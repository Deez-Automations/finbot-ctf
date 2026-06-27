"""Groq Client with configurable model

Groq serves the standard OpenAI Chat Completions wire format, not the newer
Responses API that OpenAIClient uses. FinBot's BaseAgent loop threads
Responses-API-shaped items through `messages` (plain {"role","content"} turns,
{"type":"function_call",...} entries, {"type":"function_call_output",...}
entries appended in base.py). This client converts that shape to Chat
Completions on the way in, and converts Groq's Chat Completions response back
to the same Responses-API item shape on the way out, so base.py's loop works
identically regardless of which provider is active.
"""

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from finbot.config import settings
from finbot.core.data.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# Llama models on Groq intermittently emit tool calls as literal text
# ("<function=name>{args}</function>") instead of structured tool_calls,
# especially with long system prompts and many available tools — both
# common in FinBot's agents. Groq surfaces this as a 400 'tool_use_failed'
# error. Empirically, retrying the identical request usually succeeds on
# the next attempt, so this is treated as transient rather than fatal.
MAX_TOOL_USE_RETRIES = 2


def _to_chat_completions_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert FinBot's Responses-API-shaped message items to Chat Completions messages.

    Consecutive "function_call" items (one assistant turn can emit several tool
    calls) are grouped into a single assistant message with a `tool_calls` list,
    since Chat Completions requires each tool result to immediately follow the
    assistant message that requested it.
    """
    converted: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if pending_tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": list(pending_tool_calls),
                }
            )
            pending_tool_calls.clear()

    for item in messages:
        item_type = item.get("type")

        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                }
            )
            continue

        if item_type == "function_call_output":
            flush_pending()
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item["output"],
                }
            )
            continue

        # Plain {"role", "content"} turn (system/user/assistant text-only).
        flush_pending()
        converted.append({"role": item.get("role"), "content": item.get("content")})

    flush_pending()
    return converted


def _relax_numeric_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Widen integer/number property types to also accept strings.

    Some Llama models on Groq (e.g. llama-4-scout) consistently stringify
    numeric tool-call arguments ("8" instead of 8). Groq enforces the JSON
    schema we send server-side and rejects the call outright when that
    happens, before we ever see a response to fix up. Accepting strings here
    stops Groq from rejecting the call; `_coerce_numeric_strings` converts
    the value back to a real int/float before FinBot's callables see it, so
    this is invisible outside this file.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema
    widened = dict(schema)
    widened["properties"] = {
        key: (
            {**prop, "type": [prop["type"], "string"]}
            if isinstance(prop, dict) and prop.get("type") in ("integer", "number")
            else prop
        )
        for key, prop in properties.items()
    }
    return widened


def _coerce_numeric_strings(args: dict[str, Any]) -> dict[str, Any]:
    """Convert stringified numbers back to int/float, undoing the schema widening above."""
    coerced = {}
    for key, value in args.items():
        if isinstance(value, str):
            if value.lstrip("-").isdigit():
                coerced[key] = int(value)
                continue
            try:
                coerced[key] = float(value)
                continue
            except ValueError:
                pass
        coerced[key] = value
    return coerced


def _to_chat_completions_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert FinBot's flat Responses-API tool defs to nested Chat Completions tool defs."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": _relax_numeric_types(tool.get("parameters", {})),
            },
        }
        for tool in tools
    ]


class GroqClient:
    """Groq Client with configurable model, speaking Chat Completions via the OpenAI SDK."""

    def __init__(self):
        self.default_model = settings.LLM_DEFAULT_MODEL
        self.default_temperature = settings.LLM_DEFAULT_TEMPERATURE
        self._client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL,
        )

    async def _create_with_tool_use_retry(self, create_params: dict[str, Any]):
        """Call chat.completions.create, retrying on Groq's transient tool_use_failed error."""
        last_error: BadRequestError | None = None
        for attempt in range(MAX_TOOL_USE_RETRIES + 1):
            try:
                return await self._client.chat.completions.create(**create_params)
            except BadRequestError as e:
                error_code = getattr(e, "code", None) or (e.body or {}).get("error", {}).get("code")
                if error_code != "tool_use_failed" or attempt == MAX_TOOL_USE_RETRIES:
                    raise
                last_error = e
                logger.warning(
                    "Groq tool_use_failed (attempt %d/%d), retrying: %s",
                    attempt + 1,
                    MAX_TOOL_USE_RETRIES,
                    e,
                )
                await asyncio.sleep(0.5 * (attempt + 1))
        raise last_error  # pragma: no cover — loop always returns or raises above

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Chat with Groq, translating to/from FinBot's internal message shape."""
        try:
            model = request.model or self.default_model
            temperature = (
                self.default_temperature
                if request.temperature is None
                else request.temperature
            )

            input_list: list[dict[str, Any]] = (
                list(request.messages) if request.messages else []
            )
            chat_messages = _to_chat_completions_messages(input_list)
            chat_tools = _to_chat_completions_tools(request.tools)

            create_params: dict[str, Any] = {
                "model": model,
                "messages": chat_messages,
                "temperature": temperature,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "timeout": settings.LLM_TIMEOUT,
            }
            if chat_tools:
                create_params["tools"] = chat_tools
                # Single tool calls per turn are noticeably more reliable than
                # bundled/parallel calls for Llama models on Groq — bundling is
                # when the malformed <function=...> text output shows up most.
                create_params["parallel_tool_calls"] = False

            if request.output_json_schema:
                # Groq's structured-output support is looser than OpenAI's Responses
                # API — best-effort JSON mode rather than strict schema validation.
                create_params["response_format"] = {"type": "json_object"}

            response = await self._create_with_tool_use_retry(create_params)

            if not response or not response.choices:
                logger.warning("Invalid Groq response: no choices returned")
                return LLMResponse(
                    content="",
                    provider="groq",
                    success=False,
                    messages=input_list,
                    tool_calls=[],
                )

            message = response.choices[0].message
            new_entries: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            content = message.content if isinstance(message.content, str) else ""
            if content:
                new_entries.append({"role": "assistant", "content": content})

            raw_tool_calls = getattr(message, "tool_calls", None) or []
            for tc in raw_tool_calls:
                function = getattr(tc, "function", None)
                raw_args = getattr(function, "arguments", "{}") if function else "{}"
                try:
                    parsed_args = json.loads(raw_args)
                    if isinstance(parsed_args, dict):
                        parsed_args = _coerce_numeric_strings(parsed_args)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Could not parse Groq tool call arguments: %s", raw_args)
                    parsed_args = {}

                name = getattr(function, "name", None) if function else None
                tool_calls.append(
                    {
                        "name": name,
                        "call_id": tc.id,
                        "arguments": parsed_args,
                    }
                )
                new_entries.append(
                    {
                        "type": "function_call",
                        "name": name,
                        "call_id": tc.id,
                        "arguments": raw_args,
                    }
                )

            input_list = input_list + new_entries

            metadata = {
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
            }

            return LLMResponse(
                content=content,
                provider="groq",
                success=True,
                metadata=metadata,
                messages=input_list,
                tool_calls=tool_calls,
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Groq chat failed: %s", e)
            raise Exception(f"Groq chat failed: {e}") from e  # pylint: disable=broad-exception-raised
