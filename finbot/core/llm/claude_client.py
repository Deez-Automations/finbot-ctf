"""Claude (Anthropic) Client with configurable model

Anthropic's Messages API has its own shape, different from both OpenAI's
Responses API (which FinBot's BaseAgent loop is natively built around) and
Groq's Chat Completions format. Differences that matter here:

- The system prompt is a separate top-level `system` param, never a message
  with role="system" inside the messages list (Claude rejects that).
- Assistant tool calls are content blocks ({"type":"tool_use", "id","name",
  "input"}) inside an assistant message, and `input` is already a parsed
  dict, not a JSON string.
- Tool results are content blocks ({"type":"tool_result","tool_use_id",
  "content"}) and Claude expects ALL tool results from one turn bundled into
  a SINGLE user message, not one message per result.
- Tool definitions use "input_schema", not "parameters", with no
  "type":"function" wrapper and no "strict" field.

This client converts FinBot's Responses-API-shaped message items to/from
Claude's Messages format on each call, and returns messages in FinBot's
native shape so the rest of base.py's agent loop works unmodified,
regardless of which provider is active.
"""

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic

from finbot.config import settings
from finbot.core.data.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


def _extract_system_prompt(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Pull the leading system message out of FinBot's message list.

    Claude takes the system prompt as a separate top-level param and errors
    if a "system" role appears inside the messages list itself.
    """
    system_parts = []
    remaining = []
    for item in messages:
        if item.get("role") == "system":
            system_parts.append(item.get("content", ""))
        else:
            remaining.append(item)
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, remaining


def _to_claude_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert FinBot's Responses-API-shaped message items to Claude's format.

    Consecutive "function_call" items become tool_use blocks in ONE assistant
    message; consecutive "function_call_output" items become tool_result
    blocks in ONE user message — Claude requires same-turn results bundled
    together, unlike per-result messages.
    """
    converted: list[dict[str, Any]] = []
    pending_tool_use: list[dict[str, Any]] = []
    pending_tool_result: list[dict[str, Any]] = []

    def flush_tool_use() -> None:
        if pending_tool_use:
            converted.append({"role": "assistant", "content": list(pending_tool_use)})
            pending_tool_use.clear()

    def flush_tool_result() -> None:
        if pending_tool_result:
            converted.append({"role": "user", "content": list(pending_tool_result)})
            pending_tool_result.clear()

    for item in messages:
        item_type = item.get("type")

        if item_type == "function_call":
            flush_tool_result()
            raw_args = item.get("arguments", "{}")
            try:
                parsed_input = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                parsed_input = {}
            pending_tool_use.append(
                {
                    "type": "tool_use",
                    "id": item["call_id"],
                    "name": item["name"],
                    "input": parsed_input,
                }
            )
            continue

        if item_type == "function_call_output":
            flush_tool_use()
            pending_tool_result.append(
                {
                    "type": "tool_result",
                    "tool_use_id": item["call_id"],
                    "content": item["output"],
                }
            )
            continue

        flush_tool_use()
        flush_tool_result()
        converted.append({"role": item.get("role"), "content": item.get("content")})

    flush_tool_use()
    flush_tool_result()
    return converted


def _to_claude_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert FinBot's flat Responses-API tool defs to Claude's input_schema shape."""
    if not tools:
        return None
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
        }
        for tool in tools
    ]


class ClaudeClient:
    """Claude Client with configurable model, speaking Anthropic's Messages API."""

    def __init__(self):
        self.default_model = settings.LLM_DEFAULT_MODEL
        self.default_temperature = settings.LLM_DEFAULT_TEMPERATURE
        self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Chat with Claude, translating to/from FinBot's internal message shape."""
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
            system_prompt, remaining = _extract_system_prompt(input_list)
            claude_messages = _to_claude_messages(remaining)
            claude_tools = _to_claude_tools(request.tools)

            create_params: dict[str, Any] = {
                "model": model,
                "messages": claude_messages,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": temperature,
            }
            if system_prompt:
                create_params["system"] = system_prompt
            if claude_tools:
                create_params["tools"] = claude_tools

            response = await self._client.messages.create(**create_params)

            if not response or not response.content:
                logger.warning("Invalid Claude response: no content blocks returned")
                return LLMResponse(
                    content="",
                    provider="anthropic",
                    success=False,
                    messages=input_list,
                    tool_calls=[],
                )

            new_entries: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "name": block.name,
                            "call_id": block.id,
                            "arguments": block.input,
                        }
                    )
                    new_entries.append(
                        {
                            "type": "function_call",
                            "name": block.name,
                            "call_id": block.id,
                            "arguments": json.dumps(block.input),
                        }
                    )

            content = "".join(text_parts)
            if content:
                new_entries.insert(0, {"role": "assistant", "content": content})

            input_list = input_list + new_entries

            metadata = {
                "stop_reason": response.stop_reason,
                "response_id": response.id,
            }

            return LLMResponse(
                content=content,
                provider="anthropic",
                success=True,
                metadata=metadata,
                messages=input_list,
                tool_calls=tool_calls,
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Claude chat failed: %s", e)
            raise Exception(f"Claude chat failed: {e}") from e  # pylint: disable=broad-exception-raised
