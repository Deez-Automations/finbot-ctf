"""Standalone smoke test for ClaudeClient — not part of the test suite, throwaway.

Run with: PYTHONPATH=. uv run python scripts/smoke_test_claude.py
"""

import asyncio

from finbot.core.llm.claude_client import ClaudeClient
from finbot.core.data.models import LLMRequest


async def test_plain_text():
    client = ClaudeClient()
    response = await client.chat(
        LLMRequest(
            messages=[
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Say 'pong' and nothing else."},
            ]
        )
    )
    print("--- plain text ---")
    print("success:", response.success)
    print("content:", response.content)
    assert response.success, "Plain text call failed"
    assert "pong" in (response.content or "").lower()


async def test_tool_call():
    client = ClaudeClient()
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "strict": True,
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
    ]
    response = await client.chat(
        LLMRequest(
            messages=[
                {"role": "system", "content": "You must use tools when available."},
                {"role": "user", "content": "What's the weather in Lahore?"},
            ],
            tools=tools,
        )
    )
    print("--- tool call ---")
    print("success:", response.success)
    print("tool_calls:", response.tool_calls)
    print("messages (post-call):", response.messages)
    assert response.success, "Tool call request failed"
    assert response.tool_calls, "Model did not call the tool"
    assert response.tool_calls[0]["name"] == "get_weather"
    assert "city" in response.tool_calls[0]["arguments"]

    follow_up_messages = response.messages + [
        {
            "type": "function_call_output",
            "call_id": response.tool_calls[0]["call_id"],
            "output": '{"city": "Lahore", "temp_c": 31, "condition": "sunny"}',
        }
    ]
    follow_up = await client.chat(
        LLMRequest(
            messages=[{"role": "system", "content": "You must use tools when available."}]
            + follow_up_messages,
        )
    )
    print("--- follow-up after tool result ---")
    print("success:", follow_up.success)
    print("content:", follow_up.content)
    assert follow_up.success, "Follow-up call after tool result failed"


async def test_multiple_tool_calls_in_one_turn():
    """Claude requires same-turn tool results bundled into one user message --
    this is the part most likely to break if the grouping logic is wrong.
    """
    client = ClaudeClient()
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "strict": True,
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
    ]
    response = await client.chat(
        LLMRequest(
            messages=[
                {"role": "system", "content": "You must use tools when available."},
                {
                    "role": "user",
                    "content": "What's the weather in Lahore and in Karachi? Call the tool for both cities.",
                },
            ],
            tools=tools,
        )
    )
    print("--- multiple tool calls in one turn ---")
    print("tool_calls:", response.tool_calls)
    assert response.success
    assert len(response.tool_calls) >= 1

    follow_up_messages = list(response.messages)
    for tc in response.tool_calls:
        follow_up_messages.append(
            {
                "type": "function_call_output",
                "call_id": tc["call_id"],
                "output": '{"temp_c": 30, "condition": "clear"}',
            }
        )
    follow_up = await client.chat(
        LLMRequest(
            messages=[{"role": "system", "content": "You must use tools when available."}]
            + follow_up_messages,
        )
    )
    print("follow-up success:", follow_up.success)
    print("follow-up content:", follow_up.content)
    assert follow_up.success, "Follow-up after multi-tool-call turn failed"


async def main():
    await test_plain_text()
    await test_tool_call()
    await test_multiple_tool_calls_in_one_turn()
    print("\nAll Claude smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
