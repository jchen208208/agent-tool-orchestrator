"""Converts the /chat message list into Responses API input items."""

from __future__ import annotations

from backend.chat_schema import ChatMessage


def to_responses_input(messages: list[ChatMessage]) -> list[dict]:
    items = []
    for m in messages:
        if m.role == "tool":
            # the Responses API has no plain "tool" role (tool outputs need a call_id), so pass old tool output as context
            items.append({"role": "user", "content": f"[earlier tool output]\n{m.content}"})
        else:
            items.append({"role": m.role, "content": m.content})
    return items
