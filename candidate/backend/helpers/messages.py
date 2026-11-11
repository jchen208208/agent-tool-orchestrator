from __future__ import annotations

from backend.chat_schema import ChatMessage


def to_responses_input(messages: list[ChatMessage]) -> list[dict]:
    items = []
    for m in messages:
        if m.role == "tool":
            items.append({"role": "user", "content": f"[earlier tool output]\n{m.content}"})
        else:
            items.append({"role": m.role, "content": m.content})
    return items
