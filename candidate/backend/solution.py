from __future__ import annotations

from backend.chat_schema import ChatRequest, ChatResponse
from fastapi import HTTPException


def chat(request: ChatRequest) -> ChatResponse:
    """Stub endpoint. Implement this.

    Accept the conversation in `request.messages`, decide which (if any)
    of the registered tools to call, run them, and return the final
    assistant turn appended to `messages`. Populate `tool_calls` with an
    ordered log of every tool invocation made during this turn so the
    reviewer can see how the system reasoned.

    See README.md for the full task description and scenarios.
    """
    raise HTTPException(status_code=501, detail="POST /chat is not implemented yet.")
