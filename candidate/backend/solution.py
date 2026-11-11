from __future__ import annotations

import logging
import time
from functools import lru_cache

from openai import OpenAI

from backend.chat_schema import ChatMessage, ChatRequest, ChatResponse
from backend.helpers.agent import run_agent
from backend.helpers.config import LLM_MAX_RETRIES, LLM_TIMEOUT_S
from backend.helpers.router import route

log = logging.getLogger("backend.solution")

# uvicorn only sets up its own loggers, so give ours a handler to print the trace in the server console
_backend_log = logging.getLogger("backend")
if not _backend_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    _backend_log.addHandler(_handler)
    _backend_log.setLevel(logging.INFO)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # made on the first request, not at import, so the server boots without a key
    return OpenAI(timeout=LLM_TIMEOUT_S, max_retries=LLM_MAX_RETRIES)


def chat(request: ChatRequest) -> ChatResponse:
    """POST /chat: route to a tool shortlist, run the agent loop, return the reply + every tool call made."""
    start = time.perf_counter()
    messages = request.messages

    try:
        client = _client()
    except Exception:
        log.exception("couldn't create the OpenAI client (is OPENAI_API_KEY set?)")
        return _respond(messages, "Sorry, the assistant isn't configured right now (no OpenAI API key).")

    tool_names = route(messages, client)
    result = run_agent(messages, tool_names, client)

    rounds = sum(1 for step in result.trace if "round" in step)
    log.info(
        "chat done in %d ms: %d routed tools, %d model rounds, %d tool calls",
        (time.perf_counter() - start) * 1000, len(tool_names), rounds, len(result.tool_calls),
    )
    return _respond(messages, result.reply, result.tool_calls)


def _respond(messages: list[ChatMessage], reply: str, tool_calls=None) -> ChatResponse:
    return ChatResponse(
        messages=[*messages, ChatMessage(role="assistant", content=reply)],
        tool_calls=tool_calls or [],
    )
