"""Tool router: one cheap LLM call picks which tools the agent gets to see."""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache

from backend.chat_schema import ChatMessage
from backend.helpers.config import MAX_ROUTED_TOOLS, ROUTER_MODEL
from backend.helpers.llm import create_response
from backend.helpers.messages import to_responses_input
from backend.helpers.prompts import ROUTER_PROMPT

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def tool_names() -> tuple[str, ...]:
    from backend.main import list_available_tools  # imported here to avoid a circular import with main.py

    return tuple(list_available_tools())


@lru_cache(maxsize=1)
def compact_catalog() -> str:
    # one line per tool: ~13k chars (~4k tokens) vs ~72k tokens for the full schemas
    from backend.main import get_all_tool_specs

    return "\n".join(f"{name}: {spec.description}" for name, spec in sorted(get_all_tool_specs().items()))


def _output_schema() -> dict:
    # enum = the model can only return real tool names
    return {
        "type": "json_schema",
        "name": "routed_tools",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tools": {"type": "array", "items": {"type": "string", "enum": list(tool_names())}},
            },
            "required": ["tools"],
            "additionalProperties": False,
        },
    }


def route(messages: list[ChatMessage], client) -> list[str]:
    """Returns up to MAX_ROUTED_TOOLS tool names for this conversation. [] if none needed or the call fails."""
    start = time.perf_counter()
    try:
        response = create_response(
            client,
            ROUTER_MODEL,
            # catalog goes in the instructions so the prefix is identical every call (OpenAI caches it automatically)
            instructions=f"{ROUTER_PROMPT}\n\nCatalog:\n{compact_catalog()}",
            input=to_responses_input(messages),
            text={"format": _output_schema()},
            temperature=0,
        )
        picked = json.loads(response.output_text)["tools"]
    except Exception:
        # don't fail the request: the agent can still answer directly or call search_tools
        log.exception("router call failed")
        return []

    valid = set(tool_names())
    names = list(dict.fromkeys(n for n in picked if n in valid))[:MAX_ROUTED_TOOLS]  # dedupe, keep order
    log.info("router picked %s in %.0f ms", names, (time.perf_counter() - start) * 1000)
    return names
