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

SERVICE_ANCHORS = {
    "GMAIL_": "GMAIL_FETCH_EMAILS",
    "GOOGLECALENDAR_": "GOOGLECALENDAR_EVENTS_LIST",
    "GOOGLEDRIVE_": "GOOGLEDRIVE_FIND_FILE",
    "slack_": "slack_list_conversations",
    "linear_": "linear_list_issues",
}


@lru_cache(maxsize=1)
def tool_names() -> tuple[str, ...]:
    from backend.main import list_available_tools

    return tuple(list_available_tools())


@lru_cache(maxsize=1)
def compact_catalog() -> str:
    from backend.main import get_all_tool_specs

    return "\n".join(f"{name}: {spec.description}" for name, spec in sorted(get_all_tool_specs().items()))


def _output_schema() -> dict:
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
    start = time.perf_counter()
    try:
        response = create_response(
            client,
            ROUTER_MODEL,
            instructions=f"{ROUTER_PROMPT}\n\nCatalog:\n{compact_catalog()}",
            input=to_responses_input(messages),
            text={"format": _output_schema()},
            temperature=0,
        )
        picked = json.loads(response.output_text)["tools"]
    except Exception:
        log.exception("router call failed")
        return []

    valid = set(tool_names())
    names = list(dict.fromkeys(n for n in picked if n in valid))[:MAX_ROUTED_TOOLS]
    names = with_anchors(names)
    log.info("router picked %s in %.0f ms", names, (time.perf_counter() - start) * 1000)
    return names


def with_anchors(names: list[str]) -> list[str]:
    extra = [a for prefix, a in SERVICE_ANCHORS.items() if a not in names and any(n.startswith(prefix) for n in names)]
    return names + extra
