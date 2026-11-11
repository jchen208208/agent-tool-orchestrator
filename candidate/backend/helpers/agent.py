"""The orchestration loop: ask the model, run the tools it asks for, feed the results back, repeat."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from backend.chat_schema import ChatMessage, ToolCallLog
from backend.helpers.config import AGENT_MODEL, MAX_ROUNDS, TOOL_RESULT_MAX_CHARS
from backend.helpers.messages import to_responses_input
from backend.helpers.prompts import AGENT_PROMPT
from backend.helpers.router import route

log = logging.getLogger(__name__)

SEARCH_TOOLS = "search_tools"

# meta-tool for when the router missed something. it's ours, not a registered tool, so it never goes in tool_calls
SEARCH_TOOLS_SCHEMA = {
    "type": "function",
    "name": SEARCH_TOOLS,
    "description": (
        "Find more tools when none of your current tools can do a step. Describe the capability in plain words, "
        "e.g. 'send a message to a Slack channel'. Matching tools become available on your next turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {"need": {"type": "string", "description": "What the missing tool needs to do."}},
        "required": ["need"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class AgentResult:
    reply: str = ""
    tool_calls: list[ToolCallLog] = field(default_factory=list)  # what the grader sees
    trace: list[dict] = field(default_factory=list)  # our debug log: every round, call, search and timing


def run_agent(messages: list[ChatMessage], tool_names: list[str], client) -> AgentResult:
    """Runs the loop until the model answers in text or MAX_ROUNDS is hit."""
    out = AgentResult()
    active = list(tool_names)  # tools the model can see; search_tools can add more
    seen: set[str] = set()  # exact calls already made this turn
    items = to_responses_input(messages)

    for round_no in range(1, MAX_ROUNDS + 1):
        last_round = round_no == MAX_ROUNDS
        start = time.perf_counter()
        try:
            response = client.responses.create(
                model=AGENT_MODEL,
                instructions=AGENT_PROMPT,
                input=items,
                tools=_tool_schemas(active),
                tool_choice="none" if last_round else "auto",  # last round: force a text answer
                parallel_tool_calls=True,
                temperature=0,
            )
        except Exception:
            log.exception("agent model call failed")
            out.reply = "Sorry, I couldn't reach the language model, so I couldn't finish this request."
            return out  # still return whatever tool calls already ran

        calls = [item for item in response.output if item.type == "function_call"]
        _trace(out, {"round": round_no, "ms": _ms(start), **_usage(response), "calls": [c.name for c in calls]})

        if not calls:
            out.reply = (response.output_text or "").strip() or "Sorry, I couldn't come up with an answer."
            return out

        # echo the calls back first, then one output per call_id (same shape as OpenAI's docs)
        for call in calls:
            items.append({"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments})
        # run in order on one thread: the mocks share state with no locks, and they take ms anyway
        for call in calls:
            output = _handle_call(call, active, seen, out, client)
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    # only reached if the model ignores tool_choice="none" on the last round
    out.reply = f"I stopped after {MAX_ROUNDS} steps without finishing. Nothing beyond the tool calls listed was done."
    return out


def _handle_call(call, active: list[str], seen: set[str], out: AgentResult, client) -> str:
    """Runs one function call from the model. Returns the text the model sees as that call's output."""
    try:
        raw_args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as e:
        return _error(f"Arguments were not valid JSON: {e}")
    # strict mode makes the model send null for every optional arg it doesn't use; drop them so the tool's defaults apply
    args = {k: v for k, v in raw_args.items() if v is not None}

    if call.name == SEARCH_TOOLS:
        return _search_tools(str(args.get("need", "")), active, out, client)

    if call.name not in active:
        return _error(f"Tool {call.name} isn't available. Use {SEARCH_TOOLS} to find the right one.")

    key = call.name + json.dumps(args, sort_keys=True)
    if key in seen:
        _trace(out, {"skipped_repeat": call.name, "args": args})
        return _error("You already made this exact call this turn. Use that result instead of repeating it.")
    seen.add(key)

    start = time.perf_counter()
    entry = _invoke(call.name, args)
    out.tool_calls.append(entry)
    _trace(out, {"tool": call.name, "args": args, "ms": _ms(start), "error": entry.error, "result": _preview(entry.result)})

    if entry.error:
        return _error(entry.error)
    return _truncate(json.dumps(entry.result, default=str))


def _invoke(name: str, args: dict) -> ToolCallLog:
    from backend.main import get_tool_spec  # imported here to avoid a circular import with main.py

    try:
        # invoke() validates args with pydantic (extra=forbid), then calls the mock
        envelope = get_tool_spec(name).invoke(**args)
        return ToolCallLog(name=name, arguments=args, result=envelope.model_dump(mode="json")["result"])
    except ValidationError as e:
        return ToolCallLog(name=name, arguments=args, error=f"Invalid arguments: {e}")
    except Exception as e:
        # mocks raise on bad input (unknown channel, missing id, ...). the model gets the message and can fix it
        return ToolCallLog(name=name, arguments=args, error=f"{type(e).__name__}: {e}")


def _search_tools(need: str, active: list[str], out: AgentResult, client) -> str:
    # same router as the start of the turn, but on the model's description of what it's missing
    found = route([ChatMessage(role="user", content=need)], client) if need.strip() else []
    added = [n for n in found if n not in active]
    active.extend(added)
    _trace(out, {"search_tools": need, "found": found, "added": added})

    if not found:
        return json.dumps({"tools": [], "note": "No tool matches. Tell the user this can't be done with the available tools."})

    from backend.main import get_all_tool_specs

    specs = get_all_tool_specs()
    return json.dumps({"now_available": [{"name": n, "description": specs[n].description} for n in found]})


def _tool_schemas(names: list[str]) -> list[dict]:
    from backend.main import get_openai_tools

    # careful: get_openai_tools([]) returns all 191 tools, so only call it with a non-empty list
    return (get_openai_tools(names) if names else []) + [SEARCH_TOOLS_SCHEMA]


def _error(message: str) -> str:
    return json.dumps({"error": message})


def _truncate(text: str) -> str:
    # only trims what the model reads; tool_calls keeps the full result
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    return text[:TOOL_RESULT_MAX_CHARS] + f"... [truncated, {len(text)} chars total]"


def _preview(value, limit: int = 200) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def _usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {"in_tokens": usage.input_tokens, "out_tokens": usage.output_tokens}


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _trace(out: AgentResult, step: dict) -> None:
    out.trace.append(step)
    log.info("trace %s", step)
