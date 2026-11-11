"""Unit tests for the agent loop, using a scripted fake LLM (no API key needed)."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from backend.chat_schema import ChatMessage
from backend.helpers import agent
from backend.main import reset_all_mock_state

USER = [ChatMessage(role="user", content="test")]


def call(name: str, args: dict, call_id: str) -> NS:
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=call_id)


def reply(*calls: NS, text: str = "") -> NS:
    return NS(output=list(calls), output_text=text, usage=None)


class FakeClient:
    """Stands in for OpenAI(): returns the scripted responses in order and records every request."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def outputs(request: dict) -> list[dict]:
    return [json.loads(i["output"]) for i in request["input"] if i.get("type") == "function_call_output"]


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_all_mock_state()

    def test_text_reply_with_no_tools(self) -> None:
        fake = FakeClient([reply(text="Hi!")])
        result = agent.run_agent(USER, [], fake)
        self.assertEqual(result.reply, "Hi!")
        self.assertEqual(result.tool_calls, [])
        # no routed tools: only search_tools is offered, never all 191
        self.assertEqual([t["name"] for t in fake.requests[0]["tools"]], ["search_tools"])

    def test_batched_calls_run_and_errors_go_back_to_model(self) -> None:
        fake = FakeClient([
            reply(
                call("slack_list_conversations", {"limit": None}, "c1"),
                call("slack_send_message", {"channel": "nope", "text": "hi"}, "c2"),
            ),
            reply(text="done"),
        ])
        result = agent.run_agent(USER, ["slack_list_conversations", "slack_send_message"], fake)

        self.assertEqual([c.name for c in result.tool_calls], ["slack_list_conversations", "slack_send_message"])
        self.assertEqual(result.tool_calls[0].arguments, {})  # null optional dropped
        self.assertIsNotNone(result.tool_calls[0].result)
        self.assertIn("Unknown Slack channel", result.tool_calls[1].error)
        self.assertIsNone(result.tool_calls[1].result)

        out = outputs(fake.requests[1])
        self.assertIn("conversations", out[0])
        self.assertIn("Unknown Slack channel", out[1]["error"])

    def test_repeated_call_is_skipped(self) -> None:
        fake = FakeClient([
            reply(call("slack_list_conversations", {}, "c1")),
            reply(call("slack_list_conversations", {}, "c2")),
            reply(text="done"),
        ])
        result = agent.run_agent(USER, ["slack_list_conversations"], fake)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIn("already made this exact call", outputs(fake.requests[2])[1]["error"])

    def test_tool_outside_the_list_is_refused(self) -> None:
        fake = FakeClient([reply(call("GMAIL_FETCH_EMAILS", {}, "c1")), reply(text="done")])
        result = agent.run_agent(USER, ["slack_list_conversations"], fake)
        self.assertEqual(result.tool_calls, [])
        self.assertIn("search_tools", outputs(fake.requests[1])[0]["error"])

    def test_search_tools_adds_missing_tool(self) -> None:
        fake = FakeClient([
            reply(call("search_tools", {"need": "list my Slack conversations"}, "c1")),
            NS(output_text=json.dumps({"tools": ["slack_list_conversations"]})),  # the router re-run
            reply(call("slack_list_conversations", {}, "c2")),
            reply(text="done"),
        ])
        result = agent.run_agent(USER, [], fake)

        self.assertEqual(result.reply, "done")
        self.assertEqual([c.name for c in result.tool_calls], ["slack_list_conversations"])  # search_tools not logged
        self.assertEqual(fake.requests[1]["input"][0]["content"], "list my Slack conversations")
        self.assertEqual(outputs(fake.requests[2])[0]["now_available"][0]["name"], "slack_list_conversations")
        self.assertIn("slack_list_conversations", [t["name"] for t in fake.requests[2]["tools"]])

    def test_search_tools_with_no_match(self) -> None:
        fake = FakeClient([
            reply(call("search_tools", {"need": "order a pizza"}, "c1")),
            NS(output_text=json.dumps({"tools": []})),
            reply(text="I can't do that."),
        ])
        result = agent.run_agent(USER, [], fake)
        self.assertEqual(result.tool_calls, [])
        self.assertIn("No tool matches", outputs(fake.requests[2])[0]["note"])

    def test_last_round_forces_a_text_answer(self) -> None:
        fake = FakeClient([reply(call("slack_list_conversations", {}, "c1")), reply(text="partial")])
        with mock.patch.object(agent, "MAX_ROUNDS", 2):
            result = agent.run_agent(USER, ["slack_list_conversations"], fake)
        self.assertEqual([r["tool_choice"] for r in fake.requests], ["auto", "none"])
        self.assertEqual(result.reply, "partial")

    def test_llm_failure_returns_fallback_and_keeps_log(self) -> None:
        fake = FakeClient([reply(call("slack_list_conversations", {}, "c1")), RuntimeError("boom")])
        result = agent.run_agent(USER, ["slack_list_conversations"], fake)
        self.assertIn("couldn't reach the language model", result.reply)
        self.assertEqual(len(result.tool_calls), 1)


if __name__ == "__main__":
    unittest.main()
