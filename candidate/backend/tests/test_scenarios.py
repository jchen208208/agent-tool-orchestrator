"""Live scenarios run the way the grader runs them: reset -> /chat -> check tool_calls, /state and the reply.

Uses the real OpenAI model, so it's skipped without OPENAI_API_KEY. Prints a scorecard at the end.
Run a subset with SCENARIOS=s01,s04 to save API calls. SCENARIO_PAUSE_S=20 spaces them out for low RPM limits.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import unittest
from dataclasses import dataclass, field
from typing import Callable

from fastapi.testclient import TestClient

from backend.helpers import config  # noqa: F401  (loads .env so the key check below works)
from backend.main import app

Check = Callable[[dict, dict, str], "str | None"]  # (state before, state after, reply) -> problem or None


@dataclass
class Scenario:
    id: str
    messages: list[dict]
    required: list[tuple[str, ...]] = field(default_factory=list)  # each tuple: at least one of these must be called
    forbidden: set[str] = field(default_factory=set)
    min_calls: int = 0  # fewest tool calls the task needs (for the efficiency stat, not pass/fail)
    checks: list[Check] = field(default_factory=list)


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ---- state helpers ----

def slack_texts(state: dict, channel: str) -> list[str]:
    for c in state["slack"]["channel_messages"]:
        if c["channel"] == channel:
            return [m["text"] for m in c["messages"]]
    return []


def new_items(before: list, after: list) -> list:
    old = {json.dumps(x, sort_keys=True) for x in before}
    return [x for x in after if json.dumps(x, sort_keys=True) not in old]


def calendar_events(state: dict) -> list:
    return [e for cal in state["googlecalendar"]["events"] for e in cal["events"]]


def slack_post(channel: str, *words: str) -> Check:
    def check(before, after, reply):
        added = slack_texts(after, channel)[len(slack_texts(before, channel)):]
        if not added:
            return f"no new message in Slack {channel}"
        if words and not any(all(w.lower() in t.lower() for w in words) for t in added):
            return f"new message in {channel} doesn't mention {words}: {added}"
        return None
    return check


def no_new_slack_messages(before, after, reply):
    if json.dumps(before["slack"]["channel_messages"]) != json.dumps(after["slack"]["channel_messages"]):
        return "a Slack message was posted"
    return None


def unchanged(service: str, key: str) -> Check:
    def check(before, after, reply):
        if json.dumps(before[service][key], sort_keys=True) != json.dumps(after[service][key], sort_keys=True):
            return f"{service}.{key} changed but shouldn't have"
        return None
    return check


def no_new_calendar_events(before, after, reply):
    if len(calendar_events(after)) != len(calendar_events(before)):
        return "a calendar event was created or deleted"
    return None


def reply_mentions(*options: str) -> Check:
    def check(before, after, reply):
        if not any(o.lower() in reply.lower() for o in options):
            return f"reply doesn't mention any of {options}"
        return None
    return check


def asks_question(before, after, reply):
    return None if "?" in reply else "reply doesn't ask a clarifying question"


def new_linear_issue(title_words: str, team_prefix: str) -> Check:
    def check(before, after, reply):
        added = new_items(before["linear"]["issues"], after["linear"]["issues"])
        hits = [i for i in added if title_words.lower() in i.get("title", "").lower()]
        if not hits:
            return f"no new Linear issue titled like '{title_words}'"
        if not str(hits[0]["id"]).startswith(team_prefix):  # linear mock ids look like ENG-6
            return f"issue created in the wrong team: {hits[0]['id']}"
        return None
    return check


def new_linear_comment(issue_id: str, words: str) -> Check:
    def check(before, after, reply):
        added = new_items(before["linear"]["comments"], after["linear"]["comments"])
        if not any(f'"{issue_id}"' in json.dumps(c) and words.lower() in json.dumps(c).lower() for c in added):
            return f"no new comment on {issue_id} containing '{words}'"
        return None
    return check


def new_gmail_draft_to(address: str) -> Check:
    def check(before, after, reply):
        added = new_items(before["gmail"]["drafts"], after["gmail"]["drafts"])
        if not any(address in json.dumps(d) for d in added):
            return f"no new draft addressed to {address}"
        return None
    return check


# ---- scenarios (built from the seed data in the mocks) ----

WRITE_TOOLS = {"slack_send_message", "GOOGLECALENDAR_CREATE_EVENT", "GOOGLECALENDAR_QUICK_ADD", "GMAIL_CREATE_EMAIL_DRAFT"}
DRIVE_WRITES = {"GOOGLEDRIVE_DELETE_DRIVE", "GOOGLEDRIVE_DELETE_PERMISSION", "GOOGLEDRIVE_EMPTY_TRASH", "GOOGLEDRIVE_EDIT_FILE",
                "GOOGLEDRIVE_DELETE_COMMENT", "GOOGLEDRIVE_DELETE_REPLY"}

SCENARIOS = [
    # single service, single tool
    Scenario("s01_slack_list_conversations", user("What conversations do I have in Slack?"),
             required=[("slack_list_conversations",)], min_calls=1,
             checks=[reply_mentions("engineering")]),
    Scenario("s02_calendar_today", user("What's on my calendar today?"),
             required=[("GOOGLECALENDAR_EVENTS_LIST", "GOOGLECALENDAR_FIND_EVENT")], min_calls=1,
             checks=[reply_mentions("Revenue board prep")]),
    Scenario("s03_linear_eng_issues", user("List the issues in the Engineering team on Linear."),
             required=[("linear_list_issues",)], min_calls=1,
             checks=[reply_mentions("ENG-1", "Wire /chat")]),
    Scenario("s04_github_open_issues", user("What open issues are there in the instalily/orchestrator repo?"),
             required=[("github_list_issues", "github_search_issues")], min_calls=1,
             checks=[reply_mentions("routing")]),
    Scenario("s05_perplexity", user("Search the web for recent news about AI agent frameworks."),
             required=[("perplexity_search",)], min_calls=1),

    # cross-service and multi-step
    Scenario("s06_email_to_slack",
             user("Find the most recent email about the Q1 revenue review and post a summary to the leadership Slack channel."),
             required=[("GMAIL_FETCH_EMAILS", "GMAIL_LIST_THREADS"), ("slack_send_message",)], min_calls=3,  # fetch, find channel id, post
             checks=[slack_post("C002")]),
    Scenario("s07_thread_lookup", user("Who owns the launch rollback checklist? It was discussed in #engineering."),
             required=[("slack_get_thread", "slack_conversations_history", "slack_search_messages", "slack_get_full_conversation")],
             min_calls=2, checks=[reply_mentions("Jordan")]),
    Scenario("s08_linear_create_issue", user("Create a Linear issue in the Engineering team titled 'Fix router timeout'."),
             required=[("linear_create_issue",)], min_calls=1,
             checks=[new_linear_issue("Fix router timeout", "ENG")]),
    Scenario("s09_linear_comment", user("Find the Linear issue about the launch delay and comment 'Looking into this today' on it."),
             required=[("linear_create_comment",)], min_calls=2,
             checks=[new_linear_comment("ENG-3", "Looking into this today")]),
    Scenario("s10_draft_to_contact", user("Draft an email to Morgan Lee saying the board deck is ready."),
             required=[("GMAIL_CREATE_EMAIL_DRAFT",)], min_calls=2,
             checks=[new_gmail_draft_to("finance@corp.com")]),
    Scenario("s11_parallel_reads", user("How many unread emails do I have, and what's on my calendar tomorrow?"),
             required=[("GMAIL_FETCH_EMAILS", "GMAIL_LIST_THREADS"), ("GOOGLECALENDAR_EVENTS_LIST", "GOOGLECALENDAR_FIND_EVENT")],
             min_calls=2, checks=[reply_mentions("Deep work")]),

    # conversation state
    Scenario("s12_follow_up_turn", [
        {"role": "user", "content": "What conversations do I have in Slack?"},
        {"role": "assistant", "content": "You have #engineering, #leadership-private, a DM with Avery and Morgan, and #launch-war-room."},
        {"role": "user", "content": "Post 'Standup moved to 10am' in the first one."},
    ], required=[("slack_send_message",)], min_calls=2, checks=[slack_post("C001", "10am")]),  # needs the channel id first

    # ambiguity, errors, no tool
    Scenario("s13_ambiguous_meeting", user("Schedule a 30-minute meeting with everyone on the project next week."),
             forbidden={"GOOGLECALENDAR_CREATE_EVENT", "GOOGLECALENDAR_QUICK_ADD"},
             checks=[no_new_calendar_events, asks_question]),
    Scenario("s14_missing_drive_file", user("Delete the file 'budget_2025.xlsx' from my Drive."),
             forbidden=DRIVE_WRITES, min_calls=1,
             checks=[unchanged("googledrive", "files"), reply_mentions("couldn't find", "could not find", "not found",
                                                                      "doesn't exist", "does not exist", "no file", "unable to find")]),
    Scenario("s15_missing_slack_channel", user("Post 'hello team' to the #marketing Slack channel."),
             min_calls=1, checks=[no_new_slack_messages, reply_mentions("marketing")]),
    Scenario("s16_greeting", user("Hi! What can you help me with?"), forbidden=WRITE_TOOLS),
    Scenario("s17_out_of_scope", user("Order me a large pepperoni pizza for delivery."), forbidden=WRITE_TOOLS,
             checks=[no_new_slack_messages]),
]
NO_TOOLS = {"s16_greeting", "s17_out_of_scope"}  # these must make zero tool calls


@dataclass
class Outcome:
    id: str
    ok: bool
    ms: int
    calls: int
    min_calls: int
    problems: list[str]
    state_checks: int
    state_passed: int


def run_scenario(client: TestClient, s: Scenario) -> Outcome:
    client.post("/reset")
    before = client.get("/state").json()
    start = time.perf_counter()
    response = client.post("/chat", json={"messages": s.messages})
    ms = round((time.perf_counter() - start) * 1000)
    after = client.get("/state").json()

    problems: list[str] = []
    if response.status_code != 200:
        return Outcome(s.id, False, ms, 0, s.min_calls, [f"HTTP {response.status_code}"], len(s.checks), 0)
    body = response.json()
    reply = body["messages"][-1]["content"]
    called = [c["name"] for c in body["tool_calls"]]

    if body["messages"][-1]["role"] != "assistant" or not reply.strip():
        problems.append("no assistant reply")
    for group in s.required:
        if not any(name in called for name in group):
            problems.append(f"missing required tool: {' or '.join(group)}")
    for name in sorted(s.forbidden & set(called)):
        problems.append(f"called forbidden tool: {name}")
    if s.id in NO_TOOLS and called:
        problems.append(f"should make no tool calls, made {called}")

    state_passed = 0
    for check in s.checks:
        problem = check(before, after, reply)
        if problem:
            problems.append(problem)
        else:
            state_passed += 1

    if problems:
        problems.append(f"called {called}; reply: {reply[:160]!r}")
    return Outcome(s.id, not problems, ms, len(called), s.min_calls, problems, len(s.checks), state_passed)


def print_scorecard(outcomes: list[Outcome]) -> None:
    print("\n" + "=" * 70)
    for o in outcomes:
        mark = "PASS" if o.ok else "FAIL"  # ascii: windows consoles (cp1252) can't print check marks
        line = f"  {mark}  [{o.id}]  {o.ms:>6} ms  ({o.calls}/{o.min_calls} calls)"
        print(line + (f" - {o.problems[0]}" if o.problems else ""))
        for extra in o.problems[1:]:
            print(f"        {extra}")
    n = len(outcomes)
    passed = sum(o.ok for o in outcomes)
    times = sorted(o.ms for o in outcomes)
    p95 = times[min(n - 1, round(0.95 * (n - 1)))]
    over = [o for o in outcomes if o.calls > o.min_calls]
    print(f"\n  Pass rate: {passed}/{n} ({passed / n:.0%})")
    print(f"  Latency:   p50 = {statistics.median(times) / 1000:.1f}s, p95 = {p95 / 1000:.1f}s, total = {sum(times) / 1000:.0f}s")
    print(f"  Tool use:  {len(over)}/{n} scenarios over minimum, +{sum(o.calls - o.min_calls for o in over)} extra calls total")
    print(f"  Checks:    {sum(o.state_passed for o in outcomes)}/{sum(o.state_checks for o in outcomes)} state/reply checks passed")
    print("=" * 70)


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "live scenarios need OPENAI_API_KEY")
class LiveScenarioTests(unittest.TestCase):
    def test_scenarios(self) -> None:
        wanted = {x.strip() for x in os.getenv("SCENARIOS", "").split(",") if x.strip()}
        chosen = [s for s in SCENARIOS if not wanted or s.id.split("_")[0] in wanted or s.id in wanted]
        pause = float(os.getenv("SCENARIO_PAUSE_S", "0"))  # spacing for low requests-per-minute keys
        client = TestClient(app)
        outcomes = []
        for i, s in enumerate(chosen):
            if i and pause:
                time.sleep(pause)
            outcome = run_scenario(client, s)
            outcomes.append(outcome)
            with self.subTest(scenario=s.id):
                self.assertTrue(outcome.ok, "\n".join(outcome.problems))
        print_scorecard(outcomes)


if __name__ == "__main__":
    unittest.main()
