# Design

**Loom video:** ADD LINK HERE

## Abstract

POST /chat first sends the conversation to one initial LLM routing call, which narrows down the 191 tools list to at most 17. It then runs a loop where the model calls those tools until it can generate the answer. All the tool calls are logged. If a tool call fails, the error goes back to the model to be fixed. If the router missed a tool, the model can search for it while inside the loop.

## Tool routing

All 191 schemas make up around 44k tokens, so sending them every model call is slow and costly, and tool choice gets worse with such a long list.

I considered several routing options:

1. **Send all 191 every call.** Pros: simplest + no tools missed. Cons: 44k tokens per call, slow, worse tool choice.
2. **Vector embedding similarity.** Pros: no extra model call. Cons: needs an embedding call per request, and similarity only finds tools that sound like the query.
3. **A cheap LLM call that picks tools from a list of names and one-line descriptions.** Pros: understands multi-step tasks; ~4k tokens; testable on its own. Cons: one extra LLM call per request.
4. **Pick services, then tools.** Pros: 2 smaller prompts. Cons: two extra calls = higher latency.
5. **Compact catalog in the main prompt and the agent loads tools itself.** Pros: no separate router. Cons: same number of calls as option 3; harder to test routing by itself.

I picked option 3. I first send the conversation plus the one-line description for each tool to the router and it returns up to 12 names, about 10 times fewer tokens than every schema. The code then adds the main search tool (decided by me) of each service the router picked from (e.g. `GMAIL_FETCH_EMAILS` for Gmail). This adds up to 5 more tools. I added this because small router models sometimes skip these lookup tools. The router can only answer with names from the real list of 191 tools, so it can't invent a tool. The tool list is identical on every request, so OpenAI caches it, and about 95% of the router's input came from that cache. I chose it over option 5 because routing can be tested by itself. I chose to select 12 schemas because most tasks need 2 to 5 tools, and 12 schemas are only about 2.7k tokens.

If the router misses a tool, the agent calls `search_tools` with a description ("send a message to a Slack channel"), which goes back through the router and adds tools for the next round. A `load_tools(names)` version wouldn't work because the model doesn't know names it's never seen. This recovered a missed Slack lookup twice in my live tests. `search_tools` shows up in my debug trace, but not in the `tool_calls` list the grader scores since it's a helper rather than a real tool. OpenAI's native tool search (deferred tool loading) does the same thing, but it only works on gpt-5.4 and newer, and `.env.example` uses gpt-4.1-mini.

My own live tests also showed scenarios where a similar tool returns the wrong data and the model accepts it (e.g. your own profile instead of your contacts). I changed the agent prompt to say to try another tool when a result lacks what it needs, which fixed two scenarios. The one that still failed had the right tool available, but the agent reads a total message count as the unread count.

## Orchestration loop

Each round sends the prompt, the conversation with all tool results so far, and the routed schemas to the Responses API. A reply that doesn't contain any tool calls is considered the final answer.

The loop is capped at 8 model calls. The last uses `tool_choice="none"`, which forces an answer that says what wasn't finished. Exactly repeated calls are skipped.

The prompt tells the model to request calls that don't depend on each other together in one turn, since latency comes from LLM rounds. Those calls then run one after another, because running them at the same time could give two messages the same ID in the mock data.

Bad arguments and tool exceptions go back to the model as `{"error": ...}` so it can fix the call. These cases are logged with `result: null`. If the key can't use the chosen model (gpt-5.4-mini by default), it falls back to gpt-4.1-mini.

Every request sends the full conversation to both the router and the agent, so follow-ups like "post it in the first one" work. The prompt includes the mock's current date, so phrases like "next week" resolve without an extra tool call. The server console logs every round and tool call for debugging.

## Testing and trade-offs

I built 12 unit tests against a "fake" LLM and 17 live scenarios run like the grader (reset, /chat, check tools, /state and the reply). 16/17 passed, at a median of about 5 s per request. Due to OpenAI free-tier limits, I tested on gpt-4o-mini and gpt-5.4-mini/nano, not the gpt-4.1-mini used in `.env.example`.

The main trade-off is that the router adds a call and 1 to 2 s to every request, including greetings like "hi", in exchange for much smaller prompts and routing I can test.

## If I had another 8 hours

- Add tests that check the agent reports what the tool actually returned. In the one failing scenario, it read the total number of emails as the number of unread emails.
- Put hints in tool error messages, since models act on those more than on the system prompt.
- Measure router recall at caps of 8, 12 and 16, and compare option 3 against option 5.
- Run the scenario suite in CI on the grader's model.
- Cache router picks for repeated requests.
- Ask for confirmation before destructive actions in a real product.
