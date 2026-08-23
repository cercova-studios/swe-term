# TUI model telemetry

## Goal

Make swe-term's Bubble Tea TUI expose the active model, reasoning effort, provider-reported token usage, and cumulative session cost in a compact status line inspired by oh-my-pi.

## UX

Render a muted status line below the bordered input. Model and reasoning remain visible before the first request. After each successful response, retain latest-turn input/output counts and cumulative session tokens/cost. Use compact token formatting and explicit unavailable pricing rather than reporting a false zero.

Example:

```text
gpt-5.6-luna · reasoning: medium · turn: 1.2k in / 340 out · session: 8.4k tokens · $0.0124
```

The optional `reasoning` config key accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; omitted means provider default (`auto`).

## Architecture

Usage is part of the core provider event contract. Providers emit a terminal completion event containing typed usage fields. The OpenAI adapter maps the configured reasoning effort to the Responses API, extracts authoritative usage including reasoning/cache details, and computes cost at the provider boundary from a provider/model pricing table. The TUI only aggregates and renders the core usage snapshot, keeping provider-specific pricing out of frontend state.

Unknown model pricing is represented as unavailable. Errors do not update usage totals.

## Testing

Add focused tests for config validation, reasoning request mapping, usage/cost calculation, session aggregation, compact formatting, and deterministic mock-provider metadata. Verify with focused Go tests and an actual TUI smoke run.
