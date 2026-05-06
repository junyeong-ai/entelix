# ADR 0095 — `Compactor` is genuinely operator-extensible

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: Add `CompactedHistory::group(events: &[GraphEvent]) -> Result<Self>` and `CompactedHistory::from_turns(turns: Vec<Turn>) -> Self` so external `Compactor` impls have a path to construct the return type. The `tool_call` / `tool_result` pair invariant stays sealed at the [`ToolPair`] level — operators can drop or pass through `ToolPair` instances they obtained from `group(...)` but can't synthesize new ones.

## Context

ADR-0092 (slice 109) sealed the pair invariant by making [`ToolPair`]'s fields private and giving [`CompactedHistory`] a private `turns: Vec<Turn>` field. The seal was load-bearing — type-enforced "tool_use never appears without its tool_result, ever".

But the seal extended too far. The doc on [`HeadDropCompactor`] said:

> Operators that want LLM-generated summary compaction implement [`Compactor`] directly.

And yet:

- `CompactedHistory { turns: ... }` couldn't be constructed externally (private field).
- `CompactedHistory::new` / `default` / `try_from` did not exist.
- The internal helper `group_into_turns` was a private fn.

So the trait was structurally sealed at the same time as being documented as operator-extensible. Slice 113 polish surfaced this when looking at what would actually happen to a user trying to write a Compactor outside the crate — the answer was "compile error, no path to the return type".

## Decision

### Two new public methods on `CompactedHistory`

```rust
impl CompactedHistory {
    /// Group `events` into the type-enforced [`Turn`] shape and
    /// return the un-trimmed compaction.
    pub fn group(events: &[GraphEvent]) -> Result<Self>;

    /// Build a `CompactedHistory` from a pre-grouped `Vec<Turn>`.
    /// External `Compactor` impls reach for this after filtering
    /// or transforming the turns returned by `group`.
    pub const fn from_turns(turns: Vec<Turn>) -> Self;
}
```

`group` wraps the existing private `group_into_turns` helper. `from_turns` is a thin constructor.

### Pair-invariant seal stays at the `ToolPair` level

[`ToolPair`]'s fields stay private. `Turn::Assistant { content, tools: Vec<ToolPair> }` is constructable externally with an *empty* `tools` vector or with `ToolPair`s the operator owns from `group(...)`, but no fresh `ToolPair` can ever be synthesized.

That is the right seal: compactors are *retention strategies* — they decide which turns and which tool round-trips to keep. They don't fabricate new tool round-trips. The shape mirrors `Vec::truncate` / `Vec::retain` (you can drop entries but the entries themselves came from a trusted source).

### `HeadDropCompactor` dogfoods the public API

The reference `HeadDropCompactor::compact` was rewritten to call `CompactedHistory::group(events)?` and `CompactedHistory::from_turns(...)` instead of the private `group_into_turns` + struct literal. This proves the public API is enough for a real strategy and prevents future drift between the internal and external paths.

## External-impl regression test

`crates/entelix-session/tests/external_compactor.rs` carries a `FirstNCompactor` that lives in the test crate (genuinely external from the `entelix-session` boundary). Two tests:

1. `external_compactor_can_construct_compacted_history` — proves the trait compiles and runs from outside the crate.
2. `external_compactor_passes_tool_pairs_through_unchanged` — proves a `ToolPair` from `group(...)` survives the round-trip into a rebuilt `CompactedHistory` with `id` / `name` intact.

Both lock in the contract.

## Consequences

- `CompactedHistory` gains two public methods. `entelix-session` baseline picks up two lines.
- The pair-invariant seal stays exactly as strong as before (still no path to `ToolPair { ... }` outside the grouping code).
- `Compactor` becomes the genuinely-extensible trait its doc claimed.
- `HeadDropCompactor` is now a worked example of the public API — operators reading the source see the expected pattern.

## References

- ADR-0092 — `Compactor` + sealed `CompactedHistory` (slice 109). This ADR closes the construct-from-outside hole left there.
- v3 plan slice 109 follow-up.
