# ADR 0092 — `Compactor` + sealed `CompactedHistory` with type-enforced `ToolCall`/`ToolResult` pair invariant

**Status**: Accepted
**Date**: 2026-05-06
**Decision**: A new `entelix_session::compaction` module ships:
1. A `Compactor` trait that consumes `&[GraphEvent]` and a character budget.
2. A sealed `CompactedHistory` whose constructor is private to the module.
3. A `Turn` enum (`User { content }` / `Assistant { content, tools: Vec<ToolPair> }`) that groups the event stream into model-facing chunks.
4. A `ToolPair` struct that *can only be constructed* by binding a `ToolCall` to its matching `ToolResult` — the pair invariant becomes structural.
5. A reference impl `HeadDropCompactor` that walks newest-to-oldest and keeps turns under budget.

A `CompactedHistory::to_messages()` method renders the trimmed view as `Vec<Message>` for `ChatModel::complete`. Because every `Turn::Assistant` carries `Vec<ToolPair>`, no compaction can produce output where a `tool_use` block is missing its matching `tool_result` — the foot-gun pydantic-ai's [issue #4137](https://github.com/pydantic/pydantic-ai/issues/4137) catalogues across SDKs is closed by the type system.

## Context

Long agent runs accumulate event logs that exceed the model's context window. The most common failure mode in operator-built compaction code: drop a `ToolCall` event but keep its `ToolResult` (or vice versa). The next call to the model fails with HTTP 400 — vendors reject conversations where any `tool_use` block lacks a matching `tool_result`.

pydantic-ai's `history_processors` is too low-level — operators receive `list[ModelMessage]` and write the trim logic; nothing in the type system stops them from splitting a pair. Vercel AI SDK 5's `experimental_continueSteps` and Mastra's `processConversation` have similar shapes. The recurring footgun across the industry is "the trim function looks correct but ships broken pairs to the model."

Entelix already has an event-sourced `SessionGraph` (invariant 1 — events are SSoT). The compaction layer can lift the pair invariant from "operator must remember" to "type system enforces" by closing the constructor of the compacted view.

## Decision

### Sealed `Turn` + `ToolPair`

```rust
pub struct ToolPair {
    call_id: String,
    name: String,
    input: serde_json::Value,
    result: ToolResultContent,
    is_error: bool,
}
// All fields private. Constructor is module-private (created only by
// `group_into_turns` after matching ToolCall.id with ToolResult.tool_use_id).

pub enum Turn {
    User { content: Vec<ContentPart> },
    Assistant { content: Vec<ContentPart>, tools: Vec<ToolPair> },
}
```

`ToolPair` cannot be hand-built. Operator code reads via accessors (`id()`, `name()`, `input()`, `result()`, `is_error()`) but cannot construct one. The only path to a `ToolPair` is the compactor's grouping pass — which insists on both halves matching.

### Sealed `CompactedHistory`

```rust
pub struct CompactedHistory {
    turns: Vec<Turn>,  // private; constructor is module-private
}

impl CompactedHistory {
    pub fn turns(&self) -> &[Turn];
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn to_messages(&self) -> Vec<Message>;
}
```

Operators receive `CompactedHistory` from `Compactor::compact` and read it. They cannot reach the constructor without going through a `Compactor` impl, which means they cannot synthesize a history that violates the pair invariant.

### `Compactor` trait + `HeadDropCompactor` reference

```rust
pub trait Compactor: Send + Sync + 'static {
    fn compact(&self, events: &[GraphEvent], budget_chars: usize) -> Result<CompactedHistory>;
}

pub struct HeadDropCompactor;

impl Compactor for HeadDropCompactor {
    fn compact(&self, events: &[GraphEvent], budget_chars: usize) -> Result<CompactedHistory> {
        let turns = group_into_turns(events)?;
        // walk newest-to-oldest, keep turns whose char cost fits under budget,
        // return the trimmed window
    }
}
```

`group_into_turns` is the load-bearing helper that builds the typed structure. It returns `Error::Config` on:
- `ToolResult` whose `tool_use_id` has no matching `ToolCall` in the event log.
- `ToolCall` left unmatched at the end of the event log (orphan call).
- `ToolResult` appearing before any `AssistantMessage`.

A well-formed `SessionGraph` produced by entelix's runtime never hits these — they exist for defense against synthetic event logs and corrupted persistence.

### Budget axis: `usize` characters

Token-accurate budgeting requires a tokenizer dependency (`tiktoken-rs`, `tokenizers`). Slice 109 ships character-budget approximation — every tool input / result / message body contributes its UTF-8 byte length. The relationship is monotonic (longer text = more tokens) so the strategy "drop oldest until under budget" works correctly even though the absolute number is approximate.

Operators that need token-accurate budgeting wrap their tokenizer of choice around the trait — pre-compute token costs per turn outside the compactor, then implement a custom `Compactor` that uses the precomputed costs.

## Consequences

- New module `entelix_session::compaction` (~430 lines + 8 tests).
- 5 new public types: `Compactor`, `CompactedHistory`, `Turn`, `ToolPair`, `HeadDropCompactor`.
- Facade re-exports added; xtask `facade-completeness` enforces continued coverage.
- `entelix-session` public-API baseline grows; refreshed.
- 8 regression tests: empty-log, simple round-trip, pair attachment, orphan call/result error paths, budget drops oldest, message rendering, pair invariant under partial drop.

## What this ADR does not address

- **Token-accurate budgets** — character approximation only. A future `entelix-tokenizer` companion crate could provide token-counting helpers.
- **Summary compaction** (LLM-generated synopsis of dropped turns) — operators implement `Compactor` directly to do this. The trait surface stays minimal so summary strategies do not bloat the core path.
- **Streaming compaction** — `compact` consumes the full event log. Operators streaming events through to the model do so via `SessionGraph::current_branch_messages()` (which already runs in O(n) over events).
- **Cache-control routing** — preserving Anthropic per-block cache control on retained turns. Today the compactor strips `cache_control` (constructs fresh `ContentPart::ToolResult` without the field). Operators relying on cached prompt prefixes apply the cache markers post-compaction in their own pipeline.

## References

- pydantic-ai issue #4137 — context-compaction RFC catalogues the footgun this ADR closes.
- ADR-0001 — invariant 1 (session is event SSoT). Compaction operates on the SSoT, never mutates it.
- ADR-0037 — `AuditSink`. Compaction is read-only over the audit trail; persisting compacted views is the operator's choice.
- v3 plan slice 109.
