# entelix-session

Event-sourced session state (invariant 1 — session is event SSoT). `SessionGraph::events: Vec<GraphEvent>` is the only first-class data for audit; nodes / branches / checkpoints are derived.

## Surface

- **`SessionGraph`** — in-memory append-only event log. `append(event)` / `events_since(cursor)` / `current_branch_messages()` / `fork(branch_at, new_thread_id)`.
- **`GraphEvent` enum** — `UserMessage`, `AssistantMessage`, `ThinkingDelta`, `ToolCall`, `ToolResult`, `Warning`, `RateLimit`, `BranchCreated`, `CheckpointMarker`, `Cancelled`, `Interrupt`, `Error`, plus the audit-channel variants `SubAgentInvoked` / `AgentHandoff` / `Resumed` / `MemoryRecall` / `UsageLimitExceeded`. Full field list in `crates/entelix-session/src/event.rs`.
- **`SessionLog` trait** — backend-agnostic durable persistence for `GraphEvent`. `append(key, events) -> Result<u64>` (batched, returns highest ordinal) / `load_since(key, cursor)` / `archive_before(key, watermark)`. Reference impl: `InMemorySessionLog` (test/dev). Production: `PostgresSessionLog` / `RedisSessionLog` in `entelix-persistence`.
- **`SessionAuditSink`** — fire-and-forget `AuditSink` adapter that maps `record_*` calls onto `SessionLog::append` of the corresponding `GraphEvent` variant (invariant 18). Sync `&self` so emit sites stay `.await`-free.
- **`Compactor` trait** + **`CompactedHistory`** + **`HeadDropCompactor`** — type-enforced conversation compaction (invariant 21). `tool_call` / `tool_result` pair invariant is structurally impossible to violate: `ToolPair` fields are private; the only construction path is `CompactedHistory::group(events)`; external `Compactor` impls drop or pass through `ToolPair`s and rebuild via `CompactedHistory::from_turns(turns)`. `HeadDropCompactor` ships as the reference "drop oldest until fits" strategy.

## Crate-local rules

- **No message cache anywhere** — derived state (current branch, conversation messages) is recomputed from the event log on demand. A field like `cached_messages: Vec<Message>` on `SessionGraph` violates invariant 1 and is an instant-reject review comment.
- **Append-only contract** — `SessionGraph::events` is `Vec<GraphEvent>`, never mutated in place after append. Branching is a new event (`BranchCreated { branch_at, new_thread_id }`), not a vector mutation.
- **Backend isolation tests at the persistence layer** (invariant 13) — `tests/namespace_isolation.rs` covers `InMemorySessionLog`. Production backends (`PostgresSessionLog`, `RedisSessionLog`) carry their own collision suites in `entelix-persistence/tests/`.
- **`SessionAuditSink` failures land in `tracing::warn!` and never propagate** — the audit channel is one-way by contract. A failure to persist an audit event must NOT abort the request that triggered it.
- **Archival watermark is operator-driven** — `archive_before(key, watermark)` is the only event-removal path. Per-event delete does not exist (an event is permanent until archived).

## Forbidden

- A field on `SessionGraph` or any `SessionLog` impl that caches derived state (current branch messages, current node id) outside the event log.
- Mutating a `GraphEvent` after append. Corrections happen as a new event referencing the prior one.
- A `SessionLog` impl that drops the `tenant_id` axis (invariant 13). Cross-tenant event leakage is a security boundary, not a performance hint.

## References

- Root `CLAUDE.md` invariant 1 (session-as-event-SSoT) + §"Anthropic managed-agent shape" (Session/Harness/Hand decoupling).
