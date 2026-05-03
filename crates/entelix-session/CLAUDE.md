# entelix-session

Event-sourced session state (invariant 1 — session is event SSoT). `SessionGraph::events: Vec<GraphEvent>` is the only first-class data for audit; nodes / branches / checkpoints are derived.

## Surface

- **`SessionGraph`** — append-only event log. `append(event)` / `events_since(cursor)` / `current_branch_messages()` / `events_for_node(node_id)` / `fork(parent_node)`.
- **`GraphEvent` enum** — `UserMessage`, `AssistantMessage`, `ToolCall`, `ToolResult`, `BranchCreate`, `Checkpoint`, plus the ADR-0037 audit-channel variants `SubAgentInvoked` / `AgentHandoff` / `Resumed` / `MemoryRecall`.
- **`SessionLog` trait** — backend-agnostic persistence for `GraphEvent`. `append_event(thread_id, event)` / `load_since(thread_id, cursor)` / `archive_threads(watermark)`. Reference impl: `InMemorySessionLog` (test/dev). Production: `PostgresSessionLog` / `RedisSessionLog` in `entelix-persistence`.
- **`SessionAuditSink`** — fire-and-forget `AuditSink` adapter that maps `record_*` calls onto `SessionLog::append` of the corresponding `GraphEvent` variant (ADR-0037, invariant 18). Sync `&self` so emit sites stay `.await`-free.

## Crate-local rules

- **No message cache anywhere** — derived state (current branch, conversation messages) is recomputed from the event log on demand. A field like `cached_messages: Vec<Message>` on `SessionGraph` violates invariant 1 and is an instant-reject review comment.
- **Append-only contract** — `SessionGraph::events` is `Vec<GraphEvent>`, never mutated in place after append. Branching is a new event (`BranchCreate { parent_node }`), not a vector mutation.
- **Backend isolation tests at the persistence layer** (invariant 13) — `tests/namespace_isolation.rs` covers `InMemorySessionLog`. Production backends (`PostgresSessionLog`, `RedisSessionLog`) carry their own collision suites in `entelix-persistence/tests/`.
- **`SessionAuditSink` failures land in `tracing::warn!` and never propagate** — the audit channel is one-way by contract (ADR-0037). A failure to persist an audit event must NOT abort the request that triggered it.
- **Archival watermark is operator-driven** — `archive_threads(watermark)` is the only thread-removal path. Per-event delete does not exist (an event is permanent until the whole thread is archived).

## Forbidden

- A field on `SessionGraph` or any `SessionLog` impl that caches derived state (current branch messages, current node id) outside the event log.
- Mutating a `GraphEvent` after append. Corrections happen as a new event referencing the prior one.
- A `SessionLog` impl that drops the `tenant_id` axis (invariant 13). Cross-tenant event leakage is a security boundary, not a performance hint.

## References

- ADR-0037 — `AuditSink` typed channel + `SessionAuditSink` adapter.
- ADR-0039 — `Namespace::parse` (audit-key reversibility for `MemoryRecall::namespace_key`).
- F1 mitigation — session-as-event-SSoT, no message cache.
- Anthropic managed-agents — Session/Harness/Hand decoupling (`docs/architecture/managed-agents.md`).
