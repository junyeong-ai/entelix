# CLAUDE.md — entelix

**General-purpose agentic-AI SDK.** Build any agent that runs an LLM-driven control loop with tools, memory, and durable state. Not coding-agent-specific, not vertical-specific — every domain-shaped concern (sandboxing, custom approvers, ambient task-locals, vendor-specific persistence backends) lands as a `trait` operators implement or a `feature` flag they enable, never as built-in opinion.

Web-service-native Rust. Anthropic *managed-agents* shape, LangChain + LangGraph parity, multi-tenant primitives, OpenTelemetry GenAI semconv.

## Architecture invariants — these are LAW

A PR contradicting any invariant is invalid regardless of how clean the code looks. CI enforces what is enforceable; reviewer enforces the rest.

### State

1. **Session is event SSoT** — `entelix_session::SessionGraph::events: Vec<GraphEvent>` is the only first-class data for audit. Nodes / branches / checkpoints are derived. No message cache anywhere.
2. **Harness is stateless** — `Agent` instances hold no persistent state beyond in-memory request scope. Crash → wake(thread_id) → resume from event log + checkpoint.
3. **3-tier state separation** — *StateGraph state* (per-thread typed working memory) vs *SessionGraph events* (per-thread audit log) vs *Memory Store* (cross-thread persistent knowledge). Orthogonal.

### Contracts

4. **Hand contract is `Tool::execute(input, ctx) → output`** — the only behavioural method on `Tool`. `metadata()` is a fixed-shape descriptor (name, schema, effect class) returned by reference; impls construct it once and the registry may inspect it many times. No back-channels, no shared mutable state, no extra dispatch methods.
5. **Provider IR before wire format** — every model call passes through `entelix_core::ir::ModelRequest`. Vendor JSON is codec-internal. Public API never returns vendor-shaped JSON.
6. **Lossy encoding emits warnings** — when a codec drops information the IR carried, emit `ir::ModelWarning::LossyEncode`. Silent loss is a bug.

### Composition

7. **Runnable is the composition contract** — anything composable implements `Runnable<I, O>`. `.pipe()` is the universal connector.
8. **StateGraph is the control-flow contract** — multi-step / conditional / cyclic flows are `StateGraph<S>`. Ad-hoc loops in user code are a smell.

### Security

9. **No filesystem, no shell** — first-party crates do NOT import `std::fs`, `std::path::Path`/`PathBuf` (except as opaque IDs), `std::process`, `std::os::unix::process`, `tokio::fs`, `tokio::process`, `landlock`, `seatbelt`, `tree-sitter*`, `nix`. Enforced by `cargo xtask no-fs`.
10. **Tokens never reach Tool input** — credentials live exclusively in `entelix_core::auth::CredentialProvider`. Headers are added in transport. `ExecutionContext` does NOT embed `CredentialProvider`. Type-checked.
11. **Multi-tenant Namespace is mandatory** — `entelix_memory::Namespace` requires a `tenant_id`. Cross-tenant data leak is structurally impossible by API design.

### Cost & Operations

12. **Cost is computed transactionally** — `gen_ai.usage.cost`, `gen_ai.tool.cost`, `gen_ai.embedding.cost` are emitted only inside the `Ok` branch of the corresponding service / wrapper. A failed call never produces a phantom charge.
13. **Backend isolation is row-level** — every `Store<V>`, `VectorStore`, `Checkpointer`, `SessionLog` impl includes namespace-collision tests at the persistence layer (testcontainers or equivalent), not only at the in-memory layer. Trusting the rendered namespace key alone is insufficient.

### Engineering

14. **No backwards-compatibility shims** — when a name, type, or signature changes, delete the old in the same PR. No `// deprecated`, no `pub use OldName as NewName`, no fallback constructors. Enforced by `cargo xtask no-shims` (forbids `#[deprecated]`, `// deprecated`, `// formerly …`, `// removed for backcompat`, `pub use X as YOld`).
15. **No silent fallback** — codecs, transports, and the cost meter never substitute a plausible-looking default for a missing or unknown vendor signal. Information loss surfaces through one of two channels: `ModelWarning::LossyEncode { field, detail }` for coerced values, or `StopReason::Other { raw }` for unknown vendor reasons. Vendor-mandatory IR fields (`Anthropic` `max_tokens`, …) are rejected at encode time with `Error::invalid_request`. Missing decode signals surface as `Other{raw:"missing"}` plus a `LossyEncode` warning. Default-injecting a value (e.g. `max_tokens.unwrap_or(4096)`, `cache_rate.unwrap_or(input/10)`, `stopReason.unwrap_or(EndTurn)`) is a bug regardless of how reasonable the default looks. Enforced by `cargo xtask silent-fallback` plus the `tests/codec_consistency_matrix.rs` regression suite. ADR-0032.
16. **LLM / operator channel separation** — `Display` text, source-error chains, vendor status codes, internal type identifiers, raw distance scores, and ISO-8601 timestamps never reach the model. Tool errors flow to the LLM through `entelix_core::LlmFacingError::render_for_llm`; tool input schemas through `LlmFacingSchema::strip` (drops `$schema` / `title` / `$defs` / `$ref` / integer width hints); tool outputs through default-deny exposure knobs (`HttpFetchToolBuilder::with_exposed_response_headers`, `MemoryToolConfig::expose_metadata_fields`, `MemoryToolConfig::with_entity_temporal_signals`). `AgentEvent::ToolError` carries two fields — `error` (operator-facing) + `error_for_llm` (model-facing), and the audit projection routes the model-facing rendering into `GraphEvent::ToolResult` so replay does not re-leak operator content. Operator channels (sinks, OTel, logs) keep the full diagnostic. Enforced by `crates/entelix-tools/tests/llm_context_economy.rs` regression suite. ADR-0033.
17. **Heuristic policy externalisation** — embedded heuristics (retry / fallback / routing / recursion / probability literals) live on typed `*Policy` / `*Decision` surfaces operators override, not in dispatch hot paths. Vendor-authoritative signals beat self-jitter: `RetryClassifier::should_retry → RetryDecision { retry, after }` honours `Error::Provider::retry_after` populated from the vendor's `Retry-After` header; `RetryService` propagates that cooldown ahead of its own backoff. `Error::Provider::kind: ProviderErrorKind { Network, Tls, Dns, Http(u16) }` replaces the `status: 0` sentinel so retry classifiers branch on a typed signal. `RetryService` stamps an idempotency key on the request's `ExecutionContext` on first entry — every retry attempt forwards it on the `Idempotency-Key` header, so vendor dedupe collapses N attempts into one logical call. `SupervisorDecision { Agent(String), Finish }` replaces the prior `Runnable<…, String>` + `SUPERVISOR_FINISH` sentinel; recipe routers cannot hallucinate enum variants. `ReActAgentBuilder::with_recursion_limit` exposes the graph cap on the recipe surface. Probability literals (`0.X`) in codec / transport / agent / cost paths are forbidden by `cargo xtask magic-constants`; new sites move onto an existing `*Policy` or carry a `// magic-ok:` marker. ADR-0034.
18. **Managed-agent lifecycle is auditable** — sub-agent dispatch, supervisor handoff, resume-from-checkpoint, and long-term memory recall emit through the typed `entelix_core::AuditSink` channel that operators wire onto `ExecutionContext::with_audit_sink`. `entelix-session` ships `SessionAuditSink` — a fire-and-forget adapter that maps each `record_*` call onto `SessionLog::append` of the corresponding `GraphEvent` variant (`SubAgentInvoked`, `AgentHandoff`, `Resumed`, `MemoryRecall`). The trait surface is typed `record_*` verbs (not `emit(GraphEvent)`) so `entelix-tools` / `entelix-graph` emit sites do not depend on `entelix-session`; the methods are sync `&self` so emit sites avoid `.await` ceremony in hot dispatch loops. Audit-sink failures land in `tracing::warn!` and never propagate back — the audit channel is one-way by contract. `MemoryRecall` captures the *retrieval act* (`tier`, `namespace_key`, `hits`) but never the corpus; the model-facing content already lands in the conversation transcript. Recipes that do not wire a sink incur zero overhead — `ctx.audit_sink()` returning `None` makes every emit site a no-op. ADR-0037.

## Lock ordering

Multi-lock acquisition order — always nest in this direction, never the reverse:

```
tenant > session > checkpoint > memory > tool_registry > orchestrator
```

**Never hold any lock across `.await` on a user-supplied future** (`Tool::execute`, `Layer::Service::call`). Drop the guard before `.await`, or scope the guard inside a non-async block.

## Cancellation

Every async API that may run > 100ms accepts `tokio_util::sync::CancellationToken` (re-exported as `entelix_core::cancellation::CancellationToken`). The carrier is `ExecutionContext::cancellation()`. Long loops poll `ctx.is_cancelled()`. Deadlines are `ExecutionContext::deadline()`; sub-agents inherit automatically.

## Error conventions

- **Module-internal errors** are typed enums (`CodecError`, `GraphError`, `McpError`, `PersistenceError`, …).
- **Public crate API** surfaces `entelix_core::Error`. Never expose `anyhow::Error` from public.
- `Error` is `#[non_exhaustive]` with eight variants — four user-facing failure modes, three runtime control signals, and one passthrough:
  - `Error::InvalidRequest(Cow<'static, str>)` — encode-time preflight rejections. Helper: `Error::invalid_request(msg)`.
  - `Error::Config(Cow<'static, str>)` — configuration mistakes detected at construction. Helper: `Error::config(msg)`.
  - `Error::Provider { kind: ProviderErrorKind, message: String, retry_after: Option<Duration> }` — vendor returned a failure response or transport never reached the vendor. `kind` is typed (`Network` / `Tls` / `Dns` / `Http(u16)`) so retry classifiers branch on the category instead of pattern-matching on a `status: 0` sentinel (invariant 17). Helpers: `Error::provider_http(status, msg)`, `Error::provider_network(msg)`, `Error::provider_tls(msg)`, `Error::provider_dns(msg)`. `Error::with_retry_after(self, Duration) -> Self` attaches the vendor's `Retry-After` hint so `RetryService` honours it ahead of its own backoff.
  - `Error::Auth(AuthError)` — credential resolution / token failures. Distinct from `Provider` so retry policies can split "model is down" from "key is bad".
  - `Error::Cancelled` — `ExecutionContext` cancellation token fired.
  - `Error::DeadlineExceeded` — `ExecutionContext` deadline hit.
  - `Error::Interrupted { payload: serde_json::Value }` — graph node requested human-in-the-loop. Caught by the executor; resume via `CompiledGraph::resume_with`.
  - `Error::Serde(serde_json::Error)` — JSON failure at an entelix-managed boundary (codec, tool I/O, persistence). `#[from]` enables `?` chaining.
- All errors are `Debug + Display + Send + Sync + 'static`. `Result<T> = std::result::Result<T, Error>`.

## Naming

Reference: `.claude/rules/naming.md` (mirrors `docs/adr/0010-naming-taxonomy.md`). Type-suffix table (`*Codec`, `*Transport`, `*Provider`, …), the `Runnable<Verb>` composition prefix, builder verb-prefix exception (`with_*` / `add_*` / `set_*` / `register`), and the ctx-first / ctx-last parameter-ordering split all live there. Forbidden suffixes (`*Engine`, `*Wrapper`, `*Handler`, `*Helper`, `*Util`) and `get_xxx` accessors are reviewer-rejected; `cargo xtask naming` enforces a subset.

## Anthropic managed-agent shape — non-negotiable

This SDK mirrors [Anthropic's managed-agents pattern](https://www.anthropic.com/engineering/managed-agents):

- **Session** = `SessionGraph` (event log, externally stored)
- **Harness** = `Agent` + codecs (stateless brain, replaceable)
- **Hand** = `Tool` trait (sandbox-agnostic, single `execute` interface)
- **Brain passes hand** — `Subagent::from_whitelist(model, &parent_registry, &[...])` narrows the parent `ToolRegistry` through `restricted_to` / `filter`. The narrowed view shares the parent's layer factory by `Arc` — `PolicyLayer`, `OtelLayer`, retry middleware all apply transparently to sub-agent dispatches. `Subagent` *never* constructs a fresh `ToolRegistry::new()`; the raw `Vec<Arc<dyn Tool>>` surface does not exist on this path. Enforced by `cargo xtask managed-shape` plus `tests/subagent_layer_inheritance.rs`. ADR-0035.
- **Lazy provisioning** — MCP / cloud connections opened on tool call
- **Wake / resume** — first-class

Reject feature proposals that blur these boundaries: harness caching session state, tool inputs touching credentials, sub-agent paths that bypass the parent registry's layer stack.

## Workspace layout

DAG root: `entelix-core` (depends on no other entelix crate). Sub-crate `CLAUDE.md` files live under each `crates/<name>/CLAUDE.md` and are lazy-loaded when Claude reads files in that crate.

Key references:

- Logical-flaw register (F1-Fn) → `docs/adr/` retrospective ADRs document each mitigation; reviewer rejects PRs reintroducing any.
- Feature flags → facade `crates/entelix/Cargo.toml` is the **single source of truth**; `full` aggregates everything. Never enumerate the list elsewhere — that drifts.
- Architecture decisions → `docs/adr/`.
- Public-API baselines → `docs/public-api/<crate>.txt` (drift gate).

## Commands

```bash
# Standard gates — all must be green before merge
cargo fmt --all -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo clippy --workspace --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo test --workspace --all-features

# Run every static-analysable invariant in canonical CI order
cargo xtask invariants

# Or per-invariant — each subcommand maps to one CLAUDE.md invariant or
# ADR. Implementations live in xtask/src/invariants/<name>.rs as typed-AST
# visitors over `syn::File` and `toml_edit::DocumentMut` (ADR-0073).
cargo xtask no-fs                    # invariant 9
cargo xtask managed-shape            # invariants 1, 2, 4, 10 + ADR-0035
cargo xtask naming                   # ADR-0010 taxonomy + ctx-position
cargo xtask surface-hygiene          # #[non_exhaustive] + #[source] / #[from]
cargo xtask silent-fallback          # invariant 15 + ADR-0032
cargo xtask magic-constants          # invariant 17 + ADR-0034
cargo xtask no-shims                 # invariant 14
cargo xtask lock-ordering            # await_holding_lock pinned at deny
cargo xtask dead-deps                # workspace.dependencies hygiene
cargo xtask facade-completeness      # every pub item reachable via `entelix::*`
cargo xtask doc-canonical-paths      # live docs use facade canonical paths

# Network-bound / cargo-subprocess gates — heavier, separate CI jobs
cargo xtask supply-chain             # cargo audit (RustSec) + cargo deny
cargo xtask feature-matrix           # each facade feature compiles alone
cargo xtask public-api               # per-crate public-API drift baseline

# Refreeze public-API baselines after a deliberate, ADR-documented change
cargo xtask freeze-public-api [<crate>...]

# Live integration (requires API keys, opt-in)
cargo test --workspace --all-features -- --ignored
```

All gates run in CI. PR cannot merge with red gate.
