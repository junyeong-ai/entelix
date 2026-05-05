# ADR 0080 — `RunBudget` five-axis usage cap

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: `RunBudget` is a five-axis usage cap (`request_limit`, `input_tokens_limit`, `output_tokens_limit`, `total_tokens_limit`, `tool_calls_limit`) attached to the `ExecutionContext` via `with_run_budget`. Pre-call axes (request count, tool calls) check before the wire roundtrip; post-call axes (token counts) check after the codec decodes the response. Counters are `Arc<RunBudgetState>` of atomic primitives — cloning the budget shares the underlying state, so sub-agent fan-out accumulates into the parent's counters without message passing. Breaches surface as `Error::UsageLimitExceeded { axis, limit, observed }` — distinct from `Provider` (retry classifiers short-circuit) and `InvalidRequest` (dashboards see budget signal as a first-class category). The cost meter (`entelix-policy::CostMeter`) stays separate per Fork 3 industry consensus — `RunBudget` covers token / request counts; cost per dollar lives on the policy layer.

## Context

pydantic-ai 1.90's `UsageLimits` (Fork 1 audit synthesis on `project_entelix_2026_05_05_master_plan.md`) is the cleanest cross-language reference: five axes, split before/after enforcement, one error type. Industry SDKs vary wildly past that minimum:

- **pydantic-ai 1.90** — five axes exactly, `UsageLimitExceeded` exception, sub-agent budget rollup via context manager. No native cost axis (caller adds via observer callback).
- **LangGraph 1.0 GA** — no built-in budget primitive; the pattern is per-step `recursion_limit` (count) + caller-side cost tracking.
- **OpenAI Agents SDK** — `Guardrails` (input/output filters) is the closest analogue; no token/request budget primitive.
- **Claude Agent SDK** — no budget surface; caller-managed.
- **rig** — none.
- **Vercel AI SDK 5** — `maxSteps` (recursion) + per-call `maxTokens` (vendor passthrough); no cumulative budget.
- **LiteLLM** — `BudgetManager` per cost (USD) axis only.

`pydantic-ai-middleware` adds a sixth USD axis to its budget; LiteLLM keeps cost-only. The split is contested. entelix's master plan picks the pydantic-ai shape (five count-and-token axes) and keeps the cost axis on `entelix-policy::CostMeter` — they are independent concerns: token / request counts are operational caps the SDK can enforce locally with zero pricing-table dependency; cost in dollars requires the pricing table the policy layer already owns.

Up to 1.0-RC.1 entelix exposed `RunOverrides::max_iterations` (graph recursion) and `entelix-policy::QuotaLimiter` (rate-limit + budget-USD). Per-run token / request / tool-call budgets had no surface — operators that wanted "this run cannot exceed N model calls and M tokens" had to layer it themselves through OTel observers.

## Decision

### `RunBudget` lives in `entelix-core`

Not in `entelix-policy`. `RunBudget` is per-run, in-process, and sub-agent-aware — the natural carrier is `ExecutionContext::extension::<RunBudget>()`. `entelix-policy::CostMeter` stays on the policy layer because it owns a pricing table and tenant ledger; mixing the two concerns would couple budget enforcement to vendor pricing data the budget itself does not need.

The crate-level position: `crates/entelix-core/src/run_budget.rs`. Not behind a feature flag — the type is part of the canonical `ExecutionContext` surface every dispatch site reads.

### Five axes, split enforcement

```rust
pub struct RunBudget {
    request_limit: Option<u32>,
    input_tokens_limit: Option<u64>,
    output_tokens_limit: Option<u64>,
    total_tokens_limit: Option<u64>,
    tool_calls_limit: Option<u32>,
    state: Arc<RunBudgetState>,
}
```

Pre-call axes (`request_limit`, `tool_calls_limit`) are checked at the dispatch site **before** the wire roundtrip — the SDK knows the caller is about to issue request `N+1` and refuses if the cap is `N`. Token axes are post-call: the budget sees `response.usage` only after the codec decodes, so the breach surfaces on the call that pushed the cumulative total past the limit.

The pre-call check uses `compare_exchange_weak` with `AcqRel` ordering to atomically increment-or-fail. `fetch_add` would let two concurrent dispatches both pass a `load() < limit` check and then both increment, overshooting the cap by one — the CAS loop is the only correct pattern under tokio's work-stealing executor.

### Atomic counters, Arc-shared across sub-agents

```rust
struct RunBudgetState {
    requests: AtomicU32,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    tool_calls: AtomicU32,
}
```

`RunBudget: Clone` clones the `Arc<RunBudgetState>` — the parent's budget and the sub-agent's budget point at the same counters. Sub-agent dispatch (`Subagent::execute`) inherits the parent's `ExecutionContext`, which carries the same budget through `Extensions`, which produces the same `Arc<RunBudgetState>` on lookup. The atomics are the cross-task synchronisation primitive — no message passing, no parent→child notification, no per-instance counter that needs reconciliation.

This is materially stronger than pydantic-ai's Python-GIL-coordinated `UsageLimits` (which is in-process anyway but does not surface concurrent-correctness as an explicit guarantee). entelix's tokio-aware design holds under work-stealing parallelism.

### Wiring at dispatch sites — inline, not Layer

`ChatModel::complete_full` / `complete_typed` / `stream_deltas` and `ToolRegistry::dispatch` read the budget from `ctx.run_budget()` and inline `check_pre_request` / `check_pre_tool_call` (pre-call) and `observe_usage` (post-call, `Ok` branch only). No separate `RunBudgetLayer` — three reasons:

1. **Pre-call checks need the early-exit semantic.** Layer-based budgets that wrap the inner service can refuse, but the refuse path is visually identical to a transport-class failure. Inline check at the caller's `?` lets the operator see budget refusal as a typed `Error::UsageLimitExceeded` before the layer stack even spins up.
2. **Sub-agent rollup is automatic.** A layer would need to be re-attached to every sub-agent's `ChatModel`, which `Subagent::from_whitelist` (ADR-0035) explicitly avoids — sub-agents inherit the parent's layer factory by `Arc`. Inline reads from `ctx.run_budget()` flow through the same `Extensions` clone that sub-agents already use.
3. **Streaming completion wrap composes cleanly.** `stream_deltas` already wraps the `completion` future for OtelLayer / PolicyLayer (G-1). Adding budget observation to the same wrap pattern keeps the Ok-branch semantic consistent across one-shot and streaming dispatch.

The streaming wrap mirrors G-1 exactly: pre-call `check_pre_request`, then on `completion.await.is_ok()` call `observe_usage(&response.usage)`. A stream that errors mid-flight resolves `completion` to `Err` and skips the budget observation entirely (invariant 12 — no budget drain on the error branch).

### `Error::UsageLimitExceeded` typed variant

The error enum gains a ninth variant:

```rust
Error::UsageLimitExceeded {
    axis: UsageLimitAxis,
    limit: u64,
    observed: u64,
}
```

Distinct from `Provider` (retry classifiers short-circuit — a budget breach does not retry), distinct from `InvalidRequest` (dashboards aggregate budget signals separately from caller-input mistakes), distinct from `Cancelled` / `DeadlineExceeded` (budget axis is a different ceiling shape). `axis` is `UsageLimitAxis { Requests, InputTokens, OutputTokens, TotalTokens, ToolCalls }` — operators inspecting a `UsageLimitExceeded` know exactly which axis fired without parsing strings.

`Error: #[non_exhaustive]` lets the new variant land as a non-breaking change for downstream `match` arms; the `LlmRenderable for Error` and stream `clone_error` sites add the new arm in this commit.

The `LlmRenderable` arm renders `"request quota reached"` to the model — operational signals do not flow to the LLM channel (invariant 16). Operators see the full `axis: requests, observed: 11, limit: 10` through `Display` / OTel attributes / sinks.

### `UsageSnapshot` — frozen counter view

`RunBudget::snapshot()` produces an owned `UsageSnapshot { requests, input_tokens, output_tokens, tool_calls }`. Callers that need the final tally (the agent runtime emitting `AgentEvent::Complete`, the upcoming `AgentRunResult<S>` envelope in B-5) read the snapshot once at terminal — no live-`Arc` leak through the public surface.

`UsageSnapshot::total_tokens()` derives `input + output` so callers don't reimplement the sum.

## Consequences

**Positive**:

- Operators write `ExecutionContext::new().with_run_budget(RunBudget::unlimited().with_total_tokens_limit(50_000).with_tool_calls_limit(20))` and the SDK enforces the caps across the whole run, including every sub-agent fan-out, with zero per-call wiring.
- `Error::UsageLimitExceeded` is a typed first-class signal — retry classifiers short-circuit, OTel dashboards aggregate the axis, the `LlmRenderable` rendering keeps operational signals off the model channel.
- Sub-agent rollup is automatic — the parent's budget is shared via `Arc<RunBudgetState>` through every `ExecutionContext` clone the sub-agent receives. No message passing, no reconciliation, work-stealing-correct.
- Pre-call atomic CAS prevents overshooting the cap under concurrent dispatch — two parallel `complete_full` calls racing on the same `n+1` slot do not both pass the check.
- Token observation is post-decode `Ok`-branch only — invariant 12 transactional semantics hold for budget the same way they hold for cost.
- The cost-USD axis stays on `entelix-policy::CostMeter` — separation of concerns matches Fork 3 industry consensus (LiteLLM keeps it cost-only; pydantic-ai keeps it count/token-only; entelix splits the two).

**Negative**:

- One additional public type (`RunBudget`), one additional snapshot type (`UsageSnapshot`), one additional axis enum (`UsageLimitAxis`), one additional `Error` variant. Per ADR-0064 §"Public-API baseline contract" the typed-strengthening drift is authorised at the 1.0 RC contract boundary.
- `Error: #[non_exhaustive]` already lets the new variant land non-breaking, but every internal match site in `entelix-core` added the new arm (`LlmRenderable for Error`, `stream::clone_error`). External crates that exhaustively match on `Error` need to add the arm — but invariant 14's `#[non_exhaustive]` discipline already requires every external match to fall through, so this is an enforcement pattern, not a regression.
- Inline check at every dispatch site (`complete_full`, `complete_typed`, `stream_deltas`, `ToolRegistry::dispatch`) means future dispatch sites need the same wire — a Layer-based design would carry the wire in one place. The trade-off is intentional (see Decision §"Wiring at dispatch sites — inline, not Layer").

**Migration outcome (one-shot, no shim)**: `RunOverrides::max_iterations` continues to live as the graph-level recursion cap (ADR-0028) — graph recursion is a different ceiling shape from `RunBudget::request_limit` (one is "how many graph steps", the other is "how many model dispatches"). They coexist; operators set both for full coverage.

## References

- CLAUDE.md invariant 12 — cost computed transactionally (post-decode `Ok` branch only — same pattern for budget observation)
- CLAUDE.md invariant 14 — no backwards-compatibility shims
- CLAUDE.md invariant 17 — heuristic policy externalisation (`UsageLimitAxis` typed enum, not stringly-typed)
- ADR-0028 — `recursion_limit` graph ceiling (coexists with `RunBudget::request_limit`)
- ADR-0035 — managed-agent shape (sub-agents inherit `ExecutionContext` and through it the parent's `RunBudget`)
- ADR-0064 — 1.0 release charter
- ADR-0075 — streaming dispatch on the tower::Service spine (post-completion `Ok` branch is the canonical observation site, mirrored here)
- Master plan May-05 §"B-4 — RunBudget 5-axis"
- Fork 1 (pydantic-ai 1.90 deep research) — five-axis `UsageLimits` shape, split enforcement pattern
- Fork 3 (industry survey) — cost axis kept separate from count/token axes per LiteLLM / pydantic-ai consensus
- `crates/entelix-core/src/run_budget.rs` — `RunBudget` + `UsageSnapshot` + `UsageLimitAxis` + atomics + builder
- `crates/entelix-core/src/error.rs` — `Error::UsageLimitExceeded` variant
- `crates/entelix-core/src/context.rs` — `ExecutionContext::with_run_budget` + `run_budget()` accessor
- `crates/entelix-core/src/chat.rs` — pre-call check + post-decode observation in `complete_full` / `complete_typed` / `stream_deltas`
- `crates/entelix-core/src/tools/registry.rs` — pre-call `check_pre_tool_call` in `dispatch`
