# ADR 0069 — `RunOverrides` for per-call ChatModel + graph knobs

**Status**: Accepted
**Date**: 2026-05-02
**Decision**: Per-call overrides flow through `ExecutionContext::extension::<RunOverrides>()`. Operators attach via `ctx.add_extension(...)` or the `Agent::execute_with(input, overrides, ctx)` convenience. `ChatModel::complete_full` / `stream_deltas` and `CompiledGraph::invoke` / `stream` consult the extension and patch model identifier, system prompt, and effective recursion limit — the compile-time recursion limit stays authoritative, operators can only lower the cap, never raise it.

## Context

`ChatModel` and `CompiledGraph` are built once at agent construction time and reused across many requests. A few parameters benefit from per-call override without rebuilding either component:

- **Cheap-model classification routes** inside an agent otherwise pinned to an expensive model. Picking `claude-3-5-haiku` for a triage step inside an otherwise `claude-3-5-sonnet`-pinned agent saves an order of magnitude on cost.
- **Request-specific system prompts.** A multi-tenant deployment that keys the system prompt on the request's tenant — e.g. injecting tenant-specific persona instructions — without minting one `ChatModel` per tenant.
- **Per-call recursion clamp.** An experimental run that wants to fail fast at iteration 5 instead of the agent's compile-time 25, without rebuilding the graph.

Pre-1.0 the SDK had none of these as per-call knobs. Operators worked around by minting one `ChatModel` per variant — fine in low-cardinality setups, untenable when the variation is per-request. The audit identified this as the second-largest gap (after `ToolDispatchScope`) blocking per-call configurability.

## Decision

A single `entelix_core::RunOverrides` type carries every per-call override. It rides in `ExecutionContext::extension::<RunOverrides>()` so the layered `tower::Service` stack and downstream consumers (`ChatModel`, `CompiledGraph`) pick it up automatically without API surface changes:

```rust
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct RunOverrides {
    model: Option<String>,
    system_prompt: Option<SystemPrompt>,
    max_iterations: Option<usize>,
}
```

### Why one type, not three

Three separate extension types (`ModelOverride`, `SystemPromptOverride`, `MaxIterationsOverride`) would be more orthogonal, but operators thread *bundles* of overrides per call, not single fields. One type keeps the call site terse:

```rust
agent.execute_with(input, RunOverrides::new()
    .with_model("haiku")
    .with_max_iterations(10), &ctx).await?;
```

Compared to three:

```rust
agent.execute(input, &ctx
    .add_extension(ModelOverride("haiku"))
    .add_extension(MaxIterationsOverride(10))).await?;
```

Bundling also gives `Agent::execute_with` a single overrides arg, keeping the per-call knob count fixed at one regardless of how many fields `RunOverrides` grows.

### Why the compile-time recursion limit stays authoritative

`with_max_iterations(n)` clamps to `min(compile_time_cap, n)` — the compile-time cap (set on `StateGraph::with_recursion_limit`) is the operator's design-time choice; per-call overrides can lower it for safety but must not raise it. This pins F6 mitigation (recursion-limit cap) at the level the graph was built with, regardless of any caller's per-request choice.

### Why system_prompt is replace, not append

`with_system_prompt` *replaces* the configured `SystemPrompt` rather than appending to it. Append-by-default would surprise operators in two ways: (a) a per-call instruction would inherit prior-call-flavor leakage from the configured prompt, and (b) `SystemPrompt::cached` blocks mix awkwardly with append. Operators that want to extend rather than replace pre-compose the desired `SystemPrompt` themselves:

```rust
let mut combined = SystemPrompt::text("Be terse.");
combined.append_text("Format the answer as bullet points.");
RunOverrides::new().with_system_prompt(combined)
```

### Why two entry points (`add_extension` + `execute_with`)

`add_extension` is the canonical mechanism — every other `ExecutionContext` extension follows the same shape. `Agent::execute_with` is a convenience that does the `clone() + add_extension` chain so the call site reads as `agent.execute_with(input, overrides, ctx)` instead of `agent.execute(input, &ctx.clone().add_extension(overrides))`. Both routes wire identical extension state; choose by call-site terseness.

### What this does NOT cover

- **Per-call temperature / top_p / stop sequences.** Demand-driven additions extend `RunOverrides` (the type is `#[non_exhaustive]`). The initial surface ships only the three commonly-needed fields (`model`, `max_iterations`, `system_prompt`); future fields land in follow-up slices.
- **Tool-list overrides.** Per-call tool whitelisting belongs at the registry level — operators reach for `ToolRegistry::restricted_to(allowed)?` (slice 111) and pass the narrowed registry to a fresh `Agent`. Bundling it into `RunOverrides` would conflate request-shape with capability-narrowing.
- **`ChatModel::stream_deltas` recursion limit.** `stream_deltas` does not run a graph loop; the recursion limit applies only to `CompiledGraph::execute_loop_inner` and the streaming-mode equivalent.

## Consequences

- New `entelix_core::RunOverrides` (also reachable via `entelix::RunOverrides`).
- `ChatModel::complete_full` + `stream_deltas` now consult `ctx.extension::<RunOverrides>()` and patch the underlying `ModelRequest` before encode.
- `CompiledGraph::execute_loop_inner` + the streaming-mode `build_stream` clamp the effective recursion limit to `min(compile_time_cap, overrides.max_iterations)`.
- `Agent::execute_with(input, overrides, ctx)` convenience added.
- Six regression tests: 2 in `chat_model.rs` (model + system_prompt patch + absent-keeps-defaults), 2 in `basic.rs` (max_iterations lowers, max_iterations cannot raise), 2 in `overrides.rs` unit (constructor empty, setter chain).

## References

- ADR-0017 — `ExecutionContext::extensions` (typed cross-cutting carrier, the mechanism this ADR builds on).
- ADR-0035 — `Subagent` narrowed registry (`restricted_to`); per-call tool overrides explicitly out of scope here.
- F6 mitigation (PLAN.md) — recursion-limit cap on `StateGraph` (per-call override can only lower, never raise).
- `crates/entelix-core/src/overrides.rs` — type + tests.
- `crates/entelix-core/src/chat.rs::apply_run_overrides` — ChatModel patch site.
- `crates/entelix-graph/src/compiled.rs::effective_recursion_limit` — graph clamp site.
- `crates/entelix-agents/src/agent/mod.rs::Agent::execute_with` — convenience entry.
