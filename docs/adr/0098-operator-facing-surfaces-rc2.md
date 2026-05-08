# ADR 0098 — Operator-facing surfaces in 1.0.0-rc.2

**Status**: Accepted
**Date**: 2026-05-08
**Decision**: Five new operator-supplied surfaces ship in 1.0.0-rc.2 to round out the agent-SDK feature parity bar set by Claude Agent SDK + LangChain + LangGraph + OpenAI Agents while keeping entelix's structural advantages (provider IR, sealed `RenderedForLlm`, `ToolPair`, mandatory `Namespace`, `std::fs`-free first-party). Each surface is operator-facing — codebase recipes don't dispatch through them by default — and pairs with self-tests rather than recipe wiring; this is intentional, not premature. The five close concrete parity gaps without forcing every consumer to take on the surface they don't need.

## Context

The 1.0 release positions entelix as a *general-purpose* SDK. Three orthogonal contracts shape that:

- The hand contract (`Tool::execute(input, ctx) → output`) — invariant 4.
- The composition contract (`Runnable<I, O>` + `.pipe()`) — invariant 7.
- The middleware contract (`tower::Layer<S>`) — invariant 17 (heuristic externalisation).

Every recipe (`create_chat_agent`, `create_react_agent`, `create_supervisor_agent`) is built from those three, and every operator extension point (PolicyLayer, OtelLayer, ApprovalLayer, ScopedToolLayer, RetryService) attaches at the middleware contract. The five surfaces below extend the operator-facing surface area along those same axes, never introducing a fourth contract.

A 6-probe audit against Claude Agent SDK + LangChain + LangGraph + OpenAI Agents 1.x scoped these as the genuine parity gaps; everything else the audits surfaced (fallback model, named hooks, output parser families, InjectedState sugar, session listing verbs) was either already covered by an existing surface or rejected as syntactic redundancy.

## Decision

### 1. `Toolset<D>` — reusable tool bundle declaration

`Toolset<D>` is the declaration unit operators name and reuse across agents:

```rust
let support_tools = Toolset::<()>::new("support")?
    .register(Arc::new(SearchTool::new(provider)))?
    .register(Arc::new(KbLookupTool::new(kb)))?;
let registry = ToolRegistry::<()>::new();
let registry = support_tools.install_into(registry)?;
```

Distinct from `ToolRegistry<D>`: the registry is the single dispatch path (invariant 4), `Toolset<D>` never dispatches. Duplicate names and ambiguous identifiers are rejected by the `identity::validate_config_identifier` pre-pass (shared with `Subagent::with_skills` etc.); `restricted_to(&[…])` narrows the bundle by strict-name match for sub-agent permissioning.

**Why**: Recipes need *capability declarations* the operator can attach to multiple agents (a "support team toolset", a "research toolset"). Without `Toolset`, operators wrote ad-hoc `Vec<Arc<dyn Tool>>` factories that drifted in name-validation behaviour. Closes the LangChain `Toolkit` parity gap with a typed-deps version.

**Non-goal**: a second dispatch path. Toolsets *install* into a registry; the registry remains the only place tools execute.

### 2. `ToolProgress` family — inflight progress reporting

Tools that take longer than a tight deadline (web search aggregation, multi-step shell sequences, large file processing) emit progress through a sink the operator wires onto `ExecutionContext`:

```rust
// In the tool impl:
async fn execute(&self, input: Input, ctx: &AgentContext<D>) -> Result<Value> {
    let progress = ctx.core().progress_sink();
    if let Some(sink) = progress {
        sink.record_progress(ToolProgress::started("indexing 1.4GB corpus"));
        // … long work …
        sink.record_progress(ToolProgress::complete("done"));
    }
    Ok(value)
}
```

`ToolProgressSink` is operator-supplied (pluggable into UI dashboards, OTel events, log streams). `CurrentToolInvocation` carries the `(name, tool_use_id)` identity through the sink so multi-tool dispatches multiplex correctly. `ToolProgressSinkHandle` rides on `ExecutionContext::extension` as a refcounted handle.

**Why**: Distinct from `AgentEvent::ToolInvoked` (one event per dispatch lifecycle); progress fires *during* one dispatch. Claude Agent SDK + LangGraph both expose this as a separate event class — entelix matches with a typed sink rather than a string log channel.

**Non-goal**: forced progress emission. Tools that can answer in tight latency stay silent; the sink option is `Option<Arc<dyn ToolProgressSink>>` on `ExecutionContext`.

### 3. `ToolHook` family — typed tool-dispatch policy hooks

Three lifecycle methods each return a typed decision the dispatch path honours:

```rust
struct CrmRedactor;

#[async_trait]
impl ToolHook for CrmRedactor {
    async fn before_tool(
        &self,
        request: &ToolHookRequest,
        _ctx: &ExecutionContext,
    ) -> Result<ToolHookDecision> {
        if request.tool_name == "crm_lookup" {
            let scrubbed = scrub_pii(&request.input);
            return Ok(ToolHookDecision::ReplaceInput(scrubbed));
        }
        Ok(ToolHookDecision::Continue)
    }
}

let registry = ToolHookRegistry::new().register(CrmRedactor);
let layer = ToolHookLayer::new(registry);
```

`ToolHookLayer` composes through `tower::Layer<S>` alongside `ApprovalLayer` / `ToolEventLayer` / `PolicyLayer` (invariant 19 — D-free layer ecosystem). Operator-side pattern matching on `request.tool_name` is intentional: `ToolHookRequest.tool_name` is a public field, so a hook impl writes one `if` instead of routing through a registry-side regex matcher (which would need its own grammar and tests).

**Why**: Claude Agent SDK ships `PreToolUse` / `PostToolUse` named hooks. entelix maps each onto `before_tool` / `after_tool` / `on_tool_error` returning typed `ToolHookDecision { Continue, ReplaceInput, Reject }` — three real composition shapes the prior surface (write a `tower::Layer` from scratch) didn't expose ergonomically.

**Non-goal**: a regex-keyed registry. Pattern routing belongs in operator code, not in the control surface.

### 4. `validate_vector_shape` + `first_non_finite_vector_value`

Embedder responses and operator-supplied vectors share two malformation modes — wrong dimension, non-finite element. The two helpers expose the predicate so every `Embedder` impl and every `VectorStore` impl rejects the same shapes with the same error:

```rust
// In a custom VectorStore impl:
fn add(&self, ns: &Namespace, id: &str, vector: &[f32]) -> Result<()> {
    validate_vector_shape("PgVectorStore::add", "vector", vector, self.dimension)?;
    // … insert …
    Ok(())
}
```

`validate_vector_shape` returns `Error::InvalidRequest` (caller-input boundary). `first_non_finite_vector_value` returns the offending position so embedder code can decide between `Error::Provider` (provider sent garbage) and `Error::InvalidRequest`.

**Why**: Three companion crates (`entelix-memory-pgvector`, `entelix-memory-qdrant`, future-vendor-3) inlining the same predicate diverged in error wording before this slice. Centralising the predicate normalises every backend's rejection shape.

**Non-goal**: vector arithmetic. The two helpers validate; they don't normalise, project, or compare.

### 5. `SupervisorDecision::Handoff { agent, payload }`

Supervisors now route to a named agent **with a typed JSON payload** that lands as the next agent's leading `system` message:

```rust
let router = RunnableLambda::new(|messages: Vec<Message>, ctx: ExecutionContext| async move {
    let summary = research_summary(&messages, &ctx).await?;
    Ok::<_, Error>(SupervisorDecision::handoff(
        "writer",
        json!({ "summary": summary, "evidence_count": 3 }),
    ))
});
let graph = create_supervisor_agent(
    router,
    vec![
        AgentEntry::new("researcher", researcher),
        AgentEntry::new("writer", writer),
    ],
)?;
```

`Agent` and `Handoff` share the same routing path (both consult `agent_name()`); the receiving agent_node drains `next_speaker.take()` and pushes a `Message::system("Handoff payload:\n{json}")` ahead of the agent's own turn. Audit emission (`record_agent_handoff`) fires on both variants — invariant 18 (managed-agent lifecycle is auditable) covers handoffs.

**Why**: OpenAI Agents' `Handoff(input_type, on_handoff)` is the typed-context-transfer pattern that prevents supervisors from round-tripping context through the model's natural-language channel (a prompt-injection / hallucination vector). `SupervisorDecision::Handoff` is the entelix-shaped equivalent with explicit payload semantics.

**Non-goal**: replacing `SupervisorDecision::Agent`. Agent stays the cheap, payload-free routing primitive; Handoff is reached for when the supervisor produces structured context.

## Why these five are operator-facing, not recipe-wired

Every recipe (`create_chat_agent`, `create_react_agent`, etc.) ships with *minimum-viable defaults*. Wiring the five surfaces by default would force every consumer to opt out of progress sinks, hook layers, and toolset declarations they don't need. The operator-facing pattern keeps the 5-line agent path lightweight while making the surfaces reachable when the deployment grows complex enough to justify them.

Self-tests (`Toolset` 7, `ToolProgress` 2, `ToolHook` inline, `vector` 2, `Handoff` 2) prove each surface's behaviour in isolation; consumers see the shape via examples 06+ as those grow.

## Consequences

### Positive

- **Parity with Claude Agent SDK + OpenAI Agents** on the five most-used 1.x surfaces (skills + tools + hooks + handoffs + progress) without copying their type shapes.
- **Operator-API discoverability** — five new types in the entelix facade re-export, each with a sealed self-test pinning behaviour.
- **Composable with the existing layer ecosystem** — `ToolHookLayer` plugs onto the same `Service<ToolInvocation>` spine as `ApprovalLayer` / `ToolEventLayer`, no new dispatch path.

### Negative

- Five new types in the facade re-export grow the surface area. `cargo xtask facade-curation` (this release) gates further drift; the five are pinned at the curated allowlist.
- Operators choosing to wire all five into a custom recipe author more boilerplate than a closed-form `create_<flavour>_agent` factory would. This is the cost of generality; closed-form factories ship as `entelix-agents` recipes when the pattern stabilises.

## References

- ADR-0089 — `ToolRegistry<D>` typed-deps + `SubagentBuilder` selection verbs (the carrier that lets `Toolset<D>` install onto a typed-deps registry).
- ADR-0090 — `Error::ModelRetry` + `complete_typed<O>` validation loop (the typed retry channel that `ToolHook` errors flow through).
- ADR-0034 — `RetryClassifier` + `RetryDecision` (the typed-decision surface `ToolHook` mirrors).
- ADR-0037 — `AuditSink` (invariant 18 — `record_agent_handoff` fires on both `Agent` and `Handoff`).
- `crates/entelix-agents/tests/recipes.rs` — 14 recipes tests, two of which pin Handoff payload routing.
- `crates/entelix-core/src/tools/{progress.rs,toolset.rs}` self-tests.
