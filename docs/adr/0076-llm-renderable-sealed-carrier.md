# ADR 0076 — `LlmRenderable` trait + `RenderedForLlm<T>` sealed carrier

**Status**: Accepted (supersedes the carrier portion of ADR-0033)
**Date**: 2026-05-05
**Decision**: Dynamic data crossing the agent → model boundary (errors, future sub-agent results, approval decisions, memory-recall summaries) flows through one funnel: `entelix_core::LlmRenderable<T>::for_llm(&self) -> RenderedForLlm<T>`. The carrier wraps a single `T` whose private constructor (`RenderedForLlm::new`) is `pub(crate)` to `entelix-core`, so external implementors of `LlmRenderable<T>` provide only the raw producer (`render_for_llm(&self) -> T`); the wrapping default impl cannot be overridden because no other crate can call `RenderedForLlm::new`. Emit sites that accept `RenderedForLlm<T>` therefore receive a value that structurally went through the trait — operator content cannot reach the model channel by accident or by typo.

## Context

Invariant 16 (CLAUDE.md §"Engineering — LLM / operator channel separation") demands that operator-facing diagnostics — `Display` text, source-error chains, vendor status codes, internal type identifiers, raw distance scores, ISO-8601 timestamps — never reach the model.

Up to and including 1.0-RC.1 the enforcement was a layered pair:

- `LlmFacingError::render_for_llm(&self) -> String` produced the model-safe rendering.
- `AgentEvent::ToolError` carried two fields — `error: String` (operator-facing) + `error_for_llm: String` (model-facing). The audit projection (`AgentEvent::to_graph_event`) routed the model-facing string into `GraphEvent::ToolResult`; the operator channels kept the full text.

The split was correct in spirit but brittle in mechanism. `error_for_llm: String` carries no type-level proof that the value passed through `render_for_llm`. A reviewer reading

```rust
AgentEvent::ToolError {
    error_for_llm: format!("vendor returned {status}: {body}"),
    ...
}
```

cannot distinguish "the author called `err.render_for_llm()`" from "the author concatenated operator-facing strings inline". The compiler will not catch the difference. CI test suites (`tests/llm_context_economy.rs`) catch *named* leak patterns by string-search but do not catch novel ones. As the SDK grows new emit sites — sub-agent result projection, approval-decision rationale, memory-recall summaries — the matrix of "fields that should structurally have come through the LLM-facing rendering" grows linearly with each, and `String` typing forces every reviewer of every emit site to verify by hand.

The 6-fork audit (2026-05-05, see `project_entelix_2026_05_05_master_plan.md`) confirmed that no major SDK ships a sealed carrier for model-facing content. pydantic-ai 1.90 carries error feedback as raw `str`. LangGraph 1.0 GA's middleware passes `ToolMessage.content: str` directly. OpenAI Agents SDK, Claude Agent SDK, rig, Vercel AI SDK 5 — all use untyped strings at this boundary. Type-level enforcement of the operator/model split at the SDK layer is industry-leading, mirroring the same gap `TenantId` closed for invariant 11 (ADR-0074).

## Decision

### `RenderedForLlm<T>` is a sealed newtype carrier

`entelix-core/src/llm_facing.rs`:

```rust
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RenderedForLlm<T>(T);

impl<T> RenderedForLlm<T> {
    pub(crate) const fn new(inner: T) -> Self { Self(inner) }
    pub const fn as_inner(&self) -> &T { &self.0 }
    pub fn into_inner(self) -> T { self.0 }
}
```

The constructor is `pub(crate)`. Other entelix crates (`entelix-agents`, `entelix-tools`, `entelix-session`, …) accept `RenderedForLlm<T>` as field type and call `as_inner` / `into_inner` to forward the inner value, but they cannot construct one. The seal holds across crate boundaries.

The carrier exposes:

- `as_inner` / `into_inner` — read accessors. The audit-log projection of `AgentEvent::ToolError` calls `as_inner().clone()` to emit the model-safe rendering as `GraphEvent::ToolResult` content; resume / replay paths read the persisted carrier inner value.
- `Display` — passes through to the inner type's `Display`. Sites that format the rendering for tool-message content read it through `format!("{}", rendered)` without unwrapping.
- `AsRef<str>` (when `T: AsRef<str>`) — same passthrough for `&str`-consuming APIs.
- `Serialize` / `Deserialize` — transparent (the inner value flows directly to the wire format). Audit-log replay paths re-load `AgentEvent::ToolError` events from a `SessionLog`; reconstructing the carrier around its persisted inner value is the inverse of the original emit, not a fresh fabrication. The `Deserialize` impl is the one operator-reachable construction path, but it requires a serialised value that originally came from a sealed carrier — it cannot be tricked into wrapping arbitrary `String`s because the operator-controlled side is the inner value, not the act of construction. (An adversarial operator who synthesises a JSON `{"error_for_llm": "leaked operator text"}` payload and replays it through `serde_json::from_str` is not bypassing the seal — the seal protects the *emit-time* boundary; replay-time corruption falls under the broader audit-log integrity guarantee, which is `SessionLog`'s responsibility.)

### `LlmRenderable<T>` is the funnel trait

```rust
pub trait LlmRenderable<T> {
    fn render_for_llm(&self) -> T;

    fn for_llm(&self) -> RenderedForLlm<T> {
        RenderedForLlm::new(self.render_for_llm())
    }
}
```

Implementors of `LlmRenderable<T>` define `render_for_llm` (the raw producer) and inherit `for_llm` (the carrier-producing wrapper). External crates *cannot* override `for_llm` because doing so would require calling `RenderedForLlm::new`, which is `pub(crate)` to `entelix-core`. The only path from a value to a carrier is therefore: `value.for_llm()` → trait default impl → `RenderedForLlm::new(value.render_for_llm())`.

This is the same sealing pattern Rust standard library uses for `RawVecInner` and `Box::leak` family — the constructor's privacy is the seal, not a separate `Sealed: private::Sealed` marker trait. The marker-trait pattern would seal the *trait* itself; this pattern seals the *return value* while leaving the trait operator-implementable. We want the latter — the SDK's own crates and operator-supplied error types both implement `LlmRenderable` for their own types, but only the SDK's `entelix-core` can build the carrier.

### `Error: LlmRenderable<String>`

The previous `LlmFacingError` trait is removed (invariant 14 — no shim). `Error: LlmRenderable<String>` carries the same mapping table:

| Variant | Rendering |
|---|---|
| `InvalidRequest(msg)` | `"invalid input: {msg}"` |
| `Provider { .. }` | `"upstream model error"` |
| `Auth(_)` | `"authentication failed"` |
| `Config(_)` | `"tool misconfigured"` |
| `Cancelled` | `"cancelled"` |
| `DeadlineExceeded` | `"timed out"` |
| `Interrupted { .. }` | `"awaiting human review"` |
| `Serde(_)` | `"output could not be serialised"` |

Call sites that previously wrote `err.render_for_llm()` continue to compile unchanged — `LlmRenderable::render_for_llm` has the same name and signature. Sites that need the carrier write `err.for_llm()`.

### `AgentEvent::ToolError::error_for_llm` is typed

```rust
pub enum AgentEvent<S> {
    ToolError {
        ...
        error_for_llm: RenderedForLlm<String>,
        ...
    },
    ...
}
```

`tool_event_layer.rs`'s emit site builds the field via `err.for_llm()`. The audit projection (`AgentEvent::to_graph_event`) extracts the inner string via `error_for_llm.as_inner().clone()` and emits it as `GraphEvent::ToolResult { content: ToolResultContent::Text(...) }` — replay reconstructs the model's view from the same source.

The struct test fixture in `event.rs` cannot fabricate the carrier directly; it builds the field through `Error::provider_http(503, "vendor down").for_llm()`, exercising the actual production path rather than stubbing past it. This makes the test a structural regression on the boundary, not a string-search.

### Why one trait per `T`, not one trait with associated type

`LlmRenderable<T>` is generic over `T` so the same trait name covers both `LlmRenderable<String>` (errors, summaries) and future `LlmRenderable<serde_json::Value>` (sub-agent JSON results) impls. Associated-type variants would force one impl per type per implementor; generic-parameter trait impls let one type implement `LlmRenderable<String>` for "human-readable summary" *and* `LlmRenderable<Value>` for "JSON projection" simultaneously. The carrier adapts its display ergonomics per `T` (`Display` when `T: Display`, `AsRef<str>` when `T: AsRef<str>`).

## Consequences

**Positive**:

- A reviewer reading any emit site that takes `RenderedForLlm<T>` knows by *type* that the value came through `LlmRenderable::for_llm`. No string-search test needed for the construction boundary.
- New emit sites (sub-agent result projection, approval-decision rationale, memory-recall summaries) inherit the same boundary by typing their fields `RenderedForLlm<T>`. Adding a new model-facing channel becomes one type substitution, not a new test-suite chapter.
- `AgentEvent::ToolError::error_for_llm: RenderedForLlm<String>` is forward-compatible with the planned `Error::ModelRetry { feedback: RenderedForLlm<String>, attempt: u32 }` variant (slice E-1, master plan Phase E) — the `feedback` field will share the same carrier, so retry feedback flows through the same funnel as tool error rendering.
- The carrier serialises transparently: persisted `SessionLog` events round-trip through `serde_json::to_value(&AgentEvent::ToolError { error_for_llm, … })` and back, without operator code seeing the carrier at all. Replay reconstructs the typed shape automatically.

**Negative**:

- Test fixtures that previously wrote `error_for_llm: "upstream model error".into()` now write `error_for_llm: Error::provider_http(503, "vendor down").for_llm()`. The verbosity is intentional — the test exercises the boundary instead of stubbing past it. One-line refactor per fixture, regression-locked by the failing build that the typed field forces.
- Operator-supplied `Tool` impls that produce model-facing summaries (e.g. a tool that returns a typed `RetrievalSummary`) must `impl LlmRenderable<String> for RetrievalSummary` to thread the value through. The cost is one trait impl per such type — well-amortised against the structural guarantee.
- One additional public type (`RenderedForLlm<T>`) and one renamed trait (`LlmFacingError` → `LlmRenderable<T>`) in `entelix-core`'s public surface. Per ADR-0064 §"Public-API baseline contract", typed-strengthening drift at the 1.0 RC contract boundary is authorised.

**Migration outcome (one-shot, no shim)**: `LlmFacingError` trait is deleted, not deprecated. All call sites — `entelix-agents/src/agent/event.rs`, `entelix-agents/src/agent/tool_event_layer.rs`, `entelix-agents/src/react_agent.rs`, `entelix-tools/tests/llm_context_economy.rs`, `entelix-core/src/error.rs` doc references, the facade re-export — switch to `LlmRenderable` in the same commit. `error_for_llm: String` becomes `error_for_llm: RenderedForLlm<String>`; the emit site swaps `err.render_for_llm()` for `err.for_llm()`; the audit projection swaps `error_for_llm.clone()` for `error_for_llm.as_inner().clone()`. Invariant 14 forbids deprecation, and there is no `pub use OldName as NewName` tombstone.

## References

- CLAUDE.md invariant 16 — LLM / operator channel separation (strengthened text references this ADR)
- CLAUDE.md invariant 14 — no backwards-compatibility shims
- ADR-0033 — original LLM / operator channel separation; this ADR supersedes its carrier portion (the `LlmFacingError` trait surface)
- ADR-0064 — 1.0 release charter (post-RC typed-strengthening authorised)
- ADR-0074 — `TenantId` newtype (parallel pattern: validating type as a structural boundary at a deserialise edge)
- `crates/entelix-core/src/llm_facing.rs` — `LlmRenderable<T>` trait + `RenderedForLlm<T>` carrier + `Error: LlmRenderable<String>` impl
- `crates/entelix-agents/src/agent/event.rs` — `AgentEvent::ToolError::error_for_llm: RenderedForLlm<String>`
- `crates/entelix-agents/src/agent/tool_event_layer.rs` — emit site (`err.for_llm()`)
- `crates/entelix-tools/tests/llm_context_economy.rs` — regression suite (extends with sealing test)
