# Changelog

All notable changes to **entelix** are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the workspace adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0 convention: the minor component carries breaking-change semantics).

Workspace releases ship in lockstep — every member crate carries the
same version. Annotated git tags (`vX.Y.Z`) point at the release
commit and mirror the headline summary of each entry below.

## [0.5.3] — 2026-05-12

### Changed
- `StateErasureSink<S>` 의 `PhantomData` marker — covariant +
  `Send`/`Sync` inheritance from `S`, matching the
  `impl<S: Send + Sync + 'static> AgentEventSink<S>` bound.
- `entelix-agents` crate description reflects the full surface
  (recipes + tool-side layer ecosystem + sink adapters + chat-shape
  state helpers).

### Added
- `AgentEvent::erase_state` carries an executable `# Examples` doc
  block; `StateErasureSink::new` doc references the multi-agent
  fan-in pattern.
- Wire-site regression `state_erasure_sink_pins_first_party_recipe_state_types`
  pins the adapter against the SDK's own state types
  (`ReActState` / `ChatState` / `SupervisorState`).

### Removed
- Over-scoped `#[allow(clippy::unwrap_used)]` on
  `entelix-agents/src/state.rs` test module.

## [0.5.2] — 2026-05-12

### Added
- Chat-shape state accessor — `ChatState` / `ReActState` /
  `SupervisorState::last_assistant_text(&self) -> Option<String>`
  concatenates every `ContentPart::Text` block of the most recent
  `Role::Assistant` message; the canonical SSE / chat-UI extraction
  point. `Reasoning` and `ToolUse` blocks are excluded by design.
- `AgentEvent<S>::erase_state(self) -> AgentEvent<()>` — generic
  parameter erasure, `Complete::state` replaced with `()` and every
  other variant rebuilt with identical field values.
- `StateErasureSink<S>` adapter wraps any `Arc<dyn AgentEventSink<()>>`
  and produces an `AgentEventSink<S>` for any `S`, fanning in
  heterogeneous agents to a single state-agnostic audit / SSE / OTel
  pipeline.
- `CostMeter` concurrency regression — `replace_pricing` (full table)
  and `replace_model_pricing` (single row) serialise correctly under
  cloned-meter contention.

### Documentation
- Sub-crate `CLAUDE.md` files refreshed against the 0.5.1 surface
  (`layer_named` / `replace_model_pricing` / `pricing_snapshot`) and
  the 0.5.2 additions.

## [0.5.1] — 2026-05-12

### Added
- `ChatModel::layer_named` + `ToolRegistry::layer_named` convenience
  — sugar over `WithName::new(name, layer)` for composing external
  `tower::Layer` middleware without implementing `NamedLayer`
  manually.
- `CostMeter::replace_model_pricing(model, pricing)` — single-row
  partial pricing swap, parallel to the existing
  `replace_pricing(table)` family. Inserts when the model is
  absent.
- `CostMeter::pricing_snapshot()` — owned `PricingTable` clone for
  admin diff / external-store reconciliation flows.

## [0.5.0] — 2026-05-12

### Added
- `entelix_core::ErrorEnvelope` — the canonical typed wire shape
  produced by `Error::envelope()`, bundling `wire_code`,
  `wire_class`, `retry_after_secs`, and `provider_status` into one
  `Copy` value. Derives `serde::Serialize` for SSE / audit
  forwarding.
- `AgentEvent<S>` variants carry `tenant_id: TenantId` (multi-tenant
  scope mandatory at every emit, per invariant 11). `Failed` and
  `ToolError` carry `envelope: ErrorEnvelope`.
- `ToolApprovalEventSink::record_approved` / `record_denied` take
  `tenant_id: &TenantId` as the first argument.
- `entelix_policy::UnknownModelSink` trait + `CostMeter::with_unknown_model_sink`
  — fires on every unknown-model dispatch attempt regardless of
  `UnknownModelPolicy::{Reject, WarnOnce}`.
- `PolicyRegistry::mutate_fallback` / `mutate_tenant` — closure-based
  atomic partial updates serialised by the slot's write lock.
- `entelix_core::service::NamedLayer` trait + `WithName<L>` wrapper.
  `ChatModel::layer_names()` and `ToolRegistry::layer_names()`
  expose the composed layer stack in insertion order. Eight
  first-party `NamedLayer` impls — `policy`, `otel`, `retry`
  (cross-spine); `tool_approval`, `tool_event`, `tool_hook`,
  `tool_retry`, `tool_scope` (tool-only).

### Changed
- `Error::wire_code()` / `Error::wire_class()` removed — read the
  envelope's fields via `Error::envelope()`.
- `TenantPolicyBuilder` removed — `TenantPolicy::new().with_*(...)`
  is the construction surface.
- `ChatModel::layer<L>` and `ToolRegistry::layer<L>` require
  `L: NamedLayer`; external `tower::Layer` middleware composes via
  `WithName::new(name, inner)`.

[0.5.3]: https://github.com/junyeong-ai/entelix/releases/tag/v0.5.3
[0.5.2]: https://github.com/junyeong-ai/entelix/releases/tag/v0.5.2
[0.5.1]: https://github.com/junyeong-ai/entelix/releases/tag/v0.5.1
[0.5.0]: https://github.com/junyeong-ai/entelix/releases/tag/v0.5.0
