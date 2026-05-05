# ADR 0075 — Streaming dispatch on the `tower::Service` spine + `tap_aggregator` cost emit

**Status**: Accepted
**Date**: 2026-05-05
**Decision**: `ChatModel::stream_deltas` flows through the same `tower::Service` layer factory as `ChatModel::complete_full`. The streaming path adds a parallel service shape `Service<StreamingModelInvocation, Response = ModelStream>`, where `ModelStream { stream, completion }` carries a delta-side stream taps-aggregated into a `completion` future that resolves to the terminal `ModelResponse`. `OtelLayer` and `PolicyLayer` implement both service shapes, so a single `.layer(...)` call wraps both spines; cost emission gates on the streaming-side `completion` future's `Ok` branch only — a stream that errors mid-flight, or that is dropped before its terminal `Stop`, resolves `completion` to `Err` and emits no charge.

## Context

Invariant 12 (CLAUDE.md §"Cost & Operations") demands cost be computed transactionally — `gen_ai.usage.cost`, `gen_ai.tool.cost`, `gen_ai.embedding.cost` are emitted only inside the `Ok` branch of the corresponding service / wrapper. A failed call must never produce a phantom charge.

Up to and including 1.0-RC.1 this guarantee held cleanly on the one-shot path:

- `ChatModel::complete_full` routes through `self.service().oneshot(invocation).await`, where `service()` returns the layer-wrapped `BoxedModelService`.
- `OtelLayer::Service<ModelInvocation, Response = ModelResponse>` runs `inner.oneshot(invocation).await`, then on `Ok` invokes the operator-supplied `CostCalculator` and emits `gen_ai.usage.cost` on the success-branch event.
- `PolicyLayer::Service<ModelInvocation, Response = ModelResponse>` charges the per-tenant `CostMeter` after the inner call returns `Ok`.

The streaming path bypassed this entirely. The previous `ChatModel::stream_deltas` body called `self.inner.codec().encode_streaming` → `transport.send_streaming` → `codec.decode_stream` directly, with a doc comment explicitly stating "this surface bypasses the layer stack — middleware that observes streaming should hook the underlying transport or wrap the returned stream directly." That punt:

- broke the invariant 12 guarantee for streaming dispatch — operators wiring `OtelLayer` / `PolicyLayer` saw cost only on the one-shot path.
- forced every layer to implement an alternative streaming hook (transport-level wrapping, returned-stream wrapping) that did not compose with `tower::Layer` semantics.
- left `RetryLayer` unable to wrap streaming dispatch even at the connection-establishment level (where retry is genuinely useful).
- created a per-call asymmetry where one-shot dispatch went through redaction / quota checks but streaming dispatch silently skipped them.

The 6-fork audit synthesis (`project_entelix_2026_05_05_master_plan.md` §"Phase G") confirmed the asymmetry as the largest remaining architectural debt — Phase B-E typed surfaces (RunBudget, ModelRetry, OutputValidator, AgentEvent variants) all expect to compose through the same `tower::Layer` stack on streaming dispatch, and the planned `gen_ai.usage.cost` / `gen_ai.tool.cost` symmetry between codecs and the agent-loop OTel root span is unreachable while streaming bypasses the spine.

## Decision

### `Service<Req>` is the single composition primitive

`tower::Service<Request>` carries `Response` as an associated type — one trait impl per `(Self, Request)` pair. Two `Response` types from one leaf service therefore require two distinct request types. We introduce:

```rust
pub struct StreamingModelInvocation {
    pub inner: ModelInvocation,
}
```

`InnerChatModel<C, T>` implements both `Service<ModelInvocation, Response = ModelResponse>` (the existing one-shot path, unchanged) and `Service<StreamingModelInvocation, Response = ModelStream>` (the new streaming spine). `OtelLayer` and `PolicyLayer` implement matching `Service<StreamingModelInvocation, Response = ModelStream>` impls beside their existing `Service<ModelInvocation, Response = ModelResponse>` impls — same struct, two service shapes.

`ChatModel::layer<L>` carries a dual constraint: `L: Layer<BoxedModelService> + Layer<BoxedStreamingService>`. The factory closures wrap both leaf service shapes simultaneously, so a single `.layer(OtelLayer::new(...))` call produces a `ChatModel` whose every dispatch — one-shot or streaming — runs through the same observability stack. Layers that are streaming-incompatible at the semantic level (e.g. `RetryLayer` retrying mid-stream) satisfy the constraint via tower's blanket `impl<S> Layer<S>` so the *type-system* requirement holds while their *runtime* behaviour falls through to a pass-through (the streaming side of `RetryService` simply wraps `inner.call`, never re-enters its retry loop because the future resolves to an `Ok(ModelStream)` in the connection-establishment phase, after which retry would be meaningless).

### `ModelStream { stream, completion }` is the streaming response

```rust
pub struct ModelStream {
    pub stream: BoxDeltaStream<'static>,
    pub completion: BoxFuture<'static, Result<ModelResponse>>,
}
```

The `stream` field carries the raw `StreamDelta` flow (text chunks, thinking deltas, tool-use boundaries, usage, rate-limit, warnings, terminal `Stop`). The `completion` future resolves to:

- `Ok(ModelResponse)` after the consumer drains the stream to its terminal `Stop` and the `StreamAggregator` has reconstructed the final response.
- `Err(...)` if the stream errored mid-flight, was dropped before terminal `Stop`, or violated the aggregator's protocol invariants.

The two carriers are wired by `entelix_core::stream::tap_aggregator(inner: BoxDeltaStream) -> ModelStream`. The helper wraps the inner stream in an `AggregatorTap` that:

- pushes each delta into a local `StreamAggregator` as it flows past the consumer.
- on terminal `Stop`, calls `aggregator.finalize()` and sends the result through a `oneshot::Sender`.
- on inner-stream `Err`, sends the propagated error through the same channel.
- on stream-`Drop` (consumer abandoned the stream before `Stop`), sends `Err(Error::Cancelled)` through the channel — so observability layers gating on `completion.await.is_ok()` do not fire on partial streams.

The `oneshot::Receiver` is wrapped in the `completion` future the layer can wrap further.

### Layer wrapping semantics

`OtelLayer`'s streaming `Service` impl wraps `completion`:

```rust
let user_facing = async move {
    let result = completion.await;
    match &result {
        Ok(response) => {
            let cost = cost_calculator
                .as_ref()
                .map(|c| c.compute_cost(&response.model, &response.usage, &ctx))
                .transpose()
                .await
                .flatten();
            tracing::event!(target: "gen_ai", ..., gen_ai.usage.cost = cost, "gen_ai.response");
        }
        Err(err) => {
            tracing::event!(target: "gen_ai", ..., error.message = %err, "gen_ai.error");
        }
    }
    result
};
Ok(ModelStream { stream, completion: Box::pin(user_facing) })
```

`PolicyLayer` wraps similarly — quota gate runs `pre-request` (before the dispatch even opens), cost meter charges fire on `completion.await.is_ok()` only.

Layers do **not** spawn background tasks. The contract is: callers that consume the stream and want the post-stream observability to fire MUST `await` the wrapped `completion`. This keeps the SDK runtime-agnostic (no implicit `tokio::spawn`) and keeps event ordering deterministic relative to caller drain. Consumers that drop the stream without awaiting `completion` lose the post-stream emission — mirroring what `complete_full` would do if its `oneshot(invocation)` future were dropped before resolving. The transactional guarantee still holds: dropping the future before it resolves emits nothing.

### Why not auto-spawn the completion task

A `tokio::spawn` on the wrapped completion future would emit observability events even when the caller drops the `ModelStream` early — at the cost of:

- coupling `entelix-otel` and `entelix-policy` to the tokio runtime (production deployments using async-std / smol for embedded contexts break).
- emission ordering becoming non-deterministic relative to the caller (the OTel span event might fire before or after the caller's own post-stream work).
- runtime panics in the spawned task being silent (a spawned task whose body panics does not propagate to the caller; the cost emission silently disappears).

The caller-driven design loses one corner case (dropped streams produce no observability event) but gains runtime portability + ordering determinism + caller-visible failure modes. The lost corner case is structurally identical to dropping a `complete_full` future — both surfaces share the same "emit on resolved future" semantic.

### Why `RetryLayer` exists on the streaming spine

`tower::Layer<S>` is a blanket impl on `RetryLayer`, so the dual constraint on `ChatModel::layer<L>` is type-satisfied. The runtime semantic: `RetryService<Service<StreamingModelInvocation>>` retries the *initial* call to `inner.call(invocation)` if that future resolves to `Err` — i.e., connection establishment, codec encode, transport pre-flight. If `inner.call` resolves `Ok(ModelStream)` (the stream is open), retry no longer applies; mid-stream errors propagate to the consumer through the stream itself and do not re-enter the retry loop. This is the right semantic — replaying a partially-streamed response would corrupt the conversation transcript, while retrying a connection-time failure is exactly what `RetryLayer` exists for.

### Why streaming PII redaction is deferred

`PolicyLayer::Service<StreamingModelInvocation>` redacts the *request* on the way in (operator-supplied input is fully formed before the first byte streams) but does not redact individual deltas on the way out. Streaming-aware PII redaction is non-trivial — chunk boundaries can split a PII pattern (a credit card number arriving as `"4111-1111"` + `"-1111-1111"` over two deltas evades a regex that scans each chunk individually), so a correct implementation requires a stateful sliding-window redactor with per-pattern lookback. That is post-1.0 work; the master plan §"Phase E — Retry + Validation + Observability" tracks it.

## Consequences

**Positive**:

- `gen_ai.usage.cost` emission is now symmetric across one-shot and streaming dispatch — operators wiring `OtelLayer` see cost on every model call regardless of dispatch shape.
- `PolicyLayer`'s quota gate fires on every model call regardless of dispatch shape — multi-tenant rate / budget enforcement is no longer streaming-blind.
- The same `tower::Layer<S>` composition primitive applies on every dispatch path: model one-shot, model streaming, tool dispatch. Adding a new layer is one struct + two `Service` impls; no streaming-specific hook surface exists.
- `ModelStream::completion` gives consumers a typed handle to the aggregated `ModelResponse` — agent recipes that compose streaming dispatch into a `StateGraph` node can `await` it on the post-stream branch instead of reimplementing aggregation.
- The `tap_aggregator` helper centralises the `StreamAggregator` wiring — every streaming-dispatch path runs through one well-tested aggregator, eliminating the per-codec aggregation drift risk.
- `RetryLayer` now wraps streaming dispatch at the connection-establishment level — codec / transport-level failures retry the same way they do on one-shot.

**Negative**:

- Two service shapes per layer impl (`Service<ModelInvocation>` and `Service<StreamingModelInvocation>`). Each `*Layer` carries roughly one extra `impl` block.
- Callers that consume the stream but ignore `completion` lose the post-stream observability emission. The contract is documented; the trade-off is intentional (runtime portability + ordering determinism over corner-case coverage).
- One additional public type (`ModelStream`), one new wrapper (`StreamingModelInvocation`), one new boxed alias (`BoxedStreamingService`), one new helper (`tap_aggregator`). Per ADR-0064 §"Public-API baseline contract" the typed-strengthening drift is authorised at the 1.0 RC contract boundary.

**Migration outcome (one-shot, no shim)**: `ChatModel::stream_deltas` 's previous return type (`BoxDeltaStream<'a>`) becomes `ModelStream` in the same commit. Consumers that previously consumed the boxed delta stream now consume `model_stream.stream` (one identifier substitution) and optionally `await model_stream.completion` for the aggregated final response. The doc comment that previously read "this surface bypasses the layer stack" is replaced; the layer stack now wraps streaming dispatch by default. There is no `#[deprecated]`, no `pub use OldName as NewName` — invariant 14 forbids the shim.

## References

- CLAUDE.md invariant 12 — cost computed transactionally (strengthened text references this ADR)
- CLAUDE.md invariant 14 — no backwards-compatibility shims
- ADR-0009 — OpenTelemetry GenAI semconv adoption
- ADR-0064 — 1.0 release charter (post-RC typed-strengthening authorised)
- ADR-0074 — `TenantId` newtype (parallel pattern: typed boundary at a deserialise edge)
- ADR-0076 — `LlmRenderable` + `RenderedForLlm` sealed carrier (parallel pattern: typed boundary at the model-facing channel)
- F4 mitigation — transactional cost emission
- `crates/entelix-core/src/service.rs` — `ModelStream` + `StreamingModelInvocation` + `BoxedStreamingService`
- `crates/entelix-core/src/stream.rs` — `tap_aggregator` helper + `AggregatorTap` wrapper
- `crates/entelix-core/src/chat.rs` — `ChatModel::streaming_service` + `Service<StreamingModelInvocation, Response = ModelStream>` impl on `InnerChatModel`
- `crates/entelix-otel/src/layer.rs` — `OtelLayer::Service<StreamingModelInvocation, Response = ModelStream>` cost-emit wrapping
- `crates/entelix-policy/src/layer.rs` — `PolicyLayer::Service<StreamingModelInvocation, Response = ModelStream>` quota / cost meter wrapping
- `crates/entelix-otel/tests/streaming_cost_emit.rs` — regression test (Ok-branch cost emit / mid-stream error 0 emit)
