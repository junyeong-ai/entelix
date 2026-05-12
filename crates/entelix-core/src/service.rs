//! `tower::Service` spine for entelix model + tool calls.
//!
//! Two invocation types — [`ModelInvocation`] (request + ctx for a
//! `Codec`/`Transport` model call) and [`ToolInvocation`] (name +
//! input + ctx for a `Tool` execution) — flow through a layered
//! `tower::Service<Request, Response = ...>` stack. Cross-cutting
//! concerns (PII redaction, rate limiting, cost metering, OTel
//! observability) are `tower::Layer<S>` middleware. The composition
//! primitive is `tower::ServiceBuilder`; the dyn-erased handle is
//! [`BoxedModelService`] / [`BoxedToolService`].
//!
//! ## Why `tower::Service` rather than a bespoke `Hook` trait
//!
//! `tower` is the de-facto Rust async-middleware ecosystem
//! (`axum`, `tonic`, `reqwest-middleware`, `tower-http`). Adopting
//! it means:
//!
//! - `ServiceBuilder::new().layer(PolicyLayer).layer(OtelLayer)
//!   .service(model)` is the composition contract — `model` is a
//!   [`ChatModel`](crate::ChatModel)-produced leaf service.
//! - `Service::poll_ready` gives backpressure for free; layers like
//!   `tower::limit::ConcurrencyLimitLayer`,
//!   `tower::retry::RetryLayer`, and `tower::timeout::TimeoutLayer`
//!   plug in unchanged.
//! - The same layer (e.g. `PolicyLayer`) wraps both model calls and
//!   tool calls because it has separate `Service<ModelInvocation>`
//!   and `Service<ToolInvocation>` impls behind the same struct.

use std::sync::Arc;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use serde_json::Value;
use tower::Service;
use tower::util::BoxCloneService;

use crate::codecs::BoxDeltaStream;
use crate::context::ExecutionContext;
use crate::error::Error;
use crate::ir::{ModelRequest, ModelResponse};
use crate::tools::ToolMetadata;

/// One model call's full request + request-scope context. Layers
/// read both fields; the leaf service consumes them.
#[derive(Clone, Debug)]
pub struct ModelInvocation {
    /// Provider-neutral model request the codec will encode.
    pub request: ModelRequest,
    /// Request-scope state (cancellation, deadline, tenant, thread).
    pub ctx: ExecutionContext,
}

impl ModelInvocation {
    /// Bundle `request` + `ctx` into one invocation.
    pub const fn new(request: ModelRequest, ctx: ExecutionContext) -> Self {
        Self { request, ctx }
    }
}

/// Streaming-side counterpart to [`ModelInvocation`] — the same
/// `request + ctx` payload but a distinct request type so the
/// `tower::Service` trait's associated `Response` can resolve to
/// [`ModelStream`] for the streaming spine while [`ModelInvocation`]
/// keeps resolving to [`ModelResponse`] for the one-shot spine.
///
/// Rust's `Service<Request>` carries `Response` as an associated
/// type — one trait impl per `(Self, Request)` pair. The wrapper
/// here is the cleanest way to expose two response types from one
/// leaf service: the same `InnerChatModel<C, T>` implements
/// `Service<ModelInvocation, Response = ModelResponse>` and
/// `Service<StreamingModelInvocation, Response = ModelStream>`,
/// and layers stack onto each independently.
///
/// `#[non_exhaustive]` to keep room for streaming-only knobs
/// (chunk size hints, partial-output buffers) post-1.0 without
/// breaking callers.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StreamingModelInvocation {
    /// The wrapped one-shot invocation. Streaming layers read
    /// `request` and `ctx` through this — same fields, same
    /// semantics, different `Service` shape.
    pub inner: ModelInvocation,
}

impl StreamingModelInvocation {
    /// Wrap a [`ModelInvocation`].
    #[must_use]
    pub const fn new(inner: ModelInvocation) -> Self {
        Self { inner }
    }

    /// Borrow the request — read-side shortcut so layers don't
    /// have to write `invocation.inner.request`.
    #[must_use]
    pub const fn request(&self) -> &ModelRequest {
        &self.inner.request
    }

    /// Borrow the context — read-side shortcut.
    #[must_use]
    pub const fn ctx(&self) -> &ExecutionContext {
        &self.inner.ctx
    }
}

impl From<ModelInvocation> for StreamingModelInvocation {
    fn from(inner: ModelInvocation) -> Self {
        Self::new(inner)
    }
}

/// One tool call's identifier + descriptor + input + request-scope
/// context.
///
/// `tool_use_id` carries the IR's stable id so observability layers
/// can correlate `ToolStart` / `ToolComplete` / `ToolError` events
/// for the *same* dispatch even when several parallel calls share
/// the same tool name. `metadata` is the dispatched tool's full
/// declarative descriptor — name / version / effect / idempotent /
/// retry hint flow through the layer stack from a single source so
/// `OtelLayer`, `PolicyLayer`, and retry middleware see one
/// authoritative struct. Layers may mutate `input` (e.g. PII
/// redaction).
#[derive(Clone, Debug)]
pub struct ToolInvocation {
    /// Stable tool-use id matching the originating
    /// `ContentPart::ToolUse::id`. Empty when the call did not
    /// originate from a model `ToolUse` block (e.g. recipe-driven
    /// direct dispatch); observability layers fall back to
    /// `metadata.name` in that case.
    pub tool_use_id: String,
    /// Full declarative descriptor of the tool being dispatched —
    /// shared via `Arc` so layers don't pay a clone per pass.
    pub metadata: Arc<ToolMetadata>,
    /// JSON input payload.
    pub input: Value,
    /// Request-scope state.
    pub ctx: ExecutionContext,
}

impl ToolInvocation {
    /// Bundle the fields.
    pub const fn new(
        tool_use_id: String,
        metadata: Arc<ToolMetadata>,
        input: Value,
        ctx: ExecutionContext,
    ) -> Self {
        Self {
            tool_use_id,
            metadata,
            input,
            ctx,
        }
    }

    /// Tool name (shortcut for `self.metadata.name.as_str()`).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Tool version (shortcut for `self.metadata.version.as_deref()`).
    #[must_use]
    pub fn version(&self) -> Option<&str> {
        self.metadata.version.as_deref()
    }
}

/// Type-erased, cloneable `Service<ModelInvocation>` handle. The
/// canonical pre-composed shape `ChatModel` exposes via its
/// `service()` accessor and that user code stores on agents.
pub type BoxedModelService = BoxCloneService<ModelInvocation, ModelResponse, Error>;

/// Type-erased, cloneable `Service<ToolInvocation>` handle. Tool
/// dispatch funnels through this; `ToolRegistry` builds it on
/// demand from a registered `Tool` + the registry's layer stack.
pub type BoxedToolService = BoxCloneService<ToolInvocation, Value, Error>;

/// Streaming dispatch result returned by
/// `Service<ModelInvocation, Response = ModelStream>` — the
/// caller-visible delta stream paired with a future that resolves
/// to the aggregated terminal response.
///
/// The [`Self::stream`] field carries the raw `StreamDelta` flow
/// (text chunks, tool-use boundaries, usage, rate-limit, warnings,
/// terminal `Stop`). The [`Self::completion`] future resolves to
/// `Ok(ModelResponse)` after the stream has been fully consumed
/// AND a `StreamAggregator` has reconstructed the final response;
/// it resolves to `Err(...)` if the stream errored mid-flight, was
/// dropped before terminal `Stop`, or violated the aggregator's
/// protocol invariants.
///
/// Layers (`OtelLayer`, `PolicyLayer`) wrap `completion` to emit
/// observability / cost events on the **`Ok` branch only** —
/// invariant 12. A stream that errors mid-flight surfaces the
/// error through the consumer's stream-side `Err` *and* through
/// `completion` resolving to `Err`; either way, no cost charge
/// fires.
///
/// `completion` is internally driven by the same stream
/// `stream` carries — consumers do not need to poll it
/// separately. The aggregator runs as the consumer drains the
/// stream; `completion` resolves naturally when the consumer
/// reads the terminal `Stop` (or drops the stream early, in which
/// case `completion` resolves `Err`).
pub struct ModelStream {
    /// Raw delta stream surfaced to the caller. The wrapper
    /// produced by `entelix_core::stream::tap_aggregator` taps
    /// each delta into a `StreamAggregator` as it flows past, so
    /// the caller sees an unmodified stream while
    /// [`Self::completion`] receives the aggregated final response
    /// without a second pass.
    pub stream: BoxDeltaStream<'static>,
    /// Future resolving to the aggregated `ModelResponse` after
    /// the stream has been consumed to its terminal `Stop`. Layers
    /// wrap this future to gate observability emission on success
    /// (invariant 12). Consumers that ignore the streaming-side
    /// completion (e.g. wire it into a fire-and-forget OTel layer)
    /// do not need to await it directly — dropping the
    /// `ModelStream` is the canonical "I'm done" signal that lets
    /// any wrapping layer observe stream-completion regardless of
    /// whether the consumer polled `completion` itself.
    pub completion: BoxFuture<'static, Result<ModelResponse, Error>>,
}

impl std::fmt::Debug for ModelStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelStream")
            .field("stream", &"<BoxDeltaStream>")
            .field("completion", &"<BoxFuture<Result<ModelResponse>>>")
            .finish()
    }
}

/// Type-erased, cloneable `Service<StreamingModelInvocation,
/// Response = ModelStream>` handle. Parallel to
/// [`BoxedModelService`] for the streaming dispatch path.
/// `ChatModel::streaming_service()` produces this; `OtelLayer` /
/// `PolicyLayer` wrap it the same way they wrap
/// [`BoxedModelService`] for the one-shot path.
pub type BoxedStreamingService = BoxCloneService<StreamingModelInvocation, ModelStream, Error>;

/// Convenience: an always-ready `Service` whose `poll_ready` returns
/// `Poll::Ready(Ok(()))` unconditionally. Most leaf services have
/// no internal queue and use this shape; layers inherit
/// `poll_ready` from their inner service.
pub trait AlwaysReady<Request>: Service<Request> {
    /// `poll_ready` impl that's always ready. Call from a leaf
    /// service's `poll_ready` body.
    #[inline]
    fn poll_ready_always(_cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

impl<S, Request> AlwaysReady<Request> for S where S: Service<Request> {}

/// Static identity for a [`tower::Layer`] participant in
/// [`crate::ChatModel::layer`] / `ToolRegistry::layer` composition.
///
/// [`crate::ChatModel::layer_names`] walks the composed stack and
/// surfaces the names through the typed introspection channel —
/// diagnostic dashboards read `entelix.chat_model.layers` without
/// parsing prose `Debug` output, conditional-wiring code asserts the
/// expected stack at boot ("did my staging `OtelLayer` actually
/// compose in?"), and audit trails distinguish runs whose policy
/// wiring drifted at deploy time.
///
/// **Conventions** (patch-version-stable — renaming is a breaking
/// change for dashboards keyed off the value):
///
/// - `snake_case` ASCII bucketing the layer's primary role
///   (`"policy"`, `"otel"`, `"retry"`, `"approval"`).
/// - One canonical name per concrete layer struct, surfaced as
///   `pub const NAME: &'static str` on the struct so renaming
///   happens in one place.
/// - The trait method returns `&'static str` because layer
///   composition is a startup-time event and the name is part of
///   the binary's identity — runtime-built names would defeat the
///   stable-key promise.
///
/// External `tower::Layer` middleware (e.g. `tower::limit`'s
/// `ConcurrencyLimitLayer`) wraps through [`WithName`] to
/// participate in the same channel.
pub trait NamedLayer {
    /// Stable, patch-version-stable identifier surfaced through
    /// [`crate::ChatModel::layer_names`]. See trait doc for the
    /// naming convention.
    fn layer_name(&self) -> &'static str;
}

/// Wraps any [`tower::Layer<S>`] with a static name so external
/// middleware participates in the [`crate::ChatModel::layer_names`]
/// introspection channel. The wrapper is transparent at the
/// `tower::Layer` boundary — `WithName::new("concurrency",
/// ConcurrencyLimitLayer::new(10)).layer(inner)` produces the same
/// service the underlying layer would.
///
/// First-party entelix layers (`PolicyLayer`, `OtelLayer`)
/// implement [`NamedLayer`] directly and do **not** need this
/// wrapper; it exists exclusively for external `tower` middleware.
#[derive(Clone, Copy, Debug)]
pub struct WithName<L> {
    name: &'static str,
    inner: L,
}

impl<L> WithName<L> {
    /// Stamp `inner` with the diagnostic name `name`. See
    /// [`NamedLayer`]'s trait doc for the snake_case convention.
    pub const fn new(name: &'static str, inner: L) -> Self {
        Self { name, inner }
    }

    /// Borrow the underlying layer.
    #[must_use]
    pub const fn inner(&self) -> &L {
        &self.inner
    }
}

impl<L> NamedLayer for WithName<L> {
    fn layer_name(&self) -> &'static str {
        self.name
    }
}

impl<L, S> tower::Layer<S> for WithName<L>
where
    L: tower::Layer<S>,
{
    type Service = L::Service;

    fn layer(&self, inner: S) -> Self::Service {
        self.inner.layer(inner)
    }
}
