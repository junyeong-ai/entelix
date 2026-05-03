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

use serde_json::Value;
use tower::Service;
use tower::util::BoxCloneService;

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
