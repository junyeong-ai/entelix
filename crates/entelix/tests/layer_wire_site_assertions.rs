//! Wire-site compile assertion — every first-party layer satisfies
//! the trait bound on the spine it advertises. A future bound change
//! on `ChatModel::layer` or `ToolRegistry::layer` that any layer fails
//! to track surfaces as a compile error in this file rather than as
//! a runtime "I tried to wire X and the bound shifted" surprise.
//!
//! Two assertion sites:
//! - **model spine** — layers that compose onto `ChatModel::layer`,
//!   i.e. those carrying both a `Layer<BoxedModelService>` and a
//!   `Layer<BoxedStreamingService>` impl. Currently `PolicyLayer`,
//!   `OtelLayer` (feature-gated), and `RetryLayer`.
//! - **tool spine** — layers that compose onto `ToolRegistry::layer`.
//!   The model-spine layers plus the five tool-only layers
//!   (`RetryToolLayer`, `ApprovalLayer`, `ToolEventLayer<S>`,
//!   `ToolHookLayer`, `ScopedToolLayer`).
//!
//! The bound-check functions are `fn _: L)` shells whose `where`
//! clauses mirror the spine's `.layer<L>` constraints byte-for-byte.
//! Calling them with a concrete layer instance forces the compiler
//! to verify every projection bound at the wire site.

#![allow(clippy::unused_async, dead_code)]

use std::sync::Arc;

use entelix::Error;
use entelix::ir::{ModelResponse, Usage};
use entelix::service::{
    BoxedModelService, BoxedStreamingService, BoxedToolService, ModelInvocation, ModelStream,
    StreamingModelInvocation, ToolInvocation,
};
use entelix::tools::{ScopedToolLayer, ToolDispatchScope};
use entelix::{
    AlwaysApprove, ApprovalLayer, DroppingSink, NamedLayer, ToolEventLayer, ToolHookLayer,
    ToolHookRegistry,
};
use entelix_core::ExecutionContext;
use entelix_core::transports::{RetryLayer, RetryPolicy};
use entelix_policy::{PolicyLayer, PolicyRegistry};
use futures::future::BoxFuture;
use serde_json::Value;
use tower::{Layer, Service};

// ── Model-spine bound mirror — kept in lock-step with
//    `entelix_core::chat::ChatModel::layer<L>`. ────────────────────────

fn assert_chat_model_layer<L>(_: L)
where
    L: Layer<BoxedModelService>
        + Layer<BoxedStreamingService>
        + NamedLayer
        + Clone
        + Send
        + Sync
        + 'static,
    <L as Layer<BoxedModelService>>::Service:
        Service<ModelInvocation, Response = ModelResponse, Error = Error> + Clone + Send + 'static,
    <<L as Layer<BoxedModelService>>::Service as Service<ModelInvocation>>::Future: Send + 'static,
    <L as Layer<BoxedStreamingService>>::Service: Service<StreamingModelInvocation, Response = ModelStream, Error = Error>
        + Clone
        + Send
        + 'static,
    <<L as Layer<BoxedStreamingService>>::Service as Service<StreamingModelInvocation>>::Future:
        Send + 'static,
{
}

// ── Tool-spine bound mirror — kept in lock-step with
//    `entelix_core::tools::ToolRegistry::layer<L>`. ────────────────────

fn assert_tool_registry_layer<L>(_: L)
where
    L: Layer<BoxedToolService> + NamedLayer + Clone + Send + Sync + 'static,
    L::Service: Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
    <L::Service as Service<ToolInvocation>>::Future: Send + 'static,
{
}

// ── No-op ToolDispatchScope for the ScopedToolLayer fixture. ────────

struct NoOpScope;

impl ToolDispatchScope for NoOpScope {
    fn wrap(
        &self,
        _ctx: ExecutionContext,
        fut: BoxFuture<'static, entelix::Result<Value>>,
    ) -> BoxFuture<'static, entelix::Result<Value>> {
        fut
    }
}

// ── Model-spine assertions. ─────────────────────────────────────────

#[test]
fn first_party_model_spine_layers_compose_via_chat_model_layer() {
    let registry = Arc::new(PolicyRegistry::new());
    assert_chat_model_layer(PolicyLayer::new(registry));
    assert_chat_model_layer(RetryLayer::new(RetryPolicy::standard()));
    #[cfg(feature = "otel")]
    assert_chat_model_layer(entelix::OtelLayer::new("test"));
}

// ── Tool-spine assertions. ──────────────────────────────────────────

#[test]
fn first_party_tool_spine_layers_compose_via_tool_registry_layer() {
    // Cross-spine layers also fit the tool spine.
    let registry = Arc::new(PolicyRegistry::new());
    assert_tool_registry_layer(PolicyLayer::new(Arc::clone(&registry)));
    assert_tool_registry_layer(RetryLayer::new(RetryPolicy::standard()));
    #[cfg(feature = "otel")]
    assert_tool_registry_layer(entelix::OtelLayer::new("test"));

    // Tool-only layers — every one must satisfy the bound or the
    // wire path breaks for operators wiring it.
    assert_tool_registry_layer(entelix_core::tools::RetryToolLayer::new());
    assert_tool_registry_layer(ApprovalLayer::new(Arc::new(AlwaysApprove)));
    assert_tool_registry_layer(ToolHookLayer::new(ToolHookRegistry::default()));
    assert_tool_registry_layer(ScopedToolLayer::new(NoOpScope));

    // ToolEventLayer is generic over the agent state `S`; the
    // assertion is parameterised through `S = i32` (any concrete `S`
    // works — picking one nails the bound for every monomorphisation
    // the operator wires).
    let sink: Arc<dyn entelix::AgentEventSink<i32>> = Arc::new(DroppingSink);
    assert_tool_registry_layer(ToolEventLayer::<i32>::new(sink));

    // Smoke-check Usage construction stays usable here so this file
    // breaks if the IR shape drifts independently of the bound.
    let _usage = Usage::new(0, 0);
}

// ── NAME constant audit — pins the canonical role-bucket map. ──────

#[test]
fn first_party_layer_names_are_canonical_role_buckets() {
    // Cross-spine (model + tool): role noun, no prefix.
    assert_eq!(PolicyLayer::NAME, "policy");
    assert_eq!(RetryLayer::NAME, "retry");
    #[cfg(feature = "otel")]
    assert_eq!(entelix::OtelLayer::NAME, "otel");

    // Tool-only: `tool_<role>` prefix for every entry.
    assert_eq!(entelix_core::tools::RetryToolLayer::NAME, "tool_retry");
    assert_eq!(ApprovalLayer::NAME, "tool_approval");
    assert_eq!(ToolEventLayer::<i32>::NAME, "tool_event");
    assert_eq!(ToolHookLayer::NAME, "tool_hook");
    assert_eq!(ScopedToolLayer::NAME, "tool_scope");
}
