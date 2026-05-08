//! `ApprovalLayer` — `tower::Layer<S>` that gates every
//! `Service<ToolInvocation>` dispatch through an [`Approver::decide`]
//! call. On approval the inner service runs; on rejection the layer
//! short-circuits with [`Error::InvalidRequest`] carrying the
//! approver's reason. When a [`ToolApprovalEventSinkHandle`] is
//! attached to the request's [`ExecutionContext`] (via the agent's
//! standard wiring), the layer also emits
//! [`crate::agent::AgentEvent::ToolCallApproved`] /
//! [`crate::agent::AgentEvent::ToolCallDenied`] for observability.
//!
//! ## Wiring
//!
//! Operators rarely attach the layer manually — `ReActAgentBuilder`
//! auto-wires it when an `Approver` is configured. The wiring routes:
//!
//! 1. `ReActAgentBuilder::with_approver(approver)` →
//! 2. `build()` calls `tools.layer(ApprovalLayer::new(approver))` →
//! 3. Every `tools.dispatch(...)` inside the agent's graph passes
//!    through the layer →
//! 4. `Approver::decide` runs before the inner tool service →
//! 5. `Agent::execute_inner` attaches a `ToolApprovalEventSinkHandle`
//!    to the request `ExecutionContext` so the layer can emit through
//!    the agent's typed `AgentEventSink<S>` without taking it as a
//!    constructor argument (which would tie the layer to a specific
//!    `S`).
//!
//! ## Type erasure across the sink
//!
//! `AgentEventSink<S>` is generic over the agent's state type;
//! `ApprovalLayer` lives below the agent (one layer instance, many
//! `S` shapes if the same registry feeds heterogeneous agents).
//! [`ToolApprovalEventSink`] is the type-erased trait the layer
//! actually consumes — [`ToolApprovalEventSinkHandle::for_agent_sink`]
//! is the bridge that adapts any `Arc<dyn AgentEventSink<S>>` into
//! the type-erased shape.
//!
//! ## `AwaitExternal` pause-and-resume (ADR-0071)
//!
//! When `Approver::decide` returns `ApprovalDecision::AwaitExternal`,
//! the layer raises `Error::Interrupted { payload }` with a
//! structured payload (`kind = "approval_pending"`, plus the
//! pending dispatch's `run_id` / `tool_use_id` / `tool` / `input`).
//! The graph dispatch loop catches it, persists a checkpoint with
//! pre-node state, and surfaces the typed error to the caller —
//! the agent run pauses cleanly with no inflight resources.
//!
//! Resume drops the operator's eventual decision into
//! the typed `Command::ApproveTool { tool_use_id, decision }`
//! the overrides to `ExecutionContext` before re-entering the same
//! dispatch. The layer's override-lookup runs first and short-
//! circuits the approver — the resumed run completes the pending
//! tool call without re-asking.
//!
//! ## What the layer does NOT cover
//!
//! - **Per-tool approver bypasses.** Operators that want to skip
//!   approval for a subset of tools wire a custom `Approver` impl
//!   that returns `Approve` for those names; the layer itself stays
//!   unconditional.

use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use tower::{Layer, Service};

use entelix_core::error::{Error, Result};
use entelix_core::service::ToolInvocation;
use entelix_core::{INTERRUPT_KIND_APPROVAL_PENDING, PendingApprovalDecisions};

use crate::agent::approver::{ApprovalDecision, ApprovalRequest, Approver};
use crate::agent::event::AgentEvent;
use crate::agent::sink::AgentEventSink;

/// Type-erased sink for tool-approval events. The agent runtime
/// produces an implementation by adapting its
/// `Arc<dyn AgentEventSink<S>>` (see
/// [`ToolApprovalEventSinkHandle::for_agent_sink`]); operators
/// implementing custom downstream observability (OTel direct,
/// audit-log direct) can implement this trait directly without
/// going through `AgentEventSink<S>`.
#[async_trait]
pub trait ToolApprovalEventSink: Send + Sync + 'static {
    /// Record an approval decision. The layer awaits the call so
    /// the approval marker fires *before* the inner tool service
    /// begins; observability ordering matches the operator's mental
    /// model (approve → start → complete).
    async fn record_approved(&self, run_id: &str, tool_use_id: &str, tool: &str);

    /// Record a denial decision. The layer awaits and then returns
    /// `Error::InvalidRequest` to the caller; the matching
    /// `ToolStart` does NOT fire.
    async fn record_denied(&self, run_id: &str, tool_use_id: &str, tool: &str, reason: &str);
}

/// Refcounted handle for [`ToolApprovalEventSink`]. Stored in
/// [`entelix_core::ExecutionContext`] extensions so [`ApprovalLayer`] finds the
/// sink without taking it as a constructor argument.
///
/// `Clone` is cheap (the inner sink rides behind `Arc`).
#[derive(Clone)]
pub struct ToolApprovalEventSinkHandle {
    sink: Arc<dyn ToolApprovalEventSink>,
}

impl ToolApprovalEventSinkHandle {
    /// Wrap any [`ToolApprovalEventSink`] impl. Convenient for
    /// custom direct-observability sinks that don't bridge through
    /// `AgentEventSink<S>`.
    pub fn new<E>(sink: E) -> Self
    where
        E: ToolApprovalEventSink,
    {
        Self {
            sink: Arc::new(sink),
        }
    }

    /// Adapt an agent's typed [`AgentEventSink<S>`] into the type-
    /// erased shape the layer consumes. The adapter forwards
    /// `record_approved` → [`AgentEvent::ToolCallApproved`] and
    /// `record_denied` → [`AgentEvent::ToolCallDenied`] on the
    /// underlying sink.
    pub fn for_agent_sink<S>(sink: Arc<dyn AgentEventSink<S>>) -> Self
    where
        S: Clone + Send + Sync + 'static,
    {
        Self {
            sink: Arc::new(SinkAdapter { sink }),
        }
    }

    /// Borrow the underlying erased sink. Primarily for the layer's
    /// own dispatch path; operators consume the methods through the
    /// trait object the layer reads from `ExecutionContext`.
    pub fn inner(&self) -> &Arc<dyn ToolApprovalEventSink> {
        &self.sink
    }
}

struct SinkAdapter<S> {
    sink: Arc<dyn AgentEventSink<S>>,
}

#[async_trait]
impl<S> ToolApprovalEventSink for SinkAdapter<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn record_approved(&self, run_id: &str, tool_use_id: &str, tool: &str) {
        let event: AgentEvent<S> = AgentEvent::ToolCallApproved {
            run_id: run_id.to_owned(),
            tool_use_id: tool_use_id.to_owned(),
            tool: tool.to_owned(),
        };
        let _ = self.sink.send(event).await;
    }

    async fn record_denied(&self, run_id: &str, tool_use_id: &str, tool: &str, reason: &str) {
        let event: AgentEvent<S> = AgentEvent::ToolCallDenied {
            run_id: run_id.to_owned(),
            tool_use_id: tool_use_id.to_owned(),
            tool: tool.to_owned(),
            reason: reason.to_owned(),
        };
        let _ = self.sink.send(event).await;
    }
}

/// `tower::Layer<S>` that gates a `Service<ToolInvocation, Response = Value, Error = Error>`
/// through an [`Approver`]. Construct via [`ApprovalLayer::new`];
/// attach to a `ToolRegistry` via
/// [`entelix_core::tools::ToolRegistry::layer`].
pub struct ApprovalLayer {
    approver: Arc<dyn Approver>,
}

impl ApprovalLayer {
    /// Wrap an `Arc<dyn Approver>` for layer attachment. Cloning
    /// the layer bumps the inner refcount.
    pub const fn new(approver: Arc<dyn Approver>) -> Self {
        Self { approver }
    }
}

impl Clone for ApprovalLayer {
    fn clone(&self) -> Self {
        Self {
            approver: Arc::clone(&self.approver),
        }
    }
}

impl<S> Layer<S> for ApprovalLayer {
    type Service = ApprovalService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ApprovalService {
            inner,
            approver: Arc::clone(&self.approver),
        }
    }
}

/// `tower::Service<ToolInvocation>` produced by [`ApprovalLayer`].
/// Public so operators that wire dispatch paths manually can
/// compose it directly.
pub struct ApprovalService<S> {
    inner: S,
    approver: Arc<dyn Approver>,
}

impl<S: Clone> Clone for ApprovalService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            approver: Arc::clone(&self.approver),
        }
    }
}

impl<S> Service<ToolInvocation> for ApprovalService<S>
where
    S: Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, invocation: ToolInvocation) -> Self::Future {
        let approver = Arc::clone(&self.approver);
        let mut inner = self.inner.clone();
        Box::pin(async move {
            // Override lookup runs first — when the resume path
            // (`Command::ApproveTool`) has attached
            // `PendingApprovalDecisions` carrying a decision for
            // this `tool_use_id`, skip the approver entirely. This
            // is how the AwaitExternal pause-and-resume flow
            // short-circuits on re-entry after the operator's
            // out-of-band decision lands.
            let override_decision = invocation
                .ctx
                .extension::<PendingApprovalDecisions>()
                .and_then(|o| o.get(&invocation.tool_use_id).cloned());
            let decision = if let Some(d) = override_decision {
                d
            } else {
                let request = ApprovalRequest::new(
                    invocation.tool_use_id.clone(),
                    invocation.metadata.name.clone(),
                    invocation.input.clone(),
                );
                approver.decide(&request, &invocation.ctx).await?
            };

            let sink = invocation.ctx.extension::<ToolApprovalEventSinkHandle>();
            let run_id = invocation.ctx.run_id().unwrap_or("").to_owned();
            let tool_use_id = invocation.tool_use_id.clone();
            let tool_name = invocation.metadata.name.clone();
            let input = invocation.input.clone();

            match decision {
                ApprovalDecision::Approve => {
                    if let Some(handle) = sink.as_deref() {
                        handle
                            .inner()
                            .record_approved(&run_id, &tool_use_id, &tool_name)
                            .await;
                    }
                    inner.call(invocation).await
                }
                ApprovalDecision::Reject { reason } => {
                    if let Some(handle) = sink.as_deref() {
                        handle
                            .inner()
                            .record_denied(&run_id, &tool_use_id, &tool_name, &reason)
                            .await;
                    }
                    Err(Error::invalid_request(format!(
                        "approver rejected tool '{tool_name}' dispatch: {reason}"
                    )))
                }
                ApprovalDecision::AwaitExternal => {
                    // Pause the agent via graph interrupt. The
                    // payload identifies the pending approval so
                    // the operator can match it against an out-of-
                    // band review queue. Resume via
                    // `Command::ApproveTool { tool_use_id, decision }`
                    // re-enters the same dispatch with the operator's
                    // decision attached to ctx — the override-lookup
                    // branch above short-circuits without re-asking
                    // the approver.
                    Err(Error::Interrupted {
                        payload: json!({
                            "kind": INTERRUPT_KIND_APPROVAL_PENDING,
                            "run_id": run_id,
                            "tool_use_id": tool_use_id,
                            "tool": tool_name,
                            "input": input,
                        }),
                    })
                }
                // `ApprovalDecision` is `#[non_exhaustive]` — surface
                // any future variant the layer doesn't yet wire as a
                // typed configuration error rather than a silent
                // dispatch.
                _ => Err(Error::config(format!(
                    "ApprovalLayer received an unsupported `ApprovalDecision` variant for tool '{tool_name}'; \
                     update the layer to handle the new variant"
                ))),
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_core::AgentContext;
    use entelix_core::ExecutionContext;
    use entelix_core::tools::{Tool, ToolMetadata, ToolRegistry};
    use serde_json::json;

    use super::*;
    use crate::agent::approver::{AlwaysApprove, ApprovalDecision, ApprovalRequest};

    struct EchoTool {
        metadata: ToolMetadata,
    }

    impl EchoTool {
        fn new() -> Self {
            Self {
                metadata: ToolMetadata::function(
                    "echo",
                    "Echo input verbatim.",
                    json!({ "type": "object" }),
                ),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }

        async fn execute(&self, input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
            Ok(input)
        }
    }

    struct AlwaysReject {
        reason: String,
    }

    #[async_trait]
    impl Approver for AlwaysReject {
        async fn decide(
            &self,
            _request: &ApprovalRequest,
            _ctx: &ExecutionContext,
        ) -> Result<ApprovalDecision> {
            Ok(ApprovalDecision::Reject {
                reason: self.reason.clone(),
            })
        }
    }

    struct CountingApprovalSink {
        approved: Arc<AtomicUsize>,
        denied: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ToolApprovalEventSink for CountingApprovalSink {
        async fn record_approved(&self, _run_id: &str, _tool_use_id: &str, _tool: &str) {
            self.approved.fetch_add(1, Ordering::SeqCst);
        }
        async fn record_denied(
            &self,
            _run_id: &str,
            _tool_use_id: &str,
            _tool: &str,
            _reason: &str,
        ) {
            self.denied.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn approver_approve_dispatches_inner_tool() {
        let approver: Arc<dyn Approver> = Arc::new(AlwaysApprove);
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let ctx = ExecutionContext::new();
        let result = registry
            .dispatch("", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap();
        assert_eq!(result, json!({"x": 1}));
    }

    #[tokio::test]
    async fn approver_reject_short_circuits_dispatch() {
        let approver: Arc<dyn Approver> = Arc::new(AlwaysReject {
            reason: "policy violation".to_owned(),
        });
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let ctx = ExecutionContext::new();
        let err = registry
            .dispatch("", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap_err();
        match err {
            Error::InvalidRequest(msg) => {
                assert!(msg.contains("approver rejected tool 'echo'"), "got: {msg}");
                assert!(msg.contains("policy violation"), "got: {msg}");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn approval_sink_records_both_decisions() {
        let approved = Arc::new(AtomicUsize::new(0));
        let denied = Arc::new(AtomicUsize::new(0));
        let sink = CountingApprovalSink {
            approved: Arc::clone(&approved),
            denied: Arc::clone(&denied),
        };
        let handle = ToolApprovalEventSinkHandle::new(sink);
        let ctx = ExecutionContext::new().add_extension(handle);

        // First dispatch — approver allows.
        let approver_ok: Arc<dyn Approver> = Arc::new(AlwaysApprove);
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver_ok))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        registry
            .dispatch("", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap();
        assert_eq!(approved.load(Ordering::SeqCst), 1);
        assert_eq!(denied.load(Ordering::SeqCst), 0);

        // Second dispatch — approver rejects on a fresh registry.
        let approver_no: Arc<dyn Approver> = Arc::new(AlwaysReject {
            reason: "no".into(),
        });
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver_no))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let _ = registry.dispatch("", "echo", json!({"x": 1}), &ctx).await;
        assert_eq!(approved.load(Ordering::SeqCst), 1);
        assert_eq!(denied.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn approval_layer_runs_without_sink_attached() {
        // No sink in ctx — layer must still gate dispatch correctly.
        let approver: Arc<dyn Approver> = Arc::new(AlwaysApprove);
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let result = registry
            .dispatch("", "echo", json!({"x": 1}), &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(result, json!({"x": 1}));
    }

    struct AlwaysAwait;

    #[async_trait]
    impl Approver for AlwaysAwait {
        async fn decide(
            &self,
            _request: &ApprovalRequest,
            _ctx: &ExecutionContext,
        ) -> Result<ApprovalDecision> {
            Ok(ApprovalDecision::AwaitExternal)
        }
    }

    #[tokio::test]
    async fn await_external_raises_interrupted_with_payload() {
        // The pause-and-resume contract: AwaitExternal must surface
        // as `Error::Interrupted` so the graph dispatch loop can
        // checkpoint pre-state and bubble the typed error to the
        // caller. The payload identifies the pending dispatch so
        // the operator can route it to an out-of-band review queue.
        let approver: Arc<dyn Approver> = Arc::new(AlwaysAwait);
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let err = registry
            .dispatch("tu-1", "echo", json!({"x": 1}), &ExecutionContext::new())
            .await
            .unwrap_err();
        match err {
            Error::Interrupted { payload } => {
                assert_eq!(
                    payload["kind"].as_str(),
                    Some(INTERRUPT_KIND_APPROVAL_PENDING)
                );
                assert_eq!(payload["tool_use_id"].as_str(), Some("tu-1"));
                assert_eq!(payload["tool"].as_str(), Some("echo"));
                assert_eq!(payload["input"], json!({"x": 1}));
            }
            other => panic!("expected Interrupted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn approval_decision_overrides_short_circuit_approver() {
        // Resume path simulation: the approver still says "await",
        // but the operator has attached a decision override for
        // this tool_use_id (mimicking what `agent.resume_with(...)`
        // will do once the resume API ships). The layer must use
        // the override and skip the approver.
        let approver: Arc<dyn Approver> = Arc::new(AlwaysAwait);
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let overrides = {
            let mut p = PendingApprovalDecisions::new();
            p.insert("tu-1", ApprovalDecision::Approve);
            p
        };
        let ctx = ExecutionContext::new().add_extension(overrides);

        let result = registry
            .dispatch("tu-1", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap();
        assert_eq!(result, json!({"x": 1}));
    }

    #[tokio::test]
    async fn approval_decision_overrides_propagate_reject_decision() {
        // Operator's out-of-band decision was Reject — the override
        // must propagate that reject through the same code path so
        // the resume produces a typed rejection rather than a
        // re-fired AwaitExternal.
        let approver: Arc<dyn Approver> = Arc::new(AlwaysAwait);
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let mut overrides = PendingApprovalDecisions::new();
        overrides.insert(
            "tu-1",
            ApprovalDecision::Reject {
                reason: "operator declined out-of-band".to_owned(),
            },
        );
        let ctx = ExecutionContext::new().add_extension(overrides);

        let err = registry
            .dispatch("tu-1", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap_err();
        match err {
            Error::InvalidRequest(msg) => {
                assert!(
                    msg.contains("operator declined out-of-band"),
                    "expected override reason, got: {msg}"
                );
            }
            other => panic!("expected InvalidRequest from override, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn approval_decision_overrides_only_apply_to_matching_tool_use_id() {
        // Override is registered for a different tool_use_id — the
        // current dispatch must fall through to the approver.
        let approver: Arc<dyn Approver> = Arc::new(AlwaysAwait);
        let registry = ToolRegistry::new()
            .layer(ApprovalLayer::new(approver))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let mut overrides = PendingApprovalDecisions::new();
        overrides.insert("a-different-id", ApprovalDecision::Approve);
        let ctx = ExecutionContext::new().add_extension(overrides);

        let err = registry
            .dispatch("tu-1", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap_err();
        // Approver runs (no matching override), returns AwaitExternal,
        // layer raises Interrupted.
        assert!(matches!(err, Error::Interrupted { .. }));
    }

    #[tokio::test]
    async fn approval_layer_composes_under_outer_layer() {
        // Cross-layer integration: register `ScopedToolLayer`
        // INNERMOST and `ApprovalLayer` OUTSIDE it (i.e. register
        // ScopedToolLayer first so it ends up nearer the leaf
        // tool, then register ApprovalLayer so it wraps that).
        // The dispatch flow on Approve must be:
        //   ApprovalLayer.call → ScopedToolLayer.call → tool.execute
        // proven by the wrap-counter incrementing AFTER the
        // approver's decision is applied.
        use entelix_core::tools::{ScopedToolLayer, ToolDispatchScope};
        use futures::future::BoxFuture;

        struct ApproveAfterScope {
            scope_wraps: Arc<AtomicUsize>,
        }
        impl ToolDispatchScope for ApproveAfterScope {
            fn wrap(
                &self,
                _ctx: ExecutionContext,
                fut: BoxFuture<'static, Result<Value>>,
            ) -> BoxFuture<'static, Result<Value>> {
                self.scope_wraps.fetch_add(1, Ordering::SeqCst);
                fut
            }
        }

        let scope_wraps = Arc::new(AtomicUsize::new(0));
        let scope = ApproveAfterScope {
            scope_wraps: Arc::clone(&scope_wraps),
        };
        let approver: Arc<dyn Approver> = Arc::new(AlwaysApprove);
        let registry = ToolRegistry::new()
            .layer(ScopedToolLayer::new(scope)) // innermost (registered first)
            .layer(ApprovalLayer::new(approver)) // outermost (registered last)
            .register(Arc::new(EchoTool::new()))
            .unwrap();

        registry
            .dispatch("", "echo", json!({"x": 1}), &ExecutionContext::new())
            .await
            .unwrap();
        // Scope wrap fires once = ApprovalLayer approved + flowed
        // into the inner ScopedToolLayer + then the leaf tool.
        assert_eq!(scope_wraps.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn approval_reject_short_circuits_before_inner_scope() {
        // Mirror of the above for the Reject path: the inner
        // ScopedToolLayer must NOT fire when the outer
        // ApprovalLayer rejects. Verifies the layer ordering keeps
        // approval gating outside scope setup — important for
        // perf-sensitive scopes (e.g. Postgres SET LOCAL) that
        // operators don't want to pay for on rejected calls.
        use entelix_core::tools::{ScopedToolLayer, ToolDispatchScope};
        use futures::future::BoxFuture;

        struct CountScope {
            wraps: Arc<AtomicUsize>,
        }
        impl ToolDispatchScope for CountScope {
            fn wrap(
                &self,
                _ctx: ExecutionContext,
                fut: BoxFuture<'static, Result<Value>>,
            ) -> BoxFuture<'static, Result<Value>> {
                self.wraps.fetch_add(1, Ordering::SeqCst);
                fut
            }
        }

        let wraps = Arc::new(AtomicUsize::new(0));
        let scope = CountScope {
            wraps: Arc::clone(&wraps),
        };
        let approver: Arc<dyn Approver> = Arc::new(AlwaysReject {
            reason: "no".into(),
        });
        let registry = ToolRegistry::new()
            .layer(ScopedToolLayer::new(scope)) // innermost
            .layer(ApprovalLayer::new(approver)) // outermost
            .register(Arc::new(EchoTool::new()))
            .unwrap();

        let _ = registry
            .dispatch("", "echo", json!({"x": 1}), &ExecutionContext::new())
            .await;
        assert_eq!(
            wraps.load(Ordering::SeqCst),
            0,
            "scope wrap must not fire when the outer ApprovalLayer rejects"
        );
    }
}
