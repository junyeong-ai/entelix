//! `ToolEventLayer<S>` — `tower::Layer<Service<ToolInvocation>>`
//! middleware emitting per-tool [`AgentEvent`] variants
//! (`ToolStart` / `ToolComplete` / `ToolError`) to a configured
//! `AgentEventSink`.
//!
//! Wired by recipes alongside the rest of the tool registry. The
//! agent runtime itself never auto-installs this layer; recipe
//! code reads as one explicit line:
//!
//! ```ignore
//! let registry = ToolRegistry::new()
//!     .layer(ToolEventLayer::new(sink.clone()))
//!     .register(my_tool)?;
//! ```
//!
//! When the layer fires, the `run_id` it stamps onto each event is
//! read from `ExecutionContext::run_id()`. Absent context leaves
//! events unstamped (the layer falls through to the inner service
//! without emitting), so wiring the layer outside an agent run is
//! a quiet no-op rather than a panic.
//!
//! ## Cancellation
//!
//! `ToolStart` is emitted before the inner dispatch begins. If the
//! dispatch future is dropped while awaiting the sink (cooperative
//! cancellation, deadline expiry, parent-future drop), observers may
//! see a `ToolStart` for which no matching `ToolComplete` / `ToolError`
//! ever arrives. This is inherent to any "emit-before-dispatch" shape
//! over an async sink — durable correlation lives in the
//! [`entelix_session::SessionAuditSink`] event-log channel
//! (`ToolCall` + `ToolResult` are written through
//! [`AgentEvent::to_graph_event`]), not in the in-flight `AgentEvent`
//! stream. Consumers that need orphan detection time the
//! `(ToolStart, run_id, tool_use_id)` triple against an absent
//! terminal event in their own dashboard.

use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use futures::future::BoxFuture;
use serde_json::Value;
use tower::{Layer, Service, ServiceExt};

use entelix_core::CurrentToolInvocation;
use entelix_core::LlmRenderable;
use entelix_core::error::{Error, Result};
use entelix_core::service::ToolInvocation;

use crate::agent::event::AgentEvent;
use crate::agent::sink::AgentEventSink;

/// Layer that emits per-tool `AgentEvent` variants to the
/// configured sink.
///
/// Generic over the agent state type `S` so the
/// `AgentEvent::ToolStart` / `ToolComplete` / `ToolError` variants
/// share their state-type parameter with the agent's other events
/// on a single sink.
pub struct ToolEventLayer<S>
where
    S: Clone + Send + Sync + 'static,
{
    sink: Arc<dyn AgentEventSink<S>>,
}

impl<S> ToolEventLayer<S>
where
    S: Clone + Send + Sync + 'static,
{
    /// Build with a sink. Cloning the layer is cheap (`Arc`-backed).
    #[must_use]
    pub fn new(sink: Arc<dyn AgentEventSink<S>>) -> Self {
        Self { sink }
    }
}

impl<S> Clone for ToolEventLayer<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            sink: Arc::clone(&self.sink),
        }
    }
}

impl<S> std::fmt::Debug for ToolEventLayer<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEventLayer").finish_non_exhaustive()
    }
}

impl<S, Inner> Layer<Inner> for ToolEventLayer<S>
where
    S: Clone + Send + Sync + 'static,
{
    type Service = ToolEventService<S, Inner>;
    fn layer(&self, inner: Inner) -> Self::Service {
        ToolEventService {
            inner,
            sink: Arc::clone(&self.sink),
        }
    }
}

/// `Service` produced by [`ToolEventLayer`]. Generic over the inner
/// service so the layer composes with any tower-stacked tool path.
pub struct ToolEventService<S, Inner>
where
    S: Clone + Send + Sync + 'static,
{
    inner: Inner,
    sink: Arc<dyn AgentEventSink<S>>,
}

impl<S, Inner: Clone> Clone for ToolEventService<S, Inner>
where
    S: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            sink: Arc::clone(&self.sink),
        }
    }
}

impl<S, Inner> Service<ToolInvocation> for ToolEventService<S, Inner>
where
    S: Clone + Send + Sync + 'static,
    Inner: Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
    Inner::Future: Send + 'static,
{
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut invocation: ToolInvocation) -> Self::Future {
        let inner = self.inner.clone();
        let sink = Arc::clone(&self.sink);
        Box::pin(async move {
            // Run-id stamping is a no-op outside an agent run.
            let run_id = invocation.ctx.run_id().map(str::to_owned);
            let tool = invocation.metadata.name.clone();
            let tool_version = invocation.metadata.version.clone();
            let tool_use_id = invocation.tool_use_id.clone();
            let input = invocation.input.clone();

            if let Some(rid) = &run_id {
                let _ = sink
                    .send(AgentEvent::ToolStart {
                        run_id: rid.clone(),
                        tool_use_id: tool_use_id.clone(),
                        tool: tool.clone(),
                        tool_version: tool_version.clone(),
                        input,
                    })
                    .await;
            }

            // Stamp the per-dispatch identity marker just before
            // dispatch so leaf-tool `ctx.record_phase(...)` calls
            // resolve a stable (tool_use_id, tool_name, started_at)
            // triple whose `started_at` aligns with the dispatch
            // baseline below — `dispatch_elapsed_ms` measures
            // tool-local work without layer-emit overhead leaking in.
            // `tool_use_id` falls back to the tool name when the
            // dispatch did not originate from a model `ToolUse` block.
            let marker_use_id = if tool_use_id.is_empty() {
                tool.clone()
            } else {
                tool_use_id.clone()
            };
            if let Ok(marker) = CurrentToolInvocation::new(marker_use_id, tool.clone()) {
                invocation.ctx = invocation.ctx.clone().add_extension(marker);
            }

            let started_at = Instant::now();
            let result = inner.oneshot(invocation).await;
            let duration_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);

            match (&result, run_id) {
                (Ok(output), Some(rid)) => {
                    let _ = sink
                        .send(AgentEvent::ToolComplete {
                            run_id: rid,
                            tool_use_id,
                            tool,
                            tool_version,
                            duration_ms,
                            output: output.clone(),
                        })
                        .await;
                }
                (Err(err), Some(rid)) => {
                    let _ = sink
                        .send(AgentEvent::ToolError {
                            run_id: rid,
                            tool_use_id,
                            tool,
                            tool_version,
                            error: err.to_string(),
                            error_for_llm: err.for_llm(),
                            wire_code: err.wire_code(),
                            wire_class: err.wire_class(),
                            duration_ms,
                        })
                        .await;
                }
                _ => {}
            }
            result
        })
    }
}
