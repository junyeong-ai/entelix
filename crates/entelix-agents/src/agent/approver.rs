//! `Approver` — supervised-mode gate that decides whether a pending
//! tool call should run.
//!
//! The agent (Slice F's recipe wiring) calls `Approver::decide` for
//! every tool call when [`ExecutionMode::Supervised`] is active. The
//! returned [`ApprovalDecision`] drives the agent's next step:
//!
//! - **`Approve`** — proceed with the tool call.
//! - **`Reject { reason }`** — surface the reason to the recipe's
//!   tool-dispatch path, which feeds the lean prose back into the
//!   conversation as a `ToolResult` with `is_error: true`.
//! - **`AwaitExternal`** — the recipe pauses via the underlying
//!   graph's `interrupt(payload)` until an out-of-band reviewer
//!   resolves the [`PendingApproval`] reply channel ([`mpsc`] +
//!   [`oneshot`] in [`ChannelApprover`]). External resolution
//!   translates to `entelix_graph::Command::ApproveTool {
//!   tool_use_id, decision }` on `CompiledGraph::resume_with` —
//!   see for the typed-Command resume flow.
//!
//! Two production-shape concretes ship in this slice:
//!
//! - [`AlwaysApprove`] — degenerate baseline; equivalent to running
//!   the agent in `Auto` mode but with the supervised event
//!   sequence (`ToolCallApproved` → `ToolStart`). Useful for tests
//!   of the supervised pipeline.
//! - [`ChannelApprover`] — production-shape; owns an `mpsc` of
//!   [`PendingApproval`] and lets a separate review process drive
//!   decisions back through the embedded `oneshot` reply.
//!
//! [`ExecutionMode::Supervised`]: crate::agent::mode::ExecutionMode::Supervised

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

// `ApprovalDecision` is defined in `entelix-core::approval` so the
// graph runtime (`Command::ApproveTool`) and the agent runtime
// (`Approver::decide`) share one type without violating the
// workspace DAG. Re-exported from `entelix-agents` so existing
// import paths keep working.
pub use entelix_core::ApprovalDecision;

/// Read-only context the [`Approver`] sees for each decision.
///
/// Designed for forward-extension: marked `#[non_exhaustive]` so
/// future slices can add fields (turn counter, cumulative cost,
/// prior decisions) without breaking the trait surface.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ApprovalRequest {
    /// Tool-use id correlating with the model's `tool_use` block.
    /// Recipes thread this id through the `ToolResult` they
    /// synthesize when an `AwaitExternal` decision finally
    /// resolves.
    pub id: String,
    /// Tool name as registered in the `ToolRegistry`.
    pub name: String,
    /// JSON arguments the model produced.
    pub input: Value,
}

impl ApprovalRequest {
    /// Construct a request — used by recipes (Slice F) when wiring
    /// supervised gating into the agent loop.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

/// Decision-maker for supervised tool execution.
///
/// Implementations should be cheap to clone (`Arc<dyn Approver>` is
/// the canonical handle). `decide` may be called concurrently for
/// distinct tool calls — implementations must be `Send + Sync`.
///
/// ## Idempotency requirement
///
/// `decide` MUST be idempotent on `ApprovalRequest::id`
/// (the `tool_use_id`). Both the agent runtime and the operator's
/// resume path may invoke `decide` for the same `tool_use_id`
/// across calls:
///
/// - **Multi-tool-call dispatch** — when the model emits multiple
///   `ToolUse` blocks in one assistant turn, the agent dispatches
///   them sequentially. If call N returns `AwaitExternal`, the
///   agent pauses; on resume the entire tool node re-fires from
///   the pre-node checkpoint. Calls 0..N-1 thus dispatch *twice*.
///   For consistent behaviour the approver must return the same
///   decision on the second call (typically by checking an external
///   store keyed on `tool_use_id` first).
/// - **`AwaitExternal` resumes** — `Command::ApproveTool` resumes
///   the dispatch with the operator's eventual decision pre-loaded
///   into [`entelix_core::PendingApprovalDecisions`]; the layer's
///   override-lookup runs *before* `decide`, so the approver is
///   only re-invoked for tool-use ids whose decision was not
///   pre-loaded. This bypass is the recommended pattern, but the
///   approver must still be idempotent for the multi-tool-call
///   case above.
///
/// Implementations that cannot guarantee idempotency (e.g. those
/// that consult a stateful queue without keying by `tool_use_id`)
/// will silently approve / reject duplicate requests in unexpected
/// ways. The SDK does not enforce idempotency at runtime — it is a
/// load-bearing contract operators must honour.
#[async_trait]
pub trait Approver: Send + Sync + 'static {
    /// Decide what to do with the pending tool call. Returning
    /// `Err` aborts the agent.
    async fn decide(
        &self,
        request: &ApprovalRequest,
        ctx: &ExecutionContext,
    ) -> Result<ApprovalDecision>;
}

/// Always returns `Approve` — useful as the supervised-mode default
/// when an operator wants the supervised event sequence
/// (`ToolCallApproved` → `ToolStart`) but no actual gating logic.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlwaysApprove;

#[async_trait]
impl Approver for AlwaysApprove {
    async fn decide(
        &self,
        _request: &ApprovalRequest,
        _ctx: &ExecutionContext,
    ) -> Result<ApprovalDecision> {
        Ok(ApprovalDecision::Approve)
    }
}

/// Configurable timeout for waiting on operator decisions.
#[derive(Clone, Copy, Debug)]
pub struct ChannelApproverConfig {
    /// Maximum wall-clock duration to wait for a decision before
    /// the approver auto-rejects with a timeout reason. Defaults to
    /// 5 minutes — the [`ExecutionContext::deadline`] short-circuits
    /// earlier when present.
    pub timeout: std::time::Duration,
}

impl Default for ChannelApproverConfig {
    fn default() -> Self {
        Self {
            timeout: std::time::Duration::from_mins(5),
        }
    }
}

/// One in-flight approval — the operator side. Read these from
/// the `Receiver` paired with [`ChannelApprover`]; resolve by
/// sending an [`ApprovalDecision`] on the embedded `reply`.
#[derive(Debug)]
pub struct PendingApproval {
    /// What the agent is asking about.
    pub request: ApprovalRequest,
    /// Reply channel — caller sends `ApprovalDecision` exactly once.
    pub reply: oneshot::Sender<ApprovalDecision>,
}

/// Production-shape approver backed by an `mpsc` channel.
///
/// The agent forwards every approval request to a `Receiver` owned
/// by the operator; the operator drives decisions back through the
/// embedded `oneshot` reply. A configurable timeout caps the wait
/// (default 5 min) so a stalled reviewer cannot strand the agent.
pub struct ChannelApprover {
    tx: mpsc::Sender<PendingApproval>,
    config: ChannelApproverConfig,
}

impl ChannelApprover {
    /// Build with the default config. Returns the approver and the
    /// matching receiver.
    #[must_use]
    pub fn new(capacity: usize) -> (Self, mpsc::Receiver<PendingApproval>) {
        Self::with_config(capacity, ChannelApproverConfig::default())
    }

    /// Build with an explicit config — useful for tests that pin a
    /// short timeout.
    #[must_use]
    pub fn with_config(
        capacity: usize,
        config: ChannelApproverConfig,
    ) -> (Self, mpsc::Receiver<PendingApproval>) {
        let (tx, rx) = mpsc::channel(capacity);
        (Self { tx, config }, rx)
    }

    fn deadline_for(&self, ctx: &ExecutionContext) -> tokio::time::Instant {
        let cfg_deadline = tokio::time::Instant::now() + self.config.timeout;
        ctx.deadline()
            .map_or(cfg_deadline, |ctx_deadline| ctx_deadline.min(cfg_deadline))
    }
}

impl Clone for ChannelApprover {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            config: self.config,
        }
    }
}

impl std::fmt::Debug for ChannelApprover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelApprover")
            .field("timeout", &self.config.timeout)
            .field("closed", &self.tx.is_closed())
            .finish()
    }
}

#[async_trait]
impl Approver for ChannelApprover {
    async fn decide(
        &self,
        request: &ApprovalRequest,
        ctx: &ExecutionContext,
    ) -> Result<ApprovalDecision> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        let pending = PendingApproval {
            request: request.clone(),
            reply: reply_tx,
        };
        self.tx.send(pending).await.map_err(|_| {
            Error::config("ChannelApprover: receiver dropped before approval was requested")
        })?;
        let deadline = self.deadline_for(ctx);
        let cancellation = ctx.cancellation().clone();
        tokio::select! {
            biased;
            () = cancellation.cancelled() => Err(Error::Cancelled),
            decision = reply_rx => decision.map_err(|_| {
                Error::config("ChannelApprover: reply channel dropped without decision")
            }),
            () = tokio::time::sleep_until(deadline) => Ok(ApprovalDecision::Reject {
                reason: format!(
                    "supervised approval timed out (no decision within {:?})",
                    self.config.timeout
                ),
            }),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn req() -> ApprovalRequest {
        ApprovalRequest::new("call-1", "echo", serde_json::json!({"x": 1}))
    }

    #[tokio::test]
    async fn always_approve_returns_approve() {
        let approver = AlwaysApprove;
        let decision = approver
            .decide(&req(), &ExecutionContext::new())
            .await
            .unwrap();
        assert!(matches!(decision, ApprovalDecision::Approve));
    }

    #[tokio::test]
    async fn channel_approver_round_trips_approve() {
        let (approver, mut rx) = ChannelApprover::new(4);
        let approver_clone = approver.clone();
        let decide = tokio::spawn(async move {
            approver_clone
                .decide(&req(), &ExecutionContext::new())
                .await
        });

        let pending = rx.recv().await.unwrap();
        assert_eq!(pending.request.id, "call-1");
        pending.reply.send(ApprovalDecision::Approve).unwrap();

        let decision = decide.await.unwrap().unwrap();
        assert!(matches!(decision, ApprovalDecision::Approve));
    }

    #[tokio::test]
    async fn channel_approver_round_trips_reject_with_reason() {
        let (approver, mut rx) = ChannelApprover::new(4);
        let approver_clone = approver.clone();
        let decide = tokio::spawn(async move {
            approver_clone
                .decide(&req(), &ExecutionContext::new())
                .await
        });
        let pending = rx.recv().await.unwrap();
        pending
            .reply
            .send(ApprovalDecision::Reject {
                reason: "operator denied".into(),
            })
            .unwrap();
        let decision = decide.await.unwrap().unwrap();
        match decision {
            ApprovalDecision::Reject { reason } => assert_eq!(reason, "operator denied"),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn channel_approver_times_out_when_operator_silent() {
        let (approver, _rx_keeper) = ChannelApprover::with_config(
            4,
            ChannelApproverConfig {
                timeout: Duration::from_millis(50),
            },
        );
        let decision = approver
            .decide(&req(), &ExecutionContext::new())
            .await
            .unwrap();
        match decision {
            ApprovalDecision::Reject { reason } => {
                assert!(reason.contains("timed out"), "{reason}");
            }
            other => panic!("expected Reject(timeout), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn channel_approver_propagates_cancellation() {
        let (approver, _rx_keeper) = ChannelApprover::new(4);
        let ctx = ExecutionContext::new();
        let cancellation = ctx.cancellation().clone();
        let approver_clone = approver.clone();

        let decide = tokio::spawn(async move { approver_clone.decide(&req(), &ctx).await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancellation.cancel();

        let result = decide.await.unwrap();
        assert!(matches!(result, Err(Error::Cancelled)));
    }

    #[tokio::test]
    async fn channel_approver_errors_when_receiver_dropped_before_request() {
        let (approver, rx) = ChannelApprover::new(4);
        drop(rx);
        let err = approver
            .decide(&req(), &ExecutionContext::new())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("receiver dropped"));
    }
}
