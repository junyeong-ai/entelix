//! `AuditSink` — typed channel for managed-agent audit events that
//! the SDK records into a session log.
//!
//! ## Why a `*_core` trait rather than `entelix-session::SessionLog`
//!
//! `SessionLog` is the persistence shape (`async append(events)`).
//! Tools / graphs / recipes that want to *emit* a single audit
//! event need a narrower surface: typed `record_*` methods with no
//! awareness of `GraphEvent`'s exact wire shape, ordinal accounting,
//! or `ThreadKey` plumbing. Pinning `AuditSink` in `entelix-core`
//! also breaks the dependency cycle that would otherwise force
//! `entelix-tools` and `entelix-graph` to depend on `entelix-session`.
//!
//! ## Wire-up
//!
//! Operators wire one `Arc<dyn AuditSink>` per agent run via
//! [`crate::context::ExecutionContext::with_audit_sink`]. The
//! `entelix-session` crate ships `SessionAuditSink` — a tiny
//! adapter that maps each `record_*` call onto a fire-and-forget
//! `SessionLog::append`. Recipes that don't wire a sink see no
//! change in behaviour: the absent extension makes every emit site
//! a no-op via `ctx.audit_sink()` returning `None`.
//!
//! ## Concurrency
//!
//! Methods are `&self` and synchronous so emit sites sit inside hot
//! dispatch loops without `.await` ceremony. Implementations that
//! ultimately persist via async backends spawn a detached task
//! inside the method — the audit channel is fire-and-forget by
//! design (an audit-sink failure must never block the agent).

use std::sync::Arc;

/// Typed audit-event channel. See module docs.
pub trait AuditSink: Send + Sync + 'static {
    /// Record that a sub-agent was dispatched from the parent run.
    fn record_sub_agent_invoked(&self, agent_id: &str, sub_thread_id: &str);

    /// Record that a supervisor recipe handed control between
    /// named agents. `from = None` on the first turn.
    fn record_agent_handoff(&self, from: Option<&str>, to: &str);

    /// Record that the run resumed from a prior checkpoint.
    /// `from_checkpoint` is the empty string when the resume
    /// happened from a fresh state.
    fn record_resumed(&self, from_checkpoint: &str);

    /// Record that a long-term memory tier returned `hits` records
    /// for `namespace_key`. The hits themselves stay outside the
    /// audit channel — the model-facing content already lands in
    /// `AssistantMessage` / `ToolResult`, and storing the full
    /// retrieved corpus inline would balloon the audit trail.
    fn record_memory_recall(&self, tier: &str, namespace_key: &str, hits: usize);

    /// Record that a [`crate::RunBudget`] axis hit its cap and
    /// short-circuited the run with `Error::UsageLimitExceeded`.
    /// `axis` is the lower-snake-case rendering of
    /// [`crate::run_budget::UsageLimitAxis`] (`"requests"`,
    /// `"input_tokens"`, `"output_tokens"`, `"total_tokens"`,
    /// `"tool_calls"`) — strings rather than the typed enum so
    /// `entelix-tools` / `entelix-graph` emit sites stay free of
    /// the `UsageLimitAxis` import. `limit` and `observed` carry
    /// the raw counter values for compliance / billing audits that
    /// need to attribute breaches per-tenant per-run.
    fn record_usage_limit_exceeded(&self, axis: &str, limit: u64, observed: u64);
}

/// `Arc`-shaped handle the [`crate::context::ExecutionContext`]
/// extension carries. Wraps `Arc<dyn AuditSink>` in a newtype so
/// the `Extensions` slot lookup by `TypeId` is unambiguous.
#[derive(Clone)]
pub struct AuditSinkHandle(Arc<dyn AuditSink>);

impl AuditSinkHandle {
    /// Wrap an `Arc<dyn AuditSink>` for stashing in
    /// [`crate::context::ExecutionContext`].
    #[must_use]
    pub const fn new(sink: Arc<dyn AuditSink>) -> Self {
        Self(sink)
    }

    /// Borrow the underlying sink.
    #[must_use]
    pub fn as_sink(&self) -> &dyn AuditSink {
        &*self.0
    }

    /// Clone the underlying `Arc` for use in spawned tasks or
    /// fan-out across multiple emit sites.
    #[must_use]
    pub fn clone_arc(&self) -> Arc<dyn AuditSink> {
        Arc::clone(&self.0)
    }
}

impl std::fmt::Debug for AuditSinkHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuditSinkHandle").finish_non_exhaustive()
    }
}
