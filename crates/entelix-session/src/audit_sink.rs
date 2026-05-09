//! `SessionAuditSink` — adapter that maps `entelix_core::AuditSink`
//! `record_*` calls onto fire-and-forget `SessionLog::append` calls.
//!
//! The split between `AuditSink` (typed emit channel pinned in
//! `entelix-core`) and `SessionLog` (persistence shape pinned here)
//! lets emitters in `entelix-tools`, `entelix-graph`, `entelix-agents`
//! depend only on `entelix-core` while still landing their events
//! in the durable Tier-2 log. See

use std::sync::Arc;

use chrono::Utc;

use entelix_core::{AuditSink, ThreadKey};

use crate::event::GraphEvent;
use crate::log::SessionLog;

/// Adapter that fans `AuditSink::record_*` calls into a durable
/// [`SessionLog`].
///
/// Cloning is cheap (`Arc`-shared backend handle). One adapter per
/// agent run is the typical pattern — operators construct it next to
/// the `SessionLog` itself and stash it on every spawned
/// [`entelix_core::context::ExecutionContext`] via
/// [`ExecutionContext::with_audit_sink`].
///
/// [`ExecutionContext::with_audit_sink`]:
///     entelix_core::context::ExecutionContext::with_audit_sink
pub struct SessionAuditSink {
    log: Arc<dyn SessionLog>,
    key: ThreadKey,
}

impl SessionAuditSink {
    /// Build an adapter pinned to one `(tenant_id, thread_id)` pair.
    /// Multi-thread runs allocate one adapter per thread; the
    /// adapter is stateless beyond the `Arc` handle so cloning a
    /// sink and re-keying via [`Self::with_thread_key`] is also a
    /// valid pattern.
    #[must_use]
    pub const fn new(log: Arc<dyn SessionLog>, key: ThreadKey) -> Self {
        Self { log, key }
    }

    /// Re-target an existing adapter at a different `ThreadKey`.
    /// Useful when a parent run spawns a sub-thread and wants the
    /// sub-thread's events to land under a distinct audit scope.
    #[must_use]
    pub fn with_thread_key(self, key: ThreadKey) -> Self {
        Self { log: self.log, key }
    }
}

impl std::fmt::Debug for SessionAuditSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionAuditSink")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

/// Fire-and-forget append helper. The audit channel must never
/// block the agent — a dropped event is logged via `tracing::warn!`
/// and the run continues.
fn spawn_append(log: Arc<dyn SessionLog>, key: ThreadKey, event: GraphEvent) {
    tokio::spawn(async move {
        if let Err(err) = log.append(&key, &[event]).await {
            tracing::warn!(
                target: "entelix.session.audit",
                tenant_id = %key.tenant_id(),
                thread_id = %key.thread_id(),
                error = %err,
                "audit-sink append failed; event dropped"
            );
        }
    });
}

impl AuditSink for SessionAuditSink {
    fn record_sub_agent_invoked(&self, agent_id: &str, sub_thread_id: &str) {
        spawn_append(
            Arc::clone(&self.log),
            self.key.clone(),
            GraphEvent::SubAgentInvoked {
                agent_id: agent_id.to_owned(),
                sub_thread_id: sub_thread_id.to_owned(),
                timestamp: Utc::now(),
            },
        );
    }

    fn record_agent_handoff(&self, from: Option<&str>, to: &str) {
        spawn_append(
            Arc::clone(&self.log),
            self.key.clone(),
            GraphEvent::AgentHandoff {
                from: from.map(str::to_owned),
                to: to.to_owned(),
                timestamp: Utc::now(),
            },
        );
    }

    fn record_resumed(&self, from_checkpoint: &str) {
        spawn_append(
            Arc::clone(&self.log),
            self.key.clone(),
            GraphEvent::Resumed {
                from_checkpoint: from_checkpoint.to_owned(),
                timestamp: Utc::now(),
            },
        );
    }

    fn record_memory_recall(&self, tier: &str, namespace_key: &str, hits: usize) {
        spawn_append(
            Arc::clone(&self.log),
            self.key.clone(),
            GraphEvent::MemoryRecall {
                tier: tier.to_owned(),
                namespace_key: namespace_key.to_owned(),
                hits,
                timestamp: Utc::now(),
            },
        );
    }

    fn record_usage_limit_exceeded(&self, breach: &entelix_core::UsageLimitBreach) {
        spawn_append(
            Arc::clone(&self.log),
            self.key.clone(),
            GraphEvent::UsageLimitExceeded {
                breach: breach.clone(),
                timestamp: Utc::now(),
            },
        );
    }
}
