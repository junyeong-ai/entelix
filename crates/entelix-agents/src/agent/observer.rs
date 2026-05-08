//! `AgentObserver<S>` — turn-level lifecycle observer (distinct from
//! the per-invocation `tower::Layer` middleware).
//!
//! ## Why this is *not* the deleted `Hook` trait
//!
//! The previous `Hook` trait wrapped *single invocations*
//! (`Service<ModelInvocation>` pre/post). That role belongs to
//! `tower::Layer` and stays there.
//!
//! `AgentObserver` operates at a different abstraction layer:
//! turn boundaries (start of run, terminal completion). These
//! cannot be expressed by `tower::Layer` because Layer sees one
//! invocation, not the sequence of them that defines an agent
//! turn.
//!
//! ## Stacking
//!
//! Observers are registered in order via
//! [`AgentBuilder::observe`](crate::agent::AgentBuilder::observe).
//! For each lifecycle event the agent invokes them sequentially,
//! preserving registration order. An observer returning `Err`
//! aborts the agent — observers that want best-effort semantics
//! must swallow their own errors.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};

/// Turn-level agent-lifecycle observer.
///
/// Both methods default to `Ok(())` — implementations override
/// only the lifecycle points they care about. The trait is
/// dyn-compat so the agent stores observers as
/// `Vec<Arc<dyn AgentObserver<S>>>`.
#[async_trait]
pub trait AgentObserver<S>: Send + Sync
where
    S: Clone + Send + Sync + 'static,
{
    /// Stable identifier for the observer — surfaces in panic
    /// messages and audit traces. Default: empty string.
    fn name(&self) -> &'static str {
        ""
    }

    /// Called once at the start of each agent run, after
    /// `Started` is emitted but before the inner runnable runs.
    /// The observer may inspect (read-only) the inbound `state`.
    /// Returning `Err` aborts the agent before the first
    /// model/tool interaction.
    async fn pre_turn(&self, _state: &S, _ctx: &ExecutionContext) -> Result<()> {
        Ok(())
    }

    /// Called once when the agent reaches a terminal state, just
    /// before the `Complete` event fires. Useful for terminal-
    /// state inspection: recovery detection, summary persistence,
    /// final policy logging.
    async fn on_complete(&self, _state: &S, _ctx: &ExecutionContext) -> Result<()> {
        Ok(())
    }

    /// Called once when the agent run terminates with an error,
    /// instead of [`Self::on_complete`]. Mirrors the success path
    /// for failure observation: per-run error counters, rollback
    /// of side-channel state written under [`Self::pre_turn`],
    /// alerting / pager hooks. The hook does **not** fire for
    /// [`Error::Interrupted`] — HITL pause-and-resume is a control
    /// signal, not a failure; operators observing pauses use the
    /// `AgentEvent` stream instead.
    ///
    /// Observer errors raised from `on_error` are logged via
    /// `tracing::warn!` and dropped — they do **not** replace the
    /// original runnable error. The contract mirrors the audit
    /// channel (invariant 18): failure-path observability
    /// is one-way and never disturbs the failure that's already
    /// in flight.
    async fn on_error(&self, _error: &Error, _ctx: &ExecutionContext) -> Result<()> {
        Ok(())
    }
}

/// Convenience type alias for the dynamic-dispatch handle the
/// agent stores internally.
pub type DynObserver<S> = Arc<dyn AgentObserver<S>>;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Counts each lifecycle event for assertions.
    struct CountingObserver {
        name: &'static str,
        pre_turn: AtomicUsize,
        on_complete: AtomicUsize,
        on_error: AtomicUsize,
    }

    impl CountingObserver {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                pre_turn: AtomicUsize::new(0),
                on_complete: AtomicUsize::new(0),
                on_error: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl AgentObserver<i32> for CountingObserver {
        fn name(&self) -> &'static str {
            self.name
        }
        async fn pre_turn(&self, _state: &i32, _ctx: &ExecutionContext) -> Result<()> {
            self.pre_turn.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_complete(&self, _state: &i32, _ctx: &ExecutionContext) -> Result<()> {
            self.on_complete.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_error(&self, _error: &Error, _ctx: &ExecutionContext) -> Result<()> {
            self.on_error.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn default_methods_are_no_ops() {
        struct Bare;
        #[async_trait]
        impl AgentObserver<i32> for Bare {}
        let observer = Bare;
        let ctx = ExecutionContext::new();
        observer.pre_turn(&0, &ctx).await.unwrap();
        observer.on_complete(&0, &ctx).await.unwrap();
        observer
            .on_error(&Error::config("nope"), &ctx)
            .await
            .unwrap();
        assert_eq!(observer.name(), "");
    }

    #[tokio::test]
    async fn observer_records_each_lifecycle_event() {
        let obs = CountingObserver::new("test");
        let ctx = ExecutionContext::new();
        obs.pre_turn(&0, &ctx).await.unwrap();
        obs.on_complete(&100, &ctx).await.unwrap();
        obs.on_error(&Error::config("nope"), &ctx).await.unwrap();

        assert_eq!(obs.name(), "test");
        assert_eq!(obs.pre_turn.load(Ordering::SeqCst), 1);
        assert_eq!(obs.on_complete.load(Ordering::SeqCst), 1);
        assert_eq!(obs.on_error.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dyn_observer_handle_works_for_storage() {
        let raw = Arc::new(CountingObserver::new("dyn-test"));
        let dyn_obs: DynObserver<i32> = raw.clone();
        let ctx = ExecutionContext::new();
        dyn_obs.pre_turn(&0, &ctx).await.unwrap();
        assert_eq!(raw.pre_turn.load(Ordering::SeqCst), 1);
    }
}
