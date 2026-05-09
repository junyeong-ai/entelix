//! `Agent::execute` emits `record_usage_limit_exceeded` exactly
//! once when a `RunBudget` axis breaches, ordering before the
//! `Failed` sink event so audit consumers observe the typed row
//! ahead of the operator-facing failure surface.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::sync::Arc;

use async_trait::async_trait;
use entelix_agents::{Agent, AgentEvent, CaptureSink};
use entelix_core::audit::{AuditSink, AuditSinkHandle};
use entelix_core::{Error, ExecutionContext, Result, RunBudget};
use entelix_runnable::Runnable;
use parking_lot::Mutex;

#[derive(Default)]
struct CapturingAuditSink {
    breaches: Mutex<Vec<entelix_core::UsageLimitBreach>>,
    other_calls: Mutex<usize>,
}

impl CapturingAuditSink {
    fn breaches(&self) -> Vec<entelix_core::UsageLimitBreach> {
        self.breaches.lock().clone()
    }
    fn other_call_count(&self) -> usize {
        *self.other_calls.lock()
    }
}

impl AuditSink for CapturingAuditSink {
    fn record_sub_agent_invoked(&self, _agent_id: &str, _sub_thread_id: &str) {
        *self.other_calls.lock() += 1;
    }
    fn record_agent_handoff(&self, _from: Option<&str>, _to: &str) {
        *self.other_calls.lock() += 1;
    }
    fn record_resumed(&self, _from_checkpoint: &str) {
        *self.other_calls.lock() += 1;
    }
    fn record_memory_recall(&self, _tier: &str, _namespace_key: &str, _hits: usize) {
        *self.other_calls.lock() += 1;
    }
    fn record_usage_limit_exceeded(&self, breach: &entelix_core::UsageLimitBreach) {
        self.breaches.lock().push(breach.clone());
    }

    fn record_context_compacted(&self, _dropped_chars: usize, _retained_chars: usize) {}
}

struct TwoRequestRunnable;

#[async_trait]
impl Runnable<i32, i32> for TwoRequestRunnable {
    async fn invoke(&self, input: i32, ctx: &ExecutionContext) -> Result<i32> {
        let budget = ctx.run_budget().expect("test ctx must carry a RunBudget");
        budget.check_pre_request()?;
        budget.check_pre_request()?;
        Ok(input)
    }
}

#[tokio::test]
async fn breach_emits_record_usage_limit_exceeded_with_typed_fields() {
    let audit = Arc::new(CapturingAuditSink::default());
    let agent = Agent::<i32>::builder()
        .with_name("audit-breach-test")
        .with_runnable(TwoRequestRunnable)
        .build()
        .unwrap();

    let ctx = ExecutionContext::new()
        .with_run_budget(RunBudget::unlimited().with_request_limit(1))
        .with_audit_sink(AuditSinkHandle::new(audit.clone()));

    let err = agent.execute(0, &ctx).await.unwrap_err();
    match &err {
        Error::UsageLimitExceeded(entelix_core::UsageLimitBreach::Requests {
            limit: 1,
            observed: 1,
        }) => {}
        other => panic!("unexpected: {other:?}"),
    }

    let breaches = audit.breaches();
    assert_eq!(breaches.len(), 1);
    assert_eq!(
        breaches[0],
        entelix_core::UsageLimitBreach::Requests {
            limit: 1,
            observed: 1,
        }
    );
    assert_eq!(audit.other_call_count(), 0);
}

#[tokio::test]
async fn breach_without_audit_sink_propagates_error_without_emit() {
    let agent = Agent::<i32>::builder()
        .with_name("audit-breach-noopt")
        .with_runnable(TwoRequestRunnable)
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_run_budget(RunBudget::unlimited().with_request_limit(1));

    let err = agent.execute(0, &ctx).await.unwrap_err();
    assert!(matches!(err, Error::UsageLimitExceeded { .. }));
}

#[tokio::test]
async fn breach_audit_emit_orders_before_failed_sink_event() {
    let audit = Arc::new(CapturingAuditSink::default());
    let event_sink = CaptureSink::<i32>::new();

    let agent = Agent::<i32>::builder()
        .with_name("audit-ordering-test")
        .with_runnable(TwoRequestRunnable)
        .with_sink(event_sink.clone())
        .build()
        .unwrap();

    let ctx = ExecutionContext::new()
        .with_run_budget(RunBudget::unlimited().with_request_limit(1))
        .with_audit_sink(AuditSinkHandle::new(audit.clone()));
    let _ = agent.execute(0, &ctx).await.unwrap_err();

    assert_eq!(audit.breaches().len(), 1);
    let events = event_sink.events();
    assert!(matches!(events.last(), Some(AgentEvent::Failed { .. })));
}
