//! Regression for invariant #18 — `CompiledGraph::resume_with` and
//! `CompiledGraph::resume_from` go through `dispatch_from_checkpoint`,
//! which emits `record_resumed(checkpoint_id)` on the
//! `ExecutionContext`'s audit sink. Validates the emit fires once per
//! resume, carries the resolved checkpoint id, and stays a no-op when
//! the context has no sink.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::TenantId;
use std::sync::Arc;

use parking_lot::Mutex;

use entelix_core::ThreadKey;
use entelix_core::{AuditSink, AuditSinkHandle, ExecutionContext, Result};
use entelix_graph::{Checkpoint, Checkpointer, InMemoryCheckpointer, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Workflow {
    n: i32,
}

#[derive(Default)]
struct RecordingAuditSink {
    resumes: Mutex<Vec<String>>,
}

impl AuditSink for RecordingAuditSink {
    fn record_sub_agent_invoked(&self, _agent_id: &str, _sub_thread_id: &str) {}
    fn record_agent_handoff(&self, _from: Option<&str>, _to: &str) {}
    fn record_resumed(&self, from_checkpoint: &str) {
        self.resumes.lock().push(from_checkpoint.to_owned());
    }
    fn record_memory_recall(&self, _tier: &str, _namespace_key: &str, _hits: usize) {}
    fn record_usage_limit_exceeded(&self, _breach: &entelix_core::UsageLimitBreach) {}

    fn record_context_compacted(&self, _dropped_chars: usize, _retained_chars: usize) {}
}

fn step(delta: i32) -> RunnableLambda<Workflow, Workflow> {
    RunnableLambda::new(move |mut s: Workflow, _ctx| async move {
        s.n += delta;
        Ok::<_, _>(s)
    })
}

#[tokio::test]
async fn resume_emits_resumed_with_checkpoint_id() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step(1))
        .add_node("b", step(10))
        .add_edge("a", "b")
        .set_entry_point("a")
        .add_finish_point("b")
        .with_checkpointer(cp.clone())
        .compile()?;

    let key = ThreadKey::new(TenantId::new("default"), "thread-resume");
    let checkpoint = Checkpoint::new(&key, 1, Workflow { n: 100 }, Some("b".into()));
    let manual_id = checkpoint.id.clone();
    cp.put(checkpoint).await?;

    let sink = Arc::new(RecordingAuditSink::default());
    let ctx = ExecutionContext::new()
        .with_thread_id("thread-resume")
        .with_audit_sink(AuditSinkHandle::new(Arc::clone(&sink) as Arc<dyn AuditSink>));

    let final_state = graph.resume(&ctx).await?;
    assert_eq!(final_state.n, 110);

    let recorded: Vec<_> = sink.resumes.lock().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0], manual_id.to_hyphenated_string());
    Ok(())
}

#[tokio::test]
async fn resume_without_sink_stays_silent() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step(1))
        .set_entry_point("a")
        .add_finish_point("a")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("thread-resume-2");
    let _ = graph.invoke(Workflow { n: 0 }, &ctx).await?;
    // No sink wired — resume must succeed and emit nothing.
    let _ = graph.resume(&ctx).await?;
    Ok(())
}
