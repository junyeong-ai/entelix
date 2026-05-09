//! `11_durable_session` — pod-kill / resume demonstration (invariant 2).
//!
//! Build: `cargo build --example 11_durable_session -p entelix`
//! Run:   `cargo run   --example 11_durable_session -p entelix`
//!
//! The harness is stateless: an `Agent` (or `CompiledGraph`) owns no
//! persistent state beyond the current request scope. A pod can die
//! mid-conversation and a *different* pod can pick up the same
//! `thread_id` from durable storage. This example simulates that with
//! `InMemoryCheckpointer` so it stays CI-deterministic; for production
//! the only line that changes is the checkpointer constructor:
//!
//! ```ignore
//! // CI / examples
//! let cp: Arc<dyn Checkpointer<S>> = Arc::new(InMemoryCheckpointer::<S>::new());
//!
//! // Production
//! let cp: Arc<dyn Checkpointer<S>> = Arc::new(
//!     entelix_persistence::PostgresCheckpointer::connect(database_url).await?,
//! );
//! ```
//!
//! Wire shape demonstrated:
//!
//! 1. Pod 1 runs the graph for `thread_id = "conv-42"`. The `review`
//!    node calls `interrupt()` to simulate a pause point — that's the
//!    "pod kill" trigger.
//! 2. Pod 1 is dropped. Its `CompiledGraph` Arc is gone.
//! 3. Pod 2 reconstructs an *independent* `CompiledGraph` instance from
//!    the same definition + the same `Checkpointer`.
//! 4. Pod 2 calls `resume_with(Command::Update(...), &ctx)`; the
//!    `(tenant_id, thread_id)` is derived from `ctx`, the work resumes
//!    from the checkpoint and runs to completion.

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use entelix::{
    Checkpointer, Command, CompiledGraph, Error, ExecutionContext, InMemoryCheckpointer, Result,
    Runnable, RunnableLambda, StateGraph, TenantId, ThreadKey, interrupt,
};

#[derive(Clone, Debug)]
struct WorkflowState {
    task: String,
    plan_complete: bool,
    approved: Option<bool>,
    outcome: Option<String>,
    log: Vec<String>,
}

const THREAD_ID: &str = "conv-42";

/// Build the same graph definition the way every pod would. The
/// `Checkpointer` is `Arc<dyn Checkpointer<S>>` so swapping in-memory
/// for Postgres/Redis is a one-line change at the call site.
fn build_graph(
    checkpointer: Arc<dyn Checkpointer<WorkflowState>>,
) -> Result<CompiledGraph<WorkflowState>> {
    let plan = RunnableLambda::new(|mut s: WorkflowState, _ctx| async move {
        s.plan_complete = true;
        s.log.push(format!("[plan] decomposed: {}", s.task));
        Ok::<_, _>(s)
    });

    // The review node halts the graph until a human (or another pod)
    // resumes with an `approved` decision. `interrupt()` checkpoints
    // pre-node state so a fresh harness can pick up later.
    let review = RunnableLambda::new(|s: WorkflowState, _ctx| async move {
        if s.approved.is_none() {
            return interrupt(serde_json::json!({
                "task": s.task,
                "options": ["approve", "reject"],
            }));
        }
        Ok(s)
    });

    let finish = RunnableLambda::new(|mut s: WorkflowState, _ctx| async move {
        s.outcome = Some(match s.approved {
            Some(true) => "shipped".into(),
            Some(false) => "halted".into(),
            None => "no decision".into(),
        });
        s.log.push(format!("[finish] outcome={:?}", s.outcome));
        Ok::<_, _>(s)
    });

    StateGraph::new()
        .add_node("plan", plan)
        .add_node("review", review)
        .add_node("finish", finish)
        .add_edge("plan", "review")
        .add_edge("review", "finish")
        .set_entry_point("plan")
        .add_finish_point("finish")
        .with_checkpointer(checkpointer)
        .compile()
}

#[tokio::main]
async fn main() -> Result<()> {
    // The Checkpointer is the *only* shared piece between pods. Each
    // pod builds its own `CompiledGraph` from the same definition.
    let checkpointer: Arc<dyn Checkpointer<WorkflowState>> =
        Arc::new(InMemoryCheckpointer::<WorkflowState>::new());

    let initial = WorkflowState {
        task: "ship the durable resume demo".to_owned(),
        plan_complete: false,
        approved: None,
        outcome: None,
        log: Vec::new(),
    };

    // ── Pod 1: starts the work; halts at `review` ────────────────────
    println!("── pod 1 (will be killed mid-flight) ────────────");
    let pod1_ctx = ExecutionContext::new()
        .with_tenant_id(TenantId::new("acme"))
        .with_thread_id(THREAD_ID);

    {
        let graph_pod1 = build_graph(checkpointer.clone())?;
        match graph_pod1.invoke(initial, &pod1_ctx).await {
            Err(Error::Interrupted { kind, payload }) => {
                println!("pod-1 halted at review; kind={kind:?}; payload: {payload:#}");
            }
            other => println!("unexpected: {other:?}"),
        }
        // graph_pod1 is dropped here. The harness vanishes; only the
        // checkpointer survives.
    }

    let key = ThreadKey::from_ctx(&pod1_ctx)?;
    let history = checkpointer.list_history(&key, usize::MAX).await?;
    println!(
        "checkpointer holds {} checkpoint(s) for tenant '{}' thread '{}'",
        history.len(),
        key.tenant_id(),
        key.thread_id(),
    );

    // ── Pod 2: cold start; reconstructs the graph; resumes ───────────
    println!("\n── pod 2 (cold start, fresh harness) ────────────");
    let pod2_ctx = ExecutionContext::new()
        .with_tenant_id(TenantId::new("acme"))
        .with_thread_id(THREAD_ID);

    let graph_pod2 = build_graph(checkpointer.clone())?;
    assert!(
        graph_pod2.has_checkpointer(),
        "pod 2 must see the checkpointer"
    );

    let approved = WorkflowState {
        task: "ship the durable resume demo".to_owned(),
        plan_complete: true,
        approved: Some(true),
        outcome: None,
        log: vec!["[plan] decomposed: ship the durable resume demo".to_owned()],
    };

    let final_state = graph_pod2
        .resume_with(Command::Update(approved), &pod2_ctx)
        .await?;

    println!("pod-2 resumed; outcome={:?}", final_state.outcome);
    for line in &final_state.log {
        println!("  {line}");
    }

    let key2 = ThreadKey::from_ctx(&pod2_ctx)?;
    let history_after = checkpointer.list_history(&key2, usize::MAX).await?;
    println!("\ncheckpoint count after resume: {}", history_after.len());
    println!("✓ harness was stateless across pods (invariant 2)");
    Ok(())
}
