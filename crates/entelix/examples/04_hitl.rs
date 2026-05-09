//! `04_hitl` — human-in-the-loop graph using `interrupt()` + `Command`.
//!
//! Build: `cargo build --example 04_hitl -p entelix`
//! Run:   `cargo run --example 04_hitl -p entelix`
//!
//! Demonstrates:
//! - `interrupt(payload)` from inside a node halts the graph and returns
//!   the payload to the caller.
//! - `InMemoryCheckpointer` snapshots pre-node state at the interrupt point.
//! - `CompiledGraph::resume_with(Command::Update(state), &ctx)` injects
//!   the human's reply and continues. The `(tenant_id, thread_id)` is
//!   derived from `ctx`.
//!
//! No external API dependency — runs deterministically in CI.

#![allow(clippy::print_stdout)] // example output goes to the terminal

use std::sync::Arc;

use entelix::{
    Command, CompiledGraph, Error, ExecutionContext, InMemoryCheckpointer, Result, Runnable,
    RunnableLambda, StateGraph, interrupt,
};

#[derive(Clone, Debug)]
struct Approval {
    request: String,
    /// Set by the human on resume. `None` triggers the interrupt the
    /// first time the review node runs.
    approved: Option<bool>,
    outcome: Option<String>,
}

fn build_graph(cp: Arc<InMemoryCheckpointer<Approval>>) -> Result<CompiledGraph<Approval>> {
    let review = RunnableLambda::new(|s: Approval, _ctx| async move {
        if s.approved.is_none() {
            return interrupt(serde_json::json!({
                "request": s.request,
                "options": ["approve", "reject"],
            }));
        }
        Ok(s)
    });

    let finalize = RunnableLambda::new(|mut s: Approval, _ctx| async move {
        s.outcome = Some(if matches!(s.approved, Some(true)) {
            "deploy started".into()
        } else {
            "deploy halted".into()
        });
        Ok::<_, _>(s)
    });

    StateGraph::<Approval>::new()
        .add_node("review", review)
        .add_node("finalize", finalize)
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .with_checkpointer(cp)
        .compile()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = build_graph(cp.clone())?;
    let ctx = ExecutionContext::new().with_thread_id("deploy-2026-04-26");

    let initial = Approval {
        request: "Roll out canary build to 5% of prod traffic".into(),
        approved: None,
        outcome: None,
    };

    println!("=== first invocation — should interrupt ===");
    match graph.invoke(initial, &ctx).await {
        Err(Error::Interrupted { kind, payload }) => {
            println!("graph paused for human review.");
            println!("kind: {kind:?}");
            println!("payload: {payload:#}");
        }
        Ok(_) => println!("WARNING: expected an interrupt, got success"),
        Err(other) => println!("unexpected error: {other}"),
    }

    println!();
    println!("=== human approves; resuming with Command::Update ===");
    let approved_state = Approval {
        request: "Roll out canary build to 5% of prod traffic".into(),
        approved: Some(true),
        outcome: None,
    };
    let final_state = graph
        .resume_with(Command::Update(approved_state), &ctx)
        .await?;

    println!("approved: {:?}", final_state.approved);
    println!("outcome:  {:?}", final_state.outcome);
    Ok(())
}
