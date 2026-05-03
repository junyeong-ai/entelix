//! `03_state_graph` — multi-step workflow with conditional routing.
//!
//! Build: `cargo build --example 03_state_graph -p entelix`
//! Run:   `cargo run --example 03_state_graph -p entelix`
//!
//! Demonstrates:
//! - `StateGraph<S>` builder + `compile()`
//! - Conditional edges with a state-based selector
//! - Loop with `recursion_limit` (F6 mitigation)
//! - `CompiledGraph<S>: Runnable<S, S>` — runs via the same `.invoke()`
//!   used by every other entelix runnable.
//!
//! Unlike `01_quickstart` and `02_lcel_chain`, this example has no LLM
//! dependency and runs deterministically in CI.

#![allow(clippy::print_stdout)] // example output goes to the terminal

use entelix::{CompiledGraph, ExecutionContext, Result, Runnable, RunnableLambda, StateGraph};

/// State carried between graph nodes.
#[derive(Clone, Debug)]
struct PlanState {
    task: String,
    iterations: usize,
    log: Vec<String>,
    done: bool,
}

fn build_graph() -> Result<CompiledGraph<PlanState>> {
    // Each node mutates its slice of state and returns the new full state.
    let plan = RunnableLambda::new(|mut s: PlanState, _ctx| async move {
        s.iterations += 1;
        s.log.push(format!(
            "[plan #{}] thinking about: {}",
            s.iterations, s.task
        ));
        Ok::<_, _>(s)
    });

    let execute = RunnableLambda::new(|mut s: PlanState, _ctx| async move {
        s.log
            .push(format!("[execute] iteration {} performed", s.iterations));
        Ok::<_, _>(s)
    });

    let review = RunnableLambda::new(|mut s: PlanState, _ctx| async move {
        // Toy criterion: stop after 3 iterations.
        s.done = s.iterations >= 3;
        s.log.push(format!(
            "[review] done={} after {} iteration(s)",
            s.done, s.iterations
        ));
        Ok::<_, _>(s)
    });

    let answer = RunnableLambda::new(|mut s: PlanState, _ctx| async move {
        s.log.push(format!(
            "[answer] producing final reply for task: {}",
            s.task
        ));
        Ok::<_, _>(s)
    });

    StateGraph::new()
        .add_node("plan", plan)
        .add_node("execute", execute)
        .add_node("review", review)
        .add_node("answer", answer)
        .add_edge("plan", "execute")
        .add_edge("execute", "review")
        .add_conditional_edges(
            "review",
            |s: &PlanState| (if s.done { "done" } else { "loop" }).to_owned(),
            [("loop", "plan"), ("done", "answer")],
        )
        .set_entry_point("plan")
        .add_finish_point("answer")
        .with_recursion_limit(20)
        .compile()
}

#[tokio::main]
async fn main() -> Result<()> {
    let graph = build_graph()?;

    let initial = PlanState {
        task: "Build a state-graph demo for entelix".to_owned(),
        iterations: 0,
        log: Vec::new(),
        done: false,
    };

    let final_state = graph.invoke(initial, &ExecutionContext::new()).await?;

    println!("=== final state ===");
    println!("iterations: {}", final_state.iterations);
    println!("done:       {}", final_state.done);
    println!();
    println!("=== execution log ===");
    for entry in &final_state.log {
        println!("  {entry}");
    }
    Ok(())
}
