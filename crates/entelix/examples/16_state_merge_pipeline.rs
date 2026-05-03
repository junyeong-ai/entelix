//! `16_state_merge_pipeline` — declarative per-field state merge with
//! `#[derive(StateMerge)]`, parallel fan-out, and contribution-style nodes.
//!
//! Build: `cargo build --example 16_state_merge_pipeline -p entelix`
//! Run:   `cargo run --example 16_state_merge_pipeline -p entelix`
//!
//! Demonstrates the Phase-9 `LangGraph` `TypedDict`-parity surface, end-to-end:
//! - `#[derive(Clone, Default, StateMerge)]` over a state struct with
//!   per-field `Annotated<T, R>` reducers.
//! - The macro-emitted `<Name>Contribution` companion struct + builder
//!   methods that take raw inner `T` (no `Annotated::new(...)` boilerplate).
//! - `add_contributing_node` — node returns only the slots it touched;
//!   unwritten slots keep their current value.
//! - `add_send_edges` parallel fan-out — branch results auto-merge via
//!   `<S as StateMerge>::merge` without an explicit reducer parameter.
//! - Conditional routing on the merged state.
//! - Coexistence with the classic `add_node` (full-state replace) shape.
//!
//! No LLM dependency — runs deterministically in CI like `03_state_graph`.

#![allow(clippy::print_stdout)] // example output goes to the terminal

use entelix::{
    Annotated, Append, ExecutionContext, Max, Result, Runnable, RunnableLambda, StateGraph,
    StateMerge,
};

/// State carried between graph nodes. Three field flavours show the
/// per-field reducer choices the derive recognises:
///
/// - `log: Annotated<Vec<String>, Append<String>>` — every node's log
///   contributions accumulate; nothing is overwritten.
/// - `best_score: Annotated<i32, Max<i32>>` — keeps the largest score
///   any branch produced; lower contributions are silently dropped.
/// - `phase: String` — plain field, last-write-wins (`Replace` semantics).
#[derive(Clone, Debug, Default, StateMerge)]
struct ResearchState {
    log: Annotated<Vec<String>, Append<String>>,
    best_score: Annotated<i32, Max<i32>>,
    phase: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── Planner: contribution-style. Writes only `log` and `phase`;
    //    `best_score` is left as `None` in the contribution → its
    //    *current* value (initial 0) survives unchanged. The derive's
    //    `with_*` builder takes raw `Vec<String>` / `String` — no
    //    `Annotated::new(..., Append::new())` boilerplate.
    let plan = RunnableLambda::new(|s: ResearchState, _ctx| async move {
        Ok(ResearchStateContribution::default()
            .with_log(vec![format!("[plan] task=\"{}\"", s.phase)])
            .with_phase("planning".into()))
    });

    // ── Three parallel researchers. Each writes a log entry plus a
    //    candidate score. The fan-out fold (via `<S as StateMerge>::merge`)
    //    accumulates every researcher's log entries (Append) and keeps
    //    the highest score (Max).
    let researcher_alice = RunnableLambda::new(|mut s: ResearchState, _ctx| async move {
        s.log.value.push("[research/alice] score=70".into());
        s.best_score.value = 70;
        Ok::<_, _>(s)
    });
    let researcher_bob = RunnableLambda::new(|mut s: ResearchState, _ctx| async move {
        s.log.value.push("[research/bob] score=85".into());
        s.best_score.value = 85;
        Ok::<_, _>(s)
    });
    let researcher_carol = RunnableLambda::new(|mut s: ResearchState, _ctx| async move {
        s.log.value.push("[research/carol] score=60".into());
        s.best_score.value = 60;
        Ok::<_, _>(s)
    });

    // ── Reviewer: contribution-style. Promotes the merged best_score
    //    visible after the fan-out into the next phase, leaves
    //    best_score itself untouched (Max already kept the winner).
    let review = RunnableLambda::new(|s: ResearchState, _ctx| async move {
        let next_phase = if s.best_score.value >= 80 {
            "ready"
        } else {
            "needs-refinement"
        };
        Ok(ResearchStateContribution::default()
            .with_log(vec![format!(
                "[review] best_score={} → {}",
                s.best_score.value, next_phase
            )])
            .with_phase(next_phase.into()))
    });

    // ── Refiner: contribution-style. Bumps the score floor so the next
    //    research round is more competitive — demonstrates the loop
    //    case where conditional routing brings us back to research.
    let refine = RunnableLambda::new(|s: ResearchState, _ctx| async move {
        Ok(ResearchStateContribution::default()
            .with_log(vec![format!(
                "[refine] floor lifted from best_score={}",
                s.best_score.value
            )])
            .with_phase("refining".into()))
    });

    // ── Finalize: classic full-replace style. Coexists with
    //    contribution-style nodes in the same graph — `add_node`
    //    replaces the state, `add_contributing_node` merges into it.
    let finalize = RunnableLambda::new(|mut s: ResearchState, _ctx| async move {
        s.log.value.push(format!(
            "[finalize] best_score={} after phase=\"{}\"",
            s.best_score.value, s.phase
        ));
        s.phase = "done".into();
        Ok::<_, _>(s)
    });

    let graph = StateGraph::<ResearchState>::new()
        .add_contributing_node("plan", plan)
        .add_node("alice", researcher_alice)
        .add_node("bob", researcher_bob)
        .add_node("carol", researcher_carol)
        .add_contributing_node("review", review)
        .add_contributing_node("refine", refine)
        .add_node("finalize", finalize)
        .set_entry_point("plan")
        // Three-branch parallel fan-out. The selector hands every
        // branch the same input state; each branch stamps its own log
        // + score; results auto-merge via S::merge into the
        // pre-fan-out state. No explicit reducer — derive(StateMerge)
        // wires it.
        .add_send_edges(
            "plan",
            ["alice", "bob", "carol"],
            |s: &ResearchState| {
                vec![
                    ("alice".into(), s.clone()),
                    ("bob".into(), s.clone()),
                    ("carol".into(), s.clone()),
                ]
            },
            "review",
        )
        // After review, route on `phase`. "ready" → finalize;
        // "needs-refinement" → refine and loop back through planning.
        .add_conditional_edges(
            "review",
            |s: &ResearchState| s.phase.clone(),
            [("ready", "finalize"), ("needs-refinement", "refine")],
        )
        .add_edge("refine", "plan")
        .add_finish_point("finalize")
        // F6 mitigation — bound the refinement loop just in case the
        // researchers persistently underperform the threshold.
        .with_recursion_limit(20)
        .compile()?;

    let initial = ResearchState {
        phase: "build a state-merge demo".into(),
        ..Default::default()
    };
    let final_state = graph.invoke(initial, &ExecutionContext::new()).await?;

    println!("=== final state ===");
    println!("phase:      {}", final_state.phase);
    println!("best_score: {}", final_state.best_score.value);
    println!();
    println!("=== execution log ===");
    for entry in &final_state.log.value {
        println!("  {entry}");
    }
    Ok(())
}
