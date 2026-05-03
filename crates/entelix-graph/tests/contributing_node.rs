//! `add_contributing_node` + `#[derive(StateMerge)]` integration —
//! proves the declarative-merge ergonomic composes inside a real
//! `StateGraph` and that per-field reducers fire across multiple
//! sequential nodes without per-graph closures. Also exercises the
//! `<Name>Contribution` companion struct + builder methods that
//! the derive emits.

#![allow(clippy::unwrap_used)]

use entelix_core::context::ExecutionContext;
use entelix_graph::{Annotated, Append, Max, StateGraph, StateMerge};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, Default, StateMerge)]
struct PlanState {
    log: Annotated<Vec<String>, Append<String>>,
    score: Annotated<i32, Max<i32>>,
    last_phase: String,
}

#[tokio::test]
async fn contributing_nodes_chain_per_field_reducers() {
    // Two sequential nodes; each names exactly the slots it
    // touched via the derive-emitted Contribution builder. The
    // framework auto-merges via StateMerge.
    let planner = RunnableLambda::new(|s: PlanState, _ctx| async move {
        // Planner writes log + score + last_phase.
        Ok::<_, _>(
            PlanStateContribution::default()
                .with_log(vec![format!("plan@{}", s.score.value)])
                .with_score(50)
                .with_last_phase("plan".into()),
        )
    });
    let executor = RunnableLambda::new(|s: PlanState, _ctx| async move {
        // Executor writes log + last_phase, leaves score untouched.
        // Score must stay at planner's high-water mark even though
        // executor would have produced a lower value.
        Ok::<_, _>(
            PlanStateContribution::default()
                .with_log(vec![format!("execute@{}", s.score.value)])
                .with_last_phase("execute".into()),
        )
    });

    let graph = StateGraph::<PlanState>::new()
        .add_contributing_node("plan", planner)
        .add_contributing_node("execute", executor)
        .add_edge("plan", "execute")
        .set_entry_point("plan")
        .add_finish_point("execute")
        .compile()
        .unwrap();

    let initial = PlanState {
        log: Annotated::new(vec!["seed".into()], Append::new()),
        score: Annotated::new(80, Max::new()),
        last_phase: "init".into(),
    };
    let final_state = graph
        .invoke(initial, &ExecutionContext::new())
        .await
        .unwrap();

    // Append: seed → plan@80 → execute@80
    assert_eq!(
        final_state.log.value,
        vec![
            "seed".to_owned(),
            "plan@80".to_owned(),
            "execute@80".to_owned(),
        ]
    );
    // Max: 80 (initial) > 50 (planner) > untouched (executor) → 80
    assert_eq!(final_state.score.value, 80);
    // Replace: last write wins → "execute"
    assert_eq!(final_state.last_phase, "execute");
}

#[tokio::test]
async fn contribution_with_unwritten_slot_keeps_current_value() {
    // Regression: a contribution that doesn't set `score` must
    // NOT regress a negative current value to the default 0.
    let stamper = RunnableLambda::new(|_s: PlanState, _ctx| async move {
        Ok::<_, _>(PlanStateContribution::default().with_last_phase("stamped".into()))
    });

    let graph = StateGraph::<PlanState>::new()
        .add_contributing_node("stamp", stamper)
        .set_entry_point("stamp")
        .add_finish_point("stamp")
        .compile()
        .unwrap();

    let initial = PlanState {
        log: Annotated::new(vec!["seed".into()], Append::new()),
        score: Annotated::new(-100, Max::new()),
        last_phase: "init".into(),
    };
    let final_state = graph
        .invoke(initial, &ExecutionContext::new())
        .await
        .unwrap();
    // Score: untouched → -100 keeps. The simple same-shape adapter
    // would have collapsed this via Max(-100, 0) = 0.
    assert_eq!(final_state.score.value, -100);
    assert_eq!(final_state.last_phase, "stamped");
}

#[tokio::test]
async fn contributing_node_coexists_with_plain_add_node() {
    // Mix `add_node` (full-state replace) with
    // `add_contributing_node` (declarative merge). Both shapes
    // nest in the same graph.
    let appender = RunnableLambda::new(|_s: PlanState, _ctx| async move {
        Ok::<_, _>(PlanStateContribution::default().with_log(vec!["from-contrib".into()]))
    });
    let stamper = RunnableLambda::new(|s: PlanState, _ctx| async move {
        Ok::<_, _>(PlanState {
            last_phase: "stamped".into(),
            ..s
        })
    });

    let graph = StateGraph::<PlanState>::new()
        .add_contributing_node("contrib", appender)
        .add_node("stamp", stamper)
        .add_edge("contrib", "stamp")
        .set_entry_point("contrib")
        .add_finish_point("stamp")
        .compile()
        .unwrap();

    let initial = PlanState {
        log: Annotated::new(vec!["seed".into()], Append::new()),
        score: Annotated::new(5, Max::new()),
        last_phase: "init".into(),
    };
    let final_state = graph
        .invoke(initial, &ExecutionContext::new())
        .await
        .unwrap();
    assert_eq!(
        final_state.log.value,
        vec!["seed".to_owned(), "from-contrib".to_owned()]
    );
    assert_eq!(final_state.last_phase, "stamped");
}
