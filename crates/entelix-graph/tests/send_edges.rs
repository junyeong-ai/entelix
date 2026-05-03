//! `StateGraph::add_send_edges` — parallel fan-out integration.
//!
//! Verifies the end-to-end LangGraph-Send-style shape: a source
//! node returns its post-state, a selector splits it into parallel
//! branches, each branch runs its target node concurrently, results
//! fold via the state's `StateMerge::merge` impl into the
//! pre-fan-out state, control flows to the join node.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::ExecutionContext;
use entelix_graph::{Annotated, Append, StateGraph, StateMerge};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, Default, StateMerge)]
struct State {
    log: Annotated<Vec<String>, Append<String>>,
}

fn push(mut s: State, entry: &str) -> State {
    s.log.value.push(entry.into());
    s
}

#[tokio::test]
async fn send_edges_run_branches_in_parallel_and_fold() {
    // Branch A appends "A"; branch B appends "B"; join asserts both
    // branches' log entries arrived. The state's StateMerge derive
    // wires the log-Append reducer; the fan-out fold uses it
    // automatically — no per-call reducer wiring needed.
    let planner = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(push(s, "plan")) });
    let branch_a = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(push(s, "A")) });
    let branch_b = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(push(s, "B")) });
    let join = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(push(s, "join")) });

    let graph = StateGraph::<State>::new()
        .add_node("plan", planner)
        .add_node("a", branch_a)
        .add_node("b", branch_b)
        .add_node("join", join)
        .set_entry_point("plan")
        .add_send_edges(
            "plan",
            ["a", "b"],
            |s: &State| vec![("a".to_owned(), s.clone()), ("b".to_owned(), s.clone())],
            "join",
        )
        .add_finish_point("join")
        .compile()
        .unwrap();

    let result = graph
        .invoke(State::default(), &ExecutionContext::new())
        .await
        .unwrap();
    // The fan-out folds branch contributions into the pre-fan-out
    // state via StateMerge::merge — for an Append-annotated `log`,
    // that's three concatenations: pre + branch_a + branch_b. Branch
    // order is implementation-defined because `try_join_all` polls
    // concurrently, so we sort before asserting.
    let mut log = result.log.value;
    log.sort();
    assert_eq!(
        log,
        vec![
            "A".to_owned(),
            "B".to_owned(),
            "join".to_owned(),
            "plan".to_owned(),
            "plan".to_owned(),
            "plan".to_owned(),
        ],
        "expected one 'plan' from the source plus one per branch start, \
         plus one 'A', one 'B', one 'join' — got {log:?}"
    );
}

#[tokio::test]
async fn send_edges_fail_fast_when_one_branch_errors() {
    let planner = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(push(s, "plan")) });
    let happy = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(push(s, "happy")) });
    let sad = RunnableLambda::new(|_s: State, _ctx| async move {
        Err::<State, _>(entelix_core::Error::config("branch died"))
    });
    let join = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(s) });

    let graph = StateGraph::<State>::new()
        .add_node("plan", planner)
        .add_node("happy", happy)
        .add_node("sad", sad)
        .add_node("join", join)
        .set_entry_point("plan")
        .add_send_edges(
            "plan",
            ["happy", "sad"],
            |s: &State| {
                vec![
                    ("happy".to_owned(), s.clone()),
                    ("sad".to_owned(), s.clone()),
                ]
            },
            "join",
        )
        .add_finish_point("join")
        .compile()
        .unwrap();

    let err = graph
        .invoke(State::default(), &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("branch died"));
}

#[tokio::test]
async fn send_edges_compile_rejects_unknown_join_target() {
    let n = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(s) });
    let err = StateGraph::<State>::new()
        .add_node("plan", n)
        .set_entry_point("plan")
        .add_send_edges("plan", std::iter::empty::<&str>(), |_| Vec::new(), "ghost")
        .add_finish_point("plan")
        .compile()
        .unwrap_err();
    assert!(format!("{err}").contains("ghost"));
}

#[tokio::test]
async fn send_edges_compile_rejects_clash_with_static_edge() {
    let n = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(s) });
    let join = RunnableLambda::new(|s: State, _ctx| async move { Ok::<_, _>(s) });
    let err = StateGraph::<State>::new()
        .add_node("plan", n)
        .add_node("join", join)
        .set_entry_point("plan")
        .add_edge("plan", "join")
        .add_send_edges("plan", std::iter::empty::<&str>(), |_| Vec::new(), "join")
        .add_finish_point("join")
        .compile()
        .unwrap_err();
    assert!(format!("{err}").contains("more than one outgoing edge type"));
}
