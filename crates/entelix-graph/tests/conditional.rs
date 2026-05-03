//! Conditional edges + `END` sentinel tests.

#![allow(clippy::unwrap_used)]

use entelix_core::{Error, ExecutionContext, Result};
use entelix_graph::{END, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Pipeline {
    n: i32,
    label: String,
}

fn step(label: &'static str, delta: i32) -> RunnableLambda<Pipeline, Pipeline> {
    RunnableLambda::new(move |mut s: Pipeline, _ctx| async move {
        s.n += delta;
        s.label = format!("{}->{label}", s.label);
        Ok::<_, _>(s)
    })
}

#[tokio::test]
async fn conditional_route_picks_branch_by_state() -> Result<()> {
    let graph = StateGraph::<Pipeline>::new()
        .add_node("router", step("router", 0))
        .add_node("plus", step("plus", 10))
        .add_node("minus", step("minus", -10))
        .add_conditional_edges(
            "router",
            |s: &Pipeline| (if s.n >= 0 { "pos" } else { "neg" }).to_owned(),
            [("pos", "plus"), ("neg", "minus")],
        )
        .add_edge("plus", "minus")
        .set_entry_point("router")
        .add_finish_point("minus")
        .compile()?;

    // n = 5 → pos branch (plus then minus): 5 → 15 → 5
    let out = graph
        .invoke(
            Pipeline {
                n: 5,
                label: "start".into(),
            },
            &ExecutionContext::new(),
        )
        .await?;
    assert_eq!(out.n, 5);
    assert_eq!(out.label, "start->router->plus->minus");
    Ok(())
}

#[tokio::test]
async fn conditional_route_to_end_terminates_graph() -> Result<()> {
    // ReAct-style loop: agent decides "tool" or END each turn. The
    // `unused` node is registered as a finish-point only to satisfy
    // compile-time validation (every graph needs ≥ 1 finish-point); the
    // loop terminates via the END sentinel before reaching it.
    let graph = StateGraph::<Pipeline>::new()
        .add_node("agent", step("agent", 1))
        .add_node("tool", step("tool", 5))
        .add_node("unused", step("unused", 0))
        .add_conditional_edges(
            "agent",
            |s: &Pipeline| (if s.n >= 3 { "stop" } else { "act" }).to_owned(),
            [("act", "tool"), ("stop", END)],
        )
        .add_edge("tool", "agent")
        .set_entry_point("agent")
        .add_finish_point("unused") // satisfies compile validation; unreachable here
        .compile()?;

    let out = graph
        .invoke(
            Pipeline {
                n: 0,
                label: "start".into(),
            },
            &ExecutionContext::new(),
        )
        .await?;
    // Trace: agent(0→1, "act") → tool(1→6) → agent(6→7, "stop" → END)
    assert_eq!(out.n, 7);
    assert!(out.label.ends_with("->agent"));
    Ok(())
}

#[tokio::test]
async fn missing_mapping_key_at_runtime_returns_invalid_request() {
    let graph = StateGraph::<i32>::new()
        .add_node(
            "router",
            RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x) }),
        )
        .add_node(
            "a",
            RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x) }),
        )
        .add_conditional_edges(
            "router",
            |_s: &i32| "missing-key".to_owned(),
            [("known", "a")],
        )
        .set_entry_point("router")
        .add_finish_point("a")
        .compile()
        .unwrap();

    let err = graph.invoke(0, &ExecutionContext::new()).await.unwrap_err();
    assert!(
        matches!(&err, Error::InvalidRequest(msg) if msg.contains("missing-key")),
        "got {err:?}"
    );
}

// ── compile-time validation ────────────────────────────────────────────────

#[test]
fn compile_rejects_static_and_conditional_on_same_node() {
    let err = StateGraph::<i32>::new()
        .add_node(
            "x",
            RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n) }),
        )
        .add_node(
            "y",
            RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n) }),
        )
        .add_edge("x", "y")
        .add_conditional_edges("x", |_s: &i32| "k".to_owned(), [("k", "y")])
        .set_entry_point("x")
        .add_finish_point("y")
        .compile()
        .unwrap_err();
    assert!(
        matches!(&err, Error::Config(msg) if msg.contains("both a static edge and a conditional")),
        "got {err:?}"
    );
}

#[test]
fn compile_rejects_unknown_conditional_target() {
    let err = StateGraph::<i32>::new()
        .add_node(
            "x",
            RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n) }),
        )
        .add_conditional_edges("x", |_s: &i32| "k".to_owned(), [("k", "ghost")])
        .set_entry_point("x")
        .add_finish_point("x")
        .compile()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

#[test]
fn compile_accepts_conditional_target_to_end() {
    let result = StateGraph::<i32>::new()
        .add_node(
            "x",
            RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n) }),
        )
        .add_node(
            "y",
            RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n) }),
        )
        .add_conditional_edges(
            "x",
            |_s: &i32| "stop".to_owned(),
            [("stop", END), ("go", "y")],
        )
        .set_entry_point("x")
        .add_finish_point("y")
        .compile();
    assert!(result.is_ok());
}

#[test]
fn conditional_edge_count_reflects_registrations() {
    let graph = StateGraph::<i32>::new()
        .add_node(
            "a",
            RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n) }),
        )
        .add_node(
            "b",
            RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n) }),
        )
        .add_conditional_edges("a", |_s: &i32| "x".to_owned(), [("x", "b")]);
    assert_eq!(graph.conditional_edge_count(), 1);
    assert_eq!(graph.edge_count(), 0);
}
