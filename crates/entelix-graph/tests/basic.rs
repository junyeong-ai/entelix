//! `StateGraph<S>` + `CompiledGraph<S>` basic tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::{Error, ExecutionContext, Result};
use entelix_graph::{DEFAULT_RECURSION_LIMIT, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

/// State type — a counter plus an audit trail.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Counter {
    n: i32,
    trail: Vec<&'static str>,
}

fn add_one(label: &'static str) -> RunnableLambda<Counter, Counter> {
    RunnableLambda::new(move |mut s: Counter, _ctx| async move {
        s.n += 1;
        s.trail.push(label);
        Ok::<_, _>(s)
    })
}

#[tokio::test]
async fn linear_three_node_graph_executes_in_order() -> Result<()> {
    let graph = StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .add_node("b", add_one("b"))
        .add_node("c", add_one("c"))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_entry_point("a")
        .add_finish_point("c")
        .compile()?;

    let out = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ExecutionContext::new(),
        )
        .await?;
    assert_eq!(out.n, 3);
    assert_eq!(out.trail, vec!["a", "b", "c"]);
    Ok(())
}

#[tokio::test]
async fn entry_point_finish_node_runs_once() -> Result<()> {
    let graph = StateGraph::<Counter>::new()
        .add_node("only", add_one("only"))
        .set_entry_point("only")
        .add_finish_point("only")
        .compile()?;

    let out = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ExecutionContext::new(),
        )
        .await?;
    assert_eq!(out.n, 1);
    assert_eq!(out.trail, vec!["only"]);
    Ok(())
}

#[tokio::test]
async fn recursion_limit_breaks_infinite_cycle() {
    let graph = StateGraph::<Counter>::new()
        .add_node("loop", add_one("loop"))
        .add_node("sink", add_one("sink"))
        .add_edge("loop", "loop") // self-loop
        .add_finish_point("sink") // unreachable
        .set_entry_point("loop")
        .with_recursion_limit(7)
        .compile()
        .unwrap();

    let err = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ExecutionContext::new(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidRequest(msg) if msg.contains("recursion limit")),
        "got {err:?}"
    );
}

#[tokio::test]
async fn run_overrides_max_iterations_lowers_effective_cap() {
    // Compile-time cap is 50; per-call override of 5 must lower the
    // effective cap. The self-loop should trip after 5 iterations.
    let graph = StateGraph::<Counter>::new()
        .add_node("loop", add_one("loop"))
        .add_node("sink", add_one("sink"))
        .add_edge("loop", "loop")
        .add_finish_point("sink")
        .set_entry_point("loop")
        .with_recursion_limit(50)
        .compile()
        .unwrap();
    let ctx = ExecutionContext::new()
        .add_extension(entelix_core::RunOverrides::new().with_max_iterations(5));
    let err = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidRequest(msg) if msg.contains("recursion limit (5)")),
        "expected effective cap 5 in diagnostic, got {err:?}"
    );
}

#[tokio::test]
async fn run_overrides_max_iterations_cannot_raise_compile_time_cap() {
    // Compile-time cap is 5; per-call override of 100 must clamp
    // back to 5 (compile-time cap is authoritative). The self-loop
    // should trip after 5 iterations regardless of the override.
    let graph = StateGraph::<Counter>::new()
        .add_node("loop", add_one("loop"))
        .add_node("sink", add_one("sink"))
        .add_edge("loop", "loop")
        .add_finish_point("sink")
        .set_entry_point("loop")
        .with_recursion_limit(5)
        .compile()
        .unwrap();
    let ctx = ExecutionContext::new()
        .add_extension(entelix_core::RunOverrides::new().with_max_iterations(100));
    let err = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidRequest(msg) if msg.contains("recursion limit (5)")),
        "compile-time cap must remain authoritative; got {err:?}"
    );
}

#[tokio::test]
async fn default_recursion_limit_is_25() {
    let graph: StateGraph<Counter> = StateGraph::new();
    // No public accessor on the builder, but `compile` would carry it; we
    // verify the constant directly.
    assert_eq!(DEFAULT_RECURSION_LIMIT, 25);
    let _ = graph; // silence unused — we only inspected the constant.
}

#[tokio::test]
async fn cancelled_token_short_circuits_invoke() {
    let graph = StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .add_node("b", add_one("b"))
        .add_edge("a", "b")
        .set_entry_point("a")
        .add_finish_point("b")
        .compile()
        .unwrap();

    let ctx = ExecutionContext::new();
    ctx.cancellation().cancel();
    let err = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled));
}

// ── compile-time validation ────────────────────────────────────────────────

#[test]
fn compile_without_entry_point_fails() {
    let err = StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .add_finish_point("a")
        .compile()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

#[test]
fn compile_with_unknown_entry_point_fails() {
    let err = StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .set_entry_point("ghost")
        .add_finish_point("a")
        .compile()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

#[test]
fn compile_without_finish_points_fails() {
    let err = StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .set_entry_point("a")
        .compile()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

#[test]
fn compile_with_dangling_edge_fails() {
    let err = StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .add_edge("a", "ghost")
        .set_entry_point("a")
        .add_finish_point("a")
        .compile()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

#[test]
fn compile_with_orphan_node_fails() {
    // 'b' has no outgoing edge and is not a finish point.
    let err = StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .add_node("b", add_one("b"))
        .add_edge("a", "b")
        .set_entry_point("a")
        .add_finish_point("a")
        .compile()
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

// ── composition ────────────────────────────────────────────────────────────

#[tokio::test]
async fn compiled_graph_pipes_into_runnable_chain() -> Result<()> {
    use entelix_runnable::RunnableExt;

    let graph = StateGraph::<i32>::new()
        .add_node(
            "double",
            RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) }),
        )
        .set_entry_point("double")
        .add_finish_point("double")
        .compile()?;

    let stringify = RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(format!("{x}")) });
    let chain = graph.pipe(stringify);

    let out = chain.invoke(21, &ExecutionContext::new()).await?;
    assert_eq!(out, "42");
    Ok(())
}

#[tokio::test]
async fn compiled_graph_can_be_a_node_inside_another_graph() -> Result<()> {
    // Inner graph: doubles its input.
    let inner = StateGraph::<i32>::new()
        .add_node(
            "double",
            RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) }),
        )
        .set_entry_point("double")
        .add_finish_point("double")
        .compile()?;

    // Outer graph uses the inner CompiledGraph as one of its nodes.
    let outer = StateGraph::<i32>::new()
        .add_node("inner", inner)
        .add_node(
            "plus_one",
            RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x + 1) }),
        )
        .add_edge("inner", "plus_one")
        .set_entry_point("inner")
        .add_finish_point("plus_one")
        .compile()?;

    let out = outer.invoke(10, &ExecutionContext::new()).await?;
    assert_eq!(out, 21); // (10 * 2) + 1
    Ok(())
}
