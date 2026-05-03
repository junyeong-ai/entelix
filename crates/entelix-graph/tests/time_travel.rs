//! `Checkpointer::update_state` + `CompiledGraph::resume_from` time-travel
//! tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::sync::Arc;

use entelix_core::ThreadKey;
use entelix_core::{Error, ExecutionContext, Result};
use entelix_graph::{Checkpointer, Command, InMemoryCheckpointer, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Counter {
    n: i32,
    trail: Vec<&'static str>,
}

fn step(label: &'static str, delta: i32) -> RunnableLambda<Counter, Counter> {
    RunnableLambda::new(move |mut s: Counter, _ctx| async move {
        s.n += delta;
        s.trail.push(label);
        Ok::<_, _>(s)
    })
}

fn make_graph(cp: Arc<InMemoryCheckpointer<Counter>>) -> entelix_graph::CompiledGraph<Counter> {
    StateGraph::<Counter>::new()
        .add_node("a", step("a", 1))
        .add_node("b", step("b", 10))
        .add_node("c", step("c", 100))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_entry_point("a")
        .add_finish_point("c")
        .with_checkpointer(cp)
        .compile()
        .unwrap()
}

#[tokio::test]
async fn by_id_retrieves_checkpoint() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = make_graph(cp.clone());
    let ctx = ExecutionContext::new().with_thread_id("get-by-id");
    let _ = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await?;

    let key = ThreadKey::from_ctx(&ctx)?;
    let history = cp.history(&key, 100).await?;
    let middle = &history[1]; // second-most-recent — written after node "b"

    let fetched = cp
        .by_id(&key, &middle.id)
        .await?
        .expect("checkpoint with that id should exist");
    assert_eq!(fetched.id, middle.id);
    assert_eq!(fetched.state.n, middle.state.n);
    Ok(())
}

#[tokio::test]
async fn by_id_returns_none_for_unknown_id() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = make_graph(cp.clone());
    let ctx = ExecutionContext::new().with_thread_id("nf");
    let _ = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await?;
    let key = ThreadKey::from_ctx(&ctx)?;
    let bogus = entelix_graph::CheckpointId::new();
    let result = cp.by_id(&key, &bogus).await?;
    assert!(result.is_none());
    Ok(())
}

#[tokio::test]
async fn update_state_creates_branched_checkpoint() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = make_graph(cp.clone());
    let ctx = ExecutionContext::new().with_thread_id("branch");
    let _ = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await?;

    let key = ThreadKey::from_ctx(&ctx)?;

    // History before branching: 3 checkpoints.
    let pre = cp.history(&key, 100).await?;
    assert_eq!(pre.len(), 3);
    let parent = &pre[2]; // earliest — after node "a"

    // Branch: rewrite state at the parent point.
    let new_state = Counter {
        n: -999,
        trail: vec!["rewritten"],
    };
    let new_id = cp.update_state(&key, &parent.id, new_state.clone()).await?;

    // History grew by 1, parent_id and inherited next_node tracked.
    let post = cp.history(&key, 100).await?;
    assert_eq!(post.len(), 4);
    let branched = post.iter().find(|c| c.id == new_id).unwrap();
    assert_eq!(branched.parent_id.as_ref(), Some(&parent.id));
    assert_eq!(branched.next_node, parent.next_node);
    assert_eq!(branched.state, new_state);
    Ok(())
}

#[tokio::test]
async fn update_state_unknown_parent_returns_invalid_request() {
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let key = ThreadKey::new("default", "nothing");
    let bogus = entelix_graph::CheckpointId::new();
    let err = cp
        .update_state(
            &key,
            &bogus,
            Counter {
                n: 0,
                trail: vec![],
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[tokio::test]
async fn resume_from_historical_checkpoint_with_update() -> Result<()> {
    // Run once normally; then time-travel back to the post-"a" checkpoint
    // and inject a fresh state, verifying the rest of the graph runs from
    // the branched state.
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = make_graph(cp.clone());
    let ctx = ExecutionContext::new().with_thread_id("tt");
    let final_first = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await?;
    assert_eq!(final_first.n, 111); // 1 + 10 + 100

    let key = ThreadKey::from_ctx(&ctx)?;

    // Earliest checkpoint: after "a" ran.
    let history = cp.history(&key, 100).await?;
    let earliest = history.last().unwrap().clone();
    assert_eq!(earliest.next_node.as_deref(), Some("b"));

    // Resume from there with a state override of {n: 1000, trail: ["a"]}.
    let modified = Counter {
        n: 1000,
        trail: vec!["a"],
    };
    let resumed = graph
        .resume_from(&earliest.id, Command::Update(modified), &ctx)
        .await?;

    // From n=1000, "b" adds 10 → 1010, "c" adds 100 → 1110.
    assert_eq!(resumed.n, 1110);
    assert_eq!(resumed.trail, vec!["a", "b", "c"]);
    Ok(())
}

#[tokio::test]
async fn resume_from_unknown_checkpoint_returns_invalid_request() {
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = make_graph(cp.clone());
    let ctx = ExecutionContext::new().with_thread_id("tt2");

    let bogus = entelix_graph::CheckpointId::new();
    let err = graph
        .resume_from(&bogus, Command::Resume, &ctx)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[tokio::test]
async fn update_state_then_resume_from_creates_full_branch() -> Result<()> {
    // The full LangGraph time-travel idiom: combine update_state +
    // resume_from to fork the graph at an arbitrary historical point.
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = make_graph(cp.clone());
    let ctx = ExecutionContext::new().with_thread_id("tt3");
    let _ = graph
        .invoke(
            Counter {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await?;

    let key = ThreadKey::from_ctx(&ctx)?;

    // Earliest checkpoint (post-a).
    let earliest_id = cp.history(&key, 100).await?.last().unwrap().id.clone();

    // Branch: write a new checkpoint with rewritten state.
    let branch_id = cp
        .update_state(
            &key,
            &earliest_id,
            Counter {
                n: 5000,
                trail: vec!["a", "rewritten"],
            },
        )
        .await?;

    // Resume from the branch point.
    let result = graph.resume_from(&branch_id, Command::Resume, &ctx).await?;
    assert_eq!(result.n, 5000 + 10 + 100);
    assert_eq!(result.trail, vec!["a", "rewritten", "b", "c"]);
    Ok(())
}
