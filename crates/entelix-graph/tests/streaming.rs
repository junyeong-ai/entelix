//! `CompiledGraph<S>` streaming behaviour tests.
//!
//! Validates that `Runnable::stream` on a `CompiledGraph` emits the expected
//! chunks per `StreamMode` while honouring `recursion_limit` and
//! `ExecutionContext::cancellation`.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::{ExecutionContext, Result};
use entelix_graph::StateGraph;
use entelix_runnable::stream::{DebugEvent, RunnableEvent, StreamChunk, StreamMode};
use entelix_runnable::{Runnable, RunnableLambda};
use futures::StreamExt;

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

fn build_three_step() -> Result<entelix_graph::CompiledGraph<Counter>> {
    StateGraph::<Counter>::new()
        .add_node("a", add_one("a"))
        .add_node("b", add_one("b"))
        .add_node("c", add_one("c"))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_entry_point("a")
        .add_finish_point("c")
        .compile()
}

const fn empty_counter() -> Counter {
    Counter {
        n: 0,
        trail: vec![],
    }
}

#[tokio::test]
async fn values_mode_emits_one_chunk_per_node() -> Result<()> {
    let graph = build_three_step()?;
    let ctx = ExecutionContext::new();
    let stream = graph
        .stream(empty_counter(), StreamMode::Values, &ctx)
        .await?;
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(chunks.len(), 3);
    let counts: Vec<i32> = chunks.iter().map(|c| c.output().unwrap().n).collect();
    assert_eq!(counts, vec![1, 2, 3]);
    Ok(())
}

#[tokio::test]
async fn updates_mode_tags_each_chunk_with_node_name() -> Result<()> {
    let graph = build_three_step()?;
    let ctx = ExecutionContext::new();
    let stream = graph
        .stream(empty_counter(), StreamMode::Updates, &ctx)
        .await?;
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(chunks.len(), 3);
    let nodes: Vec<&str> = chunks
        .iter()
        .map(|c| match c {
            StreamChunk::Update { node, .. } => node.as_str(),
            _ => panic!("expected Update"),
        })
        .collect();
    assert_eq!(nodes, vec!["a", "b", "c"]);
    Ok(())
}

#[tokio::test]
async fn debug_mode_emits_start_end_per_node_plus_final() -> Result<()> {
    let graph = build_three_step()?;
    let ctx = ExecutionContext::new();
    let stream = graph
        .stream(empty_counter(), StreamMode::Debug, &ctx)
        .await?;
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    // 3 nodes × (NodeStart + NodeEnd) + Final = 7
    assert_eq!(chunks.len(), 7);
    assert!(matches!(
        chunks[0],
        StreamChunk::Debug(DebugEvent::NodeStart { step: 1, .. })
    ));
    assert!(matches!(
        chunks[1],
        StreamChunk::Debug(DebugEvent::NodeEnd { step: 1, .. })
    ));
    assert!(matches!(chunks[6], StreamChunk::Debug(DebugEvent::Final)));
    Ok(())
}

#[tokio::test]
async fn events_mode_brackets_with_started_finished() -> Result<()> {
    let graph = build_three_step()?;
    let ctx = ExecutionContext::new();
    let stream = graph
        .stream(empty_counter(), StreamMode::Events, &ctx)
        .await?;
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(chunks.len(), 2);
    assert!(matches!(
        chunks[0],
        StreamChunk::Event(RunnableEvent::Started { .. })
    ));
    assert!(matches!(
        chunks[1],
        StreamChunk::Event(RunnableEvent::Finished { ok: true, .. })
    ));
    Ok(())
}

#[tokio::test]
async fn messages_mode_falls_back_to_final_value() -> Result<()> {
    let graph = build_three_step()?;
    let ctx = ExecutionContext::new();
    let stream = graph
        .stream(empty_counter(), StreamMode::Messages, &ctx)
        .await?;
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    // No node-step chunks; only the terminal Value carrier.
    assert_eq!(chunks.len(), 1);
    let final_state = chunks[0].output().unwrap();
    assert_eq!(final_state.n, 3);
    Ok(())
}

#[tokio::test]
async fn cancellation_emits_cancelled_error_chunk() -> Result<()> {
    let graph = StateGraph::<Counter>::new()
        .add_node("only", add_one("only"))
        .set_entry_point("only")
        .add_finish_point("only")
        .compile()?;
    let ctx = ExecutionContext::new();
    ctx.cancellation().cancel();
    let mut stream = graph
        .stream(empty_counter(), StreamMode::Values, &ctx)
        .await?;
    let first = stream.next().await.unwrap();
    assert!(matches!(first, Err(entelix_core::Error::Cancelled)));
    Ok(())
}

#[tokio::test]
async fn recursion_limit_breaks_streaming_loop() -> Result<()> {
    let graph = StateGraph::<Counter>::new()
        .add_node("loop", add_one("loop"))
        .add_node("sink", add_one("sink"))
        .add_edge("loop", "loop") // self-loop, sink never reached
        .set_entry_point("loop")
        .add_finish_point("sink")
        .with_recursion_limit(4)
        .compile()?;
    let ctx = ExecutionContext::new();
    let stream = graph
        .stream(empty_counter(), StreamMode::Values, &ctx)
        .await?;
    let chunks: Vec<_> = stream.collect::<Vec<_>>().await;
    // 4 successful Value chunks then an InvalidRequest (recursion-limit).
    let oks = chunks.iter().filter(|r| r.is_ok()).count();
    let errs = chunks.iter().filter(|r| r.is_err()).count();
    assert_eq!(oks, 4);
    assert_eq!(errs, 1);
    Ok(())
}
