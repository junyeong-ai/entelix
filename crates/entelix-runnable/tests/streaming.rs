//! Streaming behaviour tests for `Runnable::stream`.
//!
//! Validates the default trait impl: a single `invoke` is wrapped into one
//! chunk per mode for non-graph runnables. Graph-level streaming is
//! covered in `entelix-graph`'s test surface.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::ExecutionContext;
use entelix_runnable::stream::{DebugEvent, RunnableEvent, StreamChunk, StreamMode};
use entelix_runnable::{Runnable, RunnableExt, RunnableLambda};
use futures::StreamExt;

fn lambda_double() -> RunnableLambda<i32, i32> {
    RunnableLambda::new(|x: i32, _ctx| async move { Ok::<_, _>(x * 2) })
}

#[tokio::test]
async fn default_values_emits_one_value_chunk() {
    let ctx = ExecutionContext::new();
    let lambda = lambda_double();
    let mut stream = lambda.stream(7, StreamMode::Values, &ctx).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert!(matches!(first, StreamChunk::Value(14)));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn default_updates_emits_one_update_chunk_with_name() {
    let ctx = ExecutionContext::new();
    let lambda = lambda_double();
    let mut stream = lambda.stream(5, StreamMode::Updates, &ctx).await.unwrap();
    let chunk = stream.next().await.unwrap().unwrap();
    match chunk {
        StreamChunk::Update { node, value } => {
            assert!(!node.is_empty());
            assert_eq!(value, 10);
        }
        other => panic!("expected Update chunk, got {other:?}"),
    }
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn default_messages_falls_back_to_value() {
    let ctx = ExecutionContext::new();
    let lambda = lambda_double();
    let mut stream = lambda.stream(3, StreamMode::Messages, &ctx).await.unwrap();
    let chunk = stream.next().await.unwrap().unwrap();
    assert!(matches!(chunk, StreamChunk::Value(6)));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn default_debug_emits_lifecycle_markers() {
    let ctx = ExecutionContext::new();
    let lambda = lambda_double();
    let stream = lambda.stream(2, StreamMode::Debug, &ctx).await.unwrap();
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(chunks.len(), 4);
    assert!(matches!(
        chunks[0],
        StreamChunk::Debug(DebugEvent::NodeStart { step: 1, .. })
    ));
    assert!(matches!(chunks[1], StreamChunk::Value(4)));
    assert!(matches!(
        chunks[2],
        StreamChunk::Debug(DebugEvent::NodeEnd { step: 1, .. })
    ));
    assert!(matches!(
        chunks[3],
        StreamChunk::Debug(DebugEvent::Final)
    ));
}

#[tokio::test]
async fn default_events_brackets_value_with_started_finished() {
    let ctx = ExecutionContext::new();
    let lambda = lambda_double();
    let stream = lambda.stream(1, StreamMode::Events, &ctx).await.unwrap();
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(chunks.len(), 3);
    assert!(matches!(
        chunks[0],
        StreamChunk::Event(RunnableEvent::Started { .. })
    ));
    assert!(matches!(chunks[1], StreamChunk::Value(2)));
    assert!(matches!(
        chunks[2],
        StreamChunk::Event(RunnableEvent::Finished { ok: true, .. })
    ));
}

#[tokio::test]
async fn ext_stream_with_matches_trait_method() {
    let ctx = ExecutionContext::new();
    let lambda = lambda_double();
    let mut stream = lambda
        .stream_with(4, StreamMode::Values, &ctx)
        .await
        .unwrap();
    let chunk = stream.next().await.unwrap().unwrap();
    assert!(matches!(chunk, StreamChunk::Value(8)));
}

#[tokio::test]
async fn output_accessor_extracts_carrier_payload() {
    let v: StreamChunk<i32> = StreamChunk::Value(42);
    assert_eq!(v.output(), Some(&42));
    assert_eq!(v.into_output(), Some(42));

    let u: StreamChunk<i32> = StreamChunk::Update {
        node: "n".to_owned(),
        value: 7,
    };
    assert_eq!(u.output(), Some(&7));

    let d: StreamChunk<i32> = StreamChunk::Debug(DebugEvent::Final);
    assert!(d.output().is_none());
    assert!(d.into_output().is_none());
}
