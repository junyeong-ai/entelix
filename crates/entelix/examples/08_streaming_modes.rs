//! `08_streaming_modes` — same compiled graph, streamed under all five
//! `StreamMode`s.
//!
//! Build: `cargo build --example 08_streaming_modes -p entelix`
//! Run:   `cargo run   --example 08_streaming_modes -p entelix`
//!
//! Demonstrates:
//! - `Runnable::stream(input, mode, ctx) → BoxStream<...>`
//! - `StreamMode::{Values, Updates, Messages, Debug, Events}`
//! - `StreamChunk<O>` chunk shapes
//!
//! Deterministic, no LLM dependency — runs in CI.

#![allow(clippy::print_stdout)]

use entelix::{
    CompiledGraph, DebugEvent, ExecutionContext, Result, Runnable, RunnableEvent, RunnableLambda,
    StateGraph, StreamChunk, StreamMode,
};
use futures::StreamExt;

#[derive(Clone, Debug)]
struct PipelineState {
    counter: i32,
    trace: Vec<String>,
}

fn build_pipeline() -> Result<CompiledGraph<PipelineState>> {
    fn step(label: &'static str) -> RunnableLambda<PipelineState, PipelineState> {
        RunnableLambda::new(move |mut s: PipelineState, _ctx| async move {
            s.counter += 1;
            s.trace.push(label.to_owned());
            Ok::<_, _>(s)
        })
    }

    StateGraph::<PipelineState>::new()
        .add_node("ingest", step("ingest"))
        .add_node("transform", step("transform"))
        .add_node("emit", step("emit"))
        .add_edge("ingest", "transform")
        .add_edge("transform", "emit")
        .set_entry_point("ingest")
        .add_finish_point("emit")
        .compile()
}

async fn print_mode(
    graph: &CompiledGraph<PipelineState>,
    mode: StreamMode,
    label: &str,
) -> Result<()> {
    println!("\n── StreamMode::{label} ─────────────────────────────");
    let initial = PipelineState {
        counter: 0,
        trace: Vec::new(),
    };
    let stream = graph
        .stream(initial, mode, &ExecutionContext::new())
        .await?;
    let chunks: Vec<_> = stream.collect::<Vec<_>>().await;
    for chunk in chunks {
        match chunk? {
            StreamChunk::Value(s) => {
                println!("  Value: counter={} trace={:?}", s.counter, s.trace);
            }
            StreamChunk::Update { node, value } => {
                println!(
                    "  Update[{node}]: counter={} trace={:?}",
                    value.counter, value.trace
                );
            }
            StreamChunk::Message(delta) => {
                println!("  Message: {delta:?}");
            }
            StreamChunk::Debug(DebugEvent::NodeStart { node, step }) => {
                println!("  Debug: > start  step={step:>2} node={node}");
            }
            StreamChunk::Debug(DebugEvent::NodeEnd { node, step }) => {
                println!("  Debug: < end    step={step:>2} node={node}");
            }
            StreamChunk::Debug(DebugEvent::Final) => {
                println!("  Debug: ! final");
            }
            StreamChunk::Event(RunnableEvent::Started { name }) => {
                println!("  Event: started {name}");
            }
            StreamChunk::Event(RunnableEvent::Finished { name, ok }) => {
                println!("  Event: finished {name} ok={ok}");
            }
            _ => {}
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let graph = build_pipeline()?;
    print_mode(&graph, StreamMode::Values, "Values").await?;
    print_mode(&graph, StreamMode::Updates, "Updates").await?;
    print_mode(&graph, StreamMode::Messages, "Messages").await?;
    print_mode(&graph, StreamMode::Debug, "Debug").await?;
    print_mode(&graph, StreamMode::Events, "Events").await?;
    Ok(())
}
