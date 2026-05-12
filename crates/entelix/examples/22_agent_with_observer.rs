//! `22_agent_with_observer` — headline `Agent::builder()` walkthrough
//! with a typed `AgentObserver` that prints the lifecycle as it
//! unfolds.
//!
//! Build: `cargo build --example 22_agent_with_observer -p entelix`
//! Run (hermetic — no network):
//!     `cargo run --example 22_agent_with_observer -p entelix`
//!
//! Demonstrates the canonical 5-line shape every operator reaches
//! for first:
//!
//! 1. Build any `Runnable<S, S>` (here: a deterministic counter).
//! 2. Wrap in `Agent::<S>::builder().with_runnable(.)`.
//! 3. Attach an `AgentObserver` for lifecycle visibility
//!    (`pre_turn`, `on_complete`) and a `CaptureSink` for
//!    `AgentEvent` history.
//! 4. Drive with `agent.execute(input, &ctx)` (sync drain) or
//!    `execute_stream(.)` (incremental events).
//! 5. Inspect the captured events + final state.
//!
//! No vendor codec wiring is needed for this shape — `Agent<S>` is
//! the agent-runtime envelope; the inner runnable can be a chat
//! model, a graph, or (as here) a hand-rolled lambda.

#![allow(clippy::print_stdout)] // example output goes to the terminal

use async_trait::async_trait;
use entelix::{
    Agent, AgentEvent, AgentObserver, CaptureSink, ExecutionContext, Result, RunnableLambda,
};

/// Observer that prints `pre_turn` / `on_complete` lifecycle marks.
struct PrintingObserver;

#[async_trait]
impl AgentObserver<u32> for PrintingObserver {
    async fn pre_turn(&self, state: &u32, _ctx: &ExecutionContext) -> Result<()> {
        println!("  pre_turn  | state={state}");
        Ok(())
    }

    async fn on_complete(&self, state: &u32, _ctx: &ExecutionContext) -> Result<()> {
        println!("  on_complete | state={state}");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // (1) Inner runnable — increments by 1, deterministic.
    let counter = RunnableLambda::new(|n: u32, _ctx: ExecutionContext| async move { Ok(n + 1) });

    // (2 + 3) Agent envelope with observer + capture sink.
    let sink = CaptureSink::<u32>::new();
    let agent = Agent::<u32>::builder()
        .with_name("counter-agent")
        .with_runnable(counter)
        .add_sink(sink.clone())
        .with_observer(PrintingObserver)
        .build()?;

    // (4) Execute — emits Started → Complete on the sink, observer
    //     fires pre_turn / on_complete around the inner invoke.
    println!("=== running ===");
    let result = agent.execute(41, &ExecutionContext::new()).await?;
    println!("final state = {}", result.state);

    // (5) Captured event history.
    println!("\n=== events ===");
    for event in sink.events() {
        match event {
            AgentEvent::Started {
                run_id,
                tenant_id,
                parent_run_id,
                agent,
            } => {
                let parent = parent_run_id
                    .as_deref()
                    .map_or_else(String::new, |p| format!(" parent={p}"));
                println!("Started   | run_id={run_id} tenant={tenant_id}{parent} agent={agent}");
            }
            AgentEvent::Complete {
                run_id,
                tenant_id,
                state,
                ..
            } => {
                println!("Complete  | run_id={run_id} tenant={tenant_id} state={state}");
            }
            AgentEvent::Failed {
                run_id,
                tenant_id,
                error,
                envelope,
            } => {
                println!(
                    "Failed    | run_id={run_id} tenant={tenant_id} wire={code} class={class} error={error}",
                    code = envelope.wire_code,
                    class = envelope.wire_class,
                );
            }
            other => println!("{other:?}"),
        }
    }
    Ok(())
}
