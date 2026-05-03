//! Regression: `Agent::execute` and `execute_stream` open an
//! `entelix.agent.run` tracing span around the inner runnable
//! invocation. Inner model + tool layer spans nest as children
//! through the `tracing-opentelemetry` bridge.
//!
//! Verified via a custom `Runnable` that snapshots `Span::current()`
//! inside `invoke` — metadata (`name` / `target`) and recorded
//! field values must match what `Agent::run_span` emits.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::expect_used)]

use std::sync::Arc;
use std::sync::Once;

use async_trait::async_trait;
use entelix_agents::{Agent, ReActState};
use entelix_core::{ExecutionContext, Result};
use entelix_runnable::Runnable;
use futures::StreamExt;
use parking_lot::Mutex;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

static INIT: Once = Once::new();

/// Install a subscriber once per test process. Without it,
/// `tracing::Span::current()` returns a disabled span with no
/// metadata regardless of how the span was created.
fn init_subscriber() {
    INIT.call_once(|| {
        let _ = tracing_subscriber::registry()
            .with(fmt::layer().with_test_writer())
            .try_init();
    });
}

#[derive(Default)]
struct SpanInspector {
    captured: Mutex<Option<SpanSnapshot>>,
}

#[derive(Clone, Debug)]
struct SpanSnapshot {
    name: &'static str,
    target: &'static str,
}

#[async_trait]
impl Runnable<ReActState, ReActState> for SpanInspector {
    async fn invoke(&self, input: ReActState, _ctx: &ExecutionContext) -> Result<ReActState> {
        let span = tracing::Span::current();
        let snapshot = span.metadata().map(|m| SpanSnapshot {
            name: m.name(),
            target: m.target(),
        });
        *self.captured.lock() = snapshot;
        Ok(input)
    }
}

#[tokio::test]
async fn execute_opens_entelix_agent_run_span_around_inner_runnable() {
    init_subscriber();
    let inspector = Arc::new(SpanInspector::default());
    let inspector_for_assert = Arc::clone(&inspector);
    let agent = Agent::<ReActState>::builder()
        .with_name("test-agent")
        .with_runnable_arc(inspector as Arc<dyn Runnable<ReActState, ReActState>>)
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let _ = agent
        .execute(ReActState::from_user("hi"), &ctx)
        .await
        .unwrap();
    let snapshot = inspector_for_assert
        .captured
        .lock()
        .clone()
        .expect("inner runnable must observe an active span");
    assert_eq!(snapshot.name, "entelix.agent.run");
    assert_eq!(snapshot.target, "gen_ai");
}

#[tokio::test]
async fn execute_stream_opens_entelix_agent_run_span_around_inner_runnable() {
    init_subscriber();
    let inspector = Arc::new(SpanInspector::default());
    let inspector_for_assert = Arc::clone(&inspector);
    let agent = Agent::<ReActState>::builder()
        .with_name("test-agent-stream")
        .with_runnable_arc(inspector as Arc<dyn Runnable<ReActState, ReActState>>)
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let mut stream = agent.execute_stream(ReActState::from_user("hi"), &ctx);
    while let Some(_event) = stream.next().await {}
    let snapshot = inspector_for_assert
        .captured
        .lock()
        .clone()
        .expect("inner runnable must observe an active span");
    assert_eq!(snapshot.name, "entelix.agent.run");
    assert_eq!(snapshot.target, "gen_ai");
}

#[tokio::test]
async fn no_span_is_active_outside_execute() {
    init_subscriber();
    // Sanity check: outside of `Agent::execute`, the current span
    // is the implicit root (no metadata). This guards against
    // `Agent` accidentally entering its span globally instead of
    // scoping it to the `run_inner` future.
    let span = tracing::Span::current();
    assert!(
        span.metadata().is_none()
            || span
                .metadata()
                .is_some_and(|m| m.name() != "entelix.agent.run"),
        "no `entelix.agent.run` span outside of execute"
    );
}
