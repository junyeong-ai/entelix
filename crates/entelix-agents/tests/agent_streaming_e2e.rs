//! End-to-end agent streaming surface coverage. Exercises the
//! pieces a typical chat-streaming HTTP route binds together:
//!
//! - mock model + tool registered through `ToolRegistry`
//! - `AgentObserver` observing `on_complete` to write terminal
//!   turn state to an external side-channel after the agent finishes
//! - `agent.execute_stream(state, &ctx)` round-trips the canonical
//!   `AgentEvent` sequence an SSE renderer consumes
//!
//! Passing this test is the binary gate that the agent-stream
//! surface is wired correctly across `ToolRegistry` registration,
//! lifecycle observers, and the canonical event sequence.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use std::sync::Arc;

use async_trait::async_trait;
use entelix_agents::{Agent, AgentEvent, AgentObserver, CaptureSink, create_react_agent};
use entelix_core::AgentContext;
use entelix_core::ToolRegistry;
use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::tools::{Tool, ToolMetadata};
use entelix_runnable::{Runnable, RunnableLambda};
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::{Value, json};

/// Minimal mock model for the test: emits a tool-use on the first
/// turn, plain text on the second.
fn mock_react_model() -> impl Runnable<Vec<Message>, Message> + 'static {
    let calls = Arc::new(Mutex::new(0_u32));
    RunnableLambda::new(move |_messages: Vec<Message>, _ctx| {
        let calls = calls.clone();
        async move {
            let mut n = calls.lock();
            *n += 1;
            let turn = *n;
            drop(n);
            if turn == 1 {
                Ok::<_, _>(Message::new(
                    Role::Assistant,
                    vec![ContentPart::ToolUse {
                        id: "tool-1".into(),
                        name: "echo".into(),
                        input: json!({"value": 7}),
                        provider_echoes: Vec::new(),
                    }],
                ))
            } else {
                Ok(Message::assistant("done"))
            }
        }
    })
}

/// Trivial `Tool` impl that echoes its `value` input.
struct EchoTool {
    metadata: ToolMetadata,
}

impl EchoTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "echo",
                "echoes the input value",
                json!({"type": "object", "properties": {"value": {}}, "required": ["value"]}),
            ),
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
    async fn execute(&self, input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
        Ok(json!({"echoed": input["value"]}))
    }
}

/// Lifecycle observer that stores the terminal state on
/// `on_complete` so a downstream vector store can index the
/// conversation transcript.
struct CompletionRecorder {
    captured: Arc<Mutex<Option<entelix_agents::ReActState>>>,
}

#[async_trait]
impl AgentObserver<entelix_agents::ReActState> for CompletionRecorder {
    fn name(&self) -> &'static str {
        "completion-recorder"
    }
    async fn on_complete(
        &self,
        state: &entelix_agents::ReActState,
        _ctx: &ExecutionContext,
    ) -> Result<()> {
        *self.captured.lock() = Some(state.clone());
        Ok(())
    }
}

#[tokio::test]
async fn react_session_round_trips_canonical_event_sequence() {
    let tools = ToolRegistry::new()
        .register(Arc::new(EchoTool::new()))
        .unwrap();
    let model = mock_react_model();

    let agent = create_react_agent(model, tools).unwrap();

    // Attach a sink to mirror what an SSE driver downstream sees.
    let sink = CaptureSink::<entelix_agents::ReActState>::new();
    let agent = Agent::<entelix_agents::ReActState>::builder()
        .with_name("react")
        .with_runnable_arc(Arc::clone(agent.inner()))
        .add_sink(sink.clone())
        .build()
        .unwrap();

    // Drive the run end-to-end.
    let state = entelix_agents::ReActState::from_user("echo 7 please");
    let ctx = ExecutionContext::new();
    let mut stream = agent.execute_stream(state, &ctx);
    let mut received = Vec::new();
    while let Some(event) = stream.next().await {
        received.push(event.unwrap());
    }

    // Canonical sequence: Started → … → Complete(state).
    assert!(matches!(received[0], AgentEvent::Started { .. }));
    assert!(matches!(
        received.last(),
        Some(AgentEvent::Complete { state: _, .. })
    ));
    assert_eq!(
        received.len(),
        sink.len(),
        "caller stream and sink must observe identical event count"
    );

    // Verify the inner graph actually walked planner → tools →
    // planner → finish (4 steps, 4 messages: user + assistant
    // tool_use + tool result + assistant text).
    if let Some(AgentEvent::Complete {
        state: final_state, ..
    }) = received.last()
    {
        assert_eq!(final_state.messages.len(), 4);
        assert_eq!(final_state.steps, 2);
    }
}

#[tokio::test]
async fn completion_recorder_observes_terminal_state() {
    // Production observers receive the terminal `ReActState` via
    // `on_complete` — useful for indexing the full conversation
    // into a vector store after the agent finishes.
    let captured = Arc::new(Mutex::new(None));
    let observer = CompletionRecorder {
        captured: Arc::clone(&captured),
    };

    let tools = ToolRegistry::new()
        .register(Arc::new(EchoTool::new()))
        .unwrap();
    let recipe = create_react_agent(mock_react_model(), tools).unwrap();

    // Re-wrap with the observer attached.
    let agent = Agent::<entelix_agents::ReActState>::builder()
        .with_name("react-with-observer")
        .with_runnable_arc(Arc::clone(recipe.inner()))
        .with_observer(observer)
        .build()
        .unwrap();

    let final_state = agent
        .execute(
            entelix_agents::ReActState::from_user("echo 7"),
            &ExecutionContext::new(),
        )
        .await
        .unwrap()
        .into_state();

    // Full ReAct turn (planner → tools → planner → finish) →
    // 4 messages; observer received the same terminal state.
    assert_eq!(final_state.messages.len(), 4);
    let recorded = captured.lock().clone().expect("on_complete must fire");
    assert_eq!(recorded.messages.len(), final_state.messages.len());
}
