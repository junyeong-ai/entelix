//! Integration tests for the four pre-built agent recipes. All
//! deterministic, no LLM dependency — uses a `MockModel` that replays
//! scripted replies and a `MockTool` that doubles its input.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound,
    clippy::match_wildcard_for_single_variants
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use entelix_agents::{
    AgentEntry, ChatState, ReActAgentBuilder, ReActState, Subagent, SubagentTool,
    SupervisorDecision, SupervisorState, create_chat_agent, create_react_agent,
    create_supervisor_agent, team_from_supervisor,
};
use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::tools::{Tool, ToolMetadata};
use entelix_core::{AgentContext, ExecutionContext, Result, ToolRegistry};
use entelix_runnable::{Runnable, RunnableLambda};
use std::sync::Mutex;

/// Replays a pre-canned script of assistant replies one per invocation.
struct MockModel {
    script: Mutex<Vec<Message>>,
    calls: Arc<AtomicUsize>,
}

impl MockModel {
    fn new(script: Vec<Message>) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                script: Mutex::new(script),
                calls: calls.clone(),
            },
            calls,
        )
    }
}

#[async_trait]
impl Runnable<Vec<Message>, Message> for MockModel {
    async fn invoke(&self, _input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut script = self.script.lock().unwrap();
        if script.is_empty() {
            return Err(entelix_core::Error::invalid_request(
                "MockModel: script exhausted",
            ));
        }
        Ok(script.remove(0))
    }
}

/// Tool that returns `{"doubled": input * 2}` regardless of payload.
struct DoubleTool {
    metadata: ToolMetadata,
}

impl DoubleTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "double",
                "Doubles the integer in `n`.",
                serde_json::json!({
                    "type": "object",
                    "properties": { "n": { "type": "number" } }
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for DoubleTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &AgentContext<()>,
    ) -> Result<serde_json::Value> {
        let n = input
            .get("n")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        Ok(serde_json::json!({ "doubled": n * 2 }))
    }
}

fn assistant_text(s: &str) -> Message {
    Message::assistant(s)
}

fn assistant_tool_call(id: &str, name: &str, input: serde_json::Value) -> Message {
    Message::new(
        Role::Assistant,
        vec![ContentPart::ToolUse {
            id: id.to_owned(),
            name: name.to_owned(),
            input,
        }],
    )
}

#[tokio::test]
async fn chat_agent_appends_one_assistant_reply() -> Result<()> {
    let (model, calls) = MockModel::new(vec![assistant_text("hi back")]);
    let graph = create_chat_agent(model, "Be terse.")?;
    let final_state = graph
        .invoke(ChatState::from_user("hi"), &ExecutionContext::new())
        .await?;
    assert_eq!(final_state.messages.len(), 2);
    assert_eq!(final_state.messages[1].role, Role::Assistant);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn react_agent_builder_applies_custom_name() -> Result<()> {
    let (model, _) = MockModel::new(vec![assistant_text("ok")]);
    let agent = ReActAgentBuilder::new(model, ToolRegistry::new())
        .with_name("research")
        .build()?;
    assert_eq!(agent.name(), "research");
    Ok(())
}

#[tokio::test]
async fn react_agent_dispatches_tool_then_finalizes() -> Result<()> {
    let (model, _) = MockModel::new(vec![
        assistant_tool_call("call-1", "double", serde_json::json!({"n": 21})),
        assistant_text("Result: 42"),
    ]);
    let tools = ToolRegistry::new().register(Arc::new(DoubleTool::new()))?;
    let graph = create_react_agent(model, tools)?;

    let final_state = graph
        .invoke(
            ReActState::from_user("double 21 please"),
            &ExecutionContext::new(),
        )
        .await?;
    // user → assistant(tool_use) → tool(tool_result) → assistant(text)
    assert_eq!(final_state.messages.len(), 4);
    assert_eq!(final_state.messages[2].role, Role::Tool);
    let last = final_state.messages.last().unwrap();
    assert_eq!(last.role, Role::Assistant);
    assert_eq!(final_state.steps, 2);
    Ok(())
}

#[tokio::test]
async fn react_agent_unknown_tool_marks_result_as_error() -> Result<()> {
    let (model, _) = MockModel::new(vec![
        assistant_tool_call("call-1", "ghost", serde_json::json!({})),
        assistant_text("done"),
    ]);
    let tools = ToolRegistry::new().register(Arc::new(DoubleTool::new()))?;
    let graph = create_react_agent(model, tools)?;
    let final_state = graph
        .invoke(
            ReActState::from_user("call ghost"),
            &ExecutionContext::new(),
        )
        .await?;
    let tool_msg = &final_state.messages[2];
    let part = tool_msg.content.first().unwrap();
    if let ContentPart::ToolResult {
        is_error, content, ..
    } = part
    {
        assert!(is_error);
        match content {
            entelix_core::ir::ToolResultContent::Text(t) => assert!(t.contains("unknown")),
            other => panic!("expected Text content, got {other:?}"),
        }
    } else {
        panic!("expected ToolResult");
    }
    Ok(())
}

#[tokio::test]
async fn supervisor_routes_two_agents_then_finishes() -> Result<()> {
    // Two named sub-agents — each just appends a hard-coded reply.
    let researcher = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("research result"))
    });
    let writer = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("writeup"))
    });

    // Router: first call → "research", second → "write", third → finish.
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_inner = counter.clone();
    let router = RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = counter_inner.clone();
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            let decision = match n {
                0 => SupervisorDecision::agent("research"),
                1 => SupervisorDecision::agent("write"),
                _ => SupervisorDecision::Finish,
            };
            Ok::<_, _>(decision)
        }
    });

    let graph = create_supervisor_agent(
        router,
        vec![
            AgentEntry::new("research", researcher),
            AgentEntry::new("write", writer),
        ],
    )?;

    let final_state = graph
        .invoke(SupervisorState::from_user("plan"), &ExecutionContext::new())
        .await?;
    // user + research reply + writer reply = 3 messages
    assert_eq!(final_state.messages.len(), 3);
    assert_eq!(final_state.last_speaker.as_deref(), Some("write"));
    Ok(())
}

#[tokio::test]
async fn hierarchical_routes_via_team_supervisors() -> Result<()> {
    // Build a research team supervisor (single agent, finishes after one
    // turn) and adapt it to the chat shape.
    let researcher = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("team-research"))
    });
    let team_router_counter = Arc::new(AtomicUsize::new(0));
    let team_router_inner = team_router_counter.clone();
    let team_router = RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = team_router_inner.clone();
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok::<_, _>(if n == 0 {
                SupervisorDecision::agent("researcher")
            } else {
                SupervisorDecision::Finish
            })
        }
    });
    let team_graph =
        create_supervisor_agent(team_router, vec![AgentEntry::new("researcher", researcher)])?;
    let team_runnable = team_from_supervisor(team_graph);

    // Top-level router picks the team once, then finishes.
    let top_counter = Arc::new(AtomicUsize::new(0));
    let top_counter_inner = top_counter.clone();
    let top_router = RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = top_counter_inner.clone();
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok::<_, _>(if n == 0 {
                SupervisorDecision::agent("research-team")
            } else {
                SupervisorDecision::Finish
            })
        }
    });

    let graph = create_supervisor_agent(
        top_router,
        vec![AgentEntry::new("research-team", team_runnable)],
    )?;

    let final_state = graph
        .invoke(SupervisorState::from_user("plan"), &ExecutionContext::new())
        .await?;
    assert_eq!(final_state.messages.len(), 2);
    assert_eq!(final_state.messages[1].role, Role::Assistant);
    Ok(())
}

#[tokio::test]
async fn subagent_filters_tool_set() -> Result<()> {
    struct OtherTool {
        metadata: ToolMetadata,
    }
    impl OtherTool {
        fn new() -> Self {
            Self {
                metadata: ToolMetadata::function("other", "noop", serde_json::json!({})),
            }
        }
    }
    #[async_trait]
    impl Tool for OtherTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _ctx: &AgentContext<()>,
        ) -> Result<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }
    let parent_registry = ToolRegistry::new()
        .register(Arc::new(DoubleTool::new()) as Arc<dyn Tool>)?
        .register(Arc::new(OtherTool::new()) as Arc<dyn Tool>)?;

    let (model, _) = MockModel::new(vec![assistant_text("done")]);
    let sub = Subagent::builder(model, &parent_registry, "test_subagent", "test description")
        .restrict_to(&["double"])
        .build()?;
    assert_eq!(sub.tool_count(), 1);
    assert_eq!(sub.tool_names(), vec!["double"]);
    Ok(())
}

#[tokio::test]
async fn react_agent_builder_recursion_limit_overrides_graph_default() -> Result<()> {
    use entelix_core::tools::ToolMetadata;
    use std::sync::Arc;

    // Tool that always emits another tool_use, so the planner loops
    // forever unless the recursion limit cuts it off.
    struct LoopTool {
        metadata: ToolMetadata,
    }
    impl LoopTool {
        fn new() -> Self {
            Self {
                metadata: ToolMetadata::function("loop", "loops", serde_json::json!({})),
            }
        }
    }
    #[async_trait]
    impl Tool for LoopTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _ctx: &AgentContext<()>,
        ) -> Result<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }

    // Model script that always emits a fresh tool_use, so the loop
    // never naturally terminates.
    let template = Message::new(
        Role::Assistant,
        vec![ContentPart::ToolUse {
            id: "tu_1".into(),
            name: "loop".into(),
            input: serde_json::json!({}),
        }],
    );
    let script = std::iter::repeat_n(template, 200).collect::<Vec<_>>();
    let (model, _) = MockModel::new(script);

    let registry = ToolRegistry::new().register(Arc::new(LoopTool::new()) as Arc<dyn Tool>)?;
    let agent = ReActAgentBuilder::new(model, registry)
        .with_recursion_limit(3)
        .build()?;
    let err = agent
        .execute(ReActState::from_user("loop"), &ExecutionContext::new())
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("recursion") || msg.to_lowercase().contains("limit"),
        "expected recursion/limit error, got: {msg}"
    );
    Ok(())
}

#[tokio::test]
async fn subagent_restrict_to_rejects_unknown_tool_name() -> Result<()> {
    struct OnlyDouble {
        metadata: ToolMetadata,
    }
    impl OnlyDouble {
        fn new() -> Self {
            Self {
                metadata: ToolMetadata::function("double", "noop", serde_json::json!({})),
            }
        }
    }
    #[async_trait]
    impl Tool for OnlyDouble {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _ctx: &AgentContext<()>,
        ) -> Result<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }
    let parent_registry =
        ToolRegistry::new().register(Arc::new(OnlyDouble::new()) as Arc<dyn Tool>)?;
    let (model, _) = MockModel::new(vec![assistant_text("noop")]);
    let err = Subagent::builder(model, &parent_registry, "test_subagent", "test description")
        .restrict_to(&["double", "ghost-tool"])
        .build()
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ghost-tool"),
        "error must name the missing tool, got: {msg}"
    );
    Ok(())
}

#[tokio::test]
async fn subagent_into_tool_dispatches_full_react_loop() -> Result<()> {
    // Sub-agent's mock model returns one terminal assistant message.
    let (model, calls) = MockModel::new(vec![assistant_text("task done")]);
    let parent_registry =
        ToolRegistry::new().register(Arc::new(DoubleTool::new()) as Arc<dyn Tool>)?;
    let sub = Subagent::builder(
        model,
        &parent_registry,
        "research_team",
        "Dispatch the research sub-agent to handle the supplied task.",
    )
    .restrict_to(&["double"])
    .build()?;
    let tool: SubagentTool = sub.into_tool()?;
    // Metadata reads back as we set it.
    assert_eq!(tool.metadata().name, "research_team");
    assert!(tool.metadata().description.contains("research sub-agent"));
    assert_eq!(
        tool.metadata().effect,
        entelix_core::tools::ToolEffect::Mutating
    );
    // Dispatch through the Tool trait — same path the parent's LLM
    // would walk.
    let out = tool
        .execute(
            serde_json::json!({"task": "investigate widget bookmarks"}),
            &AgentContext::default(),
        )
        .await?;
    assert_eq!(out["output"], "task done");
    // The inner mock model was invoked exactly once.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn subagent_tool_rejects_input_without_task_field() -> Result<()> {
    let (model, _) = MockModel::new(vec![assistant_text("noop")]);
    let parent_registry = ToolRegistry::new();
    let sub = Subagent::builder(model, &parent_registry, "agent_x", "x")
        .restrict_to(&[])
        .build()?;
    let tool: SubagentTool = sub.into_tool()?;
    let err = tool
        .execute(serde_json::json!({}), &AgentContext::default())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("'task' field"));
    Ok(())
}

#[tokio::test]
async fn subagent_tool_with_effect_overrides_default() -> Result<()> {
    let (model, _) = MockModel::new(vec![assistant_text("ok")]);
    let parent_registry = ToolRegistry::new();
    let sub = Subagent::builder(model, &parent_registry, "read_only_agent", "x")
        .restrict_to(&[])
        .build()?;
    let tool = sub
        .into_tool()?
        .with_effect(entelix_core::tools::ToolEffect::ReadOnly);
    assert_eq!(
        tool.metadata().effect,
        entelix_core::tools::ToolEffect::ReadOnly
    );
    Ok(())
}

#[tokio::test]
async fn handoff_payload_injects_into_next_agent_messages() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Capture every message the receiver sees so we can assert the
    // handoff payload landed as the leading `system` content.
    let receiver_seen: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));
    let receiver_seen_inner = receiver_seen.clone();
    let receiver = RunnableLambda::new(move |msgs: Vec<Message>, _ctx| {
        let seen = receiver_seen_inner.clone();
        async move {
            seen.lock().unwrap().extend(msgs);
            Ok::<_, _>(Message::assistant("ack"))
        }
    });

    // First turn: supervisor hands off to `receiver` with a typed
    // payload. Second turn: supervisor finishes.
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_inner = counter.clone();
    let router = RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = counter_inner.clone();
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok::<_, _>(if n == 0 {
                SupervisorDecision::handoff(
                    "receiver",
                    serde_json::json!({"summary": "3 sources", "confidence": 0.92}),
                )
            } else {
                SupervisorDecision::Finish
            })
        }
    });

    let graph = create_supervisor_agent(router, vec![AgentEntry::new("receiver", receiver)])?;
    let final_state = graph
        .invoke(SupervisorState::from_user("plan"), &ExecutionContext::new())
        .await?;
    assert_eq!(final_state.last_speaker.as_deref(), Some("receiver"));

    let seen = receiver_seen.lock().unwrap().clone();
    let system_msgs: Vec<&Message> = seen
        .iter()
        .filter(|m| matches!(m.role, Role::System))
        .collect();
    assert_eq!(
        system_msgs.len(),
        1,
        "receiver should see exactly one handoff system message"
    );
    let body: String = system_msgs[0]
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(body.starts_with("Handoff payload:"), "got: {body}");
    assert!(body.contains("\"summary\""), "got: {body}");
    assert!(body.contains("\"3 sources\""), "got: {body}");
    assert!(body.contains("\"confidence\""), "got: {body}");
    Ok(())
}

#[tokio::test]
async fn handoff_to_unknown_agent_finishes() -> Result<()> {
    let receiver = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(Message::assistant("never"))
    });
    let router = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(SupervisorDecision::handoff(
            "missing-agent",
            serde_json::json!({"x": 1}),
        ))
    });
    let graph = create_supervisor_agent(router, vec![AgentEntry::new("receiver", receiver)])?;
    let final_state = graph
        .invoke(SupervisorState::from_user("plan"), &ExecutionContext::new())
        .await?;
    // Routes to FINISH like an unknown `Agent(...)` — graph terminates
    // without dispatching to the missing receiver.
    assert!(final_state.last_speaker.is_none());
    Ok(())
}
