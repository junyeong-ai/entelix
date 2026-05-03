//! Regression tests for invariant #18 producers wired in
//! `entelix-agents`: `SubagentTool::execute` emits
//! `record_sub_agent_invoked`, `build_supervisor_graph`'s router
//! node emits `record_agent_handoff`. Both gate on
//! `ctx.audit_sink()` so a context without a sink stays a no-op.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use parking_lot::Mutex;

use entelix_agents::{
    AgentEntry, Subagent, SubagentTool, SupervisorDecision, SupervisorState,
    create_supervisor_agent,
};
use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::tools::Tool;
use entelix_core::{AuditSink, AuditSinkHandle, ExecutionContext, Result, ToolRegistry};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Default)]
struct RecordingAuditSink {
    sub_agents: Mutex<Vec<(String, String)>>,
    handoffs: Mutex<Vec<(Option<String>, String)>>,
    resumes: Mutex<Vec<String>>,
}

impl AuditSink for RecordingAuditSink {
    fn record_sub_agent_invoked(&self, agent_id: &str, sub_thread_id: &str) {
        self.sub_agents
            .lock()
            .push((agent_id.to_owned(), sub_thread_id.to_owned()));
    }
    fn record_agent_handoff(&self, from: Option<&str>, to: &str) {
        self.handoffs
            .lock()
            .push((from.map(str::to_owned), to.to_owned()));
    }
    fn record_resumed(&self, from_checkpoint: &str) {
        self.resumes.lock().push(from_checkpoint.to_owned());
    }
    fn record_memory_recall(&self, _tier: &str, _namespace_key: &str, _hits: usize) {}
}

fn assistant_text(text: &str) -> Message {
    Message::new(
        Role::Assistant,
        vec![ContentPart::Text {
            text: text.to_owned(),
            cache_control: None,
        }],
    )
}

fn ctx_with_sink(sink: Arc<RecordingAuditSink>) -> ExecutionContext {
    ExecutionContext::new().with_audit_sink(AuditSinkHandle::new(sink as Arc<dyn AuditSink>))
}

#[tokio::test]
async fn subagent_tool_emits_sub_agent_invoked_with_fresh_thread_id() -> Result<()> {
    let model = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("done"))
    });
    let sub = Subagent::from_whitelist(model, &ToolRegistry::new(), &[])?;
    let tool: SubagentTool = sub.into_tool("research_team", "spec")?;

    let sink = Arc::new(RecordingAuditSink::default());
    let ctx = ctx_with_sink(Arc::clone(&sink));
    tool.execute(serde_json::json!({"task": "investigate"}), &ctx)
        .await?;

    let recorded: Vec<_> = sink.sub_agents.lock().clone();
    assert_eq!(recorded.len(), 1);
    let (agent_id, sub_thread_id) = &recorded[0];
    assert_eq!(agent_id, "research_team");
    // UUID v7 hyphenated representation is 36 chars; serves as a
    // structural sanity check without pinning a specific id.
    assert_eq!(sub_thread_id.len(), 36);
    Ok(())
}

#[tokio::test]
async fn subagent_tool_without_sink_stays_silent() -> Result<()> {
    let model = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("done"))
    });
    let sub = Subagent::from_whitelist(model, &ToolRegistry::new(), &[])?;
    let tool: SubagentTool = sub.into_tool("agent_x", "spec")?;

    let out = tool
        .execute(serde_json::json!({"task": "go"}), &ExecutionContext::new())
        .await?;
    assert_eq!(out["output"], "done");
    Ok(())
}

#[tokio::test]
async fn supervisor_emits_handoff_per_routed_turn_with_last_speaker() -> Result<()> {
    let researcher = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("research"))
    });
    let writer = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("write"))
    });

    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter_inner = counter.clone();
    let router = RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = counter_inner.clone();
        async move {
            let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok::<_, _>(match n {
                0 => SupervisorDecision::agent("research"),
                1 => SupervisorDecision::agent("write"),
                _ => SupervisorDecision::Finish,
            })
        }
    });
    let agent = create_supervisor_agent(
        router,
        vec![
            AgentEntry::new("research", researcher),
            AgentEntry::new("write", writer),
        ],
    )?;

    let sink = Arc::new(RecordingAuditSink::default());
    let ctx = ctx_with_sink(Arc::clone(&sink));
    agent
        .invoke(SupervisorState::from_user("plan"), &ctx)
        .await?;

    let recorded: Vec<_> = sink.handoffs.lock().clone();
    // Two routed turns (research, write); the third turn returned
    // Finish and produced no handoff entry.
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[0], (None, "research".to_owned()));
    assert_eq!(
        recorded[1],
        (Some("research".to_owned()), "write".to_owned())
    );
    Ok(())
}

#[tokio::test]
async fn supervisor_finish_decision_skips_handoff_emit() -> Result<()> {
    let solo = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(assistant_text("done"))
    });
    let router = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(SupervisorDecision::Finish)
    });
    let agent = create_supervisor_agent(router, vec![AgentEntry::new("solo", solo)])?;

    let sink = Arc::new(RecordingAuditSink::default());
    let ctx = ctx_with_sink(Arc::clone(&sink));
    agent
        .invoke(SupervisorState::from_user("noop"), &ctx)
        .await?;

    assert!(sink.handoffs.lock().is_empty());
    Ok(())
}
