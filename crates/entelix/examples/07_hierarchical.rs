//! `07_hierarchical` — nested supervisor (research team + writing team).
//!
//! Build: `cargo build --example 07_hierarchical -p entelix`
//! Run:   `cargo run   --example 07_hierarchical -p entelix`
//!
//! Two team supervisors are themselves supervised by a top-level
//! router. All agents are deterministic mocks — runs in CI without an
//! LLM.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use entelix::ir::{Message, Role};
use entelix::{
    AgentEntry, ExecutionContext, Result, Runnable, RunnableLambda, SupervisorDecision,
    SupervisorState, create_hierarchical_agent, create_supervisor_agent, team_from_supervisor,
};

fn one_shot_router(allow: &'static str) -> RunnableLambda<Vec<Message>, SupervisorDecision> {
    let counter = Arc::new(AtomicUsize::new(0));
    RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = counter.clone();
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok::<_, _>(if n == 0 {
                SupervisorDecision::agent(allow)
            } else {
                SupervisorDecision::Finish
            })
        }
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── research team ──────────────────────────────────────────────
    let researcher = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(Message::assistant("research-team: 3 sources gathered."))
    });
    let research_team = create_supervisor_agent(
        one_shot_router("researcher"),
        vec![AgentEntry::new("researcher", researcher)],
    )?;

    // ── writing team ───────────────────────────────────────────────
    let writer = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(Message::assistant("writing-team: drafted paragraph."))
    });
    let writing_team = create_supervisor_agent(
        one_shot_router("writer"),
        vec![AgentEntry::new("writer", writer)],
    )?;

    // Top-level: research → write → finish.
    let top_counter = Arc::new(AtomicUsize::new(0));
    let top_counter_inner = top_counter.clone();
    let top_router = RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = top_counter_inner.clone();
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok::<_, _>(match n {
                0 => SupervisorDecision::agent("research-team"),
                1 => SupervisorDecision::agent("writing-team"),
                _ => SupervisorDecision::Finish,
            })
        }
    });

    let graph = create_hierarchical_agent(
        top_router,
        vec![
            AgentEntry::new("research-team", team_from_supervisor(research_team)),
            AgentEntry::new("writing-team", team_from_supervisor(writing_team)),
        ],
    )?;

    let final_state = graph
        .invoke(
            SupervisorState::from_user("Produce a short brief on photonics."),
            &ExecutionContext::new(),
        )
        .await?;

    println!("=== hierarchical conversation ===");
    for (i, m) in final_state.messages.iter().enumerate() {
        let body = m
            .content
            .iter()
            .filter_map(|p| match p {
                entelix::ir::ContentPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" / ");
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
            _ => "?",
        };
        println!("  [{i}] {role:10} | {body}");
    }
    Ok(())
}
