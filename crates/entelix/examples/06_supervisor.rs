//! `06_supervisor` — multi-agent supervisor with two named sub-agents.
//!
//! Build: `cargo build --example 06_supervisor -p entelix`
//! Run:   `cargo run   --example 06_supervisor -p entelix`
//!
//! The supervisor router decides per turn which sub-agent (researcher /
//! writer) handles the next message, or terminates the conversation.
//! Sub-agents and router are deterministic mocks — no LLM, runs in CI.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use entelix::ir::{Message, Role};
use entelix::{
    AgentEntry, ExecutionContext, Result, Runnable, RunnableLambda, SupervisorDecision,
    SupervisorState, create_supervisor_agent,
};

#[tokio::main]
async fn main() -> Result<()> {
    let researcher = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(Message::assistant(
            "researcher: gathered three relevant facts.",
        ))
    });
    let writer = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
        Ok::<_, _>(Message::assistant(
            "writer: drafted a paragraph from those facts.",
        ))
    });

    // Route: research → write → finish.
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_inner = counter.clone();
    let router = RunnableLambda::new(move |_msgs: Vec<Message>, _ctx| {
        let counter = counter_inner.clone();
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok::<_, _>(match n {
                0 => SupervisorDecision::agent("research"),
                1 => SupervisorDecision::agent("write"),
                _ => SupervisorDecision::Finish,
            })
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
        .invoke(
            SupervisorState::from_user("Write a short note on quantum computing."),
            &ExecutionContext::new(),
        )
        .await?;

    println!("=== conversation ===");
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
    println!(
        "\nlast speaker: {}",
        final_state.last_speaker.as_deref().unwrap_or("(none)")
    );
    Ok(())
}
