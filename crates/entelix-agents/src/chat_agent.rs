//! `create_chat_agent` — simplest recipe: prepend a system message and
//! call the model once. Returns an [`Agent<ChatState>`] ready for
//! `execute` / `execute_stream`.

use std::sync::Arc;

use entelix_core::ir::Message;
use entelix_core::{ExecutionContext, Result};
use entelix_graph::{CompiledGraph, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

use crate::agent::Agent;
use crate::state::ChatState;

/// Build the single-node chat graph (system message prepend → model
/// → finish) without wrapping it into an [`Agent`]. Use this when
/// you want to configure the agent surface (name, sink, approver,
/// observers) via [`Agent::builder`] directly.
pub fn build_chat_graph<M>(model: M, system: impl Into<String>) -> Result<CompiledGraph<ChatState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let model = Arc::new(model);
    let system_text = system.into();
    let node = RunnableLambda::new(move |mut state: ChatState, ctx: ExecutionContext| {
        let model = model.clone();
        let system = system_text.clone();
        async move {
            let mut prompt = Vec::with_capacity(state.messages.len().saturating_add(1));
            if !system.is_empty() {
                prompt.push(Message::system(system));
            }
            prompt.extend(state.messages.iter().cloned());
            let reply = model.invoke(prompt, &ctx).await?;
            state.messages.push(reply);
            Ok::<_, _>(state)
        }
    });

    StateGraph::<ChatState>::new()
        .add_node("chat", node)
        .set_entry_point("chat")
        .add_finish_point("chat")
        .compile()
}

/// Build a single-turn chat agent: prepends `system` (if non-empty) to
/// the conversation, calls `model`, appends the reply, and finishes.
///
/// `model` is any `Runnable<Vec<Message>, Message>` — the
/// [`entelix_core::ChatModel`] alias being the typical instance, but a
/// mock model also satisfies the bound which makes the recipe easy to
/// unit-test.
pub fn create_chat_agent<M>(model: M, system: impl Into<String>) -> Result<Agent<ChatState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let graph = build_chat_graph(model, system)?;
    Agent::<ChatState>::builder()
        .with_name("chat")
        .with_runnable(graph)
        .build()
}
