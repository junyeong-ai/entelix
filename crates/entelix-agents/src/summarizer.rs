//! `RunnableToSummarizerAdapter` — adapt any `Runnable<Vec<Message>, Message>`
//! into a [`entelix_memory::Summarizer`] suitable for plugging into
//! [`entelix_memory::ConsolidatingBufferMemory`].
//!
//! Lives here (in `entelix-agents`) rather than in `entelix-memory`
//! so the memory crate stays decoupled from the runnable abstraction.
//! Concrete LLM-driven summarisation belongs alongside the agent
//! recipes that use it.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ir::{ContentPart, Message};
use entelix_core::{ExecutionContext, Result};
use entelix_memory::Summarizer;
use entelix_runnable::Runnable;

/// Default system instruction prepended to the buffer when no
/// explicit prompt is supplied.
const DEFAULT_SUMMARY_SYSTEM_PROMPT: &str = "You are a summarisation assistant. Compress the \
                                              conversation that follows into a concise running \
                                              summary that preserves user intent, decisions, \
                                              and outstanding questions. Reply with the summary \
                                              text only — no preamble, no formatting markers.";

/// [`Summarizer`] that delegates to any
/// `Runnable<Vec<Message>, Message>` — the same shape as a chat
/// model. Prepends a configurable system instruction so the model
/// understands the task even when the buffer itself contains only
/// raw user/assistant turns.
pub struct RunnableToSummarizerAdapter<R> {
    runnable: Arc<R>,
    system_prompt: String,
}

impl<R> RunnableToSummarizerAdapter<R>
where
    R: Runnable<Vec<Message>, Message> + 'static,
{
    /// Build a summariser using `runnable` and the default system
    /// instruction.
    pub fn new(runnable: R) -> Self {
        Self {
            runnable: Arc::new(runnable),
            system_prompt: DEFAULT_SUMMARY_SYSTEM_PROMPT.to_owned(),
        }
    }

    /// Build a summariser using an `Arc`-wrapped runnable. Useful
    /// when the same model serves multiple memories or is shared
    /// with other parts of the agent.
    pub fn from_arc(runnable: Arc<R>) -> Self {
        Self {
            runnable,
            system_prompt: DEFAULT_SUMMARY_SYSTEM_PROMPT.to_owned(),
        }
    }

    /// Override the system instruction. Use to inject domain-specific
    /// summary guidance ("preserve transaction IDs verbatim"; "always
    /// summarise in the user's preferred language").
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

#[async_trait]
impl<R> Summarizer for RunnableToSummarizerAdapter<R>
where
    R: Runnable<Vec<Message>, Message> + 'static,
{
    async fn summarize(&self, messages: Vec<Message>, ctx: &ExecutionContext) -> Result<String> {
        let mut prompt = Vec::with_capacity(messages.len() + 1);
        prompt.push(Message::system(self.system_prompt.clone()));
        prompt.extend(messages);
        let response = self.runnable.invoke(prompt, ctx).await?;
        Ok(extract_text(&response))
    }
}

/// Concatenate every `ContentPart::Text` payload in `message`
/// preserving order. Non-text parts are skipped — the consolidator
/// only needs the textual summary.
fn extract_text(message: &Message) -> String {
    let mut out = String::new();
    for part in &message.content {
        if let ContentPart::Text { text, .. } = part {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(text);
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use entelix_runnable::RunnableLambda;

    #[tokio::test]
    async fn runnable_summariser_extracts_text_from_response() {
        let runnable = RunnableLambda::new(|_msgs: Vec<Message>, _ctx| async move {
            Ok::<_, _>(Message::assistant("compressed summary"))
        });
        let summariser = RunnableToSummarizerAdapter::new(runnable);
        let ctx = ExecutionContext::new();
        let out = summariser
            .summarize(vec![Message::user("hi"), Message::assistant("hello")], &ctx)
            .await
            .unwrap();
        assert_eq!(out, "compressed summary");
    }

    #[tokio::test]
    async fn runnable_summariser_prepends_system_prompt() {
        // Capture the prompt the runnable receives — verify the
        // system instruction is present at index 0.
        use std::sync::Mutex;
        let captured: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_inner = Arc::clone(&captured);
        let runnable = RunnableLambda::new(move |msgs: Vec<Message>, _ctx| {
            let captured = Arc::clone(&captured_inner);
            async move {
                *captured.lock().unwrap() = msgs;
                Ok::<_, _>(Message::assistant("ok"))
            }
        });
        let summariser =
            RunnableToSummarizerAdapter::new(runnable).with_system_prompt("custom system prompt");
        let _ = summariser
            .summarize(vec![Message::user("hi")], &ExecutionContext::new())
            .await
            .unwrap();
        let prompt = captured.lock().unwrap().clone();
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, entelix_core::ir::Role::System);
        if let ContentPart::Text { text, .. } = &prompt[0].content[0] {
            assert_eq!(text, "custom system prompt");
        } else {
            panic!("expected Text part");
        }
    }
}
