//! Auto-compaction adapter + the canonical LLM-summary compactor.
//!
//! ## [`RunnableCompacting`] — agent-orthogonal trigger
//!
//! Compaction is a *message-history concern* that lives orthogonal to
//! recipe choice (ReAct / Supervisor / Chat). Wrapping the model
//! itself with [`RunnableCompacting`] composes through every recipe
//! unchanged — the planner / agent / chat node calls `model.invoke(.)`
//! and the wrapper transparently compacts before delegating. No
//! recipe code touches the trigger logic; new recipes inherit the
//! behaviour for free.
//!
//! The wrapper routes through [`messages_to_events`] +
//! [`Compactor::compact`] + [`CompactedHistory::to_messages`], so the
//! sealed `tool_call` / `tool_result` pair invariant
//! ([`entelix_session::ToolPair`]) survives the round-trip. The
//! vendor-side wire format never sees an unmatched tool block.
//!
//! ## [`SummaryCompactor`] — LLM-summary [`Compactor`] impl
//!
//! Operators wanting Claude Agent SDK's auto-compaction behaviour or
//! LangChain's `SummarizationMiddleware` reach for [`SummaryCompactor`]:
//! the oldest turns past `keep_recent_turns` are rendered, summarised
//! by an operator-supplied summariser model, and replaced with a single
//! synthetic `Turn::User` carrying the summary. Pair invariant survives
//! because dropped turns leave with their `ToolPair`s.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::{ExecutionContext, Result};
use entelix_runnable::Runnable;
use entelix_session::{
    CompactedHistory, Compactor, GraphEvent, Turn, messages_char_size, messages_to_events,
};

/// `Runnable<Vec<Message>, Message>` wrapper that compacts the input
/// message slice through an operator-supplied [`Compactor`] when the
/// total character count meets or exceeds `threshold_chars`. Below the
/// threshold the wrapper is a no-op delegate — the inner runnable
/// receives the original `Vec<Message>` unchanged.
///
/// Construct via [`MessageRunnableCompactionExt::with_compaction`]:
///
/// ```ignore
/// use entelix::{Compactor, HeadDropCompactor, MessageRunnableCompactionExt};
/// use std::sync::Arc;
///
/// let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
/// let model = my_chat_model.with_compaction(compactor, 8_192);
/// let agent = entelix::create_react_agent(model, tools, None)?;
/// ```
pub struct RunnableCompacting<R> {
    inner: R,
    compactor: Arc<dyn Compactor>,
    threshold_chars: usize,
}

impl<R> RunnableCompacting<R> {
    /// Threshold (in character count) at and above which the wrapper
    /// invokes the [`Compactor`]. Mirrors the `budget_chars` semantic
    /// the compactor uses to size its output.
    #[must_use]
    pub const fn threshold_chars(&self) -> usize {
        self.threshold_chars
    }

    /// Borrow the wrapped runnable.
    pub const fn inner(&self) -> &R {
        &self.inner
    }
}

#[async_trait]
impl<R> Runnable<Vec<Message>, Message> for RunnableCompacting<R>
where
    R: Runnable<Vec<Message>, Message> + Send + Sync + 'static,
{
    async fn invoke(&self, input: Vec<Message>, ctx: &ExecutionContext) -> Result<Message> {
        let input = if messages_char_size(&input) >= self.threshold_chars {
            let dropped_size = messages_char_size(&input);
            let events = messages_to_events(&input)?;
            let compacted = self
                .compactor
                .compact(&events, self.threshold_chars, ctx)
                .await?
                .to_messages();
            let retained_size = messages_char_size(&compacted);
            if let Some(handle) = ctx.audit_sink() {
                handle.as_sink().record_context_compacted(
                    dropped_size.saturating_sub(retained_size),
                    retained_size,
                );
            }
            compacted
        } else {
            input
        };
        self.inner.invoke(input, ctx).await
    }
}

/// Extension trait that attaches [`RunnableCompacting`] to any
/// `Runnable<Vec<Message>, Message>`. Blanket-impl'd for every such
/// runnable so a model accepting messages — including layered models
/// (`OtelLayer`, `PolicyLayer`, `RetryService`) — can chain `.with_compaction(.)`
/// without a separate import per concrete type.
pub trait MessageRunnableCompactionExt: Runnable<Vec<Message>, Message> + Sized {
    /// Wrap with auto-compaction. The wrapper is itself a
    /// `Runnable<Vec<Message>, Message>`, so it composes back into
    /// any recipe that takes a model.
    fn with_compaction(
        self,
        compactor: Arc<dyn Compactor>,
        threshold_chars: usize,
    ) -> RunnableCompacting<Self> {
        RunnableCompacting {
            inner: self,
            compactor,
            threshold_chars,
        }
    }
}

impl<R> MessageRunnableCompactionExt for R where R: Runnable<Vec<Message>, Message> + Sized {}

/// Default system prompt the [`SummaryCompactor`] sends to its
/// summariser model when the operator does not override. Phrased as a
/// neutral compress-the-prior-conversation instruction so it works
/// across vendors that route system prompts identically.
pub const DEFAULT_SUMMARY_SYSTEM_PROMPT: &str = "You are a conversation summariser. Distil the conversation below into 100-200 words preserving key facts, decisions, entities, and tool outcomes. Output ONLY the summary text — no preamble, no commentary.";

/// Default count of newest turns the [`SummaryCompactor`] keeps verbatim
/// before summarising the older history into one synthetic turn. Four
/// matches the typical LLM-agent rhythm (most recent user/assistant
/// pair plus one preceding pair) — small enough that summarisation
/// kicks in early, large enough that adjacent context survives.
pub const DEFAULT_SUMMARY_KEEP_RECENT_TURNS: usize = 4;

/// LLM-summary [`Compactor`] — drops the oldest turns past
/// `keep_recent_turns` into a single summarised `Turn::User`, leaving
/// the most recent turns verbatim.
///
/// Pair invariant: dropped turns carry their `ToolPair`s away with
/// them — the retained set keeps every `Turn::Assistant`'s `tools`
/// vector intact, so the wire-side codec never sees an unmatched
/// tool block.
///
/// Construct with [`SummaryCompactor::new`] then chain
/// [`SummaryCompactor::with_system_prompt`] /
/// [`SummaryCompactor::with_keep_recent_turns`] for tuning. The
/// summariser model is any `Runnable<Vec<Message>, Message>` — the
/// operator's `ChatModel`, a layered model, or a stub for tests.
pub struct SummaryCompactor<M> {
    model: Arc<M>,
    system_prompt: String,
    keep_recent_turns: usize,
}

impl<M> SummaryCompactor<M> {
    /// Construct with the default system prompt and keep-recent count.
    #[must_use]
    pub fn new(model: Arc<M>) -> Self {
        Self {
            model,
            system_prompt: DEFAULT_SUMMARY_SYSTEM_PROMPT.to_owned(),
            keep_recent_turns: DEFAULT_SUMMARY_KEEP_RECENT_TURNS,
        }
    }

    /// Override the system prompt. Operators with a custom voice or
    /// downstream-format requirement (e.g. JSON envelope) point the
    /// summariser via this knob.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Override how many newest turns are retained verbatim. Higher
    /// values preserve more recent context at the cost of leaving more
    /// budget pressure for the summariser to manage.
    #[must_use]
    pub const fn with_keep_recent_turns(mut self, n: usize) -> Self {
        self.keep_recent_turns = n;
        self
    }
}

#[async_trait]
impl<M> Compactor for SummaryCompactor<M>
where
    M: Runnable<Vec<Message>, Message> + Send + Sync + 'static,
{
    async fn compact(
        &self,
        events: &[GraphEvent],
        _budget_chars: usize,
        ctx: &ExecutionContext,
    ) -> Result<CompactedHistory> {
        let grouped = CompactedHistory::group(events)?;
        let total = grouped.len();
        if total <= self.keep_recent_turns {
            return Ok(grouped);
        }
        let split_at = total - self.keep_recent_turns;
        let mut all = grouped.turns().to_vec();
        let recent = all.split_off(split_at);
        let older = all;
        if older.is_empty() {
            return Ok(CompactedHistory::from_turns(recent));
        }
        let older_messages = CompactedHistory::from_turns(older).to_messages();
        let mut prompt = Vec::with_capacity(older_messages.len() + 1);
        prompt.push(Message::new(
            Role::System,
            vec![ContentPart::text(self.system_prompt.clone())],
        ));
        prompt.extend(older_messages);
        let summary_msg = self.model.invoke(prompt, ctx).await?;
        let summary_text = extract_text(&summary_msg.content);
        let summary_turn = Turn::User {
            content: vec![ContentPart::text(format!(
                "[Summary of earlier conversation]\n{summary_text}"
            ))],
        };
        let mut combined = Vec::with_capacity(1 + recent.len());
        combined.push(summary_turn);
        combined.extend(recent);
        Ok(CompactedHistory::from_turns(combined))
    }
}

fn extract_text(parts: &[ContentPart]) -> String {
    let mut out = String::new();
    for part in parts {
        if let ContentPart::Text { text, .. } = part {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_core::ir::{ContentPart, Message, Role};
    use entelix_session::HeadDropCompactor;
    use parking_lot::Mutex;

    use super::*;

    struct EchoModel {
        invocations: AtomicUsize,
        last_input_len: AtomicUsize,
    }

    impl EchoModel {
        fn new() -> Self {
            Self {
                invocations: AtomicUsize::new(0),
                last_input_len: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, Message> for EchoModel {
        async fn invoke(&self, input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            self.last_input_len.store(input.len(), Ordering::SeqCst);
            Ok(Message::new(Role::Assistant, vec![ContentPart::text("ok")]))
        }
    }

    fn user(text: &str) -> Message {
        Message::new(Role::User, vec![ContentPart::text(text)])
    }

    fn assistant(text: &str) -> Message {
        Message::new(Role::Assistant, vec![ContentPart::text(text)])
    }

    #[tokio::test]
    async fn passes_through_below_threshold() {
        let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
        let wrapped = EchoModel::new().with_compaction(compactor, 1024);

        let input = vec![user("short"), assistant("ok")];
        let _ = wrapped
            .invoke(input.clone(), &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(
            wrapped.inner().last_input_len.load(Ordering::SeqCst),
            input.len()
        );
    }

    #[tokio::test]
    async fn compacts_when_threshold_exceeded() {
        let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
        let model = EchoModel::new();
        // Threshold = 30 characters. Three round-trips well above that
        // trigger compaction; the head-drop strategy retains the
        // newest turns that fit under the budget.
        let wrapped = model.with_compaction(compactor, 30);

        let input = vec![
            user("one one one one"),
            assistant("first reply long enough"),
            user("two two two two"),
            assistant("second reply long enough"),
            user("three three three three"),
            assistant("third reply"),
        ];
        let _ = wrapped
            .invoke(input.clone(), &ExecutionContext::new())
            .await
            .unwrap();
        let observed_len = wrapped.inner().last_input_len.load(Ordering::SeqCst);
        assert!(
            observed_len < input.len(),
            "compaction must trim — got {observed_len}, input had {}",
            input.len()
        );
    }

    /// `AuditSink` test impl that records every `record_*` call so
    /// the compaction-emit assertion can verify the audit channel is
    /// actually crossed (invariant 18).
    struct CapturingAuditSink {
        compactions: Mutex<Vec<(usize, usize)>>,
    }

    impl CapturingAuditSink {
        fn new() -> Self {
            Self {
                compactions: Mutex::new(Vec::new()),
            }
        }
    }

    impl entelix_core::AuditSink for CapturingAuditSink {
        fn record_sub_agent_invoked(&self, _agent_id: &str, _sub_thread_id: &str) {}
        fn record_agent_handoff(&self, _from: Option<&str>, _to: &str) {}
        fn record_resumed(&self, _from_checkpoint: &str) {}
        fn record_memory_recall(&self, _tier: &str, _namespace_key: &str, _hits: usize) {}
        fn record_usage_limit_exceeded(&self, _breach: &entelix_core::UsageLimitBreach) {}
        fn record_context_compacted(&self, dropped_chars: usize, retained_chars: usize) {
            self.compactions
                .lock()
                .push((dropped_chars, retained_chars));
        }
    }

    #[tokio::test]
    async fn compaction_records_audit_event_when_threshold_exceeded() {
        let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
        let model = EchoModel::new();
        let wrapped = model.with_compaction(compactor, 30);
        let sink = Arc::new(CapturingAuditSink::new());
        let ctx = ExecutionContext::new()
            .with_audit_sink(entelix_core::AuditSinkHandle::new(sink.clone()));

        let input = vec![
            user("padding to force compaction one one one one"),
            assistant("more padding to force compaction"),
            user("trailing turn"),
            assistant("ok"),
        ];
        let _ = wrapped.invoke(input, &ctx).await.unwrap();

        let captured = sink.compactions.lock().clone();
        assert_eq!(captured.len(), 1, "exactly one compaction event expected");
        let (dropped, _retained) = captured[0];
        assert!(dropped > 0, "must report some dropped characters");
    }

    #[tokio::test]
    async fn compaction_records_no_audit_event_below_threshold() {
        let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
        let model = EchoModel::new();
        let wrapped = model.with_compaction(compactor, 1024);
        let sink = Arc::new(CapturingAuditSink::new());
        let ctx = ExecutionContext::new()
            .with_audit_sink(entelix_core::AuditSinkHandle::new(sink.clone()));

        let input = vec![user("short"), assistant("ok")];
        let _ = wrapped.invoke(input, &ctx).await.unwrap();

        assert!(
            sink.compactions.lock().is_empty(),
            "no audit event expected when threshold is not crossed"
        );
    }

    #[tokio::test]
    async fn empty_messages_pass_through() {
        let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
        let model = EchoModel::new();
        let wrapped = model.with_compaction(compactor, 1024);
        let _ = wrapped
            .invoke(Vec::new(), &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(wrapped.inner().last_input_len.load(Ordering::SeqCst), 0);
    }

    /// Stub summariser model that records the prompt it received and
    /// always replies with a fixed summary text. Lets the
    /// `SummaryCompactor` tests assert exactly which turns were sent
    /// to the summariser.
    struct StubSummariser {
        captured_prompt: Mutex<Vec<Message>>,
        reply: String,
    }

    impl StubSummariser {
        fn new(reply: impl Into<String>) -> Self {
            Self {
                captured_prompt: Mutex::new(Vec::new()),
                reply: reply.into(),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, Message> for StubSummariser {
        async fn invoke(&self, input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
            *self.captured_prompt.lock() = input;
            Ok(Message::new(
                Role::Assistant,
                vec![ContentPart::text(self.reply.clone())],
            ))
        }
    }

    fn user_event(text: &str) -> entelix_session::GraphEvent {
        entelix_session::GraphEvent::UserMessage {
            content: vec![ContentPart::text(text)],
            timestamp: chrono::Utc::now(),
        }
    }

    fn assistant_event(text: &str) -> entelix_session::GraphEvent {
        entelix_session::GraphEvent::AssistantMessage {
            content: vec![ContentPart::text(text)],
            usage: None,
            timestamp: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn summary_compactor_skips_when_under_keep_recent_threshold() {
        let summariser = Arc::new(StubSummariser::new("never invoked"));
        let compactor = SummaryCompactor::new(summariser.clone()).with_keep_recent_turns(8);
        let events = vec![
            user_event("u1"),
            assistant_event("a1"),
            user_event("u2"),
            assistant_event("a2"),
        ];
        let history = compactor
            .compact(&events, 0, &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(history.len(), 4);
        assert!(
            summariser.captured_prompt.lock().is_empty(),
            "summariser must NOT be invoked when total <= keep_recent_turns"
        );
    }

    #[tokio::test]
    async fn summary_compactor_replaces_older_turns_with_summary() {
        let summariser = Arc::new(StubSummariser::new("brief recap"));
        let compactor = SummaryCompactor::new(summariser.clone()).with_keep_recent_turns(2);
        let events = vec![
            user_event("oldest user"),
            assistant_event("oldest assistant"),
            user_event("middle user"),
            assistant_event("middle assistant"),
            user_event("newest user"),
            assistant_event("newest assistant"),
        ];
        let history = compactor
            .compact(&events, 0, &ExecutionContext::new())
            .await
            .unwrap();
        // Summary turn (1) + retained newest turns (2) = 3
        assert_eq!(history.len(), 3);
        // Head is the synthetic User summary.
        if let Turn::User { content } = &history.turns()[0] {
            if let ContentPart::Text { text, .. } = &content[0] {
                assert!(text.contains("Summary"), "summary marker missing: {text}");
                assert!(
                    text.contains("brief recap"),
                    "summariser reply missing: {text}"
                );
            }
        } else {
            panic!("expected User turn at head");
        }
        // Summariser was invoked with system + 4 older turns rendered as messages.
        let captured_len;
        let captured_role;
        {
            let captured = summariser.captured_prompt.lock();
            captured_len = captured.len();
            captured_role = captured[0].role;
        }
        assert!(
            captured_len >= 5,
            "expected system + ≥4 older messages, got {captured_len}"
        );
        assert!(matches!(captured_role, Role::System));
    }

    #[tokio::test]
    async fn summary_compactor_with_system_prompt_overrides_default() {
        let summariser = Arc::new(StubSummariser::new("ok"));
        let compactor = SummaryCompactor::new(summariser.clone())
            .with_keep_recent_turns(0)
            .with_system_prompt("CUSTOM PROMPT MARKER");
        let events = vec![user_event("hi"), assistant_event("hello")];
        let _ = compactor
            .compact(&events, 0, &ExecutionContext::new())
            .await
            .unwrap();
        let prompt_text = {
            let captured = summariser.captured_prompt.lock();
            if let ContentPart::Text { text, .. } = &captured[0].content[0] {
                text.clone()
            } else {
                panic!("expected Text part at system position");
            }
        };
        assert!(
            prompt_text.contains("CUSTOM PROMPT MARKER"),
            "operator-supplied prompt must reach the summariser, got: {prompt_text}"
        );
    }
}
