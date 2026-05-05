//! `ConsolidatingBufferMemory` — opinionated layered memory that
//! ties a `BufferMemory`, a `SummaryMemory`, and a [`Summarizer`]
//! into a single `append`-driven loop.
//!
//! Without this adapter every agent recipe rewrites the same
//! sequence: ask the buffer's policy, run an LLM summarisation pass,
//! append the result to the summary, clear the buffer, mark
//! consolidated. The wrapper bakes that loop into one type so the
//! recipe code reduces to `mem.append(ctx, msg).await?`.
//!
//! ## Provider-agnostic
//!
//! [`Summarizer`] is a trait over `(messages, ctx) -> String` —
//! independent of which LLM provider runs the summarisation. The
//! `entelix-agents` crate provides `RunnableToSummarizerAdapter`, which
//! adapts any `Runnable<Vec<Message>, Message>` (Anthropic codec,
//! `OpenAI` codec, a stubbed test runnable, …) into this trait by
//! prepending a configurable system instruction and extracting the
//! response's text content. Memory itself stays decoupled from the
//! `Runnable` abstraction so backends can plug in non-LLM
//! summarisers (heuristic compression, cached templates) without
//! pulling in the runnable dependency.
//!
//! ## Failure semantics
//!
//! When the summariser fails, the underlying buffer is **not**
//! cleared and `last_consolidated_at` is **not** updated — the next
//! call to `append` will re-attempt consolidation. This keeps a
//! transient summariser outage from silently dropping conversation
//! history.

use entelix_core::TenantId;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ir::Message;
use entelix_core::{ExecutionContext, Result};

use crate::buffer::BufferMemory;
use crate::summary::SummaryMemory;

/// Reduces a buffer of conversation messages to a summary string.
///
/// Implementations decide *how* to summarise — typically by calling
/// an LLM. The trait stays provider-agnostic; concrete impls (such
/// as `entelix_agents::RunnableToSummarizerAdapter`) wire in the model.
#[async_trait]
pub trait Summarizer: Send + Sync + 'static {
    /// Summarise `messages` into a single string. Returning `Err`
    /// signals a transient failure — the consolidating buffer
    /// keeps the original messages and re-attempts next call.
    async fn summarize(&self, messages: Vec<Message>, ctx: &ExecutionContext) -> Result<String>;
}

/// Layered memory: a [`BufferMemory`] for recent turns, a
/// [`SummaryMemory`] for the running summary, and a [`Summarizer`]
/// that bridges the two when the buffer's policy fires.
///
/// `append` drives the full loop:
///
/// ```ignore
/// let mem = ConsolidatingBufferMemory::new(buffer, summary, summariser);
/// mem.append(&ctx, Message::user("hi")).await?;
/// // — buffer now has the new message; if the policy fires,
/// //   the previous buffer has already been summarised into
/// //   `summary` and the buffer cleared.
/// ```
pub struct ConsolidatingBufferMemory {
    buffer: Arc<BufferMemory>,
    summary: Arc<SummaryMemory>,
    summarizer: Arc<dyn Summarizer>,
}

impl ConsolidatingBufferMemory {
    /// Build a layered memory from an existing buffer, summary, and
    /// summariser. The buffer must already have a
    /// [`crate::ConsolidationPolicy`] attached via
    /// [`BufferMemory::with_consolidation_policy`] — without one the
    /// adapter never consolidates and behaves as a thin
    /// `BufferMemory` proxy.
    pub fn new(
        buffer: Arc<BufferMemory>,
        summary: Arc<SummaryMemory>,
        summarizer: Arc<dyn Summarizer>,
    ) -> Self {
        Self {
            buffer,
            summary,
            summarizer,
        }
    }

    /// Borrow the underlying buffer (for direct queries that bypass
    /// the consolidation loop, such as size accounting).
    pub const fn buffer(&self) -> &Arc<BufferMemory> {
        &self.buffer
    }

    /// Borrow the underlying summary memory.
    pub const fn summary(&self) -> &Arc<SummaryMemory> {
        &self.summary
    }

    /// Append `message` to the buffer, then check the bound
    /// consolidation policy. When it fires, summarise the buffered
    /// messages, append the summary to [`SummaryMemory`], clear the
    /// buffer, and mark the buffer's `last_consolidated_at`.
    pub async fn append(&self, ctx: &ExecutionContext, message: Message) -> Result<()> {
        self.buffer.append(ctx, message).await?;
        if !self.buffer.should_consolidate(ctx).await? {
            return Ok(());
        }
        let messages = self.buffer.messages(ctx).await?;
        // Summarise BEFORE mutating either store. If summarise
        // fails, we surface the error and leave the buffer intact;
        // the caller can retry on the next append.
        let summary_text = self.summarizer.summarize(messages, ctx).await?;
        self.summary.append(ctx, &summary_text).await?;
        self.buffer.clear(ctx).await?;
        self.buffer.mark_consolidated_now();
        Ok(())
    }

    /// Fetch the current buffered messages.
    pub async fn messages(&self, ctx: &ExecutionContext) -> Result<Vec<Message>> {
        self.buffer.messages(ctx).await
    }

    /// Fetch the current running summary.
    pub async fn current_summary(&self, ctx: &ExecutionContext) -> Result<Option<String>> {
        self.summary.get(ctx).await
    }

    /// Reset both layers — buffer and summary.
    pub async fn clear(&self, ctx: &ExecutionContext) -> Result<()> {
        self.buffer.clear(ctx).await?;
        self.summary.clear(ctx).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::consolidation::{ConsolidationPolicy, OnMessageCount};
    use crate::namespace::Namespace;
    use crate::store::InMemoryStore;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Stub summariser that records call count and returns a fixed
    /// reply (or a fixed error) — keeps the memory crate's tests
    /// independent of any LLM runnable.
    struct StubSummarizer {
        calls: Arc<AtomicUsize>,
        reply: Result<String>,
    }

    impl StubSummarizer {
        fn ok(reply: &str) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: calls.clone(),
                    reply: Ok(reply.to_owned()),
                },
                calls,
            )
        }

        fn err(msg: &str) -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
                reply: Err(entelix_core::Error::config(msg.to_owned())),
            }
        }
    }

    #[async_trait]
    impl Summarizer for StubSummarizer {
        async fn summarize(
            &self,
            _messages: Vec<Message>,
            _ctx: &ExecutionContext,
        ) -> Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.reply {
                Ok(s) => Ok(s.clone()),
                Err(e) => Err(clone_error(e)),
            }
        }
    }

    fn clone_error(e: &entelix_core::Error) -> entelix_core::Error {
        // Tests only ever clone Config errors; other variants would
        // require a richer cloning strategy than this minimal stub.
        match e {
            entelix_core::Error::Config(c) => entelix_core::Error::config(c.to_string()),
            other => entelix_core::Error::config(format!("{other}")),
        }
    }

    fn make_buffer(max_turns: usize, policy: Arc<dyn ConsolidationPolicy>) -> Arc<BufferMemory> {
        Arc::new(
            BufferMemory::new(
                Arc::new(InMemoryStore::<Vec<Message>>::new()),
                Namespace::new(TenantId::new("t")).with_scope("conv"),
                max_turns,
            )
            .with_consolidation_policy(policy),
        )
    }

    fn make_summary() -> Arc<SummaryMemory> {
        Arc::new(SummaryMemory::new(
            Arc::new(InMemoryStore::<String>::new()),
            Namespace::new(TenantId::new("t")).with_scope("conv"),
        ))
    }

    #[tokio::test]
    async fn append_does_not_consolidate_below_threshold() {
        let buf = make_buffer(10, Arc::new(OnMessageCount::new(5)));
        let sum = make_summary();
        let (summariser, calls) = StubSummarizer::ok("summary");
        let mem = ConsolidatingBufferMemory::new(buf, sum.clone(), Arc::new(summariser));
        let ctx = ExecutionContext::new();
        for i in 0..3 {
            mem.append(&ctx, Message::user(format!("m{i}")))
                .await
                .unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(mem.messages(&ctx).await.unwrap().len(), 3);
        assert!(mem.current_summary(&ctx).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn append_consolidates_when_threshold_reached() {
        let buf = make_buffer(10, Arc::new(OnMessageCount::new(3)));
        let sum = make_summary();
        let (summariser, calls) = StubSummarizer::ok("compressed");
        let mem = ConsolidatingBufferMemory::new(
            Arc::clone(&buf),
            Arc::clone(&sum),
            Arc::new(summariser),
        );
        let ctx = ExecutionContext::new();
        for i in 0..3 {
            mem.append(&ctx, Message::user(format!("m{i}")))
                .await
                .unwrap();
        }
        // Threshold of 3: third append triggers consolidation.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Buffer is cleared, summary now holds the summarisation.
        assert_eq!(mem.messages(&ctx).await.unwrap().len(), 0);
        let summary = mem.current_summary(&ctx).await.unwrap().unwrap();
        assert_eq!(summary, "compressed");
        assert!(buf.last_consolidated_at().is_some());
    }

    #[tokio::test]
    async fn summariser_failure_preserves_buffer() {
        let buf = make_buffer(10, Arc::new(OnMessageCount::new(2)));
        let sum = make_summary();
        let summariser = StubSummarizer::err("summariser down");
        let mem = ConsolidatingBufferMemory::new(
            Arc::clone(&buf),
            Arc::clone(&sum),
            Arc::new(summariser),
        );
        let ctx = ExecutionContext::new();
        mem.append(&ctx, Message::user("a")).await.unwrap();
        let err = mem.append(&ctx, Message::user("b")).await.unwrap_err();
        assert!(matches!(err, entelix_core::Error::Config(_)));
        // Buffer NOT cleared — caller can retry next turn.
        assert_eq!(mem.messages(&ctx).await.unwrap().len(), 2);
        // Summary NOT touched — no partial state.
        assert!(mem.current_summary(&ctx).await.unwrap().is_none());
        // last_consolidated_at NOT advanced.
        assert!(buf.last_consolidated_at().is_none());
    }
}
