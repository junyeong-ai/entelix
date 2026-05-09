//! Auto-compaction adapter — wraps any `Runnable<Vec<Message>, Message>`
//! with a threshold-driven [`Compactor`] trigger so context-window
//! pressure is handled transparently.
//!
//! ## Why a `Runnable` adapter, not an agent-loop hook
//!
//! Compaction is a *message-history concern* that lives orthogonal to
//! recipe choice (ReAct / Supervisor / Chat). Wrapping the model
//! itself with [`RunnableCompacting`] composes through every recipe
//! unchanged — the planner / agent / chat node calls `model.invoke(.)`
//! and the wrapper transparently compacts before delegating. No
//! recipe code touches the trigger logic; new recipes inherit the
//! behaviour for free.
//!
//! ## Pair-invariant preservation
//!
//! The wrapper routes through [`messages_to_events`] +
//! [`Compactor::compact`] + [`CompactedHistory::to_messages`], so the
//! sealed `tool_call` / `tool_result` pair invariant
//! ([`entelix_session::ToolPair`]) survives the round-trip. The
//! vendor-side wire format never sees an unmatched tool block.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};
use entelix_core::ir::Message;
use entelix_runnable::Runnable;
use entelix_session::{Compactor, messages_char_size, messages_to_events};

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
            let events = messages_to_events(&input)?;
            self.compactor
                .compact(&events, self.threshold_chars)?
                .to_messages()
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_core::ir::{ContentPart, Message, Role};
    use entelix_session::HeadDropCompactor;

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
        let _ = wrapped.invoke(input.clone(), &ExecutionContext::new()).await.unwrap();
        let observed_len = wrapped.inner().last_input_len.load(Ordering::SeqCst);
        assert!(
            observed_len < input.len(),
            "compaction must trim — got {observed_len}, input had {}",
            input.len()
        );
    }

    #[tokio::test]
    async fn empty_messages_pass_through() {
        let compactor: Arc<dyn Compactor> = Arc::new(HeadDropCompactor);
        let model = EchoModel::new();
        let wrapped = model.with_compaction(compactor, 1024);
        let _ = wrapped.invoke(Vec::new(), &ExecutionContext::new()).await.unwrap();
        assert_eq!(wrapped.inner().last_input_len.load(Ordering::SeqCst), 0);
    }
}
