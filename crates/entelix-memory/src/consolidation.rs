//! `ConsolidationPolicy` — when should the buffered conversation be
//! summarised, archived, or compressed into the running summary?
//!
//! Long-running agents accumulate buffered messages indefinitely.
//! Without a consolidation pass, every model call pays for the
//! entire history; once the context window fills, useful continuity
//! is lost. This module defines the *trigger* abstraction. The
//! actual summarisation step (typically an LLM call that turns N
//! messages into a paragraph) is the operator's concern — they
//! attach a `ConsolidationPolicy` that decides *when* and supply
//! their own summariser that decides *how*.

use chrono::{DateTime, Utc};
use entelix_core::ir::Message;

/// Inputs that a [`ConsolidationPolicy`] consults when deciding
/// whether to consolidate. Carries the buffer plus optional signals
/// (last-consolidated-at, current/available context tokens) so
/// non-trivial policies — for example, "summarise once we use 80 %
/// of the model's context window" or "throttle consolidation to at
/// most once per hour" — can express their decision without
/// embedding their own clock or token counter.
///
/// Marked `#[non_exhaustive]`: callers always go through
/// [`Self::new`] and the `with_*` builders, and impls always read
/// fields by name, so adding a new signal (e.g. message count since
/// last consolidation, oldest message age) is a non-breaking change.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ConsolidationContext<'a> {
    /// The buffered conversation.
    pub buffer: &'a [Message],
    /// Wall-clock time of the most recent consolidation, if any.
    pub last_consolidated_at: Option<DateTime<Utc>>,
    /// Tokens currently consumed by the buffer in the model's
    /// context window. `None` when the host has not measured.
    pub context_tokens_used: Option<usize>,
    /// Total context-window capacity for the active model, in
    /// tokens. `None` when the host has not declared one.
    pub context_tokens_available: Option<usize>,
}

impl<'a> ConsolidationContext<'a> {
    /// Build with only the buffer — other signals default to `None`.
    /// Useful for simple agents whose policy doesn't need them.
    #[must_use]
    pub const fn new(buffer: &'a [Message]) -> Self {
        Self {
            buffer,
            last_consolidated_at: None,
            context_tokens_used: None,
            context_tokens_available: None,
        }
    }

    /// Attach the timestamp of the most recent consolidation.
    #[must_use]
    pub const fn with_last_consolidated_at(mut self, at: DateTime<Utc>) -> Self {
        self.last_consolidated_at = Some(at);
        self
    }

    /// Attach the active model's context-window state.
    #[must_use]
    pub const fn with_context_tokens(mut self, used: usize, available: usize) -> Self {
        self.context_tokens_used = Some(used);
        self.context_tokens_available = Some(available);
        self
    }
}

/// Decides whether the current buffered conversation should be
/// consolidated.
///
/// Implementations are pure functions of the supplied context — no
/// I/O, no async — so checks are free to run after every append.
/// Stateful behaviour (counters, last-trigger timestamp) lives
/// inside the impl when needed.
pub trait ConsolidationPolicy: Send + Sync + 'static {
    /// Return `true` when the buffer is ready for consolidation.
    fn should_consolidate(&self, ctx: &ConsolidationContext<'_>) -> bool;
}

/// Trigger consolidation once the buffer reaches `max_messages`.
///
/// Simplest possible policy — count messages, fire when threshold
/// crossed. Suitable for chat agents where every message is a turn.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct OnMessageCount {
    /// Maximum buffered messages before consolidation fires.
    pub max_messages: usize,
}

impl OnMessageCount {
    /// Build a policy with the given message threshold.
    #[must_use]
    pub const fn new(max_messages: usize) -> Self {
        Self { max_messages }
    }
}

impl ConsolidationPolicy for OnMessageCount {
    fn should_consolidate(&self, ctx: &ConsolidationContext<'_>) -> bool {
        ctx.buffer.len() >= self.max_messages
    }
}

/// Trigger consolidation when the buffer's total text length
/// (summed UTF-8 byte length of every `ContentPart::Text`) exceeds
/// `max_bytes`. Approximates a token-budget gate without needing a
/// tokenizer in the SDK.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct OnTokenBudget {
    /// Maximum cumulative text bytes before consolidation fires.
    pub max_bytes: usize,
}

impl OnTokenBudget {
    /// Build a policy with the given byte threshold.
    #[must_use]
    pub const fn new(max_bytes: usize) -> Self {
        Self { max_bytes }
    }
}

impl ConsolidationPolicy for OnTokenBudget {
    fn should_consolidate(&self, ctx: &ConsolidationContext<'_>) -> bool {
        let mut total: usize = 0;
        for msg in ctx.buffer {
            for part in &msg.content {
                if let entelix_core::ir::ContentPart::Text { text, .. } = part {
                    total = total.saturating_add(text.len());
                    if total >= self.max_bytes {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Never trigger. Useful as a default when consolidation is wired
/// but the operator wants to disable it temporarily without
/// rebuilding the agent graph.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct NeverConsolidate;

impl ConsolidationPolicy for NeverConsolidate {
    fn should_consolidate(&self, _ctx: &ConsolidationContext<'_>) -> bool {
        false
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use entelix_core::ir::Message;

    #[test]
    fn on_message_count_fires_at_threshold() {
        let policy = OnMessageCount::new(3);
        let one = vec![Message::user("a")];
        let three = vec![Message::user("a"), Message::user("b"), Message::user("c")];
        assert!(!policy.should_consolidate(&ConsolidationContext::new(&one)));
        assert!(policy.should_consolidate(&ConsolidationContext::new(&three)));
    }

    #[test]
    fn on_token_budget_fires_when_text_exceeds_limit() {
        let policy = OnTokenBudget::new(10);
        let small = vec![Message::user("hi")];
        let large = vec![Message::user("hello there friend")];
        assert!(!policy.should_consolidate(&ConsolidationContext::new(&small)));
        assert!(policy.should_consolidate(&ConsolidationContext::new(&large)));
    }

    #[test]
    fn never_consolidate_is_always_false() {
        let policy = NeverConsolidate;
        let buf = vec![Message::user("anything"); 1000];
        assert!(!policy.should_consolidate(&ConsolidationContext::new(&buf)));
    }
}
