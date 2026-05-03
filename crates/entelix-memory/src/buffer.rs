//! `BufferMemory` — keep the last *N* turns of a conversation in a
//! `Store<Vec<Message>>`. Simplest of the LangChain-style memory
//! patterns.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use entelix_core::ir::Message;
use entelix_core::{ExecutionContext, Result};
use parking_lot::Mutex;

use crate::consolidation::{ConsolidationContext, ConsolidationPolicy};
use crate::namespace::Namespace;
use crate::store::Store;

const DEFAULT_KEY: &str = "buffer";

/// Bounded conversation buffer.
///
/// Each call to `append` pushes one message and (if the buffer exceeds
/// `max_turns`) drops the oldest. `messages()` returns the full retained
/// list.
///
/// An optional [`ConsolidationPolicy`] can be attached via
/// [`Self::with_consolidation_policy`]. When attached,
/// [`Self::should_consolidate`] surfaces the policy's decision so the
/// caller (typically an agent recipe) can run an LLM-driven
/// summarisation pass and write the result into a sibling
/// [`crate::SummaryMemory`] before clearing the buffer. The buffer
/// tracks `last_consolidated_at` itself — recipes call
/// [`Self::mark_consolidated`] after a successful summarisation pass
/// and the time-aware policies see the updated timestamp on the next
/// check, so the consolidation loop reduces to:
///
/// ```ignore
/// if buffer.should_consolidate(&ctx).await? {
///     run_summariser(&ctx, &buffer, &summary).await?;
///     buffer.clear(&ctx).await?;
///     buffer.mark_consolidated_now();
/// }
/// ```
pub struct BufferMemory {
    store: Arc<dyn Store<Vec<Message>>>,
    namespace: Namespace,
    max_turns: usize,
    consolidation: Option<Arc<dyn ConsolidationPolicy>>,
    last_consolidated_at: Mutex<Option<DateTime<Utc>>>,
}

impl BufferMemory {
    /// Build a buffer over `store` scoped to `namespace`, retaining at
    /// most `max_turns` messages.
    pub fn new(
        store: Arc<dyn Store<Vec<Message>>>,
        namespace: Namespace,
        max_turns: usize,
    ) -> Self {
        Self {
            store,
            namespace,
            max_turns,
            consolidation: None,
            last_consolidated_at: Mutex::new(None),
        }
    }

    /// Attach a [`ConsolidationPolicy`]. The buffer itself never
    /// performs the summarisation — the policy only *decides* when
    /// the caller should — but having the policy bound here means
    /// agent recipes can ask the buffer directly via
    /// [`Self::should_consolidate`] without threading the policy
    /// through every call site.
    #[must_use]
    pub fn with_consolidation_policy(mut self, policy: Arc<dyn ConsolidationPolicy>) -> Self {
        self.consolidation = Some(policy);
        self
    }

    /// Effective retention cap.
    pub const fn max_turns(&self) -> usize {
        self.max_turns
    }

    /// Borrow the bound namespace.
    pub const fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Wall-clock time of the most recent successful consolidation,
    /// as recorded via [`Self::mark_consolidated`]. Returns `None`
    /// before the first consolidation has been marked.
    pub fn last_consolidated_at(&self) -> Option<DateTime<Utc>> {
        *self.last_consolidated_at.lock()
    }

    /// Record that a consolidation pass completed at `at`. Recipes
    /// call this after a successful summarise-and-clear cycle so
    /// time-aware policies (throttling, "at most once per hour")
    /// observe the new floor on the next check.
    pub fn mark_consolidated(&self, at: DateTime<Utc>) {
        *self.last_consolidated_at.lock() = Some(at);
    }

    /// Convenience over [`Self::mark_consolidated`] using
    /// [`chrono::Utc::now`]. Use when the caller doesn't already
    /// have a timestamp in hand.
    pub fn mark_consolidated_now(&self) {
        self.mark_consolidated(Utc::now());
    }

    /// Append `message`, dropping the oldest entries when over capacity.
    pub async fn append(&self, ctx: &ExecutionContext, message: Message) -> Result<()> {
        let mut messages = self
            .store
            .get(ctx, &self.namespace, DEFAULT_KEY)
            .await?
            .unwrap_or_default();
        messages.push(message);
        // Drop oldest while over budget.
        while messages.len() > self.max_turns {
            messages.remove(0);
        }
        self.store
            .put(ctx, &self.namespace, DEFAULT_KEY, messages)
            .await
    }

    /// Read the retained messages oldest-first.
    pub async fn messages(&self, ctx: &ExecutionContext) -> Result<Vec<Message>> {
        Ok(self
            .store
            .get(ctx, &self.namespace, DEFAULT_KEY)
            .await?
            .unwrap_or_default())
    }

    /// Clear the buffer.
    pub async fn clear(&self, ctx: &ExecutionContext) -> Result<()> {
        self.store.delete(ctx, &self.namespace, DEFAULT_KEY).await
    }

    /// Consult the bound [`ConsolidationPolicy`] (if any) against the
    /// current buffer and the buffer's tracked `last_consolidated_at`.
    /// Returns `Ok(false)` when no policy is bound — that path is the
    /// explicit "consolidation disabled" answer.
    ///
    /// For token-budget policies that need the active model's usage,
    /// use [`Self::should_consolidate_with`] and supply the values
    /// in [`PolicyExtras`].
    pub async fn should_consolidate(&self, ctx: &ExecutionContext) -> Result<bool> {
        self.should_consolidate_with(ctx, PolicyExtras::default())
            .await
    }

    /// As [`Self::should_consolidate`], but lets the caller layer on
    /// model-specific token usage signals or override the buffer's
    /// internally-tracked `last_consolidated_at` (rare — useful when
    /// a downstream system has more authoritative state).
    pub async fn should_consolidate_with(
        &self,
        ctx: &ExecutionContext,
        extras: PolicyExtras,
    ) -> Result<bool> {
        let Some(policy) = self.consolidation.as_ref() else {
            return Ok(false);
        };
        let buffer = self.messages(ctx).await?;
        let mut consolidation_ctx = ConsolidationContext::new(&buffer);
        let effective_last = extras
            .last_consolidated_at
            .or_else(|| *self.last_consolidated_at.lock());
        if let Some(at) = effective_last {
            consolidation_ctx = consolidation_ctx.with_last_consolidated_at(at);
        }
        if let (Some(used), Some(available)) =
            (extras.context_tokens_used, extras.context_tokens_available)
        {
            consolidation_ctx = consolidation_ctx.with_context_tokens(used, available);
        }
        Ok(policy.should_consolidate(&consolidation_ctx))
    }
}

/// Optional signals fed into [`BufferMemory::should_consolidate_with`].
/// Operators that don't track tokens or last-consolidated time can
/// pass [`Self::default`] and the policy will see the buffer plus
/// whatever the buffer itself tracks via [`BufferMemory::mark_consolidated`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyExtras {
    /// Wall-clock time of the most recent consolidation. When
    /// supplied, overrides the buffer's internally-tracked value
    /// for this single check.
    pub last_consolidated_at: Option<DateTime<Utc>>,
    /// Tokens currently consumed in the model's context window.
    pub context_tokens_used: Option<usize>,
    /// Total context-window capacity for the active model.
    pub context_tokens_available: Option<usize>,
}
