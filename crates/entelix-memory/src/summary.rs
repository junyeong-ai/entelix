//! `SummaryMemory` — single-string running summary keyed by namespace.
//!
//! Use case: rolling conversation summaries that fit a fixed token budget.
//! The summarization logic itself is the caller's responsibility — pair
//! with an LLM call after `messages()` to produce the new summary, then
//! `set` it.

use std::sync::Arc;

use entelix_core::{ExecutionContext, Result};

use crate::namespace::Namespace;
use crate::store::Store;

const DEFAULT_KEY: &str = "summary";

/// Single-string summary keyed by `Namespace`.
pub struct SummaryMemory {
    store: Arc<dyn Store<String>>,
    namespace: Namespace,
}

impl SummaryMemory {
    /// Build a summary memory over `store` scoped to `namespace`.
    pub fn new(store: Arc<dyn Store<String>>, namespace: Namespace) -> Self {
        Self { store, namespace }
    }

    /// Borrow the bound namespace.
    pub const fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// Replace the current summary.
    pub async fn set(&self, ctx: &ExecutionContext, summary: impl Into<String>) -> Result<()> {
        self.store
            .put(ctx, &self.namespace, DEFAULT_KEY, summary.into())
            .await
    }

    /// Append `addition` to the existing summary, separated by a blank
    /// line. If no summary exists, `addition` becomes the summary.
    pub async fn append(&self, ctx: &ExecutionContext, addition: &str) -> Result<()> {
        let merged = match self.store.get(ctx, &self.namespace, DEFAULT_KEY).await? {
            Some(existing) if !existing.is_empty() => format!("{existing}\n\n{addition}"),
            _ => addition.to_owned(),
        };
        self.store
            .put(ctx, &self.namespace, DEFAULT_KEY, merged)
            .await
    }

    /// Read the current summary.
    pub async fn get(&self, ctx: &ExecutionContext) -> Result<Option<String>> {
        self.store.get(ctx, &self.namespace, DEFAULT_KEY).await
    }

    /// Delete the summary.
    pub async fn clear(&self, ctx: &ExecutionContext) -> Result<()> {
        self.store.delete(ctx, &self.namespace, DEFAULT_KEY).await
    }
}
