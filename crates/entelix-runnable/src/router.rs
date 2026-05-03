//! `RunnableRouter` — predicate-based dispatch.
//!
//! Routes are tried in registration order; the first matching predicate
//! wins. An optional default branch handles inputs no predicate accepts.
//! No match + no default ⇒ `Error::InvalidRequest`.

use std::sync::Arc;

use entelix_core::{Error, ExecutionContext, Result};

use crate::runnable::Runnable;

type Predicate<I> = Arc<dyn Fn(&I) -> bool + Send + Sync>;
type Branch<I, O> = (Predicate<I>, Arc<dyn Runnable<I, O>>);

/// `Runnable<I, O>` that picks one of several runnables based on a predicate
/// over the input.
pub struct RunnableRouter<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    routes: Vec<Branch<I, O>>,
    fallback: Option<Arc<dyn Runnable<I, O>>>,
}

impl<I, O> RunnableRouter<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Empty router with no routes and no default.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            fallback: None,
        }
    }

    /// Register a (predicate, runnable) pair. Routes are evaluated in
    /// registration order.
    #[must_use]
    pub fn route<F, R>(mut self, predicate: F, runnable: R) -> Self
    where
        F: Fn(&I) -> bool + Send + Sync + 'static,
        R: Runnable<I, O> + 'static,
    {
        self.routes.push((Arc::new(predicate), Arc::new(runnable)));
        self
    }

    /// Set the fallback branch (used when no predicate matches). Calling
    /// twice replaces the previous default.
    #[must_use]
    pub fn fallback<R>(mut self, runnable: R) -> Self
    where
        R: Runnable<I, O> + 'static,
    {
        self.fallback = Some(Arc::new(runnable));
        self
    }

    /// Number of registered routes (excludes the fallback).
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// True when no routes are registered (the fallback alone does not
    /// count).
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
}

impl<I, O> Default for RunnableRouter<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl<I, O> Runnable<I, O> for RunnableRouter<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        for (predicate, runnable) in &self.routes {
            if predicate(&input) {
                return runnable.invoke(input, ctx).await;
            }
        }
        if let Some(fallback) = &self.fallback {
            return fallback.invoke(input, ctx).await;
        }
        Err(Error::invalid_request(
            "RunnableRouter: no route matched and no fallback was set",
        ))
    }
}
