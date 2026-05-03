//! `Configured<R, F>` — `Runnable<I, O>` adapter that mutates a
//! cloned [`ExecutionContext`] via a caller-supplied closure before
//! delegating to the inner runnable.
//!
//! Useful for branch-local context overrides (a sub-tree using a
//! different `tenant_id`, a stricter `deadline`, or a fresh
//! cancellation scope) without leaking those changes back to the
//! caller's context.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use entelix_core::ExecutionContext;
use entelix_core::error::Result;

use crate::runnable::Runnable;

/// `Runnable<I, O>` adapter that runs `configurer` on a cloned
/// `ExecutionContext` before forwarding to the inner. The original
/// `ctx` the parent passed in is left untouched.
pub struct Configured<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Fn(&mut ExecutionContext) + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    inner: Arc<R>,
    configurer: Arc<F>,
    _io: PhantomData<fn(I) -> O>,
}

impl<R, F, I, O> Configured<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Fn(&mut ExecutionContext) + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Build with the inner runnable and a context-mutating closure.
    pub fn new(inner: R, configurer: F) -> Self {
        Self {
            inner: Arc::new(inner),
            configurer: Arc::new(configurer),
            _io: PhantomData,
        }
    }
}

impl<R, F, I, O> std::fmt::Debug for Configured<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Fn(&mut ExecutionContext) + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Configured").finish_non_exhaustive()
    }
}

#[async_trait]
impl<R, F, I, O> Runnable<I, O> for Configured<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Fn(&mut ExecutionContext) + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        let mut scoped = ctx.clone();
        (self.configurer)(&mut scoped);
        self.inner.invoke(input, &scoped).await
    }
}
