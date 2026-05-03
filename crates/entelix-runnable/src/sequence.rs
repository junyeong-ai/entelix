//! Sequential composition — output of `first` becomes input of `second`.
//!
//! Constructed via [`RunnableExt::pipe`], the canonical entry point.
//!
//! [`RunnableExt::pipe`]: crate::ext::RunnableExt::pipe

use std::sync::Arc;

use entelix_core::{ExecutionContext, Result};

use crate::runnable::Runnable;

/// Two `Runnable`s composed end-to-end. `RunnableSequence<I, M, O>` reads
/// "input I → middle M → output O". Created with `.pipe()`.
pub struct RunnableSequence<I, M, O>
where
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
{
    first: Arc<dyn Runnable<I, M>>,
    second: Arc<dyn Runnable<M, O>>,
}

impl<I, M, O> RunnableSequence<I, M, O>
where
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
{
    /// Pair two type-erased `Runnable`s. Most callers go through
    /// `RunnableExt::pipe` instead of constructing this directly.
    pub fn new(first: Arc<dyn Runnable<I, M>>, second: Arc<dyn Runnable<M, O>>) -> Self {
        Self { first, second }
    }
}

#[async_trait::async_trait]
impl<I, M, O> Runnable<I, O> for RunnableSequence<I, M, O>
where
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        let mid = self.first.invoke(input, ctx).await?;
        self.second.invoke(mid, ctx).await
    }
}
