//! Closure-backed `Runnable`. Lets users drop arbitrary async logic into a
//! pipeline without defining a new type.

use std::sync::Arc;

use entelix_core::{ExecutionContext, Result};
use futures::future::BoxFuture;

use crate::runnable::Runnable;

type LambdaFn<I, O> = dyn Fn(I, ExecutionContext) -> BoxFuture<'static, Result<O>> + Send + Sync;

/// `Runnable<I, O>` backed by a user-supplied async closure.
///
/// The closure receives the input and an owned `ExecutionContext` (cheaply
/// `Clone`). It must return a future that is `Send + 'static` — common in
/// practice when the closure body uses owned data and `tokio` primitives.
pub struct RunnableLambda<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    inner: Arc<LambdaFn<I, O>>,
}

impl<I, O> RunnableLambda<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Wrap an async closure into a `Runnable`.
    ///
    /// ```ignore
    /// let double = RunnableLambda::new(|x: i32, _ctx| async move { Ok(x * 2) });
    /// ```
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn(I, ExecutionContext) -> Fut + Send + Sync + 'static,
        Fut: core::future::Future<Output = Result<O>> + Send + 'static,
    {
        Self {
            inner: Arc::new(move |input, ctx| Box::pin(f(input, ctx))),
        }
    }
}

impl<I, O> Clone for RunnableLambda<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[async_trait::async_trait]
impl<I, O> Runnable<I, O> for RunnableLambda<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        (self.inner)(input, ctx.clone()).await
    }
}
