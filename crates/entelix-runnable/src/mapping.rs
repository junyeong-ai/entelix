//! `Mapping<R, F>` — `Runnable<I, P>` adapter that pipes the inner's
//! output through a synchronous `Fn(O) -> P`.
//!
//! Equivalent to `inner.pipe(RunnableLambda::new(|o, _| async move {
//! Ok(f(o)) }))` but avoids allocating a wrapping `RunnableLambda` and
//! keeps the closure non-async at the type level.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use entelix_core::ExecutionContext;
use entelix_core::error::Result;

use crate::runnable::Runnable;

/// `Runnable<I, P>` adapter applying a pure synchronous function to
/// the inner's output.
pub struct Mapping<R, F, I, O, P>
where
    R: Runnable<I, O> + 'static,
    F: Fn(O) -> P + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    P: Send + 'static,
{
    inner: Arc<R>,
    f: Arc<F>,
    _io: PhantomData<fn(I) -> (O, P)>,
}

impl<R, F, I, O, P> Mapping<R, F, I, O, P>
where
    R: Runnable<I, O> + 'static,
    F: Fn(O) -> P + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    P: Send + 'static,
{
    /// Build with the inner runnable and a pure mapping function.
    pub fn new(inner: R, f: F) -> Self {
        Self {
            inner: Arc::new(inner),
            f: Arc::new(f),
            _io: PhantomData,
        }
    }
}

impl<R, F, I, O, P> std::fmt::Debug for Mapping<R, F, I, O, P>
where
    R: Runnable<I, O> + 'static,
    F: Fn(O) -> P + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    P: Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapping").finish_non_exhaustive()
    }
}

#[async_trait]
impl<R, F, I, O, P> Runnable<I, P> for Mapping<R, F, I, O, P>
where
    R: Runnable<I, O> + 'static,
    F: Fn(O) -> P + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    P: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<P> {
        let value = self.inner.invoke(input, ctx).await?;
        Ok((self.f)(value))
    }
}
