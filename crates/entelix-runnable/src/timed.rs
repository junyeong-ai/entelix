//! `Timed<R>` — `Runnable<I, O>` adapter that races the inner against
//! a wall-clock timeout.
//!
//! On expiry returns [`Error::DeadlineExceeded`]. Cancellation by the
//! caller's [`ExecutionContext`] still wins — the timeout sleep is
//! cancellation-aware.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use entelix_core::ExecutionContext;
use entelix_core::error::{Error, Result};

use crate::runnable::Runnable;

/// `Runnable<I, O>` adapter that aborts the inner with
/// `Error::DeadlineExceeded` if it does not complete within
/// `timeout`.
pub struct Timed<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    inner: Arc<R>,
    timeout: Duration,
    _io: PhantomData<fn(I) -> O>,
}

impl<R, I, O> Timed<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Build with the inner runnable and a wall-clock timeout.
    pub fn new(inner: R, timeout: Duration) -> Self {
        Self {
            inner: Arc::new(inner),
            timeout,
            _io: PhantomData,
        }
    }
}

impl<R, I, O> std::fmt::Debug for Timed<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timed")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<R, I, O> Runnable<I, O> for Timed<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        let token = ctx.cancellation();
        tokio::select! {
            biased;
            // 1. Caller cancellation wins — operator intent overrides
            //    the timeout.
            () = token.cancelled() => Err(Error::Cancelled),
            // 2. Timeout fires.
            () = tokio::time::sleep(self.timeout) => Err(Error::DeadlineExceeded),
            // 3. Inner completes.
            result = self.inner.invoke(input, ctx) => result,
        }
    }
}
