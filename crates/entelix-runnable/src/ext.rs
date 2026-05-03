//! Extension trait providing `.pipe()` and the standard composition
//! adapters on every `Runnable<I, O>`.

use std::sync::Arc;
use std::time::Duration;

use entelix_core::transports::RetryPolicy;
use entelix_core::{ExecutionContext, Result};

use crate::configured::Configured;
use crate::fallback::Fallback;
use crate::mapping::Mapping;
use crate::retrying::Retrying;
use crate::runnable::Runnable;
use crate::sequence::RunnableSequence;
use crate::stream::{BoxStream, StreamChunk, StreamMode};
use crate::timed::Timed;

/// Ergonomic composition surface, blanket-implemented for every
/// `Runnable<I, O>`.
///
/// Every method returns a concrete `Runnable<I, O>` — composition
/// stays zero-cost in the steady state. Boxing happens only at the
/// explicit `erase()` boundary
/// ([`AnyRunnable`](crate::any_runnable::AnyRunnable), F12).
#[async_trait::async_trait]
pub trait RunnableExt<I, O>: Runnable<I, O> + Sized + 'static
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Chain this runnable into `next`. The output `O` of `self`
    /// becomes the input of `next`, producing a `Runnable<I, P>`.
    ///
    /// ```ignore
    /// let chain = prompt.pipe(model).pipe(parser);
    /// ```
    fn pipe<P, R>(self, next: R) -> RunnableSequence<I, O, P>
    where
        P: Send + 'static,
        R: Runnable<O, P> + 'static,
    {
        RunnableSequence::new(Arc::new(self), Arc::new(next))
    }

    /// Wrap `self` with retry semantics. The returned runnable
    /// re-invokes the inner on transient errors per the `policy`.
    /// The input must be `Clone` because each retry receives a
    /// fresh copy.
    ///
    /// ```ignore
    /// let resilient = model.with_retry(RetryPolicy::standard());
    /// ```
    fn with_retry(self, policy: RetryPolicy) -> Retrying<Self, I, O>
    where
        I: Clone,
    {
        Retrying::new(self, policy)
    }

    /// Wrap `self` with an ordered fallback chain. On a transient
    /// error from the primary, the adapter tries each fallback in
    /// turn. Permanent errors surface immediately. The classifier
    /// is the same trait used by [`Self::with_retry`] —
    /// `entelix_core::transports::DefaultRetryClassifier` by default.
    ///
    /// ```ignore
    /// let resilient = primary.with_fallbacks(vec![secondary, tertiary]);
    /// ```
    fn with_fallbacks<F>(self, fallbacks: Vec<F>) -> Fallback<Self, F, I, O>
    where
        F: Runnable<I, O> + 'static,
        I: Clone,
    {
        Fallback::new(self, fallbacks)
    }

    /// Map the inner's output through a pure synchronous function.
    /// Equivalent to piping into a `RunnableLambda` but skipping the
    /// async wrapper.
    ///
    /// ```ignore
    /// let lengths = strings.map(|s: String| s.len());
    /// ```
    fn map<F, P>(self, f: F) -> Mapping<Self, F, I, O, P>
    where
        F: Fn(O) -> P + Send + Sync + 'static,
        P: Send + 'static,
    {
        Mapping::new(self, f)
    }

    /// Run `configurer` on a cloned [`ExecutionContext`] before
    /// delegating to the inner. The caller's `ctx` is not mutated.
    ///
    /// ```ignore
    /// let with_short_deadline = inner.with_config(|ctx| {
    ///     // ctx mutations apply only to the inner invocation
    /// });
    /// ```
    fn with_config<F>(self, configurer: F) -> Configured<Self, F, I, O>
    where
        F: Fn(&mut ExecutionContext) + Send + Sync + 'static,
    {
        Configured::new(self, configurer)
    }

    /// Race the inner against a wall-clock timeout. On expiry the
    /// adapter returns
    /// [`Error::DeadlineExceeded`](entelix_core::Error::DeadlineExceeded);
    /// caller cancellation still wins.
    ///
    /// ```ignore
    /// let bounded = inner.with_timeout(Duration::from_secs(30));
    /// ```
    fn with_timeout(self, timeout: Duration) -> Timed<Self, I, O> {
        Timed::new(self, timeout)
    }

    /// Convenience wrapper around [`Runnable::stream`] — same
    /// arguments, no trait import needed at the call site.
    async fn stream_with(
        &self,
        input: I,
        mode: StreamMode,
        ctx: &ExecutionContext,
    ) -> Result<BoxStream<'_, Result<StreamChunk<O>>>> {
        self.stream(input, mode, ctx).await
    }
}

impl<T, I, O> RunnableExt<I, O> for T
where
    T: Runnable<I, O> + Sized + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
}
