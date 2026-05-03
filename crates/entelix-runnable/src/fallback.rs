//! `Fallback<R, F>` — `Runnable<I, O>` adapter that, on a transient
//! error from the primary runnable, attempts an ordered list of
//! fallbacks.
//!
//! Reuses the same
//! [`RetryClassifier`](entelix_core::transports::RetryClassifier) as
//! `Retrying`: whether an error is "fallback-eligible" is the same
//! question as "retryable" (transport / 5xx / 429 = transient,
//! permanent failures stay permanent). One trait, two adapters.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use entelix_core::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::transports::{DefaultRetryClassifier, RetryClassifier};

use crate::runnable::Runnable;

/// `Runnable<I, O>` adapter that falls back through a list of
/// alternatives when the primary returns a classifier-approved
/// transient error.
///
/// Cancellation is honoured between attempts — pulling the rug
/// mid-sequence returns `Error::Cancelled` rather than continuing.
pub struct Fallback<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    primary: Arc<R>,
    fallbacks: Vec<Arc<F>>,
    classifier: Arc<dyn RetryClassifier>,
    _io: PhantomData<fn(I) -> O>,
}

impl<R, F, I, O> Fallback<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    /// Build with the primary and an ordered list of fallbacks.
    /// Empty `fallbacks` degrades to the primary alone — the adapter
    /// still type-checks and behaves identically to the inner.
    pub fn new(primary: R, fallbacks: Vec<F>) -> Self {
        Self {
            primary: Arc::new(primary),
            fallbacks: fallbacks.into_iter().map(Arc::new).collect(),
            classifier: Arc::new(DefaultRetryClassifier),
            _io: PhantomData,
        }
    }

    /// Override the classifier — useful when the operator has a
    /// stricter or laxer "transient" definition for fallback purposes
    /// than for retry purposes.
    #[must_use]
    pub fn with_classifier(mut self, classifier: Arc<dyn RetryClassifier>) -> Self {
        self.classifier = classifier;
        self
    }
}

impl<R, F, I, O> std::fmt::Debug for Fallback<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fallback")
            .field("fallback_count", &self.fallbacks.len())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<R, F, I, O> Runnable<I, O> for Fallback<R, F, I, O>
where
    R: Runnable<I, O> + 'static,
    F: Runnable<I, O> + 'static,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut attempt: u32 = 0;
        let primary_result = self.primary.invoke(input.clone(), ctx).await;
        let mut last_err = match primary_result {
            Ok(value) => return Ok(value),
            Err(err) => {
                // Fallbacks ignore the `Retry-After` hint — the
                // policy is "try the next replica now", not "wait
                // and retry the same one".
                if !self.classifier.should_retry(&err, attempt).retry {
                    return Err(err);
                }
                err
            }
        };
        for fallback in &self.fallbacks {
            attempt = attempt.saturating_add(1);
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            match fallback.invoke(input.clone(), ctx).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if !self.classifier.should_retry(&err, attempt).retry {
                        return Err(err);
                    }
                    last_err = err;
                }
            }
        }
        Err(last_err)
    }
}
