//! [`FinalizingStream`] — Stream wrapper with deterministic Drop
//! cleanup.
//!
//! `async_stream::stream! { … }` produces a state-machine future
//! whose locals (including RAII guards declared inside the macro
//! body) are released when the stream is dropped. That gives the
//! correctness we need — `_guard` runs `Drop` on early-cancel — but
//! it does *not* give us a hook to observe the cancellation from
//! outside the macro. Long-running graph streams like
//! `CompiledGraph::build_stream` benefit from a structured "the
//! consumer walked away" signal so operators can correlate
//! abandoned requests with backend resource use.
//!
//! `FinalizingStream` wraps any `Stream` and runs a finalizer
//! closure exactly once on Drop, but only when the inner stream
//! hasn't already produced a terminal `None`. The wrapper uses
//! [`pin_project_lite`] so the finalizer can safely access the
//! `done` flag through the pinned projection — no `unsafe`.
//!
//! ```rust,ignore
//! let stream = FinalizingStream::new(inner, || {
//!     tracing::debug!("graph stream dropped before completion");
//! });
//! ```

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;

pin_project! {
    /// Stream that fires `finalize` exactly once when dropped before
    /// the inner stream returned `None`. Normal completion (the
    /// inner stream yielded `None` from `poll_next`) is *not* an
    /// early-cancel — the finalizer is silently skipped.
    pub struct FinalizingStream<St, F>
    where
        F: FnOnce(),
    {
        #[pin]
        inner: St,
        done: bool,
        finalize: Option<F>,
    }

    impl<St, F> PinnedDrop for FinalizingStream<St, F>
    where
        F: FnOnce(),
    {
        fn drop(this: Pin<&mut Self>) {
            let proj = this.project();
            if !*proj.done && let Some(f) = proj.finalize.take() {
                f();
            }
        }
    }
}

impl<St, F> FinalizingStream<St, F>
where
    F: FnOnce(),
{
    /// Wrap `inner`, scheduling `finalize` to run on early Drop.
    pub const fn new(inner: St, finalize: F) -> Self {
        Self {
            inner,
            done: false,
            finalize: Some(finalize),
        }
    }
}

impl<St, F> Stream for FinalizingStream<St, F>
where
    St: Stream,
    F: FnOnce(),
{
    type Item = St::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let proj = self.project();
        // Stream contract: once we return `Ready(None)` the stream is
        // "fused" — every subsequent poll must also return `None`
        // and must not observe the inner stream again. Honouring
        // this invariant explicitly stops the wrapper from leaking
        // post-completion polls through to a `Stream` impl that
        // does not handle them defensively (some hand-rolled
        // implementations panic in that case).
        if *proj.done {
            return Poll::Ready(None);
        }
        match proj.inner.poll_next(cx) {
            Poll::Ready(None) => {
                *proj.done = true;
                // Mute the finalizer — completion is not cancellation.
                proj.finalize.take();
                Poll::Ready(None)
            }
            other => other,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            self.inner.size_hint()
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::StreamExt;
    use futures::stream;

    use super::*;

    fn finalizer(counter: &Arc<AtomicUsize>) -> impl FnOnce() + use<> {
        let counter = Arc::clone(counter);
        move || {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn finalizer_does_not_fire_on_normal_completion() {
        let counter = Arc::new(AtomicUsize::new(0));
        let inner = stream::iter(vec![1, 2, 3]);
        let mut s = FinalizingStream::new(inner, finalizer(&counter));
        while s.next().await.is_some() {}
        drop(s);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn finalizer_fires_on_early_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let inner = stream::iter(0..1000);
        let mut s = FinalizingStream::new(inner, finalizer(&counter));
        // Pull just one item.
        let _ = s.next().await;
        drop(s);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn finalizer_fires_on_drop_without_polling() {
        let counter = Arc::new(AtomicUsize::new(0));
        let inner = stream::iter(0..10);
        let s = FinalizingStream::new(inner, finalizer(&counter));
        drop(s);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn poll_after_completion_returns_none_without_polling_inner() {
        // Build an inner stream that panics if polled twice past
        // its end — the wrapper must `fuse` the stream after the
        // first `Ready(None)` and never poll inner again.
        struct PanicAfterNone {
            yielded: bool,
            ended: bool,
        }
        impl Stream for PanicAfterNone {
            type Item = u32;
            fn poll_next(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                if !self.yielded {
                    self.yielded = true;
                    Poll::Ready(Some(7))
                } else if !self.ended {
                    self.ended = true;
                    Poll::Ready(None)
                } else {
                    panic!("inner stream polled past completion");
                }
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let mut s = FinalizingStream::new(
            PanicAfterNone {
                yielded: false,
                ended: false,
            },
            finalizer(&counter),
        );
        assert_eq!(s.next().await, Some(7));
        assert_eq!(s.next().await, None);
        // Wrapper must absorb this without panicking the inner.
        assert_eq!(s.next().await, None);
        assert_eq!(s.next().await, None);
    }

    #[tokio::test]
    async fn finalizer_runs_at_most_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        // Stream that yields one item then ends.
        let inner = stream::iter(vec![1]);
        let mut s = FinalizingStream::new(inner, finalizer(&counter));
        let _ = s.next().await;
        let _ = s.next().await; // returns None — completion path
        drop(s);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "completion suppresses finalizer"
        );
    }
}
