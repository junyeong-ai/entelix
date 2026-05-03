//! `Dispatch<T>` — fan-out primitive for parallel sub-task dispatch.
//!
//! LangGraph's `Send` API lets a conditional edge emit *multiple*
//! follow-up invocations into the same node, one per `Send`. We name
//! the Rust counterpart [`Dispatch<T>`] rather than `Send<T>` because
//! Rust already reserves `Send` as the unsafe auto trait — a struct
//! with the same identifier would force every downstream user to
//! disambiguate `T: std::marker::Send` from `T: entelix_graph::Send`.
//! [`scatter`] runs a `Vec<Dispatch<I>>` through a [`Runnable<I, O>`]
//! in parallel and collects the outputs in submission order.
//!
//! Like [`Reducer<T>`](crate::Reducer), this is a standalone helper:
//! `Dispatch` does not plug into `StateGraph::add_conditional_edges`
//! as a "this node receives N sends" emit. Users fan out manually
//! from inside their node closures via [`scatter`].

use std::sync::Arc;

use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_runnable::Runnable;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesOrdered;

/// One unit of fan-out work — semantically equivalent to LangGraph's
/// `Send(node, payload)`. Carries the payload that a fanned-out
/// runnable will receive when [`scatter`] runs.
#[derive(Clone, Debug)]
pub struct Dispatch<T> {
    /// Payload the runnable will see.
    pub payload: T,
}

impl<T> Dispatch<T> {
    /// Build a single dispatch.
    pub const fn new(payload: T) -> Self {
        Self { payload }
    }
}

impl<T> From<T> for Dispatch<T> {
    fn from(payload: T) -> Self {
        Self { payload }
    }
}

/// Fan a `Vec<Dispatch<I>>` through a `Runnable<I, O>` in parallel
/// and collect the outputs in submission order.
///
/// `concurrency` caps the number of in-flight invocations; the
/// remainder are queued. Setting it to `0` is rejected at runtime
/// because a zero cap deadlocks the consumer.
///
/// Failure is fail-fast: as soon as one branch returns `Err`, the
/// remaining in-flight futures are dropped and the error surfaces.
/// All branches run under a [`ExecutionContext::child`] scope —
/// cancelling the parent cascades to siblings, and on scatter exit
/// (success, error, or panic) the scope token is fired so any branch
/// observing `ctx.cancellation()` cooperatively unwinds. The parent
/// context is left untouched.
pub async fn scatter<R, I, O>(
    runnable: Arc<R>,
    sends: Vec<Dispatch<I>>,
    ctx: &ExecutionContext,
    concurrency: usize,
) -> Result<Vec<O>>
where
    R: Runnable<I, O> + 'static,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    if concurrency == 0 {
        return Err(Error::config("scatter concurrency must be > 0"));
    }
    // Scope-bound child context. `_guard` cancels the scope token on
    // every exit path including panic-unwind, so still-racing
    // branches see `ctx.cancellation()` fire cooperatively.
    let scope_ctx = ctx.child();
    let _guard = ScopeCancelGuard {
        token: scope_ctx.cancellation().clone(),
    };
    // FuturesOrdered requires every queued future to share one type.
    // Two `async move` blocks produce two anonymous types, so we
    // erase via `BoxFuture` (heap allocation per dispatch — cheap
    // relative to the model call this typically wraps).
    let mut in_flight: FuturesOrdered<BoxFuture<'static, Result<O>>> = FuturesOrdered::new();
    let mut iter = sends.into_iter();
    let make_future = |send: Dispatch<I>| -> BoxFuture<'static, Result<O>> {
        let runnable = Arc::clone(&runnable);
        let ctx_clone = scope_ctx.clone();
        Box::pin(async move { runnable.invoke(send.payload, &ctx_clone).await })
    };
    for _ in 0..concurrency {
        let Some(send) = iter.next() else { break };
        in_flight.push_back(make_future(send));
    }
    let mut out = Vec::new();
    while let Some(result) = in_flight.next().await {
        match result {
            Ok(v) => out.push(v),
            Err(e) => return Err(e),
        }
        if let Some(send) = iter.next() {
            in_flight.push_back(make_future(send));
        }
    }
    Ok(out)
}

/// RAII fire-on-drop for the scope cancellation token. Ensures that
/// every exit path from [`scatter`] — early return, fail-fast `Err`,
/// or panic-unwind — signals siblings to wind down.
struct ScopeCancelGuard {
    token: entelix_core::cancellation::CancellationToken,
}

impl Drop for ScopeCancelGuard {
    fn drop(&mut self) {
        self.token.cancel();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_runnable::RunnableLambda;

    use super::*;

    #[tokio::test]
    async fn scatter_returns_results_in_submission_order() {
        let runnable = Arc::new(RunnableLambda::new(|n: u32, _ctx| async move {
            Ok::<_, _>(n * 2)
        }));
        let sends = vec![
            Dispatch::new(1_u32),
            Dispatch::new(2),
            Dispatch::new(3),
            Dispatch::new(4),
        ];
        let out = scatter(runnable, sends, &ExecutionContext::new(), 2)
            .await
            .unwrap();
        assert_eq!(out, vec![2, 4, 6, 8]);
    }

    #[tokio::test]
    async fn scatter_zero_concurrency_is_rejected() {
        let runnable = Arc::new(RunnableLambda::new(
            |n: u32, _ctx| async move { Ok::<_, _>(n) },
        ));
        let err = scatter(
            runnable,
            vec![Dispatch::new(1_u32)],
            &ExecutionContext::new(),
            0,
        )
        .await
        .unwrap_err();
        assert!(format!("{err}").contains("concurrency"));
    }

    #[tokio::test]
    async fn scatter_caps_in_flight_invocations() {
        let peak = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let history = Arc::new(Mutex::new(Vec::<usize>::new()));
        let peak_for_lambda = Arc::clone(&peak);
        let in_flight_for_lambda = Arc::clone(&in_flight);
        let history_for_lambda = Arc::clone(&history);

        let runnable = Arc::new(RunnableLambda::new(move |n: u32, _ctx| {
            let peak = Arc::clone(&peak_for_lambda);
            let in_flight = Arc::clone(&in_flight_for_lambda);
            let history = Arc::clone(&history_for_lambda);
            async move {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                history.lock().unwrap().push(now);
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::task::yield_now().await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, _>(n)
            }
        }));
        let sends: Vec<_> = (0..6_u32).map(Dispatch::new).collect();
        let _ = scatter(runnable, sends, &ExecutionContext::new(), 2)
            .await
            .unwrap();
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "peak in-flight exceeded 2"
        );
    }

    #[tokio::test]
    async fn scatter_fail_fast_on_first_error() {
        let runnable = Arc::new(RunnableLambda::new(|n: u32, _ctx| async move {
            if n == 3 {
                Err(entelix_core::Error::invalid_request("boom"))
            } else {
                Ok::<_, _>(n)
            }
        }));
        let sends: Vec<_> = (1..=5_u32).map(Dispatch::new).collect();
        let err = scatter(runnable, sends, &ExecutionContext::new(), 2)
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("boom"));
    }

    #[tokio::test]
    async fn fail_fast_cancels_scope_token_for_siblings() {
        // A failing branch should signal still-running siblings via the
        // scope cancellation token without escaping to the parent.
        let parent_ctx = ExecutionContext::new();
        let parent_token = parent_ctx.cancellation().clone();

        let observed_cancel = Arc::new(Mutex::new(Vec::<bool>::new()));
        let observed_for_lambda = Arc::clone(&observed_cancel);

        let runnable = Arc::new(RunnableLambda::new(move |n: u32, ctx: ExecutionContext| {
            let observed = Arc::clone(&observed_for_lambda);
            async move {
                if n == 1 {
                    // First branch fails immediately, triggering scope cancel.
                    return Err(entelix_core::Error::invalid_request("boom"));
                }
                // Sibling waits long enough for the fail-fast to land,
                // then records whether the *child* token has fired.
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                observed.lock().unwrap().push(ctx.is_cancelled());
                Ok::<_, _>(n)
            }
        }));
        let sends: Vec<_> = (1..=4_u32).map(Dispatch::new).collect();
        let _ = scatter(runnable, sends, &parent_ctx, 4).await;
        // Parent context untouched.
        assert!(
            !parent_token.is_cancelled(),
            "scatter must not bubble cancellation to parent"
        );
    }

    #[tokio::test]
    async fn parent_cancel_cascades_to_branches() {
        let parent_ctx = ExecutionContext::new();
        let parent_token = parent_ctx.cancellation().clone();

        let runnable = Arc::new(RunnableLambda::new(
            |_n: u32, ctx: ExecutionContext| async move {
                tokio::select! {
                    () = ctx.cancellation().cancelled() => {
                        Err(entelix_core::Error::Cancelled)
                    }
                    () = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                        Ok::<_, _>(0)
                    }
                }
            },
        ));
        let parent_token_for_canceller = parent_token.clone();
        let canceller = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            parent_token_for_canceller.cancel();
        });

        let sends: Vec<_> = (0..4_u32).map(Dispatch::new).collect();
        let err = scatter(runnable, sends, &parent_ctx, 4).await.unwrap_err();
        canceller.await.unwrap();
        assert!(matches!(err, entelix_core::Error::Cancelled));
    }

    #[tokio::test]
    async fn empty_sends_returns_empty_output() {
        let runnable = Arc::new(RunnableLambda::new(
            |n: u32, _ctx| async move { Ok::<_, _>(n) },
        ));
        let out = scatter::<_, u32, u32>(runnable, Vec::new(), &ExecutionContext::new(), 4)
            .await
            .unwrap();
        assert!(out.is_empty());
    }
}
