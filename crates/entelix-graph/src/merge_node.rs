//! `MergeNodeAdapter` — wraps a delta-producing `Runnable<S, U>` and
//! a merger closure into a `Runnable<S, S>` that fits unchanged into
//! the existing StateGraph node contract.
//!
//! The current StateGraph contract has each node implement
//! `Runnable<S, S>` and own its full-state replace logic. That works
//! but forces every node closure to thread the unchanged fields
//! through itself manually. LangGraph users coming from Python
//! expect a "delta-style" alternative where a node returns only its
//! contribution and the runtime merges it into the surrounding
//! state.
//!
//! `MergeNodeAdapter<S, U, F>` provides that ergonomic without
//! changing the node contract. It snapshots the inbound state,
//! runs the inner runnable to get an update of arbitrary type `U`,
//! and applies the user-supplied merger
//! `Fn(state: S, update: U) -> Result<S>` to produce the next
//! full state. Existing `add_node` (full-state replace) and the new
//! `add_node_with` (delta + merger) coexist.
//!
//! The merger has full access to both the inbound state and the
//! delta, so it composes naturally with the [`Reducer<T>`](crate::Reducer)
//! impls already shipped — the closure body is the place to call
//! `Append::<U>::new().reduce(...)` per field.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_runnable::Runnable;

/// `Runnable<S, S>` that runs an inner `Runnable<S, U>` and merges
/// the resulting `U` back into a fresh copy of the inbound `S` via
/// the supplied closure.
pub struct MergeNodeAdapter<S, U, F>
where
    S: Clone + Send + Sync + 'static,
    U: Send + Sync + 'static,
    F: Fn(S, U) -> Result<S> + Send + Sync + 'static,
{
    inner: Arc<dyn Runnable<S, U>>,
    merger: F,
}

impl<S, U, F> MergeNodeAdapter<S, U, F>
where
    S: Clone + Send + Sync + 'static,
    U: Send + Sync + 'static,
    F: Fn(S, U) -> Result<S> + Send + Sync + 'static,
{
    /// Wrap `inner` with the supplied merger.
    pub fn new<R>(inner: R, merger: F) -> Self
    where
        R: Runnable<S, U> + 'static,
    {
        Self {
            inner: Arc::new(inner),
            merger,
        }
    }
}

impl<S, U, F> std::fmt::Debug for MergeNodeAdapter<S, U, F>
where
    S: Clone + Send + Sync + 'static,
    U: Send + Sync + 'static,
    F: Fn(S, U) -> Result<S> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeNodeAdapter")
            .field("inner", &"<runnable>")
            .field("merger", &"<closure>")
            .finish()
    }
}

#[async_trait]
impl<S, U, F> Runnable<S, S> for MergeNodeAdapter<S, U, F>
where
    S: Clone + Send + Sync + 'static,
    U: Send + Sync + 'static,
    F: Fn(S, U) -> Result<S> + Send + Sync + 'static,
{
    async fn invoke(&self, input: S, ctx: &ExecutionContext) -> Result<S> {
        // Clone *before* invoking the inner runnable so the merger
        // sees the pre-call state regardless of what the inner
        // runnable does with its argument. Cheap when S is `Arc`-
        // backed; explicit and predictable for everything else.
        let snapshot = input.clone();
        let update = self.inner.invoke(input, ctx).await?;
        (self.merger)(snapshot, update)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use entelix_core::error::Error;
    use entelix_runnable::RunnableLambda;

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct State {
        log: Vec<String>,
        counter: u32,
    }

    /// Delta-style update: a node only produces its new log entries
    /// and a counter increment, not the full state.
    #[derive(Clone, Debug)]
    struct PlanDelta {
        new_entries: Vec<String>,
        increment: u32,
    }

    #[tokio::test]
    async fn merger_combines_state_with_delta() {
        let planner = RunnableLambda::new(|s: State, _ctx| async move {
            Ok::<_, _>(PlanDelta {
                new_entries: vec![format!("planned at counter={}", s.counter)],
                increment: 1,
            })
        });
        let adapter = MergeNodeAdapter::new(planner, |mut state: State, update: PlanDelta| {
            state.log.extend(update.new_entries);
            state.counter += update.increment;
            Ok(state)
        });

        let initial = State {
            log: vec!["seed".into()],
            counter: 10,
        };
        let result = adapter
            .invoke(initial, &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(
            result.log,
            vec!["seed".to_owned(), "planned at counter=10".to_owned()]
        );
        assert_eq!(result.counter, 11);
    }

    #[tokio::test]
    async fn merger_can_fail_and_propagate_error() {
        let planner = RunnableLambda::new(|_s: State, _ctx| async move {
            Ok::<_, _>(PlanDelta {
                new_entries: Vec::new(),
                increment: 0,
            })
        });
        let adapter = MergeNodeAdapter::new(planner, |_state: State, _update: PlanDelta| {
            Err(Error::invalid_request("merger refused"))
        });

        let err = adapter
            .invoke(
                State {
                    log: Vec::new(),
                    counter: 0,
                },
                &ExecutionContext::new(),
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("merger refused"));
    }

    #[tokio::test]
    async fn inner_failure_short_circuits_before_merger() {
        let merger_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let merger_calls_clone = Arc::clone(&merger_calls);

        let planner = RunnableLambda::new(|_s: State, _ctx| async move {
            Err::<PlanDelta, _>(Error::invalid_request("planner failed"))
        });
        let adapter = MergeNodeAdapter::new(planner, move |state: State, _update: PlanDelta| {
            merger_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(state)
        });

        let err = adapter
            .invoke(
                State {
                    log: Vec::new(),
                    counter: 0,
                },
                &ExecutionContext::new(),
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("planner failed"));
        assert_eq!(
            merger_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "merger must not run when inner runnable fails"
        );
    }
}
