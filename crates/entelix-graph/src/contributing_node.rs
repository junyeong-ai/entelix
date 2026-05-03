//! `ContributingNodeAdapter` — wraps a `Runnable<S, S::Contribution>`
//! whose output names exactly the slots the node touched, and folds
//! it into the inbound state through
//! [`StateMerge::merge_contribution`](crate::StateMerge::merge_contribution).
//!
//! Companion to [`MergeNodeAdapter`](crate::MergeNodeAdapter): the
//! latter takes a per-graph closure for arbitrary merge logic;
//! this one is type-driven via the state's
//! [`StateMerge`](crate::StateMerge) impl, with the
//! `<Name>Contribution` companion struct from
//! `entelix-graph-derive` providing the per-slot `Option`-wrapped
//! shape.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_runnable::Runnable;

use crate::reducer::StateMerge;

/// `Runnable<S, S>` adapter that snapshots the inbound state, runs
/// an inner `Runnable<S, S::Contribution>` to produce only the
/// slots the node touched, and folds that contribution into the
/// snapshot via `S::merge_contribution`.
pub struct ContributingNodeAdapter<S>
where
    S: StateMerge + Clone + Send + Sync + 'static,
{
    inner: Arc<dyn Runnable<S, S::Contribution>>,
}

impl<S> ContributingNodeAdapter<S>
where
    S: StateMerge + Clone + Send + Sync + 'static,
{
    /// Wrap `inner`. The inner runnable's output is
    /// `S::Contribution` — typically built via
    /// `S::Contribution::default().with_<field>(…)` chained per
    /// slot the node intends to write.
    pub fn new<R>(inner: R) -> Self
    where
        R: Runnable<S, S::Contribution> + 'static,
    {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<S> std::fmt::Debug for ContributingNodeAdapter<S>
where
    S: StateMerge + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContributingNodeAdapter")
            .field("inner", &"<runnable>")
            .finish()
    }
}

#[async_trait]
impl<S> Runnable<S, S> for ContributingNodeAdapter<S>
where
    S: StateMerge + Clone + Send + Sync + 'static,
{
    async fn invoke(&self, input: S, ctx: &ExecutionContext) -> Result<S> {
        let snapshot = input.clone();
        let contribution = self.inner.invoke(input, ctx).await?;
        Ok(snapshot.merge_contribution(contribution))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use entelix_runnable::RunnableLambda;

    use super::*;
    use crate::reducer::{Annotated, Append, Max};

    #[derive(Clone, Debug, Default)]
    struct AgentState {
        log: Annotated<Vec<String>, Append<String>>,
        score: Annotated<i32, Max<i32>>,
        tag: String,
    }

    #[derive(Default)]
    struct AgentStateContribution {
        log: Option<Annotated<Vec<String>, Append<String>>>,
        score: Option<Annotated<i32, Max<i32>>>,
        tag: Option<String>,
    }

    impl StateMerge for AgentState {
        type Contribution = AgentStateContribution;

        fn merge(self, update: Self) -> Self {
            Self {
                log: self.log.merge(update.log),
                score: self.score.merge(update.score),
                tag: update.tag,
            }
        }

        fn merge_contribution(self, c: Self::Contribution) -> Self {
            Self {
                log: match c.log {
                    Some(v) => self.log.merge(v),
                    None => self.log,
                },
                score: match c.score {
                    Some(v) => self.score.merge(v),
                    None => self.score,
                },
                tag: c.tag.unwrap_or(self.tag),
            }
        }
    }

    fn seed(log: Vec<&str>, score: i32, tag: &str) -> AgentState {
        AgentState {
            log: Annotated::new(log.into_iter().map(String::from).collect(), Append::new()),
            score: Annotated::new(score, Max::new()),
            tag: tag.into(),
        }
    }

    #[tokio::test]
    async fn contributing_adapter_writes_only_named_slots() {
        // Node writes log + tag, leaves score untouched. Score
        // contribution is `None`, so the score slot's *current*
        // value (80) survives — the LangGraph TypedDict semantic
        // that the simpler "always-merge" approach can't express
        // without per-slot intent.
        let node = RunnableLambda::new(|_input: AgentState, _ctx| async move {
            Ok::<_, _>(AgentStateContribution {
                log: Some(Annotated::new(vec!["new entry".into()], Append::new())),
                score: None,
                tag: Some("after".into()),
            })
        });
        let adapter = ContributingNodeAdapter::new(node);

        let initial = seed(vec!["seed"], 80, "before");
        let merged = adapter
            .invoke(initial, &ExecutionContext::new())
            .await
            .unwrap();

        assert_eq!(
            merged.log.value,
            vec!["seed".to_owned(), "new entry".into()]
        );
        // Score: untouched — score=80 keeps regardless of any
        // default-zero contribution might have implied.
        assert_eq!(merged.score.value, 80);
        assert_eq!(merged.tag, "after");
    }

    #[tokio::test]
    async fn contributing_adapter_unwritten_field_keeps_negative_current_value() {
        // Regression for the default-overrides edge case: a node
        // that doesn't touch `score` but returns a contribution
        // with `score: None` must keep the current value `-100`,
        // NOT collapse it via Max(current, default=0).
        let node = RunnableLambda::new(|_input: AgentState, _ctx| async move {
            Ok::<_, _>(AgentStateContribution::default())
        });
        let adapter = ContributingNodeAdapter::new(node);

        let initial = seed(vec![], -100, "x");
        let merged = adapter
            .invoke(initial, &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(merged.score.value, -100, "no contribution must leave value");
    }

    #[tokio::test]
    async fn contributing_adapter_propagates_inner_error() {
        let node = RunnableLambda::new(|_input: AgentState, _ctx| async move {
            Err::<AgentStateContribution, _>(entelix_core::error::Error::invalid_request("nope"))
        });
        let adapter = ContributingNodeAdapter::new(node);
        let err = adapter
            .invoke(seed(vec![], 0, ""), &ExecutionContext::new())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("nope"));
    }
}
