//! `RunnableParallel` — fan-out one input to many runnables, collect outputs
//! into a `HashMap<String, O>`.
//!
//! Branches run concurrently via `futures::future::try_join_all` — first
//! failure aborts the others. The input must be `Clone` because each branch
//! receives its own copy.

use std::collections::HashMap;
use std::sync::Arc;

use entelix_core::{ExecutionContext, Result};
use futures::future::try_join_all;

use crate::runnable::Runnable;

type Branch<I, O> = (String, Arc<dyn Runnable<I, O>>);

/// `Runnable<I, HashMap<String, O>>` that runs every registered branch in
/// parallel against the same input.
///
/// Construct with [`RunnableParallel::new`] and add branches via
/// [`RunnableParallel::branch`]:
///
/// ```ignore
/// let parallel = RunnableParallel::new()
///     .branch("joke", joke_chain)
///     .branch("poem", poem_chain);
/// let outputs = parallel.invoke(topic, &ctx).await?;
/// // outputs: HashMap { "joke" => Message, "poem" => Message }
/// ```
pub struct RunnableParallel<I, O>
where
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    branches: Vec<Branch<I, O>>,
}

impl<I, O> RunnableParallel<I, O>
where
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    /// Empty parallel runner.
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    /// Append a named branch. Insertion order is preserved in the output
    /// `HashMap` only by virtue of `HashMap`'s eventual lookup; consumers
    /// that need ordered results should iterate by key list.
    #[must_use]
    pub fn branch<R>(mut self, name: impl Into<String>, runnable: R) -> Self
    where
        R: Runnable<I, O> + 'static,
    {
        self.branches.push((name.into(), Arc::new(runnable)));
        self
    }

    /// Number of branches registered.
    pub fn len(&self) -> usize {
        self.branches.len()
    }

    /// True when no branches are registered.
    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }
}

impl<I, O> Default for RunnableParallel<I, O>
where
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl<I, O> Runnable<I, HashMap<String, O>> for RunnableParallel<I, O>
where
    I: Clone + Send + Sync + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<HashMap<String, O>> {
        let futures = self.branches.iter().map(|(name, runnable)| {
            let input = input.clone();
            let runnable = Arc::clone(runnable);
            let name = name.clone();
            let ctx = ctx.clone();
            async move {
                let out = runnable.invoke(input, &ctx).await?;
                Ok::<_, entelix_core::Error>((name, out))
            }
        });
        let pairs = try_join_all(futures).await?;
        Ok(pairs.into_iter().collect())
    }
}
