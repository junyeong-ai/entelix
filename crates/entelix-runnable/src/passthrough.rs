//! Identity `Runnable`. Useful as a fan-in default branch and in
//! `RunnableParallel` (later slice) to forward unchanged values.

use entelix_core::{ExecutionContext, Result};

use crate::runnable::Runnable;

/// `Runnable<T, T>` that returns its input unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunnablePassthrough;

#[async_trait::async_trait]
impl<T> Runnable<T, T> for RunnablePassthrough
where
    T: Send + 'static,
{
    async fn invoke(&self, input: T, _ctx: &ExecutionContext) -> Result<T> {
        Ok(input)
    }
}
