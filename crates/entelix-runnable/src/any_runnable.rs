//! `AnyRunnable` — `serde_json::Value`-erased object-safe trait.
//!
//! F12 mitigation: typed `Runnable<I, O>` is great for compile-time
//! correctness, but builder code that picks a runnable at runtime cannot
//! name a uniform type. `AnyRunnable` collapses the I/O generics to
//! `serde_json::Value` so runtime dispatch is possible.
//!
//! Bridging a typed `Runnable<I, O>` to `AnyRunnable` is explicit — wrap it
//! with [`erase`]. The wrapper records the I/O types via `PhantomData` so
//! the impl bound stays well-formed.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::runnable::Runnable;

/// Object-safe variant of [`Runnable`] that operates on `serde_json::Value`
/// at the public interface. Useful for tool registries, dispatch tables,
/// and dynamic plug-in points.
#[async_trait]
pub trait AnyRunnable: Send + Sync + 'static {
    /// Human-readable identifier — usually defers to the wrapped
    /// `Runnable::name`.
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("any-runnable")
    }

    /// Call the inner runnable, transcoding through JSON.
    async fn invoke_any(
        &self,
        input: serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value>;
}

/// Convenience `Arc<dyn AnyRunnable>` alias for registries / dispatch tables.
pub type AnyRunnableHandle = Arc<dyn AnyRunnable>;

/// Wrap a typed [`Runnable<I, O>`] as an [`AnyRunnable`] by transcoding I/O
/// through `serde_json::Value`. Returns `Arc<dyn AnyRunnable>` so the result
/// drops directly into a registry.
///
/// Both `I` and `O` must round-trip through `serde_json::Value`
/// (`DeserializeOwned` + `Serialize`).
pub fn erase<R, I, O>(runnable: R) -> AnyRunnableHandle
where
    R: Runnable<I, O> + 'static,
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
{
    Arc::new(ErasedRunnable {
        inner: runnable,
        _phantom: PhantomData,
    })
}

/// Internal wrapper carrying the I/O markers needed for the `AnyRunnable`
/// impl. Constructed only via [`erase`].
struct ErasedRunnable<R, I, O> {
    inner: R,
    _phantom: PhantomData<fn(I) -> O>,
}

#[async_trait]
impl<R, I, O> AnyRunnable for ErasedRunnable<R, I, O>
where
    R: Runnable<I, O> + 'static,
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
{
    fn name(&self) -> Cow<'_, str> {
        Runnable::name(&self.inner)
    }

    async fn invoke_any(
        &self,
        input: serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value> {
        let typed_in: I = serde_json::from_value(input)?;
        let typed_out = self.inner.invoke(typed_in, ctx).await?;
        Ok(serde_json::to_value(typed_out)?)
    }
}
