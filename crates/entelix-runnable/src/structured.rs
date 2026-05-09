//! `StructuredOutputAdapter` — bridges
//! [`entelix_core::chat::ChatModel`]'s `complete_typed::<O>` into the
//! `Runnable<Vec<Message>, O>` composition contract so typed
//! structured outputs drop into `.pipe()` chains alongside
//! [`ToolToRunnableAdapter`](crate::ToolToRunnableAdapter) and the
//! built-in [`ChatModel`] `Runnable<Vec<Message>, Message>` impl.
//!
//! ```ignore
//! use entelix::prelude::*;
//! use entelix_runnable::ChatModelExt;
//!
//! let chain = prompt.pipe(model.with_structured_output::<Order>());
//! let order: Order = chain.invoke(vars, &ctx).await?;
//! ```
//!
//! The adapter is `Clone`-cheap (internal `Arc`) and forwards every
//! `invoke` through `ChatModel::complete_typed`, inheriting the
//! validation-retry budget configured on the underlying `ChatModel`
//! (CLAUDE.md invariant 20).

use std::marker::PhantomData;
use std::sync::Arc;

use entelix_core::chat::ChatModel;
use entelix_core::codecs::Codec;
use entelix_core::ir::Message;
use entelix_core::transports::Transport;
use entelix_core::{ExecutionContext, Result};

use crate::runnable::Runnable;

/// Adapts a `ChatModel<C, T>` to `Runnable<Vec<Message>, O>` by
/// routing every `invoke` through
/// [`ChatModel::complete_typed::<O>`](entelix_core::chat::ChatModel::complete_typed).
///
/// `O` is constrained to the same trait set as `complete_typed`:
/// `schemars::JsonSchema + DeserializeOwned + Send + 'static`. The
/// validation-retry budget configured on the underlying `ChatModel`
/// (`ChatModelConfig::validation_retries`) flows through unchanged
/// — schema-mismatch retries reflect the parser diagnostic to the
/// model and re-invoke (CLAUDE.md invariant 20).
///
/// Construct via [`Self::new`] (consumes the model), [`Self::from_arc`]
/// (shares an existing `Arc<ChatModel>`), or — most ergonomic — via
/// [`crate::ChatModelExt::with_structured_output`].
pub struct StructuredOutputAdapter<O, C: Codec, T: Transport> {
    inner: Arc<ChatModel<C, T>>,
    _phantom: PhantomData<fn() -> O>,
}

impl<O, C: Codec, T: Transport> StructuredOutputAdapter<O, C, T> {
    /// Wrap a concrete `ChatModel`.
    pub fn new(model: ChatModel<C, T>) -> Self {
        Self {
            inner: Arc::new(model),
            _phantom: PhantomData,
        }
    }

    /// Wrap an already-shared `Arc<ChatModel>` — avoids a second
    /// `Arc::new` for operators that hold the model behind an `Arc`
    /// already.
    pub const fn from_arc(model: Arc<ChatModel<C, T>>) -> Self {
        Self {
            inner: model,
            _phantom: PhantomData,
        }
    }

    /// Borrow the wrapped `ChatModel` — useful for inspecting config
    /// or threading the same model through multiple adapters.
    pub const fn inner(&self) -> &Arc<ChatModel<C, T>> {
        &self.inner
    }
}

impl<O, C: Codec, T: Transport> Clone for StructuredOutputAdapter<O, C, T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            _phantom: PhantomData,
        }
    }
}

impl<O, C: Codec, T: Transport> std::fmt::Debug for StructuredOutputAdapter<O, C, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuredOutputAdapter")
            .field("output", &std::any::type_name::<O>())
            .finish()
    }
}

#[async_trait::async_trait]
impl<O, C, T> Runnable<Vec<Message>, O> for StructuredOutputAdapter<O, C, T>
where
    O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
    C: Codec,
    T: Transport,
{
    async fn invoke(&self, input: Vec<Message>, ctx: &ExecutionContext) -> Result<O> {
        self.inner.complete_typed::<O>(input, ctx).await
    }
}

/// Extension methods on [`ChatModel`] that produce typed `Runnable`
/// adapters.
///
/// `model.with_structured_output::<Order>()` returns a
/// `Runnable<Vec<Message>, Order>` ready for `.pipe()` composition —
/// the [`with_structured_output`](Self::with_structured_output)
/// ergonomic, typed.
pub trait ChatModelExt<C: Codec, T: Transport>: Sized {
    /// Adapt this `ChatModel` into a
    /// `Runnable<Vec<Message>, O>` that routes every invocation
    /// through `complete_typed::<O>`. The original model is consumed;
    /// operators sharing the model behind an `Arc` reach for
    /// [`StructuredOutputAdapter::from_arc`] directly.
    fn with_structured_output<O>(self) -> StructuredOutputAdapter<O, C, T>
    where
        O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static;
}

impl<C, T> ChatModelExt<C, T> for ChatModel<C, T>
where
    C: Codec,
    T: Transport,
{
    fn with_structured_output<O>(self) -> StructuredOutputAdapter<O, C, T>
    where
        O: schemars::JsonSchema + serde::de::DeserializeOwned + Send + 'static,
    {
        StructuredOutputAdapter::new(self)
    }
}
