//! Streaming surface for `Runnable`.
//!
//! Adds a 5-mode streaming protocol on top of the composition contract,
//! mirroring `LangGraph`'s `stream_mode` semantics. Every `Runnable` exposes
//! a `stream(input, mode, ctx)` method via the trait's default
//! implementation; specialized impls (`CompiledGraph<S>`,
//! `ChatModel<C, T>`) override to emit per-step chunks.
//!
//! - [`StreamMode::Values`]   — full output snapshot after each step.
//! - [`StreamMode::Updates`]  — `(node_name, output)` per step.
//! - [`StreamMode::Messages`] — provider-level token deltas
//!   (re-uses `entelix_core::stream::StreamDelta`).
//! - [`StreamMode::Debug`]    — node lifecycle markers (start/end).
//! - [`StreamMode::Events`]   — runtime events (started/finished).
//!
//! The default trait method materializes a single-shot stream by calling
//! `invoke` and emitting one chunk shaped per the requested mode. Graph
//! and model implementors emit multiple chunks as work progresses.

use std::pin::Pin;

use entelix_core::stream::StreamDelta;
use futures::Stream;

/// Boxed `Stream` alias used by every `stream()` return type.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

/// Which stream shape the caller wants.
///
/// Matches `LangGraph`'s `stream_mode`. `Values` and `Updates` are the most
/// common; `Messages` is for token-level UX; `Debug` and `Events` are for
/// observability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StreamMode {
    /// Emit the full output state after each step.
    Values,
    /// Emit only what changed at each step, tagged with the node name.
    Updates,
    /// Emit token-level deltas from any underlying chat model.
    Messages,
    /// Emit lifecycle markers (node start / node end / final).
    Debug,
    /// Emit runtime events (runnable started / runnable finished).
    Events,
}

/// One chunk of a streaming `Runnable` invocation.
///
/// Generic over the runnable's output type `O`. Chunks not relevant to a
/// chosen mode are simply not emitted; the variant the caller observes is
/// determined by the requested [`StreamMode`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum StreamChunk<O> {
    /// Full snapshot — emitted in [`StreamMode::Values`].
    Value(O),
    /// Per-node update — emitted in [`StreamMode::Updates`].
    Update {
        /// Identifier of the node (or runnable) that produced `value`.
        node: String,
        /// State (or output) the node produced this step.
        value: O,
    },
    /// Token-level delta — emitted in [`StreamMode::Messages`].
    Message(StreamDelta),
    /// Lifecycle marker — emitted in [`StreamMode::Debug`].
    Debug(DebugEvent),
    /// Runtime event — emitted in [`StreamMode::Events`].
    Event(RunnableEvent),
}

/// Lifecycle marker for [`StreamMode::Debug`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DebugEvent {
    /// Step `step` began executing node `node`.
    NodeStart {
        /// Name of the node that is about to run.
        node: String,
        /// 1-based step counter within this invocation.
        step: usize,
    },
    /// Step `step` finished executing node `node`.
    NodeEnd {
        /// Name of the node that just finished.
        node: String,
        /// 1-based step counter within this invocation.
        step: usize,
    },
    /// Graph reached a terminal state.
    Final,
}

/// Runtime event for [`StreamMode::Events`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunnableEvent {
    /// Runnable named `name` started executing.
    Started {
        /// Identifier of the runnable.
        name: String,
    },
    /// Runnable named `name` finished. `ok` is true when invocation
    /// returned successfully.
    Finished {
        /// Identifier of the runnable.
        name: String,
        /// Whether the invocation succeeded.
        ok: bool,
    },
}

impl<O> StreamChunk<O> {
    /// Borrow the inner `O` if this chunk carries one (`Value` or
    /// `Update`); otherwise `None`.
    pub const fn output(&self) -> Option<&O> {
        match self {
            Self::Value(v) | Self::Update { value: v, .. } => Some(v),
            _ => None,
        }
    }

    /// Consume the chunk, returning the inner `O` for the carrier
    /// variants.
    pub fn into_output(self) -> Option<O> {
        match self {
            Self::Value(v) | Self::Update { value: v, .. } => Some(v),
            _ => None,
        }
    }
}
