//! The `Runnable<I, O>` trait — entelix's composition contract (invariant 7).
//!
//! Anything composable in the SDK implements this trait: codecs, prompts,
//! parsers, tools (via [`crate::ToolToRunnableAdapter`]), agents, compiled
//! state graphs. `.pipe()` (see [`crate::RunnableExt`]) is the universal
//! connector.

use std::borrow::Cow;

use entelix_core::{Error, ExecutionContext, Result};
use futures::stream;

use crate::stream::{BoxStream, DebugEvent, RunnableEvent, StreamChunk, StreamMode};

/// One end of a composable computation.
///
/// Implementors describe what they consume (`I`) and produce (`O`);
/// orchestration code combines them via [`crate::RunnableExt::pipe`] without
/// caring about the concrete types in between. The trait surfaces three
/// execution shapes: `invoke` (single-shot), `batch` (sequential or
/// parallel sequence), and `stream` (5-mode incremental output, see
/// [`StreamMode`]).
#[async_trait::async_trait]
pub trait Runnable<I, O>: Send + Sync
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Single-shot invocation.
    async fn invoke(&self, input: I, ctx: &ExecutionContext) -> Result<O>;

    /// Batch invocation. The default runs `invoke` sequentially over the input
    /// vector. Implementors that can parallelize (e.g. independent provider
    /// calls) override this.
    async fn batch(&self, inputs: Vec<I>, ctx: &ExecutionContext) -> Result<Vec<O>> {
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            if ctx.is_cancelled() {
                return Err(Error::Cancelled);
            }
            out.push(self.invoke(input, ctx).await?);
        }
        Ok(out)
    }

    /// Streaming invocation, shaped by `mode` (see [`StreamMode`]).
    ///
    /// The default implementation falls back to `invoke` and yields one
    /// chunk shaped per mode. Implementors that have intermediate work
    /// to expose (compiled graphs over multiple node steps, chat models
    /// receiving SSE deltas) override this method to emit multiple
    /// chunks. Cancellation is handled per chunk via `ctx`.
    async fn stream(
        &self,
        input: I,
        mode: StreamMode,
        ctx: &ExecutionContext,
    ) -> Result<BoxStream<'_, Result<StreamChunk<O>>>> {
        let name = self.name().into_owned();
        let result = self.invoke(input, ctx).await?;
        let chunks: Vec<Result<StreamChunk<O>>> = match mode {
            StreamMode::Values | StreamMode::Messages => {
                vec![Ok(StreamChunk::Value(result))]
            }
            StreamMode::Updates => vec![Ok(StreamChunk::Update {
                node: name,
                value: result,
            })],
            StreamMode::Debug => vec![
                Ok(StreamChunk::Debug(DebugEvent::NodeStart {
                    node: name.clone(),
                    step: 1,
                })),
                Ok(StreamChunk::Value(result)),
                Ok(StreamChunk::Debug(DebugEvent::NodeEnd {
                    node: name,
                    step: 1,
                })),
                Ok(StreamChunk::Debug(DebugEvent::Final)),
            ],
            StreamMode::Events => vec![
                Ok(StreamChunk::Event(RunnableEvent::Started {
                    name: name.clone(),
                })),
                Ok(StreamChunk::Value(result)),
                Ok(StreamChunk::Event(RunnableEvent::Finished {
                    name,
                    ok: true,
                })),
            ],
        };
        Ok(Box::pin(stream::iter(chunks)))
    }

    /// Human-readable identifier used by tracing and debug output. Default is
    /// the Rust type name — implementors may override with a domain label
    /// (e.g. `"anthropic-messages"`, `"json-parser"`).
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(core::any::type_name::<Self>())
    }
}
