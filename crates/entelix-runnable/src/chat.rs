//! Closes the orphan-rule split for `ChatModel`: provides the `Runnable<
//! Vec<Message>, Message>` impl so chat models drop into `.pipe()`
//! chains and the streaming surface so `StreamMode::Messages` emits
//! real token-level deltas.
//!
//! `ChatModel` itself is defined in `entelix-core` (it depends only on
//! `Codec` + `Transport`). This module lives here because
//! `entelix-runnable` owns the `Runnable` trait — see ADR-0011.

use entelix_core::chat::ChatModel;
use entelix_core::codecs::Codec;
use entelix_core::ir::{Message, Role};
use entelix_core::stream::StreamAggregator;
use entelix_core::transports::Transport;
use entelix_core::{ExecutionContext, Result};
use futures::StreamExt;

use crate::runnable::Runnable;
use crate::stream::{BoxStream, DebugEvent, RunnableEvent, StreamChunk, StreamMode};

#[async_trait::async_trait]
impl<C, T> Runnable<Vec<Message>, Message> for ChatModel<C, T>
where
    C: Codec,
    T: Transport,
{
    async fn invoke(&self, input: Vec<Message>, ctx: &ExecutionContext) -> Result<Message> {
        self.complete(input, ctx).await
    }

    #[allow(tail_expr_drop_order, clippy::too_many_lines)]
    async fn stream(
        &self,
        input: Vec<Message>,
        mode: StreamMode,
        ctx: &ExecutionContext,
    ) -> Result<BoxStream<'_, Result<StreamChunk<Message>>>> {
        let codec_name = self.codec().name().to_owned();
        let delta_stream = self.stream_deltas(input, ctx).await?;
        Ok(Box::pin(async_stream::stream! {
            if matches!(mode, StreamMode::Events) {
                yield Ok(StreamChunk::Event(RunnableEvent::Started {
                    name: codec_name.clone(),
                }));
            }
            let mut delta_stream = delta_stream;
            let mut aggregator = StreamAggregator::new();
            while let Some(item) = delta_stream.next().await {
                match item {
                    Ok(delta) => {
                        if matches!(mode, StreamMode::Messages) {
                            yield Ok(StreamChunk::Message(delta.clone()));
                        }
                        if let Err(e) = aggregator.push(delta) {
                            if matches!(mode, StreamMode::Events) {
                                yield Ok(StreamChunk::Event(
                                    RunnableEvent::Finished {
                                        name: codec_name.clone(),
                                        ok: false,
                                    },
                                ));
                            }
                            yield Err(e);
                            return;
                        }
                    }
                    Err(e) => {
                        if matches!(mode, StreamMode::Events) {
                            yield Ok(StreamChunk::Event(RunnableEvent::Finished {
                                name: codec_name.clone(),
                                ok: false,
                            }));
                        }
                        yield Err(e);
                        return;
                    }
                }
            }
            let response = match aggregator.finalize() {
                Ok(r) => r,
                Err(e) => {
                    if matches!(mode, StreamMode::Events) {
                        yield Ok(StreamChunk::Event(RunnableEvent::Finished {
                            name: codec_name,
                            ok: false,
                        }));
                    }
                    yield Err(e);
                    return;
                }
            };
            let assistant = Message::new(Role::Assistant, response.content);
            match mode {
                StreamMode::Updates => {
                    yield Ok(StreamChunk::Update {
                        node: codec_name,
                        value: assistant,
                    });
                }
                StreamMode::Values | StreamMode::Messages => {
                    yield Ok(StreamChunk::Value(assistant));
                }
                StreamMode::Debug => {
                    yield Ok(StreamChunk::Debug(DebugEvent::NodeStart {
                        node: codec_name.clone(),
                        step: 1,
                    }));
                    yield Ok(StreamChunk::Value(assistant));
                    yield Ok(StreamChunk::Debug(DebugEvent::NodeEnd {
                        node: codec_name,
                        step: 1,
                    }));
                    yield Ok(StreamChunk::Debug(DebugEvent::Final));
                }
                StreamMode::Events => {
                    yield Ok(StreamChunk::Value(assistant));
                    yield Ok(StreamChunk::Event(RunnableEvent::Finished {
                        name: codec_name,
                        ok: true,
                    }));
                }
            }
        }))
    }
}
