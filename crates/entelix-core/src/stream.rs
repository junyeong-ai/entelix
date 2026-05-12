//! `StreamAggregator` — accumulates streaming model deltas into a coherent
//! [`ModelResponse`].
//!
//! Tool-call ordering is preserved: each `ToolUseStart` opens a fresh
//! tool block, subsequent `ToolUseInputDelta`s append into that block
//! until `ToolUseStop` closes it.
//!
//! ## Variant naming — semantic, not wire-aligned
//!
//! Variant names describe the *meaning* of each delta
//! (`TextDelta`, `ThinkingDelta`, `ToolUseStart`) rather than mirror
//! one vendor's SSE event names (`content_block_delta`,
//! `message_start`). Per invariant 5 the IR never returns
//! vendor-shaped JSON, and the same principle applies to the
//! streaming surface: codecs translate their wire events into
//! these variants so consumers writing against `StreamDelta` work
//! across Anthropic, `OpenAI` (Chat Completions and Responses),
//! Gemini, and Bedrock without renames. Renaming a variant to match
//! one provider's wire format would couple the public API to that
//! vendor's terminology and force a churn whenever they renumber an
//! event type.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use tokio::sync::oneshot;

use crate::codecs::BoxDeltaStream;
use crate::error::{Error, Result};
use crate::ir::{
    ContentPart, ModelResponse, ModelWarning, ProviderEchoSnapshot, StopReason, Usage,
};
use crate::rate_limit::RateLimitSnapshot;
use crate::service::ModelStream;

/// One chunk from a streaming model response.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamDelta {
    /// First message — vendor's response id and model identifier.
    Start {
        /// Vendor message id (echoed in the final `ModelResponse`).
        id: String,
        /// Resolved model identifier.
        model: String,
        /// Response-level vendor opaque round-trip tokens — OpenAI
        /// Responses `Response.id` (so the next request can chain via
        /// `previous_response_id` from `ModelRequest::continued_from`),
        /// or anything else the codec wants to carry at response root
        /// rather than on a single content part. The aggregator
        /// surfaces these on [`ModelResponse::provider_echoes`] at
        /// finalize time, mirroring the non-streaming decode path.
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },
    /// Append text to the in-progress text block. Consecutive `TextDelta`s
    /// fold into a single `ContentPart::Text` in the output.
    TextDelta {
        /// Text fragment to append.
        text: String,
        /// Vendor opaque round-trip tokens this fragment carries
        /// (Gemini 3.x attaches `thought_signature` to `text` parts on
        /// reasoning turns). The aggregator extends the open-text
        /// block's accumulated echoes — a single `ContentPart::Text`
        /// finalises with the union of every delta's echoes.
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },
    /// Append text (or vendor opaque tokens) to the in-progress
    /// thinking block. Consecutive `ThinkingDelta`s fold into a
    /// single `ContentPart::Thinking` in the output. A delta carrying
    /// only `provider_echoes` (empty `text`) attaches the round-trip
    /// marker without growing the body — Anthropic emits the
    /// signature on a discrete `signature_delta` SSE event with no
    /// associated text.
    ThinkingDelta {
        /// Text fragment to append. Empty when the delta carries
        /// only a `provider_echoes` update.
        text: String,
        /// Vendor opaque round-trip tokens (Anthropic `signature`,
        /// Gemini `thought_signature`, OpenAI Responses
        /// `encrypted_content`). Codecs pre-wrap the wire-shape blob
        /// into [`ProviderEchoSnapshot`] before yielding the delta;
        /// the aggregator stays codec-agnostic and just accumulates.
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },
    /// Begin a new tool-use block. Closes any open text block so the
    /// output preserves the model's intended ordering.
    ToolUseStart {
        /// Stable tool-use id.
        id: String,
        /// Tool name to call.
        name: String,
        /// Vendor opaque round-trip tokens attached to this tool call
        /// (Gemini 3.x `thought_signature` on `functionCall` parts —
        /// missing on the next turn yields HTTP 400 on the first
        /// `functionCall` of a step).
        provider_echoes: Vec<ProviderEchoSnapshot>,
    },
    /// Append partial JSON to the open tool-use block's input buffer.
    ToolUseInputDelta {
        /// Raw JSON fragment — the aggregator concatenates and parses
        /// once the block closes.
        partial_json: String,
    },
    /// Close the current tool-use block. Returns `Err` if the buffered
    /// JSON does not parse.
    ToolUseStop,
    /// Token usage update (last value wins).
    Usage(Usage),
    /// Provider rate-limit snapshot, typically emitted as the leading
    /// chunk by `ChatModel::stream_deltas` before the first content
    /// delta. Last value wins inside an aggregator.
    RateLimit(RateLimitSnapshot),
    /// Provider warning surfaced inline.
    Warning(ModelWarning),
    /// End of stream with stop reason.
    Stop {
        /// Reason the model halted.
        stop_reason: StopReason,
    },
}

/// Per-tool-block scratch space.
struct PendingTool {
    id: String,
    name: String,
    input_buffer: String,
    provider_echoes: Vec<ProviderEchoSnapshot>,
}

/// Per-thinking-block scratch space.
#[derive(Default)]
struct PendingThinking {
    text: String,
    provider_echoes: Vec<ProviderEchoSnapshot>,
}

/// Per-text-block scratch space.
#[derive(Default)]
struct PendingText {
    text: String,
    provider_echoes: Vec<ProviderEchoSnapshot>,
}

/// Accumulator that turns a sequence of `StreamDelta`s into a
/// `ModelResponse`.
///
/// Typical usage:
/// ```ignore
/// let mut agg = StreamAggregator::new();
/// while let Some(delta) = stream.next().await {
///     agg.push(delta?)?;
/// }
/// let response = agg.finalize()?;
/// ```
#[derive(Default)]
pub struct StreamAggregator {
    id: String,
    model: String,
    parts: Vec<ContentPart>,
    /// Buffer for the currently-open text block. `None` when the next
    /// `TextDelta` should start a fresh block.
    open_text: Option<PendingText>,
    /// Buffer for the currently-open thinking block. `None` when the
    /// next `ThinkingDelta` should start a fresh block. The text and
    /// tool buffers are mutually exclusive with the thinking buffer:
    /// any non-thinking delta closes an open thinking block first
    /// (intra-turn order is preserved).
    open_thinking: Option<PendingThinking>,
    pending_tool: Option<PendingTool>,
    usage: Option<Usage>,
    rate_limit: Option<RateLimitSnapshot>,
    stop_reason: Option<StopReason>,
    warnings: Vec<ModelWarning>,
    /// Response-level vendor opaque round-trip tokens captured from
    /// the streaming `Start` delta. Surfaced on
    /// [`ModelResponse::provider_echoes`] at finalize so streaming
    /// and non-streaming decode produce equivalent IR.
    response_echoes: Vec<ProviderEchoSnapshot>,
}

impl StreamAggregator {
    /// Empty aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one delta. Returns `Err` on protocol violations
    /// (`ToolUseInputDelta` outside a tool block, malformed JSON in
    /// `ToolUseStop`, double `ToolUseStart`).
    pub fn push(&mut self, delta: StreamDelta) -> Result<()> {
        match delta {
            StreamDelta::Start {
                id,
                model,
                provider_echoes,
            } => {
                if !self.id.is_empty() || !self.model.is_empty() {
                    return Err(Error::invalid_request(
                        "StreamAggregator: duplicate Start delta",
                    ));
                }
                self.id = id;
                self.model = model;
                self.response_echoes.extend(provider_echoes);
            }
            StreamDelta::TextDelta {
                text,
                provider_echoes,
            } => {
                if self.pending_tool.is_some() {
                    return Err(Error::invalid_request(
                        "StreamAggregator: TextDelta during open tool_use block",
                    ));
                }
                self.flush_thinking();
                let pending = self.open_text.get_or_insert_with(PendingText::default);
                pending.text.push_str(&text);
                pending.provider_echoes.extend(provider_echoes);
            }
            StreamDelta::ThinkingDelta {
                text,
                provider_echoes,
            } => {
                if self.pending_tool.is_some() {
                    return Err(Error::invalid_request(
                        "StreamAggregator: ThinkingDelta during open tool_use block",
                    ));
                }
                self.flush_text();
                let pending = self
                    .open_thinking
                    .get_or_insert_with(PendingThinking::default);
                pending.text.push_str(&text);
                pending.provider_echoes.extend(provider_echoes);
            }
            StreamDelta::ToolUseStart {
                id,
                name,
                provider_echoes,
            } => {
                if self.pending_tool.is_some() {
                    return Err(Error::invalid_request(
                        "StreamAggregator: ToolUseStart while another tool block is open",
                    ));
                }
                self.flush_text();
                self.flush_thinking();
                self.pending_tool = Some(PendingTool {
                    id,
                    name,
                    input_buffer: String::new(),
                    provider_echoes,
                });
            }
            StreamDelta::ToolUseInputDelta { partial_json } => {
                let pending = self.pending_tool.as_mut().ok_or_else(|| {
                    Error::invalid_request(
                        "StreamAggregator: ToolUseInputDelta with no open tool block",
                    )
                })?;
                pending.input_buffer.push_str(&partial_json);
            }
            StreamDelta::ToolUseStop => self.close_tool_block()?,
            StreamDelta::Usage(u) => self.usage = Some(u),
            StreamDelta::RateLimit(r) => self.rate_limit = Some(r),
            StreamDelta::Warning(w) => self.warnings.push(w),
            StreamDelta::Stop { stop_reason } => {
                // Refuse to overwrite a stop reason. Some providers
                // misbehave and ship a follow-up Stop delta after a
                // valid terminal one (rare, but real); silently
                // accepting the second value would change the
                // observed termination cause from `EndTurn` to
                // `MaxTokens` — a meaningful semantic flip that
                // operators would never see. Fail closed instead.
                if self.stop_reason.is_some() {
                    return Err(Error::invalid_request(
                        "StreamAggregator: duplicate Stop delta — terminal state already set",
                    ));
                }
                self.stop_reason = Some(stop_reason);
            }
        }
        Ok(())
    }

    /// Convenience: returns true after a `Stop` delta has been pushed.
    pub const fn is_finished(&self) -> bool {
        self.stop_reason.is_some()
    }

    /// Drain into a final `ModelResponse`. Returns `Err` if a tool block
    /// was left open or no `Stop` delta was seen.
    pub fn finalize(mut self) -> Result<ModelResponse> {
        if self.pending_tool.is_some() {
            return Err(Error::invalid_request(
                "StreamAggregator: stream ended with an open tool block",
            ));
        }
        let stop_reason = self.stop_reason.take().ok_or_else(|| {
            Error::invalid_request("StreamAggregator: stream ended without Stop delta")
        })?;
        self.flush_text();
        self.flush_thinking();
        // A streaming response that closes without ever emitting a
        // `Usage` delta silently zeros out the cost meter — every
        // downstream `gen_ai.usage.cost` becomes a phantom $0
        // charge. Surface a `LossyEncode` warning so operators see
        // the miss in observability instead of debugging a
        // suspiciously-cheap month at billing time.
        if self.usage.is_none() {
            self.warnings.push(crate::ir::ModelWarning::LossyEncode {
                field: "usage".to_owned(),
                detail: "streaming response closed without Usage delta — cost will be zero"
                    .to_owned(),
            });
        }
        Ok(ModelResponse {
            id: self.id,
            model: self.model,
            stop_reason,
            content: self.parts,
            usage: self.usage.unwrap_or_default(),
            rate_limit: self.rate_limit,
            warnings: self.warnings,
            provider_echoes: self.response_echoes,
        })
    }

    /// Close an open `tool_use` block — parses the buffered JSON
    /// arguments and pushes the finalised `ContentPart::ToolUse` (with
    /// any accumulated `provider_echoes`) onto `parts`. Returns
    /// `Err(Error::invalid_request)` if there is no open tool block or
    /// the buffered arguments fail to parse.
    fn close_tool_block(&mut self) -> Result<()> {
        let pending = self.pending_tool.take().ok_or_else(|| {
            Error::invalid_request("StreamAggregator: ToolUseStop with no open tool block")
        })?;
        let input: serde_json::Value = if pending.input_buffer.is_empty() {
            serde_json::json!({})
        } else {
            // Surface the tool name + id and the buffered payload so
            // operators can see which tool's arguments arrived
            // malformed. The bare serde_json::Error message is opaque
            // ("expected value at line 1 column 7"); without context,
            // a multi-tool agent run leaves the operator hunting
            // through logs.
            serde_json::from_str(&pending.input_buffer).map_err(|e| {
                Error::invalid_request(format!(
                    "StreamAggregator: ToolUse '{}' (id={}) arguments are not valid JSON: \
                     {e}; buffered={:?}",
                    pending.name,
                    pending.id,
                    truncate_for_diagnostic(&pending.input_buffer),
                ))
            })?
        };
        self.parts.push(ContentPart::ToolUse {
            id: pending.id,
            name: pending.name,
            input,
            provider_echoes: pending.provider_echoes,
        });
        Ok(())
    }

    /// Close the open text buffer, if any, into a `ContentPart::Text`.
    fn flush_text(&mut self) {
        if let Some(pending) = self.open_text.take()
            && !(pending.text.is_empty() && pending.provider_echoes.is_empty())
        {
            self.parts.push(ContentPart::Text {
                text: pending.text,
                cache_control: None,
                provider_echoes: pending.provider_echoes,
            });
        }
    }

    /// Close the open thinking buffer, if any, into a
    /// `ContentPart::Thinking`.
    fn flush_thinking(&mut self) {
        if let Some(pending) = self.open_thinking.take()
            && !(pending.text.is_empty() && pending.provider_echoes.is_empty())
        {
            self.parts.push(ContentPart::Thinking {
                text: pending.text,
                cache_control: None,
                provider_echoes: pending.provider_echoes,
            });
        }
    }
}

/// Cap a malformed-JSON payload before it rides into an error
/// message. A streaming tool-use arguments buffer can be arbitrarily
/// large under provider misbehavior; including the full buffer
/// inflates structured logs and pollutes traces. 256 bytes is enough
/// for an operator to see the rough shape and cheap to keep.
/// Wrap a raw `BoxDeltaStream` in a [`ModelStream`] whose
/// [`ModelStream::completion`] future resolves to the aggregated
/// [`ModelResponse`] after the consumer drains the stream.
///
/// The aggregator runs as a stateful side-effect inside the
/// returned stream — each delta the consumer reads is also pushed
/// into a local `StreamAggregator`. When the consumer reads the
/// terminal `Stop` (or the inner stream ends without one), the
/// aggregator finalises and the `completion` future resolves. If
/// the consumer drops the stream early, the aggregator is dropped
/// without finalising and `completion` resolves to
/// `Err(Error::Cancelled)` so observability layers gating on
/// `completion.await.is_ok()` (cost emission) do not fire on
/// abandoned streams (invariant 12).
///
/// Mid-stream `Err` propagates twofold: the consumer sees the
/// `Err` on the next `next().await`, and `completion` resolves to
/// the same error so wrapping layers see the failure path on the
/// post-stream branch.
pub fn tap_aggregator(inner: BoxDeltaStream<'static>) -> ModelStream {
    let (tx, rx) = oneshot::channel::<Result<ModelResponse>>();
    let tap = AggregatorTap {
        inner,
        agg: StreamAggregator::new(),
        completion: Some(tx),
        terminated: false,
    };
    ModelStream {
        stream: Box::pin(tap),
        completion: Box::pin(async move {
            match rx.await {
                Ok(result) => result,
                // Sender dropped before sending — the wrapping
                // stream was abandoned without reaching terminal
                // Stop. Surface as Cancelled so layers gate on Ok.
                Err(_) => Err(Error::Cancelled),
            }
        }) as BoxFuture<'static, Result<ModelResponse>>,
    }
}

/// `Stream<Item = Result<StreamDelta>>` wrapper that taps each
/// delta into a `StreamAggregator`. On terminal `Stop` (or stream
/// EOF, or mid-stream `Err`), it sends the aggregator's final
/// state through a `oneshot::Sender` so the paired
/// [`ModelStream::completion`] future resolves with the
/// aggregated response or the propagated error.
struct AggregatorTap {
    inner: BoxDeltaStream<'static>,
    agg: StreamAggregator,
    completion: Option<oneshot::Sender<Result<ModelResponse>>>,
    terminated: bool,
}

impl AggregatorTap {
    /// Send the aggregator's terminal state through the completion
    /// channel. Idempotent — subsequent calls are no-ops, so a
    /// stream that finalises on `Stop` and is then dropped does
    /// not double-send.
    fn finalize(&mut self, outcome: Result<ModelResponse>) {
        if let Some(tx) = self.completion.take() {
            // Receiver may have been dropped (operator abandoned
            // `completion` future before consuming the stream); the
            // send error is not actionable on this side.
            let _ = tx.send(outcome);
        }
    }
}

impl Stream for AggregatorTap {
    type Item = Result<StreamDelta>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.terminated {
            return Poll::Ready(None);
        }
        match self.inner.poll_next_unpin(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // Inner stream ended without terminal Stop —
                // finalise produces `Err` (the aggregator's own
                // protocol-violation message). `completion`
                // resolves Err so wrapping layers see failure.
                let agg = std::mem::take(&mut self.agg);
                let outcome = agg.finalize();
                self.finalize(outcome);
                self.terminated = true;
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                // Mid-stream error — clone the error for the
                // completion channel (consumer sees the original
                // Err on this branch).
                let cloned = clone_error(&e);
                self.finalize(Err(cloned));
                self.terminated = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(Some(Ok(delta))) => {
                let is_stop = matches!(delta, StreamDelta::Stop { .. });
                if let Err(e) = self.agg.push(delta.clone()) {
                    // Aggregator rejected a protocol violation —
                    // surface to the consumer (so they see why)
                    // *and* through completion (so layers see
                    // the failure path).
                    let cloned = clone_error(&e);
                    self.finalize(Err(cloned));
                    self.terminated = true;
                    return Poll::Ready(Some(Err(e)));
                }
                if is_stop {
                    // Terminal Stop — finalise immediately so the
                    // completion future resolves before the
                    // consumer's next `.next()` call. Any further
                    // poll returns `None`.
                    let agg = std::mem::take(&mut self.agg);
                    let outcome = agg.finalize();
                    self.finalize(outcome);
                    self.terminated = true;
                }
                Poll::Ready(Some(Ok(delta)))
            }
        }
    }
}

impl Drop for AggregatorTap {
    fn drop(&mut self) {
        // Stream dropped without terminal Stop — completion
        // resolves Err(Cancelled) so cost-emit layers gating on
        // Ok branch do not fire on abandoned streams.
        if self.completion.is_some() {
            self.finalize(Err(Error::Cancelled));
        }
    }
}

/// Best-effort clone of an `Error` for the `completion` channel.
/// `Error` is not `Clone` because `serde_json::Error` and the
/// `Auth` variant carry non-Clone payloads, but the streaming-tap
/// path needs to forward both the consumer-side `Err` and the
/// completion-future `Err`. The reconstruction preserves the
/// variant + message for observability purposes; the source
/// chain on the consumer side stays intact (the original `Err` is
/// what the consumer receives).
fn clone_error(e: &Error) -> Error {
    use crate::error::ProviderErrorKind;
    match e {
        Error::InvalidRequest(msg) => Error::invalid_request(msg.clone()),
        Error::Config(msg) => Error::config(msg.clone()),
        Error::Provider {
            kind,
            message,
            retry_after,
            ..
        } => {
            let cloned = match kind {
                ProviderErrorKind::Network => Error::provider_network(message.clone()),
                ProviderErrorKind::Tls => Error::provider_tls(message.clone()),
                ProviderErrorKind::Dns => Error::provider_dns(message.clone()),
                ProviderErrorKind::Http(status) => Error::provider_http(*status, message.clone()),
            };
            match retry_after {
                Some(after) => cloned.with_retry_after(*after),
                None => cloned,
            }
        }
        Error::Auth(_) => Error::config("authentication failed (cloned for stream completion)"),
        Error::Cancelled => Error::Cancelled,
        Error::DeadlineExceeded => Error::DeadlineExceeded,
        Error::Interrupted { kind, payload } => Error::Interrupted {
            kind: kind.clone(),
            payload: payload.clone(),
        },
        Error::Serde(_) => {
            Error::invalid_request("output serialisation failed (cloned for stream completion)")
        }
        Error::UsageLimitExceeded(breach) => Error::UsageLimitExceeded(breach.clone()),
        Error::ModelRetry { hint, attempt } => Error::ModelRetry {
            hint: hint.clone(),
            attempt: *attempt,
        },
    }
}

fn truncate_for_diagnostic(s: &str) -> String {
    const BUDGET: usize = 256;
    if s.len() <= BUDGET {
        return s.to_owned();
    }
    let mut cut = BUDGET;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}
