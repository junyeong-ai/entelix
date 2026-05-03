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

use crate::error::{Error, Result};
use crate::ir::{ContentPart, ModelResponse, ModelWarning, StopReason, Usage};
use crate::rate_limit::RateLimitSnapshot;

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
    },
    /// Append text to the in-progress text block. Consecutive `TextDelta`s
    /// fold into a single `ContentPart::Text` in the output.
    TextDelta {
        /// Text fragment to append.
        text: String,
    },
    /// Append text (or a signature) to the in-progress thinking
    /// block. Consecutive `ThinkingDelta`s fold into a single
    /// `ContentPart::Thinking` in the output. A delta carrying only
    /// a `signature` (empty `text`) attaches the redaction-resistant
    /// replay marker without growing the body.
    ThinkingDelta {
        /// Text fragment to append. Empty when the delta carries
        /// only a signature update.
        text: String,
        /// Vendor signature for redaction-resistant replay (Anthropic
        /// supplies via a discrete `signature_delta` event; other
        /// vendors leave `None`).
        signature: Option<String>,
    },
    /// Begin a new tool-use block. Closes any previously-open text block
    /// so the output preserves the model's intended ordering.
    ToolUseStart {
        /// Stable tool-use id.
        id: String,
        /// Tool name to call.
        name: String,
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
}

/// Per-thinking-block scratch space.
#[derive(Default)]
struct PendingThinking {
    text: String,
    signature: Option<String>,
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
    open_text: Option<String>,
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
            StreamDelta::Start { id, model } => {
                if !self.id.is_empty() || !self.model.is_empty() {
                    return Err(Error::invalid_request(
                        "StreamAggregator: duplicate Start delta",
                    ));
                }
                self.id = id;
                self.model = model;
            }
            StreamDelta::TextDelta { text } => {
                if self.pending_tool.is_some() {
                    return Err(Error::invalid_request(
                        "StreamAggregator: TextDelta during open tool_use block",
                    ));
                }
                self.flush_thinking();
                match &mut self.open_text {
                    Some(buf) => buf.push_str(&text),
                    None => self.open_text = Some(text),
                }
            }
            StreamDelta::ThinkingDelta { text, signature } => {
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
                if let Some(sig) = signature {
                    pending.signature = Some(sig);
                }
            }
            StreamDelta::ToolUseStart { id, name } => {
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
            StreamDelta::ToolUseStop => {
                let pending = self.pending_tool.take().ok_or_else(|| {
                    Error::invalid_request("StreamAggregator: ToolUseStop with no open tool block")
                })?;
                let input: serde_json::Value = if pending.input_buffer.is_empty() {
                    serde_json::json!({})
                } else {
                    // Surface the tool name + id and the buffered
                    // payload so operators can see which tool's
                    // arguments arrived malformed. The bare
                    // serde_json::Error message is opaque ("expected
                    // value at line 1 column 7"); without context,
                    // a multi-tool agent run leaves the operator
                    // hunting through logs.
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
                });
            }
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
        })
    }

    /// Close the open text buffer, if any, into a `ContentPart::Text`.
    fn flush_text(&mut self) {
        if let Some(text) = self.open_text.take()
            && !text.is_empty()
        {
            self.parts.push(ContentPart::Text {
                text,
                cache_control: None,
            });
        }
    }

    /// Close the open thinking buffer, if any, into a
    /// `ContentPart::Thinking`.
    fn flush_thinking(&mut self) {
        if let Some(pending) = self.open_thinking.take()
            && !(pending.text.is_empty() && pending.signature.is_none())
        {
            self.parts.push(ContentPart::Thinking {
                text: pending.text,
                signature: pending.signature,
                cache_control: None,
            });
        }
    }
}

/// Cap a malformed-JSON payload before it rides into an error
/// message. A streaming tool-use arguments buffer can be arbitrarily
/// large under provider misbehavior; including the full buffer
/// inflates structured logs and pollutes traces. 256 bytes is enough
/// for an operator to see the rough shape and cheap to keep.
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
