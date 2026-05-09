//! `Codec` trait and `EncodedRequest` — the IR ⇄ wire boundary.
//!
//! `Codec` is intentionally narrow: it converts `ModelRequest`s into
//! provider-shaped HTTP payloads and decodes the responses back into IR.
//! Streaming is a first-class concern through `decode_stream` —
//! incremental byte chunks become `StreamDelta` events, with the codec
//! owning its own parser state machine (SSE event parser, NDJSON line
//! splitter, AWS event-stream binary frame, etc.).

use std::pin::Pin;

use bytes::Bytes;
use futures::Stream;

use crate::error::Result;
use crate::ir::{Capabilities, ModelRequest, ModelResponse, ModelWarning, OutputStrategy};
use crate::rate_limit::RateLimitSnapshot;
use crate::stream::StreamDelta;

/// Boxed byte-chunk stream produced by a `Transport` and consumed by a
/// `Codec` during streaming.
pub type BoxByteStream<'a> = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send + 'a>>;

/// Boxed `StreamDelta` stream produced by `Codec::decode_stream`.
pub type BoxDeltaStream<'a> = Pin<Box<dyn Stream<Item = Result<StreamDelta>> + Send + 'a>>;

/// Bytes the transport will send to the provider plus the metadata the codec
/// learned during encoding.
///
/// `path` is the URL path the transport appends to its base URL (e.g.
/// `/v1/messages` for Anthropic). `headers` carries codec-required fields
/// like `content-type` and `anthropic-version`; transports add credentials
/// at send time.
#[derive(Clone, Debug)]
pub struct EncodedRequest {
    /// HTTP method. POST for every Phase-1 codec; pre-set as a forward hint.
    pub method: http::Method,
    /// URL path appended to the transport's base URL.
    pub path: String,
    /// Vendor-required HTTP headers (NOT credentials — transport adds those).
    pub headers: http::HeaderMap,
    /// JSON body bytes.
    pub body: Bytes,
    /// `true` if this request was produced by `encode_streaming` and the
    /// transport should stream the response body (e.g. open an SSE
    /// connection). Transports treat unset / `false` as a regular
    /// request/response.
    pub streaming: bool,
    /// Non-fatal warnings the codec produced during encoding. Carried into
    /// `ModelResponse::warnings` after the call returns.
    pub warnings: Vec<ModelWarning>,
}

impl EncodedRequest {
    /// Build a POST request with the given path and JSON body.
    pub fn post_json(path: impl Into<String>, body: Bytes) -> Self {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        Self {
            method: http::Method::POST,
            path: path.into(),
            headers,
            body,
            streaming: false,
            warnings: Vec::new(),
        }
    }

    /// Mark this request as streaming-shaped. Codecs call this from
    /// `encode_streaming` after appending any vendor-specific headers
    /// (e.g. `Accept: text/event-stream`).
    #[must_use]
    pub const fn into_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }
}

/// Stateless encoder/decoder for ONE provider wire format.
///
/// A `Codec` knows nothing about HTTP, auth, or retries. It turns IR into
/// bytes and bytes into IR. Streaming uses the same trait — `decode_stream`
/// owns the codec's parser state machine.
pub trait Codec: Send + Sync + 'static {
    /// Stable codec identifier — `"anthropic-messages"`,
    /// `"openai-chat"`, etc. Used in logs and metrics tags.
    fn name(&self) -> &'static str;

    /// Capability surface the codec advertises for the given model. Codecs
    /// vary by model (small models lacking vision, etc.).
    fn capabilities(&self, model: &str) -> Capabilities;

    /// Resolve [`OutputStrategy::Auto`] to the codec's preferred
    /// dispatch shape for `model`. Called once at codec-construction
    /// time per request — never per-delta or per-retry, so the
    /// resolved strategy is part of the SessionGraph event log's
    /// deterministic-replay surface.
    ///
    /// Default returns [`OutputStrategy::Native`] — most codecs ship
    /// vendor-native structured output (OpenAI Responses
    /// `text.format=json_schema`, Gemini `responseJsonSchema`,
    /// Bedrock Anthropic-passthrough). Codecs whose native channel
    /// is newer / less mature than the tool-call surface
    /// (Anthropic Messages today — `output_config` ships without
    /// a strict toggle) override to [`OutputStrategy::Tool`].
    fn auto_output_strategy(&self, _model: &str) -> OutputStrategy {
        OutputStrategy::Native
    }

    /// Encode IR → wire body for a one-shot (non-streaming) call.
    /// Implementors push warnings onto the returned
    /// `EncodedRequest::warnings` for any IR field they had to drop or
    /// coerce.
    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest>;

    /// Decode wire body → IR. `warnings_in` are the encode-time warnings
    /// that should be carried forward into `ModelResponse::warnings` so the
    /// caller sees the full advisory list in one place.
    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse>;

    /// Extract a [`RateLimitSnapshot`] from response headers, if the
    /// vendor exposes rate-limit state in headers. Default returns
    /// `None` — codecs whose providers publish rate-limit headers
    /// override this and parse them.
    fn extract_rate_limit(&self, _headers: &http::HeaderMap) -> Option<RateLimitSnapshot> {
        None
    }

    /// Encode IR → wire body for a streaming call. Default impl delegates
    /// to `encode` and marks the request as streaming; codecs that need a
    /// different body shape (e.g. `stream: true` field) or extra headers
    /// (e.g. `Accept: text/event-stream`) override.
    fn encode_streaming(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        Ok(self.encode(request)?.into_streaming())
    }

    /// Decode an incremental byte stream → IR `StreamDelta` stream.
    ///
    /// Implementors own their parser state machine — Anthropic walks SSE
    /// events, `OpenAI` splits `data:` lines, Gemini reads NDJSON, Bedrock
    /// parses AWS event-stream frames. Default impl is a graceful
    /// fallback: collects every chunk, runs `decode` once at the end, and
    /// emits the resulting `ModelResponse` as a single
    /// `StreamDelta::Stop`. Concrete codecs replace it as soon as they
    /// support real token-level streaming.
    #[allow(tail_expr_drop_order)]
    fn decode_stream<'a>(
        &'a self,
        bytes: BoxByteStream<'a>,
        warnings_in: Vec<ModelWarning>,
    ) -> BoxDeltaStream<'a> {
        Box::pin(async_stream::stream! {
            let mut buf: Vec<u8> = Vec::new();
            let mut bytes = bytes;
            while let Some(chunk) = futures::StreamExt::next(&mut bytes).await {
                let chunk = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                buf.extend_from_slice(&chunk);
            }
            let response = match self.decode(&buf, warnings_in) {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            for delta in deltas_from_response(&response) {
                yield Ok(delta);
            }
        })
    }
}

/// Wire literal for the OpenAI `service_tier` request field. Shared
/// between `OpenAiChatCodec` and `OpenAiResponsesCodec` so the
/// rendered string matches the documented enum exactly across both
/// endpoints.
pub fn service_tier_str(tier: crate::ir::ServiceTier) -> &'static str {
    match tier {
        crate::ir::ServiceTier::Auto => "auto",
        crate::ir::ServiceTier::Default => "default",
        crate::ir::ServiceTier::Flex => "flex",
        crate::ir::ServiceTier::Priority => "priority",
        crate::ir::ServiceTier::Scale => "scale",
    }
}

/// Shared OpenAI-style rate-limit header parser. Used by both
/// `OpenAiChatCodec` and `OpenAiResponsesCodec` because the
/// `x-ratelimit-*` family is identical across the two endpoints.
pub fn extract_openai_rate_limit(headers: &http::HeaderMap) -> Option<RateLimitSnapshot> {
    let mut snapshot = RateLimitSnapshot::default();
    let mut populated = false;
    let pairs: [(&str, &mut Option<u64>); 2] = [
        (
            "x-ratelimit-remaining-requests",
            &mut snapshot.requests_remaining,
        ),
        (
            "x-ratelimit-remaining-tokens",
            &mut snapshot.tokens_remaining,
        ),
    ];
    for (header_name, target) in pairs {
        if let Some(v) = headers.get(header_name).and_then(|h| h.to_str().ok())
            && let Ok(parsed) = v.parse::<u64>()
        {
            *target = Some(parsed);
            snapshot.raw.insert(header_name.to_owned(), v.to_owned());
            populated = true;
        }
    }
    // OpenAI emits reset deltas as durations (e.g. `6m0s`); preserve the
    // raw value for diagnostics rather than parsing into DateTime — vendor
    // doesn't surface absolute reset timestamps.
    for header_name in ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"] {
        if let Some(v) = headers.get(header_name).and_then(|h| h.to_str().ok()) {
            snapshot.raw.insert(header_name.to_owned(), v.to_owned());
            populated = true;
        }
    }
    populated.then_some(snapshot)
}

fn deltas_from_response(response: &ModelResponse) -> Vec<StreamDelta> {
    use crate::ir::ContentPart;

    let mut deltas = Vec::new();
    deltas.push(StreamDelta::Start {
        id: response.id.clone(),
        model: response.model.clone(),
    });
    for part in &response.content {
        match part {
            ContentPart::Text { text, .. } => {
                deltas.push(StreamDelta::TextDelta { text: text.clone() });
            }
            ContentPart::ToolUse { id, name, input } => {
                deltas.push(StreamDelta::ToolUseStart {
                    id: id.clone(),
                    name: name.clone(),
                });
                deltas.push(StreamDelta::ToolUseInputDelta {
                    partial_json: input.to_string(),
                });
                deltas.push(StreamDelta::ToolUseStop);
            }
            ContentPart::Thinking {
                text, signature, ..
            } => {
                deltas.push(StreamDelta::ThinkingDelta {
                    text: text.clone(),
                    signature: signature.clone(),
                });
            }
            // Multimodal inputs, citations, tool results never originate
            // from the model on the assistant streaming path — text /
            // thinking / tool_use / image_output / audio_output are the
            // assistant-emitted shapes. Output media on the synthetic
            // streaming fallback rides through `StreamDelta::Warning`
            // (codecs that natively support multimodal output emit
            // their own delta in the per-codec stream parser).
            ContentPart::Image { .. }
            | ContentPart::Audio { .. }
            | ContentPart::Video { .. }
            | ContentPart::Document { .. }
            | ContentPart::Citation { .. }
            | ContentPart::ToolResult { .. }
            | ContentPart::ImageOutput { .. }
            | ContentPart::AudioOutput { .. } => {}
        }
    }
    deltas.push(StreamDelta::Usage(response.usage.clone()));
    for w in &response.warnings {
        deltas.push(StreamDelta::Warning(w.clone()));
    }
    deltas.push(StreamDelta::Stop {
        stop_reason: response.stop_reason.clone(),
    });
    deltas
}

/// Parse a wire-format response body into a `serde_json::Value`,
/// wrapping the underlying serde error with operator-actionable
/// context: the codec's name, the response body size, and a
/// truncated peek at the first ~200 bytes. The bare
/// `serde_json::Error` ("expected value at line 4 column 12") is
/// useless for triage — operators need to know which provider
/// returned the body and roughly what shape it had.
///
/// Used by every codec's `decode` entry point so the error story is
/// uniform across Anthropic / OpenAI Chat / OpenAI Responses /
/// Gemini / Bedrock Converse paths.
pub(super) fn parse_response_body(
    body: &[u8],
    codec_name: &'static str,
) -> Result<serde_json::Value> {
    serde_json::from_slice(body).map_err(|e| {
        const PEEK_BYTES: usize = 200;
        let peek_end = peek_at_char_boundary(body, PEEK_BYTES);
        let peek = body.get(..peek_end).map_or_else(String::new, |slice| {
            String::from_utf8_lossy(slice).into_owned()
        });
        let suffix = if body.len() > PEEK_BYTES { "…" } else { "" };
        crate::error::Error::provider_network(format!(
            "{codec_name} codec failed to decode response: {e}; \
             body was {} bytes; first {peek_end} bytes: {peek:?}{suffix} \
             — the response did not parse as JSON; the upstream may have \
             returned an HTML error page, a truncated body, or a wire \
             format the codec does not yet understand",
            body.len(),
        ))
    })
}

/// Find the largest `cut <= max` such that `body[..cut]` is valid
/// UTF-8 — non-ASCII bodies produce a `%` placeholder in the
/// middle of a multi-byte codepoint otherwise.
fn peek_at_char_boundary(body: &[u8], max: usize) -> usize {
    let mut cut = max.min(body.len());
    while cut > 0
        && body
            .get(..cut)
            .is_some_and(|slice| std::str::from_utf8(slice).is_err())
    {
        cut -= 1;
    }
    cut
}
