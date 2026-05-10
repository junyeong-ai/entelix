//! `VertexAnthropicCodec` — IR ⇄ Anthropic Messages API as routed
//! through Google Cloud Vertex AI's `:rawPredict` /
//! `:streamRawPredict` endpoints.
//!
//! Wire-format reference:
//! <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude>
//! and <https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai>.
//!
//! Two — and only two — wire-shape divergences from
//! [`super::AnthropicMessagesCodec`]:
//!
//! 1. The model id rides in the URL (the `VertexTransport` path
//!    component), so the request body must NOT carry a `model`
//!    field. Direct-Anthropic does send `model` in the body.
//! 2. `anthropic_version: "vertex-2023-10-16"` is required as a
//!    body field (not the `anthropic-version` header used direct).
//!
//! Every other surface — system prompt, messages, tool config,
//! cache_control blocks, extended-thinking, anthropic-beta headers,
//! SSE event stream, rate-limit headers — is identical to direct
//! Anthropic. The codec composes [`super::AnthropicMessagesCodec`]
//! and rewrites the encoded body in those two narrow ways; the
//! decode path is delegated unchanged because the response shape
//! mirrors the direct Messages API exactly.

use bytes::Bytes;
use http::HeaderName;
use serde_json::Value;

use crate::codecs::AnthropicMessagesCodec;
use crate::codecs::codec::{BoxByteStream, BoxDeltaStream, Codec, EncodedRequest};
use crate::error::Result;
use crate::ir::{Capabilities, ModelRequest, ModelResponse, ModelWarning, OutputStrategy};
use crate::rate_limit::RateLimitSnapshot;

/// Vertex AI's required body marker selecting the Anthropic-on-Vertex
/// wire contract. Cannot be omitted; cannot be overridden — the
/// vendor pins this constant per the published partner-model spec.
pub const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// Stateless codec for Anthropic Claude routed through GCP Vertex AI.
#[derive(Clone, Copy, Debug, Default)]
pub struct VertexAnthropicCodec {
    inner: AnthropicMessagesCodec,
}

impl VertexAnthropicCodec {
    /// Create a fresh codec instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: AnthropicMessagesCodec::new(),
        }
    }
}

impl Codec for VertexAnthropicCodec {
    fn name(&self) -> &'static str {
        "vertex-anthropic"
    }

    fn capabilities(&self, model: &str) -> Capabilities {
        // Vertex Anthropic exposes the same feature surface as
        // direct Anthropic for any given Claude model — capability
        // routing therefore delegates wholesale.
        self.inner.capabilities(model)
    }

    fn auto_output_strategy(&self, model: &str) -> OutputStrategy {
        self.inner.auto_output_strategy(model)
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let mut encoded = self.inner.encode(request)?;
        rewrite_for_vertex(&mut encoded, &request.model, false)?;
        Ok(encoded)
    }

    fn encode_streaming(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let mut encoded = self.inner.encode_streaming(request)?;
        rewrite_for_vertex(&mut encoded, &request.model, true)?;
        Ok(encoded)
    }

    fn decode_stream<'a>(
        &'a self,
        bytes: BoxByteStream<'a>,
        warnings_in: Vec<ModelWarning>,
    ) -> BoxDeltaStream<'a> {
        // Vertex serves the same SSE event stream as direct
        // Anthropic — the parser is unchanged.
        self.inner.decode_stream(bytes, warnings_in)
    }

    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        // Response shape is identical to direct Anthropic Messages.
        self.inner.decode(body, warnings_in)
    }

    fn extract_rate_limit(&self, headers: &http::HeaderMap) -> Option<RateLimitSnapshot> {
        // Vertex layers its own quota headers on top, but anything
        // Anthropic-shaped that flows through is identical — defer
        // to the direct codec's parser. (GCP-specific quota headers
        // surface through `VertexTransport`'s own snapshot, so the
        // codec stays vendor-agnostic.)
        self.inner.extract_rate_limit(headers)
    }
}

/// Apply the Vertex-Anthropic wire deltas to an encoded request the
/// inner [`AnthropicMessagesCodec`] just produced:
///
/// - drop the `model` body field (Vertex routes by URL path),
/// - inject `anthropic_version: "vertex-2023-10-16"`,
/// - strip the `anthropic-version` header (Vertex carries the
///   marker in the body instead),
/// - rewrite the path to the publisher-model resource the
///   `:rawPredict` / `:streamRawPredict` endpoints expect.
///
/// `anthropic-beta` headers are left in place — Vertex honours them
/// for extended-thinking / computer-use / cache-control variants.
///
/// The emitted path is partial (`/publishers/anthropic/models/{model}:rawPredict`)
/// — the GCP project + location prefix
/// (`/v1/projects/{project}/locations/{location}`) is the
/// `VertexTransport`'s responsibility because the codec is
/// project-agnostic by contract (invariant 5 — codecs operate on
/// neutral IR, transports own connection identity).
fn rewrite_for_vertex(encoded: &mut EncodedRequest, model: &str, streaming: bool) -> Result<()> {
    let mut body: Value = serde_json::from_slice(&encoded.body)?;
    let Value::Object(ref mut obj) = body else {
        return Err(crate::error::Error::invalid_request(
            "VertexAnthropicCodec: AnthropicMessagesCodec produced a non-object body",
        ));
    };
    obj.remove("model");
    obj.insert(
        "anthropic_version".to_owned(),
        Value::String(VERTEX_ANTHROPIC_VERSION.to_owned()),
    );
    encoded.body = Bytes::from(serde_json::to_vec(&body)?);
    encoded
        .headers
        .remove(HeaderName::from_static("anthropic-version"));

    let action = if streaming {
        "streamRawPredict"
    } else {
        "rawPredict"
    };
    encoded.path = format!("/publishers/anthropic/models/{model}:{action}");
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::ir::{Message, ModelRequest};

    fn parse(body: &Bytes) -> Value {
        serde_json::from_slice(body).expect("body must be JSON")
    }

    fn req() -> ModelRequest {
        ModelRequest {
            model: "claude-opus-4-7".into(),
            messages: vec![Message::user("hi")],
            max_tokens: Some(1024),
            ..ModelRequest::default()
        }
    }

    #[test]
    fn encode_replaces_model_with_anthropic_version_in_body() {
        let codec = VertexAnthropicCodec::new();
        let encoded = codec.encode(&req()).unwrap();
        let body = parse(&encoded.body);
        let obj = body.as_object().unwrap();
        assert_eq!(obj["anthropic_version"], "vertex-2023-10-16");
        assert!(
            !obj.contains_key("model"),
            "Vertex routes by URL path — `model` must NOT appear in body"
        );
    }

    #[test]
    fn encode_preserves_messages_and_max_tokens() {
        let codec = VertexAnthropicCodec::new();
        let body = parse(&codec.encode(&req()).unwrap().body);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn encode_strips_anthropic_version_header() {
        let codec = VertexAnthropicCodec::new();
        let encoded = codec.encode(&req()).unwrap();
        assert!(
            encoded.headers.get("anthropic-version").is_none(),
            "Vertex carries the version marker in the body — header must be stripped"
        );
    }

    #[test]
    fn encode_streaming_applies_same_rewrites() {
        let codec = VertexAnthropicCodec::new();
        let encoded = codec.encode_streaming(&req()).unwrap();
        assert!(encoded.streaming);
        let body = parse(&encoded.body);
        assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
        assert!(body.get("model").is_none());
    }

    #[test]
    fn decode_delegates_to_direct_anthropic_response_shape() {
        let codec = VertexAnthropicCodec::new();
        let body = serde_json::json!({
            "id": "msg_x",
            "model": "claude-opus-4-7",
            "stop_reason": "end_turn",
            "content": [{ "type": "text", "text": "Hello!" }],
            "usage": { "input_tokens": 4, "output_tokens": 1 }
        });
        let decoded = codec
            .decode(body.to_string().as_bytes(), Vec::new())
            .unwrap();
        assert_eq!(decoded.id, "msg_x");
        assert_eq!(decoded.usage.input_tokens, 4);
        assert_eq!(decoded.usage.output_tokens, 1);
    }
}
