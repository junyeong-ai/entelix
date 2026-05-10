//! `VertexGeminiCodec` — IR ⇄ Gemini API as routed through Google
//! Cloud Vertex AI's publisher-model endpoints.
//!
//! Wire-format reference:
//! <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#syntax>.
//!
//! One — and only one — wire-shape divergence from
//! [`super::GeminiCodec`]:
//!
//! 1. The model id rides in the URL path under
//!    `/publishers/google/models/{model}`. The body shape is
//!    identical between direct AI Studio Gemini and Vertex Gemini —
//!    Gemini does not carry the model in the body, only the URL,
//!    so encode delegates the body verbatim to the inner codec and
//!    only rewrites the path.
//!
//! Every other surface — multimodal content parts, system
//! instruction, tool config, function calling, structured output,
//! thinking budget, SSE event stream, rate-limit headers — is
//! identical to direct Gemini. The codec composes
//! [`super::GeminiCodec`] and rewrites the encoded path; decode
//! paths delegate unchanged because the response shape mirrors the
//! direct generateContent / streamGenerateContent API exactly.
//!
//! The emitted path is partial
//! (`/publishers/google/models/{model}:generateContent`) — the GCP
//! project + location prefix
//! (`/v1/projects/{project}/locations/{location}`) is the
//! `VertexTransport`'s responsibility because the codec is
//! project-agnostic by contract (invariant 5 — codecs operate on
//! neutral IR, transports own connection identity).

use crate::codecs::GeminiCodec;
use crate::codecs::codec::{BoxByteStream, BoxDeltaStream, Codec, EncodedRequest};
use crate::error::Result;
use crate::ir::{Capabilities, ModelRequest, ModelResponse, ModelWarning, OutputStrategy};
use crate::rate_limit::RateLimitSnapshot;

/// Stateless codec for Google Gemini routed through GCP Vertex AI.
#[derive(Clone, Copy, Debug, Default)]
pub struct VertexGeminiCodec {
    inner: GeminiCodec,
}

impl VertexGeminiCodec {
    /// Create a fresh codec instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: GeminiCodec::new(),
        }
    }
}

impl Codec for VertexGeminiCodec {
    fn name(&self) -> &'static str {
        "vertex-gemini"
    }

    fn capabilities(&self, model: &str) -> Capabilities {
        // Vertex Gemini exposes the same feature surface as direct
        // Gemini for any given model — capability routing therefore
        // delegates wholesale.
        self.inner.capabilities(model)
    }

    fn auto_output_strategy(&self, model: &str) -> OutputStrategy {
        self.inner.auto_output_strategy(model)
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let mut encoded = self.inner.encode(request)?;
        rewrite_path_for_vertex(&mut encoded, &request.model, false);
        Ok(encoded)
    }

    fn encode_streaming(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let mut encoded = self.inner.encode_streaming(request)?;
        rewrite_path_for_vertex(&mut encoded, &request.model, true);
        Ok(encoded)
    }

    fn decode_stream<'a>(
        &'a self,
        bytes: BoxByteStream<'a>,
        warnings_in: Vec<ModelWarning>,
    ) -> BoxDeltaStream<'a> {
        // Vertex serves the same SSE event stream as direct Gemini —
        // the parser is unchanged.
        self.inner.decode_stream(bytes, warnings_in)
    }

    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        // Response shape is identical to direct Gemini.
        self.inner.decode(body, warnings_in)
    }

    fn extract_rate_limit(&self, headers: &http::HeaderMap) -> Option<RateLimitSnapshot> {
        // Vertex layers its own quota headers on top, but anything
        // Gemini-shaped that flows through is identical — defer to
        // the direct codec's parser.
        self.inner.extract_rate_limit(headers)
    }
}

/// Rewrite the path the inner [`GeminiCodec`] just produced
/// (`/v1beta/models/{model}:{action}`) into the Vertex publisher-
/// model partial path (`/publishers/google/models/{model}:{action}`)
/// that `VertexTransport::resolve_url` then prefixes with the
/// project + location segments.
fn rewrite_path_for_vertex(encoded: &mut EncodedRequest, model: &str, streaming: bool) {
    let action = if streaming {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };
    encoded.path = format!("/publishers/google/models/{model}:{action}");
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::ir::{Message, ModelRequest};

    fn req() -> ModelRequest {
        ModelRequest {
            model: "gemini-3.1-pro".into(),
            messages: vec![Message::user("hi")],
            max_tokens: Some(16),
            ..ModelRequest::default()
        }
    }

    #[test]
    fn encode_emits_publisher_partial_path() {
        let codec = VertexGeminiCodec::new();
        let encoded = codec.encode(&req()).unwrap();
        assert_eq!(
            encoded.path,
            "/publishers/google/models/gemini-2.5-flash:generateContent",
            "Vertex Gemini codec must emit the publisher-partial path so VertexTransport can prefix project + location"
        );
    }

    #[test]
    fn encode_streaming_emits_publisher_partial_path_with_sse_alt() {
        let codec = VertexGeminiCodec::new();
        let encoded = codec.encode_streaming(&req()).unwrap();
        assert!(encoded.streaming);
        assert_eq!(
            encoded.path,
            "/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
        );
    }

    #[test]
    fn encode_body_delegates_to_inner_unchanged() {
        let codec = VertexGeminiCodec::new();
        let direct = GeminiCodec::new();
        let body_v = codec.encode(&req()).unwrap().body;
        let body_d = direct.encode(&req()).unwrap().body;
        assert_eq!(
            body_v, body_d,
            "Vertex Gemini body shape is identical to direct Gemini — only the URL path differs"
        );
    }
}
