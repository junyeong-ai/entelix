//! Smoke tests for the `Codec` and `Transport` traits.
//!
//! Defines mock impls inline and validates encode/decode + send round-trip
//! at the trait level. Production codec / transport impls
//! (`AnthropicMessagesCodec`, `DirectTransport`, ...) carry their own
//! integration suites.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use bytes::Bytes;
use entelix_core::codecs::{Codec, EncodedRequest};
use entelix_core::ir::{
    Capabilities, ContentPart, Message, ModelRequest, ModelResponse, ModelWarning, StopReason,
    Usage,
};
use entelix_core::transports::{Transport, TransportResponse};
use entelix_core::{ExecutionContext, Result};

/// Echo codec: encodes the first user message verbatim, decodes any JSON body
/// as `text` content. Used to validate that the trait shapes are usable.
struct EchoCodec;

impl Codec for EchoCodec {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities {
            system_prompt: true,
            max_context_tokens: 4096,
            ..Default::default()
        }
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let body = serde_json::to_vec(request).unwrap();
        let mut req = EncodedRequest::post_json("/echo", Bytes::from(body));
        // Demonstrate the warning path — we "lose" temperature on the wire.
        if request.temperature.is_some() {
            req.warnings.push(ModelWarning::LossyEncode {
                field: "temperature".into(),
                detail: "echo codec ignores sampling parameters".into(),
            });
        }
        Ok(req)
    }

    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let req: ModelRequest = serde_json::from_slice(body)?;
        let echoed = req
            .messages
            .iter()
            .find(|m| matches!(m.role, entelix_core::ir::Role::User))
            .and_then(|m| m.content.first())
            .and_then(|c| match c {
                ContentPart::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        Ok(ModelResponse {
            id: "echo_1".into(),
            model: req.model,
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text(echoed)],
            usage: Usage::default(),
            rate_limit: None,
            warnings: warnings_in,
            provider_echoes: Vec::new(),
        })
    }
}

/// Loop-back transport: returns the request body unmodified as the response
/// body. Lets us close the encode → send → decode loop in tests.
struct LoopbackTransport;

#[async_trait::async_trait]
impl Transport for LoopbackTransport {
    fn name(&self) -> &'static str {
        "loopback"
    }

    async fn send(
        &self,
        request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        Ok(TransportResponse {
            status: 200,
            headers: http::HeaderMap::new(),
            body: request.body,
        })
    }
}

#[tokio::test]
async fn codec_encode_decode_roundtrips_via_loopback() -> Result<()> {
    let codec = EchoCodec;
    let transport = LoopbackTransport;
    let ctx = ExecutionContext::new();

    let req = ModelRequest {
        model: "echo-1".into(),
        messages: vec![Message::user("ping")],
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&req)?;
    assert_eq!(encoded.path, "/echo");
    assert_eq!(encoded.method, http::Method::POST);
    assert_eq!(
        encoded.headers.get(http::header::CONTENT_TYPE).unwrap(),
        "application/json"
    );

    let warnings = encoded.warnings.clone();
    let response = transport.send(encoded, &ctx).await?;
    assert_eq!(response.status, 200);

    let decoded = codec.decode(&response.body, warnings)?;
    assert_eq!(decoded.model, "echo-1");
    assert!(matches!(decoded.stop_reason, StopReason::EndTurn));
    assert_eq!(decoded.content.len(), 1);
    let text = match &decoded.content[0] {
        ContentPart::Text { text, .. } => text.as_str(),
        _ => panic!("expected text content"),
    };
    assert_eq!(text, "ping");
    Ok(())
}

#[tokio::test]
async fn codec_emits_lossy_warning_for_dropped_field() -> Result<()> {
    let codec = EchoCodec;
    let req = ModelRequest {
        model: "echo-1".into(),
        messages: vec![Message::user("hi")],
        temperature: Some(0.5),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req)?;
    assert_eq!(encoded.warnings.len(), 1);
    assert!(matches!(
        encoded.warnings[0],
        ModelWarning::LossyEncode { ref field, .. } if field == "temperature"
    ));
    Ok(())
}

#[tokio::test]
async fn codec_capabilities_lookup_works() {
    let codec = EchoCodec;
    let caps = codec.capabilities("any-model");
    assert!(caps.system_prompt);
    assert!(!caps.streaming);
    assert_eq!(caps.max_context_tokens, 4096);
}

#[tokio::test]
async fn codec_and_transport_object_safe_via_arc_dyn() {
    use std::sync::Arc;
    let _codecs: Vec<Arc<dyn Codec>> = vec![Arc::new(EchoCodec)];
    let _transports: Vec<Arc<dyn Transport>> = vec![Arc::new(LoopbackTransport)];
}
