//! End-to-end: a `ChatModelSamplingProvider<C, T>` driving a stub
//! `Codec` + stub `Transport` proves the IR conversion + dispatch +
//! response collapse all flow through correctly. No network, no
//! real LLM — the stub codec round-trips the request as the
//! response so the test asserts on the *adapter* shape, not on a
//! provider's behaviour.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use entelix_core::ChatModel;
use entelix_core::codecs::{Codec, EncodedRequest};
use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::ir::{
    Capabilities, ContentPart, ModelRequest, ModelResponse, ModelWarning, Role, StopReason, Usage,
};
use entelix_core::transports::{Transport, TransportResponse};
use entelix_mcp::{
    SamplingContent, SamplingMessage, SamplingProvider, SamplingRequest, StaticSamplingProvider,
};
use entelix_mcp_chatmodel::ChatModelSamplingProvider;

/// Stub codec — encodes to a placeholder body (the transport
/// ignores it) and decodes the captured `ModelRequest` snapshot
/// into a `ModelResponse` whose content is the FIRST user
/// message's text echoed back. The encode side records the
/// request the provider built so the test can assert on the
/// translated IR.
#[derive(Debug, Default)]
struct StubCodec {
    captured: Arc<Mutex<Option<ModelRequest>>>,
}

impl Codec for StubCodec {
    fn name(&self) -> &'static str {
        "stub"
    }
    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::default()
    }
    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        *self.captured.lock().unwrap() = Some(request.clone());
        Ok(EncodedRequest::post_json(
            "/stub",
            Bytes::from_static(b"{}"),
        ))
    }
    fn decode(&self, _body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let req = self.captured.lock().unwrap().clone().unwrap();
        // Echo back the last user text part as an assistant text reply.
        let echo_text = req
            .messages
            .last()
            .and_then(|m| {
                m.content.iter().find_map(|c| match c {
                    ContentPart::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_default();
        Ok(ModelResponse {
            id: "stub-response".into(),
            model: req.model,
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::Text {
                text: format!("echo: {echo_text}"),
                cache_control: None,
            }],
            usage: Usage::default(),
            rate_limit: None,
            warnings: warnings_in,
        })
    }
}

#[derive(Debug)]
struct StubTransport;

#[async_trait]
impl Transport for StubTransport {
    fn name(&self) -> &'static str {
        "stub"
    }
    async fn send(
        &self,
        _request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        Ok(TransportResponse {
            status: 200,
            headers: http::HeaderMap::new(),
            body: Bytes::from_static(b"{}"),
        })
    }
}

fn build_provider() -> (
    ChatModelSamplingProvider<StubCodec, StubTransport>,
    Arc<Mutex<Option<ModelRequest>>>,
) {
    let captured = Arc::new(Mutex::new(None));
    let codec = StubCodec {
        captured: Arc::clone(&captured),
    };
    let chat = ChatModel::new(codec, StubTransport, "test-model").with_max_tokens(512);
    let provider = ChatModelSamplingProvider::new(chat);
    (provider, captured)
}

#[tokio::test]
async fn sampling_request_translates_to_model_request_and_back() {
    let (provider, captured) = build_provider();

    let req = SamplingRequest {
        messages: vec![
            SamplingMessage {
                role: "user".into(),
                content: SamplingContent::Text {
                    text: "hello world".into(),
                },
            },
            SamplingMessage {
                role: "assistant".into(),
                content: SamplingContent::Text {
                    text: "hi back".into(),
                },
            },
            SamplingMessage {
                role: "user".into(),
                content: SamplingContent::Text {
                    text: "and again".into(),
                },
            },
        ],
        model_preferences: None,
        system_prompt: Some("be helpful".into()),
        include_context: None,
        temperature: Some(0.4),
        max_tokens: Some(128), // override the chat model's 512
        stop_sequences: vec!["STOP".into()],
        metadata: None,
    };

    let resp = provider.sample(req).await.unwrap();

    // Roundtrip echo of the LAST user message.
    assert_eq!(resp.role, "assistant");
    assert_eq!(resp.stop_reason, "endTurn");
    assert!(matches!(
        resp.content,
        SamplingContent::Text { ref text } if text == "echo: and again"
    ));
    assert_eq!(resp.model, "test-model");

    // Verify that the encode-time IR matches the per-request overrides.
    let snapshot = captured.lock().unwrap().clone().unwrap();
    assert_eq!(snapshot.model, "test-model");
    // Per-request system_prompt overrode the chat model default.
    let system_text = snapshot
        .system
        .blocks()
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        system_text.contains("be helpful"),
        "system override applied"
    );
    // temperature converted f64 → f32.
    assert!((snapshot.temperature.unwrap() - 0.4).abs() < 0.001);
    // max_tokens override (128, not 512).
    assert_eq!(snapshot.max_tokens, Some(128));
    // stop sequences carried through.
    assert_eq!(snapshot.stop_sequences, vec!["STOP".to_owned()]);
    // 3 messages, roles preserved in order.
    assert_eq!(snapshot.messages.len(), 3);
    assert_eq!(snapshot.messages[0].role, Role::User);
    assert_eq!(snapshot.messages[1].role, Role::Assistant);
    assert_eq!(snapshot.messages[2].role, Role::User);
}

#[tokio::test]
async fn invalid_role_returns_error_to_dispatcher() {
    let (provider, _captured) = build_provider();
    let req = SamplingRequest {
        messages: vec![SamplingMessage {
            role: "system".into(), // protocol violation
            content: SamplingContent::Text { text: "x".into() },
        }],
        model_preferences: None,
        system_prompt: None,
        include_context: None,
        temperature: None,
        max_tokens: None,
        stop_sequences: vec![],
        metadata: None,
    };
    let err = provider.sample(req).await.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("protocol-spec violation") || msg.contains("'system'"));
}

#[tokio::test]
async fn static_provider_still_satisfies_trait_object_use_path() {
    // Sanity: confirm that the StaticSamplingProvider in entelix-mcp
    // and the ChatModelSamplingProvider here both satisfy the same
    // SamplingProvider trait — operators can swap one for the other
    // without changing the wiring point.
    let (chat_provider, _) = build_provider();
    let static_provider = StaticSamplingProvider::text("alt-model", "ack");
    let providers: Vec<Box<dyn SamplingProvider>> =
        vec![Box::new(chat_provider), Box::new(static_provider)];
    assert_eq!(providers.len(), 2);
}
