//! Verifies that `ChatModel` implements `Runnable<Vec<Message>, Message>`
//! and composes via `.pipe()`. Mock codec + transport defined inline.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use bytes::Bytes;
use entelix_core::codecs::{Codec, EncodedRequest};
use entelix_core::ir::{
    Capabilities, ContentPart, Message, ModelRequest, ModelResponse, ModelWarning, Role,
    StopReason, Usage,
};
use entelix_core::transports::{Transport, TransportResponse};
use entelix_core::{ChatModel, ExecutionContext, Result};
use entelix_runnable::{Runnable, RunnableExt, RunnableLambda};

struct StaticCodec;

impl Codec for StaticCodec {
    fn name(&self) -> &'static str {
        "static"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::default()
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        Ok(EncodedRequest::post_json(
            "/static",
            Bytes::from(serde_json::to_vec(request).unwrap()),
        ))
    }

    fn decode(&self, _body: &[u8], _warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        Ok(ModelResponse {
            id: "s".into(),
            model: "s".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text("Hello!")],
            usage: Usage::default(),
            rate_limit: None,
            warnings: Vec::new(),
            provider_echoes: Vec::new(),
        })
    }
}

struct OkTransport;

#[async_trait::async_trait]
impl Transport for OkTransport {
    fn name(&self) -> &'static str {
        "ok"
    }

    async fn send(
        &self,
        _request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        Ok(TransportResponse {
            status: 200,
            headers: http::HeaderMap::new(),
            body: Bytes::new(),
        })
    }
}

#[tokio::test]
async fn chat_model_invoke_via_runnable_trait() -> Result<()> {
    let model = ChatModel::new(StaticCodec, OkTransport, "static");
    let ctx = ExecutionContext::new();
    let reply: Message = Runnable::invoke(&model, vec![Message::user("hi")], &ctx).await?;
    assert!(matches!(reply.role, Role::Assistant));
    match &reply.content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "Hello!"),
        other => panic!("expected text, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn chat_model_pipes_into_lambda() -> Result<()> {
    let model = ChatModel::new(StaticCodec, OkTransport, "static");

    // After the model produces an assistant Message, extract the first text.
    let extract_text = RunnableLambda::new(|m: Message, _ctx| async move {
        let text = match m.content.into_iter().next() {
            Some(ContentPart::Text { text, .. }) => text,
            _ => String::new(),
        };
        Ok::<_, _>(text)
    });

    let chain = model.pipe(extract_text);
    let out: String = chain
        .invoke(vec![Message::user("hi")], &ExecutionContext::new())
        .await?;
    assert_eq!(out, "Hello!");
    Ok(())
}
