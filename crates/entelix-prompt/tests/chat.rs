//! `ChatPromptTemplate` + `MessagesPlaceholder` tests, including a
//! `.pipe()` composition with a mock `ChatModel`.

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
use entelix_core::{ChatModel, Error, ExecutionContext, Result};
use entelix_prompt::{
    ChatPromptPart, ChatPromptTemplate, MessagesPlaceholder, PromptValue, PromptVars,
};
use entelix_runnable::{Runnable, RunnableExt};

fn vars(items: &[(&str, PromptValue)]) -> PromptVars {
    items
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

#[test]
fn from_messages_renders_system_user_pair() -> Result<()> {
    let prompt = ChatPromptTemplate::from_messages(vec![
        ChatPromptPart::system("You translate to {{ language }}.")?,
        ChatPromptPart::user("{{ question }}")?,
    ]);

    let v = vars(&[
        ("language", PromptValue::from("Korean")),
        ("question", PromptValue::from("How are you?")),
    ]);
    let messages = prompt.render(&v)?;
    assert_eq!(messages.len(), 2);
    assert!(matches!(messages[0].role, Role::System));
    match &messages[0].content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "You translate to Korean."),
        other => panic!("expected text, got {other:?}"),
    }
    assert!(matches!(messages[1].role, Role::User));
    match &messages[1].content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "How are you?"),
        other => panic!("expected text, got {other:?}"),
    }
    Ok(())
}

#[test]
fn messages_placeholder_splices_history() -> Result<()> {
    let prompt = ChatPromptTemplate::from_messages(vec![
        ChatPromptPart::system("Be concise.")?,
        ChatPromptPart::placeholder("history"),
        ChatPromptPart::user("{{ question }}")?,
    ]);

    let history = vec![
        Message::user("first turn"),
        Message::assistant("first reply"),
    ];
    let v = vars(&[
        ("history", PromptValue::from(history)),
        ("question", PromptValue::from("second turn")),
    ]);

    let out = prompt.render(&v)?;
    assert_eq!(out.len(), 4); // system + 2 history + user
    assert!(matches!(out[0].role, Role::System));
    assert!(matches!(out[1].role, Role::User));
    assert!(matches!(out[2].role, Role::Assistant));
    assert!(matches!(out[3].role, Role::User));
    Ok(())
}

#[test]
fn placeholder_with_text_value_returns_invalid_request() {
    let prompt = ChatPromptTemplate::from_messages(vec![ChatPromptPart::placeholder("history")]);
    // History should be Messages but supplied as Text → error.
    let v = vars(&[("history", PromptValue::from("oops"))]);
    let err = prompt.render(&v).unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[test]
fn templated_part_with_messages_value_returns_invalid_request() -> Result<()> {
    let prompt = ChatPromptTemplate::from_messages(vec![ChatPromptPart::system("hi {{ x }}")?]);
    let v = vars(&[("x", PromptValue::from(vec![Message::user("oops")]))]);
    let err = prompt.render(&v).unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
    Ok(())
}

#[test]
fn missing_placeholder_variable_errors() {
    let prompt = ChatPromptTemplate::from_messages(vec![ChatPromptPart::placeholder("history")]);
    let err = prompt.render(&PromptVars::new()).unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[test]
fn placeholder_accessors() {
    let p = MessagesPlaceholder::new("turns");
    assert_eq!(p.name(), "turns");
}

#[tokio::test]
async fn implements_runnable() -> Result<()> {
    let prompt = ChatPromptTemplate::from_messages(vec![ChatPromptPart::user("hi {{ name }}")?]);
    let ctx = ExecutionContext::new();
    let v = vars(&[("name", PromptValue::from("there"))]);
    let messages = Runnable::invoke(&prompt, v, &ctx).await?;
    assert_eq!(messages.len(), 1);
    Ok(())
}

// ── pipe composition with a mock ChatModel ─────────────────────────────────

struct EchoCodec;

impl Codec for EchoCodec {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::default()
    }
    fn encode(&self, req: &ModelRequest) -> Result<EncodedRequest> {
        Ok(EncodedRequest::post_json(
            "/echo",
            Bytes::from(serde_json::to_vec(req).unwrap()),
        ))
    }
    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let req: ModelRequest = serde_json::from_slice(body).unwrap();
        let echoed = req
            .messages
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .and_then(|m| m.content.first())
            .and_then(|c| match c {
                ContentPart::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        Ok(ModelResponse {
            id: "x".into(),
            model: "x".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text(echoed)],
            usage: Usage::default(),
            rate_limit: None,
            warnings: warnings_in,
        })
    }
}

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
async fn prompt_pipes_into_chat_model() -> Result<()> {
    let prompt = ChatPromptTemplate::from_messages(vec![
        ChatPromptPart::system("Be brief.")?,
        ChatPromptPart::user("{{ question }}")?,
    ]);
    let model = ChatModel::new(EchoCodec, LoopbackTransport, "echo-model");

    let chain = prompt.pipe(model);
    let v = vars(&[("question", PromptValue::from("ping"))]);

    let reply: Message = chain.invoke(v, &ExecutionContext::new()).await?;
    assert!(matches!(reply.role, Role::Assistant));
    match &reply.content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "ping"),
        other => panic!("expected text, got {other:?}"),
    }
    Ok(())
}
