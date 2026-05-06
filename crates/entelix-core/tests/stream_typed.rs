//! Regression tests for `ChatModel::stream_typed<O>` (slice 119,
//! ADR follow-on to 0079/0090). The streaming counterpart to
//! `complete_typed<O>` exposes raw `StreamDelta`s on `stream` for
//! incremental display and resolves a typed `O` on `completion`
//! after the stream drains.
//!
//! Locks two properties:
//!
//! 1. `stream` carries the model's deltas to the consumer
//!    unmodified (text fragments echo through the aggregator's
//!    tap).
//! 2. `completion` resolves to `Ok(O)` after the stream is
//!    drained and to `Err(Error::Serde(_))` on a schema-mismatch
//!    response — `stream_typed` does NOT retry (per the slice
//!    119 design note).

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Mutex;

use bytes::Bytes;
use entelix_core::codecs::{Codec, EncodedRequest};
use entelix_core::ir::{
    Capabilities, ContentPart, Message, ModelRequest, ModelResponse, ModelWarning, Role,
    StopReason, Usage,
};
use entelix_core::transports::{Transport, TransportResponse};
use entelix_core::{ChatModel, Error, ExecutionContext, Result};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Reply {
    answer: String,
    score: u32,
}

struct ScriptedCodec {
    script: Mutex<Vec<String>>,
}

impl ScriptedCodec {
    fn new(script: Vec<&str>) -> Self {
        Self {
            script: Mutex::new(script.into_iter().map(str::to_owned).collect()),
        }
    }
}

impl Codec for ScriptedCodec {
    fn name(&self) -> &'static str {
        "scripted"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::default()
    }

    fn encode(&self, _request: &ModelRequest) -> Result<EncodedRequest> {
        Ok(EncodedRequest::post_json("/scripted", Bytes::new()))
    }

    fn decode(&self, _body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let mut script = self.script.lock().unwrap();
        let text = if script.is_empty() {
            r#"{"answer":"end-of-script","score":0}"#.to_owned()
        } else {
            script.remove(0)
        };
        Ok(ModelResponse {
            id: "scripted".into(),
            model: "scripted".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text(text)],
            usage: Usage::default(),
            rate_limit: None,
            warnings: warnings_in,
        })
    }
}

struct EmptyTransport;

#[async_trait::async_trait]
impl Transport for EmptyTransport {
    fn name(&self) -> &'static str {
        "empty"
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
async fn stream_typed_resolves_completion_to_typed_value() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![
        r#"{"answer":"42","score":7}"#,
    ]));
    let model = ChatModel::from_arc(codec, std::sync::Arc::new(EmptyTransport), "test");
    let ctx = ExecutionContext::new();

    let typed_stream = model
        .stream_typed::<Reply>(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            &ctx,
        )
        .await
        .unwrap();

    // Drain the delta stream — without this the aggregator never
    // sees the terminal Stop and `completion` stays pending.
    let mut stream = typed_stream.stream;
    while let Some(delta) = stream.next().await {
        delta.unwrap();
    }
    drop(stream);

    let value: Reply = typed_stream.completion.await.unwrap();
    assert_eq!(value, Reply { answer: "42".to_owned(), score: 7 });
}

#[tokio::test]
async fn stream_typed_schema_mismatch_surfaces_serde_error() {
    // Stream sees the raw deltas as before; completion fails with
    // Error::Serde when the model's reply doesn't match the schema.
    // Unlike complete_typed, no retry — the consumer already
    // received deltas, re-invoking would emit a divergent stream.
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![
        r#"{"not_the_schema": true}"#,
    ]));
    let model = ChatModel::from_arc(codec, std::sync::Arc::new(EmptyTransport), "test")
        .with_validation_retries(2); // ignored on stream_typed by design
    let ctx = ExecutionContext::new();

    let typed_stream = model
        .stream_typed::<Reply>(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            &ctx,
        )
        .await
        .unwrap();

    let mut stream = typed_stream.stream;
    while let Some(delta) = stream.next().await {
        delta.unwrap();
    }
    drop(stream);

    let err = typed_stream.completion.await.unwrap_err();
    assert!(
        matches!(err, Error::Serde(_)),
        "expected Error::Serde on schema mismatch; got: {err:?}"
    );
}

#[tokio::test]
async fn stream_typed_emits_deltas_through_stream_field() {
    // The `stream` field surfaces raw StreamDelta events; this test
    // proves the consumer sees Start/text/Stop without inspecting
    // the typed completion. Operators echo TextDelta to a UI as
    // the model produces tokens.
    use entelix_core::stream::StreamDelta;

    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![
        r#"{"answer":"hi","score":1}"#,
    ]));
    let model = ChatModel::from_arc(codec, std::sync::Arc::new(EmptyTransport), "test");
    let ctx = ExecutionContext::new();

    let typed_stream = model
        .stream_typed::<Reply>(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            &ctx,
        )
        .await
        .unwrap();

    let mut stream = typed_stream.stream;
    let mut saw_start = false;
    let mut saw_stop = false;
    let mut text = String::new();
    while let Some(delta) = stream.next().await {
        match delta.unwrap() {
            StreamDelta::Start { .. } => saw_start = true,
            StreamDelta::TextDelta { text: t, .. } => text.push_str(&t),
            StreamDelta::Stop { .. } => saw_stop = true,
            _ => {}
        }
    }
    drop(stream);

    assert!(saw_start, "stream must surface Start");
    assert!(saw_stop, "stream must surface terminal Stop");
    assert_eq!(text, r#"{"answer":"hi","score":1}"#);

    let value: Reply = typed_stream.completion.await.unwrap();
    assert_eq!(value, Reply { answer: "hi".to_owned(), score: 1 });
}
