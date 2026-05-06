//! Regression tests for `complete_typed<O>` schema-mismatch retry
//! loop (slice 106 / ADR-0090). Operator wires
//! `with_validation_retries(n)` on the chat model; schema-mismatch
//! `Error::Serde` failures reflect the parse diagnostic to the
//! model as a corrective user message and re-invoke up to `n`
//! times before bubbling the final error.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use std::sync::Mutex;

use bytes::Bytes;
use entelix_core::codecs::{Codec, EncodedRequest};
use entelix_core::ir::{
    Capabilities, ContentPart, Message, ModelRequest, ModelResponse, ModelWarning, Role,
    StopReason, Usage,
};
use entelix_core::transports::{Transport, TransportResponse};
use entelix_core::{ChatModel, ExecutionContext, Result};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Reply {
    answer: String,
    score: u32,
}

/// Codec that returns scripted text responses sequentially. Unused
/// `schemas` / `response_format` are tolerated — the test exercises
/// the *parse* path on the response, not the wire encode.
struct ScriptedCodec {
    script: Mutex<Vec<String>>,
    calls: Mutex<u32>,
}

impl ScriptedCodec {
    fn new(script: Vec<&str>) -> Self {
        Self {
            script: Mutex::new(script.into_iter().map(str::to_owned).collect()),
            calls: Mutex::new(0),
        }
    }

    fn call_count(&self) -> u32 {
        *self.calls.lock().unwrap()
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
        drop(script);
        *self.calls.lock().unwrap() += 1;
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

/// Transport that returns an empty 200 — `ScriptedCodec` doesn't
/// inspect the wire payload.
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
async fn schema_mismatch_with_zero_retries_surfaces_serde_error() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![r#"{"not_the_schema": true}"#]));
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(EmptyTransport), "test");
    let ctx = ExecutionContext::new();
    let err = model
        .complete_typed::<Reply>(vec![Message::new(Role::User, vec![ContentPart::text("ask")])], &ctx)
        .await
        .unwrap_err();
    assert!(matches!(err, entelix_core::Error::Serde(_)), "got: {err:?}");
    assert_eq!(codec.call_count(), 1);
}

#[tokio::test]
async fn schema_mismatch_with_retries_recovers_when_second_attempt_valid() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![
        r#"{"not_the_schema": true}"#,
        r#"{"answer":"42","score":7}"#,
    ]));
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(EmptyTransport), "test")
        .with_validation_retries(2);
    let ctx = ExecutionContext::new();
    let out: Reply = model
        .complete_typed(vec![Message::new(Role::User, vec![ContentPart::text("ask")])], &ctx)
        .await
        .unwrap();
    assert_eq!(out, Reply { answer: "42".to_owned(), score: 7 });
    assert_eq!(codec.call_count(), 2);
}

#[tokio::test]
async fn schema_mismatch_retries_exhausted_surfaces_final_serde_error() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![
        r#"{"first": "bad"}"#,
        r#"{"second": "still bad"}"#,
        r#"{"third": "still bad"}"#,
    ]));
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(EmptyTransport), "test")
        .with_validation_retries(2);
    let ctx = ExecutionContext::new();
    let err = model
        .complete_typed::<Reply>(vec![Message::new(Role::User, vec![ContentPart::text("ask")])], &ctx)
        .await
        .unwrap_err();
    assert!(matches!(err, entelix_core::Error::Serde(_)), "got: {err:?}");
    assert_eq!(codec.call_count(), 3, "initial + 2 retries");
}
