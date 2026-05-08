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
use entelix_core::{ChatModel, Error, ExecutionContext, LlmRenderable, Result};
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
        .complete_typed::<Reply>(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            &ctx,
        )
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
        .complete_typed(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            &ctx,
        )
        .await
        .unwrap();
    assert_eq!(
        out,
        Reply {
            answer: "42".to_owned(),
            score: 7
        }
    );
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
        .complete_typed::<Reply>(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            &ctx,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, entelix_core::Error::Serde(_)), "got: {err:?}");
    assert_eq!(codec.call_count(), 3, "initial + 2 retries");
}

// ── complete_typed_validated — slice 106b coverage ─────────────────

#[tokio::test]
async fn validator_accepts_first_response_with_zero_retries() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![r#"{"answer":"yes","score":50}"#]));
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(EmptyTransport), "test");
    let ctx = ExecutionContext::new();
    let out: Reply = model
        .complete_typed_validated(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            |out: &Reply| {
                if out.score <= 100 {
                    Ok(())
                } else {
                    Err(Error::model_retry("score too high".to_owned().for_llm(), 0))
                }
            },
            &ctx,
        )
        .await
        .unwrap();
    assert_eq!(
        out,
        Reply {
            answer: "yes".to_owned(),
            score: 50
        }
    );
    assert_eq!(codec.call_count(), 1);
}

#[tokio::test]
async fn validator_rejection_triggers_retry_within_budget() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![
        // First response parses OK, validator rejects (score > 100).
        r#"{"answer":"yes","score":250}"#,
        // Second response passes both parse and validator.
        r#"{"answer":"yes","score":42}"#,
    ]));
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(EmptyTransport), "test")
        .with_validation_retries(2);
    let ctx = ExecutionContext::new();
    let out: Reply = model
        .complete_typed_validated(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            |out: &Reply| {
                if out.score <= 100 {
                    Ok(())
                } else {
                    Err(Error::model_retry(
                        "score must be 0-100".to_owned().for_llm(),
                        0,
                    ))
                }
            },
            &ctx,
        )
        .await
        .unwrap();
    assert_eq!(
        out,
        Reply {
            answer: "yes".to_owned(),
            score: 42
        }
    );
    assert_eq!(
        codec.call_count(),
        2,
        "validator triggered exactly one retry"
    );
}

#[tokio::test]
async fn validator_persistent_rejection_exhausts_budget_and_surfaces_model_retry() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![
        r#"{"answer":"x","score":250}"#,
        r#"{"answer":"x","score":260}"#,
        r#"{"answer":"x","score":270}"#,
    ]));
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(EmptyTransport), "test")
        .with_validation_retries(2);
    let ctx = ExecutionContext::new();
    let err = model
        .complete_typed_validated(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            |out: &Reply| {
                if out.score <= 100 {
                    Ok(())
                } else {
                    Err(Error::model_retry(
                        format!("score {} > 100", out.score).for_llm(),
                        0,
                    ))
                }
            },
            &ctx,
        )
        .await
        .unwrap_err();
    // After exhausting retries, the loop surfaces the validator's
    // last `ModelRetry` unchanged (no special wrapping).
    assert!(matches!(err, Error::ModelRetry { .. }), "got: {err:?}");
    assert_eq!(codec.call_count(), 3, "initial + 2 retries");
}

#[tokio::test]
async fn validator_non_model_retry_error_bubbles_unchanged() {
    let codec = std::sync::Arc::new(ScriptedCodec::new(vec![r#"{"answer":"x","score":50}"#]));
    let model = ChatModel::from_arc(codec.clone(), std::sync::Arc::new(EmptyTransport), "test")
        .with_validation_retries(3); // budget unused — non-retry error
    let ctx = ExecutionContext::new();
    let err = model
        .complete_typed_validated(
            vec![Message::new(Role::User, vec![ContentPart::text("ask")])],
            |_out: &Reply| Err(Error::config("operator-side validator config")),
            &ctx,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "got: {err:?}");
    assert_eq!(codec.call_count(), 1, "non-retry error short-circuits");
}
