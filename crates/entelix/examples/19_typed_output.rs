//! `19_typed_output` — typed structured output through the Phase A
//! `complete_typed` / `complete_typed_validated` / `stream_typed`
//! surface.
//!
//! Build: `cargo build --example 19_typed_output -p entelix`
//! Run:   `cargo run   --example 19_typed_output -p entelix`
//!
//! Demonstrates three complementary entry points (ADR-0090 +
//! ADR-0091 + ADR-0096):
//!
//! 1. `complete_typed::<Reply>` — one-shot typed parsing. The
//!    response is decoded into the operator's `Reply` struct via
//!    `serde_json::from_str`; `with_validation_retries` enables a
//!    retry loop that reflects parse diagnostics back to the model
//!    on schema mismatch.
//! 2. `complete_typed_validated::<Reply, _>` — same path with an
//!    `OutputValidator<Reply>` (here, an inline closure) checking
//!    a cross-field invariant the JSON schema can't express.
//!    Validator failures route through `Error::ModelRetry` and
//!    share the retry budget.
//! 3. `stream_typed::<Reply>` — streaming counterpart. The
//!    consumer drains raw `StreamDelta`s for incremental display,
//!    then awaits the `completion` future for the parsed `Reply`.
//!    No retry on this path by design (re-invoking after deltas
//!    were emitted would surface a divergent second stream).
//!
//! No API key required — runs deterministically in CI through a
//! scripted stub codec (mirrors `03_state_graph` + `17_mcp_*`).

#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use entelix::codecs::{Codec, EncodedRequest};
use entelix::ir::{
    Capabilities, ContentPart, Message, ModelRequest, ModelResponse, ModelWarning, Role,
    StopReason, Usage,
};
use entelix::transports::{Transport, TransportResponse};
use entelix::{ChatModel, Error, ExecutionContext, LlmRenderable, Result};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

/// Operator's typed output schema. Reach for any
/// `JsonSchema + DeserializeOwned + Send + 'static`. The
/// `dead_code` allowance is for the example's own Debug-only
/// reads — production code uses these fields directly.
#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
struct Reply {
    answer: String,
    confidence: u32,
}

/// Stub codec — returns scripted responses. Real deployments swap
/// this for `AnthropicMessagesCodec` / `OpenAiChatCodec` / etc.
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
            r#"{"answer":"end","confidence":0}"#.to_owned()
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

#[async_trait]
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

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = ExecutionContext::new();
    let messages = vec![Message::new(
        Role::User,
        vec![ContentPart::text("Is the sky blue?")],
    )];

    // ── (1) complete_typed: one-shot, schema-mismatch retry ─────────
    //
    // First scripted response is malformed — `complete_typed` reflects
    // the parse diagnostic to the model and re-invokes within the
    // configured budget. Second response parses cleanly.
    println!("=== (1) complete_typed::<Reply> with validation_retries(1) ===");
    let codec = Arc::new(ScriptedCodec::new(vec![
        r#"{"oops": "wrong shape"}"#,
        r#"{"answer":"yes","confidence":80}"#,
    ]));
    let model = ChatModel::from_arc(codec, Arc::new(EmptyTransport), "stub")
        .with_validation_retries(1);

    let reply: Reply = model.complete_typed(messages.clone(), &ctx).await?;
    println!("  parsed: {reply:?}");

    // ── (2) complete_typed_validated: semantic validator ────────────
    //
    // A closure validator catches a cross-field invariant the JSON
    // schema can't express (confidence must be <= 100). Validator
    // failure raises `Error::ModelRetry` with a corrective hint —
    // shares the same retry budget as schema mismatch.
    println!("\n=== (2) complete_typed_validated::<Reply, _> ===");
    let codec = Arc::new(ScriptedCodec::new(vec![
        r#"{"answer":"maybe","confidence":150}"#, // out-of-range
        r#"{"answer":"maybe","confidence":60}"#,  // valid
    ]));
    let model = ChatModel::from_arc(codec, Arc::new(EmptyTransport), "stub")
        .with_validation_retries(1);

    let reply: Reply = model
        .complete_typed_validated(
            messages.clone(),
            |out: &Reply| {
                if out.confidence <= 100 {
                    Ok(())
                } else {
                    Err(Error::model_retry(
                        format!("confidence={} is out of range (0-100)", out.confidence)
                            .for_llm(),
                        0,
                    ))
                }
            },
            &ctx,
        )
        .await?;
    println!("  parsed (after validator retry): {reply:?}");

    // ── (3) stream_typed: deltas + typed completion ────────────────
    //
    // The consumer surfaces raw `StreamDelta`s as the model produces
    // them (text fragments echo to a UI in the real world); the typed
    // payload is available on the completion future once the stream
    // drains.
    println!("\n=== (3) stream_typed::<Reply> ===");
    let codec = Arc::new(ScriptedCodec::new(vec![r#"{"answer":"yes","confidence":95}"#]));
    let model = ChatModel::from_arc(codec, Arc::new(EmptyTransport), "stub");

    let typed_stream = model.stream_typed::<Reply>(messages, &ctx).await?;
    let mut stream = typed_stream.stream;
    let mut text_seen = String::new();
    while let Some(delta) = stream.next().await {
        if let entelix::stream::StreamDelta::TextDelta { text, .. } = delta? {
            text_seen.push_str(&text);
        }
    }
    drop(stream);

    let reply: Reply = typed_stream.completion.await?;
    println!("  raw text seen during streaming: {text_seen:?}");
    println!("  parsed completion: {reply:?}");

    println!("\nAll three typed-output entry points executed.");
    Ok(())
}
