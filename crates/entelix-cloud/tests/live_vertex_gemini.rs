//! Live-API smoke for [`VertexGeminiCodec`] + [`VertexTransport`].
//!
//! Exercises the Vertex AI publisher-model path that hosts Google
//! Gemini behind GCP authentication. The path-rewrite on this
//! codec ([`VertexGeminiCodec::encode`] swaps the direct
//! `/v1beta/models/{m}:generateContent` shape for the publisher-
//! partial `/publishers/google/models/{m}:generateContent`) plus
//! the project + location prefix injected by
//! `VertexTransport::resolve_url` are both verified end-to-end
//! against the live `:generateContent` endpoint.
//!
//! `#[ignore]`-gated and feature-gated behind `gcp` so a default
//! `cargo test --workspace` skips it. Run with:
//!
//! ```text
//! ENTELIX_LIVE_VERTEX_PROJECT=my-gcp-project \
//!     ENTELIX_LIVE_VERTEX_LOCATION=asia-northeast3 \
//!     cargo test -p entelix-cloud --features gcp \
//!         --test live_vertex_gemini -- --ignored
//! ```
//!
//! Application Default Credentials are picked up via
//! [`VertexCredentialProvider::default_chain`] — `gcloud auth
//! application-default login` on a workstation, workload identity
//! in GKE, ADC metadata server on GCE.
//!
//! Optional knobs:
//! - `ENTELIX_LIVE_VERTEX_GEMINI_MODEL` — override the default
//!   `gemini-3.1-pro` model id (Gemini families publish multiple
//!   tiers per region; the operator runs the model id Vertex
//!   exposes for their region).
//! - `ENTELIX_LIVE_VERTEX_QUOTA_PROJECT` — billing project
//!   override when ADC resolves a different identity than the one
//!   carrying the Vertex quota.
//!
//! ## What is verified
//!
//! 1. ADC resolves through [`VertexCredentialProvider::default_chain`]
//!    and feeds [`VertexTransport`].
//! 2. [`VertexGeminiCodec`] rewrites the path to the publisher-
//!    partial form (`/publishers/google/models/{model}:generateContent`)
//!    while leaving the body identical to direct Gemini.
//! 3. `VertexTransport::resolve_url` prefixes the partial path with
//!    `/v1/projects/{project}/locations/{location}` so the URL
//!    `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
//!    is hit on the wire.
//! 4. Vertex returns 2xx and the response decodes into a non-empty
//!    `ModelResponse` with a recognised `StopReason` and populated
//!    `Usage` counters.
//!
//! ## Cost discipline
//!
//! `max_tokens = 16` against the Gemini Flash tier. Per-run cost
//! well under $0.001.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::codecs::{Codec, VertexGeminiCodec};
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{Message, ModelRequest, StopReason};
use entelix_core::transports::Transport;

const DEFAULT_MODEL: &str = "gemini-3.1-pro";

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_gemini_minimal_chat_round_trip() {
    install_default_tls();

    let project = std::env::var("ENTELIX_LIVE_VERTEX_PROJECT").expect(
        "set ENTELIX_LIVE_VERTEX_PROJECT (GCP project hosting the Vertex Gemini model) to run this live smoke",
    );
    let location = std::env::var("ENTELIX_LIVE_VERTEX_LOCATION")
        .expect("set ENTELIX_LIVE_VERTEX_LOCATION (e.g. `asia-northeast3`, `us-central1`)");
    let model = std::env::var("ENTELIX_LIVE_VERTEX_GEMINI_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let credentials = VertexCredentialProvider::default_chain()
        .await
        .expect("ADC must resolve — run `gcloud auth application-default login` first");

    let mut builder = VertexTransport::builder()
        .with_project_id(&project)
        .with_location(&location)
        .with_token_refresher(Arc::new(credentials));
    if let Ok(qp) = std::env::var("ENTELIX_LIVE_VERTEX_QUOTA_PROJECT") {
        builder = builder.with_quota_project(qp);
    }
    let transport = builder
        .build()
        .expect("VertexTransport built from ADC chain");

    let codec = VertexGeminiCodec::new();
    // Gemini reasoning models spend part of `max_tokens` on the
    // thinking channel before producing visible text; budget
    // generously so both channels fit in one short turn.
    let request = ModelRequest {
        model: model.clone(),
        messages: vec![Message::user("Reply with the word 'ok' and nothing else.")],
        max_tokens: Some(256),
        temperature: Some(0.0),
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&request).expect("vertex-gemini encode");
    let ctx = ExecutionContext::new();
    let response = transport
        .send(encoded, &ctx)
        .await
        .expect("VertexTransport send to :generateContent");

    assert!(
        (200..300).contains(&response.status),
        "expected 2xx, got {} — body: {}. \
         Common causes: project lacks Vertex AI API enabled, the chosen \
         location does not host the requested Gemini SKU, or the model \
         id (`{}`) is unknown to that publisher catalog",
        response.status,
        String::from_utf8_lossy(&response.body),
        model,
    );

    let decoded = codec
        .decode(&response.body, Vec::new())
        .expect("vertex-gemini decode");

    // Codec/transport verification stays focused on the wire
    // contract — at least one decoded content part (Text,
    // Thinking, or ToolUse) and populated usage counters confirm
    // the round-trip. Whether the visible-text channel itself
    // contains the requested word is a Gemini behavioural test,
    // not an integration test for entelix.
    assert!(
        !decoded.content.is_empty(),
        "expected at least one decoded content part (text / thinking / tool_use)"
    );

    assert!(
        matches!(
            decoded.stop_reason,
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence { .. }
        ),
        "unexpected stop_reason on minimal call: {:?}",
        decoded.stop_reason
    );

    assert!(
        decoded.usage.input_tokens > 0,
        "usage.input_tokens should be populated on a real Vertex Gemini call"
    );
    assert!(
        decoded.usage.output_tokens > 0,
        "usage.output_tokens should be populated on a real Vertex Gemini call"
    );
}
