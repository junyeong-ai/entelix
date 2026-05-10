//! Live-API smoke for [`VertexAnthropicCodec`] + [`VertexTransport`].
//!
//! Exercises the Vertex AI partner-model path that hosts Anthropic
//! Claude models behind GCP authentication. Two wire deltas vs the
//! direct Anthropic Messages API ride this codec
//! ([`VertexAnthropicCodec::encode`] drops `model` from the body
//! and injects `anthropic_version: "vertex-2023-10-16"`); this
//! smoke proves both still satisfy the live `:rawPredict`
//! endpoint.
//!
//! `#[ignore]`-gated and feature-gated behind `gcp` so a default
//! `cargo test --workspace` skips it. Run with:
//!
//! ```text
//! ENTELIX_LIVE_VERTEX_PROJECT=my-gcp-project \
//!     ENTELIX_LIVE_VERTEX_LOCATION=us-east5 \
//!     cargo test -p entelix-cloud --features gcp \
//!         --test live_vertex_anthropic -- --ignored
//! ```
//!
//! Application Default Credentials are picked up via
//! [`VertexCredentialProvider::default_chain`] — `gcloud auth
//! application-default login` on a workstation, workload identity
//! in GKE, ADC metadata server on GCE.
//!
//! Optional knobs:
//! - `ENTELIX_LIVE_VERTEX_MODEL` — override the default
//!   `claude-haiku-4-5@20250514` model id (Vertex pins each
//!   Claude SKU with the publisher-version suffix; the operator
//!   runs the model id Vertex publishes for their region).
//! - `ENTELIX_LIVE_VERTEX_QUOTA_PROJECT` — billing project
//!   override when ADC resolves a different identity than the one
//!   carrying the Vertex quota.
//!
//! ## What is verified
//!
//! 1. ADC resolves through [`VertexCredentialProvider::default_chain`]
//!    and feeds [`VertexTransport`].
//! 2. [`VertexAnthropicCodec`] encodes the body with the
//!    `anthropic_version` marker and *without* the `model` field
//!    (Vertex routes by URL path).
//! 3. Vertex `:rawPredict` returns 2xx and the response decodes
//!    into a non-empty `ModelResponse` with a recognised
//!    `StopReason` and populated `Usage` counters — confirming
//!    the dropped header (`anthropic-version` is stripped at
//!    encode time) does not leak as a 4xx.
//!
//! ## Cost discipline
//!
//! `max_tokens = 16` against the Haiku tier. Vertex passes through
//! Anthropic's pricing — per-run cost well under $0.001.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::codecs::{Codec, VertexAnthropicCodec};
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{ContentPart, Message, ModelRequest, StopReason};
use entelix_core::transports::Transport;

const DEFAULT_MODEL: &str = "claude-haiku-4-5@20250514";

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_anthropic_minimal_chat_round_trip() {
    install_default_tls();
    let project = std::env::var("ENTELIX_LIVE_VERTEX_PROJECT").expect(
        "set ENTELIX_LIVE_VERTEX_PROJECT (GCP project hosting the Vertex Claude model) to run this live smoke",
    );
    let location = std::env::var("ENTELIX_LIVE_VERTEX_LOCATION")
        .expect("set ENTELIX_LIVE_VERTEX_LOCATION (e.g. `us-east5`, `europe-west1`)");
    let model =
        std::env::var("ENTELIX_LIVE_VERTEX_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

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

    let codec = VertexAnthropicCodec::new();
    let request = ModelRequest {
        model: model.clone(),
        messages: vec![Message::user("Reply with the word 'ok' and nothing else.")],
        max_tokens: Some(16),
        temperature: Some(0.0),
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&request).expect("vertex-anthropic encode");
    let ctx = ExecutionContext::new();
    let response = transport
        .send(encoded, &ctx)
        .await
        .expect("VertexTransport send to :rawPredict");

    assert!(
        (200..300).contains(&response.status),
        "expected 2xx, got {} — body: {}. \
         Common causes: model id missing the `@<publisher-version>` suffix, \
         the project lacks Vertex AI API enabled, \
         or the chosen location does not host the requested Claude SKU",
        response.status,
        String::from_utf8_lossy(&response.body)
    );

    let decoded = codec
        .decode(&response.body, Vec::new())
        .expect("vertex-anthropic decode");

    let saw_text = decoded
        .content
        .iter()
        .any(|p| matches!(p, ContentPart::Text { text, .. } if !text.is_empty()));
    assert!(saw_text, "expected at least one non-empty text block");

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
        "usage.input_tokens should be populated on a real Vertex call"
    );
    assert!(
        decoded.usage.output_tokens > 0,
        "usage.output_tokens should be populated on a real Vertex call"
    );
}
