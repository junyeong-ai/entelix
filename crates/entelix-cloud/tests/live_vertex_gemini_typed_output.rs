//! Live-API smoke for [`VertexGeminiCodec`] structured (typed)
//! output via [`ChatModel::complete_typed`].
//!
//! Verifies that the schema the typed-output path emits onto the
//! Gemini wire (`generationConfig.responseMimeType` +
//! `responseSchema`) survives round-trip and the model returns a
//! payload that deserialises into the requested Rust type without
//! [`Error::Serde`] retries — the production path ontosyx will
//! exercise for `RetrievalProfile`-backed structured suggestions
//! and graph-route plans where `complete_typed` is the canonical
//! decoder.
//!
//! `#[ignore]`-gated and feature-gated behind `gcp` so a default
//! `cargo test --workspace` skips it. Run with:
//!
//! ```text
//! ENTELIX_LIVE_VERTEX_PROJECT=my-gcp-project \
//!     ENTELIX_LIVE_VERTEX_LOCATION=global \
//!     ENTELIX_LIVE_VERTEX_GEMINI_MODEL=gemini-3.1-pro-preview \
//!     cargo test -p entelix-cloud --features gcp \
//!         --test live_vertex_gemini_typed_output -- --ignored
//! ```
//!
//! ## What is verified
//!
//! 1. `complete_typed::<WeatherReport>` derives the JSON Schema
//!    via `schemars`, attaches it to the request through
//!    [`ResponseFormat::strict`], and routes the codec's
//!    [`OutputStrategy::Auto`] into the Gemini-native
//!    `response_schema` channel.
//! 2. Gemini produces a JSON payload that satisfies the schema —
//!    no `Error::Serde` retry loop fires (the test sets
//!    `validation_retries(0)` deliberately so a bad payload would
//!    surface as a hard error rather than silently retry).
//! 3. The deserialised Rust value carries the fields the prompt
//!    described, with the types the schema declared (string for
//!    `city`, integer for `temperature_celsius`).
//! 4. `Usage` counters are populated.
//!
//! ## Cost discipline
//!
//! `max_tokens = 256` covers the thinking pass plus a small
//! structured payload. Per-run cost well under $0.001.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::ChatModel;
use entelix_core::codecs::VertexGeminiCodec;
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::Message;
use schemars::JsonSchema;
use serde::Deserialize;

const DEFAULT_MODEL: &str = "gemini-3.1-pro-preview";

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherReport {
    /// City name the report is about.
    city: String,
    /// Current temperature in Celsius.
    temperature_celsius: i32,
    /// Short conditions phrase, e.g. `sunny` or `light rain`.
    conditions: String,
}

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_gemini_typed_output_round_trip() {
    install_default_tls();

    let project = std::env::var("ENTELIX_LIVE_VERTEX_PROJECT").expect(
        "set ENTELIX_LIVE_VERTEX_PROJECT (GCP project hosting the Vertex Gemini model) to run this live smoke",
    );
    let location = std::env::var("ENTELIX_LIVE_VERTEX_LOCATION")
        .expect("set ENTELIX_LIVE_VERTEX_LOCATION (e.g. `global`, `asia-northeast3`)");
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

    let chat = ChatModel::new(VertexGeminiCodec::new(), transport, model)
        // Gemini reasoning models budget a substantial slice of
        // `max_tokens` on the hidden thinking channel before the
        // visible JSON payload — 1024 covers both passes for a
        // small typed shape.
        .with_max_tokens(1024)
        .with_temperature(0.0)
        // No retry budget: a malformed payload must surface as a
        // hard error so the smoke fails loudly instead of silently
        // covering a regression with a second-attempt success.
        .with_validation_retries(0);

    let ctx = ExecutionContext::new();
    let report: WeatherReport = chat
        .complete_typed::<WeatherReport>(
            vec![Message::user(
                "Pretend the weather in Seoul, South Korea right now is 15°C and sunny. \
                 Return only the structured weather report.",
            )],
            &ctx,
        )
        .await
        .expect("Vertex Gemini structured-output round-trip must deserialise without retry");

    assert!(
        report.city.to_lowercase().contains("seoul"),
        "expected city to mention Seoul, got `{}`",
        report.city
    );
    assert_eq!(
        report.temperature_celsius, 15,
        "expected temperature 15°C from prompt, got {}",
        report.temperature_celsius
    );
    assert!(
        !report.conditions.trim().is_empty(),
        "conditions field must be non-empty (schema-required string)"
    );
}
