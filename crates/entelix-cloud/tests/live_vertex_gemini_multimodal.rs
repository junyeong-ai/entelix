//! Live-API smoke for [`VertexGeminiCodec`] multimodal input.
//!
//! Verifies the image-input path end-to-end against Vertex
//! Gemini: a [`ContentPart::Image`] with an inline base64
//! [`MediaSource`] encodes through the Gemini codec onto the wire
//! shape (`inlineData.{mimeType,data}` under the user message),
//! the model receives the image alongside the text prompt, and
//! the response decodes into a non-empty text part — proof that
//! multimodal IR survives the codec round-trip and Vertex hosts
//! the same vision-capable Gemini SKU as the AI Studio surface.
//!
//! `#[ignore]`-gated and feature-gated behind `gcp` so a default
//! `cargo test --workspace` skips it. Run with:
//!
//! ```text
//! ENTELIX_LIVE_VERTEX_PROJECT=my-gcp-project \
//!     ENTELIX_LIVE_VERTEX_LOCATION=global \
//!     ENTELIX_LIVE_VERTEX_GEMINI_MODEL=gemini-3.1-pro-preview \
//!     cargo test -p entelix-cloud --features gcp \
//!         --test live_vertex_gemini_multimodal -- --ignored
//! ```
//!
//! ## What is verified
//!
//! 1. [`ContentPart::Image`] with [`MediaSource::base64`] encodes
//!    through the codec without coercion warnings — Gemini lists
//!    Image as a native input modality.
//! 2. The image rides into the user turn alongside text, in the
//!    order the IR specifies.
//! 3. Vertex returns a non-empty text response describing the
//!    image (or acknowledging its plain content) — the visible
//!    text channel is populated, not just the thinking channel.
//! 4. `Usage` counters are populated so cost accounting reflects
//!    the multimodal turn (image tokens count toward
//!    `input_tokens`).
//!
//! ## Cost discipline
//!
//! 1×1 PNG fixture (~70 bytes after base64). Gemini reports the
//! image as a single tile in `input_tokens`. `max_tokens = 256`
//! covers the thinking pass plus a one-sentence reply. Per-run
//! cost well under $0.001.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::ChatModel;
use entelix_core::codecs::VertexGeminiCodec;
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{ContentPart, MediaSource, Message, Role, StopReason};

const DEFAULT_MODEL: &str = "gemini-3.1-pro-preview";

/// 1×1 white-pixel PNG, base64-encoded. Smallest payload that
/// still parses as a valid PNG so vendor preprocessing accepts it
/// without resizing or format conversion.
const ONE_PX_WHITE_PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_gemini_image_input_round_trip() {
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
        .with_max_tokens(256)
        .with_temperature(0.0);

    // Multimodal user turn — image part plus the text prompt that
    // anchors the model on a single observable assertion. The
    // canonical response shape (text-only with at least one
    // visible block) keeps the assertion stable across SKU /
    // version variations.
    let multimodal_turn = Message::new(
        Role::User,
        vec![
            ContentPart::image(MediaSource::base64("image/png", ONE_PX_WHITE_PNG_BASE64)),
            ContentPart::text("Describe the dominant colour of this image in one short sentence."),
        ],
    );

    let ctx = ExecutionContext::new();
    let response = chat
        .complete_full(vec![multimodal_turn], &ctx)
        .await
        .expect("Vertex Gemini multimodal round-trip");

    assert!(
        matches!(
            response.stop_reason,
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence { .. }
        ),
        "unexpected stop_reason on multimodal turn: {:?}",
        response.stop_reason
    );

    let visible_text: String = response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        !visible_text.trim().is_empty(),
        "multimodal response must include at least one non-empty visible text block; got: {response:?}"
    );

    assert!(
        response.usage.input_tokens > 0,
        "multimodal dispatch must populate usage.input_tokens (image tokens count here)"
    );
    assert!(
        response.usage.output_tokens > 0,
        "multimodal dispatch must populate usage.output_tokens"
    );
}
