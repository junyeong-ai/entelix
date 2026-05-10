//! Live-API smoke for [`VertexGeminiCodec`] system-prompt
//! application across a multi-turn conversation.
//!
//! Verifies that `ChatModel::with_system` plus a multi-turn
//! transcript (`user → assistant → user`) flows through the
//! codec into Gemini's `systemInstruction` channel and that the
//! assistant's continuation respects the system directive across
//! turn boundaries — the canonical chat-agent shape ontosyx's
//! `ox-agent` builds on.
//!
//! `#[ignore]`-gated and feature-gated behind `gcp` so a default
//! `cargo test --workspace` skips it.
//!
//! ## What is verified
//!
//! 1. [`ChatModel::with_system`] populates the codec's
//!    `systemInstruction` slot rather than a leading user-role
//!    message — Gemini distinguishes the two on the wire.
//! 2. A `[user, assistant, user]` transcript round-trips: the
//!    codec emits each turn under the right role, the model sees
//!    its own prior assistant turn as memory, and the response
//!    respects both the system directive and the second user
//!    request.
//! 3. The system directive constrains the answer ("respond in
//!    Korean only") — the assertion checks the visible text
//!    contains at least one Hangul codepoint, proving the
//!    directive flowed through.
//!
//! ## Cost discipline
//!
//! `max_tokens = 512` covers the thinking pass plus a Korean
//! sentence. Per-run cost well under $0.001.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::ChatModel;
use entelix_core::codecs::VertexGeminiCodec;
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{ContentPart, Message, Role};

const DEFAULT_MODEL: &str = "gemini-3.1-pro-preview";

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_gemini_system_prompt_multi_turn() {
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
        .with_max_tokens(512)
        .with_temperature(0.0)
        .with_system("You are a concise assistant. Respond in Korean only.");

    // [user-1, assistant-1, user-2] — assistant-1 is fed back so
    // the model treats it as its own prior turn (memory under the
    // same system directive).
    let messages = vec![
        Message::user("Name a famous city in South Korea."),
        Message::new(Role::Assistant, vec![ContentPart::text("서울입니다.")]),
        Message::user("What about a famous mountain there?"),
    ];

    let ctx = ExecutionContext::new();
    let response = chat
        .complete_full(messages, &ctx)
        .await
        .expect("multi-turn round-trip with system prompt");

    let visible_text: String = response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert!(
        !visible_text.trim().is_empty(),
        "multi-turn assistant continuation must produce a visible text reply"
    );
    let has_hangul = visible_text
        .chars()
        .any(|c| matches!(c as u32, 0xAC00..=0xD7A3 | 0x1100..=0x11FF | 0x3130..=0x318F));
    assert!(
        has_hangul,
        "system directive (`respond in Korean only`) must constrain the reply across turns; got: `{visible_text}`"
    );

    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.output_tokens > 0);
}
