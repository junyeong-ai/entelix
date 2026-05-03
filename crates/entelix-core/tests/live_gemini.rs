//! Live-API smoke for [`GeminiCodec`] + [`DirectTransport`].
//!
//! `#[ignore]`-gated. Run with:
//!
//! ```text
//! GEMINI_API_KEY=AIza... \
//!     cargo test -p entelix-core --test live_gemini -- --ignored
//! ```
//!
//! Optional override: `ENTELIX_LIVE_GEMINI_MODEL` (default
//! `gemini-2.0-flash` — the cheapest tier with general
//! availability).
//!
//! Gemini exposes API-key auth via the `x-goog-api-key` header
//! (the alternative is the `?key=…` query string; we pick the
//! header form so the key never lands in URLs / logs).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_core::auth::ApiKeyProvider;
use entelix_core::codecs::{Codec, GeminiCodec};
use entelix_core::context::ExecutionContext;
use entelix_core::ir::{ContentPart, Message, ModelRequest, StopReason};
use entelix_core::transports::{DirectTransport, Transport};
use secrecy::SecretString;

const DEFAULT_MODEL: &str = "gemini-2.0-flash";

#[tokio::test]
#[ignore = "live-API: requires GEMINI_API_KEY"]
async fn gemini_minimal_round_trip() {
    let api_key =
        std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY to run this live-API smoke");
    let model =
        std::env::var("ENTELIX_LIVE_GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let codec = GeminiCodec::new();
    let credentials = Arc::new(
        ApiKeyProvider::new("x-goog-api-key", SecretString::from(api_key))
            .expect("x-goog-api-key is a valid header name"),
    );
    let transport = DirectTransport::gemini(credentials).unwrap();

    let request = ModelRequest {
        model,
        messages: vec![Message::user("Reply with the word 'ok' and nothing else.")],
        max_tokens: Some(16),
        temperature: Some(0.0),
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&request).expect("gemini encode");
    let response = transport
        .send(encoded, &ExecutionContext::new())
        .await
        .expect("gemini send");

    assert!(
        (200..300).contains(&response.status),
        "expected 2xx, got {} — body: {}",
        response.status,
        String::from_utf8_lossy(&response.body)
    );

    let decoded = codec
        .decode(&response.body, Vec::new())
        .expect("gemini decode");
    assert!(
        decoded
            .content
            .iter()
            .any(|p| matches!(p, ContentPart::Text { text, .. } if !text.is_empty())),
        "expected at least one non-empty text block"
    );
    assert!(
        matches!(
            decoded.stop_reason,
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence { .. }
        ),
        "unexpected stop_reason: {:?}",
        decoded.stop_reason
    );
}
