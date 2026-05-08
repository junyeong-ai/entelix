//! Live-API smoke for [`OpenAiChatCodec`] + [`DirectTransport`].
//!
//! `#[ignore]`-gated. Run with:
//!
//! ```text
//! OPENAI_API_KEY=sk-... \
//!     cargo test -p entelix-core --test live_openai_chat -- --ignored
//! ```
//!
//! Optional override: `ENTELIX_LIVE_OPENAI_CHAT_MODEL` (default
//! `gpt-4o-mini` — the cheapest tier with general availability).
//!
//! ## Cost discipline
//!
//! `max_tokens = 16` against `gpt-4o-mini`. Per-run cost well
//! under $0.001.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_core::auth::BearerProvider;
use entelix_core::codecs::{Codec, OpenAiChatCodec};
use entelix_core::context::ExecutionContext;
use entelix_core::ir::{ContentPart, Message, ModelRequest, StopReason};
use entelix_core::transports::{DirectTransport, Transport};
use secrecy::SecretString;

const DEFAULT_MODEL: &str = "gpt-4o-mini";

#[tokio::test]
#[ignore = "live-API: requires OPENAI_API_KEY"]
async fn openai_chat_minimal_round_trip() {
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY to run this live-API smoke");
    let model = std::env::var("ENTELIX_LIVE_OPENAI_CHAT_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let codec = OpenAiChatCodec::new();
    let credentials = Arc::new(BearerProvider::new(SecretString::from(api_key)));
    let transport = DirectTransport::openai(credentials).unwrap();

    let request = ModelRequest {
        model,
        messages: vec![Message::user("Reply with the word 'ok' and nothing else.")],
        max_tokens: Some(16),
        temperature: Some(0.0),
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&request).expect("openai chat encode");
    let response = transport
        .send(encoded, &ExecutionContext::new())
        .await
        .expect("openai chat send");

    assert!(
        (200..300).contains(&response.status),
        "expected 2xx, got {} — body: {}",
        response.status,
        String::from_utf8_lossy(&response.body)
    );

    let decoded = codec
        .decode(&response.body, Vec::new())
        .expect("openai chat decode");
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
    assert!(decoded.usage.input_tokens > 0);
    assert!(decoded.usage.output_tokens > 0);
}
