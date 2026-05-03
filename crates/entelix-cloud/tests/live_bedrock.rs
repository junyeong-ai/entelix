//! Live-API smoke for `BedrockTransport` + `BedrockConverseCodec`.
//!
//! `#[ignore]`-gated and feature-gated behind `aws` so a default
//! `cargo test --workspace` skips it. Run with:
//!
//! ```text
//! AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \
//!     cargo test -p entelix-cloud --features aws \
//!         --test live_bedrock -- --ignored
//! ```
//!
//! The default credential chain (env vars, profile, IRSA, IMDS) is
//! consulted via `BedrockCredentialProvider::default_chain`. The
//! operator's deployment dictates which credential source flows
//! into the test process — we only require that *some* viable
//! source is wired.
//!
//! ## Cost discipline
//!
//! `max_tokens = 16` against the Haiku tier hosted in Bedrock
//! (`anthropic.claude-3-5-haiku-20241022-v1:0`). Per-run cost well
//! under $0.001.
//!
//! Optional overrides:
//! - `ENTELIX_LIVE_BEDROCK_REGION` — overrides `AWS_REGION` for
//!   the smoke (some AWS accounts run with a global default but
//!   only certain regions host Bedrock model SKUs).
//! - `ENTELIX_LIVE_BEDROCK_MODEL` — overrides the model id when
//!   the operator's account doesn't have access to Haiku.

#![cfg(feature = "aws")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use entelix_cloud::bedrock::{BedrockAuth, BedrockCredentialProvider, BedrockTransport};
use entelix_core::codecs::{BedrockConverseCodec, Codec};
use entelix_core::context::ExecutionContext;
use entelix_core::ir::{ContentPart, Message, ModelRequest, StopReason};
use entelix_core::transports::Transport;

const DEFAULT_MODEL: &str = "anthropic.claude-3-5-haiku-20241022-v1:0";

#[tokio::test]
#[ignore = "live-API: requires AWS credentials + AWS_REGION (or ENTELIX_LIVE_BEDROCK_REGION)"]
async fn bedrock_minimal_converse_round_trip() {
    let region = std::env::var("ENTELIX_LIVE_BEDROCK_REGION")
        .or_else(|_| std::env::var("AWS_REGION"))
        .expect("set AWS_REGION (or ENTELIX_LIVE_BEDROCK_REGION) to run this live-API smoke");
    let model =
        std::env::var("ENTELIX_LIVE_BEDROCK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    // Bedrock's `Converse` API is the cross-model surface here —
    // the codec emits the `/model/{model}/converse` path that
    // `BedrockTransport` signs with SigV4 and routes to the
    // Bedrock Runtime endpoint. Anthropic-via-Bedrock rides the
    // same Converse wire shape as any other model SKU.
    let codec = BedrockConverseCodec::new();
    let credentials = BedrockCredentialProvider::default_chain().await;
    let transport = BedrockTransport::builder()
        .with_region(region)
        .with_auth(BedrockAuth::SigV4 {
            provider: credentials,
        })
        .build()
        .expect("BedrockTransport built from default chain");

    let request = ModelRequest {
        model,
        messages: vec![Message::user("Reply with the word 'ok' and nothing else.")],
        max_tokens: Some(16),
        temperature: Some(0.0),
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&request).expect("bedrock converse encode");
    let response = transport
        .send(encoded, &ExecutionContext::new())
        .await
        .expect("bedrock send");

    assert!(
        (200..300).contains(&response.status),
        "expected 2xx, got {} — body: {}",
        response.status,
        String::from_utf8_lossy(&response.body)
    );

    let decoded = codec
        .decode(&response.body, Vec::new())
        .expect("bedrock converse decode");
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
