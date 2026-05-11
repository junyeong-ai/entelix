//! Sparse codec×transport compatibility matrix — type-level
//! verification that the documented `~10` valid pairings
//! all instantiate `ChatModel<C, T>` cleanly. Pairings the matrix
//! marks invalid (e.g. `OpenAiChat` × `BedrockTransport`) are
//! intentionally absent — vendors don't ship those models on those
//! routes.
//!
//! Compatible cells (test helper exists for each):
//!
//! | Codec ↓ \ Transport →      | Direct | Bedrock | Vertex | Foundry |
//! |----------------------------|--------|---------|--------|---------|
//! | `AnthropicMessagesCodec`   | ✓      | ✓       | ·      | ✓       |
//! | `VertexAnthropicCodec`     | ·      | ·       | ✓      | ·       |
//! | `OpenAiChatCodec`          | ✓      | ·       | ·      | ✓       |
//! | `OpenAiResponsesCodec`     | ✓      | ·       | ·      | ·       |
//! | `GeminiCodec`              | ✓      | ·       | ✓      | ·       |
//! | `BedrockConverseCodec`     | ·      | ✓       | ·      | ·       |
//!
//! Vertex AI hosts Anthropic Claude on a `:rawPredict` /
//! `:streamRawPredict` endpoint with a vendor-specific
//! `anthropic_version: "vertex-2023-10-16"` body field — the direct
//! `AnthropicMessagesCodec` cannot drive that route, so the
//! dedicated `VertexAnthropicCodec` carries the wire-shape rewrite.

#![cfg(all(feature = "aws", feature = "gcp", feature = "azure"))]
#![allow(clippy::unwrap_used, dead_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use entelix_cloud::CloudError;
use entelix_cloud::bedrock::{BedrockAuth, BedrockTransport};
use entelix_cloud::foundry::{FoundryAuth, FoundryTransport};
use entelix_cloud::refresh::{TokenRefresher, TokenSnapshot};
use entelix_cloud::vertex::VertexTransport;
use entelix_core::ChatModel;
use entelix_core::auth::ApiKeyProvider;
use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec, VertexAnthropicCodec,
};
use entelix_core::transports::DirectTransport;
use secrecy::SecretString;

struct StaticTokenRefresher(String);

#[async_trait]
impl TokenRefresher<SecretString> for StaticTokenRefresher {
    async fn refresh(&self) -> Result<TokenSnapshot<SecretString>, CloudError> {
        Ok(TokenSnapshot {
            value: SecretString::from(self.0.clone()),
            expires_at: Instant::now() + Duration::from_mins(60),
        })
    }
}

fn direct() -> DirectTransport {
    DirectTransport::new(
        "https://api.example.com",
        Arc::new(ApiKeyProvider::anthropic("sk-test")),
    )
    .unwrap()
}

fn bedrock() -> BedrockTransport {
    BedrockTransport::builder()
        .with_region("us-east-1")
        .with_auth(BedrockAuth::Bearer {
            token: SecretString::from("test".to_owned()),
        })
        .build()
        .unwrap()
}

fn vertex() -> VertexTransport {
    VertexTransport::builder()
        .with_project_id("p")
        .with_location("global")
        .with_token_refresher(Arc::new(StaticTokenRefresher("test".to_owned())))
        .build()
        .unwrap()
}

fn foundry() -> FoundryTransport {
    FoundryTransport::builder()
        .with_base_url("https://example.openai.azure.com")
        .with_auth(FoundryAuth::ApiKey {
            token: SecretString::from("test".to_owned()),
        })
        .build()
        .unwrap()
}

#[test]
fn anthropic_codec_pairs_with_direct_bedrock_and_foundry() {
    // Vertex is intentionally absent — Vertex AI Anthropic uses a
    // `:rawPredict` endpoint and a `vertex-2023-10-16` body marker
    // that `AnthropicMessagesCodec` does not produce; the dedicated
    // `VertexAnthropicCodec` covers that pairing.
    let _: ChatModel<_, _> =
        ChatModel::new(AnthropicMessagesCodec::new(), direct(), "claude-opus-4-7");
    let _: ChatModel<_, _> =
        ChatModel::new(AnthropicMessagesCodec::new(), bedrock(), "claude-opus-4-7");
    let _: ChatModel<_, _> =
        ChatModel::new(AnthropicMessagesCodec::new(), foundry(), "claude-opus-4-7");
}

#[test]
fn vertex_anthropic_codec_pairs_with_vertex() {
    let _: ChatModel<_, _> =
        ChatModel::new(VertexAnthropicCodec::new(), vertex(), "claude-opus-4-7");
}

#[test]
fn openai_chat_pairs_with_direct_and_foundry() {
    let _: ChatModel<_, _> = ChatModel::new(OpenAiChatCodec::new(), direct(), "gpt-4.1");
    let _: ChatModel<_, _> = ChatModel::new(OpenAiChatCodec::new(), foundry(), "gpt-4.1");
}

#[test]
fn openai_responses_pairs_with_direct() {
    let _: ChatModel<_, _> = ChatModel::new(OpenAiResponsesCodec::new(), direct(), "gpt-4.1");
}

#[test]
fn gemini_pairs_with_direct_and_vertex() {
    let _: ChatModel<_, _> = ChatModel::new(GeminiCodec::new(), direct(), "gemini-2.0-flash");
    let _: ChatModel<_, _> = ChatModel::new(GeminiCodec::new(), vertex(), "gemini-2.0-flash");
}

#[test]
fn bedrock_converse_pairs_with_bedrock() {
    let _: ChatModel<_, _> = ChatModel::new(
        BedrockConverseCodec::new(),
        bedrock(),
        "anthropic.claude-opus-4-7-v1:0",
    );
}
