//! `12_compat_matrix` — sparse codec×transport pairing matrix
//! (ADR-0018).
//!
//! Build: `cargo build --example 12_compat_matrix -p entelix --features=full`
//! Run:   `cargo run   --example 12_compat_matrix -p entelix --features=full`
//!
//! Of the naïve 5×4 = 20 combinations of `entelix-core` codecs and
//! `entelix-cloud` transports, only ~10 are real-world valid: vendors
//! don't ship `OpenAI Chat` over Bedrock, etc. The compat matrix below
//! is the canonical answer of which `ChatModel<C, T>` instances make
//! sense to construct. The verification that each cell *type-checks*
//! lives in `crates/entelix-cloud/tests/compat_matrix.rs`; this
//! example walks the matrix at runtime so a reader can inspect the
//! shape of each pairing and the wire bytes the codec produces.
//!
//! Deterministic: no live cloud calls are made. We construct each
//! valid `ChatModel`, encode an IR `ModelRequest`, and print the
//! resulting bytes' size + first few characters.
//!
//! | Codec ↓ \ Transport →  | Direct | Bedrock | Vertex | Foundry |
//! |------------------------|:------:|:-------:|:------:|:-------:|
//! | AnthropicMessagesCodec |   ✓    |    ✓    |   ✓    |    ✓    |
//! | OpenAiChatCodec        |   ✓    |         |        |    ✓    |
//! | OpenAiResponsesCodec   |   ✓    |         |        |         |
//! | GeminiCodec            |   ✓    |         |   ✓    |         |
//! | BedrockConverseCodec   |        |    ✓    |        |         |

#![cfg(all(feature = "aws", feature = "gcp", feature = "azure"))]
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::map_unwrap_or,
    clippy::unnecessary_wraps,
    clippy::used_underscore_binding,
    clippy::indexing_slicing
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use secrecy::SecretString;

use entelix::auth::ApiKeyProvider;
use entelix::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix::ir::{Message, ModelRequest};
use entelix::transports::DirectTransport;
use entelix::{
    BedrockAuth, BedrockTransport, ChatModel, CloudError, FoundryAuth, FoundryTransport, Result,
    TokenRefresher, TokenSnapshot, VertexTransport,
};

struct StaticTokenRefresher(&'static str);

#[async_trait]
impl TokenRefresher<SecretString> for StaticTokenRefresher {
    async fn refresh(&self) -> std::result::Result<TokenSnapshot<SecretString>, CloudError> {
        Ok(TokenSnapshot {
            value: SecretString::from(self.0.to_owned()),
            expires_at: Instant::now() + Duration::from_secs(3600),
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
        .with_token_refresher(Arc::new(StaticTokenRefresher("test")))
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

fn print_pair<C, T>(label: &str, codec: C, _transport: T, model: &str, request: &ModelRequest)
where
    C: Codec + 'static,
    T: entelix_core::transports::Transport + 'static,
{
    let _: ChatModel<C, T> = ChatModel::new(codec, _transport, model.to_owned());
    // Re-encode separately just for the wire-bytes preview — the
    // ChatModel above proves the *type* is constructible.
    let codec_again = encoder_for::<C>();
    let encoded = codec_again
        .encode(request)
        .expect("encode should not fail for the demo IR");
    let snippet = std::str::from_utf8(&encoded.body)
        .map(|s| s.chars().take(60).collect::<String>())
        .unwrap_or_else(|_| "<binary>".to_owned());
    println!(
        "  ✓ {label:<48} model={model:<26} body={} bytes  → {snippet}…",
        encoded.body.len()
    );
}

/// Tiny helper so the call site doesn't have to repeat per-codec
/// `Default::default()` invocations.
fn encoder_for<C: Codec + 'static>() -> Box<dyn Codec> {
    let name = std::any::type_name::<C>();
    if name.contains("AnthropicMessagesCodec") {
        Box::new(AnthropicMessagesCodec::new())
    } else if name.contains("OpenAiChatCodec") {
        Box::new(OpenAiChatCodec::new())
    } else if name.contains("OpenAiResponsesCodec") {
        Box::new(OpenAiResponsesCodec::new())
    } else if name.contains("GeminiCodec") {
        Box::new(GeminiCodec::new())
    } else if name.contains("BedrockConverseCodec") {
        Box::new(BedrockConverseCodec::new())
    } else {
        panic!("unknown codec: {{ name }}")
    }
}

fn main() -> Result<()> {
    let request = ModelRequest {
        model: "demo-model".into(),
        messages: vec![Message::user("Translate `hello world` to French.")],
        system: "Reply with the translation only.".into(),
        max_tokens: Some(64),
        ..ModelRequest::default()
    };

    println!("── Anthropic codec — pairs with all four transports ─────");
    print_pair(
        "AnthropicMessagesCodec × DirectTransport",
        AnthropicMessagesCodec::new(),
        direct(),
        "claude-opus-4-7",
        &request,
    );
    print_pair(
        "AnthropicMessagesCodec × BedrockTransport",
        AnthropicMessagesCodec::new(),
        bedrock(),
        "claude-opus-4-7",
        &request,
    );
    print_pair(
        "AnthropicMessagesCodec × VertexTransport",
        AnthropicMessagesCodec::new(),
        vertex(),
        "claude-opus-4-7",
        &request,
    );
    print_pair(
        "AnthropicMessagesCodec × FoundryTransport",
        AnthropicMessagesCodec::new(),
        foundry(),
        "claude-opus-4-7",
        &request,
    );

    println!("\n── OpenAI Chat — Direct + Foundry only ──────────────────");
    print_pair(
        "OpenAiChatCodec × DirectTransport",
        OpenAiChatCodec::new(),
        direct(),
        "gpt-4.1",
        &request,
    );
    print_pair(
        "OpenAiChatCodec × FoundryTransport",
        OpenAiChatCodec::new(),
        foundry(),
        "gpt-4.1",
        &request,
    );

    println!("\n── OpenAI Responses — Direct only ───────────────────────");
    print_pair(
        "OpenAiResponsesCodec × DirectTransport",
        OpenAiResponsesCodec::new(),
        direct(),
        "gpt-4.1",
        &request,
    );

    println!("\n── Gemini — Direct + Vertex only ────────────────────────");
    let gemini_req = ModelRequest {
        model: "gemini-2.0-flash".into(),
        ..request.clone()
    };
    print_pair(
        "GeminiCodec × DirectTransport",
        GeminiCodec::new(),
        direct(),
        "gemini-2.0-flash",
        &gemini_req,
    );
    print_pair(
        "GeminiCodec × VertexTransport",
        GeminiCodec::new(),
        vertex(),
        "gemini-2.0-flash",
        &gemini_req,
    );

    println!("\n── Bedrock Converse — Bedrock only ──────────────────────");
    print_pair(
        "BedrockConverseCodec × BedrockTransport",
        BedrockConverseCodec::new(),
        bedrock(),
        "anthropic.claude-opus-4-7-v1:0",
        &request,
    );

    println!("\n   10 valid (codec, transport) pairs — see ADR-0018.");
    Ok(())
}
