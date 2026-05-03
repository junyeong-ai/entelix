//! `09_multi_codec` — same `Vec<Message>` round-tripped through
//! every shipped codec.
//!
//! Build: `cargo build --example 09_multi_codec -p entelix`
//! Run:   `cargo run   --example 09_multi_codec -p entelix`
//!
//! Demonstrates IR neutrality (invariant 4): the same provider-neutral
//! `ModelRequest` is encoded by Anthropic / `OpenAI` Chat / `OpenAI`
//! Responses / Gemini / Bedrock Converse codecs without the caller
//! knowing the wire shape. Each codec produces a different vendor JSON
//! layout from the *same* IR. Deterministic — no live API calls.

#![allow(clippy::print_stdout)]

use entelix::Result;
use entelix::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix::ir::{Message, ModelRequest};

fn show(name: &str, body: &[u8]) -> Result<()> {
    let value: serde_json::Value = serde_json::from_slice(body).map_err(entelix::Error::from)?;
    let pretty = serde_json::to_string_pretty(&value).map_err(entelix::Error::from)?;
    println!("\n── {name} ──────────────────────────────────────────");
    println!("{pretty}");
    Ok(())
}

fn main() -> Result<()> {
    let request = ModelRequest {
        model: "demo-model".into(),
        messages: vec![Message::user("Translate `hello world` to French.")],
        system: "Reply with the translation only.".into(),
        max_tokens: Some(64),
        ..ModelRequest::default()
    };

    let anthropic = AnthropicMessagesCodec::new().encode(&request)?;
    show("anthropic-messages → /v1/messages", &anthropic.body)?;

    let openai_chat = OpenAiChatCodec::new().encode(&request)?;
    show("openai-chat → /v1/chat/completions", &openai_chat.body)?;

    let openai_responses = OpenAiResponsesCodec::new().encode(&request)?;
    show("openai-responses → /v1/responses", &openai_responses.body)?;

    let gemini_request = ModelRequest {
        model: "gemini-2.0-flash".into(),
        ..request.clone()
    };
    let gemini = GeminiCodec::new().encode(&gemini_request)?;
    show("gemini → :generateContent", &gemini.body)?;

    let bedrock = BedrockConverseCodec::new().encode(&request)?;
    show("bedrock-converse → /converse", &bedrock.body)?;

    println!("\n   ✓ same IR, five vendor JSON shapes — invariant 4 holding.");
    Ok(())
}
