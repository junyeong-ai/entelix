//! `01_quickstart` — single Anthropic Messages call.
//!
//! Build: `cargo build --example 01_quickstart -p entelix`
//! Run:   `ANTHROPIC_API_KEY=sk-... cargo run --example 01_quickstart -p entelix`
//!
//! This binary makes a real API call when run with credentials, so CI
//! validates only that it compiles. See `HANDOFF.md §9` for the live-test
//! cost trigger.

#![allow(clippy::print_stdout)] // example output goes to the terminal

use std::sync::Arc;

use entelix::auth::ApiKeyProvider;
use entelix::codecs::AnthropicMessagesCodec;
use entelix::ir::{ContentPart, Message};
use entelix::transports::DirectTransport;
use entelix::{ChatModel, Error, ExecutionContext, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| Error::config("ANTHROPIC_API_KEY env var not set"))?;

    let credentials = Arc::new(ApiKeyProvider::anthropic(api_key));
    let transport = DirectTransport::anthropic(credentials)?;
    let model = ChatModel::new(AnthropicMessagesCodec::new(), transport, "claude-opus-4-7")
        .with_max_tokens(256)
        .with_system("Answer in one sentence.");

    let reply: Message = model
        .complete(
            vec![Message::user("Define entelechy in plain English.")],
            &ExecutionContext::new(),
        )
        .await?;

    for part in &reply.content {
        if let ContentPart::Text { text, .. } = part {
            println!("{text}");
        }
    }
    Ok(())
}
