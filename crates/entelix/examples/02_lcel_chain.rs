//! `02_lcel_chain` — `prompt.pipe(model).pipe(parser)` end-to-end demo.
//!
//! Build: `cargo build --example 02_lcel_chain -p entelix`
//! Run:   `ANTHROPIC_API_KEY=sk-... cargo run --example 02_lcel_chain -p entelix`
//!
//! Demonstrates the canonical LCEL composition: a chat prompt feeds
//! the model, the model's reply is parsed as JSON into a typed Rust
//! struct.
//!
//! CI builds but does not execute — see `HANDOFF.md §9`.

#![allow(clippy::print_stdout)] // example output goes to the terminal

use std::sync::Arc;

use entelix::auth::ApiKeyProvider;
use entelix::codecs::AnthropicMessagesCodec;
use entelix::transports::DirectTransport;
use entelix::{
    ChatModel, ChatPromptPart, ChatPromptTemplate, Error, ExecutionContext, JsonOutputParser,
    PromptValue, PromptVars, Result, Runnable, RunnableExt,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Greeting {
    message: String,
    language: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| Error::config("ANTHROPIC_API_KEY env var not set"))?;

    let credentials = Arc::new(ApiKeyProvider::anthropic(api_key));
    let transport = DirectTransport::anthropic(credentials)?;
    let model = ChatModel::new(AnthropicMessagesCodec::new(), transport, "claude-opus-4-7")
        .with_max_tokens(200);

    let prompt = ChatPromptTemplate::from_messages(vec![
        ChatPromptPart::system(
            "Reply with ONLY a JSON object of the form \
             {{\"message\": \"...\", \"language\": \"...\"}}. No prose.",
        )?,
        ChatPromptPart::user("Greet someone warmly in {{ language }}.")?,
    ]);

    let chain = prompt.pipe(model).pipe(JsonOutputParser::<Greeting>::new());

    let mut vars = PromptVars::new();
    vars.insert("language".to_owned(), PromptValue::from("Korean"));

    let greeting: Greeting = chain.invoke(vars, &ExecutionContext::new()).await?;
    println!("language: {}", greeting.language);
    println!("message:  {}", greeting.message);
    Ok(())
}
