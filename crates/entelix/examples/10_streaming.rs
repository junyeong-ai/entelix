//! `10_streaming` — token-level `StreamMode::Messages` from a fake SSE
//! source.
//!
//! Build: `cargo build --example 10_streaming -p entelix`
//! Run:   `cargo run   --example 10_streaming -p entelix`
//!
//! Drives the streaming pipeline (`Codec::decode_stream`) without
//! an API key: a synthetic Anthropic SSE byte stream feeds the
//! codec's parser, which surfaces `StreamDelta`s; the
//! `StreamAggregator` finalizes them into a `ModelResponse`.
//! Deterministic, CI-friendly.

#![allow(clippy::print_stdout)]

use bytes::Bytes;
use entelix::Result;
use entelix::codecs::{AnthropicMessagesCodec, BoxByteStream, Codec};
use entelix::stream::{StreamAggregator, StreamDelta};
use futures::StreamExt;

const SSE_BODY: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_demo\",\"model\":\"claude-opus-4-7\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\", \"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world!\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

fn body_stream() -> BoxByteStream<'static> {
    Box::pin(futures::stream::iter(vec![Ok::<_, entelix::Error>(
        Bytes::from(SSE_BODY),
    )]))
}

#[tokio::main]
async fn main() -> Result<()> {
    let codec = AnthropicMessagesCodec::new();
    let mut stream = codec.decode_stream(body_stream(), Vec::new());

    let mut aggregator = StreamAggregator::new();
    println!("── streaming deltas ─────────────────────────────");
    while let Some(item) = stream.next().await {
        let delta = item?;
        match &delta {
            StreamDelta::Start {
                id,
                model,
                provider_echoes,
            } => {
                println!(
                    "  start  id={id} model={model}{}",
                    if provider_echoes.is_empty() {
                        ""
                    } else {
                        " [echo]"
                    }
                );
            }
            StreamDelta::TextDelta {
                text,
                provider_echoes,
            } => {
                print!("  text   ┊ {text}");
                if !provider_echoes.is_empty() {
                    print!(" [echo]");
                }
            }
            StreamDelta::ThinkingDelta {
                text,
                provider_echoes,
            } => {
                print!("  think  ┊ {text}");
                if !provider_echoes.is_empty() {
                    println!(" [echo]");
                }
            }
            StreamDelta::ToolUseStart {
                id,
                name,
                provider_echoes,
            } => {
                println!(
                    "  tool>  id={id} name={name}{}",
                    if provider_echoes.is_empty() {
                        ""
                    } else {
                        " [echo]"
                    }
                );
            }
            StreamDelta::ToolUseInputDelta { partial_json } => {
                println!("  tool…  {partial_json}");
            }
            StreamDelta::ToolUseStop => println!("  tool<"),
            StreamDelta::Usage(u) => {
                println!("\n  usage  in={} out={}", u.input_tokens, u.output_tokens);
            }
            StreamDelta::Warning(w) => println!("  warn   {w:?}"),
            StreamDelta::RateLimit(rl) => println!(
                "  quota  req_remain={:?} tok_remain={:?}",
                rl.requests_remaining, rl.tokens_remaining
            ),
            StreamDelta::Stop { stop_reason } => println!("  stop   {stop_reason:?}"),
            _ => {}
        }
        aggregator.push(delta)?;
    }

    let response = aggregator.finalize()?;
    println!("\n── finalized ────────────────────────────────────");
    println!("  id          : {}", response.id);
    println!("  model       : {}", response.model);
    println!("  stop_reason : {:?}", response.stop_reason);
    println!("  parts       : {}", response.content.len());
    println!("  output_tokens: {}", response.usage.output_tokens);
    Ok(())
}
