//! Streaming-thinking coverage. Each codec receives a synthetic
//! SSE / chunked-JSON corpus that contains a thinking block; the
//! test drains the resulting `StreamDelta` sequence into a
//! `StreamAggregator` and asserts that the finalized
//! `ModelResponse` carries `ContentPart::Thinking` with the right
//! text + signature.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unnecessary_find_map
)]

use bytes::Bytes;
use futures::StreamExt;
use futures::stream;
use serde_json::{Value, json};

use entelix_core::codecs::{AnthropicMessagesCodec, Codec, GeminiCodec, OpenAiResponsesCodec};
use entelix_core::ir::ContentPart;
use entelix_core::stream::{StreamAggregator, StreamDelta};

fn anthropic_sse(events: &[Value]) -> Bytes {
    let mut out = String::new();
    for event in events {
        out.push_str("event: ");
        out.push_str(event.get("type").and_then(Value::as_str).unwrap_or(""));
        out.push('\n');
        out.push_str("data: ");
        out.push_str(&event.to_string());
        out.push_str("\n\n");
    }
    Bytes::from(out)
}

fn gemini_sse(payloads: &[Value]) -> Bytes {
    let mut out = String::new();
    for payload in payloads {
        out.push_str("data: ");
        out.push_str(&payload.to_string());
        out.push_str("\n\n");
    }
    Bytes::from(out)
}

fn openai_responses_sse(events: &[Value]) -> Bytes {
    let mut out = String::new();
    for event in events {
        out.push_str("event: ");
        out.push_str(event.get("type").and_then(Value::as_str).unwrap_or(""));
        out.push('\n');
        out.push_str("data: ");
        out.push_str(&event.to_string());
        out.push_str("\n\n");
    }
    Bytes::from(out)
}

async fn drain_to_response(
    codec: &dyn Codec,
    body: Bytes,
) -> Result<entelix_core::ir::ModelResponse, entelix_core::Error> {
    let bytes_stream = stream::once(async move { Ok::<_, entelix_core::Error>(body) }).boxed();
    let mut delta_stream = codec.decode_stream(bytes_stream, vec![]);
    let mut agg = StreamAggregator::new();
    while let Some(delta) = delta_stream.next().await {
        agg.push(delta?)?;
    }
    agg.finalize()
}

#[tokio::test]
async fn anthropic_stream_yields_thinking_with_signature() {
    let codec = AnthropicMessagesCodec::new();
    let body = anthropic_sse(&[
        json!({
            "type": "message_start",
            "message": {"id": "msg_1", "model": "claude-opus-4-7", "usage": {"input_tokens": 5, "output_tokens": 0}},
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me reason. "},
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Two plus two."},
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-001"},
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }),
        json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Four."},
        }),
        json!({"type": "content_block_stop", "index": 1}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 10},
        }),
        json!({"type": "message_stop"}),
    ]);
    let response = drain_to_response(&codec, body).await.unwrap();
    assert_eq!(response.content.len(), 2);
    match &response.content[0] {
        ContentPart::Thinking {
            text,
            provider_echoes,
            ..
        } => {
            assert_eq!(text, "Let me reason. Two plus two.");
            let signature = entelix_core::ir::ProviderEchoSnapshot::find_in(
                provider_echoes,
                "anthropic-messages",
            )
            .and_then(|e| e.payload_str("signature"));
            assert_eq!(signature, Some("sig-001"));
        }
        other => panic!("expected Thinking, got {other:?}"),
    }
    match &response.content[1] {
        ContentPart::Text { text, .. } => assert_eq!(text, "Four."),
        other => panic!("expected Text, got {other:?}"),
    }
}

#[tokio::test]
async fn gemini_stream_yields_thinking_when_part_is_thought() {
    let codec = GeminiCodec::new();
    let body = gemini_sse(&[
        json!({
            "modelVersion": "gemini-2.5-pro",
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"thought": true, "text": "Reasoning step 1.", "thoughtSignature": "sig-g1"},
                ]},
            }],
        }),
        json!({
            "modelVersion": "gemini-2.5-pro",
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "Final answer."},
                ]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }),
    ]);
    let response = drain_to_response(&codec, body).await.unwrap();
    assert!(response.content.len() >= 2);
    let thinking = response
        .content
        .iter()
        .find_map(|p| matches!(p, ContentPart::Thinking { .. }).then_some(p))
        .expect("thinking block must round-trip");
    if let ContentPart::Thinking {
        text,
        provider_echoes,
        ..
    } = thinking
    {
        assert!(text.contains("Reasoning step 1"));
        let signature = entelix_core::ir::ProviderEchoSnapshot::find_in(provider_echoes, "gemini")
            .and_then(|e| e.payload_str("thought_signature"));
        assert_eq!(signature, Some("sig-g1"));
    }
    let text = response
        .content
        .iter()
        .find_map(|p| match p {
            ContentPart::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .expect("text block must round-trip");
    assert_eq!(text, "Final answer.");
}

#[tokio::test]
async fn openai_responses_stream_yields_thinking_for_reasoning_delta() {
    let codec = OpenAiResponsesCodec::new();
    let body = openai_responses_sse(&[
        json!({
            "type": "response.created",
            "response": {"id": "resp_1", "model": "gpt-5"},
        }),
        json!({
            "type": "response.reasoning.delta",
            "delta": "First, decompose.",
        }),
        json!({
            "type": "response.reasoning.delta",
            "delta": " Then synthesize.",
        }),
        json!({
            "type": "response.output_text.delta",
            "delta": "Done.",
        }),
        json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        }),
    ]);
    let response = drain_to_response(&codec, body).await.unwrap();
    assert_eq!(response.content.len(), 2);
    match &response.content[0] {
        ContentPart::Thinking { text, .. } => {
            assert_eq!(text, "First, decompose. Then synthesize.");
        }
        other => panic!("expected Thinking first, got {other:?}"),
    }
    match &response.content[1] {
        ContentPart::Text { text, .. } => assert_eq!(text, "Done."),
        other => panic!("expected Text second, got {other:?}"),
    }
}

#[tokio::test]
async fn aggregator_thinking_then_text_preserves_intra_turn_order() {
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: "x".into(),
        model: "m".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    agg.push(StreamDelta::ThinkingDelta {
        text: "thoughts".into(),
        provider_echoes: vec![entelix_core::ir::ProviderEchoSnapshot::for_provider(
            "anthropic-messages",
            "signature",
            "s",
        )],
    })
    .unwrap();
    agg.push(StreamDelta::TextDelta {
        text: "answer".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    agg.push(StreamDelta::Stop {
        stop_reason: entelix_core::ir::StopReason::EndTurn,
    })
    .unwrap();
    let resp = agg.finalize().unwrap();
    assert_eq!(resp.content.len(), 2);
    assert!(matches!(resp.content[0], ContentPart::Thinking { .. }));
    assert!(matches!(resp.content[1], ContentPart::Text { .. }));
}
