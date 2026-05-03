//! `AnthropicMessagesCodec` streaming — `encode_streaming` shape +
//! `decode_stream` SSE event parser. Synthetic byte stream feeds the
//! parser directly (no transport) so the test stays deterministic.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use bytes::Bytes;
use entelix_core::codecs::{AnthropicMessagesCodec, BoxByteStream, Codec};
use entelix_core::ir::{Message, ModelRequest, ModelWarning, StopReason};
use entelix_core::stream::{StreamAggregator, StreamDelta};
use futures::StreamExt;
use serde_json::Value;

fn sse_frames(events: &[(&str, Value)]) -> Bytes {
    let mut out = String::new();
    for (event, payload) in events {
        out.push_str("event: ");
        out.push_str(event);
        out.push('\n');
        out.push_str("data: ");
        out.push_str(&payload.to_string());
        out.push_str("\n\n");
    }
    Bytes::from(out)
}

fn body_from_bytes(b: Bytes) -> BoxByteStream<'static> {
    Box::pin(futures::stream::iter(vec![Ok::<_, entelix_core::Error>(b)]))
}

#[test]
fn encode_streaming_marks_request_and_sets_accept_header() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode_streaming(&req).unwrap();
    assert!(encoded.streaming);
    assert_eq!(
        encoded.headers.get(http::header::ACCEPT).unwrap(),
        "text/event-stream"
    );
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["stream"], Value::Bool(true));
}

#[tokio::test]
async fn decode_stream_full_text_run_round_trips() {
    let bytes = sse_frames(&[
        (
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": "msg_S1",
                    "model": "claude-opus-4-7",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": null,
                    "usage": { "input_tokens": 7 }
                }
            }),
        ),
        (
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            }),
        ),
        (
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Hello, " }
            }),
        ),
        (
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "world!" }
            }),
        ),
        (
            "content_block_stop",
            serde_json::json!({ "type": "content_block_stop", "index": 0 }),
        ),
        (
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 9 }
            }),
        ),
        (
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        ),
    ]);

    let codec = AnthropicMessagesCodec::new();
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut aggregator = StreamAggregator::new();
    while let Some(item) = stream.next().await {
        aggregator.push(item.unwrap()).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.id, "msg_S1");
    assert_eq!(response.model, "claude-opus-4-7");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.input_tokens, 7);
    assert_eq!(response.usage.output_tokens, 9);
    let part = response.content.first().unwrap();
    if let entelix_core::ir::ContentPart::Text { text, .. } = part {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("expected concatenated text part, got {part:?}");
    }
}

#[tokio::test]
async fn decode_stream_tool_use_round_trips() {
    let bytes = sse_frames(&[
        (
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": "msg_T1",
                    "model": "claude-opus-4-7",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": null,
                    "usage": { "input_tokens": 3 }
                }
            }),
        ),
        (
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "double",
                    "input": {}
                }
            }),
        ),
        (
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "{\"n\":" }
            }),
        ),
        (
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "21}" }
            }),
        ),
        (
            "content_block_stop",
            serde_json::json!({ "type": "content_block_stop", "index": 0 }),
        ),
        (
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use" },
                "usage": { "output_tokens": 4 }
            }),
        ),
        (
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        ),
    ]);

    let codec = AnthropicMessagesCodec::new();
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut aggregator = StreamAggregator::new();
    while let Some(item) = stream.next().await {
        aggregator.push(item.unwrap()).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let part = response.content.first().unwrap();
    if let entelix_core::ir::ContentPart::ToolUse { id, name, input } = part {
        assert_eq!(id, "call_1");
        assert_eq!(name, "double");
        assert_eq!(input["n"], 21);
    } else {
        panic!("expected tool_use, got {part:?}");
    }
}

#[tokio::test]
async fn decode_stream_split_across_chunks() {
    // Frame split mid-data line — parser must accumulate.
    let bytes = sse_frames(&[
        (
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": { "id": "msg_X", "model": "claude-opus-4-7", "usage": {"input_tokens": 1} }
            }),
        ),
        (
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        ),
    ]);
    let mid = bytes.len() / 2;
    let head = bytes.slice(..mid);
    let tail = bytes.slice(mid..);
    let stream: BoxByteStream<'static> = Box::pin(futures::stream::iter(vec![
        Ok::<_, entelix_core::Error>(head),
        Ok::<_, entelix_core::Error>(tail),
    ]));

    let codec = AnthropicMessagesCodec::new();
    let mut deltas = codec.decode_stream(stream, Vec::new());
    let mut count = 0;
    while let Some(item) = deltas.next().await {
        item.unwrap();
        count += 1;
    }
    assert!(count >= 2, "expected at least Start + Stop, got {count}");
}

#[tokio::test]
async fn decode_stream_emits_encode_warnings_first() {
    let bytes = sse_frames(&[(
        "message_start",
        serde_json::json!({
            "type": "message_start",
            "message": { "id": "msg_W", "model": "m", "usage": {"input_tokens": 0} }
        }),
    )]);

    let codec = AnthropicMessagesCodec::new();
    let warnings_in = vec![ModelWarning::LossyEncode {
        field: "test".into(),
        detail: "carry-forward".into(),
    }];
    let mut stream = codec.decode_stream(body_from_bytes(bytes), warnings_in);
    let first = stream.next().await.unwrap().unwrap();
    assert!(matches!(first, StreamDelta::Warning(_)));
}

#[tokio::test]
async fn decode_stream_surfaces_error_event_as_provider_error() {
    let bytes = sse_frames(&[(
        "error",
        serde_json::json!({
            "type": "error",
            "error": { "type": "overloaded_error", "message": "slow down" }
        }),
    )]);
    let codec = AnthropicMessagesCodec::new();
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let first = stream.next().await.unwrap();
    assert!(matches!(first, Err(entelix_core::Error::Provider { .. })));
}

#[tokio::test]
async fn decode_stream_emits_warning_when_content_block_start_index_is_missing() {
    // Invariant #15 — vendor SSE event missing the spec-mandated
    // `index` field used to silently fall back to slot 0 with no
    // operator-visible signal. Post-fix, the parser yields a
    // `LossyEncode` warning before falling back so the operator
    // sees the protocol violation in the response warnings list.
    let bytes = sse_frames(&[
        (
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": "msg_S1",
                    "model": "claude-opus-4-7",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": null,
                    "usage": { "input_tokens": 1 }
                }
            }),
        ),
        (
            "content_block_start",
            serde_json::json!({
                // index intentionally omitted — vendor protocol violation
                "type": "content_block_start",
                "content_block": { "type": "text", "text": "hi" }
            }),
        ),
        (
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" }
            }),
        ),
    ]);
    let codec = AnthropicMessagesCodec::new();
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut saw_index_warning = false;
    while let Some(item) = stream.next().await {
        if let Ok(StreamDelta::Warning(ModelWarning::LossyEncode { field, .. })) = item
            && field == "stream.content_block_start.index"
        {
            saw_index_warning = true;
        }
    }
    assert!(
        saw_index_warning,
        "missing 'index' on content_block_start must surface as LossyEncode warning"
    );
}
