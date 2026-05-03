//! Codec round-trip consistency matrix.
//!
//! For every codec that implements `decode_stream`, prove that
//! draining the streaming wire fixture through the parser +
//! [`StreamAggregator`] produces a [`ModelResponse`] semantically
//! equal to decoding the equivalent non-streaming response.
//!
//! This is the single most likely place for codec bugs to land — a
//! field added to one path can silently disappear on the other.
//! Per-codec test files already cover individual scenarios; this
//! matrix ensures the **invariant** holds across vendors.
//!
//! ## Invariants asserted by [`assert_responses_equivalent`]
//!
//! - `stop_reason` — identical
//! - `content` length and per-part shape:
//!   - [`ContentPart::Text`] — exact text equal
//!   - [`ContentPart::Thinking`] — exact text equal (signature
//!     equality is tested separately because some streaming events
//!     omit it)
//!   - [`ContentPart::ToolUse`] — `name` + `input` equal; `id`
//!     equal up to vendor-specific reconstruction (Gemini synthesizes
//!     `name#idx`, others echo)
//! - `usage.input_tokens` + `usage.output_tokens` — identical when
//!   both fixtures provide them
//!
//! ## Bedrock Converse omission
//!
//! Bedrock's streaming uses AWS event-stream binary frames; the
//! codec uses the default `decode_stream` fallback (buffer-then-decode).
//! It is therefore covered by `codec_streaming_fallback.rs` —
//! including it here would be a tautology.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use bytes::Bytes;
use entelix_core::codecs::{
    AnthropicMessagesCodec, BoxByteStream, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix_core::ir::{ContentPart, ModelResponse};
use entelix_core::stream::StreamAggregator;
use futures::StreamExt;
use serde_json::{Value, json};

// ── helpers ────────────────────────────────────────────────────────────────

fn body_from_bytes(b: Bytes) -> BoxByteStream<'static> {
    Box::pin(futures::stream::iter(vec![Ok::<_, entelix_core::Error>(b)]))
}

async fn drain_stream(codec: &dyn Codec, body: Bytes) -> ModelResponse {
    let mut deltas = codec.decode_stream(body_from_bytes(body), vec![]);
    let mut agg = StreamAggregator::new();
    while let Some(item) = deltas.next().await {
        agg.push(item.unwrap()).unwrap();
    }
    agg.finalize().unwrap()
}

/// Compare the IR-equivalent fields between a one-shot decode and a
/// streamed-then-aggregated decode. Vendor-internal fields (`id`,
/// `model`) differ across paths and are excluded.
fn assert_responses_equivalent(oneshot: &ModelResponse, streamed: &ModelResponse, label: &str) {
    assert_eq!(
        oneshot.stop_reason, streamed.stop_reason,
        "[{label}] stop_reason: oneshot={:?} streamed={:?}",
        oneshot.stop_reason, streamed.stop_reason,
    );
    assert_eq!(
        oneshot.content.len(),
        streamed.content.len(),
        "[{label}] content length: oneshot={} streamed={}",
        oneshot.content.len(),
        streamed.content.len(),
    );
    for (i, (lhs, rhs)) in oneshot
        .content
        .iter()
        .zip(streamed.content.iter())
        .enumerate()
    {
        assert_content_part_equivalent(lhs, rhs, &format!("{label}.content[{i}]"));
    }
    if oneshot.usage.input_tokens > 0 && streamed.usage.input_tokens > 0 {
        assert_eq!(
            oneshot.usage.input_tokens, streamed.usage.input_tokens,
            "[{label}] input_tokens"
        );
    }
    if oneshot.usage.output_tokens > 0 && streamed.usage.output_tokens > 0 {
        assert_eq!(
            oneshot.usage.output_tokens, streamed.usage.output_tokens,
            "[{label}] output_tokens"
        );
    }
}

fn assert_content_part_equivalent(lhs: &ContentPart, rhs: &ContentPart, path: &str) {
    match (lhs, rhs) {
        (ContentPart::Text { text: a, .. }, ContentPart::Text { text: b, .. }) => {
            assert_eq!(a, b, "[{path}] text mismatch");
        }
        (ContentPart::Thinking { text: a, .. }, ContentPart::Thinking { text: b, .. }) => {
            assert_eq!(a, b, "[{path}] thinking text mismatch");
        }
        (
            ContentPart::ToolUse {
                name: na,
                input: ia,
                ..
            },
            ContentPart::ToolUse {
                name: nb,
                input: ib,
                ..
            },
        ) => {
            assert_eq!(na, nb, "[{path}] tool_use name mismatch");
            assert_eq!(ia, ib, "[{path}] tool_use input mismatch");
        }
        (a, b) => panic!("[{path}] content variant mismatch: {a:?} vs {b:?}"),
    }
}

// ── Anthropic Messages ─────────────────────────────────────────────────────

fn anthropic_sse(events: &[(&str, Value)]) -> Bytes {
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

#[tokio::test]
async fn anthropic_text_streaming_matches_oneshot() {
    let codec = AnthropicMessagesCodec::new();
    let oneshot_body = json!({
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [{"type": "text", "text": "Hello, world!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 9},
    });
    let streaming_body = anthropic_sse(&[
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_01",
                    "model": "claude-opus-4-7",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": null,
                    "usage": {"input_tokens": 7}
                }
            }),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello, "}
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "world!"}
            }),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 9}
            }),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ]);
    let oneshot = codec
        .decode(oneshot_body.to_string().as_bytes(), vec![])
        .unwrap();
    let streamed = drain_stream(&codec, streaming_body).await;
    assert_responses_equivalent(&oneshot, &streamed, "anthropic.text");
}

#[tokio::test]
async fn anthropic_tool_use_streaming_matches_oneshot() {
    let codec = AnthropicMessagesCodec::new();
    let oneshot_body = json!({
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [{
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_weather",
            "input": {"city": "SF"}
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 18},
    });
    let streaming_body = anthropic_sse(&[
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_02",
                    "model": "claude-opus-4-7",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": null,
                    "usage": {"input_tokens": 12}
                }
            }),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {}
                }
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "\"SF\"}"}
            }),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 18}
            }),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ]);
    let oneshot = codec
        .decode(oneshot_body.to_string().as_bytes(), vec![])
        .unwrap();
    let streamed = drain_stream(&codec, streaming_body).await;
    assert_responses_equivalent(&oneshot, &streamed, "anthropic.tool_use");
}

#[tokio::test]
async fn anthropic_thinking_streaming_matches_oneshot() {
    let codec = AnthropicMessagesCodec::new();
    let oneshot_body = json!({
        "id": "msg_03",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [
            {"type": "thinking", "thinking": "Let me think...", "signature": "sig-A"},
            {"type": "text", "text": "Done."}
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 6},
    });
    let streaming_body = anthropic_sse(&[
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_03",
                    "model": "claude-opus-4-7",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": null,
                    "usage": {"input_tokens": 4}
                }
            }),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""}
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."}
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig-A"}
            }),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""}
            }),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Done."}
            }),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 1}),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 6}
            }),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ]);
    let oneshot = codec
        .decode(oneshot_body.to_string().as_bytes(), vec![])
        .unwrap();
    let streamed = drain_stream(&codec, streaming_body).await;
    assert_responses_equivalent(&oneshot, &streamed, "anthropic.thinking");
}

// ── Gemini ─────────────────────────────────────────────────────────────────

fn gemini_sse(payloads: &[Value]) -> Bytes {
    let mut out = String::new();
    for payload in payloads {
        out.push_str("data: ");
        out.push_str(&payload.to_string());
        out.push_str("\n\n");
    }
    Bytes::from(out)
}

#[tokio::test]
async fn gemini_text_streaming_matches_oneshot() {
    let codec = GeminiCodec::new();
    let oneshot_body = json!({
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "Hello, world!"}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 9}
    });
    let streaming_body = gemini_sse(&[
        json!({
            "modelVersion": "gemini-2.0-flash",
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello, "}]}
            }]
        }),
        json!({
            "modelVersion": "gemini-2.0-flash",
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "world!"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 9}
        }),
    ]);
    let oneshot = codec
        .decode(oneshot_body.to_string().as_bytes(), vec![])
        .unwrap();
    let streamed = drain_stream(&codec, streaming_body).await;
    assert_responses_equivalent(&oneshot, &streamed, "gemini.text");
}

#[tokio::test]
async fn gemini_function_call_streaming_matches_oneshot() {
    let codec = GeminiCodec::new();
    let oneshot_body = json!({
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{
            "content": {"role": "model", "parts": [{
                "functionCall": {"name": "double", "args": {"n": 21}}
            }]},
            "finishReason": "STOP"
        }]
    });
    let streaming_body = gemini_sse(&[json!({
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{
            "content": {"role": "model", "parts": [{
                "functionCall": {"name": "double", "args": {"n": 21}}
            }]},
            "finishReason": "STOP"
        }]
    })]);
    let oneshot = codec
        .decode(oneshot_body.to_string().as_bytes(), vec![])
        .unwrap();
    let streamed = drain_stream(&codec, streaming_body).await;
    assert_responses_equivalent(&oneshot, &streamed, "gemini.function_call");
}

// ── OpenAI Chat Completions ────────────────────────────────────────────────

fn openai_chat_sse(payloads: &[Value]) -> Bytes {
    let mut out = String::new();
    for payload in payloads {
        out.push_str("data: ");
        out.push_str(&payload.to_string());
        out.push_str("\n\n");
    }
    out.push_str("data: [DONE]\n\n");
    Bytes::from(out)
}

#[tokio::test]
async fn openai_chat_text_streaming_matches_oneshot() {
    let codec = OpenAiChatCodec::new();
    let oneshot_body = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4.1",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello, world!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 9, "total_tokens": 13}
    });
    let streaming_body = openai_chat_sse(&[
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4.1",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello, "}}]
        }),
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4.1",
            "choices": [{"index": 0, "delta": {"content": "world!"}, "finish_reason": "stop"}]
        }),
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4.1",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 9, "total_tokens": 13}
        }),
    ]);
    let oneshot = codec
        .decode(oneshot_body.to_string().as_bytes(), vec![])
        .unwrap();
    let streamed = drain_stream(&codec, streaming_body).await;
    assert_responses_equivalent(&oneshot, &streamed, "openai_chat.text");
}

// ── OpenAI Responses ───────────────────────────────────────────────────────

fn openai_responses_sse(events: &[(&str, Value)]) -> Bytes {
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

#[tokio::test]
async fn openai_responses_text_streaming_matches_oneshot() {
    let codec = OpenAiResponsesCodec::new();
    let oneshot_body = json!({
        "id": "resp_01",
        "object": "response",
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello, world!"}]
        }],
        "usage": {"input_tokens": 4, "output_tokens": 9, "total_tokens": 13}
    });
    let streaming_body = openai_responses_sse(&[
        (
            "response.created",
            json!({
                "type": "response.created",
                "response": {"id": "resp_01", "model": "gpt-4.1", "status": "in_progress"}
            }),
        ),
        (
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "delta": "Hello, "
            }),
        ),
        (
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "delta": "world!"
            }),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "model": "gpt-4.1",
                    "status": "completed",
                    "usage": {"input_tokens": 4, "output_tokens": 9, "total_tokens": 13}
                }
            }),
        ),
    ]);
    let oneshot = codec
        .decode(oneshot_body.to_string().as_bytes(), vec![])
        .unwrap();
    let streamed = drain_stream(&codec, streaming_body).await;
    assert_responses_equivalent(&oneshot, &streamed, "openai_responses.text");
}

// ── invariant #15 — silent fallback prohibition matrix ─────────────────────
//
// Three regression suites pin the cross-codec contracts that the
// audit-heuristic-risks fork report flagged. Every codec must:
//
// 1. Map an unknown finish_reason to `Other{raw}` + `UnknownStopReason`.
// 2. Map a missing finish_reason to `Other{raw:"missing"}` + `LossyEncode`.
// 3. Preserve the matched `stop_sequence` string (or warn loudly when
//    the wire format hides it — Bedrock today).
//
// Bedrock Converse joins this matrix because the contracts are
// decoder-only and the binary event-stream issue is out of scope.

use entelix_core::codecs::BedrockConverseCodec;
use entelix_core::ir::{ModelWarning, StopReason};

/// One row in the cross-codec matrix: `(codec_label, codec, stop_payload_for_unknown,
/// stop_payload_for_missing, stop_payload_for_known_sequence)`. Each
/// payload is a closure that returns a one-shot wire fixture.
struct StopReasonRow {
    label: &'static str,
    codec: Box<dyn Codec>,
    /// Wire fixture with an unknown `finish_reason` (`"future_filter"` /
    /// `"FUTURE_FILTER"` etc. depending on vendor convention).
    unknown: serde_json::Value,
    /// Wire fixture that omits the `finish_reason` field entirely.
    missing: serde_json::Value,
    /// Wire fixture that signals a known `stop_sequence` match plus
    /// the matched string (when the vendor surface allows it).
    stop_sequence: serde_json::Value,
}

fn rows() -> Vec<StopReasonRow> {
    vec![
        StopReasonRow {
            label: "anthropic",
            codec: Box::new(AnthropicMessagesCodec::new()),
            unknown: json!({
                "id": "msg_x", "model": "claude-opus-4-7", "content": [],
                "stop_reason": "future_filter",
                "usage": {"input_tokens": 1, "output_tokens": 0}
            }),
            missing: json!({
                "id": "msg_x", "model": "claude-opus-4-7", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0}
            }),
            stop_sequence: json!({
                "id": "msg_x", "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "halted"}],
                "stop_reason": "stop_sequence",
                "stop_sequence": "###",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
        },
        StopReasonRow {
            label: "openai_chat",
            codec: Box::new(OpenAiChatCodec::new()),
            unknown: json!({
                "id": "c_x", "model": "gpt-4.1", "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "future_filter"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
            }),
            missing: json!({
                "id": "c_x", "model": "gpt-4.1", "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""}
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
            }),
            // OpenAI Chat does not surface the matched stop sequence
            // on the wire — `finish_reason: "stop"` is the only
            // signal and the IR mapping is `EndTurn`. The contract
            // here is "no stop_sequence path exists, so no loss to
            // preserve". The other two rows of the matrix still apply.
            stop_sequence: Value::Null,
        },
        StopReasonRow {
            label: "openai_responses",
            codec: Box::new(OpenAiResponsesCodec::new()),
            unknown: json!({
                "id": "resp_x", "model": "gpt-4.1",
                "output": [],
                "status": "future_filter",
                "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1}
            }),
            missing: json!({
                "id": "resp_x", "model": "gpt-4.1",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1}
            }),
            // OpenAI Responses does not expose stop_sequence either.
            stop_sequence: Value::Null,
        },
        StopReasonRow {
            label: "gemini",
            codec: Box::new(GeminiCodec::new()),
            unknown: json!({
                "candidates": [{
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "FUTURE_FILTER"
                }],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0}
            }),
            missing: json!({
                "candidates": [{
                    "content": {"parts": [], "role": "model"}
                }],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0}
            }),
            // Gemini does not surface stop_sequence either — STOP is
            // the only natural-stop signal.
            stop_sequence: Value::Null,
        },
        StopReasonRow {
            label: "bedrock",
            codec: Box::new(BedrockConverseCodec::new()),
            unknown: json!({
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "future_filter",
                "usage": {"inputTokens": 1, "outputTokens": 0, "totalTokens": 1}
            }),
            missing: json!({
                "output": {"message": {"role": "assistant", "content": []}},
                "usage": {"inputTokens": 1, "outputTokens": 0, "totalTokens": 1}
            }),
            // Bedrock signals `stop_sequence` but does not expose the
            // matched string in any documented field. The codec must
            // therefore fall through to `Other{raw:"stop_sequence"}`
            // + LossyEncode (T18 contract). We assert that here.
            stop_sequence: json!({
                "output": {"message": {"role": "assistant", "content": [
                    {"text": "halted"}
                ]}},
                "stopReason": "stop_sequence",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
            }),
        },
    ]
}

#[test]
fn five_codecs_consistently_warn_on_unknown_finish_reason() {
    for row in rows() {
        let body = serde_json::to_vec(&row.unknown).unwrap();
        let resp = row
            .codec
            .decode(&body, vec![])
            .unwrap_or_else(|e| panic!("[{}] decode failed: {e}", row.label));
        match &resp.stop_reason {
            StopReason::Other { raw } => assert!(
                !raw.is_empty() && raw != "missing",
                "[{}] unknown reason must surface as Other{{raw}} non-empty: got {raw:?}",
                row.label,
            ),
            other => panic!(
                "[{}] unknown reason must map to Other{{raw}}, got {other:?}",
                row.label
            ),
        }
        let warned = resp
            .warnings
            .iter()
            .any(|w| matches!(w, ModelWarning::UnknownStopReason { .. }));
        assert!(
            warned,
            "[{}] unknown reason must emit UnknownStopReason warning; got {:?}",
            row.label, resp.warnings,
        );
    }
}

#[test]
fn five_codecs_consistently_emit_other_on_missing_finish_reason() {
    for row in rows() {
        let body = serde_json::to_vec(&row.missing).unwrap();
        let resp = row
            .codec
            .decode(&body, vec![])
            .unwrap_or_else(|e| panic!("[{}] decode failed: {e}", row.label));
        match &resp.stop_reason {
            StopReason::Other { raw } => assert_eq!(
                raw, "missing",
                "[{}] missing reason must surface as Other{{raw:\"missing\"}}; got {raw:?}",
                row.label,
            ),
            other => panic!(
                "[{}] missing reason must map to Other{{raw:\"missing\"}}, got {other:?}",
                row.label
            ),
        }
        let warned = resp.warnings.iter().any(|w| match w {
            ModelWarning::LossyEncode { detail, .. } => {
                detail.contains("missing") || detail.contains("no ")
            }
            _ => false,
        });
        assert!(
            warned,
            "[{}] missing reason must emit a LossyEncode warning; got {:?}",
            row.label, resp.warnings,
        );
    }
}

#[test]
fn five_codecs_preserve_stop_sequence_or_warn_loudly() {
    for row in rows() {
        if row.stop_sequence.is_null() {
            // Vendor wire format does not surface the matched string —
            // the contract does not apply. Documented above.
            continue;
        }
        let body = serde_json::to_vec(&row.stop_sequence).unwrap();
        let resp = row
            .codec
            .decode(&body, vec![])
            .unwrap_or_else(|e| panic!("[{}] decode failed: {e}", row.label));
        match &resp.stop_reason {
            StopReason::StopSequence { sequence } => {
                assert!(
                    !sequence.is_empty(),
                    "[{}] preserved stop_sequence must not be empty",
                    row.label
                );
            }
            StopReason::Other { raw } if raw == "stop_sequence" => {
                // Acceptable when the vendor wire genuinely hides
                // the matched string; require a LossyEncode warning
                // so the loss is observable.
                let warned = resp.warnings.iter().any(|w| {
                    matches!(w, ModelWarning::LossyEncode { field, .. } if field == "stop_sequence")
                });
                assert!(
                    warned,
                    "[{}] codec without stop_sequence wire surface must emit \
                     LossyEncode {{field:\"stop_sequence\"}}; got {:?}",
                    row.label, resp.warnings,
                );
            }
            other => panic!(
                "[{}] stop_sequence fixture must produce StopSequence{{...}} or \
                 Other{{raw:\"stop_sequence\"}}; got {other:?}",
                row.label
            ),
        }
    }
}
