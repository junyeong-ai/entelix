//! Cross-vendor opaque round-trip carrier — encode → decode → re-encode
//! parity.
//!
//! For every codec that natively carries a vendor opaque token, asserts:
//!
//! 1. Decoding a wire response that carries the token yields a
//!    `ContentPart` whose `provider_echoes` carries the wire-shape blob.
//! 2. Re-encoding that `ContentPart` reproduces the original wire bytes
//!    for the token field — bit-for-bit verbatim.
//! 3. Cross-vendor isolation: a `ContentPart` carrying entries for
//!    multiple vendors round-trips through any one codec without
//!    affecting the wire-emitted bytes; foreign entries survive in the
//!    IR for downstream consumers.
//!
//! These are the canary tests for the cross-vendor refactor — any
//! regression in encode/decode round-trip shows up here first, before
//! a live provider call.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec, VertexAnthropicCodec, VertexGeminiCodec,
};
use entelix_core::ir::{
    CacheControl, ContentPart, Message, ModelRequest, ModelResponse, ProviderEchoSnapshot, Role,
};
use entelix_core::stream::{StreamAggregator, StreamDelta};
use serde_json::{Value, json};

/// Raw wire bytes for an Anthropic Messages response containing a
/// `thinking` block with a non-empty `signature`. The signature value
/// is the load-bearing artifact every downstream test asserts on.
const ANTHROPIC_SIGNATURE: &str = "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3UR";

/// Raw wire bytes for an Anthropic Messages response containing a
/// `redacted_thinking` block. The `data` value is opaque ciphertext
/// that must round-trip verbatim.
const ANTHROPIC_REDACTED_DATA: &str = "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIw";

/// Bedrock Converse signature value (Anthropic-on-Bedrock + Nova 2
/// share the same wire shape).
const BEDROCK_SIGNATURE: &str = "abcdef0123456789BedrockSignaturePayload==";

/// Bedrock Converse `redactedContent` (Anthropic-on-Bedrock).
const BEDROCK_REDACTED: &str = "QmVkcm9ja1JlZGFjdGVkQ29udGVudFBheWxvYWQ=";

/// Gemini `thought_signature` value — opaque base64-shaped string.
const GEMINI_THOUGHT_SIGNATURE: &str = "EhsMUkVHSU9OOmFzaWEtbm9ydGhlYXN0Mw==";

/// OpenAI Responses `encrypted_content` value.
const OPENAI_ENCRYPTED_CONTENT: &str = "gAAAAABoISQ24OyVRYbk_OPAQUE_PAYLOAD_xyz";

/// OpenAI Responses reasoning item id.
const OPENAI_REASONING_ITEM_ID: &str = "rs_abc123def456";

/// OpenAI Responses Response.id (for previous_response_id chain).
const OPENAI_RESPONSE_ID: &str = "resp_def456ghi789";

fn parse(body: &[u8]) -> Value {
    serde_json::from_slice(body).expect("body must be JSON")
}

fn find_thinking(content: &[ContentPart]) -> Option<&ContentPart> {
    content
        .iter()
        .find(|p| matches!(p, ContentPart::Thinking { .. }))
}

fn find_redacted(content: &[ContentPart]) -> Option<&ContentPart> {
    content
        .iter()
        .find(|p| matches!(p, ContentPart::RedactedThinking { .. }))
}

// ── Anthropic Messages ─────────────────────────────────────────────────────

#[test]
fn anthropic_signature_round_trip_preserves_value_verbatim() {
    let codec = AnthropicMessagesCodec::new();

    // Decode wire response → IR `Thinking` part with signature on
    // `provider_echoes`.
    let wire = json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 10},
        "content": [
            {
                "type": "thinking",
                "thinking": "Let me work this out step by step.",
                "signature": ANTHROPIC_SIGNATURE,
            },
            {"type": "text", "text": "The answer is 42."},
        ],
    });
    let response: ModelResponse = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .expect("decode must succeed");

    let thinking = find_thinking(&response.content).expect("thinking part must exist");
    let ContentPart::Thinking {
        text,
        provider_echoes,
        ..
    } = thinking
    else {
        unreachable!()
    };
    assert_eq!(text, "Let me work this out step by step.");
    let echoed_sig = ProviderEchoSnapshot::find_in(provider_echoes, "anthropic-messages")
        .and_then(|e| e.payload_str("signature"))
        .expect("anthropic-messages signature must round-trip onto provider_echoes");
    assert_eq!(echoed_sig, ANTHROPIC_SIGNATURE);

    // Re-encode the IR `Thinking` part back onto the wire — signature
    // must reappear bit-for-bit.
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![thinking.clone()]),
            Message::user("continue"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded_body = parse(&codec.encode(&req).unwrap().body);
    let assistant_blocks = encoded_body["messages"][1]["content"]
        .as_array()
        .expect("assistant content must be array");
    let thinking_block = assistant_blocks
        .iter()
        .find(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
        .expect("encoded thinking block must exist");
    assert_eq!(
        thinking_block.get("signature").and_then(Value::as_str),
        Some(ANTHROPIC_SIGNATURE),
        "signature must round-trip verbatim"
    );
}

#[test]
fn anthropic_empty_signature_is_skipped_on_decode() {
    // Anthropic emits `signature: ""` as a content_block_start
    // placeholder; the real signature arrives later via signature_delta
    // (streaming) or as the final block-level value (non-streaming).
    // An empty placeholder must NOT pollute the carrier — otherwise
    // a stale empty value can shadow the real one in `find_provider_echo`.
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 10},
        "content": [
            {"type": "thinking", "thinking": "stub", "signature": ""},
        ],
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let thinking = find_thinking(&response.content).unwrap();
    let ContentPart::Thinking {
        provider_echoes, ..
    } = thinking
    else {
        unreachable!()
    };
    assert!(
        provider_echoes.is_empty(),
        "empty signature placeholder must not produce a provider_echo entry"
    );
}

#[test]
fn anthropic_redacted_thinking_round_trips_data_field() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_3",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 10},
        "content": [
            {"type": "redacted_thinking", "data": ANTHROPIC_REDACTED_DATA},
        ],
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();

    let redacted = find_redacted(&response.content).expect("redacted_thinking must round-trip");
    let ContentPart::RedactedThinking { provider_echoes } = redacted else {
        unreachable!()
    };
    let echoed_data = ProviderEchoSnapshot::find_in(provider_echoes, "anthropic-messages")
        .and_then(|e| e.payload_str("data"))
        .expect("redacted data must round-trip");
    assert_eq!(echoed_data, ANTHROPIC_REDACTED_DATA);

    // Re-encode → wire `redacted_thinking` block with the same data.
    let req = ModelRequest {
        model: "claude-3-7-sonnet".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![redacted.clone()]),
            Message::user("continue"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let blocks = body["messages"][1]["content"].as_array().unwrap();
    let block = blocks
        .iter()
        .find(|b| b.get("type").and_then(Value::as_str) == Some("redacted_thinking"))
        .expect("encoded redacted_thinking block must exist");
    assert_eq!(
        block.get("data").and_then(Value::as_str),
        Some(ANTHROPIC_REDACTED_DATA),
    );
}

// ── Vertex Anthropic — composition delegates to Anthropic ──────────────────

#[test]
fn vertex_anthropic_signature_inherits_anthropic_provider_key() {
    let codec = VertexAnthropicCodec::new();
    let wire = json!({
        "id": "msg_v1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 10},
        "content": [
            {"type": "thinking", "thinking": "cross-platform thinking", "signature": ANTHROPIC_SIGNATURE},
        ],
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let thinking = find_thinking(&response.content).unwrap();
    let ContentPart::Thinking {
        provider_echoes, ..
    } = thinking
    else {
        unreachable!()
    };
    // Vertex Anthropic shares the `anthropic-messages` provider key
    // because the wire shape is identical to first-party Anthropic
    // and Anthropic documents cross-platform signature compatibility.
    let echoed = ProviderEchoSnapshot::find_in(provider_echoes, "anthropic-messages")
        .and_then(|e| e.payload_str("signature"));
    assert_eq!(echoed, Some(ANTHROPIC_SIGNATURE));
}

// ── Bedrock Converse ───────────────────────────────────────────────────────

#[test]
fn bedrock_converse_signature_round_trip() {
    let codec = BedrockConverseCodec::new();
    let wire = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {
                        "reasoningText": {
                            "text": "Considering the question…",
                            "signature": BEDROCK_SIGNATURE,
                        },
                    }},
                    {"text": "Done."},
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 8, "totalTokens": 20},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let thinking = find_thinking(&response.content).expect("thinking must exist");
    let ContentPart::Thinking {
        text,
        provider_echoes,
        ..
    } = thinking
    else {
        unreachable!()
    };
    assert_eq!(text, "Considering the question…");
    let echoed = ProviderEchoSnapshot::find_in(provider_echoes, "bedrock-converse")
        .and_then(|e| e.payload_str("signature"))
        .expect("bedrock-converse signature must round-trip");
    assert_eq!(echoed, BEDROCK_SIGNATURE);

    // Re-encode → wire `reasoningContent.reasoningText.signature`.
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-v1:0".into(),
        messages: vec![
            Message::user("question"),
            Message::new(Role::Assistant, vec![thinking.clone()]),
            Message::user("more"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let assistant_content = body["messages"][1]["content"].as_array().unwrap();
    let reasoning = assistant_content
        .iter()
        .find_map(|b| b.get("reasoningContent"))
        .expect("reasoningContent block must exist on encoded assistant message");
    assert_eq!(
        reasoning["reasoningText"]["signature"].as_str(),
        Some(BEDROCK_SIGNATURE),
    );
}

#[test]
fn bedrock_converse_redacted_content_routes_to_redacted_thinking_variant() {
    let codec = BedrockConverseCodec::new();
    // A pure `redactedContent` block (no reasoningText) maps onto the
    // typed `RedactedThinking` IR variant — preserves invariant 6 by
    // routing through a typed channel rather than silently dropping.
    let wire = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"redactedContent": BEDROCK_REDACTED}},
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 0, "totalTokens": 5},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let redacted =
        find_redacted(&response.content).expect("RedactedThinking variant must round-trip");
    let ContentPart::RedactedThinking { provider_echoes } = redacted else {
        unreachable!()
    };
    let echoed = ProviderEchoSnapshot::find_in(provider_echoes, "bedrock-converse")
        .and_then(|e| e.payload_str("redacted_content"))
        .expect("redacted_content must ride on provider_echoes");
    assert_eq!(echoed, BEDROCK_REDACTED);
}

// ── Gemini (AI Studio + Vertex Gemini share the wire shape + key) ──────────

#[test]
fn gemini_thought_signature_round_trips_on_thinking_part() {
    let codec = GeminiCodec::new();
    let wire = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {
                        "thought": true,
                        "text": "Reasoning intermediate.",
                        "thought_signature": GEMINI_THOUGHT_SIGNATURE,
                    },
                    {"text": "Final answer."},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let thinking = find_thinking(&response.content).unwrap();
    let ContentPart::Thinking {
        provider_echoes, ..
    } = thinking
    else {
        unreachable!()
    };
    let echoed = ProviderEchoSnapshot::find_in(provider_echoes, "gemini")
        .and_then(|e| e.payload_str("thought_signature"))
        .expect("gemini thought_signature must round-trip");
    assert_eq!(echoed, GEMINI_THOUGHT_SIGNATURE);

    // Re-encode — wire field MUST be snake_case `thought_signature`,
    // never the legacy camelCase `thoughtSignature` (Vertex AI rejects
    // camelCase).
    let req = ModelRequest {
        model: "gemini-3.1-pro".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![thinking.clone()]),
            Message::user("continue"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let parts = body["contents"][1]["parts"].as_array().unwrap();
    let thinking_part = parts
        .iter()
        .find(|p| p.get("thought").and_then(Value::as_bool) == Some(true))
        .expect("thinking part must round-trip on wire");
    assert_eq!(
        thinking_part
            .get("thought_signature")
            .and_then(Value::as_str),
        Some(GEMINI_THOUGHT_SIGNATURE),
        "encoder must emit snake_case thought_signature"
    );
    assert!(
        thinking_part.get("thoughtSignature").is_none(),
        "encoder must NOT emit camelCase thoughtSignature (Vertex rejects)"
    );
}

#[test]
fn gemini_thought_signature_round_trips_on_function_call_part() {
    // Gemini 3.x rejects the next request with HTTP 400 if the first
    // `functionCall` of a step is missing its thought_signature on
    // round-trip. This test pins the round-trip end-to-end.
    let codec = GeminiCodec::new();
    let wire = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "get_weather", "args": {"city": "Seoul"}},
                        "thought_signature": GEMINI_THOUGHT_SIGNATURE,
                    },
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let tool_use = response
        .content
        .iter()
        .find(|p| matches!(p, ContentPart::ToolUse { .. }))
        .expect("ToolUse must round-trip");
    let ContentPart::ToolUse {
        name,
        provider_echoes,
        ..
    } = tool_use
    else {
        unreachable!()
    };
    assert_eq!(name, "get_weather");
    let echoed = ProviderEchoSnapshot::find_in(provider_echoes, "gemini")
        .and_then(|e| e.payload_str("thought_signature"))
        .expect("thought_signature must ride on functionCall ToolUse");
    assert_eq!(echoed, GEMINI_THOUGHT_SIGNATURE);

    // Re-encode → wire `Part` carrying both functionCall and
    // sibling thought_signature.
    let req = ModelRequest {
        model: "gemini-3.1-pro".into(),
        messages: vec![
            Message::user("weather?"),
            Message::new(Role::Assistant, vec![tool_use.clone()]),
            Message::user("more"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let parts = body["contents"][1]["parts"].as_array().unwrap();
    let fc_part = parts
        .iter()
        .find(|p| p.get("functionCall").is_some())
        .expect("functionCall part must round-trip");
    assert_eq!(
        fc_part.get("thought_signature").and_then(Value::as_str),
        Some(GEMINI_THOUGHT_SIGNATURE),
    );
}

#[test]
fn gemini_decoder_tolerates_legacy_camelcase_thoughtsignature_on_wire() {
    // Gemini servers historically emitted `thoughtSignature` (camelCase);
    // the decoder accepts both spellings to stay forward-compatible
    // with whichever transport variant the API ships. The encoder
    // strictly emits snake_case (other test) — but decode must tolerate.
    let codec = GeminiCodec::new();
    let wire = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"thought": true, "text": "hmm", "thoughtSignature": GEMINI_THOUGHT_SIGNATURE},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5, "totalTokenCount": 10},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let thinking = find_thinking(&response.content).unwrap();
    let ContentPart::Thinking {
        provider_echoes, ..
    } = thinking
    else {
        unreachable!()
    };
    let echoed = ProviderEchoSnapshot::find_in(provider_echoes, "gemini")
        .and_then(|e| e.payload_str("thought_signature"));
    assert_eq!(echoed, Some(GEMINI_THOUGHT_SIGNATURE));
}

#[test]
fn vertex_gemini_inherits_gemini_provider_key_through_composition() {
    let codec = VertexGeminiCodec::new();
    let wire = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"thought": true, "text": "vertex thinking", "thought_signature": GEMINI_THOUGHT_SIGNATURE},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5, "totalTokenCount": 10},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let thinking = find_thinking(&response.content).unwrap();
    let ContentPart::Thinking {
        provider_echoes, ..
    } = thinking
    else {
        unreachable!()
    };
    let echoed = ProviderEchoSnapshot::find_in(provider_echoes, "gemini")
        .and_then(|e| e.payload_str("thought_signature"));
    assert_eq!(echoed, Some(GEMINI_THOUGHT_SIGNATURE));
}

// ── OpenAI Responses — three-tier (part / response / request) carrier ─────

#[test]
fn openai_responses_encrypted_content_round_trips_on_reasoning_item() {
    let codec = OpenAiResponsesCodec::new();
    let wire = json!({
        "id": OPENAI_RESPONSE_ID,
        "object": "response",
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "id": OPENAI_REASONING_ITEM_ID,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Step 1: parse input."}],
                "encrypted_content": OPENAI_ENCRYPTED_CONTENT,
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();

    // Per-part: encrypted_content + reasoning item id.
    let thinking = find_thinking(&response.content).expect("reasoning item → thinking part");
    let ContentPart::Thinking {
        text,
        provider_echoes,
        ..
    } = thinking
    else {
        unreachable!()
    };
    assert_eq!(text, "Step 1: parse input.");
    let echo = ProviderEchoSnapshot::find_in(provider_echoes, "openai-responses")
        .expect("reasoning item must carry openai-responses echo");
    assert_eq!(
        echo.payload_str("encrypted_content"),
        Some(OPENAI_ENCRYPTED_CONTENT),
    );
    assert_eq!(echo.payload_str("id"), Some(OPENAI_REASONING_ITEM_ID));

    // Per-response: Response.id captured at the response root.
    let response_echo =
        ProviderEchoSnapshot::find_in(&response.provider_echoes, "openai-responses")
            .expect("Response.id must ride on ModelResponse.provider_echoes");
    assert_eq!(
        response_echo.payload_str("response_id"),
        Some(OPENAI_RESPONSE_ID),
    );
}

#[test]
fn openai_responses_previous_response_id_request_chain() {
    let codec = OpenAiResponsesCodec::new();
    // Operator chains a fresh request to a prior response by populating
    // `continued_from`. The codec must emit `previous_response_id` on
    // the request body root.
    let req = ModelRequest {
        model: "gpt-5".into(),
        messages: vec![Message::user("continue the conversation")],
        max_tokens: Some(256),
        continued_from: vec![ProviderEchoSnapshot::for_provider(
            "openai-responses",
            "response_id",
            OPENAI_RESPONSE_ID,
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(
        body.get("previous_response_id").and_then(Value::as_str),
        Some(OPENAI_RESPONSE_ID),
        "encoder must emit previous_response_id when continued_from carries the chain",
    );
}

#[test]
fn openai_responses_function_call_item_id_round_trips() {
    let codec = OpenAiResponsesCodec::new();
    let wire = json!({
        "id": "resp_xyz",
        "object": "response",
        "model": "gpt-5",
        "status": "in_progress",
        "output": [
            {
                "id": "fc_def123",
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "{\"city\":\"Seoul\"}",
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let tool_use = response
        .content
        .iter()
        .find(|p| matches!(p, ContentPart::ToolUse { .. }))
        .unwrap();
    let ContentPart::ToolUse {
        id,
        provider_echoes,
        ..
    } = tool_use
    else {
        unreachable!()
    };
    assert_eq!(id, "call_abc", "ContentPart::ToolUse.id is the call_id");
    let echo_id = ProviderEchoSnapshot::find_in(provider_echoes, "openai-responses")
        .and_then(|e| e.payload_str("id"));
    assert_eq!(
        echo_id,
        Some("fc_def123"),
        "function_call item id (fc_…) must round-trip on provider_echoes for stateless replay"
    );

    // Re-encode → assistant function_call item carries both `call_id`
    // and the per-item `id` from provider_echoes.
    let req = ModelRequest {
        model: "gpt-5".into(),
        messages: vec![
            Message::user("weather?"),
            Message::new(Role::Assistant, vec![tool_use.clone()]),
            Message::tool_result_json("call_abc", "get_weather", json!({"temp": 15})),
        ],
        max_tokens: Some(256),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let input_items = body["input"].as_array().unwrap();
    let fc_item = input_items
        .iter()
        .find(|i| i.get("type").and_then(Value::as_str) == Some("function_call"))
        .expect("function_call item must be re-emitted");
    assert_eq!(
        fc_item.get("call_id").and_then(Value::as_str),
        Some("call_abc")
    );
    assert_eq!(fc_item.get("id").and_then(Value::as_str), Some("fc_def123"));
}

// ── OpenAI Chat — no opaque round-trip token; foreign entries drop ─────────

#[test]
fn openai_chat_silently_drops_foreign_provider_echoes_on_encode() {
    let codec = OpenAiChatCodec::new();
    // OpenAI Chat Completions has no carrier slot for opaque round-trip
    // tokens. Foreign entries (e.g. an Anthropic signature carried over
    // from a prior transport) must NOT reach the wire — but should also
    // not produce a LossyEncode warning, because Chat Completions has
    // nothing to lose (no carrier slot exists).
    let part = ContentPart::Thinking {
        text: "prior thinking".into(),
        cache_control: None,
        provider_echoes: vec![ProviderEchoSnapshot::for_provider(
            "anthropic-messages",
            "signature",
            ANTHROPIC_SIGNATURE,
        )],
    };
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![part]),
            Message::user("continue"),
        ],
        max_tokens: Some(256),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    // The Anthropic signature must NOT leak onto the OpenAI Chat wire.
    let body_str = body.to_string();
    assert!(
        !body_str.contains(ANTHROPIC_SIGNATURE),
        "foreign anthropic signature must not appear on OpenAI Chat wire"
    );
}

// ── Cross-vendor isolation ─────────────────────────────────────────────────

#[test]
fn cross_vendor_anthropic_and_gemini_isolation_on_anthropic_wire() {
    // A `Thinking` part carrying entries for both `anthropic-messages`
    // and `gemini` providers — sent through the Anthropic codec —
    // emits ONLY the Anthropic signature on the wire. The Gemini blob
    // survives in IR for downstream consumers but never reaches the
    // wire (codec autonomy).
    let codec = AnthropicMessagesCodec::new();
    let part = ContentPart::Thinking {
        text: "shared thinking".into(),
        cache_control: None,
        provider_echoes: vec![
            ProviderEchoSnapshot::for_provider(
                "anthropic-messages",
                "signature",
                ANTHROPIC_SIGNATURE,
            ),
            ProviderEchoSnapshot::for_provider(
                "gemini",
                "thought_signature",
                GEMINI_THOUGHT_SIGNATURE,
            ),
        ],
    };
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![part]),
            Message::user("continue"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let body_str = body.to_string();
    assert!(
        body_str.contains(ANTHROPIC_SIGNATURE),
        "Anthropic codec must emit its own signature on the wire"
    );
    assert!(
        !body_str.contains(GEMINI_THOUGHT_SIGNATURE),
        "Anthropic codec must NOT emit the Gemini thought_signature on the wire"
    );
    assert!(
        !body_str.contains("thought_signature"),
        "Anthropic wire must not carry the Gemini wire field name"
    );
}

#[test]
fn cross_vendor_anthropic_and_gemini_isolation_on_gemini_wire() {
    // Symmetric to the previous test — the same multi-vendor part
    // through the Gemini codec emits ONLY the Gemini blob.
    let codec = GeminiCodec::new();
    let part = ContentPart::Thinking {
        text: "shared thinking".into(),
        cache_control: None,
        provider_echoes: vec![
            ProviderEchoSnapshot::for_provider(
                "anthropic-messages",
                "signature",
                ANTHROPIC_SIGNATURE,
            ),
            ProviderEchoSnapshot::for_provider(
                "gemini",
                "thought_signature",
                GEMINI_THOUGHT_SIGNATURE,
            ),
        ],
    };
    let req = ModelRequest {
        model: "gemini-3.1-pro".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![part]),
            Message::user("continue"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let body_str = body.to_string();
    assert!(
        body_str.contains(GEMINI_THOUGHT_SIGNATURE),
        "Gemini codec must emit thought_signature on the wire"
    );
    assert!(
        !body_str.contains(ANTHROPIC_SIGNATURE),
        "Gemini codec must NOT emit the Anthropic signature on the wire"
    );
}

#[test]
fn cross_vendor_ir_preserves_foreign_blobs_through_decode() {
    // Decode an Anthropic response → the Thinking part carries
    // `anthropic-messages` provider_echoes. If a downstream consumer
    // attaches a `gemini` blob to the same part (e.g. transport
    // switch), encoding through Anthropic again must preserve the
    // foreign blob in IR (audit-faithful) while emitting only the
    // Anthropic signature on the wire.
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 5},
        "content": [
            {"type": "thinking", "thinking": "reasoning", "signature": ANTHROPIC_SIGNATURE},
        ],
    });
    let response = codec
        .decode(wire.to_string().as_bytes(), Vec::new())
        .unwrap();
    let mut thinking = response.content.into_iter().next().unwrap();
    // Operator simulates a transport switch — attach a Gemini blob.
    thinking = thinking.with_provider_echo(ProviderEchoSnapshot::for_provider(
        "gemini",
        "thought_signature",
        GEMINI_THOUGHT_SIGNATURE,
    ));
    // Both echoes are present in IR.
    assert_eq!(thinking.provider_echoes().len(), 2);

    // Re-encode through Anthropic — only the Anthropic signature appears
    // on the wire.
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![thinking]),
            Message::user("continue"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let body_str = body.to_string();
    assert!(body_str.contains(ANTHROPIC_SIGNATURE));
    assert!(!body_str.contains(GEMINI_THOUGHT_SIGNATURE));
}

// ── Streaming response-level carrier (StreamDelta::Start propagation) ─────

#[test]
fn stream_aggregator_propagates_start_provider_echoes_to_response() {
    // The streaming Start delta carries a response-level
    // ProviderEchoSnapshot (e.g. OpenAI Responses Response.id). The
    // aggregator must surface it on `ModelResponse.provider_echoes`
    // at finalize so streaming decode mirrors the non-streaming
    // decode path — an operator chaining a stateless turn via
    // `previous_response_id` reads the same field regardless of
    // streaming mode.
    let mut agg = StreamAggregator::new();
    agg.push(StreamDelta::Start {
        id: OPENAI_RESPONSE_ID.into(),
        model: "gpt-5".into(),
        provider_echoes: vec![ProviderEchoSnapshot::for_provider(
            "openai-responses",
            "response_id",
            OPENAI_RESPONSE_ID,
        )],
    })
    .unwrap();
    agg.push(StreamDelta::TextDelta {
        text: "ok".into(),
        provider_echoes: Vec::new(),
    })
    .unwrap();
    agg.push(StreamDelta::Stop {
        stop_reason: entelix_core::ir::StopReason::EndTurn,
    })
    .unwrap();
    let response = agg.finalize().unwrap();

    let echoed = ProviderEchoSnapshot::find_in(&response.provider_echoes, "openai-responses")
        .and_then(|e| e.payload_str("response_id"));
    assert_eq!(
        echoed,
        Some(OPENAI_RESPONSE_ID),
        "Start.provider_echoes must surface on ModelResponse.provider_echoes",
    );
}

#[test]
fn openai_responses_streaming_captures_response_id_for_chain_continuation() {
    // Full streaming SSE → ModelResponse path: the codec emits Start
    // carrying the wrapped Response.id; the aggregator surfaces it
    // on ModelResponse.provider_echoes; an operator can then build
    // ModelRequest.continued_from from it for the next turn.
    let codec = OpenAiResponsesCodec::new();
    let sse = format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\n",
        json!({
            "type": "response.created",
            "response": {"id": OPENAI_RESPONSE_ID, "model": "gpt-5"},
        }),
        json!({
            "type": "response.output_text.delta",
            "delta": "Hello.",
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": OPENAI_RESPONSE_ID,
                "model": "gpt-5",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
            },
        }),
        json!({"type": "response.output_text.done"}),
        json!({"type": "done"}),
    );
    let bytes = bytes::Bytes::from(sse);
    let stream = codec.decode_stream(
        Box::pin(futures::stream::iter(vec![Ok::<_, entelix_core::Error>(
            bytes,
        )])),
        Vec::new(),
    );
    let response = futures::executor::block_on(async {
        let mut agg = StreamAggregator::new();
        let mut deltas = stream;
        while let Some(item) = futures::StreamExt::next(&mut deltas).await {
            agg.push(item.unwrap()).unwrap();
        }
        agg.finalize()
    })
    .unwrap();

    // Round-trip continuation: build the next ModelRequest from
    // ModelResponse.provider_echoes.
    let req_next = ModelRequest {
        model: "gpt-5".into(),
        messages: vec![Message::user("continue")],
        max_tokens: Some(64),
        continued_from: response.provider_echoes,
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req_next).unwrap().body);
    assert_eq!(
        body.get("previous_response_id").and_then(Value::as_str),
        Some(OPENAI_RESPONSE_ID),
        "streaming-decoded Response.id must survive into the next request's previous_response_id chain",
    );
}

// ── cache_control + provider_echoes co-existence ──────────────────────────

#[test]
fn cache_control_and_provider_echoes_coexist_on_same_part() {
    // A `Thinking` part may carry BOTH a `cache_control` directive
    // (operator-set) AND `provider_echoes` (model-emitted opaque
    // token). Each ride on its own native wire field; encode must
    // emit both without interference.
    let codec = AnthropicMessagesCodec::new();
    let part = ContentPart::Thinking {
        text: "step-by-step plan".into(),
        cache_control: Some(CacheControl::one_hour()),
        provider_echoes: vec![ProviderEchoSnapshot::for_provider(
            "anthropic-messages",
            "signature",
            ANTHROPIC_SIGNATURE,
        )],
    };
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![part]),
            Message::user("more"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let blocks = body["messages"][1]["content"].as_array().unwrap();
    let thinking_block = blocks
        .iter()
        .find(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
        .unwrap();
    // signature on the thinking block.
    assert_eq!(
        thinking_block.get("signature").and_then(Value::as_str),
        Some(ANTHROPIC_SIGNATURE),
    );
    // cache_control on the same block, ttl: 1h tier.
    assert_eq!(thinking_block["cache_control"]["type"], "ephemeral");
    assert_eq!(thinking_block["cache_control"]["ttl"], "1h");
}

// ── Structured output (ResponseFormat) round-trip with vendor opaque ─────

#[test]
fn structured_output_request_with_thinking_history_preserves_signature() {
    // Common multi-turn pattern: model emitted a Thinking block on
    // turn 1, the harness re-submits the conversation on turn 2 with
    // ResponseFormat::Json for typed extraction. The Thinking part's
    // signature must round-trip onto the next request's wire body
    // alongside the response_format directive.
    use entelix_core::ir::{JsonSchemaSpec, ResponseFormat};

    let codec = AnthropicMessagesCodec::new();
    let thinking_with_signature = ContentPart::Thinking {
        text: "let me reason".into(),
        cache_control: None,
        provider_echoes: vec![ProviderEchoSnapshot::for_provider(
            "anthropic-messages",
            "signature",
            ANTHROPIC_SIGNATURE,
        )],
    };
    let schema = JsonSchemaSpec::new(
        "Reply",
        json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }),
    )
    .unwrap();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![
            Message::user("question"),
            Message::new(Role::Assistant, vec![thinking_with_signature]),
            Message::user("emit JSON"),
        ],
        max_tokens: Some(1024),
        response_format: Some(ResponseFormat::strict(schema)),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    // Signature must survive the round-trip even when the request
    // also sets a structured-output format (different IR field; the
    // two must be orthogonal).
    let body_str = body.to_string();
    assert!(
        body_str.contains(ANTHROPIC_SIGNATURE),
        "thinking signature must round-trip even when ResponseFormat is set"
    );
}

// ── ContentPart::clone preserves provider_echoes ──────────────────────────

#[test]
fn content_part_clone_preserves_provider_echoes_verbatim() {
    // The auto-compaction adapter and audit replay both route through
    // `ContentPart::clone()` + reconstruction. If clone ever lost
    // `provider_echoes`, model-emitted opaque tokens would silently
    // disappear after one compaction pass. Pinning the clone semantics
    // here catches the regression at the smallest possible surface
    // (no compactor / session machinery needed — that's covered in
    // entelix-session's own integration tests).
    let original = ContentPart::Thinking {
        text: "internal".into(),
        cache_control: None,
        provider_echoes: vec![ProviderEchoSnapshot::for_provider(
            "anthropic-messages",
            "signature",
            ANTHROPIC_SIGNATURE,
        )],
    };
    let cloned = original.clone();
    // Both the original and the clone must carry identical
    // provider_echoes payloads — pinning the structural clone
    // contract that compaction and audit replay rely on.
    assert_eq!(
        ProviderEchoSnapshot::find_in(original.provider_echoes(), "anthropic-messages")
            .and_then(|e| e.payload_str("signature")),
        ProviderEchoSnapshot::find_in(cloned.provider_echoes(), "anthropic-messages")
            .and_then(|e| e.payload_str("signature")),
    );
    assert_eq!(
        ProviderEchoSnapshot::find_in(cloned.provider_echoes(), "anthropic-messages")
            .and_then(|e| e.payload_str("signature")),
        Some(ANTHROPIC_SIGNATURE),
    );
}
