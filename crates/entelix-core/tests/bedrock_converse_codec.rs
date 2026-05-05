//! `BedrockConverseCodec` encode/decode tests. Streaming uses the
//! `Codec::decode_stream` default fallback (binary event-stream lives
//! in the cloud crate alongside `SigV4` signing).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use entelix_core::codecs::{BedrockConverseCodec, Codec};
use entelix_core::ir::{
    ContentPart, Message, ModelRequest, ModelWarning, Role, StopReason, ToolChoice,
    ToolResultContent, ToolSpec,
};
use serde_json::{Value, json};

fn parse(body: &[u8]) -> Value {
    serde_json::from_slice(body).unwrap()
}

#[test]
fn encode_minimal_request_targets_converse_path() {
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-7-v1:0".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    assert_eq!(
        encoded.path,
        "/model/anthropic.claude-opus-4-7-v1:0/converse"
    );
    let body = parse(&encoded.body);
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
}

#[test]
fn encode_streaming_uses_converse_stream_path() {
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "model-1".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode_streaming(&req).unwrap();
    assert!(encoded.streaming);
    assert!(encoded.path.ends_with("/converse-stream"));
    assert_eq!(
        encoded.headers.get(http::header::ACCEPT).unwrap(),
        "application/vnd.amazon.eventstream"
    );
}

#[test]
fn encode_system_routes_into_top_level_system_array() {
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "m".into(),
        messages: vec![Message::user("hi")],
        system: "Be terse.".into(),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let system = body["system"].as_array().unwrap();
    assert_eq!(system[0]["text"], "Be terse.");
}

#[test]
fn encode_assistant_tool_use_emits_tool_use_block() {
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "m".into(),
        messages: vec![Message::new(
            Role::Assistant,
            vec![ContentPart::ToolUse {
                id: "tu-1".into(),
                name: "double".into(),
                input: json!({"n": 21}),
            }],
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let part = &body["messages"][0]["content"][0];
    assert_eq!(part["toolUse"]["toolUseId"], "tu-1");
    assert_eq!(part["toolUse"]["name"], "double");
    assert_eq!(part["toolUse"]["input"]["n"], 21);
}

#[test]
fn encode_tool_result_wraps_into_user_message_with_tool_result_block() {
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "m".into(),
        messages: vec![Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                tool_use_id: "tu-1".into(),
                name: "double".into(),
                content: ToolResultContent::Json(json!({"doubled": 42})),
                is_error: false,
                cache_control: None,
            }],
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let msg = &body["messages"][0];
    assert_eq!(msg["role"], "user");
    let result = &msg["content"][0]["toolResult"];
    assert_eq!(result["toolUseId"], "tu-1");
    assert_eq!(result["status"], "success");
    assert_eq!(result["content"][0]["json"]["doubled"], 42);
}

#[test]
fn encode_tools_emits_toolspec_array() {
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "m".into(),
        messages: vec![Message::user("calc")],
        tools: vec![ToolSpec::function(
            "double",
            "doubles n",
            json!({"type": "object"}),
        )],
        tool_choice: ToolChoice::Required,
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let tool = &body["toolConfig"]["tools"][0]["toolSpec"];
    assert_eq!(tool["name"], "double");
    assert!(body["toolConfig"]["toolChoice"]["any"].is_object());
}

// ── decode ─────────────────────────────────────────────────────────────────

#[test]
fn decode_text_response() {
    let codec = BedrockConverseCodec::new();
    let body = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{ "text": "Hello!" }]
            }
        },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 4, "outputTokens": 1 }
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.input_tokens, 4);
    assert_eq!(response.usage.output_tokens, 1);
    assert!(matches!(
        response.content[0],
        ContentPart::Text { ref text, .. } if text == "Hello!"
    ));
}

#[test]
fn decode_tool_use_response() {
    let codec = BedrockConverseCodec::new();
    let body = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "tu-1",
                        "name": "double",
                        "input": { "n": 21 }
                    }
                }]
            }
        },
        "stopReason": "tool_use"
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    if let ContentPart::ToolUse { id, name, input } = &response.content[0] {
        assert_eq!(id, "tu-1");
        assert_eq!(name, "double");
        assert_eq!(input["n"], 21);
    } else {
        panic!("expected tool_use");
    }
}

#[test]
fn decode_unknown_stop_reason_emits_warning() {
    let codec = BedrockConverseCodec::new();
    let body = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{ "text": "ok" }]
            }
        },
        "stopReason": "exotic_reason"
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert!(matches!(
        response.stop_reason,
        StopReason::Other { ref raw } if raw == "exotic_reason"
    ));
    assert!(!response.warnings.is_empty());
}

// ── ProviderExtensions wire-up ─────────────────────────────────────────────

#[test]
fn bedrock_ext_guardrail_threads_into_guardrail_config() {
    use entelix_core::ir::{BedrockExt, BedrockGuardrail, ProviderExtensions};
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-3-5-sonnet-20240620-v1:0".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default().with_bedrock(
            BedrockExt::default().with_guardrail(BedrockGuardrail {
                identifier: "gr-abc".into(),
                version: "DRAFT".into(),
            }),
        ),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["guardrailConfig"]["guardrailIdentifier"], "gr-abc");
    assert_eq!(body["guardrailConfig"]["guardrailVersion"], "DRAFT");
}

#[test]
fn bedrock_ext_performance_tier_threads_into_performance_config() {
    use entelix_core::ir::{BedrockExt, ProviderExtensions};
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-3-5-sonnet-20240620-v1:0".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_bedrock(BedrockExt::default().with_performance_config_tier("optimized")),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["performanceConfig"]["latency"], "optimized");
}

#[test]
fn bedrock_threads_anthropic_thinking_through_additional_model_request_fields() {
    use entelix_core::ir::ReasoningEffort;
    let codec = BedrockConverseCodec::new();
    // Sonnet accepts explicit budget tokens on Bedrock — Opus 4.7
    // hosted on Bedrock is adaptive-only and maps onto a different
    // shape (covered by the dedicated Opus tests).
    let req = ModelRequest {
        model: "anthropic.claude-sonnet-4-6".into(),
        messages: vec![Message::user("solve")],
        reasoning_effort: Some(ReasoningEffort::Medium),
        max_tokens: Some(8192),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    let thinking = body
        .get("additionalModelRequestFields")
        .and_then(|f| f.get("thinking"))
        .expect("thinking must ride through additionalModelRequestFields on Anthropic-on-Bedrock");
    assert_eq!(thinking["type"], "enabled");
    assert_eq!(thinking["budget_tokens"], 4096);
    // No `LossyEncode` for thinking — Anthropic-on-Bedrock honours
    // the same shape Anthropic Messages does.
    let thinking_lossy = encoded.warnings.iter().any(|w| match w {
        ModelWarning::LossyEncode { field, .. } => field.contains("thinking"),
        _ => false,
    });
    assert!(
        !thinking_lossy,
        "thinking must not emit LossyEncode on Bedrock — it rides through verbatim: {:?}",
        encoded.warnings
    );
}

#[test]
fn bedrock_emits_field_precise_lossy_for_anthropic_only_ext_fields() {
    use entelix_core::ir::{AnthropicExt, ProviderExtensions};
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default().with_anthropic(
            AnthropicExt::default()
                .with_disable_parallel_tool_use(true)
                .with_user_id("op-1"),
        ),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let warnings = codec.encode(&req).unwrap().warnings;
    let saw_disable = warnings.iter().any(|w| {
        matches!(w, ModelWarning::LossyEncode { field, .. }
            if field == "provider_extensions.anthropic.disable_parallel_tool_use")
    });
    let saw_user = warnings.iter().any(|w| {
        matches!(w, ModelWarning::LossyEncode { field, .. }
            if field == "provider_extensions.anthropic.user_id")
    });
    assert!(saw_disable && saw_user, "{warnings:?}");
}

#[test]
fn bedrock_extract_rate_limit_captures_amzn_bedrock_headers_and_retry_after() {
    let codec = BedrockConverseCodec::new();
    let mut headers = http::HeaderMap::new();
    headers.insert(
        "x-amzn-bedrock-input-token-count",
        http::HeaderValue::from_static("123"),
    );
    headers.insert(
        "x-amzn-bedrock-output-token-count",
        http::HeaderValue::from_static("45"),
    );
    headers.insert(
        "x-amzn-bedrock-invocation-latency",
        http::HeaderValue::from_static("780"),
    );
    headers.insert("retry-after", http::HeaderValue::from_static("12"));
    let snap = codec
        .extract_rate_limit(&headers)
        .expect("snapshot present");
    assert_eq!(
        snap.raw
            .get("x-amzn-bedrock-input-token-count")
            .map(String::as_str),
        Some("123")
    );
    assert_eq!(
        snap.raw
            .get("x-amzn-bedrock-output-token-count")
            .map(String::as_str),
        Some("45")
    );
    assert_eq!(
        snap.raw
            .get("x-amzn-bedrock-invocation-latency")
            .map(String::as_str),
        Some("780")
    );
    assert_eq!(snap.raw.get("retry-after").map(String::as_str), Some("12"));
}

#[test]
fn bedrock_extract_rate_limit_returns_none_when_no_signals_present() {
    let codec = BedrockConverseCodec::new();
    let headers = http::HeaderMap::new();
    assert!(codec.extract_rate_limit(&headers).is_none());
}

#[test]
fn bedrock_codec_warns_on_foreign_vendor_extension() {
    use entelix_core::ir::{GeminiExt, ProviderExtensions};
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-3-5-sonnet-20240620-v1:0".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_gemini(GeminiExt::default().with_candidate_count(2)),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let saw = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            ModelWarning::ProviderExtensionIgnored { vendor } if vendor == "gemini"
        )
    });
    assert!(
        saw,
        "expected ProviderExtensionIgnored gemini, got: {:?}",
        encoded.warnings
    );
}
