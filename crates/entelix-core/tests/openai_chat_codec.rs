//! `OpenAiChatCodec` encode/decode + streaming tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::too_many_lines)]

use bytes::Bytes;
use entelix_core::codecs::{BoxByteStream, Codec, OpenAiChatCodec};
use entelix_core::ir::{
    ContentPart, MediaSource, Message, ModelRequest, ModelWarning, Role, StopReason, ToolChoice,
    ToolResultContent, ToolSpec,
};
use entelix_core::stream::{StreamAggregator, StreamDelta};
use futures::StreamExt;
use serde_json::{Value, json};

fn parse(body: &[u8]) -> Value {
    serde_json::from_slice(body).unwrap()
}

// ── encode ─────────────────────────────────────────────────────────────────

#[test]
fn encode_minimal_request_emits_chat_completions_shape() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hello")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    assert_eq!(encoded.path, "/v1/chat/completions");
    let body = parse(&encoded.body);
    assert_eq!(body["model"], "gpt-4.1");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], "hello"); // string shorthand
    assert!(body.get("tools").is_none());
}

#[test]
fn encode_system_field_becomes_first_system_message() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        system: "Be concise.".into(),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "Be concise.");
    assert_eq!(body["messages"][1]["role"], "user");
}

#[test]
fn encode_assistant_tool_use_emits_tool_calls_array() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::new(
            Role::Assistant,
            vec![ContentPart::ToolUse {
                id: "call_1".into(),
                name: "double".into(),
                input: json!({"n": 21}),
                provider_echoes: Vec::new(),
            }],
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let msg = &body["messages"][0];
    assert_eq!(msg["role"], "assistant");
    assert!(msg["content"].is_null());
    let calls = msg["tool_calls"].as_array().unwrap();
    assert_eq!(calls[0]["id"], "call_1");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "double");
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["n"], 21);
}

#[test]
fn encode_tool_result_emits_tool_role_message() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                tool_use_id: "call_1".into(),
                name: "double".into(),
                content: ToolResultContent::Text("42".into()),
                is_error: false,
                cache_control: None,
                provider_echoes: Vec::new(),
            }],
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let msg = &body["messages"][0];
    assert_eq!(msg["role"], "tool");
    assert_eq!(msg["tool_call_id"], "call_1");
    assert_eq!(msg["content"], "42");
}

#[test]
fn encode_tools_and_tool_choice() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("calc")],
        tools: std::sync::Arc::from([ToolSpec::function(
            "double",
            "doubles n",
            json!({"type": "object"}),
        )]),
        tool_choice: ToolChoice::Required,
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["function"]["name"], "double");
    assert_eq!(body["tool_choice"], "required");
}

#[test]
fn encode_user_image_yields_array_content_with_image_url() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::new(
            Role::User,
            vec![
                ContentPart::text("look"),
                ContentPart::Image {
                    source: MediaSource::url("https://x/y.png"),
                    cache_control: None,
                    provider_echoes: Vec::new(),
                },
            ],
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let arr = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(arr[0]["type"], "text");
    assert_eq!(arr[1]["type"], "image_url");
    assert_eq!(arr[1]["image_url"]["url"], "https://x/y.png");
}

#[test]
fn encode_streaming_marks_request_and_includes_stream_options() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode_streaming(&req).unwrap();
    assert!(encoded.streaming);
    let body = parse(&encoded.body);
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
}

// ── decode ─────────────────────────────────────────────────────────────────

#[test]
fn decode_assistant_text_response() {
    let codec = OpenAiChatCodec::new();
    let body = json!({
        "id": "chatcmpl_1",
        "model": "gpt-4.1",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello, world!" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10 }
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert_eq!(response.id, "chatcmpl_1");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.input_tokens, 7);
    assert_eq!(response.usage.output_tokens, 3);
    assert!(matches!(
        response.content[0],
        ContentPart::Text { ref text, .. } if text == "Hello, world!"
    ));
}

#[test]
fn decode_assistant_tool_call_response() {
    let codec = OpenAiChatCodec::new();
    let body = json!({
        "id": "chatcmpl_2",
        "model": "gpt-4.1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_a",
                    "type": "function",
                    "function": { "name": "double", "arguments": "{\"n\":21}" }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    if let ContentPart::ToolUse {
        id, name, input, ..
    } = &response.content[0]
    {
        assert_eq!(id, "call_a");
        assert_eq!(name, "double");
        assert_eq!(input["n"], 21);
    } else {
        panic!("expected tool_use, got {:?}", response.content[0]);
    }
}

// ── streaming ──────────────────────────────────────────────────────────────

fn sse_chunks(payloads: &[Value]) -> Bytes {
    let mut out = String::new();
    for payload in payloads {
        out.push_str("data: ");
        out.push_str(&payload.to_string());
        out.push_str("\n\n");
    }
    out.push_str("data: [DONE]\n\n");
    Bytes::from(out)
}

fn body_from_bytes(b: Bytes) -> BoxByteStream<'static> {
    Box::pin(futures::stream::iter(vec![Ok::<_, entelix_core::Error>(b)]))
}

#[tokio::test]
async fn decode_stream_text_round_trips_through_aggregator() {
    let codec = OpenAiChatCodec::new();
    let bytes = sse_chunks(&[
        json!({
            "id": "chatcmpl_S1",
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" }
            }]
        }),
        json!({
            "id": "chatcmpl_S1",
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": { "content": "Hello, " }
            }]
        }),
        json!({
            "id": "chatcmpl_S1",
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": { "content": "world!" },
                "finish_reason": "stop"
            }]
        }),
        json!({
            "id": "chatcmpl_S1",
            "model": "gpt-4.1",
            "choices": [],
            "usage": { "prompt_tokens": 4, "completion_tokens": 5 }
        }),
    ]);
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut aggregator = StreamAggregator::new();
    while let Some(item) = stream.next().await {
        aggregator.push(item.unwrap()).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.id, "chatcmpl_S1");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.output_tokens, 5);
    if let ContentPart::Text { text, .. } = &response.content[0] {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("expected text part");
    }
}

#[tokio::test]
async fn decode_stream_tool_call_round_trips() {
    let codec = OpenAiChatCodec::new();
    let bytes = sse_chunks(&[
        json!({
            "id": "chatcmpl_T1",
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": { "name": "double", "arguments": "" }
                    }]
                }
            }]
        }),
        json!({
            "id": "chatcmpl_T1",
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": { "arguments": "{\"n\":" }
                    }]
                }
            }]
        }),
        json!({
            "id": "chatcmpl_T1",
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": { "arguments": "21}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
    ]);
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut deltas: Vec<StreamDelta> = Vec::new();
    while let Some(item) = stream.next().await {
        deltas.push(item.unwrap());
    }
    // start, tool_use_start, input_delta x 2, tool_use_stop (synthesized at [DONE]), stop
    let has_tool_start = deltas
        .iter()
        .any(|d| matches!(d, StreamDelta::ToolUseStart { name, .. } if name == "double"));
    let has_tool_stop = deltas.iter().any(|d| matches!(d, StreamDelta::ToolUseStop));
    assert!(has_tool_start);
    assert!(has_tool_stop);

    let mut aggregator = StreamAggregator::new();
    for d in deltas {
        aggregator.push(d).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    if let ContentPart::ToolUse { name, input, .. } = &response.content[0] {
        assert_eq!(name, "double");
        assert_eq!(input["n"], 21);
    } else {
        panic!("expected tool_use part");
    }
}

#[tokio::test]
async fn decode_stream_emits_warnings_in_first() {
    let codec = OpenAiChatCodec::new();
    let bytes = sse_chunks(&[json!({
        "id": "chatcmpl_W",
        "model": "gpt-4.1",
        "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
    })]);
    let warnings = vec![ModelWarning::LossyEncode {
        field: "test".into(),
        detail: "hi".into(),
    }];
    let mut stream = codec.decode_stream(body_from_bytes(bytes), warnings);
    let first = stream.next().await.unwrap().unwrap();
    assert!(matches!(first, StreamDelta::Warning(_)));
}

// ── ProviderExtensions wire-up ─────────────────────────────────────────────

#[test]
fn openai_chat_seed_and_end_user_id_thread_into_body() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        seed: Some(42),
        end_user_id: Some("op-3".into()),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["seed"], 42);
    assert_eq!(body["user"], "op-3");
}

#[test]
fn reasoning_effort_threads_into_body_on_o_series_models() {
    // OpenAI Chat Completions accepts `reasoning_effort` natively
    // on o-series + gpt-5 reasoning models.
    use entelix_core::ir::ReasoningEffort;
    let codec = OpenAiChatCodec::new();
    for model in ["o1-mini", "o3", "o4-mini", "gpt-5-pro"] {
        let req = ModelRequest {
            model: model.into(),
            messages: vec![Message::user("solve")],
            reasoning_effort: Some(ReasoningEffort::High),
            ..ModelRequest::default()
        };
        let encoded = codec.encode(&req).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
        assert_eq!(
            body["reasoning_effort"], "high",
            "{model} must emit native reasoning_effort"
        );
        assert!(
            encoded.warnings.iter().all(|w| !matches!(
                w,
                ModelWarning::LossyEncode { field, .. } if field == "reasoning_effort"
            )),
            "{model} must NOT emit LossyEncode for reasoning_effort"
        );
    }
}

#[test]
fn reasoning_effort_emits_lossy_encode_on_non_reasoning_models() {
    use entelix_core::ir::ReasoningEffort;
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        reasoning_effort: Some(ReasoningEffort::Medium),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    assert!(encoded.warnings.iter().any(|w| matches!(
        w,
        ModelWarning::LossyEncode { field, .. } if field == "reasoning_effort"
    )));
}

#[test]
fn service_tier_threads_into_body() {
    use entelix_core::ir::{OpenAiChatExt, ProviderExtensions, ServiceTier};
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_openai_chat(OpenAiChatExt::default().with_service_tier(ServiceTier::Flex)),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["service_tier"], "flex");
}

#[test]
fn openai_chat_codec_warns_on_foreign_vendor_extension() {
    use entelix_core::ir::{AnthropicExt, ProviderExtensions};
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_anthropic(AnthropicExt::default().with_betas(["thinking-2025"])),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let saw = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            ModelWarning::ProviderExtensionIgnored { vendor } if vendor == "anthropic"
        )
    });
    assert!(
        saw,
        "expected ProviderExtensionIgnored anthropic, got: {:?}",
        encoded.warnings
    );
}

#[test]
fn parallel_tool_calls_passes_through_natively_on_openai_chat() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        parallel_tool_calls: Some(false),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["parallel_tool_calls"], false);
}

#[test]
fn parallel_tool_calls_true_passes_through_natively_on_openai_chat() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        parallel_tool_calls: Some(true),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["parallel_tool_calls"], true);
}

#[test]
fn top_k_emits_lossy_encode_on_openai_chat() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        top_k: Some(40),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let saw = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            ModelWarning::LossyEncode { field, .. } if field == "top_k"
        )
    });
    assert!(saw, "OpenAI Chat must emit LossyEncode for top_k");
}
