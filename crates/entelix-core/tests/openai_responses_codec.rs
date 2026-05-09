//! `OpenAiResponsesCodec` encode/decode + streaming tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::too_many_lines)]

use bytes::Bytes;
use entelix_core::codecs::{BoxByteStream, Codec, OpenAiResponsesCodec};
use entelix_core::ir::{
    ContentPart, Message, ModelRequest, ModelWarning, Role, StopReason, ToolChoice,
    ToolResultContent, ToolSpec,
};
use entelix_core::stream::{StreamAggregator, StreamDelta};
use futures::StreamExt;
use serde_json::{Value, json};

fn parse(body: &[u8]) -> Value {
    serde_json::from_slice(body).unwrap()
}

#[test]
fn encode_minimal_request_emits_responses_shape() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    assert_eq!(encoded.path, "/v1/responses");
    let body = parse(&encoded.body);
    assert_eq!(body["model"], "gpt-4.1");
    assert_eq!(body["input"][0]["type"], "message");
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][0]["text"], "hi");
}

#[test]
fn encode_system_routes_into_instructions() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        system: "Be concise.".into(),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["instructions"], "Be concise.");
    // No system role in input array.
    let inputs = body["input"].as_array().unwrap();
    for input in inputs {
        if input["type"] == "message" {
            assert_ne!(input["role"], "system");
        }
    }
}

#[test]
fn encode_assistant_tool_use_emits_separate_function_call_item() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::new(
            Role::Assistant,
            vec![
                ContentPart::text("let me compute"),
                ContentPart::ToolUse {
                    id: "call_1".into(),
                    name: "double".into(),
                    input: json!({"n": 21}),
                },
            ],
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let items = body["input"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["type"], "message");
    assert_eq!(items[0]["role"], "assistant");
    assert_eq!(items[0]["content"][0]["type"], "output_text");
    assert_eq!(items[1]["type"], "function_call");
    assert_eq!(items[1]["call_id"], "call_1");
    assert_eq!(items[1]["name"], "double");
}

#[test]
fn encode_tool_result_emits_function_call_output_item() {
    let codec = OpenAiResponsesCodec::new();
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
            }],
        )],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let item = &body["input"][0];
    assert_eq!(item["type"], "function_call_output");
    assert_eq!(item["call_id"], "call_1");
    assert_eq!(item["output"], "42");
}

#[test]
fn encode_tools_emits_top_level_function_array() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
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
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "double");
    assert_eq!(body["tool_choice"], "required");
}

#[test]
fn encode_streaming_marks_request() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode_streaming(&req).unwrap();
    assert!(encoded.streaming);
    let body = parse(&encoded.body);
    assert_eq!(body["stream"], true);
}

// ── decode ─────────────────────────────────────────────────────────────────

#[test]
fn decode_text_response() {
    let codec = OpenAiResponsesCodec::new();
    let body = json!({
        "id": "resp_1",
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "Hello!" }]
        }],
        "usage": { "input_tokens": 4, "output_tokens": 1 }
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(matches!(
        response.content[0],
        ContentPart::Text { ref text, .. } if text == "Hello!"
    ));
}

#[test]
fn decode_tool_use_response() {
    let codec = OpenAiResponsesCodec::new();
    let body = json!({
        "id": "resp_2",
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "call_id": "call_a",
            "name": "double",
            "arguments": "{\"n\":21}"
        }]
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    if let ContentPart::ToolUse { id, name, input } = &response.content[0] {
        assert_eq!(id, "call_a");
        assert_eq!(name, "double");
        assert_eq!(input["n"], 21);
    } else {
        panic!("expected tool_use");
    }
}

// ── streaming ──────────────────────────────────────────────────────────────

fn sse_chunks(events: &[(&str, Value)]) -> Bytes {
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

#[tokio::test]
async fn decode_stream_text_round_trips() {
    let codec = OpenAiResponsesCodec::new();
    let bytes = sse_chunks(&[
        (
            "response.created",
            json!({
                "type": "response.created",
                "response": { "id": "resp_S1", "model": "gpt-4.1" }
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
                    "id": "resp_S1",
                    "status": "completed",
                    "usage": { "input_tokens": 2, "output_tokens": 4 },
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello, world!"}]
                    }]
                }
            }),
        ),
    ]);
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut aggregator = StreamAggregator::new();
    while let Some(item) = stream.next().await {
        aggregator.push(item.unwrap()).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.id, "resp_S1");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.output_tokens, 4);
    if let ContentPart::Text { text, .. } = &response.content[0] {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("expected text");
    }
}

#[tokio::test]
async fn decode_stream_tool_call_round_trips() {
    let codec = OpenAiResponsesCodec::new();
    let bytes = sse_chunks(&[
        (
            "response.created",
            json!({
                "type": "response.created",
                "response": { "id": "resp_T1", "model": "gpt-4.1" }
            }),
        ),
        (
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "double",
                    "arguments": ""
                }
            }),
        ),
        (
            "response.function_call_arguments.delta",
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": "{\"n\":"
            }),
        ),
        (
            "response.function_call_arguments.delta",
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": "21}"
            }),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "item": { "type": "function_call" }
            }),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_T1",
                    "status": "completed",
                    "output": [{
                        "type": "function_call",
                        "call_id": "call_a",
                        "name": "double",
                        "arguments": "{\"n\":21}"
                    }]
                }
            }),
        ),
    ]);
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut deltas: Vec<StreamDelta> = Vec::new();
    while let Some(item) = stream.next().await {
        deltas.push(item.unwrap());
    }
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
        panic!("expected tool_use");
    }
}

// ── ProviderExtensions wire-up ─────────────────────────────────────────────

#[test]
fn openai_responses_ext_seed_and_user_thread_into_body() {
    use entelix_core::ir::{OpenAiResponsesExt, ProviderExtensions};
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-5".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default().with_openai_responses(
            OpenAiResponsesExt::default()
                .with_seed(99)
                .with_user("op-9"),
        ),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["seed"], 99);
    assert_eq!(body["user"], "op-9");
}

#[test]
fn openai_responses_ext_reasoning_effort_emits_top_level_reasoning_object() {
    use entelix_core::ir::{
        OpenAiResponsesExt, ProviderExtensions, ReasoningEffort, ReasoningSummary,
    };
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "o3".into(),
        messages: vec![Message::user("solve")],
        reasoning_effort: Some(ReasoningEffort::High),
        provider_extensions: ProviderExtensions::default().with_openai_responses(
            OpenAiResponsesExt::default().with_reasoning_summary(ReasoningSummary::Detailed),
        ),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["reasoning"]["effort"], "high");
    assert_eq!(body["reasoning"]["summary"], "detailed");
}

#[test]
fn openai_responses_ext_reasoning_summary_optional() {
    use entelix_core::ir::ReasoningEffort;
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "o3".into(),
        messages: vec![Message::user("solve")],
        reasoning_effort: Some(ReasoningEffort::Low),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["reasoning"]["effort"], "low");
    assert!(body["reasoning"].get("summary").is_none());
}

#[test]
fn openai_responses_codec_warns_on_foreign_vendor_extension() {
    use entelix_core::ir::{BedrockExt, BedrockGuardrail, ProviderExtensions};
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-5".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default().with_bedrock(
            BedrockExt::default().with_guardrail(BedrockGuardrail {
                identifier: "abc".into(),
                version: "1".into(),
            }),
        ),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let saw = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            ModelWarning::ProviderExtensionIgnored { vendor } if vendor == "bedrock"
        )
    });
    assert!(
        saw,
        "expected ProviderExtensionIgnored bedrock, got: {:?}",
        encoded.warnings
    );
}

#[test]
fn parallel_tool_calls_passes_through_natively_on_openai_responses() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-5".into(),
        messages: vec![Message::user("hi")],
        parallel_tool_calls: Some(false),
        ..ModelRequest::default()
    };
    let body =
        serde_json::from_slice::<serde_json::Value>(&codec.encode(&req).unwrap().body).unwrap();
    assert_eq!(body["parallel_tool_calls"], false);
}

#[test]
fn top_k_emits_lossy_encode_on_openai_responses() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-5".into(),
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
    assert!(saw, "OpenAI Responses must emit LossyEncode for top_k");
}
