//! `GeminiCodec` encode/decode + streaming tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::too_many_lines)]

use bytes::Bytes;
use entelix_core::codecs::{BoxByteStream, Codec, GeminiCodec};
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
fn encode_minimal_request_emits_contents_array() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    assert_eq!(
        encoded.path,
        "/v1beta/models/gemini-2.0-flash:generateContent"
    );
    let body = parse(&encoded.body);
    assert_eq!(body["contents"][0]["role"], "user");
    assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
}

#[test]
fn encode_system_routes_into_system_instruction() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message::user("hi")],
        system: "Be terse.".into(),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be terse.");
    // contents must NOT contain a system role.
    let contents = body["contents"].as_array().unwrap();
    for c in contents {
        assert_ne!(c["role"], "system");
    }
}

#[test]
fn encode_assistant_uses_model_role() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message::assistant("hello")],
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["contents"][0]["role"], "model");
}

#[test]
fn encode_tool_use_emits_function_call_with_reconstructible_id() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message::new(
            Role::Assistant,
            vec![ContentPart::ToolUse {
                id: "double#0".into(),
                name: "double".into(),
                input: json!({"n": 21}),
            }],
        )],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let part = &body["contents"][0]["parts"][0];
    assert_eq!(part["functionCall"]["name"], "double");
    assert_eq!(part["functionCall"]["args"]["n"], 21);
    // The id reconstruction (`name#idx`) means no LossyEncode is needed —
    // the id is recoverable on decode without a vendor channel.
    let lossy_id = encoded.warnings.iter().any(|w| match w {
        ModelWarning::LossyEncode { field, .. } => field.contains("id"),
        _ => false,
    });
    assert!(
        !lossy_id,
        "tool_use id is reconstructible on decode; LossyEncode must not fire"
    );
}

#[test]
fn encode_tool_result_emits_function_response_with_real_name() {
    // The IR's `ContentPart::ToolResult.name` carries the original
    // function name, so Gemini's `functionResponse` keys correctly
    // and the next turn correlates against the right tool. No
    // placeholder, no `LossyEncode` warning for the name field.
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                tool_use_id: "call_1".into(),
                name: "double".into(),
                content: ToolResultContent::Json(json!({"doubled": 42})),
                is_error: false,
                cache_control: None,
            }],
        )],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let part = &body["contents"][0]["parts"][0];
    assert_eq!(part["functionResponse"]["name"], "double");
    assert_eq!(part["functionResponse"]["response"]["doubled"], 42);
    // No name-placeholder warning because the IR carried the real
    // name. (An is_error warning is still possible if the result
    // was flagged as an error — see separate test.)
    let name_warning = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            entelix_core::ir::ModelWarning::LossyEncode { detail, .. }
            if detail.contains("functionResponse needs a name")
        )
    });
    assert!(
        !name_warning,
        "name placeholder warning should not fire when IR carries the real name"
    );
}

#[test]
fn encode_tools_emits_function_declarations() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.0-flash".into(),
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
    let decls = &body["tools"][0]["functionDeclarations"];
    assert_eq!(decls[0]["name"], "double");
    assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
}

#[test]
fn encode_streaming_path_uses_stream_endpoint() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let encoded = codec.encode_streaming(&req).unwrap();
    assert!(encoded.streaming);
    assert!(encoded.path.contains(":streamGenerateContent"));
    assert!(encoded.path.contains("alt=sse"));
}

// ── decode ─────────────────────────────────────────────────────────────────

#[test]
fn decode_text_response() {
    let codec = GeminiCodec::new();
    let body = json!({
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{
            "content": { "role": "model", "parts": [{ "text": "Hello!" }] },
            "finishReason": "STOP"
        }],
        "usageMetadata": { "promptTokenCount": 4, "candidatesTokenCount": 2 }
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.input_tokens, 4);
    assert_eq!(response.usage.output_tokens, 2);
    assert!(matches!(response.content[0], ContentPart::Text { ref text, .. } if text == "Hello!"));
}

#[test]
fn decode_function_call_reconstructs_tool_use_id_from_name_and_index() {
    let codec = GeminiCodec::new();
    let body = json!({
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{
            "content": { "role": "model", "parts": [{
                "functionCall": { "name": "double", "args": { "n": 21 } }
            }]},
            "finishReason": "STOP"
        }]
    });
    let response = codec
        .decode(body.to_string().as_bytes(), Vec::new())
        .unwrap();
    if let ContentPart::ToolUse { id, name, input } = &response.content[0] {
        // Reconstruction is `name#idx` so the next turn's encode round-trips
        // bit-for-bit. The synth depends on tool name + position only — no
        // randomness, no global counter.
        assert_eq!(id, "double#0");
        assert_eq!(name, "double");
        assert_eq!(input["n"], 21);
    } else {
        panic!("expected tool_use");
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
    Bytes::from(out)
}

fn body_from_bytes(b: Bytes) -> BoxByteStream<'static> {
    Box::pin(futures::stream::iter(vec![Ok::<_, entelix_core::Error>(b)]))
}

#[tokio::test]
async fn decode_stream_text_round_trips() {
    let codec = GeminiCodec::new();
    let bytes = sse_chunks(&[
        json!({
            "modelVersion": "gemini-2.0-flash",
            "candidates": [{ "content": { "role": "model", "parts": [{"text": "Hello, "}] } }]
        }),
        json!({
            "modelVersion": "gemini-2.0-flash",
            "candidates": [{
                "content": { "role": "model", "parts": [{"text": "world!"}] },
                "finishReason": "STOP"
            }],
            "usageMetadata": { "promptTokenCount": 2, "candidatesTokenCount": 4 }
        }),
    ]);
    let mut stream = codec.decode_stream(body_from_bytes(bytes), Vec::new());
    let mut aggregator = StreamAggregator::new();
    while let Some(item) = stream.next().await {
        aggregator.push(item.unwrap()).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.output_tokens, 4);
    if let ContentPart::Text { text, .. } = &response.content[0] {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("expected text part");
    }
}

#[tokio::test]
async fn decode_stream_function_call_round_trips() {
    let codec = GeminiCodec::new();
    let bytes = sse_chunks(&[json!({
        "modelVersion": "gemini-2.0-flash",
        "candidates": [{
            "content": { "role": "model", "parts": [{
                "functionCall": { "name": "double", "args": { "n": 21 } }
            }]},
            "finishReason": "STOP"
        }]
    })]);
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
    if let ContentPart::ToolUse { name, input, .. } = &response.content[0] {
        assert_eq!(name, "double");
        assert_eq!(input["n"], 21);
    } else {
        panic!("expected tool_use");
    }
}

// ── ProviderExtensions wire-up ─────────────────────────────────────────────

#[test]
fn gemini_ext_safety_settings_thread_into_top_level() {
    use entelix_core::ir::{GeminiExt, ProviderExtensions};
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default().with_gemini(
            GeminiExt::default()
                .with_safety_override("HARM_CATEGORY_HATE_SPEECH", "BLOCK_LOW_AND_ABOVE")
                .with_safety_override("HARM_CATEGORY_HARASSMENT", "BLOCK_NONE"),
        ),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(
        body["safetySettings"][0]["category"],
        "HARM_CATEGORY_HATE_SPEECH"
    );
    assert_eq!(
        body["safetySettings"][0]["threshold"],
        "BLOCK_LOW_AND_ABOVE"
    );
    assert_eq!(
        body["safetySettings"][1]["category"],
        "HARM_CATEGORY_HARASSMENT"
    );
}

#[test]
fn gemini_ext_candidate_count_threads_into_generation_config() {
    use entelix_core::ir::{GeminiExt, ProviderExtensions};
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        temperature: Some(0.5),
        provider_extensions: ProviderExtensions::default()
            .with_gemini(GeminiExt::default().with_candidate_count(3)),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["generationConfig"]["candidateCount"], 3);
    // existing generation_config keys must survive
    assert_eq!(body["generationConfig"]["temperature"], 0.5);
}

#[test]
fn gemini_ext_candidate_count_creates_generation_config_when_missing() {
    use entelix_core::ir::{GeminiExt, ProviderExtensions};
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_gemini(GeminiExt::default().with_candidate_count(2)),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["generationConfig"]["candidateCount"], 2);
}

#[test]
fn gemini_codec_warns_on_foreign_vendor_extension() {
    use entelix_core::ir::{OpenAiChatExt, ProviderExtensions};
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_openai_chat(OpenAiChatExt::default().with_seed(7)),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let saw = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            ModelWarning::ProviderExtensionIgnored { vendor } if vendor == "openai_chat"
        )
    });
    assert!(
        saw,
        "expected ProviderExtensionIgnored openai_chat, got: {:?}",
        encoded.warnings
    );
}

#[test]
fn top_k_passes_through_natively_on_gemini() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        top_k: Some(40),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["generationConfig"]["topK"], 40);
    assert!(
        encoded.warnings.iter().all(|w| !matches!(
            w,
            ModelWarning::LossyEncode { field, .. } if field == "top_k"
        )),
        "Gemini native topK must NOT emit LossyEncode"
    );
}

#[test]
fn parallel_tool_calls_emits_lossy_encode_on_gemini() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        parallel_tool_calls: Some(true),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let saw = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            ModelWarning::LossyEncode { field, .. } if field == "parallel_tool_calls"
        )
    });
    assert!(saw, "Gemini must emit LossyEncode for parallel_tool_calls");
}
