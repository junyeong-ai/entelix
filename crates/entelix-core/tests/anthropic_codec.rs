//! `AnthropicMessagesCodec` encode/decode fixture tests.
//!
//! Each test pins a piece of the wire format. Catches accidental drift
//! before it reaches a real provider call.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use entelix_core::codecs::{AnthropicMessagesCodec, Codec};
use entelix_core::ir::{
    ContentPart, MediaSource, Message, ModelRequest, ModelWarning, Role, StopReason, ToolChoice,
    ToolResultContent, ToolSpec,
};
use serde_json::{Value, json};

fn parse(body: &[u8]) -> Value {
    serde_json::from_slice(body).unwrap()
}

// ── encode ─────────────────────────────────────────────────────────────────

#[test]
fn encode_minimal_request_emits_required_fields() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hello")],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();

    assert_eq!(encoded.method, http::Method::POST);
    assert_eq!(encoded.path, "/v1/messages");
    assert_eq!(
        encoded.headers.get("anthropic-version").unwrap(),
        "2023-06-01"
    );
    assert_eq!(
        encoded.headers.get(http::header::CONTENT_TYPE).unwrap(),
        "application/json"
    );

    let body = parse(&encoded.body);
    assert_eq!(body["model"], "claude-opus-4-7");
    assert_eq!(body["max_tokens"], 1024);
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"][0]["type"], "text");
    assert_eq!(body["messages"][0]["content"][0]["text"], "hello");
    assert!(body.get("system").is_none());
}

#[test]
fn encode_empty_messages_returns_invalid_request() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let err = codec.encode(&req).unwrap_err();
    assert!(matches!(err, entelix_core::Error::InvalidRequest(_)));
}

#[test]
fn encode_rejects_missing_max_tokens() {
    // Invariant #15 — Anthropic requires `max_tokens`; the codec
    // refuses to inject a silent default.
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let err = codec.encode(&req).unwrap_err();
    match err {
        entelix_core::Error::InvalidRequest(msg) => {
            assert!(
                msg.contains("max_tokens"),
                "missing max_tokens hint should mention the field; got: {msg}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn encode_top_level_system_passes_through() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        system: "Be terse.".into(),
        messages: vec![Message::user("hi")],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["system"], "Be terse.");
}

#[test]
fn encode_system_role_message_flattens_into_top_level() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        system: "Default system.".into(),
        messages: vec![
            Message::system("Extra rule one."),
            Message::system("Extra rule two."),
            Message::user("ok"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let system = body["system"].as_str().unwrap();
    assert!(system.contains("Default system."));
    assert!(system.contains("Extra rule one."));
    assert!(system.contains("Extra rule two."));
    // Only the user message remains in the messages array.
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    assert_eq!(body["messages"][0]["role"], "user");
}

#[test]
fn encode_tool_role_becomes_user_with_tool_result() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![
            Message::user("Use the calculator"),
            Message::new(
                Role::Assistant,
                vec![ContentPart::ToolUse {
                    id: "toolu_01".into(),
                    name: "calculator".into(),
                    input: json!({ "expr": "2+2" }),
                }],
            ),
            Message::new(
                Role::Tool,
                vec![ContentPart::ToolResult {
                    tool_use_id: "toolu_01".into(),
                    name: "calculator".into(),
                    content: ToolResultContent::Text("4".into()),
                    is_error: false,
                    cache_control: None,
                }],
            ),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);

    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][0]["type"], "tool_use");
    assert_eq!(messages[1]["content"][0]["id"], "toolu_01");
    // Tool role → user wrapper around tool_result block.
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"][0]["type"], "tool_result");
    assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_01");
    assert_eq!(messages[2]["content"][0]["content"], "4");
}

#[test]
fn encode_tool_result_json_payload_emits_warning_and_stringifies() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                tool_use_id: "x".into(),
                name: "calc".into(),
                content: ToolResultContent::Json(json!({ "value": 42 })),
                is_error: false,
                cache_control: None,
            }],
        )],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    // tool_result.content stringified.
    assert!(
        body["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap()
            .contains("\"value\":42")
    );
    // Warning emitted.
    assert!(
        encoded.warnings.iter().any(
            |w| matches!(w, ModelWarning::LossyEncode { detail, .. } if detail.contains("Json"))
        )
    );
}

#[test]
fn encode_image_url_and_base64_variants() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::new(
            Role::User,
            vec![
                ContentPart::Image {
                    source: MediaSource::url("https://example.com/x.png"),
                    cache_control: None,
                },
                ContentPart::Image {
                    source: MediaSource::Base64 {
                        media_type: "image/png".into(),
                        data: "iVBOR".into(),
                    },
                    cache_control: None,
                },
            ],
        )],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let imgs = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(imgs[0]["source"]["type"], "url");
    assert_eq!(imgs[0]["source"]["url"], "https://example.com/x.png");
    assert_eq!(imgs[1]["source"]["type"], "base64");
    assert_eq!(imgs[1]["source"]["media_type"], "image/png");
    assert_eq!(imgs[1]["source"]["data"], "iVBOR");
}

#[test]
fn encode_tools_with_choice_required() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("calc 2+2")],
        tools: vec![ToolSpec::function(
            "calculator",
            "Evaluates arithmetic",
            json!({
                "type": "object",
                "properties": { "expr": { "type": "string" } },
                "required": ["expr"],
            }),
        )],
        tool_choice: ToolChoice::Required,
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["tools"][0]["name"], "calculator");
    assert_eq!(body["tools"][0]["description"], "Evaluates arithmetic");
    assert_eq!(body["tool_choice"]["type"], "any");
}

#[test]
fn encode_tool_choice_specific_includes_name() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        tools: vec![ToolSpec::function("search", "", json!({}))],
        tool_choice: ToolChoice::Specific {
            name: "search".into(),
        },
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["tool_choice"]["type"], "tool");
    assert_eq!(body["tool_choice"]["name"], "search");
}

#[test]
fn encode_temperature_top_p_stop_sequences_pass_through() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        temperature: Some(0.7),
        top_p: Some(0.95),
        stop_sequences: vec!["###".into(), "END".into()],
        max_tokens: Some(2048),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    // f32 → f64 widening drops precision; compare within IEEE tolerance.
    let temp = body["temperature"].as_f64().unwrap();
    assert!((temp - 0.7).abs() < 1e-6, "temperature {temp}");
    let p = body["top_p"].as_f64().unwrap();
    assert!((p - 0.95).abs() < 1e-6, "top_p {p}");
    assert_eq!(body["stop_sequences"], json!(["###", "END"]));
    assert_eq!(body["max_tokens"], 2048);
}

// ── decode ─────────────────────────────────────────────────────────────────

#[test]
fn decode_text_response_with_end_turn() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_01ABC",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7-20260415",
        "content": [
            { "type": "text", "text": "Hello back!" }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4
        }
    });
    let resp = codec
        .decode(&serde_json::to_vec(&wire).unwrap(), Vec::new())
        .unwrap();

    assert_eq!(resp.id, "msg_01ABC");
    assert_eq!(resp.model, "claude-opus-4-7-20260415");
    assert!(matches!(resp.stop_reason, StopReason::EndTurn));
    assert_eq!(resp.content.len(), 1);
    match &resp.content[0] {
        ContentPart::Text { text, .. } => assert_eq!(text, "Hello back!"),
        other => panic!("expected text content, got {other:?}"),
    }
    assert_eq!(resp.usage.input_tokens, 12);
    assert_eq!(resp.usage.output_tokens, 4);
    assert!(resp.warnings.is_empty());
}

#[test]
fn decode_tool_use_response() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_02",
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_99",
                "name": "calculator",
                "input": { "expr": "2+2" }
            }
        ],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 9, "output_tokens": 8 }
    });
    let resp = codec
        .decode(&serde_json::to_vec(&wire).unwrap(), Vec::new())
        .unwrap();

    assert!(matches!(resp.stop_reason, StopReason::ToolUse));
    assert_eq!(resp.content.len(), 1);
    match &resp.content[0] {
        ContentPart::ToolUse { id, name, input } => {
            assert_eq!(id, "toolu_99");
            assert_eq!(name, "calculator");
            assert_eq!(input["expr"], "2+2");
        }
        other => panic!("expected tool_use, got {other:?}"),
    }
}

#[test]
fn decode_stop_sequence_carries_matched_string() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_03",
        "model": "x",
        "content": [{ "type": "text", "text": "halted" }],
        "stop_reason": "stop_sequence",
        "stop_sequence": "###",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec
        .decode(&serde_json::to_vec(&wire).unwrap(), Vec::new())
        .unwrap();
    match resp.stop_reason {
        StopReason::StopSequence { sequence } => assert_eq!(sequence, "###"),
        other => panic!("unexpected stop_reason: {other:?}"),
    }
}

#[test]
fn decode_unknown_stop_reason_emits_warning_and_other_variant() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_04",
        "model": "x",
        "content": [],
        "stop_reason": "future_filter",
        "usage": { "input_tokens": 1, "output_tokens": 0 }
    });
    let resp = codec
        .decode(&serde_json::to_vec(&wire).unwrap(), Vec::new())
        .unwrap();
    assert!(matches!(
        resp.stop_reason,
        StopReason::Other { ref raw } if raw == "future_filter"
    ));
    assert!(resp.warnings.iter().any(|w| matches!(
        w,
        ModelWarning::UnknownStopReason { raw } if raw == "future_filter"
    )));
}

#[test]
fn decode_unknown_content_block_emits_warning() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_05",
        "model": "x",
        "content": [
            { "type": "text", "text": "ok" },
            { "type": "future_block", "data": "..." }
        ],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let resp = codec
        .decode(&serde_json::to_vec(&wire).unwrap(), Vec::new())
        .unwrap();
    assert_eq!(resp.content.len(), 1); // future_block dropped
    assert!(resp.warnings.iter().any(|w| matches!(
        w,
        ModelWarning::LossyEncode { detail, .. } if detail.contains("future_block")
    )));
}

#[test]
fn decode_cache_token_fields() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_06",
        "model": "x",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 80
        }
    });
    let resp = codec
        .decode(&serde_json::to_vec(&wire).unwrap(), Vec::new())
        .unwrap();
    assert_eq!(resp.usage.input_tokens, 100);
    assert_eq!(resp.usage.output_tokens, 50);
    assert_eq!(resp.usage.cache_creation_input_tokens, 20);
    assert_eq!(resp.usage.cached_input_tokens, 80);
    assert_eq!(resp.usage.billable_input(), 120);
}

#[test]
fn decode_carries_encode_warnings_forward() {
    let codec = AnthropicMessagesCodec::new();
    let wire = json!({
        "id": "msg_07",
        "model": "x",
        "content": [],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 0, "output_tokens": 0 }
    });
    let prior = vec![ModelWarning::LossyEncode {
        field: "earlier".into(),
        detail: "from encode".into(),
    }];
    let resp = codec
        .decode(&serde_json::to_vec(&wire).unwrap(), prior)
        .unwrap();
    assert_eq!(resp.warnings.len(), 1);
    assert!(matches!(
        &resp.warnings[0],
        ModelWarning::LossyEncode { field, .. } if field == "earlier"
    ));
}

#[test]
fn capabilities_are_full_featured() {
    let codec = AnthropicMessagesCodec::new();
    let caps = codec.capabilities("claude-opus-4-7");
    assert!(caps.streaming);
    assert!(caps.tools);
    assert!(caps.multimodal_image);
    assert!(caps.multimodal_document);
    assert!(caps.thinking);
    assert!(caps.web_search);
    assert!(caps.computer_use);
    assert!(caps.system_prompt);
    assert!(caps.prompt_caching);
    assert_eq!(caps.max_context_tokens, 200_000);
}

#[test]
fn codec_name_is_stable() {
    let codec = AnthropicMessagesCodec::new();
    assert_eq!(codec.name(), "anthropic-messages");
}

#[test]
fn anthropic_ext_disable_parallel_tool_use_threads_into_tool_choice() {
    use entelix_core::ir::{AnthropicExt, ProviderExtensions};
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("call my tool")],
        tools: vec![ToolSpec::function(
            "calc",
            "arithmetic",
            json!({"type": "object"}),
        )],
        provider_extensions: ProviderExtensions::default()
            .with_anthropic(AnthropicExt::default().with_disable_parallel_tool_use(true)),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(
        body["tool_choice"]["disable_parallel_tool_use"], true,
        "disable_parallel_tool_use must thread into tool_choice"
    );
}

#[test]
fn anthropic_ext_user_id_threads_into_metadata() {
    use entelix_core::ir::{AnthropicExt, ProviderExtensions};
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_anthropic(AnthropicExt::default().with_user_id("op-7")),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["metadata"]["user_id"], "op-7");
}

#[test]
fn anthropic_thinking_emits_top_level_thinking_object_for_explicit_budget_models() {
    use entelix_core::ir::ReasoningEffort;
    // Sonnet accepts either adaptive or explicit budget; on
    // `Medium` the codec emits `{enabled, budget_tokens: 4096}` per
    // Opus 4.7 maps the same effort onto an adaptive
    // object — that path is exercised by the dedicated
    // `anthropic_opus_4_7_*_adaptive_*` tests.
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-sonnet-4-6".into(),
        messages: vec![Message::user("solve")],
        reasoning_effort: Some(ReasoningEffort::Medium),
        max_tokens: Some(8192),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["thinking"]["type"], "enabled");
    assert_eq!(body["thinking"]["budget_tokens"], 4096);
}

#[test]
fn anthropic_codec_warns_on_foreign_vendor_extension() {
    use entelix_core::ir::{GeminiExt, ProviderExtensions};
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_gemini(GeminiExt::default().with_candidate_count(2)),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let saw_warning = encoded.warnings.iter().any(|w| {
        matches!(
            w,
            ModelWarning::ProviderExtensionIgnored { vendor } if vendor == "gemini"
        )
    });
    assert!(
        saw_warning,
        "expected ProviderExtensionIgnored {{vendor: gemini}}, got: {:?}",
        encoded.warnings
    );
}

#[test]
fn decode_drops_document_citation_when_index_is_missing() {
    // Invariant #15 — `document_index` is the citation's
    // load-bearing pointer. Missing index used to silently default
    // to 0 (i.e. the wrong document). Post-fix, the citation is
    // dropped entirely so operators see absence rather than a
    // misleading reference. The text content survives — only the
    // unparseable citation block is filtered.
    let codec = AnthropicMessagesCodec::new();
    let body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": "from a document",
                "citations": [
                    // citation entry missing the load-bearing
                    // `document_index` field — must NOT decode as
                    // CitationSource::Document { document_index: 0, … }.
                    {
                        "type": "page_location",
                        "document_title": "report.pdf"
                        // document_index intentionally omitted
                    }
                ]
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let response = codec
        .decode(serde_json::to_vec(&body).unwrap().as_slice(), Vec::new())
        .unwrap();
    let citation_decoded = response
        .content
        .iter()
        .any(|c| matches!(c, ContentPart::Citation { .. }));
    assert!(
        !citation_decoded,
        "citation with missing document_index must be dropped, \
         not silently coerced to document_index=0; got: {:?}",
        response.content
    );
    // The text body itself still survives — only the malformed
    // citation entry is filtered.
    let text_decoded = response
        .content
        .iter()
        .any(|c| matches!(c, ContentPart::Text { text, .. } if text == "from a document"));
    assert!(text_decoded, "text body must survive citation filtering");
}

#[test]
fn decode_keeps_document_citation_when_index_is_present() {
    // Companion to the missing-index test — confirms the
    // happy-path citation still decodes when the vendor supplies
    // the document_index correctly. Guards against a regression
    // that over-broadly drops citations.
    use entelix_core::ir::CitationSource;
    let codec = AnthropicMessagesCodec::new();
    let body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": "from a document",
                "citations": [
                    {
                        "type": "page_location",
                        "document_title": "report.pdf",
                        "document_index": 7
                    }
                ]
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let response = codec
        .decode(serde_json::to_vec(&body).unwrap().as_slice(), Vec::new())
        .unwrap();
    let saw_doc_index_seven = response.content.iter().any(|c| {
        matches!(
            c,
            ContentPart::Citation { source: CitationSource::Document { document_index, .. }, .. }
                if *document_index == 7
        )
    });
    assert!(
        saw_doc_index_seven,
        "well-formed page_location citation with document_index=7 must decode; \
         got: {:?}",
        response.content
    );
}

#[test]
fn anthropic_ext_betas_emit_comma_joined_header() {
    use entelix_core::ir::{AnthropicExt, ProviderExtensions};
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default().with_anthropic(
            AnthropicExt::default()
                .with_betas(["prompt-caching-2024-07-31", "computer-use-2025-01-24"]),
        ),
        max_tokens: Some(64),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let beta = encoded
        .headers
        .get("anthropic-beta")
        .expect("anthropic-beta header must be present");
    assert_eq!(
        beta.to_str().unwrap(),
        "prompt-caching-2024-07-31,computer-use-2025-01-24"
    );
}

#[test]
fn anthropic_ext_betas_empty_emits_no_header() {
    use entelix_core::ir::{AnthropicExt, ProviderExtensions};
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_anthropic(AnthropicExt::default().with_user_id("op-3")),
        max_tokens: Some(64),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    assert!(
        !encoded.headers.contains_key("anthropic-beta"),
        "no betas configured ⇒ no anthropic-beta header"
    );
}

#[test]
fn anthropic_ext_betas_streaming_path_carries_header() {
    use entelix_core::ir::{AnthropicExt, ProviderExtensions};
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default().with_anthropic(
            AnthropicExt::default().with_betas(["interleaved-thinking-2025-05-14"]),
        ),
        max_tokens: Some(64),
        ..ModelRequest::default()
    };
    let encoded = codec.encode_streaming(&req).unwrap();
    assert!(encoded.streaming);
    assert_eq!(
        encoded
            .headers
            .get("anthropic-beta")
            .unwrap()
            .to_str()
            .unwrap(),
        "interleaved-thinking-2025-05-14",
    );
    assert_eq!(
        encoded
            .headers
            .get(http::header::ACCEPT)
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream",
    );
}
