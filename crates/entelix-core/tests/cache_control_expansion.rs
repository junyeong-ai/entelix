//! `CacheControl` IR expansion coverage — per-block, per-tool, plus
//! the per-vendor cache routing knobs that ride on `*Ext`.
//!
//! Verifies that:
//!
//! - Anthropic per-`ContentPart` cache_control surfaces as
//!   `cache_control: { type: "<ttl>" }` on the matching wire block.
//! - Anthropic `ToolSpec.cache_control` attaches to the tool
//!   declaration.
//! - `OpenAiChatExt::cache_key` / `OpenAiResponsesExt::cache_key`
//!   emit as `prompt_cache_key`.
//! - `GeminiExt::cached_content` emits as `cachedContent`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix_core::ir::{
    CacheControl, ContentPart, GeminiExt, MediaSource, Message, ModelRequest, ModelWarning,
    OpenAiChatExt, OpenAiResponsesExt, ProviderExtensions, Role, ToolSpec,
};
use serde_json::Value;

fn parse(body: &bytes::Bytes) -> Value {
    serde_json::from_slice(body).expect("body must be JSON")
}

fn lossy_for(warnings: &[ModelWarning], field: &str) -> bool {
    warnings.iter().any(|w| match w {
        ModelWarning::LossyEncode { field: f, .. } => f == field,
        _ => false,
    })
}

#[test]
fn anthropic_per_block_cache_control_emits_directive_on_wire() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::new(
            Role::User,
            vec![
                ContentPart::text("uncached input"),
                ContentPart::Image {
                    source: MediaSource::base64("image/png", "AAA"),
                    cache_control: Some(CacheControl::one_hour()),
                    provider_echoes: Vec::new(),
                },
            ],
        )],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let blocks = body["messages"][0]["content"].as_array().unwrap();
    // Block 0 — uncached text — must NOT carry cache_control.
    assert!(blocks[0].get("cache_control").is_none());
    // Block 1 — image with 1h cache — Anthropic's wire is ALWAYS
    // `{type: "ephemeral", ttl: "1h"}`. The TTL never rides in the
    // `type` field (regression-tests prior wire-format bug).
    assert_eq!(blocks[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(blocks[1]["cache_control"]["ttl"], "1h");
}

#[test]
fn anthropic_five_minute_cache_omits_ttl_field() {
    // Default 5m tier renders without an explicit `ttl` sibling —
    // matches Anthropic's documented wire shape and saves bytes on
    // the hot path. Regression-tests the `wire_ttl_field() == None`
    // branch end-to-end.
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::new(
            Role::User,
            vec![ContentPart::Image {
                source: MediaSource::base64("image/png", "AAA"),
                cache_control: Some(CacheControl::five_minutes()),
                provider_echoes: Vec::new(),
            }],
        )],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    let block = &body["messages"][0]["content"][0];
    assert_eq!(block["cache_control"]["type"], "ephemeral");
    assert!(
        block["cache_control"].get("ttl").is_none(),
        "5m default must not emit a `ttl` sibling on the wire"
    );
}

#[test]
fn anthropic_tool_spec_cache_control_attaches_to_tool_declaration() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        tools: std::sync::Arc::from([ToolSpec::function(
            "stable",
            "stable tool — cache the declaration",
            serde_json::json!({"type": "object"}),
        )
        .with_cache_control(CacheControl::five_minutes())]),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn openai_chat_cache_key_emits_prompt_cache_key() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_openai_chat(OpenAiChatExt::default().with_cache_key("user-42")),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["prompt_cache_key"], "user-42");
}

#[test]
fn openai_responses_cache_key_emits_prompt_cache_key() {
    let codec = OpenAiResponsesCodec::new();
    let req = ModelRequest {
        model: "gpt-5".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_openai_responses(OpenAiResponsesExt::default().with_cache_key("session-9")),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["prompt_cache_key"], "session-9");
}

#[test]
fn gemini_cached_content_emits_native_field() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_gemini(GeminiExt::default().with_cached_content("cachedContents/abc123")),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["cachedContent"], "cachedContents/abc123");
}

#[test]
fn openai_chat_ext_on_anthropic_request_emits_provider_extension_ignored() {
    // Cross-vendor ext placement (operator set OpenAI-only knob on
    // a request routed to AnthropicMessagesCodec) surfaces through
    // the typed `ProviderExtensionIgnored` channel — invariant 6.
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        max_tokens: Some(1024),
        provider_extensions: ProviderExtensions::default()
            .with_openai_chat(OpenAiChatExt::default().with_cache_key("ignored")),
        ..ModelRequest::default()
    };
    let warnings = codec.encode(&req).unwrap().warnings;
    assert!(warnings.iter().any(|w| matches!(
        w,
        ModelWarning::ProviderExtensionIgnored { vendor } if vendor == "openai_chat"
    )));
}

#[test]
fn gemini_ext_on_openai_request_emits_provider_extension_ignored() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        provider_extensions: ProviderExtensions::default()
            .with_gemini(GeminiExt::default().with_cached_content("cachedContents/abc")),
        ..ModelRequest::default()
    };
    let warnings = codec.encode(&req).unwrap().warnings;
    assert!(warnings.iter().any(|w| matches!(
        w,
        ModelWarning::ProviderExtensionIgnored { vendor } if vendor == "gemini"
    )));
}

#[test]
fn cache_control_helpers_round_trip_on_content_parts() {
    // The IR `with_cache_control` helper preserves all other fields.
    let part = ContentPart::text("hello").with_cache_control(CacheControl::one_hour());
    if let ContentPart::Text {
        text,
        cache_control,
        ..
    } = part
    {
        assert_eq!(text, "hello");
        assert_eq!(
            cache_control.unwrap().ttl,
            entelix_core::ir::CacheTtl::OneHour
        );
    } else {
        panic!("expected Text part");
    }
}

#[test]
fn cache_control_helper_is_noop_on_tool_use() {
    let original = ContentPart::ToolUse {
        id: "x".into(),
        name: "y".into(),
        input: serde_json::json!({}),
        provider_echoes: Vec::new(),
    };
    let after = original
        .clone()
        .with_cache_control(CacheControl::five_minutes());
    assert_eq!(original, after);
}

#[test]
fn bedrock_user_message_cache_control_emits_cache_point() {
    // Bedrock Converse `cachePoint: { type: "default" }` is the
    // first-class native caching channel (added Dec 2024). The marker
    // goes AFTER the content block it caches.
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-v1:0".into(),
        messages: vec![Message::new(
            Role::User,
            vec![
                ContentPart::text("uncached prefix"),
                ContentPart::Text {
                    text: "cached prefix".into(),
                    cache_control: Some(CacheControl::five_minutes()),
                    provider_echoes: Vec::new(),
                },
                ContentPart::text("post-marker tail"),
            ],
        )],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let content = body["messages"][0]["content"].as_array().unwrap();
    // 3 content parts + 1 cachePoint marker after block 1 = 4 wire blocks.
    assert_eq!(content.len(), 4);
    assert!(content[0].get("text").is_some() && content[0].get("cachePoint").is_none());
    assert!(content[1].get("text").is_some() && content[1].get("cachePoint").is_none());
    assert_eq!(content[2]["cachePoint"]["type"], "default");
    assert!(content[3].get("text").is_some() && content[3].get("cachePoint").is_none());
}

#[test]
fn bedrock_assistant_thinking_cache_control_emits_cache_point() {
    // Assistant-side thinking blocks accept a cache directive too.
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-v1:0".into(),
        messages: vec![Message::new(
            Role::Assistant,
            vec![ContentPart::Thinking {
                text: "reasoning".into(),
                cache_control: Some(CacheControl::five_minutes()),
                provider_echoes: Vec::new(),
            }],
        )],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert!(content[0].get("reasoningContent").is_some());
    assert_eq!(content[1]["cachePoint"]["type"], "default");
}

#[test]
fn bedrock_tool_result_cache_control_emits_cache_point() {
    // Tool results carry the heaviest payloads (RAG retrievals);
    // caching them is the canonical use case.
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-v1:0".into(),
        messages: vec![Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                tool_use_id: "t1".into(),
                name: "search".into(),
                content: entelix_core::ir::ToolResultContent::Text("retrieval result".into()),
                is_error: false,
                cache_control: Some(CacheControl::five_minutes()),
                provider_echoes: Vec::new(),
            }],
        )],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert!(content[0].get("toolResult").is_some());
    assert_eq!(content[1]["cachePoint"]["type"], "default");
}

#[test]
fn bedrock_tool_spec_cache_control_emits_cache_point_after_tool() {
    // ToolSpec.cache_control marks the end of the cacheable tool
    // declaration block — Bedrock supports this pattern for prefix
    // caching of large tool catalogs.
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-v1:0".into(),
        messages: vec![Message::new(
            Role::User,
            vec![ContentPart::text("call a tool")],
        )],
        tools: std::sync::Arc::from([
            ToolSpec::function("search", "Search the web", serde_json::json!({})),
            ToolSpec::function("read_file", "Read a file", serde_json::json!({}))
                .with_cache_control(CacheControl::five_minutes()),
        ]),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    let body = parse(&encoded.body);
    let tools = body["toolConfig"]["tools"].as_array().unwrap();
    // 2 toolSpec entries + 1 cachePoint marker after the second tool.
    assert_eq!(tools.len(), 3);
    assert!(tools[0].get("toolSpec").is_some());
    assert!(tools[1].get("toolSpec").is_some());
    assert_eq!(tools[2]["cachePoint"]["type"], "default");
}

#[test]
fn bedrock_one_hour_ttl_emits_lossy_encode_warning() {
    // Bedrock Converse `cachePoint` has no TTL knob — the cache
    // lifetime is fixed per-model server-side (5m default). When
    // the IR carries a non-default TTL we still emit the marker
    // but warn that the tier was coerced.
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4-v1:0".into(),
        messages: vec![Message::new(
            Role::User,
            vec![ContentPart::Text {
                text: "long-lived prefix".into(),
                cache_control: Some(CacheControl::one_hour()),
                provider_echoes: Vec::new(),
            }],
        )],
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&req).unwrap();
    assert!(lossy_for(
        &encoded.warnings,
        "messages[0].content[0].cache_control.ttl"
    ));
}
