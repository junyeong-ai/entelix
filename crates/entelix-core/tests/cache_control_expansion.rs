//! `CacheControl` IR expansion coverage — per-block, per-tool,
//! plus the `cache_key` and `cached_content` request-level fields
//! (ADR-0031).
//!
//! Verifies that:
//!
//! - Anthropic per-`ContentPart` cache_control surfaces as
//!   `cache_control: { type: "<ttl>" }` on the matching wire block.
//! - Anthropic `ToolSpec.cache_control` attaches to the tool
//!   declaration.
//! - OpenAI Chat / Responses `cache_key` emits as `prompt_cache_key`.
//! - Gemini `cached_content` emits as `cachedContent`.
//! - Cross-cell mismatch emits a `LossyEncode` warning rather than
//!   silent drop.

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
    CacheControl, ContentPart, MediaSource, Message, ModelRequest, ModelWarning, Role, ToolSpec,
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
        tools: vec![
            ToolSpec::function(
                "stable",
                "stable tool — cache the declaration",
                serde_json::json!({"type": "object"}),
            )
            .with_cache_control(CacheControl::five_minutes()),
        ],
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
        cache_key: Some("user-42".into()),
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
        cache_key: Some("session-9".into()),
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
        cached_content: Some("cachedContents/abc123".into()),
        ..ModelRequest::default()
    };
    let body = parse(&codec.encode(&req).unwrap().body);
    assert_eq!(body["cachedContent"], "cachedContents/abc123");
}

#[test]
fn anthropic_emits_lossy_encode_for_cache_key_and_cached_content() {
    let codec = AnthropicMessagesCodec::new();
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        cache_key: Some("ignored-on-anthropic".into()),
        cached_content: Some("ignored-on-anthropic".into()),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let warnings = codec.encode(&req).unwrap().warnings;
    assert!(lossy_for(&warnings, "cache_key"));
    assert!(lossy_for(&warnings, "cached_content"));
}

#[test]
fn openai_chat_emits_lossy_encode_for_cached_content() {
    let codec = OpenAiChatCodec::new();
    let req = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        cached_content: Some("cachedContents/abc".into()),
        ..ModelRequest::default()
    };
    let warnings = codec.encode(&req).unwrap().warnings;
    assert!(lossy_for(&warnings, "cached_content"));
    // cache_key is supported, no LossyEncode for it.
    assert!(!lossy_for(&warnings, "cache_key"));
}

#[test]
fn gemini_emits_lossy_encode_for_cache_key() {
    let codec = GeminiCodec::new();
    let req = ModelRequest {
        model: "gemini-2.5-pro".into(),
        messages: vec![Message::user("hi")],
        cache_key: Some("ignored-on-gemini".into()),
        ..ModelRequest::default()
    };
    let warnings = codec.encode(&req).unwrap().warnings;
    assert!(lossy_for(&warnings, "cache_key"));
}

#[test]
fn bedrock_emits_lossy_encode_for_cache_key_and_cached_content() {
    let codec = BedrockConverseCodec::new();
    let req = ModelRequest {
        model: "anthropic.claude-opus-4".into(),
        messages: vec![Message::user("hi")],
        cache_key: Some("x".into()),
        cached_content: Some("x".into()),
        ..ModelRequest::default()
    };
    let warnings = codec.encode(&req).unwrap().warnings;
    assert!(lossy_for(&warnings, "cache_key"));
    assert!(lossy_for(&warnings, "cached_content"));
}

#[test]
fn cache_control_helpers_round_trip_on_content_parts() {
    // The IR `with_cache_control` helper preserves all other fields.
    let part = ContentPart::text("hello").with_cache_control(CacheControl::one_hour());
    if let ContentPart::Text {
        text,
        cache_control,
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
    };
    let after = original
        .clone()
        .with_cache_control(CacheControl::five_minutes());
    assert_eq!(original, after);
}
