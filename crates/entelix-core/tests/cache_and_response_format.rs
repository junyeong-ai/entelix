//! Integration tests for [`CacheControl`] on [`SystemBlock`] and
//! [`ResponseFormat`] on [`ModelRequest`]. Verifies per-codec
//! native handling vs `LossyEncode` warning emission.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix_core::ir::{
    CacheControl, JsonSchemaSpec, Message, ModelRequest, ModelWarning, ResponseFormat, SystemBlock,
    SystemPrompt,
};
use serde_json::json;

fn req_with_cached_system() -> ModelRequest {
    ModelRequest {
        model: "claude-opus-4-7".into(),
        system: SystemPrompt::default()
            .with_block(SystemBlock::cached(
                "stable instructions",
                CacheControl::one_hour(),
            ))
            .with_block(SystemBlock::text("ephemeral context")),
        messages: vec![Message::user("hi")],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    }
}

fn assert_lossy_warning_for_field(warnings: &[ModelWarning], field: &str) {
    let found = warnings.iter().any(|w| match w {
        ModelWarning::LossyEncode { field: f, .. } => f == field,
        _ => false,
    });
    assert!(
        found,
        "expected LossyEncode warning for field '{field}'; got {warnings:?}"
    );
}

#[test]
fn anthropic_native_cache_control_emits_array_form() {
    let codec = AnthropicMessagesCodec::new();
    let request = req_with_cached_system();
    let encoded = codec.encode(&request).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).expect("body must be JSON");
    let system = body
        .get("system")
        .expect("anthropic system field must be present");
    assert!(
        system.is_array(),
        "cached system must serialize as array form: {system}"
    );
    let arr = system.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    let cache_dir = &arr[0]["cache_control"];
    // Wire shape: `type` is always `ephemeral`; the TTL rides in
    // the sibling `ttl` field (1h premium tier).
    assert_eq!(cache_dir["type"], "ephemeral");
    assert_eq!(cache_dir["ttl"], "1h");
    assert!(arr[1].get("cache_control").is_none());
    let no_lossy_for_cache = encoded.warnings.iter().all(|w| match w {
        ModelWarning::LossyEncode { field, .. } => !field.contains("cache_control"),
        _ => true,
    });
    assert!(
        no_lossy_for_cache,
        "anthropic should not emit LossyEncode for cache_control"
    );
}

#[test]
fn anthropic_uncached_system_uses_simple_string_form() {
    let codec = AnthropicMessagesCodec::new();
    let request = ModelRequest {
        model: "claude-opus-4-7".into(),
        system: "be terse".into(),
        messages: vec![Message::user("hi")],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    let system = body.get("system").unwrap();
    assert!(
        system.is_string(),
        "uncached system should use the simple string form: {system}"
    );
}

#[test]
fn openai_chat_emits_lossy_encode_warning_for_cached_system() {
    let codec = OpenAiChatCodec::new();
    let encoded = codec.encode(&req_with_cached_system()).unwrap();
    assert_lossy_warning_for_field(&encoded.warnings, "system.cache_control");
}

#[test]
fn openai_responses_emits_lossy_encode_warning_for_cached_system() {
    let codec = OpenAiResponsesCodec::new();
    let encoded = codec.encode(&req_with_cached_system()).unwrap();
    assert_lossy_warning_for_field(&encoded.warnings, "system.cache_control");
}

#[test]
fn gemini_emits_lossy_encode_warning_for_cached_system() {
    let codec = GeminiCodec::new();
    let encoded = codec.encode(&req_with_cached_system()).unwrap();
    assert_lossy_warning_for_field(&encoded.warnings, "system.cache_control");
}

#[test]
fn bedrock_converse_emits_native_cache_point_marker() {
    let codec = BedrockConverseCodec::new();
    let encoded = codec.encode(&req_with_cached_system()).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    let system = body
        .get("system")
        .and_then(|v| v.as_array())
        .expect("bedrock system must be array");
    let has_cache_point = system.iter().any(|v| v.get("cachePoint").is_some());
    assert!(
        has_cache_point,
        "bedrock should emit cachePoint marker for cached blocks: {system:?}"
    );
}

#[test]
fn openai_chat_native_response_format_emits_json_schema() {
    let codec = OpenAiChatCodec::new();
    let schema = JsonSchemaSpec::new(
        "user",
        json!({"type": "object", "properties": {"name": {"type": "string"}}}),
    )
    .unwrap();
    let request = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::strict(schema)),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    // OpenAI Chat Completions: top-level `response_format`.
    let rf = body.get("response_format").expect("native response_format");
    assert_eq!(rf["type"], "json_schema");
    assert_eq!(rf["json_schema"]["name"], "user");
    assert_eq!(rf["json_schema"]["strict"], true);
}

#[test]
fn openai_responses_native_response_format_emits_text_format() {
    let codec = OpenAiResponsesCodec::new();
    let schema = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
    let request = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::strict(schema)),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    // OpenAI Responses: nested `text.format` (not `response_format`).
    let format = body
        .get("text")
        .and_then(|t| t.get("format"))
        .expect("native text.format");
    assert_eq!(format["type"], "json_schema");
    assert_eq!(format["name"], "user");
    assert_eq!(format["strict"], true);
    // Negative: must NOT use the Chat Completions top-level shape.
    assert!(body.get("response_format").is_none());
}

#[test]
fn gemini_native_response_format_emits_response_json_schema() {
    let codec = GeminiCodec::new();
    let schema = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
    let request = ModelRequest {
        model: "gemini-2.0-flash".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::strict(schema)),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    let cfg = body
        .get("generationConfig")
        .expect("generationConfig must be present");
    assert_eq!(cfg["responseMimeType"], "application/json");
    // Gemini's modern field is `responseJsonSchema` (raw JSON
    // Schema). The legacy `responseSchema` (OpenAPI 3.0 subset) is gone.
    assert!(cfg.get("responseJsonSchema").is_some());
    assert!(cfg.get("responseSchema").is_none());
}

#[test]
fn anthropic_native_response_format_emits_output_config() {
    use entelix_core::ir::OutputStrategy;
    let codec = AnthropicMessagesCodec::new();
    let schema = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
    let request = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        // Per the Anthropic auto resolver picks `Tool`
        // (forced-tool is the more mature surface). This test
        // exercises the explicit `Native` path; the auto path is
        // covered by the dedicated forced-tool dispatch tests.
        response_format: Some(ResponseFormat::strict(schema).with_strategy(OutputStrategy::Native)),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    let format = body
        .get("output_config")
        .and_then(|v| v.get("format"))
        .expect("anthropic native output_config.format must be present");
    assert_eq!(format["type"], "json_schema");
    assert!(format.get("schema").is_some());
    let no_lossy = encoded.warnings.iter().all(|w| match w {
        ModelWarning::LossyEncode { field, .. } => !field.starts_with("response_format"),
        _ => true,
    });
    assert!(
        no_lossy,
        "anthropic must NOT emit response_format LossyEncode — native output_config: {:?}",
        encoded.warnings
    );
}

#[test]
fn anthropic_non_strict_response_format_warns_about_implicit_strict() {
    let codec = AnthropicMessagesCodec::new();
    let schema = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
    let request = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::best_effort(schema)),
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    assert_lossy_warning_for_field(&encoded.warnings, "response_format.strict");
}

#[test]
fn bedrock_native_response_format_routes_through_additional_model_request_fields() {
    use entelix_core::ir::OutputStrategy;
    let codec = BedrockConverseCodec::new();
    let schema = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
    let request = ModelRequest {
        model: "anthropic.claude-opus-4".into(),
        messages: vec![Message::user("hi")],
        // Bedrock-Anthropic auto-resolves to `Tool`;
        // the native passthrough is covered here explicitly.
        response_format: Some(ResponseFormat::strict(schema).with_strategy(OutputStrategy::Native)),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    let format = body
        .get("additionalModelRequestFields")
        .and_then(|v| v.get("output_config"))
        .and_then(|v| v.get("format"))
        .expect("bedrock must route Anthropic structured output via additionalModelRequestFields");
    assert_eq!(format["type"], "json_schema");
    assert!(format.get("schema").is_some());
    let no_lossy = encoded.warnings.iter().all(|w| match w {
        ModelWarning::LossyEncode { field, .. } => !field.starts_with("response_format"),
        _ => true,
    });
    assert!(
        no_lossy,
        "bedrock must NOT emit response_format LossyEncode for Anthropic-on-Bedrock: {:?}",
        encoded.warnings
    );
}

#[test]
fn json_schema_spec_rejects_invalid_input_at_construction() {
    let err = JsonSchemaSpec::new("", json!({"type": "object"})).unwrap_err();
    assert!(format!("{err}").contains("non-empty"));

    let err = JsonSchemaSpec::new("user", json!("not an object")).unwrap_err();
    assert!(format!("{err}").contains("JSON object"));
}

#[test]
fn response_format_strict_constructor_sets_strict_flag() {
    let schema = JsonSchemaSpec::new(
        "user",
        json!({"type": "object", "properties": {"name": {"type": "string"}}}),
    )
    .unwrap();
    let format = ResponseFormat::strict(schema.clone());
    assert!(format.strict);
    let lax = ResponseFormat::best_effort(schema);
    assert!(!lax.strict);
}
