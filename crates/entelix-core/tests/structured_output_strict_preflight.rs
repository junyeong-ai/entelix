//! Strict-mode preflight + native structured-output wire-shape
//! coverage.
//!
//! Per ADR-0024 §5 + ADR-0031, structured outputs ride native
//! vendor channels on every codec that supports them (Anthropic
//! `output_config.format`, Bedrock-routed Anthropic
//! `additionalModelRequestFields.output_config`, Gemini
//! `generationConfig.responseJsonSchema`, OpenAI Chat
//! `response_format`, OpenAI Responses `text.format`). Strict-mode
//! schemas that violate the vendor-shared constraints
//! (`additionalProperties: false` at every object level, `required`
//! lists every property) surface as `LossyEncode` so callers learn
//! at encode time that the wire request would be rejected.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::doc_markdown)]

use entelix_core::codecs::{Codec, OpenAiChatCodec, OpenAiResponsesCodec};
use entelix_core::ir::{
    JsonSchemaSpec, Message, ModelRequest, ModelWarning, ResponseFormat, StrictSchemaError,
};
use serde_json::json;

fn assert_lossy_warning_contains(
    warnings: &[ModelWarning],
    field_substr: &str,
    detail_substr: &str,
) {
    let found = warnings.iter().any(|w| match w {
        ModelWarning::LossyEncode { field, detail } => {
            field.contains(field_substr) && detail.contains(detail_substr)
        }
        _ => false,
    });
    assert!(
        found,
        "expected LossyEncode with field~='{field_substr}' detail~='{detail_substr}'; got {warnings:?}"
    );
}

fn assert_no_strict_lossy(warnings: &[ModelWarning]) {
    let found = warnings.iter().any(|w| match w {
        ModelWarning::LossyEncode { field, .. } => {
            field.contains("json_schema") || field.contains("text.format")
        }
        _ => false,
    });
    assert!(
        !found,
        "expected NO strict-preflight LossyEncode; got {warnings:?}"
    );
}

fn invalid_strict_schema() -> JsonSchemaSpec {
    // Missing additionalProperties:false AND required omits a property.
    JsonSchemaSpec::new(
        "invalid",
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }),
    )
    .unwrap()
}

fn valid_strict_schema() -> JsonSchemaSpec {
    JsonSchemaSpec::new(
        "valid",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }),
    )
    .unwrap()
}

// ── ResponseFormat::strict_preflight ───────────────────────────────────────

#[test]
fn strict_preflight_passes_for_valid_object_schema() {
    let format = ResponseFormat::strict(valid_strict_schema());
    assert!(format.strict_preflight().is_ok());
}

#[test]
fn strict_preflight_is_noop_when_strict_false() {
    let format = ResponseFormat::best_effort(invalid_strict_schema());
    assert!(format.strict_preflight().is_ok());
}

#[test]
fn strict_preflight_rejects_missing_additional_properties_false() {
    let schema = JsonSchemaSpec::new(
        "user",
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }),
    )
    .unwrap();
    let format = ResponseFormat::strict(schema);
    let err = format.strict_preflight().unwrap_err();
    assert!(matches!(
        err,
        StrictSchemaError::AdditionalPropertiesNotFalse { .. }
    ));
}

#[test]
fn strict_preflight_rejects_partial_required_list() {
    let schema = JsonSchemaSpec::new(
        "user",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }),
    )
    .unwrap();
    let format = ResponseFormat::strict(schema);
    match format.strict_preflight().unwrap_err() {
        StrictSchemaError::RequiredMissingProperties { missing, .. } => {
            assert_eq!(missing, vec!["age"]);
        }
        other => panic!("expected RequiredMissingProperties; got {other:?}"),
    }
}

#[test]
fn strict_preflight_recurses_into_nested_object() {
    let schema = JsonSchemaSpec::new(
        "outer",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}}
                },
            },
            "required": ["inner"],
        }),
    )
    .unwrap();
    let format = ResponseFormat::strict(schema);
    let err = format.strict_preflight().unwrap_err();
    match err {
        StrictSchemaError::AdditionalPropertiesNotFalse { path } => {
            assert!(
                path.contains("inner"),
                "path should point at nested object: {path}"
            );
        }
        other => panic!("expected AdditionalPropertiesNotFalse on nested; got {other:?}"),
    }
}

#[test]
fn strict_preflight_recurses_into_array_items() {
    let schema = JsonSchemaSpec::new(
        "list",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}}
                    }
                }
            },
            "required": ["items"],
        }),
    )
    .unwrap();
    let format = ResponseFormat::strict(schema);
    assert!(format.strict_preflight().is_err());
}

#[test]
fn strict_preflight_recurses_into_any_of() {
    let schema = JsonSchemaSpec::new(
        "tagged",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "kind": {
                    "anyOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}},
                    ]
                }
            },
            "required": ["kind"],
        }),
    )
    .unwrap();
    let format = ResponseFormat::strict(schema);
    let err = format.strict_preflight().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("anyOf"),
        "error path should reference anyOf: {msg}"
    );
}

// ── OpenAI Chat encode-time emission ───────────────────────────────────────

#[test]
fn openai_chat_emits_lossy_warning_for_invalid_strict_schema() {
    let codec = OpenAiChatCodec::new();
    let request = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::strict(invalid_strict_schema())),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    assert_lossy_warning_contains(
        &encoded.warnings,
        "response_format.json_schema",
        "additionalProperties",
    );
}

#[test]
fn openai_chat_clean_for_valid_strict_schema() {
    let codec = OpenAiChatCodec::new();
    let request = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::strict(valid_strict_schema())),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    assert_no_strict_lossy(&encoded.warnings);
}

#[test]
fn openai_chat_skips_preflight_when_strict_false() {
    let codec = OpenAiChatCodec::new();
    let request = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::best_effort(invalid_strict_schema())),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    assert_no_strict_lossy(&encoded.warnings);
}

// ── OpenAI Responses encode-time emission ──────────────────────────────────

#[test]
fn openai_responses_emits_lossy_warning_for_invalid_strict_schema() {
    let codec = OpenAiResponsesCodec::new();
    let request = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::strict(invalid_strict_schema())),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    assert_lossy_warning_contains(&encoded.warnings, "text.format", "additionalProperties");
}

#[test]
fn openai_responses_clean_for_valid_strict_schema() {
    let codec = OpenAiResponsesCodec::new();
    let request = ModelRequest {
        model: "gpt-4.1".into(),
        messages: vec![Message::user("hi")],
        response_format: Some(ResponseFormat::strict(valid_strict_schema())),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request).unwrap();
    assert_no_strict_lossy(&encoded.warnings);
}
