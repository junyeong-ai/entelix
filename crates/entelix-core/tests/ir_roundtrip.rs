//! Serde round-trip tests for the public IR.
//!
//! Each test encodes a representative value and decodes it back, asserting
//! equality. Catches accidental tag/field renames before they reach a codec.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::redundant_closure_for_method_calls
)]

use entelix_core::ir::{
    Capabilities, ContentPart, MediaSource, Message, ModelRequest, ModelResponse, ModelWarning,
    Role, StopReason, ToolChoice, ToolKind, ToolResultContent, ToolSpec, Usage,
};

fn roundtrip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let json = serde_json::to_string(value).unwrap();
    serde_json::from_str(&json).unwrap()
}

#[test]
fn message_roundtrip() {
    let m = Message::user("hello");
    assert_eq!(roundtrip(&m), m);
    assert_eq!(
        roundtrip(&Message::assistant("hi back")),
        Message::assistant("hi back")
    );
    assert_eq!(
        roundtrip(&Message::system("be brief")),
        Message::system("be brief")
    );
}

#[test]
fn content_part_roundtrip_all_variants() {
    let parts = vec![
        ContentPart::text("alpha"),
        ContentPart::Image {
            source: MediaSource::url("https://example.com/x.png"),
            cache_control: None,
        },
        ContentPart::Image {
            source: MediaSource::base64("image/png", "iVBORw0KGgo="),
            cache_control: None,
        },
        ContentPart::Audio {
            source: MediaSource::base64("audio/wav", "AAAA"),
            cache_control: None,
        },
        ContentPart::Video {
            source: MediaSource::url("https://example.com/clip.mp4"),
            cache_control: None,
        },
        ContentPart::Document {
            source: MediaSource::file_id("file-abc"),
            name: Some("contract.pdf".into()),
            cache_control: None,
        },
        ContentPart::Thinking {
            text: "let me reason about this".into(),
            signature: Some("sig-001".into()),
            cache_control: None,
        },
        ContentPart::Citation {
            snippet: "according to the spec".into(),
            source: entelix_core::ir::CitationSource::Url {
                url: "https://example.com/spec".into(),
                title: Some("Spec".into()),
            },
            cache_control: None,
        },
        ContentPart::ToolUse {
            id: "tool_1".into(),
            name: "calculator".into(),
            input: serde_json::json!({ "expr": "2+2" }),
        },
        ContentPart::ToolResult {
            tool_use_id: "tool_1".into(),
            name: "calculator".into(),
            content: ToolResultContent::Text("4".into()),
            is_error: false,
            cache_control: None,
        },
        ContentPart::ToolResult {
            tool_use_id: "tool_2".into(),
            name: "ping".into(),
            content: ToolResultContent::Json(serde_json::json!({ "ok": true })),
            is_error: false,
            cache_control: None,
        },
    ];
    for p in &parts {
        assert_eq!(&roundtrip(p), p);
    }
}

#[test]
fn role_serde_uses_snake_case() {
    let json = serde_json::to_string(&Role::Assistant).unwrap();
    assert_eq!(json, "\"assistant\"");
    let role: Role = serde_json::from_str("\"tool\"").unwrap();
    assert_eq!(role, Role::Tool);
}

#[test]
fn model_request_default_roundtrips() {
    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        ..ModelRequest::default()
    };
    let r2: ModelRequest = roundtrip(&req);
    assert_eq!(r2.model, "claude-opus-4-7");
    assert_eq!(r2.messages.len(), 1);
    assert!(matches!(r2.tool_choice, ToolChoice::Auto));
}

#[test]
fn tool_choice_default_is_auto() {
    let tc = ToolChoice::default();
    assert!(matches!(tc, ToolChoice::Auto));
}

#[test]
fn tool_spec_roundtrip() {
    let spec = ToolSpec::function(
        "search",
        "Web search",
        serde_json::json!({
            "type": "object",
            "properties": { "q": { "type": "string" } },
            "required": ["q"],
        }),
    );
    assert_eq!(roundtrip(&spec), spec);
    assert!(matches!(spec.kind, ToolKind::Function { .. }));

    let web = ToolSpec {
        name: "web".into(),
        description: "Web search built-in".into(),
        kind: ToolKind::WebSearch {
            max_uses: Some(3),
            allowed_domains: vec!["example.com".into()],
        },
        cache_control: None,
    };
    assert_eq!(roundtrip(&web), web);
}

#[test]
fn usage_aggregates_correctly() {
    let u = Usage::new(100, 50)
        .with_cached_input_tokens(80)
        .with_cache_creation_input_tokens(20);
    assert_eq!(u.total(), 150);
    assert_eq!(u.billable_input(), 120); // 100 + cache_creation 20
}

#[test]
fn capabilities_default_is_conservative() {
    let c = Capabilities::default();
    assert!(!c.streaming);
    assert!(!c.tools);
    assert!(!c.multimodal_image);
    assert!(!c.multimodal_audio);
    assert!(!c.thinking);
    assert!(!c.web_search);
    assert_eq!(c.max_context_tokens, 0);
    assert_eq!(roundtrip(&c), c);
}

#[test]
fn stop_reason_variants_roundtrip() {
    for sr in [
        StopReason::EndTurn,
        StopReason::MaxTokens,
        StopReason::StopSequence {
            sequence: "###".into(),
        },
        StopReason::ToolUse,
        StopReason::Refusal {
            reason: entelix_core::ir::RefusalReason::Safety,
        },
        StopReason::Other {
            raw: "filter".into(),
        },
    ] {
        assert_eq!(roundtrip(&sr), sr);
    }
}

#[test]
fn model_warning_lossy_encode() {
    let w = ModelWarning::LossyEncode {
        field: "messages[0].content[1]".into(),
        detail: "image dropped (model lacks vision)".into(),
    };
    assert_eq!(roundtrip(&w), w);
}

#[test]
fn model_response_full_roundtrip() {
    let resp = ModelResponse {
        id: "msg_01".into(),
        model: "claude-opus-4-7".into(),
        stop_reason: StopReason::EndTurn,
        content: vec![ContentPart::text("done")],
        usage: Usage::new(5, 1),
        rate_limit: None,
        warnings: vec![],
    };
    assert_eq!(roundtrip(&resp), resp);
}

#[test]
fn model_response_first_text_skips_non_text_blocks() {
    let resp = ModelResponse {
        id: "msg_01".into(),
        model: "m".into(),
        stop_reason: StopReason::ToolUse,
        content: vec![
            ContentPart::ToolUse {
                id: "tu_1".into(),
                name: "lookup".into(),
                input: serde_json::json!({"q": "x"}),
            },
            ContentPart::text("the answer"),
            ContentPart::text("with a follow-up"),
        ],
        usage: Usage::default(),
        rate_limit: None,
        warnings: vec![],
    };
    assert_eq!(resp.first_text(), Some("the answer"));
    assert_eq!(resp.full_text(), "the answer\nwith a follow-up");
    assert!(resp.has_tool_uses());
    let uses = resp.tool_uses();
    assert_eq!(uses.len(), 1);
    assert_eq!(uses[0].name, "lookup");
    assert_eq!(uses[0].id, "tu_1");
}

#[test]
fn model_response_accessors_no_text_blocks_return_none() {
    let resp = ModelResponse {
        id: "msg_01".into(),
        model: "m".into(),
        stop_reason: StopReason::ToolUse,
        content: vec![ContentPart::ToolUse {
            id: "tu_1".into(),
            name: "lookup".into(),
            input: serde_json::json!({}),
        }],
        usage: Usage::default(),
        rate_limit: None,
        warnings: vec![],
    };
    assert_eq!(resp.first_text(), None);
    assert_eq!(resp.full_text(), "");
    assert!(resp.has_tool_uses());
}

#[test]
fn message_tool_result_constructors_emit_tool_role_with_correct_id_and_name() {
    let text = Message::tool_result("tu_42", "search_web", "found 7 hits");
    assert!(matches!(text.role, Role::Tool));
    match &text.content[0] {
        ContentPart::ToolResult {
            tool_use_id,
            name,
            content: ToolResultContent::Text(t),
            is_error,
            ..
        } => {
            assert_eq!(tool_use_id, "tu_42");
            assert_eq!(name, "search_web");
            assert_eq!(t, "found 7 hits");
            assert!(!is_error);
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }

    let json_msg = Message::tool_result_json("tu_43", "search_web", serde_json::json!({"hits": 7}));
    match &json_msg.content[0] {
        ContentPart::ToolResult {
            name,
            content: ToolResultContent::Json(v),
            ..
        } => {
            assert_eq!(name, "search_web");
            assert_eq!(v.get("hits").and_then(|h| h.as_u64()), Some(7));
        }
        other => panic!("expected JSON ToolResult, got {other:?}"),
    }

    let err_msg = Message::tool_error("tu_44", "broken_tool", "tool blew up");
    match &err_msg.content[0] {
        ContentPart::ToolResult { name, is_error, .. } => {
            assert_eq!(name, "broken_tool");
            assert!(is_error);
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}
