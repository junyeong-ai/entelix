//! Forcing function for invariant 6 (lossy encoding emits warnings).
//!
//! For every modality that the IR carries, exercise each codec and assert
//! that the codec either produces a wire shape that preserves the
//! variant *or* emits exactly one `LossyEncode` warning naming the
//! variant's IR path. **Silent loss fails the test.**
//!
//! The matrix here is the truth table for; updating IR variants
//! or codec native support without updating this test is rejected by CI.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix_core::ir::{
    ContentPart, MediaSource, Message, ModelRequest, ModelWarning, Role, ToolKind, ToolSpec,
};

/// One row of the matrix: `(label, request, native_codecs)`.
///
/// `native_codecs` lists codec names where the row's IR feature must be
/// preserved on the wire (no `LossyEncode` for the IR path). All other
/// codecs *must* emit at least one `LossyEncode` whose `field` mentions
/// the IR path.
struct Row<'a> {
    label: &'a str,
    request: ModelRequest,
    /// Codecs that natively encode this feature.
    native: &'a [&'a str],
    /// Substring expected in the `field` path of the `LossyEncode`
    /// warning emitted by non-native codecs.
    field_hint: &'a str,
}

fn req_with_user_part(part: ContentPart) -> ModelRequest {
    ModelRequest {
        model: "model".into(),
        messages: vec![Message::new(Role::User, vec![part])],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    }
}

fn req_with_assistant_part(part: ContentPart) -> ModelRequest {
    ModelRequest {
        model: "model".into(),
        messages: vec![
            Message::user("hi"),
            Message::new(Role::Assistant, vec![part]),
            Message::user("more"),
        ],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    }
}

fn req_with_tool(kind: ToolKind) -> ModelRequest {
    ModelRequest {
        model: "model".into(),
        messages: vec![Message::user("hi")],
        max_tokens: Some(1024),
        tools: vec![ToolSpec {
            name: "search".into(),
            description: "vendor built-in search".into(),
            kind,
            cache_control: None,
        }],
        ..ModelRequest::default()
    }
}

#[allow(clippy::too_many_lines)]
fn rows() -> Vec<Row<'static>> {
    vec![
        Row {
            label: "audio_input",
            request: req_with_user_part(ContentPart::Audio {
                source: MediaSource::base64("audio/wav", "AAAA"),
                cache_control: None,
            }),
            native: &["openai-chat", "openai-responses", "gemini"],
            field_hint: "messages[0].content[0]",
        },
        Row {
            label: "video_input",
            request: req_with_user_part(ContentPart::Video {
                source: MediaSource::base64("video/mp4", "AAAA"),
                cache_control: None,
            }),
            native: &["gemini"],
            field_hint: "messages[0].content[0]",
        },
        Row {
            label: "document_input",
            request: req_with_user_part(ContentPart::Document {
                source: MediaSource::base64("application/pdf", "AAAA"),
                name: Some("policy.pdf".into()),
                cache_control: None,
            }),
            // Anthropic + Bedrock + Gemini accept inline base64 docs natively;
            // OpenAI Chat / Responses require Files-API FileId so a base64
            // document is reasonably emitted as LossyEncode.
            native: &["anthropic-messages", "bedrock-converse", "gemini"],
            field_hint: "messages[0].content[0]",
        },
        Row {
            label: "thinking_block_on_assistant",
            request: req_with_assistant_part(ContentPart::Thinking {
                text: "let me reason".into(),
                signature: None,
                cache_control: None,
            }),
            // Anthropic, Gemini, Bedrock (Anthropic-on-Bedrock), and OpenAI
            // Responses round-trip thinking; OpenAI Chat does not accept it on
            // input.
            native: &[
                "anthropic-messages",
                "openai-responses",
                "gemini",
                "bedrock-converse",
            ],
            field_hint: "messages[1].content[0]",
        },
        Row {
            label: "web_search_tool",
            request: req_with_tool(ToolKind::WebSearch {
                max_uses: Some(3),
                allowed_domains: vec!["example.com".into()],
            }),
            native: &["anthropic-messages", "openai-responses", "gemini"],
            field_hint: "tools[0]",
        },
        Row {
            label: "computer_use_tool",
            request: req_with_tool(ToolKind::Computer {
                display_width: 1280,
                display_height: 800,
            }),
            native: &["anthropic-messages", "openai-responses"],
            field_hint: "tools[0]",
        },
        Row {
            label: "text_editor_tool",
            request: req_with_tool(ToolKind::TextEditor),
            native: &["anthropic-messages"],
            field_hint: "tools[0]",
        },
        Row {
            label: "bash_tool",
            request: req_with_tool(ToolKind::Bash),
            native: &["anthropic-messages"],
            field_hint: "tools[0]",
        },
        Row {
            label: "code_execution_tool",
            request: req_with_tool(ToolKind::CodeExecution),
            native: &["anthropic-messages", "gemini"],
            field_hint: "tools[0]",
        },
        Row {
            label: "file_search_tool",
            request: req_with_tool(ToolKind::FileSearch {
                vector_store_ids: vec!["vs_abc".into()],
            }),
            native: &["openai-responses"],
            field_hint: "tools[0]",
        },
        Row {
            label: "code_interpreter_tool",
            request: req_with_tool(ToolKind::CodeInterpreter),
            native: &["openai-responses"],
            field_hint: "tools[0]",
        },
        Row {
            label: "image_generation_tool",
            request: req_with_tool(ToolKind::ImageGeneration),
            native: &["openai-responses"],
            field_hint: "tools[0]",
        },
        Row {
            label: "mcp_connector_tool",
            request: req_with_tool(ToolKind::McpConnector {
                name: "weather".into(),
                server_url: "https://mcp.example.com".into(),
                authorization_token: None,
            }),
            native: &["anthropic-messages"],
            field_hint: "tools[0]",
        },
        Row {
            label: "memory_tool",
            request: req_with_tool(ToolKind::Memory),
            native: &["anthropic-messages"],
            field_hint: "tools[0]",
        },
    ]
}

fn run_codec(codec: &dyn Codec, request: &ModelRequest) -> Vec<ModelWarning> {
    codec
        .encode(request)
        .expect("codec encode must not fail")
        .warnings
}

fn lossy_for_field(warnings: &[ModelWarning], hint: &str) -> bool {
    warnings.iter().any(|w| match w {
        ModelWarning::LossyEncode { field, .. } => field.contains(hint),
        _ => false,
    })
}

#[test]
fn matrix_every_ir_feature_either_native_or_lossy_encoded() {
    let codecs: Vec<(&str, Box<dyn Codec>)> = vec![
        (
            "anthropic-messages",
            Box::new(AnthropicMessagesCodec::new()),
        ),
        ("openai-chat", Box::new(OpenAiChatCodec::new())),
        ("openai-responses", Box::new(OpenAiResponsesCodec::new())),
        ("gemini", Box::new(GeminiCodec::new())),
        ("bedrock-converse", Box::new(BedrockConverseCodec::new())),
    ];

    for row in rows() {
        for (codec_name, codec) in &codecs {
            let warnings = run_codec(codec.as_ref(), &row.request);
            let has_lossy = lossy_for_field(&warnings, row.field_hint);
            let is_native = row.native.contains(codec_name);

            if is_native {
                assert!(
                    !has_lossy,
                    "row {:?}: codec {codec_name} is declared native but emitted a LossyEncode for {:?}: {:#?}",
                    row.label, row.field_hint, warnings
                );
            } else {
                assert!(
                    has_lossy,
                    "row {:?}: codec {codec_name} is non-native but emitted no LossyEncode for {:?} — invariant 6 violation: {:#?}",
                    row.label, row.field_hint, warnings
                );
            }
        }
    }
}
