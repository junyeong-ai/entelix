//! `OpenAiChatCodec` — IR ⇄ `OpenAI` Chat Completions API
//! (`POST /v1/chat/completions`).
//!
//! Wire format reference:
//! <https://platform.openai.com/docs/api-reference/chat>.
//!
//! Notable mappings:
//!
//! - IR `Role::System` messages → first `messages: [{role: "system", ...}]`
//!   entry. `OpenAI` keeps `system` inline in the messages list (unlike
//!   Anthropic).
//! - IR `Role::Tool` → `messages: [{role: "tool", tool_call_id, content}]`.
//! - IR `ContentPart::ToolUse` (assistant) →
//!   `tool_calls: [{id, type: "function", function: {name, arguments}}]`
//!   on the assistant message. `arguments` is a stringified JSON object.
//! - IR `ToolChoice::Required` → `tool_choice: "required"`.
//! - Streaming (`stream: true`) emits `data: {...}\n\n` SSE lines until a
//!   terminal `data: [DONE]\n\n`. Each chunk's `choices[0].delta` carries
//!   text (`content`) and partial tool calls
//!   (`tool_calls: [{index, id?, function: {name?, arguments?}}]`).

#![allow(clippy::cast_possible_truncation)]

use std::collections::HashSet;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::codecs::codec::{
    BoxByteStream, BoxDeltaStream, Codec, EncodedRequest, extract_openai_rate_limit,
};
use crate::error::{Error, Result};
use crate::ir::{
    Capabilities, CitationSource, ContentPart, MediaSource, ModelRequest, ModelResponse,
    ModelWarning, RefusalReason, Role, StopReason, ToolChoice, ToolKind, ToolResultContent, Usage,
};
use crate::rate_limit::RateLimitSnapshot;
use crate::stream::StreamDelta;

const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 128_000;

/// Stateless codec for the `OpenAI` Chat Completions API.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiChatCodec;

impl OpenAiChatCodec {
    /// Create a fresh codec instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Codec for OpenAiChatCodec {
    fn name(&self) -> &'static str {
        "openai-chat"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities {
            streaming: true,
            tools: true,
            multimodal_image: true,
            multimodal_audio: true,
            multimodal_video: false,
            multimodal_document: true,
            system_prompt: true,
            structured_output: true,
            prompt_caching: true,
            thinking: false,
            citations: true,
            web_search: false,
            computer_use: false,
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
        }
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let (body, warnings) = build_body(request, false)?;
        finalize_request(&body, warnings)
    }

    fn encode_streaming(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let (body, warnings) = build_body(request, true)?;
        let mut encoded = finalize_request(&body, warnings)?;
        encoded.headers.insert(
            http::header::ACCEPT,
            http::HeaderValue::from_static("text/event-stream"),
        );
        Ok(encoded.into_streaming())
    }

    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let raw: Value = super::codec::parse_response_body(body, "OpenAI Chat")?;
        let mut warnings = warnings_in;
        let id = str_field(&raw, "id").to_owned();
        let model = str_field(&raw, "model").to_owned();
        let usage = decode_usage(raw.get("usage"));
        let (content, stop_reason) = decode_choice(&raw, &mut warnings);
        Ok(ModelResponse {
            id,
            model,
            stop_reason,
            content,
            usage,
            rate_limit: None,
            warnings,
        })
    }

    fn extract_rate_limit(&self, headers: &http::HeaderMap) -> Option<RateLimitSnapshot> {
        extract_openai_rate_limit(headers)
    }

    fn decode_stream<'a>(
        &'a self,
        bytes: BoxByteStream<'a>,
        warnings_in: Vec<ModelWarning>,
    ) -> BoxDeltaStream<'a> {
        Box::pin(stream_openai_chat(bytes, warnings_in))
    }
}

// ── body builders ──────────────────────────────────────────────────────────

fn build_body(request: &ModelRequest, streaming: bool) -> Result<(Value, Vec<ModelWarning>)> {
    if request.messages.is_empty() && request.system.is_empty() {
        return Err(Error::invalid_request(
            "OpenAI Chat requires at least one message",
        ));
    }
    // Most encodes produce 0–1 warnings; pre-allocate one slot so the
    // first push doesn't reallocate.
    let mut warnings = Vec::with_capacity(1);
    let messages = encode_messages(request, &mut warnings);

    let mut body = Map::new();
    body.insert("model".into(), Value::String(request.model.clone()));
    body.insert("messages".into(), Value::Array(messages));
    if let Some(n) = request.max_tokens {
        body.insert("max_tokens".into(), json!(n));
    }
    if let Some(t) = request.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if !request.stop_sequences.is_empty() {
        body.insert("stop".into(), json!(request.stop_sequences));
    }
    if !request.tools.is_empty() {
        body.insert("tools".into(), encode_tools(&request.tools, &mut warnings));
        body.insert(
            "tool_choice".into(),
            encode_tool_choice(&request.tool_choice),
        );
    }
    if let Some(format) = &request.response_format {
        // OpenAI Chat native: response_format with json_schema.
        // Strict-mode preflight surfaces vendor-shared schema constraints
        // (additionalProperties: false at every object, required lists
        // every property) as `LossyEncode` so callers learn at encode
        // time that the wire request will be rejected.
        if let Err(err) = format.strict_preflight() {
            warnings.push(ModelWarning::LossyEncode {
                field: "response_format.json_schema".into(),
                detail: err.to_string(),
            });
        }
        body.insert(
            "response_format".into(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": format.json_schema.name,
                    "schema": format.json_schema.schema,
                    "strict": format.strict,
                }
            }),
        );
    }
    if let Some(key) = &request.cache_key {
        // OpenAI Chat Completions native prompt-cache routing key.
        body.insert("prompt_cache_key".into(), Value::String(key.clone()));
    }
    if request.cached_content.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "cached_content".into(),
            detail: "OpenAI Chat has no `cachedContents` reference field; \
                     use `cache_key` for routing into the auto-cache"
                .into(),
        });
    }
    if streaming {
        body.insert("stream".into(), Value::Bool(true));
        body.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    apply_provider_extensions(request, &mut body, &mut warnings);
    Ok((Value::Object(body), warnings))
}

/// Read [`crate::ir::OpenAiChatExt`] and merge each set field into the
/// wire body. Foreign-vendor extensions surface as
/// [`ModelWarning::ProviderExtensionIgnored`] — the operator
/// expressed an intent the OpenAI Chat format cannot honour.
fn apply_provider_extensions(
    request: &ModelRequest,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let ext = &request.provider_extensions;
    if let Some(openai_chat) = &ext.openai_chat {
        if let Some(seed) = openai_chat.seed {
            body.insert("seed".into(), json!(seed));
        }
        if let Some(user) = &openai_chat.user {
            body.insert("user".into(), Value::String(user.clone()));
        }
    }
    if ext.anthropic.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "anthropic".into(),
        });
    }
    if ext.openai_responses.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "openai_responses".into(),
        });
    }
    if ext.gemini.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "gemini".into(),
        });
    }
    if ext.bedrock.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "bedrock".into(),
        });
    }
}

fn finalize_request(body: &Value, warnings: Vec<ModelWarning>) -> Result<EncodedRequest> {
    let bytes = serde_json::to_vec(body)?;
    let mut encoded = EncodedRequest::post_json("/v1/chat/completions", Bytes::from(bytes));
    encoded.warnings = warnings;
    Ok(encoded)
}

// ── encode helpers ─────────────────────────────────────────────────────────

fn encode_messages(request: &ModelRequest, warnings: &mut Vec<ModelWarning>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    if !request.system.is_empty() {
        // OpenAI Chat has no native per-block cache directive —
        // concat block text into a single system message and emit
        // LossyEncode when any block was cached so callers see the
        // capability mismatch (ADR-0006 / ADR-0024 §5).
        if request.system.any_cached() {
            warnings.push(ModelWarning::LossyEncode {
                field: "system.cache_control".into(),
                detail: "OpenAI Chat has no native prompt-cache control; \
                         block text is concatenated and the cache directive \
                         is dropped"
                    .into(),
            });
        }
        out.push(json!({
            "role": "system",
            "content": request.system.concat_text(),
        }));
    }
    for (idx, msg) in request.messages.iter().enumerate() {
        match msg.role {
            Role::System => {
                let text = collect_text(&msg.content, warnings, idx);
                out.push(json!({ "role": "system", "content": text }));
            }
            Role::User => {
                out.push(json!({
                    "role": "user",
                    "content": encode_user_content(&msg.content, warnings, idx),
                }));
            }
            Role::Assistant => {
                out.push(encode_assistant_message(&msg.content, warnings, idx));
            }
            Role::Tool => {
                // OpenAI represents tool results as one message per
                // ToolResult part — split a multi-result IR message into
                // multiple wire messages.
                for (part_idx, part) in msg.content.iter().enumerate() {
                    if let ContentPart::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        ..
                    } = part
                    {
                        let body_str = match content {
                            ToolResultContent::Text(t) => t.clone(),
                            ToolResultContent::Json(v) => v.to_string(),
                        };
                        let mut entry = Map::new();
                        entry.insert("role".into(), Value::String("tool".into()));
                        entry.insert("tool_call_id".into(), Value::String(tool_use_id.clone()));
                        entry.insert("content".into(), Value::String(body_str));
                        if *is_error {
                            warnings.push(ModelWarning::LossyEncode {
                                field: format!("messages[{idx}].content[{part_idx}].is_error"),
                                detail: "OpenAI Chat has no tool_result is_error flag — \
                                         carrying via content text"
                                    .into(),
                            });
                        }
                        out.push(Value::Object(entry));
                    } else {
                        warnings.push(ModelWarning::LossyEncode {
                            field: format!("messages[{idx}].content[{part_idx}]"),
                            detail: "non-tool_result part on Role::Tool dropped".into(),
                        });
                    }
                }
            }
        }
    }
    out
}

fn encode_user_content(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Value {
    // OpenAI accepts a plain string when the content is text-only.
    if parts.iter().all(|p| matches!(p, ContentPart::Text { .. })) {
        let mut text = String::new();
        for part in parts {
            if let ContentPart::Text { text: t, .. } = part {
                text.push_str(t);
            }
        }
        return Value::String(text);
    }
    // Otherwise emit the array form with typed parts.
    let mut arr = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        let path = || format!("messages[{msg_idx}].content[{part_idx}]");
        match part {
            ContentPart::Text { text, .. } => {
                arr.push(json!({ "type": "text", "text": text }));
            }
            ContentPart::Image { source, .. } => {
                arr.push(json!({
                    "type": "image_url",
                    "image_url": { "url": media_to_url_chat(source) },
                }));
            }
            ContentPart::Audio { source, .. } => {
                // OpenAI Chat accepts only base64 input audio (`input_audio`).
                if let MediaSource::Base64 { media_type, data } = source {
                    let format = audio_format_from_mime(media_type);
                    arr.push(json!({
                        "type": "input_audio",
                        "input_audio": { "data": data, "format": format },
                    }));
                } else {
                    warnings.push(ModelWarning::LossyEncode {
                        field: path(),
                        detail: "OpenAI Chat input_audio requires base64 source; URL/FileId \
                                 audio dropped"
                            .into(),
                    });
                }
            }
            ContentPart::Video { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "OpenAI Chat does not accept video inputs; block dropped".into(),
            }),
            ContentPart::Document { source, name, .. } => {
                // Files API document — only `FileId` input is wire-supported on
                // Chat Completions today.
                if let MediaSource::FileId { id, .. } = source {
                    let mut o = Map::new();
                    o.insert("type".into(), Value::String("file".into()));
                    let mut file_obj = Map::new();
                    file_obj.insert("file_id".into(), Value::String(id.clone()));
                    if let Some(n) = name {
                        file_obj.insert("filename".into(), Value::String(n.clone()));
                    }
                    o.insert("file".into(), Value::Object(file_obj));
                    arr.push(Value::Object(o));
                } else {
                    warnings.push(ModelWarning::LossyEncode {
                        field: path(),
                        detail: "OpenAI Chat document input requires Files-API FileId source; \
                                 inline document dropped"
                            .into(),
                    });
                }
            }
            ContentPart::Thinking { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "OpenAI Chat does not accept thinking blocks on input; block dropped"
                    .into(),
            }),
            ContentPart::Citation { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "OpenAI Chat does not echo citations on input; block dropped".into(),
            }),
            ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "tool_use / tool_result not allowed on user role for OpenAI Chat; \
                             move to assistant or tool role"
                        .into(),
                });
            }
            ContentPart::ImageOutput { .. } | ContentPart::AudioOutput { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "OpenAI Chat does not accept assistant-produced image / audio output \
                             as input — block dropped"
                        .into(),
                });
            }
        }
    }
    Value::Array(arr)
}

fn media_to_url_chat(source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url, .. } => url.clone(),
        MediaSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
        MediaSource::FileId { id, .. } => id.clone(),
    }
}

fn audio_format_from_mime(mime: &str) -> &'static str {
    match mime {
        "audio/mp3" | "audio/mpeg" => "mp3",
        "audio/aac" => "aac",
        "audio/flac" => "flac",
        "audio/ogg" | "audio/opus" => "opus",
        // `audio/wav` / `audio/x-wav` and any unrecognised mime fall through
        // to `wav` — OpenAI Chat treats it as the default acceptable input
        // format.
        _ => "wav",
    }
}

fn encode_assistant_message(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Value {
    let mut text_buf = String::new();
    let mut tool_calls = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        match part {
            ContentPart::Text { text, .. } => text_buf.push_str(text),
            ContentPart::ToolUse { id, name, input } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": input.to_string(),
                    },
                }));
            }
            // Citations are part of the assistant reply on OpenAI — annotation
            // metadata round-trips through `decode_choice`. On encode, fold the
            // snippet text back into the reply (the model already emitted it as
            // text + annotations originally).
            ContentPart::Citation { snippet, .. } => text_buf.push_str(snippet),
            other => {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("messages[{msg_idx}].content[{part_idx}]"),
                    detail: format!(
                        "{} not supported on assistant role for OpenAI Chat — dropped",
                        debug_part_kind(other)
                    ),
                });
            }
        }
    }
    let mut entry = Map::new();
    entry.insert("role".into(), Value::String("assistant".into()));
    entry.insert(
        "content".into(),
        if text_buf.is_empty() {
            Value::Null
        } else {
            Value::String(text_buf)
        },
    );
    if !tool_calls.is_empty() {
        entry.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    Value::Object(entry)
}

fn collect_text(parts: &[ContentPart], warnings: &mut Vec<ModelWarning>, msg_idx: usize) -> String {
    let mut text = String::new();
    let mut lossy = false;
    for part in parts {
        match part {
            ContentPart::Text { text: t, .. } => text.push_str(t),
            _ => lossy = true,
        }
    }
    if lossy {
        warnings.push(ModelWarning::LossyEncode {
            field: format!("messages[{msg_idx}].content"),
            detail: "non-text parts dropped from system message".into(),
        });
    }
    text
}

const fn debug_part_kind(part: &ContentPart) -> &'static str {
    match part {
        ContentPart::Text { .. } => "text",
        ContentPart::Image { .. } => "image",
        ContentPart::Audio { .. } => "audio",
        ContentPart::Video { .. } => "video",
        ContentPart::Document { .. } => "document",
        ContentPart::Thinking { .. } => "thinking",
        ContentPart::Citation { .. } => "citation",
        ContentPart::ToolUse { .. } => "tool_use",
        ContentPart::ToolResult { .. } => "tool_result",
        ContentPart::ImageOutput { .. } => "image_output",
        ContentPart::AudioOutput { .. } => "audio_output",
    }
}

fn encode_tools(tools: &[crate::ir::ToolSpec], warnings: &mut Vec<ModelWarning>) -> Value {
    let mut arr = Vec::with_capacity(tools.len());
    for (idx, t) in tools.iter().enumerate() {
        match &t.kind {
            ToolKind::Function { input_schema } => arr.push(json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": input_schema,
                },
            })),
            // Chat Completions only ships function tools natively;
            // every vendor built-in lives on the Responses API.
            ToolKind::WebSearch { .. }
            | ToolKind::Computer { .. }
            | ToolKind::TextEditor
            | ToolKind::Bash
            | ToolKind::CodeExecution
            | ToolKind::FileSearch { .. }
            | ToolKind::CodeInterpreter
            | ToolKind::ImageGeneration
            | ToolKind::McpConnector { .. }
            | ToolKind::Memory => warnings.push(ModelWarning::LossyEncode {
                field: format!("tools[{idx}]"),
                detail: "OpenAI Chat Completions advertises only function tools — \
                         vendor built-ins (web_search, computer, file_search, …) \
                         live on the Responses API; tool dropped"
                    .into(),
            }),
        }
    }
    Value::Array(arr)
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".into()),
        ToolChoice::Required => Value::String("required".into()),
        ToolChoice::None => Value::String("none".into()),
        ToolChoice::Specific { name } => json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

// ── decode helpers ─────────────────────────────────────────────────────────

fn decode_choice(raw: &Value, warnings: &mut Vec<ModelWarning>) -> (Vec<ContentPart>, StopReason) {
    let choice = raw
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(Value::Null); // silent-fallback-ok: response with no choices array → Null (downstream nested accessors propagate as None)
    let message = choice.get("message").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
    let content = decode_assistant_message(message, warnings);
    let stop_reason = decode_finish_reason(
        choice.get("finish_reason").and_then(Value::as_str),
        warnings,
    );
    (content, stop_reason)
}

fn decode_assistant_message(message: &Value, warnings: &mut Vec<ModelWarning>) -> Vec<ContentPart> {
    let mut parts: Vec<ContentPart> = Vec::new();
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default(); // silent-fallback-ok: missing content accessor → empty string; downstream !text.is_empty() guard suppresses the empty ContentPart
    // URL annotations come as a sibling of `content` — emit them as
    // Citation parts before the text so order in the IR mirrors how the
    // model produced them (citations precede the corresponding sentence
    // the same way Anthropic emits citations *on* the text block).
    if let Some(annotations) = message.get("annotations").and_then(Value::as_array) {
        for ann in annotations {
            if ann.get("type").and_then(Value::as_str) == Some("url_citation")
                && let Some(uc) = ann.get("url_citation")
            {
                parts.push(ContentPart::Citation {
                    snippet: text.to_owned(),
                    source: CitationSource::Url {
                        url: str_field(uc, "url").to_owned(),
                        title: uc.get("title").and_then(Value::as_str).map(str::to_owned),
                    },
                    cache_control: None,
                });
            }
        }
    }
    if !text.is_empty() {
        parts.push(ContentPart::text(text));
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for (idx, call) in tool_calls.iter().enumerate() {
            let id = str_field(call, "id").to_owned();
            let function = call.get("function").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
            let name = str_field(function, "name").to_owned();
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}"); // silent-fallback-ok: tool_call without arguments = empty-args call (vendor sometimes omits when schema has no required fields)
            // Invalid-JSON branch routes through ModelWarning::LossyEncode
            // and preserves the raw string in a `Value::String` so
            // downstream replay still sees the bytes the vendor emitted
            // (invariant #15 LossyEncode channel).
            let input = if let Ok(v) = serde_json::from_str::<Value>(arguments) {
                v
            } else {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("choices[0].message.tool_calls[{idx}].function.arguments"),
                    detail: "tool arguments not valid JSON; preserved as raw string".into(),
                });
                Value::String(arguments.to_owned())
            };
            parts.push(ContentPart::ToolUse { id, name, input });
        }
    }
    parts
}

fn decode_finish_reason(reason: Option<&str>, warnings: &mut Vec<ModelWarning>) -> StopReason {
    match reason {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls" | "function_call") => StopReason::ToolUse,
        Some("content_filter") => StopReason::Refusal {
            reason: RefusalReason::Safety,
        },
        Some(other) => {
            warnings.push(ModelWarning::UnknownStopReason {
                raw: other.to_owned(),
            });
            StopReason::Other {
                raw: other.to_owned(),
            }
        }
        None => {
            // Invariant #15 — every codec must surface a missing
            // finish_reason as `Other{raw:"missing"}` plus a
            // `LossyEncode` warning. Silent EndTurn would mask "the
            // vendor truncated mid-response" for callers.
            warnings.push(ModelWarning::LossyEncode {
                field: "finish_reason".into(),
                detail: "OpenAI Chat response carried no finish_reason — \
                         IR records `Other{raw:\"missing\"}`"
                    .into(),
            });
            StopReason::Other {
                raw: "missing".to_owned(),
            }
        }
    }
}

fn decode_usage(usage: Option<&Value>) -> Usage {
    Usage {
        input_tokens: u_field(usage, "prompt_tokens"),
        output_tokens: u_field(usage, "completion_tokens"),
        cached_input_tokens: u_field_nested(usage, &["prompt_tokens_details", "cached_tokens"]),
        cache_creation_input_tokens: 0,
        reasoning_tokens: u_field_nested(usage, &["completion_tokens_details", "reasoning_tokens"]),
        safety_ratings: Vec::new(),
    }
}

fn str_field<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("") // silent-fallback-ok: missing optional string field
}

fn u_field(v: Option<&Value>, key: &str) -> u32 {
    v.and_then(|inner| inner.get(key))
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX)) // silent-fallback-ok: missing usage metric = 0 (vendor didn't report = unused); u64→u32 saturate
}

fn u_field_nested(v: Option<&Value>, path: &[&str]) -> u32 {
    let Some(mut cursor) = v else {
        return 0;
    };
    for segment in path {
        let Some(next) = cursor.get(*segment) else {
            return 0;
        };
        cursor = next;
    }
    cursor
        .as_u64()
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX)) // silent-fallback-ok: missing nested usage metric = 0 (vendor didn't report = unused); u64→u32 saturate
}

// ── SSE streaming parser ───────────────────────────────────────────────────

#[allow(tail_expr_drop_order, clippy::too_many_lines)]
fn stream_openai_chat(
    bytes: BoxByteStream<'_>,
    warnings_in: Vec<ModelWarning>,
) -> impl futures::Stream<Item = Result<StreamDelta>> + Send + '_ {
    async_stream::stream! {
        let mut bytes = bytes;
        let mut buf: Vec<u8> = Vec::new();
        // Open tool-call book-keeping keyed by tool_calls[].index.
        let mut tool_indices_open: HashSet<u64> = HashSet::new();
        let mut current_tool_index: Option<u64> = None;
        let mut started = false;
        let mut last_stop = StopReason::EndTurn;
        let mut warnings_emitted = false;

        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(b) => buf.extend_from_slice(&b),
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
            if !warnings_emitted {
                warnings_emitted = true;
                for w in &warnings_in {
                    yield Ok(StreamDelta::Warning(w.clone()));
                }
            }
            while let Some(pos) = find_double_newline(&buf) {
                let frame: Vec<u8> = buf.drain(..pos.saturating_add(2)).collect();
                let Ok(frame_str) = std::str::from_utf8(&frame) else {
                    continue;
                };
                let Some(payload) = parse_sse_data(frame_str) else {
                    continue;
                };
                if payload.trim() == "[DONE]" {
                    if current_tool_index.take().is_some() {
                        yield Ok(StreamDelta::ToolUseStop);
                    }
                    yield Ok(StreamDelta::Stop {
                        stop_reason: last_stop.clone(),
                    });
                    return;
                }
                let Ok(event) = serde_json::from_str::<Value>(&payload) else {
                    yield Err(Error::invalid_request(format!(
                        "OpenAI Chat stream: malformed chunk: {payload}"
                    )));
                    return;
                };
                if !started {
                    started = true;
                    let id = str_field(&event, "id").to_owned();
                    let model = str_field(&event, "model").to_owned();
                    yield Ok(StreamDelta::Start { id, model });
                }
                if let Some(usage) = event.get("usage").filter(|v| !v.is_null()) {
                    yield Ok(StreamDelta::Usage(decode_usage(Some(usage))));
                }
                let Some(choice) = event
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|a| a.first())
                else {
                    continue;
                };
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    last_stop = decode_finish_reason(Some(reason), &mut Vec::new());
                }
                let Some(delta) = choice.get("delta") else {
                    continue;
                };
                if let Some(text) = delta.get("content").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    if current_tool_index.take().is_some() {
                        yield Ok(StreamDelta::ToolUseStop);
                    }
                    yield Ok(StreamDelta::TextDelta {
                        text: text.to_owned(),
                    });
                }
                if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for call in tool_calls {
                        let idx = if let Some(n) = call.get("index").and_then(Value::as_u64) {
                            n
                        } else {
                            yield Ok(StreamDelta::Warning(ModelWarning::LossyEncode {
                                field: "stream.delta.tool_calls[].index".into(),
                                detail: "OpenAI Chat stream tool_call missing spec-mandated 'index' field; falling back to slot 0 (mirrors anthropic streaming idx handling)".into(),
                            }));
                            0
                        };
                        let function = call.get("function");
                        let name = function
                            .and_then(|f| f.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or(""); // silent-fallback-ok: streaming partial — name may not be present until later chunk; downstream is_empty() guards suppress empty deltas
                        let arguments = function
                            .and_then(|f| f.get("arguments"))
                            .and_then(Value::as_str)
                            .unwrap_or(""); // silent-fallback-ok: streaming partial — arguments accumulate across chunks; empty = "no addition this chunk"
                        let id = call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("") // silent-fallback-ok: streaming partial — id arrives on the first chunk for a given index, subsequent chunks omit
                            .to_owned();
                        if tool_indices_open.insert(idx) {
                            // Close any prior tool block before opening a new one.
                            if let Some(prev) = current_tool_index.take()
                                && prev != idx
                            {
                                yield Ok(StreamDelta::ToolUseStop);
                            }
                            yield Ok(StreamDelta::ToolUseStart {
                                id,
                                name: name.to_owned(),
                            });
                            current_tool_index = Some(idx);
                        }
                        if !arguments.is_empty() {
                            yield Ok(StreamDelta::ToolUseInputDelta {
                                partial_json: arguments.to_owned(),
                            });
                        }
                    }
                }
            }
        }
    }
}

fn find_double_newline(buf: &[u8]) -> Option<usize> {
    let lf = buf.windows(2).position(|w| w == b"\n\n");
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn parse_sse_data(frame: &str) -> Option<String> {
    let mut out: Option<String> = None;
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            let trimmed = rest.strip_prefix(' ').unwrap_or(rest); // silent-fallback-ok: SSE data line may or may not have leading space; idiomatic strip-or-pass-through
            match &mut out {
                Some(existing) => {
                    existing.push('\n');
                    existing.push_str(trimmed);
                }
                None => out = Some(trimmed.to_owned()),
            }
        }
    }
    out
}
