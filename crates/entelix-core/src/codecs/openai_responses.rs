//! `OpenAiResponsesCodec` — IR ⇄ `OpenAI` Responses API
//! (`POST /v1/responses`).
//!
//! Wire format reference:
//! <https://platform.openai.com/docs/api-reference/responses>.
//!
//! Notable mappings:
//!
//! - IR `Role::System` / IR `system: Option<String>` → top-level
//!   `instructions` field.
//! - IR `Role::User` / `Role::Assistant` text messages →
//!   `input: [{type: "message", role, content: [{type:"input_text"|"output_text", text}]}]`.
//! - IR `ContentPart::ToolUse` (assistant) → standalone input item
//!   `{type: "function_call", call_id, name, arguments}` (separate from
//!   the assistant `message` item).
//! - IR `Role::Tool` `ToolResult` → standalone input item
//!   `{type: "function_call_output", call_id, output}`.
//! - IR `tools` → `[{type: "function", name, description, parameters}]`.
//! - Streaming SSE events: `response.output_text.delta`,
//!   `response.output_item.added`,
//!   `response.function_call_arguments.delta`, `response.completed`,
//!   `response.error`.

#![allow(clippy::cast_possible_truncation)]

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::codecs::codec::{
    BoxByteStream, BoxDeltaStream, Codec, EncodedRequest, extract_openai_rate_limit,
    service_tier_str,
};
use crate::error::{Error, Result};
use crate::ir::{
    Capabilities, CitationSource, ContentPart, MediaSource, ModelRequest, ModelResponse,
    ModelWarning, OutputStrategy, ProviderEchoSnapshot, ReasoningEffort, ReasoningSummary,
    RefusalReason, ResponseFormat, Role, StopReason, ToolChoice, ToolKind, ToolResultContent,
    Usage, find_provider_echo,
};
use crate::rate_limit::RateLimitSnapshot;
use crate::stream::StreamDelta;

const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 256_000;

/// Provider key for [`OpenAiResponsesCodec`] — identifies this
/// vendor's entries in [`ProviderEchoSnapshot`]. Carriers ride at
/// three levels: per-content-part (reasoning `encrypted_content` +
/// item `id`), per-response (`Response.id` for chain pointers), and
/// per-request (`previous_response_id` chain pointer back to a prior
/// turn).
const PROVIDER_KEY: &str = "openai-responses";

/// Stateless codec for the `OpenAI` Responses API.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiResponsesCodec;

impl OpenAiResponsesCodec {
    /// Create a fresh codec instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Codec for OpenAiResponsesCodec {
    fn name(&self) -> &'static str {
        "openai-responses"
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
            thinking: true,
            citations: true,
            web_search: true,
            computer_use: true,
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
        let raw: Value = super::codec::parse_response_body(body, "OpenAI Responses")?;
        let mut warnings = warnings_in;
        let id = str_field(&raw, "id").to_owned();
        let model = str_field(&raw, "model").to_owned();
        let usage = decode_usage(raw.get("usage"));
        let (content, stop_reason) = decode_outputs(&raw, &mut warnings);
        // Response-level chain pointer — the next request can echo
        // this `Response.id` via `ModelRequest::continued_from` to
        // continue the conversation server-side without re-sending
        // the full transcript (`store: true` mode), or as a
        // belt-and-braces audit handle alongside per-item
        // `encrypted_content` (`store: false` mode).
        let response_echoes = if id.is_empty() {
            Vec::new()
        } else {
            vec![ProviderEchoSnapshot::for_provider(
                PROVIDER_KEY,
                "response_id",
                id.clone(),
            )]
        };
        Ok(ModelResponse {
            id,
            model,
            stop_reason,
            content,
            usage,
            rate_limit: None,
            warnings,
            provider_echoes: response_echoes,
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
        Box::pin(stream_openai_responses(bytes, warnings_in))
    }
}

// ── body builders ──────────────────────────────────────────────────────────

fn build_body(request: &ModelRequest, streaming: bool) -> Result<(Value, Vec<ModelWarning>)> {
    if request.messages.is_empty() && request.system.is_empty() {
        return Err(Error::invalid_request(
            "OpenAI Responses requires at least one message",
        ));
    }
    let mut warnings = Vec::new();
    let (instructions, input_items) = encode_inputs(request, &mut warnings);

    let mut body = Map::new();
    body.insert("model".into(), Value::String(request.model.clone()));
    body.insert("input".into(), Value::Array(input_items));
    if let Some(s) = instructions {
        body.insert("instructions".into(), Value::String(s));
    }
    if let Some(n) = request.max_tokens {
        body.insert("max_output_tokens".into(), json!(n));
    }
    if let Some(t) = request.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if request.top_k.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "top_k".into(),
            detail: "OpenAI Responses has no top_k parameter — setting dropped".into(),
        });
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
        encode_openai_responses_structured_output(format, &mut body, &mut warnings)?;
    }
    if streaming {
        body.insert("stream".into(), Value::Bool(true));
    }
    if let Some(prev) = find_provider_echo(&request.continued_from, PROVIDER_KEY)
        .and_then(|e| e.payload_str("response_id"))
    {
        body.insert(
            "previous_response_id".into(),
            Value::String(prev.to_owned()),
        );
    }
    apply_provider_extensions(request, &mut body, &mut warnings);
    Ok((Value::Object(body), warnings))
}

/// Read [`crate::ir::OpenAiResponsesExt`] and merge each set field
/// into the wire body. Foreign-vendor extensions surface as
/// [`ModelWarning::ProviderExtensionIgnored`] — the operator
/// expressed an intent the OpenAI Responses format cannot honour.
fn apply_provider_extensions(
    request: &ModelRequest,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let ext = &request.provider_extensions;
    let openai_summary = ext
        .openai_responses
        .as_ref()
        .and_then(|e| e.reasoning_summary);
    if let Some(parallel) = request.parallel_tool_calls {
        body.insert("parallel_tool_calls".into(), json!(parallel));
    }
    if let Some(seed) = request.seed {
        body.insert("seed".into(), json!(seed));
    }
    if let Some(user) = &request.end_user_id {
        body.insert("user".into(), Value::String(user.clone()));
    }
    if let Some(openai_responses) = &ext.openai_responses {
        if let Some(key) = &openai_responses.cache_key {
            body.insert("prompt_cache_key".into(), Value::String(key.clone()));
        }
        if let Some(tier) = openai_responses.service_tier {
            body.insert(
                "service_tier".into(),
                Value::String(service_tier_str(tier).into()),
            );
        }
    }
    if let Some(effort) = &request.reasoning_effort {
        encode_openai_responses_reasoning(effort, openai_summary, body, warnings);
    } else if openai_summary.is_some() {
        // Operator set summary verbosity but did not set the
        // cross-vendor effort — OpenAI Responses requires
        // `reasoning.effort` whenever any `reasoning.summary`
        // value is set, so fall through to the vendor default
        // (medium) and surface the lossy snap.
        warnings.push(ModelWarning::LossyEncode {
            field: "reasoning_effort".into(),
            detail: "openai_responses_ext.reasoning_summary set without reasoning_effort — \
                 defaulting effort to `medium`"
                .into(),
        });
        encode_openai_responses_reasoning(&ReasoningEffort::Medium, openai_summary, body, warnings);
    }
    if ext.anthropic.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "anthropic".into(),
        });
    }
    if ext.openai_chat.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "openai_chat".into(),
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

/// Resolve [`OutputStrategy`] and emit either the native
/// `text.format` shape or a forced-tool surface (parity with the
/// other codecs). `Auto` resolves to `Native` — OpenAI's native
/// json_schema strict-mode is the most mature surface and the
/// industry baseline.
fn encode_openai_responses_structured_output(
    format: &ResponseFormat,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) -> Result<()> {
    let strategy = match format.strategy {
        OutputStrategy::Auto | OutputStrategy::Native => OutputStrategy::Native,
        explicit => explicit,
    };
    match strategy {
        OutputStrategy::Native => {
            if let Err(err) = format.strict_preflight() {
                warnings.push(ModelWarning::LossyEncode {
                    field: "text.format".into(),
                    detail: err.to_string(),
                });
            }
            body.insert(
                "text".into(),
                json!({
                    "format": {
                        "type": "json_schema",
                        "name": format.json_schema.name,
                        "schema": format.json_schema.schema,
                        "strict": format.strict,
                    }
                }),
            );
        }
        OutputStrategy::Tool => {
            // Forced single tool call carrying the target schema —
            // parity with Anthropic's Tool dispatch. OpenAI
            // Responses tools live under `tools` with `tool_choice`
            // narrowing the selection.
            let tool_name = format.json_schema.name.clone();
            let synthetic_tool = json!({
                "type": "function",
                "name": tool_name,
                "description": format!(
                    "Emit the response as a JSON object matching the {tool_name} schema."
                ),
                "parameters": format.json_schema.schema.clone(),
                "strict": format.strict,
            });
            let tools = body.entry("tools").or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(arr) = tools {
                arr.insert(0, synthetic_tool);
            }
            body.insert(
                "tool_choice".into(),
                json!({
                    "type": "function",
                    "name": format.json_schema.name,
                }),
            );
        }
        OutputStrategy::Prompted => {
            return Err(Error::invalid_request(
                "OutputStrategy::Prompted is deferred to entelix 1.1; use \
                 OutputStrategy::Native or OutputStrategy::Tool",
            ));
        }
        OutputStrategy::Auto => unreachable!("Auto resolved above"),
    }
    Ok(())
}

/// Translate the cross-vendor [`ReasoningEffort`] knob onto OpenAI
/// Responses' `reasoning: { effort, summary? }`. Mapping:
///
/// - `Off` → `effort:"none"`
/// - `Minimal` → `effort:"minimal"`
/// - `Low` → `effort:"low"`
/// - `Medium` → `effort:"medium"`
/// - `High` → `effort:"high"`
/// - `Auto` → LossyEncode → `effort:"medium"` (Responses has no
///   auto bucket)
/// - `VendorSpecific(s)` → literal `effort` value (e.g. `"xhigh"`)
fn encode_openai_responses_reasoning(
    effort: &ReasoningEffort,
    summary: Option<ReasoningSummary>,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let effort_str: String = match effort {
        ReasoningEffort::Off => "none".to_owned(),
        ReasoningEffort::Minimal => "minimal".to_owned(),
        ReasoningEffort::Low => "low".to_owned(),
        ReasoningEffort::Medium => "medium".to_owned(),
        ReasoningEffort::High => "high".to_owned(),
        ReasoningEffort::Auto => {
            warnings.push(ModelWarning::LossyEncode {
                field: "reasoning_effort".into(),
                detail: "OpenAI Responses has no `Auto` bucket — snapped to `medium`".into(),
            });
            "medium".to_owned()
        }
        ReasoningEffort::VendorSpecific(literal) => literal.clone(),
    };
    let mut obj = Map::new();
    obj.insert("effort".into(), Value::String(effort_str));
    if let Some(summary) = summary {
        let summary_str = match summary {
            ReasoningSummary::Auto => "auto",
            ReasoningSummary::Concise => "concise",
            ReasoningSummary::Detailed => "detailed",
        };
        obj.insert("summary".into(), Value::String(summary_str.into()));
    }
    body.insert("reasoning".into(), Value::Object(obj));
}

fn finalize_request(body: &Value, warnings: Vec<ModelWarning>) -> Result<EncodedRequest> {
    let bytes = serde_json::to_vec(body)?;
    let mut encoded = EncodedRequest::post_json("/v1/responses", Bytes::from(bytes));
    encoded.warnings = warnings;
    Ok(encoded)
}

// ── encode helpers ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn encode_inputs(
    request: &ModelRequest,
    warnings: &mut Vec<ModelWarning>,
) -> (Option<String>, Vec<Value>) {
    let mut instructions: Vec<String> = request
        .system
        .blocks()
        .iter()
        .map(|b| b.text.clone())
        .collect();
    if request.system.any_cached() {
        warnings.push(ModelWarning::LossyEncode {
            field: "system.cache_control".into(),
            detail: "OpenAI Responses has no native prompt-cache control; \
                     block text is concatenated into instructions and the \
                     cache directive is dropped"
                .into(),
        });
    }
    let mut items = Vec::new();

    for (idx, msg) in request.messages.iter().enumerate() {
        match msg.role {
            Role::System => {
                let mut text = String::new();
                let mut lossy = false;
                for part in &msg.content {
                    if let ContentPart::Text { text: t, .. } = part {
                        text.push_str(t);
                    } else {
                        lossy = true;
                    }
                }
                if lossy {
                    warnings.push(ModelWarning::LossyEncode {
                        field: format!("messages[{idx}].content"),
                        detail: "non-text parts dropped from system message (Responses routes \
                                 system into instructions)"
                            .into(),
                    });
                }
                if !text.is_empty() {
                    instructions.push(text);
                }
            }
            Role::User => {
                items.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": encode_user_content(&msg.content, warnings, idx),
                }));
            }
            Role::Assistant => {
                let (text_content, tool_calls) =
                    split_assistant_content(&msg.content, warnings, idx);
                if !text_content.is_empty() {
                    items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": text_content,
                    }));
                }
                for tool_call in tool_calls {
                    items.push(tool_call);
                }
            }
            Role::Tool => {
                for (part_idx, part) in msg.content.iter().enumerate() {
                    if let ContentPart::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        ..
                    } = part
                    {
                        let output_str = match content {
                            ToolResultContent::Text(t) => t.clone(),
                            ToolResultContent::Json(v) => v.to_string(),
                        };
                        items.push(json!({
                            "type": "function_call_output",
                            "call_id": tool_use_id,
                            "output": output_str,
                        }));
                        if *is_error {
                            warnings.push(ModelWarning::LossyEncode {
                                field: format!("messages[{idx}].content[{part_idx}].is_error"),
                                detail: "OpenAI Responses has no function_call_output error \
                                         flag — passing through content"
                                    .into(),
                            });
                        }
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

    let instructions = if instructions.is_empty() {
        None
    } else {
        Some(instructions.join("\n\n"))
    };
    (instructions, items)
}

fn encode_user_content(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        let path = || format!("messages[{msg_idx}].content[{part_idx}]");
        match part {
            ContentPart::Text { text, .. } => out.push(json!({
                "type": "input_text",
                "text": text,
            })),
            ContentPart::Image { source, .. } => out.push(json!({
                "type": "input_image",
                "image_url": media_to_url_responses(source),
            })),
            ContentPart::Audio { source, .. } => {
                if let MediaSource::Base64 { media_type, data } = source {
                    let format = audio_format_from_mime(media_type);
                    out.push(json!({
                        "type": "input_audio",
                        "input_audio": { "data": data, "format": format },
                    }));
                } else {
                    warnings.push(ModelWarning::LossyEncode {
                        field: path(),
                        detail: "OpenAI Responses input_audio requires base64 source".into(),
                    });
                }
            }
            ContentPart::Video { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "OpenAI Responses does not accept video inputs; block dropped".into(),
            }),
            ContentPart::Document { source, name, .. } => {
                if let MediaSource::FileId { id, .. } = source {
                    let mut o = Map::new();
                    o.insert("type".into(), Value::String("input_file".into()));
                    o.insert("file_id".into(), Value::String(id.clone()));
                    if let Some(n) = name {
                        o.insert("filename".into(), Value::String(n.clone()));
                    }
                    out.push(Value::Object(o));
                } else {
                    warnings.push(ModelWarning::LossyEncode {
                        field: path(),
                        detail: "OpenAI Responses document input requires Files-API FileId source"
                            .into(),
                    });
                }
            }
            ContentPart::Thinking { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "OpenAI Responses does not accept thinking blocks on input; block dropped"
                    .into(),
            }),
            ContentPart::Citation { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "OpenAI Responses does not echo citations on input; block dropped".into(),
            }),
            ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "tool_use / tool_result not allowed on user role for OpenAI Responses"
                        .into(),
                });
            }
            ContentPart::ImageOutput { .. } | ContentPart::AudioOutput { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "OpenAI Responses does not accept assistant-produced \
                             image / audio output as input — block dropped"
                        .into(),
                });
            }
            ContentPart::RedactedThinking { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "OpenAI Responses does not accept redacted_thinking blocks; block \
                             dropped"
                        .into(),
                });
            }
        }
    }
    out
}

fn media_to_url_responses(source: &MediaSource) -> String {
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
        // to `wav` — the OpenAI Responses default for input audio.
        _ => "wav",
    }
}

fn split_assistant_content(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> (Vec<Value>, Vec<Value>) {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        match part {
            ContentPart::Text { text, .. } => {
                text_parts.push(json!({
                    "type": "output_text",
                    "text": text,
                }));
            }
            ContentPart::ToolUse {
                id,
                name,
                input,
                provider_echoes,
            } => {
                let mut entry = Map::new();
                entry.insert("type".into(), Value::String("function_call".into()));
                entry.insert("call_id".into(), Value::String(id.clone()));
                entry.insert("name".into(), Value::String(name.clone()));
                entry.insert("arguments".into(), Value::String(input.to_string()));
                if let Some(fc_id) = find_provider_echo(provider_echoes, PROVIDER_KEY)
                    .and_then(|e| e.payload_str("id"))
                {
                    entry.insert("id".into(), Value::String(fc_id.to_owned()));
                }
                tool_calls.push(Value::Object(entry));
            }
            ContentPart::Citation { snippet, .. } => {
                text_parts.push(json!({
                    "type": "output_text",
                    "text": snippet,
                }));
            }
            ContentPart::Thinking {
                text,
                provider_echoes,
                ..
            } => {
                // Reasoning items round-trip on Responses API as
                // `reasoning` items. The summary array reconstructs
                // the reader-facing text; opaque carrier keys
                // (`id`, `encrypted_content`) ride at the item root
                // when present so a stateless multi-turn replay
                // recovers prior CoT continuity.
                let mut entry = Map::new();
                entry.insert("type".into(), Value::String("reasoning".into()));
                entry.insert(
                    "summary".into(),
                    json!([{ "type": "summary_text", "text": text }]),
                );
                if let Some(echo) = find_provider_echo(provider_echoes, PROVIDER_KEY) {
                    if let Some(rid) = echo.payload_str("id") {
                        entry.insert("id".into(), Value::String(rid.to_owned()));
                    }
                    if let Some(enc) = echo.payload_str("encrypted_content") {
                        entry.insert("encrypted_content".into(), Value::String(enc.to_owned()));
                    }
                }
                text_parts.push(Value::Object(entry));
            }
            other => {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("messages[{msg_idx}].content[{part_idx}]"),
                    detail: format!(
                        "{} not supported on assistant role for OpenAI Responses — dropped",
                        debug_part_kind(other)
                    ),
                });
            }
        }
    }
    (text_parts, tool_calls)
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
        ContentPart::RedactedThinking { .. } => "redacted_thinking",
    }
}

fn encode_tools(tools: &[crate::ir::ToolSpec], warnings: &mut Vec<ModelWarning>) -> Value {
    let mut arr: Vec<Value> = Vec::with_capacity(tools.len());
    for (idx, t) in tools.iter().enumerate() {
        let value = match &t.kind {
            ToolKind::Function { input_schema } => json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": input_schema,
            }),
            ToolKind::WebSearch {
                max_uses,
                allowed_domains,
            } => {
                let mut obj = Map::new();
                obj.insert("type".into(), Value::String("web_search".into()));
                if let Some(n) = max_uses {
                    obj.insert("max_uses".into(), json!(*n));
                }
                if !allowed_domains.is_empty() {
                    let mut filters = Map::new();
                    filters.insert("allowed_domains".into(), json!(allowed_domains));
                    obj.insert("filters".into(), Value::Object(filters));
                }
                Value::Object(obj)
            }
            ToolKind::Computer {
                display_width,
                display_height,
            } => json!({
                "type": "computer_use_preview",
                "display_width": *display_width,
                "display_height": *display_height,
                "environment": "browser",
            }),
            ToolKind::FileSearch { vector_store_ids } => {
                if vector_store_ids.is_empty() {
                    warnings.push(ModelWarning::LossyEncode {
                        field: format!("tools[{idx}].vector_store_ids"),
                        detail: "OpenAI file_search requires at least one vector_store_id; \
                                 tool dropped"
                            .into(),
                    });
                    continue;
                }
                json!({
                    "type": "file_search",
                    "vector_store_ids": vector_store_ids,
                })
            }
            ToolKind::CodeInterpreter => json!({
                "type": "code_interpreter",
                "container": { "type": "auto" },
            }),
            ToolKind::ImageGeneration => json!({ "type": "image_generation" }),
            ToolKind::TextEditor
            | ToolKind::Bash
            | ToolKind::CodeExecution
            | ToolKind::McpConnector { .. }
            | ToolKind::Memory => {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("tools[{idx}]"),
                    detail: "OpenAI Responses does not natively support Anthropic-only built-ins \
                             (text_editor / bash / code_execution / mcp / memory) — tool dropped"
                        .into(),
                });
                continue;
            }
        };
        arr.push(value);
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
            "name": name,
        }),
    }
}

// ── decode helpers ─────────────────────────────────────────────────────────

fn decode_function_call_item(
    item: &Value,
    idx: usize,
    warnings: &mut Vec<ModelWarning>,
) -> ContentPart {
    let id = str_field(item, "call_id").to_owned();
    let item_id = item.get("id").and_then(Value::as_str).map(str::to_owned);
    let name = str_field(item, "name").to_owned();
    let args_str = item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}"); // silent-fallback-ok: function_call without arguments = empty-args call (vendor sometimes omits when the schema has no required fields)
    // Invalid-JSON branch routes through ModelWarning::LossyEncode
    // and preserves the raw string in a `Value::String` so
    // downstream replay still sees the bytes the vendor emitted
    // (invariant #15 LossyEncode channel).
    let input = if let Ok(v) = serde_json::from_str::<Value>(args_str) {
        v
    } else {
        warnings.push(ModelWarning::LossyEncode {
            field: format!("output[{idx}].arguments"),
            detail: "function_call arguments not valid JSON; preserved as raw".into(),
        });
        Value::String(args_str.to_owned())
    };
    let provider_echoes = if let Some(fc_id) = item_id {
        vec![ProviderEchoSnapshot::for_provider(
            PROVIDER_KEY,
            "id",
            fc_id,
        )]
    } else {
        Vec::new()
    };
    ContentPart::ToolUse {
        id,
        name,
        input,
        provider_echoes,
    }
}

/// Translate one OpenAI Responses `reasoning` output item into the IR
/// `Thinking` shape. Carries both reader-facing summary text and the
/// opaque round-trip artifacts (`encrypted_content` + per-item `id`)
/// the harness must echo on stateless multi-turn replay. Returns
/// `None` when the item is empty in every dimension.
fn decode_reasoning_item(item: &Value) -> Option<ContentPart> {
    let item_id = item.get("id").and_then(Value::as_str).map(str::to_owned);
    let encrypted = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut payload = Map::new();
    if let Some(rid) = &item_id {
        payload.insert("id".into(), Value::String(rid.clone()));
    }
    if let Some(enc) = &encrypted {
        payload.insert("encrypted_content".into(), Value::String(enc.clone()));
    }
    let provider_echoes = if payload.is_empty() {
        Vec::new()
    } else {
        vec![ProviderEchoSnapshot::new(
            PROVIDER_KEY,
            Value::Object(payload),
        )]
    };
    let summary_text: String = item
        .get("summary")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default(); // silent-fallback-ok: reasoning item with no summary array → empty text
    if summary_text.is_empty() && provider_echoes.is_empty() {
        return None;
    }
    Some(ContentPart::Thinking {
        text: summary_text,
        cache_control: None,
        provider_echoes,
    })
}

fn decode_outputs(raw: &Value, warnings: &mut Vec<ModelWarning>) -> (Vec<ContentPart>, StopReason) {
    let outputs = raw
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default(); // silent-fallback-ok: response with no output array → empty content (decode loop iterates over zero items)
    let mut content = Vec::new();
    let mut tool_use_seen = false;
    for (idx, item) in outputs.iter().enumerate() {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let parts = item
                    .get("content")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(); // silent-fallback-ok: message item with no content array → empty parts (downstream loop iterates over zero items)
                for inner in parts {
                    let text = inner
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default() // silent-fallback-ok: missing text accessor → empty string; downstream !text.is_empty() guard suppresses the empty ContentPart
                        .to_owned();
                    if let Some(annotations) = inner.get("annotations").and_then(Value::as_array) {
                        for ann in annotations {
                            if ann.get("type").and_then(Value::as_str) == Some("url_citation") {
                                content.push(ContentPart::Citation {
                                    snippet: text.clone(),
                                    source: CitationSource::Url {
                                        url: str_field(ann, "url").to_owned(),
                                        title: ann
                                            .get("title")
                                            .and_then(Value::as_str)
                                            .map(str::to_owned),
                                    },
                                    cache_control: None,
                                    provider_echoes: Vec::new(),
                                });
                            }
                        }
                    }
                    if !text.is_empty() {
                        content.push(ContentPart::text(text));
                    }
                }
            }
            Some("reasoning") => {
                if let Some(part) = decode_reasoning_item(item) {
                    content.push(part);
                }
            }
            Some("function_call") => {
                content.push(decode_function_call_item(item, idx, warnings));
                tool_use_seen = true;
            }
            Some(other) => {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("output[{idx}].type"),
                    detail: format!("unsupported output item type {other:?} dropped"),
                });
            }
            None => {}
        }
    }
    let stop_reason = decode_status(
        raw.get("status").and_then(Value::as_str),
        tool_use_seen,
        warnings,
    );
    (content, stop_reason)
}

fn decode_status(
    status: Option<&str>,
    tool_use_seen: bool,
    warnings: &mut Vec<ModelWarning>,
) -> StopReason {
    // T19: status `incomplete` AND `tool_use_seen` means the model
    // emitted a partial tool_use and then hit the token cap. Both
    // signals are load-bearing — callers must know the tool_use was
    // truncated, not naturally invoked. Surface as `Other{raw:
    // "tool_use_truncated"}` plus a LossyEncode warning so the
    // observability path sees the truncation. (Invariant #15.)
    if tool_use_seen && matches!(status, Some("incomplete")) {
        warnings.push(ModelWarning::LossyEncode {
            field: "stop_reason".into(),
            detail: "OpenAI Responses status `incomplete` paired with \
                     partial tool_use — both signals preserved as \
                     `Other{raw:\"tool_use_truncated\"}`"
                .into(),
        });
        return StopReason::Other {
            raw: "tool_use_truncated".to_owned(),
        };
    }
    if tool_use_seen {
        return StopReason::ToolUse;
    }
    match status {
        Some("completed") => StopReason::EndTurn,
        Some("incomplete") => StopReason::MaxTokens,
        Some("failed") => StopReason::Refusal {
            reason: RefusalReason::ProviderFailure,
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
            // Invariant #15 — missing status surfaces as
            // `Other{raw:"missing"}` + LossyEncode warning. Silent
            // EndTurn would mask truncated streams.
            warnings.push(ModelWarning::LossyEncode {
                field: "status".into(),
                detail: "OpenAI Responses payload carried no status — \
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
        input_tokens: u_field(usage, "input_tokens"),
        output_tokens: u_field(usage, "output_tokens"),
        cached_input_tokens: u_field_nested(usage, &["input_tokens_details", "cached_tokens"]),
        cache_creation_input_tokens: 0,
        reasoning_tokens: u_field_nested(usage, &["output_tokens_details", "reasoning_tokens"]),
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
fn stream_openai_responses(
    bytes: BoxByteStream<'_>,
    warnings_in: Vec<ModelWarning>,
) -> impl futures::Stream<Item = Result<StreamDelta>> + Send + '_ {
    async_stream::stream! {
        let mut bytes = bytes;
        let mut buf: Vec<u8> = Vec::new();
        let mut started = false;
        let mut warnings_emitted = false;
        let mut current_tool_open = false;

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
                let Ok(event) = serde_json::from_str::<Value>(&payload) else {
                    yield Err(Error::invalid_request(format!(
                        "OpenAI Responses stream: malformed chunk: {payload}"
                    )));
                    return;
                };
                let event_type = event.get("type").and_then(Value::as_str).unwrap_or(""); // silent-fallback-ok: missing event type → no-match fallthrough
                match event_type {
                    "response.created" => {
                        let response = event.get("response").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates through child .get() chain as None
                        let id = str_field(response, "id").to_owned();
                        let model = str_field(response, "model").to_owned();
                        if !started {
                            started = true;
                            yield Ok(StreamDelta::Start { id, model });
                        }
                    }
                    "response.output_item.added" => {
                        let item = event.get("item").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
                        if item.get("type").and_then(Value::as_str) == Some("function_call") {
                            if current_tool_open {
                                yield Ok(StreamDelta::ToolUseStop);
                            }
                            let id = str_field(item, "call_id").to_owned();
                            let name = str_field(item, "name").to_owned();
                            yield Ok(StreamDelta::ToolUseStart {
                                id,
                                name,
                                provider_echoes: Vec::new(),
                            });
                            current_tool_open = true;
                        }
                    }
                    "response.output_text.delta" => {
                        if let Some(delta) = event.get("delta").and_then(Value::as_str)
                            && !delta.is_empty()
                        {
                            if current_tool_open {
                                yield Ok(StreamDelta::ToolUseStop);
                                current_tool_open = false;
                            }
                            yield Ok(StreamDelta::TextDelta {
                                text: delta.to_owned(),
                                provider_echoes: Vec::new(),
                            });
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        if let Some(delta) = event.get("delta").and_then(Value::as_str)
                            && !delta.is_empty()
                        {
                            yield Ok(StreamDelta::ToolUseInputDelta {
                                partial_json: delta.to_owned(),
                            });
                        }
                    }
                    "response.reasoning.delta" | "response.reasoning_summary_text.delta" => {
                        if let Some(text) = event.get("delta").and_then(Value::as_str) {
                            yield Ok(StreamDelta::ThinkingDelta {
                                text: text.to_owned(),
                                provider_echoes: Vec::new(),
                            });
                        }
                    }
                    "response.output_item.done" => {
                        let item = event.get("item").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
                        if item.get("type").and_then(Value::as_str) == Some("function_call")
                            && current_tool_open
                        {
                            yield Ok(StreamDelta::ToolUseStop);
                            current_tool_open = false;
                        }
                    }
                    "response.completed" => {
                        let response = event.get("response").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
                        if let Some(usage) = response.get("usage") {
                            yield Ok(StreamDelta::Usage(decode_usage(Some(usage))));
                        }
                        if current_tool_open {
                            yield Ok(StreamDelta::ToolUseStop);
                        }
                        let stop = decode_status(
                            response.get("status").and_then(Value::as_str),
                            false,
                            &mut Vec::new(),
                        );
                        // If any function_call lived in output[], we've already
                        // surfaced ToolUseStart/Stop deltas — let the
                        // aggregator's content order decide. For ToolUseStop
                        // semantics we override stop_reason if any tool_use was
                        // produced.
                        let outputs = response
                            .get("output")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default(); // silent-fallback-ok: completed response with no output array → empty (saw_tool stays false)
                        let saw_tool = outputs.iter().any(|o| {
                            o.get("type").and_then(Value::as_str) == Some("function_call")
                        });
                        let final_stop = if saw_tool { StopReason::ToolUse } else { stop };
                        yield Ok(StreamDelta::Stop {
                            stop_reason: final_stop,
                        });
                        return;
                    }
                    "response.error" | "error" => {
                        let err = event
                            .get("error")
                            .or_else(|| event.get("response").and_then(|r| r.get("error")))
                            .unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
                        let kind = str_field(err, "type");
                        let message = str_field(err, "message");
                        yield Err(Error::provider_network(format!(
                            "OpenAI Responses stream error ({kind}): {message}"
                        )));
                        return;
                    }
                    _ => {
                        // Many auxiliary events (response.in_progress,
                        // response.output_text.done, response.content_part.added,
                        // ...) carry no IR-relevant data. Ignore silently.
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
