//! `GeminiCodec` — IR ⇄ Google Gemini `generateContent` API
//! (`POST /v1beta/models/{model}:generateContent`,
//!   `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`).
//!
//! Wire format reference:
//! <https://ai.google.dev/api/rest/v1beta/models/generateContent>.
//!
//! Notable mappings:
//!
//! - IR `messages` → `contents: [{role: "user"|"model", parts: [...]}]`.
//!   Gemini uses `"model"` for assistant turns.
//! - IR `system: Option<String>` + IR `Role::System` → top-level
//!   `systemInstruction: { parts: [{ text }] }`.
//! - IR `Role::Tool` → `contents: [{role: "user", parts: [{
//!   functionResponse: { name, response: { ... } } }]}]`. Gemini does
//!   not roundtrip `tool_use_id`; the codec records the `LossyEncode`.
//! - IR `ContentPart::ToolUse` → `parts: [{ functionCall: { name, args } }]`.
//! - IR `tools` → `tools: [{ functionDeclarations: [...] }]`.
//! - IR `tool_choice` → `toolConfig: { functionCallingConfig: { mode } }`.
//! - Streaming SSE: `data: {...}\n\n` per chunk; each chunk is a full
//!   `GenerateContentResponse` with delta text in `candidates[0].content`.

#![allow(clippy::cast_possible_truncation)]

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::codecs::codec::{BoxByteStream, BoxDeltaStream, Codec, EncodedRequest};
use crate::error::{Error, Result};
use crate::ir::{
    Capabilities, CitationSource, ContentPart, MediaSource, ModelRequest, ModelResponse,
    ModelWarning, OutputStrategy, ReasoningEffort, RefusalReason, ResponseFormat, Role,
    SafetyCategory, SafetyLevel, SafetyRating, StopReason, ToolChoice, ToolKind, ToolResultContent,
    Usage,
};
use crate::stream::StreamDelta;

const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 1_000_000;

/// Stateless codec for the Gemini `generateContent` family of endpoints.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeminiCodec;

impl GeminiCodec {
    /// Create a fresh codec instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Codec for GeminiCodec {
    fn name(&self) -> &'static str {
        "gemini"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities {
            streaming: true,
            tools: true,
            multimodal_image: true,
            multimodal_audio: true,
            multimodal_video: true,
            multimodal_document: true,
            system_prompt: true,
            structured_output: true,
            prompt_caching: true,
            thinking: true,
            citations: true,
            web_search: true,
            computer_use: false,
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
        }
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let (body, warnings) = build_body(request)?;
        finalize_request(&request.model, &body, warnings, false)
    }

    fn encode_streaming(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let (body, warnings) = build_body(request)?;
        let mut encoded = finalize_request(&request.model, &body, warnings, true)?;
        encoded.headers.insert(
            http::header::ACCEPT,
            http::HeaderValue::from_static("text/event-stream"),
        );
        Ok(encoded.into_streaming())
    }

    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let raw: Value = super::codec::parse_response_body(body, "Gemini")?;
        let mut warnings = warnings_in;
        let id = String::new(); // Gemini one-shot responses lack a top-level id
        let model = str_field(&raw, "modelVersion").to_owned();
        let mut usage = decode_usage(raw.get("usageMetadata"));
        // Lift the candidate-scoped safetyRatings onto Usage so consumers
        // see safety on a single canonical channel.
        if let Some(candidate) = raw
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
        {
            usage.safety_ratings = decode_safety_ratings(candidate);
        }
        let (content, stop_reason) = decode_candidate(&raw, &mut warnings);
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

    fn decode_stream<'a>(
        &'a self,
        bytes: BoxByteStream<'a>,
        warnings_in: Vec<ModelWarning>,
    ) -> BoxDeltaStream<'a> {
        Box::pin(stream_gemini(bytes, warnings_in))
    }
}

// ── body builders ──────────────────────────────────────────────────────────

fn build_body(request: &ModelRequest) -> Result<(Value, Vec<ModelWarning>)> {
    if request.messages.is_empty() && request.system.is_empty() {
        return Err(Error::invalid_request(
            "Gemini generateContent requires at least one message",
        ));
    }
    let mut warnings = Vec::new();
    let (system_text, contents) = encode_messages(request, &mut warnings);

    let mut body = Map::new();
    body.insert("contents".into(), Value::Array(contents));
    if let Some(text) = system_text {
        body.insert(
            "systemInstruction".into(),
            json!({ "parts": [{ "text": text }] }),
        );
    }

    let mut generation_config = Map::new();
    if let Some(t) = request.max_tokens {
        generation_config.insert("maxOutputTokens".into(), json!(t));
    }
    if let Some(t) = request.temperature {
        generation_config.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        generation_config.insert("topP".into(), json!(p));
    }
    if !request.stop_sequences.is_empty() {
        generation_config.insert("stopSequences".into(), json!(request.stop_sequences));
    }
    if let Some(format) = &request.response_format {
        encode_gemini_structured_output(format, &mut generation_config, &mut body, &mut warnings)?;
    }
    if let Some(effort) = &request.reasoning_effort {
        encode_gemini_thinking(
            &request.model,
            effort,
            &mut generation_config,
            &mut warnings,
        );
    }
    if !generation_config.is_empty() {
        body.insert("generationConfig".into(), Value::Object(generation_config));
    }
    if !request.tools.is_empty() {
        body.insert("tools".into(), encode_tools(&request.tools, &mut warnings));
        body.insert(
            "toolConfig".into(),
            encode_tool_choice(&request.tool_choice),
        );
    }
    if let Some(name) = &request.cached_content {
        // Gemini native: server-side cached-content reference. The
        // value is a `cachedContents/<id>` resource name minted by
        // a prior `cachedContents` API call.
        body.insert("cachedContent".into(), Value::String(name.clone()));
    }
    if request.cache_key.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "cache_key".into(),
            detail: "Gemini has no `prompt_cache_key`-style routing field; \
                     server-side `cachedContent` is the native caching channel"
                .into(),
        });
    }
    apply_provider_extensions(request, &mut body, &mut warnings);
    Ok((Value::Object(body), warnings))
}

/// Read [`crate::ir::GeminiExt`] and merge each set field into the
/// wire body. `candidate_count` lands inside `generationConfig`,
/// creating the map if `build_body` did not already emit one.
/// Foreign-vendor extensions surface as
/// [`ModelWarning::ProviderExtensionIgnored`].
fn apply_provider_extensions(
    request: &ModelRequest,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let ext = &request.provider_extensions;
    if let Some(gemini) = &ext.gemini {
        if !gemini.safety_settings.is_empty() {
            let arr: Vec<Value> = gemini
                .safety_settings
                .iter()
                .map(|o| {
                    json!({
                        "category": o.category,
                        "threshold": o.threshold,
                    })
                })
                .collect();
            body.insert("safetySettings".into(), Value::Array(arr));
        }
        if let Some(n) = gemini.candidate_count {
            let entry = body
                .entry("generationConfig")
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(map) = entry.as_object_mut() {
                map.insert("candidateCount".into(), json!(n));
            }
        }
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
    if ext.openai_responses.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "openai_responses".into(),
        });
    }
    if ext.bedrock.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "bedrock".into(),
        });
    }
}

/// Resolve [`OutputStrategy`] and emit the Gemini native
/// `responseJsonSchema` (Native) or a forced-tool surface (Tool).
/// `Auto` resolves to `Native` — Gemini's `responseJsonSchema` is
/// the most direct surface and Gemini 2.5+ always strict-validates.
fn encode_gemini_structured_output(
    format: &ResponseFormat,
    generation_config: &mut Map<String, Value>,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) -> Result<()> {
    let strategy = match format.strategy {
        OutputStrategy::Auto | OutputStrategy::Native => OutputStrategy::Native,
        explicit => explicit,
    };
    match strategy {
        OutputStrategy::Native => {
            generation_config.insert("responseMimeType".into(), json!("application/json"));
            generation_config.insert(
                "responseJsonSchema".into(),
                format.json_schema.schema.clone(),
            );
            if !format.strict {
                warnings.push(ModelWarning::LossyEncode {
                    field: "response_format.strict".into(),
                    detail: "Gemini always strict-validates structured output; \
                         the strict=false request was approximated"
                        .into(),
                });
            }
        }
        OutputStrategy::Tool => {
            // Forced single function call. Gemini wraps tools as
            // `tools[0].functionDeclarations[0]` and `toolConfig`
            // narrows the selection; `mode: "ANY"` +
            // `allowedFunctionNames: [name]` is the canonical
            // forced-call shape.
            let tool_name = format.json_schema.name.clone();
            let synthetic_decl = json!({
                "name": tool_name,
                "description": format!(
                    "Emit the response as a JSON object matching the {tool_name} schema."
                ),
                "parameters": format.json_schema.schema.clone(),
            });
            body.insert(
                "tools".into(),
                json!([{
                    "functionDeclarations": [synthetic_decl],
                }]),
            );
            body.insert(
                "toolConfig".into(),
                json!({
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [format.json_schema.name],
                    }
                }),
            );
            if !format.strict {
                warnings.push(ModelWarning::LossyEncode {
                    field: "response_format.strict".into(),
                    detail: "Gemini Tool-strategy structured output is always \
                         schema-validated; strict=false was approximated"
                        .into(),
                });
            }
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

/// Gemini model family detection — 3.x uses `thinkingLevel`
/// (discrete bucket), 2.5 uses `thinkingBudget` (integer token
/// count, with `-1` = auto and `0` = disable on Flash only).
/// Detection by model-string prefix because Gemini's API does not
/// expose a wire signal for "this model accepts which thinking
/// shape".
fn is_gemini_3(model: &str) -> bool {
    model.starts_with("gemini-3")
}

/// Gemini 2.5 Flash accepts `thinkingBudget: 0` to disable thinking;
/// Pro cannot disable. Detection by model-string prefix.
fn is_gemini_25_flash(model: &str) -> bool {
    model.starts_with("gemini-2.5-flash") || model.starts_with("gemini-2.5-flash-lite")
}

/// Translate the cross-vendor [`ReasoningEffort`] knob onto
/// Gemini's `generationConfig.thinkingConfig`. Per ADR-0078:
///
/// 2.5 (`thinkingBudget` integer):
/// - `Off` → `0` (Flash only — Pro emits LossyEncode → `512`)
/// - `Minimal` → `512`
/// - `Low` → `1024`
/// - `Medium` → `8192`
/// - `High` → `24576`
/// - `Auto` → `-1`
/// - `VendorSpecific(s)` — `s` parses as decimal `thinkingBudget`;
///   non-numeric emits LossyEncode → `Medium`.
///
/// 3.x (`thinkingLevel` enum):
/// - `Off` → LossyEncode → `"minimal"` (Gemini 3 cannot disable)
/// - `Minimal/Low/Medium/High` → `"minimal"/"low"/"medium"/"high"`
/// - `Auto` → LossyEncode → `"high"` (no auto bucket)
/// - `VendorSpecific(s)` — literal `thinkingLevel`.
fn encode_gemini_thinking(
    model: &str,
    effort: &ReasoningEffort,
    generation_config: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let mut thinking_config = Map::new();
    if is_gemini_3(model) {
        let level = match effort {
            ReasoningEffort::Off => {
                warnings.push(ModelWarning::LossyEncode {
                    field: "reasoning_effort".into(),
                    detail: "Gemini 3 cannot disable thinking — snapped to `\"minimal\"`".into(),
                });
                "minimal"
            }
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::Auto => {
                warnings.push(ModelWarning::LossyEncode {
                    field: "reasoning_effort".into(),
                    detail: "Gemini 3 has no `Auto` bucket — snapped to `\"high\"`".into(),
                });
                "high"
            }
            ReasoningEffort::VendorSpecific(literal) => {
                thinking_config.insert("thinkingLevel".into(), Value::String(literal.clone()));
                generation_config.insert("thinkingConfig".into(), Value::Object(thinking_config));
                return;
            }
        };
        thinking_config.insert("thinkingLevel".into(), Value::String(level.into()));
    } else {
        // Gemini 2.5 (default for any non-3.x prefix — falls
        // through cleanly for 2.5 Pro / Flash / Flash-Lite).
        let budget: i32 = match effort {
            ReasoningEffort::Off => {
                if is_gemini_25_flash(model) {
                    0
                } else {
                    warnings.push(ModelWarning::LossyEncode {
                        field: "reasoning_effort".into(),
                        detail: format!(
                            "Gemini 2.5 Pro ({model}) cannot disable thinking — snapped to `512`"
                        ),
                    });
                    512
                }
            }
            ReasoningEffort::Minimal => 512,
            ReasoningEffort::Low => 1024,
            ReasoningEffort::Medium => 8192,
            ReasoningEffort::High => 24576,
            ReasoningEffort::Auto => -1,
            ReasoningEffort::VendorSpecific(literal) => {
                if let Ok(parsed) = literal.parse::<i32>() {
                    parsed
                } else {
                    warnings.push(ModelWarning::LossyEncode {
                        field: "reasoning_effort".into(),
                        detail: format!(
                            "Gemini 2.5 vendor-specific reasoning_effort {literal:?} is not \
                             a numeric thinkingBudget — falling through to `Medium`"
                        ),
                    });
                    8192
                }
            }
        };
        thinking_config.insert("thinkingBudget".into(), json!(budget));
    }
    generation_config.insert("thinkingConfig".into(), Value::Object(thinking_config));
}

fn finalize_request(
    model: &str,
    body: &Value,
    warnings: Vec<ModelWarning>,
    streaming: bool,
) -> Result<EncodedRequest> {
    let bytes = serde_json::to_vec(body)?;
    let path = if streaming {
        format!("/v1beta/models/{model}:streamGenerateContent?alt=sse")
    } else {
        format!("/v1beta/models/{model}:generateContent")
    };
    let mut encoded = EncodedRequest::post_json(path, Bytes::from(bytes));
    encoded.warnings = warnings;
    Ok(encoded)
}

// ── encode helpers ─────────────────────────────────────────────────────────

fn encode_messages(
    request: &ModelRequest,
    warnings: &mut Vec<ModelWarning>,
) -> (Option<String>, Vec<Value>) {
    let mut system_parts: Vec<String> = request
        .system
        .blocks()
        .iter()
        .map(|b| b.text.clone())
        .collect();
    if request.system.any_cached() {
        warnings.push(ModelWarning::LossyEncode {
            field: "system.cache_control".into(),
            detail: "Gemini has no native prompt-cache control on \
                     systemInstruction; block text is concatenated and \
                     the cache directive is dropped"
                .into(),
        });
    }
    let mut contents = Vec::new();

    for (idx, msg) in request.messages.iter().enumerate() {
        match msg.role {
            Role::System => {
                let mut lossy_non_text = false;
                let mut text = String::new();
                for part in &msg.content {
                    if let ContentPart::Text { text: t, .. } = part {
                        text.push_str(t);
                    } else {
                        lossy_non_text = true;
                    }
                }
                if lossy_non_text {
                    warnings.push(ModelWarning::LossyEncode {
                        field: format!("messages[{idx}].content"),
                        detail: "non-text parts dropped from system message (Gemini routes \
                                 system into systemInstruction)"
                            .into(),
                    });
                }
                if !text.is_empty() {
                    system_parts.push(text);
                }
            }
            Role::User => {
                contents.push(json!({
                    "role": "user",
                    "parts": encode_user_parts(&msg.content, warnings, idx),
                }));
            }
            Role::Assistant => {
                contents.push(json!({
                    "role": "model",
                    "parts": encode_assistant_parts(&msg.content, warnings, idx),
                }));
            }
            Role::Tool => {
                contents.push(json!({
                    "role": "user",
                    "parts": encode_tool_response_parts(&msg.content, warnings, idx),
                }));
            }
        }
    }

    let system_text = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system_text, contents)
}

fn encode_user_parts(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        let path = || format!("messages[{msg_idx}].content[{part_idx}]");
        match part {
            ContentPart::Text { text, .. } => out.push(json!({ "text": text })),
            ContentPart::Image { source, .. } => out.push(encode_media_gemini(source, "image/*")),
            ContentPart::Audio { source, .. } => out.push(encode_media_gemini(source, "audio/wav")),
            ContentPart::Video { source, .. } => out.push(encode_media_gemini(source, "video/mp4")),
            ContentPart::Document { source, .. } => {
                out.push(encode_media_gemini(source, "application/pdf"));
            }
            ContentPart::Thinking { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Gemini does not accept thinking blocks on input; block dropped".into(),
            }),
            ContentPart::Citation { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Gemini does not echo citations on input; block dropped".into(),
            }),
            ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "tool_use / tool_result not allowed on user role for Gemini".into(),
                });
            }
            ContentPart::ImageOutput { .. } | ContentPart::AudioOutput { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "Gemini does not accept assistant-produced image / audio output \
                             as input — block dropped"
                        .into(),
                });
            }
        }
    }
    out
}

fn encode_media_gemini(source: &MediaSource, fallback_mime: &str) -> Value {
    match source {
        MediaSource::Base64 { media_type, data } => json!({
            "inlineData": { "mimeType": media_type, "data": data },
        }),
        MediaSource::Url { url, media_type } => {
            let mime = media_type.as_deref().unwrap_or(fallback_mime); // silent-fallback-ok: caller-supplied fallback_mime is the typed MediaSource defaulting policy
            json!({
                "fileData": { "mimeType": mime, "fileUri": url },
            })
        }
        MediaSource::FileId { id, media_type } => {
            let mime = media_type.as_deref().unwrap_or(fallback_mime); // silent-fallback-ok: caller-supplied fallback_mime is the typed MediaSource defaulting policy
            json!({
                "fileData": { "mimeType": mime, "fileUri": id },
            })
        }
    }
}

fn encode_assistant_parts(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        let path = || format!("messages[{msg_idx}].content[{part_idx}]");
        match part {
            ContentPart::Text { text, .. } => out.push(json!({ "text": text })),
            ContentPart::ToolUse { name, input, .. } => {
                // Gemini's wire shape uses the assistant-emitted function name
                // as the round-trip key — there is no separate id field. The
                // `tool_use_id` round-trip is preserved at the IR layer by
                // letting the codec re-derive the id on decode from the same
                // `name + args` shape.
                out.push(json!({ "functionCall": { "name": name, "args": input } }));
            }
            ContentPart::Thinking {
                text, signature, ..
            } => {
                let mut o = Map::new();
                o.insert("text".into(), Value::String(text.clone()));
                o.insert("thought".into(), Value::Bool(true));
                if let Some(sig) = signature {
                    o.insert("thoughtSignature".into(), Value::String(sig.clone()));
                }
                out.push(Value::Object(o));
            }
            ContentPart::Citation { snippet, .. } => out.push(json!({ "text": snippet })),
            other => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: format!(
                        "{} not supported on model role for Gemini — dropped",
                        debug_part_kind(other)
                    ),
                });
            }
        }
    }
    out
}

fn encode_tool_response_parts(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        if let ContentPart::ToolResult {
            tool_use_id: _,
            name,
            content,
            is_error,
            ..
        } = part
        {
            let response_value = match content {
                ToolResultContent::Json(v) => v.clone(),
                ToolResultContent::Text(t) => json!({ "text": t }),
            };
            // Gemini's `functionResponse` keys correlation by
            // `name`, not by id. The IR carries the original name on
            // `ContentPart::ToolResult` precisely so this codec can
            // emit it verbatim — no placeholder, no LossyEncode.
            out.push(json!({
                "functionResponse": {
                    "name": name,
                    "response": response_value,
                },
            }));
            if *is_error {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("messages[{msg_idx}].content[{part_idx}].is_error"),
                    detail: "Gemini has no functionResponse error flag — passing through content"
                        .into(),
                });
            }
        } else {
            warnings.push(ModelWarning::LossyEncode {
                field: format!("messages[{msg_idx}].content[{part_idx}]"),
                detail: "non-tool_result part on Role::Tool dropped".into(),
            });
        }
    }
    out
}

fn encode_tools(tools: &[crate::ir::ToolSpec], warnings: &mut Vec<ModelWarning>) -> Value {
    let mut declarations = Vec::new();
    let mut tool_entries: Vec<Value> = Vec::new();
    for (idx, t) in tools.iter().enumerate() {
        match &t.kind {
            ToolKind::Function { input_schema } => declarations.push(json!({
                "name": t.name,
                "description": t.description,
                "parameters": input_schema,
            })),
            ToolKind::WebSearch { .. } => {
                // Gemini's google_search built-in is parameterless — domain
                // restrictions and use caps are not exposed on the wire.
                tool_entries.push(json!({ "google_search": {} }));
            }
            // Gemini ships function tools + google_search natively; the
            // Anthropic / OpenAI vendor built-ins have no Gemini equivalent.
            ToolKind::Computer { .. }
            | ToolKind::TextEditor
            | ToolKind::Bash
            | ToolKind::CodeExecution
            | ToolKind::FileSearch { .. }
            | ToolKind::CodeInterpreter
            | ToolKind::ImageGeneration
            | ToolKind::McpConnector { .. }
            | ToolKind::Memory => warnings.push(ModelWarning::LossyEncode {
                field: format!("tools[{idx}]"),
                detail: "Gemini natively ships only google_search — other vendor \
                         built-ins (computer, text_editor, file_search, …) have no \
                         Gemini equivalent; tool dropped"
                    .into(),
            }),
        }
    }
    if !declarations.is_empty() {
        tool_entries.insert(0, json!({ "functionDeclarations": declarations }));
    }
    Value::Array(tool_entries)
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    let mode = match choice {
        ToolChoice::Auto => "AUTO",
        // Gemini's "ANY" forces a tool call; for `Specific` we additionally
        // narrow via `allowedFunctionNames` below.
        ToolChoice::Required | ToolChoice::Specific { .. } => "ANY",
        ToolChoice::None => "NONE",
    };
    let mut config = json!({ "functionCallingConfig": { "mode": mode } });
    if let ToolChoice::Specific { name } = choice
        && let Some(cfg) = config
            .get_mut("functionCallingConfig")
            .and_then(Value::as_object_mut)
    {
        cfg.insert("allowedFunctionNames".into(), json!([name]));
    }
    config
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

// ── decode helpers ─────────────────────────────────────────────────────────

fn decode_candidate(
    raw: &Value,
    warnings: &mut Vec<ModelWarning>,
) -> (Vec<ContentPart>, StopReason) {
    let candidate = raw
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(Value::Null); // silent-fallback-ok: response with no candidates array → Null (downstream nested accessors propagate as None)
    let parts_raw = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default(); // silent-fallback-ok: candidate with no parts array → empty Vec (downstream loop iterates over zero items)
    let mut parts = Vec::new();
    // Per-response counter of `functionCall` parts seen so far —
    // synthesized into the tool-use id so streaming and
    // non-streaming decoders produce the same id sequence
    // (`{name}#{tool_seq}`) for the same logical sequence of tool
    // calls. Using the part-array index directly would diverge:
    // non-streaming sees `[text, fnCall, text, fnCall]` at indices
    // 1, 3 while streaming would emit them as 0, 1.
    let mut tool_seq: usize = 0;
    for (idx, part) in parts_raw.iter().enumerate() {
        // Thinking blocks: parts marked `thought: true` carry reasoning text.
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            let text = str_field(part, "text").to_owned();
            let signature = part
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .map(str::to_owned);
            parts.push(ContentPart::Thinking {
                text,
                signature,
                cache_control: None,
            });
            continue;
        }
        if let Some(text) = part.get("text").and_then(Value::as_str)
            && !text.is_empty()
        {
            parts.push(ContentPart::text(text));
            continue;
        }
        if let Some(call) = part.get("functionCall") {
            let name = str_field(call, "name").to_owned();
            let args = call.get("args").cloned().unwrap_or_else(|| json!({})); // silent-fallback-ok: functionCall without args = empty-args call (vendor sometimes omits when schema has no required fields)
            // Gemini does not round-trip a tool-use id — derive one
            // from `(name, tool_seq)` where `tool_seq` is a per-
            // response counter of function-call parts. Streaming
            // decoder uses the identical counter, so the same
            // logical sequence of tool calls produces the same id
            // sequence regardless of code path.
            parts.push(ContentPart::ToolUse {
                id: format!("{name}#{tool_seq}"),
                name,
                input: args,
            });
            tool_seq = tool_seq.saturating_add(1);
            continue;
        }
        warnings.push(ModelWarning::LossyEncode {
            field: format!("candidates[0].content.parts[{idx}]"),
            detail: "unknown Gemini part type dropped".into(),
        });
    }
    // Grounding metadata → Citation parts.
    if let Some(meta) = candidate.get("groundingMetadata")
        && let Some(chunks) = meta.get("groundingChunks").and_then(Value::as_array)
    {
        for chunk in chunks {
            if let Some(web) = chunk.get("web") {
                let url = str_field(web, "uri").to_owned();
                let title = web.get("title").and_then(Value::as_str).map(str::to_owned);
                if !url.is_empty() {
                    parts.push(ContentPart::Citation {
                        snippet: title.clone().unwrap_or_default(), // silent-fallback-ok: grounding citation without title → snippet "" (the URL is the load-bearing pointer; title is purely descriptive)
                        source: CitationSource::Url { url, title },
                        cache_control: None,
                    });
                }
            }
        }
    }
    let stop_reason = decode_finish_reason(
        candidate.get("finishReason").and_then(Value::as_str),
        warnings,
    );
    (parts, stop_reason)
}

fn decode_finish_reason(reason: Option<&str>, warnings: &mut Vec<ModelWarning>) -> StopReason {
    match reason {
        Some("STOP") => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        // Gemini distinguishes safety blocks from copyright
        // (RECITATION) blocks — preserve the distinction in IR so
        // dashboards can split by cause instead of collapsing both
        // into a single refusal bucket.
        Some("SAFETY") => StopReason::Refusal {
            reason: RefusalReason::Safety,
        },
        Some("RECITATION") => StopReason::Refusal {
            reason: RefusalReason::Recitation,
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
            // Invariant #15 — silent EndTurn fallback was masking
            // truncated stream payloads from callers. Record as
            // Other + warning instead.
            warnings.push(ModelWarning::LossyEncode {
                field: "finishReason".into(),
                detail: "Gemini candidate carried no finishReason — \
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
        input_tokens: u_field(usage, "promptTokenCount"),
        output_tokens: u_field(usage, "candidatesTokenCount"),
        cached_input_tokens: u_field(usage, "cachedContentTokenCount"),
        cache_creation_input_tokens: 0,
        reasoning_tokens: u_field(usage, "thoughtsTokenCount"),
        safety_ratings: Vec::new(),
    }
}

fn decode_safety_ratings(candidate: &Value) -> Vec<SafetyRating> {
    let Some(raw) = candidate.get("safetyRatings").and_then(Value::as_array) else {
        return Vec::new();
    };
    raw.iter()
        .filter_map(|r| {
            let category = match r.get("category").and_then(Value::as_str)? {
                "HARM_CATEGORY_HARASSMENT" => SafetyCategory::Harassment,
                "HARM_CATEGORY_HATE_SPEECH" => SafetyCategory::HateSpeech,
                "HARM_CATEGORY_SEXUALLY_EXPLICIT" => SafetyCategory::SexuallyExplicit,
                "HARM_CATEGORY_DANGEROUS_CONTENT" => SafetyCategory::DangerousContent,
                other => SafetyCategory::Other(other.to_owned()),
            };
            let level = match r.get("probability").and_then(Value::as_str)? {
                "LOW" => SafetyLevel::Low,
                "MEDIUM" => SafetyLevel::Medium,
                "HIGH" => SafetyLevel::High,
                // `"NEGLIGIBLE"` and any unrecognised vendor label collapse
                // to the lowest bucket — the IR's four-bucket scale is the
                // canonical resolution.
                _ => SafetyLevel::Negligible,
            };
            Some(SafetyRating { category, level })
        })
        .collect()
}

fn str_field<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("") // silent-fallback-ok: missing optional string field
}

fn u_field(v: Option<&Value>, key: &str) -> u32 {
    v.and_then(|inner| inner.get(key))
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX)) // silent-fallback-ok: missing usage metric = 0 (vendor didn't report = unused); u64→u32 saturate
}

// ── SSE streaming parser ───────────────────────────────────────────────────

#[allow(tail_expr_drop_order, clippy::too_many_lines)]
fn stream_gemini(
    bytes: BoxByteStream<'_>,
    warnings_in: Vec<ModelWarning>,
) -> impl futures::Stream<Item = Result<StreamDelta>> + Send + '_ {
    async_stream::stream! {
        let mut bytes = bytes;
        let mut buf: Vec<u8> = Vec::new();
        let mut started = false;
        let mut warnings_emitted = false;
        let mut last_stop = StopReason::EndTurn;
        let mut current_tool_open = false;
        let mut tool_synth_idx: u64 = 0;

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
                        "Gemini stream: malformed chunk: {payload}"
                    )));
                    return;
                };
                if !started {
                    started = true;
                    let model = str_field(&event, "modelVersion").to_owned();
                    yield Ok(StreamDelta::Start {
                        id: String::new(),
                        model,
                    });
                }
                if let Some(usage) = event.get("usageMetadata") {
                    yield Ok(StreamDelta::Usage(decode_usage(Some(usage))));
                }
                let Some(candidate) = event
                    .get("candidates")
                    .and_then(Value::as_array)
                    .and_then(|a| a.first())
                else {
                    continue;
                };
                if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
                    last_stop = decode_finish_reason(Some(reason), &mut Vec::new());
                }
                let Some(parts) = candidate
                    .get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(Value::as_array)
                else {
                    continue;
                };
                for part in parts {
                    // Thinking branches first so a `text` payload marked
                    // `thought: true` routes to ThinkingDelta rather than
                    // TextDelta.
                    if part.get("thought").and_then(Value::as_bool) == Some(true) {
                        if current_tool_open {
                            yield Ok(StreamDelta::ToolUseStop);
                            current_tool_open = false;
                        }
                        let text = part
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("") // silent-fallback-ok: missing thinking text → empty body; downstream is_empty() guard suppresses the StreamDelta
                            .to_owned();
                        let signature = part
                            .get("thoughtSignature")
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                        if !text.is_empty() || signature.is_some() {
                            yield Ok(StreamDelta::ThinkingDelta { text, signature });
                        }
                        continue;
                    }
                    if let Some(text) = part.get("text").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        if current_tool_open {
                            yield Ok(StreamDelta::ToolUseStop);
                            current_tool_open = false;
                        }
                        yield Ok(StreamDelta::TextDelta {
                            text: text.to_owned(),
                        });
                        continue;
                    }
                    if let Some(call) = part.get("functionCall") {
                        if current_tool_open {
                            yield Ok(StreamDelta::ToolUseStop);
                        }
                        let name = str_field(call, "name").to_owned();
                        let args = call.get("args").cloned().unwrap_or_else(|| json!({})); // silent-fallback-ok: streaming functionCall without args = empty-args call
                        let synth_id = format!("{name}#{tool_synth_idx}");
                        tool_synth_idx = tool_synth_idx.saturating_add(1);
                        yield Ok(StreamDelta::ToolUseStart {
                            id: synth_id,
                            name,
                        });
                        yield Ok(StreamDelta::ToolUseInputDelta {
                            partial_json: args.to_string(),
                        });
                        current_tool_open = true;
                    }
                }
            }
        }
        if current_tool_open {
            yield Ok(StreamDelta::ToolUseStop);
        }
        yield Ok(StreamDelta::Stop {
            stop_reason: last_stop,
        });
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
