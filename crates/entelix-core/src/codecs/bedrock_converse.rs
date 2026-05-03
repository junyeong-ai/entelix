//! `BedrockConverseCodec` — IR ⇄ AWS Bedrock Converse API
//! (`POST /model/{modelId}/converse`,
//!  `POST /model/{modelId}/converse-stream`).
//!
//! Wire format reference:
//! <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>.
//!
//! Notable mappings:
//!
//! - IR `messages` → `messages: [{role, content: [{...}]}]`. Roles are
//!   `"user"` / `"assistant"`; system prompts live at the top level.
//! - IR `system: Option<String>` + IR `Role::System` → top-level
//!   `system: [{text: "..."}]` array.
//! - IR `Role::Tool` `ToolResult` → wrapped on the wire as a
//!   `role: "user"` message containing
//!   `[{toolResult: {toolUseId, content, status}}]` (Bedrock represents
//!   tool outputs as user-authored content).
//! - IR `ContentPart::ToolUse` → `[{toolUse: {toolUseId, name, input}}]`.
//! - IR `tools` / `tool_choice` → `toolConfig: {tools, toolChoice}`.
//!
//! **Streaming defers**: AWS Bedrock streams use the binary
//! `application/vnd.amazon.eventstream` framing format which lives in
//! `entelix-cloud` alongside `SigV4` signing. `decode_stream` here uses
//! the `Codec` trait's default fallback (buffer-then-decode); a real
//! token-level streaming impl lands in the cloud crate.

#![allow(clippy::cast_possible_truncation)]

use bytes::Bytes;
use serde_json::{Map, Value, json};

use crate::codecs::codec::{Codec, EncodedRequest};
use crate::error::{Error, Result};
use crate::ir::{
    Capabilities, ContentPart, MediaSource, ModelRequest, ModelResponse, ModelWarning,
    RefusalReason, Role, StopReason, ToolChoice, ToolKind, ToolResultContent, Usage,
};
use crate::rate_limit::RateLimitSnapshot;

const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 200_000;

/// Stateless codec for the AWS Bedrock Converse API.
#[derive(Clone, Copy, Debug, Default)]
pub struct BedrockConverseCodec;

impl BedrockConverseCodec {
    /// Create a fresh codec instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Codec for BedrockConverseCodec {
    fn name(&self) -> &'static str {
        "bedrock-converse"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities {
            streaming: true,
            tools: true,
            multimodal_image: true,
            multimodal_audio: false,
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
        let (body, warnings) = build_body(request)?;
        finalize_request(&request.model, &body, warnings, false)
    }

    fn encode_streaming(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        let (body, warnings) = build_body(request)?;
        let mut encoded = finalize_request(&request.model, &body, warnings, true)?;
        encoded.headers.insert(
            http::header::ACCEPT,
            http::HeaderValue::from_static("application/vnd.amazon.eventstream"),
        );
        Ok(encoded.into_streaming())
    }

    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let raw: Value = super::codec::parse_response_body(body, "Bedrock Converse")?;
        let mut warnings = warnings_in;
        let id = String::new(); // Bedrock Converse responses have no top-level id
        let model = String::new(); // model echoed via header in real responses
        let usage = decode_usage(raw.get("usage"));
        let (content, stop_reason) = decode_output(&raw, &mut warnings);
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
        // AWS Bedrock returns invocation-quality signals on
        // `x-amzn-bedrock-*` headers (input/output token counts and a
        // measured invocation latency). Throttling itself surfaces
        // through `Retry-After` / `x-amzn-errortype` on the error
        // path. Capture every Bedrock-prefixed header into the raw
        // map so operators can build dashboards without per-codec
        // wire knowledge; promote the documented token-count
        // signals into typed fields where the IR has a slot.
        let mut snapshot = RateLimitSnapshot::default();
        let mut populated = false;
        for (name, value) in headers {
            let header_name = name.as_str();
            if !header_name.starts_with("x-amzn-bedrock-") {
                continue;
            }
            if let Ok(v) = value.to_str() {
                snapshot.raw.insert(header_name.to_owned(), v.to_owned());
                populated = true;
            }
        }
        // Bedrock's throttle rate-limit headers are not standardised
        // today; if Retry-After is present (set by the gateway under
        // load) propagate it so retry classifiers can honour it.
        if let Some(v) = headers.get("retry-after").and_then(|h| h.to_str().ok()) {
            snapshot.raw.insert("retry-after".into(), v.to_owned());
            populated = true;
        }
        populated.then_some(snapshot)
    }
}

// ── body builders ──────────────────────────────────────────────────────────

fn build_body(request: &ModelRequest) -> Result<(Value, Vec<ModelWarning>)> {
    if request.messages.is_empty() && request.system.is_empty() {
        return Err(Error::invalid_request(
            "Bedrock Converse requires at least one message",
        ));
    }
    let mut warnings = Vec::new();
    let (system_blocks, messages) = encode_messages(request, &mut warnings);

    let mut body = Map::new();
    body.insert("messages".into(), Value::Array(messages));
    if !system_blocks.is_empty() {
        body.insert("system".into(), Value::Array(system_blocks));
    }

    let mut inference_config = Map::new();
    if let Some(t) = request.max_tokens {
        inference_config.insert("maxTokens".into(), json!(t));
    }
    if let Some(t) = request.temperature {
        inference_config.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        inference_config.insert("topP".into(), json!(p));
    }
    if !request.stop_sequences.is_empty() {
        inference_config.insert("stopSequences".into(), json!(request.stop_sequences));
    }
    if !inference_config.is_empty() {
        body.insert("inferenceConfig".into(), Value::Object(inference_config));
    }

    if !request.tools.is_empty() {
        let mut tool_config = Map::new();
        tool_config.insert("tools".into(), encode_tools(&request.tools, &mut warnings));
        tool_config.insert(
            "toolChoice".into(),
            encode_tool_choice(&request.tool_choice),
        );
        body.insert("toolConfig".into(), Value::Object(tool_config));
    }
    if let Some(format) = &request.response_format {
        // Anthropic-on-Bedrock: structured output rides through
        // `additionalModelRequestFields`, which Bedrock passes
        // straight through to the underlying Anthropic Messages API.
        // The wire shape inside `additionalModelRequestFields`
        // matches direct Anthropic — `output_config.format` with raw
        // JSON Schema.
        let mut additional = body
            .remove("additionalModelRequestFields")
            .and_then(|v| match v {
                Value::Object(o) => Some(o),
                _ => None,
            })
            .unwrap_or_default(); // silent-fallback-ok: caller-initiated additionalModelRequestFields nesting — fresh empty Map when absent or non-object
        additional.insert(
            "output_config".into(),
            json!({
                "format": {
                    "type": "json_schema",
                    "schema": format.json_schema.schema.clone(),
                }
            }),
        );
        body.insert(
            "additionalModelRequestFields".into(),
            Value::Object(additional),
        );
        if !format.strict {
            warnings.push(ModelWarning::LossyEncode {
                field: "response_format.strict".into(),
                detail: "Anthropic-on-Bedrock always strict-validates structured output; \
                         the strict=false request was approximated"
                    .into(),
            });
        }
    }
    if request.cache_key.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "cache_key".into(),
            detail: "Bedrock Converse has no `prompt_cache_key`-style routing field; \
                     `cachePoint` markers are the native caching channel"
                .into(),
        });
    }
    if request.cached_content.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "cached_content".into(),
            detail: "Bedrock Converse has no Gemini-style `cachedContents` reference; \
                     `cachePoint` markers are the native caching channel"
                .into(),
        });
    }
    apply_provider_extensions(request, &mut body, &mut warnings);
    Ok((Value::Object(body), warnings))
}

/// Read [`crate::ir::BedrockExt`] and merge each set field into the
/// wire body. Foreign-vendor extensions surface as
/// [`ModelWarning::ProviderExtensionIgnored`] — the operator
/// expressed an intent the Bedrock Converse format cannot honour.
fn apply_provider_extensions(
    request: &ModelRequest,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let ext = &request.provider_extensions;
    if let Some(bedrock) = &ext.bedrock {
        if let Some(guardrail) = &bedrock.guardrail {
            body.insert(
                "guardrailConfig".into(),
                json!({
                    "guardrailIdentifier": guardrail.identifier,
                    "guardrailVersion": guardrail.version,
                }),
            );
        }
        if let Some(tier) = &bedrock.performance_config_tier {
            body.insert("performanceConfig".into(), json!({ "latency": tier }));
        }
    }
    // Anthropic-on-Bedrock: a subset of `AnthropicExt` rides through
    // `additionalModelRequestFields`, the rest is genuinely
    // unreachable on the Converse wire. Emit field-precise
    // `LossyEncode` warnings so the operator sees exactly which
    // setting was honoured and which was dropped.
    if let Some(anthropic) = &ext.anthropic {
        if let Some(thinking) = &anthropic.thinking {
            let mut additional = body
                .remove("additionalModelRequestFields")
                .and_then(|v| match v {
                    Value::Object(o) => Some(o),
                    _ => None,
                })
                .unwrap_or_default(); // silent-fallback-ok: caller-initiated additionalModelRequestFields nesting — fresh empty Map when absent or non-object
            additional.insert(
                "thinking".into(),
                json!({
                    "type": "enabled",
                    "budget_tokens": thinking.budget_tokens,
                }),
            );
            body.insert(
                "additionalModelRequestFields".into(),
                Value::Object(additional),
            );
        }
        if anthropic.disable_parallel_tool_use.is_some() {
            warnings.push(ModelWarning::LossyEncode {
                field: "provider_extensions.anthropic.disable_parallel_tool_use".into(),
                detail: "Bedrock Converse exposes no equivalent toggle — \
                         setting dropped on the wire"
                    .into(),
            });
        }
        if anthropic.user_id.is_some() {
            warnings.push(ModelWarning::LossyEncode {
                field: "provider_extensions.anthropic.user_id".into(),
                detail: "Bedrock Converse has no per-request user-id \
                         metadata channel — setting dropped"
                    .into(),
            });
        }
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
    if ext.gemini.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "gemini".into(),
        });
    }
}

fn finalize_request(
    model: &str,
    body: &Value,
    warnings: Vec<ModelWarning>,
    streaming: bool,
) -> Result<EncodedRequest> {
    let bytes = serde_json::to_vec(body)?;
    let path = if streaming {
        format!("/model/{model}/converse-stream")
    } else {
        format!("/model/{model}/converse")
    };
    let mut encoded = EncodedRequest::post_json(path, Bytes::from(bytes));
    encoded.warnings = warnings;
    Ok(encoded)
}

// ── encode helpers ─────────────────────────────────────────────────────────

fn encode_messages(
    request: &ModelRequest,
    warnings: &mut Vec<ModelWarning>,
) -> (Vec<Value>, Vec<Value>) {
    // Bedrock Converse system surface is `[{text, cachePoint?}, ...]`.
    // Bedrock supports cachePoint markers between blocks for
    // Claude / Llama models — emit them when the IR block carries
    // a cache_control directive. The cachePoint goes AFTER the
    // text block it should cache (Bedrock's documented contract).
    let mut system_blocks: Vec<Value> = Vec::new();
    for block in request.system.blocks() {
        system_blocks.push(json!({ "text": block.text.clone() }));
        if let Some(cache) = block.cache_control {
            // Bedrock Converse `cachePoint` only knows `{type:
            // "default"}` today — there is no per-tier TTL channel
            // (the cache lifetime is set per-model server-side). When
            // the IR carries a non-default TTL we still emit the
            // marker but warn that the tier was coerced.
            if cache.ttl != crate::ir::CacheTtl::FiveMinutes {
                warnings.push(ModelWarning::LossyEncode {
                    field: "system.cache_control.ttl".into(),
                    detail: format!(
                        "Bedrock cachePoint has no TTL knob — IR ttl `{:?}` coerced to vendor default",
                        cache.ttl
                    ),
                });
            }
            system_blocks.push(json!({
                "cachePoint": { "type": "default" }
            }));
        }
    }
    let mut messages = Vec::new();

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
                        detail: "non-text parts dropped from system message (Bedrock routes \
                                 system into top-level system array)"
                            .into(),
                    });
                }
                if !text.is_empty() {
                    system_blocks.push(json!({ "text": text }));
                }
            }
            Role::User => {
                messages.push(json!({
                    "role": "user",
                    "content": encode_user_content(&msg.content, warnings, idx),
                }));
            }
            Role::Assistant => {
                messages.push(json!({
                    "role": "assistant",
                    "content": encode_assistant_content(&msg.content, warnings, idx),
                }));
            }
            Role::Tool => {
                messages.push(json!({
                    "role": "user",
                    "content": encode_tool_results(&msg.content, warnings, idx),
                }));
            }
        }
    }
    (system_blocks, messages)
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
            ContentPart::Text { text, .. } => out.push(json!({ "text": text })),
            ContentPart::Image { source, .. } => match source {
                MediaSource::Base64 { media_type, data } => {
                    let format_str = media_type.split('/').next_back().unwrap_or("png"); // silent-fallback-ok: defense-in-depth — split() on a non-empty MIME ("image/...") always yields ≥1 segment; "png" is the Bedrock-documented image default
                    out.push(json!({
                        "image": {
                            "format": format_str,
                            "source": { "bytes": data },
                        },
                    }));
                }
                MediaSource::Url { url, .. } => warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: format!(
                        "Bedrock Converse requires base64 inline image bytes; URL '{url}' dropped"
                    ),
                }),
                MediaSource::FileId { .. } => warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "Bedrock Converse does not accept FileId image input".into(),
                }),
            },
            ContentPart::Audio { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Bedrock Converse does not accept audio inputs; block dropped".into(),
            }),
            ContentPart::Video { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Bedrock Converse video input is not declared in the codec's default \
                         capability set (Nova-series models only); block dropped"
                    .into(),
            }),
            ContentPart::Document { source, name, .. } => match source {
                MediaSource::Base64 { media_type, data } => {
                    let format_str = media_type.split('/').next_back().unwrap_or("pdf"); // silent-fallback-ok: defense-in-depth — split() on a non-empty MIME always yields ≥1 segment; "pdf" is the Bedrock-documented document default
                    let mut inner = Map::new();
                    inner.insert("format".into(), Value::String(format_str.into()));
                    if let Some(n) = name {
                        inner.insert("name".into(), Value::String(n.clone()));
                    }
                    inner.insert("source".into(), json!({ "bytes": data }));
                    out.push(json!({ "document": Value::Object(inner) }));
                }
                _ => warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "Bedrock Converse document accepts only base64 inline; URL/FileId \
                             dropped"
                        .into(),
                }),
            },
            ContentPart::Thinking { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Bedrock Converse does not accept thinking blocks on input; block dropped"
                    .into(),
            }),
            ContentPart::Citation { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Bedrock Converse does not echo citations on input; block dropped".into(),
            }),
            ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "tool_use / tool_result not allowed on user role for Bedrock Converse"
                        .into(),
                });
            }
            ContentPart::ImageOutput { .. } | ContentPart::AudioOutput { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "Bedrock Converse does not accept assistant-produced \
                             image / audio output as input — block dropped"
                        .into(),
                });
            }
        }
    }
    out
}

fn encode_assistant_content(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        let path = || format!("messages[{msg_idx}].content[{part_idx}]");
        match part {
            ContentPart::Text { text, .. } => out.push(json!({ "text": text })),
            ContentPart::ToolUse { id, name, input } => {
                out.push(json!({
                    "toolUse": {
                        "toolUseId": id,
                        "name": name,
                        "input": input,
                    },
                }));
            }
            ContentPart::Thinking {
                text, signature, ..
            } => {
                let mut inner = Map::new();
                inner.insert("text".into(), Value::String(text.clone()));
                if let Some(sig) = signature {
                    inner.insert("signature".into(), Value::String(sig.clone()));
                }
                out.push(json!({ "reasoningContent": { "reasoningText": Value::Object(inner) } }));
            }
            ContentPart::Citation { snippet, .. } => out.push(json!({ "text": snippet })),
            other => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: format!(
                        "{} not supported on assistant role for Bedrock Converse — dropped",
                        debug_part_kind(other)
                    ),
                });
            }
        }
    }
    out
}

fn encode_tool_results(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (part_idx, part) in parts.iter().enumerate() {
        if let ContentPart::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
        } = part
        {
            let inner = match content {
                ToolResultContent::Text(t) => json!([{ "text": t }]),
                ToolResultContent::Json(v) => json!([{ "json": v }]),
            };
            out.push(json!({
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": inner,
                    "status": if *is_error { "error" } else { "success" },
                },
            }));
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
    // Bedrock Converse `toolConfig` ships function tools natively;
    // vendor built-ins ride the `additionalModelRequestFields`
    // passthrough on the underlying model (Anthropic on Bedrock).
    // Either way the surface here only emits a `toolSpec` shim per
    // function-shaped tool — vendor built-ins surface as
    // `LossyEncode` so the operator routes them through codec
    // selection rather than expecting Bedrock to bridge.
    let mut arr: Vec<Value> = Vec::with_capacity(tools.len());
    for (idx, t) in tools.iter().enumerate() {
        let ToolKind::Function { input_schema } = &t.kind else {
            warnings.push(ModelWarning::LossyEncode {
                field: format!("tools[{idx}]"),
                detail: "Bedrock Converse `toolConfig` advertises only function tools — \
                         vendor built-ins (web_search, computer, text_editor, …) ride the \
                         underlying model's native surface and are not bridged here; \
                         tool dropped"
                    .into(),
            });
            continue;
        };
        arr.push(json!({
            "toolSpec": {
                "name": t.name,
                "description": t.description,
                "inputSchema": { "json": input_schema.clone() },
            },
        }));
    }
    Value::Array(arr)
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        // Bedrock has no explicit "none" — fall back to auto so the model
        // may decline tools naturally (verified against the Converse spec).
        ToolChoice::Auto | ToolChoice::None => json!({ "auto": {} }),
        ToolChoice::Required => json!({ "any": {} }),
        ToolChoice::Specific { name } => json!({ "tool": { "name": name } }),
    }
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

fn decode_output(raw: &Value, warnings: &mut Vec<ModelWarning>) -> (Vec<ContentPart>, StopReason) {
    let message = raw
        .get("output")
        .and_then(|o| o.get("message"))
        .cloned()
        .unwrap_or(Value::Null); // silent-fallback-ok: response with no output.message → Null (downstream nested accessors propagate as None)
    let parts_raw = message
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default(); // silent-fallback-ok: message with no content array → empty Vec (downstream loop iterates over zero items)
    let mut parts = Vec::new();
    for (idx, part) in parts_raw.iter().enumerate() {
        if let Some(text) = part.get("text").and_then(Value::as_str)
            && !text.is_empty()
        {
            parts.push(ContentPart::text(text));
            continue;
        }
        if let Some(reasoning) = part.get("reasoningContent")
            && let Some(reasoning_text) = reasoning.get("reasoningText")
        {
            let text = str_field(reasoning_text, "text").to_owned();
            let signature = reasoning_text
                .get("signature")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if !text.is_empty() || signature.is_some() {
                parts.push(ContentPart::Thinking {
                    text,
                    signature,
                    cache_control: None,
                });
            }
            continue;
        }
        if let Some(tool_use) = part.get("toolUse") {
            let id = str_field(tool_use, "toolUseId").to_owned();
            let name = str_field(tool_use, "name").to_owned();
            let input = tool_use.get("input").cloned().unwrap_or_else(|| json!({})); // silent-fallback-ok: toolUse without input = empty-args call (vendor sometimes omits when schema has no required fields)
            parts.push(ContentPart::ToolUse { id, name, input });
            continue;
        }
        warnings.push(ModelWarning::LossyEncode {
            field: format!("output.message.content[{idx}]"),
            detail: "unknown Bedrock content block type dropped".into(),
        });
    }
    let stop_reason = decode_stop_reason(raw, warnings);
    (parts, stop_reason)
}

fn decode_stop_reason(raw: &Value, warnings: &mut Vec<ModelWarning>) -> StopReason {
    let reason = raw.get("stopReason").and_then(Value::as_str);
    match reason {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        // T18: Bedrock Converse does not surface the matched stop
        // sequence in a top-level field. Some model providers route
        // it through `additionalModelResponseFields.stop_sequence`;
        // when present we preserve it, otherwise we record `Other`
        // plus a LossyEncode warning so the loss is observable
        // (invariant #15) rather than silently producing `""`.
        Some("stop_sequence") => {
            let matched = raw
                .get("additionalModelResponseFields")
                .and_then(|f| f.get("stop_sequence"))
                .and_then(Value::as_str);
            match matched {
                Some(s) if !s.is_empty() => StopReason::StopSequence {
                    sequence: s.to_owned(),
                },
                _ => {
                    warnings.push(ModelWarning::LossyEncode {
                        field: "stop_sequence".into(),
                        detail: "Bedrock Converse signalled `stop_sequence` but the matched \
                                 string is not exposed on the wire — IR records \
                                 `Other{raw:\"stop_sequence\"}`"
                            .into(),
                    });
                    StopReason::Other {
                        raw: "stop_sequence".to_owned(),
                    }
                }
            }
        }
        Some("tool_use") => StopReason::ToolUse,
        Some("guardrail_intervened" | "content_filtered") => StopReason::Refusal {
            reason: RefusalReason::Guardrail,
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
            // Invariant #15 — silent EndTurn fallback would mask
            // truncated stream payloads. Emit a LossyEncode warning
            // and surface as `Other{raw:"missing"}`.
            warnings.push(ModelWarning::LossyEncode {
                field: "stopReason".into(),
                detail: "Bedrock Converse response carried no stopReason — \
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
        input_tokens: u_field(usage, "inputTokens"),
        output_tokens: u_field(usage, "outputTokens"),
        cached_input_tokens: u_field(usage, "cacheReadInputTokens"),
        cache_creation_input_tokens: u_field(usage, "cacheWriteInputTokens"),
        reasoning_tokens: 0,
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
