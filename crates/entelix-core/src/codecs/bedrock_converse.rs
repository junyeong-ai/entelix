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
    OutputStrategy, ProviderEchoSnapshot, ReasoningEffort, RefusalReason, ResponseFormat, Role,
    StopReason, ToolChoice, ToolKind, ToolResultContent, Usage,
};
use crate::rate_limit::RateLimitSnapshot;

/// Provider key for [`BedrockConverseCodec`] — matches `Codec::name`
/// and identifies this vendor's entries in [`ProviderEchoSnapshot`].
/// Bedrock Converse hosts both Anthropic Claude and Amazon Nova
/// reasoning models under the identical `reasoningContent` wire shape;
/// model-family branching lives inside this codec, not on the IR.
const PROVIDER_KEY: &str = "bedrock-converse";

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
        PROVIDER_KEY
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

    fn auto_output_strategy(&self, model: &str) -> OutputStrategy {
        // Bedrock-Anthropic mirrors direct Anthropic — prefer the
        // forced-tool surface (more mature, parity across
        // Anthropic versions on Bedrock). Non-Anthropic Bedrock
        // models default to `Native` but encode emits `LossyEncode`
        // on `response_format` since Nova / Mistral / Llama on
        // Converse have no canonical json_schema channel today.
        if is_bedrock_anthropic(model) {
            OutputStrategy::Tool
        } else {
            OutputStrategy::Native
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
            provider_echoes: Vec::new(),
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
    if let Some(k) = request.top_k {
        // Bedrock Converse `inferenceConfig` does not expose `top_k`;
        // for Anthropic-family models the parameter rides via
        // `additionalModelRequestFields.top_k`. Other Bedrock model
        // families (Nova, Mistral, Llama, …) have no `top_k`
        // equivalent — surface a typed lossy snap so the operator
        // sees the drop rather than a silently ignored field.
        if is_bedrock_anthropic(&request.model) {
            let mut additional = body
                .remove("additionalModelRequestFields")
                .and_then(|v| match v {
                    Value::Object(o) => Some(o),
                    _ => None,
                })
                .unwrap_or_default(); // silent-fallback-ok: caller-initiated additionalModelRequestFields nesting — fresh empty Map when absent or non-object
            additional.insert("top_k".into(), json!(k));
            body.insert(
                "additionalModelRequestFields".into(),
                Value::Object(additional),
            );
        } else {
            warnings.push(ModelWarning::LossyEncode {
                field: "top_k".into(),
                detail: "Bedrock Converse non-Anthropic models have no top_k parameter — \
                         setting dropped"
                    .into(),
            });
        }
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
        encode_bedrock_structured_output(format, &request.model, &mut body, &mut warnings)?;
    }
    if let Some(effort) = &request.reasoning_effort {
        encode_bedrock_thinking(&request.model, effort, &mut body, &mut warnings);
    }
    apply_provider_extensions(request, &mut body, &mut warnings);
    Ok((Value::Object(body), warnings))
}

/// Resolve [`OutputStrategy`] and emit either the Anthropic-on-
/// Bedrock native (`additionalModelRequestFields.output_config`),
/// the Bedrock-Anthropic forced-tool surface
/// (`additionalModelRequestFields.tool_choice`), or the
/// non-Anthropic Bedrock native fallback (currently `LossyEncode`
/// since Nova / Mistral / Llama on Converse have no canonical
/// json_schema channel today). `Auto` resolves to `Tool` for
/// Anthropic-family models (parity with the direct Anthropic
/// codec) and `Native` otherwise.
fn encode_bedrock_structured_output(
    format: &ResponseFormat,
    model: &str,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) -> Result<()> {
    let is_anthropic = is_bedrock_anthropic(model);
    let strategy = match format.strategy {
        OutputStrategy::Auto => {
            if is_anthropic {
                OutputStrategy::Tool
            } else {
                OutputStrategy::Native
            }
        }
        explicit => explicit,
    };
    if !is_anthropic {
        // Bedrock Nova / Mistral / Llama have no canonical
        // structured-output channel on Converse today. Future
        // codec updates can extend per-family encode here; for
        // 1.0 the typed loss surface gives operators a clear
        // signal.
        warnings.push(ModelWarning::LossyEncode {
            field: "response_format".into(),
            detail: format!(
                "Bedrock model {model:?} is not in the Anthropic family — Bedrock has no \
                 structured-output channel for non-Anthropic models on Converse; field dropped"
            ),
        });
        return Ok(());
    }
    let mut additional = body
        .remove("additionalModelRequestFields")
        .and_then(|v| match v {
            Value::Object(o) => Some(o),
            _ => None,
        })
        .unwrap_or_default(); // silent-fallback-ok: caller-initiated additionalModelRequestFields nesting — fresh empty Map when absent or non-object
    match strategy {
        OutputStrategy::Native => {
            additional.insert(
                "output_config".into(),
                json!({
                    "format": {
                        "type": "json_schema",
                        "schema": format.json_schema.schema.clone(),
                    }
                }),
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
        OutputStrategy::Tool => {
            // Forced-tool dispatch on Bedrock-Anthropic. The tool
            // and tool_choice ride through `additionalModelRequestFields`
            // since the Converse top-level `toolConfig` is the
            // wire's own tool surface for Bedrock; mixing
            // structured-output forced calls there would conflict
            // with operator-supplied tools. Anthropic Messages API
            // semantics live inside the passthrough.
            let tool_name = format.json_schema.name.clone();
            additional.insert(
                "tools".into(),
                json!([{
                    "type": "custom",
                    "name": tool_name,
                    "description": format!(
                        "Emit the response as a JSON object matching the {tool_name} schema."
                    ),
                    "input_schema": format.json_schema.schema.clone(),
                }]),
            );
            additional.insert(
                "tool_choice".into(),
                json!({
                    "type": "tool",
                    "name": format.json_schema.name,
                    "disable_parallel_tool_use": true,
                }),
            );
            if !format.strict {
                warnings.push(ModelWarning::LossyEncode {
                    field: "response_format.strict".into(),
                    detail: "Bedrock-Anthropic Tool-strategy structured output is always \
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
    body.insert(
        "additionalModelRequestFields".into(),
        Value::Object(additional),
    );
    Ok(())
}

/// Bedrock-on-Anthropic family detection — Claude models routed
/// through Converse accept Anthropic's `thinking` shape via
/// `additionalModelRequestFields`. Other Bedrock model families
/// (Nova, Mistral, Llama) have no thinking surface today; the
/// codec emits `LossyEncode` and drops the knob.
fn is_bedrock_anthropic(model: &str) -> bool {
    // Bedrock Anthropic model IDs include the cross-region inference
    // prefixes (e.g. `us.anthropic.claude-…`, `eu.anthropic.claude-…`)
    // alongside the bare `anthropic.claude-…` form.
    model.contains("anthropic.claude-")
}

/// Anthropic-on-Bedrock adaptive-only detection — Opus 4.7 hosted
/// on Bedrock inherits the same constraint (manual budget rejected).
fn is_bedrock_anthropic_adaptive_only(model: &str) -> bool {
    is_bedrock_anthropic(model) && model.contains("claude-opus-4-7")
}

/// Translate the cross-vendor [`ReasoningEffort`] knob onto the
/// Bedrock Converse `additionalModelRequestFields.thinking`
/// passthrough. Reuses the Anthropic mapping () for
/// Anthropic-family models on Bedrock; non-Anthropic models emit
/// `LossyEncode` and drop the knob.
fn encode_bedrock_thinking(
    model: &str,
    effort: &ReasoningEffort,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    if !is_bedrock_anthropic(model) {
        warnings.push(ModelWarning::LossyEncode {
            field: "reasoning_effort".into(),
            detail: format!(
                "Bedrock model {model:?} is not in the Anthropic family — Bedrock has no \
                 thinking knob for non-Anthropic models; field dropped"
            ),
        });
        return;
    }
    let adaptive_only = is_bedrock_anthropic_adaptive_only(model);
    let thinking = match effort {
        ReasoningEffort::Off => json!({"type": "disabled"}),
        ReasoningEffort::Minimal => {
            warnings.push(ModelWarning::LossyEncode {
                field: "reasoning_effort".into(),
                detail: "Anthropic on Bedrock has no `Minimal` bucket — snapped to adaptive `low`"
                    .into(),
            });
            json!({"type": "adaptive", "effort": "low"})
        }
        ReasoningEffort::Low => {
            if adaptive_only {
                json!({"type": "adaptive", "effort": "low"})
            } else {
                json!({"type": "enabled", "budget_tokens": 1024})
            }
        }
        ReasoningEffort::Medium => {
            if adaptive_only {
                json!({"type": "adaptive", "effort": "medium"})
            } else {
                json!({"type": "enabled", "budget_tokens": 4096})
            }
        }
        ReasoningEffort::High => {
            if adaptive_only {
                json!({"type": "adaptive", "effort": "high"})
            } else {
                json!({"type": "enabled", "budget_tokens": 16384})
            }
        }
        ReasoningEffort::Auto => json!({"type": "adaptive"}),
        ReasoningEffort::VendorSpecific(literal) => {
            if adaptive_only {
                warnings.push(ModelWarning::LossyEncode {
                    field: "reasoning_effort".into(),
                    detail: format!(
                        "Bedrock-Anthropic {model} is adaptive-only — manual budget \
                         {literal:?} dropped; emitting `{{type:\"adaptive\"}}` instead"
                    ),
                });
                json!({"type": "adaptive"})
            } else if let Ok(budget) = literal.parse::<u32>() {
                json!({"type": "enabled", "budget_tokens": budget})
            } else {
                warnings.push(ModelWarning::LossyEncode {
                    field: "reasoning_effort".into(),
                    detail: format!(
                        "Bedrock-Anthropic vendor-specific reasoning_effort {literal:?} is not \
                         a numeric budget_tokens — falling through to `Medium`"
                    ),
                });
                json!({"type": "enabled", "budget_tokens": 4096})
            }
        }
    };
    let mut additional = body
        .remove("additionalModelRequestFields")
        .and_then(|v| match v {
            Value::Object(o) => Some(o),
            _ => None,
        })
        .unwrap_or_default(); // silent-fallback-ok: caller-initiated additionalModelRequestFields nesting — fresh empty Map when absent or non-object
    additional.insert("thinking".into(), thinking);
    body.insert(
        "additionalModelRequestFields".into(),
        Value::Object(additional),
    );
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
    // IR `parallel_tool_calls` has no native Bedrock Converse
    // toggle (the on-Anthropic field rides under `tool_choice` but
    // Converse does not surface it); emit a field-precise lossy
    // signal so the operator sees the drop.
    if request.parallel_tool_calls.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "parallel_tool_calls".into(),
            detail: "Bedrock Converse exposes no equivalent toggle — \
                     setting dropped on the wire"
                .into(),
        });
    }
    if let Some(user_id) = &request.end_user_id {
        if is_bedrock_anthropic(&request.model) {
            // Bedrock-Anthropic relays Anthropic's `metadata.user_id`
            // through `additionalModelRequestFields`, mirroring direct
            // Anthropic. Non-Anthropic Bedrock models (Llama, Nova,
            // Mistral) lack the channel — fall through to LossyEncode.
            let entry = body
                .entry("additionalModelRequestFields")
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(map) = entry.as_object_mut() {
                let metadata = map
                    .entry("metadata")
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Some(meta_map) = metadata.as_object_mut() {
                    meta_map.insert("user_id".into(), Value::String(user_id.clone()));
                }
            }
        } else {
            warnings.push(ModelWarning::LossyEncode {
                field: "end_user_id".into(),
                detail: "Bedrock Converse non-Anthropic models have no per-request end-user \
                         attribution channel — setting dropped"
                    .into(),
            });
        }
    }
    if request.seed.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "seed".into(),
            detail: "Bedrock Converse has no deterministic-sampling knob — setting dropped".into(),
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
    // The `cachePoint` marker goes AFTER the text block it should
    // cache (Bedrock's documented contract). `attach_cache_point`
    // shares the TTL coercion + warning logic with the message and
    // tool encoders below.
    let mut system_blocks: Vec<Value> = Vec::new();
    for (idx, block) in request.system.blocks().iter().enumerate() {
        system_blocks.push(json!({ "text": block.text.clone() }));
        attach_cache_point(
            &mut system_blocks,
            block.cache_control,
            || format!("system[{idx}]"),
            warnings,
        );
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
        let cache = content_part_cache_control(part);
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
            ContentPart::RedactedThinking { .. } => {
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "Bedrock Converse does not accept redacted_thinking blocks on \
                             user-role input; block dropped"
                        .into(),
                });
            }
        }
        attach_cache_point(&mut out, cache, path, warnings);
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
        let cache = content_part_cache_control(part);
        match part {
            ContentPart::Text { text, .. } => out.push(json!({ "text": text })),
            ContentPart::ToolUse {
                id, name, input, ..
            } => {
                out.push(json!({
                    "toolUse": {
                        "toolUseId": id,
                        "name": name,
                        "input": input,
                    },
                }));
            }
            ContentPart::Thinking {
                text,
                provider_echoes,
                ..
            } => {
                let mut inner = Map::new();
                inner.insert("text".into(), Value::String(text.clone()));
                if let Some(sig) = ProviderEchoSnapshot::find_in(provider_echoes, PROVIDER_KEY)
                    .and_then(|snap| snap.payload_str("signature"))
                {
                    inner.insert("signature".into(), Value::String(sig.to_owned()));
                }
                let mut reasoning = Map::new();
                reasoning.insert("reasoningText".into(), Value::Object(inner));
                if let Some(redacted) = ProviderEchoSnapshot::find_in(provider_echoes, PROVIDER_KEY)
                    .and_then(|e| e.payload_str("redacted_content"))
                {
                    reasoning.insert("redactedContent".into(), Value::String(redacted.to_owned()));
                }
                out.push(json!({ "reasoningContent": Value::Object(reasoning) }));
            }
            ContentPart::RedactedThinking { provider_echoes } => {
                let Some(redacted) = ProviderEchoSnapshot::find_in(provider_echoes, PROVIDER_KEY)
                    .and_then(|e| e.payload_str("redacted_content"))
                else {
                    warnings.push(ModelWarning::LossyEncode {
                        field: path(),
                        detail: "redacted_thinking part missing 'bedrock-converse' \
                                 provider_echo with 'redacted_content' payload; block dropped"
                            .into(),
                    });
                    continue;
                };
                out.push(json!({
                    "reasoningContent": {
                        "redactedContent": redacted,
                    }
                }));
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
        attach_cache_point(&mut out, cache, path, warnings);
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
        let path = || format!("messages[{msg_idx}].content[{part_idx}]");
        let cache = content_part_cache_control(part);
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
                field: path(),
                detail: "non-tool_result part on Role::Tool dropped".into(),
            });
        }
        attach_cache_point(&mut out, cache, path, warnings);
    }
    out
}

/// Read the optional `cache_control` field from any `ContentPart`
/// variant. Variants the IR documents as carrying a cache directive
/// (text / image / audio / video / document / thinking / citation /
/// tool_result) return their stored value; variants that are model
/// output and emitted fresh per turn (tool_use / image_output /
/// audio_output) carry no directive and return `None`.
const fn content_part_cache_control(part: &ContentPart) -> Option<crate::ir::CacheControl> {
    match part {
        ContentPart::Text { cache_control, .. }
        | ContentPart::Image { cache_control, .. }
        | ContentPart::Audio { cache_control, .. }
        | ContentPart::Video { cache_control, .. }
        | ContentPart::Document { cache_control, .. }
        | ContentPart::Thinking { cache_control, .. }
        | ContentPart::Citation { cache_control, .. }
        | ContentPart::ToolResult { cache_control, .. } => *cache_control,
        ContentPart::ToolUse { .. }
        | ContentPart::ImageOutput { .. }
        | ContentPart::AudioOutput { .. }
        | ContentPart::RedactedThinking { .. } => None,
    }
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
        attach_cache_point(
            &mut arr,
            t.cache_control,
            || format!("tools[{idx}]"),
            warnings,
        );
    }
    Value::Array(arr)
}

/// Push a Bedrock Converse `cachePoint` marker after `out`'s most
/// recent block when `cache` is set. Emits a `LossyEncode` warning
/// when the IR's TTL diverges from Bedrock's vendor default — the
/// `cachePoint` block has no TTL knob (cache lifetime is set per
/// model server-side; 5-minute default on Converse, 1h needs the
/// InvokeModel API). The `field` prefix locates the originating IR
/// site for operator diagnostics.
fn attach_cache_point(
    out: &mut Vec<Value>,
    cache: Option<crate::ir::CacheControl>,
    field: impl FnOnce() -> String,
    warnings: &mut Vec<ModelWarning>,
) {
    let Some(cache) = cache else {
        return;
    };
    if cache.ttl != crate::ir::CacheTtl::FiveMinutes {
        warnings.push(ModelWarning::LossyEncode {
            field: format!("{}.cache_control.ttl", field()),
            detail: format!(
                "Bedrock cachePoint has no TTL knob — IR ttl `{:?}` coerced to vendor default",
                cache.ttl
            ),
        });
    }
    out.push(json!({ "cachePoint": { "type": "default" } }));
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
        ContentPart::RedactedThinking { .. } => "redacted_thinking",
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
        if let Some(reasoning) = part.get("reasoningContent") {
            let text = reasoning
                .get("reasoningText")
                .and_then(|t| t.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("") // silent-fallback-ok: reasoningContent without reasoningText.text → empty body; downstream is_empty() guard suppresses the part
                .to_owned();
            let signature = reasoning
                .get("reasoningText")
                .and_then(|t| t.get("signature"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            let redacted = reasoning
                .get("redactedContent")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let mut payload = Map::new();
            if let Some(s) = &signature {
                payload.insert("signature".into(), Value::String(s.clone()));
            }
            if let Some(r) = &redacted {
                payload.insert("redacted_content".into(), Value::String(r.clone()));
            }
            let provider_echoes = if payload.is_empty() {
                Vec::new()
            } else {
                vec![ProviderEchoSnapshot::new(
                    PROVIDER_KEY,
                    Value::Object(payload),
                )]
            };
            if text.is_empty() && signature.is_none() && redacted.is_some() {
                parts.push(ContentPart::RedactedThinking { provider_echoes });
            } else if !text.is_empty() || !provider_echoes.is_empty() {
                parts.push(ContentPart::Thinking {
                    text,
                    cache_control: None,
                    provider_echoes,
                });
            }
            continue;
        }
        if let Some(tool_use) = part.get("toolUse") {
            let id = str_field(tool_use, "toolUseId").to_owned();
            let name = str_field(tool_use, "name").to_owned();
            let input = tool_use.get("input").cloned().unwrap_or_else(|| json!({})); // silent-fallback-ok: toolUse without input = empty-args call (vendor sometimes omits when schema has no required fields)
            parts.push(ContentPart::ToolUse {
                id,
                name,
                input,
                provider_echoes: Vec::new(),
            });
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
        // Documented Converse stop reasons that don't map onto the
        // cross-vendor IR variants — surface verbatim under
        // `Other{raw}` so operators can branch on the typed tag
        // without parsing the raw string from a warning channel.
        Some(
            raw @ ("malformed_model_output"
            | "malformed_tool_use"
            | "model_context_window_exceeded"),
        ) => StopReason::Other {
            raw: raw.to_owned(),
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
