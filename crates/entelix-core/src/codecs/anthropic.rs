//! `AnthropicMessagesCodec` — IR ⇄ Anthropic Messages API
//! (`POST /v1/messages`).
//!
//! Wire format reference: <https://docs.anthropic.com/en/api/messages>.
//!
//! Notable mappings:
//!
//! - IR `system: Option<String>` → top-level `system` field.
//! - IR `Role::System` messages → flattened into `system` (Anthropic has no
//!   system role inside `messages`). Multiple system inputs are joined with
//!   blank lines.
//! - IR `Role::Tool` messages → `role: "user"` with `tool_result` blocks
//!   (Anthropic encodes tool replies as user-authored content).
//! - IR `ToolChoice` → `tool_choice` object (`auto` / `any` / `tool` /
//!   `none`).
//! - IR `Usage` cache fields ↔ Anthropic
//!   `cache_creation_input_tokens` / `cache_read_input_tokens`.
//!
//! Any feature the codec cannot preserve emits a
//! [`crate::ir::ModelWarning::LossyEncode`] (invariant 6).

#![allow(clippy::cast_possible_truncation)] // u64 → u32 token counts; saturates above

use std::collections::HashMap;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::codecs::codec::{BoxByteStream, BoxDeltaStream, Codec, EncodedRequest};
use crate::error::{Error, Result};
use crate::ir::{
    Capabilities, CitationSource, ContentPart, MediaSource, ModelRequest, ModelResponse,
    ModelWarning, OutputStrategy, ReasoningEffort, RefusalReason, ResponseFormat, Role, StopReason,
    ToolChoice, ToolKind, ToolResultContent, Usage,
};
use crate::rate_limit::RateLimitSnapshot;
use crate::stream::StreamDelta;

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Stateless codec for Anthropic's Messages API.
///
/// Construct once at agent build time and reuse — there is no internal
/// state, just static behaviour.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicMessagesCodec;

impl AnthropicMessagesCodec {
    /// Create a fresh codec instance.
    pub const fn new() -> Self {
        Self
    }
}

impl Codec for AnthropicMessagesCodec {
    fn name(&self) -> &'static str {
        "anthropic-messages"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        // Conservative full-feature defaults that match the documented
        // Messages API. Per-model precision (vision flag for haiku-3 vs
        // opus-4 etc.) lands in a follow-up slice.
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
            max_context_tokens: 200_000,
        }
    }

    fn auto_output_strategy(&self, _model: &str) -> crate::ir::OutputStrategy {
        // Anthropic's native `output_config.format = json_schema`
        // is newer than the tool-call surface and ships without a
        // `strict` toggle — every `strict=true` request emits
        // `LossyEncode { field: "response_format.strict" }`. The
        // forced-tool surface is more mature, more consistent with
        // every Anthropic version, and accepts arbitrary JSON
        // schemas without strict-mode constraints. Pick Tool until
        // Anthropic ships a native channel with parity.
        crate::ir::OutputStrategy::Tool
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

    fn decode_stream<'a>(
        &'a self,
        bytes: BoxByteStream<'a>,
        warnings_in: Vec<ModelWarning>,
    ) -> BoxDeltaStream<'a> {
        Box::pin(stream_anthropic_sse(bytes, warnings_in))
    }

    fn decode(&self, body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        let raw: Value = super::codec::parse_response_body(body, "Anthropic Messages")?;
        let mut warnings = warnings_in;

        let id = str_field(&raw, "id").to_owned();
        let model = str_field(&raw, "model").to_owned();
        let content = decode_content(&raw, &mut warnings);
        let stop_reason = decode_stop_reason(&raw, &mut warnings);
        let usage = decode_usage(&raw);

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
        let mut snapshot = RateLimitSnapshot::default();
        let mut populated = false;
        for (header_name, target) in [
            (
                "anthropic-ratelimit-requests-remaining",
                &mut snapshot.requests_remaining,
            ),
            (
                "anthropic-ratelimit-tokens-remaining",
                &mut snapshot.tokens_remaining,
            ),
        ] {
            if let Some(v) = headers.get(header_name).and_then(|h| h.to_str().ok())
                && let Ok(parsed) = v.parse::<u64>()
            {
                *target = Some(parsed);
                snapshot.raw.insert(header_name.to_owned(), v.to_owned());
                populated = true;
            }
        }
        for (header_name, target) in [
            (
                "anthropic-ratelimit-requests-reset",
                &mut snapshot.requests_reset_at,
            ),
            (
                "anthropic-ratelimit-tokens-reset",
                &mut snapshot.tokens_reset_at,
            ),
        ] {
            if let Some(v) = headers.get(header_name).and_then(|h| h.to_str().ok())
                && let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(v)
            {
                *target = Some(parsed.with_timezone(&chrono::Utc));
                snapshot.raw.insert(header_name.to_owned(), v.to_owned());
                populated = true;
            }
        }
        populated.then_some(snapshot)
    }
}

// ── body / request helpers ─────────────────────────────────────────────────

fn build_body(request: &ModelRequest, streaming: bool) -> Result<(Value, Vec<ModelWarning>)> {
    if request.messages.is_empty() {
        return Err(Error::invalid_request(
            "Anthropic Messages requires at least one message",
        ));
    }

    // Anthropic Messages requires `max_tokens` — the wire field has
    // no default. Silently injecting one would mask "model truncated
    // unexpectedly" failures (the operator sees `stop_reason:
    // max_tokens` without realising the codec, not the caller, set
    // the cap). Invariant #15 — fail loud when the IR is missing a
    // mandatory vendor field.
    let max_tokens = request.max_tokens.ok_or_else(|| {
        Error::invalid_request(
            "Anthropic Messages requires max_tokens; \
             set ModelRequest::max_tokens explicitly",
        )
    })?;

    // Most encodes produce 0–1 warnings; pre-allocate one slot so the
    // first push doesn't reallocate.
    let mut warnings = Vec::with_capacity(1);
    let (system_value, wire_messages) = encode_messages(request, &mut warnings);

    let mut body = Map::new();
    body.insert("model".into(), Value::String(request.model.clone()));
    body.insert("messages".into(), Value::Array(wire_messages));
    body.insert("max_tokens".into(), json!(max_tokens));
    if let Some(value) = system_value {
        body.insert("system".into(), value);
    }
    if let Some(temp) = request.temperature {
        body.insert("temperature".into(), json!(temp));
    }
    if let Some(p) = request.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if !request.stop_sequences.is_empty() {
        body.insert(
            "stop_sequences".into(),
            json!(request.stop_sequences.clone()),
        );
    }
    if !request.tools.is_empty() {
        body.insert("tools".into(), encode_tools(&request.tools, &mut warnings));
        body.insert(
            "tool_choice".into(),
            encode_tool_choice(&request.tool_choice),
        );
    }
    if streaming {
        body.insert("stream".into(), Value::Bool(true));
    }
    if let Some(format) = &request.response_format {
        encode_anthropic_structured_output(format, &request.model, &mut body, &mut warnings)?;
    }
    if request.cache_key.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "cache_key".into(),
            detail: "Anthropic Messages has no `prompt_cache_key`-style routing field; \
                     per-block `cache_control` is the native caching channel"
                .into(),
        });
    }
    if request.cached_content.is_some() {
        warnings.push(ModelWarning::LossyEncode {
            field: "cached_content".into(),
            detail: "Anthropic Messages has no Gemini-style `cachedContents` reference; \
                     per-block `cache_control` is the native caching channel"
                .into(),
        });
    }
    apply_provider_extensions(request, &mut body, &mut warnings);
    Ok((Value::Object(body), warnings))
}

/// Read [`crate::ir::AnthropicExt`] and merge each set field into the
/// wire body. Foreign-vendor extensions surface as
/// [`ModelWarning::ProviderExtensionIgnored`] — the operator
/// expressed an intent the Anthropic format cannot honour.
fn apply_provider_extensions(
    request: &ModelRequest,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let ext = &request.provider_extensions;
    if let Some(anthropic) = &ext.anthropic {
        if let Some(disable) = anthropic.disable_parallel_tool_use
            && body.contains_key("tool_choice")
        {
            // Anthropic threads `disable_parallel_tool_use` inside
            // `tool_choice`. Mutate the value in place so the
            // operator's `tool_choice` selection survives.
            if let Some(tc) = body.get_mut("tool_choice").and_then(Value::as_object_mut) {
                tc.insert("disable_parallel_tool_use".into(), json!(disable));
            }
        }
        if let Some(user_id) = &anthropic.user_id {
            body.insert("metadata".into(), json!({"user_id": user_id}));
        }
    }
    if let Some(effort) = &request.reasoning_effort {
        encode_anthropic_thinking(&request.model, effort, body, warnings);
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
    if ext.bedrock.is_some() {
        warnings.push(ModelWarning::ProviderExtensionIgnored {
            vendor: "bedrock".into(),
        });
    }
}

/// Resolve [`OutputStrategy::Auto`] against the codec's preferred
/// dispatch shape, then emit either the native `output_config`
/// shape or the forced-tool shape into `body`. `Prompted` is
/// rejected (1.1 — currently unsupported on every codec).
fn encode_anthropic_structured_output(
    format: &ResponseFormat,
    model: &str,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) -> Result<()> {
    let strategy = resolve_output_strategy(format.strategy, model);
    match strategy {
        OutputStrategy::Native => {
            // Anthropic native structured outputs — `output_config`
            // at the request root, raw JSON Schema (no wrapper, no
            // strict toggle). Anthropic always strict-validates
            // when the field is set.
            body.insert(
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
                    detail: "Anthropic always strict-validates structured output; \
                         the strict=false request was approximated"
                        .into(),
                });
            }
        }
        OutputStrategy::Tool => {
            // Forced single tool call carrying the target schema.
            // Mature surface, parity with every Anthropic version,
            // accepts arbitrary JSON schemas without strict-mode
            // constraints. The codec injects the synthetic tool
            // and a `tool_choice` of `{type: "tool", name: ...}`
            // ahead of any operator-supplied tools so the model
            // emits exactly one `tool_use` block whose input
            // matches the target schema.
            let tool_name = format.json_schema.name.clone();
            let synthetic_tool = json!({
                "type": "custom",
                "name": tool_name,
                "description": format!(
                    "Emit the response as a JSON object matching the {tool_name} schema."
                ),
                "input_schema": format.json_schema.schema.clone(),
            });
            // Prepend the structured-output tool so the model sees
            // it first. Operator tools survive; the new
            // `tool_choice` overrides any prior selection because
            // structured-output dispatch demands a forced single
            // tool call.
            let tools = body.entry("tools").or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(arr) = tools {
                arr.insert(0, synthetic_tool);
            }
            body.insert(
                "tool_choice".into(),
                json!({
                    "type": "tool",
                    "name": format.json_schema.name,
                    // Disable parallel tool use so the model cannot
                    // emit multiple tool_use blocks alongside the
                    // structured-output call.
                    "disable_parallel_tool_use": true,
                }),
            );
            if !format.strict {
                // Tool-input-schema validation is enforced at
                // construction time by Anthropic regardless of the
                // strict flag — surface the loss.
                warnings.push(ModelWarning::LossyEncode {
                    field: "response_format.strict".into(),
                    detail: "Anthropic Tool-strategy structured output is always \
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
        OutputStrategy::Auto => {
            // Resolved above — unreachable. Defensive arm.
            return Err(Error::invalid_request(
                "OutputStrategy::Auto did not resolve — codec invariant violation",
            ));
        }
    }
    Ok(())
}

/// Resolve [`OutputStrategy::Auto`] to Anthropic's preferred
/// dispatch shape — currently `Tool` (the native `output_config`
/// channel ships without a strict toggle and is less mature than
/// the tool-call surface). The explicit per-variant arms keep the
/// resolver readable as the cross-vendor `OutputStrategy` enum
/// gains future variants — Clippy's `match_same_arms` would have
/// us merge the identity arms but that hides the explicit
/// per-variant intent (ADR-0079).
#[allow(clippy::match_same_arms)]
const fn resolve_output_strategy(strategy: OutputStrategy, _model: &str) -> OutputStrategy {
    match strategy {
        OutputStrategy::Auto => OutputStrategy::Tool,
        OutputStrategy::Native => OutputStrategy::Native,
        OutputStrategy::Tool => OutputStrategy::Tool,
        OutputStrategy::Prompted => OutputStrategy::Prompted,
    }
}

/// Anthropic Opus 4.7 budget tokens — adaptive-only, manual budget
/// rejected at encode time. Sonnet 4.6 / 4.5 / Haiku accept either
/// adaptive or explicit budget. The codec branches on the model
/// string prefix because Anthropic does not expose a wire-level
/// "this model is adaptive-only" signal — the constraint is
/// vendor-side request validation that surfaces as 4xx without
/// useful diagnostic context.
fn is_anthropic_adaptive_only(model: &str) -> bool {
    // Opus 4.7 is the only adaptive-only model in Anthropic's 2026
    // lineup. Future models that ship adaptive-only join this list.
    model.starts_with("claude-opus-4-7")
}

/// Translate the cross-vendor [`ReasoningEffort`] knob onto the
/// Anthropic Messages API `thinking` field. Per ADR-0078:
///
/// - `Off` → `{type:"disabled"}`
/// - `Minimal` → `{type:"adaptive", effort:"low"}` (LossyEncode —
///   Anthropic's smallest adaptive bucket; closer than `enabled`
///   with a sub-1024 budget which the vendor rejects)
/// - `Low` → `{type:"enabled", budget_tokens:1024}` (or adaptive on
///   Opus 4.7)
/// - `Medium` → `{type:"enabled", budget_tokens:4096}` (or adaptive
///   on Opus 4.7)
/// - `High` → `{type:"enabled", budget_tokens:16384}` (or adaptive
///   on Opus 4.7)
/// - `Auto` → `{type:"adaptive"}`
/// - `VendorSpecific(s)` — `s` parses as decimal `budget_tokens`
///   (Opus 4.7 rejects this with `Error::invalid_request`).
///   Non-numeric `s` emits `LossyEncode` and falls through to
///   `Medium`.
fn encode_anthropic_thinking(
    model: &str,
    effort: &ReasoningEffort,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ModelWarning>,
) {
    let adaptive_only = is_anthropic_adaptive_only(model);
    let thinking = match effort {
        ReasoningEffort::Off => {
            json!({"type": "disabled"})
        }
        ReasoningEffort::Minimal => {
            warnings.push(ModelWarning::LossyEncode {
                field: "reasoning_effort".into(),
                detail:
                    "Anthropic has no `Minimal` bucket — snapped to `{type:\"adaptive\", effort:\"low\"}`"
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
        ReasoningEffort::Auto => {
            json!({"type": "adaptive"})
        }
        ReasoningEffort::VendorSpecific(literal) => {
            if adaptive_only {
                // Opus 4.7 manual-budget rejection — encode-time
                // failure with a useful diagnostic. Falling back
                // silently would let the operator's intent (raw
                // budget tokens) hit the wire and surface as a
                // 4xx with vague upstream wording.
                warnings.push(ModelWarning::LossyEncode {
                    field: "reasoning_effort".into(),
                    detail: format!(
                        "Anthropic {model} is adaptive-only — manual budget '{literal}' \
                         dropped; emitting `{{type:\"adaptive\"}}` instead"
                    ),
                });
                json!({"type": "adaptive"})
            } else if let Ok(budget) = literal.parse::<u32>() {
                json!({"type": "enabled", "budget_tokens": budget})
            } else {
                warnings.push(ModelWarning::LossyEncode {
                    field: "reasoning_effort".into(),
                    detail: format!(
                        "Anthropic vendor-specific reasoning_effort {literal:?} is not a \
                         numeric budget_tokens — falling through to `Medium`"
                    ),
                });
                json!({"type": "enabled", "budget_tokens": 4096})
            }
        }
    };
    body.insert("thinking".into(), thinking);
}

fn finalize_request(body: &Value, warnings: Vec<ModelWarning>) -> Result<EncodedRequest> {
    let bytes = serde_json::to_vec(body)?;
    let mut encoded = EncodedRequest::post_json("/v1/messages", Bytes::from(bytes));
    encoded.headers.insert(
        http::HeaderName::from_static("anthropic-version"),
        http::HeaderValue::from_static(ANTHROPIC_VERSION),
    );
    encoded.warnings = warnings;
    Ok(encoded)
}

// ── encode helpers ─────────────────────────────────────────────────────────

fn encode_messages(
    request: &ModelRequest,
    warnings: &mut Vec<ModelWarning>,
) -> (Option<Value>, Vec<Value>) {
    // Two collection paths:
    // - If any IR system block carries cache_control, we emit the
    //   array form (preserving per-block cache directives).
    // - Otherwise the simple string form is sufficient.
    let mut system_blocks: Vec<(String, Option<crate::ir::CacheControl>)> = request
        .system
        .blocks()
        .iter()
        .map(|b| (b.text.clone(), b.cache_control))
        .collect();
    let mut wire_messages = Vec::with_capacity(request.messages.len());

    for (idx, msg) in request.messages.iter().enumerate() {
        match msg.role {
            Role::System => {
                let mut lossy_non_text = false;
                let text = msg
                    .content
                    .iter()
                    .filter_map(|part| {
                        if let ContentPart::Text { text, .. } = part {
                            Some(text.clone())
                        } else {
                            lossy_non_text = true;
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if lossy_non_text {
                    warnings.push(ModelWarning::LossyEncode {
                        field: format!("messages[{idx}].content"),
                        detail: "non-text parts dropped from system message (Anthropic has no \
                                 system role)"
                            .into(),
                    });
                }
                if !text.is_empty() {
                    system_blocks.push((text, None));
                }
            }
            Role::User | Role::Assistant | Role::Tool => {
                let role_str = match msg.role {
                    Role::Assistant => "assistant",
                    _ => "user",
                };
                let content_array = encode_content_parts(&msg.content, warnings, idx);
                let mut entry = Map::new();
                entry.insert("role".into(), Value::String(role_str.into()));
                entry.insert("content".into(), Value::Array(content_array));
                wire_messages.push(Value::Object(entry));
            }
        }
    }

    // Emit the array form when any block has cache_control set;
    // otherwise the simpler string form is enough — minimizing
    // wire bytes when callers don't need caching.
    let any_cached = system_blocks.iter().any(|(_, cc)| cc.is_some());
    let system_value = if system_blocks.is_empty() {
        None
    } else if any_cached {
        let array: Vec<Value> = system_blocks
            .into_iter()
            .map(|(text, cc)| {
                let mut obj = Map::new();
                obj.insert("type".into(), Value::String("text".into()));
                obj.insert("text".into(), Value::String(text));
                if let Some(cache) = cc {
                    obj.insert("cache_control".into(), encode_cache_control(cache));
                }
                Value::Object(obj)
            })
            .collect();
        Some(Value::Array(array))
    } else {
        Some(Value::String(
            system_blocks
                .into_iter()
                .map(|(text, _)| text)
                .collect::<Vec<_>>()
                .join("\n\n"),
        ))
    };
    (system_value, wire_messages)
}

#[allow(clippy::too_many_lines)] // dispatch over every ContentPart variant
fn encode_content_parts(
    parts: &[ContentPart],
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
) -> Vec<Value> {
    let mut out = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let path = || format!("messages[{msg_idx}].content[{part_idx}]");
        match part {
            ContentPart::Text {
                text,
                cache_control,
            } => {
                let mut block = json_block("text", &[("text", Value::String(text.clone()))]);
                attach_cache_control(&mut block, cache_control.as_ref());
                out.push(Value::Object(block));
            }
            ContentPart::Image {
                source,
                cache_control,
            } => {
                let mut block = json_block(
                    "image",
                    &[("source", encode_media_source_anthropic(source))],
                );
                attach_cache_control(&mut block, cache_control.as_ref());
                out.push(Value::Object(block));
            }
            ContentPart::Audio { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Anthropic Messages does not accept audio inputs; block dropped".into(),
            }),
            ContentPart::Video { .. } => warnings.push(ModelWarning::LossyEncode {
                field: path(),
                detail: "Anthropic Messages does not accept video inputs; block dropped".into(),
            }),
            ContentPart::Document {
                source,
                name,
                cache_control,
            } => {
                let mut block = Map::new();
                block.insert("type".into(), Value::String("document".into()));
                block.insert("source".into(), encode_media_source_anthropic(source));
                if let Some(title) = name {
                    block.insert("title".into(), Value::String(title.clone()));
                }
                attach_cache_control(&mut block, cache_control.as_ref());
                out.push(Value::Object(block));
            }
            ContentPart::Thinking {
                text,
                signature,
                cache_control,
            } => {
                let mut block = Map::new();
                block.insert("type".into(), Value::String("thinking".into()));
                block.insert("thinking".into(), Value::String(text.clone()));
                if let Some(sig) = signature {
                    block.insert("signature".into(), Value::String(sig.clone()));
                }
                attach_cache_control(&mut block, cache_control.as_ref());
                out.push(Value::Object(block));
            }
            ContentPart::Citation {
                snippet,
                source,
                cache_control,
            } => {
                // Anthropic carries citations inline on `text` blocks; round-tripping
                // a standalone Citation IR variant means re-emitting the cited text
                // with a `citations` array attached.
                let citation_json = match source {
                    CitationSource::Url { url, title } => {
                        let mut o = Map::new();
                        o.insert(
                            "type".into(),
                            Value::String("web_search_result_location".into()),
                        );
                        o.insert("url".into(), Value::String(url.clone()));
                        if let Some(t) = title {
                            o.insert("title".into(), Value::String(t.clone()));
                        }
                        Value::Object(o)
                    }
                    CitationSource::Document {
                        document_index,
                        title,
                    } => {
                        let mut o = Map::new();
                        o.insert("type".into(), Value::String("char_location".into()));
                        o.insert("document_index".into(), json!(*document_index));
                        if let Some(t) = title {
                            o.insert("document_title".into(), Value::String(t.clone()));
                        }
                        Value::Object(o)
                    }
                };
                let mut block = Map::new();
                block.insert("type".into(), Value::String("text".into()));
                block.insert("text".into(), Value::String(snippet.clone()));
                block.insert("citations".into(), Value::Array(vec![citation_json]));
                attach_cache_control(&mut block, cache_control.as_ref());
                out.push(Value::Object(block));
            }
            ContentPart::ToolUse { id, name, input } => out.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            })),
            ContentPart::ToolResult {
                tool_use_id,
                name: _,
                content,
                is_error,
                cache_control,
            } => out.push(encode_tool_result(
                tool_use_id,
                content,
                *is_error,
                cache_control.as_ref(),
                warnings,
                msg_idx,
                part_idx,
            )),
            ContentPart::ImageOutput { .. } | ContentPart::AudioOutput { .. } => {
                // Assistant-produced image/audio is output-only; no
                // Anthropic input shape consumes it. Drop with a
                // LossyEncode warning so a multi-turn replay that
                // happens to surface the part is not silently
                // mis-encoded as user input.
                warnings.push(ModelWarning::LossyEncode {
                    field: path(),
                    detail: "Anthropic Messages does not accept assistant-produced \
                             image / audio output as input — block dropped"
                        .into(),
                });
            }
        }
    }
    out
}

/// Build a JSON object block of `{type: <kind>, ...fields}`.
fn json_block(kind: &str, fields: &[(&str, Value)]) -> Map<String, Value> {
    let mut block = Map::new();
    block.insert("type".into(), Value::String(kind.into()));
    for (k, v) in fields {
        block.insert((*k).to_owned(), v.clone());
    }
    block
}

/// Insert Anthropic's `cache_control` block. The wire shape is
/// always `{type: "ephemeral"}` plus an optional sibling `ttl: "1h"`
/// for the premium tier — `type` never carries the TTL string.
fn attach_cache_control(block: &mut Map<String, Value>, cache: Option<&crate::ir::CacheControl>) {
    if let Some(cache) = cache {
        block.insert("cache_control".into(), encode_cache_control(*cache));
    }
}

/// Render a [`crate::ir::CacheControl`] as Anthropic's
/// `cache_control` JSON value. Single source of truth for the wire
/// shape — every encode path (system blocks, content blocks, tool
/// blocks, tool_result blocks) goes through here.
fn encode_cache_control(cache: crate::ir::CacheControl) -> Value {
    let mut obj = Map::new();
    obj.insert("type".into(), Value::String("ephemeral".into()));
    if let Some(ttl) = cache.ttl.wire_ttl_field() {
        obj.insert("ttl".into(), Value::String(ttl.into()));
    }
    Value::Object(obj)
}

fn encode_media_source_anthropic(source: &MediaSource) -> Value {
    match source {
        MediaSource::Url { url, .. } => json!({
            "type": "url",
            "url": url,
        }),
        MediaSource::Base64 { media_type, data } => json!({
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }),
        MediaSource::FileId { id, .. } => json!({
            "type": "file",
            "file_id": id,
        }),
    }
}

fn encode_tool_result(
    tool_use_id: &str,
    content: &ToolResultContent,
    is_error: bool,
    cache_control: Option<&crate::ir::CacheControl>,
    warnings: &mut Vec<ModelWarning>,
    msg_idx: usize,
    part_idx: usize,
) -> Value {
    let content_json = match content {
        ToolResultContent::Text(s) => Value::String(s.clone()),
        ToolResultContent::Json(v) => {
            warnings.push(ModelWarning::LossyEncode {
                field: format!("messages[{msg_idx}].content[{part_idx}]"),
                detail: "tool_result Json payload stringified for Anthropic wire format".into(),
            });
            Value::String(v.to_string())
        }
    };
    let mut block = Map::new();
    block.insert("type".into(), Value::String("tool_result".into()));
    block.insert("tool_use_id".into(), Value::String(tool_use_id.into()));
    block.insert("content".into(), content_json);
    if is_error {
        block.insert("is_error".into(), Value::Bool(true));
    }
    attach_cache_control(&mut block, cache_control);
    Value::Object(block)
}

fn encode_tools(tools: &[crate::ir::ToolSpec], warnings: &mut Vec<ModelWarning>) -> Value {
    let mut arr: Vec<Value> = Vec::with_capacity(tools.len());
    for (idx, t) in tools.iter().enumerate() {
        let mut obj = match &t.kind {
            ToolKind::Function { input_schema } => {
                let mut o = Map::new();
                o.insert("name".into(), Value::String(t.name.clone()));
                o.insert("description".into(), Value::String(t.description.clone()));
                o.insert("input_schema".into(), input_schema.clone());
                o
            }
            ToolKind::WebSearch {
                max_uses,
                allowed_domains,
            } => {
                let mut o = Map::new();
                o.insert("type".into(), Value::String("web_search_20250305".into()));
                o.insert("name".into(), Value::String(t.name.clone()));
                if let Some(n) = max_uses {
                    o.insert("max_uses".into(), json!(*n));
                }
                if !allowed_domains.is_empty() {
                    o.insert("allowed_domains".into(), json!(allowed_domains));
                }
                o
            }
            ToolKind::Computer {
                display_width,
                display_height,
            } => {
                let mut o = Map::new();
                o.insert("type".into(), Value::String("computer_20250124".into()));
                o.insert("name".into(), Value::String(t.name.clone()));
                o.insert("display_width_px".into(), json!(*display_width));
                o.insert("display_height_px".into(), json!(*display_height));
                o
            }
            ToolKind::TextEditor => {
                let mut o = Map::new();
                o.insert("type".into(), Value::String("text_editor_20250124".into()));
                o.insert("name".into(), Value::String(t.name.clone()));
                o
            }
            ToolKind::Bash => {
                let mut o = Map::new();
                o.insert("type".into(), Value::String("bash_20250124".into()));
                o.insert("name".into(), Value::String(t.name.clone()));
                o
            }
            ToolKind::CodeExecution => {
                let mut o = Map::new();
                o.insert(
                    "type".into(),
                    Value::String("code_execution_20250522".into()),
                );
                o.insert("name".into(), Value::String(t.name.clone()));
                o
            }
            ToolKind::McpConnector {
                name,
                server_url,
                authorization_token,
            } => {
                let mut o = Map::new();
                o.insert("type".into(), Value::String("mcp".into()));
                o.insert("name".into(), Value::String(name.clone()));
                o.insert("server_url".into(), Value::String(server_url.clone()));
                if let Some(token) = authorization_token {
                    o.insert("authorization_token".into(), Value::String(token.clone()));
                }
                o
            }
            ToolKind::Memory => {
                let mut o = Map::new();
                o.insert("type".into(), Value::String("memory_20250818".into()));
                o.insert("name".into(), Value::String(t.name.clone()));
                o
            }
            ToolKind::FileSearch { .. } | ToolKind::CodeInterpreter | ToolKind::ImageGeneration => {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("tools[{idx}]"),
                    detail: "Anthropic does not natively support OpenAI-only built-ins \
                             (file_search / code_interpreter / image_generation) — tool dropped"
                        .into(),
                });
                continue;
            }
        };
        attach_cache_control(&mut obj, t.cache_control.as_ref());
        arr.push(Value::Object(obj));
    }
    Value::Array(arr)
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({ "type": "auto" }),
        ToolChoice::Required => json!({ "type": "any" }),
        ToolChoice::Specific { name } => json!({ "type": "tool", "name": name }),
        ToolChoice::None => json!({ "type": "none" }),
    }
}

// ── decode helpers ─────────────────────────────────────────────────────────

fn decode_content(raw: &Value, warnings: &mut Vec<ModelWarning>) -> Vec<ContentPart> {
    let Some(arr) = raw.get("content").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for (idx, block) in arr.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = str_field(block, "text").to_owned();
                // If the block carries `citations`, emit them as separate
                // ContentPart::Citation entries followed by the underlying
                // text — preserves provenance while keeping the IR variant
                // discriminated.
                if let Some(citations) = block.get("citations").and_then(Value::as_array) {
                    for c in citations {
                        if let Some(source) = decode_citation_source(c) {
                            out.push(ContentPart::Citation {
                                snippet: text.clone(),
                                source,
                                cache_control: None,
                            });
                        }
                    }
                }
                if !text.is_empty() {
                    out.push(ContentPart::text(text));
                }
            }
            Some("thinking") => {
                let thinking_text = str_field(block, "thinking").to_owned();
                let signature = block
                    .get("signature")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                out.push(ContentPart::Thinking {
                    text: thinking_text,
                    signature,
                    cache_control: None,
                });
            }
            Some("tool_use") => {
                let id = str_field(block, "id").to_owned();
                let name = str_field(block, "name").to_owned();
                let input = block.get("input").cloned().unwrap_or_else(|| json!({})); // silent-fallback-ok: tool_use without args = empty-args call (vendor sometimes omits)
                out.push(ContentPart::ToolUse { id, name, input });
            }
            Some(other) => {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("response.content[{idx}]"),
                    detail: format!("unknown content block type '{other}' dropped"),
                });
            }
            None => {
                warnings.push(ModelWarning::LossyEncode {
                    field: format!("response.content[{idx}]"),
                    detail: "content block missing 'type' field".into(),
                });
            }
        }
    }
    out
}

fn decode_citation_source(c: &Value) -> Option<CitationSource> {
    match c.get("type").and_then(Value::as_str)? {
        "web_search_result_location" => Some(CitationSource::Url {
            url: str_field(c, "url").to_owned(),
            title: c.get("title").and_then(Value::as_str).map(str::to_owned),
        }),
        "char_location" | "page_location" | "content_block_location" => {
            // Invariant #15 — `document_index` is the
            // citation's load-bearing pointer; defaulting to 0
            // would silently rewrite "vendor failed to identify the
            // document" as "the first document". Drop the citation
            // entirely when the index is missing or unparseable so
            // operators see the absence rather than a wrong link.
            let document_index = c
                .get("document_index")
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())?;
            Some(CitationSource::Document {
                document_index,
                title: c
                    .get("document_title")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            })
        }
        _ => None,
    }
}

fn decode_stop_reason(raw: &Value, warnings: &mut Vec<ModelWarning>) -> StopReason {
    match raw.get("stop_reason").and_then(Value::as_str) {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence {
            sequence: str_field(raw, "stop_sequence").to_owned(),
        },
        Some("tool_use") => StopReason::ToolUse,
        Some("refusal") => StopReason::Refusal {
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
            // Invariant #15 — silent EndTurn fallback was masking
            // truncated stream payloads from callers. Record as
            // Other + warning so observability sees the loss.
            warnings.push(ModelWarning::LossyEncode {
                field: "stop_reason".into(),
                detail: "Anthropic Messages payload carried no stop_reason — \
                         IR records `Other{raw:\"missing\"}`"
                    .into(),
            });
            StopReason::Other {
                raw: "missing".to_owned(),
            }
        }
    }
}

fn decode_usage(raw: &Value) -> Usage {
    let usage = raw.get("usage");
    Usage {
        input_tokens: u_field(usage, "input_tokens"),
        output_tokens: u_field(usage, "output_tokens"),
        cached_input_tokens: u_field(usage, "cache_read_input_tokens"),
        cache_creation_input_tokens: u_field(usage, "cache_creation_input_tokens"),
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

// ── SSE streaming parser ───────────────────────────────────────────────────

/// Per-block book-keeping the SSE parser needs to produce the right
/// `StreamDelta` on `content_block_stop`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse,
}

#[allow(tail_expr_drop_order, clippy::too_many_lines)]
fn stream_anthropic_sse(
    bytes: BoxByteStream<'_>,
    warnings_in: Vec<ModelWarning>,
) -> impl futures::Stream<Item = Result<StreamDelta>> + Send + '_ {
    async_stream::stream! {
        let mut bytes = bytes;
        let mut buf: Vec<u8> = Vec::new();
        let mut blocks: HashMap<u64, BlockKind> = HashMap::new();
        let mut last_stop_reason = StopReason::EndTurn;
        let mut accumulated_usage = Usage::default();
        let mut warnings_emitted = false;

        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(b) => buf.extend_from_slice(&b),
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
            // Emit any accumulated encode-time warnings on first byte.
            if !warnings_emitted {
                warnings_emitted = true;
                for w in &warnings_in {
                    yield Ok(StreamDelta::Warning(w.clone()));
                }
            }
            // Process every complete `\n\n`-terminated SSE frame.
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
                        "Anthropic stream: malformed event payload: {payload}"
                    )));
                    return;
                };
                let event_type = event.get("type").and_then(Value::as_str).unwrap_or(""); // silent-fallback-ok: missing event type → no-match fallthrough
                match event_type {
                    "message_start" => {
                        let message = event.get("message").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates through child .get() chain as None
                        let id = str_field(message, "id").to_owned();
                        let model = str_field(message, "model").to_owned();
                        if let Some(usage) = message.get("usage") {
                            accumulated_usage.input_tokens = u_field(Some(usage), "input_tokens");
                            accumulated_usage.cached_input_tokens =
                                u_field(Some(usage), "cache_read_input_tokens");
                            accumulated_usage.cache_creation_input_tokens =
                                u_field(Some(usage), "cache_creation_input_tokens");
                        }
                        yield Ok(StreamDelta::Start { id, model });
                    }
                    "content_block_start" => {
                        let idx = if let Some(n) = event.get("index").and_then(Value::as_u64) {
                            n
                        } else {
                            yield Ok(StreamDelta::Warning(ModelWarning::LossyEncode {
                                field: "stream.content_block_start.index".into(),
                                detail: "Anthropic SSE event missing spec-mandated 'index' field; falling back to slot 0 to keep stream parser progressing".into(),
                            }));
                            0
                        };
                        let block = event.get("content_block").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
                        match block.get("type").and_then(Value::as_str) {
                            Some("text") => {
                                blocks.insert(idx, BlockKind::Text);
                                if let Some(text) = block.get("text").and_then(Value::as_str)
                                    && !text.is_empty()
                                {
                                    yield Ok(StreamDelta::TextDelta {
                                        text: text.to_owned(),
                                    });
                                }
                            }
                            Some("thinking") => {
                                blocks.insert(idx, BlockKind::Thinking);
                                let text = block
                                    .get("thinking")
                                    .and_then(Value::as_str)
                                    .unwrap_or("") // silent-fallback-ok: missing thinking text → empty body; downstream is_empty() guard suppresses the StreamDelta
                                    .to_owned();
                                let signature = block
                                    .get("signature")
                                    .and_then(Value::as_str)
                                    .map(str::to_owned);
                                if !text.is_empty() || signature.is_some() {
                                    yield Ok(StreamDelta::ThinkingDelta { text, signature });
                                }
                            }
                            Some("tool_use") => {
                                blocks.insert(idx, BlockKind::ToolUse);
                                let id = str_field(block, "id").to_owned();
                                let name = str_field(block, "name").to_owned();
                                yield Ok(StreamDelta::ToolUseStart { id, name });
                            }
                            other => {
                                yield Ok(StreamDelta::Warning(ModelWarning::LossyEncode {
                                    field: format!("stream.content_block_start[{idx}]"),
                                    detail: format!(
                                        "unsupported block type {other:?} dropped"
                                    ),
                                }));
                            }
                        }
                    }
                    "content_block_delta" => {
                        let delta = event.get("delta").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates as None
                        match delta.get("type").and_then(Value::as_str) {
                            Some("text_delta") => {
                                if let Some(text) = delta.get("text").and_then(Value::as_str) {
                                    yield Ok(StreamDelta::TextDelta {
                                        text: text.to_owned(),
                                    });
                                }
                            }
                            Some("thinking_delta") => {
                                if let Some(text) = delta.get("thinking").and_then(Value::as_str) {
                                    yield Ok(StreamDelta::ThinkingDelta {
                                        text: text.to_owned(),
                                        signature: None,
                                    });
                                }
                            }
                            Some("signature_delta") => {
                                if let Some(sig) = delta.get("signature").and_then(Value::as_str) {
                                    yield Ok(StreamDelta::ThinkingDelta {
                                        text: String::new(),
                                        signature: Some(sig.to_owned()),
                                    });
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) =
                                    delta.get("partial_json").and_then(Value::as_str)
                                {
                                    yield Ok(StreamDelta::ToolUseInputDelta {
                                        partial_json: partial.to_owned(),
                                    });
                                }
                            }
                            other => {
                                yield Ok(StreamDelta::Warning(ModelWarning::LossyEncode {
                                    field: "stream.content_block_delta".into(),
                                    detail: format!(
                                        "unsupported delta type {other:?} dropped"
                                    ),
                                }));
                            }
                        }
                    }
                    "content_block_stop" => {
                        let idx = if let Some(n) = event.get("index").and_then(Value::as_u64) {
                            n
                        } else {
                            yield Ok(StreamDelta::Warning(ModelWarning::LossyEncode {
                                field: "stream.content_block_stop.index".into(),
                                detail: "Anthropic SSE event missing spec-mandated 'index' field; falling back to slot 0 (mirrors the content_block_start handler)".into(),
                            }));
                            0
                        };
                        if matches!(blocks.remove(&idx), Some(BlockKind::ToolUse)) {
                            yield Ok(StreamDelta::ToolUseStop);
                        }
                    }
                    "message_delta" => {
                        if let Some(delta) = event.get("delta")
                            && let Some(reason) =
                                delta.get("stop_reason").and_then(Value::as_str)
                        {
                            // Match the non-streaming `decode_stop_reason` exhaustiveness:
                            // every documented Anthropic stop reason maps explicitly,
                            // and any unknown raw value yields a `UnknownStopReason`
                            // warning delta so streaming and non-streaming clients see
                            // the same observability signal when a vendor extension
                            // arrives. Also extract `stop_sequence` so the matched
                            // sequence string is not dropped.
                            last_stop_reason = match reason {
                                "end_turn" => StopReason::EndTurn,
                                "max_tokens" => StopReason::MaxTokens,
                                "stop_sequence" => StopReason::StopSequence {
                                    sequence: delta
                                        .get("stop_sequence")
                                        .and_then(Value::as_str)
                                        .unwrap_or_default() // silent-fallback-ok: vendor reported stop_sequence stop reason without echoing the matched sequence string; "" preserves the StopReason variant choice
                                        .to_owned(),
                                },
                                "tool_use" => StopReason::ToolUse,
                                "refusal" => StopReason::Refusal {
                                    reason: RefusalReason::Safety,
                                },
                                other => {
                                    yield Ok(StreamDelta::Warning(
                                        ModelWarning::UnknownStopReason {
                                            raw: other.to_owned(),
                                        },
                                    ));
                                    StopReason::Other {
                                        raw: other.to_owned(),
                                    }
                                }
                            };
                        }
                        if let Some(usage) = event.get("usage") {
                            accumulated_usage.output_tokens =
                                u_field(Some(usage), "output_tokens");
                            yield Ok(StreamDelta::Usage(accumulated_usage.clone()));
                        }
                    }
                    "message_stop" => {
                        yield Ok(StreamDelta::Stop {
                            stop_reason: last_stop_reason.clone(),
                        });
                    }
                    "ping" => {
                        // Heartbeat — ignore.
                    }
                    "error" => {
                        let err = event.get("error").unwrap_or(&Value::Null); // silent-fallback-ok: nested accessor — Null propagates through child .get() chain as None
                        let kind = str_field(err, "type");
                        let message = str_field(err, "message");
                        yield Err(Error::provider_network(format!(
                            "Anthropic stream error ({kind}): {message}"
                        )));
                        return;
                    }
                    other => {
                        yield Ok(StreamDelta::Warning(ModelWarning::LossyEncode {
                            field: "stream.event".into(),
                            detail: format!("unknown SSE event type {other:?} ignored"),
                        }));
                    }
                }
            }
        }
    }
}

/// Find the byte offset of the first `\n\n` in `buf`. Also handles
/// `\r\n\r\n` because SSE uses CRLF in the wild.
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

/// Pull the JSON payload out of an SSE frame, concatenating multi-line
/// `data:` lines per the spec. Returns `None` for frames with no
/// `data:` line.
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
