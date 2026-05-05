//! `ResponseFormat` — vendor-agnostic structured-output IR.
//!
//! Per ADR-0024 §5: enters IR because OpenAI Chat / OpenAI
//! Responses / Gemini all natively support a JSON-Schema-shaped
//! response constraint. Anthropic does not natively, so codecs
//! synthesize a tool-use shim and emit
//! [`crate::ir::ModelWarning::LossyEncode`].
//!
//! ## Validation discipline
//!
//! [`JsonSchemaSpec::new`] performs a minimal sanity check at
//! construction (non-empty name; schema must be a JSON object).
//! Full JSON Schema validation is deferred to the codec encode
//! path where it has access to the vendor's validation rules
//! (some vendors require strict mode, draft 2020-12, etc.). Per
//! ADR-0024 §heuristic-risk: callers receive an `Err` at
//! construction for the obvious failures, not at first-call time.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// JSON Schema specification — a (name, schema) pair carried
/// through the IR and routed to vendor-canonical structured-output
/// channels.
///
/// Construct via [`Self::new`] (validates inputs) or via
/// `serde_json::from_str` (deserialization is unchecked — the
/// codec validates at encode time).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct JsonSchemaSpec {
    /// Caller-chosen identifier for the schema. Surfaces in OTel
    /// span attributes (`gen_ai.response_format.name`) and in the
    /// vendor wire format where applicable (OpenAI requires this).
    pub name: String,
    /// JSON Schema document. Must be a JSON object at the top
    /// level (per JSON Schema spec); [`Self::new`] rejects other
    /// shapes.
    pub schema: Value,
}

impl JsonSchemaSpec {
    /// Validated constructor. Returns [`Error::Config`] when:
    /// - `name` is empty after trimming, or
    /// - `schema` is not a JSON object at the top level.
    pub fn new(name: impl Into<String>, schema: Value) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(Error::config("JsonSchemaSpec: name must be non-empty"));
        }
        if !schema.is_object() {
            return Err(Error::config(
                "JsonSchemaSpec: schema must be a JSON object at the top level",
            ));
        }
        Ok(Self { name, schema })
    }
}

/// How the codec should ask the model to honour the schema.
///
/// Industry consensus (LangChain 1.0 `ProviderStrategy`/
/// `ToolStrategy`, pydantic-ai 1.90 `NativeOutput`/`ToolOutput`/
/// `PromptedOutput`/`TextOutput`, BAML SAP, Vercel AI SDK 5
/// `generateObject`, Instructor's mode flag) converges on three
/// dispatch shapes plus an automatic picker:
///
/// - `Native` — vendor-native structured output channel (OpenAI
///   `text.format = json_schema`, Gemini `responseJsonSchema`,
///   Anthropic `output_config.format = json_schema`). Strictest;
///   the vendor itself rejects malformed responses.
/// - `Tool` — single forced tool call whose input schema is the
///   target schema. Mature on every vendor (the tool-call
///   surface predates native structured output by a year+);
///   slightly less efficient because the model emits a tool_use
///   block instead of plain assistant text.
/// - `Prompted` — schema injected into the system prompt; the
///   reply is parsed best-effort. Last resort for vendors with
///   neither native nor tool support, and for "I want a typed
///   answer but the model is non-reasoning" flows. Deferred to
///   1.1 — `complete_typed` rejects this strategy at runtime
///   today.
/// - `Auto` — codec picks per-vendor at codec-construction time
///   (NOT per request — per-request resolution would let the same
///   logical request resolve differently across replays, breaking
///   the SessionGraph event log's deterministic-replay guarantee
///   per ADR-0001 §"events SSoT"). The picked strategy is what
///   `Codec::auto_output_strategy(model)` returns.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OutputStrategy {
    /// Codec picks per-vendor at codec-construction time. Default.
    #[default]
    Auto,
    /// Vendor-native structured output channel.
    Native,
    /// Forced single tool call carrying the target schema.
    Tool,
    /// Schema injected into the system prompt (1.1 — currently
    /// rejected at encode time).
    Prompted,
}

/// Structured-output directive attached to a [`ModelRequest`](crate::ir::ModelRequest).
///
/// `strict` requests the vendor's strict-mode interpretation when
/// available (OpenAI). Codecs that cannot enforce strict mode
/// natively emit a `LossyEncode` warning.
///
/// `strategy` selects the dispatch shape (vendor-native channel /
/// forced tool call / prompted). `Auto` (the default) lets each
/// codec pick its preferred shape per `auto_output_strategy(model)`
/// at codec-construction time — see [`OutputStrategy`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Schema the response must conform to.
    pub json_schema: JsonSchemaSpec,
    /// Request strict-mode validation. Defaults to `true` —
    /// callers explicitly opt out with `false` when they want
    /// best-effort schema adherence (some Anthropic shim flows).
    #[serde(default = "ResponseFormat::default_strict")]
    pub strict: bool,
    /// Dispatch shape — vendor-native, forced-tool, or prompted.
    /// Defaults to [`OutputStrategy::Auto`] which lets the codec
    /// pick per-vendor at construction time.
    #[serde(default)]
    pub strategy: OutputStrategy,
}

impl ResponseFormat {
    /// Build a strict response format from the supplied schema.
    /// `strategy` defaults to `Auto`; chain
    /// [`Self::with_strategy`] to override.
    pub fn strict(schema: JsonSchemaSpec) -> Self {
        Self {
            json_schema: schema,
            strict: true,
            strategy: OutputStrategy::Auto,
        }
    }

    /// Build a best-effort response format (no strict-mode
    /// validation requested). `strategy` defaults to `Auto`.
    pub fn best_effort(schema: JsonSchemaSpec) -> Self {
        Self {
            json_schema: schema,
            strict: false,
            strategy: OutputStrategy::Auto,
        }
    }

    /// Override the dispatch [`OutputStrategy`]. `Auto` means
    /// "codec picks at construction time"; explicit `Native` /
    /// `Tool` / `Prompted` overrides per-codec defaulting.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: OutputStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Validate the schema against the strict-mode constraints
    /// shared across `OpenAI` (Chat + Responses) and Anthropic
    /// native structured outputs (ADR-0031). Returns the offending
    /// field path on failure so codecs can attach an actionable
    /// `LossyEncode` warning.
    ///
    /// Constraints checked:
    /// - every object schema declares `additionalProperties: false`
    /// - every object schema's `required` list contains *every*
    ///   property defined in `properties` (`OpenAI` strict-mode
    ///   requirement)
    ///
    /// The check is a no-op when `self.strict == false`.
    pub fn strict_preflight(&self) -> std::result::Result<(), StrictSchemaError> {
        if !self.strict {
            return Ok(());
        }
        check_strict(&self.json_schema.schema, "$")
    }

    const fn default_strict() -> bool {
        true
    }
}

/// Reason a strict-mode `JsonSchemaSpec` did not meet the
/// vendor-shared constraints checked by
/// [`ResponseFormat::strict_preflight`].
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum StrictSchemaError {
    /// An object schema is missing `additionalProperties: false`,
    /// or carries a non-`false` value.
    #[error("strict-mode schema requires `additionalProperties: false` at {path}")]
    AdditionalPropertiesNotFalse {
        /// Dotted path into the schema (`$.properties.user`).
        path: String,
    },
    /// An object schema's `required` array does not include every
    /// property defined under `properties` — `OpenAI` strict mode
    /// rejects partial-required object schemas.
    #[error("strict-mode schema at {path} declares properties not in `required`: {}", .missing.join(", "))]
    RequiredMissingProperties {
        /// Dotted path into the schema.
        path: String,
        /// Properties declared but not required.
        missing: Vec<String>,
    },
}

fn check_strict(schema: &Value, path: &str) -> std::result::Result<(), StrictSchemaError> {
    // Only object schemas carry the constraint. Other shapes
    // (string, number, array) pass through unchecked.
    let Some(obj) = schema.as_object() else {
        return Ok(());
    };
    let kind = obj.get("type").and_then(Value::as_str);

    if kind == Some("object") {
        match obj.get("additionalProperties") {
            Some(Value::Bool(false)) => {}
            _ => {
                return Err(StrictSchemaError::AdditionalPropertiesNotFalse {
                    path: path.to_owned(),
                });
            }
        }
        if let Some(Value::Object(properties)) = obj.get("properties") {
            let required: std::collections::BTreeSet<&str> = obj
                .get("required")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let missing: Vec<String> = properties
                .keys()
                .filter(|k| !required.contains(k.as_str()))
                .cloned()
                .collect();
            if !missing.is_empty() {
                return Err(StrictSchemaError::RequiredMissingProperties {
                    path: path.to_owned(),
                    missing,
                });
            }
            // Recurse into each property schema.
            for (name, sub) in properties {
                check_strict(sub, &format!("{path}.properties.{name}"))?;
            }
        }
    } else if kind == Some("array")
        && let Some(items) = obj.get("items")
    {
        check_strict(items, &format!("{path}.items"))?;
    }
    // Recurse into composition keywords (anyOf / allOf / oneOf).
    for keyword in ["anyOf", "allOf", "oneOf"] {
        if let Some(Value::Array(arr)) = obj.get(keyword) {
            for (i, sub) in arr.iter().enumerate() {
                check_strict(sub, &format!("{path}.{keyword}[{i}]"))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn new_rejects_empty_name() {
        let err = JsonSchemaSpec::new("", json!({"type": "object"})).unwrap_err();
        assert!(format!("{err}").contains("name must be non-empty"));
    }

    #[test]
    fn new_rejects_whitespace_only_name() {
        let err = JsonSchemaSpec::new("   ", json!({"type": "object"})).unwrap_err();
        assert!(format!("{err}").contains("name must be non-empty"));
    }

    #[test]
    fn new_rejects_non_object_schema() {
        let err = JsonSchemaSpec::new("user", json!("not an object")).unwrap_err();
        assert!(format!("{err}").contains("must be a JSON object"));
        let err2 = JsonSchemaSpec::new("user", json!([1, 2, 3])).unwrap_err();
        assert!(format!("{err2}").contains("must be a JSON object"));
    }

    #[test]
    fn new_accepts_valid_object_schema() {
        let spec = JsonSchemaSpec::new(
            "user",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }),
        )
        .unwrap();
        assert_eq!(spec.name, "user");
        assert!(spec.schema.is_object());
    }

    #[test]
    fn strict_constructor_sets_strict_flag() {
        let spec = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
        let format = ResponseFormat::strict(spec);
        assert!(format.strict);
    }

    #[test]
    fn best_effort_constructor_clears_strict_flag() {
        let spec = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
        let format = ResponseFormat::best_effort(spec);
        assert!(!format.strict);
    }

    #[test]
    fn round_trips_via_serde() {
        let spec = JsonSchemaSpec::new("user", json!({"type": "object"})).unwrap();
        let format = ResponseFormat::strict(spec);
        let json = serde_json::to_string(&format).unwrap();
        let back: ResponseFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(format, back);
    }
}
