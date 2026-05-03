//! LLM-facing channel — type-level separation of operator-facing
//! diagnostics from the value the model actually sees (invariant #16).
//!
//! Two surfaces, both narrowly defined:
//!
//! - [`LlmFacingError`] — `render_for_llm()` returns a short,
//!   actionable string the model can act on. Vendor status codes,
//!   type-system identifiers, and source-error chains are
//!   deliberately omitted; they would burn the model's attention
//!   budget while supplying no information the model can use. The
//!   full error continues to flow through `Display` / `Error::source`
//!   to the operator channels (OTel, event sinks, log sinks).
//! - [`LlmFacingSchema`] — `strip(&Value) -> Value` reduces a JSON
//!   Schema to the keys vendor APIs actually consume (`type`,
//!   `properties`, `required`, `items`, `enum`, `description`,
//!   bounds…). Schemars-generated knobs (`$schema`, `title`,
//!   `$defs`, `$ref`, format specifiers like `int64`) ride out.
//!   Saves 30–120 tokens per tool per request × every turn.
//!
//! ## Why a separate trait rather than a method on `Error`
//!
//! The split lets non-`Error` types (custom tool error wrappers, MCP
//! server errors lifted into IR) opt into the same contract without
//! coupling to `entelix_core::Error`. Default impls on `Error` and
//! `String`/`&str` cover the common cases; bespoke implementors
//! override.
//!
//! ## Enforcement
//!
//! `tests/llm_context_economy.rs` (entelix-core) regression-checks
//! that built-in tool outputs and tool-spec schemas never leak the
//! forbidden patterns. CI rejects new sites silently re-introducing
//! operator-channel content into the model's view.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::error::Error;

/// Render a value (typically an error) into the short, actionable
/// string the model is allowed to see. Implementors keep prose
/// brief, omit operator-only context (status codes, type-system
/// identifiers, source chains), and never echo input payloads —
/// those are prompt-injection vectors.
pub trait LlmFacingError {
    /// The model-facing rendering. Must not include vendor status
    /// codes, `provider returned …` framing, source chains,
    /// RFC3339 timestamps, or internal type names — operator
    /// channels carry those.
    fn render_for_llm(&self) -> String;
}

impl LlmFacingError for Error {
    /// Short, model-actionable rendering. Mapping:
    ///
    /// - `InvalidRequest(msg)` → `"invalid input: {msg}"` — the
    ///   message is already caller-supplied and free of vendor
    ///   identifiers.
    /// - `Provider { .. }` → `"upstream model error"` — vendor
    ///   status is operator-only.
    /// - `Auth(_)` → `"authentication failed"` — never echo the
    ///   underlying provider's auth diagnostic.
    /// - `Config(_)` → `"tool misconfigured"` — operator must fix.
    /// - `Cancelled` → `"cancelled"`.
    /// - `DeadlineExceeded` → `"timed out"`.
    /// - `Interrupted { .. }` → `"awaiting human review"`.
    /// - `Serde(_)` → `"output could not be serialised"` — the
    ///   inner serde error names internal types.
    fn render_for_llm(&self) -> String {
        match self {
            Self::InvalidRequest(msg) => format!("invalid input: {msg}"),
            Self::Provider { .. } => "upstream model error".to_owned(),
            Self::Auth(_) => "authentication failed".to_owned(),
            Self::Config(_) => "tool misconfigured".to_owned(),
            Self::Cancelled => "cancelled".to_owned(),
            Self::DeadlineExceeded => "timed out".to_owned(),
            Self::Interrupted { .. } => "awaiting human review".to_owned(),
            Self::Serde(_) => "output could not be serialised".to_owned(),
        }
    }
}

/// JSON-Schema sanitiser — strips schemars / draft-meta keys that
/// vendor APIs ignore but that still cost tokens to ship.
pub struct LlmFacingSchema;

/// JSON-Schema key classification — drives the schema-aware walk.
///
/// Different keys hold different *kinds* of value: some carry literal
/// data (`type: "string"`, `description: "..."`), some carry a single
/// nested schema (`items`, `additionalProperties` when an object),
/// some carry an array of schemas (`anyOf`, `oneOf`, `allOf`), some
/// carry a `map<user-name, schema>` (`properties`), and some carry
/// user data that must not be schema-walked (`enum`, `default`,
/// `const`, `required`). The classifier picks the right walk for
/// each key so user-named properties survive the strip and user
/// values are not accidentally pruned to empty objects.
enum AllowedKey {
    /// Literal value — `type`, `description`, bounds, `format`, …
    /// Cloned through (with the `format` noise filter applied).
    Literal,
    /// Single nested schema — `items` (single-schema form),
    /// `additionalProperties` (when an object), `not`.
    Schema,
    /// Array of nested schemas — `anyOf`, `oneOf`, `allOf`.
    /// `items` (array form) also flows through here at runtime.
    SchemaArray,
    /// Map of user-named entries to schemas — `properties`. Keys
    /// are preserved verbatim; values are schema-walked.
    SchemaMap,
    /// User data — `enum`, `default`, `const`, `required`. Cloned
    /// verbatim; never schema-walked.
    UserData,
}

fn classify(key: &str) -> Option<AllowedKey> {
    Some(match key {
        "type" | "description" | "minimum" | "maximum" | "exclusiveMinimum"
        | "exclusiveMaximum" | "minLength" | "maxLength" | "minItems" | "maxItems"
        | "uniqueItems" | "minProperties" | "maxProperties" | "pattern" | "format" => {
            AllowedKey::Literal
        }
        "items" | "additionalProperties" | "not" => AllowedKey::Schema,
        "anyOf" | "oneOf" | "allOf" => AllowedKey::SchemaArray,
        "properties" => AllowedKey::SchemaMap,
        "enum" | "default" | "const" | "required" => AllowedKey::UserData,
        _ => return None,
    })
}

/// `format` values that read as noise to the vendor — the
/// JSON-Schema-encoded width hint is already implied by
/// `type: "integer"`/`"number"` and the model gains nothing from
/// seeing it. Removing them shrinks the wire without losing meaning.
const NOISY_FORMATS: &[&str] = &[
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float", "double",
];

impl LlmFacingSchema {
    /// Walk `schema` and return a copy containing only
    /// vendor-relevant keys. The walk inlines `$ref`/`$defs`
    /// indirection so the resulting schema is self-contained — no
    /// dangling references, no draft-meta envelope.
    #[must_use]
    pub fn strip(schema: &Value) -> Value {
        let defs = collect_defs(schema);
        strip_schema(schema, &defs)
    }
}

fn collect_defs(schema: &Value) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    if let Some(obj) = schema.as_object() {
        // Merge `$defs` (2020-12) and the legacy `definitions` key.
        for key in ["$defs", "definitions"] {
            if let Some(Value::Object(defs)) = obj.get(key) {
                for (name, body) in defs {
                    out.insert(name.clone(), body.clone());
                }
            }
        }
    }
    out
}

/// Strip one schema node. Resolves `$ref` indirection up front, then
/// dispatches each surviving key according to its [`AllowedKey`]
/// classification.
fn strip_schema(node: &Value, defs: &BTreeMap<String, Value>) -> Value {
    let Some(obj) = node.as_object() else {
        // Not an object (likely a boolean schema like
        // `additionalProperties: false` or an `items: true` shorthand)
        // — clone through unchanged.
        return node.clone();
    };

    // `$ref` short-circuits — replace the whole node with the
    // stripped definition body. Eliminates `$defs` indirection.
    if let Some(Value::String(reference)) = obj.get("$ref")
        && let Some(name) = reference
            .strip_prefix("#/$defs/")
            .or_else(|| reference.strip_prefix("#/definitions/"))
        && let Some(target) = defs.get(name)
    {
        return strip_schema(target, defs);
    }

    let mut out = Map::new();
    for (key, value) in obj {
        let Some(kind) = classify(key) else {
            continue;
        };
        match kind {
            AllowedKey::Literal => {
                if key == "format"
                    && let Some(format) = value.as_str()
                    && NOISY_FORMATS.contains(&format)
                {
                    continue;
                }
                out.insert(key.clone(), value.clone());
            }
            AllowedKey::Schema => {
                // `items` may be a single schema or an array of
                // schemas (tuple-style validation); `additionalProperties`
                // may be a boolean. Dispatch per shape.
                let stripped = match value {
                    Value::Array(arr) => {
                        Value::Array(arr.iter().map(|v| strip_schema(v, defs)).collect())
                    }
                    other => strip_schema(other, defs),
                };
                out.insert(key.clone(), stripped);
            }
            AllowedKey::SchemaArray => {
                if let Value::Array(arr) = value {
                    let stripped: Vec<Value> = arr.iter().map(|v| strip_schema(v, defs)).collect();
                    out.insert(key.clone(), Value::Array(stripped));
                } else {
                    // Malformed — keep the original; the vendor will
                    // reject it with a clearer error than we can
                    // synthesize here.
                    out.insert(key.clone(), value.clone());
                }
            }
            AllowedKey::SchemaMap => {
                // User-named keys → preserve verbatim, values → walk.
                if let Value::Object(map) = value {
                    let stripped: Map<String, Value> = map
                        .iter()
                        .map(|(k, v)| (k.clone(), strip_schema(v, defs)))
                        .collect();
                    out.insert(key.clone(), Value::Object(stripped));
                } else {
                    out.insert(key.clone(), value.clone());
                }
            }
            AllowedKey::UserData => {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    Value::Object(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_for_llm_omits_provider_status() {
        let err = Error::provider_http(503, "vendor down".to_owned());
        let rendered = err.render_for_llm();
        assert!(!rendered.contains("503"), "{rendered}");
        assert!(!rendered.contains("vendor down"), "{rendered}");
        assert!(!rendered.contains("provider returned"), "{rendered}");
    }

    #[test]
    fn render_for_llm_invalid_request_carries_caller_message() {
        let err = Error::invalid_request("missing 'task' field");
        assert_eq!(err.render_for_llm(), "invalid input: missing 'task' field");
    }

    #[test]
    fn strip_removes_schema_envelope() {
        let raw = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "DoubleInput",
            "type": "object",
            "properties": {"n": {"type": "integer", "format": "int64"}},
            "required": ["n"]
        });
        let stripped = LlmFacingSchema::strip(&raw);
        assert!(stripped.get("$schema").is_none());
        assert!(stripped.get("title").is_none());
        assert_eq!(stripped["type"], "object");
        assert_eq!(stripped["properties"]["n"]["type"], "integer");
        // int64 is the noisy width hint — dropped.
        assert!(stripped["properties"]["n"].get("format").is_none());
        assert_eq!(stripped["required"], json!(["n"]));
    }

    #[test]
    fn strip_inlines_refs_and_drops_defs_envelope() {
        let raw = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Outer",
            "type": "object",
            "properties": {"inner": {"$ref": "#/$defs/Inner"}},
            "$defs": {
                "Inner": {
                    "title": "Inner",
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"]
                }
            }
        });
        let stripped = LlmFacingSchema::strip(&raw);
        assert!(stripped.get("$defs").is_none());
        let inner = &stripped["properties"]["inner"];
        // $ref resolved → inlined object, title gone.
        assert_eq!(inner["type"], "object");
        assert_eq!(inner["properties"]["x"]["type"], "string");
        assert!(inner.get("title").is_none());
    }

    #[test]
    fn strip_keeps_meaningful_format_specifiers() {
        // `date-time`, `email`, `uri` are real vendor-honored
        // formats — the noise list only targets width hints.
        let raw = json!({
            "type": "string",
            "format": "date-time"
        });
        let stripped = LlmFacingSchema::strip(&raw);
        assert_eq!(stripped["format"], "date-time");
    }
}
