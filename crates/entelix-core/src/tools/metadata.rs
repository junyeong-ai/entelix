//! [`ToolMetadata`] — single source of truth for everything a tool
//! advertises to the runtime, the model, and observability.
//!
//! `Tool` impls hold one of these as a field and return it from
//! `Tool::metadata`. The struct is `#[non_exhaustive]` so future
//! additions (effect taxonomy, retry knobs, scheduling hints) extend
//! without touching call sites — operators always construct via
//! [`ToolMetadata::function`] and the `with_*` chain.

use std::time::Duration;

use serde_json::Value;

use crate::tools::effect::{RetryHint, ToolEffect};

/// Declarative description of a tool.
///
/// Every field is plain-data; constructed once (typically in the
/// tool's own `new()`) and returned by reference from
/// `Tool::metadata`. The runtime treats this as authoritative —
/// codecs render it into the on-the-wire `ToolSpec`, OTel layers
/// stamp `gen_ai.tool.*` attributes from it, `Approver` defaults
/// route off `effect`, and retry middleware honours `retry_hint`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToolMetadata {
    /// Stable identifier the model uses to call this tool. Must be
    /// unique within a `ToolRegistry`. Conventionally `snake_case`.
    pub name: String,
    /// Human-readable description shown to the model. Used to help
    /// the model decide when to call this tool — write it like a
    /// function docstring.
    pub description: String,
    /// JSON Schema for the `input` payload that `Tool::execute`
    /// accepts. Codecs translate this into the vendor's tool schema
    /// format.
    pub input_schema: Value,
    /// Optional JSON Schema describing the *output* shape. Vendors
    /// that support strict tool-output schemas (`OpenAI`'s
    /// `strict: true`, Anthropic's response format hints) read
    /// this. `None` = untyped JSON.
    pub output_schema: Option<Value>,
    /// Optional version string. Surfaces in OTel
    /// (`gen_ai.tool.version`) and in audit events so operators can
    /// distinguish between tool revisions when behaviour changes.
    pub version: Option<String>,
    /// Side-effect classification. Drives default `Approver`
    /// behaviour (Destructive → require approval) and is rendered
    /// to the LLM so the model can reason about safety on its own.
    pub effect: ToolEffect,
    /// `true` when calling the tool repeatedly with the same input
    /// produces the same effect (no incremental change). Retry
    /// middleware uses this as the cheap binary version of
    /// `retry_hint.is_some()`.
    pub idempotent: bool,
    /// Per-tool retry policy hint. `None` (the default) means the
    /// tool is *not* retried by middleware.
    pub retry_hint: Option<RetryHint>,
    /// Best-guess execution time for dashboards / scheduling. Used
    /// only as a hint — the runtime never enforces it as a deadline
    /// (use `ExecutionContext::deadline` for that).
    pub typical_duration: Option<Duration>,
}

impl ToolMetadata {
    /// Construct a function-tool descriptor with conservative
    /// defaults (`effect = ReadOnly`, no retry, no version).
    /// Customise via the `with_*` chain.
    #[must_use]
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema: None,
            version: None,
            effect: ToolEffect::default(),
            idempotent: false,
            retry_hint: None,
            typical_duration: None,
        }
    }

    /// Attach an output schema.
    #[must_use]
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Attach a version string.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Override the side-effect classification.
    #[must_use]
    pub const fn with_effect(mut self, effect: ToolEffect) -> Self {
        self.effect = effect;
        self
    }

    /// Mark the tool idempotent — repeat calls with the same input
    /// produce the same effect.
    #[must_use]
    pub const fn with_idempotent(mut self, idempotent: bool) -> Self {
        self.idempotent = idempotent;
        self
    }

    /// Attach a retry hint. Implies `idempotent = true` because a
    /// non-idempotent tool that opts into retries is a bug.
    #[must_use]
    pub const fn with_retry_hint(mut self, hint: RetryHint) -> Self {
        self.retry_hint = Some(hint);
        self.idempotent = true;
        self
    }

    /// Attach a typical-duration hint.
    #[must_use]
    pub const fn with_typical_duration(mut self, duration: Duration) -> Self {
        self.typical_duration = Some(duration);
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn function_defaults_are_conservative() {
        let m = ToolMetadata::function("echo", "echoes input", json!({"type": "object"}));
        assert_eq!(m.name, "echo");
        assert_eq!(m.description, "echoes input");
        assert!(m.output_schema.is_none());
        assert!(m.version.is_none());
        assert_eq!(m.effect, ToolEffect::ReadOnly);
        assert!(!m.idempotent);
        assert!(m.retry_hint.is_none());
        assert!(m.typical_duration.is_none());
    }

    #[test]
    fn with_retry_hint_implies_idempotent() {
        let m = ToolMetadata::function("get", "fetches", json!({}))
            .with_retry_hint(RetryHint::idempotent_transport());
        assert!(m.idempotent);
        assert!(m.retry_hint.is_some());
    }

    #[test]
    fn builder_chain_is_const_friendly() {
        let m = ToolMetadata::function("delete", "deletes a row", json!({}))
            .with_effect(ToolEffect::Destructive)
            .with_version("1.2.0")
            .with_typical_duration(Duration::from_millis(50));
        assert_eq!(m.effect, ToolEffect::Destructive);
        assert_eq!(m.version.as_deref(), Some("1.2.0"));
        assert_eq!(m.typical_duration, Some(Duration::from_millis(50)));
    }
}
