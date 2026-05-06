//! `SchemaTool` — typed-I/O ergonomics on top of [`entelix_core::Tool`].
//!
//! The base [`Tool`] trait takes `serde_json::Value` for both input
//! and output so the dispatcher / registry / metadata machinery can
//! be type-erased. Tool authors usually want the opposite: a typed
//! `Input` they can pattern-match and a typed `Output` whose shape
//! they fix at compile time. `SchemaTool` is the typed sibling — the
//! adapter layer ([`SchemaToolAdapter`]) bridges back to the
//! erased trait so the registry stays untouched.
//!
//! ## What you write
//!
//! ```no_run
//! use entelix_core::AgentContext;
//! use entelix_core::error::Result;
//! use entelix_tools::{SchemaTool, SchemaToolExt};
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Deserialize, JsonSchema)]
//! pub struct DoubleInput {
//!     pub n: i64,
//! }
//!
//! #[derive(Debug, Serialize, JsonSchema)]
//! pub struct DoubleOutput {
//!     pub doubled: i64,
//! }
//!
//! #[derive(Debug)]
//! pub struct DoubleTool;
//!
//! #[async_trait::async_trait]
//! impl SchemaTool for DoubleTool {
//!     type Input = DoubleInput;
//!     type Output = DoubleOutput;
//!     const NAME: &'static str = "double";
//!     fn description(&self) -> &str {
//!         "Doubles an integer."
//!     }
//!     async fn execute(
//!         &self,
//!         input: Self::Input,
//!         _ctx: &AgentContext<()>,
//!     ) -> Result<Self::Output> {
//!         Ok(DoubleOutput { doubled: input.n * 2 })
//!     }
//! }
//!
//! // Plug into any API that takes a `Tool`:
//! let _tool = DoubleTool.into_adapter();
//! ```
//!
//! ## What you get
//!
//! - **Compile-time I/O guarantees** — wrong-shape calls from inside
//!   the agent runtime fail to type-check; deserialization failures
//!   from the model surface as `Error::InvalidRequest`.
//! - **Auto-generated input schema** — `schemars` walks `Input` and
//!   produces the JSON Schema the codec advertises to the model. No
//!   hand-rolled schema literals to drift out of sync with the
//!   `Deserialize` impl.
//! - **Effect / version metadata** — same `ToolEffect`, `RetryHint`,
//!   and `version` knobs the erased `Tool` trait already exposes,
//!   surfaced through provided trait methods so the typed author
//!   never has to know about [`entelix_core::tools::ToolMetadata`].
//!
//! ## Invariant alignment
//!
//! - Invariant 4 (`Tool` is a leaf with one `execute` method):
//!   `SchemaTool` is *not* `Tool`. The adapter `SchemaToolAdapter<T>`
//!   is what implements `Tool`. Nothing on the dispatcher side needs
//!   to know `SchemaTool` exists.
//! - Invariant 10 (no tokens in tools): `SchemaTool::execute` takes
//!   `&AgentContext`. Credentials still live in transports.
//! - Cancellation (CLAUDE.md §"Cancellation"): typed authors check
//!   `ctx.is_cancelled()` for long loops just like the erased path.

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::AgentContext;
use entelix_core::LlmFacingSchema;
use entelix_core::error::{Error, Result};
use entelix_core::tools::{RetryHint, Tool, ToolEffect, ToolMetadata};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Typed-I/O sibling of [`Tool`]. Implementors get
/// `Input`/`Output` typed against the model's tool dispatch
/// without giving up the erased trait the rest of the SDK speaks.
///
/// Wrap with [`SchemaToolExt::into_adapter`] to expose the typed
/// tool to a `ToolRegistry`.
#[async_trait]
pub trait SchemaTool: Send + Sync + 'static {
    /// Typed input the model's tool call decodes into. `JsonSchema`
    /// drives auto-schema generation; `DeserializeOwned + Send`
    /// lets the adapter parse the model's `Value` payload without
    /// borrowing.
    type Input: DeserializeOwned + JsonSchema + Send + 'static;

    /// Typed output the tool returns. The adapter re-serializes it
    /// to `Value` for the codec; `Send` keeps the
    /// adapter `async fn` `Send`.
    type Output: Serialize + Send + 'static;

    /// Stable tool name surfaced to the model. Must be unique
    /// within a `ToolRegistry`. Conventionally `snake_case`.
    const NAME: &'static str;

    /// Tool description shown to the model. Implemented as a
    /// method so authors can emit dynamic strings (e.g. include
    /// the configured backend's name) — not a `const` because
    /// `&'static str` would force every author into static storage.
    fn description(&self) -> &str;

    /// Side-effect classification. Defaults to
    /// [`ToolEffect::ReadOnly`] — most tools don't mutate state.
    /// Override when the tool writes / deletes / dispatches.
    fn effect(&self) -> ToolEffect {
        ToolEffect::default()
    }

    /// Optional retry hint surfaced through OTel. Defaults to
    /// `None`; idempotent transports override.
    fn retry_hint(&self) -> Option<RetryHint> {
        None
    }

    /// Optional version label surfaced through OTel and audit
    /// events. Useful when the same `NAME` ships behavioural
    /// revisions.
    fn version(&self) -> Option<&str> {
        None
    }

    /// Optional output schema. Implementors override to enforce a
    /// vendor strict-output contract — invoke
    /// `schemars::schema_for!(Self::Output).to_value()` to mirror
    /// the auto-generation the input side gets for free. The default
    /// returns `None`.
    fn output_schema(&self) -> Option<serde_json::Value> {
        None
    }

    /// Whether the tool is idempotent — repeat calls with the same
    /// input produce the same effect. Defaults to `false`; pure
    /// computational tools (`ReadOnly` effect) and idempotent
    /// transports override to `true` so the runtime can dedupe
    /// retries server-side.
    fn idempotent(&self) -> bool {
        false
    }

    /// Run the tool against a typed input. The adapter handles
    /// JSON deserialisation upstream — implementors only see fully
    /// validated `Self::Input` and return a typed `Self::Output`.
    /// Long loops should periodically check `ctx.is_cancelled()`.
    async fn execute(&self, input: Self::Input, ctx: &AgentContext<()>) -> Result<Self::Output>;
}

/// Provided extension methods on every [`SchemaTool`]. Lives in a
/// separate trait so blanket-impls (e.g. `Box<dyn SchemaTool>`)
/// don't fight with the user-implemented `SchemaTool` trait
/// associated types.
pub trait SchemaToolExt: SchemaTool + Sized {
    /// Wrap `self` in a [`SchemaToolAdapter`] so it can be
    /// registered through any API that takes a `Tool`. The
    /// adapter generates `Input`'s JSON schema once at
    /// construction and caches it inside the metadata `Arc`.
    fn into_adapter(self) -> SchemaToolAdapter<Self> {
        SchemaToolAdapter::new(self)
    }
}

impl<T: SchemaTool> SchemaToolExt for T {}

/// Adapter that exposes any [`SchemaTool`] through the erased
/// [`Tool`] trait.
///
/// The adapter owns the inner typed tool plus a pre-built
/// [`ToolMetadata`] (input schema generated from the `Input` type,
/// effect / version / retry hint pulled from the `SchemaTool`
/// overrides) so the runtime hot path is a single pointer
/// dereference.
pub struct SchemaToolAdapter<T: SchemaTool> {
    inner: T,
    metadata: Arc<ToolMetadata>,
}

impl<T: SchemaTool> SchemaToolAdapter<T> {
    /// Build the adapter, generating `T::Input`'s JSON schema once.
    /// The schema is reduced through [`LlmFacingSchema::strip`] before
    /// landing in [`ToolMetadata`] — the model never sees schemars
    /// envelope keys (`$schema`, `title`, `$defs`, `$ref`, integer
    /// width hints), saving 30–120 tokens per tool per turn
    /// (invariant #16).
    fn new(inner: T) -> Self {
        let raw_schema: serde_json::Value = schemars::schema_for!(T::Input).to_value();
        let input_schema = LlmFacingSchema::strip(&raw_schema);
        let mut metadata = ToolMetadata::function(T::NAME, inner.description(), input_schema)
            .with_effect(inner.effect())
            .with_idempotent(inner.idempotent());
        if let Some(version) = inner.version() {
            metadata = metadata.with_version(version);
        }
        if let Some(hint) = inner.retry_hint() {
            metadata = metadata.with_retry_hint(hint);
        }
        if let Some(output_schema) = inner.output_schema() {
            metadata = metadata.with_output_schema(LlmFacingSchema::strip(&output_schema));
        }
        Self {
            inner,
            metadata: Arc::new(metadata),
        }
    }

    /// Borrow the wrapped typed tool. Useful when registry-side
    /// code wants to recover the typed handle for direct dispatch
    /// (tests, alternative invokers).
    pub const fn inner(&self) -> &T {
        &self.inner
    }
}

impl<T: SchemaTool> std::fmt::Debug for SchemaToolAdapter<T> {
    /// Surfaces the metadata identity without forcing `T: Debug` —
    /// the wrapped tool's internals are opaque to the adapter, so
    /// transitively requiring a Debug bound on every typed tool is
    /// noise. The metadata `name` is the operationally meaningful
    /// identifier in logs / crash dumps.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaToolAdapter")
            .field("name", &self.metadata.name)
            .field("inner", &std::any::type_name::<T>())
            .finish()
    }
}

#[async_trait]
impl<T: SchemaTool> Tool for SchemaToolAdapter<T> {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &AgentContext<()>,
    ) -> Result<serde_json::Value> {
        // Diagnostics surface only the tool name and the serde
        // message — internal Rust type identifiers
        // (`std::any::type_name`) are operator-only and would burn
        // model attention without informing recovery (invariant #16).
        let typed: T::Input = serde_json::from_value(input).map_err(|e| {
            Error::invalid_request(format!(
                "tool '{name}': input did not match schema: {e}",
                name = T::NAME,
            ))
        })?;
        let output = self.inner.execute(typed, ctx).await?;
        serde_json::to_value(output).map_err(Error::from)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct DoubleInput {
        n: i64,
    }

    #[derive(Debug, Serialize, JsonSchema)]
    struct DoubleOutput {
        doubled: i64,
    }

    #[derive(Debug)]
    struct DoubleTool;

    #[async_trait]
    impl SchemaTool for DoubleTool {
        type Input = DoubleInput;
        type Output = DoubleOutput;
        const NAME: &'static str = "double";

        fn description(&self) -> &str {
            "Doubles an integer."
        }

        async fn execute(
            &self,
            input: Self::Input,
            _ctx: &AgentContext<()>,
        ) -> Result<Self::Output> {
            Ok(DoubleOutput {
                doubled: input.n * 2,
            })
        }
    }

    #[derive(Debug)]
    struct VersionedTool;

    #[async_trait]
    impl SchemaTool for VersionedTool {
        type Input = DoubleInput;
        type Output = DoubleOutput;
        const NAME: &'static str = "versioned";

        fn description(&self) -> &str {
            "Versioned tool."
        }

        fn version(&self) -> Option<&str> {
            Some("1.2.3")
        }

        fn effect(&self) -> ToolEffect {
            ToolEffect::Mutating
        }

        async fn execute(
            &self,
            input: Self::Input,
            _ctx: &AgentContext<()>,
        ) -> Result<Self::Output> {
            Ok(DoubleOutput {
                doubled: input.n + 1,
            })
        }
    }

    #[derive(Debug)]
    struct RetryableTool;

    #[async_trait]
    impl SchemaTool for RetryableTool {
        type Input = DoubleInput;
        type Output = DoubleOutput;
        const NAME: &'static str = "retryable";

        fn description(&self) -> &str {
            "Retryable tool."
        }

        fn retry_hint(&self) -> Option<RetryHint> {
            Some(RetryHint::idempotent_transport())
        }

        fn output_schema(&self) -> Option<serde_json::Value> {
            Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "doubled": { "type": "integer" }
                },
                "required": ["doubled"]
            }))
        }

        async fn execute(
            &self,
            input: Self::Input,
            _ctx: &AgentContext<()>,
        ) -> Result<Self::Output> {
            Ok(DoubleOutput { doubled: input.n })
        }
    }

    #[tokio::test]
    async fn typed_round_trip_through_adapter() {
        let adapter = DoubleTool.into_adapter();
        let ctx = AgentContext::default();
        let out = adapter.execute(json!({"n": 21}), &ctx).await.unwrap();
        assert_eq!(out, json!({"doubled": 42}));
    }

    #[tokio::test]
    async fn malformed_input_surfaces_invalid_request() {
        let adapter = DoubleTool.into_adapter();
        let ctx = AgentContext::default();
        let err = adapter
            .execute(json!({"wrong_field": 21}), &ctx)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("double"), "{msg}");
        assert!(msg.contains("input did not match schema"), "{msg}");
        // Invariant #16 — diagnostic must NOT leak internal Rust
        // type identifiers (`DoubleInput`, module paths, …). The
        // model gains nothing from `entelix_tools::schema_tool::tests::DoubleInput`.
        assert!(
            !msg.contains("DoubleInput"),
            "internal type name must not surface to the model: {msg}"
        );
    }

    #[test]
    fn metadata_carries_autogenerated_input_schema() {
        let adapter = DoubleTool.into_adapter();
        let meta = adapter.metadata();
        assert_eq!(meta.name, "double");
        assert_eq!(meta.description, "Doubles an integer.");
        // schemars emits a JSON Schema that mentions `n` as the
        // expected property — exact shape varies with schemars
        // versions, so we only assert it surfaced the field name.
        let schema_str = meta.input_schema.to_string();
        assert!(schema_str.contains("\"n\""), "{schema_str}");
    }

    #[test]
    fn metadata_propagates_effect_and_version() {
        let adapter = VersionedTool.into_adapter();
        let meta = adapter.metadata();
        assert_eq!(meta.effect, ToolEffect::Mutating);
        assert_eq!(meta.version.as_deref(), Some("1.2.3"));
    }

    #[test]
    fn defaults_apply_when_overrides_absent() {
        let adapter = DoubleTool.into_adapter();
        let meta = adapter.metadata();
        assert_eq!(meta.effect, ToolEffect::ReadOnly);
        assert!(meta.version.is_none());
        assert!(meta.retry_hint.is_none());
        assert!(meta.output_schema.is_none());
    }

    #[test]
    fn metadata_propagates_retry_hint() {
        let adapter = RetryableTool.into_adapter();
        let meta = adapter.metadata();
        assert!(meta.retry_hint.is_some());
        // `with_retry_hint` flips `idempotent` to true (ToolMetadata
        // contract) — verifying both keeps the wire-through honest.
        assert!(meta.idempotent);
    }

    #[test]
    fn metadata_propagates_output_schema() {
        let adapter = RetryableTool.into_adapter();
        let meta = adapter.metadata();
        let schema = meta
            .output_schema
            .as_ref()
            .expect("output_schema override should land in metadata");
        let schema_str = schema.to_string();
        assert!(schema_str.contains("doubled"), "{schema_str}");
    }

    #[derive(Debug, Default, PartialEq, Eq)]
    struct StatefulTool {
        marker: u32,
    }

    #[async_trait]
    impl SchemaTool for StatefulTool {
        type Input = DoubleInput;
        type Output = DoubleOutput;
        const NAME: &'static str = "stateful";

        fn description(&self) -> &str {
            "Stateful tool."
        }

        async fn execute(
            &self,
            input: Self::Input,
            _ctx: &AgentContext<()>,
        ) -> Result<Self::Output> {
            Ok(DoubleOutput { doubled: input.n })
        }
    }

    #[test]
    fn inner_preserves_wrapped_instance_identity() {
        // Sentinel value the wrapper must round-trip — guards against
        // a regression where `inner()` returned a fresh `T::default()`
        // or a different cell (the type-only check is too weak).
        let adapter = StatefulTool {
            marker: 0xDEAD_BEEF,
        }
        .into_adapter();
        assert_eq!(adapter.inner().marker, 0xDEAD_BEEF);
    }
}
