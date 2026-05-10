//! `Toolset` — reusable init-time bundle of tools.
//!
//! A `Toolset` is a declaration surface, not a dispatch surface. It
//! owns a stable id plus an append-only collection of tools, then
//! installs those tools into a [`ToolRegistry`]. The registry remains
//! the single runtime source of truth for model-facing tool specs,
//! schema validation, typed deps, and `tower::Layer` dispatch.
//!
//! This mirrors the industry-standard "toolset/capability" shape:
//! operators can compose, swap, restrict, and test a bundle of tools
//! before materialising it into an agent's registry. Runtime
//! behaviour still flows through one registry path, so policy,
//! approval, retry, OTel, and typed deps cannot be bypassed.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::identity::validate_config_identifier;
use crate::ir::ToolSpec;
use crate::tools::metadata::ToolMetadata;
use crate::tools::registry::ToolRegistry;
use crate::tools::tool::Tool;

/// Reusable, append-only collection of tools.
///
/// `D` is the same typed-deps parameter used by [`Tool`] and
/// [`ToolRegistry`]. A `Toolset<D>` may only be installed into a
/// `ToolRegistry<D>`, preserving the operator's dependency boundary
/// at compile time.
pub struct Toolset<D = ()>
where
    D: Send + Sync + 'static,
{
    id: String,
    by_name: BTreeMap<String, Arc<dyn Tool<D>>>,
}

impl<D> Clone for Toolset<D>
where
    D: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            by_name: self.by_name.clone(),
        }
    }
}

impl<D> Toolset<D>
where
    D: Send + Sync + 'static,
{
    /// Create an empty toolset with a stable id.
    ///
    /// The id is operator-facing metadata used by capability
    /// manifests, test fixtures, and durable-runtime activity names.
    /// It is not sent to the model and does not alter tool names.
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        validate_identifier("Toolset::new", "id", &id)?;
        Ok(Self {
            id,
            by_name: BTreeMap::new(),
        })
    }

    /// Stable toolset id.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Number of tools in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// True when no tools are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Iterate tool names in stable lexical order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    /// Borrow a tool by exact name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool<D>>> {
        self.by_name.get(name)
    }

    /// Return model-facing tool specs in stable lexical order.
    ///
    /// This is an inspection helper for tests and capability
    /// manifests. Agents should still derive their final advertised
    /// tool catalogue from the installed [`ToolRegistry`].
    #[must_use]
    pub fn tool_specs(&self) -> Arc<[ToolSpec]> {
        self.by_name
            .values()
            .map(|tool| tool.metadata().to_tool_spec())
            .collect()
    }

    /// Append one tool to the set.
    ///
    /// Reach for this when assembling a reusable bundle the operator
    /// installs into multiple `ToolRegistry<D>` instances —
    /// `Toolset::new("support").register(tool_a)?.register(tool_b)?
    /// .install_into(registry)?` is the canonical capability-bundle
    /// path. Duplicate names are rejected: a toolset with ambiguous
    /// names cannot be installed safely because later restriction
    /// and approval policies address tools by exact name.
    pub fn register(mut self, tool: Arc<dyn Tool<D>>) -> Result<Self> {
        validate_metadata("Toolset::register", tool.metadata())?;
        let name = tool.metadata().name.clone();
        if self.by_name.contains_key(&name) {
            return Err(Error::config(format!(
                "Toolset::register: tool '{name}' is already registered in toolset '{}'",
                self.id
            )));
        }
        self.by_name.insert(name, tool);
        Ok(self)
    }

    /// Produce a strict-name restricted view of this set.
    ///
    /// Names absent from the toolset are configuration errors. Empty
    /// names and duplicate names are also rejected, keeping the
    /// declaration surface deterministic and typo-safe.
    pub fn restricted_to(&self, allowed: &[&str]) -> Result<Self> {
        validate_allowed_names("Toolset::restricted_to", allowed)?;
        let missing: Vec<&str> = allowed
            .iter()
            .copied()
            .filter(|name| !self.by_name.contains_key(*name))
            .collect();
        if !missing.is_empty() {
            return Err(Error::config(format!(
                "Toolset::restricted_to: tool name(s) not in toolset '{}': {}",
                self.id,
                missing.join(", ")
            )));
        }

        let allowed: HashSet<&str> = allowed.iter().copied().collect();
        let by_name = self
            .by_name
            .iter()
            .filter(|(name, _)| allowed.contains(name.as_str()))
            .map(|(name, tool)| (name.clone(), Arc::clone(tool)))
            .collect();
        Ok(Self {
            id: self.id.clone(),
            by_name,
        })
    }
}

impl<D> Toolset<D>
where
    D: Clone + Send + Sync + 'static,
{
    /// Install this set into an existing registry.
    ///
    /// The supplied registry's deps and layer stack are preserved.
    /// Registration remains append-only; name collisions with tools
    /// already present in the registry surface as [`Error::Config`]
    /// from [`ToolRegistry::register`].
    pub fn install_into(&self, mut registry: ToolRegistry<D>) -> Result<ToolRegistry<D>> {
        for tool in self.by_name.values() {
            registry = registry.register(Arc::clone(tool))?;
        }
        Ok(registry)
    }
}

impl<D> std::fmt::Debug for Toolset<D>
where
    D: Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Toolset")
            .field("id", &self.id)
            .field("tools", &self.by_name.keys().collect::<Vec<_>>())
            .finish()
    }
}

fn validate_identifier(surface: &str, field: &str, value: &str) -> Result<()> {
    validate_config_identifier(surface, field, value)
}

fn validate_metadata(surface: &str, metadata: &ToolMetadata) -> Result<()> {
    validate_identifier(surface, "tool name", &metadata.name)?;
    if metadata.description.trim().is_empty() {
        return Err(Error::config(format!(
            "{surface}: tool '{}' description must not be empty",
            metadata.name
        )));
    }
    jsonschema::options()
        .build(&metadata.input_schema)
        .map_err(|err| {
            Error::config(format!(
                "{surface}: tool '{}' input schema is invalid: {err}",
                metadata.name
            ))
        })?;
    if let Some(output_schema) = &metadata.output_schema {
        jsonschema::options().build(output_schema).map_err(|err| {
            Error::config(format!(
                "{surface}: tool '{}' output schema is invalid: {err}",
                metadata.name
            ))
        })?;
    }
    Ok(())
}

fn validate_allowed_names(surface: &str, allowed: &[&str]) -> Result<()> {
    for name in allowed {
        validate_identifier(surface, "requested tool name", name)?;
    }
    let mut seen = HashSet::with_capacity(allowed.len());
    let duplicates: Vec<&str> = allowed
        .iter()
        .copied()
        .filter(|name| !seen.insert(*name))
        .collect();
    if !duplicates.is_empty() {
        return Err(Error::config(format!(
            "{surface}: duplicate tool name(s): {}",
            duplicates.join(", ")
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::*;
    use crate::agent_context::AgentContext;

    struct EchoTool {
        metadata: ToolMetadata,
    }

    impl EchoTool {
        fn new(name: &str) -> Self {
            Self {
                metadata: ToolMetadata::function(
                    name,
                    format!("Echo tool {name}"),
                    json!({"type": "object", "properties": {}}),
                ),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }

        async fn execute(&self, input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
            Ok(input)
        }
    }

    #[test]
    fn toolset_rejects_empty_id() {
        let err = Toolset::<()>::new(" ").unwrap_err();
        assert!(format!("{err}").contains("id must not be empty"));
    }

    #[test]
    fn toolset_rejects_ambiguous_ids_and_tool_names() {
        let err = Toolset::<()>::new("core ").unwrap_err();
        assert!(format!("{err}").contains("leading or trailing whitespace"));

        let err = Toolset::<()>::new("core\nnext").unwrap_err();
        assert!(format!("{err}").contains("control characters"));

        let err = Toolset::new("core")
            .unwrap()
            .register(Arc::new(EchoTool::new("echo ")))
            .unwrap_err();
        assert!(format!("{err}").contains("leading or trailing whitespace"));

        let err = Toolset::new("core")
            .unwrap()
            .register(Arc::new(EchoTool::new("echo\nnext")))
            .unwrap_err();
        assert!(format!("{err}").contains("control characters"));
    }

    #[test]
    fn toolset_accepts_free_form_tool_descriptions() {
        let tool = EchoTool {
            metadata: ToolMetadata::function(
                "summarize",
                "Summarize the supplied content in two concise sentences.",
                json!({"type": "object", "properties": {}}),
            ),
        };

        let set = Toolset::new("core")
            .unwrap()
            .register(Arc::new(tool))
            .unwrap();
        assert_eq!(set.names().collect::<Vec<_>>(), vec!["summarize"]);
    }

    #[test]
    fn toolset_rejects_empty_tool_descriptions() {
        let tool = EchoTool {
            metadata: ToolMetadata::function(
                "summarize",
                " ",
                json!({"type": "object", "properties": {}}),
            ),
        };

        let err = Toolset::new("core")
            .unwrap()
            .register(Arc::new(tool))
            .unwrap_err();
        assert!(format!("{err}").contains("description must not be empty"));
    }

    #[test]
    fn toolset_rejects_duplicate_tool_names() {
        let err = Toolset::new("core")
            .unwrap()
            .register(Arc::new(EchoTool::new("echo")))
            .unwrap()
            .register(Arc::new(EchoTool::new("echo")))
            .unwrap_err();
        assert!(format!("{err}").contains("already registered"));
    }

    #[test]
    fn restricted_to_is_strict_and_stable() {
        let set = Toolset::new("core")
            .unwrap()
            .register(Arc::new(EchoTool::new("beta")))
            .unwrap()
            .register(Arc::new(EchoTool::new("alpha")))
            .unwrap();

        let narrowed = set.restricted_to(&["alpha"]).unwrap();
        assert_eq!(narrowed.names().collect::<Vec<_>>(), vec!["alpha"]);

        let err = set.restricted_to(&["alpha", "ghost"]).unwrap_err();
        assert!(format!("{err}").contains("ghost"));

        let err = set.restricted_to(&["alpha "]).unwrap_err();
        assert!(format!("{err}").contains("leading or trailing whitespace"));

        let err = set.restricted_to(&["alpha\nnext"]).unwrap_err();
        assert!(format!("{err}").contains("control characters"));

        let err = set.restricted_to(&["alpha", "alpha"]).unwrap_err();
        assert!(format!("{err}").contains("duplicate tool name"));
    }

    #[tokio::test]
    async fn install_into_preserves_registry_dispatch() {
        let set = Toolset::new("core")
            .unwrap()
            .register(Arc::new(EchoTool::new("echo")))
            .unwrap();
        let registry = set.install_into(ToolRegistry::new()).unwrap();
        let output = registry
            .dispatch(
                "tool_use_1",
                "echo",
                json!({"value": 1}),
                &crate::ExecutionContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(output, json!({"value": 1}));
    }
}
