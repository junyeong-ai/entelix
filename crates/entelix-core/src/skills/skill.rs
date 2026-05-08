//! `Skill` trait + `LoadedSkill` data — the trait surface every
//! skill implementation honours.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::ExecutionContext;
use crate::error::Result;
use crate::skills::resource::SkillResource;

/// A packaged, progressively-disclosed agent capability.
///
/// Implementations are stateless containers — `Send + Sync + Debug`
/// without interior mutability is the typical shape. State that
/// changes per call (e.g., a sandbox connection's open file handle)
/// belongs in the backing service, not the `Skill` itself.
///
/// See for the three-tier disclosure model.
#[async_trait]
pub trait Skill: Send + Sync + std::fmt::Debug {
    /// Stable kebab-case identifier. Must be unique within a
    /// `SkillRegistry`.
    fn name(&self) -> &str;

    /// One-line description. Surfaced in the registry's summary
    /// listing so the model can decide whether to activate. Should
    /// say *what* the skill does and *when* to use it in the same
    /// sentence — front-load the key use case.
    fn description(&self) -> &str;

    /// Optional semver string. Surfaces in audit traces and the
    /// `list_skills` tool output.
    fn version(&self) -> Option<&str> {
        None
    }

    /// Load the full skill body. Fires when an activator (the
    /// `activate_skill` tool, a recipe, or a host application) opts
    /// the skill in. The returned [`LoadedSkill`] carries the
    /// instructions text and any on-demand resources.
    async fn load(&self, ctx: &ExecutionContext) -> Result<LoadedSkill>;
}

/// The body produced by [`Skill::load`].
///
/// Resources are listed *by key only* in the LLM-facing tool result —
/// the bytes are read separately via [`SkillResource::read`] when the
/// model invokes the resource-read tool. This keeps T3 truly
/// on-demand: a 50-resource skill doesn't pay for any of them until
/// the model asks for one.
#[derive(Debug)]
pub struct LoadedSkill {
    /// Instructions appended to the agent's system prompt for the
    /// activated turn(s).
    pub instructions: String,

    /// Resource readers keyed by relative path
    /// (e.g. `"examples/quickstart.md"`,
    /// `"reference/api.json"`).
    pub resources: BTreeMap<String, Arc<dyn SkillResource>>,
}

impl LoadedSkill {
    /// Build with an instructions body and no resources.
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            resources: BTreeMap::new(),
        }
    }

    /// Add (or overwrite) one resource.
    #[must_use]
    pub fn with_resource(
        mut self,
        key: impl Into<String>,
        resource: Arc<dyn SkillResource>,
    ) -> Self {
        self.resources.insert(key.into(), resource);
        self
    }

    /// Sorted list of resource keys — useful for emitting the T3
    /// menu without exposing the resource handles.
    #[must_use]
    pub fn resource_keys(&self) -> Vec<&str> {
        self.resources.keys().map(String::as_str).collect()
    }
}
