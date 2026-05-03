//! `InMemorySkill` — `Skill` implementation whose body and resources
//! live in struct fields. The natural shape for embedded skills,
//! tests, and small operator-defined catalogues.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::skills::{LoadedSkill, Skill, SkillResource, SkillResourceContent};

/// `Skill` whose body lives in struct fields. Construct via
/// [`InMemorySkillBuilder`].
///
/// Cloning is cheap (`Arc` over the resource map). `Send + Sync`
/// because every field is.
#[derive(Clone, Debug)]
pub struct InMemorySkill {
    name: String,
    description: String,
    version: Option<String>,
    instructions: String,
    resources: Arc<BTreeMap<String, Arc<dyn SkillResource>>>,
}

impl InMemorySkill {
    /// Start a builder for an in-memory skill.
    #[must_use]
    pub fn builder(name: impl Into<String>) -> InMemorySkillBuilder {
        InMemorySkillBuilder {
            name: name.into(),
            description: String::new(),
            version: None,
            instructions: String::new(),
            resources: BTreeMap::new(),
        }
    }
}

#[async_trait]
impl Skill for InMemorySkill {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    async fn load(&self, _ctx: &ExecutionContext) -> Result<LoadedSkill> {
        Ok(LoadedSkill {
            instructions: self.instructions.clone(),
            resources: (*self.resources).clone(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use entelix_core::context::ExecutionContext;
    use entelix_core::skills::Skill;

    use super::*;

    #[tokio::test]
    async fn loaded_skill_carries_instructions_and_resource_keys() {
        let skill = InMemorySkill::builder("t")
            .with_description("d")
            .with_instructions("body")
            .with_text_resource("k", "v")
            .build()
            .unwrap();
        let loaded = skill.load(&ExecutionContext::new()).await.unwrap();
        assert_eq!(loaded.instructions, "body");
        assert_eq!(loaded.resource_keys(), vec!["k"]);
    }
}

/// Fluent builder for [`InMemorySkill`].
#[derive(Debug)]
pub struct InMemorySkillBuilder {
    name: String,
    description: String,
    version: Option<String>,
    instructions: String,
    resources: BTreeMap<String, Arc<dyn SkillResource>>,
}

impl InMemorySkillBuilder {
    /// Set the one-line description (T1 — always loaded).
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the optional semver version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Set the instructions body (T2 — loaded on activation).
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Insert a T3 resource. `key` is the lookup name the model
    /// passes to `read_skill_resource`; the body resolves lazily
    /// when that tool is invoked.
    #[must_use]
    pub fn with_resource(
        mut self,
        key: impl Into<String>,
        resource: Arc<dyn SkillResource>,
    ) -> Self {
        self.resources.insert(key.into(), resource);
        self
    }

    /// Convenience wrapper around [`Self::with_resource`] that wraps `text`
    /// in a [`StaticResource`] before inserting it.
    #[must_use]
    pub fn with_text_resource(self, key: impl Into<String>, text: impl Into<String>) -> Self {
        self.with_resource(
            key,
            Arc::new(StaticResource::text(text)) as Arc<dyn SkillResource>,
        )
    }

    /// Returns `Result` for forward-compat with future validation
    /// (e.g. name uniqueness, manifest-style format checks); today
    /// the call cannot fail.
    pub fn build(self) -> Result<InMemorySkill> {
        Ok(InMemorySkill {
            name: self.name,
            description: self.description,
            version: self.version,
            instructions: self.instructions,
            resources: Arc::new(self.resources),
        })
    }
}

/// `SkillResource` whose payload lives in a struct field — the
/// natural shape for embedded resources and tests.
#[derive(Clone, Debug)]
pub struct StaticResource {
    content: SkillResourceContent,
}

impl StaticResource {
    /// Build a text-payload resource.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: SkillResourceContent::Text(text.into()),
        }
    }

    /// Build a binary-payload resource.
    #[must_use]
    pub fn binary(mime_type: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            content: SkillResourceContent::Binary {
                mime_type: mime_type.into(),
                bytes,
            },
        }
    }
}

#[async_trait]
impl SkillResource for StaticResource {
    async fn read(&self, _ctx: &ExecutionContext) -> Result<SkillResourceContent> {
        Ok(self.content.clone())
    }
}
