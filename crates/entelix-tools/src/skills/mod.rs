//! Concrete `Skill` implementations and the three LLM-facing tools
//! that drive progressive disclosure (ADR-0027).
//!
//! ## Skills
//!
//! - [`InMemorySkill`] — struct holding name, description, version,
//!   instructions, and a resource map. Build via
//!   [`InMemorySkillBuilder`].
//! - [`SandboxSkill`] — backed by a sandbox-internal directory tree
//!   mirroring the Anthropic Claude Skills layout (`SKILL.md` for
//!   instructions, any other relative file becomes a resource).
//!   Filesystem access flows through `Sandbox` — invariant 9
//!   preserved.
//!
//! ## LLM-facing tools
//!
//! - [`ListSkillsTool`] — argument-less listing of registered skills
//!   (T1 metadata).
//! - [`ActivateSkillTool`] — load a skill's instructions and resource
//!   menu (T1 → T2).
//! - [`ReadSkillResourceTool`] — fetch one resource by key (T2 → T3).
//!   Binary resources surface as metadata only (`mime_type`,
//!   `size_bytes`, `sha256`); the bytes are accessible via the
//!   `SkillRegistry` API for out-of-band host-application use.

mod in_memory;
mod manifest;
mod sandbox_skill;
mod tools;

use std::sync::Arc;

use entelix_core::error::Result;
use entelix_core::skills::SkillRegistry;
use entelix_core::tools::ToolRegistry;

pub use in_memory::{InMemorySkill, InMemorySkillBuilder, StaticResource};
pub use manifest::{ManifestError, SkillManifest, parse_skill_md};
pub use sandbox_skill::{SandboxResource, SandboxSkill};
pub use tools::{ActivateSkillTool, ListSkillsTool, ReadSkillResourceTool};

/// Register the three LLM-facing skill tools (`list_skills`,
/// `activate_skill`, `read_skill_resource`) into a `ToolRegistry`,
/// each backed by the supplied `SkillRegistry`.
///
/// Recipes that accept both a `ToolRegistry` and a `SkillRegistry`
/// call this helper to surface the skills to the model — the agent
/// runtime itself does not auto-wire (per ADR-0027 §"Sub-agent
/// filtering"; the per-recipe call is the explicit opt-in).
///
/// Returns `Error::Config` from the underlying `ToolRegistry::register`
/// if any of the three names already collides — recipes typically
/// build a fresh registry, so the collision path indicates a
/// programmer error.
pub fn install(tool_registry: ToolRegistry, skill_registry: SkillRegistry) -> Result<ToolRegistry> {
    tool_registry
        .register(Arc::new(ListSkillsTool::new(skill_registry.clone())))?
        .register(Arc::new(ActivateSkillTool::new(skill_registry.clone())))?
        .register(Arc::new(ReadSkillResourceTool::new(skill_registry)))
}
