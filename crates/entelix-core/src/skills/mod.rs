//! Skills — packaged, progressively-disclosed agent capabilities
//! (ADR-0027).
//!
//! A [`Skill`] bundles four things: a stable identifier, a one-line
//! description, an instructions body, and a set of named resources.
//! The model reaches them in three tiers:
//!
//! - **T1** (always loaded): `name + description`, emitted by
//!   [`SkillRegistry::summaries`].
//! - **T2** (loaded on activation): the full `instructions` body,
//!   returned by [`Skill::load`].
//! - **T3** (loaded on demand): each [`SkillResource`] is read only
//!   when the model invokes
//!   [`crate::tools::Tool`]-fronted resource access.
//!
//! Token cost grows proportionally with what the model uses, not with
//! what the operator has registered.
//!
//! ## Backend-agnostic
//!
//! `Skill` is a capability trait: implementations can be hand-rolled
//! structs ([`crate::skills::Skill`] is `Send + Sync`), sandbox-backed
//! file trees, MCP-fronted servers, or HTTP stores. The runtime
//! contract carries an [`ExecutionContext`] through every async call
//! so backends inherit `tenant_id` (invariant 11) and cancellation.
//!
//! ## Sub-agent filtering
//!
//! [`SkillRegistry::filter`] mirrors the tool-filter pattern (F7) —
//! sub-agents receive an explicitly-named subset; no inheritance of
//! unnamed skills.
//!
//! [`ExecutionContext`]: crate::context::ExecutionContext

mod registry;
mod resource;
mod skill;

pub use registry::{SkillRegistry, SkillSummary};
pub use resource::{SkillResource, SkillResourceContent};
pub use skill::{LoadedSkill, Skill};
