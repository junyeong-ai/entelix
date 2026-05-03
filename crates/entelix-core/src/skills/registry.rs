//! `SkillRegistry` — append-only init-time collection of skills with
//! exact-name lookup, T1 summary listing, and F7 sub-agent filtering.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::skills::skill::Skill;

/// Init-time append-only collection of skills.
///
/// Cloning is cheap (`Arc<HashMap>` internally), and
/// [`Self::filter`] produces a restricted view for sub-agent scope —
/// F7 parity with [`crate::tools::ToolRegistry`].
#[derive(Clone, Default, Debug)]
pub struct SkillRegistry {
    skills: Arc<HashMap<String, Arc<dyn Skill>>>,
}

impl SkillRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one skill. Returns `Error::Config` when a skill with
    /// the same `name()` is already registered — duplicates are
    /// always a programmer error (a skill catalogue with two
    /// `code-review` entries cannot resolve unambiguously).
    pub fn register(self, skill: Arc<dyn Skill>) -> Result<Self> {
        let name = skill.name().to_owned();
        if self.skills.contains_key(&name) {
            return Err(Error::config(format!(
                "SkillRegistry: skill {name:?} is already registered"
            )));
        }
        // Cheap copy-on-write — the registry is only mutated at
        // init time so cloning the inner map per registration is fine.
        let mut next = (*self.skills).clone();
        next.insert(name, skill);
        Ok(Self {
            skills: Arc::new(next),
        })
    }

    /// Look up a skill by exact name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Skill>> {
        self.skills.get(name)
    }

    /// True when the named skill is registered.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    /// Number of registered skills.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// True when no skill is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// T1 listing — name + description + version for every skill.
    /// Stable-ordered (sorted by name) so consumers can rely on a
    /// canonical sequence.
    #[must_use]
    pub fn summaries(&self) -> Vec<SkillSummary<'_>> {
        let mut entries: Vec<&Arc<dyn Skill>> = self.skills.values().collect();
        entries.sort_by_key(|s| s.name());
        entries
            .into_iter()
            .map(|s| SkillSummary {
                name: s.name(),
                description: s.description(),
                version: s.version(),
            })
            .collect()
    }

    /// Restricted view for sub-agent scope — produces a new registry
    /// containing only the named skills. F7 parity: the parent's
    /// authority is *narrowed*, never widened. Names not present in
    /// the parent registry are silently skipped (callers asserting
    /// presence should `has(name)` first).
    #[must_use]
    pub fn filter(&self, allowed: &[&str]) -> Self {
        let mut next: HashMap<String, Arc<dyn Skill>> = HashMap::with_capacity(allowed.len());
        for name in allowed {
            if let Some(skill) = self.skills.get(*name) {
                next.insert((*name).to_owned(), Arc::clone(skill));
            }
        }
        Self {
            skills: Arc::new(next),
        }
    }
}

/// T1 summary view of one registered skill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkillSummary<'a> {
    /// Stable identifier.
    pub name: &'a str,
    /// One-line description shown to the model in the listing tool.
    pub description: &'a str,
    /// Optional version string.
    pub version: Option<&'a str>,
}
