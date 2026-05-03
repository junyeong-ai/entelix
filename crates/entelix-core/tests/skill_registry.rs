//! `SkillRegistry` behaviour tests — append-only, exact lookup,
//! summary listing, F7-style filter.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::skills::{LoadedSkill, Skill, SkillRegistry};

#[derive(Debug)]
struct TestSkill {
    name: &'static str,
    description: &'static str,
    version: Option<&'static str>,
}

#[async_trait]
impl Skill for TestSkill {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        self.description
    }
    fn version(&self) -> Option<&str> {
        self.version
    }
    async fn load(&self, _ctx: &ExecutionContext) -> Result<LoadedSkill> {
        Ok(LoadedSkill::new(format!("body of {}", self.name)))
    }
}

fn s(
    name: &'static str,
    description: &'static str,
    version: Option<&'static str>,
) -> Arc<dyn Skill> {
    Arc::new(TestSkill {
        name,
        description,
        version,
    })
}

#[test]
fn registry_register_then_lookup_round_trips() {
    let r = SkillRegistry::new()
        .register(s("alpha", "first skill", None))
        .unwrap()
        .register(s("beta", "second skill", Some("0.2.0")))
        .unwrap();
    assert_eq!(r.len(), 2);
    assert!(r.has("alpha"));
    assert!(r.has("beta"));
    assert!(!r.has("gamma"));
    let alpha = r.get("alpha").unwrap();
    assert_eq!(alpha.name(), "alpha");
}

#[test]
fn registry_rejects_duplicate_name() {
    let r = SkillRegistry::new()
        .register(s("alpha", "first", None))
        .unwrap();
    let err = r.register(s("alpha", "duplicate", None)).unwrap_err();
    assert!(format!("{err}").contains("already registered"));
}

#[test]
fn summaries_are_stable_sorted_and_carry_version() {
    let r = SkillRegistry::new()
        .register(s("zulu", "z", Some("1.0")))
        .unwrap()
        .register(s("alpha", "a", None))
        .unwrap()
        .register(s("mike", "m", Some("0.5")))
        .unwrap();
    let summaries = r.summaries();
    let names: Vec<&str> = summaries.iter().map(|s| s.name).collect();
    assert_eq!(names, vec!["alpha", "mike", "zulu"]);
    assert_eq!(summaries[0].version, None);
    assert_eq!(summaries[1].version, Some("0.5"));
}

#[test]
fn filter_narrows_authority_does_not_widen() {
    let parent = SkillRegistry::new()
        .register(s("alpha", "a", None))
        .unwrap()
        .register(s("beta", "b", None))
        .unwrap()
        .register(s("gamma", "c", None))
        .unwrap();
    let child = parent.filter(&["alpha", "gamma", "delta"]);
    // delta is not in parent — silently skipped.
    assert_eq!(child.len(), 2);
    assert!(child.has("alpha"));
    assert!(!child.has("beta"));
    assert!(child.has("gamma"));
    assert!(!child.has("delta"));
}

#[test]
fn filter_yields_independent_registry() {
    let parent = SkillRegistry::new()
        .register(s("alpha", "a", None))
        .unwrap();
    let child = parent.filter(&["alpha"]);
    // Adding to the parent does not retro-update the child view.
    let parent2 = parent.register(s("beta", "b", None)).unwrap();
    assert_eq!(parent2.len(), 2);
    assert_eq!(child.len(), 1);
}
