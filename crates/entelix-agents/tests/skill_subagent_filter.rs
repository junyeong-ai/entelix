//! `Subagent::with_skills` — F7 enforcement for the skill surface.
//!
//! A sub-agent must declare which skills it inherits. The filter
//! narrows the parent's authority; names not in the filter are
//! removed; names in the filter that are absent from the parent are
//! rejected at construction with `Error::Config` so a typo surfaces
//! at the moment it is introduced rather than as a silent
//! "skill not found" at runtime.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use async_trait::async_trait;
use entelix_core::ExecutionContext;
use entelix_core::ToolRegistry;
use entelix_core::ir::Message;
use entelix_core::skills::SkillRegistry;
use entelix_runnable::Runnable;
use entelix_tools::InMemorySkill;

use entelix_agents::Subagent;

#[derive(Debug)]
struct StubModel;

#[async_trait]
impl Runnable<Vec<Message>, Message> for StubModel {
    async fn invoke(
        &self,
        _input: Vec<Message>,
        _ctx: &ExecutionContext,
    ) -> entelix_core::Result<Message> {
        Ok(Message::assistant("ok"))
    }
}

fn parent_skills() -> SkillRegistry {
    let echo = InMemorySkill::builder("echo")
        .with_description("echo")
        .with_instructions("echo body")
        .build()
        .unwrap();
    let sql = InMemorySkill::builder("sql-expert")
        .with_description("sql")
        .with_instructions("sql body")
        .build()
        .unwrap();
    let pii = InMemorySkill::builder("pii-redaction")
        .with_description("pii")
        .with_instructions("pii body")
        .build()
        .unwrap();
    SkillRegistry::new()
        .register(Arc::new(echo))
        .unwrap()
        .register(Arc::new(sql))
        .unwrap()
        .register(Arc::new(pii))
        .unwrap()
}

#[test]
fn subagent_inherits_only_named_skills() {
    let parent = parent_skills();
    let sub = Subagent::builder(
        StubModel,
        &ToolRegistry::new(),
        "test_subagent",
        "test description",
    )
    .restrict_to(&[])
    .with_skills(&parent, &["echo", "sql-expert"])
    .build()
    .unwrap();
    let inherited = sub.skills();
    assert_eq!(inherited.len(), 2);
    assert!(inherited.has("echo"));
    assert!(inherited.has("sql-expert"));
    assert!(!inherited.has("pii-redaction"));
}

#[test]
fn subagent_with_skills_rejects_missing_names() {
    let parent = parent_skills();
    let err = Subagent::builder(
        StubModel,
        &ToolRegistry::new(),
        "test_subagent",
        "test description",
    )
    .restrict_to(&[])
    .with_skills(&parent, &["echo", "ghost-skill", "sql-expert"])
    .build()
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ghost-skill"),
        "error must name the missing skill, got: {msg}"
    );
}

#[test]
fn subagent_default_skills_are_empty_no_inheritance_without_explicit_call() {
    // F7 mitigation: never auto-inherit. The default for a freshly
    // built sub-agent is an empty skill registry.
    let sub = Subagent::builder(
        StubModel,
        &ToolRegistry::new(),
        "test_subagent",
        "test description",
    )
    .restrict_to(&[])
    .build()
    .unwrap();
    assert!(sub.skills().is_empty());
}
