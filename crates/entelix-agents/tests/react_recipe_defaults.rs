//! `ReActAgentBuilder::with_system` + auto tool-spec injection — the
//! recipe ensures every model dispatch the planner makes carries
//! the operator's system prompt and the active registry's
//! advertised tools, without forcing operators to thread either
//! through `ChatModel::with_tools` / `with_system_prompt` by hand.
//!
//! Pre-`with_system` operators had to wrap the chat runnable in a
//! `system+tools`-stamping newtype to make ReAct converge — these
//! tests pin the ergonomic that retired that boilerplate.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use entelix_agents::{ReActAgentBuilder, ReActState};
use entelix_core::ir::{ContentPart, Message, Role, SystemPrompt};
use entelix_core::tools::{Tool, ToolMetadata};
use entelix_core::{AgentContext, ExecutionContext, Result, RunOverrides, ToolRegistry};
use entelix_runnable::Runnable;

/// Captures whatever [`RunOverrides`] reaches the model on each
/// `invoke` so the test can assert the recipe defaults flowed
/// through. Returns a fixed terminator-shaped reply so the ReAct
/// loop ends after one planner turn.
struct OverrideCapturingModel {
    captured: Arc<Mutex<Option<RunOverrides>>>,
}

#[async_trait]
impl Runnable<Vec<Message>, Message> for OverrideCapturingModel {
    async fn invoke(&self, _input: Vec<Message>, ctx: &ExecutionContext) -> Result<Message> {
        if let Some(arc) = ctx.extension::<RunOverrides>() {
            *self.captured.lock().unwrap() = Some((*arc).clone());
        }
        // No `ToolUse` parts → planner exits to `finish` after one
        // turn.
        Ok(Message::new(Role::Assistant, vec![ContentPart::text("ok")]))
    }
}

/// Trivial tool that exists solely so the registry has at least one
/// advertised spec.
struct StubTool {
    metadata: ToolMetadata,
}

impl StubTool {
    fn new(name: &str) -> Self {
        Self {
            metadata: ToolMetadata::function(
                name,
                "stub for recipe-defaults integration",
                serde_json::json!({"type": "object"}),
            ),
        }
    }
}

#[async_trait]
impl Tool for StubTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &AgentContext<()>,
    ) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"ok": true}))
    }
}

fn registry_with(names: &[&str]) -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    for name in names {
        reg = reg.register(Arc::new(StubTool::new(name))).unwrap();
    }
    reg
}

fn initial_state() -> ReActState {
    ReActState {
        messages: vec![Message::new(Role::User, vec![ContentPart::text("hello")])],
        steps: 0,
    }
}

#[tokio::test]
async fn auto_tool_specs_reach_planner_without_explicit_with_tools_call() {
    let captured = Arc::new(Mutex::new(None));
    let model = OverrideCapturingModel {
        captured: Arc::clone(&captured),
    };
    let agent = ReActAgentBuilder::new(model, registry_with(&["alpha", "beta"]))
        .build()
        .unwrap();
    agent
        .execute(initial_state(), &ExecutionContext::new())
        .await
        .unwrap();
    let overrides = captured
        .lock()
        .unwrap()
        .clone()
        .expect("planner saw RunOverrides");
    let specs = overrides.tool_specs().expect("tool_specs auto-injected");
    let names: Vec<_> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["alpha", "beta"]);
    assert!(
        overrides.system_prompt().is_none(),
        "no with_system call → no system override"
    );
}

#[tokio::test]
async fn with_system_stamps_system_prompt_on_every_dispatch() {
    let captured = Arc::new(Mutex::new(None));
    let model = OverrideCapturingModel {
        captured: Arc::clone(&captured),
    };
    let agent = ReActAgentBuilder::new(model, registry_with(&["alpha"]))
        .with_system(SystemPrompt::text("You are an assistant."))
        .build()
        .unwrap();
    agent
        .execute(initial_state(), &ExecutionContext::new())
        .await
        .unwrap();
    let overrides = captured
        .lock()
        .unwrap()
        .clone()
        .expect("planner saw RunOverrides");
    assert!(overrides.system_prompt().is_some());
    assert_eq!(overrides.tool_specs().unwrap().len(), 1);
}

#[tokio::test]
async fn empty_registry_and_no_with_system_skips_run_overrides_wrapping() {
    // Zero recipe defaults — Configured wrap is bypassed entirely
    // and the model invocation observes no RunOverrides extension.
    let captured = Arc::new(Mutex::new(None));
    let model = OverrideCapturingModel {
        captured: Arc::clone(&captured),
    };
    let agent = ReActAgentBuilder::new(model, ToolRegistry::new())
        .build()
        .unwrap();
    agent
        .execute(initial_state(), &ExecutionContext::new())
        .await
        .unwrap();
    assert!(
        captured.lock().unwrap().is_none(),
        "empty defaults must not insert a RunOverrides extension"
    );
}

#[tokio::test]
async fn caller_supplied_run_overrides_win_over_recipe_defaults() {
    // Operator-provided RunOverrides on the execution context wins —
    // the recipe defaults don't clobber it. Ensures per-call
    // sophistication (different system prompt for one call, different
    // tool subset for one call) survives the recipe's auto-derive.
    let captured = Arc::new(Mutex::new(None));
    let model = OverrideCapturingModel {
        captured: Arc::clone(&captured),
    };
    let agent = ReActAgentBuilder::new(model, registry_with(&["alpha", "beta"]))
        .with_system(SystemPrompt::text("recipe default"))
        .build()
        .unwrap();
    let caller_overrides =
        RunOverrides::new().with_system_prompt(SystemPrompt::text("caller wins"));
    let ctx = ExecutionContext::new().add_extension(caller_overrides);
    agent.execute(initial_state(), &ctx).await.unwrap();
    let overrides = captured
        .lock()
        .unwrap()
        .clone()
        .expect("caller's RunOverrides reached planner");
    let system = overrides.system_prompt().expect("system stayed set");
    let blocks = system.blocks();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "caller wins");
    assert!(
        overrides.tool_specs().is_none(),
        "caller didn't set tool_specs — recipe defaults must NOT leak in"
    );
}
