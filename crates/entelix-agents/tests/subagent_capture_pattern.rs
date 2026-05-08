//! Sub-agent constructor *capture-pattern audit* — locks down the
//! brain↔hand contract beyond layer inheritance (covered by
//! `subagent_layer_inheritance.rs`).
//!
//! The constructors split on strictness:
//!
//! - `Subagent::builder(model, parent).restrict_to(allowed_names).build()` — strict.
//!   Every name in `allowed` must exist in the parent registry; a typo
//!   surfaces as `Error::Config` at construction time. Duplicate names
//!   silently dedup (`HashSet` semantic) — the deduped view is correct.
//! - `Subagent::builder(model, parent).filter(predicate).build()` — graceful.
//!   The predicate is `Fn(&dyn Tool) -> bool`, evaluated once per
//!   parent tool at construction; the resulting view is frozen.
//!   Empty results are valid (a "pure orchestration" sub-agent with
//!   no tools), an explicit shape choice the doc records.
//!
//! The skill side is the mirror image with `Subagent::with_skills`
//! (strict — typo errors at construction) and the lower-level
//! `SkillRegistry::filter` (silent-skip primitive that
//! `with_skills` wraps).
//!
//! These tests pin the asymmetry: a regression that swaps a strict
//! constructor for a silent one — or vice versa — fails here.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};

use entelix_agents::Subagent;
use entelix_core::AgentContext;
use entelix_core::ExecutionContext;
use entelix_core::Result;
use entelix_core::ir::Message;
use entelix_core::tools::{Tool, ToolMetadata};
use entelix_core::{SkillRegistry, ToolRegistry};
use entelix_runnable::Runnable;

struct EchoTool {
    metadata: ToolMetadata,
}

impl EchoTool {
    fn new(name: &str) -> Self {
        Self {
            metadata: ToolMetadata::function(name, "echo input", json!({"type": "object"})),
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

#[derive(Debug)]
struct StubModel;

#[async_trait]
impl Runnable<Vec<Message>, Message> for StubModel {
    async fn invoke(&self, _input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
        Ok(Message::assistant("ok"))
    }
}

fn parent_with(names: &[&str]) -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    for name in names {
        reg = reg
            .register(Arc::new(EchoTool::new(name)) as Arc<dyn Tool>)
            .unwrap();
    }
    reg
}

#[test]
fn restrict_to_rejects_typo_at_build_time() {
    // Strict-name contract — a missing name surfaces as
    // `Error::Config` rather than producing a sub-agent that
    // silently can't reach the tool the operator intended.
    let parent = parent_with(&["alpha", "beta"]);
    let err = Subagent::builder(StubModel, &parent, "test_subagent", "test description")
        .restrict_to(&["alpha", "ghost"])
        .build()
        .unwrap_err();
    let rendered = format!("{err}");
    assert!(
        rendered.contains("ghost") && rendered.contains("not in registry"),
        "expected diagnostic naming the missing tool; got: {rendered}"
    );
}

#[test]
fn restrict_to_dedup_handles_duplicate_names_correctly() {
    // Duplicate names in the whitelist must produce a view of the
    // *unique* set — not an error, not a duplicated tool. The
    // underlying HashSet conversion handles this; a regression that
    // swapped to Vec-based comparison would surface as either an
    // inflated tool_count or a spurious "already-registered" error.
    let parent = parent_with(&["alpha", "beta"]);
    let sub = Subagent::builder(StubModel, &parent, "test_subagent", "test description")
        .restrict_to(&["alpha", "alpha"])
        .build()
        .unwrap();
    assert_eq!(sub.tool_count(), 1);
}

#[test]
fn filter_predicate_evaluated_once_per_parent_tool() {
    // The closure is `Fn(&dyn Tool) -> bool` — one call per parent
    // entry at construction time. After construction the view is
    // frozen; subsequent dispatches don't re-evaluate.
    //
    // A regression that lazily re-evaluated the predicate per
    // dispatch (or stored it in the registry to re-run) would
    // overcount here — the dispatch later in the test would bump
    // the counter past `parent.len()`.
    let parent = parent_with(&["alpha", "beta", "gamma"]);
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in = Arc::clone(&calls);

    let sub = Subagent::builder(StubModel, &parent, "test_subagent", "test description")
        .filter(move |t| {
            calls_in.fetch_add(1, Ordering::SeqCst);
            t.metadata().name == "beta"
        })
        .build()
        .unwrap();
    let post_construction = calls.load(Ordering::SeqCst);
    assert_eq!(
        post_construction, 3,
        "predicate should be evaluated once per parent tool at construction"
    );
    assert_eq!(sub.tool_count(), 1);

    // Dispatch through the narrowed view — must NOT re-fire the
    // predicate; the registry holds the frozen filter result.
    let _ = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            sub.tool_registry()
                .dispatch("call", "beta", json!({}), &ExecutionContext::new())
                .await
        })
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        post_construction,
        "predicate must NOT re-fire on dispatch — view is frozen at construction"
    );
}

#[test]
fn filter_empty_result_is_valid_pure_orchestration_subagent() {
    // The doc records this as a deliberate trade-off: the predicate
    // form cannot detect "intended but missing", so an empty result
    // is the operator's responsibility. A regression that started
    // erroring on empty filter would break legitimate
    // pure-orchestration sub-agent shapes.
    let parent = parent_with(&["alpha", "beta"]);
    let sub = Subagent::builder(StubModel, &parent, "test_subagent", "test description")
        .filter(|_| false)
        .build()
        .unwrap();
    assert_eq!(sub.tool_count(), 0);
}

#[test]
fn metadata_inspect_without_consume() {
    // Slice 111 / ADR-0093: identity is set at builder construction
    // and is inspectable on the built `Subagent` via
    // `name()` / `description()` / `metadata()` *before* the
    // `into_tool()` conversion that consumes self. This is the
    // load-bearing capability for parent-side system-prompt
    // enrichment (parent agent lists "available sub-agents:
    // {name} — {description}" without dropping the Subagent).
    let parent = parent_with(&["alpha", "beta"]);
    let sub = Subagent::builder(
        StubModel,
        &parent,
        "research_assistant",
        "Search the web for citations.",
    )
    .restrict_to(&["alpha"])
    .build()
    .unwrap();

    assert_eq!(sub.name(), "research_assistant");
    assert_eq!(sub.description(), "Search the web for citations.");

    let md = sub.metadata();
    assert_eq!(md.name, "research_assistant");
    assert_eq!(md.description, "Search the web for citations.");
    assert_eq!(md.tool_count, 1);
    assert_eq!(md.tool_names, vec!["alpha".to_owned()]);

    // The Subagent is still usable — metadata() returned a snapshot
    // by clone, not a consuming move.
    assert_eq!(sub.tool_count(), 1);
}

#[test]
fn build_rejects_empty_name() {
    // Identity is required at the type level (`builder` signature
    // takes `impl Into<String>`) but `""` is still a valid
    // `String`. `build()` rejects it with `Error::Config` so
    // operators get a clear diagnostic at construction instead of
    // a confusing dispatch failure when the LLM receives an
    // empty tool name.
    let parent = parent_with(&["alpha"]);
    let err = Subagent::builder(StubModel, &parent, "", "test description")
        .build()
        .unwrap_err();
    let rendered = format!("{err}");
    assert!(
        rendered.contains("name cannot be empty"),
        "expected diagnostic naming the empty field; got: {rendered}"
    );
}

#[test]
fn build_rejects_empty_description() {
    // Mirror of the empty-name rejection. Description carries the
    // sub-agent's purpose to the parent's LLM; an empty string
    // there ships an unhelpful tool listing.
    let parent = parent_with(&["alpha"]);
    let err = Subagent::builder(StubModel, &parent, "test_subagent", "")
        .build()
        .unwrap_err();
    let rendered = format!("{err}");
    assert!(
        rendered.contains("description cannot be empty"),
        "expected diagnostic naming the empty field; got: {rendered}"
    );
}

#[test]
fn with_skills_rejects_typo_at_construction_time() {
    // Skill side mirrors `restrict_to` — strict, surfaces typos
    // as `Error::Config`. The lower-level `SkillRegistry::filter`
    // is silent-skip; `with_skills` is the strict wrapper around
    // it. A regression that swapped to direct `filter` would lose
    // the typo-detection.
    let parent = parent_with(&["alpha"]);
    let parent_skills = SkillRegistry::new();
    let err = Subagent::builder(StubModel, &parent, "test_subagent", "test description")
        .restrict_to(&["alpha"])
        .with_skills(&parent_skills, &["nonexistent"])
        .build()
        .unwrap_err();
    let rendered = format!("{err}");
    assert!(
        rendered.contains("nonexistent") && rendered.contains("not in parent registry"),
        "expected diagnostic naming the missing skill; got: {rendered}"
    );
}
