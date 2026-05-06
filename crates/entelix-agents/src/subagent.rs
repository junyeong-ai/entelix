//! `Subagent` — a brain↔hand pairing where both the tool surface and
//! the skill surface are explicit, filtered subsets of the parent's
//! authority. F7 mitigation: there is **no default constructor** —
//! every `Subagent` must declare which tools (and skills) it can use.
//!
//! ## Layer-stack inheritance (managed-agent shape)
//!
//! Sub-agents take the parent's [`ToolRegistry`] directly and narrow
//! it through [`ToolRegistry::restricted_to`] / [`ToolRegistry::filter`].
//! The `Arc`-backed layer factory rides over verbatim — every
//! cross-cutting concern attached at the parent (`PolicyLayer` for
//! PII redaction and quota, `OtelLayer` for `gen_ai.tool.*` events,
//! retry middleware) applies transparently to the sub-agent's
//! dispatches. There is no path through this module that constructs
//! a fresh `ToolRegistry::new()` — building one would silently drop
//! the layer stack, and `scripts/check-managed-shape.sh` rejects the
//! pattern statically.
//!
//! ## Sub-agent as tool — handoff
//!
//! Calling [`Subagent::into_tool`] consumes the sub-agent and
//! returns a [`SubagentTool`] that satisfies the [`Tool`] trait. The
//! parent's LLM dispatches the sub-agent like any other tool — the
//! `task` field on the input is rendered as a fresh user message,
//! the sub-agent's full ReAct loop runs, and the terminal assistant
//! text rides back as the tool output. This is the Anthropic
//! managed-agent "agent-as-tool" pattern in code form.
//!
//! Authority bounds carry through verbatim — the sub-agent inside
//! [`SubagentTool`] sees only the tools and skills the operator
//! whitelisted at construction. F7 holds at the dispatch boundary
//! (the parent's tool registry exposes the wrapper, not the inner
//! tools).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use entelix_core::ir::{ContentPart, Message, Role};
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};
use entelix_core::{AgentContext, Error, ExecutionContext, Result, SkillRegistry, ToolRegistry};
use entelix_runnable::Runnable;

use crate::agent::{AgentEventSink, Approver};
use crate::react_agent::react_agent_builder;
use crate::state::ReActState;

/// A bounded brain↔hand pairing.
///
/// The agent loop uses `model` as the brain, dispatching tools
/// through `tool_registry` — a narrowed view of the parent registry
/// that inherits the parent's layer stack at zero copy cost (see
/// module docs for the managed-agent rationale).
///
/// The constructor mandates an explicit filter (or whitelist). This
/// is the F7 mitigation: there is no `Default`, no `with_all_tools`
/// shortcut. Callers must say which tools and which skills they
/// trust the sub-agent with.
pub struct Subagent<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    name: String,
    description: String,
    model: Arc<M>,
    tool_registry: ToolRegistry,
    skills: SkillRegistry,
    sink: Option<Arc<dyn AgentEventSink<ReActState>>>,
    approver: Option<Arc<dyn Approver>>,
}

/// Compact metadata snapshot of a [`Subagent`] for parent-side
/// inspection — the LLM-facing identity (`name`, `description`)
/// plus the tool surface bound at construction. Operators that
/// list available sub-agents in a parent agent's system prompt
/// reach for this struct rather than calling each accessor
/// individually.
///
/// The `description` is the same one-line summary the
/// [`Subagent::builder`] constructor received; longer dev-side
/// documentation belongs in code comments, not in metadata.
#[derive(Clone, Debug)]
pub struct SubagentMetadata {
    /// LLM-facing tool name. Same value [`Subagent::name`] returns.
    pub name: String,
    /// One-line description for parent-side tool listings.
    pub description: String,
    /// Tools the sub-agent can dispatch (count).
    pub tool_count: usize,
    /// Tool names the sub-agent can dispatch — order unspecified.
    pub tool_names: Vec<String>,
}

impl<M> std::fmt::Debug for Subagent<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `model` is intentionally omitted — `M` is not bounded on
        // `Debug` and forwarding would force every model wrapper to
        // implement it. Counts are sufficient for diagnostic
        // purposes; structural fields are surfaced via accessors.
        f.debug_struct("Subagent")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("tool_count", &self.tool_registry.len())
            .field("skill_count", &self.skills.len())
            .field("has_sink", &self.sink.is_some())
            .field("has_approver", &self.approver.is_some())
            .finish_non_exhaustive()
    }
}

impl<M> Subagent<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    /// Open a [`SubagentBuilder`] anchored at `parent_registry`. The
    /// builder is the sole construction surface — operators set
    /// the LLM-facing identity (`name`, `description`), choose the
    /// tool selection (`restrict_to` / `filter`), and attach optional
    /// `sink` / `approver` / `skills` before calling
    /// [`SubagentBuilder::build`].
    ///
    /// `name` and `description` flow through `into_tool` to populate
    /// [`SubagentTool`]'s metadata; declaring them at the builder
    /// surface (not at `into_tool`) makes the sub-agent's identity
    /// inspectable via [`Subagent::name`] / [`Subagent::description`]
    /// before the conversion-to-tool boundary, which matters for
    /// operators that list sub-agent metadata in a parent agent's
    /// system prompt.
    pub fn builder(
        model: M,
        parent_registry: &ToolRegistry,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> SubagentBuilder<'_, M> {
        SubagentBuilder::new(model, parent_registry, name.into(), description.into())
    }

    /// LLM-facing name surfaced to the parent's tool dispatch.
    /// Equal to the [`SubagentTool`]'s metadata name after
    /// [`Self::into_tool`].
    pub fn name(&self) -> &str {
        &self.name
    }

    /// One-line description surfaced to the parent's tool listing.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Compact metadata snapshot for parent-side inspection.
    /// Convenient when the parent's system prompt enumerates
    /// available sub-agents and their tool surfaces without
    /// consuming the [`Subagent`] (which `into_tool` does).
    #[must_use]
    pub fn metadata(&self) -> SubagentMetadata {
        SubagentMetadata {
            name: self.name.clone(),
            description: self.description.clone(),
            tool_count: self.tool_registry.len(),
            tool_names: self.tool_registry.names().map(str::to_owned).collect(),
        }
    }

    /// Number of tools the sub-agent can dispatch.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.tool_registry.len()
    }

    /// Names of the tools the sub-agent can dispatch — useful for
    /// audit log entries. Order is unspecified.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.tool_registry.names().collect()
    }

    /// Borrow the narrowed tool registry the sub-agent dispatches
    /// through. The view inherits the parent's layer stack —
    /// `PolicyLayer`, `OtelLayer`, retry middleware all apply.
    #[must_use]
    pub const fn tool_registry(&self) -> &ToolRegistry {
        &self.tool_registry
    }

    /// Borrow the filtered skill registry the sub-agent inherited.
    /// Empty when [`SubagentBuilder::with_skills`] was never called.
    #[must_use]
    pub const fn skills(&self) -> &SkillRegistry {
        &self.skills
    }

    /// Build a `ReAct` loop bound to this sub-agent's narrowed tool
    /// registry (which inherits the parent's layer stack), model, and
    /// — when [`SubagentBuilder::with_skills`] was called — the
    /// three LLM-facing skill tools (`list_skills`, `activate_skill`,
    /// `read_skill_resource`) backed by the inherited skill registry.
    /// See ADR-0027 §"Auto-wire" and ADR-0035
    /// §"Sub-agent layer-stack inheritance". Sub-agents with no
    /// `with_skills` call build the registry without skill tools,
    /// matching their declared authority.
    pub fn into_react_agent(self) -> Result<crate::agent::Agent<ReActState>> {
        let Self {
            name: _,
            description: _,
            model,
            tool_registry,
            skills,
            sink,
            approver,
        } = self;
        let model = ArcRunnable::new(model);
        let registry_with_skills = if skills.is_empty() {
            tool_registry
        } else {
            // `skills::install` returns the same registry with three
            // additional `register(...)` calls — the layer stack is
            // preserved (registry `register` is `Self -> Self` so
            // `factory: Option<LayerFactory>` rides over).
            entelix_tools::skills::install(tool_registry, skills)?
        };
        // When an approver is configured, wrap the (skills-augmented)
        // registry with `ApprovalLayer` so every tool dispatch this
        // sub-agent issues passes through `Approver::decide` first.
        // Mirrors `ReActAgentBuilder::build` (ADR-0070) — operators
        // get HITL from `Subagent::with_approver(approver)` alone,
        // no extra registry wiring step.
        let registry = match &approver {
            Some(approver) => {
                registry_with_skills.layer(crate::agent::ApprovalLayer::new(Arc::clone(approver)))
            }
            None => registry_with_skills,
        };
        let mut builder = react_agent_builder(model, registry)?;
        if let Some(sink) = sink {
            builder = builder.with_sink_arc(sink);
        }
        if let Some(approver) = approver {
            builder = builder
                .with_execution_mode(crate::agent::ExecutionMode::Supervised)
                .with_approver_arc(approver);
        }
        builder.build()
    }

    /// Wrap this sub-agent as a [`Tool`] callable from the parent's
    /// LLM. The resulting [`SubagentTool`] reports a metadata block
    /// keyed by `name` / `description` and accepts a single
    /// `task: string` input which is rendered as the first user
    /// message of the sub-agent's ReAct loop.
    ///
    /// # Effect classification
    ///
    /// Sub-agents may dispatch arbitrary tools — without inspecting
    /// every transitive tool, a conservative
    /// [`ToolEffect::Mutating`] is reported. Operators that know the
    /// sub-agent is read-only override via [`SubagentTool::with_effect`].
    pub fn into_tool(self) -> Result<SubagentTool> {
        let Self {
            name,
            description,
            ..
        } = &self;
        let name = name.clone();
        let description = description.clone();
        let agent = self.into_react_agent()?;
        Ok(SubagentTool::new(agent, name, description))
    }
}

/// Sub-agent selection surface — `All`, strict `Restrict`, or
/// graceful `Filter`. Constructed via the [`SubagentBuilder`] verbs
/// `restrict_to` / `filter` / (default) `All`. The strict / graceful
/// asymmetry is intentional: name-typos in `restrict_to` fail at
/// construction (the operator declared a known set), whereas `filter`
/// accepts an empty result (a predicate is allowed to match nothing).
/// Boxed predicate over the parent registry's tool set; matches the
/// shape `ToolRegistry::filter` consumes. Held in a type alias so
/// the [`SubagentSelection::Filter`] variant stays readable.
type ToolPredicate = Box<dyn Fn(&dyn Tool) -> bool + Send + Sync>;

enum SubagentSelection {
    All,
    Restrict(Vec<String>),
    Filter(ToolPredicate),
}

impl SubagentSelection {
    fn apply(self, parent: &ToolRegistry) -> Result<ToolRegistry> {
        match self {
            Self::All => Ok(parent.clone()),
            Self::Restrict(allowed) => {
                let refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
                parent.restricted_to(&refs)
            }
            Self::Filter(predicate) => Ok(parent.filter(|tool| predicate(tool))),
        }
    }
}

/// Builder for [`Subagent`]. Construct via
/// [`Subagent::builder(model, &parent_registry)`](Subagent::builder).
///
/// The builder is the sole construction path — the sub-agent's
/// authority bounds (`restrict_to` / `filter`) and optional wiring
/// (`with_sink` / `with_approver` / `with_skills`) compose fluently
/// before [`Self::build`] finalises a [`Subagent`]. Authority bounds
/// are mandatory in spirit (F7 mitigation): a builder that never
/// calls `restrict_to` / `filter` produces a sub-agent inheriting
/// every parent tool — operators making that choice must do so
/// explicitly.
pub struct SubagentBuilder<'a, M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    model: M,
    parent_registry: &'a ToolRegistry,
    name: String,
    description: String,
    selection: SubagentSelection,
    skills_request: Option<(&'a SkillRegistry, Vec<String>)>,
    sink: Option<Arc<dyn AgentEventSink<ReActState>>>,
    approver: Option<Arc<dyn Approver>>,
}

impl<'a, M> SubagentBuilder<'a, M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    fn new(
        model: M,
        parent_registry: &'a ToolRegistry,
        name: String,
        description: String,
    ) -> Self {
        Self {
            model,
            parent_registry,
            name,
            description,
            selection: SubagentSelection::All,
            skills_request: None,
            sink: None,
            approver: None,
        }
    }

    /// Restrict the sub-agent to tools whose name appears in
    /// `allowed`. Returns [`entelix_core::Error::Config`] at
    /// [`Self::build`] time if any name is absent from the parent
    /// registry — strict-name lookup catches typos at the build
    /// boundary rather than at runtime tool dispatch.
    #[must_use]
    pub fn restrict_to(mut self, allowed: &[&str]) -> Self {
        let owned: Vec<String> = allowed.iter().map(|s| (*s).to_owned()).collect();
        self.selection = SubagentSelection::Restrict(owned);
        self
    }

    /// Restrict the sub-agent to tools matching `predicate`. Unlike
    /// [`Self::restrict_to`], an empty match set is accepted — the
    /// predicate form trades typo-detection for arbitrary-shape
    /// flexibility.
    #[must_use]
    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&dyn Tool) -> bool + Send + Sync + 'static,
    {
        self.selection = SubagentSelection::Filter(Box::new(predicate));
        self
    }

    /// Forward the sub-agent's lifecycle events into the parent's
    /// sink. Without this call the resulting agent uses
    /// [`crate::agent::DroppingSink`] (the AgentBuilder default) and
    /// the parent loses visibility into child runs.
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<dyn AgentEventSink<ReActState>>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Use the parent's [`Approver`] for any tool dispatch the
    /// sub-agent issues — supervised execution propagates from
    /// parent to child unless the operator explicitly overrides.
    #[must_use]
    pub fn with_approver(mut self, approver: Arc<dyn Approver>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Attach an explicitly-named subset of the parent's skill
    /// registry. Validation runs at [`Self::build`] time — if any
    /// `allowed` name is absent from `parent_skills`, build returns
    /// [`entelix_core::Error::Config`].
    #[must_use]
    pub fn with_skills(
        mut self,
        parent_skills: &'a SkillRegistry,
        allowed: &[&str],
    ) -> Self {
        let owned: Vec<String> = allowed.iter().map(|s| (*s).to_owned()).collect();
        self.skills_request = Some((parent_skills, owned));
        self
    }

    /// Finalise the sub-agent. Validates the selection and skills
    /// requests against the parent registries, returning the first
    /// configuration error encountered.
    pub fn build(self) -> Result<Subagent<M>> {
        let Self {
            model,
            parent_registry,
            name,
            description,
            selection,
            skills_request,
            sink,
            approver,
        } = self;
        let tool_registry = selection.apply(parent_registry)?;
        let skills = match skills_request {
            None => SkillRegistry::new(),
            Some((parent_skills, allowed)) => {
                let missing: Vec<&str> = allowed
                    .iter()
                    .map(String::as_str)
                    .filter(|name| !parent_skills.has(name))
                    .collect();
                if !missing.is_empty() {
                    return Err(entelix_core::Error::config(format!(
                        "SubagentBuilder::with_skills: skill name(s) not in parent registry: {}",
                        missing.join(", ")
                    )));
                }
                let allowed_refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
                parent_skills.filter(&allowed_refs)
            }
        };
        Ok(Subagent {
            name,
            description,
            model: Arc::new(model),
            tool_registry,
            skills,
            sink,
            approver,
        })
    }
}

/// Wrapper exposing a [`crate::agent::Agent`] as a [`Tool`]. Built
/// via [`Subagent::into_tool`].
///
/// The dispatch contract: caller passes `{"task": "..."}`, the
/// wrapper builds a fresh `ReActState` seeded with one
/// `Role::User` message, runs the inner agent, and returns the
/// final assistant text under `{"output": "..."}`. The full message
/// trail is reachable via the agent's event sink — observability
/// stays on the audit channel rather than the LLM-facing payload
/// (ADR-0024 §7 lean-output rule).
pub struct SubagentTool {
    inner: crate::agent::Agent<ReActState>,
    metadata: ToolMetadata,
}

impl SubagentTool {
    fn new(inner: crate::agent::Agent<ReActState>, name: String, description: String) -> Self {
        let metadata = ToolMetadata::function(
            name,
            description,
            json!({
                "type": "object",
                "required": ["task"],
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Concrete task for the sub-agent. \
                                         Phrased as you would phrase a user \
                                         message to a fresh assistant."
                    }
                },
                "additionalProperties": false
            }),
        )
        .with_effect(ToolEffect::Mutating);
        Self { inner, metadata }
    }

    /// Override the conservative [`ToolEffect::Mutating`] default
    /// when the operator knows the sub-agent only reads.
    #[must_use]
    pub fn with_effect(mut self, effect: ToolEffect) -> Self {
        self.metadata = self.metadata.with_effect(effect);
        self
    }
}

impl std::fmt::Debug for SubagentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `inner` Agent is generic over its model and not bounded
        // on Debug — surfacing it would force every model wrapper
        // to implement Debug. Print the tool-side metadata only;
        // the inner agent's identity surfaces through events.
        f.debug_struct("SubagentTool")
            .field("name", &self.metadata.name)
            .field("effect", &self.metadata.effect)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let task = input.get("task").and_then(Value::as_str).ok_or_else(|| {
            Error::invalid_request("SubagentTool: input must include a string 'task' field")
        })?;
        // Sub-agent dispatch crosses an audit boundary (invariant #18):
        // a fresh `thread_id` scopes the child run's persistence and is
        // emitted on the parent's `AuditSink` so a replay can identify
        // which child thread the parent handed off to. UUID v7 keeps the
        // identifier time-ordered, matching `CheckpointId`'s shape.
        let sub_thread_id = uuid::Uuid::now_v7().to_string();
        if let Some(handle) = ctx.audit_sink() {
            handle
                .as_sink()
                .record_sub_agent_invoked(self.metadata.name.as_str(), &sub_thread_id);
        }
        let child_ctx = ctx.core().clone().with_thread_id(sub_thread_id);
        let initial = ReActState::from_user(task);
        let final_state = self.inner.invoke(initial, &child_ctx).await?;
        // Surface only the terminal assistant text — see ADR-0024 §7
        // for the lean-output rationale. The full transcript is
        // available to the parent's event sink via the agent's
        // lifecycle events.
        let output_text = final_state
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::Assistant))
            .and_then(|m| {
                m.content.iter().find_map(|p| match p {
                    ContentPart::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_default();
        Ok(json!({ "output": output_text }))
    }
}

/// Trivial Runnable wrapper around an `Arc<R>` so the sub-agent can
/// hand its model to [`create_react_agent`] without consuming the Arc.
struct ArcRunnable<R> {
    inner: Arc<R>,
}

impl<R> ArcRunnable<R> {
    const fn new(inner: Arc<R>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl<R> Runnable<Vec<Message>, Message> for ArcRunnable<R>
where
    R: Runnable<Vec<Message>, Message>,
{
    async fn invoke(&self, input: Vec<Message>, ctx: &ExecutionContext) -> Result<Message> {
        self.inner.invoke(input, ctx).await
    }
}
