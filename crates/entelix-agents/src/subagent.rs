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
use entelix_core::{Error, ExecutionContext, Result, SkillRegistry, ToolRegistry};
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
    model: Arc<M>,
    tool_registry: ToolRegistry,
    skills: SkillRegistry,
    sink: Option<Arc<dyn AgentEventSink<ReActState>>>,
    approver: Option<Arc<dyn Approver>>,
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
    /// Build a sub-agent restricted to tools whose name is in
    /// `allowed_tools`. The skill registry starts empty; attach a
    /// filtered subset via [`Self::with_skills`] when the parent
    /// exposes skills.
    ///
    /// `parent_registry` carries the parent's layer stack — the
    /// narrowed view this sub-agent dispatches through inherits
    /// `PolicyLayer` / `OtelLayer` / retry middleware verbatim
    /// (`Arc`-shared, no copy).
    ///
    /// Returns [`entelix_core::Error::Config`] when any name in
    /// `allowed_tools` is absent from `parent_registry` — silently
    /// dropping a missing name has caused production
    /// misconfigurations (a sub-agent that quietly cannot reach a
    /// tool it was supposed to call), so the constructor fails fast
    /// at the moment the typo is introduced rather than at the moment
    /// the model issues the call.
    pub fn from_whitelist(
        model: M,
        parent_registry: &ToolRegistry,
        allowed_tools: &[&str],
    ) -> Result<Self> {
        let tool_registry = parent_registry.restricted_to(allowed_tools)?;
        Ok(Self {
            model: Arc::new(model),
            tool_registry,
            skills: SkillRegistry::new(),
            sink: None,
            approver: None,
        })
    }

    /// Build a sub-agent using a custom predicate over the parent's
    /// tools. The predicate is run once at construction; the resulting
    /// sub-agent has a frozen tool view that still inherits the
    /// parent's layer stack.
    ///
    /// Unlike [`Self::from_whitelist`], the filter form cannot detect
    /// "intended but missing" — a closure that matches nothing
    /// produces an empty sub-agent. That is the deliberate trade-off
    /// for accepting an arbitrary predicate: the strict-name path is
    /// preferred when the operator knows the tool names ahead of time.
    pub fn from_filter<F>(model: M, parent_registry: &ToolRegistry, predicate: F) -> Self
    where
        F: Fn(&dyn Tool) -> bool,
    {
        Self {
            model: Arc::new(model),
            tool_registry: parent_registry.filter(predicate),
            skills: SkillRegistry::new(),
            sink: None,
            approver: None,
        }
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
    /// registry. Mirrors [`Self::from_whitelist`] in shape and in
    /// failure mode: returns [`entelix_core::Error::Config`] when
    /// any `allowed` name is not present in `parent_skills`, so a
    /// typo surfaces at construction rather than as a silent
    /// "skill not found" at runtime.
    pub fn with_skills(mut self, parent_skills: &SkillRegistry, allowed: &[&str]) -> Result<Self> {
        let missing: Vec<&str> = allowed
            .iter()
            .copied()
            .filter(|name| !parent_skills.has(name))
            .collect();
        if !missing.is_empty() {
            return Err(entelix_core::Error::config(format!(
                "Subagent::with_skills: skill name(s) not in parent registry: {}",
                missing.join(", ")
            )));
        }
        self.skills = parent_skills.filter(allowed);
        Ok(self)
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
    /// Empty when [`Self::with_skills`] was never called.
    #[must_use]
    pub const fn skills(&self) -> &SkillRegistry {
        &self.skills
    }

    /// Build a `ReAct` loop bound to this sub-agent's narrowed tool
    /// registry (which inherits the parent's layer stack), model, and
    /// — when [`Self::with_skills`] was called — the three
    /// LLM-facing skill tools (`list_skills`, `activate_skill`,
    /// `read_skill_resource`) backed by the inherited skill registry.
    /// See ADR-0027 §"Auto-wire" and ADR-0035
    /// §"Sub-agent layer-stack inheritance". Sub-agents with no
    /// `with_skills` call build the registry without skill tools,
    /// matching their declared authority.
    pub fn into_react_agent(self) -> Result<crate::agent::Agent<ReActState>> {
        let Self {
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
    pub fn into_tool(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<SubagentTool> {
        let agent = self.into_react_agent()?;
        Ok(SubagentTool::new(agent, name.into(), description.into()))
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

    async fn execute(&self, input: Value, ctx: &ExecutionContext) -> Result<Value> {
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
        let child_ctx = ctx.clone().with_thread_id(sub_thread_id);
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
