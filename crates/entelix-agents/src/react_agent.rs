//! `create_react_agent` — `LangGraph` parity. Two nodes, one conditional
//! edge: the planner calls the model, the tool node dispatches any
//! `ToolUse` parts the model emitted, and the loop continues until the
//! model returns a tool-free reply.
//!
//! Tool dispatch routes through [`ToolRegistry::dispatch`], so any
//! `tower::Layer` attached to the registry (e.g. `PolicyLayer`,
//! `OtelLayer`) fires uniformly around every tool execution.

use std::sync::Arc;

use entelix_core::ir::{ContentPart, Message, Role, ToolResultContent};
use entelix_core::{Error, ExecutionContext, LlmRenderable, Result, ToolRegistry};
use entelix_graph::{CompiledGraph, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

use crate::agent::{Agent, AgentBuilder};
use crate::state::ReActState;

/// Builds the compiled ReAct graph (planner ↔ tools ↔ finish loop)
/// without wrapping it into an [`Agent`]. Use this when you want to
/// configure the agent-level surface (name, sink, approver,
/// observers) directly via [`Agent::builder`]:
///
/// ```ignore
/// let graph = build_react_graph(model, tools)?;
/// let agent = Agent::<ReActState>::builder()
///     .with_name("research")
///     .with_runnable(graph)
///     .with_sink(my_sink)
///     .with_approver(my_approver)
///     .build()?;
/// ```
///
/// Uses the graph-layer default recursion limit. Recipes that need
/// to override it use [`ReActAgentBuilder::with_recursion_limit`].
pub fn build_react_graph<M>(model: M, tools: ToolRegistry) -> Result<CompiledGraph<ReActState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    build_react_graph_inner(model, tools, None)
}

/// Variant of [`build_react_graph`] that overrides the
/// `StateGraph::with_recursion_limit` cap. Surfaced so recipes
/// (`ReActAgentBuilder::with_recursion_limit`) can plumb the
/// per-deployment value without duplicating the topology builder.
pub(crate) fn build_react_graph_with_recursion_limit<M>(
    model: M,
    tools: ToolRegistry,
    recursion_limit: usize,
) -> Result<CompiledGraph<ReActState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    build_react_graph_inner(model, tools, Some(recursion_limit))
}

fn build_react_graph_inner<M>(
    model: M,
    tools: ToolRegistry,
    recursion_limit: Option<usize>,
) -> Result<CompiledGraph<ReActState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let model = Arc::new(model);
    let tools = Arc::new(tools);

    let planner_node = RunnableLambda::new(move |mut state: ReActState, ctx: ExecutionContext| {
        let model = model.clone();
        async move {
            let reply = model.invoke(state.messages.clone(), &ctx).await?;
            state.messages.push(reply);
            state.steps = state.steps.saturating_add(1);
            Ok::<_, _>(state)
        }
    });

    let tool_node = RunnableLambda::new(move |mut state: ReActState, ctx: ExecutionContext| {
        let tools = tools.clone();
        async move {
            let last = state.messages.last().cloned().ok_or_else(|| {
                Error::invalid_request("ReActAgent: tool dispatch with empty conversation")
            })?;
            let mut results: Vec<ContentPart> = Vec::new();
            for part in &last.content {
                if let ContentPart::ToolUse { id, name, input } = part {
                    let (content, is_error) =
                        match tools.dispatch(id, name, input.clone(), &ctx).await {
                            Ok(value) => (ToolResultContent::Json(value), false),
                            // `Interrupted` is a graph control signal
                            // (HITL pause via `ApprovalLayer` AwaitExternal
                            // or a tool-internal `interrupt(...)`). Propagate
                            // it through the node rather than swallowing as
                            // a tool failure — the dispatch loop
                            // (`CompiledGraph::execute_loop_inner`)
                            // catches it, persists a checkpoint with PRE-
                            // node state, and surfaces the typed
                            // `Error::Interrupted` to the caller. Resume
                            // re-runs this same node with the operator's
                            // decision threaded through
                            // `Command::ApproveTool { tool_use_id, decision }`
                            // on `CompiledGraph::resume_with`.
                            Err(Error::Interrupted { kind, payload }) => {
                                return Err(Error::Interrupted { kind, payload });
                            }
                            // Invariant #16 — model sees the LLM-facing
                            // rendering (no vendor status, no source
                            // chain). The full error continues to flow
                            // through the event sink and OTel for the
                            // operator channel.
                            Err(e) => (ToolResultContent::Text(e.render_for_llm()), true),
                        };
                    results.push(ContentPart::ToolResult {
                        tool_use_id: id.clone(),
                        name: name.clone(),
                        content,
                        is_error,
                        cache_control: None,
                    });
                }
            }
            if results.is_empty() {
                return Err(Error::invalid_request(
                    "ReActAgent: tool node reached without any ToolUse parts",
                ));
            }
            state.messages.push(Message::new(Role::Tool, results));
            Ok::<_, _>(state)
        }
    });

    let finish_node =
        RunnableLambda::new(|state: ReActState, _ctx| async move { Ok::<_, _>(state) });

    let mut builder = StateGraph::<ReActState>::new()
        .add_node("planner", planner_node)
        .add_node("tools", tool_node)
        .add_node("finish", finish_node)
        .set_entry_point("planner")
        .add_finish_point("finish")
        .add_edge("tools", "planner")
        .add_conditional_edges(
            "planner",
            |state: &ReActState| {
                if last_message_has_tool_use(state) {
                    "tools".to_owned()
                } else {
                    "finish".to_owned()
                }
            },
            [("tools", "tools"), ("finish", "finish")],
        );
    if let Some(limit) = recursion_limit {
        builder = builder.with_recursion_limit(limit);
    }
    builder.compile()
}

/// Builds a ReAct-style agent: model decides; if it emits tool
/// calls, dispatch and loop back; otherwise finish.
///
/// Returns a production-ready [`Agent<ReActState>`] — call
/// `agent.execute(state, &ctx).await` for sync drain or
/// `agent.execute_stream(state, &ctx)` for a per-event stream
/// (constructed synchronously; poll with `.next().await`).
pub fn create_react_agent<M>(model: M, tools: ToolRegistry) -> Result<Agent<ReActState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let graph = build_react_graph(model, tools)?;
    Agent::<ReActState>::builder()
        .with_name("react")
        .with_runnable(graph)
        .build()
}

/// Internal: start an `AgentBuilder` for a ReAct agent. Used by
/// `Subagent::into_react_agent` to apply parent-supplied sink and
/// approver before finalising.
pub(crate) fn react_agent_builder<M>(
    model: M,
    tools: ToolRegistry,
) -> Result<AgentBuilder<ReActState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let graph = build_react_graph(model, tools)?;
    Ok(Agent::<ReActState>::builder()
        .with_name("react")
        .with_runnable(graph))
}

/// Variant of [`react_agent_builder`] that overrides the graph's
/// recursion limit. Used by [`ReActAgentBuilder::with_recursion_limit`].
pub(crate) fn react_agent_builder_with_recursion_limit<M>(
    model: M,
    tools: ToolRegistry,
    recursion_limit: usize,
) -> Result<AgentBuilder<ReActState>>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    let graph = build_react_graph_with_recursion_limit(model, tools, recursion_limit)?;
    Ok(Agent::<ReActState>::builder()
        .with_name("react")
        .with_runnable(graph))
}

/// Fluent builder for a ReAct-style [`Agent<ReActState>`].
///
/// Use this when you need to customize the per-deployment knobs
/// (name, event sink, approver for human-in-the-loop, observers) on
/// top of the standard planner ↔ tools loop. The graph topology
/// itself is fixed by the recipe; callers that need different
/// topology should compose [`entelix_graph::StateGraph`] directly.
///
/// ```ignore
/// let agent = ReActAgentBuilder::new(model, tools)
///     .with_name("research-agent")
///     .with_sink(my_broadcast_sink)
///     .with_approver(my_approver)
///     .build()?;
/// ```
pub struct ReActAgentBuilder<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    model: M,
    tools: ToolRegistry,
    name: Option<String>,
    sink: Option<Arc<dyn crate::agent::AgentEventSink<ReActState>>>,
    approver: Option<Arc<dyn crate::agent::Approver>>,
    execution_mode: Option<crate::agent::ExecutionMode>,
    observers: Vec<crate::agent::DynObserver<ReActState>>,
    recursion_limit: Option<usize>,
}

impl<M> ReActAgentBuilder<M>
where
    M: Runnable<Vec<Message>, Message> + 'static,
{
    /// Start a builder bound to `model` + `tools`.
    pub fn new(model: M, tools: ToolRegistry) -> Self {
        Self {
            model,
            tools,
            name: None,
            sink: None,
            approver: None,
            execution_mode: None,
            observers: Vec::new(),
            recursion_limit: None,
        }
    }

    /// Override the agent name (defaults to `"react"`).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Forward lifecycle events to `sink`.
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<dyn crate::agent::AgentEventSink<ReActState>>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Attach an approver. Sets [`crate::agent::ExecutionMode::Supervised`]
    /// unless an explicit mode is configured via
    /// [`Self::with_execution_mode`].
    #[must_use]
    pub fn with_approver(mut self, approver: Arc<dyn crate::agent::Approver>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Override the execution mode (defaults to
    /// [`crate::agent::ExecutionMode::Auto`], or `Supervised` when
    /// [`Self::with_approver`] was supplied).
    #[must_use]
    pub const fn with_execution_mode(mut self, mode: crate::agent::ExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
    }

    /// Register a lifecycle observer.
    #[must_use]
    pub fn with_observer(mut self, observer: crate::agent::DynObserver<ReActState>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Override the underlying graph's recursion limit (default 25
    /// per `entelix-graph`'s `DEFAULT_RECURSION_LIMIT`). Long-running
    /// research loops that intentionally cycle the planner past 25
    /// turns raise it; tightly-bounded routing agents lower it to
    /// catch run-away dispatch faster.
    #[must_use]
    pub const fn with_recursion_limit(mut self, n: usize) -> Self {
        self.recursion_limit = Some(n);
        self
    }

    /// Finalise into a runnable agent.
    pub fn build(self) -> Result<Agent<ReActState>> {
        // When an `Approver` is configured, wrap the tools registry
        // with `ApprovalLayer` so every dispatch through the recipe's
        // tool node passes through `Approver::decide` first. The
        // layer auto-emits `ToolCallApproved` / `ToolCallDenied`
        // through the sink handle that `Agent::execute` attaches to
        // the request context.
        let tools = match &self.approver {
            Some(approver) => self
                .tools
                .layer(crate::agent::ApprovalLayer::new(Arc::clone(approver))),
            None => self.tools,
        };
        let mut builder = match self.recursion_limit {
            Some(limit) => react_agent_builder_with_recursion_limit(self.model, tools, limit)?,
            None => react_agent_builder(self.model, tools)?,
        };
        if let Some(name) = self.name {
            builder = builder.with_name(name);
        }
        if let Some(sink) = self.sink {
            builder = builder.with_sink_arc(sink);
        }
        let mode = self.execution_mode.unwrap_or_else(|| {
            if self.approver.is_some() {
                crate::agent::ExecutionMode::Supervised
            } else {
                crate::agent::ExecutionMode::default()
            }
        });
        builder = builder.with_execution_mode(mode);
        if let Some(approver) = self.approver {
            builder = builder.with_approver_arc(approver);
        }
        for observer in self.observers {
            builder = builder.with_observer_arc(observer);
        }
        builder.build()
    }
}

fn last_message_has_tool_use(state: &ReActState) -> bool {
    state.messages.last().is_some_and(|m| {
        m.content
            .iter()
            .any(|p| matches!(p, ContentPart::ToolUse { .. }))
    })
}
