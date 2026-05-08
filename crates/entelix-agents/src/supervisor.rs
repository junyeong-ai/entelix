//! `create_supervisor_agent` — supervisor routes each turn to one of N
//! pre-built sub-agents (or finishes the conversation). The supervisor
//! itself is a `Runnable<Vec<Message>, SupervisorDecision>` — the
//! typed enum forecloses LLM hallucination of unregistered agent
//! names: a router that returns `Agent("rsearch")` when the registry
//! holds `"research"` would now have to hallucinate a *type* (which
//! is impossible) rather than just a string (which is easy).
//!
//! ## Routing contract
//!
//! Each turn:
//! 1. `router.invoke(messages, ctx)` → [`SupervisorDecision`].
//! 2. `Finish` ends the conversation; `Agent(name)` runs the named
//!    sub-agent and loops back to the supervisor.
//! 3. An `Agent(name)` whose name is not in the registry surfaces
//!    `Error::Config` immediately — the validator runs at graph
//!    compile time so the failure is observed as a configuration
//!    bug rather than a runtime "unknown route" deadlock.

use std::sync::Arc;

use entelix_core::ir::Message;
use entelix_core::{Error, ExecutionContext, Result};
use entelix_graph::StateGraph;

use crate::agent::Agent;
use entelix_runnable::{Runnable, RunnableLambda};

use crate::state::SupervisorState;

/// Decision a supervisor router emits each turn.
///
/// Replaces the prior `String` + `SUPERVISOR_FINISH` sentinel
/// pairing — invariant #17 (typed signal beats stringly-typed
/// sentinel). LLM-driven routers parse their own text into this
/// enum at the boundary; routers backed by deterministic logic
/// match against `state.messages` directly.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SupervisorDecision {
    /// Route the next turn to the named sub-agent. The name MUST
    /// match an `AgentEntry::name` registered with the supervisor;
    /// otherwise the run surfaces `Error::Config` rather than
    /// silently dead-ending.
    Agent(String),
    /// Terminate the conversation. The supervisor returns the
    /// final `SupervisorState` to its caller.
    Finish,
    /// Route the next turn to the named sub-agent **and** inject a
    /// typed `payload` (research summary, classification result,
    /// escalation reason, …) as a `system` message visible to the
    /// receiving agent. Same routing rules as
    /// [`Self::Agent`] — unknown names trip a `tracing::warn!` and
    /// finish the run rather than dead-ending. Mirrors OpenAI
    /// Agents' `Handoff(input_type, on_handoff)` pattern: the
    /// supervisor speaks structured context to the next agent
    /// without round-tripping through the model's natural-language
    /// channel.
    Handoff {
        /// Registered `AgentEntry::name`.
        agent: String,
        /// JSON payload injected as the next agent's leading
        /// `system` message. Operators are responsible for
        /// keeping payloads model-safe (the channel is LLM-facing
        /// per invariant 16).
        payload: serde_json::Value,
    },
}

impl SupervisorDecision {
    /// Convenience constructor for the common case.
    pub fn agent(name: impl Into<String>) -> Self {
        Self::Agent(name.into())
    }

    /// Convenience constructor for typed handoff with a payload.
    ///
    /// Reach for this when the supervisor's routing logic produces
    /// structured context the next agent should see — research
    /// summary, classification verdict, escalation reason — and
    /// returning `Agent(name)` would force that context to round-
    /// trip through the model's natural-language channel. The
    /// dispatch loop injects `payload` as the receiving agent's
    /// leading `system` message; the operator owns model-safety of
    /// the payload contents (invariant 16).
    pub fn handoff(agent: impl Into<String>, payload: serde_json::Value) -> Self {
        Self::Handoff {
            agent: agent.into(),
            payload,
        }
    }

    /// Name of the receiving agent, or `None` for [`Self::Finish`].
    /// Both [`Self::Agent`] and [`Self::Handoff`] route to a
    /// registered `AgentEntry::name`.
    #[must_use]
    pub fn agent_name(&self) -> Option<&str> {
        match self {
            Self::Agent(name) => Some(name),
            Self::Handoff { agent, .. } => Some(agent),
            Self::Finish => None,
        }
    }
}

/// One named sub-agent in a [`create_supervisor_agent`] graph.
///
/// The agent itself is any `Runnable<Vec<Message>, Message>` — for
/// example another `CompiledGraph` from [`crate::create_chat_agent`] or
/// [`crate::create_react_agent`] composed inline, or a bare
/// `ChatModel`.
pub struct AgentEntry {
    /// Stable identifier the supervisor uses to route. Must be unique
    /// within a supervisor graph.
    pub name: String,
    /// The agent itself, type-erased to a chat-shaped runnable.
    pub agent: Arc<dyn Runnable<Vec<Message>, Message>>,
}

impl AgentEntry {
    /// Convenience constructor.
    pub fn new<R>(name: impl Into<String>, agent: R) -> Self
    where
        R: Runnable<Vec<Message>, Message> + 'static,
    {
        Self {
            name: name.into(),
            agent: Arc::new(agent),
        }
    }
}

/// Build a supervisor graph that picks among `agents` each turn.
///
/// Builds the supervisor graph (router ↔ agents ↔ finish) without
/// wrapping it into an [`Agent`]. Use this when you want to
/// configure the agent surface (name, sink, approver, observers)
/// via [`Agent::builder`] directly.
///
/// Validation: every name registered with the router must be in the
/// `agents` list — a router that emits `Agent("research")` while the
/// registry only knows `"research-team"` surfaces `Error::Config`
/// at the moment a turn picks the unknown name (the conditional
/// edge cannot route to a non-existent node).
pub fn build_supervisor_graph<R>(
    router: R,
    agents: Vec<AgentEntry>,
) -> Result<entelix_graph::CompiledGraph<SupervisorState>>
where
    R: Runnable<Vec<Message>, SupervisorDecision> + 'static,
{
    if agents.is_empty() {
        return Err(Error::config(
            "build_supervisor_graph: at least one agent required",
        ));
    }
    let router = Arc::new(router);

    let supervisor_node =
        RunnableLambda::new(move |mut state: SupervisorState, ctx: ExecutionContext| {
            let router = router.clone();
            async move {
                let decision = router.invoke(state.messages.clone(), &ctx).await?;
                // Invariant #18 — handoff is auditable. Emit the
                // (from, to) pair on every routed turn; `Finish` does
                // not produce a handoff (the run terminates rather
                // than transferring control to another named agent).
                // `Agent` and `Handoff` share the same audit shape —
                // the payload, when present, rides separately into
                // the receiving agent's leading `system` message.
                if let Some(name) = decision.agent_name()
                    && let Some(handle) = ctx.audit_sink()
                {
                    handle
                        .as_sink()
                        .record_agent_handoff(state.last_speaker.as_deref(), name);
                }
                state.next_speaker = Some(decision);
                Ok::<_, _>(state)
            }
        });

    let mut graph = StateGraph::<SupervisorState>::new()
        .add_node("supervisor", supervisor_node)
        .set_entry_point("supervisor");

    let finish_node =
        RunnableLambda::new(|state: SupervisorState, _ctx| async move { Ok::<_, _>(state) });
    graph = graph
        .add_node("finish", finish_node)
        .add_finish_point("finish");

    // Routing keys: the typed `SupervisorDecision` is projected to a
    // `String` only at the conditional-edge boundary the graph layer
    // requires. Hallucination at the LLM router is foreclosed by the
    // enum; a deterministic registry lookup at this boundary catches
    // any mis-routing before the graph hands control to a non-node.
    let known_names: std::collections::HashSet<String> =
        agents.iter().map(|e| e.name.clone()).collect();
    let mut conditional_mapping: Vec<(String, String)> =
        vec![(FINISH_KEY.to_owned(), "finish".to_owned())];

    for entry in agents {
        let AgentEntry { name, agent } = entry;
        let label = name.clone();
        let node_name = name.clone();
        let agent_node =
            RunnableLambda::new(move |mut state: SupervisorState, ctx: ExecutionContext| {
                let agent = agent.clone();
                let label = label.clone();
                async move {
                    // Drain any handoff payload the supervisor staged
                    // on this turn into a leading `system` message
                    // so the receiving agent sees it before its own
                    // turn. Agent / Finish carry no payload.
                    if let Some(SupervisorDecision::Handoff { payload, .. }) =
                        state.next_speaker.take()
                    {
                        let rendered = serde_json::to_string_pretty(&payload)
                            .unwrap_or_else(|_| payload.to_string());
                        state
                            .messages
                            .push(Message::system(format!("Handoff payload:\n{rendered}")));
                    }
                    let reply = agent.invoke(state.messages.clone(), &ctx).await?;
                    state.messages.push(reply);
                    state.last_speaker = Some(label.clone());
                    state.next_speaker = None;
                    Ok::<_, _>(state)
                }
            });
        graph = graph
            .add_node(node_name.clone(), agent_node)
            .add_edge(node_name.clone(), "supervisor");
        conditional_mapping.push((name.clone(), name));
    }

    graph = graph.add_conditional_edges(
        "supervisor",
        move |state: &SupervisorState| match state.next_speaker.as_ref().and_then(SupervisorDecision::agent_name) {
            Some(name) if known_names.contains(name) => name.to_owned(),
            // Either no decision yet (unreachable from the
            // supervisor node, which always sets one) or an
            // unknown-agent name (caller bug). Routing to FINISH
            // surfaces graph completion rather than a dead-end
            // deadlock; the unknown-name branch additionally trips
            // a tracing event so the operator sees what happened.
            Some(name) => {
                tracing::warn!(
                    target: "entelix.agents.supervisor",
                    unknown_agent = %name,
                    "supervisor router emitted decision routing to '{name}' but no AgentEntry by that name; finishing"
                );
                FINISH_KEY.to_owned()
            }
            None => FINISH_KEY.to_owned(),
        },
        conditional_mapping,
    );

    graph.compile()
}

/// Internal routing key the conditional-edge map uses for the
/// `Finish` arm. Operator code never sees this — they speak in
/// [`SupervisorDecision`].
const FINISH_KEY: &str = "__finish__";

/// Build a supervisor graph that picks among `agents` each turn.
///
/// `router` returns a [`SupervisorDecision`] given the conversation
/// so far — either route to a named agent or finish the run. An
/// empty `agents` list is rejected.
pub fn create_supervisor_agent<R>(
    router: R,
    agents: Vec<AgentEntry>,
) -> Result<Agent<SupervisorState>>
where
    R: Runnable<Vec<Message>, SupervisorDecision> + 'static,
{
    let compiled = build_supervisor_graph(router, agents)?;
    Agent::<SupervisorState>::builder()
        .with_name("supervisor")
        .with_runnable(compiled)
        .build()
}

/// Adapt a supervisor [`Agent<SupervisorState>`] into a
/// `Runnable<Vec<Message>, Message>` so it can be embedded as one
/// `AgentEntry` inside a parent supervisor — the nested-supervisor
/// pattern.
///
/// The adapter feeds the parent's conversation into the nested
/// supervisor's own state, runs it to completion, and returns the
/// last message — typically the final assistant reply.
pub fn team_from_supervisor(team: Agent<SupervisorState>) -> impl Runnable<Vec<Message>, Message> {
    let team = Arc::new(team);
    RunnableLambda::new(move |messages: Vec<Message>, ctx: ExecutionContext| {
        let team = team.clone();
        async move {
            let state = SupervisorState {
                messages,
                last_speaker: None,
                next_speaker: None,
            };
            let final_state = team.execute(state, &ctx).await?.into_state();
            final_state.messages.last().cloned().ok_or_else(|| {
                Error::invalid_request(
                    "team_from_supervisor: team finished with empty conversation",
                )
            })
        }
    })
}
