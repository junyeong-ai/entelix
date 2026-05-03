//! `create_hierarchical_agent` — a supervisor of supervisors.
//!
//! Each "team" is itself a `Runnable<Vec<Message>, Message>`
//! (typically built with [`crate::create_supervisor_agent`] and then
//! adapted with the `chat_message_view` helper below). The top-level
//! router decides which team handles the next turn.

use std::sync::Arc;

use entelix_core::ir::Message;
use entelix_core::{ExecutionContext, Result};
use entelix_runnable::{Runnable, RunnableLambda};

use crate::agent::Agent;
use crate::state::SupervisorState;
use crate::supervisor::{AgentEntry, SupervisorDecision, create_supervisor_agent};

/// Build a hierarchical (nested-supervisor) agent.
///
/// Each `team` is a `Runnable<Vec<Message>, Message>` — typically
/// produced by adapting another supervisor agent via
/// [`team_from_supervisor`]. The top-level `router` is the same shape
/// as in [`create_supervisor_agent`] — `Runnable<Vec<Message>,
/// SupervisorDecision>`.
pub fn create_hierarchical_agent<R>(
    router: R,
    teams: Vec<AgentEntry>,
) -> Result<Agent<SupervisorState>>
where
    R: Runnable<Vec<Message>, SupervisorDecision> + 'static,
{
    create_supervisor_agent(router, teams)
}

/// Adapt an [`Agent<SupervisorState>`] (a team supervisor) into a
/// `Runnable<Vec<Message>, Message>` so it can be embedded in a parent
/// hierarchical agent as one team.
///
/// The adapter feeds the parent's conversation into the team's own
/// state, runs the team to completion, and returns the last message
/// the team produced — typically the final assistant reply.
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
            let final_state = team.execute(state, &ctx).await?;
            final_state.messages.last().cloned().ok_or_else(|| {
                entelix_core::Error::invalid_request(
                    "team_from_supervisor: team finished with empty conversation",
                )
            })
        }
    })
}
