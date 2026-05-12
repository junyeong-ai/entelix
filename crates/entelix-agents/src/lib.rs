//! # entelix-agents
//!
//! Production agent SDK on top of the entelix `Runnable` /
//! `StateGraph` foundation. Two layers:
//!
//! - **[`Agent<S>`]** — runtime entity wrapping any `Runnable<S, S>`
//!   with an event sink and (subsequent slices) execution mode +
//!   lifecycle observers. The production-facing surface every
//!   consumer constructs.
//! - **Recipes** — [`create_react_agent`], [`create_supervisor_agent`],
//!   [`create_chat_agent`]. Each returns a ready-to-stream
//!   `Agent<StateType>` so common patterns are a single call.
//!   Nested-supervisor topologies wire a [`team_from_supervisor`]
//!   adapter into a parent [`create_supervisor_agent`].
//!
//! Sub-agent permissions follow F7 (default-filtered hand) via
//! [`Subagent`].

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-agents/0.5.0")]
#![deny(missing_docs)]
// Doc-prose lints fire on legitimate proper nouns (ReAct,
// AgentBuilder, LangGraph) and on long opening paragraphs that
// explain recipe intent. `pub(crate)` items inside a `pub(crate)`
// module are visibility-equivalent to plain `pub` per clippy, but
// the explicit form documents intent.
#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::redundant_pub_crate,
    clippy::too_long_first_doc_paragraph
)]

pub(crate) mod agent;
mod chat_agent;
mod compaction;
mod react_agent;
mod state;
mod subagent;
mod summarizer;
mod supervisor;

pub use agent::{
    Agent, AgentBuilder, AgentEvent, AgentEventSink, AgentObserver, AgentRunResult, AlwaysApprove,
    ApprovalDecision, ApprovalLayer, ApprovalRequest, ApprovalService, Approver, BroadcastSink,
    CaptureSink, ChannelApprover, ChannelApproverConfig, ChannelSink, DroppingSink, DynObserver,
    EffectGate, ExecutionMode, FailOpenSink, FanOutSink, PendingApproval, ToolApprovalEventSink,
    ToolApprovalEventSinkHandle, ToolEventLayer, ToolEventService, ToolHook, ToolHookDecision,
    ToolHookLayer, ToolHookRegistry, ToolHookRequest, ToolHookService,
};
pub use chat_agent::{build_chat_graph, create_chat_agent};
pub use compaction::{
    DEFAULT_SUMMARY_KEEP_RECENT_TURNS, DEFAULT_SUMMARY_SYSTEM_PROMPT, MessageRunnableCompactionExt,
    RunnableCompacting, SummaryCompactor,
};
pub use react_agent::{ReActAgentBuilder, build_react_graph, create_react_agent};
pub use state::{ChatState, ReActState, SupervisorState};
pub use subagent::{Subagent, SubagentBuilder, SubagentMetadata, SubagentTool};
pub use summarizer::RunnableToSummarizerAdapter;
pub use supervisor::{
    AgentEntry, SupervisorDecision, build_supervisor_graph, create_supervisor_agent,
    team_from_supervisor,
};
