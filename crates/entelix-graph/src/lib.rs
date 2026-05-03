//! # entelix-graph
//!
//! Control-flow contract for entelix (invariant 8). Houses
//! [`StateGraph<S>`] with typed state, the [`Reducer<T>`] trait,
//! conditional / fan-out edges, subgraphs, the [`Checkpointer`]
//! trait plus [`InMemoryCheckpointer`], and the [`interrupt()`] /
//! [`Command<S>`] HITL API. `recursion_limit` (F6 mitigation) is
//! enforced here.
//!
//! Surface: [`StateGraph<S>`] builder with static and conditional
//! edges + [`CompiledGraph<S>`] executor with `recursion_limit`
//! enforcement and the [`END`] sentinel target. [`Checkpointer`] +
//! [`InMemoryCheckpointer`] for write-after-each-node persistence;
//! [`CompiledGraph::resume`] / [`CompiledGraph::resume_with`] crash
//! recovery. [`interrupt()`] + [`Command<S>`] for HITL
//! pause-and-continue.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-graph/1.0.0-rc.1")]
#![deny(missing_docs)]
// Doc prose for Reducer/Dispatch references LangGraph by name and
// uses long opening paragraphs that explain the parity intent;
// pedantic doc-style lints fire on both. We accept the trade-off.
#![allow(clippy::doc_markdown, clippy::too_long_first_doc_paragraph)]

mod checkpoint;
mod command;
mod compiled;
mod contributing_node;
mod dispatch;
mod finalizing_stream;
mod in_memory_checkpointer;
mod interrupt;
mod merge_node;
mod reducer;
mod state_graph;

pub use checkpoint::{Checkpoint, CheckpointId, Checkpointer};
pub use command::Command;
pub use compiled::{
    CompiledGraph, ConditionalEdge, EdgeSelector, SendEdge, SendMerger, SendSelector,
};
pub use contributing_node::ContributingNodeAdapter;
pub use dispatch::{Dispatch, scatter};
pub use finalizing_stream::FinalizingStream;
pub use in_memory_checkpointer::InMemoryCheckpointer;
pub use interrupt::interrupt;
pub use merge_node::MergeNodeAdapter;
pub use reducer::{Annotated, Append, Max, MergeMap, Reducer, Replace, StateMerge};
pub use state_graph::{CheckpointGranularity, DEFAULT_RECURSION_LIMIT, END, StateGraph};

// Companion `#[derive(StateMerge)]` proc-macro. Lives in the macro
// namespace, so the same-name re-export coexists with the trait
// above (serde's `Serialize` / `Deserialize` follow the same shape).
pub use entelix_graph_derive::StateMerge;
