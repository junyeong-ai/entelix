//! # entelix-session
//!
//! Session-as-event-SSoT (invariant 1). Defines [`SessionGraph`] with
//! `events: Vec<GraphEvent>` as the only durable audit truth, plus
//! fork semantics and archival watermarks. The persistent companion
//! is the [`SessionLog`] trait — concrete impls live in
//! `entelix-persistence`.
//!
//! Per F1 mitigation, this module is **distinct from** the runtime
//! [`entelix_core::events::EventBus`]: the bus is fire-and-forget
//! ephemeral pub-sub, this module is the durable replay source.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-session/0.4.2")]
#![deny(missing_docs)]

mod audit_sink;
mod compaction;
mod event;
mod log;
mod session_graph;

pub use audit_sink::SessionAuditSink;
pub use compaction::{
    CompactedHistory, Compactor, HeadDropCompactor, ToolPair, Turn, messages_char_size,
    messages_to_events,
};
pub use event::GraphEvent;
pub use log::{InMemorySessionLog, SessionLog};
pub use session_graph::SessionGraph;
