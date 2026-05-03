//! Cancellation primitive for entelix.
//!
//! All async APIs that may run > 100 ms accept cancellation via
//! `tokio_util::sync::CancellationToken` (CLAUDE.md §"Cancellation",
//! F3 mitigation). The token is carried by [`ExecutionContext`] and inherited
//! by sub-agents and tool calls.
//!
//! Re-exporting under `entelix_core::cancellation` gives downstream crates a
//! single import path that documents the contract — they should not pull
//! `tokio_util` directly.
//!
//! [`ExecutionContext`]: crate::context::ExecutionContext

pub use tokio_util::sync::CancellationToken;
