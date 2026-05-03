//! `interrupt()` helper — graph-side ergonomic shorthand for returning
//! `Error::Interrupted { payload }` from inside a node.
//!
//! Use from any `Runnable<S, S>` node body when the graph should pause and
//! hand control back to the caller for human review. The caller observes
//! `Err(Error::Interrupted { payload })`, optionally updates state, then
//! calls `CompiledGraph::resume_with` (or the lower-level `resume`) to
//! continue.

use entelix_core::{Error, Result};

/// Interrupt the graph from inside a node. Returns `Err` so the call site
/// can be `return interrupt(value);`.
pub const fn interrupt<T>(payload: serde_json::Value) -> Result<T> {
    Err(Error::Interrupted { payload })
}
