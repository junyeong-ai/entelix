//! The Hand surface — `Tool` trait + `ToolRegistry` (layered
//! `tower::Service` dispatch path) + `ToolDispatchScope` /
//! `ScopedToolLayer` (operator hook for ambient request-scope
//! state, e.g. tokio task-locals seeding RLS settings).
//!
//! See ADR-0011 for the rationale that `Tool` does not extend
//! `Runnable`. Tool dispatch composes through `tower::Layer<S>` —
//! `PolicyLayer`, `OtelLayer`, retry middleware all hang off the
//! `Service<ToolInvocation>` shape, never via a back-channel hook.
//! See ADR-0068 for `ToolDispatchScope` — the operator-supplied
//! future-wrapper that lets tools observe task-local ambient state
//! the SDK cannot supply through `ExecutionContext` directly.

mod effect;
mod metadata;
mod progress;
mod registry;
mod scope;
mod tool;
mod toolset;

pub use effect::{RetryHint, ToolEffect};
pub use metadata::ToolMetadata;
pub use progress::{
    CurrentToolInvocation, ToolProgress, ToolProgressSink, ToolProgressSinkHandle,
    ToolProgressStatus,
};
pub use registry::ToolRegistry;
pub use scope::{ScopedToolLayer, ScopedToolService, ToolDispatchScope};
pub use tool::Tool;
pub use toolset::Toolset;
