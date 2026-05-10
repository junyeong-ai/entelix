//! The Hand surface — `Tool` trait + `ToolRegistry` (layered
//! `tower::Service` dispatch path) + `ToolDispatchScope` /
//! `ScopedToolLayer` (operator hook for ambient request-scope
//! state, e.g. tokio task-locals seeding RLS settings).
//!
//! See for the rationale that `Tool` does not extend
//! `Runnable`. Tool dispatch composes through `tower::Layer<S>` —
//! `PolicyLayer`, `OtelLayer`, retry middleware all hang off the
//! `Service<ToolInvocation>` shape, never via a back-channel hook.
//! See for `ToolDispatchScope` — the operator-supplied
//! future-wrapper that lets tools observe task-local ambient state
//! the SDK cannot supply through `ExecutionContext` directly.

mod cache;
mod effect;
mod error_kind;
mod metadata;
mod progress;
mod registry;
mod retry_layer;
mod scope;
mod tool;
mod toolset;

pub use cache::ToolCacheMode;
pub use effect::{RetryHint, ToolEffect};
pub use error_kind::ToolErrorKind;
pub use metadata::ToolMetadata;
pub use progress::{
    CurrentToolInvocation, ToolProgress, ToolProgressSink, ToolProgressSinkHandle,
    ToolProgressStatus,
};
pub use registry::ToolRegistry;
pub use retry_layer::{DEFAULT_MAX_BACKOFF, RetryToolLayer, RetryToolService};
pub use scope::{ScopedToolLayer, ScopedToolService, ToolDispatchScope};
pub use tool::Tool;
pub use toolset::Toolset;
