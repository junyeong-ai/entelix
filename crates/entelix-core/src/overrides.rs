//! `RunOverrides` — per-call knobs that operators stash on
//! [`crate::context::ExecutionContext`] to override `ChatModel`
//! config and graph recursion limits without rebuilding either
//! component.
//!
//! ## Why per-call
//!
//! `ChatModel` and `CompiledGraph` are built once at agent
//! construction time and reused across many requests. A few
//! parameters benefit from per-call override — picking a different
//! model for a cheap classification step inside the same agent,
//! injecting a request-specific system prompt, or clamping the
//! recursion limit for an experimental run.
//!
//! `RunOverrides` flows through
//! [`crate::context::ExecutionContext::extension`] so the operator's
//! choice is one method call away on the call site and the layered
//! `tower::Service` stack picks it up automatically. See ADR-0069.
//!
//! ## Wiring
//!
//! Operator (explicit ctx mutation):
//!
//! ```ignore
//! let ctx = ExecutionContext::new()
//!     .add_extension(RunOverrides::new()
//!         .with_model("claude-3-5-haiku-latest")
//!         .with_max_iterations(8));
//! agent.execute(input, &ctx).await?;
//! ```
//!
//! Operator (convenience — sub-crate-specific entry points like
//! `entelix_agents::Agent::execute_with` set the extension for
//! you):
//!
//! ```ignore
//! agent.execute_with(input, RunOverrides::new()
//!     .with_system_prompt(SystemPrompt::text("You are a triage classifier.")),
//!     &ctx).await?;
//! ```
//!
//! Reading sites (internal — not operator code):
//!
//! - `ChatModel::complete_full` consults `ctx.extension::<RunOverrides>()`
//!   and patches `ModelRequest::model` and `ModelRequest::system`
//!   when the override is present.
//! - `CompiledGraph::invoke` clamps its compile-time `recursion_limit`
//!   to `min(recursion_limit, overrides.max_iterations)` when the
//!   override is present (compile-time cap stays authoritative —
//!   operators can lower but never raise).

use crate::ir::SystemPrompt;

/// Per-call knobs operators stash on `ExecutionContext` to override
/// `ChatModel` and `CompiledGraph` defaults without rebuilding either.
/// All fields are optional — `None` means "use the configured default".
///
/// `#[non_exhaustive]` so post-1.0 additions (per-call temperature,
/// stop sequences, response_format, …) ship as MINOR. Construct via
/// [`RunOverrides::new`] (or `default()`) and chain `with_*`
/// setters.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct RunOverrides {
    model: Option<String>,
    system_prompt: Option<SystemPrompt>,
    max_iterations: Option<usize>,
}

impl RunOverrides {
    /// Empty overrides — every field `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the model identifier on every `ChatModel` call that
    /// observes this `RunOverrides` through its `ExecutionContext`.
    /// Useful for cheap-model classification routes inside an agent
    /// otherwise pinned to an expensive model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Replace the system prompt on every `ChatModel` call observing
    /// this `RunOverrides`. The override completely supersedes the
    /// `ChatModel`'s configured `SystemPrompt` for the duration of
    /// the call (no merging — operators that want to extend rather
    /// than replace pre-compose the desired `SystemPrompt`
    /// themselves).
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: SystemPrompt) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    /// Cap the graph's recursion limit for this call. The
    /// compile-time `recursion_limit` (set on `StateGraph::compile`)
    /// stays authoritative — operators can only **lower** the
    /// effective cap, never raise it. The dispatch loop applies
    /// `min(compile_time_cap, overrides.max_iterations)`.
    #[must_use]
    pub const fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Borrow the model override if set.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Borrow the system-prompt override if set.
    #[must_use]
    pub const fn system_prompt(&self) -> Option<&SystemPrompt> {
        self.system_prompt.as_ref()
    }

    /// The max-iterations override if set.
    #[must_use]
    pub const fn max_iterations(&self) -> Option<usize> {
        self.max_iterations
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_overrides_have_no_fields_set() {
        let o = RunOverrides::new();
        assert!(o.model().is_none());
        assert!(o.system_prompt().is_none());
        assert!(o.max_iterations().is_none());
    }

    #[test]
    fn with_setters_chain() {
        let o = RunOverrides::new()
            .with_model("haiku")
            .with_system_prompt(SystemPrompt::text("be brief"))
            .with_max_iterations(8);
        assert_eq!(o.model(), Some("haiku"));
        assert!(o.system_prompt().is_some());
        assert_eq!(o.max_iterations(), Some(8));
    }
}
