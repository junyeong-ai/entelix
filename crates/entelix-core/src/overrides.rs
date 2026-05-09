//! Per-call overrides — operators stash these on
//! [`crate::context::ExecutionContext`] to patch defaults without
//! rebuilding `ChatModel` or `CompiledGraph`. Two carriers, scoped to
//! orthogonal concerns:
//!
//! - [`RunOverrides`] — agent-loop knobs (`model`, `system_prompt`,
//!   `max_iterations`). Replace the model identifier on a per-call
//!   classification step inside an agent otherwise pinned to an
//!   expensive model, swap the system prompt for a triage variant,
//!   or clamp graph recursion for an experimental run.
//! - [`RequestOverrides`] — `ModelRequest`-shaped sampling knobs
//!   (`temperature`, `top_p`, `max_tokens`, `stop_sequences`,
//!   `reasoning_effort`, `tool_choice`, `response_format`). Vary the
//!   sampling profile on a single call without rebuilding the
//!   `ChatModel`'s configured defaults.
//!
//! ## Why two types
//!
//! `RunOverrides` patches things the *agent loop* owns — which model
//! to dispatch against, what system prompt to introduce, how many
//! reasoning rounds to permit. `RequestOverrides` patches the
//! `ModelRequest` itself before encoding. The split lets each type
//! grow its own knobs without becoming a god-struct, and makes the
//! call-site intent self-documenting.
//!
//! ## Wiring
//!
//! Both types flow through
//! [`crate::context::ExecutionContext::add_extension`]; the chat
//! model and graph dispatch loop pick them up automatically:
//!
//! ```ignore
//! let ctx = ExecutionContext::new()
//!     .add_extension(RunOverrides::new()
//!         .with_model("claude-3-5-haiku-latest")
//!         .with_max_iterations(8))
//!     .add_extension(RequestOverrides::new()
//!         .with_temperature(0.2)
//!         .with_max_tokens(512));
//! agent.execute(input, &ctx).await?;
//! ```
//!
//! Reading sites (internal — not operator code):
//!
//! - `ChatModel::complete_full` and `ChatModel::stream_deltas` route
//!   through `apply_overrides` which patches the outgoing
//!   `ModelRequest` from both extensions when present.
//! - `CompiledGraph::invoke` clamps its compile-time
//!   `recursion_limit` to `min(recursion_limit,
//!   run_overrides.max_iterations)` when the override is present
//!   (compile-time cap stays authoritative — operators can lower but
//!   never raise).

use crate::ir::{ReasoningEffort, ResponseFormat, SystemPrompt, ToolChoice};

/// Agent-loop overrides — model identifier, system prompt, recursion
/// cap. Patched onto the call's [`crate::context::ExecutionContext`]
/// via `add_extension`.
///
/// `#[non_exhaustive]` so additive loop-level knobs ship as MINOR.
/// Construct via [`RunOverrides::new`] (or `default()`) and chain
/// `with_*` setters.
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

/// `ModelRequest`-shaped sampling overrides — temperature, top-p,
/// max tokens, stop sequences, reasoning effort, tool choice,
/// response format. Patched onto the call's
/// [`crate::context::ExecutionContext`] via `add_extension`.
///
/// Each field is `Option`-wrapped: `None` means "use the
/// `ChatModelConfig` default for this field"; `Some` overrides for
/// the duration of the call. Stop sequences use `Option<Vec<String>>`
/// so an empty `Vec` is a meaningful override (clear the list) versus
/// `None` (keep the configured list).
///
/// `#[non_exhaustive]` so additive request-level knobs ship as MINOR.
/// Construct via [`RequestOverrides::new`] (or `default()`) and
/// chain `with_*` setters.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct RequestOverrides {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    reasoning_effort: Option<ReasoningEffort>,
    tool_choice: Option<ToolChoice>,
    response_format: Option<ResponseFormat>,
    parallel_tool_calls: Option<bool>,
}

impl RequestOverrides {
    /// Empty overrides — every field `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override sampling temperature for this call. Codecs clamp to
    /// vendor range.
    #[must_use]
    pub const fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Override nucleus-sampling parameter (`top_p`) for this call.
    #[must_use]
    pub const fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Override top-k sampling parameter for this call. Native on
    /// Anthropic, Gemini, Bedrock-Anthropic; OpenAI codecs surface
    /// as `LossyEncode`.
    #[must_use]
    pub const fn with_top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Override hard output-token cap for this call.
    #[must_use]
    pub const fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Replace the stop-sequence list for this call. Pass an empty
    /// `Vec` to clear the configured list; pass `None` (the default,
    /// implicit from not calling this) to keep the configured list.
    #[must_use]
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Override the cross-vendor reasoning-effort knob for this
    /// call. See [`ReasoningEffort`]'s module doc for the per-vendor
    /// mapping.
    #[must_use]
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Override the tool-selection constraint for this call.
    #[must_use]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Override the structured-output constraint for this call.
    #[must_use]
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Override the cross-vendor parallel-tool-call toggle for this
    /// call. `Some(true)` opts in, `Some(false)` forces serial,
    /// `None` (the default, implicit from not calling this) keeps
    /// the configured default.
    #[must_use]
    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// The temperature override if set.
    #[must_use]
    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// The top-p override if set.
    #[must_use]
    pub const fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    /// The top-k override if set.
    #[must_use]
    pub const fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    /// The max-tokens override if set.
    #[must_use]
    pub const fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// Borrow the stop-sequence override if set.
    #[must_use]
    pub fn stop_sequences(&self) -> Option<&[String]> {
        self.stop_sequences.as_deref()
    }

    /// Borrow the reasoning-effort override if set.
    #[must_use]
    pub const fn reasoning_effort(&self) -> Option<&ReasoningEffort> {
        self.reasoning_effort.as_ref()
    }

    /// Borrow the tool-choice override if set.
    #[must_use]
    pub const fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }

    /// Borrow the response-format override if set.
    #[must_use]
    pub const fn response_format(&self) -> Option<&ResponseFormat> {
        self.response_format.as_ref()
    }

    /// The parallel-tool-calls override if set.
    #[must_use]
    pub const fn parallel_tool_calls(&self) -> Option<bool> {
        self.parallel_tool_calls
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn run_overrides_empty() {
        let o = RunOverrides::new();
        assert!(o.model().is_none());
        assert!(o.system_prompt().is_none());
        assert!(o.max_iterations().is_none());
    }

    #[test]
    fn run_overrides_setters_chain() {
        let o = RunOverrides::new()
            .with_model("haiku")
            .with_system_prompt(SystemPrompt::text("be brief"))
            .with_max_iterations(8);
        assert_eq!(o.model(), Some("haiku"));
        assert!(o.system_prompt().is_some());
        assert_eq!(o.max_iterations(), Some(8));
    }

    #[test]
    fn request_overrides_empty() {
        let r = RequestOverrides::new();
        assert!(r.temperature().is_none());
        assert!(r.top_p().is_none());
        assert!(r.max_tokens().is_none());
        assert!(r.stop_sequences().is_none());
        assert!(r.reasoning_effort().is_none());
        assert!(r.tool_choice().is_none());
        assert!(r.response_format().is_none());
    }

    #[test]
    fn request_overrides_setters_chain() {
        let format = ResponseFormat::strict(
            crate::ir::JsonSchemaSpec::new("answer", serde_json::json!({"type": "object"}))
                .expect("valid schema"),
        );
        let r = RequestOverrides::new()
            .with_temperature(0.3)
            .with_top_p(0.9)
            .with_max_tokens(512)
            .with_stop_sequences(vec!["</done>".into()])
            .with_reasoning_effort(ReasoningEffort::Low)
            .with_tool_choice(ToolChoice::Required)
            .with_response_format(format);
        assert_eq!(r.temperature(), Some(0.3));
        assert_eq!(r.top_p(), Some(0.9));
        assert_eq!(r.max_tokens(), Some(512));
        assert_eq!(r.stop_sequences(), Some(&["</done>".to_string()][..]));
        assert!(matches!(r.reasoning_effort(), Some(&ReasoningEffort::Low)));
        assert!(matches!(r.tool_choice(), Some(&ToolChoice::Required)));
        assert_eq!(r.response_format().expect("set").json_schema.name, "answer");
    }

    #[test]
    fn request_overrides_stop_sequences_empty_vs_none() {
        // Empty Vec is a meaningful override — distinct from None.
        let cleared = RequestOverrides::new().with_stop_sequences(Vec::new());
        assert_eq!(cleared.stop_sequences(), Some(&[][..]));

        let unset = RequestOverrides::new();
        assert!(unset.stop_sequences().is_none());
    }
}
