//! `ModelRequest` — the provider-neutral request shape (invariant 4).
//!
//! Every model call must pass through this type before reaching a `Codec`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ir::message::Message;
use crate::ir::provider_echo::ProviderEchoSnapshot;
use crate::ir::provider_extensions::ProviderExtensions;
use crate::ir::reasoning::ReasoningEffort;
use crate::ir::structured::ResponseFormat;
use crate::ir::system::SystemPrompt;
use crate::ir::tool_spec::{ToolChoice, ToolSpec};

/// One model invocation, before encoding to vendor wire format.
///
/// Built by users (or higher-level recipes) and handed to `Codec::encode`.
/// Codecs produce vendor-shaped JSON; the IR is the canonical surface and
/// never carries vendor-specific fields directly.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Vendor model identifier (e.g. `claude-opus-4-7`, `gpt-4.1`).
    pub model: String,
    /// Conversation up to this turn. Must contain at least one user message
    /// for most providers; codecs reject empty lists at encode time.
    pub messages: Vec<Message>,
    /// Ordered system-prompt blocks. Empty = "no system prompt"
    /// (codecs treat as if the field were absent). Per-block
    /// [`crate::ir::CacheControl`] is honored natively by codecs
    /// that support it (Anthropic, Bedrock Converse for Claude);
    /// other codecs concatenate block text and emit
    /// `LossyEncode` warnings when any block is cached.
    #[serde(default)]
    pub system: SystemPrompt,
    /// Hard cap on output tokens. `None` = vendor default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature `[0.0, 2.0]`. Codecs clamp to vendor range.
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter — restrict candidate-token sampling
    /// to the `k` most-likely tokens. `None` defers to the vendor
    /// default.
    ///
    /// Codec mapping (CLAUDE.md §"Provider IR promotion"; native on
    /// Anthropic, Gemini, Bedrock Converse on Claude — three
    /// vendors, criterion satisfied):
    /// - **Anthropic**, **Bedrock Converse on Claude** — pass-through
    ///   to the Messages API `top_k` field.
    /// - **Gemini** — pass-through to `generationConfig.topK`.
    /// - **OpenAI Chat** / **OpenAI Responses** — `LossyEncode` (no
    ///   native parameter).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Sequences that, when produced, halt generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Tools advertised to the model. Empty = no tool calls permitted.
    /// Held as `Arc<[ToolSpec]>` so per-dispatch cloning of the
    /// request shape is an atomic refcount bump rather than a deep
    /// walk of every tool's JSON schema. Codecs read through the
    /// `Deref<Target = [ToolSpec]>` coercion — every `&request.tools`
    /// site continues to see `&[ToolSpec]` unchanged.
    #[serde(default)]
    pub tools: Arc<[ToolSpec]>,
    /// Constraint on tool selection. Defaults to [`ToolChoice::Auto`].
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Allow the model to emit more than one tool call in a single
    /// turn. `Some(true)` opts in to parallel tool use, `Some(false)`
    /// forces serial dispatch, `None` defers to the vendor default.
    ///
    /// Codec mapping:
    /// - **Anthropic**, **Bedrock Converse on Claude** — translate to
    ///   `tool_choice.disable_parallel_tool_use` (inverted polarity);
    ///   the codec only emits when a `tool_choice` block is present.
    /// - **OpenAI Chat** / **OpenAI Responses** — pass-through to the
    ///   `parallel_tool_calls` field.
    /// - **Gemini** — `LossyEncode` (no native parallel-tool toggle).
    ///
    /// Promoted to IR per the rule "≥ 2 first-party vendors carry
    /// the concept natively → IR field" (CLAUDE.md §"Provider IR
    /// promotion").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Optional structured-output constraint. Codecs route to
    /// vendor-canonical channels (Anthropic `output_config.format`,
    /// OpenAI `response_format` / `text.format`, Gemini
    /// `responseJsonSchema`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Pseudonymous end-user identifier — abuse-monitoring,
    /// per-user rate-limit attribution, and audit trail. Vendor
    /// pseudonym, never PII (no email / IP / real name).
    ///
    /// Codec mapping (native on Anthropic + OpenAI Chat + OpenAI
    /// Responses — two distinct vendors, criterion satisfied):
    /// - **Anthropic** — `metadata.user_id`.
    /// - **OpenAI Chat** / **OpenAI Responses** — top-level `user`.
    /// - **Gemini**, **Bedrock Converse** — `LossyEncode` (no
    ///   native end-user attribution channel).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_user_id: Option<String>,
    /// Deterministic-generation seed. Same seed + same request →
    /// same output, best-effort (vendors document this as not
    /// strictly guaranteed across model versions).
    ///
    /// Codec mapping (native on OpenAI Chat + OpenAI Responses +
    /// Gemini — two distinct vendors, criterion satisfied):
    /// - **OpenAI Chat** / **OpenAI Responses** — top-level `seed`.
    /// - **Gemini** — `generationConfig.seed`.
    /// - **Anthropic**, **Bedrock Converse** — `LossyEncode` (no
    ///   native deterministic-sampling knob).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Cross-vendor reasoning-effort knob. When `Some`, codecs
    /// translate onto their native wire shape per the mapping in
    /// [`ReasoningEffort`]'s module doc — `Off`/`Minimal`/`Low`/
    /// `Medium`/`High`/`Auto` snap to vendor buckets, lossy
    /// approximations emit `ModelWarning::LossyEncode`, and
    /// `VendorSpecific(s)` passes through the literal vendor wire
    /// value. `None` ⇒ vendor default (codec emits no thinking /
    /// reasoning field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Per-vendor typed knobs that don't generalise to a
    /// cross-provider IR field — e.g. Anthropic
    /// `disable_parallel_tool_use`, Gemini `safetySettings`,
    /// Bedrock guardrails. Codecs read their own ext when encoding
    /// and emit `ModelWarning::ProviderExtensionIgnored` when
    /// another vendor's ext is present (the operator intended a
    /// knob this wire format cannot honour).
    #[serde(default, skip_serializing_if = "ProviderExtensions::is_empty")]
    pub provider_extensions: ProviderExtensions,
    /// Vendor-keyed opaque round-trip tokens carrying state from a
    /// prior turn — OpenAI Responses `previous_response_id` is the
    /// canonical example. Codecs read entries matching their own
    /// `Codec::name` and translate to the vendor's chain-pointer
    /// wire field; non-matching entries are ignored. Empty when the
    /// request does not chain from a prior turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub continued_from: Vec<ProviderEchoSnapshot>,
}

impl ModelRequest {
    /// Advance this request to the next conversational turn — append
    /// the prior assistant turn, chain the vendor's opaque echoes,
    /// and add the next message (whatever its role).
    ///
    /// The transformation is:
    ///
    /// 1. The model's prior reply (`prior_response.content`) is wrapped
    ///    in [`Message::new(Role::Assistant, ...)`](crate::ir::Message::new)
    ///    and pushed to `self.messages`.
    /// 2. `self.continued_from` is replaced with
    ///    `prior_response.provider_echoes` so vendor-specific
    ///    continuation pointers (OpenAI Responses `previous_response_id`,
    ///    Anthropic extended-thinking signatures, Gemini thought
    ///    signatures) ride the next wire encoding. Codecs whose wire
    ///    format represents the prior turn through a server-side
    ///    chain pointer (OpenAI Responses) deduplicate the appended
    ///    assistant turn against that pointer at encode time so the
    ///    transcript is not double-carried.
    /// 3. `next_message` is pushed onto `self.messages`. The role is
    ///    free — `Role::User` is the common case, `Role::Tool` is the
    ///    canonical shape when chaining a tool-round-trip result into
    ///    the next turn, and operator-driven flows that splice a
    ///    `Role::System` correction message are also valid.
    ///
    /// Model / system prompt / tools / response format / sampling
    /// knobs survive unchanged. Callers needing per-turn adjustments
    /// chain further builder methods on the returned value.
    ///
    /// ## Tool round-trip
    ///
    /// `Role::Tool` results for pending `ToolUse` blocks in
    /// `prior_response.content` are passed in via `next_message` —
    /// either directly (single tool round-trip) or by composing
    /// additional pushes on the returned request before the next
    /// dispatch (multi-tool fan-out). The helper does not
    /// auto-synthesise tool results; agent loops dispatch tools and
    /// produce the matching `Message::Tool` themselves.
    ///
    /// ## Why a self-consuming method
    ///
    /// `ModelRequest` is cheap to clone but the chain shape is
    /// fundamentally "previous turn → next turn"; consuming `self`
    /// makes accidental mutation of the old request impossible and
    /// reads naturally at the call site:
    ///
    /// ```ignore
    /// let next = prior_request.continue_turn(&prior_response, Message::user("more"));
    /// ```
    #[must_use]
    pub fn continue_turn(
        mut self,
        prior_response: &crate::ir::response::ModelResponse,
        next_message: crate::ir::message::Message,
    ) -> Self {
        self.messages.push(crate::ir::message::Message::new(
            crate::ir::message::Role::Assistant,
            prior_response.content.clone(),
        ));
        self.continued_from
            .clone_from(&prior_response.provider_echoes);
        self.messages.push(next_message);
        self
    }
}
