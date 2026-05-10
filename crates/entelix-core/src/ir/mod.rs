//! Provider-neutral intermediate representation (invariant 4).
//!
//! Every model call travels through these types before reaching a
//! [`Codec`](crate::codecs::Codec). Vendors that lack support for a given
//! IR field cause the codec to emit a [`ModelWarning::LossyEncode`]
//! (invariant 6) — silent loss is forbidden.
//!
//! Public surface is intentionally flat: `entelix_core::ir::ModelRequest`,
//! `entelix_core::ir::Message`, etc. Sub-modules organize source code.

mod cache;
mod capabilities;
mod content;
mod message;
mod provider_echo;
mod provider_extensions;
mod reasoning;
mod request;
mod response;
mod safety;
mod source;
mod structured;
mod system;
mod tool_spec;
mod usage;
mod warning;

pub use cache::{CacheControl, CacheTtl};
pub use capabilities::Capabilities;
pub use content::{ContentPart, ToolResultContent};
pub use message::{Message, Role};
pub use provider_echo::ProviderEchoSnapshot;
pub use provider_extensions::{
    AnthropicExt, BedrockExt, BedrockGuardrail, GeminiExt, GeminiSafetyOverride, OpenAiChatExt,
    OpenAiResponsesExt, ProviderExtensions, ReasoningSummary, ServiceTier, UrlContext,
};
pub use reasoning::ReasoningEffort;
pub use request::ModelRequest;
pub use response::{ModelResponse, RefusalReason, StopReason, ToolUseRef};
pub use safety::{SafetyCategory, SafetyLevel, SafetyRating};
pub use source::{CitationSource, MediaSource};
pub use structured::{JsonSchemaSpec, OutputStrategy, ResponseFormat, StrictSchemaError};
pub use system::{SystemBlock, SystemPrompt};
pub use tool_spec::{ToolChoice, ToolKind, ToolSpec};
pub use usage::Usage;
pub use warning::ModelWarning;
