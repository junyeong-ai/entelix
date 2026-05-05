//! # entelix-core
//!
//! DAG root of the entelix workspace — depends on no other entelix crate.
//! Houses the provider-neutral IR, the five codecs (Anthropic Messages,
//! `OpenAI` Chat, `OpenAI` Responses, Gemini, Bedrock Converse), the
//! `Transport` trait + `DirectTransport`, the `Tool` trait +
//! [`ToolRegistry`], `CredentialProvider`, the [`ModelInvocation`] /
//! [`ToolInvocation`] `tower::Service` spine, `EventBus`, and
//! `StreamAggregator`.
//!
//! Cross-cutting concerns (PII redaction, rate limit, cost meter, OTel
//! observability) are `tower::Layer<S>` middleware in their respective
//! sub-crates (`entelix-policy`, `entelix-otel`); the composition
//! primitive is `tower::ServiceBuilder`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-core/1.0.0-rc.1")]
#![deny(missing_docs)]
// Tower-style middleware modules use long opening doc paragraphs to
// explain composition semantics; trait-method `&str` returns satisfy
// the `Tool` trait's published shape but trip the
// `unnecessary_literal_bound` pedantic lint; `Debug` impls for
// service wrappers omit dyn-trait fields by design.
#![allow(
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_fields_in_debug,
    clippy::option_if_let_else,
    clippy::redundant_closure_for_method_calls,
    clippy::too_long_first_doc_paragraph,
    clippy::unnecessary_literal_bound
)]

pub mod approval;
pub mod audit;
pub mod auth;
pub mod backoff;
pub mod cancellation;
pub mod chat;
pub mod codecs;
pub mod context;
pub mod cost;
pub mod error;
pub mod events;
pub mod extensions;
pub mod ir;
pub mod llm_facing;
pub mod overrides;
pub mod rate_limit;
pub mod sandbox;
pub mod service;
pub mod skills;
pub mod stream;
pub mod tenant_id;
pub mod thread_key;
pub mod tools;
pub mod transports;

pub use approval::{ApprovalDecision, INTERRUPT_KIND_APPROVAL_PENDING, PendingApprovalDecisions};
pub use audit::{AuditSink, AuditSinkHandle};
pub use auth::{
    ApiKeyProvider, AuthError, BearerProvider, CachedCredentialProvider, ChainedCredentialProvider,
    CredentialProvider, Credentials,
};
pub use chat::{ChatModel, ChatModelConfig};
pub use context::ExecutionContext;
pub use cost::{CostCalculator, ToolCostCalculator};
pub use error::{Error, ProviderErrorKind, Result};
pub use extensions::Extensions;
pub use llm_facing::{LlmFacingError, LlmFacingSchema};
pub use overrides::RunOverrides;
pub use service::{BoxedModelService, BoxedToolService, ModelInvocation, ToolInvocation};
pub use skills::{
    LoadedSkill, Skill, SkillRegistry, SkillResource, SkillResourceContent, SkillSummary,
};
pub use tenant_id::{DEFAULT_TENANT_ID, TenantId};
pub use thread_key::ThreadKey;
pub use tools::{Tool, ToolRegistry};
