//! # entelix-mcp
//!
//! Native Model Context Protocol client — JSON-RPC 2.0 over MCP
//! streamable-http — with a per-tenant connection pool.
//!
//! Public surface:
//!
//! - [`McpManager`] — register servers, dispatch tool calls + roots
//!   notifications. Pool is keyed by `(TenantId, ServerName)` so
//!   cross-tenant data leakage is structurally impossible (F9
//!   mitigation; invariant 11 strengthening).
//! - [`McpServerConfig`] — HTTP-only by design (invariant 9 forbids
//!   process spawn). Operators wanting stdio MCP servers wrap them
//!   externally and expose an HTTP endpoint.
//! - [`McpClient`] trait — production impl is [`HttpMcpClient`];
//!   tests inject deterministic mocks.
//! - [`McpToolAdapter`] — implements [`entelix_core::tools::Tool`] so
//!   MCP-published tools plug straight into entelix agents.
//! - [`RootsProvider`] / [`McpRoot`] / [`StaticRootsProvider`] —
//!   server-initiated `roots/list` channel. Operators wire one
//!   provider per server through
//!   [`McpServerConfig::with_roots_provider`]; servers gate their
//!   `roots/list` traffic on the capability advertisement.
//! - [`ElicitationProvider`] / [`ElicitationRequest`] /
//!   [`ElicitationResponse`] / [`StaticElicitationProvider`] —
//!   server-initiated `elicitation/create` channel. Same shape
//!   as roots: one provider per server via
//!   [`McpServerConfig::with_elicitation_provider`], capability
//!   advertised iff a provider is wired.
//! - [`SamplingProvider`] / [`SamplingRequest`] /
//!   [`SamplingResponse`] / [`StaticSamplingProvider`] —
//!   server-initiated `sampling/createMessage` channel.
//!   Server asks the client to run an LLM completion on its
//!   behalf. Wire via
//!   [`McpServerConfig::with_sampling_provider`].
//!
//! Lazy provisioning (Anthropic managed-agent shape): the builder
//! only records configuration. Connections open on the first
//! `list_tools` / `call_tool` for a `(tenant, server)` pair.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-mcp/0.2.0")]
#![deny(missing_docs)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::redundant_closure_for_method_calls,
    clippy::match_single_binding,
    clippy::manual_let_else,
    clippy::significant_drop_in_scrutinee,
    clippy::option_if_let_else,
    clippy::match_same_arms,
    clippy::ignored_unit_patterns
)]

#[cfg(feature = "chatmodel-sampling")]
#[cfg_attr(docsrs, doc(cfg(feature = "chatmodel-sampling")))]
mod chatmodel;
mod client;
mod completion;
mod elicitation;
mod error;
mod fsm;
mod manager;
mod prompt;
mod protocol;
mod resource;
mod roots;
mod sampling;
mod server_config;
mod sse;
mod tool_adapter;
mod tool_definition;

#[cfg(feature = "chatmodel-sampling")]
#[cfg_attr(docsrs, doc(cfg(feature = "chatmodel-sampling")))]
pub use chatmodel::ChatModelSamplingProvider;
pub use client::{HttpMcpClient, McpClient};
pub use completion::{McpCompletionArgument, McpCompletionReference, McpCompletionResult};
pub use elicitation::{
    ElicitationProvider, ElicitationRequest, ElicitationResponse, StaticElicitationProvider,
};
pub use error::{McpError, McpResult, ResourceBoundKind};
pub use fsm::McpClientState;
pub use manager::{McpManager, McpManagerBuilder};
pub use prompt::{
    McpPrompt, McpPromptArgument, McpPromptContent, McpPromptInvocation, McpPromptMessage,
    McpPromptResourceRef,
};
pub use protocol::PROTOCOL_VERSION;
pub use resource::{McpResource, McpResourceContent};
pub use roots::{McpRoot, RootsProvider, StaticRootsProvider};
pub use sampling::{
    IncludeContext, ModelHint, ModelPreferences, SamplingContent, SamplingMessage,
    SamplingProvider, SamplingRequest, SamplingResponse, StaticSamplingProvider,
};
pub use server_config::{
    DEFAULT_IDLE_TTL, DEFAULT_LISTENER_CONCURRENCY, DEFAULT_MAX_FRAME_BYTES, DEFAULT_TIMEOUT,
    McpServerConfig, RequestDecorator, validate_server_name,
};
pub use tool_adapter::{McpToolAdapter, qualified_name};
pub use tool_definition::McpToolDefinition;
