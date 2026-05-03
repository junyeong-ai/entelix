//! # entelix-mcp-chatmodel
//!
//! Companion crate that bridges the MCP `sampling/createMessage`
//! server-initiated channel ([`entelix_mcp::SamplingProvider`])
//! to an [`entelix_core::ChatModel<C, T>`] dispatch surface.
//!
//! ## What this crate solves
//!
//! [`entelix_mcp`] ships the `SamplingProvider` trait but
//! deliberately keeps the conversion to a real LLM out of the
//! core MCP crate — the IR / vendor mapping is operator-specific
//! and would force a `ChatModel` dependency on every MCP user.
//! This companion is the canonical adapter: it owns one
//! `ChatModel<C, T>` and translates each
//! [`SamplingRequest`](entelix_mcp::SamplingRequest) into an IR
//! `ModelRequest`, dispatches through the chat model's full layer
//! stack (`PolicyLayer`, `OtelLayer`, retries, …), and maps the
//! resulting [`ModelResponse`](entelix_core::ir::ModelResponse)
//! back to a [`SamplingResponse`](entelix_mcp::SamplingResponse).
//!
//! ## Wiring
//!
//! ```ignore
//! use std::sync::Arc;
//! use entelix_core::ChatModel;
//! use entelix_mcp::McpServerConfig;
//! use entelix_mcp_chatmodel::ChatModelSamplingProvider;
//!
//! let chat = ChatModel::new(codec, transport, "claude-3-5-sonnet")
//!     .with_max_tokens(1024);
//! let provider = Arc::new(ChatModelSamplingProvider::new(chat));
//!
//! let server = McpServerConfig::http("https://server.example/mcp")
//!     .with_sampling_provider(provider);
//! ```
//!
//! Per-request `system_prompt` / `temperature` / `max_tokens` /
//! `stop_sequences` from the sampling request override the chat
//! model's defaults for that single dispatch — the underlying
//! `ChatModel` instance is unchanged. `model_preferences` and
//! `include_context` are advisory in the MCP spec; the default
//! adapter records them via `tracing::debug!` and otherwise
//! passes through unchanged. Operators with bespoke model-routing
//! requirements implement `SamplingProvider` directly and use
//! this crate as a reference.
//!
//! ## Multimodal
//!
//! Image and audio sampling messages translate to
//! [`ContentPart::Image`](entelix_core::ir::ContentPart::Image) /
//! [`ContentPart::Audio`](entelix_core::ir::ContentPart::Audio)
//! with [`MediaSource::Base64`](entelix_core::ir::MediaSource::Base64).
//! The reverse path collapses the response's first text /
//! image / audio content part into the single
//! [`SamplingContent`](entelix_mcp::SamplingContent) the MCP wire
//! format admits. Multi-block responses (typical when the model
//! emits `Thinking` + `Text`) drop the auxiliary blocks with a
//! `tracing::warn!` — MCP sampling cannot carry them.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-mcp-chatmodel/1.0.0-rc.1")]
#![deny(missing_docs)]

mod provider;

pub use provider::ChatModelSamplingProvider;
