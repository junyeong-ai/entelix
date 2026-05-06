//! # entelix-otel
//!
//! `OpenTelemetry` GenAI-semconv coverage for entelix. The shape is
//! intentionally minimal so the SDK stays optional for users who
//! ship their own observability stack:
//!
//! - [`semconv`] — `gen_ai.*` attribute name constants tracking the
//!   OpenTelemetry GenAI semantic conventions (snapshot 0.31).
//! - [`OtelLayer`] — `tower::Layer<S>` middleware that enriches
//!   `tracing` events around model and tool calls with `gen_ai.*`
//!   attributes and emits per-call metrics. Compose via
//!   `ChatModel::layer(OtelLayer::new("anthropic"))` for model
//!   pipelines and `ToolRegistry::layer(...)` for tool dispatch —
//!   same struct on both sides.
//! - [`GenAiMetrics`] — pre-built `opentelemetry::metrics`
//!   instrument handles (token-usage histogram, operation-duration
//!   histogram) so users don't reinvent the histogram bucket layout.
//! - `init` (cargo feature `otlp`) — convenience helpers that wire
//!   `opentelemetry-otlp` into a `tracing` subscriber. Optional —
//!   teams with their own observability bootstrap skip this.
//!
//! ## Why a tracing-driven design
//!
//! The layer never touches the OpenTelemetry SDK directly. It emits
//! `tracing::event!` events with `gen_ai.*` keys; when the caller
//! has wired `tracing-opentelemetry` (e.g. via
//! `init::init_otlp` under the `otlp` feature) those events become
//! real OTel span events. Without OTel they surface as plain
//! tracing events.
//! Three configurations work uniformly:
//!
//! 1. plain `tracing` (no OTel) — attributes surface as event fields,
//! 2. `tracing` + `tracing-opentelemetry` — events propagate to OTel,
//! 3. headless tests — the same code path with a mock subscriber.
//!
//! ## Quick start
//!
//! ```ignore
//! use entelix_otel::OtelLayer;
//!
//! let model = ChatModel::new(codec, transport, "claude-opus-4-7")
//!     .layer(OtelLayer::new("anthropic"));
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-otel/1.0.0-rc.2")]
#![deny(missing_docs)]
#![allow(
    clippy::doc_markdown,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::option_if_let_else,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_long_first_doc_paragraph
)]

mod layer;
mod metrics;
pub mod semconv;
mod trace_context;

#[cfg(feature = "otlp")]
#[cfg_attr(docsrs, doc(cfg(feature = "otlp")))]
pub mod init;

pub use layer::{DEFAULT_TOOL_IO_TRUNCATION, OtelLayer, OtelService, ToolIoCaptureMode};
pub use metrics::{GenAiMetrics, OperationKind};
pub use trace_context::{TraceContextTransport, trace_context_injector};
