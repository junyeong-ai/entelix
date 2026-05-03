//! Stateful HTTP carriers — `Transport` trait + concrete impls — and
//! the [`RetryLayer`] middleware that wraps any retry-classified
//! `Service`.
//!
//! `DirectTransport` is the simple HTTPS-to-vendor case (Anthropic,
//! OpenAI, Gemini API hosts). `BedrockTransport`, `VertexTransport`,
//! and `FoundryTransport` (cloud-routed signing variants) live in
//! `entelix-cloud`. The transports themselves are retry-naive: they
//! execute exactly one HTTP call per `send`. Retry is the
//! [`RetryLayer`]'s responsibility — operators wire it through
//! `ChatModel::layer(...)` / `ToolRegistry::layer(...)`.

mod direct;
mod retry;
mod transport;

pub use direct::DirectTransport;
pub use retry::{
    DefaultRetryClassifier, RetryClassifier, RetryDecision, RetryLayer, RetryPolicy, RetryService,
    Retryable, parse_retry_after,
};
pub use transport::{Transport, TransportResponse, TransportStream};
