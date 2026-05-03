//! `GenAiMetrics` — pre-built `opentelemetry::metrics` instrument
//! handles aligned with the GenAI semantic conventions.
//!
//! All instruments are created lazily from a `Meter` you provide
//! (typically obtained from `opentelemetry::global::meter("entelix")`).
//! The struct is `Clone` (cheap — instruments are reference-counted
//! handles) so it can be shared across threads / hooks.

use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Histogram, Meter};
use serde::{Deserialize, Serialize};

use entelix_core::ir::Usage;

use crate::semconv;

/// What kind of operation produced the metric. Mirrors the
/// `gen_ai.operation.name` semconv attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OperationKind {
    /// `chat` — multi-turn chat completion.
    Chat,
    /// `text_completion` — single-shot text completion (legacy).
    TextCompletion,
    /// `embeddings` — embedding generation.
    Embeddings,
    /// `execute_tool` — a single tool invocation.
    ExecuteTool,
}

impl OperationKind {
    /// String form, suitable for the
    /// [`semconv::OPERATION_NAME`] attribute.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::TextCompletion => "text_completion",
            Self::Embeddings => "embeddings",
            Self::ExecuteTool => "execute_tool",
        }
    }
}

/// What sort of usage tokens are being recorded. The semconv
/// ships a single `gen_ai.client.token.usage` histogram tagged
/// by `gen_ai.token.type` — we model that tag with this enum.
///
/// `Input` / `Output` are the standard semconv values;
/// `Cached` aligns with semconv 0.32 (cache reads);
/// `CacheCreation` and `Reasoning` are entelix-specific
/// extensions documented in [`crate::semconv`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Input,
    Output,
    Cached,
    CacheCreation,
    Reasoning,
}

impl TokenKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Input => semconv::TOKEN_TYPE_INPUT,
            Self::Output => semconv::TOKEN_TYPE_OUTPUT,
            Self::Cached => semconv::TOKEN_TYPE_CACHED,
            Self::CacheCreation => semconv::TOKEN_TYPE_CACHE_CREATION,
            Self::Reasoning => semconv::TOKEN_TYPE_REASONING,
        }
    }
}

/// Pre-built per-call instrument set.
#[derive(Clone, Debug)]
pub struct GenAiMetrics {
    token_usage: Histogram<u64>,
    operation_duration: Histogram<f64>,
}

impl GenAiMetrics {
    /// Build from a [`Meter`]. Typical setup:
    ///
    /// ```ignore
    /// let meter = opentelemetry::global::meter("entelix");
    /// let metrics = entelix_otel::GenAiMetrics::new(&meter);
    /// ```
    #[must_use]
    pub fn new(meter: &Meter) -> Self {
        let token_usage = meter
            .u64_histogram(semconv::METRIC_TOKEN_USAGE)
            .with_description("Number of input or output tokens used per GenAI call")
            .with_unit("{token}")
            .build();
        let operation_duration = meter
            .f64_histogram(semconv::METRIC_OPERATION_DURATION)
            .with_description("Wall-clock duration of a GenAI call, in seconds")
            .with_unit("s")
            .build();
        Self {
            token_usage,
            operation_duration,
        }
    }

    /// Record a completed call: emits one duration sample plus one
    /// token-usage sample per non-zero counter (input, output).
    pub fn record_call(
        &self,
        system: &str,
        operation: OperationKind,
        request_model: &str,
        response_model: &str,
        usage: &Usage,
        duration: Duration,
    ) {
        let attrs = base_attrs(system, operation, request_model, response_model);
        self.operation_duration
            .record(duration.as_secs_f64(), &attrs);
        if usage.input_tokens > 0 {
            self.record_tokens(&attrs, TokenKind::Input, u64::from(usage.input_tokens));
        }
        if usage.output_tokens > 0 {
            self.record_tokens(&attrs, TokenKind::Output, u64::from(usage.output_tokens));
        }
        // Cache + reasoning samples emit only when non-zero so
        // namespaces without prompt caching / reasoning models
        // don't pollute their dashboards with a constant 0
        // series. Operators filter on `gen_ai.token.type` to
        // separate the buckets at query time.
        if usage.cached_input_tokens > 0 {
            self.record_tokens(
                &attrs,
                TokenKind::Cached,
                u64::from(usage.cached_input_tokens),
            );
        }
        if usage.cache_creation_input_tokens > 0 {
            self.record_tokens(
                &attrs,
                TokenKind::CacheCreation,
                u64::from(usage.cache_creation_input_tokens),
            );
        }
        if usage.reasoning_tokens > 0 {
            self.record_tokens(
                &attrs,
                TokenKind::Reasoning,
                u64::from(usage.reasoning_tokens),
            );
        }
    }

    fn record_tokens(&self, base: &[KeyValue], kind: TokenKind, count: u64) {
        let mut attrs = base.to_vec();
        attrs.push(KeyValue::new(semconv::TOKEN_TYPE, kind.as_str()));
        self.token_usage.record(count, &attrs);
    }
}

fn base_attrs(
    system: &str,
    operation: OperationKind,
    request_model: &str,
    response_model: &str,
) -> Vec<KeyValue> {
    vec![
        KeyValue::new(semconv::SYSTEM, system.to_owned()),
        KeyValue::new(semconv::OPERATION_NAME, operation.as_str()),
        KeyValue::new(semconv::REQUEST_MODEL, request_model.to_owned()),
        KeyValue::new(semconv::RESPONSE_MODEL, response_model.to_owned()),
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use opentelemetry::global;

    use super::*;

    #[test]
    fn metrics_construct_against_global_meter() {
        // No exporter wired → instruments are no-ops, but the build
        // path itself must not panic.
        let meter = global::meter("entelix-test");
        let metrics = GenAiMetrics::new(&meter);
        metrics.record_call(
            "anthropic",
            OperationKind::Chat,
            "claude-opus-4-7",
            "claude-opus-4-7-20260415",
            &Usage::new(100, 50),
            Duration::from_millis(123),
        );
    }

    #[test]
    fn operation_kind_strings_align_with_semconv() {
        assert_eq!(OperationKind::Chat.as_str(), "chat");
        assert_eq!(OperationKind::TextCompletion.as_str(), "text_completion");
        assert_eq!(OperationKind::Embeddings.as_str(), "embeddings");
        assert_eq!(OperationKind::ExecuteTool.as_str(), "execute_tool");
    }

    #[test]
    fn zero_tokens_emit_no_token_sample() {
        // Coverage check: even with zero tokens, no panic from the
        // histogram. We can't read back recorded values without a
        // reader; this test just guards the code path.
        let meter = global::meter("entelix-test-zero");
        let metrics = GenAiMetrics::new(&meter);
        metrics.record_call(
            "anthropic",
            OperationKind::Chat,
            "claude",
            "claude",
            &Usage::default(),
            Duration::from_secs(0),
        );
    }

    #[test]
    fn cache_and_reasoning_token_kinds_emit_when_non_zero() {
        // Coverage check for the per-kind sampling extension.
        // Cache + reasoning paths must not panic and must accept
        // their respective non-zero counters. Without a metric
        // reader we can't introspect emitted values; test guards
        // the dispatch path only.
        let meter = global::meter("entelix-test-cache");
        let metrics = GenAiMetrics::new(&meter);
        metrics.record_call(
            "anthropic",
            OperationKind::Chat,
            "claude-opus-4-7",
            "claude-opus-4-7-20260415",
            &Usage::new(100, 50)
                .with_cached_input_tokens(800)
                .with_cache_creation_input_tokens(200)
                .with_reasoning_tokens(30),
            Duration::from_millis(456),
        );
    }

    #[test]
    fn token_kind_strings_align_with_semconv_constants() {
        assert_eq!(TokenKind::Input.as_str(), crate::semconv::TOKEN_TYPE_INPUT);
        assert_eq!(
            TokenKind::Output.as_str(),
            crate::semconv::TOKEN_TYPE_OUTPUT
        );
        assert_eq!(
            TokenKind::Cached.as_str(),
            crate::semconv::TOKEN_TYPE_CACHED
        );
        assert_eq!(
            TokenKind::CacheCreation.as_str(),
            crate::semconv::TOKEN_TYPE_CACHE_CREATION
        );
        assert_eq!(
            TokenKind::Reasoning.as_str(),
            crate::semconv::TOKEN_TYPE_REASONING
        );
    }
}
