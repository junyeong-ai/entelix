//! `MeteredEmbedder<E>` — wraps any `E: Embedder` and emits
//! `gen_ai.embedding.cost` (plus `usage`, `duration_ms`) per call.
//!
//! Cost calculation flows through the [`EmbeddingCostCalculator`]
//! trait so deployments can plug in any pricing source — typically
//! `entelix_policy::CostMeter` for unified billing alongside model
//! and tool costs.
//!
//! ## F4 transactional discipline
//!
//! Cost is computed and emitted **only after** `inner.embed` /
//! `embed_batch` returns Ok. A failed embedder call never produces a
//! phantom charge in telemetry — same rule the model and tool paths
//! enforce for `gen_ai.usage.cost` and `gen_ai.tool.cost`.
//!
//! ## Provider-supplied usage required
//!
//! When the inner embedder returns `Embedding::usage = None` (stub
//! embedders, hash-based encoders) the wrapper still emits a
//! `gen_ai.embedding.start`/`.end` pair for visibility but skips the
//! cost attribute — without a token count there is nothing to charge.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use entelix_core::context::ExecutionContext;
use entelix_core::cost::CostCalculator;
use entelix_core::error::Result;
use entelix_core::ir::Usage;

use crate::traits::{Embedder, Embedding, EmbeddingUsage};

/// Compute a monetary cost for one embedder call.
///
/// Implementors are pure with respect to the caller's request — they
/// may consult internal caches (a pricing table) but must not mutate
/// caller state. Implementations are typically shared across many
/// calls, so they must be `Send + Sync + 'static`.
///
/// `ctx` lets multi-tenant calculators select per-tenant pricing
/// rows via [`ExecutionContext::tenant_id`]. Single-tenant
/// calculators ignore it. Returns `None` when no pricing applies —
/// telemetry consumers omit the cost attribute (silent zero would
/// hide a missing-pricing-row deployment bug).
#[async_trait]
pub trait EmbeddingCostCalculator: Send + Sync + 'static {
    /// Compute the cost of one embedder call given the request
    /// context, the embedder model name (operator-supplied at
    /// `MeteredEmbedder` construction), and the embedder's
    /// reported usage record.
    async fn compute_cost(
        &self,
        model: &str,
        usage: &EmbeddingUsage,
        ctx: &ExecutionContext,
    ) -> Option<f64>;
}

/// `Embedder` decorator that emits OTel-compatible telemetry per
/// call (and optional cost via [`EmbeddingCostCalculator`]).
///
/// Wraps any inner `E: Embedder` and itself implements `Embedder`,
/// so the wrapper drops in transparently anywhere the bare type
/// was used.
pub struct MeteredEmbedder<E>
where
    E: Embedder,
{
    inner: Arc<E>,
    model: Arc<str>,
    cost_calculator: Option<Arc<dyn EmbeddingCostCalculator>>,
}

impl<E> MeteredEmbedder<E>
where
    E: Embedder,
{
    /// Wrap `inner` with a metered surface. `model` is the wire-name
    /// the operator wants surfaced in telemetry (`gen_ai.embedding.model`)
    /// and used as the lookup key in the cost calculator's pricing
    /// table.
    pub fn new(inner: E, model: impl Into<Arc<str>>) -> Self {
        Self {
            inner: Arc::new(inner),
            model: model.into(),
            cost_calculator: None,
        }
    }

    /// Variant for callers that already hold an `Arc<E>` (typical
    /// when the embedder is shared across multiple memory backends).
    pub fn from_arc(inner: Arc<E>, model: impl Into<Arc<str>>) -> Self {
        Self {
            inner,
            model: model.into(),
            cost_calculator: None,
        }
    }

    /// Attach an [`EmbeddingCostCalculator`]. When set, the wrapper
    /// emits `gen_ai.embedding.cost` on the success branch of every
    /// embed / embed_batch call whose `(tenant, model)` resolves to
    /// a pricing row.
    #[must_use]
    pub fn with_cost_calculator(mut self, calculator: Arc<dyn EmbeddingCostCalculator>) -> Self {
        self.cost_calculator = Some(calculator);
        self
    }

    /// Borrow the operator-supplied model name surfaced in
    /// telemetry — useful for tests and for dashboards that
    /// label rows by model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Helper: emit a single `gen_ai.embedding.end` event with the
    /// computed cost (when calculator + usage both present).
    async fn emit_end(
        &self,
        ctx: &ExecutionContext,
        usage: Option<&EmbeddingUsage>,
        duration_ms: u64,
        batch_size: usize,
    ) {
        let cost = match (usage, &self.cost_calculator) {
            (Some(u), Some(calc)) => calc.compute_cost(&self.model, u, ctx).await,
            _ => None,
        };
        let input_tokens = usage.map(|u| u.input_tokens);
        tracing::event!(
            target: "gen_ai",
            tracing::Level::INFO,
            gen_ai.system = "embedder",
            gen_ai.operation.name = "embed",
            gen_ai.embedding.model = %self.model,
            gen_ai.embedding.batch_size = batch_size,
            gen_ai.usage.input_tokens = input_tokens,
            gen_ai.embedding.cost = cost,
            duration_ms,
            entelix.tenant_id = %ctx.tenant_id(),
            entelix.run_id = ctx.run_id(),
            "gen_ai.embedding.end"
        );
    }
}

#[async_trait]
impl<E> Embedder for MeteredEmbedder<E>
where
    E: Embedder,
{
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    async fn embed(&self, text: &str, ctx: &ExecutionContext) -> Result<Embedding> {
        let started_at = Instant::now();
        let result = self.inner.embed(text, ctx).await?;
        let duration_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.emit_end(ctx, result.usage.as_ref(), duration_ms, 1)
            .await;
        Ok(result)
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Vec<Embedding>> {
        let started_at = Instant::now();
        let result = self.inner.embed_batch(texts, ctx).await?;
        let duration_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        // Sum input_tokens across the batch for one combined event.
        // Per-element emission would flood telemetry on large
        // batches; aggregated count gives dashboards the same total
        // at one event per call.
        let aggregated = aggregate_usage(&result);
        self.emit_end(ctx, aggregated.as_ref(), duration_ms, texts.len())
            .await;
        Ok(result)
    }
}

fn aggregate_usage(embeddings: &[Embedding]) -> Option<EmbeddingUsage> {
    let mut total: u32 = 0;
    let mut any = false;
    for e in embeddings {
        if let Some(u) = e.usage {
            total = total.saturating_add(u.input_tokens);
            any = true;
        }
    }
    any.then_some(EmbeddingUsage::new(total))
}

/// Adapter that bridges any [`CostCalculator`] (`ChatModel` pricing
/// source) into the [`EmbeddingCostCalculator`] surface.
///
/// Embeddings only consume input tokens, so the adapter constructs
/// a synthetic [`Usage`] with `input_tokens` populated from the
/// embedder's [`EmbeddingUsage`] and delegates to the wrapped
/// calculator. Operators with a single shared `entelix_policy::CostMeter`
/// pricing table use this to charge embedding calls from the same
/// source as model and tool calls — one pricing source, three cost
/// surfaces, no drift.
pub struct CostCalculatorAdapter {
    inner: Arc<dyn CostCalculator>,
}

impl CostCalculatorAdapter {
    /// Wrap the supplied calculator. The adapter forwards every
    /// embed call as a synthetic `Usage` and lets the inner
    /// calculator's pricing table do the lookup.
    #[must_use]
    pub const fn new(inner: Arc<dyn CostCalculator>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl EmbeddingCostCalculator for CostCalculatorAdapter {
    async fn compute_cost(
        &self,
        model: &str,
        usage: &EmbeddingUsage,
        ctx: &ExecutionContext,
    ) -> Option<f64> {
        let usage = Usage::new(usage.input_tokens, 0);
        self.inner.compute_cost(model, &usage, ctx).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Stub embedder that returns deterministic vectors and reports
    /// usage proportional to input length.
    struct StubEmbedder {
        dim: usize,
    }

    #[async_trait]
    impl Embedder for StubEmbedder {
        fn dimension(&self) -> usize {
            self.dim
        }
        async fn embed(&self, text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
            #[allow(clippy::cast_possible_truncation)]
            let tokens = text.len() as u32;
            Ok(Embedding::new(vec![0.0; self.dim]).with_usage(EmbeddingUsage::new(tokens)))
        }
    }

    /// Stub embedder that fails — used to verify the metered
    /// wrapper does NOT emit cost on the error branch.
    struct FailingEmbedder;

    #[async_trait]
    impl Embedder for FailingEmbedder {
        fn dimension(&self) -> usize {
            4
        }
        async fn embed(&self, _text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
            Err(entelix_core::Error::config("embedder down"))
        }
    }

    /// Counting calculator: returns a fixed cost and tracks calls.
    struct CountingCalculator {
        cost: f64,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EmbeddingCostCalculator for CountingCalculator {
        async fn compute_cost(
            &self,
            _model: &str,
            _usage: &EmbeddingUsage,
            _ctx: &ExecutionContext,
        ) -> Option<f64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Some(self.cost)
        }
    }

    #[tokio::test]
    async fn metered_embed_passes_through_inner_embedding() {
        let metered = MeteredEmbedder::new(StubEmbedder { dim: 8 }, "stub-model");
        let ctx = ExecutionContext::new();
        let out = metered.embed("hello", &ctx).await.unwrap();
        assert_eq!(out.vector.len(), 8);
        assert_eq!(out.usage.unwrap().input_tokens, 5);
    }

    #[tokio::test]
    async fn metered_embed_invokes_calculator_on_success() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calc = Arc::new(CountingCalculator {
            cost: 0.0001,
            calls: calls.clone(),
        });
        let metered =
            MeteredEmbedder::new(StubEmbedder { dim: 4 }, "stub-model").with_cost_calculator(calc);
        let _ = metered
            .embed("hello", &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn metered_embed_skips_calculator_on_failure() {
        // F4: a failed inner call must NEVER trigger cost
        // computation — phantom charges break billing audits.
        let calls = Arc::new(AtomicUsize::new(0));
        let calc = Arc::new(CountingCalculator {
            cost: 0.99,
            calls: calls.clone(),
        });
        let metered =
            MeteredEmbedder::new(FailingEmbedder, "stub-model").with_cost_calculator(calc);
        let err = metered
            .embed("hi", &ExecutionContext::new())
            .await
            .unwrap_err();
        assert!(matches!(err, entelix_core::Error::Config(_)));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "cost calculator must not fire on the error branch"
        );
    }

    #[tokio::test]
    async fn metered_embed_batch_aggregates_usage_into_one_event() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calc = Arc::new(CountingCalculator {
            cost: 0.0,
            calls: calls.clone(),
        });
        let metered =
            MeteredEmbedder::new(StubEmbedder { dim: 2 }, "stub-model").with_cost_calculator(calc);
        let texts = vec!["a".to_owned(), "bb".to_owned(), "ccc".to_owned()];
        let out = metered
            .embed_batch(&texts, &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(out.len(), 3);
        // Calculator should be called exactly once for the aggregate.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// Stub `CostCalculator` (`ChatModel` surface) that records the
    /// `input_tokens` it sees so the adapter test can confirm the
    /// embedding usage round-tripped into the synthetic `Usage`.
    struct ChatStyleCalculator {
        rate_per_token: f64,
        observed_input_tokens: Arc<std::sync::Mutex<Vec<u32>>>,
    }

    #[async_trait]
    impl entelix_core::CostCalculator for ChatStyleCalculator {
        async fn compute_cost(
            &self,
            _model: &str,
            usage: &entelix_core::ir::Usage,
            _ctx: &ExecutionContext,
        ) -> Option<f64> {
            self.observed_input_tokens
                .lock()
                .unwrap()
                .push(usage.input_tokens);
            Some(self.rate_per_token * f64::from(usage.input_tokens))
        }
    }

    #[tokio::test]
    async fn cost_calculator_adapter_forwards_embedding_usage_as_synthetic_usage() {
        // The adapter is the bridge that lets one shared
        // PricingTable charge model, tool, AND embedding surfaces
        // without per-surface duplication. Verify the
        // `EmbeddingUsage::input_tokens` arrives intact at the
        // wrapped `ChatModel` calculator.
        let observed = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let chat_calc = Arc::new(ChatStyleCalculator {
            rate_per_token: 0.0001,
            observed_input_tokens: Arc::clone(&observed),
        });
        let adapter = Arc::new(CostCalculatorAdapter::new(chat_calc));
        let metered = MeteredEmbedder::new(StubEmbedder { dim: 4 }, "text-embedding-3-small")
            .with_cost_calculator(adapter);
        let _ = metered
            .embed("hello world", &ExecutionContext::new())
            .await
            .unwrap();
        let saw = observed.lock().unwrap();
        assert_eq!(saw.len(), 1);
        assert_eq!(saw[0], 11, "stub embedder reports text len as input_tokens");
    }

    #[tokio::test]
    async fn metered_embed_skips_calculator_when_no_usage() {
        struct NoUsageEmbedder;
        #[async_trait]
        impl Embedder for NoUsageEmbedder {
            fn dimension(&self) -> usize {
                4
            }
            async fn embed(&self, _text: &str, _ctx: &ExecutionContext) -> Result<Embedding> {
                // No usage attached — local stub embedders.
                Ok(Embedding::new(vec![0.0; 4]))
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let calc = Arc::new(CountingCalculator {
            cost: 1.0,
            calls: calls.clone(),
        });
        let metered =
            MeteredEmbedder::new(NoUsageEmbedder, "no-usage-model").with_cost_calculator(calc);
        let _ = metered
            .embed("anything", &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "no usage → no cost computation (silent zero would mislead dashboards)"
        );
    }
}
