//! regression — `Agent::execute` mirrors the frozen
//! `RunBudget` snapshot onto the `entelix.agent.run` span as five
//! `gen_ai.usage.*` / `entelix.usage.*` attributes. Runs without a
//! budget leave the fields as `tracing::field::Empty`, which the
//! `tracing-opentelemetry` bridge omits from the exported span.
//!
//! The test installs a `tracing::subscriber` with a custom `Layer`
//! that captures `Span::record` calls so the assertion is on the
//! actual recorded field values rather than a derived metric. This
//! mirrors the pattern in `entelix-otel/tests/streaming_cost_emit.rs`
//! that asserts `gen_ai.usage.cost` emission on the streaming path.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::significant_drop_tightening
)]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use entelix_agents::Agent;
use entelix_core::{ExecutionContext, Result, RunBudget};
use entelix_runnable::Runnable;
use parking_lot::Mutex;
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context as LayerCtx, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

#[derive(Default, Clone)]
struct UsageFieldCapture {
    /// Per-span-id snapshot of recorded `u64` field values. The
    /// `entelix.agent.run` span is the only one this test exercises
    /// so a single-key map is sufficient — keyed by span name to
    /// keep the assertion site readable.
    fields: Arc<Mutex<HashMap<String, HashMap<String, u64>>>>,
}

impl UsageFieldCapture {
    fn snapshot_for(&self, span_name: &str) -> Option<HashMap<String, u64>> {
        self.fields.lock().get(span_name).cloned()
    }
}

struct U64Visitor<'a> {
    target: &'a mut HashMap<String, u64>,
}

impl Visit for U64Visitor<'_> {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.target.insert(field.name().to_owned(), value);
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        if let Ok(v) = u64::try_from(value) {
            self.target.insert(field.name().to_owned(), v);
        }
    }
    fn record_debug(&mut self, _f: &tracing::field::Field, _v: &dyn std::fmt::Debug) {}
}

impl<S> tracing_subscriber::Layer<S> for UsageFieldCapture
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: LayerCtx<'_, S>) {
        // Capture the span name so subsequent `on_record` calls can
        // be routed to the right bucket.
        let Some(meta) = ctx.span(id).map(|s| s.metadata()) else {
            return;
        };
        let mut store = self.fields.lock();
        let bucket = store.entry(meta.name().to_owned()).or_default();
        let mut visitor = U64Visitor { target: bucket };
        attrs.record(&mut visitor);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: LayerCtx<'_, S>) {
        let Some(meta) = ctx.span(id).map(|s| s.metadata()) else {
            return;
        };
        let mut store = self.fields.lock();
        let bucket = store.entry(meta.name().to_owned()).or_default();
        let mut visitor = U64Visitor { target: bucket };
        values.record(&mut visitor);
    }
}

/// Minimal runnable that returns its input unchanged so the span
/// boundary fires without any vendor wire interaction.
struct Echo;

#[async_trait]
impl Runnable<i32, i32> for Echo {
    async fn invoke(&self, input: i32, _ctx: &ExecutionContext) -> Result<i32> {
        Ok(input)
    }
}

#[tokio::test]
async fn agent_run_span_records_usage_snapshot_when_budget_attached() {
    let capture = UsageFieldCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let agent = Agent::<i32>::builder()
        .with_name("budget-span-test")
        .with_runnable(Echo)
        .build()
        .unwrap();

    // Pre-stamp counters on the budget so the snapshot has
    // observable values to assert against — the inner runnable does
    // not call `ChatModel`, so without a pre-stamp every counter is
    // zero and the test cannot distinguish "recorded 0" from "not
    // recorded". `with_*_limit` is required for the pre-call CAS to
    // actually increment — `check_pre_request` early-returns when
    // no cap is set (the realistic runtime shape sets at least one).
    let budget = RunBudget::unlimited()
        .with_request_limit(100)
        .with_tool_calls_limit(100);
    budget.check_pre_request().unwrap();
    budget.check_pre_request().unwrap();
    budget.check_pre_tool_call().unwrap();
    budget
        .observe_usage(&entelix_core::ir::Usage::new(120, 30))
        .unwrap();
    let ctx = ExecutionContext::new().with_run_budget(budget);

    let _ = agent.execute(7, &ctx).await.unwrap();

    let recorded = capture
        .snapshot_for("entelix.agent.run")
        .expect("entelix.agent.run span must have been entered");

    assert_eq!(
        recorded.get("gen_ai.usage.input_tokens").copied(),
        Some(120)
    );
    assert_eq!(
        recorded.get("gen_ai.usage.output_tokens").copied(),
        Some(30)
    );
    assert_eq!(
        recorded.get("gen_ai.usage.total_tokens").copied(),
        Some(150)
    );
    assert_eq!(recorded.get("entelix.usage.requests").copied(), Some(2));
    assert_eq!(recorded.get("entelix.usage.tool_calls").copied(), Some(1));
}

#[tokio::test]
async fn agent_run_span_omits_usage_fields_when_no_budget_attached() {
    let capture = UsageFieldCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let agent = Agent::<i32>::builder()
        .with_name("no-budget-span-test")
        .with_runnable(Echo)
        .build()
        .unwrap();
    let _ = agent.execute(7, &ExecutionContext::new()).await.unwrap();

    // The span itself was opened (other tests verify the span
    // surface), but the five usage fields were declared as
    // `tracing::field::Empty` and never `record`-ed — so they must
    // not appear in the captured bucket. Empty fields ride through
    // `on_new_span` / `on_record` as no-ops; the visitor never
    // observes them.
    let recorded = capture
        .snapshot_for("entelix.agent.run")
        .unwrap_or_default();
    assert!(
        !recorded.contains_key("gen_ai.usage.input_tokens"),
        "no budget → input_tokens stays Empty (got {recorded:?})"
    );
    assert!(
        !recorded.contains_key("entelix.usage.requests"),
        "no budget → requests stays Empty (got {recorded:?})"
    );
}
