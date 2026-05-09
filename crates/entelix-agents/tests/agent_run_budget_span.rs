//! regression — `Agent::execute` mirrors the frozen
//! `RunBudget` snapshot onto the `entelix.agent.run` span as six
//! attributes (three `gen_ai.usage.*` token counters, two
//! `entelix.usage.*` aggregates, one `entelix.agent.usage.cost`
//! roll-up). Runs without a budget leave the fields as
//! `tracing::field::Empty`, which the `tracing-opentelemetry`
//! bridge omits from the exported span.
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
    /// Per-span-id snapshot of recorded numeric field values.
    fields: Arc<Mutex<HashMap<String, HashMap<String, u64>>>>,
    /// Per-span-id snapshot of recorded string field values
    /// (cost lands here as a `Decimal`-rendered display string).
    strings: Arc<Mutex<HashMap<String, HashMap<String, String>>>>,
}

impl UsageFieldCapture {
    fn snapshot_for(&self, span_name: &str) -> Option<HashMap<String, u64>> {
        self.fields.lock().get(span_name).cloned()
    }

    fn string_snapshot_for(&self, span_name: &str) -> Option<HashMap<String, String>> {
        self.strings.lock().get(span_name).cloned()
    }
}

struct FieldVisitor<'a> {
    numeric: &'a mut HashMap<String, u64>,
    strings: &'a mut HashMap<String, String>,
}

impl Visit for FieldVisitor<'_> {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.numeric.insert(field.name().to_owned(), value);
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        if let Ok(v) = u64::try_from(value) {
            self.numeric.insert(field.name().to_owned(), v);
        }
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.strings
            .insert(field.name().to_owned(), value.to_owned());
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        // `tracing::field::display(&Decimal)` falls through to
        // `record_debug` on subscribers that don't surface a typed
        // `record_display` method — the wrapped value's Debug impl
        // is what we capture.
        self.strings
            .insert(field.name().to_owned(), format!("{value:?}"));
    }
}

impl<S> tracing_subscriber::Layer<S> for UsageFieldCapture
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: LayerCtx<'_, S>) {
        let Some(meta) = ctx.span(id).map(|s| s.metadata()) else {
            return;
        };
        let mut numeric_store = self.fields.lock();
        let mut string_store = self.strings.lock();
        let numeric_bucket = numeric_store.entry(meta.name().to_owned()).or_default();
        let string_bucket = string_store.entry(meta.name().to_owned()).or_default();
        let mut visitor = FieldVisitor {
            numeric: numeric_bucket,
            strings: string_bucket,
        };
        attrs.record(&mut visitor);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: LayerCtx<'_, S>) {
        let Some(meta) = ctx.span(id).map(|s| s.metadata()) else {
            return;
        };
        let mut numeric_store = self.fields.lock();
        let mut string_store = self.strings.lock();
        let numeric_bucket = numeric_store.entry(meta.name().to_owned()).or_default();
        let string_bucket = string_store.entry(meta.name().to_owned()).or_default();
        let mut visitor = FieldVisitor {
            numeric: numeric_bucket,
            strings: string_bucket,
        };
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
    // Cost axis observation — exercises the 6th OTel field
    // (`entelix.agent.usage.cost`) that the agent root span
    // exposes (per-run cumulative roll-up; distinct from
    // `OtelLayer`'s per-call `gen_ai.usage.cost` increment).
    budget
        .observe_cost(rust_decimal::Decimal::new(125, 4)) // $0.0125
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
    let strings = capture
        .string_snapshot_for("entelix.agent.run")
        .expect("entelix.agent.run span string fields must be captured");
    let cost = strings
        .get("entelix.agent.usage.cost")
        .expect("entelix.agent.usage.cost must be recorded on the span");
    assert!(
        cost.contains("0.0125"),
        "captured cost render must encode the observed Decimal (got {cost:?})"
    );
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
    // surface), but the six usage fields were declared as
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
