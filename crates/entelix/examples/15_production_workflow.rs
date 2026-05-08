//! `15_production_workflow` — composed end-to-end agent workflow.
//!
//! Build: `cargo build --example 15_production_workflow -p entelix --features policy`
//! Run:   `cargo run --example 15_production_workflow -p entelix --features policy`
//!
//! What this demonstrates (single deterministic run, no external
//! providers, no API keys):
//!
//! 1. **Tenant-scoped state** — every memory access threads
//!    `Namespace::new(tenant_id).with_scope(...)`. Cross-tenant data
//!    is structurally impossible (invariant 11).
//! 2. **Layered conversation memory** — `BufferMemory` for the
//!    rolling window, `SummaryMemory` as the long-horizon record,
//!    `ConsolidatingBufferMemory` automating the
//!    summarise-and-clear loop when a `ConsolidationPolicy` fires.
//!    `EntityMemory` for structured facts the model collects, with
//!    TTL-based pruning so the entity store never grows unbounded.
//! 3. **Cost telemetry** — `CostMeter` doubles as both ledger
//!    (`charge`) and `CostCalculator` (telemetry-side `compute_cost`),
//!    so `OtelLayer::with_cost_calculator` emits `gen_ai.usage.cost`
//!    per turn alongside duration / tokens. The same pricing table
//!    drives both surfaces — no double-source-of-truth drift.
//! 4. **Stable identifiers in events** — `Agent` stamps `run_id` so
//!    OTel + audit sinks correlate every step of one user request.
//!
//! The summariser is a deterministic stub `Runnable` that emits a
//! fixed string. In a real deployment swap it for any
//! `Runnable<Vec<Message>, Message>` driven by your provider of
//! choice — the surrounding plumbing does not change.

#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rust_decimal::Decimal;

use entelix::ir::Message;
use entelix::{
    BufferMemory, ConsolidatingBufferMemory, CostMeter, EntityMemory, EntityRecord,
    ExecutionContext, InMemoryStore, ModelPricing, Namespace, OnMessageCount, PricingTable, Result,
    Runnable, RunnableToSummarizerAdapter, Store, SummaryMemory, TenantId,
};

#[tokio::main]
async fn main() -> Result<()> {
    let tenant_id = TenantId::new("acme");
    let agent_scope = "concierge";
    let conversation_id = "conv-2026-04-28-001";

    // ── Memory wiring ────────────────────────────────────────────
    //
    // Every namespace combines tenant + agent + conversation so
    // distinct users, distinct agents, and distinct threads never
    // alias. This is the F2 mitigation in code form.
    let buffer_ns = Namespace::new(tenant_id.clone())
        .with_scope(agent_scope)
        .with_scope(conversation_id);
    let summary_ns = buffer_ns.clone();
    let entity_ns = Namespace::new(tenant_id.clone()).with_scope(agent_scope);

    let buffer_store: Arc<dyn Store<Vec<Message>>> = Arc::new(InMemoryStore::<Vec<Message>>::new());
    let summary_store: Arc<dyn Store<String>> = Arc::new(InMemoryStore::<String>::new());
    let entity_store: Arc<dyn Store<HashMap<String, EntityRecord>>> =
        Arc::new(InMemoryStore::<HashMap<String, EntityRecord>>::new());

    let buffer = Arc::new(
        BufferMemory::new(buffer_store, buffer_ns, 32)
            // Trigger consolidation once the buffer accumulates 4
            // messages; production deployments use OnTokenBudget.
            .with_consolidation_policy(Arc::new(OnMessageCount::new(4))),
    );
    let summary = Arc::new(SummaryMemory::new(summary_store, summary_ns));
    let entities = Arc::new(EntityMemory::new(entity_store, entity_ns));

    // The summariser would normally be a real chat model — here a
    // deterministic stub keeps the example reproducible.
    let summariser = Arc::new(RunnableToSummarizerAdapter::new(StubChatModel::new(
        "Summary: user signed up for premium and prefers Korean responses.",
    )));
    let memory =
        ConsolidatingBufferMemory::new(Arc::clone(&buffer), Arc::clone(&summary), summariser);

    let ctx = ExecutionContext::new()
        .with_tenant_id(tenant_id)
        .with_thread_id(conversation_id);

    // ── Conversation turn-by-turn ────────────────────────────────
    memory
        .append(&ctx, Message::user("Hi, I'd like to upgrade to premium."))
        .await?;
    memory
        .append(
            &ctx,
            Message::assistant("Got it. What language do you prefer?"),
        )
        .await?;
    memory.append(&ctx, Message::user("Korean please.")).await?;
    // The fourth append crosses the policy threshold — buffer is
    // summarised into `summary`, then cleared, then `last_consolidated_at`
    // advances inside the BufferMemory.
    memory
        .append(&ctx, Message::assistant("Done — anything else?"))
        .await?;

    println!(
        "buffer after consolidation: {} messages",
        memory.messages(&ctx).await?.len()
    );
    println!(
        "running summary: {}",
        memory
            .current_summary(&ctx)
            .await?
            .as_deref()
            .unwrap_or("<none>")
    );
    println!("last_consolidated_at: {:?}", buffer.last_consolidated_at());

    // ── Entity facts with TTL ────────────────────────────────────
    entities
        .set_entity(&ctx, "preferred_language", "Korean")
        .await?;
    entities
        .set_entity(&ctx, "subscription_tier", "premium")
        .await?;
    println!(
        "preferred_language: {:?}",
        entities.entity(&ctx, "preferred_language").await?
    );

    // 24-hour TTL — long enough that nothing prunes in this run, but
    // a long-running deployment calls `prune_older_than` on a cron
    // and the store stays bounded.
    let pruned = entities
        .prune_older_than(&ctx, Duration::from_mins(1440))
        .await?;
    println!("entity records pruned by 24h TTL: {pruned}");

    // ── Cost calculator (telemetry side) ─────────────────────────
    let pricing = PricingTable::new().add_model_pricing(
        "claude-opus-4-7",
        // Illustrative rates: $3 / MTok input, $15 / MTok output,
        // cache-read = 10% of input, cache-write = +25% premium —
        // matches Anthropic's published Sonnet 4.6 posture. Every
        // rate is mandatory (invariant #15 — no silent fallback).
        ModelPricing::new(
            Decimal::new(3, 3),
            Decimal::new(15, 3),
            Decimal::new(3, 4),
            Decimal::new(375, 5),
        ),
    );
    let _meter = CostMeter::new(pricing);
    // In a real deployment this becomes:
    //
    //     let layer = OtelLayer::new("anthropic")
    //         .with_cost_calculator(Arc::new(meter.clone()));
    //     let model = ChatModel::new(...).layer(layer);
    //
    // Here we just demonstrate construction — wiring a transport
    // would require provider credentials.
    println!("cost calculator wired (CostMeter implements CostCalculator)");

    Ok(())
}

/// Deterministic `Runnable<Vec<Message>, Message>` — replaces the
/// real chat model so the example produces stable output.
struct StubChatModel {
    reply: String,
}

impl StubChatModel {
    fn new(reply: &str) -> Self {
        Self {
            reply: reply.to_owned(),
        }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, Message> for StubChatModel {
    async fn invoke(&self, _input: Vec<Message>, _ctx: &ExecutionContext) -> Result<Message> {
        Ok(Message::assistant(self.reply.clone()))
    }
}
