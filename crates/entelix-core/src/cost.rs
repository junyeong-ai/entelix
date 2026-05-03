//! `CostCalculator` — small trait that computes a monetary cost
//! from a request context, a model name, and a usage record.
//!
//! Sits in `entelix-core` so observability layers (`entelix-otel`)
//! and policy layers (`entelix-policy`) can both speak the same
//! abstraction without one depending on the other. Concrete
//! implementations live in `entelix-policy::CostMeter` (for billing
//! ledgers) and operator-supplied custom types (for non-pricing
//! computations such as carbon tracking).
//!
//! ## Why async, when the canonical impl is synchronous?
//!
//! `compute_cost` is async even though the shipped
//! `entelix_policy::CostMeter` impl is purely synchronous (`Decimal`
//! arithmetic over a `parking_lot::RwLock`). The trade-off is
//! deliberate: dynamic-pricing deployments need to consult remote
//! sources at compute time — vendor pricing APIs, per-tenant rate
//! sheets in a database, FX-rate services for cross-currency
//! billing — and standardising on async now means those impls slot
//! in without a breaking surface change later. The async overhead
//! on the synchronous path is one `async_trait` desugaring per
//! call, dwarfed by the tracing event the cost feeds into. Sync
//! callers wanting maximum throughput compose a `CostMeter` directly
//! via `entelix_policy::CostMeter::charge` (which is not async).
//!
//! ## Per-tenant pricing
//!
//! `compute_cost` accepts `&ExecutionContext` so calculators that
//! need the originating tenant — multi-tenant SaaS deployments
//! charging different per-token rates per customer tier — can read
//! `ctx.tenant_id()` at compute time. Calculators with a global
//! pricing table simply ignore the parameter; the surface stays
//! identical for both shapes.
//!
//! `cost` is `Option<f64>` rather than a fixed type:
//! - `None` — the calculator does not know the model. Telemetry
//!   layers omit the `gen_ai.usage.cost` attribute rather than
//!   surfacing a misleading zero.
//! - `Some(amount)` — the computed monetary cost in the currency
//!   the operator chose at calculator construction time. Conversion
//!   to a billing currency is the caller's responsibility.
//!
//! Width: returning `f64` accepts a small loss-of-precision risk in
//! exchange for trivial JSON serialisation. Callers that need
//! financial-grade exactness use the `entelix-policy` ledger
//! directly; the value emitted into telemetry is for dashboards.

use async_trait::async_trait;

use crate::context::ExecutionContext;
use crate::ir::Usage;

/// Compute a monetary cost for one model invocation.
///
/// Implementors are pure and side-effect free with respect to the
/// caller's request — they may consult internal caches but must
/// not mutate the caller's state. Implementations are typically
/// shared across many calls, so they must be `Send + Sync + 'static`.
#[async_trait]
pub trait CostCalculator: Send + Sync + 'static {
    /// Compute the cost of a single call given the model name, the
    /// response's usage record, and the request context.
    ///
    /// `ctx` lets multi-tenant calculators select per-tenant pricing
    /// rows via [`ExecutionContext::tenant_id`]. Single-tenant
    /// calculators can ignore it.
    ///
    /// Returns `None` when the calculator does not have pricing
    /// for the supplied (tenant, model) pair. Telemetry consumers
    /// treat `None` as "do not emit" rather than as zero — silent
    /// zero in a dashboard looks like a free model and hides
    /// misconfiguration.
    async fn compute_cost(&self, model: &str, usage: &Usage, ctx: &ExecutionContext)
    -> Option<f64>;
}

/// Compute a monetary cost for one tool dispatch.
///
/// Tool calls cost wall-clock time, sometimes external-API spend
/// (paid SaaS endpoints, per-call billing on third-party search
/// engines, MCP servers behind paywalls). A separate trait keeps
/// the model-cost path lightweight while letting deployments that
/// care about tool spend roll their own calculator.
///
/// `output` is the tool's serialised JSON output — calculators
/// that price by output size, response status, or response shape
/// (e.g. row counts) read it; flat per-call calculators ignore it.
#[async_trait]
pub trait ToolCostCalculator: Send + Sync + 'static {
    /// Compute the cost of one tool dispatch. Returns `None` when
    /// no pricing applies — telemetry omits the cost attribute.
    async fn compute_cost(
        &self,
        tool_name: &str,
        output: &serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Option<f64>;
}
