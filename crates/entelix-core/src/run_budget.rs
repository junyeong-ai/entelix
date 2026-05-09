//! `RunBudget` — six-axis usage cap checked across one logical
//! run, including sub-agent fan-out.
//!
//! | Axis                    | Type      | Pre-call check | Post-call accumulation |
//! |-------------------------|-----------|----------------|------------------------|
//! | `request_limit`         | u32       | ✓              | accumulate on Ok       |
//! | `input_tokens_limit`    | u64       | —              | check on Ok            |
//! | `output_tokens_limit`   | u64       | —              | check on Ok            |
//! | `total_tokens_limit`    | u64       | —              | check on Ok            |
//! | `tool_calls_limit`      | u32       | ✓              | accumulate on Ok       |
//! | `cost_usd_limit`        | `Decimal` | —              | check on Ok            |
//!
//! Pre-call axes (`request_limit`, `tool_calls_limit`) are checked
//! before the dispatch reaches the wire — the SDK knows the
//! caller is about to issue request `N+1` and refuses if the cap
//! is `N`. Token axes are post-call: the budget sees the response
//! `Usage` only after the codec decodes, so the breach surfaces
//! on the call that pushed the cumulative total past the limit.
//!
//! ## Sub-agent fan-out
//!
//! `RunBudget` carries an `Arc<RunBudgetState>` of atomic
//! counters. Cloning the budget — done implicitly when an
//! `ExecutionContext` flows into a `Subagent::execute` — bumps
//! the Arc refcount; the sub-agent's calls accumulate into the
//! same counters as the parent's. Compared to per-instance
//! counters, this is `(a)` cheaper (no message-passing between
//! parent and child runtimes) and `(b)` correct under tokio's
//! work-stealing executor (atomic ordering is the cross-task
//! synchronisation primitive).
//!
//! ## Wiring
//!
//! Operators attach a `RunBudget` to the `ExecutionContext` via
//! [`crate::ExecutionContext::with_run_budget`]. Every `ChatModel`
//! dispatch site (`complete_full`, `complete_typed`,
//! `stream_deltas`) reads the budget from `ctx`, calls
//! `check_pre_request` before the wire roundtrip, and calls
//! `observe_usage` on the `Ok` branch (invariant 12 — never on
//! the error branch, otherwise a network failure would still
//! drain the budget). A budget breach surfaces as
//! [`crate::Error::UsageLimitExceeded`] with the breaching axis
//! and observed value.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::ir::Usage;

/// One [`RunBudget`] axis breach — typed pair of axis-discriminator
/// and magnitude. Each variant carries the magnitude shape the axis
/// uses (`u64` count for token / request / tool-call axes,
/// [`Decimal`] USD for the cost axis), so axis-magnitude pairing is
/// type-enforced rather than runtime-validated.
///
/// Carried on [`crate::Error::UsageLimitExceeded`] and
/// `entelix_session::GraphEvent::UsageLimitExceeded`; emitted to
/// `AuditSink::record_usage_limit_exceeded` for compliance /
/// billing replay.
///
/// `non_exhaustive` so post-1.0 axes ship as MINOR. Construct via
/// the typed variants directly — `UsageLimitBreach::Requests {
/// limit, observed }` is the canonical shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "axis", rename_all = "snake_case")]
#[non_exhaustive]
pub enum UsageLimitBreach {
    /// Request-count cap breached. Pre-call check fired.
    Requests {
        /// Configured cap (model dispatches per run).
        limit: u64,
        /// Counter value when the cap was hit.
        observed: u64,
    },
    /// Cumulative-input-tokens cap breached. Post-call check fired.
    InputTokens {
        /// Configured cap (cumulative input tokens).
        limit: u64,
        /// Cumulative input tokens after the breaching call.
        observed: u64,
    },
    /// Cumulative-output-tokens cap breached. Post-call check fired.
    OutputTokens {
        /// Configured cap (cumulative output tokens).
        limit: u64,
        /// Cumulative output tokens after the breaching call.
        observed: u64,
    },
    /// Cumulative input + output tokens cap breached.
    TotalTokens {
        /// Configured cap (cumulative input + output tokens).
        limit: u64,
        /// Cumulative total after the breaching call.
        observed: u64,
    },
    /// Tool-call-count cap breached. Pre-call check fired.
    ToolCalls {
        /// Configured cap (tool dispatches per run).
        limit: u64,
        /// Counter value when the cap was hit.
        observed: u64,
    },
    /// USD cost cap breached. Post-call check fired after
    /// [`RunBudget::observe_cost`] accumulated the per-call charge.
    CostUsd {
        /// Configured cap in USD.
        limit: Decimal,
        /// Cumulative cost after the breaching charge.
        observed: Decimal,
    },
}

impl UsageLimitBreach {
    /// Stable axis-name string used for OTel attribute keys,
    /// dashboards, and `AuditSink` filtering. Matches the
    /// snake-case `serde` tag.
    ///
    /// Operator-facing API even though [`std::fmt::Display`] also
    /// encodes the axis — dashboards, log filters, and serde-keyed
    /// attribute emitters need a stable `&'static str` they can
    /// compare without parsing the human-readable Display
    /// rendering.
    #[must_use]
    pub const fn axis_name(&self) -> &'static str {
        match self {
            Self::Requests { .. } => "requests",
            Self::InputTokens { .. } => "input_tokens",
            Self::OutputTokens { .. } => "output_tokens",
            Self::TotalTokens { .. } => "total_tokens",
            Self::ToolCalls { .. } => "tool_calls",
            Self::CostUsd { .. } => "cost_usd",
        }
    }
}

impl std::fmt::Display for UsageLimitBreach {
    /// Grep-consistent rendering across every axis:
    /// `run budget exceeded on <axis> axis: observed <N>, limit <N>`.
    /// Token / request / tool-call axes render `<N>` as a bare
    /// number; the cost axis renders `<N>` as a `Decimal` rendered
    /// in plain (un-prefixed) form. Dashboards regex-extracting
    /// `observed (\S+), limit (\S+)` get the magnitude on every
    /// axis without a polarity-by-axis branch.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let axis = self.axis_name();
        match self {
            Self::Requests { limit, observed }
            | Self::InputTokens { limit, observed }
            | Self::OutputTokens { limit, observed }
            | Self::TotalTokens { limit, observed }
            | Self::ToolCalls { limit, observed } => {
                write!(
                    f,
                    "run budget exceeded on {axis} axis: observed {observed}, limit {limit}"
                )
            }
            Self::CostUsd { limit, observed } => {
                write!(
                    f,
                    "run budget exceeded on {axis} axis: observed {observed}, limit {limit}"
                )
            }
        }
    }
}

/// Six-axis usage cap shared across one logical run (parent
/// agent + every sub-agent it dispatches). Cloning the budget —
/// done implicitly per `ExecutionContext` clone — bumps the
/// internal `Arc` refcount; sub-agent calls accumulate into the
/// same counters as the parent's.
///
/// Construct via [`Self::unlimited`] (every axis disabled — the
/// default) or via [`Self::default`] (alias). Set per-axis caps
/// with the `with_*_limit` builders. The result is a snapshot
/// builder; call sites read it through
/// [`crate::ExecutionContext::run_budget`].
#[derive(Clone, Debug, Default)]
pub struct RunBudget {
    request_limit: Option<u32>,
    input_tokens_limit: Option<u64>,
    output_tokens_limit: Option<u64>,
    total_tokens_limit: Option<u64>,
    tool_calls_limit: Option<u32>,
    cost_usd_limit: Option<Decimal>,
    state: Arc<RunBudgetState>,
}

#[derive(Debug, Default)]
struct RunBudgetState {
    requests: AtomicU32,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    tool_calls: AtomicU32,
    cost_usd: Mutex<Decimal>,
}

impl RunBudget {
    /// Build with every axis disabled. Caps are set via
    /// `with_*_limit` chained calls.
    #[must_use]
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Cap the number of model dispatches per run. Pre-call
    /// check — the SDK refuses request `N+1` when the cap is
    /// `N`, before the wire roundtrip.
    #[must_use]
    pub const fn with_request_limit(mut self, n: u32) -> Self {
        self.request_limit = Some(n);
        self
    }

    /// Cap cumulative input tokens. Post-call check — the SDK
    /// observes `response.usage.input_tokens` after the codec
    /// decodes and surfaces a breach on the call that pushed the
    /// running total past the cap.
    #[must_use]
    pub const fn with_input_tokens_limit(mut self, n: u64) -> Self {
        self.input_tokens_limit = Some(n);
        self
    }

    /// Cap cumulative output tokens. Post-call.
    #[must_use]
    pub const fn with_output_tokens_limit(mut self, n: u64) -> Self {
        self.output_tokens_limit = Some(n);
        self
    }

    /// Cap cumulative input + output tokens. Post-call. Lets
    /// operators set one ceiling without splitting the
    /// per-direction caps.
    #[must_use]
    pub const fn with_total_tokens_limit(mut self, n: u64) -> Self {
        self.total_tokens_limit = Some(n);
        self
    }

    /// Cap the number of tool dispatches per run. Pre-call check
    /// — the SDK refuses tool call `N+1` when the cap is `N`,
    /// before the dispatched tool's [`crate::tools::Tool::execute`]
    /// runs.
    #[must_use]
    pub const fn with_tool_calls_limit(mut self, n: u32) -> Self {
        self.tool_calls_limit = Some(n);
        self
    }

    /// Cap cumulative USD cost across the run. Operators wire a
    /// [`crate::CostCalculator`] (the same one
    /// `entelix-policy::CostMeter` consumes); the dispatch site
    /// calls [`Self::observe_cost`] on the `Ok` branch with the
    /// per-call charge, and the budget surfaces a breach when the
    /// running total crosses `limit`. Decimal precision matches
    /// the cost meter's `rust_decimal` precision.
    #[must_use]
    pub const fn with_cost_limit_usd(mut self, limit: Decimal) -> Self {
        self.cost_usd_limit = Some(limit);
        self
    }

    /// Pre-request gate — checks the request-count cap and, on
    /// success, increments the request counter. Call from the
    /// dispatch site **before** the wire roundtrip. Returns
    /// `Error::UsageLimitExceeded(UsageLimitBreach::Requests {.})`
    /// when the cap is hit; the counter is not incremented on
    /// failure (the request did not actually fire).
    pub fn check_pre_request(&self) -> Result<()> {
        if let Some(limit) = self.request_limit {
            // Atomic compare-and-swap loop: read current, refuse
            // when at-or-over cap, otherwise increment. Avoids
            // the race where two concurrent calls both pass a
            // `load() < limit` check and then both `fetch_add`,
            // overshooting the cap by one.
            loop {
                let current = self.state.requests.load(Ordering::Acquire);
                if u64::from(current) >= u64::from(limit) {
                    return Err(Error::UsageLimitExceeded(UsageLimitBreach::Requests {
                        limit: u64::from(limit),
                        observed: u64::from(current),
                    }));
                }
                if self
                    .state
                    .requests
                    .compare_exchange_weak(
                        current,
                        current.saturating_add(1),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Pre-tool-call gate — same shape as
    /// [`Self::check_pre_request`] but for the `tool_calls_limit`
    /// axis. Call from the tool dispatch site
    /// (`ToolRegistry::dispatch_*`) **before** the tool's
    /// `execute` runs.
    pub fn check_pre_tool_call(&self) -> Result<()> {
        if let Some(limit) = self.tool_calls_limit {
            loop {
                let current = self.state.tool_calls.load(Ordering::Acquire);
                if u64::from(current) >= u64::from(limit) {
                    return Err(Error::UsageLimitExceeded(UsageLimitBreach::ToolCalls {
                        limit: u64::from(limit),
                        observed: u64::from(current),
                    }));
                }
                if self
                    .state
                    .tool_calls
                    .compare_exchange_weak(
                        current,
                        current.saturating_add(1),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Post-call accumulation — adds the observed usage to the
    /// token counters and surfaces a breach on the axis that
    /// crossed its cap. Call from the dispatch site on the
    /// **`Ok` branch only** (invariant 12 — failed calls never
    /// drain the budget).
    ///
    /// When multiple axes breach simultaneously (e.g. a single
    /// large response trips both `output_tokens_limit` and
    /// `total_tokens_limit`), the function reports the first
    /// axis it encounters in the order `InputTokens` →
    /// `OutputTokens` → `TotalTokens`. Operators that need
    /// every breach surface attach observers via
    /// [`Self::snapshot`].
    pub fn observe_usage(&self, usage: &Usage) -> Result<()> {
        let new_in = self
            .state
            .input_tokens
            .fetch_add(u64::from(usage.input_tokens), Ordering::AcqRel)
            .saturating_add(u64::from(usage.input_tokens));
        let new_out = self
            .state
            .output_tokens
            .fetch_add(u64::from(usage.output_tokens), Ordering::AcqRel)
            .saturating_add(u64::from(usage.output_tokens));
        if let Some(limit) = self.input_tokens_limit
            && new_in > limit
        {
            return Err(Error::UsageLimitExceeded(UsageLimitBreach::InputTokens {
                limit,
                observed: new_in,
            }));
        }
        if let Some(limit) = self.output_tokens_limit
            && new_out > limit
        {
            return Err(Error::UsageLimitExceeded(UsageLimitBreach::OutputTokens {
                limit,
                observed: new_out,
            }));
        }
        if let Some(limit) = self.total_tokens_limit {
            let total = new_in.saturating_add(new_out);
            if total > limit {
                return Err(Error::UsageLimitExceeded(UsageLimitBreach::TotalTokens {
                    limit,
                    observed: total,
                }));
            }
        }
        Ok(())
    }

    /// Post-call cost accumulation — adds the per-call USD charge
    /// to the running total and surfaces a breach if it crossed
    /// `cost_usd_limit`. Call from the dispatch site on the **`Ok`
    /// branch only** (invariant 12 — failed calls never drain the
    /// budget). Operators integrate by computing the charge from a
    /// [`crate::CostCalculator`] and threading the result here.
    pub fn observe_cost(&self, charge_usd: Decimal) -> Result<()> {
        let observed = {
            let mut accumulated = self
                .state
                .cost_usd
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *accumulated = accumulated.saturating_add(charge_usd);
            *accumulated
        };
        if let Some(limit) = self.cost_usd_limit
            && observed > limit
        {
            return Err(Error::UsageLimitExceeded(UsageLimitBreach::CostUsd {
                limit,
                observed,
            }));
        }
        Ok(())
    }

    /// Snapshot the current counter state. Returns owned values
    /// at a single point in time; subsequent mutations on the
    /// budget do not affect the returned snapshot. Used by
    /// [`crate::ExecutionContext`] consumers and by the
    /// `AgentRunResult<S>` envelope (B-5) to expose the final
    /// usage to callers without leaking the live `Arc`.
    #[must_use]
    pub fn snapshot(&self) -> UsageSnapshot {
        let cost_usd = *self
            .state
            .cost_usd
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        UsageSnapshot {
            requests: self.state.requests.load(Ordering::Acquire),
            input_tokens: self.state.input_tokens.load(Ordering::Acquire),
            output_tokens: self.state.output_tokens.load(Ordering::Acquire),
            tool_calls: self.state.tool_calls.load(Ordering::Acquire),
            cost_usd,
        }
    }
}

/// Frozen snapshot of [`RunBudget`] counters at one point in
/// time. Carried in `AgentRunResult<S>::usage` (B-5) so callers
/// see the final tally without needing to clone the budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UsageSnapshot {
    /// Total model dispatches the run made.
    pub requests: u32,
    /// Cumulative input tokens.
    pub input_tokens: u64,
    /// Cumulative output tokens.
    pub output_tokens: u64,
    /// Total tool dispatches.
    pub tool_calls: u32,
    /// Cumulative USD cost across the run (sum of every
    /// `observe_cost` charge). Operators that don't wire a
    /// [`crate::CostCalculator`] see [`Decimal::ZERO`].
    pub cost_usd: Decimal,
}

impl UsageSnapshot {
    /// Sum of [`Self::input_tokens`] and [`Self::output_tokens`].
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::ir::Usage;

    #[test]
    fn unlimited_budget_passes_every_check() {
        let budget = RunBudget::unlimited();
        budget.check_pre_request().unwrap();
        budget.check_pre_tool_call().unwrap();
        budget
            .observe_usage(&Usage::new(1_000_000, 1_000_000))
            .unwrap();
    }

    #[test]
    fn request_limit_pre_check_increments_then_breaks() {
        let budget = RunBudget::unlimited().with_request_limit(2);
        budget.check_pre_request().unwrap();
        budget.check_pre_request().unwrap();
        let err = budget.check_pre_request().unwrap_err();
        match err {
            Error::UsageLimitExceeded(UsageLimitBreach::Requests {
                limit: 2,
                observed: 2,
            }) => {}
            other => panic!("unexpected: {other:?}"),
        }
        // Counter was not incremented past the cap.
        assert_eq!(budget.snapshot().requests, 2);
    }

    #[test]
    fn tool_calls_limit_pre_check_breaks() {
        let budget = RunBudget::unlimited().with_tool_calls_limit(1);
        budget.check_pre_tool_call().unwrap();
        let err = budget.check_pre_tool_call().unwrap_err();
        assert!(matches!(
            err,
            Error::UsageLimitExceeded(UsageLimitBreach::ToolCalls { .. })
        ));
    }

    #[test]
    fn input_tokens_limit_post_observe_breaks() {
        let budget = RunBudget::unlimited().with_input_tokens_limit(100);
        budget.observe_usage(&Usage::new(50, 0)).unwrap();
        let err = budget.observe_usage(&Usage::new(60, 0)).unwrap_err();
        match err {
            Error::UsageLimitExceeded(UsageLimitBreach::InputTokens {
                limit: 100,
                observed: 110,
            }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn output_tokens_limit_post_observe_breaks() {
        let budget = RunBudget::unlimited().with_output_tokens_limit(100);
        budget.observe_usage(&Usage::new(0, 99)).unwrap();
        let err = budget.observe_usage(&Usage::new(0, 2)).unwrap_err();
        assert!(matches!(
            err,
            Error::UsageLimitExceeded(UsageLimitBreach::OutputTokens { .. })
        ));
    }

    #[test]
    fn total_tokens_limit_combines_input_and_output() {
        let budget = RunBudget::unlimited().with_total_tokens_limit(100);
        budget.observe_usage(&Usage::new(40, 40)).unwrap();
        let err = budget.observe_usage(&Usage::new(20, 20)).unwrap_err();
        match err {
            Error::UsageLimitExceeded(UsageLimitBreach::TotalTokens {
                limit: 100,
                observed: 120,
            }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn cost_usd_limit_post_observe_breaks() {
        use rust_decimal::Decimal;
        let cap = Decimal::new(50, 2); // $0.50
        let budget = RunBudget::unlimited().with_cost_limit_usd(cap);
        budget.observe_cost(Decimal::new(30, 2)).unwrap(); // $0.30
        let err = budget.observe_cost(Decimal::new(25, 2)).unwrap_err();
        match err {
            Error::UsageLimitExceeded(UsageLimitBreach::CostUsd { limit, observed }) => {
                assert_eq!(limit, cap);
                assert_eq!(observed, Decimal::new(55, 2)); // $0.55
            }
            other => panic!("unexpected: {other:?}"),
        }
        assert_eq!(budget.snapshot().cost_usd, Decimal::new(55, 2));
    }

    #[test]
    fn cost_unlimited_accumulates_without_breaching() {
        use rust_decimal::Decimal;
        let budget = RunBudget::unlimited();
        budget.observe_cost(Decimal::new(100, 2)).unwrap(); // $1.00
        budget.observe_cost(Decimal::new(200, 2)).unwrap(); // $2.00
        assert_eq!(budget.snapshot().cost_usd, Decimal::new(300, 2));
    }

    #[test]
    fn clone_shares_atomic_state() {
        // Sub-agent fan-out invariant: cloning the budget shares
        // the underlying counters via Arc — the parent's budget
        // and the sub-agent's budget are the same logical run.
        let parent = RunBudget::unlimited().with_request_limit(2);
        let child = parent.clone();
        parent.check_pre_request().unwrap();
        child.check_pre_request().unwrap();
        // Both views see two pre-checks; the third on either side
        // breaches.
        let err = parent.check_pre_request().unwrap_err();
        assert!(matches!(
            err,
            Error::UsageLimitExceeded(UsageLimitBreach::Requests { .. })
        ));
    }

    #[test]
    fn cost_clone_shares_arc_state() {
        // Sub-agent fan-out for cost — parent + child share the
        // same `Mutex<Decimal>` accumulator via `Arc`. A cost
        // observation on either side accumulates into the
        // single logical-run total, and the cap fires from
        // whichever side pushes it over (audit gap noted in
        // post-S104 review).
        use rust_decimal::Decimal;
        let cap = Decimal::new(100, 2); // $1.00
        let parent = RunBudget::unlimited().with_cost_limit_usd(cap);
        let child = parent.clone();
        parent.observe_cost(Decimal::new(60, 2)).unwrap(); // $0.60
        child.observe_cost(Decimal::new(30, 2)).unwrap(); // $0.30 — total $0.90, under
        let err = child.observe_cost(Decimal::new(20, 2)).unwrap_err();
        match err {
            Error::UsageLimitExceeded(UsageLimitBreach::CostUsd { limit, observed }) => {
                assert_eq!(limit, cap);
                // $0.60 + $0.30 + $0.20 = $1.10
                assert_eq!(observed, Decimal::new(110, 2));
            }
            other => panic!("unexpected: {other:?}"),
        }
        assert_eq!(parent.snapshot().cost_usd, Decimal::new(110, 2));
        assert_eq!(child.snapshot().cost_usd, Decimal::new(110, 2));
    }

    #[test]
    fn snapshot_returns_owned_values() {
        // `with_request_limit` so the pre-call CAS actually
        // increments — `check_pre_request` early-returns when no
        // cap is set (`unlimited` alone leaves every counter at
        // zero, hiding the snapshot's frozen-at-call contract).
        let budget = RunBudget::unlimited().with_request_limit(100);
        budget.check_pre_request().unwrap();
        budget.observe_usage(&Usage::new(10, 5)).unwrap();
        let snap = budget.snapshot();
        assert_eq!(snap.requests, 1);
        assert_eq!(snap.input_tokens, 10);
        assert_eq!(snap.output_tokens, 5);
        assert_eq!(snap.total_tokens(), 15);
        // Subsequent mutations don't reflect on the snapshot.
        budget.check_pre_request().unwrap();
        assert_eq!(snap.requests, 1);
    }
}
