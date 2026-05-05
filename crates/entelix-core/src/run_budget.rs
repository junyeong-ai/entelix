//! `RunBudget` — five-axis usage cap checked across one logical
//! run, including sub-agent fan-out.
//!
//! Mirrors pydantic-ai 1.90's `UsageLimits` shape (Fork 1 audit
//! synthesis on `project_entelix_2026_05_05_master_plan.md`):
//!
//! | Axis                    | Type | Pre-call check | Post-call accumulation |
//! |-------------------------|------|----------------|------------------------|
//! | `request_limit`         | u32  | ✓              | accumulate on Ok       |
//! | `input_tokens_limit`    | u64  | —              | check on Ok            |
//! | `output_tokens_limit`   | u64  | —              | check on Ok            |
//! | `total_tokens_limit`    | u64  | —              | check on Ok            |
//! | `tool_calls_limit`      | u32  | ✓              | accumulate on Ok       |
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
//! and observed value. ADR-0080.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::ir::Usage;

/// Which [`RunBudget`] axis surfaced a breach.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UsageLimitAxis {
    /// `request_limit` — number of model dispatches per run.
    Requests,
    /// `input_tokens_limit` — cumulative input tokens across the run.
    InputTokens,
    /// `output_tokens_limit` — cumulative output tokens.
    OutputTokens,
    /// `total_tokens_limit` — cumulative input + output tokens.
    TotalTokens,
    /// `tool_calls_limit` — number of tool dispatches per run.
    ToolCalls,
}

impl std::fmt::Display for UsageLimitAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Requests => "requests",
            Self::InputTokens => "input_tokens",
            Self::OutputTokens => "output_tokens",
            Self::TotalTokens => "total_tokens",
            Self::ToolCalls => "tool_calls",
        };
        f.write_str(s)
    }
}

/// Five-axis usage cap shared across one logical run (parent
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
    state: Arc<RunBudgetState>,
}

#[derive(Debug, Default)]
struct RunBudgetState {
    requests: AtomicU32,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    tool_calls: AtomicU32,
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

    /// Pre-request gate — checks the request-count cap and, on
    /// success, increments the request counter. Call from the
    /// dispatch site **before** the wire roundtrip. Returns
    /// [`Error::UsageLimitExceeded`] when the cap is hit; the
    /// counter is not incremented on failure (the request did
    /// not actually fire).
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
                    return Err(Error::UsageLimitExceeded {
                        axis: UsageLimitAxis::Requests,
                        limit: u64::from(limit),
                        observed: u64::from(current),
                    });
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
                    return Err(Error::UsageLimitExceeded {
                        axis: UsageLimitAxis::ToolCalls,
                        limit: u64::from(limit),
                        observed: u64::from(current),
                    });
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
            return Err(Error::UsageLimitExceeded {
                axis: UsageLimitAxis::InputTokens,
                limit,
                observed: new_in,
            });
        }
        if let Some(limit) = self.output_tokens_limit
            && new_out > limit
        {
            return Err(Error::UsageLimitExceeded {
                axis: UsageLimitAxis::OutputTokens,
                limit,
                observed: new_out,
            });
        }
        if let Some(limit) = self.total_tokens_limit {
            let total = new_in.saturating_add(new_out);
            if total > limit {
                return Err(Error::UsageLimitExceeded {
                    axis: UsageLimitAxis::TotalTokens,
                    limit,
                    observed: total,
                });
            }
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
        UsageSnapshot {
            requests: self.state.requests.load(Ordering::Acquire),
            input_tokens: self.state.input_tokens.load(Ordering::Acquire),
            output_tokens: self.state.output_tokens.load(Ordering::Acquire),
            tool_calls: self.state.tool_calls.load(Ordering::Acquire),
        }
    }
}

/// Frozen snapshot of [`RunBudget`] counters at one point in
/// time. Carried in `AgentRunResult<S>::usage` (B-5) so callers
/// see the final tally without needing to clone the budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
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
        assert!(matches!(
            err,
            Error::UsageLimitExceeded {
                axis: UsageLimitAxis::Requests,
                limit: 2,
                observed: 2,
            }
        ));
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
            Error::UsageLimitExceeded {
                axis: UsageLimitAxis::ToolCalls,
                ..
            }
        ));
    }

    #[test]
    fn input_tokens_limit_post_observe_breaks() {
        let budget = RunBudget::unlimited().with_input_tokens_limit(100);
        budget.observe_usage(&Usage::new(50, 0)).unwrap();
        let err = budget.observe_usage(&Usage::new(60, 0)).unwrap_err();
        assert!(matches!(
            err,
            Error::UsageLimitExceeded {
                axis: UsageLimitAxis::InputTokens,
                limit: 100,
                observed: 110,
            }
        ));
    }

    #[test]
    fn output_tokens_limit_post_observe_breaks() {
        let budget = RunBudget::unlimited().with_output_tokens_limit(100);
        budget.observe_usage(&Usage::new(0, 99)).unwrap();
        let err = budget.observe_usage(&Usage::new(0, 2)).unwrap_err();
        assert!(matches!(
            err,
            Error::UsageLimitExceeded {
                axis: UsageLimitAxis::OutputTokens,
                ..
            }
        ));
    }

    #[test]
    fn total_tokens_limit_combines_input_and_output() {
        let budget = RunBudget::unlimited().with_total_tokens_limit(100);
        budget.observe_usage(&Usage::new(40, 40)).unwrap();
        let err = budget.observe_usage(&Usage::new(20, 20)).unwrap_err();
        assert!(matches!(
            err,
            Error::UsageLimitExceeded {
                axis: UsageLimitAxis::TotalTokens,
                limit: 100,
                observed: 120,
            }
        ));
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
            Error::UsageLimitExceeded {
                axis: UsageLimitAxis::Requests,
                ..
            }
        ));
    }

    #[test]
    fn snapshot_returns_owned_values() {
        let budget = RunBudget::unlimited();
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
