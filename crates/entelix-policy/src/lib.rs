//! # entelix-policy
//!
//! Multi-tenant operational primitives that `LangChain` / `LangGraph` leave to
//! the host: token-bucket rate limiting, bidirectional PII redaction
//! (F5 mitigation), `rust_decimal`-backed transactional cost accounting
//! (F4 mitigation), composite quota enforcement, and the per-tenant
//! aggregate ([`TenantPolicy`]) plus the runtime registry
//! ([`PolicyRegistry`]) that indexes them by `tenant_id`.
//!
//! ## Surface in one screen
//!
//! - [`RateLimiter`] (trait) + [`TokenBucketLimiter`] — async,
//!   per-key, time-injectable for deterministic tests.
//! - [`PiiRedactor`] (trait) + [`RegexRedactor`] — runs on both
//!   `pre_request` and `post_response` so leaks can't slip past
//!   in either direction.
//! - [`CostMeter`] + [`PricingTable`] / [`ModelPricing`] — `rust_decimal`
//!   for float-free arithmetic; charges are recorded only after the
//!   response decoder succeeds (transactional — F4).
//! - [`QuotaLimiter`] — composite: rate (RPS) + budget ceiling
//!   (per-tenant cumulative spend cap).
//! - [`TenantPolicy`] — per-tenant aggregate of optional handles to
//!   the four primitives above.
//! - [`PolicyRegistry`] — `DashMap<tenant_id, Arc<TenantPolicy>>`
//!   with a fallback default policy.
//! - [`PolicyLayer`] — `tower::Layer<S>` that wires every primitive
//!   into both `Service<ModelInvocation>` and `Service<ToolInvocation>`
//!   pipelines. Compose via `ChatModel::layer(PolicyLayer::new(mgr))`
//!   for model calls and `ToolRegistry::layer(PolicyLayer::new(mgr))`
//!   for tool calls — same struct on both sides.
//!
//! ## Layer lifecycle (model calls)
//!
//! - **before inner.call** (`Service<ModelInvocation>`):
//!   1. `PiiRedactor::redact_request` — outbound scrub.
//!   2. `QuotaLimiter::check_pre_request` — rate + budget gate.
//!      Returns `Error::Provider { status: 429 | 402, ... }` on refusal.
//! - **after inner.call**:
//!   1. `PiiRedactor::redact_response` — inbound scrub.
//!   2. `CostMeter::charge` — transactional charge (F4 — only here,
//!      after a successful inner call).
//!
//! ## Layer lifecycle (tool calls)
//!
//! - **before inner.call** (`Service<ToolInvocation>`):
//!   1. `PiiRedactor::redact_json(input)` — scrub tool input JSON.
//! - **after inner.call**:
//!   1. `PiiRedactor::redact_json(output)` — scrub tool output JSON.
//!
//! ## Tenant scoping
//!
//! Every primitive looks up state by `ExecutionContext::tenant_id()`
//! (ADR-0017). A request without an explicit tenant uses the
//! [`entelix_core::context::DEFAULT_TENANT_ID`] scope; the default
//! tenant gets the [`PolicyRegistry`]'s default policy (typically
//! "no policy" — pass-through).

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-policy/1.0.0-rc.1")]
#![deny(missing_docs)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_const_for_fn
)]

mod cost;
mod error;
mod layer;
mod pii;
mod quota;
mod rate_limit;
mod tenant;

pub use cost::{
    CostMeter, DEFAULT_MAX_TENANTS, MAX_WARNED_MODELS, ModelPricing, PricingTable,
    UnknownModelPolicy,
};
pub use error::{PolicyError, PolicyResult};
pub use layer::{PolicyLayer, PolicyService};
pub use pii::{PiiPattern, PiiRedactor, RegexRedactor, default_pii_patterns, luhn_valid};
pub use quota::{Budget, QuotaLimiter};
pub use rate_limit::{Clock, RateLimiter, SystemClock, TokenBucketLimiter};
pub use tenant::{PolicyRegistry, TenantPolicy, TenantPolicyBuilder};
