//! `PolicyLayer` — `tower::Layer<S>` middleware that fires the
//! per-tenant policy stack (PII redactor + quota gate + cost meter)
//! around every model and tool call.
//!
//! One layer struct, two `Service` impls — one for
//! [`ModelInvocation`] and one for [`ToolInvocation`]. Compose via
//! `ChatModel::layer(PolicyLayer::new(manager))` for model calls
//! and `ToolRegistry::layer(PolicyLayer::new(manager))` for tool
//! dispatch. The same `PolicyRegistry` backs both.
//!
//! ## Lifecycle (model calls)
//!
//! - **before inner.call**: PII `redact_request`, then quota
//!   pre-check (rate + budget). A pre-check refusal short-circuits
//!   before encode, surfacing as `Error::Provider { status: 429 |
//!   402, ... }` per [`PolicyError`]'s `From` impl.
//! - **after inner.call**: PII `redact_response`, then transactional
//!   cost `charge`. F4 — charge fires only when the inner call
//!   succeeded.
//!
//! ## Lifecycle (tool calls)
//!
//! - **before inner.call**: PII `redact_tool_input` walks the JSON
//!   `input` and scrubs string leaves. Quota / cost meter don't
//!   apply to tool calls — those are model-call concepts.
//! - **after inner.call**: PII `redact_tool_output` on the JSON
//!   response.

use std::sync::Arc;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use serde_json::Value;
use tower::{Layer, Service, ServiceExt};

use entelix_core::error::{Error, Result};
use entelix_core::ir::ModelResponse;
use entelix_core::service::{
    ModelInvocation, ModelStream, StreamingModelInvocation, ToolInvocation,
};

use crate::error::PolicyError;
use crate::tenant::PolicyRegistry;

/// Layer that wraps an inner service with per-tenant policy
/// enforcement.
#[derive(Clone)]
pub struct PolicyLayer {
    manager: Arc<PolicyRegistry>,
    /// Rate-limit tokens to acquire on each request. Most callers
    /// use `1` — one model call = one bucket draw.
    rate_tokens_per_request: u32,
}

impl PolicyLayer {
    /// Build with the supplied manager; one rate-limit token per
    /// request.
    #[must_use]
    pub fn new(manager: Arc<PolicyRegistry>) -> Self {
        Self {
            manager,
            rate_tokens_per_request: 1,
        }
    }

    /// Override how many rate-limit tokens each request costs.
    #[must_use]
    pub const fn with_rate_tokens(mut self, tokens: u32) -> Self {
        self.rate_tokens_per_request = tokens;
        self
    }
}

impl std::fmt::Debug for PolicyLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyLayer")
            .field("tenants", &self.manager.tenant_count())
            .field("rate_tokens_per_request", &self.rate_tokens_per_request)
            .finish()
    }
}

impl<S> Layer<S> for PolicyLayer {
    type Service = PolicyService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        PolicyService {
            inner,
            manager: Arc::clone(&self.manager),
            rate_tokens_per_request: self.rate_tokens_per_request,
        }
    }
}

/// `Service` produced by [`PolicyLayer`]. Generic over the inner
/// service type; specialised `Service<ModelInvocation>` and
/// `Service<ToolInvocation>` impls below.
#[derive(Clone)]
pub struct PolicyService<S> {
    inner: S,
    manager: Arc<PolicyRegistry>,
    rate_tokens_per_request: u32,
}

impl<S> Service<ModelInvocation> for PolicyService<S>
where
    S: Service<ModelInvocation, Response = ModelResponse, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = ModelResponse;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut invocation: ModelInvocation) -> Self::Future {
        let manager = Arc::clone(&self.manager);
        let inner = self.inner.clone();
        let tokens = self.rate_tokens_per_request;
        Box::pin(async move {
            let policy = manager.policy_for(invocation.ctx.tenant_id());

            // Pre — redact then quota.
            if let Some(redactor) = &policy.redactor {
                redactor
                    .redact_request(&mut invocation.request)
                    .await
                    .map_err(Error::from)?;
            }
            if let Some(quota) = &policy.quota {
                quota
                    .check_pre_request(invocation.ctx.tenant_id(), tokens)
                    .await
                    .map_err(Error::from)?;
            }

            let tenant = invocation.ctx.tenant_id().to_owned();
            let mut response = inner.oneshot(invocation).await?;

            // Post — redact then charge.
            if let Some(redactor) = &policy.redactor {
                redactor
                    .redact_response(&mut response)
                    .await
                    .map_err(Error::from)?;
            }
            if let Some(meter) = &policy.cost_meter {
                match meter.charge(&tenant, &response.model, &response.usage) {
                    Ok(_) => {}
                    Err(PolicyError::UnknownModel(model)) => {
                        tracing::warn!(
                            target: "entelix_policy::layer",
                            tenant = %tenant,
                            %model,
                            "no pricing configured; skipping cost charge"
                        );
                    }
                    Err(e) => return Err(Error::from(e)),
                }
            }
            Ok(response)
        })
    }
}

impl<S> Service<StreamingModelInvocation> for PolicyService<S>
where
    S: Service<StreamingModelInvocation, Response = ModelStream, Error = Error>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
{
    type Response = ModelStream;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelStream>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut invocation: StreamingModelInvocation) -> Self::Future {
        let manager = Arc::clone(&self.manager);
        let inner = self.inner.clone();
        let tokens = self.rate_tokens_per_request;
        Box::pin(async move {
            let policy = manager.policy_for(invocation.ctx().tenant_id());

            // Pre — request-side redactor + quota gate. Streaming
            // PII redaction on individual deltas is intentionally
            // left out at 1.0 (chunk boundaries can split a PII
            // pattern; streaming-aware redactors are post-1.0
            // work). The request-side redactor still applies —
            // operator-supplied input is fully formed before the
            // first byte streams.
            if let Some(redactor) = &policy.redactor {
                redactor
                    .redact_request(&mut invocation.inner.request)
                    .await
                    .map_err(Error::from)?;
            }
            if let Some(quota) = &policy.quota {
                quota
                    .check_pre_request(invocation.ctx().tenant_id(), tokens)
                    .await
                    .map_err(Error::from)?;
            }

            let tenant = invocation.ctx().tenant_id().clone();
            let model_stream = inner.oneshot(invocation).await?;
            let ModelStream { stream, completion } = model_stream;

            // Wrap completion: charge cost on `Ok` branch only —
            // mirrors the one-shot `ModelInvocation` post-call
            // path. A stream that errors mid-flight resolves
            // `completion` to `Err` and skips the charge entirely
            // (invariant 12 — no phantom cost on partial streams).
            let cost_meter = policy.cost_meter.clone();
            let user_facing = async move {
                let result = completion.await;
                if let Ok(response) = &result
                    && let Some(meter) = &cost_meter
                {
                    match meter.charge(&tenant, &response.model, &response.usage) {
                        Ok(_) => {}
                        Err(PolicyError::UnknownModel(model)) => {
                            tracing::warn!(
                                target: "entelix_policy::layer",
                                tenant = %tenant,
                                %model,
                                "no pricing configured; skipping cost charge"
                            );
                        }
                        Err(e) => return Err(Error::from(e)),
                    }
                }
                result
            };
            Ok(ModelStream {
                stream,
                completion: Box::pin(user_facing),
            })
        })
    }
}

impl<S> Service<ToolInvocation> for PolicyService<S>
where
    S: Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut invocation: ToolInvocation) -> Self::Future {
        let manager = Arc::clone(&self.manager);
        let inner = self.inner.clone();
        Box::pin(async move {
            let policy = manager.policy_for(invocation.ctx.tenant_id());
            if let Some(redactor) = &policy.redactor {
                redactor
                    .redact_json(&mut invocation.input)
                    .await
                    .map_err(Error::from)?;
            }
            let mut output = inner.oneshot(invocation).await?;
            if let Some(redactor) = &policy.redactor {
                redactor
                    .redact_json(&mut output)
                    .await
                    .map_err(Error::from)?;
            }
            Ok(output)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use entelix_core::TenantId;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::task::Context as TaskContext;

    use entelix_core::context::ExecutionContext;
    use entelix_core::ir::{ContentPart, Message, ModelRequest, StopReason, Usage};
    use rust_decimal::Decimal;
    use serde_json::json;

    use super::*;
    use crate::cost::{CostMeter, ModelPricing, PricingTable};
    use crate::pii::RegexRedactor;
    use crate::quota::{Budget, QuotaLimiter};
    use crate::rate_limit::TokenBucketLimiter;
    use crate::tenant::TenantPolicy;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    /// Trivial leaf service that returns a fixed response and counts calls.
    #[derive(Clone)]
    struct FakeModelService {
        calls: Arc<AtomicU32>,
        canned: ModelResponse,
    }

    impl FakeModelService {
        fn new(canned: ModelResponse) -> Self {
            Self {
                calls: Arc::new(AtomicU32::new(0)),
                canned,
            }
        }
    }

    impl Service<ModelInvocation> for FakeModelService {
        type Response = ModelResponse;
        type Error = Error;
        type Future = BoxFuture<'static, Result<ModelResponse>>;

        fn poll_ready(&mut self, _: &mut TaskContext<'_>) -> Poll<Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, _inv: ModelInvocation) -> Self::Future {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let canned = self.canned.clone();
            Box::pin(async move { Ok(canned) })
        }
    }

    fn make_request() -> ModelRequest {
        ModelRequest {
            model: "claude-opus-4-7".into(),
            messages: vec![Message::user("contact user@acme.io for help")],
            ..ModelRequest::default()
        }
    }

    fn make_response() -> ModelResponse {
        ModelResponse {
            id: "r1".into(),
            model: "claude-opus-4-7".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text("ack")],
            usage: Usage::new(1000, 1000),
            rate_limit: None,
            warnings: Vec::new(),
        }
    }

    fn pricing() -> PricingTable {
        PricingTable::new().add_model_pricing(
            "claude-opus-4-7",
            ModelPricing::new(d("15"), d("75"), d("1.5"), d("18.75")),
        )
    }

    #[tokio::test]
    async fn model_layer_redacts_request_then_charges_on_success() {
        let meter = Arc::new(CostMeter::new(pricing()));
        let mgr = Arc::new(
            PolicyRegistry::new().with_tenant(
                TenantId::new("acme"),
                TenantPolicy::builder()
                    .with_redactor(Arc::new(RegexRedactor::with_defaults()))
                    .with_cost_meter(meter.clone())
                    .build()
                    .unwrap(),
            ),
        );
        let leaf = FakeModelService::new(make_response());
        let calls = leaf.calls.clone();
        let layer = PolicyLayer::new(mgr);
        let service = layer.layer(leaf);

        let invocation = ModelInvocation::new(
            make_request(),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let resp = tower::ServiceExt::oneshot(service, invocation)
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // 1000*15/1000 + 1000*75/1000 = 90
        assert_eq!(meter.spent_by(&TenantId::new("acme")), d("90"));
        assert_eq!(resp.id, "r1");
    }

    #[tokio::test]
    async fn rate_refusal_returns_provider_429_and_skips_inner() {
        let mgr = Arc::new(
            PolicyRegistry::new().with_tenant(
                TenantId::new("acme"),
                TenantPolicy::builder()
                    .with_quota(Arc::new(QuotaLimiter::new(
                        Some(Arc::new(TokenBucketLimiter::new(1, 1.0).unwrap())),
                        None,
                        Budget::unlimited(),
                    )))
                    .build()
                    .unwrap(),
            ),
        );
        let leaf = FakeModelService::new(make_response());
        let calls = leaf.calls.clone();
        let layer = PolicyLayer::new(mgr);

        // First call drains the single token.
        let svc1 = layer.layer(leaf.clone());
        let _ = tower::ServiceExt::oneshot(
            svc1,
            ModelInvocation::new(
                make_request(),
                ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
            ),
        )
        .await
        .unwrap();
        // Second call refused.
        let svc2 = layer.layer(leaf);
        let err = tower::ServiceExt::oneshot(
            svc2,
            ModelInvocation::new(
                make_request(),
                ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
            ),
        )
        .await
        .unwrap_err();
        match err {
            Error::Provider { kind, .. } => {
                assert_eq!(kind, entelix_core::ProviderErrorKind::Http(429));
            }
            other => panic!("expected Provider 429, got {other:?}"),
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "inner must not run on refusal"
        );
    }

    /// Tool-side leaf service: echoes its input.
    #[derive(Clone)]
    struct EchoToolService;

    impl Service<ToolInvocation> for EchoToolService {
        type Response = serde_json::Value;
        type Error = Error;
        type Future = BoxFuture<'static, Result<serde_json::Value>>;

        fn poll_ready(&mut self, _: &mut TaskContext<'_>) -> Poll<Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, inv: ToolInvocation) -> Self::Future {
            Box::pin(async move { Ok(inv.input) })
        }
    }

    #[tokio::test]
    async fn tool_layer_redacts_input_and_output() {
        let mgr = Arc::new(
            PolicyRegistry::new().with_tenant(
                TenantId::new("acme"),
                TenantPolicy::builder()
                    .with_redactor(Arc::new(RegexRedactor::with_defaults()))
                    .build()
                    .unwrap(),
            ),
        );
        let layer = PolicyLayer::new(mgr);
        let svc = layer.layer(EchoToolService);
        let inv = ToolInvocation::new(
            "tool_use_1".into(),
            std::sync::Arc::new(entelix_core::tools::ToolMetadata::function(
                "lookup",
                "look up a record",
                json!({"type": "object"}),
            )),
            json!({"email": "user@acme.io"}),
            ExecutionContext::new().with_tenant_id(TenantId::new("acme")),
        );
        let out = tower::ServiceExt::oneshot(svc, inv).await.unwrap();
        // Both directions redacted: input redacted before echo;
        // echoed output redacted again on the way back.
        let txt = out["email"].as_str().unwrap();
        assert!(txt.contains("[REDACTED:email]"), "{txt}");
    }
}
