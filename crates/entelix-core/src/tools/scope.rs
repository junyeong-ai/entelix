//! `ToolDispatchScope` + `ScopedToolLayer` — operator-supplied hook
//! that wraps every tool dispatch future to inject ambient
//! request-scope state (tokio task-locals, tracing scopes, RLS
//! `SET LOCAL` settings, etc.).
//!
//! ## Why a hook
//!
//! Some tool implementations read state that the SDK cannot supply
//! through `ExecutionContext` directly because the state lives in a
//! thread-local or task-local rather than in a typed field. The
//! canonical example is Postgres row-level security: the tool's
//! query path reads `current_setting('entelix.tenant_id', true)`,
//! which is `SET LOCAL`-scoped to the current transaction. The SDK
//! must enter that scope before the tool's future starts polling
//! and must restore the prior scope when the future resolves.
//!
//! Operators implement [`ToolDispatchScope::wrap`] with the
//! enter/exit machinery their backend requires (e.g.
//! `tokio::task_local!::scope(value, fut)`), wrap the trait impl in
//! a [`ScopedToolLayer`], and attach it to a `ToolRegistry` via
//! [`crate::tools::ToolRegistry::layer`]. Sub-agents that narrow
//! the parent registry through `restricted_to` / `filter` inherit
//! the layer stack by `Arc` (ADR-0035) — the wrap fires for every
//! sub-agent dispatch automatically.
//!
//! ## Composition with other layers
//!
//! `ScopedToolLayer` is one more Tower middleware in the registry's
//! layer stack. The reasonable layering for the canonical entelix
//! middlewares — observability outermost, scope innermost —
//! corresponds to *registering* in inside-out order, since
//! [`crate::tools::ToolRegistry::layer`] makes the
//! **last-registered layer outermost**:
//!
//! ```ignore
//! ToolRegistry::new()
//!     .layer(ScopedToolLayer::new(my_scope))   // innermost (registered first)
//!     .layer(ApprovalLayer::new(approver))     //
//!     .layer(PolicyLayer::new(...))            //
//!     .layer(OtelLayer::new(...))              // outermost (registered last)
//!     .register(my_tool)?
//! ```
//!
//! Dispatch flow on each tool call: `OtelLayer → PolicyLayer →
//! ApprovalLayer → ScopedToolLayer → Tool::execute`. The scope is
//! active during the leaf `Tool::execute` future and through any
//! `?`-returning code on the way back; PII/cost middleware run
//! before the scope is entered. Operators that need the scope
//! active during PII redaction reverse the registration order
//! (move `.layer(ScopedToolLayer::new(...))` later in the chain).

use std::sync::Arc;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use serde_json::Value;
use tower::{Layer, Service};

use crate::context::ExecutionContext;
use crate::error::{Error, Result};
use crate::service::ToolInvocation;

/// Operator-supplied wrapper for tool-dispatch futures.
///
/// Implementations enter ambient scope state (task-locals,
/// tracing scopes, RLS settings) before the wrapped future starts
/// polling and restore the prior state when it resolves. The
/// trait is object-safe so concrete impls plug into
/// [`ScopedToolLayer`] behind `Arc<dyn ToolDispatchScope>`.
///
/// The wrap method takes `ctx` by value (the field is cheaply
/// `Clone` — `Arc<str>` for `tenant_id`, refcounted handles for
/// extensions). Implementations that need only one field
/// (typically `tenant_id`) read it once and discard the ctx; the
/// owned shape avoids lifetime gymnastics in the trait signature
/// and keeps the trait object-safe.
pub trait ToolDispatchScope: Send + Sync + 'static {
    /// Wrap `fut` to observe the scope's ambient state.
    /// `ctx` carries the current request-scope state (tenant id,
    /// thread id, extensions) the wrapper may consult to seed its
    /// task-locals.
    fn wrap(
        &self,
        ctx: ExecutionContext,
        fut: BoxFuture<'static, Result<Value>>,
    ) -> BoxFuture<'static, Result<Value>>;
}

/// `tower::Layer<S>` that wraps a `Service<ToolInvocation, Response =
/// Value, Error = Error>` so its `call` future is wrapped by the
/// operator-supplied [`ToolDispatchScope`].
///
/// Attach via [`crate::tools::ToolRegistry::layer`]. Cloning is
/// cheap (the wrapper is held behind `Arc`).
pub struct ScopedToolLayer {
    wrapper: Arc<dyn ToolDispatchScope>,
}

impl ScopedToolLayer {
    /// Wrap a concrete [`ToolDispatchScope`] for layer attachment.
    /// The boxed-trait shape lets operators stack heterogeneous
    /// scope wrappers if they need to compose (e.g. a tenant-RLS
    /// scope outside a tracing-baggage scope).
    pub fn new<W>(wrapper: W) -> Self
    where
        W: ToolDispatchScope,
    {
        Self {
            wrapper: Arc::new(wrapper),
        }
    }

    /// Wrap an already-Arc'd [`ToolDispatchScope`]. Convenient when
    /// the same scope handle is shared across multiple registries.
    #[must_use]
    pub fn from_arc(wrapper: Arc<dyn ToolDispatchScope>) -> Self {
        Self { wrapper }
    }
}

impl Clone for ScopedToolLayer {
    fn clone(&self) -> Self {
        Self {
            wrapper: Arc::clone(&self.wrapper),
        }
    }
}

impl<S> Layer<S> for ScopedToolLayer {
    type Service = ScopedToolService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ScopedToolService {
            inner,
            wrapper: Arc::clone(&self.wrapper),
        }
    }
}

/// `tower::Service<ToolInvocation>` wrapper produced by
/// [`ScopedToolLayer::layer`]. Dispatches `inner.call(invocation)`
/// and runs the resulting future under the configured
/// [`ToolDispatchScope::wrap`].
pub struct ScopedToolService<S> {
    inner: S,
    wrapper: Arc<dyn ToolDispatchScope>,
}

impl<S: Clone> Clone for ScopedToolService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            wrapper: Arc::clone(&self.wrapper),
        }
    }
}

impl<S> Service<ToolInvocation> for ScopedToolService<S>
where
    S: Service<ToolInvocation, Response = Value, Error = Error> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, invocation: ToolInvocation) -> Self::Future {
        let ctx = invocation.ctx.clone();
        let inner_fut = self.inner.call(invocation);
        self.wrapper.wrap(ctx, Box::pin(inner_fut))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;
    use crate::tools::{Tool, ToolMetadata, ToolRegistry};
    use async_trait::async_trait;

    struct CountingScope {
        wraps: Arc<AtomicUsize>,
    }

    impl ToolDispatchScope for CountingScope {
        fn wrap(
            &self,
            _ctx: ExecutionContext,
            fut: BoxFuture<'static, Result<Value>>,
        ) -> BoxFuture<'static, Result<Value>> {
            self.wraps.fetch_add(1, Ordering::SeqCst);
            fut
        }
    }

    struct EchoTool {
        metadata: ToolMetadata,
    }

    impl EchoTool {
        fn new() -> Self {
            Self {
                metadata: ToolMetadata::function(
                    "echo",
                    "Echo input verbatim.",
                    json!({ "type": "object" }),
                ),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }

        async fn execute(&self, input: Value, _ctx: &crate::AgentContext<()>) -> Result<Value> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn scope_wrap_fires_on_dispatch() {
        let wraps = Arc::new(AtomicUsize::new(0));
        let scope = CountingScope {
            wraps: Arc::clone(&wraps),
        };
        let registry = ToolRegistry::new()
            .layer(ScopedToolLayer::new(scope))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let ctx = ExecutionContext::new();
        let result = registry
            .dispatch("", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap();
        assert_eq!(result, json!({"x": 1}));
        assert_eq!(wraps.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn scope_wrap_fires_per_dispatch() {
        let wraps = Arc::new(AtomicUsize::new(0));
        let scope = CountingScope {
            wraps: Arc::clone(&wraps),
        };
        let registry = ToolRegistry::new()
            .layer(ScopedToolLayer::new(scope))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let ctx = ExecutionContext::new();
        for _ in 0..3 {
            registry
                .dispatch("", "echo", json!({"x": 1}), &ctx)
                .await
                .unwrap();
        }
        assert_eq!(wraps.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn scope_wrap_inherited_by_narrowed_view() {
        // Sub-agent narrowing pattern (ADR-0035) shares the layer
        // factory by Arc — a scope attached to the parent must
        // fire on every narrowed-view dispatch as well.
        let wraps = Arc::new(AtomicUsize::new(0));
        let scope = CountingScope {
            wraps: Arc::clone(&wraps),
        };
        let parent = ToolRegistry::new()
            .layer(ScopedToolLayer::new(scope))
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let narrowed = parent.restricted_to(&["echo"]).unwrap();
        let ctx = ExecutionContext::new();
        narrowed
            .dispatch("", "echo", json!({"x": 1}), &ctx)
            .await
            .unwrap();
        assert_eq!(wraps.load(Ordering::SeqCst), 1);
    }
}
