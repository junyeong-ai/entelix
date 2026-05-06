//! `ToolRegistry` — append-only collection of `Arc<dyn Tool>` plus
//! a layered `tower::Service` dispatch path.
//!
//! Construction is fluent: `ToolRegistry::new().register(tool).layer(L)`.
//! The registry stores the layer stack as a `BoxCloneService` factory;
//! each `dispatch` call composes the leaf [`InnerToolService`] with
//! the stack and runs it.
//!
//! Layers wrap *every* tool call. `PolicyLayer` (PII redaction,
//! quota) and `OtelLayer` (`gen_ai.tool.*` events) plug in here,
//! closing the tool-hook gap that the previous `Hook`-trait design
//! left open.

use std::collections::HashMap;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use serde_json::Value;
use tower::util::BoxCloneService;
use tower::{Layer, Service, ServiceExt};

use crate::agent_context::AgentContext;
use crate::context::ExecutionContext;
use crate::error::{Error, Result};
use crate::service::{BoxedToolService, ToolInvocation};
use crate::tools::tool::Tool;

/// `tower::Service` leaf that invokes one specific `Tool`. Cloning
/// is cheap (`Arc`-backed). The registry materialises one of these
/// per `dispatch` call (or per-tool, cached) and wraps the layer
/// stack around it.
///
/// The leaf service validates `invocation.input` against the tool's
/// declared `metadata().input_schema` before calling `Tool::execute`,
/// so malformed payloads (regardless of source — direct dispatch,
/// model-driven tool call, MCP) fail fast with
/// [`Error::InvalidRequest`] and never reach user-supplied tool code.
/// Internal leaf service — wraps one [`Tool<D>`] handle with a
/// compiled JSON-Schema validator and a clone of the registry's
/// typed deps `D`. Layer impls operate on the D-erased
/// [`BoxedToolService`] boundary; only the leaf handles `D`.
pub(super) struct InnerToolService<D> {
    tool: Arc<dyn Tool<D>>,
    deps: D,
    validator: Arc<jsonschema::Validator>,
}

impl<D: Clone> Clone for InnerToolService<D> {
    fn clone(&self) -> Self {
        Self {
            tool: Arc::clone(&self.tool),
            deps: self.deps.clone(),
            validator: Arc::clone(&self.validator),
        }
    }
}

impl<D: Send + Sync + 'static> InnerToolService<D> {
    /// Build from a shared tool handle and a clone of the
    /// registry-held deps. Compiles the input schema once at
    /// construction; subsequent dispatches reuse the compiled
    /// validator.
    pub(super) fn new(tool: Arc<dyn Tool<D>>, deps: D) -> Self {
        // Schema compilation failure leaves the validator
        // permissive — a schema bug must not silently break every
        // dispatch. The tool author still gets a best-effort
        // runtime, and the codec layer catches shape violations
        // on the wire. Surface the compilation error via
        // `tracing::warn!` so the bug isn't silently masked.
        let metadata = tool.metadata();
        let validator = jsonschema::options()
            .build(&metadata.input_schema)
            .unwrap_or_else(|err| {
                tracing::warn!(
                    target: "entelix.tools",
                    tool = %metadata.name,
                    error = %err,
                    "tool input schema failed to compile; falling back to a permissive validator"
                );
                permissive_validator()
            });
        Self {
            tool,
            deps,
            validator: Arc::new(validator),
        }
    }
}

fn permissive_validator() -> jsonschema::Validator {
    let permissive = serde_json::json!({});
    jsonschema::options()
        .build(&permissive)
        .unwrap_or_else(|_| unreachable!("empty object schema is always valid"))
}

impl<D: Send + Sync + 'static> std::fmt::Debug for InnerToolService<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerToolService")
            .field("tool", &self.tool.metadata().name)
            .finish()
    }
}

impl<D: Clone + Send + Sync + 'static> Service<ToolInvocation> for InnerToolService<D> {
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    #[inline]
    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, invocation: ToolInvocation) -> Self::Future {
        let tool = Arc::clone(&self.tool);
        let deps = self.deps.clone();
        let validator = Arc::clone(&self.validator);
        Box::pin(async move {
            if let Err(err) = validator.validate(&invocation.input) {
                return Err(Error::invalid_request(format!(
                    "tool '{}' input failed schema validation at '{}': {}",
                    tool.metadata().name,
                    err.instance_path(),
                    err
                )));
            }
            // Build `AgentContext<D>` from the invocation's infra
            // context plus the registry-held deps. Tower layers
            // upstream see only `ToolInvocation` (D-erased); D
            // surfaces solely at the leaf `Tool::execute` boundary.
            let agent_ctx = AgentContext::new(invocation.ctx, deps);
            tool.execute(invocation.input, &agent_ctx).await
        })
    }
}

/// Boxed factory that takes the leaf [`InnerToolService<D>`] and
/// returns the layered [`BoxedToolService`]. The factory itself is
/// generic over `D`, but the boxed output is D-erased so layers and
/// the dispatch hot path stay independent of the deps shape.
type LayerFactory<D> = Arc<dyn Fn(InnerToolService<D>) -> BoxedToolService + Send + Sync>;

/// Append-only tool registry with a layered dispatch path.
///
/// `D` defaults to `()`. Deps-less registries (`ToolRegistry`,
/// `ToolRegistry::new()`) work exactly as in slice 100. Operator-
/// typed tools use [`ToolRegistry::with_deps`] to thread typed
/// handles into every `Tool<D>::execute` body via the
/// [`AgentContext<D>`] carrier.
pub struct ToolRegistry<D = ()> {
    by_name: HashMap<String, Arc<dyn Tool<D>>>,
    deps: D,
    /// Wraps the leaf service with the configured layer stack. None
    /// means "no layers" — dispatch returns the inner service
    /// boxed.
    factory: Option<LayerFactory<D>>,
}

impl<D: Clone> Clone for ToolRegistry<D> {
    fn clone(&self) -> Self {
        Self {
            by_name: self.by_name.clone(),
            deps: self.deps.clone(),
            factory: self.factory.clone(),
        }
    }
}

impl Default for ToolRegistry<()> {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry<()> {
    /// Empty deps-less registry, no layers attached. Equivalent to
    /// [`ToolRegistry::with_deps(())`](Self::with_deps).
    #[must_use]
    pub fn new() -> Self {
        Self::with_deps(())
    }
}

impl<D: Clone + Send + Sync + 'static> ToolRegistry<D> {
    /// Empty registry holding `deps`, no layers attached. Every
    /// dispatch through this registry — and every sub-agent narrowed
    /// from it — surfaces `deps` to `Tool<D>::execute` via
    /// [`AgentContext<D>::deps`].
    #[must_use]
    pub fn with_deps(deps: D) -> Self {
        Self {
            by_name: HashMap::new(),
            deps,
            factory: None,
        }
    }

    /// Append a tool. Per ADR-0010, `*Registry` types are
    /// **init-time, append-only**: this method returns
    /// [`Error::Config`] if `tool.metadata().name` collides with an
    /// already-registered tool. Silent overwrite at registry
    /// construction is a configuration bug — surface it instead of
    /// papering over.
    pub fn register(mut self, tool: Arc<dyn Tool<D>>) -> Result<Self> {
        let name = tool.metadata().name.clone();
        if self.by_name.contains_key(&name) {
            return Err(Error::config(format!(
                "ToolRegistry::register: tool '{name}' is already registered \
                 (registry is append-only — see ADR-0010)"
            )));
        }
        self.by_name.insert(name, tool);
        Ok(self)
    }

    /// Append a layer to the dispatch stack. Layers stack around the
    /// leaf [`InnerToolService<D>`] — the **last-registered layer is
    /// outermost** (sees the request first, the response last).
    /// `registry.layer(A).layer(B)` resolves to `B → A → tool`.
    ///
    /// Operators wiring multiple layers attach them in inside-out
    /// order. For the canonical entelix layering — observability
    /// outermost, sandbox-side scope innermost — register
    /// inside-first:
    ///
    /// ```ignore
    /// ToolRegistry::new()
    ///     .layer(ScopedToolLayer::new(my_scope))   // innermost (registered first)
    ///     .layer(ApprovalLayer::new(approver))     //
    ///     .layer(PolicyLayer::new(...))            //
    ///     .layer(OtelLayer::new(...))              // outermost (registered last)
    ///     .register(my_tool)?
    /// ```
    ///
    /// The layer must produce a `Service<ToolInvocation, Response =
    /// Value, Error = Error>`. Layers operate on the D-erased
    /// [`BoxedToolService`] boundary so the same layer types work
    /// across every `D`.
    #[must_use]
    pub fn layer<L>(mut self, layer: L) -> Self
    where
        L: Layer<BoxedToolService> + Clone + Send + Sync + 'static,
        L::Service:
            Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
        <L::Service as Service<ToolInvocation>>::Future: Send + 'static,
    {
        let prev = self.factory.take();
        let layer = layer;
        let new_factory: LayerFactory<D> = Arc::new(move |inner: InnerToolService<D>| {
            let stacked: BoxedToolService = match &prev {
                Some(prev_factory) => prev_factory(inner),
                None => BoxCloneService::new(inner),
            };
            BoxCloneService::new(layer.clone().layer(stacked))
        });
        self.factory = Some(new_factory);
        self
    }

    /// Borrow the registry-held deps. Operators typically thread
    /// these through `Tool<D>::execute` via `ctx.deps()`; reach for
    /// this when an out-of-band consumer (e.g. an observer that
    /// inspects the dispatch path) needs the same handle.
    #[must_use]
    pub const fn deps(&self) -> &D {
        &self.deps
    }

    /// Whether the registry has at least one tool.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Iterate tool names (order unspecified).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    /// Borrow a registered tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool<D>>> {
        self.by_name.get(name)
    }

    /// Return a narrowed view of this registry containing only the
    /// tools matching `predicate`. The full layer stack and the
    /// registry-held deps ride over — `Arc`-shared at no copy cost —
    /// so observability, policy, and retry layers attached at the
    /// parent level apply transparently to dispatches through the
    /// view.
    ///
    /// This is the canonical sub-agent / brain-passes-hand path:
    /// constructing a fresh `ToolRegistry::new()` would silently drop
    /// the layer stack. The shape of the API forecloses that mistake
    /// — operators always start from the parent registry and narrow.
    #[must_use]
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&dyn Tool<D>) -> bool,
    {
        let by_name = self
            .by_name
            .iter()
            .filter(|(_, tool)| predicate(tool.as_ref()))
            .map(|(name, tool)| (name.clone(), Arc::clone(tool)))
            .collect();
        Self {
            by_name,
            deps: self.deps.clone(),
            factory: self.factory.clone(),
        }
    }

    /// Return a narrowed view restricted to the tools named in
    /// `allowed`. Returns [`Error::Config`] when any name is absent
    /// from this registry — silent name-typo drops have caused
    /// production incidents (sub-agents that quietly cannot reach a
    /// tool they were configured to call), so the view fails fast at
    /// the moment the typo lands rather than at the moment the model
    /// emits the call.
    ///
    /// The layer stack rides over (see [`Self::filter`]).
    pub fn restricted_to(&self, allowed: &[&str]) -> Result<Self> {
        let mut missing: Vec<&str> = Vec::new();
        for name in allowed {
            if !self.by_name.contains_key(*name) {
                missing.push(*name);
            }
        }
        if !missing.is_empty() {
            return Err(Error::config(format!(
                "ToolRegistry::restricted_to: tool name(s) not in registry: {}",
                missing.join(", ")
            )));
        }
        let allowed_set: std::collections::HashSet<&str> = allowed.iter().copied().collect();
        Ok(self.filter(|tool| allowed_set.contains(tool.metadata().name.as_str())))
    }

    /// Build a [`BoxedToolService`] for `name` — the leaf
    /// [`InnerToolService<D>`] wrapped in the configured layer
    /// stack. Returns `None` if the tool is not registered.
    #[must_use]
    pub fn service(&self, name: &str) -> Option<BoxedToolService> {
        let tool = self.by_name.get(name)?;
        let inner = InnerToolService::new(Arc::clone(tool), self.deps.clone());
        Some(match &self.factory {
            Some(factory) => factory(inner),
            None => BoxCloneService::new(inner),
        })
    }

    /// Dispatch `name(input)` through the layered stack. The
    /// `tool_use_id` is the IR's stable correlation id (from the
    /// originating `ContentPart::ToolUse::id`) — observability
    /// layers like `ToolEventLayer` use it to match `ToolStart` /
    /// `ToolComplete` / `ToolError` events for the same dispatch.
    /// Pass an empty string when the call has no upstream `ToolUse`
    /// (e.g. recipe-driven direct dispatch); event layers fall back
    /// to `name` in that case.
    ///
    /// `ctx` is the infra context. `D` is supplied from the registry
    /// itself — operators never thread it through `dispatch`
    /// arguments, which keeps the registry surface deps-shape
    /// independent for layer authors.
    ///
    /// Returns `Error::InvalidRequest` if `name` is not registered.
    /// Errors from the leaf tool surface unchanged after the layer
    /// stack processes them.
    pub async fn dispatch(
        &self,
        tool_use_id: &str,
        name: &str,
        input: Value,
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        if let Some(budget) = ctx.run_budget() {
            // Pre-call axes — tool-call cap. The check increments
            // the counter on success so two concurrent dispatches
            // racing on the same `n+1` slot do not both pass.
            budget.check_pre_tool_call()?;
        }
        let tool = self
            .by_name
            .get(name)
            .ok_or_else(|| Error::invalid_request(format!("unknown tool '{name}'")))?;
        let metadata = Arc::new(tool.metadata().clone());
        let service = self
            .service(name)
            .ok_or_else(|| Error::invalid_request(format!("unknown tool '{name}'")))?;
        let invocation = ToolInvocation::new(tool_use_id.to_owned(), metadata, input, ctx.clone());
        service.oneshot(invocation).await
    }
}

impl<D: Send + Sync + 'static> std::fmt::Debug for ToolRegistry<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.by_name.keys().collect::<Vec<_>>())
            .field("layers_attached", &self.factory.is_some())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::tools::metadata::ToolMetadata;

    struct EchoTool {
        metadata: ToolMetadata,
    }

    impl EchoTool {
        fn new() -> Self {
            Self {
                metadata: ToolMetadata::function("echo", "echoes input", json!({"type": "object"})),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }
        async fn execute(&self, input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn dispatch_round_trips_through_inner_service() {
        let reg = ToolRegistry::new()
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let out = reg
            .dispatch("call_1", "echo", json!({"x": 1}), &ExecutionContext::new())
            .await
            .unwrap();
        assert_eq!(out, json!({"x": 1}));
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_returns_invalid_request() {
        let reg = ToolRegistry::new();
        let err = reg
            .dispatch("", "missing", json!({}), &ExecutionContext::new())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("unknown tool 'missing'"));
    }

    #[test]
    fn registry_iterators_and_accessors() {
        let reg = ToolRegistry::new()
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        assert!(reg.get("echo").is_some());
        let names: Vec<_> = reg.names().collect();
        assert_eq!(names, vec!["echo"]);
    }

    #[test]
    fn duplicate_registration_is_a_config_error() {
        let reg = ToolRegistry::new()
            .register(Arc::new(EchoTool::new()))
            .unwrap();
        let err = reg.register(Arc::new(EchoTool::new())).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(format!("{err}").contains("already registered"));
    }

    struct NoopTool {
        metadata: ToolMetadata,
    }

    impl NoopTool {
        fn new(name: &str) -> Self {
            Self {
                metadata: ToolMetadata::function(name, "no-op", json!({"type": "object"})),
            }
        }
    }

    #[derive(Clone)]
    struct LabelLayer {
        label: char,
        log: Arc<std::sync::Mutex<Vec<char>>>,
    }

    #[derive(Clone)]
    struct LabelService<S> {
        inner: S,
        label: char,
        log: Arc<std::sync::Mutex<Vec<char>>>,
    }

    impl<S> Layer<S> for LabelLayer {
        type Service = LabelService<S>;
        fn layer(&self, inner: S) -> Self::Service {
            LabelService {
                inner,
                label: self.label,
                log: Arc::clone(&self.log),
            }
        }
    }

    impl<S> Service<ToolInvocation> for LabelService<S>
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

        fn call(&mut self, inv: ToolInvocation) -> Self::Future {
            let label = self.label;
            let log = Arc::clone(&self.log);
            let mut inner = self.inner.clone();
            Box::pin(async move {
                log.lock().unwrap().push(label);
                inner.call(inv).await
            })
        }
    }

    #[tokio::test]
    async fn layer_order_is_last_registered_outermost() {
        // Verify the dispatch order against the Tower convention.
        // Tracing the impl: `registry.layer(A).layer(B)` builds the
        // factory `|inner| B.layer(A.layer(inner))` — `B` wraps the
        // result of `A.layer(inner)`, so on dispatch `B` sees the
        // request first (B is OUTERMOST). Last-registered wins.
        let log = Arc::new(std::sync::Mutex::new(Vec::<char>::new()));
        let registry = ToolRegistry::new()
            .layer(LabelLayer {
                label: 'A',
                log: Arc::clone(&log),
            })
            .layer(LabelLayer {
                label: 'B',
                log: Arc::clone(&log),
            })
            .layer(LabelLayer {
                label: 'C',
                log: Arc::clone(&log),
            })
            .register(Arc::new(EchoTool::new()))
            .unwrap();

        registry
            .dispatch("", "echo", json!({"x": 1}), &ExecutionContext::new())
            .await
            .unwrap();

        let order: String = log.lock().unwrap().iter().collect();
        assert_eq!(
            order, "CBA",
            "last-registered layer must run first (outermost). \
             Operators wiring `OtelLayer` outermost should `.layer(OtelLayer)` \
             last in the chain."
        );
    }

    #[async_trait]
    impl Tool for NoopTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }
        async fn execute(&self, _input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
            Ok(json!(null))
        }
    }

    #[test]
    fn filter_view_narrows_by_name_set() {
        let reg = ToolRegistry::new()
            .register(Arc::new(NoopTool::new("alpha")))
            .unwrap()
            .register(Arc::new(NoopTool::new("beta")))
            .unwrap()
            .register(Arc::new(NoopTool::new("gamma")))
            .unwrap();
        let view = reg.filter(|t| t.metadata().name != "beta");
        assert_eq!(view.len(), 2);
        assert!(view.get("alpha").is_some());
        assert!(view.get("gamma").is_some());
        assert!(view.get("beta").is_none());
        // Parent untouched.
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn restricted_to_succeeds_for_known_names() {
        let reg = ToolRegistry::new()
            .register(Arc::new(NoopTool::new("alpha")))
            .unwrap()
            .register(Arc::new(NoopTool::new("beta")))
            .unwrap();
        let view = reg.restricted_to(&["alpha"]).unwrap();
        assert_eq!(view.len(), 1);
        assert!(view.get("alpha").is_some());
    }

    #[test]
    fn restricted_to_rejects_unknown_names_with_diagnostic() {
        let reg = ToolRegistry::new()
            .register(Arc::new(NoopTool::new("alpha")))
            .unwrap();
        let err = reg.restricted_to(&["alpha", "missing"]).unwrap_err();
        match err {
            Error::Config(msg) => {
                assert!(
                    msg.contains("missing"),
                    "diagnostic must name the missing tool: {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
