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
use crate::ir::ToolSpec;
use crate::service::{BoxedToolService, ToolInvocation};
use crate::tools::cache::ToolCacheMode;
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
                    target: "entelix_core::tools",
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
/// `D` defaults to `()` for the deps-less zero-cost path —
/// [`ToolRegistry::new()`] hands out an empty registry whose
/// `Tool<D>::execute` body receives `()` as `ctx.deps()`. Operator-
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
    /// Diagnostic stack — names captured at [`Self::layer`] compose
    /// time via [`crate::NamedLayer::layer_name`], in insertion order
    /// (innermost composed first → outermost composed last). Mirrors
    /// [`crate::ChatModel::layer_names`] for the tool-dispatch spine.
    layer_names: Vec<&'static str>,
    /// Cache stamping strategy applied at [`Self::tool_specs`]
    /// materialisation time. Default [`ToolCacheMode::None`] —
    /// specs leave unchanged.
    cache_mode: ToolCacheMode,
}

impl<D: Clone> Clone for ToolRegistry<D> {
    fn clone(&self) -> Self {
        Self {
            by_name: self.by_name.clone(),
            deps: self.deps.clone(),
            factory: self.factory.clone(),
            layer_names: self.layer_names.clone(),
            cache_mode: self.cache_mode,
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
            layer_names: Vec::new(),
            cache_mode: ToolCacheMode::None,
        }
    }

    /// Set the [`ToolCacheMode`] applied to specs returned by
    /// [`Self::tool_specs`]. The mode is registry-level rather
    /// than per-call because operators almost always pick once and
    /// stick — sub-agents needing a different policy can re-stamp
    /// post-materialisation via [`ToolCacheMode::apply`].
    #[must_use]
    pub fn with_cache_mode(mut self, mode: ToolCacheMode) -> Self {
        self.cache_mode = mode;
        self
    }

    /// Materialise model-facing tool specs in stable lexical order
    /// and apply the configured [`ToolCacheMode`].
    ///
    /// Operators feed the result directly to
    /// [`ChatModel::with_tools`](crate::ChatModel::with_tools) — the
    /// canonical path from a registry to an active chat model's
    /// tool catalogue. The lexical sort makes the marker placement
    /// for [`ToolCacheMode::Suffix`] deterministic across program
    /// runs even though the underlying storage is a `HashMap`.
    #[must_use]
    pub fn tool_specs(&self) -> Arc<[ToolSpec]> {
        let mut tools: Vec<&Arc<dyn Tool<D>>> = self.by_name.values().collect();
        tools.sort_by(|a, b| a.metadata().name.cmp(&b.metadata().name));
        let specs: Vec<ToolSpec> = tools
            .into_iter()
            .map(|tool| tool.metadata().to_tool_spec())
            .collect();
        Arc::from(self.cache_mode.apply(specs))
    }

    /// Deterministic SHA-256 fingerprint of the **advertised** tool
    /// surface — the identity an audit / replay consumer compares
    /// across deployments to detect schema drift.
    ///
    /// Algorithm (stable across consumers — pin this contract when
    /// integrating with external audit tooling):
    ///
    /// 1. Walk tools in lexical order by `metadata().name`.
    /// 2. Per tool, build a JSON object with the model-facing
    ///    surface only: `{name, description, input_schema,
    ///    output_schema}`. `output_schema` is rendered as `null`
    ///    when absent so its presence/absence is itself part of the
    ///    fingerprint.
    /// 3. Serialize each per-tool object to a string. `serde_json`
    ///    with default features uses `BTreeMap` for `Map`, so key
    ///    order within each object is sorted.
    /// 4. Concatenate (newline-separated) and feed into SHA-256.
    /// 5. Lower-case hexadecimal digest.
    ///
    /// **Excludes** operator-side knobs (`cache_control`,
    /// `retry_hint`, `effect`, `version`, `idempotent`,
    /// `typical_duration`) because those are deployment-shaped, not
    /// identity-shaped — two deployments with different cache
    /// strategies should still report the same fingerprint.
    #[must_use]
    pub fn canonical_fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut tools: Vec<&Arc<dyn Tool<D>>> = self.by_name.values().collect();
        tools.sort_by(|a, b| a.metadata().name.cmp(&b.metadata().name));
        let mut hasher = Sha256::new();
        for tool in tools {
            let m = tool.metadata();
            let payload = serde_json::json!({
                "name": &m.name,
                "description": &m.description,
                "input_schema": &m.input_schema,
                "output_schema": m.output_schema,
            });
            // `Value::to_string` enumerates `Map` keys in BTreeMap
            // order — canonical sort across the whole nested tree.
            // The path is total: `Value` is always serialisable.
            hasher.update(payload.to_string().as_bytes());
            hasher.update(b"\n");
        }
        format!("{:x}", hasher.finalize())
    }

    /// Append a tool. Per, `*Registry` types are
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
                 (registry is append-only — see)"
            )));
        }
        self.by_name.insert(name, tool);
        Ok(self)
    }

    /// Append a layer to the dispatch stack. Layers stack around the
    /// leaf inner tool service — the **last-registered layer is
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
    ///
    /// ## Introspection
    ///
    /// Every layer is recorded in [`Self::layer_names`] at compose
    /// time via [`crate::NamedLayer::layer_name`]. First-party
    /// entelix tool-side layers (`RetryToolLayer`, `ApprovalLayer`,
    /// `ToolEventLayer<S>`, `ToolHookLayer`, `ScopedToolLayer`, plus
    /// the shared `PolicyLayer` / `OtelLayer`) implement
    /// [`crate::NamedLayer`] directly; external `tower::Layer`
    /// middleware wraps through [`crate::service::WithName`] to
    /// participate.
    #[must_use]
    pub fn layer<L>(mut self, layer: L) -> Self
    where
        L: Layer<BoxedToolService> + crate::NamedLayer + Clone + Send + Sync + 'static,
        L::Service:
            Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
        <L::Service as Service<ToolInvocation>>::Future: Send + 'static,
    {
        // Capture the static name BEFORE the type-erasing closure
        // swallows `L`. The factory chain (Arc<dyn Fn(InnerToolService)
        // -> BoxedToolService>) discards concrete layer types, so
        // introspection has to happen here at compose time. Mirrors
        // `ChatModel::layer`.
        self.layer_names.push(layer.layer_name());

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

    /// Compose a `tower::Layer` whose only difference from a
    /// first-party tool-side layer is the missing
    /// [`crate::NamedLayer`] impl. Equivalent to
    /// `self.layer(WithName::new(name, layer))` — the wrapper
    /// supplies the identity surfaced through [`Self::layer_names`].
    ///
    /// Use this for external middleware (operator-defined wrappers,
    /// per-call task-local scope layers, etc.) without implementing
    /// [`crate::NamedLayer`] manually. First-party tool-side layers
    /// (`ApprovalLayer`, `ToolEventLayer`, `ToolHookLayer`,
    /// `ScopedToolLayer`, `RetryToolLayer`) already implement
    /// [`crate::NamedLayer`] and should compose via [`Self::layer`]
    /// directly so the canonical `tool_*` role name ships at the
    /// call site.
    #[must_use]
    pub fn layer_named<L>(self, name: &'static str, layer: L) -> Self
    where
        L: Layer<BoxedToolService> + Clone + Send + Sync + 'static,
        L::Service:
            Service<ToolInvocation, Response = Value, Error = Error> + Clone + Send + 'static,
        <L::Service as Service<ToolInvocation>>::Future: Send + 'static,
    {
        self.layer(crate::service::WithName::new(name, layer))
    }

    /// Diagnostic snapshot of the composed layer stack, in
    /// registration order: `[0]` is the first `.layer(...)` call
    /// (innermost, against the leaf inner tool service); the last
    /// element is the most recent registration (outermost, sees
    /// requests first). Empty for a registry with no layers composed.
    ///
    /// Mirrors [`crate::ChatModel::layer_names`] for the tool-
    /// dispatch spine. Operators correlate the two channels at boot
    /// to verify their wiring matches the expected production shape.
    #[must_use]
    pub fn layer_names(&self) -> &[&'static str] {
        &self.layer_names
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
            layer_names: self.layer_names.clone(),
            cache_mode: self.cache_mode,
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

    /// Build a [`BoxedToolService`] for `name` — the leaf inner
    /// tool service wrapped in the configured layer stack. Returns
    /// `None` if the tool is not registered.
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
            .field("layers", &self.layer_names)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
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
    fn layer_names_track_compose_order_and_namedlayer_identity() {
        use crate::WithName;

        let bare = ToolRegistry::new();
        assert!(
            bare.layer_names().is_empty(),
            "freshly built registry has no layers"
        );

        let log_a = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log_b = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log_c = Arc::new(std::sync::Mutex::new(Vec::new()));
        let reg = ToolRegistry::new()
            .layer(LabelLayer {
                label: 'A',
                log: log_a,
            })
            .layer(WithName::new(
                "external",
                LabelLayer {
                    label: 'B',
                    log: log_b,
                },
            ))
            .layer_named(
                "convenience",
                LabelLayer {
                    label: 'C',
                    log: log_c,
                },
            );

        assert_eq!(
            reg.layer_names(),
            ["label", "external", "convenience"],
            "layer_named threads through the same channel as WithName::new"
        );
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

    impl crate::NamedLayer for LabelLayer {
        fn layer_name(&self) -> &'static str {
            "label"
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

    fn registry_with_three_tools() -> ToolRegistry {
        ToolRegistry::new()
            .register(Arc::new(NoopTool::new("gamma")))
            .unwrap()
            .register(Arc::new(NoopTool::new("alpha")))
            .unwrap()
            .register(Arc::new(NoopTool::new("beta")))
            .unwrap()
    }

    #[test]
    fn tool_specs_default_mode_emits_lexically_sorted_specs_with_no_cache() {
        let specs = registry_with_three_tools().tool_specs();
        let names: Vec<_> = specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
        assert!(specs.iter().all(|s| s.cache_control.is_none()));
    }

    #[test]
    fn tool_specs_with_suffix_mode_marks_only_the_last_spec() {
        let cache = crate::ir::CacheControl::five_minutes();
        let specs = registry_with_three_tools()
            .with_cache_mode(ToolCacheMode::Suffix(cache))
            .tool_specs();
        assert_eq!(specs[0].cache_control, None, "alpha must be unmarked");
        assert_eq!(specs[1].cache_control, None, "beta must be unmarked");
        assert_eq!(
            specs[2].cache_control,
            Some(cache),
            "gamma (last lexical) must carry the suffix cache marker"
        );
    }

    #[test]
    fn tool_specs_with_per_spec_mode_marks_every_spec() {
        let cache = crate::ir::CacheControl::one_hour();
        let specs = registry_with_three_tools()
            .with_cache_mode(ToolCacheMode::PerSpec(cache))
            .tool_specs();
        assert!(
            specs.iter().all(|s| s.cache_control == Some(cache)),
            "every spec must carry the per-spec cache marker"
        );
    }

    #[test]
    fn cache_mode_is_inherited_through_filter_views() {
        let cache = crate::ir::CacheControl::five_minutes();
        let parent = registry_with_three_tools().with_cache_mode(ToolCacheMode::Suffix(cache));
        let view = parent.filter(|tool| tool.metadata().name != "beta");
        let specs = view.tool_specs();
        let names: Vec<_> = specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "gamma"]);
        assert_eq!(
            specs.last().and_then(|s| s.cache_control),
            Some(cache),
            "filtered view must keep the parent's cache mode"
        );
    }

    #[test]
    fn canonical_fingerprint_is_empty_sha256_for_empty_registry() {
        // SHA-256 of zero-byte input — the canonical "no tools"
        // fingerprint. Pinned because consumers compare across
        // deployments; an unexpected hash drift here would silently
        // break cross-consumer audit comparison.
        assert_eq!(
            ToolRegistry::new().canonical_fingerprint(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn canonical_fingerprint_is_order_invariant() {
        // Same three tools registered in two different orders must
        // produce the same fingerprint. The lexical sort inside
        // canonical_fingerprint guarantees this; if the sort is ever
        // dropped, this test fails.
        let a = ToolRegistry::new()
            .register(Arc::new(NoopTool::new("gamma")))
            .unwrap()
            .register(Arc::new(NoopTool::new("alpha")))
            .unwrap()
            .register(Arc::new(NoopTool::new("beta")))
            .unwrap();
        let b = ToolRegistry::new()
            .register(Arc::new(NoopTool::new("alpha")))
            .unwrap()
            .register(Arc::new(NoopTool::new("beta")))
            .unwrap()
            .register(Arc::new(NoopTool::new("gamma")))
            .unwrap();
        assert_eq!(a.canonical_fingerprint(), b.canonical_fingerprint());
    }

    #[test]
    fn canonical_fingerprint_is_insensitive_to_cache_mode() {
        // cache_control is a deployment-side optimization, not part
        // of the tool identity — two registries that differ only in
        // their `ToolCacheMode` must report the same fingerprint.
        let cache = crate::ir::CacheControl::one_hour();
        let plain = registry_with_three_tools();
        let cached = registry_with_three_tools().with_cache_mode(ToolCacheMode::PerSpec(cache));
        assert_eq!(
            plain.canonical_fingerprint(),
            cached.canonical_fingerprint(),
            "cache_mode changes must NOT shift the fingerprint"
        );
    }

    #[test]
    fn tool_specs_share_arc_across_clones_no_deep_clone() {
        // Two consecutive calls return distinct allocations (each is
        // a fresh build), but cloning a `ChatModelConfig`-shaped
        // `Arc<[ToolSpec]>` is a refcount bump — design property
        // closing the per-dispatch deep-clone hot path that
        // pre-`Arc<[T]>` `Vec<ToolSpec>` paid every planner turn.
        let registry = registry_with_three_tools();
        let specs = registry.tool_specs();
        let cloned = Arc::clone(&specs);
        assert!(
            Arc::ptr_eq(&specs, &cloned),
            "Arc::clone must share allocation; deep clone here is a regression"
        );
    }

    #[test]
    fn canonical_fingerprint_changes_when_description_changes() {
        // Description IS part of the model-facing surface — a change
        // is a meaningful identity shift. The model sees a different
        // tool catalogue and an audit consumer should detect it.
        struct Custom {
            metadata: ToolMetadata,
        }
        #[async_trait::async_trait]
        impl Tool for Custom {
            fn metadata(&self) -> &ToolMetadata {
                &self.metadata
            }
            async fn execute(&self, _input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
                Ok(Value::Null)
            }
        }
        let a = ToolRegistry::new()
            .register(Arc::new(Custom {
                metadata: ToolMetadata::function("alpha", "first", json!({"type": "object"})),
            }))
            .unwrap();
        let b = ToolRegistry::new()
            .register(Arc::new(Custom {
                metadata: ToolMetadata::function(
                    "alpha",
                    "first (rev 2)",
                    json!({"type": "object"}),
                ),
            }))
            .unwrap();
        assert_ne!(a.canonical_fingerprint(), b.canonical_fingerprint());
    }
}
