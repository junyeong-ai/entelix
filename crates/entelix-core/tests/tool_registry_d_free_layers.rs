//! Pin **invariant 19** — the tool dispatch layer ecosystem stays
//! `D`-free regardless of operator-supplied deps. //! +
//!
//! The registry holds a typed `D` clone and threads it into the leaf
//! `Tool<D>::execute` via `AgentContext<D>`. Layers, however, consume
//! the D-erased `BoxedToolService<ToolInvocation, Value>` boundary —
//! they never see `D`. This test pins the property by:
//!
//! 1. Defining a custom `D` (`MyDeps { counter, label }`) and a
//!    `Tool<MyDeps>` impl that reads from `ctx.deps()`.
//! 2. Wrapping the registry with a counting `tower::Layer` that fires
//!    on every dispatch — the *same* layer type a `D = ()` registry
//!    would use, with zero generic adaptation.
//! 3. Dispatching and asserting both (a) the layer fired and (b) the
//!    deps reached the leaf execution.
//!
//! A regression that introduced a `D` parameter on the layer signature
//! would break compilation of the `Layer<BoxedToolService>` bound;
//! a regression that severed deps flow would surface in (b).

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use async_trait::async_trait;
use entelix_core::service::{BoxedToolService, ToolInvocation};
use entelix_core::tools::{Tool, ToolMetadata, ToolRegistry};
use entelix_core::{AgentContext, ExecutionContext, Result};
use serde_json::{Value, json};
use tower::{Layer, Service};

/// Operator-supplied typed deps. Holds shared mutable counter +
/// stable label so the test asserts both that the deps clone
/// arrived at the leaf execution and that mutation through the
/// shared counter survives.
#[derive(Clone)]
struct MyDeps {
    counter: Arc<AtomicUsize>,
    label: String,
}

/// Tool that reads from `ctx.deps()` to prove deps flow. Returns
/// the post-increment counter value alongside the deps label so the
/// test inspects both fields without dragging the deps type into
/// the JSON output.
struct DepsObserverTool {
    metadata: ToolMetadata,
}

impl DepsObserverTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "deps_observer",
                "Reads from ctx.deps() and returns the read value.",
                json!({"type": "object", "properties": {}}),
            ),
        }
    }
}

#[async_trait]
impl Tool<MyDeps> for DepsObserverTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, _input: Value, ctx: &AgentContext<MyDeps>) -> Result<Value> {
        let deps = ctx.deps();
        let count = deps.counter.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(json!({"label": deps.label.clone(), "count": count}))
    }
}

/// `tower::Layer` that fires on every dispatch. The signature is
/// `Layer<BoxedToolService>` — `D`-free. A regression that added a
/// `D` parameter to the layer boundary would force this type to
/// either grow a `D` generic itself or stop compiling against the
/// new bound.
#[derive(Clone)]
struct CountingLayer {
    fires: Arc<AtomicUsize>,
}

impl CountingLayer {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let fires = Arc::new(AtomicUsize::new(0));
        (
            Self {
                fires: Arc::clone(&fires),
            },
            fires,
        )
    }
}

impl Layer<BoxedToolService> for CountingLayer {
    type Service = CountingService;

    fn layer(&self, inner: BoxedToolService) -> Self::Service {
        CountingService {
            inner,
            fires: Arc::clone(&self.fires),
        }
    }
}

#[derive(Clone)]
struct CountingService {
    inner: BoxedToolService,
    fires: Arc<AtomicUsize>,
}

impl Service<ToolInvocation> for CountingService {
    type Response = Value;
    type Error = entelix_core::Error;
    type Future = futures::future::BoxFuture<'static, Result<Value>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolInvocation) -> Self::Future {
        self.fires.fetch_add(1, Ordering::SeqCst);
        let mut inner = self.inner.clone();
        Box::pin(async move { inner.call(req).await })
    }
}

#[tokio::test]
async fn d_free_layer_wraps_typed_deps_registry() {
    let counter = Arc::new(AtomicUsize::new(0));
    let deps = MyDeps {
        counter: Arc::clone(&counter),
        label: "tenant-alpha".to_owned(),
    };

    let (layer, fires) = CountingLayer::new();
    let registry = ToolRegistry::<MyDeps>::with_deps(deps)
        .layer(layer)
        .register(Arc::new(DepsObserverTool::new()) as Arc<dyn Tool<MyDeps>>)
        .unwrap();

    let ctx = ExecutionContext::new();
    let result = registry
        .dispatch("tu_1", "deps_observer", json!({}), &ctx)
        .await
        .unwrap();

    assert_eq!(
        fires.load(Ordering::SeqCst),
        1,
        "layer must fire on dispatch"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "deps must reach leaf execution"
    );
    assert_eq!(
        result,
        json!({"label": "tenant-alpha", "count": 1}),
        "deps clone in registry must match the typed handle threaded into Tool::execute",
    );
}

#[tokio::test]
async fn same_layer_type_works_against_unit_deps_registry() {
    // Sibling assertion: the *same* CountingLayer type works against
    // `ToolRegistry::<()>` without modification. Compilation alone
    // is the load-bearing property — if a regression introduced a
    // `D` parameter on the layer signature, this block would not
    // compile.
    use std::marker::PhantomData;

    struct UnitTool {
        metadata: ToolMetadata,
        _marker: PhantomData<()>,
    }

    #[async_trait]
    impl Tool<()> for UnitTool {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }

        async fn execute(&self, _input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
            Ok(json!("ok"))
        }
    }

    let unit_tool = UnitTool {
        metadata: ToolMetadata::function("unit", "no deps", json!({"type": "object"})),
        _marker: PhantomData,
    };
    let (layer, fires) = CountingLayer::new();
    let registry = ToolRegistry::new()
        .layer(layer)
        .register(Arc::new(unit_tool) as Arc<dyn Tool<()>>)
        .unwrap();

    let ctx = ExecutionContext::new();
    let _ = registry
        .dispatch("tu_2", "unit", json!({}), &ctx)
        .await
        .unwrap();
    assert_eq!(fires.load(Ordering::SeqCst), 1);
}
