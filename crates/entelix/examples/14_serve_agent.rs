//! `14_serve_agent` — `AgentRouterBuilder` end-to-end demo.
//!
//! Build: `cargo build --example 14_serve_agent -p entelix --features=server`
//! Run:   `cargo run   --example 14_serve_agent -p entelix --features=server`
//!
//! Wires the production server stack together:
//!
//! 1. Build a `CompiledGraph<S>` agent (a tiny two-node planner →
//!    finalize loop using `RunnableLambda` — no real LLM, so the
//!    example is CI-deterministic).
//! 2. Mount it under [`entelix_server::AgentRouterBuilder`] which
//!    exposes the standard `/v1/threads/{id}/...` HTTP surface.
//! 3. Hit the service through `tower::ServiceExt::oneshot` — no
//!    socket, no real HTTP server. The response carries the
//!    `tenant_id` extracted from the request header (ADR-0017),
//!    proving multi-tenant header propagation works end-to-end.
//!
//! Production swap-ins (one-line each):
//! - the agent: replace `RunnableLambda` with a real
//!   `create_react_agent(layered_model, layered_tools)`
//! - the model: build a `ChatModel::new(codec, transport, ...)
//!   .layer(PolicyLayer::new(manager)).layer(OtelLayer::new("anthropic"))`
//! - tools: `ToolRegistry::new().register(...)
//!   .layer(PolicyLayer::new(manager)).layer(OtelLayer::new("anthropic"))`
//! - the listener: swap `tower::oneshot` for `axum::serve(listener, router)`

#![cfg(feature = "server")]
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::expect_used
)]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use entelix::{AgentRouterBuilder, ExecutionContext, Result, RunnableLambda, StateGraph};
use http_body_util::BodyExt;
use serde::{Deserialize, Serialize};
use tower::util::ServiceExt;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct AgentState {
    /// Tenant the request was routed under (filled by the planner
    /// node from `ctx.tenant_id()`).
    tenant_id: String,
    /// Toy plan steps the agent produced.
    steps: Vec<String>,
    /// Final natural-language reply.
    reply: Option<String>,
}

fn build_agent() -> Result<entelix::CompiledGraph<AgentState>> {
    // Node 1 — planner: records the tenant scope and emits two steps.
    let planner = RunnableLambda::new(|mut s: AgentState, ctx: ExecutionContext| async move {
        s.tenant_id = ctx.tenant_id().as_str().to_owned();
        s.steps.push(format!("plan@{}", s.tenant_id));
        s.steps.push("execute".to_owned());
        Ok::<_, _>(s)
    });

    // Node 2 — finalize: produces the user-facing reply.
    let finalize = RunnableLambda::new(|mut s: AgentState, _ctx| async move {
        s.reply = Some(format!(
            "Tenant '{}' ran {} step(s)",
            s.tenant_id,
            s.steps.len()
        ));
        Ok::<_, _>(s)
    });

    StateGraph::new()
        .add_node("plan", planner)
        .add_node("finalize", finalize)
        .add_edge("plan", "finalize")
        .set_entry_point("plan")
        .add_finish_point("finalize")
        .compile()
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("── building agent + multi-tenant router ────────────");
    let agent = build_agent()?;
    // `with_tenant_header` opts the router into multi-tenant mode:
    // every request MUST carry the named header. Missing → 400. The
    // single-tenant alternative omits this call entirely.
    let router = AgentRouterBuilder::new(agent)
        .with_tenant_header(entelix::SERVER_DEFAULT_TENANT_HEADER)
        .build()
        .expect("router build is infallible when the header name is valid");
    println!("  CompiledGraph mounted under /v1 (tenant header required)");

    // ── Issue a request as tenant 'acme' ─────────────────
    println!("\n── POST /v1/threads/conv-1/runs (X-Tenant-Id: acme) ─────");
    let body = serde_json::json!({"input": {"tenant_id": "", "steps": [], "reply": null}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-1/runs")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    println!("response: {v:#}");
    assert_eq!(v["output"]["tenant_id"], "acme");
    assert_eq!(v["output"]["reply"], "Tenant 'acme' ran 2 step(s)");

    // ── Same request body, different tenant header ──────
    println!("\n── POST /v1/threads/conv-2/runs (X-Tenant-Id: rocket) ─────");
    let body = serde_json::json!({"input": {"tenant_id": "", "steps": [], "reply": null}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-2/runs")
        .header("content-type", "application/json")
        .header("x-tenant-id", "rocket")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    println!("response: {v:#}");
    assert_eq!(v["output"]["tenant_id"], "rocket");

    // ── Missing header → 400 Bad Request (strict mode) ──
    println!("\n── POST /v1/threads/conv-3/runs (no header → 400) ─────");
    let body = serde_json::json!({"input": {"tenant_id": "", "steps": [], "reply": null}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-3/runs")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    println!("response: {v:#}");
    assert_eq!(v["error"]["kind"], "missing_tenant_header");

    println!("\n   ✓ AgentRouterBuilder routed two tenants through one CompiledGraph;");
    println!("     missing header rejected with 400 (multi-tenant strict mode).");
    Ok(())
}
