//! End-to-end HTTP tests over the in-memory tower stack — no
//! sockets, no async runtime gymnastics. Verifies:
//!
//! - sync run endpoint round-trips JSON state
//! - SSE stream endpoint emits per-mode events
//! - **single-tenant mode** (no `with_tenant_header`) routes every
//!   request under `DEFAULT_TENANT_ID`
//! - **multi-tenant mode** (`with_tenant_header(name)`) requires the
//!   header on every request — missing header → `400 Bad Request`
//! - cross-tenant isolation: requests carrying different tenant IDs
//!   never observe each other's state
//! - wake endpoint resumes from a checkpointed state via
//!   `Command::{Resume,Update}`
//! - error envelope shape on bad input

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use entelix_core::ThreadKey;
use entelix_core::context::ExecutionContext;
use entelix_graph::{Checkpoint, Checkpointer, InMemoryCheckpointer};
use entelix_runnable::RunnableLambda;
use entelix_server::AgentRouterBuilder;
use http_body_util::BodyExt;
use serde::{Deserialize, Serialize};
use tower::util::ServiceExt;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
struct EchoState {
    seen_tenant: String,
    counter: u32,
}

fn build_runnable() -> RunnableLambda<EchoState, EchoState> {
    RunnableLambda::new(|mut s: EchoState, ctx: ExecutionContext| {
        let tenant = ctx.tenant_id().to_owned();
        async move {
            s.seen_tenant = tenant;
            s.counter = s.counter.saturating_add(1);
            Ok::<_, _>(s)
        }
    })
}

// ── single-tenant mode (no `with_tenant_header` on the builder) ──

#[tokio::test]
async fn single_tenant_mode_routes_under_default_tenant() {
    let app = AgentRouterBuilder::new(build_runnable()).build().unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-2/runs")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["output"]["seen_tenant"], "default");
}

#[tokio::test]
async fn single_tenant_mode_ignores_present_tenant_header() {
    // Operator built single-tenant — even if a downstream proxy
    // injects `x-tenant-id`, the router does not honour it.
    let app = AgentRouterBuilder::new(build_runnable()).build().unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-1/runs")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["output"]["seen_tenant"], "default",
        "single-tenant mode must not honour an unsolicited tenant header"
    );
}

// ── multi-tenant mode (`with_tenant_header` opt-in) ──

#[tokio::test]
async fn multi_tenant_mode_extracts_default_header() {
    let app = AgentRouterBuilder::new(build_runnable())
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-1/runs")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["output"]["seen_tenant"], "acme");
    assert_eq!(v["output"]["counter"], 1);
}

#[tokio::test]
async fn multi_tenant_mode_rejects_missing_header_with_400() {
    let app = AgentRouterBuilder::new(build_runnable())
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-1/runs")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["kind"], "missing_tenant_header");
}

#[tokio::test]
async fn multi_tenant_mode_rejects_empty_header_with_400() {
    let app = AgentRouterBuilder::new(build_runnable())
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-1/runs")
        .header("content-type", "application/json")
        .header("x-tenant-id", "  ")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn multi_tenant_mode_custom_header_name_is_honoured() {
    let app = AgentRouterBuilder::new(build_runnable())
        .with_tenant_header("x-org")
        .build()
        .unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-3/runs")
        .header("content-type", "application/json")
        .header("x-org", "rocket")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["output"]["seen_tenant"], "rocket");
}

#[tokio::test]
async fn invalid_tenant_header_name_fails_at_build() {
    // Header names cannot contain control characters; deferred
    // validation surfaces at `build()`.
    let result = AgentRouterBuilder::new(build_runnable())
        .with_tenant_header("invalid header\nname")
        .build();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a valid HTTP header name"),
        "expected typed InvalidTenantHeader error, got {err}"
    );
}

#[tokio::test]
async fn cross_tenant_requests_observe_independent_state() {
    // Multi-tenant mode — two distinct tenants hitting the same
    // thread_id must each see their own tenant_id in state. The
    // runnable just echoes ctx.tenant_id() so a leak (e.g. a shared
    // mutable harness) would surface as the wrong string.
    let app = AgentRouterBuilder::new(build_runnable())
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let send = |tenant: &'static str| {
        let app = app.clone();
        async move {
            let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
            let request = Request::builder()
                .method("POST")
                .uri("/v1/threads/conv-shared/runs")
                .header("content-type", "application/json")
                .header("x-tenant-id", tenant)
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = app.oneshot(request).await.unwrap();
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v["output"]["seen_tenant"].as_str().unwrap().to_owned()
        }
    };
    let (a, b) = tokio::join!(send("acme"), send("rocket"));
    assert_eq!(a, "acme");
    assert_eq!(b, "rocket");
}

#[tokio::test]
async fn malformed_body_returns_bad_request_envelope() {
    let app = AgentRouterBuilder::new(build_runnable()).build().unwrap();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-bad/runs")
        .header("content-type", "application/json")
        .body(Body::from("not json"))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn health_returns_ok() {
    let app = AgentRouterBuilder::new(build_runnable()).build().unwrap();

    let request = Request::builder()
        .method("GET")
        .uri("/v1/health")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"ok");
}

#[tokio::test]
async fn stream_endpoint_emits_value_and_done_in_values_mode() {
    let app = AgentRouterBuilder::new(build_runnable())
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("GET")
        .uri("/v1/threads/conv-stream/stream?mode=values")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&bytes).unwrap();
    assert!(body.contains("event: value"), "{body}");
    assert!(body.contains("event: done"), "{body}");
    assert!(body.contains("acme"), "{body}");
}

#[tokio::test]
async fn unknown_stream_mode_rejected() {
    let app = AgentRouterBuilder::new(build_runnable()).build().unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("GET")
        .uri("/v1/threads/conv-x/stream?mode=oops")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["kind"], "bad_request");
}

#[tokio::test]
async fn wake_without_checkpointer_returns_config_error() {
    let app = AgentRouterBuilder::new(build_runnable()).build().unwrap();

    let body = serde_json::json!({"command": "resume"});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-w/wake")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["kind"], "config");
}

#[tokio::test]
async fn wake_resume_uses_checkpointed_state() {
    let cp = Arc::new(InMemoryCheckpointer::<EchoState>::new());
    // Seed a checkpoint so wake has something to resume from.
    let key = ThreadKey::new(TenantId::new("acme"), "conv-w");
    let checkpoint = Checkpoint::new(
        &key,
        0,
        EchoState {
            seen_tenant: "old".into(),
            counter: 41,
        },
        Some("entry".to_owned()),
    );
    cp.put(checkpoint).await.unwrap();

    let app = AgentRouterBuilder::new(build_runnable())
        .with_checkpointer(cp.clone())
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({"command": "resume"});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-w/wake")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // counter incremented from the checkpointed 41; tenant overwritten to "acme".
    assert_eq!(v["output"]["counter"], 42);
    assert_eq!(v["output"]["seen_tenant"], "acme");
}

#[tokio::test]
async fn wake_update_replaces_checkpointed_state() {
    let cp = Arc::new(InMemoryCheckpointer::<EchoState>::new());
    let key = ThreadKey::new(TenantId::new("acme"), "conv-u");
    let checkpoint = Checkpoint::new(
        &key,
        0,
        EchoState {
            seen_tenant: "old".into(),
            counter: 10,
        },
        Some("entry".to_owned()),
    );
    cp.put(checkpoint).await.unwrap();

    let app = AgentRouterBuilder::new(build_runnable())
        .with_checkpointer(cp.clone())
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({
        "command": {"update": {"seen_tenant": "ignored", "counter": 100}}
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-u/wake")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // The Update branch replaces state, so counter starts from 100 (not 10).
    assert_eq!(v["output"]["counter"], 101);
}

#[tokio::test]
async fn wake_approve_tool_attaches_pending_decisions_extension() {
    // Regression for ADR-0072: server `/wake` exposes the typed
    // `ApproveTool` resume primitive. The runnable inspects the
    // ctx-extension to verify the decision threaded through.
    use entelix_core::approval::{ApprovalDecision, PendingApprovalDecisions};

    let cp = Arc::new(InMemoryCheckpointer::<EchoState>::new());
    let key = ThreadKey::new(TenantId::new("acme"), "conv-a");
    let checkpoint = Checkpoint::new(
        &key,
        0,
        EchoState {
            seen_tenant: "old".into(),
            counter: 5,
        },
        Some("entry".to_owned()),
    );
    cp.put(checkpoint).await.unwrap();

    let runnable = RunnableLambda::new(|mut s: EchoState, ctx: ExecutionContext| async move {
        let pending = ctx.extension::<PendingApprovalDecisions>().unwrap();
        assert!(matches!(
            pending.get("tu-7"),
            Some(ApprovalDecision::Approve)
        ));
        s.counter = s.counter.saturating_add(1);
        Ok::<_, _>(s)
    });
    let app = AgentRouterBuilder::new(runnable)
        .with_checkpointer(cp)
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({
        "command": {"approve_tool": {"tool_use_id": "tu-7", "decision": "approve"}}
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-a/wake")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn wake_approve_tool_rejects_await_external() {
    // Resume with AwaitExternal would re-enter the same pause —
    // structurally rejected at the HTTP boundary.
    let cp = Arc::new(InMemoryCheckpointer::<EchoState>::new());
    let key = ThreadKey::new(TenantId::new("acme"), "conv-ax");
    let checkpoint = Checkpoint::new(
        &key,
        0,
        EchoState {
            seen_tenant: "old".into(),
            counter: 0,
        },
        Some("entry".to_owned()),
    );
    cp.put(checkpoint).await.unwrap();

    let app = AgentRouterBuilder::new(build_runnable())
        .with_checkpointer(cp)
        .with_tenant_header(entelix_server::DEFAULT_TENANT_HEADER)
        .build()
        .unwrap();

    let body = serde_json::json!({
        "command": {"approve_tool": {"tool_use_id": "tu-7", "decision": "await_external"}}
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-ax/wake")
        .header("content-type", "application/json")
        .header("x-tenant-id", "acme")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn wake_unknown_thread_returns_not_found() {
    let cp = Arc::new(InMemoryCheckpointer::<EchoState>::new());
    let app = AgentRouterBuilder::new(build_runnable())
        .with_checkpointer(cp)
        .build()
        .unwrap();

    let body = serde_json::json!({"command": "resume"});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/never-existed/wake")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn handler_drop_cancels_per_request_token() {
    // Regression: build_ctx must hand the runnable a real
    // CancellationToken (not the default never-fires token), and
    // the handler must hold its `drop_guard()` so that aborting the
    // handler future cancels the token. Without this wiring, a tool
    // polling `ctx.is_cancelled()` from inside a long-running invoke
    // can never observe a client disconnect.
    use std::sync::Arc as StdArc;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel::<()>();
    let cancel_tx = StdArc::new(StdMutex::new(Some(cancel_tx)));

    let runnable = RunnableLambda::new(move |s: EchoState, ctx: ExecutionContext| {
        let cancel_tx = cancel_tx.clone();
        async move {
            // Spawn a detached watcher: it survives the handler
            // abort and fires the channel exactly when the token
            // cancels. Validates the wiring without depending on
            // post-cancellation code running inside the aborted
            // future itself (`.await` after cancellation never
            // resumes if the future is dropped).
            let token = ctx.cancellation().clone();
            tokio::spawn(async move {
                token.cancelled().await;
                let tx = cancel_tx.lock().unwrap().take();
                if let Some(tx) = tx {
                    let _ = tx.send(());
                }
            });
            std::future::pending::<EchoState>().await;
            Ok::<_, _>(s)
        }
    });
    let app = AgentRouterBuilder::new(runnable).build().unwrap();

    let body = serde_json::json!({"input": {"seen_tenant": "", "counter": 0}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/threads/conv-cx/runs")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response_fut = tokio::spawn(app.oneshot(request));
    // Yield enough times for the handler to start and the watcher
    // task to register on the token. Both happen on the runtime;
    // a couple of yields are sufficient on the current-thread
    // executor `tokio::test` uses.
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    response_fut.abort();
    let _ = response_fut.await;

    tokio::time::timeout(Duration::from_secs(1), cancel_rx)
        .await
        .unwrap()
        .unwrap();
}
