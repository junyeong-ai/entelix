//! Streamable-http + Roots end-to-end via `wiremock`.
//!
//! Three concurrent properties verified end-to-end:
//!
//! 1. **Streamable response handling** — the `initialize` response
//!    rides as an SSE stream. The client extracts the matching
//!    JSON-RPC response (id=1) and stamps the
//!    `Mcp-Session-Id` header for sticky session.
//! 2. **Server-initiated `roots/list` dispatch** — the same SSE
//!    response embeds a second event: a server-initiated
//!    `roots/list` request with id=777. The client must
//!    spawn-dispatch it to the configured `RootsProvider` and POST
//!    the JSON-RPC response back to the same endpoint.
//! 3. **`notifications/roots/list_changed`** — the manager-level
//!    dispatcher emits the notification as a POST.
//!
//! The mock matches on `body_partial_json` so each MCP method has
//! its own ResponseTemplate; assertions read
//! `mock_server.received_requests()` to verify the client's
//! outbound shape (server-initiated response, listChanged
//! notification).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_const_for_fn,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::unnecessary_map_or,
    clippy::manual_contains
)]

use std::sync::Arc;

use entelix_mcp::{
    HttpMcpClient, McpClient, McpRoot, McpServerConfig, RootsProvider, StaticRootsProvider,
};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SESSION_ID: &str = "session-roots-e2e";

fn sse_body(events: Vec<Value>) -> String {
    let mut out = String::new();
    for event in events {
        out.push_str("event: message\ndata: ");
        out.push_str(&event.to_string());
        out.push_str("\n\n");
    }
    out
}

fn initialize_result_event(id: i64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": { "name": "mock-streamable", "version": "0.0.1" },
        }
    })
}

fn server_initiated_roots_request(id: i64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "roots/list",
        "params": {},
    })
}

async fn mount_initialize_with_server_request(server: &MockServer, server_request_id: i64) {
    let body = sse_body(vec![
        initialize_result_event(1),
        server_initiated_roots_request(server_request_id),
    ]);
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": "initialize" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("mcp-session-id", SESSION_ID)
                .set_body_raw(body.into_bytes(), "text/event-stream"),
        )
        .mount(server)
        .await;
}

async fn mount_initialize_without_server_request(server: &MockServer) {
    let body = sse_body(vec![initialize_result_event(1)]);
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": "initialize" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("mcp-session-id", SESSION_ID)
                .set_body_raw(body.into_bytes(), "text/event-stream"),
        )
        .mount(server)
        .await;
}

async fn mount_initialized_notification(server: &MockServer) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "notifications/initialized"
        })))
        .respond_with(ResponseTemplate::new(200))
        .mount(server)
        .await;
}

async fn mount_tools_list_empty(server: &MockServer) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({ "method": "tools/list" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": { "tools": [] }
        })))
        .mount(server)
        .await;
}

async fn mount_catchall_ok(server: &MockServer) {
    // The server-initiated response POST and the
    // `notifications/roots/list_changed` POST both land here. We
    // accept anything that didn't match a more specific mock.
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(server)
        .await;
}

#[tokio::test]
async fn server_initiated_roots_list_dispatches_to_static_provider() {
    let server = MockServer::start().await;
    mount_initialize_with_server_request(&server, 777).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;
    mount_catchall_ok(&server).await;

    let provider = Arc::new(StaticRootsProvider::new(vec![
        McpRoot::new("file:///workspace/repo").with_name("repo"),
        McpRoot::new("vault://team/secrets"),
    ]));
    let config = McpServerConfig::http("mock", server.uri())
        .unwrap()
        .with_roots_provider(provider);
    let client = HttpMcpClient::new(config).unwrap();
    let _ = client.initialize().await.unwrap();

    // Allow spawn-dispatched handler a brief tick to finish its
    // POST. The wiremock store is updated synchronously on the
    // mock side; the only delay is the spawn → tokio scheduling.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let received = server.received_requests().await.unwrap();
    let server_response = received
        .iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("id").and_then(Value::as_i64) == Some(777) && v.get("result").is_some())
        .expect("client must POST a JSON-RPC response for the server-initiated roots/list");

    let roots = server_response
        .pointer("/result/roots")
        .and_then(Value::as_array)
        .expect("response must carry a roots array")
        .clone();
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0]["uri"], "file:///workspace/repo");
    assert_eq!(roots[0]["name"], "repo");
    assert_eq!(roots[1]["uri"], "vault://team/secrets");
    assert!(roots[1].get("name").map_or(true, Value::is_null));

    // Sticky session id on every outbound request.
    let outbound_session_ids: Vec<_> = received
        .iter()
        .filter_map(|r| {
            r.headers
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
        })
        .collect();
    assert!(
        outbound_session_ids.iter().any(|s| *s == SESSION_ID),
        "client must echo Mcp-Session-Id once captured"
    );
}

#[tokio::test]
async fn server_initiated_roots_list_returns_method_not_found_when_provider_absent() {
    let server = MockServer::start().await;
    mount_initialize_with_server_request(&server, 778).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;
    mount_catchall_ok(&server).await;

    // No `with_roots_provider` — server-initiated `roots/list`
    // must come back as JSON-RPC -32601 (method not found).
    let config = McpServerConfig::http("mock", server.uri()).unwrap();
    let client = HttpMcpClient::new(config).unwrap();
    let _ = client.initialize().await.unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let received = server.received_requests().await.unwrap();
    let err_response = received
        .iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("id").and_then(Value::as_i64) == Some(778) && v.get("error").is_some())
        .expect("client must POST -32601 for server-initiated request without provider");

    assert_eq!(err_response["error"]["code"], -32601);
    let msg = err_response["error"]["message"].as_str().unwrap_or("");
    assert!(msg.contains("roots/list"), "{msg}");
}

#[tokio::test]
async fn provider_error_surfaces_as_jsonrpc_internal_error() {
    let server = MockServer::start().await;
    mount_initialize_with_server_request(&server, 779).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;
    mount_catchall_ok(&server).await;

    #[derive(Debug)]
    struct FailingProvider;
    #[async_trait::async_trait]
    impl RootsProvider for FailingProvider {
        async fn list_roots(&self) -> entelix_mcp::McpResult<Vec<McpRoot>> {
            Err(entelix_mcp::McpError::Config(
                "operator simulated failure".into(),
            ))
        }
    }

    let config = McpServerConfig::http("mock", server.uri())
        .unwrap()
        .with_roots_provider(Arc::new(FailingProvider));
    let client = HttpMcpClient::new(config).unwrap();
    let _ = client.initialize().await.unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let received = server.received_requests().await.unwrap();
    let err = received
        .iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("id").and_then(Value::as_i64) == Some(779) && v.get("error").is_some())
        .expect("provider failure must surface as JSON-RPC error response");
    assert_eq!(err["error"]["code"], -32603);
}

#[tokio::test]
async fn notify_roots_changed_emits_jsonrpc_notification() {
    let server = MockServer::start().await;
    mount_initialize_without_server_request(&server).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;
    mount_catchall_ok(&server).await;

    let provider = Arc::new(StaticRootsProvider::new(vec![McpRoot::new("file:///a")]));
    let config = McpServerConfig::http("mock", server.uri())
        .unwrap()
        .with_roots_provider(provider);
    let client = HttpMcpClient::new(config).unwrap();
    let _ = client.initialize().await.unwrap();

    client.notify_roots_changed().await.unwrap();

    let received = server.received_requests().await.unwrap();
    let saw_notification = received.iter().any(|r| {
        let v: Value = serde_json::from_slice(&r.body).unwrap_or(Value::Null);
        v.get("method").and_then(Value::as_str) == Some("notifications/roots/list_changed")
            && v.get("id").is_none()
    });
    assert!(
        saw_notification,
        "client must POST notifications/roots/list_changed (no id)"
    );
}

#[tokio::test]
async fn capabilities_advertise_roots_only_when_provider_wired() {
    // Two clients pointed at the same mock — one with a
    // RootsProvider, one without. Inspect the initialize body
    // each client sent to confirm capabilities differ.
    let server = MockServer::start().await;
    mount_initialize_without_server_request(&server).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;

    let with_provider = HttpMcpClient::new(
        McpServerConfig::http("mock", server.uri())
            .unwrap()
            .with_roots_provider(Arc::new(StaticRootsProvider::new(Vec::<McpRoot>::new()))),
    )
    .unwrap();
    let _ = with_provider.initialize().await.unwrap();

    let without_provider =
        HttpMcpClient::new(McpServerConfig::http("mock", server.uri()).unwrap()).unwrap();
    let _ = without_provider.initialize().await.unwrap();

    let received = server.received_requests().await.unwrap();
    let initialize_bodies: Vec<Value> = received
        .iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .filter(|v| v.get("method").and_then(Value::as_str) == Some("initialize"))
        .collect();
    assert_eq!(initialize_bodies.len(), 2);

    let advertises_roots = |body: &Value| {
        body.pointer("/params/capabilities/roots/listChanged") == Some(&Value::Bool(true))
    };
    let count_with = initialize_bodies
        .iter()
        .filter(|b| advertises_roots(b))
        .count();
    let count_without = initialize_bodies
        .iter()
        .filter(|b| !advertises_roots(b))
        .count();
    assert_eq!(count_with, 1, "exactly one initialize advertises roots");
    assert_eq!(count_without, 1, "exactly one omits the advertisement");
}
