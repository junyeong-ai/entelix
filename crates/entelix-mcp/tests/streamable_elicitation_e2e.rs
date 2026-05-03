//! Streamable-http + Elicitation end-to-end via `wiremock`.
//!
//! Mirrors `streamable_roots_e2e.rs` for the
//! `elicitation/create` server-initiated channel:
//!
//! 1. **Server-initiated `elicitation/create` dispatch** — the
//!    initialize SSE response embeds a server-initiated
//!    `elicitation/create` request. The client must
//!    spawn-dispatch it to the configured `ElicitationProvider`
//!    and POST the JSON-RPC response back.
//! 2. **Capability advertisement** — the initialize body
//!    advertises `elicitation: {}` iff a provider is wired.
//! 3. **Method-not-found path** — without a provider,
//!    server-initiated `elicitation/create` returns
//!    JSON-RPC error -32601 (Method not found).
//! 4. **Action variants** — accept / decline / cancel each
//!    serialise to the spec-mandated wire shape.

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
    ElicitationResponse, HttpMcpClient, McpClient, McpServerConfig, StaticElicitationProvider,
};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SESSION_ID: &str = "session-elicit-e2e";

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

fn server_initiated_elicit_request(id: i64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "elicitation/create",
        "params": {
            "message": "Please confirm",
            "requestedSchema": {
                "type": "object",
                "properties": { "confirm": { "type": "boolean" } },
                "required": ["confirm"],
            }
        },
    })
}

async fn mount_initialize_with_elicit(server: &MockServer, request_id: i64) {
    let body = sse_body(vec![
        initialize_result_event(1),
        server_initiated_elicit_request(request_id),
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
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(server)
        .await;
}

#[tokio::test]
async fn server_initiated_elicit_dispatches_to_accept_provider() {
    let server = MockServer::start().await;
    mount_initialize_with_elicit(&server, 901).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;
    mount_catchall_ok(&server).await;

    let provider = Arc::new(StaticElicitationProvider::accept(json!({"confirm": true})));
    let config = McpServerConfig::http("mock", server.uri())
        .unwrap()
        .with_elicitation_provider(provider);
    let client = HttpMcpClient::new(config).unwrap();
    let _ = client.initialize().await.unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let received = server.received_requests().await.unwrap();
    let server_response = received
        .iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("id").and_then(Value::as_i64) == Some(901) && v.get("result").is_some())
        .expect("client must POST a JSON-RPC response for elicitation/create");

    let result = server_response
        .pointer("/result")
        .expect("response must carry a result");
    assert_eq!(result["action"], "accept");
    assert_eq!(result["content"]["confirm"], true);
}

#[tokio::test]
async fn server_initiated_elicit_decline_serializes_action_only() {
    let server = MockServer::start().await;
    mount_initialize_with_elicit(&server, 902).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;
    mount_catchall_ok(&server).await;

    let provider = Arc::new(StaticElicitationProvider::decline());
    let config = McpServerConfig::http("mock", server.uri())
        .unwrap()
        .with_elicitation_provider(provider);
    let client = HttpMcpClient::new(config).unwrap();
    let _ = client.initialize().await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let received = server.received_requests().await.unwrap();
    let response = received
        .iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("id").and_then(Value::as_i64) == Some(902) && v.get("result").is_some())
        .expect("client must POST a JSON-RPC response for elicitation/create");
    let result = response.pointer("/result").unwrap();
    assert_eq!(result["action"], "decline");
    assert!(
        result.get("content").is_none(),
        "decline must not carry content"
    );
}

#[tokio::test]
async fn server_initiated_elicit_returns_method_not_found_when_provider_absent() {
    let server = MockServer::start().await;
    mount_initialize_with_elicit(&server, 903).await;
    mount_initialized_notification(&server).await;
    mount_tools_list_empty(&server).await;
    mount_catchall_ok(&server).await;

    let config = McpServerConfig::http("mock", server.uri()).unwrap();
    let client = HttpMcpClient::new(config).unwrap();
    let _ = client.initialize().await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let received = server.received_requests().await.unwrap();
    let error_response = received
        .iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("id").and_then(Value::as_i64) == Some(903) && v.get("error").is_some())
        .expect("client must POST a JSON-RPC error for unsupported elicitation/create");
    assert_eq!(error_response["error"]["code"], -32601);
}

#[tokio::test]
async fn capability_advertised_only_when_provider_wired() {
    let server_with = MockServer::start().await;
    mount_initialize_with_elicit(&server_with, 904).await;
    mount_initialized_notification(&server_with).await;
    mount_tools_list_empty(&server_with).await;
    mount_catchall_ok(&server_with).await;

    let provider = Arc::new(StaticElicitationProvider::cancel());
    let with_config = McpServerConfig::http("with", server_with.uri())
        .unwrap()
        .with_elicitation_provider(provider);
    let with_client = HttpMcpClient::new(with_config).unwrap();
    let _ = with_client.initialize().await.unwrap();

    let server_without = MockServer::start().await;
    mount_initialize_with_elicit(&server_without, 905).await;
    mount_initialized_notification(&server_without).await;
    mount_tools_list_empty(&server_without).await;
    mount_catchall_ok(&server_without).await;
    let without_config = McpServerConfig::http("without", server_without.uri()).unwrap();
    let without_client = HttpMcpClient::new(without_config).unwrap();
    let _ = without_client.initialize().await.unwrap();

    let with_init = server_with
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("method") == Some(&Value::String("initialize".into())))
        .expect("initialize observed");
    let without_init = server_without
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
        .find(|v| v.get("method") == Some(&Value::String("initialize".into())))
        .expect("initialize observed");

    let with_caps = with_init
        .pointer("/params/capabilities")
        .expect("capabilities present");
    assert!(
        with_caps.get("elicitation").is_some(),
        "elicitation capability advertised when provider wired"
    );
    let without_caps = without_init
        .pointer("/params/capabilities")
        .expect("capabilities present");
    assert!(
        without_caps.get("elicitation").is_none(),
        "elicitation capability omitted when no provider wired"
    );

    // Quiet the unused-binding lint.
    drop(ElicitationResponse::Cancel);
}
