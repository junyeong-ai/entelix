//! `HttpMcpClient` end-to-end via `wiremock`. Exercises the real
//! JSON-RPC envelope:
//!
//! 1. `initialize` returns capabilities.
//! 2. `notifications/initialized` is sent (no response expected).
//! 3. `tools/list` returns one tool.
//! 4. `tools/call` returns text content; the adapter joins multipart
//!    blocks with newlines.
//!
//! The mock server is intentionally permissive on `id` matching — MCP
//! lets the client choose ids freely.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_const_for_fn,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;

use entelix_core::context::ExecutionContext;
use entelix_mcp::{
    HttpMcpClient, McpClient, McpClientState, McpCompletionArgument, McpCompletionReference,
    McpManager, McpPromptContent, McpResourceContent, McpServerConfig, McpToolAdapter,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn responder() -> impl Fn(&Request) -> ResponseTemplate + Send + Sync + 'static {
    |req: &Request| {
        let body: serde_json::Value =
            serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
        let method = body.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = body.get("id").cloned();
        match method {
            "initialize" => ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": { "name": "mock", "version": "0.0.1" }
                }
            })),
            "notifications/initialized" => ResponseTemplate::new(200).set_body_string(""),
            "tools/list" => ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [{
                        "name": "echo",
                        "description": "echo back input",
                        "inputSchema": { "type": "object" }
                    }]
                }
            })),
            "tools/call" => {
                let args = body
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [
                            { "type": "text", "text": format!("you said: {args}") }
                        ],
                        "isError": false
                    }
                }))
            }
            "resources/list" => ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "resources": [{
                        "uri": "file:///etc/hosts",
                        "name": "hosts",
                        "description": "system hosts file",
                        "mimeType": "text/plain"
                    }]
                }
            })),
            "resources/read" => {
                let uri = body
                    .pointer("/params/uri")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "contents": [{
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": "127.0.0.1 localhost"
                        }]
                    }
                }))
            }
            "prompts/list" => ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "prompts": [{
                        "name": "greet",
                        "description": "say hello",
                        "arguments": [{ "name": "who", "required": true }]
                    }]
                }
            })),
            "prompts/get" => {
                let who = body
                    .pointer("/params/arguments/who")
                    .and_then(|v| v.as_str())
                    .unwrap_or("world")
                    .to_owned();
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "description": "greeting transcript",
                        "messages": [{
                            "role": "user",
                            "content": { "type": "text", "text": format!("hello {who}") }
                        }]
                    }
                }))
            }
            "completion/complete" => {
                let value = body
                    .pointer("/params/argument/value")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "completion": {
                            "values": [format!("{value}-alpha"), format!("{value}-bravo")],
                            "total": 2,
                            "hasMore": false
                        }
                    }
                }))
            }
            _ => ResponseTemplate::new(404).set_body_string("unknown method"),
        }
    }
}

#[tokio::test]
async fn http_client_initialize_caches_tools_and_advances_to_ready() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let client = HttpMcpClient::new(McpServerConfig::http("mock", server.uri()).unwrap()).unwrap();
    assert_eq!(client.state(), McpClientState::Queued);

    let tools_a = client.initialize().await.unwrap();
    assert_eq!(client.state(), McpClientState::Ready);
    assert_eq!(tools_a.len(), 1);
    assert_eq!(tools_a[0].name, "echo");

    let tools_b = client.initialize().await.unwrap();
    assert_eq!(
        tools_b, tools_a,
        "second call must hit the cached tool list"
    );
}

#[tokio::test]
async fn http_client_call_tool_joins_text_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let client = HttpMcpClient::new(McpServerConfig::http("mock", server.uri()).unwrap()).unwrap();
    let out = client
        .call_tool("echo", json!({ "msg": "hi" }))
        .await
        .unwrap();
    assert_eq!(out, json!("you said: {\"msg\":\"hi\"}"));
    assert_eq!(client.state(), McpClientState::Ready);
}

#[tokio::test]
async fn manager_routes_tool_call_through_adapter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("mock", server.uri()).unwrap())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_tenant_id("t1");

    let tools = manager.list_tools(&ctx, "mock").await.unwrap();
    let adapter = McpToolAdapter::new(manager.clone(), "mock", tools[0].clone());
    use entelix_core::tools::Tool;
    // Default namespacing surfaces the tool as `mcp:{server}:{tool}`
    // so two MCP servers can both export `echo` without colliding
    // in the same `ToolRegistry`.
    assert_eq!(adapter.metadata().name, "mcp:mock:echo");
    assert_eq!(adapter.mcp_tool_name(), "echo");

    let out = adapter.execute(json!({ "x": 1 }), &ctx).await.unwrap();
    assert_eq!(out, json!("you said: {\"x\":1}"));

    // Opt-out path for single-server deployments.
    let unq =
        McpToolAdapter::new(manager.clone(), "mock", tools[0].clone()).with_unqualified_name();
    assert_eq!(unq.metadata().name, "echo");
}

#[tokio::test]
async fn http_client_surfaces_jsonrpc_error_payload() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "initialize"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": { "code": -32601, "message": "method not found" }
        })))
        .mount(&server)
        .await;

    let client = HttpMcpClient::new(McpServerConfig::http("mock", server.uri()).unwrap()).unwrap();
    let err = client.initialize().await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("-32601"), "{msg}");
    assert_eq!(client.state(), McpClientState::Failed);
}

#[tokio::test]
async fn http_client_surfaces_non_2xx_as_network_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(503).set_body_string("upstream down"))
        .mount(&server)
        .await;

    let client = HttpMcpClient::new(McpServerConfig::http("mock", server.uri()).unwrap()).unwrap();
    let err = client.initialize().await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("503"), "{msg}");
}

#[tokio::test]
async fn tool_call_with_is_error_and_empty_content_surfaces_error() {
    // MCP spec lets a tool surface `is_error: true` independent of
    // `content` length — an empty content[] with the error flag
    // must NOT be silently treated as success. Without this guard
    // the adapter would return Value::String("") on tool failure
    // and downstream agents would assume the tool ran cleanly.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|req: &Request| {
            let body: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
            let method = body.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let id = body.get("id").cloned();
            match method {
                "initialize" => ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": { "name": "mock", "version": "0.0.1" }
                    }
                })),
                "notifications/initialized" => ResponseTemplate::new(200).set_body_string(""),
                "tools/list" => ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": { "tools": [{
                        "name": "broken",
                        "description": "always errors with empty content",
                        "inputSchema": { "type": "object" }
                    }]}
                })),
                "tools/call" => ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": { "content": [], "isError": true }
                })),
                _ => ResponseTemplate::new(404).set_body_string("unknown method"),
            }
        })
        .mount(&server)
        .await;

    let client = HttpMcpClient::new(McpServerConfig::http("mock", server.uri()).unwrap()).unwrap();
    let err = client
        .call_tool("broken", json!({}))
        .await
        .expect_err("is_error: true must surface as Err even with empty content");
    let msg = err.to_string();
    assert!(
        msg.contains("broken") || msg.contains("is_error"),
        "expected error message to identify the failing tool, got: {msg}"
    );
}

#[tokio::test]
async fn http_client_truncates_oversized_error_body_in_message() {
    // A misbehaving MCP server returning a multi-MB error body
    // must NOT inflate the resulting `McpError::Network` message.
    // Verifies log-pollution + format!-allocation DOS surface is
    // capped at the SDK boundary.
    let server = MockServer::start().await;
    let huge_body = "x".repeat(100_000);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(500).set_body_string(huge_body))
        .mount(&server)
        .await;

    let client = HttpMcpClient::new(McpServerConfig::http("mock", server.uri()).unwrap()).unwrap();
    let err = client.initialize().await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.len() < 2_000,
        "error message must be truncated, got {} bytes",
        msg.len()
    );
    assert!(
        msg.contains("truncated"),
        "expected truncation marker, got: {msg}"
    );
    assert!(msg.contains("500"), "{msg}");
}

#[tokio::test]
async fn request_decorator_panic_is_isolated_to_mcp_error() {
    // A buggy operator-supplied decorator that panics must NOT
    // tear down the async task — the SDK catches the unwind and
    // surfaces it as McpError::Config so the caller can react
    // without losing the rest of the agent run.
    use std::sync::Arc;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let decorator: entelix_mcp::RequestDecorator =
        Arc::new(|_: &mut reqwest::header::HeaderMap| {
            panic!("simulated operator-side bug");
        });

    let cfg = McpServerConfig::http("mock", server.uri())
        .unwrap()
        .with_request_decorator(decorator);
    let client = HttpMcpClient::new(cfg).unwrap();
    let err = client.initialize().await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("decorator panicked") && msg.contains("simulated"),
        "expected panic-isolation McpError, got: {msg}"
    );
}

#[tokio::test]
async fn request_decorator_injects_extra_headers_per_request() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::header;

    let server = MockServer::start().await;
    // Match only requests carrying the custom header — if the
    // decorator failed to fire, the mock returns 404 and the test
    // sees a Network error.
    Mock::given(method("POST"))
        .and(header("x-traceparent-stub", "00-abc-def-01"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_decorator = Arc::clone(&calls);
    let decorator: entelix_mcp::RequestDecorator =
        Arc::new(move |headers: &mut reqwest::header::HeaderMap| {
            calls_for_decorator.fetch_add(1, Ordering::SeqCst);
            headers.insert(
                reqwest::header::HeaderName::from_static("x-traceparent-stub"),
                reqwest::header::HeaderValue::from_static("00-abc-def-01"),
            );
        });

    let cfg = McpServerConfig::http("mock", server.uri())
        .unwrap()
        .with_request_decorator(decorator);
    let client = HttpMcpClient::new(cfg).unwrap();
    let _ = client.initialize().await.unwrap();
    // initialize fires `initialize` + `notifications/initialized` + `tools/list`
    // — decorator runs on every outbound request.
    assert!(
        calls.load(Ordering::SeqCst) >= 3,
        "decorator should fire on every request (saw {})",
        calls.load(Ordering::SeqCst)
    );
}

// ── MCP 1.5 full coverage: resources / prompts / completion ────────────────

#[tokio::test]
async fn manager_list_resources_round_trips() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("mock", server.uri()).unwrap())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_tenant_id("t1");

    let resources = manager.list_resources(&ctx, "mock").await.unwrap();
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0].uri, "file:///etc/hosts");
    assert_eq!(resources[0].name, "hosts");
    assert_eq!(resources[0].mime_type.as_deref(), Some("text/plain"));
}

#[tokio::test]
async fn manager_read_resource_round_trips() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("mock", server.uri()).unwrap())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_tenant_id("t1");

    let blocks = manager
        .read_resource(&ctx, "mock", "file:///etc/hosts")
        .await
        .unwrap();
    assert_eq!(blocks.len(), 1);
    let McpResourceContent::Text { uri, text, .. } = &blocks[0] else {
        panic!("expected text content, got {:?}", blocks[0]);
    };
    assert_eq!(uri, "file:///etc/hosts");
    assert_eq!(text, "127.0.0.1 localhost");
}

#[tokio::test]
async fn manager_list_prompts_round_trips() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("mock", server.uri()).unwrap())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_tenant_id("t1");

    let prompts = manager.list_prompts(&ctx, "mock").await.unwrap();
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0].name, "greet");
    assert_eq!(prompts[0].arguments.len(), 1);
    assert!(prompts[0].arguments[0].required);
}

#[tokio::test]
async fn manager_get_prompt_round_trips() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("mock", server.uri()).unwrap())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_tenant_id("t1");

    let mut args = BTreeMap::new();
    args.insert("who".into(), "alice".into());
    let invocation = manager
        .get_prompt(&ctx, "mock", "greet", args)
        .await
        .unwrap();
    assert_eq!(
        invocation.description.as_deref(),
        Some("greeting transcript")
    );
    assert_eq!(invocation.messages.len(), 1);
    assert_eq!(invocation.messages[0].role, "user");
    let McpPromptContent::Text { text } = &invocation.messages[0].content else {
        panic!("expected text content");
    };
    assert_eq!(text, "hello alice");
}

#[tokio::test]
async fn manager_complete_round_trips() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("mock", server.uri()).unwrap())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_tenant_id("t1");

    let result = manager
        .complete(
            &ctx,
            "mock",
            McpCompletionReference::Prompt {
                name: "greet".into(),
            },
            McpCompletionArgument {
                name: "who".into(),
                value: "al".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        result.values,
        vec!["al-alpha".to_owned(), "al-bravo".to_owned()]
    );
    assert_eq!(result.total, Some(2));
    assert!(!result.has_more);
}

#[tokio::test]
async fn manager_correlates_jsonrpc_error_with_server_and_op() {
    // Server that returns a JSON-RPC error for resources/read so the
    // correlate() path is exercised — we verify the manager prefixes
    // the message with `(server, op)` so multi-server agent runs can
    // triage which dispatch failed.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|req: &Request| {
            let body: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
            let method = body.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let id = body.get("id").cloned();
            match method {
                "initialize" => ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": { "name": "mock", "version": "0.0.1" }
                    }
                })),
                "notifications/initialized" => ResponseTemplate::new(200).set_body_string(""),
                "tools/list" => ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": id, "result": { "tools": [] }
                })),
                "resources/read" => ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32002, "message": "resource not found" }
                })),
                _ => ResponseTemplate::new(404).set_body_string("unknown method"),
            }
        })
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("mock", server.uri()).unwrap())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new().with_tenant_id("t1");

    let err = manager
        .read_resource(&ctx, "mock", "file:///missing")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("mock"), "{msg}");
    assert!(msg.contains("file:///missing"), "{msg}");
    assert!(msg.contains("resource not found"), "{msg}");
}
