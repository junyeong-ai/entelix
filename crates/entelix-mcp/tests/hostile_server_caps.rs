//! Resource-bound regression suite for `HttpMcpClient` against a
//! hostile or malfunctioning MCP server. Pairs with the inline
//! concurrency-cap unit tests in `client.rs::tests`.
//!
//! Covers:
//!
//! 1. **Frame size cap** — the SSE listener drops the connection
//!    (and `consume_sse_response` returns `McpError::Malformed`)
//!    when an unbounded frame exceeds
//!    [`McpServerConfig::max_frame_bytes`]. Without this gate, an
//!    adversarial server can stream bytes forever without a
//!    `\n\n` terminator and pin the client's heap.
//! 2. **Sane traffic stays on the happy path** — a regular SSE
//!    response that fits inside the configured cap deserializes
//!    normally. Verifies the cap doesn't false-trip on legitimate
//!    traffic.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_const_for_fn,
    clippy::indexing_slicing
)]

use entelix_mcp::{HttpMcpClient, McpClient, McpError, McpServerConfig, ResourceBoundKind};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn frame_cap_rejects_unterminated_oversize_sse_frame() {
    let server = MockServer::start().await;

    // 4 KiB of `x` with no `\n\n` terminator, served as
    // `text/event-stream`. The client's frame buffer keeps growing
    // until the cap (1 KiB here) trips.
    let oversize_payload: Vec<u8> = vec![b'x'; 4 * 1024];
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(oversize_payload, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let config = McpServerConfig::http("hostile", server.uri())
        .unwrap()
        .with_max_frame_bytes(1024);
    let client = HttpMcpClient::new(config).unwrap();

    let err = client.initialize().await.unwrap_err();
    match err {
        McpError::ResourceBounded { kind, message } => {
            assert_eq!(kind, ResourceBoundKind::FrameSize);
            assert!(
                message.contains("exceeded") && message.contains("terminator"),
                "expected frame-cap diagnostic, got: {message}"
            );
        }
        other => panic!("expected McpError::ResourceBounded; got {other:?}"),
    }
}

#[tokio::test]
async fn frame_cap_does_not_false_trip_on_legitimate_sse_response() {
    let server = MockServer::start().await;

    // A well-formed SSE-framed JSON-RPC initialize response. The
    // total body is well under the default 1 MiB cap.
    let body = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":\
        {\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\
        \"serverInfo\":{\"name\":\"mock\",\"version\":\"0.0.1\"}}}\n\n";

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |req: &wiremock::Request| {
            let body_json: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
            let m = body_json
                .get("method")
                .and_then(|m| m.as_str())
                .unwrap_or("");
            match m {
                "initialize" => ResponseTemplate::new(200)
                    .set_body_raw(body.as_bytes().to_vec(), "text/event-stream"),
                "notifications/initialized" => ResponseTemplate::new(200).set_body_string(""),
                "tools/list" => ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0",
                    "id": body_json.get("id"),
                    "result": { "tools": [] }
                })),
                _ => ResponseTemplate::new(404),
            }
        })
        .mount(&server)
        .await;

    let config = McpServerConfig::http("legit", server.uri()).unwrap();
    let client = HttpMcpClient::new(config).unwrap();
    let tools = client.initialize().await.unwrap();
    assert!(tools.is_empty());
}
