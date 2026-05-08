//! Pin invariant 16 for `McpError`: the LLM-facing rendering never
//! includes operator-only diagnostics — no source chains, no
//! tenant identifiers, no raw vendor messages, no malformed-body
//! contents. Each variant gets a static or identifier-only
//! sentence; all operator detail stays on `Display` / `Error::source`
//! / tracing.

use entelix_core::LlmRenderable;
use entelix_mcp::{McpError, ResourceBoundKind};
use std::io;

#[test]
fn network_variant_does_not_leak_source_chain() {
    let inner = io::Error::new(io::ErrorKind::ConnectionReset, "raw operator detail");
    let err = McpError::network(inner);
    let rendering = err.for_llm().into_inner();
    assert_eq!(rendering, "MCP transport failure");
    assert!(!rendering.contains("raw operator detail"));
    assert!(!rendering.contains("ConnectionReset"));
}

#[test]
fn json_rpc_variant_keeps_code_drops_message() {
    let err = McpError::JsonRpc {
        code: -32603,
        message: "internal vendor stack frame".to_owned(),
    };
    let rendering = err.for_llm().into_inner();
    assert_eq!(rendering, "MCP server returned error code -32603");
    assert!(!rendering.contains("internal vendor stack frame"));
}

#[test]
fn malformed_response_does_not_leak_body() {
    let err = McpError::malformed_msg("{\"secret\":\"value\"}");
    let rendering = err.for_llm().into_inner();
    assert_eq!(rendering, "MCP malformed response");
    assert!(!rendering.contains("secret"));
    assert!(!rendering.contains("value"));
}

#[test]
fn unknown_server_does_not_leak_tenant_id() {
    let err = McpError::UnknownServer {
        tenant_id: "tenant-42-private".to_owned(),
        server: "primary".to_owned(),
    };
    let rendering = err.for_llm().into_inner();
    assert_eq!(rendering, "MCP server 'primary' not registered");
    assert!(!rendering.contains("tenant-42-private"));
}

#[test]
fn config_variant_does_not_leak_raw_message() {
    let err = McpError::Config("DSN=postgres://user:pass@host/db".to_owned());
    let rendering = err.for_llm().into_inner();
    assert_eq!(rendering, "MCP misconfigured");
    assert!(!rendering.contains("postgres"));
    assert!(!rendering.contains("pass"));
}

#[test]
fn unreachable_keeps_server_name_drops_attempts_detail() {
    let err = McpError::Unreachable {
        server: "primary".to_owned(),
        attempts: 7,
    };
    let rendering = err.for_llm().into_inner();
    assert_eq!(rendering, "MCP server 'primary' unreachable");
}

#[test]
fn resource_bounded_keeps_kind_drops_message() {
    let err = McpError::ResourceBounded {
        kind: ResourceBoundKind::FrameSize,
        message: "frame=2 048 152 bytes from peer 198.51.100.7".to_owned(),
    };
    let rendering = err.for_llm().into_inner();
    assert_eq!(rendering, "MCP listener bound exceeded (frame_size)");
    assert!(!rendering.contains("198.51.100.7"));
    assert!(!rendering.contains("2 048 152"));
}
