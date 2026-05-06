//! `13_mcp_tools` — MCP-published tool through the entelix `Tool`
//! adapter, with F9 (per-tenant pool isolation) on display.
//!
//! Build: `cargo build --example 13_mcp_tools -p entelix`
//! Run:   `cargo run   --example 13_mcp_tools -p entelix`
//!
//! Wire shape:
//!
//! 1. A `wiremock` server stands in for an MCP HTTP endpoint. It
//!    speaks JSON-RPC 2.0 (`initialize`, `tools/list`, `tools/call`)
//!    against the 2024-11-05 MCP revision.
//! 2. `McpManager::builder().register(...)` records the server config.
//!    No connection opens until first use (lazy provisioning).
//! 3. We list the tools for tenant `alpha` (one connect), then for
//!    tenant `bravo` (a *separate* connect — F9 isolation: same server
//!    URL, different tenant scopes get independent clients with
//!    independent bearer tokens / lifecycles).
//! 4. We dispatch the same tool from both tenants through
//!    `McpToolAdapter`, which implements `entelix_core::tools::Tool` so
//!    the rest of the SDK (ReAct agents, Runnable graphs) treats MCP
//!    tools indistinguishably from first-party tools.
//!
//! Deterministic — no real MCP server needed.

#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::doc_markdown
)]

use entelix::{AgentContext, ExecutionContext};
use entelix::tools::Tool;
use entelix::{McpManager, McpServerConfig, McpToolAdapter, TenantId};
use serde_json::json;
use wiremock::matchers::method;
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
                    "serverInfo": { "name": "demo-mcp", "version": "0.0.1" }
                }
            })),
            "notifications/initialized" => ResponseTemplate::new(200).set_body_string(""),
            "tools/list" => ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [{
                        "name": "lookup",
                        "description": "Look up a record by id.",
                        "inputSchema": {
                            "type": "object",
                            "properties": { "id": { "type": "string" } },
                            "required": ["id"]
                        }
                    }]
                }
            })),
            "tools/call" => {
                let args = body
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let id_arg = args.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": format!("record({id_arg}) → ok")
                        }],
                        "isError": false
                    }
                }))
            }
            _ => ResponseTemplate::new(404).set_body_string("unknown method"),
        }
    }
}

#[tokio::main]
async fn main() -> entelix::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(responder())
        .mount(&server)
        .await;

    let manager = McpManager::builder()
        .register(McpServerConfig::http("records", server.uri()).map_err(entelix::Error::from)?)
        .build()
        .map_err(entelix::Error::from)?;

    println!("manager built — server registered, no connection yet.");
    println!(
        "registered: {:?}",
        manager.server_names().collect::<Vec<_>>()
    );

    // ── tenant alpha ─────────────────────────────────────────────────
    let ctx_alpha = ExecutionContext::new().with_tenant_id(TenantId::new("alpha"));
    let tools_alpha = manager
        .list_tools(&ctx_alpha, "records")
        .await
        .map_err(entelix::Error::from)?;
    println!(
        "\nalpha sees {} tool(s) on 'records': {:?}",
        tools_alpha.len(),
        tools_alpha.iter().map(|t| &t.name).collect::<Vec<_>>(),
    );

    // ── tenant bravo (independent client — F9) ───────────────────────
    let ctx_bravo = ExecutionContext::new().with_tenant_id(TenantId::new("bravo"));
    let tools_bravo = manager
        .list_tools(&ctx_bravo, "records")
        .await
        .map_err(entelix::Error::from)?;
    println!(
        "bravo sees {} tool(s) on 'records': {:?}",
        tools_bravo.len(),
        tools_bravo.iter().map(|t| &t.name).collect::<Vec<_>>(),
    );

    // ── adapt the MCP tool into the entelix Tool trait ───────────────
    let lookup_alpha = McpToolAdapter::new(manager.clone(), "records", tools_alpha[0].clone());
    let lookup_bravo = McpToolAdapter::new(manager.clone(), "records", tools_bravo[0].clone());

    let agent_ctx_alpha = AgentContext::<()>::from(ctx_alpha.clone());
    let agent_ctx_bravo = AgentContext::<()>::from(ctx_bravo.clone());
    let alpha_result = lookup_alpha
        .execute(json!({ "id": "alpha-001" }), &agent_ctx_alpha)
        .await?;
    let bravo_result = lookup_bravo
        .execute(json!({ "id": "bravo-001" }), &agent_ctx_bravo)
        .await?;

    println!("\nalpha tool call result: {alpha_result}");
    println!("bravo tool call result: {bravo_result}");

    println!("\n   ✓ Two tenants share one MCP server config but route through");
    println!("     two independent McpClient instances — F9 mitigation.");
    Ok(())
}
