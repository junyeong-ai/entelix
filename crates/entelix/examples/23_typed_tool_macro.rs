//! `23_typed_tool_macro` — author a typed-input tool with the
//! `#[tool]` proc-macro and dispatch it through a `ToolRegistry`.
//!
//! Build: `cargo build --example 23_typed_tool_macro -p entelix`
//! Run (hermetic — no network):
//!     `cargo run --example 23_typed_tool_macro -p entelix`
//!
//! The `#[tool]` attribute generates the canonical
//! `entelix_tools::SchemaTool` boilerplate from one `async fn`:
//!
//! - an `Input` struct (`Deserialize + JsonSchema` over the
//!   parameter list),
//! - a `PascalCase` unit struct named after the function,
//! - an `impl SchemaTool` that deserialises the JSON input,
//!   dispatches to the original `fn`, and returns the typed
//!   output.
//!
//! Operators expose the tool to an agent by registering
//! `Tool.into_adapter()` on a `ToolRegistry`. The first
//! paragraph of the doc comment becomes the model-facing
//! description (further paragraphs stay as developer notes).

#![allow(clippy::print_stdout, clippy::unused_async)]

use std::sync::Arc;

use entelix::tools::ToolRegistry;
use entelix::{AgentContext, ExecutionContext, Result, SchemaToolExt, tool};
use serde_json::json;

/// Sum two integers.
///
/// Trailing paragraphs of the doc comment are not surfaced to
/// the model — they stay inline as developer documentation.
#[tool]
async fn add(_ctx: &AgentContext<()>, a: i64, b: i64) -> Result<i64> {
    Ok(a + b)
}

/// Square an integer. Demonstrates the metadata-args form —
/// `name` override, `effect`, `idempotent`, `version`,
/// `retry_hint` all flow into the generated `ToolMetadata`.
#[tool(
    name = "calc_square",
    effect = "ReadOnly",
    idempotent,
    version = "1.0.0",
    retry_hint = "idempotent_transport"
)]
async fn square(_ctx: &AgentContext<()>, n: i64) -> Result<i64> {
    Ok(n * n)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Register both tools on a single registry — same shape an
    // operator wires when handing tools to an agent or a
    // `Subagent::builder(.).restrict_to(&[…])` view.
    let registry = ToolRegistry::new()
        .register(Arc::new(Add.into_adapter()))?
        .register(Arc::new(Square.into_adapter()))?;

    let ctx = ExecutionContext::new();

    // Dispatch through the registry — the `Input` schema
    // validates the JSON before the typed body runs.
    let sum = registry
        .dispatch("call-1", "add", json!({"a": 7, "b": 35}), &ctx)
        .await?;
    println!("add(7, 35) = {sum}");

    let sq = registry
        .dispatch("call-2", "calc_square", json!({"n": 9}), &ctx)
        .await?;
    println!("calc_square(9) = {sq}");

    // Inspect the generated metadata — dashboards and
    // capabilities surfaces consume this.
    let square_adapter = Square.into_adapter();
    let meta = entelix::tools::Tool::metadata(&square_adapter);
    println!(
        "\ncalc_square metadata: name={}, version={:?}, idempotent={}",
        meta.name, meta.version, meta.idempotent
    );
    Ok(())
}
