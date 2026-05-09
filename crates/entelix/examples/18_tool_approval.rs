//! `18_tool_approval` — HITL approval flow at the tool-dispatch
//! boundary, demonstrating the full pause-and-resume cycle.
//!
//! Build: `cargo build --example 18_tool_approval -p entelix --features policy`
//! Run:   `cargo run   --example 18_tool_approval -p entelix --features policy`
//!
//! Demonstrates the operator-facing surface for HITL approval:
//!
//! 1. `Approver::decide` returns
//!    [`ApprovalDecision::AwaitExternal`] for a high-stakes tool
//!    call. `ApprovalLayer` raises
//!    `Error::Interrupted { payload }` with a structured
//!    `kind = "approval_pending"` so the agent
//!    run pauses cleanly and the operator gets the pending
//!    dispatch context for out-of-band review.
//!
//! 2. The operator inspects the payload, makes a decision through
//!    a separate channel (web UI, Slack, e-mail — the SDK is
//!    channel-agnostic), and resumes the dispatch with the typed
//!    [`Command::ApproveTool { tool_use_id, decision }`] resume
//!    primitive. The graph's resume path attaches the decision to
//!    `ExecutionContext` internally so the layer's override-lookup
//!    short-circuits the approver — the pending tool call
//!    dispatches normally (or short-circuits with the operator's
//!    Reject reason) and the agent completes.
//!
//! 3. Compare with `04_hitl.rs` which demonstrates the
//!    graph-internal `interrupt()` primitive — same checkpoint-
//!    and-bubble mechanism, different trigger surface
//!    (graph-node interrupt vs. tool-dispatch approval).
//!
//! Runs deterministically — no external API dependency.

#![allow(clippy::print_stdout, clippy::cast_precision_loss)]
// example output + tiny i64→f64 amount conversion for display formatting

use std::sync::Arc;

use async_trait::async_trait;
use entelix::tools::{Tool, ToolMetadata, ToolRegistry};
use entelix::{
    AgentContext, ApprovalDecision, ApprovalLayer, ApprovalRequest, Approver, Error,
    ExecutionContext, Result,
};
use serde_json::{Value, json};

/// A tool whose dispatch the operator wants to review before
/// allowing — generically named so the example reads as a
/// scaffolding pattern rather than a vertical-specific snippet.
struct SensitiveActionTool {
    metadata: ToolMetadata,
}

impl SensitiveActionTool {
    fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "process_payment",
                "Charge a customer's payment method.",
                json!({
                    "type": "object",
                    "properties": {
                        "customer_id": { "type": "string" },
                        "amount_cents": { "type": "integer" },
                    },
                    "required": ["customer_id", "amount_cents"],
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for SensitiveActionTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
        // Real impl would call the payment provider. The example
        // just echoes — the gating flow is the demonstration target.
        let customer = input
            .get("customer_id")
            .and_then(Value::as_str)
            .unwrap_or("(unspecified)");
        let amount = input
            .get("amount_cents")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        Ok(json!({
            "status": "executed",
            "customer_id": customer,
            "amount_cents": amount,
            "message": format!("would have charged customer {customer} ${:.2} (example only)", amount as f64 / 100.0),
        }))
    }
}

/// An approver that always defers to out-of-band review. In a
/// real deployment this would push the request to a queue
/// (Slack channel, web UI, e-mail) and return immediately —
/// the agent pause releases inflight resources while the human
/// reviews.
struct AlwaysAwaitExternal;

#[async_trait]
impl Approver for AlwaysAwaitExternal {
    async fn decide(
        &self,
        _request: &ApprovalRequest,
        _ctx: &ExecutionContext,
    ) -> Result<ApprovalDecision> {
        Ok(ApprovalDecision::AwaitExternal)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── Setup: registry with the sensitive-action tool, gated by
    //    the always-await approver via ApprovalLayer.
    let approver: Arc<dyn Approver> = Arc::new(AlwaysAwaitExternal);
    let registry = ToolRegistry::new()
        .layer(ApprovalLayer::new(Arc::clone(&approver)))
        .register(Arc::new(SensitiveActionTool::new()))?;

    let tool_use_id = "charge-customer-acme-2026-05-03".to_owned();
    let tool_input = json!({ "customer_id": "acme", "amount_cents": 4999 });

    println!("=== first dispatch — should pause for out-of-band review ===");
    match registry
        .dispatch(
            &tool_use_id,
            "process_payment",
            tool_input.clone(),
            &ExecutionContext::new(),
        )
        .await
    {
        Err(Error::Interrupted { kind, payload }) => {
            println!("agent paused for human review.");
            println!("kind: {kind:?}");
            println!("payload: {payload:#}");
        }
        Ok(_) => {
            println!("WARNING: expected an interrupt, the dispatch ran without approval");
            return Ok(());
        }
        Err(other) => {
            println!("unexpected error: {other}");
            return Ok(());
        }
    }

    println!();
    println!("=== operator reviews out-of-band and approves ===");
    println!(
        "(pretend we routed payload['tool_use_id'] to a Slack \
         #ops-approvals channel and got a thumbs-up emoji back)"
    );
    println!();
    println!("In a real agent built on `entelix::ReActAgentBuilder`, the");
    println!("resume call uses the typed `Command::ApproveTool` primitive:");
    println!();
    println!("    let final_state = compiled_graph");
    println!("        .resume_with(");
    println!("            Command::ApproveTool {{");
    println!("                tool_use_id: \"{tool_use_id}\".into(),");
    println!("                decision: ApprovalDecision::Approve,");
    println!("            }},");
    println!("            &ctx,");
    println!("        ).await?;");
    println!();
    println!("CompiledGraph::resume_with attaches the decision to the");
    println!("ExecutionContext internally; the dispatch re-fires from the");
    println!("checkpoint and the approval layer short-circuits the approver");
    println!("for this tool_use_id.");
    println!();
    println!("Below we simulate the same effect by attaching the decision");
    println!("directly (lower-level path — useful for non-graph dispatch).");

    // Lower-level simulation for the example: attach
    // PendingApprovalDecisions directly. In production code,
    // operators reach for `Command::ApproveTool` on the resume
    // call and never construct PendingApprovalDecisions
    // themselves — see the comment block above for the typed path.
    let mut pending = entelix::PendingApprovalDecisions::new();
    pending.insert(&tool_use_id, ApprovalDecision::Approve);
    let resume_ctx = ExecutionContext::new().add_extension(pending);

    println!();
    println!("=== second dispatch with the decision attached ===");
    let result = registry
        .dispatch(&tool_use_id, "process_payment", tool_input, &resume_ctx)
        .await?;
    println!("tool result: {result:#}");

    println!();
    println!("=== alternative: operator denies out-of-band ===");
    let mut denied = entelix::PendingApprovalDecisions::new();
    denied.insert(
        &tool_use_id,
        ApprovalDecision::Reject {
            reason: "amount exceeds operator-approval ceiling".to_owned(),
        },
    );
    let denied_ctx = ExecutionContext::new().add_extension(denied);
    match registry
        .dispatch(
            &tool_use_id,
            "process_payment",
            json!({ "customer_id": "acme", "amount_cents": 4999 }),
            &denied_ctx,
        )
        .await
    {
        Err(Error::InvalidRequest(msg)) => {
            println!("denial surfaced as typed error:");
            println!("  {msg}");
        }
        Ok(value) => println!("WARNING: expected denial, got {value:#}"),
        Err(other) => println!("unexpected error: {other}"),
    }

    Ok(())
}
