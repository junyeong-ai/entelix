//! `ApprovalLayer::with_effect_gate` — operator-declared
//! [`ToolEffect`] gate decides which dispatches reach the
//! `Approver`. Tools whose effect lies outside the gate
//! auto-approve without consulting the approver.
//!
//! These tests pin the design properties:
//!
//! - `EffectGate::Always` (default) → every dispatch hits the
//!   approver, including `ReadOnly`.
//! - `EffectGate::DestructiveOnly` → `ReadOnly` and `Mutating`
//!   auto-approve; `Destructive` reaches the approver.
//! - `EffectGate::MutatingAndAbove` → `ReadOnly` auto-approves;
//!   `Mutating` and `Destructive` reach the approver.
//! - Pending decisions (resume path) take precedence over the gate
//!   so a paused-then-narrowed flow still honours the operator's
//!   recorded answer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use entelix_agents::{ApprovalDecision, ApprovalLayer, ApprovalRequest, Approver, EffectGate};
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata, ToolRegistry};
use entelix_core::{
    AgentContext, ExecutionContext, PendingApprovalDecisions, Result, ToolErrorKind,
};

/// Counts every `decide` call so tests can assert whether the gate
/// reached the approver.
struct CountingApprover {
    decisions: AtomicUsize,
}

#[async_trait]
impl Approver for CountingApprover {
    async fn decide(
        &self,
        _request: &ApprovalRequest,
        _ctx: &ExecutionContext,
    ) -> Result<ApprovalDecision> {
        self.decisions.fetch_add(1, Ordering::SeqCst);
        Ok(ApprovalDecision::Approve)
    }
}

/// Tool whose only purpose is to advertise an explicit
/// [`ToolEffect`] so `EffectGate` routing can be exercised.
struct EffectTool {
    metadata: ToolMetadata,
}

impl EffectTool {
    fn new(name: &str, effect: ToolEffect) -> Self {
        Self {
            metadata: ToolMetadata::function(
                name,
                "effect-gate fixture",
                serde_json::json!({"type": "object"}),
            )
            .with_effect(effect),
        }
    }
}

#[async_trait]
impl Tool for EffectTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &AgentContext<()>,
    ) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"ok": true}))
    }
}

fn registry_with_layer(gate: EffectGate, approver: Arc<CountingApprover>) -> ToolRegistry {
    ToolRegistry::new()
        .layer(ApprovalLayer::new(approver).with_effect_gate(gate))
        .register(Arc::new(EffectTool::new("read", ToolEffect::ReadOnly)))
        .unwrap()
        .register(Arc::new(EffectTool::new("mutate", ToolEffect::Mutating)))
        .unwrap()
        .register(Arc::new(EffectTool::new(
            "destroy",
            ToolEffect::Destructive,
        )))
        .unwrap()
}

async fn dispatch(registry: &ToolRegistry, tool: &str) -> Result<serde_json::Value> {
    registry
        .dispatch(
            "tu_1",
            tool,
            serde_json::json!({}),
            &ExecutionContext::new(),
        )
        .await
}

#[tokio::test]
async fn always_gate_reaches_approver_for_every_effect() {
    let approver = Arc::new(CountingApprover {
        decisions: AtomicUsize::new(0),
    });
    let registry = registry_with_layer(EffectGate::Always, Arc::clone(&approver));
    dispatch(&registry, "read").await.unwrap();
    dispatch(&registry, "mutate").await.unwrap();
    dispatch(&registry, "destroy").await.unwrap();
    assert_eq!(
        approver.decisions.load(Ordering::SeqCst),
        3,
        "Always gate must consult the approver on every dispatch"
    );
}

#[tokio::test]
async fn destructive_only_gate_skips_read_and_mutate() {
    let approver = Arc::new(CountingApprover {
        decisions: AtomicUsize::new(0),
    });
    let registry = registry_with_layer(EffectGate::DestructiveOnly, Arc::clone(&approver));
    dispatch(&registry, "read").await.unwrap();
    dispatch(&registry, "mutate").await.unwrap();
    dispatch(&registry, "destroy").await.unwrap();
    assert_eq!(
        approver.decisions.load(Ordering::SeqCst),
        1,
        "DestructiveOnly must only gate Destructive — exactly one approver call"
    );
}

#[tokio::test]
async fn mutating_and_above_gate_skips_only_read_only() {
    let approver = Arc::new(CountingApprover {
        decisions: AtomicUsize::new(0),
    });
    let registry = registry_with_layer(EffectGate::MutatingAndAbove, Arc::clone(&approver));
    dispatch(&registry, "read").await.unwrap();
    dispatch(&registry, "mutate").await.unwrap();
    dispatch(&registry, "destroy").await.unwrap();
    assert_eq!(
        approver.decisions.load(Ordering::SeqCst),
        2,
        "MutatingAndAbove must gate Mutating + Destructive — two approver calls"
    );
}

#[tokio::test]
async fn pending_decision_overrides_gate_for_resume_path() {
    // Operator paused on a Destructive call, gate later narrowed
    // to DestructiveOnly. The resume path still honours the
    // recorded Reject — the gate must not auto-approve a tool
    // that already has a pending decision.
    let approver = Arc::new(CountingApprover {
        decisions: AtomicUsize::new(0),
    });
    let registry = ToolRegistry::new()
        .layer(
            ApprovalLayer::new(Arc::clone(&approver) as Arc<dyn Approver>)
                .with_effect_gate(EffectGate::DestructiveOnly),
        )
        .register(Arc::new(EffectTool::new("read", ToolEffect::ReadOnly)))
        .unwrap();
    let mut pending = PendingApprovalDecisions::new();
    pending.insert(
        "tu_resume",
        ApprovalDecision::Reject {
            reason: "operator denied during pause".to_owned(),
        },
    );
    let ctx = ExecutionContext::new().add_extension(pending);
    let err = registry
        .dispatch("tu_resume", "read", serde_json::json!({}), &ctx)
        .await
        .unwrap_err();
    // The recorded Reject surfaced — the layer did not silently
    // auto-approve through the effect gate.
    assert_ne!(
        ToolErrorKind::classify(&err),
        ToolErrorKind::Internal,
        "rejection surfaced as a typed failure, not as auto-approval"
    );
    assert_eq!(
        approver.decisions.load(Ordering::SeqCst),
        0,
        "approver consulted zero times — pending decision short-circuited before gate"
    );
}
