//! `interrupt()` + `Command` HITL tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::TenantId;
use std::sync::Arc;

use entelix_core::ThreadKey;
use entelix_core::{Error, ExecutionContext, Result};
use entelix_graph::{Checkpointer, Command, InMemoryCheckpointer, StateGraph, interrupt};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Approval {
    request: String,
    approved: Option<bool>,
    reply: Option<String>,
}

/// Node that pauses and awaits a human decision the first time it runs.
fn review_node() -> RunnableLambda<Approval, Approval> {
    RunnableLambda::new(|s: Approval, _ctx| async move {
        if s.approved.is_none() {
            return interrupt(serde_json::json!({
                "request": s.request.clone(),
                "options": ["approve", "reject"],
            }));
        }
        Ok(s)
    })
}

fn finalize_node() -> RunnableLambda<Approval, Approval> {
    RunnableLambda::new(|mut s: Approval, _ctx| async move {
        s.reply = Some(if matches!(s.approved, Some(true)) {
            "proceeding".into()
        } else {
            "halted".into()
        });
        Ok::<_, _>(s)
    })
}

#[tokio::test]
async fn interrupt_returns_payload_to_caller() {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("review", review_node())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .with_checkpointer(cp.clone())
        .compile()
        .unwrap();

    let ctx = ExecutionContext::new().with_thread_id("approval-1");
    let initial = Approval {
        request: "Deploy to prod".into(),
        approved: None,
        reply: None,
    };

    let err = graph.invoke(initial, &ctx).await.unwrap_err();
    match err {
        Error::Interrupted { payload } => {
            assert_eq!(payload["request"], "Deploy to prod");
            assert!(payload["options"].is_array());
        }
        other => panic!("expected Interrupted, got {other:?}"),
    }
}

#[tokio::test]
async fn interrupt_writes_pre_node_checkpoint() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("review", review_node())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("approval-2");
    let initial = Approval {
        request: "Deploy to prod".into(),
        approved: None,
        reply: None,
    };
    let _ = graph.invoke(initial.clone(), &ctx).await;

    let key = ThreadKey::from_ctx(&ctx)?;
    let latest = cp.latest(&key).await?.unwrap();
    // Pre-node state preserved.
    assert_eq!(latest.state, initial);
    // Resume should re-run the interrupted node.
    assert_eq!(latest.next_node.as_deref(), Some("review"));
    Ok(())
}

#[tokio::test]
async fn resume_with_update_completes_the_workflow() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("review", review_node())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("approval-3");
    let initial = Approval {
        request: "Deploy to prod".into(),
        approved: None,
        reply: None,
    };
    // First invocation interrupts.
    let _ = graph.invoke(initial.clone(), &ctx).await;

    // Inject the human's "approve" decision and resume.
    let approved_state = Approval {
        approved: Some(true),
        ..initial
    };
    let final_state = graph
        .resume_with(Command::Update(approved_state), &ctx)
        .await?;

    assert_eq!(final_state.approved, Some(true));
    assert_eq!(final_state.reply.as_deref(), Some("proceeding"));
    Ok(())
}

#[tokio::test]
async fn resume_with_goto_skips_to_named_node() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("review", review_node())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("approval-4");
    let mut initial = Approval {
        request: "Hot-fix prod".into(),
        approved: None,
        reply: None,
    };
    let _ = graph.invoke(initial.clone(), &ctx).await; // interrupts

    // Caller: skip review entirely, but keep approved = Some(false) so
    // finalize emits "halted".
    initial.approved = Some(false);
    let key = entelix_core::ThreadKey::new(TenantId::new("default"), "approval-4");
    cp.put(entelix_graph::Checkpoint::new(
        &key,
        99,
        initial,
        Some("review".into()),
    ))
    .await?;

    let out = graph
        .resume_with(Command::GoTo("finalize".into()), &ctx)
        .await?;
    assert_eq!(out.reply.as_deref(), Some("halted"));
    Ok(())
}

#[tokio::test]
async fn resume_after_clean_termination_returns_state_unchanged() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("review", review_node())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("approval-5");
    let initial = Approval {
        request: "x".into(),
        approved: Some(true), // skip interrupt
        reply: None,
    };
    let final_state = graph.invoke(initial, &ctx).await?;
    assert_eq!(final_state.reply.as_deref(), Some("proceeding"));

    // resume should now see a terminated checkpoint (next_node = None) and
    // return the state without re-running anything.
    let again = graph.resume_with(Command::Resume, &ctx).await?;
    assert_eq!(again, final_state);
    Ok(())
}

#[tokio::test]
async fn interrupt_does_not_persist_when_no_thread_id() {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("review", review_node())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .with_checkpointer(cp.clone())
        .compile()
        .unwrap();

    // ExecutionContext::new() has no thread_id.
    let initial = Approval {
        request: "x".into(),
        approved: None,
        reply: None,
    };
    let _ = graph.invoke(initial, &ExecutionContext::new()).await;
    assert_eq!(cp.total_checkpoints(), 0);
}

#[tokio::test]
async fn interrupt_before_pauses_without_invoking_node() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let invocations = Arc::new(AtomicUsize::new(0));
    let invocations_inner = invocations.clone();
    let body = RunnableLambda::new(move |mut s: Approval, _ctx| {
        let invocations = invocations_inner.clone();
        async move {
            invocations.fetch_add(1, Ordering::SeqCst);
            s.reply = Some("ran".into());
            Ok::<_, _>(s)
        }
    });

    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("review", body.clone())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .interrupt_before(["review"])
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("intrpt-before");
    let initial = Approval {
        request: "x".into(),
        approved: None,
        reply: None,
    };

    let err = graph.invoke(initial.clone(), &ctx).await.unwrap_err();
    match err {
        Error::Interrupted { payload } => {
            assert_eq!(payload["kind"], "before");
            assert_eq!(payload["node"], "review");
        }
        other => panic!("expected Interrupted, got {other:?}"),
    }
    // Node body must NOT have run.
    assert_eq!(invocations.load(Ordering::SeqCst), 0);

    // Resume continues — node now runs and graph terminates.
    let key = ThreadKey::from_ctx(&ctx)?;
    let cp_count_before_resume = cp.total_checkpoints();
    assert!(cp_count_before_resume >= 1);
    let _ = key; // satisfy clippy unused — just smoke

    let final_state = graph.resume_with(Command::Resume, &ctx).await?;
    assert_eq!(final_state.reply.as_deref(), Some("halted"));
    assert_eq!(invocations.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn interrupt_after_pauses_after_node_completes() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let pre_finalize_node = RunnableLambda::new(|mut s: Approval, _ctx| async move {
        s.approved = Some(true);
        Ok::<_, _>(s)
    });
    let graph = StateGraph::<Approval>::new()
        .add_node("preflight", pre_finalize_node)
        .add_node("finalize", finalize_node())
        .add_edge("preflight", "finalize")
        .set_entry_point("preflight")
        .add_finish_point("finalize")
        .interrupt_after(["preflight"])
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("intrpt-after");
    let initial = Approval {
        request: "x".into(),
        approved: None,
        reply: None,
    };

    let err = graph.invoke(initial.clone(), &ctx).await.unwrap_err();
    match err {
        Error::Interrupted { payload } => {
            assert_eq!(payload["kind"], "after");
            assert_eq!(payload["node"], "preflight");
        }
        other => panic!("expected Interrupted, got {other:?}"),
    }

    // Resume jumps to the next node (finalize) without re-running
    // preflight — the post-state was already captured.
    let final_state = graph.resume_with(Command::Resume, &ctx).await?;
    assert_eq!(final_state.reply.as_deref(), Some("proceeding"));
    Ok(())
}

#[tokio::test]
async fn approve_tool_command_attaches_pending_approval_decisions() -> Result<()> {
    // Verify `Command::ApproveTool` writes the operator's decision
    // into the resumed dispatch's ctx via `PendingApprovalDecisions`
    // (ADR-0072). A node that reads the extension must observe the
    // decision after resume.
    use entelix_core::{ApprovalDecision, PendingApprovalDecisions};

    let observed: Arc<parking_lot::Mutex<Option<ApprovalDecision>>> =
        Arc::new(parking_lot::Mutex::new(None));
    let observed_clone = Arc::clone(&observed);

    let observe_node =
        entelix_runnable::RunnableLambda::new(move |s: Approval, ctx: ExecutionContext| {
            let observed = Arc::clone(&observed_clone);
            async move {
                if let Some(pending) = ctx.extension::<PendingApprovalDecisions>() {
                    *observed.lock() = pending.get("tu-7").cloned();
                }
                Ok::<_, _>(s)
            }
        });

    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("observe", observe_node)
        .add_finish_point("observe")
        .set_entry_point("observe")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("approve-tool-test");
    // Hand-write a checkpoint so `resume_with` has something to lift.
    let key = entelix_core::ThreadKey::new(TenantId::new("default"), "approve-tool-test");
    cp.put(entelix_graph::Checkpoint::new(
        &key,
        0,
        Approval {
            request: "x".into(),
            approved: None,
            reply: None,
        },
        Some("observe".into()),
    ))
    .await?;

    let _final_state = graph
        .resume_with(
            Command::ApproveTool {
                tool_use_id: "tu-7".into(),
                decision: ApprovalDecision::Approve,
            },
            &ctx,
        )
        .await?;

    assert!(
        matches!(*observed.lock(), Some(ApprovalDecision::Approve)),
        "node must observe the operator's decision via PendingApprovalDecisions"
    );
    Ok(())
}

#[tokio::test]
async fn approve_tool_command_rejects_await_external_decision() {
    // Resuming with AwaitExternal would pause again immediately —
    // an operator bug. The graph rejects it as InvalidRequest.
    use entelix_core::ApprovalDecision;

    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let graph = StateGraph::<Approval>::new()
        .add_node("noop", finalize_node())
        .add_finish_point("noop")
        .set_entry_point("noop")
        .with_checkpointer(cp.clone())
        .compile()
        .unwrap();

    let ctx = ExecutionContext::new().with_thread_id("await-external-reject");
    let key = entelix_core::ThreadKey::new(TenantId::new("default"), "await-external-reject");
    cp.put(entelix_graph::Checkpoint::new(
        &key,
        0,
        Approval {
            request: "x".into(),
            approved: None,
            reply: None,
        },
        Some("noop".into()),
    ))
    .await
    .unwrap();

    let err = graph
        .resume_with(
            Command::ApproveTool {
                tool_use_id: "tu-1".into(),
                decision: ApprovalDecision::AwaitExternal,
            },
            &ctx,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidRequest(msg) if msg.contains("AwaitExternal")),
        "got {err:?}"
    );
}

#[test]
fn compile_rejects_unknown_interrupt_before_node() {
    let cp = Arc::new(InMemoryCheckpointer::<Approval>::new());
    let err = StateGraph::<Approval>::new()
        .add_node("review", review_node())
        .add_node("finalize", finalize_node())
        .add_edge("review", "finalize")
        .set_entry_point("review")
        .add_finish_point("finalize")
        .interrupt_before(["ghost"])
        .with_checkpointer(cp)
        .compile()
        .unwrap_err();
    assert!(format!("{err}").contains("ghost"));
    assert!(format!("{err}").contains("interrupt_before"));
}
