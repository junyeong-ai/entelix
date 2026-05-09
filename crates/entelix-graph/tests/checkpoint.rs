//! `Checkpointer` + `InMemoryCheckpointer` + `CompiledGraph::resume` tests.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use entelix_core::TenantId;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use entelix_core::ThreadKey;
use entelix_core::{Error, ExecutionContext, Result};
use entelix_graph::{Checkpointer, InMemoryCheckpointer, StateGraph};
use entelix_runnable::{Runnable, RunnableLambda};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Workflow {
    n: i32,
    trail: Vec<&'static str>,
}

fn step(label: &'static str, delta: i32) -> RunnableLambda<Workflow, Workflow> {
    RunnableLambda::new(move |mut s: Workflow, _ctx| async move {
        s.n += delta;
        s.trail.push(label);
        Ok::<_, _>(s)
    })
}

#[tokio::test]
async fn writes_a_checkpoint_per_node_when_thread_id_set() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step("a", 1))
        .add_node("b", step("b", 2))
        .add_node("c", step("c", 3))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_entry_point("a")
        .add_finish_point("c")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("thread-1");
    let _ = graph
        .invoke(
            Workflow {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await?;

    assert_eq!(cp.total_checkpoints(), 3);
    assert_eq!(cp.thread_count(), 1);

    let key = ThreadKey::from_ctx(&ctx)?;
    let history = cp.list_history(&key, 10).await?;
    assert_eq!(history.len(), 3);
    // Most recent first.
    assert_eq!(history[0].state.n, 6);
    assert_eq!(history[0].next_node, None); // terminal
    assert_eq!(history[1].state.n, 3);
    assert_eq!(history[1].next_node.as_deref(), Some("c"));
    assert_eq!(history[2].state.n, 1);
    assert_eq!(history[2].next_node.as_deref(), Some("b"));
    // Tenant scope is recorded on every persisted row.
    assert!(history.iter().all(|cp| cp.tenant_id == "default"));
    Ok(())
}

#[tokio::test]
async fn does_not_checkpoint_when_thread_id_absent() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step("a", 1))
        .set_entry_point("a")
        .add_finish_point("a")
        .with_checkpointer(cp.clone())
        .compile()?;

    let _ = graph
        .invoke(
            Workflow {
                n: 0,
                trail: vec![],
            },
            &ExecutionContext::new(), // no thread_id
        )
        .await?;

    assert_eq!(cp.total_checkpoints(), 0);
    Ok(())
}

#[tokio::test]
async fn resume_continues_from_latest_checkpoint() -> Result<()> {
    // Track how many times the "first" node is run; resume should not
    // re-execute it.
    let first_calls = Arc::new(AtomicUsize::new(0));
    let counter = first_calls.clone();
    let first = RunnableLambda::new(move |mut s: Workflow, _ctx| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            s.n += 10;
            s.trail.push("first");
            Ok::<_, _>(s)
        }
    });

    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("first", first)
        .add_node("second", step("second", 100))
        .add_node("third", step("third", 1000))
        .add_edge("first", "second")
        .add_edge("second", "third")
        .set_entry_point("first")
        .add_finish_point("third")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("thread-A");
    let initial = Workflow {
        n: 0,
        trail: vec![],
    };
    let _ = graph.invoke(initial, &ctx).await?;
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);

    // Resume: the latest checkpoint indicates the graph already
    // terminated, so resume returns the saved state without re-running.
    let resumed = graph.resume(&ctx).await?;
    assert_eq!(first_calls.load(Ordering::SeqCst), 1); // not re-run
    assert_eq!(resumed.n, 1110);
    assert_eq!(resumed.trail, vec!["first", "second", "third"]);
    Ok(())
}

#[tokio::test]
async fn resume_picks_up_mid_graph() -> Result<()> {
    // Hand-craft an in-memory checkpoint that pretends "first" already ran
    // and the next node is "second". Resume should run only "second" and
    // "third".
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let manual_state = Workflow {
        n: 999,
        trail: vec!["first"],
    };
    let key = entelix_core::ThreadKey::new(TenantId::new("default"), "manual");
    cp.put(entelix_graph::Checkpoint::new(
        &key,
        1,
        manual_state,
        Some("second".into()),
    ))
    .await?;

    let graph = StateGraph::<Workflow>::new()
        .add_node("first", step("first", 0)) // never invoked on resume
        .add_node("second", step("second", 100))
        .add_node("third", step("third", 1000))
        .add_edge("first", "second")
        .add_edge("second", "third")
        .set_entry_point("first")
        .add_finish_point("third")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("manual");
    let resumed = graph.resume(&ctx).await?;
    assert_eq!(resumed.n, 999 + 100 + 1000);
    assert_eq!(resumed.trail, vec!["first", "second", "third"]);
    Ok(())
}

#[tokio::test]
async fn resume_without_checkpointer_returns_config_error() {
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step("a", 1))
        .set_entry_point("a")
        .add_finish_point("a")
        .compile()
        .unwrap();

    let err = graph
        .resume(&ExecutionContext::new().with_thread_id("any"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

#[tokio::test]
async fn resume_unknown_thread_returns_invalid_request() {
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step("a", 1))
        .set_entry_point("a")
        .add_finish_point("a")
        .with_checkpointer(cp)
        .compile()
        .unwrap();

    let err = graph
        .resume(&ExecutionContext::new().with_thread_id("ghost"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)));
}

#[tokio::test]
async fn resume_without_thread_id_in_ctx_returns_config_error() {
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step("a", 1))
        .set_entry_point("a")
        .add_finish_point("a")
        .with_checkpointer(cp)
        .compile()
        .unwrap();

    let err = graph.resume(&ExecutionContext::new()).await.unwrap_err();
    assert!(matches!(err, Error::Config(_)));
}

#[tokio::test]
async fn checkpointer_partitions_per_tenant() -> Result<()> {
    // Same thread_id under two different tenants must produce two
    // independent histories — Invariant 11 enforced by ThreadKey.
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step("a", 1))
        .set_entry_point("a")
        .add_finish_point("a")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx_alpha = ExecutionContext::new()
        .with_tenant_id(TenantId::new("alpha"))
        .with_thread_id("conv-1");
    let ctx_bravo = ExecutionContext::new()
        .with_tenant_id(TenantId::new("bravo"))
        .with_thread_id("conv-1");

    let _ = graph
        .invoke(
            Workflow {
                n: 0,
                trail: vec![],
            },
            &ctx_alpha,
        )
        .await?;
    let _ = graph
        .invoke(
            Workflow {
                n: 100,
                trail: vec![],
            },
            &ctx_bravo,
        )
        .await?;

    let alpha_key = ThreadKey::new(TenantId::new("alpha"), "conv-1");
    let bravo_key = ThreadKey::new(TenantId::new("bravo"), "conv-1");

    let alpha_hist = cp.list_history(&alpha_key, 10).await?;
    let bravo_hist = cp.list_history(&bravo_key, 10).await?;

    assert_eq!(alpha_hist.len(), 1);
    assert_eq!(bravo_hist.len(), 1);
    assert_eq!(alpha_hist[0].state.n, 1);
    assert_eq!(bravo_hist[0].state.n, 101);
    // Cross-read: alpha's history under bravo's key MUST be empty.
    assert!(cp.get_latest(&bravo_key).await?.is_some());
    let cross = cp.get_by_id(&alpha_key, &bravo_hist[0].id).await?;
    assert!(
        cross.is_none(),
        "tenant scope must isolate checkpoint lookups"
    );
    Ok(())
}

#[tokio::test]
async fn history_respects_limit_and_order() -> Result<()> {
    let cp = Arc::new(InMemoryCheckpointer::<Workflow>::new());
    let graph = StateGraph::<Workflow>::new()
        .add_node("a", step("a", 1))
        .add_node("b", step("b", 1))
        .add_node("c", step("c", 1))
        .add_node("d", step("d", 1))
        .add_node("e", step("e", 1))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("c", "d")
        .add_edge("d", "e")
        .set_entry_point("a")
        .add_finish_point("e")
        .with_checkpointer(cp.clone())
        .compile()?;

    let ctx = ExecutionContext::new().with_thread_id("hist");
    let _ = graph
        .invoke(
            Workflow {
                n: 0,
                trail: vec![],
            },
            &ctx,
        )
        .await?;

    let key = ThreadKey::from_ctx(&ctx)?;
    let two = cp.list_history(&key, 2).await?;
    assert_eq!(two.len(), 2);
    assert_eq!(two[0].state.n, 5); // most recent (after e)
    assert_eq!(two[1].state.n, 4); // before that (after d)
    Ok(())
}
