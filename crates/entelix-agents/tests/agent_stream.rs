//! Integration test for `Agent::execute_stream` — end-to-end event
//! sequence with multiple sinks attached.
//!
//! Verifies (Slice A done criteria):
//! - `Started` opens every run
//! - `Complete(state)` closes every successful run
//! - caller-facing stream and `CaptureSink` see the same sequence
//! - `BroadcastSink` fans out identical events to multiple
//!   subscribers
//! - `ChannelSink` capacity-bound semantics (drop receiver →
//!   agent surfaces `Cancelled`)

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use entelix_agents::{Agent, AgentEvent, BroadcastSink, CaptureSink, ChannelSink};
use entelix_core::context::ExecutionContext;
use entelix_runnable::{Runnable, RunnableLambda};
use futures::StreamExt;

fn step_runnable() -> impl Runnable<i32, i32> {
    RunnableLambda::new(|n: i32, _ctx| async move { Ok::<_, _>(n * 10) })
}

#[tokio::test]
async fn execute_stream_emits_canonical_event_sequence_to_caller_and_sink() {
    let sink = CaptureSink::<i32>::new();
    let agent = Agent::<i32>::builder()
        .with_name("canonical")
        .with_runnable(step_runnable())
        .with_sink(sink.clone())
        .build()
        .unwrap();

    let ctx = ExecutionContext::new();
    let mut stream = agent.execute_stream(5, &ctx);
    let mut caller_view = Vec::new();
    while let Some(event) = stream.next().await {
        caller_view.push(event.unwrap());
    }

    let sink_view = sink.events();
    assert_eq!(
        caller_view.len(),
        sink_view.len(),
        "caller stream and sink must observe identical event count"
    );
    assert!(matches!(
        &caller_view[0],
        AgentEvent::Started { agent, .. } if agent == "canonical"
    ));
    assert!(matches!(
        caller_view.last(),
        Some(AgentEvent::Complete { state: 50, .. })
    ));
}

#[tokio::test]
async fn broadcast_sink_fans_out_to_multiple_subscribers() {
    let sink = BroadcastSink::<i32>::new(8);
    let mut sub_a = sink.subscribe();
    let mut sub_b = sink.subscribe();

    let agent = Agent::<i32>::builder()
        .with_name("fanout")
        .with_runnable(step_runnable())
        .with_sink(sink)
        .build()
        .unwrap();

    let _final = agent.execute(7, &ExecutionContext::new()).await.unwrap();

    // Each subscriber sees Started + Complete.
    let a0 = sub_a.recv().await.unwrap();
    let a1 = sub_a.recv().await.unwrap();
    let b0 = sub_b.recv().await.unwrap();
    let b1 = sub_b.recv().await.unwrap();

    assert!(matches!(a0, AgentEvent::Started { .. }));
    assert!(matches!(a1, AgentEvent::Complete { state: 70, .. }));
    assert!(matches!(b0, AgentEvent::Started { .. }));
    assert!(matches!(b1, AgentEvent::Complete { state: 70, .. }));
}

#[tokio::test]
async fn channel_sink_drops_with_receiver_gone() {
    // Drop receiver immediately, then run agent — first sink emit
    // (Started) should fail with Cancelled.
    let (sink, rx) = ChannelSink::<i32>::new(1);
    drop(rx);

    let agent = Agent::<i32>::builder()
        .with_name("dropped-rx")
        .with_runnable(step_runnable())
        .with_sink(sink)
        .build()
        .unwrap();

    let result = agent.execute(1, &ExecutionContext::new()).await;
    assert!(
        matches!(result, Err(entelix_core::Error::Cancelled)),
        "agent must surface Cancelled when sink receiver is dropped"
    );
}

#[tokio::test]
async fn supervised_mode_requires_approver_at_build_time() {
    // Build-time guard: ExecutionMode::Supervised without an
    // Approver is a programmer error.
    use entelix_agents::ExecutionMode;
    let err = Agent::<i32>::builder()
        .with_name("supervised-no-approver")
        .with_runnable(step_runnable())
        .with_execution_mode(ExecutionMode::Supervised)
        .build()
        .unwrap_err();
    assert!(format!("{err}").contains("requires an Approver"));
}

#[tokio::test]
async fn observer_pre_turn_and_on_complete_fire_in_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_agents::AgentObserver;
    use entelix_core::context::ExecutionContext as Ctx;

    /// Records which lifecycle methods fired and in what order.
    struct OrderRecorder {
        seq: Arc<parking_lot::Mutex<Vec<&'static str>>>,
        pre_turn_count: AtomicUsize,
        on_complete_count: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl AgentObserver<i32> for OrderRecorder {
        fn name(&self) -> &'static str {
            "recorder"
        }
        async fn pre_turn(&self, _state: &i32, _ctx: &Ctx) -> entelix_core::Result<()> {
            self.seq.lock().push("pre_turn");
            self.pre_turn_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_complete(&self, _state: &i32, _ctx: &Ctx) -> entelix_core::Result<()> {
            self.seq.lock().push("on_complete");
            self.on_complete_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let seq = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let observer = OrderRecorder {
        seq: Arc::clone(&seq),
        pre_turn_count: AtomicUsize::new(0),
        on_complete_count: AtomicUsize::new(0),
    };
    let observer_arc: Arc<dyn AgentObserver<i32>> = Arc::new(observer);

    let agent = Agent::<i32>::builder()
        .with_name("observed")
        .with_runnable(step_runnable())
        .with_observer_arc(observer_arc.clone())
        .build()
        .unwrap();

    // execute() — pre_turn fires once, on_complete fires once.
    let _ = agent.execute(2, &ExecutionContext::new()).await.unwrap();
    let order = seq.lock().clone();
    assert_eq!(
        order,
        vec!["pre_turn", "on_complete"],
        "observers must fire in lifecycle order"
    );
}

#[tokio::test]
async fn observer_on_error_fires_when_runnable_fails() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_agents::AgentObserver;
    use entelix_core::context::ExecutionContext as Ctx;

    /// Records `on_error` and `on_complete` fires so the test can
    /// assert the failure-path branch is taken.
    struct FailureObserver {
        on_error: AtomicUsize,
        on_complete: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl AgentObserver<i32> for FailureObserver {
        async fn on_complete(&self, _state: &i32, _ctx: &Ctx) -> entelix_core::Result<()> {
            self.on_complete.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_error(
            &self,
            _error: &entelix_core::Error,
            _ctx: &Ctx,
        ) -> entelix_core::Result<()> {
            self.on_error.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    // Inner runnable that always fails. Distinct from the
    // observer-veto test (next) — failure originates in the
    // runnable, not in pre_turn.
    let failing = RunnableLambda::new(|_n: i32, _ctx| async move {
        Err::<i32, _>(entelix_core::Error::config("inner runnable refused"))
    });

    let observer = Arc::new(FailureObserver {
        on_error: AtomicUsize::new(0),
        on_complete: AtomicUsize::new(0),
    });
    let observer_dyn: Arc<dyn AgentObserver<i32>> = observer.clone();

    let agent = Agent::<i32>::builder()
        .with_name("on-error-test")
        .with_runnable(failing)
        .with_observer_arc(observer_dyn)
        .build()
        .unwrap();

    let err = agent
        .execute(7, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("inner runnable refused"));
    assert_eq!(
        observer.on_error.load(Ordering::SeqCst),
        1,
        "on_error must fire when runnable.invoke returns Err"
    );
    assert_eq!(
        observer.on_complete.load(Ordering::SeqCst),
        0,
        "on_complete must NOT fire on the failure path"
    );
}

#[tokio::test]
async fn observer_on_error_fires_through_streaming_path() {
    // execute_stream and execute both route through run_inner, so a
    // single on_error fire-site covers both. Pin the streaming
    // path explicitly so a future refactor that splits the two
    // doesn't silently lose the failure-observation surface for
    // streaming consumers.
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_agents::AgentObserver;
    use entelix_core::context::ExecutionContext as Ctx;

    struct StreamObserver {
        on_error: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl AgentObserver<i32> for StreamObserver {
        async fn on_error(
            &self,
            _error: &entelix_core::Error,
            _ctx: &Ctx,
        ) -> entelix_core::Result<()> {
            self.on_error.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let failing = RunnableLambda::new(|_n: i32, _ctx| async move {
        Err::<i32, _>(entelix_core::Error::config("streaming runnable refused"))
    });

    let observer = Arc::new(StreamObserver {
        on_error: AtomicUsize::new(0),
    });
    let observer_dyn: Arc<dyn AgentObserver<i32>> = observer.clone();

    let agent = Agent::<i32>::builder()
        .with_name("streaming-on-error")
        .with_runnable(failing)
        .with_observer_arc(observer_dyn)
        .build()
        .unwrap();

    let ctx = ExecutionContext::new();
    let mut stream = agent.execute_stream(0, &ctx);
    let mut saw_failed = false;
    let mut saw_err = false;
    while let Some(event) = stream.next().await {
        match event {
            Ok(entelix_agents::AgentEvent::Failed { .. }) => saw_failed = true,
            Err(_) => saw_err = true,
            _ => {}
        }
    }
    assert!(saw_failed, "streaming caller must see AgentEvent::Failed");
    assert!(saw_err, "streaming caller must see the typed Err yield");
    assert_eq!(
        observer.on_error.load(Ordering::SeqCst),
        1,
        "on_error must fire on the streaming failure path",
    );
}

#[tokio::test]
async fn observer_on_error_skipped_for_interrupted_variant() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use entelix_agents::AgentObserver;
    use entelix_core::context::ExecutionContext as Ctx;

    /// Counts `on_error` fires. Interrupt is a control signal, not
    /// a failure — operators wanting interrupt observation
    /// consume `AgentEvent::Interrupted` from the sink instead.
    struct InterruptObserver {
        on_error: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl AgentObserver<i32> for InterruptObserver {
        async fn on_error(
            &self,
            _error: &entelix_core::Error,
            _ctx: &Ctx,
        ) -> entelix_core::Result<()> {
            self.on_error.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let interrupting = RunnableLambda::new(|_n: i32, _ctx| async move {
        Err::<i32, _>(entelix_core::Error::Interrupted {
            payload: serde_json::json!({"kind": "approval_pending"}),
        })
    });

    let observer = Arc::new(InterruptObserver {
        on_error: AtomicUsize::new(0),
    });
    let observer_dyn: Arc<dyn AgentObserver<i32>> = observer.clone();

    let agent = Agent::<i32>::builder()
        .with_name("interrupt-test")
        .with_runnable(interrupting)
        .with_observer_arc(observer_dyn)
        .build()
        .unwrap();

    let _err = agent
        .execute(0, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert_eq!(
        observer.on_error.load(Ordering::SeqCst),
        0,
        "on_error must NOT fire for Error::Interrupted (HITL pause is a control signal)"
    );
}

#[tokio::test]
async fn observer_returning_err_aborts_agent() {
    use entelix_agents::AgentObserver;
    use entelix_core::context::ExecutionContext as Ctx;

    struct AbortingObserver;

    #[async_trait::async_trait]
    impl AgentObserver<i32> for AbortingObserver {
        async fn pre_turn(&self, _state: &i32, _ctx: &Ctx) -> entelix_core::Result<()> {
            Err(entelix_core::Error::config("observer veto"))
        }
    }

    let agent = Agent::<i32>::builder()
        .with_name("aborting-observer")
        .with_runnable(step_runnable())
        .with_observer(AbortingObserver)
        .build()
        .unwrap();

    let err = agent
        .execute(0, &ExecutionContext::new())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("observer veto"));
}

#[tokio::test]
async fn agent_composes_inside_outer_runnable() {
    // Agent is itself a Runnable<S, S> → wrappable as Arc<dyn ...>
    // and usable wherever a runnable is expected.
    let agent = Agent::<i32>::builder()
        .with_name("inner")
        .with_runnable(step_runnable())
        .build()
        .unwrap();
    let runnable: Arc<dyn Runnable<i32, i32>> = Arc::new(agent);

    let result = runnable.invoke(3, &ExecutionContext::new()).await.unwrap();
    assert_eq!(result, 30);
}
