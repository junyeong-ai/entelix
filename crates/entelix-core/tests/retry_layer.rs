//! `RetryLayer` integration — wraps a `tower::Service<ModelInvocation>`
//! and validates the canonical retry semantics:
//!
//! - transient `Provider` statuses (5xx, 429, 408, 425, network=0)
//!   trigger another attempt
//! - permanent statuses (4xx other than retry-eligible) surface
//!   immediately
//! - cancellation pulls the rug at the head of every iteration

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use tower::{Layer, Service};

use entelix_core::backoff::ExponentialBackoff;
use entelix_core::cancellation::CancellationToken;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::ir::{Message, ModelRequest, ModelResponse, StopReason, Usage};
use entelix_core::service::ModelInvocation;
use entelix_core::transports::{RetryLayer, RetryPolicy};

/// `Service<ModelInvocation>` whose responses are scripted by a queue
/// — each call pops the next entry. Empty queue panics, so tests
/// declare exactly the sequence they expect to see.
#[derive(Clone)]
struct ScriptedService {
    queue: Arc<Mutex<Vec<Result<ModelResponse>>>>,
    calls: Arc<AtomicU32>,
}

impl ScriptedService {
    fn new(script: Vec<Result<ModelResponse>>) -> (Self, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        (
            Self {
                queue: Arc::new(Mutex::new(script.into_iter().rev().collect())),
                calls: calls.clone(),
            },
            calls,
        )
    }
}

impl Service<ModelInvocation> for ScriptedService {
    type Response = ModelResponse;
    type Error = Error;
    type Future = BoxFuture<'static, Result<ModelResponse>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _invocation: ModelInvocation) -> Self::Future {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let queue = Arc::clone(&self.queue);
        Box::pin(async move {
            queue
                .lock()
                .pop()
                .expect("ScriptedService called more times than the script declared")
        })
    }
}

fn ok_response() -> ModelResponse {
    ModelResponse {
        id: "ok".into(),
        model: "test".into(),
        stop_reason: StopReason::EndTurn,
        content: vec![],
        usage: Usage::default(),
        rate_limit: None,
        warnings: vec![],
    }
}

fn invocation(ctx: ExecutionContext) -> ModelInvocation {
    ModelInvocation {
        request: ModelRequest {
            model: "test".into(),
            messages: vec![Message::user("hi")],
            ..ModelRequest::default()
        },
        ctx,
    }
}

fn fast_policy(max_attempts: u32) -> RetryPolicy {
    RetryPolicy::standard()
        .with_max_attempts(max_attempts)
        .with_backoff(ExponentialBackoff::new(
            Duration::from_millis(1),
            Duration::from_millis(2),
        ))
}

#[tokio::test]
async fn retries_transient_5xx_then_succeeds() {
    let (svc, calls) = ScriptedService::new(vec![
        Err(Error::provider_http(503, "transient")),
        Err(Error::provider_http(503, "still transient")),
        Ok(ok_response()),
    ]);
    let layered = RetryLayer::new(fast_policy(5)).layer(svc);
    let out = tower::ServiceExt::oneshot(layered, invocation(ExecutionContext::new()))
        .await
        .unwrap();
    assert_eq!(out.id, "ok");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retries_429_rate_limit() {
    let (svc, calls) = ScriptedService::new(vec![
        Err(Error::provider_http(429, "slow down")),
        Ok(ok_response()),
    ]);
    let layered = RetryLayer::new(fast_policy(3)).layer(svc);
    let out = tower::ServiceExt::oneshot(layered, invocation(ExecutionContext::new()))
        .await
        .unwrap();
    assert_eq!(out.id, "ok");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn retries_network_failures_status_zero() {
    let (svc, calls) = ScriptedService::new(vec![
        Err(Error::provider_network("network error: connect refused")),
        Ok(ok_response()),
    ]);
    let layered = RetryLayer::new(fast_policy(3)).layer(svc);
    let out = tower::ServiceExt::oneshot(layered, invocation(ExecutionContext::new()))
        .await
        .unwrap();
    assert_eq!(out.id, "ok");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn does_not_retry_4xx_auth() {
    let (svc, calls) = ScriptedService::new(vec![Err(Error::provider_http(401, "no auth"))]);
    let layered = RetryLayer::new(fast_policy(5)).layer(svc);
    let err = tower::ServiceExt::oneshot(layered, invocation(ExecutionContext::new()))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Provider {
            kind: entelix_core::ProviderErrorKind::Http(401),
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn exhausts_attempts_and_returns_last_error() {
    let (svc, calls) = ScriptedService::new(vec![
        Err(Error::provider_http(503, "down")),
        Err(Error::provider_http(503, "down")),
        Err(Error::provider_http(503, "down")),
    ]);
    let layered = RetryLayer::new(fast_policy(3)).layer(svc);
    let err = tower::ServiceExt::oneshot(layered, invocation(ExecutionContext::new()))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Provider {
            kind: entelix_core::ProviderErrorKind::Http(503),
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn cancellation_short_circuits_retry_loop() {
    let token = CancellationToken::new();
    let ctx = ExecutionContext::with_cancellation(token.clone());
    let (svc, _calls) = ScriptedService::new(vec![Err(Error::provider_http(503, "down"))]);
    let layered = RetryLayer::new(
        RetryPolicy::standard()
            .with_max_attempts(10)
            // Long backoff so cancellation has time to fire mid-sleep.
            .with_backoff(ExponentialBackoff::new(
                Duration::from_secs(60),
                Duration::from_secs(60),
            )),
    )
    .layer(svc);
    let handle =
        tokio::spawn(async move { tower::ServiceExt::oneshot(layered, invocation(ctx)).await });
    // Let the first call fail, then cancel during backoff.
    tokio::time::sleep(Duration::from_millis(20)).await;
    token.cancel();
    let err = handle.await.unwrap().unwrap_err();
    assert!(matches!(err, Error::Cancelled));
}

#[tokio::test]
async fn max_attempts_one_disables_retry() {
    let (svc, calls) = ScriptedService::new(vec![Err(Error::provider_http(503, "down"))]);
    let layered = RetryLayer::new(fast_policy(1)).layer(svc);
    let err = tower::ServiceExt::oneshot(layered, invocation(ExecutionContext::new()))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Provider {
            kind: entelix_core::ProviderErrorKind::Http(503),
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn deadline_truncates_backoff_and_surfaces_deadline_exceeded() {
    // Five queued 503s + a 5-attempt policy that would normally
    // backoff several times. Deadline of 60ms forces the loop to
    // bail out at the first backoff that would land past the
    // deadline, returning DeadlineExceeded — NOT exhausting all
    // 5 attempts and surfacing Provider(503).
    let (svc, calls) = ScriptedService::new(vec![
        Err(Error::provider_http(503, "down")),
        Err(Error::provider_http(503, "still down")),
        Err(Error::provider_http(503, "still still down")),
        Err(Error::provider_http(503, "still still still down")),
        Err(Error::provider_http(503, "you get the idea")),
    ]);
    let policy =
        RetryPolicy::standard()
            .with_max_attempts(5)
            .with_backoff(ExponentialBackoff::new(
                Duration::from_millis(50),
                Duration::from_millis(500),
            ));
    let layered = RetryLayer::new(policy).layer(svc);
    let ctx = ExecutionContext::new()
        .with_deadline(tokio::time::Instant::now() + Duration::from_millis(60));
    let started = tokio::time::Instant::now();
    let err = tower::ServiceExt::oneshot(layered, invocation(ctx))
        .await
        .unwrap_err();
    let elapsed = started.elapsed();

    assert!(
        matches!(err, Error::DeadlineExceeded),
        "expected DeadlineExceeded, got {err:?}"
    );
    assert!(
        elapsed < Duration::from_millis(500),
        "retry must bail at deadline, took {elapsed:?}"
    );
    // Exact attempt count depends on jitter; what matters is that
    // we did NOT exhaust all 5 attempts before bailing.
    assert!(
        calls.load(Ordering::SeqCst) < 5,
        "deadline must short-circuit before max_attempts (called {})",
        calls.load(Ordering::SeqCst)
    );
}

#[tokio::test]
async fn deadline_already_past_at_call_time_short_circuits_immediately() {
    let (svc, calls) = ScriptedService::new(vec![Ok(ok_response())]);
    let layered = RetryLayer::new(fast_policy(3)).layer(svc);
    let ctx = ExecutionContext::new()
        .with_deadline(tokio::time::Instant::now() - Duration::from_millis(1));
    let err = tower::ServiceExt::oneshot(layered, invocation(ctx))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::DeadlineExceeded));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "past-deadline must skip the inner call entirely"
    );
}
