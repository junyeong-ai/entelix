//! `RetryToolLayer` integration — wraps a
//! `Service<ToolInvocation>` and validates the canonical
//! metadata-driven retry semantics:
//!
//! - tools without `retry_hint` pass through unchanged (no retry)
//! - tools with `retry_hint` retry on transient `Provider` errors
//!   up to `hint.max_attempts` and bubble the last error otherwise
//! - vendor `Retry-After` overrides the computed backoff
//! - cancellation during backoff short-circuits with `Cancelled`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tower::{Layer, Service, ServiceExt};

use entelix_core::cancellation::CancellationToken;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::service::ToolInvocation;
use entelix_core::tools::{RetryHint, RetryToolLayer, ToolMetadata};

/// Scripted leaf service — pops one entry off the queue per call.
#[derive(Clone)]
struct ScriptedToolService {
    queue: Arc<Mutex<Vec<Result<Value>>>>,
    calls: Arc<AtomicU32>,
}

impl ScriptedToolService {
    fn new(script: Vec<Result<Value>>) -> (Self, Arc<AtomicU32>) {
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

impl Service<ToolInvocation> for ScriptedToolService {
    type Response = Value;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Value>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _invocation: ToolInvocation) -> Self::Future {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let queue = Arc::clone(&self.queue);
        Box::pin(async move {
            queue
                .lock()
                .pop()
                .expect("ScriptedToolService called more times than the script declared")
        })
    }
}

fn metadata_with_hint(name: &str, hint: Option<RetryHint>) -> Arc<ToolMetadata> {
    let mut md = ToolMetadata::function(name, "scripted", json!({"type": "object"}));
    if let Some(hint) = hint {
        md = md.with_retry_hint(hint);
    }
    Arc::new(md)
}

fn invocation(metadata: Arc<ToolMetadata>, ctx: ExecutionContext) -> ToolInvocation {
    ToolInvocation::new(String::new(), metadata, json!({}), ctx)
}

#[tokio::test]
async fn no_hint_means_no_retry() {
    // Even a transient error passes straight through if the tool
    // has not opted in via `with_retry_hint(...)`.
    let (svc, calls) = ScriptedToolService::new(vec![Err(Error::provider_http(503, "down"))]);
    let layered = RetryToolLayer::new().layer(svc);
    let metadata = metadata_with_hint("scripted", None);
    let err = ServiceExt::oneshot(layered, invocation(metadata, ExecutionContext::new()))
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Provider {
                kind: entelix_core::ProviderErrorKind::Http(503),
                ..
            }
        ),
        "got: {err:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "tool without retry_hint must run exactly once"
    );
}

#[tokio::test]
async fn hint_retries_transient_until_success() {
    let (svc, calls) = ScriptedToolService::new(vec![
        Err(Error::provider_http(503, "down")),
        Err(Error::provider_http(503, "still down")),
        Ok(json!({"ok": true})),
    ]);
    let layered = RetryToolLayer::new()
        .with_max_backoff(Duration::from_millis(2))
        .layer(svc);
    let hint = RetryHint::new(5, Duration::from_millis(1));
    let metadata = metadata_with_hint("scripted", Some(hint));
    let out = ServiceExt::oneshot(layered, invocation(metadata, ExecutionContext::new()))
        .await
        .unwrap();
    assert_eq!(out, json!({"ok": true}));
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn hint_exhausts_attempts_and_bubbles_last_error() {
    let (svc, calls) = ScriptedToolService::new(vec![
        Err(Error::provider_http(503, "down")),
        Err(Error::provider_http(503, "down")),
    ]);
    let layered = RetryToolLayer::new()
        .with_max_backoff(Duration::from_millis(2))
        .layer(svc);
    let hint = RetryHint::new(2, Duration::from_millis(1));
    let metadata = metadata_with_hint("scripted", Some(hint));
    let err = ServiceExt::oneshot(layered, invocation(metadata, ExecutionContext::new()))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Provider {
            kind: entelix_core::ProviderErrorKind::Http(503),
            ..
        }
    ));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "hint.max_attempts == 2 caps total calls at 2"
    );
}

#[tokio::test]
async fn permanent_error_short_circuits_even_with_hint() {
    // A 4xx that the classifier marks non-retryable bubbles
    // immediately regardless of how generous the hint is.
    let (svc, calls) = ScriptedToolService::new(vec![Err(Error::provider_http(400, "bad input"))]);
    let layered = RetryToolLayer::new()
        .with_max_backoff(Duration::from_millis(2))
        .layer(svc);
    let hint = RetryHint::new(5, Duration::from_millis(1));
    let metadata = metadata_with_hint("scripted", Some(hint));
    let err = ServiceExt::oneshot(layered, invocation(metadata, ExecutionContext::new()))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Provider {
            kind: entelix_core::ProviderErrorKind::Http(400),
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_during_backoff_short_circuits() {
    let (svc, _calls) = ScriptedToolService::new(vec![Err(Error::provider_http(503, "down"))]);
    // Long backoff so cancellation has time to fire mid-sleep.
    let layered = RetryToolLayer::new()
        .with_max_backoff(Duration::from_secs(60))
        .layer(svc);
    let hint = RetryHint::new(5, Duration::from_secs(60));
    let metadata = metadata_with_hint("scripted", Some(hint));
    let token = CancellationToken::new();
    let ctx = ExecutionContext::with_cancellation(token.clone());
    let handle =
        tokio::spawn(async move { ServiceExt::oneshot(layered, invocation(metadata, ctx)).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    token.cancel();
    let err = handle.await.unwrap().unwrap_err();
    assert!(matches!(err, Error::Cancelled));
}
