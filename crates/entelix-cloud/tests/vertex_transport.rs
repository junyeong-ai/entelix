//! `VertexTransport` end-to-end via wiremock + a deterministic mock
//! `TokenRefresher`. Validates auth-header injection, quota project
//! routing, and 401-driven cache invalidation.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::missing_const_for_fn)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use entelix_cloud::CloudError;
use entelix_cloud::refresh::{TokenRefresher, TokenSnapshot};
use entelix_cloud::vertex::VertexTransport;
use entelix_core::ExecutionContext;
use entelix_core::codecs::EncodedRequest;
use entelix_core::transports::Transport;
use secrecy::SecretString;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Default)]
struct StaticRefresher {
    calls: AtomicU32,
    token: String,
}

impl StaticRefresher {
    fn new(token: &str) -> Self {
        Self {
            calls: AtomicU32::new(0),
            token: token.to_owned(),
        }
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TokenRefresher<SecretString> for StaticRefresher {
    async fn refresh(&self) -> Result<TokenSnapshot<SecretString>, CloudError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(TokenSnapshot {
            value: SecretString::from(self.token.clone()),
            expires_at: Instant::now() + Duration::from_secs(3600),
        })
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn auth_header_and_quota_project_attached() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/v1/projects/p/locations/global/publishers/anthropic/models/claude:rawPredict",
        ))
        .and(header("authorization", "Bearer test-token"))
        .and(header("x-goog-user-project", "billing-project"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"ok\":true}"))
        .mount(&server)
        .await;

    let transport = VertexTransport::builder()
        .with_project_id("p")
        .with_location("global")
        .with_quota_project("billing-project")
        .with_base_url(server.uri())
        .with_token_refresher(Arc::new(StaticRefresher::new("test-token")))
        .build()
        .unwrap();

    let req = EncodedRequest::post_json(
        "/v1/projects/p/locations/global/publishers/anthropic/models/claude:rawPredict",
        Bytes::from_static(b"{}"),
    );
    let resp = transport.send(req, &ExecutionContext::new()).await.unwrap();
    assert_eq!(resp.status, 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn cached_token_reused_across_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let refresher = Arc::new(StaticRefresher::new("rotation-1"));
    let transport = VertexTransport::builder()
        .with_project_id("p")
        .with_location("us-central1")
        .with_base_url(server.uri())
        .with_token_refresher(refresher.clone())
        .build()
        .unwrap();

    for _ in 0..5 {
        let req = EncodedRequest::post_json("/v1/x", Bytes::from_static(b"{}"));
        transport.send(req, &ExecutionContext::new()).await.unwrap();
    }
    assert_eq!(
        refresher.call_count(),
        1,
        "five sends should reuse the cached token"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn unauthorized_invalidates_cached_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .mount(&server)
        .await;

    let refresher = Arc::new(StaticRefresher::new("expired"));
    let transport = VertexTransport::builder()
        .with_project_id("p")
        .with_location("us-central1")
        .with_base_url(server.uri())
        .with_token_refresher(refresher.clone())
        .build()
        .unwrap();

    let req1 = EncodedRequest::post_json("/v1/x", Bytes::from_static(b"{}"));
    let resp1 = transport
        .send(req1, &ExecutionContext::new())
        .await
        .unwrap();
    assert_eq!(resp1.status, 401);
    assert_eq!(refresher.call_count(), 1);

    let req2 = EncodedRequest::post_json("/v1/x", Bytes::from_static(b"{}"));
    let _resp2 = transport
        .send(req2, &ExecutionContext::new())
        .await
        .unwrap();
    // 401 invalidated the cache → second call refreshes again.
    assert_eq!(refresher.call_count(), 2);
}

/// Refresher that blocks for a long time — used to verify the
/// transport surfaces caller cancellation while a token refresh
/// is in flight, instead of waiting for the refresh to complete.
struct BlockingRefresher;

#[async_trait]
impl TokenRefresher<SecretString> for BlockingRefresher {
    async fn refresh(&self) -> Result<TokenSnapshot<SecretString>, CloudError> {
        // Long enough that any non-cancellation-aware path would
        // block the test, but short enough that a regression
        // doesn't hang CI for minutes.
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok(TokenSnapshot {
            value: SecretString::from("never-arrives".to_owned()),
            expires_at: Instant::now() + Duration::from_secs(3600),
        })
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_surfaces_while_token_refresh_is_in_flight() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let transport = VertexTransport::builder()
        .with_project_id("p")
        .with_location("us-central1")
        .with_base_url(server.uri())
        .with_token_refresher(Arc::new(BlockingRefresher))
        .build()
        .unwrap();

    let cancellation = entelix_core::cancellation::CancellationToken::new();
    let ctx = ExecutionContext::with_cancellation(cancellation.clone());
    let req = EncodedRequest::post_json("/v1/x", Bytes::from_static(b"{}"));

    // Cancel after a short delay — the refresher will still be
    // sleeping, so the only way `send` returns is by racing the
    // cancellation token against `build_headers`.
    let cancel_handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancellation.cancel();
    });

    let started = Instant::now();
    let result = transport.send(req, &ctx).await;
    let elapsed = started.elapsed();
    cancel_handle.await.unwrap();

    assert!(
        matches!(result, Err(entelix_core::Error::Cancelled)),
        "expected Error::Cancelled, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "send must return promptly on cancellation, took {elapsed:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_path_passes_auth_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(200).set_body_string("data: ok\n\n"))
        .mount(&server)
        .await;

    let transport = VertexTransport::builder()
        .with_project_id("p")
        .with_location("us-central1")
        .with_base_url(server.uri())
        .with_token_refresher(Arc::new(StaticRefresher::new("test-token")))
        .build()
        .unwrap();

    let req = EncodedRequest::post_json("/stream", Bytes::from_static(b"{}"));
    let stream = transport
        .send_streaming(req, &ExecutionContext::new())
        .await
        .unwrap();
    assert_eq!(stream.status, 200);
}
