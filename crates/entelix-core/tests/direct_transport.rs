//! `DirectTransport` integration tests using `wiremock` as the provider.
//!
//! Each test stands up an HTTP mock server, points a `DirectTransport` at
//! it, and asserts that headers / body / status flow through correctly.
//! Cancellation and deadline behaviour is exercised against a slow handler.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use entelix_core::auth::ApiKeyProvider;
use entelix_core::codecs::EncodedRequest;
use entelix_core::transports::{DirectTransport, Transport};
use entelix_core::{Error, ExecutionContext};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn ctx() -> ExecutionContext {
    ExecutionContext::new()
}

#[tokio::test(flavor = "multi_thread")]
async fn happy_path_propagates_headers_and_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header("content-type", "application/json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-trace", "abc123")
                .set_body_string("{\"ok\":true}"),
        )
        .mount(&server)
        .await;

    let transport =
        DirectTransport::new(server.uri(), Arc::new(ApiKeyProvider::anthropic("sk-test"))).unwrap();

    let mut req = EncodedRequest::post_json("/v1/messages", Bytes::from_static(b"{}"));
    req.headers.insert(
        http::HeaderName::from_static("anthropic-version"),
        http::HeaderValue::from_static("2023-06-01"),
    );

    let resp = transport.send(req, &ctx()).await.unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(resp.headers.get("x-trace").unwrap(), "abc123");
    assert_eq!(&resp.body[..], b"{\"ok\":true}");
}

#[tokio::test(flavor = "multi_thread")]
async fn non_2xx_status_is_returned_to_caller_for_codec_handling() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let transport =
        DirectTransport::new(server.uri(), Arc::new(ApiKeyProvider::anthropic("sk-test"))).unwrap();

    let req = EncodedRequest::post_json("/anything", Bytes::from_static(b"{}"));
    let resp = transport.send(req, &ctx()).await.unwrap();
    assert_eq!(resp.status, 429);
    assert_eq!(&resp.body[..], b"rate limited");
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_token_short_circuits_send() {
    // No mock — token is already cancelled before send is awaited, so the
    // transport must bail out without making the request.
    let transport = DirectTransport::new(
        "http://127.0.0.1:1",
        Arc::new(ApiKeyProvider::anthropic("k")),
    )
    .unwrap();

    let ctx = ExecutionContext::new();
    ctx.cancellation().cancel();

    let req = EncodedRequest::post_json("/x", Bytes::new());
    let err = transport.send(req, &ctx).await.unwrap_err();
    assert!(matches!(err, Error::Cancelled), "got {err:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn deadline_exceeded_when_provider_is_slow() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
        .mount(&server)
        .await;

    let transport =
        DirectTransport::new(server.uri(), Arc::new(ApiKeyProvider::anthropic("k"))).unwrap();

    let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
    let ctx = ExecutionContext::new().with_deadline(deadline);

    let req = EncodedRequest::post_json("/x", Bytes::new());
    let err = transport.send(req, &ctx).await.unwrap_err();
    assert!(matches!(err, Error::DeadlineExceeded), "got {err:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_helper_uses_correct_base_url() {
    let transport = DirectTransport::anthropic(Arc::new(ApiKeyProvider::anthropic("k"))).unwrap();
    assert_eq!(transport.base_url(), "https://api.anthropic.com");
    assert_eq!(transport.name(), "direct");
}
