//! `OpenAiEmbedder` end-to-end via `wiremock`. Verifies that a real
//! HTTP round-trip lands the model identifier, optional dimensions
//! parameter, and bearer-style authorization header on the wire,
//! and that the response decodes into the expected `Embedding`
//! plus `EmbeddingUsage`.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::redundant_closure_for_method_calls
)]

use std::sync::Arc;

use entelix_core::auth::ApiKeyProvider;
use entelix_core::context::ExecutionContext;
use entelix_memory::{Embedder, EmbeddingUsage};
use entelix_memory_openai::OpenAiEmbedder;
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider() -> Arc<entelix_core::auth::ApiKeyProvider> {
    Arc::new(ApiKeyProvider::new("authorization", "Bearer secret-key").unwrap())
}

#[tokio::test]
async fn embed_round_trips_vector_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer secret-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "embedding": [0.1, 0.2, 0.3], "index": 0 }],
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 5, "total_tokens": 5 }
        })))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedder::custom("text-embedding-3-small", 3)
        .with_credentials(provider())
        .with_base_url(server.uri())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let out = embedder.embed("hello", &ctx).await.unwrap();
    assert_eq!(out.vector, vec![0.1, 0.2, 0.3]);
    assert_eq!(out.usage, Some(EmbeddingUsage::new(5)));
}

#[tokio::test]
async fn embed_batch_threads_input_array_and_usage_accounting() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(|req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            // Verify the request shape includes both inputs as one
            // array — F10 mandates batch endpoint coalescing.
            let inputs = body
                .get("input")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            assert_eq!(inputs.len(), 2, "batch must send N inputs in one call");
            ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "embedding": [0.1, 0.2], "index": 0 },
                    { "embedding": [0.3, 0.4], "index": 1 }
                ],
                "model": "text-embedding-3-small",
                "usage": { "prompt_tokens": 11, "total_tokens": 11 }
            }))
        })
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedder::custom("text-embedding-3-small", 2)
        .with_credentials(provider())
        .with_base_url(server.uri())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let out = embedder
        .embed_batch(&["a".to_owned(), "b".to_owned()], &ctx)
        .await
        .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].vector, vec![0.1, 0.2]);
    assert_eq!(out[1].vector, vec![0.3, 0.4]);
    // Per-call usage attributed to slot 0 only — downstream meters
    // sum across the batch and would double-charge if replicated.
    assert_eq!(out[0].usage, Some(EmbeddingUsage::new(11)));
    assert_eq!(out[1].usage, None);
}

#[tokio::test]
async fn embed_with_dimension_override_threads_into_request_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(|req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            assert_eq!(
                body.get("dimensions").and_then(|v| v.as_u64()),
                Some(512),
                "operator dimension override must thread to wire"
            );
            ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "embedding": vec![0.0; 512], "index": 0 }],
                "model": "text-embedding-3-small",
                "usage": { "prompt_tokens": 1, "total_tokens": 1 }
            }))
        })
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedder::small()
        .with_credentials(provider())
        .with_base_url(server.uri())
        .with_dimension(512)
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let out = embedder.embed("hi", &ctx).await.unwrap();
    assert_eq!(out.vector.len(), 512);
}

#[tokio::test]
async fn http_error_surfaces_as_provider_error_with_truncated_body() {
    let server = MockServer::start().await;
    let huge = "x".repeat(100_000);
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(500).set_body_string(huge))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedder::small()
        .with_credentials(provider())
        .with_base_url(server.uri())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let err = embedder.embed("hi", &ctx).await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("500"), "{msg}");
    assert!(
        msg.contains("truncated"),
        "oversized response body must be truncated in error message: {msg}"
    );
    // The full 100k-byte body must NOT ride into the error string.
    assert!(
        msg.len() < 2_000,
        "error message too long: {} bytes",
        msg.len()
    );
}

#[tokio::test]
async fn cancelled_context_short_circuits_before_http_call() {
    // ExecutionContext::is_cancelled() must be honoured. We don't
    // mount a Mock — if the embedder reaches the HTTP layer the
    // test fails with a Network error, not Cancelled.
    let server = MockServer::start().await;

    let embedder = OpenAiEmbedder::small()
        .with_credentials(provider())
        .with_base_url(server.uri())
        .build()
        .unwrap();

    let ctx = ExecutionContext::new();
    ctx.cancellation().cancel();
    let err = embedder.embed("hi", &ctx).await.unwrap_err();
    assert!(
        matches!(err, entelix_core::Error::Cancelled),
        "expected Cancelled, got: {err:?}"
    );
}

#[tokio::test]
async fn empty_batch_short_circuits() {
    // No HTTP call must be made for an empty input slice — an
    // unconfigured Mock would surface as a Network error and fail
    // the test.
    let server = MockServer::start().await;

    let embedder = OpenAiEmbedder::small()
        .with_credentials(provider())
        .with_base_url(server.uri())
        .build()
        .unwrap();
    let ctx = ExecutionContext::new();
    let out = embedder.embed_batch(&[], &ctx).await.unwrap();
    assert!(out.is_empty());
}
