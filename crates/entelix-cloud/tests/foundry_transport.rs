//! `FoundryTransport` end-to-end via wiremock. Validates both auth
//! modes (API key + Entra/OAuth) and 401 cache invalidation.

#![cfg(feature = "azure")]
#![allow(clippy::unwrap_used, clippy::missing_const_for_fn)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use entelix_cloud::CloudError;
use entelix_cloud::foundry::{FoundryAuth, FoundryTransport};
use entelix_cloud::refresh::{TokenRefresher, TokenSnapshot};
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
            expires_at: Instant::now() + Duration::from_mins(60),
        })
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn api_key_auth_sends_api_key_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("api-key", "sk-foundry-test"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"ok\":true}"))
        .mount(&server)
        .await;

    let transport = FoundryTransport::builder()
        .with_base_url(server.uri())
        .with_auth(FoundryAuth::ApiKey {
            token: SecretString::from("sk-foundry-test".to_owned()),
        })
        .build()
        .unwrap();

    let req = EncodedRequest::post_json("/anthropic/v1/messages", Bytes::from_static(b"{}"));
    let resp = transport.send(req, &ExecutionContext::new()).await.unwrap();
    assert_eq!(resp.status, 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn entra_auth_uses_bearer_authorization_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("authorization", "Bearer entra-token"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"ok\":true}"))
        .mount(&server)
        .await;

    let transport = FoundryTransport::builder()
        .with_base_url(server.uri())
        .with_auth(FoundryAuth::Entra {
            refresher: Arc::new(StaticRefresher::new("entra-token")),
        })
        .build()
        .unwrap();

    let req = EncodedRequest::post_json("/x", Bytes::from_static(b"{}"));
    let resp = transport.send(req, &ExecutionContext::new()).await.unwrap();
    assert_eq!(resp.status, 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn entra_caches_token_across_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let refresher = Arc::new(StaticRefresher::new("entra-cached"));
    let transport = FoundryTransport::builder()
        .with_base_url(server.uri())
        .with_auth(FoundryAuth::Entra {
            refresher: refresher.clone(),
        })
        .build()
        .unwrap();

    for _ in 0..3 {
        let req = EncodedRequest::post_json("/x", Bytes::from_static(b"{}"));
        transport.send(req, &ExecutionContext::new()).await.unwrap();
    }
    assert_eq!(refresher.call_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn entra_unauthorized_invalidates_cache() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;

    let refresher = Arc::new(StaticRefresher::new("rotated"));
    let transport = FoundryTransport::builder()
        .with_base_url(server.uri())
        .with_auth(FoundryAuth::Entra {
            refresher: refresher.clone(),
        })
        .build()
        .unwrap();

    transport
        .send(
            EncodedRequest::post_json("/x", Bytes::from_static(b"{}")),
            &ExecutionContext::new(),
        )
        .await
        .unwrap();
    transport
        .send(
            EncodedRequest::post_json("/x", Bytes::from_static(b"{}")),
            &ExecutionContext::new(),
        )
        .await
        .unwrap();
    assert_eq!(refresher.call_count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_base_url_in_builder_errors() {
    let result = FoundryTransport::builder()
        .with_auth(FoundryAuth::ApiKey {
            token: SecretString::from("k".to_owned()),
        })
        .build();
    match result {
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("base_url"), "expected base_url error, got {s}");
        }
        Ok(_) => panic!("expected build to fail when base_url is missing"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_token_short_circuits_send() {
    let transport = FoundryTransport::builder()
        .with_base_url("http://127.0.0.1:1")
        .with_auth(FoundryAuth::ApiKey {
            token: SecretString::from("k".to_owned()),
        })
        .build()
        .unwrap();
    let context = ExecutionContext::new();
    context.cancellation().cancel();
    let req = EncodedRequest::post_json("/x", Bytes::from_static(b"{}"));
    let err = transport.send(req, &context).await.unwrap_err();
    assert!(
        matches!(err, entelix_core::Error::Cancelled),
        "expected Error::Cancelled, got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_token_short_circuits_send_streaming() {
    let transport = FoundryTransport::builder()
        .with_base_url("http://127.0.0.1:1")
        .with_auth(FoundryAuth::ApiKey {
            token: SecretString::from("k".to_owned()),
        })
        .build()
        .unwrap();
    let context = ExecutionContext::new();
    context.cancellation().cancel();
    let req = EncodedRequest::post_json("/x", Bytes::from_static(b"{}"));
    let err = transport.send_streaming(req, &context).await.unwrap_err();
    assert!(
        matches!(err, entelix_core::Error::Cancelled),
        "expected Error::Cancelled, got {err:?}"
    );
}
