//! Smoke tests for the auth module.
//!
//! Covers `ApiKeyProvider` (Anthropic-style header), `BearerProvider`
//! (`Authorization: Bearer ...` form), and the `Error::Config` path.

#![allow(clippy::unwrap_used)]

use entelix_core::auth::{ApiKeyProvider, BearerProvider, CredentialProvider};
use secrecy::ExposeSecret;

#[tokio::test]
async fn anthropic_provider_emits_x_api_key() {
    let p = ApiKeyProvider::anthropic("sk-test");
    let creds = p.resolve().await.unwrap();
    assert_eq!(creds.header_name.as_str(), "x-api-key");
    assert_eq!(creds.header_value.expose_secret(), "sk-test");
}

#[tokio::test]
async fn custom_header_name_accepted() {
    let p = ApiKeyProvider::new("x-custom-key", "abc123").unwrap();
    let creds = p.resolve().await.unwrap();
    assert_eq!(creds.header_name.as_str(), "x-custom-key");
}

#[tokio::test]
async fn invalid_header_name_returns_config_error() {
    // Spaces are invalid in HTTP header names.
    let err = ApiKeyProvider::new("bad header", "k").unwrap_err();
    assert!(matches!(err, entelix_core::Error::Config(_)));
}

#[tokio::test]
async fn bearer_provider_prepends_bearer_prefix() {
    let p = BearerProvider::new("token-xyz");
    let creds = p.resolve().await.unwrap();
    assert_eq!(creds.header_name.as_str(), "authorization");
    assert_eq!(creds.header_value.expose_secret(), "Bearer token-xyz");
}

#[tokio::test]
async fn provider_object_safe_via_arc_dyn() {
    use std::sync::Arc;
    let providers: Vec<Arc<dyn CredentialProvider>> = vec![
        Arc::new(ApiKeyProvider::anthropic("k")),
        Arc::new(BearerProvider::new("t")),
    ];
    for p in &providers {
        let creds = p.resolve().await.unwrap();
        assert!(!creds.header_name.as_str().is_empty());
    }
}
