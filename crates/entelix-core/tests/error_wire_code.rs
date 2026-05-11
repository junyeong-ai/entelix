//! Regression for [`Error::wire_code`] and [`Error::wire_class`].
//!
//! Wire codes are a public-API stability promise: changing one is a
//! breaking change for integrators that key i18n catalogues or metric
//! labels off them. The cases below pin every active [`Error`] variant
//! plus the HTTP status family-bucket boundaries (400/429/500) that
//! determine how new vendor statuses absorb into the right class.

#![allow(clippy::unwrap_used)]

use std::time::Duration;

use entelix_core::auth::AuthError;
use entelix_core::interruption::InterruptionKind;
use entelix_core::run_budget::UsageLimitBreach;
use entelix_core::{Error, ErrorClass, LlmRenderable};
use serde_json::json;

#[test]
fn invalid_request_maps_to_client() {
    let err = Error::invalid_request("bad");
    assert_eq!(err.wire_code(), "invalid_request");
    assert_eq!(err.wire_class(), ErrorClass::Client);
}

#[test]
fn config_maps_to_server() {
    let err = Error::config("missing key");
    assert_eq!(err.wire_code(), "config_error");
    assert_eq!(err.wire_class(), ErrorClass::Server);
}

#[test]
fn provider_http_429_is_rate_limited_client() {
    let err = Error::provider_http(429, "too many");
    assert_eq!(err.wire_code(), "rate_limited");
    assert_eq!(err.wire_class(), ErrorClass::Client);
}

#[test]
fn provider_http_401_and_403_collapse_to_upstream_unauthorized() {
    for status in [401u16, 403] {
        let err = Error::provider_http(status, "denied");
        assert_eq!(
            err.wire_code(),
            "upstream_unauthorized",
            "{status} should map to upstream_unauthorized"
        );
        assert_eq!(err.wire_class(), ErrorClass::Client);
    }
}

#[test]
fn provider_http_4xx_default_is_upstream_invalid_client() {
    // 400 / 404 / 422 / 451 — every other 4xx falls into the
    // generic-invalid bucket. New 4xx codes (e.g. RFC 9110 additions)
    // absorb here without an SDK release.
    for status in [400u16, 404, 408, 422, 451] {
        let err = Error::provider_http(status, "boom");
        assert_eq!(err.wire_code(), "upstream_invalid", "{status}");
        assert_eq!(err.wire_class(), ErrorClass::Client);
    }
}

#[test]
fn provider_http_5xx_is_upstream_unavailable_server() {
    for status in [500u16, 502, 503, 504] {
        let err = Error::provider_http(status, "down");
        assert_eq!(err.wire_code(), "upstream_unavailable", "{status}");
        assert_eq!(err.wire_class(), ErrorClass::Server);
    }
}

#[test]
fn provider_http_unknown_family_is_upstream_error_server() {
    // 1xx / 2xx / 3xx / 6xx never arrive as a `Provider::Http` in
    // practice but the family-bucket fallback exists for vendor-drift
    // safety. The bucket lands on Server (operator-actionable).
    let err = Error::provider_http(600, "anomalous");
    assert_eq!(err.wire_code(), "upstream_error");
    assert_eq!(err.wire_class(), ErrorClass::Server);
}

#[test]
fn provider_transport_class_maps_to_server() {
    let cases = [
        (Error::provider_network("reset"), "transport_failure"),
        (Error::provider_tls("handshake"), "tls_failure"),
        (Error::provider_dns("no host"), "dns_failure"),
    ];
    for (err, expected) in cases {
        assert_eq!(err.wire_code(), expected);
        assert_eq!(
            err.wire_class(),
            ErrorClass::Server,
            "transport-class failures are server-actionable: {expected}"
        );
    }
}

#[test]
fn auth_maps_to_client() {
    let err = Error::Auth(AuthError::missing());
    assert_eq!(err.wire_code(), "auth_failed");
    assert_eq!(err.wire_class(), ErrorClass::Client);
}

#[test]
fn cancelled_maps_to_client() {
    let err = Error::Cancelled;
    assert_eq!(err.wire_code(), "cancelled");
    assert_eq!(err.wire_class(), ErrorClass::Client);
}

#[test]
fn deadline_exceeded_maps_to_server() {
    let err = Error::DeadlineExceeded;
    assert_eq!(err.wire_code(), "deadline_exceeded");
    assert_eq!(err.wire_class(), ErrorClass::Server);
}

#[test]
fn interrupted_maps_to_client() {
    let err = Error::Interrupted {
        kind: InterruptionKind::Custom,
        payload: json!({"reason": "human"}),
    };
    assert_eq!(err.wire_code(), "interrupted");
    assert_eq!(err.wire_class(), ErrorClass::Client);
}

#[test]
fn model_retry_maps_to_client() {
    let hint = "schema mismatch — return only the JSON object".to_owned();
    let err = Error::model_retry(hint.for_llm(), 3);
    assert_eq!(err.wire_code(), "model_retry_exhausted");
    assert_eq!(err.wire_class(), ErrorClass::Client);
}

#[test]
fn serde_maps_to_server() {
    let err = serde_json::from_str::<serde_json::Value>("not json")
        .map_err(Error::from)
        .unwrap_err();
    assert_eq!(err.wire_code(), "serde");
    assert_eq!(err.wire_class(), ErrorClass::Server);
}

#[test]
fn usage_limit_exceeded_maps_to_client() {
    let err = Error::UsageLimitExceeded(UsageLimitBreach::Requests {
        limit: 10,
        observed: 11,
    });
    assert_eq!(err.wire_code(), "quota_exceeded");
    assert_eq!(err.wire_class(), ErrorClass::Client);
}

#[test]
fn retry_after_does_not_alter_wire_shape() {
    // Attaching a `retry_after` hint is orthogonal to wire identity —
    // a 429 stays "rate_limited" / Client regardless of whether the
    // vendor sent a Retry-After header.
    let bare = Error::provider_http(429, "x");
    let with_hint = Error::provider_http(429, "x").with_retry_after(Duration::from_secs(5));
    assert_eq!(bare.wire_code(), with_hint.wire_code());
    assert_eq!(bare.wire_class(), with_hint.wire_class());
}

#[test]
fn error_class_display_is_lowercase_stable() {
    assert_eq!(format!("{}", ErrorClass::Client), "client");
    assert_eq!(format!("{}", ErrorClass::Server), "server");
}
