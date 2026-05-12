//! Regression for [`Error::envelope`] — the canonical typed wire
//! shape that integrators read at sink / SSE / audit boundaries.
//!
//! `ErrorEnvelope::wire_code` is a public-API stability promise:
//! changing one is a breaking change for integrators that key i18n
//! catalogues or metric labels off it. The cases below pin every
//! active [`Error`] variant plus the HTTP status family-bucket
//! boundaries (400/429/500) that determine how new vendor statuses
//! absorb into the right class. The `retry_after_secs` /
//! `provider_status` fields exposed by the envelope are pinned
//! against the same variants.

#![allow(clippy::unwrap_used)]

use std::time::Duration;

use entelix_core::auth::AuthError;
use entelix_core::interruption::InterruptionKind;
use entelix_core::run_budget::UsageLimitBreach;
use entelix_core::{Error, ErrorClass, LlmRenderable};
use serde_json::json;

#[test]
fn invalid_request_maps_to_client() {
    let env = Error::invalid_request("bad").envelope();
    assert_eq!(env.wire_code, "invalid_request");
    assert_eq!(env.wire_class, ErrorClass::Client);
    assert!(env.retry_after_secs.is_none());
    assert!(env.provider_status.is_none());
}

#[test]
fn config_maps_to_server() {
    let env = Error::config("missing key").envelope();
    assert_eq!(env.wire_code, "config_error");
    assert_eq!(env.wire_class, ErrorClass::Server);
}

#[test]
fn provider_http_429_is_rate_limited_client() {
    let env = Error::provider_http(429, "too many").envelope();
    assert_eq!(env.wire_code, "rate_limited");
    assert_eq!(env.wire_class, ErrorClass::Client);
    assert_eq!(env.provider_status, Some(429));
}

#[test]
fn provider_http_401_and_403_collapse_to_upstream_unauthorized() {
    for status in [401u16, 403] {
        let env = Error::provider_http(status, "denied").envelope();
        assert_eq!(
            env.wire_code, "upstream_unauthorized",
            "{status} should map to upstream_unauthorized"
        );
        assert_eq!(env.wire_class, ErrorClass::Client);
        assert_eq!(env.provider_status, Some(status));
    }
}

#[test]
fn provider_http_4xx_default_is_upstream_invalid_client() {
    // 400 / 404 / 422 / 451 — every other 4xx falls into the
    // generic-invalid bucket. New 4xx codes (e.g. RFC 9110 additions)
    // absorb here without an SDK release.
    for status in [400u16, 404, 408, 422, 451] {
        let env = Error::provider_http(status, "boom").envelope();
        assert_eq!(env.wire_code, "upstream_invalid", "{status}");
        assert_eq!(env.wire_class, ErrorClass::Client);
        assert_eq!(env.provider_status, Some(status));
    }
}

#[test]
fn provider_http_5xx_is_upstream_unavailable_server() {
    for status in [500u16, 502, 503, 504] {
        let env = Error::provider_http(status, "down").envelope();
        assert_eq!(env.wire_code, "upstream_unavailable", "{status}");
        assert_eq!(env.wire_class, ErrorClass::Server);
        assert_eq!(env.provider_status, Some(status));
    }
}

#[test]
fn provider_http_non_terminal_status_coerces_to_network() {
    // Status `0` / 1xx / 2xx / 3xx / ≥600 do not represent a
    // terminal vendor response; the constructor coerces them to
    // `ProviderErrorKind::Network` so the wire identifier reflects
    // "we never received a terminal response" rather than a
    // plausible-looking `upstream_error` (invariant 15). The coerced
    // shape carries no `provider_status` — the original numeric is
    // intentionally dropped because it never represented a vendor
    // response.
    for status in [0u16, 100, 199, 200, 304, 600, 999] {
        let env = Error::provider_http(status, "anomalous").envelope();
        assert_eq!(
            env.wire_code, "transport_failure",
            "{status} should coerce to transport_failure"
        );
        assert_eq!(env.wire_class, ErrorClass::Server);
        assert!(
            env.provider_status.is_none(),
            "{status} coerced to Network → no provider_status"
        );
    }
}

#[test]
fn provider_transport_class_maps_to_server() {
    let cases = [
        (Error::provider_network("reset"), "transport_failure"),
        (Error::provider_tls("handshake"), "tls_failure"),
        (Error::provider_dns("no host"), "dns_failure"),
    ];
    for (err, expected) in cases {
        let env = err.envelope();
        assert_eq!(env.wire_code, expected);
        assert_eq!(
            env.wire_class,
            ErrorClass::Server,
            "transport-class failures are server-actionable: {expected}"
        );
        assert!(
            env.provider_status.is_none(),
            "{expected} has no HTTP status"
        );
    }
}

#[test]
fn auth_maps_to_client() {
    let env = Error::Auth(AuthError::missing()).envelope();
    assert_eq!(env.wire_code, "auth_failed");
    assert_eq!(env.wire_class, ErrorClass::Client);
}

#[test]
fn cancelled_maps_to_client() {
    let env = Error::Cancelled.envelope();
    assert_eq!(env.wire_code, "cancelled");
    assert_eq!(env.wire_class, ErrorClass::Client);
}

#[test]
fn deadline_exceeded_maps_to_server() {
    let env = Error::DeadlineExceeded.envelope();
    assert_eq!(env.wire_code, "deadline_exceeded");
    assert_eq!(env.wire_class, ErrorClass::Server);
}

#[test]
fn interrupted_maps_to_client() {
    let err = Error::Interrupted {
        kind: InterruptionKind::Custom,
        payload: json!({"reason": "human"}),
    };
    let env = err.envelope();
    assert_eq!(env.wire_code, "interrupted");
    assert_eq!(env.wire_class, ErrorClass::Client);
}

#[test]
fn model_retry_maps_to_client() {
    let hint = "schema mismatch — return only the JSON object".to_owned();
    let env = Error::model_retry(hint.for_llm(), 3).envelope();
    assert_eq!(env.wire_code, "model_retry_exhausted");
    assert_eq!(env.wire_class, ErrorClass::Client);
}

#[test]
fn serde_maps_to_server() {
    let env = serde_json::from_str::<serde_json::Value>("not json")
        .map_err(Error::from)
        .unwrap_err()
        .envelope();
    assert_eq!(env.wire_code, "serde");
    assert_eq!(env.wire_class, ErrorClass::Server);
}

#[test]
fn usage_limit_exceeded_maps_to_client() {
    let env = Error::UsageLimitExceeded(UsageLimitBreach::Requests {
        limit: 10,
        observed: 11,
    })
    .envelope();
    assert_eq!(env.wire_code, "quota_exceeded");
    assert_eq!(env.wire_class, ErrorClass::Client);
}

#[test]
fn retry_after_does_not_alter_wire_shape() {
    // Attaching a `retry_after` hint is orthogonal to wire identity —
    // a 429 stays "rate_limited" / Client regardless of whether the
    // vendor sent a Retry-After header.
    let bare = Error::provider_http(429, "x").envelope();
    let with_hint = Error::provider_http(429, "x")
        .with_retry_after(Duration::from_secs(5))
        .envelope();
    assert_eq!(bare.wire_code, with_hint.wire_code);
    assert_eq!(bare.wire_class, with_hint.wire_class);
    assert_eq!(bare.provider_status, with_hint.provider_status);
    assert_eq!(bare.retry_after_secs, None);
    assert_eq!(with_hint.retry_after_secs, Some(5));
}

#[test]
fn retry_after_truncates_sub_second_to_whole_seconds() {
    // `Duration` → whole-second conversion is `as_secs` (truncation).
    // 4_999ms → 4s; 5_500ms → 5s. Sub-second resolution is dropped
    // because the wire shape exposes `Option<u64>`, not `Duration`.
    let env = Error::provider_http(429, "x")
        .with_retry_after(Duration::from_millis(4_999))
        .envelope();
    assert_eq!(env.retry_after_secs, Some(4));
}

#[test]
fn retry_after_on_non_provider_variant_is_no_op() {
    // `with_retry_after` returns self unchanged for non-Provider
    // variants — the envelope mirrors that as `None`.
    let env = Error::Cancelled
        .with_retry_after(Duration::from_secs(7))
        .envelope();
    assert!(env.retry_after_secs.is_none());
}

#[test]
fn provider_status_is_none_for_transport_class_failures() {
    for err in [
        Error::provider_network("reset"),
        Error::provider_tls("handshake"),
        Error::provider_dns("no host"),
    ] {
        assert!(err.envelope().provider_status.is_none());
    }
}

#[test]
fn error_class_display_is_lowercase_stable() {
    assert_eq!(format!("{}", ErrorClass::Client), "client");
    assert_eq!(format!("{}", ErrorClass::Server), "server");
}

#[test]
fn envelope_serialises_to_json_with_lowercase_class_and_omitted_nones() {
    // The envelope is the "canonical inspector at sink / SSE / audit
    // boundaries" — sinks that forward to JSON must observe a stable
    // shape: snake_case fields, lowercase wire_class, absent options
    // skipped.
    let env = Error::provider_http(429, "x")
        .with_retry_after(Duration::from_secs(7))
        .envelope();
    let v = serde_json::to_value(env).unwrap();
    assert_eq!(v.get("wire_code").unwrap(), "rate_limited");
    assert_eq!(v.get("wire_class").unwrap(), "client");
    assert_eq!(v.get("retry_after_secs").unwrap(), 7);
    assert_eq!(v.get("provider_status").unwrap(), 429);

    // None branches omit the key entirely so SSE payloads stay lean.
    let env = Error::Cancelled.envelope();
    let v = serde_json::to_value(env).unwrap();
    assert_eq!(v.get("wire_code").unwrap(), "cancelled");
    assert_eq!(v.get("wire_class").unwrap(), "client");
    assert!(
        v.get("retry_after_secs").is_none(),
        "None retry_after_secs must be omitted from JSON, got: {v:?}"
    );
    assert!(
        v.get("provider_status").is_none(),
        "None provider_status must be omitted from JSON, got: {v:?}"
    );
}

#[test]
fn envelope_is_copy_value_semantics() {
    // `ErrorEnvelope` is `Copy` by design — sinks pass it by value
    // through fan-out without `.clone()` ceremony.
    let env = Error::provider_http(429, "x").envelope();
    let again = env;
    assert_eq!(env.wire_code, again.wire_code);
    assert_eq!(env.provider_status, again.provider_status);
}
