//! OAuth2 `refresh_token` grant against the Anthropic console token
//! endpoint.
//!
//! Pure HTTP — no storage / no credential trait involvement.

use serde::{Deserialize, Serialize};

use crate::credential::OAuthCredential;
use crate::error::{ClaudeCodeAuthError, ClaudeCodeAuthResult};

/// The OAuth2 refresh-token grant request body.
///
/// Serialised as `application/x-www-form-urlencoded` per RFC 6749 §6.
#[derive(Debug, Serialize)]
struct RefreshRequest<'a> {
    grant_type: &'static str,
    refresh_token: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    client_id: Option<&'a str>,
}

/// Token-endpoint response.
///
/// Anthropic returns the standard OAuth2 shape plus a
/// `refresh_token` (token rotation: every refresh may mint a fresh
/// refresh token).
#[derive(Debug, Deserialize)]
struct RefreshResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    /// Seconds-from-now until the new access token expires.
    expires_in: i64,
    #[serde(default)]
    scope: Option<String>,
}

/// Exchange a refresh token for a fresh `OAuthCredential` via the
/// Anthropic console token endpoint.
///
/// The returned credential carries only the fields the server
/// returned; callers merge any pre-existing metadata
/// (`subscription_type`, prior `scopes`, fallback `refresh_token`
/// when the server omits rotation) before persisting.
pub(super) async fn refresh_access_token(
    http: &reqwest::Client,
    token_url: &str,
    refresh_token: &str,
    client_id: Option<&str>,
) -> ClaudeCodeAuthResult<OAuthCredential> {
    let body = RefreshRequest {
        grant_type: "refresh_token",
        refresh_token,
        client_id,
    };
    let body_str =
        serde_urlencoded::to_string(&body).map_err(|e| ClaudeCodeAuthError::RefreshHttp {
            message: format!("encode form body: {e}"),
        })?;
    let response = http
        .post(token_url)
        .header(
            http::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(body_str)
        .send()
        .await
        .map_err(|e| ClaudeCodeAuthError::RefreshHttp {
            message: e.to_string(),
        })?;
    let status = response.status();
    let payload = response
        .text()
        .await
        .map_err(|e| ClaudeCodeAuthError::RefreshHttp {
            message: format!("read body: {e}"),
        })?;
    if !status.is_success() {
        return Err(ClaudeCodeAuthError::RefreshHttp {
            message: format!("status {status}: {payload}"),
        });
    }
    let parsed: RefreshResponse =
        serde_json::from_str(&payload).map_err(|e| ClaudeCodeAuthError::RefreshHttp {
            message: format!("parse body: {e}"),
        })?;
    let expires_at_ms = compute_expires_at_ms(parsed.expires_in)?;
    let scopes: Vec<String> = parsed
        .scope
        .as_deref()
        .map(|s| s.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();
    let mut credential = OAuthCredential::new(parsed.access_token, expires_at_ms);
    if let Some(token) = parsed.refresh_token {
        credential = credential.with_refresh_token(token);
    }
    if !scopes.is_empty() {
        credential = credential.with_scopes(scopes);
    }
    Ok(credential)
}

/// Maximum plausible `expires_in` from the token endpoint. OAuth2
/// access tokens are short-lived by design; values beyond a year
/// signal a server bug or response-shape change. Server-returned
/// non-positive values are equally invalid (would mint a credential
/// that's already expired and trigger a refresh loop).
const MAX_EXPIRES_IN_SECONDS: i64 = 365 * 24 * 60 * 60;

fn compute_expires_at_ms(expires_in_seconds: i64) -> ClaudeCodeAuthResult<i64> {
    if expires_in_seconds <= 0 || expires_in_seconds > MAX_EXPIRES_IN_SECONDS {
        return Err(ClaudeCodeAuthError::RefreshHttp {
            message: format!(
                "server returned implausible expires_in={expires_in_seconds}s \
                 (expected 0 < expires_in ≤ {MAX_EXPIRES_IN_SECONDS}s)"
            ),
        });
    }
    let now_ms = chrono::Utc::now().timestamp_millis();
    let expires_in_ms =
        expires_in_seconds
            .checked_mul(1000)
            .ok_or_else(|| ClaudeCodeAuthError::RefreshHttp {
                message: format!("expires_in={expires_in_seconds}s overflows i64 milliseconds"),
            })?;
    now_ms
        .checked_add(expires_in_ms)
        .ok_or_else(|| ClaudeCodeAuthError::RefreshHttp {
            message: "expires_at_ms overflows i64".into(),
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn successful_refresh_returns_credential_with_future_expiry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "scope": "user:inference user:profile"
            })))
            .mount(&server)
            .await;
        let http = reqwest::Client::new();
        let url = format!("{}/oauth/token", server.uri());
        let cred = refresh_access_token(&http, &url, "old-refresh", None)
            .await
            .unwrap();
        assert_eq!(cred.access_token, "new-access");
        assert_eq!(cred.refresh_token.as_deref(), Some("new-refresh"));
        assert!(cred.scopes.contains(&"user:inference".to_owned()));
        assert!(!cred.needs_refresh());
    }

    #[tokio::test]
    async fn server_error_surfaces_as_refresh_http() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(401).set_body_string("invalid_grant"))
            .mount(&server)
            .await;
        let http = reqwest::Client::new();
        let url = format!("{}/oauth/token", server.uri());
        let err = refresh_access_token(&http, &url, "old-refresh", None)
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeCodeAuthError::RefreshHttp { .. }));
    }

    #[tokio::test]
    async fn negative_expires_in_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "x",
                "expires_in": -10
            })))
            .mount(&server)
            .await;
        let http = reqwest::Client::new();
        let url = format!("{}/oauth/token", server.uri());
        let err = refresh_access_token(&http, &url, "r", None)
            .await
            .unwrap_err();
        let ClaudeCodeAuthError::RefreshHttp { message } = err else {
            panic!("expected RefreshHttp");
        };
        assert!(message.contains("implausible"), "got: {message}");
    }

    #[tokio::test]
    async fn excessive_expires_in_rejected() {
        // 10 years — beyond the 1-year plausibility cap.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "x",
                "expires_in": 315_360_000_i64
            })))
            .mount(&server)
            .await;
        let http = reqwest::Client::new();
        let url = format!("{}/oauth/token", server.uri());
        let err = refresh_access_token(&http, &url, "r", None)
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeCodeAuthError::RefreshHttp { .. }));
    }
}
