//! Live-API smoke for [`ClaudeCodeOAuthProvider`] +
//! [`AnthropicMessagesCodec`] + [`DirectTransport`].
//!
//! `#[ignore]`-gated so a default `cargo test` skips it. The
//! operator opts in by ensuring the `claude` CLI is logged in
//! (the file `~/.claude/.credentials.json` exists) and running:
//!
//! ```text
//! cargo test -p entelix-auth-claude-code \
//!     --test live_oauth_round_trip -- --ignored
//! ```
//!
//! Optional knobs:
//! - `ENTELIX_LIVE_CLAUDE_CODE_MODEL` — override the default
//!   `claude-haiku-4-5` model id when the operator's Claude.ai
//!   subscription only has access to a different SKU.
//! - `ENTELIX_LIVE_CLAUDE_CODE_PATH` — point at an alternate
//!   credential file (CI fixtures, multi-account setups). When
//!   unset, `FileCredentialStore::default_claude_path()` resolves
//!   `~/.claude/.credentials.json`.
//!
//! ## What is verified
//!
//! 1. [`FileCredentialStore`] reads the on-disk envelope the
//!    upstream `claude` CLI shares.
//! 2. [`ClaudeCodeOAuthProvider::resolve`] produces an
//!    `Authorization: Bearer <token>` header — refreshing through
//!    the OAuth2 `refresh_token` grant when the cached access
//!    token is within 60 seconds of expiry.
//! 3. The matching `anthropic-beta: claude-code-20250219`
//!    capability flows from
//!    [`crate::CLAUDE_CODE_BETA`] through
//!    [`AnthropicExt::with_betas`] onto the outgoing request.
//! 4. The full pipe ([`AnthropicMessagesCodec`] +
//!    [`DirectTransport::anthropic`]) returns 2xx with a non-empty
//!    `ModelResponse` carrying a recognised `StopReason` and
//!    populated `Usage` counters — proof that the OAuth header
//!    plus beta header satisfy Anthropic's Claude Code gateway
//!    end-to-end.
//!
//! ## Cost discipline
//!
//! One non-streaming call with `max_tokens = 16` against the
//! Haiku tier. Anthropic's price sheet (Apr 2026) puts this in the
//! cents-per-thousand range; expected cost per run is well under
//! $0.001. Pro / Team Claude.ai subscribers run inside their plan
//! quota — no per-call billing.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::path::PathBuf;
use std::sync::Arc;

use entelix_auth_claude_code::{CLAUDE_CODE_BETA, ClaudeCodeOAuthProvider, FileCredentialStore};
use entelix_core::auth::CredentialProvider;
use entelix_core::codecs::{AnthropicMessagesCodec, Codec};
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{
    AnthropicExt, ContentPart, Message, ModelRequest, ProviderExtensions, StopReason,
};
use entelix_core::transports::{DirectTransport, Transport};

const DEFAULT_MODEL: &str = "claude-haiku-4-5";

fn resolve_credential_path() -> PathBuf {
    if let Ok(custom) = std::env::var("ENTELIX_LIVE_CLAUDE_CODE_PATH") {
        return PathBuf::from(custom);
    }
    FileCredentialStore::default_claude_path()
        .expect("HOME / USERPROFILE not set — supply ENTELIX_LIVE_CLAUDE_CODE_PATH to override")
}

#[tokio::test]
#[ignore = "live-API: requires `claude` CLI login (~/.claude/.credentials.json)"]
async fn claude_code_oauth_minimal_chat_round_trip() {
    install_default_tls();
    let path = resolve_credential_path();
    assert!(
        path.exists(),
        "credential file not found at {} — run `claude login` first or set ENTELIX_LIVE_CLAUDE_CODE_PATH",
        path.display()
    );

    let model = std::env::var("ENTELIX_LIVE_CLAUDE_CODE_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let store = FileCredentialStore::with_path(path);
    let provider = ClaudeCodeOAuthProvider::new(store);

    // Resolve once first so the test fails *fast* with an OAuth
    // error rather than a confusing 401 from the Anthropic
    // gateway. Refresh — when the access token is within 60
    // seconds of expiry — happens inside this call.
    let resolved = provider
        .resolve()
        .await
        .expect("ClaudeCodeOAuthProvider::resolve must succeed (run `claude login` if expired)");
    assert_eq!(
        resolved.header_name.as_str().to_ascii_lowercase(),
        "authorization",
        "OAuth provider must produce an `Authorization` header"
    );

    let credentials: Arc<dyn CredentialProvider> = Arc::new(provider);
    let codec = AnthropicMessagesCodec::new();
    let transport = DirectTransport::anthropic(credentials).expect("transport build");

    let request = ModelRequest {
        model,
        messages: vec![Message::user("Reply with the word 'ok' and nothing else.")],
        max_tokens: Some(16),
        temperature: Some(0.0),
        provider_extensions: ProviderExtensions::default()
            .with_anthropic(AnthropicExt::default().with_betas([CLAUDE_CODE_BETA])),
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&request).expect("anthropic encode");
    let ctx = ExecutionContext::new();
    let response = transport
        .send(encoded, &ctx)
        .await
        .expect("OAuth-authenticated send to api.anthropic.com");

    assert!(
        (200..300).contains(&response.status),
        "expected 2xx, got {} — body: {}. \
         Common causes: expired refresh token (re-run `claude login`), \
         missing `claude-code-20250219` beta header, \
         or the configured model is outside the subscription tier",
        response.status,
        String::from_utf8_lossy(&response.body)
    );

    let decoded = codec
        .decode(&response.body, Vec::new())
        .expect("anthropic decode");

    let saw_text = decoded
        .content
        .iter()
        .any(|p| matches!(p, ContentPart::Text { text, .. } if !text.is_empty()));
    assert!(saw_text, "expected at least one non-empty text block");

    assert!(
        matches!(
            decoded.stop_reason,
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence { .. }
        ),
        "unexpected stop_reason on minimal call: {:?}",
        decoded.stop_reason
    );

    assert!(
        decoded.usage.input_tokens > 0,
        "usage.input_tokens should be populated on a real OAuth call"
    );
    assert!(
        decoded.usage.output_tokens > 0,
        "usage.output_tokens should be populated on a real OAuth call"
    );
}
