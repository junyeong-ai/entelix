//! `25_claude_code_oauth` — drive entelix with the OAuth token the
//! `claude` CLI manages, instead of minting an Anthropic API key.
//!
//! Build (default features omit the OAuth provider):
//! `cargo build --example 25_claude_code_oauth -p entelix --features auth-claude-code`
//! Run (hermetic — composes the credential flow without contacting Anthropic):
//! `cargo run --example 25_claude_code_oauth -p entelix --features auth-claude-code`
//!
//! Operators with a Claude.ai pro / team subscription drive entelix
//! without minting a separate Anthropic API key — the same
//! credential the `claude` CLI uses flows straight into the
//! `entelix_core::auth::CredentialProvider` chain. Refresh-token
//! rotation is automatic; storage stays compatible with the
//! upstream CLI so both tools share state.
//!
//! What this example demonstrates:
//!
//! 1. Resolve the default credential file path against `HOME` /
//!    `USERPROFILE` via [`FileCredentialStore::default_claude_path`].
//! 2. Build a [`ClaudeCodeOAuthProvider`] over the file-backed
//!    [`FileCredentialStore`].
//! 3. Resolve once to surface the `Authorization: Bearer …` header
//!    the transport will attach to every outgoing request.
//! 4. Compose the matching `anthropic-beta: claude-code-20250219`
//!    capability through [`entelix::ir::AnthropicExt::with_betas`]
//!    paired with [`CLAUDE_CODE_BETA`] — credential and codec ext
//!    stay independent (single responsibility).
//!
//! No Anthropic call happens here. In production, hand the provider
//! to a `DirectTransport` plus `AnthropicMessagesCodec` and bind the
//! `ProviderExtensions` onto the `ModelRequest` you build.

#![allow(clippy::print_stdout)]

use entelix::auth::CredentialProvider;
use entelix::ir::{AnthropicExt, ProviderExtensions};
use entelix::{CLAUDE_CODE_BETA, ClaudeCodeOAuthProvider, FileCredentialStore};

#[tokio::main]
async fn main() -> entelix::Result<()> {
    // (1) Resolve `~/.claude/.credentials.json` against the host's
    //     home directory. Returns a typed error in CI sandboxes
    //     that don't expose `HOME`.
    let path = match FileCredentialStore::default_claude_path() {
        Ok(p) => p,
        Err(err) => {
            println!("home directory unresolved — supply a path explicitly: {err}");
            return Ok(());
        }
    };
    println!("credential file: {}", path.display());

    // (2) Build a provider over the file-backed store. Refresh is
    //     automatic — the provider hits the Anthropic console
    //     endpoint when the access token is within 60 seconds of
    //     expiry, persists the rotated tokens through the same
    //     store the `claude` CLI reads.
    let store = FileCredentialStore::with_path(path);
    let provider = ClaudeCodeOAuthProvider::new(store);

    // (3) Resolve once to demonstrate the OAuth flow. In production
    //     the transport calls `resolve()` per request; refresh is
    //     transparent to the call site.
    match provider.resolve().await {
        Ok(creds) => {
            println!(
                "resolved credential header: {} (value redacted)",
                creds.header_name
            );
        }
        Err(err) => {
            println!("not authenticated yet — run `claude login` first: {err}");
            return Ok(());
        }
    }

    // (4) Compose the matching `anthropic-beta` capability so every
    //     outgoing `ModelRequest` carries the header Claude Code's
    //     tokens require. The provider deliberately does not inject
    //     the header itself — credential and codec ext stay
    //     independent.
    let extensions = ProviderExtensions::default()
        .with_anthropic(AnthropicExt::default().with_betas([CLAUDE_CODE_BETA]));
    println!("paired ProviderExtensions: {extensions:?}");
    println!(
        "wire `provider` into a DirectTransport over \
         https://api.anthropic.com/v1/messages and bind `extensions` \
         onto every `ModelRequest::provider_extensions`."
    );

    Ok(())
}
