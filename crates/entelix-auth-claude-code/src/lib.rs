//! # entelix-auth-claude-code
//!
//! Claude Code OAuth credential provider for entelix.
//!
//! Reuses the Claude.ai access token the `claude` CLI manages
//! (`~/.claude/.credentials.json` by default), refreshing through
//! the standard OAuth2 `refresh_token` grant against
//! `https://console.anthropic.com/v1/oauth/token` when the access
//! token approaches expiry.
//!
//! ## Why
//!
//! Operators with a Claude.ai pro / team subscription get to drive
//! entelix without minting a separate Anthropic API key — the same
//! credential their `claude` CLI uses flows straight into the
//! `entelix_core::auth::CredentialProvider` chain. Refresh-token
//! rotation is handled automatically and storage stays compatible
//! with the upstream CLI, so both tools share state.
//!
//! ## Layout
//!
//! ```text
//! ClaudeCodeOAuthProvider          — impl entelix_core::auth::CredentialProvider
//!   └─ CredentialStore trait       — operator-supplied backend
//!         └─ FileCredentialStore   — default; reads / writes the on-disk JSON
//!   └─ ClaudeCodeOAuthConfig       — token URL / client id / refresh timeout
//!   └─ refresh_access_token        — RFC 6749 §6 grant
//! ```
//!
//! [`CredentialStore`] is a trait so vault / Keychain / Secret
//! Service backends plug in by implementing it; the default ships
//! only the file backend so the dependency footprint stays minimal.
//!
//! ## Beta capability gate
//!
//! Claude Code's OAuth tokens require the `claude-code-20250219`
//! anthropic-beta header. The provider deliberately *does not*
//! inject the header — credentials and codec ext stay independent
//! (single responsibility). Operators wire it through
//! [`entelix_core::ir::AnthropicExt::with_betas`] using
//! [`CLAUDE_CODE_BETA`]:
//!
//! ```ignore
//! use entelix_auth_claude_code::{
//!     CLAUDE_CODE_BETA, ClaudeCodeOAuthProvider, FileCredentialStore,
//! };
//! use entelix_core::ir::{AnthropicExt, ProviderExtensions};
//!
//! let store = FileCredentialStore::with_path(
//!     FileCredentialStore::default_claude_path()?,
//! );
//! let provider = ClaudeCodeOAuthProvider::new(store);
//! // Apply the matching beta capability on every outgoing request:
//! let extensions = ProviderExtensions::default()
//!     .with_anthropic(AnthropicExt::default().with_betas([CLAUDE_CODE_BETA]));
//! ```
//!
//! ## Store hygiene
//!
//! [`FileCredentialStore`] reads and writes the credential file on a
//! `tokio::task::spawn_blocking` worker so the async runtime never
//! stalls on disk IO. Operators that need an in-memory backend
//! (env-var-driven, vault) implement [`CredentialStore`] directly
//! without touching the file path.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-auth-claude-code/0.3.0")]
#![deny(missing_docs)]
// Doc-prose lints fire on legitimate proper nouns / acronyms (Claude
// Code, OAuth2, RFC 6749, …); the redundant_pub_crate lint disagrees
// with the workspace `unreachable_pub` rule for items inside private
// modules — same exemption pattern as `entelix-tools`.
#![allow(clippy::doc_markdown, clippy::redundant_pub_crate)]

mod config;
mod credential;
mod error;
mod provider;
mod refresh;
mod store;

pub use config::{
    CLAUDE_CODE_BETA, ClaudeCodeOAuthConfig, DEFAULT_REFRESH_TIMEOUT, DEFAULT_TOKEN_URL,
};
pub use credential::{CredentialFile, OAuthCredential};
pub use error::{ClaudeCodeAuthError, ClaudeCodeAuthResult};
pub use provider::ClaudeCodeOAuthProvider;
pub use store::{CredentialStore, FileCredentialStore};
