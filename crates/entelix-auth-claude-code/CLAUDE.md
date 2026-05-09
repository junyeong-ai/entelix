# entelix-auth-claude-code

Claude Code OAuth credential provider — reuses the access token the
upstream `claude` CLI manages so operators with a Claude.ai
subscription drive entelix without minting a separate Anthropic API
key.

## Surface

- **`ClaudeCodeOAuthProvider`** — `entelix_core::auth::CredentialProvider` impl. Loads the OAuth access token from a `CredentialStore` backend, refreshes via the standard OAuth2 `refresh_token` grant when expiry is imminent, and hands the token to the transport as `Authorization: Bearer <token>`.
- **`CredentialStore` trait + `FileCredentialStore`** — backend abstraction. `FileCredentialStore` is the default (reads / writes the JSON file the `claude` CLI shares); operators that ship credentials through a vault, env-var-backed in-memory store, or platform secret API (macOS Keychain, Linux Secret Service, Windows Credential Manager) implement `CredentialStore` directly.
- **`ClaudeCodeOAuthConfig`** — token endpoint URL / OAuth client id / refresh-call timeout.
- **`OAuthCredential` + `CredentialFile`** — on-disk envelope shape; serde field names mirror the `claude` CLI exactly so storage round-trips with the upstream tool.
- **`CLAUDE_CODE_BETA`** const — the `anthropic-beta` capability gate Claude Code OAuth tokens require. Operators wire it through `entelix_core::ir::AnthropicExt::with_betas([CLAUDE_CODE_BETA])` so the codec emits the matching header. Credential and codec ext stay independent (single responsibility).

## Crate-local rules

- **Single fs-touching crate in the workspace.** `FileCredentialStore` is the *only* path that imports `std::fs` outside the `entelix-tools/sandboxed` boundary. The exception is allowlisted in `cargo xtask no-fs`'s `CREDENTIAL_STORAGE_EXEMPTIONS`. Any new fs site here needs a CLAUDE.md amendment.
- **fs IO runs on `tokio::task::spawn_blocking`.** Async runtimes must never stall on disk syscalls; `FileCredentialStore::load` / `save` already wrap their `std::fs::read` / `std::fs::write` accordingly.
- **`FileCredentialStore::save` is atomic via write-then-rename.** Bytes stream into a sibling `<file>.tmp`, then `rename(2)` replaces the destination atomically (POSIX guarantee). The `claude` CLI shares this file as its single source of truth — a partial write here would invalidate every subsequent session, so the temp-file pattern is non-negotiable. Windows operators see the rename failure surface explicitly when the destination exists (no silent corruption fallback).
- **Refresh is single-flight.** The provider holds an internal `Mutex<()>` so concurrent refresh attempts serialise — every refresh may rotate the refresh token server-side, and two parallel attempts race into rejection.
- **Never log the access token.** Tokens stay in `SecretString` from the moment they leave the file until the transport writes the header. `Debug` impls explicitly omit them.
- **Beta header lives on the codec ext.** Operators must add `CLAUDE_CODE_BETA` through `AnthropicExt::with_betas` — the provider does not silently inject it. Single responsibility + no hidden coupling between the credential surface and the wire surface.

## Forbidden

- Caching the access token outside the `CredentialStore` backend (e.g. a static `Mutex<Option<OAuthCredential>>`) — every refresh must hit storage so concurrent processes share rotation.
- Reading credentials from environment variables for "convenience" — that path belongs in `entelix_core::auth::ApiKeyProvider`. This crate is purely the Claude Code OAuth path.
- Importing `tokio::process` / `std::process` to call the `claude` CLI — credential discovery reads the file directly, never shells out.

## References

- Claude.ai OAuth token endpoint: `https://console.anthropic.com/v1/oauth/token` (per-environment override via `ClaudeCodeOAuthConfig::with_token_url`).
- RFC 6749 §6 — OAuth 2.0 refresh-token grant.
- `~/.claude/.credentials.json` — the on-disk shape the `claude` CLI writes; resolved through `FileCredentialStore::default_claude_path` against `HOME` / `USERPROFILE`.
