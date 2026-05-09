//! Error variants for the Claude Code OAuth provider.

use thiserror::Error;

/// Failure modes specific to Claude Code OAuth credential resolution.
///
/// Each variant maps onto a [`entelix_core::auth::AuthError`] when
/// surfaced through the [`entelix_core::auth::CredentialProvider`]
/// trait.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClaudeCodeAuthError {
    /// The credential file is absent — typically the operator hasn't
    /// run `claude login` yet.
    #[error("Claude Code credentials not found at `{path}` — run `claude login` first")]
    CredentialsMissing {
        /// Resolved storage path that was checked.
        path: String,
    },
    /// The credential file exists but contains no OAuth section.
    #[error("Claude Code credential file `{path}` has no OAuth section")]
    OAuthSectionMissing {
        /// Resolved storage path of the credential file.
        path: String,
    },
    /// The OAuth credential payload had no `refresh_token` — the
    /// access token is expired and cannot be renewed.
    #[error("Claude Code OAuth refresh token absent — re-authenticate via `claude login`")]
    RefreshTokenMissing,
    /// The token-refresh HTTP call failed.
    #[error("Claude Code OAuth refresh failed: {message}")]
    RefreshHttp {
        /// Human-readable cause forwarded from `reqwest` or the OAuth
        /// server's error body.
        message: String,
    },
    /// The credential file failed to parse as JSON.
    #[error("Claude Code credential file `{path}` is not valid JSON: {source}")]
    InvalidStorage {
        /// Resolved storage path of the credential file.
        path: String,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
    /// Filesystem error reading or writing the credential file.
    #[error("Claude Code credential file `{path}` IO failed: {source}")]
    Io {
        /// Resolved storage path of the credential file.
        path: String,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// `HOME` (or platform equivalent) is unset, so the default
    /// `~/.claude/.credentials.json` location cannot be resolved.
    #[error("home directory not resolvable — supply the credentials path explicitly")]
    HomeUnresolved,
}

/// Convenience result alias.
pub type ClaudeCodeAuthResult<T> = Result<T, ClaudeCodeAuthError>;

impl From<ClaudeCodeAuthError> for entelix_core::Error {
    fn from(err: ClaudeCodeAuthError) -> Self {
        use entelix_core::auth::AuthError;
        let message = err.to_string();
        match err {
            ClaudeCodeAuthError::CredentialsMissing { .. }
            | ClaudeCodeAuthError::OAuthSectionMissing { .. }
            | ClaudeCodeAuthError::HomeUnresolved => Self::Auth(AuthError::missing_from(message)),
            ClaudeCodeAuthError::RefreshTokenMissing => {
                Self::Auth(AuthError::expired_with(message))
            }
            ClaudeCodeAuthError::RefreshHttp { .. } => {
                Self::Auth(AuthError::source_unreachable(message))
            }
            ClaudeCodeAuthError::InvalidStorage { .. } | ClaudeCodeAuthError::Io { .. } => {
                Self::Auth(AuthError::refused(message))
            }
        }
    }
}
