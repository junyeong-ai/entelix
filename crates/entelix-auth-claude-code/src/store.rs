//! `CredentialStore` trait + the default `FileCredentialStore` impl.
//!
//! The trait keeps backend choice on the operator side. The default
//! reads / writes the JSON file the `claude` CLI uses, but operators
//! that ship credentials through a vault, an env-var-backed
//! in-memory store, or a platform secret API (macOS Keychain, Linux
//! Secret Service, Windows Credential Manager) implement
//! [`CredentialStore`] directly.
//!
//! The store is async — every backend may incur IO or a syscall.

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::credential::CredentialFile;
use crate::error::{ClaudeCodeAuthError, ClaudeCodeAuthResult};

/// Async credential persistence backend.
#[async_trait]
pub trait CredentialStore: Send + Sync + 'static {
    /// Load the current credential envelope, or `None` if the
    /// backend holds no credential yet.
    async fn load(&self) -> ClaudeCodeAuthResult<Option<CredentialFile>>;
    /// Persist a refreshed credential envelope. Backends that
    /// cannot retain state (e.g. a read-only env var) return
    /// [`ClaudeCodeAuthError::Io`] with a descriptive message —
    /// the provider surfaces that to the operator.
    async fn save(&self, file: &CredentialFile) -> ClaudeCodeAuthResult<()>;
}

/// File-backed [`CredentialStore`] at a caller-supplied path.
///
/// The path mirrors the on-disk shape the `claude` CLI writes
/// (`~/.claude/.credentials.json` by convention); use
/// [`FileCredentialStore::default_claude_path`] to resolve that
/// location against the host's home directory.
#[derive(Debug, Clone)]
pub struct FileCredentialStore {
    path: PathBuf,
}

impl FileCredentialStore {
    /// Build a store rooted at the given file path.
    #[must_use]
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The configured store path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Resolve `~/.claude/.credentials.json` against the host's
    /// home directory (`HOME` on Unix, `USERPROFILE` on Windows).
    /// Returns [`ClaudeCodeAuthError::HomeUnresolved`] when neither
    /// is set.
    pub fn default_claude_path() -> ClaudeCodeAuthResult<PathBuf> {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .ok_or(ClaudeCodeAuthError::HomeUnresolved)?;
        let mut path = PathBuf::from(home);
        path.push(".claude");
        path.push(".credentials.json");
        Ok(path)
    }
}

#[async_trait]
impl CredentialStore for FileCredentialStore {
    async fn load(&self) -> ClaudeCodeAuthResult<Option<CredentialFile>> {
        let path = self.path.clone();
        let read = tokio::task::spawn_blocking(move || std::fs::read(&path))
            .await
            .map_err(|join_err| ClaudeCodeAuthError::Io {
                path: self.path.display().to_string(),
                source: std::io::Error::other(join_err.to_string()),
            })?;
        let bytes = match read {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(ClaudeCodeAuthError::Io {
                    path: self.path.display().to_string(),
                    source,
                });
            }
        };
        let file: CredentialFile = serde_json::from_slice(&bytes).map_err(|source| {
            ClaudeCodeAuthError::InvalidStorage {
                path: self.path.display().to_string(),
                source,
            }
        })?;
        Ok(Some(file))
    }

    async fn save(&self, file: &CredentialFile) -> ClaudeCodeAuthResult<()> {
        let path = self.path.clone();
        let display = self.path.display().to_string();
        let bytes = serde_json::to_vec_pretty(file).map_err(|source| {
            ClaudeCodeAuthError::InvalidStorage {
                path: display.clone(),
                source,
            }
        })?;
        let display_for_blocking = display.clone();
        let write = tokio::task::spawn_blocking(move || atomic_write(&path, &bytes))
            .await
            .map_err(|join_err| ClaudeCodeAuthError::Io {
                path: display_for_blocking.clone(),
                source: std::io::Error::other(join_err.to_string()),
            })?;
        write.map_err(|source| ClaudeCodeAuthError::Io {
            path: display,
            source,
        })
    }
}

/// Write `bytes` to `path` atomically: create the parent directory,
/// stream into a sibling `<file>.tmp`, then `rename` over the
/// destination.
///
/// Atomicity: `rename(2)` is atomic on POSIX — readers either see
/// the prior file or the complete new one, never a partial write.
/// On Windows, `MoveFileExW`'s default flags reject overwriting an
/// existing destination; the error propagates so the operator sees
/// the failure rather than silent corruption (a Windows-specific
/// `MOVEFILE_REPLACE_EXISTING` flag would need a `winapi`
/// dependency we deliberately avoid in this minimal companion).
fn atomic_write(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("destination path has no file name"))?
        .to_owned();
    tmp_name.push(".tmp");
    let tmp_path = path.with_file_name(tmp_name);
    // If a stale `.tmp` lingers from a prior crashed write, ignore
    // the absence-error and proceed — the new write overwrites it.
    let _ = std::fs::remove_file(&tmp_path);
    std::fs::write(&tmp_path, bytes)?;
    std::fs::rename(&tmp_path, path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::credential::OAuthCredential;
    use chrono::Utc;

    fn tmp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "entelix-claude-code-{}-{}.json",
            std::process::id(),
            name
        ));
        path
    }

    #[tokio::test]
    async fn load_returns_none_when_file_absent() {
        let store = FileCredentialStore::with_path(tmp_path("absent"));
        let loaded = store.load().await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn save_overwrites_existing_file_via_rename() {
        // Atomic write must replace any prior credential file —
        // the upstream `claude` CLI shares this path, and a stale
        // refresh-token there would invalidate every subsequent
        // session.
        let path = tmp_path("overwrite");
        let _ = std::fs::remove_file(&path);
        let store = FileCredentialStore::with_path(&path);
        let first = CredentialFile::with_oauth(OAuthCredential::new(
            "first",
            (Utc::now() + chrono::Duration::hours(1)).timestamp_millis(),
        ));
        store.save(&first).await.unwrap();
        let second = CredentialFile::with_oauth(OAuthCredential::new(
            "second",
            (Utc::now() + chrono::Duration::hours(2)).timestamp_millis(),
        ));
        store.save(&second).await.unwrap();
        let loaded = store.load().await.unwrap().unwrap();
        assert_eq!(
            loaded.claude_ai_oauth.unwrap().access_token,
            "second",
            "second save must replace first"
        );
        // No stale `.tmp` sibling should linger after a successful
        // rename — verifies the cleanup contract.
        let mut tmp_sibling = path.clone();
        let mut tmp_name = path.file_name().unwrap().to_owned();
        tmp_name.push(".tmp");
        tmp_sibling.set_file_name(tmp_name);
        assert!(
            !tmp_sibling.exists(),
            "rename must consume the .tmp staging file"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let path = tmp_path("round_trip");
        let _ = std::fs::remove_file(&path);
        let store = FileCredentialStore::with_path(&path);
        let envelope = CredentialFile::with_oauth(
            OAuthCredential::new(
                "tok",
                (Utc::now() + chrono::Duration::hours(1)).timestamp_millis(),
            )
            .with_refresh_token("ref")
            .with_subscription_type("pro")
            .with_scopes(["user:inference"]),
        );
        store.save(&envelope).await.unwrap();
        let loaded = store.load().await.unwrap().unwrap();
        let oauth = loaded.claude_ai_oauth.unwrap();
        assert_eq!(oauth.access_token, "tok");
        assert_eq!(oauth.subscription_type.as_deref(), Some("pro"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn default_path_resolves_from_environment() {
        // Whatever HOME / USERPROFILE the test environment exposes,
        // the helper either succeeds (typical case) or returns
        // `HomeUnresolved` (CI sandboxes). Both paths are valid
        // contract outcomes; the test asserts the success branch
        // produces the documented suffix.
        if let Ok(path) = FileCredentialStore::default_claude_path() {
            assert!(
                path.ends_with(".claude/.credentials.json")
                    || path.ends_with(r".claude\.credentials.json"),
                "unexpected path shape: {}",
                path.display()
            );
        }
    }
}
