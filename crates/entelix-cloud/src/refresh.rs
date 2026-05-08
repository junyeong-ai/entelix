//! `CachedTokenProvider<T>` — single-flight cached token wrapper used by
//! every cloud transport that fronts an OAuth-style credential
//! source.
//!
//! Read path is lock-free (`parking_lot::RwLock` read guard cloned
//! into an `Option<TokenState<T>>`). Single-flight uses an atomic
//! claim flag plus `tokio::sync::Notify`: at most one task runs the
//! user-supplied `TokenRefresher::refresh()` future at a time, and
//! no lock is held across that future (CLAUDE.md "lock ordering").
//! Refresh fires `refresh_buffer` before the cached token's
//! wall-clock expiry.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::sync::Notify;

use crate::CloudError;

/// Default lead time before a token's expiry triggers a refresh.
///
/// Five minutes balances "preempt 401s on long calls" against "don't
/// thrash short-TTL credentials" — most cloud providers issue
/// 60-min tokens, so a 5-min buffer leaves ~92% of the lifetime in
/// the fast path.
pub const DEFAULT_REFRESH_BUFFER: Duration = Duration::from_mins(5);

/// Source-of-truth that yields a fresh `T` plus its absolute expiry
/// time when called.
#[async_trait]
pub trait TokenRefresher<T>: Send + Sync
where
    T: Clone + Send + Sync + 'static,
{
    /// Fetch a fresh token. Implementors hit the underlying IDP
    /// (gcp_auth, azure_identity, …); errors propagate to the
    /// caller of [`CachedTokenProvider::current`].
    async fn refresh(&self) -> Result<TokenSnapshot<T>, CloudError>;
}

/// One refresh result.
#[derive(Clone, Debug)]
pub struct TokenSnapshot<T> {
    /// The token value itself (often a [`secrecy::SecretString`]).
    pub value: T,
    /// Wall-clock instant at which the token stops being valid.
    pub expires_at: Instant,
}

#[derive(Clone)]
struct TokenState<T> {
    value: T,
    expires_at: Instant,
}

/// Cache + single-flight wrapper around a [`TokenRefresher`].
pub struct CachedTokenProvider<T>
where
    T: Clone + Send + Sync + 'static,
{
    cached: RwLock<Option<TokenState<T>>>,
    refresh_in_progress: AtomicBool,
    refresh_done: Notify,
    refresher: Arc<dyn TokenRefresher<T>>,
    refresh_buffer: Duration,
}

impl<T> CachedTokenProvider<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Build with the default [`DEFAULT_REFRESH_BUFFER`] lead time.
    pub fn new(refresher: Arc<dyn TokenRefresher<T>>) -> Self {
        Self::with_refresh_buffer(refresher, DEFAULT_REFRESH_BUFFER)
    }

    /// Build with a custom refresh-buffer.
    pub fn with_refresh_buffer(
        refresher: Arc<dyn TokenRefresher<T>>,
        refresh_buffer: Duration,
    ) -> Self {
        Self {
            cached: RwLock::new(None),
            refresh_in_progress: AtomicBool::new(false),
            refresh_done: Notify::new(),
            refresher,
            refresh_buffer,
        }
    }

    /// Return the current valid token, refreshing if the cached
    /// value is missing or within the refresh-buffer of expiry.
    ///
    /// Single-flight: only one caller runs the user-supplied
    /// [`TokenRefresher::refresh`] at a time. Other callers wait on
    /// a [`Notify`] for the refresh to complete and then re-check
    /// the cache. No lock is held across the user-supplied future.
    pub async fn current(&self) -> Result<T, CloudError> {
        loop {
            if let Some(state) = self.read_fresh() {
                return Ok(state);
            }
            // Try to claim refresh ownership atomically.
            if self
                .refresh_in_progress
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let result = self.refresher.refresh().await;
                if let Ok(snap) = &result {
                    *self.cached.write() = Some(TokenState {
                        value: snap.value.clone(),
                        expires_at: snap.expires_at,
                    });
                }
                self.refresh_in_progress.store(false, Ordering::Release);
                self.refresh_done.notify_waiters();
                return result.map(|s| s.value);
            }
            // Someone else owns the refresh — register for the
            // notification, then re-check (the writer may complete
            // between our claim attempt and our subscription).
            let waiter = self.refresh_done.notified();
            tokio::pin!(waiter);
            waiter.as_mut().enable();
            if let Some(state) = self.read_fresh() {
                return Ok(state);
            }
            waiter.await;
            // Loop: re-read cache (now likely fresh) or retry the claim.
        }
    }

    /// Drop the cached value. Useful when a 401 surfaces to force a
    /// reload regardless of the recorded expiry (clock skew defence).
    pub fn invalidate(&self) {
        *self.cached.write() = None;
    }

    fn read_fresh(&self) -> Option<T> {
        let snapshot = {
            let guard = self.cached.read();
            guard.as_ref().cloned()
        };
        let state = snapshot?;
        if Instant::now() + self.refresh_buffer < state.expires_at {
            Some(state.value)
        } else {
            None
        }
    }
}
