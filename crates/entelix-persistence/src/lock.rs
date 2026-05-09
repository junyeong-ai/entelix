//! Distributed lock primitives.
//!
//! [`DistributedLock`] is the backend-agnostic trait; concrete impls
//! live in the `postgres` (advisory locks) and `redis` (`SET NX PX` +
//! Lua release script) modules — each gated behind its corresponding
//! feature flag. [`with_session_lock`] composes the trait with
//! [`crate::AdvisoryKey::for_session`] so the canonical
//! `(tenant_id, thread_id)` lock-key derivation is the only thing
//! callers need to know about.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use uuid::Uuid;

use crate::advisory_key::AdvisoryKey;
use crate::error::{PersistenceError, PersistenceResult};

/// Acquire-then-release primitive over a distributed key.
///
/// Implementors are responsible for:
/// - exclusive acquisition by `key`
/// - per-acquire token issuance so [`Self::release`] is idempotent
///   even if the same key is held by a later attempt
/// - TTL enforcement so a crashed holder doesn't deadlock
///   indefinitely
///
/// Cancellation: implementors honour the ambient cancellation token
/// propagated via [`entelix_core::ExecutionContext`] when one is in
/// scope. The trait itself does not take a context —
/// [`with_session_lock`] is the composition point.
#[async_trait]
pub trait DistributedLock: Send + Sync + 'static {
    /// Try once to acquire `key` with the given `ttl`. Returns
    /// `Ok(Some(guard))` on success, `Ok(None)` when the key is
    /// currently held by another holder, and `Err(_)` for backend
    /// failure.
    async fn try_acquire(
        &self,
        key: &AdvisoryKey,
        ttl: Duration,
    ) -> PersistenceResult<Option<LockGuard>>;

    /// Block until the lock is acquired or `deadline` elapses.
    /// Implementors poll with backoff between attempts.
    async fn acquire(
        &self,
        key: &AdvisoryKey,
        ttl: Duration,
        deadline: Duration,
    ) -> PersistenceResult<LockGuard>;

    /// Extend the holder's TTL. Returns `Ok(false)` when the lock has
    /// already been released or expired (the guard's token no longer
    /// matches the stored value).
    async fn extend(&self, guard: &LockGuard, ttl: Duration) -> PersistenceResult<bool>;

    /// Release the lock. Consumes the guard. The implementation is a
    /// no-op when the token already mismatches (lock expired by TTL
    /// before the caller got here).
    async fn release(&self, guard: LockGuard) -> PersistenceResult<()>;
}

/// Owned proof that the holder currently has exclusive access to a
/// key.
///
/// Drop semantics: when a `LockGuard` is dropped without an explicit
/// [`DistributedLock::release`] call, a `tracing::warn!` records the
/// leak. The lock will still expire by TTL, so correctness is
/// preserved, but the warning surfaces forgotten release calls in
/// telemetry.
#[derive(Debug)]
pub struct LockGuard {
    key: AdvisoryKey,
    token: String,
    acquired_at: Instant,
    released: bool,
}

impl LockGuard {
    /// Construct a guard. Backend [`DistributedLock`] impls call this
    /// after a successful acquire — the `token` is the per-acquire
    /// ownership marker the backend stores alongside the lock value.
    pub fn new(key: AdvisoryKey) -> Self {
        Self {
            key,
            token: Uuid::new_v4().to_string(),
            acquired_at: Instant::now(),
            released: false,
        }
    }

    /// Borrow the lock key.
    pub const fn key(&self) -> &AdvisoryKey {
        &self.key
    }

    /// Borrow the per-acquire ownership token.
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Wall-clock duration the guard has been held.
    pub fn held_for(&self) -> Duration {
        self.acquired_at.elapsed()
    }

    /// Mark the guard as released — backend [`DistributedLock`] impls
    /// call this from `release()` so the [`Drop`] impl does not warn.
    /// Outside backend code there is no reason to call this directly.
    pub fn mark_released(&mut self) {
        self.released = true;
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if !self.released {
            tracing::warn!(
                target: "entelix.persistence::lock",
                key = %self.key,
                held_ms = self.acquired_at.elapsed().as_millis() as u64,
                "LockGuard dropped without explicit release; lock will expire by TTL only"
            );
        }
    }
}

/// Default TTL for session locks — 30 seconds. Long enough to bridge
/// a typical model call, short enough that a crashed holder doesn't
/// stall the next request for too long.
pub const DEFAULT_SESSION_LOCK_TTL: Duration = Duration::from_secs(30);

/// Default total deadline a caller will wait for the lock — 5 seconds.
pub const DEFAULT_SESSION_LOCK_DEADLINE: Duration = Duration::from_secs(5);

/// Acquire a session lock keyed by `(tenant, thread)`, run the
/// caller's closure, then release the lock. The lock is released even
/// if `f` returns an error.
///
/// `lock` is any backend that implements [`DistributedLock`] —
/// typically a `postgres::PostgresPersistence` or
/// `redis::RedisPersistence` handle. Pass `None` for `ttl` /
/// `deadline` to use the defaults.
///
/// The closure does not receive the guard — `with_session_lock`
/// owns the lifecycle. Callers that need to extend the lock during a
/// long-running operation use [`DistributedLock::acquire`] /
/// [`DistributedLock::extend`] / [`DistributedLock::release`]
/// directly.
pub async fn with_session_lock<L, F, Fut, T, E>(
    lock: &L,
    tenant_id: &entelix_core::TenantId,
    thread_id: &str,
    ttl: Option<Duration>,
    deadline: Option<Duration>,
    f: F,
) -> PersistenceResult<T>
where
    L: DistributedLock + ?Sized,
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
    E: Into<PersistenceError>,
{
    let key = AdvisoryKey::for_session(tenant_id, thread_id);
    let ttl = ttl.unwrap_or(DEFAULT_SESSION_LOCK_TTL);
    let deadline = deadline.unwrap_or(DEFAULT_SESSION_LOCK_DEADLINE);

    let guard = lock.acquire(&key, ttl, deadline).await?;
    let outcome = f().await;
    // Best-effort release — TTL is the safety net.
    if let Err(e) = lock.release(guard).await {
        tracing::warn!(
            target: "entelix.persistence::lock",
            error = %e,
            "lock release failed; relying on TTL expiry"
        );
    }
    outcome.map_err(Into::into)
}
