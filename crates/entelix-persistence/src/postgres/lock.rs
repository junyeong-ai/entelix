//! `PostgresLock` — [`DistributedLock`] over `pg_advisory_lock`.
//!
//! Holds a session-scoped Postgres advisory lock. Each acquisition
//! pins a `PoolConnection` to the [`LockGuard`] (via an internal
//! lookup map keyed by token) so the lock survives until the holder
//! calls `release` — at which point the connection returns to the
//! pool. TTL is advisory in the Postgres backend (the lock doesn't
//! auto-expire); callers that need true expiry use the Redis backend.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dashmap::DashMap;
use sqlx::pool::PoolConnection;
use sqlx::postgres::{PgPool, Postgres};
use tokio::time::sleep;

use crate::advisory_key::AdvisoryKey;
use crate::error::{PersistenceError, PersistenceResult};
use crate::lock::{DistributedLock, LockGuard};

const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Postgres-backed distributed lock.
pub struct PostgresLock {
    pool: Arc<PgPool>,
    held: Arc<DashMap<String, PoolConnection<Postgres>>>,
}

impl PostgresLock {
    pub(crate) fn new(pool: Arc<PgPool>) -> Self {
        Self {
            pool,
            held: Arc::new(DashMap::new()),
        }
    }
}

#[async_trait]
impl DistributedLock for PostgresLock {
    async fn try_acquire(
        &self,
        key: &AdvisoryKey,
        _ttl: Duration,
    ) -> PersistenceResult<Option<LockGuard>> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| PersistenceError::Backend(format!("pool acquire: {e}")))?;

        let (high, low) = key.halves();
        let acquired: (bool,) = sqlx::query_as("SELECT pg_try_advisory_lock($1, $2)")
            .bind(high)
            .bind(low)
            .fetch_one(&mut *conn)
            .await
            .map_err(backend_err)?;

        if !acquired.0 {
            // Connection drops back to the pool implicitly.
            return Ok(None);
        }
        let guard = LockGuard::new(*key);
        self.held.insert(guard.token().to_owned(), conn);
        Ok(Some(guard))
    }

    async fn acquire(
        &self,
        key: &AdvisoryKey,
        ttl: Duration,
        deadline: Duration,
    ) -> PersistenceResult<LockGuard> {
        let start = Instant::now();
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            if let Some(guard) = self.try_acquire(key, ttl).await? {
                return Ok(guard);
            }
            if start.elapsed() >= deadline {
                return Err(PersistenceError::LockAcquireTimeout {
                    key: key.to_string(),
                    attempts,
                });
            }
            sleep(POLL_INTERVAL).await;
        }
    }

    async fn extend(&self, _guard: &LockGuard, _ttl: Duration) -> PersistenceResult<bool> {
        // Postgres advisory locks don't expire — extend is a no-op
        // that succeeds when the lock is still tracked.
        Ok(true)
    }

    async fn release(&self, mut guard: LockGuard) -> PersistenceResult<()> {
        let Some((_, mut conn)) = self.held.remove(guard.token()) else {
            // Already released or never tracked. Mark released so
            // Drop doesn't warn.
            guard.mark_released();
            return Ok(());
        };
        let (high, low) = guard.key().halves();
        let _: (bool,) = sqlx::query_as("SELECT pg_advisory_unlock($1, $2)")
            .bind(high)
            .bind(low)
            .fetch_one(&mut *conn)
            .await
            .map_err(backend_err)?;
        guard.mark_released();
        // Connection drops back to the pool implicitly.
        Ok(())
    }
}

fn backend_err(e: sqlx::Error) -> PersistenceError {
    PersistenceError::Backend(e.to_string())
}
