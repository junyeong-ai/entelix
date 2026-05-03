//! `RedisLock` — [`DistributedLock`] over `SET NX PX` plus a Lua
//! release / extend script that verifies the per-acquire token.
//!
//! Single-node Redis only — Redlock is rejected (Kleppmann's
//! analysis: clock skew + GC pauses make multi-master Redlock
//! unsafe for our use-case). Operators that need cross-region
//! durability use the Postgres backend instead.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use redis::Script;
use redis::aio::ConnectionManager;
use tokio::time::sleep;

use crate::advisory_key::AdvisoryKey;
use crate::error::{PersistenceError, PersistenceResult};
use crate::lock::{DistributedLock, LockGuard};

const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Lua release script: deletes the key only when the stored value
/// matches the supplied token. Returns 1 on release, 0 if the lock
/// was already released or expired.
const RELEASE_SCRIPT: &str = r#"
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"#;

/// Lua extend script: bumps `pexpire` only when the token matches.
const EXTEND_SCRIPT: &str = r#"
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("pexpire", KEYS[1], ARGV[2])
else
    return 0
end
"#;

/// Redis-backed distributed lock.
pub struct RedisLock {
    manager: Arc<ConnectionManager>,
    release_script: Script,
    extend_script: Script,
}

impl RedisLock {
    pub(crate) fn new(manager: Arc<ConnectionManager>) -> Self {
        Self {
            manager,
            release_script: Script::new(RELEASE_SCRIPT),
            extend_script: Script::new(EXTEND_SCRIPT),
        }
    }
}

#[async_trait]
impl DistributedLock for RedisLock {
    async fn try_acquire(
        &self,
        key: &AdvisoryKey,
        ttl: Duration,
    ) -> PersistenceResult<Option<LockGuard>> {
        let guard = LockGuard::new(*key);
        let mut conn = (*self.manager).clone();
        let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
        let result: Option<String> = redis::cmd("SET")
            .arg(key.redis_key())
            .arg(guard.token())
            .arg("NX")
            .arg("PX")
            .arg(ttl_ms)
            .query_async(&mut conn)
            .await
            .map_err(backend_err)?;
        match result.as_deref() {
            Some("OK") => Ok(Some(guard)),
            _ => {
                drop(guard); // explicitly release — Drop will mark warning,
                // suppress with `mark_released` since acquire failed.
                Ok(None)
            }
        }
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

    async fn extend(&self, guard: &LockGuard, ttl: Duration) -> PersistenceResult<bool> {
        let mut conn = (*self.manager).clone();
        let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
        let result: i32 = self
            .extend_script
            .key(guard.key().redis_key())
            .arg(guard.token())
            .arg(ttl_ms)
            .invoke_async(&mut conn)
            .await
            .map_err(backend_err)?;
        Ok(result == 1)
    }

    async fn release(&self, mut guard: LockGuard) -> PersistenceResult<()> {
        let mut conn = (*self.manager).clone();
        let _: i32 = self
            .release_script
            .key(guard.key().redis_key())
            .arg(guard.token())
            .invoke_async(&mut conn)
            .await
            .map_err(backend_err)?;
        guard.mark_released();
        Ok(())
    }
}

fn backend_err(e: redis::RedisError) -> PersistenceError {
    PersistenceError::Backend(e.to_string())
}
