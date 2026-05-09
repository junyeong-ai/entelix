//! `DistributedLock` semantics + `with_session_lock` composition,
//! verified against a deterministic in-memory mock that tracks
//! ownership tokens.

#![allow(
    clippy::unwrap_used,
    clippy::significant_drop_tightening,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else
)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use entelix_persistence::lock::{DEFAULT_SESSION_LOCK_DEADLINE, DEFAULT_SESSION_LOCK_TTL};
use entelix_persistence::{
    AdvisoryKey, DistributedLock, LockGuard, PersistenceError, PersistenceResult, with_session_lock,
};
use parking_lot::Mutex;

#[derive(Default)]
struct InMemoryLock {
    held: Arc<Mutex<HashMap<String, String>>>,
}

#[async_trait]
impl DistributedLock for InMemoryLock {
    async fn try_acquire(
        &self,
        key: &AdvisoryKey,
        _ttl: Duration,
    ) -> PersistenceResult<Option<LockGuard>> {
        let guard = LockGuard::new(*key);
        let mut held = self.held.lock();
        let key_str = key.to_string();
        if held.contains_key(&key_str) {
            drop(guard);
            return Ok(None);
        }
        held.insert(key_str, guard.token().to_owned());
        Ok(Some(guard))
    }

    async fn acquire(
        &self,
        key: &AdvisoryKey,
        ttl: Duration,
        deadline: Duration,
    ) -> PersistenceResult<LockGuard> {
        // 단순화: try once → fail with timeout. 테스트에서는 충돌 case만 검증.
        let _ = deadline;
        match self.try_acquire(key, ttl).await? {
            Some(g) => Ok(g),
            None => Err(PersistenceError::LockAcquireTimeout {
                key: key.to_string(),
                attempts: 1,
            }),
        }
    }

    async fn extend(&self, _guard: &LockGuard, _ttl: Duration) -> PersistenceResult<bool> {
        Ok(true)
    }

    async fn release(&self, mut guard: LockGuard) -> PersistenceResult<()> {
        let mut held = self.held.lock();
        let key_str = guard.key().to_string();
        if let Some(token) = held.get(&key_str)
            && token == guard.token()
        {
            held.remove(&key_str);
        }
        guard.mark_released();
        Ok(())
    }
}

#[tokio::test]
async fn with_session_lock_acquires_and_releases() {
    let lock = InMemoryLock::default();
    let count = Arc::new(Mutex::new(0u32));
    let count_inner = count.clone();
    let tenant = entelix_core::TenantId::new("tenant-x");
    let result: PersistenceResult<u32> =
        with_session_lock(&lock, &tenant, "thread-7", None, None, || async move {
            *count_inner.lock() += 1;
            Ok::<u32, PersistenceError>(42)
        })
        .await;
    assert_eq!(result.unwrap(), 42);
    assert_eq!(*count.lock(), 1);
    // Lock is released — another acquire on the same key succeeds.
    let key = AdvisoryKey::for_session(&tenant, "thread-7");
    assert!(
        lock.try_acquire(&key, DEFAULT_SESSION_LOCK_TTL)
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn with_session_lock_releases_on_inner_error() {
    let lock = InMemoryLock::default();
    let tenant = entelix_core::TenantId::new("tenant-x");
    let result: PersistenceResult<u32> =
        with_session_lock(&lock, &tenant, "thread-7", None, None, || async move {
            Err::<u32, PersistenceError>(PersistenceError::Backend("inner failure".into()))
        })
        .await;
    assert!(result.is_err());
    // Lock released even though inner closure failed.
    let key = AdvisoryKey::for_session(&tenant, "thread-7");
    assert!(
        lock.try_acquire(&key, DEFAULT_SESSION_LOCK_TTL)
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn try_acquire_returns_none_when_held() {
    let lock = InMemoryLock::default();
    let tenant = entelix_core::TenantId::new("tenant-x");
    let key = AdvisoryKey::for_session(&tenant, "thread-7");
    let _g1 = lock
        .try_acquire(&key, DEFAULT_SESSION_LOCK_TTL)
        .await
        .unwrap()
        .unwrap();
    let g2 = lock
        .try_acquire(&key, DEFAULT_SESSION_LOCK_TTL)
        .await
        .unwrap();
    assert!(g2.is_none());
}

#[tokio::test]
async fn release_idempotent_for_already_released() {
    let lock = InMemoryLock::default();
    let tenant = entelix_core::TenantId::new("tenant-x");
    let key = AdvisoryKey::for_session(&tenant, "thread-7");
    let g1 = lock
        .try_acquire(&key, DEFAULT_SESSION_LOCK_TTL)
        .await
        .unwrap()
        .unwrap();
    lock.release(g1).await.unwrap();

    // Second concurrent acquire should now succeed.
    let g2 = lock
        .try_acquire(&key, DEFAULT_SESSION_LOCK_TTL)
        .await
        .unwrap()
        .unwrap();
    lock.release(g2).await.unwrap();
}

#[tokio::test]
async fn lock_acquire_timeout_surfaces_attempt_count() {
    let lock = InMemoryLock::default();
    let tenant = entelix_core::TenantId::new("tenant-x");
    let key = AdvisoryKey::for_session(&tenant, "thread-7");
    let _g = lock
        .try_acquire(&key, DEFAULT_SESSION_LOCK_TTL)
        .await
        .unwrap()
        .unwrap();
    let err = lock
        .acquire(
            &key,
            DEFAULT_SESSION_LOCK_TTL,
            DEFAULT_SESSION_LOCK_DEADLINE,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        PersistenceError::LockAcquireTimeout { attempts: 1, .. }
    ));
}
