//! `CachedTokenProvider<T>` single-flight + cache freshness tests.
//! Uses a deterministic mock refresher that counts calls and
//! produces tokens with controllable expiry.

#![allow(clippy::unwrap_used, clippy::missing_const_for_fn)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use entelix_cloud::CloudError;
use entelix_cloud::refresh::{CachedTokenProvider, TokenRefresher, TokenSnapshot};

#[derive(Default)]
struct CountingRefresher {
    calls: AtomicU32,
    fixed_lifetime: Duration,
}

impl CountingRefresher {
    fn new(lifetime: Duration) -> Self {
        Self {
            calls: AtomicU32::new(0),
            fixed_lifetime: lifetime,
        }
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TokenRefresher<u64> for CountingRefresher {
    async fn refresh(&self) -> Result<TokenSnapshot<u64>, CloudError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst).saturating_add(1);
        Ok(TokenSnapshot {
            value: u64::from(n),
            expires_at: Instant::now() + self.fixed_lifetime,
        })
    }
}

#[tokio::test]
async fn single_flight_collapses_concurrent_misses_to_one_refresh() {
    let refresher = Arc::new(CountingRefresher::new(Duration::from_mins(1)));
    let token = Arc::new(CachedTokenProvider::with_refresh_buffer(
        refresher.clone(),
        Duration::from_secs(5),
    ));

    let mut tasks = Vec::new();
    for _ in 0..50 {
        let t = token.clone();
        tasks.push(tokio::spawn(async move { t.current().await.unwrap() }));
    }
    for task in tasks {
        let v = task.await.unwrap();
        assert_eq!(v, 1, "single-flight must dedupe to refresh #1");
    }
    assert_eq!(refresher.call_count(), 1);
}

#[tokio::test]
async fn cache_hit_skips_refresh_within_buffer_window() {
    let refresher = Arc::new(CountingRefresher::new(Duration::from_mins(1)));
    let token =
        CachedTokenProvider::with_refresh_buffer(refresher.clone(), Duration::from_millis(100));

    assert_eq!(token.current().await.unwrap(), 1);
    assert_eq!(token.current().await.unwrap(), 1);
    assert_eq!(token.current().await.unwrap(), 1);
    assert_eq!(refresher.call_count(), 1);
}

#[tokio::test]
async fn refresh_when_within_buffer_of_expiry() {
    let refresher = Arc::new(CountingRefresher::new(Duration::from_millis(150)));
    // Buffer larger than lifetime — every call should preempt.
    let token =
        CachedTokenProvider::with_refresh_buffer(refresher.clone(), Duration::from_secs(10));

    assert_eq!(token.current().await.unwrap(), 1);
    assert_eq!(token.current().await.unwrap(), 2);
    assert_eq!(token.current().await.unwrap(), 3);
    assert_eq!(refresher.call_count(), 3);
}

#[tokio::test]
async fn invalidate_forces_next_call_to_refresh() {
    let refresher = Arc::new(CountingRefresher::new(Duration::from_mins(1)));
    let token =
        CachedTokenProvider::with_refresh_buffer(refresher.clone(), Duration::from_millis(100));

    assert_eq!(token.current().await.unwrap(), 1);
    assert_eq!(refresher.call_count(), 1);
    token.invalidate();
    assert_eq!(token.current().await.unwrap(), 2);
    assert_eq!(refresher.call_count(), 2);
}

#[tokio::test]
async fn refresh_failure_propagates() {
    struct FailingRefresher;
    #[async_trait]
    impl TokenRefresher<u64> for FailingRefresher {
        async fn refresh(&self) -> Result<TokenSnapshot<u64>, CloudError> {
            Err(CloudError::credential_msg("idp down"))
        }
    }
    let token = CachedTokenProvider::new(Arc::new(FailingRefresher));
    let err = token.current().await.unwrap_err();
    assert!(matches!(err, CloudError::Credential { .. }));
}
