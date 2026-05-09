//! `RateLimiter` trait + [`TokenBucketLimiter`] — async, per-key,
//! time-injectable.
//!
//! Token-bucket semantics: each key has a bucket of capacity
//! `capacity_tokens`. The bucket refills at `refill_per_sec` tokens
//! per second up to capacity. `try_acquire(key, n)` succeeds if the
//! bucket has at least `n` tokens (deducting them); fails with
//! [`PolicyError::RateLimited`] otherwise, indicating how long the
//! caller must wait before enough tokens accumulate.
//!
//! Time is injected via the [`entelix_core::Clock`] trait.
//! Production wires [`entelix_core::SystemClock`] (delegates to
//! `tokio::time::Instant`); tests pass a deterministic clock so
//! `tokio::time::pause()` + manual `advance` make the bucket walk
//! a known schedule.

// `parking_lot::Mutex` guards on `Bucket` are scoped inside non-async
// blocks (we never hold a guard across `.await`). clippy's
// `significant_drop_tightening` flags the binding pattern even when
// the block scope already drops correctly.
#![allow(clippy::significant_drop_tightening)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use entelix_core::{Clock, SystemClock};
use parking_lot::Mutex;

use crate::error::{PolicyError, PolicyResult};

/// Backend-agnostic rate-limit surface.
#[async_trait]
pub trait RateLimiter: Send + Sync + 'static {
    /// Try to acquire `tokens` from the bucket keyed by `key`.
    /// Returns `Ok(())` on grant. On refusal returns
    /// [`PolicyError::RateLimited`] with `retry_after_ms` indicating
    /// when retry could plausibly succeed (assuming no other
    /// concurrent draws).
    async fn try_acquire(&self, key: &str, tokens: u32) -> PolicyResult<()>;
}

/// Per-key token-bucket limiter. Buckets are created lazily on first
/// `try_acquire`; a key never seen before starts full.
pub struct TokenBucketLimiter {
    capacity: u32,
    refill_per_sec: f64,
    clock: Arc<dyn Clock>,
    buckets: DashMap<String, Mutex<Bucket>>,
}

#[derive(Clone, Copy, Debug)]
struct Bucket {
    /// Current bucket level, in micro-tokens (1 token = `1_000_000`
    /// micro-tokens). Integer math avoids float drift across many
    /// refills.
    level: u64,
    /// Last refill timestamp, in microseconds since clock origin.
    last_refill_micros: u64,
}

const MICRO: u64 = 1_000_000;

#[allow(
    clippy::missing_fields_in_debug,
    reason = "clock is dyn-trait without Debug; printed as active-key count"
)]
impl std::fmt::Debug for TokenBucketLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBucketLimiter")
            .field("capacity", &self.capacity)
            .field("refill_per_sec", &self.refill_per_sec)
            .field("active_keys", &self.buckets.len())
            .finish()
    }
}

impl TokenBucketLimiter {
    /// Build a limiter with `capacity` tokens that refills at
    /// `refill_per_sec` tokens per second. Refusing values: zero
    /// capacity (would deny all calls) or zero refill (bucket
    /// can't recover) — both rejected as
    /// [`PolicyError::Config`].
    pub fn new(capacity: u32, refill_per_sec: f64) -> PolicyResult<Self> {
        Self::with_clock(capacity, refill_per_sec, Arc::new(SystemClock))
    }

    /// Build with a custom clock. Used by tests; production calls
    /// [`Self::new`].
    pub fn with_clock(
        capacity: u32,
        refill_per_sec: f64,
        clock: Arc<dyn Clock>,
    ) -> PolicyResult<Self> {
        if capacity == 0 {
            return Err(PolicyError::Config("capacity must be > 0".into()));
        }
        if !(refill_per_sec.is_finite() && refill_per_sec > 0.0) {
            return Err(PolicyError::Config(
                "refill_per_sec must be a finite positive number".into(),
            ));
        }
        // Refusing rates so small that scaling overflows micro-token
        // precision — a single token would take more than `u64::MAX`
        // microseconds (~580_000 years) to accumulate, which is
        // almost certainly a configuration mistake. Rates from
        // `1e-6` upward are accepted (1 token per ~11.5 days at the
        // floor, which exercises the slow-rate path correctly).
        if refill_per_sec < 1.0e-6 {
            return Err(PolicyError::Config(format!(
                "refill_per_sec={refill_per_sec} is below the supported floor (1e-6 — \
                 1 token per ~11.5 days). Increase the rate or pre-aggregate at the caller."
            )));
        }
        Ok(Self {
            capacity,
            refill_per_sec,
            clock,
            buckets: DashMap::new(),
        })
    }

    /// How many distinct keys currently have buckets allocated. Test
    /// helper.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.buckets.len()
    }

    fn capacity_micro(&self) -> u64 {
        u64::from(self.capacity).saturating_mul(MICRO)
    }

    /// Micro-tokens added per second of elapsed time — i.e.
    /// `refill_per_sec * 1_000_000` rounded to u64. Storing rate at
    /// micro-token-per-second granularity (rather than the previous
    /// micro-token-per-*microsecond*) lets sub-1-RPS rates resolve
    /// correctly without silent rounding to a higher rate.
    ///
    /// Multiplication then divides by `MICRO` (microseconds per
    /// second) to recover the per-elapsed-microsecond contribution:
    ///
    /// ```ignore
    /// added_micro_tokens = (elapsed_micros * refill_micro_per_sec) / MICRO
    /// ```
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scaled value is `> 0.0` after construction-time validation and saturated to u64::MAX above the ceiling"
    )]
    fn refill_micro_per_sec(&self) -> u64 {
        let scaled = self.refill_per_sec * 1_000_000.0_f64;
        if scaled >= 18_446_744_073_709_551_615.0_f64 {
            u64::MAX
        } else {
            scaled as u64
        }
    }
}

#[async_trait]
impl RateLimiter for TokenBucketLimiter {
    async fn try_acquire(&self, key: &str, tokens: u32) -> PolicyResult<()> {
        if tokens == 0 {
            return Ok(());
        }
        let cost_micro = u64::from(tokens).saturating_mul(MICRO);
        let cap = self.capacity_micro();
        if cost_micro > cap {
            return Err(PolicyError::Config(format!(
                "request for {tokens} tokens exceeds bucket capacity {cap_tokens}",
                cap_tokens = self.capacity
            )));
        }
        let now = self.clock.now_micros();
        let rate_per_sec = self.refill_micro_per_sec();
        // Bucket-touching critical section. Lock guard is dropped before
        // any await — the lock is non-async (`parking_lot::Mutex`) but
        // we still scope it tight to keep contention minimal.
        let admitted = {
            let entry = self.buckets.entry(key.to_owned()).or_insert_with(|| {
                Mutex::new(Bucket {
                    level: cap,
                    last_refill_micros: now,
                })
            });
            let mut bucket = entry.lock();
            let elapsed_micros = now.saturating_sub(bucket.last_refill_micros);
            // added = (elapsed_micros * rate_per_sec) / MICRO. Saturating
            // mul handles enormous elapsed values; the divide brings the
            // scale back down to micro-tokens.
            let added = elapsed_micros.saturating_mul(rate_per_sec) / MICRO;
            bucket.level = bucket.level.saturating_add(added).min(cap);
            bucket.last_refill_micros = now;
            if bucket.level >= cost_micro {
                bucket.level -= cost_micro;
                Ok(())
            } else {
                let deficit = cost_micro - bucket.level;
                Err(deficit)
            }
        };
        match admitted {
            Ok(()) => Ok(()),
            Err(deficit) => {
                // retry_after_micros = ceil(deficit * MICRO / rate_per_sec).
                // Saturating mul is defensive; rate_per_sec ≥ 1 by
                // construction-time validation, so divide is safe.
                let retry_after_micros =
                    deficit.saturating_mul(MICRO).div_ceil(rate_per_sec.max(1));
                let retry_after_ms = Duration::from_micros(retry_after_micros).as_millis();
                Err(PolicyError::RateLimited {
                    key: key.to_owned(),
                    retry_after_ms: u64::try_from(retry_after_ms).unwrap_or(u64::MAX),
                })
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// Manual clock for deterministic time travel.
    #[derive(Default)]
    struct ManualClock(AtomicU64);

    impl ManualClock {
        fn advance(&self, micros: u64) {
            self.0.fetch_add(micros, Ordering::SeqCst);
        }
    }

    impl Clock for ManualClock {
        fn now_micros(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn limiter(capacity: u32, refill: f64) -> (Arc<ManualClock>, TokenBucketLimiter) {
        let clock = Arc::new(ManualClock::default());
        let limiter = TokenBucketLimiter::with_clock(capacity, refill, clock.clone()).unwrap();
        (clock, limiter)
    }

    #[tokio::test]
    async fn fresh_bucket_starts_full() {
        let (_clock, limiter) = limiter(10, 1.0);
        for _ in 0..10 {
            limiter.try_acquire("k", 1).await.unwrap();
        }
    }

    #[tokio::test]
    async fn refused_when_empty() {
        let (_clock, limiter) = limiter(3, 1.0);
        for _ in 0..3 {
            limiter.try_acquire("k", 1).await.unwrap();
        }
        let err = limiter.try_acquire("k", 1).await.unwrap_err();
        match err {
            PolicyError::RateLimited { retry_after_ms, .. } => assert!(retry_after_ms > 0),
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn refill_after_advancing_clock() {
        let (clock, limiter) = limiter(2, 1.0);
        limiter.try_acquire("k", 2).await.unwrap();
        // 1.5 seconds → 1 token refilled (rate=1/s).
        clock.advance(1_500_000);
        limiter.try_acquire("k", 1).await.unwrap();
        // Still empty — 0.5s left over isn't enough for another token.
        let err = limiter.try_acquire("k", 1).await.unwrap_err();
        assert!(matches!(err, PolicyError::RateLimited { .. }));
    }

    #[tokio::test]
    async fn keys_are_independent() {
        let (_clock, limiter) = limiter(1, 1.0);
        limiter.try_acquire("alpha", 1).await.unwrap();
        // Different key — fresh bucket.
        limiter.try_acquire("bravo", 1).await.unwrap();
        let err = limiter.try_acquire("alpha", 1).await.unwrap_err();
        assert!(matches!(err, PolicyError::RateLimited { .. }));
    }

    #[tokio::test]
    async fn request_above_capacity_is_config_error() {
        let (_clock, limiter) = limiter(5, 1.0);
        let err = limiter.try_acquire("k", 10).await.unwrap_err();
        assert!(matches!(err, PolicyError::Config(_)));
    }

    #[tokio::test]
    async fn zero_token_request_is_free() {
        let (_clock, limiter) = limiter(1, 1.0);
        limiter.try_acquire("k", 1).await.unwrap();
        // Bucket is empty; zero-token request still succeeds.
        limiter.try_acquire("k", 0).await.unwrap();
    }

    #[test]
    fn invalid_capacity_rejected() {
        let err = TokenBucketLimiter::new(0, 1.0).unwrap_err();
        assert!(matches!(err, PolicyError::Config(_)));
    }

    #[test]
    fn invalid_refill_rejected() {
        let err = TokenBucketLimiter::new(10, 0.0).unwrap_err();
        assert!(matches!(err, PolicyError::Config(_)));
        let err = TokenBucketLimiter::new(10, f64::NAN).unwrap_err();
        assert!(matches!(err, PolicyError::Config(_)));
    }

    #[test]
    fn unsupportably_slow_refill_is_rejected() {
        // 1 token per ~32 years would silently round to zero in the
        // micro-token-per-microsecond integer math; reject up front.
        let err = TokenBucketLimiter::new(1, 1.0e-9).unwrap_err();
        assert!(matches!(err, PolicyError::Config(_)));
    }

    #[tokio::test]
    async fn sub_1_rps_rate_does_not_silently_double() {
        // 0.5 tokens/sec — the previous .max(1.0) floor would have
        // delivered 1 token after 1 second instead of after 2.
        let (clock, limiter) = limiter(2, 0.5);
        // Drain the starting bucket.
        limiter.try_acquire("k", 2).await.unwrap();

        // After 1s only half a token has accumulated — must be denied.
        clock.advance(1_000_000);
        let err = limiter.try_acquire("k", 1).await.unwrap_err();
        assert!(matches!(err, PolicyError::RateLimited { .. }));

        // Another 1s elapsed — total 2s, so a full token is now ready.
        clock.advance(1_000_000);
        limiter.try_acquire("k", 1).await.unwrap();
    }

    #[tokio::test]
    async fn very_slow_refill_eventually_grants() {
        // 1 token per minute — sub-1 rps but well above the floor.
        let rate = 1.0 / 60.0;
        let (clock, limiter) = limiter(1, rate);
        limiter.try_acquire("k", 1).await.unwrap();
        // 30s in: still empty.
        clock.advance(30_000_000);
        assert!(limiter.try_acquire("k", 1).await.is_err());
        // 31s more (61s total): full token accrued.
        clock.advance(31_000_000);
        limiter.try_acquire("k", 1).await.unwrap();
    }
}
