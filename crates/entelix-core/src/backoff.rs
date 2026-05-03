//! Exponential backoff with jitter.
//!
//! `ExponentialBackoff` produces a deterministic sequence of delays
//! suitable for retry / reconnect loops. The delay for attempt `n` is
//! `min(base * 2.pow(n), max)` with optional jitter applied as a uniform
//! random fraction of the unjittered delay.
//!
//! Jitter source is injectable (`rand::Rng`), so tests pass a seeded
//! RNG to assert exact sequences. Production callers typically reuse
//! `rand::rng()` (the per-thread default).

use std::time::Duration;

use rand::{Rng, RngExt};

/// Default jitter ratio — 30% of the unjittered delay.
pub const DEFAULT_JITTER_RATIO: f64 = 0.3;

/// Default cap on retry attempts before the caller should give up.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 5;

/// Exponential backoff with optional jitter.
///
/// Cheap to clone — all fields are `Copy`.
#[derive(Clone, Copy, Debug)]
pub struct ExponentialBackoff {
    base: Duration,
    max: Duration,
    jitter_ratio: f64,
    max_attempts: u32,
}

impl ExponentialBackoff {
    /// Build with `base` doubled per attempt up to `max`.
    pub const fn new(base: Duration, max: Duration) -> Self {
        Self {
            base,
            max,
            jitter_ratio: DEFAULT_JITTER_RATIO,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
        }
    }

    /// Override the jitter ratio. `0.0` disables jitter.
    /// Values outside `[0.0, 1.0]` are clamped on read.
    #[must_use]
    pub const fn with_jitter(mut self, ratio: f64) -> Self {
        self.jitter_ratio = ratio;
        self
    }

    /// Override the retry attempt cap (used by callers that loop on
    /// [`Self::delay_for_attempt`]).
    #[must_use]
    pub const fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    /// Borrow the configured base delay.
    pub const fn base(&self) -> Duration {
        self.base
    }

    /// Borrow the configured max delay.
    pub const fn max(&self) -> Duration {
        self.max
    }

    /// Effective jitter ratio — clamped to `[0.0, 1.0]`.
    pub const fn jitter_ratio(&self) -> f64 {
        self.jitter_ratio.clamp(0.0, 1.0)
    }

    /// Configured retry attempt cap.
    pub const fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Delay for the n-th retry (0 = first retry after the original
    /// failure). Returns `Duration::ZERO` if `attempt >= max_attempts`
    /// — caller should stop retrying.
    ///
    /// `rng` is sampled when `jitter_ratio > 0`; pass a seeded RNG in
    /// tests for deterministic sequences.
    pub fn delay_for_attempt<R: Rng + ?Sized>(&self, attempt: u32, rng: &mut R) -> Duration {
        if attempt >= self.max_attempts {
            return Duration::ZERO;
        }
        let exp = attempt.min(30); // cap to avoid u128 saturation on extreme inputs
        let multiplier: u128 = 1u128 << exp;
        let base_nanos = u128::from(u64::try_from(self.base.as_nanos()).unwrap_or(u64::MAX));
        let max_nanos = u128::from(u64::try_from(self.max.as_nanos()).unwrap_or(u64::MAX));
        let unjittered = base_nanos.saturating_mul(multiplier).min(max_nanos);
        let unjittered_ns = u64::try_from(unjittered).unwrap_or(u64::MAX);

        let jitter_ratio = self.jitter_ratio();
        if jitter_ratio <= 0.0 {
            return Duration::from_nanos(unjittered_ns);
        }
        // Sample jitter as an integer fraction of the unjittered delay.
        // `random_range(0..=N)` over u128 keeps the entire calculation in
        // integer space — no float precision loss across the full
        // `Duration` range (up to 2^64 ns ≈ 584 years).
        let max_offset_ns: u128 =
            (u128::from(unjittered_ns)).saturating_mul(jitter_to_ppm(jitter_ratio)) / 1_000_000;
        let offset_ns: u128 = rng.random_range(0..=max_offset_ns);
        let offset_u64 = u64::try_from(offset_ns).unwrap_or(u64::MAX);
        Duration::from_nanos(unjittered_ns.saturating_add(offset_u64))
    }
}

/// Convert a `[0.0, 1.0]` jitter ratio into parts-per-million for
/// integer-space arithmetic.
///
/// `0.3` maps to `300_000`, `1.0` maps to `1_000_000`. Values outside
/// the unit interval are clamped.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn jitter_to_ppm(ratio: f64) -> u128 {
    // `clamped` is in [0.0, 1.0]; multiplying by 1_000_000 stays well
    // below f64's exact-integer range (2^53), so the cast is lossless.
    let clamped = ratio.clamp(0.0, 1.0);
    (clamped * 1_000_000.0).round() as u128
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self::new(Duration::from_millis(100), Duration::from_secs(30))
    }
}
