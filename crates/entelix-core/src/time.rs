//! `Clock` — monotonic-clock abstraction shared across the
//! workspace. Living in `entelix-core` so any sub-crate that needs
//! a time source (rate limiters, retry backoff, cost-rate windows,
//! TTL pruning) can take a `&dyn Clock` without depending on
//! `entelix-policy`.
//!
//! Production code wires [`SystemClock`] (delegates to
//! `tokio::time::Instant`); tests pass a deterministic clock so
//! `tokio::time::pause()` + manual `advance` make consumers walk a
//! known schedule.

/// Monotonic-clock abstraction. Implementors must produce strictly
/// non-decreasing values; jitter or skew breaks any consumer that
/// computes elapsed time from the difference of two reads
/// (rate-limit buckets, exponential backoff, TTL windows).
pub trait Clock: Send + Sync + 'static {
    /// Microseconds since some fixed origin. The origin doesn't
    /// matter — only differences are read by consumers.
    fn now_micros(&self) -> u64;
}

/// `Clock` backed by `tokio::time::Instant`. Honours
/// `tokio::time::pause` so test harnesses can simulate elapsed
/// time without real waits.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_micros(&self) -> u64 {
        // `tokio::time::Instant` uses a monotonic source; converting
        // through a stable origin keeps the absolute value bounded
        // for the process lifetime.
        let origin = origin_instant();
        let now = tokio::time::Instant::now();
        u64::try_from(now.duration_since(origin).as_micros()).unwrap_or(u64::MAX)
    }
}

fn origin_instant() -> tokio::time::Instant {
    use std::sync::OnceLock;
    static ORIGIN: OnceLock<tokio::time::Instant> = OnceLock::new();
    *ORIGIN.get_or_init(tokio::time::Instant::now)
}
