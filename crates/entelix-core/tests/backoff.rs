//! `ExponentialBackoff` deterministic sequence + bounds tests.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use std::time::Duration;

use entelix_core::backoff::ExponentialBackoff;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn no_jitter_doubles_per_attempt_until_max() {
    let backoff = ExponentialBackoff::new(Duration::from_millis(10), Duration::from_secs(1))
        .with_jitter(0.0)
        .with_max_attempts(8);
    let mut rng = StdRng::seed_from_u64(42);
    let delays: Vec<u64> = (0..8)
        .map(|n| backoff.delay_for_attempt(n, &mut rng).as_millis() as u64)
        .collect();
    assert_eq!(delays, vec![10, 20, 40, 80, 160, 320, 640, 1000]); // capped at 1s
}

#[test]
fn delay_zero_after_max_attempts() {
    let backoff = ExponentialBackoff::new(Duration::from_millis(10), Duration::from_secs(1))
        .with_jitter(0.0)
        .with_max_attempts(3);
    let mut rng = StdRng::seed_from_u64(0);
    assert_eq!(backoff.delay_for_attempt(2, &mut rng).as_millis(), 40);
    assert_eq!(backoff.delay_for_attempt(3, &mut rng), Duration::ZERO);
    assert_eq!(backoff.delay_for_attempt(99, &mut rng), Duration::ZERO);
}

#[test]
fn jitter_is_deterministic_with_seeded_rng() {
    let backoff = ExponentialBackoff::new(Duration::from_millis(100), Duration::from_secs(10))
        .with_jitter(0.3)
        .with_max_attempts(5);
    let mut rng_a = StdRng::seed_from_u64(123);
    let mut rng_b = StdRng::seed_from_u64(123);
    for n in 0..5 {
        assert_eq!(
            backoff.delay_for_attempt(n, &mut rng_a),
            backoff.delay_for_attempt(n, &mut rng_b),
            "attempt {n} diverged"
        );
    }
}

#[test]
fn jitter_stays_within_unjittered_plus_ratio() {
    let base = Duration::from_millis(100);
    let backoff = ExponentialBackoff::new(base, Duration::from_secs(10))
        .with_jitter(0.3)
        .with_max_attempts(10);
    let mut rng = StdRng::seed_from_u64(7);
    for n in 0..6 {
        let unjittered = (base.as_millis() as u64) << n;
        let unjittered = unjittered.min(10_000);
        let max_with_jitter = unjittered + (unjittered as f64 * 0.3) as u64 + 1;
        let delay = backoff.delay_for_attempt(n, &mut rng).as_millis() as u64;
        assert!(delay >= unjittered, "attempt {n}: {delay} < {unjittered}");
        assert!(
            delay <= max_with_jitter,
            "attempt {n}: {delay} > {max_with_jitter}"
        );
    }
}

#[test]
fn ratio_clamped_to_zero_one() {
    let bo =
        ExponentialBackoff::new(Duration::from_millis(10), Duration::from_secs(1)).with_jitter(2.5);
    assert_eq!(bo.jitter_ratio(), 1.0);
    let bo = ExponentialBackoff::new(Duration::from_millis(10), Duration::from_secs(1))
        .with_jitter(-1.0);
    assert_eq!(bo.jitter_ratio(), 0.0);
}

#[test]
fn default_is_sensible() {
    let bo = ExponentialBackoff::default();
    assert_eq!(bo.base(), Duration::from_millis(100));
    assert_eq!(bo.max(), Duration::from_secs(30));
    assert_eq!(bo.max_attempts(), 5);
}
