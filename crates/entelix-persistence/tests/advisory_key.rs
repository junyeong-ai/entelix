//! `AdvisoryKey` deterministic + collision-resistance tests.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::collections::HashSet;

use entelix_persistence::AdvisoryKey;

#[test]
fn deterministic_across_calls() {
    let a = AdvisoryKey::from_strings(&["alpha", "beta"]);
    let b = AdvisoryKey::from_strings(&["alpha", "beta"]);
    assert_eq!(a, b);
    assert_eq!(a.raw(), b.raw());
}

#[test]
fn different_part_compositions_diverge() {
    let a = AdvisoryKey::from_strings(&["a", "bc"]);
    let b = AdvisoryKey::from_strings(&["ab", "c"]);
    assert_ne!(a, b, "NUL separator should disambiguate");
}

#[test]
fn halves_round_trip() {
    let key = AdvisoryKey::from_strings(&["x", "y"]);
    let (high, low) = key.halves();
    let recombined = ((i64::from(high)) << 32) | (i64::from(low) & 0xFFFF_FFFF);
    assert_eq!(recombined, key.raw());
}

#[test]
fn redis_key_is_ascii_and_well_formed() {
    let key = AdvisoryKey::from_strings(&["t1", "th42"]);
    let s = key.redis_key();
    assert!(s.starts_with("entelix:lock:"));
    assert_eq!(s.len(), 29);
    let hex_part = &s[13..];
    assert!(
        hex_part
            .chars()
            .all(|c| c.is_ascii_hexdigit() && (c.is_ascii_digit() || c.is_ascii_lowercase()))
    );
}

#[test]
fn for_session_namespace_distinct_from_raw() {
    // Same parts as for_session uses internally must NOT collide with a
    // random tenant/thread combination.
    let tenant = entelix_core::TenantId::new("acme");
    let session = AdvisoryKey::for_session(&tenant, "thread-7");
    let raw = AdvisoryKey::from_strings(&["acme", "thread-7"]);
    assert_ne!(session, raw, "namespace prefix must alter the hash");
}

#[test]
fn collision_rate_below_one_in_million() {
    // 100k random tenant/thread strings → 0 collisions expected for a
    // good 64-bit hash. xxh3 has near-ideal avalanche properties; this
    // sanity-checks our derivation pipeline.
    let mut seen: HashSet<i64> = HashSet::with_capacity(100_000);
    let mut redis_seen: HashSet<String> = HashSet::with_capacity(100_000);
    for tenant_n in 0..1000u64 {
        for thread_n in 0..100u64 {
            let tenant = entelix_core::TenantId::new(format!("tenant-{tenant_n:08x}"));
            let thread = format!("thread-{thread_n:08x}");
            let key = AdvisoryKey::for_session(&tenant, &thread);
            assert!(
                seen.insert(key.raw()),
                "raw collision at ({tenant}, {thread}): {}",
                key.raw()
            );
            assert!(
                redis_seen.insert(key.redis_key().to_owned()),
                "redis_key collision at ({tenant}, {thread})"
            );
        }
    }
    assert_eq!(seen.len(), 100_000);
}

#[test]
fn high_low_uniformly_independent() {
    // Cheap sanity: across 10k keys, the high half and low half have
    // similar bit-set counts (~50% per bit).
    let mut high_set_count = 0u64;
    let mut low_set_count = 0u64;
    let total: u64 = 10_000;
    for n in 0..total {
        let tenant = entelix_core::TenantId::new("uniform");
        let key = AdvisoryKey::for_session(&tenant, &n.to_string());
        let (high, low) = key.halves();
        high_set_count += u64::from(high.count_ones());
        low_set_count += u64::from(low.count_ones());
    }
    let ideal = total * 16; // 32 bits each, average half set
    let tolerance = ideal / 10; // ±10%
    assert!(
        high_set_count.abs_diff(ideal) < tolerance,
        "high-half bit count {high_set_count} too far from ideal {ideal}"
    );
    assert!(
        low_set_count.abs_diff(ideal) < tolerance,
        "low-half bit count {low_set_count} too far from ideal {ideal}"
    );
}
