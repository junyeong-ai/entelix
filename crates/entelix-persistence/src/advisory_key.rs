//! Advisory lock key derivation.
//!
//! Postgres advisory locks come in two API shapes:
//! - `pg_try_advisory_xact_lock(bigint)` — single 64-bit key
//! - `pg_try_advisory_xact_lock(int, int)` — two 32-bit keys, used
//!   together as a 64-bit composite
//!
//! Both ultimately address a 64-bit lock space. To avoid the
//! collision risk of Postgres's built-in `hashtext()` (32-bit Jenkins
//! hash, ~7e4 collisions per million keys), we derive the key with
//! xxh3-64 — a 64-bit non-cryptographic hash with strong avalanche
//! properties.
//!
//! Redis lock keys are derived from the same hash via
//! [`AdvisoryKey::redis_key`] so a single tenant/thread pair maps to
//! the same logical lock regardless of backend.

use std::fmt;

use twox_hash::XxHash64;

const ADVISORY_NAMESPACE: &str = "entelix:lock";
const ADVISORY_SEED: u64 = 0x656e_7465_6c69_785f; // "entelix_" little-endian-ish

// 13 bytes of `entelix:lock:` prefix + 16 bytes of lowercase hex
// representing the 64-bit hash = 29 ASCII bytes total.
const REDIS_KEY_LEN: usize = 29;

/// 64-bit hashed key used for distributed lock derivation.
///
/// Stable across runs (xxh3 is deterministic) and uniformly
/// distributed across the i64 space — high half and low half are
/// each independently uniform, suitable for the two-arg Postgres
/// advisory lock variant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AdvisoryKey {
    raw: i64,
    redis_key: [u8; REDIS_KEY_LEN],
}

impl AdvisoryKey {
    /// Hash a sequence of string parts into a single key. Parts are
    /// joined with a NUL separator so `("a", "bc")` and `("ab", "c")`
    /// hash to different values.
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_strings(parts: &[&str]) -> Self {
        use std::hash::Hasher;
        let mut hasher = XxHash64::with_seed(ADVISORY_SEED);
        for part in parts {
            hasher.write(part.as_bytes());
            hasher.write(&[0u8]);
        }
        let hash = hasher.finish();
        // u64 → i64 bit-preserving cast (Postgres expects signed).
        let raw = hash as i64;
        let redis_key = encode_redis_key(hash);
        Self { raw, redis_key }
    }

    /// Convenience: namespace + `tenant_id` + `thread_id` (the
    /// canonical session-lock derivation).
    pub fn for_session(tenant_id: &str, thread_id: &str) -> Self {
        Self::from_strings(&[ADVISORY_NAMESPACE, "session", tenant_id, thread_id])
    }

    /// 64-bit key for `pg_try_advisory_xact_lock(bigint)`.
    pub const fn raw(&self) -> i64 {
        self.raw
    }

    /// Two 32-bit halves for `pg_try_advisory_xact_lock(int, int)`.
    /// `(high, low)` ordering matches Postgres docs.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn halves(&self) -> (i32, i32) {
        let high = (self.raw >> 32) as i32;
        let low = (self.raw & 0xFFFF_FFFF) as i32;
        (high, low)
    }

    /// Redis lock key string — `entelix:lock:<8-byte hex>`. Stable
    /// across processes and machines.
    pub fn redis_key(&self) -> &str {
        // SAFETY-equivalent: `encode_redis_key` only writes ASCII bytes
        // (hex digits + colon + lowercase namespace), so the returned
        // slice is always valid UTF-8.
        std::str::from_utf8(&self.redis_key).expect("redis key is ASCII")
    }
}

impl fmt::Display for AdvisoryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.redis_key())
    }
}

#[allow(clippy::cast_possible_truncation)]
fn encode_redis_key(hash: u64) -> [u8; REDIS_KEY_LEN] {
    const PREFIX: &[u8; 13] = b"entelix:lock:";
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; REDIS_KEY_LEN];
    out[..13].copy_from_slice(PREFIX);
    let bytes = hash.to_be_bytes();
    let mut idx = 13;
    for byte in bytes {
        out[idx] = HEX[((byte >> 4) & 0x0f) as usize];
        out[idx + 1] = HEX[(byte & 0x0f) as usize];
        idx += 2;
    }
    debug_assert_eq!(idx, REDIS_KEY_LEN);
    out
}
