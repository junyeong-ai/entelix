//! `TenantId` — validating newtype for the multi-tenant scope key.
//!
//! Invariant 11 (CLAUDE.md) demands cross-tenant data leak be
//! structurally impossible: every tenant-bearing field on
//! [`crate::ExecutionContext`], [`crate::ThreadKey`],
//! `entelix_memory::Namespace`, `entelix_memory::NamespacePrefix`,
//! and `entelix_graph::Checkpoint` carries this type. The only
//! constructors that accept untrusted input ([`TenantId::try_from`],
//! [`serde::Deserialize`]) reject the empty string up-front, so a
//! tenantless wire payload, a hand-crafted bad rendered key, or a
//! persistence row with an empty `tenant_id` column surface as
//! [`crate::Error::InvalidRequest`] — never as a constructed
//! instance whose downstream rendering or row filter then collapses
//! every tenant onto one effective key.
//!
//! Backed by `Arc<str>` so cloning a `TenantId` (done implicitly per
//! tool dispatch and per sub-agent context fan-out) is an atomic
//! refcount bump, not an allocation. The default-tenant Arc is
//! initialised once per process via a `OnceLock`, so a freshly-built
//! [`crate::ExecutionContext`] allocates zero strings on the hot path.

use std::borrow::Borrow;
use std::fmt;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, Result};

/// Default tenant identifier — applied when callers do not specify
/// one explicitly via [`crate::ExecutionContext::with_tenant_id`].
/// Single-tenant deployments rely on this and never construct a
/// [`TenantId`] themselves.
pub const DEFAULT_TENANT_ID: &str = "default";

/// Validating wrapper around the per-tenant scope identifier.
///
/// Two construction paths:
/// - [`TenantId::try_from`] (and [`serde::Deserialize`], which routes
///   through it) — runtime input that may be empty / malformed.
///   Surfaces empty input as [`crate::Error::InvalidRequest`].
/// - [`TenantId::new`] — `panic!` on empty. Use only for test
///   fixtures and migration tooling that have already validated the
///   inputs; production paths take a [`TenantId`] argument and
///   inherit the validation from whoever built it.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct TenantId(Arc<str>);

impl TenantId {
    /// Build a tenant id from a known non-empty literal. Panics
    /// when `s` is empty — programmer-error grade for test fixtures
    /// and migration tooling. Production input arrives through
    /// [`Self::try_from`] (or [`Deserialize`], which routes through
    /// it).
    ///
    /// # Panics
    ///
    /// Panics on empty input.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new(s: impl AsRef<str>) -> Self {
        Self::try_from(s.as_ref()).expect("TenantId::new: tenant_id must be non-empty")
    }

    /// Borrow the underlying string without an extra refcount bump.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Process-wide shared `TenantId` for [`DEFAULT_TENANT_ID`]. Every
/// `ExecutionContext::new()` clones this — an atomic refcount bump
/// instead of allocating a fresh `Arc<str>`.
fn default_shared() -> &'static TenantId {
    static SHARED: OnceLock<TenantId> = OnceLock::new();
    SHARED.get_or_init(|| TenantId(Arc::from(DEFAULT_TENANT_ID)))
}

impl Default for TenantId {
    fn default() -> Self {
        default_shared().clone()
    }
}

impl AsRef<str> for TenantId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for TenantId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

// Cross-type equality with `str` — mirrors the `PartialEq<str>`
// impls on `String`, `Path`, `OsStr`, and `Url` from the standard
// library so call sites read `ctx.tenant_id() == "acme"` without
// an `.as_str()` dance. The implementation is value-equality on
// the underlying `Arc<str>`'s contents; identity (Arc pointer
// equality) is irrelevant here.
impl PartialEq<str> for TenantId {
    fn eq(&self, other: &str) -> bool {
        &*self.0 == other
    }
}

impl PartialEq<&str> for TenantId {
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

impl PartialEq<TenantId> for str {
    fn eq(&self, other: &TenantId) -> bool {
        self == &*other.0
    }
}

impl PartialEq<TenantId> for &str {
    fn eq(&self, other: &TenantId) -> bool {
        *self == &*other.0
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TenantId").field(&&*self.0).finish()
    }
}

impl TryFrom<&str> for TenantId {
    type Error = Error;
    fn try_from(s: &str) -> Result<Self> {
        if s.is_empty() {
            return Err(Error::invalid_request(
                "tenant_id must be non-empty (invariant 11)",
            ));
        }
        Ok(Self(Arc::from(s)))
    }
}

impl TryFrom<String> for TenantId {
    type Error = Error;
    fn try_from(s: String) -> Result<Self> {
        if s.is_empty() {
            return Err(Error::invalid_request(
                "tenant_id must be non-empty (invariant 11)",
            ));
        }
        Ok(Self(Arc::from(s)))
    }
}

impl Serialize for TenantId {
    fn serialize<S: Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for TenantId {
    fn deserialize<D: Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::try_from(s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn try_from_rejects_empty_str() {
        let err = TenantId::try_from("").unwrap_err();
        assert!(format!("{err}").contains("tenant_id must be non-empty"));
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn try_from_rejects_empty_string() {
        let err = TenantId::try_from(String::new()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn try_from_accepts_non_empty() {
        let t = TenantId::try_from("acme").unwrap();
        assert_eq!(t.as_str(), "acme");
    }

    #[test]
    #[should_panic(expected = "tenant_id must be non-empty")]
    fn new_panics_on_empty() {
        let _ = TenantId::new("");
    }

    #[test]
    fn default_returns_default_tenant() {
        assert_eq!(TenantId::default().as_str(), DEFAULT_TENANT_ID);
    }

    #[test]
    fn default_clones_share_arc() {
        // Two `TenantId::default()` calls share the underlying
        // Arc<str> — refcount bump instead of allocation.
        let a = TenantId::default();
        let b = TenantId::default();
        assert!(Arc::ptr_eq(&a.0, &b.0));
    }

    #[test]
    fn deserialize_rejects_empty_string() {
        let err = serde_json::from_str::<TenantId>(r#""""#).unwrap_err();
        assert!(err.to_string().contains("tenant_id must be non-empty"));
    }

    #[test]
    fn deserialize_accepts_non_empty() {
        let t: TenantId = serde_json::from_str(r#""acme""#).unwrap();
        assert_eq!(t.as_str(), "acme");
    }

    #[test]
    fn serialize_emits_bare_string() {
        let t = TenantId::new("acme");
        assert_eq!(serde_json::to_string(&t).unwrap(), r#""acme""#);
    }

    #[test]
    fn round_trip_via_serde() {
        let t = TenantId::new("acme");
        let s = serde_json::to_string(&t).unwrap();
        let back: TenantId = serde_json::from_str(&s).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn borrow_str_enables_hashmap_lookup_by_str() {
        use std::collections::HashMap;
        let mut m: HashMap<TenantId, u32> = HashMap::new();
        m.insert(TenantId::new("acme"), 1);
        // `Borrow<str>` lets `&str` lookups hit a `HashMap<TenantId, _>`.
        assert_eq!(m.get("acme"), Some(&1));
    }
}
