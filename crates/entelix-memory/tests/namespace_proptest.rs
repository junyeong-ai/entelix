//! Property-based regression for `Namespace::render` ↔ `Namespace::parse`
//! invertibility (ADR-0039).
//!
//! The unit-test suite in `src/namespace.rs` covers hand-picked cases —
//! empty scope segments, segments containing the `:` delimiter, segments
//! containing the `\` escape, BOM-prefixed input. Property tests round-trip
//! arbitrary `tenant_id` + scope segments to catch escaping bugs the unit
//! cases miss (e.g. mixed `\:` / `:\` sequences, multiple consecutive
//! escapes, segments composed entirely of escape characters).
//!
//! Property: `Namespace::parse(&ns.render()) == Ok(ns.clone())` for every
//! `Namespace` constructed with non-empty `tenant_id` (the constructor
//! refuses empty) and any `scope` segments.

#![allow(clippy::unwrap_used)]

use entelix_core::TenantId;
use entelix_memory::Namespace;
use proptest::prelude::*;

/// Strategy for non-empty `tenant_id` strings — `Namespace::new`
/// rejects empty, so a proptest that generates `""` would test a
/// disallowed input and skew the property. Use a non-empty regex
/// admitting the full Unicode range modulo control chars (ASCII
/// printable + a Unicode-letter sample is enough for the escaping
/// path; the render layer is byte-oriented and doesn't introspect).
fn tenant_strategy() -> impl Strategy<Value = String> {
    "[A-Za-z0-9_\\-:\\\\.]{1,32}".prop_filter("tenant must be non-empty", |s| !s.is_empty())
}

/// Scope segments admit any string including empty / colon / backslash.
/// The render layer escapes `:` → `\:` and `\` → `\\` so any byte
/// sequence in a segment must round-trip through parse intact.
fn scope_segment_strategy() -> impl Strategy<Value = String> {
    "[^[:cntrl:]]{0,16}"
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// Render then parse returns the original `Namespace`.
    #[test]
    fn render_parse_round_trip(
        tenant in tenant_strategy(),
        scope in proptest::collection::vec(scope_segment_strategy(), 0..6),
    ) {
        let mut ns = Namespace::new(TenantId::new(&tenant));
        for seg in &scope {
            ns = ns.with_scope(seg);
        }
        let rendered = ns.render();
        let parsed = Namespace::parse(&rendered)
            .unwrap_or_else(|e| panic!("parse failed for rendered={rendered:?}: {e}"));
        prop_assert_eq!(
            parsed,
            ns,
            "round-trip mismatch — rendered={:?}",
            rendered
        );
    }

    /// Distinct namespace inputs render to distinct strings —
    /// the rendered key is the persistence-layer storage key, so a
    /// collision would conflate cross-tenant or cross-scope rows.
    #[test]
    fn distinct_namespaces_distinct_renders(
        tenant_a in tenant_strategy(),
        tenant_b in tenant_strategy(),
        scope_a in proptest::collection::vec(scope_segment_strategy(), 0..3),
        scope_b in proptest::collection::vec(scope_segment_strategy(), 0..3),
    ) {
        let build = |tenant: &str, scope: &[String]| {
            let mut ns = Namespace::new(TenantId::new(tenant));
            for seg in scope {
                ns = ns.with_scope(seg);
            }
            ns
        };
        let ns_a = build(&tenant_a, &scope_a);
        let ns_b = build(&tenant_b, &scope_b);
        prop_assume!(ns_a != ns_b);
        prop_assert_ne!(
            ns_a.render(),
            ns_b.render(),
            "distinct namespaces must render to distinct keys"
        );
    }
}
