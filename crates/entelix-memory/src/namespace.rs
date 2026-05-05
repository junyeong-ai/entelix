//! `Namespace` — F2 mitigation. Every memory access is keyed by a
//! `(tenant_id, scope)` pair, with `tenant_id` mandatory at the type
//! level so cross-tenant data leakage is structurally impossible
//! (invariant 11).
//!
//! There is no `Default` impl, no zero-arg constructor, and no
//! "unsafe-tenantless" escape hatch. The `tenant_id` is the
//! validating [`entelix_core::TenantId`] newtype, so a wire payload
//! carrying an empty tenant — including a rendered key starting with
//! `":"` fed to [`Namespace::parse`] — is rejected at construction
//! time rather than producing a silently-tenantless instance whose
//! row-level filter then collapses every tenant onto one key prefix.

use entelix_core::{Error, Result, TenantId};
use serde::{Deserialize, Serialize};

/// Hierarchical key prefix for memory operations.
///
/// `tenant_id` segments out per-customer data. `scope` adds nested
/// dimensions — typically `[agent_id, conversation_id]` for chat-style
/// agents.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Namespace {
    tenant_id: TenantId,
    scope: Vec<String>,
}

impl Namespace {
    /// Build a namespace bound to `tenant_id`. The mandatory argument
    /// is the F2 mitigation in code form — there is no other way to
    /// obtain a `Namespace`. The [`TenantId`] argument has already
    /// passed its validating constructor, so cross-tenant collapse
    /// (`":scope"` instead of `"acme:scope"`) is structurally
    /// impossible at this surface.
    pub const fn new(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            scope: Vec::new(),
        }
    }

    /// Append one scope segment. Builder-style.
    #[must_use]
    pub fn with_scope(mut self, segment: impl Into<String>) -> Self {
        self.scope.push(segment.into());
        self
    }

    /// Borrow the tenant identifier.
    pub const fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    /// Borrow the scope segments in registration order.
    pub fn scope(&self) -> &[String] {
        &self.scope
    }

    /// Render the namespace as a flat `:`-separated key, useful for
    /// backends that take a single string per row. Segments
    /// containing `:` or `\` are escaped (`\:` and `\\`) so two
    /// distinct namespaces can never collide on the rendered key.
    pub fn render(&self) -> String {
        let tenant_id = self.tenant_id.as_str();
        let mut out = String::with_capacity(
            tenant_id.len() + self.scope.iter().map(|s| s.len() + 1).sum::<usize>(),
        );
        push_escaped(&mut out, tenant_id);
        for s in &self.scope {
            out.push(':');
            push_escaped(&mut out, s);
        }
        out
    }

    /// Inverse of [`Self::render`]. Decodes a flat `:`-separated
    /// key back into a typed [`Namespace`], honouring the same
    /// escape rules (`\:` for `:` inside a segment, `\\` for `\`).
    /// Round-trip property: `Namespace::parse(&ns.render()) ==
    /// Ok(ns.clone())` for every well-formed namespace.
    ///
    /// The motivating consumer is the audit channel (invariant
    /// #18): `GraphEvent::MemoryRecall::namespace_key` carries
    /// rendered keys, and operators replaying the log recover the
    /// typed scope (tenant boundary, agent / conversation
    /// dimensions) by parsing.
    ///
    /// Rejects:
    /// - leading `:` (e.g. `":scope"`) — empty tenant component,
    ///   surfaces as [`Error::InvalidRequest`] via the
    ///   [`TenantId::try_from`] validator. Invariant 11 — a
    ///   tenantless `Namespace` would silently collapse every
    ///   tenant onto a single rendered key prefix.
    /// - trailing lone `\` (incomplete escape) — surfaces as
    ///   [`Error::InvalidRequest`].
    /// - `\<x>` for `x` other than `:` or `\` — unknown escape,
    ///   same error variant. Round-tripping a namespace that
    ///   never contained `:` or `\` cannot produce these inputs;
    ///   they only arise from hand-crafted (invalid) keys.
    pub fn parse(rendered: &str) -> Result<Self> {
        let mut segments: Vec<String> = Vec::new();
        let mut current = String::with_capacity(rendered.len());
        let mut chars = rendered.chars();
        while let Some(ch) = chars.next() {
            match ch {
                ':' => {
                    segments.push(std::mem::take(&mut current));
                }
                '\\' => match chars.next() {
                    Some(escaped @ (':' | '\\')) => current.push(escaped),
                    Some(other) => {
                        return Err(Error::invalid_request(format!(
                            "Namespace::parse: unknown escape \\{other}"
                        )));
                    }
                    None => {
                        return Err(Error::invalid_request(
                            "Namespace::parse: trailing backslash",
                        ));
                    }
                },
                other => current.push(other),
            }
        }
        segments.push(current);
        // `segments` is non-empty because the loop always pushes a
        // final segment after the last separator (or after exiting
        // the loop with no separators at all on an empty string,
        // in which case `segments == [""]`). The first segment is
        // the tenant component — empty surfaces as
        // `Error::InvalidRequest` via `TenantId::try_from`.
        let tenant_id = TenantId::try_from(segments.remove(0))?;
        Ok(Self {
            tenant_id,
            scope: segments,
        })
    }
}

fn push_escaped(out: &mut String, segment: &str) {
    if !segment.contains([':', '\\']) {
        out.push_str(segment);
        return;
    }
    for ch in segment.chars() {
        match ch {
            ':' | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            other => out.push(other),
        }
    }
}

/// Hierarchical prefix used by [`crate::Store::list_namespaces`].
///
/// Matches every [`Namespace`] whose tenant matches `tenant_id` and
/// whose scope segments start with the prefix's segments. An empty
/// `scope` matches every namespace under `tenant_id`. The shape
/// mirrors `Namespace` so callers compose the two consistently —
/// the tenant boundary is enforced for namespace listings just like
/// it is for individual entries (Invariant 11 / F2).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NamespacePrefix {
    tenant_id: TenantId,
    scope: Vec<String>,
}

impl NamespacePrefix {
    /// Build a prefix matching every namespace under `tenant_id`.
    /// Append scope segments via [`Self::with_scope`] to narrow.
    /// The [`TenantId`] argument has already passed its validating
    /// constructor — invariant 11 cannot be bypassed via the
    /// listing surface.
    #[must_use]
    pub const fn new(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            scope: Vec::new(),
        }
    }

    /// Append one scope segment. Builder-style.
    #[must_use]
    pub fn with_scope(mut self, segment: impl Into<String>) -> Self {
        self.scope.push(segment.into());
        self
    }

    /// Borrow the tenant identifier.
    #[must_use]
    pub const fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    /// Borrow the scope segments in registration order.
    #[must_use]
    pub fn scope(&self) -> &[String] {
        &self.scope
    }

    /// True when `ns` falls under this prefix.
    #[must_use]
    pub fn matches(&self, ns: &Namespace) -> bool {
        ns.tenant_id() == &self.tenant_id && ns.scope().starts_with(&self.scope)
    }
}

impl From<&Namespace> for NamespacePrefix {
    fn from(ns: &Namespace) -> Self {
        Self {
            tenant_id: ns.tenant_id().clone(),
            scope: ns.scope().to_vec(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn t(s: &str) -> TenantId {
        TenantId::new(s)
    }

    #[test]
    fn prefix_matches_subnamespace_and_rejects_other_tenant() {
        let parent = NamespacePrefix::new(t("acme")).with_scope("agent-a");
        assert!(parent.matches(&Namespace::new(t("acme")).with_scope("agent-a")));
        assert!(
            parent.matches(
                &Namespace::new(t("acme"))
                    .with_scope("agent-a")
                    .with_scope("conv-7")
            )
        );
        assert!(!parent.matches(&Namespace::new(t("acme")).with_scope("agent-b")));
        assert!(!parent.matches(&Namespace::new(t("other-tenant")).with_scope("agent-a")));
    }

    #[test]
    fn prefix_with_empty_scope_matches_every_namespace_under_tenant() {
        let p = NamespacePrefix::new(t("acme"));
        assert!(p.matches(&Namespace::new(t("acme"))));
        assert!(p.matches(&Namespace::new(t("acme")).with_scope("any")));
        assert!(!p.matches(&Namespace::new(t("other"))));
    }

    #[test]
    fn from_namespace_round_trips() {
        let ns = Namespace::new(t("acme"))
            .with_scope("agent-a")
            .with_scope("conv-1");
        let prefix = NamespacePrefix::from(&ns);
        assert_eq!(prefix.tenant_id().as_str(), "acme");
        assert_eq!(prefix.scope(), &["agent-a".to_owned(), "conv-1".to_owned()]);
        assert!(prefix.matches(&ns));
    }

    fn round_trip(ns: &Namespace) {
        let rendered = ns.render();
        let parsed = Namespace::parse(&rendered).unwrap();
        assert_eq!(&parsed, ns, "round-trip failed for {rendered:?}");
    }

    #[test]
    fn parse_round_trips_simple_namespace() {
        round_trip(&Namespace::new(t("acme")));
        round_trip(&Namespace::new(t("acme")).with_scope("agent-a"));
        round_trip(
            &Namespace::new(t("acme"))
                .with_scope("agent-a")
                .with_scope("conv-1"),
        );
    }

    #[test]
    fn parse_round_trips_empty_scope_segments() {
        // Empty scope segments are valid (operators sometimes use
        // them as an explicit "no further dimension" marker); empty
        // tenant_id is not (invariant 11 — see
        // `parse_rejects_leading_colon_for_empty_tenant`).
        round_trip(&Namespace::new(t("acme")).with_scope(""));
        round_trip(&Namespace::new(t("acme")).with_scope("a").with_scope(""));
    }

    #[test]
    fn parse_round_trips_segments_with_colon() {
        round_trip(&Namespace::new(t("a:b")).with_scope("c:d"));
        round_trip(&Namespace::new(t("acme")).with_scope("k8s:pod:foo"));
    }

    #[test]
    fn parse_round_trips_segments_with_backslash() {
        round_trip(&Namespace::new(t("a\\b")).with_scope("c\\d"));
        round_trip(&Namespace::new(t("acme")).with_scope("\\\\\\:"));
    }

    #[test]
    fn parse_extracts_tenant_and_scope_from_simple_input() {
        let ns = Namespace::parse("acme:agent-a:conv-1").unwrap();
        assert_eq!(ns.tenant_id().as_str(), "acme");
        assert_eq!(ns.scope(), &["agent-a".to_owned(), "conv-1".to_owned()]);
    }

    #[test]
    fn parse_decodes_escapes() {
        let ns = Namespace::parse("a\\:b:c\\\\d").unwrap();
        assert_eq!(ns.tenant_id().as_str(), "a:b");
        assert_eq!(ns.scope(), &["c\\d".to_owned()]);
    }

    #[test]
    fn parse_rejects_trailing_backslash() {
        let err = Namespace::parse("acme\\").unwrap_err();
        assert!(format!("{err}").contains("trailing backslash"));
    }

    #[test]
    fn parse_rejects_unknown_escape() {
        let err = Namespace::parse("acme\\x").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown escape"), "got {msg}");
    }

    #[test]
    fn parse_rejects_leading_colon_for_empty_tenant() {
        // Invariant 11 — `":scope"` would silently collapse every
        // tenant onto a single rendered key prefix. The
        // `TenantId::try_from` validator surfaces it as
        // `Error::InvalidRequest`.
        let err = Namespace::parse(":scope").unwrap_err();
        let msg = format!("{err}");
        assert!(matches!(err, Error::InvalidRequest(_)), "got {err:?}");
        assert!(msg.contains("tenant_id must be non-empty"), "got {msg}");
    }

    #[test]
    fn parse_rejects_empty_string_for_empty_tenant() {
        // Edge — an entirely empty rendered key produces a single
        // empty segment that maps to an empty tenant. Same
        // mitigation, same error.
        let err = Namespace::parse("").unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)), "got {err:?}");
    }

    #[test]
    fn deserialize_rejects_empty_tenant_in_wire_payload() {
        // Invariant 11 — a `Namespace` materialised from an
        // untrusted JSON payload runs the validating constructor;
        // an empty tenant cannot be hydrated.
        let err = serde_json::from_str::<Namespace>(r#"{"tenant_id":"","scope":["agent-a"]}"#)
            .unwrap_err();
        assert!(
            err.to_string().contains("tenant_id must be non-empty"),
            "got {err}"
        );
    }
}
