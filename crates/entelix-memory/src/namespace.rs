//! `Namespace` — F2 mitigation. Every memory access is keyed by a
//! `(tenant_id, scope)` pair, with `tenant_id` mandatory at the type
//! level so cross-tenant data leakage is structurally impossible
//! (invariant 11).
//!
//! There is no `Default` impl, no zero-arg constructor, and no
//! "unsafe-tenantless" escape hatch. Future persistent backends are
//! expected to enforce row-level filtering by `tenant_id`.

use entelix_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Hierarchical key prefix for memory operations.
///
/// `tenant_id` segments out per-customer data. `scope` adds nested
/// dimensions — typically `[agent_id, conversation_id]` for chat-style
/// agents.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Namespace {
    tenant_id: String,
    scope: Vec<String>,
}

impl Namespace {
    /// Build a namespace bound to `tenant_id`. The mandatory argument is
    /// the F2 mitigation in code form — there is no other way to obtain a
    /// `Namespace`.
    ///
    /// # Panics
    ///
    /// Panics when `tenant_id` is empty. Invariant 11 demands a
    /// *non-empty* tenant scope; an empty string would silently
    /// collapse every tenant onto a single rendered key prefix
    /// (`":scope"` instead of `"acme:scope"`), defeating the
    /// row-level isolation backends key off `tenant_id()`.
    /// `ExecutionContext::tenant_id` defaults to a non-empty
    /// `"default"`, so production call sites already pass a real
    /// value — this assertion is programmer-error grade for the
    /// rare hand-crafted bad input. Mirrors
    /// [`entelix_core::ThreadKey::new`].
    pub fn new(tenant_id: impl Into<String>) -> Self {
        let tenant_id = tenant_id.into();
        assert!(
            !tenant_id.is_empty(),
            "Namespace::new: tenant_id must be non-empty (invariant 11)"
        );
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
    pub fn tenant_id(&self) -> &str {
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
        let mut out = String::with_capacity(
            self.tenant_id.len() + self.scope.iter().map(|s| s.len() + 1).sum::<usize>(),
        );
        push_escaped(&mut out, &self.tenant_id);
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
        // in which case `segments == [""]`).
        let tenant_id = segments.remove(0);
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
    tenant_id: String,
    scope: Vec<String>,
}

impl NamespacePrefix {
    /// Build a prefix matching every namespace under `tenant_id`.
    /// Append scope segments via [`Self::with_scope`] to narrow.
    ///
    /// # Panics
    ///
    /// Panics when `tenant_id` is empty — invariant 11 demands a
    /// non-empty tenant scope, mirroring [`Namespace::new`].
    #[must_use]
    pub fn new(tenant_id: impl Into<String>) -> Self {
        let tenant_id = tenant_id.into();
        assert!(
            !tenant_id.is_empty(),
            "NamespacePrefix::new: tenant_id must be non-empty (invariant 11)"
        );
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
    pub fn tenant_id(&self) -> &str {
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
        ns.tenant_id() == self.tenant_id && ns.scope().starts_with(&self.scope)
    }
}

impl From<&Namespace> for NamespacePrefix {
    fn from(ns: &Namespace) -> Self {
        Self {
            tenant_id: ns.tenant_id().to_owned(),
            scope: ns.scope().to_vec(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn prefix_matches_subnamespace_and_rejects_other_tenant() {
        let parent = NamespacePrefix::new("acme").with_scope("agent-a");
        assert!(parent.matches(&Namespace::new("acme").with_scope("agent-a")));
        assert!(
            parent.matches(
                &Namespace::new("acme")
                    .with_scope("agent-a")
                    .with_scope("conv-7")
            )
        );
        assert!(!parent.matches(&Namespace::new("acme").with_scope("agent-b")));
        assert!(!parent.matches(&Namespace::new("other-tenant").with_scope("agent-a")));
    }

    #[test]
    fn prefix_with_empty_scope_matches_every_namespace_under_tenant() {
        let p = NamespacePrefix::new("acme");
        assert!(p.matches(&Namespace::new("acme")));
        assert!(p.matches(&Namespace::new("acme").with_scope("any")));
        assert!(!p.matches(&Namespace::new("other")));
    }

    #[test]
    fn from_namespace_round_trips() {
        let ns = Namespace::new("acme")
            .with_scope("agent-a")
            .with_scope("conv-1");
        let prefix = NamespacePrefix::from(&ns);
        assert_eq!(prefix.tenant_id(), "acme");
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
        round_trip(&Namespace::new("acme"));
        round_trip(&Namespace::new("acme").with_scope("agent-a"));
        round_trip(
            &Namespace::new("acme")
                .with_scope("agent-a")
                .with_scope("conv-1"),
        );
    }

    #[test]
    fn parse_round_trips_empty_scope_segments() {
        // Empty scope segments are valid (operators sometimes use
        // them as an explicit "no further dimension" marker); empty
        // tenant_id is not (invariant 11 — see
        // `namespace_new_panics_on_empty_tenant_id`).
        round_trip(&Namespace::new("acme").with_scope(""));
        round_trip(&Namespace::new("acme").with_scope("a").with_scope(""));
    }

    #[test]
    fn parse_round_trips_segments_with_colon() {
        round_trip(&Namespace::new("a:b").with_scope("c:d"));
        round_trip(&Namespace::new("acme").with_scope("k8s:pod:foo"));
    }

    #[test]
    fn parse_round_trips_segments_with_backslash() {
        round_trip(&Namespace::new("a\\b").with_scope("c\\d"));
        round_trip(&Namespace::new("acme").with_scope("\\\\\\:"));
    }

    #[test]
    fn parse_extracts_tenant_and_scope_from_simple_input() {
        let ns = Namespace::parse("acme:agent-a:conv-1").unwrap();
        assert_eq!(ns.tenant_id(), "acme");
        assert_eq!(ns.scope(), &["agent-a".to_owned(), "conv-1".to_owned()]);
    }

    #[test]
    fn parse_decodes_escapes() {
        let ns = Namespace::parse("a\\:b:c\\\\d").unwrap();
        assert_eq!(ns.tenant_id(), "a:b");
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
    #[should_panic(expected = "tenant_id must be non-empty")]
    fn namespace_new_panics_on_empty_tenant_id() {
        // Invariant 11 / F2 — empty tenant_id would silently
        // collapse every tenant onto a single rendered key prefix
        // (e.g. ":scope" rather than "acme:scope"), defeating the
        // row-level isolation backends key off `tenant_id()`.
        // Mirrors `entelix_core::ThreadKey::new`'s guard.
        let _ = Namespace::new("");
    }

    #[test]
    #[should_panic(expected = "tenant_id must be non-empty")]
    fn namespace_prefix_new_panics_on_empty_tenant_id() {
        // Same boundary as Namespace::new — the prefix surface is
        // the listing-side mirror and must enforce the same
        // non-empty guard so invariant 11 cannot be bypassed via
        // `Store::list_namespaces` either.
        let _ = NamespacePrefix::new("");
    }
}
