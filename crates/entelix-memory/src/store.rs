//! `Store<V>` trait — namespace-scoped key/value storage that
//! survives across threads (Tier 3 in the 3-tier state model).
//!
//! In-process default: [`InMemoryStore<V>`]. Postgres / Redis backed
//! `Store` impls live in `entelix-persistence`.
//!
//! ## Production primitives
//!
//! - [`PutOptions`] — declarative per-write knobs. The only field
//!   today is `ttl`; future additions ride on `#[non_exhaustive]`
//!   without touching call sites.
//! - [`Store::put`] is the simple hot path; [`Store::put_with_options`]
//!   is the configurable form. `put` has a default impl that
//!   delegates to `put_with_options(PutOptions::default())`.
//! - [`Store::list_namespaces`] returns every [`Namespace`] under a
//!   [`NamespacePrefix`] — the F2 / Invariant-11 boundary stays
//!   structural for hierarchical traversal as well as point lookups.
//! - [`Store::evict_expired`] is a default-`Ok(0)` hook that backends
//!   override when they own a TTL sweeper. Operators run it on a
//!   timer to bound store growth.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};
use parking_lot::Mutex;

use crate::namespace::{Namespace, NamespacePrefix};

/// Per-write knobs the operator may attach when calling
/// [`Store::put_with_options`]. `Default::default()` corresponds to
/// the simple [`Store::put`] path: no TTL, no extra metadata.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct PutOptions {
    /// Time-to-live for the entry. `None` = no expiry. Backends
    /// without native TTL support emit the value with no expiry and
    /// surface the request through [`Store::evict_expired`] sweeps.
    pub ttl: Option<Duration>,
}

impl PutOptions {
    /// Attach a TTL to this put. Builder-style.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }
}

/// Persistent (or in-memory) key/value store, scoped by [`Namespace`].
///
/// Every method takes [`ExecutionContext`] so remote backends can
/// honour caller-side cancellation and deadlines (invariant
/// "cancellation propagation"). In-memory impls accept the parameter
/// for trait uniformity and otherwise ignore it.
#[async_trait]
pub trait Store<V>: Send + Sync + 'static
where
    V: Clone + Send + Sync + 'static,
{
    /// Insert or replace `value` at `(ns, key)` with the supplied
    /// per-write options (TTL, future fields). This is the only
    /// required write — [`Self::put`] is a thin convenience that
    /// delegates here.
    async fn put_with_options(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        key: &str,
        value: V,
        options: PutOptions,
    ) -> Result<()>;

    /// Insert or replace `value` at `(ns, key)` with default options
    /// (no TTL). The default impl delegates to
    /// [`Self::put_with_options`] — backends only need to provide one.
    async fn put(&self, ctx: &ExecutionContext, ns: &Namespace, key: &str, value: V) -> Result<()> {
        self.put_with_options(ctx, ns, key, value, PutOptions::default())
            .await
    }

    /// Look up `(ns, key)`. Returns `None` if absent or expired.
    async fn get(&self, ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<Option<V>>;

    /// Delete `(ns, key)`. Idempotent — deleting an absent key
    /// succeeds.
    async fn delete(&self, ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<()>;

    /// List keys under `ns` whose names start with `prefix` (or all
    /// keys if `prefix` is `None`). Order is unspecified.
    async fn list(
        &self,
        ctx: &ExecutionContext,
        ns: &Namespace,
        prefix: Option<&str>,
    ) -> Result<Vec<String>>;

    /// List every [`Namespace`] under `prefix` that holds at least
    /// one entry. The default impl returns an empty list — backends
    /// override when they can enumerate cheaply (Postgres index
    /// scan, Redis `SCAN`). Order is unspecified.
    ///
    /// Useful for "list all conversations under agent-X" or
    /// admin tooling that audits per-tenant storage.
    async fn list_namespaces(
        &self,
        _ctx: &ExecutionContext,
        _prefix: &NamespacePrefix,
    ) -> Result<Vec<Namespace>> {
        Ok(Vec::new())
    }

    /// Sweep expired entries. Returns the number of rows removed.
    /// Default impl returns `Ok(0)` — only backends that natively
    /// track TTL implement this. Operators schedule it on a timer
    /// (or trigger from cron / periodic graph) to bound store
    /// growth in deployments where the store does not auto-expire
    /// (e.g. plain `put` into Postgres without a TTL trigger).
    async fn evict_expired(&self, _ctx: &ExecutionContext) -> Result<usize> {
        Ok(0)
    }
}

/// In-process `Store<V>` backed by a `HashMap` keyed by
/// `(rendered_namespace, key)`. Cheap to clone — internal state is
/// `Arc<Mutex<...>>`-shared.
///
/// TTL is honoured: entries written via
/// [`Store::put_with_options`] with a non-`None` `ttl` are dropped
/// from `get` / `list` results once their absolute expiry passes.
/// The sweep ([`Store::evict_expired`]) cleans the map structure;
/// callers may run it from a periodic graph if memory pressure
/// matters.
pub struct InMemoryStore<V>
where
    V: Clone + Send + Sync + 'static,
{
    inner: Arc<Mutex<EntryMap<V>>>,
}

type EntryMap<V> = HashMap<(String, String), Entry<V>>;

struct Entry<V> {
    value: V,
    expires_at: Option<Instant>,
}

impl<V> InMemoryStore<V>
where
    V: Clone + Send + Sync + 'static,
{
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Total entry count across all namespaces, including not-yet-
    /// swept-but-expired ones. Useful for tests; production callers
    /// should run [`Store::evict_expired`] first if they care about
    /// the live count.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.inner.lock().len()
    }
}

impl<V> Default for InMemoryStore<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Clone for InMemoryStore<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[async_trait]
impl<V> Store<V> for InMemoryStore<V>
where
    V: Clone + Send + Sync + 'static,
{
    async fn put_with_options(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        key: &str,
        value: V,
        options: PutOptions,
    ) -> Result<()> {
        let composite = (ns.render(), key.to_owned());
        let expires_at = options.ttl.map(|d| Instant::now() + d);
        {
            let mut guard = self.inner.lock();
            guard.insert(composite, Entry { value, expires_at });
        }
        Ok(())
    }

    async fn get(&self, _ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<Option<V>> {
        let composite = (ns.render(), key.to_owned());
        let now = Instant::now();
        let result = {
            let guard = self.inner.lock();
            guard
                .get(&composite)
                .filter(|entry| entry.expires_at.is_none_or(|exp| exp > now))
                .map(|entry| entry.value.clone())
        };
        Ok(result)
    }

    async fn delete(&self, _ctx: &ExecutionContext, ns: &Namespace, key: &str) -> Result<()> {
        let composite = (ns.render(), key.to_owned());
        {
            let mut guard = self.inner.lock();
            guard.remove(&composite);
        }
        Ok(())
    }

    async fn list(
        &self,
        _ctx: &ExecutionContext,
        ns: &Namespace,
        prefix: Option<&str>,
    ) -> Result<Vec<String>> {
        let ns_key = ns.render();
        let now = Instant::now();
        let out = {
            let guard = self.inner.lock();
            guard
                .iter()
                .filter(|((n, _), entry)| {
                    n == &ns_key && entry.expires_at.is_none_or(|exp| exp > now)
                })
                .filter(|((_, k), _)| prefix.is_none_or(|p| k.starts_with(p)))
                .map(|((_, k), _)| k.clone())
                .collect::<Vec<_>>()
        };
        Ok(out)
    }

    async fn list_namespaces(
        &self,
        _ctx: &ExecutionContext,
        prefix: &NamespacePrefix,
    ) -> Result<Vec<Namespace>> {
        let prefix_render = render_prefix(prefix);
        let now = Instant::now();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        {
            let guard = self.inner.lock();
            for ((rendered_ns, _), entry) in guard.iter() {
                if entry.expires_at.is_some_and(|exp| exp <= now) {
                    continue;
                }
                if rendered_ns == &prefix_render
                    || rendered_ns.starts_with(&format!("{prefix_render}:"))
                {
                    seen.insert(rendered_ns.clone());
                }
            }
        }
        // `Namespace::parse` recovers the typed `(tenant_id, scope)`
        // tuple from the rendered key — the structural identity is
        // preserved through the round-trip render → store → list →
        // parse. The trait contract ("every distinct Namespace
        // under prefix") is honoured as written rather than
        // approximated with a synthetic clone of the prefix.
        seen.into_iter().map(|key| Namespace::parse(&key)).collect()
    }

    async fn evict_expired(&self, _ctx: &ExecutionContext) -> Result<usize> {
        let now = Instant::now();
        let removed = {
            let mut guard = self.inner.lock();
            let before = guard.len();
            guard.retain(|_, entry| entry.expires_at.is_none_or(|exp| exp > now));
            before - guard.len()
        };
        Ok(removed)
    }
}

fn render_prefix(prefix: &NamespacePrefix) -> String {
    // Mirror Namespace::render layout so InMemoryStore prefix matches
    // are textually consistent with stored namespace keys.
    let mut tmp = Namespace::new(prefix.tenant_id().clone());
    for s in prefix.scope() {
        tmp = tmp.with_scope(s.clone());
    }
    tmp.render()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use entelix_core::TenantId;

    fn ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn ns() -> Namespace {
        Namespace::new(TenantId::new("acme")).with_scope("agent-a")
    }

    #[tokio::test]
    async fn put_then_get_round_trips() {
        let store: InMemoryStore<String> = InMemoryStore::new();
        store.put(&ctx(), &ns(), "k", "v".into()).await.unwrap();
        let got = store.get(&ctx(), &ns(), "k").await.unwrap();
        assert_eq!(got.as_deref(), Some("v"));
    }

    #[tokio::test]
    async fn ttl_expires_on_get() {
        let store: InMemoryStore<String> = InMemoryStore::new();
        store
            .put_with_options(
                &ctx(),
                &ns(),
                "k",
                "v".into(),
                PutOptions::default().with_ttl(Duration::from_millis(20)),
            )
            .await
            .unwrap();
        // Live before expiry.
        assert!(store.get(&ctx(), &ns(), "k").await.unwrap().is_some());
        tokio::time::sleep(Duration::from_millis(40)).await;
        // Expired — get returns None even though sweep has not run.
        assert!(store.get(&ctx(), &ns(), "k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn evict_expired_returns_count_and_drops_rows() {
        let store: InMemoryStore<String> = InMemoryStore::new();
        // One TTL row (will expire) + one no-TTL row (survives).
        store
            .put_with_options(
                &ctx(),
                &ns(),
                "doomed",
                "v".into(),
                PutOptions::default().with_ttl(Duration::from_millis(10)),
            )
            .await
            .unwrap();
        store.put(&ctx(), &ns(), "alive", "v".into()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        let removed = store.evict_expired(&ctx()).await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.total_entries(), 1);
    }

    #[tokio::test]
    async fn list_namespaces_finds_subscopes_under_prefix() {
        let store: InMemoryStore<String> = InMemoryStore::new();
        let ns_a = Namespace::new(TenantId::new("acme")).with_scope("agent-a");
        let ns_b = Namespace::new(TenantId::new("acme"))
            .with_scope("agent-a")
            .with_scope("conv-1");
        let ns_other = Namespace::new(TenantId::new("acme")).with_scope("agent-b");
        store.put(&ctx(), &ns_a, "k", "v".into()).await.unwrap();
        store.put(&ctx(), &ns_b, "k", "v".into()).await.unwrap();
        store.put(&ctx(), &ns_other, "k", "v".into()).await.unwrap();
        let prefix = NamespacePrefix::new(TenantId::new("acme")).with_scope("agent-a");
        let found = store.list_namespaces(&ctx(), &prefix).await.unwrap();
        // ns_a + ns_b match; ns_other does not.
        assert_eq!(found.len(), 2);
        // Returned namespaces structurally match the originals, not
        // a prefix-shape clone — the round-trip render → parse
        // recovers the typed scope.
        let mut got: Vec<Namespace> = found;
        got.sort_by_key(|x| x.scope().len());
        assert_eq!(got[0], ns_a);
        assert_eq!(got[1], ns_b);
    }

    #[tokio::test]
    async fn list_namespaces_recovers_escaped_segments() {
        let store: InMemoryStore<String> = InMemoryStore::new();
        let ns_colon = Namespace::new(TenantId::new("acme"))
            .with_scope("agent-a")
            .with_scope("k8s:pod:foo");
        store.put(&ctx(), &ns_colon, "k", "v".into()).await.unwrap();
        let prefix = NamespacePrefix::new(TenantId::new("acme")).with_scope("agent-a");
        let found = store.list_namespaces(&ctx(), &prefix).await.unwrap();
        assert_eq!(found.len(), 1);
        // The `:`-bearing scope segment survives the render → store
        // → list → parse round-trip — escapes are not silently
        // chopped at substring boundaries.
        assert_eq!(found[0], ns_colon);
    }

    #[tokio::test]
    async fn delete_then_get_returns_none() {
        let store: InMemoryStore<String> = InMemoryStore::new();
        store.put(&ctx(), &ns(), "k", "v".into()).await.unwrap();
        store.delete(&ctx(), &ns(), "k").await.unwrap();
        assert!(store.get(&ctx(), &ns(), "k").await.unwrap().is_none());
    }
}
