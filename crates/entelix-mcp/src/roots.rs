//! `RootsProvider` + `McpRoot` — client-side answer to the
//! server-initiated `roots/list` request (MCP 2024-11-05 §"Roots").
//!
//! Roots are the operator's contract about where the agent's tools
//! are allowed to operate — typically a workspace directory URI, a
//! repository root, or any other addressable resource boundary the
//! server is expected to honour. The MCP server pulls this list
//! from the client when it needs to scope its own filesystem /
//! workspace operations.
//!
//! ## Why a trait, not a `Vec`
//!
//! Static roots are the common case (the deployment knows them at
//! boot time) and ride through [`StaticRootsProvider`]. Dynamic
//! roots — gathered from a per-request workspace, a database, a
//! sandbox-creation event — need a callback shape, so the
//! manager-facing surface is a trait. Operators who outgrow the
//! static helper implement [`RootsProvider`] directly.
//!
//! ## No `ExecutionContext` parameter
//!
//! Roots are connection-level, not per-request — the MCP server
//! issues `roots/list` outside of any client-driven call, from a
//! background SSE listener. Threading an `ExecutionContext` through
//! the listener would force the listener to choose one (which
//! request's context?) and that choice has no honest answer. The
//! signature mirrors [`entelix_core::auth::CredentialProvider::resolve`]
//! — also context-free, also process-level state.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::McpResult;

/// One workspace / filesystem root the client exposes to the server.
///
/// `uri` is required (per MCP spec); `name` is operator-facing
/// metadata the server may surface in audit logs or its own UI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpRoot {
    /// Absolute URI of the root (e.g. `file:///workspace/repo`,
    /// `vault://team/secrets`). Servers parse this verbatim — the
    /// client makes no claim about the scheme it advertises.
    pub uri: String,
    /// Human-readable label. Optional per spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl McpRoot {
    /// Construct a root with no display name.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            name: None,
        }
    }

    /// Builder-style `name` setter.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Async source-of-truth for the roots a server may discover via
/// `roots/list`. Mirrors the `*Provider` taxonomy
/// (ADR-0010) — async, single-purpose, replaceable.
///
/// Operators wire one provider per server through
/// [`crate::McpServerConfig::with_roots_provider`]. Servers that
/// don't request roots get the provider for free as dormant state
/// (no calls into it).
///
/// `Debug` is required so [`crate::McpServerConfig`]'s own `Debug`
/// derivation surfaces the provider's presence in operator logs;
/// the typical shape is a `#[derive(Debug)]` newtype around a
/// `Vec<McpRoot>` or a closure.
#[async_trait]
pub trait RootsProvider: Send + Sync + 'static + std::fmt::Debug {
    /// Return the current roots. Called once per server-initiated
    /// `roots/list` request — the trait makes no caching claim, so
    /// providers that compute roots dynamically should cache
    /// internally if the cost is non-trivial.
    async fn list_roots(&self) -> McpResult<Vec<McpRoot>>;
}

/// In-memory [`RootsProvider`] holding a fixed slice of roots.
///
/// Use this when the deployment knows its roots at boot — typically
/// for sandbox-rooted agents or single-workspace deployments.
/// Dynamic roots ride a custom [`RootsProvider`] impl.
#[derive(Clone, Debug)]
pub struct StaticRootsProvider {
    roots: Vec<McpRoot>,
}

impl StaticRootsProvider {
    /// Build from any iterable of roots.
    pub fn new(roots: impl IntoIterator<Item = McpRoot>) -> Self {
        Self {
            roots: roots.into_iter().collect(),
        }
    }
}

#[async_trait]
impl RootsProvider for StaticRootsProvider {
    async fn list_roots(&self) -> McpResult<Vec<McpRoot>> {
        Ok(self.roots.clone())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn root_serializes_without_optional_name() {
        let r = McpRoot::new("file:///workspace");
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, r#"{"uri":"file:///workspace"}"#);
    }

    #[test]
    fn root_serializes_with_name_when_set() {
        let r = McpRoot::new("file:///workspace").with_name("repo");
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""name":"repo""#));
    }

    #[tokio::test]
    async fn static_provider_returns_configured_roots() {
        let provider = StaticRootsProvider::new(vec![
            McpRoot::new("file:///a"),
            McpRoot::new("file:///b").with_name("beta"),
        ]);
        let roots = provider.list_roots().await.unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].uri, "file:///a");
        assert_eq!(roots[1].name.as_deref(), Some("beta"));
    }

    #[tokio::test]
    async fn empty_static_provider_returns_empty_list() {
        let provider = StaticRootsProvider::new(Vec::<McpRoot>::new());
        let roots = provider.list_roots().await.unwrap();
        assert!(roots.is_empty());
    }
}
