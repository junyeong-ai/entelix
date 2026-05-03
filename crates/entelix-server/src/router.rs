//! `AgentRouterBuilder` — fluent builder that mounts a single
//! [`Runnable<S, S>`] (and an optional [`Checkpointer<S>`]) under the
//! standard `/v1/threads/{thread_id}/...` HTTP surface and returns a
//! ready-to-serve [`axum::Router`].
//!
//! ## Tenant routing
//!
//! A router carries a [`TenantMode`] that the request middleware
//! consults on every dispatch:
//!
//! - **[`TenantMode::Default`]** (default): every request runs under
//!   [`entelix_core::DEFAULT_TENANT_ID`]. The `Namespace { tenant_id }`
//!   invariant 11 enforcement still fires (no row can ever be tagged
//!   "no tenant"), but only one tenant is ever projected.
//! - **[`TenantMode::RequiredHeader`]** (B2B SaaS): every request
//!   MUST carry the named header; missing-header requests reject with
//!   `400 Bad Request` and a typed
//!   [`crate::ServerError::MissingTenantHeader`]. There is no silent
//!   fall-through to the default tenant — once strict mode is opted
//!   into, ambiguity is structurally impossible.
//!
//! Choose the mode at builder time via either
//! [`AgentRouterBuilder::with_tenant_header`] (the convenience method
//! that wraps the most common multi-tenant case) or
//! [`AgentRouterBuilder::with_tenant_mode`] (the canonical method when
//! the mode value is computed elsewhere). The router does not
//! auto-detect; the type signature does.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use http::HeaderName;
use serde::{Serialize, de::DeserializeOwned};

use entelix_graph::Checkpointer;
use entelix_runnable::Runnable;

use crate::error::{BuildError, BuildResult};
use crate::handlers;

/// Header name conventionally used when the operator opts into
/// multi-tenant mode. Pass to [`AgentRouterBuilder::with_tenant_header`].
pub const DEFAULT_TENANT_HEADER: &str = "x-tenant-id";

/// How the router resolves the `tenant_id` for an incoming request.
///
/// `#[non_exhaustive]` so future modes (claim extraction from a JWT,
/// header-with-fallback, etc.) can be added without breaking match
/// arms in operator code. Today the two variants exhaust the
/// production patterns the SDK ships behaviour for.
///
/// Each variant uses struct-style fields (not tuple) so a future
/// extension that grows a sibling parameter — for example
/// `RequiredHeader { header, fallback_to_default: bool }` — can land
/// without breaking match arms that already destructured the variant.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum TenantMode {
    /// Single-tenant — every request runs under
    /// [`entelix_core::DEFAULT_TENANT_ID`]. Suitable for solo-tenant
    /// deployments and local development.
    #[default]
    Default,
    /// Multi-tenant strict — extract `tenant_id` from the named
    /// HTTP header on every request. Missing or empty header rejects
    /// with `400 Bad Request` and a typed
    /// [`crate::ServerError::MissingTenantHeader`]. There is no
    /// silent fall-through.
    RequiredHeader {
        /// Header name to extract `tenant_id` from on every request.
        header: HeaderName,
    },
}

/// Internal axum state shared across handlers via `axum::State`.
pub(crate) struct AgentRouterState<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    runnable: Arc<dyn Runnable<S, S>>,
    checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    tenant_mode: TenantMode,
}

impl<S> AgentRouterState<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    pub(crate) fn runnable(&self) -> &Arc<dyn Runnable<S, S>> {
        &self.runnable
    }

    pub(crate) fn checkpointer(&self) -> Option<&Arc<dyn Checkpointer<S>>> {
        self.checkpointer.as_ref()
    }

    pub(crate) fn tenant_mode(&self) -> &TenantMode {
        &self.tenant_mode
    }
}

/// Fluent builder that produces an [`axum::Router`] mounted at `/v1`.
///
/// Add your own middleware (CORS, tracing, rate-limit, …) to the
/// returned router before passing it to `axum::serve`.
pub struct AgentRouterBuilder<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    runnable: Arc<dyn Runnable<S, S>>,
    checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    tenant_mode: TenantMode,
    invalid_tenant_header: Option<String>,
}

impl<S> AgentRouterBuilder<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    /// Start a builder around the supplied runnable. Defaults to
    /// [`TenantMode::Default`] — call [`Self::with_tenant_header`]
    /// (or [`Self::with_tenant_mode`]) to opt into multi-tenant.
    pub fn new<R>(runnable: R) -> Self
    where
        R: Runnable<S, S> + 'static,
    {
        Self {
            runnable: Arc::new(runnable),
            checkpointer: None,
            tenant_mode: TenantMode::Default,
            invalid_tenant_header: None,
        }
    }

    /// Attach a checkpointer for the `/wake` endpoint. Without one,
    /// `/wake` returns `503 Service Unavailable`.
    #[must_use]
    pub fn with_checkpointer(mut self, cp: Arc<dyn Checkpointer<S>>) -> Self {
        self.checkpointer = Some(cp);
        self
    }

    /// Set the [`TenantMode`] explicitly. The canonical method when
    /// the mode value is computed elsewhere (config layer, test
    /// fixture, future variant). Most callers reach for the
    /// convenience wrapper [`Self::with_tenant_header`].
    #[must_use]
    pub fn with_tenant_mode(mut self, mode: TenantMode) -> Self {
        self.tenant_mode = mode;
        self.invalid_tenant_header = None;
        self
    }

    /// Convenience — opt into [`TenantMode::RequiredHeader`]
    /// without naming the variant. Once registered, requests missing
    /// the header fail at the middleware boundary with `400 Bad
    /// Request` + [`crate::ServerError::MissingTenantHeader`] —
    /// there is no silent fall-through to a default tenant.
    ///
    /// The conventional header name is [`DEFAULT_TENANT_HEADER`]
    /// (`x-tenant-id`); pass it explicitly to keep the deployment's
    /// intent visible at the builder call site.
    ///
    /// Header-name validation is deferred to [`Self::build`] so the
    /// builder chain stays infallible (mirrors the Tower middleware
    /// idiom).
    #[must_use]
    pub fn with_tenant_header(mut self, name: impl AsRef<[u8]>) -> Self {
        let bytes = name.as_ref();
        if let Ok(header) = HeaderName::from_bytes(bytes) {
            self.tenant_mode = TenantMode::RequiredHeader { header };
            self.invalid_tenant_header = None;
        } else {
            self.tenant_mode = TenantMode::Default;
            self.invalid_tenant_header = Some(String::from_utf8_lossy(bytes).into_owned());
        }
        self
    }

    /// Finalize. Surfaces deferred validation (see
    /// [`Self::with_tenant_header`]) as a typed [`BuildError`].
    pub fn build(self) -> BuildResult<Router> {
        if let Some(name) = self.invalid_tenant_header {
            return Err(BuildError::InvalidTenantHeader { name });
        }
        let state = Arc::new(AgentRouterState {
            runnable: self.runnable,
            checkpointer: self.checkpointer,
            tenant_mode: self.tenant_mode,
        });
        Ok(Router::new()
            .route(
                "/v1/threads/{thread_id}/runs",
                post(handlers::run_sync::<S>),
            )
            .route(
                "/v1/threads/{thread_id}/stream",
                get(handlers::run_stream::<S>),
            )
            .route("/v1/threads/{thread_id}/wake", post(handlers::wake::<S>))
            .route("/v1/health", get(handlers::health))
            .with_state(state))
    }
}
