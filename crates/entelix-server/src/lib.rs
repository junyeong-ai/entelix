//! # entelix-server
//!
//! Production HTTP surface for entelix agents, built on `axum`. Mounts
//! a single [`entelix_runnable::Runnable`]-shaped agent under
//! `/v1/threads/{thread_id}/...` with synchronous, streaming, and
//! resume endpoints. Multi-tenant via a configurable header that the
//! middleware forwards into `ExecutionContext::with_tenant_id`.
//!
//! ## API
//!
//! | Method | Path                                    | Purpose |
//! |--------|-----------------------------------------|---------|
//! | POST   | `/v1/threads/{thread_id}/runs`          | sync invoke; JSON request → JSON response |
//! | GET    | `/v1/threads/{thread_id}/stream`        | SSE; 5-mode (`?mode=values\|updates\|messages\|debug\|events`) |
//! | POST   | `/v1/threads/{thread_id}/wake`          | resume from latest checkpoint with `Command::{Resume,Update,GoTo}` |
//! | GET    | `/v1/health`                            | liveness probe |
//!
//! ## Wiring
//!
//! ```ignore
//! use std::sync::Arc;
//! use axum::serve;
//! use entelix_server::{AgentRouterBuilder, DEFAULT_TENANT_HEADER};
//!
//! let router = AgentRouterBuilder::new(my_compiled_graph)
//!     .with_checkpointer(Arc::clone(&checkpointer))
//!     .with_tenant_header(DEFAULT_TENANT_HEADER)
//!     .build()?;
//!
//! let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
//! serve(listener, router).await?;
//! ```
//!
//! ## Tenant mode
//!
//! - **Single-tenant** (default — omit `with_tenant_header`): every
//!   request runs under [`entelix_core::DEFAULT_TENANT_ID`].
//! - **Multi-tenant strict** (call `with_tenant_header(name)`): every
//!   request MUST carry the named header; missing-header requests
//!   reject with `400 Bad Request` and a typed
//!   [`ServerError::MissingTenantHeader`]. There is no silent
//!   fall-through to the default tenant.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/entelix-server/0.1.0")]
#![deny(missing_docs)]
#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_const_for_fn,
    clippy::redundant_pub_crate
)]

mod error;
mod handlers;
mod router;

pub use error::{BuildError, BuildResult, ServerError, ServerResult};
pub use router::{AgentRouterBuilder, DEFAULT_TENANT_HEADER, TenantMode};
