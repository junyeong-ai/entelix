//! `DocumentLoader` — async source-side trait.
//!
//! Loaders stream `Document`s so ingestion pipelines stay
//! memory-bounded over arbitrarily large corpora. A backend that
//! produces millions of records (S3 bucket walk, Confluence space
//! enumeration) must not buffer the whole catalogue — the
//! `BoxStream` return forces incremental yield.
//!
//! Loaders are *async* (network IO, paginated APIs) but a
//! `DocumentLoader` impl that wraps a sync source pre-loads the
//! payload eagerly inside `load(...)` and yields synchronously
//! from there.

use std::pin::Pin;

use async_trait::async_trait;
use entelix_core::{ExecutionContext, Result};
use futures::Stream;

use crate::document::Document;

/// Boxed stream type alias for documents produced by a
/// [`DocumentLoader`]. Items are `Result` so a partial-success
/// stream can yield successful documents while reporting per-item
/// errors — a single mid-walk failure does not abort the whole
/// ingestion run.
pub type DocumentStream<'a> = Pin<Box<dyn Stream<Item = Result<Document>> + Send + 'a>>;

/// Source-side trait the ingestion pipeline pulls documents from.
///
/// Implementations cover network sources (HTTP, REST APIs, GraphQL),
/// SaaS connectors (Notion, Confluence, GDrive, Slack), object
/// stores (S3, GCS, Azure Blob), and filesystem walkers (the latter
/// behind invariant 9 sandbox exemption — typically in a coding-agent
/// companion crate, not this surface).
///
/// The cancellation token on the supplied
/// [`ExecutionContext`](entelix_core::ExecutionContext) gates the
/// walk; long-running loaders poll `ctx.is_cancelled()` between
/// pages so an abandoned ingestion run releases backend resources
/// promptly.
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    /// Stable loader identifier — surfaces on every produced
    /// document's [`Source::loader`](crate::Source::loader) field
    /// and in audit dashboards. `"web"`, `"s3"`, `"notion"`, etc.
    fn name(&self) -> &'static str;

    /// Open the loader and stream documents. The returned stream
    /// is bounded by the loader's view of the source — pagination,
    /// rate limits, and per-item errors all surface through the
    /// stream's items, not the outer `Result`.
    ///
    /// The outer `Result` is reserved for *opening* failures
    /// (auth rejection, source unreachable, configuration error)
    /// — once the stream starts, per-document failures land on
    /// the `Result<Document>` items so the pipeline can decide
    /// whether to skip or abort.
    async fn load<'a>(&'a self, ctx: &'a ExecutionContext) -> Result<DocumentStream<'a>>;
}
