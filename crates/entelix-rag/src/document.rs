//! `Document` — the unit RAG pipelines move around.
//!
//! Distinct from [`entelix_memory::Document`]: the latter is a
//! *retrieval result* (carries a similarity score from a vector
//! store); this is the *ingestion / processing* shape (carries
//! provenance under [`Source`] and split-history under
//! [`Lineage`]). Splitters, chunkers, and ingestion pipelines move
//! `entelix_rag::Document` values; the final write-to-vector-store
//! step converts to `entelix_memory::Document`.
//!
//! The two shapes are deliberately uncoupled — retrieval has no
//! need for `Source` / `Lineage` (that information lives in
//! `metadata` on persistent storage), and ingestion has no
//! similarity score until retrieval happens.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use entelix_memory::Namespace;
use serde::{Deserialize, Serialize};

/// Stable identifier for a `Document` within its [`Namespace`].
/// Loaders mint these from the source's natural id (S3 object key,
/// Notion page id, file path); splitters derive child ids by
/// suffixing the parent id with `:<chunk_index>`.
///
/// Held as `Arc<str>` so cloning a `Document` (and the chunk tree
/// a splitter produces) is an atomic refcount bump rather than a
/// fresh string allocation per chunk. Mirrors the
/// [`entelix_core::TenantId`] interning pattern.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentId(Arc<str>);

impl DocumentId {
    /// Build an id from any string-like value. Empty ids are
    /// rejected at construction time — silent mismatch with stored
    /// records on retrieval is a class of bug not worth admitting.
    ///
    /// # Panics
    ///
    /// Panics when `id` is empty after `Into::into`. Empty
    /// document ids are a programmer error, not a runtime
    /// condition the pipeline should silently paper over.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        let s: String = id.into();
        assert!(!s.is_empty(), "DocumentId must not be empty");
        Self(Arc::from(s))
    }

    /// Borrow the id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DocumentId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for DocumentId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Where a `Document` originated. Survives every split and chunker
/// pass — the leaf chunk knows the source URI of the parent
/// document and which loader produced it.
///
/// `etag` enables idempotent re-ingestion: pipelines compare the
/// loader-reported etag against the stored value and skip
/// reprocessing when unchanged. Loaders that lack a natural etag
/// (in-memory, ephemeral) leave it `None` and the pipeline falls
/// back to time-based deduplication.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Source {
    /// URI of the source — `file:///path`, `s3://bucket/key`,
    /// `notion://workspace/page-id`, etc. Pipelines treat this as
    /// opaque; loaders define the scheme.
    pub uri: String,
    /// Stable identifier of the loader implementation. `"web"`,
    /// `"s3"`, `"notion"`, etc. — used for dashboards and replay
    /// routing, not for behaviour gating. Carried as `String`
    /// (rather than `&'static str`) so persisted documents
    /// reconstruct via serde without forcing every loader name to
    /// be a string literal.
    pub loader: String,
    /// When the loader fetched this document. Pipelines stamp this
    /// at fetch time so re-ingestion ages-out properly.
    pub fetched_at: DateTime<Utc>,
    /// Optional content-version tag the source surfaces (HTTP
    /// `ETag`, S3 object `etag`, Notion `last_edited_time` hash).
    /// `None` when the source has no natural content-versioning
    /// signal.
    pub etag: Option<String>,
}

impl Source {
    /// Build a source descriptor stamped at the current wall
    /// clock. Loaders whose natural fetch time differs from "now"
    /// (replay, batch import) construct via the struct literal.
    #[must_use]
    pub fn now(uri: impl Into<String>, loader: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            loader: loader.into(),
            fetched_at: Utc::now(),
            etag: None,
        }
    }

    /// Builder-style etag attachment.
    #[must_use]
    pub fn with_etag(mut self, etag: impl Into<String>) -> Self {
        self.etag = Some(etag.into());
        self
    }
}

/// Split-history — survives every transformation. A leaf chunk's
/// `Lineage` describes which parent it came from, which split
/// produced it, and which chunkers ran over it. Audit / debug
/// flows reconstruct the path from a retrieval hit back to the
/// ingestion source by walking the lineage chain (parent_id →
/// loader's source URI).
///
/// `None` on the original `Document` produced by a `DocumentLoader`
/// — only chunked descendants carry lineage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Lineage {
    /// Document id of the parent this chunk descends from. Walks
    /// up through every split layer to the original loaded
    /// document.
    pub parent_id: DocumentId,
    /// Position of this chunk within the immediate parent's split
    /// output. Zero-based.
    pub chunk_index: u32,
    /// Total number of chunks the parent produced. Lets retrieval
    /// surfaces show "chunk 3 of 12" provenance.
    pub total_chunks: u32,
    /// Stable identifier of the splitter that produced this
    /// chunk. `"recursive-character"`, `"markdown-structure"`,
    /// etc. — surfaces in audit dashboards.
    pub splitter: String,
    /// Stable identifiers of every chunker that processed this
    /// chunk after the split, in order. `"contextual"`,
    /// `"hyde"`, … — empty when no chunker ran.
    pub chunker_chain: Vec<String>,
}

impl Lineage {
    /// Build the lineage entry a splitter stamps onto each child
    /// chunk. The chunker chain starts empty; downstream chunkers
    /// append themselves via [`Self::push_chunker`].
    #[must_use]
    pub fn from_split(
        parent_id: DocumentId,
        chunk_index: u32,
        total_chunks: u32,
        splitter: impl Into<String>,
    ) -> Self {
        Self {
            parent_id,
            chunk_index,
            total_chunks,
            splitter: splitter.into(),
            chunker_chain: Vec::new(),
        }
    }

    /// Append a chunker identifier to this chunk's chain — called
    /// by `Chunker` impls when they transform a chunk.
    pub fn push_chunker(&mut self, chunker: impl Into<String>) {
        self.chunker_chain.push(chunker.into());
    }
}

/// The unit a RAG pipeline moves around — content plus everything
/// downstream needs to know about where it came from.
///
/// Splitters consume one `Document` and emit several. Chunkers
/// consume a sequence and emit a transformed sequence (typically
/// the same length, with mutated `content` or `metadata`).
/// Loaders produce them; ingestion pipelines consume them.
///
/// `metadata` is operator-defined free-form JSON for filtering at
/// the vector-store layer — explicit fields above (`source`,
/// `lineage`, `namespace`) are the SDK-stamped boundary that every
/// pipeline must preserve.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Document {
    /// Stable identifier for this document within its namespace.
    pub id: DocumentId,
    /// The textual content. Splitters slice this; chunkers may
    /// rewrite it (Contextual Retrieval prepends a generated
    /// context prefix).
    pub content: String,
    /// Operator-supplied free-form metadata. Vector stores
    /// typically expose this as a filterable JSON column.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
    /// Origin of this document or its top-level ancestor.
    pub source: Source,
    /// Split / chunk ancestry. `None` on the loader-produced root;
    /// `Some` on every chunked descendant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<Lineage>,
    /// Multi-tenant boundary (invariant 11). Every persistent
    /// boundary the pipeline crosses respects this — silent
    /// cross-tenant leakage is structurally impossible because
    /// every loader, splitter, and pipeline takes a `Namespace`
    /// at construction time.
    pub namespace: Namespace,
}

impl Document {
    /// Construct a fresh root document — the shape a
    /// [`crate::DocumentLoader`] emits before any splitter has
    /// run. `lineage` is `None`; chunked descendants populate it.
    #[must_use]
    pub fn root(
        id: impl Into<DocumentId>,
        content: impl Into<String>,
        source: Source,
        namespace: Namespace,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: serde_json::Value::Null,
            source,
            lineage: None,
            namespace,
        }
    }

    /// Builder-style metadata setter.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Derive a child document for a chunk. The splitter supplies
    /// the per-chunk content + lineage; the new id is the parent's
    /// id with a `:{chunk_index}` suffix so leaves carry
    /// hierarchical identifiers stable across re-runs.
    #[must_use]
    pub fn child(&self, content: impl Into<String>, lineage: Lineage) -> Self {
        let child_id = format!("{}:{}", self.id, lineage.chunk_index);
        Self {
            id: DocumentId::new(child_id),
            content: content.into(),
            metadata: self.metadata.clone(),
            source: self.source.clone(),
            lineage: Some(lineage),
            namespace: self.namespace.clone(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn ns() -> Namespace {
        Namespace::new(entelix_core::TenantId::new("acme"))
    }

    fn src() -> Source {
        Source::now("file:///tmp/doc.md", "test")
    }

    #[test]
    fn document_id_rejects_empty() {
        let result = std::panic::catch_unwind(|| DocumentId::new(""));
        assert!(result.is_err(), "empty DocumentId must panic");
    }

    #[test]
    fn document_id_clone_shares_arc() {
        let id = DocumentId::new("doc-1");
        let cloned = id.clone();
        // Same `Arc<str>` allocation under both handles.
        assert_eq!(Arc::as_ptr(&id.0), Arc::as_ptr(&cloned.0));
    }

    #[test]
    fn child_id_suffixes_with_chunk_index() {
        let root = Document::root("paper", "full text", src(), ns());
        let lineage = Lineage::from_split(root.id.clone(), 3, 10, "recursive");
        let child = root.child("slice", lineage);
        assert_eq!(child.id.as_str(), "paper:3");
        assert_eq!(child.lineage.as_ref().unwrap().chunk_index, 3);
        assert_eq!(child.lineage.as_ref().unwrap().total_chunks, 10);
        assert_eq!(child.source.uri, root.source.uri);
        assert_eq!(child.namespace, root.namespace);
    }

    #[test]
    fn lineage_push_chunker_records_chain_order() {
        let mut lineage = Lineage::from_split(DocumentId::new("d"), 0, 1, "recursive");
        lineage.push_chunker("contextual");
        lineage.push_chunker("hyde");
        assert_eq!(lineage.chunker_chain, vec!["contextual", "hyde"]);
    }

    #[test]
    fn source_with_etag_preserves_other_fields() {
        let s = Source::now("https://example.com/p", "web").with_etag("W/\"abc\"");
        assert_eq!(s.etag.as_deref(), Some("W/\"abc\""));
        assert_eq!(s.loader, "web");
    }

    #[test]
    fn document_round_trips_through_serde() {
        let doc = Document::root("d", "hello", src(), ns())
            .with_metadata(serde_json::json!({"locale": "en"}));
        let json = serde_json::to_string(&doc).unwrap();
        let back: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(doc, back);
    }
}
