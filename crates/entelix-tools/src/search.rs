//! `SearchProvider` trait + [`SearchTool`] adapter.
//!
//! Concrete providers (Brave / Tavily / Perplexity / SerpAPI / …)
//! are deferred to 1.1 — same trait-only policy as
//! [`entelix_memory::Embedder`] (ADR-0008). Operators wire whatever
//! provider matches their compliance/cost stance and the SDK stays
//! out of the credentials game.
//!
//! ## Wiring example
//!
//! ```ignore
//! struct BraveProvider { api_key: SecretString }
//!
//! #[async_trait]
//! impl SearchProvider for BraveProvider {
//!     async fn search(
//!         &self,
//!         query: &str,
//!         max_results: usize,
//!     ) -> ToolResult<Vec<SearchResult>> { /* … */ }
//! }
//!
//! let tool = SearchTool::new(Arc::new(BraveProvider { api_key }));
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use entelix_core::AgentContext;
use entelix_core::error::Result;
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};

use crate::error::{ToolError, ToolResult};

/// Default cap on results returned per query.
pub const DEFAULT_MAX_RESULTS: usize = 5;

/// One search hit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result title.
    pub title: String,
    /// Canonical URL of the result.
    pub url: String,
    /// Short snippet / description.
    #[serde(default)]
    pub snippet: String,
    /// Optional vendor-supplied relevance score (`0.0`-`1.0`,
    /// caller's responsibility to normalize). Higher is better.
    #[serde(default)]
    pub score: Option<f32>,
}

/// Adapter trait the [`SearchTool`] dispatches to.
#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Run a query and return up to `max_results` hits, in
    /// descending relevance order. Implementations must respect
    /// `max_results` by truncation; callers depend on the cap as a
    /// cost-control signal.
    async fn search(&self, query: &str, max_results: usize) -> ToolResult<Vec<SearchResult>>;
}

/// `Tool` wrapper around a [`SearchProvider`].
pub struct SearchTool {
    provider: Arc<dyn SearchProvider>,
    default_max_results: usize,
    metadata: ToolMetadata,
}

#[allow(
    clippy::missing_fields_in_debug,
    reason = "provider is dyn-trait without Debug; printed via default cap"
)]
impl std::fmt::Debug for SearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchTool")
            .field("default_max_results", &self.default_max_results)
            .finish()
    }
}

impl SearchTool {
    /// Build with the default `max_results` cap.
    #[must_use]
    pub fn new(provider: Arc<dyn SearchProvider>) -> Self {
        Self {
            provider,
            default_max_results: DEFAULT_MAX_RESULTS,
            metadata: search_tool_metadata(),
        }
    }

    /// Override the default `max_results` (the cap remains caller-
    /// overridable per call via the input schema).
    #[must_use]
    pub fn with_default_max_results(mut self, n: usize) -> Self {
        self.default_max_results = n;
        self
    }
}

fn search_tool_metadata() -> ToolMetadata {
    ToolMetadata::function(
        "search",
        "Run a web search and return the top-N hits. Returns title, url, snippet for each hit.",
        json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": { "type": "string", "description": "Search query string." },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of hits to return."
                }
            }
        }),
    )
    .with_effect(ToolEffect::ReadOnly)
    .with_idempotent(true)
}

#[derive(Debug, Deserialize)]
struct SearchInput {
    query: String,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Serialize)]
struct SearchOutput {
    query: String,
    results: Vec<SearchResult>,
}

#[async_trait]
impl Tool for SearchTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, _ctx: &AgentContext<()>) -> Result<Value> {
        let parsed: SearchInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let n = parsed
            .max_results
            .unwrap_or(self.default_max_results)
            .max(1);
        let results = self.provider.search(&parsed.query, n).await?;
        let truncated = results.into_iter().take(n).collect();
        let output = SearchOutput {
            query: parsed.query,
            results: truncated,
        };
        Ok(serde_json::to_value(output).map_err(ToolError::from)?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Mock provider that records the requested query / cap and
    /// returns a fixed result list (truncated to `max_results`).
    struct MockProvider {
        recorded: Mutex<Vec<(String, usize)>>,
        canned: Vec<SearchResult>,
    }

    impl MockProvider {
        fn new(canned: Vec<SearchResult>) -> Self {
            Self {
                recorded: Mutex::new(Vec::new()),
                canned,
            }
        }
    }

    #[async_trait]
    impl SearchProvider for MockProvider {
        async fn search(&self, query: &str, max_results: usize) -> ToolResult<Vec<SearchResult>> {
            self.recorded
                .lock()
                .unwrap()
                .push((query.to_owned(), max_results));
            Ok(self.canned.iter().take(max_results).cloned().collect())
        }
    }

    fn hit(title: &str, url: &str) -> SearchResult {
        SearchResult {
            title: title.into(),
            url: url.into(),
            snippet: format!("snippet for {title}"),
            score: None,
        }
    }

    #[tokio::test]
    async fn dispatches_to_provider_with_default_cap() {
        let provider = Arc::new(MockProvider::new(vec![
            hit("a", "https://a"),
            hit("b", "https://b"),
        ]));
        let tool = SearchTool::new(provider.clone());
        let out = tool
            .execute(json!({"query": "rust async"}), &AgentContext::default())
            .await
            .unwrap();
        assert_eq!(out["query"], "rust async");
        let recorded = provider.recorded.lock().unwrap();
        assert_eq!(recorded[0].0, "rust async");
        assert_eq!(recorded[0].1, DEFAULT_MAX_RESULTS);
    }

    #[tokio::test]
    async fn caller_can_override_max_results() {
        let provider = Arc::new(MockProvider::new(vec![
            hit("a", "https://a"),
            hit("b", "https://b"),
            hit("c", "https://c"),
        ]));
        let tool = SearchTool::new(provider.clone());
        let out = tool
            .execute(
                json!({"query": "x", "max_results": 2}),
                &AgentContext::default(),
            )
            .await
            .unwrap();
        let arr = out["results"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["url"], "https://a");
    }

    #[tokio::test]
    async fn rejects_missing_query() {
        let provider: Arc<dyn SearchProvider> = Arc::new(MockProvider::new(Vec::new()));
        let tool = SearchTool::new(provider);
        let err = tool
            .execute(json!({"not_a_query": 1}), &AgentContext::default())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("missing field"));
    }

    #[tokio::test]
    async fn provider_error_surfaces_via_tool() {
        struct FailingProvider;
        #[async_trait]
        impl SearchProvider for FailingProvider {
            async fn search(
                &self,
                _query: &str,
                _max_results: usize,
            ) -> ToolResult<Vec<SearchResult>> {
                Err(ToolError::network_msg("upstream 503"))
            }
        }
        let tool = SearchTool::new(Arc::new(FailingProvider));
        let err = tool
            .execute(json!({"query": "x"}), &AgentContext::default())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("upstream 503"));
    }
}
