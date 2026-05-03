//! `OpenAiEmbedder` — concrete `Embedder` over OpenAI `/v1/embeddings`.

use std::sync::Arc;

use async_trait::async_trait;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::json;

use entelix_core::auth::CredentialProvider;
use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_memory::{Embedder, Embedding, EmbeddingUsage};

use crate::error::{OpenAiEmbedderError, OpenAiEmbedderResult};

/// OpenAI's lower-cost embedding model. Native dimension 1536; can
/// be reduced via the `dimensions` request parameter (operator
/// `with_dimension` on the builder).
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";

/// Native dimension of [`TEXT_EMBEDDING_3_SMALL`].
pub const TEXT_EMBEDDING_3_SMALL_DIMENSION: usize = 1536;

/// OpenAI's higher-quality embedding model. Native dimension 3072;
/// can be reduced via the `dimensions` request parameter.
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";

/// Native dimension of [`TEXT_EMBEDDING_3_LARGE`].
pub const TEXT_EMBEDDING_3_LARGE_DIMENSION: usize = 3072;

/// Default OpenAI API base URL. Override via
/// [`OpenAiEmbedderBuilder::with_base_url`] for proxies, regional
/// endpoints, or test fixtures.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Concrete [`Embedder`] backed by OpenAI's `/v1/embeddings` HTTPS
/// endpoint. Stateless beyond the connection pool inside
/// `reqwest::Client`; clone freely or wrap in `Arc` per F10.
#[derive(Clone)]
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    base_url: Arc<str>,
    credentials: Arc<dyn CredentialProvider>,
    model: Arc<str>,
    dimension: usize,
    /// `Some` when the operator explicitly chose a reduced
    /// dimension via `with_dimension`. We forward this on the
    /// `dimensions` field to the API; native dimension is sent as
    /// `None` so OpenAI applies its own default.
    dimension_override: Option<usize>,
}

impl std::fmt::Debug for OpenAiEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbedder")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("dimension", &self.dimension)
            .field("dimension_override", &self.dimension_override)
            .finish_non_exhaustive()
    }
}

impl OpenAiEmbedder {
    /// Builder for the lower-cost `text-embedding-3-small` model
    /// (native dimension 1536).
    pub fn small() -> OpenAiEmbedderBuilder {
        OpenAiEmbedderBuilder::new(TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_3_SMALL_DIMENSION)
    }

    /// Builder for the higher-quality `text-embedding-3-large` model
    /// (native dimension 3072).
    pub fn large() -> OpenAiEmbedderBuilder {
        OpenAiEmbedderBuilder::new(TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_LARGE_DIMENSION)
    }

    /// Builder for an operator-supplied custom model identifier.
    /// Use for OpenAI-compatible APIs (Azure OpenAI, vLLM-compatible
    /// gateways) or for future models the SDK has not yet promoted
    /// to a `pub const`.
    pub fn custom(model: impl Into<String>, dimension: usize) -> OpenAiEmbedderBuilder {
        OpenAiEmbedderBuilder::new(model, dimension)
    }

    fn embeddings_url(&self) -> String {
        format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'))
    }

    /// Send one batch (1 or N inputs) to `/v1/embeddings` and decode
    /// the response. The response's `usage` is split across the
    /// returned `Embedding`s — OpenAI reports a single per-call
    /// `prompt_tokens` count which we attribute to the first
    /// `Embedding` to keep aggregate accounting accurate without
    /// double-charging.
    async fn call(&self, inputs: Vec<String>) -> OpenAiEmbedderResult<Vec<Embedding>> {
        let credentials = self
            .credentials
            .resolve()
            .await
            .map_err(OpenAiEmbedderError::Credential)?;

        let body = self.build_request_body(&inputs);
        let response = self
            .client
            .post(self.embeddings_url())
            .header(
                credentials.header_name.clone(),
                http::HeaderValue::from_str(credentials.header_value.expose_secret()).map_err(
                    |e| OpenAiEmbedderError::Config(format!("invalid credential header: {e}")),
                )?,
            )
            .json(&body)
            .send()
            .await
            .map_err(OpenAiEmbedderError::network)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(OpenAiEmbedderError::HttpStatus {
                status: status.as_u16(),
                body: truncate_for_error(&body),
            });
        }

        let parsed: EmbeddingsResponse = response
            .json()
            .await
            .map_err(OpenAiEmbedderError::network)?;
        self.decode(&parsed, inputs.len())
    }

    fn build_request_body(&self, inputs: &[String]) -> serde_json::Value {
        let mut body = json!({
            "model": &*self.model,
            "input": inputs,
            "encoding_format": "float",
        });
        if let Some(dim) = self.dimension_override
            && let Some(obj) = body.as_object_mut()
        {
            obj.insert("dimensions".into(), json!(dim));
        }
        body
    }

    fn decode(
        &self,
        parsed: &EmbeddingsResponse,
        expected_len: usize,
    ) -> OpenAiEmbedderResult<Vec<Embedding>> {
        if parsed.data.len() != expected_len {
            return Err(OpenAiEmbedderError::Malformed(format!(
                "expected {expected_len} embeddings, server returned {}",
                parsed.data.len()
            )));
        }
        // OpenAI does not guarantee response-`index` ordering matches
        // request order — sort by `index` to match the input slot.
        let mut sorted: Vec<&EmbeddingsDataItem> = parsed.data.iter().collect();
        sorted.sort_by_key(|d| d.index);

        let usage = parsed.usage.map(|u| EmbeddingUsage::new(u.prompt_tokens));
        let mut out = Vec::with_capacity(expected_len);
        for (i, item) in sorted.iter().enumerate() {
            if item.embedding.len() != self.dimension {
                return Err(OpenAiEmbedderError::Malformed(format!(
                    "embedding {} dimension {} does not match configured {}",
                    i,
                    item.embedding.len(),
                    self.dimension
                )));
            }
            // Attribute the per-call usage to slot 0 only — downstream
            // meters sum across the batch and would double-charge if
            // we replicated the count on every slot.
            let mut emb = Embedding::new(item.embedding.clone());
            if i == 0
                && let Some(u) = usage
            {
                emb = emb.with_usage(u);
            }
            out.push(emb);
        }
        Ok(out)
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, text: &str, ctx: &ExecutionContext) -> Result<Embedding> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut out = self
            .call(vec![text.to_owned()])
            .await
            .map_err(Error::from)?;
        out.pop()
            .ok_or_else(|| Error::provider_network("OpenAI returned no embedding".to_owned()))
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Vec<Embedding>> {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // One HTTP call per batch — F10 amortization. Default
        // sequential impl would issue N round-trips; we override.
        self.call(texts.to_vec()).await.map_err(Error::from)
    }
}

/// Builder for [`OpenAiEmbedder`].
#[must_use]
pub struct OpenAiEmbedderBuilder {
    model: String,
    dimension: usize,
    dimension_override: Option<usize>,
    base_url: String,
    credentials: Option<Arc<dyn CredentialProvider>>,
    client: Option<reqwest::Client>,
}

impl OpenAiEmbedderBuilder {
    fn new(model: impl Into<String>, native_dimension: usize) -> Self {
        Self {
            model: model.into(),
            dimension: native_dimension,
            dimension_override: None,
            base_url: DEFAULT_BASE_URL.to_owned(),
            credentials: None,
            client: None,
        }
    }

    /// Attach a credential provider. Required.
    pub fn with_credentials(mut self, credentials: Arc<dyn CredentialProvider>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Override the API base URL (defaults to
    /// [`DEFAULT_BASE_URL`]). Used for Azure OpenAI, regional
    /// endpoints, or test fixtures.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Request a reduced dimension via the API's `dimensions`
    /// parameter. Must be ≤ the model's native dimension. Storage
    /// savings come from `text-embedding-3` family's matryoshka
    /// representation — quality degrades gracefully toward the
    /// chosen dimension.
    pub const fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension_override = Some(dimension);
        self.dimension = dimension;
        self
    }

    /// Override the underlying HTTP client. Useful when the operator
    /// runs a shared `reqwest::Client` to consolidate connection
    /// pools across embedder + chat + tool transports.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Finalize the builder. Returns
    /// [`OpenAiEmbedderError::Config`] if credentials are missing
    /// or the configured dimension exceeds the native maximum.
    pub fn build(self) -> OpenAiEmbedderResult<OpenAiEmbedder> {
        let credentials = self
            .credentials
            .ok_or_else(|| OpenAiEmbedderError::Config("credentials required".into()))?;
        if self.dimension == 0 {
            return Err(OpenAiEmbedderError::Config("dimension must be > 0".into()));
        }
        let client = self.client.unwrap_or_default();
        Ok(OpenAiEmbedder {
            client,
            base_url: self.base_url.into(),
            credentials,
            model: self.model.into(),
            dimension: self.dimension,
            dimension_override: self.dimension_override,
        })
    }
}

// ── wire format ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingsDataItem>,
    #[serde(default)]
    usage: Option<EmbeddingsUsageItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsDataItem {
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
struct EmbeddingsUsageItem {
    prompt_tokens: u32,
}

const ERROR_BODY_TRUNCATION_BYTES: usize = 512;

fn truncate_for_error(body: &str) -> String {
    if body.len() <= ERROR_BODY_TRUNCATION_BYTES {
        return body.to_owned();
    }
    let mut cut = ERROR_BODY_TRUNCATION_BYTES;
    while cut > 0 && !body.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}… ({} bytes truncated)", &body[..cut], body.len() - cut)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use entelix_core::auth::ApiKeyProvider;

    fn provider() -> Arc<dyn CredentialProvider> {
        Arc::new(ApiKeyProvider::new("authorization", "Bearer test").unwrap())
    }

    #[test]
    fn small_builder_defaults_to_native_dimension() {
        let e = OpenAiEmbedder::small()
            .with_credentials(provider())
            .build()
            .unwrap();
        assert_eq!(e.dimension(), TEXT_EMBEDDING_3_SMALL_DIMENSION);
        assert_eq!(&*e.model, TEXT_EMBEDDING_3_SMALL);
    }

    #[test]
    fn large_builder_defaults_to_native_dimension() {
        let e = OpenAiEmbedder::large()
            .with_credentials(provider())
            .build()
            .unwrap();
        assert_eq!(e.dimension(), TEXT_EMBEDDING_3_LARGE_DIMENSION);
    }

    #[test]
    fn dimension_override_threads_into_request_body() {
        let e = OpenAiEmbedder::small()
            .with_credentials(provider())
            .with_dimension(512)
            .build()
            .unwrap();
        assert_eq!(e.dimension(), 512);
        let body = e.build_request_body(&["hi".to_owned()]);
        assert_eq!(body["dimensions"], 512);
    }

    #[test]
    fn missing_credentials_rejected_at_build() {
        let err = OpenAiEmbedder::small().build().unwrap_err();
        assert!(matches!(err, OpenAiEmbedderError::Config(_)));
    }

    #[test]
    fn zero_dimension_rejected_at_build() {
        let err = OpenAiEmbedder::custom("custom-model", 0)
            .with_credentials(provider())
            .build()
            .unwrap_err();
        assert!(matches!(err, OpenAiEmbedderError::Config(_)));
    }

    #[test]
    fn embeddings_url_strips_trailing_slash() {
        let e = OpenAiEmbedder::small()
            .with_credentials(provider())
            .with_base_url("https://example.test/")
            .build()
            .unwrap();
        assert_eq!(e.embeddings_url(), "https://example.test/v1/embeddings");
    }

    #[test]
    fn decode_attributes_usage_to_first_slot_only() {
        let e = OpenAiEmbedder::custom("test-model", 3)
            .with_credentials(provider())
            .build()
            .unwrap();
        let parsed = EmbeddingsResponse {
            data: vec![
                EmbeddingsDataItem {
                    embedding: vec![0.1, 0.2, 0.3],
                    index: 0,
                },
                EmbeddingsDataItem {
                    embedding: vec![0.4, 0.5, 0.6],
                    index: 1,
                },
            ],
            usage: Some(EmbeddingsUsageItem { prompt_tokens: 7 }),
        };
        let out = e.decode(&parsed, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].usage, Some(EmbeddingUsage::new(7)));
        assert!(
            out[1].usage.is_none(),
            "usage must NOT replicate across slots"
        );
    }

    #[test]
    fn decode_sorts_by_index_when_response_order_shuffled() {
        let e = OpenAiEmbedder::custom("test-model", 2)
            .with_credentials(provider())
            .build()
            .unwrap();
        let parsed = EmbeddingsResponse {
            data: vec![
                EmbeddingsDataItem {
                    embedding: vec![0.9, 0.9],
                    index: 1,
                },
                EmbeddingsDataItem {
                    embedding: vec![0.1, 0.1],
                    index: 0,
                },
            ],
            usage: None,
        };
        let out = e.decode(&parsed, 2).unwrap();
        assert_eq!(out[0].vector, vec![0.1, 0.1]);
        assert_eq!(out[1].vector, vec![0.9, 0.9]);
    }

    #[test]
    fn decode_rejects_dimension_mismatch() {
        let e = OpenAiEmbedder::custom("test-model", 3)
            .with_credentials(provider())
            .build()
            .unwrap();
        let parsed = EmbeddingsResponse {
            data: vec![EmbeddingsDataItem {
                embedding: vec![0.1, 0.2], // 2 != 3
                index: 0,
            }],
            usage: None,
        };
        let err = e.decode(&parsed, 1).unwrap_err();
        assert!(matches!(err, OpenAiEmbedderError::Malformed(_)));
    }

    #[test]
    fn decode_rejects_count_mismatch() {
        let e = OpenAiEmbedder::custom("test-model", 1)
            .with_credentials(provider())
            .build()
            .unwrap();
        let parsed = EmbeddingsResponse {
            data: vec![EmbeddingsDataItem {
                embedding: vec![0.1],
                index: 0,
            }],
            usage: None,
        };
        let err = e.decode(&parsed, 2).unwrap_err();
        assert!(matches!(err, OpenAiEmbedderError::Malformed(_)));
    }

    #[test]
    fn truncate_for_error_caps_oversized_body() {
        let huge = "x".repeat(10_000);
        let truncated = truncate_for_error(&huge);
        assert!(truncated.contains("truncated"));
        assert!(truncated.len() < 1000);
    }
}
