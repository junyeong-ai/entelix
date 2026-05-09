//! Verifies that `ChatModelExt::with_structured_output::<O>()` produces a
//! `Runnable<Vec<Message>, O>` that composes via `.pipe()`. Uses an inline
//! mock codec/transport that emits a JSON-shaped assistant text block so
//! `complete_typed` can parse it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unnecessary_literal_bound
)]

use bytes::Bytes;
use entelix_core::codecs::{Codec, EncodedRequest};
use entelix_core::ir::{
    Capabilities, ContentPart, Message, ModelRequest, ModelResponse, ModelWarning, StopReason,
    Usage,
};
use entelix_core::transports::{Transport, TransportResponse};
use entelix_core::{ChatModel, ExecutionContext, Result};
use entelix_runnable::{ChatModelExt, Runnable, RunnableExt, RunnableLambda};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq)]
struct Order {
    sku: String,
    quantity: u32,
}

/// Codec that decodes by producing a hard-coded assistant text reply
/// containing valid JSON for `Order`. This mimics a vendor that
/// returns `complete_typed`'s prompted-strategy output as a single
/// JSON text block.
struct OrderCodec;

impl Codec for OrderCodec {
    fn name(&self) -> &'static str {
        "order-codec"
    }

    fn capabilities(&self, _model: &str) -> Capabilities {
        Capabilities::default()
    }

    fn encode(&self, request: &ModelRequest) -> Result<EncodedRequest> {
        Ok(EncodedRequest::post_json(
            "/order",
            Bytes::from(serde_json::to_vec(request).unwrap()),
        ))
    }

    fn decode(&self, _body: &[u8], warnings_in: Vec<ModelWarning>) -> Result<ModelResponse> {
        Ok(ModelResponse {
            id: "o_1".into(),
            model: "order-model".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text(r#"{"sku": "ABC-123", "quantity": 4}"#)],
            usage: Usage::default(),
            rate_limit: None,
            warnings: warnings_in,
        })
    }
}

struct OkTransport;

#[async_trait::async_trait]
impl Transport for OkTransport {
    fn name(&self) -> &'static str {
        "ok"
    }

    async fn send(
        &self,
        _request: EncodedRequest,
        _ctx: &ExecutionContext,
    ) -> Result<TransportResponse> {
        Ok(TransportResponse {
            status: 200,
            headers: http::HeaderMap::new(),
            body: Bytes::new(),
        })
    }
}

#[tokio::test]
async fn with_structured_output_produces_runnable_returning_typed_value() -> Result<()> {
    let model = ChatModel::new(OrderCodec, OkTransport, "order-model");
    let runnable = model.with_structured_output::<Order>();
    let order: Order = runnable
        .invoke(vec![Message::user("place order")], &ExecutionContext::new())
        .await?;
    assert_eq!(
        order,
        Order {
            sku: "ABC-123".into(),
            quantity: 4
        }
    );
    Ok(())
}

#[tokio::test]
async fn with_structured_output_pipes_into_downstream_runnable() -> Result<()> {
    let model = ChatModel::new(OrderCodec, OkTransport, "order-model");

    // Pipe the typed output into a downstream `RunnableLambda` that
    // computes total cost — proves the typed adapter composes via
    // `.pipe()` exactly like any other `Runnable<Vec<Message>, O>`.
    let chain = model
        .with_structured_output::<Order>()
        .pipe(RunnableLambda::new(
            |order: Order, _ctx: ExecutionContext| async move {
                Ok::<u32, entelix_core::Error>(order.quantity * 25)
            },
        ));

    let total: u32 = chain
        .invoke(vec![Message::user("place order")], &ExecutionContext::new())
        .await?;
    assert_eq!(total, 100);
    Ok(())
}
