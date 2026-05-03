//! `DirectTransport::send_streaming` integration tests using `wiremock`.
//!
//! Validates the streaming pipeline end-to-end: SSE bytes flow through
//! reqwest's `bytes_stream`, into a `TransportStream`, and (in the
//! round-trip test) through the Anthropic codec into IR.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use bytes::Bytes;
use entelix_core::ExecutionContext;
use entelix_core::auth::ApiKeyProvider;
use entelix_core::codecs::{AnthropicMessagesCodec, Codec, EncodedRequest};
use entelix_core::ir::{Message, ModelRequest, StopReason};
use entelix_core::stream::StreamAggregator;
use entelix_core::transports::{DirectTransport, Transport};
use futures::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn ctx() -> ExecutionContext {
    ExecutionContext::new()
}

const ANTHROPIC_SSE_BODY: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_E2E\",\"model\":\"claude-opus-4-7\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":2}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi from stream\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

#[tokio::test(flavor = "multi_thread")]
async fn send_streaming_returns_byte_stream_with_full_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(ANTHROPIC_SSE_BODY),
        )
        .mount(&server)
        .await;

    let transport =
        DirectTransport::new(server.uri(), Arc::new(ApiKeyProvider::anthropic("sk-test"))).unwrap();

    let req = EncodedRequest::post_json("/v1/messages", Bytes::from_static(b"{}"));
    let mut stream = transport.send_streaming(req, &ctx()).await.unwrap();
    assert_eq!(stream.status, 200);

    let mut accumulated = Vec::new();
    while let Some(chunk) = stream.body.next().await {
        accumulated.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(accumulated, ANTHROPIC_SSE_BODY.as_bytes());
}

#[tokio::test(flavor = "multi_thread")]
async fn send_streaming_through_anthropic_codec_round_trips() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(ANTHROPIC_SSE_BODY),
        )
        .mount(&server)
        .await;

    let transport =
        DirectTransport::new(server.uri(), Arc::new(ApiKeyProvider::anthropic("sk-test"))).unwrap();
    let codec = AnthropicMessagesCodec::new();

    let req = ModelRequest {
        model: "claude-opus-4-7".into(),
        messages: vec![Message::user("hi")],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode_streaming(&req).unwrap();
    let stream = transport.send_streaming(encoded, &ctx()).await.unwrap();
    let mut deltas = codec.decode_stream(stream.body, Vec::new());

    let mut aggregator = StreamAggregator::new();
    while let Some(delta) = deltas.next().await {
        aggregator.push(delta.unwrap()).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.id, "msg_E2E");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    let part = response.content.first().unwrap();
    if let entelix_core::ir::ContentPart::Text { text, .. } = part {
        assert_eq!(text, "hi from stream");
    } else {
        panic!("expected text part, got {part:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn send_streaming_non_2xx_returns_buffered_error_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let transport =
        DirectTransport::new(server.uri(), Arc::new(ApiKeyProvider::anthropic("sk-test"))).unwrap();

    let req = EncodedRequest::post_json("/v1/messages", Bytes::from_static(b"{}"));
    let mut stream = transport.send_streaming(req, &ctx()).await.unwrap();
    assert_eq!(stream.status, 429);
    let chunk = stream.body.next().await.unwrap().unwrap();
    assert_eq!(&chunk[..], b"rate limited");
}
