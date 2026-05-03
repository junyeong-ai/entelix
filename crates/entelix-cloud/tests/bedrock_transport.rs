//! `BedrockTransport` end-to-end tests via `wiremock`. Validates:
//! - Bearer auth path (Anthropic-on-Bedrock bridge)
//! - Streaming response unwrapping (binary event-stream → JSON
//!   payload bytes)
//! - Non-2xx status surfaces buffered error body

#![cfg(feature = "aws")]
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use bytes::Bytes;
use entelix_cloud::bedrock::{
    BedrockAuth, BedrockTransport, EventStreamHeader, EventStreamHeaderValue, encode_frame,
};
use entelix_core::ExecutionContext;
use entelix_core::codecs::EncodedRequest;
use entelix_core::transports::Transport;
use futures::StreamExt;
use secrecy::SecretString;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn ctx() -> ExecutionContext {
    ExecutionContext::new()
}

fn bearer_transport(base_url: &str) -> BedrockTransport {
    BedrockTransport::builder()
        .with_region("us-east-1")
        .with_base_url(base_url)
        .with_auth(BedrockAuth::Bearer {
            token: SecretString::from("test-token".to_owned()),
        })
        .build()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn bearer_auth_attaches_authorization_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/m/converse"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"ok\":true}"))
        .mount(&server)
        .await;

    let transport = bearer_transport(&server.uri());
    let req = EncodedRequest::post_json("/model/m/converse", Bytes::from_static(b"{}"));
    let resp = transport.send(req, &ctx()).await.unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(&resp.body[..], b"{\"ok\":true}");
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_unwraps_binary_event_stream_into_json_payloads() {
    let server = MockServer::start().await;

    let frame_a = encode_frame(
        &[EventStreamHeader {
            name: ":event-type".into(),
            value: EventStreamHeaderValue::String("messageStart".into()),
        }],
        br#"{"role":"assistant"}"#,
    );
    let frame_b = encode_frame(
        &[EventStreamHeader {
            name: ":event-type".into(),
            value: EventStreamHeaderValue::String("contentBlockDelta".into()),
        }],
        br#"{"delta":{"text":"hi"}}"#,
    );
    let frame_c = encode_frame(
        &[EventStreamHeader {
            name: ":event-type".into(),
            value: EventStreamHeaderValue::String("messageStop".into()),
        }],
        br#"{"stopReason":"end_turn"}"#,
    );
    let mut combined = Vec::new();
    combined.extend_from_slice(&frame_a);
    combined.extend_from_slice(&frame_b);
    combined.extend_from_slice(&frame_c);

    Mock::given(method("POST"))
        .and(path("/model/m/converse-stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_bytes(combined),
        )
        .mount(&server)
        .await;

    let transport = bearer_transport(&server.uri());
    let req = EncodedRequest::post_json("/model/m/converse-stream", Bytes::from_static(b"{}"));
    let mut stream = transport.send_streaming(req, &ctx()).await.unwrap();
    assert_eq!(stream.status, 200);

    let mut payloads = Vec::new();
    while let Some(chunk) = stream.body.next().await {
        payloads.push(chunk.unwrap());
    }
    assert_eq!(payloads.len(), 3);
    assert_eq!(&payloads[0][..], br#"{"role":"assistant"}"#);
    assert_eq!(&payloads[1][..], br#"{"delta":{"text":"hi"}}"#);
    assert_eq!(&payloads[2][..], br#"{"stopReason":"end_turn"}"#);
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_non_2xx_returns_buffered_error_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403).set_body_string("AccessDenied"))
        .mount(&server)
        .await;

    let transport = bearer_transport(&server.uri());
    let req = EncodedRequest::post_json("/model/m/converse-stream", Bytes::from_static(b"{}"));
    let mut stream = transport.send_streaming(req, &ctx()).await.unwrap();
    assert_eq!(stream.status, 403);
    let chunk = stream.body.next().await.unwrap().unwrap();
    assert_eq!(&chunk[..], b"AccessDenied");
}

#[tokio::test(flavor = "multi_thread")]
async fn corrupted_event_stream_surfaces_error_chunk() {
    let server = MockServer::start().await;
    let mut bytes = encode_frame(&[], b"x");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff; // corrupt message CRC
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_bytes(bytes),
        )
        .mount(&server)
        .await;

    let transport = bearer_transport(&server.uri());
    let req = EncodedRequest::post_json("/model/m/converse-stream", Bytes::from_static(b"{}"));
    let mut stream = transport.send_streaming(req, &ctx()).await.unwrap();
    let first = stream.body.next().await.unwrap();
    assert!(
        first.is_err(),
        "corrupted CRC must surface as Err in the byte stream"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_token_short_circuits_send() {
    let transport = bearer_transport("http://127.0.0.1:1");
    let context = ExecutionContext::new();
    context.cancellation().cancel();
    let req = EncodedRequest::post_json("/model/m/converse", Bytes::from_static(b"{}"));
    let err = transport.send(req, &context).await.unwrap_err();
    assert!(
        matches!(err, entelix_core::Error::Cancelled),
        "expected Error::Cancelled, got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_token_short_circuits_send_streaming() {
    let transport = bearer_transport("http://127.0.0.1:1");
    let context = ExecutionContext::new();
    context.cancellation().cancel();
    let req = EncodedRequest::post_json("/model/m/converse-stream", Bytes::from_static(b"{}"));
    let err = transport.send_streaming(req, &context).await.unwrap_err();
    assert!(
        matches!(err, entelix_core::Error::Cancelled),
        "expected Error::Cancelled, got {err:?}"
    );
}
