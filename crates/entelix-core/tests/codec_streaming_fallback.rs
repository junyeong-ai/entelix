//! `Codec::decode_stream` default-impl fallback round-trips a buffered
//! decode through the same `StreamDelta` shape a true streaming codec
//! would emit. Validates that codecs which haven't implemented
//! token-level decoding remain useful — `StreamAggregator::finalize`
//! reproduces the same `ModelResponse` as a one-shot `decode`.
//!
//! Exercised against `BedrockConverseCodec` because Bedrock's
//! `vnd.amazon.eventstream` binary stream lives in `entelix-cloud` —
//! the codec subset relies on the trait default until the cloud crate
//! ships a real impl.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use bytes::Bytes;
use entelix_core::codecs::{BedrockConverseCodec, BoxByteStream, Codec};
use entelix_core::ir::{ModelWarning, StopReason};
use entelix_core::stream::{StreamAggregator, StreamDelta};
use futures::StreamExt;

const BEDROCK_BODY: &str = r#"{
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                { "text": "hello world" },
                { "toolUse": { "toolUseId": "tu-1", "name": "echo", "input": {"x": 1} } }
            ]
        }
    },
    "stopReason": "end_turn",
    "usage": { "inputTokens": 4, "outputTokens": 5 }
}"#;

fn body_stream(parts: Vec<&'static [u8]>) -> BoxByteStream<'static> {
    Box::pin(futures::stream::iter(
        parts
            .into_iter()
            .map(|b| Ok::<_, entelix_core::Error>(Bytes::from(b))),
    ))
}

#[tokio::test]
async fn fallback_decode_stream_emits_full_delta_sequence() {
    let codec = BedrockConverseCodec::new();
    let bytes = body_stream(vec![BEDROCK_BODY.as_bytes()]);
    let mut stream = codec.decode_stream(bytes, Vec::<ModelWarning>::new());

    let mut deltas = Vec::new();
    while let Some(item) = stream.next().await {
        deltas.push(item.unwrap());
    }

    // start, text, tool_use_start, tool_use_input_delta, tool_use_stop, usage, stop
    assert!(matches!(deltas[0], StreamDelta::Start { .. }));
    assert!(matches!(deltas[1], StreamDelta::TextDelta { .. }));
    assert!(matches!(deltas[2], StreamDelta::ToolUseStart { .. }));
    assert!(matches!(deltas[3], StreamDelta::ToolUseInputDelta { .. }));
    assert!(matches!(deltas[4], StreamDelta::ToolUseStop));
    assert!(matches!(deltas[5], StreamDelta::Usage(_)));
    assert!(matches!(
        deltas.last().unwrap(),
        StreamDelta::Stop {
            stop_reason: StopReason::EndTurn
        }
    ));
}

#[tokio::test]
async fn fallback_streaming_round_trips_through_aggregator() {
    // Multi-chunk byte split to prove the fallback's accumulation buffer
    // works.
    let codec = BedrockConverseCodec::new();
    let mid = BEDROCK_BODY.len() / 2;
    let all = BEDROCK_BODY.as_bytes();
    let head = all[..mid].to_vec();
    let tail = all[mid..].to_vec();
    let bytes: BoxByteStream<'static> = Box::pin(futures::stream::iter(vec![
        Ok::<_, entelix_core::Error>(Bytes::from(head)),
        Ok::<_, entelix_core::Error>(Bytes::from(tail)),
    ]));

    let mut stream = codec.decode_stream(bytes, Vec::new());
    let mut aggregator = StreamAggregator::new();
    while let Some(delta) = stream.next().await {
        aggregator.push(delta.unwrap()).unwrap();
    }
    let response = aggregator.finalize().unwrap();
    assert_eq!(response.usage.output_tokens, 5);
    assert!(matches!(response.stop_reason, StopReason::EndTurn));
}

#[tokio::test]
async fn fallback_stream_propagates_byte_error() {
    let codec = BedrockConverseCodec::new();
    let bytes: BoxByteStream<'static> = Box::pin(futures::stream::iter(vec![
        Ok::<_, entelix_core::Error>(Bytes::from_static(b"{partial")),
        Err(entelix_core::Error::invalid_request("network died")),
    ]));
    let mut stream = codec.decode_stream(bytes, Vec::new());
    let first = stream.next().await.unwrap();
    assert!(first.is_err());
}
