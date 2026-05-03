//! Binary `vnd.amazon.eventstream` round-trip + protocol error tests.
//! Decoder is exercised against fixtures encoded via the same crate's
//! `encode_frame` helper, plus hand-built malformed frames.

#![cfg(feature = "aws")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::checked_conversions,
    clippy::needless_pass_by_value,
    clippy::redundant_clone
)]

use entelix_cloud::bedrock::{
    EventStreamDecoder, EventStreamFrame, EventStreamHeader, EventStreamHeaderValue,
    EventStreamParseError, encode_frame,
};

fn encode(headers: Vec<EventStreamHeader>, payload: &[u8]) -> Vec<u8> {
    encode_frame(&headers, payload)
}

fn header(name: &str, value: EventStreamHeaderValue) -> EventStreamHeader {
    EventStreamHeader {
        name: name.to_owned(),
        value,
    }
}

#[test]
fn round_trip_empty_payload_no_headers() {
    let bytes = encode(Vec::new(), &[]);
    let mut dec = EventStreamDecoder::new();
    dec.push(&bytes);
    let frame = dec.next_frame().unwrap().unwrap();
    assert!(frame.headers.is_empty());
    assert!(frame.payload.is_empty());
}

#[test]
fn round_trip_all_nine_header_types() {
    let headers = vec![
        header("bool-true", EventStreamHeaderValue::Bool(true)),
        header("bool-false", EventStreamHeaderValue::Bool(false)),
        header("byte", EventStreamHeaderValue::Byte(-7)),
        header("int16", EventStreamHeaderValue::Int16(-1234)),
        header("int32", EventStreamHeaderValue::Int32(0x1122_3344)),
        header(
            "int64",
            EventStreamHeaderValue::Int64(-0x0102_0304_0506_0708),
        ),
        header(
            "bytes",
            EventStreamHeaderValue::Bytes(vec![0xde, 0xad, 0xbe, 0xef]),
        ),
        header("string", EventStreamHeaderValue::String("hello".into())),
        header("ts", EventStreamHeaderValue::Timestamp(1_700_000_000_000)),
        header(
            "uuid",
            EventStreamHeaderValue::Uuid([
                0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ]),
        ),
    ];
    let bytes = encode(headers.clone(), b"payload");
    let mut dec = EventStreamDecoder::new();
    dec.push(&bytes);
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.headers, headers);
    assert_eq!(frame.payload, b"payload");
    assert_eq!(
        frame.header_index.get("string"),
        Some(&EventStreamHeaderValue::String("hello".into()))
    );
}

#[test]
fn frame_split_across_chunks_assembles() {
    let bytes = encode(
        vec![header(
            ":event-type",
            EventStreamHeaderValue::String("contentBlockDelta".into()),
        )],
        br#"{"delta": "hi"}"#,
    );
    let mid = bytes.len() / 2;
    let mut dec = EventStreamDecoder::new();
    dec.push(&bytes[..mid]);
    assert!(dec.next_frame().unwrap().is_none());
    dec.push(&bytes[mid..]);
    let frame = dec.next_frame().unwrap().unwrap();
    assert_eq!(frame.event_type(), Some("contentBlockDelta"));
    assert_eq!(frame.payload, br#"{"delta": "hi"}"#);
}

#[test]
fn two_back_to_back_frames_decode() {
    let frame_a = encode(vec![], b"first");
    let frame_b = encode(vec![], b"second");
    let mut combined = frame_a.clone();
    combined.extend_from_slice(&frame_b);
    let mut dec = EventStreamDecoder::new();
    dec.push(&combined);
    let f1 = dec.next_frame().unwrap().unwrap();
    let f2 = dec.next_frame().unwrap().unwrap();
    assert_eq!(f1.payload, b"first");
    assert_eq!(f2.payload, b"second");
    assert!(dec.next_frame().unwrap().is_none());
}

#[test]
fn corrupted_prelude_crc_surfaces_error() {
    let mut bytes = encode(vec![], b"x");
    // Flip a byte in the prelude_crc field.
    bytes[8] ^= 0xff;
    let mut dec = EventStreamDecoder::new();
    dec.push(&bytes);
    let err = dec.next_frame().unwrap_err();
    assert!(matches!(
        err,
        EventStreamParseError::PreludeCrcMismatch { .. }
    ));
}

#[test]
fn corrupted_message_crc_surfaces_error() {
    let mut bytes = encode(vec![], b"x");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    let mut dec = EventStreamDecoder::new();
    dec.push(&bytes);
    let err = dec.next_frame().unwrap_err();
    assert!(matches!(
        err,
        EventStreamParseError::MessageCrcMismatch { .. }
    ));
}

#[test]
fn frame_too_short_in_total_length_field_rejected() {
    // total_length = 8 (less than min 16). We have to forge headers
    // by hand because encode_frame asserts on valid input.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&8u32.to_be_bytes()); // total
    bytes.extend_from_slice(&0u32.to_be_bytes()); // headers_length
    let prelude_crc = crc32fast::hash(&bytes[..8]);
    bytes.extend_from_slice(&prelude_crc.to_be_bytes());
    // No payload, no message CRC — but we need at least PRELUDE_LEN
    // to trigger the parse path.
    let mut dec = EventStreamDecoder::new();
    dec.push(&bytes);
    let err = dec.next_frame().unwrap_err();
    assert!(matches!(err, EventStreamParseError::FrameTooShort(8)));
}

#[test]
fn unknown_header_type_rejected() {
    // Manually craft a frame with a bogus header type byte.
    let header_name = b"x";
    let bogus_type: u8 = 200;
    let mut header_bytes = Vec::new();
    header_bytes.push(header_name.len() as u8);
    header_bytes.extend_from_slice(header_name);
    header_bytes.push(bogus_type);
    // Bogus header has no value bytes — decoder should reject the
    // type byte before reading further.
    let payload = b"";
    let total_length = (12 + header_bytes.len() + payload.len() + 4) as u32;
    let headers_length = header_bytes.len() as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&total_length.to_be_bytes());
    bytes.extend_from_slice(&headers_length.to_be_bytes());
    let prelude_crc = crc32fast::hash(&bytes[..8]);
    bytes.extend_from_slice(&prelude_crc.to_be_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(payload);
    let message_crc = crc32fast::hash(&bytes);
    bytes.extend_from_slice(&message_crc.to_be_bytes());

    let mut dec = EventStreamDecoder::new();
    dec.push(&bytes);
    let err = dec.next_frame().unwrap_err();
    assert!(matches!(err, EventStreamParseError::UnknownHeaderType(200)));
}

#[test]
fn event_type_helper_returns_none_when_absent() {
    let frame = EventStreamFrame {
        headers: Vec::new(),
        header_index: std::collections::HashMap::new(),
        payload: Vec::new(),
    };
    assert!(frame.event_type().is_none());
}
