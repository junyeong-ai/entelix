//! AWS `vnd.amazon.eventstream` binary frame decoder.
//!
//! Wire format (all integers big-endian):
//!
//! ```text
//! ┌───────────────────────┐
//! │ total_length      u32 │  full message bytes incl. length fields
//! │ headers_length    u32 │  header section bytes
//! │ prelude_crc32     u32 │  CRC-32 over the two length fields above
//! ├───────────────────────┤
//! │ headers...            │  zero or more typed headers
//! │ payload...            │  arbitrary bytes
//! ├───────────────────────┤
//! │ message_crc32     u32 │  CRC-32 over every byte before this one
//! └───────────────────────┘
//! ```
//!
//! Each header is `[name_len: u8][name: utf-8][type: u8][value: per-type]`.
//! Nine value types (0..=9) cover bool / signed integers / bytes /
//! string / timestamp / uuid — see [`EventStreamHeaderValue`].
//!
//! The decoder is incremental: feed bytes via [`EventStreamDecoder::push`]
//! and pull complete frames with [`EventStreamDecoder::next_frame`].
//! Implementation is self-contained — no AWS SDK dep.

use std::collections::HashMap;

use bytes::{Buf, BytesMut};
use thiserror::Error;

const PRELUDE_LEN: usize = 12;
const MESSAGE_CRC_LEN: usize = 4;
const MIN_FRAME_LEN: usize = PRELUDE_LEN + MESSAGE_CRC_LEN;

const TYPE_BOOL_TRUE: u8 = 0;
const TYPE_BOOL_FALSE: u8 = 1;
const TYPE_BYTE: u8 = 2;
const TYPE_INT16: u8 = 3;
const TYPE_INT32: u8 = 4;
const TYPE_INT64: u8 = 5;
const TYPE_BYTE_ARRAY: u8 = 6;
const TYPE_STRING: u8 = 7;
const TYPE_TIMESTAMP: u8 = 8;
const TYPE_UUID: u8 = 9;

/// One typed header value.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EventStreamHeaderValue {
    /// `bool` — wire types 0 (true) and 1 (false).
    Bool(bool),
    /// `i8` — wire type 2.
    Byte(i8),
    /// `i16` big-endian — wire type 3.
    Int16(i16),
    /// `i32` big-endian — wire type 4.
    Int32(i32),
    /// `i64` big-endian — wire type 5.
    Int64(i64),
    /// Variable-length byte array (u16 length prefix) — wire type 6.
    Bytes(Vec<u8>),
    /// UTF-8 string (u16 length prefix) — wire type 7.
    String(String),
    /// Milliseconds since the Unix epoch (`i64` BE) — wire type 8.
    Timestamp(i64),
    /// 16-byte UUID — wire type 9.
    Uuid([u8; 16]),
}

/// One header `(name, value)` pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventStreamHeader {
    /// Header name — UTF-8, max 255 bytes per AWS spec.
    pub name: String,
    /// Typed value.
    pub value: EventStreamHeaderValue,
}

/// One decoded frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventStreamFrame {
    /// Headers in the order they appeared on the wire.
    pub headers: Vec<EventStreamHeader>,
    /// Header lookup by name — O(1) access for callers that only
    /// care about specific keys.
    pub header_index: HashMap<String, EventStreamHeaderValue>,
    /// Payload bytes (typically a JSON event for Bedrock streams).
    pub payload: Vec<u8>,
}

impl EventStreamFrame {
    /// Convenience: pull the `:event-type` header value as a string,
    /// matching AWS service convention.
    pub fn event_type(&self) -> Option<&str> {
        match self.header_index.get(":event-type") {
            Some(EventStreamHeaderValue::String(s)) => Some(s),
            _ => None,
        }
    }
}

/// Errors from frame parsing.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EventStreamParseError {
    /// `total_length` field claims a frame size below the protocol
    /// minimum.
    #[error("frame too short: total_length={0} (min {MIN_FRAME_LEN})")]
    FrameTooShort(u32),

    /// CRC over the prelude (first 8 bytes) did not match the
    /// stored value.
    #[error("prelude CRC mismatch (computed {computed:#010x}, header {expected:#010x})")]
    PreludeCrcMismatch {
        /// CRC value the decoder computed locally.
        computed: u32,
        /// CRC value carried in the prelude.
        expected: u32,
    },

    /// CRC over the full message did not match the trailing value.
    #[error("message CRC mismatch (computed {computed:#010x}, trailer {expected:#010x})")]
    MessageCrcMismatch {
        /// CRC value the decoder computed locally.
        computed: u32,
        /// CRC value carried in the message trailer.
        expected: u32,
    },

    /// A header value type byte was outside the documented `0..=9`
    /// range.
    #[error("unknown header value type: {0}")]
    UnknownHeaderType(u8),

    /// A length-prefixed field claimed more bytes than the frame
    /// supplied.
    #[error("frame underrun reading {context}")]
    Underrun {
        /// Human-readable site that ran out of bytes (e.g. "string
        /// value", "header name").
        context: &'static str,
    },

    /// A `String` header could not be decoded as UTF-8.
    #[error("non-UTF-8 string in header {0}")]
    NonUtf8String(String),
}

/// Incremental binary frame decoder.
#[derive(Default)]
pub struct EventStreamDecoder {
    buf: BytesMut,
}

impl EventStreamDecoder {
    /// Empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `chunk` to the internal buffer.
    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Try to parse the next complete frame. Returns `Ok(None)` when
    /// the buffer does not yet contain a full frame, `Ok(Some(_))` on
    /// success, or `Err(_)` when the next frame's prelude is
    /// malformed (in which case the decoder's state is corrupted —
    /// callers should treat the stream as failed).
    pub fn next_frame(
        &mut self,
    ) -> std::result::Result<Option<EventStreamFrame>, EventStreamParseError> {
        if self.buf.len() < PRELUDE_LEN {
            return Ok(None);
        }
        let total_length = u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]);
        let total_usize = total_length as usize;
        if total_usize < MIN_FRAME_LEN {
            return Err(EventStreamParseError::FrameTooShort(total_length));
        }
        if self.buf.len() < total_usize {
            return Ok(None);
        }
        let headers_length =
            u32::from_be_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;
        let prelude_crc_expected =
            u32::from_be_bytes([self.buf[8], self.buf[9], self.buf[10], self.buf[11]]);
        let prelude_crc_computed = crc32fast::hash(&self.buf[..8]);
        if prelude_crc_computed != prelude_crc_expected {
            return Err(EventStreamParseError::PreludeCrcMismatch {
                computed: prelude_crc_computed,
                expected: prelude_crc_expected,
            });
        }
        // Verify message CRC over [0, total - 4).
        let message_body_end = total_usize - MESSAGE_CRC_LEN;
        let message_crc_expected = u32::from_be_bytes([
            self.buf[message_body_end],
            self.buf[message_body_end + 1],
            self.buf[message_body_end + 2],
            self.buf[message_body_end + 3],
        ]);
        let message_crc_computed = crc32fast::hash(&self.buf[..message_body_end]);
        if message_crc_computed != message_crc_expected {
            return Err(EventStreamParseError::MessageCrcMismatch {
                computed: message_crc_computed,
                expected: message_crc_expected,
            });
        }
        // Slice headers and payload.
        let header_start = PRELUDE_LEN;
        let header_end = header_start + headers_length;
        let payload_start = header_end;
        let payload_end = message_body_end;

        let headers = parse_headers(&self.buf[header_start..header_end])?;
        let payload = self.buf[payload_start..payload_end].to_vec();
        let mut header_index = HashMap::with_capacity(headers.len());
        for h in &headers {
            header_index.insert(h.name.clone(), h.value.clone());
        }
        // Drop the consumed frame from the buffer.
        self.buf.advance(total_usize);
        Ok(Some(EventStreamFrame {
            headers,
            header_index,
            payload,
        }))
    }

    /// True when the internal buffer has bytes that have not yet
    /// completed a frame.
    pub fn has_residual(&self) -> bool {
        !self.buf.is_empty()
    }
}

fn parse_headers(
    mut bytes: &[u8],
) -> std::result::Result<Vec<EventStreamHeader>, EventStreamParseError> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let name_len = take_u8(&mut bytes, "header name length")? as usize;
        let name = take_str(&mut bytes, name_len, "header name")?;
        let type_byte = take_u8(&mut bytes, "header type")?;
        let value = parse_header_value(&mut bytes, type_byte, &name)?;
        out.push(EventStreamHeader { name, value });
    }
    Ok(out)
}

fn parse_header_value(
    bytes: &mut &[u8],
    type_byte: u8,
    header_name: &str,
) -> std::result::Result<EventStreamHeaderValue, EventStreamParseError> {
    match type_byte {
        TYPE_BOOL_TRUE => Ok(EventStreamHeaderValue::Bool(true)),
        TYPE_BOOL_FALSE => Ok(EventStreamHeaderValue::Bool(false)),
        TYPE_BYTE => {
            let v = take_u8(bytes, "byte value")? as i8;
            Ok(EventStreamHeaderValue::Byte(v))
        }
        TYPE_INT16 => {
            let v = take_n::<2>(bytes, "int16 value")?;
            Ok(EventStreamHeaderValue::Int16(i16::from_be_bytes(v)))
        }
        TYPE_INT32 => {
            let v = take_n::<4>(bytes, "int32 value")?;
            Ok(EventStreamHeaderValue::Int32(i32::from_be_bytes(v)))
        }
        TYPE_INT64 => {
            let v = take_n::<8>(bytes, "int64 value")?;
            Ok(EventStreamHeaderValue::Int64(i64::from_be_bytes(v)))
        }
        TYPE_BYTE_ARRAY => {
            let len_bytes = take_n::<2>(bytes, "byte array length")?;
            let len = u16::from_be_bytes(len_bytes) as usize;
            let payload = take_slice(bytes, len, "byte array value")?;
            Ok(EventStreamHeaderValue::Bytes(payload.to_vec()))
        }
        TYPE_STRING => {
            let len_bytes = take_n::<2>(bytes, "string length")?;
            let len = u16::from_be_bytes(len_bytes) as usize;
            let payload = take_slice(bytes, len, "string value")?;
            let s = std::str::from_utf8(payload)
                .map_err(|_| EventStreamParseError::NonUtf8String(header_name.to_owned()))?;
            Ok(EventStreamHeaderValue::String(s.to_owned()))
        }
        TYPE_TIMESTAMP => {
            let v = take_n::<8>(bytes, "timestamp value")?;
            Ok(EventStreamHeaderValue::Timestamp(i64::from_be_bytes(v)))
        }
        TYPE_UUID => {
            let v = take_n::<16>(bytes, "uuid value")?;
            Ok(EventStreamHeaderValue::Uuid(v))
        }
        other => Err(EventStreamParseError::UnknownHeaderType(other)),
    }
}

fn take_u8(
    bytes: &mut &[u8],
    context: &'static str,
) -> std::result::Result<u8, EventStreamParseError> {
    if bytes.is_empty() {
        return Err(EventStreamParseError::Underrun { context });
    }
    let v = bytes[0];
    *bytes = &bytes[1..];
    Ok(v)
}

fn take_n<const N: usize>(
    bytes: &mut &[u8],
    context: &'static str,
) -> std::result::Result<[u8; N], EventStreamParseError> {
    if bytes.len() < N {
        return Err(EventStreamParseError::Underrun { context });
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes[..N]);
    *bytes = &bytes[N..];
    Ok(out)
}

fn take_slice<'a>(
    bytes: &mut &'a [u8],
    n: usize,
    context: &'static str,
) -> std::result::Result<&'a [u8], EventStreamParseError> {
    if bytes.len() < n {
        return Err(EventStreamParseError::Underrun { context });
    }
    let (head, tail) = bytes.split_at(n);
    *bytes = tail;
    Ok(head)
}

fn take_str(
    bytes: &mut &[u8],
    n: usize,
    context: &'static str,
) -> std::result::Result<String, EventStreamParseError> {
    let slice = take_slice(bytes, n, context)?;
    let s = std::str::from_utf8(slice)
        .map_err(|_| EventStreamParseError::NonUtf8String(context.to_owned()))?;
    Ok(s.to_owned())
}

// ── Frame encoder (test-only helper, kept on the public API for
// integration tests and downstream tooling) ───────────────────────

/// Encode a single frame into AWS `vnd.amazon.eventstream` bytes.
/// Used by tests and any caller that needs to mint synthetic frames
/// for fixtures.
#[doc(hidden)]
pub fn encode_frame(headers: &[EventStreamHeader], payload: &[u8]) -> Vec<u8> {
    let mut header_bytes = Vec::new();
    for h in headers {
        let name_bytes = h.name.as_bytes();
        debug_assert!(name_bytes.len() <= u8::MAX as usize);
        #[allow(clippy::cast_possible_truncation)]
        let len = name_bytes.len() as u8;
        header_bytes.push(len);
        header_bytes.extend_from_slice(name_bytes);
        match &h.value {
            EventStreamHeaderValue::Bool(true) => header_bytes.push(TYPE_BOOL_TRUE),
            EventStreamHeaderValue::Bool(false) => header_bytes.push(TYPE_BOOL_FALSE),
            EventStreamHeaderValue::Byte(v) => {
                header_bytes.push(TYPE_BYTE);
                #[allow(clippy::cast_sign_loss)]
                header_bytes.push(*v as u8);
            }
            EventStreamHeaderValue::Int16(v) => {
                header_bytes.push(TYPE_INT16);
                header_bytes.extend_from_slice(&v.to_be_bytes());
            }
            EventStreamHeaderValue::Int32(v) => {
                header_bytes.push(TYPE_INT32);
                header_bytes.extend_from_slice(&v.to_be_bytes());
            }
            EventStreamHeaderValue::Int64(v) => {
                header_bytes.push(TYPE_INT64);
                header_bytes.extend_from_slice(&v.to_be_bytes());
            }
            EventStreamHeaderValue::Bytes(b) => {
                header_bytes.push(TYPE_BYTE_ARRAY);
                #[allow(clippy::cast_possible_truncation)]
                let len = b.len() as u16;
                header_bytes.extend_from_slice(&len.to_be_bytes());
                header_bytes.extend_from_slice(b);
            }
            EventStreamHeaderValue::String(s) => {
                header_bytes.push(TYPE_STRING);
                let bytes = s.as_bytes();
                #[allow(clippy::cast_possible_truncation)]
                let len = bytes.len() as u16;
                header_bytes.extend_from_slice(&len.to_be_bytes());
                header_bytes.extend_from_slice(bytes);
            }
            EventStreamHeaderValue::Timestamp(v) => {
                header_bytes.push(TYPE_TIMESTAMP);
                header_bytes.extend_from_slice(&v.to_be_bytes());
            }
            EventStreamHeaderValue::Uuid(u) => {
                header_bytes.push(TYPE_UUID);
                header_bytes.extend_from_slice(u);
            }
        }
    }
    let total_length =
        u32::try_from(PRELUDE_LEN + header_bytes.len() + payload.len() + MESSAGE_CRC_LEN)
            .expect("frame fits in u32");
    let headers_length = u32::try_from(header_bytes.len()).expect("headers fit in u32");

    let mut out = Vec::with_capacity(total_length as usize);
    out.extend_from_slice(&total_length.to_be_bytes());
    out.extend_from_slice(&headers_length.to_be_bytes());
    let prelude_crc = crc32fast::hash(&out[..8]);
    out.extend_from_slice(&prelude_crc.to_be_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(payload);
    let message_crc = crc32fast::hash(&out);
    out.extend_from_slice(&message_crc.to_be_bytes());
    out
}
