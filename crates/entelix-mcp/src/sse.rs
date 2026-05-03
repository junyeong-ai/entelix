//! SSE-frame helpers shared by the streamable-http transport.
//!
//! Streamable HTTP carries every JSON-RPC message — both POST
//! responses and the long-lived `GET /` listener — as Server-Sent
//! Events. The frame format is the W3C SSE spec subset MCP relies
//! on:
//!
//! - Frames terminate on `\n\n` (or `\r\n\r\n`).
//! - The JSON-RPC payload rides on `data:` lines; multi-line `data:`
//!   entries concatenate with `\n` between them.
//! - All other lines (`event:`, `id:`, `:` comments) are ignored —
//!   MCP does not put significance on them.
//!
//! These helpers mirror the SSE handling already proven in the
//! Anthropic Messages codec (`entelix_core::codecs::anthropic`)
//! without reaching across crate boundaries; SSE parsing is
//! crate-internal infrastructure, not a public surface.

#![allow(clippy::redundant_pub_crate)]

/// Find the byte offset of the first `\n\n` (or `\r\n\r\n`) in
/// `buf`. Returns the index of the *first* terminator byte.
pub(crate) fn find_double_newline(buf: &[u8]) -> Option<usize> {
    let lf = buf.windows(2).position(|w| w == b"\n\n");
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Pull the JSON payload out of an SSE frame, concatenating
/// multi-line `data:` lines per spec. Returns `None` for frames
/// with no `data:` line (heartbeats, comment-only frames).
pub(crate) fn parse_sse_data(frame: &str) -> Option<String> {
    let mut out: Option<String> = None;
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            let trimmed = rest.strip_prefix(' ').unwrap_or(rest);
            match &mut out {
                Some(existing) => {
                    existing.push('\n');
                    existing.push_str(trimmed);
                }
                None => out = Some(trimmed.to_owned()),
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn double_newline_lf_only() {
        assert_eq!(find_double_newline(b"data: x\n\nrest"), Some(7));
    }

    #[test]
    fn double_newline_crlf() {
        assert_eq!(find_double_newline(b"data: x\r\n\r\nrest"), Some(7));
    }

    #[test]
    fn double_newline_picks_earliest() {
        // CRLF starts at byte 7 (\r\n\r\n), LF at byte 9 (the second \n\n).
        // The function should pick the earliest terminator.
        let buf = b"data: x\r\n\r\ny\n\n";
        assert_eq!(find_double_newline(buf), Some(7));
    }

    #[test]
    fn parse_data_strips_optional_leading_space() {
        let frame = "event: message\ndata: {\"a\":1}\n";
        assert_eq!(parse_sse_data(frame).as_deref(), Some("{\"a\":1}"));
    }

    #[test]
    fn parse_data_concatenates_multi_line_data() {
        let frame = "data: line1\ndata: line2\n";
        assert_eq!(parse_sse_data(frame).as_deref(), Some("line1\nline2"));
    }

    #[test]
    fn parse_data_returns_none_for_frame_without_data_line() {
        let frame = "event: ping\n: comment\n";
        assert!(parse_sse_data(frame).is_none());
    }
}
