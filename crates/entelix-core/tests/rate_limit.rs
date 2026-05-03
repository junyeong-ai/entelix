//! `RateLimitSnapshot` extraction across providers + integration with
//! `ChatModel::complete_full` and `stream_deltas`.

#![allow(clippy::unwrap_used)]

use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix_core::rate_limit::RateLimitSnapshot;
use http::HeaderMap;

fn header_map(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, v) in pairs {
        h.insert(
            http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
            http::HeaderValue::from_str(v).unwrap(),
        );
    }
    h
}

#[test]
fn anthropic_extracts_remaining_and_reset() {
    let codec = AnthropicMessagesCodec::new();
    let headers = header_map(&[
        ("anthropic-ratelimit-requests-remaining", "42"),
        ("anthropic-ratelimit-tokens-remaining", "9876"),
        ("anthropic-ratelimit-requests-reset", "2026-04-26T12:00:00Z"),
        ("anthropic-ratelimit-tokens-reset", "2026-04-26T12:01:00Z"),
    ]);
    let snap = codec.extract_rate_limit(&headers).unwrap();
    assert_eq!(snap.requests_remaining, Some(42));
    assert_eq!(snap.tokens_remaining, Some(9876));
    assert!(snap.requests_reset_at.is_some());
    assert!(snap.tokens_reset_at.is_some());
    assert!(
        snap.raw
            .contains_key("anthropic-ratelimit-requests-remaining")
    );
}

#[test]
fn openai_chat_extracts_remaining() {
    let codec = OpenAiChatCodec::new();
    let headers = header_map(&[
        ("x-ratelimit-remaining-requests", "100"),
        ("x-ratelimit-remaining-tokens", "50000"),
        ("x-ratelimit-reset-requests", "6m0s"),
    ]);
    let snap = codec.extract_rate_limit(&headers).unwrap();
    assert_eq!(snap.requests_remaining, Some(100));
    assert_eq!(snap.tokens_remaining, Some(50_000));
    assert!(snap.raw.contains_key("x-ratelimit-reset-requests"));
}

#[test]
fn openai_responses_uses_same_extractor_as_chat() {
    let codec = OpenAiResponsesCodec::new();
    let headers = header_map(&[("x-ratelimit-remaining-requests", "7")]);
    let snap = codec.extract_rate_limit(&headers).unwrap();
    assert_eq!(snap.requests_remaining, Some(7));
}

#[test]
fn gemini_returns_none_when_no_rate_limit_header() {
    let codec = GeminiCodec::new();
    let headers = header_map(&[("content-type", "application/json")]);
    assert!(codec.extract_rate_limit(&headers).is_none());
}

#[test]
fn bedrock_returns_none_for_default_impl() {
    let codec = BedrockConverseCodec::new();
    let headers = header_map(&[]);
    assert!(codec.extract_rate_limit(&headers).is_none());
}

#[test]
fn empty_headers_yield_none_for_anthropic() {
    let codec = AnthropicMessagesCodec::new();
    assert!(codec.extract_rate_limit(&header_map(&[])).is_none());
}

#[test]
fn unparseable_count_is_dropped_silently() {
    let codec = AnthropicMessagesCodec::new();
    let headers = header_map(&[("anthropic-ratelimit-requests-remaining", "not-a-number")]);
    assert!(codec.extract_rate_limit(&headers).is_none());
}

#[test]
fn approaching_limit_heuristic_fires_under_floor() {
    let snap = RateLimitSnapshot {
        requests_remaining: Some(3),
        tokens_remaining: Some(50_000),
        ..RateLimitSnapshot::default()
    };
    assert!(snap.is_approaching_limit(0.1));

    let healthy = RateLimitSnapshot {
        requests_remaining: Some(1000),
        tokens_remaining: Some(1_000_000),
        ..RateLimitSnapshot::default()
    };
    assert!(!healthy.is_approaching_limit(0.1));
}

#[test]
fn approaching_limit_returns_false_when_no_counters() {
    let snap = RateLimitSnapshot::default();
    assert!(!snap.is_approaching_limit(0.5));
}
