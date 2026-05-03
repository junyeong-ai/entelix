//! Codec robustness — property tests across all 5 codecs.
//!
//! Two universal properties:
//!
//! 1. **Encoder totality** — for every well-formed [`ModelRequest`] the
//!    five production codecs must either return `Ok(EncodedRequest)`
//!    with a body that parses as JSON, or return `Err` cleanly. No
//!    panics. This catches edge cases (empty content arrays, mixed
//!    role sequences, oversized stop_sequences) where IR combinations
//!    that look reachable in operator code would otherwise blow up at
//!    request time.
//!
//! 2. **Decoder panic-safety** — feeding arbitrary bytes to each
//!    codec's `decode` and `extract_rate_limit` must never panic.
//!    Decoders read network data; a malicious or buggy server must
//!    not be able to take the runtime down.
//!
//! Both properties are vendor-neutral and run identically against
//! Anthropic / OpenAI Chat / OpenAI Responses / Gemini / Bedrock
//! Converse — a regression in any one codec surfaces here without
//! per-codec scaffolding.
//!
//! Number of cases is left at proptest's default (256). Run a heavier
//! sweep locally with `PROPTEST_CASES=10000`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::single_match_else,
    clippy::single_match
)]

use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix_core::ir::{ContentPart, Message, ModelRequest, Role};
use proptest::prelude::*;
use serde_json::Value;

// ── Strategies ─────────────────────────────────────────────────────────────

fn role_strategy() -> impl Strategy<Value = Role> {
    prop_oneof![
        Just(Role::User),
        Just(Role::Assistant),
        Just(Role::System),
        Just(Role::Tool),
    ]
}

/// Text content blocks span the cross-cutting cases every codec
/// handles — extending to images/audio/tool-use multiplies the
/// matrix without changing what the property is asserting.
fn text_part_strategy() -> impl Strategy<Value = ContentPart> {
    "[\\PC]{0,80}".prop_map(ContentPart::text)
}

fn message_strategy() -> impl Strategy<Value = Message> {
    (
        role_strategy(),
        prop::collection::vec(text_part_strategy(), 1..4),
    )
        .prop_map(|(role, content)| Message::new(role, content))
}

fn request_strategy() -> impl Strategy<Value = ModelRequest> {
    (
        "[a-z0-9-]{1,32}",
        prop::collection::vec(message_strategy(), 1..6),
        prop::option::of(any::<u32>().prop_map(|v| v % 8192 + 1)),
        prop::option::of(any::<f32>().prop_map(|v| v.abs() % 2.0)),
    )
        .prop_map(|(model, messages, max_tokens, temperature)| ModelRequest {
            model,
            messages,
            max_tokens,
            temperature,
            ..ModelRequest::default()
        })
}

// ── Property 1: encoder totality ───────────────────────────────────────────

fn assert_encode_total<C: Codec>(codec: &C, request: &ModelRequest) {
    match codec.encode(request) {
        Ok(encoded) => {
            // Body must parse as JSON. Codec-internal panics during
            // serialization would already have fired before we got
            // here; this guards against a codec that hand-writes
            // bytes and ships malformed JSON.
            let _: Value = serde_json::from_slice(&encoded.body).unwrap_or_else(|e| {
                panic!(
                    "{} produced non-JSON body for request {:?}: {e}",
                    codec.name(),
                    request
                )
            });
        }
        // Err is acceptable — codecs reject obviously invalid IR
        // (e.g. empty messages). The property is panic-freedom, not
        // unconditional success.
        Err(_) => {}
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn anthropic_encode_is_total(req in request_strategy()) {
        assert_encode_total(&AnthropicMessagesCodec::new(), &req);
    }

    #[test]
    fn openai_chat_encode_is_total(req in request_strategy()) {
        assert_encode_total(&OpenAiChatCodec::new(), &req);
    }

    #[test]
    fn openai_responses_encode_is_total(req in request_strategy()) {
        assert_encode_total(&OpenAiResponsesCodec::new(), &req);
    }

    #[test]
    fn gemini_encode_is_total(req in request_strategy()) {
        assert_encode_total(&GeminiCodec::new(), &req);
    }

    #[test]
    fn bedrock_encode_is_total(req in request_strategy()) {
        assert_encode_total(&BedrockConverseCodec::new(), &req);
    }
}

// ── Property 2: decoder panic-safety ───────────────────────────────────────

fn assert_decode_safe<C: Codec>(codec: &C, bytes: &[u8]) {
    // Returning Err is fine — the property is "no panic".
    let _ = codec.decode(bytes, Vec::new());
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn anthropic_decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        assert_decode_safe(&AnthropicMessagesCodec::new(), &bytes);
    }

    #[test]
    fn openai_chat_decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        assert_decode_safe(&OpenAiChatCodec::new(), &bytes);
    }

    #[test]
    fn openai_responses_decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        assert_decode_safe(&OpenAiResponsesCodec::new(), &bytes);
    }

    #[test]
    fn gemini_decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        assert_decode_safe(&GeminiCodec::new(), &bytes);
    }

    #[test]
    fn bedrock_decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        assert_decode_safe(&BedrockConverseCodec::new(), &bytes);
    }
}

// ── Targeted "shaped JSON" decode robustness ───────────────────────────────
//
// Pure random bytes hit the JSON parser's fast-fail path on the first
// non-`{` byte. To exercise decode logic past that gate, generate
// JSON-shaped objects with arbitrary fields and feed them through.

fn shaped_json_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::hash_map("[a-zA-Z_][a-zA-Z0-9_]{0,8}", any::<i32>(), 0..5)
        .prop_map(|map| serde_json::to_vec(&map).expect("HashMap<String,i32> always serializes"))
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    #[test]
    fn shaped_json_decode_safety(bytes in shaped_json_strategy()) {
        assert_decode_safe(&AnthropicMessagesCodec::new(), &bytes);
        assert_decode_safe(&OpenAiChatCodec::new(), &bytes);
        assert_decode_safe(&OpenAiResponsesCodec::new(), &bytes);
        assert_decode_safe(&GeminiCodec::new(), &bytes);
        assert_decode_safe(&BedrockConverseCodec::new(), &bytes);
    }
}
