//! `ProviderEchoSnapshot` — vendor-keyed opaque round-trip carrier.
//!
//! Vendors emit per-turn opaque tokens the harness must echo verbatim
//! on the next request. Each vendor names and shapes the token
//! differently, but the contract is identical: *opaque to the harness,
//! mandatory on the wire, lost = next turn rejected or degraded*. This
//! module ships the cross-vendor IR carrier that all such tokens ride
//! on.
//!
//! ## Vendors that produce opaque round-trip tokens (2026-05)
//!
//! | Vendor | Token | Wire location | Echo failure mode |
//! |---|---|---|---|
//! | Anthropic Messages | `signature` | `content[].signature` (`type: "thinking"`) | Tool round → HTTP 400; non-tool round → silent thinking-disabled |
//! | Anthropic Messages | `redacted_thinking.data` | `content[].data` | Same as `signature` (full block round-trip) |
//! | Anthropic on Bedrock Converse | `reasoningText.signature` + `redactedContent` | `content[].reasoningContent.*` | Hash-validated; any tampering throws |
//! | Amazon Nova 2 (Bedrock Converse) | `signature` (Anthropic-on-Bedrock shape) | `content[].reasoningContent.reasoningText.signature` | Required for extended-thinking continuity |
//! | Anthropic on Vertex AI | `signature` (wire-shape identical to first-party) | `content[].signature` | Same as first-party Anthropic |
//! | Gemini (AI Studio + Vertex) 3.x | `thought_signature` (snake_case strict on Vertex) | `Part.thought_signature` on `text` / `functionCall` / thinking parts | First `functionCall` of step missing → HTTP 400 |
//! | OpenAI Responses | `encrypted_content` | `output[].encrypted_content` | `store: false` + dropped → silent CoT loss |
//! | OpenAI Responses | `previous_response_id` | request root | Optional alternative to manual echo |
//! | OpenAI Responses | item `id` (`rs_…` / `fc_…` / `msg_…`) | per-item | Required when echoing prior output items in stateless mode |
//! | xAI Grok 4.x | `reasoning.encrypted_content` | OpenAI-Responses-shaped | Same as OpenAI Responses |
//!
//! Five distinct native vendors satisfy invariant 22 — promote to
//! cross-vendor IR.
//!
//! ## Invariants
//!
//! - **Codec autonomy.** Each codec reads and writes only entries
//!   whose `provider` matches its own
//!   [`Codec::name`](crate::codecs::Codec::name). Cross-codec
//!   passthrough is a structural property: a transcript with
//!   entries for multiple vendors round-trips through any codec
//!   without affecting wire bytes. Enforced by convention + the
//!   `cross_vendor_*_isolation_*` regression suite in
//!   `tests/provider_echo_round_trip.rs`; the type itself is
//!   open-constructable so external codec crates can stamp their
//!   own provider key without a sealed-trait bottleneck (a
//!   prerequisite for invariant 22's "new vendor = one codec impl,
//!   zero IR change" promise).
//! - **Harness never inspects.** The harness forwards whichever
//!   blob lands at decode time, untouched. The single-emit
//!   invariant — at most one entry per `(part, provider key)` pair
//!   — is documented on [`Self::find_in`].
//! - **`Vec`, not `Option`.** A transcript that has crossed
//!   transports may carry blobs from multiple vendors
//!   simultaneously; the IR is an audit-faithful record, not a
//!   single-vendor projection.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Vendor-issued opaque data the harness must echo verbatim on the
/// next turn — Anthropic `signature`, Anthropic `redacted_thinking.data`,
/// Gemini `thought_signature`, OpenAI `encrypted_content`, OpenAI item
/// ids, Bedrock `reasoningText.signature` / `redactedContent`, Nova 2
/// `signature`, xAI `encrypted_content`, OpenAI `code_interpreter`
/// `container_id`.
///
/// Codecs decode their own vendor's blob into this carrier and read
/// it back verbatim on the encode side. The harness never inspects
/// the payload.
///
/// Cross-vendor: a transcript may carry entries for multiple vendors
/// after a transport switch; each codec only emits its own entries
/// on the wire and silently leaves the other vendor's blob alone in
/// IR.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderEchoSnapshot {
    /// Provider key matching [`Codec::name`](crate::codecs::Codec::name)
    /// (`"anthropic-messages"`, `"gemini"`, `"openai-responses"`,
    /// `"bedrock-converse"`, `"vertex-anthropic"`, `"vertex-gemini"`).
    /// Codecs select their own blobs by this key.
    pub provider: Cow<'static, str>,
    /// Vendor-defined opaque payload. Typically a JSON object whose
    /// keys are the vendor's wire field names (`signature`,
    /// `thought_signature`, `encrypted_content`, `data`,
    /// `redacted_content`, `container_id`, `id`). The harness never
    /// reads or rewrites these keys — only the originating codec
    /// does.
    pub payload: serde_json::Value,
}

impl ProviderEchoSnapshot {
    /// Construct a snapshot with an arbitrary payload value.
    #[must_use]
    pub fn new(provider: impl Into<Cow<'static, str>>, payload: serde_json::Value) -> Self {
        Self {
            provider: provider.into(),
            payload,
        }
    }

    /// Construct a snapshot whose payload is a single-key JSON object
    /// `{ <field>: <value> }`. The common shape for vendors that emit
    /// one opaque field per part (`{ "signature": "…" }`,
    /// `{ "thought_signature": "…" }`).
    #[must_use]
    pub fn for_provider(
        provider: impl Into<Cow<'static, str>>,
        field: &str,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        let mut map = serde_json::Map::with_capacity(1);
        map.insert(field.to_owned(), value.into());
        Self {
            provider: provider.into(),
            payload: serde_json::Value::Object(map),
        }
    }

    /// Borrow a single field from the payload object, when the payload
    /// is a JSON object and the field exists. Returns `None` for any
    /// other shape — codecs use this on encode to look up their own
    /// wire field.
    #[must_use]
    pub fn payload_field(&self, field: &str) -> Option<&serde_json::Value> {
        self.payload.as_object().and_then(|obj| obj.get(field))
    }

    /// Borrow a single string field from the payload object — the
    /// hot-path accessor for opaque-token-as-string vendors
    /// (Anthropic `signature`, Gemini `thought_signature`, OpenAI
    /// `encrypted_content`).
    #[must_use]
    pub fn payload_str(&self, field: &str) -> Option<&str> {
        self.payload_field(field)
            .and_then(serde_json::Value::as_str)
    }

    /// Find the first snapshot in `echoes` whose `provider` matches
    /// `name`. Codecs call this on the encode side to recover their
    /// own opaque blob from a part's cross-vendor carrier; `None`
    /// means no entry for this codec, so no wire-side opaque field
    /// is emitted.
    ///
    /// **Codec single-emit invariant**: a codec MUST attach at most
    /// one entry per `(part, provider key)` pair on decode. Multiple
    /// matches in `echoes` would indicate the IR was hand-built or a
    /// codec violated the invariant — `find_in` returns the first
    /// match, leaving subsequent entries inert.
    #[must_use]
    pub fn find_in<'a>(echoes: &'a [Self], name: &str) -> Option<&'a Self> {
        echoes.iter().find(|e| e.provider.as_ref() == name)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip_serde() {
        let snap =
            ProviderEchoSnapshot::for_provider("anthropic-messages", "signature", "WaUjzkypQ2…");
        let s = serde_json::to_string(&snap).unwrap();
        let back: ProviderEchoSnapshot = serde_json::from_str(&s).unwrap();
        assert_eq!(snap, back);
    }

    #[test]
    fn payload_field_lookup() {
        let snap = ProviderEchoSnapshot::for_provider("gemini", "thought_signature", "EhsM…");
        assert_eq!(snap.payload_str("thought_signature"), Some("EhsM…"));
        assert_eq!(snap.payload_str("missing"), None);
    }

    #[test]
    fn payload_field_handles_non_object() {
        let snap = ProviderEchoSnapshot::new("x", json!("scalar"));
        assert_eq!(snap.payload_field("anything"), None);
    }

    #[test]
    fn find_picks_first_match() {
        let echoes = vec![
            ProviderEchoSnapshot::for_provider("anthropic-messages", "signature", "a"),
            ProviderEchoSnapshot::for_provider("gemini", "thought_signature", "g"),
        ];
        assert_eq!(
            ProviderEchoSnapshot::find_in(&echoes, "anthropic-messages")
                .and_then(|s| s.payload_str("signature")),
            Some("a"),
        );
        assert_eq!(
            ProviderEchoSnapshot::find_in(&echoes, "gemini")
                .and_then(|s| s.payload_str("thought_signature")),
            Some("g"),
        );
        assert!(ProviderEchoSnapshot::find_in(&echoes, "openai-responses").is_none());
    }
}
