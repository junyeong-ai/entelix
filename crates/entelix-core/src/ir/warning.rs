//! `ModelWarning` — non-fatal codec advisories (invariant 6).
//!
//! When a codec drops or coerces information that the IR carried, it must
//! emit a `LossyEncode` here. Silent loss is an invariant violation.

use serde::{Deserialize, Serialize};

/// One non-fatal advisory from a codec or transport. Carried in
/// `ModelResponse::warnings` and surfaced via observability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelWarning {
    /// The codec could not preserve every IR field on the wire. The vendor
    /// will not see (or will see a coerced version of) the named field.
    LossyEncode {
        /// IR field that was dropped or coerced (e.g. `messages[2].content[0]`).
        field: String,
        /// Why the loss happened.
        detail: String,
    },
    /// The provider returned a stop reason this codec does not yet model;
    /// the IR carries it as `StopReason::Other`.
    UnknownStopReason {
        /// Raw vendor reason string.
        raw: String,
    },
    /// The IR requested a feature the model doesn't support (e.g. tools on a
    /// vision-only model). The codec proceeded with a degraded request.
    UnsupportedCapability {
        /// Capability name (e.g. `streaming`, `prompt_caching`).
        capability: String,
        /// Detail of what fallback was applied.
        detail: String,
    },
    /// The IR carries a [`ProviderExtensions`](crate::ir::ProviderExtensions)
    /// entry for a vendor different from the active codec — the
    /// codec ignored the foreign knobs because the wire format
    /// cannot express them. Operators that route the same request
    /// across multiple codecs see one of these warnings per inactive
    /// vendor that had ext set.
    ProviderExtensionIgnored {
        /// Vendor identifier matching the inactive ext field name
        /// (`anthropic`, `openai_chat`, `openai_responses`, `gemini`,
        /// `bedrock`).
        vendor: String,
    },
}
