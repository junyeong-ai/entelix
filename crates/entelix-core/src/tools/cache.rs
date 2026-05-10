//! Tool-block cache stamping strategy.
//!
//! [`crate::ir::CacheControl`] already lives on individual
//! [`crate::ir::ToolSpec`]s — what the registry adds is a strategy
//! knob that picks **which** specs in the materialised list get
//! marked. Operators set the mode once on the registry; every
//! [`crate::tools::ToolRegistry::tool_specs`] call thereafter
//! returns a list with the markers applied.
//!
//! ## Cross-vendor handling (invariant 22)
//!
//! - **Anthropic Messages**, **Bedrock Converse (Claude)**,
//!   **Vertex Anthropic** — native per-spec `cache_control`. The
//!   codec stamps each marker inline; the Anthropic 4-breakpoint
//!   total (system + tools + messages) is enforced at the codec
//!   edge.
//! - **OpenAI Chat / Responses**, **Gemini**, **Vertex Gemini** —
//!   no per-spec cache breakpoint. The codec emits
//!   [`crate::ir::ModelWarning::LossyEncode`] for the dropped
//!   marker; the operator leans on the vendor's automatic prefix
//!   caching where it exists.
//!
//! Why this is its own knob rather than a global "cache everything"
//! switch: system-prompt cache (`SystemPrompt::cached`) and
//! message-block cache live on their own surfaces. Operators need
//! independent control because Anthropic's 4-breakpoint budget is
//! shared across system + tools + messages, and a single coarse
//! switch would exhaust it on the tools half alone.

use crate::ir::{CacheControl, ToolSpec};

/// Cache-stamping mode for the tool block.
///
/// Resolved at materialisation time: the strategy lives on
/// [`crate::tools::ToolRegistry`] (or any caller that holds a spec
/// list) and translates into per-spec `cache_control` markers via
/// [`Self::apply`]. The mode itself never travels on the wire —
/// codecs see only the final [`ToolSpec::cache_control`] field.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ToolCacheMode {
    /// No cache stamping. Specs leave the registry exactly as
    /// `Tool::metadata().to_tool_spec()` produced them.
    #[default]
    None,
    /// Stamp `cache_control` on the **last** spec only — Anthropic
    /// reads this as a single cache prefix covering everything up
    /// to and including that spec (the system block, every
    /// preceding tool, and this last tool). Cheapest in
    /// breakpoints; correct for the 99% case where operators want
    /// "the tool block cached".
    Suffix(CacheControl),
    /// Stamp `cache_control` on **every** spec. Burns N
    /// breakpoints — vendors cap at their own limit (Anthropic = 4
    /// across system + tools + messages) and codecs emit
    /// [`crate::ir::ModelWarning::LossyEncode`] when overflow
    /// drops a marker. Useful for partial-prefix re-use scenarios
    /// where the operator wants distinct cache anchors at every
    /// tool boundary.
    PerSpec(CacheControl),
}

impl ToolCacheMode {
    /// Apply this mode to a spec list, returning the list with
    /// `cache_control` set per the variant's contract. Pure: no
    /// allocation when [`Self::None`], in-place mutation through
    /// the owned `Vec` for the marking variants.
    #[must_use]
    pub fn apply(self, mut specs: Vec<ToolSpec>) -> Vec<ToolSpec> {
        match self {
            Self::None => specs,
            Self::Suffix(cache) => {
                if let Some(last) = specs.last_mut() {
                    last.cache_control = Some(cache);
                }
                specs
            }
            Self::PerSpec(cache) => {
                for spec in &mut specs {
                    spec.cache_control = Some(cache);
                }
                specs
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fake(name: &str) -> ToolSpec {
        ToolSpec::function(
            name.to_owned(),
            format!("{name} description"),
            json!({"type": "object"}),
        )
    }

    #[test]
    fn none_leaves_specs_unchanged() {
        let specs = vec![fake("alpha"), fake("beta")];
        let out = ToolCacheMode::None.apply(specs);
        assert!(out.iter().all(|s| s.cache_control.is_none()));
    }

    #[test]
    fn suffix_marks_only_the_last_spec() {
        let cache = CacheControl::five_minutes();
        let specs = vec![fake("alpha"), fake("beta"), fake("gamma")];
        let out = ToolCacheMode::Suffix(cache).apply(specs);
        assert!(out[0].cache_control.is_none());
        assert!(out[1].cache_control.is_none());
        assert_eq!(out[2].cache_control, Some(cache));
    }

    #[test]
    fn per_spec_marks_every_spec() {
        let cache = CacheControl::one_hour();
        let specs = vec![fake("alpha"), fake("beta"), fake("gamma")];
        let out = ToolCacheMode::PerSpec(cache).apply(specs);
        assert!(out.iter().all(|s| s.cache_control == Some(cache)));
    }

    #[test]
    fn apply_on_empty_input_is_a_noop() {
        let out = ToolCacheMode::Suffix(CacheControl::five_minutes()).apply(Vec::new());
        assert!(out.is_empty());
        let out = ToolCacheMode::PerSpec(CacheControl::one_hour()).apply(Vec::new());
        assert!(out.is_empty());
    }
}
