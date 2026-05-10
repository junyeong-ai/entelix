//! `PiiRedactor` trait + [`RegexRedactor`].
//!
//! F5 mitigation: redaction is **bidirectional** — the same redactor
//! runs at `pre_request` (outbound scrub of user-supplied messages
//! before they're sent to the model) *and* `post_response` (inbound
//! scrub of model-generated content before it's stored or surfaced
//! back to the user). Either direction alone leaks; both together
//! close the gap.

use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;

use entelix_core::ir::{ContentPart, ModelRequest, ModelResponse};

use crate::error::{PolicyError, PolicyResult};

/// Bidirectional PII redaction surface.
///
/// Implementations cover three surfaces — model requests / model
/// responses (the F5 bidirectional pair) plus tool-call JSON
/// payloads (input + output). `PolicyLayer` wires this trait into
/// both `Service<ModelInvocation>` and `Service<ToolInvocation>`.
#[async_trait]
pub trait PiiRedactor: Send + Sync + 'static {
    /// Scrub the outbound request in-place. Walks every text content
    /// part of every message and rewrites matching substrings with
    /// the configured replacement.
    async fn redact_request(&self, request: &mut ModelRequest) -> PolicyResult<()>;

    /// Scrub the inbound response in-place. Walks every text content
    /// part of the response.
    async fn redact_response(&self, response: &mut ModelResponse) -> PolicyResult<()>;

    /// Scrub a JSON payload in-place. Walks every string leaf and
    /// rewrites matching substrings. Used by `PolicyLayer` for tool
    /// invocations (both `input` and `output` JSON go through here).
    async fn redact_json(&self, value: &mut serde_json::Value) -> PolicyResult<()>;
}

/// One named redaction pattern.
#[derive(Clone, Debug)]
pub struct PiiPattern {
    /// Diagnostic name (e.g. `"email"`, `"us_ssn"`). Surfaced in
    /// telemetry; not exposed to the model.
    pub name: &'static str,
    /// Regex matching one PII instance.
    pub regex: Regex,
    /// String the matched substring is replaced with. Conventionally
    /// `"[REDACTED:<name>]"` so the model can still reason about the
    /// shape ("a number was here") without seeing the value.
    pub replacement: String,
    /// Optional secondary check applied to each regex match. If set
    /// and returns `false`, the match is *not* redacted — useful for
    /// patterns prone to false positives (a 16-digit run that fails
    /// the Luhn checksum is almost certainly not a real card number).
    /// `None` redacts every regex hit.
    pub validator: Option<fn(&str) -> bool>,
    /// When `true`, additionally require that the character
    /// immediately preceding each match is **not** a word
    /// character (`[A-Za-z0-9_]`). Use for patterns that cannot
    /// rely on a leading regex `\b` because the first character
    /// of the regex is itself non-word (`+`, `(`) — without this
    /// check the regex would match in the middle of identifiers
    /// like `order_id_+1-415-555-0199`. Default `false`.
    pub left_word_boundary: bool,
}

impl PartialEq for PiiPattern {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.regex.as_str() == other.regex.as_str()
            && self.replacement == other.replacement
            && self.left_word_boundary == other.left_word_boundary
            // Function pointers compare by address — equality only
            // holds when both sides reference the same `fn` item.
            && match (self.validator, other.validator) {
                (None, None) => true,
                (Some(a), Some(b)) => std::ptr::fn_addr_eq(a, b),
                _ => false,
            }
    }
}

/// Luhn-checksum validator — reject candidate runs that aren't
/// well-formed payment-card numbers. Strips spaces, dashes, and
/// non-digit noise before computing.
#[must_use]
pub fn luhn_valid(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    // Standard PAN length range is 13–19; the regex narrows further but
    // we double-check here so a custom regex with broader bounds still
    // gets sane behavior.
    if !(13..=19).contains(&digits.len()) {
        return false;
    }
    let mut sum: u32 = 0;
    for (i, d) in digits.iter().rev().enumerate() {
        let mut v = *d;
        if i % 2 == 1 {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
    }
    sum.is_multiple_of(10)
}

/// A small starter set of PII patterns. Production deployments
/// almost always extend or replace these per jurisdiction; this
/// list exists so a `RegexRedactor::default()` is non-trivial out
/// of the box.
///
/// The patterns are intentionally permissive (some false positives
/// are preferable to silent leaks). Tighten per environment.
pub fn default_pii_patterns() -> Vec<PiiPattern> {
    fn re(s: &str) -> Regex {
        // Patterns below are static literals validated by the unit
        // test `default_patterns_compile`; `unwrap` is safe.
        #[allow(clippy::unwrap_used)]
        Regex::new(s).unwrap()
    }
    vec![
        PiiPattern {
            name: "email",
            regex: re(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
            replacement: "[REDACTED:email]".into(),
            validator: None,
            left_word_boundary: false,
        },
        PiiPattern {
            name: "us_ssn",
            regex: re(r"\b\d{3}-\d{2}-\d{4}\b"),
            replacement: "[REDACTED:us_ssn]".into(),
            validator: None,
            left_word_boundary: false,
        },
        PiiPattern {
            name: "credit_card",
            // 13–19 digit groups with optional space/dash separators.
            // Anchored on word boundaries; the Luhn validator rejects
            // false positives (long order numbers, trackers, ISBNs).
            regex: re(r"\b(?:\d[ -]?){12,18}\d\b"),
            replacement: "[REDACTED:credit_card]".into(),
            validator: Some(luhn_valid),
            left_word_boundary: false,
        },
        PiiPattern {
            name: "phone",
            // Require an explicit phone marker (`+CC` country prefix
            // OR `(area)` parentheses) — bare digit runs collide
            // with order numbers, tracker IDs, etc.
            //
            // No leading `\b` — `+` / `(` are non-word chars; `\b`
            // requires a word/non-word transition that fails when
            // the preceding char is whitespace. `left_word_boundary`
            // below adds the manual check the regex can't express
            // (Rust regex has no lookbehind), preventing matches
            // mid-identifier (e.g. `order-id-+1-415-555-0199`).
            regex: re(r"(?:\+\d{1,3}\s*\(?\d{2,4}\)?|\(\d{2,4}\))[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b"),
            replacement: "[REDACTED:phone]".into(),
            validator: None,
            left_word_boundary: true,
        },
    ]
}

/// Regex-driven PII redactor.
pub struct RegexRedactor {
    patterns: Arc<Vec<PiiPattern>>,
}

impl std::fmt::Debug for RegexRedactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegexRedactor")
            .field(
                "patterns",
                &self.patterns.iter().map(|p| p.name).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl RegexRedactor {
    /// Build with the supplied patterns. Order matters — earlier
    /// patterns run first, so a credit-card regex placed before a
    /// generic digit-run regex catches the structured cases first.
    pub fn new(patterns: Vec<PiiPattern>) -> Self {
        Self {
            patterns: Arc::new(patterns),
        }
    }

    /// Build with [`default_pii_patterns`].
    pub fn with_defaults() -> Self {
        Self::new(default_pii_patterns())
    }

    /// Compile a `(name, regex_str, replacement)` tuple list. Returns
    /// [`PolicyError::Config`] when any regex fails to compile.
    pub fn from_strs(
        triples: impl IntoIterator<Item = (&'static str, &'static str, String)>,
    ) -> PolicyResult<Self> {
        let mut patterns = Vec::new();
        for (name, pat, repl) in triples {
            let regex = Regex::new(pat)
                .map_err(|e| PolicyError::Config(format!("invalid PII pattern '{name}': {e}")))?;
            patterns.push(PiiPattern {
                name,
                regex,
                replacement: repl,
                validator: None,
                left_word_boundary: false,
            });
        }
        Ok(Self::new(patterns))
    }

    /// Number of configured patterns.
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    fn redact_text(&self, text: &mut String) {
        for pat in self.patterns.iter() {
            // Fast path: regex-only, no validator and no left-boundary
            // requirement — let `replace_all` do the work and return
            // a borrowed Cow when nothing matched.
            if pat.validator.is_none() && !pat.left_word_boundary {
                if let std::borrow::Cow::Owned(replaced) =
                    pat.regex.replace_all(text, pat.replacement.as_str())
                {
                    *text = replaced;
                }
                continue;
            }
            // Slow path: walk matches, applying validator and/or
            // manual left-boundary check per hit.
            let bytes = text.as_bytes();
            let mut rewritten = String::with_capacity(text.len());
            let mut last_end = 0;
            let mut changed = false;
            for m in pat.regex.find_iter(text) {
                rewritten.push_str(&text[last_end..m.start()]);
                let mut keep = true;
                if pat.left_word_boundary && m.start() > 0 {
                    // Bytes are safe to inspect at index `start-1`
                    // because `find_iter` returns char-aligned
                    // boundaries and we only check ASCII-class
                    // properties. `bytes.get` keeps clippy
                    // `indexing_slicing` happy without changing
                    // semantics — `m.start() > 0` already gates.
                    if let Some(&prev) = bytes.get(m.start().saturating_sub(1))
                        && (prev.is_ascii_alphanumeric() || prev == b'_')
                    {
                        keep = false;
                    }
                }
                if keep && let Some(check) = pat.validator {
                    keep = check(m.as_str());
                }
                if keep {
                    rewritten.push_str(pat.replacement.as_str());
                    changed = true;
                } else {
                    rewritten.push_str(m.as_str());
                }
                last_end = m.end();
            }
            if changed {
                rewritten.push_str(&text[last_end..]);
                *text = rewritten;
            }
        }
    }

    fn redact_parts(&self, parts: &mut [ContentPart]) {
        for part in parts {
            if let ContentPart::Text { text, .. } = part {
                self.redact_text(text);
            }
        }
    }
}

impl Default for RegexRedactor {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl RegexRedactor {
    fn redact_json_value(&self, value: &mut serde_json::Value) {
        match value {
            serde_json::Value::String(s) => self.redact_text(s),
            serde_json::Value::Array(arr) => {
                for v in arr {
                    self.redact_json_value(v);
                }
            }
            serde_json::Value::Object(obj) => {
                for (_, v) in obj.iter_mut() {
                    self.redact_json_value(v);
                }
            }
            // Numbers, bools, nulls have no PII surface.
            _ => {}
        }
    }
}

#[async_trait]
impl PiiRedactor for RegexRedactor {
    async fn redact_request(&self, request: &mut ModelRequest) -> PolicyResult<()> {
        for block in request.system.blocks_mut() {
            self.redact_text(&mut block.text);
        }
        for msg in &mut request.messages {
            self.redact_parts(&mut msg.content);
        }
        Ok(())
    }

    async fn redact_response(&self, response: &mut ModelResponse) -> PolicyResult<()> {
        self.redact_parts(&mut response.content);
        Ok(())
    }

    async fn redact_json(&self, value: &mut serde_json::Value) -> PolicyResult<()> {
        self.redact_json_value(value);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use entelix_core::ir::Usage;
    use entelix_core::ir::{ContentPart, Message, ModelRequest, ModelResponse, Role, StopReason};

    use super::*;

    fn req(text: &str) -> ModelRequest {
        ModelRequest {
            model: "x".into(),
            messages: vec![Message::user(text)],
            ..ModelRequest::default()
        }
    }

    fn resp(text: &str) -> ModelResponse {
        ModelResponse {
            id: "r".into(),
            model: "x".into(),
            stop_reason: StopReason::EndTurn,
            content: vec![ContentPart::text(text)],
            usage: Usage::default(),
            rate_limit: None,
            warnings: Vec::new(),
            provider_echoes: Vec::new(),
        }
    }

    fn extract_text(parts: &[ContentPart]) -> String {
        parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    #[tokio::test]
    async fn redacts_email_in_request() {
        let redactor = RegexRedactor::with_defaults();
        let mut request = req("contact me at jane.doe@example.com please");
        redactor.redact_request(&mut request).await.unwrap();
        let txt = extract_text(&request.messages[0].content);
        assert!(txt.contains("[REDACTED:email]"));
        assert!(!txt.contains("jane.doe@example.com"));
    }

    #[tokio::test]
    async fn redacts_email_in_response() {
        let redactor = RegexRedactor::with_defaults();
        let mut response = resp("Sure — write to support@acme.io for help.");
        redactor.redact_response(&mut response).await.unwrap();
        let txt = extract_text(&response.content);
        assert!(txt.contains("[REDACTED:email]"));
    }

    #[tokio::test]
    async fn redacts_ssn_and_phone() {
        let redactor = RegexRedactor::with_defaults();
        let mut request = req("SSN 123-45-6789, call +1 (415) 555-0199");
        redactor.redact_request(&mut request).await.unwrap();
        let txt = extract_text(&request.messages[0].content);
        assert!(txt.contains("[REDACTED:us_ssn]"), "{txt}");
        assert!(txt.contains("[REDACTED:phone]"), "{txt}");
    }

    #[tokio::test]
    async fn redacts_system_prompt() {
        let redactor = RegexRedactor::with_defaults();
        let mut request = ModelRequest {
            system: "admin@acme.io owns this".into(),
            ..req("hi")
        };
        redactor.redact_request(&mut request).await.unwrap();
        assert!(request.system.concat_text().contains("[REDACTED:email]"));
    }

    #[tokio::test]
    async fn non_text_parts_pass_through_untouched() {
        let redactor = RegexRedactor::with_defaults();
        let mut request = ModelRequest {
            messages: vec![Message::new(
                Role::User,
                vec![ContentPart::ToolUse {
                    id: "tu_1".into(),
                    name: "lookup".into(),
                    input: serde_json::json!({"email": "leaks@example.com"}),
                    provider_echoes: Vec::new(),
                }],
            )],
            ..req("")
        };
        redactor.redact_request(&mut request).await.unwrap();
        // ToolUse part is not text — redactor doesn't reach into JSON
        // arguments by design (tool inputs travel through `pre_tool_call`
        // hooks instead). Verify the email is still present.
        if let ContentPart::ToolUse { input, .. } = &request.messages[0].content[0] {
            assert_eq!(input["email"], "leaks@example.com");
        } else {
            panic!("expected ToolUse part");
        }
    }

    #[tokio::test]
    async fn no_match_no_allocation() {
        let redactor = RegexRedactor::with_defaults();
        let mut request = req("hello world");
        let original = extract_text(&request.messages[0].content);
        redactor.redact_request(&mut request).await.unwrap();
        assert_eq!(extract_text(&request.messages[0].content), original);
    }

    #[test]
    fn default_patterns_compile() {
        // Sanity check that the static literals in `default_pii_patterns`
        // are real regexes — locks down the `unwrap` in `re(...)`.
        let patterns = default_pii_patterns();
        assert!(patterns.len() >= 4, "{patterns:?}");
    }

    #[test]
    fn from_strs_rejects_invalid_regex() {
        let err = RegexRedactor::from_strs([("bad", "[", "[X]".into())]).unwrap_err();
        assert!(matches!(err, PolicyError::Config(_)));
    }

    #[tokio::test]
    async fn pattern_order_is_respected() {
        // Earlier pattern wins on overlapping matches.
        let redactor = RegexRedactor::from_strs([
            ("first", r"\d+", "<n>".into()),
            ("second", r"\d{4}", "<year>".into()),
        ])
        .unwrap();
        let mut request = req("year 2026");
        redactor.redact_request(&mut request).await.unwrap();
        let txt = extract_text(&request.messages[0].content);
        // First pattern (\d+) consumes 2026 entirely; second never sees it.
        assert!(txt.contains("<n>"));
        assert!(!txt.contains("<year>"));
    }

    #[test]
    fn luhn_recognizes_well_known_test_numbers() {
        // Visa, Mastercard, Amex test numbers (commonly published).
        assert!(luhn_valid("4111 1111 1111 1111"));
        assert!(luhn_valid("5500-0000-0000-0004"));
        assert!(luhn_valid("340000000000009"));
        // 16-digit run that fails Luhn — tracking number, order id, etc.
        assert!(!luhn_valid("1234 5678 9012 3456"));
        // Too short / too long.
        assert!(!luhn_valid("4111111"));
        assert!(!luhn_valid(&"4".repeat(20)));
    }

    #[tokio::test]
    async fn phone_left_word_boundary_blocks_mid_identifier_match() {
        // `+1-415-555-0199` embedded *inside* an identifier (preceded
        // by a word character) must not be redacted. The regex itself
        // can't express this (it has no leading `\b` because `+` is
        // non-word) — the `left_word_boundary` post-check is what
        // closes the gap.
        let redactor = RegexRedactor::with_defaults();
        let mut request =
            req("order-id-x+1-415-555-0199-tail and standalone +1 (415) 555-0199 here");
        redactor.redact_request(&mut request).await.unwrap();
        let txt = extract_text(&request.messages[0].content);
        // Standalone phone redacted.
        assert!(
            txt.contains("[REDACTED:phone]"),
            "standalone phone missed: {txt}"
        );
        // Mid-identifier phone left intact.
        assert!(
            txt.contains("order-id-x+1-415-555-0199-tail"),
            "phone redacted inside identifier: {txt}"
        );
    }

    #[tokio::test]
    async fn credit_card_luhn_passes_real_number_blocks_fakes() {
        let redactor = RegexRedactor::with_defaults();
        let mut request = req(
            "card 4111 1111 1111 1111 valid; ticket 1234 5678 9012 3456 invalid; ord 9876543210",
        );
        redactor.redact_request(&mut request).await.unwrap();
        let txt = extract_text(&request.messages[0].content);
        assert!(
            txt.contains("[REDACTED:credit_card]"),
            "real card was missed: {txt}"
        );
        assert!(
            txt.contains("1234 5678 9012 3456"),
            "Luhn-fail run got falsely redacted: {txt}"
        );
        assert!(
            txt.contains("9876543210"),
            "10-digit order id got falsely redacted: {txt}"
        );
    }
}
