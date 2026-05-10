//! `SkillManifest` + `parse_skill_md` — minimal parser for the
//! `SKILL.md` frontmatter shape used by [`SandboxSkill`].
//!
//! The format is a YAML-style frontmatter block delimited by `---`,
//! followed by the markdown instructions body:
//!
//! ```text
//! ---
//! name: code-review
//! description: Review pull requests with focus on correctness.
//! version: 1.0.0
//! ---
//! Body of the skill — this becomes the activated instructions.
//! ```
//!
//! The parser is deliberately small and conservative: it accepts the
//! frontmatter fields the trait actually uses (`name`, `description`,
//! `version`), and treats anything else as forward-compatibility
//! metadata that the caller can read separately if it cares. No
//! dependency on a full YAML engine — the keys we accept are
//! single-line strings and the format never needs lists or nested
//! mappings.

use thiserror::Error;

/// Parsed contents of a `SKILL.md` file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkillManifest {
    /// Stable kebab-case identifier (required).
    pub name: String,
    /// One-line description (required).
    pub description: String,
    /// Optional semver version.
    pub version: Option<String>,
    /// Markdown body that follows the frontmatter — becomes
    /// `LoadedSkill::instructions`.
    pub body: String,
}

/// Reason a `SKILL.md` failed to parse.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ManifestError {
    /// The file did not begin with the expected `---` frontmatter
    /// fence.
    #[error("SKILL.md must begin with a `---` frontmatter fence")]
    MissingOpeningFence,
    /// The opening `---` fence was found but no closing fence appeared.
    #[error("SKILL.md frontmatter has no closing `---` fence")]
    MissingClosingFence,
    /// A required field (`name` or `description`) was absent.
    #[error("SKILL.md frontmatter missing required `{0}`")]
    MissingField(&'static str),
    /// A field value was empty after trimming.
    #[error("SKILL.md frontmatter `{0}` is empty")]
    EmptyField(&'static str),
}

/// Parse a `SKILL.md` document into its frontmatter + body.
pub fn parse_skill_md(input: &str) -> Result<SkillManifest, ManifestError> {
    let trimmed = input.trim_start_matches('\u{feff}'); // strip BOM if present
    let after_open = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))
        .ok_or(ManifestError::MissingOpeningFence)?;
    let close_idx = find_closing_fence(after_open).ok_or(ManifestError::MissingClosingFence)?;
    let frontmatter = &after_open[..close_idx];
    // Skip past the closing fence line. The fence itself is `---`
    // optionally followed by `\r\n` or `\n`; everything after is the
    // body. `find_closing_fence` returns the offset *to* the fence, so
    // we have to advance past the fence and its trailing newline.
    let after_close = after_open[close_idx..]
        .strip_prefix("---\n")
        .or_else(|| after_open[close_idx..].strip_prefix("---\r\n"))
        .or_else(|| after_open[close_idx..].strip_prefix("---"))
        .unwrap_or("");

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut version: Option<String> = None;

    for line in frontmatter.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = strip_inline_quotes(value.trim());
        match key {
            "name" => name = Some(value.to_owned()),
            "description" => description = Some(value.to_owned()),
            "version" => version = Some(value.to_owned()),
            _ => { /* forward-compatibility — ignore unknown keys */ }
        }
    }

    let name = require(name, "name")?;
    let description = require(description, "description")?;

    Ok(SkillManifest {
        name,
        description,
        version: version.filter(|v| !v.is_empty()),
        body: after_close.trim_start_matches('\n').to_owned(),
    })
}

fn require(value: Option<String>, field: &'static str) -> Result<String, ManifestError> {
    let v = value.ok_or(ManifestError::MissingField(field))?;
    if v.is_empty() {
        return Err(ManifestError::EmptyField(field));
    }
    Ok(v)
}

/// Find the byte offset of the first `---` line *after* the opening
/// frontmatter. The fence must appear at the start of a line; `---`
/// inside string values is not matched.
fn find_closing_fence(body: &str) -> Option<usize> {
    let mut offset = 0;
    for line in body.split_inclusive('\n') {
        let line_end = offset + line.len();
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed == "---" {
            return Some(offset);
        }
        offset = line_end;
    }
    if body.trim_end_matches(['\n', '\r']) == "---" {
        return Some(0);
    }
    None
}

/// Remove a single layer of matched single or double quotes, if
/// present. YAML allows quoting string values to preserve leading /
/// trailing whitespace; this lightweight stripper handles the common
/// case.
fn strip_inline_quotes(s: &str) -> &str {
    let bytes = s.as_bytes();
    let (Some(&first), Some(&last)) = (bytes.first(), bytes.last()) else {
        return s;
    };
    if bytes.len() >= 2 && ((first == b'"' && last == b'"') || (first == b'\'' && last == b'\'')) {
        return &s[1..s.len() - 1];
    }
    s
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_manifest() {
        let m =
            parse_skill_md("---\nname: echo\ndescription: Echo back text.\n---\nBody.\n").unwrap();
        assert_eq!(m.name, "echo");
        assert_eq!(m.description, "Echo back text.");
        assert_eq!(m.version, None);
        assert_eq!(m.body, "Body.\n");
    }

    #[test]
    fn parses_version_field() {
        let m = parse_skill_md("---\nname: code-review\ndescription: x\nversion: 1.2.3\n---\nbody")
            .unwrap();
        assert_eq!(m.version, Some("1.2.3".into()));
    }

    #[test]
    fn ignores_unknown_keys_for_forward_compat() {
        let m =
            parse_skill_md("---\nname: t\ndescription: t\nfuture-field: ignored\n---\n").unwrap();
        assert_eq!(m.name, "t");
    }

    #[test]
    fn strips_inline_quotes_from_values() {
        let m =
            parse_skill_md("---\nname: \"quoted-name\"\ndescription: 'single quoted'\n---\nbody")
                .unwrap();
        assert_eq!(m.name, "quoted-name");
        assert_eq!(m.description, "single quoted");
    }

    #[test]
    fn missing_opening_fence_errors() {
        let err = parse_skill_md("name: foo\n").unwrap_err();
        assert!(matches!(err, ManifestError::MissingOpeningFence));
    }

    #[test]
    fn missing_closing_fence_errors() {
        let err = parse_skill_md("---\nname: foo\ndescription: bar\n").unwrap_err();
        assert!(matches!(err, ManifestError::MissingClosingFence));
    }

    #[test]
    fn missing_required_field_errors() {
        let err = parse_skill_md("---\nname: foo\n---\nbody").unwrap_err();
        assert!(matches!(err, ManifestError::MissingField("description")));
    }

    #[test]
    fn empty_required_field_errors() {
        let err = parse_skill_md("---\nname: foo\ndescription:\n---\nbody").unwrap_err();
        assert!(matches!(err, ManifestError::EmptyField("description")));
    }

    #[test]
    fn handles_crlf_line_endings() {
        let m = parse_skill_md("---\r\nname: t\r\ndescription: d\r\n---\r\nbody").unwrap();
        assert_eq!(m.name, "t");
        assert_eq!(m.body, "body");
    }

    #[test]
    fn body_preserves_internal_horizontal_rules() {
        // `---` inside the body (not at the start of a line by itself)
        // must not be treated as a fence; here it's after content.
        let m = parse_skill_md("---\nname: t\ndescription: d\n---\n# Heading\n\n---\nMore body.\n")
            .unwrap();
        assert!(m.body.contains("---\nMore body."));
    }
}
