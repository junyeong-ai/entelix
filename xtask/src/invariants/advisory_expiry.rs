//! Advisory-ignore expiry enforcement. Every entry in `deny.toml`'s
//! `[advisories].ignore` array carries a `REVIEW-BY: YYYY-MM-DD` date
//! in its `reason` field; when the date is in the past, this gate
//! fails so the entry cannot quietly outlive its rationale.
//!
//! Rationale: cargo-deny's native `expires` field was added in a
//! later release than the version this workspace pins; rather than
//! couple the supply-chain gate to a moving cargo-deny revision, the
//! tripwire lives in xtask. The `reason` field accepts arbitrary
//! prose, so the date convention is stable across every cargo-deny
//! revision.
//!
//! Failure mode:
//!   * missing `REVIEW-BY:` prefix in any `reason` string
//!   * past date in any `REVIEW-BY:` field
//!   * unparseable date string (must be `YYYY-MM-DD`)
//!
//! Recovery: re-evaluate whether the upstream fix has shipped or the
//! structural-absence rationale still holds, then either delete the
//! ignore entry or extend the date with refreshed prose.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::visitor::{Violation, read, repo_root, report};

const ADVISORY_FILE: &str = "deny.toml";
const REVIEW_PREFIX: &str = "REVIEW-BY:";

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let path = root.join(ADVISORY_FILE);
    let text = read(&path)?;
    let doc: toml_edit::DocumentMut = text
        .parse()
        .with_context(|| format!("parse {}", path.display()))?;

    let ignore = doc
        .get("advisories")
        .and_then(|a| a.as_table())
        .and_then(|t| t.get("ignore"))
        .and_then(|i| i.as_array())
        .with_context(|| format!("`[advisories].ignore` array missing in {}", path.display()))?;

    let today = today_iso();
    let mut violations = Vec::new();

    for (idx, entry) in ignore.iter().enumerate() {
        let table = match entry.as_inline_table() {
            Some(t) => t,
            None => {
                violations.push(Violation::file(
                    path.clone(),
                    format!(
                        "ignore entry #{idx} is a bare ID instead of an inline table — every entry must be `{{ id = \"…\", reason = \"REVIEW-BY: YYYY-MM-DD …\" }}`"
                    ),
                ));
                continue;
            }
        };
        let id = table
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing>");
        let reason = match table.get("reason").and_then(|v| v.as_str()) {
            Some(r) => r,
            None => {
                violations.push(Violation::file(
                    path.clone(),
                    format!("`{id}` ignore is missing the `reason` field"),
                ));
                continue;
            }
        };
        let date = match extract_review_by(reason) {
            Some(d) => d,
            None => {
                violations.push(Violation::file(
                    path.clone(),
                    format!(
                        "`{id}` ignore reason lacks `REVIEW-BY: YYYY-MM-DD` prefix — current reason: {reason:?}"
                    ),
                ));
                continue;
            }
        };
        if !is_valid_iso_date(&date) {
            violations.push(Violation::file(
                path.clone(),
                format!(
                    "`{id}` ignore reason has `REVIEW-BY: {date}` but the date is not a valid `YYYY-MM-DD` string"
                ),
            ));
            continue;
        }
        if date.as_str() < today.as_str() {
            violations.push(Violation::file(
                path.clone(),
                format!(
                    "`{id}` ignore expired on {date} (today {today}) — re-evaluate the upstream pin and either delete the entry or refresh the `REVIEW-BY:` date with new rationale"
                ),
            ));
        }
    }

    report(
        "advisory-expiry",
        violations,
        "Every advisory ignore in deny.toml must carry a `REVIEW-BY:\n\
         YYYY-MM-DD` prefix in its `reason` field. When the date is\n\
         past, the gate fires — refresh the rationale and extend the\n\
         date, or remove the ignore entry if the upstream fix has\n\
         shipped. Permanent ignores are reviewer-rejected.",
    )
}

/// Pull a `YYYY-MM-DD` date out of a reason string that begins with
/// `REVIEW-BY:`. Whitespace and the colon are skipped. Returns the
/// canonical 10-character form `YYYY-MM-DD` on success, `None` if the
/// prefix is missing or the token does not match the shape.
fn extract_review_by(reason: &str) -> Option<String> {
    let after = reason.trim_start().strip_prefix(REVIEW_PREFIX)?;
    let date = after.trim_start().chars().take(10).collect::<String>();
    if date.len() == 10 { Some(date) } else { None }
}

/// Validate the canonical `YYYY-MM-DD` shape — four-digit year,
/// hyphen, two-digit month (01–12), hyphen, two-digit day (01–31).
/// Calendar accuracy (Feb 30, etc.) is not checked; the gate is a
/// tripwire, not a calendar.
fn is_valid_iso_date(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    if !bytes[..4].iter().all(u8::is_ascii_digit) {
        return false;
    }
    if !bytes[5..7].iter().all(u8::is_ascii_digit) {
        return false;
    }
    if !bytes[8..].iter().all(u8::is_ascii_digit) {
        return false;
    }
    let month: u32 = std::str::from_utf8(&bytes[5..7])
        .unwrap_or("0")
        .parse()
        .unwrap_or(0);
    let day: u32 = std::str::from_utf8(&bytes[8..])
        .unwrap_or("0")
        .parse()
        .unwrap_or(0);
    (1..=12).contains(&month) && (1..=31).contains(&day)
}

/// Today's date in UTC, rendered `YYYY-MM-DD`. Std-only — computes
/// the civil-date proleptic Gregorian from the Unix epoch via Howard
/// Hinnant's well-known algorithm
/// (<https://howardhinnant.github.io/date_algorithms.html>).
fn today_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days-since-1970-01-01 → `(year, month, day)`. Shifts to the
/// algorithm's `0000-03-01` epoch, runs the era / day-of-era /
/// year-of-era decomposition, then shifts back.
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_review_by_strips_prefix_and_whitespace() {
        assert_eq!(
            extract_review_by("REVIEW-BY: 2026-08-11 — explanation"),
            Some("2026-08-11".into())
        );
        assert_eq!(
            extract_review_by("  REVIEW-BY:2026-08-11"),
            Some("2026-08-11".into())
        );
        assert_eq!(extract_review_by("no prefix here"), None);
    }

    #[test]
    fn iso_date_shape_validation() {
        assert!(is_valid_iso_date("2026-08-11"));
        assert!(is_valid_iso_date("0001-01-01"));
        assert!(is_valid_iso_date("9999-12-31"));
        assert!(!is_valid_iso_date("2026-13-01")); // month
        assert!(!is_valid_iso_date("2026-08-32")); // day
        assert!(!is_valid_iso_date("2026/08/11")); // wrong separator
        assert!(!is_valid_iso_date("26-08-11")); // short year
        assert!(!is_valid_iso_date("2026-8-11")); // unpadded month
    }

    #[test]
    fn civil_date_round_trip() {
        // Reference points from Howard Hinnant's paper.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // Year 2000 — leap year, well past the 1970 epoch.
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        // 20_637 days from 1970-01-01 = 2026-07-03 (56 years +
        // partial 2026 of 183 days; Jan 31 + Feb 28 + Mar 31 +
        // Apr 30 + May 31 + Jun 30 = 181, then Jul 2 = day 183).
        assert_eq!(civil_from_days(20_637), (2026, 7, 3));
    }
}
