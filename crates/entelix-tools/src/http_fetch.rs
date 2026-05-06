//! `HttpFetchTool` — `Tool` impl for outbound HTTP fetches.
//!
//! ## Threat model
//!
//! Naïve "let the model fetch any URL" tools are SSRF magnets:
//! agents have been talked into hitting `http://169.254.169.254/...`
//! (cloud metadata) and internal services. The defense lives in
//! three orthogonal layers:
//!
//! 1. **Host allowlist** — explicit allow-by-domain list. The
//!    builder requires at least one entry; an unconfigured tool
//!    refuses every URL. [`HostAllowlist`] supports exact matches,
//!    wildcard subdomains (`*.example.com`), and explicit IP-range
//!    permits.
//! 2. **Scheme guard** — only `http` and `https`. `file://`,
//!    `javascript:`, `data:`, `gopher://`, and IP-of-ftp tricks all
//!    bounce here.
//! 3. **Private-IP block** — by default literal IPs in
//!    loopback / private / link-local / metadata ranges are
//!    rejected even when the surface allowlist would otherwise
//!    permit them. Override with [`HostRule::IpExact`] when an
//!    on-prem deployment genuinely needs `127.0.0.1:8080`.
//!
//! Layered defense rather than a single check — any one layer
//! could be misconfigured but all three together close the
//! reasonable SSRF surface.
//!
//! ## Resource caps
//!
//! - **Method allowlist** — defaults to `[GET]`. POST / PATCH must
//!   be opted in.
//! - **Redirect cap** — defaults to 5; `0` disables redirects.
//! - **Body cap** — defaults to 1 MiB; the response stream aborts
//!   with [`ToolError::BodyTooLarge`] once the cap is exceeded
//!   instead of buffering the whole tail.
//! - **Per-call timeout** — defaults to 30 s; respects
//!   [`entelix_core::context::ExecutionContext::cancellation`] as a
//!   secondary kill switch.

use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::BytesMut;
use futures::StreamExt;
use reqwest::Method;
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;

use entelix_core::AgentContext;
use entelix_core::error::Result;
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};

use crate::error::{ToolError, ToolResult};

/// Default cap on redirect chain length.
pub const DEFAULT_MAX_REDIRECTS: usize = 5;

/// Default cap on response body size (1 MiB).
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 1024 * 1024;

/// Default per-call timeout.
pub const DEFAULT_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// One allowlist rule.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HostRule {
    /// Exact case-insensitive hostname match (e.g. `api.example.com`).
    Exact(String),
    /// Wildcard subdomain — `*.example.com` matches `a.example.com`
    /// and `b.c.example.com` but not `example.com` itself.
    Wildcard(String),
    /// Exact IP literal (e.g. `127.0.0.1`). Use sparingly — bypasses
    /// the private-IP block.
    IpExact(IpAddr),
}

/// Host allowlist. Fail-closed: empty allowlist rejects everything.
#[derive(Clone, Debug, Default)]
pub struct HostAllowlist {
    rules: Vec<HostRule>,
}

impl HostAllowlist {
    /// Empty (fail-closed) allowlist.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Normalize a hostname to its ASCII-Punycode form (UTS-46 +
    /// IDNA-2008 transitional rules) and lower-case it. IDN inputs
    /// like `пример.рф` round-trip to `xn--e1afmkfd.xn--p1ai`, which
    /// is the form `Url::host_str()` returns at check time. Falls
    /// back to a plain lowercase when normalization fails (preserves
    /// the previous behavior for hostnames the IDNA pass rejects).
    fn normalize(host: &str) -> String {
        idna::domain_to_ascii(host).map_or_else(|_| host.to_lowercase(), |s| s.to_lowercase())
    }

    /// Append an exact hostname rule (case-insensitive). IDN inputs
    /// are normalized to Punycode so a Cyrillic-look-alike domain
    /// cannot bypass an entry registered in Latin script (or vice
    /// versa).
    #[must_use]
    pub fn add_exact_host(mut self, host: impl Into<String>) -> Self {
        self.rules
            .push(HostRule::Exact(Self::normalize(&host.into())));
        self
    }

    /// Append a wildcard-subdomain rule. The leading `*.` is
    /// optional in the supplied string; both `*.example.com` and
    /// `example.com` are accepted as input and stored without the
    /// `*.` prefix for matching. Inputs are normalized to Punycode
    /// the same way as [`Self::add_exact_host`].
    #[must_use]
    pub fn add_subdomain_root(mut self, host: impl Into<String>) -> Self {
        let raw = host.into();
        let stripped = raw.strip_prefix("*.").unwrap_or(&raw);
        self.rules
            .push(HostRule::Wildcard(Self::normalize(stripped)));
        self
    }

    /// Append an exact IP literal rule. Intended for narrow on-prem
    /// allowances; prefer `allow_exact` over this for hostnames.
    #[must_use]
    pub fn add_exact_ip(mut self, ip: IpAddr) -> Self {
        self.rules.push(HostRule::IpExact(ip));
        self
    }

    /// Number of registered rules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the allowlist has zero rules (rejects everything).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Borrow every IP registered via [`Self::add_exact_ip`].
    /// Used by [`HttpFetchToolBuilder`] to seed the SSRF-safe DNS
    /// resolver's explicit-allow set so on-prem private-IP
    /// allowances pass the connect-time block.
    pub fn explicit_ips(&self) -> std::collections::HashSet<IpAddr> {
        self.rules
            .iter()
            .filter_map(|r| match r {
                HostRule::IpExact(ip) => Some(*ip),
                _ => None,
            })
            .collect()
    }

    fn check(&self, url: &Url) -> ToolResult<()> {
        let host = url.host_str().ok_or_else(|| ToolError::HostBlocked {
            host: "<no host>".to_owned(),
        })?;
        // `Url::parse` already Punycode-encodes IDN hosts, but we run
        // the same normalize pass so rule and check stay symmetric —
        // any future change to the normalizer applies everywhere at
        // once.
        let host_lower = Self::normalize(host);

        // 1. IP literal short-circuit — only allowed via explicit
        //    IpExact rule (overrides the private-IP block).
        if let Ok(ip) = host_lower.parse::<IpAddr>() {
            for rule in &self.rules {
                if let HostRule::IpExact(allowed) = rule
                    && *allowed == ip
                {
                    return Ok(());
                }
            }
            return Err(ToolError::HostBlocked { host: host_lower });
        }

        // 2. Hostname rules.
        for rule in &self.rules {
            match rule {
                HostRule::Exact(h) if h == &host_lower => return Ok(()),
                HostRule::Wildcard(suffix) => {
                    if host_lower == *suffix {
                        // `*.example.com` does NOT match the apex
                        // `example.com` — that's the whole point of
                        // wildcard-subdomain: subdomains, not the
                        // bare host.
                        continue;
                    }
                    if host_lower.ends_with(&format!(".{suffix}")) {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        Err(ToolError::HostBlocked { host: host_lower })
    }
}

/// Builder for [`HttpFetchTool`].
pub struct HttpFetchToolBuilder {
    allowlist: HostAllowlist,
    max_redirects: usize,
    max_response_bytes: usize,
    timeout: Duration,
    allowed_methods: HashSet<Method>,
    user_agent: String,
    /// Lower-cased response header names the tool surfaces to the
    /// model. Empty = no headers reach the LLM (default — most
    /// vendor headers like `set-cookie`, `cf-ray`, `x-amz-*` are
    /// noise that burns model attention without informing reasoning,
    /// invariant #16).
    exposed_response_headers: HashSet<String>,
}

impl HttpFetchToolBuilder {
    /// Start a builder with no allowlist (fail-closed), the
    /// `[GET]` default method allowlist, and no response headers
    /// exposed to the model.
    #[must_use]
    pub fn new() -> Self {
        let mut methods = HashSet::new();
        methods.insert(Method::GET);
        Self {
            allowlist: HostAllowlist::new(),
            max_redirects: DEFAULT_MAX_REDIRECTS,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            timeout: DEFAULT_FETCH_TIMEOUT,
            allowed_methods: methods,
            user_agent: format!("entelix-http-fetch/{}", env!("CARGO_PKG_VERSION")),
            exposed_response_headers: HashSet::new(),
        }
    }

    /// Set the host allowlist outright.
    #[must_use]
    pub fn with_allowlist(mut self, allowlist: HostAllowlist) -> Self {
        self.allowlist = allowlist;
        self
    }

    /// Cap redirect chain length. `0` disables redirects entirely.
    #[must_use]
    pub const fn with_max_redirects(mut self, n: usize) -> Self {
        self.max_redirects = n;
        self
    }

    /// Cap the response body in bytes.
    #[must_use]
    pub const fn with_max_response_bytes(mut self, n: usize) -> Self {
        self.max_response_bytes = n;
        self
    }

    /// Per-call timeout.
    #[must_use]
    pub const fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    /// Set the method allowlist outright. Intersect with the actual
    /// HTTP method on the input — anything not in this set is
    /// rejected.
    #[must_use]
    pub fn with_allowed_methods<I: IntoIterator<Item = Method>>(mut self, methods: I) -> Self {
        self.allowed_methods = methods.into_iter().collect();
        self
    }

    /// Override the `User-Agent` header.
    #[must_use]
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Allow the tool to surface specific response headers to the
    /// model (LLM-facing). Header names are lower-cased and matched
    /// case-insensitively. Default is the empty set — every response
    /// header is dropped from the tool output, sparing the model
    /// from `set-cookie` / `cf-ray` / `x-amz-request-id` /
    /// `content-encoding` noise that costs tokens without informing
    /// reasoning (invariant #16). Operators that need a header
    /// (e.g. `content-type` for the model to branch on payload
    /// shape) opt it in explicitly.
    #[must_use]
    pub fn with_exposed_response_headers<I, S>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.exposed_response_headers = headers
            .into_iter()
            .map(|h| h.as_ref().to_ascii_lowercase())
            .collect();
        self
    }

    /// Finalize. Returns [`ToolError::Config`] when the allowlist
    /// is empty (fail-closed: an unconfigured tool would refuse
    /// every URL anyway, but explicit early failure is friendlier
    /// to operators).
    pub fn build(self) -> ToolResult<HttpFetchTool> {
        if self.allowlist.is_empty() {
            return Err(ToolError::config_msg(
                "HttpFetchTool requires at least one HostAllowlist rule",
            ));
        }
        // Allowlist + scheme guard re-applied on every redirect hop.
        // Without this, a 302 from an allowlisted host to an
        // unlisted (but DNS-public) one would succeed: the
        // host-allowlist check only ran on the first URL and the
        // SSRF DNS resolver alone does not enforce host policy.
        let allowlist_for_policy = Arc::new(self.allowlist.clone());
        let max_redirects = self.max_redirects;
        let policy = if max_redirects == 0 {
            Policy::none()
        } else {
            Policy::custom(move |attempt| {
                if attempt.previous().len() >= max_redirects {
                    return attempt.error(redirect_error(format!(
                        "redirect cap exceeded ({max_redirects})"
                    )));
                }
                let scheme = attempt.url().scheme().to_owned();
                if !matches!(scheme.as_str(), "http" | "https") {
                    return attempt.error(redirect_error(format!(
                        "redirect to disallowed scheme '{scheme}'"
                    )));
                }
                if let Err(e) = allowlist_for_policy.check(attempt.url()) {
                    return attempt.error(redirect_error(format!(
                        "redirect to non-allowlisted host: {e}"
                    )));
                }
                attempt.follow()
            })
        };
        // SSRF-safe DNS resolver: filters every connect-time lookup
        // against the private/loopback/metadata block. IP literals
        // explicitly registered on the allowlist override the block
        // (on-prem proxies bind 127.0.0.1, etc.).
        let resolver = crate::dns::SsrfSafeDnsResolver::from_system()?
            .with_explicit_allow(self.allowlist.explicit_ips());
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .redirect(policy)
            .user_agent(self.user_agent)
            .dns_resolver(Arc::new(resolver))
            .build()
            .map_err(|e| ToolError::Config {
                message: format!("HTTP client: {e}"),
                source: Some(Box::new(e)),
            })?;
        let metadata = ToolMetadata::function(
            "http_fetch",
            "Fetch a URL over HTTP/HTTPS. Returns status, final_url (post-redirect), \
             headers, body. Restricted to the configured host allowlist.",
            json!({
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http(s) URL to fetch."
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method (default: GET).",
                        "enum": ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]
                    },
                    "headers": {
                        "type": "object",
                        "description": "Extra request headers.",
                        "additionalProperties": { "type": "string" }
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body (for non-GET methods)."
                    }
                }
            }),
        )
        .with_effect(ToolEffect::Mutating);
        Ok(HttpFetchTool {
            client,
            allowlist: Arc::new(self.allowlist),
            max_response_bytes: self.max_response_bytes,
            allowed_methods: Arc::new(self.allowed_methods),
            exposed_response_headers: Arc::new(self.exposed_response_headers),
            metadata: Arc::new(metadata),
        })
    }
}

/// Wraps a redirect-rejection message into a `Box<dyn Error>` so
/// `reqwest::redirect::Attempt::error` accepts it.
fn redirect_error(message: String) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(RedirectRejected(message))
}

/// Internal error type produced by the redirect policy. The text is
/// surfaced through reqwest's `Error::Display` so callers see the
/// rejection reason in their `ToolError::Network` chain.
#[derive(Debug)]
struct RedirectRejected(String);

impl std::fmt::Display for RedirectRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RedirectRejected {}

impl Default for HttpFetchToolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP fetch [`Tool`] for agentic workflows.
///
/// Cloning is cheap (handles are `Arc`-backed). Share one tool
/// instance across the process and across the hooks pipeline.
#[derive(Clone)]
pub struct HttpFetchTool {
    client: reqwest::Client,
    allowlist: Arc<HostAllowlist>,
    max_response_bytes: usize,
    allowed_methods: Arc<HashSet<Method>>,
    exposed_response_headers: Arc<HashSet<String>>,
    metadata: Arc<ToolMetadata>,
}

#[allow(
    clippy::missing_fields_in_debug,
    reason = "`reqwest::Client` is opaque; printed as configured-rule counts"
)]
impl std::fmt::Debug for HttpFetchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpFetchTool")
            .field("allowlist_rules", &self.allowlist.len())
            .field("max_response_bytes", &self.max_response_bytes)
            .field("allowed_methods", &self.allowed_methods.len())
            .finish()
    }
}

impl HttpFetchTool {
    /// Start a builder.
    #[must_use]
    pub fn builder() -> HttpFetchToolBuilder {
        HttpFetchToolBuilder::new()
    }
}

#[derive(Debug, Deserialize)]
struct FetchInput {
    url: String,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    headers: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    body: Option<String>,
}

#[derive(Debug, Serialize)]
struct FetchOutput {
    status: u16,
    final_url: String,
    headers: std::collections::HashMap<String, String>,
    body: String,
    truncated: bool,
}

#[async_trait]
impl Tool for HttpFetchTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &AgentContext<()>) -> Result<Value> {
        let parsed: FetchInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let url = Url::parse(&parsed.url)
            .map_err(|e| ToolError::InvalidInput(format!("malformed URL: {e}")))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(ToolError::UnsupportedScheme {
                scheme: url.scheme().to_owned(),
            }
            .into());
        }
        self.allowlist.check(&url)?;

        let method = match parsed.method.as_deref() {
            Some(m) => Method::from_bytes(m.to_uppercase().as_bytes())
                .map_err(|_| ToolError::InvalidInput(format!("unknown method '{m}'")))?,
            None => Method::GET,
        };
        if !self.allowed_methods.contains(&method) {
            return Err(ToolError::MethodBlocked {
                method: method.to_string(),
            }
            .into());
        }

        let mut request = self.client.request(method, url.clone());
        if let Some(headers) = &parsed.headers {
            for (k, v) in headers {
                request = request.header(k, v);
            }
        }
        if let Some(body) = parsed.body {
            request = request.body(body);
        }

        // Race the HTTP send against cancellation.
        let cancel = ctx.cancellation().clone();
        let response = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                return Err(ToolError::network_msg("cancelled").into());
            }
            r = request.send() => r.map_err(ToolError::network)?,
        };

        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        // Default-deny: only headers the operator explicitly opted
        // in via `with_exposed_response_headers` flow to the LLM. Vendor
        // chrome (`set-cookie`, `cf-ray`, `x-amz-*`, `via`,
        // `content-encoding`, …) costs the model tokens without
        // informing reasoning (invariant #16).
        let allow = &*self.exposed_response_headers;
        let response_headers = if allow.is_empty() {
            std::collections::HashMap::new()
        } else {
            response
                .headers()
                .iter()
                .filter(|(k, _)| allow.contains(k.as_str()))
                .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_owned())))
                .collect::<std::collections::HashMap<_, _>>()
        };

        // Stream-and-cap body collection.
        let mut buf = BytesMut::new();
        let mut truncated = false;
        let mut stream = response.bytes_stream();
        let cancel = ctx.cancellation().clone();
        loop {
            let chunk = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    return Err(ToolError::network_msg("cancelled").into());
                }
                next = stream.next() => match next {
                    Some(Ok(c)) => c,
                    Some(Err(e)) => {
                        return Err(ToolError::network(e).into());
                    }
                    None => break,
                },
            };
            if buf.len().saturating_add(chunk.len()) > self.max_response_bytes {
                let take = self
                    .max_response_bytes
                    .saturating_sub(buf.len())
                    .min(chunk.len());
                buf.extend_from_slice(chunk.get(..take).unwrap_or(&[]));
                truncated = true;
                break;
            }
            buf.extend_from_slice(&chunk);
        }

        // Treat the body as UTF-8 text when it parses, otherwise
        // hex-prefixed binary marker. Tools shouldn't surface
        // arbitrary bytes inline; the agent loop expects strings.
        let body = match std::str::from_utf8(&buf) {
            Ok(s) => s.to_owned(),
            Err(_) => format!("<binary {} bytes>", buf.len()),
        };

        let output = FetchOutput {
            status,
            final_url,
            headers: response_headers,
            body,
            truncated,
        };
        Ok(serde_json::to_value(output).map_err(ToolError::from)?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::ip_constant)]
mod tests {
    use std::net::Ipv4Addr;

    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn empty_allowlist_rejects_everything() {
        let allow = HostAllowlist::new();
        assert!(allow.check(&url("https://example.com/x")).is_err());
    }

    #[test]
    fn exact_host_match() {
        let allow = HostAllowlist::new().add_exact_host("api.example.com");
        assert!(allow.check(&url("https://api.example.com/path")).is_ok());
        assert!(allow.check(&url("https://other.example.com/")).is_err());
    }

    #[test]
    fn case_insensitive_hostname_match() {
        let allow = HostAllowlist::new().add_exact_host("API.example.com");
        assert!(allow.check(&url("https://api.example.com/")).is_ok());
        assert!(allow.check(&url("https://API.EXAMPLE.COM/")).is_ok());
    }

    #[test]
    fn wildcard_matches_subdomains_only_not_apex() {
        let allow = HostAllowlist::new().add_subdomain_root("example.com");
        assert!(allow.check(&url("https://a.example.com/")).is_ok());
        assert!(allow.check(&url("https://x.y.example.com/")).is_ok());
        // Apex must NOT match a wildcard rule.
        assert!(allow.check(&url("https://example.com/")).is_err());
    }

    #[test]
    fn wildcard_input_strips_leading_star_dot() {
        let allow = HostAllowlist::new().add_subdomain_root("*.example.com");
        assert!(allow.check(&url("https://a.example.com/")).is_ok());
    }

    #[test]
    fn ip_literals_require_explicit_rule() {
        let allow = HostAllowlist::new().add_exact_host("example.com");
        assert!(allow.check(&url("http://127.0.0.1/x")).is_err());
        assert!(allow.check(&url("http://10.0.0.5/x")).is_err());
    }

    #[test]
    fn explicit_ip_exact_admits() {
        let allow = HostAllowlist::new().add_exact_ip(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
        assert!(allow.check(&url("http://127.0.0.1/x")).is_ok());
        assert!(allow.check(&url("http://127.0.0.2/x")).is_err());
    }

    #[test]
    fn builder_requires_non_empty_allowlist() {
        let err = HttpFetchToolBuilder::new().build().unwrap_err();
        assert!(matches!(err, ToolError::Config { .. }));
    }

    #[test]
    fn idn_rule_matches_punycode_url() {
        // Rule provided in human form; URL arrives in Punycode (which
        // is what `Url::parse` produces) — they must agree.
        let allow = HostAllowlist::new().add_exact_host("пример.рф");
        // `xn--e1afmkfd.xn--p1ai` is the canonical Punycode of пример.рф.
        let parsed = url("https://xn--e1afmkfd.xn--p1ai/");
        assert_eq!(parsed.host_str(), Some("xn--e1afmkfd.xn--p1ai"));
        assert!(allow.check(&parsed).is_ok());
    }

    #[test]
    fn punycode_rule_matches_idn_input_via_url_parse() {
        // Symmetric: rule given in Punycode; URL passes through
        // `Url::parse` which canonicalizes IDNs to ASCII.
        let allow = HostAllowlist::new().add_exact_host("xn--e1afmkfd.xn--p1ai");
        let parsed = url("https://пример.рф/path");
        assert!(allow.check(&parsed).is_ok());
    }

    #[test]
    fn cyrillic_lookalike_blocked_when_only_latin_is_allowed() {
        // Cyrillic 'е' (U+0435) is visually identical to Latin 'e'
        // (U+0065). An allowlist for the Latin domain must NOT admit
        // a homograph attack: post-IDNA the two normalize to
        // different ASCII (Punycode) forms.
        let allow = HostAllowlist::new().add_exact_host("example.com");
        // "еxample.com" with the leading 'e' replaced by Cyrillic 'е'.
        let homograph = "\u{0435}xample.com";
        // `Url::parse` runs IDNA on this; the resulting host_str is
        // the Punycode form, which is not "example.com".
        let parsed = Url::parse(&format!("https://{homograph}/")).unwrap();
        assert_ne!(parsed.host_str(), Some("example.com"));
        assert!(allow.check(&parsed).is_err());
    }

    #[test]
    fn idn_wildcard_matches_subdomain() {
        let allow = HostAllowlist::new().add_subdomain_root("пример.рф");
        let parsed = url("https://api.xn--e1afmkfd.xn--p1ai/");
        assert!(allow.check(&parsed).is_ok());
    }
}
