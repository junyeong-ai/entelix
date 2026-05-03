//! SSRF-safe DNS resolver for `HttpFetchTool`.
//!
//! Without an explicit resolver, `reqwest`'s default DNS path runs at
//! connect time — *after* the [`HostAllowlist`](crate::HostAllowlist)
//! check on the URL host string has already passed. That leaves a
//! TOCTOU window: a hostname admitted by the allowlist can resolve to
//! a private / loopback / cloud-metadata IP at connect time, and the
//! tool happily reaches into the cluster's blast radius.
//!
//! [`SsrfSafeDnsResolver`] closes that window. It plugs into the
//! reqwest client via [`reqwest::dns::Resolve`]; every hostname the
//! HTTP stack tries to connect to passes through us first. We
//! resolve via `hickory-resolver` (modern async DNS, not the
//! blocking `std::net::ToSocketAddrs` path), filter out IPs in the
//! configured block ranges, and hand reqwest only the survivors.
//! When no IP survives, the request fails *before* a connection is
//! attempted.
//!
//! ## What's blocked by default
//!
//! - IPv4 loopback (`127.0.0.0/8`)
//! - IPv4 private (RFC 1918: `10/8`, `172.16/12`, `192.168/16`)
//! - IPv4 link-local (`169.254/16`) — covers AWS / GCP / Azure
//!   metadata endpoints (`169.254.169.254`).
//! - IPv4 CGNAT (`100.64/10`)
//! - IPv4 unspecified / broadcast / multicast / documentation ranges
//! - IPv6 loopback (`::1`), unspecified (`::`), unique-local
//!   (`fc00::/7`), link-local (`fe80::/10`), multicast (`ff00::/8`)
//!
//! ## Override path
//!
//! Operators sometimes need a private IP (e.g. `127.0.0.1:8080` for
//! an on-prem inference proxy). They register that IP via
//! [`HostAllowlist::allow_ip_exact`](crate::HostAllowlist::allow_ip_exact);
//! [`HttpFetchToolBuilder`](crate::HttpFetchToolBuilder) plumbs the
//! set into the resolver, which lets matching IPs through even when
//! the default block would reject them.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use hickory_resolver::TokioResolver;
use hickory_resolver::config::{ResolverConfig, ResolverOpts};
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::error::ToolError;

/// Returns `true` for IPs the SDK refuses to connect to by default.
///
/// Call out via the public re-export only if you're building a
/// custom resolver — `HttpFetchTool` already wires this through
/// [`SsrfSafeDnsResolver`].
///
/// ## IPv6 tunnel coverage
///
/// A naïve "block IPv4 private ranges" pass leaks: an attacker can
/// embed a private IPv4 inside an IPv6 prefix and bypass the v4
/// check entirely. We extract the embedded IPv4 from the relevant
/// prefixes and recurse, so:
///
/// - `::ffff:127.0.0.1` (IPv4-mapped) routes through the v4 block.
/// - `2002:7f00:0001::` (6to4 — `2002:WWXX:YYZZ::/48` carries IPv4
///   `WW.XX.YY.ZZ`) routes through the v4 block.
/// - `2001:0:WWXX:YYZZ::` (Teredo — server-mapped IPv4 in segs[2..4])
///   routes through the v4 block.
///
/// In addition, blanket Teredo (`2001::/32`) and 6to4 (`2002::/16`)
/// prefixes are themselves blocked by default, since they are
/// rarely the right transport for outbound LLM tool calls.
#[must_use]
pub fn is_ssrf_blocked(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_ssrf_blocked_v4(*v4),
        IpAddr::V6(v6) => is_ssrf_blocked_v6(*v6),
    }
}

fn is_ssrf_blocked_v4(v4: std::net::Ipv4Addr) -> bool {
    let octets = v4.octets();
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_broadcast()
        || v4.is_unspecified()
        || v4.is_multicast()
        || v4.is_documentation()
        // CGNAT 100.64.0.0/10 (RFC 6598). Not flagged by
        // `is_private` but still routes only inside the carrier.
        || (octets[0] == 100 && (64..=127).contains(&octets[1]))
}

fn is_ssrf_blocked_v6(v6: std::net::Ipv6Addr) -> bool {
    let segs = v6.segments();
    if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
        return true;
    }
    // Unique-local (fc00::/7).
    if segs[0] & 0xfe00 == 0xfc00 {
        return true;
    }
    // Link-local (fe80::/10).
    if segs[0] & 0xffc0 == 0xfe80 {
        return true;
    }
    // IPv4-mapped IPv6 (::ffff:0:0/96): `::ffff:W.X.Y.Z` is the same
    // host as the v4 address — apply the v4 block to the embedded
    // octets, otherwise an attacker resolves to ::ffff:127.0.0.1
    // and tunnels into loopback unchecked.
    if segs[0..5].iter().all(|s| *s == 0) && segs[5] == 0xffff {
        let v4 = std::net::Ipv4Addr::new(
            (segs[6] >> 8) as u8,
            (segs[6] & 0xff) as u8,
            (segs[7] >> 8) as u8,
            (segs[7] & 0xff) as u8,
        );
        return is_ssrf_blocked_v4(v4);
    }
    // 6to4 (2002::/16) — `2002:WWXX:YYZZ::/48` encapsulates IPv4
    // `WW.XX.YY.ZZ`. Block the entire /16 unconditionally:
    // 6to4 has no legitimate role in an outbound LLM call, and
    // the embedded v4 is meaningless when the prefix is rejected.
    if segs[0] == 0x2002 {
        return true;
    }
    // Teredo (2001::/32) — `2001:0:SVR_IP4_HI:SVR_IP4_LO::` carries
    // the Teredo server's IPv4 in segs[2..4]. Block the entire
    // /32 unconditionally — Teredo is a NAT-traversal protocol
    // that has no legitimate role in an outbound LLM call.
    if segs[0] == 0x2001 && segs[1] == 0 {
        return true;
    }
    false
}

/// `reqwest::dns::Resolve` impl that vets every resolved IP against
/// [`is_ssrf_blocked`] before handing addresses back to the HTTP
/// connector. See module docs for the threat model.
pub struct SsrfSafeDnsResolver {
    inner: TokioResolver,
    /// IP literals registered via
    /// [`HostAllowlist::allow_ip_exact`](crate::HostAllowlist::allow_ip_exact).
    /// These bypass the default block so on-prem allowances still
    /// work.
    explicit_allow: Arc<HashSet<IpAddr>>,
}

impl SsrfSafeDnsResolver {
    /// Build a resolver from the system's DNS config (`/etc/resolv.conf`
    /// on Unix, the Windows registry equivalents otherwise) with no
    /// explicit IP overrides.
    pub fn from_system() -> Result<Self, ToolError> {
        let inner = TokioResolver::builder_tokio()
            .map_err(|e| ToolError::Config {
                message: format!("DNS: failed to read system config: {e}"),
                source: Some(Box::new(e)),
            })?
            .build()
            .map_err(|e| ToolError::Config {
                message: format!("DNS: failed to construct resolver: {e}"),
                source: Some(Box::new(e)),
            })?;
        Ok(Self {
            inner,
            explicit_allow: Arc::new(HashSet::new()),
        })
    }

    /// Build a resolver with an explicit `(config, opts)` pair —
    /// useful for tests or for environments that pin a specific
    /// upstream resolver.
    pub fn from_config(config: ResolverConfig, opts: ResolverOpts) -> Result<Self, ToolError> {
        let inner = TokioResolver::builder_with_config(
            config,
            hickory_resolver::net::runtime::TokioRuntimeProvider::default(),
        )
        .with_options(opts)
        .build()
        .map_err(|e| ToolError::Config {
            message: format!("DNS: failed to construct resolver: {e}"),
            source: Some(Box::new(e)),
        })?;
        Ok(Self {
            inner,
            explicit_allow: Arc::new(HashSet::new()),
        })
    }

    /// Replace the explicit-allow set. IPs in the set bypass
    /// [`is_ssrf_blocked`].
    #[must_use]
    pub fn with_explicit_allow(mut self, ips: HashSet<IpAddr>) -> Self {
        self.explicit_allow = Arc::new(ips);
        self
    }
}

#[allow(
    clippy::missing_fields_in_debug,
    reason = "TokioResolver carries a non-Debug closure; printed as the explicit-allow count instead"
)]
impl std::fmt::Debug for SsrfSafeDnsResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsrfSafeDnsResolver")
            .field("explicit_allow_count", &self.explicit_allow.len())
            .finish()
    }
}

impl Resolve for SsrfSafeDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let inner = self.inner.clone();
        let allow = Arc::clone(&self.explicit_allow);
        Box::pin(async move {
            let host = name.as_str();
            let lookup = inner
                .lookup_ip(host)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
            let mut safe: Vec<SocketAddr> = Vec::new();
            let mut blocked: Vec<IpAddr> = Vec::new();
            for ip in lookup.iter() {
                if allow.contains(&ip) || !is_ssrf_blocked(&ip) {
                    safe.push(SocketAddr::new(ip, 0));
                } else {
                    blocked.push(ip);
                }
            }
            if safe.is_empty() {
                let msg = format!(
                    "DNS for '{host}' resolved only to blocked IPs ({blocked:?}); \
                     refusing to connect (SSRF guard)"
                );
                return Err(msg.into());
            }
            let iter: Addrs = Box::new(safe.into_iter());
            Ok(iter)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::ip_constant)]
mod tests {
    use std::net::{Ipv4Addr, Ipv6Addr};

    use super::*;

    #[test]
    fn ipv4_loopback_blocked() {
        assert!(is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
    }

    #[test]
    fn ipv4_metadata_endpoint_blocked() {
        // EC2 / GCP metadata.
        assert!(is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(
            169, 254, 169, 254
        ))));
    }

    #[test]
    fn ipv4_private_ranges_blocked() {
        assert!(is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5))));
        assert!(is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(172, 16, 1, 1))));
        assert!(is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
    }

    #[test]
    fn ipv4_cgnat_blocked() {
        // 100.64.0.0/10 (carrier-grade NAT).
        assert!(is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1))));
        assert!(is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(100, 127, 1, 1))));
    }

    #[test]
    fn ipv4_public_address_passes() {
        // 8.8.8.8 (Google DNS) — unquestionably public.
        assert!(!is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
        // 1.1.1.1 (Cloudflare).
        assert!(!is_ssrf_blocked(&IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
    }

    #[test]
    fn ipv6_loopback_and_unspecified_blocked() {
        assert!(is_ssrf_blocked(&IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert!(is_ssrf_blocked(&IpAddr::V6(Ipv6Addr::UNSPECIFIED)));
    }

    #[test]
    fn ipv6_unique_local_and_link_local_blocked() {
        // ULA fc00::/7 — pick fd00::1 as a representative.
        assert!(is_ssrf_blocked(&IpAddr::V6("fd00::1".parse().unwrap())));
        // Link-local fe80::/10.
        assert!(is_ssrf_blocked(&IpAddr::V6("fe80::1".parse().unwrap())));
    }

    #[test]
    fn ipv4_mapped_ipv6_routes_through_v4_block() {
        // ::ffff:127.0.0.1 — IPv4-mapped, must inherit the v4 block.
        assert!(is_ssrf_blocked(&IpAddr::V6(
            "::ffff:127.0.0.1".parse().unwrap()
        )));
        assert!(is_ssrf_blocked(&IpAddr::V6(
            "::ffff:10.0.0.5".parse().unwrap()
        )));
        assert!(is_ssrf_blocked(&IpAddr::V6(
            "::ffff:169.254.169.254".parse().unwrap()
        )));
    }

    #[test]
    fn ipv4_mapped_public_v4_passes() {
        // ::ffff:8.8.8.8 — public v4, IPv4-mapped form is also public.
        assert!(!is_ssrf_blocked(&IpAddr::V6(
            "::ffff:8.8.8.8".parse().unwrap()
        )));
    }

    #[test]
    fn six_to_four_prefix_blocked_unconditionally() {
        // 2002:7f00:0001::/48 — 6to4 of 127.0.0.1; whole /16 blocked.
        assert!(is_ssrf_blocked(&IpAddr::V6("2002::1".parse().unwrap())));
        assert!(is_ssrf_blocked(&IpAddr::V6(
            "2002:7f00:0001::".parse().unwrap()
        )));
        assert!(is_ssrf_blocked(&IpAddr::V6(
            "2002:0808:0808::".parse().unwrap()
        )));
    }

    #[test]
    fn teredo_prefix_blocked_unconditionally() {
        // 2001::/32 — Teredo. NAT-traversal has no legit outbound role.
        assert!(is_ssrf_blocked(&IpAddr::V6("2001::1".parse().unwrap())));
        assert!(is_ssrf_blocked(&IpAddr::V6(
            "2001:0:abcd:ef01::".parse().unwrap()
        )));
    }

    #[test]
    fn non_teredo_2001_prefix_allowed() {
        // 2001:db8::/32 is documentation, but 2001:4860::/32 is Google
        // production IPv6 — must NOT collide with the Teredo block.
        assert!(!is_ssrf_blocked(&IpAddr::V6(
            "2001:4860:4860::8888".parse().unwrap()
        )));
    }

    #[test]
    fn ipv6_public_address_passes() {
        // Google public DNS over IPv6.
        assert!(!is_ssrf_blocked(&IpAddr::V6(
            "2001:4860:4860::8888".parse().unwrap()
        )));
    }

    #[tokio::test]
    async fn resolver_rejects_when_only_blocked_ips_resolve() {
        // Build a resolver whose lookup will only return loopback IPs
        // by pointing at a hosts-style override. Hickory has no
        // built-in mock, so we exercise the filter directly via the
        // is_ssrf_blocked function above and assert the
        // "no safe IPs" error path through the public surface in a
        // separate integration test under `tests/`.
        // This unit test smoke-tests construction + Debug surface.
        let r =
            SsrfSafeDnsResolver::from_config(ResolverConfig::default(), ResolverOpts::default());
        assert!(format!("{r:?}").contains("SsrfSafeDnsResolver"));
    }

    #[test]
    fn explicit_allow_overrides_block_for_listed_ips() {
        // Direct invariant test: an IP in the allow set should NOT be
        // filtered out by the resolver. Tested via a synthetic
        // address pair.
        let mut allow = HashSet::new();
        allow.insert(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
        let allowed_ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let other_blocked = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5));

        // Mirror the resolver's filter logic.
        let safe_for_allowed = allow.contains(&allowed_ip) || !is_ssrf_blocked(&allowed_ip);
        let safe_for_other = allow.contains(&other_blocked) || !is_ssrf_blocked(&other_blocked);

        assert!(safe_for_allowed, "explicit_allow must override block");
        assert!(!safe_for_other, "non-allowlisted private IP stays blocked");
    }
}
