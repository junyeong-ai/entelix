//! Process-wide TLS / crypto-provider installation.
//!
//! `reqwest 0.13`'s `rustls` feature delegates [`rustls::crypto::CryptoProvider`]
//! selection to the application — without an installed provider the
//! first HTTPS request panics inside `rustls`. entelix transports
//! ([`crate::transports::DirectTransport`], `BedrockTransport`,
//! `VertexTransport`, `FoundryTransport`) all flow through reqwest, so
//! every binary that constructs an entelix transport must install a
//! provider before issuing a request.
//!
//! [`install_default_tls`] is the one-liner the application calls
//! near the top of `main`. It installs the `aws-lc-rs` provider — a
//! FIPS-validated, hardware-accelerated build that aligns with the
//! AWS SigV4 stack `entelix-cloud` already pulls in for Bedrock.
//! Idempotent: the first install wins, subsequent calls are no-ops,
//! and applications that pre-install a different provider keep
//! their selection (the helper does not overwrite).

/// Install the default `rustls` crypto provider for entelix
/// transports. Call once near the top of `main`, before any
/// `Transport::send` runs.
///
/// Idempotent — repeated calls are no-ops, and an
/// application-installed provider takes precedence over this
/// helper. Recommended placement:
///
/// ```ignore
/// fn main() {
///     entelix::install_default_tls();
///     // … construct transports, build agents, run …
/// }
/// ```
///
/// The helper exists because `reqwest 0.13` defers crypto provider
/// selection to the application; without an installed provider the
/// first reqwest call panics inside `rustls::crypto`. entelix wraps
/// the install behind one symbol so consumers do not reach for the
/// `rustls` crate directly.
pub fn install_default_tls() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
}
