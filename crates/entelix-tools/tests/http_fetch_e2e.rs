//! `HttpFetchTool` end-to-end via `wiremock`. Exercises:
//!
//! - happy path — allowlist permit, GET, JSON body roundtrip.
//! - body cap — large response truncated, `truncated: true`.
//! - method allowlist — POST refused when only GET allowed.
//! - host allowlist — bare IP refused.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::net::IpAddr;

use entelix_core::AgentContext;
use entelix_core::tools::Tool;
use entelix_tools::{HostAllowlist, HttpFetchTool};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn allowlist_for(server: &MockServer) -> HostAllowlist {
    let url = url::Url::parse(&server.uri()).unwrap();
    let host = url.host_str().unwrap();
    let mut allow = HostAllowlist::new();
    if let Ok(ip) = host.parse::<IpAddr>() {
        allow = allow.add_exact_ip(ip);
    } else {
        allow = allow.add_exact_host(host);
    }
    allow
}

#[tokio::test]
async fn happy_path_get_returns_status_and_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/hello"))
        .respond_with(ResponseTemplate::new(200).set_body_string("hi there"))
        .mount(&server)
        .await;

    let tool = HttpFetchTool::builder()
        .with_allowlist(allowlist_for(&server))
        .build()
        .unwrap();
    let out = tool
        .execute(
            json!({"url": format!("{}/hello", server.uri())}),
            &AgentContext::default(),
        )
        .await
        .unwrap();
    assert_eq!(out["status"], 200);
    assert_eq!(out["body"], "hi there");
    assert_eq!(out["truncated"], false);
}

#[tokio::test]
async fn body_cap_truncates_and_marks_truncated() {
    let server = MockServer::start().await;
    let big = "x".repeat(2048);
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(big))
        .mount(&server)
        .await;

    let tool = HttpFetchTool::builder()
        .with_allowlist(allowlist_for(&server))
        .with_max_response_bytes(512)
        .build()
        .unwrap();
    let out = tool
        .execute(json!({"url": server.uri()}), &AgentContext::default())
        .await
        .unwrap();
    assert_eq!(out["truncated"], true);
    assert!(
        out["body"].as_str().unwrap().len() <= 512,
        "body length: {}",
        out["body"].as_str().unwrap().len()
    );
}

#[tokio::test]
async fn host_outside_allowlist_is_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    // Build with an allowlist that doesn't cover the mock server.
    let tool = HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_exact_host("not-the-server.test"))
        .build()
        .unwrap();
    let err = tool
        .execute(json!({"url": server.uri()}), &AgentContext::default())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("not on the allowlist"));
}

#[tokio::test]
async fn method_allowlist_rejects_post_by_default() {
    let server = MockServer::start().await;
    let tool = HttpFetchTool::builder()
        .with_allowlist(allowlist_for(&server))
        .build()
        .unwrap();
    let err = tool
        .execute(
            json!({"url": server.uri(), "method": "POST"}),
            &AgentContext::default(),
        )
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("not allowed"));
}

#[tokio::test]
async fn unsupported_scheme_is_rejected_before_network() {
    let tool = HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_exact_host("example.com"))
        .build()
        .unwrap();
    let err = tool
        .execute(
            json!({"url": "file:///etc/passwd"}),
            &AgentContext::default(),
        )
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("unsupported URL scheme"));
}

#[tokio::test]
async fn dns_rebinding_block_rejects_hostname_resolving_to_loopback() {
    // Verifies the SsrfSafeDnsResolver is actually wired into the
    // reqwest client — the URL-host allowlist check passes
    // (`localhost` is admitted), but the DNS resolver intercepts
    // the connect-time lookup and refuses because the only A
    // record for `localhost` (per the operating system's hosts
    // file) is `127.0.0.1`, which is in the SSRF block range.
    //
    // If the resolver were silently ignored at builder time
    // (regression risk the audit flagged), this request would
    // succeed and read from a local service. The assertion below
    // catches that.
    let tool = HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_exact_host("localhost"))
        .build()
        .unwrap();
    let err = tool
        .execute(
            // Port 9 (discard) so even if the SSRF block somehow
            // failed and the request hit loopback, no real local
            // service would respond. The block fires earlier
            // anyway — the connect never starts.
            json!({"url": "http://localhost:9/probe"}),
            &AgentContext::default(),
        )
        .await
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("blocked") || msg.contains("SSRF") || msg.contains("network"),
        "expected SSRF block error, got: {msg}"
    );
}

#[tokio::test]
async fn explicit_ip_allow_round_trips_against_loopback_listener() {
    // Counterpart to the test above: when the operator opts a
    // specific loopback address into the allowlist via
    // `allow_ip_exact`, the SsrfSafeDnsResolver must seed the
    // explicit-allow set with that IP and let the connect through.
    // Uses wiremock (which binds to 127.0.0.1) so we have a real
    // local listener to round-trip against.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/probe"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    let server_url = url::Url::parse(&server.uri()).unwrap();
    let server_ip: IpAddr = server_url.host_str().unwrap().parse().unwrap();

    let tool = HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_exact_ip(server_ip))
        .build()
        .unwrap();
    let out = tool
        .execute(
            json!({"url": format!("{}/probe", server.uri())}),
            &AgentContext::default(),
        )
        .await
        .unwrap();
    assert_eq!(out["status"], 204);
}

#[tokio::test]
async fn redirect_to_non_allowlisted_host_is_rejected() {
    // The first server is in the allowlist and returns a 302 to a
    // hostname that is not. The redirect policy must refuse the
    // hop *before* DNS — the target uses the reserved `.invalid`
    // TLD (RFC 2606) so a successful test does not accidentally
    // hit the public internet.
    //
    // Without the policy gate, the host-allowlist check on the
    // initial URL is meaningless on any path that redirects.
    let permit = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/start"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("location", "http://forbidden.invalid/secret"),
        )
        .mount(&permit)
        .await;

    let permit_ip: IpAddr = url::Url::parse(&permit.uri())
        .unwrap()
        .host_str()
        .unwrap()
        .parse()
        .unwrap();
    let tool = HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_exact_ip(permit_ip))
        .build()
        .unwrap();

    let err = tool
        .execute(
            json!({"url": format!("{}/start", permit.uri())}),
            &AgentContext::default(),
        )
        .await
        .unwrap_err();
    let msg = format!("{err}");
    // reqwest collapses redirect-policy errors into "error following
    // redirect for url (...)" at Display — the inner reason is in
    // the source chain. The presence of that phrase is sufficient
    // evidence that the policy fired and refused the hop.
    assert!(
        msg.contains("error following redirect"),
        "expected redirect-rejection error, got: {msg}"
    );
}

#[tokio::test]
async fn redirect_within_allowlist_is_followed() {
    // Sanity: same setup but with both servers in the allowlist —
    // the redirect must succeed end-to-end.
    let a = MockServer::start().await;
    let b = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/start"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("location", format!("{}/end", b.uri())),
        )
        .mount(&a)
        .await;
    Mock::given(method("GET"))
        .and(path("/end"))
        .respond_with(ResponseTemplate::new(200).set_body_string("done"))
        .mount(&b)
        .await;

    let a_ip: IpAddr = url::Url::parse(&a.uri())
        .unwrap()
        .host_str()
        .unwrap()
        .parse()
        .unwrap();
    let b_ip: IpAddr = url::Url::parse(&b.uri())
        .unwrap()
        .host_str()
        .unwrap()
        .parse()
        .unwrap();
    let tool = HttpFetchTool::builder()
        .with_allowlist(HostAllowlist::new().add_exact_ip(a_ip).add_exact_ip(b_ip))
        .build()
        .unwrap();
    let out = tool
        .execute(
            json!({"url": format!("{}/start", a.uri())}),
            &AgentContext::default(),
        )
        .await
        .unwrap();
    assert_eq!(out["status"], 200);
    assert_eq!(out["body"], "done");
    assert!(out["final_url"].as_str().unwrap().ends_with("/end"));
}
