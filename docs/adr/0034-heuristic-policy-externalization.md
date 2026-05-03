# ADR 0034 — Heuristic policy externalisation (invariant #17)

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 3 of the post-7-차원-audit roadmap

## Context

The 7-차원 heuristic-risk audit (H4 / H5 / H6 + F-rec) flagged four
sites where embedded heuristics were silently shaping behaviour:

- **H5** — `RetryClassifier::should_retry → bool` discarded the
  vendor's `Retry-After` header. The retry layer ran its own
  exponential backoff with jitter, ignoring an explicit cooldown
  the vendor had taken the trouble to compute. `Error::Provider {
  status: 0 }` collapsed network / TLS / DNS / connect failures
  into a single sentinel, foreclosing typed retry decisions. And
  there was no idempotency identifier flowing on the wire — a
  client timeout that raced a server-side success could double-charge.
- **H6** — supervisor router emitted a bare `String` for the next
  agent. An LLM-driven router that hallucinated a name not in the
  registry would either silently dead-end or trip a runtime
  conditional-edge mismatch — a single typo became an opaque
  routing failure.
- **F-rec** — `ReActAgentBuilder` never exposed
  `with_recursion_limit`; recipes that needed to override the
  graph's `DEFAULT_RECURSION_LIMIT` (25) had to drop down to
  `build_react_graph` and rebuild the agent shell themselves.
- **probability literals in heuristic-prone code paths** — the
  pattern that shows up across the codebase whenever a heuristic
  hides as a literal (jitter ratio, MMR lambda, summarisation
  threshold).

Each instance was small. Aggregated, they let an SDK look like it
made deliberate decisions when in fact it was deferring to an
embedded coin-flip the operator could not see, override, or even
inspect.

## Decision

Land four typed surfaces plus a static-gate guardrail in one slice.
All variants `#[non_exhaustive]`, every new field type public so
downstream operators depend on them at the type system level.

### `Error::Provider` redesign

`status: u16` sentinel removed. Replaced with:

```rust
pub enum ProviderErrorKind {
    Network, Tls, Dns, Http(u16),
}
pub enum Error {
    Provider {
        kind: ProviderErrorKind,
        message: String,
        hint: Option<String>,
        retry_after: Option<Duration>,
    },
    /* ... */
}
```

Helpers: `Error::provider_http(status, msg, hint)`,
`provider_network(msg, hint)`, `provider_tls(msg, hint)`,
`provider_dns(msg, hint)`, plus `with_retry_after(duration)` to
attach the vendor cooldown when present. Retry classifiers branch
on the typed `kind` rather than parsing a sentinel.

### `RetryDecision` + `Retry-After` honour

`RetryClassifier::should_retry → RetryDecision { retry, after }`.
The `DefaultRetryClassifier` propagates
`Error::Provider::retry_after` straight into `decision.after`;
`RetryService` honours the vendor cooldown ahead of its self-jitter
plan, capped at the configured backoff cap so a malicious vendor
cannot pin a retry loop forever. `parse_retry_after` (RFC 7231,
integer-seconds form) is the canonical parser the chat path calls
when building `Error::provider_http(...)`.

### Idempotency-Key auto-stamp

`ExecutionContext::idempotency_key: Option<Arc<str>>` field +
accessor + `with_idempotency_key` setter. `RetryService` calls
`ctx.ensure_idempotency_key(|| Uuid::new_v4().to_string())` on
first entry — every retry attempt clones the same key, so vendor
dedupe sees one logical call across N attempts.
`DirectTransport::send` and `send_streaming` forward the key on
the `Idempotency-Key` request header.

### `SupervisorDecision` enum

```rust
pub enum SupervisorDecision {
    Agent(String),
    Finish,
}
```

Replaces the prior `Runnable<Vec<Message>, String>` +
`SUPERVISOR_FINISH` sentinel pairing. The `String` arm still
admits hallucinated agent names — if the router produces
`Agent("rsearch")` for a registry that knows `"research"`, the
supervisor logs a structured `tracing::warn!` and routes to
`Finish` rather than dead-ending. LLM-driven routers parse text
into the enum at the boundary; deterministic routers match against
`state.messages` directly.

### `ReActAgentBuilder::with_recursion_limit`

New builder method. `build_react_graph` keeps the default for
backwards-compatible callers; `build_react_graph_with_recursion_limit`
is the explicit-cap variant. The two share an internal
`build_react_graph_inner` so the topology lives in one place.

### `scripts/check-magic-constants.sh`

Static gate over codecs, transports, recipe agents, and the cost
meter. Forbids embedded probability literals (`0.X`) — the
canonical "heuristic in disguise" tell. Doc lines, full-line
comments, version literals, and explicitly-marked
(`// magic-ok: <reason>`) sites are excluded. Other classes of
magic numbers (HTTP statuses, byte caps, token budgets) carry
their own narrower gates already (silent-fallback, codec
consistency, lossy-warning completeness).

## Consequences

✅ Vendors that publish `Retry-After` get their cooldown honoured —
fewer thundering-herd 429 storms. Network / TLS / DNS failures
retry by default (transport class), HTTP 408/425/429/5xx retry
(documented set), 4xx other than those don't. The classification
shows up in `Error::Provider.kind` for observability.
✅ Client timeouts that race server-side success no longer
double-charge — the second attempt presents the same
`Idempotency-Key`.
✅ Supervisor router hallucinations route to `Finish` with a
structured warn instead of silently dead-ending. Routes registered
in the agent list are matched literally.
✅ Operators that need a non-default recursion limit on ReAct
loops set it on the recipe builder, not by dropping down to graph
internals.
✅ The static gate catches future heuristic literals at PR time.
❌ `Error::provider(status, msg, hint)` removed — every call site
chooses an explicit `_http` / `_network` / `_tls` / `_dns` helper.
One-line change at every site (17 sites this slice).
❌ `Error::Provider { status, .. }` pattern matches replaced with
`{ kind, .. }` plus a `ProviderErrorKind::Http(s)` arm. ~10 test
sites.
❌ `SUPERVISOR_FINISH` const removed; existing routers move to
`SupervisorDecision::Finish`. Single-find-replace at call sites.
❌ Probability literals in heuristic-prone paths now require
either externalisation onto a `*Policy` or an inline marker. The
gate tells the operator which.

## Alternatives considered

1. **Keep `status: u16` and add `kind` alongside** — silent
   redundancy: status=0 sentinel persists, retry classifiers
   pattern-match both. Rejected; ADR-0032 (no silent fallback)
   forbids the duplication.
2. **`should_retry → Option<Duration>`** — overloads `Some(0)` and
   `None`. `RetryDecision` separates the two questions cleanly.
   Rejected.
3. **Idempotency key on `EncodedRequest::headers`** — codec would
   stamp the key, but retry would re-encode and produce a new
   UUID. The key has to flow through `ExecutionContext` (the only
   thing that survives one logical call across N attempts).
   Rejected.
4. **`SupervisorDecision::Goto(String) | Finish | Reject`** —
   wider enum than the audit warranted. Started simple; the
   `#[non_exhaustive]` lets us grow without breaking callers.

## References

- 7-차원 audit fork report `audit-heuristic-risks` H4 / H5 / H6 / F-rec.
- ADR-0028 — `RetryClassifier` introduced (this slice extends).
- ADR-0032 — invariant #15 (silent fallback prohibition) — sibling
  contract: `Error::Provider` redesign closes the `status: 0`
  sentinel that #15 implicitly forbade.
- ADR-0035 — sub-agent layer-stack inheritance — same "narrow at
  the typed boundary, never reconstruct" principle.
