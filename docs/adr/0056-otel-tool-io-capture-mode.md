# ADR 0056 — Tool I/O capture mode for OTel `gen_ai.tool.*` events

**Status**: Accepted
**Date**: 2026-05-01
**Decision**: Phase 8 of the post-7-차원-audit roadmap (fourth sub-slice)

## Context

ADR-0009 (OTel adoption) shipped the `gen_ai.tool.start` /
`gen_ai.tool.end` event surface with `gen_ai.tool.input` and
`gen_ai.tool.output` fields populated from `invocation.input`
and `output` respectively. The fields emit verbatim — every
byte of the JSON payload makes it onto the span.

That worked for early development but isn't safe for
production:

- **Size**: A `HttpFetchTool` returning a 100KB JSON response
  blows up span size 100x compared to a slim `Calculator`
  call. OTel exporters aren't infinitely scalable; large
  spans degrade ingest throughput, push storage cost, and
  trip backend payload caps (Datadog 1MB, Grafana Cloud
  varies).
- **PII**: Tool I/O routinely carries user content (raw
  prompts, captured form input, scraped HTML), credentials
  captured from elicitation (ADR-0053), API keys forwarded
  to tools. Operators with PII obligations need a way to
  suppress the field surface entirely without losing the
  rest of the trace.
- **No opt-out**: Operators wanting the cheap-and-quiet
  default had to fork the layer.

Phase 8 of the roadmap explicitly named this as remaining
OTel work alongside cache token telemetry (ADR-0055) and
agent OTel span (next slice candidate).

## Decision

Add a `ToolIoCaptureMode` enum on `OtelLayer` with three
variants — `Off`, `Truncated { max_bytes }`, `Full` — and a
builder method `OtelLayer::with_tool_io_capture(mode)` that
overrides the default. Default mode is
`Truncated { max_bytes: 4096 }`, exposed as the public
constant `DEFAULT_TOOL_IO_TRUNCATION`.

```rust
#[non_exhaustive]
pub enum ToolIoCaptureMode {
    Off,
    Truncated { max_bytes: usize },
    Full,
}

impl OtelLayer {
    pub const fn with_tool_io_capture(mut self, mode: ToolIoCaptureMode) -> Self;
}

pub const DEFAULT_TOOL_IO_TRUNCATION: usize = 4096;
```

The tool dispatch path in `OtelService<S>::call(ToolInvocation)`
runs both `invocation.input.to_string()` and
`output.to_string()` through a `capture_tool_payload` helper
that applies the mode before the field hits
`tracing::event!`.

### Why default Truncated, not Off

Default-Off would sacrifice debuggability for the safe
default. Operators who *want* tool I/O visibility (the
common case in development and most production deployments)
would have to opt in — which means they wouldn't, until they
hit a debugging session and realised the field was missing.

`Truncated { max_bytes: 4096 }` strikes the balance: typical
tool calls (`{"task": "..."}` / `{"output": "..."}`,
calculator results, small JSON payloads) ride through in
full; the worst offenders (HTTP fetch, web search results,
large model-generated content) get capped at a useful
prefix. 4 KiB picked because:

- It's well under the 32 KiB tag-value cap of major OTel
  backends (Datadog, Honeycomb).
- It accommodates the 90th-percentile tool I/O size in
  observed agent workloads.
- It's a power of 2 that round-trips through doc strings
  without rounding-error confusion.

### Why a marker on truncation, not silent cut

`… [truncated, N bytes total]` (with the unicode ellipsis to
distinguish from text containing `...`) tells the operator:

1. The field was truncated (vs. the tool returning a 4 KiB
   response that happened to end at the cap).
2. How much was lost (the `N` is the original full length
   so operators can decide whether to switch the mode for a
   re-run).

Silent truncation would force every operator to inspect
`output_kind` and length math to reason about whether they
saw the full payload.

### Why Off emits `<omitted>`, not absent attribute

A missing field on a span looks identical to "the layer
forgot the field" — operators can't distinguish policy from
bug. `<omitted>` is a load-bearing string: present means
"yes, the layer ran, the policy was Off". Costs ~10 bytes
per dispatch event; cheaper than the operator confusion of
silent absence.

### UTF-8 boundary handling

Truncation is byte-based for predictability (operators
configure `max_bytes` because their backend has a byte cap,
not a char cap). But the byte cap may land mid-UTF-8 char,
which would produce an invalid string in the rendered span.
The helper snaps `cut` back to the nearest char boundary —
typical 1-3 byte penalty, vs. the alternative of either
emitting invalid UTF-8 (downstream parsers fail) or rounding
up past the cap (operator's max_bytes promise broken).

### Why no `Redacted` mode

A `Redacted { redactor: Arc<dyn Redactor> }` mode would let
operators run their own PII scrubber inline. Real
production redactors carry state (regex caches, ML models)
and per-tool config (allowlists). Putting the trait inside
this enum would force the trait into `entelix-otel` and
constrain the operator's redactor design. The right separation:
operators apply redaction in a layer above `OtelLayer` (e.g.,
`PolicyLayer` from `entelix-policy`) so the redacted output
is what the OTel layer sees. Reserved for a future slice if
the layered approach proves clumsy.

### Tests

- 6 unit tests covering each mode shape:
  - `Full` returns input verbatim.
  - `Off` returns `<omitted>` literal.
  - `Truncated` under the cap passes through.
  - `Truncated` over the cap appends marker with full size.
  - `Truncated` snaps to UTF-8 boundary at 4-byte char.
  - Default mode is `Truncated(4096)`.
- Existing 17 tests continue to pass (default mode preserves
  prior visibility for tests with small payloads).

## Consequences

✅ Operators with PII obligations switch to
`ToolIoCaptureMode::Off` per server / per pipeline; they
keep the rest of the OTel surface (timing, model name, cost)
without leaking the I/O body.
✅ Production deployments don't blow up span size on a
single 100KB HTTP fetch — the truncated marker preserves
operator visibility into "what was returned" without the
full bytes.
✅ Default mode preserves backward-compatible behaviour for
small tool payloads (under 4 KiB) — existing tests pass
unchanged.
✅ The `Off` mode emits `<omitted>` instead of absent
attribute — operators distinguish policy from bug.
✅ Truncation marker is informative (full byte count
preserved) so operators can decide whether to re-run with
`Full` mode.
❌ Default 4 KiB cap loses the tail of large tool responses
in the span. Operators who need full payloads switch to
`Full` (with PII responsibility on their layer above).
❌ Public-API baseline grew (3 new exports:
`ToolIoCaptureMode`, `DEFAULT_TOOL_IO_TRUNCATION`, builder
method). Refrozen.
❌ `Redacted` mode not shipped — operators wanting inline
redaction either layer it above `OtelLayer` or wait for the
follow-up slice.

## Alternatives considered

1. **Default `Off` (safest)** — sacrifices debuggability
   for the safe default. Operators who want visibility have
   to opt in, which means they won't until they need it.
   Wrong default for the development experience.
2. **Default `Full` (preserve current)** — keeps current
   behaviour but ships the existing problems (size, PII)
   forward. Production-hostile default.
3. **Per-tool capture mode (`HashMap<ToolName, Mode>`)** —
   over-engineered for the current operator request profile.
   The two-level distinction (off / capped / full) covers
   the real policy decisions. Per-tool granularity can land
   later if demand materialises.
4. **`Redacted { redactor }` mode** — see "Why no Redacted"
   above. Layered approach (PolicyLayer above OtelLayer)
   is the right separation; this slice doesn't preclude it.
5. **Char-based truncation (`max_chars` instead of `max_bytes`)**
   — operator backends cap on bytes, not chars. Byte cap +
   UTF-8 snap is the predictable shape; char cap would be a
   lie for any backend that counts bytes.

## Operator usage patterns

**Production with PII obligations** (banking, healthcare):
```rust
let layer = OtelLayer::new("anthropic")
    .with_tool_io_capture(ToolIoCaptureMode::Off);
```

**Default development / staging** (no opt-in needed):
```rust
let layer = OtelLayer::new("anthropic"); // 4 KiB truncation
```

**Production with own redactor + needing full tool I/O**:
```rust
let layer = OtelLayer::new("anthropic")
    .with_tool_io_capture(ToolIoCaptureMode::Full);
// Compose with PolicyLayer above for redaction
```

**Cost-sensitive backend (Datadog seat-priced spans)**:
```rust
let layer = OtelLayer::new("anthropic")
    .with_tool_io_capture(ToolIoCaptureMode::Truncated { max_bytes: 1024 });
```

## References

- ADR-0009 — OTel GenAI semconv adoption (parent).
- ADR-0055 — cache token telemetry (sibling Phase 8 slice).
- 7-차원 roadmap §S9 — Phase 8 (MCP + OTel completion),
  fourth sub-slice.
- entelix-otel CLAUDE.md — invariant 12 cost emission +
  attribute-name discipline.
- `crates/entelix-otel/src/layer.rs` — enum + builder +
  `capture_tool_payload` helper + 6 new unit tests.
- `crates/entelix-otel/src/lib.rs` — re-exports.
