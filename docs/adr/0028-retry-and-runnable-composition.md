# ADR-0028 — Retry layer + `Runnable` composition surface

* **Status**: Accepted
* **Date**: 2026-04-27
* **Drivers**: Phase 8C
* **Supersedes**: nothing — formalises retry semantics that prior
  phases left as undeclared territory.

## Context

Two gaps surfaced in audit:

1. **Retry is unwired**. `entelix_core::backoff::ExponentialBackoff`
   produces correct delay sequences with jitter, but no production
   path *invokes* it. `DirectTransport::send` makes one HTTP request
   and returns whatever comes back — transient 5xx and network blips
   fail the call instead of being retried. Three earlier phases
   referenced "retry" in comments and documentation; the
   implementation never landed.
2. **`RunnableExt` is too narrow**. The composition trait exposes
   `.pipe()` and `.stream_with()` only. Operators reaching for the
   LCEL-equivalent surface (`.with_retry`, `.with_fallbacks`, `.map`,
   `.with_config`, `.with_timeout`) have to write boilerplate
   wrappers — every consumer reinvents the same five adapters with
   subtly different semantics.

Both gaps are addressed together because they share an organising
question: **where does retry policy live?** Three plausible homes:

- **Inside the transport.** Worst choice. Couples retry policy to
  the wire layer; impossible to compose with tools, `Runnable`s, or
  agent recipes without duplicating logic.
- **Inside `tower::Layer<Service<ModelInvocation>>`.** Right place
  for model-call retry. Already aligned with `OtelLayer` /
  `PolicyLayer` and the `*Layer` / `*Service` paired naming
  (ADR-0010). Composes uniformly with every other middleware.
- **As a `RunnableExt::with_retry` adapter.** Right place for
  generic retry on any `Runnable<I, O>` — useful when the unit being
  retried is an entire chain (`prompt.pipe(model).pipe(parser)`),
  not a single model invocation.

These are not in tension; they answer different questions. Both ship.

## Decision

### `RetryLayer` + `RetryService` — for the model / tool service path

A standard `tower::Layer<S>` paired with `RetryService<S>` (ADR-0010
`*Layer` / `*Service` rule). Wraps any `Service` whose error is
`entelix_core::Error` and whose response is `Clone`-free in the
happy path (we never clone the success value).

```rust
pub struct RetryLayer { policy: RetryPolicy }

#[derive(Clone, Debug)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff: ExponentialBackoff,
    pub classifier: Arc<dyn RetryClassifier>,
}

pub trait RetryClassifier: Send + Sync + std::fmt::Debug {
    fn should_retry(&self, error: &Error, attempt: u32) -> bool;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultRetryClassifier;
```

`DefaultRetryClassifier` retries:

- `Error::Network` (transport / DNS / connect failures)
- `Error::Provider { status: 408 | 429 | 5xx, .. }` (transient HTTP
  classes per RFC-9457 + vendor practice)

It does **not** retry:

- `Error::Auth` (4xx auth never recovers without operator action)
- `Error::Provider { status: 4xx other than 408 | 429 }` (caller bug)
- `Error::InvalidRequest` (encode-time rejection, deterministic)
- `Error::Cancelled` (operator intent — retrying is a bug)
- `Error::Config` (programming error)

Cancellation is checked on **every iteration** (head of each retry
attempt). `ExecutionContext::cancellation` short-circuits the loop
to `Error::Cancelled` rather than burning attempts and clock time
after the operator already pulled the rug.

Composition (an operator wiring retry around a model call):

```rust
let model = ChatModel::new(codec, transport, "claude-opus-4-7")
    .layer(RetryLayer::new(RetryPolicy::standard()))
    .layer(OtelLayer::new("anthropic"));
```

### `RunnableExt` — five new methods

For composition at the `Runnable<I, O>` level, regardless of whether
the inner is a model call, a tool, or an arbitrary chain:

```rust
pub trait RunnableExt<I, O>: Runnable<I, O> + Sized + 'static {
    fn pipe<P, R>(self, next: R) -> RunnableSequence<I, O, P>;       // existing

    fn with_retry(self, policy: RetryPolicy) -> Retrying<Self, I, O>;
    fn with_fallbacks<R>(self, fallbacks: Vec<R>) -> Fallback<Self, R, I, O>
        where R: Runnable<I, O> + 'static;
    fn map<F, P>(self, f: F) -> Mapping<Self, F, I, O, P>
        where F: Fn(O) -> P + Send + Sync + 'static, P: Send + 'static;
    fn with_config<F>(self, configurer: F) -> Configured<Self, F, I, O>
        where F: Fn(&mut ExecutionContext) + Send + Sync + 'static;
    fn with_timeout(self, timeout: Duration) -> Timed<Self, I, O>;
}
```

Each adapter is a concrete `pub struct` implementing
`Runnable<I, O>`. **No boxing** in the composition path — chains
stay zero-cost in the steady state, with `dyn Runnable` only at the
explicit `erase()` boundary (`AnyRunnable`, F12).

#### `RetryClassifier` reused for fallbacks

`with_fallbacks` is "try this; on a retryable error, try the next".
Whether an error is "retryable" or "fallback-eligible" is the same
question (5xx/429/Network = transient; 4xx/Auth = permanent). The
fallback adapter reuses `RetryClassifier`, parameterised the same
way `RetryPolicy` is — no parallel taxonomy.

```rust
let model = primary
    .with_fallbacks(vec![secondary, tertiary])
    // uses DefaultRetryClassifier; pass a custom one to override.
```

`with_fallbacks` with no fallbacks degrades to the inner runnable —
zero-cost when the operator hasn't configured any. The classifier
also receives the `attempt` counter (just like in retry) so the same
trait can express "after the 3rd retryable failure, give up and
fall back".

### `DirectTransport` keeps no retry trace

Per the long-term design discipline (no half-implementations, no
"// retry will be wired in a later slice" comments), the transport
is now declared **retry-naive** in its module docs. Retry is purely
the Layer's responsibility; the transport implements one HTTP call
end-to-end and returns the result.

## Consequences

- New module `entelix_core::transports::retry` containing
  `RetryLayer`, `RetryService`, `RetryPolicy`, `RetryClassifier`,
  `DefaultRetryClassifier`. Re-exported from `entelix_core` and the
  `entelix` facade.
- New modules in `entelix_runnable`: `retrying`, `fallback`,
  `mapping`, `configured`, `timed`. Each carries one concrete
  adapter type. Re-exported from `entelix_runnable` lib.rs.
- `RunnableExt` gains five default-method-only entries. Blanket impl
  on every `Runnable<I, O>` keeps `.with_retry(policy)` callable
  without explicit imports beyond the trait.
- Public-api baselines refrozen: `entelix-core`, `entelix-runnable`,
  `entelix`.
- `DirectTransport::send` retains its single-call shape; module doc
  states retry is operator-wired through `RetryLayer`.

## Alternatives considered

1. **Retry semantics on `Tool::execute` directly.** Rejected.
   `Tool::execute` is the single-method invariant 4 contract — adding
   retry there bloats every implementation with policy code that
   belongs at composition.
2. **Hard-coded retry inside `DirectTransport`.** Rejected. Forces
   one policy on every consumer; impossible to compose with tools or
   `Runnable` chains.
3. **`Result<T, Either<Permanent, Transient>>` taxonomy.** Rejected.
   Bifurcating the error type duplicates the typed `Error` enum and
   leaks retry concerns into every caller.
4. **Separate `FallbackClassifier` trait.** Rejected. The error
   shape that justifies a retry is the same one that justifies a
   fallback — single `RetryClassifier` keeps users out of trait-
   selection paralysis.

## References

- ADR-0010 §"Type suffix table" (`*Layer` / `*Service` paired rule)
- F3 (cancellation propagation through async APIs)
- F12 (`Runnable<I,O>` dynamic dispatch via `AnyRunnable`)
- `entelix_core::backoff::ExponentialBackoff` (existing primitive,
  now wired)
