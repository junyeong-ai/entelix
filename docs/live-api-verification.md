# Live-API verification

The workspace ships `#[ignore]`-gated smoke tests that exercise the
real vendor APIs end-to-end. They are **not** part of the default
`cargo test --workspace` sweep — running them costs cents and
requires operator-supplied credentials.

The smokes serve two release-gate purposes:

1. **Wire-format truth** — verify the codecs / transports continue
   to produce bytes the vendors accept, against whatever spec
   variants those vendors silently roll out.
2. **Account viability** — verify the operator's keys, regions,
   and model SKUs work in their target deployment before a
   release tag is cut.

## Vendor matrix

| Vendor | Crate | Test | Required env vars | Default model | Per-run cost (Apr 2026) |
|---|---|---|---|---|---|
| Anthropic | `entelix-core` | `live_anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` | ~$0.0001 |
| OpenAI Chat | `entelix-core` | `live_openai_chat` | `OPENAI_API_KEY` | `gpt-4o-mini` | ~$0.0001 |
| OpenAI Responses | `entelix-core` | `live_openai_responses` | `OPENAI_API_KEY` | `gpt-4o-mini` | ~$0.0001 |
| Gemini | `entelix-core` | `live_gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` | free tier or ~$0.00001 |
| Bedrock | `entelix-cloud` (`aws` feature) | `live_bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION` (or `ENTELIX_LIVE_BEDROCK_REGION`) | `anthropic.claude-3-5-haiku-20241022-v1:0` | ~$0.0001 |
| OpenAI Embedder | `entelix-memory-openai` | `live_openai_embedder` | `OPENAI_API_KEY` | `text-embedding-3-small` | ~$0.00001 |

A full sweep (one call per vendor) costs **well under $0.01**
total. The tests deliberately set `max_tokens = 16` /
`temperature = 0.0` so the cost ceiling is fixed regardless of
how chatty the model wants to be.

## Optional model overrides

Every smoke honours a `ENTELIX_LIVE_{VENDOR}_MODEL` env var so
operators whose accounts only have access to alternate SKUs can
substitute in-place — for example, a free-tier OpenAI account
forced onto `gpt-4o-mini` while an enterprise account uses
`gpt-4.1-nano`. See the test file's `DEFAULT_MODEL` constant for
the override slot name.

## Running the smokes

### Single vendor

```bash
ANTHROPIC_API_KEY=sk-ant-... \
    cargo test -p entelix-core --test live_anthropic -- --ignored
```

```bash
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \
    cargo test -p entelix-cloud --features aws \
        --test live_bedrock -- --ignored
```

### Full sweep (everything you have keys for)

```bash
ANTHROPIC_API_KEY=... \
OPENAI_API_KEY=... \
GEMINI_API_KEY=... \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \
    cargo test --workspace --all-features -- --ignored \
        live_anthropic live_openai_chat live_openai_responses \
        live_gemini live_bedrock live_openai_embedder
```

A vendor whose key is missing fails fast with an `expect` panic
on the env-var lookup — operators see exactly which credential
to set.

### CI manual trigger

`.github/workflows/live-api.yml` exposes a `workflow_dispatch`
button that runs the full sweep with secrets pulled from the
repository's GitHub Secrets store. Trigger it manually before
cutting a release tag — there is **no scheduled run**, no
trigger on push, and no automatic invocation. Cost discipline
demands explicit operator intent.

## What each smoke verifies

- **Status** — the vendor accepts our codec-produced bytes (2xx).
- **Decode** — the response decodes into a non-empty
  `ModelResponse` (chat smokes) or a properly-dimensioned
  `Embedding` (embedder smoke).
- **Stop reason** — chat smokes assert the response carries a
  recognised `StopReason` variant (no silent unknown). The model
  is asked for a one-word reply (`"ok"`); the smoke does **not**
  match the literal text — vendors paraphrase.
- **Usage** — `Usage::input_tokens` / `Usage::output_tokens` (or
  `EmbeddingUsage::input_tokens`) populate so cost telemetry
  downstream of these surfaces is honest.

## What each smoke does NOT verify

- **Streaming** — non-streaming round-trips only. Streaming
  smokes ride on a follow-up slice when the vendor SSE shape is
  worth budgeting for.
- **Tool calling** — the smoke uses a one-turn user message with
  no tools. Tool-call smokes need a multi-turn fixture and are
  out of scope for the release gate.
- **Production load** — one call per run. Latency / throughput /
  rate-limit behaviour is operator territory, not a release gate.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `expected 2xx, got 401` | Key invalid or expired. Rotate. |
| `expected 2xx, got 404 — model …` | Model id not granted to this account. Set `ENTELIX_LIVE_{VENDOR}_MODEL`. |
| `expected 2xx, got 429` | Rate-limited. Wait or move to a higher tier. |
| `expected 2xx, got 500` | Vendor-side outage. Consult the vendor's status page; not an entelix bug. |
| Bedrock smoke fails with `NoCredentialProvider` | Default chain came up empty. Set explicit `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` for the test process. |
| `unexpected stop_reason` | Vendor introduced a new stop reason. The codec emits a `ModelWarning::UnknownStopReason` on the same call (visible in the response's `warnings` field) — file a tracking issue and update the smoke once a backwards-compat plan exists. |

## Adding a new vendor smoke

1. Add `crates/<crate>/tests/live_<vendor>.rs` mirroring
   `live_anthropic.rs`'s shape (env-var lookup, `DEFAULT_MODEL`
   constant, codec + transport wiring, status + decode
   assertions).
2. Update the matrix table in this document with the new env-var
   names and per-run cost estimate.
3. Add the test name to the full-sweep CI workflow's manual
   trigger.
4. Open a PR; CI runs the rest of the gate suite (the new live
   smoke is `#[ignore]`d so it does not block).
