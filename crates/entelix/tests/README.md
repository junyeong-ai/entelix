# entelix integration tests — operator pattern

The shipped test suite exercises every codec, transport, and agent
surface against deterministic mocks. **No live-provider tests run
in CI** — they require API keys and are flaky against vendor rate
limits. Operators with credentials run their own integration tests
following the pattern below.

## Recommended layout

```
your-app/
├─ Cargo.toml
└─ tests/
    └─ live_anthropic.rs   ← marked #[ignore], opt-in only
```

```rust
//! Live Anthropic round-trip — `cargo test -- --ignored`.

use entelix::{ChatModel, ExecutionContext};
use entelix::codecs::AnthropicMessagesCodec;
use entelix::transports::DirectTransport;
use entelix::ir::{Message, ModelRequest};
use entelix::auth::ApiKeyProvider;
use secrecy::SecretString;

#[tokio::test]
#[ignore = "live API call — opt-in via `cargo test -- --ignored`"]
async fn anthropic_round_trip() {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY required for live test");
    let auth = ApiKeyProvider::anthropic(SecretString::from(api_key));
    let transport = DirectTransport::new("https://api.anthropic.com", auth);
    let model = ChatModel::new(
        AnthropicMessagesCodec::new(),
        transport,
        "claude-haiku-4-5",
    );
    let request = ModelRequest::default()
        .with_model("claude-haiku-4-5")
        .with_messages(vec![Message::user("Reply with exactly: pong")])
        .with_max_tokens(20);

    let resp = model
        .complete(request, &ExecutionContext::new().with_tenant_id("test-tenant"))
        .await
        .unwrap();
    assert!(
        resp.content.iter().any(|p| matches!(p, entelix::ir::ContentPart::Text { text, .. } if text.contains("pong"))),
        "expected 'pong' in response, got {:?}",
        resp.content,
    );
}
```

## Running

```bash
ANTHROPIC_API_KEY="sk-ant-..." cargo test --test live_anthropic -- --ignored
```

Without `--ignored`, the test is filtered out by the harness — the
`#[ignore]` attribute is the standard Rust idiom for opt-in tests.

## Per-provider env-var conventions

| Provider | Env var | Crate / codec |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `AnthropicMessagesCodec` (entelix-core) |
| OpenAI Chat | `OPENAI_API_KEY` | `OpenAiChatCodec` (entelix-core) |
| OpenAI Responses | `OPENAI_API_KEY` | `OpenAiResponsesCodec` (entelix-core) |
| Gemini | `GEMINI_API_KEY` | `GeminiCodec` (entelix-core) |
| Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION` | `BedrockConverseCodec` + `BedrockTransport` (entelix-cloud) |
| Vertex | `GOOGLE_APPLICATION_CREDENTIALS` (path to service-account JSON) | `VertexTransport` (entelix-cloud) |
| Azure Foundry | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `FoundryTransport` (entelix-cloud) |

## Cancellation budgets

Live tests should set a deadline so a stuck vendor call doesn't
hang the harness:

```rust
let ctx = ExecutionContext::new()
    .with_tenant_id("test-tenant")
    .with_deadline(tokio::time::Instant::now() + std::time::Duration::from_secs(30));
```

## Cost discipline

Live tests cost money. Bound them:

- Use the cheapest tier of each provider (`claude-haiku-4-5`, `gpt-5.1-mini`, `gemini-2.0-flash-lite`, `nova-micro`).
- Set `max_tokens` low (50–200).
- Run only on demand, not in CI loops.

Dashboards built from `gen_ai.usage.cost` (see
`docs/observability/`) surface the spend per test if you forward
traces during the run.
