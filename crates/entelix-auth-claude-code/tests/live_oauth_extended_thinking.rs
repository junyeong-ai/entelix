//! Live-API smoke for the cross-vendor opaque round-trip carrier
//! (`ProviderEchoSnapshot`) under Claude Code OAuth +
//! extended-thinking + tool round-trip.
//!
//! The end-to-end contract being pinned:
//!
//! 1. Turn 1 — operator calls Anthropic Messages API with
//!    `extended_thinking` enabled and a function tool. The assistant
//!    emits a `Thinking` block carrying a non-empty `signature`
//!    (decoded onto `ContentPart::Thinking::provider_echoes` under
//!    the `"anthropic-messages"` provider key) followed by a
//!    `ToolUse` block.
//! 2. Turn 2 — operator re-submits the conversation as
//!    `[user, assistant(Thinking + ToolUse), tool_result]`. The
//!    Anthropic gateway hash-validates the round-tripped signature;
//!    a missing or tampered signature on this turn rejects the
//!    request with HTTP 400 (`invalid signature`). A clean 2xx +
//!    final assistant text proves the carrier round-tripped
//!    bit-for-bit.
//!
//! Variants:
//!
//! - `ENTELIX_LIVE_CLAUDE_CODE_MODEL` overrides the default model
//!   (`claude-haiku-4-5`). All Claude 4.x tiers support the
//!   extended-thinking + signature surface; haiku is the default
//!   here because the Claude.ai Max tier rate-limits sonnet/opus
//!   for low-volume smoke calls.
//! - `ENTELIX_LIVE_CLAUDE_CODE_PATH` overrides the credential file
//!   location (default: `~/.claude/.credentials.json`). Mirrors the
//!   convention used by the basic-chat live test.
//!
//! ## Cost discipline
//!
//! Two non-streaming Messages calls, `max_tokens = 2048` with
//! `ReasoningEffort::Low` (Anthropic budget = 1024). Anthropic
//! requires `max_tokens > thinking.budget_tokens`. Per-run cost
//! typically pennies; suitable for manual smoke.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::path::PathBuf;
use std::sync::Arc;

use entelix_auth_claude_code::{CLAUDE_CODE_BETA, ClaudeCodeOAuthProvider, FileCredentialStore};
use entelix_core::auth::CredentialProvider;
use entelix_core::codecs::{AnthropicMessagesCodec, Codec};
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{
    AnthropicExt, ContentPart, Message, ModelRequest, ProviderEchoSnapshot, ProviderExtensions,
    ReasoningEffort, Role, ToolKind, ToolSpec,
};
use entelix_core::transports::{DirectTransport, Transport};
use serde_json::json;

const DEFAULT_MODEL: &str = "claude-haiku-4-5";
const TOOL_NAME: &str = "get_weather";

fn resolve_credential_path() -> PathBuf {
    if let Ok(custom) = std::env::var("ENTELIX_LIVE_CLAUDE_CODE_PATH") {
        return PathBuf::from(custom);
    }
    FileCredentialStore::default_claude_path()
        .expect("HOME / USERPROFILE not set — supply ENTELIX_LIVE_CLAUDE_CODE_PATH to override")
}

#[tokio::test]
#[ignore = "live-API: requires `claude` CLI login (~/.claude/.credentials.json) + reasoning-tier model access"]
#[allow(clippy::too_many_lines)]
async fn claude_code_oauth_extended_thinking_signature_round_trip() {
    install_default_tls();
    let path = resolve_credential_path();
    assert!(
        path.exists(),
        "credential file not found at {} — run `claude login` first or set ENTELIX_LIVE_CLAUDE_CODE_PATH",
        path.display()
    );

    let model = std::env::var("ENTELIX_LIVE_CLAUDE_CODE_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let store = FileCredentialStore::with_path(path);
    let provider = ClaudeCodeOAuthProvider::new(store);
    provider
        .resolve()
        .await
        .expect("ClaudeCodeOAuthProvider::resolve must succeed (run `claude login` if expired)");

    let credentials: Arc<dyn CredentialProvider> = Arc::new(provider);
    let codec = AnthropicMessagesCodec::new();
    let transport = DirectTransport::anthropic(credentials).expect("transport build");

    let weather_tool = ToolSpec {
        name: TOOL_NAME.into(),
        description: "Look up the current weather for one city. Always call when asked about \
                      weather."
            .into(),
        kind: ToolKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name, e.g. 'Seoul'" }
                },
                "required": ["city"],
            }),
        },
        cache_control: None,
    };

    let mut transcript = vec![Message::user(
        "What is the weather in Seoul right now? Use the tool, do not guess.",
    )];

    // ── Turn 1 — extended thinking + tool dispatch ───────────────────
    let request_turn1 = ModelRequest {
        model: model.clone(),
        messages: transcript.clone(),
        max_tokens: Some(2048),
        tools: vec![weather_tool.clone()],
        reasoning_effort: Some(ReasoningEffort::Low),
        provider_extensions: ProviderExtensions::default()
            .with_anthropic(AnthropicExt::default().with_betas([CLAUDE_CODE_BETA])),
        ..ModelRequest::default()
    };

    let encoded = codec.encode(&request_turn1).expect("turn 1 encode");
    let ctx = ExecutionContext::new();
    let raw = transport
        .send(encoded, &ctx)
        .await
        .expect("turn 1 OAuth-authenticated send");
    assert!(
        (200..300).contains(&raw.status),
        "turn 1 must succeed; got {}: {}",
        raw.status,
        String::from_utf8_lossy(&raw.body)
    );
    let response_turn1 = codec.decode(&raw.body, Vec::new()).expect("turn 1 decode");

    // Pull the Thinking part — its provider_echoes must carry a
    // non-empty `signature` under `"anthropic-messages"`.
    let thinking_part = response_turn1
        .content
        .iter()
        .find(|p| matches!(p, ContentPart::Thinking { .. }))
        .expect(
            "extended-thinking turn must produce a Thinking block; \
             the model may have skipped reasoning if the prompt did not \
             trigger it — adjust the prompt or use a higher-effort model",
        );
    let ContentPart::Thinking {
        text,
        provider_echoes,
        ..
    } = thinking_part
    else {
        unreachable!()
    };
    assert!(
        !text.is_empty(),
        "Thinking text must not be empty when extended thinking is engaged"
    );
    let signature = ProviderEchoSnapshot::find_in(provider_echoes, "anthropic-messages")
        .and_then(|e| e.payload_str("signature"))
        .expect(
            "Thinking part must carry a non-empty signature on provider_echoes — \
             missing here means the codec failed to decode the wire `signature` field",
        );
    assert!(!signature.is_empty(), "decoded signature must not be empty");

    // Pull the ToolUse part.
    let tool_use = response_turn1
        .content
        .iter()
        .find(|p| matches!(p, ContentPart::ToolUse { .. }))
        .expect("turn 1 must produce a ToolUse block (the prompt forces a tool call)");
    let ContentPart::ToolUse {
        id: tool_use_id, ..
    } = tool_use
    else {
        unreachable!()
    };
    let tool_use_id = tool_use_id.clone();

    // ── Turn 2 — round-trip the Thinking block + ToolResult ────────
    //
    // The Anthropic gateway hash-validates `signature`. If our codec
    // dropped a byte or re-encoded a field, the gateway returns
    // `400 invalid_request_error: signature does not match` —
    // which is exactly the production-blocker semantics we need to
    // verify the carrier handles end-to-end.
    transcript.push(Message::new(
        Role::Assistant,
        response_turn1.content.clone(),
    ));
    transcript.push(Message::tool_result_json(
        &tool_use_id,
        TOOL_NAME,
        json!({
            "city": "Seoul",
            "temperature_celsius": 16,
            "conditions": "partly cloudy"
        }),
    ));

    let request_turn2 = ModelRequest {
        model,
        messages: transcript,
        max_tokens: Some(2048),
        tools: vec![weather_tool],
        reasoning_effort: Some(ReasoningEffort::Low),
        provider_extensions: ProviderExtensions::default()
            .with_anthropic(AnthropicExt::default().with_betas([CLAUDE_CODE_BETA])),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request_turn2).expect("turn 2 encode");
    let raw = transport
        .send(encoded, &ctx)
        .await
        .expect("turn 2 OAuth-authenticated send");
    assert!(
        (200..300).contains(&raw.status),
        "turn 2 must succeed — a 400 here means the signature did not round-trip \
         verbatim (the cross-vendor carrier introduced byte drift). Got {}: {}",
        raw.status,
        String::from_utf8_lossy(&raw.body)
    );
    let response_turn2 = codec.decode(&raw.body, Vec::new()).expect("turn 2 decode");
    let saw_final_text = response_turn2
        .content
        .iter()
        .any(|p| matches!(p, ContentPart::Text { text, .. } if !text.is_empty()));
    assert!(
        saw_final_text,
        "turn 2 must emit a final non-empty text reply citing the tool result"
    );
}
