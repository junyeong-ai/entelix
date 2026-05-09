//! Compaction → codec wire-format round trip.
//!
//! The auto-compaction pipeline routes through three pair-aware
//! transformations:
//!
//! 1. `messages_to_events(.)` — `Vec<Message>` → `Vec<GraphEvent>`
//! 2. `Compactor::compact(.)` — typed `Turn` grouping with sealed
//!    `ToolPair` invariant
//! 3. `CompactedHistory::to_messages(.)` — back to `Vec<Message>`
//!
//! The round-tripped messages must encode cleanly through every
//! shipped codec. A regression here would manifest as a wire-format
//! 400 the moment a vendor saw a compacted conversation — exactly
//! the footgun pydantic-ai #4137 catalogues — so we pin the
//! contract here once per codec.
//!
//! For each codec the test asserts:
//! - `Codec::encode` succeeds without `Error::InvalidRequest`
//! - The encode emits **zero** `ModelWarning::LossyEncode` for the
//!   compacted shape (the round-trip itself is lossless; lossy
//!   encode would mean a compaction bug, not a pre-existing IR
//!   limitation)
//! - The wire body is non-empty (compacted history was actually
//!   serialised, not silently dropped)
//!
//! Tests return `Result` and propagate errors via `?` so a regression
//! surfaces as a typed failure rather than a panic from `.unwrap()` —
//! cleaner test-failure diagnostics and no `#![allow(unwrap_used)]`
//! hatch over the workspace lint.

use entelix_core::ExecutionContext;
use entelix_core::Result;
use entelix_core::codecs::{
    AnthropicMessagesCodec, BedrockConverseCodec, Codec, GeminiCodec, OpenAiChatCodec,
    OpenAiResponsesCodec,
};
use entelix_core::ir::{
    ContentPart, Message, ModelRequest, ModelWarning, Role, ToolResultContent, ToolSpec,
};
use entelix_session::{Compactor, HeadDropCompactor, messages_to_events};

fn tool_use_conversation() -> Vec<Message> {
    vec![
        Message::new(Role::User, vec![ContentPart::text("compute 1+1")]),
        Message::new(
            Role::Assistant,
            vec![
                ContentPart::text("calling calculator"),
                ContentPart::ToolUse {
                    id: "call_1".to_owned(),
                    name: "calculator".to_owned(),
                    input: serde_json::json!({"expr": "1+1"}),
                },
            ],
        ),
        Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                tool_use_id: "call_1".to_owned(),
                name: "calculator".to_owned(),
                content: ToolResultContent::Text("2".to_owned()),
                is_error: false,
                cache_control: None,
            }],
        ),
        Message::new(Role::Assistant, vec![ContentPart::text("the answer is 2")]),
    ]
}

async fn round_trip_through_compactor(messages: Vec<Message>) -> Result<Vec<Message>> {
    let events = messages_to_events(&messages)?;
    let history = HeadDropCompactor
        .compact(&events, usize::MAX, &ExecutionContext::new())
        .await?;
    Ok(history.to_messages())
}

fn calculator_tool_spec() -> ToolSpec {
    ToolSpec::function(
        "calculator",
        "simple arithmetic",
        serde_json::json!({
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        }),
    )
}

fn assert_clean_encode<C: Codec>(codec: &C, messages: Vec<Message>) -> Result<()> {
    let request = ModelRequest {
        model: "test-model".to_owned(),
        messages,
        tools: vec![calculator_tool_spec()],
        max_tokens: Some(1024),
        ..ModelRequest::default()
    };
    let encoded = codec.encode(&request)?;
    assert!(
        !encoded.body.is_empty(),
        "encoded wire body must be non-empty",
    );
    let lossy: Vec<&ModelWarning> = encoded
        .warnings
        .iter()
        .filter(|w| matches!(w, ModelWarning::LossyEncode { .. }))
        .collect();
    assert!(
        lossy.is_empty(),
        "lossy encode on compacted round-trip: {lossy:?}",
    );
    Ok(())
}

#[tokio::test]
async fn anthropic_encodes_compacted_round_trip_cleanly() -> Result<()> {
    let messages = round_trip_through_compactor(tool_use_conversation()).await?;
    assert_clean_encode(&AnthropicMessagesCodec, messages)
}

#[tokio::test]
async fn openai_chat_encodes_compacted_round_trip_cleanly() -> Result<()> {
    let messages = round_trip_through_compactor(tool_use_conversation()).await?;
    assert_clean_encode(&OpenAiChatCodec, messages)
}

#[tokio::test]
async fn openai_responses_encodes_compacted_round_trip_cleanly() -> Result<()> {
    let messages = round_trip_through_compactor(tool_use_conversation()).await?;
    assert_clean_encode(&OpenAiResponsesCodec, messages)
}

#[tokio::test]
async fn gemini_encodes_compacted_round_trip_cleanly() -> Result<()> {
    let messages = round_trip_through_compactor(tool_use_conversation()).await?;
    assert_clean_encode(&GeminiCodec, messages)
}

#[tokio::test]
async fn bedrock_converse_encodes_compacted_round_trip_cleanly() -> Result<()> {
    let messages = round_trip_through_compactor(tool_use_conversation()).await?;
    assert_clean_encode(&BedrockConverseCodec, messages)
}
