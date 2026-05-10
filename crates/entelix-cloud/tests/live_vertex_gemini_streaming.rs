//! Live-API smoke for [`VertexGeminiCodec`] +
//! [`VertexTransport::send_streaming`] driven through the
//! high-level [`ChatModel::stream_deltas`] surface.
//!
//! Verifies the streaming spine end-to-end against Vertex
//! `:streamGenerateContent?alt=sse`: the codec's path rewrite
//! survives streaming encode, the transport opens the SSE
//! connection through ADC-issued bearer tokens, the SSE parser
//! decodes each event into a typed [`StreamDelta`], the tapped
//! `StreamAggregator` reconstructs the final
//! [`entelix_core::ir::ModelResponse`], and `ModelStream.completion`
//! resolves with that aggregated shape.
//!
//! `#[ignore]`-gated and feature-gated behind `gcp` so a default
//! `cargo test --workspace` skips it. Run with:
//!
//! ```text
//! ENTELIX_LIVE_VERTEX_PROJECT=my-gcp-project \
//!     ENTELIX_LIVE_VERTEX_LOCATION=global \
//!     ENTELIX_LIVE_VERTEX_GEMINI_MODEL=gemini-3.1-pro-preview \
//!     cargo test -p entelix-cloud --features gcp \
//!         --test live_vertex_gemini_streaming -- --ignored
//! ```
//!
//! ## What is verified
//!
//! 1. `ChatModel::stream_deltas` opens the SSE call without
//!    requiring callers to compose codec / transport directly —
//!    the same surface ontosyx's `ox-brain` will dispatch
//!    through.
//! 2. The delta stream produces at least one terminal
//!    [`StreamDelta::Stop`] carrying a recognised
//!    [`entelix_core::ir::StopReason`] and at least one
//!    progress-bearing delta (`TextDelta` *or* `ThinkingDelta` —
//!    Gemini reasoning models often emit thinking deltas before
//!    the visible reply).
//! 3. The tapped aggregator delivered through
//!    `ModelStream.completion` reconstructs a non-empty
//!    [`entelix_core::ir::ModelResponse`] with populated
//!    `Usage` counters — i.e. the streaming dispatch path emits
//!    the same `Usage` shape the non-streaming path does, so the
//!    cost meter (invariant 12) sees identical accounting whether
//!    callers stream or one-shot.
//! 4. SSE end-of-stream cleanly terminates the consumer loop —
//!    no hang on the polling future after the terminal `Stop`.
//!
//! ## Cost discipline
//!
//! `max_tokens = 256` against the Gemini Flash / Pro tier.
//! Per-run cost well under $0.001. Reasoning models budget part
//! of `max_tokens` on the thinking channel before producing
//! visible text, so the budget covers both passes.

#![cfg(feature = "gcp")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use entelix_cloud::vertex::{VertexCredentialProvider, VertexTransport};
use entelix_core::ChatModel;
use entelix_core::codecs::VertexGeminiCodec;
use entelix_core::context::ExecutionContext;
use entelix_core::install_default_tls;
use entelix_core::ir::{Message, StopReason};
use entelix_core::stream::StreamDelta;
use futures::StreamExt;

const DEFAULT_MODEL: &str = "gemini-3.1-pro-preview";

#[tokio::test]
#[ignore = "live-API: requires GCP ADC + ENTELIX_LIVE_VERTEX_PROJECT + ENTELIX_LIVE_VERTEX_LOCATION"]
async fn vertex_gemini_streaming_round_trip() {
    install_default_tls();

    let project = std::env::var("ENTELIX_LIVE_VERTEX_PROJECT").expect(
        "set ENTELIX_LIVE_VERTEX_PROJECT (GCP project hosting the Vertex Gemini model) to run this live smoke",
    );
    let location = std::env::var("ENTELIX_LIVE_VERTEX_LOCATION")
        .expect("set ENTELIX_LIVE_VERTEX_LOCATION (e.g. `global`, `asia-northeast3`)");
    let model = std::env::var("ENTELIX_LIVE_VERTEX_GEMINI_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let credentials = VertexCredentialProvider::default_chain()
        .await
        .expect("ADC must resolve — run `gcloud auth application-default login` first");

    let mut builder = VertexTransport::builder()
        .with_project_id(&project)
        .with_location(&location)
        .with_token_refresher(Arc::new(credentials));
    if let Ok(qp) = std::env::var("ENTELIX_LIVE_VERTEX_QUOTA_PROJECT") {
        builder = builder.with_quota_project(qp);
    }
    let transport = builder
        .build()
        .expect("VertexTransport built from ADC chain");

    let chat = ChatModel::new(VertexGeminiCodec::new(), transport, model)
        .with_max_tokens(256)
        .with_temperature(0.0);

    let ctx = ExecutionContext::new();
    let stream_envelope = chat
        .stream_deltas(
            vec![Message::user("Reply with the word 'ok' and nothing else.")],
            &ctx,
        )
        .await
        .expect("ChatModel::stream_deltas must open the SSE call");

    let mut stream = stream_envelope.stream;
    let completion = stream_envelope.completion;

    let mut text_chunks: Vec<String> = Vec::new();
    let mut thinking_chunks: Vec<String> = Vec::new();
    let mut saw_stop = false;
    let mut terminal_stop_reason: Option<StopReason> = None;

    while let Some(delta) = stream.next().await {
        let delta = delta.expect("stream delta must not error mid-flight");
        match delta {
            StreamDelta::TextDelta { text, .. } => {
                if !text.is_empty() {
                    text_chunks.push(text);
                }
            }
            StreamDelta::ThinkingDelta { text, .. } => {
                if !text.is_empty() {
                    thinking_chunks.push(text);
                }
            }
            StreamDelta::Stop { stop_reason, .. } => {
                saw_stop = true;
                terminal_stop_reason = Some(stop_reason);
            }
            // Other variants (Start / ToolUse* / Usage / RateLimit /
            // Warning) are observed silently — this smoke focuses on
            // the visible-progress + terminal-stop contract.
            _ => {}
        }
    }

    assert!(saw_stop, "stream must emit a terminal `Stop` delta");
    let stop = terminal_stop_reason.expect("terminal Stop must carry a StopReason");
    assert!(
        matches!(
            stop,
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence { .. }
        ),
        "unexpected terminal stop_reason: {stop:?}"
    );
    assert!(
        !text_chunks.is_empty() || !thinking_chunks.is_empty(),
        "stream must produce at least one progress-bearing delta (text or thinking) before Stop"
    );

    let final_response = completion
        .await
        .expect("ModelStream.completion must resolve to the aggregated ModelResponse");

    assert!(
        !final_response.content.is_empty(),
        "aggregated ModelResponse must carry at least one content part"
    );
    assert!(
        matches!(
            final_response.stop_reason,
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence { .. }
        ),
        "aggregated stop_reason must match the terminal Stop: {:?}",
        final_response.stop_reason
    );
    assert!(
        final_response.usage.input_tokens > 0,
        "streaming dispatch must populate usage.input_tokens (cost meter dependency, invariant 12)"
    );
    assert!(
        final_response.usage.output_tokens > 0,
        "streaming dispatch must populate usage.output_tokens"
    );
}
