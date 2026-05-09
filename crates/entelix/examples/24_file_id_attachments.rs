//! `24_file_id_attachments` — large attachments via
//! `MediaSource::FileId` reference instead of inline base64.
//!
//! Build: `cargo build --example 24_file_id_attachments -p entelix`
//! Run (hermetic — composes the IR shape; no network):
//!     `cargo run --example 24_file_id_attachments -p entelix`
//!
//! For attachments larger than a few MB, base64-inlining the bytes
//! into every request is wasteful — vendors expose Files / cached-
//! content APIs that mint a stable id once, then accept that id on
//! subsequent calls.
//!
//! `MediaSource` carries three sources for every modality (image,
//! audio, video, document):
//!
//! - `Base64 { media_type, data }`     — inline; small attachments.
//! - `Url { url, media_type? }`        — vendor fetches by URL.
//! - `FileId { id, media_type? }`      — vendor-minted reference.
//!
//! Per-vendor mapping for `FileId`:
//!
//! | Vendor          | Upload API                | Reference shape                   |
//! |-----------------|---------------------------|-----------------------------------|
//! | `Anthropic`     | Files API (`/v1/files`)   | `file_id` on `image` / `document` |
//! | `OpenAI`        | Files API (`/v1/files`)   | `file_id` on `image_file`         |
//! | `Gemini`        | `cachedContents` resource | `cached_content` on the request   |
//!
//! In production the upload step uses the vendor's REST API or SDK
//! and stamps the returned id onto the IR `ContentPart`. This
//! example shows the IR shape that flows through the codec stack
//! once the upload completes.

#![allow(clippy::print_stdout)]

use entelix::ir::{ContentPart, MediaSource, Message, Role};

#[tokio::main]
async fn main() -> entelix::Result<()> {
    // (1) Operator's pre-upload step (sketched as a constant).
    //     Anthropic Files API would return e.g. `file_xyz123`.
    //     The id is opaque to the SDK — vendor metadata only.
    let anthropic_pdf = "file_xyz123";
    let openai_image = "file-abc456";
    let gemini_cached_content = "cachedContents/proj-123/abc";

    // (2) Build a multimodal user turn referencing the uploads by
    //     id. `MediaSource::file_id(.)` wraps the vendor token; the
    //     codec routes the wire encoding (Anthropic emits
    //     `{type:"document", source:{type:"file", file_id:"…"}}`,
    //     OpenAI Responses emits `{type:"image_file", file_id:"…"}`
    //     in the canonical content_part shape).
    let user_turn = Message::new(
        Role::User,
        vec![
            ContentPart::text("Summarise the attached deck and the screenshot together."),
            ContentPart::document(MediaSource::file_id(anthropic_pdf), Some("deck.pdf".into())),
            ContentPart::image(MediaSource::file_id(openai_image)),
        ],
    );

    println!("=== user turn IR ===");
    for (i, part) in user_turn.content.iter().enumerate() {
        match part {
            ContentPart::Text { text, .. } => println!("[{i}] text: {text}"),
            ContentPart::Document { source, .. } => match source {
                MediaSource::FileId { id, .. } => println!("[{i}] document: file_id={id}"),
                other => println!("[{i}] document: {other:?}"),
            },
            ContentPart::Image { source, .. } => match source {
                MediaSource::FileId { id, .. } => println!("[{i}] image: file_id={id}"),
                other => println!("[{i}] image: {other:?}"),
            },
            other => println!("[{i}] {other:?}"),
        }
    }

    // (3) Gemini-style cached-content reference rides on the
    //     `ModelRequest::cached_content` field rather than per-part
    //     — the vendor caches an entire conversational prefix
    //     (system prompt + early turns), keyed by the operator's
    //     prior `cachedContents.create` call. The codec emits the
    //     id verbatim; non-Gemini codecs surface
    //     `ModelWarning::LossyEncode`.
    println!("\n=== Gemini cached_content (request-level) ===");
    println!("ModelRequest.cached_content = Some({gemini_cached_content:?})");
    println!("(non-Gemini codecs emit ModelWarning::LossyEncode for this field)");

    Ok(())
}
