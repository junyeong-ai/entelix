//! `SandboxSkill` round-trip via `MockSandbox`.
//!
//! Verifies that a directory tree of the form
//!
//! ```text
//! /skills/<name>/
//!   SKILL.md           ← YAML frontmatter + instructions body
//!   reference/api.md
//!   examples/quickstart.md
//!   icon.png
//! ```
//!
//! parses into a `SandboxSkill` whose name + description + version
//! come from the manifest, whose `LoadedSkill::instructions` is the
//! body that follows the frontmatter, and whose other files are
//! enumerated as lazy resources.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use entelix_core::context::ExecutionContext;
use entelix_core::sandbox::Sandbox;
use entelix_core::skills::{Skill, SkillResourceContent};
use entelix_tools::SandboxSkill;
use entelix_tools::sandboxed::MockSandbox;

const SKILL_MD: &str = "---
name: code-review
description: Review pull requests with focus on correctness and security.
version: 0.1.0
---
Review the supplied PR step by step:

1. Read the diff.
2. Identify behaviour changes.
3. Suggest tests for any new branch.
";

fn build_sandbox() -> Arc<dyn Sandbox> {
    let sandbox = MockSandbox::new();
    sandbox.seed_file("/skills/code-review/SKILL.md", SKILL_MD.as_bytes().to_vec());
    sandbox.seed_file(
        "/skills/code-review/reference/api.md",
        b"## review(diff) -> comments".to_vec(),
    );
    sandbox.seed_file(
        "/skills/code-review/examples/quickstart.md",
        b"User: review my PR\nAssistant: ...".to_vec(),
    );
    sandbox.seed_file(
        "/skills/code-review/icon.png",
        vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    );
    Arc::new(sandbox)
}

#[tokio::test]
async fn from_sandbox_parses_manifest_metadata() {
    let sandbox = build_sandbox();
    let ctx = ExecutionContext::new();
    let skill = SandboxSkill::from_sandbox(sandbox, "/skills/code-review", &ctx)
        .await
        .unwrap();
    assert_eq!(skill.name(), "code-review");
    assert_eq!(
        skill.description(),
        "Review pull requests with focus on correctness and security."
    );
    assert_eq!(skill.version(), Some("0.1.0"));
}

#[tokio::test]
async fn loaded_instructions_match_markdown_body() {
    let sandbox = build_sandbox();
    let ctx = ExecutionContext::new();
    let skill = SandboxSkill::from_sandbox(sandbox, "/skills/code-review", &ctx)
        .await
        .unwrap();
    let loaded = skill.load(&ctx).await.unwrap();
    assert!(loaded.instructions.starts_with("Review the supplied PR"));
    // Frontmatter must NOT leak into the body.
    assert!(!loaded.instructions.contains("description:"));
    assert!(!loaded.instructions.contains("version:"));
    // The manifest is the instructions, not a resource.
    assert!(!loaded.resources.contains_key("SKILL.md"));
}

#[tokio::test]
async fn enumerates_other_files_as_resources() {
    let sandbox = build_sandbox();
    let ctx = ExecutionContext::new();
    let skill = SandboxSkill::from_sandbox(sandbox, "/skills/code-review", &ctx)
        .await
        .unwrap();
    let loaded = skill.load(&ctx).await.unwrap();
    let mut keys = loaded.resource_keys();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["examples/quickstart.md", "icon.png", "reference/api.md"]
    );
}

#[tokio::test]
async fn text_resource_returns_text_content() {
    let sandbox = build_sandbox();
    let ctx = ExecutionContext::new();
    let skill = SandboxSkill::from_sandbox(sandbox, "/skills/code-review", &ctx)
        .await
        .unwrap();
    let loaded = skill.load(&ctx).await.unwrap();
    let res = loaded.resources.get("reference/api.md").unwrap();
    let content = res.read(&ctx).await.unwrap();
    assert!(matches!(
        content,
        SkillResourceContent::Text(ref t) if t.contains("review(diff)")
    ));
}

#[tokio::test]
async fn binary_resource_returns_binary_content() {
    let sandbox = build_sandbox();
    let ctx = ExecutionContext::new();
    let skill = SandboxSkill::from_sandbox(sandbox, "/skills/code-review", &ctx)
        .await
        .unwrap();
    let loaded = skill.load(&ctx).await.unwrap();
    let res = loaded.resources.get("icon.png").unwrap();
    let content = res.read(&ctx).await.unwrap();
    match content {
        SkillResourceContent::Binary { mime_type, bytes } => {
            assert_eq!(mime_type, "image/png");
            assert_eq!(bytes.len(), 8);
        }
        other => panic!("expected binary content, got {other:?}"),
    }
}

#[tokio::test]
async fn malformed_manifest_surfaces_a_config_error() {
    let sandbox = MockSandbox::new();
    sandbox.seed_file(
        "/skills/broken/SKILL.md",
        // Missing closing fence.
        b"---\nname: broken\ndescription: x\n".to_vec(),
    );
    let err = SandboxSkill::from_sandbox(
        Arc::new(sandbox),
        "/skills/broken",
        &ExecutionContext::new(),
    )
    .await
    .unwrap_err();
    assert!(
        format!("{err}").contains("malformed"),
        "expected malformed-manifest message, got {err}"
    );
}
