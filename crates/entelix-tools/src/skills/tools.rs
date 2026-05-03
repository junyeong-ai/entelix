//! `ListSkillsTool`, `ActivateSkillTool`, `ReadSkillResourceTool` —
//! the three LLM-facing tools that drive progressive disclosure
//! (ADR-0027 §"Three built-in tools").

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::skills::{SkillRegistry, SkillResourceContent};
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};

use crate::error::ToolError;

/// Tool name surfaced to the LLM for the listing tool.
const LIST_TOOL_NAME: &str = "list_skills";
/// Tool name surfaced to the LLM for the activation tool.
const ACTIVATE_TOOL_NAME: &str = "activate_skill";
/// Tool name surfaced to the LLM for the resource-read tool.
const READ_RESOURCE_TOOL_NAME: &str = "read_skill_resource";

// ── list_skills ────────────────────────────────────────────────────────────

/// T1 listing — returns `[{name, description, version}, ...]` for
/// every skill registered in the bound `SkillRegistry`.
#[derive(Clone, Debug)]
pub struct ListSkillsTool {
    registry: SkillRegistry,
    metadata: ToolMetadata,
}

impl ListSkillsTool {
    /// Build with the given registry. Cloning is cheap (`Arc`-backed).
    #[must_use]
    pub fn new(registry: SkillRegistry) -> Self {
        Self {
            registry,
            metadata: ToolMetadata::function(
                LIST_TOOL_NAME,
                "List available skills with their names and descriptions. Use this to \
                 discover what skills exist before activating one.",
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            )
            .with_effect(ToolEffect::ReadOnly)
            .with_idempotent(true),
        }
    }
}

#[async_trait]
impl Tool for ListSkillsTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, _input: Value, _ctx: &ExecutionContext) -> Result<Value> {
        let summaries = self.registry.summaries();
        let entries: Vec<Value> = summaries
            .iter()
            .map(|s| {
                let mut obj = serde_json::Map::new();
                obj.insert("name".into(), Value::String(s.name.to_owned()));
                obj.insert(
                    "description".into(),
                    Value::String(s.description.to_owned()),
                );
                if let Some(v) = s.version {
                    obj.insert("version".into(), Value::String(v.to_owned()));
                }
                Value::Object(obj)
            })
            .collect();
        Ok(json!({ "skills": entries }))
    }
}

// ── activate_skill ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ActivateInput {
    name: String,
}

/// T1 → T2 transition — load one skill's full instructions and the
/// list of resource keys it exposes. Resource bytes are *not*
/// included; they are read separately via [`ReadSkillResourceTool`].
#[derive(Clone, Debug)]
pub struct ActivateSkillTool {
    registry: SkillRegistry,
    metadata: ToolMetadata,
}

impl ActivateSkillTool {
    /// Build with the given registry.
    #[must_use]
    pub fn new(registry: SkillRegistry) -> Self {
        Self {
            registry,
            metadata: ToolMetadata::function(
                ACTIVATE_TOOL_NAME,
                "Activate a skill by name. Returns its instructions and the list of \
                 available resource keys. Read individual resources with \
                 read_skill_resource.",
                json!({
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Skill name from list_skills."
                        }
                    },
                    "additionalProperties": false
                }),
            )
            .with_effect(ToolEffect::ReadOnly)
            .with_idempotent(true),
        }
    }
}

#[async_trait]
impl Tool for ActivateSkillTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &ExecutionContext) -> Result<Value> {
        let parsed: ActivateInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let skill = self.registry.get(&parsed.name).ok_or_else(|| {
            Error::config(format!(
                "activate_skill: skill {:?} is not registered",
                parsed.name
            ))
        })?;
        let loaded = skill.load(ctx).await?;
        let keys: Vec<Value> = loaded
            .resource_keys()
            .into_iter()
            .map(|k| Value::String(k.to_owned()))
            .collect();
        Ok(json!({
            "instructions": loaded.instructions,
            "resources": keys,
        }))
    }
}

// ── read_skill_resource ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ReadResourceInput {
    skill: String,
    key: String,
}

/// T2 → T3 transition — read one resource from one skill.
///
/// Text resources return their body; binary resources return
/// metadata only (`mime_type`, `size_bytes`, `sha256`). The host
/// application can fetch the bytes via [`SkillRegistry`] for
/// out-of-band use (e.g., uploading to a vendor file API and
/// handing the resulting `FileId` back to the model).
#[derive(Clone, Debug)]
pub struct ReadSkillResourceTool {
    registry: SkillRegistry,
    metadata: ToolMetadata,
}

impl ReadSkillResourceTool {
    /// Build with the given registry.
    #[must_use]
    pub fn new(registry: SkillRegistry) -> Self {
        Self {
            registry,
            metadata: ToolMetadata::function(
                READ_RESOURCE_TOOL_NAME,
                "Read a resource from an activated skill. Text resources return their \
                 contents; binary resources return only metadata (mime_type, size_bytes, \
                 sha256).",
                json!({
                    "type": "object",
                    "required": ["skill", "key"],
                    "properties": {
                        "skill": {
                            "type": "string",
                            "description": "Skill name (from list_skills / activate_skill)."
                        },
                        "key": {
                            "type": "string",
                            "description": "Resource key (from activate_skill response)."
                        }
                    },
                    "additionalProperties": false
                }),
            )
            .with_effect(ToolEffect::ReadOnly)
            .with_idempotent(true),
        }
    }
}

#[async_trait]
impl Tool for ReadSkillResourceTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, ctx: &ExecutionContext) -> Result<Value> {
        let parsed: ReadResourceInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let skill = self.registry.get(&parsed.skill).ok_or_else(|| {
            Error::config(format!(
                "read_skill_resource: skill {:?} is not registered",
                parsed.skill
            ))
        })?;
        let loaded = skill.load(ctx).await?;
        let resource = loaded.resources.get(&parsed.key).ok_or_else(|| {
            Error::config(format!(
                "read_skill_resource: skill {:?} has no resource {:?}",
                parsed.skill, parsed.key
            ))
        })?;
        let content = resource.read(ctx).await?;
        match content {
            SkillResourceContent::Text(text) => Ok(json!({ "text": text })),
            SkillResourceContent::Binary { mime_type, bytes } => {
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                let digest = hasher.finalize();
                let sha256 = hex_lowercase(&digest);
                Ok(json!({
                    "mime_type": mime_type,
                    "size_bytes": bytes.len(),
                    "sha256": sha256,
                }))
            }
            other => Err(Error::config(format!(
                "read_skill_resource: unsupported resource shape {other:?}"
            ))),
        }
    }
}

/// Lowercase hex encoder for the SHA-256 placeholder display path.
fn hex_lowercase(bytes: &[u8]) -> String {
    fn nibble(n: u8) -> char {
        // Each nibble is in 0..16 by construction of `>> 4` and `& 0x0f`.
        match n {
            0..=9 => (b'0' + n) as char,
            10..=15 => (b'a' + n - 10) as char,
            _ => unreachable!(),
        }
    }
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(nibble(b >> 4));
        out.push(nibble(b & 0x0f));
    }
    out
}
