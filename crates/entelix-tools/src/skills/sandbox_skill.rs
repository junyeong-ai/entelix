//! `SandboxSkill` — `Skill` implementation backed by a sandbox-internal
//! directory tree.
//!
//! Layout:
//!
//! - `<root>/SKILL.md` — required. YAML frontmatter delimited by `---`
//!   carries the skill metadata (`name`, `description`, optional
//!   `version`); the markdown body becomes the activated instructions.
//! - `<root>/<any other relative file>` — exposed as a `SkillResource`
//!   keyed by path relative to `<root>`.
//!
//! All filesystem access flows through `Arc<dyn Sandbox>` — invariant 9
//! preserved (no `std::fs` import here).

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use entelix_core::context::ExecutionContext;
use entelix_core::error::{Error, Result};
use entelix_core::sandbox::{DirEntry, Sandbox};
use entelix_core::skills::{LoadedSkill, Skill, SkillResource, SkillResourceContent};

use crate::skills::manifest::{SkillManifest, parse_skill_md};

/// Conventional name of the manifest file under each skill root.
const SKILL_MANIFEST: &str = "SKILL.md";

/// `Skill` whose name, description, and instructions come from a
/// `SKILL.md` markdown file on a sandbox-internal filesystem.
///
/// The constructor only records *where* the skill lives. Reading the
/// manifest happens at [`Skill::load`] / `register_from_sandbox` time
/// so a misconfigured skill root reports the error at the time it
/// matters — close to the wire boundary, not at registration.
#[derive(Clone)]
pub struct SandboxSkill {
    name: String,
    description: String,
    version: Option<String>,
    sandbox: Arc<dyn Sandbox>,
    /// Sandbox-internal directory holding `SKILL.md` and any
    /// resources.
    root: String,
}

impl std::fmt::Debug for SandboxSkill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Skip `description` and `version` — they are derivable from
        // the sandbox-stored manifest and would inflate every debug
        // print of an Agent's skill registry.
        f.debug_struct("SandboxSkill")
            .field("name", &self.name)
            .field("backend", &self.sandbox.backend())
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl SandboxSkill {
    /// Build by reading `<root>/SKILL.md` from `sandbox` and parsing
    /// its frontmatter for name + description + optional version.
    /// The instructions body and resource enumeration are deferred
    /// to [`Skill::load`] — the registration path stays cheap.
    pub async fn from_sandbox(
        sandbox: Arc<dyn Sandbox>,
        root: impl Into<String>,
        ctx: &ExecutionContext,
    ) -> Result<Self> {
        let root = root.into();
        let manifest_path = join_sandbox_path(&root, SKILL_MANIFEST);
        let bytes = sandbox.read_file(&manifest_path, ctx).await?;
        let text = String::from_utf8(bytes).map_err(|e| {
            Error::config(format!(
                "SandboxSkill: SKILL.md at {manifest_path:?} is not valid UTF-8: {e}"
            ))
        })?;
        let SkillManifest {
            name,
            description,
            version,
            body: _,
        } = parse_skill_md(&text).map_err(|e| {
            Error::config(format!(
                "SandboxSkill: SKILL.md at {manifest_path:?} is malformed: {e}"
            ))
        })?;
        Ok(Self {
            name,
            description,
            version,
            sandbox,
            root,
        })
    }
}

#[async_trait]
impl Skill for SandboxSkill {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    async fn load(&self, ctx: &ExecutionContext) -> Result<LoadedSkill> {
        // Re-read the manifest at activation time so edits to
        // SKILL.md reflect without a process restart. The backing
        // sandbox is the source of truth.
        let manifest_path = join_sandbox_path(&self.root, SKILL_MANIFEST);
        let manifest_bytes = self.sandbox.read_file(&manifest_path, ctx).await?;
        let manifest_text = String::from_utf8(manifest_bytes).map_err(|e| {
            Error::config(format!(
                "SandboxSkill {:?}: SKILL.md is not valid UTF-8: {}",
                self.name, e
            ))
        })?;
        let SkillManifest { body, .. } = parse_skill_md(&manifest_text).map_err(|e| {
            Error::config(format!(
                "SandboxSkill {:?}: SKILL.md is malformed: {}",
                self.name, e
            ))
        })?;

        // Walk the root directory and register every other file as a
        // lazy resource handle.
        let mut resources: BTreeMap<String, Arc<dyn SkillResource>> = BTreeMap::new();
        walk_collect_resources(self.sandbox.clone(), ctx, &self.root, "", &mut resources).await?;

        Ok(LoadedSkill {
            instructions: body,
            resources,
        })
    }
}

/// Lazy-read resource backed by a sandbox-internal file.
#[derive(Clone)]
pub struct SandboxResource {
    sandbox: Arc<dyn Sandbox>,
    path: String,
    /// MIME type guessed from the file extension at registration time.
    /// Not authoritative — operators that need a precise content-type
    /// implement `SkillResource` themselves.
    mime_type: String,
}

impl std::fmt::Debug for SandboxResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxResource")
            .field("backend", &self.sandbox.backend())
            .field("path", &self.path)
            .field("mime_type", &self.mime_type)
            .finish()
    }
}

#[async_trait]
impl SkillResource for SandboxResource {
    async fn read(&self, ctx: &ExecutionContext) -> Result<SkillResourceContent> {
        let bytes = self.sandbox.read_file(&self.path, ctx).await?;
        if mime_is_text(&self.mime_type) {
            match String::from_utf8(bytes) {
                Ok(text) => Ok(SkillResourceContent::Text(text)),
                Err(e) => Ok(SkillResourceContent::Binary {
                    mime_type: self.mime_type.clone(),
                    bytes: e.into_bytes(),
                }),
            }
        } else {
            Ok(SkillResourceContent::Binary {
                mime_type: self.mime_type.clone(),
                bytes,
            })
        }
    }
}

// ── walk helpers ───────────────────────────────────────────────────────────

fn walk_collect_resources<'a>(
    sandbox: Arc<dyn Sandbox>,
    ctx: &'a ExecutionContext,
    root: &'a str,
    rel: &'a str,
    out: &'a mut BTreeMap<String, Arc<dyn SkillResource>>,
) -> futures::future::BoxFuture<'a, Result<()>> {
    Box::pin(async move {
        let dir_path = if rel.is_empty() {
            root.to_owned()
        } else {
            join_sandbox_path(root, rel)
        };
        let entries = sandbox.list_dir(&dir_path, ctx).await?;
        for entry in entries {
            let entry_rel = if rel.is_empty() {
                entry.name.clone()
            } else {
                format!("{rel}/{}", entry.name)
            };
            handle_entry(&sandbox, ctx, root, rel, &entry, &entry_rel, out).await?;
        }
        Ok(())
    })
}

async fn handle_entry(
    sandbox: &Arc<dyn Sandbox>,
    ctx: &ExecutionContext,
    root: &str,
    rel: &str,
    entry: &DirEntry,
    entry_rel: &str,
    out: &mut BTreeMap<String, Arc<dyn SkillResource>>,
) -> Result<()> {
    if entry.is_dir {
        walk_collect_resources(Arc::clone(sandbox), ctx, root, entry_rel, out).await?;
        return Ok(());
    }
    if rel.is_empty() && entry.name == SKILL_MANIFEST {
        // The manifest is the instructions body, not a resource.
        return Ok(());
    }
    let abs_path = join_sandbox_path(root, entry_rel);
    let mime = guess_mime_from_extension(&entry.name);
    out.insert(
        entry_rel.to_owned(),
        Arc::new(SandboxResource {
            sandbox: Arc::clone(sandbox),
            path: abs_path,
            mime_type: mime,
        }) as Arc<dyn SkillResource>,
    );
    Ok(())
}

fn join_sandbox_path(left: &str, right: &str) -> String {
    let l = left.trim_end_matches('/');
    let r = right.trim_start_matches('/');
    if l.is_empty() {
        r.to_owned()
    } else if r.is_empty() {
        l.to_owned()
    } else {
        format!("{l}/{r}")
    }
}

fn guess_mime_from_extension(name: &str) -> String {
    let ext = name
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase());
    match ext.as_deref() {
        Some("md" | "markdown") => "text/markdown",
        Some("txt") => "text/plain",
        Some("json") => "application/json",
        Some("yaml" | "yml") => "application/yaml",
        Some("toml") => "application/toml",
        Some("csv") => "text/csv",
        Some("tsv") => "text/tab-separated-values",
        Some("xml") => "application/xml",
        Some("html" | "htm") => "text/html",
        Some("js" | "mjs") => "text/javascript",
        Some("ts" | "tsx") => "application/typescript",
        Some("py") => "text/x-python",
        Some("rs") => "text/x-rust",
        Some("go") => "text/x-go",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("pdf") => "application/pdf",
        _ => "application/octet-stream",
    }
    .to_owned()
}

fn mime_is_text(mime: &str) -> bool {
    if mime.starts_with("text/") {
        return true;
    }
    matches!(
        mime,
        "application/json"
            | "application/yaml"
            | "application/toml"
            | "application/xml"
            | "application/typescript"
            | "application/javascript"
    )
}
