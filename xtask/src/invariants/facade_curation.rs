//! Facade top-level curation gate.
//!
//! `facade-completeness` guarantees every public sub-crate item remains
//! reachable through the facade. This gate enforces the complementary rule:
//! backend, adapter, and durable session surfaces must stay under named
//! facade modules instead of drifting back into `entelix::Type`.

use std::collections::BTreeSet;

use anyhow::Result;

use crate::visitor::{Violation, parse, repo_root, report, span_loc};

const BANNED_TOP_LEVEL_CRATES: &[&str] = &[
    "entelix_cloud",
    "entelix_graphmemory_pg",
    "entelix_mcp",
    "entelix_mcp_chatmodel",
    "entelix_memory",
    "entelix_memory_openai",
    "entelix_memory_pgvector",
    "entelix_memory_qdrant",
    "entelix_otel",
    "entelix_persistence",
    "entelix_policy",
    "entelix_server",
    "entelix_session",
    "entelix_tools",
];

const BANNED_TOP_LEVEL_MODULES: &[&str] = &[
    "builtin_tools",
    "cloud",
    "embedders",
    "graph_memory",
    "mcp",
    "mcp_chatmodel",
    "memory",
    "otel",
    "persistence",
    "policy",
    "server",
    "session",
    "vectorstores",
];

const TOP_LEVEL_AGENT_ALLOWLIST: &[&str] = &[
    "Agent",
    "AgentBuilder",
    "AgentRunEvent",
    "AgentRunResult",
    "ChatAgentBuilder",
    "ChatState",
    "ReActAgentBuilder",
    "ReActState",
    "Subagent",
    "SubagentBuilder",
    "SubagentTool",
    "SupervisorState",
    "build_chat_graph",
    "build_react_graph",
    "build_supervisor_graph",
    "create_chat_agent",
    "create_react_agent",
    "create_supervisor_agent",
];

const TOP_LEVEL_CORE_ALLOWLIST: &[&str] = &[
    "AgentContext",
    "ApprovalDecision",
    "AuditSink",
    "AuditSinkHandle",
    "ChatModel",
    "ChatModelConfig",
    "CostCalculator",
    "DEFAULT_TENANT_ID",
    "Error",
    "ExecutionContext",
    "Extensions",
    "INTERRUPT_KIND_APPROVAL_PENDING",
    "LlmFacingSchema",
    "LlmRenderable",
    "ModelCatalog",
    "ModelEndpoint",
    "ModelToolCatalog",
    "OutputValidator",
    "PendingApprovalDecisions",
    "ProviderErrorKind",
    "RenderedForLlm",
    "Result",
    "RunBudget",
    "RunOverrides",
    "TenantId",
    "ThreadKey",
    "ToolCostCalculator",
    "ToolProgress",
    "ToolProgressSink",
    "ToolProgressSinkHandle",
    "ToolProgressStatus",
    "Toolset",
    "TypedModelStream",
    "TypedOutputOptions",
    "UsageLimitAxis",
    "UsageSnapshot",
    "auth",
    "cancellation",
    "codecs",
    "ir",
    "model_catalog",
    "sandbox",
    "service",
    "skills",
    "stream",
    "tools",
    "transports",
];

const TOP_LEVEL_GRAPH_ALLOWLIST: &[&str] = &[
    "Annotated",
    "Append",
    "Checkpoint",
    "CheckpointGranularity",
    "CheckpointId",
    "Checkpointer",
    "Command",
    "CompiledGraph",
    "ConditionalEdge",
    "ContributingNodeAdapter",
    "DEFAULT_RECURSION_LIMIT",
    "END",
    "EdgeSelector",
    "InMemoryCheckpointer",
    "Max",
    "MergeMap",
    "MergeNodeAdapter",
    "Reducer",
    "Replace",
    "SendEdge",
    "SendMerger",
    "SendSelector",
    "StateGraph",
    "StateMerge",
    "interrupt",
];

const TOP_LEVEL_PROMPT_ALLOWLIST: &[&str] = &[
    "ChatFewShotPromptTemplate",
    "ChatPromptPart",
    "ChatPromptTemplate",
    "Example",
    "ExampleSelector",
    "FewShotPromptTemplate",
    "FixedExampleSelector",
    "LengthBasedExampleSelector",
    "MessagesPlaceholder",
    "PromptTemplate",
    "PromptValue",
    "PromptVars",
    "SharedExampleSelector",
];

const TOP_LEVEL_RUNNABLE_ALLOWLIST: &[&str] = &[
    "AnyRunnable",
    "AnyRunnableHandle",
    "BoxStream",
    "Configured",
    "DebugEvent",
    "Fallback",
    "JsonOutputParser",
    "Mapping",
    "Retrying",
    "Runnable",
    "RunnableEvent",
    "RunnableExt",
    "RunnableLambda",
    "RunnableParallel",
    "RunnablePassthrough",
    "RunnableRouter",
    "RunnableSequence",
    "StreamChunk",
    "StreamMode",
    "Timed",
    "ToolToRunnableAdapter",
    "erase",
];

const PRELUDE_ALLOWLIST: &[&str] = &[
    "entelix_core::ir::ContentPart",
    "entelix_core::ir::Message",
    "entelix_core::ir::Role",
    "entelix_core::AgentContext",
    "entelix_core::ChatModel",
    "entelix_core::Error",
    "entelix_core::ExecutionContext",
    "entelix_core::Result",
    "entelix_prompt::ChatPromptPart",
    "entelix_prompt::ChatPromptTemplate",
    "entelix_prompt::PromptValue",
    "entelix_prompt::PromptVars",
    "entelix_runnable::JsonOutputParser",
    "entelix_runnable::Runnable",
    "entelix_runnable::RunnableExt",
];

pub(crate) fn run() -> Result<()> {
    let root = repo_root()?;
    let facade = root.join("crates/entelix/src/lib.rs");
    let (_, ast) = parse(&facade)?;
    let allowed_agents: BTreeSet<&str> = TOP_LEVEL_AGENT_ALLOWLIST.iter().copied().collect();
    let allowed_core: BTreeSet<&str> = TOP_LEVEL_CORE_ALLOWLIST.iter().copied().collect();
    let allowed_graph: BTreeSet<&str> = TOP_LEVEL_GRAPH_ALLOWLIST.iter().copied().collect();
    let allowed_prompt: BTreeSet<&str> = TOP_LEVEL_PROMPT_ALLOWLIST.iter().copied().collect();
    let allowed_runnable: BTreeSet<&str> = TOP_LEVEL_RUNNABLE_ALLOWLIST.iter().copied().collect();
    let banned_crates: BTreeSet<&str> = BANNED_TOP_LEVEL_CRATES.iter().copied().collect();
    let banned_modules: BTreeSet<&str> = BANNED_TOP_LEVEL_MODULES.iter().copied().collect();
    let prelude_allowed: BTreeSet<&str> = PRELUDE_ALLOWLIST.iter().copied().collect();
    let mut violations = Vec::new();

    for item in ast.items {
        match item {
            syn::Item::Use(item_use) => {
                if !matches!(item_use.vis, syn::Visibility::Public(_)) {
                    continue;
                }
                let (line, col) = span_loc(item_use.use_token.span);
                let mut exports = Vec::new();
                collect_use_exports(&item_use.tree, &mut Vec::new(), &mut exports);
                for export in exports {
                    let path = &export.source;
                    let Some(root_ident) = path.first() else {
                        continue;
                    };
                    if banned_crates.contains(root_ident.as_str()) {
                        violations.push(Violation::new(
                            facade.clone(),
                            line,
                            col,
                            format!(
                                "`pub use {root_ident}::...` exposes backend/adapter/session \
                                 types at the facade root"
                            ),
                        ));
                        continue;
                    }
                    if is_banned_facade_module_reexport(&path, &banned_modules) {
                        violations.push(Violation::new(
                            facade.clone(),
                            line,
                            col,
                            "`pub use <named_module>::...` re-exports advanced facade module \
                             items at the facade root",
                        ));
                        continue;
                    }

                    if is_agent_reexport(&path) {
                        let name = export.exposed.as_str();
                        if !allowed_agents.contains(name) {
                            violations.push(Violation::new(
                                facade.clone(),
                                line,
                                col,
                                format!(
                                    "`{}` is not part of the curated top-level agent recipe \
                                     surface",
                                    path.join("::")
                                ),
                            ));
                        }
                        continue;
                    }

                    if let Some((surface, allowed)) = curated_surface_allowlist(
                        &path,
                        &allowed_core,
                        &allowed_graph,
                        &allowed_prompt,
                        &allowed_runnable,
                    ) {
                        let name = export.exposed.as_str();
                        if !allowed.contains(name) {
                            violations.push(Violation::new(
                                facade.clone(),
                                line,
                                col,
                                format!(
                                    "`{}` is not part of the curated top-level {surface} surface",
                                    path.join("::")
                                ),
                            ));
                        }
                    }
                }
            }
            syn::Item::Type(item_type) => {
                if !matches!(item_type.vis, syn::Visibility::Public(_)) {
                    continue;
                }
                let Some(path) = type_path_segments(&item_type.ty) else {
                    continue;
                };
                let (line, col) = span_loc(item_type.ident.span());
                if let Some(root) = path.first() {
                    if banned_crates.contains(root.as_str())
                        || is_banned_facade_module_reexport(&path, &banned_modules)
                    {
                        violations.push(Violation::new(
                            facade.clone(),
                            line,
                            col,
                            format!(
                                "`pub type {}` aliases advanced facade surface `{}` at the \
                                 facade root",
                                item_type.ident,
                                path.join("::")
                            ),
                        ));
                        continue;
                    }
                }
                if is_agent_reexport(&path) {
                    let source_name = path.last().map(String::as_str).unwrap_or("*");
                    let exposed_name = item_type.ident.to_string();
                    if !allowed_agents.contains(source_name)
                        || source_name != exposed_name.as_str()
                    {
                        violations.push(Violation::new(
                            facade.clone(),
                            line,
                            col,
                            format!(
                                "`pub type {}` aliases non-curated agent surface `{}` at the \
                                 facade root",
                                item_type.ident,
                                path.join("::")
                            ),
                        ));
                    }
                    continue;
                }
                if let Some((surface, allowed)) = curated_surface_allowlist(
                    &path,
                    &allowed_core,
                    &allowed_graph,
                    &allowed_prompt,
                    &allowed_runnable,
                ) {
                    let source_name = path.last().map(String::as_str).unwrap_or("*");
                    let exposed_name = item_type.ident.to_string();
                    if !allowed.contains(source_name)
                        || source_name != exposed_name.as_str()
                    {
                        violations.push(Violation::new(
                            facade.clone(),
                            line,
                            col,
                            format!(
                                "`pub type {}` aliases non-curated {surface} surface `{}` at the \
                                 facade root",
                                item_type.ident,
                                path.join("::")
                            ),
                        ));
                    }
                }
            }
            syn::Item::Struct(item_struct) => {
                if matches!(item_struct.vis, syn::Visibility::Public(_))
                    && banned_modules.contains(item_struct.ident.to_string().as_str())
                {
                    let (line, col) = span_loc(item_struct.ident.span());
                    violations.push(Violation::new(
                        facade.clone(),
                        line,
                        col,
                        format!(
                            "`pub struct {}` shadows a named advanced facade module at the \
                             facade root",
                            item_struct.ident
                        ),
                    ));
                }
            }
            syn::Item::Mod(item_mod) => {
                if matches!(item_mod.vis, syn::Visibility::Public(_))
                    && item_mod.ident == "prelude"
                {
                    check_prelude(
                        &facade,
                        &item_mod,
                        &prelude_allowed,
                        &banned_crates,
                        &banned_modules,
                        &mut violations,
                    );
                }
            }
            _ => {}
        }
    }

    report(
        "facade-curation",
        violations,
        "Keep the `entelix` root focused on the common SDK spine: core request\n\
         types, ChatModel, Runnable, prompt, StateGraph, and common agent\n\
         recipes. `entelix::prelude` is even smaller: message primitives,\n\
         ChatModel, ExecutionContext, Result, prompt primitives, Runnable,\n\
         RunnableExt, and JsonOutputParser. Backend, adapter, policy,\n\
         observability, session, memory, MCP, server, and built-in tool\n\
         surfaces belong under their named facade modules (`entelix::memory`,\n\
         `entelix::session`, `entelix::policy`, etc.).",
    )
}

fn check_prelude(
    facade: &std::path::Path,
    item_mod: &syn::ItemMod,
    prelude_allowed: &BTreeSet<&str>,
    banned_crates: &BTreeSet<&str>,
    banned_modules: &BTreeSet<&str>,
    violations: &mut Vec<Violation>,
) {
    let Some((_, items)) = &item_mod.content else {
        return;
    };
    for item in items {
        let syn::Item::Use(item_use) = item else {
            continue;
        };
        if !matches!(item_use.vis, syn::Visibility::Public(_)) {
            continue;
        }
        let (line, col) = span_loc(item_use.use_token.span);
        let mut exports = Vec::new();
        collect_use_exports(&item_use.tree, &mut Vec::new(), &mut exports);
        for export in exports {
            let joined = export.source.join("::");
            let exposed_matches_source = export
                .source
                .last()
                .map(|last| last == &export.exposed)
                .unwrap_or(false);
            if !prelude_allowed.contains(joined.as_str()) || !exposed_matches_source {
                violations.push(Violation::new(
                    facade.to_path_buf(),
                    line,
                    col,
                    format!("`prelude::{joined}` is not part of the curated prelude"),
                ));
            }
            let Some(root_ident) = export.source.first() else {
                continue;
            };
            if banned_crates.contains(root_ident.as_str())
                || is_banned_facade_module_reexport(&export.source, banned_modules)
            {
                violations.push(Violation::new(
                    facade.to_path_buf(),
                    line,
                    col,
                    format!("`prelude::{joined}` exposes advanced facade surface"),
                ));
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct UseExport {
    source: Vec<String>,
    exposed: String,
}

fn collect_use_exports(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<UseExport>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_exports(&path.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut path = prefix.clone();
            let exposed = name.ident.to_string();
            path.push(exposed.clone());
            out.push(UseExport {
                source: path,
                exposed,
            });
        }
        syn::UseTree::Rename(rename) => {
            let mut path = prefix.clone();
            path.push(rename.ident.to_string());
            out.push(UseExport {
                source: path,
                exposed: rename.rename.to_string(),
            });
        }
        syn::UseTree::Glob(_) => {
            let mut path = prefix.clone();
            path.push("*".to_owned());
            out.push(UseExport {
                source: path,
                exposed: "*".to_owned(),
            });
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_exports(item, prefix, out);
            }
        }
    }
}

fn is_banned_facade_module_reexport(path: &[String], banned_modules: &BTreeSet<&str>) -> bool {
    match path {
        [first, second, ..] if first == "crate" || first == "self" => {
            banned_modules.contains(second.as_str())
        }
        [first, ..] => banned_modules.contains(first.as_str()),
        [] => false,
    }
}

fn is_agent_reexport(path: &[String]) -> bool {
    match path {
        [first, ..] if first == "entelix_agents" => true,
        [first, ..] if first == "agents" => true,
        [first, second, ..] if (first == "crate" || first == "self") && second == "agents" => true,
        _ => false,
    }
}

fn curated_surface_allowlist<'a>(
    path: &[String],
    allowed_core: &'a BTreeSet<&str>,
    allowed_graph: &'a BTreeSet<&str>,
    allowed_prompt: &'a BTreeSet<&str>,
    allowed_runnable: &'a BTreeSet<&str>,
) -> Option<(&'static str, &'a BTreeSet<&'a str>)> {
    let root = match path {
        [first, second, ..] if first == "crate" || first == "self" => second.as_str(),
        [first, ..] => first.as_str(),
        [] => return None,
    };
    if root == "entelix_core" || TOP_LEVEL_CORE_ALLOWLIST.contains(&root) {
        return Some(("core", allowed_core));
    }
    match root {
        "entelix_graph" | "graph" => Some(("graph", allowed_graph)),
        "entelix_prompt" | "prompt" => Some(("prompt", allowed_prompt)),
        "entelix_runnable" | "runnable" => Some(("runnable", allowed_runnable)),
        _ => None,
    }
}

fn type_path_segments(ty: &syn::Type) -> Option<Vec<String>> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segments: Vec<String> = path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    (!segments.is_empty()).then_some(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_underlying_crate_reexport_is_seen() {
        let paths = use_paths("pub use {entelix_session::GraphEvent};");

        assert_eq!(
            paths,
            vec![vec![
                "entelix_session".to_owned(),
                "GraphEvent".to_owned()
            ]]
        );
    }

    #[test]
    fn crate_named_module_reexport_is_banned() {
        let paths = use_paths("pub use crate::{session::GraphEvent};");
        let banned_modules: BTreeSet<&str> = BANNED_TOP_LEVEL_MODULES.iter().copied().collect();

        assert!(is_banned_facade_module_reexport(
            &paths[0],
            &banned_modules
        ));
    }

    #[test]
    fn advanced_agent_module_reexport_is_subject_to_allowlist() {
        let paths = use_paths("pub use crate::agents::ApprovalLayer;");

        assert!(is_agent_reexport(&paths[0]));
        assert!(!TOP_LEVEL_AGENT_ALLOWLIST.contains(&"ApprovalLayer"));
    }

    #[test]
    fn renamed_underlying_crate_reexport_uses_source_path() {
        let exports = use_exports("pub use entelix_session as durable_session;");

        assert_eq!(
            exports,
            vec![UseExport {
                source: vec!["entelix_session".to_owned()],
                exposed: "durable_session".to_owned(),
            }]
        );
    }

    #[test]
    fn renamed_named_module_reexport_uses_source_path() {
        let paths = use_paths("pub use crate::session as durable_session;");
        let banned_modules: BTreeSet<&str> = BANNED_TOP_LEVEL_MODULES.iter().copied().collect();

        assert!(is_banned_facade_module_reexport(
            &paths[0],
            &banned_modules
        ));
    }

    #[test]
    fn disallowed_core_module_leaf_is_rejected() {
        let paths = use_paths("pub use crate::codecs::AnthropicMessagesCodec;");
        let allowed_core: BTreeSet<&str> = TOP_LEVEL_CORE_ALLOWLIST.iter().copied().collect();
        let allowed_graph: BTreeSet<&str> = TOP_LEVEL_GRAPH_ALLOWLIST.iter().copied().collect();
        let allowed_prompt: BTreeSet<&str> = TOP_LEVEL_PROMPT_ALLOWLIST.iter().copied().collect();
        let allowed_runnable: BTreeSet<&str> = TOP_LEVEL_RUNNABLE_ALLOWLIST.iter().copied().collect();

        let (surface, allowed) = curated_surface_allowlist(
            &paths[0],
            &allowed_core,
            &allowed_graph,
            &allowed_prompt,
            &allowed_runnable,
        )
        .expect("core module is curated");

        assert_eq!(surface, "core");
        assert!(!allowed.contains("AnthropicMessagesCodec"));
    }

    #[test]
    fn renamed_allowed_core_leaf_is_rejected_by_exposed_name() {
        let exports = use_exports("pub use entelix_core::auth as auth2;");
        let allowed_core: BTreeSet<&str> = TOP_LEVEL_CORE_ALLOWLIST.iter().copied().collect();
        let allowed_graph: BTreeSet<&str> = TOP_LEVEL_GRAPH_ALLOWLIST.iter().copied().collect();
        let allowed_prompt: BTreeSet<&str> = TOP_LEVEL_PROMPT_ALLOWLIST.iter().copied().collect();
        let allowed_runnable: BTreeSet<&str> = TOP_LEVEL_RUNNABLE_ALLOWLIST.iter().copied().collect();

        let (surface, allowed) = curated_surface_allowlist(
            &exports[0].source,
            &allowed_core,
            &allowed_graph,
            &allowed_prompt,
            &allowed_runnable,
        )
        .expect("core module is curated");

        assert_eq!(surface, "core");
        assert!(allowed.contains("auth"));
        assert!(!allowed.contains(exports[0].exposed.as_str()));
    }

    #[test]
    fn disallowed_graph_leaf_is_rejected() {
        let paths = use_paths("pub use entelix_graph::Dispatch;");
        let allowed_core: BTreeSet<&str> = TOP_LEVEL_CORE_ALLOWLIST.iter().copied().collect();
        let allowed_graph: BTreeSet<&str> = TOP_LEVEL_GRAPH_ALLOWLIST.iter().copied().collect();
        let allowed_prompt: BTreeSet<&str> = TOP_LEVEL_PROMPT_ALLOWLIST.iter().copied().collect();
        let allowed_runnable: BTreeSet<&str> = TOP_LEVEL_RUNNABLE_ALLOWLIST.iter().copied().collect();

        let (surface, allowed) = curated_surface_allowlist(
            &paths[0],
            &allowed_core,
            &allowed_graph,
            &allowed_prompt,
            &allowed_runnable,
        )
        .expect("graph is curated");

        assert_eq!(surface, "graph");
        assert!(!allowed.contains("Dispatch"));
    }

    #[test]
    fn prelude_rejects_advanced_session_surface() {
        let item_mod = syn::parse_str::<syn::ItemMod>(
            r#"
            pub mod prelude {
                pub use entelix_session::GraphEvent;
            }
            "#,
        )
        .expect("valid module");
        let prelude_allowed: BTreeSet<&str> = PRELUDE_ALLOWLIST.iter().copied().collect();
        let banned_crates: BTreeSet<&str> = BANNED_TOP_LEVEL_CRATES.iter().copied().collect();
        let banned_modules: BTreeSet<&str> = BANNED_TOP_LEVEL_MODULES.iter().copied().collect();
        let mut violations = Vec::new();

        check_prelude(
            std::path::Path::new("facade.rs"),
            &item_mod,
            &prelude_allowed,
            &banned_crates,
            &banned_modules,
            &mut violations,
        );

        assert_eq!(violations.len(), 2);
    }

    #[test]
    fn prelude_rejects_renamed_allowed_surface() {
        let item_mod = syn::parse_str::<syn::ItemMod>(
            r#"
            pub mod prelude {
                pub use entelix_core::ChatModel as Model;
            }
            "#,
        )
        .expect("valid module");
        let prelude_allowed: BTreeSet<&str> = PRELUDE_ALLOWLIST.iter().copied().collect();
        let banned_crates: BTreeSet<&str> = BANNED_TOP_LEVEL_CRATES.iter().copied().collect();
        let banned_modules: BTreeSet<&str> = BANNED_TOP_LEVEL_MODULES.iter().copied().collect();
        let mut violations = Vec::new();

        check_prelude(
            std::path::Path::new("facade.rs"),
            &item_mod,
            &prelude_allowed,
            &banned_crates,
            &banned_modules,
            &mut violations,
        );

        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn type_alias_preserves_full_source_path() {
        let path = type_alias_path("pub type ApiKeyProvider = entelix_core::auth::ApiKeyProvider;");

        assert_eq!(
            path,
            vec![
                "entelix_core".to_owned(),
                "auth".to_owned(),
                "ApiKeyProvider".to_owned()
            ]
        );
    }

    #[test]
    fn type_alias_to_advanced_core_surface_is_not_allowed() {
        let path = type_alias_path("pub type ApiKeyProvider = entelix_core::auth::ApiKeyProvider;");
        let allowed_core: BTreeSet<&str> = TOP_LEVEL_CORE_ALLOWLIST.iter().copied().collect();
        let allowed_graph: BTreeSet<&str> = TOP_LEVEL_GRAPH_ALLOWLIST.iter().copied().collect();
        let allowed_prompt: BTreeSet<&str> = TOP_LEVEL_PROMPT_ALLOWLIST.iter().copied().collect();
        let allowed_runnable: BTreeSet<&str> = TOP_LEVEL_RUNNABLE_ALLOWLIST.iter().copied().collect();

        let (surface, allowed) = curated_surface_allowlist(
            &path,
            &allowed_core,
            &allowed_graph,
            &allowed_prompt,
            &allowed_runnable,
        )
        .expect("core is curated");

        assert_eq!(surface, "core");
        assert!(!allowed.contains("ApiKeyProvider"));
    }

    #[test]
    fn type_alias_to_advanced_agent_surface_is_not_allowed() {
        let path = type_alias_path("pub type ApprovalLayer = crate::agents::ApprovalLayer;");

        assert!(is_agent_reexport(&path));
        assert!(!TOP_LEVEL_AGENT_ALLOWLIST.contains(&"ApprovalLayer"));
    }

    #[test]
    fn type_alias_renaming_allowed_core_surface_is_not_allowed() {
        let path = type_alias_path("pub type Model = entelix_core::ChatModel;");
        let exposed = "Model";
        let source = path.last().map(String::as_str).unwrap_or("*");
        let allowed_core: BTreeSet<&str> = TOP_LEVEL_CORE_ALLOWLIST.iter().copied().collect();

        assert!(allowed_core.contains(source));
        assert_ne!(source, exposed);
    }

    #[test]
    fn type_alias_renaming_allowed_agent_surface_is_not_allowed() {
        let path = type_alias_path("pub type Worker = crate::agents::Subagent;");
        let exposed = "Worker";
        let source = path.last().map(String::as_str).unwrap_or("*");

        assert!(is_agent_reexport(&path));
        assert!(TOP_LEVEL_AGENT_ALLOWLIST.contains(&source));
        assert_ne!(source, exposed);
    }

    fn use_paths(src: &str) -> Vec<Vec<String>> {
        use_exports(src)
            .into_iter()
            .map(|export| export.source)
            .collect()
    }

    fn use_exports(src: &str) -> Vec<UseExport> {
        let item = syn::parse_str::<syn::ItemUse>(src).expect("valid use item");
        let mut exports = Vec::new();
        collect_use_exports(&item.tree, &mut Vec::new(), &mut exports);
        exports
    }

    fn type_alias_path(src: &str) -> Vec<String> {
        let item = syn::parse_str::<syn::ItemType>(src).expect("valid type alias");
        type_path_segments(&item.ty).expect("path type")
    }
}
