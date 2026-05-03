# ADR 0003 — No filesystem, no shell, no local sandbox

**Status**: Accepted
**Date**: 2026-04-26

## Context

Mixing a coding-agent surface (filesystem, shell, Landlock / Seatbelt, BashAnalyzer) with a web-service surface in one library produces two distinct user personas in one rustdoc — and confuses both.

The web-service persona (multi-tenant SaaS, customer support bot, RAG endpoint, scheduled agent) does not want:
- `Read`/`Write`/`Edit`/`Glob`/`Grep` tools — these read process-local files, irrelevant in containerized deployments
- `Bash` execution — multi-tenancy makes shell-level sandboxing the wrong place to enforce isolation
- `Landlock`/`Seatbelt` — these are POSIX/macOS sandbox primitives; in production agents run inside containers/serverless, where the deployment layer enforces isolation

## Decision

entelix forbids — at the architecture level — filesystem and shell access from any first-party crate.

**Forbidden imports** (CI-enforced via grep gate `scripts/check-no-fs.sh`):

```
std::fs::*
std::process::*
std::os::unix::process::*
tokio::fs::*
tokio::process::*
```

**Forbidden dependencies** in any first-party crate's `Cargo.toml`:

```
landlock        # Linux LSM sandbox
tree-sitter*    # AST parsing for shell analysis
seatbelt        # macOS sandbox
nix             # POSIX wrapper (often used for fs ops)
rustix (fs feat)# direct syscall fs ops
```

`Path`/`PathBuf` may be used as opaque identifiers (e.g., MCP stdio binary path) but never to read/write file content.

**Sandboxing is the deployment layer's job.** Users deploy entelix agents in:
- Containers (Docker, Kubernetes) with seccomp/AppArmor profiles
- Serverless (Lambda, Cloud Run, Workers) with vendor-managed isolation
- Restate (durable execution) with virtual-object isolation
- Custom orchestrators (Temporal, Inngest)

entelix supplies the agent runtime; the platform supplies the isolation.

## Consequences

✅ Smaller dependency surface — no `landlock`, no `tree-sitter*`, no `rustix-fs`.
✅ Single deployment story — every entelix agent is HTTP-callable, container-deployable.
✅ No platform-specific code paths — same code on Linux/macOS/Windows.
✅ Smaller compiled binary — no Landlock / Seatbelt / tree-sitter linkage.
✅ Cleaner threat model — credentials live in the transport layer; tools never see the local FS.
❌ Users who want a coding agent in Rust must build on `entelix-core` directly with their own host-primitive tools, or pick a coding-agent harness with built-in primitives.
❌ Some tools that traditionally read local files (e.g., loading a PDF) need to fetch via HTTP instead — accepted, since web services rarely have local PDFs anyway.

## References

- Anthropic managed-agents: *"the sandbox is the deployment layer's responsibility"*
- Container security best practices (seccomp, AppArmor, gVisor)
