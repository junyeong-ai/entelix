# entelix-tools-coding

Coding-agent vertical companion to `entelix-tools`. Every shell / code / filesystem operation delegates through `entelix_core::sandbox::Sandbox` — invariant 9 holds inside this crate (no `std::fs` / `std::process` import lands here).

## Surface

- **`SandboxedShellTool`** + **`ShellPolicy`** — execute commands through `Sandbox::run` under an operator-supplied allowlist (command name, argument shape, env-var prefix, working-directory restriction). `ShellPolicy::READ_ONLY_BASELINE` is the canonical conservative default.
- **`SandboxedCodeTool`** + **`CodePolicy`** — execute source code in a `SandboxLanguage` chosen at construction. `CodePolicy` gates language family + per-language interpreter args.
- **`SandboxedReadFileTool` / `SandboxedWriteFileTool` / `SandboxedListDirTool`** — filesystem ops scoped to a sandbox root supplied by the operator. Path resolution + canonicalisation happen inside the `Sandbox` impl, not in the tool.
- **`SandboxSkill`** + **`SandboxResource`** + **`SkillManifest`** + **`parse_skill_md`** — `entelix_core::Skill` impl backed by a sandbox-internal directory tree mirroring the Anthropic Claude Skills layout (`SKILL.md` + sibling resource files). Lazy resource bodies; binary resources return metadata only on the LLM-facing channel.
- **`MockSandbox`** — programmable in-memory `Sandbox` impl for tests and as the regression baseline downstream sandbox companions (`entelix-sandbox-e2b`, `entelix-sandbox-modal`, …) check their behaviour against.

## Crate-local rules

- **Invariant 9 holds inside this crate too.** Every syscall routes through `Sandbox` — adding `tokio::fs::read` / `tokio::process::Command::spawn` defeats the whole boundary. `cargo xtask no-fs` enforces.
- **Policy allowlists default-deny.** `ShellPolicy::new()` permits nothing; `CodePolicy::new()` permits nothing. Operators add language families / commands explicitly. Convenience constants like `ShellPolicy::READ_ONLY_BASELINE` exist for the common shape but are still opt-in.
- **Skill manifest `parse_skill_md` is total.** A malformed `SKILL.md` surfaces as `ManifestError` carrying file path + reason; never panics, never produces an empty-but-valid manifest as silent fallback.
- **`MockSandbox` is the test baseline.** Downstream sandbox impls (`Landlock`, `Seatbelt`, `e2b`, `modal`) should run the same regression suite this crate ships in `tests/` so behaviour drift surfaces under CI.

## Forbidden

- A `std::fs` / `tokio::fs` / `std::process` / `tokio::process` import anywhere in this crate.
- A `Tool` impl that reads credentials from `ExecutionContext` (invariant 10).
- A default-allow `ShellPolicy` / `CodePolicy` — fail-closed at construction is non-negotiable.

## References

- Root `CLAUDE.md` invariant 9 — filesystem / shell boundary.
- `entelix-tools` CLAUDE.md — horizontal-tools surface that this crate complements.
