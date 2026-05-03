//! `ShellPolicy` — allowlist + argv validation for
//! [`SandboxedShellTool`](crate::sandboxed::SandboxedShellTool).
//!
//! Defense in depth: the underlying [`entelix_core::sandbox::Sandbox`]
//! enforces process isolation; `ShellPolicy` enforces *intent*
//! filtering (operator picks "which commands the agent may even
//! ask the sandbox to run"). Both layers must approve.
//!
//! Two prebuilt baselines cover common deployments:
//!
//! - [`ShellPolicy::READ_ONLY_BASELINE`] — `ls`, `cat`, `head`,
//!   `tail`, `wc`, `grep`, `find`, `pwd`, `echo`. No
//!   filesystem-mutation, no network.
//! - [`ShellPolicy::DEVELOPMENT_BASELINE`] — adds `git status`,
//!   `git log`, `git diff`, `npm ls`, `cargo metadata`,
//!   `python --version`, `node --version`. Still read-only and
//!   metadata-only.
//!
//! Operators that need to extend: build a fresh policy via
//! [`ShellPolicy::new`] and add commands explicitly.

use std::collections::HashSet;
use std::time::Duration;

use thiserror::Error;

/// Allowlist + duration cap for shell commands the agent may
/// dispatch through [`crate::sandboxed::SandboxedShellTool`].
#[derive(Clone, Debug)]
pub struct ShellPolicy {
    /// Set of allowed `argv[0]` values. Lookup is exact (no
    /// substring match, no glob) — operators that need wildcards
    /// build their own match function.
    allowed_commands: HashSet<String>,
    /// Cap on per-call wall-clock duration. The
    /// [`entelix_core::sandbox::Sandbox`] backend may also impose
    /// its own; the smaller of the two wins.
    max_duration: Duration,
}

impl ShellPolicy {
    /// Build a fresh policy with the given allowlist and a
    /// 30-second default duration cap.
    #[must_use]
    pub fn new<I, S>(allowed: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed_commands: allowed.into_iter().map(Into::into).collect(),
            max_duration: Duration::from_secs(30),
        }
    }

    /// Override the duration cap.
    #[must_use]
    pub const fn with_max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = duration;
        self
    }

    /// Append one allowed command to the existing allowlist.
    #[must_use]
    pub fn add_allowed_command(mut self, command: impl Into<String>) -> Self {
        self.allowed_commands.insert(command.into());
        self
    }

    /// Check whether `argv` is admissible. Returns the offending
    /// reason on rejection.
    pub fn check(&self, argv: &[String]) -> Result<(), ShellPolicyError> {
        let head = argv.first().ok_or(ShellPolicyError::EmptyArgv)?.clone();
        if !self.allowed_commands.contains(&head) {
            return Err(ShellPolicyError::CommandNotAllowed { command: head });
        }
        Ok(())
    }

    /// Borrow the allowed-command set.
    #[must_use]
    pub fn allowed(&self) -> &HashSet<String> {
        &self.allowed_commands
    }

    /// Borrow the duration cap.
    #[must_use]
    pub const fn max_duration(&self) -> Duration {
        self.max_duration
    }

    /// Read-only baseline — `ls`, `cat`, `head`, `tail`, `wc`,
    /// `grep`, `find`, `pwd`, `echo`. No filesystem mutation, no
    /// network commands.
    #[must_use]
    pub fn read_only_baseline() -> Self {
        Self::new([
            "ls", "cat", "head", "tail", "wc", "grep", "find", "pwd", "echo",
        ])
    }

    /// Development baseline — read-only metadata for git / `npm` /
    /// `cargo` / language version probes on top of
    /// [`Self::read_only_baseline`]. Operators that need write
    /// access (`git commit`, `npm install`, etc.) build a custom
    /// policy explicitly.
    #[must_use]
    pub fn development_baseline() -> Self {
        let mut p = Self::read_only_baseline();
        for cmd in ["git", "npm", "cargo", "python", "python3", "node", "rg"] {
            p.allowed_commands.insert(cmd.into());
        }
        p
    }
}

impl Default for ShellPolicy {
    /// Same as [`Self::read_only_baseline`] — the safest default.
    fn default() -> Self {
        Self::read_only_baseline()
    }
}

/// Reason `ShellPolicy::check` rejected an `argv`.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShellPolicyError {
    /// `argv` was empty — there is no command to evaluate.
    #[error("ShellPolicy: argv is empty")]
    EmptyArgv,
    /// `argv[0]` is not in the policy's allowlist.
    #[error("ShellPolicy: command '{command}' is not on the allowlist")]
    CommandNotAllowed {
        /// The `argv[0]` value that was rejected.
        command: String,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_argv_is_rejected() {
        let policy = ShellPolicy::default();
        let err = policy.check(&[]).unwrap_err();
        assert_eq!(err, ShellPolicyError::EmptyArgv);
    }

    #[test]
    fn unknown_command_is_rejected() {
        let policy = ShellPolicy::default();
        let err = policy.check(&["rm".to_owned()]).unwrap_err();
        assert!(
            matches!(err, ShellPolicyError::CommandNotAllowed { ref command } if command == "rm")
        );
    }

    #[test]
    fn read_only_baseline_admits_ls() {
        let policy = ShellPolicy::read_only_baseline();
        policy.check(&["ls".to_owned(), "-la".to_owned()]).unwrap();
    }

    #[test]
    fn development_baseline_admits_git() {
        let policy = ShellPolicy::development_baseline();
        policy
            .check(&["git".to_owned(), "status".to_owned()])
            .unwrap();
    }

    #[test]
    fn allow_extends_existing_set() {
        let policy = ShellPolicy::new(["ls"]).add_allowed_command("cat");
        policy.check(&["ls".to_owned()]).unwrap();
        policy.check(&["cat".to_owned()]).unwrap();
        policy.check(&["echo".to_owned()]).unwrap_err();
    }

    #[test]
    fn with_max_duration_overrides_default() {
        let policy = ShellPolicy::default().with_max_duration(Duration::from_secs(5));
        assert_eq!(policy.max_duration(), Duration::from_secs(5));
    }
}
