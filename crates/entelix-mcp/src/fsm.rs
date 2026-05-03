//! Forward-only client lifecycle state machine. Every transition
//! advances; no state can ever go backwards. `Failed` is terminal;
//! the surrounding [`crate::McpClient`] is replaced rather than
//! rewound.

// `pub(crate)` is required to satisfy the workspace `unreachable_pub` rust
// lint; clippy nursery's `redundant_pub_crate` disagrees and we side with
// the rust lint.
#![allow(clippy::redundant_pub_crate)]

use std::sync::atomic::{AtomicU8, Ordering};

/// Lifecycle position of an [`crate::McpClient`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum McpClientState {
    /// Configured but not yet attempted to connect.
    Queued = 0,
    /// HTTP client built; about to dial the server.
    Spawn = 1,
    /// TCP+TLS handshake in progress.
    Handshake = 2,
    /// JSON-RPC `initialize` request in flight.
    InitializeProtocol = 3,
    /// Server's reply parsed; capability set being recorded.
    NegotiateCapabilities = 4,
    /// `tools/list` request in flight.
    ListTools = 5,
    /// `resources/list` request in flight (skipped if not advertised).
    ListResources = 6,
    /// `prompts/list` request in flight (skipped if not advertised).
    ListPrompts = 7,
    /// Optional warmup step — advanced policies may pre-resolve
    /// expensive tool descriptors at this point.
    CacheWarmup = 8,
    /// Ready to serve `tools/call` requests.
    Ready = 9,
    /// Terminal failure. Owner must drop and rebuild.
    Failed = 10,
}

impl McpClientState {
    fn rank(self) -> u8 {
        self as u8
    }

    /// Forward-only check: `to` must be `>=` `from`. Used by the
    /// atomic transition helper to refuse rewinds in debug builds.
    pub fn is_forward(from: Self, to: Self) -> bool {
        to.rank() >= from.rank()
    }
}

/// Atomic state cell that enforces forward-only transitions.
#[derive(Debug)]
pub(crate) struct StateCell(AtomicU8);

impl StateCell {
    pub(crate) const fn new() -> Self {
        Self(AtomicU8::new(McpClientState::Queued as u8))
    }

    pub(crate) fn load(&self) -> McpClientState {
        from_rank(self.0.load(Ordering::Acquire))
    }

    /// Move to `next`. Returns `true` on success, `false` (and leaves
    /// the value unchanged) when `next` would be a rewind. Debug
    /// builds panic on rewind to surface bugs in the calling code.
    pub(crate) fn advance(&self, next: McpClientState) -> bool {
        let mut current_rank = self.0.load(Ordering::Acquire);
        loop {
            let current = from_rank(current_rank);
            if !McpClientState::is_forward(current, next) {
                debug_assert!(false, "McpClientState rewind: {current:?} → {next:?}");
                return false;
            }
            match self.0.compare_exchange_weak(
                current_rank,
                next as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current_rank = observed,
            }
        }
    }
}

fn from_rank(rank: u8) -> McpClientState {
    match rank {
        0 => McpClientState::Queued,
        1 => McpClientState::Spawn,
        2 => McpClientState::Handshake,
        3 => McpClientState::InitializeProtocol,
        4 => McpClientState::NegotiateCapabilities,
        5 => McpClientState::ListTools,
        6 => McpClientState::ListResources,
        7 => McpClientState::ListPrompts,
        8 => McpClientState::CacheWarmup,
        9 => McpClientState::Ready,
        _ => McpClientState::Failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_transitions_succeed() {
        let cell = StateCell::new();
        assert_eq!(cell.load(), McpClientState::Queued);
        assert!(cell.advance(McpClientState::Spawn));
        assert!(cell.advance(McpClientState::Handshake));
        assert!(cell.advance(McpClientState::InitializeProtocol));
        assert!(cell.advance(McpClientState::Ready));
        assert_eq!(cell.load(), McpClientState::Ready);
    }

    #[test]
    fn idempotent_self_transition_is_allowed() {
        let cell = StateCell::new();
        assert!(cell.advance(McpClientState::Queued));
        assert_eq!(cell.load(), McpClientState::Queued);
    }

    #[test]
    #[should_panic(expected = "rewind")]
    fn rewind_panics_in_debug() {
        let cell = StateCell::new();
        assert!(cell.advance(McpClientState::Ready));
        // Ready -> InitializeProtocol is a rewind.
        cell.advance(McpClientState::InitializeProtocol);
    }

    #[test]
    fn failed_is_terminal_high_rank() {
        let cell = StateCell::new();
        assert!(cell.advance(McpClientState::Failed));
        // Failed > Ready, so this is also rejected as a rewind.
        let result = std::panic::catch_unwind(|| cell.advance(McpClientState::Ready));
        assert!(result.is_err(), "Ready after Failed must be rejected");
    }
}
