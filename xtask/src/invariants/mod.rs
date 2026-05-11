//! Per-invariant visitor modules. Each exposes a single `pub fn run() ->
//! Result<()>` entry point so `main.rs` can dispatch uniformly.

pub(crate) mod advisory_expiry;
pub(crate) mod dead_deps;
pub(crate) mod doc_canonical_paths;
pub(crate) mod facade_completeness;
pub(crate) mod feature_matrix;
pub(crate) mod lock_ordering;
pub(crate) mod magic_constants;
pub(crate) mod managed_shape;
pub(crate) mod naming;
pub(crate) mod no_fs;
pub(crate) mod no_shims;
pub(crate) mod public_api;
pub(crate) mod silent_fallback;
pub(crate) mod supply_chain;
pub(crate) mod surface_hygiene;
