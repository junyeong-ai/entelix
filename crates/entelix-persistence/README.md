# entelix-persistence

[![docs.rs](https://docs.rs/entelix-persistence/badge.svg)](https://docs.rs/entelix-persistence)
[![crates.io](https://img.shields.io/crates/v/entelix-persistence.svg)](https://crates.io/crates/entelix-persistence)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/junyeong-ai/entelix/blob/main/LICENSE)

Postgres + Redis backends for `Checkpointer` (entelix-graph) + `Store` (entelix-memory) + `SessionLog` (entelix-session) + `DistributedLock` (this crate). Aggregate facade `*Persistence` carries a wired pool plus the trait impls.

Part of the [`entelix`](https://github.com/junyeong-ai/entelix) agentic-AI SDK workspace — see the [workspace README](https://github.com/junyeong-ai/entelix#readme) for project overview, quickstart, examples, and architecture.

## Documentation

- API reference: <https://docs.rs/entelix-persistence>
- Workspace overview: <https://github.com/junyeong-ai/entelix>

## License

MIT — see [LICENSE](https://github.com/junyeong-ai/entelix/blob/main/LICENSE).
