# entelix-graphmemory-pg

[![docs.rs](https://docs.rs/entelix-graphmemory-pg/badge.svg)](https://docs.rs/entelix-graphmemory-pg)
[![crates.io](https://img.shields.io/crates/v/entelix-graphmemory-pg.svg)](https://crates.io/crates/entelix-graphmemory-pg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/junyeong-ai/entelix/blob/main/LICENSE)

Companion crate. Concrete `GraphMemory<N, E>` impl backed by Postgres (sqlx) — production graph-memory tier with row-level security, single-SQL BFS via `WITH RECURSIVE`, and `INSERT … SELECT FROM UNNEST` bulk insert.

Part of the [`entelix`](https://github.com/junyeong-ai/entelix) agentic-AI SDK workspace — see the [workspace README](https://github.com/junyeong-ai/entelix#readme) for project overview, quickstart, examples, and architecture.

## Documentation

- API reference: <https://docs.rs/entelix-graphmemory-pg>
- Workspace overview: <https://github.com/junyeong-ai/entelix>

## License

MIT — see [LICENSE](https://github.com/junyeong-ai/entelix/blob/main/LICENSE).
