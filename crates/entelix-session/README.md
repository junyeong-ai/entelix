# entelix-session

[![docs.rs](https://docs.rs/entelix-session/badge.svg)](https://docs.rs/entelix-session)
[![crates.io](https://img.shields.io/crates/v/entelix-session.svg)](https://crates.io/crates/entelix-session)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/junyeong-ai/entelix/blob/main/LICENSE)

Event-sourced session state (invariant 1 — session is event SSoT). `SessionGraph::events: Vec<GraphEvent>` is the only first-class data for audit; nodes / branches / checkpoints are derived.

Part of the [`entelix`](https://github.com/junyeong-ai/entelix) agentic-AI SDK workspace — see the [workspace README](https://github.com/junyeong-ai/entelix#readme) for project overview, quickstart, examples, and architecture.

## Documentation

- API reference: <https://docs.rs/entelix-session>
- Workspace overview: <https://github.com/junyeong-ai/entelix>

## License

MIT — see [LICENSE](https://github.com/junyeong-ai/entelix/blob/main/LICENSE).
