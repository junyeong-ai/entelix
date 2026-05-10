# entelix-graph-derive

[![docs.rs](https://docs.rs/entelix-graph-derive/badge.svg)](https://docs.rs/entelix-graph-derive)
[![crates.io](https://img.shields.io/crates/v/entelix-graph-derive.svg)](https://crates.io/crates/entelix-graph-derive)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/junyeong-ai/entelix/blob/main/LICENSE)

Proc-macro for `#[derive(StateMerge)]`. Emits a `<Name>Contribution` companion struct (per-field `Option<T>`) + `with_<field>` builder methods + the `StateMerge` impl that auto-wraps raw `T` into `Annotated::new(value, R::default())` for annotated fields.

Part of the [`entelix`](https://github.com/junyeong-ai/entelix) agentic-AI SDK workspace — see the [workspace README](https://github.com/junyeong-ai/entelix#readme) for project overview, quickstart, examples, and architecture.

## Documentation

- API reference: <https://docs.rs/entelix-graph-derive>
- Workspace overview: <https://github.com/junyeong-ai/entelix>

## License

MIT — see [LICENSE](https://github.com/junyeong-ai/entelix/blob/main/LICENSE).
