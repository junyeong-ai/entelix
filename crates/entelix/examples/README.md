# entelix — examples catalogue

Runnable, deterministic examples. None hit a real LLM — every example
uses an in-process model stub or a pre-canned wire response so it runs
in CI without API keys. Read the source and the doc-comment in each
file for the full walkthrough.

## Reading order

| Stage | Examples | Goal |
|---|---|---|
| **Foundation** | 01, 02, 09 | Single model call → LCEL pipe → multi-codec parity |
| **Control flow** | 03, 16 | `StateGraph` conditional routing → `#[derive(StateMerge)]` per-field reducers |
| **Streaming** | 10, 08 | Token-level deltas → 5-mode graph streaming |
| **Memory** | 05, 20, 21 | Tier-3 KV memory → graph memory → episodic log |
| **Agents** | 06, 07, 15 | Supervisor → nested supervisor → production workflow |
| **HITL & approval** | 04, 18 | Graph-level `interrupt()` → tool-dispatch approval gate |
| **Durable & multi-cloud** | 11, 12 | Resume after pod restart → codec×transport sparse matrix |
| **Tools & MCP** | 13, 17 | MCP-published tools → MCP sampling provider |
| **Server & typed output** | 14, 19 | `AgentRouterBuilder` HTTP serve → `complete_typed_validated` |

## Catalogue

| # | Example | Demonstrates | Feature gate |
|---|---|---|---|
| 01 | `01_quickstart.rs` | Single Anthropic Messages call | (default) |
| 02 | `02_lcel_chain.rs` | `prompt.pipe(model).pipe(parser)` end-to-end | (default) |
| 03 | `03_state_graph.rs` | Multi-step workflow with conditional routing | (default) |
| 04 | `04_hitl.rs` | Human-in-the-loop with `interrupt()` + `Command` | (default) |
| 05 | `05_memory.rs` | Tier-3 cross-thread memory (`BufferMemory` + `EntityMemory` + tenant isolation) | (default) |
| 06 | `06_supervisor.rs` | Multi-agent supervisor with two named sub-agents | (default) |
| 07 | `07_hierarchical.rs` | Nested supervisor (`team_from_supervisor` adapter) | (default) |
| 08 | `08_streaming_modes.rs` | Same graph, streamed under all five `StreamMode`s | (default) |
| 09 | `09_multi_codec.rs` | Same `Vec<Message>` through every shipped codec | (default) |
| 10 | `10_streaming.rs` | Token-level `StreamMode::Messages` from a fake SSE source | (default) |
| 11 | `11_durable_session.rs` | Pod-kill / resume — invariant 2 (stateless harness) | (default) |
| 12 | `12_compat_matrix.rs` | Sparse codec×transport pairing matrix | `aws`, `gcp`, `azure` |
| 13 | `13_mcp_tools.rs` | MCP-published tool through the `Tool` adapter, per-tenant pool isolation | `mcp` |
| 14 | `14_serve_agent.rs` | `AgentRouterBuilder` HTTP server end-to-end | `server` |
| 15 | `15_production_workflow.rs` | Composed end-to-end agent workflow with policy/observability | `policy` |
| 16 | `16_state_merge_pipeline.rs` | `#[derive(StateMerge)]` per-field reducers + parallel fan-out + contribution nodes | (default) |
| 17 | `17_mcp_sampling_provider.rs` | MCP `sampling/createMessage` dispatched through `ChatModelSamplingProvider` | `mcp`, `mcp-chatmodel` |
| 18 | `18_tool_approval.rs` | HITL approval at the tool-dispatch boundary, pause-and-resume cycle | `policy` |
| 19 | `19_typed_output.rs` | `complete_typed` / `complete_typed_validated` / `stream_typed` (invariant 20) | (default) |
| 20 | `20_graph_memory.rs` | `GraphMemory<N, E>` — `add_node` / `add_edge` / `neighbors` / BFS `traverse` / `find_path` | (default) |
| 21 | `21_episodic_memory.rs` | `EpisodicMemory<V>` — `append_at` / `recent` / `range` / `since` / `prune_older_than` | (default) |
| 22 | `22_agent_with_observer.rs` | Headline `Agent::builder()` + `AgentObserver` + `CaptureSink` lifecycle observation | (default) |
| 23 | `23_typed_tool_macro.rs` | `#[tool]` proc-macro typed-input tool authoring + dispatch | (default) |
| 24 | `24_file_id_attachments.rs` | `MediaSource::FileId` for large-attachment workflows (Anthropic Files / OpenAI Files / `GeminiExt::cached_content`) | (default) |
| 25 | `25_claude_code_oauth.rs` | `ClaudeCodeOAuthProvider` + `FileCredentialStore` — drive entelix with the OAuth token the `claude` CLI manages | `auth-claude-code` |

## Running

```bash
# Default-features example
cargo run --example 01_quickstart -p entelix

# Feature-gated example
cargo run --example 13_mcp_tools  -p entelix --features mcp
cargo run --example 14_serve_agent -p entelix --features server
cargo run --example 18_tool_approval -p entelix --features policy

# Multi-cloud example (requires the three cloud features)
cargo run --example 12_compat_matrix -p entelix --features "aws,gcp,azure"
```

## Invariant coverage

Every architecture invariant from the root [`CLAUDE.md`](../../../CLAUDE.md)
is exercised by at least one example below. The mapping makes
regressions visible at the example layer, not only at unit-test depth.

| Invariant | Example(s) |
|---|---|
| 1. Session is event SSoT | 11 |
| 2. Harness is stateless | 11 |
| 3. 3-tier state separation | 05 |
| 4. `Tool::execute` single contract | 13, 18 |
| 5. Provider IR before wire format | 09, 12 |
| 6. Lossy encoding emits warnings | 09 |
| 7. Runnable is the composition contract | 02, 16 |
| 8. StateGraph is the control-flow contract | 03, 16 |
| 11. Multi-tenant `Namespace` mandatory | 05, 13, 20, 21 |
| 16. LLM/operator channel separation | 19 |
| 18. Managed-agent lifecycle is auditable | 06, 07, 18 |
| 20. Validation retry budget is unified | 19 |
