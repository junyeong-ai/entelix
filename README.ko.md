# entelix

[![Rust 1.95](https://img.shields.io/badge/rust-1.95.0%2B-orange?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![Edition 2024](https://img.shields.io/badge/edition-2024-blueviolet?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

> **[English](README.md)** | **한국어**

> **모든 에이전트가 자기 목적을 실현하는 곳.**

**Rust 로 작성된 범용 에이전트 AI SDK.** LLM 기반 control loop 에 도구·메모리·내구 상태를 묶은 어떤 종류의 에이전트도 구축할 수 있습니다. 멀티테넌트 구조가 처음부터 일급. OpenTelemetry GenAI semconv 네이티브. 22 가지 아키텍처 invariant 가 CI 로 강제됩니다. Anthropic *managed-agents* 구조. LangChain + LangGraph parity.

---

## 왜 entelix 인가?

- **멀티테넌트 일급** — `TenantId` 가 모든 영속화 경계에 강제. Postgres `FORCE ROW LEVEL SECURITY` 가 모든 백엔드에 defense-in-depth. 테넌트 간 데이터 누설은 API 설계상 구조적으로 불가능.
- **타입 강제 아키텍처** — sealed `RenderedForLlm<T>` 캐리어 (LLM / 운영자 채널 분리 컴파일 타임 강제); sealed `CompactedHistory` (`tool_call` / `tool_result` 페어 invariant 깨뜨릴 수 없음); `TenantId` 검증 newtype.
- **production observability 처음부터** — OpenTelemetry GenAI semconv `gen_ai.*` 속성. cost 는 트랜잭셔널 emit (`Ok` 분기에서만). 6 verb 의 typed `AuditSink` 채널 (`sub_agent_invoked` / `agent_handoff` / `resumed` / `memory_recall` / `usage_limit_exceeded` / `context_compacted`).
- **5 codec × 4 transport** — Anthropic Messages, OpenAI Chat, OpenAI Responses, Gemini, Bedrock Converse 가 Direct (어떤 HTTPS), Bedrock (SigV4), Vertex (gcp_auth), Foundry (api-key + AAD) 위에서 동작. 능력 정직성: 모든 변환 손실 필드에 `LossyEncode` 경고, 미지 vendor 시그널은 `Other { raw }` — silent fallback 없음.
- **MCP 일급** — 3 가지 server-initiated 채널 모두 (`Roots`, `Elicitation`, `Sampling`). per-tenant `(tenant_id, server_name)` pool 격리. HTTP-only 설계.
- **CI 강제 invariants** — `cargo xtask invariants` 가 매 push 마다 typed-AST visitor 들을 실행 (filesystem-free / naming taxonomy / silent-fallback / lock-ordering / public-API drift / supply-chain / feature-matrix gates).

---

## 빠른 시작

```bash
cargo add entelix
```

```rust
use std::sync::Arc;
use entelix::auth::ApiKeyProvider;
use entelix::codecs::AnthropicMessagesCodec;
use entelix::ir::Message;
use entelix::transports::DirectTransport;
use entelix::{ChatModel, ExecutionContext};

#[tokio::main]
async fn main() -> entelix::Result<()> {
    let creds = Arc::new(ApiKeyProvider::anthropic(std::env::var("ANTHROPIC_API_KEY")?));
    let transport = DirectTransport::anthropic(creds)?;
    let model = ChatModel::new(AnthropicMessagesCodec, transport, "claude-opus-4-7")
        .with_system("한 문장으로 답하세요.");

    let reply = model
        .complete(vec![Message::user("entelechy 를 정의해.")], &ExecutionContext::new())
        .await?;
    println!("{reply:?}");
    Ok(())
}
```

---

## 주요 기능

### LLM 호출 — 5 codec, 단일 IR

```rust
// 같은 `ModelRequest` IR 가 모든 codec 통과.
// 모든 변환 손실 필드는 `LossyEncode` 로 노출.
let model = ChatModel::new(AnthropicMessagesCodec, transport, "claude-opus-4-7")
    .with_validation_retries(2);                     // schema-mismatch 재시도 예산

let reply: Translation = model
    .complete_typed::<Translation>(messages, &ctx)   // 타입 구조화 출력
    .await?;
```

### LCEL 합성

```rust
use entelix::{ChatPromptTemplate, JsonOutputParser, RunnableExt};

let chain = prompt.pipe(model).pipe(parser);       // 컴파일 타임 I/O 검사
let result = chain.invoke(input, &ctx).await?;
```

### StateGraph control flow (LangGraph parity)

```rust
use entelix::{Annotated, Append, Max, StateGraph, StateMerge, RunnableLambda};

#[derive(Clone, Default, StateMerge)]
struct AgentState {
    log: Annotated<Vec<String>, Append<String>>,    // 노드 간 누적
    score: Annotated<i32, Max<i32>>,                // branch 간 best-of
    last_phase: String,                             // last-write-wins
}

let graph = StateGraph::<AgentState>::new()
    .add_contributing_node("plan", plan)            // 타입 delta + 필드별 merge
    .add_send_edges("plan", ["a", "b", "c"], scatter, "score")  // 병렬 fan-out
    .add_conditional_edges("score", router, [("plan", "plan"), ("done", entelix::END)])
    .set_entry_point("plan")
    .with_checkpointer(Arc::new(postgres_checkpointer))         // crash 후 resume
    .compile()?;
```

### Auto-compaction (Claude Agent SDK parity)

```rust
use entelix::{HeadDropCompactor, MessageRunnableCompactionExt, SummaryCompactor};

// context 가 임계 근접 시 oldest turn drop.
let model = my_model.with_compaction(Arc::new(HeadDropCompactor), 8_192);

// 또는 LLM 요약 압축 (LangChain SummarizationMiddleware parity).
let summariser = SummaryCompactor::new(Arc::new(summary_model));
let model = my_model.with_compaction(Arc::new(summariser), 8_192);
```

### 프로덕션 HTTP 서버

```rust
use entelix::{AgentRouterBuilder, SERVER_DEFAULT_TENANT_HEADER};

let router = AgentRouterBuilder::new(agent)
    .with_checkpointer(Arc::clone(&postgres_checkpointer))
    .with_tenant_header(SERVER_DEFAULT_TENANT_HEADER)   // 멀티테넌트 strict 모드
    .build()?;

let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
axum::serve(listener, router).await?;
```

엔드포인트: `POST /v1/threads/{thread_id}/runs`, `GET /v1/threads/{thread_id}/stream`, `POST /v1/threads/{thread_id}/wake`, `GET /v1/health`.

### Tools — built-in + `#[tool]` 매크로

```rust
use entelix::tool;

#[tool]
/// 두 수를 더합니다.                              // 첫 단락 → LLM-facing description
async fn add(_ctx: &AgentContext<()>, a: i64, b: i64) -> Result<i64> {
    Ok(a + b)
}
// `Add` unit struct + `SchemaTool` impl + JSON Schema 자동 생성.
```

Built-ins: `HttpFetchTool` (3-layer SSRF 방어), `Calculator`, `SchemaTool` 타입 I/O 어댑터. sandboxed shell / file / code / list-dir 도구는 `entelix-tools-coding` 컴패니언 crate 가 제공하고 모든 syscall 은 `Sandbox` trait 통해 delegate.

### Memory 패턴

`BufferMemory` (sliding window), `SummaryMemory` (rolling LLM 요약), `ConsolidatingBufferMemory` (size 임계 자동 summary), `EntityMemory` (엔티티 키 기반 fact), `EpisodicMemory<V>` (시간순 episode), `SemanticMemory<E, V>` (벡터 retrieval), `GraphMemory<N, E>` (타입 노드 + 타임스탬프 edge, BFS traversal + shortest-path).

### MCP — 3 server-initiated 채널 모두

```rust
use entelix::{ChatModelSamplingProvider, McpServerConfig, StaticRootsProvider};

let server = McpServerConfig::http("https://server.example/mcp")
    .with_roots_provider(Arc::new(StaticRootsProvider::new(roots)))
    .with_sampling_provider(Arc::new(ChatModelSamplingProvider::new(chat)));
```

---

## 다른 SDK 와의 차이

| 기능 | LangChain | LangGraph | Claude Agent SDK | pydantic-ai | **entelix** |
|---|---|---|---|---|---|
| 언어 | Python | Python | TypeScript | Python | **Rust** |
| 멀티테넌트 일급 | ✗ | △ | ✗ | ✗ | **✓** |
| Postgres FORCE RLS | ✗ | ✗ | ✗ | ✗ | **✓** |
| Graph memory (타입 노드/엣지 + BFS) | ✗ | ✗ | ✗ | ✗ | **✓** |
| Auto-compaction (drop + LLM-summary) | △ | △ | ✓ | ✗ | **✓** |
| 타입 구조화 출력 + retry budget | ✓ | ✓ | △ | ✓ | **✓** |
| Sealed LLM/operator 채널 캐리어 | ✗ | ✗ | ✗ | ✗ | **✓** |
| OTel GenAI semconv 네이티브 | △ | △ | △ | ✓ | **✓** |
| Typed audit sink (`record_*` verbs) | ✗ | ✗ | △ | ✗ | **✓** |
| Run budget axes (USD cost 포함) | ✗ | △ | ✗ | ✓ (5-axis, USD 없음) | **✓ (6-axis incl. USD)** |
| Vendor 확장 escape-hatch (`*Ext`) | △ | △ | ✗ | ✓ | **✓** |
| 분산 session lock (Postgres + Redis) | ✗ | ✗ | ✗ | ✗ | **✓** |
| MCP 3 server-initiated 채널 모두 | △ | △ | ✓ | △ | **✓** |
| CI 강제 아키텍처 invariants | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## entelix 가 하지 않는 것

- **first-party crate 에서 직접 filesystem / shell 호출 없음** — sandboxed wrapper (`SandboxedShellTool`, `SandboxedReadFileTool`, …) 는 `entelix-tools-coding` 컴패니언 crate 가 제공하고 모든 syscall 은 `Sandbox` trait 통해 delegate. 구체 `Sandbox` impl (Landlock / Seatbelt / e2b / modal) 은 컴패니언 crate, core 에 없음.
- **로컬 추론 없음** — application layer SDK; 필요 시 `candle` / `mistral.rs` 와 페어링.
- **벡터 DB 재구현 없음** — 프로덕션 `VectorStore` impl 은 `entelix-memory-qdrant` / `entelix-memory-pgvector` 컴패니언으로 ship; trait 통해 BYO.
- **document loader 없음** — `swiftide` 영역.
- **Python interop 없음** — Rust 우선.

---

## Workspace

```
entelix                  — 파사드 (feature flag 기반 re-export)
entelix-core             — IR, Codec, Transport, Tool, Auth, ChatModel + tower::Service spine
entelix-runnable         — Runnable trait + LCEL .pipe() + Sequence/Parallel/Router/Lambda
entelix-prompt           — PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShot
entelix-graph            — StateGraph, Reducer, StateMerge trait, Dispatch, Checkpointer, interrupts
entelix-graph-derive     — proc-macro: #[derive(StateMerge)] 가 Contribution + builder + impl emit
entelix-tool-derive      — proc-macro: #[tool] 이 async fn 시그니처에서 SchemaTool impl 생성
entelix-session          — SessionGraph 이벤트 로그 + Compactor + SessionAuditSink, fork, archival watermark
entelix-memory           — Store + Embedder/Retriever/EmbeddingRetriever/VectorStore/GraphMemory trait + memory pattern
entelix-memory-openai    — OpenAI Embeddings 구체 Embedder (컴패니언)
entelix-memory-qdrant    — qdrant gRPC 구체 VectorStore (컴패니언)
entelix-memory-pgvector  — Postgres + pgvector 구체 VectorStore + row-level security (컴패니언)
entelix-graphmemory-pg   — Postgres 구체 GraphMemory + WITH RECURSIVE BFS + UNNEST bulk insert (컴패니언)
entelix-rag              — RAG 프리미티브 (Document/Lineage, splitter, Chunker, IngestionPipeline) + corrective-RAG (CRAG) 레시피
entelix-persistence      — Postgres + Redis Checkpointer/Store/SessionLog + row-level security + advisory lock
entelix-tokenizer-tiktoken — tiktoken-rs 래핑 vendor-accurate TokenCounter (OpenAI BPE: cl100k_base / o200k_base / p50k_base / r50k_base)
entelix-tokenizer-hf     — HuggingFace `tokenizers` 래핑 vendor-accurate TokenCounter (Llama / Qwen / Mistral / DeepSeek / Gemma / Phi)
entelix-tools            — HttpFetchTool, Calculator, SchemaTool, skill, memory tool
entelix-tools-coding     — Sandbox trait 기반 shell / code / fs 도구 + Anthropic Skills 레이아웃 (수직 컴패니언)
entelix-mcp              — 네이티브 JSON-RPC 2.0 over MCP streamable-http; Roots + Elicitation + Sampling 채널; ChatModelSamplingProvider 가 `chatmodel-sampling` feature 뒤
entelix-cloud            — Bedrock (SigV4) / Vertex (gcp_auth) / Foundry (AAD) transport
entelix-policy           — TenantPolicy, RateLimiter, PiiRedactor, CostMeter, QuotaLimiter, PolicyLayer
entelix-otel             — OpenTelemetry GenAI semconv tower::Layer + cache token telemetry + agent root span
entelix-server           — axum HTTP + 5-mode SSE + tenant middleware
entelix-auth-claude-code — Claude.ai OAuth credential provider (Claude Code CLI 공유 저장소)
entelix-agents           — ReAct, Supervisor, Hierarchical, Chat 레시피 + Subagent
```

`entelix-core` 는 다른 entelix crate 의존하지 않음. DAG 가 workspace 차원에서 강제.

파사드 `entelix` crate 가 optional sub-crate 들을 feature flag 뒤에 gate — 안 쓰는 layer 비용 안 냄. canonical list 는 [`crates/entelix/Cargo.toml`](crates/entelix/Cargo.toml); `full` 이 모든 feature 활성화.

---

## 예제

[`crates/entelix/examples/`](crates/entelix/examples/) 아래 working example — quickstart 부터 end-to-end production workflow까지. LCEL 합성, StateGraph control flow (`16_state_merge_pipeline` 가 `derive(StateMerge)` + `add_contributing_node` + `add_send_edges` end-to-end 시연), HITL graph interrupt (`04_hitl`) 와 HITL tool-dispatch approval pause-and-resume (`18_tool_approval`), memory, multi-agent supervisor / hierarchical 레시피, 모든 streaming mode, 모든 codec × transport pair, MCP per-tenant 격리, MCP sampling via `ChatModelSamplingProvider` (`17_mcp_sampling_provider`), auto-compaction (`25_auto_compaction`), axum `AgentRouterBuilder`.

---

## 영감

- **Anthropic [Managed Agents](https://www.anthropic.com/engineering/managed-agents)** — Session / Harness / Hand 분리, cattle-not-pets, lazy provisioning
- **LangChain LCEL** — `Runnable` + `.pipe()` 합성 primitive
- **LangGraph** — typed state-graph control flow + `Annotated[T, reducer]` + Checkpointer + HITL
- **OpenTelemetry [GenAI semconv](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai)** — vendor-neutral observability 어휘

---

## 읽는 순서

1. [`CLAUDE.md`](CLAUDE.md) — invariant, lock 순서, 에러 컨벤션, managed-agent 구조
2. [`docs/architecture/principles.md`](docs/architecture/principles.md) — living design contract
3. crate 별 `crates/<name>/CLAUDE.md` — surface, crate-local 규칙, forbidden 패턴
4. [`docs/public-api/`](docs/public-api/) — frozen per-crate API baseline (파사드는 의도적 제외)

---

## 지원

- [GitHub Issues](https://github.com/junyeong-ai/entelix/issues)
- [아키텍처 원칙](docs/architecture/principles.md)
- [Provider Capabilities Matrix](docs/architecture/provider-capabilities.md)

---

## 라이선스

MIT.

---

<div align="center">

**[English](README.md)** | **한국어**

Made with Rust 🦀

</div>
