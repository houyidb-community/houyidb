# HouyiDB v0.2.0-beta.5

> [中文](README.md)

<p align="center"><img src="docs/assets/logo-h-en-red.png" alt="HouyiDB" width="380"></p>

<p align="center"><a href="https://artifacthub.io/packages/helm/houyidb/houyidb"><img src="https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/houyidb" alt="Artifact Hub"></a></p>

A single-node, memory-first vector database: HNSW vector search + PAX row-column hybrid storage + pg_wire/HTTP dual interfaces + WAL persistence.

![HouyiDB 60-second demo: upsert, search, filter, kill -9, restart recovery](demo/houyidb-demo.gif)

> First time here? See [`docs/USER_MANUAL.en.md`](docs/USER_MANUAL.en.md) (30 minutes from setup to restart recovery).

## Quick Start

### 0) Docker (simplest)

```bash
docker compose up -d        # 或：docker build -t houyidb . && docker run -d -p 7000:7000 -p 5432:5432 -v houyidb-data:/data houyidb
curl http://127.0.0.1:7000/v1/health
```

Interactive documentation for the HTTP API: `http://127.0.0.1:7000/v1/openapi.json` (paste it into Swagger UI to browse).

### 1) Start the main server from source

```bash
cargo run -p houyi-server --bin houyi-server
```

Optional parameters (CLI > ENV > defaults):

```bash
cargo run -p houyi-server --bin houyi-server -- \
  --worker-threshold 256 \
  --worker-tick-ms 20 \
  --flush-policy hybrid \
  --metrics-interval-secs 1
```

### 2) Performance smoke tests and benchmarks

```bash
./scripts/perf_smoke_fast.sh    # 快速（跳过 recall 组）
./scripts/perf_smoke.sh         # 完整（含 recall_check_hnsw_after_flush）

# Recall gate (L2 ground truth, default mean@k ≥ 0.99)
cargo run -p houyi-vector --example recall_gate --release
```

For the reproducible VectorDBBench protocol, see `VDBBENCH_PROTOCOL.en.md` and
`benchmarks/vectordb_bench_houyidb/`. Public reports in this repository
contain HouyiDB self-measurements only; no third-party comparison results are
published.

HNSW connectivity density: the server defaults to **2×M** at level 0 (recall first, compatible with existing graph snapshots); for high-QPS scenarios explicitly set `HOUYI_HNSW_LAYER0_MULT=1` (the VDB campaign defaults to 1×). 1× improves QPS by roughly 35–60% over 2×, at about −0.9pp recall (Cohere). Measured comparisons of both modes are in `docs/benchmarks/houyidb-performance-2026-10.en.md`.

HNSW graph merging defaults to "plan under read lock + commit under short write lock": searches can concurrently read the graph during construction; in the streaming 50% phase, serial query wall time dropped from 104s to 3.9s and p99 from 1.72s to 21ms, with mature recall unchanged. Set `HOUYI_HNSW_PLAN_APPLY=0` to fall back to the old path. See the streaming section of `docs/benchmarks/houyidb-performance-2026-10.en.md`.

## Current Capabilities

- `houyi-core`: 64-byte aligned page allocation, PAX page abstraction, lock-free message bus
- `houyi-vector`: HNSW dual-speed writes (HotBuffer + Async-Builder), SIMD distance computation (NEON/AVX feature-gated), Clustered/PQ/HII-L0 backends
- `houyi-query`: pg_wire protocol (text/binary), SQL hybrid search, predicate-pushdown bitset pruning
- `houyi-persistence`: WAL (async writer thread + checkpoint/rotate), disk snapshots, cold-start recovery
- `houyi-server`: HTTP API (collection upsert with payload, hybrid search with scores), pg_wire connection handling, metrics output
- `houyi-mobile`: lightweight C ABI index (iOS/Android), see `docs/HOUYIDB_MOBILE_MANUAL.en.md`

## Client SDKs (`sdk/`)

- **Python**: `sdk/python/houyidb` + LangChain / LlamaIndex adapters
- **TypeScript**: `sdk/typescript/houyidb` (bigint handling for ids > 2^53)
- **Go**: `sdk/go/houyidb`
- **Rust**: `sdk/rust/houyidb-client`
- **.NET**: `sdk/dotnet` (net8.0 + net10.0)
- **Java**: `sdk/java/houyidb` (17+)

The six-language `stable_id` (BLAKE2b-8 big-endian & 2^63-1) is cross-checked against the server gateway: the same string id yields the same u64 in any language, and deletions can be replayed across SDKs.

**MCP**: `mcp/houyidb-mcp` provides a stdio MCP server (collection listing, search, hybrid search, opt-in writes) for Claude Code, Cursor, Cline, and other MCP clients.

## Documentation

- User guide: [`docs/USER_MANUAL.en.md`](docs/USER_MANUAL.en.md)
- Server manual: [`docs/HOUYIDB_SERVER_MANUAL.en.md`](docs/HOUYIDB_SERVER_MANUAL.en.md)
- Deployment guide: [`docs/DEPLOYMENT_GUIDE.en.md`](docs/DEPLOYMENT_GUIDE.en.md)
- Embeddings (OpenAI-compatible gateway): [`docs/EMBEDDING_GUIDE.en.md`](docs/EMBEDDING_GUIDE.en.md)
- VectorDBBench reproduction protocol: [`VDBBENCH_PROTOCOL.en.md`](VDBBENCH_PROTOCOL.en.md)
- Performance measurements report: [`docs/benchmarks/houyidb-performance-2026-10.en.md`](docs/benchmarks/houyidb-performance-2026-10.en.md)
- Mobile (C ABI): [`docs/HOUYIDB_MOBILE_MANUAL.en.md`](docs/HOUYIDB_MOBILE_MANUAL.en.md)
- MCP server: [`mcp/houyidb-mcp/README.en.md`](mcp/houyidb-mcp/README.en.md)
- SDK release guide: [`docs/SDK_RELEASE.en.md`](docs/SDK_RELEASE.en.md)

## License & Governance

- Community edition: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE);
  licensing notes in [LICENSING.en.md](LICENSING.en.md).
- Contributing: [CONTRIBUTING.en.md](CONTRIBUTING.en.md) (external contributions require a [CLA](CLA.en.md));
  report security vulnerabilities through the private channel in [SECURITY.en.md](SECURITY.en.md);
  trademark policy in [TRADEMARK.en.md](TRADEMARK.en.md).
