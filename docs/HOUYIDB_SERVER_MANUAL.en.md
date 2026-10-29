# HouyiDB Server Manual

> [中文](HOUYIDB_SERVER_MANUAL.md)

> A single-node, memory-first vector database — server manual.  
> Supports both the pgwire / HTTP protocols and multiple switchable index backends; for
> production deployment, authentication, TLS, encryption at rest, and primary/standby
> configuration, see `docs/DEPLOYMENT_GUIDE.en.md`.

---

## 1. Quick Start

### 1.1 Build

```bash
cargo build --release -p houyi-server
```

### 1.2 Start the Service

```bash
./target/release/houyi-server \
  --benchmark-http-addr 127.0.0.1:7000 \
  --pgwire-addr 127.0.0.1:5432 \
  --vector-dim 1536 \
  --worker-threshold 256 \
  --flush-policy hybrid \
  --index-type hnsw
```

### 1.3 Health Check

```bash
curl http://127.0.0.1:7000/v1/health
# → {"status":"ok"}
```

---

## 2. CLI Parameters

| Parameter | Environment variable | Default | Description |
|------|---------|--------|------|
| `--benchmark-http-addr` | `HOUYI_SERVER_BENCHMARK_HTTP_ADDR` | `127.0.0.1:7000` | HTTP API listen address |
| `--pgwire-addr` | `HOUYI_SERVER_PGWIRE_ADDR` | `127.0.0.1:5432` | pgwire listen address |
| `--vector-dim` | `HOUYI_SERVER_VECTOR_DIM` | `1536` | Vector dimension |
| `--worker-threshold` | `HOUYI_SERVER_WORKER_THRESHOLD` | `256` | Builder trigger threshold (vector count) |
| `--worker-tick-ms` | `HOUYI_SERVER_WORKER_TICK_MS` | `20` | Builder polling interval (ms) |
| `--flush-policy` | `HOUYI_SERVER_FLUSH_POLICY` | `hybrid` | `threshold` / `interval` / `hybrid` |
| `--index-type` | `HOUYI_SERVER_INDEX_TYPE` | `hnsw` | `hnsw` / `clustered` / `flat-clustered` / `flat` / `pq-flat` / `hii-l0` |
| `--hnsw-ef-search` | `HOUYI_SERVER_HNSW_EF_SEARCH` | Auto-inferred | Override HNSW ef_search |
| `--metrics-interval-secs` | `HOUYI_SERVER_METRICS_INTERVAL_SECS` | `1` | Metrics output interval |
| `--wal-path` | `HOUYI_SERVER_WAL_PATH` | Temp directory | WAL file path |
| `--wal-write-mode` | `HOUYI_SERVER_WAL_WRITE_MODE` | `async` | `async` / `sync` |
| `--openai-upstream` | `HOUYI_OPENAI_UPSTREAM` | Not configured | OpenAI-compatible upstream base_url (enables the `/v1/embeddings` gateway) |
| `--openai-timeout-secs` | `HOUYI_OPENAI_TIMEOUT_SECS` | `30` | Timeout for gateway calls to the upstream (seconds) |
| — | `HOUYI_OPENAI_API_KEY` | Empty | Upstream API key (**ENV only**, no CLI — command lines are visible to `ps` and therefore leak) |

### 2.1 HNSW-Specific Environment Variables

| Variable | Default | Description |
|------|------|------|
| `HOUYI_HNSW_PARALLEL_BATCH_BUILD` | `1` (server) | Enable Rayon parallel graph construction |
| `HOUYI_HNSW_MAX_BUILD_BATCH` | `8192` (server) | Maximum number of vectors per build batch |
| `HOUYI_HNSW_SEED_SERIAL_POINTS` | `256` | Serial insertion threshold for the small-graph phase |
| `HOUYI_VECTOR_AUTO_HNSW` | `1` | Automatically infer M / ef parameters |
| `HOUYI_VECTOR_RECALL_OPTIMIZED` | `0` | High-recall mode (raises ef_c) |
| `HOUYI_VECTOR_ESTIMATED_VECTORS` | Auto | Estimated vector count (affects parameter inference) |

---

## 3. Index Backends

| Index type | `--index-type` | Build time | QPS | Recall | Use case |
|------|:---:|:---:|:---:|:---:|------|
| **HNSW** | `hnsw` | Slow | Medium | **Highest (0.9995)** | High recall, offline indexing |
| **ClusteredHnsw** | `clustered` | **Fast (8x)** | On par with HNSW | 0.995 | Write-heavy, read-light; fast rebuilds |
| **FlatClustered** | `flat-clustered` | **Very fast** | **Highest (231)** | 0.998 | Brute-force retrieval on small clusters |
| **Flat** | `flat` | **0** | Low (14) | **1.0** | N < 5K, exact retrieval |
| **PQ-Flat** | `pq-flat` | Fast | High | Lossy | Memory-constrained; distance is L2 |
| **HII L0** | `hii-l0` | Fast | Medium | Small-graph ANN | Real-time writes + small-graph retrieval (L1/L2 not implemented) |

> Benchmark data: OpenAI 50K × 1536D, COSINE, Apple Silicon 8 cores (HNSW/Clustered/
> FlatClustered/Flat); PQ-Flat and HII L0 are not included in that benchmark table.

---

## 4. pgwire Protocol (Port 5432)

### 4.1 Connection

```bash
# Use a standard PostgreSQL client
psql -h 127.0.0.1 -p 5432 -U tester -d houyi
```

### 4.2 Vector Retrieval

```sql
-- k=10, no filter
/* houyi:ef_search=256 */ /* houyi:vector=0.1,0.2,... */ SELECT id FROM docs LIMIT 10;

-- k=10, with filter
/* houyi:vector=0.1,0.2,... */ SELECT id FROM docs WHERE id IN (1,2,3) LIMIT 10;
```

### 4.3 Batch Writes

```sql
-- Via Extended Query + BYTEA binary parameters
/* houyi:upsert_batch=$1 */ SELECT 1;
-- Bind: $1 = AUB1 binary payload (dim|count|id+vec[])
```

### 4.4 Client Drivers

```python
# Python (psycopg)
import psycopg
conn = psycopg.connect("host=127.0.0.1 port=5432 user=tester dbname=houyi")
conn.execute("/* houyi:vector=%s */ SELECT id FROM docs LIMIT 10", ("0.1,0.2,...",))
```

```rust
// Rust (tokio-postgres)
use tokio_postgres::connect;
let (client, conn) = connect("host=127.0.0.1 port=5432 user=tester dbname=houyi", NoTls).await?;
let rows = client.query("/* houyi:vector=$1 */ SELECT id FROM docs LIMIT 10", &[&"0.1,0.2,..."]).await?;
```

---

## 5. HTTP API (Port 7000)

### 5.1 Health Check

```bash
GET /v1/health
# → {"status":"ok"}
```

### 5.2 Batch Writes

```bash
POST /v1/vectors/upsert
Content-Type: application/json

{
  "vectors": [
    {"id": 1, "values": [0.1, 0.2, ...]},
    {"id": 2, "values": [0.3, 0.4, ...]}
  ],
  "index": {
    "m": 24,
    "ef_construction": 400
  }
}
# → {"inserted": 2}
```

### 5.3 Vector Search

```bash
POST /v1/vectors/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, ...],
  "k": 10,
  "ef_search": 256
}
# → {"ids": [1, 3, 7, ...]}
```

### 5.4 Index Build / Optimize

```bash
POST /v1/optimize
# → {"flushed_items": 50000, "pending_items": 0, ...}
```

### 5.5 Index Status

```bash
GET /v1/index/status
# → {"indexed_count": 50000, "pending_count": 0, "hot_count": 0,
#     "hnsw_m": 24, "hnsw_ef_construction": 400, "hnsw_ef_search": 256, ...}
```

### 5.6 WAL Management

```bash
GET  /v1/wal/status     # WAL 状态 (next_lsn, applied_lsn, archived_segments)
POST /v1/wal/checkpoint  # {"through_lsn": 42} → 截断 WAL
POST /v1/wal/rotate       # 归档当前 WAL
POST /v1/wal/snapshot     # 创建向量快照
```

### 5.7 pgwire Metrics

```bash
GET /v1/pgwire/status
# → {"handshake_count": 5, "simple_query_count": 100, ...}
```

### 5.8 OpenAI-Compatible Gateway (/v1/embeddings)

When started with `--openai-upstream`, HouyiDB exposes an OpenAI-shaped embeddings endpoint.
The core has **no** embedding inference capability — the gateway forwards requests to the
configured upstream (OpenAI / Ollama / vLLM or any compatible endpoint) and does not compute
vectors itself.

**Pure proxy mode** (without the `collection` field): the request body is forwarded as-is
(unknown fields are not dropped), and the upstream response (including error status codes and
body) is returned as-is — "swap base_url from OpenAI to HouyiDB" and it works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7000/v1", api_key="not-needed")
resp = client.embeddings.create(model="text-embedding-3-small", input="hello")
```

**Auto-ingest mode** (with the `collection` extension field, passed by OpenAI SDKs via
`extra_body`): the gateway opportunistically upserts the embedding returned by the upstream
into that collection — "change the URL and data is ingested":

```python
resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["apple is a fruit", "banana is yellow"],
    extra_body={"collection": "docs"},
)
print(resp.houyi)  # {"collection": "docs", "upserted": 2}
```

- Ingest id = `stable_id(text)` (blake2b-8, the same scheme as the Python SDK; idempotent on
  re-runs, and deletions are replayable across SDKs).
- The original text goes into the payload `{"text": ...}` (consistent with the text_key
  convention of the langchain / llama-index adapters).
- The `houyi` extension field is injected at the top level of the response; OpenAI clients
  ignore unknown fields, so it is safe.
- Auto-ingest requires input to be a string (array); token arrays cannot reconstruct the
  original text and explicitly return 400.
- When `--openai-upstream` is not configured, the endpoint returns 501.

---

## 6. VectorDBBench Standard Test

### 6.1 Environment Preparation

```bash
cd "$VDB_ROOT"        # 含 houyidb client 适配器的 VectorDBBench 工作副本
source .venv/bin/activate
```

### 6.2 Run the Benchmark

```bash
# Start the server
./target/release/houyi-server \
  --benchmark-http-addr 127.0.0.1:17000 \
  --pgwire-addr 127.0.0.1:15432 \
  --vector-dim 1536 \
  --index-type hnsw \
  --worker-threshold 256 \
  --flush-policy hybrid

# VDB benchmark
python3 -m vectordb_bench.cli.vectordbbench houyidb \
  --host 127.0.0.1 --port 15432 --http-port 17000 \
  --transport pgwire \
  --case-type Performance1536D50K \
  --k 10 --m 24 --ef-construction 400 --ef-search 256 \
  --num-concurrency 1 \
  --task-label "my-benchmark"
```

### 6.3 Interpreting Results

| Metric | Meaning |
|------|------|
| `load_duration` | Total load time (insert + optimize) |
| `insert_duration` | Pure insert time |
| `optimize_duration` | Index construction time |
| `qps` | Queries per second |
| `serial_latency_p99` | P99 latency (seconds) |
| `recall` | Recall@k |
| `ndcg` | NDCG@k |

---

## 7. Full Test Commands

```bash
# Full unit test suite
cargo test --all

# Single-crate tests
cargo test -p houyi-core
cargo test -p houyi-vector
cargo test -p houyi-query
cargo test -p houyi-persistence
cargo test -p houyi-server

# Performance smoke (skip recall)
./scripts/perf_smoke_fast.sh

# Full smoke (including recall)
./scripts/perf_smoke.sh

# VectorDBBench comparison
python3 benchmarks/vectordb_bench_houyidb/install.py --vdb-root "$VDB_ROOT"
VDB_ROOT="$VDB_ROOT" ./benchmarks/vectordb_bench_houyidb/run_bench.sh
```

---

## 8. Benchmark Data

### 8.1 50K 1536D (OpenAI, COSINE, k=10)

| Index | Load | QPS | P99 | Recall@10 |
|------|:---:|:---:|:---:|:---:|
| HNSW | 64.6s | 182 | 6.6ms | 0.9995 |
| ClusteredHnsw | 17.6s | 179 | 6.8ms | 0.9948 |
| FlatClustered | 17.0s | **231** | **5.1ms** | 0.9978 |
| Flat | **9.5s** | 14 | 71.9ms | **1.0000** |

### 8.2 1M 768D (Cohere, COSINE, k=100)

| Index | Load | QPS | P99 | Recall@100 |
|------|:---:|:---:|:---:|:---:|
| HNSW | 1050s | 176 | 6.8ms | 0.969 |
| ClusteredHnsw | 480s | 70 | 16.4ms | 0.961 |

---

## 9. Deployment Recommendations

| Scenario | Recommended index |
|------|------|
| High recall (≥ 0.999) | `hnsw` |
| Fast loading + stable QPS | `clustered` |
| Extreme QPS + latency | `flat-clustered` |
| N < 5K, exact retrieval | `flat` |
| N > 1M | `hnsw` or `clustered` |

---

## 10. Hybrid Retrieval (Vector + Structured Filtering)

### 10.1 HTTP API

```bash
POST /v1/collections/{name}/hybrid/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, ...],
  "k": 10,
  "filters": [
    {"column": "tag", "op": "eq", "value": "beach"},
    {"column": "date", "op": "gt", "value": "1717200000"},
    {"column": "date", "op": "lt", "value": "1728000000"},
    {"column": "people", "op": "in", "value": ["Alice", "Bob"]}
  ]
}
```

### 10.2 Supported Filter Operations

| Operation | Description | Example |
|:-----|:-----|:-----|
| `eq` | Equal | `{"column":"tag","op":"eq","value":"beach"}` |
| `gt` | Greater than | `{"column":"date","op":"gt","value":"1717200000"}` |
| `lt` | Less than | `{"column":"date","op":"lt","value":"1728000000"}` |
| `gte` | Greater than or equal | `{"column":"score","op":"gte","value":"0.8"}` |
| `lte` | Less than or equal | `{"column":"score","op":"lte","value":"0.5"}` |
| `in` | Set membership | `{"column":"people","op":"in","value":["Alice","Bob"]}` |

### 10.3 Predicate Tree (PAX Layer)

```rust
ScalarPredicate::And(
    Box::new(ScalarPredicate::Gt(ScalarValue::Timestamp(1717200000))),
    Box::new(ScalarPredicate::Lt(ScalarValue::Timestamp(1728000000))),
)
```

- `ScalarValue`: Int64 / Float64 / Bool / Text / Timestamp / JsonText
- `ScalarPredicate`: Eq / Gt / Lt / GtEq / LtEq / In / And / Or
- `filter_multi_column()`: multi-column AND composition
