# HouyiDB Trial Manual (v0.2.0-beta.8)

> [中文](USER_MANUAL.md)

> For developers new to HouyiDB: go from startup to "write → search → filter →
> delete → restart recovery" within 30 minutes. All commands have been tested on
> a local macOS machine (Apple Silicon native + Intel package self-checked via
> Rosetta 2).
>
> Project positioning: **a single-node, memory-first vector database**. HNSW
> vector search + scalar filtering + dual pgwire/HTTP protocols + WAL
> persistence. Currently in beta; suitable for feature validation and performance
> trials, not recommended for carrying critical production data directly.

---

## 0. What's in the Release Package

Prebuilt packages (macOS; one each for Apple Silicon and Intel):

```
houyidb-0.2.0-beta.8-<date>-darwin-arm64/     # Apple Silicon (M1/M2/M3/M4)
houyidb-0.2.0-beta.8-<date>-darwin-x86_64/    # Intel Mac
├── bin/houyi-server          # 主服务（约 10MB，无外部 C 依赖）
├── scripts/trial_start.sh    # 一键启动（默认 17000/15432）
├── scripts/trial_smoke.sh    # 自检：写入/检索/过滤/删除
├── scripts/embed_smoke.sh    # 向量化链路自检（需配置上游）
├── scripts/mock_embedding_server.py  # 无语义 mock 上游（只验链路）
├── examples/text_ingest.py   # 文本切块 → 自动入库 → 检索示例
├── USER_MANUAL.md            # 本文件
├── README.md  CHANGELOG.md  LICENSE  NOTICE
├── docs/                     # 服务端手册、部署指南、向量化指南
└── sdk/python/               # Python 客户端 + LangChain/LlamaIndex 适配器
```

Source package: `houyidb-0.2.0-beta.8-src.tar.gz` (any platform; includes the
Dockerfile and all SDKs).

> Choose the prebuilt package by CPU architecture (`uname -m`: use the arm64
> package on `arm64` and the x86_64 package on `x86_64`). On Linux/Windows, use
> the source package with `cargo build` or Docker.

---

## 1. Three Ways to Start

### 1.1 Prebuilt Package (macOS, fastest)

Use `darwin-arm64` on Apple Silicon (M series) and `darwin-x86_64` on Intel Macs:

```bash
uname -m        # arm64 → darwin-arm64；x86_64 → darwin-x86_64
tar xzf houyidb-0.2.0-beta.8-*-darwin-<arch>.tar.gz
cd houyidb-0.2.0-beta.8-*-darwin-<arch>

# If the binary was quarantined by Gatekeeper after network transfer, remove the quarantine first (a common issue with unsigned beta builds):
xattr -dr com.apple.quarantine . 2>/dev/null || true

./scripts/trial_start.sh
```

Once you see `HTTP : http://127.0.0.1:17000`, open another terminal:

```bash
./scripts/trial_smoke.sh
```

### 1.2 Docker (any platform; recommended for Linux/Windows/WSL)

After extracting the source package:

```bash
tar xzf houyidb-0.2.0-beta.8-src.tar.gz
cd houyidb-0.2.0-beta.8-src
docker compose up -d --build        # 首次会编译 Rust，约 5–15 分钟
curl http://127.0.0.1:7000/v1/health
```

> **macOS note**: The system AirPlay Receiver occupies ports 5000/7000. If port
> 7000 fails to start, use host port mapping instead (the container still uses
> 7000/5432 internally):
>
> ```bash
> docker run -d --name houyidb -p 17000:7000 -p 15432:5432 \
>   -v houyidb-data:/data houyidb:latest
> curl http://127.0.0.1:17000/v1/health
> ```
>
> Subsequent commands in this manual assume HTTP at `http://127.0.0.1:17000`;
> with Docker's default mapping, replace 17000 with 7000.

### 1.3 Build from Source (requires Rust 1.87+)

```bash
tar xzf houyidb-0.2.0-beta.8-src.tar.gz
cd houyidb-0.2.0-beta.8-src
cargo build --release -p houyi-server
./scripts/trial_start.sh
```

Verify:

```bash
curl -sS --noproxy '*' http://127.0.0.1:17000/v1/health
# {"status":"ok"}
```

If the ports are in use, override them with environment variables:

```bash
HOUYI_TRIAL_HTTP_ADDR=127.0.0.1:18000 \
HOUYI_TRIAL_PGWIRE_ADDR=127.0.0.1:16432 \
./scripts/trial_start.sh
```

---

## 2. Five-Minute Quick Start (HTTP API)

Data model: a **Collection** (similar to a table; created automatically on the
first write, with dimensionality determined by the first batch of vectors; it
can also be created explicitly with `PUT` to fix dims/metric/backend up front)
→ each Collection holds `id + vector + label + payload`. `id` is `u64`;
string ids can be deterministically hashed to u64 by the SDK.

### 2.1 Write

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/upsert \
  -H 'Content-Type: application/json' -d '{
  "vectors": [
    {"id": 1, "values": [0.92,0.88,0.06,0.05,0.05,0.04,0.05,0.05],
     "label": "fruit", "payload": {"name": "apple", "price": 3}},
    {"id": 2, "values": [0.05,0.04,0.92,0.90,0.06,0.05,0.05,0.04],
     "label": "electronics", "payload": {"name": "laptop", "price": 999}},
    {"id": 3, "values": [0.04,0.05,0.05,0.06,0.90,0.92,0.05,0.04],
     "label": "book", "payload": {"name": "rust-book", "price": 45}}
  ]
}'
# {"inserted":3}
```

- `label` is a reserved top-level field (string) and is used as the filter
  column `label`; a `label` key inside `payload` is rejected (400).
- Single request body limit is 64MB; larger batches yield better throughput.

### 2.1.1 Explicit Collection Creation (Optional)

Implicit creation uses the server's default backend and metric (hnsw → cosine).
When L2 / inner product / L1 or custom HNSW parameters are needed, fix everything
at once with `PUT /v1/collections/{name}` (write permission, no admin token):

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/goods \
  -H 'Content-Type: application/json' -d '{
  "dims": 8,
  "metric": "ip",
  "index_type": "hnsw",
  "hnsw_m": 16, "hnsw_ef_construction": 100, "hnsw_ef_search": 64
}'
# 201 {"created":true,"collection":"goods","dims":8,"metric":"ip",
#      "index_type":"hnsw","hnsw":{"m":16,"ef_construction":100,"ef_search":64}}
```

- `metric`: `l2` / `cosine` / `ip` / `l1`; omitted means the backend default.
  **Immutable after creation**—changing it requires drop-and-recreate.
- `index_type`: `hnsw` / `flat` / `clustered` / `flat-clustered` / `pq-flat` /
  `hii-l0`; omitted means the server default. Support matrix: the first four
  support all four metrics; `pq-flat` supports `l2` only; `hii-l0` supports
  `cosine` only.
- The three HNSW parameters must be supplied together (`m` 1..1024,
  `ef_construction` / `ef_search` 1..65536) and only matter for the hnsw backend.
- An existing name yields 409 (even with identical parameters); invalid
  parameters 400; follower/read-only 503.

### 2.2 Vector Search

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/search \
  -H 'Content-Type: application/json' -d '{
  "vector": [0.90,0.85,0.05,0.05,0.05,0.05,0.05,0.05],
  "k": 3,
  "include_payload": true
}'
# {"ids":[1,3,2],"scores":[...],"payloads":{"1":{...},"3":{...},"2":{...}}}
```

- `scores` has the same length as `ids`; higher means more similar, and the
  conversion depends on the collection's `metric`: `cosine` → `1 -
  cosine_distance` (clamped to `[-1,1]`); `l2` / `l1` → `1 - distance`
  (unclamped); `ip` → the raw dot product (higher is more relevant). The
  response also echoes the `metric` field.
- `ef_search` is optional; higher values increase recall and latency.

### 2.3 Vector + Scalar Filtering (Hybrid Search)

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/hybrid/search \
  -H 'Content-Type: application/json' -d '{
  "vector": [0.90,0.85,0.05,0.05,0.05,0.05,0.05,0.05],
  "k": 3,
  "filters": [{"column": "price", "op": "lt", "value": 10}],
  "include_payload": true
}'
# Only the fruit category remains: {"ids":[1],...}
```

Filter operators: `eq` / `gt` / `lt` / `gte` / `lte` / `in` (`value` for `in`
must be a JSON array). Multiple `filters` are combined with AND. `label` can be
used directly as a column name. **Filtered search is supported only by the
hnsw / flat backends** (other backends return 400).

### 2.4 Delete and Status

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/delete \
  -H 'Content-Type: application/json' -d '{"ids": [2]}'
# {"deleted":1}

curl -sS --noproxy '*' http://127.0.0.1:17000/v1/collections/goods/status
# {"name":"goods","vector_count":2,"pending_count":0,"dims":8,...}
```

Delete also supports `{"filters":[...]}` (delete by condition, unioned with
explicit ids).

### 2.4.1 Dropping a Whole Collection (admin, irreversible)

```bash
# With RBAC enabled use an admin token; omit Authorization when auth is off
curl -sS --noproxy '*' -X DELETE \
  -H "Authorization: Bearer $HOUYI_ADMIN_TOKEN" \
  http://127.0.0.1:17000/v1/admin/collections/goods
# {"deleted":true,"collection":"goods"}
```

- **Irreversible**: unconditional (no empty-collection requirement and no
  second-confirmation flag). Export a backup first with
  `GET /v1/collections/{name}/export`.
- **Permission**: the path lives under `/v1/admin/*` and requires an admin
  principal (`Action::Admin`); the audit log records actor/route/status.
- **Durability and replication**: WAL-first (`collection_drop`), so crash replay
  and followers converge to "gone"; a same-name upsert afterwards creates a
  brand-new empty collection.
- **Status codes**: unknown/already-dropped 404; delete races 409;
  follower/read-only 503.

### 2.5 Other Common Endpoints

| Endpoint | Purpose |
|------|------|
| `GET /v1/health` | Health check (authentication exempt) |
| `GET /readyz` | Readiness check (authentication exempt) |
| `GET /v1/openapi.json` | OpenAPI document (can be pasted into Swagger UI) |
| `GET /v1/collections` | List Collections |
| `PUT /v1/collections/{name}` | Explicitly create a collection (write permission; fixes dims/metric/backend/HNSW in one shot) |
| `GET /v1/collections/{name}/export` | Streaming NDJSON logical export (header/records/footer; backup/migration) |
| `DELETE /v1/admin/collections/{name}` | Drop a whole collection (admin, irreversible, WAL-first) |
| `GET /metrics` | Prometheus metrics (authentication exempt) |
| `GET /v1/index/status` | Primary index (/v1/vectors path) status |
| `POST /v1/optimize` | Drain the build queue / trigger compaction |
| `POST /v1/wal/checkpoint` | WAL checkpoint |
| `POST /v1/vectors/upsert`, `/v1/vectors/search` | Single-primary-index compatibility API (dimensionality fixed by startup argument) |

### 2.6 Text Embedding (optional, OpenAI-compatible gateway)

The core does not perform embedding inference; once an OpenAI-compatible
upstream is configured, `/v1/embeddings` converts text into vectors, and the
`collection` extension field enables **automatic ingestion**.

Ollama (recommended for offline use):

```bash
ollama pull bge-m3
HOUYI_OPENAI_UPSTREAM=http://127.0.0.1:11434 ./scripts/trial_start.sh
# Another terminal:
EMBED_MODEL=bge-m3 ./scripts/embed_smoke.sh
EMBED_MODEL=bge-m3 python3 examples/text_ingest.py
```

OpenAI (requires a key):

```bash
HOUYI_OPENAI_UPSTREAM=https://api.openai.com \
HOUYI_OPENAI_API_KEY=sk-... ./scripts/trial_start.sh
EMBED_MODEL=text-embedding-3-small ./scripts/embed_smoke.sh
```

Automatic ingestion from code:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:17000/v1", api_key="not-needed")
client.embeddings.create(model="bge-m3", input=["苹果是水果"],
                         extra_body={"collection": "docs"})
# resp.houyi = {"collection": "docs", "upserted": 1}
```

- id = the text's `stable_id` (idempotent across reruns); the original text is
  stored in payload `text`;
- **identical texts overwrite each other**; business metadata cannot be
  attached; changing the model requires rebuilding the collection;
- For the complete recipe (Ollama/TEI/OpenAI, model selection, limits, and
  troubleshooting), see `docs/EMBEDDING_GUIDE.md`.

---

## 3. Python SDK

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e sdk/python/houyidb
```

```python
from houyidb import HouyiClient, Vector, Filter

client = HouyiClient("http://127.0.0.1:17000")
col = client.collection("goods")          # 首次 upsert 隐式创建

col.upsert([
    Vector(id=1, values=[0.92, 0.88, 0.06, 0.05, 0.05, 0.04, 0.05, 0.05],
           label="fruit", payload={"name": "apple", "price": 3}),
])

hits = col.search([0.90, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                  3, include_payload=True)
print(hits)

hits = col.hybrid_search(
    [0.90, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], 3,
    filters=[Filter("price", "lt", 10)], include_payload=True)
print(hits)

col.delete([1])
```

- String ids are hashed automatically into stable u64 values (consistent across
  languages), so business primary keys can be used directly.
- LangChain / LlamaIndex adapters live in the same directory:
  `pip install -e sdk/python/langchain-houyidb` /
  `pip install -e sdk/python/llama-index-vector-stores-houyidb`.
  See each package's README for usage (embeddings are computed by the
  framework/caller; HouyiDB only stores vectors).

---

## 4. pgwire (PostgreSQL Protocol)

Any PostgreSQL client can connect directly; no actual database or table creation
is needed. When authentication is disabled, any user/password works.

```bash
psql -h 127.0.0.1 -p 15432 -U trial -d trial
```

```sql
-- Pass the query vector through a comment; k is specified by LIMIT
/* houyi:vector=0.92,0.88,0.06,0.05,0.05,0.04,0.05,0.05 */
SELECT id FROM goods LIMIT 3;

-- With id filtering
/* houyi:vector=0.92,0.88,0.06,0.05,0.05,0.04,0.05,0.05 */
SELECT id FROM goods WHERE id IN (1, 3) LIMIT 3;
```

- The dimensionality of the pgwire primary index is `HOUYI_TRIAL_DIM` at startup
  (default 768).
- Batch writes use Extended Query + the `/* houyi:upsert_batch=$1 */` binary
  protocol; see `docs/HOUYIDB_SERVER_MANUAL.md` in the package for details.
- After enabling `HOUYI_AUTH_TOKEN`, the pgwire password equals that token.

---

## 5. Persistence, Restart, and Recovery

- **WAL-first**: every upsert/delete is written to disk before memory is
  modified; after the process is killed with kill -9 and restarted, acknowledged
  writes are replayed and recovered.
- Collection data is stored in `data/` (the trial script defaults to
  `./data/houyidb.wal` and `houyidb.collections.wal`). Stopping and restarting:

```bash
# Graceful stop with Ctrl-C
./scripts/trial_start.sh          # 再次启动，自动重放 WAL
```

- Periodic graph snapshots occur every 600s by default; a graph snapshot is also
  written on shutdown (`HOUYI_SHUTDOWN_SNAPSHOT=0` disables it).
- **Resetting trial data** (data will be lost): stop the service, then
  `rm -rf data/`.
- The collections WAL uses **bounded retention** (P5.0): by default the oldest
  whole segments are removed beyond 16 × 64 MiB, starting from the manifest's
  safe checkpoint; followers pin the floor through their durable LSN. With a
  clustered / flat-clustered / pq-flat collection present, retention stops at
  the last floor and reports `blocked_reason` in `/v1/wal/status`.

---

## 6. Configuration and Performance Profiles

### 6.1 Common Startup Arguments

| Argument / Environment Variable | Default | Description |
|---|---|---|
| `--benchmark-http-addr` / `HOUYI_SERVER_BENCHMARK_HTTP_ADDR` | `127.0.0.1:7000` | HTTP listen address |
| `--pgwire-addr` / `HOUYI_SERVER_PGWIRE_ADDR` | `127.0.0.1:5432` | pgwire listen address |
| `--vector-dim` / `HOUYI_SERVER_VECTOR_DIM` | `1536` | Dimensionality of `/v1/vectors` and the pgwire primary index |
| `--index-type` / `HOUYI_SERVER_INDEX_TYPE` | `hnsw` | `hnsw`/`clustered`/`flat-clustered`/`flat`/`pq-flat`/`hii-l0` |
| `--wal-path` / `HOUYI_SERVER_WAL_PATH` | Temporary directory | WAL file path |
| `--wal-write-mode` / `HOUYI_SERVER_WAL_WRITE_MODE` | `async` | `async` / `sync` (sync is more durable and slower) |
| `--snapshot-interval-secs` | `600` | Periodic graph snapshots; 0=disabled |
| `--max-http-inflight` | `512` | Maximum in-flight requests; exceeding the limit returns 503 |
| `--config <toml>` | None | TOML configuration (RBAC, etc.) |

### 6.2 HNSW Knobs (choose by scenario)

| Environment Variable | Default | Description |
|---|---|---|
| `HOUYI_HNSW_LAYER0_MULT` | `2` | L0 connection multiplier. `1` = high QPS (about +35–60%), recall about −0.9pp; `2` = recall first |
| `HOUYI_HNSW_UPPER_MULT` | `0.5` | Upper-layer degree multiplier |
| `HOUYI_HNSW_PLAN_APPLY` | On | Read-lock planning + short write-lock commit; `0`/`false`/`off`/`no`/empty falls back to the old whole-graph locking path (about 11% faster bulk loading, slower streaming search) |
| `HOUYI_HNSW_PARALLEL_BATCH_BUILD` | Server-side `1` | Rayon parallel graph construction |
| `HOUYI_HNSW_MAX_BUILD_BATCH` | Server-side `8192` | Maximum inserts per build batch |
| `HOUYI_HNSW_KEEP_PRUNED` | `0` | keepPruned links (slightly higher recall, slower build/query; not recommended for large collections) |

> Recommendation: start with the defaults. For QPS, set
> `HOUYI_HNSW_LAYER0_MULT=1`; for recall, keep the default 2×. Do not switch
> between the two directly on the same old graph snapshot (snapshot
> compatibility triggers a rebuild).

### 6.3 Filtering

`label` uses an exact inverted index; when the number of matches is large, a
contiguous per-label vector cache is built automatically, and filtered search
degrades to a sequential SIMD scan. No configuration is needed; the cache is
invalidated and rebuilt automatically after writes.

---

## 7. Observability

```bash
curl -sS --noproxy '*' http://127.0.0.1:17000/metrics | head -n 40
curl -sS --noproxy '*' http://127.0.0.1:17000/v1/index/status
curl -sS --noproxy '*' http://127.0.0.1:17000/v1/collections/goods/status
```

`/metrics` is in Prometheus text format and can be connected directly to
Grafana. Server logs go to stdout/stderr (Rust `tracing`; adjustable via
`RUST_LOG=info`).

---

## 8. Security (essential reading for LAN trials)

By default the server listens only on `127.0.0.1`, so local trials need no
authentication. **To let friends or other machines access it**:

```bash
HOUYI_AUTH_TOKEN='请换成长随机串' \
HOUYI_TRIAL_HTTP_ADDR=0.0.0.0:17000 \
./scripts/trial_start.sh
```

- Thereafter, all `/v1/*` (except `/v1/health` and `/readyz`) require
  `Authorization: Bearer <token>` or `X-Api-Key: <token>`.
- For finer-grained permissions, use TOML `[[auth_principals]]` (admin/writer/reader +
  collection scopes); see `docs/HOUYIDB_SERVER_MANUAL.md`.
- For public exposure, add TLS (`--tls-cert` / `--tls-key`) and a reverse proxy;
  do not run an unauthenticated instance bare on the public internet.

---

## 9. Known Limitations (beta)

- Single-node architecture, no sharding; high availability is leader/follower
  with manual promote, with no automatic leader election.
- The collections WAL has bounded retention (default 16 × 64 MiB). A follower
  that falls behind the retained range receives `BEHIND_RETAINED` and must run
  the full resync in `docs/runbooks/collections-retention-resync.md`; it never
  wipes itself. Replication protocol v8 is not rolling-compatible across
  versions, so upgrade all replicas together (see the server manual).
- Scalar pages are append-only: re-upserting the same id appends a new row, and
  both filtering and payload read-back only see the last row per id (stale
  values stop matching as soon as they are overwritten). A check runs every
  10 minutes and compacts when at least 50% of rows are stale, reclaiming
  physical rows without changing query results. clustered / flat-clustered /
  pq-flat do not support snapshot compaction yet.
- Array/object values in payload are treated as JSON text columns, so filtering
  semantics are limited.
- Backend/metric support matrix: hnsw / flat / clustered / flat-clustered
  support l2/cosine/ip/l1; `pq-flat` supports `l2` only (HTTP score is
  `1 − L2²`, unbounded); `hii-l0` supports `cosine` only.
- HII L1 (GPU CAGRA) / L2 (NVMe DiskANN) are not implemented; `simd-avx512` is
  only a skeleton.
- Filtered search is supported only by the hnsw / flat backends.
- Single request body limit is 64MB.

---

## 10. Troubleshooting

| Symptom | Resolution |
|------|------|
| `Address already in use` | Change ports: `HOUYI_TRIAL_HTTP_ADDR=127.0.0.1:18000` (on macOS, 5000/7000 are often occupied by AirPlay) |
| macOS reports "cannot verify the developer" | `xattr -dr com.apple.quarantine .`, or right-click → Open |
| Docker build takes a long time | Normal: Rust is compiled from source inside the container; later builds use the cache |
| Health check fails after startup | Check the server stderr; verify the port and WAL directory are writable |
| Search reports a 400 for dimensionality | The query vector length must equal the dimensionality of the Collection's first write |
| Filtering returns empty | Confirm the backend is hnsw/flat; confirm the column name exists and `op` is valid |
| Want a complete reset | Stop the service, then `rm -rf data/` |

---

## 11. Feedback

If you encounter issues during the trial, please include the following
information:

```bash
uname -a
./bin/houyi-server --help        # 或 cargo run -p houyi-server -- --help
# Reproduction commands, request body, response, and server logs (stdout/stderr)
```

We wish you a smooth trial.
