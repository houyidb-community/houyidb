# HouyiDB Trial Manual (v0.2.0-beta.9)

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
houyidb-0.2.0-beta.9-<date>-darwin-arm64/     # Apple Silicon (M1/M2/M3/M4)
houyidb-0.2.0-beta.9-<date>-darwin-x86_64/    # Intel Mac
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

Source package: `houyidb-0.2.0-beta.9-src.tar.gz` (any platform; includes the
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
tar xzf houyidb-0.2.0-beta.9-*-darwin-<arch>.tar.gz
cd houyidb-0.2.0-beta.9-*-darwin-<arch>

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
tar xzf houyidb-0.2.0-beta.9-src.tar.gz
cd houyidb-0.2.0-beta.9-src
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
tar xzf houyidb-0.2.0-beta.9-src.tar.gz
cd houyidb-0.2.0-beta.9-src
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
- `vector_dtype`: `f32` (default) / `f16`. `f16` halves resident vector memory,
  is hnsw-only, and is **immutable after creation**. Query vectors are always
  computed in f32 (asymmetric distance: f16 data is first rounded onto the f16
  grid). A non-f32 dtype with a non-hnsw backend yields 400. The response and
  `GET .../status` echo the effective `vector_dtype`.
- `quantization`: `none` (default) / `int8` / `rabitq`. It compresses **only
  the HNSW graph-side vector copy**: the main store / hot buffer / builder
  queue always keep the raw (f32 or f16) vectors; search takes candidates from
  the quantized graph (candidate width ×2 for int8, ×16 for rabitq) and then
  reranks exactly on the raw vectors, so the final top-k scores never pass
  through quantization. hnsw-only, **immutable after creation**; a non-hnsw
  backend explicitly carrying `int8` / `rabitq` yields 400; a same-name
  collection with different parameters yields 409. `int8` is 1 B/dim;
  `rabitq` is 1 bit/dim (RaBitQ binary quantization plus a deterministic random
  rotation) and **supports `metric = l2 | cosine` only** (`ip` / `l1` yield
  400); query vectors stay f32 (asymmetric estimation). The response and
  `GET .../status` echo the effective `quantization`.
- Quantization memory ledger (honest; never plan capacity on "halved"): main
  store memory is unchanged (the graph-side `source` shares the main-store
  `Arc`). Graph-side codes = `ceil(D/64)·8` B + 8 B factors, where D is dims
  rounded up to a power of two; at dims=768 (D=1024) each vector costs f32
  3072 B / int8 768 B + 8 B / rabitq 128 B + 8 B. rabitq adds only ~136 B per
  vector over the f32 graph (≈1/24) and drops to ~1/6 of int8; it also needs
  shared rotation signs of 4·D bytes per collection plus one D×f32 scratch
  buffer per query. For cosine the raw and normalized copies coincide, so
  total memory shrinks versus unquantized (int8 ≈ −3 B/dim, rabitq ≈
  −3.8 B/dim); outside cosine it grows slightly (int8 ≈ +1 B/dim, rabitq ≈
  +0.18 B/dim). The win is the in-graph distance working set and bandwidth.
- `sparse`: `none` (default) / `bm25`. `bm25` builds a side inverted index for
  the collection and enables sparse writes plus fusion search; **hnsw / flat
  backends only**, **immutable after creation**, and implicit creation is
  always `none`. The response and `GET .../status` echo the effective `sparse`;
  changing sparse on the same name yields 409, and `bm25` with any other backend
  yields 400. The sparse index never changes dense search results; the main
  store / hot buffer / builder queue still hold only dense vectors.
- An existing name yields 409 (even with identical parameters); invalid
  parameters 400; follower/read-only 503.

f16 example (memory-sensitive workloads that tolerate ~10⁻³ rounding):

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/imgf16 \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "hnsw", "vector_dtype": "f16"
}'
# 201 {"created":true,"collection":"imgf16",...,"vector_dtype":"f16"}
```

int8 graph-side quantization example (raw f32 main store, still reranked exactly):

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/imgq \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "hnsw",
  "vector_dtype": "f32", "quantization": "int8"
}'
# 201 {"created":true,"collection":"imgq",..."quantization":"int8"}
```

rabitq graph-side quantization example (1 bit/dim, l2/cosine only; still reranked exactly):

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/imgq1bit \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "hnsw",
  "quantization": "rabitq"
}'
# 201 {"created":true,"collection":"imgq1bit",..."quantization":"rabitq"}
# metric=ip/l1 with rabitq → 400; rabitq on a non-hnsw backend → 400
```

bm25 sparse-index example (side inverted index enabling sparse writes and fusion search):

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/docs \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "flat", "sparse": "bm25"
}'
# 201 {"created":true,"collection":"docs",...,"sparse":"bm25"}
# bm25 on a non-hnsw/flat backend → 400; implicit creation is always sparse=none
```

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

### 2.3.1 Sparse Vectors (BM25) and Fusion Search

Sparse terms are supplied by the caller (**the core does not tokenize**): each
item is `(term_id, weight)`, where `term_id` is an integer in `1..=2^24-1` and
`weight` is a finite positive number. Writes and queries accept the JSON array
`[[term_id, weight], …]`; duplicate terms are summed by the server and order
does not affect the result. At most 1024 pairs per document and 64 pairs per
query; anything beyond that yields 400.

Write (the collection must already be `sparse=bm25`; a `sparse` key on a
`sparse=none` collection always yields 400 and is never silently dropped):

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/docs/upsert \
  -H 'Content-Type: application/json' -d '{
  "vectors": [
    {"id": 1, "values": [0.9,0.1,0.0,0.0], "label": "fruit",
     "payload": {"text": "apple is a fruit"},
     "sparse": [[7, 1.0], [8, 0.8], [11, 0.5]]},
    {"id": 2, "values": [0.1,0.9,0.0,0.0], "label": "device",
     "payload": {"text": "laptop is a device"},
     "sparse": [[7, 0.6], [9, 1.0]]}
  ]
}'
# {"inserted":2}
```

- An absent / `null` `sparse` key means **unchanged**; **`[]`** clears the
  document's sparse terms.
- Updates are **whole-document replacement**: re-upserting the same id
  overwrites the old terms with the new ones (not an incremental merge), which
  makes replay idempotent.
- Deletes by id / filters remove the sparse terms too, so deleted documents
  become unreachable immediately.

Fusion search (both dense and sparse are required; the collection must be
`sparse=bm25`):

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/docs/fusion/search \
  -H 'Content-Type: application/json' -d '{
  "vector": [0.9,0.1,0.0,0.0],
  "sparse": [[7, 1.0], [9, 0.5]],
  "k": 2,
  "fusion": "rrf",
  "include_payload": true
}'
# {"ids":[1,2],"scores":[...],"dense_scores":[0.99,null],
#  "sparse_scores":[2.1,1.7],"metric":"cosine",
#  "fusion":{"mode":"rrf","rrf_k":60,"alpha":null},"payloads":{...}}
```

- `fusion`: `rrf` (default) or `weighted`. `rrf_k` defaults to 60 and must be
  within `1..=1000`; `alpha` defaults to 0.5 and must be within `[0,1]`
  (weighted only). Explicitly supplied values are validated even when the
  active mode does not use them (invalid values always yield 400).
- **`scores` is a fusion ranking score only and carries no metric semantics**:
  RRF is `Σ 1/(rrf_k + rank)` (rank starts at 1, one term per hit list); weighted
  is `alpha * norm(dense) + (1-alpha) * norm(sparse)`, with each modality
  min-max normalized over the filtered candidates (a single-element list
  normalizes to 1.0, a missing modality counts as 0).
- `dense_scores` follows the collection `metric` (same conversion as 2.2);
  `sparse_scores` is the raw BM25 score. A hit that missed a modality has
  `null` in that position. `metric` explains `dense_scores` only; the `fusion`
  block echoes the effective mode and the parameters actually used.
- `filters` / `ef_search` / `include_payload` mean the same as in normal
  search; filtering is hnsw / flat only and applies to both paths.
- Both paths over-fetch `4×k` candidates, fuse, then cut to `k`; the `k` /
  `ef_search` limits match normal search.
- Error codes: missing collection 404; `sparse=none` collection or empty
  `sparse` 400; dimensionality / term / fusion-parameter violations 400;
  read-only / follower / fencing 503.
- BM25 (`k1=1.2`, `b=0.75`): `idf = ln(1 + (N-df+0.5)/(df+0.5))`, `tf` is
  the client weight, `dl = Σ weights`, `avgdl = Σdl / N`; N=0 returns empty.
  Higher scores are more relevant. A sparse collection's `status` additionally
  echoes `doc_count` / `term_count` / `posting_count` / `dead_postings`.
- Sparse data is written only through HTTP and the SDKs; pgwire AUB1 batch
  rows carry no sparse field (see Section 4: pgwire offers fusion **queries**
  only).
- Memory ledger (one posting per (doc, term)): ~24 B per posting, ~12–16 B per
  (doc, term) in the forward index, plus term/doc map buckets and fragmentation.
  1M documents × 8 terms is roughly 192 MB of postings + 100–150 MB forward
  index + 40–60 MB maps ≈ 330–400 MB; trust `status` / `dead_postings`
  measurements and **never plan on smaller per-entry numbers**. Dead postings
  are reclaimed by compaction on the write path once `dead > live`;
  `dead_postings` exposes the amplification.

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
| `POST /v1/collections/{name}/fusion/search` | Dense+sparse fusion search (collection must be `sparse=bm25`; `scores` is ranking-only) |
| `GET /v1/collections/{name}/export` | Streaming NDJSON logical export (header/records/footer; backup/migration) |
| `DELETE /v1/admin/collections/{name}` | Drop a whole collection (admin, irreversible, WAL-first) |
| `GET /metrics` | Prometheus metrics (authentication exempt; includes the per-collection `houyidb_collection_vectors` gauge) |
| `GET /v1/version` | Version/capability probe; `build_sha` is the short git SHA embedded at build time |
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
  troubleshooting), see `docs/EMBEDDING_GUIDE.en.md`.

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

# Sparse + fusion (requires an explicit sparse="bm25" collection; see 2.3.1)
client.create_collection("docs", 8, index_type="flat", sparse="bm25")
docs = client.collection("docs")
docs.upsert([
    Vector(id=1, values=[0.92, 0.88, 0.06, 0.05, 0.05, 0.04, 0.05, 0.05],
           sparse=[(7, 1.0), (8, 0.8)]),
])
hits = docs.fusion_search(
    [0.90, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [(7, 1.0), (9, 0.5)], 3,
    include_payload=True)
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
  protocol; see `docs/HOUYIDB_SERVER_MANUAL.en.md` in the package for details.
- After enabling `HOUYI_AUTH_TOKEN`, the pgwire password equals that token.

### 4.1 Fusion Search Directives (P4, query only)

```sql
-- The sparse and vector directives must appear together; select the collection
-- with the houyi:collection directive
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ SELECT id FROM docs LIMIT 2;

-- Weighted fusion (alpha applies to weighted only; fusion defaults to rrf and
-- alpha defaults to 0.5)
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ /* houyi:fusion=weighted */ \
  /* houyi:alpha=0.4 */ SELECT id FROM docs LIMIT 2;
```

- Sparse terms use `term:weight[,term:weight…]`; duplicate terms are summed.
  A missing `vector` directive, or fusion/alpha parameters alone, is a syntax
  error (SQLSTATE 42601).
- The collection must already have been created as `sparse=bm25` through
  HTTP/SDK, and a `houyi:collection` directive is required: a missing directive
  or a sparse-disabled collection yields `0A000`; a missing collection yields
  `42P01` (pgwire never creates collections implicitly); term / alpha / fusion /
  k violations yield `22023`.
- pgwire offers **no sparse writes**: AUB1 batch rows contain only `id + dense
  vector`; send sparse terms through HTTP `upsert` or any SDK.
- Fusion query results are still a single `id` column (oid 20), same as normal
  vector search.

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
  safe checkpoint; followers pin the floor through their durable LSN. All six
  index backends support snapshot compaction, so an `unsupported_backend`
  `blocked_reason` should no longer occur in normal operation.

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
| `HOUYI_HNSW_PARALLEL_CHUNK_SIZE` | `256` | Planning chunk size |
| `HOUYI_HNSW_PARALLEL_PLAN_GROUP_CHUNKS` | `4` | Chunks per group (configured width = chunk × this value; acts as the cap under adaptation) |
| `HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR` | `256` | Adaptive group-width cap = `max(plan_group_min, graph size / this value)`, then min-ed with the configured width; `0` disables adaptation and keeps the fixed chunk × group_chunks width |
| `HOUYI_HNSW_PLAN_GROUP_MIN` | `16` | Floor for the adaptive **cap**: `max(this, graph size/divisor)`, then min-ed with the configured width (the final width may stay below this value) |
| `HOUYI_HNSW_MAX_BUILD_BATCH` | Server-side `8192` | Maximum inserts per build batch |
| `HOUYI_HNSW_GROUP_REPAIR` | `4,16` | Post-group orphan repair: group members with L0 degree below `min_degree` are re-linked within an id-order ±`window` neighborhood (grouped commit paths only; format `min[,window]`, window defaults to 16; `off`/`0`/empty disables it for A/B; invalid values warn and fall back to the default) |
| `HOUYI_HNSW_UPPER_EF` | `64` | **Floor** for the query-side upper-layer descent beam (search entry navigation only; does not change build semantics): with adaptation on, the actual width is `max(this, ceil(layer node estimate/8))` with the adaptive part capped at 4096 (~12.5% layer coverage at 250k, ~6.5% at 1M); this value can be explicitly set to `1..=4096` (higher = slower) to force a wider beam on smaller graphs for A/B. Invalid values warn and fall back to 64. The build side intentionally keeps beam 1 |
| `HOUYI_HNSW_UPPER_EF_ADAPTIVE` | On | Adaptation switch for the upper beam; `0/false/off/no` pins the width to `HOUYI_HNSW_UPPER_EF` (latency A/B / rollback only; disabling it can bring back wrong-cluster landings on 50k clustered corpora) |
| `HOUYI_HNSW_KEEP_PRUNED` | `0` | keepPruned links (slightly higher recall, slower build/query; not recommended for large collections) |

> Recommendation: start with the defaults. For QPS, set
> `HOUYI_HNSW_LAYER0_MULT=1`; for recall, keep the default 2×. Do not switch
> between the two directly on the same old graph snapshot (snapshot
> compatibility triggers a rebuild).
>
> Plan group width: during batched construction each group plans its neighbors
> against the **pre-group graph snapshot**, so siblings within a group are
> invisible to each other. On clustered data, in-batch siblings are each
> other's nearest neighbors; a fixed large group lets symmetric pruning turn
> them into islands after commit, and recall stays flat as ef grows. The
> default adaptive width **cap** is about 1/256 of the graph and never below 16;
> the final width is further min-ed with the configured width. This preserves
> quality on small graphs and throughput on large ones; only raise
> `HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR` (or set `0`) when deliberately trading
> quality for build throughput.
>
> On large clustered collections (e.g. a single 50k-point collection with 64
> clusters of ~800 points) the adaptive width (~195) still stays below the
> cluster size: a whole group of siblings is pruned into a pocket unreachable
> from the entry point, so **whole-cluster queries return the wrong cluster and
> raising ef barely helps**. After a grouped commit, members with L0 degree
> below 4 are therefore re-linked to robust siblings within an id-order ±16
> window (`HOUYI_HNSW_GROUP_REPAIR`, default `4,16`); the query-side
> upper-layer beam scales with layer size (floor `HOUYI_HNSW_UPPER_EF`, default
> 64; the layer node estimate is `graph points/M^layer`, and the width is
> estimate/8 with the adaptive part capped at 4096, navigation only;
> `HOUYI_HNSW_UPPER_EF_ADAPTIVE=0` disables adaptation for latency A/B, but
> disabling it can bring back wrong-cluster landings on 50k clustered corpora;
> adaptation adds a small query-time cost)
> so cross-cluster navigation does not fall into a wrong local minimum. The 4096
> cap keeps ~6.5% coverage at 1M; larger graphs can explicitly set
> `HOUYI_HNSW_UPPER_EF` to 4096 for a wider beam (query cost rises accordingly). In
> addition, once the build queue drains, the engine runs a connectivity
> backstop on **every layer**: a
> per-layer BFS from the entry point reconnects each broken in-layer component
> to the nearest reachable anchor with spare capacity (same rule as L0). An
> upper-layer orphan is otherwise fatal — the query descent lands directly on
> a wrong cluster's L0 entry and a wider ef cannot recover it.
> 50k large-cluster regression gate:
> `./scripts/clustered_recall_50k_gate.sh` (release; covers the default,
> legacy serial, legacy HTTP, and production-shaped cases).

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
Grafana; `houyidb_collection_vectors{collection=...}` reports the indexed vector
count per collection. `/v1/version`'s `build_sha` identifies the exact built
commit. Server logs go to stdout/stderr (Rust `tracing`; adjustable via
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
  collection scopes); see the [TOML configuration example in the deployment guide](DEPLOYMENT_GUIDE.en.md).
- For public exposure, add TLS (`--tls-cert` / `--tls-key`) and a reverse proxy;
  do not run an unauthenticated instance bare on the public internet.

---

## 9. Known Limitations (beta)

- Single-node architecture, no sharding; high availability is leader/follower
  with manual promote, with no automatic leader election.
- The collections WAL has bounded retention (default 16 × 64 MiB). A follower
  that falls behind the retained range receives `BEHIND_RETAINED` and must run
  the full resync in `docs/runbooks/collections-retention-resync.en.md`; it never
  wipes itself. Replication protocol v12 is not rolling-compatible across
  versions, so upgrade all replicas together (see the server manual).
- Scalar pages are append-only: re-upserting the same id appends a new row, and
  both filtering and payload read-back only see the last row per id (stale
  values stop matching as soon as they are overwritten). A check runs every
  10 minutes and compacts when at least 50% of rows are stale, reclaiming
  physical rows without changing query results.
- Every index backend supports collection snapshot compaction. Lazy-build
  backends (clustered / flat-clustered / pq-flat) are **one-shot**: they do
  **not** auto-build on restore, so a collection that was unbuilt before restart
  stays unbuilt (empty searches) until an explicit build. A built pq-flat
  restores its compact state bit-for-bit (no retraining) and keeps post-build
  buffered rows as pending.
  **Once built (`built=true`), new writes only fill the pending buffer; they are
  never merged into the index and remain unsearchable** until the collection is
  dropped and recreated. The `lazy_build` object in
  `GET /v1/collections/{name}/status` reports `built` / `buffered` /
  `rebuild_required` (`rebuild_required=true` means a built index still has
  buffered writes; eager backends omit this field).
- Array/object values in payload are treated as JSON text columns, so filtering
  semantics are limited.
- Backend/metric support matrix: hnsw / flat / clustered / flat-clustered
  support l2/cosine/ip/l1; `pq-flat` supports `l2` only (HTTP score is
  `1 − L2²`, unbounded); `hii-l0` supports `cosine` only.
- HII L1 (GPU CAGRA) / L2 (NVMe DiskANN) are not implemented; `simd-avx512` is
  only a skeleton.
- Filtered search is supported only by the hnsw / flat backends.
- Sparse terms are client-supplied and the core **does not tokenize**;
  `sparse=bm25` is hnsw / flat only, immutable after creation, allows ≤1024
  pairs per document and ≤64 pairs per query, with `term_id` in
  `1..=2^24-1`. Sparse writes exist only in HTTP/SDK; pgwire supports fusion
  queries only.
- **Downgrade prohibition (sparse)**: collection snapshot v3, manifest
  `format_version=4`, and replication protocol v12 make older binaries reject
  cleanly (no silent downgrade). However, the sparse fields in the collections
  WAL are additive optional fields that older binaries **silently ignore** on
  replay—downgrading before a snapshot exists loses the sparse index. Take a
  snapshot first, or confirm there are no `sparse=bm25` collections, before
  downgrading.
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
