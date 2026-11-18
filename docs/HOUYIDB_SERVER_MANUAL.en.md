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

### 4.2.1 Fusion Search (P4, query only)

```sql
-- sparse and vector must appear together; the collection is selected by the
-- houyi:collection directive
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ SELECT id FROM docs LIMIT 2;

-- Weighted fusion; fusion defaults to rrf and alpha defaults to 0.5
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ /* houyi:fusion=weighted */ \
  /* houyi:alpha=0.4 */ SELECT id FROM docs LIMIT 2;
```

- Directive grammar: `/* houyi:sparse=term:weight[,term:weight…] */` (duplicate
  terms are summed; ≤64 raw pairs), `/* houyi:fusion=rrf|weighted */`, and
  `/* houyi:alpha=0..1 */`. Sparse + vector → fusion search; sparse alone or
  fusion/alpha alone → syntax error (SQLSTATE 42601).
- Execution semantics: an explicit `/* houyi:collection=name */` is required
  (missing directive `0A000`); a missing collection yields `42P01` (never
  creates implicitly); a `sparse=none` collection yields `0A000`; term / alpha
  / fusion / k violations yield `22023`; internal errors `XX000`.
- Results remain a single `id` column (oid 20). pgwire offers **no sparse
  writes** (AUB1 rows carry only `id + dense vector`); send sparse terms
  through HTTP `upsert` / SDKs.

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
#     "distance_metric": "cosine", "hnsw_m": 24,
#     "hnsw_ef_construction": 400, "hnsw_ef_search": 256, ...}
```

### 5.5.1 Explicit Creation and Distance Metrics

```bash
PUT /v1/collections/{name}
# {"dims":8,"metric":"ip","index_type":"hnsw",
#  "hnsw_m":16,"hnsw_ef_construction":100,"hnsw_ef_search":64}
# -> 201 {"created":true,"collection":"goods","dims":8,"metric":"ip",
#         "index_type":"hnsw","hnsw":{...}}
```

- RBAC classifies it as `Action::Write` (write permission, not admin); names
  outside a tenant scope are still rejected.
- Parameters are validated before any WAL append (`validate_create`): invalid
  metric/backend/dims/bounds return 400 and never leave an unreplayable record.
  An existing name (even with identical parameters) yields 409; follower/read-only
  503. On success a `collection_create` record is appended and replicated.
- `metric` and `index_type` are immutable after creation; the three HNSW
  parameters must be supplied together (`m` 1..=1024,
  `ef_construction`/`ef_search` 1..=65536).
- `vector_dtype`: `f32` (default) / `f16` (aliases float32/fp32/float16/fp16/half,
  case-insensitive). `f16` halves resident vector memory, is hnsw-only, and is
  immutable after creation; the kernel queries in f32 against f16 storage
  (asymmetric quantization, f16 values are first rounded onto the f16 grid).
  Non-f32 with a non-hnsw backend yields 400; an existing name with a different
  dtype (f32→f16 or vice versa) yields 409. Creation responses, list summaries
  and status all echo `vector_dtype`.
- `quantization`: `none` (default) / `int8` / `rabitq`, **independent of**
  `vector_dtype`: `vector_dtype` selects the main-store encoding while
  `quantization` only compresses the HNSW **graph-side copy**. The main store /
  hot buffer / builder queue always keep the raw (f32/f16) vectors; search takes
  candidates from the quantized graph (candidate width ×2 for int8, ×16 for
  rabitq: `INT8_ANN_K_MULTIPLIER = 2`, `RABITQ_ANN_K_MULTIPLIER = 16`, whose
  code bandwidth matches the former) and
  reranks exactly on the raw vectors, so the final top-k scores never pass
  through quantization. `int8` / `rabitq` are hnsw-only (other backends yield
  400) and immutable after creation; an existing name explicitly carrying a
  different quantization yields 409. Cosine normalizes first and then
  quantizes; the inserted vector acts as a query and is **always f32**
  (asymmetric quantization). Non-finite components are already rejected by the
  server's write gate; the library layer defensively replaces them. pgwire has
  no `CREATE COLLECTION` grammar, so quantized collections can only be created
  via HTTP / SDK. Creation responses, list summaries and status all echo
  `quantization`.
- `int8` kernel: per-dimension int8 codes plus scalar factors; graph-side
  codes are 1 B/dim. Memory ledger (honest; never plan capacity on "halved"):
  non-cosine f32 4→5 B/dim (+1), non-cosine f16 2→3 B/dim (+1), cosine f32
  8→5 B/dim (−3), cosine f16 4→3 B/dim (−1); the win is the in-graph distance
  working set (4→1 B/dim, cache locality) and memory bandwidth.
- `rabitq` kernel: D = dims rounded up to a power of two with zero padding;
  deterministic random rotation P = H_D ∘ diag(signs) (the seed depends only
  on D, so results are bit-reproducible across processes/nodes/restarts);
  encoding = sign bits (`ceil(D/64)` u64 words) + `alpha = Σ|ō_i|` +
  `norm = ||x||`. The query side does the zero-padding and rotation once at
  the traversal entrance (O(D log D)); asymmetric estimation
  `est_cos = clamp(sign_dot/alpha, -1, 1)`: cosine distance `1−est`, squared
  L2 `max(0, q²+x²−2qx·est)`; stored×stored comparisons (construction
  heuristic/pruning) use Hamming distance plus the arcsine law. Zero-norm
  vectors use the sentinel `{codes=0, alpha=1.0, norm=0}`; when `q_norm` or
  `x_norm` is 0 the cross term is 0 (cosine distance is naturally 1).
  **Supports `metric = l2 | cosine` only**: `ip` / `l1` fail fast at three
  layers—creation, WAL replay, and snapshot expectations (400 / dirty record
  rejection). Memory ledger (honest; never plan capacity on "halved"): main
  store memory is unchanged (the graph-side `source` shares the main-store
  `Arc`); graph-side codes = `ceil(D/64)·8` B + 8 B factors, and at dims=768
  (D=1024) each vector costs 128 B + 8 B (same yardstick: f32 3072 B, int8
  768 B + 8 B). Versus the f32 graph that is only ~136 B more per vector
  (≈1/24), and ~1/6 of int8; it also needs shared rotation signs of 4·D bytes
  per collection plus one D×f32 scratch buffer per query. Outside cosine total
  memory grows slightly (≈+0.18 B/dim); for cosine it shrinks versus
  unquantized (≈−3.8 B/dim).
- `sparse`: `none` (default) / `bm25`, hnsw / flat backends only, immutable
  after creation, and implicit creation is always `none`; `bm25` builds a side
  inverted index on the collection (dense search results are unchanged). Writes
  and fusion queries are covered in 5.5.2. Changing sparse on an existing name
  yields 409; an illegal combination yields 400.
- Future extension point (documented only, no code): codebooks for stronger
  quantizers such as PQ would be trained once by the leader, distributed via a
  WAL quantizer record applied by followers by LSN, and embedded in graph
  snapshots; this release ships only `int8` / `rabitq` and exposes no
  half-built API.
- Support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered` support
  `l2`/`cosine`/`ip`/`l1`; `pq-flat` supports `l2` only; `hii-l0` supports
  `cosine` only.
- Kernel distances are always "smaller = closer" (L2 is squared, IP is `-dot`,
  L1 is Σ|a-b|). HTTP scores are converted at the single edge by metric:
  cosine `1-d` (clamped to [-1,1]), l2/l1 `1-d` (unclamped), ip `-d` (the raw
  dot product). Collection/hybrid search responses echo `metric`; the
  benchmark-compatibility path omits it to keep bytes identical.
- Snapshots and recovery: collection snapshot v2 carries `metric`, optional
  HNSW parameters, `vector_dtype` (flags bit0), and `quantization` (flags
  bit1 = int8, bit2 = RaBitQ (`0x04`); without a sparse index the payload
  version stays 2 with no extra body bytes (with `bm25` see 5.5.2: payload v3
  + flags bit3); the two bits are mutually exclusive; older binaries accept
  only bits 0–1 and fail fast on the RaBitQ flag). The manifest stores
  `vector_dtype` / `quantization` as optional fields (quantization: 1=int8,
  2=rabitq); v1 decodes to the backend default metric/f32/unquantized. Graph
  snapshots carry the metric, dtype and quantization and reject mismatches
  (a quantized collection must be restored as the matching backend from the
  snapshot header, then reranked exactly on the raw source vectors instead of
  degrading to dequantized approximations). Standalone HNSW graph snapshots
  use an asymmetric version gate: f32 writes version=1 (readable by older
  binaries), f16 writes version=2, int8 writes version=3, and rabitq writes
  version=4 (flags bit3 = `0x08` mandatory, bit2 forbidden; older binaries
  reject it with `UnsupportedVersion` and fall back to a full WAL replay,
  instead of silently downgrading to f32/unquantized and freezing that on the
  next snapshot). Manifest `format_version`: f32-only → 1, int8 present → 2,
  rabitq present → 3 (exact match, not loose). The `quantization` field value
  1 requires ≥2 and hnsw + `HnswGraph`; value 2 requires ≥3, hnsw +
  `HnswGraph`, and metric l2/cosine. Every backend supports snapshot
  compaction (P5): `vector_kind` gains 3 = `built pq-flat compact state` and
  4 = `lazy-backend buffered raw rows` (additive; older binaries cleanly reject
  unknown kinds at payload decode, and `format_version` is unchanged). Lazy
  snapshots carry: clustered / flat-clustered = built raw vectors (boundary
  copies deduped by id) + pending rows; built pq-flat = compact built state
  (centroids/codebook/cluster ids+codes) + pending rows; unbuilt pq-flat =
  pending rows. Restoring built pq-flat is bit-identical without retraining;
  lazy backends only refill the buffer and never auto-build (matching a cold
  start via full replay, empty searches until an explicit build). `kind=4` is
  valid only for clustered/flat-clustered/pq-flat and `kind=3` only for
  pq-flat, cross-checked in both manifest and payload. Retention no longer
  reports an `unsupported_backend` block in normal operation.
  `collection_create` WAL records carry
  metric/HNSW/`vector_dtype`/`quantization` too (hnsw always writes it,
  including an explicit `"none"`; non-hnsw omits it), so cold replay never
  resets historical collections to cosine / f32 / unquantized.

### 5.5.2 Sparse Index (BM25) and Fusion Search (P4)

- Data model: sparse terms are caller-supplied `(term_id, weight)`; `term_id ∈
  [1, 2^24-1]`, weight finite and >0; ≤1024 raw pairs per document and ≤64 per
  query; duplicate terms are summed and order is irrelevant; **no tokenizer**.
  A `sparse=bm25` collection keeps a side `Bm25Index` (term → postings plus a
  forward index) while the main store / hot buffer / builder queue still hold
  only dense vectors, so dense search results are unaffected.
- Writes: each vector in `POST /v1/collections/{name}/upsert` may carry
  `"sparse": [[term, weight], …]`; an absent key or `null` means unchanged,
  `[]` clears, and re-writing the same id is **whole-document replacement**
  (not an incremental merge), making WAL covered replay idempotent. Sparse
  updates land synchronously with scalar rows inside the write lock, so vectors
  just written and still in the async graph-build queue are immediately
  matchable by sparse search/delete. A `sparse` key on a `sparse=none`
  collection yields 400 (never silently dropped).
- Fusion search: `POST /v1/collections/{name}/fusion/search` with
  `{vector, sparse, k, filters?, ef_search?, include_payload?, fusion?,
  rrf_k?, alpha?}`; both dense and sparse must be non-empty. Both paths
  over-fetch 4×k candidates (after filters), fuse, then cut to k; ordering is
  fused score descending with id ascending as the tie-break.
  - RRF (default, `rrf_k=60`, range 1..=1000): `Σ 1/(rrf_k + rank)` with rank
    starting at 1, one term per hit list.
  - Weighted (`alpha` default 0.5, range 0..=1): each modality is min-max
    normalized over the filtered candidates (a single element normalizes to
    1.0, a missing modality counts as 0), then
    `score = alpha*dense + (1-alpha)*sparse`.
  - Response fields: `ids`; `scores` (**fusion ranking semantics only, no
    metric**); `dense_scores` (collection-metric conversion, null for an absent
    modality); `sparse_scores` (raw BM25, null for an absent modality);
    `metric` (explains `dense_scores` only); `fusion:{mode, rrf_k, alpha}`
    (echoes only the parameters that actually scored); optional `payloads`.
  - Explicitly supplied values are validated even when the active mode does
    not use them (invalid values always yield 400); `k` / `ef_search` limits
    match normal search; filters are hnsw / flat only.
- BM25: `k1=1.2`, `b=0.75`, `idf = ln(1 + (N-df+0.5)/(df+0.5))`, `tf` is the
  client weight, `dl = Σ weights`, `avgdl = Σdl/N`; N=0 returns empty. Dead
  postings are compacted on the write path once `dead > live`.
  `GET /v1/collections/{name}/status` additionally echoes
  `sparse: {mode, doc_count, term_count, posting_count, dead_postings}` for
  bm25 collections (`none` echoes only `mode`, avoiding a misleading all-zero
  reading).
- Export: `GET /v1/collections/{name}/export` appends an optional
  `"sparse":[[term,weight]…]` to each record on bm25 collections (an additive
  field so migrations do not lose sparse data).
- Observability: `/metrics` gains `houyidb_sparse_docs`,
  `houyidb_sparse_terms`, `houyidb_sparse_postings`,
  `houyidb_sparse_dead_postings` (cross-collection aggregated gauges) and
  `houyidb_fusion_searches_total` (counter); `/v1/version` capabilities gain
  `sparse_bm25` and `fusion_search`.
- Memory ledger (honest; never plan on smaller per-entry numbers): ~24 B per
  (doc, term) posting, ~12–16 B per (doc, term) in the forward index, plus
  term/doc map buckets and growth fragmentation; 1M documents × 8 terms ≈
  192 MB postings + 100–150 MB forward index + 40–60 MB maps ≈ 330–400 MB.
  Trust status/metrics measurements.
- Concurrency: the sparse index sits behind a collection-level `RwLock` and is
  updated inside the write lock (never nested with scalar pages); queries
  compute synchronously under the read lock and **never hold it across
  `.await`**; no global Mutex, no sharding.
- Version discipline: collection snapshot payload **v3** (flags bit3=`0x08`
  mandatory, v1/v2 must not set it, older binaries reject v3), manifest entry
  `sparse` (1=bm25) with `format_version=4`, and replication protocol **v12**;
  older binaries reject v3 / format 4 / v12 cleanly. The WAL sparse fields are
  additive optional fields (`COLLECTION_WAL_VERSION` stays 1) that older
  binaries **silently ignore** on replay—downgrading before a snapshot exists
  loses the sparse index, so never hand a v12 WAL to an older binary.

### 5.6 WAL Management

```bash
GET  /v1/wal/status      # main WAL plus collections segments/retention/manifest
POST /v1/wal/checkpoint  # {"through_lsn": 42} -> truncate WAL
POST /v1/wal/rotate       # archive current WAL
POST /v1/wal/snapshot     # create vector snapshot

# Storage lifecycle (P5.0)
GET  /v1/collections/{name}/status   # hot/indexed/pending + applied_lsn + scalar stats
                                     # one-shot backends (clustered/flat-clustered/pq-flat)
                                     # also return lazy_build:{built,buffered,rebuild_required}
POST /v1/collections/{name}/compact  # {"force": true} in-memory scalar compaction
POST /v1/admin/collections/compact   # {"rotate":true,"hold_retention_secs":3600}
                                     # snapshot cycle -> manifest -> optional rotate -> retention

# Admin: drop a whole collection (irreversible; requires admin credentials)
DELETE /v1/admin/collections/{name}
```

`DELETE /v1/admin/collections/{name}` responds
`{"deleted":true,"collection":"<name>"}` on success; unknown/already-dropped 404,
delete races 409, follower/read-only or expired lease 503. Deletion is WAL-first:
a `{"v":1,"op":"collection_drop","name":"<name>"}` record is appended before the
in-memory removal, so crash replay and follower live apply both converge to
"gone"; live apply on a follower does not append to the local WAL. A same-name
recreate yields a fresh empty collection and never resurrects old vectors.
Replay treats drops like any other record under the per-collection snapshot
checkpoint: after `create→drop→recreate→snapshot`, the stale drop LSN is below
the new collection's checkpoint and is skipped, so live snapshot data survives.
Periodic snapshots skip a collection being dropped (the cycle is not aborted);
after dropping everything an empty manifest is written with
`safe_checkpoint_lsn = global_barrier_lsn`, allowing WAL truncation to that
barrier. Followers **never** publish empty-name manifests: their local WAL
appends before apply, so `next_lsn - 1` may outrun a not-yet-applied
create/upsert; publishing it as safe would let retention truncate unapplied
records. A follower keeps the previous manifest instead (temporarily using
more local WAL) until a collection lands or it is promoted to leader.
Replication protocol **v11** extends v10's `collection_create` quantization
(int8) semantics by admitting `rabitq` into the canonical value domain behind
a fail-fast version gate: v10/v11 cannot be mixed, so upgrades must be
coordinated cluster-wide (stop replication, upgrade everything, resume).
**Never** cold-replay v11 WAL on a v10 or older binary: a v10 binary treats
the `rabitq` string as non-canonical and fails fast during replay (no silent
downgrade, but it is rejected at the handshake already); v9 and older
binaries additionally ignore the `quantization` field and build int8
collections as **unquantized** graphs (a silent fork — recall and memory
diverge from the source after restart), and also skip unknown drop records and
ignore the metric/dtype fields, rebuilding as cosine/f32 — resurrecting
dropped collections and silently diverging on metrics and storage encoding.
v10 WAL is still readable by v11 binaries (backward compatible); int8
snapshots written by v11 (graph snapshot v3 / manifest v2) and rabitq
snapshots (graph snapshot v4 / manifest format 3) are not downgradable
either.
Replication protocol **v12** adds sparse fields (`collection_create.sparse`
and the optional `sparse` on upsert entries) on top of v11. v11 and older
binaries **silently ignore** those additive fields when decoding v12 WAL —
losing the sparse index while no snapshot exists; therefore v12 must not be
mixed with older versions and upgrades require cluster-wide coordination
(stop replication, upgrade everything, resume). **Never** replay v12 WAL on a
v11 or older binary; collection snapshot v3 and manifest `format_version=4`
written by v12 are rejected cleanly by older binaries (version range / unknown
format) instead of silently degrading.

`collections_wal_segment_bytes` (default 64 MiB; 0 = manual rotation only),
`collections_wal_max_segments` (default 16; 0 = disable deletion),
`collections_wal_follower_grace_secs` (default 60),
`collections_scalar_compact_interval_secs` (default 600; 0 = disabled),
`collections_scalar_compact_stale_ratio` (default 0.5). Scalar pages are
append-only: re-upserting the same id appends a new row, and filtering plus
payload read-back only see the last row per id; compaction reclaims physical
rows without changing query results (the semantics after the public issue #6
fix).

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
