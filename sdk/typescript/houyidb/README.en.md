# houyidb (TypeScript/JavaScript)

> [中文](README.md)

The official TS/JS client for the HouyiDB vector database. Node ≥ 18 (global fetch), with a single runtime dependency (`@noble/hashes`, used for the cross-language-consistent `stableId`).

```bash
npm install houyidb
```

## Quick Start

```ts
import { HouyiClient, stableId } from "houyidb";

const client = new HouyiClient("http://127.0.0.1:7000");
const col = client.collection("docs");

// Write (the collection is created implicitly on first write; dimensionality is fixed by the first vector)
await col.upsert([
  { id: stableId("doc-1"), values: embedding, payload: { text: "...", category: "fruit" } },
]);

// Search (scores are metric-derived; higher means more similar — see "Explicit Creation" below)
const result = await col.search(queryEmbedding, 5, { includePayload: true });

// Hybrid search (filters are intersected; op: eq/gt/gte/lt/lte/in)
const filtered = await col.hybridSearch(queryEmbedding, 5, {
  filters: [{ column: "category", op: "eq", value: "fruit" }],
  includePayload: true,
});

// Delete (by id and/or filter conditions, idempotent)
await col.delete({ ids: [stableId("doc-1")] });
await col.delete({ filters: [{ column: "category", op: "eq", value: "fruit" }] });
```

## Key Design: Vector IDs Are bigint

HouyiDB's id keyspace is u64, beyond the JS Number safe range (2^53-1). This SDK:

- `SearchHit.id` is always `bigint`; `stableId()` returns `bigint`;
- the write side accepts `bigint | number | string` and normalizes it;
- request serialization and response parsing protect against 2^53 corruption internally—**do not** `JSON.parse` search responses yourself to compare ids; use the `SearchResult` returned by the SDK.

`stableId` uses the same algorithm as the Python SDK and the server OpenAI gateway (BLAKE2b-8 big-endian & 2^63-1): the same string id yields the same u64 in any language, so deletes can be replayed across SDKs.

## Explicit Creation (Dims/Metric/Backend/HNSW Fixed Up Front)

`PUT /v1/collections/{name}` is a normal write path (write permission, no admin
token). Explicit creation locks dimensionality and metric: an existing name
yields `HouyiError` statusCode 409; metric/backend are immutable afterwards,
and omitted fields take server defaults (`cosine` + `hnsw`).

```ts
const created = await client.createCollection("docs", 768, {
  metric: "ip",          // l2 / cosine / ip / l1
  indexType: "hnsw",     // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
  vectorDtype: "f16",    // optional: f32 (default) / f16 (hnsw only, main-store memory halved)
  quantization: "int8",  // optional: none (default) / int8 / rabitq (hnsw only, graph-side compression)
  sparse: "bm25",        // optional: none (default) / bm25 (hnsw/flat only, immutable after creation)
  hnsw: { m: 16, efConstruction: 100, efSearch: 64 },
});
created.metric;        // "ip"; the server echoes the effective parameters
created.vectorDtype;   // "f16"
created.quantization;  // "int8"
created.sparse;        // "bm25"; undefined when an older server omits it
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

`vectorDtype` allows `f16` for the hnsw backend only and is immutable after
creation; queries stay f32 and distances are computed against f16-rounded data
(asymmetric quantization). When an older server omits the field it stays
`undefined`; never default it to `f32` yourself.

`quantization` is independent of `vectorDtype`: `int8` (1 B/dim) and
`rabitq` (1 bit/dim) compress only the HNSW graph-side copy while the main
store keeps the raw vectors, and search reranks exactly on those after taking
graph candidates, so final top-k precision is unchanged. `rabitq` supports
`metric = l2 | cosine` only (`ip`/`l1` return 400). It is hnsw-only and
immutable after creation; unset
options keep the key out of the request body, and an older server that omits
the field leaves no own `quantization` key on the result.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

## Sparse Vectors (BM25) and Fusion Search (P4)

The sparse index is an **explicit creation switch**: `sparse: "bm25"` works on
`hnsw` / `flat` backends only and is immutable after creation; implicitly
created collections are always `none`, and writing `sparse` on a `sparse=none`
collection (or combining `bm25` with another backend at creation) returns 400.
When an older server omits the field in the creation response,
`created.sparse` is `undefined` and is tolerated as an optional field.

Terms are client-supplied (no tokenizer): `SparseTerm` is `[termId, weight]`,
term ∈ [1, 2^24−1] and must be an integer, weight finite and > 0; at most 1024
terms per document (the server sums duplicate terms), and at most 64 terms per
fusion query.

```ts
// Sparse writes ride the dense batch (whole-document replacement): absent = unchanged, [] = clear
await col.upsert([
  { id: stableId("doc-2"), values: embedding, sparse: [[7, 1.0], [8, 0.8]] },
]);

// Fusion search: dense and sparse are both required; the k limit matches search
const fused = await col.fusionSearch(
  queryEmbedding, [[7, 1.0], [8, 0.8]], 10,
  {
    fusion: "rrf",        // rrf (default) / weighted
    rrfK: 60,             // 1..=1000, rrf only
    // alpha: 0.5,        // 0..=1, weighted only (default 0.5)
    filters: [{ column: "category", op: "eq", value: "fruit" }],
    includePayload: true,
  },
);
for (const hit of fused.hits) {
  // hit.score only expresses fused ranking; hit.denseScore is metric-converted and hit.sparseScore is raw BM25
  console.log(hit.id, hit.score, hit.denseScore, hit.sparseScore);
}
```

Two fusion modes: `rrf` (default, `score = Σ 1/(rrf_k + rank)`) and
`weighted` (min-max normalize each path, then
`alpha*dense + (1-alpha)*sparse`). Both paths over-fetch `4*k` and apply
filters; documents missing from both paths are excluded. Explicitly supplied
`rrfK`/`alpha` values are always strictly validated (400 when out of range,
even under the inactive mode). `fused.metric` explains `denseScore` only;
`fused.fusionMode` / `rrfK` / `alpha` echo the effective parameters.

Fusion search is pure read and never creates a collection implicitly; a
`sparse=none` collection returns 400; an older server without the endpoint
returns 404, passed through verbatim. Sparse **writes** are HTTP/SDK only
(pgwire supports fusion queries only).

## Admin Operation: Dropping a Whole Collection (Irreversible)

Requires server admin permission (RBAC `Action::Admin`). WAL-first, so crash
replay and follower replication converge to "gone". **Irreversible**; export
first. A same-name recreate starts empty and never resurrects old vectors.

```ts
// adminToken is attached only to /v1/admin/* requests; normal reads/writes never carry it
const admin = new HouyiClient(url, { adminToken });
await admin.deleteCollection("docs"); // unknown → HouyiError statusCode 404
```

## API Mapping with the Python SDK

| Python | TypeScript |
|---|---|
| `collection.upsert(vectors)` | `col.upsert(vectors)` |
| `collection.search(v, k, ef_search=, include_payload=)` | `col.search(v, k, { efSearch, includePayload })` |
| `collection.hybrid_search(v, k, filters=, ...)` | `col.hybridSearch(v, k, { filters, ... })` |
| `collection.fusion_search(v, sparse, k, fusion=, rrf_k=, alpha=, ...)` | `col.fusionSearch(v, sparse, k, { fusion, rrfK, alpha, ... })` |
| `collection.delete(ids=, filters=)` | `col.delete({ ids, filters })` |
| `collection.status()` | `col.status()` |
| `client.health() / list_collections() / index_status()` | `client.health() / listCollections() / indexStatus()` |
| `stable_id(text) -> int` | `stableId(text) -> bigint` |
| `HouyiError(status_code)` | `HouyiError(statusCode)` |

## Tests

```bash
npm test          # 单元（stable_id 三方对拍）+ 集成（打 127.0.0.1:17000 真实服务，HOUYIDB_URL 可覆盖）
npm run build     # 产出 dist/（ESM + d.ts）
```

Integration prerequisite: `./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
