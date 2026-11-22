# houyidb-client

> [中文](README.md)

The official Rust client for the HouyiDB vector database (collections HTTP API). Synchronous blocking (`ureq` + rustls), with only 4 direct dependencies; it shares the same `stable_id` algorithm and dependency versions with the server.

> You may also depend on the facade crate `houyidb` (a full re-export of this
> crate): `houyidb = "0.2.0-beta.9"`.

```toml
[dependencies]
houyidb-client = { path = "sdk/rust/houyidb-client" }
```

## Quick Start

```rust
use houyidb_client::{stable_id, Client, Filter, FilterOp, SearchOptions, Vector};
use std::collections::HashMap;

let client = Client::new("http://127.0.0.1:7000");
let col = client.collection("docs");

// Write (the collection is created implicitly on first write; dimensionality is fixed by the first vector)
// `sparse` is only used on `sparse=bm25` collections: None = unchanged, Some([]) = clear
col.upsert(&[Vector {
    id: stable_id("doc-1"),
    values: embedding, // Vec<f32>
    label: None,
    payload: Some(HashMap::from([
        ("text".into(), "...".into()),
        ("category".into(), "fruit".into()),
    ])),
    sparse: None,
}])?;

// Search (scores are metric-derived and higher means more similar; see "Explicit Creation" below)
let result = col.search(&query, 5, &SearchOptions {
    include_payload: true,
    ..Default::default()
})?;

// Hybrid search (filters are intersected; op: eq/gt/gte/lt/lte/in)
let filtered = col.hybrid_search(&query, 5, &SearchOptions {
    filters: vec![Filter::new("category", FilterOp::Eq, "fruit")],
    include_payload: true,
    ..Default::default()
})?;

// Delete (by id and/or filter conditions, idempotent; the two are unioned)
col.delete(&[stable_id("doc-1")], &[])?;
```

## Explicit Creation (Dims/Metric/Backend/HNSW Fixed Up Front)

`PUT /v1/collections/{name}` is a normal write path (write permission, no admin
token). Explicit creation locks dimensionality and metric: an existing name
yields 409, and metric/backend are immutable afterwards. Omitted fields take
server defaults (`cosine` + `hnsw`).

```rust
use houyidb_client::{Client, CreateCollectionOptions, HnswConfig};

let created = client.create_collection("docs", 768, &CreateCollectionOptions {
    metric: Some("ip".into()),          // l2 / cosine / ip / l1
    index_type: Some("hnsw".into()),    // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    vector_dtype: Some("f16".into()),   // optional: f32 (default) / f16 (hnsw only, main-store memory halved)
    quantization: Some("int8".into()),  // optional: none (default) / int8 / rabitq (hnsw only, graph-side compression)
    sparse: Some("bm25".into()),        // optional: none (default) / bm25 (hnsw/flat only, immutable after creation)
    hnsw: Some(HnswConfig { m: 16, ef_construction: 100, ef_search: 64 }),
})?;
assert_eq!(created.metric, "ip");
assert_eq!(created.vector_dtype.as_deref(), Some("f16"));
assert_eq!(created.quantization.as_deref(), Some("int8"));
assert_eq!(created.sparse.as_deref(), Some("bm25"));
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

`vector_dtype` allows `f16` for the hnsw backend only and is immutable after
creation; queries stay f32 and distances are computed against f16-rounded data
(asymmetric quantization). When an older server omits the field the result is
`None`; callers must not default it to `f32`.

`quantization` is independent of `vector_dtype`: `int8` (1 B/dim) and
`rabitq` (1 bit/dim) compress only the HNSW graph-side copy while the main
store keeps the raw vectors, and search reranks exactly on those after taking
graph candidates, so final top-k precision is unchanged. `rabitq` supports
`metric = l2 | cosine` only (`ip`/`l1` return 400). It is hnsw-only and
immutable after creation; unknown values and explicit non-hnsw combinations
are rejected by the server (400). When an older server omits the field the
result is `None`; unset options keep the key out of the request body.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

## Sparse Vectors (BM25) and Fusion Search (P4)

The sparse index is an **explicit creation switch**: `sparse=bm25` works on
`hnsw` / `flat` backends only and is immutable after creation; implicitly
created collections are always `none`, and writing `sparse` on a `sparse=none`
collection (or combining `bm25` with another backend at creation) returns 400.
When an older server omits the field in the creation response,
`created.sparse` is `None` and is tolerated as an optional field.

Terms are client-supplied (no tokenizer): `(term_id: u32, weight: f32)`,
term ∈ [1, 2^24−1], weight finite and > 0; at most 1024 terms per document
(the server sums duplicate terms), and at most 64 terms per fusion query.

```rust
use houyidb_client::{FusionOptions, SparseTerms, Vector};

// Sparse writes ride the dense batch (whole-document replacement): None = unchanged, Some([]) = clear
let terms: SparseTerms = vec![(7, 1.0), (8, 0.8)];
col.upsert(&[Vector {
    id: stable_id("doc-2"),
    values: embedding,
    label: None,
    payload: None,
    sparse: Some(terms),
}])?;

// Fusion search: dense and sparse are both required; the k limit matches search
let fused = col.fusion_search(&query, &[(7, 1.0), (8, 0.8)], 10, &FusionOptions {
    fusion: Some("rrf".into()),      // rrf (default) / weighted
    rrf_k: Some(60),                 // 1..=1000, rrf only
    // alpha: Some(0.5),             // 0..=1, weighted only (default 0.5)
    filters: vec![Filter::new("category", FilterOp::Eq, "fruit")],
    include_payload: true,
    ..Default::default()
})?;
for hit in &fused.hits {
    // `score` only expresses fused ranking; `dense_score` is metric-converted and `sparse_score` is raw BM25
    println!("{} fused={} dense={:?} sparse={:?}",
        hit.id, hit.score, hit.dense_score, hit.sparse_score);
}
```

Two fusion modes: `rrf` (default, `score = Σ 1/(rrf_k + rank)`) and
`weighted` (min-max normalize each path, then
`alpha*dense + (1-alpha)*sparse`). Both paths over-fetch `4*k` and apply
filters; documents missing from both paths are excluded. Explicitly supplied
`rrf_k`/`alpha` values are always strictly validated (400 when out of range,
even under the inactive mode). `fused.metric` explains `dense_score` only;
`fused.fusion_mode` / `rrf_k` / `alpha` echo the effective parameters.

Fusion search is pure read and never creates a collection implicitly; a
`sparse=none` collection returns 400; an older server without the endpoint
returns 404, passed through verbatim. Sparse **writes** are HTTP/SDK only
(pgwire supports fusion queries only).

## Admin Operation: Dropping a Whole Collection (Irreversible)

Requires server admin permission (RBAC `Action::Admin`). Deletion is WAL-first:
the record lands before the in-memory drop, so crash replay and follower
replication converge to "gone". **Irreversible**; export via
`GET /v1/collections/{name}/export` first. A same-name recreate starts empty and
never resurrects the old vectors.

```rust
// The admin token is attached only to /v1/admin/* requests; normal reads/writes never carry it
let admin = Client::with_timeout(url, timeout).with_admin_token(admin_token);
admin.delete_collection("docs")?; // unknown → HouyiError::Http { status: 404, .. }
```

## Mapping to Other Language SDKs

- Vector ids are `u64`—native keyspace coverage, avoiding JS's 2^53 precision issue;
- `stable_id` uses the same algorithm as the Python/TS/Go SDKs and the server OpenAI gateway
  (five parties; BLAKE2b-8 big-endian & 2^63-1; the digest length is an algorithm parameter, not truncation): the same string id
  yields the same u64 in any language, so deletes can be replayed across SDKs;
- Errors are uniformly `HouyiError`: `Http { status, message }` (the server's plain-text diagnostic is passed through verbatim)
  or `Transport` (connection/read failure);
- Collection names are path-escaped per RFC 3986 (same semantics as Python `quote(safe="")`),
  so names containing `/ ? #` or Chinese characters do not alter routing.

## Tests

```bash
cargo test -p houyidb-client            # 单元（stable_id 五方对拍）+ 集成
```

Integration tests hit the real service at 127.0.0.1:17000 (`HOUYIDB_URL` can override; skipped automatically when unreachable).
Prerequisite: `./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
