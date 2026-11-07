# houyidb-client

> [中文](README.md)

The official Rust client for the HouyiDB vector database (collections HTTP API). Synchronous blocking (`ureq` + rustls), with only 4 direct dependencies; it shares the same `stable_id` algorithm and dependency versions with the server.

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
col.upsert(&[Vector {
    id: stable_id("doc-1"),
    values: embedding, // Vec<f32>
    label: None,
    payload: Some(HashMap::from([
        ("text".into(), "...".into()),
        ("category".into(), "fruit".into()),
    ])),
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
    hnsw: Some(HnswConfig { m: 16, ef_construction: 100, ef_search: 64 }),
})?;
assert_eq!(created.metric, "ip");
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

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
