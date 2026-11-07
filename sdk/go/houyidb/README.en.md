# houyidb-go

> [中文](README.md)

The official Go client for the HouyiDB vector database. Only one external dependency (`golang.org/x/crypto`, for the cross-language-consistent `StableID`); Go ≥ 1.17.

```bash
go get github.com/houyidb-community/houyidb/sdk/go/houyidb
```

## Quick Start

```go
package main

import (
    "context"

    houyidb "github.com/houyidb-community/houyidb/sdk/go/houyidb"
)

func main() {
    ctx := context.Background()
    client := houyidb.NewClient("http://127.0.0.1:7000")
    col := client.Collection("docs")

    // Write (the collection is created implicitly on first write; dimensionality is fixed by the first vector)
    _, _ = col.Upsert(ctx, []houyidb.Vector{{
        ID:      houyidb.StableID("doc-1"),
        Values:  embedding, // []float64
        Payload: map[string]interface{}{"text": "...", "category": "fruit"},
    }})

    // Search (scores are metric-derived; higher means more similar — see "Explicit Creation" below)
    result, _ := col.Search(ctx, queryEmbedding, 5,
        &houyidb.SearchOptions{IncludePayload: true})

    // Hybrid search (Filters are intersected; Op: eq/gt/gte/lt/lte/in)
    filtered, _ := col.HybridSearch(ctx, queryEmbedding, 5, &houyidb.HybridSearchOptions{
        Filters:        []houyidb.Filter{{Column: "category", Op: houyidb.OpEq, Value: "fruit"}},
        IncludePayload: true,
    })
    _, _ = result, filtered

    // Delete (by id and/or filter conditions, idempotent)
    _, _ = col.Delete(ctx, &houyidb.DeleteOptions{IDs: []uint64{houyidb.StableID("doc-1")}})
}
```

## Explicit Creation (Dims/Metric/Backend/HNSW Fixed Up Front)

`PUT /v1/collections/{name}` is a normal write path (write permission, no admin
token). Explicit creation locks dimensionality and metric: an existing name
yields 409, and metric/backend are immutable afterwards. Zero-value options
equal server defaults (`cosine` + `hnsw`).

```go
created, err := client.CreateCollection(ctx, "docs", 768, &houyidb.CreateCollectionOptions{
    Metric:    "ip",   // l2 / cosine / ip / l1
    IndexType: "hnsw", // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    Hnsw:      &houyidb.HnswConfig{M: 16, EFConstruction: 100, EFSearch: 64},
})
// created.Metric == "ip"; created.Hnsw echoes the effective HNSW parameters
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

## Admin Operation: Dropping a Whole Collection (Irreversible)

Requires server admin permission (RBAC `Action::Admin`). WAL-first, so crash
replay and follower replication converge to "gone". **Irreversible**; export
first. A same-name recreate starts empty and never resurrects old vectors.

```go
client := houyidb.NewClient(url, houyidb.WithAdminToken(adminToken))
if err := client.DeleteCollection(ctx, "docs"); err != nil {
    // unknown → *houyidb.HouyiError with StatusCode == 404
}
```

## Mapping to Other Language SDKs

- Vector ids are `uint64`—Go natively covers the keyspace, avoiding JS's 2^53 precision issue;
- `StableID` uses the same algorithm as the Python/TS SDKs and the server OpenAI gateway
  (four parties; BLAKE2b-8 big-endian & 2^63-1): the same string id yields the same u64 in any language, so deletes can be replayed across SDKs;
- Errors are uniformly `*houyidb.HouyiError` (`StatusCode` + server diagnostic text) and can be detected with `errors.As`.

## Tests

```bash
go test -race ./...   # StableID 四方对拍 + 集成（打 127.0.0.1:17000 真实服务，HOUYIDB_URL 可覆盖；不可达自动 Skip）
```

Integration prerequisite: `./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
