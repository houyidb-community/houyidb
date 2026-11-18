# houyidb-go

> [中文](README.md)

The official Go client for the HouyiDB vector database. Only one external dependency (`golang.org/x/crypto`, for the cross-language-consistent `StableID`); Go ≥ 1.25 (required by the security-fixed x/crypto).

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
    // `Sparse` is only used on `sparse=bm25` collections: nil = unchanged, empty slice = clear
    _, _ = col.Upsert(ctx, []houyidb.Vector{{
        ID:      houyidb.StableID("doc-1"),
        Values:  embedding, // []float64
        Payload: map[string]interface{}{"text": "...", "category": "fruit"},
        Sparse:  nil,
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
    Metric:       "ip",   // l2 / cosine / ip / l1
    IndexType:    "hnsw", // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    VectorDtype:  "f16",  // optional: f32 (default) / f16 (hnsw only, main-store memory halved)
    Quantization: "int8", // optional: none (default) / int8 / rabitq (hnsw only, graph-side compression)
    Sparse:       "bm25", // optional: none (default) / bm25 (hnsw/flat only, immutable after creation)
    Hnsw:         &houyidb.HnswConfig{M: 16, EFConstruction: 100, EFSearch: 64},
})
// created.Metric == "ip"; created.Hnsw echoes the effective HNSW parameters;
// created.VectorDtype/Quantization/Sparse echo f16/int8/bm25 (empty when an older server omits them)
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

`VectorDtype` allows `f16` for the hnsw backend only and is immutable after
creation; queries stay f32 and distances are computed against f16-rounded data
(asymmetric quantization). An older server that omits the field echoes an
empty string.

`Quantization` is independent of `VectorDtype`: `int8` (1 B/dim) and
`rabitq` (1 bit/dim) compress only the HNSW graph-side copy while the main
store keeps the raw vectors, and search reranks exactly on those after taking
graph candidates, so final top-k precision is unchanged. `rabitq` supports
`Metric = l2 | cosine` only (`ip`/`l1` return 400). It is hnsw-only and
immutable after creation; unset options keep the key out of the request body.
An older server that omits the field echoes an empty string.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

## Sparse Vectors (BM25) and Fusion Search (P4)

The sparse index is an **explicit creation switch**: `Sparse: "bm25"` works on
`hnsw` / `flat` backends only and is immutable after creation; implicitly
created collections are always `none`, and writing `Sparse` on a `sparse=none`
collection (or combining `bm25` with another backend at creation) returns 400.
When an older server omits the field in the creation response,
`created.Sparse` is an empty string and is tolerated as an optional field.

Terms are client-supplied (no tokenizer): `SparseTerm` is `[2]float64`
(`{term_id, weight}`), term ∈ [1, 2^24−1] and must be an integer, weight finite
and > 0; at most 1024 terms per document (the server sums duplicate terms), and
at most 64 terms per fusion query.

```go
// Sparse writes ride the dense batch (whole-document replacement): nil = unchanged, empty slice = clear
_, _ = col.Upsert(ctx, []houyidb.Vector{{
    ID:     houyidb.StableID("doc-2"),
    Values: embedding,
    Sparse: []houyidb.SparseTerm{{7, 1.0}, {8, 0.8}},
}})

// Fusion search: dense and sparse are both required; the k limit matches Search
fused, err := col.FusionSearch(ctx, queryEmbedding,
    []houyidb.SparseTerm{{7, 1.0}, {8, 0.8}}, 10,
    &houyidb.FusionSearchOptions{
        Fusion:         "rrf", // rrf (default) / weighted
        RRFK:           60,    // 1..=1000, rrf only
        Filters:        []houyidb.Filter{{Column: "category", Op: houyidb.OpEq, Value: "fruit"}},
        IncludePayload: true,
        // Alpha: &alpha,      // 0..=1, weighted only (default 0.5; a pointer distinguishes "unset" from an explicit 0)
    })
// fused.Hits[i].Score only expresses fused ranking; DenseScore is metric-converted (dense hits only),
// SparseScore is the raw BM25 score (sparse hits only); fused.Metric explains DenseScore only.
```

Two fusion modes: `rrf` (default, `score = Σ 1/(rrf_k + rank)`) and
`weighted` (min-max normalize each path, then
`alpha*dense + (1-alpha)*sparse`). Both paths over-fetch `4*k` and apply
Filters; documents missing from both paths are excluded. Explicitly supplied
`RRFK`/`Alpha` values are always strictly validated (400 when out of range,
even under the inactive mode).

Fusion search is pure read and never creates a collection implicitly; a
`sparse=none` collection returns 400; an older server without the endpoint
returns 404, passed through verbatim. Sparse **writes** are HTTP/SDK only
(pgwire supports fusion queries only).

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
