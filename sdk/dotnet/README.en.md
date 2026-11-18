# houyidb-dotnet

> [中文](README.md)

The official .NET client for the HouyiDB vector database (collections HTTP API). Fully asynchronous with
async/await, targeting net8.0 + net10.0; the only external dependency is `Konscious.Security.Cryptography.Blake2`
(for the cross-language-consistent `StableId`).

```bash
dotnet add package HouyiDb
```

## Quick Start

```csharp
using HouyiDb;

var client = new HouyiClient("http://127.0.0.1:7000");
var col = client.GetCollection("docs");

// Write (the collection is created implicitly on first write; dimensionality is fixed by the first vector)
await col.UpsertAsync(new[]
{
    new Vector
    {
        Id = StableId.Of("doc-1"),
        Values = embedding, // float[]
        Payload = new Dictionary<string, object?>
        {
            ["text"] = "...",
            ["category"] = "fruit",
        },
    },
});

// Search (scores are metric-derived; higher means more similar — see "Explicit Creation" below)
var result = await col.SearchAsync(queryEmbedding, 5,
    new SearchOptions { IncludePayload = true });

// Hybrid search (Filters are intersected; FilterOp: Eq/Gt/Gte/Lt/Lte/In)
var options = new SearchOptions { IncludePayload = true };
options.Filters.Add(new Filter("category", FilterOp.Eq, "fruit"));
var filtered = await col.HybridSearchAsync(queryEmbedding, 5, options);

// Delete (by id and/or filter conditions, idempotent; the two are unioned)
await col.DeleteAsync(ids: new[] { StableId.Of("doc-1") });
```

## Explicit Creation (Dims/Metric/Backend/HNSW Fixed Up Front)

`PUT /v1/collections/{name}` is a normal write path (write permission, no admin
token). Explicit creation locks dimensionality and metric: an existing name
yields `HouyiException(409)`; metric/backend are immutable afterwards, and
omitted fields take server defaults (`cosine` + `hnsw`).

```csharp
var created = await client.CreateCollectionAsync("docs", 768, new CreateCollectionOptions
{
    Metric = "ip",          // l2 / cosine / ip / l1
    IndexType = "hnsw",     // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    VectorDtype = "f16",    // optional: f32 (default) / f16 (hnsw only, main-store memory halved)
    Quantization = "int8",  // optional: none (default) / int8 / rabitq (hnsw only, graph-side compression)
    Sparse = "bm25",        // optional: none (default) / bm25 (hnsw/flat only, immutable after creation)
    Hnsw = new HnswParams(M: 16, EfConstruction: 100, EfSearch: 64),
});
// created.Metric == "ip"; created.VectorDtype/Quantization/Sparse echo f16/int8/bm25;
// the server echoes the effective parameters
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

`VectorDtype` allows `f16` for the hnsw backend only and is immutable after
creation; queries stay f32 and distances are computed against f16-rounded data
(asymmetric quantization). An older server that omits the field yields `null`;
never default it to `f32` yourself.

`Quantization` is independent of `VectorDtype`: `int8` (1 B/dim) and
`rabitq` (1 bit/dim) compress only the HNSW graph-side copy while the main
store keeps the raw vectors, and search reranks exactly on those after taking
graph candidates, so final top-k precision is unchanged. `rabitq` supports
`Metric = l2 | cosine` only (`ip`/`l1` return 400). It is hnsw-only and
immutable after creation; unset
options keep the key out of the request body, and an older server that omits
the field yields `null`.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

## Sparse Vectors (BM25) and Fusion Search (P4)

The sparse index is an **explicit creation switch**: `Sparse = "bm25"` works
on `hnsw` / `flat` backends only and is immutable after creation; implicitly
created collections are always `none`, and writing `Sparse` on a `sparse=none`
collection (or combining `bm25` with another backend at creation) returns 400.
When an older server omits the field in the creation response,
`created.Sparse` is `null` and is tolerated as an optional field.

Terms are client-supplied (no tokenizer): `new SparseTerm(termId, weight)`,
term ∈ [1, 2^24−1], weight finite and > 0; at most 1024 terms per document
(the server sums duplicate terms), and at most 64 terms per fusion query.

```csharp
// Sparse writes ride the dense batch (whole-document replacement): null = unchanged, empty list = clear
await col.UpsertAsync(new[]
{
    new Vector
    {
        Id = StableId.Of("doc-2"),
        Values = embedding,
        Sparse = new List<SparseTerm> { new(7, 1.0f), new(8, 0.8f) },
    },
});

// Fusion search: dense and sparse are both required; the k limit matches Search
var fusionOptions = new FusionOptions
{
    Fusion = "rrf",        // rrf (default) / weighted
    RrfK = 60,             // 1..=1000, rrf only
    // Alpha = 0.5f,       // 0..=1, weighted only (default 0.5)
    IncludePayload = true,
};
fusionOptions.Filters.Add(new Filter("category", FilterOp.Eq, "fruit"));
var fused = await col.FusionSearchAsync(
    queryEmbedding, new[] { new SparseTerm(7, 1.0f), new SparseTerm(8, 0.8f) }, 10,
    fusionOptions);
foreach (var hit in fused.Hits)
{
    // hit.Score only expresses fused ranking; hit.DenseScore is metric-converted and hit.SparseScore is raw BM25
    Console.WriteLine($"{hit.Id} {hit.Score} {hit.DenseScore} {hit.SparseScore}");
}
```

Two fusion modes: `rrf` (default, `score = Σ 1/(rrf_k + rank)`) and
`weighted` (min-max normalize each path, then
`alpha*dense + (1-alpha)*sparse`). Both paths over-fetch `4*k` and apply
Filters; documents missing from both paths are excluded. Explicitly supplied
`RrfK`/`Alpha` values are always strictly validated (400 when out of range,
even under the inactive mode). `fused.Metric` explains `DenseScore` only;
`fused.FusionMode` / `RrfK` / `Alpha` echo the effective parameters.

Fusion search is pure read and never creates a collection implicitly; a
`sparse=none` collection returns 400; an older server without the endpoint
returns 404, passed through verbatim. Sparse **writes** are HTTP/SDK only
(pgwire supports fusion queries only).

## Admin Operation: Dropping a Whole Collection (Irreversible)

Requires server admin permission (RBAC `Action::Admin`). WAL-first, so crash
replay and follower replication converge to "gone". **Irreversible**; export
first. A same-name recreate starts empty and never resurrects old vectors. The
admin token is attached only to `/v1/admin/*` requests.

```csharp
var admin = HouyiClient.WithAdminToken(baseUrl, adminToken);
await admin.DeleteCollectionAsync("docs"); // unknown → HouyiException(404)
```

## Mapping to Other Language SDKs

- Vector ids are `ulong`—native keyspace coverage, avoiding JS's 2^53 precision issue;
- `StableId.Of` uses the same algorithm as the Python/TS/Go/Rust SDKs and the server OpenAI gateway (five parties;
  BLAKE2b-8 big-endian & 2^63-1; `HMACBlake2B(64)` without a key is standard BLAKE2b,
  and the digest length is an algorithm parameter, not truncation): the same string id yields the same u64 in any language,
  so deletes can be replayed across SDKs;
- Non-2xx server responses uniformly raise `HouyiException` (`StatusCode` + server diagnostic text);
- Collection names are path-escaped per RFC 3986 (`Uri.EscapeDataString`, same semantics as Python
  `quote(safe="")`), so names containing `/ ? #` or Chinese characters do not alter routing.

## Tests

```bash
dotnet test   # StableId 五方对拍 + 集成（打 127.0.0.1:17000 真实服务，
              # HOUYIDB_URL can override; skipped automatically when unreachable)
```

Integration prerequisite: `./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
