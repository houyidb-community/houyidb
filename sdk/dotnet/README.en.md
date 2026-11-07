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
    Hnsw = new HnswParams(M: 16, EfConstruction: 100, EfSearch: 64),
});
// created.Metric == "ip"; the server echoes the effective parameters
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
