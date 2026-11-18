# houyidb-java

> [中文](README.md)

The official Java client for the HouyiDB vector database (collections HTTP API). Synchronous blocking
(`java.net.http.HttpClient`), Java 17+, with only two external dependencies: BouncyCastle
(cross-language-consistent `StableId`) and Jackson (JSON).

```groovy
dependencies {
    implementation 'top.houyidb:houyidb-client:0.2.0-beta.8'
}
```

## Quick Start

```java
import com.houyidb.client.*;
import com.houyidb.client.Types.*;

var client = new HouyiClient("http://127.0.0.1:7000");
var col = client.collection("docs");

// Write (the collection is created implicitly on first write; dimensionality is fixed by the first vector)
col.upsert(List.of(new Vector(
        StableId.of("doc-1"),
        embedding, // double[]
        Map.of("text", "...", "category", "fruit"))));

// Search (scores are metric-derived; higher means more similar — see "Explicit Creation" below)
SearchResult result = col.search(queryEmbedding, 5,
        new SearchOptions().includePayload(true));

// Hybrid search (filters are intersected; FilterOp: EQ/GT/GTE/LT/LTE/IN)
SearchResult filtered = col.hybridSearch(queryEmbedding, 5,
        new SearchOptions().includePayload(true)
                .filter("category", FilterOp.EQ, "fruit"));

// Delete (by id and/or filter conditions, idempotent; the two are unioned)
col.delete(List.of(StableId.of("doc-1")), null);
```

## Explicit Creation (Dims/Metric/Backend/HNSW Fixed Up Front)

`PUT /v1/collections/{name}` is a normal write path (write permission, no admin
token). Explicit creation locks dimensionality and metric: an existing name
yields `HouyiException(409)`; metric/backend are immutable afterwards, and
omitted fields take server defaults (`cosine` + `hnsw`).

```java
var created = client.createCollection("docs", 768,
        new CreateCollectionOptions()
                .metric("ip")            // l2 / cosine / ip / l1
                .indexType("hnsw")       // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
                .vectorDtype("f16")      // optional: f32 (default) / f16 (hnsw only, half the memory)
                .quantization("int8")    // optional: none (default) / int8 / rabitq (hnsw only, graph-side compression)
                .sparse("bm25")          // optional: none (default) / bm25 (hnsw/flat only, immutable after creation)
                .hnsw(new HnswParams(16, 100, 64)));
// created.metric == "ip"; created.vectorDtype == "f16"; created.quantization == "int8";
// created.sparse == "bm25" (null when an older server omits it); the server echoes the effective parameters
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

`vectorDtype` allows `f16` for the hnsw backend only and is immutable after
creation; queries stay f32 and distances are computed against f16-rounded data
(asymmetric quantization). An older server that omits the field yields `null`;
never default it to `f32` yourself.

`quantization` is independent of `vectorDtype`: `int8` (1 B/dim) and
`rabitq` (1 bit/dim) compress only the HNSW graph-side copy while the main
store keeps the raw vectors, and search reranks exactly on those after taking
graph candidates, so final top-k precision is unchanged. `rabitq` supports
`metric = l2 | cosine` only (`ip`/`l1` return 400). It is hnsw-only and
immutable after creation; unset
options keep the key out of the request body, and an older server that omits
the field yields `null`.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

## Sparse Vectors (BM25) and Fusion Search (P4)

The sparse index is an **explicit creation switch**: `.sparse("bm25")` works on
`hnsw` / `flat` backends only and is immutable after creation; implicitly
created collections are always `none`, and writing `sparse` on a `sparse=none`
collection (or combining `bm25` with another backend at creation) returns 400.
When an older server omits the field in the creation response,
`created.sparse` is `null` and is tolerated as an optional field.

Terms are client-supplied (no tokenizer): `new SparseTerm(termId, weight)`,
term ∈ [1, 2^24−1], weight finite and > 0; at most 1024 terms per document
(the server sums duplicate terms), and at most 64 terms per fusion query.

```java
// Sparse writes ride the dense batch (whole-document replacement): null = unchanged, empty list = clear
col.upsert(List.of(new Vector(
        StableId.of("doc-2"),
        embedding,
        Map.of("category", "fruit"),
        List.of(new SparseTerm(7, 1.0), new SparseTerm(8, 0.8)))));

// Fusion search: dense and sparse are both required; the k limit matches search
FusionResult fused = col.fusionSearch(
        queryEmbedding, List.of(new SparseTerm(7, 1.0), new SparseTerm(8, 0.8)), 10,
        new FusionOptions()
                .fusion("rrf")                        // rrf (default) / weighted
                .rrfK(60)                             // 1..=1000, rrf only
                // .alpha(0.5)                        // 0..=1, weighted only (default 0.5)
                .filter("category", FilterOp.EQ, "fruit")
                .includePayload(true));
for (FusionHit hit : fused.hits) {
    // hit.score only expresses fused ranking; hit.denseScore is metric-converted and hit.sparseScore is raw BM25
    System.out.println(hit.id + " " + hit.score + " " + hit.denseScore + " " + hit.sparseScore);
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
first. A same-name recreate starts empty and never resurrects old vectors. The
admin token is attached only to `/v1/admin/*` requests.

```java
HouyiClient admin = HouyiClient.withAdminToken(baseUrl, adminToken);
admin.deleteCollection("docs"); // unknown → HouyiException(404)
```

## Mapping to Other Language SDKs

- Vector ids are `long`—stable_id falls within [0, 2^63-1], carried losslessly by signed long,
  avoiding JS's 2^53 precision issue;
- `StableId.of` uses the same algorithm as the Python/TS/Go/Rust/.NET SDKs and the server OpenAI gateway (six parties;
  BLAKE2b-8 big-endian & 2^63-1; the digest length of `Blake2bDigest(64)` is an algorithm parameter,
  not truncation): the same string id yields the same u64 in any language, so deletes can be replayed across SDKs;
- Non-2xx server responses uniformly raise `HouyiException` (`statusCode()` + server diagnostic text),
  while transport-layer failures remain `IOException`;
- Collection names are path-escaped per RFC 3986 (same semantics as Python `quote(safe="")`),
  so names containing `/ ? #` or Chinese characters do not alter routing.

## Tests

```bash
gradle test   # StableId 六方对拍 + 路径转义 + 集成（打 127.0.0.1:17000 真实服务，
              # HOUYIDB_URL can override; skipped automatically via JUnit Assumptions when unreachable)
```

Integration prerequisite: `./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
