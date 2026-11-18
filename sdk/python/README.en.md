# HouyiDB Python Ecosystem

> [中文](README.md)

Requires Python ≥ 3.10. Three packages, each with a distinct role:

| Package | Role | Dependencies |
|----|------|------|
| `houyidb` | collections HTTP API client (transport layer) | `requests` |
| `langchain-houyidb` | LangChain `VectorStore` adapter (semantic dialect layer) | `langchain-core` + `houyidb` |
| `llama-index-vector-stores-houyidb` | LlamaIndex `BasePydanticVectorStore` adapter | `llama-index-core` + `houyidb` |

The SDK owns the wire protocol (endpoints, JSON, error mapping, deterministic str→u64 id hashing);
the adapters own framework semantics (Document/node models, embedding delegation, filter DSL translation,
score conventions, lifecycle contracts). Embeddings are always computed by the caller/framework; HouyiDB only stores and retrieves vectors.

## Installation (Development)

```bash
pip install -e sdk/python/houyidb \
            -e sdk/python/langchain-houyidb \
            -e sdk/python/llama-index-vector-stores-houyidb
```

## Minimal Example

```python
from houyidb import HouyiClient, Vector, Filter

client = HouyiClient("http://127.0.0.1:7000")
col = client.collection("docs")  # 首次 upsert 隐式创建
col.upsert([Vector(id=1, values=emb, payload={"text": "apple", "price": 5})])

hits = col.search(query_emb, 3, include_payload=True)          # 纯向量
hits = col.hybrid_search(query_emb, 3,                         # 向量 × 过滤
                         filters=[Filter("price", "lt", 10)], include_payload=True)
col.delete([1])                                                # 按 id
col.delete(filters=[Filter("price", "lt", 10)])                # 按条件（见下）
```

LangChain / LlamaIndex usage is documented in each package's README.

## Explicit Creation (Dims/Metric/Backend/HNSW Fixed Up Front)

`PUT /v1/collections/{name}` is a normal write path (write permission, no admin
token). Explicit creation locks dimensionality and metric: an existing name
yields `HouyiError(status_code=409)`; metric/backend are immutable afterwards,
and omitted fields take server defaults (`cosine` + `hnsw`).

```python
from houyidb import HnswParams

created = client.create_collection(
    "docs", 768,
    metric="ip",          # l2 / cosine / ip / l1
    index_type="hnsw",    # hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    vector_dtype="f16",   # optional: f32 (default) / f16 (hnsw only, main-store memory halved)
    quantization="int8",  # optional: none (default) / int8 / rabitq (hnsw only, graph-side compression)
    sparse="bm25",        # optional: none (default) / bm25 (hnsw/flat only, immutable after creation)
    hnsw=HnswParams(m=16, ef_construction=100, ef_search=64),
)
created["metric"]        # "ip"; the server echoes the effective parameters
created.get("vector_dtype")  # "f16"; None when an older server omits it
created.get("quantization")  # "int8"; None when an older server omits it
created.get("sparse")    # "bm25"; None when an older server omits it
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

`vector_dtype` allows `f16` for the hnsw backend only and is immutable after
creation; queries stay f32 and distances are computed against f16-rounded data
(asymmetric quantization).

`quantization` is independent of `vector_dtype`: `int8` (1 B/dim) and
`rabitq` (1 bit/dim) compress only the HNSW graph-side copy while the main
store keeps the raw vectors, and search reranks exactly on those after taking
graph candidates, so final top-k precision is unchanged. `rabitq` supports
`metric = l2 | cosine` only (`ip`/`l1` return 400). It is hnsw-only and
immutable after creation; unset options keep the key out of the request body.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

## Sparse Vectors (BM25) and Fusion Search (P4)

The sparse index is an **explicit creation switch**: `sparse="bm25"` works on
`hnsw` / `flat` backends only and is immutable after creation; implicitly
created collections are always `none`, and writing `sparse` on a `sparse=none`
collection (or combining `bm25` with another backend at creation) returns 400.
When an older server omits the field in the creation response,
`created.get("sparse")` is `None` and is tolerated as an optional field.

Terms are client-supplied (no tokenizer): `(term_id: int, weight: float)`,
term ∈ [1, 2^24−1], weight finite and > 0; at most 1024 terms per document
(the server sums duplicate terms), and at most 64 terms per fusion query.

```python
# Sparse writes ride the dense batch (whole-document replacement): None (default) = unchanged, [] = clear
col.upsert([Vector(id=2, values=emb, sparse=[(7, 1.0), (8, 0.8)])])

# Fusion search: dense and sparse are both required; the k limit matches search
fused = col.fusion_search(
    query_emb, [(7, 1.0), (8, 0.8)], 10,
    fusion="rrf",        # rrf (default) / weighted
    rrf_k=60,            # 1..=1000, rrf only
    # alpha=0.5,         # 0..=1, weighted only (default 0.5)
    filters=[Filter("category", "eq", "fruit")],
    include_payload=True,
)
for hit in fused.hits:
    # hit.score only expresses fused ranking; hit.dense_score is metric-converted and hit.sparse_score is raw BM25
    print(hit.id, hit.score, hit.dense_score, hit.sparse_score)
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

## Write Visibility (Dual-Speed Writes)

Batch upserts first sit in the asynchronous graph-build queue (builder_queue) and become
visible to **search** only after a flush (typically tens of milliseconds in normal operation);
scalar rows, however, are written to their pages **synchronously** with the upsert, therefore:

- `delete(filters=...)` is evaluated against scalar pages, so freshly written vectors can be deleted by condition immediately—
  do not use the "search first for ids, then delete" pattern (search cannot see vectors still in the queue, so deletes will miss them);
- when determinism is required before asserting on search results, poll `col.status()["pending_count"] == 0`
  (test helper `sdk/python/tests/conftest.py::wait_flushed`).

## OpenAI-Compatible Gateway (Optional)

Started with `--openai-upstream <upstream base_url>`, the server exposes an OpenAI-shaped
`POST /v1/embeddings`: without `collection` it is a transparent proxy ("just change the base_url");
with `collection` (passed by the SDK via `extra_body`) it also ingests the embeddings—
"change the URL and data is ingested":

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:17000/v1", api_key="not-needed")
resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["apple is a fruit"],
    extra_body={"collection": "docs"},
)
resp.houyi  # {"collection": "docs", "upserted": 1} —— id = stable_id(text)，重跑幂等
```

## Admin Operation: Dropping a Whole Collection (Irreversible)

Requires server admin permission (RBAC `Action::Admin`). Deletion is WAL-first,
so crash replay and follower replication converge to "gone". **Irreversible**;
export via `GET /v1/collections/{name}/export` first. A same-name recreate starts
empty and never resurrects old vectors. `admin_token` is attached only to
`/v1/admin/*` requests; normal reads/writes never carry it.

```python
admin = HouyiClient(base_url, admin_token=admin_token)
admin.delete_collection("docs")  # unknown → HouyiError(status_code=404)
```

## Tests

```bash
# Requires a server running the new binary (with delete/include_payload/OpenAI gateway)
cargo build --release -p houyi-server
.venv-eco/bin/python sdk/python/tests/fake_openai_upstream.py &   # 假 OpenAI 上游 :18090
./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 \
    --pgwire-addr 127.0.0.1:15432 --wal-path /tmp/houyidb-eco.wal \
    --openai-upstream http://127.0.0.1:18090

.venv-eco/bin/python -m pytest sdk/python/tests -v   # HOUYIDB_URL 可覆盖默认地址
```

## Common Known Limitations

- Collection payload scalar pages are volatile in v1: payload/filter/delete tombstones are lost after a process restart,
  while the vector index (WAL/snapshot) is unaffected.
- The payload/metadata key `label` is a reserved column.
- Filter operators are limited to `eq/gt/gte/lt/lte/in`, and multiple conditions are ANDed; only the hnsw/flat backends support
  filtering and deletion (other backends fail fast with 400).
- Vector delete is an in-memory operation (same semantics as scalar pages; not written to the WAL/snapshot);
  whole-collection `delete_collection` is an admin operation recorded in the WAL (`collection_drop`) and replicated to followers.
