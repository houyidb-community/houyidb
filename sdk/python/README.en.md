# HouyiDB Python Ecosystem

> [中文](README.md)

Three packages, each with a distinct role:

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
    hnsw=HnswParams(m=16, ef_construction=100, ef_search=64),
)
created["metric"]  # "ip"; the server echoes the effective parameters
```

Metric support matrix: `hnsw` / `flat` / `clustered` / `flat-clustered`
support all four; `pq-flat` supports `l2` only; `hii-l0` supports `cosine`
only.

Search scores are metric-derived (higher means more similar): `cosine` →
`1 - distance` in [-1, 1]; `l2` / `l1` → `1 - distance` (unclamped); `ip` →
the raw dot product.

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
