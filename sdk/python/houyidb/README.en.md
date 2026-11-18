# houyidb — HouyiDB Python SDK

> [中文](README.md)

A thin wrapper around the HouyiDB collections HTTP API: transport + types, with no framework semantics. Requires Python ≥ 3.10.

```python
from houyidb import HouyiClient, Vector, Filter

client = HouyiClient("http://127.0.0.1:7000")
col = client.collection("goods")          # 首次 upsert 时隐式创建

col.upsert([
    Vector(id=1, values=[0.9, 0.1], payload={"name": "apple", "price": 3.5}),
    Vector(id=2, values=[0.1, 0.9], payload={"name": "laptop", "price": 8999}),
])

res = col.hybrid_search(
    [0.85, 0.15], k=3,
    filters=[Filter("price", "lt", 1000)],
    include_payload=True,
)
for hit in res.hits:
    print(hit.id, hit.score, hit.payload)

col.delete([1])                                # 按 id
col.delete(filters=[Filter("price", "lt", 100)])  # 按条件（标量页求值，刚写入即可命中）
```

## Contract Highlights

- **Similarity score**: `1 - cosine_distance`; larger is more similar (cosine backend).
- **Filter operators**: `eq` / `gt` / `lt` / `gte` / `lte` / `in`; multiple filters are intersected.
- **Reserved key**: payload must not use the `label` key (400); label is passed as a top-level field.
- **ID space**: u64; `houyidb.stable_id(str)` deterministically maps a string id into u64.
- **v1 volatility**: collection payload/scalar pages are not written to the WAL and are lost after a process restart (the vector index is unaffected).
- **Errors**: any non-2xx response raises `HouyiError(status_code, message)`.

## Ecosystem Adapters

- `langchain-houyidb` — LangChain `VectorStore` implementation
- `llama-index-vector-stores-houyidb` — LlamaIndex `BasePydanticVectorStore` implementation
