# houyidb — HouyiDB Python SDK

> [English](README.en.md)

HouyiDB collections HTTP API 的薄封装：传输 + 类型，不含框架语义。需要 Python ≥ 3.10。

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

## 契约要点

- **相似度分数**：`1 - cosine_distance`，越大越相似（余弦后端）。
- **过滤算子**：`eq` / `gt` / `lt` / `gte` / `lte` / `in`，多条取交集。
- **保留键**：payload 不得使用 `label` 键（400），label 走顶层字段。
- **id 空间**：u64；`houyidb.stable_id(str)` 可把字符串 id 确定性映射进 u64。
- **v1 易失性**：collection 的 payload/标量页不写 WAL，进程重启后丢失（向量索引不受影响）。
- **错误**：非 2xx 一律抛 `HouyiError(status_code, message)`。

## 生态 adapter

- `langchain-houyidb` — LangChain `VectorStore` 实现
- `llama-index-vector-stores-houyidb` — LlamaIndex `BasePydanticVectorStore` 实现
