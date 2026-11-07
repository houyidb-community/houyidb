# houyidb (TypeScript/JavaScript)

> [English](README.en.md)

HouyiDB 向量数据库的官方 TS/JS 客户端。Node ≥ 18（全局 fetch），运行时仅一个依赖（`@noble/hashes`，用于跨语言一致的 `stableId`）。

```bash
npm install houyidb
```

## 快速开始

```ts
import { HouyiClient, stableId } from "houyidb";

const client = new HouyiClient("http://127.0.0.1:7000");
const col = client.collection("docs");

// 写入（collection 首次写入隐式创建，维度由首个向量定型）
await col.upsert([
  { id: stableId("doc-1"), values: embedding, payload: { text: "...", category: "fruit" } },
]);

// 检索（score 由度量换算、越大越相似，详见下文「显式建集合」）
const result = await col.search(queryEmbedding, 5, { includePayload: true });

// 混合检索（filters 取交集；op: eq/gt/gte/lt/lte/in）
const filtered = await col.hybridSearch(queryEmbedding, 5, {
  filters: [{ column: "category", op: "eq", value: "fruit" }],
  includePayload: true,
});

// 删除（按 id 和/或过滤条件，幂等）
await col.delete({ ids: [stableId("doc-1")] });
await col.delete({ filters: [{ column: "category", op: "eq", value: "fruit" }] });
```

## 关键设计：向量 id 是 bigint

HouyiDB 的 id 键空间是 u64，超出 JS Number 安全域（2^53-1）。本 SDK：

- `SearchHit.id` 一律为 `bigint`；`stableId()` 返回 `bigint`；
- 写入侧接受 `bigint | number | string` 并归一；
- 请求序列化与响应解析内部做 2^53 防腐蚀处理——**不要**自己 `JSON.parse` 检索响应再比较 id，请使用 SDK 返回的 `SearchResult`。

`stableId` 与 Python SDK、服务端 OpenAI 网关三方同算法（BLAKE2b-8 大端 & 2^63-1）：同一字符串 id 在任何语言得到同一 u64，删除可跨 SDK 重放。

## 显式建集合（维度/度量/后端/HNSW 一次定型）

`PUT /v1/collections/{name}` 是普通写路径（写权限，不需要 admin token）。
显式创建会锁定维度与度量：同名已存在 → `HouyiError` statusCode 409；度量与
后端创建后不可变，省略字段走服务端默认（`cosine` + `hnsw`）。

```ts
const created = await client.createCollection("docs", 768, {
  metric: "ip",          // l2 / cosine / ip / l1
  indexType: "hnsw",     // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
  hnsw: { m: 16, efConstruction: 100, efSearch: 64 },
});
created.metric; // "ip"；服务端回显实际生效参数
```

度量支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持全部四种；
`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。

检索分数按度量换算（越大越相似）：`cosine` → `1 - distance` ∈ [-1, 1]；
`l2` / `l1` → `1 - distance`（不截断）；`ip` → 点积本身。

## 管理操作：删除整个 collection（不可逆）

需服务端管理员权限（RBAC `Action::Admin`）。删除走 WAL-first：崩溃重放与
follower 复制都收敛到「已删除」。**不可逆**，删除前请先导出备份。同名重建
得到全新的空 collection，旧向量不会复活。

```ts
// adminToken 只附加在 /v1/admin/* 请求上，普通读写不携带
const admin = new HouyiClient(url, { adminToken });
await admin.deleteCollection("docs"); // 不存在 → HouyiError statusCode 404
```

EN: Admin-only on the server. WAL-first, so crash replay and follower
replication converge to "gone". **Irreversible**; export first. A same-name
recreate starts empty and never resurrects old vectors.

## 与 Python SDK 的 API 对应

| Python | TypeScript |
|---|---|
| `collection.upsert(vectors)` | `col.upsert(vectors)` |
| `collection.search(v, k, ef_search=, include_payload=)` | `col.search(v, k, { efSearch, includePayload })` |
| `collection.hybrid_search(v, k, filters=, ...)` | `col.hybridSearch(v, k, { filters, ... })` |
| `collection.delete(ids=, filters=)` | `col.delete({ ids, filters })` |
| `collection.status()` | `col.status()` |
| `client.health() / list_collections() / index_status()` | `client.health() / listCollections() / indexStatus()` |
| `stable_id(text) -> int` | `stableId(text) -> bigint` |
| `HouyiError(status_code)` | `HouyiError(statusCode)` |

## 测试

```bash
npm test          # 单元（stable_id 三方对拍）+ 集成（打 127.0.0.1:17000 真实服务，HOUYIDB_URL 可覆盖）
npm run build     # 产出 dist/（ESM + d.ts）
```

集成前置：`./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
