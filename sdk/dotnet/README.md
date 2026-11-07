# houyidb-dotnet

> [English](README.en.md)

HouyiDB 向量数据库的官方 .NET 客户端（collections HTTP API）。async/await 全异步，
目标框架 net8.0 + net10.0，唯一外部依赖是 `Konscious.Security.Cryptography.Blake2`
（用于跨语言一致的 `StableId`）。

```bash
dotnet add package HouyiDb
```

## 快速开始

```csharp
using HouyiDb;

var client = new HouyiClient("http://127.0.0.1:7000");
var col = client.GetCollection("docs");

// 写入（collection 首次写入隐式创建，维度由首个向量定型）
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

// 检索（Score 由度量换算、越大越相似，详见下文「显式建集合」）
var result = await col.SearchAsync(queryEmbedding, 5,
    new SearchOptions { IncludePayload = true });

// 混合检索（Filters 取交集；FilterOp: Eq/Gt/Gte/Lt/Lte/In）
var options = new SearchOptions { IncludePayload = true };
options.Filters.Add(new Filter("category", FilterOp.Eq, "fruit"));
var filtered = await col.HybridSearchAsync(queryEmbedding, 5, options);

// 删除（按 id 和/或过滤条件，幂等；两者取并集）
await col.DeleteAsync(ids: new[] { StableId.Of("doc-1") });
```

## 显式建集合（维度/度量/后端/HNSW 一次定型）

`PUT /v1/collections/{name}` 是普通写路径（写权限，不需要 admin token）。
显式创建会锁定维度与度量：同名已存在 → `HouyiException(409)`；度量与后端
创建后不可变，省略字段走服务端默认（`cosine` + `hnsw`）。

```csharp
var created = await client.CreateCollectionAsync("docs", 768, new CreateCollectionOptions
{
    Metric = "ip",          // l2 / cosine / ip / l1
    IndexType = "hnsw",     // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    Hnsw = new HnswParams(M: 16, EfConstruction: 100, EfSearch: 64),
});
// created.Metric == "ip"；服务端回显实际生效参数
```

度量支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持全部四种；
`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。

检索分数按度量换算（越大越相似）：`cosine` → `1 - distance` ∈ [-1, 1]；
`l2` / `l1` → `1 - distance`（不截断）；`ip` → 点积本身。

## 管理操作：删除整个 collection（不可逆）

需服务端管理员权限（RBAC `Action::Admin`）。删除走 WAL-first：崩溃重放与
follower 复制都收敛到「已删除」。**不可逆**，删除前请先导出备份。同名重建
得到全新的空 collection，旧向量不会复活。adminToken 只附加在 `/v1/admin/*`
请求上，普通读写不携带。

```csharp
var admin = HouyiClient.WithAdminToken(baseUrl, adminToken);
await admin.DeleteCollectionAsync("docs"); // 不存在 → HouyiException(404)
```

EN: Admin-only on the server. WAL-first, so crash replay and follower
replication converge to "gone". **Irreversible**; export first. A same-name
recreate starts empty and never resurrects old vectors.

## 与其他语言 SDK 的对应

- 向量 id 是 `ulong`——原生覆盖键空间，无 JS 的 2^53 精度问题；
- `StableId.Of` 与 Python/TS/Go/Rust SDK、服务端 OpenAI 网关五方同算法
  （BLAKE2b-8 大端 & 2^63-1；`HMACBlake2B(64)` 无 key 构造即标准 BLAKE2b，
  digest 长度是算法参数而非截断）：同一字符串 id 在任何语言得到同一 u64，
  删除可跨 SDK 重放；
- 服务端非 2xx 统一抛 `HouyiException`（`StatusCode` + 服务端诊断文本）；
- collection 名经 RFC 3986 路径转义（`Uri.EscapeDataString`，与 Python
  `quote(safe="")` 同口径），含 `/ ? #` 或中文的名字不会改写路由。

## 测试

```bash
dotnet test   # StableId 五方对拍 + 集成（打 127.0.0.1:17000 真实服务，
              # HOUYIDB_URL 可覆盖；不可达自动跳过）
```

集成前置：`./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
