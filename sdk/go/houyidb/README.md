# houyidb-go

> [English](README.en.md)

HouyiDB 向量数据库的官方 Go 客户端。仅一个外部依赖（`golang.org/x/crypto`，用于跨语言一致的 `StableID`），Go ≥ 1.17。

```bash
go get github.com/houyidb-community/houyidb/sdk/go/houyidb
```

## 快速开始

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

    // 写入（collection 首次写入隐式创建，维度由首个向量定型）
    _, _ = col.Upsert(ctx, []houyidb.Vector{{
        ID:      houyidb.StableID("doc-1"),
        Values:  embedding, // []float64
        Payload: map[string]interface{}{"text": "...", "category": "fruit"},
    }})

    // 检索（Score 由度量换算、越大越相似，详见下文「显式建集合」）
    result, _ := col.Search(ctx, queryEmbedding, 5,
        &houyidb.SearchOptions{IncludePayload: true})

    // 混合检索（Filters 取交集；Op: eq/gt/gte/lt/lte/in）
    filtered, _ := col.HybridSearch(ctx, queryEmbedding, 5, &houyidb.HybridSearchOptions{
        Filters:        []houyidb.Filter{{Column: "category", Op: houyidb.OpEq, Value: "fruit"}},
        IncludePayload: true,
    })
    _, _ = result, filtered

    // 删除（按 id 和/或过滤条件，幂等）
    _, _ = col.Delete(ctx, &houyidb.DeleteOptions{IDs: []uint64{houyidb.StableID("doc-1")}})
}
```

## 显式建集合（维度/度量/后端/HNSW 一次定型）

`PUT /v1/collections/{name}` 是普通写路径（写权限，不需要 admin token）。
显式创建会锁定维度与度量：同名已存在返回 409，度量与后端创建后不可变；
零值选项等价服务端默认（`cosine` + `hnsw`）。

```go
created, err := client.CreateCollection(ctx, "docs", 768, &houyidb.CreateCollectionOptions{
    Metric:    "ip",   // l2 / cosine / ip / l1
    IndexType: "hnsw", // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    Hnsw:      &houyidb.HnswConfig{M: 16, EFConstruction: 100, EFSearch: 64},
})
// created.Metric == "ip"，created.Hnsw 回显实际生效的 HNSW 参数
```

度量支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持全部四种；
`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。

检索分数按度量换算（越大越相似）：`cosine` → `1 - distance` ∈ [-1, 1]；
`l2` / `l1` → `1 - distance`（不截断）；`ip` → 点积本身。

## 管理操作：删除整个 collection（不可逆）

需服务端管理员权限（RBAC `Action::Admin`）。删除走 WAL-first：崩溃重放与
follower 复制都收敛到「已删除」。**不可逆**，删除前请先导出备份。同名重建
得到全新的空 collection，旧向量不会复活。

```go
client := houyidb.NewClient(url, houyidb.WithAdminToken(adminToken))
if err := client.DeleteCollection(ctx, "docs"); err != nil {
    // 不存在 → *houyidb.HouyiError，StatusCode == 404
}
```

EN: Admin-only on the server. WAL-first, so crash replay and follower
replication converge to "gone". **Irreversible**; export first. A same-name
recreate starts empty and never resurrects old vectors.

## 与其他语言 SDK 的对应

- 向量 id 是 `uint64`——Go 原生覆盖键空间，无 JS 的 2^53 精度问题；
- `StableID` 与 Python/TS SDK、服务端 OpenAI 网关四方同算法
  （BLAKE2b-8 大端 & 2^63-1）：同一字符串 id 在任何语言得到同一 u64，删除可跨 SDK 重放；
- 错误统一为 `*houyidb.HouyiError`（`StatusCode` + 服务端诊断文本），`errors.As` 可判。

## 测试

```bash
go test -race ./...   # StableID 四方对拍 + 集成（打 127.0.0.1:17000 真实服务，HOUYIDB_URL 可覆盖；不可达自动 Skip）
```

集成前置：`./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
