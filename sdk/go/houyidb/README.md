# houyidb-go

> [English](README.en.md)

HouyiDB 向量数据库的官方 Go 客户端。仅一个外部依赖（`golang.org/x/crypto`，用于跨语言一致的 `StableID`），Go ≥ 1.25（安全修复后的 x/crypto 要求）。

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
    // Sparse 仅在 sparse=bm25 的集合使用：nil = 不改，空切片 = 清空
    _, _ = col.Upsert(ctx, []houyidb.Vector{{
        ID:      houyidb.StableID("doc-1"),
        Values:  embedding, // []float64
        Payload: map[string]interface{}{"text": "...", "category": "fruit"},
        Sparse:  nil,
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
    Metric:       "ip",   // l2 / cosine / ip / l1
    IndexType:    "hnsw", // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    VectorDtype:  "f16",  // 可选：f32（缺省）/ f16（仅 hnsw，主表内存减半）
    Quantization: "int8", // 可选：none（缺省）/ int8 / rabitq（仅 hnsw，压缩图侧）
    Sparse:       "bm25", // 可选：none（缺省）/ bm25（仅 hnsw/flat，创建后不可变）
    Hnsw:         &houyidb.HnswConfig{M: 16, EFConstruction: 100, EFSearch: 64},
})
// created.Metric == "ip"，created.Hnsw 回显实际生效的 HNSW 参数，
// created.VectorDtype/Quantization/Sparse 回显 f16/int8/bm25（旧服务端未配置时为空串）
```

度量支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持全部四种；
`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。

`VectorDtype` 仅 hnsw 后端可指定 `f16`，创建后不可变；查询向量仍为 f32，
距离按 f16 舍入后的数据计算（非对称量化）。旧服务端不回显该字段时为空串。

`Quantization` 与 `VectorDtype` 相互独立：`int8`（1 B/dim）与 `rabitq`
（1 bit/dim）只压缩 HNSW 图侧副本，主表仍保留原始向量，检索时先取图候选
再精确重排，最终 top-k 精度不降。`rabitq` 仅支持 `Metric = l2 | cosine`
（`ip`/`l1` 返回 400）。仅 hnsw 后端可用，创建后不可变；未设置时请求体
不出现该键。旧服务端不回显时为空串。

检索分数按度量换算（越大越相似）：`cosine` → `1 - distance` ∈ [-1, 1]；
`l2` / `l1` → `1 - distance`（不截断）；`ip` → 点积本身。

## 稀疏向量（BM25）与融合检索（P4）

稀疏索引是**显式创建开关**：`Sparse: "bm25"` 仅 `hnsw` / `flat` 后端可用，
创建后不可变；隐式创建的集合恒为 `none`，在 `sparse=none` 集合上写入
`Sparse`（或创建时组合其他后端）一律 400。旧服务端创建响应不回显该字段时
`created.Sparse` 为空串，按可选字段容忍。

词项由客户端给定（无分词器）：`SparseTerm` 为 `[2]float64`（`{term_id,
weight}`），term ∈ [1, 2^24−1] 且必须为整数，weight 有限且 > 0；每文档
≤ 1024 个词项（重复 term 服务端求和），融合查询 ≤ 64 个词项。

```go
// 稀疏写入与 dense 同批（整文档替换语义）：nil = 不改，空切片 = 清空
_, _ = col.Upsert(ctx, []houyidb.Vector{{
    ID:     houyidb.StableID("doc-2"),
    Values: embedding,
    Sparse: []houyidb.SparseTerm{{7, 1.0}, {8, 0.8}},
}})

// 融合检索：dense 与 sparse 必须同时非空；k 上限与 Search 一致
fused, err := col.FusionSearch(ctx, queryEmbedding,
    []houyidb.SparseTerm{{7, 1.0}, {8, 0.8}}, 10,
    &houyidb.FusionSearchOptions{
        Fusion:         "rrf", // rrf（缺省）/ weighted
        RRFK:           60,    // 1..=1000，仅 rrf 生效
        Filters:        []houyidb.Filter{{Column: "category", Op: houyidb.OpEq, Value: "fruit"}},
        IncludePayload: true,
        // Alpha: &alpha,      // 0..=1，仅 weighted 生效（缺省 0.5；指针区分「未给」与显式 0）
    })
// fused.Hits[i].Score 仅表达融合排序；DenseScore 按集合度量换算（仅稠密命中），
// SparseScore 为 BM25 原始分（仅稀疏命中）；fused.Metric 只解释 DenseScore。
```

融合两种模式：`rrf`（默认，`score = Σ 1/(rrf_k + rank)`）、`weighted`
（两路各 min-max 归一化后 `alpha*dense + (1-alpha)*sparse`）。两路候选均按
`4*k` 过取并按 Filters 过滤；未命中任一路的文档不入选。显式给出的
`RRFK`/`Alpha` 服务端一律严格校验（越界 400，即使当前模式不参与计算）。

融合检索是纯读，不隐式建集合；`sparse=none` 集合返回 400；旧服务端没有该
端点（404 原样透出）。稀疏**写入**仅 HTTP/SDK 提供（pgwire 仅支持融合查询）。

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
