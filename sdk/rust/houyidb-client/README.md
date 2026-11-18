# houyidb-client

> [English](README.en.md)

HouyiDB 向量数据库的官方 Rust 客户端（collections HTTP API）。同步阻塞式（`ureq` + rustls），仅 4 个直接依赖，与服务端共用同一套 `stable_id` 算法与依赖版本。

```toml
[dependencies]
houyidb-client = { path = "sdk/rust/houyidb-client" }
```

## 快速开始

```rust
use houyidb_client::{stable_id, Client, Filter, FilterOp, SearchOptions, Vector};
use std::collections::HashMap;

let client = Client::new("http://127.0.0.1:7000");
let col = client.collection("docs");

// 写入（collection 首次写入隐式创建，维度由首个向量定型）
// sparse 仅在 sparse=bm25 的集合使用：None = 不改，Some([]) = 清空
col.upsert(&[Vector {
    id: stable_id("doc-1"),
    values: embedding, // Vec<f32>
    label: None,
    payload: Some(HashMap::from([
        ("text".into(), "...".into()),
        ("category".into(), "fruit".into()),
    ])),
    sparse: None,
}])?;

// 检索（score 由度量换算、越大越相似，详见下文「显式建集合」）
let result = col.search(&query, 5, &SearchOptions {
    include_payload: true,
    ..Default::default()
})?;

// 混合检索（filters 取交集；op: eq/gt/gte/lt/lte/in）
let filtered = col.hybrid_search(&query, 5, &SearchOptions {
    filters: vec![Filter::new("category", FilterOp::Eq, "fruit")],
    include_payload: true,
    ..Default::default()
})?;

// 删除（按 id 和/或过滤条件，幂等；两者取并集）
col.delete(&[stable_id("doc-1")], &[])?;
```

## 显式建集合（维度/度量/后端/HNSW 一次定型）

`PUT /v1/collections/{name}` 是普通写路径（写权限，不需要 admin token）。
显式创建会锁定维度与度量：同名已存在返回 409，度量与后端创建后不可变；
省略的字段走服务端默认（`cosine` + `hnsw`）。

```rust
use houyidb_client::{Client, CreateCollectionOptions, HnswConfig};

let created = client.create_collection("docs", 768, &CreateCollectionOptions {
    metric: Some("ip".into()),          // l2 / cosine / ip / l1
    index_type: Some("hnsw".into()),    // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
    vector_dtype: Some("f16".into()),   // 可选：f32（缺省）/ f16（仅 hnsw，主表内存减半）
    quantization: Some("int8".into()),  // 可选：none（缺省）/ int8 / rabitq（仅 hnsw，压缩图侧）
    sparse: Some("bm25".into()),        // 可选：none（缺省）/ bm25（仅 hnsw/flat，创建后不可变）
    hnsw: Some(HnswConfig { m: 16, ef_construction: 100, ef_search: 64 }),
})?;
assert_eq!(created.metric, "ip");
assert_eq!(created.vector_dtype.as_deref(), Some("f16"));
assert_eq!(created.quantization.as_deref(), Some("int8"));
assert_eq!(created.sparse.as_deref(), Some("bm25"));
```

度量支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持全部四种；
`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。

`vector_dtype` 仅 hnsw 后端可指定 `f16`，创建后不可变；查询向量仍为 f32，
距离按 f16 舍入后的数据计算（非对称量化）。旧服务端不回显该字段时结果为
`None`，调用方无需兜底成 `f32`。

`quantization` 与 `vector_dtype` 相互独立：`int8`（1 B/dim）与 `rabitq`
（1 bit/dim）只压缩 HNSW 图侧副本，主表仍保留原始向量，检索时先取图候选
再精确重排，最终 top-k 精度不降。`rabitq` 仅支持 `metric = l2 | cosine`
（`ip`/`l1` 返回 400）。仅 hnsw 后端可用，创建后不可变；未知值与显式非
hnsw 组合由服务端 400 拒绝。旧服务端不回显时结果为 `None`；未设置该选项
时请求体不出现该键。

检索分数按度量换算（越大越相似）：`cosine` → `1 - distance` ∈ [-1, 1]；
`l2` / `l1` → `1 - distance`（不截断）；`ip` → 点积本身。

## 稀疏向量（BM25）与融合检索（P4）

稀疏索引是**显式创建开关**：`sparse=bm25` 仅 `hnsw` / `flat` 后端可用，
创建后不可变；隐式创建的集合恒为 `none`，在 `sparse=none` 集合上写入
`sparse`（或创建时组合其他后端）一律 400。旧服务端创建响应不回显该字段时
`created.sparse` 为 `None`，按可选字段容忍。

词项由客户端给定（无分词器）：`(term_id: u32, weight: f32)`，
term ∈ [1, 2^24−1]，weight 有限且 > 0；每文档 ≤ 1024 个词项（重复 term
服务端求和），融合查询 ≤ 64 个词项。

```rust
use houyidb_client::{FusionOptions, SparseTerms, Vector};

// 稀疏写入与 dense 同批（整文档替换语义）：None = 不改，Some([]) = 清空
let terms: SparseTerms = vec![(7, 1.0), (8, 0.8)];
col.upsert(&[Vector {
    id: stable_id("doc-2"),
    values: embedding,
    label: None,
    payload: None,
    sparse: Some(terms),
}])?;

// 融合检索：dense 与 sparse 必须同时非空；k 上限与 search 一致
let fused = col.fusion_search(&query, &[(7, 1.0), (8, 0.8)], 10, &FusionOptions {
    fusion: Some("rrf".into()),      // rrf（缺省）/ weighted
    rrf_k: Some(60),                 // 1..=1000，仅 rrf 生效
    // alpha: Some(0.5),             // 0..=1，仅 weighted 生效（缺省 0.5）
    filters: vec![Filter::new("category", FilterOp::Eq, "fruit")],
    include_payload: true,
    ..Default::default()
})?;
for hit in &fused.hits {
    // score 仅表达融合排序；dense_score 才按集合度量换算，sparse_score 是 BM25 原始分
    println!("{} fused={} dense={:?} sparse={:?}",
        hit.id, hit.score, hit.dense_score, hit.sparse_score);
}
```

融合两种模式：`rrf`（默认，`score = Σ 1/(rrf_k + rank)`）、`weighted`
（两路各 min-max 归一化后 `alpha*dense + (1-alpha)*sparse`）。两路候选均
按 `4*k` 过取并按 filters 过滤；未命中任一路的文档不入选。`FusionOptions`
中显式给出的 `rrf_k`/`alpha` 服务端一律严格校验（越界 400，即使当前模式不
参与计算）。`fused.metric` 仅解释 `dense_score` 口径；`fused.fusion_mode` /
`rrf_k` / `alpha` 回显生效参数。

融合检索是纯读，不隐式建集合；`sparse=none` 集合返回 400；旧服务端没有该
端点（404 原样透出）。稀疏**写入**仅 HTTP/SDK 提供（pgwire 仅支持融合查询）。

## 管理操作：删除整个 collection（不可逆）

需服务端管理员权限（RBAC `Action::Admin`）。删除走 WAL-first：记录先落盘，
崩溃重放与 follower 复制都收敛到「已删除」。**不可逆**，删除前请先用导出
端点（`GET /v1/collections/{name}/export`）备份。同名重建会得到全新的空
collection，旧向量不会复活。

```rust
// admin token 只附加在 /v1/admin/* 请求上，普通读写不携带
let admin = Client::with_timeout(url, timeout).with_admin_token(admin_token);
admin.delete_collection("docs")?; // 不存在 → HouyiError::Http { status: 404, .. }
```

EN: Admin-only on the server (`Action::Admin`). Deletion is WAL-first, so
crash replay and follower replication converge to "gone". **Irreversible**;
export via `GET /v1/collections/{name}/export` first. A same-name recreate
starts empty and never resurrects the old vectors.

## 与其他语言 SDK 的对应

- 向量 id 是 `u64`——原生覆盖键空间，无 JS 的 2^53 精度问题；
- `stable_id` 与 Python/TS/Go SDK、服务端 OpenAI 网关五方同算法
  （BLAKE2b-8 大端 & 2^63-1，digest 长度是算法参数而非截断）：同一字符串 id
  在任何语言得到同一 u64，删除可跨 SDK 重放；
- 错误统一为 `HouyiError`：`Http { status, message }`（服务端纯文本诊断原样透出）
  或 `Transport`（连接/读取失败）；
- collection 名经 RFC 3986 路径转义（与 Python `quote(safe="")` 同口径），
  含 `/ ? #` 或中文的名字不会改写路由。

## 测试

```bash
cargo test -p houyidb-client            # 单元（stable_id 五方对拍）+ 集成
```

集成测试打 127.0.0.1:17000 真实服务（`HOUYIDB_URL` 可覆盖；不可达自动跳过）。
前置：`./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
