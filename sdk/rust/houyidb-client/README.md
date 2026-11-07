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
col.upsert(&[Vector {
    id: stable_id("doc-1"),
    values: embedding, // Vec<f32>
    label: None,
    payload: Some(HashMap::from([
        ("text".into(), "...".into()),
        ("category".into(), "fruit".into()),
    ])),
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
    hnsw: Some(HnswConfig { m: 16, ef_construction: 100, ef_search: 64 }),
})?;
assert_eq!(created.metric, "ip");
```

度量支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持全部四种；
`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。

检索分数按度量换算（越大越相似）：`cosine` → `1 - distance` ∈ [-1, 1]；
`l2` / `l1` → `1 - distance`（不截断）；`ip` → 点积本身。

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
