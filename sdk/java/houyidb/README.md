# houyidb-java

> [English](README.en.md)

HouyiDB 向量数据库的官方 Java 客户端（collections HTTP API）。同步阻塞式
（`java.net.http.HttpClient`），Java 17+，仅两个外部依赖：BouncyCastle
（跨语言一致的 `StableId`）与 Jackson（JSON）。

```groovy
dependencies {
    implementation 'top.houyidb:houyidb-client:0.2.0-beta.8'
}
```

## 快速开始

```java
import com.houyidb.client.*;
import com.houyidb.client.Types.*;

var client = new HouyiClient("http://127.0.0.1:7000");
var col = client.collection("docs");

// 写入（collection 首次写入隐式创建，维度由首个向量定型）
col.upsert(List.of(new Vector(
        StableId.of("doc-1"),
        embedding, // double[]
        Map.of("text", "...", "category", "fruit"))));

// 检索（score 由度量换算、越大越相似，详见下文「显式建集合」）
SearchResult result = col.search(queryEmbedding, 5,
        new SearchOptions().includePayload(true));

// 混合检索（filters 取交集；FilterOp: EQ/GT/GTE/LT/LTE/IN）
SearchResult filtered = col.hybridSearch(queryEmbedding, 5,
        new SearchOptions().includePayload(true)
                .filter("category", FilterOp.EQ, "fruit"));

// 删除（按 id 和/或过滤条件，幂等；两者取并集）
col.delete(List.of(StableId.of("doc-1")), null);
```

## 显式建集合（维度/度量/后端/HNSW 一次定型）

`PUT /v1/collections/{name}` 是普通写路径（写权限，不需要 admin token）。
显式创建会锁定维度与度量：同名已存在 → `HouyiException(409)`；度量与后端
创建后不可变，省略字段走服务端默认（`cosine` + `hnsw`）。

```java
var created = client.createCollection("docs", 768,
        new CreateCollectionOptions()
                .metric("ip")        // l2 / cosine / ip / l1
                .indexType("hnsw")   // hnsw / flat / clustered / flat-clustered / pq-flat / hii-l0
                .hnsw(new HnswParams(16, 100, 64)));
// created.metric == "ip"；服务端回显实际生效参数
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

```java
HouyiClient admin = HouyiClient.withAdminToken(baseUrl, adminToken);
admin.deleteCollection("docs"); // 不存在 → HouyiException(404)
```

EN: Admin-only on the server. WAL-first, so crash replay and follower
replication converge to "gone". **Irreversible**; export first. A same-name
recreate starts empty and never resurrects old vectors.

## 与其他语言 SDK 的对应

- 向量 id 是 `long`——stable_id 落在 [0, 2^63-1]，有符号 long 无损承载，
  无 JS 的 2^53 精度问题；
- `StableId.of` 与 Python/TS/Go/Rust/.NET SDK、服务端 OpenAI 网关六方同算法
  （BLAKE2b-8 大端 & 2^63-1；`Blake2bDigest(64)` 的 digest 长度是算法参数
  而非截断）：同一字符串 id 在任何语言得到同一 u64，删除可跨 SDK 重放；
- 服务端非 2xx 统一抛 `HouyiException`（`statusCode()` + 服务端诊断文本），
  传输层失败仍为 `IOException`；
- collection 名经 RFC 3986 路径转义（与 Python `quote(safe="")` 同口径），
  含 `/ ? #` 或中文的名字不会改写路由。

## 测试

```bash
gradle test   # StableId 六方对拍 + 路径转义 + 集成（打 127.0.0.1:17000 真实服务，
              # HOUYIDB_URL 可覆盖；不可达经 JUnit Assumptions 自动跳过）
```

集成前置：`./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000 --wal-path /tmp/houyidb-eco.wal`
