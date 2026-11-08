# HouyiDB Server 使用手册

> [English](HOUYIDB_SERVER_MANUAL.en.md)

> 单机、内存优先的向量数据库 — 服务端手册。  
> 支持 pgwire / HTTP 双协议与多种可切换索引后端；生产部署、认证、TLS、
> 静态加密与主备配置见 `docs/DEPLOYMENT_GUIDE.md`。

---

## 1. 快速启动

### 1.1 编译

```bash
cargo build --release -p houyi-server
```

### 1.2 启动服务

```bash
./target/release/houyi-server \
  --benchmark-http-addr 127.0.0.1:7000 \
  --pgwire-addr 127.0.0.1:5432 \
  --vector-dim 1536 \
  --worker-threshold 256 \
  --flush-policy hybrid \
  --index-type hnsw
```

### 1.3 健康检查

```bash
curl http://127.0.0.1:7000/v1/health
# → {"status":"ok"}
```

---

## 2. CLI 参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| `--benchmark-http-addr` | `HOUYI_SERVER_BENCHMARK_HTTP_ADDR` | `127.0.0.1:7000` | HTTP API 监听地址 |
| `--pgwire-addr` | `HOUYI_SERVER_PGWIRE_ADDR` | `127.0.0.1:5432` | pgwire 监听地址 |
| `--vector-dim` | `HOUYI_SERVER_VECTOR_DIM` | `1536` | 向量维度 |
| `--worker-threshold` | `HOUYI_SERVER_WORKER_THRESHOLD` | `256` | Builder 触发阈值 (向量数) |
| `--worker-tick-ms` | `HOUYI_SERVER_WORKER_TICK_MS` | `20` | Builder 轮询间隔 (ms) |
| `--flush-policy` | `HOUYI_SERVER_FLUSH_POLICY` | `hybrid` | `threshold` / `interval` / `hybrid` |
| `--index-type` | `HOUYI_SERVER_INDEX_TYPE` | `hnsw` | `hnsw` / `clustered` / `flat-clustered` / `flat` / `pq-flat` / `hii-l0` |
| `--hnsw-ef-search` | `HOUYI_SERVER_HNSW_EF_SEARCH` | 自动推断 | 覆盖 HNSW ef_search |
| `--metrics-interval-secs` | `HOUYI_SERVER_METRICS_INTERVAL_SECS` | `1` | 指标输出间隔 |
| `--wal-path` | `HOUYI_SERVER_WAL_PATH` | 临时目录 | WAL 文件路径 |
| `--wal-write-mode` | `HOUYI_SERVER_WAL_WRITE_MODE` | `async` | `async` / `sync` |
| `--openai-upstream` | `HOUYI_OPENAI_UPSTREAM` | 未配置 | OpenAI 兼容上游 base_url（启用 `/v1/embeddings` 网关） |
| `--openai-timeout-secs` | `HOUYI_OPENAI_TIMEOUT_SECS` | `30` | 网关调用上游的超时（秒） |
| — | `HOUYI_OPENAI_API_KEY` | 空 | 上游 API key（**仅 ENV 注入**，不接 CLI——命令行对 `ps` 可见即泄漏） |

### 2.1 HNSW 专用环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `HOUYI_HNSW_PARALLEL_BATCH_BUILD` | `1` (服务端) | 启用 Rayon 并行构图 |
| `HOUYI_HNSW_MAX_BUILD_BATCH` | `8192` (服务端) | 单批最大构图向量数 |
| `HOUYI_HNSW_SEED_SERIAL_POINTS` | `256` | 小图阶段串行插入阈值 |
| `HOUYI_VECTOR_AUTO_HNSW` | `1` | 自动推断 M / ef 参数 |
| `HOUYI_VECTOR_RECALL_OPTIMIZED` | `0` | 高 recall 模式 (提升 ef_c) |
| `HOUYI_VECTOR_ESTIMATED_VECTORS` | 自动 | 预估向量数 (影响参数推断) |

---

## 3. 索引后端

| 索引类型 | `--index-type` | 构建时间 | QPS | Recall | 适用场景 |
|------|:---:|:---:|:---:|:---:|------|
| **HNSW** | `hnsw` | 慢 | 中 | **最高 (0.9995)** | 高 recall、离线索引 |
| **ClusteredHnsw** | `clustered` | **快 (8x)** | 持平 HNSW | 0.995 | 写多读少、快速重建 |
| **FlatClustered** | `flat-clustered` | **极快** | **最高 (231)** | 0.998 | 小集群暴力检索 |
| **Flat** | `flat` | **0** | 低 (14) | **1.0** | N < 5K，精确检索 |
| **PQ-Flat** | `pq-flat` | 快 | 高 | 有损 | 内存受限；距离为 L2 |
| **HII L0** | `hii-l0` | 快 | 中 | 小图 ANN | 实时写入 + 小图检索（L1/L2 未实现） |

> 基准数据: OpenAI 50K × 1536D, COSINE, Apple Silicon 8 核（HNSW/Clustered/
> FlatClustered/Flat）；PQ-Flat 与 HII L0 未列入该基准表。

---

## 4. pgwire 协议 (端口 5432)

### 4.1 连接

```bash
# 使用标准 PostgreSQL 客户端
psql -h 127.0.0.1 -p 5432 -U tester -d houyi
```

### 4.2 向量检索

```sql
-- k=10, 无过滤
/* houyi:ef_search=256 */ /* houyi:vector=0.1,0.2,... */ SELECT id FROM docs LIMIT 10;

-- k=10, 带过滤
/* houyi:vector=0.1,0.2,... */ SELECT id FROM docs WHERE id IN (1,2,3) LIMIT 10;
```

### 4.3 批量写入

```sql
-- 通过 Extended Query + BYTEA binary 参数
/* houyi:upsert_batch=$1 */ SELECT 1;
-- Bind: $1 = AUB1 binary payload (dim|count|id+vec[])
```

### 4.4 客户端驱动

```python
# Python (psycopg)
import psycopg
conn = psycopg.connect("host=127.0.0.1 port=5432 user=tester dbname=houyi")
conn.execute("/* houyi:vector=%s */ SELECT id FROM docs LIMIT 10", ("0.1,0.2,...",))
```

```rust
// Rust (tokio-postgres)
use tokio_postgres::connect;
let (client, conn) = connect("host=127.0.0.1 port=5432 user=tester dbname=houyi", NoTls).await?;
let rows = client.query("/* houyi:vector=$1 */ SELECT id FROM docs LIMIT 10", &[&"0.1,0.2,..."]).await?;
```

---

## 5. HTTP API (端口 7000)

### 5.1 健康检查

```bash
GET /v1/health
# → {"status":"ok"}
```

### 5.2 批量写入

```bash
POST /v1/vectors/upsert
Content-Type: application/json

{
  "vectors": [
    {"id": 1, "values": [0.1, 0.2, ...]},
    {"id": 2, "values": [0.3, 0.4, ...]}
  ],
  "index": {
    "m": 24,
    "ef_construction": 400
  }
}
# → {"inserted": 2}
```

### 5.3 向量检索

```bash
POST /v1/vectors/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, ...],
  "k": 10,
  "ef_search": 256
}
# → {"ids": [1, 3, 7, ...]}
```

### 5.4 索引构建/优化

```bash
POST /v1/optimize
# → {"flushed_items": 50000, "pending_items": 0, ...}
```

### 5.5 索引状态

```bash
GET /v1/index/status
# → {"indexed_count": 50000, "pending_count": 0, "hot_count": 0,
#     "distance_metric": "cosine", "hnsw_m": 24,
#     "hnsw_ef_construction": 400, "hnsw_ef_search": 256, ...}
```

### 5.5.1 显式建集合与距离度量

```bash
PUT /v1/collections/{name}
# {"dims":8,"metric":"ip","index_type":"hnsw",
#  "hnsw_m":16,"hnsw_ef_construction":100,"hnsw_ef_search":64}
# → 201 {"created":true,"collection":"goods","dims":8,"metric":"ip",
#        "index_type":"hnsw","hnsw":{...}}
```

- RBAC 归属 `Action::Write`（写权限，不是 admin）；租户作用域外的名字仍被拒绝。
- 参数在落 WAL 前统一校验（`validate_create`）：非法度量/后端/维度/边界返回
  400，不会留下无法重放的创建记录；同名已存在（即使参数一致）返回 409；
  follower/只读 503。成功时追加 `collection_create` 记录并复制到 follower。
- `metric` 与 `index_type` 创建后不可变；HNSW 三参数必须同时给出
  （`m` 1..=1024，`ef_construction`/`ef_search` 1..=65536）。
- 支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持
  `l2`/`cosine`/`ip`/`l1`；`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。
- 内核距离一律「越小越近」（L2 为平方距离、IP 为 `-dot`、L1 为 Σ|a-b|）；
  HTTP 分数在唯一出口按度量换算：cosine `1-d`（钳制 [-1,1]）、l2/l1 `1-d`
  （不截断）、ip `-d`（即原始点积）。集合/混合检索响应回传 `metric` 字段；
  benchmark 兼容路径不带该字段以保持字节不变。
- 快照与恢复：collection 快照 v2 携带 `metric` 与可选 HNSW 参数（v1 解码时
  按后端默认度量）；图快照带度量，度量不匹配拒绝加载。WAL 的
  `collection_create` 记录同样携带度量/HNSW，冷重放不会把历史集合重置为
  cosine。

### 5.6 WAL 管理

```bash
GET  /v1/wal/status      # 主 WAL 状态 + collections 段/保留/manifest 状态
POST /v1/wal/checkpoint  # {"through_lsn": 42} → 截断 WAL
POST /v1/wal/rotate      # 归档当前 WAL
POST /v1/wal/snapshot    # 创建向量快照

# 存储生命周期（P5.0）
GET  /v1/collections/{name}/status   # hot/indexed/pending + applied_lsn + 标量统计
POST /v1/collections/{name}/compact  # {"force": true} 标量页内存压实
POST /v1/admin/collections/compact   # {"rotate":true,"hold_retention_secs":3600}
                                     # 快照周期 → manifest → 可选轮转 → 保留

# 管理员：整 collection 删除（不可逆，需 admin 凭据）
DELETE /v1/admin/collections/{name}
```

`DELETE /v1/admin/collections/{name}` 成功返回
`{"deleted":true,"collection":"<name>"}`；未知/重复删除 404，与并发删除竞争 409，
follower/只读或 lease 过期 503。删除 WAL-first：先追加
`{"v":1,"op":"collection_drop","name":"<name>"}` 再移除内存对象，崩溃重放与
follower 实时 apply 都会收敛到「已删除」；follower 实时 apply 不追加本地 WAL。
同名重建得到全新空 collection，旧向量不会复活。重放时 drop 与其它记录一样
受 per-collection 快照 checkpoint 约束：`创建→删除→重建→快照` 后，旧 drop
的 LSN 低于新 collection 的 checkpoint，会被跳过，不会删掉快照中存活的数据。
删除期间周期快照跳过该 collection（不中止本轮）；全部删除后写入空 manifest，其
`safe_checkpoint_lsn = global_barrier_lsn`，允许 WAL 截断到该屏障。follower
**不**提交空名单 manifest：其本地 WAL 先 append 后 apply，`next_lsn - 1` 可能
越过尚未应用的 create/upsert，写成 safe 会让保留策略截断未应用记录；因此
follower 保留旧 manifest（仅暂时多占本地 WAL），直到有新 collection 应用落地
或被 promote 为 leader 后再推进。
复制协议 **v8** 在 v7（`collection_drop`）之上补齐 `collection_create` 的
度量/HNSW 语义（线格式不变），版本门 fail-fast：v7/v8 不能混跑，升级须全集群
协调（先停复制、全量升级、再恢复）。**禁止**把 v8 WAL 交给 v7 二进制冷重放：
旧二进制会跳过未知 drop 记录、忽略创建记录里的度量字段而按 cosine 重建，
造成「已删集合复活 + 度量静默漂移」的分叉。

`collections_wal_segment_bytes`（默认 64 MiB，0=仅手动轮转）、
`collections_wal_max_segments`（默认 16，0=关闭删除）、
`collections_wal_follower_grace_secs`（默认 60，leader 断连进度保留）、
`collections_scalar_compact_interval_secs`（默认 600，0=关闭）、
`collections_scalar_compact_stale_ratio`（默认 0.5）。标量页为 append-only：重复
upsert 同一 id 追加新行，过滤与 payload 回读只看每个 id 的最后一行；压实只回收
物理行，不改变查询结果（公共 issue #6 修复后的语义）。

### 5.7 pgwire 指标

```bash
GET /v1/pgwire/status
# → {"handshake_count": 5, "simple_query_count": 100, ...}
```

### 5.8 OpenAI 兼容网关 (/v1/embeddings)

以 `--openai-upstream` 启动后，HouyiDB 暴露 OpenAI 形状的嵌入端点。内核**没有** embedding
推理能力——网关把请求转发到配置的上游（OpenAI / Ollama / vLLM 等任意兼容端点），
自己不算向量。

**纯代理模式**（不带 `collection` 字段）：请求体原样转发（未知字段不丢），上游响应
（含错误状态码与正文）原样回传——"把 base_url 从 OpenAI 换成 HouyiDB"即可工作：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7000/v1", api_key="not-needed")
resp = client.embeddings.create(model="text-embedding-3-small", input="hello")
```

**自动入库模式**（带 `collection` 扩展字段，OpenAI SDK 经 `extra_body` 传入）：
网关把上游返回的 embedding 顺手 upsert 进该 collection——"换 URL 即入库"：

```python
resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["apple is a fruit", "banana is yellow"],
    extra_body={"collection": "docs"},
)
print(resp.houyi)  # {"collection": "docs", "upserted": 2}
```

- 入库 id = `stable_id(text)`（blake2b-8，与 Python SDK 同方案，重跑幂等、跨 SDK 可重放删除）。
- 原文进 payload `{"text": ...}`（与 langchain / llama-index adapter 的 text_key 约定一致）。
- `houyi` 扩展字段注入响应顶层，OpenAI 客户端忽略未知字段，安全。
- 自动入库要求 input 是字符串（数组）；token 数组无法还原原文，显式 400。
- 未配置 `--openai-upstream` 时该端点回 501。

---

## 6. VectorDBBench 标准测试

### 6.1 环境准备

```bash
cd "$VDB_ROOT"        # 含 houyidb client 适配器的 VectorDBBench 工作副本
source .venv/bin/activate
```

### 6.2 运行 benchmark

```bash
# 服务端启动
./target/release/houyi-server \
  --benchmark-http-addr 127.0.0.1:17000 \
  --pgwire-addr 127.0.0.1:15432 \
  --vector-dim 1536 \
  --index-type hnsw \
  --worker-threshold 256 \
  --flush-policy hybrid

# VDB 测试
python3 -m vectordb_bench.cli.vectordbbench houyidb \
  --host 127.0.0.1 --port 15432 --http-port 17000 \
  --transport pgwire \
  --case-type Performance1536D50K \
  --k 10 --m 24 --ef-construction 400 --ef-search 256 \
  --num-concurrency 1 \
  --task-label "my-benchmark"
```

### 6.3 结果解读

| 指标 | 含义 |
|------|------|
| `load_duration` | 总加载时间 (插入 + 优化) |
| `insert_duration` | 纯插入耗时 |
| `optimize_duration` | 索引构建耗时 |
| `qps` | 每秒查询数 |
| `serial_latency_p99` | P99 延迟 (秒) |
| `recall` | Recall@k |
| `ndcg` | NDCG@k |

---

## 7. 完整测试命令

```bash
# 全量单元测试
cargo test --all

# 单 crate 测试
cargo test -p houyi-core
cargo test -p houyi-vector
cargo test -p houyi-query
cargo test -p houyi-persistence
cargo test -p houyi-server

# 性能冒烟 (跳过 recall)
./scripts/perf_smoke_fast.sh

# 完整冒烟 (含 recall)
./scripts/perf_smoke.sh

# VectorDBBench 对照
python3 benchmarks/vectordb_bench_houyidb/install.py --vdb-root "$VDB_ROOT"
VDB_ROOT="$VDB_ROOT" ./benchmarks/vectordb_bench_houyidb/run_bench.sh
```

---

## 8. 基准数据

### 8.1 50K 1536D (OpenAI, COSINE, k=10)

| 索引 | 加载 | QPS | P99 | Recall@10 |
|------|:---:|:---:|:---:|:---:|
| HNSW | 64.6s | 182 | 6.6ms | 0.9995 |
| ClusteredHnsw | 17.6s | 179 | 6.8ms | 0.9948 |
| FlatClustered | 17.0s | **231** | **5.1ms** | 0.9978 |
| Flat | **9.5s** | 14 | 71.9ms | **1.0000** |

### 8.2 1M 768D (Cohere, COSINE, k=100)

| 索引 | 加载 | QPS | P99 | Recall@100 |
|------|:---:|:---:|:---:|:---:|
| HNSW | 1050s | 176 | 6.8ms | 0.969 |
| ClusteredHnsw | 480s | 70 | 16.4ms | 0.961 |

---

## 9. 部署建议

| 场景 | 推荐索引 |
|------|------|
| 高 recall (≥ 0.999) | `hnsw` |
| 快速加载 + 稳定 QPS | `clustered` |
| 极致 QPS + 延迟 | `flat-clustered` |
| N < 5K, 精确检索 | `flat` |
| N > 1M | `hnsw` 或 `clustered` |

---

## 10. 混合检索 (向量 + 结构化过滤)

### 10.1 HTTP API

```bash
POST /v1/collections/{name}/hybrid/search
Content-Type: application/json

{
  "vector": [0.1, 0.2, ...],
  "k": 10,
  "filters": [
    {"column": "tag", "op": "eq", "value": "beach"},
    {"column": "date", "op": "gt", "value": "1717200000"},
    {"column": "date", "op": "lt", "value": "1728000000"},
    {"column": "people", "op": "in", "value": ["Alice", "Bob"]}
  ]
}
```

### 10.2 支持的过滤操作

| 操作 | 说明 | 示例 |
|:-----|:-----|:-----|
| `eq` | 等于 | `{"column":"tag","op":"eq","value":"beach"}` |
| `gt` | 大于 | `{"column":"date","op":"gt","value":"1717200000"}` |
| `lt` | 小于 | `{"column":"date","op":"lt","value":"1728000000"}` |
| `gte` | 大于等于 | `{"column":"score","op":"gte","value":"0.8"}` |
| `lte` | 小于等于 | `{"column":"score","op":"lte","value":"0.5"}` |
| `in` | 集合匹配 | `{"column":"people","op":"in","value":["Alice","Bob"]}` |

### 10.3 谓词树 (PAX 层)

```rust
ScalarPredicate::And(
    Box::new(ScalarPredicate::Gt(ScalarValue::Timestamp(1717200000))),
    Box::new(ScalarPredicate::Lt(ScalarValue::Timestamp(1728000000))),
)
```

- `ScalarValue`: Int64 / Float64 / Bool / Text / Timestamp / JsonText
- `ScalarPredicate`: Eq / Gt / Lt / GtEq / LtEq / In / And / Or
- `filter_multi_column()`: 多列 AND 组合
