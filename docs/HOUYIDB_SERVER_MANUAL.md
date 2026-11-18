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

### 4.2.1 融合检索（P4，仅查询）

```sql
-- sparse 与 vector 必须同时出现；collection 由 houyi:collection 指令指定
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ SELECT id FROM docs LIMIT 2;

-- 加权融合；fusion 缺省 rrf、alpha 缺省 0.5
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ /* houyi:fusion=weighted */ \
  /* houyi:alpha=0.4 */ SELECT id FROM docs LIMIT 2;
```

- 指令语法：`/* houyi:sparse=term:weight[,term:weight…] */`（重复 term
  求和；≤64 个原始对）、`/* houyi:fusion=rrf|weighted */`、
  `/* houyi:alpha=0..1 */`。sparse 与 vector 同时存在 → 融合检索；只有
  sparse 或只有 fusion/alpha → 语法错误（SQLSTATE 42601）。
- 执行语义：必须显式 `/* houyi:collection=name */`（缺指令 `0A000`）；
  集合不存在 `42P01`（不会隐式建集合）；集合 `sparse=none` → `0A000`；
  词项 / alpha / fusion / k 语义越界 → `22023`；内部错误 `XX000`。
- 结果仍是单列 `id`（oid 20）。pgwire **不提供稀疏写入**（AUB1 行只有
  `id + dense 向量`），稀疏词项走 HTTP `upsert` / SDK。

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
- `vector_dtype`：`f32`（缺省）/ `f16`（别名 float32/fp32/float16/fp16/half，
  大小写不敏感）。`f16` 将常驻向量内存减半，仅 hnsw 后端允许，创建后不可变；
  内核以 f32 查询、f16 存储计算距离（非对称量化，f16 值先舍入到 f16 网格）。
  非 f32 与非 hnsw 组合 400；同名 f32→f16 或反向 409。创建响应、列表摘要与
  status 均回显 `vector_dtype`。
- `quantization`：`none`（缺省）/ `int8` / `rabitq`。与 `vector_dtype`
  **相互独立**：`vector_dtype` 决定主表编码，`quantization` 只压缩 HNSW
  **图侧副本**。主表 / hot buffer / 构建队列始终保留原始（f32/f16）向量；
  检索时用量化图取候选（int8 宽度 ×2、rabitq ×16：
  `INT8_ANN_K_MULTIPLIER = 2`、`RABITQ_ANN_K_MULTIPLIER = 16`，后者码字
  带宽与前者等价），再在原始
  向量上精确重排，最终 top-k 分数不经过量化。`int8` / `rabitq` 仅 hnsw 后端
  允许（其他后端 400），创建后不可变；同名集合显式携带不一致的
  quantization → 409。余弦度量先归一化、后量化；插入时新向量充当查询且
  **始终为 f32**（非对称量化）。非有限分量在服务端写入总闸门已被拒绝；
  库层对非有限值做防御性替换。pgwire 无 `CREATE COLLECTION` 语法，量化集合
  只能经 HTTP / SDK 创建。创建响应、列表摘要与 status 均回显 `quantization`。
- `int8` 内核：逐维 int8 码 + 标量因子，图侧码流 1 B/dim。内存账目（如实，
  不得按「减半」规划）：非余弦 f32 4→5 B/dim（+1）、非余弦 f16 2→3 B/dim
  （+1）、余弦 f32 8→5 B/dim（−3）、余弦 f16 4→3 B/dim（−1）；收益在图内
  距离工作集（4→1 B/dim，缓存局部性）与内存带宽。
- `rabitq` 内核：D = dims 向上取 2 的幂、尾部补 0；确定性随机旋转
  P = H_D ∘ diag(signs)（种子只依赖 D，跨进程/节点/重启逐位可复现）；编码
  = 符号位（`ceil(D/64)` 个 u64）+ `alpha = Σ|ō_i|` + `norm = ||x||`。查询
  侧仅在遍历入口做一次补 0 + 旋转（O(D log D)），非对称估计
  `est_cos = clamp(符号内积/alpha, -1, 1)`：余弦距离 `1−est`，L2 平方距离
  `max(0, q²+x²−2qx·est)`；存储×存储（构图启发式/剪枝）用汉明距离 +
  arcsine 律估计。零范数向量用哨兵 `{codes=0, alpha=1.0, norm=0}`，
  `q_norm` 或 `x_norm` 为 0 时交叉项为 0（余弦距离自然为 1）。
  **仅支持 `metric = l2 | cosine`**：`ip` / `l1` 在创建、WAL 重放、快照期望
  三层 fail-fast（400 / 脏记录拒绝）。内存账目（如实，不得按「减半」规划）：
  主表内存不变（图侧 `source` 与主表共享 `Arc`）；图侧码流
  `ceil(D/64)·8` B + 8 B 因子，dims=768（D=1024）时每向量 128 B + 8 B
  （同口径 f32 3072 B、int8 768 B + 8 B）。相对 f32 图侧仅多 ~136 B/向量
  （≈1/24），相对 int8 降至约 1/6；另需每集合共享旋转符号 4·D 字节与每查询
  一个 D×f32 临时缓冲。非余弦总内存略增（≈+0.18 B/dim），余弦总内存较未
  量化下降（≈−3.8 B/dim）。
- `sparse`：`none`（缺省）/ `bm25`，仅 `hnsw` / `flat` 后端，创建后不可变，
  隐式创建恒为 `none`；`bm25` 在集合上建立旁路倒排索引（不改变 dense 检索
  结果），写入与融合查询见 5.5.2。同名集合改变 sparse → 409，非法组合 → 400。
- 后续扩展点（仅文档，不落代码）：PQ 等更强量化器的码本由 leader 训练一次、
  以 WAL quantizer 记录随复制分发（副本按 LSN 装载），并内嵌进图快照；本轮
  只有 `int8` / `rabitq`，不提供半成品 API。
- 支持矩阵：`hnsw` / `flat` / `clustered` / `flat-clustered` 支持
  `l2`/`cosine`/`ip`/`l1`；`pq-flat` 仅 `l2`；`hii-l0` 仅 `cosine`。
- 内核距离一律「越小越近」（L2 为平方距离、IP 为 `-dot`、L1 为 Σ|a-b|）；
  HTTP 分数在唯一出口按度量换算：cosine `1-d`（钳制 [-1,1]）、l2/l1 `1-d`
  （不截断）、ip `-d`（即原始点积）。集合/混合检索响应回传 `metric` 字段；
  benchmark 兼容路径不带该字段以保持字节不变。
- 快照与恢复：collection 快照 v2 携带 `metric`、可选 HNSW 参数、
  `vector_dtype`（flags bit0）与 `quantization`（flags bit1 = int8、
  bit2 = RaBitQ（`0x04`）；无稀疏索引时 payload 版本保持 2、不新增 body
  字节（含 `bm25` 时见 5.5.2：payload v3 + flags bit3）；两位互斥；
  旧二进制只接受 bit0–bit1，遇 RaBitQ flag fail-fast），manifest 的
  `vector_dtype`/`quantization` 为可选字段（quantization：1=int8、
  2=rabitq）；v1 解码按后端默认度量/f32/未量化。图快照带度量、dtype 与
  quantization，不匹配拒绝加载（量化集合必须按快照头建对应后端，恢复后用
  原始 source 向量精确 rescore，不会退化为反量化近似）。独立 HNSW 图快照
  采用不对称版本门：f32 写 version=1（旧二进制可读），f16 写 version=2，
  int8 写 version=3，rabitq 写 version=4（flags bit3 = `0x08` 必置、禁止
  bit2；旧二进制以 `UnsupportedVersion` 拒绝并回退全量 WAL 重放，不会静默
  降为 f32/未量化后再被下次快照固化）。manifest `format_version`：仅 f32
  → 1，含 int8 → 2，含 rabitq → 3（精确匹配，不松散）；`quantization`
  字段值 1 要求 ≥2 且必须 hnsw + `HnswGraph`，值 2 要求 ≥3、hnsw +
  `HnswGraph` 且度量为 l2/cosine。WAL 的 `collection_create` 记录同样携带
  度量/HNSW/`vector_dtype`/`quantization`（hnsw 恒写，含显式 `"none"`；
  非 hnsw 不写），冷重放不会把历史集合重置为 cosine / f32 / 未量化。
- 全后端快照压缩（P5）：payload/manifest 的 `vector_kind` 新增 3 =
  `pq-flat 已建紧凑态`、4 = `懒后端缓冲原始行`（加法扩展；旧二进制在
  payload 解码时对未知 kind 干净拒绝，`format_version` 不变）。懒后端快照
  表示：clustered / flat-clustered = 已建原始向量（边界复制按 id 去重）+
  待建行；pq-flat 已建 = 紧凑构建态（centroids/codebook/聚类 id+codes）+
  待建行；pq-flat 未建 = 待建行。恢复时的 `pq-flat` 紧凑态逐位保真、不重新
  训练；懒后端一律只装回缓冲、**不**自动构图（与无快照全量重放冷启动一致，
  构图前检索为空）。`kind=4` 仅允许 clustered/flat-clustered/pq-flat，
  `kind=3` 仅允许 pq-flat，manifest 与 payload 双侧一致校验。正常运行时
  保留不再出现 `unsupported_backend` 阻塞。

### 5.5.2 稀疏索引（BM25）与融合检索（P4）

- 数据模型：稀疏词项由客户端给定 `(term_id, weight)`；`term_id ∈
  [1, 2^24-1]`，weight 有限且 >0；每文档 ≤1024 个原始对、查询 ≤64 个；
  重复 term 求和、顺序无关；**无分词器**。`sparse=bm25` 集合维护旁路
  `Bm25Index`（term → postings + 正向索引），主表 / hot buffer / 构建队列
  仍只保存稠密向量，dense 检索结果不受影响。
- 写入：`POST /v1/collections/{name}/upsert` 的每条向量可选
  `"sparse": [[term, weight], …]`；键缺省或 `null` = 不改，`[]` = 清空；
  同一 id 重写是**整文档替换**（不是增量合并），因此 WAL covered 重放幂等。
  稀疏更新在写锁内与标量页同步完成，刚写入、尚在异步构图队列的向量也能
  立即被稀疏检索/删除命中。`sparse=none` 集合带 `sparse` 键 → 400
  （不静默丢弃）。
- 融合检索：`POST /v1/collections/{name}/fusion/search`，请求
  `{vector, sparse, k, filters?, ef_search?, include_payload?, fusion?,
  rrf_k?, alpha?}`；dense 与 sparse 必须同时非空。两路各取 4×k 候选
  （先应用过滤），融合后截前 k；排序为融合分降序、id 升序破平。
  - RRF（缺省，`rrf_k=60`，范围 1..=1000）：`Σ 1/(rrf_k + rank)`，
    rank 从 1 起，仅命中该路列表各计一项。
  - 加权（`alpha` 缺省 0.5，范围 0..=1）：两路在过滤后候选上各自
    min-max 归一（单元素归一为 1.0，缺失模态记 0），
    `score = alpha*dense + (1-alpha)*sparse`。
  - 响应字段：`ids`、`scores`（**仅融合排序语义，不属于任何 metric**）、
    `dense_scores`（按集合 metric 换算，缺席模态为 null）、
    `sparse_scores`（BM25 原始分，缺席模态为 null）、`metric`（仅用于解释
    `dense_scores`）、`fusion:{mode, rrf_k, alpha}`（只回显实际参与计算的
    参数）、可选 `payloads`。
  - 显式提供但当前模式不用的参数也会校验（非法一律 400）；`k` /
    `ef_search` 上限与普通检索同口径；filters 仅 hnsw / flat。
- BM25：`k1=1.2`、`b=0.75`，`idf = ln(1 + (N-df+0.5)/(df+0.5))`，
  `tf` = 客户端权重，`dl = Σ权重`，`avgdl = Σdl/N`；N=0 返回空。死 posting
  在 `dead > live` 时于写路径压缩回收。`GET /v1/collections/{name}/status`
  对 bm25 集合额外回显
  `sparse: {mode, doc_count, term_count, posting_count, dead_postings}`
  （`none` 时只回显 `mode`，避免被误读为全 0 统计）。
- 导出：`GET /v1/collections/{name}/export` 在 bm25 集合上为每行追加可选
  `"sparse":[[term,weight]…]`（加法字段，迁移不丢稀疏数据）。
- 可观测性：`/metrics` 新增 `houyidb_sparse_docs`、`houyidb_sparse_terms`、
  `houyidb_sparse_postings`、`houyidb_sparse_dead_postings`（跨集合聚合
  gauge）与 `houyidb_fusion_searches_total`（counter）；`/v1/version` 的
  capabilities 增加 `sparse_bm25`、`fusion_search`。
- 内存账目（如实，不得按更小单位成本规划）：每 (doc, term) 一条 posting
  约 24 B，正向索引每 (doc, term) 约 12–16 B，另有 term/doc 映射桶与扩容
  碎片；100 万文档 × 8 term ≈ postings 192 MB + 正向 100–150 MB + 映射
  40–60 MB ≈ 330–400 MB。以 status / metrics 实测为准。
- 并发：稀疏索引位于集合级 `RwLock`，写锁内更新（与标量页不嵌套），查询
  在读锁内同步计算，**不跨 `.await` 持锁**；无全局 Mutex、无分片。
- 版本纪律：collection 快照 payload **v3**（flags bit3=`0x08` 必置、v1/v2
  必须不置、旧二进制拒绝 v3）、manifest 条目 `sparse`（1=bm25）且
  `format_version=4`、复制协议 **v12**；旧二进制对 v3 / format 4 / v12
  干净拒绝。WAL 的稀疏字段是加法可选字段（`COLLECTION_WAL_VERSION` 仍为
  1），**旧二进制重放会静默忽略**——尚无快照时降级会丢稀疏索引，禁止把
  v12 WAL 交给旧二进制。

### 5.6 WAL 管理

```bash
GET  /v1/wal/status      # 主 WAL 状态 + collections 段/保留/manifest 状态
POST /v1/wal/checkpoint  # {"through_lsn": 42} → 截断 WAL
POST /v1/wal/rotate      # 归档当前 WAL
POST /v1/wal/snapshot    # 创建向量快照

# 存储生命周期（P5.0）
GET  /v1/collections/{name}/status   # hot/indexed/pending + applied_lsn + 标量统计
                                     # 一次性构图后端（clustered/flat-clustered/pq-flat）
                                     # 另回显 lazy_build:{built,buffered,rebuild_required}
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
复制协议 **v11** 在 v10 的 `collection_create` quantization（int8）语义之上
把 `rabitq` 纳入规范取值域，版本门 fail-fast：v10/v11 不能混跑，升级须全
集群协调（先停复制、全量升级、再恢复）。**禁止**把 v11 WAL 交给 v10 或更旧
二进制冷重放：v10 二进制会把 `rabitq` 串判为非规范值而在重放时 fail-fast
（不会静默降级，但握手阶段即拒绝）；v9 及更早二进制还会忽略 `quantization`
字段而把 int8 集合建成**未量化**图（静默分叉，重启后召回/内存与源端不一致），
并跳过未知 drop 记录、忽略度量与 dtype 字段而按 cosine/f32 重建，造成
「已删集合复活 + 度量/存储编码静默漂移」。v10 WAL 仍可被 v11 二进制读取
（向后兼容）；v11 写出的 int8 collection 快照（图快照 v3 / manifest v2）
与 rabitq 快照（图快照 v4 / manifest format 3）同样不可降级。
复制协议 **v12** 在 v11 之上新增稀疏字段（`collection_create.sparse` 与
upsert 条目可选 `sparse`）。v11 及更早二进制解码 v12 WAL 会**静默忽略**
这些加法字段——尚无快照时会丢失稀疏索引；因此 v12 与更早版本禁止混跑，
升级须全集群协调（先停复制、全量升级、再恢复）。**禁止**用 v11 或更旧
二进制重放 v12 WAL；v12 写出的 collection 快照 v3 与 manifest
`format_version=4` 会被旧二进制干净拒绝（版本范围 / 未知 format），
不会静默降级。

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
