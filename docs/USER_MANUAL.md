# HouyiDB 试用手册（v0.2.0-beta.9）

> [English](USER_MANUAL.en.md)

> 面向第一次接触 HouyiDB 的开发者：30 分钟内从启动走到「写入 → 检索 →
> 过滤 → 删除 → 重启恢复」。命令均已在本机 macOS 实测（Apple Silicon 原生 +
> Intel 包经 Rosetta 2 自检）。
>
> 项目定位：**单机、内存优先的向量数据库**。HNSW 向量检索 + 标量过滤 +
> pgwire/HTTP 双协议 + WAL 持久化。当前是 beta，适合功能验证与性能试用，
> 不建议直接承载关键生产数据。

---

## 0. 发布包里有什么

预编译包（macOS；Apple Silicon 与 Intel 各一份）：

```
houyidb-0.2.0-beta.9-<date>-darwin-arm64/     # Apple Silicon (M1/M2/M3/M4)
houyidb-0.2.0-beta.9-<date>-darwin-x86_64/    # Intel Mac
├── bin/houyi-server          # 主服务（约 10MB，无外部 C 依赖）
├── scripts/trial_start.sh    # 一键启动（默认 17000/15432）
├── scripts/trial_smoke.sh    # 自检：写入/检索/过滤/删除
├── scripts/embed_smoke.sh    # 向量化链路自检（需配置上游）
├── scripts/mock_embedding_server.py  # 无语义 mock 上游（只验链路）
├── examples/text_ingest.py   # 文本切块 → 自动入库 → 检索示例
├── USER_MANUAL.md            # 本文件
├── README.md  CHANGELOG.md  LICENSE  NOTICE
├── docs/                     # 服务端手册、部署指南、向量化指南
└── sdk/python/               # Python 客户端 + LangChain/LlamaIndex 适配器
```

源码包：`houyidb-0.2.0-beta.9-src.tar.gz`（任意平台；含 Dockerfile 与全部 SDK）。

> 请按 CPU 架构选择预编译包（`uname -m`：`arm64` 用 arm64 包，`x86_64` 用
> x86_64 包）。Linux/Windows 请用源码包 `cargo build` 或 Docker。

---

## 1. 三种启动方式

### 1.1 预编译包（macOS，最快）

Apple Silicon（M 系列）用 `darwin-arm64`，Intel Mac 用 `darwin-x86_64`：

```bash
uname -m        # arm64 → darwin-arm64；x86_64 → darwin-x86_64
tar xzf houyidb-0.2.0-beta.9-*-darwin-<arch>.tar.gz
cd houyidb-0.2.0-beta.9-*-darwin-<arch>

# 若二进制来自网络传输被 Gatekeeper 隔离，先解除（无签名 beta 的常见问题）：
xattr -dr com.apple.quarantine . 2>/dev/null || true

./scripts/trial_start.sh
```

看到 `HTTP : http://127.0.0.1:17000` 后，另开一个终端：

```bash
./scripts/trial_smoke.sh
```

### 1.2 Docker（任意平台，推荐给 Linux/Windows/WSL）

解压源码包后：

```bash
tar xzf houyidb-0.2.0-beta.9-src.tar.gz
cd houyidb-0.2.0-beta.9-src
docker compose up -d --build        # 首次会编译 Rust，约 5–15 分钟
curl http://127.0.0.1:7000/v1/health
```

> **macOS 注意**：系统 AirPlay 接收器会占用 5000/7000。若 7000 起不来，
> 改用宿主端口映射（容器内仍是 7000/5432）：
>
> ```bash
> docker run -d --name houyidb -p 17000:7000 -p 15432:5432 \
>   -v houyidb-data:/data houyidb:latest
> curl http://127.0.0.1:17000/v1/health
> ```
>
> 本手册后续命令默认 HTTP 为 `http://127.0.0.1:17000`；Docker 默认映射时
> 请把 17000 换成 7000。

### 1.3 源码构建（需要 Rust 1.87+）

```bash
tar xzf houyidb-0.2.0-beta.9-src.tar.gz
cd houyidb-0.2.0-beta.9-src
cargo build --release -p houyi-server
./scripts/trial_start.sh
```

验证：

```bash
curl -sS --noproxy '*' http://127.0.0.1:17000/v1/health
# {"status":"ok"}
```

端口被占用时可用环境变量覆盖：

```bash
HOUYI_TRIAL_HTTP_ADDR=127.0.0.1:18000 \
HOUYI_TRIAL_PGWIRE_ADDR=127.0.0.1:16432 \
./scripts/trial_start.sh
```

---

## 2. 五分钟上手（HTTP API）

数据模型：**Collection**（类似表，首次写入自动创建，维度由第一批向量决定；
也可用 `PUT` 显式创建并一次定型维度/度量/后端）→ 每个 Collection 内是
`id + 向量 + label + payload`。`id` 为 `u64`；字符串 id 可由 SDK 确定性哈希成 u64。

### 2.1 写入

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/upsert \
  -H 'Content-Type: application/json' -d '{
  "vectors": [
    {"id": 1, "values": [0.92,0.88,0.06,0.05,0.05,0.04,0.05,0.05],
     "label": "fruit", "payload": {"name": "apple", "price": 3}},
    {"id": 2, "values": [0.05,0.04,0.92,0.90,0.06,0.05,0.05,0.04],
     "label": "electronics", "payload": {"name": "laptop", "price": 999}},
    {"id": 3, "values": [0.04,0.05,0.05,0.06,0.90,0.92,0.05,0.04],
     "label": "book", "payload": {"name": "rust-book", "price": 45}}
  ]
}'
# {"inserted":3}
```

- `label` 是保留的顶层字段（字符串），会作为过滤列 `label` 使用；
  `payload` 里的 `label` 键会被拒绝（400）。
- 单请求体上限 64MB；批量越大吞吐越好。

### 2.1.1 显式创建 collection（可选）

隐式创建使用服务端默认后端与默认度量（hnsw → cosine）。需要 L2 / 内积 / L1
或自定义 HNSW 参数时，用 `PUT /v1/collections/{name}` 一次定型（普通写权限，
不需要 admin token）：

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/goods \
  -H 'Content-Type: application/json' -d '{
  "dims": 8,
  "metric": "ip",
  "index_type": "hnsw",
  "hnsw_m": 16, "hnsw_ef_construction": 100, "hnsw_ef_search": 64
}'
# 201 {"created":true,"collection":"goods","dims":8,"metric":"ip",
#      "index_type":"hnsw","hnsw":{"m":16,"ef_construction":100,"ef_search":64}}
```

- `metric`：`l2` / `cosine` / `ip` / `l1`，省略为后端默认；**创建后不可变**，
  改度量只能删除重建。
- `index_type`：`hnsw` / `flat` / `clustered` / `flat-clustered` / `pq-flat` /
  `hii-l0`，省略为服务端默认。支持矩阵：前四者支持全部四种度量；`pq-flat`
  仅 `l2`；`hii-l0` 仅 `cosine`。
- HNSW 三参数必须同时给出（`m` 1..1024，`ef_construction` / `ef_search`
  1..65536），且仅在 hnsw 后端有意义。
- `vector_dtype`：`f32`（缺省）/ `f16`。`f16` 把常驻向量内存减半，仅
  hnsw 后端支持，**创建后不可变**；查询向量始终以 f32 计算（非对称距离，
  f16 数据先舍入到 f16 网格）。非 f32 与非 hnsw 组合 → 400。响应与
  `GET .../status` 回显实际生效的 `vector_dtype`。
- `quantization`：`none`（缺省）/ `int8` / `rabitq`。**只压缩 HNSW 图侧向量
  副本**：主表 / hot buffer / 构建队列始终保留原始（f32 或 f16）向量；检索
  先用图候选（int8 候选宽度 ×2、rabitq ×16），再在原始向量上精确重排，最终
  top-k 分数不经过量化。仅 hnsw 后端支持，**创建后不可变**；非 hnsw 显式
  携带 `int8` / `rabitq` → 400；同名集合参数不一致 → 409。`int8` 为
  1 B/dim；`rabitq` 为 1 bit/dim（RaBitQ 二值量化 + 确定性随机旋转），
  **仅支持 `metric = l2 | cosine`**（`ip` / `l1` → 400），查询向量始终 f32
  （非对称估计）。响应与 `GET .../status` 回显实际生效的 `quantization`。
- 量化内存账目（如实，不得按「减半」规划）：主表内存不变（图侧 `source` 与
  主表共享 `Arc`）。图侧码流 = `ceil(D/64)·8` B + 8 B 因子，D 为 dims
  向上取 2 的幂；dims=768（D=1024）时每向量 f32 3072 B / int8 768 B +
  8 B / rabitq 128 B + 8 B。rabitq 相对 f32 图侧仅多 ~136 B（≈1/24），
  相对 int8 降至约 1/6；另需每集合共享的旋转符号 4·D 字节与每查询一个
  D×f32 临时缓冲。余弦场景原始与归一化副本合一，总内存较未量化下降
  （int8 ≈ −3 B/dim、rabitq ≈ −3.8 B/dim）；非余弦总内存略增
  （int8 ≈ +1 B/dim、rabitq ≈ +0.18 B/dim）。收益在图内距离工作集与
  内存带宽。
- `sparse`：`none`（缺省）/ `bm25`。`bm25` 为集合建立旁路倒排索引，支持
  稀疏写入与融合检索；**仅 hnsw / flat 后端**，**创建后不可变**，隐式创建
  恒为 `none`。响应与 `GET .../status` 回显实际生效的 `sparse`；同名集合
  改变 sparse → 409；`bm25` + 其它后端 → 400。稀疏索引不改变 dense 检索
  结果，主表 / hot buffer / 构建队列仍只保存稠密向量。
- 同名已存在 → 409（参数一致也返回 409）；参数非法 → 400；follower/只读 →
  503。

f16 示例（内存敏感、可接受 ~10⁻³ 舍入误差的场景）：

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/imgf16 \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "hnsw", "vector_dtype": "f16"
}'
# 201 {"created":true,"collection":"imgf16",...,"vector_dtype":"f16"}
```

int8 图侧量化示例（保留 f32 主表，检索仍精确重排）：

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/imgq \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "hnsw",
  "vector_dtype": "f32", "quantization": "int8"
}'
# 201 {"created":true,"collection":"imgq",..."quantization":"int8"}
```

rabitq 图侧量化示例（1 bit/dim，仅 l2/cosine；检索仍精确重排）：

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/imgq1bit \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "hnsw",
  "quantization": "rabitq"
}'
# 201 {"created":true,"collection":"imgq1bit",..."quantization":"rabitq"}
# metric=ip/l1 搭配 rabitq → 400；非 hnsw 后端携带 rabitq → 400
```

bm25 稀疏索引示例（旁路倒排索引，支持稀疏写入与融合检索）：

```bash
curl -sS --noproxy '*' -X PUT http://127.0.0.1:17000/v1/collections/docs \
  -H 'Content-Type: application/json' -d '{
  "dims": 512, "metric": "cosine", "index_type": "flat", "sparse": "bm25"
}'
# 201 {"created":true,"collection":"docs",...,"sparse":"bm25"}
# 非 hnsw/flat 后端携带 bm25 → 400；隐式创建（首次 upsert）恒为 sparse=none
```

### 2.2 向量检索

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/search \
  -H 'Content-Type: application/json' -d '{
  "vector": [0.90,0.85,0.05,0.05,0.05,0.05,0.05,0.05],
  "k": 3,
  "include_payload": true
}'
# {"ids":[1,3,2],"scores":[...],"payloads":{"1":{...},"3":{...},"2":{...}}}
```

- `scores` 与 `ids` 等长，越大越相似；换算由 collection 的 `metric` 决定：
  `cosine` → `1 - cosine_distance`（钳制到 `[-1,1]`）；`l2` / `l1` →
  `1 - distance`（不截断）；`ip` → 点积本身（越大越相关）。响应同时回传
  `metric` 字段。
- `ef_search` 可选，越大召回越高、延迟越高。

### 2.3 向量 + 标量过滤（混合检索）

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/hybrid/search \
  -H 'Content-Type: application/json' -d '{
  "vector": [0.90,0.85,0.05,0.05,0.05,0.05,0.05,0.05],
  "k": 3,
  "filters": [{"column": "price", "op": "lt", "value": 10}],
  "include_payload": true
}'
# 只剩水果类：{"ids":[1],...}
```

过滤操作符：`eq` / `gt` / `lt` / `gte` / `lte` / `in`（`in` 的 `value`
必须是 JSON 数组）。多个 `filters` 之间是 AND。`label` 可直接作为列名。
**过滤检索仅 hnsw / flat 后端支持**（其他后端返回 400）。

### 2.3.1 稀疏向量（BM25）与融合检索

稀疏词项由调用方给定（**内核不做分词**）：每项是 `(term_id, weight)`，
`term_id` 为 `1..=2^24-1` 的整数，`weight` 为有限正数。写入与查询都接受
`[[term_id, weight], …]` 的 JSON 数组；重复 term 由服务端求和，顺序不影响
结果。单文档最多 1024 对、单查询最多 64 对，越界 400。

写入（须先在 `sparse=bm25` 的集合上；在 `sparse=none` 集合上带 `sparse` 键
一律 400，不会静默丢弃）：

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/docs/upsert \
  -H 'Content-Type: application/json' -d '{
  "vectors": [
    {"id": 1, "values": [0.9,0.1,0.0,0.0], "label": "fruit",
     "payload": {"text": "apple is a fruit"},
     "sparse": [[7, 1.0], [8, 0.8], [11, 0.5]]},
    {"id": 2, "values": [0.1,0.9,0.0,0.0], "label": "device",
     "payload": {"text": "laptop is a device"},
     "sparse": [[7, 0.6], [9, 1.0]]}
  ]
}'
# {"inserted":2}
```

- `sparse` 键**缺省 / `null`** = 不改该文档的稀疏词项；**`[]`** = 清空。
- 更新语义是**整文档替换**：同一 id 再次 upsert 会以新词项覆盖旧词项，
  不是增量合并（幂等重放安全）。
- 按 id / 过滤器删除会同步移除稀疏词项，删除后立即不可检索。

融合检索（dense 与 sparse 同时必填；集合须为 `sparse=bm25`）：

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/docs/fusion/search \
  -H 'Content-Type: application/json' -d '{
  "vector": [0.9,0.1,0.0,0.0],
  "sparse": [[7, 1.0], [9, 0.5]],
  "k": 2,
  "fusion": "rrf",
  "include_payload": true
}'
# {"ids":[1,2],"scores":[...],"dense_scores":[0.99,null],
#  "sparse_scores":[2.1,1.7],"metric":"cosine",
#  "fusion":{"mode":"rrf","rrf_k":60,"alpha":null},"payloads":{...}}
```

- `fusion`：`rrf`（缺省）或 `weighted`。`rrf_k` 缺省 60，范围 `1..=1000`；
  `alpha` 缺省 0.5，范围 `[0,1]`（仅 `weighted` 生效）。显式提供但当前模式
  不用的参数也会校验（非法一律 400）。
- **`scores` 只是融合排序分，不属于任何度量口径**：RRF 为
  `Σ 1/(rrf_k + rank)`（rank 从 1 起，仅命中该路的列表各计一项）；加权为
  `alpha * norm(dense) + (1-alpha) * norm(sparse)`，两路各自在过滤后的候选集
  上做 min-max 归一化（单元素列表归一为 1.0，缺失模态记 0）。
- `dense_scores` 才按集合 `metric` 换算（口径同 2.2），`sparse_scores`
  为 BM25 原始分；某一路未命中的文档对应位置为 `null`。`metric` 仅用于
  解释 `dense_scores`，`fusion` 块回显生效模式与实际使用的参数。
- `filters` / `ef_search` / `include_payload` 与普通检索同义；过滤仅
  hnsw / flat 后端支持，两路候选都会被过滤。
- 两路各取 `4×k` 候选后融合再截前 `k`；`k` / `ef_search` 上限与普通检索相同。
- 错误码：缺集合 404；`sparse=none` 集合或 `sparse` 为空 400；维度、词项、
  融合参数越界 400；只读 / follower / fencing 503。
- BM25（`k1=1.2`、`b=0.75`）：`idf = ln(1 + (N-df+0.5)/(df+0.5))`，
  `tf` 即客户端权重，`dl = Σ权重`，`avgdl = Σdl / N`；N=0 返回空。
  分数越高越相关。`sparse` 集合的 `status` 额外回显
  `doc_count` / `term_count` / `posting_count` / `dead_postings`。
- 稀疏数据只经 HTTP 与 SDK 写入；pgwire 的 AUB1 批量行不含稀疏字段
  （见第 4 节，pgwire 仅提供融合**查询**）。
- 内存账目（每 (doc, term) 一条 posting）：posting 约 24 B，正向索引每
  (doc, term) 约 12–16 B，另有 term/doc 映射桶与扩容碎片。100 万文档 × 8
  term 约为 postings 192 MB + 正向 100–150 MB + 映射 40–60 MB ≈ 330–400 MB；
  以 `status` / `dead_postings` 实测为准，**不要按更小的字节数规划**。死
  posting 在 `dead > live` 时于写路径压缩回收，`dead_postings` 暴露放大程度。

### 2.4 删除与状态

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/delete \
  -H 'Content-Type: application/json' -d '{"ids": [2]}'
# {"deleted":1}

curl -sS --noproxy '*' http://127.0.0.1:17000/v1/collections/goods/status
# {"name":"goods","vector_count":2,"pending_count":0,"dims":8,...}
```

删除也支持 `{"filters":[...]}`（按条件删，与显式 id 取并集）。

### 2.4.1 删除整个 collection（管理员，不可逆）

```bash
# RBAC 开启时使用 admin token；未开启鉴权时可省略 Authorization
curl -sS --noproxy '*' -X DELETE \
  -H "Authorization: Bearer $HOUYI_ADMIN_TOKEN" \
  http://127.0.0.1:17000/v1/admin/collections/goods
# {"deleted":true,"collection":"goods"}
```

- **不可逆**：无条件删除（不要求 collection 为空，也没有二次确认参数）。
  删除前建议先导出备份：`GET /v1/collections/{name}/export`。
- **权限**：路径归属 `/v1/admin/*`，需要 admin 主体（`Action::Admin`）；审计
  日志自动记录 actor/路由/状态。
- **持久化与复制**：WAL-first（`collection_drop`），崩溃重放与 follower 都会
  收敛到「已删除」；删除后的同名 upsert 会创建全新的空 collection。
- **状态码**：不存在或重复删除 404；并发删除竞争 409；follower/只读模式 503。

### 2.5 其它常用端点

| 端点 | 作用 |
|------|------|
| `GET /v1/health` | 健康检查（认证豁免） |
| `GET /readyz` | 就绪检查（认证豁免） |
| `GET /v1/openapi.json` | OpenAPI 文档（可贴进 Swagger UI） |
| `GET /v1/collections` | 列出 Collection |
| `PUT /v1/collections/{name}` | 显式创建 collection（写权限；一次定型 dims/度量/后端/HNSW） |
| `POST /v1/collections/{name}/fusion/search` | dense+sparse 融合检索（集合须 `sparse=bm25`；`scores` 仅排序语义） |
| `GET /v1/collections/{name}/export` | 流式 NDJSON 逻辑导出（header/记录/footer，备份/迁移用） |
| `DELETE /v1/admin/collections/{name}` | 删除整个 collection（管理员，不可逆，WAL-first） |
| `GET /metrics` | Prometheus 指标（认证豁免；含 per-collection `houyidb_collection_vectors` gauge） |
| `GET /v1/version` | 版本/能力探测；`build_sha` 为构建时 Git 短 SHA |
| `GET /v1/index/status` | 主索引（/v1/vectors 路径）状态 |
| `POST /v1/optimize` | 排空构图队列 / 触发压缩 |
| `POST /v1/wal/checkpoint` | WAL checkpoint |
| `POST /v1/vectors/upsert`、`/v1/vectors/search` | 单主索引兼容 API（维度固定为启动参数） |

### 2.6 文本向量化（可选，OpenAI 兼容网关）

内核不做 embedding 推理；配置一个 OpenAI 兼容上游后，`/v1/embeddings` 把文本
转成向量，并可用 `collection` 扩展字段**自动入库**。

Ollama（离线推荐）：

```bash
ollama pull bge-m3
HOUYI_OPENAI_UPSTREAM=http://127.0.0.1:11434 ./scripts/trial_start.sh
# 另一个终端：
EMBED_MODEL=bge-m3 ./scripts/embed_smoke.sh
EMBED_MODEL=bge-m3 python3 examples/text_ingest.py
```

OpenAI（需 key）：

```bash
HOUYI_OPENAI_UPSTREAM=https://api.openai.com \
HOUYI_OPENAI_API_KEY=sk-... ./scripts/trial_start.sh
EMBED_MODEL=text-embedding-3-small ./scripts/embed_smoke.sh
```

代码里自动入库：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:17000/v1", api_key="not-needed")
client.embeddings.create(model="bge-m3", input=["苹果是水果"],
                         extra_body={"collection": "docs"})
# resp.houyi = {"collection": "docs", "upserted": 1}
```

- id = 文本的 `stable_id`（重跑幂等）；原文进 payload `text`；
- **相同文本会互相覆盖**；不能携带业务元数据；换模型需要重建 collection；
- 完整配方（Ollama/TEI/OpenAI、模型选型、限制与排障）见
  `docs/EMBEDDING_GUIDE.md`。

---

## 3. Python SDK

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e sdk/python/houyidb
```

```python
from houyidb import HouyiClient, Vector, Filter

client = HouyiClient("http://127.0.0.1:17000")
col = client.collection("goods")          # 首次 upsert 隐式创建

col.upsert([
    Vector(id=1, values=[0.92, 0.88, 0.06, 0.05, 0.05, 0.04, 0.05, 0.05],
           label="fruit", payload={"name": "apple", "price": 3}),
])

hits = col.search([0.90, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                  3, include_payload=True)
print(hits)

hits = col.hybrid_search(
    [0.90, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], 3,
    filters=[Filter("price", "lt", 10)], include_payload=True)
print(hits)

# 稀疏 + 融合（须显式建 sparse="bm25" 集合，语义见 2.3.1）
client.create_collection("docs", 8, index_type="flat", sparse="bm25")
docs = client.collection("docs")
docs.upsert([
    Vector(id=1, values=[0.92, 0.88, 0.06, 0.05, 0.05, 0.04, 0.05, 0.05],
           sparse=[(7, 1.0), (8, 0.8)]),
])
hits = docs.fusion_search(
    [0.90, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [(7, 1.0), (9, 0.5)], 3,
    include_payload=True)
print(hits)

col.delete([1])
```

- 字符串 id 会自动哈希成稳定的 u64（跨语言一致），可直接用业务主键。
- LangChain / LlamaIndex 适配器在同目录：
  `pip install -e sdk/python/langchain-houyidb` /
  `pip install -e sdk/python/llama-index-vector-stores-houyidb`。
  用法见各自包内 README（embedding 由框架/调用方计算，HouyiDB 只存取向量）。

---

## 4. pgwire（PostgreSQL 协议）

任意 PostgreSQL 客户端可直连，无需真实建库建表。认证关闭时用户/密码随意。

```bash
psql -h 127.0.0.1 -p 15432 -U trial -d trial
```

```sql
-- 以注释传递查询向量；k 由 LIMIT 指定
/* houyi:vector=0.92,0.88,0.06,0.05,0.05,0.04,0.05,0.05 */
SELECT id FROM goods LIMIT 3;

-- 带 id 过滤
/* houyi:vector=0.92,0.88,0.06,0.05,0.05,0.04,0.05,0.05 */
SELECT id FROM goods WHERE id IN (1, 3) LIMIT 3;
```

- pgwire 主索引的维度是启动时的 `HOUYI_TRIAL_DIM`（默认 768）。
- 批量写入使用 Extended Query + `/* houyi:upsert_batch=$1 */` 二进制协议，
  细节见包内 `docs/HOUYIDB_SERVER_MANUAL.md`。
- 开启 `HOUYI_AUTH_TOKEN` 后，pgwire 密码 = 该 token。

### 4.1 融合检索指令（P4，仅查询）

```sql
-- sparse 与 vector 指令必须同时出现；collection 用 houyi:collection 指定
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ SELECT id FROM docs LIMIT 2;

-- 加权融合（alpha 仅 weighted 生效；fusion 缺省 rrf、alpha 缺省 0.5）
/* houyi:collection=docs */ /* houyi:vector=0.9,0.1,0.0,0.0 */ \
  /* houyi:sparse=7:1.0,9:0.5 */ /* houyi:fusion=weighted */ \
  /* houyi:alpha=0.4 */ SELECT id FROM docs LIMIT 2;
```

- `sparse` 词项格式 `term:weight[,term:weight…]`，重复 term 求和；
  缺 `vector` 指令或只有 fusion/alpha 参数 → 语法错误（SQLSTATE 42601）。
- 集合必须已用 HTTP/SDK 建为 `sparse=bm25` 并带 `houyi:collection` 指令：
  缺 collection 指令或集合未启用稀疏 → `0A000`；集合不存在 → `42P01`
  （pgwire 不会隐式建集合）；词项 / alpha / fusion / k 语义越界 → `22023`。
- pgwire **不提供稀疏写入**：AUB1 批量行只有 `id + dense 向量`，稀疏词项
  请走 HTTP `upsert` 或任一 SDK。
- 融合查询结果仍是单列 `id`（oid 20），与普通向量检索一致。

---

## 5. 持久化、重启与恢复

- **WAL-first**：每次 upsert/delete 先落盘再改内存；进程被 kill -9 后重启，
  已确认的写入会重放恢复。
- Collection 数据保存在 `data/`（试用脚本默认 `./data/houyidb.wal` 与
  `houyidb.collections.wal`）。停止与重启：

```bash
# Ctrl-C 优雅停止
./scripts/trial_start.sh          # 再次启动，自动重放 WAL
```

- 周期图快照默认 600s 一次；关停时也会写图快照
  （`HOUYI_SHUTDOWN_SNAPSHOT=0` 可关）。
- **重置试用数据**（会丢数据）：停止服务后 `rm -rf data/`。
- collections WAL **有界保留**（P5.0）：默认每 16 段 × 64 MiB 从 manifest 的
  safe checkpoint 起按整段删除；follower 由已确认 durable LSN 保护 floor。
  全部六种索引后端均可快照压缩，正常运行时不应再出现
  `unsupported_backend` 的 `blocked_reason`。

---

## 6. 配置与性能档位

### 6.1 常用启动参数

| 参数 / 环境变量 | 默认 | 说明 |
|---|---|---|
| `--benchmark-http-addr` / `HOUYI_SERVER_BENCHMARK_HTTP_ADDR` | `127.0.0.1:7000` | HTTP 监听地址 |
| `--pgwire-addr` / `HOUYI_SERVER_PGWIRE_ADDR` | `127.0.0.1:5432` | pgwire 监听地址 |
| `--vector-dim` / `HOUYI_SERVER_VECTOR_DIM` | `1536` | `/v1/vectors` 与 pgwire 主索引维度 |
| `--index-type` / `HOUYI_SERVER_INDEX_TYPE` | `hnsw` | `hnsw`/`clustered`/`flat-clustered`/`flat`/`pq-flat`/`hii-l0` |
| `--wal-path` / `HOUYI_SERVER_WAL_PATH` | 临时目录 | WAL 文件路径 |
| `--wal-write-mode` / `HOUYI_SERVER_WAL_WRITE_MODE` | `async` | `async` / `sync`（sync 更稳、更慢） |
| `--snapshot-interval-secs` | `600` | 周期图快照；0=关闭 |
| `--max-http-inflight` | `512` | 在途请求上限，超限 503 |
| `--config <toml>` | 无 | TOML 配置（RBAC 等） |

### 6.2 HNSW 旋钮（按场景选）

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `HOUYI_HNSW_LAYER0_MULT` | `2` | L0 连接数乘数。`1`＝高 QPS（约 +35–60%），召回约 −0.9pp；`2`＝召回优先 |
| `HOUYI_HNSW_UPPER_MULT` | `0.5` | 上层度数乘数 |
| `HOUYI_HNSW_PLAN_APPLY` | 开 | 读锁规划 + 短写锁提交；`0/false/off/no/空` 回退旧整图持锁路径（批量加载快约 11%，流式检索变慢） |
| `HOUYI_HNSW_PARALLEL_BATCH_BUILD` | 服务端 `1` | Rayon 并行构图 |
| `HOUYI_HNSW_PARALLEL_CHUNK_SIZE` | `256` | 规划 chunk 大小 |
| `HOUYI_HNSW_PARALLEL_PLAN_GROUP_CHUNKS` | `4` | 每组 chunk 数（配置组宽 = chunk × 本值，自适应时作为上限） |
| `HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR` | `256` | 自适应组宽上限 = `max(plan_group_min, 图点数/该值)`，再与配置组宽取小；`0` 关闭自适应、固定用 chunk × group_chunks |
| `HOUYI_HNSW_PLAN_GROUP_MIN` | `16` | 自适应组宽**上限**的下限：`max(本值, 图点数/divisor)`，再与配置组宽取小（最终组宽可低于本值） |
| `HOUYI_HNSW_MAX_BUILD_BATCH` | 服务端 `8192` | 单批最大构图数 |
| `HOUYI_HNSW_GROUP_REPAIR` | `4,16` | 分组提交后的孤点补救：对 L0 度低于 `min_degree` 的组员按 id 邻域 ±`window` 补连（仅分组提交路径；格式 `min[,window]`，window 缺省 16；`off`/`0`/空关闭供 A/B；非法值告警后回落默认） |
| `HOUYI_HNSW_UPPER_EF` | `64` | 查询侧上层下降束宽**下限**（只影响检索入口导航，不改构图语义）：自适应开启时实际宽度为 `max(本值, ceil(层节点估计/8))`、自适应部分封顶 4096（250k 约 12.5%、1M 约 6.5% 的层覆盖率）；本值可显式设为 `1..=4096`（越高越慢），用于小图强制宽束做 A/B。非法值告警后回落 64。构图侧有意保持 1 |
| `HOUYI_HNSW_UPPER_EF_ADAPTIVE` | 开 | 上层束宽自适应开关；`0/false/off/no` 关闭后束宽恒为 `HOUYI_HNSW_UPPER_EF`（延迟敏感部署/回滚用；关闭后 50k 成簇语料可能重现错簇，仅建议 A/B） |
| `HOUYI_HNSW_KEEP_PRUNED` | `0` | keepPruned 连接（召回略升，构图/查询变慢；不建议大库开启） |

> 建议：先用默认值。追求 QPS 时设 `HOUYI_HNSW_LAYER0_MULT=1`；追求召回
> 用默认 2×。两者不要在同一份旧图快照上直接切换（快照兼容性会触发重建）。
>
> 规划组宽：批量构图时每组只对**组前图快照**规划近邻，组内兄弟互相不可见；
> 成簇数据里同批兄弟互为最近邻，固定大组会让它们在提交后被对称剪枝剪成
> 孤岛，召回随 ef 增大也上不去。默认自适应组宽**上限**约为图规模的 1/256
> 且不低于 16，最终组宽再与配置组宽取小；在小图上保质量、大图上保吞吐；仅在明确要压吞吐时才调大
> `HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR` 或用 `0` 关闭自适应。
>
> 大规模成簇集合上（如单 collection 5 万点、64 簇每簇约 800 点），自适应组宽
> （约 195）仍小于簇规模：整组兄弟会被剪成入口点不可达的口袋，表现为**整簇
> 查询返回错误簇且加大 ef 几乎不改善**。因此分组提交后还会对 L0 度低于 4 的
> 组员补连到 id 邻域（±16）内的健壮兄弟（`HOUYI_HNSW_GROUP_REPAIR`，默认
> `4,16`）；查询侧上层下降束宽按层规模自适应（下限 `HOUYI_HNSW_UPPER_EF`，
> 默认 64；估计层节点数为 `图点数/M^层`，束宽取估计值/8 且自适应部分封顶
> 4096，仅用于查询入口导航；自适应会略增查询开销，可用
> `HOUYI_HNSW_UPPER_EF_ADAPTIVE=0` 关闭做延迟 A/B，但关闭后 50k 成簇语料可能
> 重现错簇），保证跨簇导航不落入错误
> 局部极小。上限 4096 在 1M 规模仍维持约 6.5% 覆盖率；更大的图如需更宽束，
> 可将 `HOUYI_HNSW_UPPER_EF` 显式设为 4096（查询成本随之上升）。除此之外，构图排空队列后
> 会对**每一层**做一次连通性兜底：从入口点逐层 BFS，把断开的层内分量接到
> 向量距离最近、仍有余量的可达锚点（L0 与上层同规则）；上层孤点若不补回，
> 查询下降会直接落到错误簇的 L0 入口，加宽 ef 也救不回来。
> 50k 大簇回归门禁：`./scripts/clustered_recall_50k_gate.sh`（release，含默认、
> legacy 串行、legacy HTTP、生产形态四条用例）。

### 6.3 过滤

`label` 走精确倒排索引；命中量大时自动构建按标签的连续向量缓存，过滤检索
退化为顺序 SIMD 扫描。无需配置，写后自动失效重建。

---

## 7. 可观测性

```bash
curl -sS --noproxy '*' http://127.0.0.1:17000/metrics | head -n 40
curl -sS --noproxy '*' http://127.0.0.1:17000/v1/index/status
curl -sS --noproxy '*' http://127.0.0.1:17000/v1/collections/goods/status
```

`/metrics` 为 Prometheus 文本格式，可直接接 Grafana；`houyidb_collection_vectors{collection=...}`
给出每个 collection 的已索引向量数。`/v1/version` 的 `build_sha` 可确认实例对应的构建提交。服务端日志输出到
stdout/stderr（Rust `tracing`，`RUST_LOG=info` 可调）。

---

## 8. 安全（局域网试用必读）

默认只监听 `127.0.0.1`，本机试用无需认证。**要让朋友/其它机器访问时**：

```bash
HOUYI_AUTH_TOKEN='请换成长随机串' \
HOUYI_TRIAL_HTTP_ADDR=0.0.0.0:17000 \
./scripts/trial_start.sh
```

- 之后所有 `/v1/*`（除 `/v1/health`、`/readyz`）需
  `Authorization: Bearer <token>` 或 `X-Api-Key: <token>`。
- 更细粒度权限用 TOML `[[auth_principals]]`（admin/writer/reader +
  collection 作用域），见[部署指南的 TOML 配置示例](DEPLOYMENT_GUIDE.md)。
- 公网暴露请加 TLS（`--tls-cert` / `--tls-key`）与反向代理；不要在公网
  裸跑无认证实例。

---

## 9. 已知限制（beta）

- 单机架构，无分片；高可用是 leader/follower 手动 promote，无自动选主。
- collections WAL 有界保留（默认 16 段 × 64 MiB）。follower 落后到保留范围
  之外会收到 `BEHIND_RETAINED`，必须按
  `docs/runbooks/collections-retention-resync.md` 做全量重同步；不会自动清库。
  复制协议 v12 不做跨版本滚动兼容，升级需所有副本一起进行（见服务手册）。
- 标量页为 append-only：重复 upsert 同一 id 会追加新行，过滤与 payload 回读
  只看每个 id 的最后一行（旧值从写入起即不再命中）。默认每 10 分钟检查、
  stale 行 ≥50% 时压实回收物理行，压实不改变查询结果。
- 全部索引后端均支持 collection 快照压缩。懒构建后端（clustered /
  flat-clustered / pq-flat）是**一次性构图**：恢复后不自动构图，重启前未构图的
  集合重启后同样检索为空，直到显式构图；pq-flat 已建态按紧凑态逐位恢复、
  不重新训练，构图后的待建缓冲行原样保留。
  **已建态（built=true）之后的新写入只进待建缓冲，不会自动并入索引、检索不可见**；
  需删除重建集合才能把新数据入图。`GET /v1/collections/{name}/status` 的
  `lazy_build` 字段回显 `built` / `buffered` / `rebuild_required`
  （`rebuild_required=true` 即已建态仍有缓冲写；非惰性后端省略该字段）。
- payload 的数组/对象值按 JSON 文本列处理，过滤语义有限。
- 后端与度量支持矩阵：hnsw / flat / clustered / flat-clustered 支持
  l2/cosine/ip/l1；`pq-flat` 仅 `l2`（HTTP score 为 `1 − L2²`，无界）；
  `hii-l0` 仅 `cosine`。
- HII L1（GPU CAGRA）/ L2（NVMe DiskANN）未实现；`simd-avx512` 仅骨架。
- 过滤检索仅 hnsw / flat 后端支持。
- 稀疏词项由客户端提供，内核**不做分词**；`sparse=bm25` 仅 hnsw / flat 后端，
  创建后不可变，单文档 ≤1024 对、查询 ≤64 对，`term_id` 限
  `1..=2^24-1`。稀疏写入只有 HTTP/SDK，pgwire 只支持融合查询。
- **降级禁令（稀疏）**：collection 快照 v3、manifest `format_version=4`、
  复制协议 v12 会让旧二进制干净拒绝（不会静默降级）。但 collections WAL
  的稀疏字段是加法可选字段，旧二进制重放时会**静默忽略**——若尚未生成快照
  就降级，稀疏索引会丢失。降级前必须先完成快照，或确认没有
  `sparse=bm25` 集合。
- 单请求体上限 64MB。

---

## 10. 故障排查

| 现象 | 处理 |
|------|------|
| `Address already in use` | 换端口：`HOUYI_TRIAL_HTTP_ADDR=127.0.0.1:18000`（macOS 的 5000/7000 常被 AirPlay 占用） |
| macOS 提示「无法验证开发者」 | `xattr -dr com.apple.quarantine .`，或右键 → 打开 |
| Docker 构建很久 | 正常：容器内从源码编译 Rust；之后会走缓存 |
| 启动后健康检查失败 | 看服务端 stderr；确认端口与 WAL 目录可写 |
| 检索维度报 400 | 查询向量长度必须等于该 Collection 首次写入的维度 |
| 过滤返回空 | 确认后端为 hnsw/flat；确认列名存在且 `op` 合法 |
| 想彻底重置 | 停止服务后 `rm -rf data/` |

---

## 11. 反馈

试用中遇到问题，请把以下信息带回：

```bash
uname -a
./bin/houyi-server --help        # 或 cargo run -p houyi-server -- --help
# 复现命令、请求体、响应、服务端日志（stdout/stderr）
```

祝试用顺利。
