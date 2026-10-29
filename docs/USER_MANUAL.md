# HouyiDB 试用手册（v0.2.0-beta.5）

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
houyidb-0.2.0-beta.5-<date>-darwin-arm64/     # Apple Silicon (M1/M2/M3/M4)
houyidb-0.2.0-beta.5-<date>-darwin-x86_64/    # Intel Mac
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

源码包：`houyidb-0.2.0-beta.5-src.tar.gz`（任意平台；含 Dockerfile 与全部 SDK）。

> 请按 CPU 架构选择预编译包（`uname -m`：`arm64` 用 arm64 包，`x86_64` 用
> x86_64 包）。Linux/Windows 请用源码包 `cargo build` 或 Docker。

---

## 1. 三种启动方式

### 1.1 预编译包（macOS，最快）

Apple Silicon（M 系列）用 `darwin-arm64`，Intel Mac 用 `darwin-x86_64`：

```bash
uname -m        # arm64 → darwin-arm64；x86_64 → darwin-x86_64
tar xzf houyidb-0.2.0-beta.5-*-darwin-<arch>.tar.gz
cd houyidb-0.2.0-beta.5-*-darwin-<arch>

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
tar xzf houyidb-0.2.0-beta.5-src.tar.gz
cd houyidb-0.2.0-beta.5-src
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
tar xzf houyidb-0.2.0-beta.5-src.tar.gz
cd houyidb-0.2.0-beta.5-src
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

数据模型：**Collection**（类似表，首次写入自动创建，维度由第一批向量决定）
→ 每个 Collection 内是 `id + 向量 + label + payload`。`id` 为 `u64`；
字符串 id 可由 SDK 确定性哈希成 u64。

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

- `scores` 与 `ids` 等长，越大越相似（COSINE 后端为 `1 - cosine_distance`，
  钳制到 `[-1,1]`）。
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

### 2.4 删除与状态

```bash
curl -sS --noproxy '*' -X POST http://127.0.0.1:17000/v1/collections/goods/delete \
  -H 'Content-Type: application/json' -d '{"ids": [2]}'
# {"deleted":1}

curl -sS --noproxy '*' http://127.0.0.1:17000/v1/collections/goods/status
# {"name":"goods","vector_count":2,"pending_count":0,"dims":8,...}
```

删除也支持 `{"filters":[...]}`（按条件删，与显式 id 取并集）。

### 2.5 其它常用端点

| 端点 | 作用 |
|------|------|
| `GET /v1/health` | 健康检查（认证豁免） |
| `GET /readyz` | 就绪检查（认证豁免） |
| `GET /v1/openapi.json` | OpenAPI 文档（可贴进 Swagger UI） |
| `GET /v1/collections` | 列出 Collection |
| `GET /metrics` | Prometheus 指标（认证豁免） |
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
- 已知限制：collections WAL **只增不减**，尚无压实/截断；长期大量写入会让
  该文件持续增长（试用阶段可定期重置）。

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
| `HOUYI_HNSW_MAX_BUILD_BATCH` | 服务端 `8192` | 单批最大构图数 |
| `HOUYI_HNSW_KEEP_PRUNED` | `0` | keepPruned 连接（召回略升，构图/查询变慢；不建议大库开启） |

> 建议：先用默认值。追求 QPS 时设 `HOUYI_HNSW_LAYER0_MULT=1`；追求召回
> 用默认 2×。两者不要在同一份旧图快照上直接切换（快照兼容性会触发重建）。

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

`/metrics` 为 Prometheus 文本格式，可直接接 Grafana。服务端日志输出到
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
  collection 作用域），见[部署指南的 TOML 配置示例](https://github.com/houyidb-community/houyidb/blob/main/docs/DEPLOYMENT_GUIDE.md)。
- 公网暴露请加 TLS（`--tls-cert` / `--tls-key`）与反向代理；不要在公网
  裸跑无认证实例。

---

## 9. 已知限制（beta）

- 单机架构，无分片；高可用是 leader/follower 手动 promote，无自动选主。
- collections WAL 只增不减（无压实）。
- 重复 upsert 时标量页 append-only，会保留旧行；检索/过滤按 id 去重。
  payload 的数组/对象值按 JSON 文本列处理，过滤语义有限。
- `pq-flat` 后端的距离全链路为 L2，HTTP score 为 `1 − L2²`（无界）。
- HII L1（GPU CAGRA）/ L2（NVMe DiskANN）未实现；`simd-avx512` 仅骨架。
- 过滤检索仅 hnsw / flat 后端支持。
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
