# HouyiDB v0.2.0-beta.8

> [English](README.en.md)

<p align="center"><img src="docs/assets/logo-h-zh-red.png" alt="HouyiDB 后羿向量数据库" width="420"></p>

单机、内存优先的向量数据库：HNSW 向量检索 + PAX 行列混存 + pg_wire/HTTP 双接口 + WAL 持久化。

![HouyiDB 60 秒演示：写入、检索、过滤、kill -9 后重启恢复](demo/houyidb-demo.gif)

> 第一次使用？看 [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md)（30 分钟从上手到重启恢复）。

## 快速启动

### 0) Docker（最简单）

```bash
docker compose up -d        # 或：docker build -t houyidb . && docker run -d -p 7000:7000 -p 5432:5432 -v houyidb-data:/data houyidb
curl http://127.0.0.1:7000/v1/health
```

HTTP API 的交互式文档：`http://127.0.0.1:7000/v1/openapi.json`（可贴进 Swagger UI 浏览）。

### 1) 从源码启动主服务

```bash
cargo run -p houyi-server --bin houyi-server
```

可选参数（CLI > ENV > 默认值）：

```bash
cargo run -p houyi-server --bin houyi-server -- \
  --worker-threshold 256 \
  --worker-tick-ms 20 \
  --flush-policy hybrid \
  --metrics-interval-secs 1
```

### 2) 性能冒烟与基准

```bash
./scripts/perf_smoke_fast.sh    # 快速（跳过 recall 组）
./scripts/perf_smoke.sh         # 完整（含 recall_check_hnsw_after_flush）

# Recall 门禁（L2 真值，默认 mean@k ≥ 0.99）
cargo run -p houyi-vector --example recall_gate --release
```

VectorDBBench 复现协议见 `VDBBENCH_PROTOCOL.md` 与
`benchmarks/vectordb_bench_houyidb/`；本仓公开报告只含 HouyiDB 自测数据，
不发布第三方系统对比结果。

HNSW 连接密度：服务端 level-0 默认 **2×M**（召回优先，兼容旧图快照）；
高 QPS 场景显式设 `HOUYI_HNSW_LAYER0_MULT=1`（VDB campaign 默认 1×）。
1× 相对 2× 的 QPS 提升约 35–60%，召回约 −0.9pp（Cohere）。
两种模式的实测对照见 `docs/benchmarks/houyidb-performance-2026-10.md`。

HNSW 合图默认「读锁规划 + 短写锁提交」：构图期间检索可并发读图，
streaming 50% 阶段串行查询总耗时从 104s 降到 3.9s、p99 从 1.72s 降到 21ms，
成熟 recall 不变。`HOUYI_HNSW_PLAN_APPLY=0` 可回退旧路径。详见
`docs/benchmarks/houyidb-performance-2026-10.md` 的 streaming 一节。

## 当前能力

- `houyi-core`: 64 字节对齐页分配、PAX 页抽象、无锁消息总线
- `houyi-vector`: HNSW 双速写入（HotBuffer + Async-Builder）、SIMD 距离计算（NEON/AVX feature 隔离）、Clustered/PQ/HII-L0 多后端
- `houyi-query`: pg_wire 协议（text/binary）、SQL 混合检索、谓词下推 bitset 剪枝
- `houyi-persistence`: WAL（异步写线程 + checkpoint/rotate）、磁盘快照、冷启动恢复
- `houyi-server`: HTTP API（collection upsert 带 payload、hybrid search 带 scores）、pg_wire 连接处理、指标输出
- `houyi-mobile`: C ABI 轻量索引（iOS/Android），见 `docs/HOUYIDB_MOBILE_MANUAL.md`

## 客户端 SDK（`sdk/`）

- **Python**：`sdk/python/houyidb` + LangChain / LlamaIndex adapter
- **TypeScript**：`sdk/typescript/houyidb`（bigint 处理 >2^53 id）
- **Go**：`sdk/go/houyidb`
- **Rust**：`sdk/rust/houyidb-client`
- **.NET**：`sdk/dotnet`（net8.0 + net10.0）
- **Java**：`sdk/java/houyidb`（17+）

六语言 `stable_id`（BLAKE2b-8 大端 & 2^63-1）与服务端网关互相对拍：同一字符串 id 在任何语言得到同一 u64，删除可跨 SDK 重放。

**MCP**：`mcp/houyidb-mcp` 提供 stdio MCP server（collection 列表、检索、混合检索、可选写入），可接入 Claude Code、Cursor、Cline 等客户端。

## 文档

- 试用手册：[`docs/USER_MANUAL.md`](docs/USER_MANUAL.md)
- 服务端手册：[`docs/HOUYIDB_SERVER_MANUAL.md`](docs/HOUYIDB_SERVER_MANUAL.md)
- 部署指南：[`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md)
- 向量化（OpenAI 兼容网关）：[`docs/EMBEDDING_GUIDE.md`](docs/EMBEDDING_GUIDE.md)
- VectorDBBench 对标协议：[`VDBBENCH_PROTOCOL.md`](VDBBENCH_PROTOCOL.md)
- 性能实测报告：[`docs/benchmarks/houyidb-performance-2026-10.md`](docs/benchmarks/houyidb-performance-2026-10.md)
- 移动端（C ABI）：[`docs/HOUYIDB_MOBILE_MANUAL.md`](docs/HOUYIDB_MOBILE_MANUAL.md)
- MCP 服务：[`mcp/houyidb-mcp/README.md`](mcp/houyidb-mcp/README.md)
- SDK 发布手册：[`docs/SDK_RELEASE.md`](docs/SDK_RELEASE.md)

## License & Governance

- 社区版：Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)；
  许可说明见 [LICENSING.md](LICENSING.md)。
- 贡献：[CONTRIBUTING.md](CONTRIBUTING.md)（外部贡献需 [CLA](CLA.md)）；
  安全漏洞请走私密渠道 [SECURITY.md](SECURITY.md)；商标政策
  [TRADEMARK.md](TRADEMARK.md)。
