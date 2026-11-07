# HouyiDB 部署指南

> [English](DEPLOYMENT_GUIDE.en.md)

> 适用版本：v0.2.0-beta.8。HouyiDB 是**单机、内存优先**的向量数据库；
> 高可用为 leader/follower 手动 promote，不做自动选主。beta 阶段建议先用于
> 功能验证与非关键业务。
>
> EN: deployment guide for the single-node, memory-first vector database;
> HA is manual leader/follower failover (no automatic election).

## 1. 部署形态

| 形态 | 适用场景 | 入口 |
|------|----------|------|
| Docker Compose | 快速启动、单机试用 | `docker-compose.yml` |
| 预编译二进制 + systemd | Linux 单机部署 | `docs/USER_MANUAL.md` 的发布包 |
| 源码构建 | 自定义 feature / 开发调试 | `cargo build --release -p houyi-server` |
| Kubernetes Helm | 集群内单机或主备 | `deploy/helm/houyidb/` |

## 2. 端口与数据目录

| 用途 | 默认地址 | 覆盖方式 |
|------|----------|----------|
| HTTP API | `127.0.0.1:7000` | `--benchmark-http-addr` / `HOUYI_SERVER_BENCHMARK_HTTP_ADDR` |
| pgwire | `127.0.0.1:5432` | `--pgwire-addr` / `HOUYI_SERVER_PGWIRE_ADDR` |
| 复制流（仅 leader） | `0.0.0.0:9876` | `--replication-listen-addr` / `HOUYI_SERVER_REPLICATION_LISTEN_ADDR` |

数据目录由 `--wal-path` 决定，同目录下会派生：

```
<wal-path>                     # 向量 WAL
<stem>.collections.wal         # collection 元数据/向量/payload 的 WAL
<stem>.applied_lsn             # 已应用 LSN 旁路文件
<stem>.wal.next_lsn            # 下一 LSN 旁路文件
snapshots/                     # 各 collection 的 HNSW 图快照
```

**生产必须显式设置 `--wal-path`**：默认值落在系统临时目录，重启或清理
`/tmp` 会丢数据。该目录需对服务用户可写，并纳入备份。

> `*.collections.wal` 当前只增不减（无压实）。长期高频写入会持续增长，
> 容量规划时按写入量预留，并定期做冷启动恢复演练。
>
> EN: production must set an explicit `--wal-path`; the default lives under the
> system temp dir. `collections.wal` currently grows monotonically.

## 3. Docker Compose（推荐起步）

```bash
docker compose up -d --build
curl -fsS http://127.0.0.1:7000/v1/health
```

`docker-compose.yml` 的关键约定：

- 数据卷 `houyidb-data` 挂到容器 `/data`，镜像内
  `HOUYI_SERVER_WAL_PATH=/data/houyidb.wal`；
- HTTP 7000、pgwire 5432 映射到宿主；
- 可选 OpenAI 兼容网关：设置 `HOUYI_OPENAI_UPSTREAM`（及
  `HOUYI_OPENAI_API_KEY`）后 `/v1/embeddings` 可用。

端口冲突时只改宿主侧映射（容器内保持 7000/5432）：

```bash
docker run -d --name houyidb \
  -p 17000:7000 -p 15432:5432 \
  -v houyidb-data:/data houyidb:latest
```

macOS 的 AirPlay 接收器可能占用 5000/7000；优先换宿主端口。

生产环境通过环境变量或 TOML 开启认证（不要写进镜像）：

```yaml
environment:
  HOUYI_AUTH_TOKEN: "${HOUYI_AUTH_TOKEN}"
  HOUYI_SERVER_WAL_WRITE_MODE: "async"
```

## 4. 二进制 + systemd

```bash
# 1) 准备用户与目录
sudo useradd --system --home /var/lib/houyidb --shell /usr/sbin/nologin houyidb
sudo install -d -o houyidb -g houyidb -m 750 /var/lib/houyidb /etc/houyidb
sudo install -o houyidb -g houyidb -m 750 houyi-server /usr/local/bin/houyi-server

# 2) 环境文件（权限 600，含 token 时尤其重要）
sudo install -o root -g houyidb -m 640 /dev/null /etc/houyidb/houyidb.env
```

`/etc/houyidb/houyidb.env`：

```bash
HOUYI_SERVER_BENCHMARK_HTTP_ADDR=0.0.0.0:7000
HOUYI_SERVER_PGWIRE_ADDR=0.0.0.0:5432
HOUYI_SERVER_WAL_PATH=/var/lib/houyidb/houyidb.wal
HOUYI_SERVER_WAL_WRITE_MODE=async
HOUYI_AUTH_TOKEN=<长随机串>
# 可选用 TOML 承载 RBAC/TLS/加密等完整配置：
# HOUYI_CONFIG=/etc/houyidb/config.toml
```

`/etc/systemd/system/houyidb.service`：

```ini
[Unit]
Description=HouyiDB server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=houyidb
Group=houyidb
EnvironmentFile=/etc/houyidb/houyidb.env
ExecStart=/usr/local/bin/houyi-server
Restart=on-failure
RestartSec=2
LimitNOFILE=65536
# SIGTERM 触发优雅关停（含关停图快照）；留足时间避免被强杀
KillSignal=SIGTERM
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now houyidb
curl -fsS http://127.0.0.1:7000/v1/health
```

## 5. 源码构建

需要 stable Rust（建议 1.87+）：

```bash
cargo build --release -p houyi-server
./target/release/houyi-server --help
```

Docker 构建默认走跨 CPU 通用路径；需要 AVX2 时：

```bash
docker build --build-arg CARGO_FEATURES=simd-avx2 -t houyidb:avx2 .
```

## 6. Kubernetes（Helm）

单机：

```bash
helm install houyi ./deploy/helm/houyidb -n houyidb --create-namespace \
  --set persistence.size=100Gi
```

默认镜像为 `ghcr.io/houyidb-community/houyidb:v0.2.0-beta.8`；自建镜像时
覆盖 `image.repository` / `image.tag`。

主备是**两个 release**（v1 无自动选主，`replicaCount` 调大只会得到多个独立
单机）：

```bash
helm install houyi-leader ./deploy/helm/houyidb -n houyidb \
  --set role=leader --set replication.authTokenSecret=houyi-replication

helm install houyi-follower ./deploy/helm/houyidb -n houyidb \
  --set role=follower \
  --set replication.leaderAddr=houyi-leader-houyidb.houyidb.svc:9876 \
  --set replication.authTokenSecret=houyi-replication
```

Secret 约定、TLS/加密/RBAC 的挂载方式见
`deploy/helm/houyidb/README.md`；完整 `config.toml` 走 Secret，不要用
`--set` 传凭据（会留在 Helm release 与 shell 历史里）。

## 7. 配置优先级与 TOML

优先级：**CLI 旗标 > 环境变量 > TOML > 默认值**。配置文件由 `--config`
或 `HOUYI_CONFIG` 指定；未知键、枚举笔误、空 token 都会启动失败
（fail-fast / fail-closed）。

```toml
# /etc/houyidb/config.toml
benchmark_http_addr = "0.0.0.0:7000"
pgwire_addr = "0.0.0.0:5432"
wal_path = "/var/lib/houyidb/houyidb.wal"
wal_write_mode = "async"          # async | sync（复制要求 async）
vector_dim = 1536
index_type = "hnsw"               # hnsw|clustered|flat-clustered|flat|pq-flat|hii-l0
snapshot_interval_secs = 600

# TLS（HTTP 与 pgwire 共用同一对证书）
tls_cert_path = "/etc/houyidb/tls.crt"
tls_key_path  = "/etc/houyidb/tls.key"

# 静态加密（AES-256-GCM；keyring 每行 `id=<64位hex>`，可加 `active=<id>`）
encryption_key_file = "/etc/houyidb/keyring"

# 审计日志（SHA-256 哈希链；启动时校验，链断拒绝启动）
audit_log_path = "/var/lib/houyidb/audit.log"
audit_include_reads = false

# RBAC：role = admin | writer | reader；token 与 token_sha256 二选一
[[auth_principals]]
name = "app-writer"
role = "writer"
token_sha256 = "<sha256-hex>"
collections = ["docs", "images"]
max_collections = 8
max_inflight = 64
writes_per_sec = 1000
```

- `auth_token` / `HOUYI_AUTH_TOKEN` 是遗留的**全权 admin** 通道；
  多主体请用 `[[auth_principals]]`。
- 未知凭据返回 401，已知主体越权返回 403；配额超限 HTTP 429、
  pgwire `SQLSTATE 54000`。
- pgwire 的密码即主体 token；仅在 TLS 或 localhost 场景使用。
- keyring 丢失将无法解密 WAL/快照；请与数据分开备份、权限 600。

## 8. 认证、TLS 与加密的最小生产组合

1. 反向代理终止 TLS，或直接给服务配 `tls_cert_path` / `tls_key_path`
   （HTTP 与 pgwire 同时生效）；
2. 启用 `HOUYI_AUTH_TOKEN` 或 `[[auth_principals]]`；公网不要裸跑无认证实例；
3. 需要静态加密时先创建 keyring 再首次启动（缺密钥会 fail-closed）；
4. 审计日志开启后可用
   `GET /v1/admin/audit/verify` 校验哈希链。

> 复制端口没有 TLS，只有共享 token 认证；必须用内网/防火墙隔离。
>
> EN: the replication port has token auth but no TLS — network-isolate it.

## 9. 备份与恢复

推荐以「停止写入后复制整个数据目录」为准：

```bash
sudo systemctl stop houyidb
sudo tar czf houyidb-backup-$(date +%F).tgz \
  -C /var/lib houyidb
sudo systemctl start houyidb
```

- 目录必须包含 WAL、`*.collections.wal` 及其 `*.wal.archived.*` 归档段、
  `*.wal.next_lsn`、`snapshots/`（含 `collections.manifest`）、follower 的
  `replication.follower_id` 和 keyring（如启用加密）；
- 运行中 WAL 与快照文件仍在变化，热拷贝**不能保证跨文件一致**；关键数据
  请用停写备份，或走主备（follower 追平后在其上做备份/promote）；
- 恢复：把备份解到原路径、权限归 `houyidb`，启动后自动重放 WAL 与图快照；
- 定期用副本做冷启动演练，验证备份可用，而不只是文件存在。

## 10. 主备与手动 failover

```bash
# leader
houyi-server --role leader \
  --replication-listen-addr 0.0.0.0:9876 \
  --wal-path /data/houyidb.wal

# follower（空目录启动即可，自动全量同步）
houyi-server --role follower \
  --replication-leader-addr 10.0.0.10:9876 \
  --wal-path /data/houyidb.wal
```

- leader/follower 必须配置相同的 `replication_auth_token`（ENV 或 TOML）；
- 复制要求 `wal_write_mode=async`；
- collection 级 HA：`async` / `semi_sync` / `full_sync`，通过
  `POST /v1/collections/{name}/ha_level` 设置；`full_sync` 的已确认写入
  满足 RPO=0；
- promotion 只能在 follower 上执行：

```bash
curl -fsS -X POST http://<follower>:7000/v1/admin/promote \
  -H "Authorization: Bearer $TOKEN"
# {"promoted":true,"next_lsn":...}
```

- promote 幂等：已是 leader 返回 409，`role=disabled` 返回 400；
- failover 后**旧 leader 必须停止**并清空数据目录，再以 follower 身份重连
  新主，避免脑裂：
  `rm -rf /data/* && houyi-server --role follower ...`；
- 复制范围仅 collection 路径；`/v1/vectors`（benchmark 主索引）不复制，
  failover 后新主没有该路径数据。生产数据请走 Collections API；
- v1 没有自动选主/quorum：监控复制滞后，人工判断后再 promote。

### 10.1 Sentinel 控制面原语（EE 前置，默认关闭）

自动选主本身在企业版，但 **term/租约/写 fencing 等安全原语在社区版**（不设
付费墙——没有它们，被分区的旧 leader 会继续写而丢数据）：

```bash
# 1. 开启写 fencing（leader 必须有有效租约才可写）
houyi-server --role follower --require-leader-lease \
  --replication-sync-quorum 1 ...     # quorum 可选，0 = 任一 ACK

# 2. sentinel 专用秘密只经 ENV/TOML 注入（不上 CLI），并需同时配置
#    HOUYI_AUTH_TOKEN 或 [[auth_principals]]，否则数据面无主体可用
export HOUYI_SERVER_SENTINEL_TOKEN=...

# 3. 授予租约（term 只增）+ 带租约 promote
curl -X POST http://<follower>:7000/v1/admin/lease \
  -H "Authorization: Bearer $SENTINEL_TOKEN" \
  -H "Content-Type: application/json" -d '{"term":7,"ttl_ms":10000}'
curl -X POST http://<follower>:7000/v1/admin/promote \
  -H "Authorization: Bearer $SENTINEL_TOKEN" \
  -H "Content-Type: application/json" -d '{"term":7,"ttl_ms":10000}'

# 4. 旧 leader 降级；带 follower_of 时运行时转回 follower（推荐）
curl -X POST http://<old-leader>:7000/v1/admin/demote \
  -H "Authorization: Bearer $SENTINEL_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"term":7,"follower_of":"<new-leader>:9876"}'
#    不带 follower_of 则保持只读 detached，需重启为 follower 才能重新加入。
```

- term 持久化在 WAL 目录的 `cluster_state.json`（原子写 + fsync）；
  文件损坏时**启动失败**，不带着未知 term 继续写；
- 租约过期不会自动翻角色，写路径返回 503 `lease_expired`（pgwire 25006）；
  sentinel 或运维显式 demote，避免与续租竞争；
- `GET /v1/admin/replication/status` 返回 role/term/lease/follower 进度/
  quorum；`GET /v1/version` 供客户端探测能力；
- demote 带 `follower_of`（新主复制地址）时节点运行时转回 follower 并自动追平；
  不带时是 `detached_readonly`，需以 follower 角色重启才能重新加入。

## 11. 可观测性

| 端点 | 用途 |
|------|------|
| `GET /v1/health` | 存活检查（认证豁免） |
| `GET /readyz` | 就绪检查（认证豁免） |
| `GET /metrics` | Prometheus 文本指标（认证豁免） |
| `GET /v1/index/status` | 主索引状态 |
| `GET /v1/collections/{name}/status` | 单个 collection 状态 |
| `GET /v1/pgwire/status` | pgwire 会话统计 |
| `GET /v1/admin/replication/status` | 角色/term/租约/follower 进度 |
| `GET /v1/admin/diagnostics` | 诊断包结构化状态（无密钥/向量） |
| `GET /v1/admin/query-log` | 最近查询元数据（需开启查询日志） |
| `GET /v1/admin/query-stats` | 按 collection 的 p50/p95/p99 |

- 告警规则：`deploy/prometheus/houyidb-alerts.yml`，每条告警带 runbook 链接
  （`docs/runbooks/`）；
- SLO 与错误预算：`docs/SLO.md`；
- 日志走 stdout/stderr（`tracing`），`RUST_LOG=info` 可调；
  `--slow-query-ms` 可打印慢查询。

## 12. 升级与回滚

1. 先备份数据目录（见 §9）；
2. 单机：停服 → 替换二进制/镜像 → 启动 → 验证 `/v1/health` 与抽样检索；
3. 主备：复制协议 **v6 不与 v5 滚动兼容**，不能“先升 follower”。先静默写入
   （或停写），**同时升级所有副本**，再按 leader → follower 顺序启动；
   升级前建议先 `POST /v1/admin/collections/compact {"rotate":true}` 提交完整
   快照，缩短首次启动的 WAL 重放。
4. 升级后首次启动会按需重放 WAL/图快照，耗时随数据量增长；
5. **不承诺降级兼容**：首次截断 collections WAL 后旧二进制无法安全打开新数据；
   回滚前先恢复升级前的备份，或用 `scripts/collections_resync.sh` 重建基线。

## 13. 故障排查

| 现象 | 处理 |
|------|------|
| `Address already in use` | 换端口；macOS 检查 AirPlay 占用的 5000/7000 |
| 启动失败提示缺少密钥 | keyring 未配置或路径错误；加密开启后不可无钥启动 |
| 审计链校验失败 | 日志被截断/篡改；恢复备份或走安全事件流程 |
| 复制 `AUTH REJECTED` | 两侧 `replication_auth_token` 不一致 |
| follower 只读 503 / pgwire 25006 | 正常：follower 不接受写；需先 promote |
| 复制滞后持续增长 | 检查网络与 follower 磁盘；见 `docs/runbooks/replication-lag.md` |
| 检索维度 400 | 查询向量长度必须等于 collection 首次写入的维度 |
| 过滤无结果 | 确认后端为 hnsw/flat，列名与操作符合法 |
| WAL 重放很久 | 数据量大或 `collections.wal` 长期未压实；评估快照与容量 |

更多试用/客户端问题见 `docs/USER_MANUAL.md`；协议细节见
`docs/HOUYIDB_SERVER_MANUAL.md`。
