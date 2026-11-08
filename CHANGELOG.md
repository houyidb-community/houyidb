# Changelog

> [English](CHANGELOG.en.md)

## Unreleased

### Fixed
- **payload 过滤只看每个 id 的最后一行**（公共 issue #6）：重复 upsert 同一 id
  后，旧 payload 不再参与过滤。修复前 append-only 页里的旧行仍会命中——先写
  `{"color":"red"}` 再改写为 `{"color":"blue"}` 后，过滤 `color=red` 会返回该
  id 却回读出 blue payload，重启后同样复现。过滤、按条件删除与 payload 回读
  现在统一为 last-wins 口径；压实只回收物理行，不改变查询结果。

### Changed
- **英文文档导航修正**：英文文档中的互链统一指向 `.en.md` 目标，
  `SECURITY(.en).md` 的安全验证入口改指部署指南与告警 Runbooks（原
  `docs/ENTERPRISE_READINESS.md` 不在公共快照内）；修正失效的相对引用。
  （信用 @Aqiu16717，公共 PR #4）

## v0.2.0-beta.8 (2026-11-07)

### Added
- **`DELETE /v1/admin/collections/{name}`**：无条件删除整个 collection
  （管理员 `Action::Admin`，审计自动记录）。删除走 WAL-first：先追加
  `collection_drop` 记录再移除内存对象，崩溃重放与 follower 实时 apply 都会
  收敛到「已删除」；同名重建得到全新的空 collection，旧向量不会复活。
  响应 `{"deleted":true,"collection":"<name>"}`；不存在/重复删除 404，
  并发删除竞争 409，follower/只读 503。`/v1/version` 增加
  `collection_delete` 能力标记。
- **六语言 SDK + MCP**：Rust/Go/Python/TypeScript/.NET/Java 客户端新增可选
  admin token（仅附加在 `/v1/admin/*` 请求）与 `delete_collection` /
  `DeleteCollection` API；`houyidb-mcp` 新增 `delete_collection` 工具
  （仅在读写模式注册，要求 `HOUYI_ADMIN_TOKEN`，仍受白名单约束）。
- **`PUT /v1/collections/{name}` 显式建集合**（普通写权限
  `Action::Write`）：一次定型 `dims` / `metric` / `index_type` / HNSW 三元组，
  参数在落 WAL 前统一校验；同名已存在 409，度量与后端创建后不可变。
  `/v1/version` 增加 `collection_create` / `collection_metric` 能力标记。
- **collection 级距离度量**：`l2` / `cosine` / `ip` / `l1`（内核统一
  「越小越近」，HTTP 分数在唯一出口换算：cosine `1-d` 钳制 [-1,1]，
  l2/l1 `1-d` 不截断，ip 为原始点积）。支持矩阵：hnsw / flat / clustered /
  flat-clustered 支持全部四种；pq-flat 仅 l2；hii-l0 仅 cosine。集合/混合
  检索响应新增 `metric` 字段（benchmark 路径保持字节不变）。
- **六语言 SDK + MCP 建集合接口**：`create_collection` / `createCollection` /
  `CreateCollection` / `CreateCollectionAsync`，支持 metric/后端/HNSW 选项并
  回显服务端实际生效参数；MCP 新增 `create_collection` 工具，走普通
  `HOUYI_AUTH_TOKEN`，绝不使用 admin token。

### Changed
- **复制协议 v8**：在 v7（`collection_drop`）之上补齐 `collection_create` 的
  度量/HNSW 语义（线格式不变）。版本门 fail-fast 隔离 v8 与旧版本混跑——旧
  二进制会静默跳过未知 op 并按 cosine 重建创建记录（已删集合复活 + 度量静默
  漂移），升级需全集群协调。
- **collection 快照 v2**：携带 `metric` 与可选 HNSW 参数；v1 解码按后端默认
  度量。图快照带度量，不匹配拒绝加载；`collection_create` WAL 记录同样携带
  度量/HNSW，冷重放不会把历史集合重置为 cosine。
- **快照/保留**：删除中的 collection 不再中止本轮快照（跳过即可）；全部删除
  后写空 manifest，`safe_checkpoint_lsn = global_barrier_lsn` 以允许 WAL
  截断；barrier 之后新建 collection 的 Create LSN 严格更大，截断安全。
  follower 不写空名单 manifest（append 先于 apply，barrier 可能越过未应用
  记录），保留旧 manifest 直到有新 collection 落地或被 promote。

### Upgrade
- 复制协议 **v8 不与 v7 及更早版本（beta.7 = v6）滚动兼容**：先停复制，
  全量升级所有副本，再恢复。
- **禁止降级**：v8 WAL 不得交给 v7 二进制冷重放——旧二进制会跳过
  `collection_drop` 并忽略创建记录的度量字段，把已删 collection 复活且按
  cosine 静默漂移。回退路径是备份或删除后重建（重建需重新指定度量）。

## v0.2.0-beta.7 (2026-11-02)

### Added
- `GET /v1/collections/{name}/export`：流式 NDJSON 逻辑导出（header + 按 id
  升序的记录 + footer），向量共享 `Arc` 不复制主表；payload 取最后写入值并
  跳过墓碑。`/v1/version` 增加 `collection_export` 能力标记。
- collection 列表/状态暴露只读 `ha_level`（变更仍走
  `POST /v1/collections/{name}/ha_level`，WAL-first）。

### Fixed
- 并发构建回归用例改为自适应等待队列排空，消除高负载 Linux CI 的偶发红灯。

## v0.2.0-beta.6 (2026-10-30) — P5.0 存储生命周期

### Added
- **collections 快照 + manifest**：HOUYISNAP/HOUYIMAN 格式，冷启动按
  checkpoint 跳过已覆盖记录；manifest 缺失且历史已截断时启动失败（禁止带洞重放）。
- **WAL 分段与有界保留**：默认 16 段 × 64 MiB，整段删除；`/v1/wal/status`、
  `/metrics`、`POST /v1/admin/collections/compact` 暴露保留状态与手动快照/轮转。
- **复制协议 v6**：握手携带本地最早可读 LSN / manifest checkpoint / 稳定
  follower_id；ANCHOR/HNDA 锚点协商；`BEHIND_RETAINED` + 全量重同步 runbook；
  保留 floor 跟随活跃 follower 的 durable LSN（断连 grace 可配）。
- **标量页内存压实**：默认每 10 分钟检查、stale ≥50% 时压实；status 暴露
  physical/live/stale/tombstone；`POST /v1/collections/{name}/compact` 手动触发。
- promote 前按 collection `applied_lsn` 补齐本地 WAL 尾部，失败拒绝 promote。
- **Sentinel 控制面原语（EE 前置，非付费墙）**：`cluster_state.json` 持久化
  term（只增、损坏 fail-closed）；leader 租约与写 fencing（
  `--require-leader-lease`，过期 503 `lease_expired` / pgwire 25006）；
  `POST/DELETE/GET /v1/admin/lease`、`POST /v1/admin/demote`、
  `GET /v1/admin/replication/status`、`GET /v1/version`、
  `GET /v1/admin/whoami`、`GET /v1/admin/diagnostics`；
  promote 可选 `{term, ttl_ms}` 原子授租；demote 可选 `follower_of`
  （新主复制地址）在运行时把旧 leader 转回 follower 并自动追平，无需重启。
  默认全关时行为与手动 promote 一致。
- **复制 quorum 等待**：`--replication-sync-quorum N`（默认 0 = 任一 ACK）；
  N>0 时 sync 写等待 N 个活跃 follower 的 DURABLE ≥ 写入 LSN（K-th 大）。
- **查询日志与统计**：`--query-log-capacity` / `--query-log-sample-every`
  （默认关闭）；有界 ring buffer 只记元数据（不记向量/payload/SQL）；
  `GET /v1/admin/query-log`、`GET /v1/admin/query-stats`（p50/p95/p99）。
- **sentinel RBAC 角色**：专用 `sentinel_token`（ENV/TOML，不上 CLI）
  仅可访问 lease/demote/promote/replication-status；admin 隐含。
- fuzz 目标 `collection_snapshot_decode`。

### Changed
- **行为变更（过滤）**：压实后过滤只命中每个 id 的最后一行；此前 append-only
  页会让旧行也命中。
- `drop_old`/基准客户端：启动自检探针改用保留 ID 并清理，`/v1/index/status`
  报告生效的 `distance_metric` / `hnsw_m` / `hnsw_ef_construction`。

### Upgrade
- 复制协议 **v6 不与 v5 滚动兼容**：所有副本一起升级。
- 首次截断 WAL 后不可降级到旧二进制；回退路径是备份或全量重同步。
- 启动前建议先做一次 `POST /v1/admin/collections/compact`，让旧历史被快照覆盖。

## v0.2.0-beta.5 (2026-10-25)

### Added
- **向量化接入（路线 A）**：`docs/EMBEDDING_GUIDE.md`（Ollama / OpenAI /
  TEI / mock 配方）、`examples/text_ingest.py`（切块→自动入库→检索）、
  `scripts/embed_smoke.sh`（链路自检）、`scripts/mock_embedding_server.py`
  （无语义 mock 上游）。试用手册新增 §2.6；发布包原生包含 docs/examples/scripts。
- 试用手册支持 Apple Silicon / Intel 双平台预编译包与向量化章节。

### Changed
- `async-trait 0.1.89 → 0.1.92`：Rust 1.99 新增 `clippy::double_must_use`，
  旧版宏展开触发 `-D warnings`；升级后 1.99 clippy 全绿。
- **CI/发布修复**：5 个失效的 action 固定 SHA 替换为有效 v-tag commit；
  Release 的 `macos-x86_64` 改为在 `macos-latest` 交叉编译（`macos-13` 退役）；
  CI Docker build 与 Release 流水线恢复。

### Fixed
- 向量化示例/脚本修正：忽略系统代理、非 UTF-8/缺失文件友好报错、
  chunk/overlap/batch/k 参数校验、smoke 响应解析守卫。
- 依赖安全：`cmov 0.5.4`、`vitest ≥4.1.11`（TS SDK 开发依赖）、
  `tokio` workspace 下限 `1.54.1`。

## v0.2.0-beta.4 (2026-10-25)

### Security
- **依赖加固**：`tokio 1.52.0 → 1.54.1`（RUSTSEC-2026-0386 `AsyncFd` I/O
  safety，unsound 警告）；`ctutils 0.4.2 → 0.4.5`（RUSTSEC-2026-0385
  非常量时间掩码；链路为 dev 依赖
  `tokio-postgres → postgres-protocol → digest → ctutils`）。
  `cargo audit` 复核：0 漏洞 / 0 警告；`cargo test --all` 660 项全绿。

## v0.2.0-beta.3 (2026-10-25)

### Added
- **标签过滤连续向量缓存**：命中向量按标签惰性拷贝为连续 `Vec<f32>`（Cosine
  预先单位化），过滤查询退化为顺序 SIMD 扫描 + top-k，写后失效。Cohere 1M
  1% 过滤实测 ef128 **1365 QPS**、ef256 **1386 QPS**，recall 1.0。
- **HNSW 实验旋钮**（默认关闭）：`HOUYI_HNSW_KEEP_PRUNED=1`（keepPruned
  connections，成熟召回 .996→.9976）、`HOUYI_HNSW_RERANK_EXPAND=N`（查询侧
  邻居扩展）、`HOUYI_HNSW_COMMIT_CHUNK=N`（串行合图短锁提交）。
- `/metrics` 构图日志新增 `graph`（图内节点数），用于诊断 main 与图的覆盖差。

### Changed
- **HNSW 合图默认改为「读锁规划 + 短写锁提交」**：`run_async_builder_once`
  的 ef_construction 规划移出全局写锁（读锁内并行完成），仅连边提交持短写锁
  并按 `parallel_chunk_size × parallel_plan_group_chunks` 分组。Streaming
  实测 50% 阶段 1000 条串行查询 104s → **3.9s**、p99 1.72s → **21ms**；
  80/100% 检查点 206s/211s → **164s/201s**；成熟 recall 不变（1× .9876、
  2× .996），写后 QPS 1× **1355/1353**。提交窗口用独立 gate 与 `remove`
  互斥（不占墓碑锁，避免批量写入排队）；`HOUYI_HNSW_PLAN_APPLY=0` 可回退
  旧路径。同日 Cohere 1M 批量加载总墙钟 +11%（1055s → 1170s，recall/QPS
  同档；对加载时间敏感时用回退开关）。详见
  `docs/benchmarks/houyidb-performance-2026-10.md`。
- **HNSW 默认连接密度保持 2×M**：1× 会把旧 2× 图快照判为不兼容并回退重放；
  若 WAL 已被 checkpoint 截断，存在数据丢失窗口。因此代码默认仍为 2×（兼容
  旧快照），高 QPS 场景显式设 `HOUYI_HNSW_LAYER0_MULT=1`（VDB campaign
  默认 1×）。同机实测 1× 的 OpenAI/Cohere QPS 提升约 35–60%，Cohere
  recall 0.9971 → 0.9883。

## v0.2.0-beta.2 (2026-10-21)

### Added
- **正确性验证（P4.5）**: proptest 属性化、cargo-fuzz、shuttle/loom、检索 oracle、
  crash chaos、turmoil 确定性网络仿真、Stateright 模型检查、Jepsen 式线性化、soak 哨兵。
- **复制硬化（P4.6）**: 协议 v5（日志前缀指纹 + 认证 token）、promote/关停互斥、
  HaLevel 枚举、FileWal 毒化、apply 原地重试、DNS/握手独立超时、SDK 集成 hermetic 化。
- **审计日志（P4.7.0）**: SHA-256 哈希链、启动校验、写失败 poison、审计校验端点。
- **RBAC（P4.7.1）**: 配置式主体 admin/writer/reader + collection 作用域。
- **租户配额（P4.7.2）**: 每主体 collection 数/在途/写速率限制（HTTP 429、pgwire 54000）。
- **静态加密（P4.7.3）**: AES-256-GCM；WAL v3、图快照 v2、旧 JSON 快照加密；
  本地 keyring + KeyProvider、密钥轮换、复制跨密钥指纹一致。
- **供应链与运维（P4.8）**: CI cargo audit/deny + CycloneDX SBOM；告警 Runbook 与
  Prometheus 规则；SLO/错误预算；RTO/MTTR 实测脚本；Helm chart
  （StatefulSet/PVC/Secret）；psql/psycopg2/pgbench 外部客户端一致性；
  移除未维护的 hwloc 原生依赖（NUMA 改读 sysfs）。
- **pgwire 兼容**: 接受标准客户端的 BEGIN/COMMIT/ROLLBACK/SET 空操作
  （libpq 系驱动隐式发送），修复 psycopg2 无法连接的问题。

### Changed
- **pgwire 授权**：Simple Query 与 Extended Execute 共用拒绝链；配额拒绝
  （54000）与越权拒绝（42501）指标分账。
- **依赖升级**：ureq 3 / rustls-pki-types（移除 rustls 0.21 与未维护的
  rustls-pemfile）；anyhow、crossbeam-epoch、tokio-postgres、rustls-webpki 安全升级。
- **TLS PEM 解析**迁移到 rustls-pki-types；Docker/CI 不再需要 libhwloc。

### Fixed
- 复制分歧（同 LSN 不同内容）不再被幂等跳过错失；缺密钥启动 fail-closed。
- pgwire SELECT 隐式建集合绕过配额；DELETE 未知 collection 误报 429。
- 代码审查批次三十二至三十五共 24 项（含 3 HIGH）。

## v0.2.0-beta (2026-05-13)

### Added
- **多 Collection 支持**: `CollectionManager` 管理独立维度/索引的 Collection，HTTP + pgwire 双协议路由
- **四种索引后端**: `--index-type hnsw|clustered|flat-clustered|flat`
- **ClusteredHnsw**: K-Means 分区 + 并行子图构建，加载快 73%（50K）/ 55%（1M）
- **FlatClustered**: K-Means 分区 + SIMD 暴力检索，QPS 231（50K 最高）
- **自适应 nprobe**: 查询时动态计算 (k × 2500 / cluster_size)，小 k 不浪费算力
- **Product Quantization**: 全局共享码本 + SDC 查表检索，128D 压缩 32x（16B/向量）
- **PqFlatClustered + mmap 重排**: 内存 15MB + 磁盘 488MB，SIFT1M recall 0.937, 延迟 3.9ms
- **houyi-mobile**: 独立 C ABI 库，支持 iOS/Android 原生集成
- **houyi-mobile MobileDB**: 多 Collection 管理，与服务器 API 一脉相承
- **SIFT1M 对标 MicroNN**: 延迟 3.9ms, recall 0.937, RAM 15MB
- **RegionPool 无锁化**: CAS 替代 Mutex
- **NEON SIMD 补全**: 4 累加器展开，Cosine/L2/点积全路径
- **统一配置系统**: houyi-core/config.rs 集中管理

### Changed
- **IndexBackend 枚举**: 抽象 HNSW/ClusteredHnsw/FlatClustered/Flat 四种后端
- **main.rs 模块拆分**: 6339 → 3040 行（types.rs / tests.rs 独立）
- **向量数据共享 Arc**: main_index 与 HnswGraph 同一份数据，消除三重存储
- **PAX 页清理**: 移除装饰性 ptr/layout，向量列改为列式连续存储
- **optimize() 实现**: HnswGraph.optimize_compact() 启发式邻居重选

### Fixed
- 配置散落：41 处 env::var 集中到 21 处
- houyi-core 零 Mutex（CAS 替代）
- SQL 解析器升级 (sqlparser 0.62 AST)

### Known Issues
- 15 个 pgwire 字节级测试因 IndexBackend 重构出现 1 字节偏差（不影响功能）
- ClusterGraph 自动分裂未完成（占位代码）
- PQ 未接入 --index-type
- simd-avx512 仅骨架

---

## v0.1.0-alpha (2026-04-24)

### Added
- 超融合单体内存向量数据库骨架
- HNSW 图引擎 + 双速写入路径
- pgwire + HTTP 双协议
- WAL 持久化 + 冷启动恢复
- PAX 页抽象 + Region 池
