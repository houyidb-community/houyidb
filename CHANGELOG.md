# Changelog

> [English](CHANGELOG.en.md)

## Unreleased

## v0.2.0-beta.9 (2026-11-21)

### Added
- **houyi-mobile JNI/AAR**：新增 `houyi-mobile-jni`（Kotlin
  `cc.houyidb.mobile.HouyiDB` 的 native 绑定，检索结果用 u64 id）与
  `houyi-mobile/android` Gradle library 工程；`scripts/build_mobile_android_aar.sh`
  构建 arm64-v8a AAR（约 404 KB，native 未压缩约 763 KB），
  `scripts/mobile_jni_smoke.sh` 在主机 JVM 验证 JNI 名称/编组/u64/save-load。
  同时提供 `include/houyi_mobile.h` C 头文件。
- **houyi-mobile 过滤下推**：标量行新增显式向量 id（`insert_scalars_v2` /
  Kotlin `insertScalarsWithId`），`FlatClustered::search_with_target_filtered`
  在候选扫描阶段跳过不允许的 id（不再先搜后过滤导致高选择性时不足 k）；
  `db_search_hybrid_u64`/Kotlin `searchHybrid` 返回 u64 结果。移动快照升级
  v3 记录标量行 id，v1/v2 仍按行号兼容。旧标量接口保留行号语义。
- **houyi-mobile i8 存储**：`FlatStorageDtype` 新增 i8（每维 1 B + 每向量
  4 B scale，仅 flat-clustered），`build_with_storage_dtype` 与 C ABI/Kotlin
  `dtype=2` 接入；f16/i8 共用余弦范数缓存。host 实测：真实 7.3k×512
  recall 0.996/文件 4.3 MB（f16 8.3 MB）；100k×192 recall 0.953/文件
  19.9 MB（f16 38 MB），查询延迟与 f16 持平或略优。服务端默认 f32 不变。
- **houyi-mobile 检索优化**：f16 余弦路径预计算候选/质心范数（每向量
  +4 B，100k 约 0.4 MB），查询热循环从「点积 + 两个范数」降到一次点积；
  新增可调候选预算 `houyi_mobile_{db_,}set_search_target` / Kotlin
  `setSearchBudget`（默认 2500/k）。真实 7.3k×512 人脸 host 实测：
  `per_k=500` 时 p50 ≈ 0.30 ms、recall 0.994（默认 2500 为 0.43 ms/1.000）；
  服务端默认 f32 路径不启用缓存，行为与性能不变。
- **houyi-mobile ABI v2**：新增批量写入 `insert_batch`/`db_insert_batch` 与
  u64 结果 `search_u64`/`db_search_u64`（旧 f32 id 在 >2^24 时丢精度的路径保留
  但不再推荐）；新增 `create_v2`/`db_collection_v2` 显式 metric（L2/Cosine/IP/L1）
  与 dtype（f32/f16，f16 限 flat-clustered）；同名不同配置拒绝复用。批量失败
  返回首个错误且不回滚已写入行。持久化格式 v2 记录 metric/dtype。
- **houyi-mobile f16 存储与 save/load**：`flat-clustered` 在移动端使用 f16
  分簇存储（`houyi-vector` 新增 `build_with_dtype`，默认 `build` 仍为 f32，
  服务端既有集合与快照行为不变）：100k×192 索引常驻约 37 MB（f32 约 74 MB），
  真实 7.3k×512 recall@10 保持 1.0、查询 0.47 ms；高噪声 100k 合成 recall
  0.955 与 f32 相同。新增 `houyi_mobile_db_save/load`：原子临时文件 + fsync +
  rename、magic/版本/长度/CRC32；flat-clustered 保存 f16 构建态，加载不重建；
  flat 存原始行，clustered-hnsw 加载时重建。100k×192 文件 37.6 MB、save
  0.10 s、load 0.064 s；真实 7.3k×512 文件 7.3 MB、0.023/0.012 s（host）。
- **houyi-mobile 增量写入与内存去重**：`flat-clustered` 建图后按最近质心
  增量插入（立即可查，同 id 覆盖为原位更新/跨簇搬移且计数不变），再次
  `build` 从当前索引导出重聚类；建图成功后释放独立写入缓冲，向量只由索引
  持有（100k×192 的索引常驻内存约为原来的一半）。`clustered-hnsw` 建图后
  允许追加新 id（立即可查）、覆盖已有 id 返回 -2，需重建索引后 upsert；`flat` 行为不变。
  测试：`flat-clustered` 增量插入/跨簇 upsert、C ABI 立即可查；真实 7.3k×512
  人脸数据 recall@10=1.0、构建 0.11 s、查询 0.44 ms。
- **实例可观测性**：`/metrics` 新增 per-collection gauge
  `houyidb_collection_vectors{collection="..."}`（集合名按 Prometheus
  规则转义，取 `indexed_count` 廉价计数）；`/v1/version` 新增
  `build_sha`，由 `houyi-server/build.rs` 在构建时嵌入 Git 短 SHA（源码
  包无 `.git` 时为 `unknown`；Docker 构建上下文没有 `.git`，经
  `HOUYIDB_GIT_SHA` build-arg/ENV 由 CI 显式传入，镜像不再恒为 `unknown`）。
  调用方可用它确认运行实例对应的提交。
- **`vector_dtype` f16 存储**：`PUT /v1/collections/{name}` 新增可选
  `vector_dtype`（`f32` 缺省 / `f16`；大小写不敏感，接受常用别名），仅
  `index_type=hnsw` 可用，创建后不可变。f16 把常驻向量内存减半；查询向量
  始终为 f32、距离按 f16 舍入后的数据计算（非对称量化）。缺省 f32 的行为
  与旧版本逐位一致。创建响应、列表摘要与 status 回显实际生效的
  `vector_dtype`；`/v1/version` 增加 `vector_dtype_f16` 能力标记。collection
  WAL 的 `collection_create` 与 collection/图快照（v2 flags bit0、manifest
  `vector_dtype`）均携带该编码，冷重启与快照恢复保持 dtype（f16 集合按快照
  头建 typed hnsw 后端，不会静默退化为 f32）。独立 HNSW 图快照采用不对称
  版本门：f32 仍写 version=1（旧二进制可继续读取），f16 写 version=2（旧
  二进制以 `UnsupportedVersion` 明确拒绝并回退全量 WAL 重放，绝不静默降为
  f32 后在下次快照固化）。同名集合改变 dtype 返回 409；非 f32 + 非 hnsw
  组合 400。
- **六语言 SDK + MCP f16 支持**：Rust/Go/Python/TypeScript/.NET/Java 的建
  集合接口新增可选 `vector_dtype`（响应回显，旧服务端缺省时保持未设置）；
  `houyidb-mcp` 的 `create_collection` 工具新增 `vector_dtype` 参数（本地
  拒绝 f16 + 显式非 hnsw 后端，与服务端同口径）。
- **`quantization` 图侧 int8 量化**：`PUT /v1/collections/{name}` 新增可选
  `quantization`（`none` 缺省 / `int8`），仅 `index_type=hnsw` 可用，创建后
  不可变，与 `vector_dtype` 相互独立。int8 只压缩 HNSW **图侧**副本（逐向量
  对称仿射，1 B/dim），主表 / hot buffer / 构建队列仍保留原始（f32/f16）向量；
  检索以 `k.max(ef)` × 2 取图候选后在原始向量上精确重排，最终 top-k 精度
  不降。余弦先归一化再量化，查询向量恒为 f32（非对称量化）；零范数带保持
  原值。内存不为非余弦集合带来净收益（主表必须保留，4→5 / 2→3 B/dim；余弦
  f32 8→5、f16 4→3 才有下降），收益在图内距离工作集与带宽。缺省 none 的
  行为与旧版本逐位一致。创建响应、列表摘要与 status 回显生效的
  `quantization`；`/v1/version` 增加 `quantization_int8` 能力标记；同名集合
  改变 quantization 返回 409；int8 + 非 hnsw 组合 400（显式 `"none"` 在
  任何后端合法）。
- **六语言 SDK + MCP int8 支持**：六个 SDK 的建集合接口新增可选
  `quantization`（未设置时请求体不出现该键；旧服务端不回显时保持未设置，
  不自行默认为 `none`）；`houyidb-mcp` 的 `create_collection` 工具新增
  `quantization` 参数（本地拒绝 int8 + 显式非 hnsw 后端）。
- **`quantization` 图侧 RaBitQ（1 bit/dim）**：`quantization` 取值域扩展为
  `none` / `int8` / `rabitq`。RaBitQ 仅 `index_type=hnsw` 且
  `metric=l2|cosine`（`ip` / `l1` 创建 400；WAL 重放与快照期望同样
  fail-fast），创建后不可变。内核：D = dims 向上取 2 的幂并补 0；确定性
  随机旋转 P = H_D ∘ diag(signs)（种子只依赖 D，跨进程/节点/重启逐位
  可复现）；符号位打包为 `ceil(D/64)` 个 u64 + `alpha=Σ|ō_i|` + `norm`。
  查询侧仅在遍历入口补 0 + 旋转一次（O(D log D)），非对称估计
  `est_cos = clamp(符号内积/alpha, -1, 1)`：余弦距离 `1−est`、L2 平方距离
  `max(0, q²+x²−2qx·est)`；存储×存储（构图启发式/剪枝）用汉明距离 +
  arcsine 律估计。零范数用哨兵 `{codes=0, alpha=1, norm=0}`。候选宽度 ×16
  （码字带宽与 int8 ×2 等价：`16·D/8 = 2·D` 字节）；
  与 int8 一样主表 / hot buffer / 构建队列保留原始向量，最终 top-k 精确
  重排。图快照 v4（`FLAG_VECTOR_RABITQ=0x08` bit3 必置、禁止 bit2）、
  collection payload flags bit2=`0x04`、manifest `quantization=2` 且
  `format_version=3`，旧二进制一律干净拒绝，禁止降级。内存不为非余弦集合
  带来净收益（主表不变，图侧每向量仅 ~136 B@768 维，比 int8 再降至
  ~1/6），收益在图内距离工作集与带宽。
- **六语言 SDK + MCP rabitq 支持**：六个 SDK 的建集合接口接受
  `quantization="rabitq"`（未设置时请求体不出现该键；旧服务端不回显时
  保持未设置，不自行默认为 `none`）；`houyidb-mcp` 的
  `create_collection` 工具枚举与本地校验同步扩展（int8 / rabitq +
  显式非 hnsw、rabitq + 显式 ip/l1 均在本地拒绝）。
- **稀疏向量 + BM25/IDF + dense+sparse 融合检索（P4）**：`PUT
  /v1/collections/{name}` 新增可选 `sparse`（`none` 缺省 / `bm25`），仅
  `hnsw` / `flat` 后端，创建后不可变，隐式创建恒为 `none`；同名改变
  sparse → 409，非法组合 → 400。稀疏词项由客户端给定（无分词器）：
  `(term_id, weight)`，term ∈ [1, 2^24-1]、weight 有限 >0、单文档 ≤1024
  个原始对、查询 ≤64 个，重复 term 求和。upsert 条目可选
  `"sparse": [[t,w],…]`（缺省 / null = 不改，`[]` = 清空；整文档替换
  语义，WAL 重放幂等），`sparse=none` 集合带 sparse 键 → 400。新端点
  `POST /v1/collections/{name}/fusion/search`（既有 `/hybrid/search`
  语义不变）：dense 与 sparse 必须同时非空，两路各取 4×k 后按 RRF
  （默认 `rrf_k=60`）或加权（`alpha` 默认 0.5、各路 min-max 归一）融合；
  `scores` **仅表达融合排序、不属于任何 metric**，`dense_scores` 按
  集合 metric 换算、`sparse_scores` 为 BM25 原始分，缺席模态为 null。
  BM25 `k1=1.2`、`b=0.75`；删除（按 id / 过滤器）同步移除稀疏词项；
  `status` 回显 sparse 统计，`/metrics` 新增 `houyidb_sparse_*` 与
  `houyidb_fusion_searches_total`，`/v1/version` 能力增加 `sparse_bm25`
  / `fusion_search`；`/export` 每行追加可选 sparse 字段。内存账目如实：
  每 (doc, term) posting ~24 B + 正向 12–16 B，100 万文档 × 8 term ≈
  330–400 MB（不得按更小单位成本规划）。
- **六语言 SDK + MCP 稀疏/融合支持**：Rust/Go/Python/TypeScript/.NET/Java
  的建集合接口新增可选 `sparse`（旧服务端响应缺省按可选容忍），upsert
  条目支持稀疏词项，新增 `fusion_search` / `fusionSearch` /
  `FusionSearchAsync` 方法与响应模型（`dense_scores` / `sparse_scores` /
  `fusion` 块均按可选字段容忍）；`houyidb-mcp` 的 `create_collection`
  新增 `sparse` 参数（本地拒绝 bm25 + 非 hnsw/flat），`upsert_vectors`
  支持 sparse，新增 `fusion_search` 工具。
- **pgwire 融合查询指令（P4）**：新增 `/* houyi:sparse=t:w,… */`、
  `/* houyi:fusion=rrf|weighted */`、`/* houyi:alpha=… */`；sparse 与
  vector 同时存在才进入融合检索。仅稀疏或仅融合参数 → 42601；缺
  collection 指令或集合 `sparse=none` → 0A000；缺集合 → 42P01；词项 /
  参数越界 → 22023。pgwire 不提供稀疏写入（AUB1 行不变）。
- **全后端 collection 快照压缩（P5）**：此前 clustered / flat-clustered /
  pq-flat 不支持快照，会令 collections WAL 保留停在旧 floor 并在
  `/v1/wal/status` 报 `unsupported_backend`。现在六种后端全部可快照：
  - payload/manifest 的 `vector_kind` 新增 3 = pq-flat 已建紧凑态（
    centroids/codebook/聚类 id+codes，恢复逐位保真、不重新训练）、
    4 = 懒后端缓冲原始行；加法扩展不 bump payload/manifest 版本，旧二进制
    在解码时对未知 kind 干净拒绝（`format_version` 规则不变）。
  - clustered / flat-clustered 快照 = 已建原始向量（边界复制按 id 去重）+
    待建行；pq-flat 已建 = 紧凑构建态 + 待建行；pq-flat 未建 = 待建行。
    恢复只装回缓冲、不自动构图（与无快照全量 WAL 重放冷启动一致）；
    `kind=4` 仅 clustered/flat-clustered/pq-flat，`kind=3` 仅 pq-flat，
    manifest 与 payload 双侧一致校验。
  - `export_snapshot_vectors` 对懒后端改为返回「已建原始向量 + 待建行」；
    pq-flat 构图后原始向量已丢弃，导出显式报错而非静默返回残缺数据。

### Changed
- **复制协议 v9**：在 v8 的 `collection_create` 度量/HNSW 语义之上新增
  `vector_dtype`（线格式不变）。v8 二进制解码时忽略该字段，follower 会把
  f16 集合建成 f32（静默分叉），因此 v8/v9 禁止混跑，升级须全集群协调；
  **禁止**用 v8 或更旧二进制重放 v9 WAL（f16 collection 会被重新解释为 f32）。
- **复制协议 v10**：在 v9 之上新增 `quantization`（线格式不变）。v9 二进制
  解码时忽略该字段，follower 会把 int8 集合建成未量化（静默分叉），因此
  v9/v10 禁止混跑，升级须全集群协调；**禁止**用 v9 或更旧二进制重放 v10
  WAL（int8 collection 会静默建为未量化图）。
- **复制协议 v11**：在 v10 之上把 `rabitq` 纳入 `quantization` 规范取值域
  （线格式不变）。v10 二进制把 rabitq 串判为非规范值并在重放时 fail-fast
  （不会静默降级），握手阶段即拒绝；v10/v11 禁止混跑，升级须全集群协调；
  **禁止**用 v10 或更旧二进制重放 v11 WAL。
- **复制协议 v12**：在 v11 之上新增稀疏字段（`collection_create.sparse`
  与 upsert 条目可选 `sparse`，线格式仍为加法字段）。v11 及更早二进制
  解码时**静默忽略**这些字段（尚无快照时丢失稀疏索引），因此 v12 与更早
  版本禁止混跑，升级须全集群协调；**禁止**用 v11 或更旧二进制重放 v12
  WAL。
- **快照版本门扩展（稀疏）**：collection 快照 payload v3（flags bit3=
  `0x08` 必置，v1/v2 必须不置；旧二进制只接受 1..=2，遇 v3 干净拒绝），
  稀疏段位置式追加在 tombstones 之后（term 升序、term 内 doc 升序，
  与墓碑交叉校验）；manifest 条目新增可选 `sparse`（1=bm25），含 bm25
  条目的 `format_version` 必须为 4（接受 1..=4，精确推导）。WAL 稀疏字段
  沿用加法可选字段策略（`COLLECTION_WAL_VERSION` 不变），仅有文档声明的
  静默降级风险（降级前须先完成快照）。
- **快照版本门扩展**：独立 HNSW 图快照新增 version=3（n 条记录携带
  `[scale][bias][dims×i8][dims×f32 source]`，`FLAG_VECTOR_INT8=0x04`）；
  v1/v2 二进制以 `UnsupportedVersion` 拒绝 v3 并回退全量 WAL 重放。collection
  快照 payload 版本仍为 2，`flags` 新增 bit1 `VECTOR_INT8=0x02`（不新增
  body 字节）；manifest entry 新增可选 `quantization`（仅 int8 写 1），且
  含 int8 条目的 manifest `format_version` 必须为 2（旧二进制只接受 1，显式
  拒绝而非静默降级）。RaBitQ 再新增图快照 version=4（记录携带
  `[id][norm][alpha][packed codes][dims×f32 source]`，
  `FLAG_VECTOR_RABITQ=0x08` 必置、禁止 bit2 的 int8 flag；旧二进制以
  `UnsupportedVersion` 拒绝），collection `flags` 新增 bit2
  `VECTOR_RABITQ=0x04`（与 bit1 互斥，不新增 body 字节），manifest
  `quantization` 新增值 2 且含 rabitq 条目的 `format_version` 必须为 3。
- **英文文档导航修正**：英文文档中的互链统一指向 `.en.md` 目标，
  `SECURITY(.en).md` 的安全验证入口改指部署指南与告警 Runbooks（原
  `docs/ENTERPRISE_READINESS.md` 不在公共快照内）；修正失效的相对引用。
  （信用 @Aqiu16717，公共 PR #4）

### Fixed
- **依赖安全整改（Dependabot 36 条，10 critical / 7 high / 18 medium /
  1 low）**：Go SDK `golang.org/x/crypto` 由 v0.1.0 升到 v0.52.0（`go`
  指令 1.17 → 1.25；覆盖 18 条 x/crypto 公告，含 8 条 critical）；Python
  SDK 依赖下限声明为 `requests>=2.34.2,<3`、`urllib3>=2.8.0,<3`（上界与
  requests 自身的 urllib3 约束对齐，不放宽消费者已有约束），并同步升级
  `uv.lock`；`requires-python` 由 ≥3.9 提到 ≥3.10（修复版本要求 ≥3.10，
  且 Python 3.9 已 EOL），两个 adapter 也同步 ≥3.10 + `houyidb>=0.2.0b8`。
  MCP 依赖 `hono` 4.13.18、`@hono/node-server` 2.1.8、`vitest` 4.1.12
  （连带修复 tinypool/@vitest/mocker critical/medium），`engines.node`
  为 ≥20（生产依赖 `@hono/node-server` 与 Vitest 4 的共同下限，
  `@types/node` 同步对齐 ^20，README 推荐 22 LTS），`npm audit` 0。
  验证：Go `go test ./...`、Python wire 契约 7 passed、MCP clean
  `npm ci` + build + 43 tests。另：本地 cargo audit 仍有 1 条
  `rustix` unsound 警告（非 Dependabot 条目，属 allowed warning）。
- **生产形态 legacy 构图的上层导航束不足（sm 实测残余孤岛）**：sm 在
  `c631596` 上用 100 行/批的 HTTP 写入（worker 阈值式排空）复现
  `clustered_1794652908806`：recall@10 在 ef=16/64/256/512 恒为 0.9656，
  q45/q62 返回错误簇（相似度 0.11–0.12，真值 0.95，与 ef 无关），而
  level-0 已 100% 可达、兜底补边为 0、冷重启与再次触发兜底均不改变。用
  mulberry32 逐位复刻其语料后定位：layer-1 有 3118 节点时，固定 64/128 的
  上层贪心束从入口点找不到部分簇的层内代表（layer-2 以上与 L0 均全连通，
  缺陷只在**层内束覆盖不足**），且 layer-1 另有 12 个可达性孤点。修复两处：
  (1) 上层束宽按层规模自适应：估计层节点数 `nodes/m^layer`，束宽取
  `max(HOUYI_HNSW_UPPER_EF, ceil(估计/8))`、自适应部分封顶 4096（env 由
  「固定值」改为「下限」，上限同为 4096；新增
  `HOUYI_HNSW_UPPER_EF_ADAPTIVE=0` 关闭自适应做延迟 A/B；构图侧仍保持贪心
  1）；(2) 连通性兜底从仅 L0 推广到每一层，逐层 BFS 并把断开分量接回最近可达
  锚点；每层节点数在单次全图扫描中预计算，连通正常时只遍历可达集合，避免
  在 drain 写锁内按层重复扫全图。验证：精确生产语料 legacy 均值
  recall@10 由 0.9656 升至 **0.9953**（唯一低于下限的是 q30 的 TIE_EPS
  并列近邻噪声），q45/q62 均回到真值簇；默认 plan/apply 保持 1.0000；官方
  三个 50k 用例全过。门禁新增第四个 50k 用例
  `legacy_prod_shape_batches_keep_all_layer_recall_50k`（精确复刻 sm 语料 +
  服务端 collection 的 `BuildConfig::default()` + ~300 行/tick 排空），断言
  每层可达率 100%、各 ef 均值 ≥0.99 且 q45/q62 recall ≥0.9。新增大图门禁
  `large_graph_upper_navigation_keeps_cluster_at_quarter_million`（250k×512 / 64 簇，
  layer-1 约 15.6k 节点）：ef=16…4096 的返回 id 必须留在查询自己的簇（禁止
  跨簇错落），ef=8192 均值召回 ≥0.99；实测低 ef 的 q8 召回可为 0 但始终在
  簇 8，ef=8192 恢复 1.0，说明大图瓶颈是 L0 束宽与簇规模，不是上层导航。
  自适应封顶相应由 1024 提到 4096：250k 实际宽度 1954（约 12.5%，未触顶）、
  1M 约 6.5%，避免覆盖率随 N 线性衰减；本机 50k×512、ef=256 的多次 A/B 实测
  固定 64 束宽约 0.46–0.55ms、自适应约 0.64–0.90ms（视机器负载 +0.18–0.35ms），可用
  `HOUYI_HNSW_UPPER_EF_ADAPTIVE=0` 回滚（`scripts/upper_beam_latency_ab.sh`）；
  nightly 增加 250k 导航 job。所有 50k/4653
  召回门禁同时修正 ef 扫描：`search_scored_with_ef_override` 只能抬高
  `ef_search`，此前 ef=16/64 实际静默复用推断默认束宽；现在每轮先经 setter
  真正钉到 16/64/256/512，并按每 tick 排空行数与 legacy 分支断言门禁没有
  测错构图路径。
- **串行/HTTP 小批次构图遗留整簇不可达分量**：串行插入与 HTTP 100 行/批
  累积 drain 在成簇语料上会因对称剪枝丢掉唯一桥边，留下成员度数都
  ≥`min_degree` 的不可达分量（分组补救只补度数 <`min_degree` 的孤点，
  覆盖不到这种「内部互连、对外无桥」的整簇形态）。修复：每次 drain 排空
  构图队列后从入口点做 L0 BFS，把每个不可达分量接到**向量距离最近的
  可达锚点**（优先仍有余量者，同距按 id 破平；新边被对端剪枝时最多重试
  4 轮），保证整图可达；分组路径与 plan/apply 路径同样生效。官方 50k
  门禁新增 HTTP 形态用例（100 行/批、worker 阈值式排空，覆盖 300 行串行
  与 600 行分组两种 drain），要求 level-0 可达率 **100%**。
- **HNSW 重写/删除节点孤立第三方邻居**：替换已有节点时 `remove_node` 只
  清理被删节点的回边，旧邻居可能失去最后一条边跌到 L0 度 0，成为入口点
  不可达孤岛（服务端后端模型差分门禁在 5 节点重写序列上实证：节点 8 被
  完全孤立、查询 0 命中）。修复：替换与删除路径对「本批 id + 受影响旧
  邻居」执行局部孤点补救（复用分组补救的 id 窗口规则，成本只随受影响
  邻居数增长，不做全图扫描；`HOUYI_HNSW_GROUP_REPAIR=off` 仍可关闭）。
  新增三层回归：服务端后端模型差分门禁（小 id 域强制碰撞的随机
  upsert/删除/查询流对拍 `HashMap` 模型，覆盖 flat 重复 id 与图孤岛两类
  事故）、内核 5 节点重写回归、HTTP 稳定 id 重索引 + 冷重启（hnsw/flat
  双后端）；recall oracle 增加 level-0 可达率与零度节点不变量；50k 大簇
  门禁接入 nightly CI。
- **flat 后端重复 id 计数与 top-k 重复**：flat 的写入缓冲只做 `extend`，
  同一 id 重复 upsert 会累积多行——`indexed_count` 虚高（upsert 1,2,1 后
  为 3）、检索时同一 id 占多个 top-k 名额（返回 `[1,1,2]`）且旧向量仍在
  参与排序。修复：flat 写入改为按 id 幂等的 upsert，批内后写覆盖先写、
  跨批替换同 id 旧行（与 HNSW 主图的按 id 更新语义一致）；WAL 重放与快照
  恢复共用该写入通道，重启/恢复同样去重。查询侧不加去重开销。存储为
  `rows` + id→行下标，并以 `RwLock` 保护：upsert O(batch) 原地替换、删除
  O(1)（`swap_remove` 只修正被移动行），多个只读扫描并行；`FlatStore`
  暴露只读访问器（`rows`/`len`/`is_empty`）。
- **50k×512 成簇语料整簇孤岛（两条构图路径）**：50k / 64 簇（每簇 ~781
  点）下自适应组宽仅 ~195，仍小于簇规模，plan/apply 与 legacy 串行两条
  路径都出现整簇不可达（失败查询返回完全错误的簇，相似度 0.03–0.10，真值
  0.95，加大 ef 几乎不改善）：默认 plan/apply 无补救时 level-0 前向可达仅
  359/50000（5003 个 L0 度为 0 的新点）。修复两处：(1) 上层下降束宽从逐层
  贪心 16 提到 64——串行路径 #38 在束宽 16 时仍落入错误簇（错误入口相似度
  0.881 vs 真簇 0.0495），束宽 64 时两条路径均 0/64 错落；(2) 组提交后默认
  执行孤点补救（`HOUYI_HNSW_GROUP_REPAIR=4,16`，设 `off` 关闭）：对 L0 度
  <4 的成员在同批 ±16 id 窗口内补双向边（仅 L0）。plan/apply 前向可达恢复
  到 49958/50000（99.92%；余下 42 个节点为成员度 ≥4 的整簇小孤岛，仍远超
  ≥95% 门禁阈值），串行路径不补救亦达 96.18%；两者在所有 ef 上均值召回
  1.0000、ef=512 单查询下限 0.9 全过。新增 release 门禁
  `./scripts/clustered_recall_50k_gate.sh`（两条 `#[ignore]` 用例；debug 下
  50k 暴力真值 + 构图需数分钟，故不进每次提交的 `cargo test --all`）。
  门禁与可调性：上层束宽改为可调 `HOUYI_HNSW_UPPER_EF`（默认 64，取值
  1..=1024，非法值告警后回落 64；构图侧有意保持 1——加大它只会放大
  ef_construction 成本并改变构图语义）。补救候选按「健康 > 低度」两档排序，
  采集时即排除已饱和的兄弟（避免触发对端剪枝回退刚加的边），加边决策时再复测
  一次以兜住本循环内补满的候选；窗口边界用饱和算术，候选缓冲预留量按批内实际
  候选数封顶（手误的巨大窗口不会在写锁内预留 GB 级内存）。`GROUP_REPAIR`
  提供了窗口令牌但无法解析（如 `4,abc`）或带多余字段（如 `4,16,32`）时整体
  告警回落缺省，不再静默半应用。门禁显式钉死 PLAN_APPLY / GROUP_REPAIR / UPPER_EF 三个
  进程级 OnceLock，并以新的 `plan_apply` 报告字段断言真实走到的构图分支（防止
  静默回退串行构图而召回数字仍然达标）；`clustered_recall_50k_gate.sh` 额外清空
  KEEP_PRUNED / RERANK_EXPAND、单线程运行并逐行校验两条 `ok`。
- **HNSW 分组构图在成簇小语料上召回崩塌**：plan/apply 与 legacy 分组路径按
  固定组宽（默认 `chunk × group_chunks` = 1024）规划，每组只对组前图快照
  取近邻、组内兄弟互不可见；成簇语料里同批兄弟互为最近邻，提交后被两侧
  对称剪枝剪成孤岛。表现为 recall@10 在 ef=16/64/256 上恒为 6–50%（与 ef
  无关），level-0 从入口只能触达约一半节点。修复：新增自适应组宽
  `HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR`（默认 256）与
  `HOUYI_HNSW_PLAN_GROUP_MIN`（默认 16），组宽取
  `min(configured, max(min, graph_len/divisor))`；`divisor=0` 可关闭自适应。
  自适应收窄的组仍走规划提交路径（legacy 串行子分支会重抽层级并丢弃本批
  预抽层级，实测仍只有 91.3% 均值）。新增默认与 legacy 两条回归门禁
  （4653×512 成簇语料：level-0 可达 ≥95%、ef 16/64/256/512 recall 均值
  ≥0.95；单查询最小值 ≥0.9 只在 ef=512 强制，该点比本题 512 维语料的推断
  默认 ef_search=256 更严，其余 ef 另要求低于下限的查询数 ≤2——64 簇全覆盖
  下实测 #49 recall 0.7、#55 recall 0.0 属批次边界桥接方差，ef=512 全部恢复；
  上限 2 使第 3 个破簇在任何 ef 都被计数断言拒绝，历史固定组宽缺陷仍被
  均值/计数/可达率断言当场抓住）。
- **`HOUYI_HNSW_SEED_SERIAL_POINTS=0` 开关失效**：`usize` 解析的正数过滤会把
  显式 0 回落成默认 256，使文档承诺的「0 = 关闭」从环境变量不可达。
  `env_usize` 现由 `env_usize_allow_zero` 派生，两者在缺失/非法语义上不会
  漂移。
- **一次性构图后端的状态不可见**：clustered / flat-clustered / pq-flat 在
  `built=true` 之后的新写入只进待建缓冲、永不入图，`pending_count` 还会归零，
  此前没有任何字段能表达「需重建」。`GET /v1/collections/{name}/status`
  新增可选 `lazy_build`（`built` / `buffered` / `rebuild_required`，非惰性
  后端省略）；构图进行中 `built` 返回 false（状态端点不阻塞等待构图锁）。
  文档写明需删库重建才能让新数据可检索。

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
