use ahash::{AHashMap, AHashSet};
use async_trait::async_trait;
use houyi_core::config::{BuildConfig, HnswConfig, VectorConfig};
use parking_lot::RwLock as PlRwLock;
use rayon::prelude::*;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{self, Duration};

pub mod cluster;
mod graph_snapshot;
pub mod hii;
mod hnsw_graph;

pub use graph_snapshot::{peek_graph_snapshot_dims_metric, GraphSnapshotError};
// 公开 `set_layer_multipliers` 供 houyi-server 启动时调用。
// Export `set_layer_multipliers` for houyi-server startup.
pub use hnsw_graph::set_layer_multipliers;
// 测试支撑（P4.5.0）：零范数阈值带的唯一真源。恢复属性测试必须与图路径用
// **同一个**常量分类，按值复刻会在阈值变更时静默失配（OCR 批次修订）。
// 非测试代码不应依赖：带内语义由 `normalize_for_cosine` 内部实现。
// EN: Test-support re-export so recovery properties classify with the same
// constant the graph uses instead of restating its value.
#[doc(hidden)]
pub use hnsw_graph::ZERO_NORM_BAND_NORM2;
pub mod distance;
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
mod distance_aarch64_neon;
/// x86_64 + `feature = "simd-avx2"`：AVX2 距离内联模块（`distance` 内部分派）。
/// EN: x86_64 AVX2 distance intrinsics; dispatched from `distance` module.
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
mod distance_x86_avx2;
pub mod hnsw_tuning;
pub mod pq;

pub use hnsw_tuning::{
    default_estimated_cardinality, infer_hnsw_params, load_hnsw_initial_params, InferredHnswParams,
};

pub type VectorId = u64;
const EMPTY_NEIGHBOR: u64 = u64::MAX;

/// 串行合图拆成「读锁规划 + 短写锁提交」。默认开启；
/// `HOUYI_HNSW_PLAN_APPLY` 取 `0`/`false`/`off`/`no`/空值（大小写不敏感、
/// 忽略首尾空白）时回退到旧的「整批持写锁合图」路径（兼容排查）。
/// EN: Plan(read-lock)/apply(short write-lock) split for the merge. Enabled by
/// default; any of `0`/`false`/`off`/`no`/empty (case-insensitive, trimmed)
/// disables it and falls back to the legacy whole-batch write-locked merge.
fn plan_apply_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("HOUYI_HNSW_PLAN_APPLY") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no" | ""
        ),
        Err(_) => true,
    })
}

/// 读锁规划/短写锁提交路径的逐批诊断日志（`HOUYI_HNSW_PLAN_APPLY_LOG=1`，默认关闭）。
/// EN: Per-batch diagnostics for the plan/apply path (opt-in).
fn plan_apply_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HOUYI_HNSW_PLAN_APPLY_LOG").ok().as_deref() == Some("1"))
}

/// 查询侧邻居扩展数量（`HOUYI_HNSW_RERANK_EXPAND`，0=关闭，进程内一次读取）。
/// EN: Query-side neighbor expansion count; 0 = off, read once per process.
fn rerank_expand() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("HOUYI_HNSW_RERANK_EXPAND")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// 距离度量。内核统一约定「距离越小越近」——内积因此取负点积，而非直接返回点积，
/// 否则所有上层排序/近邻选择都需要按度量翻转比较方向（双轨语义是历史 bug 来源）。
/// EN: Distance metric. The kernel uniformly treats smaller distance as closer, so
/// inner product is expressed as the negated dot product; returning the raw dot
/// would force every sorter/neighbor selector to flip comparison direction by metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// 欧氏距离的平方（不开方，单调等价且省一次 sqrt）。
    /// EN: Squared Euclidean distance (no sqrt — monotonic and cheaper).
    L2,
    /// 余弦距离 `1 - cosine_similarity`；图/写入路径按 L2 单位化后走预归一化点积。
    /// EN: Cosine distance `1 - cosine_similarity`; graph/write paths pre-normalize.
    Cosine,
    /// 内积距离 `-dot(a,b)`：点积越大（距离越小）越近，与 Milvus/DM 的 IP 语义一致。
    /// EN: Inner-product distance `-dot(a,b)`; larger dot (smaller distance) is
    /// closer, matching Milvus/DM IP semantics.
    InnerProduct,
    /// L1 / 曼哈顿距离 `Σ|a_i - b_i|`。
    /// EN: L1 / Manhattan distance `Σ|a_i - b_i|`.
    L1,
}

impl DistanceMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetric::L2 => "l2",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::InnerProduct => "ip",
            DistanceMetric::L1 => "l1",
        }
    }

    /// 从快照/配置字符串解析（ASCII 大小写不敏感）。接受常见别名，便于客户端
    /// 从 Milvus（`IP`）、DM（`MANHATTAN`）等迁移时直接复用其度量名。
    /// EN: Parses from a snapshot/config string (ASCII case-insensitive). Common
    /// aliases are accepted so clients migrating from Milvus (`IP`) or DM
    /// (`MANHATTAN`) can reuse their metric names directly.
    pub fn parse_str(s: &str) -> Option<Self> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("l2") || t.eq_ignore_ascii_case("euclidean") {
            Some(DistanceMetric::L2)
        } else if t.eq_ignore_ascii_case("cosine") {
            Some(DistanceMetric::Cosine)
        } else if t.eq_ignore_ascii_case("ip")
            || t.eq_ignore_ascii_case("inner_product")
            || t.eq_ignore_ascii_case("inner-product")
            || t.eq_ignore_ascii_case("dot")
        {
            Some(DistanceMetric::InnerProduct)
        } else if t.eq_ignore_ascii_case("l1") || t.eq_ignore_ascii_case("manhattan") {
            Some(DistanceMetric::L1)
        } else {
            None
        }
    }

    /// 图快照头部的度量字节（0=L2、1=Cosine、2=IP、3=L1）。旧二进制读到未知
    /// 字节会 fail-fast 拒绝快照并回退，而不是静默按错误度量解释图结构——因此
    /// 新增度量不需要 bump 快照版本，但降级重放新快照不可用。
    /// EN: Metric byte in the graph-snapshot header (0=L2, 1=Cosine, 2=IP, 3=L1).
    /// Old binaries reject unknown bytes and fall back instead of silently
    /// interpreting the graph under the wrong metric, so no snapshot version bump
    /// is needed — but downgraded binaries cannot replay newer snapshots.
    pub fn to_snapshot_byte(self) -> u8 {
        match self {
            DistanceMetric::L2 => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::InnerProduct => 2,
            DistanceMetric::L1 => 3,
        }
    }

    /// 解析图快照头部的度量字节；未知值返回 `None`（调用方映射 `InvalidMetric`）。
    /// EN: Parse the snapshot metric byte; unknown values yield `None`
    /// (callers map it to `InvalidMetric`).
    pub fn from_snapshot_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(DistanceMetric::L2),
            1 => Some(DistanceMetric::Cosine),
            2 => Some(DistanceMetric::InnerProduct),
            3 => Some(DistanceMetric::L1),
            _ => None,
        }
    }
}

#[cfg(feature = "simd-avx512")]
#[inline(always)]
pub(crate) fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    // AVX-512 feature 路径骨架：用 16 路分块累加，在接入真实 intrinsic 前保持独立实现边界。
    // EN: AVX-512 feature path skeleton: use 16-lane chunk accumulation to keep
    // EN: a dedicated implementation boundary before real intrinsics are wired.
    let mut sum = 0.0_f32;
    let mut i = 0usize;
    while i + 16 <= a.len() && i + 16 <= b.len() {
        for j in 0..16 {
            let d = a[i + j] - b[i + j];
            sum += d * d;
        }
        i += 16;
    }
    while i < a.len() && i < b.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(not(feature = "simd-avx512"))]
pub(crate) fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    distance::l2_squared(a, b)
}

pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    distance::cosine(a, b)
}

/// 连续向量缓存上的精确 top-k 扫描（HTTP label 过滤路径用）。
///
/// `ids` 与 `data` 一一对应（第 i 个向量是 `data[i*dims..(i+1)*dims]`）。
/// **Cosine 要求 `data` 已单位化**（缓存构建时做一次）；查询向量在函数内
/// 单位化。返回按距离升序、id 兜底的 top-k，语义与索引检索一致。
///
/// EN: Exact top-k over a contiguous vector cache (HTTP label-filter path).
/// `ids[i]` owns `data[i*dims..(i+1)*dims]`; for Cosine `data` must already be
/// L2-normalized (done once at cache build), and the query is normalized here.
pub fn search_flat_contiguous(
    query: &[f32],
    ids: &[VectorId],
    data: &[f32],
    dims: usize,
    k: usize,
    metric: DistanceMetric,
) -> Vec<(VectorId, f32)> {
    if k == 0 || dims == 0 || query.len() != dims || data.len() != ids.len() * dims {
        return Vec::new();
    }
    let q_storage;
    let q = if metric == DistanceMetric::Cosine {
        q_storage = {
            let mut v = query.to_vec();
            distance::normalize_in_place(&mut v);
            v
        };
        q_storage.as_slice()
    } else {
        query
    };
    let mut scored: Vec<(VectorId, f32)> = data
        .par_chunks(dims)
        .zip(ids.par_iter())
        .map(|(v, id)| {
            let d = match metric {
                DistanceMetric::L2 => distance::l2_squared(q, v),
                DistanceMetric::Cosine => distance::cosine_distance_prenormalized(q, v),
                DistanceMetric::InnerProduct => distance::inner_product_distance(q, v),
                DistanceMetric::L1 => distance::l1_distance(q, v),
            };
            (*id, d)
        })
        .collect();
    let cmp = |a: &(VectorId, f32), b: &(VectorId, f32)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
    if scored.len() > k {
        scored.select_nth_unstable_by(k - 1, cmp);
        scored.truncate(k);
    }
    scored.sort_by(cmp);
    scored
}

/// 向量数据标准接口.
/// Standard vector data interface.
pub trait VectorData: Send + Sync {
    fn dims(&self) -> usize;
    fn as_slice(&self) -> &[f32];
}

/// 无锁邻居存储接口（最小契约）.
/// Lock-free neighbor storage contract.
pub trait NeighborStore: Send + Sync {
    fn add_neighbor(&self, node: VectorId, neighbor: VectorId) -> bool;
    fn neighbors_of(&self, node: VectorId) -> Vec<VectorId>;
}

struct NodeNeighbors {
    len: AtomicUsize,
    slots: Vec<AtomicU64>,
}

impl NodeNeighbors {
    fn with_capacity(capacity: usize) -> Self {
        let slots = (0..capacity)
            .map(|_| AtomicU64::new(EMPTY_NEIGHBOR))
            .collect();
        Self {
            len: AtomicUsize::new(0),
            slots,
        }
    }

    fn push_lock_free(&self, neighbor: VectorId) -> bool {
        let capacity = self.slots.len();
        loop {
            let idx = self.len.load(Ordering::Acquire);
            if idx >= capacity {
                return false;
            }
            for scan in 0..idx {
                if self.slots[scan].load(Ordering::Acquire) == neighbor {
                    return true;
                }
            }
            if self
                .len
                .compare_exchange(idx, idx + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.slots[idx].store(neighbor, Ordering::Release);
                return true;
            }
        }
    }

    fn snapshot(&self) -> Vec<VectorId> {
        let limit = self.len.load(Ordering::Acquire).min(self.slots.len());
        (0..limit)
            .filter_map(|idx| {
                let v = self.slots[idx].load(Ordering::Acquire);
                if v == EMPTY_NEIGHBOR {
                    None
                } else {
                    Some(v)
                }
            })
            .collect()
    }
}

/// 基于原子数组 + CAS 的最小无锁邻居实现.
/// Minimal lock-free neighbor store using atomics and CAS.
pub struct LockFreeNeighborStore {
    slots_per_node: usize,
    nodes: PlRwLock<HashMap<VectorId, Arc<NodeNeighbors>>>,
}

impl LockFreeNeighborStore {
    pub fn new(slots_per_node: usize) -> Self {
        Self {
            slots_per_node,
            nodes: PlRwLock::new(HashMap::new()),
        }
    }

    fn get_or_create_node(&self, node: VectorId) -> Arc<NodeNeighbors> {
        if let Some(existing) = self.nodes.read().get(&node).cloned() {
            return existing;
        }
        let mut guard = self.nodes.write();
        guard
            .entry(node)
            .or_insert_with(|| Arc::new(NodeNeighbors::with_capacity(self.slots_per_node)))
            .clone()
    }
}

impl NeighborStore for LockFreeNeighborStore {
    fn add_neighbor(&self, node: VectorId, neighbor: VectorId) -> bool {
        if node == neighbor {
            return true;
        }
        let entry = self.get_or_create_node(node);
        entry.push_lock_free(neighbor)
    }

    fn neighbors_of(&self, node: VectorId) -> Vec<VectorId> {
        self.nodes
            .read()
            .get(&node)
            .map(|x| x.snapshot())
            .unwrap_or_default()
    }
}

/// 向量索引核心 Trait.
/// Core vector index contract.
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// 插入单条向量（HotBuffer 路径）。
    /// EN: Insert a single vector (HotBuffer path).
    async fn insert(&self, id: VectorId, vector: Vec<f32>) -> Result<(), String>;

    /// 批量插入（加载路径，后续接入异步构图）。
    /// EN: Batch insert (load path; async building is wired later).
    async fn insert_batch(&self, vectors: Vec<(VectorId, Vec<f32>)>) -> Result<(), String>;

    /// 混合检索：`filter_mask` 是**集合**（重复 id 按首次出现去重、不存在的
    /// id 忽略）；`None` = 不过滤，`Some(&[])` = 命中为空。
    /// Hybrid search: `filter_mask` is a set (duplicates deduped, unknown ids
    /// ignored); `None` = unfiltered, `Some(&[])` = empty result.。
    async fn search(
        &self,
        query: &[f32],
        k: usize,
        filter_mask: Option<&[VectorId]>,
    ) -> Vec<VectorId>;

    /// 混合检索并可选覆盖 `ef_search`（pg_wire / 基准与默认 `search` 对齐）。
    /// Optional per-query HNSW beam; defaults to [`search`] when unimplemented.
    async fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        filter_mask: Option<&[VectorId]>,
        ef_search_override: Option<usize>,
    ) -> Vec<VectorId> {
        let _ = ef_search_override;
        self.search(query, k, filter_mask).await
    }

    /// 触发全局优化（占位）。
    /// EN: Trigger global optimization (placeholder).
    fn optimize(&self);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushPolicy {
    ThresholdOnly,
    IntervalOnly,
    Hybrid,
}

#[derive(Default)]
struct BuilderMetrics {
    flush_count: AtomicUsize,
    total_flushed_items: AtomicUsize,
    max_pending_queue: AtomicUsize,
    last_flush_items: AtomicUsize,
    graph_edges_built: AtomicUsize,
    graph_edges_level0: AtomicUsize,
    graph_edges_upper: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuilderMetricsSnapshot {
    pub flush_count: usize,
    pub total_flushed_items: usize,
    pub max_pending_queue: usize,
    pub last_flush_items: usize,
    pub graph_edges_built: usize,
    pub graph_edges_level0: usize,
    pub graph_edges_upper: usize,
}

/// 单次 `run_async_builder_once` 的构图模式：批内 Rayon `plan`+apply vs 逐点 `insert`。
/// EN: One drain: batched parallel `plan_insert` path vs per-vector insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswBatchPlanMode {
    Parallel,
    Serial,
}

/// 单次 `run_async_builder_once` 的结果；若本队列为空未 drain，则 `flushed==0` 且 `plan_mode==None`。
/// EN: When the queue was empty, `flushed==0` and `plan_mode==None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsyncBuilderOnceReport {
    pub flushed: usize,
    pub plan_mode: Option<HnswBatchPlanMode>,
}

/// 热缓冲/构建队列条目：向量 ID + 行内拷贝的向量数据。
/// EN: hot-buffer/builder-queue entry: vector id plus an inline copy of the vector.
type BufferEntry = (VectorId, Vec<f32>);

/// 第一版 HotBuffer：保证插入后立即可见.
/// v0.1 hot buffer: read-your-write visibility.
pub struct HotBufferIndex {
    dims: usize,
    metric: DistanceMetric,
    hot_buffer: Arc<RwLock<Vec<BufferEntry>>>,
    /// 主存向量表：按 `id` 索引，Arc 共享避免与 HnswGraph 重复存储。
    /// EN: `HashMap` with `Arc` sharing to avoid duplicating vectors in HnswGraph.
    main_index: Arc<RwLock<AHashMap<VectorId, Arc<Vec<f32>>>>>,
    builder_queue: Arc<RwLock<VecDeque<BufferEntry>>>,
    metrics: Arc<BuilderMetrics>,
    /// Full HNSW graph (vectors duplicated for O(1) distance during search).
    /// 中文：标准 HNSW 分层图；与 `main_index` 同步插入。
    /// `Arc` 便于后续将构图 CPU 移出 Tokio（`spawn_blocking`）而不改调用方 API。
    /// EN: `Arc` enables future offloading graph CPU via `spawn_blocking` without API churn.
    hnsw: Arc<PlRwLock<hnsw_graph::HnswGraph>>,
    /// HNSW `m`；可在无 `&mut` 下由 HTTP 调参。 / EN: Layer-0 degree cap; tunable via HTTP `index.m`.
    /// `Arc`：合图任务在拿到串行门之后才读取，保证排队期间的新调参生效。
    /// EN: `Arc` so the build task reads it *after* acquiring the builder gate.
    hnsw_m: Arc<AtomicUsize>,
    /// 构图 `ef_construction` 目标值。 / EN: Build-time greedy list size target.
    hnsw_ef_construction: Arc<AtomicUsize>,
    /// 查询 `ef_search`（与单次请求 `ef_search` 取 max(k, ef)）。 / EN: Query beam; combined with k in `bounded_candidate_ids`.
    hnsw_ef_search: Arc<AtomicUsize>,
    force_flat_search: bool,
    /// 已删除 id 的墓碑集：闭合 `remove` 与并发 flush 的竞态——worker 合图前
    /// 按墓碑过滤在途批次；同 id 再次写入（insert/insert_batch）时清除（复活语义）。
    /// 集合规模上界 = 「已删除且未复活」的 id 数，不会随写入无限增长。
    /// EN: Tombstones of deleted ids closing the remove-vs-concurrent-flush race —
    /// the builder filters in-flight batches against them before merging; a later
    /// re-insert of the same id clears its tombstone (resurrection semantics).
    /// Bounded by "deleted but not resurrected" ids; does not grow with writes.
    tombstones: Arc<RwLock<AHashSet<VectorId>>>,
    /// 提交门：让 `plan_apply` 的「墓碑过滤 + 提交」与 `remove` 全程互斥，
    /// 但**不阻塞 insert**（insert 只与墓碑锁交互）。若改为在提交全程持墓碑
    /// 读锁，批量加载时每次 apply 都会让 insert_batch 的墓碑写锁排队：
    /// Cohere 1M 实测总加载 +11%（insert 854s → 987s）。
    /// EN: Commit gate serializing remove against the tombstone-filter + commit
    /// window without blocking inserts (which only take the tombstone lock).
    /// Holding the tombstone read lock across apply cost +11% on a 1M bulk load.
    commit_gate: Arc<RwLock<()>>,
    /// 合图串行门：`run_async_builder_once` 的调用方不止后台 worker——HTTP
    /// `/v1/optimize`、状态泵、恢复/快照排水都可能并发调用。两个 builder 各自
    /// drain 出同一 id（旧批与重插批）时，plan/apply 的前置 fresh 检查发生在
    /// 规划前、提交时不再校验，`apply_insert_plans` 的 `nodes.insert` 会静默
    /// 替换对方已提交的节点（泄漏 link_arena 块并丢边），`main.insert` 还可能
    /// 让旧向量覆盖新值。此门把整次合图串行化——同 id 的后续批次必然在本次
    /// 结束后才被 drain，届时走 upsert 回退路径先摘旧节点再插入。
    /// EN: Serializes `run_async_builder_once` callers (worker, HTTP optimize,
    /// status pump, recovery drains). Without it two builders can plan the same
    /// id from different in-flight batches and `apply_insert_plans` would replace
    /// the other node silently (leaking its link block) while `main.insert` could
    /// overwrite a newer value with a stale one. With the gate, any later batch
    /// for the same id drains only after this build finishes and then takes the
    /// upsert fallback (remove old node, re-insert).
    builder_gate: Arc<Mutex<()>>,
    /// 写入代数：每次 insert/insert_batch/remove/合图完成后自增。过滤向量缓存
    /// 用它在每次查询时校验新鲜度（builder 把批次 drain 到本地 staged 的窗口
    /// 无法被 vectors_by_ids 看到，只有合图完成后的自增能标记该窗口失效）。
    /// EN: Write generation — bumped on every insert/insert_batch/remove and after
    /// each builder merge. Filter caches compare it per query so a cache built while
    /// a batch was staged (invisible to vectors_by_ids) is invalidated on merge.
    write_generation: Arc<AtomicU64>,
    /// 批量构图配置（进程启动时读取一次，替代散落 env var）。
    /// Build config snapshot at startup; replaces scattered env reads.
    build_config: BuildConfig,
}

/// HNSW 图锁取 guard 的统一入口。锁用 parking_lot::RwLock：
/// (1) 写者优先公平调度——std RwLock 读优先,高 QPS 读锁风暴会饿死合图写锁
///     (spawn_blocking flush 永远拿不到锁,builder 队列无限积压;并发测试
///     60s 超时三次实证:无间隔读/1ms 间隔/workspace 并行各一次);
/// (2) 无中毒语义——std RwLock 持锁 panic 即中毒,`.expect("poisoned")` 会把
///     单点图异常放大成服务级连环 panic(混合读写压测实证);parking_lot 锁
///     不中毒,搜索路径对半更新图的防御性跳过(见 hnsw_graph)继续兜底。
/// EN: Unified guard accessors for the HNSW graph lock, backed by
/// parking_lot::RwLock: (1) write-preferring fairness — std RwLock is
/// reader-preferring, so a high-QPS read storm starves the merge write lock
/// (spawn_blocking flush never acquires it; the builder queue grows
/// unboundedly — three 60s test timeouts proved: gapless reads, 1ms gaps,
/// full-parallel workspace runs); (2) no poisoning — a panic under std RwLock
/// poisons it and `.expect("poisoned")` escalates a local glitch into a
/// service-wide cascade; parking_lot locks never poison, and the defensive
/// dangling-link skips in hnsw_graph remain as the safety net.
fn hnsw_read_guard(
    hnsw: &PlRwLock<hnsw_graph::HnswGraph>,
) -> parking_lot::RwLockReadGuard<'_, hnsw_graph::HnswGraph> {
    hnsw.read()
}

fn hnsw_write_guard(
    hnsw: &PlRwLock<hnsw_graph::HnswGraph>,
) -> parking_lot::RwLockWriteGuard<'_, hnsw_graph::HnswGraph> {
    hnsw.write()
}

pub struct BuilderWorker {
    stop: Arc<AtomicBool>,
    /// `Option` 只因 Drop 与 by-value `shutdown` 共存：`shutdown` 需要 `take()`
    /// 出 JoinHandle 才能 await（实现了 Drop 的类型禁止部分移动字段）。
    /// EN: `Option` exists solely so by-value `shutdown` can `take()` the
    /// JoinHandle — types implementing `Drop` forbid moving fields out.
    handle: Option<JoinHandle<()>>,
}

impl BuilderWorker {
    pub async fn shutdown(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

/// 句柄被丢弃时置停止位：worker 任务在下一个 tick 自行退出。
/// 没有此 Drop 时，仅丢弃句柄会 detach 任务——它带着 `Arc<HotBufferIndex>`
/// 永远空转（Collection 被移除场景的任务+索引内存双泄漏）。
/// `shutdown()` 仍是即时退出的首选路径；Drop 只保证「最终停止」。
/// EN: Dropping the handle sets the stop flag so the worker task exits on its next
/// tick. Without this, a dropped handle would detach the task — spinning forever
/// while holding an `Arc<HotBufferIndex>` (task + index memory leak when a
/// collection is removed). `shutdown()` remains the preferred immediate path;
/// Drop only guarantees eventual stop.
impl Drop for BuilderWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl HotBufferIndex {
    pub fn new(dims: usize) -> Self {
        Self::new_with_metric_and_config(
            dims,
            DistanceMetric::L2,
            &VectorConfig::from_env(),
            &HnswConfig::from_env(),
        )
    }

    pub fn new_with_metric(dims: usize, metric: DistanceMetric) -> Self {
        Self::new_with_metric_and_config(
            dims,
            metric,
            &VectorConfig::from_env(),
            &HnswConfig::from_env(),
        )
    }

    /// 使用外部配置构造（用于集中配置管理）。
    /// Construct with external configs.
    pub fn new_with_metric_and_config(
        dims: usize,
        metric: DistanceMetric,
        vector_cfg: &VectorConfig,
        hnsw_cfg: &HnswConfig,
    ) -> Self {
        let (p, _) = crate::hnsw_tuning::load_hnsw_initial_params(dims, vector_cfg);
        Self {
            dims,
            metric,
            hot_buffer: Arc::new(RwLock::new(Vec::new())),
            main_index: Arc::new(RwLock::new(AHashMap::new())),
            builder_queue: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(BuilderMetrics::default()),
            hnsw: Arc::new(PlRwLock::new(hnsw_graph::HnswGraph::new(
                dims,
                metric,
                p.m,
                p.ef_construction,
                p.ef_search,
                hnsw_cfg.seed,
                hnsw_cfg.max_layers,
            ))),
            hnsw_m: Arc::new(AtomicUsize::new(p.m)),
            hnsw_ef_construction: Arc::new(AtomicUsize::new(p.ef_construction)),
            hnsw_ef_search: Arc::new(AtomicUsize::new(p.ef_search)),
            force_flat_search: vector_cfg.force_flat_search,
            tombstones: Arc::new(RwLock::new(AHashSet::new())),
            commit_gate: Arc::new(RwLock::new(())),
            builder_gate: Arc::new(Mutex::new(())),
            write_generation: Arc::new(AtomicU64::new(0)),
            build_config: BuildConfig::from_env(),
        }
    }

    /// 使用外部 build 配置（供 houyi-server 传入集中管理的配置）。
    /// EN: Use an external build config (houyi-server passes centrally managed config).
    pub fn set_build_config(&mut self, cfg: BuildConfig) {
        self.build_config = cfg;
    }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // 内积/IP 走 `-a·b`、L1 走曼哈顿原值，二者都保持「越小越近」的全序约定。
        // EN: IP uses `-a·b` and L1 the raw Manhattan sum; both keep the
        // "smaller = closer" total order the kernel relies on.
        match self.metric {
            DistanceMetric::L2 => l2_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::InnerProduct => distance::inner_product_distance(a, b),
            DistanceMetric::L1 => distance::l1_distance(a, b),
        }
    }

    pub fn set_hnsw_ef_search(&self, ef: usize) -> Result<(), String> {
        if ef == 0 {
            return Err("hnsw_ef_search must be > 0".to_string());
        }
        self.hnsw_ef_search.store(ef, Ordering::Release);
        hnsw_write_guard(&self.hnsw).set_ef_search(ef);
        Ok(())
    }

    pub fn hnsw_ef_search(&self) -> usize {
        self.hnsw_ef_search.load(Ordering::Acquire)
    }

    /// 设置构图 `m`（仅影响后续 `insert`；与 benchmark 的 index 参数对齐）。
    /// EN: Neighbor cap per layer; affects inserts after call.
    pub fn set_hnsw_m(&self, m: usize) -> Result<(), String> {
        if m < 2 {
            return Err("hnsw_m must be >= 2".to_string());
        }
        self.hnsw_m.store(m, Ordering::Release);
        hnsw_write_guard(&self.hnsw).set_m(m);
        Ok(())
    }

    pub fn hnsw_m(&self) -> usize {
        self.hnsw_m.load(Ordering::Acquire)
    }

    /// 设置构图 `ef_construction`（bulk 阶段仍受 `HOUYI_HNSW_BULK_EF_CONSTRUCTION` 约束）。
    /// EN: Sets build `ef_construction` (bulk phase is still bound by `HOUYI_HNSW_BULK_EF_CONSTRUCTION`).
    pub fn set_hnsw_ef_construction(&self, ef: usize) -> Result<(), String> {
        if ef == 0 {
            return Err("hnsw_ef_construction must be > 0".to_string());
        }
        self.hnsw_ef_construction.store(ef, Ordering::Release);
        Ok(())
    }

    pub fn hnsw_ef_construction_value(&self) -> usize {
        self.hnsw_ef_construction.load(Ordering::Acquire)
    }

    fn maybe_run_sanity_check(
        &self,
        query: &[f32],
        k: usize,
        approx_sorted: &[(VectorId, f32)],
        hot: &[(VectorId, Vec<f32>)],
        main: &AHashMap<VectorId, Arc<Vec<f32>>>,
    ) {
        if std::env::var("HOUYI_VECTOR_SANITY_CHECK").ok().as_deref() != Some("1") {
            return;
        }
        if k == 0 || main.is_empty() {
            return;
        }

        let sample_cap = 100usize.min(main.len());
        let mut brute: Vec<(VectorId, f32)> = main
            .iter()
            .take(sample_cap)
            .map(|(id, v)| (*id, self.distance(query, v)))
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let brute_top: Vec<VectorId> = brute.into_iter().take(k).map(|(id, _)| id).collect();
        let approx_top: Vec<VectorId> = approx_sorted.iter().take(k).map(|(id, _)| *id).collect();
        if brute_top.is_empty() || approx_top.is_empty() {
            return;
        }
        let hit = approx_top
            .iter()
            .filter(|id| brute_top.contains(id))
            .count();
        if hit == 0 {
            let fallback_source = hot
                .iter()
                .map(|(id, _)| *id)
                .chain(main.keys().copied())
                .take(5)
                .collect::<Vec<_>>();
            tracing::warn!(
                k,
                sample = sample_cap,
                ?approx_top,
                ?brute_top,
                ids_head = ?fallback_source,
                "[sanity-check] ANN top-k has zero overlap with brute-force sample"
            );
        }
    }

    /// 将批量加载队列异步并行处理后合并入主索引。
    /// Drain async-builder queue and merge into main index.
    ///
    /// 取消安全：整个合图（drain → 过滤 → spawn_blocking join）在独立任务中
    /// 执行并持有 `builder_gate`。调用方 future 被 drop（HTTP 断连、任务 abort）
    /// 时构图仍会完成——门锁不会提前释放，已 drain 的批次也不会丢失。
    /// EN: Cancellation-safe — the whole build (drain → filter → blocking join)
    /// runs in a detached task that owns the builder gate. Dropping the caller
    /// future (client disconnect, abort) neither releases the gate early nor
    /// drops already-drained batches.
    pub async fn run_async_builder_once(&self) -> AsyncBuilderOnceReport {
        let cfg = self.build_config.clone();
        let builder_gate = Arc::clone(&self.builder_gate);
        let builder_queue = Arc::clone(&self.builder_queue);
        let tombstones = Arc::clone(&self.tombstones);
        let commit_gate = Arc::clone(&self.commit_gate);
        let main_index = Arc::clone(&self.main_index);
        let hnsw = Arc::clone(&self.hnsw);
        let metrics = Arc::clone(&self.metrics);
        let write_generation = Arc::clone(&self.write_generation);
        let hnsw_ef_search = Arc::clone(&self.hnsw_ef_search);
        let hnsw_m = Arc::clone(&self.hnsw_m);
        let hnsw_ef_construction = Arc::clone(&self.hnsw_ef_construction);
        // 外层任务只做观察：即使调用方 future 被 drop，内层构图 panic 仍会被
        // 记录（否则 detached 构建失败完全不可观测）。
        // EN: The outer task only observes; even when the caller is dropped, an
        // inner build panic is still logged instead of vanishing silently.
        let join = tokio::spawn(async move {
            let inner = tokio::spawn(Self::run_async_builder_once_inner(
                cfg,
                builder_gate,
                builder_queue,
                tombstones,
                commit_gate,
                main_index,
                hnsw,
                metrics,
                write_generation,
                hnsw_ef_search,
                hnsw_m,
                hnsw_ef_construction,
            ));
            match inner.await {
                Ok(report) => report,
                Err(e) if e.is_panic() => {
                    let payload = e.into_panic();
                    let msg = payload
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
                        .unwrap_or("<non-string payload>");
                    tracing::error!(
                        panic = %msg,
                        "[builder] merge task panicked; in-flight batch left to WAL replay"
                    );
                    std::panic::resume_unwind(payload);
                }
                Err(e) => panic!("builder task cancelled: {e}"),
            }
        });
        match join.await {
            Ok(report) => report,
            Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
            Err(e) => panic!("builder task cancelled: {e}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_async_builder_once_inner(
        cfg: BuildConfig,
        builder_gate: Arc<Mutex<()>>,
        builder_queue: Arc<RwLock<VecDeque<BufferEntry>>>,
        tombstones: Arc<RwLock<AHashSet<VectorId>>>,
        commit_gate: Arc<RwLock<()>>,
        main_index: Arc<RwLock<AHashMap<VectorId, Arc<Vec<f32>>>>>,
        hnsw: Arc<PlRwLock<hnsw_graph::HnswGraph>>,
        metrics: Arc<BuilderMetrics>,
        write_generation: Arc<AtomicU64>,
        hnsw_ef_search: Arc<AtomicUsize>,
        hnsw_m: Arc<AtomicUsize>,
        hnsw_ef_construction: Arc<AtomicUsize>,
    ) -> AsyncBuilderOnceReport {
        // 串行化合图调用（多个调用方会并发进入）：见 `builder_gate` 字段说明。
        // EN: Serialize builder invocations (multiple callers enter concurrently) —
        // see the `builder_gate` field docs.
        let _builder_gate = builder_gate.lock().await;
        // 调参值在拿到串行门之后再读：排队等待长合图期间经 HTTP set_* 或
        // 重放恢复所做的变更必须对本批次生效。
        // EN: Read tuning knobs *after* acquiring the gate so changes made while
        // queued behind a long merge (HTTP set_* / replay restore) take effect.
        let hnsw_ef_search = hnsw_ef_search.load(Ordering::Acquire);
        let hnsw_m = hnsw_m.load(Ordering::Acquire);
        let hnsw_ef_construction = hnsw_ef_construction.load(Ordering::Acquire);
        let max_build_batch = cfg.max_build_batch;
        let drain_all = cfg.optimize_drain_all;
        let breakdown_enabled = cfg.breakdown_enabled;
        let drain_t0 = std::time::Instant::now();
        let staged = {
            let mut queue = builder_queue.write().await;
            if queue.is_empty() {
                Vec::new()
            } else if drain_all {
                queue.drain(..).collect::<Vec<_>>()
            } else {
                let n = queue.len().min(max_build_batch);
                queue.drain(..n).collect::<Vec<_>>()
            }
        };
        let drain_elapsed_us = drain_t0.elapsed().as_micros();
        if staged.is_empty() {
            return AsyncBuilderOnceReport {
                flushed: 0,
                plan_mode: None,
            };
        }

        // Build graph incrementally: each new vector attaches through ef-construction search.
        // 中文：单图必须串行 `insert`；同一 `write` 锁内一次处理本批全部点，减少锁/段间 yield 开销。
        // EN: Sequential inserts; one lock scope per drained batch (fewer yields vs 32-sized sub-chunks).
        // 并行 `link_new_point` 需 GraphLayersBuilder 级细粒度同步（见 HNSW_ENGINE_ROADMAP §4），未在本函数内用 Rayon 抢全局写锁。
        //
        // 墓碑过滤（第一道）：drain 之后、合图之前被 `remove` 的 id 不得进图。
        // 这道快照只是廉价早退——它无法覆盖「快照之后、闭包拿到 main 写锁之前落完
        // 整个临界区的 remove」（其显式摘除找不到尚未合入的 id，会漏）；真正的收敛
        // 靠闭包内拿到合并锁后的第二道过滤（见下）。
        // EN: First-pass tombstone filter — a cheap early-out only. It cannot cover a
        // remove() whose whole critical section lands between this snapshot and the
        // closure acquiring the main write lock (its explicit purge finds nothing yet
        // and misses); convergence comes from the second filter inside the closure.
        let built: Vec<(VectorId, Vec<f32>)> = {
            let ts = tombstones.read().await;
            if ts.is_empty() {
                staged
            } else {
                staged
                    .into_iter()
                    .filter(|(id, _)| !ts.contains(id))
                    .collect()
            }
        };
        if built.is_empty() {
            return AsyncBuilderOnceReport {
                flushed: 0,
                plan_mode: None,
            };
        }
        let collect_edge_metrics = cfg.collect_edge_metrics;
        let parallel_batch_build = cfg.parallel_batch_build;
        let parallel_min_batch = cfg.parallel_min_batch;
        let parallel_min_graph = cfg.parallel_min_graph;
        let parallel_chunk_size = cfg.parallel_chunk_size;
        let parallel_min_chunk = cfg.parallel_min_chunk;
        let parallel_plan_group_chunks = cfg.parallel_plan_group_chunks;
        let plan_group_size = parallel_chunk_size
            .saturating_mul(parallel_plan_group_chunks)
            .max(1);
        let seed_serial_points = cfg.seed_serial_points;
        let commit_chunk = cfg.commit_chunk;
        let bulk_ef_cap = cfg.bulk_ef_construction_cap;
        let build_t0 = std::time::Instant::now();
        let report = tokio::task::spawn_blocking(move || {
            let mut part_us: u128 = 0;
            let mut levels_us: u128 = 0;
            let mut fresh_plan_us: u128 = 0;
            let mut fresh_apply_us: u128 = 0;
            let mut fresh_serial_us: u128 = 0;
            let mut upsert_serial_us: u128 = 0;
            let mut serial_fallback_us: u128 = 0;
            let mut group_count: usize = 0;
            let mut fresh_count: usize = 0;
            let mut upsert_count: usize = 0;
            // 墓碑过滤（第二道，收敛关键）：在持有合并锁的临界区内按最新墓碑集
            // 再滤一次。remove() 全程持墓碑写锁（先记墓碑再清除），因此要么
            // 它已整体完成（墓碑可见，这里滤掉），要么它被本闭包挡在 ts 读锁外
            // （合图完成后其显式摘除兜底）——不存在第三种时序。
            // 锁序与全局一致：tombstones → main → hnsw（remove 同序，无逆序成环）。
            // EN: Second-pass filter inside the merge critical section — the actual
            // convergence point. remove() holds the tombstone write lock across its
            // whole purge, so it either completed before this read (tombstone
            // visible, filtered here) or is blocked on the ts read lock until the
            // merge finishes (its explicit purge then takes over). No third
            // interleaving. Lock order matches the global order ts → main → hnsw.
            let mut built: Vec<(VectorId, Vec<f32>)> = {
                let ts = tombstones.blocking_read();
                if ts.is_empty() {
                    built
                } else {
                    built
                        .into_iter()
                        .filter(|(id, _)| !ts.contains(id))
                        .collect()
                }
            };
            if built.is_empty() {
                return AsyncBuilderOnceReport {
                    flushed: 0,
                    plan_mode: None,
                };
            }
            // 默认路径：读锁规划 + 短写锁提交（fresh 批次）；失败回退既有路径。
            // EN: Default read-lock plan + short write-lock apply for fresh batches.
            if plan_apply_enabled() {
                let m = hnsw_m;
                let requested = hnsw_ef_construction
                    .max(m.saturating_mul(2))
                    .clamp(16, 4096);
                let ef_c = bulk_ef_cap.map_or(requested, |cap| requested.min(cap));
                match Self::plan_apply_fresh(
                    &main_index,
                    &hnsw,
                    &metrics,
                    &tombstones,
                    &commit_gate,
                    built,
                    m,
                    ef_c,
                    hnsw_ef_search,
                    seed_serial_points,
                    plan_group_size,
                    parallel_batch_build,
                ) {
                    Ok(report) => return report,
                    Err(returned) => built = returned,
                }
            }
            let count = built.len();
            let mut main = main_index.blocking_write();
            let mut guard = hnsw_write_guard(&hnsw);
            let m = hnsw_m;
            guard.set_m(m);
            let requested = hnsw_ef_construction
                .max(m.saturating_mul(2))
                .clamp(16, 4096);
            let ef_c = bulk_ef_cap.map_or(requested, |cap| requested.min(cap));
            guard.set_ef_construction(ef_c);
            guard.set_ef_search(hnsw_ef_search);
            let mut visited_build = hnsw_graph::HnswVisitTracker::with_capacity(
                hnsw_graph::HnswGraph::visited_insert_capacity_hint(ef_c),
            );
            let edge_before = if collect_edge_metrics {
                guard.edge_count_hint()
            } else {
                0
            };
            // Phase-1 parallel attempt: precompute entry hints in parallel, then serially commit graph links.
            // 中文：先并行做只读近邻 hint，再串行提交，保证图结构一致性与 recall。
            let initial_graph_len = guard.len();
            let use_parallel_mode = parallel_batch_build
                && count >= parallel_min_batch
                && initial_graph_len.saturating_add(count) >= parallel_min_graph;
            if use_parallel_mode {
                // 整批先分片：fresh 走并行规划，upsert 固定串行安全路径。
                // EN: Partition once at batch-level; fresh for parallel planning, upserts stay serial-safe.
                let part_t0 = std::time::Instant::now();
                let mut fresh_all = Vec::with_capacity(count);
                let mut upserts_all = Vec::new();
                for (id, vector) in built {
                    if main.contains_key(&id) {
                        upserts_all.push((id, vector));
                    } else {
                        fresh_all.push((id, vector));
                    }
                }
                part_us = part_t0.elapsed().as_micros();
                fresh_count = fresh_all.len();
                upsert_count = upserts_all.len();

                if seed_serial_points > 0 {
                    let gsz = guard.len();
                    if gsz < seed_serial_points && !fresh_all.is_empty() {
                        let need = seed_serial_points - gsz;
                        let take = need.min(fresh_all.len());
                        let seed_t0 = std::time::Instant::now();
                        for (id, vector) in fresh_all.drain(..take) {
                            let arc_v = Arc::new(vector);
                            main.insert(id, Arc::clone(&arc_v));
                            guard.insert_with_visited_entry_hint(
                                id,
                                arc_v,
                                &mut visited_build,
                                None,
                            );
                        }
                        fresh_serial_us =
                            fresh_serial_us.saturating_add(seed_t0.elapsed().as_micros());
                    }
                }

                // fresh 分组并行规划 + 单次批量原子合入，降低提交轮次。
                // EN: Grouped parallel planning + single batch apply to reduce serial commit rounds.
                let plan_group_size =
                    parallel_chunk_size.saturating_mul(parallel_plan_group_chunks);
                let levels_t0 = std::time::Instant::now();
                let fresh_levels = guard.draw_random_levels(fresh_all.len());
                levels_us = levels_t0.elapsed().as_micros();
                let mut level_cursor = 0usize;
                while !fresh_all.is_empty() {
                    group_count = group_count.saturating_add(1);
                    let take = plan_group_size.min(fresh_all.len());
                    let group: Vec<(VectorId, Vec<f32>)> = fresh_all.drain(..take).collect();
                    let levels_slice = &fresh_levels[level_cursor..level_cursor + take];
                    level_cursor += take;
                    if group.len() < parallel_min_chunk {
                        let serial_t0 = std::time::Instant::now();
                        for (id, vector) in group.into_iter() {
                            let arc_v = Arc::new(vector);
                            main.insert(id, Arc::clone(&arc_v));
                            guard.insert_with_visited_entry_hint(
                                id,
                                arc_v,
                                &mut visited_build,
                                None,
                            );
                        }
                        fresh_serial_us =
                            fresh_serial_us.saturating_add(serial_t0.elapsed().as_micros());
                        continue;
                    }
                    let plan_t0 = std::time::Instant::now();
                    let mut apply_plans: Vec<_> = {
                        let g_ref: &hnsw_graph::HnswGraph = &guard;
                        group
                            .into_par_iter()
                            .zip(levels_slice.par_iter().copied())
                            .map(|((id, vector), level)| {
                                g_ref.plan_insert_with_level_unchecked(
                                    id,
                                    Arc::new(vector),
                                    level,
                                    None,
                                )
                            })
                            .collect()
                    };
                    fresh_plan_us = fresh_plan_us.saturating_add(plan_t0.elapsed().as_micros());
                    let apply_t0 = std::time::Instant::now();
                    for plan in &mut apply_plans {
                        main.insert(plan.id, Arc::clone(&plan.vector));
                    }
                    guard.apply_insert_plans(apply_plans);
                    fresh_apply_us = fresh_apply_us.saturating_add(apply_t0.elapsed().as_micros());
                }

                let upsert_t0 = std::time::Instant::now();
                for (id, vector) in upserts_all.into_iter() {
                    let arc_v = Arc::new(vector);
                    match main.entry(id) {
                        Entry::Occupied(mut occ) => {
                            let _ = guard.remove_node(id);
                            occ.insert(Arc::clone(&arc_v));
                        }
                        Entry::Vacant(vac) => {
                            vac.insert(Arc::clone(&arc_v));
                        }
                    }
                    guard.insert_with_visited_entry_hint(id, arc_v, &mut visited_build, None);
                }
                upsert_serial_us = upsert_t0.elapsed().as_micros();
            } else if commit_chunk > 0 {
                // 短锁提交（P4.x streaming 调优）：保留原串行合图的工作量与吞吐，
                // 但每 `commit_chunk` 点释放一次 main/hnsw 写锁，让等待中的检索
                // 在子批之间拿到读锁。锁序仍为 ts → main → hnsw；每个子批重新
                // 获取墓碑读锁并二次过滤，删除不会漏进图。
                // EN: Short-lock commit — same serial merge work/throughput, but the
                // main/hnsw write locks are released every `commit_chunk` inserts so
                // waiting searches can run between sub-batches. Lock order stays
                // ts → main → hnsw, and each sub-batch re-checks tombstones.
                let serial_t0 = std::time::Instant::now();
                drop(guard);
                drop(main);
                let mut iter = built.into_iter().peekable();
                loop {
                    let chunk: Vec<(VectorId, Vec<f32>)> =
                        iter.by_ref().take(commit_chunk).collect();
                    if chunk.is_empty() {
                        break;
                    }
                    let ts = tombstones.blocking_read();
                    let chunk: Vec<(VectorId, Vec<f32>)> = chunk
                        .into_iter()
                        .filter(|(id, _)| !ts.contains(id))
                        .collect();
                    let mut m = main_index.blocking_write();
                    let mut g = hnsw_write_guard(&hnsw);
                    for (id, vector) in chunk {
                        let arc_v = Arc::new(vector);
                        match m.entry(id) {
                            Entry::Occupied(mut occ) => {
                                let _ = g.remove_node(id);
                                occ.insert(Arc::clone(&arc_v));
                            }
                            Entry::Vacant(vac) => {
                                vac.insert(Arc::clone(&arc_v));
                            }
                        }
                        g.insert_with_visited_entry_hint(id, arc_v, &mut visited_build, None);
                    }
                    drop(g);
                    drop(m);
                    drop(ts);
                    if iter.peek().is_some() {
                        // 让出 CPU，给已排队的读锁一个窗口（写者不饥饿）。
                        // EN: Yield so queued readers get a window between chunks.
                        std::thread::yield_now();
                    }
                }
                // 恢复 guard 供后面的 edge 指标读取（main 已无后续使用）。
                // EN: Reacquire the graph guard for the edge metric; main is not used after.
                guard = hnsw_write_guard(&hnsw);
                serial_fallback_us = serial_t0.elapsed().as_micros();
            } else {
                let serial_t0 = std::time::Instant::now();
                for (id, vector) in built.into_iter() {
                    let arc_v = Arc::new(vector);
                    match main.entry(id) {
                        Entry::Occupied(mut occ) => {
                            let _ = guard.remove_node(id);
                            occ.insert(Arc::clone(&arc_v));
                        }
                        Entry::Vacant(vac) => {
                            vac.insert(Arc::clone(&arc_v));
                        }
                    }
                    guard.insert_with_visited_entry_hint(id, arc_v, &mut visited_build, None);
                }
                serial_fallback_us = serial_t0.elapsed().as_micros();
            }
            let delta = if collect_edge_metrics {
                let edge_after = guard.edge_count_hint();
                edge_after.saturating_sub(edge_before)
            } else {
                // 关闭全图扫描时，使用近似值维持指标单调性：每新点最多贡献约 `m` 条无向边。
                // EN: When exact scan is disabled, use an approximate monotonic edge delta.
                count.saturating_mul(m)
            };
            metrics
                .graph_edges_built
                .fetch_add(delta, Ordering::Relaxed);
            let level0 = delta / 2;
            let upper = delta.saturating_sub(level0);
            metrics
                .graph_edges_level0
                .fetch_add(level0, Ordering::Relaxed);
            metrics
                .graph_edges_upper
                .fetch_add(upper, Ordering::Relaxed);

            // 另见 `AsyncBuilderOnceReport` / `POST /v1/optimize` 的 `hnsw_*_batch_rounds`；本行仅在 HOUYI_HNSW_OPTIMIZE_BREAKDOWN=1 时打印。
            // See also `AsyncBuilderOnceReport` and optimize JSON; this line only when OPTIMIZE_BREAKDOWN=1.
            if breakdown_enabled {
                let build_total_us = build_t0.elapsed().as_micros();
                tracing::info!(
                    count,
                    mode = if use_parallel_mode {
                        "parallel"
                    } else {
                        "serial"
                    },
                    drain_us = drain_elapsed_us,
                    build_total_us,
                    partition_us = part_us,
                    levels_us,
                    fresh_plan_us,
                    fresh_apply_us,
                    fresh_serial_us,
                    upsert_serial_us,
                    serial_fallback_us,
                    groups = group_count,
                    fresh = fresh_count,
                    upserts = upsert_count,
                    edges_delta = delta,
                    "[optimize-breakdown]"
                );
            }
            AsyncBuilderOnceReport {
                flushed: count,
                plan_mode: Some(if use_parallel_mode {
                    HnswBatchPlanMode::Parallel
                } else {
                    HnswBatchPlanMode::Serial
                }),
            }
        })
        .await
        .expect("hnsw spawn_blocking join failed");
        if report.flushed > 0 {
            write_generation.fetch_add(1, Ordering::Release);
        }
        if cfg.builder_yield {
            tokio::task::yield_now().await;
        }
        report
    }

    /// fresh 批次先读锁规划、再短写锁提交。失败（upsert/未播种）时原样返回
    /// 批次，调用方回退既有合图路径。
    /// EN: Fresh-batch plan(read-lock)/apply(write-lock) path. Returns the batch
    /// back on precondition failure so the caller falls back.
    #[allow(clippy::too_many_arguments)]
    fn plan_apply_fresh(
        main_index: &RwLock<AHashMap<VectorId, Arc<Vec<f32>>>>,
        hnsw: &PlRwLock<hnsw_graph::HnswGraph>,
        metrics: &BuilderMetrics,
        tombstones: &RwLock<AHashSet<VectorId>>,
        commit_gate: &RwLock<()>,
        built: Vec<BufferEntry>,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seed_serial_points: usize,
        plan_group_size: usize,
        parallel_plan: bool,
    ) -> Result<AsyncBuilderOnceReport, Vec<BufferEntry>> {
        if built.is_empty() {
            return Ok(AsyncBuilderOnceReport {
                flushed: 0,
                plan_mode: Some(HnswBatchPlanMode::Serial),
            });
        }
        // 只处理 fresh 批次；upsert 走原路径（需要先摘旧节点）。
        // EN: Only processes fresh batches; upserts use the original path (detach old nodes first).
        {
            let main = main_index.blocking_read();
            if built.iter().any(|(id, _)| main.contains_key(id)) {
                return Err(built);
            }
        }
        // 图必须先播种，规划才有入口点；首批仍走原串行路径。
        // EN: The graph must be seeded first so planning has an entry point; the first batch stays serial.
        {
            let g = hnsw_read_guard(hnsw);
            if g.len() < seed_serial_points.max(1) {
                return Err(built);
            }
        }
        let count = built.len();
        let levels = {
            let mut g = hnsw_write_guard(hnsw);
            g.set_m(m);
            g.set_ef_construction(ef_construction);
            g.set_ef_search(ef_search);
            g.draw_random_levels(count)
        };
        let arcs: Vec<(VectorId, Arc<Vec<f32>>)> =
            built.into_iter().map(|(id, v)| (id, Arc::new(v))).collect();
        // 按既有并行分组（chunk×group_chunks）滚动：每组先读锁并行规划，
        // 再短写锁提交，组间保持「后组能看到前组」的可见性（与旧并行路径
        // 的图质量口径一致）。规划阶段检索可并发读图。
        // EN: Roll in the existing parallel grouping (chunk × group_chunks):
        // each group plans in parallel under the read lock, then commits under a
        // short write lock. Later groups see earlier ones (same visibility as the
        // old parallel path); searches read the graph concurrently while planning.
        let mut kept_total = 0usize;
        let mut cursor = 0usize;
        let mut plan_us: u128 = 0;
        let mut apply_us: u128 = 0;
        while cursor < count {
            let take = plan_group_size.min(count - cursor).max(1);
            let group = &arcs[cursor..cursor + take];
            let group_levels = &levels[cursor..cursor + take];
            let plan_t0 = std::time::Instant::now();
            let plans: Vec<_> = {
                let g = hnsw_read_guard(hnsw);
                if parallel_plan {
                    group
                        .par_iter()
                        .zip(group_levels.par_iter().copied())
                        .map(|((id, v), level)| {
                            g.plan_insert_with_level_unchecked(*id, Arc::clone(v), level, None)
                        })
                        .collect()
                } else {
                    group
                        .iter()
                        .zip(group_levels.iter().copied())
                        .map(|((id, v), level)| {
                            g.plan_insert_with_level_unchecked(*id, Arc::clone(v), level, None)
                        })
                        .collect()
                }
            };
            plan_us = plan_us.saturating_add(plan_t0.elapsed().as_micros());
            cursor += take;
            let apply_t0 = std::time::Instant::now();
            // 第三道墓碑过滤（规划期间可能删点）。提交门（gate）覆盖「过滤 +
            // 提交」：remove 持 gate 写锁贯穿「记墓碑 + 逐结构清除」，因此要么
            // 删除已完成（这里滤掉），要么被挡在门外（提交后其清除兜底）。
            // gate 与 insert 无关——insert 只取墓碑写锁，批量加载不会被 apply
            // 阻塞。锁序：gate → ts → main → hnsw（remove 同序）。
            // EN: Third-pass filter for deletes that landed during planning. The
            // commit gate spans filter + commit; remove() holds its write lock
            // across "record tombstone + purge". So a delete is either already
            // visible (filtered here) or blocked outside (its purge covers the
            // committed nodes). Inserts never take the gate (only the tombstone
            // lock), so bulk loading is not stalled by apply windows. Lock order:
            // gate → ts → main → hnsw (same as remove()).
            let gate = commit_gate.blocking_read();
            let ts = tombstones.blocking_read();
            let mut kept: Vec<_> = plans.into_iter().filter(|p| !ts.contains(&p.id)).collect();
            drop(ts);
            if !kept.is_empty() {
                kept_total += kept.len();
                // 向量入 main 用短写锁；图链路提交同样短写锁。
                // EN: Brief write locks for the main map and link commit.
                {
                    let mut main = main_index.blocking_write();
                    for plan in &kept {
                        main.insert(plan.id, Arc::clone(&plan.vector));
                    }
                }
                {
                    let mut g = hnsw_write_guard(hnsw);
                    g.retain_live_plan_neighbors(&mut kept);
                    g.apply_insert_plans(kept);
                }
            }
            drop(gate);
            apply_us = apply_us.saturating_add(apply_t0.elapsed().as_micros());
        }
        if plan_apply_log_enabled() {
            let graph_len = hnsw_read_guard(hnsw).len();
            tracing::info!(
                batch = count,
                flushed = kept_total,
                graph = graph_len,
                groups = count.div_ceil(plan_group_size),
                plan_us,
                apply_us,
                "[plan-apply]"
            );
        }
        // 近似边数指标（与 collect_edge_metrics=false 的旧口径一致）。
        // EN: Approximate edge count metric (matches the old collect_edge_metrics=false semantics).
        let delta = kept_total.saturating_mul(m);
        metrics
            .graph_edges_built
            .fetch_add(delta, Ordering::Relaxed);
        metrics
            .graph_edges_level0
            .fetch_add(delta / 2, Ordering::Relaxed);
        metrics
            .graph_edges_upper
            .fetch_add(delta.saturating_sub(delta / 2), Ordering::Relaxed);
        Ok(AsyncBuilderOnceReport {
            flushed: kept_total,
            plan_mode: Some(if parallel_plan {
                HnswBatchPlanMode::Parallel
            } else {
                HnswBatchPlanMode::Serial
            }),
        })
    }

    pub async fn pending_build_count(&self) -> usize {
        self.builder_queue.read().await.len()
    }

    pub async fn indexed_count(&self) -> usize {
        self.main_index.read().await.len()
    }

    /// 当前写入代数（过滤缓存新鲜度校验）。
    /// EN: Current write generation (filter-cache freshness check).
    pub fn write_generation(&self) -> u64 {
        self.write_generation.load(Ordering::Acquire)
    }

    fn bump_write_generation(&self) {
        self.write_generation.fetch_add(1, Ordering::Release);
    }

    /// 图内节点数（诊断用；与 main 计数不一致意味着有点未入图）。
    /// EN: Graph node count (diagnostic; a gap vs main means points missed the graph).
    pub async fn graph_node_count(&self) -> usize {
        hnsw_read_guard(&self.hnsw).len()
    }

    pub async fn hot_buffer_count(&self) -> usize {
        self.hot_buffer.read().await.len()
    }

    pub fn vector_dim(&self) -> usize {
        self.dims
    }

    pub fn distance_metric(&self) -> DistanceMetric {
        self.metric
    }

    /// 共享指针导出：已构建向量只克隆 `Arc`，未构建的 hot/queue 行克隆少量数据。
    /// 用于逻辑备份/迁移的流式导出，避免 10M 规模时把整个向量表再复制一份。
    /// 三层读锁沿用 `export_snapshot_vectors` 的嵌套模式，但**不单独构成**
    /// 「行不会丢失」的保证：构建器的 drain→merge 跨越队列写锁与主表写锁，
    /// 读锁之间仍有窗口。真正的保证来自调用方——先 `drain_pending_builder`
    /// 等在途合并结束，再由集合写锁阻止新入队（见 HTTP 导出处理器）。
    /// EN: Shared-pointer export — built vectors clone only the `Arc`; the small
    /// hot/queue tail is copied. Used by streaming logical backup so a 10M-vector
    /// table is not duplicated in memory. The nested read guards mirror
    /// `export_snapshot_vectors` but do **not** alone guarantee no lost rows: the
    /// builder's drain→merge spans the queue and main write locks, leaving a
    /// window between our read guards. The guarantee comes from the caller —
    /// drain the builder first, then hold the collection write lock so nothing
    /// new is enqueued (see the HTTP export handler).
    pub async fn export_vectors_shared(&self) -> Result<Vec<(VectorId, Arc<Vec<f32>>)>, String> {
        let main = self.main_index.read().await;
        let hot = self.hot_buffer.read().await;
        let queue = self.builder_queue.read().await;
        let mut map: AHashMap<VectorId, Arc<Vec<f32>>> =
            AHashMap::with_capacity(main.len() + hot.len() + queue.len());
        for (id, v) in main.iter() {
            map.insert(*id, Arc::clone(v));
        }
        for (id, v) in hot.iter() {
            map.insert(*id, Arc::new(v.clone()));
        }
        for (id, v) in queue.iter() {
            map.insert(*id, Arc::new(v.clone()));
        }
        drop(main);
        drop(hot);
        drop(queue);
        let mut out: Vec<_> = Vec::with_capacity(map.len());
        out.extend(map);
        out.sort_by_key(|(id, _)| *id);
        for (id, v) in &out {
            if v.len() != self.dims {
                return Err(format!("export: id={id} dim {} != {}", v.len(), self.dims));
            }
        }
        Ok(out)
    }

    /// 导出磁盘快照：main ∪ hot ∪ builder_queue，同 id 以后写为准（queue > hot > main）。
    /// Export for disk snapshot: last-wins per id (queue over hot over main).
    pub async fn export_snapshot_vectors(&self) -> Result<Vec<(VectorId, Vec<f32>)>, String> {
        let mut map: AHashMap<VectorId, Vec<f32>> = AHashMap::new();
        let main = self.main_index.read().await;
        let hot = self.hot_buffer.read().await;
        let queue = self.builder_queue.read().await;
        for (id, v) in main.iter() {
            map.insert(*id, (**v).clone());
        }
        for (id, v) in hot.iter() {
            map.insert(*id, v.clone());
        }
        for (id, v) in queue.iter() {
            map.insert(*id, v.clone());
        }
        drop(main);
        drop(hot);
        drop(queue);
        let mut out: Vec<_> = map.into_iter().collect();
        out.sort_by_key(|(id, _)| *id);
        for (id, v) in &out {
            if v.len() != self.dims {
                return Err(format!(
                    "snapshot export: id={id} dim {} != {}",
                    v.len(),
                    self.dims
                ));
            }
        }
        Ok(out)
    }

    /// 按 id 批量取向量（last-wins：queue > hot > main；墓碑持读锁保证与 remove
    /// 互斥）。只复制请求的 id，供过滤路径构建连续向量缓存。
    /// EN: Bulk fetch by id (last-wins: queue > hot > main; the tombstone read guard
    /// serializes against remove). Only requested ids are cloned.
    pub async fn vectors_by_ids(&self, ids: &[VectorId]) -> Vec<(VectorId, Vec<f32>)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let want: AHashSet<VectorId> = ids.iter().copied().collect();
        let mut found: AHashMap<VectorId, Vec<f32>> = AHashMap::with_capacity(ids.len());
        // 锁序 ts → main → hot → queue（逐个释放）；ts 读锁挡住 remove 的清除。
        // EN: Lock order ts → main → hot → queue, one at a time; ts read blocks remove.
        let ts = self.tombstones.read().await;
        {
            let main = self.main_index.read().await;
            for id in ids {
                if let Some(v) = main.get(id) {
                    found.insert(*id, (**v).clone());
                }
            }
        }
        {
            let hot = self.hot_buffer.read().await;
            for (id, v) in hot.iter() {
                if want.contains(id) {
                    found.insert(*id, v.clone());
                }
            }
        }
        {
            let queue = self.builder_queue.read().await;
            for (id, v) in queue.iter() {
                if want.contains(id) {
                    found.insert(*id, v.clone());
                }
            }
        }
        drop(ts);
        ids.iter()
            .filter_map(|id| found.remove(id).map(|v| (*id, v)))
            .collect()
    }

    /// 从全部写入路径删除单个 id：hot buffer、待构图队列、主存向量表、HNSW 图。
    /// 返回该 id 是否真实存在于任一结构中（幂等：重复删除返回 false）。
    /// EN: Remove one id from every write path — hot buffer, builder queue, main
    /// vector map, HNSW graph. Returns whether the id existed anywhere
    /// (idempotent: a repeated delete returns false).
    ///
    /// 与并发 flush 的竞态通过墓碑收敛：墓碑写锁全程持有——若 worker 已 drain 出含
    /// 该 id 的批次但尚未合图，合图临界区内的第二道墓碑过滤（持合并锁后按最新
    /// 墓碑集再滤）将其丢弃；若合图已完成，则下方的显式摘除负责清出。两种时序
    /// 最终一致。同一 id 之后的重新写入会撤销墓碑（复活语义，见
    /// `insert`/`insert_batch`——它们持同一把锁做「清墓碑 + 写数据」，与本方法
    /// 互斥）。
    /// EN: The race with a concurrent flush converges via the tombstone: the
    /// tombstone write lock is held throughout — if the worker already drained a
    /// batch containing this id but has not merged yet, the second-pass tombstone
    /// filter inside the merge critical section (re-checked under the merge locks)
    /// drops it; if the merge already finished, the explicit purge below takes it
    /// out. A later re-insert of the same id clears the tombstone (resurrection —
    /// insert paths hold the same lock across clear+write, mutually exclusive with
    /// this method).
    ///
    /// 可见性说明：当前仅 houyi-server 的 collection 内存路径使用（collection 标量/
    /// 向量 v1 易失、不进 WAL）；WAL-first 的 benchmark 主索引暴露 delete 需要同步
    /// 设计 WAL 删除记录与重放语义，属独立的持久化工作项。
    /// EN: Used today only by the server's volatile collection path; exposing delete
    /// on the WAL-first benchmark index requires WAL delete records and replay
    /// semantics — a separate persistence work item.
    pub async fn remove(&self, id: VectorId) -> bool {
        // 墓碑写锁贯穿全程，两个目的：
        // 1. 与 insert/insert_batch 互斥——它们持同一把锁做「清墓碑 + 写数据」，
        //    删除与同 id 并发写入的胜者由锁序唯一定义，无第三窗口。
        // 2. 各结构在**独立锁域**内逐个清除（不嵌套）。曾经的嵌套写法
        //    hot→queue→main 与 `export_snapshot_vectors` 的嵌套读 main→hot→queue
        //    构成 ABBA 死锁（remove 持 hot 写等 main，export 持 main 读等 hot）；
        //    独立锁域后本方法不存在多锁嵌套，任何其他路径的嵌套序都不再成环。
        // EN: The tombstone write lock is held across the whole purge, for two
        // reasons: (1) mutual exclusion with insert/insert_batch, which hold the
        // same lock across clear+write, so a delete racing a same-id write has its
        // winner defined by lock order alone; (2) each structure is purged in its
        // own independent scope (no nesting) — the previous nested hot→queue→main
        // writes ABBA-deadlocked against export_snapshot_vectors' nested
        // main→hot→queue reads.
        //
        // 取消安全：若 future 在中途被 drop（如客户端断连），已记录的墓碑仍然
        // 有效——在途批次会被合并期过滤；已完成的清除不回滚，未执行到的结构里
        // 残留的数据可能被再次检索到，但客户端并未收到成功响应（无虚假确认），
        // 重发删除即可收敛。
        // EN: Cancellation — if the future is dropped mid-purge (client
        // disconnect), the recorded tombstone still covers in-flight batches via
        // the merge-time filter; completed purges don't roll back and unpurged
        // remnants may be searchable again, but the client never received a
        // success response (no false acknowledgement) — retrying the delete
        // converges.
        //
        // 提交门（gate）先于墓碑锁：`plan_apply` 的「墓碑过滤 + 提交」持 gate
        // 读锁，保证删除要么在过滤前完成（被滤掉）、要么等提交完成后再清除。
        // 锁序 gate → ts → main → hnsw 与合图路径一致；insert 不取 gate。
        // EN: The commit gate is acquired before the tombstone lock. `plan_apply`
        // holds it shared across filter + commit, so a delete either finishes
        // before the filter (dropped there) or waits until commit and then purges.
        // Lock order gate → ts → main → hnsw, matching the builder; inserts never
        // take the gate.
        let _commit_gate = self.commit_gate.write().await;
        let mut ts = self.tombstones.write().await;
        ts.insert(id);
        let mut removed = false;
        {
            let mut hot = self.hot_buffer.write().await;
            let before = hot.len();
            hot.retain(|(i, _)| *i != id);
            removed |= before != hot.len();
        }
        {
            let mut queue = self.builder_queue.write().await;
            let before = queue.len();
            queue.retain(|(i, _)| *i != id);
            removed |= before != queue.len();
        }
        {
            let mut main = self.main_index.write().await;
            removed |= main.remove(&id).is_some();
        }
        {
            let mut guard = hnsw_write_guard(&self.hnsw);
            removed |= guard.remove_node(id);
        }
        if removed {
            self.bump_write_generation();
        }
        removed
    }

    pub fn builder_metrics(&self) -> BuilderMetricsSnapshot {
        BuilderMetricsSnapshot {
            flush_count: self.metrics.flush_count.load(Ordering::Relaxed),
            total_flushed_items: self.metrics.total_flushed_items.load(Ordering::Relaxed),
            max_pending_queue: self.metrics.max_pending_queue.load(Ordering::Relaxed),
            last_flush_items: self.metrics.last_flush_items.load(Ordering::Relaxed),
            graph_edges_built: self.metrics.graph_edges_built.load(Ordering::Relaxed),
            graph_edges_level0: self.metrics.graph_edges_level0.load(Ordering::Relaxed),
            graph_edges_upper: self.metrics.graph_edges_upper.load(Ordering::Relaxed),
        }
    }

    pub fn neighbors_of(&self, node: VectorId) -> Vec<VectorId> {
        hnsw_read_guard(&self.hnsw).neighbors_layer0(node)
    }

    pub fn upper_neighbors_of(&self, node: VectorId) -> Vec<VectorId> {
        hnsw_read_guard(&self.hnsw).neighbors_at_layer(node, 1)
    }

    pub fn entry_point(&self) -> Option<VectorId> {
        hnsw_read_guard(&self.hnsw).entry_point()
    }

    pub fn node_level(&self, id: VectorId) -> Option<usize> {
        hnsw_read_guard(&self.hnsw).node_level(id)
    }

    fn vector_by_id_ref<'a>(
        id: VectorId,
        hot: &'a [(VectorId, Vec<f32>)],
        main: &'a AHashMap<VectorId, Arc<Vec<f32>>>,
    ) -> Option<&'a [f32]> {
        hot.iter()
            .find(|(x, _)| *x == id)
            .map(|(_, v)| v.as_slice())
            .or_else(|| main.get(&id).map(|v| v.as_slice()))
    }

    /// `k` = 查询 top-k；`ef_override` = 单次查询覆盖 `ef_search`（与 VectorDBBench 一致）。
    /// `filter_set` = 可选过滤集；非 None 时推送入 HNSW 图遍历内部。
    ///
    /// 锁纪律：`hot`/`main` 由调用方已持有的读 guard 传入，本函数**不得**再
    /// `read().await` 自取——tokio RwLock 写者优先，嵌套重入读 + 中间插入的挂起
    /// 写者 = 确定性自死锁（并发回归测试实证：writer 等 hot 写锁，searcher 持
    /// 外层 hot 读锁等内层 hot 读锁，互相等死）。
    /// EN: Lock discipline — callers pass in their already-held read guards; this
    /// function must NOT re-acquire via `read().await`: tokio RwLock is
    /// writer-preferring, so a nested read re-acquire with a pending writer in
    /// between is a deterministic self-deadlock (proven by the concurrency
    /// regression test: writer waits for the hot write lock while the searcher
    /// holds the outer hot read guard and waits on the inner hot read).
    fn bounded_candidate_ids(
        &self,
        hot: &[(VectorId, Vec<f32>)],
        main: &AHashMap<VectorId, Arc<Vec<f32>>>,
        query: &[f32],
        k: usize,
        ef_override: Option<usize>,
        filter_set: Option<&std::collections::HashSet<u64>>,
    ) -> Vec<VectorId> {
        let ef_cfg = ef_override.unwrap_or_else(|| self.hnsw_ef_search());
        let ann_k = k.max(ef_cfg).max(1);
        if self.force_flat_search {
            return hot
                .iter()
                .map(|(id, _)| *id)
                .chain(main.keys().copied())
                .collect();
        }
        if !main.is_empty() {
            let g = hnsw_read_guard(&self.hnsw);
            // 有 filter 则推入图搜索内部，否则走原始 ANN
            // With filter: push into graph traversal; without: original ANN
            let mut candidates = if let Some(fs) = filter_set {
                g.search_with_filter(query, ann_k, fs)
            } else {
                g.search(query, ann_k)
            };
            // 查询侧邻居扩展（实验）：把 top-N 候选的 L0 邻居加入重排集。
            // HNSW 的 ef 束宽可能漏掉真近邻；这些点往往仍是 top 候选的邻居，
            // 扩展后用精确距离重排即可补回，而无需加大构图度数。
            // EN: Query-side neighbor expansion — add the L0 neighbors of the top-N
            // candidates to the rerank set. Recovers neighbors missed by the ef beam
            // without increasing the graph degree.
            let expand = rerank_expand();
            if expand > 0 && filter_set.is_none() {
                let mut seen: AHashSet<VectorId> = candidates.iter().map(|(id, _)| *id).collect();
                let top: Vec<VectorId> =
                    candidates.iter().take(expand).map(|(id, _)| *id).collect();
                for id in top {
                    for nbr in g.neighbors_layer0(id) {
                        if seen.insert(nbr) {
                            candidates.push((nbr, f32::INFINITY));
                        }
                    }
                }
            }
            return candidates.into_iter().map(|(id, _)| id).collect();
        }
        // 主图尚未构建时回退到 hot buffer 的 id。
        // EN: fallback to hot buffer ids when main graph is not built yet.
        hot.iter().map(|(id, _)| *id).take(ann_k).collect()
    }

    /// 启动异步构建 worker：支持阈值、周期或混合策略。
    /// Start async builder worker with threshold/interval/hybrid policy.
    pub fn start_async_builder_worker(
        self: &Arc<Self>,
        batch_threshold: usize,
        tick_ms: u64,
        flush_policy: FlushPolicy,
    ) -> BuilderWorker {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let index = Arc::clone(self);
        let metrics = Arc::clone(&self.metrics);

        let handle = tokio::spawn(async move {
            let mut ticker = time::interval(Duration::from_millis(tick_ms));
            loop {
                ticker.tick().await;
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }

                let pending = index.pending_build_count().await;
                metrics
                    .max_pending_queue
                    .fetch_max(pending, Ordering::Relaxed);

                let should_flush = match flush_policy {
                    FlushPolicy::ThresholdOnly => pending >= batch_threshold,
                    FlushPolicy::IntervalOnly => pending > 0,
                    FlushPolicy::Hybrid => pending >= batch_threshold || pending > 0,
                };
                if should_flush {
                    // 一次唤醒内连续执行多轮 `run_async_builder_once`，直到本轮 queue 上无新批或已排空。
                    // 不依赖 VectorDBBench 用 HTTP 逐次 POST `/v1/optimize`；客户端可只轮询 `GET /v1/index/status`。
                    // EN: Multi-burst per tick; build completes in-server; bench waits on `pending_count` only.
                    const MAX_BURSTS: usize = 1_000_000;
                    for _ in 0..MAX_BURSTS {
                        let rep = index.run_async_builder_once().await;
                        if rep.flushed == 0 {
                            break;
                        }
                        metrics.flush_count.fetch_add(1, Ordering::Relaxed);
                        metrics
                            .total_flushed_items
                            .fetch_add(rep.flushed, Ordering::Relaxed);
                        metrics
                            .last_flush_items
                            .store(rep.flushed, Ordering::Relaxed);
                        if stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                }
            }
        });

        BuilderWorker {
            stop,
            handle: Some(handle),
        }
    }
}

impl Default for HotBufferIndex {
    fn default() -> Self {
        Self::new(1)
    }
}

#[async_trait]
impl VectorIndex for HotBufferIndex {
    async fn insert(&self, id: VectorId, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dims {
            return Err(format!(
                "dims mismatch: expect {}, got {}",
                self.dims,
                vector.len()
            ));
        }
        // 复活语义 + 与 remove 的原子性：持墓碑写锁贯穿「清墓碑 + 写 hot」。
        // 分锁实现会在两步之间留出窗口——并发 remove(id) 整体落进窗口时会重新
        // 记下墓碑而清不到尚未 push 的数据，随后 hot 里的 id 永久可被检索
        // （hot 不经 builder 队列，合并期的墓碑过滤永远看不到它）。持锁贯穿后
        // remove 与 insert 由同一把墓碑锁互斥，胜者由锁序唯一定义。
        // 锁序 ts → hot 与 remove() 一致，无逆序。
        // EN: Resurrection + atomicity vs remove — hold the tombstone write lock
        // across clear + hot push. Separate scopes left a window where a concurrent
        // remove(id) re-tombstones yet purges nothing (data not yet pushed), and
        // the hot entry (which never drains through the merge-time filter) stays
        // searchable indefinitely. Holding the lock serializes insert/remove on the
        // same mutex; winner is defined by lock order. Lock order ts → hot matches
        // remove().
        let mut ts = self.tombstones.write().await;
        ts.remove(&id);
        let mut hot = self.hot_buffer.write().await;
        hot.retain(|(i, _)| *i != id);
        hot.push((id, vector));
        self.bump_write_generation();
        Ok(())
    }

    async fn insert_batch(&self, vectors: Vec<(VectorId, Vec<f32>)>) -> Result<(), String> {
        // 中文：同批 / 跨批重试时同一 id 以后写为准；覆盖待构图队列中旧条目，避免重复主键。
        // EN: Last-wins per id within batch; drop pending queue rows with same ids (retry-safe).
        //
        // 白皮书 §3.2 / `houyidb.md`：`insert_batch` 使用 Rayon 并行校验维度并对去重结果排序；
        // 幂等合并（last-wins）仍须顺序扫描 `vectors` 以保证与单线程语义一致。
        // Whitepaper: Rayon parallelizes validation + sort; sequential HashMap merge keeps last-wins deterministic.
        let dims = self.dims;
        vectors.par_iter().try_for_each(|(id, v)| {
            if v.len() != dims {
                return Err(format!(
                    "dims mismatch for id={id}: expect {dims}, got {}",
                    v.len()
                ));
            }
            Ok(())
        })?;

        let mut last: AHashMap<VectorId, Vec<f32>> = AHashMap::new();
        for (id, v) in vectors {
            last.insert(id, v);
        }
        let mut deduped: Vec<_> = last.into_iter().collect();
        deduped.par_sort_unstable_by_key(|(id, _)| *id);
        let new_ids: AHashSet<VectorId> = deduped.iter().map(|(id, _)| *id).collect();
        // 复活语义 + 与 remove 的原子性（与单条 insert 同口径、同根因）：
        // 持墓碑写锁贯穿「清墓碑 + 入队」。分锁实现下，并发 remove(id) 可在两步
        // 之间重新记下墓碑而清不到尚未入队的 id——已确认的写入随后被合并期的
        // 墓碑过滤静默丢弃（id 既不在 hot 也不在 main，客户端无任何报错）。
        // EN: Resurrection + atomicity vs remove (same contract and root cause as
        // single-row insert): hold the tombstone write lock across clear + enqueue.
        // With separate scopes, a concurrent remove(id) could re-tombstone between
        // the steps while purging nothing (id not yet queued) — the acknowledged
        // write is then silently dropped by the merge-time tombstone filter (id
        // ends up in neither hot nor main, with no error to the client).
        let mut ts = self.tombstones.write().await;
        for id in &new_ids {
            ts.remove(id);
        }
        let mut queue = self.builder_queue.write().await;
        queue.retain(|(qid, _)| !new_ids.contains(qid));
        queue.extend(deduped);
        self.bump_write_generation();
        Ok(())
    }

    async fn search(
        &self,
        _query: &[f32],
        k: usize,
        filter_mask: Option<&[VectorId]>,
    ) -> Vec<VectorId> {
        self.search_with_ef_override(_query, k, filter_mask, None)
            .await
    }

    async fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        filter_mask: Option<&[VectorId]>,
        ef_search_override: Option<usize>,
    ) -> Vec<VectorId> {
        self.search_with_ef_override(query, k, filter_mask, ef_search_override)
            .await
    }

    fn optimize(&self) {
        let pruned = hnsw_write_guard(&self.hnsw).optimize_compact();
        if pruned > 0 {
            let metrics = &self.metrics;
            let delta = pruned as isize;
            let current = metrics.graph_edges_built.load(Ordering::Relaxed) as isize;
            if current >= delta {
                metrics
                    .graph_edges_built
                    .store((current - delta) as usize, Ordering::Relaxed);
            }
        }
    }
}

impl HotBufferIndex {
    /// 与 [`VectorIndex::search`] 相同，但允许单次请求覆盖 `ef_search`（HTTP benchmark）。
    /// EN: Same as [`VectorIndex::search`] but allows a per-request `ef_search` override.
    pub async fn search_with_ef_override(
        &self,
        query: &[f32],
        k: usize,
        filter_mask: Option<&[VectorId]>,
        ef_search_override: Option<usize>,
    ) -> Vec<VectorId> {
        // 唯一排序/截断逻辑位于 scored 变体；id-only 链路只剥离距离，
        // 避免两条链路在候选补全/mask 策略上漂移。
        // EN: Single source of truth for candidate scoring/sort/truncate lives in the
        // scored variant; the id-only path just strips distances, preventing drift.
        self.search_scored_with_ef_override(query, k, filter_mask, ef_search_override)
            .await
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// scored 检索：返回 `(向量 id, 距离)`，按距离**升序**排列（距离相等按 id 升序），
    /// 至多 `k` 条；允许单次请求覆盖 `ef_search`（HTTP benchmark / 边缘 score 换算）。
    /// EN: Scored search — returns `(vector id, distance)` pairs sorted by distance
    /// ascending (id ascending on ties), at most `k` pairs; `ef_search` may be
    /// overridden per request. Distance semantics follow the index metric:
    /// - [`DistanceMetric::Cosine`] → cosine distance `1 - cosine_similarity`
    ///   （由 [`cosine_distance`] 内部按范数归一化计算，存储/查询向量无需预归一化，
    ///   完全相同的向量距离为 0）；
    /// - [`DistanceMetric::L2`] → L2 距离平方（不开方）；
    /// - [`DistanceMetric::InnerProduct`] → 负点积 `-a·b`（点积越大距离越小）；
    /// - [`DistanceMetric::L1`] → 曼哈顿距离 `Σ|a_i - b_i|`。
    ///
    /// 四种语义均为「越小越近」；服务端在响应边缘把 IP 再取负回正为点积分数。
    /// EN: All four keep "smaller = closer"; the server flips the IP sign back
    /// to a raw dot-product score at the response edge.
    ///
    /// 为什么同时保留 scored 与 id-only 两个入口：内核内部链路（HNSW 图遍历、
    /// sanity check）只需要 id，而对外生产 HTTP 检索需要返回相似度分数；后端统一保持
    /// 「距离、越小越近」语义，距离→相似度（`1 - distance`）的换算只在 HTTP 边缘做一次，
    /// 避免内核层出现距离/相似度双轨语义。
    /// EN: Why both entry points exist — internal graph paths need ids only, while the
    /// external HTTP API returns similarity scores. Backends stay in distance
    /// semantics (smaller = closer); the distance→similarity conversion happens once at
    /// the HTTP edge, avoiding dual semantics inside the kernel.
    pub async fn search_scored_with_ef_override(
        &self,
        _query: &[f32],
        k: usize,
        filter_mask: Option<&[VectorId]>,
        ef_search_override: Option<usize>,
    ) -> Vec<(VectorId, f32)> {
        let hot = self.hot_buffer.read().await;
        let main = self.main_index.read().await;

        // —— 智能策略选择：基于 mask 大小 ——
        // (1) 小 mask (≤50K)：rayon 全量并行暴力 + NEON SIMD（已验证 306.9 QPS）
        // (2) 大 mask (>50K)：filter 推入 HNSW 图遍历内部（图内过滤）
        // (3) 无 mask：原有 ANN 逻辑
        //
        // EN: Intelligent strategy based on mask size:
        // (1) Small mask (≤50K): rayon parallel brute-force + NEON SIMD (validated 306.9 QPS)
        // (2) Large mask (>50K): filter pushed into HNSW graph traversal (in-graph filtering)
        // (3) No mask: original ANN logic
        let is_small_mask = filter_mask.map(|m| m.len() <= 50_000).unwrap_or(false);

        let source_ids: Vec<VectorId> = if is_small_mask {
            // 小 mask：rayon 并行暴力（提取引用绕过 tokio RwLockSend 限制）。
            //
            // mask 的语义是**集合**（「只在这些 id 里搜」），调用方传入重复 id 时
            // 原样当候选会让同一 id 各占一个 top-k 名额——`k=2` 返回 `[7, 7]`，
            // 既浪费名额又污染响应。此处按首次出现去重，与 >50K 分支的 `seen`
            // 同口径（P4.5.3 检索 oracle 的掩码属性抓出）。
            // EN: The mask is a SET; a duplicated id must not consume two top-k
            // slots (`k=2` returned `[7, 7]`). Order-preserving dedupe, mirroring
            // the large-mask branch's `seen` (found by the P4.5.3 retrieval oracle).
            let mask = filter_mask.unwrap();
            let mut seen: std::collections::HashSet<VectorId> =
                std::collections::HashSet::with_capacity(mask.len());
            mask.iter().copied().filter(|id| seen.insert(*id)).collect()
        } else if let Some(mask) = filter_mask {
            // 大 mask：全图 HNSW 候选 + HashSet filter + mask 补充
            // 注意：不推入 HNSW 图遍历内部（子图边断裂导致 recall 骤降）；
            // 改为全图 ANN 搜索后再 filter，不足时从 mask 补充
            // EN: Large mask: full-graph HNSW candidates + HashSet filter + mask fill-in.
            // EN: Note: do not push into HNSW traversal (subgraph edge breaks collapse recall);
            // EN: instead run full-graph ANN, then filter, and fill shortages from the mask.
            let mask_set: std::collections::HashSet<u64> = mask.iter().copied().collect();
            let candidates =
                self.bounded_candidate_ids(&hot, &main, _query, k, ef_search_override, None);
            let mut ids: Vec<VectorId> = candidates
                .iter()
                .filter(|id| mask_set.contains(id))
                .copied()
                .collect();
            let mut seen: std::collections::HashSet<u64> = ids.iter().copied().collect();
            for id in mask.iter() {
                if seen.len() >= k.saturating_mul(10).max(k) {
                    break;
                }
                if seen.insert(*id) {
                    ids.push(*id);
                }
            }
            ids
        } else {
            // 无 filter：原始 ANN + fallback
            // EN: No filter: raw ANN + fallback.
            let mut ids =
                self.bounded_candidate_ids(&hot, &main, _query, k, ef_search_override, None);
            if ids.is_empty() {
                ids = hot
                    .iter()
                    .map(|(id, _)| *id)
                    .chain(main.keys().copied())
                    .take(k.saturating_mul(8).max(k))
                    .collect();
            } else if ids.len() < k {
                for id in hot.iter().map(|(id, _)| *id).chain(main.keys().copied()) {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                    if ids.len() >= k {
                        break;
                    }
                }
            }
            ids
        };

        // 距离计算 + 排序。
        // 小 mask: 先串行提取引用（10K 次 HashMap 查找），再对连续的
        // `(id, &[f32])` 做 par_iter 距离计算。
        // 不要合并成「查表+距离」的单层 par_iter：实测 ef128 645 QPS vs
        // 分两阶段 729 QPS——并行随机查表破坏局部性并放大 rayon 调度开销。
        // EN: Small mask — serial ref extraction, then par_iter distance over the
        // contiguous ref Vec. Do NOT fuse lookup+distance into one par_iter:
        // measured 645 vs 729 QPS (ef128); parallel random lookups hurt locality
        // and add rayon overhead.
        let mut scored: Vec<(VectorId, f32)> = if is_small_mask {
            let mask_vecs: Vec<(VectorId, &[f32])> = source_ids
                .iter()
                .filter_map(|id| Self::vector_by_id_ref(*id, &hot, &main).map(|v| (*id, v)))
                .collect();
            use rayon::prelude::*;
            mask_vecs
                .par_iter()
                .map(|(id, v)| (*id, self.distance(_query, v)))
                .collect()
        } else if source_ids.len() >= 512 {
            // 邻居扩展后候选可达数千：与 small-mask 同套路，先串行取引用，
            // 再 par_iter 算距离。无扩展时候选通常 <512，仍走下面的串行路径。
            // EN: After neighbor expansion the candidate set can reach thousands;
            // extract refs serially, then par_iter distances. Without expansion the
            // set is usually <512 and stays on the serial path.
            let refs: Vec<(VectorId, &[f32])> = source_ids
                .iter()
                .filter_map(|id| Self::vector_by_id_ref(*id, &hot, &main).map(|v| (*id, v)))
                .collect();
            use rayon::prelude::*;
            refs.par_iter()
                .map(|(id, v)| (*id, self.distance(_query, v)))
                .collect()
        } else {
            source_ids
                .into_iter()
                .filter_map(|id| {
                    Self::vector_by_id_ref(id, &hot, &main).map(|v| (id, self.distance(_query, v)))
                })
                .collect()
        };
        // 小 mask 路径会对全部命中做距离计算（1% of 1M ≈ 10K）。只排序并保留
        // top-k 即可：`select_nth_unstable_by` 平均 O(n)，再对 k 个元素排序，
        // 避免 10K 全排序。语义与全排序后 truncate 一致（total_cmp + id 兜底）。
        // EN: The small-mask path scores every hit (~10K at 1% of 1M). Select the
        // top-k in O(n) via select_nth_unstable_by, then sort only k, instead of
        // sorting all 10K. Same ordering as the previous full sort + truncate.
        let cmp =
            |a: &(VectorId, f32), b: &(VectorId, f32)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
        if scored.len() > k {
            if k == 0 {
                scored.clear();
            } else {
                scored.select_nth_unstable_by(k - 1, cmp);
                scored.truncate(k);
            }
        }
        scored.sort_by(cmp);
        self.maybe_run_sanity_check(_query, k, &scored, &hot, &main);
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::{
        l2_distance, search_flat_contiguous, DistanceMetric, FlushPolicy, HotBufferIndex,
        LockFreeNeighborStore, NeighborStore, VectorIndex,
    };
    use crate::distance;
    use houyi_core::config::BuildConfig;
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};

    async fn drain_builder_queue(index: &HotBufferIndex) {
        while index.pending_build_count().await > 0 {
            let _ = index.run_async_builder_once().await;
        }
    }

    /// 度量枚举的配置/快照字节契约：别名解析、字符串回写与快照字节往返一致。
    /// EN: Metric enum config/snapshot-byte contract: alias parsing, string
    /// roundtrip, and snapshot-byte roundtrip all agree.
    #[test]
    fn distance_metric_parse_and_snapshot_bytes_roundtrip() {
        let all = [
            DistanceMetric::L2,
            DistanceMetric::Cosine,
            DistanceMetric::InnerProduct,
            DistanceMetric::L1,
        ];
        for m in all {
            assert_eq!(DistanceMetric::parse_str(m.as_str()), Some(m));
            assert_eq!(
                DistanceMetric::from_snapshot_byte(m.to_snapshot_byte()),
                Some(m)
            );
        }
        // 常见迁移别名（Milvus `IP` / DM `MANHATTAN` / 欧氏）。
        // EN: Common migration aliases (Milvus `IP`, DM `MANHATTAN`, Euclidean).
        assert_eq!(
            DistanceMetric::parse_str("IP"),
            Some(DistanceMetric::InnerProduct)
        );
        assert_eq!(
            DistanceMetric::parse_str("dot"),
            Some(DistanceMetric::InnerProduct)
        );
        assert_eq!(
            DistanceMetric::parse_str("EUCLIDEAN"),
            Some(DistanceMetric::L2)
        );
        assert_eq!(
            DistanceMetric::parse_str("manhattan"),
            Some(DistanceMetric::L1)
        );
        assert_eq!(DistanceMetric::parse_str("hamming"), None);
        assert_eq!(DistanceMetric::from_snapshot_byte(255), None);
    }

    /// IP/L1 的图检索 top-k 与暴力排序一致（HotBuffer 热路径同一 `distance` 分派）。
    /// EN: Graph top-k under IP/L1 equals brute-force ranking through the same
    /// HotBuffer `distance` dispatch.
    #[tokio::test]
    async fn hotbuffer_ip_and_l1_topk_match_brute_force() {
        for metric in [DistanceMetric::InnerProduct, DistanceMetric::L1] {
            let dims = 8usize;
            let index = HotBufferIndex::new_with_metric(dims, metric);
            // 固定 HNSW 参数：`new_with_metric` 从环境变量推导 m/ef，默认值
            // 随 `HOUYI_VECTOR_*` 变化，低 ef_search 会让 beam 漏候选，
            // 「top-k 与暴力一致」变成偶发失败（OCR #18）。
            // EN: Pin HNSW params: `new_with_metric` derives m/ef from env vars,
            // so defaults shift with `HOUYI_VECTOR_*`; a low ef_search makes the
            // beam miss candidates and turns the exact top-k assertion flaky
            // (OCR #18).
            index.set_hnsw_m(16).unwrap();
            index.set_hnsw_ef_construction(200).unwrap();
            index.set_hnsw_ef_search(128).unwrap();
            let mut rows = Vec::new();
            for id in 0..64u64 {
                let v: Vec<f32> = (0..dims)
                    .map(|i| ((id as usize * 7 + i * 13) % 23) as f32 * 0.05 - 0.5)
                    .collect();
                rows.push((id, v));
            }
            index.insert_batch(rows.clone()).await.unwrap();
            drain_builder_queue(&index).await;
            let query: Vec<f32> = (0..dims).map(|i| (i % 5) as f32 * 0.1).collect();
            let got = index
                .search_scored_with_ef_override(&query, 5, None, None)
                .await;
            let mut scored: Vec<(u64, f32)> = rows
                .iter()
                .map(|(id, v)| {
                    let d = match metric {
                        DistanceMetric::InnerProduct => distance::inner_product_distance(&query, v),
                        DistanceMetric::L1 => distance::l1_distance(&query, v),
                        _ => unreachable!(),
                    };
                    (*id, d)
                })
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let want: Vec<u64> = scored.iter().take(5).map(|(id, _)| *id).collect();
            let got_ids: Vec<u64> = got.iter().take(5).map(|(id, _)| *id).collect();
            assert_eq!(got_ids, want, "metric={metric:?}");
        }
    }

    /// 写入代数：insert_batch 与 builder 合图都必须推进，供过滤缓存失效。
    /// EN: Write generation advances on insert_batch and again after the builder merge.
    #[tokio::test]
    async fn write_generation_advances_on_write_and_merge() {
        let index = HotBufferIndex::new_with_metric(4, DistanceMetric::L2);
        let g0 = index.write_generation();
        index
            .insert_batch(vec![(1, vec![0.1, 0.2, 0.3, 0.4])])
            .await
            .unwrap();
        let g1 = index.write_generation();
        assert!(g1 > g0, "insert_batch must bump the generation");
        drain_builder_queue(&index).await;
        assert!(
            index.write_generation() > g1,
            "builder merge must bump the generation"
        );
    }

    /// 连续缓存扫描 == 逐点暴力 top-k（L2 与 Cosine 两条口径）。
    /// EN: Contiguous-cache scan equals per-point brute-force top-k for both metrics.
    #[test]
    fn flat_contiguous_matches_brute_force() {
        let dims = 8usize;
        let n = 100u64;
        let ids: Vec<u64> = (0..n).collect();
        let data: Vec<f32> = (0..(n as usize * dims))
            .map(|i| ((i * 17) % 97) as f32 / 97.0)
            .collect();
        let query: Vec<f32> = (0..dims).map(|j| (j as f32 * 0.13).sin()).collect();
        let k = 10usize;

        let got = search_flat_contiguous(&query, &ids, &data, dims, k, DistanceMetric::L2);
        let mut want: Vec<(u64, f32)> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                (
                    *id,
                    distance::l2_squared(&query, &data[i * dims..(i + 1) * dims]),
                )
            })
            .collect();
        want.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        want.truncate(k);
        assert_eq!(got, want);

        let mut ndata = data;
        for chunk in ndata.chunks_mut(dims) {
            distance::normalize_in_place(chunk);
        }
        let mut nq = query;
        distance::normalize_in_place(&mut nq);
        let got = search_flat_contiguous(&nq, &ids, &ndata, dims, k, DistanceMetric::Cosine);
        let mut want: Vec<(u64, f32)> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                (
                    *id,
                    distance::cosine_distance_prenormalized(&nq, &ndata[i * dims..(i + 1) * dims]),
                )
            })
            .collect();
        want.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        want.truncate(k);
        assert_eq!(got, want);
    }

    /// 短锁提交不改变图内容：同一批数据分别以 commit_chunk=0 / 8 / 64 构建，
    /// 查询结果应逐 id 一致（插入顺序不变，只有写锁作用域变化）。
    /// EN: Short-lock commit must not change graph content; same data and insertion
    /// order, only the write-lock scopes differ -> identical query results.
    #[tokio::test]
    async fn commit_chunk_preserves_graph_results() {
        async fn build(chunk: usize) -> Vec<Vec<u64>> {
            let dims = 16usize;
            let mut index = HotBufferIndex::new_with_metric(dims, DistanceMetric::L2);
            index.set_hnsw_m(16).unwrap();
            index.set_hnsw_ef_construction(128).unwrap();
            index.set_hnsw_ef_search(64).unwrap();
            let cfg = BuildConfig {
                commit_chunk: chunk,
                max_build_batch: 64,
                parallel_batch_build: false,
                ..BuildConfig::default()
            };
            index.set_build_config(cfg);
            let batch: Vec<(u64, Vec<f32>)> = (0..200_u64)
                .map(|id| {
                    (
                        id,
                        (0..dims)
                            .map(|j| ((id as usize * 13 + j * 7) % 200) as f32 / 200.0)
                            .collect(),
                    )
                })
                .collect();
            index.insert_batch(batch).await.unwrap();
            drain_builder_queue(&index).await;
            let mut out = Vec::new();
            for q in 0..10 {
                let query: Vec<f32> = (0..dims)
                    .map(|j| ((q * 17 + j * 11) % 200) as f32 / 200.0)
                    .collect();
                out.push(index.search(&query, 10, None).await);
            }
            out
        }
        assert_eq!(build(0).await, build(8).await);
        assert_eq!(build(0).await, build(64).await);
    }

    /// TDD：构图完成后，`k` 应等于返回 id 数（数据量足够时）；覆盖「HNSW 只截断到 ef_search 而非 k」类回归。
    /// EN: TDD: after building, `k` should equal the returned id count (with enough data).
    #[tokio::test]
    async fn search_returns_k_ids_after_hnsw_graph_built() {
        let dims = 16usize;
        let index = HotBufferIndex::new_with_metric(dims, DistanceMetric::L2);
        index.set_hnsw_m(16).unwrap();
        index.set_hnsw_ef_construction(128).unwrap();
        index.set_hnsw_ef_search(64).unwrap();
        let batch: Vec<(u64, Vec<f32>)> = (0..200_u64)
            .map(|id| {
                (
                    id,
                    (0..dims)
                        .map(|j| ((id as usize * 13 + j * 7) % 200) as f32 / 200.0)
                        .collect(),
                )
            })
            .collect();
        index.insert_batch(batch).await.unwrap();
        drain_builder_queue(&index).await;
        assert_eq!(index.indexed_count().await, 200);
        let q = vec![0.42_f32; dims];
        let ids = index.search(&q, 10, None).await;
        assert_eq!(ids.len(), 10, "top-k must return k ids when N>=k");
    }

    /// TDD：全量入图后 ANN recall@k 应显著高于随机（防距离/截断 bug）。
    /// EN: TDD: after full ingestion, ANN recall@k should be well above random (guards distance bugs).
    #[tokio::test]
    async fn ann_recall_non_trivial_after_hnsw_flush() {
        let dims = 32usize;
        let n = 500usize;
        let k = 10usize;
        let index = HotBufferIndex::new_with_metric(dims, DistanceMetric::L2);
        index.set_hnsw_m(16).unwrap();
        index.set_hnsw_ef_construction(200).unwrap();
        index.set_hnsw_ef_search(128).unwrap();
        let data: Vec<(u64, Vec<f32>)> = (0..n as u64)
            .map(|id| {
                (
                    id,
                    (0..dims)
                        .map(|j| ((id as usize * 17 + j * 11) % 500) as f32 / 500.0)
                        .collect(),
                )
            })
            .collect();
        index.insert_batch(data.clone()).await.unwrap();
        drain_builder_queue(&index).await;
        let query = data[100].1.clone();
        let mut brute: Vec<(u64, f32)> = data
            .iter()
            .map(|(id, v)| (*id, l2_distance(&query, v)))
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let truth: HashSet<u64> = brute.into_iter().take(k).map(|(id, _)| id).collect();
        let pred = index.search(&query, k, None).await;
        let hit = pred.iter().filter(|id| truth.contains(id)).count();
        let recall = hit as f32 / k as f32;
        assert!(
            recall >= 0.35,
            "expected non-trivial recall@{}, got {:.2}",
            k,
            recall
        );
    }

    #[tokio::test]
    async fn insert_is_visible_immediately_via_hot_buffer() {
        let index = HotBufferIndex::new(4);
        index.insert(1, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();

        let ids = index.search(&[0.1, 0.2, 0.3, 0.4], 10, None).await;
        assert_eq!(ids, vec![1]);
    }

    #[tokio::test]
    async fn batch_insert_is_queued_until_builder_flush() {
        let index = HotBufferIndex::new(4);
        index
            .insert_batch(vec![
                (11, vec![1.0, 1.0, 1.0, 1.0]),
                (12, vec![2.0, 2.0, 2.0, 2.0]),
            ])
            .await
            .unwrap();

        assert_eq!(index.pending_build_count().await, 2);
        let before = index.search(&[0.0; 4], 10, None).await;
        assert!(before.is_empty());

        let rep = index.run_async_builder_once().await;
        assert_eq!(rep.flushed, 2);

        let after = index.search(&[0.0; 4], 10, None).await;
        assert_eq!(after, vec![11, 12]);
        let n11 = index.neighbors_of(11);
        let n12 = index.neighbors_of(12);
        assert_eq!(n11, vec![12]);
        assert_eq!(n12, vec![11]);
    }

    #[tokio::test]
    async fn batch_insert_rejects_invalid_dims() {
        let index = HotBufferIndex::new(4);
        let err = index
            .insert_batch(vec![(42, vec![1.0, 2.0, 3.0])])
            .await
            .expect_err("invalid dims should fail");

        assert!(err.contains("dims mismatch"));
    }

    #[tokio::test]
    async fn insert_batch_last_wins_same_id_within_batch() {
        let index = HotBufferIndex::new_with_metric(4, DistanceMetric::L2);
        index
            .insert_batch(vec![
                (1, vec![1.0, 0.0, 0.0, 0.0]),
                (1, vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .await
            .unwrap();
        drain_builder_queue(&index).await;
        let q = vec![0.0_f32, 1.0, 0.0, 0.0];
        let found = index.search(&q, 1, None).await;
        assert_eq!(found, vec![1]);
        assert_eq!(index.indexed_count().await, 1);
    }

    #[tokio::test]
    async fn insert_batch_reupsert_replaces_after_flush() {
        let index = HotBufferIndex::new_with_metric(4, DistanceMetric::L2);
        index
            .insert_batch(vec![(5, vec![0.0, 0.0, 0.0, 1.0])])
            .await
            .unwrap();
        drain_builder_queue(&index).await;
        index
            .insert_batch(vec![(5, vec![0.0, 0.0, 1.0, 0.0])])
            .await
            .unwrap();
        drain_builder_queue(&index).await;
        let q = vec![0.0_f32, 0.0, 1.0, 0.0];
        let found = index.search(&q, 1, None).await;
        assert_eq!(found, vec![5]);
        assert_eq!(index.indexed_count().await, 1);
    }

    /// remove 覆盖全部写入路径：hot buffer 条目、待构图队列条目、已合图的主表+图节点。
    /// EN: remove() covers every write path — hot entry, queued entry, merged main+graph.
    #[tokio::test]
    async fn remove_covers_hot_queue_and_graph_paths() {
        let index = HotBufferIndex::new_with_metric(4, DistanceMetric::Cosine);
        // id=1 走单条 insert（hot buffer），id=2/3 走批量（builder_queue），
        // id=4 先合图进主索引。
        // EN: id=1 uses single insert (hot buffer); id=2/3 use batch (builder_queue);
        // EN: id=4 is merged into the main index first.
        index.insert(1, vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        index
            .insert_batch(vec![
                (2, vec![0.0, 1.0, 0.0, 0.0]),
                (3, vec![0.0, 0.0, 1.0, 0.0]),
                (4, vec![0.0, 0.0, 0.0, 1.0]),
            ])
            .await
            .unwrap();
        drain_builder_queue(&index).await;
        // id=2 重新入队（不排空），制造「待构图」状态。
        // EN: Re-enqueue id=2 without draining to create a pending-build state.
        index
            .insert_batch(vec![(2, vec![0.0, 1.0, 0.0, 0.0])])
            .await
            .unwrap();

        assert!(index.remove(1).await, "hot entry existed");
        assert!(index.remove(2).await, "queued entry existed");
        assert!(index.remove(4).await, "merged entry existed");
        assert!(!index.remove(99).await, "unknown id reports false");

        assert_eq!(index.hot_buffer_count().await, 0);
        assert_eq!(index.pending_build_count().await, 0);
        assert_eq!(index.indexed_count().await, 1, "only id=3 survives");

        // 已删 id 对检索不可见：分别用其自身向量精确查询。
        // EN: Deleted ids are invisible to search: query exactly with their own vectors.
        for (gone, q) in [
            (1_u64, vec![1.0_f32, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (4, vec![0.0, 0.0, 0.0, 1.0]),
        ] {
            let found = index.search(&q, 3, None).await;
            assert!(
                !found.contains(&gone),
                "removed id {gone} must be invisible, got {found:?}"
            );
        }
        let found = index.search(&[0.0, 0.0, 1.0, 0.0], 3, None).await;
        assert_eq!(found.first(), Some(&3));
    }

    /// remove 与并发 flush 的收敛：批量在队列中即被删除，排空后不得进图。
    /// EN: remove-vs-flush convergence — ids deleted while still queued never merge.
    #[tokio::test]
    async fn remove_queued_id_never_enters_graph_after_flush() {
        let index = HotBufferIndex::new_with_metric(4, DistanceMetric::Cosine);
        index
            .insert_batch(vec![
                (1, vec![1.0, 0.0, 0.0, 0.0]),
                (2, vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .await
            .unwrap();
        assert!(index.remove(2).await);
        drain_builder_queue(&index).await;
        assert_eq!(index.indexed_count().await, 1);
        let found = index.search(&[0.0, 1.0, 0.0, 0.0], 2, None).await;
        assert!(
            !found.contains(&2),
            "tombstoned id must not merge: {found:?}"
        );
    }

    /// 复活语义：删除后同一 id 重新写入必须撤销墓碑并重新可检索。
    /// EN: Resurrection — re-inserting a deleted id clears the tombstone.
    #[tokio::test]
    async fn reinsert_after_remove_resurrects_id() {
        let index = HotBufferIndex::new_with_metric(4, DistanceMetric::Cosine);
        index
            .insert_batch(vec![(7, vec![1.0, 0.0, 0.0, 0.0])])
            .await
            .unwrap();
        drain_builder_queue(&index).await;
        assert!(index.remove(7).await);
        assert!(index
            .search(&[1.0, 0.0, 0.0, 0.0], 1, None)
            .await
            .is_empty());

        index
            .insert_batch(vec![(7, vec![0.0, 1.0, 0.0, 0.0])])
            .await
            .unwrap();
        drain_builder_queue(&index).await;
        let found = index.search(&[0.0, 1.0, 0.0, 0.0], 1, None).await;
        assert_eq!(found, vec![7], "resurrected id must be searchable again");
    }

    /// 单条 insert 路径同样遵循复活语义（hot buffer 写入撤销墓碑）。
    /// EN: The single-row insert path also resurrects (hot-buffer write clears tombstone).
    #[tokio::test]
    async fn single_insert_after_remove_resurrects_id() {
        let index = HotBufferIndex::new_with_metric(4, DistanceMetric::Cosine);
        index.insert(3, vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        assert!(index.remove(3).await);
        index.insert(3, vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        let found = index.search(&[1.0, 0.0, 0.0, 0.0], 1, None).await;
        assert_eq!(found, vec![3]);
    }

    /// 并发回归：remove 曾经嵌套持锁 hot→queue→main（写），与
    /// export_snapshot_vectors 的嵌套读 main→hot→queue 锁序相反——并发即 ABBA
    /// 死锁。修复后 remove 全程持墓碑写锁、各结构在独立锁域内清除（无嵌套）。
    /// 测试用 timeout 包裹：死锁复发时表现为超时失败，而非挂死 CI。
    /// EN: Concurrency regression — remove() used to nest write guards
    /// hot→queue→main, the opposite order of export_snapshot_vectors' nested
    /// main→hot→queue reads: an ABBA deadlock. Fixed by holding the tombstone
    /// lock across independent per-structure scopes (no nesting). The timeout
    /// wrapper turns a deadlock relapse into a test failure instead of a hung CI.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn remove_and_snapshot_export_do_not_deadlock() {
        let index = Arc::new(HotBufferIndex::new_with_metric(4, DistanceMetric::Cosine));
        let writer = {
            let index = Arc::clone(&index);
            tokio::spawn(async move {
                for i in 0..300_u64 {
                    let id = i % 16;
                    if i % 2 == 0 {
                        index.insert(id, vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
                    } else {
                        index
                            .insert_batch(vec![(id, vec![0.0, 1.0, 0.0, 0.0])])
                            .await
                            .unwrap();
                    }
                    if i % 3 == 0 {
                        index.remove(id).await;
                    }
                }
            })
        };
        let exporter = {
            let index = Arc::clone(&index);
            tokio::spawn(async move {
                for _ in 0..300 {
                    index.export_snapshot_vectors().await.unwrap();
                }
            })
        };
        tokio::time::timeout(Duration::from_secs(30), async {
            writer.await.unwrap();
            exporter.await.unwrap();
        })
        .await
        .expect("remove vs snapshot export deadlocked");
    }

    /// 并发回归（线上实证事故的 lib 级复现）：remove 与 search、builder 合图
    /// 高频交织下，HNSW 图曾单边丢边留下悬空链接 → search 索引已删向量
    /// panic → 持锁 panic 毒化 RwLock → 后续所有 HNSW 操作连环失败。修复
    /// （对称剪枝 + 入口点最高层不变量 + 连边双端预检 + 搜索防御性跳过 +
    /// 锁中毒恢复）后，本测试断言交织压力下：无任何 panic（任务 join 成功）、
    /// 锁不中毒（收尾操作正常）、已删 id 对检索不可见。
    /// EN: Concurrency regression (lib-level replay of the production
    /// incident): under heavy remove/search/graph-merge interleaving the HNSW
    /// graph used to drop edges one-sidedly, leaving dangling links → search
    /// indexed a deleted vector and panicked → the panicking holder poisoned
    /// the RwLock → every later HNSW op failed. After the fix (symmetric
    /// pruning + top-level entry-point invariant + dual-endpoint link
    /// pre-check + defensive search + poison recovery), assert under
    /// interleaved pressure: no panic (task joins succeed), no poison
    /// (wrap-up ops work), deleted ids invisible to search.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_remove_search_and_build_stays_alive() {
        fn mkvec(id: u64) -> Vec<f32> {
            (0..16_u64)
                .map(|j| ((id.wrapping_mul(31).wrapping_add(j * 17)) % 1000) as f32 / 1000.0)
                .collect()
        }

        let index = Arc::new(HotBufferIndex::new_with_metric(16, DistanceMetric::L2));
        let base: Vec<(u64, Vec<f32>)> = (1..=400_u64).map(|i| (i, mkvec(i))).collect();
        index.insert_batch(base).await.unwrap();
        drain_builder_queue(&index).await;

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut searchers = Vec::new();
        for _ in 0..2 {
            let index = Arc::clone(&index);
            let stop = Arc::clone(&stop);
            searchers.push(tokio::spawn(async move {
                let mut q = 0_u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = index.search(&mkvec(q % 400 + 1), 10, None).await;
                    q = q.wrapping_add(1);
                    // 无间隔读风暴是本测试的核心压力面：std RwLock 读优先时,
                    // 连续读让合图写锁永远拿不到(60s 超时三次实证:无间隔/1ms/
                    // workspace 并行各一次,确诊饥饿非死锁)。hnsw 锁换 parking_lot
                    // (写者优先公平调度)后,读风暴下写锁保证可获取——不再需要
                    // sleep 让步,读压拉满反而更贴近生产高 QPS 形态。
                    // EN: The gapless read storm is this test's core pressure:
                    // under std RwLock (reader-preferring) it starved the merge
                    // write lock three times (60s timeouts: gapless, 1ms gaps,
                    // full workspace parallelism — starvation, not deadlock).
                    // After switching the hnsw lock to parking_lot (writer-
                    // preferring fairness), the write lock is guaranteed
                    // acquisition under read storms — no sleep yield needed,
                    // and maxed-out read pressure better matches production.
                }
            }));
        }

        const ROUNDS: u64 = 60;
        // 推进戳:writer 每轮 +1,watchdog 据此判定「真停滞」(饥饿/死锁)。
        // EN: Progress stamp bumped per writer round; the watchdog judges
        // true stalls (starvation/deadlock) from it.
        let progress = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let writer = {
            let index = Arc::clone(&index);
            let progress = Arc::clone(&progress);
            tokio::spawn(async move {
                for r in 0..ROUNDS {
                    let nid = 1000 + r;
                    index.insert_batch(vec![(nid, mkvec(nid))]).await.unwrap();
                    // 周期排空：让合图与 remove/search 真实交织。
                    // EN: Periodic draining: genuinely interleave merges with remove/search.
                    if r % 5 == 4 {
                        drain_builder_queue(&index).await;
                    }
                    // 删旧 id（含已合图的与仍在队列的）。
                    // EN: Delete old ids (both merged and still queued).
                    index.remove(r % 400 + 1).await;
                    progress.store(r + 1, std::sync::atomic::Ordering::Relaxed);
                }
                drain_builder_queue(&index).await;
            })
        };

        // 推进 watchdog 而非整体时间盒:「60s 完成 60 轮」在 CPU 超订的并行
        // 测试套件下天然 flaky(绝对吞吐随负载波动,三次 60s 超时实证:std
        // RwLock 读优先饥饿 ×2 + parking_lot 后全量并行 ×1——后者纯负载
        // 拖慢,非不推进)。本测试要抓的是「系统不再推进」(饥饿/死锁),不是
        // 「跑得慢」:连续 QUIET_LIMIT 次 WATCH_INTERVAL 检查零推进才判 fail。
        // 阈值 3 次而非 2 次:进度戳按整轮打,drain 轮(insert+整批合图
        // +remove)在 CPU 极端超订下单轮可能越过两次检查,把「慢」误判成
        // 「停滞」(OCR 裁决)。
        // EN: A progress watchdog, not a wall-clock budget: "60 rounds in 60s"
        // is inherently flaky under CPU-oversubscribed parallel test suites
        // (three 60s timeouts: std RwLock starvation ×2, then a pure
        // load-induced slowdown ×1 after parking_lot — slow, not stalled).
        // What this test must catch is "no forward progress" (starvation /
        // deadlock), not "slow": fail only after QUIET_LIMIT consecutive
        // zero-advancement checks. The limit is 3, not 2: progress is stamped
        // per full round, and a drain round (insert + full-batch merge +
        // remove) can legitimately span two checks under extreme CPU
        // oversubscription — slow misread as stall (OCR ruling).
        const WATCH_INTERVAL: Duration = Duration::from_secs(5);
        /// 停滞判定阈值 = QUIET_LIMIT × WATCH_INTERVAL(当前 3×5s = 15s);
        /// 两处任一改动须同步,panic 消息由二者现算,不写死秒数。
        /// EN: Stall verdict = QUIET_LIMIT × WATCH_INTERVAL (now 3×5s = 15s);
        /// keep them in sync — the panic message is computed, never hardcoded.
        const QUIET_LIMIT: u32 = 3;
        let stalled = {
            let progress = Arc::clone(&progress);
            tokio::spawn(async move {
                let mut last = 0_u64;
                let mut quiet_checks = 0_u32;
                loop {
                    sleep(WATCH_INTERVAL).await;
                    let p = progress.load(std::sync::atomic::Ordering::Relaxed);
                    if p >= ROUNDS {
                        return false; // 正常完成
                    }
                    if p == last {
                        quiet_checks += 1;
                        if quiet_checks >= QUIET_LIMIT {
                            return true; // 连续 QUIET_LIMIT 次零推进:真停滞
                        }
                    } else {
                        quiet_checks = 0;
                        last = p;
                    }
                }
            })
        };

        // JoinHandle 的 expect 只打 "task N panicked",吞掉底层 panic 负载与
        // 栈;一律 unwrap_or_else + resume_unwind 让真实 panic 原样抛出。
        // EN: `expect` on a JoinError discards the payload ("task N panicked"
        // only); always unwrap_or_else + resume_unwind so the real panic
        // message/backtrace propagates.
        let mut writer = Some(writer);
        let mut stalled = stalled;
        tokio::select! {
            w = writer.as_mut().unwrap() => {
                w.unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()));
                writer = None; // JoinHandle 已消费,尾部不得再 poll
                stalled.abort();
            }
            s = &mut stalled => {
                let stall =
                    s.unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()));
                assert!(
                    !stall,
                    "concurrent write path stalled (no progress ≥{}s) — \
                     writer starvation or deadlock",
                    QUIET_LIMIT * WATCH_INTERVAL.as_secs() as u32
                );
            }
        }
        // 收尾阶段统一 60s 时限:watchdog 在 writer 完成后已退出,若本测试要防
        // 的死锁类故障此刻才显形(如 searcher 持 guard 卡死、收尾 remove 被挡
        // 在其后),无时限的 await 会一路挂到 CI 外层超时(OCR 裁决)。尾部正常
        // 工作量为毫秒级(落旗后 searcher 立即退出 + 两次 trivial 操作),60s
        // 只捕挂死不捕慢。
        // EN: Uniform 60s bound on the tail — the watchdog is gone once the
        // writer finishes, so a deadlock-class failure surfacing only now (a
        // searcher stuck holding guards, the wrap-up remove queued behind it)
        // would otherwise hang until the outer CI timeout (OCR ruling).
        // Normal tail work is milliseconds; 60s catches hangs, not slowness.
        tokio::time::timeout(Duration::from_secs(60), async {
            // watchdog 分支里 writer 可能仅剩收尾 drain，在此一并等完。
            // EN: In the watchdog branch the writer may only have the final drain left; wait here.
            if let Some(w) = writer {
                w.await
                    .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()));
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for h in searchers {
                h.await
                    .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()));
            }
            // 锁未中毒：收尾 search/remove 正常执行；已删 id 不可见。
            // EN: Lock not poisoned: final search/remove run normally; deleted ids stay invisible.
            let found = index.search(&mkvec(3), 10, None).await;
            assert!(
                !found.contains(&3),
                "removed id 3 must be invisible: {found:?}"
            );
            assert!(index.remove(1001).await, "id 1001 was inserted by writer");
        })
        .await
        .expect("tail phase hung — post-writer deadlock (searcher stuck or final op blocked)");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn worker_flushes_concurrent_batch_inserts() {
        let index = Arc::new(HotBufferIndex::new(4));
        let worker = index.start_async_builder_worker(16, 10, FlushPolicy::Hybrid);

        let mut tasks = Vec::new();
        for i in 0..64_u64 {
            let index_clone = Arc::clone(&index);
            tasks.push(tokio::spawn(async move {
                index_clone
                    .insert_batch(vec![(i, vec![1.0, 2.0, 3.0, 4.0])])
                    .await
                    .unwrap();
            }));
        }

        for task in tasks {
            task.await.unwrap();
        }

        sleep(Duration::from_millis(120)).await;
        worker.shutdown().await;

        assert_eq!(index.pending_build_count().await, 0);
        assert_eq!(index.indexed_count().await, 64);
        let metrics = index.builder_metrics();
        assert!(metrics.flush_count >= 1);
        assert_eq!(metrics.total_flushed_items, 64);
        assert!(metrics.max_pending_queue >= 1);
    }

    #[tokio::test]
    async fn threshold_only_policy_waits_for_threshold() {
        let index = Arc::new(HotBufferIndex::new(4));
        let worker = index.start_async_builder_worker(10, 10, FlushPolicy::ThresholdOnly);

        index
            .insert_batch(vec![
                (1, vec![0.0, 0.0, 0.0, 0.0]),
                (2, vec![1.0, 1.0, 1.0, 1.0]),
            ])
            .await
            .unwrap();
        sleep(Duration::from_millis(60)).await;
        assert_eq!(index.indexed_count().await, 0);

        let mut batch = Vec::new();
        for i in 3..=12_u64 {
            batch.push((i, vec![2.0, 2.0, 2.0, 2.0]));
        }
        index.insert_batch(batch).await.unwrap();
        sleep(Duration::from_millis(60)).await;
        worker.shutdown().await;

        assert_eq!(index.pending_build_count().await, 0);
        assert_eq!(index.indexed_count().await, 12);
    }

    #[test]
    fn lock_free_neighbor_store_appends_neighbors() {
        let store = LockFreeNeighborStore::new(4);
        assert!(store.add_neighbor(1, 2));
        assert!(store.add_neighbor(1, 3));
        assert_eq!(store.neighbors_of(1), vec![2, 3]);
    }

    #[tokio::test]
    async fn builder_builds_hnsw_level0_stub_edges() {
        let index = HotBufferIndex::new(4);
        index
            .insert_batch(vec![
                (100, vec![0.0, 0.0, 0.0, 0.0]),
                (101, vec![1.0, 1.0, 1.0, 1.0]),
                (102, vec![2.0, 2.0, 2.0, 2.0]),
            ])
            .await
            .unwrap();
        let rep = index.run_async_builder_once().await;
        assert_eq!(rep.flushed, 3);

        let n102 = index.neighbors_of(102);
        assert!(
            !n102.is_empty(),
            "HNSW should assign at least one level-0 neighbor"
        );
        let metrics = index.builder_metrics();
        assert!(metrics.graph_edges_built > 0);
        assert!(metrics.graph_edges_level0 > 0);
    }

    #[tokio::test]
    async fn builder_assigns_entry_point_and_upper_layer_links() {
        let index = HotBufferIndex::new(4);
        index
            .insert_batch(vec![
                (200, vec![0.0, 0.0, 0.0, 0.0]),
                (201, vec![1.0, 1.0, 1.0, 1.0]),
                (202, vec![2.0, 2.0, 2.0, 2.0]),
                (203, vec![3.0, 3.0, 3.0, 3.0]),
            ])
            .await
            .unwrap();
        let _ = index.run_async_builder_once().await;

        let ep = index.entry_point();
        assert!(ep.is_some());
        assert!(index.node_level(ep.unwrap_or_default()).is_some());
        let metrics = index.builder_metrics();
        assert!(metrics.graph_edges_built >= metrics.graph_edges_level0);
    }

    #[tokio::test]
    async fn search_returns_distance_sorted_ids_with_filter() {
        let index = HotBufferIndex::new(2);
        index.insert(1, vec![0.0, 0.0]).await.unwrap();
        index.insert(2, vec![1.0, 0.0]).await.unwrap();
        index.insert(3, vec![2.0, 0.0]).await.unwrap();

        let all = index.search(&[0.9, 0.0], 3, None).await;
        assert_eq!(all, vec![2, 1, 3]);

        let filtered = index.search(&[0.9, 0.0], 3, Some(&[1, 3])).await;
        assert_eq!(filtered, vec![1, 3]);
    }

    #[tokio::test]
    async fn known_vectors_search_returns_self_ids() {
        let index = HotBufferIndex::new(4);
        let mut batch = Vec::new();
        for id in 0_u64..10 {
            let mut v = vec![0.0_f32; 4];
            v[(id as usize) % 4] = 1.0 + (id as f32) * 0.01;
            batch.push((id, v));
        }
        index.insert_batch(batch.clone()).await.unwrap();
        let _ = index.run_async_builder_once().await;

        for (id, v) in batch {
            let top = index.search(&v, 1, None).await;
            assert_eq!(top, vec![id], "query by exact vector should hit self id");
        }
    }

    /// TDD：scored 检索返回 (id, 距离) 对——距离按升序排列，且 id 序列与
    /// 仅返回 id 的 `search_with_ef_override` 在相同入参下完全一致（防两条链路排序漂移）。
    /// EN: Scored search must return ascending distances and the exact same id order
    /// as the id-only variant on identical inputs.
    #[tokio::test]
    async fn scored_search_ascending_distances_and_matches_id_only() {
        let dims = 16usize;
        let index = HotBufferIndex::new_with_metric(dims, DistanceMetric::Cosine);
        index.set_hnsw_m(16).unwrap();
        index.set_hnsw_ef_construction(128).unwrap();
        index.set_hnsw_ef_search(64).unwrap();
        let batch: Vec<(u64, Vec<f32>)> = (0..200_u64)
            .map(|id| {
                (
                    id,
                    (0..dims)
                        .map(|j| ((id as usize * 13 + j * 7) % 200) as f32 / 200.0)
                        .collect(),
                )
            })
            .collect();
        index.insert_batch(batch).await.unwrap();
        drain_builder_queue(&index).await;

        let q = vec![0.42_f32; dims];
        let scored = index
            .search_scored_with_ef_override(&q, 10, None, None)
            .await;
        assert_eq!(scored.len(), 10);
        let scored_ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
        let ids_only = index.search_with_ef_override(&q, 10, None, None).await;
        assert_eq!(scored_ids, ids_only, "scored/id-only 链路 id 顺序必须一致");
        for pair in scored.windows(2) {
            assert!(
                pair[0].1 <= pair[1].1,
                "距离必须升序: {} > {}",
                pair[0].1,
                pair[1].1
            );
        }
        assert!(scored.iter().all(|(_, d)| *d >= 0.0), "余弦距离必须非负");
    }

    /// TDD：查询向量与某条已写入向量完全相同时，其距离必须 ≈ 0（余弦距离约定，
    /// 无论向量是否预归一化）；带 mask 时同样成立。
    /// EN: Query identical to a stored vector yields distance ~0 (cosine distance),
    /// including through the filter-mask brute-force branch.
    #[tokio::test]
    async fn scored_search_exact_match_distance_near_zero() {
        let dims = 4usize;
        let index = HotBufferIndex::new_with_metric(dims, DistanceMetric::Cosine);
        // 非单位向量：验证全余弦公式（内部按范数归一化）而非假设输入已归一化。
        // Non-unit vector: full cosine formula normalizes by norm internally.
        let target = vec![3.0_f32, 0.0, -1.0, 2.0];
        index
            .insert_batch(vec![
                (1, vec![0.0, 1.0, 0.0, 0.0]),
                (2, target.clone()),
                (3, vec![1.0, 1.0, 1.0, 1.0]),
            ])
            .await
            .unwrap();
        // 批量写入先进 builder_queue，排空后才进入主图可被检索（与既有测试约定一致）。
        // Batched rows land in builder_queue first; drain before they are searchable.
        drain_builder_queue(&index).await;

        let scored = index
            .search_scored_with_ef_override(&target, 3, None, None)
            .await;
        assert_eq!(scored.first().map(|(id, _)| *id), Some(2));
        assert!(
            scored.first().map(|(_, d)| *d).unwrap_or(f32::NAN).abs() < 1e-5,
            "exact match distance must be ~0, got {:?}",
            scored.first().map(|(_, d)| *d)
        );

        // 小 mask 走 rayon 暴力分支，精确匹配距离同样必须 ≈ 0。
        // Small mask takes the rayon brute-force branch.
        let masked = index
            .search_scored_with_ef_override(&target, 3, Some(&[1, 2]), None)
            .await;
        assert_eq!(
            masked.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(masked.first().map(|(_, d)| *d).unwrap_or(f32::NAN).abs() < 1e-5);
    }

    #[test]
    fn hnsw_ef_search_can_be_configured() {
        // 1536D 默认推断 ef_search=512（见 hnsw_tuning）。
        // EN: 1536D defaults to ef_search=512 (see hnsw_tuning).
        let index = HotBufferIndex::new(1536);
        assert_eq!(index.hnsw_ef_search(), 512);
        index.set_hnsw_ef_search(32).unwrap();
        assert_eq!(index.hnsw_ef_search(), 32);
        assert!(index.set_hnsw_ef_search(0).is_err());
    }

    #[test]
    fn simd_feature_l2_distance_matches_baseline_math() {
        let a = vec![1.0_f32; 32];
        let b = vec![2.0_f32; 32];
        let d = super::l2_distance(&a, &b);
        assert_eq!(d, 32.0);
    }

    async fn run_concurrent_build_case(total: u64, worker_threads: usize) {
        let index = Arc::new(HotBufferIndex::new(4));
        let worker = index.start_async_builder_worker(64, 5, FlushPolicy::Hybrid);

        let mut tasks = Vec::new();
        for i in 0..total {
            let index_clone = Arc::clone(&index);
            tasks.push(tokio::spawn(async move {
                index_clone
                    .insert_batch(vec![(i, vec![1.0, 2.0, 3.0, 4.0])])
                    .await
                    .unwrap();
            }));
            if tasks.len() >= worker_threads {
                for task in tasks.drain(..) {
                    task.await.unwrap();
                }
            }
        }
        for task in tasks {
            task.await.unwrap();
        }

        // 自适应等待队列排空：Linux CI 高并发下固定 2.5s 偶发不足（OOM/抢占），
        // 会让「并发构建一致性」用例误报；超时后仍由下面的断言暴露真实停滞。
        // EN: adaptively wait for the queue to drain — a fixed 2.5s window is
        // flaky on loaded Linux CI. A real stall still fails the assertion below.
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        while index.pending_build_count().await != 0 {
            if std::time::Instant::now() >= deadline {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
        worker.shutdown().await;

        assert_eq!(index.pending_build_count().await, 0);
        assert_eq!(index.indexed_count().await as u64, total);

        let metrics = index.builder_metrics();
        assert!(metrics.flush_count >= 1);
        assert_eq!(metrics.total_flushed_items as u64, total);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_build_regression_128() {
        run_concurrent_build_case(128, 64).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_build_regression_256() {
        run_concurrent_build_case(256, 64).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_build_regression_512() {
        run_concurrent_build_case(512, 64).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_build_regression_1024_high_thread_consistency() {
        run_concurrent_build_case(1024, 128).await;
    }

    /// 并发调用 `run_async_builder_once`（后台 worker、HTTP optimize、恢复排水
    /// 都可能同时进入）：builder_gate 必须让同一 id 不会被两个在途批次重复提交，
    /// 最终 main 与图节点数一致、所有点可检索。
    /// EN: Concurrent builder invocations must serialize so the same id cannot be
    /// committed from two in-flight batches; main and graph stay in sync.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_builder_invocations_keep_main_and_graph_in_sync() {
        fn mkvec(id: u64) -> Vec<f32> {
            (0..8)
                .map(|j| ((id as f32) * 0.01 + j as f32 * 0.1) % 1.0)
                .collect()
        }

        let index = Arc::new(HotBufferIndex::new_with_metric(8, DistanceMetric::L2));
        // 第一批把图播种到 seed_serial_points(默认 256)，后续批次才会走 plan/apply。
        // EN: The first batch seeds to seed_serial_points (default 256); later batches use plan/apply.
        let seed: Vec<(u64, Vec<f32>)> = (1..=256_u64).map(|i| (i, mkvec(i))).collect();
        index.insert_batch(seed).await.unwrap();
        drain_builder_queue(&index).await;
        // 600 点 > max_build_batch(默认 512)，保证并发调用会 drain 出多个批次。
        // EN: 600 points > max_build_batch (default 512), ensuring concurrent calls drain multiple batches.
        let batch: Vec<(u64, Vec<f32>)> = (1001..=1600_u64).map(|i| (i, mkvec(i))).collect();
        index.insert_batch(batch).await.unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let index = Arc::clone(&index);
            handles.push(tokio::spawn(async move {
                for _ in 0..64 {
                    if index.run_async_builder_once().await.flushed == 0 {
                        break;
                    }
                }
            }));
        }
        // 并发构建期间重插已提交 id：必须走 upsert 回退（先摘旧节点），
        // 最终值以后写为准、节点数不增。串行门保证重插批次不会与在途旧批
        // 同时以 fresh 规划。
        // EN: Re-insert committed ids while builders run — must take the upsert
        // fallback (remove old node first); last write wins, no node growth.
        let updated = vec![0.999_f32; 8];
        let rows: Vec<(u64, Vec<f32>)> = (1..=128_u64).map(|i| (i, updated.clone())).collect();
        index.insert_batch(rows).await.unwrap();
        for h in handles {
            h.await.unwrap();
        }
        drain_builder_queue(&index).await;

        let main = index.indexed_count().await;
        let graph = index.graph_node_count().await;
        assert_eq!(main, 856, "all points must reach main (got {main})");
        assert_eq!(graph, main, "graph and main must not diverge");
        let ids: Vec<u64> = (1..=128).collect();
        let got = index.vectors_by_ids(&ids).await;
        assert_eq!(got.len(), 128, "every re-inserted id must be visible");
        assert!(
            got.iter()
                .all(|(_, v)| v.iter().all(|x| (*x - 0.999).abs() < 1e-6)),
            "last write must win for re-inserted ids"
        );
        let hits = index.search(&[0.05_f32; 8], 10, None).await;
        assert!(!hits.is_empty(), "search must return results after build");
    }

    /// 合图排队期间修改的调参值必须在拿到串行门后生效（而非 spawn 时快照）。
    /// EN: Tuning updated while a build is queued behind the gate must take
    /// effect after the gate is acquired (not snapshotted at spawn time).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tuning_changes_queued_behind_builder_gate_take_effect() {
        fn mkvec(id: u64) -> Vec<f32> {
            (0..8)
                .map(|j| ((id as f32) * 0.01 + j as f32 * 0.1) % 1.0)
                .collect()
        }

        let index = Arc::new(HotBufferIndex::new_with_metric(8, DistanceMetric::L2));
        let seed: Vec<(u64, Vec<f32>)> = (1..=256_u64).map(|i| (i, mkvec(i))).collect();
        index.insert_batch(seed).await.unwrap();
        drain_builder_queue(&index).await;

        let default_ef = index.hnsw_ef_construction_value();
        let new_ef = (default_ef + 100).min(4096);
        assert_ne!(new_ef, default_ef);

        // 占住串行门。`started` 只证明任务被 poll 过一次；给足调度余量，
        // 让「旧的 spawn 前快照实现」也必然已完成快照——这是 best-effort
        // 同步（生产代码没有内建信号点），200ms 对单个测试无成本。
        // EN: `started` only proves one poll. The 200ms margin is best-effort,
        // ensuring a hypothetical spawn-time snapshot has already happened;
        // there is no built-in signal point in production code.
        let gate = index.builder_gate.lock().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let idx = Arc::clone(&index);
        let build = tokio::spawn(async move {
            let _ = started_tx.send(());
            idx.run_async_builder_once().await
        });
        started_rx.await.unwrap();
        sleep(Duration::from_millis(200)).await;
        // 排队期间改调参并入队；释放门后本批应使用新值。
        // EN: Change knobs and enqueue while queued; this batch must use the new values after the gate.
        index.set_hnsw_ef_construction(new_ef).unwrap();
        let batch: Vec<(u64, Vec<f32>)> = (1001..=1100_u64).map(|i| (i, mkvec(i))).collect();
        index.insert_batch(batch).await.unwrap();
        drop(gate);
        let rep = build.await.unwrap();
        assert!(rep.flushed > 0, "queued batch must be built");
        drain_builder_queue(&index).await;

        // 与 builder 同口径计算最终写入图的束宽（含 bulk 上限）。
        // EN: Mirror the builder's cap-aware computation of the applied beam.
        let requested = new_ef.max(index.hnsw_m().saturating_mul(2)).clamp(16, 4096);
        // 直接复用生产解析（含 `0 = 不截断` 哨兵），避免镜像逻辑漂移。
        // EN: Reuse the production parser (incl. the `0 = uncapped` sentinel).
        let expected = BuildConfig::from_env()
            .bulk_ef_construction_cap
            .map_or(requested, |cap| requested.min(cap));
        let g = crate::hnsw_read_guard(&index.hnsw);
        assert_eq!(
            g.ef_construction_for_test(),
            expected,
            "queued build must use tuning set after acquiring the gate"
        );
    }
}
