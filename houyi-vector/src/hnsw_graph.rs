//! Full HNSW graph (Malkov–Yashunin style): layered greedy search, heuristic neighbor selection.
//! 中文：标准分层可导航小世界图；与 `HotBufferIndex` 的向量存储解耦，本模块持有构图副本以便检索。
//! EN: Standalone graph + vector map; duplicates vectors from main index for O(1) distance during search.

use crate::quantization::{distance_query, distance_stored, Int8Ref, Int8Vector, Quantization};
use crate::rabitq::{self, RaBitQQuery, RaBitQRef, RaBitQVector};
use crate::stored_vector::{distance_between, StoredVector, VectorDtype, VectorRef};
use crate::{DistanceMetric, VectorId};
use ahash::{AHashMap, AHashSet};
use houyi_core::config::env_bool;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

pub(crate) const MAX_HNSW_LAYERS: usize = 16;
// 连通兜底的告警位图宽度必须覆盖全部合法层：层号 0..MAX_HNSW_LAYERS-1 的
// 位移不得越界（否则 debug 在持写锁的路径上 panic、release 回绕到错误层）。
// EN: The connectivity-warn bitmap must cover every legal layer: shifts for
// layer ids 0..MAX_HNSW_LAYERS-1 must stay in range (otherwise a debug build
// panics under the write lock and release wraps to the wrong layer).
const _: () = assert!(MAX_HNSW_LAYERS <= u64::BITS as usize);

const NEIGHBOR_LINK_SLACK: usize = 2;
const LINK_PAD: VectorId = VectorId::MAX;

/// 世代式 visited；避免每次 `search_layer_inner` 对 `AHashSet::clear` 的全表失效开销。
/// EN: Generational visited marks; avoids per-search full clear on a hash set.
#[derive(Debug, Default, Clone)]
pub struct HnswVisitTracker {
    epoch: u64,
    marks: AHashMap<VectorId, u64>,
}

impl HnswVisitTracker {
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            epoch: 0,
            marks: AHashMap::with_capacity(capacity),
        }
    }

    /// 每次 `search_layer_inner` 入口调用，取代 `visited.clear()`。
    /// EN: One call per `search_layer_inner` entry; replaces `visited.clear()`.
    #[inline]
    pub fn begin_layer_search(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.clear();
        }
    }

    /// 与 `AHashSet::insert(id)` 语义对齐：新见到则 `true`。
    /// EN: Aligned with `AHashSet::insert`: `true` when newly marked this epoch.
    #[inline]
    pub fn try_visit(&mut self, id: VectorId) -> bool {
        match self.marks.get(&id).copied() {
            Some(e) if e == self.epoch => false,
            _ => {
                self.marks.insert(id, self.epoch);
                true
            }
        }
    }
}

/// 最小堆候选：距离最小者优先。
/// EN: Min-heap candidate: smallest distance first.
#[derive(Debug, Clone, Copy)]
struct Cd {
    dist: f32,
    id: VectorId,
}

impl PartialEq for Cd {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Cd {}
impl PartialOrd for Cd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cd {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.dist.partial_cmp(&other.dist) {
            Some(Ordering::Equal) => other.id.cmp(&self.id),
            Some(o) => o.reverse(),
            None => other.id.cmp(&self.id),
        }
    }
}

/// 用于动态结果列表的最大堆（最差在堆顶）。
/// EN: Max-heap for dynamic result list (worst on top).
#[derive(Debug, Clone, Copy)]
struct Wd {
    dist: f32,
    id: VectorId,
}

impl PartialEq for Wd {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Wd {}
impl PartialOrd for Wd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Wd {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.dist.partial_cmp(&other.dist) {
            Some(Ordering::Equal) => self.id.cmp(&other.id),
            Some(o) => o,
            None => self.id.cmp(&other.id),
        }
    }
}

/// 图内向量：完整存储、int8 或 RaBitQ 量化（P2/P3）。量化变体同时持有原始
/// `source`，与主表共享同一 `Arc`——快照恢复/导出必须还原原始值，而不是反量化
/// 近似值。
/// EN: Graph-side vector: full stored value, int8, or RaBitQ quantized (P2/P3).
/// The quantized variants also hold the original `source`, sharing the same `Arc`
/// as the main store, so snapshot restore/export recover raw values rather than
/// dequantized approximations.
#[derive(Clone)]
pub(crate) enum GraphVector {
    /// 未量化（图内直接使用 `StoredVector` 视图）。
    /// EN: Unquantized (the graph uses the `StoredVector` view directly).
    Full(Arc<StoredVector>),
    /// 图侧 int8：`codes` 独立分配，`source` 与主表共享。
    /// EN: Graph-side int8: `codes` is separately allocated, `source` is shared
    /// with the main store.
    Int8 {
        codes: Arc<Int8Vector>,
        source: Arc<StoredVector>,
    },
    /// 图侧 RaBitQ 1 bit/dim：`codes` 独立分配（符号位 + alpha + norm），
    /// `source` 与主表共享；维数由 `source` 提供。
    /// EN: Graph-side RaBitQ 1 bit/dim: `codes` is separately allocated (sign
    /// bits + alpha + norm) and `source` is shared with the main store; dims
    /// come from `source`.
    RabitQ {
        codes: Arc<RaBitQVector>,
        source: Arc<StoredVector>,
    },
}

impl GraphVector {
    /// 原始（未量化）向量句柄：供主表重建/导出使用。
    /// EN: Handle to the raw (unquantized) vector for main-store rebuild/export.
    pub(crate) fn source_arc(&self) -> Arc<StoredVector> {
        match self {
            GraphVector::Full(v) => Arc::clone(v),
            GraphVector::Int8 { source, .. } => Arc::clone(source),
            GraphVector::RabitQ { source, .. } => Arc::clone(source),
        }
    }

    /// 原始向量的借用视图（非量化语义的距离计算）。
    /// EN: Borrowed view of the raw vector (distance math under unquantized semantics).
    pub(crate) fn source_view(&self) -> VectorRef<'_> {
        match self {
            GraphVector::Full(v) => v.as_view(),
            GraphVector::Int8 { source, .. } => source.as_view(),
            GraphVector::RabitQ { source, .. } => source.as_view(),
        }
    }

    /// 维度数（量化 codes 与 source 同维）。
    /// EN: Dimensions (quantized codes and source always share the same dims).
    pub(crate) fn dims(&self) -> usize {
        match self {
            GraphVector::Full(v) => v.dims(),
            GraphVector::Int8 { codes, .. } => codes.dims(),
            // RaBitQ 码本身不记维数：source 恒与码同维，取源头更可靠。
            // EN: RaBitQ codes do not record dims: source always shares them, so
            // taking it from the source is the more robust choice.
            GraphVector::RabitQ { source, .. } => source.dims(),
        }
    }

    /// 原始向量的存储 dtype（主表编码，不是图侧量化方式）。
    /// EN: Storage dtype of the raw vector (main-store encoding, not graph quantization).
    pub(crate) fn source_dtype(&self) -> VectorDtype {
        match self {
            GraphVector::Full(v) => v.dtype(),
            GraphVector::Int8 { source, .. } => source.dtype(),
            GraphVector::RabitQ { source, .. } => source.dtype(),
        }
    }

    /// 图侧量化方式。
    /// EN: Graph-side quantization.
    pub(crate) fn quantization(&self) -> Quantization {
        match self {
            GraphVector::Full(_) => Quantization::None,
            GraphVector::Int8 { .. } => Quantization::Int8,
            GraphVector::RabitQ { .. } => Quantization::RaBitQ,
        }
    }

    /// 借用为图内距离核使用的视图。
    /// EN: Borrows as the view consumed by in-graph distance kernels.
    pub(crate) fn as_graph_ref(&self) -> GraphVectorRef<'_> {
        match self {
            GraphVector::Full(v) => GraphVectorRef::Full(v.as_view()),
            GraphVector::Int8 { codes, .. } => GraphVectorRef::Int8(codes.view()),
            GraphVector::RabitQ { codes, source } => GraphVectorRef::RabitQ {
                codes: codes.view(source.dims()),
                source: source.as_view(),
            },
        }
    }
}

/// 与 [`GraphVector`] 对应的借用视图；不拥有数据。
/// EN: Borrowed view over a [`GraphVector`]; owns nothing.
#[derive(Clone, Copy)]
pub(crate) enum GraphVectorRef<'a> {
    /// 完整存储视图。
    /// EN: Full stored view.
    Full(VectorRef<'a>),
    /// int8 量化视图。
    /// EN: int8 quantized view.
    Int8(Int8Ref<'a>),
    /// RaBitQ 量化视图：码视图 + 原始向量视图。原始视图供构图/维护期的
    /// 存储×存储距离走精确计算（见 [`metric_pair_stored`] 的说明）。
    /// EN: RaBitQ quantized view: the code view plus the raw-vector view. The
    /// raw view lets construction/maintenance stored×stored distances stay
    /// exact (see the rationale on [`metric_pair_stored`]).
    RabitQ {
        codes: RaBitQRef<'a>,
        source: VectorRef<'a>,
    },
}

/// HNSW 图节点；邻接为图级 `link_arena` 中一段连续子块（真 Flat-Link）。
/// EN: HNSW node; adjacency is a contiguous sub-block in the graph `link_arena` (true flat link).
#[derive(Debug, Clone)]
pub(crate) struct HnswNode {
    /// 本节点各层平坦块在 `HnswGraph::link_arena` 中的基索引。
    /// EN: Base index in `HnswGraph::link_arena` for this node's flat per-layer block.
    pub(crate) link_base: usize,
    /// 分配到的最高层 L（0..=L 共 L+1 个邻接层）。
    /// EN: Top assigned layer L; `level_count` = L+1.
    pub(crate) max_layer: u8,
    /// 每层已存邻居数，合法 id 位于 `link_arena[base+off..base+off+len)`。
    /// EN: Per-layer used count; valid ids lie in the arena subslice of length `layer_len[layer]`.
    pub(crate) layer_len: [u16; MAX_HNSW_LAYERS],
}

impl HnswNode {
    /// EN: `neighbors.len()` 等价：参与构图的层数 = `max_layer + 1`。
    #[inline]
    pub(crate) fn level_count(&self) -> usize {
        self.max_layer as usize + 1
    }

    /// 本节点第 `l` 层区间在 `link_arena` 中的绝对起始索引（容量 = `link_cap(m,l)`）。
    /// EN: Absolute `link_arena` start index for this node's layer-`l` run (capacity = `link_cap(m,l)`).
    #[inline]
    fn layer_start(&self, m: usize, l: usize) -> usize {
        self.link_base + layer_start_offset(m, l)
    }

    /// `layer` 层活跃（已用）邻居 id 的只读视图（不含尾部 slack）。
    /// EN: Read-only view of the active (used) neighbor ids for `layer` (excludes slack tail).
    fn link_active_ref<'a>(
        &'a self,
        arena: &'a [VectorId],
        m: usize,
        layer: usize,
    ) -> Option<&'a [VectorId]> {
        if layer > self.max_layer as usize {
            return None;
        }
        let s = self.layer_start(m, layer);
        let n = self.layer_len[layer] as usize;
        if s + n > arena.len() {
            return None;
        }
        if n == 0 {
            return Some(&[]);
        }
        Some(&arena[s..s + n])
    }
}

/// 基于只读图快照计算出的批量构建插入计划。
/// EN: Batch build insert plan computed against a read-only graph snapshot.
pub(crate) struct InsertPlan {
    pub(crate) id: VectorId,
    /// 已按图量化方式准备的图侧向量（余弦度量下图侧副本已归一化）；提交时直接入图。
    /// EN: Graph-side vector prepared for the graph's quantization (the graph copy
    /// is normalized under cosine); applied to the graph as-is at commit time.
    pub(crate) vector: GraphVector,
    level: usize,
    layers_neighbors: Vec<(usize, Vec<VectorId>)>,
}

/// 快照恢复的全部组成部分（P4.4.0）：`decode_snapshot` 解码校验后打包成单一
/// 命名结构传入 [`HnswGraph::restore_from_snapshot`]——13 个值里 7 个是相邻
/// 数字类型，位置参数换位可静默编译通过，具名字段让错配不可能（OCR ae8d7e3）。
/// EN: All decoded snapshot sections bundled into one named struct, so a
/// transposed positional argument cannot compile silently.
pub(crate) struct SnapshotGraphParts {
    pub(crate) dims: usize,
    pub(crate) metric: DistanceMetric,
    /// 图内存储的向量 dtype；快照格式本身按 f32 字节落盘，解码时按此下转。
    /// EN: Vector dtype stored in the graph; the snapshot format itself writes
    /// f32 bytes and down-converts to this dtype on decode.
    pub(crate) dtype: VectorDtype,
    /// 图侧量化方式（创建后不可变；快照 v3 必须与之一致）。
    /// EN: Graph-side quantization (immutable after creation; snapshot v3 must
    /// agree with it).
    pub(crate) quantization: Quantization,
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) ef_search: usize,
    pub(crate) num_layers: usize,
    pub(crate) seed: u64,
    pub(crate) nodes: AHashMap<VectorId, HnswNode>,
    pub(crate) link_arena: Vec<VectorId>,
    pub(crate) link_free: AHashMap<usize, Vec<usize>>,
    pub(crate) vectors: AHashMap<VectorId, GraphVector>,
    pub(crate) entry_point: Option<VectorId>,
    pub(crate) max_layer: usize,
}

/// In-memory HNSW index: owns a copy of inserted vectors for distance evaluation.
/// 邻接边存储在 `link_arena` 的连续子块中（真 Flat-Link）；`link_free` 复用等尺寸已释放块。
/// EN: Adjacency in graph-wide `link_arena` sub-blocks; `link_free` reuses equal-size released blocks.
pub struct HnswGraph {
    pub(crate) dims: usize,
    pub(crate) metric: DistanceMetric,
    /// 图内向量存储的 dtype（创建后不可变；查询向量始终保持 f32）。
    /// EN: Vector dtype stored inside the graph (immutable after creation; query
    /// vectors always stay f32).
    pub(crate) dtype: VectorDtype,
    /// 图侧量化方式（创建后不可变；查询向量始终保持 f32）。
    /// EN: Graph-side quantization (immutable after creation; query vectors
    /// always stay f32).
    pub(crate) quantization: Quantization,
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) ef_search: usize,
    pub(crate) num_layers: usize,
    /// 构造种子：`StdRng` 无法取回种子，快照需要持久化它以在恢复时重播种。
    /// EN: Construction seed — `StdRng` cannot return its seed, but the graph
    /// snapshot needs it to reseed on restore.
    pub(crate) seed: u64,
    // 保持私有：rng 活状态不得被外部改写——快照持久化的是 seed，外部直接赋值
    // rng 会让「恢复后同种子重播种」的确定性契约静默失效（OCR ae8d7e3）。
    // EN: Keep private: live rng state must not be externally assigned — snapshots persist the seed,
    // EN: and assigning rng directly silently breaks the reseed-on-restore determinism contract.
    rng: StdRng,
    pub(crate) nodes: AHashMap<VectorId, HnswNode>,
    /// 所有节点的全部邻居 id，按固定每节点块布局（见 `HnswNode::link_base` / `link_cap`）。
    /// EN: All neighbor ids for all nodes, laid out in fixed per-node blocks (see `HnswNode::link_base` / `link_cap`).
    pub(crate) link_arena: Vec<VectorId>,
    /// 块大小 `slots` -> 空闲 `link_base` 列表，用于 `remove` 后复用、降低 arena 无界增长风险。
    /// EN: `block_slots -> [bases]` for reusing on delete.
    pub(crate) link_free: AHashMap<usize, Vec<usize>>,
    pub(crate) vectors: AHashMap<VectorId, GraphVector>,
    pub(crate) entry_point: Option<VectorId>,
    pub(crate) max_layer: usize,
    /// 已告警过的断开层位图（第 L 位 = 第 L 层兜底后仍有不可达分量）。兜底在
    /// drain 排空钩子的图写锁内执行，持久断裂若每次排空都打 warn 会刷屏并可能
    /// 阻塞持锁线程；按层去重，恢复全可达后清除对应位，再次断裂时重新告警一次。
    /// EN: Bitmap of layers already warned about (bit L = layer L still had
    /// unreachable components after the sweep). The sweep runs under the graph
    /// write lock from the drain-empty hook; a persistent break would flood the
    /// log and could block the lock holder, so warn once per layer and clear the
    /// bit once the layer becomes fully reachable (a later break warns again).
    /// Width is tied to `MAX_HNSW_LAYERS` by the compile-time guard above; a
    /// `clear()`/`remove_node()` lifecycle transition also resets/masks stale
    /// bits so a later break in the same layer warns again.
    incomplete_warn_mask: u64,
}

/// 图内距离：余弦走「已归一化」快速路径，其余度量复用跨 dtype 距离核。
/// EN: In-graph distance: cosine takes the pre-normalized fast path; other
/// metrics reuse the cross-dtype kernels.
#[inline]
fn metric_pair(metric: DistanceMetric, a: VectorRef<'_>, b: VectorRef<'_>) -> f32 {
    match metric {
        // Cosine 路径约定输入已单位化；任意 dtype 组合都在 f32 上累加点积。
        // EN: Normalized inputs; every dtype combination accumulates the dot in f32.
        DistanceMetric::Cosine => crate::cosine_prenormalized(a, b),
        // IP 取负点积、L1 取曼哈顿原值：图排序只依赖全序，不依赖具体量纲。
        // EN: IP negates the dot and L1 uses the raw Manhattan sum; the graph
        // only relies on the ordering, not on the scale.
        other => distance_between(a, b, other),
    }
}

/// 图遍历的查询上下文：入口按图侧量化方式构造一次，之后逐邻居比较零预处理。
/// `Exact`/`Int8` 的 f32 查询分量在距离分派里走同一条路径，分开只为让「未量化图
/// vs int8 图」在断言/日志里可区分；RaBitQ 携带入口处已补 0、已旋转、已单位化的
/// 查询（O(D log D) 只发生一次，不进邻接循环）。
/// EN: Query context for graph traversal: built once at the entrance from the
/// graph quantization, so per-neighbor comparisons pay zero preprocessing.
/// `Exact`/`Int8` share the same f32 dispatch path and are separated only to keep
/// "unquantized vs int8 graph" distinguishable in asserts/logs; RaBitQ carries
/// the query already padded, rotated, and unitized at the entrance (one O(D log D)
/// pass, never inside the adjacency loop).
#[derive(Clone, Copy)]
pub(crate) enum QueryContext<'a> {
    /// 未量化图：原始（余弦已归一化）f32 查询。
    /// EN: Unquantized graph: raw (cosine-normalized) f32 query.
    Exact(VectorRef<'a>),
    /// int8 图：f32 查询（余弦已归一化）。
    /// EN: int8 graph: f32 query (cosine-normalized).
    Int8(VectorRef<'a>),
    /// RaBitQ 图：已补 0、已旋转、已单位化的查询 + 原始范数。
    /// EN: RaBitQ graph: padded, rotated, unitized query plus its raw norm.
    RabitQ(RaBitQQuery<'a>),
}

/// 查询 × 图内存储的距离分派（搜索/插入下降路径）。`Full` 分支保留既有
/// `metric_pair` 口径，旧图数值逐位不变；`Int8` 分支按反量化值计算；`RabitQ`
/// 在搜索热路径走 1 bit/dim 非对称估计，构图期（`Exact` 查询）则与原始向量
/// 精确比较。除此之外的混合形态只可能来自损坏或未来新格式：debug 构建断言，
/// release 退化为最远距离（只影响损坏图的排序，不影响正常路径）。
/// EN: Query × stored distance dispatch (search / insert descent). The `Full`
/// branch keeps the established `metric_pair` contract, leaving legacy graph
/// values bit-identical; `Int8` evaluates dequantized values; `RabitQ` uses the
/// 1-bit asymmetric estimator on the search hot path and exact raw-vector
/// comparisons during construction (`Exact` queries). Any other mixed shape can
/// only come from corruption or a future format: debug builds assert, release
/// degrades to the farthest distance (only affecting corrupted-graph ordering,
/// never normal paths).
#[inline]
fn metric_pair_query(
    metric: DistanceMetric,
    query: QueryContext<'_>,
    stored: GraphVectorRef<'_>,
) -> f32 {
    match (query, stored) {
        (QueryContext::Exact(q) | QueryContext::Int8(q), GraphVectorRef::Full(v)) => {
            metric_pair(metric, q, v)
        }
        (QueryContext::Exact(q) | QueryContext::Int8(q), GraphVectorRef::Int8(v)) => {
            distance_query(metric, q, v)
        }
        (QueryContext::RabitQ(q), GraphVectorRef::RabitQ { codes, .. }) => {
            rabitq::query_distance(metric, q, codes)
        }
        // RaBitQ 构图期：新节点的精确 f32 查询对已存节点的原始向量视图。
        // 与 `metric_pair_stored` 同理，构图不量化查询侧。
        // EN: RaBitQ build time: the new node's exact f32 query against an
        // existing node's raw-vector view. As with `metric_pair_stored`, the
        // build path does not quantize the query side.
        (QueryContext::Exact(q), GraphVectorRef::RabitQ { source, .. }) => {
            distance_between(q, source, metric)
        }
        _ => {
            debug_assert!(false, "mixed query/stored graph quantization");
            f32::INFINITY
        }
    }
}

/// 图内两个存储向量之间的距离分派（启发式选边/剪枝/压缩路径）。Full/Int8
/// 混合形态（仅可能来自损坏或未来新格式）沿用查询式接口兜底；RaBitQ 与其它
/// 形态混合没有几何意义，debug 断言并在 release 退化为最远。
///
/// RaBitQ 的存储×存储**刻意走原始向量精确距离**，而不是两段码位的
/// arcsine 估计：召回门实测 1 bit 存储对估计在非零均值（全正分量）数据上
/// 严重失真（1024 维 L2 数据候选召回在束宽 512 时仍只有 0.9），据此构图的
/// 图连通性崩塌。`source` 本就与主表共享 `Arc`、构图期读它零额外内存；
/// 精确比较只付 O(dims) 构图算力，查询热路径仍走 1 bit 码。
/// EN: RaBitQ stored×stored **deliberately uses exact raw-vector distances**
/// instead of the arcsine estimator over the two code words: the recall gate
/// measured that 1-bit stored-pair estimates distort badly on non-zero-mean
/// (all-positive) data (1024-d L2 candidate recall stayed at 0.9 even with a
/// 512 beam), collapsing the graph built from them. `source` already shares its
/// `Arc` with the main store, so reading it costs no extra memory during
/// construction; exact comparisons pay only O(dims) build-time compute, while
/// the query hot path still runs on the 1-bit codes.
#[inline]
fn metric_pair_stored(metric: DistanceMetric, a: GraphVectorRef<'_>, b: GraphVectorRef<'_>) -> f32 {
    match (a, b) {
        (GraphVectorRef::Full(a), GraphVectorRef::Full(b)) => metric_pair(metric, a, b),
        (GraphVectorRef::Int8(a), GraphVectorRef::Int8(b)) => distance_stored(metric, a, b),
        (GraphVectorRef::RabitQ { source: a, .. }, GraphVectorRef::RabitQ { source: b, .. }) => {
            distance_between(a, b, metric)
        }
        (GraphVectorRef::Full(a), GraphVectorRef::Int8(b)) => distance_query(metric, a, b),
        (GraphVectorRef::Int8(a), GraphVectorRef::Full(b)) => distance_query(metric, b, a),
        _ => {
            debug_assert!(false, "mixed stored graph quantization");
            f32::INFINITY
        }
    }
}

/// 零范数阈值带：`norm² ≤ f32::EPSILON`（即 `norm ≲ 3.45e-4`）内的向量不做单位化，
/// 原样保留（全 0 输入仍为全 0）。图路径（[`normalize_for_cosine`]）与 HII L0 写入/
/// 查询归一化共用此**阈值常量**，消除阈值漂移（WHY: 常量分叉会让同一向量在暴力路径
/// 被单位化、在图路径保持原值，距离口径不一致）。
///
/// 注意共享常量只消除「阈值」漂移，不消除「累加精度」漂移：L0 侧 norm² 用 f64 累加，
/// 图路径仍用 f32 累加，真实 norm² 落在阈值边缘一个 f32 ulp 内的向量，两侧可能给出
/// 相反判定（带宽 `norm ≈ 3.45e-4`，实际影响可忽略）。该残余窗口随图路径 f32→f64
/// 统一一并消除；图路径改动会影响已建图数值，需走 recall/性能门禁，故刻意延后。
/// EN: Shared zero-norm band (`norm² ≤ f32::EPSILON`, norm ≲ 3.45e-4): vectors inside
/// the band are kept untouched. Sharing the *constant* removes threshold drift between
/// the graph path and HII L0. It does NOT remove accumulation drift — L0 accumulates
/// norm² in f64 while the graph path still uses f32, so within one f32 ulp of the band
/// edge the two can still classify a vector differently (a negligible window around
/// norm ≈ 3.45e-4). The residual window resolves with the deliberately deferred
/// graph-path f32→f64 unification, which needs the recall/perf gate.
/// 可见性：模块私有 + `lib.rs` 的 `#[doc(hidden)] pub use` 再导出（P4.5.0 恢复
/// 属性测试要用**同一个**常量做分类判定，按值复刻会在阈值变更时静默失配）。
pub const ZERO_NORM_BAND_NORM2: f32 = f32::EPSILON;

/// 查询侧的余弦归一化：始终产出 f32（查询向量不量化，保证非对称量化精度）。
/// 非余弦度量原样拷贝；零范数带（`norm² ≤ ZERO_NORM_BAND_NORM2`）保持原值。
/// EN: Query-side cosine normalization; always stays f32 (queries are never
/// quantized, preserving asymmetric-quantization accuracy). Non-cosine metrics
/// copy verbatim; the zero-norm band (`norm² ≤ ZERO_NORM_BAND_NORM2`) is kept.
#[inline]
fn normalize_query_for_cosine(metric: DistanceMetric, v: &[f32]) -> Vec<f32> {
    if !matches!(metric, DistanceMetric::Cosine) {
        return v.to_vec();
    }
    let norm2: f32 = v.iter().map(|x| x * x).sum();
    if norm2 <= ZERO_NORM_BAND_NORM2 {
        return v.to_vec();
    }
    let inv = norm2.sqrt().recip();
    v.iter().map(|x| x * inv).collect()
}

/// 存储侧的余弦归一化 + dtype 量化：先按 f32 归一化，再一次性下转到图 dtype。
/// 归一化后再量化可让单位范数在 f16 上保持（误差仅来自分量舍入）。
/// EN: Storage-side cosine normalization plus dtype quantization: normalize in
/// f32 first, then down-convert once to the graph dtype. Normalizing before
/// quantization preserves the unit norm (error comes only from component rounding).
#[inline]
fn normalize_for_cosine(
    metric: DistanceMetric,
    dtype: VectorDtype,
    v: &[f32],
) -> Arc<StoredVector> {
    let normalized = normalize_query_for_cosine(metric, v);
    Arc::new(StoredVector::from_f32(normalized, dtype))
}

/// 构图查询：量化图专用（非对称量化下查询侧始终保持 f32）。int8 与 RaBitQ
/// 都携带度量预处理后的 f32 原向量（余弦已归一化）。
///
/// RaBitQ 的构图**不量化查询侧**：构图期查询×存储的距离走精确 f32 计算
/// （原因见 [`metric_pair_stored`] 的召回门结论），码只在检索热路径使用。
/// 因此这里只需保留 f32 查询本身，旋转结果仅在编码时一次性使用。
/// EN: Build query, only populated for quantized graphs (queries always stay
/// f32 under asymmetric quantization). Both int8 and RaBitQ carry the
/// metric-preprocessed f32 vector (normalized under cosine).
///
/// RaBitQ construction **does not quantize the query side**: build-time
/// query×stored distances are exact f32 (see the recall-gate rationale on
/// [`metric_pair_stored`]); codes are used only on the search hot path. So this
/// only keeps the f32 query itself — the rotation is used once during encoding.
#[derive(Debug, Clone)]
pub(crate) struct PreparedGraphQuery {
    /// 构图查询的 f32 分量（余弦度量下已归一化）。
    /// EN: f32 build-query components (normalized under cosine).
    pub(crate) data: Arc<[f32]>,
}

impl PreparedGraphQuery {
    /// 构图查询：度量预处理后的 f32 原向量。
    /// EN: Build query: the metric-preprocessed f32 vector.
    fn exact(data: Arc<[f32]>) -> Self {
        Self { data }
    }
}

/// 插入准备的产物：主存储句柄 + 图侧向量 + 构图查询。
///
/// `graph_query` 只在量化路径非空：图内查询必须保持 f32（非对称量化），余弦
/// 度量下取归一化后的值；Full 路径的查询直接借用图向量自身，无需单独副本。
///
/// EN: Output of insert preparation: main-store handle + graph-side vector +
/// graph query. `graph_query` is only populated on quantized paths: in-graph
/// queries must stay f32 (asymmetric quantization), normalized under cosine;
/// the Full path borrows the graph vector itself and needs no extra copy.
pub(crate) struct PreparedInsert {
    /// 主存储句柄（原始 f32/f16 值，用于精确 rescore 与导出）。
    /// EN: Main-store handle (raw f32/f16 values for exact rescoring and export).
    pub(crate) main: Arc<StoredVector>,
    /// 图侧向量（可能已量化/归一化）。
    /// EN: Graph-side vector (possibly quantized/normalized).
    pub(crate) graph: GraphVector,
    /// 仅量化路径：构图规划用的 f32 查询（余弦为归一化后的口径）；Full 为 None。
    /// EN: Quantized paths only: the f32 build-planning query (normalized under
    /// cosine); None on the Full path.
    pub(crate) graph_query: Option<PreparedGraphQuery>,
}

/// 插入前把原始 f32 向量准备成主存储与图侧两套表示，一次量化、两处共享：
/// - Full 非余弦：两侧共享**同一个** `Arc`（零额外拷贝）；
/// - Full 余弦：主存储保留原始向量（dtype 量化），图侧持归一化向量——主存储
///   保留原始分量是既有语义（恢复等价性口径依赖它），因此必须拆成两份；
/// - int8：图侧码位独立分配（仿射量化后的归一化向量），`source` 与主存储共享
///   同一 `Arc`——精确 rescore 与快照恢复都要拿到原始值；
/// - RaBitQ：图侧只存 1 bit/dim 符号码 + `alpha`/`norm`，`source` 同样与主存储
///   共享；构图期比较走精确 f32（查询侧不量化），编码只旋转一次。
///
/// EN: Prepare a raw f32 vector into main-store and graph representations,
/// quantizing once and sharing where possible:
/// - Full non-cosine: both sides share the **same** `Arc` (no extra copy).
/// - Full cosine: the main store keeps the raw vector (dtype-quantized) while
///   the graph holds the normalized one — raw components in the main store are
///   the established semantics (recovery-equivalence tests rely on it), so two
///   allocations are required.
/// - int8: graph codes are separately allocated (affine-quantized normalized
///   vector) while `source` shares the main-store `Arc` — exact rescoring and
///   snapshot restore both need the raw values.
/// - RaBitQ: the graph stores only 1 bit/dim sign codes plus `alpha`/`norm`,
///   `source` again shares the main-store `Arc`, build-time comparisons run on
///   exact f32 (the query side is never quantized), and encoding rotates once.
pub(crate) fn prepare_insert_stored(
    metric: DistanceMetric,
    dtype: VectorDtype,
    quantization: Quantization,
    vector: Vec<f32>,
) -> PreparedInsert {
    match quantization {
        Quantization::None => {
            if matches!(metric, DistanceMetric::Cosine) {
                // 先归一化（只借用），再把原始 `vector` 移动进主存储：归一化本身会分配
                // 新缓冲区，主存储无需再 clone 一份全量向量。
                // EN: Normalize first (borrows only), then move the raw `vector` into
                // the main store: normalization allocates its own buffer, so the
                // main store no longer needs a full-vector clone.
                let graph = normalize_for_cosine(metric, dtype, &vector);
                let main = Arc::new(StoredVector::from_f32(vector, dtype));
                PreparedInsert {
                    main: Arc::clone(&main),
                    graph: GraphVector::Full(graph),
                    graph_query: None,
                }
            } else {
                let shared = Arc::new(StoredVector::from_f32(vector, dtype));
                PreparedInsert {
                    main: Arc::clone(&shared),
                    graph: GraphVector::Full(shared),
                    graph_query: None,
                }
            }
        }
        Quantization::Int8 => {
            // 余弦必须「先归一化、后量化」：反量化向量的范数仍近似 1，图内
            // `1 - dot` 口径才成立。非余弦原样拷贝，兼作构图查询。
            // EN: Cosine must normalize *before* quantizing so dequantized vectors
            // stay approximately unit-norm and the in-graph `1 - dot` contract
            // holds. Non-cosine copies verbatim and reuses that copy as the build
            // query.
            let graph_f32 = normalize_query_for_cosine(metric, &vector);
            let codes = Arc::new(Int8Vector::quantize(&graph_f32));
            let source = Arc::new(StoredVector::from_f32(vector, dtype));
            PreparedInsert {
                main: Arc::clone(&source),
                graph: GraphVector::Int8 { codes, source },
                graph_query: Some(PreparedGraphQuery::exact(Arc::from(graph_f32))),
            }
        }
        Quantization::RaBitQ => {
            // 与 int8 同口径：余弦先归一化、后编码；非余弦原样。编码用
            // `prepare_query` 的旋转结果；构图查询保留预处理后的 f32 原向量
            // ——构图期比较走精确距离，不量化查询侧。
            // EN: Same contract as int8 — cosine normalizes before encoding,
            // other metrics pass the raw vector. Encoding consumes
            // `prepare_query`'s rotation; the build query keeps the preprocessed
            // f32 vector because build-time comparisons are exact, not quantized.
            let graph_f32 = normalize_query_for_cosine(metric, &vector);
            let (rotated, norm) = rabitq::prepare_query(&graph_f32);
            let codes = Arc::new(RaBitQVector::from_rotated_unit(
                &rotated,
                norm,
                graph_f32.len(),
            ));
            let source = Arc::new(StoredVector::from_f32(vector, dtype));
            PreparedInsert {
                main: Arc::clone(&source),
                graph: GraphVector::RabitQ { codes, source },
                graph_query: Some(PreparedGraphQuery::exact(Arc::from(graph_f32))),
            }
        }
    }
}

/// 搜索入口预处理的查询：Plain 为度量预处理后的 f32 向量（未量化与 int8 图共用）；
/// RaBitQ 为补 0、旋转、单位化后的查询 + 原始范数。
/// EN: Query preprocessed at the search entrance: `Plain` is the metric-preprocessed
/// f32 vector (shared by unquantized and int8 graphs); `RabitQ` holds the padded,
/// rotated, unitized query plus its raw norm.
enum PreparedQuery {
    Plain(Vec<f32>),
    RabitQ(Vec<f32>, f32),
}

impl PreparedQuery {
    /// 按图量化方式借用为遍历上下文；形态与量化方式不符只可能来自内部 bug。
    /// EN: Borrows as the traversal context for the graph quantization; a shape
    /// mismatch can only come from an internal bug.
    fn context(&self, quantization: Quantization) -> QueryContext<'_> {
        match (self, quantization) {
            (PreparedQuery::Plain(v), Quantization::None) => {
                QueryContext::Exact(VectorRef::F32(v.as_slice()))
            }
            (PreparedQuery::Plain(v), Quantization::Int8) => {
                QueryContext::Int8(VectorRef::F32(v.as_slice()))
            }
            (PreparedQuery::RabitQ(rotated, norm), Quantization::RaBitQ) => {
                QueryContext::RabitQ(RaBitQQuery {
                    rotated: rotated.as_slice(),
                    norm: *norm,
                })
            }
            _ => {
                debug_assert!(false, "prepared query does not match graph quantization");
                QueryContext::Exact(VectorRef::F32(&[]))
            }
        }
    }
}

/// 搜索入口的查询预处理：先按度量归一化（余弦），RaBitQ 图再补 0 + 旋转 +
/// 单位化。O(D log D) 只在这里发生一次，图内邻接循环只做逐邻居点积。
/// EN: Search-entrance query preprocessing: metric normalization (cosine) first,
/// then pad + rotate + unitize for RaBitQ graphs. The O(D log D) pass happens
/// exactly once here; the in-graph adjacency loop only runs per-neighbor dots.
fn prepare_search_query(
    metric: DistanceMetric,
    quantization: Quantization,
    query: &[f32],
) -> PreparedQuery {
    let graph_side = normalize_query_for_cosine(metric, query);
    if quantization.is_rabitq() {
        let (rotated, norm) = rabitq::prepare_query(&graph_side);
        PreparedQuery::RabitQ(rotated, norm)
    } else {
        PreparedQuery::Plain(graph_side)
    }
}

/// 每层连接数上限。乘数由 `set_layer_multipliers()` 设置，默认 L0=2x / 上层=0.5x。
/// Per-layer max degree. Multipliers set via `set_layer_multipliers()`; defaults: L0=2x, upper=0.5x.
///
/// 高 QPS（旧行为）: 设置 L0=1.0, upper=0.5。
/// 上层对齐（L0=2M/upper=M）: 设置 L0=2.0, upper=1.0。
fn layer_m(m: usize, layer: usize) -> usize {
    let mults = effective_layer_multipliers();
    if layer == 0 {
        ((m as f64 * mults.0).round() as usize).max(2)
    } else {
        ((m as f64 * mults.1).round() as usize).max(1)
    }
}

/// 当前进程生效的层乘数（未显式设置时为默认 2.0/0.5）。
/// 图快照记录该值，恢复时比对——`link_base` 布局解释依赖它。
/// EN: Effective layer multipliers for this process (defaults 2.0/0.5).
/// Snapshots persist this; restore compares it since `link_base` layout
/// interpretation depends on it.
pub(crate) fn effective_layer_multipliers() -> (f64, f64) {
    LAYER_MULTS.get().copied().unwrap_or((2.0_f64, 0.5_f64))
}

/// 全局层乘数缓存（`set_layer_multipliers` 写入，`layer_m` 读取）。
/// Global multiplier cache: written by `set_layer_multipliers`, read by `layer_m`.
static LAYER_MULTS: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();

/// 设置全局层连接数乘数（进程启动时调用一次，必须在首次 `layer_m` 调用前完成）。
/// Set global per-layer degree multipliers. Call once at startup, before any `layer_m` usage.
///
/// `l0_mult` — level 0 乘数 (默认 2.0, 召回优先且兼容旧快照; 1.0 = 高 QPS)
/// `upper_mult` — 上层乘数 (默认 0.5)
///
/// # Examples
/// ```ignore
/// // 默认 / 召回优先: L0=2*m, upper=m/2
/// set_layer_multipliers(2.0, 0.5);
/// // 高 QPS (旧行为): L0=m, upper=m/2
/// set_layer_multipliers(1.0, 0.5);
/// ```
#[doc(hidden)]
pub fn set_layer_multipliers(l0_mult: f64, upper_mult: f64) {
    if l0_mult <= 0.0 || upper_mult <= 0.0 {
        return;
    }
    let _ = LAYER_MULTS.set((l0_mult, upper_mult));
}

/// 单层的邻接槽容量 = `layer_m` + 松弛位（临时 push 后 prune 不会溢出）。
/// EN: Per-layer cap = `layer_m` + slack so push-then-prune never overflows the flat block.
#[inline]
pub(crate) fn link_cap(m: usize, l: usize) -> usize {
    layer_m(m, l) + NEIGHBOR_LINK_SLACK
}

/// 连续块内第 `l` 层在节点块中的起始 offset（0-based，`l` 之前各层 `link_cap` 之和）。
/// EN: In-node byte offset: sum of prior layers' `link_cap`.
fn layer_start_offset(m: usize, l: usize) -> usize {
    (0..l).map(|i| link_cap(m, i)).sum()
}

/// 为「最高到 `max_level` 层」的节点分配整段 `link_arena` 时的槽总长度（各层 `link_cap` 之和）。
/// EN: Total `VectorId` slots for levels `0..=max_level` in the flat per-node block.
pub(crate) fn block_size_for_node(m: usize, max_level: usize) -> usize {
    (0..=max_level).map(|i| link_cap(m, i)).sum()
}

fn random_level(m: usize, num_layers: usize, rng: &mut impl Rng) -> usize {
    let m = m.max(2) as f64;
    let ml = 1.0 / m.ln();
    let max_l = num_layers
        .saturating_sub(1)
        .min(MAX_HNSW_LAYERS.saturating_sub(1));
    let draw = -rng.gen::<f64>().ln() * ml;
    (draw.floor() as usize).min(max_l)
}

/// 两个搜索变体共享的入口守卫：ef 边界 → ep 层级 → ep 向量,任一不满足返回
/// None(调用方降级为空结果);通过时完成 visited 层初始化并返回 ep 向量。
/// 单源维护——两变体的守卫语义(含日志/断言)不得各自漂移。
///
/// 「ep 缺层 = 腐败信号」的前提由调用方契约保证:所有合法调用方
/// (search/search_with_filter/insert 下降链/plan)已把下降区间钳制到 ep 层级
/// 跨度,故此处命中即入口点/连边不变量真被破坏。ep 向量缺失同理:对称剪枝
/// 维持「链接目标必有向量」,命中即半更新图(锁中毒恢复)——跳过优于 panic
/// (持锁 panic 会毒化 RwLock,把单点图异常放大成服务级故障)。
///
/// EN: Shared entry guard for both search variants: ef bound → ep level → ep
/// vector; returns None when the caller must degrade to an empty result, else
/// primes the visited tracker and returns the ep vector. Single source — the
/// guard semantics (log/assert) must not drift between variants.
///
/// The "ep lacks layer ⇒ corruption" premise is upheld by caller contract:
/// every legitimate caller (search / search_with_filter / insert descent /
/// plan) clamps its descent to the ep's level span, so hitting this branch
/// means the entry-point/linking invariant was truly violated. Same for a
/// missing ep vector: symmetric pruning maintains "link target ⇒ vector
/// exists", so a miss means a half-updated graph (poison recovery) — skipping
/// beats panicking (a panic under lock poisons the RwLock, escalating a local
/// glitch into a service-wide outage).
fn search_entry_guard<'a>(
    visited: &mut HnswVisitTracker,
    vectors: &'a AHashMap<VectorId, GraphVector>,
    nodes: &AHashMap<VectorId, HnswNode>,
    ep: VectorId,
    ef: usize,
    layer: usize,
) -> Option<&'a GraphVector> {
    if ef == 0 {
        return None;
    }
    let ep_has_layer = nodes
        .get(&ep)
        .map(|n| layer < n.level_count())
        .unwrap_or(false);
    if !ep_has_layer {
        // 冷路径,诊断日志每查询至多一行;debug 构建断言暴露,release 降级。
        // EN: Cold path — at most one log line per query; debug builds assert,
        // release degrades to an empty result.
        tracing::error!(
            ep,
            layer,
            "hnsw: search ep lacks layer — link invariant violated"
        );
        debug_assert!(false, "search ep {ep} lacks layer {layer}");
        return None;
    }
    visited.begin_layer_search();
    let _ = visited.try_visit(ep);
    match vectors.get(&ep) {
        Some(v) => Some(v),
        None => {
            tracing::error!(ep, "hnsw: search ep vector missing — half-updated graph");
            debug_assert!(false, "search ep {ep} vector missing");
            None
        }
    }
}

/// 由调用方复用的 [`HnswVisitTracker`]；`begin_layer_search` 取代全表 clear。
/// EN: Reused [`HnswVisitTracker`]; `begin_layer_search` replaces a full set clear.
// 热路径内部函数：参数即图的拆解视图，打包成结构体只会增加间接层。
#[allow(clippy::too_many_arguments)]
fn search_layer_inner(
    visited: &mut HnswVisitTracker,
    vectors: &AHashMap<VectorId, GraphVector>,
    link_arena: &[VectorId],
    nodes: &AHashMap<VectorId, HnswNode>,
    m: usize,
    query: QueryContext<'_>,
    ep: VectorId,
    ef: usize,
    layer: usize,
    metric: DistanceMetric,
) -> Vec<(VectorId, f32)> {
    // 入口守卫(ef/ep 层级/ep 向量)单源在 search_entry_guard——ep 必须持有该层,
    // 否则被调用方(insert 下降链/连接阶段)当作合法邻居连边,产生单边悬空边。
    // EN: Entry guard (ef / ep level / ep vector) lives in search_entry_guard —
    // ep must own the layer, or callers (insert descent/connect) would link it
    // as a neighbor there, creating a one-sided dangling edge.
    let ep_vec = match search_entry_guard(visited, vectors, nodes, ep, ef, layer) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let dist_ep = metric_pair_query(metric, query, ep_vec.as_graph_ref());
    let cap_hint = ef.saturating_mul(4).max(64);
    let mut candidates = BinaryHeap::with_capacity(cap_hint);
    candidates.push(Cd {
        dist: dist_ep,
        id: ep,
    });
    let mut w = BinaryHeap::with_capacity(cap_hint);
    w.push(Wd {
        dist: dist_ep,
        id: ep,
    });

    while let Some(Cd {
        dist: dist_c,
        id: c,
    }) = candidates.pop()
    {
        let furthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
        if dist_c > furthest && w.len() >= ef {
            break;
        }

        let node = match nodes.get(&c) {
            Some(n) => n,
            None => {
                // 邻接表引用的节点已不存在 ⇒ 悬空链接(不变量曾被破坏)。循环内
                // 不打日志防刷屏,debug 构建断言暴露。
                // EN: Adjacency references a gone node ⇒ dangling link (invariant
                // was violated). No log inside the loop (flood risk); debug builds assert.
                debug_assert!(false, "candidate node {c} missing — dangling link");
                continue;
            }
        };
        if layer >= node.level_count() {
            continue;
        }
        let nbrs = match node.link_active_ref(link_arena, m, layer) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        for &e in nbrs {
            if visited.try_visit(e) {
                let e_vec = match vectors.get(&e) {
                    Some(v) => v,
                    None => {
                        // 悬空链接(邻居向量已删): 同上,循环内只断言不打日志。
                        // EN: Dangling link (neighbor vector gone): assert only,
                        // no log inside the loop.
                        debug_assert!(false, "neighbor {e} vector missing — dangling link");
                        continue;
                    }
                };
                let dist_e = metric_pair_query(metric, query, e_vec.as_graph_ref());
                let furthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if dist_e < furthest || w.len() < ef {
                    candidates.push(Cd {
                        dist: dist_e,
                        id: e,
                    });
                    if w.len() < ef {
                        w.push(Wd {
                            dist: dist_e,
                            id: e,
                        });
                    } else if dist_e < furthest {
                        w.pop();
                        w.push(Wd {
                            dist: dist_e,
                            id: e,
                        });
                    }
                }
            }
        }
    }

    let mut out: Vec<(VectorId, f32)> = w.into_iter().map(|wd| (wd.id, wd.dist)).collect();
    out.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

/// 与 `search_layer_inner` 相同，但在遍历邻居时跳过不在 filter_set 中的节点。
/// Same as `search_layer_inner`, but skips neighbors not in filter_set during traversal.
#[allow(clippy::too_many_arguments)]
fn search_layer_inner_filtered(
    visited: &mut HnswVisitTracker,
    vectors: &AHashMap<VectorId, GraphVector>,
    link_arena: &[VectorId],
    nodes: &AHashMap<VectorId, HnswNode>,
    m: usize,
    query: QueryContext<'_>,
    ep: VectorId,
    ef: usize,
    layer: usize,
    metric: DistanceMetric,
    filter_set: &std::collections::HashSet<VectorId>,
) -> Vec<(VectorId, f32)> {
    // 入口守卫单源在 search_entry_guard(与 search_layer_inner 共用)。
    // EN: Entry guard lives in search_entry_guard (shared with search_layer_inner).
    let ep_vec = match search_entry_guard(visited, vectors, nodes, ep, ef, layer) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let dist_ep = metric_pair_query(metric, query, ep_vec.as_graph_ref());
    let cap_hint = ef.saturating_mul(4).max(64);
    let mut candidates = BinaryHeap::with_capacity(cap_hint);
    candidates.push(Cd {
        dist: dist_ep,
        id: ep,
    });
    let mut w = BinaryHeap::with_capacity(cap_hint);
    w.push(Wd {
        dist: dist_ep,
        id: ep,
    });

    while let Some(Cd {
        dist: dist_c,
        id: c,
    }) = candidates.pop()
    {
        let furthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
        if dist_c > furthest && w.len() >= ef {
            break;
        }
        let node = match nodes.get(&c) {
            Some(n) => n,
            None => {
                // 邻接表引用的节点已不存在 ⇒ 悬空链接(不变量曾被破坏)。循环内
                // 不打日志防刷屏,debug 构建断言暴露。
                // EN: Adjacency references a gone node ⇒ dangling link (invariant
                // was violated). No log inside the loop (flood risk); debug builds assert.
                debug_assert!(false, "candidate node {c} missing — dangling link");
                continue;
            }
        };
        if layer >= node.level_count() {
            continue;
        }
        let nbrs = match node.link_active_ref(link_arena, m, layer) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        for &e in nbrs {
            // 跳过不在 filter_set 中的邻居 — 图内过滤（参考常见 payload index 实现）
            // EN: Skip neighbors not in filter_set — in-graph filtering (cf. common payload-index implementations).
            if !filter_set.contains(&e) {
                continue;
            }
            if visited.try_visit(e) {
                let e_vec = match vectors.get(&e) {
                    Some(v) => v,
                    None => {
                        // 悬空链接(邻居向量已删): 同上,循环内只断言不打日志。
                        // EN: Dangling link (neighbor vector gone): assert only,
                        // no log inside the loop.
                        debug_assert!(false, "neighbor {e} vector missing — dangling link");
                        continue;
                    }
                };
                let dist_e = metric_pair_query(metric, query, e_vec.as_graph_ref());
                let furthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if dist_e < furthest || w.len() < ef {
                    candidates.push(Cd {
                        dist: dist_e,
                        id: e,
                    });
                    if w.len() < ef {
                        w.push(Wd {
                            dist: dist_e,
                            id: e,
                        });
                    } else if dist_e < furthest {
                        w.pop();
                        w.push(Wd {
                            dist: dist_e,
                            id: e,
                        });
                    }
                }
            }
        }
    }

    let mut out: Vec<(VectorId, f32)> = w.into_iter().map(|wd| (wd.id, wd.dist)).collect();
    out.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

/// hnswlib 风格启发式 2：邻居多样性。
/// EN: hnswlib-style heuristic 2: diversity among neighbors.
fn select_neighbors_heuristic_from_sorted(
    vectors: &AHashMap<VectorId, GraphVector>,
    w: &[(VectorId, f32)],
    m: usize,
    metric: DistanceMetric,
) -> Vec<VectorId> {
    if m == 0 || w.is_empty() {
        return Vec::new();
    }
    let mut selected: Vec<VectorId> = Vec::with_capacity(m);
    let mut sel_vecs: Vec<GraphVectorRef<'_>> = Vec::with_capacity(m);
    for &(eid, d_qe) in w.iter() {
        if selected.len() >= m {
            break;
        }
        let ev = match vectors.get(&eid) {
            Some(v) => v.as_graph_ref(),
            None => continue,
        };
        let mut ok = true;
        for rv in sel_vecs.iter().copied() {
            if metric_pair_stored(metric, ev, rv) < d_qe {
                ok = false;
                break;
            }
        }
        if ok {
            selected.push(eid);
            sel_vecs.push(ev);
        }
    }
    // 实验开关：保留被多样性启发式剪掉的近邻，补足到 m（HNSW 论文
    // keepPrunedConnections）。不改变候选顺序，只增加边数/连通性。
    // EN: Optional keepPrunedConnections: fill to m with the closest candidates
    // rejected by the diversity heuristic. More edges/connectivity, same order.
    if selected.len() < m && keep_pruned_enabled() {
        for &(eid, _) in w.iter() {
            if selected.len() >= m {
                break;
            }
            if vectors.contains_key(&eid) && !selected.contains(&eid) {
                selected.push(eid);
            }
        }
    }
    selected
}

/// `HOUYI_HNSW_KEEP_PRUNED=1` 时启用 keepPrunedConnections（进程内一次读取）。
/// EN: Read once per process; off by default, experimental knob.
fn keep_pruned_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HOUYI_HNSW_KEEP_PRUNED").ok().as_deref() == Some("1"))
}

/// 在 `link_arena` 中尝试加入邻接，若已存在同 id 则不改。
/// EN: Append neighbor in flat link slot; no-op on duplicate.
fn try_push_link_unique(
    link_arena: &mut [VectorId],
    n: &mut HnswNode,
    m: usize,
    layer: usize,
    id: VectorId,
) {
    if layer >= n.level_count() {
        return;
    }
    let st = n.layer_start(m, layer);
    let l = n.layer_len[layer] as usize;
    let cap = link_cap(m, layer);
    for i in 0..l {
        if link_arena[st + i] == id {
            return;
        }
    }
    if l >= cap {
        debug_assert!(false, "HNSW flat link overflow (cap too small)");
        return;
    }
    link_arena[st + l] = id;
    n.layer_len[layer] = (l + 1) as u16;
}

// 热路径内部函数：参数即图的拆解视图，打包成结构体只会增加间接层。
// EN: Hot-path internal fn: args are a decomposed graph view; a struct would only add indirection.
#[allow(clippy::too_many_arguments)]
fn prune_neighbors(
    vectors: &AHashMap<VectorId, GraphVector>,
    link_arena: &mut [VectorId],
    nodes: &mut AHashMap<VectorId, HnswNode>,
    m: usize,
    nid: VectorId,
    layer: usize,
    max_deg: usize,
    metric: DistanceMetric,
) {
    let nbr_len = match nodes.get(&nid) {
        Some(n) if layer < n.level_count() => n.layer_len[layer] as usize,
        _ => return,
    };
    if nbr_len <= max_deg {
        return;
    }
    // scored 用于启发式选边（需向量算距离，filter_map 滤掉向量缺失的悬空邻居）；
    // old_neighbors 保留原始邻接表——对称清理必须遍历它而非 scored，否则悬空
    // kid 的正向边被写回丢弃、反向边却存活，在半更新图上重现单边边。
    // EN: `scored` feeds the heuristic (needs vectors, so filter_map drops dangling
    // neighbors); `old_neighbors` keeps the raw adjacency — symmetric cleanup must
    // iterate it, not `scored`, or a dangling kid loses its forward edge at
    // write-back while its back-edge survives, recreating a one-sided edge.
    let (scored, old_neighbors): (Vec<(VectorId, f32)>, Vec<VectorId>) = {
        let host = match vectors.get(&nid) {
            Some(v) => v.as_graph_ref(),
            None => return,
        };
        let n = match nodes.get(&nid) {
            Some(n) if layer < n.level_count() => n,
            _ => return,
        };
        let neighbor_ids = match n.link_active_ref(link_arena, m, layer) {
            Some(s) if !s.is_empty() => s,
            _ => return,
        };
        let old: Vec<VectorId> = neighbor_ids.to_vec();
        let scored = neighbor_ids
            .iter()
            .filter_map(|&id| {
                vectors
                    .get(&id)
                    .map(|on| (id, metric_pair_stored(metric, host, on.as_graph_ref())))
            })
            .collect();
        (scored, old)
    };
    let mut scored = scored;
    scored.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let keep = select_neighbors_heuristic_from_sorted(vectors, &scored, max_deg, metric);
    if let Some(n) = nodes.get_mut(&nid) {
        if layer < n.level_count() {
            let st = n.layer_start(m, layer);
            let cap = link_cap(m, layer);
            for (i, &kid) in keep.iter().enumerate() {
                if i >= cap {
                    break;
                }
                link_arena[st + i] = kid;
            }
            n.layer_len[layer] = (keep.len().min(cap)) as u16;
        }
    }
    // 对称剪枝（边对称不变量的唯一维护点）：启发式每丢弃一条 nid→kid 边，
    // 必须同步摘除 kid→nid 反向边。remove_node 的「只扫被删节点自身邻接」
    // 清理依赖「A→B 存在 ⟺ B→A 存在」——此处若单向丢边，反向边即成悬空
    // 链接：remove_node(kid) 清不到它，其后 search 遍历到 kid 时会索引已被
    // 删除的 vectors[&nid] 直接 panic（线上锁中毒连环炸的源头）。
    // EN: Symmetric pruning — the sole maintenance point of the edge-symmetry
    // invariant. Every heuristic-dropped nid→kid edge must also drop the
    // kid→nid back-edge. remove_node's "only scan the victim's own lists"
    // cleanup relies on A→B ⟺ B→A; a one-sided drop leaves a dangling
    // back-edge that remove_node(kid) cannot reach, and a later search
    // traversing kid indexes the deleted vectors[&nid] and panics (the
    // source of the production poisoned-lock cascade).
    for &kid in &old_neighbors {
        if keep.contains(&kid) {
            continue;
        }
        if let Some(kn) = nodes.get_mut(&kid) {
            remove_link_id(link_arena, kn, m, layer, nid);
        }
    }
}

// 热路径内部函数：参数即图的拆解视图，打包成结构体只会增加间接层。
// EN: Hot-path internal fn: args are a decomposed graph view; a struct would only add indirection.
#[allow(clippy::too_many_arguments)]
fn add_bidirectional_edge(
    vectors: &AHashMap<VectorId, GraphVector>,
    link_arena: &mut [VectorId],
    nodes: &mut AHashMap<VectorId, HnswNode>,
    m: usize,
    q: VectorId,
    e: VectorId,
    layer: usize,
    max_deg: usize,
    metric: DistanceMetric,
) {
    if q == e {
        return;
    }
    // 双端层级预检：任一端缺该层 ⇒ 无法建立对称边，整体跳过。不得先推正向
    // 边再在反向块里早退——那会留下单边边（remove_node 扫不到的悬空链接）。
    // EN: Pre-check both endpoints' levels: if either lacks the layer, no
    // symmetric edge is possible — skip entirely. Pushing the forward edge
    // first and early-returning in the reverse block would leave a one-sided
    // edge (a dangling link remove_node cannot reach).
    let both_have_layer = [&q, &e].iter().all(|x| {
        nodes
            .get(x)
            .map(|n| layer < n.level_count())
            .unwrap_or(false)
    });
    if !both_have_layer {
        // 任一端缺层还想连边 ⇒ 上游(入口点/搜索过滤)不变量曾被破坏。
        // 静默跳过保 release 安全,debug 构建断言暴露。
        // EN: Linking an endpoint that lacks the layer ⇒ an upstream invariant
        // (entry-point / search filtering) was violated. Skip silently for
        // release safety; debug builds assert.
        debug_assert!(
            false,
            "link {q}<->{e} at layer {layer}: endpoint lacks layer"
        );
        return;
    }
    {
        let nq = match nodes.get_mut(&q) {
            Some(n) if layer < n.level_count() => n,
            _ => return,
        };
        try_push_link_unique(link_arena, nq, m, layer, e);
    }
    {
        let ne = match nodes.get_mut(&e) {
            Some(n) if layer < n.level_count() => n,
            _ => return,
        };
        try_push_link_unique(link_arena, ne, m, layer, q);
    }
    prune_neighbors(vectors, link_arena, nodes, m, q, layer, max_deg, metric);
    prune_neighbors(vectors, link_arena, nodes, m, e, layer, max_deg, metric);
}

/// 从 `node` 的 `layer` 中删除一条到 `id` 的边（`swap_remove` 语义）。
/// EN: Remove one edge in flat layer; `swap_remove` semantics.
fn remove_link_id(
    link_arena: &mut [VectorId],
    n: &mut HnswNode,
    m: usize,
    layer: usize,
    id: VectorId,
) {
    if layer >= n.level_count() {
        return;
    }
    let st = n.layer_start(m, layer);
    let l = n.layer_len[layer] as usize;
    for i in 0..l {
        if link_arena[st + i] == id {
            if l > 1 {
                link_arena[st + i] = link_arena[st + l - 1];
            }
            n.layer_len[layer] = (l - 1) as u16;
            return;
        }
    }
}

impl HnswGraph {
    /// 为邻接块向 `link_arena` 申请 `size` 个 `VectorId` 槽，优先复用 `link_free` 同尺寸块。
    /// EN: Allocate `size` `VectorId` slots; reuses a same-size free block when available.
    fn alloc_link_block(&mut self, size: usize) -> usize {
        if size == 0 {
            return 0;
        }
        if let Some(fv) = self.link_free.get_mut(&size) {
            if let Some(b) = fv.pop() {
                return b;
            }
        }
        let base = self.link_arena.len();
        self.link_arena.resize(base + size, LINK_PAD);
        base
    }

    /// 将整块邻接还回空闲表；与 [`alloc_link_block`] 成对出现（见 `remove_node`）。
    /// EN: Return a full node link block; paired with alloc on `remove_node`.
    fn free_link_block(&mut self, base: usize, max_level: usize) {
        let sz = block_size_for_node(self.m, max_level);
        if sz == 0 {
            return;
        }
        self.link_free.entry(sz).or_default().push(base);
    }

    /// 分配带图级 `link_arena` 连续槽的节点（真 Flat-Link）。
    /// EN: Allocate an [`HnswNode`] with a contiguous per-node sub-block in `link_arena`.
    fn new_hnsw_node(&mut self, level: usize) -> HnswNode {
        let m = self.m;
        let sz = block_size_for_node(m, level);
        let base = self.alloc_link_block(sz);
        HnswNode {
            link_base: base,
            max_layer: level as u8,
            layer_len: [0; MAX_HNSW_LAYERS],
        }
    }

    pub fn new(
        dims: usize,
        metric: DistanceMetric,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
        max_layers: usize,
    ) -> Self {
        Self::new_with_dtype(
            dims,
            metric,
            VectorDtype::F32,
            m,
            ef_construction,
            ef_search,
            seed,
            max_layers,
        )
    }

    /// 带存储 dtype 的构造器；`new` 等价于 dtype=f32（保持既有调用方行为不变）。
    /// EN: Constructor with an explicit storage dtype; `new` is the f32 alias that
    /// keeps existing call sites' behavior unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_dtype(
        dims: usize,
        metric: DistanceMetric,
        dtype: VectorDtype,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
        max_layers: usize,
    ) -> Self {
        Self::new_with_dtype_and_quantization(
            dims,
            metric,
            dtype,
            Quantization::None,
            m,
            ef_construction,
            ef_search,
            seed,
            max_layers,
        )
    }

    /// 带存储 dtype 与图侧量化的构造器（P2）；旧构造器等价于 quantization=None。
    /// EN: Constructor with an explicit storage dtype and graph-side quantization
    /// (P2); legacy constructors are the quantization=None alias.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_dtype_and_quantization(
        dims: usize,
        metric: DistanceMetric,
        dtype: VectorDtype,
        quantization: Quantization,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
        max_layers: usize,
    ) -> Self {
        let num_layers = max_layers.clamp(1, MAX_HNSW_LAYERS);
        Self {
            dims,
            metric,
            dtype,
            quantization,
            m: m.max(2),
            ef_construction: ef_construction.max(1),
            ef_search: ef_search.max(1),
            num_layers,
            seed,
            rng: StdRng::seed_from_u64(seed),
            nodes: AHashMap::new(),
            link_arena: Vec::new(),
            link_free: AHashMap::new(),
            vectors: AHashMap::new(),
            entry_point: None,
            max_layer: 0,
            incomplete_warn_mask: 0,
        }
    }

    /// 图内向量存储 dtype。
    /// EN: Storage dtype of vectors held by the graph.
    pub fn dtype(&self) -> VectorDtype {
        self.dtype
    }

    /// 图侧量化方式。
    /// EN: Graph-side quantization.
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// 图的距离度量（构图/检索共用；调用方据此准备共享插入向量）。
    /// EN: The graph's distance metric (shared by build and search; callers use
    /// it to prepare shareable insert vectors).
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// 测试专用：读取图内某 id 的图侧向量，用于验证主表/图 `Arc` 共享关系、
    /// 余弦归一化口径以及 int8 的 codes/source 分离。
    /// EN: Test-only: read a node's graph vector to verify main/graph `Arc`
    /// sharing, the cosine normalization contract, and int8 codes/source split.
    #[cfg(test)]
    pub(crate) fn graph_vector(&self, id: VectorId) -> Option<&GraphVector> {
        self.vectors.get(&id)
    }

    /// 从快照各段直接重建图（P4.4.0）。调用方（`graph_snapshot::decode_snapshot`）
    /// 已完成全部结构校验；本函数只做与 `new` 相同的取值钳制，不重复校验——
    /// 关键不变量以 debug_assert 钉死，防未来第二个调用方绕过 decode 的校验。
    /// EN: Rebuild a graph from decoded snapshot sections (P4.4.0). The caller
    /// has validated everything; this only applies the same clamps as `new`.
    /// Key invariants are pinned with debug_asserts so a future second caller
    /// that skips decode-time validation fails loudly in debug builds.
    pub(crate) fn restore_from_snapshot(parts: SnapshotGraphParts) -> Self {
        let SnapshotGraphParts {
            dims,
            metric,
            dtype,
            quantization,
            m,
            ef_construction,
            ef_search,
            num_layers,
            seed,
            nodes,
            link_arena,
            link_free,
            vectors,
            entry_point,
            max_layer,
        } = parts;
        debug_assert!(
            m >= 2,
            "decode must reject m below the clamp floor (layout depends on it)"
        );
        debug_assert!(
            entry_point.is_none_or(|ep| nodes.contains_key(&ep)),
            "entry point must reference a restored node"
        );
        debug_assert!(
            nodes.is_empty() == entry_point.is_none(),
            "non-empty graph must carry an entry point (and vice versa)"
        );
        debug_assert!(
            nodes.values().all(
                |n| n.link_base + block_size_for_node(m, n.max_layer as usize) <= link_arena.len()
            ),
            "every node block must stay within link_arena"
        );
        Self {
            dims,
            metric,
            dtype,
            quantization,
            m: m.max(2),
            ef_construction: ef_construction.max(1),
            ef_search: ef_search.max(1),
            num_layers: num_layers.clamp(1, MAX_HNSW_LAYERS),
            seed,
            // rng 状态不序列化：同种子重播种，后续插入的随机层抽取从头开始。
            // EN: rng state is not serialized: reseeding restarts random level draws for later inserts.
            rng: StdRng::seed_from_u64(seed),
            nodes,
            link_arena,
            link_free,
            vectors,
            entry_point,
            max_layer,
            // 恢复后的首次兜底允许告警；位图不随快照持久化。
            // EN: The first sweep after restore may warn; the bitmap is not
            // part of the persisted snapshot.
            incomplete_warn_mask: 0,
        }
    }

    pub fn set_m(&mut self, m: usize) {
        self.m = m.max(2);
    }

    pub fn set_ef_construction(&mut self, ef: usize) {
        self.ef_construction = ef.max(1);
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef.max(1);
    }

    /// 保留给指标 / 未来的压缩钩子。
    /// EN: Reserved for metrics / future compaction hooks.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn entry_point(&self) -> Option<VectorId> {
        self.entry_point
    }

    /// 入口点的全局最高层（导航深度）。
    /// EN: Global max layer of the entry point (navigation depth).
    #[allow(dead_code)]
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    /// 测试用：当前构图束宽（验证调参在合图门释放后生效）。
    /// EN: Test-only: current build beam width (verifies post-gate tuning).
    #[cfg(test)]
    pub(crate) fn ef_construction_for_test(&self) -> usize {
        self.ef_construction
    }

    /// 返回近似边数（无向边按每层一次计；add 中双向计数）。
    /// EN: Returns approximate edge count (undirected edges counted once per layer, both directions in add).
    pub fn edge_count_hint(&self) -> usize {
        let mut c = 0usize;
        for n in self.nodes.values() {
            // 用切片迭代替代下标循环：clippy 1.100 的 needless_range_loop
            // 会拒绝后者，且语义完全等价。
            // EN: Iterate over the slice rather than indexing: clippy 1.100's
            // needless_range_loop rejects the latter; semantics are identical.
            c += n.layer_len[..n.level_count()]
                .iter()
                .map(|&len| len as usize)
                .sum::<usize>();
        }
        c / 2
    }

    pub fn neighbors_layer0(&self, id: VectorId) -> Vec<VectorId> {
        self.neighbors_at_layer(id, 0)
    }

    pub fn neighbors_at_layer(&self, id: VectorId, layer: usize) -> Vec<VectorId> {
        self.nodes
            .get(&id)
            .and_then(|n| n.link_active_ref(&self.link_arena, self.m, layer))
            .map(|s| s.to_vec())
            .unwrap_or_default()
    }

    pub fn node_level(&self, id: VectorId) -> Option<usize> {
        self.nodes.get(&id).map(|n| n.max_layer as usize)
    }

    /// 单次 `insert` / `insert_with_visited` 中 [`HnswVisitTracker`] 的预分配提示（与内部一致）。
    /// EN: Capacity hint for [`HnswVisitTracker`] during insert-time greedy search.
    #[inline]
    pub fn visited_insert_capacity_hint(ef_construction: usize) -> usize {
        ef_construction.max(1).saturating_mul(16).max(128)
    }

    /// Insert one point. Duplicate `id` is ignored.
    /// 对外单点 API；本仓库内批量构图优先 [`insert_with_visited`]。
    /// EN: Public single-point API; in-tree bulk build uses [`insert_with_visited`].
    #[allow(dead_code)] // 供下游 crate 使用；工作区内构图走 `insert_with_visited`，故 lib 构建会误报未用。
    pub fn insert(&mut self, id: VectorId, vector: Vec<f32>) -> usize {
        let mut visited = HnswVisitTracker::with_capacity(Self::visited_insert_capacity_hint(
            self.ef_construction,
        ));
        self.insert_with_visited(id, vector, &mut visited)
    }

    /// 与 [`insert`] 语义相同，但由调用方复用 [`HnswVisitTracker`]，减少批量构图时的每点分配。
    /// EN: Same as [`insert`], but reuses [`HnswVisitTracker`] across points to cut allocations in bulk build.
    pub fn insert_with_visited(
        &mut self,
        id: VectorId,
        vector: Vec<f32>,
        visited: &mut HnswVisitTracker,
    ) -> usize {
        self.insert_with_visited_entry_hint(id, Arc::new(vector), visited, None)
    }

    /// 与 [`insert_with_visited`] 语义相同，但允许调用方提供入口点 hint。
    /// EN: Same as `insert_with_visited`, with optional entry-point hint for faster greedy warm start.
    pub fn insert_with_visited_entry_hint(
        &mut self,
        id: VectorId,
        vector: Arc<Vec<f32>>,
        visited: &mut HnswVisitTracker,
        entry_hint: Option<VectorId>,
    ) -> usize {
        // 廉价早退先于归一化/量化：被拒绝的插入（维度不符、重复 id）不应付出
        // O(dims) 的整向量分配。prepared 路径保留同样的守卫供直接调用方使用。
        // EN: Cheap early-outs before normalization/quantization: rejected inserts
        // (dims mismatch, duplicate id) must not pay an O(dims) full-vector
        // allocation. The prepared path keeps the same guards for direct callers.
        if self.dims > 0 && vector.len() != self.dims {
            return 0;
        }
        if self.nodes.contains_key(&id) {
            return 0;
        }
        // `unwrap_or_clone` 在调用方独占 Arc 时避免整向量拷贝；共享时退回 clone，
        // 与本路径此前 `normalize_for_cosine` 已产生的拷贝等价。
        // EN: `unwrap_or_clone` skips the full-vector copy when the caller owns the
        // Arc; under sharing it falls back to clone, the same cost the previous
        // `normalize_for_cosine` path already paid.
        let PreparedInsert {
            main: _main,
            graph,
            graph_query,
        } = prepare_insert_stored(
            self.metric,
            self.dtype,
            self.quantization,
            Arc::unwrap_or_clone(vector),
        );
        self.insert_prepared_with_visited_entry_hint(
            id,
            graph,
            graph_query.as_ref(),
            visited,
            entry_hint,
        )
    }

    /// 已准备图侧向量（已按图 dtype/量化方式准备、余弦已归一化）的插入路径：
    /// 语义与 [`insert_with_visited_entry_hint`] 相同，但跳过内部归一化，让调用方
    /// 把原始 `Arc` 同时交给主存储（见 `prepare_insert_stored`）。
    ///
    /// 调用方契约：`source_dtype` 等于图 dtype、`quantization` 等于图量化方式；
    /// debug 构建用断言把关，release 下由构造方（`prepare_insert_stored`）保证。
    /// 量化路径必须携带构图查询（非对称量化）；缺失时 debug 断言并在 release
    /// 下退化为由 source 现场构造的查询（仅错误路径，不影响正常写入）。
    ///
    /// EN: Insert path for an already-prepared graph vector (graph dtype/quantization
    /// prepared, cosine-normalized). Same semantics as
    /// [`insert_with_visited_entry_hint`] but skips internal normalization so the
    /// caller shares the raw `Arc` with the main store (see `prepare_insert_stored`).
    ///
    /// Caller contract: `source_dtype` equals the graph dtype and `quantization`
    /// equals the graph quantization; debug builds assert it, release builds rely
    /// on the producer. Quantized paths must carry a build query (asymmetric
    /// quantization); when missing, debug builds assert and release falls back to
    /// one built from the source (an error-only path that never affects normal
    /// writes).
    pub(crate) fn insert_prepared_with_visited_entry_hint(
        &mut self,
        id: VectorId,
        graph: GraphVector,
        graph_query: Option<&PreparedGraphQuery>,
        visited: &mut HnswVisitTracker,
        entry_hint: Option<VectorId>,
    ) -> usize {
        debug_assert_eq!(
            graph.source_dtype(),
            self.dtype,
            "prepared vector dtype must match graph dtype"
        );
        debug_assert_eq!(
            graph.quantization(),
            self.quantization,
            "prepared vector quantization must match graph quantization"
        );
        if self.dims > 0 && graph.dims() != self.dims {
            return 0;
        }
        if self.nodes.contains_key(&id) {
            return 0;
        }

        let metric = self.metric;
        // 量化路径的构图查询必须是 f32（非对称量化）；缺失仅可能来自绕过
        // `prepare_insert_stored` 的直接调用，debug 断言、release 用原始值现场
        // 构造兜底。余弦度量下查询与图内距离核约定「已预归一化」，兜底路径也
        // 必须先归一化，否则余弦距离与既有图数值口径不一致。
        // EN: Quantized build queries must be f32 (asymmetric quantization); a
        // missing query can only come from a direct call bypassing
        // `prepare_insert_stored` — debug builds assert and release builds one
        // from the source on the spot. Under cosine the query is expected
        // pre-normalized by the in-graph kernel; the fallback normalizes first so
        // cosine distances keep matching existing graph numerics.
        let fallback_query;
        let q: QueryContext<'_> = match &graph {
            GraphVector::Full(v) => QueryContext::Exact(v.as_view()),
            GraphVector::Int8 { .. } => match graph_query {
                Some(q) => QueryContext::Int8(VectorRef::F32(&q.data[..])),
                None => {
                    debug_assert!(false, "int8 graph vector requires an f32 graph query");
                    let raw = graph.source_view().to_f32_vec();
                    let normalized = normalize_query_for_cosine(metric, &raw);
                    fallback_query = PreparedGraphQuery::exact(Arc::from(normalized));
                    QueryContext::Int8(VectorRef::F32(&fallback_query.data[..]))
                }
            },
            GraphVector::RabitQ { .. } => match graph_query {
                // 构图期精确比较：查询侧不量化（见 `metric_pair_stored`）。
                // EN: Exact build-time comparisons: the query side is not quantized
                // (see `metric_pair_stored`).
                Some(q) => QueryContext::Exact(VectorRef::F32(&q.data[..])),
                None => {
                    debug_assert!(false, "rabitq graph vector requires a prepared graph query");
                    let raw = graph.source_view().to_f32_vec();
                    let normalized = normalize_query_for_cosine(metric, &raw);
                    fallback_query = PreparedGraphQuery::exact(Arc::from(normalized));
                    QueryContext::Exact(VectorRef::F32(&fallback_query.data[..]))
                }
            },
        };
        let level = random_level(self.m, self.num_layers, &mut self.rng);

        if self.entry_point.is_none() {
            self.vectors.insert(id, graph);
            let node = self.new_hnsw_node(level);
            self.nodes.insert(id, node);
            self.entry_point = Some(id);
            self.max_layer = level;
            return 0;
        }

        let ep = entry_hint
            .filter(|eid| self.nodes.contains_key(eid))
            .or(self.entry_point)
            .expect("entry_point must exist for non-empty graph");
        let global_max = self.max_layer;
        let upper = global_max.min(level);

        // 下降起点钳制到 ep 层级 + hint 覆盖检查:entry_hint 只保证存活、不保证
        // 层级。hint 层级低于插入顶层 upper 时,连边循环会拿它搜它没有的层——
        // 「ep 缺层 = 腐败」断言误伤合法输入(debug panic),release 下 w 为空、
        // 新节点上层零邻居静默断链(钳制只护住了下降循环,误伤只是被挪到连边
        // 阶段)。hint 够不到 upper 时回退图入口点(恒拥有 max_layer ≥ upper),
        // 此后下降/连边全程 ep 层级 ≥ 搜索层,守卫前提恒真。
        // EN: Clamp the descent start to the ep's level AND check hint coverage:
        // entry_hint guarantees liveness, not level. A hint below the top insert
        // layer `upper` would be used by the linking loop at layers it does not
        // own — the "ep lacks layer ⇒ corruption" guard misfires on legal input
        // (debug panic), and in release the empty candidate set leaves the new
        // node's upper layers silently unlinked (the clamp only guarded the
        // descent; the misfire merely moved to the linking phase). Fall back to
        // the graph entry point (always owns max_layer ≥ upper) when the hint
        // cannot reach `upper`, keeping the guard premise always true.
        let ep_top = self.node_level(ep).unwrap_or(0);
        let (ep, ep_top) = if ep_top < upper {
            let gep = self
                .entry_point
                .expect("entry_point must exist for non-empty graph");
            (gep, self.node_level(gep).unwrap_or(0))
        } else {
            (ep, ep_top)
        };
        let mut ep_curr = ep;
        for lc in ((level + 1)..=global_max.min(ep_top)).rev() {
            // 构图侧保持贪心 ef=1（有意为之，理由见 `upper_layer_ef` 的设计说明）。
            // EN: Build descent intentionally stays greedy at ef=1; rationale in
            // the `upper_layer_ef` doc.
            let w = search_layer_inner(
                visited,
                &self.vectors,
                &self.link_arena,
                &self.nodes,
                self.m,
                q,
                ep_curr,
                1,
                lc,
                metric,
            );
            ep_curr = w.first().map(|(i, _)| *i).unwrap_or(ep_curr);
        }

        let mut layers_neighbors: Vec<(usize, Vec<VectorId>)> = Vec::new();
        for lc in (0..=upper).rev() {
            let ef = self.ef_construction.max(1);
            let w = search_layer_inner(
                visited,
                &self.vectors,
                &self.link_arena,
                &self.nodes,
                self.m,
                q,
                ep_curr,
                ef,
                lc,
                metric,
            );
            let m_lc = layer_m(self.m, lc);
            let neigh = select_neighbors_heuristic_from_sorted(&self.vectors, &w, m_lc, metric);
            ep_curr = w.first().map(|(i, _)| *i).unwrap_or(ep_curr);
            layers_neighbors.push((lc, neigh));
        }

        self.vectors.insert(id, graph);
        let node = self.new_hnsw_node(level);
        self.nodes.insert(id, node);

        for (lc, neigh) in layers_neighbors {
            let m_lc = layer_m(self.m, lc);
            for &e in &neigh {
                add_bidirectional_edge(
                    &self.vectors,
                    &mut self.link_arena,
                    &mut self.nodes,
                    self.m,
                    id,
                    e,
                    lc,
                    m_lc,
                    metric,
                );
            }
        }

        if level > global_max {
            self.max_layer = level;
            self.entry_point = Some(id);
        }

        0
    }

    /// 在一次可变遍历中抽取随机层级（受 RNG 约束）。
    /// EN: Draw random levels in one mutable pass (RNG-bound).
    pub fn draw_random_levels(&mut self, count: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(random_level(self.m, self.num_layers, &mut self.rng));
        }
        out
    }

    /// 在调用方已完成前置校验（维度/主键）时的快速规划路径，避免 Option 分支与重复校验。
    /// 仅测试使用：生产路径（upsert/回退/plan_apply_fresh）一律先 `prepare_insert_stored`，
    /// 走 prepared 变体以共享同一个已量化 `Arc`。
    /// EN: Fast planning path when caller already validated dims/id, avoiding
    /// Option/duplicate checks. Test-only: production paths
    /// (upsert/fallback/plan_apply_fresh) call `prepare_insert_stored` first and
    /// use the prepared variant to share one quantized `Arc`.
    #[cfg(test)]
    pub(crate) fn plan_insert_with_level_unchecked(
        &self,
        id: VectorId,
        vector: Arc<Vec<f32>>,
        level: usize,
        entry_hint: Option<VectorId>,
    ) -> InsertPlan {
        let PreparedInsert {
            main: _main,
            graph,
            graph_query,
        } = prepare_insert_stored(
            self.metric,
            self.dtype,
            self.quantization,
            Arc::unwrap_or_clone(vector),
        );
        self.plan_insert_with_level_unchecked_prepared(
            id,
            graph,
            graph_query.as_ref(),
            level,
            entry_hint,
        )
    }

    /// 已准备图侧向量的规划路径：语义与 `plan_insert_with_level_unchecked` 相同，
    /// 但跳过内部归一化，让调用方在主存储与图之间共享原始向量 `Arc`。
    /// 调用方契约：`source_dtype`/`quantization` 必须与图一致；debug 构建断言把关，
    /// int8 必须携带 f32 构图查询（缺失时同插入路径的 debug 断言 + release 兜底）。
    /// EN: Planning path for an already-prepared graph vector. Same semantics as
    /// `plan_insert_with_level_unchecked` but skips internal normalization so the
    /// caller shares the raw `Arc` between the main store and the graph. Caller
    /// contract: `source_dtype`/`quantization` must match the graph; debug builds
    /// assert it, and int8 must carry an f32 build query (missing queries assert in
    /// debug and fall back in release, exactly like the insert path).
    pub(crate) fn plan_insert_with_level_unchecked_prepared(
        &self,
        id: VectorId,
        graph: GraphVector,
        graph_query: Option<&PreparedGraphQuery>,
        level: usize,
        entry_hint: Option<VectorId>,
    ) -> InsertPlan {
        debug_assert_eq!(
            graph.source_dtype(),
            self.dtype,
            "prepared vector dtype must match graph dtype"
        );
        debug_assert_eq!(
            graph.quantization(),
            self.quantization,
            "prepared vector quantization must match graph quantization"
        );
        let metric = self.metric;
        if self.entry_point.is_none() {
            return InsertPlan {
                id,
                vector: graph,
                level,
                layers_neighbors: Vec::new(),
            };
        }
        // 构图查询选择：同插入路径（f32 恒非对称，缺失仅错误路径兜底；
        // 余弦兜底同样先归一化，保持图内距离口径）。
        // EN: Build-query selection mirrors the insert path (f32 is always
        // asymmetric; a missing query only occurs on error paths, and the cosine
        // fallback normalizes first to keep in-graph distance semantics).
        let fallback_query;
        let q: QueryContext<'_> = match &graph {
            GraphVector::Full(v) => QueryContext::Exact(v.as_view()),
            GraphVector::Int8 { .. } => match graph_query {
                Some(q) => QueryContext::Int8(VectorRef::F32(&q.data[..])),
                None => {
                    debug_assert!(false, "int8 graph vector requires an f32 graph query");
                    let raw = graph.source_view().to_f32_vec();
                    let normalized = normalize_query_for_cosine(metric, &raw);
                    fallback_query = PreparedGraphQuery::exact(Arc::from(normalized));
                    QueryContext::Int8(VectorRef::F32(&fallback_query.data[..]))
                }
            },
            GraphVector::RabitQ { .. } => match graph_query {
                // 构图期精确比较：查询侧不量化（见 `metric_pair_stored`）。
                // EN: Exact build-time comparisons: the query side is not quantized
                // (see `metric_pair_stored`).
                Some(q) => QueryContext::Exact(VectorRef::F32(&q.data[..])),
                None => {
                    debug_assert!(false, "rabitq graph vector requires a prepared graph query");
                    let raw = graph.source_view().to_f32_vec();
                    let normalized = normalize_query_for_cosine(metric, &raw);
                    fallback_query = PreparedGraphQuery::exact(Arc::from(normalized));
                    QueryContext::Exact(VectorRef::F32(&fallback_query.data[..]))
                }
            },
        };
        let ep = entry_hint
            .filter(|eid| self.nodes.contains_key(eid))
            .or(self.entry_point)
            .expect("entry_point must exist");
        let global_max = self.max_layer;
        let upper = global_max.min(level);
        let mut visited = HnswVisitTracker::with_capacity(Self::visited_insert_capacity_hint(
            self.ef_construction,
        ));
        // 下降起点钳制 + hint 覆盖回退：同 insert_with_visited_entry_hint 的说明。
        // EN: Descent clamp + hint-coverage fallback — see insert_with_visited_entry_hint.
        let ep_top = self.node_level(ep).unwrap_or(0);
        let (ep, ep_top) = if ep_top < upper {
            let gep = self.entry_point.expect("entry_point must exist");
            (gep, self.node_level(gep).unwrap_or(0))
        } else {
            (ep, ep_top)
        };
        let mut ep_curr = ep;
        for lc in ((level + 1)..=global_max.min(ep_top)).rev() {
            // 规划侧同样保持贪心 ef=1（有意为之，与查询侧束宽不对称，理由见
            // `upper_layer_ef` 的设计说明）。
            // EN: The planning descent also stays greedy at ef=1 — deliberately
            // asymmetric with the query-side beam; see the `upper_layer_ef` doc.
            let w = search_layer_inner(
                &mut visited,
                &self.vectors,
                &self.link_arena,
                &self.nodes,
                self.m,
                q,
                ep_curr,
                1,
                lc,
                metric,
            );
            ep_curr = w.first().map(|(i, _)| *i).unwrap_or(ep_curr);
        }
        let mut layers_neighbors: Vec<(usize, Vec<VectorId>)> = Vec::new();
        for lc in (0..=upper).rev() {
            let ef = self.ef_construction.max(1);
            let w = search_layer_inner(
                &mut visited,
                &self.vectors,
                &self.link_arena,
                &self.nodes,
                self.m,
                q,
                ep_curr,
                ef,
                lc,
                metric,
            );
            let m_lc = layer_m(self.m, lc);
            let neigh = select_neighbors_heuristic_from_sorted(&self.vectors, &w, m_lc, metric);
            ep_curr = w.first().map(|(i, _)| *i).unwrap_or(ep_curr);
            layers_neighbors.push((lc, neigh));
        }
        InsertPlan {
            id,
            vector: graph,
            level,
            layers_neighbors,
        }
    }

    /// 提交前剔除计划中已不存在的邻居：规划与 `remove` 并发时，被删节点可能
    /// 仍留在计划的邻居表里；`add_bidirectional_edge` 遇缺失端点会调试断言。
    /// 在持图写锁（与删除互斥）时调用，保证计划只引用当前图中的节点。
    /// EN: Drop planned neighbors that no longer exist. A `remove` racing the
    /// read-lock planning phase may leave stale ids in a plan; the edge adder
    /// debug-asserts on missing endpoints. Call under the graph write lock.
    pub(crate) fn retain_live_plan_neighbors(&self, plans: &mut [InsertPlan]) {
        for plan in plans.iter_mut() {
            let id = plan.id;
            for (_layer, neigh) in plan.layers_neighbors.iter_mut() {
                neigh.retain(|e| *e != id && self.nodes.contains_key(e));
            }
        }
    }

    /// 分组规划后的孤点补救：组内兄弟互不可见 + 对称剪枝会令部分新节点
    /// 失去全部旧图边（L0 度 0 或极低），成为整簇查询的不可达口袋。提交后
    /// 对低度成员补连到同组兄弟：候选限于 id 顺序 ±`window` 的邻域（成簇
    /// 语料块内连续 id 即近邻，扰动极小）。候选分两档排序：健壮
    /// （`min_degree <= 度 < max_deg`）优先，同为孤点的兄弟次之；档内按
    /// （距离, id）升序，补到 `min_degree` 为止。采集时已饱和（`度 >=
    /// max_deg`）的兄弟直接排除：补边必然触发对端剪枝（见下）。返回实际
    /// 补边数。
    ///
    /// EN: Post-group orphan repair: siblings inside a plan group are invisible
    /// to each other and symmetric pruning can strip a new node of every legacy
    /// edge (L0 degree 0 or near), turning it into an unreachable pocket for
    /// whole-cluster queries. After apply, low-degree members are re-linked to
    /// siblings within a bounded id-order window (consecutive ids are near
    /// neighbors on block-clustered corpora). Candidates are ranked in two
    /// tiers: robust (`min_degree <= degree < max_deg`) first, fellow orphans
    /// next; within a tier by (distance, id), until `min_degree`. Siblings
    /// already saturated at
    /// collection time (`degree >= max_deg`) are excluded outright: adding to
    /// them must trigger their prune (see below). Returns the number of edges
    /// added.
    ///
    /// 只补 L0：上层图节点稀疏，导航缺口已由查询侧束宽
    /// （`upper_layer_ef`）覆盖；L0 才是整簇召回的主体。
    /// EN: L0 only: upper layers are sparse and the query-side beam
    /// (`upper_layer_ef`) already covers navigation gaps; L0 carries recall.
    ///
    /// 边对称不变量：补边走 `add_bidirectional_edge`，与常规插入共用同一条
    /// 剪枝/反向清理路径。对端饱和状态在**决策时复测**：本循环早先的补边
    /// 可能已把原本有余量的兄弟补满，此时跳过，绝不触发对端剪枝——剪枝若
    /// 丢掉刚补的边，对称清理会连正向边一起撤销；并且被剪掉的边可能属于
    /// 不在组内的健康节点，使其跌破 `min_degree` 而无从补救。采集时已饱和
    /// 的兄弟根本不入档，决策时复测只兜住本循环内新近补满者。
    /// EN: Symmetry invariant: repair goes through `add_bidirectional_edge`,
    /// the same prune/reciprocal-cleanup path as regular inserts. The far
    /// side's saturation is re-measured at decision time: earlier adds in this
    /// loop may have filled a previously spare sibling; skip it then and never
    /// trigger its prune. Pruning may drop the fresh edge (reciprocal cleanup
    /// then reverts the forward edge too), and the evicted edge may belong to a
    /// healthy node outside the group, pushing it below `min_degree` with no
    /// repair coverage. Candidates already saturated at collection time never
    /// enter the ranking; the decision-time re-measurement only catches those
    /// that fill up during this loop.
    ///
    /// 不做距离门限：补救的首要目标是可达，窗口内最近邻即使跨簇也远优于
    /// 0 度；每点至多补到 `min_degree`（缺省 4，远低于 `max_deg`），保留主要
    /// 度数预算给后续正常插入的近邻。代价上界为 O(低度点数 × 窗口 × 一次
    /// 距离计算)，只在分组提交后的写锁内执行一次。
    /// EN: No distance cutoff: connectivity is the goal, and the nearest window
    /// sibling — even across a cluster boundary — beats degree 0. Each node is
    /// only filled to `min_degree` (default 4, far below `max_deg`), leaving the
    /// main degree budget for later regular inserts' near neighbors. Cost is
    /// bounded by O(low-degree nodes x window x one distance computation) and
    /// runs once under the post-group write lock.
    pub(crate) fn repair_group_links(
        &mut self,
        ids: &[VectorId],
        min_degree: usize,
        window: usize,
    ) -> usize {
        if ids.len() < 2 || min_degree == 0 || window == 0 {
            return 0;
        }
        let m = self.m;
        let max_deg = layer_m(m, 0);
        let min_degree = min_degree.min(max_deg);
        let metric = self.metric;
        let mut added = 0usize;
        // 候选缓冲跨节点复用，避免每个孤点分配多份 Vec；窗口可经环境变量
        // 覆写，预留量按批内实际候选数封顶，手误的巨大窗口不会在写锁内预留
        // GB 级内存（真实候选数不超过 `ids.len() - 1`）。
        // EN: Candidate buffer reused across nodes so each orphan allocates
        // none. The window is env-overridable, so cap the reservation by the
        // actual per-node candidate count: a typo'd huge window must not
        // reserve GBs under the write lock (real candidates <= ids.len() - 1).
        let cand_cap = window.saturating_mul(2).min(ids.len().saturating_sub(1));
        let mut cands: Vec<(u8, f32, VectorId)> = Vec::with_capacity(cand_cap);
        for i in 0..ids.len() {
            if self.l0_degree(ids[i]) >= min_degree {
                continue;
            }
            let id = ids[i];
            let lo = i.saturating_sub(window);
            // 窗口可经环境变量覆写，取饱和加法防止越界值在 debug 下溢出 panic。
            // EN: The window can be overridden via env; saturate the upper bound
            // so an out-of-range value cannot overflow-panic in debug builds.
            let hi = i.saturating_add(window).min(ids.len() - 1);
            cands.clear();
            {
                let host = match self.vectors.get(&id) {
                    Some(v) => v.as_graph_ref(),
                    None => continue,
                };
                for (j, &cid) in ids.iter().enumerate().take(hi + 1).skip(lo) {
                    if j == i {
                        continue;
                    }
                    let Some(other) = self.vectors.get(&cid) else {
                        continue;
                    };
                    let deg = self.l0_degree(cid);
                    // 已饱和兄弟在补边时必然触发对端剪枝，采集阶段直接排除；
                    // 决策时的复测只兜住本循环内新近补满的候选。
                    // EN: Saturated siblings would necessarily trigger their
                    // prune on add, so exclude them at collection time; the
                    // decision-time re-measurement only catches candidates
                    // that fill up during this loop.
                    if deg >= max_deg {
                        continue;
                    }
                    let tier = if deg >= min_degree { 0u8 } else { 1u8 };
                    let d = metric_pair_stored(metric, host, other.as_graph_ref());
                    cands.push((tier, d, cid));
                }
            }
            cands.sort_unstable_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.total_cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            for &(_, _, cid) in &cands {
                if self.l0_degree(id) >= min_degree {
                    break;
                }
                if self.l0_degree(cid) >= max_deg {
                    continue;
                }
                let before = self.l0_degree(id);
                add_bidirectional_edge(
                    &self.vectors,
                    &mut self.link_arena,
                    &mut self.nodes,
                    m,
                    id,
                    cid,
                    0,
                    max_deg,
                    metric,
                );
                // 以实际度增判定存活：对端仍有余量时补边不会触发剪枝。
                // EN: Judge survival by the measured degree gain: with spare
                // capacity on the far side the add cannot prune.
                if self.l0_degree(id) > before {
                    added += 1;
                }
            }
        }
        added
    }

    /// L0 出度上限（`layer_m(m, 0)`；默认 m 下为 32）。
    /// EN: L0 out-degree cap (`layer_m(m, 0)`; 32 under the default m).
    pub(crate) fn l0_max_degree(&self) -> usize {
        self.layer_max_degree(0)
    }

    /// 指定层的出度上限（L0=2×m，上层=0.5×m，受 `layer_m` 乘数控制）。
    /// EN: Out-degree cap for a layer (L0=2*m, upper=0.5*m, controlled by
    /// `layer_m` multipliers).
    fn layer_max_degree(&self, layer: usize) -> usize {
        layer_m(self.m, layer)
    }

    /// 找一个尚有 L0 余量的存活节点作补救锚点（排除 `exclude`）：单候选孤儿
    /// 形态下入口点可能已饱和，`repair_group_links` 会把它过滤掉，需要另一个
    /// 有余量的伙伴才可能补边。只在极端罕见路径调用，O(n) 扫描可接受。
    /// EN: Find a live node with spare L0 capacity as a repair anchor (excluding
    /// `exclude`): in the single-candidate orphan shape the entry point may be
    /// saturated and `repair_group_links` filters it out, so another spare peer
    /// is needed for any edge to be addable. Only called on a very rare path,
    /// where the O(n) scan is acceptable.
    pub(crate) fn find_spare_l0_anchor(&self, exclude: VectorId) -> Option<VectorId> {
        // `nodes` 是 AHashMap（每进程随机种子）：取「第一个空位节点」会让同一
        // 操作序列在不同运行间产生不同补边，快照不可复现；且空闲节点往往本身
        // 接近孤儿，拿它作锚点无法恢复可达性。改为选择**仍有余量中度数最高**
        // 的节点，度数相同按 id 升序破平；容量判定复用 `l0_degree`，与
        // `repair_group_links` 的饱和过滤同源。
        // EN: `nodes` is an AHashMap (per-process random seed): picking the
        // "first node with a free slot" makes identical operation sequences
        // produce different repair edges across runs (non-reproducible graph /
        // snapshots), and a spare node is often another near-orphan, so the
        // added edge would not restore reachability. Choose the
        // best-connected node that still has capacity, ties broken by
        // ascending id; capacity reuses `l0_degree`, the same measure as
        // `repair_group_links`'s saturation filter.
        let max_deg = self.l0_max_degree();
        // 单趟扫描、每节点只算一次度数：`l0_degree` 走 AHashMap 查找，若在
        // `max_by` 比较器里反复调用会变成 ~3n 次哈希查找（写锁内延迟放大）。
        // 选择规则显式化：度数最大者优先，同度取最小 id（确定性）。
        // EN: Single pass with the degree computed once per node: calling
        // `l0_degree` inside `max_by` would cost ~3n hash lookups (latency
        // amplified under the write lock). The rule is explicit: highest
        // degree wins, ties go to the smallest id (deterministic).
        let mut best: Option<(usize, VectorId)> = None;
        for (id, node) in &self.nodes {
            if *id == exclude {
                continue;
            }
            let deg = self.node_l0_degree(node);
            if deg >= max_deg {
                continue;
            }
            let better = match best {
                None => true,
                Some((best_deg, best_id)) => deg > best_deg || (deg == best_deg && *id < best_id),
            };
            if better {
                best = Some((deg, *id));
            }
        }
        best.map(|(_, id)| id)
    }

    /// 给定节点引用的 L0 出度（与 [`Self::l0_degree`] 同一口径，省去哈希查找）。
    /// EN: L0 out-degree from a node reference (same definition as
    /// [`Self::l0_degree`], without the hash lookup).
    fn node_l0_degree(&self, node: &HnswNode) -> usize {
        self.node_layer_degree(node, 0)
    }

    /// 给定节点引用在指定层的出度（节点缺该层时为 0）。
    /// EN: Out-degree from a node reference at a layer (0 when the node lacks it).
    fn node_layer_degree(&self, node: &HnswNode, layer: usize) -> usize {
        node.link_active_ref(&self.link_arena, self.m, layer)
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// 节点在 L0 的实际出度（过滤 `EMPTY_NEIGHBOR` 哨兵后的邻接长度）。
    /// EN: Actual L0 out-degree (link length after filtering the `EMPTY_NEIGHBOR` sentinel).
    pub(crate) fn l0_degree(&self, id: VectorId) -> usize {
        self.layer_degree(id, 0)
    }

    /// 节点在指定层的实际出度（过滤 `EMPTY_NEIGHBOR` 哨兵后的邻接长度）。
    /// EN: Actual out-degree at a layer (link length after filtering the
    /// `EMPTY_NEIGHBOR` sentinel).
    fn layer_degree(&self, id: VectorId, layer: usize) -> usize {
        self.nodes
            .get(&id)
            .map(|n| self.node_layer_degree(n, layer))
            .unwrap_or(0)
    }

    /// 节点是否存在且持有指定层（悬空 id / 低层节点返回 false）。
    /// EN: Whether the node exists and owns the layer (false for dangling ids
    /// and lower-level nodes).
    fn layer_has_node(&self, id: VectorId, layer: usize) -> bool {
        self.nodes
            .get(&id)
            .map(|n| layer < n.level_count())
            .unwrap_or(false)
    }

    /// 某层的告警位（层号由 `MAX_HNSW_LAYERS` 编译期约束在 u64 范围内）。
    /// EN: Warn bit for a layer (the layer id is compile-time bounded to the
    /// u64 range by `MAX_HNSW_LAYERS`).
    fn warn_bit(layer: usize) -> u64 {
        debug_assert!(layer < MAX_HNSW_LAYERS);
        1u64 << layer
    }

    /// 仅保留 `0..=max_layer` 的告警位（层数缩小时清除已不存在层的残留位）。
    /// EN: Keep only warn bits for `0..=max_layer` (clears stale bits of layers
    /// that no longer exist when the layer count shrinks).
    fn warn_mask_upto(max_layer: usize) -> u64 {
        if max_layer >= u64::BITS as usize - 1 {
            u64::MAX
        } else {
            (1u64 << (max_layer + 1)) - 1
        }
    }

    /// 从入口点做指定层 BFS 的可达集合（只沿存活节点的链接扩展，避免为每个
    /// 访问节点分配 `Vec`）。返回集合可能包含悬空 link id；调用方按
    /// `nodes` 键域与层成员计数，不能直接拿 `len()` 与节点总数比较。
    /// EN: Reachable set from the entry point at a given layer (expands only
    /// through live nodes, avoiding a `Vec` allocation per visited node). The
    /// set may contain dangling link ids; callers count within the `nodes` key
    /// domain and the layer's node set instead of comparing `len()` with the
    /// total node count.
    fn layer_reachable_set(&self, ep: VectorId, layer: usize) -> AHashSet<VectorId> {
        // 预分配有界（16k）：碎片化时可达集可能很小，1M 图按节点数预留会在
        // 写锁内白白占 ~10MB；集合填满时自然增长即可。
        // EN: Bounded pre-size (16k): fragmented graphs may have a tiny reachable
        // set, and reserving the node count on a 1M graph would waste ~10MB under
        // the write lock; letting the set grow when it actually fills is enough.
        let mut seen: AHashSet<VectorId> = AHashSet::with_capacity(self.nodes.len().min(16_384));
        let mut stack = vec![ep];
        seen.insert(ep);
        while let Some(cur) = stack.pop() {
            let Some(node) = self.nodes.get(&cur) else {
                continue;
            };
            if let Some(links) = node.link_active_ref(&self.link_arena, self.m, layer) {
                for &n in links {
                    if seen.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
        seen
    }

    /// 全层连通性兜底：逐层从入口点 BFS，把每个不可达连通分量接到一个可达
    /// 且仍有余量的节点。L0 分量断裂表现为召回整体崩塌；上层分量断裂表现为
    /// 查询下降后落入错误簇的 L0 入口（宽束也救不回来，因为层内根本没有路径）。
    /// 这是 `repair_group_links` 的补充——后者只补 L0「度数 < min_degree」的
    /// 孤点，而整簇分量可能每个成员度数都 ≥min_degree（内部互连、对外无桥），
    /// 串行构图在对称剪枝丢掉唯一桥边后正是这种形态。分量成员按 id 排序保证
    /// 确定性；端点取分量内度数最低者以尽量避开对端剪枝；新边被剪掉时最多重试
    /// 4 轮。只在 drain 排空队列后调用：一次 O(N) 的层节点数统计 + 每层
    /// O(该层节点·m) 的 BFS，相对构图成本可忽略。返回实际新增的桥边总数。
    /// EN: All-layer connectivity backstop: per layer, BFS from the entry point
    /// and attach each unreachable component to a reachable node with spare
    /// capacity. A broken L0 component collapses recall outright; a broken
    /// upper-layer component lands queries on a wrong-cluster L0 entry (even a
    /// wide beam cannot recover because no path exists within the layer). This
    /// complements `repair_group_links`, which only fills L0 nodes below
    /// `min_degree`: a whole-cluster component can have every member at >=
    /// `min_degree` (internally linked, no bridge), exactly the shape a serial
    /// build produces after symmetric pruning drops the only bridge edge.
    /// Component members are id-ordered for determinism; the endpoint is the
    /// component member with the lowest degree to minimize far-side pruning; up
    /// to 4 passes retry if the new edge is pruned away. Only called after a
    /// drain empties the queue: one O(N) layer-population pass plus per-layer
    /// O(layer nodes * m) BFS, negligible against the build. Returns the total
    /// number of bridge edges added.
    pub(crate) fn ensure_connectivity(&mut self) -> usize {
        if self.entry_point.is_none() {
            return 0;
        }
        // 入口点持有全部层（max_layer 由最高层节点定义）；逐层扫，任何一层
        // 的断裂都必须补回，否则上层导航会把查询带进错误盆地。
        // EN: The entry point owns every layer (max_layer is defined by the
        // highest-level node); sweep each layer, because any break sends upper
        // navigation into a wrong basin.
        //
        // 每层节点总数只在这里扫一遍 `nodes` 求一次（按最高层做后缀和），而不是
        // 每层再扫一遍全图：drain 排空钩子在写锁内执行，层数越多重复全图扫描
        // 的延迟放大越明显。单层内部只在**发现断裂时**才为收集不可达集合扫
        // `nodes`；连通正常时只遍历可达集合（层节点数）。
        // EN: Layer populations are computed once here by scanning `nodes` a
        // single time (suffix sums over each node's top level) instead of once
        // per layer: the drain-empty hook runs under the graph write lock, and
        // repeated full-map scans amplify latency with the layer count. Within
        // a layer, `nodes` is scanned only when a break is actually found; the
        // healthy path walks just the reachable set (layer-sized).
        let max_layer = self.max_layer;
        let mut exact_top_level = vec![0usize; max_layer + 1];
        for node in self.nodes.values() {
            let top = (node.level_count() - 1).min(max_layer);
            exact_top_level[top] += 1;
        }
        let mut layer_totals = vec![0usize; max_layer + 1];
        let mut acc = 0usize;
        for layer in (0..=max_layer).rev() {
            acc += exact_top_level[layer];
            layer_totals[layer] = acc;
        }
        let mut added = 0usize;
        for (layer, &total) in layer_totals.iter().enumerate() {
            added += self.ensure_layer_connectivity(layer, total);
        }
        added
    }

    /// 单层连通性兜底：断开的层内分量各接一条回边，返回新增边数。
    /// `total_nodes` 是调用方一次性预计算的该层节点总数。
    /// EN: Single-layer connectivity backstop: attach each disconnected
    /// in-layer component with one back-edge and return the number added.
    /// `total_nodes` is the layer population precomputed once by the caller.
    fn ensure_layer_connectivity(&mut self, layer: usize, total_nodes: usize) -> usize {
        let Some(ep) = self.entry_point else {
            return 0;
        };
        // 入口点必须持有该层；否则 BFS 无意义（调用方契约：entry 是最高层节点）。
        // EN: The entry must own the layer; otherwise the BFS is meaningless
        // (caller contract: entry is the highest-level node).
        if !self.layer_has_node(ep, layer) {
            return 0;
        }
        let max_deg = self.layer_max_degree(layer);
        let mut added = 0usize;
        // 已尝试失败的 (endpoint, anchor) 对：重试轮次据此换锚点，否则同一
        // 组合会被确定性重选、4 轮空转。
        // EN: Failed (endpoint, anchor) pairs so retry passes pick a different
        // anchor instead of deterministically re-selecting the same one.
        let mut failed_links: AHashSet<(VectorId, VectorId)> = AHashSet::new();
        for _pass in 0..4 {
            let reachable = self.layer_reachable_set(ep, layer);
            // 连通正常路径只遍历可达集合：可达集合可能混入悬空 link id，
            // 因此按「节点存在且持有本层」过滤后再计数；不得再扫全图。
            // EN: The healthy path walks only the reachable set: it may contain
            // dangling link ids, so count entries that exist and own this layer
            // — never rescan the whole node map here.
            let reachable_nodes = reachable
                .iter()
                .copied()
                .filter(|id| self.layer_has_node(*id, layer))
                .count();
            if reachable_nodes >= total_nodes {
                // 本层已全可达：清除告警位，后续若再次断裂可重新告警。
                // EN: The layer is fully reachable: clear the warn bit so a
                // future break warns again.
                self.incomplete_warn_mask &= !Self::warn_bit(layer);
                return added;
            }
            // 断裂路径（罕见）：才为收集不可达集合扫一次 `nodes`。
            // EN: Broken path (rare): scan `nodes` once to collect the
            // unreachable set.
            let mut unreachable: Vec<VectorId> = self
                .nodes
                .iter()
                .filter(|(id, n)| layer < n.level_count() && !reachable.contains(id))
                .map(|(id, _)| *id)
                .collect();
            unreachable.sort_unstable();

            // 锚点候选池：可达节点 ≤ 上限时全量；更大时用**确定性 id 取模**
            // 采样（含入口点及其邻居），避免每分量再扫全图造成 O(C·R·dims)
            // 写锁内延迟。采样按簇内 id 块仍保留足量同簇候选。
            // EN: Anchor candidate pool: all reachable nodes up to the cap; above
            // it, a deterministic id-modulo sample (plus the entry point and its
            // neighbors) avoids an O(C·R·dims) scan per component under the write
            // lock. Id-block clustering still leaves enough same-cluster anchors.
            const ANCHOR_POOL_MAX: usize = 16_384;
            let mut pool: Vec<VectorId> = Vec::new();
            if reachable_nodes <= ANCHOR_POOL_MAX {
                // 全量锚点池同样只从可达集合取（层节点数），不扫全图。
                // EN: The full anchor pool also comes from the reachable set
                // (layer-sized), never from a full-map scan.
                pool.extend(
                    reachable
                        .iter()
                        .copied()
                        .filter(|id| self.layer_has_node(*id, layer)),
                );
                pool.sort_unstable();
            } else {
                pool.push(ep);
                if let Some(node) = self.nodes.get(&ep) {
                    if let Some(links) = node.link_active_ref(&self.link_arena, self.m, layer) {
                        for &n in links {
                            if self
                                .nodes
                                .get(&n)
                                .map(|nn| layer < nn.level_count())
                                .unwrap_or(false)
                            {
                                pool.push(n);
                            }
                        }
                    }
                }
                // 步长按**可达**节点数算（分支前提 reachable_nodes >
                // ANCHOR_POOL_MAX）；先收集所有取模命中的 id 再排序截断，
                // 不依赖 AHashMap 随机迭代序（同序列跨运行必须产出同一张图）。
                // EN: Derive the stride from the **reachable** count (this branch
                // guarantees reachable_nodes > ANCHOR_POOL_MAX); collect all
                // modulo hits first, then sort and truncate — never depend on
                // AHashMap's per-process iteration order (identical operation
                // sequences must produce identical graphs).
                // 向上取整：reachable 落在 (上限, 2×上限] 时 floor 会得到
                // stride=1，随后 truncate 只保留低位 id，高 id 分量失去同簇
                // 锚点；div_ceil 保证采样均匀覆盖整个 id 区间。
                // EN: Round up: when reachable sits in (cap, 2*cap], floor yields
                // stride=1 and the subsequent truncate keeps only low ids, so
                // high-id components lose same-cluster anchors; div_ceil spreads
                // the sample across the whole id range.
                let stride = reachable_nodes.div_ceil(ANCHOR_POOL_MAX).max(1) as u64;
                let mut sampled: Vec<VectorId> = reachable
                    .iter()
                    .copied()
                    .filter(|id| self.layer_has_node(*id, layer) && *id % stride == 0)
                    .collect();
                sampled.sort_unstable();
                if sampled.len() > ANCHOR_POOL_MAX {
                    let step = sampled.len().div_ceil(ANCHOR_POOL_MAX);
                    sampled = sampled.into_iter().step_by(step).collect();
                }
                // 取模命中可能接近 2×上限（stride 取整）；排序后直接截断，
                // 确定性且不再放大下游逐分量距离扫描。
                // EN: Modulo hits can approach 2x the cap (floor division);
                // truncating after the sort is deterministic and keeps the
                // per-component distance scan bounded.
                sampled.truncate(ANCHOR_POOL_MAX);
                pool.extend(sampled);
                pool.sort_unstable();
                pool.dedup();
            }

            let mut linked_this_pass = 0usize;
            let mut consumed: AHashSet<VectorId> = AHashSet::new();
            for start in unreachable {
                if consumed.contains(&start) || reachable.contains(&start) {
                    continue;
                }
                // 不可达分量 BFS（只在不可达集合内扩展）。
                // EN: Unreachable-component BFS (expands within the unreachable set only).
                let mut members = Vec::new();
                let mut seen = AHashSet::new();
                let mut st = vec![start];
                seen.insert(start);
                while let Some(cur) = st.pop() {
                    let Some(node) = self.nodes.get(&cur) else {
                        continue;
                    };
                    members.push(cur);
                    if let Some(links) = node.link_active_ref(&self.link_arena, self.m, layer) {
                        for &n in links {
                            if !reachable.contains(&n) && seen.insert(n) {
                                st.push(n);
                            }
                        }
                    }
                }
                for id in &members {
                    consumed.insert(*id);
                }
                members.sort_unstable();
                // 端点：分量内度数最低者（同度取最小 id）；缺向量/节点的悬空
                // id 直接排除——它们在 `l0_degree` 里读作 0 会被优先选中，
                // 随后 add edge 时 panic（OCR 高优先）。
                // EN: Endpoint: lowest-degree component member (smallest id on ties);
                // dangling ids without a node/vector are excluded — `l0_degree`
                // reads them as 0, so they would be selected preferentially and
                // then panic while adding the edge (OCR high).
                let endpoint = match members
                    .iter()
                    .filter(|id| {
                        self.vectors.contains_key(id)
                            && self
                                .nodes
                                .get(id)
                                .map(|n| layer < n.level_count())
                                .unwrap_or(false)
                    })
                    .min_by_key(|id| (self.layer_degree(**id, layer), **id))
                {
                    Some(id) => *id,
                    None => continue,
                };
                // 锚点选**向量距离最近**的可达节点（优先有余量者），而不是
                // 度数最高的节点：任意远锚虽然恢复理论可达性，但桥边距离太远，
                // ef 束内永远排不上号，搜索仍然找不到碎片——同簇近锚才能让
                // 桥边进入候选束、真正恢复检索质量。同距离按 id 升序破平。
                // EN: Choose the reachable node **nearest in vector distance** (spare
                // capacity preferred) instead of the highest-degree node: an arbitrary
                // far anchor restores theoretical reachability but its long bridge
                // never enters the ef beam, so search still misses the fragments —
                // only a nearby anchor (typically same-cluster) makes the bridge
                // search-traversable. Ties break to the smallest id.
                let endpoint_vec = match self.vectors.get(&endpoint) {
                    Some(v) => v.as_graph_ref(),
                    // 缺向量（悬空 id）：跳过该分量，绝不 panic（写锁内 panic
                    // 会毒化图锁并级联）。
                    // EN: Missing vector (dangling id): skip this component; never
                    // panic here (a panic under the write lock poisons the graph
                    // lock and cascades).
                    None => continue,
                };
                let mut nearest_spare: Option<(f32, VectorId)> = None;
                let mut nearest_any: Option<(f32, VectorId)> = None;
                for &id in &pool {
                    if failed_links.contains(&(endpoint, id)) {
                        continue;
                    }
                    let Some(other) = self.vectors.get(&id) else {
                        continue;
                    };
                    let d = metric_pair_stored(self.metric, endpoint_vec, other.as_graph_ref());
                    let better = |best: Option<(f32, VectorId)>| match best {
                        None => true,
                        Some((bd, bid)) => d < bd || (d == bd && id < bid),
                    };
                    if better(nearest_any) {
                        nearest_any = Some((d, id));
                    }
                    if self.layer_degree(id, layer) < max_deg && better(nearest_spare) {
                        nearest_spare = Some((d, id));
                    }
                }
                let mut candidates: Vec<VectorId> = Vec::new();
                if let Some((_, id)) = nearest_spare {
                    candidates.push(id);
                }
                if let Some((_, id)) = nearest_any {
                    if !candidates.contains(&id) {
                        candidates.push(id);
                    }
                }
                let mut linked_this = false;
                for anchor_id in candidates {
                    add_bidirectional_edge(
                        &self.vectors,
                        &mut self.link_arena,
                        &mut self.nodes,
                        self.m,
                        endpoint,
                        anchor_id,
                        layer,
                        max_deg,
                        self.metric,
                    );
                    // BFS 从入口点沿**出边**扩展：只有锚点侧的 `anchor -> endpoint`
                    // 回边存在，分量才真正被接回。单侧边不算成功，记录失败对，
                    // 重试轮次会换下一个候选锚点。
                    // EN: The BFS expands along **outgoing** edges from the entry
                    // point: only the anchor-side `anchor -> endpoint` back-edge
                    // actually reconnects the component. A one-sided edge does not
                    // count; record the failed pair so retries pick another anchor.
                    if self
                        .neighbors_at_layer(anchor_id, layer)
                        .contains(&endpoint)
                    {
                        linked_this = true;
                        break;
                    }
                    failed_links.insert((endpoint, anchor_id));
                }
                if linked_this {
                    linked_this_pass += 1;
                }
            }
            added += linked_this_pass;
            if linked_this_pass == 0 {
                // 无法推进：停止避免空转（残余由调用方诊断日志暴露）。
                // EN: No progress: stop instead of spinning (leftovers surface via
                // the caller's diagnostic log).
                break;
            }
        }
        // 最终再点一次可达数；残留时首次打 warn，同层重复断裂降到 debug
        // （既防持锁刷屏，又保留恶化趋势的可见性），恢复全可达后清除告警位。
        // EN: Final reachability recount; the first leftover warns, repeated
        // breaks in the same layer drop to debug (anti-flood under the lock
        // while keeping deterioration visible), and the bit clears once the
        // layer is fully reachable.
        let seen = self.layer_reachable_set(ep, layer);
        let reachable_nodes = self
            .nodes
            .iter()
            .filter(|(id, n)| layer < n.level_count() && seen.contains(id))
            .count();
        let still_unreachable = total_nodes.saturating_sub(reachable_nodes);
        let warn_bit = Self::warn_bit(layer);
        if still_unreachable > 0 {
            // 上层断裂只靠 L0 门禁不可见（L0 可达率与召回均值都可能仍绿），
            // 因此提升到 warn；但同一层持续断裂只告警一次（写锁内 stderr
            // 反复刷屏可能阻塞持锁线程），恢复全可达后清除告警位允许再次告警。
            // EN: Upper-layer breaks are invisible to L0-only gates (L0
            // reachability and mean recall can both stay green), so escalate to
            // warn; but a persistent break warns once per layer (repeated stderr
            // writes under the write lock could block the holder), and the bit
            // is cleared once the layer is fully reachable so a later break
            // warns again.
            if self.incomplete_warn_mask & warn_bit == 0 {
                self.incomplete_warn_mask |= warn_bit;
                tracing::warn!(
                    layer,
                    total_nodes,
                    still_unreachable,
                    added,
                    "layer connectivity sweep incomplete"
                );
            } else {
                tracing::debug!(
                    layer,
                    total_nodes,
                    still_unreachable,
                    added,
                    "layer connectivity sweep still incomplete (warn suppressed)"
                );
            }
        } else {
            self.incomplete_warn_mask &= !warn_bit;
        }
        added
    }

    /// 批量应用插入计划：先落向量/节点，再统一连边，减少逐条提交的重复开销。
    /// EN: Batch apply plans (materialize nodes first, then connect edges) to reduce per-plan overhead.
    pub(crate) fn apply_insert_plans(&mut self, mut plans: Vec<InsertPlan>) {
        if plans.is_empty() {
            return;
        }
        let metric = self.metric;
        let mut max_level = self.max_layer;
        let mut new_entry = self.entry_point;

        // 第一遍仅落节点，确保第二遍连边时两端节点都已可见。
        // EN: First pass materializes all nodes so both edge endpoints exist in second pass.
        for plan in &mut plans {
            self.vectors.insert(plan.id, plan.vector.clone());
            let node = self.new_hnsw_node(plan.level);
            self.nodes.insert(plan.id, node);
            if plan.level > max_level {
                max_level = plan.level;
                new_entry = Some(plan.id);
            } else if new_entry.is_none() {
                new_entry = Some(plan.id);
            }
        }

        for plan in plans {
            let id = plan.id;
            for (lc, neigh) in plan.layers_neighbors {
                let m_lc = layer_m(self.m, lc);
                for &e in &neigh {
                    add_bidirectional_edge(
                        &self.vectors,
                        &mut self.link_arena,
                        &mut self.nodes,
                        self.m,
                        id,
                        e,
                        lc,
                        m_lc,
                        metric,
                    );
                }
            }
        }

        self.max_layer = max_level;
        self.entry_point = new_entry;
    }

    /// Remove a point from the graph so the same `id` can be inserted again (idempotent upsert).
    /// 删除节点以便同一 `id` 可再次 `insert`（幂等覆盖写入）。
    pub fn remove_node(&mut self, id: VectorId) -> bool {
        let removed = match self.nodes.remove(&id) {
            Some(node) => node,
            None => return false,
        };
        let removed_level = removed.max_layer as usize;
        let m = self.m;
        // 在释放 `link_arena` 子块前拷出各层真实邻接，避免回收后再读。
        // EN: Snapshot per-layer links before `free_link_block` can reuse the slot.
        let max_l = removed.max_layer as usize;
        let per_layer: Vec<Vec<VectorId>> = (0..=max_l)
            .map(|layer| {
                let s = removed.layer_start(m, layer);
                let l = removed.layer_len[layer] as usize;
                if l == 0 {
                    return Vec::new();
                }
                self.link_arena[s..s + l].to_vec()
            })
            .collect();
        self.free_link_block(removed.link_base, max_l);
        self.vectors.remove(&id);
        let removed_entry = self.entry_point == Some(id);
        // 仅访问被删节点邻接，避免对全图做 O(N) 扫描；Flat-Link 上用 `remove_link_id` 等价于 `swap_remove`。
        // EN: Only touch neighbors; flat `remove_link_id` == old `Vec::swap_remove` for an edge.
        let mut uniq = AHashSet::new();
        for (layer_idx, neighbors) in per_layer.into_iter().enumerate() {
            if neighbors.len() <= 16 {
                for nid in neighbors {
                    if let Some(n) = self.nodes.get_mut(&nid) {
                        remove_link_id(&mut self.link_arena, n, m, layer_idx, id);
                    }
                }
                continue;
            }
            if neighbors.len() > 24 {
                let mut sorted = neighbors;
                sorted.sort_unstable();
                sorted.dedup();
                for nid in sorted {
                    if let Some(n) = self.nodes.get_mut(&nid) {
                        remove_link_id(&mut self.link_arena, n, m, layer_idx, id);
                    }
                }
            } else {
                uniq.clear();
                let cap = uniq.capacity();
                if cap < neighbors.len() {
                    uniq.reserve(neighbors.len() - cap);
                }
                for nid in neighbors {
                    if !uniq.insert(nid) {
                        continue;
                    }
                    if let Some(n) = self.nodes.get_mut(&nid) {
                        remove_link_id(&mut self.link_arena, n, m, layer_idx, id);
                    }
                }
            }
        }
        if removed_entry {
            // 入口点必须是「全局最高层」的节点（经典 HNSW 不变量：逐层下降要求
            // 入口点覆盖 max_layer）。此前取 min id 可能选中低层节点——
            // search_layer 对 ep 无层级检查，会把它放进上层搜索的结果集 w，
            // insert 的下降链把它带到更低层后被选为邻居连边；低层节点没有该层
            // 邻接表，反向边写入被静默跳过，产生单边悬空边（fuzz 实证）。
            // 平手取最小 id 保证可复现。
            // EN: The entry point must be a top-level node (classic HNSW
            // invariant: layer descent assumes it spans max_layer). The old
            // min-id pick could select a low-level node — search_layer admits
            // ep into result set w with no level check, the insert descent
            // chain carries it down, it gets linked as a neighbor at a layer
            // it does not have, and the back-edge push is silently skipped,
            // yielding a one-sided dangling edge (proven by fuzz). Ties break
            // to the smallest id for determinism.
            self.entry_point = self
                .nodes
                .iter()
                .map(|(nid, n)| (n.max_layer, std::cmp::Reverse(*nid)))
                .max()
                .map(|(_, std::cmp::Reverse(nid))| nid);
        }
        if removed_entry || removed_level >= self.max_layer {
            self.max_layer = self
                .nodes
                .keys()
                .filter_map(|nid| self.node_level(*nid))
                .max()
                .unwrap_or(0);
        }
        // 层数可能缩小：清除已不存在层的历史告警位，后续该层重新出现并断裂时
        // 能再次告警。空图必须整体清零——`max_layer` 会回落为 0，但位 0 若
        // 残留，下一代图重建后的首个 L0 断裂会被静默（`ensure_connectivity`
        // 在无入口点时提前返回，无法清除它）。
        // EN: The layer count may shrink: clear stale warn bits for layers that
        // no longer exist so a future break in that layer warns again. An empty
        // graph must reset the whole mask — `max_layer` falls back to 0, and a
        // leftover bit 0 would silence the next generation's first L0 break
        // (`ensure_connectivity` returns early without an entry point and
        // cannot clear it).
        if self.nodes.is_empty() {
            self.incomplete_warn_mask = 0;
        } else {
            self.incomplete_warn_mask &= Self::warn_mask_upto(self.max_layer);
        }
        true
    }

    /// 查询侧上层下降束宽缺省值：逐层贪心（ef=1）在稠密成簇数据上会卡在错误
    /// 簇的局部极小。早期实测（另一随机语料）显示 50k×512 / 64 簇下 ef=1 时
    /// 13/64 查询错落，束宽 64 为 0/64；但 sm 生产形态语料（mulberry32
    /// 0xA11CE）在 layer-1 有 3118 个节点时，束宽 64/128 仍让 #45 落入错误簇
    /// （返回簇 41、相似度 0.887 vs 真簇 0.052），束宽 256/392 才恢复；
    /// 该语料 L0 已 100% 可达、layer-2 以上也全连通，缺陷只在层内贪心束的
    /// 覆盖不足。因此缺省束宽不再是固定值，而按层规模自适应放大
    /// （见 [`Self::upper_layer_ef_for`]）。
    /// EN: Default upper-layer descent beam: greedy per-layer search (ef=1)
    /// stalls in the wrong cluster's local minimum on dense clustered data. An
    /// earlier corpus measured 13/64 wrong landings at ef=1 and 0/64 at width
    /// 64; however, the production-shaped sm corpus (mulberry32 0xA11CE) with
    /// 3118 layer-1 nodes still lands query #45 in the wrong cluster at width
    /// 64/128 (returned cluster 41, similarity 0.887 vs 0.052 in the true
    /// cluster) and only recovers at width 256/392 — while L0 is 100% reachable
    /// and layer >=2 is fully connected. The defect is insufficient beam
    /// coverage *within* a layer, so the default is no longer a fixed width;
    /// it scales with the layer size (see [`Self::upper_layer_ef_for`]).
    const DEFAULT_UPPER_LAYER_EF: usize = 64;

    /// 自适应上层束宽的分母：估计层节点数 / 该值作为束宽下限。50k 语料
    /// layer-1 估计 3125，得 391，刚好覆盖上面实测所需的 256~392 区间。
    /// EN: Divisor for the adaptive upper beam: estimated layer nodes / this
    /// value is the width floor. At 50k, layer-1 estimates to 3125 -> 391,
    /// covering the measured 256..392 requirement.
    const UPPER_LAYER_EF_DIVISOR: usize = 8;

    /// 自适应束宽封顶 4096。250k 生产形态实测：layer-1 约 15.6k 节点时
    /// 1024（6.6% 覆盖）仍能把查询送进正确簇；4096 让 1M 图（layer-1 约
    /// 62.5k）保持同一比例（6.5%），避免覆盖率随 N 线性衰减。默认路径的
    /// 最坏查询开销仍由该封顶约束。
    /// EN: Adaptive width is capped at 4096. Measured on the 250k
    /// production-shaped corpus: ~15.6k layer-1 nodes with 1024 (6.6%
    /// coverage) still lands queries in the correct cluster, and 4096 keeps a
    /// 1M graph (~62.5k layer-1 nodes) at the same ratio (6.5%) instead of
    /// letting coverage decay linearly with N. The cap still bounds the default
    /// worst-case query cost.
    const MAX_UPPER_LAYER_EF: usize = 4096;

    /// 查询侧上层下降束宽：`HOUYI_HNSW_UPPER_EF`（1..=4096，非法值警告后回落
    /// 64，进程内一次读取）作为**下限**，实际宽度按层规模自适应放大：
    /// 层节点数估计为 `nodes.len() / m^layer`（层分配几何分布），束宽取
    /// `max(env, ceil(estimate / 8))`，自适应部分封顶 4096；显式 env 高于
    /// 自适应宽度时按 env 生效。`HOUYI_HNSW_UPPER_EF_ADAPTIVE=0`（默认开）关闭
    /// 自适应，束宽恒为 env（延迟敏感部署/回滚用）。
    /// 为什么自适应：固定 64 在 50k 生产语料上仍会落入错误簇（见
    /// [`Self::DEFAULT_UPPER_LAYER_EF`]），而更大的图上层节点更多、需要更宽的
    /// 束。构图上侧的逐层下降**保持**贪心 ef=1（见
    /// `insert_prepared_with_visited_entry_hint` 与
    /// `plan_insert_with_level_unchecked*` 处的注释）：加宽插入侧下降会成倍
    /// 放大每点 `ef_construction` 的候选扩展成本并改变全图构建语义，而实测
    /// 缺陷在查询入口落入错误簇，因此只调整查询侧。
    /// EN: Query-side upper-layer descent beam: `HOUYI_HNSW_UPPER_EF`
    /// (1..=4096; invalid values warn and fall back to 64; read once per
    /// process) is a **floor**, and the actual width scales with the layer:
    /// layer nodes are estimated as `nodes.len() / m^layer` (geometric level
    /// distribution) and the width is `max(env, ceil(estimate / 8))`, with the
    /// adaptive part capped at 4096; an explicit env above the adaptive width
    /// wins as-is.
    /// `HOUYI_HNSW_UPPER_EF_ADAPTIVE=0` (on by default) disables adaptation
    /// and pins the width to env exactly (latency-sensitive deployments /
    /// rollback). Why adaptive: a fixed 64 still lands in the wrong cluster on
    /// the 50k production corpus (see [`Self::DEFAULT_UPPER_LAYER_EF`]), and
    /// larger graphs have more upper nodes needing a wider beam. Build/plan
    /// descents **stay** greedy at ef=1 (see the comments at
    /// `insert_prepared_with_visited_entry_hint` and
    /// `plan_insert_with_level_unchecked*`): widening the insert-side descent
    /// multiplies the per-point `ef_construction` candidate expansion and
    /// changes whole-graph build semantics, while the measured defect was the
    /// query entry landing in a wrong cluster — so only the query side changes.
    fn upper_layer_ef() -> usize {
        static CFG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CFG.get_or_init(|| {
            Self::parse_upper_layer_ef(std::env::var("HOUYI_HNSW_UPPER_EF").ok().as_deref())
        })
    }

    /// 自适应开关：`HOUYI_HNSW_UPPER_EF_ADAPTIVE`（缺省开；`0/false/off/no`
    /// 关闭），进程内一次读取。
    /// EN: Adaptation switch: `HOUYI_HNSW_UPPER_EF_ADAPTIVE` (on by default;
    /// `0/false/off/no` disables), read once per process.
    fn adaptive_upper_layer_ef_enabled() -> bool {
        static CFG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CFG.get_or_init(|| env_bool("HOUYI_HNSW_UPPER_EF_ADAPTIVE", true))
    }

    /// 指定层的实际查询束宽（自适应开启时 env 为下限并按层规模放大；关闭时
    /// 严格等于 env）。
    /// EN: Actual query beam for a specific layer (env as a floor, widened by
    /// layer size when adaptation is on; exactly env when disabled).
    fn upper_layer_ef_for(&self, layer: usize) -> usize {
        Self::effective_upper_layer_ef(
            Self::upper_layer_ef(),
            Self::adaptive_upper_layer_ef_enabled(),
            self.nodes.len(),
            self.m,
            layer,
        )
    }

    /// 纯函数形式的最终束宽（供单元测试钉住回滚分支）：自适应关闭时严格返回
    /// `base`，开启时按层规模放大。环境开关经 `OnceLock` 进程内记忆，无法在
    /// 同一测试二进制里安全切换，因此把决策抽成纯函数而不是直接测 env。
    /// EN: Pure final-width decision (unit tests pin the rollback branch):
    /// exactly `base` when adaptation is off, layer-scaled when on. The env
    /// switch is memoized process-wide via `OnceLock` and cannot be flipped
    /// safely inside one test binary, so the decision is a pure function.
    fn effective_upper_layer_ef(
        base: usize,
        adaptive: bool,
        total_nodes: usize,
        m: usize,
        layer: usize,
    ) -> usize {
        if !adaptive {
            return base;
        }
        Self::compute_upper_layer_ef(base, total_nodes, m, layer)
    }

    /// 纯函数形式的自适应束宽计算（供单元测试钉住公式）：
    /// `max(base, ceil(estimate / 8))`，`estimate = total_nodes / m^layer`，
    /// 封顶 4096。`m=0` 视作 1（构造器不会产生 0，防御除零）。
    /// EN: Pure adaptive-width formula (unit tests pin it): `max(base,
    /// ceil(estimate / 8))` with `estimate = total_nodes / m^layer`, capped at
    /// 4096. `m=0` is treated as 1 (constructors never produce 0; division
    /// guard).
    fn compute_upper_layer_ef(base: usize, total_nodes: usize, m: usize, layer: usize) -> usize {
        let m = m.max(1);
        let mut estimate = total_nodes;
        for _ in 0..layer {
            estimate /= m;
        }
        let adaptive = estimate
            .div_ceil(Self::UPPER_LAYER_EF_DIVISOR)
            .clamp(1, Self::MAX_UPPER_LAYER_EF);
        base.max(adaptive)
    }

    /// 解析 `HOUYI_HNSW_UPPER_EF` 原始值：`1..=4096` 原样采用（更大值截断），
    /// 缺失或非法值回落缺省 64 并警告一次（绝不静默降到 0/关）。
    /// EN: Parse the raw `HOUYI_HNSW_UPPER_EF` value: `1..=4096` is used as-is
    /// (larger values clamp), missing or invalid values fall back to the default
    /// 64 with one warning (never silently drops to 0/off).
    fn parse_upper_layer_ef(raw: Option<&str>) -> usize {
        let Some(raw) = raw else {
            return Self::DEFAULT_UPPER_LAYER_EF;
        };
        match raw.trim().parse::<usize>() {
            Ok(v) if v >= 1 => v.min(Self::MAX_UPPER_LAYER_EF),
            _ => {
                tracing::warn!(value = raw, "invalid HOUYI_HNSW_UPPER_EF; using default 64");
                Self::DEFAULT_UPPER_LAYER_EF
            }
        }
    }

    /// 多层贪心搜索 + 第 0 层 beam；返回排序后的 (id, 距离)，距离最小者在前。
    /// EN: Multi-layer greedy search + layer-0 beam; returns sorted (id, distance), smallest first.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(VectorId, f32)> {
        if self.dims > 0 && query.len() != self.dims {
            return Vec::new();
        }
        let ep = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };
        let metric = self.metric;
        let prepared = prepare_search_query(metric, self.quantization, query);
        let query = prepared.context(self.quantization);
        let max_layer = self.max_layer;
        let mut current_ep = ep;

        let mut visited = HnswVisitTracker::with_capacity(
            self.ef_search.max(k).max(1).saturating_mul(16).max(128),
        );
        for layer in (1..=max_layer).rev() {
            let w = search_layer_inner(
                &mut visited,
                &self.vectors,
                &self.link_arena,
                &self.nodes,
                self.m,
                query,
                current_ep,
                self.upper_layer_ef_for(layer),
                layer,
                metric,
            );
            if let Some((closest, _)) = w.first() {
                current_ep = *closest;
            }
        }

        let ef = self.ef_search.max(k).max(1);
        let mut candidates = search_layer_inner(
            &mut visited,
            &self.vectors,
            &self.link_arena,
            &self.nodes,
            self.m,
            query,
            current_ep,
            ef,
            0,
            metric,
        );

        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(k);
        candidates
    }

    /// 带 filter 的多层搜索：遍历时仅访问 filter_set 内节点（图内过滤）。
    /// Filtered multi-layer search: only visits nodes in filter_set during traversal.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter_set: &std::collections::HashSet<VectorId>,
    ) -> Vec<(VectorId, f32)> {
        if self.dims > 0 && query.len() != self.dims {
            return Vec::new();
        }
        // 确保 entry_point 在 filter 内，否则从 filter 中找一个存在的节点替代。
        // 选取 filter 中 level 最高的节点作为入口，保证多层搜索效率。
        // Ensure entry point is in filter; if not, pick the highest-level filter node.
        let default_ep = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };
        let ep = if filter_set.contains(&default_ep) {
            default_ep
        } else {
            // 从 filter 中找 level 最高的节点作为替代入口
            // EN: Find the highest-level node in the filter as the alternate entry.
            filter_set
                .iter()
                .filter_map(|id| self.node_level(*id).map(|lvl| (*id, lvl)))
                .max_by_key(|(_, lvl)| *lvl)
                .map(|(id, _)| id)
                .unwrap_or(default_ep)
        };
        let metric = self.metric;
        let prepared = prepare_search_query(metric, self.quantization, query);
        let query = prepared.context(self.quantization);
        let max_layer = self.max_layer;
        let mut current_ep = ep;

        let mut visited = HnswVisitTracker::with_capacity(
            self.ef_search.max(k).max(1).saturating_mul(16).max(128),
        );
        // 下降起点钳制到所选 ep 的层级跨度：filter 排除图入口点时,替代入口取自
        // filter 集,其层级可能低于 max_layer——此前上层搜索对它是「空 w 空转、
        // current_ep 不变」的合法 no-op;直接从其持有层开始语义完全等价,且保
        // search_layer_inner_filtered 入口「ep 缺层 = 腐败信号」的日志/断言不被
        // 合法替代入口路径触发。
        // EN: Clamp the descent start to the chosen ep's span. A substitute entry
        // (picked from the filter set) may sit below max_layer; upper-layer calls
        // used to be legal no-ops (empty w, ep unchanged), so starting at its own
        // top layer is semantically identical — and keeps the inner guard's
        // "ep lacks layer ⇒ corruption" premise truthful.
        let ep_top = self.node_level(ep).unwrap_or(0);
        for layer in (1..=max_layer.min(ep_top)).rev() {
            let w = search_layer_inner_filtered(
                &mut visited,
                &self.vectors,
                &self.link_arena,
                &self.nodes,
                self.m,
                query,
                current_ep,
                self.upper_layer_ef_for(layer),
                layer,
                metric,
                filter_set,
            );
            if let Some((closest, _)) = w.first() {
                current_ep = *closest;
            }
        }

        let ef = self.ef_search.max(k).max(1);
        let mut candidates = search_layer_inner_filtered(
            &mut visited,
            &self.vectors,
            &self.link_arena,
            &self.nodes,
            self.m,
            query,
            current_ep,
            ef,
            0,
            metric,
            filter_set,
        );

        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(k);
        candidates
    }

    /// 导出图中所有向量。
    /// 供 collection 快照（clustered 后端的构建态数据）导出使用。
    /// EN: Exports all graph vectors. Used by collection snapshots to export the
    /// built state of the clustered backend.
    pub fn export_vectors(&self) -> Vec<(VectorId, Arc<StoredVector>)> {
        self.vectors
            .iter()
            .map(|(id, v)| (*id, v.source_arc()))
            .collect()
    }

    /// 重置图（例如 benchmark 中删除 collection 后）。
    /// EN: Reset graph (e.g. after collection drop in benchmarks).
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.link_arena.clear();
        self.link_free.clear();
        self.vectors.clear();
        self.entry_point = None;
        self.max_layer = 0;
        // 新一代图不得继承上一代的告警位，否则同层真实断裂会被静默。
        // EN: A rebuilt graph must not inherit the previous generation's warn
        // bits, or a real break in that layer would stay silent.
        self.incomplete_warn_mask = 0;
    }

    /// 图优化：对各级邻居执行启发式重选，剪除冗余边，提升 recall/内存比。
    /// Graph compaction: re-select neighbors with heuristic, prune redundant edges.
    pub fn optimize_compact(&mut self) -> usize {
        let metric = self.metric;
        let m = self.m;
        let mut total_pruned = 0usize;
        let node_ids: Vec<VectorId> = self.nodes.keys().copied().collect();

        for &nid in &node_ids {
            let max_level = match self.nodes.get(&nid) {
                Some(n) => n.max_layer as usize,
                None => continue,
            };

            for layer in 0..=max_level {
                let max_deg = layer_m(m, layer);
                let current_len = match self.nodes.get(&nid) {
                    Some(n) if layer < n.level_count() => n.layer_len[layer] as usize,
                    _ => 0,
                };
                if current_len <= max_deg {
                    continue;
                }

                // 按到宿主节点的距离给现有邻居打分。
                // EN: Score existing neighbors by distance from host.
                let host = match self.vectors.get(&nid) {
                    Some(v) => v.as_graph_ref(),
                    None => continue,
                };
                let neighbor_ids: Vec<VectorId> = self.neighbors_at_layer(nid, layer);
                let mut scored: Vec<(VectorId, f32)> = neighbor_ids
                    .iter()
                    .filter_map(|&nbr| {
                        self.vectors
                            .get(&nbr)
                            .map(|v| (nbr, metric_pair_stored(metric, host, v.as_graph_ref())))
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });

                // 用启发式重新选择。
                // EN: Re-select with heuristic.
                let keep =
                    select_neighbors_heuristic_from_sorted(&self.vectors, &scored, max_deg, metric);
                let pruned = current_len.saturating_sub(keep.len());
                total_pruned += pruned;

                // 写回剪枝后的邻居列表。
                // EN: Write back the pruned neighbor list.
                if let Some(n) = self.nodes.get_mut(&nid) {
                    if layer < n.level_count() {
                        let st = n.layer_start(m, layer);
                        let cap = link_cap(m, layer);
                        for (i, &kid) in keep.iter().enumerate() {
                            if i >= cap {
                                break;
                            }
                            self.link_arena[st + i] = kid;
                        }
                        // 清空剩余槽位。
                        // EN: Clear remaining slots.
                        for i in keep.len().min(cap)..current_len.min(cap) {
                            self.link_arena[st + i] = LINK_PAD;
                        }
                        n.layer_len[layer] = (keep.len().min(cap)) as u16;
                    }
                }
                // 与 `prune_neighbors` 同一不变量：丢边必须对称（摘除被弃邻居的反向边），
                // 否则留下 remove_node 扫不到的悬空链接。遍历原始邻接表 neighbor_ids
                // 而非 scored——scored 滤掉了向量缺失的悬空邻居，其反向边会漏清。
                // EN: Same invariant as `prune_neighbors` — drops must be symmetric,
                // or dangling back-edges survive that remove_node cannot reach.
                // Iterate the raw neighbor_ids, not scored: scored excludes neighbors
                // whose vector is missing, and their back-edges would be skipped.
                for &kid in &neighbor_ids {
                    if keep.contains(&kid) {
                        continue;
                    }
                    if let Some(kn) = self.nodes.get_mut(&kid) {
                        remove_link_id(&mut self.link_arena, kn, m, layer, nid);
                    }
                }
            }
        }

        // Compact link_arena: rebuild from scratch to eliminate fragmentation.
        // 返回值是「释放字节数」而非边数——不得并入 total_pruned（边数语义），
        // 否则调用方按边数扣减 graph_edges_built 指标会被字节数污染。
        // EN: The return value is freed BYTES, not edges — do not fold into
        // total_pruned (edge-count semantics), or callers subtracting it from
        // the graph_edges_built metric get corrupted by a byte count.
        let _ = self.compact_link_arena();

        total_pruned
    }

    /// 压缩 link_arena：移除碎片，更新所有 link_base 指针。
    /// Compact link_arena to eliminate fragmentation from remove_node.
    fn compact_link_arena(&mut self) -> usize {
        let m = self.m;
        let mut new_arena = Vec::new();

        // 计算新偏移 + 拷贝邻接数据（活跃条目 + 层内 padding，布局与原分配一致）。
        // EN: Compute the new offset + copy adjacency (active entries + padding, same layout as before).
        for node in self.nodes.values_mut() {
            let max_level = node.max_layer as usize;
            let old_base = node.link_base;
            let new_base = new_arena.len();
            node.link_base = new_base;

            // 以下标枚举层，保留 layer 的偏移计算；clippy 1.100 要求用
            // enumerate 而非裸范围循环。
            // EN: Enumerate layers by index to keep the offset math; clippy
            // 1.100 requires enumerate instead of a bare range loop.
            for (layer, &used_raw) in node.layer_len.iter().enumerate().take(max_level + 1) {
                let old_start = old_base + layer_start_offset(m, layer);
                let used = used_raw as usize;
                let cap = link_cap(m, layer);
                for i in 0..used.min(cap) {
                    new_arena.push(self.link_arena[old_start + i]);
                }
                let pad = cap.saturating_sub(used);
                new_arena.resize(new_arena.len() + pad, LINK_PAD); // padding keeps alignment
            }
        }

        // 释放量 = 旧 arena 与新 arena 的长度差（= 全部空闲块）换成字节。
        // 旧实现逐节点累加「旧块尾 − 新块首」，该量既无意义又会在 AHashMap
        // 任意迭代序下减法溢出（debug panic / release 环绕），fuzz 实证。
        // EN: Freed bytes = (old − new) arena length (= all free-listed blocks)
        // in bytes. The old per-node accumulation of "old block end − new base"
        // was meaningless and underflowed under AHashMap's arbitrary iteration
        // order (debug panic / release wraparound) — proven by fuzz.
        let old_len = self.link_arena.len();
        let freed = (old_len - new_arena.len()) * std::mem::size_of::<VectorId>();
        self.link_arena = new_arena;
        self.link_free.clear(); // old free list invalidated
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn hnsw_visit_tracker_mimics_hashset_per_layer() {
        let mut t = HnswVisitTracker::with_capacity(4);
        t.begin_layer_search();
        assert!(t.try_visit(1));
        assert!(!t.try_visit(1));
        assert!(t.try_visit(2));
        t.begin_layer_search();
        assert!(t.try_visit(1));
        assert!(!t.try_visit(1));
    }

    /// 查询侧上层束宽的环境解析：缺省 64、1..=4096 原样、越界截断、非法值
    /// 回落缺省（绝不静默关成 0）。
    /// EN: Env parsing for the query-side upper beam: default 64, 1..=4096
    /// as-is, overflow clamps, invalid falls back to the default (never
    /// silently becomes 0/off).
    #[test]
    fn upper_layer_ef_env_parsing() {
        assert_eq!(HnswGraph::parse_upper_layer_ef(None), 64);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some(" 64 ")), 64);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some("1")), 1);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some("1024")), 1024);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some("4096")), 4096);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some("9999")), 4096);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some("0")), 64);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some("abc")), 64);
        assert_eq!(HnswGraph::parse_upper_layer_ef(Some("")), 64);
    }

    /// 自适应束宽公式：层规模估计 = nodes/m^layer，束宽取估计值/8，且不低于
    /// 环境下限、不超过 4096；50k 图上 layer-1 必须达到实测所需的 256~392。
    /// EN: Adaptive beam formula: layer estimate = nodes/m^layer, width =
    /// estimate/8, floored by the env value and capped at 4096; on a 50k graph
    /// layer-1 must reach the measured 256..392 requirement.
    #[test]
    fn adaptive_upper_layer_ef_scales_with_layer_size() {
        // 50000/16=3125，3125/8 向上取整 391（实测 256 已修复、128 不足）。
        // EN: 50000/16=3125, ceil(3125/8)=391 (measured: 256 fixes it, 128 does not).
        assert_eq!(HnswGraph::compute_upper_layer_ef(64, 50_000, 16, 1), 391);
        // 50000/256=195，195/8=25，低于下限 64。
        // EN: 50000/256=195, 195/8=25, below the 64 floor.
        assert_eq!(HnswGraph::compute_upper_layer_ef(64, 50_000, 16, 2), 64);
        // 4653/16=290，290/8=37，低于下限 64（小图保持 64）。
        // EN: 4653/16=290, 290/8=37, below the 64 floor (small graphs stay at 64).
        assert_eq!(HnswGraph::compute_upper_layer_ef(64, 4_653, 16, 1), 64);
        // 大图自适应放大但封顶 4096。
        // EN: Large graphs widen adaptively but cap at 4096.
        assert_eq!(
            HnswGraph::compute_upper_layer_ef(64, 10_000_000, 16, 1),
            4096
        );
        // 环境值只能抬高下限，不能压到自适应宽度以下。
        // EN: The env value can only raise the floor, never go below the adaptive width.
        assert_eq!(HnswGraph::compute_upper_layer_ef(1, 50_000, 16, 1), 391);
        assert_eq!(HnswGraph::compute_upper_layer_ef(1024, 50_000, 16, 1), 1024);
        // m=0 防御除零（按 1 处理，估计值不缩小 → 封顶 4096）；高层估计值
        // 降到 0 时仍有下限。
        // EN: m=0 guards against division by zero (treated as 1, so the
        // estimate does not shrink -> capped at 4096); a zero layer estimate
        // still keeps the floor.
        assert_eq!(HnswGraph::compute_upper_layer_ef(64, 50_000, 0, 1), 4096);
        assert_eq!(HnswGraph::compute_upper_layer_ef(64, 100, 16, 5), 64);
        // 回滚分支：关闭自适应时严格等于 env（即使 env 低于所需宽度/高于封顶）；
        // 开启时按层规模放大，显式高 env 原样生效。
        // EN: Rollback branch: adaptation off returns env exactly (even when it
        // is below the needed width or above the cap); adaptation on widens by
        // layer size and an explicit high env wins as-is.
        assert_eq!(
            HnswGraph::effective_upper_layer_ef(64, false, 50_000, 16, 1),
            64
        );
        assert_eq!(
            HnswGraph::effective_upper_layer_ef(64, true, 50_000, 16, 1),
            391
        );
        assert_eq!(
            HnswGraph::effective_upper_layer_ef(2048, false, 50_000, 16, 1),
            2048
        );
        assert_eq!(
            HnswGraph::effective_upper_layer_ef(2048, true, 50_000, 16, 1),
            2048
        );
    }

    /// 测试侧逐层可达集合（与 `ensure_layer_connectivity` 同一 BFS 口径）。
    /// EN: Test-side per-layer reachable set (same BFS as the sweep).
    fn layer_reachable_ids(g: &HnswGraph, layer: usize) -> HashSet<VectorId> {
        let ep = g.entry_point().expect("entry point");
        let mut seen = HashSet::new();
        seen.insert(ep);
        let mut stack = vec![ep];
        while let Some(cur) = stack.pop() {
            for n in g.neighbors_at_layer(cur, layer) {
                if g.node_level(n).map(|l| l >= layer).unwrap_or(false) && seen.insert(n) {
                    stack.push(n);
                }
            }
        }
        seen
    }

    /// 上层孤立节点必须被全层兜底接回：L0 全可达但 layer-1 有孤立点时，查询
    /// 上层导航仍可能落入错误盆地；兜底后该层从入口点全可达，且重复调用幂等。
    /// EN: An isolated upper-layer node must be reconnected by the all-layer
    /// backstop: with L0 fully reachable but a layer-1 orphan, upper
    /// navigation can still land in a wrong basin; the sweep must restore
    /// layer-1 reachability and be idempotent on repeat calls.
    #[test]
    fn ensure_connectivity_repairs_isolated_upper_layer_node() {
        let mut g = HnswGraph::new(4, DistanceMetric::Cosine, 8, 64, 32, 0xA11CE, 10);
        let mut rng = StdRng::seed_from_u64(0x1234);
        for i in 0..256_u64 {
            let v: Vec<f32> = (0..4).map(|_| rng.gen::<f32>()).collect();
            g.insert(i, v);
        }
        let ep = g.entry_point().expect("entry point");
        // 选一个非入口、持有 layer-1 且确有 layer-1 邻接的节点，摘掉它的全部
        // layer-1 边（含对端反向边），制造一个真实的层内孤立分量。
        // EN: Pick a non-entry node that owns layer 1 and has layer-1 links,
        // then remove all its layer-1 edges (including the reverse side) to
        // create a genuine in-layer isolated component.
        let victim = g
            .nodes
            .iter()
            .filter(|(id, node)| **id != ep && node.level_count() > 1 && node.layer_len[1] > 0)
            .map(|(id, _)| *id)
            .min()
            .expect("some non-entry node has layer-1 links");
        let nbrs = g.neighbors_at_layer(victim, 1);
        for nbr in &nbrs {
            if let Some(n) = g.nodes.get_mut(nbr) {
                remove_link_id(&mut g.link_arena, n, g.m, 1, victim);
            }
        }
        g.nodes.get_mut(&victim).expect("victim").layer_len[1] = 0;
        assert!(
            !layer_reachable_ids(&g, 1).contains(&victim),
            "constructing the defect: victim must be unreachable at layer 1"
        );
        let layer_nodes = g.nodes.iter().filter(|(_, n)| n.level_count() > 1).count();
        let added = g.ensure_connectivity();
        assert!(added >= 1, "sweep must add at least the layer-1 bridge");
        let reachable = layer_reachable_ids(&g, 1);
        assert!(reachable.contains(&victim), "victim must be reconnected");
        assert_eq!(
            reachable.len(),
            layer_nodes,
            "every layer-1 node must be reachable after the sweep"
        );
        // 幂等：第二次兜底不得再改图。
        // EN: Idempotent: a second sweep must not change the graph.
        assert_eq!(g.ensure_connectivity(), 0);
        assert_eq!(layer_reachable_ids(&g, 1).len(), layer_nodes);
    }

    /// 整簇形态的上层分量：成员保留内部边、只断外部桥（与整簇孤岛同构）。
    /// 上层度数上限只有 `m/2`，补桥会与内部邻居竞争 `prune_neighbors`，与
    /// 度数 0 的孤点形态是不同代码路径（add-then-prune/换锚点重试）。
    /// EN: Whole-cluster-shaped upper component: members keep their internal
    /// edges and only lose external bridges (isomorphic to a whole-cluster
    /// island). The upper degree cap is only `m/2`, so the bridge competes with
    /// internal neighbors inside `prune_neighbors` — a different path from the
    /// degree-0 orphan (add-then-prune / alternate-anchor retry).
    #[test]
    fn ensure_connectivity_repairs_internal_upper_component() {
        let mut g = HnswGraph::new(4, DistanceMetric::Cosine, 8, 64, 32, 0x5EED, 10);
        let mut rng = StdRng::seed_from_u64(0x4321);
        for i in 0..512_u64 {
            let v: Vec<f32> = (0..4).map(|_| rng.gen::<f32>()).collect();
            g.insert(i, v);
        }
        let ep = g.entry_point().expect("entry point");
        // 选一对确定性的 layer-1 内部边 (a<b)，两端都不是入口点。
        // EN: Pick a deterministic layer-1 internal edge (a<b) whose endpoints
        // are not the entry point.
        let mut pairs: Vec<(VectorId, VectorId)> = g
            .nodes
            .iter()
            .filter(|(id, n)| **id != ep && n.level_count() > 1)
            .flat_map(|(id, _)| {
                g.neighbors_at_layer(*id, 1)
                    .into_iter()
                    .filter(|nbr| *nbr != ep && g.node_level(*nbr).map(|l| l >= 1).unwrap_or(false))
                    .map(|nbr| if *id < nbr { (*id, nbr) } else { (nbr, *id) })
                    .collect::<Vec<_>>()
            })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let (a, b) = pairs
            .into_iter()
            .next()
            .expect("some layer-1 internal edge exists");
        // 摘掉 a/b 的所有外部 layer-1 边（保留 a<->b），构造只有内部边的分量。
        // EN: Strip all external layer-1 edges of a/b (keeping a<->b), leaving a
        // component with only its internal edge.
        for (keep, victim) in [(b, a), (a, b)] {
            for nbr in g.neighbors_at_layer(victim, 1) {
                if nbr == keep {
                    continue;
                }
                if let Some(node) = g.nodes.get_mut(&nbr) {
                    remove_link_id(&mut g.link_arena, node, g.m, 1, victim);
                }
            }
        }
        let m = g.m;
        for (victim, keep) in [(a, b), (b, a)] {
            let start = g.nodes.get(&victim).expect("victim node").layer_start(m, 1);
            g.link_arena[start] = keep;
            g.nodes.get_mut(&victim).expect("victim node").layer_len[1] = 1;
        }
        assert!(g.neighbors_at_layer(a, 1).contains(&b));
        assert!(g.neighbors_at_layer(b, 1).contains(&a));
        let before = layer_reachable_ids(&g, 1);
        assert!(
            !before.contains(&a) && !before.contains(&b),
            "constructing the defect: the internal component must be unreachable at layer 1"
        );
        let layer_nodes = g.nodes.values().filter(|n| n.level_count() > 1).count();
        let added = g.ensure_connectivity();
        assert!(
            added >= 1,
            "sweep must bridge the internal layer-1 component"
        );
        let after = layer_reachable_ids(&g, 1);
        assert!(after.contains(&a) && after.contains(&b));
        assert_eq!(
            after.len(),
            layer_nodes,
            "every layer-1 node must be reachable after the sweep"
        );
        assert_eq!(g.ensure_connectivity(), 0, "sweep must be idempotent");
    }

    /// 告警位图的生命周期：层数缩小时只保留存在的层；`clear()` 清空。
    /// EN: Warn-bitmap lifecycle: shrinking the layer count keeps only existing
    /// layers; `clear()` resets it.
    #[test]
    fn connectivity_warn_mask_tracks_layer_lifecycle() {
        assert_eq!(HnswGraph::warn_mask_upto(0), 0b1);
        assert_eq!(HnswGraph::warn_mask_upto(3), 0b1111);
        assert_eq!(
            HnswGraph::warn_mask_upto(MAX_HNSW_LAYERS - 1),
            (1u64 << MAX_HNSW_LAYERS) - 1
        );
        let mut g = HnswGraph::new(4, DistanceMetric::Cosine, 8, 64, 32, 0xA11CE, 10);
        g.incomplete_warn_mask = 0b111;
        g.clear();
        assert_eq!(g.incomplete_warn_mask, 0, "clear must reset the warn mask");

        // remove_node 令 max_layer 缩小时，高于新 max_layer 的告警位必须被掩掉。
        // EN: When remove_node shrinks max_layer, warn bits above the new
        // max_layer must be cleared.
        let mut g = HnswGraph::new(4, DistanceMetric::Cosine, 8, 64, 32, 0xA11CE, 10);
        let mut rng = StdRng::seed_from_u64(0x9E37_79B9);
        for i in 0..1024_u64 {
            let v: Vec<f32> = (0..4).map(|_| rng.gen::<f32>()).collect();
            g.insert(i, v);
        }
        let old_max = g.max_layer;
        assert!(old_max >= 2, "setup needs at least two upper layers");
        g.incomplete_warn_mask = u64::MAX;
        while g.max_layer == old_max && !g.nodes.is_empty() {
            let top = g
                .nodes
                .iter()
                .filter(|(_, n)| (n.max_layer as usize) == old_max)
                .map(|(id, _)| *id)
                .min()
                .expect("top-layer node exists");
            assert!(g.remove_node(top));
        }
        assert!(
            g.max_layer < old_max,
            "setup must actually shrink the top layer"
        );
        assert_eq!(
            g.incomplete_warn_mask,
            HnswGraph::warn_mask_upto(g.max_layer),
            "warn bits above the shrunk max_layer must be cleared"
        );
        // 空图必须整体清零：残留位 0 会静默下一代图的首次 L0 断裂。
        // EN: An empty graph must reset the whole mask: a leftover bit 0 would
        // silence the next generation's first L0 break.
        let mut tiny = HnswGraph::new(4, DistanceMetric::Cosine, 8, 64, 32, 0xDEAD, 10);
        tiny.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        let id = tiny.entry_point().expect("single node entry");
        tiny.incomplete_warn_mask = 0b1;
        assert!(tiny.remove_node(id));
        assert_eq!(tiny.incomplete_warn_mask, 0, "empty graph resets the mask");
    }

    #[test]
    fn hnsw_self_query_cosine() {
        let seed = 0xA7E5_72D8_B7DB_BEEF;
        let mut g = HnswGraph::new(4, DistanceMetric::Cosine, 8, 64, 32, seed, 10);
        for i in 0_u64..20 {
            let mut v = vec![0.0_f32; 4];
            v[0] = i as f32 * 0.01 + 1.0;
            v[1] = (i * 7) as f32 * 0.001;
            g.insert(i, v);
        }
        for i in 0_u64..20 {
            let mut v = vec![0.0_f32; 4];
            v[0] = i as f32 * 0.01 + 1.0;
            v[1] = (i * 7) as f32 * 0.001;
            let r = g.search(&v, 1);
            assert_eq!(r.first().map(|x| x.0), Some(i));
        }
    }

    #[test]
    fn hnsw_recall_vs_bruteforce_small_l2() {
        let dim = 32_usize;
        let n = 400_usize;
        let k = 10_usize;
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut g = HnswGraph::new(dim, DistanceMetric::L2, 16, 128, 64, 0xC0FFEE, 10);
        let mut stored: Vec<(VectorId, Vec<f32>)> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            stored.push((i as u64, v.clone()));
            g.insert(i as u64, v);
        }
        let mut min_r = 1.0f32;
        for _ in 0..20 {
            let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let ann: HashSet<VectorId> = g.search(&q, k).into_iter().map(|(id, _)| id).collect();
            let mut brute: Vec<(VectorId, f32)> = stored
                .iter()
                .map(|(id, v)| (*id, crate::l2_distance(&q, v)))
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let truth: HashSet<VectorId> = brute.into_iter().take(k).map(|(id, _)| id).collect();
            let recall = ann.intersection(&truth).count() as f32 / k as f32;
            min_r = min_r.min(recall);
        }
        assert!(
            min_r >= 0.4,
            "recall@{} min too low: {:.2} (graph may be broken)",
            k,
            min_r
        );
    }

    #[test]
    fn insert_with_visited_matches_insert_sequential() {
        let dim = 16_usize;
        let n = 80_usize;
        let seed = 0xBEEF;
        let mut g1 = HnswGraph::new(dim, DistanceMetric::Cosine, 8, 64, 32, seed, 10);
        let mut g2 = HnswGraph::new(dim, DistanceMetric::Cosine, 8, 64, 32, seed, 10);
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            vecs.push((0..dim).map(|_| rng.gen::<f32>()).collect());
        }
        for (i, v) in vecs.iter().enumerate() {
            g1.insert(i as u64, v.clone());
        }
        let mut visited =
            HnswVisitTracker::with_capacity(HnswGraph::visited_insert_capacity_hint(64));
        for (i, v) in vecs.iter().enumerate() {
            g2.insert_with_visited(i as u64, v.clone(), &mut visited);
        }
        assert_eq!(g1.edge_count_hint(), g2.edge_count_hint());
        assert_eq!(g1.len(), g2.len());
        assert_eq!(g1.entry_point(), g2.entry_point());
    }

    /// 补救锚点选择：必须跳过被排除节点，且返回的邻居仍有 L0 余量
    /// （饱和锚点会被 `repair_group_links` 在采集阶段过滤，选了也补不上边）。
    /// EN: Repair-anchor selection must skip the excluded id and return a node
    /// with spare L0 capacity (a saturated anchor is filtered out at collection
    /// time by `repair_group_links`, so selecting it would add nothing).
    #[test]
    fn find_spare_l0_anchor_skips_excluded_and_respects_capacity() {
        // 常规图：排除被替换 id，返回的锚点必须是仍有余量者中度数最高者
        // （AHashMap 迭代序随进程随机，选「第一个」不可复现且质量差）。
        // EN: Normal graph: skip the excluded id, and return the spare node with
        // the highest degree (AHashMap iteration order is per-process random,
        // so "first spare" is neither reproducible nor a good anchor).
        let mut g = HnswGraph::new(4, DistanceMetric::Cosine, 8, 64, 32, 0x5EED, 0);
        for id in 1..=4u64 {
            g.insert(id, vec![id as f32, 1.0, 0.0, 0.0]);
        }
        let max_deg = g.l0_max_degree();
        let anchor = g
            .find_spare_l0_anchor(1)
            .expect("small sparse graph has spare nodes");
        assert_ne!(anchor, 1, "anchor must exclude the requested id");
        assert!(g.l0_degree(anchor) < max_deg, "anchor must have capacity");
        let best_spare_deg = (2..=4u64)
            .filter(|id| g.l0_degree(*id) < max_deg)
            .map(|id| g.l0_degree(id))
            .max()
            .expect("at least one spare candidate");
        // 同度破平必须取最小 id：这条把「确定性」本身钉住，而不只是比较度数
        // （本图所有节点度数相同，只比度数任何候选都能通过）。
        // EN: Tie-breaking must pick the smallest id: this pins determinism
        // itself, not just degree (all nodes here share one degree, so any
        // candidate would pass a degree-only comparison).
        let expected = (2..=4u64)
            .find(|id| g.l0_degree(*id) == best_spare_deg)
            .expect("best degree has a candidate");
        assert_eq!(
            anchor, expected,
            "anchor must be the smallest id among best-connected spare nodes"
        );

        // 小上限图（m=1）：验证「返回 None 当且仅当其余节点全部饱和」的契约；
        // 具体是否真的饱和由插入内部决定，因此两种形态都按契约校验。
        // EN: Small-cap graph (m=1): verify "None iff every other node is
        // saturated". Whether the insert internals actually saturate is
        // topology-dependent, so both shapes are checked against the contract.
        let mut tiny = HnswGraph::new(4, DistanceMetric::Cosine, 1, 64, 32, 0xDEAD, 0);
        for id in 1..=5u64 {
            tiny.insert(id, vec![id as f32, 0.0, 0.0, 1.0]);
        }
        let tiny_cap = tiny.l0_max_degree();
        match tiny.find_spare_l0_anchor(1) {
            Some(anchor) => {
                assert_ne!(anchor, 1);
                assert!(
                    tiny.l0_degree(anchor) < tiny_cap,
                    "picker returned a saturated node (degree {}, cap {tiny_cap})",
                    tiny.l0_degree(anchor)
                );
            }
            None => {
                for id in 2..=5 {
                    assert!(
                        tiny.l0_degree(id) >= tiny_cap,
                        "None is only valid when every other node is saturated; \
                         node {id} has degree {} < cap {tiny_cap}",
                        tiny.l0_degree(id)
                    );
                }
            }
        }
    }

    #[test]
    fn remove_node_then_insert_same_id() {
        let mut g = HnswGraph::new(4, DistanceMetric::L2, 8, 64, 32, 0xDEAD, 10);
        let v1 = vec![1.0_f32, 0.0, 0.0, 0.0];
        let v2 = vec![0.0_f32, 1.0, 0.0, 0.0];
        g.insert(1, v1.clone());
        assert!(g.remove_node(1));
        g.insert(1, v2.clone());
        let r = g.search(&v2, 1);
        assert_eq!(r.first().map(|x| x.0), Some(1));
    }

    /// 图链接不变量断言器：任何层的任何边 (a@L → b) 必须满足两条：
    /// (1) b 的节点与向量都存活（无悬空链接）；(2) 反向边 b@L → a 存在
    /// （边对称——remove_node「只扫自身邻接」清理的正确性前提）。
    /// entry_point 必须指向存活节点。
    /// EN: Link-invariant checker: every edge (a@L → b) at any layer requires
    /// (1) b's node and vector both live (no dangling links), and (2) the
    /// back-edge b@L → a exists (symmetry — the premise of remove_node's
    /// "scan only the victim's own lists" cleanup). entry_point must
    /// reference a live node.
    fn assert_link_invariants(g: &HnswGraph) {
        for (id, node) in &g.nodes {
            for layer in 0..node.level_count() {
                let nbrs = node
                    .link_active_ref(&g.link_arena, g.m, layer)
                    .unwrap_or(&[]);
                for &e in nbrs {
                    assert!(
                        g.nodes.contains_key(&e),
                        "dangling link {id}@{layer} -> {e}: node missing"
                    );
                    assert!(
                        g.vectors.contains_key(&e),
                        "dangling link {id}@{layer} -> {e}: vector missing"
                    );
                    let back = g.nodes[&e]
                        .link_active_ref(&g.link_arena, g.m, layer)
                        .unwrap_or(&[]);
                    assert!(
                        back.contains(id),
                        "asymmetric edge {id}@{layer} -> {e}: back-edge missing \
                         (level_count({id})={}, level_count({e})={})",
                        g.node_level(*id).map(|l| l + 1).unwrap_or(0),
                        g.node_level(e).map(|l| l + 1).unwrap_or(0),
                    );
                }
            }
        }
        if let Some(ep) = g.entry_point {
            assert!(g.nodes.contains_key(&ep), "entry point {ep} not in nodes");
            // 入口点必须横跨全图最高层——insert 下降链/上层搜索的前提;
            // remove_node 曾可选中低层入口破坏该不变量(fuzz 实证),此处兜底。
            // EN: The entry point must span the graph's max layer — the premise
            // of insert descent / upper-layer search; remove_node once broke
            // this by re-selecting a low-level entry (proven by fuzz).
            assert_eq!(
                g.node_level(ep),
                Some(g.max_layer),
                "entry point {ep} level != graph max_layer {}",
                g.max_layer
            );
        }
    }

    /// 随机 insert/delete/search/过滤检索/compact 混合 fuzz：
    /// 回归「对称剪枝前 remove_node 留下悬空反向边 → search 索引已删向量 panic」。
    /// 规模足以让 layer-0 邻接超过 max_deg 触发启发式剪枝（不对称丢边的案发点）。
    /// 不变量每轮断言；收尾清空全图，断言无残留节点/向量/入口。
    /// EN: Mixed insert/delete/search/filtered-search/compact fuzz — regression
    /// for "pre-symmetric-pruning remove_node left dangling back-edges → search
    /// indexed a deleted vector and panicked". Large enough to push layer-0
    /// degrees past max_deg so heuristic pruning (the asymmetric-drop crime
    /// scene) actually fires. Invariants asserted every round; the graph is
    /// drained at the end to assert nothing leaks.
    #[test]
    fn remove_fuzz_maintains_link_invariants() {
        let dim = 16_usize;
        let mut rng = StdRng::seed_from_u64(0xD311_9F1E_55AA_0001);
        let mut g = HnswGraph::new(dim, DistanceMetric::L2, 16, 128, 64, 0xF122, 10);
        let mut live: Vec<VectorId> = Vec::new();
        let mut next_id = 1_u64;

        for round in 0..250 {
            for _ in 0..8 {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
                g.insert(next_id, v);
                live.push(next_id);
                next_id += 1;
            }
            for _ in 0..4 {
                if live.len() <= 1 {
                    break;
                }
                let idx = rng.gen_range(0..live.len());
                let id = live.swap_remove(idx);
                assert!(g.remove_node(id), "remove live id {id} must succeed");
            }
            // 普通检索 + 过滤检索各一次（两条搜索路径都必须不 panic）。
            // EN: One plain search + one filtered search (neither search path may panic).
            let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let _ = g.search(&q, 10);
            let filter: HashSet<VectorId> = live.iter().copied().collect();
            let _ = g.search_with_filter(&q, 10, &filter);
            // 中段触发一次 compact（其写回同样走对称清理）。
            // EN: Trigger one compact midway (its write-back also runs symmetric cleanup).
            if round == 120 {
                let _ = g.optimize_compact();
            }
            assert_link_invariants(&g);
        }

        for &id in &live {
            assert!(g.remove_node(id));
        }
        assert_link_invariants(&g);
        assert_eq!(g.len(), 0, "drained graph must have no nodes");
        assert!(g.vectors.is_empty(), "drained graph must have no vectors");
        assert_eq!(g.entry_point(), None);
    }

    /// 过滤检索的替代入口为低层节点时,不得触发入口守卫的腐败信号。
    /// 场景: filter 排除图入口点(层级=max_layer) → 从 filter 内选最高层节点
    /// 作替代入口,其层级 < max_layer——这是替代入口的合法设计,不是腐败。
    /// 回归: 守卫加 debug_assert 之初,该路径被误判腐败,debug 构建下普通
    /// 过滤查询即 panic(OCR 评审实证);修复 = 下降起点钳制到 ep 层级跨度。
    /// 本测试在 debug 构建下若钳制失效即被 debug_assert 拦停。
    /// EN: A low-level substitute entry in filtered search must not trip the
    /// entry guard's corruption signal. When the filter excludes the graph
    /// entry (at max_layer), a substitute picked from the filter may sit below
    /// max_layer — legal by design, not corruption. Regression: the guard's
    /// debug_assert originally misfired on this path (proven by OCR review);
    /// the fix clamps the descent to the ep's span. If the clamp regresses,
    /// the debug_assert stops this test in debug builds.
    #[test]
    fn filtered_search_low_level_substitute_entry_stays_legal() {
        let dim = 8_usize;
        let mut rng = StdRng::seed_from_u64(0x5EED_2024_AA01);
        let mut g = HnswGraph::new(dim, DistanceMetric::L2, 16, 64, 32, 0xBEEF, 10);
        let mut ids = Vec::new();
        for i in 1..=300_u64 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            g.insert(i, v);
            ids.push(i);
        }
        assert!(
            g.max_layer() >= 1,
            "test premise: graph must be multi-layer for the substitute-entry path"
        );
        // filter 只保留层级低于 max_layer 的节点 ⇒ 图入口被排除,替代入口
        // 必为低层节点(触发钳制路径)。
        // EN: The filter keeps only nodes below max_layer ⇒ the graph entry is excluded, and the
        // EN: alternate entry must be a lower-level node (triggering the clamp path).
        let filter: HashSet<VectorId> = ids
            .iter()
            .copied()
            .filter(|id| g.node_level(*id).unwrap_or(0) < g.max_layer())
            .collect();
        assert!(
            !filter.is_empty(),
            "test premise: some nodes must sit below max_layer"
        );
        assert!(!filter.contains(&g.entry_point().unwrap()));

        let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        // debug 构建下: 若守卫被该合法路径触发,debug_assert 在此 panic。
        // EN: In debug builds, if the guard fires on this valid path, debug_assert panics here.
        let r = g.search_with_filter(&q, 5, &filter);
        assert!(!r.is_empty(), "filtered search must still return results");
        assert!(
            r.iter().all(|(id, _)| filter.contains(id)),
            "filtered results must all be in the filter set"
        );
        assert_link_invariants(&g);
    }

    /// 回归(OCR 6a61a4f Finding 1/2):低层 entry_hint 是合法输入,但其层级低于
    /// 新节点插入顶层 upper 时,连边循环曾拿它搜它没有的层——「ep 缺层」守卫在
    /// debug 构建 panic、release 静默留新节点上层零邻居(下降钳制只护住了下降
    /// 循环,误伤被挪到连边阶段)。修复后:hint 够不到 upper 时回退图入口点,
    /// 各层正常连边。
    /// EN: Regression — a low-level entry hint is legal input, but when its
    /// level is below the insert's top layer `upper`, the linking loop used to
    /// search layers the hint does not own: the guard panicked in debug builds
    /// and release builds silently left the new node's upper layers unlinked
    /// (the descent clamp only guarded the descent; the misfire moved to the
    /// linking phase). After the fix, hints that cannot reach `upper` fall back
    /// to the graph entry point and every layer links normally.
    #[test]
    fn insert_with_low_level_entry_hint_links_all_layers() {
        let dim = 8_usize;
        let mut rng = StdRng::seed_from_u64(0x5EED_2024_BB02);
        // m=4 → 层级增长快,少量插入即得 max_layer ≥ 2 的入口点。
        // EN: m=4 → levels grow fast; a few inserts yield an entry with max_layer ≥ 2.
        let mut g = HnswGraph::new(dim, DistanceMetric::L2, 4, 64, 32, 0xF00D, 10);
        let mut level0_hint = None;
        for i in 1..=200_u64 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            g.insert(i, v);
            if level0_hint.is_none() && g.node_level(i) == Some(0) {
                level0_hint = Some(i);
            }
        }
        let hint = level0_hint.expect("test premise: a level-0 node must exist");
        assert!(
            g.max_layer() >= 2,
            "test premise: entry point must own ≥2 layers"
        );

        // 路径 1(显式层级规划):level=1 < max_layer ⇒ upper=1 > hint 层级 0,
        // 无回退时连边循环在 layer 1 触发守卫(debug panic)/零邻居(release)。
        // EN: Path 1 (explicit level plan): level=1 < max_layer ⇒ upper=1 > hint layer 0;
        // EN: without fallback the link loop fires the guard at layer 1 (debug panic) / zero neighbors (release).
        let new_id = 10_001;
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let plan = g.plan_insert_with_level_unchecked(new_id, Arc::new(v), 1, Some(hint));
        let layer1 = plan
            .layers_neighbors
            .iter()
            .find(|(lc, _)| *lc == 1)
            .expect("plan must include layer 1");
        assert!(
            !layer1.1.is_empty(),
            "layer-1 neighbors must be found via entry-point fallback"
        );
        g.apply_insert_plans(vec![plan]);

        // 路径 2(随机层级插入):持续以同一 level-0 hint 插入,凡抽中 level ≥ 1
        // 的新节点在修复前都会触发守卫。
        // 路径 1b(显式 level=2 规划):upper=2 > hint 层级 0,层 1/2 邻居均须
        // 经回退找到(修复前 release 下两层皆空、debug 下守卫 panic)。
        // EN: Path 2 (random-level insert): insert with the same level-0 hint; any node with level ≥ 1
        // EN: fired the guard before the fix.
        // EN: Path 1b (explicit level=2 plan): upper=2 > hint layer 0; layer-1/2 neighbors must both
        // EN: be found via fallback (before the fix: both empty in release, guard panic in debug).
        let v2: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let plan2 = g.plan_insert_with_level_unchecked(30_001, Arc::new(v2), 2, Some(hint));
        for lc in [1_usize, 2] {
            let layer = plan2
                .layers_neighbors
                .iter()
                .find(|(l, _)| *l == lc)
                .unwrap_or_else(|| panic!("plan must include layer {lc}"));
            assert!(
                !layer.1.is_empty(),
                "layer-{lc} neighbors must be found via entry-point fallback"
            );
        }
        g.apply_insert_plans(vec![plan2]);

        // 路径 2(随机层级插入):持续以同一 level-0 hint 插入,凡抽中 level ≥ 1
        // 的新节点在修复前都会触发守卫——debug 构建下不 panic 即断言成立。
        // 注意:不断言「新节点上层前向边非空」——邻接剪枝是对称收敛的,后续
        // 插入可把它的前向边合法剪空(实证:入口点层 1 容量 2,反向剪枝 + 对称
        // 清理会把新节点的前向边一并摘除),那是 HNSW 正常形态而非断链。
        // EN: Path 2 (random-level insert): insert with the same level-0 hint; any node with level ≥ 1
        // EN: fired the guard before the fix — no panic in debug builds means the assertion holds.
        // EN: Note: do not assert upper forward edges are non-empty — pruning converges symmetrically,
        // EN: and later inserts may legitimately prune them away (evidence: entry layer-1 capacity 2;
        // EN: reverse pruning + cleanup removes them), which is normal HNSW, not a broken link.
        let mut saw_upper = false;
        for i in 20_000..20_400_u64 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let mut visited =
                HnswVisitTracker::with_capacity(HnswGraph::visited_insert_capacity_hint(64));
            g.insert_with_visited_entry_hint(i, Arc::new(v), &mut visited, Some(hint));
            if g.node_level(i).expect("inserted node must exist") >= 1 {
                saw_upper = true;
            }
        }
        assert!(saw_upper, "test premise: some insert must draw level ≥ 1");
        assert_link_invariants(&g);
    }

    // ── P2：图侧 int8 量化（插入准备与图内检索） ─────────────────
    // ── EN: P2: graph-side int8 quantization (insert prep + search) ──

    /// 未量化非余弦：主表与图必须共享**同一个** `Arc`（零额外拷贝），构图查询
    /// 借用图向量自身（`graph_query=None`）。
    /// EN: Unquantized non-cosine: main store and graph must share the **same**
    /// `Arc` (zero extra copy) and the build query borrows the graph vector
    /// itself (`graph_query=None`).
    #[test]
    fn prepare_insert_stored_full_non_cosine_shares_arc() {
        let prepared = prepare_insert_stored(
            DistanceMetric::L2,
            VectorDtype::F32,
            Quantization::None,
            vec![0.5, -1.0, 2.0, 0.25],
        );
        assert!(matches!(&prepared.graph, GraphVector::Full(_)));
        assert!(
            Arc::ptr_eq(&prepared.main, &prepared.graph.source_arc()),
            "non-cosine full path must share one Arc"
        );
        assert!(prepared.graph_query.is_none());
    }

    /// int8 非余弦：`source` 与主表共享同一 `Arc`（精确 rescore/恢复要原始值），
    /// codes 独立分配；构图查询仍是原始 f32（非对称量化）。
    /// EN: int8 non-cosine: `source` shares the main-store `Arc` (exact rescoring
    /// and restore need raw values) while codes are separately allocated; the
    /// build query stays raw f32 (asymmetric quantization).
    #[test]
    fn prepare_insert_stored_int8_shares_source_and_separates_codes() {
        let raw = vec![0.25f32, -1.5, 3.0, 2.0];
        let prepared = prepare_insert_stored(
            DistanceMetric::L2,
            VectorDtype::F32,
            Quantization::Int8,
            raw.clone(),
        );
        assert!(
            Arc::ptr_eq(&prepared.main, &prepared.graph.source_arc()),
            "int8 source must share the main-store Arc"
        );
        let GraphVector::Int8 { codes, source } = &prepared.graph else {
            panic!("int8 quantization must produce GraphVector::Int8");
        };
        assert_eq!(
            source.to_f32_vec(),
            raw,
            "source keeps raw main-store values"
        );
        assert_eq!(codes.dims(), raw.len());
        let gq = prepared.graph_query.as_ref().expect("int8 graph query");
        assert_eq!(
            &gq.data[..],
            raw.as_slice(),
            "non-cosine build query is the raw f32 vector"
        );
        for (a, b) in codes.to_f32_vec().iter().zip(raw.iter()) {
            assert!(
                (a - b).abs() <= codes.scale / 2.0 + 1e-6,
                "dequantized error must stay within scale/2: {a} vs {b}"
            );
        }
    }

    /// int8 余弦：量化发生在归一化之后——source 保留原始分量，构图查询为归一化
    /// f32，反量化向量范数仍近似 1（图内 `1 - dot` 口径成立）。
    /// EN: int8 cosine quantizes after normalization — `source` keeps raw
    /// components, the build query is normalized f32, and the dequantized vector
    /// stays unit-norm (the in-graph `1 - dot` contract holds).
    #[test]
    fn prepare_insert_stored_int8_cosine_normalizes_before_quantizing() {
        let raw = vec![3.0f32, 4.0, 0.0, 0.0];
        let prepared = prepare_insert_stored(
            DistanceMetric::Cosine,
            VectorDtype::F32,
            Quantization::Int8,
            raw.clone(),
        );
        let GraphVector::Int8 { codes, source } = &prepared.graph else {
            panic!("int8 quantization must produce GraphVector::Int8");
        };
        assert_eq!(source.to_f32_vec(), raw, "source keeps raw components");
        let gq = prepared.graph_query.as_ref().expect("int8 graph query");
        let qnorm: f32 = gq.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (qnorm - 1.0).abs() <= 1e-5,
            "cosine build query must be normalized: {qnorm}"
        );
        let dnorm: f32 = codes.to_f32_vec().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (dnorm - 1.0).abs() <= 0.05,
            "dequantized cosine vector must stay near unit norm: {dnorm}"
        );
    }

    /// int8 图自查询命中自身（四度量 smoke），并验证 `insert` 公共入口实际落到
    /// `GraphVector::Int8`、source 保留原始值。
    /// EN: int8 self-query hits itself across all four metrics, and the public
    /// `insert` entry actually stores `GraphVector::Int8` with raw `source`.
    #[test]
    fn int8_graph_search_matches_self_for_all_metrics() {
        let dims = 8usize;
        for metric in [
            DistanceMetric::L2,
            DistanceMetric::InnerProduct,
            DistanceMetric::Cosine,
            DistanceMetric::L1,
        ] {
            let mut g = HnswGraph::new_with_dtype_and_quantization(
                dims,
                metric,
                VectorDtype::F32,
                Quantization::Int8,
                8,
                64,
                64,
                42,
                8,
            );
            assert_eq!(g.quantization(), Quantization::Int8);
            let mut stored = Vec::new();
            for i in 0u64..16 {
                // 单位向量保证四种度量下「自身最近」严格成立（同范数不同方向）。
                // EN: Unit vectors make "self is closest" strict under all four
                // metrics (same norm, distinct directions).
                let mut v = vec![0.0f32; dims];
                let b = (i % dims as u64) as usize;
                v[b] = 1.0;
                v[(b + 1) % dims] = i as f32 / 15.0;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                stored.push(v.clone());
                g.insert(i, v);
            }
            for (i, q) in stored.iter().enumerate() {
                let gv = g.graph_vector(i as u64).expect("inserted node");
                let GraphVector::Int8 { codes, source } = gv else {
                    panic!("int8 graph must store GraphVector::Int8");
                };
                assert_eq!(source.to_f32_vec(), *q, "source keeps raw values for {i}");
                assert_eq!(codes.dims(), dims);
                let hit = g.search(q, 1);
                assert_eq!(
                    hit.first().map(|(id, _)| *id),
                    Some(i as u64),
                    "metric={metric:?} self must rank first, got {hit:?}"
                );
            }
        }
    }

    // ── P3：图侧 RaBitQ 量化（插入准备与图内检索） ─────────────────
    // ── EN: P3: graph-side RaBitQ quantization (insert prep + search) ──

    /// RaBitQ 非余弦：`source` 与主表共享同一 `Arc`；构图查询保持度量预处理后
    /// 的 f32 原向量（不旋转、不量化——构图期比较走精确距离）。
    /// EN: RaBitQ non-cosine: `source` shares the main-store `Arc`; the build
    /// query stays the metric-preprocessed f32 vector (not rotated, not
    /// quantized — build-time comparisons are exact).
    #[test]
    fn prepare_insert_stored_rabitq_shares_source_and_keeps_exact_build_query() {
        let raw = vec![0.5f32, -1.25, 2.0, 0.75, -0.5];
        let dims = raw.len();
        let raw_norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let prepared = prepare_insert_stored(
            DistanceMetric::L2,
            VectorDtype::F32,
            Quantization::RaBitQ,
            raw.clone(),
        );
        assert!(
            Arc::ptr_eq(&prepared.main, &prepared.graph.source_arc()),
            "rabitq source must share the main-store Arc"
        );
        let GraphVector::RabitQ { codes, source } = &prepared.graph else {
            panic!("rabitq quantization must produce GraphVector::RabitQ");
        };
        assert_eq!(
            source.to_f32_vec(),
            raw,
            "source keeps raw main-store values"
        );
        assert_eq!(
            codes.codes.len(),
            crate::rabitq::words_for_dims(dims),
            "sign codes must pack ceil(D/64) words"
        );
        assert!(codes.alpha > 0.0, "non-sentinel alpha must be positive");
        assert!(
            (codes.norm - raw_norm).abs() <= 1e-5,
            "codes norm must be the raw norm: {} vs {raw_norm}",
            codes.norm
        );

        let gq = prepared.graph_query.as_ref().expect("rabitq graph query");
        assert_eq!(
            &gq.data[..],
            raw.as_slice(),
            "L2 build query must stay the raw f32 vector (exact construction)"
        );
    }

    /// RaBitQ 余弦：编码发生在归一化之后——source 保留原始分量，构图查询是
    /// 归一化后的 f32 原向量（分量未旋转）。
    /// EN: RaBitQ cosine encodes after normalization — `source` keeps raw
    /// components while the build query is the normalized f32 vector (its
    /// components are not rotated).
    #[test]
    fn prepare_insert_stored_rabitq_cosine_normalizes_before_encoding() {
        let raw = vec![3.0f32, 4.0, 0.0, 0.0];
        let prepared = prepare_insert_stored(
            DistanceMetric::Cosine,
            VectorDtype::F32,
            Quantization::RaBitQ,
            raw.clone(),
        );
        let GraphVector::RabitQ { codes, source } = &prepared.graph else {
            panic!("rabitq quantization must produce GraphVector::RabitQ");
        };
        assert_eq!(source.to_f32_vec(), raw, "source keeps raw components");
        let gq = prepared.graph_query.as_ref().expect("rabitq graph query");
        assert_eq!(gq.data.len(), raw.len(), "build query keeps raw dims");
        let qnorm: f32 = gq.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (qnorm - 1.0).abs() <= 1e-5,
            "cosine build query must be normalized: {qnorm}"
        );
        assert!(
            (codes.norm - 1.0).abs() <= 1e-5,
            "cosine codes norm must be 1: {}",
            codes.norm
        );
        assert!(codes.alpha > 0.0, "non-sentinel alpha must be positive");
    }

    /// RaBitQ 小图检索对暴力：一热基向量的旋转符号码互为正交/自身全同，
    /// 1 bit 估计在此数据上是**精确**的，因此图检索 top-1 必须与暴力一致。
    /// 随机稠密数据的图内 top-k 受 1 bit 方差影响本就有噪声，严格召回门在
    /// `tests/recall_gate.rs` 走「扩宽候选 + 主表精确重排」的生产口径。
    /// EN: RaBitQ small-graph search vs brute force: rotated one-hot basis
    /// vectors have orthogonal (or identical, for self) sign codes, so 1-bit
    /// estimates are exact on this dataset and graph top-1 must match brute
    /// force. In-graph top-k on dense random data is inherently noisy at 1
    /// bit; the strict recall gate lives in `tests/recall_gate.rs` and uses
    /// the production path (widened candidates + exact main-store rerank).
    #[test]
    fn rabitq_graph_small_index_matches_bruteforce() {
        let dims = 16_usize;
        for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
            let mut g = HnswGraph::new_with_dtype_and_quantization(
                dims,
                metric,
                VectorDtype::F32,
                Quantization::RaBitQ,
                8,
                64,
                64,
                0xC0FFEE,
                10,
            );
            let mut stored: Vec<Vec<f32>> = Vec::with_capacity(dims);
            for i in 0..dims {
                let mut v = vec![0.0f32; dims];
                v[i] = 1.0;
                stored.push(v.clone());
                g.insert(i as u64, v);
            }
            for (i, query) in stored.iter().enumerate() {
                let hits = g.search(query, 3);
                let ids: Vec<VectorId> = hits.iter().map(|(id, _)| *id).collect();
                assert_eq!(
                    ids.first().copied(),
                    Some(i as u64),
                    "metric={metric:?} query e_{i}: self must win, got {ids:?}"
                );
                // 自查询估计精确为 1 → 距离 0（L2）/ 0（1-est）；并列者按 id 排。
                // EN: self estimate is exactly 1 → distance 0 for both metrics.
                assert!(
                    hits[0].1.abs() <= 1e-6,
                    "metric={metric:?} self distance must be 0, got {}",
                    hits[0].1
                );
            }
        }
    }

    /// RaBitQ 零范数：零向量编码为哨兵（全 0 码、alpha=1、norm=0），零查询与
    /// 零存储距离为 0（L2）/ 1（余弦），全链路不得 panic 或产出 NaN。
    /// EN: RaBitQ zero norm: the zero vector encodes as the sentinel (zero codes,
    /// alpha=1, norm=0); a zero query against zero storage yields distance 0
    /// under L2 and 1 under cosine, with no panic or NaN anywhere.
    #[test]
    fn rabitq_zero_norm_vectors_are_safe() {
        let dims = 8usize;
        for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
            let mut g = HnswGraph::new_with_dtype_and_quantization(
                dims,
                metric,
                VectorDtype::F32,
                Quantization::RaBitQ,
                4,
                64,
                64,
                42,
                8,
            );
            g.insert(1, vec![0.0f32; dims]);
            let gv = g.graph_vector(1).expect("inserted node");
            let GraphVector::RabitQ { codes, .. } = gv else {
                panic!("rabitq graph must store GraphVector::RabitQ");
            };
            assert!(
                codes.view(dims).is_zero_norm(),
                "zero vector must be sentinel"
            );
            assert_eq!(codes.alpha, 1.0);
            assert_eq!(codes.codes, vec![0u64; crate::rabitq::words_for_dims(dims)]);

            let hits = g.search(&vec![0.0f32; dims], 1);
            assert_eq!(
                hits.first().map(|(id, _)| *id),
                Some(1),
                "single node is returned"
            );
            let dist = hits.first().expect("one hit").1;
            assert!(
                dist.is_finite(),
                "metric={metric:?} distance must be finite"
            );
            let expected = if metric == DistanceMetric::L2 {
                0.0
            } else {
                1.0
            };
            assert!(
                (dist - expected).abs() <= 1e-6,
                "metric={metric:?} zero-zero distance: {dist} vs {expected}"
            );
        }
    }

    /// RaBitQ 自查询（L2/余弦）：单位向量自比较估计精确为 1，自身必须进入
    /// top-3（并列估计在 1 bit 下可能发生，故不钉死 top-1）。
    /// EN: RaBitQ self-query (L2/cosine): the unit-vector self estimate is exactly
    /// 1, so the self id must land in top-3 (exact ties are possible under 1-bit
    /// estimates, hence not pinned to top-1).
    #[test]
    fn rabitq_graph_self_query_lands_in_top_three_for_l2_and_cosine() {
        let dims = 8usize;
        for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
            let mut g = HnswGraph::new_with_dtype_and_quantization(
                dims,
                metric,
                VectorDtype::F32,
                Quantization::RaBitQ,
                8,
                64,
                64,
                42,
                8,
            );
            let mut stored = Vec::new();
            for i in 0u64..16 {
                let mut v = vec![0.0f32; dims];
                let b = (i % dims as u64) as usize;
                v[b] = 1.0;
                v[(b + 1) % dims] = i as f32 / 15.0;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                stored.push(v.clone());
                g.insert(i, v);
            }
            for (i, q) in stored.iter().enumerate() {
                let hit = g.search(q, 3);
                assert!(
                    hit.iter().any(|(id, _)| *id == i as u64),
                    "metric={metric:?} self {i} must stay in top-3, got {hit:?}"
                );
            }
        }
    }
}
