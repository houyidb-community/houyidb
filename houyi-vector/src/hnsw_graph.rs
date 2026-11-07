//! Full HNSW graph (Malkov–Yashunin style): layered greedy search, heuristic neighbor selection.
//! 中文：标准分层可导航小世界图；与 `HotBufferIndex` 的向量存储解耦，本模块持有构图副本以便检索。
//! EN: Standalone graph + vector map; duplicates vectors from main index for O(1) distance during search.

use crate::distance::{self, cosine_distance_prenormalized};
use crate::{l2_distance, DistanceMetric, VectorId};
use ahash::{AHashMap, AHashSet};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

pub(crate) const MAX_HNSW_LAYERS: usize = 16;

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
    pub(crate) vector: Arc<Vec<f32>>,
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
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) ef_search: usize,
    pub(crate) num_layers: usize,
    pub(crate) seed: u64,
    pub(crate) nodes: AHashMap<VectorId, HnswNode>,
    pub(crate) link_arena: Vec<VectorId>,
    pub(crate) link_free: AHashMap<usize, Vec<usize>>,
    pub(crate) vectors: AHashMap<VectorId, Arc<Vec<f32>>>,
    pub(crate) entry_point: Option<VectorId>,
    pub(crate) max_layer: usize,
}

/// In-memory HNSW index: owns a copy of inserted vectors for distance evaluation.
/// 邻接边存储在 `link_arena` 的连续子块中（真 Flat-Link）；`link_free` 复用等尺寸已释放块。
/// EN: Adjacency in graph-wide `link_arena` sub-blocks; `link_free` reuses equal-size released blocks.
pub struct HnswGraph {
    pub(crate) dims: usize,
    pub(crate) metric: DistanceMetric,
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
    pub(crate) vectors: AHashMap<VectorId, Arc<Vec<f32>>>,
    pub(crate) entry_point: Option<VectorId>,
    pub(crate) max_layer: usize,
}

#[inline]
fn metric_pair(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_distance(a, b),
        // Cosine 路径约定输入已单位化；委托 `distance::cosine_distance_prenormalized`（分块点积）。
        // EN: Normalized inputs; delegate to `distance` for chunked dot (hot path).
        DistanceMetric::Cosine => cosine_distance_prenormalized(a, b),
        // IP 取负点积、L1 取曼哈顿原值：图排序只依赖全序，不依赖具体量纲。
        // EN: IP negates the dot and L1 uses the raw Manhattan sum; the graph
        // only relies on the ordering, not on the scale.
        DistanceMetric::InnerProduct => distance::inner_product_distance(a, b),
        DistanceMetric::L1 => distance::l1_distance(a, b),
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

#[inline]
fn normalize_for_cosine(metric: DistanceMetric, v: &[f32]) -> Arc<Vec<f32>> {
    if !matches!(metric, DistanceMetric::Cosine) {
        return Arc::new(v.to_vec());
    }
    let norm2: f32 = v.iter().map(|x| x * x).sum();
    if norm2 <= ZERO_NORM_BAND_NORM2 {
        return Arc::new(v.to_vec());
    }
    let inv = norm2.sqrt().recip();
    let normalized: Vec<f32> = v.iter().map(|x| x * inv).collect();
    Arc::new(normalized)
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
    vectors: &'a AHashMap<VectorId, Arc<Vec<f32>>>,
    nodes: &AHashMap<VectorId, HnswNode>,
    ep: VectorId,
    ef: usize,
    layer: usize,
) -> Option<&'a Arc<Vec<f32>>> {
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
    vectors: &AHashMap<VectorId, Arc<Vec<f32>>>,
    link_arena: &[VectorId],
    nodes: &AHashMap<VectorId, HnswNode>,
    m: usize,
    query: &[f32],
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
    let dist_ep = metric_pair(metric, query, ep_vec);
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
                let dist_e = metric_pair(metric, query, e_vec);
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
    vectors: &AHashMap<VectorId, Arc<Vec<f32>>>,
    link_arena: &[VectorId],
    nodes: &AHashMap<VectorId, HnswNode>,
    m: usize,
    query: &[f32],
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
    let dist_ep = metric_pair(metric, query, ep_vec);
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
                let dist_e = metric_pair(metric, query, e_vec);
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
    vectors: &AHashMap<VectorId, Arc<Vec<f32>>>,
    w: &[(VectorId, f32)],
    m: usize,
    metric: DistanceMetric,
) -> Vec<VectorId> {
    if m == 0 || w.is_empty() {
        return Vec::new();
    }
    let mut selected: Vec<VectorId> = Vec::with_capacity(m);
    let mut sel_vecs: Vec<&[f32]> = Vec::with_capacity(m);
    for &(eid, d_qe) in w.iter() {
        if selected.len() >= m {
            break;
        }
        let ev = match vectors.get(&eid) {
            Some(v) => v.as_slice(),
            None => continue,
        };
        let mut ok = true;
        for rv in sel_vecs.iter().copied() {
            if metric_pair(metric, ev, rv) < d_qe {
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
    vectors: &AHashMap<VectorId, Arc<Vec<f32>>>,
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
            Some(v) => v.as_slice(),
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
                    .map(|on| (id, metric_pair(metric, host, on.as_slice())))
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
    vectors: &AHashMap<VectorId, Arc<Vec<f32>>>,
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
        let num_layers = max_layers.clamp(1, MAX_HNSW_LAYERS);
        Self {
            dims,
            metric,
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
        }
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
            for l in 0..n.level_count() {
                c += n.layer_len[l] as usize;
            }
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
        if self.dims > 0 && vector.len() != self.dims {
            return 0;
        }
        if self.nodes.contains_key(&id) {
            return 0;
        }

        let metric = self.metric;
        let vector = normalize_for_cosine(metric, &vector);
        let q = vector.as_slice();
        let level = random_level(self.m, self.num_layers, &mut self.rng);

        if self.entry_point.is_none() {
            self.vectors.insert(id, vector);
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

        self.vectors.insert(id, vector);
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
    /// EN: Fast planning path when caller already validated dims/id, avoiding Option/duplicate checks.
    pub(crate) fn plan_insert_with_level_unchecked(
        &self,
        id: VectorId,
        vector: Arc<Vec<f32>>,
        level: usize,
        entry_hint: Option<VectorId>,
    ) -> InsertPlan {
        let metric = self.metric;
        let vector = normalize_for_cosine(metric, &vector);
        if self.entry_point.is_none() {
            return InsertPlan {
                id,
                vector,
                level,
                layers_neighbors: Vec::new(),
            };
        }
        let ep = entry_hint
            .filter(|eid| self.nodes.contains_key(eid))
            .or(self.entry_point)
            .expect("entry_point must exist");
        let global_max = self.max_layer;
        let upper = global_max.min(level);
        let q = vector.as_slice();
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
            vector,
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
            self.vectors.insert(plan.id, Arc::clone(&plan.vector));
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
        true
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
        let query_vec = normalize_for_cosine(metric, query);
        let query = query_vec.as_slice();
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
                1,
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
        let query_vec = normalize_for_cosine(metric, query);
        let query = query_vec.as_slice();
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
                1,
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
    /// EN: Export all graph vectors.
    // 预留 API：供快照/迁移路径使用（HiiL0 已有对应导出），当前无调用方。
    // EN: Reserved API for snapshot/migration paths (HiiL0 has an equivalent); currently unused.
    #[allow(dead_code)]
    pub fn export_vectors(&self) -> Vec<(VectorId, Arc<Vec<f32>>)> {
        self.vectors
            .iter()
            .map(|(id, v)| (*id, Arc::clone(v)))
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
                    Some(v) => v.as_slice(),
                    None => continue,
                };
                let neighbor_ids: Vec<VectorId> = self.neighbors_at_layer(nid, layer);
                let mut scored: Vec<(VectorId, f32)> = neighbor_ids
                    .iter()
                    .filter_map(|&nbr| {
                        self.vectors
                            .get(&nbr)
                            .map(|v| (nbr, metric_pair(metric, host, v.as_slice())))
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

            for layer in 0..=max_level {
                let old_start = old_base + layer_start_offset(m, layer);
                let used = node.layer_len[layer] as usize;
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
                .map(|(id, v)| (*id, l2_distance(&q, v)))
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
}
