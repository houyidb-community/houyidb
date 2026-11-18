//! CollectionManager: 多 Collection 路由，每 Collection 独立维度/索引。
//! Multi-collection router: each collection has its own dims, metric, and IndexBackend.

use crate::index_backend::IndexBackend;
use houyi_core::{HouyiPage, ScalarPredicate, ScalarValue};
use houyi_vector::sparse::{Bm25Index, SparseMode, SparseStats};
use houyi_vector::{BuilderWorker, DistanceMetric, FlushPolicy, Quantization, VectorDtype};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub const DEFAULT_COLLECTION: &str = "default";

/// 显式建集合时请求的 HNSW 图参数；缺省由库层按维度推导。
/// 对外只读暴露（`Collection::hnsw_params` 供二进制测试/观测读取生效值）。
/// EN: HNSW graph parameters requested at explicit collection creation; defaults
/// are derived by the library from the dimension otherwise. Public read-only so
/// binary tests/observability can read the effective values via
/// `Collection::hnsw_params`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl HnswParams {
    /// 合法性边界：全部为 0 会导致构图/检索立即失效；上限防止单集合吞掉内存。
    /// EN: Validity bounds — zero disables construction/search outright; the upper
    /// bound keeps a single collection from exhausting memory.
    pub(crate) fn validate(&self) -> Result<(), String> {
        // 下限 2 与库层 `set_hnsw_m` 一致：m=1 会在后端构造时被拒绝，
        // 在这里提前给出同一边界与措辞，避免「校验通过、构造失败」的错位（OCR #39）。
        // EN: Lower bound 2 matches the library's `set_hnsw_m`: m=1 is rejected at
        // backend construction, so validate with the same bound and message here
        // instead of passing validation only to fail later (OCR #39).
        if self.m < 2 || self.m > 1024 {
            return Err(format!("hnsw m must be in 2..=1024 (got {})", self.m));
        }
        for (name, v) in [
            ("ef_construction", self.ef_construction),
            ("ef_search", self.ef_search),
        ] {
            if v == 0 || v > 65_536 {
                return Err(format!("hnsw {name} must be in 1..=65536 (got {v})"));
            }
        }
        Ok(())
    }
}

/// 每 Collection 标量页的建议容量（字节）。
/// Advisory capacity (bytes) for each collection's scalar PAX page.
const COLLECTION_SCALAR_PAGE_CAPACITY: usize = 64 * 1024 * 1024;

/// 物理标量行（id + 列值）；快照/压实共用，避免重复的复杂类型。
/// EN: Physical scalar rows (id + column values), shared by snapshots and
/// compaction to avoid repeating the complex type.
pub(crate) type ScalarRows = Vec<(u64, HashMap<String, ScalarValue>)>;

/// 标量页统计快照（status/压实判定用）。
/// Scalar-page statistics snapshot (status and compaction decisions).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ScalarStats {
    /// 物理行数（含 stale 行与墓碑行）。 / EN: Physical rows (stale and tombstoned included).
    pub physical_rows: usize,
    /// 非墓碑 id 的去重数（可见行数）。 / EN: Distinct non-tombstoned ids (visible rows).
    pub live_rows: usize,
    /// 物理行中不贡献可见 id 的行数。 / EN: Physical rows that do not contribute a visible id.
    pub stale_rows: usize,
    /// 墓碑 id 数。 / EN: Number of tombstoned ids.
    pub tombstone_ids: usize,
}

/// 一次标量压实的结果（管理端点/周期任务共用）。
/// Result of one scalar compaction (shared by the admin endpoint and periodic task).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScalarCompactionResult {
    pub physical_before: usize,
    pub physical_after: usize,
    pub dropped_rows: usize,
    /// None = 已执行；Some(reason) = 跳过（empty/below_threshold/concurrent_write）。
    /// EN: None = performed; Some(reason) = skipped (empty/below_threshold/concurrent_write).
    pub skipped_reason: Option<String>,
}

/// 每 Collection 的内存标量存储：PAX 页 + 行号→向量 id 映射。
/// Per-collection in-memory scalar store: PAX page + row_idx→vector id map.
///
/// **持久化**：label/payload 以原始 JSON 随 `collection_upsert_batch` 落 collections
/// WAL（WAL-first），冷启动重放经 `json_value_to_scalar` 逐行重建本页——写入侧
/// 与重放侧走同一派生函数，语义逐字节等价。删除以 `collection_delete` 记录落盘，
/// 重放时重建墓碑。
/// EN: **Durability** — label/payload ride the `collection_upsert_batch` records as
/// original JSON (WAL-first); cold replay rebuilds this page row-by-row through the
/// same `json_value_to_scalar` derivation used on the write path. Deletes persist
/// as `collection_delete` records and replay as tombstones.
///
/// 追加写（append-only）+ 内存压实（P5.0.1）：重复 upsert 同一 id 先追加新行；
/// 默认每 10 分钟按 stale 比例压实，只保留每个 id 的最后一行并丢弃墓碑行。
/// 过滤与 payload 回读一律只看最后一行（旧行是已被覆盖的旧值，从写入起即失效）；
/// 压实只是回收物理行，不改变查询语义。压实不写 WAL，重启后由 WAL/快照恢复，
/// 随后周期任务再次压实。
/// EN: Append-only with in-memory compaction (P5.0.1): a re-upsert appends a new
/// row; the default 10-minute task compacts by stale ratio, keeping only the last
/// row per id and dropping tombstoned rows. Filtering and payload read-back only
/// see the last row — stale rows are superseded values and stay invisible from the
/// write on; compaction merely reclaims physical rows and never changes query
/// semantics. It writes no WAL, so restart restores from WAL/snapshots and the
/// periodic task compacts again.
pub struct CollectionScalars {
    /// 字段私有：页行与 `row_to_id` 的 1:1 对齐是结构性不变量，
    /// 只能经由 [`append_row`](Self::append_row) 原子推进，禁止外部只改其一。
    /// EN: Private by design — page rows and `row_to_id` must advance together;
    /// callers go through `append_row` so the alignment cannot be broken by accident.
    page: HouyiPage,
    /// 第 i 项 = PAX 页第 i 行对应的向量 id；与页内行一一对应。
    /// Entry i = vector id backing page row i; aligned 1:1 with page rows.
    row_to_id: Vec<u64>,
    /// 每个 id 的最后一行行号（last-wins 过滤索引）：由 [`append_row`](Self::append_row)
    /// 与 `row_to_id` 同步推进，`from_rows` 重建；被墓碑删除的 id 条目会保留到同 id
    /// 重写或重建为止，由 `matching_ids` 的墓碑检查屏蔽。过滤只认该行，追加写留下的
    /// 旧行不参与；命中行经本表判定，无需再按页去重。
    /// EN: Last row index per id (the last-wins filter index), advanced together with
    /// `row_to_id` by [`append_row`](Self::append_row) and rebuilt by `from_rows`.
    /// Entries of tombstoned ids linger until a same-id re-upsert or a rebuild and are
    /// masked by the tombstone check in `matching_ids`. Filters only accept that row,
    /// so stale rows from append-only re-upserts never participate; matched rows are
    /// resolved through this index instead of per-page dedupe.
    last_row: HashMap<u64, u64>,
    /// 已删除向量 id 的墓碑集：append-only 页无法物理删行，删除以墓碑表达——
    /// 过滤命中映射与 payload 回读都跳过墓碑 id；同 id 重新 upsert 时复活（清除）。
    /// EN: Tombstones of deleted vector ids. The append-only page cannot physically
    /// drop rows, so deletes are logical: id mapping and payload read-back both skip
    /// tombstoned ids; a re-upsert of the same id resurrects it (clears the mark).
    tombstones: HashSet<u64>,
    /// 写代数：每次 append/tombstone/压实 swap 递增。压实用「读代数 → 锁外重建 →
    /// 代数未变才换」避免长时间持锁；并发写会让代数变化，压实放弃本轮。
    /// EN: Write generation, bumped by append/tombstone/compaction swap. Compaction
    /// captures it, rebuilds outside the lock, and swaps only if unchanged; a
    /// concurrent write aborts this round instead of losing data.
    generation: u64,
}

impl CollectionScalars {
    fn new() -> Self {
        // 注意：capacity 仅为建议值——`HouyiPage::insert_row` 不做容量检查，
        // 标量列由普通 Vec 支撑，此处参数只影响对外暴露的 capacity()。
        // EN: capacity is advisory only — `insert_row` never checks it (scalar columns
        // are plain Vecs); the value only surfaces via `HouyiPage::capacity()`.
        Self {
            page: HouyiPage::new(COLLECTION_SCALAR_PAGE_CAPACITY),
            row_to_id: Vec::new(),
            last_row: HashMap::new(),
            tombstones: HashSet::new(),
            generation: 0,
        }
    }

    /// 原子追加一行标量并登记其向量 id（二者必须同时推进以保持 1:1 对齐）。
    /// Append one scalar row and register its backing vector id atomically.
    ///
    /// 空 map 不建行：裸向量（无 label/payload）不进标量页，否则会出现「有行无列」
    /// 的占位行并把后续 `row_to_id` 映射整体错位。调用方需先判断是否有标量可写。
    /// EN: An empty map appends nothing — bare vectors must not create placeholder
    /// rows that would shift every subsequent row→id mapping. Callers skip bare vectors.
    pub(crate) fn append_row(&mut self, id: u64, scalars: HashMap<String, ScalarValue>) {
        if scalars.is_empty() {
            return;
        }
        // 复活语义：携带新标量行的写入即撤销该 id 的墓碑（与向量侧 remove/insert
        // 的墓碑约定同口径；裸向量 upsert 不触碰标量墓碑，与「裸向量不建行」一致）。
        // EN: Resurrection — a write carrying fresh scalar rows clears the tombstone
        // (same contract as the vector-side tombstones; bare-vector upserts don't
        // touch scalar tombstones, consistent with "bare vectors add no row").
        self.tombstones.remove(&id);
        let row_idx = self.page.insert_row(scalars, None) as u64;
        self.row_to_id.push(id);
        self.last_row.insert(id, row_idx);
        self.generation = self.generation.wrapping_add(1);
    }

    /// 登记删除墓碑：该 id 之后的过滤命中映射与 payload 回读一律跳过。
    /// EN: Record a delete tombstone — id mapping and payload read-back skip it.
    pub(crate) fn tombstone(&mut self, id: u64) {
        self.tombstones.insert(id);
        self.generation = self.generation.wrapping_add(1);
    }

    /// 当前写代数（压实 swap 的乐观锁）。
    /// EN: Current write generation (the optimistic lock for compaction swaps).
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// 廉价计数（物理行、墓碑数），供 /metrics 每次 scrape 使用，不做 O(n) 去重。
    /// EN: Cheap counts (physical rows, tombstones) for every /metrics scrape —
    /// no O(n) dedupe scan.
    pub(crate) fn raw_counts(&self) -> (usize, usize) {
        (self.row_to_id.len(), self.tombstones.len())
    }

    /// 统计物理/可见/stale 行与墓碑数（status 端点按需调用，O(physical_rows)）。
    /// EN: Physical/visible/stale rows and tombstones (called on demand by status,
    /// O(physical_rows)).
    pub(crate) fn stats_snapshot(&self) -> ScalarStats {
        let physical_rows = self.row_to_id.len();
        let mut live_ids = HashSet::with_capacity(physical_rows);
        for id in &self.row_to_id {
            if !self.tombstones.contains(id) {
                live_ids.insert(*id);
            }
        }
        let live_rows = live_ids.len();
        ScalarStats {
            physical_rows,
            live_rows,
            stale_rows: physical_rows.saturating_sub(live_rows),
            tombstone_ids: self.tombstones.len(),
        }
    }

    /// 每个可见 id 的最后一行（按最后一次出现的顺序），供压实重建。
    /// Last row per visible id, ordered by last occurrence, for compaction rebuilds.
    ///
    /// 只读扫描，不改动本页；调用方持有标量锁期间完成克隆，锁外重建新页。
    /// EN: Read-only scan; callers clone under the scalar lock and rebuild outside it.
    pub(crate) fn live_rows_last_wins(&self) -> ScalarRows {
        let mut out: ScalarRows = Vec::with_capacity(self.row_to_id.len());
        let mut seen = HashSet::with_capacity(self.row_to_id.len());
        for row_idx in (0..self.row_to_id.len()).rev() {
            let id = self.row_to_id[row_idx];
            if self.tombstones.contains(&id) || !seen.insert(id) {
                continue;
            }
            if let Some(view) = self.page.get_row(row_idx) {
                out.push((id, view.scalars));
            }
        }
        // 反向扫描得到「最后出现」的逆序；翻转后保持与原页一致的先后关系。
        // EN: The reverse scan yields last occurrences backwards; flip to preserve
        // the page's relative order.
        out.reverse();
        out
    }

    /// 完整物理页视图 + 墓碑集合（快照用，保持 append-only 语义）。
    /// Full physical-page view plus tombstones (snapshot use — preserves append-only
    /// semantics).
    ///
    /// 与 [`live_rows_last_wins`](Self::live_rows_last_wins) 的区别：快照必须逐物理行
    /// 保存（包括 stale 行），使「快照 + 尾段重放」复原的物理/存活/stale 行数与快照
    /// 时点一致（压实节奏与诊断指标不变）；查询语义由 last-wins 保证，与物理行数无关。
    /// EN: Unlike `live_rows_last_wins`, snapshots must keep every physical row
    /// (stale rows included), so snapshot+tail replay reproduces the same
    /// physical/live/stale row counts as of the snapshot (compaction cadence and
    /// diagnostics stay unchanged). Query semantics are guaranteed by last-wins and
    /// do not depend on physical row counts.
    // P5.0.2 快照写入接线后使用；先保留单一实现。
    // EN: Used once P5.0.2 wires snapshot writes; keep the single implementation now.
    #[allow(dead_code)]
    pub(crate) fn snapshot_view(&self) -> (ScalarRows, Vec<u64>) {
        let rows = (0..self.row_to_id.len())
            .filter_map(|row_idx| {
                self.page
                    .get_row(row_idx)
                    .map(|view| (self.row_to_id[row_idx], view.scalars))
            })
            .collect();
        let tombstones = self.tombstones.iter().copied().collect();
        (rows, tombstones)
    }

    /// 从物理行 + 墓碑恢复（快照加载；generation 从 0 起）。
    /// EN: Restore from physical rows plus tombstones (snapshot load; generation
    /// starts at zero).
    pub(crate) fn from_rows(rows: ScalarRows, tombstones: Vec<u64>) -> Self {
        let mut out = Self::new();
        for (id, scalars) in rows {
            let row_idx = out.page.insert_row(scalars, None) as u64;
            out.row_to_id.push(id);
            out.last_row.insert(id, row_idx);
        }
        out.tombstones = tombstones.into_iter().collect();
        out
    }

    /// 返回列上谓词命中的向量 id（只看每个 id 的最后一行）。
    /// Vector ids matched by `pred` on `column` (last row per id only).
    ///
    /// append-only 重写会让多行指向同一 id，只有最后一行代表该 id 的当前值；旧行
    /// 不得再参与过滤（公共 issue #6：重写 payload 后旧值仍命中，与 `payloads_for`
    /// 的 last-wins 回读口径矛盾）。压实前后结果一致——压实只是回收物理行。
    /// EN: Append-only re-upserts map several rows to one id; only the last row is
    /// current, so stale rows must not participate (public issue #6: after
    /// re-upserting a payload the old value still matched, contradicting the
    /// last-wins read-back in `payloads_for`). The result is identical before and
    /// after compaction — compaction only reclaims physical rows.
    pub(crate) fn matching_ids(&self, column: &str, pred: &ScalarPredicate) -> Vec<u64> {
        let matched_rows = self.page.filter_to_bitset(column, pred);
        let mut ids = Vec::with_capacity(matched_rows.len());
        for row_idx in matched_rows {
            let Some(&vector_id) = self.row_to_id.get(row_idx as usize) else {
                continue;
            };
            if self.tombstones.contains(&vector_id) {
                continue;
            }
            // 命中行不是该 id 的最后一行 ⇒ 被重写覆盖的旧值，丢弃（去重随之自然成立）。
            // EN: A match on a non-last row is a superseded value — drop it (dedupe
            // follows automatically).
            if self.last_row.get(&vector_id) != Some(&row_idx) {
                continue;
            }
            ids.push(vector_id);
        }
        ids
    }

    /// 按向量 id 回读 payload（label + payload 键值），供检索响应携带文档内容。
    /// Read payloads back by vector id (label + payload keys) for search responses.
    ///
    /// append-only 语义：同一 id 多行时**最后一行生效**，与过滤侧
    /// [`matching_ids`](Self::matching_ids) 的 last-wins 口径一致（公共 issue #6）；
    /// 墓碑 id 一律缺席。行扫描为 O(页行数)，仅在 `include_payload=true` 的检索尾部
    /// 调用（k 个 id 的查找合并成一次扫页），不在单点查询热路径。
    /// EN: Append-only semantics — the LAST row per id wins, matching the last-wins
    /// rule of [`matching_ids`](Self::matching_ids) (public issue #6); tombstoned ids
    /// are always absent. The row scan is O(page rows), invoked only at the tail of
    /// `include_payload=true` searches (one merged scan for k ids), never on a
    /// point-lookup hot path.
    pub(crate) fn payloads_for(&self, ids: &[u64]) -> HashMap<u64, HashMap<String, ScalarValue>> {
        let want: HashSet<u64> = ids.iter().copied().collect();
        // 反向扫页：每个目标 id 只取最后一行。
        // EN: Reverse page scan — keep only the last row per wanted id.
        let mut out: HashMap<u64, HashMap<String, ScalarValue>> = HashMap::new();
        for row_idx in (0..self.row_to_id.len()).rev() {
            let id = self.row_to_id[row_idx];
            if !want.contains(&id) || self.tombstones.contains(&id) || out.contains_key(&id) {
                continue;
            }
            if let Some(view) = self.page.get_row(row_idx) {
                // 剔除 PAX 页缺失列的占位值（`HouyiPage::insert_row` 以
                // JsonText("null") 补齐列长）：占位不是用户数据，回读不得透出。
                // 已知取舍：用户真实写入的 JSON null 与该占位无法区分，一并隐藏
                // （页级表示限制，与 `json_value_to_scalar` 的 null→JsonText 约定相关）。
                // EN: Drop the page's missing-column padding (JsonText("null")) — it is
                // a layout artifact, not user data. Trade-off: a genuinely stored JSON
                // null is indistinguishable from padding and is hidden too.
                let scalars = view
                    .scalars
                    .into_iter()
                    .filter(|(_, v)| !matches!(v, ScalarValue::JsonText(s) if s == "null"))
                    .collect();
                out.insert(id, scalars);
            }
        }
        out
    }

    /// 已落标的量行数（= `row_to_id` 长度；裸向量不计）。
    /// Number of scalar rows (bare vectors excluded).
    pub(crate) fn row_count(&self) -> usize {
        self.row_to_id.len()
    }
}

/// Collection 内 HNSW 后端的异步构图参数（与 benchmark 主索引 worker 同源）。
/// Async-builder parameters for per-collection HNSW backends (same knobs as the
/// benchmark main index worker).
///
/// 为什么需要它：`insert_batch` 只进待构图队列，向量要经 worker flush 才可被检索。
/// 不带该配置创建的 Manager（单元测试）不启动 worker，队列保持静态、行为确定；
/// 生产装配（main）必须注入，否则 Collection 的 HNSW 写入永远不可检索（冒烟发现的
/// 生产缺口）。
/// EN: Why this exists — `insert_batch` only enqueues; vectors become searchable
/// solely after a worker flush. Managers built without it (unit tests) start no
/// worker and stay deterministic; production assembly (main) must inject it, or
/// collection HNSW writes are never searchable (a gap found by smoke testing).
#[derive(Debug, Clone, Copy)]
pub struct CollectionBuilderConfig {
    pub threshold: usize,
    pub tick_ms: u64,
    pub policy: FlushPolicy,
}

/// 写路径拿到锁后发现 collection 已被整库删除：拒绝本次写，避免向一个已从
/// manager 摘除的孤儿 Arc 写入并返回成功。
/// EN: A write found the collection dropped after acquiring its write lock;
/// reject it instead of mutating and acknowledging against an orphaned Arc that
/// is already absent from the manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CollectionDeleted;

impl std::fmt::Display for CollectionDeleted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("collection is deleted")
    }
}

impl std::error::Error for CollectionDeleted {}

pub struct CollectionManager {
    collections: Mutex<HashMap<String, Arc<Collection>>>,
    default_dims: usize,
    default_index_type: String,
    builder_cfg: Option<CollectionBuilderConfig>,
    /// 新建 collection 的默认 HA 级别（P4.4.2，TOML `default_ha_level`，
    /// 缺省 async）。优先级：collection 级设置（WAL 记录）> 本默认值。
    /// P4.6.3 起为枚举——非法级别在类型层不可表达。
    /// EN: Default HA level for newly created collections; a collection-level
    /// setting (WAL record) always wins over this default (enum since P4.6.3).
    default_ha_level: houyi_persistence::ha_level::HaLevel,
}

pub struct Collection {
    // 元数据字段：当前路由/状态端点尚未全部透出，先保留供后续 status/可观测性使用。
    // Metadata retained for future status/observability surfaces; not all read today.
    #[allow(dead_code)]
    pub(crate) name: String,
    pub(crate) dims: usize,
    /// 实际计分度量（pq-flat 恒 L2、hii-l0 恒 Cosine，其余为创建时请求值）。
    /// EN: Effective scoring metric (always L2 for pq-flat, Cosine for hii-l0,
    /// otherwise the metric requested at creation).
    pub(crate) metric: DistanceMetric,
    pub(crate) index_type: String,
    /// 存储向量编码（创建后不可变）：f32 默认；f16 仅 hnsw 后端。
    /// EN: Stored vector encoding (immutable after creation): f32 by default;
    /// f16 only for the hnsw backend.
    pub(crate) vector_dtype: VectorDtype,
    /// 图侧量化（创建后不可变）：none 默认；int8 仅 hnsw 后端。主表始终保留
    /// 原始向量，int8 只压缩图内副本，最终 top-k 在原始向量上精确重排。
    /// EN: Graph-side quantization (immutable after creation): none by default;
    /// int8 only for the hnsw backend. The main store always keeps raw vectors;
    /// int8 only shrinks the in-graph copy, and the final top-k is reranked
    /// exactly on raw vectors.
    pub(crate) quantization: Quantization,
    /// 稀疏索引模式（P4）：创建后不可变；唯一例外是冷启动「legacy 图快照恢复
    /// 先行、WAL create 记录后重放」时由 none 对账升级为 bm25（图快照头不含
    /// 稀疏模式，create 记录才是事实）。以 [`SparseMode::code`] 存储，对外一律
    /// 经 [`Self::sparse_mode`]。
    /// EN: Sparse index mode (P4): immutable after creation. The sole exception is
    /// cold-start reconciliation from none to bm25 when a legacy graph-snapshot
    /// restore created the collection first and the replayed `collection_create`
    /// record (the authoritative fact) requires bm25. Stored as the enum code;
    /// only [`Self::sparse_mode`] crosses the type boundary.
    sparse_mode: std::sync::atomic::AtomicU8,
    /// 稀疏倒排索引（仅 bm25 模式使用）。`OnceLock` 使冷启动对账可在 `&self`
    /// 上惰性初始化空索引，随后由 WAL 重放/快照填充；喷模式集合永不触碰。
    /// EN: Sparse inverted index (bm25 mode only). `OnceLock` lets cold-start
    /// reconciliation lazily initialize an empty index through `&self`, which
    /// WAL replay/snapshot then fills; none-mode collections never touch it.
    sparse: std::sync::OnceLock<std::sync::RwLock<Bm25Index>>,
    #[allow(dead_code)]
    pub(crate) created_at: u64,
    /// 索引后端：crate 外的二进制/集成测试会直接驱动（排空异步 builder）。
    /// Index backend: also driven directly by in-package binary tests (builder drain).
    pub backend: IndexBackend,
    /// 该 Collection 的内存标量页（label/payload；随 collections WAL 持久化，详见类型文档）。
    /// Per-collection in-memory scalar page (label/payload; persisted via collections WAL).
    /// 每 Collection 一把锁而非全局锁：不同 Collection 的写入/过滤互不阻塞。
    /// EN: per-collection lock (not a global one) so collections never contend.
    pub(crate) scalars: Mutex<CollectionScalars>,
    /// 写互斥锁：串行化「WAL append + 内存变更」这组复合写（upsert/delete 共用），
    /// 保证 WAL 顺序与内存顺序一致——否则并发下同 id 的 upsert/delete 会让重放
    /// 结果与已确认状态分叉（重放复活已删 id、或过期匹配集销毁已确认写入）。
    /// 用 tokio Mutex 而非 std：guard 需跨 `.await`（WAL append / backend.delete /
    /// insert_batch）持有，std MutexGuard 非 Send 无法用于 axum handler。
    /// 锁序约定：write_lock 先于 scalars 获取，任何路径不得反向。
    /// EN: Write mutex serializing the compound "WAL append + in-memory mutation"
    /// (shared by upsert/delete) so WAL order always matches memory order —
    /// otherwise concurrent same-id upsert/delete makes replay diverge from the
    /// acknowledged state (resurrecting a deleted id, or letting a stale match
    /// set destroy an acknowledged write). tokio Mutex because the guard must be
    /// held across `.await` (WAL append / backend ops). Lock order: write_lock
    /// is always taken before scalars, never the reverse.
    pub(crate) write_lock: tokio::sync::Mutex<()>,
    /// HA 级别（P4.4.2）：collection 粒度的复制持久性语义，热路径一次
    /// load(Relaxed) 零锁读取。内部以 `HaLevel::code()` 存储（std 无
    /// AtomicHaLevel）；对外一律经 [`Self::ha_level`] / [`Self::set_ha_level`]
    /// 的枚举接口——裸码不得越过本类型边界（P4.6.3 类型硬化）。
    /// 运行期经 `POST /v1/collections/{name}/ha_level`（WAL-first）与复制流
    /// apply 改写，重启经 collections.wal 重放恢复。
    /// EN: Stored as the enum's code (std has no AtomicHaLevel); the enum
    /// accessors are the only crossing points. Mutated WAL-first via the
    /// endpoint or the replication stream, restored by replay.
    ha_level: std::sync::atomic::AtomicU8,
    /// 整库删除哨兵：一旦置位，后续写路径（`begin_write`）拒绝写入。删除流程
    /// 在持写锁时置位并同时从 manager 摘除，防止「拿到 Arc 但尚未拿锁」的写
    /// 落在一个已删除的孤儿对象上。
    /// EN: Whole-collection drop sentinel. Once set, `begin_write` rejects later
    /// writers. The delete path sets it while holding the write lock and removes
    /// the Arc from the manager, so a writer that fetched the Arc just before
    /// the drop cannot mutate and acknowledge an orphan.
    pub(crate) deleted: std::sync::atomic::AtomicBool,
    /// 建集合记录是否已落盘（collections WAL `collection_create`）。
    /// 置位前任何持写锁的写路径都必须先补落 create 记录（见
    /// [`ensure_create_logged`]），保证 WAL 中 create 永远先于该 Arc 的任何
    /// upsert/delete/ha_level/drop；删除→同名重建的竞态窗口由此收敛。
    /// EN: Whether this collection's `collection_create` record is already durable.
    /// Until it is set, every write-lock holder must log the create first (see
    /// [`ensure_create_logged`]), so the create always precedes any mutation of
    /// this Arc in the WAL — including across a delete/recreate race.
    create_logged: std::sync::atomic::AtomicBool,
    /// 是否由 collections WAL 的 upsert 重放入口在「无 create 记录」时隐式建立。
    /// 这是稀疏对账（none → bm25）的唯一依据：快照恢复的集合 `create_logged`
    /// 同样为 false，但其稀疏模式由快照/manifest 承载，后续稀疏 upsert 属腐坏
    /// 记录，必须拒绝而不是静默升级（OCR 9fda139）。
    /// EN: Whether this collection was implicitly built by the collections-WAL
    /// upsert replay entry while the create record was missing. This is the sole
    /// basis for sparse reconciliation (none -> bm25): a snapshot-restored
    /// collection also has `create_logged == false`, but its sparse mode comes
    /// from the snapshot/manifest, so a later sparse upsert is corruption and
    /// must be rejected instead of silently upgrading (OCR 9fda139).
    implicit_upsert_built: std::sync::atomic::AtomicBool,
    /// 已成功应用的最后一条 WAL LSN（P5.0）：快照 checkpoint 只能用到它。
    /// WAL 已 append 但后端/标量写失败的记录不推进——留给重放，不能被快照跳过。
    /// EN: Last successfully applied WAL LSN (P5.0). Snapshot checkpoints may only
    /// cover this value; records appended but not applied must stay replayable.
    pub applied_lsn: AtomicU64,
    /// 最近一次成功压实的时间（Unix 秒；0 = 从未）。
    /// EN: Unix seconds of the last successful compaction (0 = never).
    pub(crate) last_compaction_unix: AtomicU64,
    /// 最近一次压实丢弃的行数。 / EN: Rows dropped by the latest compaction.
    pub(crate) last_compaction_dropped: AtomicU64,
    pub(crate) compactions_ok_total: AtomicU64,
    pub(crate) compactions_skipped_total: AtomicU64,
    pub(crate) compaction_failures_total: AtomicU64,
    pub(crate) compaction_dropped_rows_total: AtomicU64,
    /// sync 等待超时告警的限流计数（P4.4.2）：挂在 collection 上，生命周期
    /// 随 collection 回收——无进程级映射表的无界增长问题（OCR 6bfb210[5]）。
    /// EN: Rate-limit counter for sync-wait timeout warns; dies with the
    /// collection, so no process-lifetime map can grow unboundedly.
    pub sync_timeout_warns: std::sync::atomic::AtomicU64,
    /// HNSW 后端的异步构图 worker 句柄：仅持有保活，进程退出随 runtime 回收。
    /// Builder worker keep-alive handle for the HNSW backend; reclaimed with the
    /// runtime at process exit.
    _builder_worker: Option<BuilderWorker>,
}

/// bm25 稀疏索引只允许可过滤后端（集合创建与冷启动对账共用一条判据，
/// 避免两处规则漂移；两处分属 Collection 与 CollectionManager 的 impl，
/// 因此作为模块级自由函数）。
/// EN: bm25 sparse indexes only allow the filterable backends (creation and
/// cold-start reconciliation share one predicate so the rule cannot drift;
/// they live in Collection's and CollectionManager's impls respectively, so the
/// predicate is a module-level free function).
fn sparse_bm25_backend_supported(index_type: &str) -> bool {
    matches!(index_type, "hnsw" | "flat")
}

impl Collection {
    /// 是否已被整库删除（Acquire 读取，与 `mark_deleted` 的 Release 写入配对）。
    /// EN: Whether the collection has been dropped (Acquire pairs with the
    /// Release store in `mark_deleted`).
    pub(crate) fn is_deleted(&self) -> bool {
        self.deleted.load(std::sync::atomic::Ordering::Acquire)
    }

    /// 当前生效的 HNSW 参数（WAL 记录、快照头与状态端点共用）。直接读后端
    /// 的原子值而非创建时的缓存：运行期调参（pgwire `SET`/管理端点）只改后端，
    /// 缓存会永久陈旧（OCR #34）。非 HNSW 后端返回库层占位默认值，调用方按
    /// `index_type == "hnsw"` 过滤后才透出。
    /// EN: HNSW parameters currently in effect (shared by WAL records, snapshot
    /// headers and status endpoints). Read the backend's atomics instead of a
    /// creation-time cache: runtime tuning only mutates the backend, so a cache
    /// would stay stale forever (OCR #34). Non-HNSW backends return placeholder
    /// library defaults; callers expose them only when `index_type == "hnsw"`.
    pub fn hnsw_params(&self) -> HnswParams {
        HnswParams {
            m: self.backend.hnsw_m(),
            ef_construction: self.backend.hnsw_ef_construction_value(),
            ef_search: self.backend.hnsw_ef_search(),
        }
    }

    /// 当前生效度量（创建后不可变；pq-flat 恒 L2、hii-l0 恒 Cosine）。
    /// EN: Effective metric (immutable after creation; always L2 for pq-flat and
    /// Cosine for hii-l0).
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// 存储向量编码（创建后不可变；WAL/快照记录与状态端点共用）。
    /// EN: Stored vector encoding (immutable after creation; shared by WAL/snapshot
    /// records and status endpoints).
    pub fn vector_dtype(&self) -> VectorDtype {
        self.vector_dtype
    }

    /// 图侧量化（创建后不可变；仅 hnsw 后端可为 int8）。
    /// EN: Graph-side quantization (immutable after creation; only the hnsw
    /// backend may use int8).
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// 当前稀疏索引模式（Acquire 读，与对账升级的 Release 写配对）。
    /// 存储码未知属内存腐坏：panic 留痕，绝不静默当作 none。
    /// EN: Current sparse mode (Acquire load, paired with the Release store in
    /// reconciliation). An unknown code is memory corruption: panic with a trace
    /// instead of silently degrading to none.
    pub fn sparse_mode(&self) -> SparseMode {
        let code = self.sparse_mode.load(Ordering::Acquire);
        SparseMode::from_code(code).unwrap_or_else(|| {
            panic!("collection sparse_mode atomic holds unknown code {code} (memory corruption?)")
        })
    }

    /// 稀疏索引句柄：仅 bm25 模式可见；none 模式恒 `None`（调用方据此拒绝
    /// 稀疏写入/查询）。bm25 集合必定返回可用的 `RwLock<Bm25Index>`。
    /// EN: Sparse index handle: only visible in bm25 mode; none-mode collections
    /// always return `None` (callers reject sparse writes/queries on that basis).
    /// A bm25 collection always yields a usable `RwLock<Bm25Index>`.
    pub(crate) fn sparse_index(&self) -> Option<&std::sync::RwLock<Bm25Index>> {
        if self.sparse_mode() != SparseMode::Bm25 {
            return None;
        }
        Some(
            self.sparse
                .get_or_init(|| std::sync::RwLock::new(Bm25Index::new())),
        )
    }

    /// 状态/指标用稀疏统计；none 模式返回 `Ok(None)`（不是 0 统计）。
    /// http status 端点（Phase D）接入前暂由测试使用。
    /// EN: Sparse stats for status/metrics; none mode returns `Ok(None)` (not a
    /// zeroed stat set). Used by tests until the HTTP status endpoint (Phase D)
    /// consumes it.
    #[allow(dead_code)]
    pub(crate) fn sparse_stats(&self) -> Result<Option<SparseStats>, String> {
        let Some(lock) = self.sparse_index() else {
            return Ok(None);
        };
        let index = lock
            .read()
            .map_err(|e| format!("sparse lock poisoned: {e}"))?;
        Ok(Some(index.stats()))
    }

    /// 冷启动稀疏模式对账：现存集合为 none 而记录要求 bm25 时升级（事实来源
    /// 是重放的 create 记录，或 create 记录丢失时的稀疏 upsert）；反向与任何
    /// 其他分叉一律拒绝。升级只发生一次且仅由重放入口调用（serve 之前 /
    /// follower apply 串行），之后模式重新成为不可变事实。空索引由后续
    /// upsert/快照重放填充。
    /// EN: Cold-start reconciliation: upgrade none to bm25 when a record requires
    /// it (the fact comes from a replayed create record, or from a sparse upsert
    /// when that record is missing); the reverse and any other divergence are
    /// rejected. Upgrade happens once, only from the replay entry (before
    /// serving / serialized follower apply); afterwards the mode is immutable
    /// again. Later upsert/snapshot replay fills the empty index.
    pub(crate) fn reconcile_sparse_mode(&self, requested: SparseMode) -> Result<(), String> {
        let current = self.sparse_mode();
        if current == requested {
            return Ok(());
        }
        if current == SparseMode::None && requested == SparseMode::Bm25 {
            // 升级同样受「bm25 仅限 hnsw|flat」创建约束：隐式建出的集合采用
            // 服务端默认后端（可能是 clustered/pq-flat/hii-l0），不校验会让
            // 集合变成永远无法编码快照的形态（manifest 编码会拒绝，重启即失败）。
            // EN: The upgrade is bound by the same "bm25 only on hnsw|flat"
            // creation rule; implicitly built collections use the server default
            // backend (possibly clustered/pq-flat/hii-l0), and skipping the check
            // yields a shape whose snapshot can never encode (the manifest
            // encoder rejects it), failing the next restart.
            if !sparse_bm25_backend_supported(&self.index_type) {
                return Err(format!(
                    "collection '{}' has index_type={:?}, sparse bm25 requires 'hnsw'|'flat'",
                    self.name, self.index_type
                ));
            }
            self.sparse_mode
                .store(SparseMode::Bm25.code(), Ordering::Release);
            // 显式初始化空索引：升级后 [`Self::sparse_index`] 恒可用。
            // EN: Initialize the empty index explicitly so `sparse_index` is
            // always usable after the upgrade.
            let _ = self.sparse_index();
            tracing::warn!(
                collection = %self.name,
                "collection sparse mode reconciled none -> bm25 during replay"
            );
            return Ok(());
        }
        Err(format!(
            "collection '{}' has sparse={}, record requires {}",
            self.name,
            current.as_str(),
            requested.as_str()
        ))
    }

    /// 置位删除哨兵；由删除流程在持有写锁时调用（重放/follower apply 同）。
    /// EN: Set the drop sentinel; called by the delete flow while holding the
    /// write lock (cold replay and follower apply do the same).
    pub(crate) fn mark_deleted(&self) {
        self.deleted
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// 建集合记录是否已落盘（Acquire 与 `mark_create_logged` 的 Release 配对）。
    /// EN: Whether the create record is durable (Acquire pairs with the Release
    /// store in `mark_create_logged`).
    pub(crate) fn create_logged(&self) -> bool {
        self.create_logged
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// 标记建集合记录已落盘（仅在记录 append 成功之后调用）。
    /// EN: Mark the create record durable (only after a successful append).
    pub(crate) fn mark_create_logged(&self) {
        self.create_logged
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// 是否由无 create 记录的 upsert 重放隐式建立（Acquire 读，与
    /// `mark_implicit_upsert_built` 的 Release 写配对）。
    /// EN: Whether the collection was implicitly built by upsert replay without
    /// a create record (Acquire pairs with the Release store in
    /// `mark_implicit_upsert_built`).
    pub(crate) fn implicit_upsert_built(&self) -> bool {
        self.implicit_upsert_built
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// 标记「由无 create 记录的 upsert 重放隐式建立」（仅重放入口在新建时调用）。
    /// 标记的读者（后续 upsert 记录的稀疏对账）与写者串行在同一条 apply 流里
    /// ——启动重放或 follower 实时 apply 二选一；follower 上该集合可与服务线程
    /// 共享同一 `Arc`，因此原子类型 + Release/Acquire 按跨线程可见性设计，
    /// 不依赖「单线程」假设。
    /// EN: Mark "implicitly built by upsert replay without a create record"
    /// (called by the replay entry only, and only when it created the collection).
    /// The marker's reader (sparse reconciliation of later upsert records) and
    /// writer run serially inside one apply stream — either startup replay or a
    /// follower's live apply; on a follower the collection may be shared with
    /// serving threads through the same `Arc`, so the atomic type with
    /// Release/Acquire is designed for cross-thread visibility and does not
    /// assume single-threadedness.
    pub(crate) fn mark_implicit_upsert_built(&self) {
        self.implicit_upsert_built
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// 获取写锁并在拿锁后再次确认未删除。双检关闭「取 Arc 后、拿锁前被删除」
    /// 的窗口；已删除返回 [`CollectionDeleted`]，调用方映射 409/只读语义。
    /// EN: Acquire the write lock and re-check deletion after acquiring it. The
    /// double check closes the "fetched Arc, dropped before locking" window;
    /// a dropped collection yields [`CollectionDeleted`] for the caller to map
    /// to 409/read-only semantics.
    pub(crate) async fn begin_write(
        &self,
    ) -> Result<tokio::sync::MutexGuard<'_, ()>, CollectionDeleted> {
        let guard = self.write_lock.lock().await;
        if self.is_deleted() {
            return Err(CollectionDeleted);
        }
        Ok(guard)
    }

    /// 读取 HA 级别（热路径一次 Relaxed load）。
    /// 存储码损坏在进程内不可达（所有写入点都经 [`Self::set_ha_level`]）；
    /// 真出现即属于内存腐坏，按不可恢复错误panic 并留痕，绝不静默落 async。
    /// EN: one Relaxed load on the hot path; an impossible bad code panics with
    /// a trace instead of silently degrading to async.
    pub fn ha_level(&self) -> houyi_persistence::ha_level::HaLevel {
        use houyi_persistence::ha_level::HaLevel;
        let code = self.ha_level.load(std::sync::atomic::Ordering::Relaxed);
        HaLevel::from_code(code).unwrap_or_else(|| {
            panic!("collection ha_level atomic holds unknown code {code} (memory corruption?)")
        })
    }

    /// 写 HA 级别（WAL-first 调用方在日志落盘成功后调用；复制 apply 同）。
    /// EN: set the level after the WAL record is durable (or via replication apply).
    pub fn set_ha_level(&self, level: houyi_persistence::ha_level::HaLevel) {
        self.ha_level
            .store(level.code(), std::sync::atomic::Ordering::Relaxed);
    }

    /// 读取已应用 WAL LSN（快照 checkpoint 的唯一来源）。
    /// EN: Read the last applied WAL LSN (the sole snapshot checkpoint source).
    pub fn applied_lsn(&self) -> u64 {
        self.applied_lsn.load(Ordering::Relaxed)
    }

    /// 成功应用一条 WAL record 后推进水位线（fetch_max 容忍幂等重放）。
    /// EN: Advance the applied watermark after a successful record (fetch_max
    /// tolerates idempotent replay).
    pub(crate) fn note_applied_lsn(&self, lsn: u64) {
        self.applied_lsn.fetch_max(lsn, Ordering::Relaxed);
    }

    /// 内存标量压实：只保留每个 id 的最后一行、丢弃墓碑行，锁外重建、代数校验后
    /// swap。默认按 stale 比例触发；`force` 忽略阈值（管理端点用）。
    /// EN: In-memory scalar compaction — keep only the last row per id, drop
    /// tombstoned rows, rebuild outside the lock, and swap if the generation is
    /// unchanged. The stale ratio gates the automatic path; `force` ignores it.
    ///
    /// 不写 WAL：压实是内存优化，崩溃/重启后由 WAL 或快照恢复；下一次周期任务会
    /// 在可见 stale 行重新积累后再次压实。
    /// EN: No WAL write — compaction is an in-memory optimization; crash/restart
    /// restores from WAL or snapshots, and the periodic task re-runs it later.
    pub async fn compact_scalars(
        &self,
        force: bool,
        stale_ratio: f64,
    ) -> Result<ScalarCompactionResult, String> {
        let ratio = if stale_ratio.is_finite() {
            stale_ratio.clamp(0.0, 1.0)
        } else {
            1.0
        };
        // 阶段 1：持写锁 + 标量锁取一致性视图；写锁保证没有 in-flight 写，
        // 标量锁保证页与墓碑同时定格。只做克隆，不做重建。
        // EN: Phase 1 — take write_lock + scalar lock for a consistent view. The
        // write lock excludes in-flight writes; the scalar lock freezes page and
        // tombstones. Clone only; rebuild happens outside the locks.
        let (generation, physical_before, rows) = {
            let _write = self.write_lock.lock().await;
            let scalars = self.scalars.lock().map_err(|e| {
                self.compaction_failures_total
                    .fetch_add(1, Ordering::Relaxed);
                format!("scalar lock poisoned: {e}")
            })?;
            let physical = scalars.row_count();
            if physical == 0 {
                self.compactions_skipped_total
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(ScalarCompactionResult {
                    physical_before: 0,
                    physical_after: 0,
                    dropped_rows: 0,
                    skipped_reason: Some("empty".to_string()),
                });
            }
            let rows = scalars.live_rows_last_wins();
            let stale = physical.saturating_sub(rows.len());
            if !force && (stale as f64 / physical as f64) < ratio {
                self.compactions_skipped_total
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(ScalarCompactionResult {
                    physical_before: physical,
                    physical_after: physical,
                    dropped_rows: 0,
                    skipped_reason: Some("below_threshold".to_string()),
                });
            }
            (scalars.generation(), physical, rows)
        };

        // 阶段 2：锁外重建新页（O(n) 内存分配/复制），随后重新加锁；代数变化说明
        // 期间有写，放弃本轮而不是覆盖新数据。
        // EN: Phase 2 — rebuild outside the locks; if the generation changed, a
        // concurrent write landed and this round aborts instead of overwriting it.
        let new_state = CollectionScalars::from_rows(rows, Vec::new());
        let _write = self.write_lock.lock().await;
        let mut scalars = self.scalars.lock().map_err(|e| {
            self.compaction_failures_total
                .fetch_add(1, Ordering::Relaxed);
            format!("scalar lock poisoned: {e}")
        })?;
        if scalars.generation() != generation {
            self.compactions_skipped_total
                .fetch_add(1, Ordering::Relaxed);
            return Ok(ScalarCompactionResult {
                physical_before: scalars.row_count(),
                physical_after: scalars.row_count(),
                dropped_rows: 0,
                skipped_reason: Some("concurrent_write".to_string()),
            });
        }
        let physical_after = new_state.row_count();
        let old = std::mem::replace(&mut *scalars, new_state);
        // from_rows 的 generation 从 0 起；swap 后必须延续旧代数 +1。
        // EN: from_rows starts at generation 0; after the swap continue from old+1.
        scalars.generation = old.generation.wrapping_add(1);
        let dropped_rows = old.row_count().saturating_sub(physical_after);
        drop(scalars);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_compaction_unix.store(now, Ordering::Relaxed);
        self.last_compaction_dropped
            .store(dropped_rows as u64, Ordering::Relaxed);
        self.compactions_ok_total.fetch_add(1, Ordering::Relaxed);
        self.compaction_dropped_rows_total
            .fetch_add(dropped_rows as u64, Ordering::Relaxed);

        Ok(ScalarCompactionResult {
            physical_before,
            physical_after,
            dropped_rows,
            skipped_reason: None,
        })
    }
}

/// 重放时稀疏模式分叉的对账策略（创建后模式不可变，分叉只能来自重放）。
/// EN: Divergence reconciliation policy for the sparse mode during replay
/// (the mode is immutable after creation, so divergence only comes from replay).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SparseReconcile {
    /// 严格比对，任何分叉都拒绝（实时写入 / 快照恢复入口）。
    /// EN: Strict; any divergence is rejected (live write / snapshot restore).
    Never,
    /// 允许 none → bm25（create 记录重放：该记录就是模式事实）。
    /// EN: Allow none -> bm25 (create-record replay: the record is the fact).
    Always,
    /// 仅当集合由无 create 记录的 upsert 重放隐式建立时允许 none → bm25；
    /// create 记录丢失时，后续稀疏 upsert 才携带模式事实。快照恢复的集合
    /// 不适用（模式由快照/manifest 承载）。
    /// EN: Allow none -> bm25 only when the collection was implicitly built by
    /// upsert replay without a create record; with a lost create record, a later
    /// sparse upsert carries the mode fact. Snapshot-restored collections do not
    /// qualify (the snapshot/manifest carries the mode).
    IfImplicitUpsert,
}

impl CollectionManager {
    pub fn new(default_dims: usize, default_index_type: &str) -> Self {
        CollectionManager {
            collections: Mutex::new(HashMap::new()),
            default_dims,
            default_index_type: default_index_type.to_string(),
            builder_cfg: None,
            default_ha_level: houyi_persistence::ha_level::HaLevel::Async,
        }
    }

    /// 注入 Collection 异步构图配置（生产装配必须调用，详见 [`CollectionBuilderConfig`]）。
    /// Inject the per-collection builder config (mandatory in production assembly;
    /// see [`CollectionBuilderConfig`]).
    pub fn with_builder_config(mut self, cfg: CollectionBuilderConfig) -> Self {
        self.builder_cfg = Some(cfg);
        self
    }

    /// 注入默认 HA 级别（P4.4.2；TOML `default_ha_level` 装配入口）。
    /// 枚举参数自带合法性，无需运行时守卫（P4.6.3）。
    /// EN: Inject the default HA level; the enum carries validity (P4.6.3).
    pub fn with_default_ha_level(mut self, level: houyi_persistence::ha_level::HaLevel) -> Self {
        self.default_ha_level = level;
        self
    }

    /// 旧 WAL/快照记录（无 metric 字段）的缺省度量：pq-flat 内部恒为平方 L2，
    /// hii-l0 内部单位化恒为 Cosine，其余历史后端实际按 Cosine 计分。
    /// EN: Default metric for legacy records without a metric field: pq-flat is
    /// squared-L2 internally, hii-l0 normalizes to cosine; all other historical
    /// backends actually scored by cosine.
    pub(crate) fn default_metric_for_index_type(index_type: &str) -> DistanceMetric {
        match index_type {
            "pq-flat" => DistanceMetric::L2,
            _ => DistanceMetric::Cosine,
        }
    }

    /// 隐式创建（upsert/pgwire 首次写入）实际采用的 index_type 与缺省度量。
    /// 创建记录必须与 get_or_create 的推断规则一致，否则重放会建成另一种后端。
    /// EN: The index_type and default metric an implicit creation actually uses;
    /// the create record must match `get_or_create`'s inference exactly, or replay
    /// would build a different backend.
    pub(crate) fn implicit_create_params(&self) -> (String, DistanceMetric) {
        let index_type = self.default_index_type.clone();
        let metric = Self::default_metric_for_index_type(&index_type);
        (index_type, metric)
    }

    /// 默认 index_type（恢复路径据此决定走历史隐式创建还是按快照度量显式建后端）。
    /// EN: Default index type (recovery uses it to choose between legacy implicit
    /// creation and an explicit backend built with the snapshot's metric).
    pub fn default_index_type(&self) -> &str {
        &self.default_index_type
    }

    /// 建集合参数的前置校验（不落 WAL 就能证明后端可构造）。
    /// EN: Pre-validate creation parameters so a WAL record is never written for a
    /// backend that cannot be constructed (that would poison cold replay).
    pub(crate) fn validate_create(
        index_type: &str,
        dims: usize,
        metric: DistanceMetric,
        hnsw: Option<HnswParams>,
        vector_dtype: VectorDtype,
        quantization: Quantization,
        sparse: SparseMode,
    ) -> Result<(), String> {
        if dims == 0 {
            return Err("collection dims must be > 0".to_string());
        }
        Self::create_backend(
            index_type,
            dims,
            metric,
            hnsw,
            vector_dtype,
            quantization,
            sparse,
        )
        .map(|_| ())
    }

    /// index_type → 后端构造（集合创建与快照恢复共用，未知类型/不支持度量 fail-fast）。
    /// `hnsw` 参数仅对 hnsw 后端有意义；传给其他后端属于调用方错误而非静默忽略。
    /// EN: index_type → backend construction (shared by creation and snapshot
    /// restore; unknown types and unsupported metric/index combinations fail fast).
    /// HNSW parameters are meaningful only for the hnsw backend; passing them to
    /// another backend is a caller error rather than a silent no-op.
    #[allow(clippy::too_many_arguments)]
    fn create_backend(
        index_type: &str,
        dims: usize,
        metric: DistanceMetric,
        hnsw: Option<HnswParams>,
        vector_dtype: VectorDtype,
        quantization: Quantization,
        sparse: SparseMode,
    ) -> Result<IndexBackend, String> {
        IndexBackend::metric_supported(index_type, metric)?;
        // 稀疏融合检索的过滤器路径只覆盖 hnsw/flat；其他后端建 bm25 会让
        // 「建了却查不了（带过滤）」成为静默陷阱，创建期直接 fail-fast。
        // EN: Filtered fusion search is only implemented for hnsw/flat; allowing
        // bm25 on other backends would create a silent "built but can't query
        // with filters" trap, so reject it at creation.
        if sparse == SparseMode::Bm25 && !sparse_bm25_backend_supported(index_type) {
            return Err(format!(
                "sparse bm25 is only supported for index_type 'hnsw'|'flat' (got {index_type:?})"
            ));
        }
        if let Some(p) = hnsw {
            if index_type != "hnsw" {
                return Err(format!(
                    "hnsw parameters are only supported for index_type 'hnsw' (got {index_type:?})"
                ));
            }
            p.validate()?;
        }
        // f16 存储只存在于 HNSW 路径；其他后端没有 dtype 形参，静默忽略会让
        // 创建声明与后端实际存储分叉，因此在构造入口统一 fail-fast。
        // EN: f16 storage exists only on the HNSW path; other backends have no
        // dtype parameter, and silently ignoring it would diverge the creation
        // declaration from actual storage — fail fast here.
        if vector_dtype != VectorDtype::F32 && index_type != "hnsw" {
            return Err(format!(
                "vector_dtype {} is only supported for index_type 'hnsw' (got {index_type:?})",
                vector_dtype.as_str()
            ));
        }
        // 图侧量化同理：int8 只存在于 HNSW 路径，其他后端没有量化形参；
        // 静默忽略会让创建声明与后端实际存储分叉（fail-fast）。
        // EN: Graph-side quantization follows the same rule: int8 exists only on
        // the HNSW path and other backends have no quantization parameter;
        // silently ignoring it would diverge the declaration from actual storage
        // (fail fast).
        if quantization != Quantization::None && index_type != "hnsw" {
            return Err(format!(
                "quantization {} is only supported for index_type 'hnsw' (got {index_type:?})",
                quantization.as_str()
            ));
        }
        // RaBitQ 的 1 bit/dim 估计器依赖余弦/平方 L2 的几何语义；ip/l1
        // 没有对应的角度表述，创建期必须拒绝，避免落到无意义的距离核。
        // EN: The RaBitQ 1-bit/dim estimator relies on cosine/squared-L2
        // geometry; ip/l1 have no angle-based form, so reject them at creation
        // rather than falling through to a meaningless distance kernel.
        if quantization == Quantization::RaBitQ
            && !matches!(metric, DistanceMetric::L2 | DistanceMetric::Cosine)
        {
            return Err(format!(
                "rabitq quantization requires metric l2|cosine (got {})",
                metric.as_str()
            ));
        }
        let backend = match index_type {
            "hnsw" => IndexBackend::new_hnsw_with_metric_and_dtype_and_quantization(
                dims,
                metric,
                vector_dtype,
                quantization,
            ),
            "clustered" => IndexBackend::new_clustered_with_metric(dims, metric),
            "flat-clustered" => IndexBackend::new_flat_clustered_with_metric(dims, metric),
            "flat" => IndexBackend::new_flat_with_metric(dims, metric),
            "pq-flat" => IndexBackend::new_pq_flat_with_metric(dims, metric),
            "hii-l0" => IndexBackend::new_hii_l0_with_metric(dims, metric),
            other => {
                return Err(format!(
                    "unsupported collection index_type {other:?} \
                     (expect hnsw|clustered|flat-clustered|flat|pq-flat|hii-l0)"
                ))
            }
        };
        if let Some(p) = hnsw {
            backend.set_hnsw_m(p.m)?;
            backend.set_hnsw_ef_construction(p.ef_construction)?;
            backend.set_hnsw_ef_search(p.ef_search)?;
        }
        Ok(backend)
    }

    /// 获取 Collection handle（无需持有锁）
    /// EN: Get a Collection handle (no lock held).
    pub fn get(&self, name: &str) -> Option<Arc<Collection>> {
        self.collections.lock().unwrap().get(name).cloned()
    }

    /// 获取或隐式创建 Collection。new_dims=None 时用默认维度。
    /// 仅 crate 内路由（HTTP/pgwire）调用。
    /// Crate-internal routing only (HTTP/pgwire handlers).
    pub(crate) fn get_or_create(
        &self,
        name: &str,
        new_dims: Option<usize>,
    ) -> Result<Arc<Collection>, String> {
        Ok(self.get_or_create_flagged(name, new_dims)?.0)
    }

    /// 同 [`get_or_create`](Self::get_or_create)，额外返回「本次是否新建」——
    /// collections WAL 只在新建时落 `collection_create` 记录（幂等重放的依据）。
    /// EN: Same as `get_or_create`, plus a "created this call" flag — the
    /// collections WAL writes `collection_create` only on actual creation
    /// (the basis for idempotent replay).
    pub(crate) fn get_or_create_flagged(
        &self,
        name: &str,
        new_dims: Option<usize>,
    ) -> Result<(Arc<Collection>, bool), String> {
        self.get_or_create_impl(
            name,
            new_dims,
            None,
            None,
            None,
            None,
            None,
            None,
            true,
            SparseReconcile::Never,
        )
    }

    /// 按快照/WAL 记录（含度量与 HNSW 参数）创建 collection；记录中的度量缺失时
    /// 退回 [`Self::default_metric_for_index_type`]。仅保留给既有单元测试；
    /// 生产恢复/重放路径一律显式传 quantization。
    /// EN: Create a collection from a snapshot/WAL record carrying the metric and
    /// HNSW parameters; a missing metric falls back to
    /// [`Self::default_metric_for_index_type`]. Kept only for existing unit
    /// tests; production restore/replay paths always pass quantization explicitly.
    #[cfg(test)]
    pub(crate) fn get_or_create_typed_with_metric(
        &self,
        name: &str,
        dims: usize,
        index_type: &str,
        metric: Option<DistanceMetric>,
        hnsw: Option<HnswParams>,
        vector_dtype: Option<VectorDtype>,
    ) -> Result<(Arc<Collection>, bool), String> {
        self.get_or_create_typed_with_metric_and_quantization(
            name,
            dims,
            index_type,
            metric,
            hnsw,
            vector_dtype,
            None,
            None,
        )
    }

    /// 同 [`get_or_create_typed_with_metric`](Self::get_or_create_typed_with_metric)，
    /// 额外携带图侧量化（P2）与稀疏模式（P4）；`None` = 旧记录（不比对、按
    /// 未量化/无稀疏创建）。legacy 图快照恢复拿不到稀疏模式，故传 `None`。
    /// EN: Same as `get_or_create_typed_with_metric` plus graph-side quantization
    /// (P2) and sparse mode (P4); `None` = legacy record (not compared, built
    /// unquantized/without sparse). Legacy graph-snapshot restore cannot know the
    /// sparse mode and therefore passes `None`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_create_typed_with_metric_and_quantization(
        &self,
        name: &str,
        dims: usize,
        index_type: &str,
        metric: Option<DistanceMetric>,
        hnsw: Option<HnswParams>,
        vector_dtype: Option<VectorDtype>,
        quantization: Option<Quantization>,
        sparse: Option<SparseMode>,
    ) -> Result<(Arc<Collection>, bool), String> {
        self.get_or_create_impl(
            name,
            Some(dims),
            Some(index_type),
            metric,
            hnsw,
            vector_dtype,
            quantization,
            sparse,
            true,
            SparseReconcile::Never,
        )
    }

    /// 重放专用创建入口：现存集合只校验 dims/index_type/metric，**不**校验
    /// HNSW 参数。图快照恢复会把快照头中的运行期调参写回后端 atomics，而
    /// `collection_create` 记录保存的是创建时参数；严格比对会让「先恢复快照、
    /// 再重放覆盖段内的 create 记录」在冷启动时失败（OCR 195e6be #11）。
    ///
    /// EN: Replay-only creation entry: an existing collection is checked for
    /// dims/index_type/metric but **not** HNSW parameters. Graph-snapshot restore
    /// writes runtime-tuned values from the snapshot header back into the backend
    /// atomics, while the `collection_create` record holds creation-time
    /// parameters; a strict comparison would abort cold start when a covered
    /// create record is replayed after snapshot restore (OCR 195e6be #11).
    // 仅保留给既有单元测试的兼容入口；生产复制/重放路径一律显式传 quantization。
    // EN: Compatibility entry kept only for existing unit tests; production
    // replication/replay paths always pass quantization explicitly.
    #[cfg(test)]
    pub(crate) fn get_or_create_replayed(
        &self,
        name: &str,
        dims: usize,
        index_type: &str,
        metric: Option<DistanceMetric>,
        hnsw: Option<HnswParams>,
        vector_dtype: Option<VectorDtype>,
    ) -> Result<(Arc<Collection>, bool), String> {
        self.get_or_create_replayed_with_quantization(
            name,
            dims,
            index_type,
            metric,
            hnsw,
            vector_dtype,
            None,
            None,
        )
    }

    /// 重放专用入口 + 图侧量化（P2）+ 稀疏模式（P4）；`None` = 旧记录（不比对、
    /// 按未量化/无稀疏创建）。稀疏对账升级（none → bm25）在本入口（create
    /// 记录重放）与 [`Self::get_or_create_for_collection_upsert`]（无 create
    /// 记录的 upsert 重放）生效；其余入口严格拒绝。
    /// EN: Replay-only entry plus graph-side quantization (P2) and sparse mode
    /// (P4); `None` = legacy record (not compared, built unquantized/without
    /// sparse). The sparse reconciliation upgrade (none -> bm25) applies through
    /// this entry (create-record replay) and
    /// [`Self::get_or_create_for_collection_upsert`] (upsert replay without a
    /// create record); every other entry rejects it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_create_replayed_with_quantization(
        &self,
        name: &str,
        dims: usize,
        index_type: &str,
        metric: Option<DistanceMetric>,
        hnsw: Option<HnswParams>,
        vector_dtype: Option<VectorDtype>,
        quantization: Option<Quantization>,
        sparse: Option<SparseMode>,
    ) -> Result<(Arc<Collection>, bool), String> {
        self.get_or_create_impl(
            name,
            Some(dims),
            Some(index_type),
            metric,
            hnsw,
            vector_dtype,
            quantization,
            sparse,
            false,
            SparseReconcile::Always,
        )
    }

    /// collections WAL 重放专用入口：`collection_create` 记录缺失时，按
    /// 本批 upsert 是否携带稀疏数据决定新建集合的稀疏模式（带稀疏 → bm25，
    /// 否则 none）。现存集合的模式对账取决于它是否由本重放流的 upsert
    /// 隐式建立（`implicit_upsert_built`，与是否存在 create 记录无关，
    /// 因为快照恢复的集合同样没有 create 记录）：隐式建立时允许
    /// none → bm25——否则「先 dense-only 批、后稀疏批」的合法历史会被
    /// 误判为腐坏（OCR 裁决）；其余情况（显式 create 记录、快照恢复）
    /// 模式是权威事实，严格比对，不允许静默升级。
    /// 本方法由单条串行 apply 流调用（启动重放与 follower 实时 apply 共用），
    /// 标记位按跨线程可见性设计，不依赖单线程假设。
    ///
    /// EN: Collections-WAL replay entry: when the `collection_create` record is
    /// missing, the new collection's sparse mode follows whether this upsert
    /// batch carries sparse data (sparse -> bm25, otherwise none). For an
    /// existing collection, reconciliation depends on whether it was
    /// implicitly built by this replay stream's upserts (`implicit_upsert_built`,
    /// independent of whether a create record exists, since snapshot-restored
    /// collections also lack one): implicit builds allow none -> bm25 —
    /// otherwise the legitimate "dense-only batch first, sparse batch later"
    /// history would be misclassified as corruption (OCR adjudication); all
    /// other cases (explicit create record, snapshot restore) treat the mode as
    /// authoritative and compare strictly, never silently upgrading. This
    /// method is called from one serial apply stream (startup replay and live
    /// follower apply share it); the marker is designed for cross-thread
    /// visibility rather than assuming single-threadedness.
    pub(crate) fn get_or_create_for_collection_upsert(
        &self,
        name: &str,
        new_dims: Option<usize>,
        sparse_hint: Option<SparseMode>,
    ) -> Result<(Arc<Collection>, bool), String> {
        let (col, created) = self.get_or_create_impl(
            name,
            new_dims,
            None,
            None,
            None,
            None,
            None,
            sparse_hint,
            false,
            SparseReconcile::IfImplicitUpsert,
        )?;
        if created {
            // 本入口新建 = create 记录缺失时的隐式建立；后续 upsert 记录若携带
            // 稀疏数据，可据此对账 none → bm25（快照恢复的集合不会走到这里）。
            // EN: Created by this entry means an implicit build while the create
            // record was missing; later upsert records carrying sparse data may
            // reconcile none -> bm25 on that basis (snapshot restores never do).
            col.mark_implicit_upsert_built();
        }
        Ok((col, created))
    }

    #[allow(clippy::too_many_arguments)]
    fn get_or_create_impl(
        &self,
        name: &str,
        new_dims: Option<usize>,
        requested_index_type: Option<&str>,
        requested_metric: Option<DistanceMetric>,
        requested_hnsw: Option<HnswParams>,
        requested_vector_dtype: Option<VectorDtype>,
        requested_quantization: Option<Quantization>,
        requested_sparse: Option<SparseMode>,
        strict_hnsw: bool,
        sparse_reconcile: SparseReconcile,
    ) -> Result<(Arc<Collection>, bool), String> {
        let mut guard = self.collections.lock().unwrap();
        if let Some(col) = guard.get(name) {
            if let Some(d) = new_dims {
                if col.dims != d {
                    return Err(format!(
                        "collection '{name}' has dims={}, cannot insert with dims={d}",
                        col.dims
                    ));
                }
            }
            if let Some(t) = requested_index_type {
                if col.index_type != t {
                    return Err(format!(
                        "collection '{name}' has index_type={:?}, snapshot requires {t:?}",
                        col.index_type
                    ));
                }
            }
            if let Some(m) = requested_metric {
                if col.metric != m {
                    return Err(format!(
                        "collection '{name}' has metric={}, record requires {}",
                        col.metric.as_str(),
                        m.as_str()
                    ));
                }
            }
            // 存储编码创建后不可变：记录/快照显式携带时必须与现存集合一致，
            // 分叉意味着 WAL/快照与内存态描述的不是同一个集合（重放/恢复拒绝）。
            // EN: The storage encoding is immutable after creation: when a record
            // or snapshot carries one explicitly it must match the live collection;
            // divergence means WAL/snapshot and memory describe different
            // collections (replay/restore rejects it).
            if let Some(d) = requested_vector_dtype {
                if col.vector_dtype != d {
                    return Err(format!(
                        "collection '{name}' has vector_dtype={}, record requires {}",
                        col.vector_dtype.as_str(),
                        d.as_str()
                    ));
                }
            }
            // 图侧量化创建后不可变：记录/快照显式携带时同样严格比对（含显式
            // none），分叉意味着记录与内存态描述的不是同一个集合。
            // EN: Graph-side quantization is immutable after creation: an explicit
            // record/snapshot value (including an explicit none) must match the
            // live collection; divergence means they describe different collections.
            if let Some(q) = requested_quantization {
                if col.quantization != q {
                    return Err(format!(
                        "collection '{name}' has quantization={}, record requires {}",
                        col.quantization.as_str(),
                        q.as_str()
                    ));
                }
            }
            // 稀疏模式创建后不可变：记录显式携带时严格比对。例外只有冷启动
            // 对账（none → bm25）：create 记录重放（legacy 图快照恢复先建集合、
            // 快照头不含稀疏模式，create 记录才是事实），以及 create 记录丢失
            // 后由 dense-only 批次隐式建立的集合（后续稀疏 upsert 携带事实）。
            // 对账能力由显式策略控制，避免和 HNSW 严格比对策略耦联（OCR 裁决）。
            // EN: The sparse mode is immutable after creation: an explicit
            // record value must match. The only exceptions are cold-start
            // reconciliation (none -> bm25): create-record replay (a legacy
            // graph-snapshot restore creates the collection before the mode is
            // known, and the create record is the fact), and a collection
            // implicitly built by dense-only batches after the create record was
            // lost (a later sparse upsert carries the fact). The capability is an
            // explicit policy so it no longer rides on the HNSW strictness flag
            // (OCR adjudication).
            if let Some(s) = requested_sparse {
                let current = col.sparse_mode();
                if current != s {
                    let may_reconcile = match sparse_reconcile {
                        SparseReconcile::Never => false,
                        SparseReconcile::Always => true,
                        // 只认「upsert 重放隐式建立」，不是「create_logged=false」：
                        // 快照恢复的集合同样没有 create 记录，但模式已是权威事实。
                        // EN: Only "implicitly built by upsert replay", not
                        // "create_logged == false": snapshot-restored collections
                        // also lack a create record, yet their mode is authoritative.
                        SparseReconcile::IfImplicitUpsert => col.implicit_upsert_built(),
                    };
                    if may_reconcile && current == SparseMode::None && s == SparseMode::Bm25 {
                        col.reconcile_sparse_mode(s)?;
                    } else {
                        return Err(format!(
                            "collection '{name}' has sparse={}, record requires {}",
                            current.as_str(),
                            s.as_str()
                        ));
                    }
                }
            }
            // 记录带 HNSW 参数时同样要求与现存集合一致：只比 dims/index_type/metric
            // 会让「同度量、不同图参数」的显式重建被静默当作已存在（OCR #08）。
            // 仅限显式创建（strict_hnsw）：重放路径下现存集合的参数可能已被
            // 运行期调参/快照恢复改写，严格比对会拒绝合法的冷启动重放。
            // EN: A record carrying HNSW parameters must match the existing
            // collection too; comparing only dims/index_type/metric would silently
            // treat an explicit recreate with different graph parameters as an
            // existing collection (OCR #08). Only for explicit creation
            // (strict_hnsw): on replay, live parameters may have been tuned or
            // overwritten by snapshot restore, and a strict comparison would
            // reject a legitimate cold-start replay.
            if let Some(p) = requested_hnsw {
                let current = col.hnsw_params();
                if current != p {
                    if strict_hnsw {
                        return Err(format!(
                            "collection '{name}' has hnsw m={}, ef_construction={}, ef_search={}, \
                             record requires m={}, ef_construction={}, ef_search={}",
                            current.m,
                            current.ef_construction,
                            current.ef_search,
                            p.m,
                            p.ef_construction,
                            p.ef_search
                        ));
                    }
                    // 重放路径容忍漂移（参数可能被运行期调参/快照恢复改写），
                    // 但必须留痕：真正的陈旧/损坏 WAL 与合法漂移不能都静默吞掉
                    // （OCR cf1d19f #01）。
                    // EN: Replay tolerates drift (parameters may have been
                    // runtime-tuned or overwritten by snapshot restore), but it
                    // must leave a trace: a genuinely stale/corrupt WAL record
                    // and a legitimate drift must not both be silently swallowed
                    // (OCR cf1d19f #01).
                    tracing::warn!(
                        collection = name,
                        current_m = current.m,
                        current_ef_construction = current.ef_construction,
                        current_ef_search = current.ef_search,
                        recorded_m = p.m,
                        recorded_ef_construction = p.ef_construction,
                        recorded_ef_search = p.ef_search,
                        "replayed collection_create hnsw params differ from live collection; keeping live values"
                    );
                }
            }
            return Ok((col.clone(), false));
        }

        let dims = new_dims.unwrap_or(self.default_dims);
        let index_type = requested_index_type.unwrap_or(&self.default_index_type);
        let metric =
            requested_metric.unwrap_or_else(|| Self::default_metric_for_index_type(index_type));
        // 缺省 f32：旧 WAL/快照（无 dtype 字段）与隐式创建都落在这里。
        // EN: f32 by default: legacy WAL/snapshot records (no dtype field) and
        // implicit creation both land here.
        let vector_dtype = requested_vector_dtype.unwrap_or(VectorDtype::F32);
        // 缺省 none：旧 WAL/快照（无量化字段）与隐式创建都落在这里。
        // EN: none by default: legacy WAL/snapshot records (no quantization field)
        // and implicit creation both land here.
        let quantization = requested_quantization.unwrap_or(Quantization::None);
        // 缺省 none：旧记录与隐式创建（首批 upsert 建集合）恒为无稀疏。
        // EN: none by default: legacy records and implicit creation (first
        // upsert) always start without sparse.
        let sparse_mode = requested_sparse.unwrap_or(SparseMode::None);
        let backend = Self::create_backend(
            index_type,
            dims,
            metric,
            requested_hnsw,
            vector_dtype,
            quantization,
            sparse_mode,
        )?;

        // HNSW 后端需要异步构图 worker 才能把待构图队列 flush 成可检索状态；
        // 「哪个后端需要 worker」的分发规则集中在 IndexBackend 适配层，此处不重复
        // 编码（否则未来新后端需要 worker 时，这里的第二处分发会静默返回 None，
        // 重新引入写入不可检索的缺口）。
        // EN: The which-backend-needs-a-worker dispatch lives solely in the
        // IndexBackend adapter; re-encoding it here would silently return None for
        // any future worker-needing backend, reintroducing the unsearchable gap.
        let builder_worker = self.builder_cfg.and_then(|cfg| {
            backend.start_async_builder_worker(cfg.threshold, cfg.tick_ms, cfg.policy)
        });

        // 生产必须注入 builder 配置的契约没有任何静态保障：HNSW 后端 + 无配置
        // 意味着写入永不检索（本修复的缺口）。单测故意不配置，故只告警一次不
        // panic；告警走 tracing（生产由 main 安装订阅器），不污染正常路径。
        // EN: Nothing statically enforces "production must inject the config":
        // an HNSW backend without it means writes stay unsearchable forever (the
        // gap this fixes). Tests intentionally run unconfigured, so warn once via
        // tracing instead of panicking (subscriber installed in main).
        if self.builder_cfg.is_none() && matches!(backend, IndexBackend::Hnsw(_)) {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    "[houyi-collection] HNSW backend created without builder config; \
                     writes stay unsearchable until manually drained (tests only — \
                     production must inject CollectionBuilderConfig)"
                );
            });
        }

        let col = Arc::new(Collection {
            name: name.to_string(),
            dims,
            metric,
            index_type: index_type.to_string(),
            vector_dtype,
            quantization,
            sparse_mode: std::sync::atomic::AtomicU8::new(sparse_mode.code()),
            // bm25 集合预置空索引（mode==Bm25 ⇒ 句柄恒可用）；none 保持未初始化。
            // EN: bm25 collections pre-install an empty index (mode==Bm25 implies
            // a usable handle); none stays uninitialized.
            sparse: if sparse_mode == SparseMode::Bm25 {
                std::sync::OnceLock::from(std::sync::RwLock::new(Bm25Index::new()))
            } else {
                std::sync::OnceLock::new()
            },
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            backend,
            scalars: Mutex::new(CollectionScalars::new()),
            write_lock: tokio::sync::Mutex::new(()),
            ha_level: std::sync::atomic::AtomicU8::new(self.default_ha_level.code()),
            deleted: std::sync::atomic::AtomicBool::new(false),
            create_logged: std::sync::atomic::AtomicBool::new(false),
            implicit_upsert_built: std::sync::atomic::AtomicBool::new(false),
            applied_lsn: AtomicU64::new(0),
            last_compaction_unix: AtomicU64::new(0),
            last_compaction_dropped: AtomicU64::new(0),
            compactions_ok_total: AtomicU64::new(0),
            compactions_skipped_total: AtomicU64::new(0),
            compaction_failures_total: AtomicU64::new(0),
            compaction_dropped_rows_total: AtomicU64::new(0),
            sync_timeout_warns: std::sync::atomic::AtomicU64::new(0),
            _builder_worker: builder_worker,
        });

        guard.insert(name.to_string(), col.clone());
        Ok((col, true))
    }

    /// 列出所有 Collection 名称（HTTP 列表端点与服务端周期任务共用）。
    /// EN: List all collection names (shared by the HTTP list endpoint and the
    /// server's periodic tasks).
    pub fn list(&self) -> Vec<String> {
        self.collections.lock().unwrap().keys().cloned().collect()
    }

    /// Collection 数量（当前仅 crate 内使用）。
    /// EN: Number of collections (currently crate-internal only).
    #[allow(dead_code)]
    pub(crate) fn count(&self) -> usize {
        self.collections.lock().unwrap().len()
    }

    /// 仅当 map 中仍是**同一个** Arc 时摘除，避免「旧删除删掉同名重建的新
    /// collection」：删除流程先查名取 Arc，期间该名字可能已被删除并重新创建。
    /// EN: Remove only when the map still holds **the same** Arc, preventing an
    /// old drop from removing a same-named collection recreated concurrently:
    /// the delete flow resolves the Arc by name, and the name may be dropped
    /// and recreated before the removal.
    pub(crate) fn remove_if_same(&self, name: &str, col: &Arc<Collection>) -> bool {
        let mut guard = self.collections.lock().unwrap();
        match guard.get(name) {
            Some(current) if Arc::ptr_eq(current, col) => guard.remove(name).is_some(),
            _ => false,
        }
    }
}

/// 若 collection 的 `collection_create` 尚未落盘，由当前写锁持有者按 collection
/// 自身的创建参数补落，并推进 applied_lsn。**必须在持有该 collection 的
/// write_lock 时调用**——这把锁是「WAL 中 create 永远先于该 Arc 任何变更记录」
/// 的唯一保证（含删除→同名重建竞态：新 Arc 在首个写操作前补落 create）。
/// 幂等：已落盘时只做一次 Acquire 读，热路径零开销。
/// EN: If the collection's `collection_create` record is not durable yet, log it
/// from the collection's own creation parameters and advance applied_lsn. Must be
/// called with the collection's write lock held — that lock is the sole guarantee
/// that create precedes any mutation of this Arc in the WAL (including across a
/// delete/recreate race, where the new Arc logs its create before its first
/// mutation). Idempotent: once logged, the hot path costs one Acquire load.
pub(crate) fn ensure_create_logged(
    wal: &crate::types::WalHandle,
    name: &str,
    col: &Collection,
) -> Result<Option<u64>, String> {
    if col.create_logged() {
        return Ok(None);
    }
    // HNSW 参数只对 hnsw 后端写入：其他后端持有的只是占位默认值，写进创建记录
    // 会被重放的 create_backend 判为「参数给了错误后端」而 fail-fast。
    // EN: HNSW parameters are logged only for the hnsw backend. Other backends
    // only carry placeholder defaults; writing them into the create record would
    // make replay's create_backend fail fast on params for the wrong backend.
    let hnsw = (col.index_type == "hnsw").then_some(());
    let p = col.hnsw_params();
    let options = houyi_persistence::collection_wal::CollectionCreateOptions {
        metric: Some(col.metric.as_str().to_string()),
        hnsw_m: hnsw.map(|()| p.m),
        hnsw_ef_construction: hnsw.map(|()| p.ef_construction),
        hnsw_ef_search: hnsw.map(|()| p.ef_search),
        // 编码是创建时事实：hnsw 恒写入（含显式 f32），非 hnsw 为 None。
        // EN: The encoding is a creation-time fact: always written for hnsw
        // (explicit f32 included), None for other backends.
        vector_dtype: hnsw.map(|()| col.vector_dtype.as_str().to_string()),
        // 量化同 dtype 纪律：hnsw 恒写入（含显式 none），非 hnsw 为 None。
        // EN: Quantization follows the dtype discipline: always written for hnsw
        // (explicit none included), None for other backends.
        quantization: hnsw.map(|()| col.quantization.as_str().to_string()),
        // 稀疏模式是创建事实：恒写入规范串（含 none）。bm25 仅 hnsw/flat
        // 可构造，非法组合在解码/重放侧 fail-fast。
        // EN: The sparse mode is a creation fact: always written as its canonical
        // string (none included). bm25 is only constructible on hnsw/flat; illegal
        // combinations fail fast at decode/replay.
        sparse: Some(col.sparse_mode().as_str().to_string()),
    };
    let payload = houyi_persistence::collection_wal::encode_collection_create_with_options(
        name,
        col.dims,
        &col.index_type,
        &options,
    )?;
    let entry = wal.append(payload).map_err(|e| e.to_string())?;
    col.note_applied_lsn(entry.lsn);
    col.mark_create_logged();
    Ok(Some(entry.lsn))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(color: &str) -> HashMap<String, ScalarValue> {
        HashMap::from([("color".to_string(), ScalarValue::Text(color.to_string()))])
    }

    fn color_of(payloads: &HashMap<u64, HashMap<String, ScalarValue>>, id: u64) -> String {
        match payloads.get(&id).and_then(|r| r.get("color")) {
            Some(ScalarValue::Text(s)) => s.clone(),
            other => panic!("missing color for id {id}: {other:?}"),
        }
    }

    /// `Collection` 不实现 Debug，`unwrap_err` 的 `T: Debug` 约束不成立。
    /// EN: `Collection` implements no Debug, so `unwrap_err`'s `T: Debug` bound
    /// does not hold.
    fn expect_create_err<T>(result: Result<T, String>, what: &str) -> String {
        match result {
            Ok(_) => panic!("{what}: expected an error"),
            Err(e) => e,
        }
    }

    #[test]
    fn scalar_stats_and_last_wins_view() {
        let mut s = CollectionScalars::new();
        s.append_row(1, row("red"));
        s.append_row(1, row("blue"));
        s.append_row(2, row("green"));
        s.tombstone(2);

        let stats = s.stats_snapshot();
        assert_eq!(stats.physical_rows, 3);
        assert_eq!(stats.live_rows, 1);
        assert_eq!(stats.stale_rows, 2);
        assert_eq!(stats.tombstone_ids, 1);

        let live = s.live_rows_last_wins();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].0, 1);
        assert_eq!(
            live[0].1.get("color"),
            Some(&ScalarValue::Text("blue".to_string()))
        );
    }

    /// 回归（公共 issue #6）：同 id 重写后，旧 payload 不得再命中过滤。
    /// EN: Regression (public issue #6): after re-upserting an id, its old payload
    /// must no longer match a filter.
    #[test]
    fn matching_ids_only_sees_the_last_row_per_id() {
        use houyi_core::ScalarPredicate;

        let mut s = CollectionScalars::new();
        s.append_row(1, row("red"));
        s.append_row(1, row("blue"));
        s.append_row(2, row("red"));
        s.tombstone(2);

        let red = ScalarPredicate::Eq(ScalarValue::Text("red".to_string()));
        let blue = ScalarPredicate::Eq(ScalarValue::Text("blue".to_string()));
        assert!(s.matching_ids("color", &red).is_empty());
        assert_eq!(s.matching_ids("color", &blue), vec![1]);

        // 压实前后必须同结果（修复前只有压实后才不再命中旧值）。
        // EN: The result must be identical before and after compaction (before the
        // fix only compaction stopped stale values from matching).
        let compacted = CollectionScalars::from_rows(s.live_rows_last_wins(), Vec::new());
        assert!(compacted.matching_ids("color", &red).is_empty());
        assert_eq!(compacted.matching_ids("color", &blue), vec![1]);

        // 快照恢复保留物理旧行；last_row 索引随 from_rows 重建，过滤语义不变。
        // EN: Snapshot restore keeps the physical stale rows; from_rows rebuilds the
        // last-row index, so filter semantics are unchanged.
        let (rows, tombstones) = s.snapshot_view();
        let restored = CollectionScalars::from_rows(rows, tombstones);
        assert!(restored.matching_ids("color", &red).is_empty());
        assert_eq!(restored.matching_ids("color", &blue), vec![1]);
    }

    #[test]
    fn hnsw_params_reflect_runtime_tuning() {
        // OCR #34：运行期调参只改后端；hnsw_params 必须读当前值而非创建缓存。
        // EN: OCR #34 — runtime tuning mutates only the backend; hnsw_params must
        // read current values, not a creation-time cache.
        let mgr = CollectionManager::new(4, "hnsw");
        let col = mgr.get_or_create("c", Some(4)).unwrap();
        let before = col.hnsw_params();
        // 调参值必须由运行期默认值派生：`HOUYI_VECTOR_AUTO_HNSW=false` 时
        // 默认 ef_search 本身就是 512，硬编码 512 会让 assert_ne 因环境而失败
        // （OCR 195e6be #12）。
        // EN: Derive the tuned values from the runtime defaults: with
        // `HOUYI_VECTOR_AUTO_HNSW=false` the default ef_search is already 512,
        // so hardcoding 512 makes assert_ne fail for an unrelated environment
        // reason (OCR 195e6be #12).
        let tuned_ef_search = before.ef_search.saturating_add(1);
        let tuned_m = before.m.saturating_add(1).max(2);
        let tuned_efc = before.ef_construction.saturating_add(1).max(1);
        col.backend.set_hnsw_ef_search(tuned_ef_search).unwrap();
        assert_eq!(col.hnsw_params().ef_search, tuned_ef_search);
        col.backend.set_hnsw_m(tuned_m).unwrap();
        assert_eq!(col.hnsw_params().m, tuned_m);
        col.backend.set_hnsw_ef_construction(tuned_efc).unwrap();
        assert_eq!(col.hnsw_params().ef_construction, tuned_efc);
        assert_ne!(col.hnsw_params().ef_search, before.ef_search);
    }

    #[test]
    fn recreate_with_different_hnsw_params_conflicts() {
        // OCR #08：显式重建只改 HNSW 参数也必须拒绝，不得静默当作已存在。
        // EN: OCR #08 — an explicit recreate that only changes HNSW parameters
        // must be rejected instead of silently treated as an existing collection.
        let mgr = CollectionManager::new(4, "hnsw");
        let a = HnswParams {
            m: 16,
            ef_construction: 400,
            ef_search: 256,
        };
        let (_, created) = mgr
            .get_or_create_typed_with_metric(
                "c",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                Some(a),
                None,
            )
            .unwrap();
        assert!(created);
        // 同参数重复创建：幂等（重放重复 create 记录）。
        // EN: Same parameters are idempotent (duplicate create records on replay).
        let (_, created) = mgr
            .get_or_create_typed_with_metric(
                "c",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                Some(a),
                None,
            )
            .unwrap();
        assert!(!created);
        let b = HnswParams { m: 32, ..a };
        // Ok 侧含非 Debug 的 Arc<Collection>，用 match 而非 expect_err。
        // EN: the Ok side holds a non-Debug Arc<Collection>, so match instead of
        // expect_err.
        let err = match mgr.get_or_create_typed_with_metric(
            "c",
            4,
            "hnsw",
            Some(DistanceMetric::Cosine),
            Some(b),
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("different hnsw params must conflict"),
        };
        assert!(err.contains("hnsw"), "got: {err}");
    }

    #[test]
    fn replayed_create_tolerates_runtime_tuned_hnsw_params() {
        // 图快照恢复会把运行期调参写回后端 atomics，而 create 记录保存创建时
        // 参数；重放入口必须接受这种漂移，否则合法的冷启动会被拒绝
        // （OCR 195e6be #11）。backend/metric 漂移仍必须 fail-fast。
        // EN: Graph-snapshot restore writes runtime-tuned values back into the
        // backend atomics while the create record holds creation-time parameters;
        // the replay entry must tolerate this drift or a legitimate cold start is
        // rejected (OCR 195e6be #11). backend/metric drift still fails fast.
        let mgr = CollectionManager::new(4, "hnsw");
        let a = HnswParams {
            m: 16,
            ef_construction: 400,
            ef_search: 256,
        };
        let (col, created) = mgr
            .get_or_create_typed_with_metric(
                "c",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                Some(a),
                None,
            )
            .unwrap();
        assert!(created);
        col.backend.set_hnsw_m(32).unwrap();
        col.backend.set_hnsw_ef_construction(800).unwrap();
        col.backend.set_hnsw_ef_search(512).unwrap();
        let (_, created) = mgr
            .get_or_create_replayed("c", 4, "hnsw", Some(DistanceMetric::Cosine), Some(a), None)
            .unwrap();
        assert!(!created);
        let err = match mgr.get_or_create_replayed(
            "c",
            4,
            "flat",
            Some(DistanceMetric::Cosine),
            None,
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("index_type drift must still conflict on replay"),
        };
        assert!(err.contains("index_type"), "got: {err}");
        // 度量漂移同样必须 fail-fast，不能只锁定 index_type 分支
        // （OCR cf1d19f #02）。
        // EN: metric drift must fail fast too; do not pin only the index_type
        // branch (OCR cf1d19f #02).
        let err = match mgr.get_or_create_replayed(
            "c",
            4,
            "hnsw",
            Some(DistanceMetric::InnerProduct),
            None,
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("metric drift must still conflict on replay"),
        };
        assert!(err.contains("metric"), "got: {err}");
    }

    #[test]
    fn hnsw_param_bounds_match_library() {
        // OCR #39：m=1 在库层 set_hnsw_m 被拒，校验边界必须同为 2..=1024。
        // EN: OCR #39 — m=1 is rejected by the library's set_hnsw_m, so validation
        // must use the same 2..=1024 bound.
        let base = HnswParams {
            m: 16,
            ef_construction: 400,
            ef_search: 256,
        };
        assert!(HnswParams { m: 1, ..base }.validate().is_err());
        assert!(HnswParams { m: 1025, ..base }.validate().is_err());
        assert!(HnswParams {
            ef_construction: 0,
            ..base
        }
        .validate()
        .is_err());
        assert!(HnswParams { m: 2, ..base }.validate().is_ok());
        assert!(HnswParams { m: 1024, ..base }.validate().is_ok());
    }

    #[test]
    fn vector_dtype_is_immutable_and_hnsw_only() {
        // 编码是创建时事实：同值幂等、异值冲突、非 hnsw 后端拒绝。
        // EN: The encoding is a creation-time fact: same value is idempotent,
        // a different value conflicts, and non-hnsw backends are rejected.
        let mgr = CollectionManager::new(4, "hnsw");
        let (col, created) = mgr
            .get_or_create_typed_with_metric(
                "c",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                None,
                Some(VectorDtype::F16),
            )
            .unwrap();
        assert!(created);
        assert_eq!(col.vector_dtype(), VectorDtype::F16);
        let (_, created) = mgr
            .get_or_create_replayed(
                "c",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                None,
                Some(VectorDtype::F16),
            )
            .unwrap();
        assert!(!created);
        let err = match mgr.get_or_create_replayed(
            "c",
            4,
            "hnsw",
            Some(DistanceMetric::Cosine),
            None,
            Some(VectorDtype::F32),
        ) {
            Err(e) => e,
            Ok(_) => panic!("vector_dtype drift must conflict"),
        };
        assert!(err.contains("vector_dtype"), "got: {err}");
        assert!(mgr
            .get_or_create_typed_with_metric(
                "flatc",
                4,
                "flat",
                Some(DistanceMetric::Cosine),
                None,
                Some(VectorDtype::F16),
            )
            .is_err());
    }

    #[test]
    fn snapshot_view_keeps_stale_rows_and_roundtrips() {
        let mut s = CollectionScalars::new();
        s.append_row(1, row("red"));
        s.append_row(1, row("blue"));
        s.tombstone(9);

        let (rows, tombstones) = s.snapshot_view();
        assert_eq!(rows.len(), 2, "stale rows must be preserved in snapshots");
        assert_eq!(tombstones, vec![9]);

        let restored = CollectionScalars::from_rows(rows, tombstones);
        assert_eq!(restored.row_count(), 2);
        assert_eq!(restored.stats_snapshot(), s.stats_snapshot());
        assert_eq!(color_of(&restored.payloads_for(&[1]), 1), "blue");
    }

    #[tokio::test]
    async fn compact_drops_stale_and_tombstoned_rows() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (col, _) = mgr.get_or_create_flagged("c", Some(4)).unwrap();
        {
            let mut s = col.scalars.lock().unwrap();
            s.append_row(1, row("red"));
            s.append_row(1, row("blue"));
            s.append_row(2, row("green"));
            s.tombstone(2);
        }

        let result = col.compact_scalars(true, 0.5).await.unwrap();
        assert_eq!(result.physical_before, 3);
        assert_eq!(result.physical_after, 1);
        assert_eq!(result.dropped_rows, 2);
        assert_eq!(result.skipped_reason, None);

        let s = col.scalars.lock().unwrap();
        assert_eq!(s.row_count(), 1);
        let stats = s.stats_snapshot();
        assert_eq!(stats.live_rows, 1);
        assert_eq!(stats.stale_rows, 0);
        assert_eq!(stats.tombstone_ids, 0);
        assert_eq!(color_of(&s.payloads_for(&[1]), 1), "blue");
    }

    #[tokio::test]
    async fn compact_respects_stale_ratio_unless_forced() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (col, _) = mgr.get_or_create_flagged("c2", Some(4)).unwrap();
        {
            let mut s = col.scalars.lock().unwrap();
            s.append_row(1, row("red"));
            s.append_row(1, row("blue"));
        }
        // 1 stale / 2 physical = 50%，阈值 90% → 跳过。
        // EN: 1 stale / 2 physical = 50%, threshold 90% → skip.
        let skipped = col.compact_scalars(false, 0.9).await.unwrap();
        assert_eq!(skipped.skipped_reason.as_deref(), Some("below_threshold"));
        assert_eq!(col.scalars.lock().unwrap().row_count(), 2);

        let forced = col.compact_scalars(true, 0.9).await.unwrap();
        assert_eq!(forced.dropped_rows, 1);
        assert_eq!(col.scalars.lock().unwrap().row_count(), 1);
    }

    #[test]
    fn applied_lsn_is_monotonic() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (col, _) = mgr.get_or_create_flagged("c3", Some(4)).unwrap();
        assert_eq!(col.applied_lsn(), 0);
        col.note_applied_lsn(5);
        col.note_applied_lsn(3);
        col.note_applied_lsn(7);
        assert_eq!(col.applied_lsn(), 7);
    }

    #[tokio::test]
    async fn begin_write_rejects_writer_after_drop() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (col, _) = mgr.get_or_create_flagged("dropped", Some(4)).unwrap();
        assert!(!col.is_deleted());
        assert!(col.begin_write().await.is_ok());

        col.mark_deleted();
        assert!(col.is_deleted());
        // 拿锁后双检：即使 Arc 已被取出，删除后也不得再写。
        // EN: Post-lock re-check: even with a pre-fetched Arc, no writes after drop.
        let err = col.begin_write().await.unwrap_err();
        assert_eq!(err, CollectionDeleted);
        assert!(err.to_string().contains("deleted"));
    }

    #[test]
    fn remove_if_same_never_removes_recreated_collection() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (old, _) = mgr.get_or_create_flagged("same-name", Some(4)).unwrap();
        assert!(mgr.remove_if_same("same-name", &old));
        assert!(mgr.get("same-name").is_none());

        // 同名重建后，旧删除流程的迟到摘除不得误删新对象。
        // EN: After a same-name recreation, a delayed removal from the old
        // delete flow must not remove the new object.
        let (new, created) = mgr.get_or_create_flagged("same-name", Some(4)).unwrap();
        assert!(created);
        assert!(!mgr.remove_if_same("same-name", &old));
        let current = mgr.get("same-name").expect("recreated collection stays");
        assert!(Arc::ptr_eq(&current, &new));
        assert!(!new.is_deleted());
    }

    /// P4：稀疏模式仅允许 hnsw/flat；none 在任何后端都合法。
    /// EN: P4 — bm25 is only valid on hnsw/flat; none is valid on every backend.
    #[test]
    fn sparse_bm25_is_only_valid_for_hnsw_and_flat() {
        for ok in ["hnsw", "flat"] {
            CollectionManager::validate_create(
                ok,
                4,
                DistanceMetric::Cosine,
                None,
                VectorDtype::F32,
                Quantization::None,
                SparseMode::Bm25,
            )
            .expect("bm25 must be accepted on hnsw|flat");
        }
        // 每个后端用其原生度量，让失败面落在稀疏校验本身而非度量校验。
        // EN: Give each backend a metric it natively supports so the failure
        // lands on the sparse rule rather than the metric check.
        for (bad, metric) in [
            ("clustered", DistanceMetric::L2),
            ("flat-clustered", DistanceMetric::L2),
            ("pq-flat", DistanceMetric::L2),
            ("hii-l0", DistanceMetric::Cosine),
        ] {
            let err = CollectionManager::validate_create(
                bad,
                4,
                metric,
                None,
                VectorDtype::F32,
                Quantization::None,
                SparseMode::Bm25,
            )
            .expect_err("bm25 must be rejected on non hnsw|flat backends");
            assert!(err.contains("sparse bm25"), "unexpected error: {err}");
        }
        CollectionManager::validate_create(
            "clustered",
            4,
            DistanceMetric::Cosine,
            None,
            VectorDtype::F32,
            Quantization::None,
            SparseMode::None,
        )
        .expect("none remains valid anywhere");
    }

    /// P4：稀疏模式是不可变事实——严格（非重放）路径的显式分叉双向拒绝。
    /// EN: P4 — the sparse mode is immutable; explicit divergence on the strict
    /// (non-replay) path is rejected in both directions.
    #[test]
    fn strict_create_conflicts_on_sparse_mode_divergence() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (col, created) = mgr
            .get_or_create_typed_with_metric_and_quantization(
                "bm25",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                None,
                None,
                None,
                Some(SparseMode::Bm25),
            )
            .unwrap();
        assert!(created);
        assert_eq!(col.sparse_mode(), SparseMode::Bm25);
        let err = expect_create_err(
            mgr.get_or_create_typed_with_metric_and_quantization(
                "bm25",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                None,
                None,
                None,
                Some(SparseMode::None),
            ),
            "recreate as none must be rejected",
        );
        assert!(
            err.contains("sparse=bm25") && err.contains("requires none"),
            "unexpected error: {err}"
        );

        let (plain, _) = mgr
            .get_or_create_typed_with_metric_and_quantization(
                "plain",
                4,
                "flat",
                Some(DistanceMetric::Cosine),
                None,
                None,
                None,
                Some(SparseMode::None),
            )
            .unwrap();
        assert!(plain.sparse_index().is_none());
        let err = expect_create_err(
            mgr.get_or_create_typed_with_metric_and_quantization(
                "plain",
                4,
                "flat",
                Some(DistanceMetric::Cosine),
                None,
                None,
                None,
                Some(SparseMode::Bm25),
            ),
            "recreate as bm25 must be rejected",
        );
        assert!(
            err.contains("sparse=none") && err.contains("requires bm25"),
            "unexpected error: {err}"
        );
    }

    /// P4：仅重放入口允许 none → bm25 的冷启动对账；反向永远拒绝。
    /// EN: P4 — only the replay entry reconciles none -> bm25 at cold start;
    /// the reverse is always rejected.
    #[test]
    fn replay_entry_reconciles_none_to_bm25_only() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (col, _) = mgr
            .get_or_create_typed_with_metric_and_quantization(
                "legacy",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                None,
                None,
                None,
                Some(SparseMode::None),
            )
            .unwrap();
        assert!(col.sparse_index().is_none());
        let (same, created) = mgr
            .get_or_create_replayed_with_quantization(
                "legacy",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                None,
                None,
                None,
                Some(SparseMode::Bm25),
            )
            .unwrap();
        assert!(
            !created,
            "reconciliation must reuse the restored collection"
        );
        assert!(Arc::ptr_eq(&same, &col));
        assert_eq!(col.sparse_mode(), SparseMode::Bm25);
        let lock = col
            .sparse_index()
            .expect("reconciled collection owns a sparse index");
        assert_eq!(lock.read().unwrap().stats().doc_count, 0);

        let err = expect_create_err(
            mgr.get_or_create_replayed_with_quantization(
                "legacy",
                4,
                "hnsw",
                Some(DistanceMetric::Cosine),
                None,
                None,
                None,
                Some(SparseMode::None),
            ),
            "reverse reconciliation must be rejected",
        );
        assert!(err.contains("sparse=bm25"), "unexpected error: {err}");
    }

    /// P4：隐式创建（首次 upsert 落地）恒为 none，不得凭空建稀疏索引。
    /// EN: P4 — implicit creation (first upsert materialization) is always none
    /// and must never allocate a sparse index.
    #[test]
    fn implicit_creation_is_always_sparse_none() {
        let mgr = CollectionManager::new(4, "hnsw");
        let col = mgr.get_or_create("implicit", Some(4)).unwrap();
        assert_eq!(col.sparse_mode(), SparseMode::None);
        assert!(col.sparse_index().is_none());
    }

    /// P4 OCR 9fda139：快照恢复的集合（`create_logged=false` 但不是 upsert
    /// 隐式建立）不参与 none → bm25 对账；后续稀疏 upsert 属腐坏记录，必须
    /// 拒绝而不是静默升级。
    /// EN: P4 OCR 9fda139 — a snapshot-restored collection (`create_logged ==
    /// false` but not implicitly built by upsert replay) never reconciles
    /// none -> bm25; a later sparse upsert is corruption and must be rejected
    /// instead of silently upgrading.
    #[test]
    fn snapshot_restored_collection_rejects_sparse_upsert_reconciliation() {
        let mgr = CollectionManager::new(4, "hnsw");
        let (restored, created) = mgr
            .get_or_create_typed_with_metric_and_quantization(
                "restored",
                4,
                "hnsw",
                None,
                None,
                None,
                None,
                Some(SparseMode::None),
            )
            .unwrap();
        assert!(created);
        // 快照恢复不会 mark_create_logged，但它也绝不是 upsert 隐式建立。
        // EN: Snapshot restore never marks the create logged, and it is also not
        // an upsert implicit build.
        assert!(!restored.create_logged());
        assert!(!restored.implicit_upsert_built());
        let err = expect_create_err(
            mgr.get_or_create_for_collection_upsert("restored", Some(4), Some(SparseMode::Bm25)),
            "sparse upsert after snapshot restore must be rejected",
        );
        assert!(
            err.contains("sparse=none") && err.contains("requires bm25"),
            "unexpected error: {err}"
        );
        assert_eq!(restored.sparse_mode(), SparseMode::None);
    }

    /// P4 OCR 9fda139：允许的升级同样受「bm25 仅限 hnsw|flat」创建约束——
    /// 隐式建立集合采用的服务端默认后端（此处 clustered）上不得升级。
    /// EN: P4 OCR 9fda139 — the allowed upgrade is still bound by the
    /// "bm25 only on hnsw|flat" creation rule: an implicitly built collection
    /// using the server default backend (clustered here) must not upgrade.
    #[test]
    fn sparse_reconcile_rejects_non_hnsw_flat_backend() {
        let mgr = CollectionManager::new(4, "clustered");
        mgr.get_or_create_for_collection_upsert("implicit", Some(4), None)
            .unwrap();
        let err = expect_create_err(
            mgr.get_or_create_for_collection_upsert("implicit", Some(4), Some(SparseMode::Bm25)),
            "bm25 upgrade on clustered must be rejected",
        );
        assert!(
            err.contains("hnsw") && err.contains("flat"),
            "unexpected error: {err}"
        );
        let col = mgr.get("implicit").expect("collection stays");
        assert_eq!(col.sparse_mode(), SparseMode::None);
    }
}
