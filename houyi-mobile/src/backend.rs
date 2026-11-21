//! 移动端索引后端：单索引句柄与多 Collection 共用同一状态机。
//! Mobile index backend shared by the single-index handle and multi-collection DB.
//!
//! 建图后的惰性后端只有两种合法状态：`Pending(store)`（尚未建图）或
//! `Built(index)`（索引独占向量）。用枚举编码这个不变量，避免两个 `Option`
//! 允许「两者同时存在/同时缺失」的非法状态，也避免跨 FFI 的 `expect` panic。
//! EN: A lazy backend has exactly two legal states after construction:
//! `Pending(store)` (not built yet) or `Built(index)` (the index owns the
//! vectors). The enum encodes the invariant so two `Option`s cannot represent
//! impossible states (both present or both absent) and no FFI-facing `expect`
//! can panic.

use crate::storage::VectorStore;
use houyi_vector::cluster::{
    ClusterConfig, ClusteredHnsw, FlatClustered, FlatStorageDtype, DEFAULT_SEARCH_TARGET_PER_K,
};
use houyi_vector::distance;
use houyi_vector::DistanceMetric;

/// 错误前缀：FFI 用前缀而不是完整文案映射 -1/-2，改写说明文字不会改变返回码。
/// EN: Error prefixes let the FFI map -1/-2 by prefix instead of full prose, so
/// rewording a message cannot silently change the return code.
pub(crate) const ERR_INVALID: &str = "invalid: ";
pub(crate) const ERR_REJECTED: &str = "rejected: ";

/// 惰性构图后端的内部状态；`known_ids` 只对 clustered-hnsw 维护（其写入
/// 无 upsert 语义，必须拒绝重复 id）。
/// EN: Internal state of a lazily built backend. `known_ids` is maintained only
/// for clustered-hnsw, whose append-only insert has no upsert semantics and
/// therefore must reject duplicate ids.
enum LazyIndex<I> {
    Pending(VectorStore),
    Built {
        index: I,
        known_ids: Option<ahash::AHashSet<u64>>,
    },
}

/// 移动端索引后端（flat / flat-clustered / clustered-hnsw）。
/// EN: Mobile index backend (flat / flat-clustered / clustered-hnsw).
pub(crate) struct Backend {
    dims: usize,
    metric: DistanceMetric,
    dtype: FlatStorageDtype,
    /// flat-clustered 余弦检索的候选预算（每 k）。默认 2500（k=10 时 25k 候选，
    /// 与旧行为一致）；移动端可按场景调低换取延迟。
    /// EN: Candidate budget per k for flat-clustered cosine search. Default
    /// 2500 (25k candidates at k=10, matching old behavior); mobile may lower
    /// it to trade recall for latency.
    search_target_per_k: usize,
    /// 下一次 build 使用的 nprobe（None = 自适应公式）。
    /// EN: nprobe for the next build (None = adaptive formula).
    build_nprobe: Option<usize>,
    /// 运行期 nprobe / ef_search 设置（建图后立即生效，待建图则建图时应用）。
    /// EN: Runtime nprobe / ef_search settings (applied immediately when built,
    /// or at build time when still pending).
    search_nprobe: NprobeSetting,
    search_ef_search: Option<usize>,
    kind: BackendKind,
}

/// nprobe 设置：绝对值或按簇数推导，避免在未建图时猜簇数量。
/// EN: nprobe setting: absolute or derived from the cluster count, so nothing
/// has to guess the cluster count before a build.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NprobeSetting {
    /// 沿用构建期值。
    /// EN: Keep the build-time value.
    Default,
    /// 显式簇数（夹取到 1..=num_clusters）。
    /// EN: Explicit cluster count (clamped to 1..=num_clusters).
    Absolute(usize),
    /// 全部簇（高召回）。
    /// EN: All clusters (high recall).
    All,
    /// `max(4, num_clusters / divisor)`（快速档）。
    /// EN: `max(4, num_clusters / divisor)` (fast preset).
    Div(usize),
}

/// 移动端召回/延迟预设。
/// EN: Mobile recall/latency presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchPreset {
    /// 低延迟：per_k=500，少量簇。
    /// EN: Low latency: per_k=500, few clusters.
    Fast,
    /// 默认平衡档：沿用默认预算与构建期 nprobe。
    /// EN: Default balanced preset: built-in budget and build-time nprobe.
    Balanced,
    /// 高召回：per_k=5000，全簇路由；clustered-hnsw 提高 ef_search。
    /// EN: High recall: per_k=5000, all clusters; clustered-hnsw raises
    /// ef_search.
    HighRecall,
}

impl SearchPreset {
    /// 从 C/JNI 的整数码解析：0=fast、1=balanced、2=high_recall。
    /// EN: Parse the C/JNI integer code: 0=fast, 1=balanced, 2=high_recall.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Fast),
            1 => Some(Self::Balanced),
            2 => Some(Self::HighRecall),
            _ => None,
        }
    }

    fn per_k(self) -> usize {
        match self {
            Self::Fast => 500,
            Self::Balanced => DEFAULT_SEARCH_TARGET_PER_K,
            Self::HighRecall => 5000,
        }
    }

    fn nprobe(self) -> NprobeSetting {
        match self {
            Self::Fast => NprobeSetting::Div(16),
            Self::Balanced => NprobeSetting::Default,
            Self::HighRecall => NprobeSetting::All,
        }
    }

    fn ef_search(self) -> Option<usize> {
        match self {
            Self::Fast => Some(64),
            Self::Balanced => None,
            Self::HighRecall => Some(512),
        }
    }
}

enum BackendKind {
    Flat(VectorStore),
    FlatClustered(LazyIndex<FlatClustered>),
    ClusteredHnsw(LazyIndex<ClusteredHnsw>),
}

/// 单个集合的持久化视图：flat-clustered 构建态保存编码后的索引状态，
/// 其余后端保存原始行（加载时重建）。
/// EN: Persistence view of one collection: a built flat-clustered saves its
/// encoded index state; other backends save raw rows (rebuilt on load).
pub(crate) enum BackendSnapshot {
    Flat(Vec<(u64, Vec<f32>)>),
    FlatClusteredPending(Vec<(u64, Vec<f32>)>),
    FlatClusteredBuilt(Vec<u8>),
    ClusteredHnsw {
        rows: Vec<(u64, Vec<f32>)>,
        built: bool,
    },
    /// clustered-hnsw 构建态（质心 + 每簇图），加载不重建。
    /// EN: Clustered-hnsw built state (centroids + per-cluster graphs); loading
    /// does not rebuild.
    ClusteredHnswBuilt(Vec<u8>),
}

impl Backend {
    /// 旧构造：保持历史默认（flat=L2/f32、flat-clustered=Cosine/f16、
    /// clustered-hnsw=Cosine/f32）。
    /// EN: Legacy constructor preserving historical defaults (flat=L2/f32,
    /// flat-clustered=Cosine/f16, clustered-hnsw=Cosine/f32).
    pub(crate) fn new(dims: usize, index_type: &str) -> Result<Self, String> {
        // 未知后端沿用历史降级（clustered-hnsw），其余走统一旧默认表。
        // EN: Unknown backends keep the historical clustered-hnsw fallback;
        // everything else uses the single legacy-default table.
        let index_type = match index_type {
            "flat" | "flat-clustered" | "clustered-hnsw" => index_type,
            _ => "clustered-hnsw",
        };
        let (metric, dtype) = Self::legacy_config(index_type);
        Self::new_with_config(dims, index_type, metric, dtype)
    }

    /// 无分配的配置校验（避免为校验构造 VectorStore 的大块容量）。
    /// EN: Allocation-free config validation (avoid constructing a VectorStore
    /// with large preallocated capacity just to validate).
    pub(crate) fn validate_config(
        dims: usize,
        index_type: &str,
        metric: DistanceMetric,
        dtype: FlatStorageDtype,
    ) -> Result<(), String> {
        if dims == 0 || dims > 65_536 {
            return Err("invalid dims".to_string());
        }
        match index_type {
            "flat" => {
                if dtype != FlatStorageDtype::F32 {
                    return Err("flat supports only f32 storage".to_string());
                }
            }
            "flat-clustered" => {}
            "clustered-hnsw" => {
                if dtype != FlatStorageDtype::F32 {
                    return Err("clustered-hnsw supports only f32 storage".to_string());
                }
            }
            other => return Err(format!("invalid index type {other:?}")),
        }
        let _ = metric;
        Ok(())
    }

    /// 显式配置构造；dtype=f16 目前只允许 flat-clustered（HNSW 图内 dtype
    /// 由 ClusterConfig 未接线，flat 存储固定 f32）。
    /// EN: Explicit config constructor. f16 is currently limited to
    /// flat-clustered (the clustered HNSW config has no dtype plumbing yet and
    /// flat storage is f32-only).
    pub(crate) fn new_with_config(
        dims: usize,
        index_type: &str,
        metric: DistanceMetric,
        dtype: FlatStorageDtype,
    ) -> Result<Self, String> {
        Self::validate_config(dims, index_type, metric, dtype)?;
        let kind = match index_type {
            "flat" => {
                if dtype != FlatStorageDtype::F32 {
                    return Err("flat supports only f32 storage".to_string());
                }
                BackendKind::Flat(VectorStore::new(dims, 1024))
            }
            "flat-clustered" => {
                BackendKind::FlatClustered(LazyIndex::Pending(VectorStore::new(dims, 1024)))
            }
            "clustered-hnsw" => {
                if dtype != FlatStorageDtype::F32 {
                    return Err("clustered-hnsw supports only f32 storage".to_string());
                }
                BackendKind::ClusteredHnsw(LazyIndex::Pending(VectorStore::new(dims, 1024)))
            }
            other => return Err(format!("invalid index type {other:?}")),
        };
        Ok(Backend {
            dims,
            metric,
            dtype,
            search_target_per_k: DEFAULT_SEARCH_TARGET_PER_K,
            build_nprobe: None,
            search_nprobe: NprobeSetting::Default,
            search_ef_search: None,
            kind,
        })
    }

    /// 历史默认配置：flat=L2/f32、flat-clustered=Cosine/f16、
    /// clustered-hnsw=Cosine/f32；未知类型沿用旧行为（Cosine/f32）。
    /// EN: Legacy defaults (flat=L2/f32, flat-clustered=Cosine/f16,
    /// clustered-hnsw=Cosine/f32); unknown types keep the old Cosine/f32 fallback.
    pub(crate) fn legacy_config(index_type: &str) -> (DistanceMetric, FlatStorageDtype) {
        match index_type {
            "flat" => (DistanceMetric::L2, FlatStorageDtype::F32),
            "flat-clustered" => (DistanceMetric::Cosine, FlatStorageDtype::F16),
            _ => (DistanceMetric::Cosine, FlatStorageDtype::F32),
        }
    }

    pub(crate) fn set_search_target(&mut self, per_k: usize) -> Result<(), String> {
        if per_k == 0 || per_k > 100_000 {
            return Err(format!(
                "{ERR_INVALID}search target per k must be in 1..=100000"
            ));
        }
        self.search_target_per_k = per_k;
        Ok(())
    }

    /// 设置检索选项（None = 该项保持不变）。
    ///
    /// - `per_k`：候选预算（1..=100000）。
    /// - `nprobe`：路由簇数（1..=100000，建图后被夹取到簇数）。
    /// - `ef_search`：仅 clustered-hnsw 生效（1..=4096），逐簇写入图实例。
    ///
    /// EN: Set search options (None leaves a field unchanged).
    /// - `per_k`: candidate budget (1..=100000).
    /// - `nprobe`: routed cluster count (clamped to the cluster count after a
    ///   build).
    /// - `ef_search`: clustered-hnsw only (1..=4096), written into every
    ///   cluster graph.
    pub(crate) fn set_search_options(
        &mut self,
        per_k: Option<usize>,
        nprobe: Option<usize>,
        ef_search: Option<usize>,
    ) -> Result<(), String> {
        if let Some(per_k) = per_k {
            self.set_search_target(per_k)?;
        }
        if let Some(nprobe) = nprobe {
            if nprobe == 0 || nprobe > 100_000 {
                return Err(format!("{ERR_INVALID}nprobe must be in 1..=100000"));
            }
            self.search_nprobe = NprobeSetting::Absolute(nprobe);
        }
        if let Some(ef_search) = ef_search {
            if ef_search == 0 || ef_search > 4096 {
                return Err(format!("{ERR_INVALID}ef_search must be in 1..=4096"));
            }
            self.search_ef_search = Some(ef_search);
        }
        self.apply_search_settings();
        Ok(())
    }

    /// 应用预设（fast / balanced / high_recall）。
    /// EN: Apply a preset (fast / balanced / high_recall).
    pub(crate) fn set_search_preset(&mut self, preset: SearchPreset) -> Result<(), String> {
        self.search_target_per_k = preset.per_k();
        self.search_nprobe = preset.nprobe();
        self.search_ef_search = preset.ef_search();
        self.apply_search_settings();
        Ok(())
    }

    /// 下一次 build 使用的 nprobe；0 = 恢复自适应公式。已建图需重建才生效
    /// （运行期可用 `set_search_options` 的 nprobe 直接抬高路由簇数）。
    /// EN: nprobe for the next build; 0 restores the adaptive formula. A built
    /// index only picks it up on rebuild (use `set_search_options` nprobe to
    /// raise routing immediately).
    pub(crate) fn set_build_nprobe(&mut self, nprobe: usize) -> Result<(), String> {
        if nprobe > 100_000 {
            return Err(format!("{ERR_INVALID}nprobe must be <= 100000"));
        }
        self.build_nprobe = if nprobe == 0 { None } else { Some(nprobe) };
        Ok(())
    }

    /// 把已保存的运行期设置应用到已构建的索引（未建图时留待 build 处理）。
    /// EN: Apply saved runtime settings to a built index (pending indexes pick
    /// them up at build time).
    fn apply_search_settings(&mut self) {
        let nprobe = self.search_nprobe;
        let ef = self.search_ef_search;
        match &mut self.kind {
            BackendKind::FlatClustered(LazyIndex::Built { index, .. }) => {
                index.set_search_nprobe(Self::resolve_nprobe(nprobe, index.num_clusters()));
            }
            BackendKind::ClusteredHnsw(LazyIndex::Built { index, .. }) => {
                index.set_search_nprobe(Self::resolve_nprobe(nprobe, index.num_clusters()));
                if let Some(ef) = ef {
                    index.set_search_ef_search(ef);
                }
            }
            _ => {}
        }
    }

    /// 把 nprobe 设置解析成绝对值：Default=0、All=全部、Div=按簇数推导。
    /// EN: Resolve an nprobe setting: Default=0, All=every cluster, Div=derived
    /// from the cluster count.
    fn resolve_nprobe(setting: NprobeSetting, num_clusters: usize) -> usize {
        match setting {
            NprobeSetting::Default => 0,
            NprobeSetting::Absolute(value) => value.min(num_clusters),
            NprobeSetting::All => num_clusters,
            NprobeSetting::Div(divisor) => (num_clusters / divisor.max(1)).max(4),
        }
    }

    /// 插入/覆盖；建图后的 flat-clustered 立即可见。
    ///
    /// `clustered-hnsw` 建图后仅接受**新 id**（追加即查）；同 id 覆盖返回
    /// Err——该后端的 insert 不返回拒绝信号，静默重复会留下陈旧副本并抬高
    /// 计数。需要 upsert 时重建索引后批量写入。
    /// EN: Insert or overwrite; a built flat-clustered row is immediately
    /// searchable.
    ///
    /// A built clustered-hnsw accepts **new ids only** (append and searchable).
    /// Overwriting an existing id returns Err: its insert reports no rejection,
    /// so a silent duplicate would leave a stale copy and inflate the count.
    /// Recreate the index and batch-write when an upsert is required.
    pub(crate) fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dims {
            return Err(format!(
                "{}dim mismatch: expected {}, got {}",
                ERR_INVALID,
                self.dims,
                vector.len()
            ));
        }
        match &mut self.kind {
            BackendKind::Flat(store) => store.try_upsert(id, vector),
            BackendKind::FlatClustered(state) => match state {
                LazyIndex::Pending(store) => store.try_upsert(id, vector),
                LazyIndex::Built { index, .. } => index
                    .insert(id, vector)
                    .map_err(|message| format!("{ERR_REJECTED}{message}")),
            },
            BackendKind::ClusteredHnsw(state) => match state {
                LazyIndex::Pending(store) => store.try_upsert(id, vector),
                LazyIndex::Built { index, known_ids } => {
                    let Some(known) = known_ids.as_mut() else {
                        return Err(format!(
                            "{ERR_REJECTED}clustered-hnsw id set missing; recreate the index"
                        ));
                    };
                    if known.contains(&id) {
                        return Err(format!(
                            "{ERR_REJECTED}clustered-hnsw cannot overwrite an existing id after build; recreate the index to upsert"
                        ));
                    }
                    index.insert(id, vector);
                    known.insert(id);
                    Ok(())
                }
            },
        }
    }

    /// 构建/重聚类；返回索引原子数。已建索引时从索引导出重建。
    /// EN: Build or recluster; returns the indexed count. A built index is
    /// exported and rebuilt from its current rows.
    pub(crate) fn build(&mut self) -> usize {
        // 建图前先取出设置：match 借用了 self.kind，之后无法再借 self。
        // EN: Snapshot settings before the match borrows self.kind.
        let build_nprobe = self.build_nprobe;
        let runtime_nprobe = self.search_nprobe;
        let runtime_ef = self.search_ef_search;
        match &mut self.kind {
            BackendKind::Flat(store) => store.len(),
            BackendKind::FlatClustered(state) => match state {
                LazyIndex::Pending(store) => {
                    let n = store.len();
                    if n < 2 {
                        return n;
                    }
                    let (ids, vecs) = store.take_for_build();
                    let c = (n as f64).sqrt().ceil() as usize;
                    let fc = FlatClustered::build_with_storage_dtype_and_nprobe(
                        &vecs,
                        &ids,
                        self.dims,
                        self.metric,
                        c,
                        10,
                        0,
                        self.dtype,
                        build_nprobe,
                    );
                    fc.set_search_nprobe(Backend::resolve_nprobe(
                        runtime_nprobe,
                        fc.num_clusters(),
                    ));
                    let count = fc.total_vectors();
                    *state = LazyIndex::Built {
                        index: fc,
                        known_ids: None,
                    };
                    count
                }
                LazyIndex::Built {
                    index,
                    known_ids: _,
                } => {
                    let n = index.total_vectors();
                    let (ids, vecs) = flatten_export(index.export_vectors(), self.dims);
                    let c = (n as f64).sqrt().ceil() as usize;
                    let fc = FlatClustered::build_with_storage_dtype_and_nprobe(
                        &vecs,
                        &ids,
                        self.dims,
                        self.metric,
                        c,
                        10,
                        0,
                        self.dtype,
                        build_nprobe,
                    );
                    fc.set_search_nprobe(Backend::resolve_nprobe(
                        runtime_nprobe,
                        fc.num_clusters(),
                    ));
                    let count = fc.total_vectors();
                    *state = LazyIndex::Built {
                        index: fc,
                        known_ids: None,
                    };
                    count
                }
            },
            BackendKind::ClusteredHnsw(state) => match state {
                LazyIndex::Pending(store) => {
                    let n = store.len();
                    if n < 2 {
                        return n;
                    }
                    let (ids, vecs) = store.take_for_build();
                    let known_ids: ahash::AHashSet<u64> = ids.iter().copied().collect();
                    let mut cfg = cluster_config(n, self.dims);
                    if let Some(nprobe) = build_nprobe {
                        cfg.nprobe = nprobe.clamp(1, cfg.num_clusters);
                    }
                    let ch = ClusteredHnsw::build(&vecs, &ids, self.dims, self.metric, &cfg);
                    ch.set_search_nprobe(Backend::resolve_nprobe(
                        runtime_nprobe,
                        ch.num_clusters(),
                    ));
                    if let Some(ef) = runtime_ef {
                        ch.set_search_ef_search(ef);
                    }
                    let count = ch.total_vectors();
                    *state = LazyIndex::Built {
                        index: ch,
                        known_ids: Some(known_ids),
                    };
                    count
                }
                LazyIndex::Built {
                    index,
                    known_ids: _,
                } => {
                    let n = index.total_vectors();
                    let (ids, vecs) = flatten_export(index.export_vectors(), self.dims);
                    let known_ids: ahash::AHashSet<u64> = ids.iter().copied().collect();
                    let mut cfg = cluster_config(n, self.dims);
                    if let Some(nprobe) = build_nprobe {
                        cfg.nprobe = nprobe.clamp(1, cfg.num_clusters);
                    }
                    let ch = ClusteredHnsw::build(&vecs, &ids, self.dims, self.metric, &cfg);
                    ch.set_search_nprobe(Backend::resolve_nprobe(
                        runtime_nprobe,
                        ch.num_clusters(),
                    ));
                    if let Some(ef) = runtime_ef {
                        ch.set_search_ef_search(ef);
                    }
                    let count = ch.total_vectors();
                    *state = LazyIndex::Built {
                        index: ch,
                        known_ids: Some(known_ids),
                    };
                    count
                }
            },
        }
    }

    pub(crate) fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_filtered(query, k, None)
    }

    /// 带允许 id 集合的检索；flat-clustered 在候选扫描阶段过滤，其余后端
    /// 先搜后过滤（结果可能不足 k，与过滤选择性有关）。
    /// EN: Search with an allowed-id set; flat-clustered filters during
    /// candidate scanning, while other backends post-filter (result count may
    /// be below k depending on selectivity).
    pub(crate) fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&ahash::AHashSet<u64>>,
    ) -> Vec<(u64, f32)> {
        // 统一在这里拒绝维度不符：所有后端（含 flat 的逐项 zip）都不得
        // 用截断后的查询产生「看似合理」的错误距离。
        // EN: Reject dimension mismatches here so no backend (including flat's
        // zip-based scan) can produce plausible-looking but wrong distances
        // from a truncated query.
        if query.len() != self.dims {
            return Vec::new();
        }
        match &self.kind {
            BackendKind::Flat(store) => {
                // flat 全量扫描：有过滤时先取全量候选再过滤，避免 k 截断后
                // 过滤把真实命中丢掉。
                // EN: Full scan: with a mask, score the whole store before
                // filtering so the k truncation cannot drop matching rows.
                let Some(allowed) = allowed else {
                    return flat_search(store, query, k, self.dims, self.metric);
                };
                let mut found = flat_search(store, query, store.len(), self.dims, self.metric);
                found.retain(|(id, _)| allowed.contains(id));
                found.truncate(k);
                found
            }
            BackendKind::FlatClustered(state) => match state {
                LazyIndex::Pending(_) => Vec::new(),
                LazyIndex::Built { index, .. } => {
                    index.search_with_target_filtered(query, k, self.search_target_per_k, allowed)
                }
            },
            BackendKind::ClusteredHnsw(state) => match state {
                LazyIndex::Pending(_) => Vec::new(),
                LazyIndex::Built { index, .. } => {
                    let Some(allowed) = allowed else {
                        return index.search(query, k);
                    };
                    // ANN 只能后过滤；先过取再过滤，降低不足 k 的概率。
                    // EN: ANN can only post-filter; over-fetch before filtering
                    // to reduce under-fill.
                    let total = index.total_vectors();
                    let overfetch = k.saturating_mul(4).min(total);
                    let mut found = index.search(query, overfetch);
                    found.retain(|(id, _)| allowed.contains(id));
                    found.truncate(k);
                    found
                }
            },
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.kind {
            BackendKind::Flat(store) => store.len(),
            BackendKind::FlatClustered(state) => match state {
                LazyIndex::Pending(store) => store.len(),
                LazyIndex::Built { index, .. } => index.total_vectors(),
            },
            BackendKind::ClusteredHnsw(state) => match state {
                LazyIndex::Pending(store) => store.len(),
                LazyIndex::Built { index, .. } => index.total_vectors(),
            },
        }
    }
}

impl Backend {
    pub(crate) fn snapshot(&self) -> Result<BackendSnapshot, String> {
        Ok(match &self.kind {
            BackendKind::Flat(store) => BackendSnapshot::Flat(store.rows()),
            BackendKind::FlatClustered(LazyIndex::Pending(store)) => {
                BackendSnapshot::FlatClusteredPending(store.rows())
            }
            BackendKind::FlatClustered(LazyIndex::Built { index, .. }) => {
                BackendSnapshot::FlatClusteredBuilt(index.encode_state()?)
            }
            BackendKind::ClusteredHnsw(LazyIndex::Pending(store)) => {
                BackendSnapshot::ClusteredHnsw {
                    rows: store.rows(),
                    built: false,
                }
            }
            BackendKind::ClusteredHnsw(LazyIndex::Built { index, .. }) => {
                BackendSnapshot::ClusteredHnswBuilt(index.encode_state())
            }
        })
    }

    /// 从持久化视图重建后端；维度/索引状态不一致返回 Err（不 panic）。
    /// EN: Restore a backend from its persistence view; inconsistent
    /// dimensions/index state return Err instead of panicking.
    pub(crate) fn restore(
        snapshot: BackendSnapshot,
        dims: usize,
        index_type: &str,
        metric: DistanceMetric,
        dtype: FlatStorageDtype,
    ) -> Result<Self, String> {
        if dims == 0 || dims > 65_536 {
            return Err(format!("snapshot: invalid dims {dims}"));
        }
        let expected_type = match &snapshot {
            BackendSnapshot::Flat(_) => "flat",
            BackendSnapshot::FlatClusteredPending(_) | BackendSnapshot::FlatClusteredBuilt(_) => {
                "flat-clustered"
            }
            BackendSnapshot::ClusteredHnswBuilt(_) => "clustered-hnsw",
            BackendSnapshot::ClusteredHnsw { .. } => "clustered-hnsw",
        };
        if index_type != expected_type {
            return Err(format!(
                "snapshot backend tag '{expected_type}' != collection index type '{index_type}'"
            ));
        }
        match snapshot {
            BackendSnapshot::Flat(rows) => {
                let mut backend = Backend::new_with_config(dims, "flat", metric, dtype)?;
                insert_rows(&mut backend, &rows, dims)?;
                Ok(backend)
            }
            BackendSnapshot::FlatClusteredPending(rows) => {
                let mut backend = Backend::new_with_config(dims, "flat-clustered", metric, dtype)?;
                insert_rows(&mut backend, &rows, dims)?;
                Ok(backend)
            }
            BackendSnapshot::FlatClusteredBuilt(state) => {
                let index = FlatClustered::decode_state(&state)?;
                if index.state_dims() != dims {
                    return Err(format!(
                        "flat-clustered state dims {} != collection dims {dims}",
                        index.state_dims()
                    ));
                }
                if index.vector_metric() != metric || index.storage_dtype() != dtype {
                    return Err(
                        "flat-clustered state metric/dtype mismatch with collection config"
                            .to_string(),
                    );
                }
                Ok(Backend {
                    dims,
                    metric,
                    dtype,
                    search_target_per_k: DEFAULT_SEARCH_TARGET_PER_K,
                    build_nprobe: None,
                    search_nprobe: NprobeSetting::Default,
                    search_ef_search: None,
                    kind: BackendKind::FlatClustered(LazyIndex::Built {
                        index,
                        known_ids: None,
                    }),
                })
            }
            BackendSnapshot::ClusteredHnswBuilt(state) => {
                let index = ClusteredHnsw::decode_state(&state)?;
                if index.dims() != dims || index.metric() != metric {
                    return Err(
                        "clustered-hnsw state dims/metric mismatch with collection config"
                            .to_string(),
                    );
                }
                Ok(Backend {
                    dims,
                    metric,
                    dtype,
                    search_target_per_k: DEFAULT_SEARCH_TARGET_PER_K,
                    build_nprobe: None,
                    search_nprobe: NprobeSetting::Default,
                    search_ef_search: None,
                    kind: BackendKind::ClusteredHnsw(LazyIndex::Built {
                        index,
                        known_ids: None,
                    }),
                })
            }
            BackendSnapshot::ClusteredHnsw { rows, built } => {
                let mut backend = Backend::new_with_config(dims, "clustered-hnsw", metric, dtype)?;
                insert_rows(&mut backend, &rows, dims)?;
                if built {
                    backend.build();
                }
                Ok(backend)
            }
        }
    }
}

fn insert_rows(backend: &mut Backend, rows: &[(u64, Vec<f32>)], dims: usize) -> Result<(), String> {
    for (id, vector) in rows {
        if vector.len() != dims {
            return Err(format!(
                "snapshot row {id}: dim mismatch expected {dims}, got {}",
                vector.len()
            ));
        }
        backend.insert(*id, vector)?;
    }
    Ok(())
}

fn cluster_config(n: usize, dims: usize) -> ClusterConfig {
    ClusterConfig {
        hnsw_ef_construction: 200,
        hnsw_ef_search: 128,
        hnsw_m: 12, // 移动端用小 M 节约内存
        ..ClusterConfig::infer(n, dims)
    }
}

/// 把逐行导出的向量摊平成 `(ids, flat_values)`；预分配避免逐行扩容。
/// EN: Flatten exported rows into `(ids, flat_values)` with preallocation.
fn flatten_export(rows: Vec<(u64, Vec<f32>)>, dims: usize) -> (Vec<u64>, Vec<f32>) {
    let mut ids = Vec::with_capacity(rows.len());
    let mut flat = Vec::with_capacity(rows.len() * dims);
    for (id, vector) in rows {
        ids.push(id);
        flat.extend_from_slice(&vector);
    }
    (ids, flat)
}

fn distance_metric(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => distance::l2_squared(a, b),
        DistanceMetric::Cosine => distance::cosine(a, b),
        DistanceMetric::InnerProduct => distance::inner_product_distance(a, b),
        DistanceMetric::L1 => distance::l1_distance(a, b),
    }
}

fn flat_search(
    store: &VectorStore,
    query: &[f32],
    k: usize,
    dims: usize,
    metric: DistanceMetric,
) -> Vec<(u64, f32)> {
    let (ids, vecs) = store.raw_data();
    let n = ids.len();
    if n == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(u64, f32)> = (0..n)
        .map(|i| {
            let v = &vecs[i * dims..(i + 1) * dims];
            (ids[i], distance_metric(metric, query, v))
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}
