//! IndexBackend：HNSW / ClusteredHnsw / FlatClustered 三种索引后端。
//! EN: IndexBackend: three index backends — HNSW / ClusteredHnsw / FlatClustered.

use houyi_persistence::collection_snapshot_file::SnapshotVectorKind;
use houyi_vector::cluster::{ClusterConfig, ClusteredHnsw, FlatClustered, PqFlatClustered};
use houyi_vector::hii::{HiiL0, HiiL0Config};
use houyi_vector::{
    AsyncBuilderOnceReport, BuilderMetricsSnapshot, DistanceMetric, FlushPolicy, HnswBatchPlanMode,
    HotBufferIndex, VectorId, VectorIndex,
};
use std::sync::{Arc, Mutex};

/// 延迟构图后端共享的「已写入未构图」行缓冲（id, 向量）。
/// Pending (not-yet-built) rows shared by lazy-build backends: `(id, vector)`.
///
/// WHY：四个变体各自内联 `Arc<Mutex<Vec<(u64, Vec<f32>)>>>` 会触发
/// clippy::type_complexity 且容易漂移；提取别名让「缓冲行」这一概念只有一处定义。
/// EN: One alias for the buffered-rows shape shared by four variants.
type BufferedRows = Arc<Mutex<Vec<(u64, Vec<f32>)>>>;

#[derive(Clone)]
pub enum IndexBackend {
    Hnsw(Arc<HotBufferIndex>),
    Clustered {
        buf: BufferedRows,
        index: Arc<Mutex<Option<ClusteredHnsw>>>,
        dims: usize,
        built: Arc<Mutex<bool>>,
        /// 构图度量：决定 scored 距离是否为余弦（L2 构图返回 L2²，边缘不得 clamp）。
        /// Build metric: decides whether scored distances are cosine (L2 build → L2²,
        /// the edge must not clamp).
        metric: DistanceMetric,
    },
    FlatClustered {
        buf: BufferedRows,
        index: Arc<Mutex<Option<FlatClustered>>>,
        dims: usize,
        built: Arc<Mutex<bool>>,
        /// 构图度量：同 `Clustered`，纵深防御 L2 构图被误当余弦 clamp。
        /// Build metric — same fail-closed gate as `Clustered`.
        metric: DistanceMetric,
    },
    Flat {
        data: BufferedRows,
        dims: usize,
        /// 计分度量：flat 是暴力扫描，直接按该度量计算距离（不再固定余弦）。
        /// Scoring metric — flat is brute-force, so distances are computed
        /// directly under this metric instead of a fixed cosine.
        metric: DistanceMetric,
    },
    PqFlat {
        buf: BufferedRows,
        index: Arc<Mutex<Option<PqFlatClustered>>>,
        dims: usize,
        built: Arc<Mutex<bool>>,
    },
    /// HII L0: 内存增量层 — 实时写入可见，小图 ANN 检索。
    /// HII L0: memory incremental layer — real-time writes, small graph ANN search.
    /// L1/L2 TODO (待 GPU/NVMe 硬件就绪).
    HiiL0 {
        inner: Arc<Mutex<HiiL0>>,
        dims: usize,
    },
}

impl IndexBackend {
    /// 旧签名兼容入口：默认 Cosine（历史上服务端所有后端的实际度量）。
    /// EN: Legacy-compatible entry point: Cosine (the historical effective metric).
    pub fn new_hnsw(dims: usize) -> Self {
        Self::new_hnsw_with_metric(dims, DistanceMetric::Cosine)
    }
    pub fn new_hnsw_with_metric(dims: usize, metric: DistanceMetric) -> Self {
        IndexBackend::Hnsw(Arc::new(HotBufferIndex::new_with_metric(dims, metric)))
    }
    pub fn new_clustered(dims: usize) -> Self {
        Self::new_clustered_with_metric(dims, DistanceMetric::Cosine)
    }
    pub fn new_clustered_with_metric(dims: usize, metric: DistanceMetric) -> Self {
        IndexBackend::Clustered {
            buf: Arc::new(Mutex::new(Vec::new())),
            index: Arc::new(Mutex::new(None)),
            dims,
            built: Arc::new(Mutex::new(false)),
            metric,
        }
    }
    pub fn new_flat_clustered(dims: usize) -> Self {
        Self::new_flat_clustered_with_metric(dims, DistanceMetric::Cosine)
    }
    pub fn new_flat_clustered_with_metric(dims: usize, metric: DistanceMetric) -> Self {
        IndexBackend::FlatClustered {
            buf: Arc::new(Mutex::new(Vec::new())),
            index: Arc::new(Mutex::new(None)),
            dims,
            built: Arc::new(Mutex::new(false)),
            metric,
        }
    }
    pub fn new_flat(dims: usize) -> Self {
        Self::new_flat_with_metric(dims, DistanceMetric::Cosine)
    }
    pub fn new_flat_with_metric(dims: usize, metric: DistanceMetric) -> Self {
        IndexBackend::Flat {
            data: Arc::new(Mutex::new(Vec::new())),
            dims,
            metric,
        }
    }
    /// pq-flat 只支持 L2：内部构建与查表均为平方欧氏，`metric` 形参仅为保持
    /// 统一构造面；调用方（集合创建/重放）必须先按 [`metric_supported`] 校验。
    /// EN: pq-flat supports L2 only (training/lookup are squared Euclidean); the
    /// `metric` parameter just keeps a uniform surface — callers must validate
    /// via [`metric_supported`] first.
    pub fn new_pq_flat(dims: usize) -> Self {
        Self::new_pq_flat_with_metric(dims, DistanceMetric::L2)
    }
    pub fn new_pq_flat_with_metric(dims: usize, metric: DistanceMetric) -> Self {
        // 构造期自守不变量：报错型调用面（集合创建）已按 [`metric_supported`]
        // 校验，这里再断言一次，防止绕过校验的调用方得到一个「按 L2 计分却
        // 声称别的度量」的后端（OCR #35）。
        // EN: Enforce the invariant at construction: the validated path already
        // checks [`metric_supported`]; asserting here prevents a caller that
        // bypasses validation from getting a backend that scores L2 while claiming
        // another metric (OCR #35).
        assert_eq!(
            metric,
            DistanceMetric::L2,
            "pq-flat only supports L2 (got {metric:?})"
        );
        IndexBackend::PqFlat {
            buf: Arc::new(Mutex::new(Vec::new())),
            index: Arc::new(Mutex::new(None)),
            dims,
            built: Arc::new(Mutex::new(false)),
        }
    }
    /// hii-l0 只支持 Cosine（写入/查询都做 L2 单位化）；非 Cosine 由调用方拒绝。
    /// EN: hii-l0 supports Cosine only (writes/queries are L2-normalized); callers
    /// reject other metrics.
    pub fn new_hii_l0(dims: usize) -> Self {
        Self::new_hii_l0_with_metric(dims, DistanceMetric::Cosine)
    }
    pub fn new_hii_l0_with_metric(dims: usize, metric: DistanceMetric) -> Self {
        // 与 pq-flat 同理：构造期自守「只接受 Cosine」不变量（OCR #35）。
        // EN: Same as pq-flat: enforce the Cosine-only invariant at construction (OCR #35).
        assert_eq!(
            metric,
            DistanceMetric::Cosine,
            "hii-l0 only supports Cosine (got {metric:?})"
        );
        let config = HiiL0Config {
            dims,
            ..Default::default()
        };
        IndexBackend::HiiL0 {
            inner: Arc::new(Mutex::new(HiiL0::new(config))),
            dims,
        }
    }

    /// index_type × metric 支持矩阵（集合创建与 WAL/快照恢复共用）。
    /// pq-flat 仅 L2（PQ 查表恒为平方欧氏）；hii-l0 仅 Cosine（内部单位化）；
    /// 其余后端支持全部四种度量。
    /// EN: index_type × metric support matrix shared by creation and replay.
    /// pq-flat is L2-only (PQ tables are squared Euclidean); hii-l0 is Cosine-only
    /// (it normalizes internally); every other backend supports all four metrics.
    pub fn metric_supported(index_type: &str, metric: DistanceMetric) -> Result<(), String> {
        match index_type {
            "pq-flat" if metric != DistanceMetric::L2 => Err(format!(
                "index_type 'pq-flat' supports metric 'l2' only (requested '{}')",
                metric.as_str()
            )),
            "hii-l0" if metric != DistanceMetric::Cosine => Err(format!(
                "index_type 'hii-l0' supports metric 'cosine' only (requested '{}')",
                metric.as_str()
            )),
            _ => Ok(()),
        }
    }
    /// PqFlat scored 透传：聚类索引自身已返回按距离升序的 (id, dist)；未 build → 空。
    /// PqFlat scored passthrough: the clustered index already yields ascending
    /// (id, dist); unbuilt index → empty.
    ///
    /// 已知限制（TODO/后续 issue）：这里的距离不是余弦距离——无 mmap 时是 PQ 子空间
    /// 平方 L2 查表距离之和，有 mmap 时是 Phase-2 对原始向量重排的平方 L2（L2²）。
    /// HTTP 边缘统一套用 `score = 1 - distance` 后，pq-flat 的分数不保证落在
    /// `[-1, 1]`，且对模长不同的向量排序可能与余弦不同；后续应让 pq-flat 重排
    /// 按 Cosine 度量计算，或在响应中显式透出每后端的分数类型。
    /// EN: Known limitation — pq-flat distances are squared-L2 (PQ table sums without
    /// mmap; exact L2² rerank with mmap), NOT cosine. `score = 1 - distance` is therefore
    /// unbounded and ranking can differ from cosine for vectors of differing magnitude.
    fn search_pq_scored(
        index: &Mutex<Option<PqFlatClustered>>,
        q: &[f32],
        k: usize,
    ) -> Vec<(VectorId, f32)> {
        index
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.search(q, k))
            .unwrap_or_default()
    }

    /// 服务端所有写入路径的分量有限性总闸门（HTTP 两条 upsert、pgwire `upsert_batch`、
    /// 冷启动重放都经 `insert`/`insert_batch` 进入后端）。
    /// EN: The single finiteness gate for every server ingestion path (both HTTP
    /// upserts, pgwire `upsert_batch`, and cold-start WAL/snapshot replay).
    ///
    /// WHY（为什么闸门必须在服务端适配层而不能只靠库层或 HTTP 校验）：库层校验是
    /// 不对称的——[`HiiL0`] 硬拒非有限分量，[`HotBufferIndex`] 只校验维度，而
    /// Clustered/FlatClustered/PqFlat/Flat 的 buf 分支连 `Err` 都不返回（直接 extend）。
    /// 同时 pgwire binary float4/float8 经 `f32::from_bits` 解码，NaN/±Inf 原样通过
    /// 维度校验（OCR 第五轮发现的唯一毒数据入口）。在适配层统一 fail-fast，整批原子
    /// 拒绝、任何行都不部分写入。维度仍由各库层与协议边界各自负责，这里不重复。
    /// 冷启动重放自身先经 `partition_finite_batch` 过滤历史毒行，闸门对其为 no-op。
    /// EN: Library validation is asymmetric (HiiL0 hard-rejects; HotBuffer checks dims
    /// only; the buf-based backends never return Err), and pgwire binary floats decoded
    /// via f32::from_bits sailed through the dim check. This adapter-level gate rejects
    /// atomically. Replay filters poisoned rows beforehand, so the gate is a no-op there.
    fn reject_non_finite<'a>(
        rows: impl Iterator<Item = (VectorId, &'a [f32])>,
    ) -> Result<(), String> {
        for (id, v) in rows {
            if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
                return Err(format!(
                    "non-finite component (NaN/Infinity) in vector id={id}: {bad}"
                ));
            }
        }
        Ok(())
    }

    pub async fn insert(&self, id: VectorId, vector: Vec<f32>) -> Result<(), String> {
        Self::reject_non_finite(std::iter::once((id, vector.as_slice())))?;
        match self {
            Self::Hnsw(idx) => idx.insert(id, vector).await,
            _ => self.insert_batch(vec![(id, vector)]).await,
        }
    }
    pub async fn insert_batch(&self, batch: Vec<(u64, Vec<f32>)>) -> Result<(), String> {
        Self::reject_non_finite(batch.iter().map(|(id, v)| (*id, v.as_slice())))?;
        match self {
            Self::Hnsw(idx) => idx.insert_batch(batch).await,
            Self::HiiL0 { inner, .. } => inner.lock().unwrap().insert_batch(batch),
            Self::Clustered { buf, .. }
            | Self::FlatClustered { buf, .. }
            | Self::PqFlat { buf, .. } => {
                buf.lock().unwrap().extend(batch);
                Ok(())
            }
            Self::Flat { data, .. } => {
                data.lock().unwrap().extend(batch);
                Ok(())
            }
        }
    }

    /// Flat 暴力 scored 检索：先按 mask 过滤，再按变体记录的度量计分排序。
    /// EN: Flat brute-force scored search — apply the allow-list mask first, then
    /// score by the variant's recorded metric.
    ///
    /// 历史背景：旧实现固定余弦，`distance_metric()` 也因此对 Flat 报 Cosine；
    /// 现在 Flat 直接承载请求度量（Cosine/L2/IP/L1），Cosine 分支与 HotBufferIndex
    /// 共用同一全余弦公式（内部按范数归一化），**存储向量保持原样**；零向量按
    /// 余弦约定距离为 1。
    /// EN: Historic note — the old implementation hard-coded cosine. Flat now carries
    /// the requested metric (Cosine/L2/IP/L1); the cosine arm shares the full-cosine
    /// formula (norm-normalized internally) with HotBufferIndex, so stored vectors
    /// are never mutated; zero vectors map to distance 1 under cosine.
    #[allow(clippy::type_complexity)]
    fn flat_search_scored(
        data: &Mutex<Vec<(u64, Vec<f32>)>>,
        metric: DistanceMetric,
        q: &[f32],
        k: usize,
        filter: Option<&[VectorId]>,
    ) -> Vec<(u64, f32)> {
        let guard = data.lock().unwrap();
        if guard.is_empty() {
            return Vec::new();
        }
        // 过滤 mask 每次调用只构建一次 HashSet：flat 是暴力扫描，旧实现直接忽略
        // mask，会让混合检索的结构化过滤静默失效（返回未过滤结果）。
        // EN: Build the allow-list HashSet once per brute-force scan. The old flat
        // path ignored the mask, so structured filters silently returned unfiltered hits.
        let allow: Option<std::collections::HashSet<u64>> =
            filter.map(|ids| ids.iter().copied().collect());
        let mut scored: Vec<(u64, f32)> = guard
            .iter()
            .filter(|(id, _)| allow.as_ref().is_none_or(|set| set.contains(id)))
            .map(|(id, v)| {
                // 逐度量计分；Cosine 内部按范数归一化，存储向量不做任何改写。
                // Per-metric scoring; cosine normalizes by norm inside and never
                // mutates stored vectors.
                let d: f32 = match metric {
                    DistanceMetric::Cosine => houyi_vector::distance::cosine(q, v),
                    DistanceMetric::L2 => houyi_vector::distance::l2_squared(q, v),
                    DistanceMetric::InnerProduct => {
                        houyi_vector::distance::inner_product_distance(q, v)
                    }
                    DistanceMetric::L1 => houyi_vector::distance::l1_distance(q, v),
                };
                (*id, d)
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    pub async fn search(&self, q: &[f32], k: usize, f: Option<&[VectorId]>) -> Vec<VectorId> {
        Self::strip(self.search_scored_with_ef_override(q, k, f, None).await)
    }
    pub async fn search_with_ef(
        &self,
        q: &[f32],
        k: usize,
        f: Option<&[VectorId]>,
        ef: Option<usize>,
    ) -> Vec<VectorId> {
        Self::strip(self.search_scored_with_ef_override(q, k, f, ef).await)
    }
    pub async fn search_with_ef_override(
        &self,
        q: &[f32],
        k: usize,
        f: Option<&[VectorId]>,
        ef: Option<usize>,
    ) -> Vec<VectorId> {
        Self::strip(self.search_scored_with_ef_override(q, k, f, ef).await)
    }

    /// id-only 入口统一剥离距离：所有后端只维护 scored 一条排序链路。
    /// EN: All id-only entry points strip distances here — each backend maintains a
    /// single scored sorting path.
    fn strip(pairs: Vec<(VectorId, f32)>) -> Vec<VectorId> {
        pairs.into_iter().map(|(id, _)| id).collect()
    }

    /// scored 检索：所有后端统一返回 `(id, 距离)`，按距离升序（越小越近）。
    /// 距离语义 = 各变体 [`Self::distance_metric`] 记录的度量：hnsw/clustered/
    /// flat-clustered/flat 支持 Cosine/L2/IP/L1；**pq-flat 恒为平方 L2**（PQ 查表
    /// 距离和 / mmap 精确 L2² 重排）；hii-l0 恒为 Cosine。距离→相似度的换算只在
    /// HTTP 边缘做，按 [`Self::distance_metric`] 返回的度量逐种换算（仅 Cosine 钳制
    /// 到 `[-1,1]`；ip 取反为原始内积）。
    /// EN: Scored search across backends: ascending (id, distance), smaller = closer.
    /// The distance semantics are those recorded by [`Self::distance_metric`]:
    /// hnsw/clustered/flat-clustered/flat support Cosine/L2/IP/L1; pq-flat is always
    /// squared-L2 (PQ table sum / exact L2² rerank); hii-l0 is always cosine. The
    /// distance→similarity conversion happens only at the HTTP edge, per metric
    /// (only Cosine clamps to [-1,1]; ip is negated to the raw inner product).
    #[allow(clippy::type_complexity)]
    pub async fn search_scored_with_ef_override(
        &self,
        q: &[f32],
        k: usize,
        f: Option<&[VectorId]>,
        ef: Option<usize>,
    ) -> Vec<(VectorId, f32)> {
        match self {
            Self::HiiL0 { inner, .. } => {
                let mut l0 = inner.lock().unwrap();
                l0.search_scored(q, k)
            }
            Self::Hnsw(idx) => idx.search_scored_with_ef_override(q, k, f, ef).await,
            Self::Clustered { index, .. } => Self::search_c_scored(index, q, k),
            Self::FlatClustered { index, .. } => Self::search_fc_scored(index, q, k),
            Self::Flat { data, metric, .. } => Self::flat_search_scored(data, *metric, q, k, f),
            Self::PqFlat { index, .. } => Self::search_pq_scored(index, q, k),
        }
    }

    /// Clustered scored 透传：c.search 自身已返回按距离升序的 (id, dist)；未 build → 空。
    /// Clustered scored passthrough: c.search already returns ascending pairs;
    /// an unbuilt index yields an empty vec.
    fn search_c_scored(
        index: &Mutex<Option<ClusteredHnsw>>,
        q: &[f32],
        k: usize,
    ) -> Vec<(VectorId, f32)> {
        index
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.search(q, k))
            .unwrap_or_default()
    }
    fn search_fc_scored(
        index: &Mutex<Option<FlatClustered>>,
        q: &[f32],
        k: usize,
    ) -> Vec<(VectorId, f32)> {
        index
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.search(q, k))
            .unwrap_or_default()
    }

    /// 删除单个向量 id：hnsw（墓碑 + 四路摘除）与 flat（直接剔除）支持；
    /// 其余后端 fail-fast 返回 Err——与混合检索 filter 的支持面（hnsw/flat）同口径，
    /// 宁可显式拒绝也不静默「删了还能搜到」。
    /// EN: Delete one vector id — supported on hnsw (tombstone + four-way removal)
    /// and flat (direct retain); all other backends fail fast with Err, mirroring
    /// the hybrid-filter support surface. Better an explicit refusal than a delete
    /// whose hits keep showing up.
    pub async fn delete(&self, id: VectorId) -> Result<bool, String> {
        match self {
            Self::Hnsw(idx) => Ok(idx.remove(id).await),
            Self::Flat { data, .. } => {
                let mut guard = data.lock().unwrap();
                let before = guard.len();
                guard.retain(|(vid, _)| *vid != id);
                Ok(before != guard.len())
            }
            other => Err(format!(
                "delete is unsupported for index backend '{}' (supported: hnsw, flat)",
                other.backend_name()
            )),
        }
    }

    /// 后端名（错误诊断用；不引入 Display 以免被当成稳定 API 依赖）。
    /// EN: Backend name for diagnostics (no Display impl — not a stable API surface).
    fn backend_name(&self) -> &'static str {
        match self {
            Self::Hnsw(_) => "hnsw",
            Self::Clustered { .. } => "clustered",
            Self::FlatClustered { .. } => "flat-clustered",
            Self::Flat { .. } => "flat",
            Self::PqFlat { .. } => "pq-flat",
            Self::HiiL0 { .. } => "hii-l0",
        }
    }

    pub async fn pending_build_count(&self) -> usize {
        match self {
            Self::HiiL0 { inner, .. } => inner.lock().unwrap().size(),
            Self::Hnsw(idx) => idx.pending_build_count().await,
            Self::Clustered { buf, .. }
            | Self::FlatClustered { buf, .. }
            | Self::PqFlat { buf, .. } => buf.lock().unwrap().len(),
            Self::Flat { .. } => 0,
        }
    }
    pub async fn indexed_count(&self) -> usize {
        match self {
            Self::HiiL0 { inner, .. } => inner.lock().unwrap().size(),
            Self::Hnsw(idx) => idx.indexed_count().await,
            Self::Clustered { index, .. } => index
                .lock()
                .unwrap()
                .as_ref()
                .map(|c| c.total_vectors())
                .unwrap_or(0),
            Self::FlatClustered { index, .. } => index
                .lock()
                .unwrap()
                .as_ref()
                .map(|c| c.total_vectors())
                .unwrap_or(0),
            Self::Flat { data, .. } => data.lock().unwrap().len(),
            Self::PqFlat { index, .. } => index
                .lock()
                .unwrap()
                .as_ref()
                .map(|c| c.total_vectors())
                .unwrap_or(0),
        }
    }
    pub async fn hot_buffer_count(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.hot_buffer_count().await,
            _ => 0,
        }
    }

    /// 后端类型名（诊断/状态端点用，与 `--index-type` 取值一致）。
    /// EN: backend kind name for diagnostics; matches `--index-type` values.
    pub fn index_kind(&self) -> &'static str {
        self.backend_name()
    }

    pub fn hnsw_m(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.hnsw_m(),
            _ => 24,
        }
    }
    pub fn hnsw_ef_search(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.hnsw_ef_search(),
            _ => 256,
        }
    }
    pub fn hnsw_ef_construction_value(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.hnsw_ef_construction_value(),
            _ => 400,
        }
    }
    pub fn set_hnsw_m(&self, m: usize) -> Result<(), String> {
        if let Self::Hnsw(idx) = self {
            idx.set_hnsw_m(m)
        } else {
            Ok(())
        }
    }
    pub fn set_hnsw_ef_search(&self, ef: usize) -> Result<(), String> {
        if let Self::Hnsw(idx) = self {
            idx.set_hnsw_ef_search(ef)
        } else {
            Ok(())
        }
    }
    pub fn set_hnsw_ef_construction(&self, ef: usize) -> Result<(), String> {
        if let Self::Hnsw(idx) = self {
            idx.set_hnsw_ef_construction(ef)
        } else {
            Ok(())
        }
    }

    pub fn builder_metrics(&self) -> BuilderMetricsSnapshot {
        match self {
            Self::Hnsw(idx) => idx.builder_metrics(),
            _ => BuilderMetricsSnapshot {
                flush_count: 0,
                total_flushed_items: 0,
                max_pending_queue: 0,
                last_flush_items: 0,
                graph_edges_built: 0,
                graph_edges_level0: 0,
                graph_edges_upper: 0,
            },
        }
    }
    pub async fn run_async_builder_once(&self) -> AsyncBuilderOnceReport {
        match self {
            Self::Hnsw(idx) => idx.run_async_builder_once().await,
            _ => AsyncBuilderOnceReport {
                flushed: 0,
                plan_mode: Some(HnswBatchPlanMode::Serial),
            },
        }
    }
    pub fn start_async_builder_worker(
        &self,
        th: usize,
        t: u64,
        p: FlushPolicy,
    ) -> Option<houyi_vector::BuilderWorker> {
        match self {
            Self::Hnsw(idx) => Some(idx.start_async_builder_worker(th, t, p)),
            _ => None,
        }
    }

    /// 共享指针导出（Hnsw 零拷贝主表；其余后端克隆行缓冲）。
    /// EN: shared-pointer export (Hnsw shares the main table; other backends
    /// clone their row buffers).
    pub async fn export_vectors_shared(&self) -> Result<Vec<(u64, Arc<Vec<f32>>)>, String> {
        match self {
            Self::Hnsw(idx) => idx.export_vectors_shared().await,
            _ => {
                let mut out: Vec<(u64, Arc<Vec<f32>>)> = self
                    .export_snapshot_vectors()
                    .await?
                    .into_iter()
                    .map(|(id, v)| (id, Arc::new(v)))
                    .collect();
                // 非 HNSW 后端按插入顺序返回；导出契约要求 id 升序。
                // EN: non-HNSW backends return insertion order; the export
                // contract requires ascending ids.
                out.sort_by_key(|(id, _)| *id);
                Ok(out)
            }
        }
    }

    pub async fn export_snapshot_vectors(&self) -> Result<Vec<(u64, Vec<f32>)>, String> {
        match self {
            Self::HiiL0 { inner, .. } => Ok(inner.lock().unwrap().export_vectors()),
            Self::Hnsw(idx) => idx.export_snapshot_vectors().await,
            Self::Clustered { buf, .. }
            | Self::FlatClustered { buf, .. }
            | Self::PqFlat { buf, .. } => Ok(buf.lock().unwrap().clone()),
            Self::Flat { data, .. } => Ok(data.lock().unwrap().clone()),
        }
    }

    /// 按 id 批量取向量（仅 Hnsw 支持；其余返回空，调用方回退到索引检索）。
    /// EN: Bulk fetch by id (Hnsw only; others return empty and callers fall back).
    pub async fn vectors_by_ids(&self, ids: &[u64]) -> Vec<(u64, Vec<f32>)> {
        match self {
            Self::Hnsw(idx) => idx.vectors_by_ids(ids).await,
            _ => Vec::new(),
        }
    }

    /// 图内节点数（诊断；仅 Hnsw）。
    /// EN: Graph node count (diagnostic; Hnsw only).
    pub async fn graph_node_count(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.graph_node_count().await,
            _ => 0,
        }
    }

    /// 写入代数（过滤缓存新鲜度；仅 Hnsw）。
    /// EN: Write generation for filter-cache freshness (Hnsw only).
    pub fn write_generation(&self) -> u64 {
        match self {
            Self::Hnsw(idx) => idx.write_generation(),
            _ => 0,
        }
    }

    /// 编码 HNSW 图快照（P4.4.0）：仅 `Hnsw` 变体支持，其余返回 `None`。
    /// 调用前必须先 drain 构建队列（由调用方保证）。
    /// EN: Encode the HNSW graph snapshot (P4.4.0) — `Hnsw` variant only;
    /// other backends return `None`. Caller drains the builder queue first.
    pub async fn encode_graph_snapshot(&self) -> Option<Vec<u8>> {
        match self {
            Self::Hnsw(idx) => Some(idx.encode_graph_snapshot().await),
            _ => None,
        }
    }

    /// P5.0.2 的 collection WAL 压缩是否支持该后端。
    /// EN: Whether P5.0.2 collection WAL compaction supports this backend.
    pub fn supports_collection_snapshot(&self) -> bool {
        matches!(self, Self::Hnsw(_) | Self::Flat { .. } | Self::HiiL0 { .. })
    }

    /// 编码 collection 快照向量：HNSW 图字节，Flat/HII L0 原始 f32。
    /// EN: Encode collection snapshot vectors: HNSW graph bytes, or raw f32 for
    /// Flat/HII L0.
    pub async fn encode_collection_snapshot_vectors(
        &self,
    ) -> Result<(SnapshotVectorKind, u64, Vec<u8>), String> {
        match self {
            Self::Hnsw(idx) => {
                let bytes = idx.encode_graph_snapshot().await;
                let count = idx.graph_node_count().await as u64;
                Ok((SnapshotVectorKind::HnswGraph, count, bytes))
            }
            Self::Flat { data, .. } => {
                let rows = data.lock().unwrap().clone();
                raw_vectors_payload(&rows)
            }
            Self::HiiL0 { inner, .. } => {
                let rows = inner.lock().unwrap().export_vectors();
                raw_vectors_payload(&rows)
            }
            other => Err(format!(
                "collection snapshot unsupported for index backend '{}'",
                other.backend_name()
            )),
        }
    }

    /// 恢复 collection 快照向量：HNSW 图恢复，Flat/HII L0 批量插入并重建。
    /// EN: Restore collection snapshot vectors: HNSW graph restore, or batch insert
    /// + rebuild for Flat/HII L0.
    pub async fn restore_collection_snapshot_vectors(
        &self,
        kind: SnapshotVectorKind,
        count: u64,
        bytes: &[u8],
    ) -> Result<(), String> {
        match kind {
            SnapshotVectorKind::Empty => {
                if count != 0 || !bytes.is_empty() {
                    return Err("empty snapshot vectors require count=0 and no bytes".to_string());
                }
                Ok(())
            }
            SnapshotVectorKind::HnswGraph => match self.restore_graph_snapshot(bytes).await {
                Some(Ok(())) => Ok(()),
                Some(Err(e)) => Err(format!("graph snapshot restore failed: {e}")),
                None => Err("graph snapshot kind used with non-HNSW backend".to_string()),
            },
            SnapshotVectorKind::RawF32 => {
                let rows = decode_raw_vectors(bytes, count, self.dims())?;
                self.insert_batch(rows).await?;
                // HII L0 需要重建小图；Flat 为 no-op。
                // EN: HII L0 needs a graph rebuild; Flat is a no-op.
                self.optimize_build().await;
                Ok(())
            }
        }
    }

    /// 后端维度（快照恢复校验用）。
    /// EN: Backend dimensions (snapshot restore checks).
    pub fn dims(&self) -> usize {
        match self {
            Self::Hnsw(_) => 0, // HNSW 维度由图快照自带并在 restore 内校验。
            Self::Clustered { dims, .. }
            | Self::FlatClustered { dims, .. }
            | Self::Flat { dims, .. }
            | Self::PqFlat { dims, .. }
            | Self::HiiL0 { dims, .. } => *dims,
        }
    }

    /// 从图快照恢复：仅 `Hnsw` 变体支持（外层 `Some`），失败（结构不符/
    /// 损坏/非空目标）在内层 `Err`，调用方弃用快照全量重放。
    /// EN: Restore from a graph snapshot — `Hnsw` only (outer `Some`); the
    /// inner `Err` covers mismatch/corruption/non-empty target, in which case
    /// the caller discards the snapshot and falls back to full replay.
    pub async fn restore_graph_snapshot(
        &self,
        payload: &[u8],
    ) -> Option<Result<(), houyi_vector::GraphSnapshotError>> {
        match self {
            Self::Hnsw(idx) => Some(idx.restore_graph_snapshot(payload).await),
            _ => None,
        }
    }
    /// 报告后端**实际计分的度量**，用于快照头/状态端点等「按什么度量构建与检索」的
    /// 元数据场景，也是 HTTP 边缘距离→相似度换算的唯一依据（pq-flat 如实报告 L2，
    /// HiiL0 报告 Cosine）。
    /// EN: Reports the metric the backend **actually scores with**, for snapshot
    /// headers and status endpoints; it is also the sole input to the edge
    /// distance→similarity conversion (pq-flat truthfully reports L2, HiiL0 reports
    /// Cosine).
    pub fn distance_metric(&self) -> DistanceMetric {
        match self {
            Self::Hnsw(idx) => idx.distance_metric(),
            // Clustered 系列按构图时记录的实际度量报告（库 build 公开接受 L2）。
            // Clustered variants report the actual recorded build metric (the library
            // build APIs publicly accept L2).
            Self::Clustered { metric, .. } | Self::FlatClustered { metric, .. } => *metric,
            // Flat 按变体记录度量计分；HiiL0 内部单位化，恒为余弦。
            // PqFlat 内部恒为平方 L2——现在如实报告 L2（旧实现报名义 Cosine，
            // 与 scored 距离口径不一致），边缘据此按 `1 - d` 换算，分数数值与旧版
            // 完全一致（旧版 pq-flat 不钳制）。
            // EN: Flat scores by its recorded metric; HiiL0 normalizes internally so
            // it is always cosine. PqFlat is squared-L2 internally and now reports L2
            // truthfully (the old nominal Cosine disagreed with scored distances);
            // the edge converts with `1 - d`, numerically identical to the legacy
            // unclamped pq-flat output.
            Self::Flat { metric, .. } => *metric,
            Self::PqFlat { .. } => DistanceMetric::L2,
            Self::HiiL0 { .. } => DistanceMetric::Cosine,
        }
    }

    /// scored 距离是否为余弦语义（派生判定，保留给测试与防御性断言）。
    /// HTTP 边缘的换算已改为直接匹配 [`Self::distance_metric`]（三种度量各自换算）。
    /// pq-flat（平方 L2 距离，分数合法越界）恒为 false；HNSW、
    /// Clustered/FlatClustered/Flat 按变体记录的实际度量判定；HiiL0 恒为余弦。
    /// EN: Derived predicate "are scored distances cosine semantics?" retained for
    /// tests and defensive assertions. The HTTP edge now converts by matching
    /// [`Self::distance_metric`] directly. pq-flat (squared L2, legitimately out of
    /// range) is always false; HNSW/Clustered/FlatClustered/Flat follow their
    /// recorded metric; HiiL0 is always cosine.
    pub fn distance_is_cosine(&self) -> bool {
        match self {
            // pq-flat 建表标称 Cosine，但 scored search 返回 L2²：必须显式排除。
            // pq-flat nominally reports Cosine yet scored search returns L2².
            Self::PqFlat { .. } => false,
            Self::Hnsw(idx) => idx.distance_metric() == DistanceMetric::Cosine,
            // Clustered/FlatClustered 与 HNSW 同一口径：库层 build 公开接受 L2 并在
            // search 时按 L2² 计分（metric 只存在库类型私有字段里、无访问器），因此
            // 变体显式记录构图度量。服务器当前只走 Cosine 构图，L2 判定属纵深防御。
            // EN: Same gate as HNSW — the library build APIs publicly accept L2 and
            // honor it at search (the metric lives in a private field with no accessor),
            // so the variant records its build metric. Server assembly builds Cosine
            // only today; the L2 arm is defense in depth.
            Self::Clustered { metric, .. } | Self::FlatClustered { metric, .. } => {
                *metric == DistanceMetric::Cosine
            }
            // Flat 按记录度量判定（L2/IP/L1 分数都不保证落在 [-1,1]）；
            // HiiL0 恒余弦。
            // EN: Flat follows its recorded metric (L2/IP/L1 scores are not bounded
            // to [-1,1]); HiiL0 is always cosine.
            Self::Flat { metric, .. } => *metric == DistanceMetric::Cosine,
            Self::HiiL0 { .. } => true,
        }
    }

    pub async fn optimize_build(&self) -> usize {
        match self {
            Self::HiiL0 { inner, .. } => {
                let mut l0 = inner.lock().unwrap();
                l0.rebuild_graph();
                l0.size()
            }
            Self::Hnsw(_) => 0,
            Self::Flat { data, .. } => {
                let n = data.lock().unwrap().len();
                n
            }
            Self::Clustered {
                buf,
                index,
                dims,
                built,
                metric,
            } => {
                let mut b = built.lock().unwrap();
                if *b {
                    return 0;
                }
                let data = {
                    let mut bg = buf.lock().unwrap();
                    std::mem::take(&mut *bg)
                };
                if data.is_empty() {
                    return 0;
                }
                let n = data.len();
                let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
                let vecs: Vec<f32> = data.iter().flat_map(|(_, v)| v.clone()).collect();
                drop(data);
                let cfg = ClusterConfig {
                    hnsw_ef_construction: 400,
                    hnsw_ef_search: 256,
                    hnsw_m: 24,
                    ..ClusterConfig::infer(n, *dims)
                };
                // 透传变体记录的实际构图度量：build 的度量决定 scored 距离是余弦还是
                // L2²，旧实现硬编码 Cosine 会让 metric=L2 的变体「自称 L2、按 Cosine
                // 构图」，与 distance_metric()/distance_is_cosine() 三处分歧。
                // EN: Thread the variant's recorded metric — the build metric decides
                // whether scored distances are cosine or L2²; hard-coding Cosine made an
                // L2 variant report L2 while actually building/searching by cosine.
                let ch = ClusteredHnsw::build(&vecs, &ids, *dims, *metric, &cfg);
                let count = ch.total_vectors();
                *index.lock().unwrap() = Some(ch);
                *b = true;
                count
            }
            Self::FlatClustered {
                buf,
                index,
                dims,
                built,
                metric,
            } => {
                let mut b = built.lock().unwrap();
                if *b {
                    return 0;
                }
                let data = {
                    let mut bg = buf.lock().unwrap();
                    std::mem::take(&mut *bg)
                };
                if data.is_empty() {
                    return 0;
                }
                let n = data.len();
                let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
                let vecs: Vec<f32> = data.iter().flat_map(|(_, v)| v.clone()).collect();
                drop(data);
                let c = (n as f64).sqrt().ceil() as usize;
                // 同 Clustered 分支：透传变体度量，保证构图口径与报告口径一致。
                // Same as the Clustered arm: thread the variant metric so the built and
                // reported metrics never diverge.
                let fc =
                    FlatClustered::build(&vecs, &ids, *dims, *metric, c, 15, 0xA7E5_72D8_B7DB_BEEF);
                let count = fc.total_vectors();
                *index.lock().unwrap() = Some(fc);
                *b = true;
                count
            }
            Self::PqFlat {
                buf,
                index,
                dims,
                built,
                ..
            } => {
                let mut b = built.lock().unwrap();
                if *b {
                    return 0;
                }
                let data = {
                    let mut bg = buf.lock().unwrap();
                    std::mem::take(&mut *bg)
                };
                if data.is_empty() {
                    return 0;
                }
                let n = data.len();
                let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
                let vecs: Vec<f32> = data.iter().flat_map(|(_, v)| v.clone()).collect();
                drop(data);
                let c = (n as f64).sqrt().ceil() as usize;
                // PqFlatClustered::build 的 metric 形参 by design 被忽略（库层签名为
                // `_metric`：PQ 距离恒为平方 L2，建表度量不参与距离计算），变体也不
                // 存储 metric；继续传标称 Cosine，真实 score 语义由 distance_is_cosine()
                // 对 PqFlat 恒 false 单独表达。
                // EN: PqFlatClustered::build ignores its metric parameter by design
                // (`_metric` — PQ distances are always squared-L2); the variant stores no
                // metric. Keep passing the nominal Cosine; real score semantics are
                // expressed by distance_is_cosine() returning false for PqFlat.
                let pf = PqFlatClustered::build(
                    &vecs,
                    &ids,
                    *dims,
                    DistanceMetric::Cosine,
                    c,
                    15,
                    0xA7E5_72D8_B7DB_BEEF,
                );
                let count = pf.total_vectors();
                *index.lock().unwrap() = Some(pf);
                *b = true;
                count
            }
        }
    }
}

/// 原始 f32 向量段编码：[id u64 BE][dims × f32 BE] 重复 count 次。
/// EN: Raw f32 vector section: repeated [id u64 BE][dims × f32 BE].
fn raw_vectors_payload(
    rows: &[(u64, Vec<f32>)],
) -> Result<(SnapshotVectorKind, u64, Vec<u8>), String> {
    if rows.is_empty() {
        return Ok((SnapshotVectorKind::Empty, 0, Vec::new()));
    }
    let dims = rows[0].1.len();
    if dims == 0 {
        return Err("collection snapshot: zero-dimension vector".to_string());
    }
    let per = 8usize
        .checked_add(
            dims.checked_mul(4)
                .ok_or_else(|| "collection snapshot: dims overflow".to_string())?,
        )
        .ok_or_else(|| "collection snapshot: dims overflow".to_string())?;
    let total = per
        .checked_mul(rows.len())
        .ok_or_else(|| "collection snapshot: vector bytes overflow".to_string())?;
    let mut bytes = Vec::with_capacity(total);
    for (id, values) in rows {
        if values.len() != dims {
            return Err(format!(
                "collection snapshot: inconsistent dims for id {id} ({} != {dims})",
                values.len()
            ));
        }
        bytes.extend_from_slice(&id.to_be_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_bits().to_be_bytes());
        }
    }
    Ok((SnapshotVectorKind::RawF32, rows.len() as u64, bytes))
}

fn decode_raw_vectors(
    bytes: &[u8],
    count: u64,
    dims: usize,
) -> Result<Vec<(u64, Vec<f32>)>, String> {
    if dims == 0 {
        return Err("collection snapshot: zero dimensions".to_string());
    }
    let count: usize = count
        .try_into()
        .map_err(|_| "collection snapshot: vector count too large".to_string())?;
    let per = 8usize
        .checked_add(
            dims.checked_mul(4)
                .ok_or_else(|| "collection snapshot: dims overflow".to_string())?,
        )
        .ok_or_else(|| "collection snapshot: dims overflow".to_string())?;
    let expected = per
        .checked_mul(count)
        .ok_or_else(|| "collection snapshot: vector bytes overflow".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "collection snapshot: raw vector bytes {} != expected {expected}",
            bytes.len()
        ));
    }
    let mut rows = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(per) {
        let id = u64::from_be_bytes(chunk[..8].try_into().unwrap());
        let mut values = Vec::with_capacity(dims);
        for i in 0..dims {
            let start = 8 + i * 4;
            values.push(f32::from_bits(u32::from_be_bytes(
                chunk[start..start + 4].try_into().unwrap(),
            )));
        }
        rows.push((id, values));
    }
    Ok(rows)
}

impl From<Arc<HotBufferIndex>> for IndexBackend {
    fn from(idx: Arc<HotBufferIndex>) -> Self {
        IndexBackend::Hnsw(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn drain_hnsw(backend: &IndexBackend) {
        for _ in 0..1000 {
            if backend.run_async_builder_once().await.flushed == 0 {
                break;
            }
        }
    }

    fn batch_200_d16() -> Vec<(u64, Vec<f32>)> {
        let dims = 16usize;
        (0..200_u64)
            .map(|id| {
                (
                    id,
                    (0..dims)
                        .map(|j| ((id as usize * 13 + j * 7) % 200) as f32 / 200.0)
                        .collect(),
                )
            })
            .collect()
    }

    /// scored 契约：结果数 == k（数据足够时）、距离升序、id 序列与 id-only 入口一致。
    /// EN: Scored contract across backends: k results, ascending distances,
    /// ids identical to the id-only entry point.
    #[tokio::test]
    async fn hnsw_scored_search_contract() {
        let backend = IndexBackend::new_hnsw(16);
        backend.insert_batch(batch_200_d16()).await.unwrap();
        drain_hnsw(&backend).await;
        assert_eq!(backend.indexed_count().await, 200);

        let q = vec![0.42_f32; 16];
        let scored = backend
            .search_scored_with_ef_override(&q, 10, None, Some(256))
            .await;
        assert_eq!(scored.len(), 10);
        for pair in scored.windows(2) {
            assert!(pair[0].1 <= pair[1].1, "distances must be ascending");
        }
        let scored_ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
        let ids_only = backend
            .search_with_ef_override(&q, 10, None, Some(256))
            .await;
        assert_eq!(scored_ids, ids_only);
    }

    /// HNSW 精确匹配距离 ≈ 0（全余弦公式，非单位向量也成立）。
    /// EN: Exact-match query yields distance ~0 even for non-unit vectors.
    #[tokio::test]
    async fn hnsw_scored_exact_match_near_zero() {
        let backend = IndexBackend::new_hnsw(4);
        let target = vec![3.0_f32, 0.0, -1.0, 2.0];
        backend
            .insert_batch(vec![
                (1, vec![0.0, 1.0, 0.0, 0.0]),
                (2, target.clone()),
                (3, vec![1.0, 1.0, 1.0, 1.0]),
            ])
            .await
            .unwrap();
        // 批量写入先进 builder_queue，排空后才进入主图可被检索。
        // Batched rows land in builder_queue first; drain before searching.
        drain_hnsw(&backend).await;
        let scored = backend
            .search_scored_with_ef_override(&target, 3, None, Some(256))
            .await;
        assert_eq!(scored.first().map(|(id, _)| *id), Some(2));
        assert!(scored.first().unwrap().1.abs() < 1e-5);
    }

    /// Flat 余弦修正：同向但模长大（L2²=81）的向量必须排在正交但 L2 更近
    /// （L2²=2）的向量之前——旧 flat 用裸 L2² 排序，会把正交向量误排第一。
    /// EN: Flat cosine correction — a same-direction far-magnitude vector
    /// (L2²=81) must outrank an orthogonal vector (L2²=2); the old raw-L2²
    /// scoring ranked the orthogonal one first despite distance_metric()=Cosine.
    #[tokio::test]
    async fn flat_scored_uses_cosine_not_raw_l2() {
        let backend = IndexBackend::new_flat(2);
        backend
            .insert_batch(vec![
                // q=[1,0]：id=1 同向模长 10（余弦距离 0，L2²=81）；
                // id=2 正交单位向量（余弦距离 1，L2²=2）。
                // EN: q=[1,0]: id=1 is same-direction with magnitude 10 (cosine
                // distance 0, L2²=81); id=2 is an orthogonal unit vector (cosine
                // distance 1, L2²=2).
                (1, vec![10.0, 0.0]),
                (2, vec![0.0, 1.0]),
            ])
            .await
            .unwrap();
        let q = vec![1.0, 0.0];
        let scored = backend
            .search_scored_with_ef_override(&q, 2, None, None)
            .await;
        assert_eq!(scored.len(), 2);
        assert_eq!(
            scored[0].0, 1,
            "cosine: same-direction vector must rank first"
        );
        assert!(
            scored[0].1.abs() < 1e-6,
            "same-direction cosine distance ~0"
        );
        assert!(
            (scored[1].1 - 1.0).abs() < 1e-6,
            "orthogonal cosine distance ~1"
        );
        for pair in scored.windows(2) {
            assert!(pair[0].1 <= pair[1].1);
        }
        // id-only 链路同步改为余弦排序。
        // EN: The id-only entry point inherits the corrected cosine ordering.
        let ids = backend.search_with_ef_override(&q, 2, None, None).await;
        assert_eq!(ids, vec![1, 2]);

        // mask 仍然生效：先按 mask 过滤再计分。
        // EN: Mask support retained — filter before scoring.
        let masked = backend
            .search_scored_with_ef_override(&q, 2, Some(&[2]), None)
            .await;
        assert_eq!(
            masked.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![2]
        );

        // 空结果（无命中 mask）→ 空 pairs。
        // EN: Empty result (mask matches nothing) → empty pairs.
        let empty = backend
            .search_scored_with_ef_override(&q, 2, Some(&[999]), None)
            .await;
        assert!(empty.is_empty());
    }

    /// OCR #37：Flat 的 L2/IP/L1 分支必须按各自的原始距离计分且排序方向正确——
    /// 同一组向量在 L2 与 L1 下顺序不同，可证两个分支没有再共用余弦公式。
    /// EN: OCR #37 — Flat's L2/IP/L1 arms must score by their own raw distances
    /// with the right ordering. The same vectors order differently under L2 vs L1,
    /// proving the arms no longer share the cosine formula.
    #[tokio::test]
    async fn flat_scored_l2_ip_l1_use_own_distances() {
        // q=[1,0]；id=1 的 [5,4]：L2²=32、L1=8、IP=-5；
        // id=2 的 [1,6]：L2²=36、L1=6、IP=-1。
        // L2 序 [1,2]，L1 序 [2,1]，IP 序 [1,2]（更负 = 内积更大 = 更近）。
        // EN: q=[1,0]; id=1 [5,4]: L2²=32, L1=8, IP=-5; id=2 [1,6]: L2²=36,
        // L1=6, IP=-1. L2 order [1,2], L1 order [2,1], IP order [1,2].
        let cases = [
            (DistanceMetric::L2, vec![1_u64, 2], vec![32.0_f32, 36.0]),
            (DistanceMetric::L1, vec![2, 1], vec![6.0, 8.0]),
            (DistanceMetric::InnerProduct, vec![1, 2], vec![-5.0, -1.0]),
        ];
        for (metric, ids, dists) in cases {
            let backend = IndexBackend::new_flat_with_metric(2, metric);
            backend
                .insert_batch(vec![(1, vec![5.0, 4.0]), (2, vec![1.0, 6.0])])
                .await
                .unwrap();
            let q = vec![1.0, 0.0];
            let scored = backend
                .search_scored_with_ef_override(&q, 2, None, None)
                .await;
            assert_eq!(scored.len(), 2);
            let got_ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
            assert_eq!(got_ids, ids, "{metric:?}: wrong ordering");
            for ((_, got), want) in scored.iter().zip(dists.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "{metric:?}: distance {got} != expected {want}"
                );
            }
            assert_eq!(backend.distance_metric(), metric);
        }
    }

    /// HII L0 scored：暴力路径距离升序、精确匹配 ≈ 0、与 id-only 一致。
    /// EN: HiiL0 scored brute-force path: ascending, exact match ~0, ids match.
    #[tokio::test]
    async fn hii_l0_scored_search_contract() {
        let backend = IndexBackend::new_hii_l0(4);
        backend
            .insert_batch(vec![
                (1, vec![1.0, 0.0, 0.0, 0.0]),
                (2, vec![0.0, 1.0, 0.0, 0.0]),
                (3, vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .await
            .unwrap();
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let scored = backend
            .search_scored_with_ef_override(&q, 2, None, None)
            .await;
        assert_eq!(scored.len(), 2);
        assert_eq!(scored[0].0, 1);
        assert!(scored[0].1.abs() < 1e-6);
        assert!(scored[0].1 <= scored[1].1);
        let scored_ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            scored_ids,
            backend.search_with_ef_override(&q, 2, None, None).await
        );
    }

    /// Clustered/FlatClustered/PqFlat：c.search(q,k) 的 (id, dist) 对直接透出，
    /// None 索引（未 build）→ 空。
    /// EN: Clustered backends pass (id, dist) pairs through; unbuilt index → empty.
    #[tokio::test]
    async fn clustered_backends_scored_passthrough() {
        for (backend, name) in [
            (IndexBackend::new_clustered(16), "clustered"),
            (IndexBackend::new_flat_clustered(16), "flat-clustered"),
            (IndexBackend::new_pq_flat(16), "pq-flat"),
        ] {
            // 未构建索引时必须返回空而非 panic。
            // EN: An unbuilt index must return empty rather than panic.
            let q = vec![0.42_f32; 16];
            assert!(
                backend
                    .search_scored_with_ef_override(&q, 10, None, None)
                    .await
                    .is_empty(),
                "{name}: unbuilt index must return empty"
            );

            backend.insert_batch(batch_200_d16()).await.unwrap();
            let built = backend.optimize_build().await;
            assert_eq!(built, 200, "{name}: build must index all rows");
            let scored = backend
                .search_scored_with_ef_override(&q, 10, None, None)
                .await;
            assert!(!scored.is_empty(), "{name}: scored search must return hits");
            for pair in scored.windows(2) {
                assert!(pair[0].1 <= pair[1].1, "{name}: distances ascending");
            }
            let scored_ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
            assert_eq!(
                scored_ids,
                backend.search_with_ef_override(&q, 10, None, None).await,
                "{name}: scored/id-only ids must match"
            );
        }
    }

    /// OCR 第五轮回归：服务端写入总闸门必须对**所有后端**拒绝 NaN/±Inf。库层校验
    /// 不对称——HiiL0 硬拒非有限分量，HotBufferIndex 只查维度，Clustered/FlatClustered/
    /// PqFlat/Flat 连 Err 都不返回；pgwire binary float4/float8 经 `f32::from_bits`
    /// 解码，毒分量原样通过维度校验，是唯一仍开放的毒数据入口。整批原子拒绝：毒批次
    /// 内的合法行也不得部分写入；单行入口 `insert` 共用同一闸门。
    /// EN: The server-side write choke point must reject NaN/±Inf on EVERY backend:
    /// library validation is asymmetric (HiiL0 hard-rejects; HotBuffer checks dims only;
    /// Clustered/Flat/PqFlat/Flat never return Err) and pgwire binary floats decoded via
    /// f32::from_bits sailed through the dim check. A poisoned batch fails atomically and
    /// the single-row `insert` entry shares the gate.
    #[tokio::test]
    async fn insert_gate_rejects_non_finite_on_every_backend() {
        #[rustfmt::skip]
        let makers: [fn(usize) -> IndexBackend; 6] = [
            IndexBackend::new_hnsw,
            IndexBackend::new_clustered,
            IndexBackend::new_flat_clustered,
            IndexBackend::new_flat,
            IndexBackend::new_pq_flat,
            IndexBackend::new_hii_l0,
        ];
        for make in makers {
            let backend = make(4);
            // 一条合法行先落（用于确认毒批次前后没有部分写入）。
            // One good row lands first; it marks the pre-poison state.
            backend
                .insert_batch(vec![(1, vec![1.0, 0.0, 0.0, 0.0])])
                .await
                .unwrap();

            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let err = backend
                    .insert_batch(vec![
                        (2, vec![1.0, 0.0, 0.0, 0.0]),
                        (3, vec![1.0, bad, 0.0, 0.0]),
                    ])
                    .await
                    .expect_err("poisoned batch must be rejected on every backend");
                assert!(err.contains("non-finite"), "unexpected message: {err}");

                let err = backend
                    .insert(9, vec![0.0, bad, 0.0, 0.0])
                    .await
                    .expect_err("single-row insert shares the finiteness gate");
                assert!(err.contains("non-finite"), "unexpected message: {err}");
            }

            // 原子性：导出的行里只有最初的合法 id=1。
            // Atomicity: only the original good row is observable afterwards.
            let mut ids: Vec<u64> = backend
                .export_snapshot_vectors()
                .await
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![1], "no poisoned or partial rows may land");
        }
    }

    /// OCR 回归：clamp 判定必须按后端真实语义——pq-flat 永远非余弦（标称 Cosine 但
    /// scored search 返回 L2²）；HNSW 按建索引实际度量（库层公开支持 L2）；其余后端
    /// 当前均为余弦。
    /// EN: The clamp gate follows real semantics — pq-flat is never cosine (reports
    /// Cosine but returns L2²), HNSW follows its actual build metric (L2 is publicly
    /// constructible at library level), every other current backend is cosine.
    #[tokio::test]
    async fn distance_is_cosine_follows_metric_and_pq_flat_exception() {
        assert!(IndexBackend::new_hnsw(4).distance_is_cosine());
        assert!(IndexBackend::new_flat(4).distance_is_cosine());
        assert!(IndexBackend::new_clustered(4).distance_is_cosine());
        assert!(IndexBackend::new_flat_clustered(4).distance_is_cosine());
        assert!(IndexBackend::new_hii_l0(4).distance_is_cosine());
        assert!(!IndexBackend::new_pq_flat(4).distance_is_cosine());
        let l2_hnsw = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new_with_metric(
            4,
            DistanceMetric::L2,
        )));
        assert!(!l2_hnsw.distance_is_cosine());
    }

    /// OCR 回归（fail-closed 纵深防御）：ClusteredHnsw/FlatClustered 的库层 build 公开
    /// 接受 L2 并在 search 时按 L2² 计分（metric 仅存于库类型私有字段）。若变体不记录
    /// 实际构图度量，`distance_is_cosine()` 会硬编码 true，把 L2² 分数误钳进 [-1,1]——
    /// 与 pq-flat 同类的 fail-open。L2 构图的变体必须报告非余弦。
    /// EN: The library build APIs publicly accept L2 for clustered indexes and score by
    /// L2²; a variant that failed to record the build metric would hard-code cosine and
    /// wrongly clamp L2² scores into [-1,1] (the pq-flat fail-open class). L2-built
    /// clustered variants must report non-cosine.
    #[tokio::test]
    async fn clustered_l2_build_reports_non_cosine_and_still_searches() {
        let dims = 16usize;
        let batch = batch_200_d16();
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        let vecs: Vec<f32> = batch.iter().flat_map(|(_, v)| v.clone()).collect();
        let cfg = ClusterConfig {
            hnsw_ef_construction: 400,
            hnsw_ef_search: 256,
            hnsw_m: 24,
            ..ClusterConfig::infer(200, dims)
        };

        let ch = ClusteredHnsw::build(&vecs, &ids, dims, DistanceMetric::L2, &cfg);
        let l2_clustered = IndexBackend::Clustered {
            buf: Arc::new(Mutex::new(Vec::new())),
            index: Arc::new(Mutex::new(Some(ch))),
            dims,
            built: Arc::new(Mutex::new(true)),
            metric: DistanceMetric::L2,
        };
        assert!(!l2_clustered.distance_is_cosine());
        assert_eq!(l2_clustered.distance_metric(), DistanceMetric::L2);
        let q = vec![0.42_f32; dims];
        let hits = l2_clustered
            .search_scored_with_ef_override(&q, 5, None, None)
            .await;
        assert!(!hits.is_empty(), "L2 clustered index still returns hits");

        let clusters = (200.0_f64).sqrt().ceil() as usize;
        let fc = FlatClustered::build(
            &vecs,
            &ids,
            dims,
            DistanceMetric::L2,
            clusters,
            15,
            0xA7E5_72D8_B7DB_BEEF,
        );
        let l2_flat_clustered = IndexBackend::FlatClustered {
            buf: Arc::new(Mutex::new(Vec::new())),
            index: Arc::new(Mutex::new(Some(fc))),
            dims,
            built: Arc::new(Mutex::new(true)),
            metric: DistanceMetric::L2,
        };
        assert!(!l2_flat_clustered.distance_is_cosine());
        assert_eq!(l2_flat_clustered.distance_metric(), DistanceMetric::L2);
        let hits = l2_flat_clustered
            .search_scored_with_ef_override(&q, 5, None, None)
            .await;
        assert!(
            !hits.is_empty(),
            "L2 flat-clustered index still returns hits"
        );
    }

    /// OCR 回归：`optimize_build` 必须把变体记录的构图度量透传给库层 build，而不是
    /// 硬编码 Cosine——否则变体字段自称 L2、实际图按 Cosine 构建，`distance_metric()`/
    /// `distance_is_cosine()` 与 scored 距离口径三者分裂（边缘会把 L2² 分数误当余弦
    /// clamp）。行为证据：远离全部数据（分量 0..1）的查询 q=[10;16] 在 L2² 口径下
    /// 最近距离远超余弦距离上界 2；若 build 被硬编码成 Cosine，本断言失败。
    /// EN: optimize_build must thread the variant's stored metric into the library build
    /// instead of hard-coding Cosine; otherwise metric reporting and actual scored
    /// distances diverge and the edge mis-clamps L2² scores. A far query (q=10 vs data
    /// in 0..1) yields squared-L2 distances far above the cosine upper bound 2.
    #[tokio::test]
    async fn optimize_build_threads_stored_l2_metric() {
        let dims = 16usize;
        let batch = batch_200_d16();
        let q = vec![10.0_f32; dims];

        // ClusteredHnsw 变体：未构图、metric=L2。
        // ClusteredHnsw variant: unbuilt buf, recorded metric L2.
        let clustered = IndexBackend::Clustered {
            buf: Arc::new(Mutex::new(batch.clone())),
            index: Arc::new(Mutex::new(None)),
            dims,
            built: Arc::new(Mutex::new(false)),
            metric: DistanceMetric::L2,
        };
        assert_eq!(clustered.optimize_build().await, 200);
        assert_eq!(clustered.distance_metric(), DistanceMetric::L2);
        assert!(!clustered.distance_is_cosine());
        let hits = clustered
            .search_scored_with_ef_override(&q, 5, None, None)
            .await;
        assert!(!hits.is_empty(), "built L2 clustered index returns hits");
        assert!(
            hits[0].1 > 2.0,
            "built metric must be L2 (squared-L2 distance > 2), got {}",
            hits[0].1
        );

        // FlatClustered 变体同口径。
        // Same contract for the FlatClustered variant.
        let flat_clustered = IndexBackend::FlatClustered {
            buf: Arc::new(Mutex::new(batch)),
            index: Arc::new(Mutex::new(None)),
            dims,
            built: Arc::new(Mutex::new(false)),
            metric: DistanceMetric::L2,
        };
        assert_eq!(flat_clustered.optimize_build().await, 200);
        assert_eq!(flat_clustered.distance_metric(), DistanceMetric::L2);
        assert!(!flat_clustered.distance_is_cosine());
        let hits = flat_clustered
            .search_scored_with_ef_override(&q, 5, None, None)
            .await;
        assert!(
            !hits.is_empty(),
            "built L2 flat-clustered index returns hits"
        );
        assert!(
            hits[0].1 > 2.0,
            "built metric must be L2 (squared-L2 distance > 2), got {}",
            hits[0].1
        );
    }

    /// P5.0.2：HNSW/Flat/HII L0 的 collection 快照向量 roundtrip；其他后端不支持。
    /// EN: P5.0.2 — collection-snapshot vector roundtrip for HNSW/Flat/HII L0;
    /// other backends are unsupported.
    #[tokio::test]
    async fn collection_snapshot_vectors_roundtrip_supported_backends() {
        let rows = vec![
            (1u64, vec![1.0f32, 0.0, 0.0, 0.0]),
            (2u64, vec![0.0f32, 1.0, 0.0, 0.0]),
        ];
        for backend in [
            IndexBackend::new_hnsw(4),
            IndexBackend::new_flat(4),
            IndexBackend::new_hii_l0(4),
        ] {
            assert!(backend.supports_collection_snapshot());
            backend.insert_batch(rows.clone()).await.unwrap();
            if matches!(&backend, IndexBackend::Hnsw(_)) {
                drain_hnsw(&backend).await;
            }
            let (kind, count, bytes) = backend.encode_collection_snapshot_vectors().await.unwrap();
            assert_eq!(count, 2);
            let restored = match backend {
                IndexBackend::Hnsw(_) => IndexBackend::new_hnsw(4),
                IndexBackend::Flat { .. } => IndexBackend::new_flat(4),
                IndexBackend::HiiL0 { .. } => IndexBackend::new_hii_l0(4),
                _ => unreachable!(),
            };
            restored
                .restore_collection_snapshot_vectors(kind, count, &bytes)
                .await
                .unwrap();
            if matches!(&restored, IndexBackend::Hnsw(_)) {
                drain_hnsw(&restored).await;
            }
            let hits = restored
                .search_scored_with_ef_override(&[1.0, 0.0, 0.0, 0.0], 1, None, None)
                .await;
            assert_eq!(hits[0].0, 1, "{kind:?} roundtrip must return id 1");
        }

        for unsupported in [
            IndexBackend::new_clustered(4),
            IndexBackend::new_flat_clustered(4),
            IndexBackend::new_pq_flat(4),
        ] {
            assert!(!unsupported.supports_collection_snapshot());
            assert!(unsupported
                .encode_collection_snapshot_vectors()
                .await
                .is_err());
        }
    }
}
