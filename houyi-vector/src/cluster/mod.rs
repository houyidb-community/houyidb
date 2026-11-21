//! 聚类分治 HNSW：K-Means 路由器 + C 个独立小 HNSW 图。
//! Clustered HNSW: K-Means router + C independent sub-graph HNSWs.
//!
//! 不改 HNSW 本质，通过空间划分消除构建阶段的串行瓶颈。
//! Build-time serial bottleneck removed via spatial partitioning.

pub mod boundary;
pub mod flat_clustered;
pub mod kmeans;
pub mod pq_flat;
pub mod router;

pub use flat_clustered::{FlatClustered, FlatStorageDtype, DEFAULT_SEARCH_TARGET_PER_K};
pub use pq_flat::PqFlatClustered;

use crate::hnsw_graph::HnswGraph;
use crate::{DistanceMetric, VectorDtype, VectorId};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// 聚类分治配置。
/// EN: Cluster-partitioning configuration.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub num_clusters: usize,
    pub nprobe: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub kmeans_iters: usize,
    pub seed: u64,
    /// 边界复制阈值 (默认 1.2, 0 = 禁用)
    /// EN: Boundary replication threshold (default 1.2, 0 = disabled).
    pub boundary_alpha: f32,
}

impl ClusterConfig {
    pub fn infer(n: usize, dims: usize) -> Self {
        let c = (n as f64).sqrt().ceil() as usize;
        let c = c.clamp(2, 4096);
        let m = if dims <= 128 {
            12
        } else if dims <= 768 {
            16
        } else {
            20
        };
        Self {
            num_clusters: c,
            // 调优结论 (50K1536D OpenAI 实测): alpha=0 + nprobe=80 得 recall 0.999。
            // 固定阈值: C≤100 → 50%, C≤300 → 35%, C≤1000 → 15%, 更大 → 8%
            // EN: Tuning result (50K×1536D OpenAI measurement): alpha=0 + nprobe=80 gives recall 0.999.
            // EN: Fixed ratio: C≤100 → 50%, C≤300 → 35%, C≤1000 → 15%, larger → 8%.
            nprobe: {
                let frac = if c <= 100 {
                    0.5
                } else if c <= 300 {
                    0.35
                } else if c <= 1000 {
                    0.15
                } else {
                    0.08
                };
                (c as f64 * frac).ceil() as usize
            },
            hnsw_m: m,
            hnsw_ef_construction: 400,
            hnsw_ef_search: 256,
            kmeans_iters: 15,
            seed: 0xA7E5_72D8_B7DB_BEEF,
            boundary_alpha: 0.0, // 实测调优: nprobe 加大即可覆盖 recall, 无需边界复制
        }
    }
}

/// 单个聚类的子图。
/// EN: Subgraph of a single cluster.
struct ClusterGraph {
    #[allow(dead_code)]
    id: usize,
    vector_ids: Vec<VectorId>,
    graph: HnswGraph,
    /// 分裂阈值 = 2 × √N_expected（构建后固定）
    /// EN: Split threshold = 2 × √N_expected (fixed after build).
    split_threshold: usize,
}

/// 聚类分治 HNSW 索引。
/// EN: Cluster-partitioned HNSW index.
pub struct ClusteredHnsw {
    #[allow(dead_code)]
    dims: usize,
    metric: DistanceMetric,
    num_clusters: usize,
    nprobe: usize,
    /// 边界复制阈值（与构建期 `ClusterConfig::boundary_alpha` 一致；0 = 禁用）
    /// EN: Boundary replication threshold (matches the build-time
    /// `ClusterConfig::boundary_alpha`; 0 = disabled).
    boundary_alpha: f32,
    centroids: Vec<f32>,
    clusters: Vec<RwLock<ClusterGraph>>,
    /// 运行期 nprobe 覆盖（0 = 构建期值）；提高只会多路由簇。
    /// EN: Runtime nprobe override (0 = build-time value); raising it only
    /// routes more clusters.
    search_nprobe: AtomicUsize,
    /// 已索引的向量总数（含边界复制）
    /// EN: Total indexed vectors (including boundary replication).
    total_indexed: AtomicUsize,
    #[allow(dead_code)]
    next_id: std::sync::atomic::AtomicU64,
}

impl ClusteredHnsw {
    /// 从平坦向量数据构建聚类 HNSW 索引（含边界复制）。
    /// EN: Builds a clustered HNSW index from flat vector data (with boundary replication).
    pub fn build(
        vectors: &[f32],
        ids: &[VectorId],
        dims: usize,
        metric: DistanceMetric,
        config: &ClusterConfig,
    ) -> Self {
        assert_eq!(vectors.len(), ids.len() * dims);
        let n = ids.len();
        let c = config.num_clusters.min(n).max(2);
        let alpha = config.boundary_alpha;
        let split_threshold = 2 * (n as f64).sqrt().ceil() as usize;

        // 1) K-Means
        let km = kmeans::kmeans(vectors, dims, c, config.kmeans_iters, config.seed, metric);

        // 2) 边界感知分配
        // EN: 2) Boundary-aware assignment.
        let use_boundary = alpha > 0.0;
        let groups: Vec<Vec<(usize, bool)>> = if use_boundary {
            let ba = boundary::assign_with_boundary(vectors, dims, &km.centroids, alpha, metric);
            boundary::group_with_boundary(&ba, c)
        } else {
            let simple_assign = router::assign_vectors(vectors, dims, &km.centroids, metric);
            let simple_groups = router::group_by_cluster(&simple_assign, c);
            simple_groups
                .into_iter()
                .map(|g| g.into_iter().map(|vi| (vi, true)).collect())
                .collect()
        };

        // 3) 并行构建各聚类 HNSW（含复制的边界点）
        // EN: 3) Build each cluster's HNSW in parallel (including copied boundary points).
        let clusters: Vec<RwLock<ClusterGraph>> = groups
            .par_iter()
            .enumerate()
            .map(|(cluster_id, members)| {
                let mut cids = Vec::with_capacity(members.len());
                let mut graph = HnswGraph::new(
                    dims,
                    metric,
                    config.hnsw_m,
                    config.hnsw_ef_construction,
                    config.hnsw_ef_search,
                    config.seed.wrapping_add(cluster_id as u64),
                    10,
                );

                for &(vec_idx, _is_primary) in members {
                    let vid = ids[vec_idx];
                    let v = &vectors[vec_idx * dims..(vec_idx + 1) * dims];
                    cids.push(vid);
                    graph.insert(vid, v.to_vec());
                }

                RwLock::new(ClusterGraph {
                    id: cluster_id,
                    vector_ids: cids,
                    graph,
                    split_threshold,
                })
            })
            .collect();

        let total: usize = clusters.iter().map(|c| c.read().vector_ids.len()).sum();
        let max_id = ids.iter().copied().max().unwrap_or(0);

        ClusteredHnsw {
            dims,
            metric,
            num_clusters: c,
            nprobe: config.nprobe.min(c),
            boundary_alpha: alpha,
            centroids: km.centroids,
            clusters,
            search_nprobe: AtomicUsize::new(0),
            total_indexed: AtomicUsize::new(total),
            next_id: std::sync::atomic::AtomicU64::new(max_id + 1),
        }
    }

    /// 序列化整个聚类 HNSW 构建态（质心 + 每簇图 + 边界复制信息）。
    ///
    /// 复用单图 `graph_snapshot` 的编码：每簇写一份图 payload，恢复时直接
    /// 反序列化，不再逐点重建（100k 冷加载从数秒降到百毫秒级）。乘数（l0/upper）
    /// 与 dtype/量化随图快照写入，环境不一致会在解码时显式拒绝，而不是静默
    /// 得到不同的图语义。
    /// EN: Serialize the whole clustered-HNSW built state (centroids + every
    /// cluster graph + boundary replication). Each cluster reuses the single
    /// graph snapshot encoding so restore decodes instead of re-inserting points
    /// (100k cold load drops from seconds to sub-second). Layer multipliers and
    /// dtype/quantization travel with each graph, so a mismatched environment is
    /// rejected explicitly instead of silently changing graph semantics.
    pub fn encode_state(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(CH_STATE_MAGIC);
        out.extend_from_slice(&CH_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.dims as u32).to_le_bytes());
        out.push(self.metric.to_snapshot_byte());
        out.extend_from_slice(&(self.num_clusters as u32).to_le_bytes());
        out.extend_from_slice(&(self.nprobe as u32).to_le_bytes());
        out.extend_from_slice(&self.boundary_alpha.to_le_bytes());
        out.extend_from_slice(&(self.total_vectors() as u64).to_le_bytes());
        for &c in &self.centroids {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for cluster in &self.clusters {
            let guard = cluster.read();
            out.extend_from_slice(&(guard.vector_ids.len() as u32).to_le_bytes());
            for &id in &guard.vector_ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
            out.extend_from_slice(&(guard.split_threshold as u64).to_le_bytes());
            out.extend_from_slice(&(guard.graph.num_layers as u32).to_le_bytes());
            let payload = guard.graph.encode_snapshot();
            out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            out.extend_from_slice(&payload);
        }
        out
    }

    /// 从构建态字节恢复聚类 HNSW；magic/版本/长度/图 payload 任一不一致即拒绝。
    /// EN: Restore a clustered HNSW from state bytes; bad magic/version/length or
    /// a bad graph payload is rejected.
    pub fn decode_state(bytes: &[u8]) -> Result<Self, String> {
        use crate::graph_snapshot::SnapshotExpectations;
        let mut r = StateReader::new(bytes);
        if r.take(CH_STATE_MAGIC.len())? != CH_STATE_MAGIC {
            return Err("clustered-hnsw state: bad magic".to_string());
        }
        let version = r.u32()?;
        if version != CH_STATE_VERSION {
            return Err(format!(
                "clustered-hnsw state: unsupported version {version}"
            ));
        }
        let dims = r.u32()? as usize;
        let metric_byte = r.u8()?;
        let metric = DistanceMetric::from_snapshot_byte(metric_byte)
            .ok_or_else(|| format!("clustered-hnsw state: bad metric {metric_byte}"))?;
        let num_clusters = r.u32()? as usize;
        let nprobe = r.u32()? as usize;
        let boundary_alpha = r.f32()?;
        let _total = r.u64()?;
        if dims == 0 || dims > 65_536 || num_clusters == 0 || num_clusters > 65_536 {
            return Err("clustered-hnsw state: dims/clusters out of range".to_string());
        }
        if !boundary_alpha.is_finite() {
            return Err("clustered-hnsw state: non-finite boundary alpha".to_string());
        }
        let mut centroids = Vec::with_capacity(num_clusters * dims);
        for _ in 0..num_clusters * dims {
            centroids.push(r.f32()?);
        }
        let (l0_mult, upper_mult) = crate::hnsw_graph::effective_layer_multipliers();
        let mut clusters = Vec::with_capacity(num_clusters);
        for _ in 0..num_clusters {
            let id_count = r.u32()? as usize;
            if id_count > r.remaining() / 8 {
                return Err("clustered-hnsw state: id count exceeds payload".to_string());
            }
            let mut vector_ids = Vec::with_capacity(id_count);
            for _ in 0..id_count {
                vector_ids.push(r.u64()?);
            }
            let split_threshold = r.u64()? as usize;
            let num_layers = r.u32()? as usize;
            let payload_len = usize::try_from(r.u64()?)
                .map_err(|_| "clustered-hnsw state: payload too large".to_string())?;
            if payload_len == 0 || payload_len > r.remaining() {
                return Err("clustered-hnsw state: graph payload out of bounds".to_string());
            }
            let payload = r.take(payload_len)?;
            let expectations = SnapshotExpectations {
                dims,
                metric,
                dtype: VectorDtype::F32,
                quantization: crate::quantization::Quantization::None,
                num_layers,
                l0_mult,
                upper_mult,
            };
            let (graph, consumed) = HnswGraph::decode_snapshot(payload, &expectations)
                .map_err(|e| format!("clustered-hnsw state: {e:?}"))?;
            if consumed != payload.len() {
                return Err("clustered-hnsw state: trailing bytes in graph payload".to_string());
            }
            clusters.push(RwLock::new(ClusterGraph {
                id: clusters.len(),
                vector_ids,
                graph,
                split_threshold,
            }));
        }
        if r.remaining() != 0 {
            return Err("clustered-hnsw state: trailing bytes".to_string());
        }
        let total: usize = clusters.iter().map(|c| c.read().vector_ids.len()).sum();
        Ok(ClusteredHnsw {
            dims,
            metric,
            num_clusters,
            nprobe: nprobe.min(num_clusters),
            boundary_alpha,
            centroids,
            clusters,
            search_nprobe: AtomicUsize::new(0),
            total_indexed: AtomicUsize::new(total),
            next_id: AtomicU64::new(0),
        })
    }

    /// 检索 top-k（自适应 nprobe：固定候选预算，不随数据规模膨胀）。
    /// EN: Search top-k (adaptive nprobe: fixed candidate budget, independent of data size).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(VectorId, f32)> {
        if query.len() != self.dims {
            return Vec::new();
        }
        let k = k.max(1);
        // 自适应 nprobe：候选预算 = k × 2500，感知 top-k 大小
        // k=10  → 搜 ~25K 候选（50K 数据实测 recall 0.999）
        // k=100 → 搜 ~250K 候选（1M 数据预期 recall 0.95+）
        // EN: Adaptive nprobe: candidate budget = k × 2500, aware of top-k size.
        // EN: k=10 → search ~25K candidates (recall 0.999 measured on 50K data).
        // EN: k=100 → search ~250K candidates (recall 0.95+ expected on 1M data).
        let adaptive_nprobe = {
            let total = self.total_vectors();
            let cluster_sz = (total / self.num_clusters).max(1);
            let target = k.saturating_mul(DEFAULT_SEARCH_TARGET_PER_K);
            let base = self.nprobe.max(
                self.search_nprobe
                    .load(Ordering::Relaxed)
                    .min(self.num_clusters),
            );
            base.min((target / cluster_sz).max(4))
        };
        let top_clusters = router::route_query(
            query,
            &self.centroids,
            self.dims,
            adaptive_nprobe,
            self.metric,
        );

        let all_candidates: Vec<(VectorId, f32)> = top_clusters
            .par_iter()
            .flat_map(|&cid| {
                let guard = self.clusters[cid].read();
                guard
                    .graph
                    .search(query, k.saturating_mul(2))
                    .into_par_iter()
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut deduped: Vec<(VectorId, f32)> = {
            let mut seen = ahash::AHashSet::with_capacity(all_candidates.len());
            let mut out = Vec::with_capacity(all_candidates.len());
            for (id, dist) in all_candidates {
                if seen.insert(id) {
                    out.push((id, dist));
                }
            }
            out
        };

        deduped.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        deduped.truncate(k);
        deduped
    }

    /// 插入单个向量（含边界检测 + 自动分裂检查）。
    /// EN: Insert one vector (with boundary detection + automatic split check).
    pub fn insert(&self, id: VectorId, vector: &[f32]) {
        // 插入期与构建期使用同一 alpha、同一度量距离、同一判定函数：任何一侧
        // 单独硬编码（旧插入路径曾固定 alpha=1.2 + 平方 L2 + 乘法判定）都会让
        // IP/Cosine 下运行期写入的复制集合与构建期分叉（OCR 195e6be #04）。
        // EN: Insert time uses the same alpha, metric-aware distances and predicate
        // as build time; hardcoding on either side (the old insert path fixed
        // alpha=1.2 with squared-L2 distances and the multiplicative test) makes
        // runtime writes replicate differently from the build for IP/Cosine
        // (OCR 195e6be #04).
        let alpha = self.boundary_alpha;
        let (primary, secondary_opt) = boundary::find_two_nearest(
            vector,
            &self.centroids,
            self.dims,
            self.num_clusters,
            self.metric,
        );

        // 插入主聚类。
        // EN: Insert into primary.
        {
            let mut guard = self.clusters[primary].write();
            guard.vector_ids.push(id);
            guard.graph.insert(id, vector.to_vec());
        }
        self.total_indexed.fetch_add(1, Ordering::Relaxed);

        // 边界复制。
        // EN: Boundary replication.
        if let Some(sec) = secondary_opt {
            let d_primary = boundary::dist_fn(
                self.metric,
                vector,
                &self.centroids[primary * self.dims..(primary + 1) * self.dims],
            );
            let d_sec = boundary::dist_fn(
                self.metric,
                vector,
                &self.centroids[sec * self.dims..(sec + 1) * self.dims],
            );
            if boundary::is_boundary(d_primary, d_sec, alpha) {
                let mut guard = self.clusters[sec].write();
                guard.vector_ids.push(id);
                guard.graph.insert(id, vector.to_vec());
                self.total_indexed.fetch_add(1, Ordering::Relaxed);
            }
        }

        // 检查主聚类的分裂阈值。
        // EN: Check split threshold for primary cluster.
        let primary_size = {
            let guard = self.clusters[primary].read();
            guard.vector_ids.len()
        };
        let threshold = {
            let guard = self.clusters[0].read();
            guard.split_threshold
        };
        if primary_size > threshold {
            self.split_cluster(primary);
        }
    }

    /// 聚类分裂：当聚类超过 2×√N 阈值时，二分该聚类。
    /// 注意：完全实现需要从 HnswGraph 中导出向量数据。
    /// 当前为占位 — 触发时打印日志，实际分裂待后续版本。
    /// EN: Cluster split: bisect a cluster when it exceeds the 2×√N threshold.
    /// EN: Note: a full implementation needs to export vector data from HnswGraph.
    /// EN: Currently a placeholder — logs on trigger; actual splitting is deferred.
    fn split_cluster(&self, cluster_id: usize) {
        let size = {
            let guard = self.clusters[cluster_id].read();
            guard.vector_ids.len()
        };
        tracing::warn!(
            cluster_id,
            size,
            "[cluster] split_cluster: size exceeds threshold; \
             full impl requires vector export from HnswGraph — skipping"
        );
    }

    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// 运行期 nprobe 覆盖：0 恢复构建期值，其余夹取到 1..=num_clusters。
    /// EN: Runtime nprobe override: 0 restores the build-time value; other
    /// values are clamped to 1..=num_clusters.
    pub fn set_search_nprobe(&self, nprobe: usize) {
        let value = if nprobe == 0 {
            0
        } else {
            nprobe.clamp(1, self.num_clusters)
        };
        self.search_nprobe.store(value, Ordering::Relaxed);
    }

    /// 运行期 ef_search 覆盖：逐簇写入图实例（一次性，热路径零改动）。
    /// EN: Runtime ef_search override: writes each cluster graph once, leaving
    /// the hot search path untouched.
    pub fn set_search_ef_search(&self, ef_search: usize) {
        let ef = ef_search.max(1);
        for cluster in &self.clusters {
            cluster.write().graph.set_ef_search(ef);
        }
    }

    /// 构建期 nprobe。
    /// EN: Build-time nprobe.
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// 向量维度（构建态校验用）。
    /// EN: Vector dimensionality (for built-state validation).
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// 距离度量（构建态校验用）。
    /// EN: Distance metric (for built-state validation).
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn total_vectors(&self) -> usize {
        self.total_indexed.load(Ordering::Relaxed)
    }

    /// 导出全部去重后的原始向量（供 collection 快照使用）。
    /// 边界复制会让同一 id 出现在多个子图中；快照只需一份，按 id 去重。
    /// EN: Exports all unique raw vectors for collection snapshots. Boundary
    /// replication puts the same id in several sub-graphs; the snapshot needs
    /// one copy, so ids are deduplicated.
    pub fn export_vectors(&self) -> Vec<(VectorId, Vec<f32>)> {
        // 锁内只克隆 Arc，锁外再转 f32：逐条转换会长时间阻塞该簇的写入。
        // EN: Clone only Arcs under the lock and convert outside; per-row
        // conversion would block writes to that cluster for the whole export.
        let mut shared = Vec::new();
        for cluster in &self.clusters {
            let guard = cluster.read();
            shared.extend(guard.graph.export_vectors());
        }
        let mut seen = ahash::AHashSet::with_capacity(shared.len());
        let mut out = Vec::with_capacity(shared.len());
        for (id, stored) in shared {
            if seen.insert(id) {
                out.push((id, stored.to_f32_vec()));
            }
        }
        out
    }

    /// 边界复制开销比例（仅构建时计算，非实时）。
    /// EN: Boundary replication overhead ratio (computed at build time only, not live).
    pub fn replication_overhead(&self) -> f64 {
        let unique_ids: usize = self
            .clusters
            .iter()
            .map(|c| c.read().vector_ids.len())
            .sum();
        (unique_ids as f64 - self.total_vectors() as f64).max(0.0) / self.total_vectors() as f64
    }
}

/// 聚类 HNSW 构建态容器格式（每簇图 payload 复用 `graph_snapshot` 的格式）。
/// EN: Clustered-HNSW state container (each cluster payload reuses the
/// `graph_snapshot` format).
const CH_STATE_MAGIC: &[u8; 8] = b"HOUYICH1";
const CH_STATE_VERSION: u32 = 1;

/// 构建态解码的边界安全读取器（越界一律返回 Err，不做无界分配）。
/// EN: Bounds-checked reader for state decoding (out-of-range input returns Err
/// instead of allocating unbounded memory).
struct StateReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> StateReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err("clustered-hnsw state: truncated payload".to_string());
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f32(&mut self) -> Result<f32, String> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceMetric;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn generate_random_vectors(n: usize, dims: usize, seed: u64) -> (Vec<f32>, Vec<u64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            for _ in 0..dims {
                data.push(rng.gen::<f32>());
            }
        }
        (data, ids)
    }

    /// 导出全部去重后的原始向量（边界复制副本只保留一份）。
    /// EN: Exports all unique raw vectors (boundary copies are deduplicated).
    #[test]
    fn test_export_vectors_dedupes_boundary_copies() {
        let dims = 4;
        let n = 100;
        let (vectors, ids) = generate_random_vectors(n, dims, 777);
        let mut config = ClusterConfig::infer(n, dims);
        config.boundary_alpha = 100.0; // 强制几乎所有点复制到次聚类
        let ch = ClusteredHnsw::build(&vectors, &ids, dims, DistanceMetric::L2, &config);
        assert!(
            ch.total_vectors() > n,
            "alpha=100 must replicate: total={}",
            ch.total_vectors()
        );

        let mut exported = ch.export_vectors();
        assert_eq!(exported.len(), n, "export must dedupe boundary copies");
        exported.sort_by_key(|(id, _)| *id);
        for (i, (id, v)) in exported.iter().enumerate() {
            assert_eq!(*id, ids[i]);
            assert_eq!(v.as_slice(), &vectors[i * dims..(i + 1) * dims]);
        }
    }

    /// 构建态往返：同查询的 top-k 必须逐项一致（id 与距离），坏 magic/截断拒绝。
    /// EN: Built-state roundtrip: top-k for the same query must match item for
    /// item (ids and distances); bad magic/truncation are rejected.
    #[test]
    fn test_clustered_state_roundtrip_and_rejection() {
        let dims = 8;
        let n = 400usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            for _ in 0..dims {
                vectors.push(rng.gen::<f32>() * 2.0 - 1.0);
            }
        }
        let cfg = ClusterConfig::infer(n, dims);
        let ch = ClusteredHnsw::build(&vectors, &ids, dims, DistanceMetric::Cosine, &cfg);
        let q = &vectors[..dims];
        let before = ch.search(q, 10);
        assert_eq!(before.len(), 10);

        let state = ch.encode_state();
        let restored = ClusteredHnsw::decode_state(&state).expect("state must decode");
        assert_eq!(restored.total_vectors(), ch.total_vectors());
        assert_eq!(restored.dims(), dims);
        assert_eq!(restored.metric(), DistanceMetric::Cosine);
        let after = restored.search(q, 10);
        assert_eq!(after.len(), before.len());
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.0, b.0, "ids must match");
            assert!(
                (a.1 - b.1).abs() < 1.0e-6,
                "distances must match: {} vs {}",
                a.1,
                b.1
            );
        }

        let mut bad_magic = state.clone();
        bad_magic[0] ^= 0xFF;
        assert!(ClusteredHnsw::decode_state(&bad_magic).is_err());
        assert!(ClusteredHnsw::decode_state(&state[..state.len() - 1]).is_err());
        assert!(ClusteredHnsw::decode_state(&state[..16]).is_err());
    }

    #[test]
    fn test_clustered_build_and_search() {
        let dims = 16;
        let n = 500;
        let (vectors, ids) = generate_random_vectors(n, dims, 42);
        let config = ClusterConfig::infer(n, dims);

        let ch = ClusteredHnsw::build(&vectors, &ids, dims, DistanceMetric::L2, &config);
        assert!(ch.total_vectors() >= n); // 含边界复制可能 > n

        let mut found_self = 0;
        for i in 0..n.min(50) {
            let q = &vectors[i * dims..(i + 1) * dims];
            let results = ch.search(q, 1);
            if results.first().map(|r| r.0) == Some(ids[i]) {
                found_self += 1;
            }
        }
        let self_recall = found_self as f32 / 50.0;
        assert!(self_recall > 0.7, "self-recall too low: {:.2}", self_recall);
    }

    #[test]
    fn test_clustered_insert_and_search() {
        let dims = 8;
        let n = 200;
        let (vectors, ids) = generate_random_vectors(n, dims, 99);
        let config = ClusterConfig::infer(n, dims);

        let ch = ClusteredHnsw::build(&vectors, &ids, dims, DistanceMetric::L2, &config);

        let new_id = 9999;
        let new_vec: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();
        ch.insert(new_id, &new_vec);
        assert!(ch.total_vectors() > n);

        let results = ch.search(&new_vec, 1);
        assert_eq!(results.first().map(|r| r.0), Some(new_id));
    }

    #[test]
    fn test_clustered_recall_vs_flat() {
        let dims = 32;
        let n = 300;
        let k = 10;
        let (vectors, ids) = generate_random_vectors(n, dims, 0xBEEF);
        let config = ClusterConfig::infer(n, dims);

        let ch = ClusteredHnsw::build(&vectors, &ids, dims, DistanceMetric::L2, &config);

        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let mut total_recall = 0.0_f32;
        let queries = 30;

        for _ in 0..queries {
            let q: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
            let mut brute: Vec<(u64, f32)> = (0..n)
                .map(|i| {
                    let v = &vectors[i * dims..(i + 1) * dims];
                    let d: f32 = q
                        .iter()
                        .zip(v.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum();
                    (ids[i], d)
                })
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let truth: ahash::AHashSet<u64> = brute.iter().take(k).map(|(id, _)| *id).collect();
            let pred: ahash::AHashSet<u64> =
                ch.search(&q, k).into_iter().map(|(id, _)| id).collect();

            let hit = pred.intersection(&truth).count();
            total_recall += hit as f32 / k as f32;
        }

        let avg_recall = total_recall / queries as f32;
        if avg_recall < 0.9 {
            panic!("clustered recall@{k} too low: {avg_recall:.3}");
        }
    }

    #[test]
    fn test_insert_boundary_replication_honors_alpha_and_metric() {
        // 插入期必须复用构建期的 alpha/度量/判定：旧实现固定 alpha=1.2、用
        // 平方 L2 距离和乘法判定，IP 下会漏复制或复制错聚类
        // （OCR 195e6be #04）。
        // EN: Insert time must reuse the build-time alpha/metric/predicate: the
        // old implementation fixed alpha=1.2, used squared-L2 distances and the
        // multiplicative test, so IP either missed replication or copied the
        // wrong secondary (OCR 195e6be #04).
        let dims = 2;
        let make = |metric: DistanceMetric, alpha: f32| {
            let clusters = (0..2)
                .map(|id| {
                    RwLock::new(ClusterGraph {
                        id,
                        vector_ids: Vec::new(),
                        graph: HnswGraph::new(dims, metric, 16, 200, 128, id as u64, 10),
                        split_threshold: usize::MAX,
                    })
                })
                .collect();
            ClusteredHnsw {
                dims,
                metric,
                num_clusters: 2,
                nprobe: 2,
                boundary_alpha: alpha,
                // c0=(1,0)、c1=(0.9,0.1)：v=(1,0) 时 IP 距离 d0=-1、d1=-0.9。
                // EN: c0=(1,0), c1=(0.9,0.1): for v=(1,0), IP distances are d0=-1 and d1=-0.9.
                centroids: vec![1.0, 0.0, 0.9, 0.1],
                clusters,
                search_nprobe: AtomicUsize::new(0),
                total_indexed: AtomicUsize::new(0),
                next_id: std::sync::atomic::AtomicU64::new(1),
            }
        };

        // IP：加性余量 α=1.2 → 阈值 -0.8，d_sec=-0.9 命中，应复制到次聚类。
        // EN: IP with the additive margin α=1.2 gives threshold -0.8; d_sec=-0.9
        // hits it and must be replicated to the secondary cluster.
        let ip = make(DistanceMetric::InnerProduct, 1.2);
        ip.insert(1, &[1.0, 0.0]);
        assert_eq!(ip.total_vectors(), 2, "IP boundary point must replicate");
        assert!(ip.clusters[1].read().vector_ids.contains(&1));

        // α=0 关闭复制：插入期必须遵循配置值，不能回退到硬编码 1.2。
        // EN: α=0 disables replication: insert must honor the configured value
        // instead of falling back to hardcoded 1.2.
        let off = make(DistanceMetric::InnerProduct, 0.0);
        off.insert(1, &[1.0, 0.0]);
        assert_eq!(off.total_vectors(), 1);
        assert!(off.clusters[1].read().vector_ids.is_empty());

        // L2 并行性：服务端 build 的 α 恒为 0，真正走度量感知插入分支的是
        // L2/Cosine 等非 IP 度量；断言插入期与 assign_with_boundary 的复制
        // 判定逐例一致（OCR cf1d19f #04）。
        // v=(0,1)：d0=2.0、d1=1.62 → 主聚类 c1。α=1.3 → 阈值 2.106 ≥ 2.0
        // 复制；α=1.2 → 阈值 1.944 < 2.0 不复制。
        // EN: L2 parity: the server always builds with α=0, so the only metrics
        // that actually exercise the metric-aware insert branch are non-IP ones
        // such as L2/Cosine; assert the insert decision matches
        // `assign_with_boundary` case by case (OCR cf1d19f #04). For v=(0,1):
        // d0=2.0, d1=1.62 → primary c1. α=1.3 gives threshold 2.106 ≥ 2.0
        // (replicate); α=1.2 gives 1.944 < 2.0 (no replication).
        for (alpha, expect_replicate) in [(1.3, true), (1.2, false)] {
            let built = boundary::assign_with_boundary(
                &[0.0, 1.0],
                2,
                &[1.0, 0.0, 0.9, 0.1],
                alpha,
                DistanceMetric::L2,
            );
            assert_eq!(built[0].primary, 1, "L2 primary must match build time");
            let expected_secondaries: Vec<usize> = if expect_replicate {
                vec![0]
            } else {
                Vec::new()
            };
            assert_eq!(
                built[0].secondaries, expected_secondaries,
                "build-time secondaries at α={alpha}"
            );

            let ch = make(DistanceMetric::L2, alpha);
            ch.insert(1, &[0.0, 1.0]);
            assert_eq!(
                ch.clusters[0].read().vector_ids.contains(&1),
                expect_replicate,
                "insert must mirror build-time replication at α={alpha}"
            );
            assert_eq!(ch.total_vectors(), if expect_replicate { 2 } else { 1 });
        }
    }

    #[test]
    fn test_boundary_replication_increases_total_count() {
        let dims = 4;
        let n = 100;
        let (vectors, ids) = generate_random_vectors(n, dims, 777);

        // 带边界复制构建。
        // EN: build with boundary replication.
        let mut config_with = ClusterConfig::infer(n, dims);
        config_with.boundary_alpha = 1.5; // aggressive → more replication
        let ch_with = ClusteredHnsw::build(&vectors, &ids, dims, DistanceMetric::L2, &config_with);

        // 不带边界复制构建。
        // EN: build without boundary replication.
        let mut config_without = ClusterConfig::infer(n, dims);
        config_without.boundary_alpha = 0.0; // disabled
        let ch_without =
            ClusteredHnsw::build(&vectors, &ids, dims, DistanceMetric::L2, &config_without);

        // 带边界复制的计数应 ≥ 总数。
        // EN: with boundary should have ≥ total counts.
        assert!(ch_with.total_vectors() >= ch_without.total_vectors());
    }
}
