//! FlatClustered: K-Means 分区 + 每区 SIMD 暴力检索（无 HNSW 子图）。
//! K-Means partitioning + per-cluster brute-force SIMD search (no HNSW sub-graphs).
//!
//! 适合: 每分区向量数 < 5K、频繁全量更新、NUMA 多节点部署。
//! Ideal for: <5K vectors per cluster, frequent full reloads, NUMA-aware deploys.

use super::boundary::dist_fn;
use super::kmeans;
use super::router;
use crate::distance::dot_product_f32;
use crate::DistanceMetric;
use crate::VectorDtype;
use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;

/// 默认候选预算（每 k）：k=10 时约 25k 候选，保持历史召回。
/// EN: Default candidate budget per k: ~25k candidates at k=10, preserving
/// historical recall.
pub const DEFAULT_SEARCH_TARGET_PER_K: usize = 2500;
#[inline]
fn inv_norm_from_norm2(norm2: f32) -> f32 {
    if norm2 > 0.0 {
        1.0 / norm2.sqrt()
    } else {
        0.0
    }
}

/// flat-clustered 簇内存储编码：f32 / f16 / i8（i8 额外每向量 4 B scale）。
/// EN: Flat-clustered per-cluster storage encoding: f32 / f16 / i8 (i8 adds a
/// 4-byte scale per vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlatStorageDtype {
    /// 全精度 f32。
    /// EN: Full-precision f32.
    F32,
    /// 半精度 IEEE binary16 格式。
    /// EN: Half-precision IEEE binary16.
    F16,
    /// 每向量对称 int8（每维 1 B + 每向量 4 B scale）。
    /// EN: Per-vector symmetric int8 (1 B per dimension + 4 B scale per vector).
    I8,
}

impl FlatStorageDtype {
    pub fn from_vector_dtype(dtype: VectorDtype) -> Self {
        match dtype {
            VectorDtype::F32 => FlatStorageDtype::F32,
            VectorDtype::F16 => FlatStorageDtype::F16,
        }
    }

    pub fn to_snapshot_byte(self) -> u8 {
        match self {
            FlatStorageDtype::F32 => 0,
            FlatStorageDtype::F16 => 1,
            FlatStorageDtype::I8 => 2,
        }
    }

    pub fn from_snapshot_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(FlatStorageDtype::F32),
            1 => Some(FlatStorageDtype::F16),
            2 => Some(FlatStorageDtype::I8),
            _ => None,
        }
    }

    /// 每维字节数（i8 另计每向量 scale）。
    /// EN: Bytes per dimension (i8 adds a per-vector scale separately).
    pub fn bytes_per_dim(self) -> usize {
        match self {
            FlatStorageDtype::F32 => 4,
            FlatStorageDtype::F16 => 2,
            FlatStorageDtype::I8 => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            FlatStorageDtype::F32 => "f32",
            FlatStorageDtype::F16 => "f16",
            FlatStorageDtype::I8 => "i8",
        }
    }
}

/// 聚类内向量存储：f32 或 f16（移动端用 f16 减半内存；服务端默认仍走 f32）。
/// EN: Per-cluster vector storage: f32 or f16 (mobile uses f16 to halve memory;
/// the server default stays f32).
enum ClusterVectors {
    F32(Vec<Vec<f32>>),
    F16(Vec<Vec<f16>>),
    /// i8 对称量化：codes 每维 1 B，scales 每向量 1 个 f32。
    /// EN: Symmetric i8 quantization: 1 B per dimension plus one f32 scale per
    /// vector.
    I8 {
        codes: Vec<Vec<i8>>,
        scales: Vec<Vec<f32>>,
    },
}

/// f16 量化时饱和到最大有限值，避免 ±inf 让距离变成 inf/NaN（与 StoredVector 同一规则）。
/// EN: Saturate f16 quantization to the largest finite value so ±inf cannot
/// make distances inf/NaN (same rule as StoredVector).
#[inline]
fn to_f16(x: f32) -> f16 {
    f16::from_f32(x.clamp(-65504.0, 65504.0))
}

/// 每向量对称 i8 量化：scale = max|x| / 127；全零向量 scale=0。
/// EN: Per-vector symmetric i8 quantization: scale = max|x| / 127; an all-zero
/// vector uses scale 0.
fn quantize_i8(vector: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = vector.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
    if max_abs == 0.0 {
        return (vec![0; vector.len()], 0.0);
    }
    let scale = max_abs / 127.0;
    let inv = 1.0 / scale;
    let codes = vector
        .iter()
        .map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (codes, scale)
}

impl ClusterVectors {
    fn clusters(&self) -> usize {
        match self {
            ClusterVectors::F32(v) => v.len(),
            ClusterVectors::F16(v) => v.len(),
            ClusterVectors::I8 { codes, .. } => codes.len(),
        }
    }

    fn cluster_len(&self, cluster: usize) -> usize {
        match self {
            ClusterVectors::F32(v) => v[cluster].len(),
            ClusterVectors::F16(v) => v[cluster].len(),
            ClusterVectors::I8 { codes, .. } => codes[cluster].len(),
        }
    }

    fn dtype(&self) -> FlatStorageDtype {
        match self {
            ClusterVectors::F32(_) => FlatStorageDtype::F32,
            ClusterVectors::F16(_) => FlatStorageDtype::F16,
            ClusterVectors::I8 { .. } => FlatStorageDtype::I8,
        }
    }

    fn push(&mut self, cluster: usize, vector: &[f32]) {
        match self {
            ClusterVectors::F32(v) => v[cluster].extend_from_slice(vector),
            ClusterVectors::F16(v) => v[cluster].extend(vector.iter().map(|&x| to_f16(x))),
            ClusterVectors::I8 { codes, scales } => {
                let (q, scale) = quantize_i8(vector);
                codes[cluster].extend(q);
                scales[cluster].push(scale);
            }
        }
    }

    fn overwrite(&mut self, cluster: usize, slot: usize, dims: usize, vector: &[f32]) {
        match self {
            ClusterVectors::F32(v) => {
                let start = slot * dims;
                v[cluster][start..start + dims].copy_from_slice(vector);
            }
            ClusterVectors::F16(v) => {
                debug_assert_eq!(vector.len(), dims);
                let start = slot * dims;
                let dst = &mut v[cluster][start..start + dims];
                for (dst, &src) in dst.iter_mut().zip(vector) {
                    *dst = to_f16(src);
                }
            }
            ClusterVectors::I8 { codes, scales } => {
                let (q, scale) = quantize_i8(vector);
                let start = slot * dims;
                codes[cluster][start..start + dims].copy_from_slice(&q);
                scales[cluster][slot] = scale;
            }
        }
    }

    /// 删除槽位：把该簇最后一个向量块搬到 slot 并截断（与 ids.swap_remove 同步）。
    /// EN: Remove a slot by moving the cluster's last vector block into it and
    /// truncating (in lockstep with ids.swap_remove).
    fn remove_slot(&mut self, cluster: usize, slot: usize, dims: usize) {
        match self {
            ClusterVectors::F32(v) => {
                let last = v[cluster].len() / dims - 1;
                let last_start = last * dims;
                let start = slot * dims;
                v[cluster].copy_within(last_start..last_start + dims, start);
                v[cluster].truncate(last * dims);
            }
            ClusterVectors::F16(v) => {
                let last = v[cluster].len() / dims - 1;
                let last_start = last * dims;
                let start = slot * dims;
                v[cluster].copy_within(last_start..last_start + dims, start);
                v[cluster].truncate(last * dims);
            }
            ClusterVectors::I8 { codes, scales } => {
                let last = codes[cluster].len() / dims - 1;
                let last_start = last * dims;
                let start = slot * dims;
                codes[cluster].copy_within(last_start..last_start + dims, start);
                codes[cluster].truncate(last * dims);
                scales[cluster].swap_remove(slot);
            }
        }
    }

    fn export_row(&self, cluster: usize, slot: usize, dims: usize) -> Vec<f32> {
        let start = slot * dims;
        match self {
            ClusterVectors::F32(v) => v[cluster][start..start + dims].to_vec(),
            ClusterVectors::F16(v) => v[cluster][start..start + dims]
                .iter()
                .map(|x| x.to_f32())
                .collect(),
            ClusterVectors::I8 { codes, scales } => {
                let scale = scales[cluster][slot];
                codes[cluster][start..start + dims]
                    .iter()
                    .map(|&x| x as f32 * scale)
                    .collect()
            }
        }
    }

    /// 第 (cluster, slot) 个向量的平方 L2 范数（按实际存储编码计算）。
    /// EN: Squared L2 norm of the vector at (cluster, slot), computed from the
    /// actual stored encoding.
    fn norm2(&self, cluster: usize, slot: usize, dims: usize) -> f32 {
        let start = slot * dims;
        match self {
            ClusterVectors::F32(v) => {
                let row = &v[cluster][start..start + dims];
                row.iter().map(|x| x * x).sum()
            }
            ClusterVectors::F16(v) => {
                let row = &v[cluster][start..start + dims];
                row.iter()
                    .map(|x| {
                        let x = x.to_f32();
                        x * x
                    })
                    .sum()
            }
            ClusterVectors::I8 { codes, scales } => {
                let scale = scales[cluster][slot];
                let ss: i32 = codes[cluster][start..start + dims]
                    .iter()
                    .map(|&x| (x as i32) * (x as i32))
                    .sum();
                scale * scale * ss as f32
            }
        }
    }
}

/// i8 量化向量的点积：先算 `Σ q[i] * code[i]`，最后乘以该向量的 scale。
/// aarch64+NEON 用 i8→i16→i32→f32 宽化转换与 FMA（每次 8 维）；其他平台
/// 用标量循环。仅用于 i8 余弦缓存的点积热路径。
/// EN: Dot product for an i8-quantized vector: compute sum(q[i] * code[i]) first,
/// then multiply by the vector scale. aarch64+NEON widens i8→i16→i32→f32 and
/// uses FMA (8 dimensions per iteration); other targets use a scalar loop.
/// Used only by the i8 cosine-cache dot-product path.
#[inline]
fn dot_i8_scaled(query: &[f32], codes: &[i8], scale: f32) -> f32 {
    let n = query.len().min(codes.len());
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // aarch64 上 NEON 是强制要求，无需运行时检测。
        // EN: NEON is mandatory on aarch64; no runtime check needed.
        return unsafe { dot_i8_scaled_neon(query, codes, scale, n) };
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += query[i] * codes[i] as f32;
        }
        sum * scale
    }
}

#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
unsafe fn dot_i8_scaled_neon(query: &[f32], codes: &[i8], scale: f32, n: usize) -> f32 {
    use std::arch::aarch64::*;
    // 防御性夹取：即使调用方传入超界 n 也不会越界读。
    // EN: Defensive clamp: an out-of-range n cannot cause out-of-bounds reads.
    let n = n.min(query.len()).min(codes.len());
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        // SAFETY：边界已检查，切片指针有效。
        // EN: SAFETY: bounds checked; slice pointers are valid.
        let c = vld1_s8(codes.as_ptr().add(i));
        let lo16 = vmovl_s8(c);
        let lo32 = vmovl_s16(vget_low_s16(lo16));
        let hi32 = vmovl_s16(vget_high_s16(lo16));
        let qlo = vld1q_f32(query.as_ptr().add(i));
        let qhi = vld1q_f32(query.as_ptr().add(i + 4));
        acc = vfmaq_f32(
            vfmaq_f32(acc, vcvtq_f32_s32(lo32), qlo),
            vcvtq_f32_s32(hi32),
            qhi,
        );
        i += 8;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += query[i] * codes[i] as f32;
        i += 1;
    }
    sum * scale
}

/// 余弦检索的预计算范数缓存：把每个候选的 `‖v‖` 与每个质心的 `‖c‖`
/// 在构建期算好，检索时只做一次点积 + 标量乘法。代价是每向量 4 B
/// （f16/i8 100k 约 0.4 MB），只对 f16/i8 余弦路径启用，服务端默认 f32
/// 路径完全不变。
/// EN: Precomputed norms for cosine search: per-candidate `‖v‖` and per-centroid
/// `‖c‖` are computed at build time so each query does one dot product plus a
/// scalar multiply. Cost is 4 B per vector (~0.4 MB at 100k f16/i8); enabled
/// only for the f16/i8 cosine paths, leaving the server default f32 path
/// untouched.
struct CosineNormCache {
    inv_vec_norms: Vec<Vec<f32>>,
}

impl CosineNormCache {
    fn build(vecs: &ClusterVectors, dims: usize) -> Self {
        let mut inv_vec_norms = Vec::new();
        for cluster in 0..vecs.clusters() {
            let count = vecs.cluster_len(cluster) / dims;
            let mut inv = Vec::with_capacity(count);
            for slot in 0..count {
                let norm2 = vecs.norm2(cluster, slot, dims);
                inv.push(inv_norm_from_norm2(norm2));
            }
            inv_vec_norms.push(inv);
        }
        CosineNormCache { inv_vec_norms }
    }

    fn set_norm2(&mut self, cluster: usize, slot: usize, norm2: f32) {
        self.inv_vec_norms[cluster][slot] = inv_norm_from_norm2(norm2);
    }

    fn push_norm2(&mut self, cluster: usize, norm2: f32) {
        self.inv_vec_norms[cluster].push(inv_norm_from_norm2(norm2));
    }

    fn remove_slot(&mut self, cluster: usize, slot: usize) {
        self.inv_vec_norms[cluster].swap_remove(slot);
    }
}

/// Flat 聚类索引：分 C 个聚类，每聚类平坦存储向量 + 暴力检索。
/// EN: Flat cluster index: C clusters, each storing vectors flatly with brute-force search.
pub struct FlatClustered {
    dims: usize,
    metric: DistanceMetric,
    num_clusters: usize,
    nprobe: usize,

    centroids: Vec<f32>,
    /// 每聚类的向量 ID
    /// EN: Vector ids per cluster.
    cluster_ids: Vec<Vec<u64>>,
    /// 每聚类的平坦向量数据 [cluster_size × dims]（f32 或 f16）。
    /// EN: Flat vector data per cluster [cluster_size × dims], f32 or f16.
    cluster_vecs: ClusterVectors,
    /// id → (聚类, 槽位) 映射；仅在首次增量插入时惰性建立，构建路径不付
    /// 这份内存（服务端平簇后端仍只批量构建）。
    /// EN: id -> (cluster, slot) map built lazily on the first incremental
    /// insert; the batch build path does not pay this memory (server
    /// flat-clustered collections only build in bulk).
    id_index: Option<ahash::AHashMap<u64, (usize, usize)>>,
    /// 重复 id 等不可恢复的增量错误；缓存后让后续 insert 立即失败，避免
    /// 每批重试都 O(n) 重建映射（只有重建索引才能恢复）。
    /// EN: Sticky incremental-insert error (e.g. duplicate ids); caching it makes
    /// later inserts fail immediately instead of re-scanning O(n) per retry
    /// (only a rebuilt index can recover).
    id_index_error: Option<String>,
    /// f16 余弦路径的预计算范数缓存（f32/其他度量不启用，服务端默认路径零开销）。
    /// EN: Precomputed norm cache for the f16 cosine path (not enabled for f32 or
    /// other metrics, so the server default path pays nothing).
    cosine_norms: Option<CosineNormCache>,
}

impl FlatClustered {
    pub fn build(
        vectors: &[f32],
        ids: &[u64],
        dims: usize,
        metric: DistanceMetric,
        num_clusters: usize,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        // 服务端默认 f32 构建不启用余弦范数缓存：保持既有内存/性能基线。
        // EN: The server's default f32 build does not enable the cosine norm
        // cache, preserving the existing memory/performance baseline.
        Self::build_internal(
            vectors,
            ids,
            dims,
            metric,
            num_clusters,
            kmeans_iters,
            seed,
            FlatStorageDtype::F32,
            false,
        )
    }

    /// 与 [`build`] 相同，但可指定簇内存储编码；f16 + Cosine 会启用范数缓存
    /// （f32 实测受内存带宽限制反而更慢，不启用）。供移动端按 dtype 选择。
    /// EN: Same as [`build`] with an explicit per-cluster storage encoding;
    /// f16 + Cosine enables the norm cache (f32 is measurably slower due to
    /// memory bandwidth, so it stays uncached), letting mobile choose a dtype.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_dtype(
        vectors: &[f32],
        ids: &[u64],
        dims: usize,
        metric: DistanceMetric,
        num_clusters: usize,
        kmeans_iters: usize,
        seed: u64,
        dtype: VectorDtype,
    ) -> Self {
        Self::build_internal(
            vectors,
            ids,
            dims,
            metric,
            num_clusters,
            kmeans_iters,
            seed,
            FlatStorageDtype::from_vector_dtype(dtype),
            true,
        )
    }

    /// 显式存储编码版本（支持 i8）；供移动端选择 f32/f16/i8。
    /// EN: Explicit storage encoding (including i8) for mobile f32/f16/i8
    /// selection.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_storage_dtype(
        vectors: &[f32],
        ids: &[u64],
        dims: usize,
        metric: DistanceMetric,
        num_clusters: usize,
        kmeans_iters: usize,
        seed: u64,
        dtype: FlatStorageDtype,
    ) -> Self {
        Self::build_internal(
            vectors,
            ids,
            dims,
            metric,
            num_clusters,
            kmeans_iters,
            seed,
            dtype,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_internal(
        vectors: &[f32],
        ids: &[u64],
        dims: usize,
        metric: DistanceMetric,
        num_clusters: usize,
        kmeans_iters: usize,
        seed: u64,
        dtype: FlatStorageDtype,
        cache_norms: bool,
    ) -> Self {
        assert_eq!(vectors.len(), ids.len() * dims);
        let n = ids.len();
        let c = num_clusters.min(n).max(1);

        // 1) K-Means
        let km = kmeans::kmeans(vectors, dims, c, kmeans_iters, seed, metric);

        // 2) 分配向量
        // EN: 2) Assign vectors.
        let assignments = router::assign_vectors(vectors, dims, &km.centroids, metric);
        let groups = router::group_by_cluster(&assignments, c);

        // 3) 按聚类整理数据（f32 保留原批量 extend 路径，避免服务端热路径
        // 因新增 dtype 分支而变慢）。
        // EN: 3) Organize data by cluster (the f32 path keeps the original bulk
        // extend so the server hot path does not slow down for the new dtype).
        let mut cluster_ids: Vec<Vec<u64>> = (0..c).map(|_| Vec::new()).collect();
        let cluster_vecs = match dtype {
            FlatStorageDtype::F32 => {
                let mut cluster_vecs: Vec<Vec<f32>> = (0..c).map(|_| Vec::new()).collect();
                for (cluster_id, vec_indices) in groups.iter().enumerate() {
                    let mut ids_list = Vec::with_capacity(vec_indices.len());
                    let mut vec_list = Vec::with_capacity(vec_indices.len() * dims);
                    for &vi in vec_indices {
                        ids_list.push(ids[vi]);
                        let start = vi * dims;
                        vec_list.extend_from_slice(&vectors[start..start + dims]);
                    }
                    cluster_ids[cluster_id] = ids_list;
                    cluster_vecs[cluster_id] = vec_list;
                }
                ClusterVectors::F32(cluster_vecs)
            }
            FlatStorageDtype::F16 => {
                let mut cluster_vecs: Vec<Vec<f16>> = (0..c).map(|_| Vec::new()).collect();
                for (cluster_id, vec_indices) in groups.iter().enumerate() {
                    let mut ids_list = Vec::with_capacity(vec_indices.len());
                    let mut vec_list = Vec::with_capacity(vec_indices.len() * dims);
                    for &vi in vec_indices {
                        ids_list.push(ids[vi]);
                        let start = vi * dims;
                        vec_list.extend(vectors[start..start + dims].iter().map(|&x| to_f16(x)));
                    }
                    cluster_ids[cluster_id] = ids_list;
                    cluster_vecs[cluster_id] = vec_list;
                }
                ClusterVectors::F16(cluster_vecs)
            }
            FlatStorageDtype::I8 => {
                let mut codes: Vec<Vec<i8>> = (0..c).map(|_| Vec::new()).collect();
                let mut scales: Vec<Vec<f32>> = (0..c).map(|_| Vec::new()).collect();
                for (cluster_id, vec_indices) in groups.iter().enumerate() {
                    let mut ids_list = Vec::with_capacity(vec_indices.len());
                    let mut code_list = Vec::with_capacity(vec_indices.len() * dims);
                    let mut scale_list = Vec::with_capacity(vec_indices.len());
                    for &vi in vec_indices {
                        ids_list.push(ids[vi]);
                        let start = vi * dims;
                        let (q, scale) = quantize_i8(&vectors[start..start + dims]);
                        code_list.extend(q);
                        scale_list.push(scale);
                    }
                    cluster_ids[cluster_id] = ids_list;
                    codes[cluster_id] = code_list;
                    scales[cluster_id] = scale_list;
                }
                ClusterVectors::I8 { codes, scales }
            }
        };

        // nprobe: 自适应公式（与 ClusteredHnsw 一致）
        // 下界不得超过上界：c ∈ {2,3}（小数据集）时 clamp(4, c) 会 panic。
        // EN: lower bound must not exceed upper bound — clamp(4, c) panics for c < 4.
        let cluster_sz = n / c;
        let target = DEFAULT_SEARCH_TARGET_PER_K.saturating_mul(10); // k=10 预算
        let nprobe = (target / cluster_sz.max(1)).clamp(4.min(c), c);

        let mut index = FlatClustered {
            dims,
            metric,
            num_clusters: c,
            nprobe,
            centroids: km.centroids,
            cluster_ids,
            cluster_vecs,
            id_index: None,
            id_index_error: None,
            cosine_norms: None,
        };
        if cache_norms {
            index.rebuild_cosine_norms();
        }
        index
    }

    /// 重建余弦范数缓存（仅 f16 + cosine 启用）。构建、解码与增量写入后调用。
    /// EN: Rebuild the cosine norm cache (f16 + cosine only). Called after a
    /// build, a state decode, and incremental writes.
    fn rebuild_cosine_norms(&mut self) {
        self.cosine_norms = if self.metric == DistanceMetric::Cosine
            && matches!(
                self.cluster_vecs.dtype(),
                FlatStorageDtype::F16 | FlatStorageDtype::I8
            ) {
            Some(CosineNormCache::build(&self.cluster_vecs, self.dims))
        } else {
            None
        };
    }

    /// 簇内存储编码（f32/f16）。
    /// EN: Per-cluster storage encoding (f32/f16).
    pub fn storage_dtype(&self) -> FlatStorageDtype {
        self.cluster_vecs.dtype()
    }

    /// 索引使用的距离度量。
    /// EN: Distance metric used by the index.
    pub fn vector_metric(&self) -> DistanceMetric {
        self.metric
    }

    /// 构建态编码里的维度（持久化加载时与集合维度交叉校验）。
    /// EN: Dimensions inside the built-state encoding (cross-checked against
    /// collection dims on load).
    pub fn state_dims(&self) -> usize {
        self.dims
    }

    /// 增量插入/覆盖一个向量（按最近质心归簇），插入后立即可被检索命中。
    /// upsert 语义：同 id 的旧向量被移除（必要时跨簇搬移），总计数不变。
    ///
    /// 为什么惰性建映射：批量构建是服务端与首次导入的主路径，额外维护一份
    /// id→位置表会让每个平簇集合多付约 40 B/向量；移动端在首次增量插入时才
    /// 扫描一次分簇块建立映射，随后 O(1) 更新。
    /// EN: Incrementally insert or overwrite a vector (assigned to the nearest
    /// centroid); it becomes searchable immediately. Upsert semantics: an old
    /// row with the same id is removed (moving clusters when needed) so counts
    /// stay exact.
    ///
    /// WHY lazy map construction: bulk build is the main path for servers and
    /// first imports; maintaining an id->position table would cost ~40 B per
    /// vector for every flat-clustered collection. Mobile builds the map once
    /// by scanning the cluster blocks on the first incremental insert, then
    /// updates it in O(1).
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dims {
            return Err(format!(
                "dim mismatch: expected {}, got {}",
                self.dims,
                vector.len()
            ));
        }
        if self.num_clusters == 0 {
            return Err("flat-clustered index has no clusters".to_string());
        }
        if let Some(message) = &self.id_index_error {
            return Err(message.clone());
        }
        let target = *router::route_query(vector, &self.centroids, self.dims, 1, self.metric)
            .first()
            .expect("route_query returns one cluster when num_clusters > 0");
        if self.id_index.is_none() {
            self.build_id_index()?;
        }
        let map = self.id_index.as_mut().expect("id index built above");

        if let Some(&(old_cluster, slot)) = map.get(&id) {
            // 槽位必须真实指向该 id：映射与分簇块一旦失步，继续写会覆盖
            // 别的向量（比返回错误更隐蔽）。这里显式拒绝并要求 rebuild。
            // EN: The slot must actually hold this id; if the map and cluster
            // blocks desync, continuing would overwrite another vector (more
            // insidious than failing). Reject and require a rebuild.
            let valid = self
                .cluster_ids
                .get(old_cluster)
                .and_then(|ids| ids.get(slot))
                .is_some_and(|existing| *existing == id);
            if !valid {
                return Err(format!(
                    "flat-clustered id map desync for id {id} (cluster {old_cluster}, slot {slot}); rebuild required"
                ));
            }
            if old_cluster == target {
                self.cluster_vecs
                    .overwrite(old_cluster, slot, self.dims, vector);
                if let Some(cache) = self.cosine_norms.as_mut() {
                    let norm2 = self.cluster_vecs.norm2(old_cluster, slot, self.dims);
                    cache.set_norm2(old_cluster, slot, norm2);
                }
                return Ok(());
            }
            // 跨簇搬移：swap_remove 保持其余槽位连续，并把被交换的尾部 id
            // 的位置同步到映射，避免悬空槽位被后续检索读到。
            // EN: Cross-cluster move: swap_remove keeps the remaining slots
            // contiguous and re-points the swapped tail id in the map so no
            // dangling slot survives for later searches.
            let moved_id = {
                let ids = &mut self.cluster_ids[old_cluster];
                let last = ids.len() - 1;
                ids.swap_remove(slot);
                (slot != last).then(|| ids[slot])
            };
            self.cluster_vecs.remove_slot(old_cluster, slot, self.dims);
            if let Some(cache) = self.cosine_norms.as_mut() {
                cache.remove_slot(old_cluster, slot);
            }
            if let Some(moved_id) = moved_id {
                map.insert(moved_id, (old_cluster, slot));
            }
        }

        self.cluster_ids[target].push(id);
        self.cluster_vecs.push(target, vector);
        let slot = self.cluster_ids[target].len() - 1;
        if let Some(cache) = self.cosine_norms.as_mut() {
            let norm2 = self.cluster_vecs.norm2(target, slot, self.dims);
            cache.push_norm2(target, norm2);
        }
        map.insert(id, (target, slot));
        Ok(())
    }

    /// 扫描当前分簇块建立 id→位置映射（仅在首次增量插入时执行）。
    /// 重复 id 会让「同 id 只应有一个位置」的增量语义不成立（旧的副本会
    /// 继续被检索到）；这里拒绝建图并要求调用方先 rebuild/去重，而不是
    /// 静默保留最后一个位置。
    /// EN: Scan the cluster blocks to build the id->position map (first
    /// incremental insert only). Duplicate ids break the one-position-per-id
    /// invariant (a stale copy would stay searchable); reject the index and
    /// require a rebuild/dedup instead of silently keeping the last slot.
    fn build_id_index(&mut self) -> Result<(), String> {
        let mut map = ahash::AHashMap::with_capacity(self.total_vectors());
        for (cluster, ids) in self.cluster_ids.iter().enumerate() {
            for (slot, &id) in ids.iter().enumerate() {
                if map.insert(id, (cluster, slot)).is_some() {
                    let message = format!(
                        "duplicate id {id} in built flat-clustered index; deduplicate the build input before rebuilding (incremental inserts disabled)"
                    );
                    self.id_index_error = Some(message.clone());
                    return Err(message);
                }
            }
        }
        self.id_index = Some(map);
        Ok(())
    }

    /// 检索 top-k（自适应 nprobe）。
    /// EN: Searches top-k with adaptive nprobe.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_with_target(query, k, DEFAULT_SEARCH_TARGET_PER_K)
    }

    /// 以指定「每 k 的候选预算」检索（默认 2500/k，即 k=10 时 25k 候选）。
    /// 移动端可据此在 recall 允许范围内换延迟；默认 `search` 行为不变。
    /// EN: Search with an explicit candidate budget per k (default 2500/k, i.e.
    /// 25k candidates at k=10). Mobile can trade recall for latency; the default
    /// `search` behavior is unchanged.
    #[doc(hidden)]
    pub fn search_with_target(
        &self,
        query: &[f32],
        k: usize,
        target_per_k: usize,
    ) -> Vec<(u64, f32)> {
        self.search_with_target_filtered(query, k, target_per_k, None)
    }

    /// 带允许 id 集合的检索：候选扫描阶段直接跳过不允许的向量（索引内过滤），
    /// 而不是先搜后过滤。`allowed=None` 等价于无过滤。
    ///
    /// 限制：簇路由的 nprobe 仍按全量总量计算；若允许集全部落在构建期
    /// nprobe 之外的簇，返回值仍可能少于 k。`target_per_k` 只能把运行期
    /// nprobe 提到构建期上限，无法越过该上限；此类场景需重建更大的 nprobe
    /// 或退回后过滤。
    /// EN: Search with an allowed-id set: disallowed vectors are skipped during
    /// candidate scanning (in-index filtering) instead of post-filtering.
    /// `allowed=None` is equivalent to no filter.
    ///
    /// Limitation: cluster routing still derives nprobe from the unfiltered
    /// totals; if the allowed set lives entirely in clusters outside the
    /// build-time nprobe set, the result may still contain fewer than k.
    /// `target_per_k` can only raise the runtime nprobe up to the build-time
    /// cap, never beyond it; such cases require rebuilding with a larger
    /// nprobe or falling back to post-filtering.
    #[doc(hidden)]
    pub fn search_with_target_filtered(
        &self,
        query: &[f32],
        k: usize,
        target_per_k: usize,
        allowed: Option<&ahash::AHashSet<u64>>,
    ) -> Vec<(u64, f32)> {
        if query.len() != self.dims {
            return Vec::new();
        }
        let k = k.max(1);
        let target_per_k = target_per_k.max(1);

        // 自适应 nprobe
        // EN: Adaptive nprobe.
        let total = self.total_vectors();
        let cluster_sz = (total / self.num_clusters).max(1);
        let target = k.saturating_mul(target_per_k);
        let adaptive_nprobe = self.nprobe.min((target / cluster_sz).max(4));

        let top_clusters = router::route_query(
            query,
            &self.centroids,
            self.dims,
            adaptive_nprobe,
            self.metric,
        );

        // 并行暴力检索各聚类
        // EN: Brute-force each cluster in parallel.
        let all_candidates: Vec<(u64, f32)> = match &self.cluster_vecs {
            ClusterVectors::F32(vecs) => top_clusters
                .par_iter()
                .flat_map(|&cid| {
                    let ids = &self.cluster_ids[cid];
                    let vecs = &vecs[cid];
                    let mut local: Vec<(u64, f32)> = Vec::with_capacity(ids.len());
                    for (i, &id) in ids.iter().enumerate() {
                        if allowed.is_some_and(|set| !set.contains(&id)) {
                            continue;
                        }
                        let v = &vecs[i * self.dims..(i + 1) * self.dims];
                        local.push((id, dist_fn(self.metric, query, v)));
                    }
                    // 只保留每聚类 top-2k
                    // EN: Keep only the top-2k per cluster.
                    local.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                    local.truncate(k.saturating_mul(2));
                    local
                })
                .collect(),
            ClusterVectors::F16(vecs) => {
                // 余弦路径：查询范数只算一次，候选范数用构建期缓存，热循环只剩
                // 一次点积与一次标量乘法（与 dist_fn 的余弦公式逐项等价）。
                // EN: Cosine path: the query norm is computed once and candidate
                // norms come from the build-time cache, leaving one dot product
                // and one scalar multiply per candidate (item-for-item equivalent
                // to dist_fn's cosine formula).
                let cache = self.cosine_norms.as_ref();
                let q_inv = match cache {
                    Some(_) => inv_norm_from_norm2(dot_product_f32(query, query)),
                    None => 0.0,
                };
                top_clusters
                    .par_iter()
                    .flat_map(|&cid| {
                        let ids = &self.cluster_ids[cid];
                        let vecs = &vecs[cid];
                        let inv_norms = cache.map(|c| c.inv_vec_norms[cid].as_slice());
                        let mut scratch = vec![0f32; self.dims];
                        let mut local: Vec<(u64, f32)> = Vec::with_capacity(ids.len());
                        for (i, &id) in ids.iter().enumerate() {
                            if allowed.is_some_and(|set| !set.contains(&id)) {
                                continue;
                            }
                            let start = i * self.dims;
                            vecs[start..start + self.dims].convert_to_f32_slice(&mut scratch);
                            let d = match inv_norms {
                                Some(inv_norms) => {
                                    let dot = dot_product_f32(query, &scratch);
                                    let inv_v = inv_norms[i];
                                    if q_inv > 0.0 && inv_v > 0.0 {
                                        1.0 - dot * q_inv * inv_v
                                    } else {
                                        1.0
                                    }
                                }
                                None => dist_fn(self.metric, query, &scratch),
                            };
                            local.push((id, d));
                        }
                        local.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                        local.truncate(k.saturating_mul(2));
                        local
                    })
                    .collect()
            }
            ClusterVectors::I8 { codes, scales } => {
                let cache = self.cosine_norms.as_ref();
                let q_inv = match cache {
                    Some(_) => inv_norm_from_norm2(dot_product_f32(query, query)),
                    None => 0.0,
                };
                top_clusters
                    .par_iter()
                    .flat_map(|&cid| {
                        let ids = &self.cluster_ids[cid];
                        let codes = &codes[cid];
                        let scales = &scales[cid];
                        let inv_norms = cache.map(|c| c.inv_vec_norms[cid].as_slice());
                        let mut scratch: Option<Vec<f32>> = None;
                        let mut local: Vec<(u64, f32)> = Vec::with_capacity(ids.len());
                        for (i, &id) in ids.iter().enumerate() {
                            if allowed.is_some_and(|set| !set.contains(&id)) {
                                continue;
                            }
                            let start = i * self.dims;
                            let scale = scales[i];
                            let d = match inv_norms {
                                Some(inv_norms) => {
                                    let dot = dot_i8_scaled(
                                        query,
                                        &codes[start..start + self.dims],
                                        scale,
                                    );
                                    let inv_v = inv_norms[i];
                                    if q_inv > 0.0 && inv_v > 0.0 {
                                        1.0 - dot * q_inv * inv_v
                                    } else {
                                        1.0
                                    }
                                }
                                None => {
                                    let scratch =
                                        scratch.get_or_insert_with(|| vec![0f32; self.dims]);
                                    for (dst, &code) in
                                        scratch.iter_mut().zip(&codes[start..start + self.dims])
                                    {
                                        *dst = code as f32 * scale;
                                    }
                                    dist_fn(self.metric, query, scratch)
                                }
                            };
                            local.push((id, d));
                        }
                        local.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                        local.truncate(k.saturating_mul(2));
                        local
                    })
                    .collect()
            }
        };

        // 去重 + 全局排序
        // EN: Deduplicate + global sort.
        let mut deduped: Vec<(u64, f32)> = {
            let mut seen = ahash::AHashSet::with_capacity(all_candidates.len());
            let mut out = Vec::with_capacity(all_candidates.len());
            for (id, dist) in all_candidates {
                if seen.insert(id) {
                    out.push((id, dist));
                }
            }
            out
        };

        deduped.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        deduped.truncate(k);
        deduped
    }

    pub fn total_vectors(&self) -> usize {
        self.cluster_ids.iter().map(|v| v.len()).sum()
    }

    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// 导出全部原始向量（供 collection 快照使用）。
    /// FlatClustered 不做边界复制，因此每个 id 只出现一次；顺序按聚类块。
    /// EN: Exports every raw vector for collection snapshots. FlatClustered does
    /// no boundary replication, so each id appears exactly once; order follows
    /// the cluster blocks.
    pub fn export_vectors(&self) -> Vec<(u64, Vec<f32>)> {
        let mut out = Vec::with_capacity(self.total_vectors());
        for (cluster, ids) in self.cluster_ids.iter().enumerate() {
            for (slot, &id) in ids.iter().enumerate() {
                out.push((id, self.cluster_vecs.export_row(cluster, slot, self.dims)));
            }
        }
        out
    }

    /// 编码**构建态**（质心 + 分簇 id/向量 + dtype/nprobe），供 mobile
    /// save/load 免重建；外层文件负责 CRC/原子写，这里只做结构与边界校验。
    ///
    /// 格式（小端）：
    /// magic(8) | version u32 | dtype u8 | metric u8 | dims u32 | nprobe u32 |
    /// clusters u32 | centroids (clusters × dims f32) | 每簇 [ids u32][ids × u64]
    /// [vecs 与 ids 等量、按 dtype 逐分量]。
    /// EN: Encode the **built state** (centroids + per-cluster ids/vectors +
    /// dtype/nprobe) so mobile save/load avoids a rebuild. The outer file owns
    /// CRC and atomic writes; this codec only validates structure/bounds.
    ///
    /// Layout (little endian): magic(8) | version u32 | dtype u8 | metric u8 |
    /// dims u32 | nprobe u32 | clusters u32 | centroids (clusters x dims f32) |
    /// per cluster [ids u32][ids x u64][vectors, one component per id per dtype].
    pub fn encode_state(&self) -> Result<Vec<u8>, String> {
        if self.num_clusters == 0 || self.dims == 0 {
            return Err("flat-clustered state: empty dims/clusters".to_string());
        }
        let centroid_len = self
            .num_clusters
            .checked_mul(self.dims)
            .ok_or_else(|| "flat-clustered state: centroid size overflow".to_string())?;
        if self.centroids.len() != centroid_len || self.cluster_ids.len() != self.num_clusters {
            return Err("flat-clustered state: centroid/cluster count mismatch".to_string());
        }
        let per_vector = 8usize
            .checked_add(
                self.dims
                    .checked_mul(self.cluster_vecs.dtype().bytes_per_dim())
                    .ok_or_else(|| "flat-clustered state: vector size overflow".to_string())?,
            )
            .and_then(|v| {
                v.checked_add(if self.cluster_vecs.dtype() == FlatStorageDtype::I8 {
                    4
                } else {
                    0
                })
            })
            .ok_or_else(|| "flat-clustered state: per-vector size overflow".to_string())?;
        let capacity = 26usize
            .checked_add(self.centroids.len() * 4)
            .and_then(|v| v.checked_add(self.num_clusters * 4))
            .and_then(|v| v.checked_add(self.total_vectors().checked_mul(per_vector)?))
            .ok_or_else(|| "flat-clustered state: capacity overflow".to_string())?;
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(FLAT_CLUSTERED_STATE_MAGIC);
        out.extend_from_slice(&FLAT_CLUSTERED_STATE_VERSION.to_le_bytes());
        out.push(self.cluster_vecs.dtype().to_snapshot_byte());
        out.push(metric_code(self.metric));
        out.extend_from_slice(&(self.dims as u32).to_le_bytes());
        out.extend_from_slice(&(self.nprobe as u32).to_le_bytes());
        out.extend_from_slice(&(self.num_clusters as u32).to_le_bytes());
        for &x in &self.centroids {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for (cluster, ids) in self.cluster_ids.iter().enumerate() {
            out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
            for &id in ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
            match &self.cluster_vecs {
                ClusterVectors::F32(v) => {
                    for &x in &v[cluster] {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                ClusterVectors::F16(v) => {
                    for x in &v[cluster] {
                        out.extend_from_slice(&x.to_bits().to_le_bytes());
                    }
                }
                ClusterVectors::I8 { codes, scales } => {
                    for &x in &codes[cluster] {
                        out.push(x as u8);
                    }
                    for &scale in &scales[cluster] {
                        out.extend_from_slice(&scale.to_le_bytes());
                    }
                }
            }
        }
        Ok(out)
    }

    /// 解码构建态；任何长度/重复 id/尾随字节都拒绝（不 panic、不分配放大）。
    /// EN: Decode a built state; reject length mismatches, duplicate ids, and
    /// trailing bytes (no panic, no allocation blow-up).
    pub fn decode_state(bytes: &[u8]) -> Result<Self, String> {
        let mut r = StateReader::new(bytes);
        if r.take(FLAT_CLUSTERED_STATE_MAGIC.len())? != FLAT_CLUSTERED_STATE_MAGIC {
            return Err("flat-clustered state: bad magic".to_string());
        }
        let version = r.u32()?;
        if version != FLAT_CLUSTERED_STATE_VERSION {
            return Err(format!(
                "flat-clustered state: unsupported version {version}"
            ));
        }
        let dtype = FlatStorageDtype::from_snapshot_byte(r.u8()?)
            .ok_or_else(|| "flat-clustered state: bad storage dtype".to_string())?;
        let metric = metric_from_code(r.u8()?)?;
        let dims = r.u32()? as usize;
        let nprobe = r.u32()? as usize;
        let num_clusters = r.u32()? as usize;
        if dims == 0
            || num_clusters == 0
            || nprobe == 0
            || nprobe > num_clusters
            || dims > MAX_STATE_DIMS
            || num_clusters > MAX_STATE_CLUSTERS
        {
            return Err("flat-clustered state: invalid dims/nprobe/clusters".to_string());
        }
        let centroid_len = num_clusters
            .checked_mul(dims)
            .ok_or_else(|| "flat-clustered state: centroid size overflow".to_string())?;
        // 分配前先按剩余字节拒绝放大（每个质心 4 B，每簇至少 4 B 计数）。
        // EN: Reject allocation blow-up against the remaining payload first
        // (each centroid is 4 B and each cluster needs at least 4 B).
        if centroid_len > r.remaining() / 4
            || num_clusters > r.remaining().saturating_sub(centroid_len * 4) / 4
        {
            return Err("flat-clustered state: centroids/clusters exceed payload".to_string());
        }
        let mut centroids = Vec::with_capacity(centroid_len);
        for _ in 0..centroid_len {
            centroids.push(r.f32()?);
        }
        let mut cluster_ids: Vec<Vec<u64>> = Vec::with_capacity(num_clusters);
        let mut cluster_vecs = match dtype {
            FlatStorageDtype::F32 => ClusterVectors::F32(Vec::with_capacity(num_clusters)),
            FlatStorageDtype::F16 => ClusterVectors::F16(Vec::with_capacity(num_clusters)),
            FlatStorageDtype::I8 => ClusterVectors::I8 {
                codes: Vec::with_capacity(num_clusters),
                scales: Vec::with_capacity(num_clusters),
            },
        };
        let mut seen = ahash::AHashSet::new();
        for _ in 0..num_clusters {
            let count = r.u32()? as usize;
            // ids 至少占 8 B，先按剩余字节拒绝明显的分配放大。
            // EN: Each id costs at least 8 B; reject obvious allocation blow-up
            // against the remaining input first.
            if count > r.remaining() / 8 {
                return Err("flat-clustered state: id count exceeds payload".to_string());
            }
            let mut ids = Vec::with_capacity(count);
            for _ in 0..count {
                let id = r.u64()?;
                if !seen.insert(id) {
                    return Err(format!("flat-clustered state: duplicate id {id}"));
                }
                ids.push(id);
            }
            let vec_len = count
                .checked_mul(dims)
                .ok_or_else(|| "flat-clustered state: vector size overflow".to_string())?;
            match &mut cluster_vecs {
                ClusterVectors::F32(v) => {
                    let bytes = vec_len
                        .checked_mul(4)
                        .ok_or_else(|| "flat-clustered state: vector bytes overflow".to_string())?;
                    if bytes > r.remaining() {
                        return Err("flat-clustered state: vector payload truncated".to_string());
                    }
                    let mut list = Vec::with_capacity(vec_len);
                    for _ in 0..vec_len {
                        list.push(r.f32()?);
                    }
                    v.push(list);
                }
                ClusterVectors::F16(v) => {
                    let bytes = vec_len
                        .checked_mul(2)
                        .ok_or_else(|| "flat-clustered state: vector bytes overflow".to_string())?;
                    if bytes > r.remaining() {
                        return Err("flat-clustered state: vector payload truncated".to_string());
                    }
                    let mut list = Vec::with_capacity(vec_len);
                    for _ in 0..vec_len {
                        list.push(f16::from_bits(r.u16()?));
                    }
                    v.push(list);
                }
                ClusterVectors::I8 { codes, scales } => {
                    let code_bytes = vec_len;
                    if code_bytes > r.remaining() {
                        return Err("flat-clustered state: i8 payload truncated".to_string());
                    }
                    let mut list = Vec::with_capacity(vec_len);
                    for _ in 0..vec_len {
                        list.push(r.u8()? as i8);
                    }
                    let scale_bytes = count
                        .checked_mul(4)
                        .ok_or_else(|| "flat-clustered state: scale bytes overflow".to_string())?;
                    if scale_bytes > r.remaining() {
                        return Err("flat-clustered state: i8 scales truncated".to_string());
                    }
                    let mut scale_list = Vec::with_capacity(count);
                    for _ in 0..count {
                        scale_list.push(r.f32()?);
                    }
                    codes.push(list);
                    scales.push(scale_list);
                }
            }
            cluster_ids.push(ids);
        }
        if r.remaining() != 0 {
            return Err("flat-clustered state: trailing bytes".to_string());
        }
        let mut index = FlatClustered {
            dims,
            metric,
            num_clusters,
            nprobe,
            centroids,
            cluster_ids,
            cluster_vecs,
            id_index: None,
            id_index_error: None,
            cosine_norms: None,
        };
        index.rebuild_cosine_norms();
        Ok(index)
    }
}

const FLAT_CLUSTERED_STATE_MAGIC: &[u8; 8] = b"HOUYIFC1";
const FLAT_CLUSTERED_STATE_VERSION: u32 = 1;
/// 构建态解码的防御上限：移动端实际远小于此值，避免恶意/损坏头部放大分配。
/// EN: Defensive caps for built-state decoding; real mobile values are far
/// smaller, preventing crafted/corrupt headers from blowing up allocations.
const MAX_STATE_DIMS: usize = 65_536;
const MAX_STATE_CLUSTERS: usize = 65_536;

fn metric_code(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::L2 => 0,
        DistanceMetric::Cosine => 1,
        DistanceMetric::InnerProduct => 2,
        DistanceMetric::L1 => 3,
    }
}

fn metric_from_code(code: u8) -> Result<DistanceMetric, String> {
    match code {
        0 => Ok(DistanceMetric::L2),
        1 => Ok(DistanceMetric::Cosine),
        2 => Ok(DistanceMetric::InnerProduct),
        3 => Ok(DistanceMetric::L1),
        other => Err(format!("flat-clustered state: bad metric {other}")),
    }
}

/// 构建态解码的定长读取器：越界一律返回错误（不 panic）。
/// EN: Fixed-width reader for state decoding; out-of-bounds returns errors.
struct StateReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> StateReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        StateReader { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err("flat-clustered state: truncated".to_string());
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
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

    use rand::{Rng, SeedableRng};

    /// 快照导出必须返回全部原始向量，且每 id 唯一。
    /// EN: Snapshot export must return every raw vector exactly once per id.
    #[test]
    fn test_export_vectors_roundtrip() {
        let dims = 16;
        let n = 200;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).map(|i| i * 3 + 7).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>());
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let fc = FlatClustered::build(&vectors, &ids, dims, DistanceMetric::L2, c, 10, 0);

        let mut exported = fc.export_vectors();
        assert_eq!(exported.len(), n);
        exported.sort_by_key(|(id, _)| *id);
        for (i, (id, v)) in exported.iter().enumerate() {
            assert_eq!(*id, ids[i]);
            assert_eq!(v.len(), dims);
            assert_eq!(v.as_slice(), &vectors[i * dims..(i + 1) * dims]);
        }
    }

    /// 回归：小数据集（c < 4）时 nprobe 的 clamp 下界不得超过上界，否则 build panic。
    /// EN: regression — for c < 4 the nprobe clamp lower bound must not exceed
    /// the upper bound (pre-existing panic on small collections).
    #[test]
    fn test_flat_clustered_small_collection_no_clamp_panic() {
        let dims = 8;
        let n = 6;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>());
        }
        let fc = FlatClustered::build(&vectors, &ids, dims, DistanceMetric::L2, 3, 5, 0);
        assert_eq!(fc.total_vectors(), n);
    }

    #[test]
    fn test_flat_clustered_build_and_search() {
        let dims = 16;
        let n = 500;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            for _ in 0..dims {
                vectors.push(rng.gen::<f32>());
            }
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let fc = FlatClustered::build(&vectors, &ids, dims, DistanceMetric::L2, c, 10, 0);

        assert_eq!(fc.total_vectors(), n);

        // 自检索
        // EN: Self-search.
        let mut found = 0;
        for i in 0..50 {
            let q = &vectors[i * dims..(i + 1) * dims];
            let r = fc.search(q, 1);
            if r.first().map(|x| x.0) == Some(ids[i]) {
                found += 1;
            }
        }
        assert!(found as f32 / 50.0 > 0.7, "self-recall too low: {found}/50");
    }

    /// 增量插入必须立即可查；同 id upsert 覆盖旧向量且不增加计数。
    /// EN: Incremental inserts must be searchable immediately; an upsert with
    /// the same id overwrites the old vector without changing the count.
    #[test]
    fn test_flat_clustered_incremental_insert_and_upsert() {
        let dims = 8;
        let n = 200;
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>());
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let mut fc = FlatClustered::build(&vectors, &ids, dims, DistanceMetric::L2, c, 10, 0);

        // 新 id 的向量贴着 0 号向量：应立刻成为 0 号查询的最近邻。
        // EN: A new id next to vector 0 must become the nearest result
        // immediately.
        let mut probe = vectors[..dims].to_vec();
        probe[0] += 0.001;
        fc.insert(9999, &probe).expect("insert");
        let found = fc.search(&probe, 1);
        assert_eq!(found.first().map(|x| x.0), Some(9999));
        assert_eq!(fc.total_vectors(), n + 1);

        // 同 id 覆盖到另一端：旧位置被移除，新位置可查，计数不变。
        // EN: Upsert the same id to the opposite end: the old slot is removed,
        // the new one is searchable, and the count stays unchanged.
        let mut far = vectors[(n - 1) * dims..].to_vec();
        far[0] += 0.001;
        fc.insert(9999, &far).expect("upsert");
        let found = fc.search(&far, 1);
        assert_eq!(found.first().map(|x| x.0), Some(9999));
        assert_eq!(fc.total_vectors(), n + 1);

        // 新位置只出现一次（搬移后旧槽位不得残留同 id 副本）。
        // EN: The new slot appears exactly once (no duplicate id may survive
        // the cross-cluster move).
        let dup = fc.search(&far, 10);
        assert_eq!(
            dup.iter().filter(|(id, _)| *id == 9999).count(),
            1,
            "id must not be duplicated after cross-cluster upsert"
        );
    }

    /// 同簇覆盖走原位写路径；导出配对必须正确（供快照/重建）。
    /// EN: Same-cluster overwrite takes the in-place path; exported id/vector
    /// pairing must stay correct (snapshot/rebuild depend on it).
    #[test]
    fn test_flat_clustered_same_cluster_upsert_and_pairing() {
        let dims = 8;
        let n = 200;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>());
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let mut fc = FlatClustered::build(&vectors, &ids, dims, DistanceMetric::L2, c, 10, 0);
        let mut probe = vectors[..dims].to_vec();
        probe[0] += 0.01;
        fc.insert(8888, &probe).expect("insert");
        // 先精确同向量插入：路由必然相同，锁定 copy_from_slice 分支。
        // EN: First re-insert the exact same vector: routing is identical, so
        // the copy_from_slice branch runs deterministically.
        fc.insert(8888, &probe).expect("same-cluster upsert");
        let old_loc = *fc
            .id_index
            .as_ref()
            .and_then(|map| map.get(&8888))
            .expect("incremental id indexed");
        // 再写入可区分的向量：位置不变且字节可观测，才证明原位覆盖真的发生。
        // EN: Then write a distinguishable vector: an unchanged slot plus
        // observable bytes proves the in-place overwrite actually happened.
        let mut changed = probe.clone();
        changed[1] += 0.001;
        fc.insert(8888, &changed).expect("same-cluster overwrite");
        assert_eq!(
            *fc.id_index
                .as_ref()
                .and_then(|map| map.get(&8888))
                .expect("id remains indexed"),
            old_loc,
            "same-cluster upsert must not move the slot"
        );
        assert_eq!(fc.total_vectors(), n + 1);
        let found = fc.search(&changed, 1);
        assert_eq!(found.first().map(|x| x.0), Some(8888));

        // 导出配对：每个原始 id 仍对应原向量，8888 对应覆盖后的向量。
        // EN: Export pairing: every original id still maps to its vector and
        // 8888 maps to the overwritten vector.
        let mut exported: ahash::AHashMap<u64, Vec<f32>> =
            fc.export_vectors().into_iter().collect();
        assert_eq!(exported.len(), n + 1);
        for i in 0..n as u64 {
            let v = exported.remove(&i).expect("original id");
            assert_eq!(
                v,
                vectors[i as usize * dims..(i as usize + 1) * dims].to_vec()
            );
        }
        assert_eq!(exported.remove(&8888).unwrap(), changed);
        assert!(exported.is_empty());
    }

    /// 构建输入含重复 id 时，增量入口必须拒绝（否则旧副本永远可被检索）。
    /// EN: Duplicate ids in the build input must make the incremental entry
    /// reject the index (otherwise a stale copy stays searchable forever).
    #[test]
    fn test_flat_clustered_duplicate_build_rejects_incremental_insert() {
        let dims = 4;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut vectors = Vec::new();
        let mut ids = Vec::new();
        for i in 0..10u64 {
            ids.push(if i == 5 { 1 } else { i });
            for _ in 0..dims {
                vectors.push(rng.gen::<f32>());
            }
        }
        let mut fc = FlatClustered::build(&vectors, &ids, dims, DistanceMetric::L2, 3, 5, 0);
        let probe: Vec<f32> = (0..dims).map(|_| 0.5).collect();
        assert!(
            fc.insert(99, &probe).is_err(),
            "duplicate build ids must be rejected by incremental insert"
        );
        // 失败是粘性的：重试立即返回同一错误（不重复 O(n) 扫描）。
        // EN: The failure is sticky: retries return the same error immediately
        // without re-scanning O(n).
        let first = fc.insert(98, &probe).unwrap_err();
        let second = fc.insert(97, &probe).unwrap_err();
        assert_eq!(first, second);
        assert!(fc.id_index_error.is_some(), "failure must be cached");
        assert_eq!(fc.total_vectors(), ids.len());
    }

    /// f16 存储：编码正确、检索可用、增量插入/导出仍以 f32 视图工作。
    /// EN: f16 storage: correct encoding, searchable, and incremental insert /
    /// export still work through the f32 view.
    /// 构建态编解码：f32/f16 往返后 dtype/metric/检索/增量行为一致。
    /// EN: Built-state roundtrip: dtype/metric/search/incremental behavior
    /// survives encode/decode for both f32 and f16.
    #[test]
    fn test_flat_clustered_state_roundtrip() {
        let dims = 12;
        let n = 300;
        let mut rng = rand::rngs::StdRng::seed_from_u64(31);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>() * 2.0 - 1.0);
        }
        let c = (n as f64).sqrt().ceil() as usize;
        for dtype in [
            FlatStorageDtype::F32,
            FlatStorageDtype::F16,
            FlatStorageDtype::I8,
        ] {
            let fc = FlatClustered::build_with_storage_dtype(
                &vectors,
                &ids,
                dims,
                DistanceMetric::Cosine,
                c,
                10,
                0,
                dtype,
            );
            let bytes = fc.encode_state().expect("encode");
            let mut decoded = FlatClustered::decode_state(&bytes).expect("state must decode");
            assert_eq!(decoded.storage_dtype(), dtype);
            let mut before = fc.export_vectors();
            let mut after = decoded.export_vectors();
            before.sort_by_key(|(id, _)| *id);
            after.sort_by_key(|(id, _)| *id);
            assert_eq!(before.len(), after.len());
            for ((id_a, va), (id_b, vb)) in before.iter().zip(&after) {
                assert_eq!(id_a, id_b);
                assert_eq!(va, vb);
            }
            // 解码后的索引仍可增量写入。
            // EN: The decoded index still accepts incremental writes.
            let mut probe = vec![0.0f32; dims];
            probe[0] = 1.0;
            decoded.insert(9999, &probe).expect("insert after decode");
            let got = decoded.search(&probe, 1);
            assert_eq!(got.first().map(|x| x.0), Some(9999));

            // 坏 magic / 截断 / 尾随字节必须拒绝。
            // EN: Bad magic, truncation, and trailing bytes must be rejected.
            let mut bad = bytes.clone();
            bad[0] ^= 0xFF;
            assert!(FlatClustered::decode_state(&bad).is_err());
            let truncated = &bytes[..bytes.len() - 1];
            assert!(FlatClustered::decode_state(truncated).is_err());
            let mut trailing = bytes.clone();
            trailing.push(0);
            assert!(FlatClustered::decode_state(&trailing).is_err());
        }

        // 退化单向量集合：build 的簇数必须与实际质心数一致，状态可往返。
        // EN: Degenerate one-vector collection: build's cluster count must match
        // the actual centroid count and the state must roundtrip.
        let one = FlatClustered::build_with_dtype(
            &vectors[..dims],
            &[7],
            dims,
            DistanceMetric::L2,
            2,
            5,
            0,
            VectorDtype::F16,
        );
        let bytes = one.encode_state().expect("single encode");
        let decoded = FlatClustered::decode_state(&bytes).expect("single decode");
        assert_eq!(decoded.state_dims(), dims);
        assert_eq!(decoded.total_vectors(), 1);
        let found = decoded.search(&vectors[..dims], 1);
        assert_eq!(found.first().map(|x| x.0), Some(7));
    }

    #[test]
    fn test_flat_clustered_f16_storage_and_incremental() {
        let dims = 16;
        let n = 300;
        let mut rng = rand::rngs::StdRng::seed_from_u64(21);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>() * 2.0 - 1.0);
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let mut fc = FlatClustered::build_with_dtype(
            &vectors,
            &ids,
            dims,
            DistanceMetric::L2,
            c,
            10,
            0,
            VectorDtype::F16,
        );
        assert_eq!(fc.storage_dtype(), FlatStorageDtype::F16);
        // 导出为 f32 视图，数值等于 f16 舍入后的值。
        // EN: Export is the f32 view and matches the f16-rounded values.
        let exported = fc.export_vectors();
        assert_eq!(exported.len(), n);
        for (id, v) in &exported {
            let i = *id as usize;
            for j in 0..dims {
                let expected = to_f16(vectors[i * dims + j]).to_f32();
                assert_eq!(v[j], expected);
            }
        }
        // 自检索应命中自身（f16 舍入不改变精确自匹配的排序）。
        // EN: Self-search should hit itself (f16 rounding preserves exact
        // self-matches).
        for i in 0..50 {
            let q = &vectors[i * dims..(i + 1) * dims];
            let r = fc.search(q, 1);
            assert_eq!(r.first().map(|x| x.0), Some(ids[i]));
        }
        // 增量插入与覆盖在 f16 下同样立即可查。
        // EN: Incremental insert/overwrite are immediately searchable under f16.
        let mut probe = vec![0.0f32; dims];
        probe[0] = 1.0;
        fc.insert(9999, &probe).expect("insert f16");
        assert_eq!(fc.total_vectors(), n + 1);
        let found = fc.search(&probe, 1);
        assert_eq!(found.first().map(|x| x.0), Some(9999));
        fc.insert(9999, &probe).expect("same-cluster overwrite");
        assert_eq!(fc.total_vectors(), n + 1);
    }

    /// f16 + Cosine 的范数缓存路径：构建、跨簇 upsert、状态往返与
    /// 默认距离核（dist_fn）对拍；这是移动端默认配置。
    /// EN: f16 + Cosine norm-cache path: build, cross-cluster upsert, state
    /// roundtrip, and parity with the default dist_fn kernel. This is the
    /// mobile default configuration.
    #[test]
    fn test_flat_clustered_cosine_f16_cache_path() {
        let dims = 16;
        let n = 400;
        let mut rng = rand::rngs::StdRng::seed_from_u64(71);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            let mut v = vec![0f32; dims];
            for x in &mut v {
                *x = rng.gen::<f32>() * 2.0 - 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            vectors.extend_from_slice(&v);
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let mut fc = FlatClustered::build_with_dtype(
            &vectors,
            &ids,
            dims,
            DistanceMetric::Cosine,
            c,
            10,
            0,
            VectorDtype::F16,
        );
        assert!(
            fc.cosine_norms.is_some(),
            "f16 cosine must build the norm cache"
        );
        // 与默认 dist_fn 对拍：同一条查询的 top-1 距离应一致（f16 舍入内）。
        // EN: Parity with dist_fn: top-1 distance must match (within f16
        // rounding).
        let q = &vectors[..dims];
        let got = fc.search(q, 1);
        let expected = fc
            .export_vectors()
            .into_iter()
            .map(|(id, v)| (id, dist_fn(DistanceMetric::Cosine, q, &v)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .expect("non-empty");
        assert_eq!(got[0].0, expected.0);
        assert!(
            (got[0].1 - expected.1).abs() < 1.0e-3,
            "cached cosine {} != dist_fn {}",
            got[0].1,
            expected.1
        );

        // 跨簇 upsert：缓存中的位置/范数必须同步。先在同一簇尾追加两个近邻，
        // 让 9999 处于非尾槽位，搬移时真正触发 swap_remove 的尾部重定位。
        // EN: Cross-cluster upsert must keep cached slot/norm in sync. Append two
        // close neighbours first so 9999 sits in a non-tail slot; the move then
        // exercises swap_remove's tail relocation.
        let mut far = vec![0.0f32; dims];
        far[0] = 1.0;
        fc.insert(9999, &far).expect("insert");
        let mut near = far.clone();
        near[1] += 0.001;
        fc.insert(9998, &near).expect("neighbour insert");
        let mut nearer = near.clone();
        nearer[2] += 0.001;
        fc.insert(9997, &nearer).expect("neighbour insert");
        let map = fc.id_index.as_ref().expect("id index");
        assert_eq!(
            map.get(&9999).map(|p| p.0),
            map.get(&9998).map(|p| p.0),
            "precondition: neighbours must be same cluster"
        );
        let mut other = vec![0.0f32; dims];
        other[dims - 1] = 1.0;
        fc.insert(9999, &other).expect("cross-cluster upsert");
        let found = fc.search(&other, 1);
        assert_eq!(found.first().map(|x| x.0), Some(9999));
        // 被 swap 到空位的邻居仍需可检索，说明位置与范数缓存同步搬移。
        // EN: The neighbour swapped into the vacated slot must remain
        // searchable, proving slot and norm cache moved together.
        assert_eq!(fc.search(&near, 1).first().map(|x| x.0), Some(9998));
        assert_eq!(fc.search(&nearer, 1).first().map(|x| x.0), Some(9997));
        assert_eq!(fc.total_vectors(), n + 3);

        // 状态往返后缓存重建，检索仍命中。
        // EN: The cache is rebuilt after a state roundtrip and search still hits.
        let bytes = fc.encode_state().expect("encode");
        let decoded = FlatClustered::decode_state(&bytes).expect("decode");
        assert!(decoded.cosine_norms.is_some());
        let found = decoded.search(&other, 1);
        assert_eq!(found.first().map(|x| x.0), Some(9999));
    }

    /// i8 存储：余弦缓存启用、量化召回接近暴力余弦、增量/状态往返可用。
    /// EN: i8 storage: cosine cache enabled, quantized recall close to
    /// brute-force cosine, incremental/state roundtrip usable.
    #[test]
    fn test_flat_clustered_i8_cosine_recall() {
        let dims = 16;
        let n = 500;
        let mut rng = rand::rngs::StdRng::seed_from_u64(91);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            let mut v = vec![0f32; dims];
            for x in &mut v {
                *x = rng.gen::<f32>() * 2.0 - 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            vectors.extend_from_slice(&v);
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let mut fc = FlatClustered::build_with_storage_dtype(
            &vectors,
            &ids,
            dims,
            DistanceMetric::Cosine,
            c,
            10,
            0,
            FlatStorageDtype::I8,
        );
        assert_eq!(fc.storage_dtype(), FlatStorageDtype::I8);
        assert!(fc.cosine_norms.is_some(), "i8 cosine must cache norms");
        let mut hits = 0;
        let mut recall_queries = 0;
        for qi in 0..20 {
            let q = &vectors[qi * dims..(qi + 1) * dims];
            // 真值用原始 f32 向量，而不是 i8 反量化后的视图，才能检验量化质量。
            // EN: Ground truth uses the original f32 rows, not the i8-dequantized
            // view, so quantizer defects are observable.
            let mut truth: Vec<(u64, f32)> = vectors
                .chunks_exact(dims)
                .enumerate()
                .map(|(i, v)| (i as u64, dist_fn(DistanceMetric::Cosine, q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let truth: Vec<u64> = truth.into_iter().take(10).map(|(id, _)| id).collect();
            let found = fc.search(q, 10);
            let tset: ahash::AHashSet<u64> = truth.into_iter().collect();
            hits += found.iter().filter(|(id, _)| tset.contains(id)).count();
            recall_queries += 1;
        }
        let recall = hits as f32 / (recall_queries * 10) as f32;
        assert!(recall >= 0.95, "i8 cosine recall too low: {recall:.3}");
        // 增量插入与状态往返。
        // EN: Incremental insert and state roundtrip.
        let mut probe = vec![0.0f32; dims];
        probe[0] = 1.0;
        fc.insert(9999, &probe).expect("insert");
        assert_eq!(fc.search(&probe, 1).first().map(|x| x.0), Some(9999));
        // 同 id 覆盖到另一方向：i8 overwrite + scale/cache 同步。
        // EN: Overwrite the same id in another direction: i8 overwrite plus
        // scale/cache sync.
        let mut other = vec![0.0f32; dims];
        other[dims - 1] = 1.0;
        fc.insert(9999, &other).expect("i8 upsert");
        assert_eq!(fc.search(&other, 1).first().map(|x| x.0), Some(9999));
        let bytes = fc.encode_state().expect("encode");
        let decoded = FlatClustered::decode_state(&bytes).expect("decode");
        assert_eq!(decoded.storage_dtype(), FlatStorageDtype::I8);
        assert_eq!(decoded.search(&other, 1).first().map(|x| x.0), Some(9999));
    }

    /// 索引内过滤：结果只来自 allowed 集合，且与 allowed 子集上的暴力排序一致。
    /// EN: In-index filtering: results come only from the allowed set and match
    /// brute-force ordering over that subset.
    #[test]
    fn test_flat_clustered_search_filter_mask() {
        let dims = 16;
        let n = 300;
        let mut rng = rand::rngs::StdRng::seed_from_u64(101);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).map(|i| i * 7 + 3).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>() * 2.0 - 1.0);
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let fc = FlatClustered::build_with_storage_dtype(
            &vectors,
            &ids,
            dims,
            DistanceMetric::Cosine,
            c,
            10,
            0,
            FlatStorageDtype::F16,
        );
        let allowed: ahash::AHashSet<u64> = ids.iter().copied().step_by(3).collect();
        let q = &vectors[..dims];
        let mut truth: Vec<(u64, f32)> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| allowed.contains(id))
            .map(|(i, id)| {
                (
                    *id,
                    dist_fn(
                        DistanceMetric::Cosine,
                        q,
                        &vectors[i * dims..(i + 1) * dims],
                    ),
                )
            })
            .collect();
        truth.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let found =
            fc.search_with_target_filtered(q, 10, DEFAULT_SEARCH_TARGET_PER_K, Some(&allowed));
        assert_eq!(
            found.len(),
            10,
            "filtered search should fill k when clusters cover the data"
        );
        assert!(found.iter().all(|(id, _)| allowed.contains(id)));
        let expected: Vec<u64> = truth
            .into_iter()
            .take(found.len())
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            found.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            expected
        );
    }

    /// i8 + 非余弦（L2）走无范数缓存的解码分支；真值用原始 f32 向量。
    /// EN: i8 + non-cosine (L2) exercises the dequantize-only branch; ground
    /// truth uses the original f32 rows.
    #[test]
    fn test_flat_clustered_i8_l2_recall() {
        let dims = 16;
        let n = 400;
        let mut rng = rand::rngs::StdRng::seed_from_u64(93);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>() * 2.0 - 1.0);
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let fc = FlatClustered::build_with_storage_dtype(
            &vectors,
            &ids,
            dims,
            DistanceMetric::L2,
            c,
            10,
            0,
            FlatStorageDtype::I8,
        );
        assert!(fc.cosine_norms.is_none(), "L2 must not build cosine cache");
        let mut hits = 0;
        for qi in 0..20 {
            let q = &vectors[qi * dims..(qi + 1) * dims];
            let mut truth: Vec<(u64, f32)> = vectors
                .chunks_exact(dims)
                .enumerate()
                .map(|(i, v)| (i as u64, dist_fn(DistanceMetric::L2, q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let truth: Vec<u64> = truth.into_iter().take(10).map(|(id, _)| id).collect();
            let found = fc.search(q, 10);
            let tset: ahash::AHashSet<u64> = truth.into_iter().collect();
            hits += found.iter().filter(|(id, _)| tset.contains(id)).count();
        }
        let recall = hits as f32 / 200.0;
        assert!(recall >= 0.95, "i8 L2 recall too low: {recall:.3}");
    }

    #[test]
    fn test_flat_clustered_recall() {
        let dims = 32;
        let n = 300;
        let k = 10;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let mut vectors = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            for _ in 0..dims {
                vectors.push(rng.gen::<f32>());
            }
        }

        let c = (n as f64).sqrt().ceil() as usize;
        let fc = FlatClustered::build(&vectors, &ids, dims, DistanceMetric::L2, c, 10, 0);

        // 暴力 ground truth
        // EN: Brute-force ground truth.
        let mut hits = 0;
        for _ in 0..30 {
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
            brute.sort_by(|a, b| a.1.total_cmp(&b.1));
            let truth: ahash::AHashSet<u64> = brute.iter().take(k).map(|x| x.0).collect();
            let pred: ahash::AHashSet<u64> = fc.search(&q, k).into_iter().map(|x| x.0).collect();
            hits += truth.intersection(&pred).count();
        }
        let recall = hits as f32 / (30 * k) as f32;
        assert!(recall > 0.85, "recall too low: {recall:.3}");
    }
}
