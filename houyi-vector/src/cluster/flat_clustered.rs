//! FlatClustered: K-Means 分区 + 每区 SIMD 暴力检索（无 HNSW 子图）。
//! K-Means partitioning + per-cluster brute-force SIMD search (no HNSW sub-graphs).
//!
//! 适合: 每分区向量数 < 5K、频繁全量更新、NUMA 多节点部署。
//! Ideal for: <5K vectors per cluster, frequent full reloads, NUMA-aware deploys.

use super::boundary::dist_fn;
use super::kmeans;
use super::router;
use crate::DistanceMetric;
use rayon::prelude::*;

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
    /// 每聚类的平坦向量数据 [cluster_size × dims]
    /// EN: Flat vector data per cluster [cluster_size × dims].
    cluster_vecs: Vec<Vec<f32>>,
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
        assert_eq!(vectors.len(), ids.len() * dims);
        let n = ids.len();
        let c = num_clusters.min(n).max(2);

        // 1) K-Means
        let km = kmeans::kmeans(vectors, dims, c, kmeans_iters, seed, metric);

        // 2) 分配向量
        // EN: 2) Assign vectors.
        let assignments = router::assign_vectors(vectors, dims, &km.centroids, metric);
        let groups = router::group_by_cluster(&assignments, c);

        // 3) 按聚类整理数据
        // EN: 3) Organize data by cluster.
        let mut cluster_ids: Vec<Vec<u64>> = (0..c).map(|_| Vec::new()).collect();
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

        // nprobe: 自适应公式（与 ClusteredHnsw 一致）
        // 下界不得超过上界：c ∈ {2,3}（小数据集）时 clamp(4, c) 会 panic。
        // EN: lower bound must not exceed upper bound — clamp(4, c) panics for c < 4.
        let cluster_sz = n / c;
        let target = 25_000; // k=10 预算
        let nprobe = (target / cluster_sz.max(1)).clamp(4.min(c), c);

        FlatClustered {
            dims,
            metric,
            num_clusters: c,
            nprobe,
            centroids: km.centroids,
            cluster_ids,
            cluster_vecs,
        }
    }

    /// 检索 top-k（自适应 nprobe）。
    /// EN: Searches top-k with adaptive nprobe.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if query.len() != self.dims {
            return Vec::new();
        }
        let k = k.max(1);

        // 自适应 nprobe
        // EN: Adaptive nprobe.
        let total = self.total_vectors();
        let cluster_sz = (total / self.num_clusters).max(1);
        let target = k.saturating_mul(2500);
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
        let all_candidates: Vec<(u64, f32)> = top_clusters
            .par_iter()
            .flat_map(|&cid| {
                let ids = &self.cluster_ids[cid];
                let vecs = &self.cluster_vecs[cid];
                let mut local: Vec<(u64, f32)> = ids
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| {
                        let v = &vecs[i * self.dims..(i + 1) * self.dims];
                        let d = dist_fn(self.metric, query, v);
                        (id, d)
                    })
                    .collect();
                // 只保留每聚类 top-2k
                // EN: Keep only the top-2k per cluster.
                local.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                local.truncate(k.saturating_mul(2));
                local
            })
            .collect();

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceMetric;

    use rand::{Rng, SeedableRng};

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
