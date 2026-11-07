//! K-Means 聚类：Rayon 并行距离计算，返回平坦的聚类中心向量。
//! K-Means clustering with Rayon-parallel distance computation.

use super::boundary::dist_fn;
use crate::DistanceMetric;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// K-Means 聚类结果。
/// EN: K-Means clustering result.
pub struct KMeansResult {
    /// 平坦存储的聚类中心 [num_clusters × dims]
    /// EN: Centroids stored flat [num_clusters × dims].
    pub centroids: Vec<f32>,
    /// 每个向量的主聚类 ID
    /// EN: Primary cluster id for each vector.
    pub assignments: Vec<usize>,
    /// 最终迭代的 inertia (簇内平方和)
    /// EN: Inertia at the final iteration (within-cluster sum of squares).
    pub inertia: f64,
    /// 实际迭代次数
    /// EN: Actual number of iterations.
    pub iterations: usize,
}

/// 运行 K-Means 聚类。
/// EN: Runs K-Means clustering.
///
/// - `vectors`: 平坦存储的向量 [N × dims]
///   EN: - `vectors`: flat vectors [N × dims]
/// - `dims`: 向量维度
///   EN: - `dims`: vector dimension
/// - `num_clusters`: 聚类数 C
///   EN: - `num_clusters`: number of clusters C
/// - `max_iters`: 最大迭代次数
///   EN: - `max_iters`: maximum iterations
/// - `seed`: 随机种子
///   EN: - `seed`: RNG seed
/// - `metric`: 距离度量
///   EN: - `metric`: distance metric
pub fn kmeans(
    vectors: &[f32],
    dims: usize,
    num_clusters: usize,
    max_iters: usize,
    seed: u64,
    metric: DistanceMetric,
) -> KMeansResult {
    assert!(!vectors.is_empty(), "vectors must not be empty");
    assert!(dims > 0, "dims must be > 0");
    assert!(num_clusters > 0, "num_clusters must be > 0");
    assert_eq!(
        vectors.len() % dims,
        0,
        "vectors len must be multiple of dims"
    );

    let n = vectors.len() / dims;
    let c = num_clusters.min(n);

    // 初始化中心: k-means++ 风格（第一个随机，后续加权采样）
    // EN: Initialize centroids: k-means++ style (first random, then weighted sampling).
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = vec![0.0_f32; c * dims];
    let mut assignments = vec![0_usize; n];

    // 第一个中心: 随机选一个向量
    // EN: First centroid: pick a random vector.
    let first_idx = rng.gen_range(0..n);
    centroids[0..dims].copy_from_slice(&vectors[first_idx * dims..(first_idx + 1) * dims]);

    // 后续中心: 按距离加权随机采样 (k-means++ 初始化)
    // EN: Remaining centroids: weighted random sampling by distance (k-means++ init).
    let mut min_dists = vec![f32::MAX; n];
    for k in 1..c {
        // 并行更新每个点到最近中心的距离
        // EN: Update each point's distance to its nearest centroid in parallel.
        min_dists.par_iter_mut().enumerate().for_each(|(i, md)| {
            let vi = &vectors[i * dims..(i + 1) * dims];
            let ck_start = (k - 1) * dims;
            // 只检查最新加入的中心
            // EN: Only check the newly added centroid.
            let d = dist_fn(metric, vi, &centroids[ck_start..ck_start + dims]);
            if d < *md {
                *md = d;
            }
        });

        // 加权随机采样下一个中心。IP 的 -dot 距离可为负——直接求和会得到负
        // total 并按「全部重合」兜底随机选，k-means++ 加权完全失效；先把最小值
        // 按 0 截断后平移（仅当 min_d < 0 时生效），既修复负 IP，又与旧行为在
        // L2/Cosine 的 min_d ≥ 0 区间逐位一致——无条件的 min_d 平移会把采样权重
        // 从 d 改成 d - min_d，静默改变既有非 IP 聚类的初始化结果（OCR #15、#10）。
        // EN: Weighted-random-sample the next centroid. IP's -dot distances may
        // be negative — summing them directly yields a negative total and falls
        // into the "all coincide" random pick, disabling k-means++ weighting
        // entirely. Clamp the minimum at 0 and shift only when it is negative:
        // this fixes negative IP while staying bit-identical to the old behavior
        // for L2/Cosine (min_d ≥ 0). An unconditional min_d shift would change
        // the sampling weights from d to d - min_d, silently altering the
        // initialization of every existing non-IP clustering run (OCR #15, #10).
        let min_d = min_dists.iter().copied().fold(f32::INFINITY, f32::min);
        let shift = min_d.min(0.0);
        let total: f32 = min_dists.iter().map(|&d| d - shift).sum();
        if total <= 0.0 {
            // 所有点已经重合, 随机选
            // EN: All points coincide; pick randomly.
            let idx = rng.gen_range(0..n);
            centroids[k * dims..(k + 1) * dims]
                .copy_from_slice(&vectors[idx * dims..(idx + 1) * dims]);
        } else {
            let threshold = rng.gen::<f32>() * total;
            let mut cumulative = 0.0_f32;
            let mut chosen = 0;
            for (i, &md) in min_dists.iter().enumerate() {
                cumulative += md - shift;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids[k * dims..(k + 1) * dims]
                .copy_from_slice(&vectors[chosen * dims..(chosen + 1) * dims]);
        }
    }

    // 迭代优化
    // EN: Iterative refinement.
    let mut cluster_counts = vec![0_usize; c];
    let mut new_centroids = vec![0.0_f32; c * dims];
    let mut iter = 0;

    for _ in 0..max_iters {
        iter += 1;

        // E-step: 并行分配每个向量到最近中心
        // EN: E-step: assign each vector to its nearest centroid in parallel.
        let changed = AtomicUsize::new(0);
        assignments
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, assign)| {
                let vi = &vectors[i * dims..(i + 1) * dims];
                let mut best_c = 0;
                let mut best_d = dist_fn(metric, vi, &centroids[0..dims]);
                for k in 1..c {
                    let d = dist_fn(metric, vi, &centroids[k * dims..(k + 1) * dims]);
                    if d < best_d {
                        best_d = d;
                        best_c = k;
                    }
                }
                if *assign != best_c {
                    changed.fetch_add(1, Ordering::Relaxed);
                }
                *assign = best_c;
            });

        if changed.load(Ordering::Relaxed) == 0 {
            break;
        }

        // M-step: 重新计算中心
        // EN: M-step: recompute centroids.
        new_centroids.fill(0.0);
        cluster_counts.fill(0);

        for (i, &cluster_id) in assignments.iter().enumerate() {
            let vi = &vectors[i * dims..(i + 1) * dims];
            let c_start = cluster_id * dims;
            for d in 0..dims {
                new_centroids[c_start + d] += vi[d];
            }
            cluster_counts[cluster_id] += 1;
        }

        for (k, &count) in cluster_counts.iter().enumerate() {
            if count > 0 {
                let count = count as f32;
                let c_start = k * dims;
                for d in 0..dims {
                    centroids[c_start + d] = new_centroids[c_start + d] / count;
                }
            }
        }
    }

    // 计算最终 inertia
    // EN: Compute the final inertia.
    let inertia: f64 = assignments
        .par_iter()
        .enumerate()
        .map(|(i, &cluster_id)| {
            let vi = &vectors[i * dims..(i + 1) * dims];
            let c = &centroids[cluster_id * dims..(cluster_id + 1) * dims];
            dist_fn(metric, vi, c) as f64
        })
        .sum();

    KMeansResult {
        centroids,
        assignments,
        inertia,
        iterations: iter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceMetric;

    /// 生成模拟数据: 3 个自然聚类，每聚类 100 点，2 维。
    /// EN: Generate synthetic data: 3 natural clusters, 100 points each, 2-D.
    fn generate_gaussian_clusters() -> (Vec<f32>, usize, usize) {
        let dims = 2;
        let per_cluster = 100;
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = Vec::with_capacity(3 * per_cluster * dims);
        let centers = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)];

        for (cx, cy) in &centers {
            for _ in 0..per_cluster {
                let x = *cx + rng.gen::<f32>() * 3.0 - 1.5;
                let y = *cy + rng.gen::<f32>() * 3.0 - 1.5;
                data.push(x);
                data.push(y);
            }
        }
        (data, dims, per_cluster)
    }

    #[test]
    fn test_kmeans_converges_on_gaussian_clusters() {
        let (data, dims, _per_cluster) = generate_gaussian_clusters();
        let result = kmeans(&data, dims, 3, 50, 0xC0FFEE, DistanceMetric::L2);

        // 应该收敛到 3 个聚类
        // EN: It should converge to 3 clusters.
        assert!(result.iterations <= 50);
        assert!(result.inertia > 0.0);

        // 每个聚类都应该有分配
        // EN: Every cluster should have assignments.
        let mut cluster_sizes = vec![0_usize; 3];
        for &a in &result.assignments {
            cluster_sizes[a] += 1;
        }
        for size in &cluster_sizes {
            assert!(*size > 0, "every cluster should have points assigned");
        }
    }

    #[test]
    fn test_kmeans_single_cluster_trivial() {
        let dims = 4;
        let data: Vec<f32> = (0..100 * dims).map(|i| (i % 13) as f32 * 0.01).collect();
        let result = kmeans(&data, dims, 1, 10, 0, DistanceMetric::Cosine);

        assert_eq!(result.centroids.len(), dims);
        assert!(result.assignments.iter().all(|&a| a == 0));
        assert_eq!(result.iterations, 1); // 单聚类一次迭代即收敛
    }

    #[test]
    fn test_kmeans_inner_product_negative_distances_still_partition() {
        // OCR #15：所有点积为正 → 所有 IP 距离为负；旧实现 min_dists 求和为负，
        // 每次都走「全部重合」随机兜底。最小平移后 k-means++ 加权恢复，两个
        // 方向簇应各自成簇。这里用固定种子锁死结果形态。
        // EN: OCR #15 — all dots are positive, so every IP distance is negative;
        // the old implementation summed to a negative total and always took the
        // "all coincide" random fallback. The min-shift restores k-means++ weighting,
        // so the two directional groups partition cleanly; a fixed seed pins the shape.
        let dims = 2;
        let mut data = Vec::new();
        for i in 0..20 {
            data.push(10.0_f32 + i as f32 * 0.1);
            data.push(5.0);
        }
        for i in 0..20 {
            data.push(1.0);
            data.push(10.0_f32 + i as f32 * 0.1);
        }
        let result = kmeans(&data, dims, 2, 30, 42, DistanceMetric::InnerProduct);
        let mut sizes = vec![0_usize; 2];
        for &a in &result.assignments {
            sizes[a] += 1;
        }
        assert_eq!(
            sizes,
            vec![20, 20],
            "each directional group must form its own cluster"
        );
    }

    #[test]
    fn test_kmeans_stable_with_fixed_seed() {
        let dims = 4;
        let n = 200;
        let data: Vec<f32> = (0..n * dims).map(|i| (i % 17) as f32 * 0.03).collect();

        let r1 = kmeans(&data, dims, 4, 20, 0xDEAD, DistanceMetric::L2);
        let r2 = kmeans(&data, dims, 4, 20, 0xDEAD, DistanceMetric::L2);

        // 相同种子应产生相同结果
        // EN: The same seed should produce the same result.
        assert_eq!(r1.iterations, r2.iterations);
        assert!((r1.inertia - r2.inertia).abs() < 0.001);
        assert_eq!(r1.centroids, r2.centroids);
    }
}
