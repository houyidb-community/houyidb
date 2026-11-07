//! 路由器：查询路由到 top-nprobe 聚类 + 向量分配到聚类。
//! Router: query-to-cluster routing + vector assignment.

use super::boundary::dist_fn;
use crate::DistanceMetric;
use rayon::prelude::*;

/// 计算查询向量到所有聚类中心的距离，返回最近的 `nprobe` 个聚类 ID。
/// EN: Computes distances from the query vector to all centroids; returns the nearest `nprobe` ids.
pub fn route_query(
    query: &[f32],
    centroids: &[f32],
    dims: usize,
    nprobe: usize,
    metric: DistanceMetric,
) -> Vec<usize> {
    let num_clusters = centroids.len() / dims;
    let nprobe = nprobe.min(num_clusters);

    let mut dists: Vec<(usize, f32)> = (0..num_clusters)
        .into_par_iter()
        .map(|cid| {
            let c = &centroids[cid * dims..(cid + 1) * dims];
            let d = dist_fn(metric, query, c);
            (cid, d)
        })
        .collect();

    // 部分排序取 top nprobe
    // EN: Partial sort to take the top nprobe.
    dists.select_nth_unstable_by(nprobe - 1, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    dists.truncate(nprobe);
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    dists.into_iter().map(|(cid, _)| cid).collect()
}

/// 并行分配向量到聚类（仅主聚类，不含边界复制）。
/// 返回每个向量的主聚类 ID。
/// EN: Assigns vectors to clusters in parallel (primary cluster only, no boundary replication).
/// EN: Returns the primary cluster id for each vector.
pub fn assign_vectors(
    vectors: &[f32],
    dims: usize,
    centroids: &[f32],
    metric: DistanceMetric,
) -> Vec<usize> {
    let n = vectors.len() / dims;
    let num_clusters = centroids.len() / dims;

    (0..n)
        .into_par_iter()
        .map(|i| {
            let v = &vectors[i * dims..(i + 1) * dims];
            let mut best_c = 0;
            let mut best_d = dist_fn(metric, v, &centroids[0..dims]);
            for c in 1..num_clusters {
                let d = dist_fn(metric, v, &centroids[c * dims..(c + 1) * dims]);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            best_c
        })
        .collect()
}

/// 按聚类分组向量索引。
/// 返回 `clusters[c] = Vec<原始向量索引>`
/// EN: Groups vector indices by cluster.
/// EN: Returns `clusters[c] = Vec<original vector index>`.
pub fn group_by_cluster(assignments: &[usize], num_clusters: usize) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = (0..num_clusters).map(|_| Vec::new()).collect();
    for (vec_idx, &cluster_id) in assignments.iter().enumerate() {
        clusters[cluster_id].push(vec_idx);
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceMetric;

    #[test]
    fn test_route_query_returns_nprobe_clusters() {
        let dims = 4;
        let centroids: Vec<f32> = (0..8 * dims).map(|i| i as f32).collect(); // 8 个假中心
        let query: Vec<f32> = (0..dims).map(|i| i as f32 * 0.5).collect();

        let top = route_query(&query, &centroids, dims, 3, DistanceMetric::L2);
        assert_eq!(top.len(), 3);
        // 所有返回的聚类 ID 应在有效范围
        // EN: All returned cluster ids must be within the valid range.
        assert!(top.iter().all(|&c| c < 8));
    }

    #[test]
    fn test_assign_vectors_each_vector_has_valid_cluster() {
        let dims = 3;
        let n = 100;
        let vectors: Vec<f32> = (0..n * dims).map(|i| (i % 11) as f32 * 0.1).collect();
        let centroids: Vec<f32> = (0..5 * dims).map(|i| (i % 7) as f32 * 0.2).collect();

        let assignments = assign_vectors(&vectors, dims, &centroids, DistanceMetric::Cosine);
        assert_eq!(assignments.len(), n);
        assert!(assignments.iter().all(|&c| c < 5));
    }

    #[test]
    fn test_group_by_cluster_preserves_total_count() {
        let assignments = vec![0, 1, 0, 2, 1, 1, 2, 0];
        let groups = group_by_cluster(&assignments, 3);

        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 8);
        assert_eq!(groups[0], vec![0, 2, 7]);
        assert_eq!(groups[1], vec![1, 4, 5]);
        assert_eq!(groups[2], vec![3, 6]);
    }
}
