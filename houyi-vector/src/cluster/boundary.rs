//! 边界点检测与跨聚类复制。
//! Boundary detection and cross-cluster replication.
//!
//! 当向量到次近中心的距离 ≤ 最近距离 + |最近距离| × (α - 1) 时，判定为边界点，
//! 复制到次近聚类。这消除了聚类方法的主要 recall 损失来源（边界效应）。加性
//! 余量在最近距离非负时与乘法形式等价，且对 IP 的负值距离（内积取负）方向正确。
//! EN: A vector is a boundary point when its second-nearest distance ≤ nearest +
//! |nearest| × (α - 1), and is copied to the second cluster. The additive margin
//! is equivalent to the multiplicative form for non-negative distances and stays
//! direction-correct for negative IP distances (-dot).

use crate::distance;
use crate::DistanceMetric;
use rayon::prelude::*;

/// 边界分配结果：每个向量的主聚类 + 可选次要聚类列表。
/// EN: Boundary assignment: each vector's primary cluster + optional secondary cluster list.
#[derive(Debug, Clone)]
pub struct BoundaryAssignment {
    /// 主聚类
    /// EN: Primary cluster.
    pub primary: usize,
    /// 因边界效应而复制的次要聚类（可为空）
    /// EN: Secondary clusters copied due to the boundary effect (may be empty).
    pub secondaries: Vec<usize>,
}

/// 并行检测边界点并分配聚类（含复制）。
/// EN: Detects boundary points and assigns clusters in parallel (with replication).
///
/// - `vectors`: 平坦向量 [N × dims]
///   EN: - `vectors`: flat vectors [N × dims]
/// - `centroids`: 聚类中心 [C × dims]
///   EN: - `centroids`: cluster centroids [C × dims]
/// - `alpha`: 边界阈值（默认 1.2）。α 越小，复制越少，recall 越低。
///   EN: - `alpha`: boundary threshold (default 1.2). Smaller α means fewer copies, lower recall.
///
/// 返回每个向量的 `BoundaryAssignment`。
/// EN: Returns a `BoundaryAssignment` for each vector.
pub fn assign_with_boundary(
    vectors: &[f32],
    dims: usize,
    centroids: &[f32],
    alpha: f32,
    metric: DistanceMetric,
) -> Vec<BoundaryAssignment> {
    let n = vectors.len() / dims;
    let c = centroids.len() / dims;

    (0..n)
        .into_par_iter()
        .map(|i| {
            let v = &vectors[i * dims..(i + 1) * dims];

            // 找最近和次近中心
            // EN: Find the nearest and second-nearest centroids.
            let (primary, secondary) = find_two_nearest(v, centroids, dims, c, metric);

            let secondaries = if let Some(sec) = secondary {
                let d_primary =
                    dist_fn(metric, v, &centroids[primary * dims..(primary + 1) * dims]);
                let d_sec = dist_fn(metric, v, &centroids[sec * dims..(sec + 1) * dims]);

                // 边界条件见 `is_boundary`：构建期与插入期共用同一判定，
                // 避免两条路径对 IP 正负号/度量距离的处理分叉（OCR 195e6be #04）。
                // EN: The boundary condition lives in `is_boundary`, shared by the
                // build and insert paths so IP sign handling and metric-specific
                // distances cannot drift between them (OCR 195e6be #04).
                if is_boundary(d_primary, d_sec, alpha) {
                    vec![sec]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            BoundaryAssignment {
                primary,
                secondaries,
            }
        })
        .collect()
}

/// 找到最近中心（返回 index）和次近中心（可选，单聚类时为 None）。
/// EN: Finds the nearest centroid (returns its index) and second-nearest (None with one cluster).
pub fn find_two_nearest(
    v: &[f32],
    centroids: &[f32],
    dims: usize,
    num_clusters: usize,
    metric: DistanceMetric,
) -> (usize, Option<usize>) {
    if num_clusters == 1 {
        return (0, None);
    }

    let mut best = 0;
    let mut best_d = dist_fn(metric, v, &centroids[0..dims]);
    let mut second = 1;
    let mut second_d = dist_fn(metric, v, &centroids[dims..2 * dims]);

    if second_d < best_d {
        std::mem::swap(&mut best, &mut second);
        std::mem::swap(&mut best_d, &mut second_d);
    }

    for k in 2..num_clusters {
        let d = dist_fn(metric, v, &centroids[k * dims..(k + 1) * dims]);
        if d < best_d {
            second = best;
            second_d = best_d;
            best = k;
            best_d = d;
        } else if d < second_d {
            second = k;
            second_d = d;
        }
    }

    (best, Some(second))
}

/// 按边界分配结果分组向量索引。返回 `clusters[c] = Vec<(向量索引, is_primary)>`。
/// `is_primary`: true = 主聚类, false = 因边界复制而来。
/// EN: Groups vector indices by boundary assignments. Returns `clusters[c] = Vec<(index, is_primary)>`.
/// EN: `is_primary`: true = primary cluster, false = added by boundary replication.
pub fn group_with_boundary(
    assignments: &[BoundaryAssignment],
    num_clusters: usize,
) -> Vec<Vec<(usize, bool)>> {
    let mut clusters: Vec<Vec<(usize, bool)>> = (0..num_clusters).map(|_| Vec::new()).collect();
    for (vec_idx, a) in assignments.iter().enumerate() {
        clusters[a.primary].push((vec_idx, true));
        for &sec in &a.secondaries {
            clusters[sec].push((vec_idx, false));
        }
    }
    clusters
}

/// 统计因边界复制导致的额外存储开销。
/// EN: Counts the extra storage cost caused by boundary replication.
pub fn replication_overhead(assignments: &[BoundaryAssignment]) -> f64 {
    let primary_count = assignments.len();
    let total_copies: usize = assignments.iter().map(|a| 1 + a.secondaries.len()).sum();
    (total_copies - primary_count) as f64 / primary_count as f64
}

/// 边界判定：次近距离 ≤ 最近距离 + |最近距离| × (α - 1)。
/// EN: Boundary test: second-nearest ≤ nearest + |nearest| × (α - 1).
///
/// IP 距离为 -dot（可为负）：直接乘 α 会把负值变得更负、方向反转（次近“更
/// 相似”的反而复制，真正边缘点漏掉）；加性形式对正负号都正确，且
/// `d_primary ≥ 0` 时与旧式 `d_primary × α` 等价（OCR #14）。
/// EN: IP distances are -dot and may be negative; multiplying directly by α makes
/// larger-magnitude negatives even more negative (reversing the direction: truly
/// similar secondaries get copied while borderline points are missed). The
/// additive form is sign-correct and equals `d_primary × α` whenever
/// `d_primary ≥ 0` (OCR #14).
///
/// 构建期（`assign_with_boundary`）与插入期（`ClusteredHnsw::insert`）必须共用
/// 本函数：任一侧单独硬编码旧式乘法都会让两条路径的复制集合不一致（OCR
/// 195e6be #04）。
/// EN: Build time (`assign_with_boundary`) and insert time
/// (`ClusteredHnsw::insert`) must share this predicate; hardcoding the old
/// multiplicative test on either side makes the replicated sets disagree
/// (OCR 195e6be #04).
#[inline]
pub(crate) fn is_boundary(d_primary: f32, d_sec: f32, alpha: f32) -> bool {
    alpha > 0.0 && d_sec <= d_primary + d_primary.abs() * (alpha - 1.0)
}

/// 按度量计算两点距离：聚类内所有路径（构建、插入、路由、K-Means、
/// FlatClustered 检索）共用这一份分发，未来度量语义变化只改这里。
/// EN: Distance between two points under the given metric: every clustering
/// path (build, insert, routing, K-Means, FlatClustered search) shares this
/// single dispatch, so a future metric-semantics change is made in one place
/// (OCR cf1d19f #03).
#[inline]
pub(crate) fn dist_fn(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => distance::l2_squared(a, b),
        DistanceMetric::Cosine => distance::cosine(a, b),
        DistanceMetric::InnerProduct => distance::inner_product_distance(a, b),
        DistanceMetric::L1 => distance::l1_distance(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceMetric;

    #[test]
    fn test_boundary_detection_near_edge() {
        let dims = 2;
        // 两个聚类中心分别在 (0,0) 和 (10,10)
        // EN: Two centroids at (0,0) and (10,10).
        let centroids = vec![0.0_f32, 0.0, 10.0, 10.0];
        // 向量在正中间: (5,5) → 距离两个中心相等，应被判定为边界点
        // EN: Vector in the middle: (5,5) → equidistant to both centroids, so it is a boundary point.
        let vectors = vec![5.0_f32, 5.0];
        let assignments = assign_with_boundary(&vectors, dims, &centroids, 1.2, DistanceMetric::L2);

        assert_eq!(assignments.len(), 1);
        // 应该在两个聚类之间
        // EN: It should appear in both clusters.
        assert!(!assignments[0].secondaries.is_empty());
    }

    #[test]
    fn test_boundary_detection_far_from_edge() {
        let dims = 2;
        let centroids = vec![0.0_f32, 0.0, 10.0, 10.0];
        // 向量紧邻第一个中心: (1,1) → 距离 (0,0)=√2 ≈1.4, 距离 (10,10)=√162 ≈12.7
        // EN: Vector next to the first centroid: (1,1) → dist to (0,0)=√2 ≈1.4, to (10,10)=√162 ≈12.7.
        let vectors = vec![1.0_f32, 1.0];
        let assignments = assign_with_boundary(&vectors, dims, &centroids, 1.2, DistanceMetric::L2);

        assert_eq!(assignments.len(), 1);
        // 不应该有次要聚类
        // EN: It should have no secondary clusters.
        assert!(assignments[0].secondaries.is_empty());
    }

    #[test]
    fn test_boundary_detection_inner_product_uses_additive_margin() {
        // OCR #14：IP 距离为 -dot（负值）。旧式 `d_primary × α` 把 -1 变成
        // -1.2，方向反转：d_sec=-0.9 的真正的边缘点被判为非边界。加性余量
        // 得到阈值 -0.8，正确复制；远点（d_sec=0）仍不复制。
        // EN: OCR #14 — IP distances are -dot (negative). The old `d_primary × α`
        // turned -1 into -1.2 (direction reversed), missing a genuine borderline
        // point at d_sec=-0.9. The additive margin yields -0.8 and copies it,
        // while the far point (d_sec=0) is still not copied.
        let dims = 2;
        // c0=(1,0) 与 c1=(0.9,0.1)：v=(1,0) 时 d0=-1，d1=-0.9。
        // EN: c0=(1,0), c1=(0.9,0.1): for v=(1,0), d0=-1 and d1=-0.9.
        let centroids = vec![1.0_f32, 0.0, 0.9, 0.1];
        let vectors = vec![1.0_f32, 0.0];
        let assignments = assign_with_boundary(
            &vectors,
            dims,
            &centroids,
            1.2,
            DistanceMetric::InnerProduct,
        );
        assert_eq!(assignments[0].primary, 0);
        assert_eq!(assignments[0].secondaries, vec![1]);

        // c1=(0,1)：d1=0，远不构成边界。
        // EN: c1=(0,1): d1=0, far from the boundary.
        let far = vec![1.0_f32, 0.0, 0.0, 1.0];
        let assignments =
            assign_with_boundary(&vectors, dims, &far, 1.2, DistanceMetric::InnerProduct);
        assert!(assignments[0].secondaries.is_empty());
    }

    #[test]
    fn test_replication_overhead_statistic() {
        let assignments = vec![
            BoundaryAssignment {
                primary: 0,
                secondaries: vec![],
            },
            BoundaryAssignment {
                primary: 0,
                secondaries: vec![1],
            },
            BoundaryAssignment {
                primary: 1,
                secondaries: vec![],
            },
            BoundaryAssignment {
                primary: 1,
                secondaries: vec![],
            },
        ];
        // 4 个主分配 + 1 个额外复制 = 总共 5 份 → 25% 开销
        // EN: 4 primary assignments + 1 extra copy = 5 total → 25% overhead.
        let overhead = replication_overhead(&assignments);
        assert!((overhead - 0.25).abs() < 0.01);
    }
}
