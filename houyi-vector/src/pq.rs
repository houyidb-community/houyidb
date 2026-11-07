//! Product Quantization: 将向量压缩为短码，用 ADC 近似距离。
//! Product Quantization: compress vectors to short codes, approximate distance via ADC.
//!
//! 参数: m 段 × k 码本/段（默认 256 → 8-bit 码）。
//! 128D: m=16 → 16 bytes/向量 (32x 压缩)
//! 768D: m=48 → 48 bytes/向量 (64x 压缩)
//! 1536D: m=96 → 96 bytes/向量 (64x 压缩)

use crate::DistanceMetric;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// PQ 配置。
/// EN: PQ configuration.
#[derive(Debug, Clone)]
pub struct PqConfig {
    /// 段数 m（每段维度 = dims / m）
    /// EN: Number of segments m (sub-dimension = dims / m).
    pub m: usize,
    /// 每段码本大小 k（默认 256 = 8-bit）
    /// EN: Codebook size k per segment (default 256 = 8-bit).
    pub k: usize,
    /// K-Means 训练迭代次数
    /// EN: K-Means training iterations.
    pub train_iters: usize,
}

impl PqConfig {
    pub fn for_dims(dims: usize) -> Self {
        // 移动端优先精度：每段 16 维（更少段 = 更大子向量 = 更高精度）
        // EN: Mobile-first precision: 16 dims per segment (fewer segments = larger subvectors).
        let m = if dims <= 128 {
            (dims / 16).max(4)
        } else if dims <= 768 {
            (dims / 16).max(8)
        } else {
            (dims / 16).max(12)
        };
        PqConfig {
            m,
            k: 256,
            train_iters: 15,
        }
    }

    pub fn sub_dims(&self, full_dims: usize) -> usize {
        full_dims / self.m
    }
}

/// PQ 索引：码本 + 压缩编码表。
/// EN: PQ index: codebooks + compressed codes.
pub struct PqIndex {
    config: PqConfig,
    #[allow(dead_code)]
    dims: usize,
    sub_dims: usize,
    /// 码本: [m × k × sub_dims] 平坦
    /// EN: Codebooks: [m × k × sub_dims], flat.
    pub codebooks: Vec<f32>,
    /// 编码: [n × m] u8 平坦
    /// EN: Codes: [n × m] u8, flat.
    codes: Vec<u8>,
    /// 距离度量
    /// EN: Distance metric.
    metric: DistanceMetric,
}

impl PqIndex {
    /// 训练 PQ：运行 m 个独立 K-Means。
    /// EN: Train PQ: run m independent K-Means jobs.
    pub fn train(vectors: &[f32], dims: usize, config: &PqConfig) -> Self {
        let n = vectors.len() / dims;
        let sub_dims = config.sub_dims(dims);
        let m = config.m;
        let k = config.k;

        assert_eq!(dims, m * sub_dims, "dims must be divisible by m");

        // 训练 m 个码本（并行）
        // EN: Train m codebooks (in parallel).
        let codebooks: Vec<f32> = (0..m)
            .into_par_iter()
            .flat_map(|seg| {
                // 提取第 seg 段的所有子向量
                // EN: Extract all subvectors of segment seg.
                let subvecs: Vec<f32> = (0..n)
                    .flat_map(|i| {
                        let start = i * dims + seg * sub_dims;
                        vectors[start..start + sub_dims].to_vec()
                    })
                    .collect();
                // 对该段做 K-Means
                // EN: Run K-Means on that segment.
                kmeans_segment(&subvecs, sub_dims, k, config.train_iters, seg as u64)
            })
            .collect();

        // 编码所有向量
        // EN: Encode all vectors.
        let codes: Vec<u8> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let v = &vectors[i * dims..(i + 1) * dims];
                (0..m)
                    .map(|seg| {
                        let sub = &v[seg * sub_dims..(seg + 1) * sub_dims];
                        let cb_start = seg * k * sub_dims;
                        nearest_centroid_u8(
                            sub,
                            &codebooks[cb_start..cb_start + k * sub_dims],
                            sub_dims,
                            k,
                        )
                    })
                    .collect::<Vec<u8>>()
            })
            .collect();

        PqIndex {
            config: config.clone(),
            dims,
            sub_dims,
            codebooks,
            codes,
            metric: DistanceMetric::L2,
        }
    }

    pub fn num_vectors(&self) -> usize {
        self.codes.len() / self.config.m
    }

    /// ADC 距离: query vs PQ-encoded vector i
    /// EN: ADC distance: query vs PQ-encoded vector i.
    pub fn adc_distance(&self, query: &[f32], vec_idx: usize) -> f32 {
        let m = self.config.m;
        let sub_dims = self.sub_dims;
        let k = self.config.k;
        let code_start = vec_idx * m;
        let mut dist = 0.0f32;
        for seg in 0..m {
            let code = self.codes[code_start + seg] as usize;
            let cb_start = seg * k * sub_dims + code * sub_dims;
            let q_sub = &query[seg * sub_dims..(seg + 1) * sub_dims];
            let c_sub = &self.codebooks[cb_start..cb_start + sub_dims];
            dist += dist_fn(self.metric, q_sub, c_sub);
        }
        dist
    }

    /// 批量 ADC 距离: query vs 一个区间内的所有向量
    /// EN: Batch ADC distance: query vs all vectors in a range.
    pub fn adc_batch(&self, query: &[f32], start: usize, end: usize) -> Vec<(usize, f32)> {
        (start..end)
            .map(|i| (i, self.adc_distance(query, i)))
            .collect()
    }

    /// 预计算查询到所有码字的距离表（加速批量查询）。
    /// EN: Precomputes the query-to-codeword distance table (speeds up batch queries).
    pub fn precompute_table(&self, query: &[f32]) -> Vec<f32> {
        let m = self.config.m;
        let k = self.config.k;
        let sub_dims = self.sub_dims;
        let mut table = vec![0.0f32; m * k];
        for seg in 0..m {
            let q_sub = &query[seg * sub_dims..(seg + 1) * sub_dims];
            for c in 0..k {
                let cb = &self.codebooks
                    [seg * k * sub_dims + c * sub_dims..seg * k * sub_dims + (c + 1) * sub_dims];
                table[seg * k + c] = dist_fn(self.metric, q_sub, cb);
            }
        }
        table
    }

    /// 查表式 SDC (Symmetric Distance Computation) — 极速。
    /// EN: Table-lookup SDC (Symmetric Distance Computation) — very fast.
    pub fn sdc_distance(&self, table: &[f32], vec_idx: usize) -> f32 {
        let m = self.config.m;
        let k = self.config.k;
        let code_start = vec_idx * m;
        let mut dist = 0.0f32;
        for seg in 0..m {
            let code = self.codes[code_start + seg] as usize;
            dist += table[seg * k + code];
        }
        dist
    }

    /// 查表式批量检索 top-k。
    /// EN: Table-lookup batch top-k search.
    pub fn search_table(&self, table: &[f32], k: usize) -> Vec<(usize, f32)> {
        let n = self.num_vectors();
        let mut scored: Vec<(usize, f32)> =
            (0..n).map(|i| (i, self.sdc_distance(table, i))).collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    pub fn memory_bytes(&self) -> usize {
        let codebook_bytes = self.codebooks.len() * 4;
        let code_bytes = self.codes.len();
        codebook_bytes + code_bytes
    }
}

/// 段内 K-Means（小规模，串行即可）。
/// EN: Per-segment K-Means (small scale; serial is fine).
fn kmeans_segment(subvecs: &[f32], sub_dims: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = subvecs.len() / sub_dims;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = vec![0.0f32; k * sub_dims];
    let mut assignments = vec![0usize; n];

    // 随机初始化
    // EN: Random initialization.
    for c in 0..k {
        let idx = rng.gen_range(0..n);
        centroids[c * sub_dims..(c + 1) * sub_dims]
            .copy_from_slice(&subvecs[idx * sub_dims..(idx + 1) * sub_dims]);
    }

    for _ in 0..iters {
        // E-step
        let mut changed = false;
        for i in 0..n {
            let v = &subvecs[i * sub_dims..(i + 1) * sub_dims];
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let d = l2_dist(v, &centroids[c * sub_dims..(c + 1) * sub_dims], sub_dims);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                changed = true;
            }
            assignments[i] = best_c;
        }
        if !changed {
            break;
        }

        // M-step
        centroids.fill(0.0);
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            let v = &subvecs[i * sub_dims..(i + 1) * sub_dims];
            let start = c * sub_dims;
            for d in 0..sub_dims {
                centroids[start + d] += v[d];
            }
            counts[c] += 1;
        }
        for (c, &count) in counts.iter().enumerate() {
            if count > 0 {
                let inv = 1.0 / count as f32;
                let start = c * sub_dims;
                for d in 0..sub_dims {
                    centroids[start + d] *= inv;
                }
            }
        }
    }
    centroids
}

fn nearest_centroid_u8(sub: &[f32], codebook: &[f32], sub_dims: usize, k: usize) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::MAX;
    for c in 0..k {
        let d = l2_dist(sub, &codebook[c * sub_dims..(c + 1) * sub_dims], sub_dims);
        if d < best_d {
            best_d = d;
            best = c as u8;
        }
    }
    best
}

#[inline]
fn l2_dist(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[inline]
fn dist_fn(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_dist(a, b, a.len()),
        // PQ 仅支持 L2（构建/查表都按平方欧氏编码）；服务端在建索引前已拒绝
        // 非 L2 组合。这里必须显式 panic 而不是返回「看起来合理」的距离：若
        // 私有 metric 字段将来被改写，ADC 的码本/查表仍是平方欧氏，返回
        // IP/L1 数值只会静默算错；Cosine 同理——ADC 逐段余弦之和并非整向量
        // 余弦，属于最危险的一类「看似合理但算错」。宁可当场暴露（OCR #17、
        // 195e6be #05）。
        // EN: PQ only supports L2 (both training and table lookup encode squared
        // Euclidean), and the server rejects non-L2 before building. Panic
        // explicitly rather than returning a plausible-looking distance: if the
        // private metric field ever gains a setter, the ADC codebooks/lookup are
        // still squared-Euclidean, so IP/L1 numbers would be silently wrong.
        // Cosine is no different — the sum of per-segment cosine values is not the
        // full-vector cosine, the most dangerous kind of "plausible but wrong".
        // Fail loudly instead (OCR #17, 195e6be #05).
        DistanceMetric::Cosine | DistanceMetric::InnerProduct | DistanceMetric::L1 => {
            unreachable!("PQ index only supports L2; cosine/IP/L1 are rejected before construction")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_train_and_search() {
        let dims = 32;
        let n = 500;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vectors = Vec::with_capacity(n * dims);
        for _ in 0..n {
            for _ in 0..dims {
                vectors.push(rng.gen::<f32>());
            }
        }

        let config = PqConfig {
            m: 4,
            k: 64,
            train_iters: 10,
        };
        let pq = PqIndex::train(&vectors, dims, &config);

        assert_eq!(pq.num_vectors(), n);
        assert_eq!(pq.codes.len(), n * 4); // m=4 bytes per vector

        // 自查询：自身向量应排在前面。
        // EN: Self-query: own vector should rank high.
        let q = &vectors[100 * dims..101 * dims];
        let table = pq.precompute_table(q);
        let results = pq.search_table(&table, 5);
        // PQ 编码向量应接近其自身位置。
        // EN: The PQ-encoded vector should be near its own position.
        assert!(
            results.iter().any(|(i, _)| *i == 100),
            "self should be in top-5 after PQ"
        );

        // 内存：500 × 4 字节 = 2KB 编码 + 码本。
        // EN: Memory: 500 × 4 bytes = 2KB codes + codebook.
        let mem = pq.memory_bytes();
        assert!(mem < 100_000, "PQ memory should be small, got {mem}");
    }

    #[test]
    fn test_pq_compression_ratio() {
        let dims = 128;
        let n = 1000;
        let config = PqConfig::for_dims(128);
        // m=8, k=256 → 每向量 8 字节（64 倍压缩）。
        // EN: m=8, k=256 → 8 bytes per vector (64x compression).
        assert_eq!(config.m, 8);
        assert_eq!(config.k, 256);

        let raw = n * dims * 4; // 512KB
        let pq = n * config.m; // 8KB
        let ratio = raw / pq;
        assert!(
            ratio >= 32,
            "compression ratio should be ≥ 32x, got {ratio}x"
        );
    }
}
