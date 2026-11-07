//! 向量距离：portable 热点路径按 8 元块累加，减少循环开销并利于 LLVM 向量化。
//! Vector distances: 8-element block accumulation for fewer loop iterations and better auto-vectorization.
//!
//! **aarch64 + `feature = "simd-neon"`**：L2/点积在 **NEON** 可用时走 `distance_aarch64_neon`；进程内**仅首测一次** `is_aarch64_feature_detected!("neon")`（HNSW 热路径中不可每次探测）。
//! **x86_64 + `feature = "simd-avx2"`**：L2/点积在 **AVX2+FMA** 可用时走 `distance_x86_avx2`；同样**缓存** AVX2/FMA 探测结果。
//! Default（无 `simd-neon`/`simd-avx2`）不编入上述分派，与 portable 全路径一致。
//! `lib.rs` 的 `simd-avx512` 为另一套 L2 入口，仍与本模块分治。
//! `simd-avx512` in `lib.rs` is a separate L2 entry, kept isolated.

const CHUNK: usize = 8;

/// NEON：运行期 `is_aarch64_feature_detected!` 在百万级距离计算下开销过大，进程内缓存一次。
/// NEON: feature probe is too expensive in HNSW hot path; cache once per process.
#[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
#[inline]
fn simd_neon_dist_runtime_ok() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

/// AVX2+FMA：同上，两路 `is_x86_feature_detected!` 每调用各一次在构图阶段不可接受；合并为一次读缓存。
/// AVX2+FMA: same idea—two probes per call are folded into one `OnceLock`.
#[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
#[inline]
fn simd_avx2_fma_dist_runtime_ok() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

/// 纯可移植 L2 平方和。
/// EN: Portable L2² only.
#[inline(always)]
fn l2_squared_portable(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut i = 0usize;
    while i + CHUNK <= n {
        let mut block = 0.0_f32;
        for j in 0..CHUNK {
            let d = a[i + j] - b[i + j];
            block += d * d;
        }
        sum += block;
        i += CHUNK;
    }
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// L2 距离平方和（与既有 `l2_distance` 语义一致：不开方）。
/// Sum of squared differences (same contract as `crate::l2_distance`).
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        if n >= 4 && simd_neon_dist_runtime_ok() {
            return unsafe { crate::distance_aarch64_neon::l2_squared(a, b, n) };
        }
    }
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        if n >= 8 && simd_avx2_fma_dist_runtime_ok() {
            return unsafe { crate::distance_x86_avx2::l2_squared(a, b, n) };
        }
    }
    l2_squared_portable(a, b, n)
}

/// Cosine 距离：`1 - cos_sim`。
/// Cosine distance `1 - cos_sim`.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        if n >= 4 && simd_neon_dist_runtime_ok() {
            return unsafe { crate::distance_aarch64_neon::cosine(a, b, n) };
        }
    }
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        if n >= 8 && simd_avx2_fma_dist_runtime_ok() {
            // AVX2 cosine 路径——暂委托给可移植实现；后续再补 AVX2 cosine intrinsic。
            // EN: AVX2 cosine path — delegates to portable for now; add AVX2 cosine intrinsic later.
        }
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    let mut i = 0usize;
    while i + CHUNK <= n {
        let mut bd = 0.0_f32;
        let mut bna = 0.0_f32;
        let mut bnb = 0.0_f32;
        for j in 0..CHUNK {
            let x = a[i + j];
            let y = b[i + j];
            bd += x * y;
            bna += x * x;
            bnb += y * y;
        }
        dot += bd;
        na += bna;
        nb += bnb;
        i += CHUNK;
    }
    while i < n {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
        i += 1;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (na.sqrt() * nb.sqrt()))
}

/// 纯可移植点积。
/// EN: Portable dot only.
#[inline(always)]
fn dot_product_f32_portable(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut dot = 0.0_f32;
    let mut i = 0usize;
    while i + CHUNK <= n {
        let mut bd = 0.0_f32;
        for j in 0..CHUNK {
            bd += a[i + j] * b[i + j];
        }
        dot += bd;
        i += CHUNK;
    }
    while i < n {
        dot += a[i] * b[i];
        i += 1;
    }
    dot
}

/// 等长向量点积；与 HNSW 中「已 L2 单位化余弦」分支共享分块实现。
/// Dot product for equal-length slices; shared pre-normalized cosine hot path in HNSW.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        if n >= 4 && simd_neon_dist_runtime_ok() {
            return unsafe { crate::distance_aarch64_neon::dot_f32(a, b, n) };
        }
    }
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        if n >= 8 && simd_avx2_fma_dist_runtime_ok() {
            return unsafe { crate::distance_x86_avx2::dot_f32(a, b, n) };
        }
    }
    dot_product_f32_portable(a, b, n)
}

/// L1 / 曼哈顿距离 `Σ|a_i - b_i|`（开源内核对任何度量都保持「越小越近」，
/// 因此 L1 直接用原值，不做取负）。
/// EN: L1 / Manhattan distance `Σ|a_i - b_i|` (the kernel keeps "smaller =
/// closer" for every metric, so L1 is used as-is with no negation).
#[inline]
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0_f32;
    let mut i = 0usize;
    while i + CHUNK <= n {
        let mut block = 0.0_f32;
        for j in 0..CHUNK {
            block += (a[i + j] - b[i + j]).abs();
        }
        sum += block;
        i += CHUNK;
    }
    while i < n {
        sum += (a[i] - b[i]).abs();
        i += 1;
    }
    sum
}

/// 内积距离 `-a·b`：内积越大越相似，取负后与「越小越近」的全序约定一致。
/// 对外分数会在服务端再取负回正（`-d = a·b`），因此这里的符号不影响排序。
/// EN: Inner-product distance `-a·b`: a larger dot means more similar, so the
/// negation keeps the kernel's "smaller = closer" ordering. The server flips the
/// sign back when producing user-facing scores (`-d = a·b`), so ranking is
/// unaffected by the negation here.
#[inline]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot_product_f32(a, b)
}

/// 输入已 **L2 单位化** 时的余弦距离：`1 - dot`（与 `HnswGraph` 构图/检索余弦约定一致）。
/// Cosine distance when both inputs are already L2-normalized: `1 - dot`.
#[inline]
pub fn cosine_distance_prenormalized(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_f32(a, b);
    1.0 - dot.clamp(-1.0, 1.0)
}

/// 原地 L2 单位化；零范数带（norm² ≤ f32::EPSILON）保持原样，与图路径
/// `normalize_for_cosine` 的阈值一致。
/// EN: In-place L2 normalization; the zero-norm band (norm² ≤ f32::EPSILON) is
/// left untouched, matching the graph path's `normalize_for_cosine` threshold.
#[inline]
pub fn normalize_in_place(v: &mut [f32]) {
    let norm2 = dot_product_f32(v, v);
    if norm2 > f32::EPSILON {
        let inv = norm2.sqrt().recip();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_ref(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn cosine_ref(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        1.0 - (dot / (na.sqrt() * nb.sqrt()))
    }

    #[test]
    fn l2_squared_matches_reference_dims() {
        for dims in [1usize, 7, 8, 9, 15, 16, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|i| (i % 17) as f32 * 0.03).collect();
            let b: Vec<f32> = (0..dims).map(|i| (i % 13) as f32 * 0.07).collect();
            let x = l2_squared(&a, &b);
            let y = l2_ref(&a, &b);
            let err = (x - y).abs();
            assert!(
                err < 1e-4 || err / y.max(1e-12) < 1e-5,
                "dims={dims} got {x} ref {y} err {err}"
            );
        }
    }

    #[test]
    fn cosine_matches_reference_dims() {
        for dims in [1usize, 7, 8, 9, 15, 16, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|i| (i % 17) as f32 * 0.03).collect();
            let b: Vec<f32> = (0..dims).map(|i| (i % 13) as f32 * 0.07).collect();
            let x = cosine(&a, &b);
            let y = cosine_ref(&a, &b);
            let err = (x - y).abs();
            assert!(
                err < 1e-4 || err / y.max(1e-12).abs().max(1e-6) < 1e-4,
                "dims={dims} got {x} ref {y} err {err}"
            );
        }
    }

    fn prenormalized_ref(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        for i in 0..a.len().min(b.len()) {
            dot += a[i] * b[i];
        }
        1.0 - dot.clamp(-1.0, 1.0)
    }

    #[test]
    fn l1_distance_matches_reference() {
        for dims in [1usize, 7, 8, 9, 15, 16, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|i| (i % 17) as f32 * 0.03 - 0.2).collect();
            let b: Vec<f32> = (0..dims).map(|i| (i % 13) as f32 * 0.07 + 0.1).collect();
            let x = l1_distance(&a, &b) as f64;
            // 参考值用 f64 累加：dims=1537 时和约 800+，f32 累加噪音可达 1e-3
            // 量级，绝对容差在 f32 参考里会变成 ~1e-6 相对容差，属于 CI 抖动源
            // （OCR #16）。与 `l2_squared_matches_reference_dims` 同口径用绝对或
            // 相对二选一。
            // EN: Accumulate the reference in f64: at dims=1537 the sum is ~800+,
            // where f32 accumulation noise reaches ~1e-3, making an absolute
            // tolerance on an f32 reference an effective ~1e-6 relative bound —
            // a CI flake source (OCR #16). Use the absolute-or-relative criterion
            // from `l2_squared_matches_reference_dims`.
            let y: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(p, q)| ((*p as f64) - (*q as f64)).abs())
                .sum();
            let err = (x - y).abs();
            assert!(
                err < 1e-5 || err / y.abs().max(1e-12) < 1e-5,
                "dims={dims} got {x} ref {y} err {err}"
            );
        }
    }

    #[test]
    fn inner_product_distance_is_negated_dot() {
        let a = [0.5_f32, -1.0, 2.0, 0.25];
        let b = [2.0_f32, 3.0, -0.5, 4.0];
        let dot = dot_product_f32(&a, &b);
        assert_eq!(inner_product_distance(&a, &b), -dot);
        // 更相似（点积更大）=> 距离更小；与内核「越小越近」约定一致。
        // EN: More similar (larger dot) => smaller distance, matching the kernel order.
        let c = [0.5_f32, -1.0, 2.0, 0.25];
        assert!(inner_product_distance(&a, &c) < inner_product_distance(&a, &b));
    }

    #[test]
    fn cosine_prenormalized_matches_scalar_loop() {
        for dims in [1usize, 3, 8, 1536] {
            let a: Vec<f32> = (0..dims).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
            let b: Vec<f32> = (0..dims).map(|i| (i % 5) as f32 * 0.11 - 0.2).collect();
            let x = cosine_distance_prenormalized(&a, &b);
            let y = prenormalized_ref(&a, &b);
            assert!((x - y).abs() < 1e-5, "dims={dims} got {x} ref {y}");
        }
    }
}
