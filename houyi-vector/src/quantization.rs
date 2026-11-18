//! 图侧 int8 标量量化（P2）：逐向量仿射映射，查询恒 f32（非对称量化）。
//!
//! 设计取舍：
//! - **只压缩图侧向量**，主表/hot buffer/builder queue 永远保留原始 f32/f16 值；
//!   检索时 int8 图只负责「取候选」，最终 top-k 由主表原始向量精确重排（rescore）
//!   决定，因此量化误差不影响最终排序精度。
//! - 逐向量仿射（per-vector affine）而不是逐维/全局 scale：实现零依赖、解码只需
//!   两个标量，且对任意维度分布都能把 256 个码位铺满当前值域，误差界为 scale/2。
//! - 查询向量不量化（非对称量化）：查询侧保持 f32，避免「双端量化」的复合误差。
//! - 余弦路径沿用图内既有约定「存储侧预归一化、距离 = 1 - dot」：量化发生在归一化
//!   之后，反量化向量的范数仍近似 1。
//!
//! EN: Graph-side int8 scalar quantization (P2): per-vector affine mapping with
//! always-f32 queries (asymmetric quantization).
//!
//! Design trade-offs:
//! - Only graph-side vectors are compressed; the main store / hot buffer / builder
//!   queue always keep raw f32/f16 values. The int8 graph only proposes candidates
//!   and the final top-k is decided by exact rescoring on the main store, so
//!   quantization error never affects final ranking precision.
//! - Per-vector affine (rather than per-dimension or global scale): zero external
//!   dependencies, two scalars suffice to decode, and 256 codes always span the
//!   current value range with an error bound of scale/2.
//! - Queries are never quantized (asymmetric quantization): they stay f32, avoiding
//!   the compounded error of double-sided quantization.
//! - The cosine path keeps the graph's "store-side pre-normalization, distance =
//!   1 - dot" contract: quantization happens after normalization, so dequantized
//!   vectors stay approximately unit-norm.

use crate::stored_vector::VectorRef;
use crate::DistanceMetric;
use half::f16;

/// 图侧量化方式。`vector_dtype` 决定主表编码，本枚举只压缩图内距离计算用的副本。
/// EN: Graph-side quantization. `vector_dtype` decides the main-store encoding;
/// this enum only compresses the copy used for in-graph distance evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Quantization {
    /// 不量化（默认；行为与旧版本逐位一致）。
    /// EN: No quantization (default; bit-identical to older releases).
    #[default]
    None,
    /// 逐向量仿射 int8（仅 hnsw 图支持）。
    /// EN: Per-vector affine int8 (hnsw graphs only).
    Int8,
    /// RaBitQ 1 bit/dim 符号码（仅 hnsw 图、仅 l2/cosine）。
    /// EN: RaBitQ 1-bit-per-dim sign codes (hnsw graphs only, l2/cosine only).
    RaBitQ,
}

impl Quantization {
    /// 对外/配置/JSON 使用的规范名。
    /// EN: Canonical name used in APIs, config, and JSON.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Quantization::None => "none",
            Quantization::Int8 => "int8",
            Quantization::RaBitQ => "rabitq",
        }
    }

    /// 从字符串解析（ASCII 大小写不敏感、忽略首尾空白）。只接受规范名：
    /// WAL 的 canonical 列表与 HTTP 校验共用这一份口径，别名会制造第二种真源。
    /// EN: Parses from a string (ASCII case-insensitive, trimmed). Only canonical
    /// names are accepted: the WAL canonical list and HTTP validation share this
    /// single source of truth, and aliases would create a second one.
    pub fn parse_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Quantization::None),
            "int8" => Some(Quantization::Int8),
            "rabitq" => Some(Quantization::RaBitQ),
            _ => None,
        }
    }

    /// 是否启用图侧 int8。
    /// EN: Whether graph-side int8 is enabled.
    pub const fn is_int8(&self) -> bool {
        matches!(self, Quantization::Int8)
    }

    /// 是否启用图侧 RaBitQ。
    /// EN: Whether graph-side RaBitQ is enabled.
    pub const fn is_rabitq(&self) -> bool {
        matches!(self, Quantization::RaBitQ)
    }

    /// 图侧每维额外字节（用于容量估算与内存账目；主表字节另计）。
    /// RaBitQ 实际为 `8·ceil(D/64)/dims` 再加 8 字节标量（D 为补齐到 2 的幂的
    /// 维数），此处按 1 给保守上界；生产路径不依赖该值（仅测试/账目用）。
    /// EN: Extra graph-side bytes per dimension (capacity estimates and memory
    /// accounting; main-store bytes are counted separately). RaBitQ actually
    /// costs `8·ceil(D/64)/dims` plus 8 scalar bytes (D is the next power of
    /// two), so 1 is a conservative upper bound here; production paths do not
    /// rely on this value (tests/accounting only).
    pub const fn bytes_per_dim(&self) -> usize {
        match self {
            Quantization::None => 0,
            Quantization::Int8 => 1,
            Quantization::RaBitQ => 1,
        }
    }

    /// 快照里的稳定字节编码（0=none，1=int8，2=rabitq）；追加新值即可向前兼容。
    /// EN: Stable byte encoding for snapshots (0=none, 1=int8, 2=rabitq);
    /// appending new values stays forward-compatible.
    pub const fn to_snapshot_byte(&self) -> u8 {
        match self {
            Quantization::None => 0,
            Quantization::Int8 => 1,
            Quantization::RaBitQ => 2,
        }
    }

    /// 从快照字节恢复；未知值返回 None，由调用方按损坏/版本不兼容处理。
    /// EN: Restores from a snapshot byte; unknown values return None so callers
    /// treat them as corruption/unsupported version.
    pub const fn from_snapshot_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Quantization::None),
            1 => Some(Quantization::Int8),
            2 => Some(Quantization::RaBitQ),
            _ => None,
        }
    }
}

/// 逐向量仿射 int8 编码：`x' = code * scale + bias`。
/// 端点精确：`code=-128 → min`，`code=127 → max`；内部分量误差 ≤ scale/2。
/// EN: Per-vector affine int8 encoding: `x' = code * scale + bias`.
/// Endpoints are exact: `code=-128 → min`, `code=127 → max`; interior
/// components err by at most scale/2.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Int8Vector {
    /// 量化码位（-128..=127）。
    /// EN: Quantized codes (-128..=127).
    pub(crate) codes: Vec<i8>,
    /// 码位步长；值为常量时退化为 0。
    /// EN: Code step; degenerates to 0 when the value is constant.
    pub(crate) scale: f32,
    /// 零点偏移：`code=0` 对应的值。
    /// EN: Zero-point offset: the value at `code=0`.
    pub(crate) bias: f32,
}

impl Int8Vector {
    /// 用仿射映射量化 f32 分量。非有限分量由服务端写入闸门提前拒绝，本层仍做
    /// 防御性替换（记为 0.0），保证坏输入不会把 scale/bias 污染成 inf/NaN。
    /// EN: Quantizes f32 components with an affine map. Non-finite components are
    /// rejected earlier by the server write gate; this layer still replaces them
    /// defensively (as 0.0) so bad input cannot poison scale/bias with inf/NaN.
    pub(crate) fn quantize(vector: &[f32]) -> Self {
        if vector.is_empty() {
            return Self {
                codes: Vec::new(),
                scale: 0.0,
                bias: 0.0,
            };
        }
        // 非有限分量：调试构建断言（正常写入路径不会到达），release 下按 0.0 处理。
        // EN: Non-finite components: debug builds assert (the normal write path
        // never reaches this) and release builds treat them as 0.0.
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &x in vector {
            debug_assert!(
                x.is_finite(),
                "non-finite component reached int8 quantization"
            );
            let x = if x.is_finite() { x } else { 0.0 };
            min = min.min(x);
            max = max.max(x);
        }
        let range = max - min;
        // 常量向量（range == 0）与极端值域溢出（`max - min` 溢出成 inf）：退化为
        // 全部码位 0。后者用 `0.5*min + 0.5*max` 求中点，避免 `min + max` 再溢出。
        // EN: Constant vectors (range == 0) and extreme-range overflow (`max - min`
        // overflows to inf) degenerate to all-zero codes. The latter uses
        // `0.5*min + 0.5*max` for the midpoint so `min + max` cannot overflow again.
        if !range.is_finite() {
            return Self {
                codes: vec![0; vector.len()],
                scale: 0.0,
                bias: 0.5 * min + 0.5 * max,
            };
        }
        if range == 0.0 {
            return Self {
                codes: vec![0; vector.len()],
                scale: 0.0,
                bias: min,
            };
        }
        let scale = range / 255.0;
        let bias = min + 128.0 * scale;
        let codes = vector
            .iter()
            .map(|&x| {
                let x = if x.is_finite() { x } else { 0.0 };
                // 理论上落在 0..=255；`round` 后仍 clamp 一次，防极端量级下的
                // 浮点舍入把码位推出 i8 值域。
                // EN: Mathematically within 0..=255; clamp once after `round` so
                // extreme magnitudes cannot push the code outside the i8 range.
                let q = ((x - min) / scale).round();
                let q = (q as i32 - 128).clamp(-128, 127);
                q as i8
            })
            .collect();
        Self { codes, scale, bias }
    }

    /// 维度数。
    /// EN: Number of dimensions.
    pub(crate) fn dims(&self) -> usize {
        self.codes.len()
    }

    /// 借用为只读视图（图内距离核的入口）。
    /// EN: Borrows as a read-only view (the in-graph distance-kernel entry).
    pub(crate) fn view(&self) -> Int8Ref<'_> {
        Int8Ref {
            codes: &self.codes,
            scale: self.scale,
            bias: self.bias,
        }
    }

    /// 展开为 f32（仅测试/诊断用；生产路径不应反量化主表值）。
    /// EN: Expands to f32 (tests/diagnostics only; production paths must not
    /// dequantize main-store values).
    #[cfg(test)]
    pub(crate) fn to_f32_vec(&self) -> Vec<f32> {
        self.codes
            .iter()
            .map(|&c| c as f32 * self.scale + self.bias)
            .collect()
    }
}

/// 与 `Int8Vector` 对应的借用视图；不拥有数据，生命周期跟随底层编码。
/// EN: Borrowed view over an `Int8Vector`; owns nothing and lives as long as
/// the underlying encoding.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Int8Ref<'a> {
    /// 量化码位视图。
    /// EN: View over the quantized codes.
    pub(crate) codes: &'a [i8],
    /// 码位步长。
    /// EN: Code step.
    pub(crate) scale: f32,
    /// 零点偏移。
    /// EN: Zero-point offset.
    pub(crate) bias: f32,
}

impl Int8Ref<'_> {
    /// 维度数（int8 codes 与 source 同维）。
    /// EN: Dimensions (int8 codes and source always share the same dims).
    pub(crate) fn dims(&self) -> usize {
        self.codes.len()
    }
}

/// 查询元素提升契约：让 f32/f16 查询复用同一套累加循环。
/// EN: Query-element lifting contract so f32/f16 queries share one accumulation loop.
trait Lift {
    /// 提升为 f32。
    /// EN: Lifts to f32.
    fn lift(self) -> f32;
}

impl Lift for f32 {
    #[inline(always)]
    fn lift(self) -> f32 {
        self
    }
}

impl Lift for f16 {
    #[inline(always)]
    fn lift(self) -> f32 {
        self.to_f32()
    }
}

/// 反量化单个码位。
/// EN: Dequantizes a single code.
#[inline(always)]
fn dequant(code: i8, scale: f32, bias: f32) -> f32 {
    code as f32 * scale + bias
}

/// 任意 dtype 查询 × int8 存储的四个度量累加（长度不等取较短侧）。
/// 查询元素按 `Lift` 提升为 f32 后复用同一套循环；内存序保持 f32 累加，
/// 与现有距离核口径一致。
/// EN: Four-metric accumulation for any-dtype query against int8 storage
/// (unequal lengths use the shorter side). Query elements are lifted to f32
/// via `Lift` and share one loop; accumulation stays in f32 to match the
/// existing distance kernels.
#[inline]
fn metric_query<Q: Lift + Copy>(q: &[Q], stored: Int8Ref<'_>, metric: DistanceMetric) -> f32 {
    let n = q.len().min(stored.dims());
    let q = &q[..n];
    let codes = &stored.codes[..n];
    let (scale, bias) = (stored.scale, stored.bias);
    let lifted = || codes.iter().map(move |&c| dequant(c, scale, bias));
    match metric {
        DistanceMetric::L2 => {
            let sum: f32 = q
                .iter()
                .zip(lifted())
                .map(|(&x, x_stored)| {
                    let d = x.lift() - x_stored;
                    d * d
                })
                .sum();
            sum
        }
        DistanceMetric::InnerProduct => {
            let dot: f32 = q.iter().zip(lifted()).map(|(&x, xs)| x.lift() * xs).sum();
            -dot
        }
        DistanceMetric::L1 => {
            let sum: f32 = q
                .iter()
                .zip(lifted())
                .map(|(&x, xs)| (x.lift() - xs).abs())
                .sum();
            sum
        }
        // 余弦口径与图内既有路径一致：输入已预归一化，距离 = 1 - clamp(dot, -1, 1)。
        // EN: Cosine matches the existing in-graph path: inputs are pre-normalized
        // and the distance is 1 - clamp(dot, -1, 1).
        DistanceMetric::Cosine => {
            let dot: f32 = q.iter().zip(lifted()).map(|(&x, xs)| x.lift() * xs).sum();
            1.0 - dot.clamp(-1.0, 1.0)
        }
    }
}

/// 任意 dtype 查询 × int8 存储的距离（搜索/插入下降路径）；仅按元素类型分派。
/// EN: Distance between any-dtype query and int8 storage (search / insert
/// descent); dispatch is only on the element type.
#[inline]
pub(crate) fn distance_query(
    metric: DistanceMetric,
    query: VectorRef<'_>,
    stored: Int8Ref<'_>,
) -> f32 {
    match query {
        VectorRef::F32(q) => metric_query(q, stored, metric),
        VectorRef::F16(q) => metric_query(q, stored, metric),
    }
}

/// int8 × int8 存储的四个度量距离（启发式选边/剪枝路径，两端同属一张 int8 图）。
/// EN: Four-metric distance between two int8 stored vectors (heuristic selection
/// and pruning, where both endpoints belong to the same int8 graph).
#[inline]
pub(crate) fn distance_stored(metric: DistanceMetric, a: Int8Ref<'_>, b: Int8Ref<'_>) -> f32 {
    let n = a.codes.len().min(b.codes.len());
    let a_codes = &a.codes[..n];
    let b_codes = &b.codes[..n];
    let (a_scale, a_bias) = (a.scale, a.bias);
    let (b_scale, b_bias) = (b.scale, b.bias);
    let pairs = || {
        a_codes
            .iter()
            .zip(b_codes.iter())
            .map(move |(&ca, &cb)| (dequant(ca, a_scale, a_bias), dequant(cb, b_scale, b_bias)))
    };
    match metric {
        DistanceMetric::L2 => {
            let sum: f32 = pairs()
                .map(|(xa, xb)| {
                    let d = xa - xb;
                    d * d
                })
                .sum();
            sum
        }
        DistanceMetric::InnerProduct => {
            let dot: f32 = pairs().map(|(xa, xb)| xa * xb).sum();
            -dot
        }
        DistanceMetric::L1 => {
            let sum: f32 = pairs().map(|(xa, xb)| (xa - xb).abs()).sum();
            sum
        }
        DistanceMetric::Cosine => {
            let dot: f32 = pairs().map(|(xa, xb)| xa * xb).sum();
            1.0 - dot.clamp(-1.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance_between;

    fn sample(dims: usize) -> Vec<f32> {
        (0..dims).map(|i| ((i as f32) - 5.0) / 7.0).collect()
    }

    #[test]
    fn quantization_names_and_snapshot_bytes_roundtrip() {
        assert_eq!(Quantization::None.as_str(), "none");
        assert_eq!(Quantization::Int8.as_str(), "int8");
        assert_eq!(Quantization::RaBitQ.as_str(), "rabitq");
        assert_eq!(Quantization::default(), Quantization::None);
        assert_eq!(Quantization::None.bytes_per_dim(), 0);
        assert_eq!(Quantization::Int8.bytes_per_dim(), 1);
        assert_eq!(Quantization::RaBitQ.bytes_per_dim(), 1);
        assert!(!Quantization::None.is_int8());
        assert!(Quantization::Int8.is_int8());
        assert!(!Quantization::Int8.is_rabitq());
        assert!(Quantization::RaBitQ.is_rabitq());
        for q in [Quantization::None, Quantization::Int8, Quantization::RaBitQ] {
            assert_eq!(
                Quantization::from_snapshot_byte(q.to_snapshot_byte()),
                Some(q)
            );
            assert_eq!(Quantization::parse_str(q.as_str()), Some(q));
        }
        assert_eq!(Quantization::parse_str(" INT8 "), Some(Quantization::Int8));
        assert_eq!(
            Quantization::parse_str("RABITQ"),
            Some(Quantization::RaBitQ)
        );
        assert_eq!(Quantization::parse_str("sq8"), None);
        assert_eq!(
            Quantization::from_snapshot_byte(2),
            Some(Quantization::RaBitQ)
        );
        assert_eq!(Quantization::from_snapshot_byte(3), None);
    }

    #[test]
    fn quantize_roundtrip_error_within_half_scale() {
        let raw = sample(23);
        let q = Int8Vector::quantize(&raw);
        assert_eq!(q.dims(), raw.len());
        assert!(q.scale > 0.0);
        let lifted = q.to_f32_vec();
        for (a, b) in raw.iter().zip(lifted.iter()) {
            assert!(
                (a - b).abs() <= q.scale * 0.5 + 1e-6,
                "component error exceeds half-scale: raw={a} lifted={b} scale={}",
                q.scale
            );
        }
        // 端点精确：最小/最大分量反量化后与原始值逐位相等。
        // EN: Endpoints are exact: the min/max components round-trip bit-exactly.
        let min_raw = raw.iter().copied().fold(f32::INFINITY, f32::min);
        let max_raw = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_idx = raw.iter().position(|&x| x == min_raw).unwrap();
        let max_idx = raw.iter().position(|&x| x == max_raw).unwrap();
        assert_eq!(lifted[min_idx], min_raw);
        assert_eq!(lifted[max_idx], max_raw);
    }

    #[test]
    fn quantize_constant_vector_degenerates_to_zero_scale() {
        let raw = vec![3.25_f32; 9];
        let q = Int8Vector::quantize(&raw);
        assert_eq!(q.scale, 0.0);
        assert_eq!(q.bias, 3.25);
        assert!(q.codes.iter().all(|&c| c == 0));
        assert!(q.to_f32_vec().iter().all(|&x| x == 3.25));
    }

    #[test]
    fn quantize_zero_and_empty_vectors_are_safe() {
        let zeros = vec![0.0_f32; 7];
        let q = Int8Vector::quantize(&zeros);
        assert_eq!(q.scale, 0.0);
        assert_eq!(q.bias, 0.0);
        assert!(q.codes.iter().all(|&c| c == 0));
        assert!(q.to_f32_vec().iter().all(|&x| x == 0.0));

        let empty = Int8Vector::quantize(&[]);
        assert_eq!(empty.dims(), 0);
        assert_eq!(empty.scale, 0.0);
        assert_eq!(empty.bias, 0.0);
        assert!(empty.codes.is_empty());
    }

    #[test]
    fn quantize_huge_range_degenerates_without_nan() {
        // `max - min` 在 f32 上溢出成 inf：退化路径必须给出有限 bias 与全 0 码。
        // EN: `max - min` overflows to inf in f32: the degenerate path must yield a
        // finite bias and all-zero codes.
        let raw = vec![f32::MAX, -f32::MAX, 0.0];
        let q = Int8Vector::quantize(&raw);
        assert_eq!(q.scale, 0.0);
        assert!(q.bias.is_finite(), "bias must stay finite: {}", q.bias);
        assert!(q.codes.iter().all(|&c| c == 0));
        assert!(q.to_f32_vec().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn int8_distances_match_f32_reference_for_all_metrics() {
        let query = sample(16);
        let raw_a = sample(16);
        let raw_b: Vec<f32> = (0..16).map(|i| ((i as f32) - 8.0) / 9.0).collect();
        let qa = Int8Vector::quantize(&raw_a);
        let qb = Int8Vector::quantize(&raw_b);

        for metric in [
            DistanceMetric::L2,
            DistanceMetric::InnerProduct,
            DistanceMetric::L1,
        ] {
            let reference =
                distance_between(VectorRef::F32(&query), VectorRef::F32(&raw_a), metric);
            let got = distance_query(metric, VectorRef::F32(&query), qa.view());
            let scale = reference.abs().max(1.0);
            // 量化噪声界：16 维、值域 ≤ 2 时 3% 相对容差宽松且能捕获语义错误。
            // EN: Quantization-noise bound: 3% relative tolerance is loose for 16
            // dims within a range of 2 while still catching semantic errors.
            assert!(
                (got - reference).abs() <= 0.03 * scale,
                "metric={metric:?} int8={got} reference={reference}"
            );

            let stored_ref =
                distance_between(VectorRef::F32(&raw_a), VectorRef::F32(&raw_b), metric);
            let got_stored = distance_stored(metric, qa.view(), qb.view());
            assert!(
                (got_stored - stored_ref).abs() <= 0.03 * stored_ref.abs().max(1.0),
                "metric={metric:?} stored={got_stored} reference={stored_ref}"
            );
        }

        // 余弦：输入先归一化，int8 距离与 f32 预归一化点积口径一致。
        // EN: Cosine: normalize inputs first; int8 distance matches the f32
        // pre-normalized dot-product contract.
        let mut a = raw_a.clone();
        let mut b = raw_b.clone();
        let mut qv = query.clone();
        crate::distance::normalize_in_place(&mut a);
        crate::distance::normalize_in_place(&mut b);
        crate::distance::normalize_in_place(&mut qv);
        let qa = Int8Vector::quantize(&a);
        let qb = Int8Vector::quantize(&b);
        let reference = 1.0 - (0..16).map(|i| qv[i] * a[i]).sum::<f32>().clamp(-1.0, 1.0);
        let got = distance_query(DistanceMetric::Cosine, VectorRef::F32(&qv), qa.view());
        assert!(
            (got - reference).abs() <= 0.02,
            "cosine int8={got} reference={reference}"
        );
        let got_stored = distance_stored(DistanceMetric::Cosine, qa.view(), qb.view());
        let reference_stored = 1.0 - (0..16).map(|i| a[i] * b[i]).sum::<f32>().clamp(-1.0, 1.0);
        assert!(
            (got_stored - reference_stored).abs() <= 0.02,
            "cosine stored int8={got_stored} reference={reference_stored}"
        );
    }
}
