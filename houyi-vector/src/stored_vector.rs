//! 向量存储编码（P1.1）：默认 f32，可选 f16；写入时只量化一次，检索时按 dtype
//! 分派距离核，查询向量始终保持 f32（非对称量化，精度优于双端 f16）。
//!
//! 设计取舍：
//! - 存储层用枚举而非泛型参数，避免 `HnswGraph`/`HotBufferIndex` 被 `D: Elem`
//!   参数化后传染到 houyi-server 的整个类型面；代价是天量距离计算处有一次
//!   枚举分支，但该分支在逐向量调用中摊销，远小于 f32/f16 转换本身。
//! - f16 与混合组合一律用 f32 累加（与 `distance.rs` 的 portable 核同构），
//!   保证召回口径与 f32 基线只差量化噪声，不引入额外的累加舍入差。
//! - 图/快照路径的余弦仍沿用「存储侧预归一化」约定：f16 在归一化之后再下转，
//!   混合点积仍在 f32 上累加。
//!
//! EN: Vector storage encoding (P1.1): f32 by default, optional f16. Quantization
//! happens once at write time; searches dispatch distance kernels by dtype while
//! the query vector stays f32 (asymmetric quantization, more accurate than f16 on
//! both sides).
//!
//! Design trade-offs:
//! - The storage layer uses an enum instead of a generic `D: Elem` parameter so
//!   `HnswGraph`/`HotBufferIndex` do not infect the whole `houyi-server` type
//!   surface; the cost is one enum branch per distance call, amortized across a
//!   whole vector and far smaller than the f32/f16 conversion itself.
//! - f16 and mixed combinations always accumulate in f32 (same structure as the
//!   portable kernel in `distance.rs`), so recall differs from the f32 baseline
//!   only by quantization noise, not by extra accumulation rounding.
//! - Cosine on the graph/snapshot path keeps the "store-side pre-normalization"
//!   contract: f16 is down-converted after normalization, and mixed dot products
//!   still accumulate in f32.

use crate::distance;
use crate::DistanceMetric;
use half::f16;
use std::borrow::Cow;

/// f16 的最大有限值。写入量化时越界分量饱和到该值，而不是溢出成 ±inf。
/// EN: Largest finite f16 value. Out-of-range components saturate to this at
/// quantization time instead of overflowing to ±inf.
const F16_MAX_FINITE: f32 = 65504.0;

/// 向量存储元素类型。默认 f32；f16 把每维从 4 字节降到 2 字节。
/// EN: Element type of stored vectors. f32 by default; f16 cuts bytes per
/// dimension from 4 to 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum VectorDtype {
    /// 全精度（默认，行为与 beta.7 之前完全一致）。
    /// EN: Full precision (default; behavior identical to releases before f16 support).
    #[default]
    F32,
    /// IEEE 754 binary16；存储内存减半，召回损失由量化噪声界定。
    /// EN: IEEE 754 binary16; halves storage memory, with recall loss bounded by
    /// quantization noise.
    F16,
}

impl VectorDtype {
    /// 对外/配置/JSON 使用的规范名。
    /// EN: Canonical name used in APIs, config, and JSON.
    pub const fn as_str(&self) -> &'static str {
        match self {
            VectorDtype::F32 => "f32",
            VectorDtype::F16 => "f16",
        }
    }

    /// 每维字节数（用于容量估算与内存账目）。
    /// EN: Bytes per dimension (capacity estimates and memory accounting).
    pub const fn bytes_per_dim(&self) -> usize {
        match self {
            VectorDtype::F32 => 4,
            VectorDtype::F16 => 2,
        }
    }

    /// 从字符串解析（ASCII 大小写不敏感）。接受常见别名，便于客户端从
    /// Milvus（`FLOAT16`）等迁移时直接复用其类型名。
    /// EN: Parses from a string (ASCII case-insensitive). Common aliases are
    /// accepted so clients migrating from Milvus (`FLOAT16`) can reuse their
    /// type names directly.
    pub fn parse_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "fp32" | "float" | "single" => Some(VectorDtype::F32),
            "f16" | "float16" | "fp16" | "half" => Some(VectorDtype::F16),
            _ => None,
        }
    }

    /// 快照/清单里的稳定字节编码（0=f32，1=f16）；追加新值即可向前兼容。
    /// EN: Stable byte encoding for snapshots/manifests (0=f32, 1=f16); appending
    /// new values is forward-compatible.
    pub const fn to_snapshot_byte(&self) -> u8 {
        match self {
            VectorDtype::F32 => 0,
            VectorDtype::F16 => 1,
        }
    }

    /// 从快照字节恢复；未知值返回 None，由调用方按损坏/版本不兼容处理。
    /// EN: Restores from a snapshot byte; unknown values return None so callers
    /// can treat them as corruption/unsupported version.
    pub const fn from_snapshot_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(VectorDtype::F32),
            1 => Some(VectorDtype::F16),
            _ => None,
        }
    }
}

/// 已存储的向量值：dims 在两种变体中都由 `len()` 导出，且创建后不可变。
/// EN: A stored vector value: dimensions derive from `len()` in both variants and
/// contents are immutable after construction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StoredVector {
    /// 原始 f32 分量（默认）。
    /// EN: Raw f32 components (default).
    F32(Vec<f32>),
    /// f16 分量；由 f32 在写入时一次性量化。
    /// EN: f16 components; quantized once from f32 at write time.
    F16(Vec<f16>),
}

impl StoredVector {
    /// 由 f32 输入按目标 dtype 构造；f32 目标零拷贝，f16 目标逐元素量化。
    /// EN: Builds from f32 input for the target dtype; f32 targets take the
    /// vector as-is and f16 targets quantize element-wise.
    pub fn from_f32(vector: Vec<f32>, dtype: VectorDtype) -> Self {
        match dtype {
            VectorDtype::F32 => StoredVector::F32(vector),
            // `f16::from_f32` 是 round-to-nearest；但 >|65504| 的有限值会溢出成
            // ±inf，进入 L2/IP 后距离变 inf、余弦的 inf*inf 归一化还可能产出 NaN。
            // 因此在量化前饱和到 f16 最大有限值：保持排序语义与有限距离，代价是
            // 越界分量的绝对数值被截断（NaN 仍按 IEEE 传播，与本分支无关）。
            // EN: `f16::from_f32` rounds to nearest, but finite values beyond
            // |65504| overflow to ±inf, making L2/IP distances infinite and cosine
            // possibly NaN. Saturate to f16's largest finite value before
            // quantization: ordering and finite distances are preserved at the cost
            // of clipping the magnitude of out-of-range components (NaN still
            // propagates per IEEE, independent of this branch).
            VectorDtype::F16 => StoredVector::F16(
                vector
                    .into_iter()
                    .map(|x| f16::from_f32(x.clamp(-F16_MAX_FINITE, F16_MAX_FINITE)))
                    .collect(),
            ),
        }
    }

    /// 维度数。
    /// EN: Number of dimensions.
    pub fn dims(&self) -> usize {
        match self {
            StoredVector::F32(v) => v.len(),
            StoredVector::F16(v) => v.len(),
        }
    }

    /// 存储 dtype。
    /// EN: Storage dtype.
    pub fn dtype(&self) -> VectorDtype {
        match self {
            StoredVector::F32(_) => VectorDtype::F32,
            StoredVector::F16(_) => VectorDtype::F16,
        }
    }

    /// 借用为与 dtype 无关的只读视图（距离核的入口）。
    /// EN: Borrows as a dtype-agnostic read-only view (the distance-kernel entry).
    /// 取借用视图。刻意不叫 `as_ref`：`Arc<StoredVector>` 自带 `AsRef<StoredVector>`，
    /// 若同名会解析成 `&StoredVector` 而非本方法的 `VectorRef`（调用点能编译，
    /// 但下游距离核期望 `VectorRef` 时才报错），独立命名把误用变成显式编译错误。
    /// EN: Borrowed view. Deliberately not named `as_ref`: `Arc<StoredVector>` already
    /// provides `AsRef<StoredVector>`, so a same-named method would resolve to
    /// `&StoredVector` instead of this `VectorRef` view — the call site compiles and
    /// the error only surfaces where a `VectorRef` is expected. A distinct name makes
    /// the mix-up an explicit compile error.
    pub fn as_view(&self) -> VectorRef<'_> {
        match self {
            StoredVector::F32(v) => VectorRef::F32(v.as_slice()),
            StoredVector::F16(v) => VectorRef::F16(v.as_slice()),
        }
    }

    /// 展开为 f32（导出/重排/测试用）；f16 变体逐元素提升。
    /// EN: Expands to f32 (export/rerank/tests); the f16 variant lifts element-wise.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            StoredVector::F32(v) => v.clone(),
            StoredVector::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        }
    }

    /// 以 f32 视图导出：f32 变体借用底层切片（零拷贝），f16 变体逐元素提升。
    /// 用于逐记录序列化（如 HTTP 导出）等不需要持有 `Vec<f32>` 的边界。
    /// EN: Exports as an f32 view: the f32 variant borrows the underlying slice
    /// (zero-copy) while the f16 variant lifts element-wise. Intended for
    /// per-record serialization boundaries (e.g. HTTP export) that need no
    /// owned `Vec<f32>`.
    pub fn to_f32_cow(&self) -> Cow<'_, [f32]> {
        match self {
            StoredVector::F32(v) => Cow::Borrowed(v.as_slice()),
            StoredVector::F16(v) => Cow::Owned(v.iter().map(|x| x.to_f32()).collect()),
        }
    }
}

/// 与 dtype 无关的向量借用视图。不拥有数据，生命周期跟随底层 `StoredVector`。
/// EN: Dtype-agnostic borrowed view of a vector. It owns nothing and lives as
/// long as the underlying `StoredVector`.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum VectorRef<'a> {
    /// f32 分量视图。
    /// EN: View over f32 components.
    F32(&'a [f32]),
    /// f16 分量视图。
    /// EN: View over f16 components.
    F16(&'a [f16]),
}

impl<'a> VectorRef<'a> {
    /// 维度数。
    /// EN: Number of dimensions.
    pub fn dims(&self) -> usize {
        match self {
            VectorRef::F32(v) => v.len(),
            VectorRef::F16(v) => v.len(),
        }
    }

    /// 视图的 dtype。
    /// EN: Dtype of the view.
    pub fn dtype(&self) -> VectorDtype {
        match self {
            VectorRef::F32(_) => VectorDtype::F32,
            VectorRef::F16(_) => VectorDtype::F16,
        }
    }

    /// 展开为 f32（仅用于非批量路径，如导出与调试）。
    /// EN: Expands to f32 (only for non-bulk paths such as export and debugging).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            VectorRef::F32(v) => v.to_vec(),
            VectorRef::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        }
    }
}

/// 距离核的公共元素契约：允许 f32/f16 及混合组合共用同一套累加循环。
/// EN: Shared element contract for distance kernels so f32/f16 and mixed
/// combinations reuse the same accumulation loops.
trait Elem: Copy {
    /// 提升为 f32。
    /// EN: Lifts to f32.
    fn lift(self) -> f32;
}

impl Elem for f32 {
    #[inline(always)]
    fn lift(self) -> f32 {
        self
    }
}

impl Elem for f16 {
    #[inline(always)]
    fn lift(self) -> f32 {
        self.to_f32()
    }
}

/// 通用平方 L2 累加（逐元素提升到 f32，避免 f16 中间累加溢出/欠精度）。
/// EN: Generic squared-L2 accumulation (each element lifted to f32 to avoid
/// f16 intermediate overflow/precision loss).
#[inline(always)]
fn l2_generic<A: Elem, B: Elem>(a: &[A], b: &[B], n: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut i = 0usize;
    while i < n {
        let d = a[i].lift() - b[i].lift();
        sum += d * d;
        i += 1;
    }
    sum
}

/// 通用点积累加。
/// EN: Generic dot-product accumulation.
#[inline(always)]
fn dot_generic<A: Elem, B: Elem>(a: &[A], b: &[B], n: usize) -> f32 {
    let mut dot = 0.0_f32;
    let mut i = 0usize;
    while i < n {
        dot += a[i].lift() * b[i].lift();
        i += 1;
    }
    dot
}

/// 通用 L1 累加。
/// EN: Generic L1 accumulation.
#[inline(always)]
fn l1_generic<A: Elem, B: Elem>(a: &[A], b: &[B], n: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut i = 0usize;
    while i < n {
        sum += (a[i].lift() - b[i].lift()).abs();
        i += 1;
    }
    sum
}

/// 非 f32×f32 组合的统一分派：四则度量语义与 `distance.rs` 完全一致。
/// EN: Unified dispatch for non-f32×f32 pairs; the four metric semantics match
/// `distance.rs` exactly.
#[inline]
fn mixed_metric<A: Elem, B: Elem>(a: &[A], b: &[B], n: usize, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_generic(a, b, n),
        DistanceMetric::Cosine => {
            let dot = dot_generic(a, b, n);
            let na = dot_generic(a, a, n);
            let nb = dot_generic(b, b, n);
            // 零范数与 `distance::cosine` 保持同一约定：返回最大距离 1.0。
            // EN: Zero norm follows `distance::cosine`: return the maximum distance 1.0.
            if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                1.0 - (dot / (na.sqrt() * nb.sqrt()))
            }
        }
        DistanceMetric::InnerProduct => -dot_generic(a, b, n),
        DistanceMetric::L1 => l1_generic(a, b, n),
    }
}

/// 任意 dtype 组合的距离计算（余弦为完整余弦，适用于存储原始向量的扁平/重排路径）。
/// 长度不等时取较短的一侧，与 `distance.rs` 的既有约定一致。
/// EN: Distance for any dtype combination (Cosine is the full cosine, suitable for
/// flat/rerank paths over raw stored vectors). Unequal lengths use the shorter
/// side, matching `distance.rs`.
#[inline]
pub fn distance_between(a: VectorRef<'_>, b: VectorRef<'_>, metric: DistanceMetric) -> f32 {
    let n = a.dims().min(b.dims());
    match (a, b) {
        (VectorRef::F32(a), VectorRef::F32(b)) => match metric {
            DistanceMetric::L2 => distance::l2_squared(a, b),
            DistanceMetric::Cosine => distance::cosine(a, b),
            DistanceMetric::InnerProduct => distance::inner_product_distance(a, b),
            DistanceMetric::L1 => distance::l1_distance(a, b),
        },
        (VectorRef::F32(a), VectorRef::F16(b)) => mixed_metric(a, b, n, metric),
        (VectorRef::F16(a), VectorRef::F32(b)) => mixed_metric(a, b, n, metric),
        (VectorRef::F16(a), VectorRef::F16(b)) => mixed_metric(a, b, n, metric),
    }
}

/// 已预归一化输入的余弦距离 `1 - dot`（图内部近邻选择专用）。
/// 任意 dtype 组合都在 f32 上累加；点积先 clamp 到 [-1, 1] 抑制舍入越界。
/// EN: Cosine over pre-normalized inputs `1 - dot` (internal graph neighbor
/// selection). Every dtype combination accumulates in f32, and the dot product is
/// clamped to [-1, 1] to suppress rounding overshoot.
#[inline]
pub fn cosine_prenormalized(a: VectorRef<'_>, b: VectorRef<'_>) -> f32 {
    match (a, b) {
        (VectorRef::F32(a), VectorRef::F32(b)) => distance::cosine_distance_prenormalized(a, b),
        (VectorRef::F32(a), VectorRef::F16(b)) => {
            let n = a.len().min(b.len());
            1.0 - dot_generic(a, b, n).clamp(-1.0, 1.0)
        }
        (VectorRef::F16(a), VectorRef::F32(b)) => {
            let n = a.len().min(b.len());
            1.0 - dot_generic(a, b, n).clamp(-1.0, 1.0)
        }
        (VectorRef::F16(a), VectorRef::F16(b)) => {
            let n = a.len().min(b.len());
            1.0 - dot_generic(a, b, n).clamp(-1.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(dims: usize) -> Vec<f32> {
        (0..dims).map(|i| ((i as f32) - 3.0) / 7.0).collect()
    }

    #[test]
    fn dtype_names_and_snapshot_bytes_roundtrip() {
        assert_eq!(VectorDtype::F32.as_str(), "f32");
        assert_eq!(VectorDtype::F16.as_str(), "f16");
        assert_eq!(VectorDtype::F32.bytes_per_dim(), 4);
        assert_eq!(VectorDtype::F16.bytes_per_dim(), 2);
        assert_eq!(VectorDtype::default(), VectorDtype::F32);
        for dtype in [VectorDtype::F32, VectorDtype::F16] {
            assert_eq!(
                VectorDtype::from_snapshot_byte(dtype.to_snapshot_byte()),
                Some(dtype)
            );
            assert_eq!(VectorDtype::parse_str(dtype.as_str()), Some(dtype));
        }
        assert_eq!(VectorDtype::parse_str("FLOAT16"), Some(VectorDtype::F16));
        assert_eq!(VectorDtype::parse_str(" Half "), Some(VectorDtype::F16));
        assert_eq!(VectorDtype::parse_str("float32"), Some(VectorDtype::F32));
        assert_eq!(VectorDtype::parse_str("int8"), None);
        assert_eq!(VectorDtype::from_snapshot_byte(2), None);
    }

    #[test]
    fn stored_vector_quantizes_and_lifts() {
        let raw = sample(13);
        let f32_vec = StoredVector::from_f32(raw.clone(), VectorDtype::F32);
        assert_eq!(f32_vec.dims(), 13);
        assert_eq!(f32_vec.dtype(), VectorDtype::F32);
        assert_eq!(f32_vec.to_f32_vec(), raw);

        let f16_vec = StoredVector::from_f32(raw.clone(), VectorDtype::F16);
        assert_eq!(f16_vec.dims(), 13);
        assert_eq!(f16_vec.dtype(), VectorDtype::F16);
        let lifted = f16_vec.to_f32_vec();
        for (a, b) in raw.iter().zip(lifted.iter()) {
            // f16 相对误差约 2^-11；分量量级 ≤ 1 时绝对误差应远小于 1e-3。
            // EN: f16 relative error is ~2^-11; with components ≤ 1 the absolute
            // error must be far below 1e-3.
            assert!((a - b).abs() < 1e-3, "a={a} b={b}");
        }
        assert_eq!(f16_vec.as_view().dtype(), VectorDtype::F16);
        assert_eq!(f16_vec.as_view().dims(), 13);
    }

    #[test]
    fn f16_quantization_saturates_out_of_range_components() {
        // 回归：f16::from_f32 对超界有限值与 ±inf 会产出 ±inf，L2/IP 变成 inf、
        // cosine 变成 NaN（污染排序）。必须饱和到 ±65504 并保持有限。
        // EN: Regression: f16::from_f32 maps out-of-range finite values and ±inf
        // to ±inf, turning L2/IP into inf and cosine into NaN (poisoning ranking).
        // Quantization must saturate at ±65504 and stay finite.
        let raw = vec![1e9_f32, -1e9, f32::INFINITY, f32::NEG_INFINITY, 1.0];
        let stored = StoredVector::from_f32(raw, VectorDtype::F16);
        assert_eq!(
            stored.to_f32_vec(),
            vec![65504.0, -65504.0, 65504.0, -65504.0, 1.0]
        );
        assert!(
            stored.to_f32_vec().iter().all(|x| x.is_finite()),
            "quantized components must all be finite"
        );
        // inf 不能参与距离计算的传播：两行饱和向量之间的协方差必须有限。
        // EN: inf must not propagate through distance math: the distance between
        // two saturated rows must stay finite.
        let other = StoredVector::from_f32(vec![-1e9_f32; 5], VectorDtype::F16);
        let d = distance_between(stored.as_view(), other.as_view(), DistanceMetric::L2);
        assert!(
            d.is_finite(),
            "L2 between saturated rows must be finite: {d}"
        );
    }

    #[test]
    fn f16_and_mixed_distances_match_f32_reference() {
        let raw_a = sample(16);
        let raw_b: Vec<f32> = (0..16).map(|i| ((i as f32) - 5.0) / 9.0).collect();
        let f16_a = StoredVector::from_f32(raw_a.clone(), VectorDtype::F16);
        let f16_b = StoredVector::from_f32(raw_b.clone(), VectorDtype::F16);

        for metric in [
            DistanceMetric::L2,
            DistanceMetric::Cosine,
            DistanceMetric::InnerProduct,
            DistanceMetric::L1,
        ] {
            let baseline = distance_between(VectorRef::F32(&raw_a), VectorRef::F32(&raw_b), metric);
            let f16_pair = distance_between(f16_a.as_view(), f16_b.as_view(), metric);
            let mixed = distance_between(VectorRef::F32(&raw_a), f16_b.as_view(), metric);
            let scale = baseline.abs().max(1.0);
            // 量化噪声界：16 维、分量 ≤ 1 时 2% 相对容差足够宽松且能捕获语义错误。
            // EN: Quantization-noise bound: 2% relative tolerance is loose enough
            // for 16 dims with components ≤ 1 while still catching semantic bugs.
            assert!(
                (f16_pair - baseline).abs() <= 0.02 * scale,
                "metric={metric:?} f16={f16_pair} base={baseline}"
            );
            assert!(
                (mixed - baseline).abs() <= 0.02 * scale,
                "metric={metric:?} mixed={mixed} base={baseline}"
            );
        }
    }

    #[test]
    fn cosine_prenormalized_matches_full_cosine_for_unit_vectors() {
        let mut a = sample(8);
        let mut b: Vec<f32> = (0..8).map(|i| (i as f32) / 5.0).collect();
        distance::normalize_in_place(&mut a);
        distance::normalize_in_place(&mut b);

        let full = distance_between(
            VectorRef::F32(&a),
            VectorRef::F32(&b),
            DistanceMetric::Cosine,
        );
        let pre = cosine_prenormalized(VectorRef::F32(&a), VectorRef::F32(&b));
        assert!((full - pre).abs() < 1e-5);

        let f16_a = StoredVector::from_f32(a, VectorDtype::F16);
        let f16_b = StoredVector::from_f32(b, VectorDtype::F16);
        let pre_f16 = cosine_prenormalized(f16_a.as_view(), f16_b.as_view());
        assert!((pre_f16 - pre).abs() < 2e-3, "pre_f16={pre_f16} pre={pre}");
    }

    #[test]
    fn zero_norm_cosine_returns_one_for_mixed_dtypes() {
        let zeros = vec![0.0_f32; 6];
        let ones = vec![1.0_f32; 6];
        let f16_zeros = StoredVector::from_f32(zeros.clone(), VectorDtype::F16);
        let f16_ones = StoredVector::from_f32(ones.clone(), VectorDtype::F16);
        assert_eq!(
            distance_between(
                f16_zeros.as_view(),
                VectorRef::F32(&ones),
                DistanceMetric::Cosine
            ),
            1.0
        );
        assert_eq!(
            distance_between(
                f16_zeros.as_view(),
                f16_ones.as_view(),
                DistanceMetric::Cosine
            ),
            1.0
        );
    }

    #[test]
    fn unequal_lengths_use_shorter_side() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let f16_a = StoredVector::from_f32(a.clone(), VectorDtype::F16);
        let expected = l2_generic(&a[..3], &b, 3);
        let got = distance_between(f16_a.as_view(), VectorRef::F32(&b), DistanceMetric::L2);
        assert!((got - expected).abs() < 1e-4);
    }
}
