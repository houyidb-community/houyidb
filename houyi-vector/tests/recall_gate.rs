//! Recall 门禁测试：与 `benches/vector_baseline.rs` 的 `recall_check_hnsw_after_flush` 同配置，
//! 但把 recall 从「算了不判」变成真正的阈值断言。数据集/对照/阈值共享自 `tests/common/recall.rs`。
//!
//! Recall gate: same configs as the `recall_check_hnsw_after_flush` bench, but asserts a
//! minimum recall@k so regressions fail CI instead of being silently black_boxed away.
//!
//! 环境前提 / EN: This gate is only meaningful under default `HOUYI_*` env.
//! `HotBufferIndex::new` reads `VectorConfig/HnswConfig/BuildConfig::from_env()` — e.g.
//! `HOUYI_VECTOR_FORCE_FLAT=1` turns search into an exact brute-force scan (recall ≡ 1.0)
//! and the gate passes without exercising HNSW at all; `HOUYI_HNSW_*` knobs silently
//! switch the build path. CI must run with `HOUYI_*` unset; `BuildConfig` is pinned
//! explicitly below (the rest are not pinnable post-construction).

use half::f16;
use houyi_core::config::{BuildConfig, HnswConfig, VectorConfig};
use houyi_vector::{DistanceMetric, HotBufferIndex, Quantization, VectorDtype, VectorIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[path = "common/recall.rs"]
mod recall;
use recall::{brute_force_topk, make_vector, recall_at_k, RECALL_MIN};

/// 跑一组 (dims, total, k, ef)，返回实测 recall。 / Run one config, return measured recall.
/// EN: Run one config and return the measured recall.
async fn measure_recall(dims: usize, total: usize, k: usize, ef: usize) -> f32 {
    let dataset: Vec<(u64, Vec<f32>)> = (0..total as u64)
        .map(|id| (id, make_vector(id, dims)))
        .collect();
    let query = make_vector((total / 3) as u64, dims);
    let truth = brute_force_topk(&dataset, &query, k);

    let mut index = HotBufferIndex::new(dims);
    // 钉住可钉的构建配置，使被测构建路径不受环境变量影响（串行/512，与实测基线一致）。
    // EN: Pin the build path so ambient HOUYI_HNSW_* env cannot silently switch it.
    index.set_build_config(BuildConfig::default());
    index
        .set_hnsw_ef_search(ef)
        .expect("set ef_search should succeed");
    index
        .insert_batch(dataset)
        .await
        .expect("batch insert should succeed");
    while index.pending_build_count().await > 0 {
        let _ = index.run_async_builder_once().await;
    }
    let pred = index.search(&query, k, None).await;
    recall_at_k(&pred, &truth, k)
}

#[tokio::test]
async fn recall_gate_after_flush_meets_threshold() {
    for &(dims, total, k) in &[(128usize, 1000usize, 10usize), (512, 2000, 10)] {
        for ef in [8usize, 16, 32] {
            let recall = measure_recall(dims, total, k, ef).await;
            assert!(
                recall >= RECALL_MIN,
                "recall regression: dims={dims} total={total} k={k} ef={ef} \
                 recall={recall:.4} < {RECALL_MIN}"
            );
        }
    }
}

/// P1.7 f16 召回门：f16 存储 + f32 查询（非对称距离）在相同 HNSW 配置下的
/// recall@10 必须达到同一门槛。对照真值按**实际语义**计算——数据集先按 f16
/// 舍入、再与 f32 查询算平方 L2——否则「真值取 f32 原始值」会把纯量化误差
/// 误判为检索回归（两者本就不是同一个向量集合）。
/// EN: P1.7 f16 recall gate: f16 storage + f32 query (asymmetric distance) must
/// meet the same recall@10 bound under identical HNSW configs. The ground truth
/// models the actual semantics — the dataset is f16-rounded first, then squared
/// L2 against the f32 query — otherwise f32-original truth would misreport pure
/// quantization error as a retrieval regression (they are different vector sets).
async fn measure_recall_f16(dims: usize, total: usize, k: usize, ef: usize) -> f32 {
    // 数据集本身可被 f16 精确表示（make_vector 的 997 分母不在二进制小数网格上，
    // 但插入内核时同样会舍入，因此以「舍入后的 f32 值」作为对照数据）。
    // EN: the dataset is f16-rounded here for the reference; the kernel rounds
    // the same way on insert, so both sides see identical stored values.
    let rounded: Vec<(u64, Vec<f32>)> = (0..total as u64)
        .map(|id| {
            (
                id,
                make_vector(id, dims)
                    .into_iter()
                    .map(|x| f16::from_f32(x).to_f32())
                    .collect(),
            )
        })
        .collect();
    let query = make_vector((total / 3) as u64, dims);
    let truth = brute_force_topk(&rounded, &query, k);

    // 显式配置 + dtype：环境变量不能切换构图路径，f32 缺省也不能悄悄顶替 f16。
    // EN: explicit config + dtype so env cannot switch the build path and the
    // f32 default cannot silently replace f16.
    let mut index = HotBufferIndex::new_with_metric_and_config_and_dtype(
        dims,
        DistanceMetric::L2,
        VectorDtype::F16,
        &VectorConfig::default(),
        &HnswConfig::default(),
    );
    assert_eq!(
        index.vector_dtype(),
        VectorDtype::F16,
        "f16 gate must actually exercise f16 storage"
    );
    index.set_build_config(BuildConfig::default());
    index
        .set_hnsw_ef_search(ef)
        .expect("set ef_search should succeed");
    index
        .insert_batch(rounded)
        .await
        .expect("batch insert should succeed");
    while index.pending_build_count().await > 0 {
        let _ = index.run_async_builder_once().await;
    }
    let pred = index.search(&query, k, None).await;
    recall_at_k(&pred, &truth, k)
}

#[tokio::test]
async fn recall_gate_f16_storage_meets_threshold() {
    for &(dims, total, k) in &[(128usize, 1000usize, 10usize), (512, 2000, 10)] {
        for ef in [8usize, 16, 32] {
            let recall = measure_recall_f16(dims, total, k, ef).await;
            assert!(
                recall >= RECALL_MIN,
                "f16 recall regression: dims={dims} total={total} k={k} ef={ef} \
                 recall={recall:.4} < {RECALL_MIN}"
            );
        }
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn brute_force_topk_metric(
    data: &[(u64, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = data
        .iter()
        .map(|(id, v)| {
            let d = match metric {
                DistanceMetric::L2 => recall::l2_distance(query, v),
                DistanceMetric::Cosine => cosine_distance(query, v),
                _ => unreachable!("int8 recall gate covers L2/Cosine only"),
            };
            (*id, d)
        })
        .collect();
    scored.sort_by(|(id_a, d_a), (id_b, d_b)| d_a.total_cmp(d_b).then(id_a.cmp(id_b)));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// 门禁数据集分布口径。
/// EN: Dataset-distribution convention for the recall gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateDatasetKind {
    /// 锯齿周期数据集（`make_vector`）：int8 门沿用，与既有基线同口径。
    /// EN: Sawtooth periodic dataset (`make_vector`), kept for the int8 gates to
    /// stay on the existing baseline.
    Sawtooth,
    /// 零均值 i.i.d. 均匀数据集：RaBitQ 门专用。`make_vector` 是周期锯齿，
    /// dims>997 时按 997 回绕，各向量只是同一波形的平移，强公共方向会吞掉
    /// 1 bit 符号码的排序分辨率（Hadamard 旋转把全 1 方向集中到单个坐标，
    /// 均值分量抢走绝大部分 L1 质量）。改用固定种子的零均值均匀数据，与
    /// RaBitQ 的推荐前置处理（零均值化）一致；生产侧同样建议先去均值。
    /// EN: Zero-mean i.i.d. uniform dataset, used by the RaBitQ gates only.
    /// `make_vector` is a periodic sawtooth that wraps with period 997 past dims
    /// 997, so vectors are shifts of one waveform; the strong common direction
    /// swamps the 1-bit sign codes (the Hadamard rotation concentrates the
    /// all-ones direction onto a single coordinate, and the mean component
    /// dominates the L1 mass). A fixed-seed zero-mean uniform dataset matches
    /// RaBitQ's recommended preprocessing (mean-centering); production use
    /// should center data as well.
    ZeroMeanUniform,
}

/// 数据集与查询的固定种子（只影响门禁，不进入生产路径）。
/// EN: Fixed seeds for the dataset and query (gate-only, never on the
/// production path).
const GATE_DATA_SEED: u64 = 0x5EED_2024_9A17_C0DE;
const GATE_QUERY_SEED: u64 = 0x5EED_2024_9A17_1D0C;

fn uniform_gate_vector(rng: &mut StdRng, dims: usize) -> Vec<f32> {
    (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect()
}

/// 量化门禁数据集：余弦对照需零均值向量（全正向量彼此近平行，top-k 对比度
/// 不足）；锯齿口径仅在余弦时做平移，`ZeroMeanUniform` 本身零均值。
/// f16 主表把值预舍入，使真值与存储值同源，避免把纯量化误差判成回归。
/// EN: Quantization-gate dataset: cosine truth needs zero-mean vectors
/// (all-positive vectors are nearly parallel and leave the top-k almost no
/// contrast); the sawtooth convention shifts only for cosine, while
/// `ZeroMeanUniform` is zero-mean by construction. The f16 main store pre-rounds
/// values so ground truth and stored values are the same vector set, avoiding
/// quantization error being read as a retrieval regression.
fn gate_dataset(
    kind: GateDatasetKind,
    dims: usize,
    total: usize,
    metric: DistanceMetric,
    dtype: VectorDtype,
) -> Vec<(u64, Vec<f32>)> {
    let mut rng = StdRng::seed_from_u64(GATE_DATA_SEED);
    (0..total as u64)
        .map(|id| {
            let mut v = match kind {
                GateDatasetKind::Sawtooth => make_vector(id, dims),
                GateDatasetKind::ZeroMeanUniform => uniform_gate_vector(&mut rng, dims),
            };
            if kind == GateDatasetKind::Sawtooth && metric == DistanceMetric::Cosine {
                for x in &mut v {
                    *x -= 0.5;
                }
            }
            if dtype == VectorDtype::F16 {
                v = v.into_iter().map(|x| f16::from_f32(x).to_f32()).collect();
            }
            (id, v)
        })
        .collect()
}

/// 与 [`gate_dataset`] 同口径的查询向量。
/// EN: Query vector under the same convention as [`gate_dataset`].
fn gate_query(
    kind: GateDatasetKind,
    dims: usize,
    total: usize,
    metric: DistanceMetric,
) -> Vec<f32> {
    match kind {
        GateDatasetKind::Sawtooth => {
            let q = make_vector((total / 3) as u64, dims);
            if metric == DistanceMetric::Cosine {
                q.into_iter().map(|x| x - 0.5).collect()
            } else {
                q
            }
        }
        GateDatasetKind::ZeroMeanUniform => {
            let mut rng = StdRng::seed_from_u64(GATE_QUERY_SEED);
            uniform_gate_vector(&mut rng, dims)
        }
    }
}

/// 量化图（int8 / RaBitQ）召回门的共用驱动：主表 dtype + 图侧量化显式构造，
/// 钉住构建配置，批量入图后按生产路径检索（扩宽候选 + 主表精确重排）。
/// 参数多是有意为之——每个维度都要在失败信息里可辨认；测试代码不加分组结构。
/// EN: Shared driver for quantized-graph (int8 / RaBitQ) recall gates: explicit
/// main-store dtype + graph-side quantization, pinned build config, batch
/// ingestion, then production-path search (widened candidates + exact
/// main-store rerank). The wide parameter list is deliberate — every dimension
/// must stay identifiable in failure messages — so no grouping struct here.
#[allow(clippy::too_many_arguments)]
async fn measure_recall_quantized(
    dims: usize,
    total: usize,
    k: usize,
    ef: usize,
    metric: DistanceMetric,
    dtype: VectorDtype,
    quantization: Quantization,
    kind: GateDatasetKind,
) -> f32 {
    let dataset = gate_dataset(kind, dims, total, metric, dtype);
    let query = gate_query(kind, dims, total, metric);
    let truth = brute_force_topk_metric(&dataset, &query, k, metric);

    let mut index = HotBufferIndex::new_with_metric_and_config_and_dtype_and_quantization(
        dims,
        metric,
        dtype,
        quantization,
        &VectorConfig::default(),
        &HnswConfig::default(),
    );
    assert_eq!(
        index.vector_quantization(),
        quantization,
        "gate must actually exercise the intended graph-side quantization"
    );
    assert_eq!(
        index.vector_dtype(),
        dtype,
        "main-store dtype must match the gate"
    );
    index.set_build_config(BuildConfig::default());
    index
        .set_hnsw_ef_search(ef)
        .expect("set ef_search should succeed");
    index
        .insert_batch(dataset)
        .await
        .expect("batch insert should succeed");
    while index.pending_build_count().await > 0 {
        let _ = index.run_async_builder_once().await;
    }
    let pred = index.search(&query, k, None).await;
    recall_at_k(&pred, &truth, k)
}

/// P2 int8 图召回门：图侧 int8 只改变候选集合，最终 top-k 由主表原始向量精确
/// 重排（`bounded_candidate_ids` 在 int8 时把束宽乘 2 补偿近似距离）。真值按
/// 主表语义计算——L2 用原始 f32（或 f16 舍入后的值），余弦在零均值数据集上
/// 用精确余弦距离——否则会把「候选近似」误判成排序回归。
/// EN: P2 int8-graph recall gate: graph-side int8 only changes the candidate set;
/// the final top-k is reranked exactly on raw main-store vectors
/// (`bounded_candidate_ids` doubles the beam width under int8 to offset
/// approximate distances). Ground truth follows main-store semantics — L2 over
/// raw f32 (or f16-rounded) values, cosine over a zero-mean dataset with exact
/// cosine distance — otherwise approximate candidates would be misreported as a
/// ranking regression.
async fn measure_recall_int8(
    dims: usize,
    total: usize,
    k: usize,
    ef: usize,
    metric: DistanceMetric,
    dtype: VectorDtype,
) -> f32 {
    measure_recall_quantized(
        dims,
        total,
        k,
        ef,
        metric,
        dtype,
        Quantization::Int8,
        GateDatasetKind::Sawtooth,
    )
    .await
}

#[tokio::test]
#[ignore = "量化为多种子/多配置建库，debug 下约 18 分钟；CI 在 release 模式用 --ignored 运行 / EN: quantized gates build many indexes (~18 min in debug); CI runs them in release with --ignored"]
async fn recall_gate_int8_graph_meets_threshold() {
    for &(dims, total, k) in &[(128usize, 1000usize, 10usize), (512, 2000, 10)] {
        for ef in [8usize, 16, 32] {
            let recall =
                measure_recall_int8(dims, total, k, ef, DistanceMetric::L2, VectorDtype::F32).await;
            assert!(
                recall >= RECALL_MIN,
                "int8 recall regression: dims={dims} total={total} k={k} ef={ef} \
                 recall={recall:.4} < {RECALL_MIN}"
            );
        }
    }
    for ef in [8usize, 16, 32] {
        let recall =
            measure_recall_int8(128, 1000, 10, ef, DistanceMetric::Cosine, VectorDtype::F32).await;
        assert!(
            recall >= RECALL_MIN,
            "int8 cosine recall regression: ef={ef} recall={recall:.4} < {RECALL_MIN}"
        );
    }
}

#[tokio::test]
#[ignore = "量化为多种子/多配置建库，debug 下约 18 分钟；CI 在 release 模式用 --ignored 运行 / EN: quantized gates build many indexes (~18 min in debug); CI runs them in release with --ignored"]
async fn recall_gate_int8_graph_f16_main_meets_threshold() {
    for &(dims, total, k) in &[(128usize, 1000usize, 10usize), (512, 2000, 10)] {
        for ef in [8usize, 16, 32] {
            let recall =
                measure_recall_int8(dims, total, k, ef, DistanceMetric::L2, VectorDtype::F16).await;
            assert!(
                recall >= RECALL_MIN,
                "int8+f16 recall regression: dims={dims} total={total} k={k} ef={ef} \
                 recall={recall:.4} < {RECALL_MIN}"
            );
        }
    }
}

/// P3 RaBitQ 图召回门：图侧 1 bit/dim 只改变候选集合，最终 top-k 由主表原始
/// 向量精确重排（`bounded_candidate_ids` 在 RaBitQ 时把束宽乘 16 补偿近似距离）。
/// 数据集统一零均值（含 L2；理由见 [`gate_dataset`]）；真值口径与 int8 门一致。
/// 覆盖 l2/cosine × f32/f16 主表 × dims 128/768/1024；dims=128 另跑 8000
/// 向量稀释档（束宽只覆盖语料 12.8%/19.2%，让门禁真正检验 1 bit 排序质量）。
/// EN: P3 RaBitQ-graph recall gate: graph-side 1 bit/dim only changes the
/// candidate set; the final top-k is reranked exactly on raw main-store vectors
/// (`bounded_candidate_ids` multiplies the beam width by 16 under RaBitQ to
/// offset approximate distances). The dataset is always zero-mean, L2 included
/// (rationale in [`gate_dataset`]); ground-truth semantics match the int8 gate.
/// Covers l2/cosine × f32/f16 main stores × dims 128/768/1024; dims=128 also
/// runs an 8000-vector dilution tier (the beam covers only 12.8%/19.2% of the
/// corpus, so the gate truly exercises 1-bit ranking quality).
async fn measure_recall_rabitq(
    dims: usize,
    total: usize,
    k: usize,
    ef: usize,
    metric: DistanceMetric,
    dtype: VectorDtype,
) -> f32 {
    measure_recall_quantized(
        dims,
        total,
        k,
        ef,
        metric,
        dtype,
        Quantization::RaBitQ,
        GateDatasetKind::ZeroMeanUniform,
    )
    .await
}

/// RaBitQ 召回门：随机高维数据上的对照（未量化）在 ef=64 才达到满分 recall
/// （ef=32 时 dims=128 为 0.9、dims=768 余弦为 0.7），因此本门用 ef=64/96，
/// 保证「量化门」不会把未量化路径本就达不到的 recall 记到 RaBitQ 头上。
/// EN: RaBitQ recall gate. On random high-dim data the unquantized control needs
/// ef=64 for perfect recall (ef=32 gives 0.9 at dims=128 and 0.7 for cosine at
/// dims=768), so this gate runs at ef=64/96 — a quantization gate must not blame
/// RaBitQ for recall the unquantized path cannot reach either.
#[tokio::test]
#[ignore = "量化为多种子/多配置建库，debug 下约 18 分钟；CI 在 release 模式用 --ignored 运行 / EN: quantized gates build many indexes (~18 min in debug); CI runs them in release with --ignored"]
async fn recall_gate_rabitq_graph_meets_threshold() {
    let mut failures = Vec::new();
    // dims=128 另跑一个 8000 向量档：×16 束宽在 ef=64/96 下只覆盖 12.8%/19.2%
    // 语料，让门禁真正检验 1 bit 排序质量——1000 向量档的候选集占语料 51%+
    // （×8 时 51%~77%，×16 时封顶为全图），几乎等价于暴力检索，秩退化时也不会
    // 掉召回。
    // EN: dims=128 also runs an 8000-vector tier: the ×16 beam covers only
    // 12.8%/19.2% of the corpus at ef=64/96, so the gate actually exercises the
    // 1-bit ranker — the 1000-vector tier's candidate set is 51%+ of the corpus
    // (51%-77% at ×8, capped at the whole graph at ×16), effectively a
    // brute-force check that cannot detect rank degradation.
    for &(dims, total, k) in &[
        (128usize, 1000usize, 10usize),
        (128, 8000, 10),
        (768, 1000, 10),
        (1024, 1000, 10),
    ] {
        for dtype in [VectorDtype::F32, VectorDtype::F16] {
            for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
                for ef in [64usize, 96] {
                    let recall = measure_recall_rabitq(dims, total, k, ef, metric, dtype).await;
                    eprintln!(
                        "rabitq dims={dims} dtype={dtype:?} metric={metric:?} ef={ef} \
                         recall={recall:.4}"
                    );
                    if recall < RECALL_MIN {
                        failures.push(format!(
                            "dims={dims} dtype={dtype:?} metric={metric:?} ef={ef} \
                             recall={recall:.4} < {RECALL_MIN}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "rabitq recall regressions:\n{}",
        failures.join("\n")
    );
}
