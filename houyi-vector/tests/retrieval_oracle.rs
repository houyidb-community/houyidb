//! 检索 oracle（P4.5.3）：**metamorphic 关系 + 暴力对照**，无外部真值也能判定对错。
//! Retrieval oracle: metamorphic relations plus brute-force differential testing.
//!
//! 两条腿 / Two legs:
//!
//! 1. **严格断言（暴力后端）**——把 `force_flat_search` 钉死为 true，检索退化为精确
//!    暴力扫描，于是「实现结果 == 参考实现 top-k」是逐位可判的等式：k 前缀性、过滤
//!    掩码等价、重复插入幂等、删除不可见、重插复活、距离序与并列打破规则。这一层
//!    一旦红就是**真缺陷**，不存在近似误差的解释空间。
//! 2. **弱断言（HNSW 近似）**——图检索本身允许漏邻，故只断言**在近似算法下仍必须
//!    成立的关系**：k 前缀交集比例、ef 增大 recall 不降、噪声扰动下邻居稳定性、
//!    足够远的新插入不得顶掉旧 top-1。阈值取自实测（见各常量注释）并留余量。
//!
//! EN: Leg 1 pins `force_flat_search = true` so search degenerates to exact brute
//! force and every property is a bit-exact equality against a reference top-k.
//! Leg 2 only asserts relations that must survive approximation (prefix overlap
//! ratio, recall monotonicity in `ef`, stability under small noise, a far-away
//! insert never displacing an old top-1). Thresholds are measured with margin.
//!
//! 环境前提 / EN: configs are constructed explicitly (`new_with_metric_and_config`),
//! so ambient `HOUYI_*` env cannot silently switch the searched path — the same
//! discipline `tests/recall_gate.rs` documents for `HOUYI_HNSW_*`.

use houyi_core::config::{HnswConfig, VectorConfig};
use houyi_vector::{DistanceMetric, HotBufferIndex, VectorIndex};
use proptest::prelude::*;

/// 维度：32 维 + 聚簇数据 —— HNSW 在均匀高维随机数据上近似邻域本身无意义
/// （几乎所有点等距），弱断言会变成噪声。聚簇让「近邻」有真实含义。
/// Dims: 32 with clustered data — on uniform high-dim data every point is nearly
/// equidistant and HNSW weak assertions degenerate into noise.
const DIMS: usize = 32;
/// 聚簇中心数（数据集生成用）。
/// EN: Number of cluster centers (used for dataset generation).
const CLUSTERS: usize = 4;

// ── 确定性数据集生成（splitmix64；不引外部 RNG，避免版本 API 漂移）──────────
// Deterministic dataset generation — splitmix64, no external RNG dependency.

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// [0, 1) 均匀 / uniform in [0, 1).
    /// EN: Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    fn vector(&mut self, dims: usize) -> Vec<f32> {
        (0..dims).map(|_| self.unit()).collect()
    }
}

/// 聚簇数据集：`clusters` 个中心，每点 = 中心 + 0.15 幅度噪声。
/// Clustered dataset: a point is a cluster center plus 0.15-amplitude noise.
fn clustered_dataset(seed: u64, n: usize) -> Vec<(u64, Vec<f32>)> {
    let mut rng = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..CLUSTERS).map(|_| rng.vector(DIMS)).collect();
    (0..n as u64)
        .map(|id| {
            let center = &centers[(id as usize) % CLUSTERS];
            let v = center
                .iter()
                .map(|c| c + (rng.unit() - 0.5) * 0.15)
                .collect();
            (id, v)
        })
        .collect()
}

// ── 参考实现（暴力）───────────────────────────────────────────────────
// EN: Reference implementation (brute force).

/// L2 距离平方（与内核 `distance()` 的 L2 语义一致：不开方）。
/// EN: Squared L2 distance (matches the kernel's `distance()` L2 semantics: no
/// square root).
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// 参考 top-k：按距离升序、距离相等按 id 升序（与内核排序规则一致）。
/// Reference top-k: distance ascending, ties broken by ascending id.
fn brute_topk(data: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = data.iter().map(|(id, v)| (*id, l2_sq(query, v))).collect();
    scored.sort_by(|(ia, da), (ib, db)| da.total_cmp(db).then(ia.cmp(ib)));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

// ── 索引构造／Index builders ────────────────────────────────────────
// EN: Index builders.

/// 精确（暴力）索引：`force_flat_search = true` ⇒ `search` 在全集上精确打分。
/// Exact (brute-force) index: `force_flat_search = true`.
fn flat_index(dims: usize) -> HotBufferIndex {
    let cfg = VectorConfig {
        force_flat_search: true,
        ..VectorConfig::default()
    };
    HotBufferIndex::new_with_metric_and_config(
        dims,
        DistanceMetric::L2,
        &cfg,
        &HnswConfig::default(),
    )
}

/// HNSW 索引：显式给 HnswConfig（不走 from_env），使环境变量无法切换构图路径。
/// HNSW index with an explicit config so ambient env cannot switch the build path.
fn hnsw_index(dims: usize, m: usize, ef_construction: usize, ef_search: usize) -> HotBufferIndex {
    let cfg = HnswConfig {
        m,
        ef_construction,
        ef_search,
        ..HnswConfig::default()
    };
    HotBufferIndex::new_with_metric_and_config(
        dims,
        DistanceMetric::L2,
        &VectorConfig::default(),
        &cfg,
    )
}

async fn drain(index: &HotBufferIndex) {
    while index.pending_build_count().await > 0 {
        let _ = index.run_async_builder_once().await;
    }
}

fn query_for(seed: u64) -> Vec<f32> {
    Rng::new(seed ^ 0xDEAD_BEEF).vector(DIMS)
}

// ── 随机输入策略／Strategies ────────────────────────────────────────
// EN: Strategies.

/// 随机数据集：1~14 条，**id 唯一**。
/// Random dataset: 1–14 rows with unique ids.
///
/// 为什么强制唯一（WHY）：存储以 id 为键，重复 id 会让「参考实现逐行对照」失去
/// 意义（参考侧会把同一 id 的两行当作两个点，索引侧只留一个）。重复插入的幂等
/// 性由专门的属性覆盖，不靠这条策略生成。
/// EN: The store keys on id, so duplicate ids would invalidate the row-by-row
/// differential (the reference would count two points where the index keeps one).
/// Insert idempotency has its own property instead of riding on this strategy.
fn arb_dataset() -> impl Strategy<Value = Vec<(u64, Vec<f32>)>> {
    (1usize..=6)
        .prop_flat_map(|n| prop::collection::vec(0u64..32, n..=(n + 8)))
        .prop_map(|mut ids| {
            ids.sort_unstable();
            ids.dedup();
            // 种子取自 id 内容：只按长度播种会让所有同长度 id 集合共享同一批
            // 向量坐标，压缩输入多样性（OCR 批次二十六）。
            // EN: seed from id contents — length-only seeding made every id set
            // of the same size share the same point cloud.
            let seed = ids.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, id| {
                (h ^ id).wrapping_mul(0x100_0000_01B3)
            });
            let mut rng = Rng::new(seed);
            ids.into_iter().map(|id| (id, rng.vector(DIMS))).collect()
        })
}

// ── 严格断言（暴力后端）／Strict properties ──────────────────────────
// EN: Strict properties (brute-force backend).

proptest! {
    #![proptest_config(ProptestConfig { cases: 96, ..ProptestConfig::default() })]

    /// 主对照：精确后端的结果必须与暴力参考**逐位相同**（id 与顺序）。
    /// Primary differential: the exact backend must match brute force bit for bit.
    #[test]
    fn flat_search_equals_brute_force(
        data in arb_dataset(),
        k in 1usize..6,
        seed in any::<u64>(),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let index = flat_index(DIMS);
            index.insert_batch(data.clone()).await.unwrap();
            drain(&index).await;
            let query = query_for(seed);
            let got = index.search(&query, k, None).await;
            let want = brute_topk(&data, &query, k);
            prop_assert_eq!(&got, &want, "query seed {} k {}", seed, k);
            Ok(())
        })?;
    }

    /// k 前缀性（严格）：`top-k` 必须是 `top-2k` 的前缀（精确后端下即前缀相等）。
    /// k-prefix (strict): `top-k` must be the prefix of `top-2k`.
    #[test]
    fn flat_top_k_is_prefix_of_top_2k(
        data in arb_dataset(),
        k in 1usize..5,
        seed in any::<u64>(),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let index = flat_index(DIMS);
            index.insert_batch(data.clone()).await.unwrap();
            drain(&index).await;
            let query = query_for(seed);
            let small = index.search(&query, k, None).await;
            let large = index.search(&query, k * 2, None).await;
            prop_assert_eq!(&small, &large[..small.len()].to_vec(), "prefix violated");
            Ok(())
        })?;
    }

    /// 过滤掩码（严格）：带 mask 的结果必须等于「先按 mask 限制数据集再取 top-k」。
    /// Filter mask (strict): equals brute force restricted to the mask.
    #[test]
    fn flat_filter_mask_equals_brute_force_on_mask(
        data in arb_dataset(),
        mask_ids in prop::collection::vec(0u64..12, 1..6),
        k in 1usize..5,
        seed in any::<u64>(),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let index = flat_index(DIMS);
            index.insert_batch(data.clone()).await.unwrap();
            drain(&index).await;
            let query = query_for(seed);
            let restricted: Vec<(u64, Vec<f32>)> = data
                .iter()
                .filter(|(id, _)| mask_ids.contains(id))
                .cloned()
                .collect();
            let want = brute_topk(&restricted, &query, k);
            let got = index.search(&query, k, Some(&mask_ids)).await;
            prop_assert_eq!(&got, &want, "mask {:?}", mask_ids);
            // 掩码外的 id 绝不允许出现。
            // EN: IDs outside the mask must never appear.
            for id in &got {
                prop_assert!(mask_ids.contains(id), "id {} outside mask", id);
            }
            Ok(())
        })?;
    }

    /// 重复插入幂等：同一 (id, vector) 反复写入不得改变结果集或计数。
    /// Duplicate insert is idempotent.
    #[test]
    fn duplicate_insert_is_idempotent(
        data in arb_dataset(),
        seed in any::<u64>(),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let index = flat_index(DIMS);
            index.insert_batch(data.clone()).await.unwrap();
            drain(&index).await;
            let query = query_for(seed);
            let before = index.search(&query, 5, None).await;
            let count_before = index.indexed_count().await;

            index.insert_batch(data.clone()).await.unwrap();
            drain(&index).await;
            let after = index.search(&query, 5, None).await;

            prop_assert_eq!(before, after, "duplicate insert changed results");
            prop_assert_eq!(count_before, index.indexed_count().await, "count changed");
            Ok(())
        })?;
    }

    /// 删除不可见 + 重插复活：`remove` 后任何 top-k 都不得含该 id；重插后必须可见。
    /// Delete hides the id everywhere; re-insert resurrects it.
    #[test]
    fn delete_hides_and_reinsert_resurrects(
        data in arb_dataset(),
        seed in any::<u64>(),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let index = flat_index(DIMS);
            index.insert_batch(data.clone()).await.unwrap();
            drain(&index).await;
            let query = query_for(seed);
            let victim = data[0].0;

            prop_assert!(index.remove(victim).await, "remove of an existing id must report true");
            // k 取全集大小：任何残留都会被看到。
            // EN: Take k as the full set size so any residue is visible.
            let after = index.search(&query, data.len() + 4, None).await;
            prop_assert!(!after.contains(&victim), "deleted id {} still visible: {:?}", victim, after);

            // 重插（用新向量）后必须复活，且位置与新向量一致。
            // EN: After re-inserting (with a new vector) the id must come back,
            // ranked according to the new vector.
            let mut rng = Rng::new(seed);
            let fresh = rng.vector(DIMS);
            index.insert(victim, fresh.clone()).await.unwrap();
            drain(&index).await;
            let resurrected = index.search(&query, data.len() + 4, None).await;
            prop_assert!(resurrected.contains(&victim), "re-inserted id {} not visible", victim);

            let mut expected: Vec<(u64, Vec<f32>)> = data
                .iter()
                .filter(|(id, _)| *id != victim)
                .cloned()
                .collect();
            expected.push((victim, fresh));
            let want = brute_topk(&expected, &query, data.len() + 4);
            prop_assert_eq!(resurrected, want, "resurrected set must match brute force");
            Ok(())
        })?;
    }

    /// 删除不存在的 id 必须报 false（不得谎报成功）。
    /// EN: Removing an absent id must report false (never falsely report success).
    #[test]
    fn remove_absent_id_reports_false(
        data in arb_dataset(),
        ghost in 100u64..200,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let index = flat_index(DIMS);
            index.insert_batch(data.clone()).await.unwrap();
            drain(&index).await;
            prop_assert!(!index.remove(ghost).await, "absent id must not report removal");
            Ok(())
        })?;
    }
}

/// 回归（P4.5.3 oracle 抓出）：mask 是**集合**，重复 id 不得各占一个 top-k 名额。
///
/// 旧实现里 ≤50K 的小 mask 分支把调用方传入的 mask 原样当候选列表（>50K 分支有
/// `seen` 去重），于是 `k=2` + `mask=[7, 7]` 返回 `[7, 7]`——同一 id 出现两次，浪费
/// 一个名额并污染响应。此处钉死「结果去重 + 与集合语义下的暴力参考一致」。
/// Regression: a duplicated mask id used to consume two top-k slots (`[7, 7]`), since
/// only the >50K branch deduped. The mask is a set; results must be distinct.
#[tokio::test]
async fn duplicate_mask_ids_do_not_duplicate_results() {
    let data = clustered_dataset(11, 40);
    let index = flat_index(DIMS);
    index.insert_batch(data.clone()).await.unwrap();
    drain(&index).await;

    let query = query_for(77);
    // mask 里每个 id 重复三次。
    // EN: Repeat every id in the mask three times.
    let mask: Vec<u64> = data.iter().flat_map(|(id, _)| [*id, *id, *id]).collect();
    let got = index.search(&query, 5, Some(&mask)).await;

    let unique: std::collections::HashSet<u64> = got.iter().copied().collect();
    assert_eq!(
        got.len(),
        unique.len(),
        "duplicate mask ids leaked into results: {got:?}"
    );
    // 与「掩码去重后」的暴力参考逐位一致。
    // EN: Bit-for-bit identical to the brute-force reference over the deduplicated mask.
    let want = brute_topk(&data, &query, 5);
    assert_eq!(got, want, "deduped-mask results must match brute force");
}

// ── 弱断言（HNSW 近似）／Weak properties ────────────────────────────
// EN: Weak properties (HNSW approximation).

/// HNSW 弱断言阈值——**实测基线**（macOS/aarch64，n=400，dims=32，聚簇，M=16/efc=200）：
///
/// | 关系 | 实测最小值 | 阈值 | 余量 |
/// |------|-----------|------|------|
/// | recall@10（ef=64） | 1.000（均值亦 1.000） | 0.95 | 与 `tests/recall_gate.rs` 同门 |
/// | recall 均值 vs ef | ef=16: 0.975 → ef=64: 1.0 → ef=256: 1.0 | 单调（容差 0.02） | — |
/// | top-5 ⊂ top-10 交集 | 1.000 | 0.80 | 只拦「整批邻居换人」 |
/// | 噪声扰动（±0.01/维）Jaccard | 0.818 | 0.70 | 防采样抖动误报 |
///
/// 阈值取在实测值下方一个安全距离：门禁要抓的是**结构性劣化**（图断边、ef 失效、
/// 距离语义漂移），不是单次采样的随机抖动。若将来 HNSW 参数或数据分布改动导致
/// 实测基线移动，请重新测量后再调阈值——不要凭感觉放宽。
/// EN: Thresholds sit a safe distance below measured minima so the gate catches
/// structural breakage (broken graph edges, ignored `ef`, drifted distance
/// semantics) rather than sampling jitter. Re-measure before adjusting.
const HNSW_RECALL_MIN: f32 = 0.95;
const HNSW_PREFIX_OVERLAP_MIN: f32 = 0.80;
const HNSW_NOISE_JACCARD_MIN: f32 = 0.70;
/// level-0 可达率下限，口径与 `tests/common/clustered_recall.rs` 的
/// `REACHABLE_MIN_RATIO` 一致；该模块未被本文件引入（会额外编译整套语料
/// 生成器），因此用本地常量显式标注耦合，调整任一处必须同时调整另一处。
/// EN: Level-0 reachability floor, matching `tests/common/clustered_recall.rs`'s
/// `REACHABLE_MIN_RATIO`. That module is not pulled in here (it would compile
/// the whole corpus generator), so this local constant names the coupling
/// explicitly — tuning one requires tuning the other.
const GRAPH_REACHABLE_MIN_RATIO: f64 = 0.95;

/// 图健康门禁：level-0 从入口点 BFS 的可达率 ≥95%，且不存在 L0 度 0 的存活
/// 节点。召回均值会掩盖整簇不可达（4653 事故时 L0 可达率已跌到 48% 才开始
/// 报警）；这两条不变量让结构断裂在召回数字崩塌之前暴露，口径与
/// `tests/common/clustered_recall.rs` 的 `REACHABLE_MIN_RATIO` 一致。
/// EN: Graph-health gate: level-0 reachability from the entry point >= 95% and
/// no live node with L0 degree 0. Recall averages hide whole-cluster islands
/// (the 4653 incident was already at 48% reachability before recall alarms
/// fired); these invariants expose structural breaks before the recall number
/// collapses, matching `tests/common/clustered_recall.rs`.
fn assert_graph_health_for_data(index: &HotBufferIndex, data: &[(u64, Vec<f32>)], label: &str) {
    let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
    assert_graph_health(index, &ids, label);
}

fn assert_graph_health(index: &HotBufferIndex, ids: &[u64], label: &str) {
    let ep = match index.entry_point() {
        Some(ep) => ep,
        None => {
            assert!(
                ids.is_empty(),
                "{label}: non-empty data without entry point"
            );
            return;
        }
    };
    let valid: std::collections::HashSet<u64> = ids.iter().copied().collect();
    let mut seen = std::collections::HashSet::new();
    seen.insert(ep);
    let mut queue = std::collections::VecDeque::from([ep]);
    while let Some(cur) = queue.pop_front() {
        for n in index.neighbors_of(cur) {
            if valid.contains(&n) && seen.insert(n) {
                queue.push_back(n);
            }
        }
    }
    let reachable_min = (ids.len() as f64 * GRAPH_REACHABLE_MIN_RATIO).ceil() as usize;
    assert!(
        seen.len() >= reachable_min,
        "{label}: level-0 reachability {}/{} < {reachable_min}",
        seen.len(),
        ids.len()
    );
    // 单节点图合法地只有一个入口点且有 0 条边：该不变量仅适用于多节点图。
    // EN: A single-node graph legitimately has only the entry point and zero
    // edges; this invariant applies to multi-node graphs only.
    if ids.len() > 1 {
        let zero_degree = ids
            .iter()
            .filter(|&&id| index.neighbors_of(id).is_empty())
            .count();
        assert_eq!(
            zero_degree, 0,
            "{label}: {zero_degree} live nodes have L0 degree 0 (unreachable islands)"
        );
    }
}

fn recall(pred: &[u64], truth: &[u64]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let hit = pred.iter().filter(|id| truth.contains(id)).count();
    hit as f32 / truth.len() as f32
}

fn jaccard(a: &[u64], b: &[u64]) -> f32 {
    let sa: std::collections::HashSet<u64> = a.iter().copied().collect();
    let sb: std::collections::HashSet<u64> = b.iter().copied().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    inter / union
}

/// 随机 recall 门：多 seed 数据集 × 多 query，recall@10 必须过阈值。
/// 这是 `tests/recall_gate.rs` 固定配置的随机化扩展——同一门槛、随机数据。
/// Randomized recall gate: an extension of the fixed-config gate in
/// `tests/recall_gate.rs` to random datasets and queries.
#[tokio::test]
async fn hnsw_random_recall_gate() {
    for seed in [1u64, 7, 31, 97, 512, 4096] {
        let data = clustered_dataset(seed, 400);
        let index = hnsw_index(DIMS, 16, 200, 64);
        index.insert_batch(data.clone()).await.unwrap();
        drain(&index).await;
        assert_graph_health_for_data(&index, &data, &format!("random_recall seed={seed}"));

        for q in 0..8u64 {
            let query = query_for(seed.wrapping_mul(31).wrapping_add(q));
            let got = index.search(&query, 10, None).await;
            let truth = brute_topk(&data, &query, 10);
            let r = recall(&got, &truth);
            assert!(
                r >= HNSW_RECALL_MIN,
                "recall regression: seed={seed} q={q} recall={r:.3} < {HNSW_RECALL_MIN} \
                 (got={got:?} truth={truth:?})"
            );
        }
    }
}

/// ef 单调性：同一图、同一 query 集，`ef_search` 增大时平均 recall 不得下降。
/// Recall monotonicity in `ef_search` — a larger beam must not recall less on average.
#[tokio::test]
async fn hnsw_recall_is_monotonic_in_ef() {
    let data = clustered_dataset(99, 400);
    let index = hnsw_index(DIMS, 16, 200, 16);
    index.insert_batch(data.clone()).await.unwrap();
    drain(&index).await;
    assert_graph_health_for_data(&index, &data, "ef_monotonic");

    let queries: Vec<Vec<f32>> = (0..8u64).map(|q| query_for(1000 + q)).collect();
    let truth: Vec<Vec<u64>> = queries.iter().map(|q| brute_topk(&data, q, 10)).collect();

    let mut means = Vec::new();
    for ef in [16usize, 64, 256] {
        index.set_hnsw_ef_search(ef).unwrap();
        let mut total = 0.0f32;
        for (query, want) in queries.iter().zip(&truth) {
            let got = index.search(query, 10, None).await;
            total += recall(&got, want);
        }
        means.push(total / queries.len() as f32);
    }
    assert!(
        means[1] + 0.02 >= means[0],
        "recall dropped when ef grew: {means:?}"
    );
    assert!(
        means[2] + 0.02 >= means[1],
        "recall dropped when ef grew: {means:?}"
    );
}

/// k 前缀性（弱）：`top-k` 与 `top-2k` 的重叠比例不得低于阈值。
/// 近似下不要求前缀相等，但「完全换了一批邻居」是结构性故障。
/// Weak k-prefix: overlap ratio between top-k and top-2k must stay above the bound.
#[tokio::test]
async fn hnsw_top_k_overlaps_top_2k() {
    let data = clustered_dataset(7, 400);
    let index = hnsw_index(DIMS, 16, 200, 64);
    index.insert_batch(data.clone()).await.unwrap();
    drain(&index).await;
    assert_graph_health_for_data(&index, &data, "top_k_prefix");

    for q in 0..8u64 {
        let query = query_for(500 + q);
        let small = index.search(&query, 5, None).await;
        let large = index.search(&query, 10, None).await;
        let overlap = small.iter().filter(|id| large.contains(id)).count() as f32 / 5.0;
        assert!(
            overlap >= HNSW_PREFIX_OVERLAP_MIN,
            "top-5/top-10 overlap {overlap:.2} < {HNSW_PREFIX_OVERLAP_MIN} \
             (small={small:?} large={large:?})"
        );
    }
}

/// 噪声稳定性（metamorphic）：查询加微小噪声后邻居集合的 Jaccard 不得崩塌。
/// Metamorphic noise stability: a slightly perturbed query keeps most neighbors.
#[tokio::test]
async fn hnsw_neighbors_stable_under_small_query_noise() {
    let data = clustered_dataset(31, 400);
    let index = hnsw_index(DIMS, 16, 200, 64);
    index.insert_batch(data.clone()).await.unwrap();
    drain(&index).await;
    assert_graph_health_for_data(&index, &data, "noise_stability");

    let mut rng = Rng::new(4242);
    for q in 0..8u64 {
        let query = query_for(900 + q);
        let noisy: Vec<f32> = query
            .iter()
            .map(|x| x + (rng.unit() - 0.5) * 0.02)
            .collect();
        let a = index.search(&query, 10, None).await;
        let b = index.search(&noisy, 10, None).await;
        let j = jaccard(&a, &b);
        assert!(
            j >= HNSW_NOISE_JACCARD_MIN,
            "neighbor set unstable under small noise: jaccard={j:.2} (a={a:?} b={b:?})"
        );
    }
}

/// 远插入不改旧 top-1（metamorphic，近似下也应严格成立）：
/// 新点与查询的距离比当前 top-1 远出一大截时，它不可能顶掉 top-1。
/// A far-away insert must never displace an existing top-1.
#[tokio::test]
async fn far_insert_does_not_displace_top1() {
    let data = clustered_dataset(512, 300);
    let index = hnsw_index(DIMS, 16, 200, 64);
    index.insert_batch(data.clone()).await.unwrap();
    drain(&index).await;
    assert_graph_health_for_data(&index, &data, "far_insert");

    let query = query_for(2024);
    let before = index.search(&query, 1, None).await;
    assert_eq!(before.len(), 1);
    let top1 = before[0];
    let top1_dist = l2_sq(&query, &data.iter().find(|(id, _)| *id == top1).unwrap().1);

    // 构造一个与查询距离远超 top-1 的新点：各维平移 +2.0，远离数据所在的 [0,1) 流形。
    // 注意不能只用 `1.0 - x`——聚簇数据的中心本身随机落在 [0,1)，取反后可能落在
    // 另一个簇附近（实测仅 3.7×），setup 断言会挡住这种「假远点」。
    // EN: Shift every dim by +2.0 to leave the [0,1) manifold the data lives on.
    // `1.0 - x` is not enough — cluster centers are random in [0,1), so the mirror
    // point can land near another cluster (measured only 3.7× farther).
    let far: Vec<f32> = query.iter().map(|x| x + 2.0).collect();
    let far_dist = l2_sq(&query, &far);
    assert!(
        far_dist > top1_dist * 10.0,
        "test setup: far point must be much farther ({far_dist} vs {top1_dist})"
    );

    let new_id = 999_999u64;
    // 必须走 `insert_batch` + drain：单条 `insert` 只进热缓冲，不排队建图；
    // 主图非空时 HNSW 无掩码检索只从图里取候选，新点根本不会成为候选——
    // 那样这个断言就永远为真（OCR 批次二十六）。
    // EN: use insert_batch + drain so the point is actually built into the graph;
    // a hot-only insert never becomes a candidate on the no-mask HNSW path.
    index.insert_batch(vec![(new_id, far)]).await.unwrap();
    drain(&index).await;

    let after = index.search(&query, 1, None).await;
    assert_eq!(
        after,
        vec![top1],
        "a far insert displaced top-1 (was {top1}, now {:?})",
        after
    );
}
