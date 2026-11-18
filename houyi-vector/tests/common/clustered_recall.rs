//! 成簇数据召回回归门禁的共享数据/度量工具：`plan_apply_group_recall.rs` 与
//! `legacy_group_recall.rs` 经 `#[path]` 各自引入同一份实现，保证两种构图路径
//! 用完全相同的数据分布、查询、阈值与断言对拍，避免门禁口径漂移。每个文件另
//! 带一个 `#[ignore]` 的 50k 大簇用例（release 下由
//! `scripts/clustered_recall_50k_gate.sh` 运行：debug 的 50k 暴力真值与构图
//! 需要数分钟，超出每次提交的预算）。
//!
//! Shared data/metric helpers for the clustered-data recall gates. Both the
//! default plan/apply gate and the legacy-grouped gate include this file via
//! `#[path]`, so both build paths are compared on identical data, queries,
//! thresholds, and assertions instead of drifting apart. Each file also carries
//! an `#[ignore]` 50k large-cluster case, run in release by
//! `scripts/clustered_recall_50k_gate.sh` (the 50k brute-force truth and build
//! take minutes in debug, beyond the per-commit budget).

// 召回阈值与 `recall_at_k` 复用既有 Recall-Check 共享模块（tests/recall_gate.rs
// 同源）；本模块只用到其中一部分函数，未用项显式放行。
// EN: Reuse the canonical Recall-Check helpers (the same module backing
// tests/recall_gate.rs); only part of it is used here, so allow dead code.
#[path = "recall.rs"]
#[allow(dead_code)]
mod recall_common;

use houyi_vector::HotBufferIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashSet, VecDeque};

/// 语料规模 / 维度 / 簇数：与知识库评测的 4653 × 512 维文本嵌入同量级。
/// EN: Corpus size / dims / cluster count, matching the knowledge-base
/// benchmark's 4653 × 512-d text embeddings.
pub const DATA_N: usize = 4653;
/// 50k 大簇门禁的语料规模：与线上单 collection 的 50k+ 量级对齐（同一
/// `DATA_CLUSTERS` 下每簇约 781 点，远大于自适应组宽 ~195，是复现整簇孤岛的
/// 规模条件）。
/// EN: Corpus size of the 50k large-cluster gate, aligned with the 50k+
/// single-collection scale seen in production (with the same `DATA_CLUSTERS`,
/// each cluster holds ~781 points, far above the adaptive group width ~195 —
/// the scale at which whole-cluster islands reproduce).
pub const DATA_N_LARGE: usize = 50_000;
pub const DATA_DIMS: usize = 512;
pub const DATA_CLUSTERS: usize = 64;
/// 固定种子：两条构图路径必须吃到逐位相同的语料。
/// EN: Fixed seed so both build paths consume bit-identical corpora.
pub const DATA_SEED: u64 = 0xA11CE;
/// 簇内噪声：故意压得很低，使每批写入的兄弟节点互为最近邻——这是复现
/// 「组内兄弟不可见 + 对称剪枝」的最小压力分布。
/// EN: Intra-cluster noise deliberately kept tiny so in-flight siblings are
/// each other's nearest neighbors — the minimal stress distribution for the
/// "siblings invisible inside a group + symmetric pruning" defect.
pub const DATA_NOISE: f32 = 0.02;
/// 查询数 / k：每簇恰好一条查询（全覆盖，避免个别簇的断裂漏检），
/// k 与 Recall-Check 的 k=10 口径一致。
/// EN: Query count / k: exactly one query per cluster (full coverage so a
/// single broken cluster cannot slip through); k follows the k=10
/// Recall-Check convention.
pub const QUERY_COUNT: usize = DATA_CLUSTERS;
pub const K: usize = 10;
/// ef 扫描点：每个 ef 的**均值**都要达标，避免「靠大 ef 掩盖图结构断裂」。
/// 本门禁语料 512 维且 `VectorConfig::default().auto_hnsw = true`，服务端据此
/// 推断默认 `ef_search = 256`（129..=1024 档）；末点 512 比该推断默认更严。
/// 每轮先用 setter 把底层束宽真正钉到该值（override 只能抬高、不能降低）。
/// EN: ef sweep points: the **mean** must pass at every ef, so a broken graph
/// cannot be masked by a large ef. This gate's corpus is 512-d with
/// `VectorConfig::default().auto_hnsw = true`, so the server infers the
/// default `ef_search = 256` (the 129..=1024 bucket); the last point 512 is
/// deliberately stricter than that inferred default. Each round pins the L0
/// beam through the setter first (the override can only raise, never lower).
pub const EF_SWEEP: [usize; 4] = [16, 64, 256, 512];
/// 单查询召回下限（mean 用共享 `RECALL_MIN`；k=10 时 0.9 = 漏 1 个邻居）。
/// **每个 ef 的每个查询都必须达标（零容差）**：ef 扫描改为每轮 setter 真正
/// 钉到 16/64/256/512 之后（此前 override 只能抬高，低 ef 实际复用推断默认
/// 256），4653 与 50k 全部构图路径实测 min=1.0000、below_floor=0；旧注释里
/// 记录的 #49/#55 是「低 ef 实际跑 256」时代的数字，已随 ef 钉定失效。均值
/// 门禁在 64 查询下单个破簇仍可能 ≥0.984 > 0.95，因此单查询下限必须零容差，
/// 否则第二个破簇会藏进均值里。
/// EN: Per-query recall floor (mean uses the shared `RECALL_MIN`; at k=10,
/// 0.9 means one missed neighbor). **Every query at every ef must pass (zero
/// tolerance).** After the ef sweep started pinning 16/64/256/512 through the
/// setter (the override can only raise, so low ef used to replay the inferred
/// 256), the 4653 and 50k build paths all measure min=1.0000 and
/// below_floor=0; the old #49/#55 note came from that replayed-256 era. With
/// 64 queries, a single broken cluster can still leave the mean at ≥0.984 >
/// 0.95, so the per-query floor must be zero-tolerance or a second broken
/// cluster could hide inside the mean.
pub const PER_QUERY_MIN: f32 = 0.9;
/// 近邻等价容差：合成语料簇内噪声极低（`DATA_NOISE`），同簇近邻的余弦分差常
/// 低于 1e-3，ANN 无法区分；把「与第 k 名分差 ≤ 该值」的点都视为等价真值，
/// 避免把浮点级排序差异误判为结构回归。结构断裂（整簇不可达/sibling 被剪成
/// 孤岛）不会因此被掩盖：那时返回集与真值的分差远大于该阈值。
/// EN: Neighbor-equivalence tolerance: with the very low intra-cluster noise
/// (`DATA_NOISE`), same-cluster cosine scores differ by less than 1e-3 and ANN
/// cannot rank them reliably; points within this delta of the k-th score count
/// as equivalent ground truth, so float-level ordering is not misread as a
/// structural regression. A real break (unreachable cluster / sibling islands)
/// is not masked — its returned scores sit far below the k-th threshold.
pub const TIE_EPS: f32 = 1e-3;
/// level-0 可达率下限：远端孤岛比召回崩塌更早暴露结构断裂。
/// EN: Minimum level-0 reachability ratio: distant islands expose structural
/// breaks earlier than recall collapse.
pub const REACHABLE_MIN_RATIO: f64 = 0.95;

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// 块状局部性成簇语料：id 按连续块归属同一簇（模拟文本嵌入的连续 chunk）。
/// EN: Block-local clustered corpus: consecutive id blocks belong to one
/// cluster, mimicking contiguous text-embedding chunks.
pub fn clustered_data(n: usize, dims: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centroids: Vec<Vec<f32>> = (0..DATA_CLUSTERS)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect();
    (0..n)
        .map(|id| {
            let c = (id * DATA_CLUSTERS / n).min(DATA_CLUSTERS - 1);
            let noise: Vec<f32> = (0..dims)
                .map(|_| (rng.gen::<f32>() - 0.5) * DATA_NOISE)
                .collect();
            let v: Vec<f32> = (0..dims)
                .map(|j| 0.45 * centroids[c][j] + noise[j])
                .collect();
            (id as u64, normalize(v))
        })
        .collect()
}

/// 每个簇取一条查询：按与 `clustered_data` 相同的块划分 `[start, end)` 取块内
/// 中心点（而非等距步长——步长采样仍可能整簇漏掉），再叠加固定 RNG 扰动
/// 模拟「问题嵌入落在语料附近」。
/// EN: One query per cluster: uses the same block partition `[start, end)` as
/// `clustered_data` and samples the block's interior midpoint (evenly spaced
/// stepping could still miss whole clusters), then perturbation with a fixed
/// RNG mimics query embeddings landing near corpus points.
pub fn perturbed_queries(data: &[(u64, Vec<f32>)]) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let n = data.len();
    let mut queries = Vec::with_capacity(DATA_CLUSTERS);
    for c in 0..DATA_CLUSTERS {
        let start = c * n / DATA_CLUSTERS;
        let end = (c + 1) * n / DATA_CLUSTERS;
        // 语料比簇数还小时跳过空块（本门禁语料远大于簇数，仅为防御）。
        // EN: Skip empty blocks when the corpus is smaller than the cluster
        // count (defensive only; the gate corpus is far larger).
        if end <= start {
            continue;
        }
        let (_, v) = &data[start + (end - start) / 2];
        let q: Vec<f32> = v
            .iter()
            .map(|x| x + (rng.gen::<f32>() - 0.5) * 0.05)
            .collect();
        queries.push(normalize(q));
    }
    queries
}

/// 归一化向量的余弦相似度 = 点积；降序、id 升序破平。返回与第 k 名分差
/// ≤ `eps` 的**全部** id（等价近邻集，理由见 `TIE_EPS`），至少包含前 k 名。
/// EN: Cosine similarity of normalized vectors = dot product; descending,
/// id-ascending tie-break. Returns ALL ids within `eps` of the k-th score (the
/// equivalence set, see `TIE_EPS`), always including the exact top-k.
pub fn brute_topk(data: &[(u64, Vec<f32>)], query: &[f32], k: usize, eps: f32) -> Vec<u64> {
    // 边界：k=0 或空语料无真值可言；k 超过语料时以末名分数为阈值，等价近邻集
    // 即全体（与旧的 `take(k)` 语义一致）。
    // EN: Edge cases: k=0 or an empty corpus has no ground truth; when k exceeds
    // the corpus the last score becomes the threshold and the equivalence set is
    // everything (matching the old `take(k)` semantics).
    if k == 0 || data.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(u64, f32)> = data
        .iter()
        .map(|(id, v)| (*id, query.iter().zip(v.iter()).map(|(a, b)| a * b).sum()))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let kth = scored[k.min(scored.len()) - 1].1;
    scored
        .into_iter()
        .take_while(|(_, score)| *score >= kth - eps)
        .map(|(id, _)| id)
        .collect()
}

/// 容差召回：命中判定的真值集是 `brute_topk` 的等价近邻集（可多于 k 个），
/// 因此不能复用 `recall_common::recall_at_k`（后者会把真值截断到 k）。
/// EN: Tolerance-aware recall: the ground truth is `brute_topk`'s equivalence
/// set (possibly larger than k), so `recall_common::recall_at_k` cannot be
/// reused here — it truncates the truth set to k.
fn tolerant_recall_at_k(pred: &[u64], truth: &[u64], k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let hit = pred
        .iter()
        .take(k)
        .filter(|id| truth.iter().any(|x| x == *id))
        .count();
    hit as f32 / k as f32
}

/// level-0 从入口点 BFS 的可达节点数：整批规划若让新节点互为孤岛，这个数会
/// 显著低于 `ids.len()`，比单看 recall 更早暴露结构断裂。
/// EN: Level-0 reachability from the entry point. If grouped planning leaves
/// new nodes as islands, this falls well below `ids.len()` and exposes the
/// structural break earlier than recall alone.
pub fn level0_reachable(index: &HotBufferIndex, ids: &[u64]) -> usize {
    let valid: HashSet<u64> = ids.iter().copied().collect();
    let ep = match index.entry_point() {
        Some(ep) => ep,
        None => return 0,
    };
    let mut seen = HashSet::new();
    seen.insert(ep);
    let mut queue = VecDeque::from([ep]);
    while let Some(cur) = queue.pop_front() {
        // `neighbors_of` 已过滤 `EMPTY_NEIGHBOR` 哨兵，这里只做成员校验。
        // EN: `neighbors_of` already filters the `EMPTY_NEIGHBOR` sentinel;
        // only membership needs checking here.
        for n in index.neighbors_of(cur) {
            if valid.contains(&n) && seen.insert(n) {
                queue.push_back(n);
            }
        }
    }
    seen.len()
}

/// 一次生成两条门禁共用的语料、查询与暴力真值。
/// EN: Build once the corpus, queries, and brute-force ground truth shared by
/// both gates.
pub struct ClusteredCorpus {
    pub data: Vec<(u64, Vec<f32>)>,
    pub queries: Vec<Vec<f32>>,
    pub truth: Vec<Vec<u64>>,
    pub ids: Vec<u64>,
}

impl ClusteredCorpus {
    /// 按语料基数生成：50k 大簇门禁复用同一数据分布、查询扰动与真值口径，
    /// 使两条规模的门禁不会因构造差异而漂移。
    /// 前置条件：`n >= DATA_CLUSTERS`（每簇至少一个点）。
    /// EN: Build for the given corpus size: the 50k large-cluster gate reuses the
    /// same distribution, query perturbation, and ground-truth semantics, so the
    /// gates at the two scales cannot drift apart.
    /// Precondition: `n >= DATA_CLUSTERS` (at least one point per cluster).
    pub fn new_with(n: usize) -> Self {
        // `n < DATA_CLUSTERS` 会出现空簇：采样拿不到向量，随后的 QUERY_COUNT
        // 断言会以「查询数量不对」失败而掩盖真实原因，先在此命名实际约束。
        // EN: With `n < DATA_CLUSTERS` some clusters are empty, sampling yields
        // no vector for them, and the later QUERY_COUNT assertion would fail
        // with a misleading "wrong query count" message — name the real
        // constraint first.
        assert!(
            n >= DATA_CLUSTERS,
            "corpus size {n} must cover all {DATA_CLUSTERS} clusters; \
             smaller corpora leave empty clusters and break query sampling"
        );
        let data = clustered_data(n, DATA_DIMS, DATA_SEED);
        let queries = perturbed_queries(&data);
        // 硬断言而非 `debug_assert_eq!`：release 测试构建会编译掉后者，若语料
        // 缩小到出现空块，门禁会静默少跑查询而均值语义仍「成立」。
        // EN: Hard assertion rather than `debug_assert_eq!`: release test builds
        // compile the latter out, and if the corpus shrank enough for empty
        // blocks the gate would silently run fewer queries while the mean
        // assertion still "passes".
        assert_eq!(
            queries.len(),
            QUERY_COUNT,
            "sampling must yield exactly one query per cluster"
        );
        let truth = queries
            .iter()
            .map(|q| brute_topk(&data, q, K, TIE_EPS))
            .collect();
        let ids = data.iter().map(|(id, _)| *id).collect();
        Self {
            data,
            queries,
            truth,
            ids,
        }
    }

    pub fn new() -> Self {
        Self::new_with(DATA_N)
    }
}

impl Default for ClusteredCorpus {
    fn default() -> Self {
        Self::new()
    }
}

/// 对已建好的索引跑完整门禁：level-0 可达率 + 每个 ef 的 mean/min 召回。
/// `label` 只用于失败信息，标明是哪条构图路径。
/// EN: Run the full gate against a built index: level-0 reachability plus
/// mean/min recall at every ef. `label` only appears in failure messages to
/// identify the build path.
pub async fn assert_clustered_recall(
    index: &HotBufferIndex,
    corpus: &ClusteredCorpus,
    label: &str,
) {
    let reachable = level0_reachable(index, &corpus.ids);
    // 向上取整：截断会让 95% 门槛悄悄变松（4653 × 0.95 = 4420.35 → 4420）。
    // EN: Round up: truncation would silently loosen the 95% floor
    // (4653 × 0.95 = 4420.35 → 4420).
    let reachable_min = (corpus.ids.len() as f64 * REACHABLE_MIN_RATIO).ceil() as usize;
    let mut recalls = Vec::new();
    for ef in EF_SWEEP {
        // `search_scored_with_ef_override` 的 override 只能抬高 `ef_search`，
        // 不能降低；必须先用 setter 把本轮的底层束宽真正钉到 ef，否则
        // ef=16/64 会静默复用推断出的 256（OCR：门禁的「ef 扫描」变成重复点）。
        // EN: The override can only raise `ef_search`, never lower it; pin the
        // L0 beam for this round through the setter, otherwise ef=16/64 silently
        // replay the inferred 256 (OCR: the "ef sweep" degenerates into
        // duplicate points).
        index.set_hnsw_ef_search(ef).expect("set ef_search");
        let mut sum = 0.0f32;
        let mut min = 1.0f32;
        let mut below_floor = 0usize;
        for (q, truth) in corpus.queries.iter().zip(corpus.truth.iter()) {
            let pred = index.search_scored_with_ef_override(q, K, None, None).await;
            let got: Vec<u64> = pred.into_iter().map(|(id, _)| id).collect();
            let recall = tolerant_recall_at_k(&got, truth, K);
            if recall < PER_QUERY_MIN {
                below_floor += 1;
            }
            sum += recall;
            min = min.min(recall);
        }
        recalls.push((ef, sum / corpus.queries.len() as f32, min, below_floor));
    }
    assert!(
        reachable >= reachable_min,
        "{label}: level-0 unreachable nodes: reachable={reachable}/{} \
         (need >= {reachable_min}); recalls={recalls:?}",
        corpus.ids.len()
    );
    for &(ef, mean, min, below_floor) in &recalls {
        assert!(
            mean >= recall_common::RECALL_MIN,
            "{label}: clustered recall regression at ef={ef}: mean={mean:.4} \
             min={min:.4} below_floor={below_floor} (need mean >= {}); \
             reachable={reachable}/{}; recalls={recalls:?}",
            recall_common::RECALL_MIN,
            corpus.ids.len()
        );
        // 单查询下限在每个 ef 都零容差（等价于 min ≥ PER_QUERY_MIN）。ef 扫描
        // 真正钉到 16/64/256/512 后，4653 与 50k 各构图路径实测 min=1.0000、
        // below_floor=0；旧的 #49/#55 记录来自低 ef 实际复用推断 256 的时代，
        // 已随 ef 钉定失效。零容差能拒绝任何藏在低 ef 均值里的破簇：64 查询
        // 下单簇破簇的均值仍可能 ≥0.984 > 0.95，只靠均值会漏掉更宽回归。历史
        // 缺陷（固定 1024 组宽）在每个 ef mean≈0.53、min=0、大量查询低于下限
        // 且可达率 2229/4653，均值/计数/可达率断言都会当场抓住。
        // EN: The per-query floor is zero-tolerance at every ef (equivalent to
        // min >= PER_QUERY_MIN). After the ef sweep genuinely pins
        // 16/64/256/512, the 4653 and 50k build paths all measure min=1.0000 and
        // below_floor=0; the old #49/#55 note came from the era when low ef
        // replayed the inferred 256 and is obsolete. Zero tolerance rejects any
        // broken cluster hidden behind a low-ef mean: with 64 queries one broken
        // cluster can still leave the mean at ≥0.984 > 0.95. The historical
        // defect (fixed 1024 width) has mean≈0.53, min=0, many queries below the
        // floor, and reachability 2229/4653 at every ef, so the mean, count, and
        // reachability assertions all fail it immediately.
        assert!(
            below_floor == 0,
            "{label}: {below_floor} queries below the per-query floor at ef={ef} \
             (require 0); mean={mean:.4} min={min:.4}; reachable={reachable}/{}; \
             recalls={recalls:?}",
            corpus.ids.len()
        );
    }
}
