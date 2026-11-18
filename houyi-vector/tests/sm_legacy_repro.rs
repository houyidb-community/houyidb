//! sm 生产形态 legacy 构图门禁：逐位复刻 sm 的 mulberry32 成簇语料（50k×512
//! 维、64 簇、噪声 0.02、数据 seed 0xA11CE、查询 seed 0xBEEF），按 collection
//! 后端的真实 `BuildConfig::default()`（max_batch=512、串行）分批 drain，模拟
//! 服务端 worker 每 tick 约 300 行的写-构图节奏。
//!
//! 背景（sm 在生产形态上实测到的残余缺陷）：
//! - level-0 已 100% 可达、`ensure_connectivity` 也补不出任何桥，但 legacy
//!   路径的 q45/q62 在所有 ef 下都返回错误簇（0.9656，与 ef 无关）；
//! - 根因是**上层导航束**：layer-1 估计 3125 个节点时，固定束宽 64/128 的
//!   贪心下降从入口点找不到目标簇的层内代表，束宽 256/392 才恢复；同时
//!   layer-1 还有 12 个可达性孤点（L0 兜底覆盖不到）。
//! - 修复：上层束宽按层规模自适应（见 `HnswGraph::upper_layer_ef_for`），
//!   连通性兜底推广到全层（见 `HnswGraph::ensure_connectivity`）。
//!
//! 本门禁钉住修复后的两项结果：全层可达 100%；legacy 各 ef 容差召回 ≥0.99，
//! 且 q45/q62 必须回到真值簇。
//!
//! EN: Production-shaped legacy gate replicating sm's mulberry32 clustered
//! corpus bit-for-bit (50k x 512-d, 64 clusters, noise 0.02, data seed
//! 0xA11CE, query seed 0xBEEF), built through the real collection-backend
//! `BuildConfig::default()` (max_batch=512, serial) and drained in ~300-row
//! simulated worker ticks.
//!
//! Background (the residual defect sm measured in production shape):
//! - L0 is 100% reachable and the sweep adds nothing, yet legacy q45/q62
//!   return the wrong cluster at every ef (0.9656, ef-independent);
//! - the root cause is the **upper navigation beam**: with ~3125 layer-1
//!   nodes, fixed widths 64/128 cannot route the greedy descent from the entry
//!   to the target cluster's in-layer representative; 256/392 recovers. Layer 1
//!   also had 12 reachability orphans that the L0 sweep could not cover.
//! - fix: the upper beam scales adaptively with layer size (see
//!   `HnswGraph::upper_layer_ef_for`), and the connectivity backstop covers
//!   every layer (see `HnswGraph::ensure_connectivity`).
//!
//! This gate pins both outcomes: 100% all-layer reachability; legacy tolerant
//! recall >=0.99 at every ef, with q45/q62 restored to their truth clusters.

use houyi_core::config::{BuildConfig, HnswConfig, VectorConfig};
use houyi_vector::{DistanceMetric, HnswBatchPlanMode, HotBufferIndex, VectorIndex};
use std::collections::{HashSet, VecDeque};

const N: usize = 50_000;
/// 大图导航探针语料规模：layer-1 估计约 15625 节点，自适应束宽为约
/// 1954（覆盖率 ~12.5%，封顶 4096 尚未触顶）。
/// EN: Large-graph navigation probe size: layer-1 estimates ~15625 nodes, so
/// the adaptive beam is ~1954 (~12.5% coverage; the 4096 cap is not reached).
const LARGE_N: usize = 250_000;
const DIMS: usize = 512;
const CLUSTERS: usize = 64;
const NOISE: f64 = 0.02;
const DATA_SEED: u32 = 0xA11CE;
const QUERY_SEED: u32 = 0xBEEF;
const K: usize = 10;
const TIE_EPS: f64 = 1e-3;
/// 模拟 worker tick 时的请求批量：3×100 行（生产实测写入约 7ms/批、tick 20ms）。
/// EN: Requests accumulated per simulated worker tick: 3 x 100 rows (production
/// measured ~7ms per request with a 20ms tick).
const DRAIN_EVERY: usize = 3;

/// `(id, vector)` 语料；id 从 1 开始（与服务端写入一致）。
/// EN: `(id, vector)` corpus; ids start at 1 (matching server writes).
type Corpus = Vec<(u64, Vec<f32>)>;

struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// 与 sm/bench 的 JS `mulberry32` 逐位一致（u32 wrapping 语义）。
    /// EN: Bit-identical to the JS `mulberry32` used by sm/bench (u32 wrapping).
    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6D2B79F5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }
}

fn normalize_f64(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let n = if n == 0.0 { 1.0 } else { n };
    for x in v {
        *x /= n;
    }
}

/// 精确复刻 sm E2E 的语料与查询生成（RNG 消费顺序不可变）。
/// EN: Exact replica of sm's E2E corpus and query generation (RNG order fixed).
fn build_corpus(n: usize) -> (Corpus, Vec<Vec<f32>>) {
    let mut rng = Mulberry32::new(DATA_SEED);
    let mut centroids = vec![vec![0.0f64; DIMS]; CLUSTERS];
    for c in &mut centroids {
        for x in c.iter_mut() {
            *x = rng.next() - 0.5;
        }
    }
    let mut data: Corpus = Vec::with_capacity(n);
    for i in 0..n {
        let c = (i * CLUSTERS / n).min(CLUSTERS - 1);
        let mut v = vec![0.0f64; DIMS];
        for (j, x) in v.iter_mut().enumerate() {
            *x = 0.45 * centroids[c][j] + (rng.next() - 0.5) * NOISE;
        }
        normalize_f64(&mut v);
        data.push((i as u64 + 1, v.into_iter().map(|x| x as f32).collect()));
    }

    let mut qrng = Mulberry32::new(QUERY_SEED);
    let mut queries = Vec::with_capacity(CLUSTERS);
    for c in 0..CLUSTERS {
        let start = (c * n) as f64 / CLUSTERS as f64;
        let end = ((c + 1) * n) as f64 / CLUSTERS as f64;
        if end <= start {
            continue;
        }
        let base = (start + (end - start) / 2.0).floor() as usize;
        let mut q: Vec<f64> = data[base].1.iter().map(|x| *x as f64).collect();
        for x in q.iter_mut() {
            *x += (qrng.next() - 0.5) * 0.05;
        }
        normalize_f64(&mut q);
        queries.push(q.into_iter().map(|x| x as f32).collect());
    }
    (data, queries)
}

/// 容差真值：与第 k 名分差 ≤ TIE_EPS 的所有 id（与 sm 的判定一致）。
/// EN: Tolerant truth: all ids within TIE_EPS of the k-th score (sm's rule).
fn tolerant_truth(data: &Corpus, query: &[f32]) -> Vec<u64> {
    let mut scored: Vec<(u64, f64)> = data
        .iter()
        .map(|(id, v)| {
            let s = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            (*id, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let kth = scored[K.min(scored.len()) - 1].1;
    scored
        .into_iter()
        .take_while(|(_, s)| *s >= kth - TIE_EPS)
        .map(|(id, _)| id)
        .collect()
}

/// 指定层的可达节点数（只统计持有该层的节点）。
/// EN: Reachable node count at a layer (counting only nodes that own it).
fn layer_reachable(index: &HotBufferIndex, layer: usize) -> usize {
    let Some(ep) = index.entry_point() else {
        return 0;
    };
    let mut seen = HashSet::new();
    seen.insert(ep);
    let mut q = VecDeque::from([ep]);
    while let Some(cur) = q.pop_front() {
        for n in index.neighbors_at_layer_of(cur, layer) {
            if index.node_level(n).map(|l| l >= layer).unwrap_or(false) && seen.insert(n) {
                q.push_back(n);
            }
        }
    }
    seen.into_iter()
        .filter(|id| index.node_level(*id).map(|l| l >= layer).unwrap_or(false))
        .count()
}

/// 指定层的节点总数。
/// EN: Total node count owning a layer.
fn layer_nodes(index: &HotBufferIndex, layer: usize, ids: &[u64]) -> usize {
    ids.iter()
        .filter(|id| index.node_level(**id).map(|l| l >= layer).unwrap_or(false))
        .count()
}

/// 按服务端 collection 的真实默认构建参数与 ~300 行/tick 的 worker 节奏建图，
/// 并逐 tick 校验「legacy 分支 + 无丢行 + 全量排空」。
/// EN: Build with the collection backend's real defaults and ~300-row worker
/// ticks, validating per tick that the legacy branch ran, no rows were dropped,
/// and the queue fully drained.
async fn build_legacy_index(data: &Corpus) -> HotBufferIndex {
    let mut index = HotBufferIndex::new_with_metric_and_config(
        DIMS,
        DistanceMetric::Cosine,
        &VectorConfig::default(),
        &HnswConfig::default(),
    );
    index.set_hnsw_m(16).expect("set m");
    index.set_hnsw_ef_construction(200).expect("set efc");
    index.set_hnsw_ef_search(128).expect("set ef_search");
    let build_cfg = BuildConfig::default();
    let max_build_batch = build_cfg.max_build_batch;
    index.set_build_config(build_cfg);

    let mut flushed_total = 0usize;
    let mut rows_since_drain = 0usize;
    let mut saw_serial = false;
    for (i, chunk) in data.chunks(100).enumerate() {
        rows_since_drain += chunk.len();
        index.insert_batch(chunk.to_vec()).await.expect("insert");
        if (i + 1) % DRAIN_EVERY == 0 {
            let report = index.run_async_builder_once().await;
            // 与服务端一致：legacy 开关下绝不能走 plan/apply，否则门禁会
            // 静默测成默认路径（默认路径召回本就 1.0000，测错也全绿）。
            // EN: Match the server: under the legacy switch plan/apply must
            // never run, or the gate silently measures the default path
            // (which recall 1.0000 anyway and would stay green).
            assert!(
                !report.plan_apply,
                "HOUYI_HNSW_PLAN_APPLY=0 must not take the plan/apply path"
            );
            assert!(report.flushed > 0, "a non-empty drain must flush rows");
            assert!(
                rows_since_drain <= max_build_batch,
                "one tick accumulated {rows_since_drain} rows, above the \
                 {max_build_batch}-row cap; adjust the tick or the assertion"
            );
            // 本 tick 的全部待构图行必须一次排空（无墓碑/无拒绝），否则
            // 静默丢行会让可达率断言失去意义（分母和分子一起缩小）。
            // EN: A tick must flush all rows accumulated for it (no
            // tombstones/rejections); silently dropped rows would shrink
            // both numerator and denominator and make the reachability
            // assertion vacuous.
            assert_eq!(
                report.flushed, rows_since_drain,
                "tick drain must flush exactly the rows accumulated for it"
            );
            if report.plan_mode == Some(HnswBatchPlanMode::Serial) {
                saw_serial = true;
            }
            flushed_total += report.flushed;
            rows_since_drain = 0;
        }
    }
    while index.pending_build_count().await > 0 {
        let report = index.run_async_builder_once().await;
        // 尾批 drain 必须推进，否则 pending 永不归零、门禁挂到超时；
        // 同时校验它同样没走 plan/apply。
        // EN: The tail drain must make progress, otherwise pending never
        // reaches zero and the gate hangs until timeout; also verify it does
        // not take the plan/apply path.
        assert!(report.flushed > 0, "tail drain must make progress");
        assert!(
            !report.plan_apply,
            "tail drain must stay on the legacy path"
        );
        if report.plan_mode == Some(HnswBatchPlanMode::Serial) {
            saw_serial = true;
        }
        flushed_total += report.flushed;
    }
    assert!(
        saw_serial,
        "no batch reported the legacy serial plan mode; the gate measured the wrong path"
    );
    assert_eq!(
        flushed_total,
        data.len(),
        "every row must be flushed exactly once (got {flushed_total})"
    );
    // 显式全层兜底（构建期自动兜底可能已覆盖；这里钉住兜底能力本身）。
    // EN: Explicit all-layer backstop (the build-time sweep may already have
    // covered it; this pins the backstop capability itself).
    index.ensure_connectivity().await;
    index
}

#[test]
#[ignore = "50k x 512-d production-shaped legacy gate; run via scripts/clustered_recall_50k_gate.sh (release)"]
fn legacy_prod_shape_batches_keep_all_layer_recall_50k() {
    // legacy 构图 + 分组补救开启；UPPER_EF 不设、自适应显式开启，走进程内
    // 自适应缺省（与生产一致）；ADAPTIVE 也是进程级 OnceLock，外部残留的 0
    // 会让本用例静默改测回滚束宽，必须钉死。
    // EN: Legacy build with group repair on; UPPER_EF unset and adaptation
    // explicitly enabled to exercise the in-process adaptive default (matching
    // production). ADAPTIVE is also a process-wide OnceLock, so a leaked 0
    // would silently make this case measure the rollback beam; pin it.
    std::env::set_var("HOUYI_HNSW_PLAN_APPLY", "0");
    std::env::set_var("HOUYI_HNSW_GROUP_REPAIR", "4,16");
    std::env::remove_var("HOUYI_HNSW_UPPER_EF");
    std::env::set_var("HOUYI_HNSW_UPPER_EF_ADAPTIVE", "1");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build runtime");
    rt.block_on(async {
        let (data, queries) = build_corpus(N);
        let truth: Vec<Vec<u64>> = queries.iter().map(|q| tolerant_truth(&data, q)).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

        let index = build_legacy_index(&data).await;

        let max_layer = index
            .entry_point()
            .and_then(|ep| index.node_level(ep))
            .unwrap_or(0);
        for layer in 0..=max_layer {
            let total = layer_nodes(&index, layer, &ids);
            let reachable = layer_reachable(&index, layer);
            assert_eq!(
                reachable, total,
                "level-{layer} reachability must be 100% after the all-layer sweep \
                 (got {reachable}/{total})"
            );
        }

        let mut below_floor = Vec::new();
        let mut below_queries = HashSet::new();
        for ef in [16usize, 64, 256, 512] {
            // override 只能抬高 `ef_search`、不能降低；先用 setter 把本轮
            // 底层束宽真正钉到 ef，否则 ef=16/64 会静默复用 128（OCR #8/#10）。
            // EN: The override can only raise `ef_search`, never lower it; pin
            // the L0 beam for this round via the setter, otherwise ef=16/64
            // silently replay the 128-wide beam (OCR #8/#10).
            index.set_hnsw_ef_search(ef).expect("set ef_search");
            let mut sum = 0.0f64;
            for (qi, (q, t)) in queries.iter().zip(truth.iter()).enumerate() {
                let got = index.search_scored_with_ef_override(q, K, None, None).await;
                let got_ids: Vec<u64> = got.into_iter().map(|(id, _)| id).collect();
                let hits = got_ids.iter().take(K).filter(|id| t.contains(id)).count();
                let recall = hits as f64 / K as f64;
                sum += recall;
                // q45/q62 是本次缺陷的回归锚点：必须回到真值簇（recall ≥0.9）。
                // EN: q45/q62 are the regression anchors for this defect: they
                // must return to the truth cluster (recall >= 0.9).
                if qi == 45 || qi == 62 {
                    assert!(
                        recall >= 0.9,
                        "ef={ef}: q{qi} recall={recall} — upper-layer navigation regressed"
                    );
                }
                if recall < 0.9 {
                    below_queries.insert(qi);
                    below_floor.push((ef, qi, recall));
                }
            }
            let mean = sum / queries.len() as f64;
            assert!(
                mean >= 0.99,
                "ef={ef}: mean tolerant recall {mean:.4} < 0.99; below_floor={below_floor:?}"
            );
        }
        // 唯一允许低于单查询下限的是 q30 的 TIE_EPS 边界噪声（185 个并列近邻，
        // 返回分数与第 k 名差 <1e-3）；任何其它查询低于下限即回归（每个 ef 都会
        // 记录，因此按「涉及的查询 id 集合」判定，而不是按记录条数）。
        // EN: The only permitted below-floor query is q30's TIE_EPS boundary
        // noise (185 tied neighbours whose scores are within 1e-3 of k-th);
        // any other query below the floor is a regression (every ef records a
        // hit, so judge by the set of query ids, not the record count).
        assert!(
            below_queries.len() <= 1 && below_queries.iter().all(|qi| *qi == 30),
            "unexpected below-floor queries: {below_floor:?}"
        );
    });
}

/// 大图导航门禁：250k×512 / 64 簇（layer-1 约 15.6k 节点）。断言两层结果：
/// (1) ef=16…4096 的所有返回 id 必须仍落在查询自己的簇（不发生跨簇错落）；
/// (2) ef=8192 时均值召回 ≥0.99 且无查询低于下限（图结构完整、增加束宽可恢复）。
/// 250k 实测：低 ef 的 q8 召回可能为 0，但返回 id 始终在簇 8；ef=8192 恢复
/// 1.0000。这说明大图瓶颈是 L0 束宽 vs 簇规模（3906 点），不是上层导航错簇。
/// EN: Large-graph gate at 250k x 512 / 64 clusters (~15.6k layer-1 nodes).
/// Asserts two layers: (1) at ef=16..4096 every returned id must stay in the
/// query's own cluster (no cross-cluster landings); (2) at ef=8192 mean recall
/// ≥ 0.99 with zero below-floor queries (the graph is structurally complete and
/// wider beams recover it). Measured at 250k: low-ef q8 may have recall 0, but
/// its returned ids are always in cluster 8; ef=8192 restores 1.0000 — the
/// bottleneck is L0 beam vs cluster size (3906 points), not upper navigation.
#[test]
#[ignore = "250k x 512-d large-graph navigation gate; release only"]
fn large_graph_upper_navigation_keeps_cluster_at_quarter_million() {
    std::env::set_var("HOUYI_HNSW_PLAN_APPLY", "0");
    std::env::set_var("HOUYI_HNSW_GROUP_REPAIR", "4,16");
    std::env::remove_var("HOUYI_HNSW_UPPER_EF");
    std::env::set_var("HOUYI_HNSW_UPPER_EF_ADAPTIVE", "1");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build runtime");
    rt.block_on(async {
        let (data, queries) = build_corpus(LARGE_N);
        let truth: Vec<Vec<u64>> = queries.iter().map(|q| tolerant_truth(&data, q)).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
        let index = build_legacy_index(&data).await;

        let max_layer = index
            .entry_point()
            .and_then(|ep| index.node_level(ep))
            .unwrap_or(0);
        for layer in 0..=max_layer {
            let total = layer_nodes(&index, layer, &ids);
            let reachable = layer_reachable(&index, layer);
            eprintln!("large probe layer={layer} reachable={reachable}/{total}");
            assert_eq!(reachable, total, "all layers must be reachable");
        }

        let cluster_of = |id: u64| (((id as usize - 1) * CLUSTERS) / LARGE_N).min(CLUSTERS - 1);
        // 低/中 ef：不得跨簇错落（召回可以低，但返回必须仍在查询簇内）。
        // EN: Low/mid ef: no cross-cluster landings (recall may be low, but every
        // returned id must stay in the query's own cluster).
        for ef in [16usize, 64, 256, 512, 1024, 2048, 4096] {
            index.set_hnsw_ef_search(ef).expect("set ef_search");
            let mut sum = 0.0f64;
            for (qi, (q, t)) in queries.iter().zip(truth.iter()).enumerate() {
                assert!(
                    t.iter().all(|id| cluster_of(*id) == qi),
                    "truth for q{qi} must stay in cluster {qi}"
                );
                let got = index.search_scored_with_ef_override(q, K, None, None).await;
                let got_ids: Vec<u64> = got.into_iter().map(|(id, _)| id).collect();
                assert!(!got_ids.is_empty(), "ef={ef} q{qi} returned nothing");
                for id in &got_ids {
                    assert_eq!(
                        cluster_of(*id),
                        qi,
                        "ef={ef}: q{qi} returned id {id} from the wrong cluster"
                    );
                }
                let hits = got_ids.iter().take(K).filter(|id| t.contains(id)).count();
                sum += hits as f64 / K as f64;
            }
            eprintln!("large gate ef={ef} mean={:.4}", sum / queries.len() as f64);
        }
        // 高 ef：图结构完整时束宽足够即可恢复全部召回。
        // EN: High ef: a structurally complete graph recovers full recall once
        // the beam is wide enough.
        index.set_hnsw_ef_search(8192).expect("set ef_search");
        let mut below = 0usize;
        let mut sum = 0.0f64;
        for (qi, (q, t)) in queries.iter().zip(truth.iter()).enumerate() {
            let got = index.search_scored_with_ef_override(q, K, None, None).await;
            let got_ids: Vec<u64> = got.into_iter().map(|(id, _)| id).collect();
            let hits = got_ids.iter().take(K).filter(|id| t.contains(id)).count();
            let recall = hits as f64 / K as f64;
            sum += recall;
            if recall < 0.9 {
                below += 1;
                eprintln!("large gate ef=8192 fail qi={qi} recall={recall}");
            }
        }
        let mean = sum / queries.len() as f64;
        assert!(
            mean >= 0.99 && below == 0,
            "ef=8192: mean={mean:.4} below_floor={below}; graph must be structurally complete"
        );
    });
}

/// 上层束宽 50k 延迟 A/B 探针（不作为硬门禁，机器相关）：同一语料/构建下
/// 分别以 ADAPTIVE=0（固定 env 束宽）与 ADAPTIVE=1（按层规模自适应）运行，
/// 打印 ef=256 的单查询平均延迟；由 `scripts/upper_beam_latency_ab.sh` 分别
/// 起两个进程采集，避免 OnceLock 在同进程内无法切换。
/// EN: 50k upper-beam latency A/B probe (not a hard gate; machine-dependent):
/// with the same corpus/build, run ADAPTIVE=0 (fixed env width) and ADAPTIVE=1
/// (size-adaptive), printing average ef=256 latency per query.
/// `scripts/upper_beam_latency_ab.sh` launches two separate processes because
/// the OnceLock cannot be switched inside one process.
#[test]
#[ignore = "50k latency A/B probe; release only"]
fn upper_beam_latency_ab() {
    let adaptive = std::env::var("TEST_ADAPTIVE").unwrap_or_else(|_| "1".to_string());
    // env_bool 对未知值静默回落默认 true；这里必须拒绝，否则两臂会跑成同一配置。
    // EN: env_bool silently falls back to the default true for unknown values;
    // reject them here, or both A/B arms run the same configuration.
    assert!(
        matches!(adaptive.as_str(), "0" | "1"),
        "TEST_ADAPTIVE must be 0 or 1, got {adaptive:?}"
    );
    std::env::set_var("HOUYI_HNSW_PLAN_APPLY", "0");
    std::env::set_var("HOUYI_HNSW_GROUP_REPAIR", "4,16");
    std::env::remove_var("HOUYI_HNSW_UPPER_EF");
    std::env::set_var("HOUYI_HNSW_UPPER_EF_ADAPTIVE", &adaptive);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build runtime");
    rt.block_on(async {
        let (data, queries) = build_corpus(N);
        let index = build_legacy_index(&data).await;
        index.set_hnsw_ef_search(256).expect("set ef_search");
        // 预热 16 条，避免把首次搜索的冷路径算入均值。
        // EN: Warm up with 16 queries so the first-search cold path is not
        // counted in the average.
        for q in queries.iter().take(16) {
            let _ = index.search_scored_with_ef_override(q, K, None, None).await;
        }
        // 配置指纹：固定 64 与自适应（50k 下约 391）在 sm 语料上应命中
        // 不同簇（固定束宽下 q45 落入簇 41，自适应回到簇 45）。非空 + 打印
        // 簇号让两臂即使数值接近也能自证跑的是预期配置。
        // EN: Config fingerprint: fixed 64 and adaptive (~391 at 50k) should hit
        // different clusters on the sm corpus (fixed lands q45 in cluster 41,
        // adaptive returns cluster 45). Non-empty results plus the printed cluster
        // let each arm prove it ran the intended config even when timings are close.
        let probe = index
            .search_scored_with_ef_override(&queries[45], K, None, None)
            .await;
        assert!(!probe.is_empty(), "q45 probe must return results");
        let top_id = probe[0].0;
        let top_cluster = (((top_id as usize - 1) * CLUSTERS) / N).min(CLUSTERS - 1);
        // 固定束宽的已知回归形态是 q45 找不到真值簇 45；自适应应恢复它。
        // 不硬编码窄束具体落入哪个错簇（41 只是当前的偶发现象），只钉住
        // 「窄束未恢复真值、自适应恢复真值」这一 A/B 前提。
        // EN: The narrow beam's known regression shape is failing to recover the
        // truth cluster 45; adaptation should recover it. Do not hardcode which
        // wrong cluster (41 is only the current incidental symptom); pin the A/B
        // premise "narrow beam misses the truth, adaptive finds it".
        let truth_cluster = 45usize;
        if adaptive == "0" {
            assert_ne!(
                top_cluster, truth_cluster,
                "adaptive=0 unexpectedly recovered q45 truth cluster {truth_cluster} \
                 (top_id={top_id}); the narrow-beam A/B anchor changed"
            );
        } else {
            assert_eq!(
                top_cluster, truth_cluster,
                "adaptive=1 must recover q45 truth cluster {truth_cluster} (top_id={top_id})"
            );
        }
        let rounds = 3usize;
        let t0 = std::time::Instant::now();
        for _ in 0..rounds {
            for q in &queries {
                let _ = index.search_scored_with_ef_override(q, K, None, None).await;
            }
        }
        let per_query_ms = t0.elapsed().as_secs_f64() * 1000.0 / (rounds * queries.len()) as f64;
        println!(
            "latency_ab adaptive={adaptive} avg_ms={per_query_ms:.3} \
             q45_top_id={top_id} q45_top_cluster={top_cluster}"
        );
    });
}
