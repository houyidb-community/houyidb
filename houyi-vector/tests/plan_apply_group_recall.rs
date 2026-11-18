//! 默认 plan/apply 分组构图的召回回归门禁。
//!
//! 历史缺陷：每组只对「组前图快照」做近邻规划，组内兄弟节点互相不可见；成簇
//! 数据下整批兄弟互为最近邻，而对称剪枝会把旧节点的回边剪掉，新节点因此成为
//! 入口点不可达的孤岛——recall 与 ef_search 无关地崩塌（实测 recall@10 6%、
//! level-0 可达 2229/4653）。本门禁用同一压力分布钉住该缺陷：修复前失败，
//! 修复后（组宽随图规模收敛 + 提交后孤点补救）恢复 ≥95% 召回与近似全图可达。
//!
//! 门禁显式钉住 `HOUYI_HNSW_PLAN_APPLY=1`、`HOUYI_HNSW_GROUP_REPAIR=4,16` 与
//! `HOUYI_HNSW_UPPER_EF=64`（查询侧上层束宽下限，实际宽度按层规模自适应）与
//! `HOUYI_HNSW_UPPER_EF_ADAPTIVE=1`：四者都是进程级 OnceLock，外部 shell 遗留的 `=0`/`off`/更高束宽会让本门禁
//! 静默改测别的构图/补救/查询行为。同一测试二进制
//! 内的用例共享这些 OnceLock，两个用例必须分进程运行（默认过滤与
//! `--ignored` 互斥），不要用 `--include-ignored` 同进程并行。
//!
//! 50k 大簇用例（`#[ignore]`）：4653 规模下自适应组宽已收缩到 ~18，压不出
//! 50k 的整簇孤岛；50k 下组宽 ~195 仍小于簇规模（~781），正是生产上报的
//! 「整簇查询返回错误簇」规模。debug 下 50k 的暴力真值与构图需要数分钟，
//! 因此默认不跑，由 `scripts/clustered_recall_50k_gate.sh` 以 release 运行。
//!
//! Regression gate for the default plan/apply grouped build. Each group plans
//! against the pre-group graph snapshot, so siblings inside a group are
//! invisible to each other. On clustered data those siblings are each other's
//! nearest neighbors, symmetric pruning drops the old nodes' back-edges, and
//! the new nodes become islands unreachable from the entry point — recall
//! collapses independently of ef_search (measured recall@10 6%, level-0
//! reachability 2229/4653). This gate pins that failure shape: it fails before
//! the fix and restores >=95% recall with near-full reachability after the
//! adaptive group-width fix plus post-apply orphan repair.
//!
//! The gate explicitly pins `HOUYI_HNSW_PLAN_APPLY=1`,
//! `HOUYI_HNSW_GROUP_REPAIR=4,16`, `HOUYI_HNSW_UPPER_EF=64` (a floor for
//! the size-adaptive upper beam), and `HOUYI_HNSW_UPPER_EF_ADAPTIVE=1`: all
//! four are process-wide OnceLocks, so a
//! stale `=0`/`off`/higher beam in the shell would silently make this gate
//! exercise different build/repair/query behavior. The cases in this binary share those OnceLocks and therefore
//! must run in separate processes (default filter vs `--ignored`); do not
//! use `--include-ignored` to run both in one process.
//!
//! The 50k large-cluster case (`#[ignore]`): at 4653 the adaptive group width
//! already shrinks to ~18, so the 50k whole-cluster islands cannot be pressed
//! out; at 50k the width (~195) still sits below the cluster size (~781) — the
//! exact scale of the production report ("whole-cluster queries reach the wrong
//! cluster"). The 50k brute-force truth and build take minutes in debug, so it
//! is skipped by default and run in release by
//! `scripts/clustered_recall_50k_gate.sh`.

#[path = "common/clustered_recall.rs"]
mod clustered;

use houyi_core::config::{BuildConfig, HnswConfig, VectorConfig};
use houyi_vector::{DistanceMetric, HnswBatchPlanMode, HotBufferIndex, VectorIndex};

/// 请求批大小与知识库评测一致（500 条/批，每批排空后继续）。
/// EN: Request batch size matches the knowledge-base benchmark (500 rows per
/// batch, drained before the next batch).
const REQUEST_BATCH: usize = 500;

/// 用门禁配置构建索引：m=16 / ef_construction=200 与知识库评测一致，plan/
/// apply 组宽取仓库默认（1024 上限 + 图规模/256 自适应）。
///
/// 排空时同时断言真实构图路径：所有行恰好 flush 一次、至少有一批由
/// `plan_apply` 接管且报告 `Parallel`。否则播种门槛/upsert 判定一旦放宽，
/// 门禁可能悄悄退回旧路径而召回数字仍然达标（串行构图单独也能过 95%）。
/// EN: Build the index with the gate configuration: m=16 / ef_construction=200
/// match the knowledge-base benchmark; the plan/apply group width uses the repo
/// defaults (1024 cap, adaptive graph-size/256).
///
/// Draining also asserts the actual build path: every row flushes exactly
/// once and at least one batch is taken by `plan_apply` and reports
/// `Parallel`. Otherwise, if the seed gate or the upsert check ever widened,
/// the gate could silently fall back to the legacy path while recall still
/// passes (the serial build alone clears 95%).
async fn build_gate_index(rows: &[(u64, Vec<f32>)]) -> HotBufferIndex {
    let mut index = HotBufferIndex::new_with_metric_and_config(
        clustered::DATA_DIMS,
        DistanceMetric::Cosine,
        &VectorConfig::default(),
        &HnswConfig::default(),
    );
    index.set_hnsw_m(16).expect("set m should succeed");
    index
        .set_hnsw_ef_construction(200)
        .expect("set ef_construction should succeed");
    let cfg = BuildConfig {
        parallel_batch_build: true,
        parallel_chunk_size: 256,
        parallel_plan_group_chunks: 4,
        ..BuildConfig::default()
    };
    index.set_build_config(cfg);

    let mut flushed_total = 0usize;
    let mut plan_apply_batches = 0usize;
    for chunk in rows.chunks(REQUEST_BATCH) {
        index
            .insert_batch(chunk.to_vec())
            .await
            .expect("batch insert should succeed");
        // 每批排空：批内兄弟不可见是本门禁的原始触发场景。
        // EN: Drain after every batch: intra-batch sibling invisibility is
        // the original trigger this gate pins.
        while index.pending_build_count().await > 0 {
            let report = index.run_async_builder_once().await;
            assert!(
                report.flushed > 0,
                "a drain call with pending rows must flush something"
            );
            flushed_total += report.flushed;
            if report.plan_apply {
                plan_apply_batches += 1;
                assert_eq!(
                    report.plan_mode,
                    Some(HnswBatchPlanMode::Parallel),
                    "plan/apply batches must plan in parallel"
                );
            }
        }
    }
    assert_eq!(
        flushed_total,
        rows.len(),
        "every row must be flushed exactly once"
    );
    // 首批在 256 点播种门槛前只能串行；此后必须真实走 plan/apply。
    // EN: Only the first pre-seed batch may stay serial; afterwards the
    // plan/apply path must really run.
    assert!(
        plan_apply_batches > 0,
        "no batch took the plan/apply path; the gate would be measuring the legacy build"
    );
    index
}

#[test]
fn plan_apply_grouped_build_keeps_clustered_recall() {
    // 必须在首次构图/检索前设置：OnceLock 只读一次 env。四条都显式钉死
    // （构图路径、补救参数、查询侧上层束宽与自适应开关）；shell 残留任一
    // `=0`/`off` 都会让门禁静默改测别的行为。
    // EN: Must be set before the first build/search — the OnceLocks read env
    // once. All four knobs are pinned explicitly (build path, repair params,
    // query-side upper beam and adaptation switch); a stale `=0`/`off` in the
    // shell would silently make this gate measure different behavior.
    std::env::set_var("HOUYI_HNSW_PLAN_APPLY", "1");
    std::env::set_var("HOUYI_HNSW_GROUP_REPAIR", "4,16");
    std::env::set_var("HOUYI_HNSW_UPPER_EF", "64");
    std::env::set_var("HOUYI_HNSW_UPPER_EF_ADAPTIVE", "1");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build runtime");
    rt.block_on(async {
        let corpus = clustered::ClusteredCorpus::new();
        let index = build_gate_index(&corpus.data).await;
        clustered::assert_clustered_recall(&index, &corpus, "plan/apply grouped build").await;
    });
}

/// 50k 大簇回归：默认 plan/apply 路径在同一成簇分布 ×10.7 的语料上必须保持
/// ≥95% 可达与各 ef 的均值召回；修复前该规模只剩 359/50000 前向可达
/// （5003 个 L0 度为 0 的新点）。
/// EN: 50k large-cluster regression: the default plan/apply path must keep >=95%
/// reachability and mean recall at every ef on the same clustered distribution
/// scaled ×10.7. Before the fix only 359/50000 nodes were forward-reachable
/// (5003 new nodes with L0 degree 0).
#[test]
#[ignore = "50k corpus takes minutes in debug; run via scripts/clustered_recall_50k_gate.sh (release)"]
fn plan_apply_grouped_build_keeps_clustered_recall_50k() {
    // 同上：四条 env 都在首次构图/检索前钉死（补救参数必须显式开启）。
    // EN: Same as above: all four env vars are pinned before the first
    // build/search (the repair pass must be explicitly enabled).
    std::env::set_var("HOUYI_HNSW_PLAN_APPLY", "1");
    std::env::set_var("HOUYI_HNSW_GROUP_REPAIR", "4,16");
    std::env::set_var("HOUYI_HNSW_UPPER_EF", "64");
    std::env::set_var("HOUYI_HNSW_UPPER_EF_ADAPTIVE", "1");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build runtime");
    rt.block_on(async {
        let corpus = clustered::ClusteredCorpus::new_with(clustered::DATA_N_LARGE);
        let index = build_gate_index(&corpus.data).await;
        clustered::assert_clustered_recall(&index, &corpus, "plan/apply 50k grouped build").await;
    });
}
