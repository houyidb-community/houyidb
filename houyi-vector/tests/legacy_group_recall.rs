//! legacy 分组构图路径的召回回归门禁（`HOUYI_HNSW_PLAN_APPLY=0`）。
//!
//! 与默认 plan/apply 同缺陷：单请求整批（≥ `parallel_min_batch` 且
//! graph+count ≥ `parallel_min_graph`）时，每组只对组前快照规划，组内兄弟互不
//! 可见。本文件是独立测试二进制：构图路径开关、补救参数与查询侧上层束宽都是
//! 进程级 OnceLock，必须在本进程首次构图/检索前（且在任何 tokio 运行时启动
//! 前）设置。四条都显式钉死：`HOUYI_HNSW_PLAN_APPLY=0`（legacy 路径本身）、
//! `HOUYI_HNSW_GROUP_REPAIR`（4653 用例 `4,16` 开启补救；50k 串行用例 `off`，
//! 串行路径不经补救，钉死防止 shell 残留改写门禁语义）、
//! `HOUYI_HNSW_UPPER_EF=64`（查询侧上层束宽下限，实际宽度按层规模自适应；
//! 钉住防止外部残留更高值改测查询而非构图）与 `HOUYI_HNSW_UPPER_EF_ADAPTIVE=1`。两个用例钉的 GROUP_REPAIR 不同，且共享同一批 OnceLock，必须分进程
//! 运行（默认过滤与 `--ignored` 互斥），不要用 `--include-ignored` 同进程并跑。
//! 默认只跑 4653 用例；50k 用例（`#[ignore]`）以 500 条/批走批大小低于并行
//! 门槛时的逐点串行路径，由 `scripts/clustered_recall_50k_gate.sh` 在 release
//! 下运行。
//!
//! Regression gate for the legacy grouped build path (`HOUYI_HNSW_PLAN_APPLY=0`).
//! Same defect: with a single large request (>= `parallel_min_batch` and
//! graph+count >= `parallel_min_graph`) each group plans against the pre-group
//! snapshot, so siblings stay invisible to each other. This is a separate test
//! binary on purpose — the build-path switch, repair params, and query-side
//! upper beam are process-wide OnceLocks, so env vars must be set before this
//! process builds/searches anything (and before any tokio runtime starts). All
//! four are pinned explicitly: `HOUYI_HNSW_PLAN_APPLY=0` (the legacy path
//! itself), `HOUYI_HNSW_GROUP_REPAIR` (the 4653 case uses `4,16` with repair
//! on; the 50k serial case uses `off` because the serial path never invokes
//! repair, pinned so a shell leftover cannot change gate semantics),
//! `HOUYI_HNSW_UPPER_EF=64` (floor for the size-adaptive query-side upper
//! beam; pinned so a stale higher value cannot change the query side while
//! the build stays the same), and `HOUYI_HNSW_UPPER_EF_ADAPTIVE=1` (keeps the
//! size-adaptive beam on). The
//! two cases pin different GROUP_REPAIR values and share those OnceLocks, so
//! they must run in separate processes (default filter vs `--ignored`); do not
//! use `--include-ignored` to run both in one process. By default only the
//! 4653 case runs. The 50k case (`#[ignore]`) uses 500-row requests, below the
//! parallel threshold, exercising the per-point serial path; it is run in
//! release by `scripts/clustered_recall_50k_gate.sh`.

#[path = "common/clustered_recall.rs"]
mod clustered;

use houyi_core::config::{BuildConfig, HnswConfig, VectorConfig};
use houyi_vector::{DistanceMetric, HnswBatchPlanMode, HotBufferIndex, VectorIndex};

#[test]
fn legacy_grouped_build_keeps_clustered_recall() {
    // 四条进程级 OnceLock 全显式钉死：legacy 构图路径（=0）、补救参数（本用例
    // 需要补救开启）、查询侧上层束宽下限与自适应开关；shell 残留任一值都会让
    // 门禁静默改测别的行为。必须在首次构图（乃至运行时启动）之前设置：
    // OnceLock 只读一次 env。
    // EN: Pin all four process-wide OnceLocks explicitly: the legacy build path
    // (=0), repair params (this case needs repair on), the query-side upper
    // beam floor, and the adaptation switch; any stale shell value would
    // silently change what this gate measures. Must be set before the first
    // build (and before the runtime starts) — the OnceLock reads env once.
    std::env::set_var("HOUYI_HNSW_PLAN_APPLY", "0");
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
        // 单请求整批 + 满足 legacy 并行门槛（count >= 512、graph+count >= 4096）。
        // EN: One single big request that meets the legacy parallel gates
        // (count >= 512, graph+count >= 4096).
        let cfg = BuildConfig {
            max_build_batch: 8192,
            parallel_batch_build: true,
            parallel_min_batch: 512,
            parallel_min_graph: 4096,
            parallel_chunk_size: 256,
            parallel_plan_group_chunks: 4,
            ..BuildConfig::default()
        };
        index.set_build_config(cfg);

        index
            .insert_batch(corpus.data.clone())
            .await
            .expect("batch insert should succeed");
        let mut flushed_total = 0usize;
        let mut parallel_batches = 0usize;
        while index.pending_build_count().await > 0 {
            let report = index.run_async_builder_once().await;
            assert!(
                report.flushed > 0,
                "a drain call with pending rows must flush something"
            );
            // legacy 开关下绝不能出现 plan/apply；整批应派发到并行分组分支，
            // 否则门禁测到的只是串行构图（串行构图单独也能过 95%）。
            // EN: Under the legacy switch plan/apply must never run; the big
            // batch should dispatch to the parallel grouped branch, otherwise
            // the gate only measures the serial build (which clears 95% alone).
            assert!(
                !report.plan_apply,
                "HOUYI_HNSW_PLAN_APPLY=0 must not take the plan/apply path"
            );
            if report.plan_mode == Some(HnswBatchPlanMode::Parallel) {
                parallel_batches += 1;
            }
            flushed_total += report.flushed;
        }
        assert_eq!(
            flushed_total,
            corpus.data.len(),
            "every row must be flushed exactly once"
        );
        assert!(
            parallel_batches > 0,
            "no batch took the parallel grouped path; the gate would be measuring the serial build"
        );

        clustered::assert_clustered_recall(&index, &corpus, "legacy grouped build").await;
    });
}

/// 50k 串行构图回归：`HOUYI_HNSW_PLAN_APPLY=0` + 500 条/批时批大小低于
/// `parallel_min_batch`（512），走逐点串行插入——sm 报告中「加大 ef 几乎不
/// 改善」的 legacy 复现路径。四条 env 在首次构图/检索前显式钉死；补救参数取
/// `off`（串行路径不经提交后孤点补救，显式关闭以免 shell 残留 `4,16` 混入）。
/// 并行门槛在 `BuildConfig` 里显式钉死而不依赖 default：默认值一旦变更，500
/// 条/批可能被并行分组接管，门禁会静默改测别的构图路径。修复前上层下降束宽
/// 16 时 #38 落入错误簇（错误簇入口相似度 0.881 vs 真簇 0.0495），增大 L0 ef
/// 无法纠正；束宽 64 后两条路径均 0/64 错落。同分布 50k 串行构图的 L0 可达率
/// 实测 96.2%，未套用提交后孤点补救（补救只作用于分组提交路径）。
/// EN: 50k serial-build regression: with `HOUYI_HNSW_PLAN_APPLY=0` and 500-row
/// requests the batch stays below `parallel_min_batch` (512), so points are
/// inserted one by one — the legacy path from the sm report where raising ef
/// barely helped. All four env vars are pinned explicitly before the first
/// build/search; repair params are `off` (the serial path never invokes the
/// post-apply orphan repair, pinned off so a shell leftover `4,16` cannot
/// leak in). The parallel gates are pinned in `BuildConfig` rather than
/// inherited from defaults: if a default changed, 500-row batches could be
/// taken over by the parallel grouped path and the gate would silently
/// measure a different build. Before the fix, upper-layer width 16 left query
/// #38 in the wrong cluster (wrong entry similarity 0.881 vs 0.0495 in the
/// true cluster), and a larger L0 ef could not correct it; width 64 gives 0/64
/// wrong landings on both paths. On this 50k serial build the measured L0
/// reachability is 96.2%; the post-apply orphan repair does not apply (it
/// only covers grouped commit paths).
#[test]
#[ignore = "50k corpus takes minutes in debug; run via scripts/clustered_recall_50k_gate.sh (release)"]
fn legacy_serial_build_keeps_clustered_recall_50k() {
    // 同上：四条 OnceLock 都在首次构图/检索前钉死（本用例补救必须关闭）。
    // EN: Same as above: all four OnceLocks are pinned before the first
    // build/search (repair must be off for this case).
    std::env::set_var("HOUYI_HNSW_PLAN_APPLY", "0");
    std::env::set_var("HOUYI_HNSW_GROUP_REPAIR", "off");
    std::env::set_var("HOUYI_HNSW_UPPER_EF", "64");
    std::env::set_var("HOUYI_HNSW_UPPER_EF_ADAPTIVE", "1");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build runtime");
    rt.block_on(async {
        let corpus = clustered::ClusteredCorpus::new_with(clustered::DATA_N_LARGE);
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
        // 门槛显式钉死（不依赖 default）：500 条/批必须始终低于
        // `parallel_min_batch` 且 graph+count 永远达不到 `parallel_min_graph`，
        // 否则并行分组路径会接管，门禁静默改测别的构图方式。
        // EN: Pin the gates explicitly (do not rely on defaults): 500-row
        // requests must stay below `parallel_min_batch` and graph+count must
        // never reach `parallel_min_graph`, otherwise the parallel grouped
        // path takes over and the gate silently measures a different build.
        let cfg = BuildConfig {
            max_build_batch: 8192,
            parallel_batch_build: true,
            parallel_min_batch: 512,
            parallel_min_graph: 4096,
            parallel_chunk_size: 256,
            parallel_plan_group_chunks: 4,
            ..BuildConfig::default()
        };
        index.set_build_config(cfg);
        const REQUEST_BATCH: usize = 500;
        let mut flushed_total = 0usize;
        for rows in corpus.data.chunks(REQUEST_BATCH) {
            index
                .insert_batch(rows.to_vec())
                .await
                .expect("batch insert should succeed");
            while index.pending_build_count().await > 0 {
                let report = index.run_async_builder_once().await;
                assert!(
                    report.flushed > 0,
                    "a drain call with pending rows must flush something"
                );
                // 500 条/批低于并行门槛：必须逐点串行，且 legacy 开关下不得
                // 出现 plan/apply。任一断言失守都说明门禁改测了别的路径。
                // EN: 500-row batches stay below the parallel gates, so the
                // build must stay serial; and under the legacy switch
                // plan/apply must never run. Either failure means the gate is
                // measuring a different path.
                assert!(
                    !report.plan_apply,
                    "HOUYI_HNSW_PLAN_APPLY=0 must not take the plan/apply path"
                );
                assert_eq!(
                    report.plan_mode,
                    Some(HnswBatchPlanMode::Serial),
                    "500-row batches must take the serial path"
                );
                flushed_total += report.flushed;
            }
        }
        assert_eq!(
            flushed_total,
            corpus.data.len(),
            "every row must be flushed exactly once"
        );
        clustered::assert_clustered_recall(&index, &corpus, "legacy 50k serial build").await;
    });
}
