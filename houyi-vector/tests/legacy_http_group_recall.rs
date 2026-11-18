//! HTTP 形态的 legacy 构图召回门禁：100 行/批、按 worker 阈值排空，
//! 覆盖官方 `legacy_group_recall` 未触及的「小请求 + 累积批次」路径。
//!
//! 背景：官方 legacy 50k 用例以 500 条/批（低于 `parallel_min_batch`=512）走
//! 逐点串行路径；真实 HTTP 写入常是 100 条/批，worker 在队列达阈值时一次
//! drain 300~数千条——可能进入 legacy 分组路径，且串行路径在成簇语料上仍会
//! 因对称剪枝丢掉唯一桥边，留下「整簇分量」（成员度数都 ≥min_degree、对外
//! 无桥）。本文件同时覆盖两种 drain 规模（300 串行 / 600+ 分组），断言
//! level-0 可达率 100% 与各 ef 的容差召回——连通性兜底（排空后 BFS +
//! 最近可达锚点补桥）必须把残余分量全部接回入口点。
//!
//! HTTP-shaped legacy gate: 100-row requests drained at the worker threshold,
//! covering the "small requests + accumulated batches" path the official
//! `legacy_group_recall` case never reaches. The official case uses 500-row
//! requests (below `parallel_min_batch`=512) and stays per-point serial; real
//! HTTP writes are usually 100 rows per request, and the worker drain of
//! 300..thousands can take the legacy grouped path — while the serial path on
//! clustered data can still lose its only bridge to symmetric pruning,
//! leaving whole-cluster components (every member at >= min_degree, no
//! external bridge). This file covers both drain sizes (300 serial / 600+
//! grouped) and asserts 100% level-0 reachability plus tolerant recall at
//! every ef — the post-drain connectivity backstop (BFS + nearest-reachable
//! anchor bridges) must reconnect every residual component.
//!
//! 进程级 OnceLock：`HOUYI_HNSW_PLAN_APPLY=0`、`HOUYI_HNSW_GROUP_REPAIR=4,16`、
//! `HOUYI_HNSW_UPPER_EF=64`（作为自适应束宽的下限，实际宽度由层规模决定）与
//! `HOUYI_HNSW_UPPER_EF_ADAPTIVE=1` 四条都显式钉死；本文件只有一个用例，不与其它测试二进制共享这些开关。由 `scripts/clustered_recall_50k_gate.sh` 在
//! release 下运行（debug 的 50k 暴力真值与构图需要数分钟）。
//! EN: Process-wide OnceLocks: all four knobs are pinned (UPPER_EF=64 is a
//! floor and ADAPTIVE=1 keeps the size-adaptive beam on); this file carries a
//! single test and shares no switches with other test binaries. Run in release
//! by `scripts/clustered_recall_50k_gate.sh` (the 50k brute-force truth plus
//! build take minutes in debug).

#[path = "common/clustered_recall.rs"]
mod clustered;

use houyi_core::config::{BuildConfig, HnswConfig, VectorConfig};
use houyi_vector::{DistanceMetric, HotBufferIndex, VectorIndex, MIN_CONNECTIVITY_SWEEP_ROWS};

/// HTTP 客户端请求批大小（与知识库/bench 一致）。
/// EN: HTTP request batch size (matches the knowledge-base bench).
const REQUEST_BATCH: usize = 100;
/// 单次 worker drain 前的请求数：3 → 300 行走串行路径（<512）。
/// EN: Requests per worker drain: 3 -> 300 rows take the serial path (<512).
const SERIAL_DRAIN_REQUESTS: usize = 3;
/// 6 → 600 行走 legacy 分组路径（≥512 且图+批 ≥4096，图播种后）。
/// EN: 6 -> 600 rows take the legacy grouped path (>=512 and graph+batch
/// >=4096 once seeded).
const GROUPED_DRAIN_REQUESTS: usize = 6;

/// 构建并返回 `(索引, flush 总数, 是否走过 Parallel 分组, 是否走过 Serial)`。
/// EN: Build and return `(index, flushed total, took Parallel grouped, took Serial)`.
async fn build_with_drain_pattern(
    data: &[(u64, Vec<f32>)],
    requests_per_drain: usize,
) -> (HotBufferIndex, usize, bool, bool) {
    let mut index = HotBufferIndex::new_with_metric_and_config(
        clustered::DATA_DIMS,
        DistanceMetric::Cosine,
        &VectorConfig::default(),
        &HnswConfig::default(),
    );
    index.set_hnsw_m(16).expect("set m");
    index.set_hnsw_ef_construction(200).expect("set efc");
    // 与服务端 HTTP 路径一致：单次 drain 上限 8192，并行批量构图开启。
    // EN: Match the server HTTP path: 8192 per-drain cap, parallel batch build.
    // 门槛显式钉死（不依赖 default）：门槛一旦漂移，300/600 两种 drain 会
    // 静默落到另一种构图路径，门禁仍绿却测错了对象。
    // EN: Pin the thresholds explicitly (never inherit defaults): if a default
    // drifts, the 300/600 drains silently land on the other build path and the
    // gate stays green while measuring the wrong thing.
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

    let mut flushed_total = 0usize;
    let mut saw_parallel = false;
    let mut saw_serial = false;
    let chunks: Vec<&[(u64, Vec<f32>)]> = data.chunks(REQUEST_BATCH).collect();
    let mut i = 0usize;
    while i < chunks.len() {
        let end = (i + requests_per_drain).min(chunks.len());
        for chunk in &chunks[i..end] {
            index
                .insert_batch(chunk.to_vec())
                .await
                .expect("batch insert");
        }
        i = end;
        // 一次 worker tick：单次 drain（队列超过 8192 时留到下一轮）。
        // EN: One worker tick: a single drain (leftovers wait for the next
        // tick when the queue exceeds 8192).
        let report = index.run_async_builder_once().await;
        flushed_total += report.flushed;
        match report.plan_mode {
            Some(houyi_vector::HnswBatchPlanMode::Parallel) => saw_parallel = true,
            Some(houyi_vector::HnswBatchPlanMode::Serial) => saw_serial = true,
            None => {}
        }
    }
    while index.pending_build_count().await > 0 {
        let report = index.run_async_builder_once().await;
        flushed_total += report.flushed;
        match report.plan_mode {
            Some(houyi_vector::HnswBatchPlanMode::Parallel) => saw_parallel = true,
            Some(houyi_vector::HnswBatchPlanMode::Serial) => saw_serial = true,
            None => {}
        }
    }
    // 尾批（如 200 行）低于累计阈值；与服务端 worker 的空闲 tick 等价，
    // 显式再扫一次保证门禁与写入节奏无关。先按实际语料推导尾批大小并断言
    // 计数语义：尾批 ≥256 或为 0 时应已被构建期兜底清零，唯一未扫的情形是
    // 0<尾批<256——这样语料/参数变化不会让断言变成假绿或误报。
    // EN: The tail batch (e.g. 200 rows) stays below the accumulation threshold;
    // mirror the server worker's idle tick with an explicit final sweep so the
    // gate does not depend on write pacing. Derive the actual tail size first:
    // a tail of >=256 or 0 must already have been swept (counter 0); only
    // 0<tail<256 may remain unswept — corpus/parameter changes then cannot turn
    // this assertion into a false green or a misleading failure.
    let tail = data.len() % (REQUEST_BATCH * requests_per_drain);
    if tail == 0 || tail >= MIN_CONNECTIVITY_SWEEP_ROWS {
        assert_eq!(
            index.unswept_rows_since_connectivity_sweep(),
            0,
            "drain of {tail} rows must have triggered the builder sweep itself"
        );
    } else {
        assert!(
            index.unswept_rows_since_connectivity_sweep() > 0,
            "expected a {tail}-row unswept tail before the idle-equivalent sweep"
        );
    }
    index.ensure_connectivity().await;
    assert_eq!(
        index.unswept_rows_since_connectivity_sweep(),
        0,
        "explicit sweep must clear the unswept counter"
    );
    (index, flushed_total, saw_parallel, saw_serial)
}

/// 50k 成簇语料 × 两种 drain 规模：level-0 可达率必须 100%，容差召回过门。
/// EN: 50k clustered corpus x two drain sizes: level-0 reachability must be
/// 100% and tolerant recall must pass at every ef.
#[test]
#[ignore = "50k corpus takes minutes in debug; run via scripts/clustered_recall_50k_gate.sh (release)"]
fn legacy_http_batches_keep_clustered_recall_50k() {
    // 四条进程级 OnceLock 全显式钉死：legacy 构图、补救开启、查询侧束宽下限
    // 与自适应开关。
    // EN: Pin all four process-wide OnceLocks explicitly: legacy build,
    // repair enabled, query-side beam floor, adaptation switch.
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
        let corpus = clustered::ClusteredCorpus::new_with(clustered::DATA_N_LARGE);
        for (label, requests_per_drain, expect_parallel) in [
            ("serial-300", SERIAL_DRAIN_REQUESTS, false),
            ("grouped-600", GROUPED_DRAIN_REQUESTS, true),
        ] {
            let (index, flushed, saw_parallel, saw_serial) =
                build_with_drain_pattern(&corpus.data, requests_per_drain).await;
            assert_eq!(
                flushed,
                corpus.data.len(),
                "{label}: every row must be flushed exactly once"
            );
            assert!(
                saw_serial,
                "{label}: at least one batch must report the serial path"
            );
            assert_eq!(
                saw_parallel, expect_parallel,
                "{label}: recorded plan mode does not match the documented drain size"
            );
            let reachable = clustered::level0_reachable(&index, &corpus.ids);
            assert_eq!(
                reachable,
                corpus.ids.len(),
                "{label}: connectivity sweep must reach every L0 node, got {reachable}/{}",
                corpus.ids.len()
            );
            clustered::assert_clustered_recall(&index, &corpus, label).await;
        }
    });
}
