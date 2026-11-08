//! HTTP API 处理器：benchmark 路由、Collection 路由、混合检索、WAL 操作。
//! HTTP API handlers: benchmark routes, collection routes, hybrid search, WAL operations.

use crate::collection::CollectionManager;
use crate::index_backend::IndexBackend;
use crate::recovery::drain_pending_builder;
use crate::scalar_json::filter_predicate;
use crate::types::*;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use futures_util::StreamExt;
use houyi_core::config::{BuildConfig, OptimizeLogConfig};
use houyi_core::ScalarPredicate;
use houyi_persistence::vector_snapshot::{
    encode_vector_snapshot_v1, write_vector_snapshot_atomic_with_encryption,
};
use houyi_persistence::vector_wal::{encode_vector_upsert_batch, VectorItem};
use houyi_persistence::wal_sidecar::{read_applied_lsn, write_applied_lsn_atomic};
use houyi_vector::{DistanceMetric, HnswBatchPlanMode};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;

/// 已安装标签缓存的字节数（跨标签总预算；预留走原子 CAS）。
/// EN: Bytes of installed label caches (global budget; reserved atomically).
static LABEL_CACHE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 单次检索 `k` 的硬上界。
/// EN: Hard upper bound for a single search's `k`.
///
/// WHY（为什么必须在所有网络边缘封死上界）：`k` 是请求里的裸 `usize`，而检索链路按 k
/// 线性放大内存与工作量——HTTP 边缘 4k 过取、HNSW 访问器约 16×k 预分配、聚类后端按
/// `k*2500` 估算 nprobe。不设上界时，一条构造的请求即可触发数十亿元素的预分配（分配
/// 失败直接 abort 整个进程）/近似全图扫描，构成以小博大的内存放大攻击面。10_000 远超
/// VectorDBBench 等正常客户端使用的 top-k（通常 ≤ 100），同时把放大锁死在常数倍内。
/// 覆盖边缘：三条 HTTP 检索路径 + pgwire 的 `LIMIT n`（见 `pgwire_collect_rows`）。
/// EN: Without an edge cap, a raw request-supplied `k` drives 4k over-fetch, ~16×k HNSW
/// preallocation and `k*2500` nprobe estimates — a cheap memory-amplification vector that
/// can abort the whole process on allocation failure, or force a near-full scan. 10,000 is
/// far above normal top-k workloads (≤100) yet bounds the amplification to a constant
/// multiple. Enforced on all network edges: the three HTTP handlers AND pgwire `LIMIT n`
/// (see `pgwire_collect_rows`).
pub(crate) const MAX_SEARCH_K: usize = 10_000;

/// 单次检索 `ef_search`（HNSW beam 宽度）的硬上界。
/// EN: Hard upper bound for a single search's `ef_search` (HNSW beam width).
///
/// WHY：检索链取 `ann_k = k.max(ef_search)`，ef 与 k 驱动**同一个**放大面——
/// `HnswVisitTracker::with_capacity(16 × ann_k)` 与 beam 堆宽。只封 k 而放行
/// `ef_search: Some(1<<30)` 等于没封：单请求即可请求数百 GB 预分配并 abort 进程。
/// 取与 k 相同的 10_000（基准客户端实际只用 256；库内构图 ef 上限为 4096），
/// 保证 `16 × ann_k` 最坏约 16 万个槽位。pgwire `/* houyi:ef_search=N */` 同口径。
/// EN: The retrieval chain computes `ann_k = k.max(ef_search)`, so ef drives the SAME
/// amplification surface (16×ann_k visit-tracker preallocation + beam width). Capping k
/// without ef would be void — `ef_search: Some(1<<30)` requests hundreds of GB and aborts
/// the process. 10,000 matches the k cap (benchmarks use 256; build-time ef is capped at
/// 4096), bounding the tracker to ~160k slots worst case. The pgwire ef directive shares it.
pub(crate) const MAX_SEARCH_EF: usize = 10_000;

/// 与传输协议无关的检索放大面封边：`k` 或 `ef_search` 越界时返回错误文案，
/// 由各边缘映射成自己的错误信封（HTTP → 400；pgwire → SQLSTATE 22023）。
/// EN: Transport-agnostic edge check for the retrieval amplification surface; returns an
/// error message that each edge maps to its own envelope (HTTP 400; pgwire SQLSTATE 22023).
///
/// 注意不含 k==0 判定：benchmark 路径刻意保留 VectorDBBench 语义（k==0 → 200 空结果，
/// 见 `benchmark_search`），由调用方在短路处自行处理；collection/hybrid/pgwire 路径在
/// 调用本函数前显式拒绝 k==0。WHY: 两条语义都是有意为之，藏进统一函数会让差异不可见。
/// EN: k==0 is deliberately NOT handled here: the benchmark path keeps VectorDBBench's
/// empty-200 semantics and short-circuits itself, while collection/hybrid/pgwire reject
/// zero explicitly before calling. Keeping the divergence visible at the call sites.
pub(crate) fn search_limit_violation(k: usize, ef_search: Option<usize>) -> Option<String> {
    if k > MAX_SEARCH_K {
        return Some(format!(
            "k={k} exceeds the maximum allowed value {MAX_SEARCH_K} \
             (bound on per-query preallocation/scan amplification)"
        ));
    }
    if let Some(ef) = ef_search {
        if ef > MAX_SEARCH_EF {
            return Some(format!(
                "ef_search={ef} exceeds the maximum allowed value {MAX_SEARCH_EF} \
                 (bound on HNSW beam width / visit-tracker preallocation)"
            ));
        }
    }
    None
}

/// HTTP 边缘封边：越界映射为 400。三条检索路径共用这唯一实现。
/// EN: HTTP-edge wrapper mapping any search-limit violation to 400; shared by all
/// three search handlers.
fn reject_search_limits(
    k: usize,
    ef_search: Option<usize>,
) -> Result<(), (axum::http::StatusCode, String)> {
    match search_limit_violation(k, ef_search) {
        Some(msg) => Err((axum::http::StatusCode::BAD_REQUEST, msg)),
        None => Ok(()),
    }
}

pub fn bench_build_config(base: BuildConfig) -> BuildConfig {
    BuildConfig {
        parallel_batch_build: if std::env::var("HOUYI_HNSW_PARALLEL_BATCH_BUILD").is_err() {
            true
        } else {
            base.parallel_batch_build
        },
        max_build_batch: if std::env::var("HOUYI_HNSW_MAX_BUILD_BATCH").is_err() {
            8192
        } else {
            base.max_build_batch
        },
        ..base
    }
}

/// 每次 `POST /v1/optimize` 结束发一条 tracing 事件（VectorDBench 在后台发 optimize 时，看 houyi-server 终端即可）。
/// 若配置 `log_file` 则同时把原始一行追加到该路径。`quiet=true` 时仅写文件、不发事件。
/// EN: One tracing event per optimize; optional raw-line append file; quiet skips the event.
fn log_houyi_optimize_post_result(resp: &OptimizeResponse, log_cfg: &OptimizeLogConfig) {
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    if !log_cfg.quiet {
        tracing::info!(
            ts_ms,
            flushed = resp.flushed_items,
            pending = resp.pending_items,
            hnsw_parallel_batch_rounds = resp.hnsw_parallel_batch_rounds,
            hnsw_serial_batch_rounds = resp.hnsw_serial_batch_rounds,
            "[houyi-optimize]"
        );
    }
    if let Some(p) = log_cfg.log_file.as_deref() {
        let p = p.trim();
        if !p.is_empty() {
            let line = format!(
                "[houyi-optimize] ts_ms={} flushed={} pending={} hnsw_parallel_batch_rounds={} hnsw_serial_batch_rounds={}",
                ts_ms,
                resp.flushed_items,
                resp.pending_items,
                resp.hnsw_parallel_batch_rounds,
                resp.hnsw_serial_batch_rounds
            );
            if let Err(e) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .and_then(|mut f| writeln!(&mut f, "{line}"))
            {
                tracing::warn!(path = %p, error = %e, "[houyi-optimize] append log file failed");
            }
        }
    }
}

/// 列出所有 Collection
/// EN: List all collections.
async fn handle_list_collections(
    State(state): State<BenchmarkHttpState>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
) -> Json<Vec<serde_json::Value>> {
    let cm = &state.collections;
    let names = cm.list();
    // 作用域过滤（P4.7.2）：租户/受限主体只看到自己的 collection 名单；
    // 未启用认证（扩展缺失）或未设作用域时返回全部。
    // EN: scoped principals only see their own collection names.
    let names: Vec<String> = names
        .into_iter()
        .filter(|name| principal.as_ref().is_none_or(|ext| ext.0.may_see(name)))
        .collect();
    let mut infos = Vec::new();
    for name in names {
        if let Some(col) = cm.get(&name) {
            // 注意: 此端点同步，indexed_count 可能为近似值（异步后端）
            // EN: Note: this endpoint is synchronous; indexed_count may be approximate
            // (async backend).
            let indexed = col.backend.indexed_count().await;
            let mut info = serde_json::json!({
                "name": name,
                "dims": col.dims,
                "index_type": col.index_type,
                "vectors": indexed,
                // 度量与 HNSW 参数为创建时定型值（不可变）；非 hnsw 后端不返回。
                // EN: metric and HNSW parameters are fixed at creation (immutable);
                // the hnsw object is omitted for non-hnsw backends.
                "metric": col.metric.as_str(),
                // 复制级别只读展示；变更仍走 POST .../ha_level（WAL-first）。
                // EN: read-only replication level for display; changes still go
                // through POST .../ha_level (WAL-first).
                "ha_level": col.ha_level().as_str(),
            });
            // 条件插入而非 `null`：`json!` 对 `.then(...)` 的 None 会输出
            // `"hnsw": null`，与「非 hnsw 后端不返回该字段」的契约不符（OCR #19）。
            // EN: Conditional insert instead of `null`: `json!` renders a `.then(...)`
            // None as `"hnsw": null`, breaking the "omitted for non-hnsw backends"
            // contract (OCR #19).
            if col.index_type == "hnsw" {
                let p = col.hnsw_params();
                if let Some(obj) = info.as_object_mut() {
                    obj.insert(
                        "hnsw".to_string(),
                        serde_json::json!({
                            "m": p.m,
                            "ef_construction": p.ef_construction,
                            "ef_search": p.ef_search,
                        }),
                    );
                }
            }
            infos.push(info);
        }
    }
    Json(infos)
}

/// 向指定 Collection 写入向量 / Upsert vectors into a named collection
///
/// 向量进入索引后端；携带的 `label`/`payload` 在向量写入成功后追加到该 Collection
/// 的内存 PAX 标量页（供 `/hybrid/search` 结构化过滤）。标量页随 collections WAL
/// 与快照持久化，重启后由重放/快照恢复（详见
/// [`crate::collection::CollectionScalars`]）。
/// EN: Vectors go to the index backend; on success each vector's `label`/`payload`
/// is appended to the collection's in-memory PAX scalar page for hybrid filtering.
/// The page is persisted through the collections WAL and snapshots and restored
/// on restart (see [`crate::collection::CollectionScalars`]).
/// 只读副本拦截（P4.4.1）：role=follower 时写端点入口首行 503 `read_only`——
/// 数据面变更只允许经复制流到达（apply 直调 CollectionManager，天然豁免）。
/// 检索/metrics/health 不调用本守卫，只读副本可查是卖点。
/// EN: Read-only guard at the head of every write endpoint — 503 `read_only`
/// while the runtime role is follower; data changes may only arrive via the
/// replication stream (its apply bypasses HTTP entirely). Read paths never
/// call this guard: serving reads is the follower's raison d'être.
pub(crate) fn reject_if_read_only(
    state: &BenchmarkHttpState,
) -> Result<(), (axum::http::StatusCode, String)> {
    if state.role.is_follower() {
        return Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "read_only: follower replica rejects writes".to_string(),
        ));
    }
    // 租约 fencing（EE M1）：`require_leader_lease=false`（默认）时直接放行，
    // 行为与手动 promote 逐字节兼容；开启后 leader 无有效租约即拒写。
    // 过期不自动翻角色：sentinel 会显式 demote；此处只做写闸门，避免与续租
    // 竞争产生「角色 follower、但没有任何人重新 promote」的死局。
    // EN: lease fencing (EE M1). With the default off, behavior is unchanged.
    // Expiry does not flip the role here — sentinel demotes explicitly; fencing
    // only gates writes, avoiding a renew/flip race that would leave the node
    // follower with nobody re-promoting it.
    match state.cluster.check_writable() {
        Ok(()) => Ok(()),
        Err(crate::cluster::LeaseError::Missing) => Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "lease_missing: leader has no active lease (require_leader_lease)".to_string(),
        )),
        Err(crate::cluster::LeaseError::Expired) => Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "lease_expired: leader lease expired; writes are fenced".to_string(),
        )),
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("lease check failed: {e}"),
        )),
    }
}

/// `begin_write` 失败的统一映射：Collection 正在删除/已删除 → 409 Conflict。
/// EN: Uniform 409 mapping for a failed `begin_write` (collection being/deleted).
///
/// WHY：写路径先 `get` 拿到 `Arc` 再取写锁；删除方在两步之间可能已
/// `mark_deleted`，此时继续写会破坏「删除后不复活」的线性化语义。
/// `begin_write` 的双检把这个窗口收敛成确定的 409，而不是让写静默落进已删对象。
/// EN: Write paths fetch the `Arc` before locking; the deleter may mark it between
/// the two steps. Continuing would resurrect a deleted collection. The double
/// check inside `begin_write` turns that window into a deterministic 409.
pub(crate) fn collection_deleted_conflict(
    name: &str,
    _err: crate::collection::CollectionDeleted,
) -> (axum::http::StatusCode, String) {
    (
        axum::http::StatusCode::CONFLICT,
        format!("collection '{name}' is being deleted"),
    )
}

/// 记录一条查询日志（EE M1）。关闭时零开销；只记元数据，不含向量/payload/SQL。
/// EN: record one query-log entry (EE M1); free when disabled, metadata only.
#[allow(clippy::too_many_arguments)]
fn record_query_log(
    state: &BenchmarkHttpState,
    principal: Option<&crate::rbac::AuthenticatedPrincipal>,
    route: &'static str,
    collection: Option<&str>,
    k: usize,
    status: u16,
    started: std::time::Instant,
    result_count: usize,
    filtered: bool,
) {
    let Some(log) = &state.query_log else {
        return;
    };
    log.record(crate::query_log::QueryLogEntry {
        ts_ms: crate::query_log::now_ms(),
        request_id: 0,
        principal: principal
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "-".to_string()),
        route,
        collection: collection.map(str::to_string),
        k,
        status,
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
        result_count,
        filtered,
    });
}

/// 测试用集群控制句柄（不授予租约；open 只读，不写盘）。
/// EN: test-only cluster control (no grants; open only reads).
#[cfg(test)]
fn test_cluster() -> Arc<crate::cluster::ClusterControl> {
    // 每次调用独占目录：cluster_state.json 落在 WAL 父目录，共享目录会让
    // 并行测试在同一个 .json.tmp 上竞争（与 cluster.rs 单测同一纪律）。
    // EN: every call gets its own directory — cluster_state.json lives beside
    // the WAL, and a shared directory races on the same .json.tmp (same rule
    // as the cluster.rs unit tests).
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "houyi-http-test-cluster-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    Arc::new(
        crate::cluster::ClusterControl::open(&dir.join("test.wal"), false)
            .expect("cluster control"),
    )
}

/// 数据面 sync 等待（P4.4.2）：写端点响应前按 collection 的 ha_level 等 follower ACK。
/// Sync wait at the tail of a write endpoint, keyed by the collection's ha_level.
///
/// 语义 / Semantics:
/// - `async`（默认）：一次 Relaxed load 即返回，热路径零额外开销。
/// - `semi_sync`：等任一 follower 的 ACK(RECEIVED)（收帧即回，+1 RTT）。
/// - `full_sync`：等 ACK(DURABLE)（follower fsync + 进 HotBuffer 可查后回；
///   与 leader 自身语义同级，不等异步 HNSW 建图）。
/// - 无健康 follower / 等待超时：照常返回成功，语义降级为 async——写已本地
///   durable；超时计 `sync_timeouts_total` 并 rate-limited warn（MySQL
///   semi-sync 降级路线，指标承载完整计数）。
///
/// 调用纪律：WAL append + 内存变更完成之后、响应客户端之前，且**已释放**
/// collection 的 write_lock——等待是纯观测，持锁等待会把并发写串行到 sync
/// RTT 上，摧毁 group commit 的吞吐意义。等待的 LSN 必须是本次写入的**最后
/// 一条** WAL 记录（create+upsert 两条时只等 upsert 那条，LSN 有序即全覆盖）。
/// `level` 则在**持锁期间**捕获（级别必须定格在本写 WAL LSN 处，见函数内
/// 注释）——「级别锁内取、等待锁外做」是同一纪律的两半。
/// EN: Invoke after the WAL append + in-memory mutations, before the response,
/// with the write_lock already released — waiting is pure observation, and
/// holding the lock would serialize concurrent writes onto the sync RTT. Wait
/// on the LAST WAL LSN of this write (for create+upsert, the upsert record;
/// LSN ordering covers the earlier one). The ha_level, by contrast, is read
/// while the lock is still held so it matches the write's WAL position.
///
/// 非 leader（视图缺失或 follower）：无等待语义直接返回——follower 上的写在
/// 入口已 503，此分支仅防御装配缺口。std 阻塞等待经 spawn_blocking 桥接，
/// 不占 tokio worker。HTTP 与 pgwire（P4.4.3 补齐）共用本函数——故参数是
/// 指标槽而非 HTTP state。
pub(crate) async fn sync_wait_if_required(
    replication_metrics: &crate::types::SharedReplicationMetrics,
    col: &crate::collection::Collection,
    level: houyi_persistence::ha_level::HaLevel,
    lsn: u64,
) {
    use houyi_persistence::ha_level::HaLevel;
    // level 由调用方在**持 write_lock 期间**捕获传入（OCR f942e45[9]）：
    // set_ha_level 也走同一把锁（WAL-first 后改原子量），锁内读到的级别
    // 即本写 WAL LSN 处的在效级别——锁释放后再读会被并发降级抢走
    // semi/full_sync 承诺（静默剥语义，且无指标痕迹）。
    // EN: `level` is captured by the caller while write_lock is still held,
    // making it exactly the level in effect at this write's WAL LSN.
    if level == HaLevel::Async {
        return;
    }
    let (hub, watermarks, timeout, quorum) = {
        // 毒锁不得丢观测（benchmark_metrics 同款纪律）：写者 panic 后读侧
        // into_inner 恢复——此处恢复语义是「跳过等待降级 async」，而非 500。
        // 但恢复出的视图可能是 None/Follower 旧值，sync 等待会被静默跳过
        // （OCR 6bfb210[4]）——留诊断痕迹；毒化是持续态，once 告警防抖。
        // EN: A poisoned lock must not lose observability (same discipline as
        // benchmark_metrics): recover via into_inner after a writer panic — here
        // recovery means "skip the wait, degrade to async", not a 500. The
        // recovered view may be a stale None/Follower value, so sync waits can be
        // silently skipped (OCR 6bfb210[4]) — leave a diagnostic trace; poisoning
        // persists, so a once-only warning avoids spam.
        let guard = replication_metrics.read().unwrap_or_else(|e| {
            static POISON_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !POISON_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "[houyi-sync] replication metrics lock poisoned; recovered view may be \
                     stale — sync waits may be silently skipped (degraded to async)"
                );
            }
            e.into_inner()
        });
        match &*guard {
            Some(ReplicationMetricsView::Leader {
                hub,
                watermarks,
                sync_timeout,
                sync_quorum,
                ..
            }) => (
                Arc::clone(hub),
                Arc::clone(watermarks),
                *sync_timeout,
                *sync_quorum,
            ),
            _ => return,
        }
    };
    let stage = if level == HaLevel::FullSync {
        houyi_persistence::wal_replication::ACK_STAGE_DURABLE
    } else {
        houyi_persistence::wal_replication::ACK_STAGE_RECEIVED
    };
    let outcome = tokio::task::spawn_blocking(move || {
        // quorum > 0 时升级为「K 个活跃 follower 的 DURABLE ≥ lsn」等待；
        // 语义从「任一 ACK」收紧到多数派，代价是 follower 故障时写可用性。
        // EN: quorum > 0 upgrades the wait to "K active followers durable ≥
        // lsn" — a majority instead of any-ACK, trading write availability
        // when followers are down.
        if quorum > 0 {
            return watermarks.wait_durable_quorum(lsn, &hub, quorum, timeout);
        }
        // follower 活性在等待决策点采样（OCR f942e45[4]）：spawn 前的快照
        // 会被 blocking 池排队延迟放陈——恰好在停滞/饱和场景误差最大。
        // EN: Sample follower liveness at the decision point (OCR f942e45[4]): a
        // snapshot taken before spawn can go stale in the blocking-pool queue —
        // worst exactly under stall/saturation.
        let has_followers = hub.follower_count() > 0;
        watermarks.wait(lsn, stage, has_followers, timeout)
    })
    .await;
    use houyi_persistence::replication_leader::SyncWaitOutcome;
    match outcome {
        Ok(SyncWaitOutcome::Satisfied) | Ok(SyncWaitOutcome::NoFollower) => {}
        Ok(SyncWaitOutcome::Timeout) => {
            // rate-limited warn 计数器挂在 Collection 上（OCR 6bfb210[5]）：
            // 生命周期随 collection（drop 即回收，无进程级 HashMap 泄漏/
            // 无界增长问题），分键语义天然成立——前 8 次全打，之后每 1024
            // 次一条，超时风暴期间日志不刷屏。
            // EN: The rate-limited warn counter lives on the Collection (OCR
            // 6bfb210[5]): lifetime follows the collection (dropped with it, no
            // process-wide HashMap leak/unbounded growth), and per-key semantics
            // hold naturally — the first 8 all log, then one every 1024, so
            // timeout storms do not flood the log.
            let n = col
                .sync_timeout_warns
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 || n.is_multiple_of(1024) {
                tracing::warn!(
                    collection = %col.name,
                    lsn,
                    suppressed = n.saturating_sub(8),
                    "[houyi-sync] replication sync wait timed out; write degraded to async"
                );
            }
        }
        Err(join_err) => {
            tracing::warn!(error = %join_err, "[houyi-sync] sync wait task join failed");
        }
    }
}

pub async fn handle_collection_upsert(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    // 边界校验先于任何写入：payload 不得占用保留列名 "label"，否则其值会与顶层
    // label 列冲突（旧实现静默覆盖，造成用户数据无声丢失）。
    // EN: Validate before any write: payload must not use the reserved key "label",
    // which collides with the top-level label column.
    for v in &req.vectors {
        let collides = v
            .payload
            .as_ref()
            .is_some_and(|payload| payload.contains_key("label"));
        if collides {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                "payload key 'label' is reserved; use the top-level 'label' field".to_string(),
            ));
        }
    }

    // Collection 维度以首批首个向量定型；空批次沿用服务默认维度。
    // Collection dims are fixed by the first vector of the first batch; empty batch
    // falls back to the server default.
    let dims = req
        .vectors
        .first()
        .map(|v| v.values.len())
        .unwrap_or(state.vector_dim);
    // 统一边界校验（维度/非有限）先于 Collection 创建与索引写入。
    // EN: Shared boundary validation (dim/finiteness) before any side effect.
    validate_upsert_vectors(&req.vectors, dims)?;
    // 内存水位线（P4.2.4）：校验后、Collection 创建/WAL/索引写入前裁决，超限 503。
    // EN: Memory watermark admission after validation, before any side effect.
    state
        .pgwire_metrics
        .memory_watermark
        .check_admission()
        .map_err(|msg| (axum::http::StatusCode::SERVICE_UNAVAILABLE, msg))?;
    let col = state
        .collections
        .get_or_create(&name, Some(dims))
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    // 写互斥：WAL append 与后续内存变更（insert_batch + 标量行）必须同序——
    // 否则并发同 id 的两批 upsert 可能出现 WAL 顺序 A,B 而内存顺序 B,A，
    // 重放收敛到与已确认状态不同的值（last-writer 分叉）。
    // EN: Write mutex — the WAL append and the in-memory mutations (insert_batch +
    // scalar rows) must share one order; otherwise two concurrent same-id batches
    // can land as WAL A,B but memory B,A, replaying to a value that diverges
    // from the acknowledged last write.
    let write_guard = col
        .begin_write()
        .await
        .map_err(|e| collection_deleted_conflict(&name, e))?;
    // WAL-first：先落盘再改内存，宕机后冷启动重放可完整恢复本批。
    // collection_create 由首个写锁持有者补落（新建时即本请求；删除→同名重建的
    // 竞态由同一函数收敛），记录参数取自 collection 自身——WAL 中 create 永远
    // 先于该 Arc 的任何 upsert，重放不依赖重放期的服务端默认配置。
    // EN: WAL-first — persist before any in-memory mutation so crash + cold replay
    // recovers the batch. The first write-lock holder logs `collection_create`
    // (this request on normal creation; the same helper heals a delete/recreate
    // race) using the collection's own parameters, so create always precedes any
    // upsert of this Arc and replay never depends on replay-time server defaults.
    crate::collection::ensure_create_logged(&state.collections_wal, &name, &col)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let wal_items: Vec<houyi_persistence::collection_wal::CollectionWalItem> = req
        .vectors
        .iter()
        .map(|v| houyi_persistence::collection_wal::CollectionWalItem {
            id: v.id,
            values: v.values.clone(),
            label: v.label.clone(),
            payload: v.payload.clone(),
        })
        .collect();
    let payload =
        houyi_persistence::collection_wal::encode_collection_upsert_batch(&name, dims, &wal_items)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // 本批写入的最后一条 WAL 记录（create 在前，LSN 有序——sync 等待只等这一条
    // 即覆盖整批，见 sync_wait_if_required 文档）。
    // EN: The last WAL record of this write — the sync wait keys on its LSN.
    let wal_entry = state
        .collections_wal
        .append(payload)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let batch: Vec<(u64, Vec<f32>)> = req
        .vectors
        .iter()
        .map(|v| (v.id, v.values.clone()))
        .collect();
    let n = batch.len();
    // insert_batch 的 Err 全部来自输入校验：非有限分量由 IndexBackend 适配层总闸门
    // 对所有后端统一拒绝（库层校验不对称——HiiL0 硬拒、HNSW 只查维度、buf 类后端
    // 连 Err 都不返回），维度由各库层自行拒绝。映射 400 而非裸 500。HTTP 边界已
    // 预校验这两个条件，走到这里理论上不可达；若未来某后端新增非校验类 Err
    // （持久化/锁中毒/内部状态），这行会把服务端故障误记为 4xx，故先打日志让误分类
    // 可观测（WHY: 不要静默改变错误分类）。
    // EN: insert_batch errors are input validation only: the IndexBackend adapter gate
    // rejects non-finite components uniformly (library validation is asymmetric) and each
    // library rejects dim mismatches itself. Map to 400. The edge pre-validates both, so
    // this is effectively unreachable; log to keep any future non-validation backend fault
    // observable instead of silently polluting 4xx.
    col.backend.insert_batch(batch).await.map_err(|e| {
        tracing::warn!(error = %e, "[houyi-upsert] collection insert_batch rejected (expected validation-only)");
        (axum::http::StatusCode::BAD_REQUEST, e)
    })?;

    // 仅当至少一个向量携带 label/非空 payload 时才建行；
    // row_to_id 必须与页内行严格一一对应，因此「裸向量」不插入标量页。
    // EN: Build scalar rows only for vectors carrying label/non-empty payload;
    // row_to_id must stay aligned 1:1 with page rows, so bare vectors skip the page.
    let has_scalars = req
        .vectors
        .iter()
        .any(|v| v.label.is_some() || v.payload.as_ref().is_some_and(|p| !p.is_empty()));
    if has_scalars {
        let mut scalars = col
            .scalars
            .lock()
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        for v in &req.vectors {
            // 行构建与 collections WAL 重放共用同一函数（WHY 见 build_collection_scalar_row）。
            // EN: Row construction is shared with collections-WAL replay.
            let row = crate::scalar_json::build_collection_scalar_row(
                v.label.as_deref(),
                v.payload.as_ref(),
            );
            if !row.is_empty() {
                // 页行与 row_to_id 在 append_row 内原子推进，1:1 对齐不再靠调用方自觉。
                // EN: append_row advances the page row and row_to_id atomically — the
                // 1:1 alignment no longer depends on callers remembering both pushes.
                //
                // 旧行暂时保留，但过滤只看每个 id 的最后一行（旧值从写入起即失效，
                // 公共 issue #6）；周期内存压实（P5.0.1）只按 stale 比例回收物理行，
                // 不改变查询结果。
                // EN: Stale rows are retained, but filters only see the last row per id
                // (superseded values stop matching from the write on; public issue #6).
                // The periodic in-memory compaction (P5.0.1) only reclaims physical rows
                // by stale ratio and never changes query results.
                scalars.append_row(v.id, row);
            }
        }
    }
    // 后端与标量行均已成功写入：推进 applied_lsn，快照 checkpoint 可以覆盖本批。
    // EN: Backend and scalar rows are durable in memory — advance applied_lsn so a
    // snapshot checkpoint may cover this batch.
    col.note_applied_lsn(wal_entry.lsn);

    // sync 等待（P4.4.2）在 write_lock 释放后进行：等待是纯观测，持锁等待会
    // 把并发写串行到 sync RTT 上（详见 sync_wait_if_required 文档）。级别在
    // 持锁期间捕获——定格在本写 WAL LSN 处的在效级别。
    // EN: The sync wait runs after releasing the write_lock; the level is
    // captured while the lock is still held.
    let ha_level = col.ha_level();
    drop(write_guard);
    sync_wait_if_required(&state.replication_metrics, &col, ha_level, wal_entry.lsn).await;

    Ok(Json(UpsertResponse { inserted: n }))
}

/// 显式建集合（`PUT /v1/collections/{name}`）。
///
/// 与隐式创建（upsert/pgwire 首批写入）的区别：维度、度量、后端与 HNSW 参数在
/// 创建时定型并由响应回显，客户端无需猜测服务端默认值。
///
/// 状态码 / status codes:
/// - 201 创建成功（`collection_create` 记录已落 WAL 后才应答）；
/// - 400 参数非法（dims=0、未知度量/后端、后端不支持该度量、半套 HNSW 参数）；
/// - 409 同名 collection 已存在（配置变更必须删库重建：旧图按旧度量构建，
///   换度量后不可复用）；
/// - 503 follower/只读/租约 fencing。
///
/// EN: Explicit collection creation. Unlike implicit creation (first upsert), the
/// dims, metric, backend and HNSW parameters are fixed here and echoed back.
/// 201 on success (acked only after the `collection_create` record is durable);
/// 400 for invalid parameters; 409 when the name exists (metric/backend changes
/// require drop-and-recreate because graphs are built under the old metric);
/// 503 on followers/read-only/lease fencing.
pub async fn handle_create_collection(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<
    (axum::http::StatusCode, Json<CreateCollectionResponse>),
    (axum::http::StatusCode, String),
> {
    reject_if_read_only(&state)?;
    // 全部校验先于任何副作用：坏请求不得在 WAL 留下记录（与 drop/delete 同纪律）。
    // EN: all validation precedes any side effect — a bad request must never leave
    // a WAL record (same discipline as drop/delete).
    let index_type = req
        .index_type
        .unwrap_or_else(|| state.collections.implicit_create_params().0);
    // 缺省度量取决于最终生效的 index_type：pq-flat 实际按 L2 计分，若在此先
    // 硬编码 Cosine 再校验，会把非法组合报成 400 而不是采用真实缺省（OCR #20）。
    // EN: The default metric depends on the effective index_type: pq-flat scores
    // by L2, so hard-coding Cosine before resolving the backend would reject the
    // default combination as 400 instead of using its real default (OCR #20).
    let metric = match req.metric.as_deref() {
        Some(m) => DistanceMetric::parse_str(m).ok_or_else(|| {
            (
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid metric {m:?} (expect l2|cosine|ip|l1)"),
            )
        })?,
        None => CollectionManager::default_metric_for_index_type(&index_type),
    };
    let hnsw = match (req.hnsw_m, req.hnsw_ef_construction, req.hnsw_ef_search) {
        (None, None, None) => None,
        (Some(m), Some(ef_construction), Some(ef_search)) => Some(crate::collection::HnswParams {
            m,
            ef_construction,
            ef_search,
        }),
        _ => {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                "hnsw_m/hnsw_ef_construction/hnsw_ef_search must be provided together".to_string(),
            ))
        }
    };
    CollectionManager::validate_create(&index_type, req.dims, metric, hnsw)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let (col, created) = state
        .collections
        .get_or_create_typed_with_metric(&name, req.dims, &index_type, Some(metric), hnsw)
        .map_err(|e| {
            // validate_create 已证明同一组参数可构造，因此这里只有「同名已存在且
            // 参数不符」一种失败：名字被占用是 409，不是请求格式错误。不再回读
            // map 猜测状态码（旧写法在并发删除窗口内会误报 400，OCR #23）。
            // EN: validate_create already proved these parameters constructible, so
            // the only remaining failure is "same name exists with different
            // parameters": the name is taken, so 409 — not malformed input. Do not
            // re-read the map to guess (the old TOCTOU could report 400 inside a
            // concurrent drop window, OCR #23).
            (axum::http::StatusCode::CONFLICT, e)
        })?;
    if !created {
        return Err((
            axum::http::StatusCode::CONFLICT,
            format!("collection '{name}' already exists"),
        ));
    }
    // 与 upsert 同一复合写纪律：create 记录在首个写锁持有者手里落盘。显式创建
    // 通常就是首个持有者；若并发写抢先在锁内补落（参数同样取自本 Arc），
    // ensure 直接返回 None，此时记录也已持久。
    // EN: same compound-write discipline as upsert — the first write-lock holder
    // logs the create record. Usually this handler; if a concurrent write got the
    // lock first it logged the identical record from the same Arc, and ensure
    // returns None (already durable).
    let write_guard = col
        .begin_write()
        .await
        .map_err(|e| collection_deleted_conflict(&name, e))?;
    let create_lsn =
        match crate::collection::ensure_create_logged(&state.collections_wal, &name, &col) {
            Ok(lsn) => lsn,
            Err(e) => {
                // create 记录落盘失败时不得把「未持久化」的 collection 留在内存里：
                // 进程内它可见可写，但重启后会凭空消失（OCR #21）。持写锁置删除
                // 哨兵并从 manager 摘除，与 drop 同一收尾；在途写入拿锁后会看到
                // 哨兵并失败，不会有任何写入被确认到孤儿对象。
                // EN: A failed create-record append must not leave a non-durable
                // collection visible/writable in memory — it would vanish on restart
                // (OCR #21). Mark the drop sentinel under the write lock and remove it
                // from the manager, exactly like a drop; in-flight writers will see
                // the sentinel after locking and fail, so nothing acknowledges to an
                // orphan.
                col.mark_deleted();
                state.collections.remove_if_same(&name, &col);
                return Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e));
            }
        };
    let ha_level = col.ha_level();
    drop(write_guard);
    // 应答前等待 create 记录达到配置的复制级别（与 upsert 同纪律：等待在锁外）。
    // EN: wait for the create record to reach the configured replication level
    // before acking (same discipline as upsert: wait outside the lock).
    if let Some(lsn) = create_lsn {
        sync_wait_if_required(&state.replication_metrics, &col, ha_level, lsn).await;
    }
    let hnsw_response = (col.index_type == "hnsw").then(|| {
        let p = col.hnsw_params();
        HnswParamsResponse {
            m: p.m,
            ef_construction: p.ef_construction,
            ef_search: p.ef_search,
        }
    });
    Ok((
        axum::http::StatusCode::CREATED,
        Json(CreateCollectionResponse {
            created: true,
            collection: name,
            dims: col.dims,
            metric: col.metric.as_str().to_string(),
            index_type: col.index_type.clone(),
            hnsw: hnsw_response,
        }),
    ))
}

/// 把后端返回的「(id, 距离) 升序」对转换为 HTTP 响应：按 collection 的度量把距离
/// 换算成相似度分数（越大越相似，保持输入的非递增顺序）。
/// EN: Convert backend pairs (id, distance, ascending) into the HTTP shape, converting
/// distance → similarity score by the collection's metric (larger = closer, preserving
/// the non-increasing input order). The conversion lives only here at the HTTP edge so
/// the kernel can stay uniformly "smaller distance = closer".
///
/// 换算表 / Conversion:
/// - cosine：`1 - d`，钳制 `[-1, 1]`（f32 舍入可让相同向量距离略负、分数略超 1）；
/// - l2 / l1：`1 - d`，不钳制（平方 L2 / 曼哈顿距离合法越界，钳制会改写真实排序信号）；
/// - ip：`-d`（内核距离为 `-dot`，取反即原始内积，与 Milvus IP 语义一致）。
///
/// EN: cosine → `1-d` clamped to [-1,1] (rounding can push identical vectors slightly
/// past 1); l2/l1 → `1-d` unclamped (squared-L2 / Manhattan distances legitimately fall
/// outside [-1,1]); ip → `-d` (the kernel distance is `-dot`, so negation is the raw
/// inner product, matching Milvus IP semantics).
///
/// 非有限距离（NaN/+Inf/-Inf，pg_wire binary float4/float8 可写入）一律成对丢弃：
/// serde_json 不会报错，而是把非有限浮点序列化为 `null`（合法 JSON 却静默破坏
/// `scores` 的数值契约，强类型客户端反序列化即失败），且 NaN 距离无法参与有意义排序。
/// EN: Pairs with non-finite distance are dropped together. serde_json does not reject
/// non-finite floats — it serializes them as `null` (valid JSON, but it silently breaks
/// the numeric `scores` contract and typed-client decoding), and NaN cannot be sorted.
fn scored_pairs_to_response(pairs: Vec<(u64, f32)>, metric: DistanceMetric) -> SearchResponse {
    let mut dropped_non_finite = 0usize;
    let (ids, scores) = pairs
        .into_iter()
        .filter_map(|(id, distance)| {
            if !distance.is_finite() {
                dropped_non_finite += 1;
                return None;
            }
            let score = match metric {
                DistanceMetric::Cosine => (1.0 - distance).clamp(-1.0, 1.0),
                DistanceMetric::L2 | DistanceMetric::L1 => 1.0 - distance,
                DistanceMetric::InnerProduct => -distance,
            };
            Some((id, score))
        })
        .unzip();
    // 结果数可能因此少于 k；打一行诊断让「返回数缩水」可归因（通常是 pg_wire binary
    // float4/float8 写入了 NaN/Inf 的脏向量）。
    // EN: Results may shrink below k; log once so shrinking stays attributable (usually
    // NaN/Inf vectors persisted via pg_wire binary floats).
    if dropped_non_finite > 0 {
        tracing::warn!(
            dropped_non_finite,
            "[houyi-search] dropped non-finite hit(s) (NaN/±Inf distance); \
             result count reduced — check for poisoned vectors"
        );
    }
    SearchResponse {
        ids,
        scores,
        payloads: None,
        metric: None,
    }
}

/// 为 collection 检索响应填充 payload（`include_payload=true` 时调用）。
/// Attach per-hit payloads to a collection search response (include_payload=true).
///
/// 锁纪律：持标量页锁做一次合并扫页（k 个 id 一次 O(页行数) 反向扫描，详见
/// [`crate::collection::CollectionScalars::payloads_for`]），锁不跨 .await。
/// EN: One merged reverse page scan under the scalar lock (no guard across .await).
fn attach_payloads(
    col: &crate::collection::Collection,
    resp: &mut SearchResponse,
) -> Result<(), (axum::http::StatusCode, String)> {
    let scalars = col
        .scalars
        .lock()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let found = scalars.payloads_for(&resp.ids);
    let mut payloads = std::collections::HashMap::with_capacity(found.len());
    for (id, row) in found {
        let obj: serde_json::Map<String, serde_json::Value> = row
            .iter()
            .map(|(k, v)| (k.clone(), crate::scalar_json::scalar_to_json(v)))
            .collect();
        payloads.insert(id.to_string(), serde_json::Value::Object(obj));
    }
    resp.payloads = Some(payloads);
    Ok(())
}

/// 从「按 4k 过取回来的候选对」中剔除非有限距离，再截断到 k。
/// EN: Drop non-finite distances from an over-fetched (4k) candidate list, then
/// truncate to k.
///
/// WHY（为什么必须先剔除再截断，且每条检索路径都要 4k 过取）：每个后端都在自己
/// 内部先截断到调用方给的 k。NaN/±Inf 距离可能占据前部槽位——历史毒向量经
/// pg_wire binary `f32::from_bits` 写入（服务端写入闸门建立前的存量数据），即使
/// 写入全有限，`distance::cosine` 的 f32 累加在 |x| ≳ 1.8e19 时也会溢出成
/// NaN/Inf；部分图后端以 `partial_cmp().unwrap_or(Equal)` 排序，NaN 与谁都相等、
/// 可落在任意位置。若调用方只要 k 行，毒行被最终转换器丢弃后响应就缩水且无候选
/// 可回填；按 4k 过取后，k..4k 区间内已取回的有限候选在此回填。顺序不可颠倒。
/// [`scored_pairs_to_response`] 内的丢弃保留为最后一道安全网。
/// EN: Each backend truncates internally to the requested k, and non-finite distances
/// can occupy leading slots (legacy pg_wire binary poison rows, or f32 overflow inside
/// distance::cosine even for fully-finite inputs; graph sorts using partial_cmp with
/// NaN→Equal, so a NaN row may land anywhere). Over-fetching 4k and dropping BEFORE
/// truncate lets finite candidates in k..4k backfill; reversing the order cannot.
/// The response converter keeps its own drop as the final safety net.
///
/// 可见性说明：本函数仅为 crate 内部辅助，但单测模块挂在 bin target（`main.rs` 的
/// `mod tests`）、经库 crate 的外部路径 `houyi_server::http_handler` 引用，
/// `pub(crate)` 对该路径不可见，故声明为 `pub`；不属于对外承诺的稳定 API。
/// 加 `#[doc(hidden)]` 显式从 rustdoc 公共面隐藏：签名随时可能收回为 `pub(crate)`，
/// 下游不得依赖（WHY: 编译器可见性被迫放宽，但文档不必把它承诺成 API）。
/// EN: Visibility is `pub` only because the test module lives in the bin target and
/// reaches this through the library crate's external path; not a stable public API.
/// `#[doc(hidden)]` keeps it out of the documented surface — it may shrink back to
/// `pub(crate)` at any time and downstream code must not rely on it.
#[doc(hidden)]
pub fn finite_pairs_truncated_to_k(mut pairs: Vec<(u64, f32)>, k: usize) -> Vec<(u64, f32)> {
    pairs.retain(|(_, distance)| distance.is_finite());
    pairs.truncate(k);
    pairs
}

/// 写入边界统一校验：逐向量检查维度与分量有限性。benchmark 与 collection 两条 upsert
/// 路径共用这唯一实现（WHY: 校验曾经两边各写一份并漂移——benchmark 路径漏掉非有限
/// 拦截，毒行先写 WAL 再被库层拒绝，残留进日志）。必须在任何副作用（WAL/label 索引/
/// Collection 创建/构图参数修改）之前调用。
/// 返回值直接沿用各 handler 的错误元组类型（不引入只被两处使用的别名——半采用的别名
/// 会重新漂移；全文件 handler 签名都显式写出该元组）。
/// EN: The single write-boundary validator (dim + finiteness) for both upsert paths.
/// Checks previously drifted apart: the benchmark path let a poisoned vector reach the
/// WAL before the library rejected it. Must run before any side effect (WAL, label index,
/// collection creation, build hints). Returns the handlers' explicit error tuple rather
/// than a half-adopted alias used only here.
fn validate_upsert_vectors(
    vectors: &[UpsertVector],
    expected_dim: usize,
) -> Result<(), (axum::http::StatusCode, String)> {
    for v in vectors {
        if v.values.len() != expected_dim {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "dim mismatch: expect {expected_dim}, got {}",
                    v.values.len()
                ),
            ));
        }
        // NaN 毒化范数/点积与排序（clamp 对 NaN 无效）；Inf 让余弦归一化溢出成全 0。
        // JSON 1e40 会被反序列化为 f32::Inf，光靠 JSON 解析拦不住。
        // EN: NaN poisons norms/dots/ranking (clamp can't fix NaN); Inf overflows cosine
        // normalization to zeros. JSON `1e40` decodes to f32::Inf, parsing won't reject it.
        if let Some(bad) = v.values.iter().find(|x| !x.is_finite()) {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "vector id={} contains non-finite component (NaN/Infinity): {bad}",
                    v.id
                ),
            ));
        }
    }
    Ok(())
}

/// 查询边界统一校验：维度 + 分量有限性。非有限查询的距离恒 NaN、排序未定义
/// （HiiL0 库层另有返回空集的最后防线），必须在进索引前以 400 拒绝。
/// EN: Single read-boundary validator (dim + finiteness). A non-finite query yields NaN
/// distances and undefined ranking (HiiL0 defends itself by returning empty); reject 400
/// before touching the index.
fn validate_query_vector(
    vector: &[f32],
    expected_dim: usize,
    context: &str,
) -> Result<(), (axum::http::StatusCode, String)> {
    if vector.len() != expected_dim {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "{context} dim {} != expected dim {expected_dim}",
                vector.len()
            ),
        ));
    }
    if let Some(bad) = vector.iter().find(|x| !x.is_finite()) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            format!("{context} vector contains non-finite component (NaN/Infinity): {bad}"),
        ));
    }
    Ok(())
}

/// 在指定 Collection 中检索
/// EN: Search within a named collection.
pub async fn handle_collection_search(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    let started = std::time::Instant::now();
    let k = req.k;
    let result = collection_search_inner(
        State(state.clone()),
        axum::extract::Path(name.clone()),
        principal.clone(),
        Json(req),
    )
    .await;
    // 失败路径同样入查询日志：否则 errors 桶永远为 0，统计失真。
    // EN: log failures too; otherwise the errors bucket is structurally zero.
    if let Err((status, _)) = &result {
        record_query_log(
            &state,
            principal.as_ref().map(|ext| &ext.0),
            "collection_search",
            Some(&name),
            k,
            status.as_u16(),
            started,
            0,
            false,
        );
    }
    result
}

async fn collection_search_inner(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;
    // k 边界先于任何检索：collection 路径下 k==0 语义非法（400），超上界同样 400。
    // EN: k boundary before touching the index — zero is a 400 on this path (unlike the
    // benchmark compat path) and over-cap k is rejected too.
    if req.k == 0 {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "k must be >= 1 (use a positive top-k)".to_string(),
        ));
    }
    reject_search_limits(req.k, req.ef_search)?;
    validate_query_vector(&req.vector, col.dims, "query")?;
    // 与 benchmark_search 同一纪律：4k 过取 + truncate 前剔除非有限距离，
    // 防止毒行占据前 k 槽位导致响应缩水（详见 finite_pairs_truncated_to_k）。
    // EN: Same discipline as benchmark_search — over-fetch 4k and drop non-finite
    // distances before truncate (see finite_pairs_truncated_to_k).
    // 饱和乘法后的 `.max(req.k)` 守卫实际不可达（4k 恒 ≥ k），保留以对齐库内
    // `saturating_mul(n).max(k)` 惯例（见 houyi-vector 检索链路）。
    // EN: The `.max(req.k)` after saturating_mul is unreachable (4k ≥ k always); kept
    // deliberately to mirror the in-library `saturating_mul(n).max(k)` convention.
    let fetch_k = req.k.saturating_mul(4).max(req.k);
    // 慢查询计时（P4.3.2）：阈值 0 = 禁用，超阈值打 warn（不含向量本体）。
    // EN: Slow-query timing (P4.3.2): threshold 0 disables; over threshold emits a
    // warn (without the vector itself).
    let t0 = std::time::Instant::now();
    let pairs = col
        .backend
        .search_scored_with_ef_override(&req.vector, fetch_k, None, req.ef_search)
        .await;
    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let threshold = state
        .pgwire_metrics
        .slow_query_ms
        .load(std::sync::atomic::Ordering::Relaxed);
    if crate::types::slow_query_exceeded(elapsed_ms, threshold) {
        tracing::warn!(
            elapsed_ms,
            threshold_ms = threshold,
            collection = %name,
            k = req.k,
            "slow http collection search"
        );
    }
    let pairs = finite_pairs_truncated_to_k(pairs, req.k);
    // 距离 → 分数按 collection 度量换算（cosine 钳制、其余不钳制；ip 取反）。
    // EN: distance → score by the collection metric (cosine clamps, others don't;
    // ip negates).
    let mut resp = scored_pairs_to_response(pairs, col.metric);
    resp.metric = Some(col.metric.as_str().to_string());
    if req.include_payload {
        attach_payloads(&col, &mut resp)?;
    }
    // 普通 collection 检索不接受标量过滤（过滤走 hybrid_search），
    // SearchRequest 里的 filter 字段在此路径被忽略，故 filtered 恒为 false。
    // EN: plain collection search ignores the filter fields (filters go through
    // hybrid_search), so the filtered flag is always false here.
    record_query_log(
        &state,
        principal.as_ref().map(|ext| &ext.0),
        "collection_search",
        Some(&name),
        req.k,
        200,
        t0,
        resp.ids.len(),
        false,
    );
    Ok(Json(resp))
}

/// 从 Collection 删除向量 / Delete vectors from a named collection
///
/// 语义：按 id 删除，幂等——不存在的 id 静默跳过，`deleted` 只计真实存在的删除。
/// 向量侧经 [`crate::index_backend::IndexBackend::delete`]（hnsw/flat 支持，其余
/// 后端 400 fail-fast）；标量侧登记墓碑（append-only 页不能物理删行），后续的
/// 过滤命中与 payload 回读都跳过已删 id。删除 WAL-first（`collection_delete`
/// 记录先于内存变更），墓碑随快照持久化，重启后由重放/快照恢复。
/// EN: Delete by ids, idempotent — unknown ids are skipped and `deleted` counts only
/// ids that existed. The vector side goes through `IndexBackend::delete` (hnsw/flat
/// only; other backends get 400); the scalar side records tombstones (the append-only
/// page cannot drop rows) so filters and payload read-back skip deleted ids. Deletion
/// is WAL-first (`collection_delete` records precede the in-memory change) and the
/// tombstones are captured by snapshots, so restart restores them via replay/snapshot.
pub async fn handle_collection_delete(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<DeleteResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;
    if req.ids.is_empty() && req.filters.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "delete requires at least one of 'ids' or 'filters'".to_string(),
        ));
    }
    // 过滤求值与混合检索同纪律：谓词构造（op 白名单/字面值形状）先于加锁；支持面
    // 也一致（仅 hnsw/flat，其余 fail-fast——这些后端的 delete 本来就 400，提前拒绝
    // 可避免「显式 ids 已删、过滤部分才报错」的半完成状态）。
    // EN: Same discipline as hybrid search — build predicates (op whitelist, literal
    // shapes) before locking, same support surface (hnsw/flat only; other backends
    // fail fast — their delete is a 400 anyway, and rejecting up front avoids a
    // half-done state where explicit ids were deleted before the filter part errs).
    let mut ids: Vec<u64> = req.ids;
    // 两条删除路径最终都落到 IndexBackend::delete（仅 hnsw/flat 支持，其余 400）。
    // 预检必须先于任何 WAL 记录：否则一条会被 400 拒绝的请求已在 WAL 留下
    // collection_delete 记录，而重放对 backend.delete 的 Err 是 fail-fast（其不变式
    // 为「Err 即后端从不支持 delete，这种记录不可能被写出」）——一条坏请求让服务
    // 永远无法冷启动（WAL 投毒）。
    // EN: Both delete paths end at IndexBackend::delete (hnsw/flat only; others 400).
    // The pre-check must precede any WAL record — otherwise a rejected request
    // leaves a collection_delete record behind, and replay fails fast on the
    // backend Err (whose invariant is "an Err means the backend never supported
    // delete, so no such record could exist"): one bad request bricks cold start.
    if col.index_type != "hnsw" && col.index_type != "flat" {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "delete is unsupported for index_type '{}' (supported: hnsw, flat)",
                col.index_type
            ),
        ));
    }
    if !req.filters.is_empty() {
        let predicates: Vec<ScalarPredicate> = req
            .filters
            .iter()
            .map(|filter| filter_predicate(&filter.column, &filter.op, &filter.value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
        // 写互斥：匹配集求值 → WAL 落盘 → 墓碑登记 → 删向量 全程持有 write_lock，
        // 保证 WAL 顺序与内存顺序一致。拆锁版本有两个已确认的破坏窗口（OCR 评审
        // 发现）：其一，并发同 id upsert 在求值与墓碑之间完成（已 200 确认），随后
        // 被过期匹配集墓碑 + 删向量——已确认写被静默销毁；其二，delete 记录先于
        // upsert 记录落 WAL 而墓碑晚于 upsert 标量行——重放按 delete→upsert 顺序
        // 复活了崩溃前已确认删除的 id，恢复态与已确认态分叉。
        // EN: Write mutex spans match evaluation → WAL append → tombstone
        // registration → vector deletes, keeping WAL order identical to memory
        // order. The split-lock version had two confirmed break windows (found in
        // OCR review): a concurrent same-id upsert could complete (200 acked)
        // between evaluation and tombstoning, then be destroyed by the stale
        // match set; and a delete record could precede the upsert record in the
        // WAL while its tombstone landed after the upsert's scalar row — replay
        // (delete→upsert) resurrects an id whose pre-crash delete was
        // acknowledged, so recovered state diverges from acknowledged state.
        let write_guard = col
            .begin_write()
            .await
            .map_err(|e| collection_deleted_conflict(&name, e))?;
        // 建集合记录先于本删除落盘（新建竞态兜底，见 ensure_create_logged）。
        // EN: the create record precedes this delete (create-race backstop; see
        // ensure_create_logged).
        crate::collection::ensure_create_logged(&state.collections_wal, &name, &col)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
        // 匹配集求值在标量锁内完成；随后 WAL 落盘 → 重加锁登记墓碑 →
        // 删向量（墓碑先于向量删除，避免「向量没了、payload 还在」的半删除）。
        // EN: Match evaluation runs under the scalar lock; then WAL append →
        // re-lock to register tombstones → vector deletes (tombstones before
        // vector deletes, avoiding a half-deleted "vector gone, payload alive"
        // state).
        {
            let scalars = col
                .scalars
                .lock()
                .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            let mut matched: Option<Vec<u64>> = None;
            for (filter, predicate) in req.filters.iter().zip(predicates.iter()) {
                let hits = scalars.matching_ids(&filter.column, predicate);
                matched = intersect_opt(matched, &hits);
            }
            if let Some(hits) = matched {
                ids.extend(hits);
            }
        }
        // WAL-first：删除向量（不可逆步骤）之前先落盘；墓碑是内存态，重放时按
        // WAL 顺序重建。write_lock 保证 WAL 顺序与内存顺序一致（见上方）。
        // EN: WAL-first — persist before the irreversible vector deletes;
        // tombstones are in-memory state rebuilt from the WAL on replay. The
        // write_lock keeps WAL order identical to memory order (see above).
        let delete_lsn = if !ids.is_empty() {
            let payload = houyi_persistence::collection_wal::encode_collection_delete(&name, &ids)
                .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
            let entry = state
                .collections_wal
                .append(payload)
                .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            let mut scalars = col
                .scalars
                .lock()
                .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            for id in &ids {
                scalars.tombstone(*id);
            }
            Some(entry.lsn)
        } else {
            None
        };
        // 同 id 的 delete/upsert 并发由 write_lock 串行化：后拿到锁的一方看到
        // 前者的完整结果（墓碑/新行），WAL 顺序即锁顺序，重放收敛于同一终态。
        // EN: Concurrent same-id delete/upsert is serialized by the write_lock:
        // the later lock holder observes the former's complete result
        // (tombstone/new row), WAL order equals lock order, and replay converges
        // to the same final state.
        let mut deleted = 0usize;
        for id in &ids {
            match col.backend.delete(*id).await {
                Ok(true) => deleted += 1,
                Ok(false) => {}
                Err(e) => return Err((axum::http::StatusCode::BAD_REQUEST, e)),
            }
        }
        if let Some(lsn) = delete_lsn {
            col.note_applied_lsn(lsn);
        }
        // sync 等待在 write_lock 释放后（纯观测，不占锁，见 sync_wait_if_required）；
        // 过滤未命中时无 WAL 记录，无需等待。级别锁内捕获（同 upsert 纪律）。
        // EN: Sync wait after releasing the lock; no WAL record → nothing to wait on.
        let ha_level = col.ha_level();
        drop(write_guard);
        if let Some(lsn) = delete_lsn {
            sync_wait_if_required(&state.replication_metrics, &col, ha_level, lsn).await;
        }
        return Ok(Json(DeleteResponse { deleted }));
    }
    // 仅显式 ids 路径同样 WAL-first（先于不可逆的向量删除），且与过滤路径
    // 共用 write_lock：WAL 顺序与内存顺序一致（同 id 并发 upsert/delete 的
    // 重放终态 = 已确认终态）。
    // EN: The explicit-ids path is WAL-first too (ahead of the irreversible
    // vector deletes) and shares the write_lock with the filter path: WAL order
    // matches memory order (replay of concurrent same-id upsert/delete converges
    // to the acknowledged final state).
    let write_guard = col
        .begin_write()
        .await
        .map_err(|e| collection_deleted_conflict(&name, e))?;
    // 建集合记录先于本删除落盘（新建竞态兜底，见 ensure_create_logged）。
    // EN: the create record precedes this delete (create-race backstop; see
    // ensure_create_logged).
    crate::collection::ensure_create_logged(&state.collections_wal, &name, &col)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let delete_entry = {
        let payload = houyi_persistence::collection_wal::encode_collection_delete(&name, &ids)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
        state
            .collections_wal
            .append(payload)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };
    // 仅显式 ids 路径：先删向量再记标量墓碑——不支持 delete 的后端在首个调用即
    // 报错返回，标量页不被污染（不会出现「向量还在、payload 却消失」的半删除）。
    // EN: Explicit-ids-only path — vectors first, scalar tombstones second: a
    // backend that rejects deletes errors on the very first call, before any
    // tombstone is registered, so the scalar page stays untouched (no half-deleted
    // state where the vector survives but its payload vanished).
    let mut deleted = 0usize;
    let mut removed_ids = Vec::new();
    for id in &ids {
        match col.backend.delete(*id).await {
            Ok(true) => {
                deleted += 1;
                removed_ids.push(*id);
            }
            Ok(false) => {}
            Err(e) => return Err((axum::http::StatusCode::BAD_REQUEST, e)),
        }
    }
    if !removed_ids.is_empty() {
        let mut scalars = col
            .scalars
            .lock()
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        for id in removed_ids {
            scalars.tombstone(id);
        }
    }
    col.note_applied_lsn(delete_entry.lsn);
    // sync 等待在 write_lock 释放后（与 upsert 同纪律，级别锁内捕获）。
    // EN: The sync wait starts after write_lock is released (same as upsert; the
    // level is captured inside the lock).
    let ha_level = col.ha_level();
    drop(write_guard);
    sync_wait_if_required(&state.replication_metrics, &col, ha_level, delete_entry.lsn).await;
    Ok(Json(DeleteResponse { deleted }))
}

/// `DELETE /v1/admin/collections/{name}`：整库删除。
/// EN: Drop an entire collection.
///
/// 无条件删除（不要求 collection 为空），不可逆——仅 RBAC admin 可调用，审计中间件
/// 自动记录 actor/route/status。流程是 WAL-first 复合写：先落 `collection_drop`
/// 再标记/摘除内存；崩溃后冷重放与 follower 实时 apply 走同一
/// [`crate::recovery::apply_collection_wal_record`] 完成删除，语义不会分叉。
/// EN: Unconditional and irreversible (RBAC admin only, audited by the global
/// middleware). WAL-first: the `collection_drop` record lands before the in-memory
/// mark/removal, so crash replay and follower live apply converge through the same
/// apply function.
///
/// 状态码 / status codes:
/// - 200 删除成功；404 不存在（含重复删除）；
/// - 409 并发删除竞争（collection 已在删除中）；
/// - 403 非 admin（RBAC 中间件）；503 follower/只读/租约 fencing。
///
/// EN: 200 deleted; 404 unknown (including repeat delete); 409 concurrent
/// deletion; 403 non-admin (RBAC middleware); 503 follower/read-only/lease fencing.
pub async fn handle_admin_collection_delete(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<DropCollectionResponse>, (axum::http::StatusCode, String)> {
    drop_collection(&state, name).await
}

/// `DELETE /v1/admin/collections/compact`：删除**名为 `compact` 的 collection**。
/// EN: Delete a collection literally named `compact`.
///
/// WHY 单独一个 handler：axum/matchit 的静态段优先于 `{name}`，上面的
/// `/v1/admin/collections/compact`（快照/压实管理端点）会先匹配，参数路由永远
/// 收不到这个 DELETE；不补这个 handler，名为 `compact` 的 collection 无法删除。
/// EN: matchit prefers static segments over `{name}`, so the admin compact route
/// above shadows the parameterized DELETE. Without this handler a collection
/// named `compact` could never be dropped.
pub async fn handle_admin_collection_delete_compact(
    State(state): State<BenchmarkHttpState>,
) -> Result<Json<DropCollectionResponse>, (axum::http::StatusCode, String)> {
    drop_collection(&state, "compact".to_string()).await
}

/// 整库删除的共享实现（参数路由与 `compact` 静态路由共用）。
/// EN: Shared drop implementation for the parameterized and static `compact` routes.
async fn drop_collection(
    state: &BenchmarkHttpState,
    name: String,
) -> Result<Json<DropCollectionResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(state)?;
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;
    // 删除期间持 write_lock：已进入的写先完成；随后的写由 begin_write 双检拒绝
    // （见 [`collection_deleted_conflict`]）。
    // EN: hold write_lock for the deletion so in-flight writes finish first and
    // subsequent writers are rejected by begin_write's double check.
    let write_guard = col
        .begin_write()
        .await
        .map_err(|e| collection_deleted_conflict(&name, e))?;
    // 建集合记录先于 drop 落盘：同名重建但 create 尚未落盘的竞态下，补落后再删，
    // 重放顺序 create→drop 收敛为「不存在」。（缺 create 的 drop 会被重放跳过，
    // 但 create 若随后落盘会复活 collection，故必须补齐。）
    // EN: the create record precedes the drop; across a same-name recreate race the
    // missing create is logged first, so replay sees create→drop (absent) instead
    // of skipping the drop and later resurrecting the collection via its create.
    crate::collection::ensure_create_logged(&state.collections_wal, &name, &col)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // WAL-first：先落 drop 记录（崩溃后重放可完成删除），再标记并摘除内存。
    // EN: WAL-first — the drop record is durable before the in-memory mark/removal.
    let payload = houyi_persistence::collection_wal::encode_collection_drop(&name)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let entry = state
        .collections_wal
        .append(payload)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    col.mark_deleted();
    // 只摘除同一个 Arc：并发同名重建得到的新 collection 不会被旧删除误摘。
    // EN: remove only this Arc — a concurrent same-name rebuild must survive.
    state.collections.remove_if_same(&name, &col);
    // sync 等待在锁外（与 upsert 同纪律）；级别在锁内捕获，col 仍由本作用域持有。
    // EN: the sync wait runs after releasing the lock (same discipline as upsert);
    // the level is captured under it and `col` stays alive in this scope.
    let ha_level = col.ha_level();
    drop(write_guard);
    sync_wait_if_required(&state.replication_metrics, &col, ha_level, entry.lsn).await;
    Ok(Json(DropCollectionResponse {
        deleted: true,
        collection: name,
    }))
}

/// 设置 Collection 的 HA 级别（P4.4.2）/ Set a collection's HA level.
///
/// WAL-first：先落 `collection_set_ha_level` 记录再改内存 AtomicU8——宕机后冷启动
/// 重放恢复到同一级别；重放幂等（同值重复 store 无副作用）。写互斥与 upsert/delete
/// 同一把 write_lock：HA 变更与数据写共用「WAL append + 内存变更」复合写纪律，
/// 保证 WAL 顺序与内存顺序一致（65b 的 sync 等待按 WAL LSN 判定级别，顺序分叉会让
/// 等待用到过期级别）。
/// EN: WAL-first — the `collection_set_ha_level` record lands before the in-memory
/// AtomicU8 store, so crash + cold replay restores the same level; replay is
/// idempotent. Shares the write_lock with upsert/delete (same compound-write
/// discipline) so WAL order matches memory order — the 65b sync wait reads the
/// level by WAL LSN and a reordered level would corrupt that decision.
/// follower 上 503（只读守卫）：级别变更只允许经复制流 apply 到达。
/// EN: 503 on followers — level changes arrive only via the replication stream.
pub async fn handle_collection_set_ha_level(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(req): Json<SetHaLevelRequest>,
) -> Result<Json<SetHaLevelResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    // 校验先于任何副作用：collection 缺失 404、级别笔误 400——与 delete 端点
    // 「预检必须先于 WAL 记录」同纪律，坏请求不得在 WAL 留下记录。
    // EN: Validation precedes every side effect (404 unknown collection, 400 typo)
    // — same discipline as delete: a rejected request must not reach the WAL.
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;
    let level = houyi_persistence::ha_level::HaLevel::parse(&req.ha_level)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let _write = col
        .begin_write()
        .await
        .map_err(|e| collection_deleted_conflict(&name, e))?;
    // 建集合记录先于 HA 变更落盘（新建竞态兜底，见 ensure_create_logged）。
    // EN: the create record precedes this HA change (create-race backstop; see
    // ensure_create_logged).
    crate::collection::ensure_create_logged(&state.collections_wal, &name, &col)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // 编码器收级别码、内部渲染规范串——「落盘必为规范串」由签名保证。
    // EN: The encoder takes the level code and renders the canonical string
    // internally — the signature guarantees a canonical string lands on disk.
    let payload = houyi_persistence::collection_wal::encode_collection_set_ha_level(&name, level)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let entry = state
        .collections_wal
        .append(payload)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    col.set_ha_level(level);
    col.note_applied_lsn(entry.lsn);
    Ok(Json(SetHaLevelResponse {
        collection: name,
        ha_level: level.to_string(),
    }))
}

/// 手动 failover：`POST /v1/admin/promote`（P4.4.3，仅 follower；认证走全局
/// Bearer 中间件）。幂等：已是 leader（起始角色或 promoted）返回 409；
/// disabled 返回 400（本端点不是复制开关）。
/// EN: Manual failover promote — followers only; auth rides the global Bearer
/// middleware. Idempotent: an existing leader gets 409, disabled gets 400.
///
/// 步骤顺序即不变量（每步失败留下的状态都可归因）：
/// 1. 停 follower 复制线程（此后数据面不再被旧主流入）；
/// 2. drain 全部 collection 的 builder 队列（promote 前已 apply 的写全部转为
///    可检索，RPO 语义对 ANN 检索同样成立）；
/// 3. 起 ReplicationLeader（hub 复用，WAL 句柄零切换；watermarks 新建——
///    promote 前本机没有「被 ACK」的概念）；
/// 4. **先换指标视图后翻角色**：Leader 视图 + follower 角色的窗口里写仍 503、
///    sync 等待不可达（无害）；反过来的窗口会出现「写放行但水位线缺失」，
///    full_sync 写静默走 async——不可接受；
/// 5. 翻角色（只读即时解除）→ 200。
///
/// EN: The step order is the invariant: stop the follower thread, drain builder
/// queues, start the leader (hub reused, WAL handle untouched), swap the
/// metrics view BEFORE flipping the role (the reverse order opens a window
/// where full_sync writes silently degrade to async), then flip the role.
/// `POST /v1/admin/promote` 请求体（EE M1，可选）。同时给出 term 与 ttl_ms
/// 时先授予租约再切角色；bodyless 保持手动 promote 兼容，但节点开启
/// `require_leader_lease` 后 bodyless 请求被拒绝（否则手动路径可绕过 fencing）。
/// EN: optional promote body (EE M1). Supplying term + ttl_ms grants the lease
/// before the role flip; a bodyless request stays compatible with manual
/// promote, but is rejected when `require_leader_lease` is enabled.
#[derive(Debug, Default, Deserialize)]
pub struct PromoteRequest {
    pub term: Option<u64>,
    pub ttl_ms: Option<u64>,
}

fn lease_error_response(e: crate::cluster::LeaseError) -> (axum::http::StatusCode, String) {
    use axum::http::StatusCode;
    match e {
        crate::cluster::LeaseError::TermRegression { .. } => (StatusCode::CONFLICT, e.to_string()),
        crate::cluster::LeaseError::InvalidTerm => (StatusCode::BAD_REQUEST, e.to_string()),
        crate::cluster::LeaseError::InvalidTtl { .. } => (StatusCode::BAD_REQUEST, e.to_string()),
        crate::cluster::LeaseError::Io(ref inner) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("lease io: {inner}"),
        ),
        other => (StatusCode::CONFLICT, other.to_string()),
    }
}

/// 前置校验虽只读 term，但需等待 store 的 Mutex——可能被另一线程的 fsync
/// 持有，因此同样经 spawn_blocking 桥接。
/// EN: the pre-check only reads the term, but waits on the store Mutex, which
/// another thread may hold across an fsync — bridge it the same way.
async fn validate_grant_blocking(
    cluster: Arc<crate::cluster::ClusterControl>,
    term: u64,
    ttl_ms: u64,
) -> Result<(), (axum::http::StatusCode, String)> {
    tokio::task::spawn_blocking(move || cluster.validate_grant(term, ttl_ms))
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("lease validation task failed: {e}"),
            )
        })?
        .map_err(lease_error_response)
}

/// term 前进时 `observe_term` 会原子写 + fsync，属阻塞 IO：经 spawn_blocking
/// 桥接，不占 tokio worker（与复制会话的 wal.fsync 同纪律）。
/// EN: advancing the term fsyncs atomically — bridge via spawn_blocking so the
/// runtime worker never carries the blocking write.
async fn grant_lease_blocking(
    cluster: Arc<crate::cluster::ClusterControl>,
    term: u64,
    ttl_ms: u64,
) -> Result<crate::cluster::LeaseGrant, (axum::http::StatusCode, String)> {
    tokio::task::spawn_blocking(move || cluster.grant(term, ttl_ms))
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("lease grant task failed: {e}"),
            )
        })?
        .map_err(lease_error_response)
}

/// 同上的 demote 版本：先 fencing term（可能 fsync）再撤销租约。
/// EN: demote counterpart — fence the term (possibly fsync), then revoke.
async fn fence_and_revoke_blocking(
    cluster: Arc<crate::cluster::ClusterControl>,
    term: Option<u64>,
) -> Result<u64, (axum::http::StatusCode, String)> {
    tokio::task::spawn_blocking(move || cluster.fence_and_revoke(term))
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("demote task failed: {e}"),
            )
        })?
        .map_err(lease_error_response)
}

/// 从可选请求体解析 (term, ttl_ms)，要求成对出现。
/// EN: parse the optional body into a (term, ttl_ms) pair; both or neither.
fn parse_lease_pair(
    body: Option<Json<PromoteRequest>>,
) -> Result<Option<(u64, u64)>, (axum::http::StatusCode, String)> {
    match body.map(|Json(req)| req) {
        None => Ok(None),
        Some(PromoteRequest {
            term: None,
            ttl_ms: None,
        }) => Ok(None),
        Some(PromoteRequest {
            term: Some(term),
            ttl_ms: Some(ttl_ms),
        }) => Ok(Some((term, ttl_ms))),
        Some(_) => Err((
            axum::http::StatusCode::BAD_REQUEST,
            "promote body must supply both term and ttl_ms (or neither)".to_string(),
        )),
    }
}

pub async fn handle_admin_promote(
    State(state): State<BenchmarkHttpState>,
    body: Option<Json<PromoteRequest>>,
) -> Result<Json<PromoteResponse>, (axum::http::StatusCode, String)> {
    match state.role.get() {
        ReplicationRole::Leader => {
            return Err((
                axum::http::StatusCode::CONFLICT,
                "already leader (promote is idempotent, not a no-op)".to_string(),
            ));
        }
        ReplicationRole::Disabled => {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                "replication is disabled on this node".to_string(),
            ));
        }
        ReplicationRole::Follower => {}
    }
    let lease = parse_lease_pair(body)?;
    // 开启租约要求后，手动 promote 不得绕过 fencing——必须带 term/ttl。
    // EN: with leases required, a manual promote must not bypass fencing.
    if state.cluster.lease_required() && lease.is_none() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "require_leader_lease is enabled: promote body {term, ttl_ms} is required".to_string(),
        ));
    }
    // 拆除 follower 之前先做可预测校验（无 IO，仅锁内读 term）：陈旧 term /
    // term=0 / 越界 TTL 必须在节点仍是健康 follower 时被拒绝。真正的授权检查
    // 仍在 promote 门内（并发 demote 仍可能推进 term）。
    // EN: predictable validation before tearing the follower down (no IO, only
    // a locked term read); the authoritative check still runs under the gate.
    if let Some((term, ttl_ms)) = lease {
        validate_grant_blocking(Arc::clone(&state.cluster), term, ttl_ms).await?;
    }
    let ctx = state.promote.clone().ok_or_else(|| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "promote context not assembled".to_string(),
        )
    })?;
    // P4.5.5：复制层已 tokio 化——leader bind 与句柄关停都是 async，直接
    // await 即可，不再需要 spawn_blocking + Handle::block_on 的桥接（旧桥接
    // 在 async 上下文里 block_on 本身就是隐患）。
    // EN: P4.5.5: the replication layer is now tokio-native — leader bind and
    // handle shutdown are async, so await directly; the old spawn_blocking +
    // Handle::block_on bridge is gone (block_on inside async was itself a hazard).
    promote_to_leader(state, ctx, lease).await
}

async fn promote_to_leader(
    state: BenchmarkHttpState,
    ctx: Arc<PromoteContext>,
    lease: Option<(u64, u64)>,
) -> Result<Json<PromoteResponse>, (axum::http::StatusCode, String)> {
    // promote 互斥：tokio Mutex 可跨 await 持有（P4.5.5 起 promote 内部要
    // 关停 follower、起 leader，都是 async）。`handles` 的 std 锁只在同步段
    // 短暂交接句柄，绝不跨 await——否则 handler future 不再是 Send。
    // EN: the tokio gate serializes promotes; the std handles lock is only
    // held synchronously (a guard across `.await` would make the future !Send).
    let _gate = ctx.promote_gate.lock().await;
    // 门内复核：demote 也走同一把门，因此在拆除 follower 之前这里的校验结果
    // 不会被并发 demote 推进的 term 推翻。
    // EN: re-validate under the gate; demote shares this gate, so the check
    // cannot be invalidated by a concurrent term advance before teardown.
    if let Some((term, ttl_ms)) = lease {
        validate_grant_blocking(Arc::clone(&state.cluster), term, ttl_ms).await?;
    }
    if state.role.get() == ReplicationRole::Leader {
        return Err((
            axum::http::StatusCode::CONFLICT,
            "already leader (concurrent promote won the race)".to_string(),
        ));
    }
    // 1. 停 follower 复制 task（abort + 5s 收尾预算；挂死只会 warn）。
    // EN: 1. Stop the follower replication task (abort + 5s shutdown budget; a hang
    // only warns).
    let follower = ctx
        .handles
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .follower
        .take();
    if let Some(f) = follower {
        f.shutdown().await;
    }
    // 1.5) 尾部补齐（P5.0.4）：本地 WAL 可能包含「已接收但未 apply」的尾部
    //      （apply 失败 / 关停竞争）。带着缺口 promote 会让新 leader 回 ACK
    //      一份自己内存里并不存在的历史。逐 collection 只重放 applied_lsn 之后
    //      的记录；任何解码/apply 失败都拒绝 promote（503），人工处置。
    // EN: Tail replay (P5.0.4): the local WAL may hold received-but-unapplied
    // entries. Promoting with a gap would ACK history the new leader does not
    // hold in memory. Replay only records above each collection's applied_lsn;
    // any decode/apply failure rejects the promote (503) for manual action.
    let wal_for_replay = Arc::clone(&ctx.collections_wal);
    let entries = tokio::task::spawn_blocking(move || wal_for_replay.replay_entries())
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("promote tail replay join failed: {e}"),
            )
        })?
        .map_err(|e| {
            (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!("promote tail replay cannot read local WAL: {e}"),
            )
        })?;
    let mut tail = Vec::new();
    for entry in entries {
        let record = match houyi_persistence::collection_wal::decode_collection_wal_payload(
            &entry.payload,
        ) {
            Ok(Some(record)) => record,
            Ok(None) => continue,
            Err(e) => {
                return Err((
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "promote refused: collections WAL record at lsn {} is corrupt: {e}",
                        entry.lsn
                    ),
                ));
            }
        };
        let applied = state
            .collections
            .get(record.collection_name())
            .map(|col| col.applied_lsn())
            .unwrap_or(0);
        if entry.lsn > applied {
            tail.push(entry);
        }
    }
    if !tail.is_empty() {
        let replayed = tail.len();
        crate::recovery::replay_collections_wal_into_manager(
            &tail,
            &state.collections,
            &std::collections::HashMap::new(),
        )
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!("promote refused: tail replay failed ({replayed} records): {e}"),
            )
        })?;
        tracing::info!(replayed, "promote: replayed local WAL tail");
    }
    // 2. drain 全部 collection 的 builder 队列（列表快照后逐个 drain；
    //    follower task 已停，期间无新 apply 进队列——leader 角色的直接写尚未
    //    放行，队列是静止的）。
    // EN: 2. Drain every collection's builder queue (snapshot the list, then drain
    // one by one; the follower task is stopped, so no new applies arrive — direct
    // writes for the leader role are not allowed yet, so the queues are quiescent).
    for name in state.collections.list() {
        if let Some(col) = state.collections.get(&name) {
            crate::recovery::drain_pending_builder(&col.backend).await;
        }
    }
    // 2.5 租约先行（EE M1）：先持久化 term + 设 deadline，再起 leader。若
    //     leader 启动失败，租约随进程内 deadline 自然过期（角色仍是 follower，
    //     写路径依旧拒绝）；term 只增，失败不会让 sentinel 拿到回退的 term。
    // EN: 2.5 lease first (EE M1) — persist the term and set the deadline before
    // starting the leader. If startup fails the lease self-expires; the role is
    // still follower, and the term never regresses.
    let granted_lease = match lease {
        Some((term, ttl_ms)) => {
            Some(grant_lease_blocking(Arc::clone(&state.cluster), term, ttl_ms).await?)
        }
        None => None,
    };
    // 3. 起 leader：hub 复用（follower 的 WAL 写线程本就向它 publish，WAL 句柄
    //    零切换）；watermarks 全新——promote 前本机无人 ACK 它，旧 follower 的
    //    进度语义不能冒充 leader 水位线。
    // EN: 3. Start the leader: reuse the hub (the follower's WAL writer already
    // publishes to it, so the WAL handle never switches); brand-new watermarks —
    // nobody ACKed this node before promote, so old follower progress must not
    // masquerade as leader watermarks.
    let watermarks = Arc::new(houyi_persistence::replication_leader::AckWatermarks::new());
    let wal_for_next = Arc::clone(&ctx.collections_wal);
    let leader_next_lsn: Arc<dyn Fn() -> u64 + Send + Sync> =
        Arc::new(move || wal_for_next.next_lsn());
    // 先设租约再起 leader（P5.0.4 OCR）：起 leader 到设租约之间，角色仍是
    // follower，周期保留会按本地 manifest 删除旧段——恰在此刻握手的旧 follower
    // 会被误判 BEHIND_RETAINED。租约是进程内状态，启动失败也会自然过期。
    // EN: set the lease before starting the leader (P5.0.4 OCR): between leader
    // start and lease creation the role is still follower, so periodic retention
    // could delete old segments and force a handshaking follower into a full
    // resync. The lease is process-local and self-expires if startup fails.
    if ctx.follower_grace_secs > 0 {
        ctx.lifecycle.hold_retention(ctx.follower_grace_secs);
    }
    let wal_for_first = Arc::clone(&ctx.collections_wal);
    let leader_first_available: Arc<dyn Fn() -> std::io::Result<u64> + Send + Sync> =
        Arc::new(move || wal_for_first.first_available_lsn());
    let leader = houyi_persistence::replication_leader::ReplicationLeader::start(
        &ctx.listen_addr,
        Arc::clone(&ctx.hub),
        ctx.collections_wal_path.clone(),
        Arc::clone(&leader_next_lsn),
        Arc::clone(&watermarks),
        // 认证随 promote 延续（OCR 批次二十八 HIGH）：统一走生产构造函数，
        // 不再靠 `..Default::default()` 手工补 token。
        // EN: token survives promote via the single production constructor.
        houyi_persistence::replication_leader::LeaderSessionConfig::for_replication(
            ctx.queue_batches,
            ctx.replication_auth_token.clone(),
            ctx.encryption.clone(),
        )
        .with_retention(leader_first_available, ctx.lifecycle.retention_gate()),
    )
    .await
    .map_err(|e| {
        // 失败时角色未翻、视图未换：本机仍是只读（follower 线程已停，复制
        // 断流）——响亮降级态，运维按错误处置，不会出现「以为可写实际裸奔」。
        // EN: On failure the role is unchanged and the view is not swapped: the
        // node stays read-only (follower thread stopped, replication cut) — a loud
        // degraded state for ops to handle, never "believed writable while open".
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("replication leader start failed (node stays read-only): {e}"),
        )
    })?;
    tracing::info!(
        listen_addr = %ctx.listen_addr,
        bound = ?leader.local_addr().ok(),
        "promote: replication leader ready"
    );
    // leader 一旦存活必须即刻对关停序列可见（OCR 16a7ae9[2]）：视图/角色
    // 翻转与句柄登记之间留 panic 窗口会让运行中的 leader 无法被关停。
    // EN: A live leader must immediately be visible to the shutdown sequence
    // (OCR 16a7ae9[2]): a panic window between view/role flips and handle
    // registration would leave a running leader that cannot be shut down.
    ctx.handles
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .leader = Some(leader);
    // 4. 先换视图（sync 等待的水位线/hub 即刻就位）。
    // EN: 4. Swap the view first (watermarks/hub for sync waits become available
    // immediately).
    *state.replication_metrics.write().unwrap() = Some(ReplicationMetricsView::Leader {
        hub: Arc::clone(&ctx.hub),
        watermarks,
        leader_next_lsn,
        sync_timeout: ctx.sync_timeout,
        sync_quorum: ctx.sync_quorum,
    });
    // 5. 翻角色：只读即时解除，写路径与 sync 等待同步生效（Release 存储与
    //    读侧 Acquire 配对，见 types.rs RoleState——视图发布先于角色可见）。
    // EN: 5. Flip the role: read-only lifts immediately and write paths plus sync
    // waits take effect together (Release store paired with Acquire reads, see
    // types.rs RoleState — the view is published before the role becomes visible).
    state.role.set(ReplicationRole::Leader);
    let next_lsn = ctx.collections_wal.next_lsn();
    tracing::info!(next_lsn, term = state.cluster.term(), "promoted to leader");
    Ok(Json(PromoteResponse {
        promoted: true,
        next_lsn,
        term: state.cluster.term(),
        lease: granted_lease,
    }))
}

// ── Sentinel 控制面（EE M1）──────────────────────────────────
// EN: Sentinel control plane (EE M1).

/// `POST /v1/admin/lease` 请求体：授予/续租。 EN: Lease grant/renew request.
#[derive(Debug, Deserialize)]
pub struct LeaseRequest {
    pub term: u64,
    pub ttl_ms: u64,
}

/// 授予或续租：term 不得回退；TTL 受硬上限约束。
/// EN: grant or renew; the term never regresses and the TTL is capped.
pub async fn handle_admin_lease_grant(
    State(state): State<BenchmarkHttpState>,
    Json(req): Json<LeaseRequest>,
) -> Result<Json<crate::cluster::LeaseGrant>, (axum::http::StatusCode, String)> {
    grant_lease_blocking(Arc::clone(&state.cluster), req.term, req.ttl_ms)
        .await
        .map(Json)
}

/// `GET /v1/admin/lease`：当前 term/租约状态与上限。
/// EN: current term/lease status and limits.
pub async fn handle_admin_lease_status(
    State(state): State<BenchmarkHttpState>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "term": state.cluster.term(),
        "lease": state.cluster.status(),
        "require_leader_lease": state.cluster.lease_required(),
        "max_ttl_ms": state.cluster.max_ttl_ms(),
    }))
}

/// `DELETE /v1/admin/lease`：撤销租约（不改角色；角色切换由 demote 完成）。
/// EN: revoke the lease without changing the role (demote flips the role).
pub async fn handle_admin_lease_revoke(
    State(state): State<BenchmarkHttpState>,
) -> Json<serde_json::Value> {
    state.cluster.revoke();
    Json(serde_json::json!({"revoked": true, "term": state.cluster.term()}))
}

/// `POST /v1/admin/demote` 请求体（term 可选，用于 fencing 陈旧请求）。
/// EN: demote body (term optional; fences stale requests).
#[derive(Debug, Default, Deserialize)]
pub struct DemoteRequest {
    pub term: Option<u64>,
    /// 新 leader 的复制地址：给出时旧 leader 在 demote 后直接以 follower
    /// 身份连回，无需重启（M3 运行时转换）。
    /// EN: new leader's replication address; when present the demoted node
    /// immediately rejoins as a follower without a restart (M3).
    pub follower_of: Option<String>,
}

/// 降级结果。 EN: Demote result.
#[derive(Debug, serde::Serialize)]
pub struct DemoteResponse {
    pub demoted: bool,
    /// 调用前就已经是 follower（幂等重放）。 EN: True when already a follower.
    pub already: bool,
    pub role: &'static str,
    pub term: u64,
}

/// `POST /v1/admin/demote`：撤销租约 → 翻只读 → 停 leader 监听。
/// EN: revoke the lease, flip to read-only, stop the leader listener.
///
/// 两种归宿：带 `follower_of`（新主复制地址）时运行时转换回 follower 复制流
/// （M3）；不带则保持只读 `detached_readonly`，需要重启或以 follower 角色
/// 启动才能重新加入复制。顺序不可倒置：先翻角色（写立即 fence）再撤指标
/// 视图——反过来会出现「角色仍 leader 但水位线视图已消失」的窗口，full_sync
/// 写静默降级 async。
/// EN: two outcomes — with `follower_of` (the new leader's replication address)
/// the node converts back into a follower stream at runtime (M3); without it
/// the node stays read-only detached and needs a restart (or a follower start)
/// to rejoin replication. Ordering is load-bearing: flip the role first, then
/// clear the view; the reverse opens a window where full_sync writes silently
/// degrade to async.
pub async fn handle_admin_demote(
    State(state): State<BenchmarkHttpState>,
    body: Option<Json<DemoteRequest>>,
) -> Result<Json<DemoteResponse>, (axum::http::StatusCode, String)> {
    if state.role.get() == ReplicationRole::Disabled {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "replication is disabled on this node".to_string(),
        ));
    }
    let ctx = state.promote.clone().ok_or_else(|| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "promote context not assembled".to_string(),
        )
    })?;
    // follower_of 决定一次破坏性动作之后的归宿，必须在撤租约/停监听**之前**
    // 校验：畸形地址若等到会话启动才失败，节点已经降级且重试被占用。
    // EN: follower_of decides the outcome of a destructive action, so validate
    // it before revoking the lease or stopping the listener — a malformed
    // address that only fails session startup leaves the node demoted with its
    // retry slot occupied.
    let (requested_term, follower_of) = match body {
        Some(Json(request)) => {
            let follower_of = match request.follower_of {
                Some(raw) if !raw.trim().is_empty() => Some(validate_follower_of(&raw)?),
                _ => None,
            };
            (request.term, follower_of)
        }
        None => (None, None),
    };
    // 与 promote 共用同一把门：demote 与 promote 不得交错执行。
    // EN: share the promote gate so demote and promote never interleave.
    let _gate = ctx.promote_gate.lock().await;
    let term = fence_and_revoke_blocking(Arc::clone(&state.cluster), requested_term).await?;
    if state.role.get() != ReplicationRole::Leader {
        // 幂等重放。若上一次转换在启动复制会话时失败（角色已翻、会话没起来），
        // 这里允许用相同的 demote 请求重试，避免节点卡在「只读 follower」。
        // EN: idempotent replay. If a previous conversion flipped the role but
        // failed to start the replication session, the same demote request can
        // retry it instead of leaving the node stuck as a detached read-only.
        if let Some(leader_addr) = follower_of {
            let needs_start = ctx
                .handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .follower
                .is_none();
            if needs_start {
                start_follower_after_demote(&state, &ctx, leader_addr).await?;
                tracing::warn!(term, "retried the demote→follower conversion");
            } else {
                // 会话已存在时 follower_of 被忽略；纠正地址需要停掉会话后重来。
                // EN: follower_of is ignored when a session already exists;
                // correcting the address needs the session torn down first.
                tracing::warn!(
                    term,
                    ignored_address = %leader_addr,
                    "demote replay ignored follower_of: a follower session already exists"
                );
            }
        }
        return Ok(Json(DemoteResponse {
            demoted: true,
            already: true,
            role: "follower",
            term,
        }));
    }
    state.role.set(ReplicationRole::Follower);
    *state
        .replication_metrics
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    let leader = ctx
        .handles
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .leader
        .take();
    if let Some(leader) = leader {
        leader.shutdown().await;
    }
    if let Some(leader_addr) = follower_of {
        start_follower_after_demote(&state, &ctx, leader_addr).await?;
        tracing::warn!(
            term,
            "leader demoted and rejoined as a follower (runtime conversion)"
        );
    } else {
        tracing::warn!(
            term,
            "leader demoted to read-only (detached; restart to rejoin)"
        );
    }
    Ok(Json(DemoteResponse {
        demoted: true,
        already: false,
        role: "follower",
        term,
    }))
}

/// 校验 `follower_of` 与复制层 `split_host_port` 的可拨号契约一致：
/// `host:port` 或 `[ipv6]:port`，无 scheme/路径/空白/`@`，非空 host，
/// 端口 1–65535。未加括号的 IPv6（含多个冒号）一律拒绝——它会被
/// `rsplit_once` 误拆成不可解析的 host。畸形值在破坏性动作之前返回 400。
/// EN: validate `follower_of` against the replication layer's dialable
/// contract — `host:port` or `[ipv6]:port`, no scheme/path/whitespace/`@`,
/// non-empty host, port 1–65535. Unbracketed IPv6 (multiple colons) is
/// rejected because `rsplit_once` would split it into an unresolvable host.
/// Bad values return 400 before any destructive action.
fn validate_follower_of(raw: &str) -> Result<String, (axum::http::StatusCode, String)> {
    let addr = raw.trim();
    let bad = |why: &str| {
        (
            axum::http::StatusCode::BAD_REQUEST,
            format!("invalid follower_of {raw:?}: {why}"),
        )
    };
    if addr.is_empty() {
        return Err(bad("must not be empty"));
    }
    if addr.contains("://") || addr.contains('/') {
        return Err(bad("must be host:port without a scheme or path"));
    }
    if addr.contains(char::is_whitespace) || addr.contains('@') {
        return Err(bad("must not contain whitespace or userinfo"));
    }
    let port = if let Some(rest) = addr.strip_prefix('[') {
        let close = rest
            .find(']')
            .ok_or_else(|| bad("missing ']' in the IPv6 literal"))?;
        // 方括号里必须是真正的 IPv6 字面量：`[foo]`、`[[::1]`、`[::1::2]`
        // 这类值在复制层拆括号后无法解析，会陷入无限重试。
        // EN: the bracket content must be a real IPv6 literal — `[foo]`,
        // `[[::1]` or `[::1::2]` cannot resolve after the replication layer
        // strips the brackets and would retry forever.
        rest[..close]
            .parse::<std::net::Ipv6Addr>()
            .map_err(|_| bad("the bracketed host must be an IPv6 literal"))?;
        rest[close + 1..]
            .strip_prefix(':')
            .ok_or_else(|| bad("missing ':port' after the bracketed IPv6 host"))?
    } else {
        if addr.contains('[') || addr.contains(']') {
            return Err(bad("unbalanced brackets"));
        }
        let (host, port) = addr
            .rsplit_once(':')
            .ok_or_else(|| bad("must be host:port"))?;
        if host.contains(':') {
            return Err(bad("an IPv6 literal must be written as [addr]:port"));
        }
        if host.is_empty() {
            return Err(bad("host is empty"));
        }
        // getaddrinfo 只接受可打印 ASCII：控制字符/非 ASCII 会永久解析失败。
        // EN: getaddrinfo only accepts printable ASCII; control or non-ASCII
        // characters fail resolution forever.
        if !host.chars().all(|c| c.is_ascii_graphic()) {
            return Err(bad("host must be printable ASCII"));
        }
        port
    };
    match port.parse::<u16>() {
        Ok(port) if port != 0 => Ok(addr.to_string()),
        Ok(_) => Err(bad("port must be 1-65535")),
        Err(_) => Err(bad("port must be a numeric port (1-65535)")),
    }
}

/// demote 后的运行时 leader→follower 转换：复用启动期的 follower 身份、
/// WAL 句柄、apply sink 与快照 checkpoint，直接连回新主。
/// EN: runtime leader→follower conversion after demote — reuse the follower
/// identity, WAL handle, apply sink and snapshot checkpoint to reconnect.
async fn start_follower_after_demote(
    state: &BenchmarkHttpState,
    ctx: &Arc<PromoteContext>,
    leader_addr: String,
) -> Result<(), (axum::http::StatusCode, String)> {
    let wal_async = ctx.collections_wal.async_handle().ok_or_else(|| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "demote: collections WAL has no async handle".to_string(),
        )
    })?;
    let metrics = Arc::new(houyi_persistence::replication_follower::FollowerMetrics::default());
    let sink = Arc::new(crate::replication_apply::CollectionsApplySink::new(
        Arc::clone(&ctx.collections),
    ));
    let poison_skipped = sink.poison_skipped_handle();
    let follower = houyi_persistence::replication_follower::ReplicationFollower::start(
        leader_addr,
        wal_async,
        sink,
        Arc::clone(&metrics),
        houyi_persistence::replication_follower::FollowerConfig::for_replication(
            ctx.replication_auth_token.clone(),
            ctx.encryption.clone(),
        )
        .with_follower_identity(ctx.follower_id, Arc::clone(&ctx.snapshot_checkpoint)),
    )
    .map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("demoted to follower but replication session failed: {e}"),
        )
    })?;
    ctx.handles
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .follower = Some(follower);
    *state
        .replication_metrics
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
        Some(ReplicationMetricsView::Follower {
            metrics,
            poison_skipped,
        });
    Ok(())
}

/// 复制状态视图（`/v1/admin/replication/status` 与 diagnostics 共用）。
/// EN: replication status view shared by the status endpoint and diagnostics.
fn replication_status_json(
    state: &BenchmarkHttpState,
    first_available_lsn: Option<u64>,
) -> serde_json::Value {
    let role = match state.role.get() {
        ReplicationRole::Leader => "leader",
        ReplicationRole::Follower => "follower",
        ReplicationRole::Disabled => "disabled",
    };
    let mut body = serde_json::json!({
        "role": role,
        "term": state.cluster.term(),
        "lease": state.cluster.status(),
        "next_lsn": state.collections_wal.next_lsn(),
        "first_available_lsn": first_available_lsn,
    });
    let guard = state
        .replication_metrics
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match guard.as_ref() {
        Some(ReplicationMetricsView::Leader {
            hub,
            watermarks,
            leader_next_lsn,
            sync_quorum,
            ..
        }) => {
            let grace = state
                .promote
                .as_ref()
                .map(|ctx| std::time::Duration::from_secs(ctx.follower_grace_secs))
                .unwrap_or_default();
            let followers: Vec<serde_json::Value> = hub
                .progress_snapshot(grace)
                .into_iter()
                .map(|f| {
                    let follower_id = f
                        .follower_id
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>();
                    serde_json::json!({
                        "follower_id": follower_id,
                        "session_id": f.session_id,
                        "durable_lsn": f.durable_lsn,
                        "lag": leader_next_lsn().saturating_sub(f.durable_lsn),
                        "active": f.active,
                        "last_seen_ms": f.last_seen_ms,
                    })
                })
                .collect();
            body["leader"] = serde_json::json!({
                "next_lsn": leader_next_lsn(),
                "durable_watermark": watermarks.durable_lsn.load(std::sync::atomic::Ordering::Relaxed),
                "recv_watermark": watermarks.recv_lsn.load(std::sync::atomic::Ordering::Relaxed),
                "sync_quorum": sync_quorum,
                "active_followers": hub.active_follower_count(),
                "followers": followers,
                "slow_follower_kicks_total": hub.slow_follower_kicks_total(),
            });
        }
        Some(ReplicationMetricsView::Follower { metrics, .. }) => {
            body["follower"] = serde_json::json!({
                "recv_lsn": metrics.recv_lsn.load(std::sync::atomic::Ordering::Relaxed),
                "durable_lsn": metrics.durable_lsn.load(std::sync::atomic::Ordering::Relaxed),
                "leader_next_lsn": metrics
                    .leader_next_lsn
                    .load(std::sync::atomic::Ordering::Relaxed),
                "leader_first_available_lsn": metrics
                    .leader_first_available_lsn
                    .load(std::sync::atomic::Ordering::Relaxed),
                "lag": metrics
                    .leader_next_lsn
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .saturating_sub(
                        metrics
                            .recv_lsn
                            .load(std::sync::atomic::Ordering::Relaxed)
                    ),
                "behind_retained_rejections_total": metrics
                    .behind_retained_rejections_total
                    .load(std::sync::atomic::Ordering::Relaxed),
                "reconnects_total": metrics
                    .reconnects_total
                    .load(std::sync::atomic::Ordering::Relaxed),
            });
        }
        None => {}
    }
    body
}

/// `GET /v1/admin/replication/status`：角色/term/租约/水位线/follower 进度。
/// EN: role, term, lease, watermarks and follower progress.
pub async fn handle_admin_replication_status(
    State(state): State<BenchmarkHttpState>,
) -> Json<serde_json::Value> {
    // 段目录扫描是阻塞 IO：移出 runtime worker（与握手路径同纪律）。
    // EN: the segment directory scan is blocking IO — keep it off the runtime.
    let wal = Arc::clone(&state.collections_wal);
    let first_available = tokio::task::spawn_blocking(move || wal.first_available_lsn().ok())
        .await
        .ok()
        .flatten();
    Json(replication_status_json(&state, first_available))
}

/// `GET /v1/version`：版本与能力探测（EE 客户端用）。
/// EN: version and capability probe used by EE clients.
pub async fn handle_version() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "houyidb",
        "version": env!("CARGO_PKG_VERSION"),
        "api": "v1",
        "capabilities": ["lease", "demote", "quorum", "replication_status", "query_log",
                         "demote_follower_of", "collection_export", "collection_delete",
                         "collection_create", "collection_metric"],
    }))
}

/// `GET /v1/admin/whoami`：当前主体（认证关闭时标记匿名）。
/// EN: current principal (anonymous when auth is disabled).
pub async fn handle_admin_whoami(
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
) -> Json<serde_json::Value> {
    match principal {
        Some(axum::extract::Extension(p)) => Json(serde_json::json!({
            "authenticated": true,
            "name": p.name,
            "role": p.role,
            "collections": p.collections,
            "max_collections": p.max_collections,
        })),
        None => Json(serde_json::json!({
            "authenticated": false,
            "name": "anonymous",
            "role": "admin",
        })),
    }
}

/// `GET /v1/admin/diagnostics`：诊断包所需的结构化状态（不含密钥/向量）。
/// EN: structured state for diagnostic bundles (no secrets or vectors).
pub async fn handle_admin_diagnostics(
    State(state): State<BenchmarkHttpState>,
) -> Json<serde_json::Value> {
    let wal = Arc::clone(&state.collections_wal);
    let first_available = tokio::task::spawn_blocking(move || wal.first_available_lsn().ok())
        .await
        .ok()
        .flatten();
    let collection_count = state.collections.list().len();
    Json(serde_json::json!({
        "version": {
            "name": "houyidb",
            "version": env!("CARGO_PKG_VERSION"),
            "api": "v1",
        },
        "index": {
            "type": state.index.index_kind(),
            "distance_metric": state.index.distance_metric().as_str(),
            "hnsw_m": state.index.hnsw_m(),
            "hnsw_ef_construction": state.index.hnsw_ef_construction_value(),
            "vector_dim": state.vector_dim,
        },
        "collections": {
            "count": collection_count,
            "names": state.collections.list(),
        },
        "wal": {
            "next_lsn": state.collections_wal.next_lsn(),
            "first_available_lsn": first_available,
        },
        "replication": replication_status_json(&state, first_available),
        "security": {
            "audit_enabled": state.audit.is_some(),
            "encryption_at_rest": state.encryption.is_some(),
        },
        "query_log": {
            "enabled": state.query_log.is_some(),
            "capacity": state.query_log.as_ref().map(|q| q.capacity()),
        },
    }))
}

/// `GET /v1/admin/query-log`：最近查询（新→旧，可按条件过滤）。
/// EN: recent queries (newest first, filterable).
pub async fn handle_admin_query_log(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Query(query): axum::extract::Query<crate::query_log::QueryLogQuery>,
) -> Json<serde_json::Value> {
    match &state.query_log {
        Some(log) => Json(serde_json::json!({
            "enabled": true,
            "capacity": log.capacity(),
            "entries": log.recent(&query),
        })),
        None => Json(serde_json::json!({"enabled": false, "entries": []})),
    }
}

/// `GET /v1/admin/query-stats`：按 collection 的计数/错误/p50/p95/p99。
/// EN: per-collection counts, errors and latency percentiles.
pub async fn handle_admin_query_stats(
    State(state): State<BenchmarkHttpState>,
) -> Json<serde_json::Value> {
    match &state.query_log {
        Some(log) => {
            let stats = log.stats();
            Json(serde_json::json!({"enabled": true, "stats": stats}))
        }
        None => Json(serde_json::json!({"enabled": false})),
    }
}

/// 查询 Collection 状态
/// EN: Get collection status.
pub async fn handle_collection_status(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;
    let pending = col.backend.pending_build_count().await;
    let indexed = col.backend.indexed_count().await;
    // hot 缓冲里的点已进入内存但尚未进构图队列；只报 indexed+pending 的
    // 消费方（演练/对账脚本）会把它们误判为丢失。
    // EN: Hot-buffer points are in memory but not yet queued for building;
    // consumers that only sum indexed+pending (drills, reconcile scripts)
    // would misread them as lost.
    let hot = col.backend.hot_buffer_count().await;
    // 标量页：physical 行包含 stale/墓碑行；live/stale 按 id 去重后计算（status
    // 低频调用，允许 O(rows) 扫描；/metrics 不做同样扫描）。
    // EN: The scalar page — physical rows include stale/tombstoned rows; live/stale
    // are deduped by id (status is low-QPS and may scan; /metrics does not).
    let scalar_stats = col
        .scalars
        .lock()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .stats_snapshot();
    let last_compaction_unix = col
        .last_compaction_unix
        .load(std::sync::atomic::Ordering::Relaxed);
    let last_compaction_dropped = col
        .last_compaction_dropped
        .load(std::sync::atomic::Ordering::Relaxed);
    let mut body = serde_json::json!({
        "name": name,
        "hot_count": hot,
        "indexed_count": indexed,
        "pending_count": pending,
        "dims": col.dims,
        "metric": col.metric.as_str(),
        "index_type": col.index_type,
        "ha_level": col.ha_level().as_str(),
        "applied_lsn": col.applied_lsn(),
        "scalar_row_count": scalar_stats.physical_rows,
        "scalar_physical_rows": scalar_stats.physical_rows,
        "scalar_live_rows": scalar_stats.live_rows,
        "scalar_stale_rows": scalar_stats.stale_rows,
        "scalar_tombstone_ids": scalar_stats.tombstone_ids,
        "last_compaction_unix": last_compaction_unix,
        "last_compaction_dropped_rows": last_compaction_dropped,
    });
    // 非 hnsw 后端省略该字段（而非输出 null）。参数读后端当前值，运行期调参
    // 后此处立即反映生效值（OCR #19/#34）。
    // EN: Omit the field for non-hnsw backends instead of emitting null. The
    // parameters read the backend's current values, so runtime tuning shows up
    // here immediately (OCR #19/#34).
    if col.index_type == "hnsw" {
        let p = col.hnsw_params();
        if let Some(obj) = body.as_object_mut() {
            obj.insert(
                "hnsw".to_string(),
                serde_json::json!({
                    "m": p.m,
                    "ef_construction": p.ef_construction,
                    "ef_search": p.ef_search,
                }),
            );
        }
    }
    Ok(Json(body))
}

/// `GET /v1/collections/{name}/export`：流式 NDJSON 逻辑导出（备份/迁移）。
/// EN: streaming NDJSON logical export for backup/migration.
///
/// 行格式 / line format:
/// - header：`{"format":"houyidb-export","version":1,"collection":..,"dims":..,"metric":..,"index_type":..,"count":N}`
/// - 记录：`{"id":..,"vector":[..],"payload":{..}|null}`（按 id 升序）
/// - footer：`{"end":true,"count":N}`
///
/// 一致性 / consistency: 捕获阶段短暂持有集合写锁，向量表与 payload 页来自同一
/// 时刻；随后释放锁流式输出，不阻塞写路径。并发写入可能在此之后到达，调用方
/// 若需要严格时点快照，应先暂停写入（备份额度场景）。
/// EN: the capture phase briefly holds the collection write lock so vectors and
/// payloads come from one instant, then streams without the lock. Writes after
/// the capture are not included; pause writes for a strict point-in-time copy.
pub async fn handle_collection_export(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
) -> Result<axum::response::Response, (axum::http::StatusCode, String)> {
    if let Some(ext) = &principal {
        if !ext.0.may_see(&name) {
            return Err((
                axum::http::StatusCode::FORBIDDEN,
                "collection is not in the principal's scope".to_string(),
            ));
        }
    }
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;
    let (vectors, payloads) = {
        // 锁序与写路径一致：write_lock 先于 scalars。
        // EN: lock order matches writes: write_lock before scalars.
        let _write = col
            .begin_write()
            .await
            .map_err(|e| collection_deleted_conflict(&name, e))?;
        // 异步构建器不受 write_lock 保护：先把队列排空（等待在途合并结束），
        // 否则「从队列摘除、尚未并入 main」的行会在三层快照里丢失。
        // EN: the async builder is not covered by write_lock; drain it first
        // (waiting for any in-flight merge) so a batch drained from the queue
        // but not yet merged into main cannot vanish from the snapshot.
        drain_pending_builder(&col.backend).await;
        let vectors = col.backend.export_vectors_shared().await.map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("export vectors: {e}"),
            )
        })?;
        // 行数据在锁内取得（自有所有权），HashMap 重建放到锁外，缩短大页
        // 临界区。 / EN: rows are cloned under the lock; the map is built after.
        let rows = {
            let page = col.scalars.lock().map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("export payloads: {e}"),
                )
            })?;
            page.live_rows_last_wins()
        };
        let payloads: HashMap<u64, HashMap<String, houyi_core::ScalarValue>> =
            rows.into_iter().collect();
        (vectors, payloads)
    };
    let count = vectors.len();
    let header = serde_json::json!({
        "format": "houyidb-export",
        "version": 1,
        "collection": name,
        "dims": col.dims,
        "metric": col.metric.as_str(),
        "index_type": col.index_type,
        "count": count,
    });
    let mut header_line = serde_json::to_vec(&header).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("export header: {e}"),
        )
    })?;
    header_line.push(b'\n');
    let mut footer_line = serde_json::to_vec(&serde_json::json!({"end": true, "count": count}))
        .expect("static footer serializes");
    footer_line.push(b'\n');

    let header_stream =
        futures_util::stream::once(
            async move { Ok::<Bytes, std::io::Error>(Bytes::from(header_line)) },
        );
    let records = futures_util::stream::unfold(
        (vectors, payloads, 0usize, count),
        |(vectors, payloads, idx, count)| async move {
            if idx >= count {
                return None;
            }
            let (id, vector) = &vectors[idx];
            let payload = payloads.get(id).map(|row| {
                serde_json::Value::Object(
                    row.iter()
                        .map(|(key, value)| {
                            (key.clone(), crate::scalar_json::scalar_to_json(value))
                        })
                        .collect(),
                )
            });
            let record =
                serde_json::json!({"id": id, "vector": vector.as_ref(), "payload": payload});
            let item = match serde_json::to_vec(&record) {
                Ok(mut line) => {
                    line.push(b'\n');
                    Ok(Bytes::from(line))
                }
                Err(e) => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("export record {id}: {e}"),
                )),
            };
            Some((item, (vectors, payloads, idx + 1, count)))
        },
    );
    let footer_stream =
        futures_util::stream::once(
            async move { Ok::<Bytes, std::io::Error>(Bytes::from(footer_line)) },
        );
    let body = Body::from_stream(header_stream.chain(records).chain(footer_stream));
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        body,
    )
        .into_response())
}

/// 管理端点：对单个 collection 的标量页做内存压实。
/// EN: Admin endpoint: compact one collection's in-memory scalar page.
#[derive(Debug, serde::Deserialize)]
pub struct ScalarCompactRequest {
    /// true = 忽略 stale 比例阈值，强制重建。
    /// EN: true = ignore the stale-ratio threshold and force a rebuild.
    #[serde(default)]
    pub force: bool,
}

pub async fn handle_collection_compact(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(req): Json<ScalarCompactRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    // 压实只改内存、不改逻辑数据，但它是管理动作：follower 上仍按只读守卫拒绝，
    // 自动周期任务在 follower 内部独立运行。
    // EN: Compaction is an in-memory optimization but still an admin action; the
    // follower guard rejects it while the automatic task runs locally on followers.
    reject_if_read_only(&state)?;
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;
    // 非 force 请求沿用默认 50% 阈值；force 请求忽略阈值。
    // EN: Non-force requests use the default 50% threshold; force ignores it.
    let result = col
        .compact_scalars(req.force, 0.5)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({
        "collection": name,
        "physical_before": result.physical_before,
        "physical_after": result.physical_after,
        "dropped_rows": result.dropped_rows,
        "skipped_reason": result.skipped_reason,
    })))
}

/// 管理端点：完整快照周期（可选轮转）→ 保留。
/// EN: Admin endpoint: full snapshot cycle (optional rotation) -> retention.
pub async fn handle_admin_collections_compact(
    State(state): State<BenchmarkHttpState>,
    Json(req): Json<CollectionsCompactRequest>,
) -> Result<Json<CollectionsCompactResponse>, (axum::http::StatusCode, String)> {
    // follower 是只读副本；保留由 follower 自己的周期任务执行。
    // EN: Followers are read-only; retention runs in their own periodic task.
    reject_if_read_only(&state)?;
    let _guard = state.lifecycle.snapshot_lock.lock().await;
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&state.wal_path);
    // 租约必须先于快照生效：否则快照提交与下一轮保留 tick 之间会有一段
    // 未受保护的删除窗口。快照失败时清掉本次租约，不留下空窗口。
    // EN: The lease must take effect before the snapshot, otherwise a retention
    // tick between manifest commit and lease creation could delete unprotected
    // segments. A failed snapshot clears this request's lease.
    let previous_lease_secs = if req.hold_retention_secs.is_some() {
        state.lifecycle.lease_remaining_secs()
    } else {
        0
    };
    if let Some(secs) = req.hold_retention_secs {
        state.lifecycle.hold_retention(secs);
    }
    // 入口已拒 follower；仍按角色传门控，关闭「demote 与本轮快照并发」的窗口
    // （follower 的空名单 barrier 可能越过未 apply 的 create）。
    // EN: Followers are rejected at entry; the gate still follows the runtime
    // role to close the demote-vs-snapshot window (a follower's empty barrier
    // can outrun a not-yet-applied create).
    let allow_empty_manifest = !state.role.is_follower();
    let outcome = crate::collection_lifecycle::write_collection_snapshots(
        &state.collections,
        &state.collections_wal,
        &collections_wal_path,
        state.encryption.clone(),
        &state.graph_snapshot_metrics,
        allow_empty_manifest,
    )
    .await;
    let manifest_committed = outcome.manifest.is_some();
    if let Some(manifest) = &outcome.manifest {
        state
            .lifecycle
            .set_manifest_checkpoint(manifest.safe_checkpoint_lsn);
    }
    if !manifest_committed && req.hold_retention_secs.is_some() {
        // 恢复本请求之前的租约，而不是撤销更长的保护窗口。
        // EN: Restore the pre-existing lease instead of revoking a longer window.
        state.lifecycle.hold_retention(previous_lease_secs);
    }

    // 与复制握手互斥（P5.0.4）：轮转/保留期间不允许握手拿到旧的
    // first_available 与半删状态。读锁握手会在本写锁前排队。
    // EN: mutually exclusive with replication handshakes (P5.0.4): no handshake
    // may sample an old first-available or a half-deleted state.
    let gate = state.lifecycle.retention_gate();
    let _retention_write = gate.write().await;
    let rotated = if manifest_committed && req.rotate {
        let wal = Arc::clone(&state.collections_wal);
        tokio::task::spawn_blocking(move || wal.rotate_archive())
            .await
            .map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("rotate task join failed: {e}"),
                )
            })?
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .is_some()
    } else {
        false
    };

    let wal = Arc::clone(&state.collections_wal);
    let role = state.role.clone();
    let lifecycle = Arc::clone(&state.lifecycle);
    let retention = tokio::task::spawn_blocking(move || {
        crate::collection_lifecycle::run_retention_for_admin(
            &wal,
            &collections_wal_path,
            &role,
            &lifecycle,
        )
    })
    .await
    .map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("retention task join failed: {e}"),
        )
    })?;

    Ok(Json(CollectionsCompactResponse {
        manifest_committed,
        manifest_seq: outcome.manifest.as_ref().map(|m| m.manifest_seq),
        safe_checkpoint_lsn: outcome.manifest.as_ref().map(|m| m.safe_checkpoint_lsn),
        blocked_reason: outcome.blocked_reason,
        rotated,
        retention: RetentionResponse {
            deleted_segments: retention.deleted_segments,
            bytes_freed: retention.bytes_freed,
            first_available_lsn: retention.first_available_lsn,
            blocked_reason: retention.blocked_reason,
            lease_until_unix: state.lifecycle.lease_until_unix(),
        },
    }))
}

// ── 混合检索 / Hybrid search ────────────────────────────
// EN: ── Hybrid search ──────────────────────────────────

/// Collection 混合检索请求：向量近邻 + 可选结构化过滤。
/// Collection hybrid-search request: vector ANN plus optional structured filters.
#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    /// 查询向量，长度必须等于 Collection 维度，否则 400。
    /// Query vector; its length must equal the collection dims, else 400.
    pub vector: Vec<f32>,
    /// 返回上限 top-k。
    /// Top-k limit.
    pub k: usize,
    /// 结构化过滤条件；多条之间取交集；仅 hnsw/flat 后端支持（其余 400）。
    /// Structured filters (AND semantics); supported on hnsw/flat backends only.
    #[serde(default)]
    pub filters: Vec<HybridSearchFilter>,
    /// 检索响应是否携带每命中的 payload（与 `SearchRequest::include_payload` 同语义）。
    /// EN: Include per-hit payloads (same semantics as `SearchRequest::include_payload`).
    #[serde(default)]
    pub include_payload: bool,
    /// 可选 HNSW ef 覆盖（flat/暴力后端忽略）。
    /// Optional HNSW ef override (ignored by brute-force backends).
    #[serde(default)]
    pub ef_search: Option<usize>,
}

/// 混合检索（Collection 路由 + 维度校验）/ Hybrid search with collection routing + dim validation.
///
/// 过滤条件在该 Collection 的内存 PAX 标量页上求值：先得到匹配的页行号，再经
/// `row_to_id` 映射成向量 id，只看每个 id 的最后一行（append-only 重写产生的旧行
/// 不参与过滤），多个过滤条件之间取交集。标量页随 collections WAL 与快照持久化，
/// 重启后由重放/快照恢复。
/// EN: Filters run against the collection's in-memory PAX scalar page. Matched row
/// indices are mapped to vector ids via `row_to_id`, keeping only the last row per
/// id (stale rows from append-only re-upserts never match), then intersected.
/// The scalar page is persisted through the collections WAL and snapshots and
/// restored on restart.
pub async fn handle_collection_hybrid_search(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
    Json(req): Json<HybridSearchRequest>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    let started = std::time::Instant::now();
    let k = req.k;
    let filtered = !req.filters.is_empty();
    let result = collection_hybrid_search_inner(
        State(state.clone()),
        axum::extract::Path(name.clone()),
        principal.clone(),
        Json(req),
    )
    .await;
    if let Err((status, _)) = &result {
        record_query_log(
            &state,
            principal.as_ref().map(|ext| &ext.0),
            "hybrid_search",
            Some(&name),
            k,
            status.as_u16(),
            started,
            0,
            filtered,
        );
    }
    result
}

async fn collection_hybrid_search_inner(
    State(state): State<BenchmarkHttpState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
    Json(req): Json<HybridSearchRequest>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    let col = state.collections.get(&name).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("collection '{name}' not found"),
        )
    })?;

    // k 边界先于谓词构造/加锁/检索：与普通 collection 检索同口径，k==0 与超上界均 400。
    // EN: k boundary before predicate construction, locking or search — same contract
    // as plain collection search: zero and over-cap k both yield 400.
    if req.k == 0 {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "k must be >= 1 (use a positive top-k)".to_string(),
        ));
    }
    reject_search_limits(req.k, req.ef_search)?;

    // 维度 + 非有限分量校验（先于谓词构造与加锁）。
    // Dim + finiteness validation before predicate construction or any lock.
    validate_query_vector(&req.vector, col.dims, "query")?;

    // 边界校验先于任何锁/检索，并在同一遍历里把谓词构造好：op 白名单 + 字面值形状
    // 校验 + JSON→Scalar 转换只有 `filter_predicate` 这一个解释入口，持锁后直接复用，
    // 不再二次 match（旧实现两处解释操作符、应用侧靠 unreachable! 兜底，漂移即 panic；
    // 旧代码还对 "in" 非数组静默按「零命中」处理、对 gt+数组进入 compare 产生脏结果）。
    // EN: Validate and build every predicate before taking the lock. `filter_predicate`
    // is the single interpreter of the op whitelist, literal shapes and JSON→scalar
    // conversion; the under-lock path reuses it verbatim — no second match, no
    // `unreachable!`. The old code double-interpreted ops, treated a non-array `in`
    // as silent zero-match, and fed array/object/null into scalar compare.
    let predicates: Vec<ScalarPredicate> = if req.filters.is_empty() {
        Vec::new()
    } else {
        req.filters
            .iter()
            .map(|filter| filter_predicate(&filter.column, &filter.op, &filter.value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?
    };
    if !req.filters.is_empty() {
        // 仅 hnsw/flat 实现了 mask 下推；其余后端的 search 会忽略过滤集合，
        // 必须显式拒绝（fail-fast），不能静默返回未过滤结果。
        // EN: Only hnsw/flat honor the filter mask; other backends ignore it, so
        // refuse explicitly rather than silently returning unfiltered results.
        if col.index_type != "hnsw" && col.index_type != "flat" {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                "filtered hybrid search is unsupported for index_type '{}' (supported: hnsw, flat)",
                col.index_type
            ),
            ));
        }
    }

    // 结构化过滤：持锁仅覆盖「页过滤 + 行号→id 映射」，std MutexGuard 不跨 .await。
    // EN: Hold the per-collection lock only for page filtering + row→id mapping;
    // the std MutexGuard is released before the async index search (no guard across .await).
    let mut filter_ids: Option<Vec<u64>> = None;
    if !req.filters.is_empty() {
        let scalars = col
            .scalars
            .lock()
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        for (filter, predicate) in req.filters.iter().zip(predicates.iter()) {
            // 谓词由 filter_predicate 统一构造：过滤字面值与写入侧共用同一
            // JSON→ScalarValue 转换，修掉旧实现 Text("\"red\"") 的引号 bug。
            // EN: Predicates come from the single filter_predicate construction
            // point; filter literals share the write-side conversion, fixing the
            // old Text("\"red\"") double-quoting that made eq filters unmatchable.
            let ids = scalars.matching_ids(&filter.column, predicate);
            filter_ids = intersect_opt(filter_ids, &ids);
        }
    }

    // 与另两条检索路径同一纪律：4k 过取（带 mask）+ truncate 前剔除非有限距离。
    // EN: Same discipline as the other two search paths — 4k over-fetch (with the
    // mask) and drop non-finite distances before truncate.
    // 饱和乘法后的 `.max(req.k)` 守卫实际不可达（4k 恒 ≥ k），保留以对齐库内
    // `saturating_mul(n).max(k)` 惯例（见 houyi-vector 检索链路）。
    // EN: The `.max(req.k)` after saturating_mul is unreachable (4k ≥ k always); kept
    // deliberately to mirror the in-library `saturating_mul(n).max(k)` convention.
    let fetch_k = req.k.saturating_mul(4).max(req.k);
    // 慢查询计时（P4.3.2）：阈值 0 = 禁用，超阈值打 warn（不含向量本体）。
    // EN: Slow-query timing (P4.3.2): threshold 0 disables; over threshold emits a
    // warn (without the vector itself).
    let t0 = std::time::Instant::now();
    let pairs = col
        .backend
        .search_scored_with_ef_override(&req.vector, fetch_k, filter_ids.as_deref(), req.ef_search)
        .await;
    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let threshold = state
        .pgwire_metrics
        .slow_query_ms
        .load(std::sync::atomic::Ordering::Relaxed);
    if crate::types::slow_query_exceeded(elapsed_ms, threshold) {
        tracing::warn!(
            elapsed_ms,
            threshold_ms = threshold,
            collection = %name,
            k = req.k,
            filtered = filter_ids.is_some(),
            "slow http hybrid search"
        );
    }
    let pairs = finite_pairs_truncated_to_k(pairs, req.k);
    // 距离 → 分数按 collection 度量换算（cosine 钳制、其余不钳制；ip 取反）。
    // EN: distance → score by the collection metric (cosine clamps, others don't;
    // ip negates).
    let mut resp = scored_pairs_to_response(pairs, col.metric);
    resp.metric = Some(col.metric.as_str().to_string());
    if req.include_payload {
        attach_payloads(&col, &mut resp)?;
    }
    record_query_log(
        &state,
        principal.as_ref().map(|ext| &ext.0),
        "hybrid_search",
        Some(&name),
        req.k,
        200,
        t0,
        resp.ids.len(),
        filter_ids.is_some() || !req.filters.is_empty(),
    );
    Ok(Json(resp))
}

fn intersect_opt(prev: Option<Vec<u64>>, new: &[u64]) -> Option<Vec<u64>> {
    match prev {
        None => Some(new.to_vec()),
        Some(p) => {
            let set: std::collections::HashSet<u64> = new.iter().copied().collect();
            Some(p.into_iter().filter(|id| set.contains(id)).collect())
        }
    }
}

async fn benchmark_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

/// readiness 反压水位线（P4.0.5）：待构图队列超过该值即判 not_ready。
/// 经验值：单轮并行构图约 8k 点，64k ≈ 8 轮积压，新写入检索可见性延迟已达秒级。
/// Backpressure watermark for readiness: pending graph-build items beyond this
/// means ~8 build rounds of backlog and seconds of write-visibility lag.
// pub：bin target 的集成测试（tests.rs）派生超水位样本量，与 /metrics HELP
// 文案共用同一常量——水位线调参时测试与文档同步漂移，不会静默失真。
pub const READYZ_PENDING_WATERMARK: usize = 65536;

/// 就绪判定纯函数（抽出单测边界：≤ 水位线就绪）。
/// Pure readiness decision (unit-testable boundary: ready iff pending ≤ watermark).
fn readyz_is_ready(pending: usize, watermark: usize) -> bool {
    pending <= watermark
}

/// `/readyz` 就绪探针（P4.0.5）——与 `/v1/health`（liveness，进程活着即 ok）分离。
/// 就绪 = 索引可响应 且 待构图队列低于反压水位线；超水位 503，K8s 摘流让积压排空。
/// Readiness probe, separate from `/v1/health` liveness: ready = index responsive
/// and builder queue below the backpressure watermark; 503 above it so
/// orchestrators stop routing until the backlog drains.
// pub：bin target 的集成测试（tests.rs）跨 crate 装配路由做行为守卫。
pub async fn benchmark_readyz(
    State(state): State<BenchmarkHttpState>,
) -> (axum::http::StatusCode, Json<crate::types::ReadyzResponse>) {
    let pending = state.index.pending_build_count().await;
    let hot = state.index.hot_buffer_count().await;
    let indexed = state.index.indexed_count().await;
    let ready = readyz_is_ready(pending, READYZ_PENDING_WATERMARK);
    let body = crate::types::ReadyzResponse {
        status: if ready { "ready" } else { "not_ready" },
        pending_build: pending,
        hot_buffer: hot,
        indexed,
        watermark: READYZ_PENDING_WATERMARK,
    };
    if ready {
        (axum::http::StatusCode::OK, Json(body))
    } else {
        tracing::warn!(pending, "readyz: builder queue backlog above watermark");
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, Json(body))
    }
}

/// WAL 指标读取的抓取超时上限：写线程挂死时，每次抓取押住一个阻塞线程，
/// 必须限时降级（超时后省略该族指标），warn 文案引用此常量防止漂移。
/// EN: Scrape-side timeout for the WAL metric read — a wedged writer thread
/// would otherwise pin a blocking thread per scrape. Warn text references
/// this constant so it can't drift.
const METRICS_WAL_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// `/metrics` Prometheus 文本暴露端点（P4.0.4）：8 个核心指标，全部取自现有
/// 原子计数/快照，抓取路径零额外采集开销；不引 prometheus client 依赖——
/// 指标数量少且格式简单，手写 text exposition 比拖一个 client crate 更透明。
/// Prometheus text exposition endpoint: 8 core metrics sourced from existing
/// atomics/snapshots, so scrapes add no collection cost. No client crate —
/// the metric set is small enough that hand-written exposition is clearer.
///
/// 命名遵循 Prometheus 约定（`_total` 后缀 = counter；单位后缀 `_bytes`）。
/// Naming follows Prometheus conventions (`_total` = counter, `_bytes` = bytes unit).
// pub：bin target 的集成测试（tests.rs）跨 crate 装配路由做行为守卫。
pub async fn benchmark_metrics(
    State(state): State<BenchmarkHttpState>,
) -> ([(axum::http::header::HeaderName, &'static str); 1], String) {
    let hot = state.index.hot_buffer_count().await;
    let indexed = state.index.indexed_count().await;
    let pending = state.index.pending_build_count().await;
    let bm = state.index.builder_metrics();
    // active_file_len 在 Async 模式要穿有界通道等写线程应答（阻塞 recv），
    // 移出 runtime worker；读取失败（如写线程死亡 BrokenPipe）降级为省略
    // 该族指标 + warn，而非整篇 500——WAL 出问题时恰恰不能连索引/pgwire
    // 的可观测性一起丢。
    // EN: Degrade instead of failing the whole scrape: on WAL read error omit
    // houyidb_wal_active_bytes (with a warn) and keep the other 7 series.
    // await 再加 2s 超时：写线程若挂死（而非死亡——通道不断、recv 永不返），
    // 每次抓取都会永久押住一个阻塞线程与一个 pending future。超时后降级省略
    // 该族指标；泄漏有界（每次抓取至多一个线程，且阻塞池线程随 runtime 退出
    // 回收），换来 /metrics 永不挂死。
    // EN: Bound the await with a 2s timeout — a wedged (not dead) writer thread
    // would pin a blocking thread per scrape forever. On timeout we degrade and
    // omit the series; the leak is bounded and reclaimed at runtime exit.
    let wal = Arc::clone(&state.wal);
    let wal_read = tokio::task::spawn_blocking(move || wal.active_file_len());
    let wal_section = match tokio::time::timeout(METRICS_WAL_READ_TIMEOUT, wal_read).await {
        Ok(Ok(Ok(bytes))) => format!(
            "# HELP houyidb_wal_active_bytes Size of the active vector WAL segment.\n\
             # TYPE houyidb_wal_active_bytes gauge\n\
             houyidb_wal_active_bytes {bytes}\n"
        ),
        Ok(Ok(Err(e))) => {
            tracing::warn!(error = %e, "metrics: wal active_file_len failed, omitting houyidb_wal_active_bytes");
            String::new()
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "metrics: wal read task join failed, omitting houyidb_wal_active_bytes");
            String::new()
        }
        Err(_) => {
            tracing::warn!(
                timeout_ms = METRICS_WAL_READ_TIMEOUT.as_millis() as u64,
                "metrics: wal read task timed out, omitting houyidb_wal_active_bytes"
            );
            String::new()
        }
    };
    let pg = state.pgwire_metrics.snapshot();
    // RSS（P4.2.4）：/proc/self/statm 是内存映射的伪文件，读取微秒级，
    // 直接在 async 上下文读；读不到（非 Linux）省略该族，不拖垮整篇抓取。
    // EN: statm is a memory-backed pseudo-file (microsecond reads) — read it
    // inline; omit the series when unavailable instead of failing the scrape.
    let rss_section = match crate::mem::current_rss_bytes() {
        Some(rss) => format!(
            "# HELP houyidb_process_rss_bytes Process resident set size (P4.2.4 watermark source).\n\
             # TYPE houyidb_process_rss_bytes gauge\n\
             houyidb_process_rss_bytes {rss}\n"
        ),
        None => String::new(),
    };
    // 熔断器（P4.3.3）：仅网关启用时有熔断器实例，未启用省略该族（与 wal/rss
    // 同纪律——指标缺失比恒 0 更能表达「功能未启用」）。
    // EN: Breaker series only when the gateway is configured; absent beats a
    // misleading constant-zero for "feature off".
    let breaker_section = match &state.openai_upstream {
        Some(upstream) => {
            let b = &upstream.breaker;
            format!(
                "# HELP houyidb_openai_breaker_state Circuit breaker state: 0=closed 1=open 2=half-open (P4.3.3).\n\
                 # TYPE houyidb_openai_breaker_state gauge\n\
                 houyidb_openai_breaker_state {}\n\
                 # HELP houyidb_openai_breaker_rejected_total Requests fast-failed by the open/half-open breaker (503).\n\
                 # TYPE houyidb_openai_breaker_rejected_total counter\n\
                 houyidb_openai_breaker_rejected_total {}\n\
                 # HELP houyidb_openai_breaker_open_total Times the breaker tripped open since startup.\n\
                 # TYPE houyidb_openai_breaker_open_total counter\n\
                 houyidb_openai_breaker_open_total {}\n",
                b.state_code(),
                b.rejected_open_total(),
                b.open_total()
            )
        }
        None => String::new(),
    };
    // 复制（P4.4.1）：role≠disabled 才输出该节（与 wal/rss/breaker 同纪律——
    // 指标缺失比恒 0 更能表达「功能未启用」）。锁毒化不丢观测：into_inner 取回
    // 只读视图（指标槽只被装配/promote 写，读数不受毒化影响）。
    // EN: Replication series only when role≠disabled; a poisoned lock still
    // yields the read view (the slot is written only at assembly/promote).
    let replication_section = {
        use crate::types::ReplicationMetricsView;
        let role = state.role.get();
        let guard = state
            .replication_metrics
            .read()
            .unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(ReplicationMetricsView::Leader {
                hub,
                watermarks,
                leader_next_lsn,
                ..
            }) => {
                let next = leader_next_lsn();
                let recv = watermarks
                    .recv_lsn
                    .load(std::sync::atomic::Ordering::Relaxed);
                let durable = watermarks
                    .durable_lsn
                    .load(std::sync::atomic::Ordering::Relaxed);
                // durable 水印是「最大已确认 LSN」，next 是「下一条将分配」——
                // 差值近似滞后（从未收过 ACK 时差 1，v1 接受并文档化）。
                // EN: Approximate lag (off-by-one before the first ACK).
                let lag = next.saturating_sub(durable.saturating_add(1));
                format!(
                    "# HELP houyidb_replication_role Replication role: 0=disabled 1=leader 2=follower (P4.4.1).\n\
                     # TYPE houyidb_replication_role gauge\n\
                     houyidb_replication_role {}\n\
                     # HELP houyidb_replication_followers_active Follower sessions currently registered on the leader.\n\
                     # TYPE houyidb_replication_followers_active gauge\n\
                     houyidb_replication_followers_active {}\n\
                     # HELP houyidb_replication_recv_lsn Max LSN acked RECEIVED by any follower.\n\
                     # TYPE houyidb_replication_recv_lsn gauge\n\
                     houyidb_replication_recv_lsn {recv}\n\
                     # HELP houyidb_replication_durable_lsn Max LSN acked DURABLE (fsync+apply) by any follower.\n\
                     # TYPE houyidb_replication_durable_lsn gauge\n\
                     houyidb_replication_durable_lsn {durable}\n\
                     # HELP houyidb_replication_lag_entries Approximate unreplicated entries (leader next_lsn vs follower watermark).\n\
                     # TYPE houyidb_replication_lag_entries gauge\n\
                     houyidb_replication_lag_entries {lag}\n\
                     # HELP houyidb_replication_shipped_batches_total WAL batches published to follower queues since startup.\n\
                     # TYPE houyidb_replication_shipped_batches_total counter\n\
                     houyidb_replication_shipped_batches_total {}\n\
                     # HELP houyidb_replication_shipped_bytes_total WAL payload bytes published to follower queues since startup.\n\
                     # TYPE houyidb_replication_shipped_bytes_total counter\n\
                     houyidb_replication_shipped_bytes_total {}\n\
                     # HELP houyidb_replication_slow_follower_kicks_total Followers kicked for a full ship queue since startup.\n\
                     # TYPE houyidb_replication_slow_follower_kicks_total counter\n\
                     houyidb_replication_slow_follower_kicks_total {}\n\
                     # HELP houyidb_replication_sync_waits_total Sync waits (ha_level semi_sync/full_sync) entered since startup (P4.4.2).\n\
                     # TYPE houyidb_replication_sync_waits_total counter\n\
                     houyidb_replication_sync_waits_total {}\n\
                     # HELP houyidb_replication_sync_timeouts_total Sync waits that timed out and degraded to async since startup.\n\
                     # TYPE houyidb_replication_sync_timeouts_total counter\n\
                     houyidb_replication_sync_timeouts_total {}\n\
                     # HELP houyidb_replication_sync_no_follower_total Sync waits degraded immediately because no follower was connected.\n\
                     # TYPE houyidb_replication_sync_no_follower_total counter\n\
                     houyidb_replication_sync_no_follower_total {}\n\
                     # HELP houyidb_replication_sync_wait_nanos_total Total nanoseconds spent inside sync waits.\n\
                     # TYPE houyidb_replication_sync_wait_nanos_total counter\n\
                     houyidb_replication_sync_wait_nanos_total {}\n",
                    role.code(),
                    hub.follower_count(),
                    hub.shipped_batches_total(),
                    hub.shipped_bytes_total(),
                    hub.slow_follower_kicks_total(),
                    watermarks.sync_waits_total(),
                    watermarks.sync_timeouts_total(),
                    watermarks.sync_no_follower_total(),
                    watermarks.sync_wait_nanos_total(),
                )
            }
            Some(ReplicationMetricsView::Follower {
                metrics,
                poison_skipped,
            }) => {
                let recv = metrics.recv_lsn.load(std::sync::atomic::Ordering::Relaxed);
                let durable = metrics
                    .durable_lsn
                    .load(std::sync::atomic::Ordering::Relaxed);
                let leader_next = metrics
                    .leader_next_lsn
                    .load(std::sync::atomic::Ordering::Relaxed);
                // recv_lsn 是「下一条期望」、leader_next 是 leader 的「下一条将分配」——
                // 两者同基（都指向下一条），差值即精确滞后条数。
                // EN: Both counters point at the next entry, so the difference is exact.
                let lag = leader_next.saturating_sub(recv);
                format!(
                    "# HELP houyidb_replication_role Replication role: 0=disabled 1=leader 2=follower (P4.4.1).\n\
                     # TYPE houyidb_replication_role gauge\n\
                     houyidb_replication_role {}\n\
                     # HELP houyidb_replication_recv_lsn Next LSN the follower expects (local durable point + 1).\n\
                     # TYPE houyidb_replication_recv_lsn gauge\n\
                     houyidb_replication_recv_lsn {recv}\n\
                     # HELP houyidb_replication_durable_lsn Max LSN fsynced and applied on this follower.\n\
                     # TYPE houyidb_replication_durable_lsn gauge\n\
                     houyidb_replication_durable_lsn {durable}\n\
                     # HELP houyidb_replication_lag_entries Entries the leader has that this follower has not yet received.\n\
                     # TYPE houyidb_replication_lag_entries gauge\n\
                     houyidb_replication_lag_entries {lag}\n\
                     # HELP houyidb_replication_applied_entries_total Streamed records applied into collections since startup.\n\
                     # TYPE houyidb_replication_applied_entries_total counter\n\
                     houyidb_replication_applied_entries_total {}\n\
                     # HELP houyidb_replication_apply_failures_total Streamed records that failed to apply (durable in follower WAL; gap heals on restart replay).\n\
                     # TYPE houyidb_replication_apply_failures_total counter\n\
                     houyidb_replication_apply_failures_total {}\n\
                     # HELP houyidb_replication_apply_retries_total In-place apply retries (transient-error self-heal).\n\
                     # TYPE houyidb_replication_apply_retries_total counter\n\
                     houyidb_replication_apply_retries_total {}\n\
                     # HELP houyidb_replication_apply_poison_skipped_total Poisoned rows (non-finite components) skipped by the apply sink since startup.\n\
                     # TYPE houyidb_replication_apply_poison_skipped_total counter\n\
                     houyidb_replication_apply_poison_skipped_total {}\n\
                     # HELP houyidb_replication_fsync_total Group-commit fsync rounds on the follower WAL since startup.\n\
                     # TYPE houyidb_replication_fsync_total counter\n\
                     houyidb_replication_fsync_total {}\n\
                     # HELP houyidb_replication_reconnects_total Replication reconnects (backoff retries) since startup.\n\
                     # TYPE houyidb_replication_reconnects_total counter\n\
                     houyidb_replication_reconnects_total {}\n\
                     # HELP houyidb_replication_follower_ahead_rejections_total Handshakes rejected because this follower is ahead of the leader (split-brain signal).\n\
                     # TYPE houyidb_replication_follower_ahead_rejections_total counter\n\
                     houyidb_replication_follower_ahead_rejections_total {}\n\
                     # HELP houyidb_replication_diverged_rejections_total Handshakes rejected because the local log diverges from the leader at the same LSN (manual intervention).\n\
                     # TYPE houyidb_replication_diverged_rejections_total counter\n\
                     houyidb_replication_diverged_rejections_total {}\n\
                     # HELP houyidb_replication_auth_rejections_total Handshakes rejected because the replication auth token does not match (configuration error).\n\
                     # TYPE houyidb_replication_auth_rejections_total counter\n\
                     houyidb_replication_auth_rejections_total {}\n\
                     # HELP houyidb_replication_local_log_errors_total Sessions aborted because the local log could not produce a fingerprint (corrupt/truncated/gapped).\n\
                     # TYPE houyidb_replication_local_log_errors_total counter\n\
                     houyidb_replication_local_log_errors_total {}\n\
                     # HELP houyidb_replication_behind_retained_rejections_total Handshakes rejected because this follower's history predates the leader's retained range (manual full resync).\n\
                     # TYPE houyidb_replication_behind_retained_rejections_total counter\n\
                     houyidb_replication_behind_retained_rejections_total {}\n\
                     # HELP houyidb_replication_leader_first_available_lsn Earliest LSN the leader can still serve (from WELCOME).\n\
                     # TYPE houyidb_replication_leader_first_available_lsn gauge\n\
                     houyidb_replication_leader_first_available_lsn {}\n",
                    role.code(),
                    metrics
                        .applied_entries_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .apply_failures_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .apply_retries_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    poison_skipped.load(std::sync::atomic::Ordering::Relaxed),
                    metrics.fsync_total.load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .reconnects_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .follower_ahead_rejections_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .diverged_rejections_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .auth_rejections_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .local_log_errors_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .behind_retained_rejections_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    metrics
                        .leader_first_available_lsn
                        .load(std::sync::atomic::Ordering::Relaxed),
                )
            }
            None => String::new(),
        }
    };
    // 审计（P4.7.0）：未启用时省略该节（与 wal/rss/breaker 同纪律）。
    // EN: audit series only when enabled (absent beats constant zero).
    let audit_section = match &state.audit {
        Some(a) => format!(
            "# HELP houyidb_audit_entries_total Audit records appended since startup (P4.7.0).\n\
             # TYPE houyidb_audit_entries_total counter\n\
             houyidb_audit_entries_total {}\n\
             # HELP houyidb_audit_write_errors_total Audit appends that failed (successful responses become 500).\n\
             # TYPE houyidb_audit_write_errors_total counter\n\
             houyidb_audit_write_errors_total {}\n",
            a.entries_total(),
            a.write_errors_total()
        ),
        None => String::new(),
    };
    // 标量页聚合指标（P5.0.1）：只做廉价计数；live/stale 精确值在 status 端点。
    // EN: Aggregate scalar metrics (P5.0.1) — cheap counts only; exact live/stale
    // values live on the status endpoint.
    let (
        scalar_physical_rows,
        scalar_tombstones,
        scalar_ok,
        scalar_skipped,
        scalar_failed,
        scalar_dropped,
    ) = {
        let mut physical = 0u64;
        let mut tombstones = 0u64;
        let mut ok = 0u64;
        let mut skipped = 0u64;
        let mut failed = 0u64;
        let mut dropped = 0u64;
        for name in state.collections.list() {
            let Some(col) = state.collections.get(&name) else {
                continue;
            };
            let (p, t) = col.scalars.lock().map(|s| s.raw_counts()).unwrap_or((0, 0));
            physical += p as u64;
            tombstones += t as u64;
            ok += col
                .compactions_ok_total
                .load(std::sync::atomic::Ordering::Relaxed);
            skipped += col
                .compactions_skipped_total
                .load(std::sync::atomic::Ordering::Relaxed);
            failed += col
                .compaction_failures_total
                .load(std::sync::atomic::Ordering::Relaxed);
            dropped += col
                .compaction_dropped_rows_total
                .load(std::sync::atomic::Ordering::Relaxed);
        }
        (physical, tombstones, ok, skipped, failed, dropped)
    };
    let scalar_section = format!(
        "# HELP houyidb_collections_scalar_physical_rows Scalar physical rows across collections (stale/tombstoned included).\n\
         # TYPE houyidb_collections_scalar_physical_rows gauge\n\
         houyidb_collections_scalar_physical_rows {scalar_physical_rows}\n\
         # HELP houyidb_collections_scalar_tombstones Scalar tombstone ids across collections.\n\
         # TYPE houyidb_collections_scalar_tombstones gauge\n\
         houyidb_collections_scalar_tombstones {scalar_tombstones}\n\
         # HELP houyidb_collections_scalar_compactions_total Scalar compactions by outcome (ok/skipped/failed).\n\
         # TYPE houyidb_collections_scalar_compactions_total counter\n\
         houyidb_collections_scalar_compactions_total{{result=\"ok\"}} {scalar_ok}\n\
         houyidb_collections_scalar_compactions_total{{result=\"skipped\"}} {scalar_skipped}\n\
         houyidb_collections_scalar_compactions_total{{result=\"failed\"}} {scalar_failed}\n\
         # HELP houyidb_collections_scalar_compaction_dropped_rows_total Rows dropped by scalar compactions.\n\
         # TYPE houyidb_collections_scalar_compaction_dropped_rows_total counter\n\
         houyidb_collections_scalar_compaction_dropped_rows_total {scalar_dropped}\n"
    );
    // Collections WAL 段/保留指标（P5.0.3）。
    // EN: Collections WAL segment/retention metrics (P5.0.3).
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&state.wal_path);
    let collections_metrics = tokio::task::spawn_blocking({
        let wal = Arc::clone(&state.collections_wal);
        let path = collections_wal_path.clone();
        let role = state.role.clone();
        let lifecycle = Arc::clone(&state.lifecycle);
        move || -> std::io::Result<_> {
            let stats = wal.segment_stats()?;
            let report = crate::collection_lifecycle::retention_report_from_stats(
                &path, &role, &lifecycle, &stats, None, false,
            );
            Ok((stats, report))
        }
    })
    .await;
    let lifecycle_metrics = &state.lifecycle.metrics;
    let collections_section = match collections_metrics {
        Ok(Ok((collections_stats, collections_report))) => format!(
        "# HELP houyidb_collections_wal_first_available_lsn Earliest readable collections WAL LSN.\n\
         # TYPE houyidb_collections_wal_first_available_lsn gauge\n\
         houyidb_collections_wal_first_available_lsn {}\n\
         # HELP houyidb_collections_wal_active_bytes Active collections WAL segment bytes.\n\
         # TYPE houyidb_collections_wal_active_bytes gauge\n\
         houyidb_collections_wal_active_bytes {}\n\
         # HELP houyidb_collections_wal_archived_segments Archived collections WAL segments.\n\
         # TYPE houyidb_collections_wal_archived_segments gauge\n\
         houyidb_collections_wal_archived_segments {}\n\
         # HELP houyidb_collections_wal_archived_bytes Archived collections WAL bytes.\n\
         # TYPE houyidb_collections_wal_archived_bytes gauge\n\
         houyidb_collections_wal_archived_bytes {}\n\
         # HELP houyidb_collections_wal_segments_deleted_total Whole segments deleted by retention.\n\
         # TYPE houyidb_collections_wal_segments_deleted_total counter\n\
         houyidb_collections_wal_segments_deleted_total {}\n\
         # HELP houyidb_collections_wal_bytes_freed_total Bytes freed by retention.\n\
         # TYPE houyidb_collections_wal_bytes_freed_total counter\n\
         houyidb_collections_wal_bytes_freed_total {}\n\
         # HELP houyidb_collections_wal_safe_floor_lsn Retention safe floor (manifest safe checkpoint).\n\
         # TYPE houyidb_collections_wal_safe_floor_lsn gauge\n\
         houyidb_collections_wal_safe_floor_lsn {}\n\
         # HELP houyidb_collections_wal_retention_blocked Retention currently blocked (1/0).\n\
         # TYPE houyidb_collections_wal_retention_blocked gauge\n\
         houyidb_collections_wal_retention_blocked {}\n\
         # HELP houyidb_collections_snapshot_safe_checkpoint_lsn Manifest safe checkpoint LSN.\n\
         # TYPE houyidb_collections_snapshot_safe_checkpoint_lsn gauge\n\
         houyidb_collections_snapshot_safe_checkpoint_lsn {}\n\
         # HELP houyidb_collections_manifest_writes_total Manifests successfully committed.\n\
         # TYPE houyidb_collections_manifest_writes_total counter\n\
         houyidb_collections_manifest_writes_total {}\n\
         # HELP houyidb_collections_manifest_rejects_total Manifest writes rejected/failed.\n\
         # TYPE houyidb_collections_manifest_rejects_total counter\n\
         houyidb_collections_manifest_rejects_total {}\n\
         # HELP houyidb_collections_snapshot_failures_total Collection snapshot cycles that failed.\n\
         # TYPE houyidb_collections_snapshot_failures_total counter\n\
         houyidb_collections_snapshot_failures_total {}\n",
        collections_stats.first_available_lsn,
        collections_stats.active_bytes,
        collections_stats.archives.len(),
        collections_stats
            .archives
            .iter()
            .map(|m| m.bytes)
            .sum::<u64>(),
        lifecycle_metrics
            .segments_deleted_total
            .load(std::sync::atomic::Ordering::Relaxed),
        lifecycle_metrics
            .bytes_freed_total
            .load(std::sync::atomic::Ordering::Relaxed),
        collections_report.safe_floor_lsn.unwrap_or(0),
        u64::from(collections_report.blocked_reason.is_some()),
        collections_report.safe_floor_lsn.unwrap_or(0),
        state
            .graph_snapshot_metrics
            .manifest_writes_total
            .load(std::sync::atomic::Ordering::Relaxed),
        state
            .graph_snapshot_metrics
            .manifest_rejects_total
            .load(std::sync::atomic::Ordering::Relaxed),
        state
            .graph_snapshot_metrics
            .failures_total
            .load(std::sync::atomic::Ordering::Relaxed),
    ),
        // 段统计失败时不要输出“健康空 WAL”的零值；保留一条可诊断注释。
        // EN: Do not emit healthy-looking zeroes when stats fail; keep a
        // diagnosable comment instead.
        Ok(Err(e)) => format!("# collections WAL stats unavailable: {e}\n"),
        Err(e) => format!("# collections WAL stats task failed: {e}\n"),
    };
    let body = format!(
        "# HELP houyidb_index_hot_buffer Vectors held in the hot write buffer (not yet searchable).\n\
         # TYPE houyidb_index_hot_buffer gauge\n\
         houyidb_index_hot_buffer {hot}\n\
         # HELP houyidb_index_indexed_vectors Vectors built into the ANN graph (searchable).\n\
         # TYPE houyidb_index_indexed_vectors gauge\n\
         houyidb_index_indexed_vectors {indexed}\n\
         # HELP houyidb_index_pending_build Vectors queued for async graph build (readiness watermark {watermark}).\n\
         # TYPE houyidb_index_pending_build gauge\n\
         houyidb_index_pending_build {pending}\n\
         # HELP houyidb_index_flush_total Async-builder flush rounds since startup.\n\
         # TYPE houyidb_index_flush_total counter\n\
         houyidb_index_flush_total {flush_count}\n\
         # HELP houyidb_index_flushed_items_total Vector items flushed into the graph since startup.\n\
         # TYPE houyidb_index_flushed_items_total counter\n\
         houyidb_index_flushed_items_total {flushed_items}\n\
         {wal_section}\
         # HELP houyidb_pgwire_handshakes_total pgwire connections accepted since startup.\n\
         # TYPE houyidb_pgwire_handshakes_total counter\n\
         houyidb_pgwire_handshakes_total {handshakes}\n\
         # HELP houyidb_pgwire_queries_total pgwire simple-query messages since startup.\n\
         # TYPE houyidb_pgwire_queries_total counter\n\
         houyidb_pgwire_queries_total {queries}\n\
         # HELP houyidb_pgwire_auth_failures_total pgwire authentication rejections since startup (all paths).\n\
         # TYPE houyidb_pgwire_auth_failures_total counter\n\
         houyidb_pgwire_auth_failures_total {pg_auth_failures}\n\
         # HELP houyidb_pgwire_authz_denials_total pgwire statements rejected by RBAC (42501).\n\
         # TYPE houyidb_pgwire_authz_denials_total counter\n\
         houyidb_pgwire_authz_denials_total {pg_authz_denials}\n\
         # HELP houyidb_pgwire_quota_denials_total pgwire statements rejected by quota (54000).\n\
         # TYPE houyidb_pgwire_quota_denials_total counter\n\
         houyidb_pgwire_quota_denials_total {pg_quota_denials}\n\
         # HELP houyidb_pgwire_connections_active pgwire connections currently admitted (P4.2.3 watermark).\n\
         # TYPE houyidb_pgwire_connections_active gauge\n\
         houyidb_pgwire_connections_active {pg_conn_active}\n\
         # HELP houyidb_pgwire_connections_rejected_total pgwire connections refused by the watermark (FATAL 53300).\n\
         # TYPE houyidb_pgwire_connections_rejected_total counter\n\
         houyidb_pgwire_connections_rejected_total {pg_conn_rejected}\n\
         # HELP houyidb_http_inflight_requests HTTP requests currently in flight (P4.2.3 watermark).\n\
         # TYPE houyidb_http_inflight_requests gauge\n\
         houyidb_http_inflight_requests {http_active}\n\
         # HELP houyidb_http_inflight_rejected_total HTTP requests refused by the watermark (503).\n\
         # TYPE houyidb_http_inflight_rejected_total counter\n\
         houyidb_http_inflight_rejected_total {http_rejected}\n\
         {rss_section}\
         {breaker_section}\
         {audit_section}\
         # HELP houyidb_memory_watermark_rejected_total Writes refused by the memory watermark (P4.2.4).\n\
         # TYPE houyidb_memory_watermark_rejected_total counter\n\
         houyidb_memory_watermark_rejected_total {mem_rejected}\n\
         {scalar_section}\
         # HELP houyidb_graph_snapshot_writes_total Graph snapshot files written since startup (P4.4.0; main index + collections).\n\
         # TYPE houyidb_graph_snapshot_writes_total counter\n\
         houyidb_graph_snapshot_writes_total {gs_writes}\n\
         # HELP houyidb_graph_snapshot_loads_total Graph snapshots successfully restored at startup.\n\
         # TYPE houyidb_graph_snapshot_loads_total counter\n\
         houyidb_graph_snapshot_loads_total {gs_loads}\n\
         # HELP houyidb_graph_snapshot_rejects_total Graph snapshots discarded (corrupt/bad header/restore failure) at startup.\n\
         # TYPE houyidb_graph_snapshot_rejects_total counter\n\
         houyidb_graph_snapshot_rejects_total {gs_rejects}\n\
         # HELP houyidb_graph_snapshot_bytes_written_total Graph snapshot payload bytes written since startup (envelope excluded).\n\
         # TYPE houyidb_graph_snapshot_bytes_written_total counter\n\
         houyidb_graph_snapshot_bytes_written_total {gs_bytes}\n\
         {collections_section}\
         {replication_section}",
        watermark = READYZ_PENDING_WATERMARK,
        flush_count = bm.flush_count,
        flushed_items = bm.total_flushed_items,
        handshakes = pg.handshake_count,
        queries = pg.simple_query_count,
        pg_auth_failures = pg.auth_failures,
        pg_authz_denials = pg.authz_denials,
        pg_quota_denials = pg.quota_denials,
        pg_conn_active = state.pgwire_metrics.connection_watermark.active(),
        pg_conn_rejected = state.pgwire_metrics.connection_watermark.rejected_total(),
        http_active = state.http_inflight.active(),
        http_rejected = state.http_inflight.rejected_total(),
        rss_section = rss_section,
        breaker_section = breaker_section,
        mem_rejected = state.pgwire_metrics.memory_watermark.rejected_total(),
        gs_writes = state
            .graph_snapshot_metrics
            .writes_total
            .load(std::sync::atomic::Ordering::Relaxed),
        gs_loads = state
            .graph_snapshot_metrics
            .loads_total
            .load(std::sync::atomic::Ordering::Relaxed),
        gs_rejects = state
            .graph_snapshot_metrics
            .rejects_total
            .load(std::sync::atomic::Ordering::Relaxed),
        gs_bytes = state
            .graph_snapshot_metrics
            .bytes_written_total
            .load(std::sync::atomic::Ordering::Relaxed),
        collections_section = collections_section,
        replication_section = replication_section,
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// 内嵌的 OpenAPI 3.1 文档（构建期随二进制打包，离线可用、与版本严格同源）。
/// Embedded OpenAPI 3.1 spec (compiled into the binary; always matches the build).
// pub 而非 pub(crate)：bin target 的集成测试（tests.rs）跨 crate 引用做文档守卫。
pub const OPENAPI_SPEC: &str = include_str!("../openapi.json");

/// 提供 OpenAPI 文档
/// EN: Serve the OpenAPI spec at /v1/openapi.json.
async fn handle_openapi_spec() -> impl axum::response::IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        OPENAPI_SPEC,
    )
}

pub async fn benchmark_upsert(
    State(state): State<BenchmarkHttpState>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    let inserted = req.vectors.len();

    // 校验先于一切副作用（WHY: 旧实现先写 label_index、改构图参数、append WAL，维度
    // 错误或 NaN/Inf 随后才返回 400——毒记录已残留进 WAL（本路径 WAL-first，冷启动重放
    // 会被库层拒绝而中止），label 映射也被脏 id 污染）。
    // EN: Validate BEFORE any side effect. The old ordering wrote label_index, mutated
    // build hints and appended the WAL before the 400: a poisoned record then survived in
    // the log (this path is WAL-first) and poisoned label mappings.
    validate_upsert_vectors(&req.vectors, state.vector_dim)?;

    // 内存水位线（P4.2.4）：校验后、副作用前裁决，超限 503。
    // EN: Memory watermark admission after validation, before side effects.
    state
        .pgwire_metrics
        .memory_watermark
        .check_admission()
        .map_err(|msg| (axum::http::StatusCode::SERVICE_UNAVAILABLE, msg))?;

    // 记录 label → ids 映射 (用于混合检索的标量过滤)
    // Store label→ids mapping for hybrid/scalar-filtered search
    {
        let mut label_map = state
            .label_index
            .write()
            .map_err(|e: std::sync::PoisonError<_>| {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?;
        let mut touched_labels: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for v in &req.vectors {
            if let Some(ref label) = v.label {
                // Arc::make_mut：无并发读者时原地追加；有读者持 Arc 时克隆一次
                // 列表（写少读多，读取侧不再复制 10K id）。
                // EN: Arc::make_mut appends in place unless readers hold the Arc.
                let entry = label_map.entry(label.clone()).or_default();
                Arc::make_mut(&mut entry.ids).push(v.id);
                touched_labels.insert(label.clone());
            }
        }
        // 每批每标签只失效一次缓存（逐向量抢缓存锁会让 1M 批量加载多付
        // 1M 次锁/原子开销）。
        // EN: Invalidate each touched label once per batch — per-vector cache locks
        // add 1M lock/atomic ops to a 1M-row load.
        for label in touched_labels {
            if let Some(entry) = label_map.get(&label) {
                if let Ok(mut cache) = entry.cache.write() {
                    if let Some(old) = cache.take() {
                        LABEL_CACHE_BYTES
                            .fetch_sub(old.data.len() * 4, std::sync::atomic::Ordering::AcqRel);
                    }
                }
            }
        }
    }

    if let Some(ref hints) = req.index {
        if let Some(m) = hints.m {
            // 避免与 HTTP 批量 upsert 重复 set：每次抢 `hnsw.write()` 会与异步构图 worker 长时间持锁叠加。
            // EN: Skip redundant `set_hnsw_m` to avoid lock contention with the async builder on every batch.
            if m != state.index.hnsw_m() {
                state
                    .index
                    .set_hnsw_m(m)
                    .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
            }
        }
        if let Some(efc) = hints.ef_construction {
            if efc != state.index.hnsw_ef_construction_value() {
                state
                    .index
                    .set_hnsw_ef_construction(efc)
                    .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
            }
        }
    }
    // 维度/非有限校验已在函数入口统一完成（validate_upsert_vectors），此处直接落批。
    // EN: Dim/finiteness validation already ran at the top (validate_upsert_vectors).
    let batch: Vec<(u64, Vec<f32>)> = req.vectors.into_iter().map(|v| (v.id, v.values)).collect();
    let payload = encode_vector_upsert_batch(state.vector_dim, &batch)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // WAL-first: persist before in-memory apply so crash + cold replay can recover the batch.
    // 先写 WAL 再 insert_batch；若 insert 失败，重放仍会应用该条（全量重放自空索引，语义一致）。
    let committed_lsn = state
        .wal
        .append(payload)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .lsn;
    // 同 collection upsert：预校验后此处理论上不可达；记录日志让未来的非校验类后端
    // 故障不会被静默误分类成 4xx。注意 WAL 已先落盘（WAL-first 语义），重放会自愈。
    // Same as collection upsert: effectively unreachable after pre-validation; log so a
    // future non-validation backend fault stays observable. The WAL is already durable
    // (WAL-first) — replay re-applies the batch on restart.
    state.index.insert_batch(batch).await.map_err(|e| {
        tracing::warn!(error = %e, "[houyi-upsert] benchmark insert_batch rejected (expected validation-only)");
        (axum::http::StatusCode::BAD_REQUEST, e)
    })?;
    {
        let _guard = state
            .applied_sidecar_lock
            .lock()
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        // EN: Parallel requests can complete out of order; persist the max LSN only.
        // 中文：并发请求完成顺序可能乱序，仅持久化最大 LSN，避免侧车回退。
        let current = read_applied_lsn(&state.applied_sidecar)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        write_applied_lsn_atomic(&state.applied_sidecar, current.max(committed_lsn))
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    Ok(Json(UpsertResponse { inserted }))
}

pub async fn benchmark_search(
    State(state): State<BenchmarkHttpState>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    let started = std::time::Instant::now();
    let k = req.k;
    let filtered = req.filter_label.is_some() || req.filter_gt_id.is_some();
    let result = benchmark_search_inner(State(state.clone()), principal.clone(), Json(req)).await;
    if let Err((status, _)) = &result {
        record_query_log(
            &state,
            principal.as_ref().map(|ext| &ext.0),
            "benchmark_search",
            None,
            k,
            status.as_u16(),
            started,
            0,
            filtered,
        );
    }
    result
}

async fn benchmark_search_inner(
    State(state): State<BenchmarkHttpState>,
    principal: Option<axum::extract::Extension<crate::rbac::AuthenticatedPrincipal>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    // 维度 + 非有限查询拦截先于 k==0 短路（NaN/Inf 距离恒 NaN、排序未定义；
    // JSON 1e40 → f32::Inf）。WHY: 若先短路，同一条坏查询 k=0 得 200 空结果、
    // k>0 得 400，契约自相矛盾；坏输入必须与 k 无关地被拒绝。
    // EN: Validate before the k==0 short-circuit — otherwise the same invalid query
    // would yield 200-empty for k=0 but 400 for k>0, an inconsistent contract.
    validate_query_vector(&req.vector, state.vector_dim, "query")?;
    if req.k == 0 {
        // 刻意保留：VectorDBBench 基准语义下 k==0 等价于「不取结果」，返回 200 空响应，
        // 与历史压测契约保持一致；本路径只封上界（见下 reject_search_limits）。
        // 注意：collection/hybrid 路径面向业务调用，k==0 在那里是 400——差异是有意为之。
        // EN: Deliberate — under VectorDBBench benchmark semantics k==0 means "fetch
        // nothing", so return 200-empty for wire compatibility; only the upper cap is
        // enforced here. On the business-facing collection/hybrid paths k==0 is a 400;
        // the divergence is intentional.
        return Ok(Json(SearchResponse {
            ids: Vec::new(),
            scores: Vec::new(),
            payloads: None,
            metric: None,
        }));
    }
    // k==0 已短路；此处只封上界：HTTP 边缘拒绝超大 k，防止 4k 过取与库内 ~16×k
    // 预分配（HNSW）构成的内存放大攻击面。
    // EN: k==0 already short-circuited; enforce only the upper cap at the HTTP edge to
    // bound the 4k over-fetch and the in-library ~16×k HNSW preallocation.
    reject_search_limits(req.k, req.ef_search)?;

    // label 过滤 → 匹配 id + 连续向量缓存（精确顺序扫描）。
    // 1% of 1M ≈ 10K 命中：把向量拷进连续内存后顺序 SIMD 扫描，避免每 query
    // 10K 次 HashMap 查表 + 散列指针跳转；缓存按标签惰性构建、写后失效。
    // EN: Label filter → ids plus a contiguous vector cache (exact sequential scan).
    // 10K hits at 1% of 1M are copied once into contiguous memory; writes invalidate.
    const MAX_LABEL_CACHE_BYTES: usize = 256 * 1024 * 1024;
    // 全局预算：标签缓存按标签懒构建；总占用超限时整体清空（下一次查询再建），
    // 避免多标签/大标签把内存推高。上限 512MB + 单个新缓存（≤256MB）。
    // EN: Global budget — clear all label caches when the total exceeds 512MB, so
    // many/queries-large labels cannot grow memory without bound.
    const MAX_TOTAL_LABEL_CACHE_BYTES: usize = 512 * 1024 * 1024;
    // 标签过滤的精确扫描必须按索引的实际度量计分（ip/l1 不再被误映射为 l2）。
    // EN: the label-filter exact scan must score by the index's actual metric
    // (ip/l1 are no longer mis-mapped to l2).
    let metric = state.index.distance_metric();
    // 写入代数：缓存构建/读取都要与之对齐（含 builder staged 窗口）。
    // EN: Write generation; caches must match it (covers the builder staged window).
    let write_gen = state.index.write_generation();
    let (label_filter_ids, label_cache) = if let Some(ref label) = req.filter_label {
        let (ids, cache_slot) = {
            let label_map = state
                .label_index
                .read()
                .map_err(|e: std::sync::PoisonError<_>| {
                    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                })?;
            match label_map.get(label) {
                Some(entry) => (Some(Arc::clone(&entry.ids)), Some(Arc::clone(&entry.cache))),
                None => (None, None),
            }
        };
        let cached = cache_slot
            .as_ref()
            .and_then(|slot| slot.read().ok().and_then(|g| g.clone()))
            // 代数不一致 = 构建后发生过写入（含 builder 合图），视为 miss。
            // EN: A generation mismatch means a write/merge happened after build.
            .filter(|c| c.generation == write_gen);
        let cache = if cached.is_some() {
            cached
        } else if let (Some(ids), Some(slot)) = (ids.as_ref(), cache_slot.as_ref()) {
            let dims = state.vector_dim;
            let expected_bytes = ids.len().saturating_mul(dims).saturating_mul(4);
            if dims > 0 && expected_bytes <= MAX_LABEL_CACHE_BYTES {
                // 原子预留；超总预算则本标签不缓存（回退索引检索）。
                // EN: Atomic reservation; over budget -> skip cache for this label.
                let prev = LABEL_CACHE_BYTES
                    .fetch_add(expected_bytes, std::sync::atomic::Ordering::AcqRel);
                if prev.saturating_add(expected_bytes) > MAX_TOTAL_LABEL_CACHE_BYTES {
                    LABEL_CACHE_BYTES
                        .fetch_sub(expected_bytes, std::sync::atomic::Ordering::AcqRel);
                    None
                } else {
                    let vectors = state.index.vectors_by_ids(ids.as_slice()).await;
                    // 构建窗口内发生写入（含 builder 合图）→ 本次结果作废。
                    // EN: A write/merge during the build window invalidates this build.
                    if state.index.write_generation() != write_gen || vectors.is_empty() {
                        LABEL_CACHE_BYTES
                            .fetch_sub(expected_bytes, std::sync::atomic::Ordering::AcqRel);
                        None
                    } else {
                        let mut out_ids = Vec::with_capacity(vectors.len());
                        let mut data = Vec::with_capacity(vectors.len() * dims);
                        for (id, mut v) in vectors {
                            if v.len() != dims {
                                continue;
                            }
                            if metric == DistanceMetric::Cosine {
                                houyi_vector::distance::normalize_in_place(&mut v);
                            }
                            out_ids.push(id);
                            data.extend_from_slice(&v);
                        }
                        let actual = data.len() * 4;
                        if actual > expected_bytes {
                            LABEL_CACHE_BYTES.fetch_add(
                                actual - expected_bytes,
                                std::sync::atomic::Ordering::AcqRel,
                            );
                        } else if actual < expected_bytes {
                            LABEL_CACHE_BYTES.fetch_sub(
                                expected_bytes - actual,
                                std::sync::atomic::Ordering::AcqRel,
                            );
                        }
                        let c = Arc::new(LabelVectorCache {
                            ids: out_ids,
                            data,
                            dims,
                            generation: write_gen,
                        });
                        // 替换旧缓存并释放旧预留；安装失败则释放本次预留。
                        // EN: Replace the old cache and release its reservation; on
                        // install failure release this build's reservation.
                        let mut installed = false;
                        let old = if let Ok(mut guard) = slot.write() {
                            installed = true;
                            guard.replace(Arc::clone(&c))
                        } else {
                            None
                        };
                        if let Some(old) = old {
                            LABEL_CACHE_BYTES
                                .fetch_sub(old.data.len() * 4, std::sync::atomic::Ordering::AcqRel);
                        }
                        if !installed {
                            LABEL_CACHE_BYTES
                                .fetch_sub(actual, std::sync::atomic::Ordering::AcqRel);
                        }
                        Some(c)
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        (ids, cache)
    } else {
        (None, None)
    };

    // 一律按 4k 过取（不再只对过滤查询）：4× 过取同时服务两件事——过滤谓词的
    // false-empty 回补，以及 NaN/±Inf 毒槽位的有限候选回填（OCR 第五轮：旧代码无
    // 过滤分支 fetch_k==k，毒行落在前 k 时响应直接缩水）。见
    // finite_pairs_truncated_to_k 的 WHY；与 collection/hybrid 三条路径口径一致。
    // EN: Always over-fetch 4k (not for filtered queries only) — the same headroom
    // backfills both post-filter false-empties and poisoned (non-finite) top slots;
    // the unfiltered branch previously fetched k, so a poisoned top-k slot shrank the
    // response. Aligned with the collection/hybrid handlers.
    // 饱和乘法后的 `.max(req.k)` 守卫实际不可达（4k 恒 ≥ k），保留以对齐库内
    // `saturating_mul(n).max(k)` 惯例（见 houyi-vector 检索链路）。
    // EN: The `.max(req.k)` after saturating_mul is unreachable (4k ≥ k always); kept
    // deliberately to mirror the in-library `saturating_mul(n).max(k)` convention.
    let fetch_k = req.k.saturating_mul(4).max(req.k);
    // 慢查询计时（P4.3.2）：阈值 0 = 禁用，超阈值打 warn（不含向量本体）。
    // EN: Slow-query timing (P4.3.2): threshold 0 disables; over threshold emits a
    // warn (without the vector itself).
    let t0 = std::time::Instant::now();
    // 取带距离的 (id, distance) 对：后续 retain/truncate 成对进行，
    // 保证响应里 ids[i]/scores[i] 始终对齐。
    // EN: Fetch scored pairs and apply retain/truncate pair-wise so ids/scores stay aligned.
    let mut hits = if let Some(cache) = label_cache.as_ref() {
        // rayon 扫描是同步 CPU 工作：移出 tokio worker，避免阻塞异步调度。
        // EN: The rayon scan is synchronous CPU work; move it off the tokio worker.
        let cache = Arc::clone(cache);
        let query = req.vector.clone();
        match tokio::task::spawn_blocking(move || {
            houyi_vector::search_flat_contiguous(
                &query,
                &cache.ids,
                &cache.data,
                cache.dims,
                fetch_k,
                metric,
            )
        })
        .await
        {
            Ok(hits) => hits,
            Err(_) => {
                // JoinError（panic/取消）：回退索引检索，避免把内部错误变成空结果。
                // EN: JoinError — fall back to the index path instead of empty results.
                state
                    .index
                    .search_scored_with_ef_override(
                        &req.vector,
                        fetch_k,
                        label_filter_ids.as_ref().map(|ids| ids.as_slice()),
                        req.ef_search,
                    )
                    .await
            }
        }
    } else {
        state
            .index
            .search_scored_with_ef_override(
                &req.vector,
                fetch_k,
                label_filter_ids.as_ref().map(|ids| ids.as_slice()),
                req.ef_search,
            )
            .await
    };
    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let threshold = state
        .pgwire_metrics
        .slow_query_ms
        .load(std::sync::atomic::Ordering::Relaxed);
    if crate::types::slow_query_exceeded(elapsed_ms, threshold) {
        tracing::warn!(
            elapsed_ms,
            threshold_ms = threshold,
            k = req.k,
            filtered = label_filter_ids.is_some(),
            "slow http benchmark search"
        );
    }

    // 额外的 id > N 过滤（与 label 过滤可叠加）
    // Additional id > N filter (stackable with label filter)
    if let Some(min_id) = req.filter_gt_id {
        hits.retain(|(id, _)| *id > min_id);
    }
    // 非有限距离在 truncate 之前剔除并由 4k 过取回填（三条检索路径共用的唯一实现）。
    // scored_pairs_to_response 内的丢弃保留为最后一道安全网。
    // EN: Shared pre-truncate drop + 4k backfill across all three search paths;
    // the response converter keeps its own drop as the final safety net.
    let hits = finite_pairs_truncated_to_k(hits, req.k);
    // 距离 → 分数按 benchmark 索引的度量换算（cosine 钳制、其余不钳制；ip 取反）。
    // EN: distance → score by the benchmark index's metric (cosine clamps, others
    // don't; ip negates).
    let resp = scored_pairs_to_response(hits, state.index.distance_metric());
    record_query_log(
        &state,
        principal.as_ref().map(|ext| &ext.0),
        "benchmark_search",
        None,
        req.k,
        200,
        t0,
        resp.ids.len(),
        label_filter_ids.is_some() || req.filter_gt_id.is_some(),
    );
    Ok(Json(resp))
}

pub async fn benchmark_wal_checkpoint(
    State(state): State<BenchmarkHttpState>,
    Json(req): Json<WalCheckpointRequest>,
) -> Result<Json<WalCheckpointResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    let applied = read_applied_lsn(&state.applied_sidecar)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let through = req.through_lsn.unwrap_or(applied);
    if through > applied {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "through_lsn {through} exceeds applied_lsn {applied} (refuse truncating uncommitted WAL)"
            ),
        ));
    }
    const MAX_DRAIN: usize = 1024;
    for _ in 0..MAX_DRAIN {
        if state.index.run_async_builder_once().await.flushed == 0 {
            break;
        }
    }
    let removed = state
        .wal
        .truncate_prefix_through(through)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let next_lsn = state.wal.next_lsn();
    Ok(Json(WalCheckpointResponse {
        removed_entries: removed,
        next_lsn,
        through_lsn: through,
        data_loss_on_restart_without_snapshot: true,
    }))
}

pub async fn benchmark_wal_status(
    State(state): State<BenchmarkHttpState>,
) -> Result<Json<WalStatusResponse>, (axum::http::StatusCode, String)> {
    let applied = read_applied_lsn(&state.applied_sidecar)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let next_lsn = state.wal.next_lsn();
    let active_bytes = state
        .wal
        .active_file_len()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let archived_segments = state
        .wal
        .archived_segment_count()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    // Collections WAL 的段/快照/保留视图（P5.0.3）。
    // EN: Collections WAL segment/snapshot/retention view (P5.0.3).
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&state.wal_path);
    // 段统计/next_lsn/manifest 读取都是阻塞操作：统一移出 runtime worker。
    // EN: Segment stats, next_lsn and the manifest read are blocking; run them
    // together off the runtime worker.
    let (stats, collections_next_lsn, report) = tokio::task::spawn_blocking({
        let wal = Arc::clone(&state.collections_wal);
        let path = collections_wal_path.clone();
        let role = state.role.clone();
        let lifecycle = Arc::clone(&state.lifecycle);
        move || -> std::io::Result<_> {
            let stats = wal.segment_stats()?;
            let next = wal.next_lsn();
            let report = crate::collection_lifecycle::retention_report_from_stats(
                &path, &role, &lifecycle, &stats, None, false,
            );
            Ok((stats, next, report))
        }
    })
    .await
    .map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("collections status task join failed: {e}"),
        )
    })?
    .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let collections = CollectionsWalStatus {
        first_available_lsn: stats.first_available_lsn,
        next_lsn: collections_next_lsn,
        active_bytes: stats.active_bytes,
        archived_segments: stats.archives.len(),
        archived_bytes: stats.archives.iter().map(|m| m.bytes).sum(),
        oldest_archive_first_lsn: stats.archives.first().map(|m| m.first_lsn),
        newest_archive_last_lsn: stats.archives.last().map(|m| m.last_lsn),
        manifest_present: report.manifest_present,
        manifest_seq: report.manifest_seq,
        safe_checkpoint_lsn: report.safe_floor_lsn,
        retention_safe_floor_lsn: report.safe_floor_lsn,
        retention_blocked_reason: report.blocked_reason,
        retention_lease_until_unix: state.lifecycle.lease_until_unix(),
        segments_deleted_total: state
            .lifecycle
            .metrics
            .segments_deleted_total
            .load(std::sync::atomic::Ordering::Relaxed),
        bytes_freed_total: state
            .lifecycle
            .metrics
            .bytes_freed_total
            .load(std::sync::atomic::Ordering::Relaxed),
    };
    Ok(Json(WalStatusResponse {
        next_lsn,
        applied_lsn: applied,
        active_bytes,
        archived_segments,
        collections,
    }))
}

pub async fn benchmark_pgwire_status(
    State(state): State<BenchmarkHttpState>,
) -> Result<Json<PgWireMetricsSnapshot>, (axum::http::StatusCode, String)> {
    Ok(Json(state.pgwire_metrics.snapshot()))
}

async fn benchmark_wal_rotate(
    State(state): State<BenchmarkHttpState>,
) -> Result<Json<WalRotateResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    let archived = state
        .wal
        .rotate_archive()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(WalRotateResponse {
        archived_path: archived.map(|p| p.to_string_lossy().into_owned()),
    }))
}

async fn benchmark_vector_snapshot(
    State(state): State<BenchmarkHttpState>,
) -> Result<Json<VectorSnapshotWriteResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    drain_pending_builder(&state.index).await;
    let pairs = state
        .index
        .export_snapshot_vectors()
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let checkpoint = read_applied_lsn(&state.applied_sidecar)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let metric = state.index.distance_metric().as_str();
    let n = pairs.len();
    let items: Vec<VectorItem> = pairs
        .into_iter()
        .map(|(id, values)| VectorItem { id, values })
        .collect();
    let bytes = encode_vector_snapshot_v1(state.vector_dim, metric, checkpoint, items)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let bytes_written = bytes.len();
    // 加密层在写入函数内生成信封；`bytes` 只用于上报明文大小。
    // EN: the writer builds the encrypted envelope; `bytes` only feeds the
    // reported size.
    write_vector_snapshot_atomic_with_encryption(
        &state.wal_path,
        &bytes,
        state.encryption.as_deref(),
    )
    .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(VectorSnapshotWriteResponse {
        checkpoint_lsn: checkpoint,
        vector_count: n,
        bytes_written,
    }))
}

async fn benchmark_optimize(
    State(state): State<BenchmarkHttpState>,
) -> Result<Json<OptimizeResponse>, (axum::http::StatusCode, String)> {
    reject_if_read_only(&state)?;
    // ClusteredHnsw：从缓冲区构建（HNSW：空操作）。
    // EN: ClusteredHnsw: build from buffer (HNSW: no-op).
    let clustered_count = state.index.optimize_build().await;

    let mut flushed = clustered_count;
    let mut parallel_rounds: u32 = 0;
    let mut serial_rounds: u32 = 0;
    // 主路径：async builder worker 在进程内多轮刷队列；`optimize` 为兼容/手动用。
    // 若客户端仍用 POST 泵队列，本路由可在单次请求内多轮 `run_async_builder_once`，减少 HTTP 往返。
    // EN: Default build is the tokio worker; this POST can loop for legacy/ops (VectorDBBench may poll `index/status` only).
    const MAX_ROUNDS_PER_REQUEST: usize = 100_000;
    for _ in 0..MAX_ROUNDS_PER_REQUEST {
        let rep = state.index.run_async_builder_once().await;
        if rep.flushed == 0 {
            break;
        }
        flushed += rep.flushed;
        match rep.plan_mode {
            Some(HnswBatchPlanMode::Parallel) => {
                parallel_rounds = parallel_rounds.saturating_add(1)
            }
            Some(HnswBatchPlanMode::Serial) => serial_rounds = serial_rounds.saturating_add(1),
            None => {}
        }
    }
    let pending = state.index.pending_build_count().await;
    let resp = OptimizeResponse {
        flushed_items: flushed,
        pending_items: pending,
        hnsw_parallel_batch_rounds: parallel_rounds,
        hnsw_serial_batch_rounds: serial_rounds,
    };
    let log_cfg = OptimizeLogConfig::from_env();
    log_houyi_optimize_post_result(&resp, &log_cfg);
    Ok(Json(resp))
}

async fn benchmark_index_status(
    State(state): State<BenchmarkHttpState>,
) -> Result<Json<IndexStatusResponse>, (axum::http::StatusCode, String)> {
    let hot_count = state.index.hot_buffer_count().await;
    let indexed_count = state.index.indexed_count().await;
    let pending_count = state.index.pending_build_count().await;
    let bm = state.index.builder_metrics();
    Ok(Json(IndexStatusResponse {
        hot_count,
        indexed_count,
        pending_count,
        flush_count: bm.flush_count,
        total_flushed_items: bm.total_flushed_items,
        max_pending_queue: bm.max_pending_queue,
        last_flush_items: bm.last_flush_items,
        distance_metric: state.index.distance_metric().as_str().to_string(),
        hnsw_m: state.index.hnsw_m(),
        hnsw_ef_construction: state.index.hnsw_ef_construction_value(),
    }))
}

// 认证 + RBAC 中间件（P4.2.1 认证 → P4.7.1 授权）实现在 `crate::rbac`；
// 这里保留 re-export，bin target 的集成测试与路由装配按原路径使用。
// 契约不变：Bearer/X-Api-Key 二选一，探针豁免，未知凭据 401（带
// WWW-Authenticate），已知但越权 403。
// EN: authentication + authorization live in `crate::rbac`; re-exported here.
pub use crate::rbac::require_auth;

/// 在途请求水位线中间件（P4.2.3）：超限立即 503，许可随响应完成（RAII）归还。
/// 豁免 `/v1/health`、`/readyz`、`/metrics`——饱和时编排器探针与 Prometheus
/// 抓取必须仍可达，否则水位线会把自己打成不可观测、不可探活的黑盒。
/// EN: In-flight watermark middleware — immediate 503 on overflow, RAII release.
/// Probes and /metrics stay reachable under saturation by design.
pub async fn limit_http_inflight(
    axum::extract::State(limiter): axum::extract::State<Arc<InflightWatermark>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, (axum::http::StatusCode, &'static str)> {
    let path = req.uri().path();
    if path == "/v1/health" || path == "/readyz" || path == "/metrics" {
        return Ok(next.run(req).await);
    }
    match limiter.try_acquire() {
        Some(_permit) => Ok(next.run(req).await),
        None => Err((axum::http::StatusCode::SERVICE_UNAVAILABLE, "server busy")),
    }
}

/// 启动 Collection HTTP 服务。`collections` 是与 pgwire 共享的同一 CollectionManager
/// （默认索引后端已在 main 装配 Manager 时定型，两侧天然一致——旧实现 HTTP 侧自建
/// Manager 并硬编码 "flat"，既造成 HTTP/pgwire 数据分裂，又会把生产 `--index-type`
/// 静默降级）。
/// EN: `collections` is the single CollectionManager shared with pgwire (the default
/// backend is fixed when main assembles it, so both sides agree by construction —
/// the old code built a second manager here with hardcoded "flat", splitting
/// HTTP/pgwire state and silently downgrading `--index-type`).
///
/// 装配边界函数（main 唯一调用点），参数即各子系统句柄，不做无谓聚合。
/// `shutdown` 触发后 axum 进入优雅关停：停止 accept、等在途请求完成后返回（P4.0.3）。
/// 返回 `(监听地址, serve 任务句柄)`——句柄在 drain 完成时收尾，调用方必须
/// await，否则关停流程无法观测在途请求是否真正排空（OCR 裁决修复）。
/// Assembly boundary called only from main; args are subsystem handles, left ungrouped.
/// Once `shutdown` fires, axum stops accepting and drains in-flight requests; the
/// returned JoinHandle resolves when the drain completes and MUST be awaited.
#[allow(clippy::too_many_arguments)]
pub async fn run_benchmark_http_api<F>(
    addr: &str,
    index: IndexBackend,
    wal: Arc<WalHandle>,
    collections_wal: Arc<WalHandle>,
    lifecycle: Arc<crate::collection_lifecycle::CollectionLifecycleState>,
    collections: Arc<CollectionManager>,
    vector_dim: usize,
    pgwire_metrics: Arc<PgWireMetrics>,
    applied_sidecar: PathBuf,
    wal_path: PathBuf,
    openai_upstream: Option<Arc<crate::types::OpenAIUpstreamConfig>>,
    authz: Option<Arc<crate::rbac::AuthzPolicy>>,
    tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
    http_inflight: Arc<InflightWatermark>,
    graph_snapshot_metrics: Arc<crate::recovery::GraphSnapshotMetrics>,
    role: crate::types::RoleState,
    replication_metrics: crate::types::SharedReplicationMetrics,
    promote: Option<Arc<PromoteContext>>,
    audit: Option<Arc<crate::audit::AuditLog>>,
    encryption: Option<Arc<houyi_persistence::encryption::Encryption>>,
    cluster: Arc<crate::cluster::ClusterControl>,
    query_log: Option<Arc<crate::query_log::QueryLog>>,
    shutdown: F,
) -> std::io::Result<(SocketAddr, tokio::task::JoinHandle<()>)>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    // 集合配额中间件需要目录；state 会移走 collections，先留一份。
    // EN: The collection-quota middleware needs the directory; state takes
    // ownership of collections, so keep a clone first.
    let auth_collections = Arc::clone(&collections);
    let app = Router::new()
        .route("/v1/health", get(benchmark_health))
        .route("/readyz", get(benchmark_readyz))
        .route("/metrics", get(benchmark_metrics))
        .route("/v1/openapi.json", get(handle_openapi_spec))
        .route("/v1/hybrid/search", post(handle_collection_hybrid_search))
        .route("/v1/collections", get(handle_list_collections))
        // 显式建集合：度量/后端/HNSW 参数在创建时定型，而不是等首批 upsert 猜。
        // EN: explicit creation fixes metric/backend/HNSW up front.
        .route("/v1/collections/{name}", put(handle_create_collection))
        .route(
            "/v1/collections/{name}/hybrid/search",
            post(handle_collection_hybrid_search),
        )
        .route(
            "/v1/collections/{name}/upsert",
            post(handle_collection_upsert),
        )
        .route(
            "/v1/collections/{name}/search",
            post(handle_collection_search),
        )
        .route(
            "/v1/collections/{name}/delete",
            post(handle_collection_delete),
        )
        .route(
            "/v1/collections/{name}/ha_level",
            post(handle_collection_set_ha_level),
        )
        .route("/v1/version", get(handle_version))
        .route("/v1/admin/promote", post(handle_admin_promote))
        .route("/v1/admin/demote", post(handle_admin_demote))
        .route(
            "/v1/admin/lease",
            get(handle_admin_lease_status)
                .post(handle_admin_lease_grant)
                .delete(handle_admin_lease_revoke),
        )
        .route(
            "/v1/admin/replication/status",
            get(handle_admin_replication_status),
        )
        .route("/v1/admin/whoami", get(handle_admin_whoami))
        .route("/v1/admin/diagnostics", get(handle_admin_diagnostics))
        .route("/v1/admin/query-log", get(handle_admin_query_log))
        .route("/v1/admin/query-stats", get(handle_admin_query_stats))
        .route(
            "/v1/admin/collections/compact",
            // 静态段优先于 {name}：同时挂 DELETE，名为 `compact` 的 collection 才可删除。
            // EN: static segments win over {name}; mount DELETE here too so a
            // collection named `compact` remains droppable.
            post(handle_admin_collections_compact).delete(handle_admin_collection_delete_compact),
        )
        .route(
            "/v1/admin/collections/{name}",
            delete(handle_admin_collection_delete),
        )
        .route(
            "/v1/admin/audit/verify",
            get(crate::audit::handle_admin_audit_verify),
        )
        .route(
            "/v1/collections/{name}/status",
            get(handle_collection_status),
        )
        .route(
            "/v1/collections/{name}/export",
            get(handle_collection_export),
        )
        .route(
            "/v1/collections/{name}/compact",
            post(handle_collection_compact),
        )
        .route("/v1/vectors/upsert", post(benchmark_upsert))
        .route("/v1/vectors/search", post(benchmark_search))
        .route("/v1/optimize", post(benchmark_optimize))
        .route("/v1/index/status", get(benchmark_index_status))
        .route("/v1/wal/checkpoint", post(benchmark_wal_checkpoint))
        .route("/v1/wal/status", get(benchmark_wal_status))
        .route("/v1/pgwire/status", get(benchmark_pgwire_status))
        .route("/v1/wal/rotate", post(benchmark_wal_rotate))
        .route("/v1/wal/snapshot", post(benchmark_vector_snapshot))
        .route(
            "/v1/embeddings",
            post(crate::openai_gateway::handle_openai_embeddings),
        )
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(BenchmarkHttpState {
            index,
            collections,
            wal,
            collections_wal,
            lifecycle,
            vector_dim,
            pgwire_metrics,
            applied_sidecar,
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path,
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream,
            http_inflight: http_inflight.clone(),
            graph_snapshot_metrics,
            role,
            replication_metrics,
            promote,
            audit: audit.clone(),
            encryption,
            cluster,
            query_log,
        });
    let app = if let Some(policy) = authz {
        let auth_state = Arc::new(crate::rbac::AuthzState::new(policy, Some(auth_collections)));
        app.layer(axum::middleware::from_fn_with_state(
            auth_state,
            crate::rbac::require_auth,
        ))
    } else {
        app
    };
    // 审计在认证外侧：401/403 也必须入账（认证中间件短路后不会再进内层）。
    // EN: audit sits outside auth so 401/403 responses are recorded too.
    let app = app.layer(axum::middleware::from_fn_with_state(
        audit,
        crate::audit::audit_middleware,
    ));
    // 在途水位线（P4.2.3）最外层：503 拒绝比认证比较更便宜，饱和时先挡。
    // EN: In-flight watermark outermost — a 503 is cheaper than the auth check.
    let app = app.layer(axum::middleware::from_fn_with_state(
        http_inflight,
        limit_http_inflight,
    ));
    // TLS（P4.2.2）：配置 acceptor 时经 TlsHttpListener 在每个连接上先做带超时的
    // TLS 握手再交给 hyper；未配置则保持明文直通。
    // EN: With a TLS acceptor, TlsHttpListener performs a bounded TLS handshake
    // per connection before hyper sees it; otherwise plaintext as before.
    let serve_task = tokio::spawn(async move {
        let result = match tls_acceptor {
            Some(acceptor) => {
                axum::serve(crate::tls::TlsHttpListener::new(listener, acceptor), app)
                    .with_graceful_shutdown(shutdown)
                    .await
            }
            None => {
                axum::serve(listener, app)
                    .with_graceful_shutdown(shutdown)
                    .await
            }
        };
        if let Err(e) = result {
            tracing::error!(error = %e, "benchmark http server stopped");
        }
    });
    Ok((local_addr, serve_task))
}

#[cfg(test)]
mod score_edge_tests {
    use super::*;

    /// 余弦语义后端：f32 舍入可能让相同向量的预归一化距离略负（如 -1e-7），
    /// 裸换算会得到 1.0000001；边缘必须 clamp 到 [-1,1]，相同向量 score 恰为 1。
    /// EN: Cosine backends clamp rounding-induced overshoot (distance -1e-7 →
    /// raw score 1.0000001) so identical vectors score exactly 1.
    #[test]
    fn cosine_scores_clamped_to_unit_interval() {
        let pairs = vec![(1_u64, -1e-7_f32), (2, 1.0_f32), (3, 2.0_f32)];
        let resp = scored_pairs_to_response(pairs, DistanceMetric::Cosine);
        assert_eq!(resp.ids, vec![1, 2, 3]);
        assert_eq!(resp.scores[0], 1.0, "identical-vector score must be <= 1");
        assert!(resp.scores[1].abs() < 1e-7);
        assert!((resp.scores[2] + 1.0).abs() < 1e-7);
        for score in &resp.scores {
            assert!(
                (-1.0..=1.0).contains(score),
                "cosine score {score} out of [-1,1]"
            );
        }
    }

    /// l2/l1 的分数是 `1 - distance`（L2² / 曼哈顿距离合法越界），禁止 clamp。
    /// EN: l2/l1 scores are `1 - distance` (squared-L2 / Manhattan distances
    /// legitimately fall outside [-1,1]); they must pass through unclamped.
    #[test]
    fn l2_and_l1_scores_pass_through_without_clamp() {
        let pairs = vec![(1_u64, 0.5_f32), (2, 5.0_f32)];
        let resp = scored_pairs_to_response(pairs.clone(), DistanceMetric::L2);
        assert_eq!(resp.ids, vec![1, 2]);
        assert!((resp.scores[0] - 0.5).abs() < 1e-7);
        assert!(
            (resp.scores[1] - (-4.0)).abs() < 1e-7,
            "1 - L2² must stay out-of-range"
        );
        let resp = scored_pairs_to_response(pairs, DistanceMetric::L1);
        assert!((resp.scores[0] - 0.5).abs() < 1e-7);
        assert!(
            (resp.scores[1] - (-4.0)).abs() < 1e-7,
            "1 - L1 must stay out-of-range"
        );
    }

    /// ip 的内核距离是 `-dot`，分数取反即原始内积（越大越相似，与 Milvus 一致）。
    /// EN: the ip kernel distance is `-dot`, so negating yields the raw inner
    /// product (larger = closer, matching Milvus).
    #[test]
    fn ip_scores_are_raw_dot_product() {
        let pairs = vec![(1_u64, -5.0_f32), (2, 0.0_f32), (3, 2.5_f32)];
        let resp = scored_pairs_to_response(pairs, DistanceMetric::InnerProduct);
        assert_eq!(resp.ids, vec![1, 2, 3]);
        assert!((resp.scores[0] - 5.0).abs() < 1e-7);
        assert!(resp.scores[1].abs() < 1e-7);
        assert!((resp.scores[2] + 2.5).abs() < 1e-7);
    }

    /// NaN/Inf 防御：pg_wire binary float4/float8 可写入非有限值；serde_json 会把
    /// 非有限浮点序列化为 `null`（合法 JSON 但破坏 scores 数值契约）。非有限命中必须
    /// 成对丢弃且不错位。
    /// EN: Non-finite distances (storable via pg_wire binary floats) serialize as
    /// `null` (valid JSON that breaks the numeric contract); drop those pairs and
    /// keep ids/scores aligned.
    #[test]
    fn non_finite_distances_are_filtered() {
        let pairs = vec![
            (1_u64, 0.1_f32),
            (2, f32::NAN),
            (3, f32::INFINITY),
            (4, f32::NEG_INFINITY),
            (5, 0.3_f32),
        ];
        let resp = scored_pairs_to_response(pairs, DistanceMetric::Cosine);
        assert_eq!(resp.ids, vec![1, 5]);
        assert_eq!(
            resp.scores.len(),
            2,
            "ids/scores stay aligned after filtering"
        );
        assert!(resp.scores.iter().all(|s| s.is_finite()));
        assert!((resp.scores[0] - 0.9).abs() < 1e-6);
        assert!((resp.scores[1] - 0.7).abs() < 1e-6);
    }

    /// FIX-5：HNSW 分数锚点——相同 ≈ 1（且严格 ≤ 1）、正交 ≈ 0、反向 ≈ -1。
    /// EN: HNSW score anchors — identical ≈ 1 (and strictly ≤ 1), orthogonal ≈ 0,
    /// opposite ≈ -1.
    #[tokio::test]
    async fn hnsw_score_anchors_identical_orthogonal_opposite() {
        let backend = IndexBackend::new_hnsw(4);
        backend
            .insert_batch(vec![
                (1, vec![1.0, 0.0, 0.0, 0.0]),
                (2, vec![0.0, 1.0, 0.0, 0.0]),
                (3, vec![-1.0, 0.0, 0.0, 0.0]),
            ])
            .await
            .unwrap();
        for _ in 0..1000 {
            if backend.run_async_builder_once().await.flushed == 0 {
                break;
            }
        }
        let pairs = backend
            .search_scored_with_ef_override(&[1.0, 0.0, 0.0, 0.0], 3, None, Some(256))
            .await;
        assert_eq!(pairs.len(), 3);
        let resp = scored_pairs_to_response(pairs, DistanceMetric::Cosine);
        let score_of = |id: u64| {
            let pos = resp.ids.iter().position(|x| *x == id).unwrap();
            resp.scores[pos]
        };
        let identical = score_of(1);
        let orthogonal = score_of(2);
        let opposite = score_of(3);
        assert!(
            identical <= 1.0,
            "identical score must be exactly <= 1.0, got {identical}"
        );
        assert!(
            (identical - 1.0).abs() < 1e-4,
            "identical score ~1, got {identical}"
        );
        assert!(
            orthogonal.abs() < 1e-4,
            "orthogonal score ~0, got {orthogonal}"
        );
        assert!(
            opposite >= -1.0,
            "opposite score must be >= -1.0, got {opposite}"
        );
        assert!(
            (opposite + 1.0).abs() < 1e-4,
            "opposite score ~-1, got {opposite}"
        );
    }
}

#[cfg(test)]
mod collection_status_tests {
    use super::*;
    use axum::extract::Path;

    /// 临时文件守卫：断言失败提前 unwind 时也删除 WAL/sidecar，避免 /tmp 泄漏。
    /// 与 src/tests.rs 中 collection 测试使用同一 RAII 模式（happy path 末尾
    /// 手工删除会在 panic 路径漏删）。
    /// Drop guard for temp WAL/sidecar files: cleanup still runs when an assert
    /// fails — same RAII pattern as the collection tests in src/tests.rs.
    struct TempFileCleanup(Vec<std::path::PathBuf>);
    impl Drop for TempFileCleanup {
        fn drop(&mut self) {
            for path in &self.0 {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    fn temp_state(index_type: &str) -> (BenchmarkHttpState, TempFileCleanup) {
        let wal_path = std::env::temp_dir().join(format!(
            "houyidb-status-test-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let sidecar = wal_path.with_extension("applied");
        let wal = Arc::new(WalHandle::open(&wal_path, WalWriteMode::Async).unwrap().0);
        let collections_wal_path =
            houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
        let collections_wal = Arc::new(
            WalHandle::open(&collections_wal_path, WalWriteMode::Async)
                .unwrap()
                .0,
        );
        let dims = 4;
        let state = BenchmarkHttpState {
            promote: None,
            audit: None,
            encryption: None,
            index: IndexBackend::new_hnsw(dims),
            collections: Arc::new(CollectionManager::new(dims, index_type)),
            wal,
            collections_wal,
            vector_dim: dims,
            pgwire_metrics: Arc::new(PgWireMetrics::default()),
            applied_sidecar: sidecar.clone(),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path: wal_path.clone(),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
            graph_snapshot_metrics: Arc::new(crate::recovery::GraphSnapshotMetrics::default()),
            role: crate::types::RoleState::default(),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: crate::types::shared_replication_metrics(None),
            lifecycle: std::sync::Arc::new(
                crate::collection_lifecycle::CollectionLifecycleState::for_test(),
            ),
        };
        (
            state,
            TempFileCleanup(vec![wal_path, sidecar, collections_wal_path]),
        )
    }

    async fn upsert_rows(state: &BenchmarkHttpState, rows: &[(u64, bool)]) {
        let vectors = rows
            .iter()
            .map(|(id, with_payload)| UpsertVector {
                id: *id,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: if *with_payload {
                    Some(
                        serde_json::json!({"color": "red"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    )
                } else {
                    None
                },
            })
            .collect();
        let _ = handle_collection_upsert(
            State(state.clone()),
            Path("st".to_string()),
            Json(UpsertRequest {
                vectors,
                index: None,
            }),
        )
        .await
        .unwrap();
        // 单测无异步 builder worker：同步排空，使 indexed_count 可见。
        // No async builder worker in unit tests: drain synchronously.
        if let Some(col) = state.collections.get("st") {
            for _ in 0..1000 {
                if col.backend.run_async_builder_once().await.flushed == 0 {
                    break;
                }
            }
        }
    }

    #[tokio::test]
    async fn status_reports_scalar_row_count_for_payload_vectors_only() {
        let (state, _cleanup) = temp_state("hnsw");

        // 先写一个裸向量：Collection 已建但标量页应为 0 行。
        // A bare vector first: collection exists, scalar page must still show 0 rows.
        upsert_rows(&state, &[(1, false)]).await;
        let status = handle_collection_status(State(state.clone()), Path("st".to_string()))
            .await
            .unwrap()
            .0;
        assert_eq!(status["scalar_row_count"], 0);
        assert_eq!(status["indexed_count"], 1);

        // 再写一个裸向量 + 两个 payload 向量：只有后两者建行。
        // One bare + two payload vectors: only the payload ones add scalar rows.
        upsert_rows(&state, &[(2, false), (3, true), (4, true)]).await;
        // 只读 ha_level 必须反映 WAL-first 变更后的级别。
        // EN: the read-only ha_level must reflect the WAL-first change.
        state
            .collections
            .get("st")
            .expect("collection")
            .set_ha_level(houyi_persistence::ha_level::HaLevel::SemiSync);
        let status = handle_collection_status(State(state.clone()), Path("st".to_string()))
            .await
            .unwrap()
            .0;
        assert_eq!(status["scalar_row_count"], 2);
        assert_eq!(status["indexed_count"], 4);
        assert_eq!(status["ha_level"], "semi_sync");
        // WAL/sidecar 由 TempFileCleanup 在 drop 时删除（panic 路径同样执行）。
        // The guard removes the WAL/sidecar on drop, including the panic path.
    }

    /// 生产装配级认证守卫（OCR 359d85d[1]）：tests.rs 的迷你路由只验证
    /// require_auth 中间件本体；本测试驱动 `run_benchmark_http_api` 的真实
    /// 装配——axum 的 layer 只包裹其注册之前的路由，若未来有人在 layer 之后
    /// 注册新路由、或反转 auth_token 条件，迷你路由测试全绿而生产裸奔，
    /// 只有这个 wiring 级测试会红。
    /// EN: Wiring-level guard: drives the real assembly so a future regression
    /// in route/layer registration order (or an inverted auth condition) fails
    /// here instead of shipping silently.
    #[tokio::test]
    async fn production_router_applies_require_auth_when_token_set() {
        let (state, _cleanup) = temp_state("hnsw");
        let (addr, _server) = run_benchmark_http_api(
            "127.0.0.1:0",
            state.index.clone(),
            state.wal.clone(),
            state.collections_wal.clone(),
            state.lifecycle.clone(),
            state.collections.clone(),
            state.vector_dim,
            state.pgwire_metrics.clone(),
            state.applied_sidecar.clone(),
            state.wal_path.clone(),
            None,
            Some(Arc::new(
                crate::rbac::AuthzPolicy::from_config(Some("s3cret"), &[]).expect("legacy policy"),
            )),
            None,
            state.http_inflight.clone(),
            state.graph_snapshot_metrics.clone(),
            state.role.clone(),
            std::sync::Arc::clone(&state.replication_metrics),
            None,
            None,
            None,
            test_cluster(),
            None,
            std::future::pending(),
        )
        .await
        .expect("test server must bind");

        // ureq 是同步客户端：真实 HTTP 调用放进 spawn_blocking。
        // EN: ureq is a synchronous client: run the real HTTP calls inside
        // spawn_blocking.
        let base = format!("http://{addr}");
        tokio::task::spawn_blocking(move || {
            let get = |path: &str, auth: Option<&str>| {
                let mut req = ureq::get(&format!("{base}{path}"));
                if let Some(v) = auth {
                    req = req.header("authorization", v);
                }
                match req.call() {
                    Ok(r) => r.status().as_u16(),
                    Err(ureq::Error::StatusCode(code)) => code,
                    Err(e) => panic!("GET {path} transport failed: {e}"),
                }
            };
            // 探针豁免（orchestrator 不带凭据）。
            // EN: Probe exemption (the orchestrator carries no credentials).
            assert_eq!(get("/v1/health", None), 200);
            // /metrics 与业务路由均在认证层内。
            // EN: /metrics and business routes are both inside the auth layer.
            assert_eq!(get("/metrics", None), 401);
            assert_eq!(get("/v1/collections", None), 401);
            assert_eq!(get("/metrics", Some("Bearer s3cret")), 200);
            assert_eq!(get("/v1/collections", Some("Bearer s3cret")), 200);
        })
        .await
        .expect("blocking assertions must not panic");
    }

    /// P4.7.0 wiring：真实路由 + 审计中间件 —— 成功的写与 401 都必须入账，
    /// 且原始 token 绝不落盘。
    /// EN: real router + audit middleware — a successful write and a 401 are
    /// both recorded, and the raw token never reaches the log.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn production_router_writes_audit_entries() {
        let (state, _cleanup) = temp_state("hnsw");
        let audit_path = std::env::temp_dir().join(format!(
            "houyidb-audit-wiring-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _audit_cleanup = TempFileCleanup(vec![audit_path.clone()]);
        let audit = Arc::new(
            crate::audit::AuditLog::open(audit_path.clone(), false, 0)
                .expect("audit log must open"),
        );
        let (addr, server) = run_benchmark_http_api(
            "127.0.0.1:0",
            state.index.clone(),
            state.wal.clone(),
            state.collections_wal.clone(),
            state.lifecycle.clone(),
            state.collections.clone(),
            state.vector_dim,
            state.pgwire_metrics.clone(),
            state.applied_sidecar.clone(),
            state.wal_path.clone(),
            None,
            Some(Arc::new(
                crate::rbac::AuthzPolicy::from_config(Some("s3cret"), &[]).expect("legacy policy"),
            )),
            None,
            state.http_inflight.clone(),
            state.graph_snapshot_metrics.clone(),
            state.role.clone(),
            std::sync::Arc::clone(&state.replication_metrics),
            None,
            Some(Arc::clone(&audit)),
            None,
            test_cluster(),
            None,
            std::future::pending(),
        )
        .await
        .expect("test server must bind");

        let base = format!("http://{addr}");
        tokio::task::spawn_blocking(move || {
            // 未认证读 → 401，必须入账（审计在认证外层）。
            // EN: Unauthenticated read → 401, and it must be audited (audit sits
            // outside auth).
            let code = match ureq::get(&format!("{base}/v1/collections")).call() {
                Ok(r) => r.status().as_u16(),
                Err(ureq::Error::StatusCode(code)) => code,
                Err(e) => panic!("GET transport failed: {e}"),
            };
            assert_eq!(code, 401);
            // 认证写 → 200，必须入账。
            // EN: Authenticated write → 200, and it must be audited.
            let body = r#"{"vectors":[{"id":1,"values":[1.0,0.0,0.0,0.0]}]}"#;
            let resp = ureq::post(&format!("{base}/v1/collections/docs/upsert"))
                .header("authorization", "Bearer s3cret")
                .header("content-type", "application/json")
                .send(body)
                .expect("authenticated upsert must succeed");
            assert_eq!(resp.status().as_u16(), 200);
        })
        .await
        .expect("blocking assertions must not panic");

        let report = audit.verify().await.expect("verify");
        assert!(report.valid);
        assert_eq!(report.entries, 2, "401 + successful write must both audit");
        let text = std::fs::read_to_string(&audit_path).unwrap();
        assert!(!text.contains("s3cret"), "raw token must never be logged");
        // RBAC 后 actor 记录人可读主体名（legacy token = legacy-admin）。
        // EN: After RBAC the actor records the human-readable principal
        // (legacy token = legacy-admin).
        assert!(
            text.contains("principal:legacy-admin"),
            "audit must record the authenticated principal, got: {text}"
        );
        assert!(text.contains("/v1/collections/docs/upsert"));
        server.abort();
    }

    /// 导出流：header/记录/footer 完整，按 id 升序，payload 与向量可回读。
    /// EN: export stream carries header/records/footer, sorted by id, with the
    /// vector and payload round-tripping.
    #[tokio::test]
    async fn export_streams_sorted_records_with_payloads() {
        let (state, _cleanup) = temp_state("hnsw");
        upsert_rows(&state, &[(1, true), (2, false), (3, true)]).await;
        let response = handle_collection_export(State(state.clone()), Path("st".to_string()), None)
            .await
            .expect("export");
        let bytes = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
            .await
            .expect("body");
        let text = std::str::from_utf8(&bytes).expect("utf8");
        let mut lines = text.lines();
        let header: serde_json::Value =
            serde_json::from_str(lines.next().expect("header")).expect("header json");
        assert_eq!(header["format"], "houyidb-export");
        assert_eq!(header["version"], 1);
        assert_eq!(header["collection"], "st");
        assert_eq!(header["dims"], 4);
        assert_eq!(header["count"], 3);

        let mut ids = Vec::new();
        let mut footer = None;
        for line in lines {
            let record: serde_json::Value = serde_json::from_str(line).expect("record json");
            if record.get("end") == Some(&serde_json::Value::Bool(true)) {
                footer = Some(record);
                continue;
            }
            assert_eq!(record["vector"].as_array().map(Vec::len), Some(4));
            let id = record["id"].as_u64().expect("id");
            if id == 1 || id == 3 {
                assert_eq!(record["payload"]["color"], "red");
            } else {
                assert!(record["payload"].is_null());
            }
            ids.push(id);
        }
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(footer.expect("footer")["count"], 3);

        // 作用域外的读取主体被拒绝。 / EN: out-of-scope principals are rejected.
        let principal = crate::rbac::AuthenticatedPrincipal {
            name: "tenant".to_string(),
            role: crate::rbac::Role::Reader,
            collections: Some(vec!["other".to_string()]),
            max_collections: None,
        };
        let denied = handle_collection_export(
            State(state),
            Path("st".to_string()),
            Some(axum::extract::Extension(principal)),
        )
        .await
        .expect_err("scope must be enforced");
        assert_eq!(denied.0, axum::http::StatusCode::FORBIDDEN);
    }

    /// 非 HNSW 后端按插入顺序存储；导出必须仍按 id 升序输出。
    /// EN: non-HNSW backends store insertion order; exports stay id-sorted.
    #[tokio::test]
    async fn export_sorts_flat_backend_records() {
        let (state, _cleanup) = temp_state("flat");
        upsert_rows(&state, &[(3, true), (1, false), (2, true)]).await;
        let response = handle_collection_export(State(state), Path("st".to_string()), None)
            .await
            .expect("export");
        let bytes = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
            .await
            .expect("body");
        let text = std::str::from_utf8(&bytes).expect("utf8");
        let ids: Vec<u64> = text
            .lines()
            .skip(1)
            .filter_map(|line| {
                let record: serde_json::Value = serde_json::from_str(line).expect("record json");
                (record.get("end").is_none())
                    .then(|| record["id"].as_u64())
                    .flatten()
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    /// P4.7.2 作用域过滤：受限主体只看到自己的 collection 名单。
    /// EN: scoped principals only see their own collections in the list endpoint.
    #[tokio::test]
    async fn list_collections_filters_by_principal_scope() {
        let (state, _cleanup) = temp_state("hnsw");
        state.collections.get_or_create("alpha", Some(4)).unwrap();
        state.collections.get_or_create("beta", Some(4)).unwrap();
        state
            .collections
            .get("alpha")
            .expect("alpha")
            .set_ha_level(houyi_persistence::ha_level::HaLevel::FullSync);
        let principal = crate::rbac::AuthenticatedPrincipal {
            name: "tenant".to_string(),
            role: crate::rbac::Role::Reader,
            collections: Some(vec!["alpha".to_string()]),
            max_collections: None,
        };
        let Json(infos) =
            handle_list_collections(State(state), Some(axum::extract::Extension(principal))).await;
        let names: Vec<&str> = infos
            .iter()
            .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
            .collect();
        assert_eq!(names, vec!["alpha"], "only in-scope names may be listed");
        assert_eq!(
            infos[0]["ha_level"], "full_sync",
            "the list view must expose the read-only ha_level"
        );
    }

    /// 管理删除：成功返回 `deleted:true` 并摘除内存；重复删除 404。
    /// EN: admin drop returns `deleted:true` and removes the collection;
    /// a repeat drop is 404.
    #[tokio::test]
    async fn admin_drop_removes_collection_and_repeat_is_404() {
        let (state, _cleanup) = temp_state("hnsw");
        upsert_rows(&state, &[(1, true), (2, false)]).await;

        let Json(resp) =
            handle_admin_collection_delete(State(state.clone()), Path("st".to_string()))
                .await
                .expect("drop must succeed");
        assert!(resp.deleted);
        assert_eq!(resp.collection, "st");
        assert!(
            state.collections.get("st").is_none(),
            "dropped collection must leave the catalog"
        );
        let err = handle_collection_status(State(state.clone()), Path("st".to_string()))
            .await
            .expect_err("status of a dropped collection must be 404");
        assert_eq!(err.0, axum::http::StatusCode::NOT_FOUND);

        let err = handle_admin_collection_delete(State(state), Path("st".to_string()))
            .await
            .expect_err("repeat drop must be 404");
        assert_eq!(err.0, axum::http::StatusCode::NOT_FOUND);
    }

    /// 名为 `compact` 的 collection 必须能经静态路由删除（否则 {name} 被遮蔽）。
    /// EN: a collection literally named `compact` must be droppable through the
    /// static route (otherwise {name} is shadowed).
    #[tokio::test]
    async fn compact_named_collection_drops_via_static_route() {
        let (state, _cleanup) = temp_state("hnsw");
        state
            .collections
            .get_or_create("compact", Some(4))
            .expect("create compact collection");

        let Json(resp) = handle_admin_collection_delete_compact(State(state.clone()))
            .await
            .expect("drop must succeed");
        assert_eq!(resp.collection, "compact");
        assert!(state.collections.get("compact").is_none());
    }

    /// 删除后同名重建：得到全新空 collection，旧向量不复活。
    /// EN: a same-name recreate after a drop yields a fresh empty collection;
    /// old vectors must not resurrect.
    #[tokio::test]
    async fn dropped_collection_recreates_empty_without_old_vectors() {
        let (state, _cleanup) = temp_state("hnsw");
        upsert_rows(&state, &[(1, true), (2, true)]).await;
        let _ = handle_admin_collection_delete(State(state.clone()), Path("st".to_string()))
            .await
            .expect("drop");

        let _ = handle_collection_upsert(
            State(state.clone()),
            Path("st".to_string()),
            Json(UpsertRequest {
                vectors: vec![UpsertVector {
                    id: 9,
                    values: vec![0.0, 1.0, 0.0, 0.0],
                    label: None,
                    payload: None,
                }],
                index: None,
            }),
        )
        .await
        .expect("recreate");
        let col = state.collections.get("st").expect("recreated collection");
        for _ in 0..1000 {
            if col.backend.run_async_builder_once().await.flushed == 0 {
                break;
            }
        }
        let status = handle_collection_status(State(state), Path("st".to_string()))
            .await
            .expect("status")
            .0;
        assert_eq!(status["indexed_count"], 1);
        assert_eq!(
            status["scalar_row_count"], 0,
            "old payload rows must not survive the drop"
        );
    }

    /// follower 是只读副本：删除必须 503 且不改变内存状态。
    /// EN: a follower is read-only — the drop must 503 and leave memory untouched.
    #[tokio::test]
    async fn admin_drop_on_follower_is_503() {
        let (state, _cleanup) = temp_state("hnsw");
        upsert_rows(&state, &[(1, false)]).await;
        state.role.set(crate::types::ReplicationRole::Follower);
        let err = handle_admin_collection_delete(State(state.clone()), Path("st".to_string()))
            .await
            .expect_err("follower must reject the drop");
        assert_eq!(err.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            state.collections.get("st").is_some(),
            "a rejected drop must not remove the collection"
        );
    }

    /// 删除与写入竞争：删除等待写锁，拿锁后置位哨兵；同一 Arc 上排队的写
    /// 在双检时被 409 拒绝（不 panic、不死锁）。
    /// EN: drop/write race — the drop waits for the lock, sets the sentinel,
    /// and the queued writer is rejected with 409 by the double check.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_upsert_behind_drop_is_409() {
        let (state, _cleanup) = temp_state("hnsw");
        upsert_rows(&state, &[(1, false)]).await;
        // 预先持锁，让删除与写入按 FIFO 排队：删除先入队，写入后入队。
        // EN: hold the lock first so the FIFO queue order is drop then write.
        let col = state.collections.get("st").expect("collection");
        let guard = col.write_lock.lock().await;

        let delete_state = state.clone();
        let delete = tokio::spawn(async move {
            handle_admin_collection_delete(State(delete_state), Path("st".to_string())).await
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let upsert_state = state.clone();
        let upsert = tokio::spawn(async move {
            handle_collection_upsert(
                State(upsert_state),
                Path("st".to_string()),
                Json(UpsertRequest {
                    vectors: vec![UpsertVector {
                        id: 2,
                        values: vec![0.0, 1.0, 0.0, 0.0],
                        label: None,
                        payload: None,
                    }],
                    index: None,
                }),
            )
            .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        drop(guard);
        let _ = delete
            .await
            .expect("delete task panicked")
            .expect("drop must succeed");
        match upsert.await.expect("upsert task panicked") {
            Err((status, msg)) => {
                assert_eq!(status, axum::http::StatusCode::CONFLICT);
                assert!(msg.contains("being deleted"), "unexpected message: {msg}");
            }
            Ok(_) => panic!("a queued write must not succeed on a dropped collection"),
        }
        assert!(state.collections.get("st").is_none());
    }
}

/// 检索 k 上界（OCR：裸 usize k 的内存放大攻击面）回归测试。
/// EN: Regression tests for the search-k upper bound (raw usize k was a memory
/// amplification attack surface: 4k over-fetch + ~16×k HNSW preallocation).
#[cfg(test)]
mod search_k_bound_tests {
    use super::*;
    use axum::extract::Path;

    /// 临时文件守卫：panic 路径也清理 WAL/sidecar，避免 /tmp 泄漏。
    /// Drop guard for temp WAL/sidecar files (panic-safe cleanup).
    struct TempFileCleanup(Vec<std::path::PathBuf>);
    impl Drop for TempFileCleanup {
        fn drop(&mut self) {
            for path in &self.0 {
                // best-effort：异步 WalHandle 可能仍持有文件句柄，个别平台上删除会
                // 静默失败；临时目录最终由系统回收，错误有意忽略。
                // best-effort: the async WalHandle may still hold the file open and
                // removal can fail silently on file-locking platforms; the error is
                // intentionally ignored — temp dirs are reclaimed by the OS anyway.
                let _ = std::fs::remove_file(path);
            }
        }
    }

    fn temp_state() -> (BenchmarkHttpState, TempFileCleanup) {
        let wal_path = std::env::temp_dir().join(format!(
            "houyidb-kbound-test-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let sidecar = wal_path.with_extension("applied");
        let wal = Arc::new(WalHandle::open(&wal_path, WalWriteMode::Async).unwrap().0);
        let collections_wal_path =
            houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
        let collections_wal = Arc::new(
            WalHandle::open(&collections_wal_path, WalWriteMode::Async)
                .unwrap()
                .0,
        );
        let dims = 4;
        let state = BenchmarkHttpState {
            promote: None,
            audit: None,
            encryption: None,
            index: IndexBackend::new_hnsw(dims),
            collections: Arc::new(CollectionManager::new(dims, "hnsw")),
            wal,
            collections_wal,
            vector_dim: dims,
            pgwire_metrics: Arc::new(PgWireMetrics::default()),
            applied_sidecar: sidecar.clone(),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path: wal_path.clone(),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
            graph_snapshot_metrics: Arc::new(crate::recovery::GraphSnapshotMetrics::default()),
            role: crate::types::RoleState::default(),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: crate::types::shared_replication_metrics(None),
            lifecycle: std::sync::Arc::new(
                crate::collection_lifecycle::CollectionLifecycleState::for_test(),
            ),
        };
        (
            state,
            TempFileCleanup(vec![wal_path, sidecar, collections_wal_path]),
        )
    }

    fn search_req(k: usize) -> SearchRequest {
        SearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k,
            filter_gt_id: None,
            ef_search: None,
            filter_label: None,
            include_payload: false,
        }
    }

    fn hybrid_req(k: usize) -> HybridSearchRequest {
        HybridSearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k,
            filters: Vec::new(),
            ef_search: None,
            include_payload: false,
        }
    }

    /// benchmark 路径：超上界 k 在 HTTP 边缘 400，错误信息带上界数值。
    /// EN: Benchmark path — over-cap k is rejected at the edge with 400 and a
    /// message naming the bound.
    #[tokio::test]
    async fn benchmark_search_rejects_k_above_cap() {
        let (state, _cleanup) = temp_state();
        let err = benchmark_search(State(state), None, Json(search_req(MAX_SEARCH_K + 1)))
            .await
            .expect_err("over-cap k must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.1.contains(&MAX_SEARCH_K.to_string()),
            "error must name the bound, got: {}",
            err.1
        );
    }

    /// benchmark 路径刻意保留 VectorDBBench 兼容语义：k==0 → 200 空响应（不是 400）。
    /// EN: Benchmark path deliberately keeps VectorDBBench compat: k==0 → 200 empty.
    #[tokio::test]
    async fn benchmark_search_k_zero_stays_empty_200() {
        let (state, _cleanup) = temp_state();
        let resp = benchmark_search(State(state), None, Json(search_req(0)))
            .await
            .expect("k==0 stays 200-empty on the benchmark path")
            .0;
        assert!(resp.ids.is_empty());
        assert!(resp.scores.is_empty());
    }

    /// 恰好等于上界的 k 必须放行（空索引返回空结果，仅验证不被边缘误杀）。
    /// EN: k exactly at the cap passes (empty index → empty result; the edge
    /// must not reject the boundary value).
    #[tokio::test]
    async fn benchmark_search_accepts_k_at_cap() {
        let (state, _cleanup) = temp_state();
        let Json(resp) = benchmark_search(State(state), None, Json(search_req(MAX_SEARCH_K)))
            .await
            .expect("k == MAX_SEARCH_K must be accepted");
        assert!(resp.ids.is_empty());
        assert!(resp.scores.is_empty());
    }

    /// ef_search 是 k 的兄弟放大面（ann_k = k.max(ef) → ~16×ann_k 访问器预分配）：
    /// 三条 HTTP 路径超上界一律 400，错误信息带上界数值。
    /// EN: ef_search is the sibling amplification field (ann_k = k.max(ef) drives
    /// ~16×ann_k visit-tracker preallocation): over-cap ef is 400 on all three paths.
    #[tokio::test]
    async fn benchmark_search_rejects_ef_above_cap() {
        let (state, _cleanup) = temp_state();
        let req = SearchRequest {
            ef_search: Some(MAX_SEARCH_EF + 1),
            ..search_req(10)
        };
        let err = benchmark_search(State(state), None, Json(req))
            .await
            .expect_err("over-cap ef_search must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.1.contains(&MAX_SEARCH_EF.to_string()), "got: {}", err.1);
    }

    #[tokio::test]
    async fn collection_search_rejects_ef_above_cap() {
        let (state, _cleanup) = temp_state();
        state.collections.get_or_create("c", Some(4)).unwrap();
        let req = SearchRequest {
            ef_search: Some(MAX_SEARCH_EF + 1),
            ..search_req(10)
        };
        let err =
            handle_collection_search(State(state.clone()), Path("c".to_string()), None, Json(req))
                .await
                .expect_err("over-cap ef_search must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.1.contains(&MAX_SEARCH_EF.to_string()), "got: {}", err.1);
    }

    #[tokio::test]
    async fn hybrid_search_rejects_ef_above_cap() {
        let (state, _cleanup) = temp_state();
        state.collections.get_or_create("c", Some(4)).unwrap();
        let req = HybridSearchRequest {
            ef_search: Some(MAX_SEARCH_EF + 1),
            ..hybrid_req(10)
        };
        let err = handle_collection_hybrid_search(
            State(state.clone()),
            Path("c".to_string()),
            None,
            Json(req),
        )
        .await
        .expect_err("over-cap ef_search must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.1.contains(&MAX_SEARCH_EF.to_string()), "got: {}", err.1);
    }

    /// 与协议无关的统一封边判定：k/ef 各自封上界，k==0 不在此判定（k==0 的语义
    /// 分歧由各边缘自行处理），恰好等于上界放行。
    /// EN: Protocol-agnostic edge check: caps both k and ef; k==0 is deliberately
    /// not handled here (paths diverge), boundary values pass.
    #[test]
    fn search_limit_violation_flags_only_over_cap_values() {
        assert!(
            search_limit_violation(0, None).is_none(),
            "k==0 not this helper's concern"
        );
        assert!(search_limit_violation(1, None).is_none());
        assert!(search_limit_violation(MAX_SEARCH_K, Some(MAX_SEARCH_EF)).is_none());
        let msg = search_limit_violation(MAX_SEARCH_K + 1, None).unwrap();
        assert!(msg.contains(&MAX_SEARCH_K.to_string()));
        let msg = search_limit_violation(10, Some(MAX_SEARCH_EF + 1)).unwrap();
        assert!(msg.contains(&MAX_SEARCH_EF.to_string()));
    }

    /// collection 路径：k==0 与超上界均为 400（业务路径，无 benchmark 兼容特例）。
    /// EN: Collection path — zero and over-cap k both yield 400 (no benchmark
    /// compat carve-out on a business-facing path).
    #[tokio::test]
    async fn collection_search_rejects_zero_and_oversize_k() {
        let (state, _cleanup) = temp_state();
        state.collections.get_or_create("c", Some(4)).unwrap();
        for bad_k in [0usize, MAX_SEARCH_K + 1] {
            let err = handle_collection_search(
                State(state.clone()),
                Path("c".to_string()),
                None,
                Json(search_req(bad_k)),
            )
            .await
            .expect_err("k boundary must be rejected");
            assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST, "k={bad_k}");
            // 按越界情形断言特征子串：零值必须点明下界，超上界必须带出上界数值。
            // EN: Assert a distinctive substring per case: the floor for zero, the
            // numeric bound for over-cap — a bare contains('k') would catch neither drift.
            let expected = if bad_k == 0 {
                "must be >= 1"
            } else {
                &MAX_SEARCH_K.to_string()
            };
            assert!(
                err.1.contains(expected),
                "error must name the enforced limit, got: {}",
                err.1
            );
        }
    }

    /// hybrid 路径与普通 collection 检索同口径：k==0 与超上界均为 400，
    /// 且该判断先于谓词构造/加锁（空 filters 即可覆盖）。
    /// EN: Hybrid path shares the collection contract — zero and over-cap k both
    /// yield 400 before predicate construction or locking.
    #[tokio::test]
    async fn hybrid_search_rejects_zero_and_oversize_k() {
        let (state, _cleanup) = temp_state();
        state.collections.get_or_create("c", Some(4)).unwrap();
        for bad_k in [0usize, MAX_SEARCH_K + 1] {
            let err = handle_collection_hybrid_search(
                State(state.clone()),
                Path("c".to_string()),
                None,
                Json(hybrid_req(bad_k)),
            )
            .await
            .expect_err("k boundary must be rejected");
            assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST, "k={bad_k}");
            // 按越界情形断言特征子串：零值必须点明下界，超上界必须带出上界数值。
            // EN: Assert a distinctive substring per case: the floor for zero, the
            // numeric bound for over-cap — a bare contains('k') would catch neither drift.
            let expected = if bad_k == 0 {
                "must be >= 1"
            } else {
                &MAX_SEARCH_K.to_string()
            };
            assert!(
                err.1.contains(expected),
                "error must name the enforced limit, got: {}",
                err.1
            );
        }
    }
}

/// readyz 探针测试（P4.0.5）：与 score_edge_tests 分离——就绪判定与分数语义无关。
/// Readiness-probe tests, kept apart from score edge cases (unrelated concerns).
#[cfg(test)]
mod readyz_tests {
    use super::*;

    /// readyz 判定边界：pending ≤ 水位线就绪，超水位摘流（含等值边界）。
    /// EN: readyz decision boundary — ready iff pending ≤ watermark (inclusive).
    #[test]
    fn readyz_decision_boundary_is_watermark_inclusive() {
        let w = READYZ_PENDING_WATERMARK;
        assert!(readyz_is_ready(0, w));
        assert!(readyz_is_ready(w, w));
        assert!(!readyz_is_ready(w + 1, w));
    }
}

/// 只读副本测试（P4.4.1）：follower 写端点 503 `read_only`，检索与指标放行。
/// Read-only replica tests — writes get 503 on followers; reads/metrics pass.
#[cfg(test)]
mod read_only_tests {
    use super::*;
    use crate::types::{ReplicationMetricsView, ReplicationRole, RoleState};
    use axum::extract::Path;

    /// 本模块自用的最小状态装配（各 cfg(test) 模块的 temp_state 互不可见）。
    /// EN: Local minimal state builder (sibling test modules' helpers are private).
    fn follower_state(role: ReplicationRole) -> BenchmarkHttpState {
        let dims = 4;
        let wal_path = std::env::temp_dir().join(format!(
            "houyidb-readonly-test-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let wal = Arc::new(WalHandle::open(&wal_path, WalWriteMode::Async).unwrap().0);
        let collections_wal_path =
            houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
        let collections_wal = Arc::new(
            WalHandle::open(&collections_wal_path, WalWriteMode::Async)
                .unwrap()
                .0,
        );
        BenchmarkHttpState {
            promote: None,
            audit: None,
            encryption: None,
            index: IndexBackend::new_hnsw(dims),
            collections: Arc::new(CollectionManager::new(dims, "hnsw")),
            wal,
            collections_wal,
            vector_dim: dims,
            pgwire_metrics: Arc::new(PgWireMetrics::default()),
            applied_sidecar: wal_path.with_extension("applied"),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path,
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(
                crate::types::DEFAULT_MAX_HTTP_INFLIGHT as u64,
            )),
            graph_snapshot_metrics: Arc::new(crate::recovery::GraphSnapshotMetrics::default()),
            role: RoleState::new(role),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: crate::types::shared_replication_metrics(match role {
                ReplicationRole::Follower => Some(ReplicationMetricsView::Follower {
                    metrics: std::sync::Arc::new(
                        houyi_persistence::replication_follower::FollowerMetrics::default(),
                    ),
                    poison_skipped: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                }),
                _ => None,
            }),
            lifecycle: std::sync::Arc::new(
                crate::collection_lifecycle::CollectionLifecycleState::for_test(),
            ),
        }
    }

    fn upsert_req() -> UpsertRequest {
        UpsertRequest {
            vectors: vec![UpsertVector {
                id: 1,
                values: vec![0.0; 4],
                label: None,
                payload: None,
            }],
            index: None,
        }
    }

    /// follower 上 collection upsert 必须 503 且报文点名 read_only。
    /// EN: Collection upsert on a follower → 503 naming read_only.
    #[tokio::test]
    async fn follower_rejects_collection_upsert() {
        let state = follower_state(ReplicationRole::Follower);
        let err = handle_collection_upsert(State(state), Path("c".to_string()), Json(upsert_req()))
            .await
            .expect_err("follower must reject writes");
        assert_eq!(err.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.1.contains("read_only"), "got: {}", err.1);
    }

    /// follower 上 benchmark upsert 同纪律 503（主索引路径不豁免）。
    /// EN: The benchmark upsert path is guarded too.
    #[tokio::test]
    async fn follower_rejects_benchmark_upsert() {
        let state = follower_state(ReplicationRole::Follower);
        let err = benchmark_upsert(State(state), Json(upsert_req()))
            .await
            .expect_err("follower must reject writes");
        assert_eq!(err.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.1.contains("read_only"), "got: {}", err.1);
    }

    /// 检索/metrics 在 follower 上放行——只读副本可查是卖点。
    /// EN: Reads and /metrics stay available on followers.
    #[tokio::test]
    async fn follower_serves_search_and_metrics() {
        let state = follower_state(ReplicationRole::Follower);
        state.collections.get_or_create("c", Some(4)).unwrap();
        let req = SearchRequest {
            vector: vec![0.0; 4],
            k: 1,
            filter_gt_id: None,
            ef_search: None,
            filter_label: None,
            include_payload: false,
        };
        let _ =
            handle_collection_search(State(state.clone()), Path("c".to_string()), None, Json(req))
                .await
                .expect("search must pass on followers");
        let (_headers, body) = benchmark_metrics(State(state)).await;
        assert!(
            body.contains("houyidb_replication_role 2"),
            "metrics must render the follower role gauge, got: {body}"
        );
        assert!(
            body.contains("houyidb_replication_recv_lsn"),
            "follower metrics section must be present"
        );
        assert!(
            body.contains("houyidb_replication_apply_failures_total 0"),
            "follower metrics must expose apply_failures_total, got: {body}"
        );
        assert!(
            body.contains("houyidb_replication_apply_poison_skipped_total 0"),
            "follower metrics must expose poison_skipped, got: {body}"
        );
    }

    /// role=disabled 时 /metrics 省略复制节（缺失比恒 0 更能表达未启用）。
    /// EN: The replication section is omitted entirely when disabled.
    #[tokio::test]
    async fn disabled_role_omits_replication_metrics() {
        let state = follower_state(ReplicationRole::Disabled);
        let (_headers, body) = benchmark_metrics(State(state)).await;
        assert!(
            !body.contains("houyidb_replication_role"),
            "disabled role must omit the replication section"
        );
    }

    /// leader 视图渲染扇出与水印族。
    /// EN: The leader view renders fan-out and watermark series.
    #[tokio::test]
    async fn leader_metrics_render_hub_and_watermarks() {
        let state = follower_state(ReplicationRole::Leader);
        let hub = std::sync::Arc::new(houyi_persistence::replication_leader::ShipHub::new(8));
        let watermarks =
            std::sync::Arc::new(houyi_persistence::replication_leader::AckWatermarks::new());
        *state.replication_metrics.write().unwrap() = Some(ReplicationMetricsView::Leader {
            hub,
            watermarks,
            leader_next_lsn: std::sync::Arc::new(|| 42),
            sync_timeout: std::time::Duration::from_millis(2000),
            sync_quorum: 0,
        });
        let (_headers, body) = benchmark_metrics(State(state)).await;
        assert!(body.contains("houyidb_replication_role 1"), "got: {body}");
        assert!(body.contains("houyidb_replication_followers_active 0"));
        assert!(body.contains("houyidb_replication_shipped_batches_total 0"));
        // lag = next(42) - (durable(0) + 1) = 41（从未收过 ACK 的差 1 已文档化）。
        // EN: lag = next(42) - (durable(0) + 1) = 41 (the off-by-one for a
        // never-ACKed follower is documented).
        assert!(body.contains("houyidb_replication_lag_entries 41"));
        // P4.4.2 sync 等待四系列必须在 leader 节渲染。
        // EN: The four P4.4.2 sync-wait series must render in the leader section.
        assert!(
            body.contains("houyidb_replication_sync_waits_total 0"),
            "got: {body}"
        );
        assert!(body.contains("houyidb_replication_sync_timeouts_total 0"));
        assert!(body.contains("houyidb_replication_sync_no_follower_total 0"));
        assert!(body.contains("houyidb_replication_sync_wait_nanos_total 0"));
    }

    // ── Sentinel 控制面（EE M1）──────────────────────────────
    // EN: Sentinel control plane (EE M1).

    /// 每个测试独占一个目录（cluster_state.json 写在 WAL 父目录；共享
    /// temp_dir 会让并行测试在同一个 `cluster_state.json.tmp` 上竞争 rename）。
    /// EN: each test gets its own directory — cluster_state.json lives beside
    /// the WAL, and a shared temp_dir races on the same `.json.tmp` rename.
    struct LeaseTestDir(std::path::PathBuf);

    impl Drop for LeaseTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn lease_control(require_lease: bool) -> (Arc<crate::cluster::ClusterControl>, LeaseTestDir) {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "houyi-lease-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("lease test dir");
        let wal = dir.join("test.wal");
        let control = Arc::new(
            crate::cluster::ClusterControl::open(&wal, require_lease).expect("lease control"),
        );
        (control, LeaseTestDir(dir))
    }

    /// 开启租约后：无租约拒写、授予后放行、过期再次拒写、撤销回 Missing。
    /// EN: fencing: missing → 503, granted → pass, expired → 503, revoked → missing.
    #[tokio::test]
    async fn lease_fencing_blocks_and_unblocks_writes() {
        let mut state = follower_state(ReplicationRole::Leader);
        let (cluster, _lease_dir) = lease_control(true);
        state.cluster = cluster;
        let err = reject_if_read_only(&state).expect_err("missing lease must fence");
        assert_eq!(err.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.1.contains("lease_missing"), "got: {}", err.1);

        let grant = handle_admin_lease_grant(
            State(state.clone()),
            Json(LeaseRequest {
                term: 1,
                ttl_ms: 5_000,
            }),
        )
        .await
        .expect("grant");
        assert_eq!(grant.term, 1);
        reject_if_read_only(&state).expect("active lease allows writes");

        // 过期：同一 term 续租到极短 TTL，sleep 后必须拒写。
        // EN: expire by renewing with a tiny TTL, then sleep.
        let _ = handle_admin_lease_grant(
            State(state.clone()),
            Json(LeaseRequest { term: 1, ttl_ms: 1 }),
        )
        .await
        .expect("short renewal");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let err = reject_if_read_only(&state).expect_err("expired lease must fence");
        assert!(err.1.contains("lease_expired"), "got: {}", err.1);

        let _ = handle_admin_lease_revoke(State(state.clone())).await;
        let err = reject_if_read_only(&state).expect_err("revoked lease must fence");
        assert!(err.1.contains("lease_missing"), "got: {}", err.1);
    }

    /// term 只增：回退授权 409；状态端点回显租约与上限。
    /// EN: term monotonicity: a regressing grant is 409; status echoes lease/limits.
    #[tokio::test]
    async fn lease_grant_rejects_regression_and_status_is_readable() {
        let mut state = follower_state(ReplicationRole::Leader);
        let (cluster, _lease_dir) = lease_control(true);
        state.cluster = cluster;
        let _ = handle_admin_lease_grant(
            State(state.clone()),
            Json(LeaseRequest {
                term: 3,
                ttl_ms: 1_000,
            }),
        )
        .await
        .expect("grant term 3");
        let err = handle_admin_lease_grant(
            State(state.clone()),
            Json(LeaseRequest {
                term: 2,
                ttl_ms: 1_000,
            }),
        )
        .await
        .expect_err("term regression must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::CONFLICT);
        assert!(err.1.contains("term regression"), "got: {}", err.1);

        let Json(status) = handle_admin_lease_status(State(state.clone())).await;
        assert_eq!(status["term"], 3);
        assert_eq!(status["require_leader_lease"], true);
        assert_eq!(status["lease"]["active"], true);
    }

    /// demote：角色翻只读、租约撤销、term 落盘；重复调用幂等。
    /// EN: demote flips the role, revokes the lease, persists the term; idempotent.
    #[tokio::test]
    async fn demote_flips_role_and_is_idempotent() {
        let mut state = follower_state(ReplicationRole::Leader);
        let (cluster, _lease_dir) = lease_control(true);
        state.cluster = cluster;
        let collections_wal_path =
            houyi_persistence::collection_wal::collections_wal_path_for(&state.wal_path);
        state.promote = Some(Arc::new(PromoteContext {
            handles: Arc::new(Mutex::new(crate::types::ReplicationHandles::default())),
            promote_gate: tokio::sync::Mutex::new(()),
            hub: Arc::new(houyi_persistence::replication_leader::ShipHub::new(8)),
            listen_addr: "127.0.0.1:0".to_string(),
            collections_wal_path,
            collections_wal: Arc::clone(&state.collections_wal),
            queue_batches: 8,
            sync_timeout: std::time::Duration::from_millis(2000),
            sync_quorum: 0,
            lifecycle: Arc::clone(&state.lifecycle),
            follower_grace_secs: 0,
            replication_auth_token: None,
            encryption: None,
            collections: Arc::clone(&state.collections),
            follower_id: [1u8; 16],
            snapshot_checkpoint: Arc::new(|| 0),
        }));
        let _ = handle_admin_lease_grant(
            State(state.clone()),
            Json(LeaseRequest {
                term: 4,
                ttl_ms: 5_000,
            }),
        )
        .await
        .expect("grant");

        let demoted = handle_admin_demote(
            State(state.clone()),
            Some(Json(DemoteRequest {
                term: Some(4),
                follower_of: None,
            })),
        )
        .await
        .expect("demote");
        assert!(demoted.demoted);
        assert!(!demoted.already);
        assert_eq!(demoted.term, 4);
        assert_eq!(state.role.get(), ReplicationRole::Follower);
        assert!(reject_if_read_only(&state)
            .expect_err("demoted node must be read-only")
            .1
            .contains("read_only"));
        assert!(state.replication_metrics.read().unwrap().is_none());

        // 幂等：再次 demote 返回 already，且陈旧 term（< 4）被 409 拒绝。
        // EN: Idempotent: a second demote reports already, and a stale term
        // (< 4) is rejected with 409.
        let again = handle_admin_demote(State(state.clone()), None)
            .await
            .expect("idempotent demote");
        assert!(again.already);
        let stale = handle_admin_demote(
            State(state.clone()),
            Some(Json(DemoteRequest {
                term: Some(1),
                follower_of: None,
            })),
        )
        .await
        .expect_err("stale term must be rejected");
        assert_eq!(stale.0, axum::http::StatusCode::CONFLICT);
    }

    /// 畸形 follower_of 必须在撤租约/停监听之前返回 400，节点保持 leader 可写。
    /// EN: a malformed follower_of returns 400 before revoking the lease or
    /// stopping the listener; the node stays a writable leader.
    #[tokio::test]
    async fn demote_rejects_malformed_follower_of_before_teardown() {
        let mut state = follower_state(ReplicationRole::Leader);
        let (cluster, _lease_dir) = lease_control(true);
        state.cluster = cluster;
        state.promote = Some(Arc::new(PromoteContext {
            handles: Arc::new(Mutex::new(crate::types::ReplicationHandles::default())),
            promote_gate: tokio::sync::Mutex::new(()),
            hub: Arc::new(houyi_persistence::replication_leader::ShipHub::new(8)),
            listen_addr: "127.0.0.1:0".to_string(),
            collections_wal_path: houyi_persistence::collection_wal::collections_wal_path_for(
                &state.wal_path,
            ),
            collections_wal: Arc::clone(&state.collections_wal),
            queue_batches: 8,
            sync_timeout: std::time::Duration::from_millis(2000),
            sync_quorum: 0,
            lifecycle: Arc::clone(&state.lifecycle),
            follower_grace_secs: 0,
            replication_auth_token: None,
            encryption: None,
            collections: Arc::clone(&state.collections),
            follower_id: [4u8; 16],
            snapshot_checkpoint: Arc::new(|| 0),
        }));
        let _ = handle_admin_lease_grant(
            State(state.clone()),
            Json(LeaseRequest {
                term: 2,
                ttl_ms: 10_000,
            }),
        )
        .await
        .expect("grant");
        for bad in [
            "http://leader:9876",
            "leader",
            "leader:70000",
            " leader:9876/x",
            "fe80::1",
            "a:b:9876",
            "host :9876",
            "[leader:9876",
            "[::1]:9876:9",
            "[::1]",
            "user:pass@leader:9876",
            "leader:0",
            "[foo]:9876",
            "[[::1]:9876",
            "[::1::2]:9876",
            "[fe80::1%eth0]:9876",
            "host\u{0}:9876",
            "hosté:9876",
            "host:",
        ] {
            let err = handle_admin_demote(
                State(state.clone()),
                Some(Json(DemoteRequest {
                    term: Some(2),
                    follower_of: Some(bad.to_string()),
                })),
            )
            .await
            .expect_err("malformed follower_of must be rejected");
            assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST, "{bad}");
        }
        assert_eq!(state.role.get(), ReplicationRole::Leader);
        assert!(state.cluster.status().active);
        assert!(reject_if_read_only(&state).is_ok());
    }

    /// demote 带 follower_of：运行时转成 follower，无需重启。
    /// EN: demote with follower_of performs runtime conversion to follower.
    #[tokio::test]
    async fn demote_with_follower_of_starts_follower() {
        let mut state = follower_state(ReplicationRole::Leader);
        let (cluster, _lease_dir) = lease_control(true);
        state.cluster = cluster;
        let collections_wal_path =
            houyi_persistence::collection_wal::collections_wal_path_for(&state.wal_path);
        state.promote = Some(Arc::new(PromoteContext {
            handles: Arc::new(Mutex::new(crate::types::ReplicationHandles::default())),
            promote_gate: tokio::sync::Mutex::new(()),
            hub: Arc::new(houyi_persistence::replication_leader::ShipHub::new(8)),
            listen_addr: "127.0.0.1:0".to_string(),
            collections_wal_path,
            collections_wal: Arc::clone(&state.collections_wal),
            queue_batches: 8,
            sync_timeout: std::time::Duration::from_millis(2000),
            sync_quorum: 0,
            lifecycle: Arc::clone(&state.lifecycle),
            follower_grace_secs: 0,
            replication_auth_token: None,
            encryption: None,
            collections: Arc::clone(&state.collections),
            follower_id: [3u8; 16],
            snapshot_checkpoint: Arc::new(|| 0),
        }));
        let demoted = handle_admin_demote(
            State(state.clone()),
            Some(Json(DemoteRequest {
                term: Some(2),
                follower_of: Some("127.0.0.1:1".to_string()),
            })),
        )
        .await
        .expect("demote with follower_of");
        assert!(demoted.demoted);
        assert_eq!(state.role.get(), ReplicationRole::Follower);
        assert!(matches!(
            state.replication_metrics.read().unwrap().as_ref(),
            Some(ReplicationMetricsView::Follower { .. })
        ));
        // 重试路径：模拟复制会话启动失败（角色已翻、会话缺失），再次 demote
        // 必须重建 follower 会话而不是直接返回 already。
        // EN: retry path — simulate a failed session start (role flipped, no
        // session) and verify a repeated demote rebuilds the follower session.
        {
            let ctx = state.promote.as_ref().expect("promote ctx");
            ctx.handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .follower = None;
            *state
                .replication_metrics
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        }
        let retried = handle_admin_demote(
            State(state.clone()),
            Some(Json(DemoteRequest {
                term: Some(2),
                follower_of: Some("127.0.0.1:1".to_string()),
            })),
        )
        .await
        .expect("retried demote");
        assert!(retried.already);
        assert!(matches!(
            state.replication_metrics.read().unwrap().as_ref(),
            Some(ReplicationMetricsView::Follower { .. })
        ));
        // 会话已存在时再带一个不同的 follower_of：必须忽略、不拆掉现有会话。
        // EN: replay with a different follower_of must keep the existing session.
        let replayed = handle_admin_demote(
            State(state.clone()),
            Some(Json(DemoteRequest {
                term: Some(2),
                follower_of: Some("127.0.0.1:2".to_string()),
            })),
        )
        .await
        .expect("replayed demote");
        assert!(replayed.already);
        {
            let ctx = state.promote.as_ref().expect("promote ctx");
            assert!(ctx
                .handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .follower
                .is_some());
        }
        assert!(matches!(
            state.replication_metrics.read().unwrap().as_ref(),
            Some(ReplicationMetricsView::Follower { .. })
        ));
    }

    /// 拆除 follower 之前拒绝陈旧 term：即使 promote 上下文缺失也应先返回
    /// 409（证明预校验发生在任何拆除动作之前）。
    /// EN: a stale term is rejected before teardown — even with no promote
    /// context, the handler must return 409 rather than 500.
    #[tokio::test]
    async fn promote_rejects_stale_term_before_teardown() {
        let mut state = follower_state(ReplicationRole::Follower);
        let (cluster, _lease_dir) = lease_control(true);
        cluster.grant(5, 1_000).expect("seed term 5");
        cluster.revoke();
        state.cluster = cluster;
        state.promote = None;
        let err = handle_admin_promote(
            State(state),
            Some(Json(PromoteRequest {
                term: Some(4),
                ttl_ms: Some(1_000),
            })),
        )
        .await
        .expect_err("stale term must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::CONFLICT);
    }

    /// term=0 的 grant 是永久性客户端错误，必须 400 而不是 409。
    /// EN: term=0 is a permanent client error and must map to 400.
    #[tokio::test]
    async fn lease_term_zero_is_a_bad_request() {
        let mut state = follower_state(ReplicationRole::Leader);
        let (cluster, _lease_dir) = lease_control(true);
        state.cluster = cluster;
        let err = handle_admin_lease_grant(
            State(state),
            Json(LeaseRequest {
                term: 0,
                ttl_ms: 1_000,
            }),
        )
        .await
        .expect_err("term 0 must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
    }

    /// 失败检索也要记录过滤标志与真实状态码。
    /// EN: failed searches record the real status and filter flag.
    #[tokio::test]
    async fn query_log_records_failed_search_metadata() {
        let mut state = follower_state(ReplicationRole::Leader);
        state.query_log = Some(Arc::new(crate::query_log::QueryLog::new(8, 1)));
        let req = SearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k: MAX_SEARCH_K + 1,
            filter_gt_id: Some(1),
            ef_search: None,
            filter_label: None,
            include_payload: false,
        };
        let err = benchmark_search(State(state.clone()), None, Json(req))
            .await
            .expect_err("over-cap k must be rejected");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        let Json(body) = handle_admin_query_log(
            State(state),
            axum::extract::Query(crate::query_log::QueryLogQuery::default()),
        )
        .await;
        assert_eq!(body["entries"].as_array().unwrap().len(), 1);
        assert_eq!(body["entries"][0]["status"], 400);
        assert_eq!(body["entries"][0]["filtered"], true);
        assert_eq!(body["entries"][0]["route"], "benchmark_search");
    }

    /// 查询日志端点：关闭时 enabled=false；开启后统计按 collection 聚合。
    /// EN: query-log endpoints: disabled reports enabled=false; enabled aggregates.
    #[tokio::test]
    async fn query_log_endpoints_report_entries_and_stats() {
        let mut state = follower_state(ReplicationRole::Leader);
        let Json(disabled) = handle_admin_query_log(
            State(state.clone()),
            axum::extract::Query(crate::query_log::QueryLogQuery::default()),
        )
        .await;
        assert_eq!(disabled["enabled"], false);

        state.query_log = Some(Arc::new(crate::query_log::QueryLog::new(8, 1)));
        record_query_log(
            &state,
            None,
            "collection_search",
            Some("docs"),
            10,
            200,
            std::time::Instant::now(),
            3,
            false,
        );
        let Json(body) = handle_admin_query_log(
            State(state.clone()),
            axum::extract::Query(crate::query_log::QueryLogQuery::default()),
        )
        .await;
        assert_eq!(body["enabled"], true);
        assert_eq!(body["entries"].as_array().unwrap().len(), 1);
        assert_eq!(body["entries"][0]["collection"], "docs");
        let Json(stats) = handle_admin_query_stats(State(state)).await;
        assert_eq!(stats["stats"]["by_collection"][0]["collection"], "docs");
        assert_eq!(stats["stats"]["by_collection"][0]["count"], 1);
    }
}

/// `POST /v1/collections/{name}/ha_level`（P4.4.2）端点行为测试：
/// 成功回显规范字符串并改内存 / 未知 collection 404 / 笔误 400 且无副作用 /
/// follower 503。WAL 落盘 + 重放恢复的端到端链路由 src/tests.rs 的
/// `collections_restart_recovers_ha_level` 覆盖（与 upsert/delete 同模式）。
/// EN: Endpoint behavior tests for the ha_level route; the WAL-persistence +
/// replay-restore chain is covered end-to-end in src/tests.rs.
#[cfg(test)]
mod collection_ha_level_tests {
    use super::*;
    use crate::types::{ReplicationRole, RoleState};
    use axum::extract::Path;

    struct TempFileCleanup(Vec<std::path::PathBuf>);
    impl Drop for TempFileCleanup {
        fn drop(&mut self) {
            for path in &self.0 {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    fn temp_state(role: ReplicationRole) -> (BenchmarkHttpState, TempFileCleanup) {
        let dims = 4;
        let wal_path = std::env::temp_dir().join(format!(
            "houyidb-ha-level-test-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let wal = Arc::new(WalHandle::open(&wal_path, WalWriteMode::Async).unwrap().0);
        let collections_wal_path =
            houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
        let collections_wal = Arc::new(
            WalHandle::open(&collections_wal_path, WalWriteMode::Async)
                .unwrap()
                .0,
        );
        let state = BenchmarkHttpState {
            promote: None,
            audit: None,
            encryption: None,
            index: IndexBackend::new_hnsw(dims),
            collections: Arc::new(CollectionManager::new(dims, "hnsw")),
            wal,
            collections_wal,
            vector_dim: dims,
            pgwire_metrics: Arc::new(PgWireMetrics::default()),
            applied_sidecar: wal_path.with_extension("applied"),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path: wal_path.clone(),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
            graph_snapshot_metrics: Arc::new(crate::recovery::GraphSnapshotMetrics::default()),
            role: RoleState::new(role),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: crate::types::shared_replication_metrics(None),
            lifecycle: std::sync::Arc::new(
                crate::collection_lifecycle::CollectionLifecycleState::for_test(),
            ),
        };
        (
            state,
            TempFileCleanup(vec![
                wal_path.with_extension("applied"),
                wal_path,
                collections_wal_path,
            ]),
        )
    }

    fn current_level(
        state: &BenchmarkHttpState,
        name: &str,
    ) -> houyi_persistence::ha_level::HaLevel {
        state.collections.get(name).unwrap().ha_level()
    }

    #[tokio::test]
    async fn set_ha_level_success_echoes_canonical_and_updates_memory() {
        let (state, _cleanup) = temp_state(ReplicationRole::Disabled);
        state.collections.get_or_create("docs", Some(4)).unwrap();
        assert_eq!(
            current_level(&state, "docs"),
            houyi_persistence::ha_level::HaLevel::Async,
            "default must be async"
        );
        // 大小写 + 连字符变体必须被归一（与 HaLevel::parse 约定一致）。
        // EN: Case and hyphen variants must be normalized (consistent with
        // HaLevel::parse).
        let resp = handle_collection_set_ha_level(
            State(state.clone()),
            Path("docs".to_string()),
            Json(SetHaLevelRequest {
                ha_level: "Semi-Sync".to_string(),
            }),
        )
        .await
        .expect("valid level must succeed")
        .0;
        assert_eq!(resp.collection, "docs");
        assert_eq!(resp.ha_level, "semi_sync");
        assert_eq!(
            current_level(&state, "docs"),
            houyi_persistence::ha_level::HaLevel::SemiSync
        );
        // 同值重复设置幂等成功（重放同语义的写入侧投影）。
        // EN: Setting the same value again is idempotent (replays the same write-side
        // projection).
        let resp = handle_collection_set_ha_level(
            State(state.clone()),
            Path("docs".to_string()),
            Json(SetHaLevelRequest {
                ha_level: "semi_sync".to_string(),
            }),
        )
        .await
        .expect("idempotent re-set must succeed")
        .0;
        assert_eq!(resp.ha_level, "semi_sync");
    }

    #[tokio::test]
    async fn set_ha_level_unknown_collection_is_404() {
        let (state, _cleanup) = temp_state(ReplicationRole::Disabled);
        let err = handle_collection_set_ha_level(
            State(state),
            Path("ghost".to_string()),
            Json(SetHaLevelRequest {
                ha_level: "full_sync".to_string(),
            }),
        )
        .await
        .expect_err("unknown collection must 404");
        assert_eq!(err.0, axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn set_ha_level_typo_is_400_without_side_effects() {
        let (state, _cleanup) = temp_state(ReplicationRole::Disabled);
        state.collections.get_or_create("docs", Some(4)).unwrap();
        let err = handle_collection_set_ha_level(
            State(state.clone()),
            Path("docs".to_string()),
            Json(SetHaLevelRequest {
                ha_level: "sycn".to_string(),
            }),
        )
        .await
        .expect_err("typo must 400");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            current_level(&state, "docs"),
            houyi_persistence::ha_level::HaLevel::Async,
            "rejected request must not mutate the level"
        );
    }

    #[tokio::test]
    async fn follower_rejects_set_ha_level() {
        let (state, _cleanup) = temp_state(ReplicationRole::Follower);
        state.collections.get_or_create("docs", Some(4)).unwrap();
        let err = handle_collection_set_ha_level(
            State(state),
            Path("docs".to_string()),
            Json(SetHaLevelRequest {
                ha_level: "full_sync".to_string(),
            }),
        )
        .await
        .expect_err("follower must reject writes");
        assert_eq!(err.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.1.contains("read_only"), "got: {}", err.1);
    }

    // ── P4.4.2 数据面：写端点的 sync 等待 ─────────────────────
    // EN: ── P4.4.2 data plane: sync wait on write endpoints ────────────────

    use houyi_persistence::replication_leader::{AckWatermarks, ShipHub};

    /// 给 state 装上 leader 视图（hub + watermarks + 短超时），返回 watermarks
    /// 供断言。`connect_fake_follower=true` 时注册一个永不 ACK 的假 follower
    /// session（follower_count>0，但水位线永不推进）。
    /// EN: Install a leader view (hub + watermarks + short timeout) into state and
    /// return the watermarks for assertions. With `connect_fake_follower=true`,
    /// register a fake follower session that never ACKs (follower_count>0, but the
    /// watermark never advances).
    fn install_leader_view(
        state: &BenchmarkHttpState,
        sync_timeout: std::time::Duration,
        connect_fake_follower: bool,
    ) -> Arc<AckWatermarks> {
        let hub = Arc::new(ShipHub::new(8));
        if connect_fake_follower {
            // 接收端持有但不取帧、不 ACK——模拟在线但无确认推进的 follower。
            // EN: The receiver holds frames without reading or ACKing — a follower
            // that is online but makes no progress.
            std::mem::forget(hub.register());
        }
        let watermarks = Arc::new(AckWatermarks::new());
        *state.replication_metrics.write().unwrap() = Some(ReplicationMetricsView::Leader {
            hub,
            watermarks: Arc::clone(&watermarks),
            leader_next_lsn: Arc::new(|| 0),
            sync_timeout,
            sync_quorum: 0,
        });
        watermarks
    }

    async fn upsert_one(state: &BenchmarkHttpState, name: &str) {
        let _ = handle_collection_upsert(
            State(state.clone()),
            Path(name.to_string()),
            Json(UpsertRequest {
                vectors: vec![UpsertVector {
                    id: 1,
                    values: vec![1.0, 0.0, 0.0, 0.0],
                    label: None,
                    payload: None,
                }],
                index: None,
            }),
        )
        .await
        .expect("upsert must succeed");
    }

    /// 无健康 follower：full_sync 写照常成功、立即返回（不阻塞到超时），
    /// 计 no_follower 降级——裁决语义「无 follower 不等待」的 handler 级护栏。
    /// EN: No healthy follower: full_sync writes still succeed and return
    /// immediately (no blocking until timeout), counting no_follower degradation —
    /// a handler-level guard for the ruling "no follower means no wait".
    #[tokio::test]
    async fn sync_write_without_follower_degrades_immediately() {
        let (state, _cleanup) = temp_state(ReplicationRole::Disabled);
        let watermarks = install_leader_view(&state, std::time::Duration::from_secs(30), false);
        state.collections.get_or_create("docs", Some(4)).unwrap();
        state
            .collections
            .get("docs")
            .unwrap()
            .set_ha_level(houyi_persistence::ha_level::HaLevel::FullSync);
        let start = std::time::Instant::now();
        upsert_one(&state, "docs").await;
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "no-follower must not block"
        );
        assert_eq!(watermarks.sync_waits_total(), 1);
        assert_eq!(watermarks.sync_no_follower_total(), 1);
        assert_eq!(watermarks.sync_timeouts_total(), 0);
    }

    /// follower 在线但 ACK 不推进：写照常成功但需等满超时，计 timeout 降级。
    /// EN: Follower online but ACK does not advance: writes still succeed but must
    /// wait the full timeout, counting timeout degradation.
    #[tokio::test]
    async fn sync_write_with_stalled_follower_succeeds_after_timeout() {
        let (state, _cleanup) = temp_state(ReplicationRole::Disabled);
        let watermarks = install_leader_view(&state, std::time::Duration::from_millis(50), true);
        state.collections.get_or_create("docs", Some(4)).unwrap();
        state
            .collections
            .get("docs")
            .unwrap()
            .set_ha_level(houyi_persistence::ha_level::HaLevel::SemiSync);
        upsert_one(&state, "docs").await;
        assert_eq!(watermarks.sync_waits_total(), 1);
        assert_eq!(watermarks.sync_timeouts_total(), 1);
        assert_eq!(watermarks.sync_no_follower_total(), 0);
        assert!(
            watermarks.sync_wait_nanos_total() >= 40_000_000,
            "wait must have blocked ~50ms, got {} ns",
            watermarks.sync_wait_nanos_total()
        );
    }

    /// 水位线已满足：full_sync 写走零注册快路径（等待计数但不计降级）。
    /// EN: Watermark already satisfied: full_sync writes take the zero-registration
    /// fast path (wait counted, no degradation).
    #[tokio::test]
    async fn sync_write_fast_path_when_watermark_covers_lsn() {
        let (state, _cleanup) = temp_state(ReplicationRole::Disabled);
        let watermarks = install_leader_view(&state, std::time::Duration::from_millis(50), false);
        // 预推进 DURABLE 水位线：upsert 的 WAL LSN（0/1）必被覆盖。
        // EN: Advance the DURABLE watermark in advance: the upsert WAL LSN (0/1) is
        // necessarily covered.
        watermarks.on_ack(houyi_persistence::wal_replication::ACK_STAGE_DURABLE, 1000);
        state.collections.get_or_create("docs", Some(4)).unwrap();
        state
            .collections
            .get("docs")
            .unwrap()
            .set_ha_level(houyi_persistence::ha_level::HaLevel::FullSync);
        upsert_one(&state, "docs").await;
        assert_eq!(watermarks.sync_waits_total(), 1);
        assert_eq!(watermarks.sync_timeouts_total(), 0);
        assert_eq!(watermarks.sync_no_follower_total(), 0);
    }

    /// 默认 async 级别：即使 leader 视图在位也完全不进等待（热路径零开销）。
    /// EN: Default async level: even with a leader view installed there is no wait
    /// at all (zero hot-path overhead).
    #[tokio::test]
    async fn async_level_never_waits() {
        let (state, _cleanup) = temp_state(ReplicationRole::Disabled);
        let watermarks = install_leader_view(&state, std::time::Duration::from_millis(50), true);
        state.collections.get_or_create("docs", Some(4)).unwrap();
        upsert_one(&state, "docs").await;
        assert_eq!(watermarks.sync_waits_total(), 0, "async must not wait");
    }
}

/// P4.4.3 手动 failover：promote 端点测试——角色门（400/409/幂等）、
/// 翻转后写解锁与视图就位、以及最关键的「hub 复用端到端」：promote 前的
/// WAL 句柄（以 follower 身份打开）在 promote 后无需切换即可向新 follower
/// 扇出，且 LSN 连续（follower resume 零空洞）。
/// EN: Promote endpoint tests — role gates, write unlock + view swap, and the
/// hub-reuse end-to-end: the WAL handle opened as a follower ships to a new
/// follower after promote with no handle swap and no LSN gap.
#[cfg(test)]
mod promote_tests {
    use super::*;
    use crate::types::{ReplicationHandles, ReplicationRole, RoleState, SharedReplicationHandles};
    use axum::extract::Path;

    struct TempFileCleanup(Vec<std::path::PathBuf>);
    impl Drop for TempFileCleanup {
        fn drop(&mut self) {
            for path in &self.0 {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    /// follower 角色的完整 promote 装配：hub 注入 collections.wal 写线程
    /// （open_with_ship——复用语义的唯一真实来源）、共享句柄槽、随机端口监听。
    /// EN: Full promote wiring for the follower role: hub injected into the
    /// collections.wal writer (open_with_ship — the single source of truth for
    /// reuse semantics), shared handle slot, random-port listener.
    fn promote_state() -> (
        BenchmarkHttpState,
        SharedReplicationHandles,
        TempFileCleanup,
    ) {
        promote_state_with_token(None)
    }

    fn promote_state_with_token(
        token: Option<&str>,
    ) -> (
        BenchmarkHttpState,
        SharedReplicationHandles,
        TempFileCleanup,
    ) {
        let dims = 4;
        let wal_path = std::env::temp_dir().join(format!(
            "houyidb-promote-test-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let wal = Arc::new(WalHandle::open(&wal_path, WalWriteMode::Async).unwrap().0);
        let collections_wal_path =
            houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
        let hub = Arc::new(houyi_persistence::replication_leader::ShipHub::new(64));
        let collections_wal = Arc::new(
            WalHandle::open_with_ship(
                &collections_wal_path,
                WalWriteMode::Async,
                Some(Arc::clone(&hub)),
            )
            .unwrap()
            .0,
        );
        let handles: SharedReplicationHandles =
            Arc::new(std::sync::Mutex::new(ReplicationHandles::default()));
        let lifecycle =
            std::sync::Arc::new(crate::collection_lifecycle::CollectionLifecycleState::for_test());
        let promote_ctx = Arc::new(PromoteContext {
            handles: Arc::clone(&handles),
            promote_gate: tokio::sync::Mutex::new(()),
            hub,
            listen_addr: "127.0.0.1:0".to_string(),
            collections_wal_path: collections_wal_path.clone(),
            collections_wal: Arc::clone(&collections_wal),
            queue_batches: 64,
            sync_timeout: std::time::Duration::from_millis(2000),
            sync_quorum: 0,
            lifecycle: Arc::clone(&lifecycle),
            follower_grace_secs: 0,
            replication_auth_token: token.map(str::to_string),
            encryption: None,
            collections: Arc::new(CollectionManager::new(dims, "hnsw")),
            follower_id: [2u8; 16],
            snapshot_checkpoint: Arc::new(|| 0),
        });
        let state = BenchmarkHttpState {
            audit: None,
            encryption: None,
            promote: Some(promote_ctx),
            index: IndexBackend::new_hnsw(dims),
            collections: Arc::new(CollectionManager::new(dims, "hnsw")),
            wal,
            collections_wal,
            vector_dim: dims,
            pgwire_metrics: Arc::new(PgWireMetrics::default()),
            applied_sidecar: wal_path.with_extension("applied"),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path: wal_path.clone(),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
            graph_snapshot_metrics: Arc::new(crate::recovery::GraphSnapshotMetrics::default()),
            role: RoleState::new(ReplicationRole::Follower),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: crate::types::shared_replication_metrics(None),
            lifecycle,
        };
        (
            state,
            handles,
            TempFileCleanup(vec![
                wal_path.with_extension("applied"),
                wal_path,
                collections_wal_path,
            ]),
        )
    }

    /// 关停 promote 起的 leader（测试收尾纪律：监听 task 不得随测试泄漏）。
    /// EN: Shut down the leader started by promote (test teardown discipline:
    /// listener tasks must not leak with the test).
    async fn shutdown_leader(handles: &SharedReplicationHandles) {
        let leader = handles.lock().unwrap().leader.take();
        if let Some(l) = leader {
            l.shutdown().await;
        }
    }

    /// disabled 角色不是复制开关：400。
    /// EN: A disabled role is not a replication switch: 400.
    #[tokio::test]
    async fn promote_on_disabled_is_400() {
        let (state, handles, _c) = promote_state();
        state.role.set(ReplicationRole::Disabled);
        let err = handle_admin_promote(State(state), None)
            .await
            .expect_err("disabled must 400");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        assert!(handles.lock().unwrap().leader.is_none());
    }

    /// 已是 leader（起始角色）：幂等 409，不得重起监听。
    /// EN: Already leader (initial role): idempotent 409, the listener must not restart.
    #[tokio::test]
    async fn promote_on_leader_is_409() {
        let (state, handles, _c) = promote_state();
        state.role.set(ReplicationRole::Leader);
        let err = handle_admin_promote(State(state), None)
            .await
            .expect_err("leader must 409");
        assert_eq!(err.0, axum::http::StatusCode::CONFLICT);
        assert!(handles.lock().unwrap().leader.is_none());
    }

    /// 核心路径：promote 成功 → 角色翻转 + 视图就位 + 写解锁 + next_lsn 合理；
    /// 重复 promote 409（幂等）。
    /// EN: Core path: promote succeeds → role flips + view installed + writes
    /// unlock + next_lsn sane; repeating promote returns 409 (idempotent).
    #[tokio::test]
    async fn promote_flips_role_unlocks_writes_and_is_idempotent() {
        let (state, handles, _c) = promote_state();
        // promote 前带一个 collection（drain 循环必须有真实对象可遍历）。
        // EN: Create a collection before promote (the drain loop needs a real object to
        // iterate).
        state.collections.get_or_create("docs", Some(4)).unwrap();
        // promote 前写被只读拦截（前置条件自证）。
        // EN: Writes are blocked by read-only before promote (precondition self-check).
        let pre = handle_collection_upsert(
            State(state.clone()),
            Path("docs".to_string()),
            Json(UpsertRequest {
                vectors: vec![UpsertVector {
                    id: 7,
                    values: vec![1.0, 0.0, 0.0, 0.0],
                    label: None,
                    payload: None,
                }],
                index: None,
            }),
        )
        .await;
        assert_eq!(
            pre.expect_err("follower must reject writes").0,
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );

        let resp = handle_admin_promote(State(state.clone()), None)
            .await
            .expect("promote must succeed")
            .0;
        assert!(resp.promoted);
        assert_eq!(
            resp.next_lsn,
            state.collections_wal.next_lsn(),
            "next_lsn must match the collections WAL"
        );
        assert_eq!(state.role.get(), ReplicationRole::Leader);
        // 视图已换成 Leader 变体（sync 等待的水位线/hub 就位）。
        // EN: The view has been swapped to the Leader variant (watermarks/hub for sync
        // waits installed).
        assert!(
            matches!(
                &*state.replication_metrics.read().unwrap(),
                Some(ReplicationMetricsView::Leader { .. })
            ),
            "metrics view must be swapped to Leader"
        );
        // 写解锁：同一 upsert 现在成功。
        // EN: Writes unlocked: the same upsert now succeeds.
        let _ = handle_collection_upsert(
            State(state.clone()),
            Path("docs".to_string()),
            Json(UpsertRequest {
                vectors: vec![UpsertVector {
                    id: 7,
                    values: vec![1.0, 0.0, 0.0, 0.0],
                    label: None,
                    payload: None,
                }],
                index: None,
            }),
        )
        .await
        .expect("writes must unlock after promote");
        // 幂等：第二次 promote 409。
        // EN: Idempotency: the second promote returns 409.
        let err = handle_admin_promote(State(state.clone()), None)
            .await
            .expect_err("second promote must 409");
        assert_eq!(err.0, axum::http::StatusCode::CONFLICT);
        shutdown_leader(&handles).await;
    }

    /// hub 复用端到端：promote 后新 follower 从 resume_lsn=0 全量追上，
    /// promote 前后的写（同一 WAL 句柄、LSN 连续）全部到达且可查。
    /// EN: Hub reuse end to end: after promote a new follower catches up fully from
    /// resume_lsn=0, and writes before/after promote (same WAL handle, contiguous
    /// LSN) all arrive and are queryable.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn promoted_leader_ships_to_new_follower_without_gap() {
        let (state, handles, _c) = promote_state();
        // promote 前的「历史」：模拟 follower 时代经复制 apply 落下的一条记录
        // （直接 WAL append——follower 的 AppendEntry 与客户端写共享同一句柄）。
        // EN: "History" before promote: one record applied via replication during
        // the follower era (direct WAL append — the follower's AppendEntry shares
        // the same handle as client writes).
        let history =
            houyi_persistence::collection_wal::encode_collection_create("hist", 4, "hnsw").unwrap();
        let history_lsn = state.collections_wal.append(history).unwrap().lsn;

        let resp = handle_admin_promote(State(state.clone()), None)
            .await
            .expect("promote must succeed")
            .0;
        assert!(resp.next_lsn > history_lsn);
        let leader_addr = handles
            .lock()
            .unwrap()
            .leader
            .as_ref()
            .expect("leader handle must be stored")
            .local_addr()
            .expect("leader must be bound");

        // 新 follower（promote 后才上线）：resume_lsn=0，必须全量 catch-up。
        // EN: New follower (comes online only after promote): resume_lsn=0, must catch
        // up fully.
        let wal_b_path = std::env::temp_dir().join(format!(
            "houyidb-promote-follower-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _c2 = TempFileCleanup(vec![wal_b_path.clone()]);
        let wal_b = WalHandle::open(&wal_b_path, WalWriteMode::Async).unwrap().0;
        let wal_b_async = wal_b.async_handle().expect("async wal handle");
        let manager_b = Arc::new(CollectionManager::new(4, "hnsw"));
        let sink = Arc::new(crate::replication_apply::CollectionsApplySink::new(
            Arc::clone(&manager_b),
        ));
        let metrics_b =
            Arc::new(houyi_persistence::replication_follower::FollowerMetrics::default());
        let follower_b = houyi_persistence::replication_follower::ReplicationFollower::start(
            leader_addr.to_string(),
            wal_b_async,
            sink,
            Arc::clone(&metrics_b),
            houyi_persistence::replication_follower::FollowerConfig::default(),
        )
        .expect("follower must start");

        // promote 后的写：走 HTTP 端点（真实客户端路径），同一 WAL 句柄扇出。
        // EN: Post-promote write: goes through the HTTP endpoint (real client path),
        // fanning out on the same WAL handle.
        let _ = handle_collection_upsert(
            State(state.clone()),
            Path("hist".to_string()),
            Json(UpsertRequest {
                vectors: vec![UpsertVector {
                    id: 42,
                    values: vec![1.0, 0.0, 0.0, 0.0],
                    label: None,
                    payload: None,
                }],
                index: None,
            }),
        )
        .await
        .expect("write on promoted leader must succeed");

        // 等 follower B 追到 upsert 之后（10s IO 超时纪律，轮询消调度竞争）。
        // EN: Wait for follower B to catch up past the upsert (10s IO-timeout
        // discipline; polling tames scheduling races).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let caught_up = metrics_b
                .durable_lsn
                .load(std::sync::atomic::Ordering::Relaxed)
                >= state.collections_wal.next_lsn().saturating_sub(1)
                && manager_b.get("hist").is_some();
            if caught_up {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "follower did not catch up: durable_lsn={} next={}",
                metrics_b
                    .durable_lsn
                    .load(std::sync::atomic::Ordering::Relaxed),
                state.collections_wal.next_lsn()
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        // LSN 连续（保持 leader LSN 纪律）：B 的本地 next_lsn == A 的 next_lsn。
        // EN: LSN continuity (leader LSN discipline): B's local next_lsn == A's next_lsn.
        assert_eq!(
            metrics_b
                .recv_lsn
                .load(std::sync::atomic::Ordering::Relaxed),
            state.collections_wal.next_lsn(),
            "follower must preserve the leader LSN space with no gap"
        );
        follower_b.shutdown().await;
        shutdown_leader(&handles).await;
    }

    /// OCR 批次二十八 HIGH 回归：promote 重建 LeaderSessionConfig 时必须沿用
    /// 启动时的 auth_token，否则新 leader 静默变成无认证（任何 TCP 客户端
    /// 可拉全量日志）。错误 token 的 follower 必须收到 AUTH 拒绝。
    /// EN: promote must carry the auth token; a wrong-token follower gets AUTH.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn promote_preserves_replication_auth_token() {
        let (state, handles, _c) = promote_state_with_token(Some("s3cret"));
        let _ = handle_admin_promote(State(state), None)
            .await
            .expect("promote must succeed");
        let leader_addr = handles
            .lock()
            .unwrap()
            .leader
            .as_ref()
            .expect("leader handle must be stored")
            .local_addr()
            .expect("leader must be bound");

        let wal_b_path = std::env::temp_dir().join(format!(
            "houyidb-promote-auth-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _c2 = TempFileCleanup(vec![wal_b_path.clone()]);
        let wal_b = WalHandle::open(&wal_b_path, WalWriteMode::Async).unwrap().0;
        let wal_b_async = wal_b.async_handle().expect("async wal handle");
        let manager_b = Arc::new(CollectionManager::new(4, "hnsw"));
        let sink = Arc::new(crate::replication_apply::CollectionsApplySink::new(
            Arc::clone(&manager_b),
        ));
        let metrics_b =
            Arc::new(houyi_persistence::replication_follower::FollowerMetrics::default());
        let follower_b = houyi_persistence::replication_follower::ReplicationFollower::start(
            leader_addr.to_string(),
            wal_b_async,
            sink,
            Arc::clone(&metrics_b),
            houyi_persistence::replication_follower::FollowerConfig {
                auth_token: Some("wrong-token".to_string()),
                ahead_retry_interval: std::time::Duration::from_millis(100),
                handshake_timeout: std::time::Duration::from_secs(2),
                ..Default::default()
            },
        )
        .expect("follower must start");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while metrics_b
            .auth_rejections_total
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            assert!(
                std::time::Instant::now() < deadline,
                "promoted leader must reject a wrong-token follower (token was likely dropped)"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            metrics_b
                .durable_lsn
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "rejected follower must not apply anything"
        );
        follower_b.shutdown().await;
        shutdown_leader(&handles).await;
    }
}
