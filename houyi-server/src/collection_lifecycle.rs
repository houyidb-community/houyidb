//! Collection 快照周期与恢复（P5.0.2）。
//! Collection snapshot cycle and restore (P5.0.2).
//!
//! 提交顺序：逐 collection 写快照文件（tmp + fsync + rename）→ 原子写 manifest。
//! manifest 是唯一提交点；任何文件/manifest 失败都保留旧 manifest，不进入删除阶段。
//! EN: Commit order — per-collection snapshot files (tmp + fsync + rename) first,
//! then the atomic manifest. The manifest is the sole commit point; any failure
//! keeps the previous manifest and never enters deletion.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use houyi_core::ScalarValue;
use houyi_persistence::collection_snapshot_file::{
    collection_snapshots_dir, decode_collection_snapshot_payload,
    encode_collection_snapshot_payload, snapshot_file_name, CollectionManifest,
    CollectionManifestEntry, CollectionSnapshotPayload, SnapshotHnswParams, SnapshotScalar,
    SnapshotScalarRow,
};
use houyi_persistence::encryption::Encryption;
use houyi_persistence::graph_snapshot_file::{
    read_graph_snapshot_with_encryption, write_graph_snapshot_atomic_with_encryption,
};
use houyi_persistence::ha_level::HaLevel;
use houyi_vector::DistanceMetric;

use crate::collection::CollectionManager;
use crate::recovery::{drain_pending_builder, GraphSnapshotMetrics};
use crate::types::{ReplicationRole, RoleState, WalHandle};

/// 一次快照周期的结果。
/// EN: Result of one snapshot cycle.
#[derive(Debug, Clone, Default)]
pub struct CollectionSnapshotOutcome {
    /// Some = manifest 已落盘；None = 本轮未提交（blocked_reason 必有值）。
    /// EN: Some = manifest committed; None = not committed (blocked_reason set).
    pub manifest: Option<CollectionManifest>,
    pub blocked_reason: Option<String>,
    pub files_written: usize,
    pub bytes_written: u64,
}

/// 恢复结果：per-collection checkpoint + manifest。
/// EN: Restore result: per-collection checkpoints plus the manifest.
#[derive(Debug, Clone)]
pub struct LoadedCollectionSnapshots {
    pub manifest: CollectionManifest,
    pub checkpoints: HashMap<String, u64>,
}

/// 每次保留运行最多删除的整段数（计划默认 8）。
/// EN: Maximum whole segments deleted per retention run (plan default 8).
pub const DEFAULT_MAX_DELETES_PER_RUN: usize = 8;

/// P5.0.3：快照/保留的共享协调状态。
/// EN: P5.0.3 shared coordination state for snapshot and retention.
pub struct CollectionLifecycleState {
    /// 快照写出与管理员 compact 共用；周期任务与手动触发不得并发编码同一图。
    /// EN: Shared by the periodic writer and the admin compact endpoint so they
    /// never encode the same graph concurrently.
    pub snapshot_lock: tokio::sync::Mutex<()>,
    /// 握手/保留互斥门（P5.0.4）：握手读锁完成校验+注册，保留写锁删除段。
    /// EN: handshake/retention gate (P5.0.4): handshakes hold the read lock
    /// through validation+registration; retention deletes under the write lock.
    retention_gate: std::sync::Arc<tokio::sync::RwLock<()>>,
    /// 归档段保留上限；0 = 关闭删除。
    /// EN: Archived-segment cap; 0 disables deletion.
    pub max_segments: usize,
    pub max_deletes_per_run: usize,
    /// 保留租约截止时间（Unix 秒，仅用于状态展示）。
    /// EN: Retention lease deadline (Unix seconds; display only).
    lease_until_unix: AtomicU64,
    /// 单调时钟截止（毫秒，自 `started` 起）；0 = 无租约。active 判定只看它。
    /// EN: Monotonic deadline (ms since `started`); 0 = no lease. The active
    /// check uses only this value.
    lease_deadline_ms: AtomicU64,
    started: std::time::Instant,
    /// 最近一次成功提交的 manifest safe checkpoint（0 = 尚无 manifest）。
    /// follower 握手只读这个原子量，不在 runtime worker 上做磁盘 IO。
    /// EN: safe checkpoint of the last committed manifest (0 = none). Follower
    /// handshakes read this atomic and never touch disk on a runtime worker.
    manifest_checkpoint_lsn: AtomicU64,
    pub metrics: RetentionMetrics,
}

impl CollectionLifecycleState {
    pub fn new(max_segments: usize, max_deletes_per_run: usize) -> Self {
        Self {
            snapshot_lock: tokio::sync::Mutex::new(()),
            retention_gate: std::sync::Arc::new(tokio::sync::RwLock::new(())),
            max_segments,
            max_deletes_per_run,
            lease_until_unix: AtomicU64::new(0),
            lease_deadline_ms: AtomicU64::new(0),
            started: std::time::Instant::now(),
            manifest_checkpoint_lsn: AtomicU64::new(0),
            metrics: RetentionMetrics::default(),
        }
    }

    /// 测试默认：关闭删除，避免单元测试误删夹具。
    /// EN: Test default: deletion disabled so unit tests never delete fixtures.
    pub fn for_test() -> Self {
        Self::new(0, DEFAULT_MAX_DELETES_PER_RUN)
    }

    /// 握手/保留互斥门的共享句柄（P5.0.4）。
    /// EN: shared handshake/retention gate handle (P5.0.4).
    pub fn retention_gate(&self) -> std::sync::Arc<tokio::sync::RwLock<()>> {
        std::sync::Arc::clone(&self.retention_gate)
    }

    /// 记录最近一次成功提交的 manifest safe checkpoint（0 = 无 manifest）。
    /// EN: Records the safe checkpoint of the last committed manifest (0 = none).
    pub fn set_manifest_checkpoint(&self, lsn: u64) {
        self.manifest_checkpoint_lsn.store(lsn, Ordering::Release);
    }

    pub fn manifest_checkpoint(&self) -> u64 {
        self.manifest_checkpoint_lsn.load(Ordering::Acquire)
    }

    /// 设置保留租约（人工重同步/备份窗口）。0 秒表示清除租约。
    /// EN: Set a retention lease (manual resync/backup window). Zero clears it.
    pub fn hold_retention(&self, secs: u64) {
        let until = if secs == 0 {
            0
        } else {
            unix_now().saturating_add(secs)
        };
        let deadline_ms = if secs == 0 {
            0
        } else {
            self.now_ms().saturating_add(secs.saturating_mul(1000))
        };
        self.lease_until_unix.store(until, Ordering::Release);
        self.lease_deadline_ms.store(deadline_ms, Ordering::Release);
    }

    pub fn lease_until_unix(&self) -> u64 {
        self.lease_until_unix.load(Ordering::Acquire)
    }

    pub fn lease_active(&self) -> bool {
        let deadline = self.lease_deadline_ms.load(Ordering::Acquire);
        deadline > 0 && deadline > self.now_ms()
    }

    /// 剩余租约秒数（向上取整）；无租约/已过期返回 0。
    /// EN: Remaining lease seconds (rounded up); 0 when inactive/expired.
    pub fn lease_remaining_secs(&self) -> u64 {
        let deadline = self.lease_deadline_ms.load(Ordering::Acquire);
        let now = self.now_ms();
        if deadline <= now {
            return 0;
        }
        (deadline - now).div_ceil(1000)
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis().min(u64::MAX as u128) as u64
    }
}

/// 保留侧计数器；snapshot 写入/恢复继续复用 GraphSnapshotMetrics。
/// EN: Retention counters; snapshot write/load counters stay in GraphSnapshotMetrics.
#[derive(Default)]
pub struct RetentionMetrics {
    pub runs_total: AtomicU64,
    pub blocked_total: AtomicU64,
    pub segments_deleted_total: AtomicU64,
    pub bytes_freed_total: AtomicU64,
    pub last_safe_floor_lsn: AtomicU64,
    pub last_deleted_segments: AtomicU64,
}

/// 一次保留运行/状态查询的结果。
/// EN: Result of one retention run or status query.
#[derive(Debug, Clone, Default)]
pub struct RetentionReport {
    pub manifest_present: bool,
    pub manifest_seq: Option<u64>,
    pub safe_floor_lsn: Option<u64>,
    pub first_available_lsn: u64,
    pub archived_segments: usize,
    pub archived_bytes: u64,
    pub deleted_segments: usize,
    pub bytes_freed: u64,
    pub blocked_reason: Option<String>,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 只读查询保留状态：绝不删除段。
/// EN: Read-only retention status; never deletes segments.
pub fn retention_status(
    wal: &WalHandle,
    collections_wal_path: &Path,
    role: &RoleState,
    state: &CollectionLifecycleState,
) -> RetentionReport {
    retention_status_with(wal, collections_wal_path, role, state, None, false)
}

fn retention_status_with(
    wal: &WalHandle,
    collections_wal_path: &Path,
    role: &RoleState,
    state: &CollectionLifecycleState,
    follower_floor: Option<u64>,
    ignore_lease: bool,
) -> RetentionReport {
    let stats = match wal.segment_stats() {
        Ok(stats) => stats,
        Err(e) => {
            // 不吞错：stats 失败必须显式 blocked，而不是伪装成空 WAL。
            // EN: Never swallow: a stats failure must surface as explicitly
            // blocked rather than masquerading as an empty WAL.
            return RetentionReport {
                blocked_reason: Some(format!("segment_stats_failed:{e}")),
                ..Default::default()
            };
        }
    };
    retention_report_from_stats(
        collections_wal_path,
        role,
        state,
        &stats,
        follower_floor,
        ignore_lease,
    )
}

/// 用已取得的段统计构造保留报告（状态/metrics 复用，避免重复阻塞查询）。
/// EN: Build a retention report from already-fetched segment stats (reused by
/// status/metrics to avoid duplicate blocking queries).
pub fn retention_report_from_stats(
    collections_wal_path: &Path,
    role: &RoleState,
    state: &CollectionLifecycleState,
    stats: &houyi_persistence::SegmentStats,
    follower_floor: Option<u64>,
    ignore_lease: bool,
) -> RetentionReport {
    let manifest = houyi_persistence::collection_snapshot_file::read_collection_manifest(
        &collection_snapshots_dir(collections_wal_path).join("collections.manifest"),
    )
    .ok()
    .flatten();
    let mut report = RetentionReport {
        manifest_present: manifest.is_some(),
        manifest_seq: manifest.as_ref().map(|m| m.manifest_seq),
        safe_floor_lsn: manifest.as_ref().map(|m| m.safe_checkpoint_lsn),
        first_available_lsn: stats.first_available_lsn,
        archived_segments: stats.archives.len(),
        archived_bytes: stats.archives.iter().map(|m| m.bytes).sum(),
        ..Default::default()
    };
    if role.get() == ReplicationRole::Leader {
        if let (Some(manifest_floor), Some(follower_floor)) =
            (report.safe_floor_lsn, follower_floor)
        {
            report.safe_floor_lsn = Some(manifest_floor.min(follower_floor));
        }
    }
    report.blocked_reason =
        retention_blocked_reason(role, state, &report, stats, follower_floor, ignore_lease);
    report
}

fn retention_blocked_reason(
    role: &RoleState,
    state: &CollectionLifecycleState,
    report: &RetentionReport,
    stats: &houyi_persistence::SegmentStats,
    follower_floor: Option<u64>,
    ignore_lease: bool,
) -> Option<String> {
    if state.max_segments == 0 {
        return Some("retention_disabled".to_string());
    }
    if !ignore_lease && state.lease_active() {
        return Some(format!("lease_active_until:{}", state.lease_until_unix()));
    }
    // leader 必须跟随活跃 follower 的 durable 进度；没有 floor 时不得删除。
    // EN: A leader must follow active followers' durable progress; without a
    // floor it must not delete.
    if role.get() == ReplicationRole::Leader && follower_floor.is_none() {
        return Some("leader_retention_pending_no_follower_floor".to_string());
    }
    if !report.manifest_present {
        return Some("no_manifest".to_string());
    }
    if report.archived_segments <= state.max_segments {
        return None;
    }
    let floor = report.safe_floor_lsn.unwrap_or(0);
    if stats
        .archives
        .first()
        .is_some_and(|oldest| oldest.last_lsn > floor)
    {
        return Some(format!(
            "floor_below_oldest_segment: safe_floor={floor}, oldest_last_lsn={}",
            stats.archives[0].last_lsn
        ));
    }
    None
}

/// 执行整段保留（只删归档段；活动文件永不删除）。
/// EN: Run whole-segment retention (archives only; the active file is never deleted).
pub fn run_retention(
    wal: &WalHandle,
    collections_wal_path: &Path,
    role: &RoleState,
    state: &CollectionLifecycleState,
) -> RetentionReport {
    run_retention_with(wal, collections_wal_path, role, state, None, false)
}

/// leader 周期任务专用：把活跃 follower 的最小 durable LSN 作为下限。
/// EN: Leader periodic-task entry: use the minimum active follower durable LSN
/// as the floor.
pub fn run_retention_with_follower_floor(
    wal: &WalHandle,
    collections_wal_path: &Path,
    role: &RoleState,
    state: &CollectionLifecycleState,
    follower_floor: Option<u64>,
) -> RetentionReport {
    run_retention_with(
        wal,
        collections_wal_path,
        role,
        state,
        follower_floor,
        false,
    )
}

/// 管理员 compact 专用：保留租约已在本请求中设置，但仍要执行本轮保留。
/// EN: Admin compact only: the request already set the lease but still needs
/// this round of retention to run.
pub fn run_retention_for_admin(
    wal: &WalHandle,
    collections_wal_path: &Path,
    role: &RoleState,
    state: &CollectionLifecycleState,
) -> RetentionReport {
    run_retention_with(wal, collections_wal_path, role, state, None, true)
}

fn run_retention_with(
    wal: &WalHandle,
    collections_wal_path: &Path,
    role: &RoleState,
    state: &CollectionLifecycleState,
    follower_floor: Option<u64>,
    ignore_lease: bool,
) -> RetentionReport {
    let mut report = retention_status_with(
        wal,
        collections_wal_path,
        role,
        state,
        follower_floor,
        ignore_lease,
    );
    if report.blocked_reason.is_some() {
        state.metrics.blocked_total.fetch_add(1, Ordering::Relaxed);
        return report;
    }
    let Some(safe_floor) = report.safe_floor_lsn else {
        return report;
    };
    state
        .metrics
        .last_safe_floor_lsn
        .store(safe_floor, Ordering::Relaxed);
    match wal.retain_segments(safe_floor, state.max_segments, state.max_deletes_per_run) {
        Ok(outcome) => {
            report.deleted_segments = outcome.deleted_segments;
            report.bytes_freed = outcome.bytes_freed;
            report.first_available_lsn = outcome.first_available_lsn;
            report.blocked_reason = outcome.blocked_reason;
            report.archived_segments = report
                .archived_segments
                .saturating_sub(outcome.deleted_segments);
            report.archived_bytes = report.archived_bytes.saturating_sub(outcome.bytes_freed);
            state.metrics.runs_total.fetch_add(1, Ordering::Relaxed);
            state
                .metrics
                .last_deleted_segments
                .store(outcome.deleted_segments as u64, Ordering::Relaxed);
            if outcome.deleted_segments > 0 {
                state
                    .metrics
                    .segments_deleted_total
                    .fetch_add(outcome.deleted_segments as u64, Ordering::Relaxed);
                state
                    .metrics
                    .bytes_freed_total
                    .fetch_add(outcome.bytes_freed, Ordering::Relaxed);
            }
        }
        Err(e) => {
            report.blocked_reason = Some(format!("retain_failed:{e}"));
            state.metrics.blocked_total.fetch_add(1, Ordering::Relaxed);
        }
    }
    report
}

/// 快照被拒原因：名单为空且此前没有 manifest（保持既有行为，不凭空生成记录）。
/// 跨模块（recovery 决定日志级别）以常量匹配，避免字面量改名后静默失配。
/// EN: Snapshot blocked reason: empty list and no prior manifest (legacy
/// behavior; never fabricate records). Matched as a constant across modules
/// (recovery decides the log level) so a literal rename cannot silently break it.
pub const BLOCKED_NO_COLLECTIONS: &str = "no_collections";
/// 快照被拒原因：follower 的空名单不可信（append 先于 apply），保留旧 manifest。
/// EN: Snapshot blocked reason: a follower's empty list is untrustworthy
/// (append precedes apply); keep the previous manifest.
pub const BLOCKED_NO_COLLECTIONS_FOLLOWER: &str = "no_collections_follower";

/// 写一轮 collection 快照 + manifest；不删除任何 WAL。
///
/// `allow_empty_manifest`：名单为空时是否允许提交空 manifest（推进 barrier）。
/// follower 必须传 `false`：其本地 WAL 是 append 先于 apply，`next_lsn - 1`
/// 可能覆盖尚未应用的 create/upsert；把这种 barrier 写成 safe 会让保留策略
/// 截断未应用记录，重启后该 collection 永久丢失。follower 保留旧 manifest
/// （保守、不丢数据），待有 collection 应用落地或 promote 后再由 leader 路径推进。
/// EN: Write one collection snapshot cycle plus manifest; deletes no WAL.
/// `allow_empty_manifest` gates the empty-name empty-manifest commit. Followers
/// must pass `false`: their local WAL appends before apply, so `next_lsn - 1`
/// may cover a not-yet-applied create/upsert; publishing that barrier as safe
/// lets retention truncate unapplied records and loses the collection on
/// restart. Followers keep the previous manifest (conservative, no data loss)
/// until a collection lands or the node is promoted.
pub async fn write_collection_snapshots(
    manager: &Arc<CollectionManager>,
    collections_wal: &Arc<WalHandle>,
    collections_wal_path: &Path,
    encryption: Option<Arc<Encryption>>,
    metrics: &Arc<GraphSnapshotMetrics>,
    allow_empty_manifest: bool,
) -> CollectionSnapshotOutcome {
    let initial = manager.list();
    let dir = collection_snapshots_dir(collections_wal_path);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        metrics.inc_failures();
        return CollectionSnapshotOutcome {
            blocked_reason: Some(format!("create_snapshots_dir_failed:{e}")),
            ..Default::default()
        };
    }
    let manifest_path = dir.join("collections.manifest");
    let previous_manifest =
        houyi_persistence::collection_snapshot_file::read_collection_manifest(&manifest_path)
            .ok()
            .flatten();
    let previous_seq = previous_manifest
        .as_ref()
        .map(|m| m.manifest_seq)
        .unwrap_or(0);
    // 全部 collection 已删除、但磁盘上仍留有 manifest：写一份空 manifest 把
    // barrier 前推，使 WAL 保留策略可以回收已无读者的历史段。没有历史 manifest
    // 时保持 no_collections（此前行为，也避免首启即空库时凭空生成记录）。
    // EN: all collections were dropped but a manifest exists: publish an empty
    // manifest to advance the barrier so WAL retention can reclaim history with
    // no readers. With no prior manifest, keep the legacy `no_collections` outcome.
    if initial.is_empty() {
        // follower 的空名单快照不可信：manager 为空不代表 WAL 里没有尚未 apply
        // 的 create/upsert（append 已经推进 next_lsn），barrier 会越过未应用状态。
        // 保留旧 manifest 不动——不推进 safe 最多让本地 WAL 多留一段，绝不会丢
        // 数据。该判定必须先于 manifest 是否存在：刚引导的空 leader 副本同样没有
        // 旧 manifest，不能因此落回 leader 的 no_collections 语义。
        // EN: An empty snapshot is not trustworthy on a follower: an empty manager
        // does not mean the WAL has no pending create/upsert (append already
        // advanced next_lsn), so the barrier would outrun applied state. Leaving
        // the previous manifest untouched costs some extra local WAL but can
        // never lose data. This check must precede the manifest-existence check:
        // a freshly bootstrapped empty-leader replica has no prior manifest
        // either and must not fall back to the leader's `no_collections` semantics.
        if !allow_empty_manifest {
            return CollectionSnapshotOutcome {
                blocked_reason: Some(BLOCKED_NO_COLLECTIONS_FOLLOWER.to_string()),
                ..Default::default()
            };
        }
        let Some(previous) = previous_manifest.as_ref() else {
            return CollectionSnapshotOutcome {
                blocked_reason: Some(BLOCKED_NO_COLLECTIONS.to_string()),
                ..Default::default()
            };
        };
        let barrier = match read_next_lsn(collections_wal).await {
            Ok(next) => next.saturating_sub(1),
            Err(reason) => {
                metrics.inc_failures();
                return CollectionSnapshotOutcome {
                    blocked_reason: Some(reason),
                    ..Default::default()
                };
            }
        };
        // 读 barrier 之后再列一次名单：barrier 是 `next_lsn - 1`，若某个
        // collection 在首次 list 之后、barrier 读取之前完成 Create，其
        // Create LSN ≤ barrier；此刻写空 manifest（safe = barrier）会把它的
        // create/upsert 记录标成可截断，重启后数据凭空消失。发现新名单则退回
        // 正常快照路径（其 re-list + stability 循环保证 barrier 前创建必被收录）。
        // EN: Re-list after reading the barrier: barrier = next_lsn - 1, so a
        // collection created after the first list but before the barrier read has
        // Create LSN <= barrier. Publishing an empty manifest (safe = barrier)
        // would mark its create/upsert records truncatable and lose it on restart.
        // Fall through to the normal path when the list is no longer empty (its
        // re-list/stability loop guarantees every pre-barrier create is captured).
        if manager.list().is_empty() {
            let manifest = CollectionManifest {
                format_version: 1,
                manifest_seq: previous.manifest_seq.saturating_add(1),
                created_at_unix: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                global_barrier_lsn: barrier,
                // 空名单的不变式：safe = barrier（见 validate_manifest 的同口径）。
                // EN: empty-manifest invariant: safe == barrier (same rule in validate_manifest).
                safe_checkpoint_lsn: barrier,
                collections: Vec::new(),
            };
            if let Err(e) =
                houyi_persistence::collection_snapshot_file::write_collection_manifest_atomic(
                    &manifest_path,
                    &manifest,
                )
            {
                metrics.inc_manifest_rejects();
                metrics.inc_failures();
                return CollectionSnapshotOutcome {
                    blocked_reason: Some(format!("manifest_write_failed:{e}")),
                    ..Default::default()
                };
            }
            metrics.inc_manifest_writes();
            cleanup_unreferenced(&dir, &manifest, 2);
            tracing::info!(
                barrier_lsn = barrier,
                "collection snapshots: empty manifest committed after drop"
            );
            return CollectionSnapshotOutcome {
                manifest: Some(manifest),
                blocked_reason: None,
                ..Default::default()
            };
        }
    }
    let epoch_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let mut entries = Vec::with_capacity(initial.len());
    let mut done = HashSet::with_capacity(initial.len());
    let mut written_paths = Vec::new();
    let mut bytes_written = 0u64;
    let mut barrier = 0u64;
    let mut stable = false;
    // 每轮：补拍所有未覆盖 collection → 读 barrier → 再列名单。任何在 barrier
    // 之后创建的 collection 其 Create LSN 必大于 barrier，可安全不在本轮 manifest。
    // EN: Each round: snapshot every uncovered collection, read the barrier, then
    // list again. A collection created after the barrier necessarily has a greater
    // Create LSN and may safely be absent from this manifest.
    for _ in 0..3 {
        let mut current = manager.list();
        current.sort();
        for name in current {
            if done.contains(&name) {
                continue;
            }
            let Some(col) = manager.get(&name) else {
                // 删除导致 list→get 间隙缺席：跳过这一项，本轮其余 collection
                // 继续；已写文件不清理（同一轮仍要提交 manifest）。
                // EN: dropped between list and get: skip it and keep snapshotting
                // the rest; already-written files stay (the round still commits).
                continue;
            };
            if !col.backend.supports_collection_snapshot() {
                cleanup_files(&written_paths);
                return CollectionSnapshotOutcome {
                    blocked_reason: Some(format!("unsupported_backend:{name}")),
                    ..Default::default()
                };
            }
            match snapshot_one(
                &col,
                &name,
                &dir,
                epoch_nanos,
                SEQ.fetch_add(1, Ordering::Relaxed),
                encryption.clone(),
            )
            .await
            {
                Ok(Some((entry, path, bytes))) => {
                    metrics.inc_writes(bytes as usize);
                    bytes_written += bytes;
                    written_paths.push(path);
                    entries.push(entry);
                    done.insert(name);
                }
                // 快照期间被删除：按 skip 处理，不计失败、不写 manifest 条目。
                // EN: dropped mid-snapshot — a skip, not a failure or manifest entry.
                Ok(None) => {}
                Err(reason) => {
                    cleanup_files(&written_paths);
                    metrics.inc_failures();
                    return CollectionSnapshotOutcome {
                        blocked_reason: Some(reason),
                        ..Default::default()
                    };
                }
            }
        }
        match read_next_lsn(collections_wal).await {
            Ok(next) => barrier = next.saturating_sub(1),
            Err(reason) => {
                cleanup_files(&written_paths);
                metrics.inc_failures();
                return CollectionSnapshotOutcome {
                    blocked_reason: Some(reason),
                    ..Default::default()
                };
            }
        }
        if manager.list().iter().all(|name| done.contains(name)) {
            stable = true;
            break;
        }
    }
    if !stable {
        cleanup_files(&written_paths);
        metrics.inc_failures();
        return CollectionSnapshotOutcome {
            blocked_reason: Some("collection_names_unstable".to_string()),
            ..Default::default()
        };
    }
    // 本轮快照途中所有 collection 都被删除时会落到空名单：safe 退化为 barrier，
    // 与「入口即空」分支同一语义，follower 同样不得提交（其 next_lsn 可能已
    // 越过尚未应用 create/upsert）。有旧 manifest 就保留不动；从未有过 manifest
    // 时维持既有的 no_collections（不在首启空库凭空生成记录）。follower 判定
    // 同样先于 manifest 存在性。
    // EN: When every collection was dropped during the round the empty-name case
    // is reached and safe degrades to the barrier — the same semantics as the
    // initially-empty branch, so followers must not commit either (their
    // next_lsn may outrun an unapplied create/upsert). Keep the previous
    // manifest if one exists; with no prior manifest keep the legacy
    // `no_collections` outcome (never fabricate records on an empty first boot).
    // The follower check also precedes manifest existence here.
    if entries.is_empty() {
        if !allow_empty_manifest {
            return CollectionSnapshotOutcome {
                blocked_reason: Some(BLOCKED_NO_COLLECTIONS_FOLLOWER.to_string()),
                ..Default::default()
            };
        }
        if previous_manifest.is_none() {
            return CollectionSnapshotOutcome {
                blocked_reason: Some(BLOCKED_NO_COLLECTIONS.to_string()),
                ..Default::default()
            };
        }
    }
    // 空名单（本轮全部被删除）时 safe 取 barrier：与 manifest 校验口径一致，
    // 且保证 barrier 之后新建 collection 的 Create LSN 不会被截断。
    // EN: empty list (all dropped this round): safe = barrier, matching the
    // manifest validator and keeping post-barrier Create LSNs un-truncated.
    let safe_checkpoint_lsn = entries
        .iter()
        .map(|e| e.checkpoint_lsn)
        .min()
        .unwrap_or(barrier);
    let max_checkpoint = entries.iter().map(|e| e.checkpoint_lsn).max().unwrap_or(0);
    let manifest = CollectionManifest {
        format_version: 1,
        manifest_seq: previous_seq.saturating_add(1),
        created_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        global_barrier_lsn: barrier.max(max_checkpoint),
        safe_checkpoint_lsn,
        collections: entries,
    };
    if let Err(e) = houyi_persistence::collection_snapshot_file::write_collection_manifest_atomic(
        &manifest_path,
        &manifest,
    ) {
        cleanup_files(&written_paths);
        metrics.inc_manifest_rejects();
        metrics.inc_failures();
        return CollectionSnapshotOutcome {
            blocked_reason: Some(format!("manifest_write_failed:{e}")),
            ..Default::default()
        };
    }
    metrics.inc_manifest_writes();
    cleanup_unreferenced(&dir, &manifest, 2);
    tracing::info!(
        collections = manifest.collections.len(),
        safe_checkpoint_lsn = manifest.safe_checkpoint_lsn,
        barrier_lsn = manifest.global_barrier_lsn,
        files = written_paths.len(),
        "collection snapshots: manifest committed"
    );
    CollectionSnapshotOutcome {
        manifest: Some(manifest),
        blocked_reason: None,
        files_written: written_paths.len(),
        bytes_written,
    }
}

async fn read_next_lsn(collections_wal: &Arc<WalHandle>) -> Result<u64, String> {
    let wal = Arc::clone(collections_wal);
    match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::task::spawn_blocking(move || wal.next_lsn()),
    )
    .await
    {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(format!("next_lsn_task_failed:{e}")),
        Err(_) => Err("next_lsn_timeout".to_string()),
    }
}

/// `Ok(None)`：collection 在快照期间被删除（调用方按 skip 处理，不计失败）。
/// EN: `Ok(None)` means the collection was dropped mid-snapshot (caller skips it).
async fn snapshot_one(
    col: &crate::collection::Collection,
    name: &str,
    dir: &Path,
    epoch_nanos: u128,
    seq: u64,
    encryption: Option<Arc<Encryption>>,
) -> Result<Option<(CollectionManifestEntry, PathBuf, u64)>, String> {
    // 锁内只做 O(n) 克隆/导出：标量完整物理视图 + Flat/HII 原始向量。
    // HNSW 图编码在锁外（可能较慢），期间的新写由尾段重放补齐。
    // EN: Inside the lock we only clone/export (O(n)): the full physical scalar
    // view plus raw vectors for Flat/HII. HNSW graph encoding happens outside the
    // lock; newer writes are replayed from the tail.
    let store = {
        // begin_write 的双检同时覆盖「拿到 Arc 后、取锁前被删除」的窗口。
        // EN: begin_write's double check also covers the fetched-Arc/deleted-before-lock window.
        let _write = match col.begin_write().await {
            Ok(guard) => guard,
            Err(_) => return Ok(None),
        };
        let (rows, tombstones) = col
            .scalars
            .lock()
            .map_err(|e| format!("scalar_lock_poisoned:{name}:{e}"))?
            .snapshot_view();
        let checkpoint = col.applied_lsn();
        let ha_level = col.ha_level().code();
        // 度量/HNSW 参数在锁内定格：图快照字节本身按该度量构建，恢复时若按
        // 历史默认 Cosine 重建，HNSW 恢复会因 metric 不符拒绝整份快照。
        // EN: metric/HNSW are captured under the lock: the graph bytes were built
        // under this metric, and rebuilding under the historical Cosine default
        // would make HNSW restore reject the whole snapshot on metric mismatch.
        let metric = col.metric.to_snapshot_byte();
        let hnsw = (col.index_type == "hnsw").then(|| {
            let p = col.hnsw_params();
            SnapshotHnswParams {
                m: p.m as u32,
                ef_construction: p.ef_construction as u32,
                ef_search: p.ef_search as u32,
            }
        });
        let early_vectors = if matches!(&col.backend, crate::index_backend::IndexBackend::Hnsw(_)) {
            None
        } else {
            Some(
                col.backend
                    .encode_collection_snapshot_vectors()
                    .await
                    .map_err(|e| format!("vector_encode_failed:{name}:{e}"))?,
            )
        };
        (
            rows,
            tombstones,
            checkpoint,
            ha_level,
            metric,
            hnsw,
            early_vectors,
        )
    };
    let (rows, tombstones, checkpoint, ha_level, metric, hnsw, early_vectors) = store;
    let (kind, vector_count, vector_bytes) = match early_vectors {
        Some(v) => v,
        None => {
            drain_pending_builder(&col.backend).await;
            col.backend
                .encode_collection_snapshot_vectors()
                .await
                .map_err(|e| format!("vector_encode_failed:{name}:{e}"))?
        }
    };
    let scalars = rows
        .into_iter()
        .map(|(id, columns)| SnapshotScalarRow {
            id,
            columns: columns
                .into_iter()
                .map(|(k, v)| (k, snapshot_scalar(&v)))
                .collect(),
        })
        .collect();
    let payload = CollectionSnapshotPayload {
        checkpoint_lsn: checkpoint,
        name: name.to_string(),
        dims: col.dims as u64,
        index_type: col.index_type.clone(),
        ha_level,
        metric: Some(metric),
        hnsw,
        vector_kind: kind,
        vector_count,
        vector_bytes,
        scalars,
        tombstones,
    };
    let payload_bytes = encode_collection_snapshot_payload(&payload)
        .map_err(|e| format!("payload_encode_failed:{name}:{e}"))?;
    let payload_crc32 = crc32fast::hash(&payload_bytes);
    // 编码阶段不持锁（可能耗时）：写文件前再查一次删除哨兵——已删则整份丢弃，
    // 避免把结束于删除之后的 payload 写进目录（manifest 也不含它）。
    // EN: encoding runs unlocked; re-check the drop sentinel before writing so a
    // bogus post-drop payload never lands on disk (nor in the manifest).
    if col.is_deleted() {
        return Ok(None);
    }
    let file = snapshot_file_name(name, checkpoint, epoch_nanos, seq);
    let path = dir.join(&file);
    let write_path = path.clone();
    let write_enc = encryption.clone();
    let write_bytes = payload_bytes.clone();
    tokio::task::spawn_blocking(move || {
        write_graph_snapshot_atomic_with_encryption(
            &write_path,
            checkpoint,
            &write_bytes,
            write_enc.as_deref(),
        )
    })
    .await
    .map_err(|e| format!("snapshot_task_join_failed:{name}:{e}"))?
    .map_err(|e| format!("snapshot_write_failed:{name}:{e}"))?;
    let file_len = std::fs::metadata(&path)
        .map_err(|e| format!("snapshot_stat_failed:{name}:{e}"))?
        .len();
    let entry = CollectionManifestEntry {
        name: name.to_string(),
        checkpoint_lsn: checkpoint,
        dims: col.dims as u64,
        index_type: col.index_type.clone(),
        ha_level,
        metric: Some(metric),
        vector_kind: kind as u8,
        file,
        file_len,
        payload_crc32,
        vector_count,
        scalar_rows: payload.scalars.len() as u64,
    };
    Ok(Some((entry, path, payload_bytes.len() as u64)))
}

/// 加载 manifest + 全部快照；任何失败在两态下的处理见调用方：
/// - `first_available == 0`：返回 Ok(None)，调用方回退 legacy graph snapshot + 全量重放；
/// - `first_available > 0`：返回 Err（截断历史没有 manifest 不可恢复）。
///
/// EN: Load manifest + all snapshots. On failure the caller either falls back
/// (full history) or treats it as fatal (truncated history).
pub async fn load_collection_snapshots(
    manager: &Arc<CollectionManager>,
    collections_wal_path: &Path,
    first_available_lsn: u64,
    encryption: Option<Arc<Encryption>>,
    metrics: &Arc<GraphSnapshotMetrics>,
) -> std::io::Result<Option<LoadedCollectionSnapshots>> {
    match load_collection_snapshots_inner(
        manager,
        collections_wal_path,
        first_available_lsn,
        encryption,
        metrics,
    )
    .await
    {
        Err(e) if e.to_string().contains("__SNAPSHOT_FALLBACK__") => Ok(None),
        other => other,
    }
}

async fn load_collection_snapshots_inner(
    manager: &Arc<CollectionManager>,
    collections_wal_path: &Path,
    first_available_lsn: u64,
    encryption: Option<Arc<Encryption>>,
    metrics: &Arc<GraphSnapshotMetrics>,
) -> std::io::Result<Option<LoadedCollectionSnapshots>> {
    let dir = collection_snapshots_dir(collections_wal_path);
    let manifest_path = dir.join("collections.manifest");
    let manifest = match houyi_persistence::collection_snapshot_file::read_collection_manifest(
        &manifest_path,
    ) {
        Ok(Some(m)) => m,
        Ok(None) => {
            return if first_available_lsn > 0 {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "collections WAL starts at lsn {first_available_lsn} but manifest {} is missing",
                        manifest_path.display()
                    ),
                ))
            } else {
                Ok(None)
            };
        }
        Err(e) => {
            return if first_available_lsn > 0 {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "collections WAL starts at lsn {first_available_lsn} but manifest {} is invalid: {e}",
                        manifest_path.display()
                    ),
                ))
            } else {
                metrics.inc_rejects();
                tracing::warn!(error = %e, "collection snapshots: invalid manifest; falling back to full replay");
                Ok(None)
            };
        }
    };

    // 先读取 + 解码全部文件，任何失败都不触碰 manager（避免半恢复后回退困难）。
    // EN: Read and decode every file first; a failure must not partially mutate
    // the manager (otherwise legacy fallback becomes unsafe).
    let mut decoded = Vec::with_capacity(manifest.collections.len());
    for entry in &manifest.collections {
        let path = dir.join(&entry.file);
        let file_len = match std::fs::metadata(&path) {
            Ok(m) => m.len(),
            Err(e) => {
                return snapshot_load_failure(
                    first_available_lsn,
                    metrics,
                    format!("snapshot {} missing/unreadable: {e}", entry.file),
                );
            }
        };
        if file_len != entry.file_len {
            return snapshot_load_failure(
                first_available_lsn,
                metrics,
                format!(
                    "snapshot {} length {file_len} != manifest {}",
                    entry.file, entry.file_len
                ),
            );
        }
        let envelope = match read_graph_snapshot_with_encryption(&path, encryption.as_deref()) {
            Ok(Some(f)) => f,
            Ok(None) => {
                return snapshot_load_failure(
                    first_available_lsn,
                    metrics,
                    format!("snapshot {} vanished", entry.file),
                );
            }
            Err(e) => {
                if e.is_missing_key() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("collection snapshot missing encryption key: {e}"),
                    ));
                }
                return snapshot_load_failure(
                    first_available_lsn,
                    metrics,
                    format!("snapshot {} invalid: {e}", entry.file),
                );
            }
        };
        if envelope.checkpoint_lsn != entry.checkpoint_lsn {
            return snapshot_load_failure(
                first_available_lsn,
                metrics,
                format!(
                    "snapshot {} checkpoint {} != manifest {}",
                    entry.file, envelope.checkpoint_lsn, entry.checkpoint_lsn
                ),
            );
        }
        if crc32fast::hash(&envelope.payload) != entry.payload_crc32 {
            return snapshot_load_failure(
                first_available_lsn,
                metrics,
                format!("snapshot {} payload CRC mismatch", entry.file),
            );
        }
        let payload = match decode_collection_snapshot_payload(&envelope.payload) {
            Ok(p) => p,
            Err(e) => {
                return snapshot_load_failure(
                    first_available_lsn,
                    metrics,
                    format!("snapshot {} payload invalid: {e}", entry.file),
                );
            }
        };
        // v1 旧 manifest 的 metric 为 None：按契约不做度量相等约束（由 payload /
        // 后端历史语义决定）；manifest 携带度量时（v2 起）两侧必须一致（OCR #36）。
        // EN: legacy v1 manifests carry metric=None and the documented contract is
        // "no equality enforcement" (payload/backend history decides); when the
        // manifest carries a metric (v2 on), both sides must agree (OCR #36).
        if payload.name != entry.name
            || payload.dims != entry.dims
            || payload.index_type != entry.index_type
            || payload.ha_level != entry.ha_level
            || entry.metric.is_some_and(|m| payload.metric != Some(m))
            || payload.vector_kind as u8 != entry.vector_kind
        {
            return snapshot_load_failure(
                first_available_lsn,
                metrics,
                format!("snapshot {} metadata mismatch with manifest", entry.file),
            );
        }
        decoded.push(payload);
    }

    let mut checkpoints = HashMap::with_capacity(decoded.len());
    for payload in decoded {
        // 先按快照记录的度量/HNSW 参数重建后端：HNSW 图字节按原度量构建，
        // 恢复期期望匹配会拒绝度量不符的快照。v1 快照的 None 交给管理器按
        // index_type 历史语义推断（与旧服务端行为逐字节一致）。
        // EN: rebuild the backend from the snapshot's metric/HNSW first — graph
        // bytes were built under that metric and restore rejects a mismatch.
        // v1's None defers to the manager's historical index_type inference.
        let metric = match payload.metric {
            Some(b) => Some(DistanceMetric::from_snapshot_byte(b).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("collection snapshot: unsupported metric byte {b}"),
                )
            })?),
            None => None,
        };
        let hnsw = payload.hnsw.map(|p| crate::collection::HnswParams {
            m: p.m as usize,
            ef_construction: p.ef_construction as usize,
            ef_search: p.ef_search as usize,
        });
        let (col, _) = manager
            .get_or_create_typed_with_metric(
                &payload.name,
                payload.dims as usize,
                &payload.index_type,
                metric,
                hnsw,
            )
            .map_err(std::io::Error::other)?;
        col.backend
            .restore_collection_snapshot_vectors(
                payload.vector_kind,
                payload.vector_count,
                &payload.vector_bytes,
            )
            .await
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("restore {} vectors: {e}", payload.name),
                )
            })?;
        let rows = payload
            .scalars
            .into_iter()
            .map(|row| {
                (
                    row.id,
                    row.columns
                        .into_iter()
                        .map(|(k, v)| (k, scalar_value(v)))
                        .collect(),
                )
            })
            .collect();
        *col.scalars
            .lock()
            .map_err(|e| std::io::Error::other(e.to_string()))? =
            crate::collection::CollectionScalars::from_rows(rows, payload.tombstones);
        if let Some(level) = HaLevel::from_code(payload.ha_level) {
            col.set_ha_level(level);
        }
        col.note_applied_lsn(payload.checkpoint_lsn);
        checkpoints.insert(payload.name, payload.checkpoint_lsn);
        metrics.inc_loads();
    }
    // 启动时也清理孤儿 generation，但保留最近 2 代用于排障。
    // EN: Clean orphan generations on startup too, keeping the two most recent
    // for debugging.
    cleanup_unreferenced(&dir, &manifest, 2);
    tracing::info!(
        collections = manifest.collections.len(),
        safe_checkpoint_lsn = manifest.safe_checkpoint_lsn,
        "collection snapshots: manifest restored"
    );
    Ok(Some(LoadedCollectionSnapshots {
        manifest,
        checkpoints,
    }))
}

fn snapshot_load_failure<T>(
    first_available_lsn: u64,
    metrics: &Arc<GraphSnapshotMetrics>,
    reason: String,
) -> std::io::Result<T> {
    if first_available_lsn > 0 {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "collection snapshot restore failed with truncated history (first_available={first_available_lsn}): {reason}"
            ),
        ))
    } else {
        metrics.inc_rejects();
        tracing::warn!(error = %reason, "collection snapshots: falling back to full replay");
        // 调用方用 Ok(None) 表示回退；这里用 sentinel 错误类型在上一层转换。
        // EN: Callers treat Ok(None) as a fallback; a sentinel error is converted one layer up.
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("__SNAPSHOT_FALLBACK__:{reason}"),
        ))
    }
}

fn snapshot_scalar(value: &ScalarValue) -> SnapshotScalar {
    match value {
        ScalarValue::Int64(v) => SnapshotScalar::I64(*v),
        ScalarValue::Float64(v) => SnapshotScalar::F64(*v),
        ScalarValue::Bool(v) => SnapshotScalar::Bool(*v),
        ScalarValue::Text(v) => SnapshotScalar::Text(v.clone()),
        ScalarValue::Timestamp(v) => SnapshotScalar::Timestamp(*v),
        ScalarValue::JsonText(v) => SnapshotScalar::JsonText(v.clone()),
    }
}

fn scalar_value(value: SnapshotScalar) -> ScalarValue {
    match value {
        SnapshotScalar::I64(v) => ScalarValue::Int64(v),
        SnapshotScalar::F64(v) => ScalarValue::Float64(v),
        SnapshotScalar::Bool(v) => ScalarValue::Bool(v),
        SnapshotScalar::Text(v) => ScalarValue::Text(v),
        SnapshotScalar::Timestamp(v) => ScalarValue::Timestamp(v),
        SnapshotScalar::JsonText(v) => ScalarValue::JsonText(v),
    }
}

fn cleanup_files(paths: &[PathBuf]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

fn cleanup_unreferenced(dir: &Path, manifest: &CollectionManifest, keep_recent: usize) {
    let keep: HashSet<&str> = manifest
        .collections
        .iter()
        .map(|e| e.file.as_str())
        .collect();
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    // 中文：孤儿按 mtime 倒序保留最近 N 代（同名时间戳时按文件名倒序），
    // 其余删除。保留窗口用于崩溃后人工排障，不影响 manifest 的提交语义。
    // EN: Keep the newest N orphan generations by mtime (filename descending
    // on ties) and delete the rest. The window is for post-crash debugging and
    // does not affect manifest commit semantics.
    let mut orphans: Vec<(std::time::SystemTime, String, PathBuf)> = Vec::new();
    for entry in read.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if name.starts_with('.')
            || !name.ends_with(".bin")
            || !name.contains(".collection_snapshot.")
            || keep.contains(name.as_str())
        {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        orphans.push((modified, name, entry.path()));
    }
    orphans.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    for (_, _, path) in orphans.into_iter().skip(keep_recent) {
        if std::fs::remove_file(&path).is_err() {
            tracing::warn!(path = %path.display(), "collection snapshots: cleanup failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::CollectionsReplayStats;
    use houyi_persistence::collection_wal::{CollectionWalItem, CollectionWalRecord};
    use houyi_persistence::FileWal;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "houyi-col-lifecycle-{tag}-{}-{}-{n}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 建一个 flat collection，经 WAL 记录 + apply 写入一行标量/向量。
    /// EN: Build a flat collection and write one scalar/vector row through WAL + apply.
    async fn seeded(
        dir: &Path,
    ) -> (
        Arc<CollectionManager>,
        PathBuf,
        Vec<houyi_persistence::WalEntry>,
        Arc<GraphSnapshotMetrics>,
    ) {
        let wal_path = dir.join("bench.wal");
        let cols_path = houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let create =
            houyi_persistence::collection_wal::encode_collection_create("docs", 4, "flat").unwrap();
        let create_entry = wal.append(create).unwrap();
        let items = vec![CollectionWalItem {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("a".to_string()),
            payload: Some(
                serde_json::json!({"color": "red"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        }];
        let upsert =
            houyi_persistence::collection_wal::encode_collection_upsert_batch("docs", 4, &items)
                .unwrap();
        let upsert_entry = wal.append(upsert).unwrap();
        drop(wal);

        let manager = Arc::new(CollectionManager::new(4, "flat"));
        let mut stats = CollectionsReplayStats::default();
        let mut skipped = 0usize;
        let mut poison = crate::recovery::PoisonedRowLog::new();
        crate::recovery::apply_collection_wal_record(
            &manager,
            create_entry.lsn,
            CollectionWalRecord::Create {
                name: "docs".to_string(),
                dims: 4,
                index_type: "flat".to_string(),
                options: Default::default(),
            },
            |_| None,
            &mut stats,
            &mut skipped,
            &mut poison,
        )
        .await
        .unwrap();
        crate::recovery::apply_collection_wal_record(
            &manager,
            upsert_entry.lsn,
            CollectionWalRecord::UpsertBatch {
                name: "docs".to_string(),
                dim: 4,
                items,
            },
            |_| None,
            &mut stats,
            &mut skipped,
            &mut poison,
        )
        .await
        .unwrap();
        let (_wal, entries) = FileWal::open(&cols_path).unwrap();
        let metrics = Arc::new(GraphSnapshotMetrics::default());
        (manager, cols_path, entries, metrics)
    }

    /// P5.0.2：manifest + 尾段重放的可见状态与全量重放一致，覆盖记录整条跳过。
    /// EN: P5.0.2 — manifest + tail replay matches full replay; covered records are
    /// skipped whole.
    #[tokio::test]
    async fn manifest_and_tail_replay_match_full_replay() {
        let dir = temp_dir("manifest-tail");
        let (manager, cols_path, entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        let manifest = outcome.manifest.expect("manifest must commit for flat");
        assert_eq!(manifest.collections.len(), 1);
        assert_eq!(manifest.safe_checkpoint_lsn, 1);
        assert_eq!(manifest.collections[0].checkpoint_lsn, 1);
        drop(wal);

        let fresh = Arc::new(CollectionManager::new(4, "flat"));
        let loaded = load_collection_snapshots(&fresh, &cols_path, 0, None, &metrics)
            .await
            .unwrap()
            .expect("manifest must load");
        let stats = crate::recovery::replay_collections_wal_into_manager_with_mode(
            &entries,
            &fresh,
            &loaded.checkpoints,
            true,
        )
        .await
        .unwrap();
        assert_eq!(stats.records_skipped_snapshot, 2);

        let a = manager.get("docs").unwrap();
        let b = fresh.get("docs").unwrap();
        assert_eq!(a.dims, b.dims);
        assert_eq!(a.index_type, b.index_type);
        assert_eq!(a.applied_lsn(), b.applied_lsn());
        let pa = a.scalars.lock().unwrap().payloads_for(&[1]);
        let pb = b.scalars.lock().unwrap().payloads_for(&[1]);
        assert_eq!(pa, pb);
        assert_eq!(b.scalars.lock().unwrap().row_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P5.0.2：截断历史（first_available>0）但没有 manifest 时必须拒绝启动路径。
    /// EN: P5.0.2 — truncated history without a manifest must fail the load path.
    #[tokio::test]
    async fn truncated_history_without_manifest_is_fatal() {
        let dir = temp_dir("manifest-missing");
        let (_manager, cols_path, _entries, metrics) = seeded(&dir).await;
        let fresh = Arc::new(CollectionManager::new(4, "flat"));
        let err = load_collection_snapshots(&fresh, &cols_path, 1, None, &metrics)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("manifest") && err.to_string().contains("missing"),
            "got: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P5.0.2：manifest 缺失但历史完整（first_available=0）时回退全量重放。
    /// EN: P5.0.2 — a missing manifest with full history (first_available=0)
    /// falls back to full replay.
    #[tokio::test]
    async fn missing_manifest_with_full_history_falls_back() {
        let dir = temp_dir("manifest-full-fallback");
        let (manager, cols_path, entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        assert!(outcome.manifest.is_some());
        drop(wal);
        std::fs::remove_file(collection_snapshots_dir(&cols_path).join("collections.manifest"))
            .unwrap();

        let fresh = Arc::new(CollectionManager::new(4, "flat"));
        let loaded = load_collection_snapshots(&fresh, &cols_path, 0, None, &metrics)
            .await
            .unwrap();
        assert!(loaded.is_none(), "full history must fall back, not fail");
        assert!(fresh.list().is_empty(), "fallback must not mutate manager");

        let stats = crate::recovery::replay_collections_wal_into_manager_with_mode(
            &entries,
            &fresh,
            &HashMap::new(),
            false,
        )
        .await
        .unwrap();
        assert_eq!(stats.creates, 1);
        assert_eq!(fresh.get("docs").unwrap().applied_lsn(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P5.0.2：损坏快照在完整历史下回退，在截断历史下致命。
    /// EN: P5.0.2 — a corrupt snapshot falls back with full history and is
    /// fatal with truncated history.
    #[tokio::test]
    async fn corrupt_snapshot_falls_back_or_fails_by_history() {
        let dir = temp_dir("corrupt-snapshot");
        let (manager, cols_path, entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        let manifest = outcome.manifest.expect("manifest must commit for flat");
        drop(wal);

        let path = collection_snapshots_dir(&cols_path).join(&manifest.collections[0].file);
        let mut bytes = std::fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();

        let fresh = Arc::new(CollectionManager::new(4, "flat"));
        let loaded = load_collection_snapshots(&fresh, &cols_path, 0, None, &metrics)
            .await
            .unwrap();
        assert!(loaded.is_none(), "full history must fall back");
        assert!(
            fresh.list().is_empty(),
            "fallback must not partially restore"
        );

        let err = load_collection_snapshots(&fresh, &cols_path, 1, None, &metrics)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("truncated history") || err.to_string().contains("snapshot"),
            "got: {err}"
        );

        // 回退后的全量重放必须仍能恢复可见状态。
        // EN: Full replay after fallback must still restore the visible state.
        let stats = crate::recovery::replay_collections_wal_into_manager_with_mode(
            &entries,
            &fresh,
            &HashMap::new(),
            false,
        )
        .await
        .unwrap();
        assert_eq!(stats.creates, 1);
        assert_eq!(fresh.get("docs").unwrap().applied_lsn(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P5.0.2：不支持的 backend 只阻塞 manifest，不产生删除/半快照。
    /// EN: P5.0.2 — an unsupported backend blocks the manifest without deleting
    /// or half-writing snapshots.
    #[tokio::test]
    async fn unsupported_backend_blocks_manifest() {
        let dir = temp_dir("unsupported-backend");
        let cols_path = dir.join("bench.wal");
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let manager = Arc::new(CollectionManager::new(4, "pq-flat"));
        manager
            .get_or_create_typed_with_metric("docs", 4, "pq-flat", None, None)
            .unwrap();
        let metrics = Arc::new(GraphSnapshotMetrics::default());

        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        assert!(outcome.manifest.is_none());
        assert!(
            outcome
                .blocked_reason
                .as_deref()
                .is_some_and(|r| r.contains("unsupported_backend:docs")),
            "got: {:?}",
            outcome.blocked_reason
        );
        assert!(
            !collection_snapshots_dir(&cols_path)
                .join("collections.manifest")
                .exists(),
            "blocked cycle must not commit a manifest"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 全部 collection 删除后写空 manifest：safe = barrier，且下一轮新建
    /// collection 的 Create LSN 严格大于该 barrier（截断安全）。
    /// EN: after dropping every collection, an empty manifest is committed with
    /// safe == barrier; a collection created afterwards has a strictly greater
    /// Create LSN (safe to truncate).
    #[tokio::test]
    async fn dropping_all_collections_commits_empty_manifest_at_barrier() {
        let dir = temp_dir("drop-all-empty-manifest");
        let (manager, cols_path, _entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let first =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        assert!(
            first.manifest.is_some(),
            "seeded collection must commit a full manifest first"
        );

        let col = manager.get("docs").expect("seeded collection");
        {
            let _write = col.begin_write().await.unwrap();
            col.mark_deleted();
            manager.remove_if_same("docs", &col);
        }

        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        let manifest = outcome
            .manifest
            .expect("empty manifest must commit after dropping everything");
        assert!(manifest.collections.is_empty());
        assert_eq!(
            manifest.safe_checkpoint_lsn, manifest.global_barrier_lsn,
            "empty manifest safe checkpoint must equal the barrier"
        );
        let barrier = manifest.global_barrier_lsn;
        assert_eq!(
            barrier,
            wal.next_lsn().saturating_sub(1),
            "barrier must be the last appended WAL lsn"
        );

        // barrier 之后新建 collection：Create LSN 严格大于 barrier，可安全截断。
        // EN: a post-barrier create has a strictly greater LSN, so truncation is safe.
        let create =
            houyi_persistence::collection_wal::encode_collection_create("fresh", 4, "flat")
                .unwrap();
        assert_eq!(
            wal.append(create).unwrap().lsn,
            barrier + 1,
            "post-barrier create must outrank the barrier"
        );

        // 空 manifest 必须可重新加载（校验口径自洽）。
        // EN: the empty manifest must round-trip through the loader.
        drop(wal);
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        drop(wal);
        let fresh = Arc::new(CollectionManager::new(4, "flat"));
        let loaded = load_collection_snapshots(&fresh, &cols_path, 1, None, &metrics)
            .await
            .expect("empty manifest with truncated history must load");
        assert!(loaded.is_some(), "empty manifest must be accepted");
        assert!(fresh.list().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// follower 不得提交空名单 manifest：create 已 append 但尚未 apply 时，
    /// `next_lsn - 1` 会越过应用前沿；写成 safe 会让保留策略截断它，重启后
    /// 该 collection 永久消失。旧 manifest 必须原样保留。
    /// EN: Followers must not publish empty-name manifests: when a create is
    /// appended but not yet applied, `next_lsn - 1` outruns the applied frontier;
    /// marking it safe would let retention truncate it and lose the collection on
    /// restart. The previous manifest must stay untouched.
    #[tokio::test]
    async fn follower_skips_empty_manifest_for_unapplied_create() {
        let dir = temp_dir("follower-empty-manifest");
        let (manager, cols_path, _entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let first = write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true)
            .await
            .manifest
            .expect("seed must commit a manifest first");
        let col = manager.get("docs").expect("seeded collection");
        {
            let _write = col.begin_write().await.unwrap();
            col.mark_deleted();
            manager.remove_if_same("docs", &col);
        }
        // 模拟 follower：create 已 append 进本地 WAL，但 apply 还没发生——manager 仍为空。
        // EN: emulate a follower: the create is already appended to the local WAL
        // while apply has not happened yet, so the manager is still empty.
        let create =
            houyi_persistence::collection_wal::encode_collection_create("fresh", 4, "flat")
                .unwrap();
        let appended = wal.append(create).unwrap();

        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, false).await;
        assert!(
            outcome.manifest.is_none(),
            "follower must not commit an empty manifest"
        );
        assert_eq!(
            outcome.blocked_reason.as_deref(),
            Some(BLOCKED_NO_COLLECTIONS_FOLLOWER)
        );
        let on_disk = houyi_persistence::collection_snapshot_file::read_collection_manifest(
            &collection_snapshots_dir(&cols_path).join("collections.manifest"),
        )
        .expect("read manifest")
        .expect("previous manifest must remain on disk");
        assert_eq!(on_disk.manifest_seq, first.manifest_seq);
        assert_eq!(on_disk.safe_checkpoint_lsn, first.safe_checkpoint_lsn);
        assert!(
            appended.lsn > on_disk.safe_checkpoint_lsn,
            "unapplied create must stay above the safe checkpoint"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// follower 的非空名单照常提交 manifest，但 safe 必须钳在 collection 的
    /// `applied_lsn`：本地 WAL 的 barrier（next_lsn-1）会越过尚未 apply 的记录。
    /// 这条断言锁住「有 collection 落地就继续推进」的宣传口径，并防止把门控
    /// 挪到空名单分支之外时漏掉非空路径的钳制。
    /// EN: Followers with a non-empty list still commit a manifest, but safe must
    /// be clamped to the collection's `applied_lsn`: the local-WAL barrier
    /// (next_lsn - 1) outruns records that are appended but not yet applied.
    /// This pins the "advance once a collection lands" claim and catches a
    /// hoisted gate that forgets the clamp on the non-empty path.
    #[tokio::test]
    async fn follower_non_empty_manifest_clamps_safe_to_applied_lsn() {
        let dir = temp_dir("follower-nonempty-manifest");
        let (manager, cols_path, _entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        // follower 已把 upsert 追加进本地 WAL，但 apply 还没发生：next_lsn 前进，
        // applied_lsn 不动。
        // EN: the follower already appended an upsert locally but apply has not
        // happened: next_lsn advances while applied_lsn stays put.
        let items = vec![CollectionWalItem {
            id: 2,
            values: vec![0.0, 1.0, 0.0, 0.0],
            label: None,
            payload: None,
        }];
        let appended = wal
            .append(
                houyi_persistence::collection_wal::encode_collection_upsert_batch(
                    "docs", 4, &items,
                )
                .unwrap(),
            )
            .unwrap();
        let applied = manager
            .get("docs")
            .expect("seeded collection")
            .applied_lsn();
        assert!(
            appended.lsn > applied,
            "the extra upsert must be appended but unapplied"
        );

        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, false).await;
        let manifest = outcome
            .manifest
            .expect("a non-empty follower list must still commit a manifest");
        assert_eq!(manifest.collections.len(), 1);
        assert_eq!(
            manifest.safe_checkpoint_lsn, applied,
            "safe must clamp to applied_lsn, not the WAL barrier"
        );
        assert_eq!(manifest.collections[0].checkpoint_lsn, applied);
        assert!(
            manifest.global_barrier_lsn > manifest.safe_checkpoint_lsn,
            "the barrier must actually outrun applied state for this test to bite"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 驱动一轮「快照途中全部删除」：先持 docs 写锁，单次 poll 让本轮停在
    /// `snapshot_one` 的 `begin_write`，再删除并从 manager 摘除，最后放锁等
    /// 本轮结束。返回结果供调用方按角色断言。
    /// EN: Drive one "all dropped mid-round" snapshot: hold docs' write lock, poll
    /// the round once so it stalls in `snapshot_one`'s `begin_write`, then delete
    /// and remove it from the manager, release the lock, and await the round.
    /// Returns the outcome for role-specific assertions.
    async fn drop_docs_mid_round(
        manager: &Arc<CollectionManager>,
        wal: &Arc<WalHandle>,
        cols_path: &Path,
        metrics: &Arc<GraphSnapshotMetrics>,
        allow_empty_manifest: bool,
    ) -> CollectionSnapshotOutcome {
        let col = manager.get("docs").expect("seeded collection");
        let guard = col.begin_write().await.expect("live collection");
        let mut fut = Box::pin(write_collection_snapshots(
            manager,
            wal,
            cols_path,
            None,
            metrics,
            allow_empty_manifest,
        ));
        // 单次 poll 足以跑到 begin_write（其前全是同步逻辑）；若未阻塞说明
        // 测试前提已失效，直接失败，而不是用 sleep 掩盖竞态。noop waker 保证
        // 这次 poll 不会注册任何唤醒，也不会让出 runtime。
        // EN: One poll suffices to reach begin_write (everything before it is
        // sync); if it did not block the premise is broken — fail instead of
        // papering over the race with a sleep. A no-op waker guarantees this
        // poll registers no wakeup and yields nothing to the runtime.
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(
            std::future::Future::poll(fut.as_mut(), &mut cx).is_pending(),
            "the round must stall on the held write lock"
        );
        col.mark_deleted();
        manager.remove_if_same("docs", &col);
        drop(guard);
        fut.await
    }

    /// follower 在「途中全部删除」的窗口同样不得提交空 manifest：barrier 越过
    /// 未应用的 create 时，空 manifest 会让保留截断它。旧 manifest 原样保留。
    /// EN: A follower must not commit an empty manifest in the "all dropped
    /// mid-round" window either: when the barrier outruns an unapplied create,
    /// an empty manifest would let retention truncate it. The previous manifest
    /// stays untouched.
    #[tokio::test]
    async fn follower_mid_round_drop_keeps_previous_manifest() {
        let dir = temp_dir("follower-mid-round-drop");
        let (manager, cols_path, _entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let first = write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true)
            .await
            .manifest
            .expect("baseline manifest");
        // follower 已 append 一个尚未 apply 的 create：barrier 必须领先 safe。
        // EN: the follower has appended an unapplied create: barrier > safe.
        let appended = wal
            .append(
                houyi_persistence::collection_wal::encode_collection_create("fresh", 4, "flat")
                    .unwrap(),
            )
            .unwrap();
        assert!(appended.lsn > first.safe_checkpoint_lsn);

        let outcome = drop_docs_mid_round(&manager, &wal, &cols_path, &metrics, false).await;
        assert!(
            outcome.manifest.is_none(),
            "follower must keep the previous manifest"
        );
        assert_eq!(
            outcome.blocked_reason.as_deref(),
            Some(BLOCKED_NO_COLLECTIONS_FOLLOWER)
        );
        let on_disk = houyi_persistence::collection_snapshot_file::read_collection_manifest(
            &collection_snapshots_dir(&cols_path).join("collections.manifest"),
        )
        .expect("read manifest")
        .expect("previous manifest must remain on disk");
        assert_eq!(on_disk.manifest_seq, first.manifest_seq);
        assert_eq!(on_disk.safe_checkpoint_lsn, first.safe_checkpoint_lsn);
        assert!(on_disk.collections.iter().any(|e| e.name == "docs"));
        assert!(
            appended.lsn > on_disk.safe_checkpoint_lsn,
            "the unapplied create must stay above the safe checkpoint"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// leader（allow_empty_manifest=true）在同样的途中全删窗口仍提交空 manifest，
    /// safe = barrier：保留可以推进，且 barrier 之后新建的 collection 不会被截断。
    /// EN: A leader (allow_empty_manifest = true) still commits the empty manifest
    /// in the same mid-round window with safe = barrier: retention advances and a
    /// post-barrier create is never truncated.
    #[tokio::test]
    async fn leader_mid_round_drop_commits_empty_manifest_at_barrier() {
        let dir = temp_dir("leader-mid-round-drop");
        let (manager, cols_path, _entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let first = write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true)
            .await
            .manifest
            .expect("baseline manifest");
        assert_eq!(first.collections.len(), 1);

        let outcome = drop_docs_mid_round(&manager, &wal, &cols_path, &metrics, true).await;
        let manifest = outcome.manifest.expect("leader commits the empty manifest");
        assert!(manifest.collections.is_empty());
        assert_eq!(manifest.safe_checkpoint_lsn, manifest.global_barrier_lsn);
        assert!(manifest.global_barrier_lsn >= first.safe_checkpoint_lsn);
        assert!(manager.list().is_empty());
        // 空 manifest 必须能在截断历史下重新加载（校验口径自洽）。
        // EN: the empty manifest must reload under truncated history.
        drop(wal);
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        drop(wal);
        let fresh = Arc::new(CollectionManager::new(4, "flat"));
        let loaded = load_collection_snapshots(
            &fresh,
            &cols_path,
            manifest.safe_checkpoint_lsn,
            None,
            &metrics,
        )
        .await
        .expect("empty manifest with truncated history must load");
        assert!(loaded.is_some());
        assert!(fresh.list().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 已删除的 collection 不进入本轮 manifest，其余 collection 继续快照。
    /// EN: a dropped collection is absent from the manifest while the remaining
    /// collections are still snapshotted.
    #[tokio::test]
    async fn dropped_collection_is_absent_from_manifest() {
        let dir = temp_dir("dropped-absent");
        let (manager, cols_path, _entries, metrics) = seeded(&dir).await;
        manager
            .get_or_create_typed_with_metric("other", 4, "flat", None, None)
            .unwrap();
        let other = manager.get("other").unwrap();
        {
            let _write = other.begin_write().await.unwrap();
            other.mark_deleted();
            manager.remove_if_same("other", &other);
        }
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        let manifest = outcome.manifest.expect("remaining collection must commit");
        let names: Vec<&str> = manifest
            .collections
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["docs"],
            "dropped collection must not be snapshotted"
        );
        // 文件名形如 `other.<hash>.<checkpoint>.<epoch>.<seq>.collection_snapshot.bin`，
        // 只能按前缀扫描（拼一个不存在的固定文件名是永真断言）。
        // EN: file names look like
        // `other.<hash>.<checkpoint>.<epoch>.<seq>.collection_snapshot.bin`, so
        // scan by prefix — a fabricated fixed name is a vacuous assertion.
        // read_dir 失败必须响亮（unwrap_or(false) 会让断言永真）。
        // EN: a read_dir failure must be loud (unwrap_or(false) makes the
        // assertion vacuous).
        // 逐项读取失败也必须响亮：`.flatten()` 会吞掉 entry 级 io::Error，
        // 让 any(...) 静默返回 false 而断言永真。
        // EN: per-entry read failures must be loud too: `.flatten()` swallows
        // entry-level io::Errors, letting any(...) return false silently and
        // making the assertion vacuous.
        let leaked = std::fs::read_dir(collection_snapshots_dir(&cols_path))
            .expect("snapshot directory must exist")
            .collect::<std::io::Result<Vec<_>>>()
            .expect("snapshot directory entries must be readable")
            .iter()
            .any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with("other."))
            });
        assert!(!leaked, "dropped collection must not leave a snapshot file");
        drop(wal);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P5.0.2 关键不变量：WAL 已 append 但未成功应用的记录不得被快照 checkpoint 覆盖。
    /// EN: Key P5.0.2 invariant — a WAL record appended but not applied must not
    /// be covered by the snapshot checkpoint.
    #[tokio::test]
    async fn checkpoint_excludes_unapplied_wal_record() {
        let dir = temp_dir("unapplied-checkpoint");
        let (manager, cols_path, _entries, metrics) = seeded(&dir).await;
        let (wal, _) = WalHandle::open(&cols_path, crate::types::WalWriteMode::Sync).unwrap();
        let wal = Arc::new(wal);
        let items = vec![CollectionWalItem {
            id: 2,
            values: vec![0.0, 1.0, 0.0, 0.0],
            label: Some("b".to_string()),
            payload: Some(
                serde_json::json!({"color": "blue"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        }];
        let payload =
            houyi_persistence::collection_wal::encode_collection_upsert_batch("docs", 4, &items)
                .unwrap();
        let appended = wal.append(payload).unwrap();
        assert_eq!(appended.lsn, 2);

        let outcome =
            write_collection_snapshots(&manager, &wal, &cols_path, None, &metrics, true).await;
        let manifest = outcome.manifest.expect("manifest must commit");
        assert_eq!(
            manifest.collections[0].checkpoint_lsn, 1,
            "checkpoint must stop at the last applied record"
        );
        drop(wal);

        let (_wal, all_entries) = FileWal::open(&cols_path).unwrap();
        let fresh = Arc::new(CollectionManager::new(4, "flat"));
        let loaded = load_collection_snapshots(&fresh, &cols_path, 0, None, &metrics)
            .await
            .unwrap()
            .expect("manifest must load");
        let stats = crate::recovery::replay_collections_wal_into_manager_with_mode(
            &all_entries,
            &fresh,
            &loaded.checkpoints,
            true,
        )
        .await
        .unwrap();
        assert_eq!(stats.records_skipped_snapshot, 2);
        assert_eq!(fresh.get("docs").unwrap().applied_lsn(), 2);
        let fresh_col = fresh.get("docs").unwrap();
        let payloads = fresh_col.scalars.lock().unwrap().payloads_for(&[1, 2]);
        assert_eq!(payloads.len(), 2, "unapplied record must be replayed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 构造一个只用于保留测试的 manifest；不要求快照文件存在。
    /// EN: Build a retention-only manifest; snapshot files need not exist.
    fn write_retention_manifest(path: &Path, floor: u64) {
        let entry = CollectionManifestEntry {
            name: "dummy".to_string(),
            checkpoint_lsn: floor,
            dims: 4,
            index_type: "flat".to_string(),
            ha_level: 0,
            metric: Some(1),
            vector_kind: 2,
            file: snapshot_file_name("dummy", floor, 1, 1),
            file_len: 0,
            payload_crc32: 0,
            vector_count: 0,
            scalar_rows: 0,
        };
        let manifest = CollectionManifest {
            format_version: 1,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: floor,
            safe_checkpoint_lsn: floor,
            collections: vec![entry],
        };
        let dir = collection_snapshots_dir(path);
        std::fs::create_dir_all(&dir).unwrap();
        let manifest_path = dir.join("collections.manifest");
        houyi_persistence::collection_snapshot_file::write_collection_manifest_atomic(
            &manifest_path,
            &manifest,
        )
        .unwrap();
    }

    /// P5.0.3：只删整段、活动文件保留、first_available 推进且不超过 max_segments。
    /// EN: P5.0.3 — delete whole segments only, keep the active file, advance
    /// first_available, and stop at max_segments.
    #[test]
    fn retention_deletes_whole_segments_up_to_floor() {
        let dir = temp_dir("retention-delete");
        let path = dir.join("collections.wal");
        let (mut wal, _) = FileWal::open(&path).unwrap();
        wal.set_segment_bytes(128);
        for i in 0..16 {
            wal.append_no_flush(vec![i as u8; 40]).unwrap();
        }
        wal.sync_data().unwrap();
        let before = wal.segment_stats();
        assert!(before.archives.len() > 3, "need multiple archives");
        let floor = before.archives.last().unwrap().last_lsn;
        let expected_deleted = (before.archives.len() - 2).min(DEFAULT_MAX_DELETES_PER_RUN);
        drop(wal);
        write_retention_manifest(&path, floor);

        let handle = WalHandle::open(&path, crate::types::WalWriteMode::Sync)
            .unwrap()
            .0;
        let role = RoleState::new(ReplicationRole::Disabled);
        let lifecycle = CollectionLifecycleState::new(2, DEFAULT_MAX_DELETES_PER_RUN);
        let report = run_retention(&handle, &path, &role, &lifecycle);
        assert_eq!(report.blocked_reason, None);
        assert_eq!(report.deleted_segments, expected_deleted);
        assert!(report.first_available_lsn > before.first_available_lsn);
        assert!(path.exists(), "active file must survive retention");
        let after = handle.segment_stats().unwrap();
        assert_eq!(
            after.archives.len(),
            before.archives.len() - expected_deleted
        );
        assert_eq!(
            lifecycle
                .metrics
                .segments_deleted_total
                .load(Ordering::Relaxed),
            expected_deleted as u64
        );
        assert_eq!(
            lifecycle
                .metrics
                .last_safe_floor_lsn
                .load(Ordering::Relaxed),
            floor
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P5.0.4：leader 的保留 floor 取 manifest 与活跃 follower 的最小值。
    /// EN: P5.0.4 — the leader's retention floor is the minimum of the manifest
    /// checkpoint and the active follower floor.
    #[test]
    fn leader_retention_uses_follower_floor() {
        let dir = temp_dir("retention-leader-floor");
        let path = dir.join("collections.wal");
        let (mut wal, _) = FileWal::open(&path).unwrap();
        wal.set_segment_bytes(128);
        for i in 0..16 {
            wal.append_no_flush(vec![i as u8; 40]).unwrap();
        }
        wal.sync_data().unwrap();
        let before = wal.segment_stats();
        assert!(before.archives.len() > 3);
        let manifest_floor = before.archives.last().unwrap().last_lsn;
        let follower_floor = before.archives[1].last_lsn;
        drop(wal);
        write_retention_manifest(&path, manifest_floor);

        let handle = WalHandle::open(&path, crate::types::WalWriteMode::Sync)
            .unwrap()
            .0;
        let leader = RoleState::new(ReplicationRole::Leader);
        let lifecycle = CollectionLifecycleState::new(2, DEFAULT_MAX_DELETES_PER_RUN);
        let report = run_retention_with_follower_floor(
            &handle,
            &path,
            &leader,
            &lifecycle,
            Some(follower_floor),
        );
        assert_eq!(report.safe_floor_lsn, Some(follower_floor));
        assert_eq!(report.deleted_segments, 2);
        assert!(report.first_available_lsn > before.first_available_lsn);
        let after = handle.segment_stats().unwrap();
        assert_eq!(after.archives.len(), before.archives.len() - 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P5.0.3：leader 在 P5.0.4 前、租约有效、无 manifest 都必须阻塞。
    /// EN: P5.0.3 — leader before P5.0.4, an active lease, and a missing
    /// manifest must all block retention.
    #[test]
    fn retention_blocks_leader_lease_and_missing_manifest() {
        let dir = temp_dir("retention-blocked");
        let path = dir.join("collections.wal");
        let (wal, _) = FileWal::open(&path).unwrap();
        drop(wal);
        let handle = WalHandle::open(&path, crate::types::WalWriteMode::Sync)
            .unwrap()
            .0;
        let lifecycle = CollectionLifecycleState::new(2, DEFAULT_MAX_DELETES_PER_RUN);

        // 无 manifest。
        // EN: No manifest.
        let disabled = RoleState::new(ReplicationRole::Disabled);
        let report = run_retention(&handle, &path, &disabled, &lifecycle);
        assert_eq!(report.blocked_reason.as_deref(), Some("no_manifest"));

        // 有 manifest 后：租约优先于 leader 检查。
        // EN: With a manifest, the lease takes precedence over the leader check.
        write_retention_manifest(&path, 1);
        lifecycle.hold_retention(3600);
        let leader = RoleState::new(ReplicationRole::Leader);
        let report = run_retention(&handle, &path, &leader, &lifecycle);
        assert!(
            report
                .blocked_reason
                .as_deref()
                .is_some_and(|r| r.starts_with("lease_active_until:")),
            "got: {:?}",
            report.blocked_reason
        );
        lifecycle.hold_retention(0);
        let report = run_retention(&handle, &path, &leader, &lifecycle);
        assert_eq!(
            report.blocked_reason.as_deref(),
            Some("leader_retention_pending_no_follower_floor")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
