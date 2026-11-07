//! 集群控制面原语（EE 前置，CE 非付费能力）：持久化 term 与 leader 租约。
//! Cluster control primitives (EE prerequisite, non-paywalled CE): persisted
//! term and leader lease.
//!
//! 为什么放在 CE：没有 term/租约的自动选主无法阻止被分区的旧 leader 继续
//! 写入——那会直接丢数据。基础 HA 安全原语不设付费墙；sentinel 选主服务本身
//! 在企业版（见 `docs/OPEN_SOURCE_PLAN.md`）。
//! EN: Without term/lease, automatic failover cannot stop a partitioned old
//! leader from writing — that is data loss. Base HA safety primitives are not
//! paywalled; the sentinel election service itself lives in EE.
//!
//! 默认 `require_lease=false` 时行为与手动 promote 完全一致；开启后 leader
//! 只有持有效租约才可写，租约过期由调用方统一降为只读。
//! EN: With `require_lease=false` behavior matches manual promote; when enabled,
//! a leader may write only while its lease is valid, and callers flip it to
//! read-only once the lease expires.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// 租约 TTL 上限（毫秒）：超过上限的请求被拒绝，限制时钟偏斜/配置错误的影响面。
/// EN: Lease TTL cap; larger requests are rejected to bound clock-skew/config risk.
pub const MAX_LEASE_TTL_MS: u64 = 60_000;

/// 持久化的集群状态（term 只增）。
/// EN: Persisted cluster state; the term only increases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedClusterState {
    pub term: u64,
    pub max_seen_term: u64,
}

/// term 持久化存储：数据目录下的 `cluster_state.json`，原子写 + 父目录 fsync。
/// EN: Term store at `<data_dir>/cluster_state.json`, atomic write + parent fsync.
#[derive(Debug)]
pub struct ClusterStateStore {
    path: PathBuf,
    state: Mutex<PersistedClusterState>,
}

impl ClusterStateStore {
    /// 打开或初始化（文件缺失 = 全新部署，term 0；损坏 = fail-closed）。
    /// EN: Open or initialize (missing = fresh deploy, term 0; corrupt = fail closed).
    pub fn open(wal_path: &Path) -> io::Result<Self> {
        // `Path::new("data.wal").parent()` 是 `Some("")` 而不是 None：
        // 空父目录必须归一为 "."，否则目录 fsync 会打开 "" 报 ENOENT。
        // EN: `Path::new("data.wal").parent()` is `Some("")`, not None — an
        // empty parent must normalize to "." or the fsync opens "" and fails.
        let dir = houyi_persistence::parent_dir_or_dot(wal_path);
        let path = dir.join("cluster_state.json");
        let state = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("cluster_state {} is invalid: {e}", path.display()),
                )
            })?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => PersistedClusterState::default(),
            Err(e) => return Err(e),
        };
        Ok(Self {
            path,
            state: Mutex::new(state),
        })
    }

    pub fn term(&self) -> u64 {
        self.state.lock().unwrap().term
    }

    /// 观察一个 term：只增不减；前进时先落盘再返回。
    /// EN: Observe a term (monotonic); persist before returning on advancement.
    pub fn observe_term(&self, term: u64) -> io::Result<u64> {
        self.observe_term_and_get(term)
    }

    /// 原子「观察并返回生效 term」：term 检查与前进在同一把锁内完成，
    /// 调用方据此拒绝低 term 的 grant（check-then-act 会把更高 term 覆盖掉）。
    /// EN: atomically observe-and-return the effective term; the check and the
    /// advance share one lock so a lower-term grant cannot overwrite a higher one.
    pub fn observe_term_and_get(&self, term: u64) -> io::Result<u64> {
        let mut state = self.state.lock().unwrap();
        if term <= state.term {
            state.max_seen_term = state.max_seen_term.max(term);
            return Ok(state.term);
        }
        let next = PersistedClusterState {
            term,
            max_seen_term: state.max_seen_term.max(term),
        };
        self.persist_locked(&next)?;
        *state = next;
        Ok(term)
    }

    fn persist_locked(&self, state: &PersistedClusterState) -> io::Result<()> {
        let parent = houyi_persistence::parent_dir_or_dot(&self.path);
        std::fs::create_dir_all(parent)?;
        let tmp = self.path.with_extension("json.tmp");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp)?;
            f.write_all(serde_json::to_vec(state)?.as_slice())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        #[cfg(unix)]
        {
            File::open(parent)?.sync_all()?;
        }
        Ok(())
    }
}

/// 租约错误。 / Lease errors.
///
/// 手写 Display/Error（不引入 thiserror）：本 crate 目前无该依赖，一处错误
/// 枚举不值得新增供应链面。
/// EN: hand-written Display/Error instead of pulling in thiserror for one enum.
#[derive(Debug)]
pub enum LeaseError {
    TermRegression { current: u64, requested: u64 },
    InvalidTerm,
    InvalidTtl { ttl_ms: u64 },
    Missing,
    Expired,
    Io(io::Error),
}

impl std::fmt::Display for LeaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TermRegression { current, requested } => write!(
                f,
                "term regression: current {current}, requested {requested}"
            ),
            Self::InvalidTerm => write!(f, "lease term must be >= 1 (0 means no lease)"),
            Self::InvalidTtl { ttl_ms } => write!(
                f,
                "lease ttl {ttl_ms}ms is outside 1..={MAX_LEASE_TTL_MS}ms"
            ),
            Self::Missing => write!(f, "leader lease is missing"),
            Self::Expired => write!(f, "leader lease expired"),
            Self::Io(e) => write!(f, "cluster state io: {e}"),
        }
    }
}

impl std::error::Error for LeaseError {}

impl From<io::Error> for LeaseError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// 租约授予结果。 EN: Lease grant result.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct LeaseGrant {
    pub term: u64,
    pub ttl_ms: u64,
    pub expires_in_ms: u64,
}

/// 租约状态视图。 EN: Lease status view.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct LeaseStatus {
    pub term: u64,
    pub active: bool,
    pub expires_in_ms: u64,
    pub required: bool,
}

/// 单调时钟租约：deadline 相对 `started`，不依赖墙钟。
/// EN: Monotonic lease; deadlines are relative to `started`, not wall time.
pub struct LeaderLease {
    enabled: bool,
    started: Instant,
    term: AtomicU64,
    deadline_ms: AtomicU64,
    /// grant/revoke 的短临界区：term 与 deadline 必须作为一个整体更新，
    /// 仅靠 term CAS 无法给 deadline 写入排序（低 term grant 的 deadline
    /// 可能覆盖高 term grant 的）。低频管理路径，允许短 Mutex。
    /// EN: short critical section for grant/revoke — term and deadline must
    /// move as one unit; a term CAS alone cannot order the deadline write.
    /// This is a low-frequency admin path, so a short Mutex is fine.
    state_lock: Mutex<()>,
}

impl LeaderLease {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            started: Instant::now(),
            term: AtomicU64::new(0),
            deadline_ms: AtomicU64::new(0),
            state_lock: Mutex::new(()),
        }
    }

    pub fn term(&self) -> u64 {
        self.term.load(Ordering::Acquire)
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis().min(u64::MAX as u128) as u64
    }

    /// 授予：仅当 `term` 不低于当前租约 term 时生效；返回 None 表示已有
    /// 更高 term 的租约（调用方报 TermRegression）。读-判-写全程持有
    /// `state_lock` 短临界区，因此 term 与 deadline 作为整体推进；单靠
    /// store 侧的原子 observe 不够——两个并发 grant 在 observe 与 lease
    /// 更新之间仍可能交错。
    /// EN: grant — applies only when `term` is not below the current lease
    /// term; None means a higher-term lease already exists. The whole
    /// read-compare-write runs under the short `state_lock` critical section,
    /// so term and deadline advance as one unit. The store-side atomic observe
    /// alone is insufficient: concurrent grants can still interleave between
    /// it and the lease update.
    pub fn grant(&self, term: u64, ttl_ms: u64) -> Option<LeaseGrant> {
        let _guard = self
            .state_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.term.load(Ordering::Acquire);
        if term < current {
            return None;
        }
        let now = self.now_ms();
        self.term.store(term, Ordering::Release);
        self.deadline_ms
            .store(now.saturating_add(ttl_ms), Ordering::Release);
        Some(LeaseGrant {
            term,
            ttl_ms,
            expires_in_ms: ttl_ms,
        })
    }

    pub fn revoke(&self) {
        let _guard = self
            .state_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.deadline_ms.store(0, Ordering::Release);
    }

    pub fn status(&self) -> LeaseStatus {
        let now = self.now_ms();
        let guard = self
            .state_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // term 与 deadline 必须在同一临界区内成对读取：先取 deadline 再在锁外
        // 读 term 会拼出一个从未存在过的快照（新 term + 旧 deadline），把刚
        // 续租的租约误报成 active=false。
        // EN: read term and deadline as a pair inside the critical section. A
        // deadline load under the lock followed by a term load after dropping
        // it can splice a snapshot that never existed (new term + old
        // deadline) and misreport a just-renewed lease as active=false.
        let term = self.term.load(Ordering::Acquire);
        let deadline = self.deadline_ms.load(Ordering::Acquire);
        drop(guard);
        let remaining = deadline.saturating_sub(now);
        // active 描述「租约本身是否有效」（term>0 且未过期）；是否强制由
        // required 单独表达。未开启强制时授予的租约依然应显示 active=true，
        // 否则控制面会误报「授了租却没生效」。
        // EN: active describes the lease itself (term>0 and unexpired); whether
        // it is enforced is `required`. A lease granted while enforcement is
        // off must still report active, or the control plane misreports it.
        LeaseStatus {
            term,
            active: term > 0 && deadline > now,
            expires_in_ms: remaining,
            required: self.enabled,
        }
    }

    /// 写前检查：未启用租约时永远放行（手动模式兼容）。
    /// EN: Pre-write check; always allowed when leases are disabled.
    pub fn check_writable(&self) -> Result<(), LeaseError> {
        if !self.enabled {
            return Ok(());
        }
        // 写路径不加 state_lock（工程禁令：热路径优先无锁）。先读 term 再读
        // deadline 是 fail-safe 顺序：要返回 Ok，deadline 必须非零且未过期，
        // 而 grant 在锁内先写 term 后写 deadline——读到有效 deadline 时租约
        // 至少已授予到该 deadline；期间发生 grant 最多导致一次瞬时拒绝，绝不
        // 会把从未授予的租约放行。revoke 与检查之间的固有竞态由 fencing 窗口
        // 兜底（撤销方先翻角色/过期，检查只是最后一道闸门）。
        // EN: the write path takes no lock (hot-path lock-free rule). Reading
        // term before deadline is fail-safe: Ok requires a non-zero unexpired
        // deadline, and a grant stores term before deadline inside the lock, so
        // an observed valid deadline means the lease already covered it. A
        // concurrent grant can only cause a transient rejection, never admit a
        // lease that was never granted. The inherent revoke-vs-check race is
        // covered by the fencing window (the revoker flips role/expiry first;
        // this check is the last gate).
        let term = self.term();
        let deadline = self.deadline_ms.load(Ordering::Acquire);
        if term == 0 || deadline == 0 {
            return Err(LeaseError::Missing);
        }
        if self.now_ms() >= deadline {
            return Err(LeaseError::Expired);
        }
        Ok(())
    }
}

/// term 存储 + 租约的组合入口。 EN: Combined term store + lease control.
pub struct ClusterControl {
    store: ClusterStateStore,
    lease: LeaderLease,
    max_ttl_ms: u64,
}

impl ClusterControl {
    pub fn open(wal_path: &Path, require_lease: bool) -> io::Result<Self> {
        Self::open_with_max_ttl(wal_path, require_lease, MAX_LEASE_TTL_MS)
    }

    /// 自定义 TTL 上限（配置 `leader_lease_max_ttl_ms`）；仍被硬上限截断——
    /// 运维无法通过配置把时钟偏斜风险放大到硬上限之外。
    /// EN: custom TTL cap; still clamped by the hard cap so configuration can
    /// never widen the clock-skew exposure beyond it.
    pub fn open_with_max_ttl(
        wal_path: &Path,
        require_lease: bool,
        max_ttl_ms: u64,
    ) -> io::Result<Self> {
        Ok(Self {
            store: ClusterStateStore::open(wal_path)?,
            lease: LeaderLease::new(require_lease),
            max_ttl_ms: max_ttl_ms.clamp(1, MAX_LEASE_TTL_MS),
        })
    }

    /// 是否要求有效租约才可写。 EN: Whether a valid lease is required for writes.
    pub fn lease_required(&self) -> bool {
        self.lease.status().required
    }

    pub fn max_ttl_ms(&self) -> u64 {
        self.max_ttl_ms
    }

    pub fn term(&self) -> u64 {
        self.store.term()
    }

    pub fn observe_term(&self, term: u64) -> Result<u64, LeaseError> {
        Ok(self.store.observe_term(term)?)
    }

    /// 无落盘的前置校验（promote 在拆除 follower 之前先调用它）。
    /// EN: pre-validation without persistence (promote calls it before tearing
    /// down the follower).
    pub fn validate_grant(&self, term: u64, ttl_ms: u64) -> Result<(), LeaseError> {
        if ttl_ms == 0 || ttl_ms > self.max_ttl_ms {
            return Err(LeaseError::InvalidTtl { ttl_ms });
        }
        if term == 0 {
            return Err(LeaseError::InvalidTerm);
        }
        let current = self.store.term();
        if term < current {
            return Err(LeaseError::TermRegression {
                current,
                requested: term,
            });
        }
        Ok(())
    }

    pub fn grant(&self, term: u64, ttl_ms: u64) -> Result<LeaseGrant, LeaseError> {
        self.validate_grant(term, ttl_ms)?;
        // 原子观察：若并发 grant 已把 term 推进到更高值，本请求必须以
        // TermRegression 失败，绝不能把租约降回低 term。
        // EN: atomic observe — if a concurrent grant advanced the term, this
        // request must fail with TermRegression instead of lowering the lease.
        let effective = self.store.observe_term_and_get(term)?;
        if effective > term {
            return Err(LeaseError::TermRegression {
                current: effective,
                requested: term,
            });
        }
        self.lease
            .grant(term, ttl_ms)
            .ok_or(LeaseError::TermRegression {
                current: self.store.term().max(self.lease.term()),
                requested: term,
            })
    }

    pub fn revoke(&self) {
        self.lease.revoke();
    }

    /// demote 的 fencing 入口：先观察/校验 term（只增），再撤销租约。
    /// term 回退返回错误且**不**撤销——调用方拿到 409 时应视为陈旧请求。
    /// EN: demote fencing — observe/validate the term (monotonic), then revoke.
    /// A regression returns an error and leaves the lease untouched so a stale
    /// request cannot demote the current leader.
    pub fn fence_and_revoke(&self, term: Option<u64>) -> Result<u64, LeaseError> {
        if let Some(requested) = term {
            // 同一把锁内完成「比较 + 前进」，避免并发 grant 推动 term 后
            // 陈旧 demote 仍然撤销当前租约。
            // EN: compare and advance under one lock so a stale demote cannot
            // revoke a lease after a concurrent grant advanced the term.
            let effective = self.store.observe_term_and_get(requested)?;
            if effective > requested {
                return Err(LeaseError::TermRegression {
                    current: effective,
                    requested,
                });
            }
        }
        self.lease.revoke();
        Ok(self.store.term())
    }

    pub fn status(&self) -> LeaseStatus {
        self.lease.status()
    }

    pub fn check_writable(&self) -> Result<(), LeaseError> {
        self.lease.check_writable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// 每个测试独占一个目录：`cluster_state.json` 写在 WAL 父目录，共享
    /// temp_dir 会让并行测试互相覆盖彼此的状态（term 断言随机失败）。
    /// EN: each test gets its own directory — cluster_state.json lives beside
    /// the WAL, and a shared temp_dir makes parallel tests clobber each other.
    fn temp_wal() -> PathBuf {
        // 时间戳可能在同一纳秒内重复；加原子序号保证每个测试独占目录。
        // EN: timestamps can collide within a nanosecond; a counter guarantees
        // a unique directory per test.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "houyi-cluster-state-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("test.wal")
    }

    fn cleanup(wal: &Path) {
        if let Some(dir) = wal.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn term_is_monotonic_and_persisted() {
        let wal = temp_wal();
        {
            let store = ClusterStateStore::open(&wal).unwrap();
            assert_eq!(store.term(), 0);
            assert_eq!(store.observe_term(3).unwrap(), 3);
            assert_eq!(store.observe_term(2).unwrap(), 3);
            assert_eq!(store.observe_term(5).unwrap(), 5);
        }
        let reopened = ClusterStateStore::open(&wal).unwrap();
        assert_eq!(reopened.term(), 5);
        cleanup(&wal);
    }

    #[test]
    fn corrupt_state_is_fatal() {
        let wal = temp_wal();
        let path = wal.parent().unwrap().join("cluster_state.json");
        std::fs::write(&path, b"{not json").unwrap();
        let err = ClusterStateStore::open(&wal).unwrap_err();
        assert!(err.to_string().contains("is invalid"));
        cleanup(&wal);
    }

    #[test]
    fn lease_rejects_regression_and_expires() {
        let wal = temp_wal();
        let control = ClusterControl::open(&wal, true).unwrap();
        assert!(matches!(control.check_writable(), Err(LeaseError::Missing)));
        control.grant(2, 10).unwrap();
        control.check_writable().unwrap();
        assert!(matches!(
            control.grant(1, 10),
            Err(LeaseError::TermRegression { .. })
        ));
        std::thread::sleep(Duration::from_millis(20));
        assert!(matches!(control.check_writable(), Err(LeaseError::Expired)));
        control.revoke();
        assert!(matches!(control.check_writable(), Err(LeaseError::Missing)));
        cleanup(&wal);
    }

    #[test]
    fn disabled_lease_is_always_writable() {
        let wal = temp_wal();
        let control = ClusterControl::open(&wal, false).unwrap();
        let lease = control.grant(1, 1_000).unwrap();
        assert_eq!(lease.term, 1);
        assert!(
            control.status().active,
            "a granted lease is active even when not enforced"
        );
        std::thread::sleep(Duration::from_millis(5));
        control.check_writable().unwrap();
        assert!(!control.status().required);
        cleanup(&wal);
    }

    #[test]
    fn term_zero_is_rejected() {
        let wal = temp_wal();
        let control = ClusterControl::open(&wal, true).unwrap();
        assert!(matches!(
            control.grant(0, 1_000),
            Err(LeaseError::InvalidTerm)
        ));
        cleanup(&wal);
    }

    /// 并发 grant 不得把租约 term 降回低值：store 侧原子 observe 之后、
    /// lease 更新之前仍可能交错，租约自身必须在 `state_lock` 短临界区内保持
    /// term/deadline 单调。
    /// EN: concurrent grants must never lower the lease term; the store-side
    /// atomic observe alone is not enough — the lease's term/deadline pair must
    /// stay monotonic inside the short `state_lock` critical section.
    #[test]
    fn concurrent_grants_never_lower_the_lease_term() {
        for _ in 0..50 {
            let wal = temp_wal();
            let control = Arc::new(ClusterControl::open(&wal, true).unwrap());
            let low = Arc::clone(&control);
            let high = Arc::clone(&control);
            let a = std::thread::spawn(move || {
                let _ = low.grant(5, 50);
            });
            let b = std::thread::spawn(move || {
                let _ = high.grant(6, 50);
            });
            a.join().unwrap();
            b.join().unwrap();
            assert_eq!(control.status().term, 6);
            assert_eq!(control.term(), 6);
            cleanup(&wal);
        }
    }

    #[test]
    fn ttl_is_bounded() {
        let wal = temp_wal();
        let control = ClusterControl::open(&wal, true).unwrap();
        assert!(matches!(
            control.grant(1, 0),
            Err(LeaseError::InvalidTtl { .. })
        ));
        assert!(matches!(
            control.grant(1, MAX_LEASE_TTL_MS + 1),
            Err(LeaseError::InvalidTtl { .. })
        ));
        cleanup(&wal);
    }
}
