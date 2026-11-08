//! Shared types for houyi-server: config, WAL handle, pgwire state, HTTP request/response.
//! 共享类型：配置、WAL 句柄、pgwire 状态、HTTP 请求/响应结构体。

use crate::collection::CollectionManager;
use crate::index_backend::IndexBackend;
use houyi_persistence::wal_async::WalAsyncHandle;
use houyi_persistence::FileWal;
#[allow(unused_imports)]
use houyi_persistence::Wal;
use houyi_vector::FlushPolicy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

// ── WAL 写入模式 ─────────────────────────────────────────────
// EN: WAL write mode.

/// WAL 写入模式：仅 **进程启动时** 读取；运行中切换需重启。
/// EN: WAL write mode: read only at process startup; runtime switching needs a restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WalWriteMode {
    #[default]
    Async,
    Sync,
}

pub fn parse_wal_write_mode(raw: &str) -> Result<WalWriteMode, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "async" => Ok(WalWriteMode::Async),
        "sync" => Ok(WalWriteMode::Sync),
        _ => Err(format!(
            "invalid wal write mode {raw:?}, expected async | sync"
        )),
    }
}

// ── WAL 句柄 ────────────────────────────────────────────────
// EN: WAL handle.

pub enum WalHandle {
    Async(WalAsyncHandle),
    Sync(Mutex<FileWal>),
}

impl WalHandle {
    pub fn open(
        path: &std::path::Path,
        mode: WalWriteMode,
    ) -> std::io::Result<(Self, Vec<houyi_persistence::WalEntry>)> {
        Self::open_with_ship(path, mode, None)
    }

    /// `open` + 复制扇出（P4.4.1）：Async 模式把 `ShipHub` 接进 WAL 写线程
    /// （flush 成功的批 publish 给全部 follower session）。Sync 模式无扇出
    /// 通道——role≠disabled 的装配在配置校验期已 fast-fail，此处防御性报错。
    /// EN: `open` plus a replication fan-out hub wired into the async WAL
    /// writer thread. Sync mode has no fan-out path; role≠disabled + sync
    /// was already rejected during config validation, so this is defensive.
    pub fn open_with_ship(
        path: &std::path::Path,
        mode: WalWriteMode,
        ship: Option<Arc<houyi_persistence::replication_leader::ShipHub>>,
    ) -> std::io::Result<(Self, Vec<houyi_persistence::WalEntry>)> {
        Self::open_with_ship_and_encryption(path, mode, ship, None)
    }

    /// `open_with_ship` + 静态加密（P4.7.3）：两种写模式均支持。
    /// EN: shipping plus at-rest encryption, supported in both write modes.
    pub fn open_with_ship_and_encryption(
        path: &std::path::Path,
        mode: WalWriteMode,
        ship: Option<Arc<houyi_persistence::replication_leader::ShipHub>>,
        encryption: Option<Arc<houyi_persistence::encryption::Encryption>>,
    ) -> std::io::Result<(Self, Vec<houyi_persistence::WalEntry>)> {
        Self::open_with_config(
            path,
            mode,
            ship,
            encryption,
            houyi_persistence::wal_async::WalOpenConfig::default(),
        )
    }

    /// `open_with_ship_and_encryption` + 分段配置（P5.0.2）。
    /// EN: shipping/encryption plus segment config (P5.0.2).
    pub fn open_with_config(
        path: &std::path::Path,
        mode: WalWriteMode,
        ship: Option<Arc<houyi_persistence::replication_leader::ShipHub>>,
        encryption: Option<Arc<houyi_persistence::encryption::Encryption>>,
        config: houyi_persistence::wal_async::WalOpenConfig,
    ) -> std::io::Result<(Self, Vec<houyi_persistence::WalEntry>)> {
        match mode {
            WalWriteMode::Async => {
                let (h, entries) =
                    WalAsyncHandle::open_with_config(path, ship, encryption, config)?;
                Ok((WalHandle::Async(h), entries))
            }
            WalWriteMode::Sync => {
                if ship.is_some() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "replication requires wal_write_mode=async",
                    ));
                }
                let (mut w, entries) = FileWal::open_with_encryption(path, encryption)?;
                w.set_segment_bytes(config.segment_bytes);
                Ok((WalHandle::Sync(Mutex::new(w)), entries))
            }
        }
    }

    /// 复制 follower 专用：取底层异步句柄（`append_entry`/`fsync` 仅异步
    /// 模式有）。Sync 模式返回 None——role=follower + sync 已在配置校验
    /// fast-fail，此处 None 只会出现在测试装配。
    /// EN: Clone the underlying async handle for the replication follower
    /// (explicit-LSN append / fsync exist only there). None in Sync mode.
    pub fn async_handle(&self) -> Option<WalAsyncHandle> {
        match self {
            WalHandle::Async(h) => Some(h.clone()),
            WalHandle::Sync(_) => None,
        }
    }

    pub fn append(&self, payload: Vec<u8>) -> std::io::Result<houyi_persistence::WalEntry> {
        match self {
            WalHandle::Async(h) => h.append(payload),
            WalHandle::Sync(m) => Ok(m.lock().expect("wal mutex poisoned").append(payload)),
        }
    }

    pub fn truncate_prefix_through(&self, through: u64) -> std::io::Result<usize> {
        match self {
            WalHandle::Async(h) => h.truncate_prefix_through(through),
            WalHandle::Sync(m) => m
                .lock()
                .expect("wal mutex poisoned")
                .truncate_prefix_through(through),
        }
    }

    pub fn rotate_archive(&self) -> std::io::Result<Option<PathBuf>> {
        match self {
            WalHandle::Async(h) => h.rotate_archive(),
            WalHandle::Sync(m) => m.lock().expect("wal mutex poisoned").rotate_archive(),
        }
    }

    pub fn next_lsn(&self) -> u64 {
        match self {
            WalHandle::Async(h) => h.next_lsn(),
            WalHandle::Sync(m) => m.lock().expect("wal mutex poisoned").next_lsn(),
        }
    }

    pub fn active_file_len(&self) -> std::io::Result<u64> {
        match self {
            WalHandle::Async(h) => h.active_file_len(),
            WalHandle::Sync(m) => m.lock().expect("wal mutex poisoned").active_file_len(),
        }
    }

    pub fn archived_segment_count(&self) -> std::io::Result<usize> {
        match self {
            WalHandle::Async(h) => h.archived_segment_count(),
            WalHandle::Sync(m) => m
                .lock()
                .expect("wal mutex poisoned")
                .archived_segment_count(),
        }
    }

    /// 最早可读 LSN（P5.0.2）。
    /// EN: Earliest readable LSN (P5.0.2).
    pub fn first_available_lsn(&self) -> std::io::Result<u64> {
        match self {
            WalHandle::Async(h) => h.first_available_lsn(),
            WalHandle::Sync(m) => Ok(m.lock().expect("wal mutex poisoned").first_available_lsn()),
        }
    }

    /// 全量本地 WAL 条目（promote 尾部补齐，P5.0.4）。
    /// EN: Every local WAL entry (promote tail replay, P5.0.4).
    pub fn replay_entries(&self) -> std::io::Result<Vec<houyi_persistence::WalEntry>> {
        match self {
            WalHandle::Async(h) => h.replay_entries(),
            WalHandle::Sync(m) => m.lock().expect("wal mutex poisoned").replay_all(),
        }
    }

    /// 段统计（P5.0.2）。
    /// EN: Segment statistics (P5.0.2).
    pub fn segment_stats(&self) -> std::io::Result<houyi_persistence::SegmentStats> {
        match self {
            WalHandle::Async(h) => h.segment_stats(),
            WalHandle::Sync(m) => Ok(m.lock().expect("wal mutex poisoned").segment_stats()),
        }
    }

    /// 按整段保留（P5.0.2）。
    /// EN: Whole-segment retention (P5.0.2).
    pub fn retain_segments(
        &self,
        safe_floor: u64,
        max_segments: usize,
        max_deletes: usize,
    ) -> std::io::Result<houyi_persistence::RetentionOutcome> {
        match self {
            WalHandle::Async(h) => h.retain_segments(safe_floor, max_segments, max_deletes),
            WalHandle::Sync(m) => m.lock().expect("wal mutex poisoned").retain_segments(
                safe_floor,
                max_segments,
                max_deletes,
            ),
        }
    }

    /// 优雅关停（P4.0.3）：Async 模式排空写线程队列并 join（落盘保证）；
    /// Sync 模式逐条已落盘，无需动作。
    /// Graceful shutdown: Async drains and joins the writer thread; Sync flushes
    /// per record already, so it's a no-op.
    pub fn shutdown(&self) {
        match self {
            WalHandle::Async(h) => h.shutdown(),
            WalHandle::Sync(_) => {}
        }
    }
}

// ── 复制角色（P4.4.1） ───────────────────────────────────────
// EN: replication role (P4.4.1).

/// 复制角色（P4.4.1）：disabled = 单机（默认）；leader = 接受 follower 连接并
/// 扇出 collections.wal；follower = 只读副本，从 leader 流式接收并实时 apply。
/// EN: Replication role — disabled (standalone, default), leader (streams
/// collections.wal to followers), follower (read-only replica applying the stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplicationRole {
    #[default]
    Disabled = 0,
    Leader = 1,
    Follower = 2,
}

impl ReplicationRole {
    /// /metrics `houyidb_replication_role` gauge 数值。
    /// EN: Numeric code for the metrics gauge.
    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Leader => "leader",
            Self::Follower => "follower",
        }
    }
}

pub fn parse_replication_role(raw: &str) -> Result<ReplicationRole, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "disabled" => Ok(ReplicationRole::Disabled),
        "leader" => Ok(ReplicationRole::Leader),
        "follower" => Ok(ReplicationRole::Follower),
        _ => Err(format!(
            "invalid replication role {raw:?}, expected disabled | leader | follower"
        )),
    }
}

/// 运行态角色（HTTP 只读拦截 / pgwire 25006 / 指标共用一份；P4.4.3 promote
/// 需要运行中切换，故用原子量而非启动期固化）。
/// EN: Runtime role shared by the HTTP read-only guard, the pgwire 25006
/// rejection, and /metrics — atomic because promote (P4.4.3) flips it live.
#[derive(Clone, Default)]
pub struct RoleState(Arc<AtomicU64>);

impl RoleState {
    pub fn new(role: ReplicationRole) -> Self {
        Self(Arc::new(AtomicU64::new(role.code() as u64)))
    }

    pub fn get(&self) -> ReplicationRole {
        // Acquire/Release 配对（OCR 16a7ae9[1]）：promote 的顺序不变量
        // 「先发布指标视图、后翻角色」要成立，写路径必须以 Acquire 读角色——
        // 观察到 Leader 即与视图发布形成 happens-before，弱序架构上才不会
        // 出现「写放行但读到旧视图」的静默降级窗口。
        // EN: Acquire/Release pairing (OCR 16a7ae9[1]): the promote ordering
        // EN: invariant "publish metrics view first, flip role second" holds only
        // EN: if the write path reads the role with Acquire — observing Leader
        // EN: then happens-before the view publish, preventing a silent downgrade
        // EN: window (writes admitted but stale view read) on weak memory models.
        match self.0.load(Ordering::Acquire) {
            1 => ReplicationRole::Leader,
            2 => ReplicationRole::Follower,
            _ => ReplicationRole::Disabled,
        }
    }

    pub fn set(&self, role: ReplicationRole) {
        self.0.store(role.code() as u64, Ordering::Release);
    }

    /// follower = 只读副本（写端点 503 / pgwire 25006）。
    /// EN: follower is the read-only role.
    pub fn is_follower(&self) -> bool {
        self.get() == ReplicationRole::Follower
    }
}

/// /metrics 的复制读数视图（P4.4.1）：role≠disabled 时输出对应族；RwLock 为
/// P4.4.3 promote 后换视图预留（抓取是低频路径，读锁开销可忽略）。
/// EN: Replication metrics source for /metrics, emitted only when role≠disabled;
/// the RwLock lets promote (P4.4.3) swap the view (scrapes are infrequent).
pub enum ReplicationMetricsView {
    Leader {
        hub: Arc<houyi_persistence::replication_leader::ShipHub>,
        watermarks: Arc<houyi_persistence::replication_leader::AckWatermarks>,
        leader_next_lsn: Arc<dyn Fn() -> u64 + Send + Sync>,
        /// sync 等待超时（P4.4.2，`replication_sync_timeout_ms`）。挂在视图上
        /// 而非 ServerConfig：写 handler 只经本槽位触达复制态，promote
        /// （P4.4.3）换视图时 sync 等待随角色出现/消失，无需第二处接线。
        /// EN: Sync-wait timeout, carried on the view so write handlers reach
        /// all replication state through this one slot (and promote swaps it
        /// atomically with the role).
        sync_timeout: std::time::Duration,
        /// sync 等待的 follower quorum（0 = 现行「任一 ACK」）。挂在视图上
        /// 与 timeout 同理：promote 换视图时同步生效。
        /// EN: follower quorum for sync waits (0 = legacy "any ACK"), carried
        /// on the view so promote swaps it atomically with the role.
        sync_quorum: usize,
    },
    Follower {
        metrics: Arc<houyi_persistence::replication_follower::FollowerMetrics>,
        /// apply 汇跳过的毒行计数（非有限分量）——sink 持于 follower 线程，
        /// 视图另持同一 Arc 供 /metrics 读取。
        /// EN: Poisoned-row skip counter shared with the apply sink.
        poison_skipped: Arc<std::sync::atomic::AtomicU64>,
    },
}

/// 共享复制指标槽：None = 未启用复制（指标节省略，与 breaker/rss 同纪律）。
/// EN: Shared replication metrics slot; None = replication off (section omitted).
pub type SharedReplicationMetrics = Arc<RwLock<Option<ReplicationMetricsView>>>;

pub fn shared_replication_metrics(
    view: Option<ReplicationMetricsView>,
) -> SharedReplicationMetrics {
    Arc::new(RwLock::new(view))
}

/// leader/follower 运行句柄共享槽（P4.4.3）：main 装配时填充、promote 端点
/// 取 follower 关停并放入新 leader、关停序列统一取出 shutdown——三方经同一
/// 槽位交接，杜绝「promote 后关停序列不知道新 leader 存在」的句柄泄漏。
/// EN: Shared slot for the live replication handles — main fills it at startup,
/// the promote endpoint swaps follower→leader through it, and the shutdown
/// sequence drains it. One slot prevents handle leaks across promote.
#[derive(Default)]
pub struct ReplicationHandles {
    pub leader: Option<houyi_persistence::replication_leader::ReplicationLeader>,
    pub follower: Option<houyi_persistence::replication_follower::ReplicationFollower>,
}

/// std::sync::Mutex：句柄交接是低频管理路径（启动/promote/关停各一次），
/// 且 shutdown 的 join 是阻塞调用，本就不该进 tokio Mutex。
/// EN: A std Mutex suffices — handle handoff is a rare admin path and the
/// shutdown join is blocking anyway.
pub type SharedReplicationHandles = Arc<std::sync::Mutex<ReplicationHandles>>;

/// promote 所需的全部构造参数（P4.4.3）：启动期装配进 HTTP state，promote 时
/// 无需回问 main 的局部变量。
///
/// promote 互斥由 `promote_gate`（tokio Mutex）承担：P4.5.5 起 promote 内部
/// 要 await（关停 follower / 起 leader），std MutexGuard 跨 await 会让整个
/// handler future 变成非 Send。`handles` 的 std 锁只用于同步段交接，不跨 await。
/// EN: `promote_gate` (a tokio Mutex) serializes promotes now that the body
/// awaits; the std `handles` lock is only ever held synchronously.
pub struct PromoteContext {
    pub handles: SharedReplicationHandles,
    /// promote 串行化（可跨 await 持有的异步锁）。
    /// EN: Serializes promotes (async lock held across awaits).
    pub promote_gate: tokio::sync::Mutex<()>,
    /// 扇出枢纽复用：follower 的 WAL 写线程本就注入同一 hub（其 AppendEntry
    /// 批次也过 publish），promote 后 WAL 句柄零切换。
    /// EN: The hub is reused as-is — the follower's WAL writer already
    /// publishes into it, so promote needs no WAL handle swap.
    pub hub: Arc<houyi_persistence::replication_leader::ShipHub>,
    pub listen_addr: String,
    pub collections_wal_path: PathBuf,
    /// next_lsn 闭包来源（leader 视图水位线/滞后计算与新 session 的 WELCOME）。
    /// EN: Source of the next-LSN closure for the leader view and sessions.
    pub collections_wal: Arc<WalHandle>,
    pub queue_batches: usize,
    pub sync_timeout: std::time::Duration,
    /// sync 等待的 follower quorum（0 = 任一 ACK，M1 新增）。
    /// EN: follower quorum for sync waits (0 = any ACK; added in M1).
    pub sync_quorum: usize,
    /// promote 后的 follower 重连宽限期与租约入口（P5.0.4）：新 leader 必须
    /// 先留出窗口，旧 follower 才来得及握手。
    /// EN: reconnect grace and lease handle after promote (P5.0.4); the new
    /// leader must hold retention long enough for old followers to handshake.
    pub lifecycle: Arc<crate::collection_lifecycle::CollectionLifecycleState>,
    pub follower_grace_secs: u64,
    /// 复制认证共享秘密（P4.6.3；OCR 批次二十八 HIGH）：promote 重建
    /// LeaderSessionConfig 时必须沿用启动时的 token，否则新 leader 静默
    /// 变为无认证——任何 TCP 客户端都能拉走全量日志。
    /// EN: the replication secret must survive promote; rebuilding the leader
    /// config without it silently turns the new leader unauthenticated.
    /// 字段名带 replication_ 前缀：本 crate 另有 HTTP/pgwire 的
    /// `ServerConfig.auth_token`，同名会让 `auth_token: cfg.auth_token.clone()`
    /// 这类笔误静默通过编译（OCR 批次二十九）。
    /// EN: prefixed to make mixing it up with the HTTP Bearer token impossible.
    pub replication_auth_token: Option<String>,
    /// 本地 WAL 加密句柄（P4.7.3）：promote 后的新 leader 用它读本机段
    /// （catch-up/指纹）；线上传输仍是明文，由各节点本地加密落盘。
    /// EN: local WAL encryption handle for the promoted leader; the wire stays
    /// plaintext and each node encrypts at rest with its own keyring.
    pub encryption: Option<Arc<houyi_persistence::encryption::Encryption>>,
    /// 运行时 demote→follower 转换所需的句柄（M3）：demote 后旧 leader
    /// 直接以 follower 身份连回新主，无需重启。
    /// EN: handles for runtime demote→follower conversion (M3) so a demoted
    /// leader can rejoin as a follower without a restart.
    pub collections: Arc<CollectionManager>,
    /// 本节点稳定的复制 follower 身份；demote 转换沿用它参加握手。
    /// EN: stable replication follower identity for this node; the demote
    /// conversion reuses it in the handshake.
    pub follower_id: [u8; 16],
    /// 最近已提交 manifest 的 checkpoint；follower 握手用它协商补洞起点。
    /// EN: checkpoint of the latest committed manifest; the follower handshake
    /// uses it to negotiate where replay starts.
    pub snapshot_checkpoint: Arc<dyn Fn() -> u64 + Send + Sync>,
}

/// `POST /v1/admin/promote` 响应。
/// EN: Promote response body.
#[derive(Debug, Serialize)]
pub struct PromoteResponse {
    pub promoted: bool,
    /// promote 完成时 collections.wal 的下一条 LSN（复制进度校验用）。
    /// EN: collections.wal next LSN at promote time (replication progress check).
    pub next_lsn: u64,
    /// promote 后的持久化 term（EE M1）。 EN: Persisted term after promote (EE M1).
    pub term: u64,
    /// 本次授予的租约；bodyless 手动 promote 为 null。
    /// EN: lease granted by this promote; null for bodyless manual promotes.
    pub lease: Option<crate::cluster::LeaseGrant>,
}

// ── 服务配置 ────────────────────────────────────────────────
// EN: Server config.

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub worker_threshold: usize,
    pub worker_tick_ms: u64,
    pub flush_policy: FlushPolicy,
    pub metrics_interval_secs: u64,
    pub hnsw_ef_search: Option<usize>,
    pub vector_dim: usize,
    pub benchmark_http_addr: String,
    pub pgwire_addr: String,
    pub wal_path: PathBuf,
    pub wal_write_mode: WalWriteMode,
    pub pgwire_hnsw_m: Option<usize>,
    pub pgwire_hnsw_ef_construction: Option<usize>,
    pub index_type: String,
    /// OpenAI 兼容网关上游 base URL；None = 网关未启用。
    /// EN: Upstream base URL for the OpenAI-compatible gateway; None = disabled.
    pub openai_upstream: Option<String>,
    /// 上游调用超时（秒）。
    /// EN: Upstream call timeout (seconds).
    pub openai_timeout_secs: u64,
    /// 认证令牌（P4.2.1）：经 ENV HOUYI_AUTH_TOKEN 或 TOML `auth_token` 注入
    /// （不接 CLI 旗标——ps 可见即泄漏，与 HOUYI_OPENAI_API_KEY 同规；
    /// TOML 通道的磁盘权限由运维自理，加载时对组/他人可读发警告）。
    /// Some 时 HTTP /v1/*（除 health/readyz）要求 Bearer/X-Api-Key，
    /// pgwire 走明文密码质询。
    /// Auth token (ENV or TOML; never a CLI flag); None = auth disabled.
    pub auth_token: Option<String>,
    /// TLS 证书链 PEM 路径（P4.2.2）。与 `tls_key_path` 必须成对出现；
    /// 一对均 Some 时 HTTP 与 pgwire 共用同一 rustls acceptor（单证书）。
    /// PEM cert chain path; must be paired with `tls_key_path`.
    pub tls_cert_path: Option<String>,
    /// TLS 私钥 PEM 路径（P4.2.2）。
    /// PEM private key path.
    pub tls_key_path: Option<String>,
    /// pgwire 并发连接上限（P4.2.3）：超限连接在握手后收 FATAL 53300 被拒。
    /// Max concurrent pgwire connections; overflow gets FATAL 53300 after handshake.
    pub max_pgwire_connections: usize,
    /// HTTP 在途请求上限（P4.2.3）：超限请求立即 503（探针与 /metrics 豁免）。
    /// Max in-flight HTTP requests; overflow gets an immediate 503
    /// (probes and /metrics are exempt).
    pub max_http_inflight: usize,
    /// 内存水位线字节数（P4.2.4）：RSS 超限拒绝写入（HTTP 503 / pgwire 53200）。
    /// 0 = 禁用（默认，opt-in）；仅 Linux 有 RSS 源，其他平台 fail-open。
    /// Memory watermark in bytes; over the mark writes are rejected
    /// (HTTP 503 / pgwire 53200). 0 = disabled (default, opt-in).
    pub max_memory_bytes: u64,
    /// 慢查询阈值毫秒数（P4.3.2）：pgwire/HTTP 检索路径超阈值打 warn 日志。
    /// 0 = 禁用（默认，opt-in）。
    /// Slow-query threshold in milliseconds; pgwire/HTTP search paths over the
    /// mark emit a warn log. 0 = disabled (default, opt-in).
    pub slow_query_ms: u64,
    /// 图快照周期间隔秒数（P4.4.0e）：默认 600；0 = 关闭周期触发。
    /// Periodic graph snapshot interval in seconds. 0 = disabled.
    pub snapshot_interval_secs: u64,
    /// 图快照 WAL 增量阈值字节数（P4.4.0e）：主 WAL + collections.wal 自上次
    /// 快照以来的增量超阈值即触发；默认 256MB；0 = 关闭阈值触发。
    /// WAL-bytes-since-last-snapshot threshold triggering a graph snapshot.
    /// 0 = disabled.
    pub snapshot_wal_threshold_bytes: u64,
    /// 标量页自动压实间隔秒数（P5.0.1）：0 = 关闭自动压实。
    /// EN: Scalar-page auto-compaction interval in seconds; 0 disables it.
    pub collections_scalar_compact_interval_secs: u64,
    /// 触发自动压实的 stale 行占比（0..=1）。
    /// EN: Stale-row ratio that triggers auto-compaction (0..=1).
    pub collections_scalar_compact_stale_ratio: f64,
    /// collections WAL 自动轮转阈值字节（P5.0.2；0 = 只手动轮转）。
    /// EN: collections WAL auto-rotation threshold (P5.0.2; 0 = manual only).
    pub collections_wal_segment_bytes: u64,
    /// collections WAL 保留段数（P5.0.3；0 = 关闭删除）。
    /// EN: collections WAL retained segment count (P5.0.3; 0 disables deletion).
    pub collections_wal_max_segments: usize,
    /// leader 断连 follower 进度保留秒数 / 启动与 promote 宽限（P5.0.4）。
    /// EN: leader grace for disconnected follower progress and startup/promote
    /// (P5.0.4).
    pub collections_wal_follower_grace_secs: u64,
    /// 复制角色（P4.4.1）：默认 disabled（单机）。
    /// Replication role; disabled (standalone) by default.
    pub role: ReplicationRole,
    /// leader 监听地址（role=leader 生效）。
    /// EN: Replication listen address (leader).
    pub replication_listen_addr: String,
    /// leader 地址（role=follower 必填）。
    /// EN: Leader address (required for followers).
    pub replication_leader_addr: Option<String>,
    /// 复制认证共享秘密（P4.6.3，ENV `HOUYI_SERVER_REPLICATION_AUTH_TOKEN` /
    /// TOML `replication_auth_token`）：leader 与 follower 必须一致；未配置 =
    /// 不认证（可信网络）。敏感项纪律与 `auth_token` 相同：不接 CLI 旗标，
    /// 空白值启动期报错（fail-closed）。
    /// EN: shared replication secret; both sides must match. Unset = no auth.
    /// Same sensitive-value discipline as `auth_token`: no CLI flag, blank is
    /// fatal at startup.
    pub replication_auth_token: Option<String>,
    /// 每 follower 扇出队列深度（批数）：慢 follower 积满即被踢出（不拖累
    /// wal worker 与其他 follower）。 / Per-follower ship queue depth in batches;
    /// a slow follower that fills it is kicked rather than back-pressuring the
    /// WAL writer.
    pub replication_queue_batches: usize,
    /// sync 等待超时毫秒（P4.4.2）：semi_sync/full_sync collection 的写入
    /// 等 follower ACK 的上限；超时照常返回成功（语义降级 async）+
    /// `sync_timeouts_total` 指标 + rate-limited warn。
    /// EN: Timeout for semi/full-sync waits; on expiry the write still
    /// succeeds (degraded to async semantics) with a metric and warn.
    pub replication_sync_timeout_ms: u64,
    /// 新建 collection 的默认 HA 级别（P4.4.2，TOML `default_ha_level`，
    /// 缺省 async；collection 级设置优先）。P4.6.3 起为枚举——配置通道
    /// parse 即定型，非法值不可表达。
    /// EN: Default HA level for new collections (enum since P4.6.3).
    pub default_ha_level: houyi_persistence::ha_level::HaLevel,
    /// 审计日志路径（P4.7.0；ENV `HOUYI_SERVER_AUDIT_LOG` / TOML
    /// `audit_log_path`）。None = 关闭；设置后启动即校验哈希链，链断拒绝启动。
    /// EN: audit log path; None disables it. When set, the chain is verified on
    /// startup and a broken chain refuses to start.
    pub audit_log_path: Option<PathBuf>,
    /// 是否把读请求也计入审计（默认 false：只记写/管理/认证失败）。
    /// EN: include reads in the audit stream (default false).
    pub audit_include_reads: bool,
    /// 审计段轮转阈值字节数（0 = 不轮转）。轮转不删除旧段。
    /// EN: segment rotation threshold in bytes (0 = never rotate); old segments
    /// are never deleted.
    pub audit_max_bytes: u64,
    /// RBAC 主体（P4.7.1，TOML `[[auth_principals]]`）。legacy `auth_token`
    /// 仍作为全权 admin 追加；两者都为空 = 不启用认证。
    /// EN: RBAC principals; the legacy auth_token is appended as an unscoped
    /// admin. Both empty = authentication disabled.
    pub auth_principals: Vec<PrincipalConfig>,
    /// 静态加密 keyring 文件（P4.7.3；ENV `HOUYI_SERVER_ENCRYPTION_KEY_FILE` /
    /// TOML `encryption_key_file`）。None = 不加密。设置后 WAL 与图快照
    /// 一律加密落盘，旧明文历史在打开时一次性迁移。
    /// EN: keyring path; None disables encryption at rest. When set, WAL and
    /// graph snapshots are encrypted, and legacy plaintext is migrated once.
    pub encryption_key_file: Option<PathBuf>,
    /// 新写入使用的活动密钥 id（ENV `HOUYI_SERVER_ENCRYPTION_KEY_ID` / TOML
    /// `encryption_key_id`；缺省取 keyring 内的 `active=` 或唯一键）。
    /// EN: active key id for new writes; defaults to the keyring's `active=`.
    pub encryption_key_id: Option<String>,
    /// Sentinel 专用共享秘密（M1；ENV `HOUYI_SERVER_SENTINEL_TOKEN` / TOML
    /// `sentinel_token`）。与 `auth_token` 同属敏感项：不接 CLI 旗标，空白值
    /// 启动期报错。设置后 `sentinel` 角色只能访问 lease/demote/promote/
    /// replication-status（admin 仍隐含全部权限）。
    /// EN: dedicated sentinel shared secret (M1; ENV/TOML only, never a CLI
    /// flag). Grants the `sentinel` role its control endpoints; admin keeps
    /// everything.
    pub sentinel_token: Option<String>,
    /// 写 fencing（M1；ENV `HOUYI_SERVER_REQUIRE_LEADER_LEASE` / TOML
    /// `require_leader_lease`，CLI `--require-leader-lease`）。开启后 leader
    /// 必须持有未过期租约才可写；默认 false 与手动 promote 逐字节兼容。
    /// EN: require a valid leader lease for writes; off by default so manual
    /// promote stays byte-for-byte compatible.
    pub require_leader_lease: bool,
    /// 租约 TTL 上限（毫秒，M1；默认 60000，硬上限同为 60000）。
    /// EN: lease TTL cap in milliseconds (hard cap is the same constant).
    pub leader_lease_max_ttl_ms: u64,
    /// sync 等待的 follower quorum（M1；0 = 任一 ACK）。
    /// EN: follower quorum for sync waits (0 = any ACK).
    pub replication_sync_quorum: usize,
    /// 查询日志容量（M1；0 = 关闭，默认）。有界 ring buffer，只记元数据。
    /// EN: query-log capacity (0 = disabled, the default); a bounded ring
    /// buffer that records metadata only.
    pub query_log_capacity: usize,
    /// 查询日志采样：每 N 条记录 1 条（默认 1 = 全量）。
    /// EN: query-log sampling — record 1 in N queries (default 1 = all).
    pub query_log_sample_every: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            worker_threshold: 256,
            worker_tick_ms: 20,
            flush_policy: FlushPolicy::Hybrid,
            metrics_interval_secs: 1,
            hnsw_ef_search: None,
            vector_dim: 1536,
            benchmark_http_addr: "127.0.0.1:7000".to_string(),
            pgwire_addr: "127.0.0.1:5432".to_string(),
            wal_path: std::env::temp_dir().join("houyidb-dev.wal"),
            wal_write_mode: WalWriteMode::default(),
            pgwire_hnsw_m: None,
            pgwire_hnsw_ef_construction: None,
            index_type: "hnsw".to_string(),
            openai_upstream: None,
            openai_timeout_secs: 30,
            auth_token: None,
            tls_cert_path: None,
            tls_key_path: None,
            max_pgwire_connections: DEFAULT_MAX_PGWIRE_CONNECTIONS,
            max_http_inflight: DEFAULT_MAX_HTTP_INFLIGHT,
            max_memory_bytes: 0,
            slow_query_ms: 0,
            snapshot_interval_secs: DEFAULT_SNAPSHOT_INTERVAL_SECS,
            snapshot_wal_threshold_bytes: DEFAULT_SNAPSHOT_WAL_THRESHOLD_BYTES,
            collections_scalar_compact_interval_secs:
                DEFAULT_COLLECTIONS_SCALAR_COMPACT_INTERVAL_SECS,
            collections_scalar_compact_stale_ratio: DEFAULT_COLLECTIONS_SCALAR_COMPACT_STALE_RATIO,
            collections_wal_segment_bytes: DEFAULT_COLLECTIONS_WAL_SEGMENT_BYTES,
            collections_wal_max_segments: DEFAULT_COLLECTIONS_WAL_MAX_SEGMENTS,
            collections_wal_follower_grace_secs: DEFAULT_COLLECTIONS_WAL_FOLLOWER_GRACE_SECS,
            role: ReplicationRole::default(),
            replication_listen_addr: DEFAULT_REPLICATION_LISTEN_ADDR.to_string(),
            replication_leader_addr: None,
            replication_auth_token: None,
            replication_queue_batches: DEFAULT_REPLICATION_QUEUE_BATCHES,
            replication_sync_timeout_ms: DEFAULT_REPLICATION_SYNC_TIMEOUT_MS,
            default_ha_level: houyi_persistence::ha_level::HaLevel::Async,
            audit_log_path: None,
            audit_include_reads: false,
            audit_max_bytes: DEFAULT_AUDIT_MAX_BYTES,
            auth_principals: Vec::new(),
            encryption_key_file: None,
            encryption_key_id: None,
            sentinel_token: None,
            require_leader_lease: false,
            leader_lease_max_ttl_ms: crate::cluster::MAX_LEASE_TTL_MS,
            replication_sync_quorum: 0,
            query_log_capacity: 0,
            query_log_sample_every: 1,
        }
    }
}

/// TOML 配置文件镜像（P4.3.1）：全 Option——只有显式写出的键才覆盖默认值。
/// 合并优先级：CLI 旗标 > ENV > TOML > default（`--config` 路径本身 CLI > ENV）。
/// 敏感项说明：auth_token 允许入文件（磁盘权限 600 自担），仍不接 CLI 旗标。
/// `deny_unknown_fields`：拼错的键必须 fast-fail——枚举笔误致命是既定纪律，
/// 键名笔误静默落默认只会更隐蔽（OCR 批次三）。
/// EN: TOML mirror with all-optional fields — only written keys override the
/// default. Precedence: CLI flag > ENV > TOML > default. Unknown keys are
/// fatal (a misspelled key must not silently fall back to the default).
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ServerConfigFile {
    pub worker_threshold: Option<usize>,
    pub worker_tick_ms: Option<u64>,
    pub flush_policy: Option<String>,
    pub metrics_interval_secs: Option<u64>,
    pub hnsw_ef_search: Option<usize>,
    pub vector_dim: Option<usize>,
    pub benchmark_http_addr: Option<String>,
    pub pgwire_addr: Option<String>,
    pub wal_path: Option<String>,
    pub wal_write_mode: Option<String>,
    pub pgwire_hnsw_m: Option<usize>,
    pub pgwire_hnsw_ef_construction: Option<usize>,
    pub index_type: Option<String>,
    pub openai_upstream: Option<String>,
    pub openai_timeout_secs: Option<u64>,
    pub auth_token: Option<String>,
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
    pub max_pgwire_connections: Option<usize>,
    pub max_http_inflight: Option<usize>,
    pub max_memory_bytes: Option<u64>,
    pub slow_query_ms: Option<u64>,
    pub snapshot_interval_secs: Option<u64>,
    pub snapshot_wal_threshold_bytes: Option<u64>,
    pub collections_scalar_compact_interval_secs: Option<u64>,
    pub collections_scalar_compact_stale_ratio: Option<f64>,
    pub collections_wal_segment_bytes: Option<u64>,
    pub collections_wal_max_segments: Option<usize>,
    pub collections_wal_follower_grace_secs: Option<u64>,
    pub role: Option<String>,
    pub replication_listen_addr: Option<String>,
    pub replication_leader_addr: Option<String>,
    pub replication_auth_token: Option<String>,
    pub replication_queue_batches: Option<usize>,
    pub replication_sync_timeout_ms: Option<u64>,
    pub default_ha_level: Option<String>,
    pub audit_log_path: Option<String>,
    pub audit_include_reads: Option<bool>,
    pub audit_max_bytes: Option<u64>,
    pub encryption_key_file: Option<String>,
    pub encryption_key_id: Option<String>,
    pub sentinel_token: Option<String>,
    pub require_leader_lease: Option<bool>,
    pub leader_lease_max_ttl_ms: Option<u64>,
    pub replication_sync_quorum: Option<usize>,
    pub query_log_capacity: Option<usize>,
    pub query_log_sample_every: Option<u32>,
    /// `[[auth_principals]]` 数组；缺省为空。
    /// RBAC principals; absent means empty.
    #[serde(default)]
    pub auth_principals: Vec<PrincipalConfig>,
}

/// 单个 RBAC 主体配置（P4.7.1）。
/// One RBAC principal from config.
///
/// `token` 与 `token_sha256` 二选一：前者方便小规模私有化（明文只在启动时
/// 读入并立即哈希），后者用于不希望配置文件出现明文的场景。
/// EN: exactly one of `token` / `token_sha256`; the plaintext is hashed at load.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct PrincipalConfig {
    pub name: String,
    pub role: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub token_sha256: Option<String>,
    /// None = 全部 collection；`["*"]` 等价全部；否则为白名单。
    /// None = all collections; ["*"] is equivalent; otherwise a whitelist.
    #[serde(default)]
    pub collections: Option<Vec<String>>,
    /// 可见 collection 数量上限（None = 不限；仅约束新建，不删除已有数据）。
    /// EN: max visible collections for this principal (None = unlimited;
    /// only creation is bounded, existing data is never deleted).
    #[serde(default)]
    pub max_collections: Option<usize>,
    /// 在途请求上限（None = 默认 64；0 = 不限）。超限 429。
    /// EN: in-flight request cap (None = 64; 0 = unlimited); 429 on overflow.
    #[serde(default)]
    pub max_inflight: Option<usize>,
    /// 写请求速率（次/秒，突发 = 1 秒额度；None = 不限）。超限 429。
    /// EN: write requests per second, burst = 1s worth (None = unlimited).
    #[serde(default)]
    pub writes_per_sec: Option<u32>,
}

impl ServerConfigFile {
    /// 把显式写出的键应用到 `cfg`；枚举字符串（flush_policy/wal_write_mode）
    /// 解析失败即报错（配置文件的笔误不应静默落回默认）。
    /// EN: Apply written keys onto `cfg`; enum parse failures are fatal —
    /// a typo in a config file must not silently fall back to the default.
    pub fn apply_to(self, cfg: &mut ServerConfig) -> Result<(), String> {
        if let Some(v) = self.worker_threshold {
            cfg.worker_threshold = v;
        }
        if let Some(v) = self.worker_tick_ms {
            cfg.worker_tick_ms = v;
        }
        if let Some(v) = self.flush_policy {
            cfg.flush_policy = parse_flush_policy(&v)?;
        }
        if let Some(v) = self.metrics_interval_secs {
            cfg.metrics_interval_secs = v;
        }
        if let Some(v) = self.hnsw_ef_search {
            cfg.hnsw_ef_search = Some(v);
        }
        if let Some(v) = self.vector_dim {
            cfg.vector_dim = v;
        }
        if let Some(v) = self.benchmark_http_addr {
            cfg.benchmark_http_addr = v;
        }
        if let Some(v) = self.pgwire_addr {
            cfg.pgwire_addr = v;
        }
        if let Some(v) = self.wal_path {
            // 空白 wal_path 视为未写出（空白 PathBuf 会让 WAL open 在启动后才炸，
            // 与其他 Option<String> 键的空白过滤纪律对齐——OCR 批次三）。
            // EN: Blank wal_path counts as not written (a blank PathBuf makes WAL open
            // EN: fail only after startup; aligns with blank filtering of Option<String>
            // EN: keys — OCR batch 3).
            if let Some(v) = Some(v).filter(|s| !s.trim().is_empty()) {
                cfg.wal_path = PathBuf::from(v);
            }
        }
        if let Some(v) = self.wal_write_mode {
            cfg.wal_write_mode = parse_wal_write_mode(&v)?;
        }
        if let Some(v) = self.pgwire_hnsw_m {
            cfg.pgwire_hnsw_m = Some(v);
        }
        if let Some(v) = self.pgwire_hnsw_ef_construction {
            cfg.pgwire_hnsw_ef_construction = Some(v);
        }
        if let Some(v) = self.index_type {
            cfg.index_type = v;
        }
        if let Some(v) = self.openai_upstream {
            cfg.openai_upstream = Some(v).filter(|s| !s.trim().is_empty());
        }
        if let Some(v) = self.openai_timeout_secs {
            cfg.openai_timeout_secs = v;
        }
        if let Some(v) = self.auth_token {
            // fail-closed（OCR 359d85d[6]，与 ENV 通道同规）：TOML 里写了
            // 空白 token = 运维以为认证已开——启动期报错，不静默降级 None。
            // 存 trim 后的值（比较侧不 trim，原样存会永远匹配不上）。
            // EN: fail-closed (OCR 359d85d[6], same rule as the ENV channel): a blank
            // EN: TOML token means the operator thinks auth is on — error at startup,
            // EN: never silently degrade to None. Store the trimmed value (comparison
            // EN: side does not trim, so storing it raw would never match).
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(
                    "config file auth_token is blank; refusing to start unauthenticated \
                     (remove the key to disable auth explicitly)"
                        .to_string(),
                );
            }
            cfg.auth_token = Some(trimmed.to_string());
        }
        if let Some(v) = self.tls_cert_path {
            cfg.tls_cert_path = Some(v).filter(|s| !s.trim().is_empty());
        }
        if let Some(v) = self.tls_key_path {
            cfg.tls_key_path = Some(v).filter(|s| !s.trim().is_empty());
        }
        if let Some(v) = self.max_pgwire_connections {
            cfg.max_pgwire_connections = v;
        }
        if let Some(v) = self.max_http_inflight {
            cfg.max_http_inflight = v;
        }
        if let Some(v) = self.max_memory_bytes {
            cfg.max_memory_bytes = v;
        }
        if let Some(v) = self.slow_query_ms {
            cfg.slow_query_ms = v;
        }
        if let Some(v) = self.snapshot_interval_secs {
            cfg.snapshot_interval_secs = v;
        }
        if let Some(v) = self.snapshot_wal_threshold_bytes {
            cfg.snapshot_wal_threshold_bytes = v;
        }
        if let Some(v) = self.collections_scalar_compact_interval_secs {
            cfg.collections_scalar_compact_interval_secs = v;
        }
        if let Some(v) = self.collections_scalar_compact_stale_ratio {
            cfg.collections_scalar_compact_stale_ratio = v;
        }
        if let Some(v) = self.collections_wal_segment_bytes {
            cfg.collections_wal_segment_bytes = v;
        }
        if let Some(v) = self.collections_wal_max_segments {
            cfg.collections_wal_max_segments = v;
        }
        if let Some(v) = self.collections_wal_follower_grace_secs {
            cfg.collections_wal_follower_grace_secs = v;
        }
        if let Some(v) = self.role {
            cfg.role = parse_replication_role(&v)?;
        }
        if let Some(v) = self.replication_listen_addr {
            cfg.replication_listen_addr = v;
        }
        if let Some(v) = self.replication_leader_addr {
            cfg.replication_leader_addr = Some(v).filter(|s| !s.trim().is_empty());
        }
        if let Some(v) = self.replication_auth_token {
            // fail-closed（与 auth_token 同规）：空白 token 会被误读为“认证已开”。
            // EN: a blank token looks like auth is on — refuse to start.
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(
                    "config file replication_auth_token is blank; remove the key to \
                     disable replication auth explicitly"
                        .to_string(),
                );
            }
            houyi_persistence::wal_replication::validate_auth_token(trimmed)
                .map_err(|e| format!("replication_auth_token: {e}"))?;
            cfg.replication_auth_token = Some(trimmed.to_string());
        }
        if let Some(v) = self.replication_queue_batches {
            cfg.replication_queue_batches = v;
        }
        if let Some(v) = self.replication_sync_timeout_ms {
            cfg.replication_sync_timeout_ms = v;
        }
        if let Some(v) = self.default_ha_level {
            // 枚举笔误 fast-fail（与 role/flush_policy 同纪律）：静默落 async
            // 会让「以为有 HA」的部署实际裸奔。
            // EN: A typo is fatal — silently landing on async would leave a
            // deployment that believes it has HA actually unprotected.
            cfg.default_ha_level = houyi_persistence::ha_level::HaLevel::parse(&v)
                .map_err(|e| format!("default_ha_level: {e}"))?;
        }
        if let Some(v) = self.audit_log_path {
            // 空白路径 = 运维以为审计已开（fail-closed）。
            // EN: A blank path means the operator thinks audit is on (fail-closed).
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(
                    "config file audit_log_path is blank; remove the key to disable \
                     auditing explicitly"
                        .to_string(),
                );
            }
            cfg.audit_log_path = Some(PathBuf::from(trimmed));
        }
        if let Some(v) = self.audit_include_reads {
            cfg.audit_include_reads = v;
        }
        if let Some(v) = self.audit_max_bytes {
            cfg.audit_max_bytes = v;
        }
        if let Some(v) = self.encryption_key_file {
            // 空白路径 = 运维以为加密已开（fail-closed，与审计同规）。
            // EN: a blank path looks like encryption is on — refuse to start.
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(
                    "config file encryption_key_file is blank; remove the key to disable \
                     encryption at rest explicitly"
                        .to_string(),
                );
            }
            cfg.encryption_key_file = Some(PathBuf::from(trimmed));
        }
        if let Some(v) = self.encryption_key_id {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(
                    "config file encryption_key_id is blank; remove the key to use the \
                     keyring's active= selection"
                        .to_string(),
                );
            }
            cfg.encryption_key_id = Some(trimmed.to_string());
        }
        if let Some(v) = self.sentinel_token {
            // 与 auth_token 同规（fail-closed）：空白 = 运维以为 sentinel 控制面
            // 已启用；静默降级 None 会让租约/降级端点落到普通认证面。
            // EN: same fail-closed rule as auth_token — a blank sentinel token
            // looks enabled but would silently drop the control-plane credential.
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(
                    "config file sentinel_token is blank; remove the key to disable the \
                     sentinel control plane explicitly"
                        .to_string(),
                );
            }
            cfg.sentinel_token = Some(trimmed.to_string());
        }
        if let Some(v) = self.require_leader_lease {
            cfg.require_leader_lease = v;
        }
        if let Some(v) = self.leader_lease_max_ttl_ms {
            cfg.leader_lease_max_ttl_ms = v;
        }
        if let Some(v) = self.replication_sync_quorum {
            cfg.replication_sync_quorum = v;
        }
        if let Some(v) = self.query_log_capacity {
            cfg.query_log_capacity = v;
        }
        if let Some(v) = self.query_log_sample_every {
            cfg.query_log_sample_every = v;
        }
        if !self.auth_principals.is_empty() {
            cfg.auth_principals = self.auth_principals;
        }
        Ok(())
    }
}

pub fn parse_flush_policy(raw: &str) -> Result<FlushPolicy, String> {
    match raw {
        "threshold" => Ok(FlushPolicy::ThresholdOnly),
        "interval" => Ok(FlushPolicy::IntervalOnly),
        "hybrid" => Ok(FlushPolicy::Hybrid),
        _ => Err(format!(
            "invalid policy: {raw}, expected one of threshold|interval|hybrid"
        )),
    }
}

// ── pgwire 共享状态 ─────────────────────────────────────────
// EN: pgwire shared state.

#[derive(Clone)]
pub struct PgWireSharedState {
    pub index: IndexBackend,
    pub collections: Arc<CollectionManager>,
    pub vector_dim: usize,
    pub pgwire_hnsw_m: Option<usize>,
    pub pgwire_hnsw_ef_construction: Option<usize>,
    /// 主向量 WAL（无 collection 指令的 pgwire upsert 走主索引，与 HTTP benchmark
    /// 路径同纪律 WAL-first）。None = 测试装配，不写 WAL。
    /// EN: Main vector WAL for directive-less pgwire upserts into the main index
    /// (WAL-first, same discipline as the HTTP benchmark path). None in tests.
    pub wal: Option<Arc<WalHandle>>,
    /// Collections WAL（带 collection 指令的 upsert）。None = 测试装配，不写 WAL。
    /// EN: Collections WAL for directed upserts. None in tests.
    pub collections_wal: Option<Arc<WalHandle>>,
    /// 运行态复制角色（P4.4.1）：follower 时 upsert 拒绝（SQLSTATE 25006）。
    /// EN: Runtime replication role — followers reject upserts with 25006.
    pub role: RoleState,
    /// 复制指标槽（P4.4.3）：collection 写路径的 sync 等待（P4.4.2）经此触达
    /// 水位线与 hub——与 HTTP 共用同一槽，promote 换视图后 pgwire 写同样即时
    /// 获得/失去 sync 语义。
    /// EN: Shared replication slot so pgwire collection writes get the same
    /// P4.4.2 sync-wait semantics (and promote swaps the view live).
    pub replication_metrics: SharedReplicationMetrics,
    /// 集群控制原语（M1）：pgwire 写路径与 HTTP 共用同一租约句柄。
    /// EN: cluster primitives (M1) shared with the HTTP write path.
    pub cluster: Arc<crate::cluster::ClusterControl>,
    /// 查询日志（M1）：与 HTTP 共用同一个有界缓冲区。
    /// EN: query log (M1) shared with the HTTP side.
    pub query_log: Option<Arc<crate::query_log::QueryLog>>,
}

#[derive(Clone)]
pub struct PreparedStatementEntry {
    pub sql: String,
    pub param_count: usize,
    pub param_oids: Vec<u32>,
}

#[derive(Clone)]
pub struct PgWireColumnMeta {
    pub name: String,
    pub type_oid: u32,
}

#[derive(Default)]
pub struct PgWireConnState {
    pub unnamed_stmt: Option<String>,
    pub unnamed_param_count: usize,
    pub unnamed_param_oids: Vec<u32>,
    pub named_statements: HashMap<String, PreparedStatementEntry>,
    pub bound_unnamed_sql: Option<String>,
    pub bound_statement_name: Option<String>,
    pub bound_unnamed_row_description_sent: bool,
    pub unnamed_result_formats: Vec<i16>,
    pub pending_portal: Option<PendingPortalResult>,
    pub pending_upsert_batch: Option<HouyiUpsertBatch>,
}

pub struct PendingPortalResult {
    pub columns: Vec<PgWireColumnMeta>,
    pub rows: Vec<Vec<String>>,
    pub next_idx: usize,
    pub total_rows: usize,
}

#[derive(Clone, Debug)]
pub struct HouyiUpsertBatch {
    pub dim: usize,
    pub items: Vec<(u64, Vec<f32>)>,
}

// ── pgwire 指标 ─────────────────────────────────────────────
// EN: pgwire metrics.

/// pgwire 并发连接上限默认值（P4.2.3）。
/// Default cap on concurrent pgwire connections.
pub const DEFAULT_MAX_PGWIRE_CONNECTIONS: usize = 200;
/// HTTP 在途请求上限默认值（P4.2.3）。
/// Default cap on in-flight HTTP requests.
pub const DEFAULT_MAX_HTTP_INFLIGHT: usize = 512;
/// 图快照周期间隔默认值秒（P4.4.0e）。
/// EN: Default periodic snapshot interval.
pub const DEFAULT_SNAPSHOT_INTERVAL_SECS: u64 = 600;
/// 图快照 WAL 增量阈值默认值字节（256MB，P4.4.0e）。
/// Default WAL-delta snapshot threshold (256 MiB).
pub const DEFAULT_SNAPSHOT_WAL_THRESHOLD_BYTES: u64 = 268_435_456;
/// 标量页自动压实默认间隔秒（P5.0.1）。
/// EN: Default scalar-page auto-compaction interval.
pub const DEFAULT_COLLECTIONS_SCALAR_COMPACT_INTERVAL_SECS: u64 = 600;
/// 标量页自动压实默认 stale 比例（P5.0.1）。
/// EN: Default stale-row ratio for scalar auto-compaction.
pub const DEFAULT_COLLECTIONS_SCALAR_COMPACT_STALE_RATIO: f64 = 0.5;
/// collections WAL 默认段大小（64 MiB，P5.0.2）。
/// EN: Default collections WAL segment size (64 MiB, P5.0.2).
pub const DEFAULT_COLLECTIONS_WAL_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;
/// collections WAL 默认保留段数（P5.0.3）。
/// EN: Default retained collections WAL segment count (P5.0.3).
pub const DEFAULT_COLLECTIONS_WAL_MAX_SEGMENTS: usize = 16;
/// leader 断连/启动/promote 默认宽限秒数（P5.0.4）。
/// EN: Default leader grace seconds for disconnects/startup/promote (P5.0.4).
pub const DEFAULT_COLLECTIONS_WAL_FOLLOWER_GRACE_SECS: u64 = 60;
/// 复制 leader 默认监听地址（P4.4.1）。
/// EN: Default replication listen address.
pub const DEFAULT_REPLICATION_LISTEN_ADDR: &str = "0.0.0.0:9876";
/// 每 follower 扇出队列默认深度（批数，P4.4.1）。
/// Default per-follower ship queue depth in batches.
pub const DEFAULT_REPLICATION_QUEUE_BATCHES: usize = 4096;
/// sync 等待超时默认毫秒（P4.4.2）。
/// EN: Default semi/full-sync wait timeout.
pub const DEFAULT_REPLICATION_SYNC_TIMEOUT_MS: u64 = 2000;
/// 审计段轮转阈值默认 256MB（0 = 不轮转）。
/// Default audit segment rotation threshold (0 disables rotation).
pub const DEFAULT_AUDIT_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// 在途水位线（P4.2.3）：gauge + 上限 + 拒绝计数，CAS 占位、RAII 归还。
/// 连接（pgwire）与请求（HTTP）共用同一原语；拒绝计数单调递增供 /metrics 暴露。
/// EN: Shared in-flight watermark primitive — CAS-acquired slot, RAII release,
/// monotonic rejection counter for /metrics.
#[derive(Debug)]
pub struct InflightWatermark {
    max: u64,
    active: AtomicU64,
    rejected_total: AtomicU64,
}

impl Default for InflightWatermark {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_PGWIRE_CONNECTIONS as u64)
    }
}

impl InflightWatermark {
    pub fn new(max: u64) -> Self {
        Self {
            max,
            active: AtomicU64::new(0),
            rejected_total: AtomicU64::new(0),
        }
    }

    /// 尝试占一个名额；超限则计入拒绝并返回 None。
    /// Try to take a slot; on overflow count a rejection and return None.
    pub fn try_acquire(self: &Arc<Self>) -> Option<InflightPermit> {
        let mut cur = self.active.load(Ordering::Relaxed);
        loop {
            if cur >= self.max {
                self.rejected_total.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            match self.active.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(InflightPermit {
                        watermark: Arc::clone(self),
                    })
                }
                Err(actual) => cur = actual,
            }
        }
    }

    pub fn active(&self) -> u64 {
        self.active.load(Ordering::Relaxed)
    }

    pub fn rejected_total(&self) -> u64 {
        self.rejected_total.load(Ordering::Relaxed)
    }

    pub fn max(&self) -> u64 {
        self.max
    }
}

/// 水位线许可（RAII）：Drop 归还 gauge，杜绝各种退出路径漏还。
/// EN: RAII slot guard — Drop releases the gauge on every exit path.
#[derive(Debug)]
pub struct InflightPermit {
    watermark: Arc<InflightWatermark>,
}

impl Drop for InflightPermit {
    fn drop(&mut self) {
        self.watermark.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug, Default)]
pub struct PgWireMetrics {
    pub handshake_count: AtomicU64,
    pub frontend_messages: AtomicU64,
    pub decode_errors: AtomicU64,
    /// 认证失败计数（OCR 359d85d[12]）：凭据拒绝与协议 decode 错误分账——
    /// 混在 decode_errors 里，暴力破解会被 dashboard 误读成客户端 bug。
    /// 三条拒绝路径（非 'p' / 长度畸形 / 密码错误）全部计入。
    /// EN: Dedicated auth-failure counter (all three rejection paths) —
    /// brute-force attempts must not masquerade as client decode bugs.
    pub auth_failures: AtomicU64,
    /// RBAC 越权拒绝计数（P4.7.1）：与认证失败分账（身份未知 vs 权限不足）。
    /// EN: authorization denials, separate from authentication failures.
    pub authz_denials: AtomicU64,
    /// 配额拒绝计数（P4.7.2，SQLSTATE 54000）：与 42501 分账——配额是
    /// 背压/容量信号，混入越权计数会污染安全告警（OCR 批次三十四）。
    /// EN: quota rejections (54000) kept apart from 42501 so routine
    /// backpressure does not pollute authorization alarms.
    pub quota_denials: AtomicU64,
    pub simple_query_count: AtomicU64,
    pub parse_count: AtomicU64,
    pub bind_count: AtomicU64,
    pub execute_count: AtomicU64,
    pub describe_count: AtomicU64,
    pub close_count: AtomicU64,
    pub sync_count: AtomicU64,
    pub flush_count: AtomicU64,
    pub copy_count: AtomicU64,
    pub terminate_count: AtomicU64,
    pub unsupported_count: AtomicU64,
    pub sql_text_bytes: AtomicU64,
    pub bind_param_text_bytes: AtomicU64,
    pub frontend_payload_bytes: AtomicU64,
    pub upsert_batch_execute_count: AtomicU64,
    pub upsert_batch_rows_inserted: AtomicU64,
    /// 连接水位线（P4.2.3）：listener accept 后、查询循环前占位；
    /// 上限在 main 按配置覆盖（`Arc::new(InflightWatermark::new(max))`）。
    /// EN: Connection watermark, acquired post-handshake; main overrides the cap.
    pub connection_watermark: Arc<InflightWatermark>,
    /// 内存水位线（P4.2.4）：HTTP/pgwire 写入路径共用（默认禁用，main 按配置覆盖）。
    /// 挂在 metrics 里是为了让两个写入面零装配 churn 地共享同一份。
    /// EN: Memory watermark shared by both write paths (disabled by default);
    /// lives here so HTTP and pgwire share one instance with no wiring churn.
    pub memory_watermark: Arc<crate::mem::MemoryWatermark>,
    /// 慢查询阈值毫秒数（P4.3.2）：沿用 config-via-metrics 先例（同水位线），
    /// pgwire/HTTP 检索路径共享一份，0 = 禁用（默认，main 按配置覆盖）。
    /// EN: Slow-query threshold in ms, shared by both query surfaces via the
    /// config-via-metrics precedent; 0 = disabled (default, main overrides).
    pub slow_query_ms: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PgWireMetricsSnapshot {
    pub handshake_count: u64,
    pub frontend_messages: u64,
    pub decode_errors: u64,
    pub auth_failures: u64,
    pub authz_denials: u64,
    pub quota_denials: u64,
    pub simple_query_count: u64,
    pub parse_count: u64,
    pub bind_count: u64,
    pub execute_count: u64,
    pub describe_count: u64,
    pub close_count: u64,
    pub sync_count: u64,
    pub flush_count: u64,
    pub copy_count: u64,
    pub terminate_count: u64,
    pub unsupported_count: u64,
    pub sql_text_bytes: u64,
    pub bind_param_text_bytes: u64,
    pub frontend_payload_bytes: u64,
    pub upsert_batch_execute_count: u64,
    pub upsert_batch_rows_inserted: u64,
}

impl PgWireMetrics {
    pub fn snapshot(&self) -> PgWireMetricsSnapshot {
        PgWireMetricsSnapshot {
            handshake_count: self.handshake_count.load(Ordering::Relaxed),
            frontend_messages: self.frontend_messages.load(Ordering::Relaxed),
            decode_errors: self.decode_errors.load(Ordering::Relaxed),
            auth_failures: self.auth_failures.load(Ordering::Relaxed),
            authz_denials: self.authz_denials.load(Ordering::Relaxed),
            quota_denials: self.quota_denials.load(Ordering::Relaxed),
            simple_query_count: self.simple_query_count.load(Ordering::Relaxed),
            parse_count: self.parse_count.load(Ordering::Relaxed),
            bind_count: self.bind_count.load(Ordering::Relaxed),
            execute_count: self.execute_count.load(Ordering::Relaxed),
            describe_count: self.describe_count.load(Ordering::Relaxed),
            close_count: self.close_count.load(Ordering::Relaxed),
            sync_count: self.sync_count.load(Ordering::Relaxed),
            flush_count: self.flush_count.load(Ordering::Relaxed),
            copy_count: self.copy_count.load(Ordering::Relaxed),
            terminate_count: self.terminate_count.load(Ordering::Relaxed),
            unsupported_count: self.unsupported_count.load(Ordering::Relaxed),
            sql_text_bytes: self.sql_text_bytes.load(Ordering::Relaxed),
            bind_param_text_bytes: self.bind_param_text_bytes.load(Ordering::Relaxed),
            frontend_payload_bytes: self.frontend_payload_bytes.load(Ordering::Relaxed),
            upsert_batch_execute_count: self.upsert_batch_execute_count.load(Ordering::Relaxed),
            upsert_batch_rows_inserted: self.upsert_batch_rows_inserted.load(Ordering::Relaxed),
        }
    }
}

// ── HTTP 请求/响应类型 ───────────────────────────────────────
// EN: HTTP request/response types.

/// 标签连续向量缓存：`ids[i]` 对应 `data[i*dims..(i+1)*dims]`。
/// 过滤查询退化为顺序内存扫描（缓存构建时一次拷贝，Cosine 预先单位化），
/// 避免每 query 10K 次 HashMap 查表 + 散列指针跳转。
/// EN: Contiguous vector cache for a label; filtered queries become a sequential
/// scan (one copy at build, Cosine pre-normalized), avoiding 10K hash lookups.
pub struct LabelVectorCache {
    pub ids: Vec<u64>,
    pub data: Vec<f32>,
    pub dims: usize,
    /// 构建时的索引写入代数；查询时不等即失效（含 builder staged 窗口）。
    /// EN: Index write generation at build time; a mismatch invalidates the cache
    /// (including the builder's staged-batch window).
    pub generation: u64,
}

/// 单标签索引项：id 列表 + 惰性连续向量缓存。
/// EN: Per-label entry — id list plus a lazily built contiguous vector cache.
pub struct LabelEntry {
    pub ids: Arc<Vec<u64>>,
    pub cache: Arc<RwLock<Option<Arc<LabelVectorCache>>>>,
}

impl Default for LabelEntry {
    fn default() -> Self {
        Self {
            ids: Arc::new(Vec::new()),
            cache: Arc::new(RwLock::new(None)),
        }
    }
}

/// 标量标签索引: label → ids + 向量缓存
/// Label index: label → ids plus the contiguous vector cache.
pub type LabelIndex = Arc<RwLock<HashMap<String, LabelEntry>>>;

/// OpenAI 兼容 embeddings 网关的上游配置（`--openai-upstream` / `HOUYI_OPENAI_UPSTREAM`）。
/// EN: Upstream config for the OpenAI-compatible embeddings gateway.
///
/// API key 只经 ENV `HOUYI_OPENAI_API_KEY` 注入、不接 CLI 旗标——CLI 参数对
/// `ps` 可见，密钥落命令行等于泄漏。`base_url` 不含路径，`/v1/embeddings` 由
/// 网关拼接（与 OpenAI SDK 的 base_url 约定一致，Ollama/vLLM 等同款）。
#[derive(Clone)]
pub struct OpenAIUpstreamConfig {
    /// 上游 OpenAI 兼容服务 base URL，如 `https://api.openai.com`。
    /// EN: Upstream OpenAI-compatible base URL, e.g. `https://api.openai.com`.
    pub base_url: String,
    /// 上游 Bearer key（可选；本地 Ollama 等不需要）。
    /// EN: Upstream Bearer key (optional; local Ollama etc. do not need one).
    pub api_key: Option<String>,
    /// 单次上游调用超时。
    /// EN: Per-call upstream timeout.
    pub timeout: std::time::Duration,
    /// 共享 ureq Agent（连接池）：每请求新建 Agent 会让每次 embedding 调用都付
    /// 完整 TCP(+TLS) 握手，上游还会看到持续的连接 churn。Agent 内部 Arc，
    /// clone 廉价。
    /// EN: Shared ureq Agent (connection pool) — building one per request defeats
    /// keep-alive/TLS session reuse (full handshake per embeddings call) and shows
    /// the upstream constant connection churn. Internally Arc'd; cheap to clone.
    pub agent: ureq::Agent,
    /// 上游熔断器（P4.3.3）：Arc 共享——Config 每请求 clone，熔断状态必须跨
    /// 请求与跨 clone 唯一（与 agent 同纪律）。
    /// EN: Circuit breaker — Arc-shared so per-request Config clones observe
    /// one breaker state (same discipline as the pooled agent).
    pub breaker: Arc<crate::breaker::UpstreamBreaker>,
}

impl OpenAIUpstreamConfig {
    /// 构造并按 timeout 预建共享 Agent。
    /// EN: Build the config together with its pooled Agent (timeout baked in).
    pub fn new(base_url: String, api_key: Option<String>, timeout: std::time::Duration) -> Self {
        Self {
            base_url,
            api_key,
            timeout,
            // `http_status_as_error(false)`：4xx/5xx 也返回 Response，
            // 网关需要把上游错误正文原样透传（ureq 3 默认把它们变成
            // 不带正文的 Error::StatusCode）。
            // EN: keep non-2xx as responses so the gateway can forward the body.
            agent: ureq::Agent::new_with_config(
                ureq::Agent::config_builder()
                    .timeout_global(Some(timeout))
                    .http_status_as_error(false)
                    .build(),
            ),
            breaker: Arc::new(crate::breaker::UpstreamBreaker::new()),
        }
    }
}

impl std::fmt::Debug for OpenAIUpstreamConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 密钥永不进日志：只报告是否配置，不报告内容。
        // EN: Keys never enter logs: only report whether configured, never the content.
        f.debug_struct("OpenAIUpstreamConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[derive(Clone)]
pub struct BenchmarkHttpState {
    pub index: IndexBackend,
    /// HTTP 与 pgwire 共享的同一 CollectionManager（main 装配时注入）——
    /// 历史上 HTTP/pgwire 各自持有一份，同名 collection 两侧数据分裂；
    /// 共享后 collections WAL 只需向一个实体重放。
    /// EN: The single CollectionManager shared by HTTP and pgwire (injected by
    /// main). Historically each side held its own and same-named collections
    /// diverged; sharing lets the collections WAL replay into one entity.
    pub collections: Arc<CollectionManager>,
    pub wal: Arc<WalHandle>,
    /// Collections 专用 WAL（独立于主 WAL，WHY 见 houyi_persistence::collection_wal）。
    /// EN: Dedicated collections WAL (separate from the main WAL — see
    /// houyi_persistence::collection_wal for the rationale).
    pub collections_wal: Arc<WalHandle>,
    /// P5.0.3：快照/保留协调状态（锁、段上限、租约、计数器）。
    /// EN: P5.0.3 snapshot/retention coordination (lock, segment cap, lease, counters).
    pub lifecycle: Arc<crate::collection_lifecycle::CollectionLifecycleState>,
    pub vector_dim: usize,
    pub pgwire_metrics: Arc<PgWireMetrics>,
    pub applied_sidecar: PathBuf,
    pub applied_sidecar_lock: Arc<Mutex<()>>,
    pub wal_path: PathBuf,
    /// label → ids，供 benchmark API 做标量过滤。
    /// EN: label -> ids for benchmark API scalar filtering.
    pub label_index: LabelIndex,
    /// OpenAI 兼容 `/v1/embeddings` 网关的上游；None = 网关未启用（路由 501）。
    /// EN: Upstream for the OpenAI-compatible `/v1/embeddings` gateway; None = disabled (501).
    pub openai_upstream: Option<Arc<OpenAIUpstreamConfig>>,
    /// HTTP 在途请求水位线（P4.2.3）：中间件占位，/metrics 读 gauge 与拒绝计数。
    /// EN: In-flight HTTP request watermark — middleware acquires, /metrics reads.
    pub http_inflight: Arc<InflightWatermark>,
    /// 图快照指标（P4.4.0e）：恢复/写出两侧共用，/metrics 暴露。
    /// EN: Graph snapshot counters shared by the recovery and writer sides.
    pub graph_snapshot_metrics: Arc<crate::recovery::GraphSnapshotMetrics>,
    /// 运行态复制角色（P4.4.1）：follower 时写端点 503 `read_only`。
    /// EN: Runtime replication role — write endpoints return 503 on followers.
    pub role: RoleState,
    /// 复制指标槽（P4.4.1）：None = 未启用（/metrics 省略该节）。
    /// EN: Replication metrics slot; None = disabled (section omitted).
    pub replication_metrics: SharedReplicationMetrics,
    /// promote 上下文（P4.4.3）：role≠disabled 时 Some——仅 follower 会真正
    /// 走完流程（leader/disabled 在端点入口即 409/400），但 leader 初始角色
    /// 同样装配以保持「状态只经 state 单点触达」的纪律。
    /// EN: Promote context, present whenever replication is enabled; only a
    /// follower completes the flow, but leaders get it too so all replication
    /// state is reached through the state struct alone.
    pub promote: Option<Arc<PromoteContext>>,
    /// 审计日志（P4.7.0）：None = 未启用；中间件与 /metrics 共用。
    /// EN: audit log; None when disabled (shared by middleware and metrics).
    pub audit: Option<Arc<crate::audit::AuditLog>>,
    /// 静态加密句柄（P4.7.3）：旧 JSON 向量快照写入时据此加密。
    /// EN: at-rest encryption handle; the legacy JSON snapshot writer uses it.
    pub encryption: Option<Arc<houyi_persistence::encryption::Encryption>>,
    /// 集群控制原语（M1）：term 持久化 + leader 租约，写路径 fencing 与
    /// lease/demote 端点共用同一句柄。
    /// EN: cluster primitives (M1) — persisted term + leader lease shared by
    /// write fencing and the lease/demote endpoints.
    pub cluster: Arc<crate::cluster::ClusterControl>,
    /// 查询日志（M1）：None = 关闭（默认），关闭时写路径零开销。
    /// EN: query log (M1); None = disabled (default), zero cost on write paths.
    pub query_log: Option<Arc<crate::query_log::QueryLog>>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub vectors: Vec<UpsertVector>,
    #[serde(default)]
    pub index: Option<UpsertIndexHints>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertIndexHints {
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertVector {
    pub id: u64,
    pub values: Vec<f32>,
    #[serde(default)]
    pub label: Option<String>,
    /// 自由 JSON 标量字段；写入所属 Collection 的内存 PAX 标量页，供混合检索过滤。
    /// 标量页随 collections WAL 与快照持久化，重启后由重放/快照恢复；键名 `label`
    /// 为保留列，请改用顶层 `label` 字段（占用将返回 400）。
    /// EN: Free-form JSON scalar fields; stored in the collection's in-memory PAX
    /// scalar page for hybrid-search filtering. The page is persisted through the
    /// collections WAL and snapshots and restored on restart. The key `label` is
    /// reserved for the top-level field and rejected.
    /// `null`/array/object values land in a JSON-text column via the
    /// crate-private `json_value_to_scalar` conversion (module `scalar_json`).
    #[serde(default)]
    pub payload: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct UpsertResponse {
    pub inserted: usize,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    #[serde(default)]
    pub filter_gt_id: Option<u64>,
    #[serde(default)]
    pub ef_search: Option<usize>,
    #[serde(default)]
    pub filter_label: Option<String>,
    /// 检索响应是否携带每命中的 payload（label + payload 键值，键为 id 的字符串）。
    /// 仅 collection 路径生效（payload 存在 collection 的内存标量页）；benchmark
    /// 主索引路径无 payload 存储，该标志被忽略。默认 false 保持响应形状不变。
    /// EN: Whether to include per-hit payloads (label + payload keys, keyed by the
    /// stringified id). Collection paths only — the benchmark main index has no
    /// payload store and ignores the flag. Defaults to false; response shape is
    /// unchanged unless requested.
    #[serde(default)]
    pub include_payload: bool,
}

/// Collection 向量删除请求 / Delete-by-ids request for a collection.
///
/// 删除 WAL-first（`collection_delete` 记录先于内存变更）；墓碑随标量页一起进
/// 快照，重启后由重放/快照恢复。不存在的 id 静默跳过（幂等）；`deleted` 计真实
/// 存在的删除数。
/// EN: Deletion is WAL-first (`collection_delete` records precede the in-memory
/// change); tombstones are captured by snapshots together with the scalar page and
/// restored on restart via replay/snapshot. Unknown ids are skipped (idempotent);
/// `deleted` counts ids that actually existed.
#[derive(Debug, Deserialize)]
pub struct DeleteRequest {
    /// 显式 id 列表；与 `filters` 命中结果取**并集**（两者可单独或同时使用）。
    /// EN: Explicit ids; unioned with filter matches (either or both may be given).
    #[serde(default)]
    pub ids: Vec<u64>,
    /// 结构化过滤（AND 交集），命中的 id 一并删除。
    /// 为什么需要它：按条件删除必须走标量页求值——标量行随 upsert **同步**落页，
    /// 覆盖尚在 builder_queue 待构图的向量；若让调用方「先检索拿 id 再删」，
    /// 检索看不到队列内向量（双速写入），刚写入即按条件删除会静默漏删。
    /// EN: Structured filters (AND); matched ids are deleted too. Required because
    /// delete-by-condition must evaluate the scalar page — rows land synchronously
    /// at upsert, covering vectors still queued for graph building. A
    /// "search-then-delete" flow cannot see queued vectors (dual-speed writes) and
    /// would silently miss them.
    #[serde(default)]
    pub filters: Vec<HybridSearchFilter>,
}

/// 单条结构化过滤条件（标量列 + 操作符 + 字面值），混合检索与按条件删除共用。
/// One structured filter (scalar column + operator + literal), shared by hybrid
/// search and delete-by-filter.
#[derive(Debug, Deserialize)]
pub struct HybridSearchFilter {
    /// 标量列名（payload 键名，或保留列 `label`）。
    /// Scalar column name (a payload key, or the reserved `label` column).
    pub column: String,
    /// 操作符：`eq`/`gt`/`lt`/`gte`/`lte`/`in`；其余一律 400。
    /// Operator: `eq`/`gt`/`lt`/`gte`/`lte`/`in`; anything else yields 400.
    pub op: String,
    /// 比较字面值：`in` 必须是 JSON 数组，且元素只能是布尔/数字/字符串；
    /// 标量比较不得为数组/对象/null。形状不符在持锁前统一校验 → 400。
    /// Literal: must be a JSON array of boolean/number/string elements for `in`;
    /// must not be array/object/null for scalar comparisons. Bad shapes are
    /// rejected with 400 before taking the lock.
    pub value: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub deleted: usize,
}

/// 整库删除响应（`DELETE /v1/admin/collections/{name}`）。
/// EN: Whole-collection drop response.
#[derive(Debug, Serialize)]
pub struct DropCollectionResponse {
    /// 固定为 true：目标不存在时返回 404，不会出现 `deleted:false` 的 200。
    /// EN: Always true — an absent target yields 404, never a 200 with `false`.
    pub deleted: bool,
    pub collection: String,
}

/// 显式建集合请求（`PUT /v1/collections/{name}`）。
///
/// `metric` 缺省 Cosine（与历史隐式创建一致）；`index_type` 缺省服务端默认后端；
/// HNSW 三项必须全给或全不给（半套 400）。后端与度量的兼容矩阵由
/// `IndexBackend::metric_supported` 裁决：pq-flat 仅 L2、hii-l0 仅 Cosine，
/// 其余后端四种都支持。
/// EN: Explicit collection-creation request. `metric` defaults to Cosine (same as
/// legacy implicit creation); `index_type` defaults to the server's backend; the
/// three HNSW parameters must be all-present or all-absent. Backend/metric
/// compatibility is decided by `IndexBackend::metric_supported`: pq-flat is L2-only,
/// hii-l0 is Cosine-only, all other backends accept all four metrics.
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    /// 向量维度，必须 > 0（0 维度后端不可构造 → 400）。
    /// EN: Vector dimension; must be > 0 (a zero-dim backend cannot be built → 400).
    pub dims: usize,
    /// `l2` / `cosine` / `ip` / `l1`（大小写不敏感；常用别名见 DistanceMetric::parse_str）。
    /// EN: One of l2/cosine/ip/l1 (case-insensitive aliases accepted).
    #[serde(default)]
    pub metric: Option<String>,
    /// 后端类型：hnsw / pq-flat / flat / clustered / flat-clustered / hii-l0。
    /// EN: Backend kind: hnsw / pq-flat / flat / clustered / flat-clustered / hii-l0.
    #[serde(default)]
    pub index_type: Option<String>,
    #[serde(default)]
    pub hnsw_m: Option<usize>,
    #[serde(default)]
    pub hnsw_ef_construction: Option<usize>,
    #[serde(default)]
    pub hnsw_ef_search: Option<usize>,
}

/// 建集合时生效的 HNSW 参数（响应与状态端点共用）。
/// EN: Effective HNSW parameters at creation (shared by responses and status).
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct HnswParamsResponse {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

/// 显式建集合响应：回显实际生效的度量/后端/HNSW 参数（缺省值已展开）。
/// EN: Explicit-creation response echoing the effective metric/backend/HNSW
/// parameters (defaults already resolved).
#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    /// 固定 true：同名已存在返回 409，不会有 `created:false` 的 200。
    /// EN: Always true — an existing name yields 409, never 200 with `false`.
    pub created: bool,
    pub collection: String,
    pub dims: usize,
    /// 规范度量串（l2|cosine|ip|l1），与实际计分口径一致。
    /// EN: Canonical metric string matching the actual scoring metric.
    pub metric: String,
    pub index_type: String,
    /// 仅 `index_type=hnsw` 时存在；其他后端不序列化此字段。
    /// EN: Present only for `index_type=hnsw`; omitted for other backends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw: Option<HnswParamsResponse>,
}

/// 设置 Collection HA 级别请求（P4.4.2）/ Set a collection's HA level.
///
/// 取值 `async` / `semi_sync` / `full_sync`（大小写与连字符变体均可，
/// 规范定义在 `houyi_persistence::ha_level`）；笔误 400 fail-fast。
/// EN: Accepts `async` / `semi_sync` / `full_sync` (case/hyphen variants
/// tolerated; canonical definitions live in `houyi_persistence::ha_level`);
/// typos are a 400.
#[derive(Debug, Deserialize)]
pub struct SetHaLevelRequest {
    pub ha_level: String,
}

/// HA 级别设置响应：回显规范字符串（与落盘值一致）。
/// EN: Canonical level echoed back.
#[derive(Debug, Serialize)]
pub struct SetHaLevelResponse {
    pub collection: String,
    pub ha_level: String,
}

/// 向量检索响应：`ids[i]` 与 `scores[i]` 按下标一一对应，按相似度从高到低排列。
///
/// Vector search response: `ids[i]` and `scores[i]` are aligned by index and ordered
/// by descending similarity.
///
/// `scores` 的语义按 collection 的**度量**（`metric`，创建时定型）区分，由后端返回
/// 的距离在 HTTP 边界一次性换算（内核各后端统一「距离越小越近」）：
/// Score semantics depend on the collection's **metric** (fixed at creation); the
/// backend distance is converted once at the HTTP edge (the kernel is uniformly
/// "smaller distance = closer"):
///
/// - `cosine`：余弦相似度 `score = 1.0 - cosine_distance`，取值 `[-1, 1]`，值越大
///   越相似。注意单位范数只保证上界 `[-1, 1]`；要进一步落在 `[0, 1]` 还需向量
///   元素全部非负。HTTP 边缘会把 f32 舍入导致的轻微越界钳制回 `[-1, 1]`。
///   EN: cosine similarity, range [-1,1], larger = closer; unit norm alone only
///   guarantees [-1,1] (non-negative elements are required for [0,1]); rounding
///   overshoot is clamped back to [-1,1] at the edge.
/// - `l2`：距离是平方欧氏距离（L2²）；`score = 1.0 - L2²`，**不保证有界**（越负越远）。
///   对模长不同的向量，排序与余弦不同（这是度量语义的直接结果）。
///   EN: squared Euclidean distance; `score = 1.0 - L2²`, unbounded (more negative =
///   farther). Ranking differs from cosine for differing magnitudes by design.
/// - `l1`：距离是曼哈顿距离 Σ|a-b|；`score = 1.0 - L1`，**不保证有界**。
///   EN: Manhattan distance Σ|a-b|; `score = 1.0 - L1`, unbounded.
/// - `ip`：内核距离为 `-dot`，因此 `score = -distance = dot(a, b)`，即原始内积，
///   **不保证有界**（越大越相似，与 Milvus 的 IP 语义一致）。
///   EN: kernel distance is `-dot`, so `score = -distance = dot(a, b)` — the raw inner
///   product, unbounded (larger = closer, matching Milvus IP semantics).
///
/// 非有限距离（NaN/±Inf，可由 pg_wire binary float4/float8 写入）会在边缘成对丢弃，
/// 因此结果可能少于 k；目前该计数只打到服务端 stderr（后续可用机器可读字段透出）。
/// EN: non-finite distances (NaN/±Inf, storable via pg_wire binary floats) are dropped
/// pairwise at the edge, so results may number fewer than k; the count is currently
/// logged server-side only (a machine-readable field is a follow-up).
///
/// 空结果时 `ids` 与 `scores` 均为空数组。换算只在 HTTP 边缘做一次，内核各层始终
/// 保持「距离越小越近」的统一语义（WHY: 避免距离/相似度两种约定在内核里混用导致
/// 排序方向分歧）。
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub ids: Vec<u64>,
    /// 相似度分数，与 `ids` 等长对齐，非递增（单调不增）排列（越大越相似）。
    /// 换算按 collection 度量：cosine → `1-d` 且钳制 `[-1,1]`；l2/l1 → `1-d`；
    /// ip → `-d`（原始内积）。后三者不保证有界。详见结构体文档。
    /// Similarity scores aligned with `ids`, ordered non-increasing (larger = closer).
    /// By metric: cosine → `1-d` clamped to [-1,1]; l2/l1 → `1-d`; ip → `-d`
    /// (raw inner product). The latter three are unbounded. See the struct docs.
    pub scores: Vec<f32>,
    /// 每命中的 payload（label + payload 键值），键为 `ids` 元素的字符串形式。
    /// 仅当请求 `include_payload=true` 且为 collection 路径时存在；命中行无标量
    /// （裸向量）或已被删除时该 id 缺席。None 时整个字段不序列化（benchmark 响应
    /// 形状与旧版逐字节一致）。
    /// EN: Per-hit payloads (label + payload keys) keyed by the stringified id,
    /// present only for collection searches with `include_payload=true`. Ids with
    /// no scalar row (bare vectors) or already deleted are absent. `None` skips
    /// serialization entirely (benchmark responses stay byte-identical).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payloads: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 本次检索所用度量（l2|cosine|ip|l1），collection/hybrid 路径返回；benchmark
    /// 路径省略该字段以保持响应形状与旧版逐字节一致。
    /// EN: the metric used for this search (l2|cosine|ip|l1) on the collection/hybrid
    /// paths; the benchmark path omits it to keep responses byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OptimizeResponse {
    pub flushed_items: usize,
    pub pending_items: usize,
    pub hnsw_parallel_batch_rounds: u32,
    pub hnsw_serial_batch_rounds: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct IndexStatusResponse {
    pub hot_count: usize,
    pub indexed_count: usize,
    pub pending_count: usize,
    pub flush_count: usize,
    pub total_flushed_items: usize,
    pub max_pending_queue: usize,
    pub last_flush_items: usize,
    // 生效的索引配置：客户端（如 VectorDBBench）在测量前核对，避免把
    // 服务端实际参数与请求参数不一致的跑分当成功。
    // EN: Effective index configuration so clients (e.g. VectorDBBench) can
    // verify the graph they are about to measure instead of trusting labels.
    pub distance_metric: String,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
}

#[derive(Debug, Deserialize)]
pub struct WalCheckpointRequest {
    #[serde(default)]
    pub through_lsn: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct WalCheckpointResponse {
    pub removed_entries: usize,
    pub next_lsn: u64,
    pub through_lsn: u64,
    pub data_loss_on_restart_without_snapshot: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct WalRotateResponse {
    pub archived_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WalStatusResponse {
    pub next_lsn: u64,
    pub applied_lsn: u64,
    pub active_bytes: u64,
    pub archived_segments: usize,
    /// P5.0.3：collections WAL 的段/快照/保留视图。
    /// EN: P5.0.3 collections WAL segment/snapshot/retention view.
    pub collections: CollectionsWalStatus,
}

#[derive(Debug, Serialize)]
pub struct CollectionsWalStatus {
    pub first_available_lsn: u64,
    pub next_lsn: u64,
    pub active_bytes: u64,
    pub archived_segments: usize,
    pub archived_bytes: u64,
    pub oldest_archive_first_lsn: Option<u64>,
    pub newest_archive_last_lsn: Option<u64>,
    pub manifest_present: bool,
    pub manifest_seq: Option<u64>,
    pub safe_checkpoint_lsn: Option<u64>,
    pub retention_safe_floor_lsn: Option<u64>,
    pub retention_blocked_reason: Option<String>,
    pub retention_lease_until_unix: u64,
    pub segments_deleted_total: u64,
    pub bytes_freed_total: u64,
}

#[derive(Debug, serde::Deserialize, Default)]
pub struct CollectionsCompactRequest {
    #[serde(default)]
    pub rotate: bool,
    #[serde(default)]
    pub hold_retention_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct CollectionsCompactResponse {
    pub manifest_committed: bool,
    pub manifest_seq: Option<u64>,
    pub safe_checkpoint_lsn: Option<u64>,
    pub blocked_reason: Option<String>,
    pub rotated: bool,
    pub retention: RetentionResponse,
}

#[derive(Debug, Serialize)]
pub struct RetentionResponse {
    pub deleted_segments: usize,
    pub bytes_freed: u64,
    pub first_available_lsn: u64,
    pub blocked_reason: Option<String>,
    pub lease_until_unix: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct VectorSnapshotWriteResponse {
    pub checkpoint_lsn: u64,
    pub vector_count: usize,
    pub bytes_written: usize,
}

/// 恒定时间比较（认证路径防时序侧信道；长度对发送方本非秘密，不等直接 false）。
/// Constant-time byte compare for auth paths (HTTP bearer / pgwire password).
///
/// `black_box` 冻积累加器（OCR 359d85d[9]）：LLVM 在法律上有权把 XOR-fold
/// 重写为数据依赖的提前退出——没有屏障时"恒定时间"只是编译器心情。
/// 不引 subtle crate：为零依赖取舍，black_box 已能阻断该优化通道。
/// EN: The accumulator is frozen with `black_box` — without a barrier LLVM may
/// legally turn the XOR-fold into a data-dependent early exit, making the
/// "constant time" guarantee compiler-dependent. Zero-dependency alternative
/// to the subtle crate.
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let acc = a
        .iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(acc) == 0
}

#[derive(Debug, Serialize)]
pub(crate) struct HealthResponse {
    pub status: String,
}

/// `/readyz` 响应（P4.0.5）：与 openapi.json 的命名组件
/// `#/components/schemas/ReadyzResponse` 对应——用类型而不是手工 json! 组装；
/// openapi 守卫测试断言组件字段集与本结构体字段一致，双向漂移都会爆炸。
/// Typed /readyz body mirroring the named OpenAPI component schema; the spec
/// guard test asserts the field sets stay in lockstep.
// pub：benchmark_readyz 是 pub handler（bin target 集成测试跨 crate 调用），
// 返回类型不能比函数更私有。
#[derive(Debug, Serialize)]
pub struct ReadyzResponse {
    pub status: &'static str,
    pub pending_build: usize,
    pub hot_buffer: usize,
    pub indexed: usize,
    pub watermark: usize,
}

/// 慢查询判定（P4.3.2）：阈值 0 = 禁用（opt-in），启用后 elapsed >= threshold 即触发。
/// EN: Slow-query check; 0 disables it (opt-in), otherwise elapsed >= threshold fires.
pub fn slow_query_exceeded(elapsed_ms: u64, threshold_ms: u64) -> bool {
    threshold_ms > 0 && elapsed_ms >= threshold_ms
}

/// 慢查询日志的 SQL 清洗：截 256 字节（回退到 char 边界）+ 控制字符转义。
/// 日志不整段吐 SQL（防刷屏/防把大 payload 写进日志）；换行/制表/其他控制
/// 字符必须转义——SQL 是客户端可控输入，原样进日志即成日志注入（伪造日志行、
/// 污染采集管道，OCR 批次三）。无控制字符且未超限时零拷贝借用。
/// EN: Sanitize SQL for warn logs — truncate at 256 bytes (char-boundary) and
/// escape control characters; raw client-controlled SQL in logs is log
/// injection. Zero-copy borrow when clean and short.
pub fn sanitize_sql_for_log(sql: &str) -> std::borrow::Cow<'_, str> {
    const MAX: usize = 256;
    let truncated = if sql.len() <= MAX {
        sql
    } else {
        let mut end = MAX;
        while !sql.is_char_boundary(end) {
            end -= 1;
        }
        &sql[..end]
    };
    if !truncated.chars().any(|c| c.is_control()) {
        return std::borrow::Cow::Borrowed(truncated);
    }
    let mut out = String::with_capacity(truncated.len());
    for c in truncated.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    std::borrow::Cow::Owned(out)
}

#[cfg(test)]
mod slow_query_tests {
    use super::*;

    /// 阈值 0 禁用；启用后边界（恰好等于阈值）即触发。
    /// EN: 0 disables; at exactly the threshold it fires.
    #[test]
    fn slow_query_threshold_semantics() {
        assert!(!slow_query_exceeded(u64::MAX, 0));
        assert!(!slow_query_exceeded(99, 100));
        assert!(slow_query_exceeded(100, 100));
        assert!(slow_query_exceeded(101, 100));
    }

    /// 截断不 panic：ASCII 整截；多字节字符回退到 char 边界。
    /// EN: ASCII truncates whole; multibyte backs off to the char boundary.
    #[test]
    fn truncate_sql_respects_char_boundary() {
        let short = "select 1";
        assert_eq!(sanitize_sql_for_log(short), short);

        let long_ascii = "a".repeat(300);
        assert_eq!(sanitize_sql_for_log(&long_ascii).len(), 256);

        // '汉' 3 字节：让第 256 字节落在字符中间，必须回退。
        // EN: '汉' is 3 bytes: the 256th byte lands mid-character, so fallback is required.
        let mut s = "x".repeat(255);
        s.push('汉');
        s.push_str(&"y".repeat(100));
        let truncated = sanitize_sql_for_log(&s);
        assert_eq!(truncated.len(), 255, "must back off before the 3-byte char");
        assert!(truncated.ends_with('x'));
    }

    /// 控制字符转义：换行/回车/制表转义为可见序列，其余控制字符压成空格——
    /// 客户端 SQL 不得在日志里伪造新行（日志注入，OCR 批次三）。
    /// EN: Newlines/CR/TAB escape to visible sequences, other control chars
    /// collapse to spaces — client SQL must not forge log lines.
    #[test]
    fn sanitize_sql_escapes_control_chars() {
        // 注意必须用 \u{1}：'\\x01' 是字面四个字符（反斜杠+x01），会让
        // 控制字符断言沦为 vacuous（OCR 96bed7e）。
        // EN: Must use \u{1}: '\\x01' is four literal characters (backslash+x01),
        // EN: which would make the control-character assertion vacuous (OCR 96bed7e).
        let dirty = "select *\nfrom t\r\nwhere x =\t '\u{1}\u{7f}'";
        // U+0001/U+007F 不属于 \n\r\t 转义臂，走「其余控制字符压空格」臂。
        // EN: U+0001/U+007F are outside the \n\r\t escape arm; the other-controls arm strips them.
        let cleaned = sanitize_sql_for_log(dirty);
        assert!(!cleaned.contains('\n'), "raw newline must not survive");
        assert!(!cleaned.contains('\r'));
        assert!(!cleaned.contains('\t'));
        assert!(
            !cleaned.chars().any(|c| c.is_control()),
            "no control char may survive: {cleaned:?}"
        );
        assert!(cleaned.contains("\\n"), "escaped newline marker expected");
        assert!(cleaned.contains("\\t"));
        // 干净输入零拷贝借用。
        // EN: Clean input borrows without copying.
        let clean = "select 1";
        assert!(matches!(
            sanitize_sql_for_log(clean),
            std::borrow::Cow::Borrowed(_)
        ));
    }
}

#[cfg(test)]
mod inflight_watermark_tests {
    use super::*;

    /// 水位线原语（P4.2.3）：占满后拒绝并计数，归还后可再占。
    /// EN: Acquire up to max, overflow rejected + counted, release re-enables.
    #[test]
    fn watermark_acquire_reject_release_cycle() {
        let wm = Arc::new(InflightWatermark::new(2));
        let p1 = wm.try_acquire().expect("slot 1");
        let p2 = wm.try_acquire().expect("slot 2");
        assert_eq!(wm.active(), 2);

        assert!(wm.try_acquire().is_none(), "third acquire must be rejected");
        assert!(wm.try_acquire().is_none());
        assert_eq!(wm.rejected_total(), 2);
        assert_eq!(wm.active(), 2, "rejection must not leak gauge");

        drop(p1);
        assert_eq!(wm.active(), 1);
        let _p3 = wm.try_acquire().expect("released slot re-acquirable");
        assert_eq!(wm.active(), 2);
        drop(p2);
    }
}
