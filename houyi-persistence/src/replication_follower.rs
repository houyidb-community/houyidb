//! 复制 follower：常驻流接收 + group commit 落盘 + 幂等 apply。
//! Replication follower: persistent stream receive, group-commit durability,
//! idempotent apply.
//!
//! 关键结构决策 / Key decisions:
//! - **保持 leader LSN**：collections.wal 无截断、LSN 单调无洞，follower 用
//!   同一 WalAsyncHandle 落盘 → 本地文件即复制进度（免新侧车）→ promote
//!   后 LSN 连续、句柄零切换。
//! - **group commit**：接收线程攒批 fire-and-forget N 条 AppendEntry + 一条
//!   Fsync 等回（FIFO 保证顺序，一次 syscall 覆盖整批）；批内失败经
//!   Fsync 上报 → 断连重 catch-up。
//! - **幂等吸收**：lsn < expected 跳过仍 ACK（leader「先注册后 catch-up」
//!   的重叠推送）；lsn > expected = 空洞，断连重 catch-up 自愈。
//! - **ACK 时机**：帧解析后立即 ACK(RECEIVED)；fsync+apply 完 ACK(DURABLE)；
//!   HEARTBEAT 也回 ACK(RECEIVED, 已收最大 lsn)——leader 的 ack_timeout
//!   活性契约要求空闲链路也有 ACK 流，否则零写入部署每 5s 被掐线重连。
//! - **退出兜底 drain**：session 结束（断连/shutdown）时 pending 已 append
//!   进本地 WAL（next_lsn 已越过），不 apply 会造成运行态内存与本地 WAL
//!   分叉（重启冷重放才愈合）——退出前尽力 fsync+apply 收尾。
//! - **follower_ahead**（resume > leader_next，脑裂信号）：ERROR kind 字节
//!   /WELCOME 双向识别（结构化字段，不靠消息文本匹配）→ error 日志 +
//!   无限退避 + 指标，人工处置（v1 无自动demote）。

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(not(loom))]
use crate::net_runtime as runtime;
#[cfg(not(loom))]
use crate::net_runtime::TcpStream;
use crate::wal_async::WalAsyncHandle;
use crate::wal_replication as proto;
#[cfg(not(loom))]
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// 阻塞调用的并发预算（P4.6.2）：复制会话里的 WAL 操作是到 std wal worker
/// 的**阻塞通道往返**，必须离开 tokio worker（`spawn_blocking`），否则慢盘/
/// 排队时会把 runtime 线程钉住，影响同 runtime 上的其它任务。信号量上限 64
/// 给所有会话的阻塞面兜底（单 follower 同一时刻至多 1 个在途）。
/// EN: WAL calls are blocking round-trips to the std wal worker; they run on
/// `spawn_blocking` under a 64-permit budget so a slow disk cannot pin tokio
/// worker threads.
static BLOCKING_BUDGET: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();

/// 在阻塞池执行一个 WAL 调用（受全局预算限流）。
/// Run a blocking WAL call on the blocking pool under the global budget.
async fn blocking_wal<T, F>(f: F) -> io::Result<T>
where
    F: FnOnce() -> io::Result<T> + Send + 'static,
    T: Send + 'static,
{
    let sem = BLOCKING_BUDGET.get_or_init(|| tokio::sync::Semaphore::new(64));
    let _permit = sem
        .acquire()
        .await
        .map_err(|_| io::Error::other("blocking budget closed"))?;
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| io::Error::other(format!("blocking join failed: {e}")))?
}

/// 已 fsync 记录的应用回调（server 侧注入 CollectionsApplySink）。
/// 毒行由实现侧跳过+计数；解码硬错返回 Err——该记录已随批 fsync 落盘、
/// next_lsn 已越过，重连 resume 不会重发（错误不自旋），但运行态 apply
/// 缺口要等重启冷重放才愈合，故计 `apply_failures_total` + error 日志告警。
/// apply 回调（P4.5.5 起为 async：follower 已是 tokio task，回调里不得再
/// `block_on` 一个 runtime——那会 panic「Cannot start a runtime from within a
/// runtime」）。实现方直接在回调里 await 即可。
/// EN: async since P4.5.5 — the follower runs inside tokio, so an implementation
/// must not `block_on` a runtime (that panics).
#[async_trait::async_trait]
pub trait ReplicationApply: Send + Sync {
    async fn apply(&self, lsn: u64, payload: &[u8]) -> io::Result<()>;
}

#[derive(Default)]
pub struct FollowerMetrics {
    pub applied_entries_total: AtomicU64,
    /// 日志分叉拒绝次数（同 resume_lsn 但指纹不一致，人工处置）。
    /// Divergence rejections (equal length, different content; manual action).
    pub diverged_rejections_total: AtomicU64,
    /// 认证拒绝次数（token 不匹配，配置错误；P4.6.3）。
    /// Auth rejections (token mismatch; configuration error).
    pub auth_rejections_total: AtomicU64,
    /// 本地日志故障次数（指纹前置失败：损坏/截断/不连续；P4.6.3 OCR）。
    /// Local-log failures (fingerprint precondition: corrupt/truncated/gapped).
    pub local_log_errors_total: AtomicU64,
    /// apply 硬错（记录已落本地 WAL、运行态缺口重启愈合）——必须保持可见。
    /// EN: Hard apply errors (record already in the local WAL; the runtime gap
    /// heals on restart) — must stay visible.
    pub apply_failures_total: AtomicU64,
    /// apply 原地重试次数（P4.6.3）：瞬时错误在会话内自愈的路径；持续增长
    /// 说明 apply 面在恶化。EN: in-place apply retries (transient-error self-heal).
    pub apply_retries_total: AtomicU64,
    pub fsync_total: AtomicU64,
    pub reconnects_total: AtomicU64,
    pub follower_ahead_rejections_total: AtomicU64,
    /// 落后于 leader 保留范围的拒绝次数（人工全量重同步）。
    /// EN: rejections because local history is behind the leader's retained
    /// range (manual full resync required).
    pub behind_retained_rejections_total: AtomicU64,
    /// 本地下一条期望 LSN（= 已连续落盘的最大 lsn + 1）。
    /// EN: Next expected local LSN (= highest contiguously persisted lsn + 1).
    pub recv_lsn: AtomicU64,
    /// 已 fsync+apply 的最大 LSN。
    /// EN: Highest LSN fsynced and applied.
    pub durable_lsn: AtomicU64,
    /// HEARTBEAT 观测到的 leader_next_lsn（lag = leader_next - recv_lsn）。
    /// EN: leader_next_lsn observed via HEARTBEAT (lag = leader_next - recv_lsn).
    pub leader_next_lsn: AtomicU64,
    /// WELCOME 观测到的 leader 最早可读 LSN（保留基线可见性）。
    /// EN: leader first-available LSN observed via WELCOME (retention visibility).
    pub leader_first_available_lsn: AtomicU64,
}

pub struct FollowerConfig {
    /// 单批最多记录数（达到即 fsync+apply）。
    /// EN: Maximum records per batch (reaching it triggers fsync+apply).
    pub batch_max: usize,
    /// 读超时（超过即断连重连；TCP 建连单地址超时复用此值）。
    /// EN: Read timeout (exceeding it disconnects and reconnects; the TCP
    /// single-address connect timeout reuses this value).
    pub read_timeout: Duration,
    /// DNS 解析超时（P4.6.3）。与 connect 分离：系统解析器可能被吞包钉住
    /// 分钟级，必须单独设预算。
    /// EN: DNS resolution budget, separate from connect (the system resolver can
    /// hang for minutes on dropped packets).
    pub dns_timeout: Duration,
    /// 握手阶段预算（P4.6.3）：leader 在回 WELCOME 前要扫一遍日志指纹，
    /// 大 WAL 上秒级；不能用 read_timeout（5s）否则 follower 反复超时。
    /// EN: handshake budget — the leader scans its log for the fingerprint
    /// before WELCOME, which takes seconds on large WALs.
    pub handshake_timeout: Duration,
    /// 重连初始退避（指数 ×2 至 max）。
    /// EN: Initial reconnect backoff (doubles up to max).
    pub backoff_initial: Duration,
    pub backoff_max: Duration,
    /// follower_ahead 无限退避间隔（人工处置前的重试节奏）。
    /// EN: follower_ahead infinite-backoff retry interval (pending manual action).
    pub ahead_retry_interval: Duration,
    /// 复制认证共享秘密（P4.6.3）：必须与 leader 配置一致；None = 空 token。
    /// EN: shared secret; must match the leader. None sends the empty token.
    pub auth_token: Option<String>,
    /// 本地 WAL 静态加密句柄（P4.7.3）：指纹计算需要先解密本机段。
    /// EN: local WAL encryption handle; the local fingerprint needs decryption.
    pub encryption: Option<Arc<crate::encryption::Encryption>>,
    /// 稳定 follower 身份（P5.0.4）：跨重连保留 retention grace；服务器从
    /// 数据目录的 `replication.follower_id` 载入/生成，默认全零仅供测试。
    /// EN: stable follower identity (P5.0.4) so retention grace survives
    /// reconnects; the server loads/generates it from the data directory, and
    /// the all-zero default is for tests only.
    pub follower_id: [u8; 16],
    /// 本地 manifest checkpoint 回调（0 = 无 manifest）。每次握手读取最新值，
    /// leader 用它区分「快照已覆盖」与「需要重放」的尾部。
    /// EN: local manifest checkpoint callback (0 = no manifest), read on every
    /// handshake so the leader can distinguish snapshot-covered tails.
    pub snapshot_checkpoint: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// 落后于保留范围时的退避（人工重同步前不自动清库）。
    /// EN: backoff when behind the retained range; never auto-wipe before a
    /// manual resync.
    pub behind_retained_retry_interval: Duration,
}

impl FollowerConfig {
    /// 生产装配的唯一入口（与 `LeaderSessionConfig::for_replication` 同纪律）：
    /// 安全字段收进构造函数，`..Default::default()` 的静默丢失在结构上不可
    /// 发生。计时参数仍可用字面量在测试里覆盖。
    /// EN: single production constructor — security fields cannot be silently
    /// dropped by a `..Default::default()` literal.
    pub fn for_replication(
        auth_token: Option<String>,
        encryption: Option<Arc<crate::encryption::Encryption>>,
    ) -> Self {
        Self {
            auth_token,
            encryption,
            ..Self::default()
        }
    }

    /// 注入稳定 follower 身份与本地 manifest checkpoint（P5.0.4）。
    /// EN: injects the stable follower identity and local manifest checkpoint (P5.0.4).
    pub fn with_follower_identity(
        mut self,
        follower_id: [u8; 16],
        snapshot_checkpoint: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        self.follower_id = follower_id;
        self.snapshot_checkpoint = snapshot_checkpoint;
        self
    }
}

impl Default for FollowerConfig {
    fn default() -> Self {
        Self {
            batch_max: 32,
            read_timeout: Duration::from_secs(5),
            dns_timeout: Duration::from_secs(5),
            handshake_timeout: Duration::from_secs(30),
            backoff_initial: Duration::from_millis(100),
            backoff_max: Duration::from_secs(5),
            ahead_retry_interval: Duration::from_secs(30),
            auth_token: None,
            encryption: None,
            follower_id: [0u8; 16],
            snapshot_checkpoint: Arc::new(|| 0),
            behind_retained_retry_interval: Duration::from_secs(60),
        }
    }
}

/// session 终止原因分类。
/// EN: Session termination classification.
#[cfg(not(loom))]
enum SessionEnd {
    /// 网络/协议错误：退避后重连。
    /// EN: Network/protocol error: reconnect after backoff.
    Retry(io::Error),
    /// follower_ahead（脑裂信号）：无限退避 + 人工处置。
    /// EN: follower_ahead (split-brain signal): infinite backoff + manual action.
    FollowerAhead(String),
    /// 日志分叉（同长度不同内容）：无限退避 + 人工处置（清目录重同步）。
    /// EN: Log divergence (equal length, different content): infinite backoff +
    /// manual action (wipe and resync).
    Diverged(String),
    /// 认证失败（token 不匹配）：配置错误，长退避 + 人工修正。
    /// EN: Auth failure (token mismatch): configuration error, long backoff + manual fix.
    Auth(String),
    /// 本地日志故障（指纹无法计算）：重试无法自愈，长退避 + 人工处置。
    /// EN: local log cannot produce a fingerprint — retries cannot heal it.
    LocalLog(String),
    /// 落后于 leader 的保留范围：人工全量重同步，绝不自动清库。
    /// EN: behind the leader's retained range — manual full resync; never wipe
    /// automatically.
    BehindRetained(String),
    /// 外部 shutdown。
    /// EN: External shutdown.
    Stopped,
}

#[cfg(not(loom))]
pub struct ReplicationFollower {
    shutdown: Arc<AtomicBool>,
    handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// 优雅退出预算：必须 > read_timeout，让 frame_loop 在阻塞读超时后看到
    /// 关停标志并走 `session_inner` 的退出兜底 drain（fsync+apply 残留批）。
    /// EN: Graceful-exit budget: must exceed read_timeout so frame_loop sees the
    /// stop flag after a blocking-read timeout and runs `session_inner`'s exit
    /// drain (fsync+apply the pending batch).
    graceful_stop: Duration,
}

#[cfg(not(loom))]
impl ReplicationFollower {
    /// 起常驻复制 task（必须在 tokio 运行时上下文中调用）。
    /// EN: spawns the long-running replication task; requires a tokio runtime.
    pub fn start(
        leader_addr: String,
        wal: WalAsyncHandle,
        apply: Arc<dyn ReplicationApply>,
        metrics: Arc<FollowerMetrics>,
        config: FollowerConfig,
    ) -> io::Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let graceful_stop = config.read_timeout + Duration::from_secs(1);
        let handle = {
            let shutdown = Arc::clone(&shutdown);
            tokio::spawn(async move {
                run_loop(leader_addr, wal, apply, metrics, config, shutdown).await
            })
        };
        Ok(Self {
            shutdown,
            handle: Mutex::new(Some(handle)),
            graceful_stop,
        })
    }

    /// 关停：**先协作、后 abort**。
    ///
    /// 为什么不能直接 abort（WHY）：apply 是 async，abort 可能在单条记录 apply
    /// 到一半时切断——记录已随批 fsync 进本地 WAL，但内存态停在半途；promote
    /// 随即以 leader 身份对外服务，这条已持久化的记录在运行态不可见（要等重启
    /// 冷重放）。先置标志让 frame_loop 在下一次阻塞读超时（≤ read_timeout）后
    /// 退出并执行 `session_inner` 的兜底 drain，再以 `graceful_stop` 为预算等待；
    /// 只有任务卡死（例如 writer 卡在对端）才 abort。
    /// EN: abort can slice an apply in half; the record is durable in the local
    /// WAL but invisible in memory, and promote would serve as leader with that
    /// gap. Signal first, let the frame loop exit at its next read timeout and
    /// run the exit drain; abort only if it wedges.
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // guard 不得跨 await 持有：先 take（语句结束即释锁）再 await。
        // EN: The guard must not be held across await: take first (lock released
        // at statement end) and await after.
        let handle = self.handle.lock().unwrap().take();
        if let Some(mut h) = handle {
            // timeout 借用 `&mut h`：超时后仍持有句柄，可以 abort。
            // EN: timeout borrows `&mut h`: the handle is still held after a timeout,
            // so it can be aborted.
            if runtime::timeout(self.graceful_stop, &mut h).await.is_err() {
                h.abort();
                let _ = runtime::timeout(Duration::from_secs(5), h).await;
            }
        }
    }
}

#[cfg(not(loom))]
async fn run_loop(
    leader_addr: String,
    wal: WalAsyncHandle,
    apply: Arc<dyn ReplicationApply>,
    metrics: Arc<FollowerMetrics>,
    config: FollowerConfig,
    shutdown: Arc<AtomicBool>,
) {
    let mut backoff = config.backoff_initial;
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        let session_started = Instant::now();
        match run_session(&leader_addr, &wal, &*apply, &metrics, &config, &shutdown).await {
            SessionEnd::Stopped => return,
            SessionEnd::Retry(e) => {
                metrics.reconnects_total.fetch_add(1, Ordering::Relaxed);
                // 健康 session（存活 ≥ backoff_max）失败后退避重置：一段抖动期
                // 顶到 5s 上限后，不该让未来每次单击闪断都永久睡满上限。
                // EN: Reset backoff after a healthy session (lifetime >= backoff_max)
                // fails: after a jittery spell hits the 5s cap, future single blips
                // must not sleep the full cap forever.
                if session_started.elapsed() >= config.backoff_max {
                    backoff = config.backoff_initial;
                }
                tracing::warn!(error = %e, backoff_ms = backoff.as_millis() as u64,
                    "replication follower: session ended, reconnecting");
                jittered_sleep(backoff, &shutdown).await;
                backoff = (backoff * 2).min(config.backoff_max);
            }
            SessionEnd::FollowerAhead(msg) => {
                metrics
                    .follower_ahead_rejections_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    msg = %msg,
                    "replication follower: FOLLOWER AHEAD (split-brain signal) — \
                     refusing to replicate; manual intervention required \
                     (wipe data dir and re-sync, or promote)"
                );
                jittered_sleep(config.ahead_retry_interval, &shutdown).await;
            }
            SessionEnd::Diverged(msg) => {
                metrics
                    .diverged_rejections_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    msg = %msg,
                    "replication follower: LOG DIVERGED (same LSN, different content) — \
                     refusing to replicate; wipe this node's data dir and re-sync"
                );
                jittered_sleep(config.ahead_retry_interval, &shutdown).await;
            }
            SessionEnd::BehindRetained(msg) => {
                metrics
                    .behind_retained_rejections_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    msg = %msg,
                    "replication follower: BEHIND RETAINED — local history predates the \
                     leader's retained range; a full manual resync is required \
                     (see docs/runbooks/collections-retention-resync.md); never auto-wipe"
                );
                jittered_sleep(config.behind_retained_retry_interval, &shutdown).await;
            }
            SessionEnd::Auth(msg) => {
                metrics
                    .auth_rejections_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    msg = %msg,
                    "replication follower: AUTH REJECTED — replication_auth_token must \
                     match the leader's; retrying slowly until fixed"
                );
                jittered_sleep(config.ahead_retry_interval, &shutdown).await;
            }
            SessionEnd::LocalLog(msg) => {
                metrics
                    .local_log_errors_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    msg = %msg,
                    "replication follower: LOCAL LOG UNUSABLE (fingerprint failed) — \
                     retrying slowly; inspect/repair or wipe this node's data dir"
                );
                jittered_sleep(config.ahead_retry_interval, &shutdown).await;
            }
        }
    }
}

/// ±20% 抖动睡眠（免 rand 依赖：纳秒时间戳低位伪随机）。
///
/// tokio 化后不再需要 10ms 分段轮询 shutdown：task abort 会在 sleep 的 await
/// 点直接取消（这正是 wedged-writer/退避长睡泄漏的根治手段）。turmoil 下
/// `runtime::sleep` 走模拟时钟，退避不会拖慢仿真。
/// EN: ±20% jittered sleep (no rand dependency: low bits of the nanosecond
/// timestamp serve as pseudo-randomness). After the tokio conversion, 10ms
/// segmented shutdown polling is gone: task abort cancels directly at the sleep
/// await point (the root fix for wedged-writer/backoff long-sleep leaks). Under
/// turmoil, `runtime::sleep` uses the simulated clock, so backoff does not slow
/// simulation.
#[cfg(not(loom))]
async fn jittered_sleep(base: Duration, shutdown: &AtomicBool) {
    if shutdown.load(Ordering::Relaxed) {
        return;
    }
    // 抖动源必须是**确定性**的：turmoil 的模拟时钟只推进 sleep 本身，墙钟低位
    // 会让同一 seed 的重连时序逐次漂移，破坏 DST 的可复现性。用进程内计数器。
    // EN: deterministic jitter — wall-clock nanos would make the same turmoil
    // seed produce different reconnect timings.
    static JITTER_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = JITTER_SEQ.fetch_add(1, Ordering::Relaxed);
    let jitter_pct = (n % 41) as i64 - 20; // [-20, +20]
    let ms = base.as_millis() as i64 * (100 + jitter_pct) / 100;
    runtime::sleep(Duration::from_millis(ms.max(1) as u64)).await;
}

#[cfg(not(loom))]
async fn run_session(
    leader_addr: &str,
    wal: &WalAsyncHandle,
    apply: &dyn ReplicationApply,
    metrics: &FollowerMetrics,
    config: &FollowerConfig,
    shutdown: &Arc<AtomicBool>,
) -> SessionEnd {
    match session_inner(leader_addr, wal, apply, metrics, config, shutdown).await {
        Ok(()) => SessionEnd::Stopped,
        Err(SessionError::Retry(e)) => SessionEnd::Retry(e),
        Err(SessionError::Ahead(m)) => SessionEnd::FollowerAhead(m),
        Err(SessionError::Diverged(m)) => SessionEnd::Diverged(m),
        Err(SessionError::Auth(m)) => SessionEnd::Auth(m),
        Err(SessionError::LocalLog(m)) => SessionEnd::LocalLog(m),
        Err(SessionError::BehindRetained(m)) => SessionEnd::BehindRetained(m),
    }
}

#[cfg(not(loom))]
#[derive(Debug)]
enum SessionError {
    Retry(io::Error),
    Ahead(String),
    Diverged(String),
    Auth(String),
    LocalLog(String),
    BehindRetained(String),
}

#[cfg(not(loom))]
impl From<io::Error> for SessionError {
    fn from(e: io::Error) -> Self {
        SessionError::Retry(e)
    }
}

/// 拆分 `host:port`（DNS 名留给连接层解析）。turmoil 的仿真网络按 **host
/// 标签**寻址（`("leader", port)`），必须走元组形式；tokio 同样接受元组并
/// 对 host 做 DNS 解析——两侧行为一致。
/// EN: split host:port; the tuple form is what turmoil's simulated network
/// addresses, and tokio accepts it too (resolving host names as usual).
#[cfg(not(loom))]
fn split_host_port(addr: &str) -> io::Result<(&str, u16)> {
    let (host, port) = addr.rsplit_once(':').ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("leader addr '{addr}' is not host:port"),
        )
    })?;
    // IPv6 字面量 `[::1]:9000`：去掉方括号，交给连接层按地址解析。
    // EN: IPv6 literal `[::1]:9000`: strip the brackets and let the connection
    // layer resolve by address.
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    let port: u16 = port.parse().map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("bad port in '{addr}'"))
    })?;
    if host.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("empty host in '{addr}'"),
        ));
    }
    Ok((host, port))
}

/// 握手阶段的 leader 回包（WELCOME / ANCHOR / ERROR）。
/// EN: Leader replies during the handshake (WELCOME / ANCHOR / ERROR).
#[cfg(not(loom))]
enum HandshakeReply {
    Welcome(proto::Welcome),
    Anchor(u64),
    Error { kind: u8, msg: String },
}

/// 读一个完整握手回包（ERROR / ANCHOR / WELCOME），统一超时与长度校验。
/// EN: Reads one complete handshake reply with a shared timeout and length checks.
#[cfg(not(loom))]
async fn read_handshake_reply(
    stream: &mut TcpStream,
    budget: Duration,
) -> io::Result<HandshakeReply> {
    let mut head = [0u8; 4];
    runtime::timeout(budget, stream.read_exact(&mut head))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "handshake reply timeout"))??;
    if head[0] == proto::FRAME_ERROR {
        let kind = head[1];
        let msg_len = u16::from_be_bytes([head[2], head[3]]) as usize;
        if msg_len > proto::MAX_ERROR_MSG_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("error message too large: {msg_len}"),
            ));
        }
        let mut msg = vec![0u8; msg_len];
        runtime::timeout(budget, stream.read_exact(&mut msg))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "error-frame read timeout"))??;
        return Ok(HandshakeReply::Error {
            kind,
            msg: proto::decode_error_msg(&msg),
        });
    }
    match &head {
        b"ANCH" => {
            let mut full = [0u8; proto::ANCHOR_LEN];
            full[..4].copy_from_slice(&head);
            runtime::timeout(budget, stream.read_exact(&mut full[4..]))
                .await
                .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "anchor read timeout"))??;
            Ok(HandshakeReply::Anchor(proto::decode_anchor(&full)?))
        }
        b"WLCM" => {
            let mut full = [0u8; proto::WELCOME_LEN];
            full[..4].copy_from_slice(&head);
            runtime::timeout(budget, stream.read_exact(&mut full[4..]))
                .await
                .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "welcome read timeout"))??;
            Ok(HandshakeReply::Welcome(proto::decode_welcome(&full)?))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected handshake reply magic",
        )),
    }
}

/// 结构化 kind → 会话错误（禁止靠消息文本分类安全信号）。
/// EN: structured kind → session error (never classify safety signals by text).
#[cfg(not(loom))]
fn classify_leader_error(kind: u8, msg: String) -> SessionError {
    match kind {
        proto::ERROR_KIND_FOLLOWER_AHEAD => SessionError::Ahead(msg),
        proto::ERROR_KIND_DIVERGED => SessionError::Diverged(msg),
        proto::ERROR_KIND_AUTH => SessionError::Auth(msg),
        proto::ERROR_KIND_BEHIND_RETAINED => SessionError::BehindRetained(msg),
        _ => SessionError::Retry(io::Error::new(
            io::ErrorKind::ConnectionRefused,
            format!("leader rejected: {msg}"),
        )),
    }
}

#[cfg(not(loom))]
async fn session_inner(
    leader_addr: &str,
    wal: &WalAsyncHandle,
    apply: &dyn ReplicationApply,
    metrics: &FollowerMetrics,
    config: &FollowerConfig,
    shutdown: &Arc<AtomicBool>,
) -> Result<(), SessionError> {
    let resume_lsn = wal.next_lsn();
    metrics.recv_lsn.store(resume_lsn, Ordering::Relaxed);
    let first_available = wal
        .first_available_lsn()
        .map_err(|e| SessionError::LocalLog(format!("local first-available LSN failed: {e}")))?;
    if first_available > resume_lsn {
        return Err(SessionError::LocalLog(format!(
            "local first_available_lsn {first_available} > next_lsn {resume_lsn}"
        )));
    }

    // 先校验本地配置形态（零成本），再算本地前缀指纹。
    // EN: validate the address form first (free), then fingerprint.
    let (host, port) = split_host_port(leader_addr)?;

    // 本地日志范围指纹：**建连之前**在阻塞池算完（P4.6.3 OCR 修正）——
    // 若连上后再扫，扫描时长会吃掉 leader 的握手读窗口；且本地日志损坏
    // 属于本地永久故障，应与网络抖动区分（SessionError::LocalLog）。
    // v6 只哈希本地仍保留的 [log_start, resume) 区间。本机保留任务可能在
    // 采样 first_available 之后删掉最旧段：`Unavailable` 携带新的起点时
    // 用它重试一次，而不是把这种瞬时竞态误报成本地永久故障。
    // EN: compute the local range hash on the blocking pool *before*
    // connecting — scanning after connect eats the leader's handshake window,
    // and a corrupt local log is a local permanent fault, not a network flap.
    // v6 hashes only the locally retained [log_start, resume) range. Local
    // retention may delete the oldest segment after sampling first-available;
    // on `Unavailable` with a newer start, retry from that start instead of
    // misreporting a transient race as a permanent local fault.
    let mut log_start = first_available;
    // 本地保留可能在「列段 → 逐个打开」之间删掉/轮转段：Unavailable 携带新
    // 起点，NotFound/Corrupt 也可能只是竞态。两类都先重采样 floor，确认真的
    // 推进才重试；重试上限防止保留策略跑赢扫描时反复全量扫盘。
    // EN: Local retention may unlink/rotate a listed segment between listing and
    // opening: `Unavailable` carries a new start, and NotFound/Corrupt may also
    // be a race. Re-sample the floor for both and retry only when it actually
    // advanced; a cap bounds full-log rescans if retention outruns the scan.
    const MAX_FINGERPRINT_ATTEMPTS: usize = 3;
    let mut attempts = 0usize;
    let local_hash = loop {
        attempts += 1;
        match crate::log_fingerprint::range_hash_blocking_with_encryption(
            wal.path().to_path_buf(),
            log_start,
            resume_lsn,
            config.encryption.clone(),
        )
        .await
        {
            Ok(hash) => break hash,
            Err(err) => {
                let current = wal.first_available_lsn().unwrap_or(log_start);
                let advanced = current > log_start && current <= resume_lsn;
                if advanced && attempts < MAX_FINGERPRINT_ATTEMPTS {
                    tracing::warn!(
                        old_start = log_start,
                        new_start = current,
                        error = %err,
                        "replication follower: local retention advanced during hashing; retrying"
                    );
                    log_start = current;
                    continue;
                }
                return Err(SessionError::LocalLog(format!(
                    "local log fingerprint failed after {attempts} attempt(s): {err}"
                )));
            }
        }
    };

    // 解析与建连分开设预算（P4.6.3）：DNS 走独立 timeout（系统解析器被吞包
    // 时可能钉住分钟级）；每个解析地址的 TCP 建连复用 read_timeout。
    // EN: separate DNS and connect budgets (P4.6.3); per-address connect reuses
    // read_timeout.
    let addrs = match runtime::timeout(config.dns_timeout, runtime::resolve_host(host, port)).await
    {
        Ok(Ok(addrs)) if !addrs.is_empty() => addrs,
        Ok(Ok(_)) => {
            return Err(SessionError::Retry(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no addresses resolved for {leader_addr}"),
            )))
        }
        Ok(Err(e)) => return Err(SessionError::Retry(e)),
        Err(_) => {
            return Err(SessionError::Retry(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("dns resolve for {leader_addr} timed out"),
            )))
        }
    };
    // 总预算覆盖整轮地址尝试（OCR 批次二十八）：多 A/AAAA 记录 + 黑洞地址
    // 时，逐地址各给 read_timeout 会让重连延迟线性膨胀；剩余预算随 deadline
    // 收缩，最后一个地址可能只拿到零头（与旧版单次超时语义对齐）。
    // EN: one overall connect budget across all resolved addresses; per-address
    // timeouts share it instead of multiplying reconnect latency.
    let connect_deadline = Instant::now() + config.read_timeout;
    let mut stream = None;
    let mut last_err = None;
    for addr in addrs {
        let remaining = connect_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            // 保留先前真正的根因（通常是黑洞地址的单个超时）；预算消息只
            // 在没有任何失败记录时兜底（OCR 批次二十九）。
            // EN: keep the earlier root cause (the address that consumed the
            // budget); the budget message is a fallback only.
            if last_err.is_none() {
                last_err = Some(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("connect budget exhausted before {addr}"),
                ));
            }
            break;
        }
        match runtime::timeout(remaining, TcpStream::connect(addr)).await {
            Ok(Ok(s)) => {
                stream = Some(s);
                break;
            }
            Ok(Err(e)) => last_err = Some(e),
            Err(_) => {
                last_err = Some(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("connect to {addr} timed out"),
                ))
            }
        }
    }
    let mut stream = match stream {
        Some(s) => s,
        None => {
            return Err(SessionError::Retry(last_err.unwrap_or_else(|| {
                io::Error::new(
                    io::ErrorKind::ConnectionRefused,
                    format!("connect to {leader_addr} failed"),
                )
            })))
        }
    };
    let _ = stream.set_nodelay(true);

    // HANDSHAKE → WELCOME / ERROR（握手阶段用独立 handshake_timeout：
    // leader 侧也要扫一遍日志指纹，预算必须大于单次扫描而非 read_timeout）。
    // 参与方在 resume_lsn 处比对内容，长度相同也必须内容相同（P4.6.1）。
    // EN: handshake phase uses its own timeout — the leader also scans its log
    // for the fingerprint, so the budget must cover a scan, not one read.
    let handshake = proto::encode_handshake(
        resume_lsn,
        proto::STREAM_MASK_COLLECTIONS,
        local_hash,
        log_start,
        (config.snapshot_checkpoint)(),
        config.follower_id,
        config.auth_token.as_deref().unwrap_or(""),
    )?;
    runtime::timeout(config.handshake_timeout, stream.write_all(&handshake))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "handshake write timeout"))??;
    let mut reply = read_handshake_reply(&mut stream, config.handshake_timeout).await?;
    if let HandshakeReply::Anchor(anchor) = reply {
        // 锚点协商（P5.0.4）：leader 的 first_available 已越过本地最早可读点，
        // 从锚点重算范围指纹。无法计算时发 (0,0) 哨兵，leader 必判 BEHIND_RETAINED。
        // EN: anchor negotiation: the leader's first-available passed our earliest
        // readable LSN; recompute the fingerprint from the anchor. If it cannot be
        // computed, send the (0,0) sentinel so the leader rejects with
        // BEHIND_RETAINED.
        let anchored = if anchor < log_start || anchor > resume_lsn {
            proto::encode_handshake_anchored(0, 0)
        } else {
            match crate::log_fingerprint::range_hash_blocking_with_encryption(
                wal.path().to_path_buf(),
                anchor,
                resume_lsn,
                config.encryption.clone(),
            )
            .await
            {
                Ok(h) => proto::encode_handshake_anchored(anchor, h),
                Err(e) => {
                    tracing::warn!(anchor, error = %e,
                        "replication follower: anchor fingerprint unavailable; sending sentinel");
                    proto::encode_handshake_anchored(0, 0)
                }
            }
        };
        runtime::timeout(config.handshake_timeout, stream.write_all(&anchored))
            .await
            .map_err(|_| {
                io::Error::new(io::ErrorKind::TimedOut, "anchored handshake write timeout")
            })??;
        reply = read_handshake_reply(&mut stream, config.handshake_timeout).await?;
    }
    let welcome = match reply {
        HandshakeReply::Welcome(w) => w,
        HandshakeReply::Error { kind, msg } => return Err(classify_leader_error(kind, msg)),
        HandshakeReply::Anchor(_) => {
            return Err(SessionError::Retry(io::Error::new(
                io::ErrorKind::InvalidData,
                "leader sent a second ANCHOR after anchored handshake",
            )))
        }
    };
    metrics
        .leader_first_available_lsn
        .store(welcome.leader_first_available_lsn, Ordering::Relaxed);
    let leader_next = welcome.leader_next_lsn;
    metrics
        .leader_next_lsn
        .store(leader_next, Ordering::Relaxed);
    if resume_lsn > leader_next {
        // WELCOME 侧双向识别（leader 侧握手检查是主防线，此为兜底）。
        // EN: Two-way detection on the WELCOME side (the leader-side handshake check
        // is the primary defense; this is a backstop).
        return Err(SessionError::Ahead(format!(
            "follower ahead: resume_lsn {resume_lsn} > leader_next_lsn {leader_next}"
        )));
    }
    tracing::info!(leader = %leader_addr, resume_lsn, leader_next,
        "replication follower: connected");

    let mut expected = resume_lsn;
    let mut pending: Vec<(u64, Vec<u8>)> = Vec::new();
    let result = frame_loop(
        &mut stream,
        wal,
        apply,
        metrics,
        config,
        shutdown,
        &mut expected,
        &mut pending,
    )
    .await;
    // 退出兜底 drain：pending 记录已 append 进本地 WAL（next_lsn 已越过），
    // 不 apply 会造成运行态内存与本地 WAL 分叉（重启冷重放才愈合，promote
    // 更会带着缺口上线）。断流时末尾 DURABLE ACK 写失败属正常，apply 成果
    // 不受影响。
    // EN: Exit backstop drain: pending records were appended to the local WAL
    // (next_lsn already advanced); not applying them diverges runtime memory from
    // the local WAL (heals only via cold replay on restart, and promote would go
    // live with the gap). A failed trailing DURABLE ACK on stream loss is normal;
    // applied progress is unaffected.
    if !pending.is_empty() {
        if let Err(e) = flush_batch(&mut stream, wal, apply, metrics, &mut pending).await {
            tracing::warn!(error = ?e,
                "replication follower: final drain incomplete; unapplied entries heal via local-WAL cold replay on restart");
        }
    }
    result
}

#[cfg(not(loom))]
#[allow(clippy::too_many_arguments)]
async fn frame_loop(
    stream: &mut TcpStream,
    wal: &WalAsyncHandle,
    apply: &dyn ReplicationApply,
    metrics: &FollowerMetrics,
    config: &FollowerConfig,
    shutdown: &Arc<AtomicBool>,
    expected: &mut u64,
    pending: &mut Vec<(u64, Vec<u8>)>,
) -> Result<(), SessionError> {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut tag = [0u8; 1];
        // 读超时 = 断连判定（与旧 std `set_read_timeout` 同语义）：链路空闲
        // 超过 read_timeout 即重连，从本地 LSN 续传。
        // EN: Read timeout = disconnect decision (same semantics as the old std
        // `set_read_timeout`): idleness beyond read_timeout reconnects and resumes
        // from the local LSN.
        match runtime::timeout(config.read_timeout, stream.read_exact(&mut tag)).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err(SessionError::Retry(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "replication read timeout",
                )))
            }
        }
        let hlen = proto::frame_header_len(tag[0])?;
        let mut hbuf = vec![0u8; hlen];
        hbuf[0] = tag[0];
        runtime::timeout(config.read_timeout, stream.read_exact(&mut hbuf[1..]))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "frame header read timeout"))??;
        match proto::decode_frame_header(&hbuf)? {
            proto::FrameHeader::Data {
                stream: s,
                lsn,
                payload_len,
            } => {
                // v1 仅 stream 0：未知流的记录静默收下会写进错误的 WAL。
                // EN: v1 has only stream 0: silently accepting records from an unknown
                // stream would write them into the wrong WAL.
                if s != proto::STREAM_COLLECTIONS {
                    return Err(SessionError::Retry(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected stream {s} in DATA frame (v1 ships stream 0 only)"),
                    )));
                }
                let mut payload = vec![0u8; payload_len as usize];
                runtime::timeout(config.read_timeout, stream.read_exact(&mut payload))
                    .await
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::TimedOut, "data payload read timeout")
                    })??;
                // 帧解析完成 → 立即 ACK(RECEIVED)
                // EN: Frame parsed → immediately ACK(RECEIVED).
                let ack =
                    proto::encode_ack(proto::STREAM_COLLECTIONS, proto::ACK_STAGE_RECEIVED, lsn);
                let _ = runtime::timeout(config.read_timeout, stream.write_all(&ack)).await;
                if lsn < *expected {
                    continue; // 重叠推送：幂等跳过（已 ACK）
                }
                if lsn > *expected {
                    return Err(SessionError::Retry(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("lsn gap: got {lsn}, expect {expected}; reconnect for catch-up"),
                    )));
                }
                {
                    let wal = wal.clone();
                    let to_write = payload.clone();
                    blocking_wal(move || wal.append_entry_no_wait(lsn, to_write)).await?;
                }
                *expected += 1;
                metrics.recv_lsn.store(*expected, Ordering::Relaxed);
                pending.push((lsn, payload));
                if pending.len() >= config.batch_max {
                    flush_batch(stream, wal, apply, metrics, pending).await?;
                }
            }
            proto::FrameHeader::Heartbeat {
                stream: s,
                leader_next_lsn,
            } => {
                if s != proto::STREAM_COLLECTIONS {
                    return Err(SessionError::Retry(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected stream {s} in HEARTBEAT frame"),
                    )));
                }
                metrics
                    .leader_next_lsn
                    .store(leader_next_lsn, Ordering::Relaxed);
                // 低流量落地：心跳驱动在途批 fsync+apply（最大滞留 1 个心跳周期）。
                // EN: Low-traffic landing: heartbeats drive fsync+apply of the in-flight
                // batch (residue bounded by one heartbeat period).
                if !pending.is_empty() {
                    flush_batch(stream, wal, apply, metrics, pending).await?;
                }
                // 心跳回 ACK(RECEIVED, 已收最大 lsn)：leader 的 ack_timeout
                // 活性契约要求空闲链路也有 ACK 流——零写入部署若不回 ACK，
                // 每 5s 被 leader 掐线重连（重连风暴）。
                // EN: Heartbeat replies ACK(RECEIVED, max lsn received): the leader's
                // ack_timeout liveness contract requires ACK traffic even on idle links —
                // a zero-write deployment that never ACKs gets cut every 5s (storm).
                let ack = proto::encode_ack(
                    proto::STREAM_COLLECTIONS,
                    proto::ACK_STAGE_RECEIVED,
                    expected.saturating_sub(1),
                );
                let _ = runtime::timeout(config.read_timeout, stream.write_all(&ack)).await;
            }
            proto::FrameHeader::Error { kind, msg_len } => {
                let mut msg = vec![0u8; msg_len as usize];
                runtime::timeout(config.read_timeout, stream.read_exact(&mut msg))
                    .await
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::TimedOut, "error msg read timeout")
                    })??;
                let msg = proto::decode_error_msg(&msg);
                if kind == proto::ERROR_KIND_FOLLOWER_AHEAD {
                    return Err(SessionError::Ahead(msg));
                }
                if kind == proto::ERROR_KIND_DIVERGED {
                    return Err(SessionError::Diverged(msg));
                }
                if kind == proto::ERROR_KIND_AUTH {
                    return Err(SessionError::Auth(msg));
                }
                return Err(SessionError::Retry(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    format!("leader error frame: {msg}"),
                )));
            }
            proto::FrameHeader::Ack { .. } => {
                return Err(SessionError::Retry(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected ACK frame from leader",
                )));
            }
        }
    }
}

/// apply 原地重试预算：1 次原始尝试 + 3 次重试（100/200/400ms 退避）。
/// EN: one attempt plus three in-place retries (100/200/400 ms backoff).
const APPLY_MAX_RETRIES: u32 = 3;

/// 带原地重试的 apply（P4.6.3）。
///
/// `InvalidData` 是解码腐败（确定性错误），重试只会拖延会话终止——直接上抛。
/// 其余错误（内存压力、后端瞬时失败）视为可重试。
/// EN: `InvalidData` is deterministic decode corruption — surface immediately;
/// everything else is treated as transient and retried.
#[cfg(not(loom))]
async fn apply_with_retry(
    apply: &dyn ReplicationApply,
    lsn: u64,
    payload: &[u8],
    metrics: &FollowerMetrics,
) -> io::Result<()> {
    let mut attempt = 0u32;
    loop {
        match apply.apply(lsn, payload).await {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::InvalidData => return Err(e),
            Err(e) if attempt >= APPLY_MAX_RETRIES => return Err(e),
            Err(e) => {
                attempt += 1;
                metrics.apply_retries_total.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    lsn,
                    attempt,
                    error = %e,
                    "replication follower: transient apply failure, retrying in place"
                );
                runtime::sleep(Duration::from_millis(100u64 << (attempt - 1))).await;
            }
        }
    }
}

/// group commit 收尾：fsync（批内失败在此上报）→ 逐条 apply → ACK(DURABLE)。
///
/// 先 take 后 fsync（OCR 批次十七 HIGH 修正）：fsync 失败时本批必须随 Err
/// 一并丢弃，不得留在 pending 里——否则退出兜底 drain 会二次调用本函数，
/// 此时 pending_error 已被首次 fsync 消费、重试 fsync 成功，把「可能从未
/// 落盘」的批 apply 并 ACK DURABLE（append 失败不消耗 LSN，这些记录与新
/// session 重推的同号记录双重 apply，运行态分叉）。
/// 丢弃的条目分两类：失败前已写盘的（lsn < next_lsn）保持「durable 未
/// apply」与既有 apply 失败语义一致，重启冷重放愈合；未写盘的由重连后
/// leader 从 resume_lsn 重推补齐。两类都不会假 DURABLE。
/// EN: Take the batch *before* fsync so a failed fsync drops it with the Err
/// instead of leaving it in `pending` — otherwise the session-exit drain would
/// retry flush, succeed (pending_error already consumed), and ACK DURABLE for
/// records that may never have reached disk.
#[cfg(not(loom))]
async fn flush_batch(
    stream: &mut TcpStream,
    wal: &WalAsyncHandle,
    apply: &dyn ReplicationApply,
    metrics: &FollowerMetrics,
    pending: &mut Vec<(u64, Vec<u8>)>,
) -> Result<(), SessionError> {
    let batch = std::mem::take(pending);
    if batch.is_empty() {
        return Ok(());
    }
    {
        let wal = wal.clone();
        blocking_wal(move || wal.fsync()).await?;
    }
    metrics.fsync_total.fetch_add(1, Ordering::Relaxed);
    let mut max_lsn = 0u64;
    for (lsn, payload) in batch {
        // 注入点（P4.5.4）：记录已随批 fsync 落盘、apply 之前崩溃 —— 运行态
        // 出现 apply 缺口，只能靠重启冷重放愈合。演练据此验证「WAL 耐久但
        // 内存态滞后」时冷启动能把 follower 拉回与 leader 一致。
        // EN: the record is durable in the local WAL but not yet applied; only a
        // cold replay can heal the runtime gap. The drill validates that.
        crate::fail_point!("follower_apply_pre");
        // 原地重试（P4.6.3）：apply 失败后**不能**指望重连自愈——记录已随批
        // fsync 落盘、next_lsn 已越过，新 session 的 resume_lsn 不会重发它。
        // 旧行为直接把会话判死，瞬时错误因此变成「运行态缺口 + 重启才愈合」。
        // 现在对可重试错误（非解码腐败）原地重试至多 3 次，指数退避；耗尽的
        // 才按硬错结束会话。apply 实现需幂等（upsert/delete/create/set 均满足）。
        // EN: apply failures cannot heal via reconnect — the record is already
        // durable locally and resume_lsn moves past it. Retry retryable errors
        // in place (decode corruption stays fatal); exhausting retries ends the
        // session as before. Apply implementations are idempotent by contract.
        apply_with_retry(apply, lsn, &payload, metrics)
            .await
            .map_err(|e| {
                metrics.apply_failures_total.fetch_add(1, Ordering::Relaxed);
                tracing::error!(lsn, error = %e,
                    "replication follower: apply failed (record durable in local WAL; runtime gap heals via cold replay on restart)");
                SessionError::Retry(io::Error::new(
                    e.kind(),
                    format!("apply fatal at lsn {lsn}: {e}"),
                ))
            })?;
        max_lsn = max_lsn.max(lsn);
        metrics
            .applied_entries_total
            .fetch_add(1, Ordering::Relaxed);
    }
    metrics.durable_lsn.fetch_max(max_lsn, Ordering::Relaxed);
    // 注入点（P4.5.4）：DURABLE ACK 发送前崩溃/报错 —— leader 收不到持久确认，
    // full_sync 的等待必须降级为超时（而非谎报成功），且 follower 重启后
    // 靠本地 WAL 续传不丢已持久化记录。
    // EN: crash/error right before the DURABLE ack — the leader must degrade to a
    // timeout instead of reporting false durability; the follower must resume
    // from its local WAL without losing durable records.
    crate::fail_point!("repl_ack_send_pre");
    let ack = proto::encode_ack(proto::STREAM_COLLECTIONS, proto::ACK_STAGE_DURABLE, max_lsn);
    runtime::timeout(Duration::from_secs(10), stream.write_all(&ack))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "durable ack write timeout"))??;
    Ok(())
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::replication_leader::{
        AckWatermarks, LeaderSessionConfig, ReplicationLeader, ShipHub,
    };
    use crate::FileWal;
    use std::io::{Read, Write};

    struct RecordingApply {
        seen: Mutex<Vec<(u64, Vec<u8>)>>,
    }

    #[async_trait::async_trait]
    impl ReplicationApply for RecordingApply {
        async fn apply(&self, lsn: u64, payload: &[u8]) -> io::Result<()> {
            self.seen.lock().unwrap().push((lsn, payload.to_vec()));
            Ok(())
        }
    }

    /// 前 N 次失败、随后成功的 apply（验证原地重试）。
    /// Fails N times, then succeeds (in-place retry probe).
    struct FlakyApply {
        failures_left: AtomicU64,
        kind: io::ErrorKind,
        seen: Mutex<Vec<u64>>,
    }

    #[async_trait::async_trait]
    impl ReplicationApply for FlakyApply {
        async fn apply(&self, lsn: u64, _payload: &[u8]) -> io::Result<()> {
            let left = self.failures_left.load(Ordering::Relaxed);
            if left > 0 {
                self.failures_left.store(left - 1, Ordering::Relaxed);
                return Err(io::Error::new(self.kind, "injected apply failure"));
            }
            self.seen.lock().unwrap().push(lsn);
            Ok(())
        }
    }

    #[tokio::test]
    async fn apply_retries_transient_errors_in_place() {
        let apply = FlakyApply {
            failures_left: AtomicU64::new(2),
            kind: io::ErrorKind::Other,
            seen: Mutex::new(Vec::new()),
        };
        let metrics = FollowerMetrics::default();
        apply_with_retry(&apply, 7, b"rec", &metrics)
            .await
            .expect("two transient failures must be absorbed by retries");
        assert_eq!(apply.seen.lock().unwrap().as_slice(), &[7]);
        assert_eq!(metrics.apply_retries_total.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn apply_does_not_retry_decode_corruption() {
        let apply = FlakyApply {
            failures_left: AtomicU64::new(1),
            kind: io::ErrorKind::InvalidData,
            seen: Mutex::new(Vec::new()),
        };
        let metrics = FollowerMetrics::default();
        let err = apply_with_retry(&apply, 9, b"corrupt", &metrics)
            .await
            .expect_err("InvalidData is deterministic and must surface immediately");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            metrics.apply_retries_total.load(Ordering::Relaxed),
            0,
            "decode corruption must not be retried"
        );
        assert!(apply.seen.lock().unwrap().is_empty());
    }

    struct Rig {
        dir: std::path::PathBuf,
        leader_wal_path: std::path::PathBuf,
        follower_wal_path: std::path::PathBuf,
    }

    impl Rig {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "houyi-repl-rig-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self {
                leader_wal_path: dir.join("leader.collections.wal"),
                follower_wal_path: dir.join("follower.collections.wal"),
                dir,
            }
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn wait_until(mut ok: impl FnMut() -> bool, timeout: Duration, what: &str) {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if ok() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("timeout waiting for {what}");
    }

    async fn start_leader(
        rig: &Rig,
    ) -> (
        WalAsyncHandle,
        Arc<ShipHub>,
        ReplicationLeader,
        Arc<AckWatermarks>,
        u16,
    ) {
        start_leader_cfg(rig, LeaderSessionConfig::default()).await
    }

    async fn start_leader_cfg(
        rig: &Rig,
        cfg: LeaderSessionConfig,
    ) -> (
        WalAsyncHandle,
        Arc<ShipHub>,
        ReplicationLeader,
        Arc<AckWatermarks>,
        u16,
    ) {
        let hub = Arc::new(ShipHub::new(64));
        let (wal, _) =
            WalAsyncHandle::open_with_ship(&rig.leader_wal_path, Some(Arc::clone(&hub))).unwrap();
        let wm = Arc::new(AckWatermarks::new());
        let wal_for_next = wal.clone();
        let leader = ReplicationLeader::start(
            "127.0.0.1:0",
            Arc::clone(&hub),
            rig.leader_wal_path.clone(),
            Arc::new(move || wal_for_next.next_lsn()),
            Arc::clone(&wm),
            cfg,
        )
        .await
        .unwrap();
        let port = leader.local_addr().unwrap().port();
        (wal, hub, leader, wm, port)
    }

    async fn start_follower(
        rig: &Rig,
        port: u16,
        apply: Arc<RecordingApply>,
        metrics: Arc<FollowerMetrics>,
    ) -> (WalAsyncHandle, ReplicationFollower) {
        start_follower_cfg(rig, port, apply, metrics, FollowerConfig::default()).await
    }

    async fn start_follower_cfg(
        rig: &Rig,
        port: u16,
        apply: Arc<RecordingApply>,
        metrics: Arc<FollowerMetrics>,
        cfg: FollowerConfig,
    ) -> (WalAsyncHandle, ReplicationFollower) {
        let (wal, _) = WalAsyncHandle::open(&rig.follower_wal_path).unwrap();
        let follower = ReplicationFollower::start(
            format!("127.0.0.1:{port}"),
            wal.clone(),
            apply,
            metrics,
            cfg,
        )
        .unwrap();
        (wal, follower)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn end_to_end_live_replication() {
        let rig = Rig::new("e2e");
        let (leader_wal, _hub, leader, wm, port) = start_leader(&rig).await;
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let (fwal, follower) =
            start_follower(&rig, port, Arc::clone(&apply), Arc::clone(&metrics)).await;

        leader_wal.append(b"rec-0".to_vec()).unwrap();
        leader_wal.append(b"rec-1".to_vec()).unwrap();
        leader_wal.append(b"rec-2".to_vec()).unwrap();

        wait_until(
            || metrics.durable_lsn.load(Ordering::Relaxed) >= 2,
            Duration::from_secs(10),
            "follower durable lsn >= 2",
        );
        {
            // 块作用域确保 guard 在后续 await 前释放（clippy await_holding_lock）。
            // EN: Block scope ensures the guard is released before later awaits
            // (clippy await_holding_lock).
            let seen = apply.seen.lock().unwrap();
            assert_eq!(seen.len(), 3);
            assert_eq!(seen[0], (0, b"rec-0".to_vec()));
            assert_eq!(seen[2], (2, b"rec-2".to_vec()));
        }
        // follower 本地 WAL 保持 leader LSN（promote 零切换的基础）
        // EN: The follower local WAL keeps the leader's LSN (basis of zero-switch promote).
        assert_eq!(fwal.next_lsn(), 3);
        // leader 侧水位线被 DURABLE ACK 推进
        // EN: The leader-side watermark is advanced by DURABLE ACKs.
        wait_until(
            || wm.durable_lsn.load(Ordering::Relaxed) >= 2,
            Duration::from_secs(5),
            "leader watermark",
        );

        follower.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    /// P4.4.2：真组件对接的 sync 等待——leader WAL 写入经 ShipHub 扇出到真
    /// follower（loopback），其 RECEIVED/DURABLE ACK 推进 leader 水位线并
    /// 唤醒 wait()；DURABLE 唤醒意味着 follower 侧确已 fsync+apply 到该 LSN。
    /// EN: Sync waits over a real leader+follower pair on loopback — the two
    /// stages map to RECEIVED vs DURABLE acks, and a DURABLE wake implies the
    /// follower actually fsynced+applied up to the waited LSN.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sync_wait_satisfied_by_real_follower_acks() {
        let rig = Rig::new("syncwait");
        let (leader_wal, hub, leader, wm, port) = start_leader(&rig).await;
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let (_fwal, follower) =
            start_follower(&rig, port, Arc::clone(&apply), Arc::clone(&metrics)).await;
        wait_until(
            || hub.follower_count() == 1,
            Duration::from_secs(5),
            "follower registered",
        );

        // semi_sync 语义：RECEIVED（收帧即 ACK）即满足。
        // EN: semi_sync semantics: RECEIVED (ACK on frame receipt) satisfies it.
        let e1 = leader_wal.append(b"sync-semi".to_vec()).unwrap();
        let outcome = wm.wait(
            e1.lsn,
            proto::ACK_STAGE_RECEIVED,
            hub.follower_count() > 0,
            Duration::from_secs(10),
        );
        assert_eq!(
            outcome,
            crate::replication_leader::SyncWaitOutcome::Satisfied
        );

        // full_sync 语义：DURABLE（follower fsync+apply 后 ACK）才满足。
        // EN: full_sync semantics: only DURABLE (ACK after follower fsync+apply) satisfies it.
        let e2 = leader_wal.append(b"sync-full".to_vec()).unwrap();
        let outcome = wm.wait(
            e2.lsn,
            proto::ACK_STAGE_DURABLE,
            hub.follower_count() > 0,
            Duration::from_secs(10),
        );
        assert_eq!(
            outcome,
            crate::replication_leader::SyncWaitOutcome::Satisfied
        );
        // DURABLE 唤醒不是空头水位线：follower 侧确认已 fsync+apply 到该 LSN。
        // EN: A DURABLE wake is not a hollow watermark: the follower confirmed
        // fsync+apply up to that LSN.
        assert!(metrics.durable_lsn.load(Ordering::Relaxed) >= e2.lsn);
        assert_eq!(wm.sync_waits_total(), 2);
        assert_eq!(wm.sync_timeouts_total(), 0);
        assert_eq!(wm.sync_no_follower_total(), 0);

        follower.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn catch_up_from_existing_history() {
        let rig = Rig::new("catchup");
        let (leader_wal, _hub, leader, _wm, port) = start_leader(&rig).await;
        // follower 启动前 leader 已有 5 条历史
        // EN: The leader already has 5 records before the follower starts.
        for i in 0..5u8 {
            leader_wal.append(vec![i]).unwrap();
        }
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let (_fwal, follower) =
            start_follower(&rig, port, Arc::clone(&apply), Arc::clone(&metrics)).await;

        wait_until(
            || apply.seen.lock().unwrap().len() >= 5,
            Duration::from_secs(10),
            "catch-up 5 entries",
        );
        {
            let seen = apply.seen.lock().unwrap();
            let lsns: Vec<u64> = seen.iter().map(|(l, _)| *l).collect();
            assert_eq!(lsns, vec![0, 1, 2, 3, 4]);
        }

        follower.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconnect_resumes_without_duplicates() {
        let rig = Rig::new("resume");
        let (leader_wal, _hub, leader, _wm, port) = start_leader(&rig).await;
        leader_wal.append(b"a".to_vec()).unwrap();
        leader_wal.append(b"b".to_vec()).unwrap();
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let (fwal, follower) =
            start_follower(&rig, port, Arc::clone(&apply), Arc::clone(&metrics)).await;
        wait_until(
            || apply.seen.lock().unwrap().len() >= 2,
            Duration::from_secs(10),
            "initial sync",
        );
        follower.shutdown().await;
        let n_after_first = apply.seen.lock().unwrap().len();
        assert_eq!(fwal.next_lsn(), 2);
        drop(fwal);

        // leader 在 follower 离线期间再写 2 条
        // EN: The leader writes 2 more while the follower is offline.
        leader_wal.append(b"c".to_vec()).unwrap();
        leader_wal.append(b"d".to_vec()).unwrap();

        // follower 以同一 WAL 重启 → resume_lsn=2，只补 c/d
        // EN: The follower restarts on the same WAL → resume_lsn=2, catching up only c/d.
        let (fwal2, follower2) =
            start_follower(&rig, port, Arc::clone(&apply), Arc::clone(&metrics)).await;
        wait_until(
            || apply.seen.lock().unwrap().len() >= n_after_first + 2,
            Duration::from_secs(10),
            "resume catch-up",
        );
        {
            let seen = apply.seen.lock().unwrap();
            let lsns: Vec<u64> = seen.iter().map(|(l, _)| *l).collect();
            assert_eq!(lsns, vec![0, 1, 2, 3], "no duplicates, no holes");
        }
        assert_eq!(fwal2.next_lsn(), 4);

        follower2.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    /// 回归（P4.6.1）：同 resume_lsn、不同内容 ⇒ 日志分叉拒绝。
    /// 旧实现只比长度，旧主等长日志重连会被接受，同 LSN 的两条不同记录被
    /// `lsn < expected` 幂等跳过永久固化（P4.5.6 模型反例）。
    /// EN: equal length, different content must be rejected (the P4.5.6
    /// counterexample); the old handshake compared length only.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn diverged_log_is_rejected_and_counted() {
        let rig = Rig::new("diverged");
        let (leader_wal, _hub, leader, _wm, port) = start_leader(&rig).await;
        // 两侧各写 2 条、内容不同：长度相同 ⇒ 旧握手会放行。
        // EN: Both sides write 2 records with different content: equal length ⇒ the
        // old handshake would admit it.
        leader_wal.append(b"alpha".to_vec()).unwrap();
        leader_wal.append(b"beta".to_vec()).unwrap();
        {
            let (mut fw, _) = FileWal::open(&rig.follower_wal_path).unwrap();
            use crate::Wal;
            fw.append(b"rogue-0".to_vec());
            fw.append(b"rogue-1".to_vec());
        }
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let (fwal, follower) =
            start_follower(&rig, port, Arc::clone(&apply), Arc::clone(&metrics)).await;

        wait_until(
            || metrics.diverged_rejections_total.load(Ordering::Relaxed) >= 1,
            Duration::from_secs(10),
            "diverged rejection",
        );
        // 拒绝后不得应用/采纳任何 leader 记录，本地日志保持原样。
        // EN: After rejection no leader record may be applied or adopted; the local
        // log stays untouched.
        {
            let seen = apply.seen.lock().unwrap();
            assert!(
                seen.is_empty(),
                "diverged follower must not apply: {seen:?}"
            );
        }
        assert_eq!(fwal.next_lsn(), 2, "local log must not be adopted");

        follower.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    /// P5.0.4：follower 所需历史早于 leader 保留范围 ⇒ BEHIND_RETAINED，
    /// 不应用任何记录，也绝不自动清库。
    /// EN: P5.0.4 — when the follower needs history older than the leader's
    /// retained range, reject with BEHIND_RETAINED, apply nothing, and never
    /// auto-wipe.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn behind_retained_is_rejected_and_counted() {
        let rig = Rig::new("behind-retained");
        let gate = Arc::new(tokio::sync::RwLock::new(()));
        let (leader_wal, _hub, leader, _wm, port) = start_leader_cfg(
            &rig,
            LeaderSessionConfig::default().with_retention(Arc::new(|| Ok(5)), gate),
        )
        .await;
        // 物理记录不必存在：first_available 由 leader 注入的闭包给出。
        // EN: Physical records are unnecessary: first_available comes from the
        // injected closure.
        let _ = &leader_wal;
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let cfg = FollowerConfig {
            behind_retained_retry_interval: Duration::from_millis(100),
            ..FollowerConfig::default()
        };
        let (_fwal, follower) =
            start_follower_cfg(&rig, port, Arc::clone(&apply), Arc::clone(&metrics), cfg).await;

        wait_until(
            || {
                metrics
                    .behind_retained_rejections_total
                    .load(Ordering::Relaxed)
                    >= 1
            },
            Duration::from_secs(10),
            "behind-retained rejection",
        );
        assert!(apply.seen.lock().unwrap().is_empty());
        follower.shutdown().await;
        leader.shutdown().await;
    }

    /// P5.0.4：follower 的基线早于 leader first_available 但尾部重叠 ⇒
    /// ANCHOR 协商后放行，WELCOME 携带 leader first_available。
    /// EN: P5.0.4 — the follower base predates the leader's first-available but
    /// the tail overlaps; ANCHOR negotiation admits it and WELCOME carries the
    /// leader's first-available.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn anchor_negotiation_admits_overlapping_tail() {
        let rig = Rig::new("anchor");
        let gate = Arc::new(tokio::sync::RwLock::new(()));
        let (leader_wal, _hub, leader, _wm, port) = start_leader_cfg(
            &rig,
            LeaderSessionConfig::default().with_retention(Arc::new(|| Ok(4)), gate),
        )
        .await;
        for i in 0..5u64 {
            leader_wal.append(format!("r{i}").into_bytes()).unwrap();
        }
        // follower：同样的 5 条，然后只保留 [3,5)（first_available=3，next=5）。
        // EN: The follower has the same 5 records, then keeps only [3,5)
        // (first_available = 3, next = 5).
        {
            let (mut fw, _) = crate::FileWal::open(&rig.follower_wal_path).unwrap();
            use crate::Wal;
            for i in 0..5u64 {
                fw.append(format!("r{i}").into_bytes());
            }
            fw.truncate_prefix_through(2).unwrap();
        }
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let cfg = FollowerConfig::default().with_follower_identity([7u8; 16], Arc::new(|| 0));
        let (_fwal, follower) =
            start_follower_cfg(&rig, port, Arc::clone(&apply), Arc::clone(&metrics), cfg).await;

        wait_until(
            || metrics.leader_first_available_lsn.load(Ordering::Relaxed) == 4,
            Duration::from_secs(10),
            "anchored handshake WELCOME",
        );
        assert_eq!(
            metrics
                .behind_retained_rejections_total
                .load(Ordering::Relaxed),
            0
        );
        assert!(apply.seen.lock().unwrap().is_empty());
        follower.shutdown().await;
        leader.shutdown().await;
    }

    /// P4.6.3：leader 配置了 token、follower 未配 ⇒ 拒绝并计 auth 指标，
    /// 不应用任何记录（此前任何 TCP 客户端都能伪装 follower 拉走全量日志）。
    /// EN: a leader with a token rejects an unauthenticated follower and applies
    /// nothing (before this, any TCP client could pull the whole log).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auth_mismatch_is_rejected_and_counted() {
        let rig = Rig::new("auth-bad");
        let (leader_wal, _hub, leader, _wm, port) = start_leader_cfg(
            &rig,
            LeaderSessionConfig {
                auth_token: Some("s3cret".to_string()),
                ..LeaderSessionConfig::default()
            },
        )
        .await;
        leader_wal.append(b"secret-record".to_vec()).unwrap();
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let cfg = FollowerConfig {
            // 认证失败走长退避；测试压到 100ms 以免关停等满 30s。
            // EN: Auth failure takes a long backoff; the test lowers it to 100ms so
            // shutdown does not wait 30s.
            ahead_retry_interval: Duration::from_millis(100),
            ..FollowerConfig::default()
        };
        let (fwal, follower) =
            start_follower_cfg(&rig, port, Arc::clone(&apply), Arc::clone(&metrics), cfg).await;

        wait_until(
            || metrics.auth_rejections_total.load(Ordering::Relaxed) >= 1,
            Duration::from_secs(10),
            "auth rejection",
        );
        assert!(apply.seen.lock().unwrap().is_empty(), "no record may apply");
        assert_eq!(fwal.next_lsn(), 0, "local log must stay untouched");

        follower.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    /// P4.6.3：两侧 token 一致 ⇒ 正常复制。
    /// EN: matching tokens replicate normally.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn matching_auth_token_replicates() {
        let rig = Rig::new("auth-ok");
        let (leader_wal, _hub, leader, _wm, port) = start_leader_cfg(
            &rig,
            LeaderSessionConfig {
                auth_token: Some("s3cret".to_string()),
                ..LeaderSessionConfig::default()
            },
        )
        .await;
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let cfg = FollowerConfig {
            auth_token: Some("s3cret".to_string()),
            ..FollowerConfig::default()
        };
        let (_fwal, follower) =
            start_follower_cfg(&rig, port, Arc::clone(&apply), Arc::clone(&metrics), cfg).await;
        leader_wal.append(b"authed".to_vec()).unwrap();

        wait_until(
            || apply.seen.lock().unwrap().len() == 1,
            Duration::from_secs(10),
            "authenticated replication",
        );
        assert_eq!(apply.seen.lock().unwrap()[0], (0, b"authed".to_vec()));
        assert_eq!(
            metrics.auth_rejections_total.load(Ordering::Relaxed),
            0,
            "matching token must not be counted as rejection"
        );

        follower.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    /// P4.6.3 OCR 批次二十七：本地日志无法出指纹（LSN 洞）必须归为
    /// LocalLog（长退避 + 专门指标），而不是通用 Retry 重连风暴；且应在
    /// 建连之前终止（不产生任何网络重连）。
    /// EN: an unfingerprintable local log is classified as LocalLog and aborts
    /// before any network attempt.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn local_log_fingerprint_failure_is_classified() {
        let rig = Rig::new("locallog");
        {
            let (mut w, _) = FileWal::open(&rig.follower_wal_path).unwrap();
            use crate::Wal;
            w.append(b"zero".to_vec());
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&rig.follower_wal_path)
                .unwrap();
            // 跳过 LSN 1/2 直接写 3：历史有洞，指纹无法定义。
            // EN: Skip LSN 1/2 and write 3 directly: the history has a hole, so no
            // fingerprint is definable.
            f.write_all(&crate::encode_record(3, b"three", crate::WalFormat::V2, None).unwrap())
                .unwrap();
        }
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let cfg = FollowerConfig {
            ahead_retry_interval: Duration::from_millis(100),
            ..FollowerConfig::default()
        };
        // 目标端口无人监听也无妨：LocalLog 在建连之前就终止会话。
        // EN: No listener on the target port is fine: LocalLog ends the session
        // before connecting.
        let (_wal, follower) =
            start_follower_cfg(&rig, 1, Arc::clone(&apply), Arc::clone(&metrics), cfg).await;
        wait_until(
            || metrics.local_log_errors_total.load(Ordering::Relaxed) >= 1,
            Duration::from_secs(10),
            "local log classification",
        );
        assert_eq!(
            metrics.reconnects_total.load(Ordering::Relaxed),
            0,
            "local-log fault must not be counted as a reconnect"
        );
        assert!(apply.seen.lock().unwrap().is_empty());
        follower.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn follower_ahead_is_rejected_and_counted() {
        let rig = Rig::new("ahead");
        let (leader_wal, _hub, leader, _wm, port) = start_leader(&rig).await;
        // follower WAL 领先 leader（脑裂现场）
        // EN: The follower WAL leads the leader (split-brain scene).
        {
            let (mut fw, _) = FileWal::open(&rig.follower_wal_path).unwrap();
            use crate::Wal;
            for _ in 0..3 {
                let _ = fw.append(b"rogue".to_vec());
            }
        }
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let cfg = FollowerConfig {
            ahead_retry_interval: Duration::from_millis(100),
            ..FollowerConfig::default()
        };
        let (fwal, _) = WalAsyncHandle::open(&rig.follower_wal_path).unwrap();
        let follower = ReplicationFollower::start(
            format!("127.0.0.1:{port}"),
            fwal.clone(),
            apply.clone(),
            Arc::clone(&metrics),
            cfg,
        )
        .unwrap();
        wait_until(
            || {
                metrics
                    .follower_ahead_rejections_total
                    .load(Ordering::Relaxed)
                    >= 1
            },
            Duration::from_secs(10),
            "follower_ahead rejection",
        );
        // apply 零调用：脏数据不被覆盖也不被信任
        // EN: Zero apply calls: dirty data is neither overwritten nor trusted.
        assert_eq!(fwal.next_lsn(), 3);
        assert!(
            apply.seen.lock().unwrap().is_empty(),
            "follower_ahead must never apply a single record"
        );
        follower.shutdown().await;
        leader.shutdown().await;
        drop(leader_wal);
    }

    /// 空闲链路活性：零写入时 follower 必须靠心跳 ACK 喂活 leader 的
    /// ack_timeout——不回 ACK 的话 leader 每 5s 掐线， follower 陷入重连风暴。
    /// EN: Idle-link liveness: with zero writes the follower must feed the
    /// leader's ack_timeout with heartbeat ACKs — without them the leader cuts
    /// the link every 5s and the follower falls into a reconnect storm.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn idle_link_survives_ack_timeout_via_heartbeat_acks() {
        let rig = Rig::new("idle");
        let (_leader_wal, hub, leader, _wm, port) = start_leader_cfg(
            &rig,
            LeaderSessionConfig {
                ack_timeout: Duration::from_secs(2),
                heartbeat_interval: Duration::from_millis(200),
                ..LeaderSessionConfig::default()
            },
        )
        .await;
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let (_fwal, follower) = start_follower(&rig, port, apply, Arc::clone(&metrics)).await;
        // 零写入空转 5s（> 2×ack_timeout）：无心跳 ACK 时 session 早被掐了
        // EN: Idle for 5s with zero writes (> 2×ack_timeout): without heartbeat ACKs
        // the session would have been cut already.
        std::thread::sleep(Duration::from_secs(5));
        assert_eq!(
            hub.follower_count(),
            1,
            "idle session must survive via heartbeat ACKs"
        );
        assert_eq!(
            metrics.reconnects_total.load(Ordering::Relaxed),
            0,
            "no reconnect storm on idle link"
        );
        follower.shutdown().await;
        leader.shutdown().await;
    }

    /// 退出兜底 drain：不满批 + 无心跳冲刷时 leader 断连，pending 已进本地
    /// WAL（next_lsn 越过）——session 退出路径必须兜底 fsync+apply，否则
    /// 运行态内存与本地 WAL 分叉直到重启。手工假 leader 保确定性。
    /// EN: Exit backstop drain: when the leader disconnects before a full batch or
    /// a heartbeat flush, pending records are already in the local WAL (next_lsn
    /// advanced) — the session exit path must fsync+apply them, or runtime memory
    /// diverges from the local WAL until restart. A hand-written fake leader keeps
    /// it deterministic.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_entries_are_applied_on_session_end() {
        let rig = Rig::new("drain");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (fwal, _) = WalAsyncHandle::open(&rig.follower_wal_path).unwrap();
        let apply = Arc::new(RecordingApply {
            seen: Mutex::new(Vec::new()),
        });
        let metrics = Arc::new(FollowerMetrics::default());
        let follower = ReplicationFollower::start(
            format!("127.0.0.1:{port}"),
            fwal.clone(),
            apply.clone(),
            Arc::clone(&metrics),
            FollowerConfig::default(),
        )
        .unwrap();
        let fake_leader = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut hbuf = [0u8; proto::HANDSHAKE_PREFIX_LEN];
            s.read_exact(&mut hbuf).unwrap();
            s.write_all(&proto::encode_welcome(2, 0)).unwrap();
            for (lsn, p) in [(0u64, b"r0".as_slice()), (1, b"r1")] {
                let h = proto::encode_data_header(proto::STREAM_COLLECTIONS, lsn, p.len() as u32);
                s.write_all(&h).unwrap();
                s.write_all(p).unwrap();
            }
            // 不满批（2 < 32）、不发心跳，直接断连
            // EN: Below the batch threshold (2 < 32), no heartbeat, disconnect directly.
        });
        // OCR 批次十七：等 durable 水位线而非 apply 计数——fetch_max 在 apply
        // 循环之后，前者覆盖后者且消调度竞争（flake 修复）。
        // EN: OCR batch 17: wait for the durable watermark instead of the apply count —
        // fetch_max runs after the apply loop, so the former covers the latter and
        // removes scheduling races (flake fix).
        wait_until(
            || metrics.durable_lsn.load(Ordering::Relaxed) == 1,
            Duration::from_secs(10),
            "drain applies pending and advances durable watermark",
        );
        assert_eq!(apply.seen.lock().unwrap().len(), 2);
        assert_eq!(fwal.next_lsn(), 2);
        fake_leader.join().unwrap();
        follower.shutdown().await;
    }

    /// OCR 批次十七 HIGH 回归：fsync 失败的批必须随 Err 整体丢弃——留在
    /// pending 会被退出兜底 drain 二次 flush（pending_error 已消费、重试
    /// 成功）apply 成假 DURABLE；append 失败不消耗 LSN，同号记录会被新
    /// session 重推后双重 apply。
    /// EN: A batch whose fsync fails is dropped with the Err, never left in
    /// `pending` for the exit drain to re-flush into a false DURABLE ack.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_batch_drops_pending_on_fsync_error() {
        let rig = Rig::new("flushdrop");
        let (wal, _) = WalAsyncHandle::open(&rig.follower_wal_path).unwrap();
        let apply = RecordingApply {
            seen: Mutex::new(Vec::new()),
        };
        let metrics = FollowerMetrics::default();
        // loopback 对：flush 失败路径不写 ACK，流仅作形参占位。
        // EN: loopback pair: the flush-failure path writes no ACK; the stream is only
        // a parameter placeholder.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (_server, _) = listener.accept().unwrap();

        let mut pending = vec![(0u64, b"r0".to_vec()), (1u64, b"r1".to_vec())];
        // 关停 WAL worker：fsync 走 BrokenPipe 失败路径（等价于写线程上报
        // pending_error 的首次 fsync）。
        // EN: Shut down the WAL worker: fsync takes the BrokenPipe failure path
        // (equivalent to the first fsync after the writer reports pending_error).
        wal.shutdown();
        let err = flush_batch(&mut client, &wal, &apply, &metrics, &mut pending)
            .await
            .expect_err("fsync on a stopped wal must fail");
        assert!(
            matches!(err, SessionError::Retry(_)),
            "fsync failure maps to Retry, got: {err:?}"
        );
        assert!(
            pending.is_empty(),
            "failed batch must be dropped with the Err, not left for the exit drain"
        );
        assert!(
            apply.seen.lock().unwrap().is_empty(),
            "unconfirmed batch must never reach apply"
        );
        assert_eq!(
            metrics.durable_lsn.load(Ordering::Relaxed),
            0,
            "no DURABLE watermark advance on failed flush"
        );
        // drain 幂等：pending 已空，二次 flush 是 no-op（成功且零副作用）。
        // EN: drain is idempotent: pending is already empty, so a second flush is a
        // successful no-op.
        flush_batch(&mut client, &wal, &apply, &metrics, &mut pending)
            .await
            .expect("empty pending flush is a no-op");
        assert!(apply.seen.lock().unwrap().is_empty());
    }
}
