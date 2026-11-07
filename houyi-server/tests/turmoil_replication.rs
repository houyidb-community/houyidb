//! turmoil 确定性网络仿真（P4.5.5）。
//! Deterministic network simulation for the replication layer (P4.5.5).
//!
//! 运行 / Run:
//! ```text
//! cargo test -p houyi-server --features turmoil-dst --test turmoil_replication
//! ```
//!
//! **这个 feature 会改变 houyi-persistence 的编译**（网络类型换到 turmoil 的
//! drop-in 实现），故只在本测试目标、且必须带 `--test turmoil_replication`
//! 过滤时使用——别把它带进常规全量测试。
//! EN: this feature swaps houyi-persistence's network types; run it only for
//! this test target.
//!
//! 设计 / Design:
//! - leader 与 follower 各跑在 sim 的一个 host 里（各自的 tokio runtime）；
//! - **写入由测试本体驱动**（`FileWal::append` + `ShipHub::publish`），是
//!   host 外的确定性输入源——与生产「wal worker 是 std 线程」的边界一致；
//! - 网络事件（partition / repair）由测试本体在 `step()` 之间注入；
//! - 断言只看**行为**（apply 序列无洞无重、RPO、拒绝语义），不断言耗时。
//!
//! 覆盖 / Covered:
//! 1. 基础链路：live 推送到达、DURABLE ACK 推进水位线；
//! 2. 分区 → 恢复：分区期间写落 WAL，恢复后 follower 从文件 catch-up，
//!    apply 序列无洞无重复；
//! 3. follower_ahead：follower 的 LSN 超前 leader 时被拒绝并计数（脑裂信号）；
//! 4. 确定性：同 seed 两次运行得到完全相同的 apply 序列。

#![cfg(feature = "turmoil-dst")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use houyi_persistence::replication_follower::{
    FollowerConfig, FollowerMetrics, ReplicationApply, ReplicationFollower,
};
use houyi_persistence::replication_leader::{
    AckWatermarks, LeaderSessionConfig, ReplicationLeader, ShipHub,
};
use houyi_persistence::wal_async::WalAsyncHandle;
use houyi_persistence::{FileWal, Wal};

/// 临时数据目录（Drop 清理）。
/// EN: Temporary data directory (cleaned in Drop).
struct Rig {
    dir: PathBuf,
}

impl Rig {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::AtomicU64 as Seq;
        static SEQ: Seq = Seq::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "houyidb-turmoil-{}-{}-{}-{}",
            tag,
            std::process::id(),
            nanos,
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::create_dir_all(&dir);
        Self { dir }
    }

    fn leader_wal(&self) -> PathBuf {
        self.dir.join("leader.wal")
    }

    fn follower_wal(&self) -> PathBuf {
        self.dir.join("follower.wal")
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// apply 记录器：只记 LSN 序列（缺洞/重复一眼可见）。
/// EN: Apply recorder: records only LSN sequences (gaps/duplicates are obvious).
struct RecordingSink {
    seen: Arc<Mutex<Vec<u64>>>,
}

#[async_trait::async_trait]
impl ReplicationApply for RecordingSink {
    async fn apply(&self, lsn: u64, _payload: &[u8]) -> std::io::Result<()> {
        self.seen.lock().unwrap().push(lsn);
        Ok(())
    }
}

/// 测试用 follower 配置：把退避/超时压到毫秒量级，让仿真的模拟时间快速推进。
/// EN: Test follower config: shrink backoff/timeouts to milliseconds so simulated time advances
/// EN: fast.
fn fast_follower_config() -> FollowerConfig {
    FollowerConfig {
        batch_max: 4,
        read_timeout: Duration::from_millis(100),
        dns_timeout: Duration::from_millis(50),
        handshake_timeout: Duration::from_millis(200),
        backoff_initial: Duration::from_millis(10),
        backoff_max: Duration::from_millis(50),
        ahead_retry_interval: Duration::from_millis(50),
        auth_token: None,
        encryption: None,
        follower_id: [1u8; 16],
        snapshot_checkpoint: Arc::new(|| 0),
        behind_retained_retry_interval: Duration::from_millis(50),
    }
}

fn leader_config() -> LeaderSessionConfig {
    LeaderSessionConfig {
        queue_batches: 16,
        heartbeat_interval: Duration::from_millis(30),
        ack_timeout: Duration::from_millis(150),
        ..LeaderSessionConfig::default()
    }
}

/// 推进仿真：每步 = 一个 tick（默认 1ms 模拟时间）。
/// EN: Advance the simulation: each step = one tick (default 1ms simulated time).
fn step_n(sim: &mut turmoil::Sim<'_>, steps: usize) {
    for _ in 0..steps {
        let _ = sim.step().expect("sim step");
    }
}

/// 写一条记录进 WAL 并推给 hub（顺序即 LSN 顺序）。
/// EN: Write one record to the WAL and push it to the hub (order is LSN order).
fn publish_one(wal: &Mutex<FileWal>, hub: &ShipHub, next_lsn: &AtomicU64, payload: &[u8]) -> u64 {
    let lsn = {
        let mut w = wal.lock().unwrap();
        // `Wal::append` 自 panics（写失败即测试失败），无需 Result。
        // EN: `Wal::append` panics on its own (write failure = test failure); no Result needed.
        w.append(payload.to_vec()).lsn
    };
    next_lsn.store(lsn + 1, Ordering::Relaxed);
    hub.publish(lsn, payload);
    lsn
}

/// 搭一对 leader/follower host，返回共享观测点。
/// EN: Build a leader/follower host pair; returns the shared observation points.
struct Harness {
    hub: Arc<ShipHub>,
    watermarks: Arc<AckWatermarks>,
    applied: Arc<Mutex<Vec<u64>>>,
    metrics: Arc<FollowerMetrics>,
    wal: Mutex<FileWal>,
    next_lsn: Arc<AtomicU64>,
}

fn harness(rig: &Rig) -> Harness {
    let wal = FileWal::open(rig.leader_wal()).expect("open leader wal").0;
    Harness {
        hub: Arc::new(ShipHub::new(16)),
        watermarks: Arc::new(AckWatermarks::new()),
        applied: Arc::new(Mutex::new(Vec::new())),
        metrics: Arc::new(FollowerMetrics::default()),
        wal: Mutex::new(wal),
        next_lsn: Arc::new(AtomicU64::new(0)),
    }
}

/// 注册 leader host。
/// EN: Register the leader host.
fn register_leader(sim: &mut turmoil::Sim<'_>, h: &Harness, rig: &Rig) {
    let hub = Arc::clone(&h.hub);
    let wm = Arc::clone(&h.watermarks);
    let next = Arc::clone(&h.next_lsn);
    let wal_path = rig.leader_wal();
    sim.host("leader", move || {
        let (hub, wm, next, wal_path) = (
            Arc::clone(&hub),
            Arc::clone(&wm),
            Arc::clone(&next),
            wal_path.clone(),
        );
        async move {
            let _leader = ReplicationLeader::start(
                "0.0.0.0:9000",
                hub,
                wal_path,
                Arc::new(move || next.load(Ordering::Relaxed)),
                wm,
                leader_config(),
            )
            .await?;
            // 保持 host 存活到仿真结束。
            // EN: Keep the host alive until the simulation ends.
            std::future::pending::<()>().await;
            Ok(())
        }
    });
}

/// 注册 follower host。
/// EN: Register the follower host.
fn register_follower(sim: &mut turmoil::Sim<'_>, h: &Harness, rig: &Rig) {
    let seen = Arc::clone(&h.applied);
    let metrics = Arc::clone(&h.metrics);
    let wal_path = rig.follower_wal();
    sim.host("follower", move || {
        let (seen, metrics, wal_path) = (Arc::clone(&seen), Arc::clone(&metrics), wal_path.clone());
        async move {
            let (wal, _) = WalAsyncHandle::open(&wal_path)?;
            let _follower = ReplicationFollower::start(
                "leader:9000".to_string(),
                wal,
                Arc::new(RecordingSink { seen }),
                metrics,
                fast_follower_config(),
            )?;
            std::future::pending::<()>().await;
            Ok(())
        }
    });
}

/// 等到条件成立（按仿真步推进），否则 panic。
/// EN: Wait until the condition holds (advancing simulation steps), else panic.
fn wait_until(sim: &mut turmoil::Sim<'_>, steps: usize, mut ok: impl FnMut() -> bool, what: &str) {
    for _ in 0..steps {
        if ok() {
            return;
        }
        let _ = sim.step().expect("sim step");
    }
    panic!("timeout waiting for {what}");
}

fn sim(seed: u64) -> turmoil::Sim<'static> {
    turmoil::Builder::new().rng_seed(seed).build()
}

// ── 场景 1：基础链路 ──────────────────────────────────────────────
// EN: Scenario 1: basic link.

#[test]
fn turmoil_live_replication_reaches_follower() {
    let rig = Rig::new("live");
    let h = harness(&rig);
    let mut sim = sim(11);
    register_leader(&mut sim, &h, &rig);
    register_follower(&mut sim, &h, &rig);

    wait_until(
        &mut sim,
        2000,
        || h.hub.follower_count() == 1,
        "follower registered",
    );

    for i in 0..8u64 {
        publish_one(&h.wal, &h.hub, &h.next_lsn, format!("rec-{i}").as_bytes());
    }
    wait_until(
        &mut sim,
        3000,
        || h.applied.lock().unwrap().len() == 8,
        "all records applied",
    );
    assert_eq!(*h.applied.lock().unwrap(), (0..8).collect::<Vec<_>>());
    // DURABLE ACK 推进水位线（full_sync 的同步等待依赖它）。
    // EN: DURABLE ACK advances the watermark (full_sync's synchronous wait depends on it).
    assert!(
        h.metrics.durable_lsn.load(Ordering::Relaxed) >= 7,
        "durable watermark must reach the last applied lsn"
    );
    // 心跳也可能触发 flush_batch（pending 非空时），故 fsync 次数是下界而非
    // 精确值——精确断言会把仿真调度的合法差异当成失败（OCR 批次二十三）。
    // EN: Heartbeats may also trigger flush_batch (when pending is non-empty), so the fsync
    // EN: count is a lower bound, not exact — precise assertions would flag legal scheduler
    // EN: differences as failures (OCR batch 23).
    assert!(
        h.metrics.fsync_total.load(Ordering::Relaxed) >= 2,
        "batch_max=4 → at least 2 fsyncs"
    );
}

// ── 场景 2：分区 → 恢复 catch-up ─────────────────────────────────
// EN: Scenario 2: partition -> recovery catch-up.

#[test]
fn turmoil_partition_then_catch_up_without_gaps_or_duplicates() {
    let rig = Rig::new("partition");
    let h = harness(&rig);
    let mut sim = sim(22);
    register_leader(&mut sim, &h, &rig);
    register_follower(&mut sim, &h, &rig);

    wait_until(
        &mut sim,
        2000,
        || h.hub.follower_count() == 1,
        "follower registered",
    );
    for i in 0..4u64 {
        publish_one(&h.wal, &h.hub, &h.next_lsn, format!("pre-{i}").as_bytes());
    }
    wait_until(
        &mut sim,
        3000,
        || h.applied.lock().unwrap().len() == 4,
        "pre-partition records applied",
    );

    // 断开：之后的写只进 WAL（live 推送断流，session 因 ACK 超时退出）。
    // EN: Disconnect: later writes only enter the WAL (live push stalls; the session exits
    // EN: on ACK timeout).
    sim.partition("leader", "follower");
    for i in 4..10u64 {
        publish_one(
            &h.wal,
            &h.hub,
            &h.next_lsn,
            format!("during-{i}").as_bytes(),
        );
    }
    step_n(&mut sim, 500);
    assert!(
        h.applied.lock().unwrap().len() <= 10,
        "sanity: applied cannot exceed published"
    );
    let during_partition = h.applied.lock().unwrap().clone();
    assert!(
        during_partition.len() < 10,
        "partition must stall delivery (applied {during_partition:?})"
    );

    // 恢复：follower 重连 → 从 WAL 尾部 catch-up，序列无洞无重。
    // EN: Recovery: the follower reconnects -> catches up from the WAL tail, no gaps or duplicates.
    sim.repair("leader", "follower");
    wait_until(
        &mut sim,
        5000,
        || h.applied.lock().unwrap().len() >= 10,
        "catch-up after repair",
    );
    let seen = h.applied.lock().unwrap().clone();
    assert_eq!(
        seen,
        (0..10).collect::<Vec<_>>(),
        "apply sequence must be gapless and duplicate-free (got {seen:?})"
    );
}

// ── 场景 3：follower_ahead（脑裂信号）─────────────────────────────
// EN: Scenario 3: follower_ahead (split-brain signal).

#[test]
fn turmoil_follower_ahead_is_rejected_and_counted() {
    let rig = Rig::new("ahead");
    let h = harness(&rig);
    // 先给 follower 的 WAL 灌 5 条「脑裂」记录——必须在 follower 打开 WAL
    // **之前**（`WalAsyncHandle::open` 的 next_lsn 来自重放；进程内改文件不会
    // 改变已打开句柄的 next_lsn）。
    // EN: First feed 5 "split-brain" records into the follower's WAL — must be **before**
    // EN: the follower opens the WAL (WalAsyncHandle::open derives next_lsn from replay;
    // EN: changing the file in-process does not change an open handle's next_lsn).
    {
        let (mut fwal, _) = FileWal::open(rig.follower_wal()).expect("open follower wal");
        for i in 0..5u64 {
            fwal.append(format!("ahead-{i}").into_bytes());
        }
    }
    let mut sim = sim(33);
    register_leader(&mut sim, &h, &rig); // leader next_lsn = 0
    register_follower(&mut sim, &h, &rig); // follower resume_lsn = 5

    wait_until(
        &mut sim,
        3000,
        || {
            h.metrics
                .follower_ahead_rejections_total
                .load(Ordering::Relaxed)
                >= 1
        },
        "follower_ahead rejection",
    );
    // 拒绝后不 apply 任何东西（脏数据不被覆盖也不被信任），且 follower 保持
    // 自身的 5（无限退避等待人工处置）。
    // EN: After rejection nothing is applied (dirty data is neither overwritten nor trusted),
    // EN: and the follower keeps its own 5 (infinite backoff awaiting manual action).
    assert!(
        h.applied.lock().unwrap().is_empty(),
        "an ahead follower must not apply anything"
    );
    assert_eq!(
        h.hub.follower_count(),
        0,
        "ahead follower must not stay registered"
    );
}

// ── 场景 4：确定性 ───────────────────────────────────────────────
// EN: Scenario 4: determinism.

/// 同一 seed 两次运行必须得到完全相同的 apply 序列。
/// EN: Two runs with the same seed must produce exactly the same apply sequence.
fn run_determinism_round(seed: u64) -> Vec<u64> {
    let rig = Rig::new("determinism");
    let h = harness(&rig);
    let mut sim = sim(seed);
    register_leader(&mut sim, &h, &rig);
    register_follower(&mut sim, &h, &rig);

    wait_until(
        &mut sim,
        2000,
        || h.hub.follower_count() == 1,
        "follower registered",
    );
    for i in 0..6u64 {
        publish_one(&h.wal, &h.hub, &h.next_lsn, format!("d-{i}").as_bytes());
    }
    wait_until(
        &mut sim,
        3000,
        || h.applied.lock().unwrap().len() == 6,
        "determinism round applied",
    );
    let out = h.applied.lock().unwrap().clone();
    out
}

#[test]
fn turmoil_same_seed_is_deterministic() {
    let a = run_determinism_round(7);
    let b = run_determinism_round(7);
    assert_eq!(a, b, "same seed must reproduce the same apply sequence");
    assert_eq!(a, (0..6).collect::<Vec<_>>());
}
