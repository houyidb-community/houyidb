//! 异步 WAL 写入：单后台线程独占 `FileWal`，高并发下合并多次 `append` 为一次 `flush`。
//! Async WAL: dedicated writer thread owns `FileWal`; batches multiple appends per `flush` under load.
//!
//! **语义 / Semantics**：与原先 `FileWal::append`（每条 `write` + `flush`）相比，仍为「数据进内核缓冲区后返回」；
//! 未默认 `fsync`。顺序与 LSN 单调性与单线程 `FileWal` 一致。
//! Compared to per-record `flush`, same visibility class (no default `fsync`); LSN order preserved.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::replication_leader::ShipHub;
use crate::{FileWal, RetentionOutcome, SegmentStats, WalEntry};

const WAL_CHANNEL_BOUND: usize = 4096;

/// 打开配置（P5.0.2）：目前只有自动轮转阈值。
/// EN: Open config (P5.0.2): currently only the auto-rotation threshold.
#[derive(Debug, Clone, Copy, Default)]
pub struct WalOpenConfig {
    pub segment_bytes: u64,
}

/// 单次从 channel 合并的最多 Append（一次 `flush_wal`）。
/// Max coalesced appends per `flush_wal` syscall batch.
pub const WAL_APPEND_BATCH_MAX: usize = 64;

enum WalMsg {
    Append {
        payload: Vec<u8>,
        reply: std::sync::mpsc::Sender<io::Result<WalEntry>>,
    },
    /// 复制 follower 专用：指定 LSN 追加（保持 leader LSN 空间）。
    /// EN: follower-only append with explicit LSN (keeps the leader's LSN space).
    AppendEntry {
        lsn: u64,
        payload: Vec<u8>,
        reply: std::sync::mpsc::Sender<io::Result<WalEntry>>,
    },
    /// fdatasync 屏障——follower group commit：接收线程攒批 fire-and-forget
    /// N 条 AppendEntry 后发一条 Fsync 等回，FIFO 保证一次 syscall 覆盖整批。
    /// 批内任一 append 失败经 pending_error 在本消息上报（fire-and-forget
    /// 无回执，错误只能顺 FIFO 传到下一个等待点）。
    /// EN: fdatasync barrier for follower group commit: the receive thread batches
    /// N fire-and-forget AppendEntry messages, then sends one Fsync; FIFO order
    /// makes one syscall cover the batch. Any append failure is reported on this
    /// message via pending_error (fire-and-forget has no reply of its own).
    Fsync {
        reply: std::sync::mpsc::Sender<io::Result<()>>,
    },
    Truncate {
        through: u64,
        reply: std::sync::mpsc::Sender<io::Result<usize>>,
    },
    Rotate {
        reply: std::sync::mpsc::Sender<io::Result<Option<PathBuf>>>,
    },
    NextLsn {
        reply: std::sync::mpsc::Sender<u64>,
    },
    FirstAvailable {
        reply: std::sync::mpsc::Sender<io::Result<u64>>,
    },
    SegmentStats {
        reply: std::sync::mpsc::Sender<io::Result<SegmentStats>>,
    },
    Retain {
        safe_floor: u64,
        max_segments: usize,
        max_deletes: usize,
        reply: std::sync::mpsc::Sender<io::Result<RetentionOutcome>>,
    },
    ActiveFileLen {
        reply: std::sync::mpsc::Sender<io::Result<u64>>,
    },
    ArchivedCount {
        reply: std::sync::mpsc::Sender<io::Result<usize>>,
    },
    /// 全量读取本地 WAL（promote 尾部补齐，P5.0.4）：writer 独占 FileWal，
    /// 读取必须在同一线程做，避免第二份句柄与追加竞争。
    /// EN: full local WAL read for promote tail replay (P5.0.4); the writer owns
    /// the FileWal, so the read must run on that thread.
    ReplayEntries {
        reply: std::sync::mpsc::Sender<io::Result<Vec<WalEntry>>>,
    },
    Shutdown,
}

/// 批内一条追加请求：`None` = 自主分配 LSN（leader 路径），`Some(lsn)` =
/// 指定 LSN（follower 路径）。
/// EN: One append request in a batch: `None` = allocate a LSN (leader path),
/// `Some(lsn)` = explicit LSN (follower path).
type AppendItem = (
    Option<u64>,
    Vec<u8>,
    std::sync::mpsc::Sender<io::Result<WalEntry>>,
);

fn worker_loop(mut wal: FileWal, rx: Receiver<WalMsg>, ship: Option<Arc<ShipHub>>) {
    let mut stash: Option<WalMsg> = None;
    // fire-and-forget 追加的失败回执无人接收，顺 FIFO 暂存到下一条 Fsync 上报。
    // EN: Nobody receives fire-and-forget append failures; stash them along the
    // FIFO until the next Fsync reports them.
    let mut pending_error: Option<(io::ErrorKind, String)> = None;
    loop {
        let msg = if let Some(m) = stash.take() {
            m
        } else {
            match rx.recv() {
                Ok(m) => m,
                Err(_) => break,
            }
        };

        match msg {
            WalMsg::Shutdown => break,
            WalMsg::Append { payload, reply } => {
                handle_append_batch(
                    &mut wal,
                    &rx,
                    (None, payload, reply),
                    &mut stash,
                    &mut pending_error,
                    &ship,
                );
            }
            WalMsg::AppendEntry {
                lsn,
                payload,
                reply,
            } => {
                handle_append_batch(
                    &mut wal,
                    &rx,
                    (Some(lsn), payload, reply),
                    &mut stash,
                    &mut pending_error,
                    &ship,
                );
            }
            WalMsg::Fsync { reply } => {
                let r = match pending_error.take() {
                    Some((kind, msg)) => Err(io::Error::new(kind, msg)),
                    None => wal.sync_data(),
                };
                let _ = reply.send(r);
            }
            WalMsg::Truncate { through, reply } => {
                let _ = reply.send(wal.truncate_prefix_through(through));
            }
            WalMsg::Rotate { reply } => {
                let _ = reply.send(wal.rotate_archive());
            }
            WalMsg::NextLsn { reply } => {
                let _ = reply.send(wal.next_lsn());
            }
            WalMsg::FirstAvailable { reply } => {
                let _ = reply.send(Ok(wal.first_available_lsn()));
            }
            WalMsg::SegmentStats { reply } => {
                let _ = reply.send(Ok(wal.segment_stats()));
            }
            WalMsg::Retain {
                safe_floor,
                max_segments,
                max_deletes,
                reply,
            } => {
                let _ = reply.send(wal.retain_segments(safe_floor, max_segments, max_deletes));
            }
            WalMsg::ActiveFileLen { reply } => {
                let _ = reply.send(wal.active_file_len());
            }
            WalMsg::ArchivedCount { reply } => {
                let _ = reply.send(wal.archived_segment_count());
            }
            WalMsg::ReplayEntries { reply } => {
                let _ = reply.send(wal.replay_all());
            }
        }
    }
}

/// 处理一批合并追加：以 `first` 起批，try_recv 尽量合并后续 Append/AppendEntry
/// （一次 flush 覆盖整批）。任一失败 → 整批回 Err（既有语义）并置 pending_error
/// 供下一条 Fsync 上报。非追加消息 stash 回外层循环。
/// EN: Handles a coalesced append batch: start with `first`, try_recv to merge
/// further Append/AppendEntry (one flush covers the batch). Any failure returns
/// Err for the whole batch and stores pending_error for the next Fsync; other
/// message kinds are stashed back to the outer loop.
fn handle_append_batch(
    wal: &mut FileWal,
    rx: &Receiver<WalMsg>,
    first: AppendItem,
    stash: &mut Option<WalMsg>,
    pending_error: &mut Option<(io::ErrorKind, String)>,
    ship: &Option<Arc<ShipHub>>,
) {
    let mut batch = vec![first];
    while batch.len() < WAL_APPEND_BATCH_MAX {
        match rx.try_recv() {
            Ok(WalMsg::Append { payload, reply }) => {
                batch.push((None, payload, reply));
            }
            Ok(WalMsg::AppendEntry {
                lsn,
                payload,
                reply,
            }) => {
                batch.push((Some(lsn), payload, reply));
            }
            Ok(other) => {
                *stash = Some(other);
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
    }

    let mut entries = Vec::with_capacity(batch.len());
    let mut failed: Option<io::Error> = None;
    for (lsn, payload, _) in batch.iter() {
        let r = match lsn {
            Some(l) => wal.append_entry_no_flush(*l, payload.clone()),
            None => wal.append_no_flush(payload.clone()),
        };
        match r {
            Ok(e) => entries.push(e),
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }

    if let Some(e) = failed {
        let _ = wal.flush_wal();
        let kind = e.kind();
        let msg = e.to_string();
        for (_, _, r) in batch {
            let _ = r.send(Err(io::Error::new(kind, msg.clone())));
        }
        *pending_error = Some((kind, msg));
        return;
    }

    if let Err(e) = wal.flush_wal() {
        let kind = e.kind();
        let msg = e.to_string();
        for (_, _, r) in batch {
            let _ = r.send(Err(io::Error::new(kind, msg.clone())));
        }
        *pending_error = Some((kind, msg));
        return;
    }

    // flush 成功后、reply 前 publish：leader 侧 follower 只见已落内核缓冲
    // 的批；publish 是 try_send 永不阻塞 wal worker（慢 follower 由 hub 踢）。
    // EN: Publish after a successful flush but before the reply: followers only
    // see batches already in the kernel buffer; publish uses try_send and never
    // blocks the wal worker (the hub evicts slow followers).
    if let Some(hub) = ship {
        for e in &entries {
            hub.publish(e.lsn, &e.payload);
        }
    }

    for (entry, (_, _, reply)) in entries.into_iter().zip(batch) {
        let _ = reply.send(Ok(entry));
    }
}

/// 共享句柄（`Arc`）：热路径无 `Mutex<FileWal>`。
/// Shared via `Arc`: no global `Mutex` on the upsert path.
pub struct WalAsyncHandle {
    inner: Arc<WalAsyncInner>,
}

struct WalAsyncInner {
    tx: SyncSender<WalMsg>,
    join: Mutex<Option<JoinHandle<()>>>,
    path: PathBuf,
}

impl WalAsyncHandle {
    /// 打开 WAL 并启动写线程；返回的 `entries` 与 `FileWal::open` 相同，供启动重放。
    /// EN: Opens the WAL and starts the writer thread; `entries` mirrors
    /// `FileWal::open` output for startup replay.
    pub fn open(path: impl AsRef<Path>) -> io::Result<(Self, Vec<WalEntry>)> {
        Self::open_with_ship_and_encryption(path, None, None)
    }

    /// `open` + 静态加密（P4.7.3）。 / open with at-rest encryption.
    /// EN: `open` plus at-rest encryption (P4.7.3).
    pub fn open_with_encryption(
        path: impl AsRef<Path>,
        encryption: Option<Arc<crate::encryption::Encryption>>,
    ) -> io::Result<(Self, Vec<WalEntry>)> {
        Self::open_with_ship_and_encryption(path, None, encryption)
    }

    /// `open` + 复制扇出：flush 成功的批 publish 到 ShipHub（role=leader 时
    /// 装配；follower 路径的 AppendEntry 同样会 publish——followers-of-follower
    /// 级联 v1 不支持，但 promote 后原 follower 的 hub 直接可用，句柄零切换）。
    /// EN: `open` plus replication fan-out: flushed batches publish to ShipHub
    /// (installed when role=leader; follower AppendEntry also publishes —
    /// follower-of-follower cascading is unsupported in v1, but after promote the
    /// former follower's hub is usable as-is, with zero handle switching).
    pub fn open_with_ship(
        path: impl AsRef<Path>,
        ship: Option<Arc<ShipHub>>,
    ) -> io::Result<(Self, Vec<WalEntry>)> {
        Self::open_with_ship_and_encryption(path, ship, None)
    }

    /// `open_with_ship` + 静态加密。
    /// EN: shipping plus at-rest encryption.
    pub fn open_with_ship_and_encryption(
        path: impl AsRef<Path>,
        ship: Option<Arc<ShipHub>>,
        encryption: Option<Arc<crate::encryption::Encryption>>,
    ) -> io::Result<(Self, Vec<WalEntry>)> {
        Self::open_with_config(path, ship, encryption, WalOpenConfig::default())
    }

    /// `open_with_ship_and_encryption` + 分段配置（P5.0.2）。
    /// EN: shipping/encryption plus segment config (P5.0.2).
    pub fn open_with_config(
        path: impl AsRef<Path>,
        ship: Option<Arc<ShipHub>>,
        encryption: Option<Arc<crate::encryption::Encryption>>,
        config: WalOpenConfig,
    ) -> io::Result<(Self, Vec<WalEntry>)> {
        let (mut wal, entries) = FileWal::open_with_encryption(path.as_ref(), encryption)?;
        wal.set_segment_bytes(config.segment_bytes);
        let path = wal.path().to_path_buf();
        let (tx, rx) = sync_channel(WAL_CHANNEL_BOUND);
        let join = std::thread::Builder::new()
            .name("houyi-wal-writer".into())
            .spawn(move || worker_loop(wal, rx, ship))
            .map_err(io::Error::other)?;
        let inner = Arc::new(WalAsyncInner {
            tx,
            join: Mutex::new(Some(join)),
            path,
        });
        Ok((WalAsyncHandle { inner }, entries))
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    pub fn append(&self, payload: Vec<u8>) -> io::Result<WalEntry> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::Append {
                payload,
                reply: reply_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    /// 复制 follower：以指定 LSN 追加（等回执）。
    /// EN: follower append with explicit LSN, waited.
    pub fn append_entry(&self, lsn: u64, payload: Vec<u8>) -> io::Result<WalEntry> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::AppendEntry {
                lsn,
                payload,
                reply: reply_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    /// 复制 follower：以指定 LSN 追加（fire-and-forget）——接收线程攒批用；
    /// 失败经 FIFO 由随后的 `fsync()` 上报，调用方必须以 fsync 收尾整批。
    /// EN: fire-and-forget variant for batched follower receive loops; failures
    /// surface at the closing `fsync()` that every batch must end with.
    pub fn append_entry_no_wait(&self, lsn: u64, payload: Vec<u8>) -> io::Result<()> {
        let (reply_tx, _reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::AppendEntry {
                lsn,
                payload,
                reply: reply_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        // rx 随作用域丢弃，worker 的 `let _ = send` 吸收回执。
        // EN: rx is dropped with the scope; the worker's `let _ = send` absorbs the reply.
        Ok(())
    }

    /// fdatasync 屏障（等回执）：整批 fire-and-forget 追加的收尾；批内任一
    /// 追加失败在本调用上报（Err）。
    /// EN: fdatasync barrier (waits for a reply) closing a batch of
    /// fire-and-forget appends; any failure in the batch is reported here as Err.
    pub fn fsync(&self) -> io::Result<()> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::Fsync { reply: reply_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn truncate_prefix_through(&self, through: u64) -> io::Result<usize> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::Truncate {
                through,
                reply: reply_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn rotate_archive(&self) -> io::Result<Option<PathBuf>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::Rotate { reply: reply_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn next_lsn(&self) -> u64 {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        if self
            .inner
            .tx
            .send(WalMsg::NextLsn { reply: reply_tx })
            .is_err()
        {
            return 0;
        }
        reply_rx.recv().unwrap_or(0)
    }

    pub fn first_available_lsn(&self) -> io::Result<u64> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::FirstAvailable { reply: reply_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    /// 读取本地 WAL 的全部条目（promote 尾部补齐，P5.0.4）。
    /// EN: Reads every local WAL entry for promote tail replay (P5.0.4).
    pub fn replay_entries(&self) -> io::Result<Vec<WalEntry>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::ReplayEntries { reply: reply_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn segment_stats(&self) -> io::Result<SegmentStats> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::SegmentStats { reply: reply_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn retain_segments(
        &self,
        safe_floor: u64,
        max_segments: usize,
        max_deletes: usize,
    ) -> io::Result<RetentionOutcome> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::Retain {
                safe_floor,
                max_segments,
                max_deletes,
                reply: reply_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn active_file_len(&self) -> io::Result<u64> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::ActiveFileLen { reply: reply_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn archived_segment_count(&self) -> io::Result<usize> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.inner
            .tx
            .send(WalMsg::ArchivedCount { reply: reply_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal writer stopped"))?;
        reply_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "wal reply missing"))?
    }

    pub fn shutdown(&self) {
        let _ = self.inner.tx.send(WalMsg::Shutdown);
        if let Ok(mut g) = self.inner.join.lock() {
            if let Some(j) = g.take() {
                let _ = j.join();
            }
        }
    }
}

impl Clone for WalAsyncHandle {
    fn clone(&self) -> Self {
        WalAsyncHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Drop for WalAsyncInner {
    fn drop(&mut self) {
        let _ = self.tx.send(WalMsg::Shutdown);
        if let Ok(mut g) = self.join.lock() {
            if let Some(j) = g.take() {
                let _ = j.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::WalAsyncHandle;
    use crate::FileWal;

    #[test]
    fn append_entry_keeps_explicit_lsn_and_roundtrips() {
        let path = crate::unique_temp_path("houyi-async-wal-entry", "wal");
        let _ = std::fs::remove_file(&path);
        let (wal, _) = WalAsyncHandle::open(&path).unwrap();
        let e0 = wal.append_entry(0, b"r0".to_vec()).unwrap();
        let e1 = wal.append_entry(1, b"r1".to_vec()).unwrap();
        assert_eq!((e0.lsn, e1.lsn), (0, 1));
        wal.fsync().unwrap();
        drop(wal);
        let (_w, entries) = FileWal::open(&path).unwrap();
        let payloads: Vec<&[u8]> = entries.iter().map(|e| e.payload.as_slice()).collect();
        assert_eq!(payloads, vec![b"r0".as_slice(), b"r1".as_slice()]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_entry_rejects_lsn_gap_and_state_unaffected() {
        let path = crate::unique_temp_path("houyi-async-wal-gap", "wal");
        let _ = std::fs::remove_file(&path);
        let (wal, _) = WalAsyncHandle::open(&path).unwrap();
        let err = wal.append_entry(5, b"hole".to_vec()).unwrap_err();
        assert!(err.to_string().contains("lsn gap"), "got: {err}");
        assert_eq!(
            wal.next_lsn(),
            0,
            "rejected append must not advance next_lsn"
        );
        // 错序拒绝后正确 LSN 仍可写入
        // EN: A correct LSN is still writable after an out-of-order rejection.
        wal.append_entry(0, b"r0".to_vec()).unwrap();
        assert_eq!(wal.next_lsn(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fire_and_forget_failure_surfaces_at_fsync() {
        let path = crate::unique_temp_path("houyi-async-wal-fnw", "wal");
        let _ = std::fs::remove_file(&path);
        let (wal, _) = WalAsyncHandle::open(&path).unwrap();
        // fire-and-forget 空洞追加：无回执，错误须经 fsync 上报
        // EN: Fire-and-forget hole append: no reply, the failure must surface via fsync.
        wal.append_entry_no_wait(9, b"hole".to_vec()).unwrap();
        let err = wal.fsync().unwrap_err();
        assert!(err.to_string().contains("lsn gap"), "got: {err}");
        // 上报后 pending_error 已取走，后续正常批次不受影响
        // EN: pending_error is consumed after reporting; later normal batches are unaffected.
        wal.append_entry_no_wait(0, b"r0".to_vec()).unwrap();
        wal.append_entry_no_wait(1, b"r1".to_vec()).unwrap();
        wal.fsync().unwrap();
        drop(wal);
        let (_w, entries) = FileWal::open(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].payload, b"r1");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ship_publishes_after_flush() {
        let path = crate::unique_temp_path("houyi-async-wal-ship", "wal");
        let _ = std::fs::remove_file(&path);
        let hub = std::sync::Arc::new(crate::replication_leader::ShipHub::new(8));
        let (_sid, rx) = hub.register();
        let (wal, _) =
            WalAsyncHandle::open_with_ship(&path, Some(std::sync::Arc::clone(&hub))).unwrap();
        wal.append(b"s1".to_vec()).unwrap();
        wal.append(b"s2".to_vec()).unwrap();
        let mut rx = rx;
        let b1 =
            crate::replication_leader::hub_recv_timeout(&mut rx, std::time::Duration::from_secs(2))
                .expect("first batch");
        let b2 =
            crate::replication_leader::hub_recv_timeout(&mut rx, std::time::Duration::from_secs(2))
                .expect("second batch");
        assert_eq!((b1.lsn, b2.lsn), (0, 1));
        assert_eq!(&b1.payload[..], b"s1");
        assert_eq!(&b2.payload[..], b"s2");
        assert_eq!(hub.shipped_batches_total(), 2);
        drop(wal);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn async_wal_append_replay_matches_file_wal() {
        let path = crate::unique_temp_path("houyi-async-wal", "wal");
        let _ = std::fs::remove_file(&path);
        let (wal, initial) = WalAsyncHandle::open(&path).unwrap();
        assert!(initial.is_empty());
        let e1 = wal.append(b"alpha".to_vec()).unwrap();
        let e2 = wal.append(b"beta".to_vec()).unwrap();
        assert_eq!(e1.lsn, 0);
        assert_eq!(e2.lsn, 1);
        drop(wal);
        let (_w, entries) = FileWal::open(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].payload, b"alpha".to_vec());
        assert_eq!(entries[1].payload, b"beta".to_vec());
        let _ = std::fs::remove_file(&path);
    }
}
