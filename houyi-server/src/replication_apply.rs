//! 复制 follower 的实时 apply 汇：把 collections.wal 流式记录应用到 [`CollectionManager`]。
//! Live apply sink for the replication follower: applies streamed collections.wal
//! records into the [`CollectionManager`].
//!
//! 设计要点 / Design notes:
//! - **与冷重放同源**：解码后的记录直接调 [`apply_collection_wal_record`]（`checkpoint_of`
//!   恒 `None`——实时记录必在任一图快照 checkpoint 之后），杜绝「重放语义」与「实时
//!   apply 语义」分叉。
//!   EN: Same code path as cold replay (`checkpoint_of` is always `None` — streamed
//!   records always postdate any snapshot checkpoint), so live apply can never drift
//!   from replay semantics.
//! - **直接在 tokio 任务里 await**（P4.5.5 起）：follower 会话本身是 tokio task，
//!   `ReplicationApply::apply` 是 async trait 方法——不再需要 `Handle::block_on`
//!   桥接（在 runtime 内 block_on 会直接 panic）。严格的 LSN 顺序由 follower 的
//!   逐条 apply 循环保证。
//!   EN: apply is awaited directly inside the follower's tokio task (P4.5.5); the
//!   old `Handle::block_on` bridge is gone (it panics inside a runtime). Strict LSN
//!   ordering comes from the follower's sequential apply loop.
//! - **错误分级**：未知 op（`Ok(None)`，新版 leader 写给旧版 follower）兼容跳过；解码
//!   硬错返回 `InvalidData`——该记录在 follower 本地 WAL 已随批 fsync 落盘（next_lsn
//!   已越过它），重连后 leader 从 resume_lsn 起推、不会重发，错误不会无限自旋。
//!   EN: unknown ops are skipped for forward compatibility; a hard decode error
//!   surfaces once — the record is already durable in the local WAL so a reconnect
//!   resumes past it instead of spinning.
//! - **毒行限流跨记录共享**：`PoisonedRowLog` 持于 `Mutex`（apply 是 `&self`），前 16 条
//!   毒行打全量明细、其余折叠计数，语义与冷重放一致。
//!   EN: one `PoisonedRowLog` behind a `Mutex` shared across records, matching the
//!   cold-replay rate-limiting discipline.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use houyi_persistence::collection_wal::decode_collection_wal_payload;
use houyi_persistence::replication_follower::ReplicationApply;

use crate::collection::CollectionManager;
use crate::recovery::{apply_collection_wal_record, CollectionsReplayStats, PoisonedRowLog};

/// collections.wal 实时 apply 汇。
/// EN: Live apply sink for the collections.wal stream.
pub struct CollectionsApplySink {
    manager: Arc<CollectionManager>,
    /// tokio Mutex：apply 是 async 且要把毒行计数锁**跨 await** 借给
    /// `apply_collection_wal_record`——std MutexGuard 会让 future 非 Send。
    /// EN: tokio Mutex because the guard is held across awaits inside the async
    /// apply (a std guard would make the future non-Send).
    poison_log: tokio::sync::Mutex<PoisonedRowLog>,
    poison_skipped_total: Arc<AtomicU64>,
}

impl CollectionsApplySink {
    pub fn new(manager: Arc<CollectionManager>) -> Self {
        Self {
            manager,
            poison_log: tokio::sync::Mutex::new(PoisonedRowLog::new()),
            poison_skipped_total: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 毒行计数的共享句柄：/metrics follower 视图经此读取（sink 本体沉在
    /// follower 的 tokio 任务里，不出借引用）——唯一读取通道，不再另设快照 getter
    /// （两个读通道会让同一计数出现两份可漂移的文档/语义，OCR 批次十八）。
    /// EN: Shared handle to the poison counter — the single canonical read path
    /// (a second snapshot getter would drift).
    pub fn poison_skipped_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.poison_skipped_total)
    }
}

#[async_trait::async_trait]
impl ReplicationApply for CollectionsApplySink {
    async fn apply(&self, lsn: u64, payload: &[u8]) -> io::Result<()> {
        let record = match decode_collection_wal_payload(payload) {
            Ok(Some(r)) => r,
            // 未知 op：leader 比 follower 新时的向前兼容跳过。
            // EN: Unknown op — forward-compat skip when the leader is newer.
            Ok(None) => return Ok(()),
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
        };
        let mut stats = CollectionsReplayStats::default();
        let mut skipped = 0usize;
        let mut poison_log = self.poison_log.lock().await;
        apply_collection_wal_record(
            &self.manager,
            lsn,
            record,
            |_| None,
            &mut stats,
            &mut skipped,
            &mut poison_log,
        )
        .await?;
        if skipped > 0 {
            self.poison_skipped_total
                .fetch_add(skipped as u64, Ordering::Relaxed);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use houyi_persistence::collection_wal::{
        encode_collection_create, encode_collection_drop, encode_collection_upsert_batch,
        CollectionWalItem,
    };

    fn rig() -> (Arc<CollectionManager>, CollectionsApplySink) {
        // P4.5.5：apply 已是 async，测试直接在 tokio 运行时里跑（与 follower
        // task 的调用形态一致），不再需要 Handle::block_on 桥接。
        // EN: P4.5.5: apply is async now; the test runs directly on a tokio runtime
        // EN: (matching the follower task's call shape), no Handle::block_on bridge.
        let manager = Arc::new(CollectionManager::new(4, "hnsw"));
        let sink = CollectionsApplySink::new(Arc::clone(&manager));
        (manager, sink)
    }

    /// create + upsert 经 sink 落进 Manager：collection 建成、向量进热缓冲。
    /// EN: Create + upsert through the sink land in the manager.
    #[tokio::test]
    async fn sink_applies_create_and_upsert() {
        let (manager, sink) = rig();
        let create = encode_collection_create("c", 4, "hnsw").unwrap();
        sink.apply(0, &create).await.unwrap();
        let items = vec![CollectionWalItem {
            id: 7,
            values: vec![0.1, 0.2, 0.3, 0.4],
            label: Some("a".to_string()),
            payload: None,
        }];
        let upsert = encode_collection_upsert_batch("c", 4, &items).unwrap();
        sink.apply(1, &upsert).await.unwrap();
        let col = manager.get("c").expect("collection must exist after apply");
        assert_eq!(col.dims, 4);
        // 排空构图队列后向量必须可检索（与冷重放同语义）。
        // EN: Vectors must be searchable after draining the build queue (same as cold replay).
        crate::recovery::drain_pending_builder(&col.backend).await;
        let indexed = col.backend.indexed_count().await;
        assert_eq!(indexed, 1, "applied vector must be searchable after drain");
        assert_eq!(
            col.applied_lsn(),
            1,
            "applied_lsn must advance to the last successfully applied record"
        );
    }

    /// follower 实时 apply 整库删除：从本地 manager 摘除；重复/未知目标幂等跳过，
    /// 且不得追加本地 WAL（apply sink 只改内存）。
    /// EN: The follower drops the collection locally on a live apply; duplicate
    /// or unknown targets are idempotent skips, and the sink never appends to
    /// the local WAL (it only mutates memory).
    #[tokio::test]
    async fn sink_applies_collection_drop_idempotently() {
        let (manager, sink) = rig();
        sink.apply(0, &encode_collection_create("c", 4, "hnsw").unwrap())
            .await
            .unwrap();
        assert!(manager.get("c").is_some());

        let drop = encode_collection_drop("c").unwrap();
        sink.apply(1, &drop).await.unwrap();
        assert!(
            manager.get("c").is_none(),
            "follower must drop the collection"
        );

        // 重复 apply 与未知目标均幂等跳过（不改变 manager、不报错）。
        // EN: Duplicate apply and unknown targets are idempotent skips.
        sink.apply(2, &drop).await.unwrap();
        sink.apply(3, &encode_collection_drop("ghost").unwrap())
            .await
            .unwrap();
        assert!(manager.get("c").is_none());
        assert!(manager.get("ghost").is_none());
    }

    /// 未知 op / 非 JSON 记录兼容跳过（新版 leader → 旧版 follower）；
    /// 腐败的已知 op 是 InvalidData 硬错。
    /// EN: Unknown ops and non-JSON records are skipped; a corrupt known op
    /// is a hard InvalidData error.
    #[tokio::test]
    async fn sink_skips_unknown_and_rejects_corrupt() {
        let (_manager, sink) = rig();
        sink.apply(0, b"not json at all").await.unwrap();
        sink.apply(1, br#"{"v":1,"op":"future_op"}"#).await.unwrap();
        // 已知 op 缺字段 = 腐败硬错（重放侧同口径）。
        // EN: A known op missing a field is a hard corruption error (same as replay).
        let err = sink
            .apply(2, br#"{"v":1,"op":"collection_create","name":"c"}"#)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "got: {err}");
    }
}
