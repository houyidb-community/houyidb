//! CXL 持久内存抽象——崩溃恢复跳过 WAL 重放。
//! CXL persistent memory abstraction for crash recovery without WAL replay.
//!
//! 启用 `cxl-memory` feature 后：
//! - 生产环境：通过 DAX 文件系统映射 CXL Type-3 内存区域
//! - 开发/测试：`HOUYI_CXL_SIM_DIR` 指定文件模拟持久内存
//!
//! 恢复逻辑：
//! 1. 启动时检查 CXL 区域是否有有效快照
//! 2. 若有，直接加载快照（跳过 WAL 重放）
//! 3. 若无，走正常冷启动路径（快照 + WAL 重放）

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

/// CXL 持久内存区域句柄。
/// EN: Handle to a CXL persistent memory region.
pub struct CxlRegion {
    path: PathBuf,
    size: usize,
}

/// CXL 快照元数据。
/// EN: CXL snapshot metadata.
#[derive(Debug, Clone)]
pub struct CxlSnapshotMeta {
    pub checkpoint_lsn: u64,
    pub vector_count: usize,
    pub dim: usize,
    pub metric: String,
    pub timestamp_ms: u128,
}

impl CxlRegion {
    /// 打开 CXL 区域。
    /// EN: Opens the CXL region.
    ///
    /// 生产环境：`HOUYI_CXL_REGION_PATH` 指向 DAX 文件系统挂载点。
    /// EN: Production: `HOUYI_CXL_REGION_PATH` points to a DAX filesystem mount point.
    /// 开发环境：`HOUYI_CXL_SIM_DIR` 模拟持久内存。
    /// EN: Development: `HOUYI_CXL_SIM_DIR` simulates persistent memory.
    pub fn open() -> io::Result<Option<Self>> {
        if let Ok(path) = std::env::var("HOUYI_CXL_REGION_PATH") {
            let p = PathBuf::from(&path);
            if !p.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("CXL region path not found: {path}"),
                ));
            }
            // 获取区域大小（DAX 设备）
            // EN: Get the region size (DAX device).
            let size = fs::metadata(&p)?.len() as usize;
            tracing::info!(path = %p.display(), size_bytes = size, "houyi-core cxl: region open");
            return Ok(Some(CxlRegion { path: p, size }));
        }

        // 开发模拟模式
        // EN: Development simulation mode.
        if let Ok(sim_dir) = std::env::var("HOUYI_CXL_SIM_DIR") {
            let p = PathBuf::from(&sim_dir).join("houyi-cxl-sim.bin");
            let size = 64 * 1024 * 1024; // 64 MiB 模拟区域
            if !p.exists() {
                let _ = fs::create_dir_all(&sim_dir);
                // 预分配文件
                // EN: Preallocate the file.
                let f = fs::OpenOptions::new().create(true).write(true).open(&p)?;
                f.set_len(size as u64)?;
            }
            tracing::info!(path = %p.display(), size_bytes = size, "houyi-core cxl: sim mode");
            return Ok(Some(CxlRegion { path: p, size }));
        }

        Ok(None)
    }

    /// 写入快照数据（向量 payload）到持久内存。
    /// EN: Writes snapshot data (vector payload) to persistent memory.
    pub fn write_snapshot(&self, meta: &CxlSnapshotMeta, payload: &[u8]) -> io::Result<()> {
        if meta.checkpoint_lsn == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CXL snapshot: checkpoint_lsn must be > 0",
            ));
        }

        let snap_path = self.path.with_extension("snap");
        let meta_path = self.path.with_extension("meta");

        // 原子写入：先写 tmp，再 rename
        // EN: Atomic write: write tmp files first, then rename.
        let tmp_snap = self.path.with_extension("snap.tmp");
        let tmp_meta = self.path.with_extension("meta.tmp");

        // 写快照正文
        // EN: Write the snapshot payload.
        let mut f = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_snap)?;
        f.write_all(payload)?;
        f.sync_all()?;

        // 写元数据
        // EN: Write metadata.
        let meta_json = serde_json::json!({
            "checkpoint_lsn": meta.checkpoint_lsn,
            "vector_count": meta.vector_count,
            "dim": meta.dim,
            "metric": meta.metric,
            "timestamp_ms": meta.timestamp_ms,
        });
        let meta_bytes = serde_json::to_vec(&meta_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(&tmp_meta, &meta_bytes)?;

        // 原子 rename
        // EN: Atomic rename.
        fs::rename(&tmp_snap, &snap_path)?;
        fs::rename(&tmp_meta, &meta_path)?;

        tracing::info!(
            lsn = meta.checkpoint_lsn,
            vectors = meta.vector_count,
            bytes = payload.len(),
            "houyi-core cxl: snapshot written"
        );
        Ok(())
    }

    /// 尝试从 CXL 区域恢复快照。
    /// EN: Attempts to restore a snapshot from the CXL region.
    /// 返回 (元数据, 快照字节)；若快照不存在或损坏则返回 None。
    /// EN: Returns (metadata, snapshot bytes); returns None if absent or corrupted.
    pub fn read_snapshot(&self) -> io::Result<Option<(CxlSnapshotMeta, Vec<u8>)>> {
        let snap_path = self.path.with_extension("snap");
        let meta_path = self.path.with_extension("meta");

        if !snap_path.exists() || !meta_path.exists() {
            return Ok(None);
        }

        let meta_bytes = match fs::read(&meta_path) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };

        let meta_json: serde_json::Value = match serde_json::from_slice(&meta_bytes) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "houyi-core cxl: corrupted meta, ignoring snapshot");
                return Ok(None);
            }
        };

        let meta = CxlSnapshotMeta {
            checkpoint_lsn: meta_json["checkpoint_lsn"].as_u64().unwrap_or(0),
            vector_count: meta_json["vector_count"].as_u64().unwrap_or(0) as usize,
            dim: meta_json["dim"].as_u64().unwrap_or(0) as usize,
            metric: meta_json["metric"].as_str().unwrap_or("cosine").to_string(),
            timestamp_ms: meta_json["timestamp_ms"].as_u64().unwrap_or(0) as u128,
        };

        if meta.checkpoint_lsn == 0 || meta.vector_count == 0 {
            tracing::warn!(
                lsn = meta.checkpoint_lsn,
                vectors = meta.vector_count,
                "houyi-core cxl: empty snapshot, ignoring"
            );
            return Ok(None);
        }

        let payload = fs::read(&snap_path)?;
        tracing::info!(
            lsn = meta.checkpoint_lsn,
            vectors = meta.vector_count,
            bytes = payload.len(),
            "houyi-core cxl: snapshot found"
        );
        Ok(Some((meta, payload)))
    }

    /// 清除 CXL 快照（用于测试或重置）。
    /// EN: Clears the CXL snapshot (for tests or reset).
    pub fn clear_snapshot(&self) -> io::Result<()> {
        let snap_path = self.path.with_extension("snap");
        let meta_path = self.path.with_extension("meta");
        if snap_path.exists() {
            fs::remove_file(&snap_path)?;
        }
        if meta_path.exists() {
            fs::remove_file(&meta_path)?;
        }
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // 进程级 `env` 读写与 crate 内其它测试共享同一把锁，并统一用 `EnvRestore`
    // 在测试结束时（含 panic）恢复快照——否则一个用例改写/删除的变量会泄漏到
    // 并行执行的其它用例。
    // EN: Process-env reads/writes share one lock with the rest of the crate's
    // tests and restore via `EnvRestore` on exit (including on panic); otherwise
    // a variable set/removed by one case leaks into other parallel cases.
    use crate::config::{env_test_lock, EnvRestore};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_cxl_dir() -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("houyi-cxl-test-{ts}"))
    }

    #[test]
    fn cxl_snapshot_write_read_roundtrip() {
        let _guard = env_test_lock();
        let _sim_dir = EnvRestore::new("HOUYI_CXL_SIM_DIR");
        // `open()` 优先读取 DAX 路径；若宿主机导出了该变量，本测试会写真实
        // CXL 区域而不是临时模拟目录，必须先清空并由 `EnvRestore` 还原。
        // EN: `open()` checks the DAX path first; if the host exports it this
        // test would write the real CXL region instead of the temp sim dir,
        // so clear it first and let `EnvRestore` put it back.
        let _region_path = EnvRestore::new("HOUYI_CXL_REGION_PATH");
        std::env::remove_var("HOUYI_CXL_REGION_PATH");
        let dir = temp_cxl_dir();
        let _ = fs::create_dir_all(&dir);
        // 通过环境变量启用模拟模式
        // EN: Enable simulation mode via env var.
        std::env::set_var("HOUYI_CXL_SIM_DIR", dir.to_string_lossy().as_ref());

        let region = CxlRegion::open().unwrap().expect("should open sim region");

        let payload = b"houyidb test vector payload data for cxl persistence";
        let meta = CxlSnapshotMeta {
            checkpoint_lsn: 42,
            vector_count: 1000,
            dim: 128,
            metric: "cosine".to_string(),
            timestamp_ms: 1234567890,
        };

        // 写入
        // EN: Write.
        region.write_snapshot(&meta, payload).unwrap();

        // 读取
        // EN: Read.
        let (restored_meta, restored_payload) =
            region.read_snapshot().unwrap().expect("should restore");

        assert_eq!(restored_meta.checkpoint_lsn, 42);
        assert_eq!(restored_meta.vector_count, 1000);
        assert_eq!(restored_meta.dim, 128);
        assert_eq!(restored_meta.metric, "cosine");
        assert_eq!(restored_payload, payload);

        // 清理（环境变量由 `EnvRestore` 在 drop 时恢复）。
        // EN: Clean up (env vars are restored by `EnvRestore` on drop).
        region.clear_snapshot().unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cxl_empty_snapshot_returns_none() {
        let _guard = env_test_lock();
        let _sim_dir = EnvRestore::new("HOUYI_CXL_SIM_DIR");
        // 同 roundtrip：`open()` 优先走真实 DAX 路径，先清空再还原。
        // EN: Same as roundtrip: `open()` prefers the real DAX path; clear it
        // first and restore it afterwards.
        let _region_path = EnvRestore::new("HOUYI_CXL_REGION_PATH");
        std::env::remove_var("HOUYI_CXL_REGION_PATH");
        let dir = temp_cxl_dir();
        let _ = fs::create_dir_all(&dir);
        std::env::set_var("HOUYI_CXL_SIM_DIR", dir.to_string_lossy().as_ref());

        let region = CxlRegion::open().unwrap().expect("should open");
        let result = region.read_snapshot().unwrap();
        assert!(result.is_none());

        region.clear_snapshot().unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cxl_no_config_returns_none() {
        let _guard = env_test_lock();
        let _region_path = EnvRestore::new("HOUYI_CXL_REGION_PATH");
        let _sim_dir = EnvRestore::new("HOUYI_CXL_SIM_DIR");
        std::env::remove_var("HOUYI_CXL_REGION_PATH");
        std::env::remove_var("HOUYI_CXL_SIM_DIR");
        let result = CxlRegion::open().unwrap();
        assert!(result.is_none());
    }
}
