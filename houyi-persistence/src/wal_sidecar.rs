//! WAL 旁路文件：记录「已成功 apply 到索引的最末 LSN」.
//! Sidecar file: last LSN that was successfully applied to the index (after WAL append + insert).
//!
//! 另见 `*.wal.next_lsn`：当 WAL 文件被 **截断为空** 时，记录下一个待分配 LSN（否则 `open` 无法从空文件推断）。
//! See also `*.wal.next_lsn`: when the WAL file is **truncated to empty**, stores the next LSN to allocate.
//!
//! 用途：运维可见性、未来 WAL 截断/检查点；**当前**冷启动仍 **全量重放** WAL（空索引），不依赖本字段跳过记录。
//! Use: observability; future truncation/checkpoint. Cold start still does full WAL replay into an empty index.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// 与 `wal` 同目录、同主名：`foo.wal` → `foo.applied_lsn`.
/// Same directory and stem as the WAL file: `foo.wal` → `foo.applied_lsn`.
pub fn sidecar_path_for_wal(wal_path: &Path) -> PathBuf {
    let parent = wal_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = wal_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("wal");
    parent.join(format!("{stem}.applied_lsn"))
}

/// `foo.wal` → `foo.wal.next_lsn`（与 `with_extension` 规则一致）.
/// `foo.wal` → `foo.wal.next_lsn` (same `with_extension` convention).
pub fn wal_next_lsn_meta_path(wal_path: &Path) -> PathBuf {
    wal_path.with_extension("wal.next_lsn")
}

/// 读取截断后空 WAL 的下一 LSN；无文件则 `None`.
/// EN: Reads the next LSN for an empty post-truncate WAL; `None` when absent.
pub fn read_wal_next_lsn_meta(wal_path: &Path) -> std::io::Result<Option<u64>> {
    let p = wal_next_lsn_meta_path(wal_path);
    match std::fs::read_to_string(&p) {
        Ok(s) => {
            let t = s.trim();
            if t.is_empty() {
                return Ok(None);
            }
            t.parse::<u64>()
                .map(Some)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// 原子写入 `wal.next_lsn`（截断空 WAL 后由 `FileWal::truncate_prefix_through` 调用）.
/// EN: Atomically writes `wal.next_lsn`; called by `FileWal::truncate_prefix_through`.
pub fn write_wal_next_lsn_meta_atomic(wal_path: &Path, next_lsn: u64) -> std::io::Result<()> {
    let path = wal_next_lsn_meta_path(wal_path);
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("wal.next_lsn");
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!("{name}.tmp"));
    // P5.0.2：保留可能删掉最后一个归档段，活动文件只剩头——此时 next_lsn
    // 侧车是唯一的 LSN 来源，必须 fsync 文件与父目录后才能删除归档。
    // EN: Retention may delete the last archive leaving a header-only active
    // segment; the next_lsn sidecar is then the only source and must be durable
    // (file + parent dir) before archives are removed.
    {
        let mut f = File::create(&tmp)?;
        f.write_all(format!("{next_lsn}\n").as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    crate::sync_parent_dir(&path)?;
    Ok(())
}

/// 读取侧车中的 LSN；文件不存在或空则返回 0。
/// EN: Reads the sidecar LSN; returns 0 when the file is missing or empty.
pub fn read_applied_lsn(path: &Path) -> std::io::Result<u64> {
    match std::fs::read_to_string(path) {
        Ok(s) => {
            let t = s.trim();
            if t.is_empty() {
                return Ok(0);
            }
            t.parse::<u64>()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e),
    }
}

/// 原子写入：先写 `*.tmp` 再 `rename`。
/// EN: Atomic write: write `*.tmp` first, then `rename`.
pub fn write_applied_lsn_atomic(path: &Path, lsn: u64) -> std::io::Result<()> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("applied_lsn");
    let tmp = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{name}.tmp"));
    std::fs::write(&tmp, format!("{lsn}\n"))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unique_temp_path;

    fn temp_sidecar() -> PathBuf {
        unique_temp_path("houyidb-sidecar-test", "applied_lsn")
    }

    #[test]
    fn sidecar_path_matches_parent_stem() {
        let p = Path::new("/tmp/foo.wal");
        assert_eq!(
            sidecar_path_for_wal(p),
            PathBuf::from("/tmp/foo.applied_lsn")
        );
    }

    #[test]
    fn read_write_roundtrip() {
        let path = temp_sidecar();
        let _ = std::fs::remove_file(&path);
        write_applied_lsn_atomic(&path, 42).unwrap();
        assert_eq!(read_applied_lsn(&path).unwrap(), 42);
        write_applied_lsn_atomic(&path, 43).unwrap();
        assert_eq!(read_applied_lsn(&path).unwrap(), 43);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_missing_is_zero() {
        let path = temp_sidecar();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read_applied_lsn(&path).unwrap(), 0);
    }

    #[test]
    fn wal_next_lsn_meta_roundtrip() {
        let wal = Path::new("/tmp/x.wal");
        let p = wal_next_lsn_meta_path(wal);
        assert!(p.to_string_lossy().ends_with(".wal.next_lsn"));
        let tmp = unique_temp_path("houyidb-nextlsn", "wal");
        let _ = std::fs::remove_file(&tmp);
        write_wal_next_lsn_meta_atomic(&tmp, 7).unwrap();
        assert_eq!(read_wal_next_lsn_meta(&tmp).unwrap(), Some(7));
        let _ = std::fs::remove_file(wal_next_lsn_meta_path(&tmp));
    }
}
