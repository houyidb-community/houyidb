//! 移动端数据库快照：原子写 + CRC32 + 版本化格式，无 WAL。
//! Mobile database snapshot: atomic write + CRC32 + versioned format, no WAL.
//!
//! 格式（小端）：
//! magic(8) | version u32 | body_len u64 | crc32 u32 | body
//! body = collections u32，每个集合：name、dims、index_type、metric/dtype、
//! backend 标签、向量负载（原始行或 flat-clustered 构建态）、标量页。
//!
//! 写入先落同目录临时文件并 fsync，再 rename；任何解析失败都返回错误而不
//! panic/越界分配。flat-clustered 构建态保存 f16/f32 分簇数据，加载无需重建；
//! flat 与 clustered-hnsw 保存原始行（后者加载时重建）。
//! EN: Layout (little endian): magic | version | body_len | crc32 | body. Body
//! stores collections with name, dims, index type, backend tag, vector payload
//! (raw rows or a built flat-clustered state), and the scalar page. Writes land
//! in a sibling temp file, fsync, then rename; parsing failures return errors
//! without panics or allocation blow-ups. A built flat-clustered state carries
//! f16/f32 cluster data and loads without rebuilding; flat and clustered-hnsw
//! store raw rows (the latter rebuilds on load).

use crate::backend::BackendSnapshot;
use crate::db::{CollectionSnapshot, MobileDB};
use crate::filter::{ColType, ScalarStoreSnapshot, ScalarVal};
use houyi_vector::cluster::FlatStorageDtype;
use houyi_vector::DistanceMetric;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"HOUYIMB1";
const VERSION: u32 = 3;
const HEADER_LEN: usize = 8 + 4 + 8 + 4;
const MAX_NAME_BYTES: usize = 4096;
const MAX_COLLECTIONS: usize = 4096;
const MAX_DIMS: usize = 65_536;
const MAX_SCALAR_ROWS: usize = 10_000_000;
/// 文件总大小上限：移动端集合远小于此值；先查元数据再读，避免坏文件
/// 触发无界分配。
/// EN: Total file-size cap; real mobile collections are far smaller. Check
/// metadata before reading so a corrupt file cannot force an unbounded read.
const MAX_SNAPSHOT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// 将数据库原子写入 `path`。
/// EN: Atomically write the database to `path`.
pub(crate) fn save_to(db: &MobileDB, path: &Path) -> Result<(), String> {
    let body = encode_body(db)?;
    let crc = crc32fast::hash(&body);
    let body_len =
        u64::try_from(body.len()).map_err(|_| "snapshot: body length overflows u64".to_string())?;

    // 头与 body 分开写：避免把整个 body 再复制一份（大集合的峰值内存）。
    // EN: Write the header and body separately so the whole body is not copied
    // once more (peak memory on large collections).
    let tmp = temp_path(path);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| format!("snapshot: create temp {}: {e}", tmp.display()))?;
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&body_len.to_le_bytes());
        header.extend_from_slice(&crc.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| format!("snapshot: write header {}: {e}", tmp.display()))?;
        file.write_all(&body)
            .map_err(|e| format!("snapshot: write temp {}: {e}", tmp.display()))?;
        file.sync_all()
            .map_err(|e| format!("snapshot: fsync temp {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            format!(
                "snapshot: rename {} -> {}: {e}",
                tmp.display(),
                path.display()
            )
        })?;
        // rename 的目录项也要持久化；部分移动文件系统不支持目录 fsync，失败忽略。
        // EN: Persist the directory entry created by rename too; some mobile
        // filesystems do not support directory fsync, so ignore its error.
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// 读取并校验数据库快照；坏 magic/版本/CRC/截断/尾随字节一律拒绝。
/// EN: Read and validate a database snapshot; bad magic/version/CRC,
/// truncation, and trailing bytes are all rejected.
pub(crate) fn load_from(path: &Path) -> Result<MobileDB, String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("snapshot: stat {}: {e}", path.display()))?
        .len();
    if size > MAX_SNAPSHOT_BYTES {
        return Err(format!("snapshot: file too large ({size} bytes)"));
    }
    let bytes =
        std::fs::read(path).map_err(|e| format!("snapshot: read {}: {e}", path.display()))?;
    if bytes.len() < HEADER_LEN {
        return Err("snapshot: truncated header".to_string());
    }
    if &bytes[..8] != MAGIC {
        return Err("snapshot: bad magic".to_string());
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if !(1..=VERSION).contains(&version) {
        return Err(format!("snapshot: unsupported version {version}"));
    }
    let body_len_u64 = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);
    let body_len = usize::try_from(body_len_u64)
        .map_err(|_| "snapshot: body length exceeds address space".to_string())?;
    let crc = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    if body_len != bytes.len() - HEADER_LEN {
        return Err("snapshot: body length mismatch".to_string());
    }
    let body = &bytes[HEADER_LEN..];
    if crc32fast::hash(body) != crc {
        return Err("snapshot: crc mismatch".to_string());
    }
    let snapshots = decode_body(body, version)?;
    MobileDB::from_snapshots(snapshots)
}

fn temp_path(path: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("snapshot");
    let attempt = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    path.with_file_name(format!(
        "{name}.tmp-{}-{nanos}-{attempt}",
        std::process::id()
    ))
}

fn encode_body(db: &MobileDB) -> Result<Vec<u8>, String> {
    let snapshots = db.snapshot_collections()?;
    if snapshots.len() > MAX_COLLECTIONS {
        return Err("snapshot: too many collections".to_string());
    }
    let mut w = Writer::new();
    w.u32(snapshots.len() as u32);
    for snap in snapshots {
        validate_name(&snap.name)?;
        if snap.dims == 0 || snap.dims > MAX_DIMS {
            return Err(format!(
                "snapshot: collection '{}' has invalid dims {}",
                snap.name, snap.dims
            ));
        }
        w.string(&snap.name);
        w.u32(snap.dims as u32);
        w.u8(snap.metric.to_snapshot_byte());
        w.u8(snap.dtype.to_snapshot_byte());
        let index_code = match snap.index_type.as_str() {
            "flat" => 0,
            "flat-clustered" => 1,
            "clustered-hnsw" => 2,
            other => {
                return Err(format!(
                    "snapshot: collection '{}' has unsupported index type {other:?}",
                    snap.name
                ))
            }
        };
        w.u8(index_code);
        match snap.backend {
            BackendSnapshot::Flat(rows) => {
                w.u8(0);
                encode_rows(&mut w, &rows, snap.dims)?;
            }
            BackendSnapshot::FlatClusteredPending(rows) => {
                w.u8(1);
                encode_rows(&mut w, &rows, snap.dims)?;
            }
            BackendSnapshot::FlatClusteredBuilt(state) => {
                w.u8(2);
                w.u64(
                    u64::try_from(state.len())
                        .map_err(|_| "snapshot: state length overflow".to_string())?,
                );
                w.bytes(&state);
            }
            BackendSnapshot::ClusteredHnsw { rows, built } => {
                w.u8(if built { 4 } else { 3 });
                encode_rows(&mut w, &rows, snap.dims)?;
            }
        }
        encode_scalars(&mut w, &snap.scalars)?;
    }
    Ok(w.finish())
}

fn encode_rows(w: &mut Writer, rows: &[(u64, Vec<f32>)], dims: usize) -> Result<(), String> {
    w.u64(rows.len() as u64);
    for (id, vector) in rows {
        if vector.len() != dims {
            return Err(format!(
                "snapshot: row {id} has {} dims, expected {dims}",
                vector.len()
            ));
        }
        w.u64(*id);
        for &x in vector {
            w.f32(x);
        }
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_NAME_BYTES {
        return Err("snapshot: invalid collection name".to_string());
    }
    Ok(())
}

fn encode_scalars(w: &mut Writer, snapshot: &ScalarStoreSnapshot) -> Result<(), String> {
    if snapshot.rows > MAX_SCALAR_ROWS {
        return Err("snapshot: scalar row count too large".to_string());
    }
    if snapshot.row_ids.len() != snapshot.rows {
        return Err(format!(
            "snapshot: scalar row ids have {} entries, expected {}",
            snapshot.row_ids.len(),
            snapshot.rows
        ));
    }
    // 保存前也校验唯一性：重复 id 会让 row_index 与 row_ids 指向不同行，
    // 若写入成功而加载端拒绝，用户会得到“能存不能读”的快照。
    // EN: Validate uniqueness before writing too: duplicate ids make row_index
    // and row_ids disagree, and a file that saves but fails to load is worse
    // than an early error.
    let mut seen = HashSet::with_capacity(snapshot.row_ids.len());
    w.u64(snapshot.rows as u64);
    for &id in &snapshot.row_ids {
        if !seen.insert(id) {
            return Err(format!("snapshot: duplicate scalar row id {id}"));
        }
        w.u64(id);
    }
    w.u32(snapshot.columns.len() as u32);
    for (name, ty, values) in &snapshot.columns {
        validate_name(name)?;
        if values.len() != snapshot.rows {
            return Err(format!(
                "snapshot: scalar column '{name}' has {} values, expected {}",
                values.len(),
                snapshot.rows
            ));
        }
        w.string(name);
        w.u8(match ty {
            ColType::Int => 0,
            ColType::Float => 1,
            ColType::Text => 2,
            ColType::Timestamp => 3,
        });
        w.u64(values.len() as u64);
        for value in values {
            match value {
                ScalarVal::Missing => w.u8(0),
                ScalarVal::Int(v) => {
                    w.u8(1);
                    w.i64(*v);
                }
                ScalarVal::Float(v) => {
                    w.u8(2);
                    w.f64(*v);
                }
                ScalarVal::Text(v) => {
                    w.u8(3);
                    w.string(v);
                }
                ScalarVal::Timestamp(v) => {
                    w.u8(4);
                    w.i64(*v);
                }
            }
        }
    }
    Ok(())
}

fn decode_body(body: &[u8], version: u32) -> Result<Vec<CollectionSnapshot>, String> {
    let mut r = Reader::new(body);
    let count = r.u32()? as usize;
    if count > MAX_COLLECTIONS || count > r.remaining() / 8 {
        return Err("snapshot: collection count exceeds payload".to_string());
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let name = r.string()?;
        if name.is_empty() || name.len() > MAX_NAME_BYTES {
            return Err("snapshot: invalid collection name".to_string());
        }
        let dims = r.u32()? as usize;
        if dims == 0 || dims > MAX_DIMS {
            return Err("snapshot: invalid dims".to_string());
        }
        // v1 没有 metric/dtype 字节：按旧默认（按后端类型）推断，保证升级后
        // 旧快照仍可加载。
        // EN: v1 has no metric/dtype bytes; infer the legacy per-backend
        // defaults so snapshots from older builds remain loadable.
        let (metric, dtype, index_code) = if version >= 2 {
            let metric = DistanceMetric::from_snapshot_byte(r.u8()?)
                .ok_or_else(|| "snapshot: bad metric".to_string())?;
            let dtype = FlatStorageDtype::from_snapshot_byte(r.u8()?)
                .ok_or_else(|| "snapshot: bad vector dtype".to_string())?;
            (metric, dtype, r.u8()?)
        } else {
            let index_code = r.u8()?;
            let index_type = match index_code {
                0 => "flat",
                1 => "flat-clustered",
                2 => "clustered-hnsw",
                _ => return Err("snapshot: bad index type".to_string()),
            };
            let (metric, dtype) = crate::backend::Backend::legacy_config(index_type);
            (metric, dtype, index_code)
        };
        let tag = r.u8()?;
        let backend = match tag {
            0 | 1 | 3 | 4 => {
                let rows = decode_rows(&mut r, dims)?;
                match tag {
                    0 => BackendSnapshot::Flat(rows),
                    1 => BackendSnapshot::FlatClusteredPending(rows),
                    3 => BackendSnapshot::ClusteredHnsw { rows, built: false },
                    _ => BackendSnapshot::ClusteredHnsw { rows, built: true },
                }
            }
            2 => {
                let len = r.u64()? as usize;
                BackendSnapshot::FlatClusteredBuilt(r.take(len)?.to_vec())
            }
            other => return Err(format!("snapshot: bad backend tag {other}")),
        };
        let index_type = match (index_code, tag) {
            (0, 0) => "flat",
            (1, 1 | 2) => "flat-clustered",
            (2, 3 | 4) => "clustered-hnsw",
            _ => return Err("snapshot: index type/backend tag mismatch".to_string()),
        };
        let scalars = decode_scalars(&mut r, version)?;
        out.push(CollectionSnapshot {
            name,
            dims,
            index_type: index_type.to_string(),
            metric,
            dtype,
            backend,
            scalars,
        });
    }
    if r.remaining() != 0 {
        return Err("snapshot: trailing bytes".to_string());
    }
    Ok(out)
}

fn decode_rows(r: &mut Reader, dims: usize) -> Result<Vec<(u64, Vec<f32>)>, String> {
    let count = r.u64()? as usize;
    let row_bytes = dims
        .checked_mul(4)
        .and_then(|v| v.checked_add(8))
        .ok_or_else(|| "snapshot: row size overflow".to_string())?;
    if row_bytes == 0 || count > r.remaining() / row_bytes {
        return Err("snapshot: row count exceeds payload".to_string());
    }
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let id = r.u64()?;
        let mut vector = Vec::with_capacity(dims);
        for _ in 0..dims {
            vector.push(r.f32()?);
        }
        rows.push((id, vector));
    }
    Ok(rows)
}

fn decode_scalars(r: &mut Reader, version: u32) -> Result<ScalarStoreSnapshot, String> {
    let rows = r.u64()? as usize;
    if rows > MAX_SCALAR_ROWS {
        return Err("snapshot: scalar row count too large".to_string());
    }
    let row_ids = if version >= 3 {
        let mut ids = Vec::with_capacity(rows.min(1024));
        for _ in 0..rows {
            ids.push(r.u64()?);
        }
        ids
    } else {
        // v1/v2 没有显式 id：按行号兼容。
        // EN: v1/v2 have no explicit ids; fall back to row indices.
        (0..rows as u64).collect()
    };
    let columns = r.u32()? as usize;
    if columns > r.remaining() / 8 {
        return Err("snapshot: scalar column count exceeds payload".to_string());
    }
    let mut out = Vec::with_capacity(columns);
    for _ in 0..columns {
        let name = r.string()?;
        if name.is_empty() || name.len() > MAX_NAME_BYTES {
            return Err("snapshot: invalid scalar column name".to_string());
        }
        let ty = match r.u8()? {
            0 => ColType::Int,
            1 => ColType::Float,
            2 => ColType::Text,
            3 => ColType::Timestamp,
            other => return Err(format!("snapshot: bad scalar column type {other}")),
        };
        let value_count = r.u64()? as usize;
        if value_count != rows {
            return Err(format!(
                "snapshot: scalar column '{name}' has {value_count} values, expected {rows}"
            ));
        }
        if value_count > r.remaining() {
            // 每个值至少 1 B；此处只做分配上界，细读仍逐步校验。
            // EN: Each value is at least 1 B; this only bounds allocation and
            // the actual reads still validate step by step.
            return Err("snapshot: scalar values exceed payload".to_string());
        }
        let mut values = Vec::with_capacity(value_count.min(1024));
        for _ in 0..value_count {
            values.push(match r.u8()? {
                0 => ScalarVal::Missing,
                1 => ScalarVal::Int(r.i64()?),
                2 => ScalarVal::Float(r.f64()?),
                3 => ScalarVal::Text(r.text()?),
                4 => ScalarVal::Timestamp(r.i64()?),
                other => return Err(format!("snapshot: bad scalar value tag {other}")),
            });
        }
        out.push((name, ty, values));
    }
    Ok(ScalarStoreSnapshot {
        rows,
        row_ids,
        columns: out,
    })
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { bytes: Vec::new() }
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
    fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.bytes.extend_from_slice(v);
    }
    fn string(&mut self, v: &str) {
        self.u32(v.len() as u32);
        self.bytes(v.as_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err("snapshot: truncated payload".to_string());
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn i64(&mut self) -> Result<i64, String> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f32(&mut self) -> Result<f32, String> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f64(&mut self) -> Result<f64, String> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        if len > MAX_NAME_BYTES || len > self.remaining() {
            return Err("snapshot: string length out of bounds".to_string());
        }
        self.utf8(len)
    }

    /// 标量文本值不设命名上限，只受剩余字节约束（UTF-8 仍严格校验）。
    /// EN: Scalar text values have no name-sized cap; only the remaining payload
    /// bounds them (UTF-8 is still validated).
    fn text(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        if len > self.remaining() {
            return Err("snapshot: string length out of bounds".to_string());
        }
        self.utf8(len)
    }

    fn utf8(&mut self, len: usize) -> Result<String, String> {
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| "snapshot: invalid utf-8 string".to_string())
    }
}

impl MobileDB {
    /// 保存到 `path`（原子写）。
    /// EN: Save to `path` (atomic write).
    pub fn save_to(&self, path: impl AsRef<Path>) -> Result<(), String> {
        save_to(self, path.as_ref())
    }

    /// 从 `path` 加载数据库（新句柄）。
    /// EN: Load a database from `path` (new handle).
    pub fn load_from(path: impl AsRef<Path>) -> Result<MobileDB, String> {
        load_from(path.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v1 快照带非空标量页但没有 row_ids：按行号回退，加载不得错位。
    /// EN: A v1 snapshot with a non-empty scalar page but no row ids must fall
    /// back to row indices and load without byte-range misalignment.
    #[test]
    fn test_v1_snapshot_scalar_rows_fallback_ids() {
        let mut w = Writer::new();
        w.u32(1);
        w.string("legacy-v1");
        w.u32(4);
        w.u8(0); // flat
        w.u8(0); // backend tag: flat rows
        w.u64(1); // vector rows
        w.u64(0);
        for x in [1.0f32, 0.0, 0.0, 0.0] {
            w.f32(x);
        }
        // 标量页：rows=1、columns=1（tag: text）。
        // EN: scalar page: rows=1, columns=1 (tag: text).
        w.u64(1);
        w.u32(1);
        w.string("tag");
        w.u8(2); // Text
        w.u64(1);
        w.u8(3); // ScalarVal::Text
        w.string("beach");
        let body = w.finish();
        let crc = crc32fast::hash(&body);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&body);
        let path = std::env::temp_dir().join(format!(
            "houyi-mobile-v1-scalars-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let db = load_from(&path).expect("v1 snapshot must load");
        assert_eq!(db.count("legacy-v1").unwrap(), 1);
        let _ = std::fs::remove_file(&path);
    }

    /// v1 快照（无 metric/dtype 字节）仍可加载：按旧默认推断。
    /// EN: A v1 snapshot (no metric/dtype bytes) still loads, inferring the
    /// legacy defaults.
    #[test]
    fn test_v1_snapshot_reads_with_legacy_defaults() {
        let mut w = Writer::new();
        w.u32(1); // collections
        w.string("legacy");
        w.u32(4); // dims
        w.u8(0); // index type: flat
        w.u8(0); // backend tag: flat rows
        w.u64(1); // rows
        w.u64(7);
        for x in [1.0f32, 0.0, 0.0, 0.0] {
            w.f32(x);
        }
        w.u64(0); // scalar rows
        w.u32(0); // scalar columns
        let body = w.finish();
        let crc = crc32fast::hash(&body);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&body);
        let path = std::env::temp_dir().join(format!("houyi-mobile-v1-{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let db = load_from(&path).expect("v1 snapshot must load");
        assert_eq!(db.count("legacy").unwrap(), 1);
        let col = db.get("legacy").unwrap();
        assert_eq!(col.metric, DistanceMetric::L2);
        assert_eq!(col.dtype, FlatStorageDtype::F32);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encode_rejects_duplicate_scalar_row_ids() {
        let snapshot = ScalarStoreSnapshot {
            rows: 2,
            row_ids: vec![7, 7],
            columns: Vec::new(),
        };
        let mut writer = Writer::new();
        let err = encode_scalars(&mut writer, &snapshot).unwrap_err();
        assert!(err.contains("duplicate scalar row id"), "{err}");
    }
}
