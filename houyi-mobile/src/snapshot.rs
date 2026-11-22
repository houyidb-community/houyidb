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
use houyi_vector::cluster::{FlatStorageDtype, SnapshotBytes};
use houyi_vector::DistanceMetric;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"HOUYIMB1";
/// 快照版本：4 = v3 + clustered-hnsw 构建态标签（旧二进制对 v4 干净拒绝）。
/// EN: Snapshot version: 4 = v3 plus the clustered-hnsw built-state tag (older
/// binaries reject v4 cleanly).
const VERSION: u32 = 4;
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

/// mmap 快照字节源（view 模式）：只读映射，不整块复制。
/// EN: mmap snapshot byte source (view mode): a read-only mapping without a
/// whole-file copy.
struct MmapSource(memmap2::Mmap);

impl SnapshotBytes for MmapSource {
    fn bytes(&self) -> &[u8] {
        &self.0
    }
}

/// 堆缓冲字节源（mmap 不可用时的回退，保持 view 的按簇惰性解码）。
/// EN: Heap-buffer byte source (fallback when mmap is unavailable; keeps the
/// lazy per-cluster decode of view mode).
struct VecSource(Vec<u8>);

impl SnapshotBytes for VecSource {
    fn bytes(&self) -> &[u8] {
        &self.0
    }
}

/// 并行 CRC32：分块计算后用 `Hasher::combine` 按原始顺序合并。
///
/// `combine` 的语义是「self 的数据在前、other 的数据在后」，因此按块顺序
/// 合并与整段单遍计算逐位一致。小于 1 MiB 时并行调度不划算，仍走单遍。
/// EN: Parallel CRC-32: hash chunks and combine them in original order. The
/// result is bit-identical to one pass over the whole slice because `combine`
/// treats self's data as preceding other's. Below 1 MiB the parallel scheduling
/// overhead is not worth it, so a single pass is used.
fn crc32_parallel(bytes: &[u8]) -> u32 {
    const PARALLEL_MIN_BYTES: usize = 1 << 20;
    if bytes.len() < PARALLEL_MIN_BYTES {
        return crc32fast::hash(bytes);
    }
    let chunks = rayon::current_num_threads().clamp(2, 8);
    let chunk_len = bytes.len().div_ceil(chunks);
    let hashes: Vec<crc32fast::Hasher> = bytes
        .par_chunks(chunk_len)
        .map(|chunk| {
            let mut h = crc32fast::Hasher::new();
            h.update(chunk);
            h
        })
        .collect();
    let mut acc = crc32fast::Hasher::new();
    for h in &hashes {
        acc.combine(h);
    }
    acc.finalize()
}

/// 把数据库编码成完整快照字节（magic + 版本 + 长度 + CRC32 + body）。
///
/// 先在缓冲区预留头部再把 body 直接追加进去，最后回填头部：整个编码过程只有
/// 一份 body 所在的大缓冲，不会为「头部 + body」再复制一次（大集合的峰值内存）。
/// EN: Encode the database into complete snapshot bytes (magic | version | length
/// | CRC32 | body). The header is reserved first and filled in after the body is
/// appended, so the encoding holds exactly one large buffer instead of copying
/// the body a second time (peak memory on large collections).
pub(crate) fn encode_snapshot(db: &MobileDB) -> Result<Vec<u8>, String> {
    let mut w = Writer::with_prefix(HEADER_LEN);
    encode_body_into(&mut w, db)?;
    let mut bytes = w.finish();
    let body_len = bytes.len() - HEADER_LEN;
    let crc = crc32_parallel(&bytes[HEADER_LEN..]);
    let body_len =
        u64::try_from(body_len).map_err(|_| "snapshot: body length overflows u64".to_string())?;
    bytes[..8].copy_from_slice(MAGIC);
    bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
    bytes[12..20].copy_from_slice(&body_len.to_le_bytes());
    bytes[20..24].copy_from_slice(&crc.to_le_bytes());
    Ok(bytes)
}

/// 解码并校验快照字节；坏 magic/版本/CRC/截断/尾随字节一律拒绝。
/// EN: Decode and validate snapshot bytes; bad magic/version/CRC, truncation,
/// and trailing bytes are all rejected.
pub(crate) fn decode_snapshot(bytes: &[u8]) -> Result<MobileDB, String> {
    let size = u64::try_from(bytes.len()).map_err(|_| "snapshot: buffer too large".to_string())?;
    if size > MAX_SNAPSHOT_BYTES {
        return Err(format!("snapshot: buffer too large ({size} bytes)"));
    }
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
    if crc32_parallel(body) != crc {
        return Err("snapshot: crc mismatch".to_string());
    }
    let snapshots = decode_body(body, version)?;
    MobileDB::from_snapshots(snapshots)
}

/// view 模式解码：只校验头部与结构，**不计算整文件 CRC**；flat-clustered
/// 构建态的向量按簇惰性解码。调用方需保证 `source` 在数据库生命周期内有效。
/// EN: View-mode decode: validate the header and structure but **skip the
/// whole-file CRC**; flat-clustered built-state vectors decode lazily per
/// cluster. The caller must keep `source` valid for the database's lifetime.
pub(crate) fn decode_snapshot_view(source: Arc<dyn SnapshotBytes>) -> Result<MobileDB, String> {
    let version = {
        let bytes = source.bytes();
        let size =
            u64::try_from(bytes.len()).map_err(|_| "snapshot: buffer too large".to_string())?;
        if size > MAX_SNAPSHOT_BYTES {
            return Err(format!("snapshot: buffer too large ({size} bytes)"));
        }
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
        if body_len != bytes.len() - HEADER_LEN {
            return Err("snapshot: body length mismatch".to_string());
        }
        version
    };
    let snapshots = decode_body_view(&source, HEADER_LEN, version)?;
    MobileDB::from_snapshots(snapshots)
}

/// 将数据库原子写入 `path`。
/// EN: Atomically write the database to `path`.
pub(crate) fn save_to(db: &MobileDB, path: &Path) -> Result<(), String> {
    let bytes = encode_snapshot(db)?;
    write_snapshot_bytes(path, &bytes)
}

/// 原子写入已编码的快照字节：同目录临时文件 + fsync + rename。
/// EN: Atomically write already-encoded snapshot bytes: sibling temp file,
/// fsync, then rename.
fn write_snapshot_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = temp_path(path);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| format!("snapshot: create temp {}: {e}", tmp.display()))?;
        file.write_all(bytes)
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
    // 只读映射省掉整块 read 拷贝与一份峰值内存；解码会在返回前把所有数据
    // 复制进自有的簇缓冲，映射随后即可释放。SIGBUS 风险来自外部前提：映射
    // 期间不得有他方原地截断/改写该文件；本库 save_to 的「临时文件 + rename」
    // 原子替换满足该前提，宿主自行原地覆写时请改用 load_bytes（见公开文档）。
    // EN: A read-only mapping removes the whole-file read copy and one peak
    // memory copy; decoding copies everything into owned cluster buffers before
    // returning, so the mapping can be dropped. The SIGBUS risk is an external
    // precondition: nobody may truncate or overwrite the file in place while it
    // is mapped. This library's atomic temp-file + rename satisfies it; hosts
    // that overwrite in place should use load_bytes (see the public docs).
    if let Ok(file) = std::fs::File::open(path) {
        // SAFETY：映射只读，文件句柄在 map 生命周期内保持打开；并发截断是
        // 调用方需遵守的外部前提（见上方说明），不是本调用能建立的保证。
        // EN: SAFETY: the mapping is read-only and the handle stays open for the
        // mapping's lifetime; concurrent truncation is an external precondition
        // for the caller (see above), not something this call can guarantee.
        if let Ok(map) = unsafe { memmap2::Mmap::map(&file) } {
            return decode_snapshot(&map);
        }
    }
    // 映射不可用（如部分特殊文件系统）时退回整块读取，语义一致。
    // EN: Fall back to a whole-file read when mapping is unavailable (e.g. some
    // special filesystems); semantics are identical.
    let bytes =
        std::fs::read(path).map_err(|e| format!("snapshot: read {}: {e}", path.display()))?;
    decode_snapshot(&bytes)
}

/// view 模式从文件加载：只读 mmap，元数据解析后按簇惰性解码。
///
/// 不计算整文件 CRC；调用方不得在数据库使用期间原地截断/改写该文件。
/// EN: Load from a file in view mode: read-only mmap, metadata parsed up front
/// and clusters decoded lazily. The whole-file CRC is skipped; the caller must
/// not truncate or overwrite the file in place while the database is in use.
pub(crate) fn load_view_from(path: &Path) -> Result<MobileDB, String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("snapshot: stat {}: {e}", path.display()))?
        .len();
    if size > MAX_SNAPSHOT_BYTES {
        return Err(format!("snapshot: file too large ({size} bytes)"));
    }
    let file =
        std::fs::File::open(path).map_err(|e| format!("snapshot: open {}: {e}", path.display()))?;
    // SAFETY：只读映射；文件句柄在映射建立后即可关闭，映射本身保持有效。
    // 调用方必须保证使用期间没有他方原地截断/改写（见公开文档）。
    // EN: SAFETY: read-only mapping; the handle may close after mapping and the
    // mapping stays valid. The caller must ensure nobody truncates or
    // overwrites the file in place during use (see the public docs).
    match unsafe { memmap2::Mmap::map(&file) } {
        Ok(map) => decode_snapshot_view(Arc::new(MmapSource(map))),
        Err(_) => {
            // 部分文件系统不支持映射：退回整块读取，仍保持按簇惰性解码。
            // EN: Some filesystems cannot map; fall back to a whole-file copy
            // while keeping the lazy per-cluster decode.
            let bytes = std::fs::read(path)
                .map_err(|e| format!("snapshot: read {}: {e}", path.display()))?;
            decode_snapshot_view(Arc::new(VecSource(bytes)))
        }
    }
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

fn encode_body_into(w: &mut Writer, db: &MobileDB) -> Result<(), String> {
    let snapshots = db.snapshot_collections()?;
    if snapshots.len() > MAX_COLLECTIONS {
        return Err("snapshot: too many collections".to_string());
    }
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
                encode_rows(w, &rows, snap.dims)?;
            }
            BackendSnapshot::FlatClusteredPending(rows) => {
                w.u8(1);
                encode_rows(w, &rows, snap.dims)?;
            }
            BackendSnapshot::FlatClusteredBuilt(state) => {
                w.u8(2);
                w.u64(
                    u64::try_from(state.len())
                        .map_err(|_| "snapshot: state length overflow".to_string())?,
                );
                w.bytes(&state);
            }
            // view 状态保存时直接写回原始构建态字节（未修改时与 eager 完全一致）。
            // EN: Persist a view state by writing back its raw built-state bytes
            // (identical to eager when unmodified).
            BackendSnapshot::FlatClusteredView { source, base, len } => {
                let end = base
                    .checked_add(len)
                    .ok_or_else(|| "snapshot: view state range overflow".to_string())?;
                let bytes = source
                    .bytes()
                    .get(base..end)
                    .ok_or_else(|| "snapshot: view state out of range".to_string())?;
                w.u8(2);
                w.u64(
                    u64::try_from(bytes.len())
                        .map_err(|_| "snapshot: state length overflow".to_string())?,
                );
                w.bytes(bytes);
            }
            BackendSnapshot::ClusteredHnsw { rows, built } => {
                w.u8(if built { 4 } else { 3 });
                encode_rows(w, &rows, snap.dims)?;
            }
            BackendSnapshot::ClusteredHnswBuilt(state) => {
                w.u8(5);
                w.u64(
                    u64::try_from(state.len())
                        .map_err(|_| "snapshot: state length overflow".to_string())?,
                );
                w.bytes(&state);
            }
        }
        encode_scalars(w, &snap.scalars)?;
    }
    Ok(())
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
    decode_body_impl(body, version, None)
}

/// view 模式：只解析元数据；flat-clustered 构建态记录窗口并惰性解码。
/// EN: View mode: parse metadata only; flat-clustered built states record
/// windows and decode lazily.
fn decode_body_view(
    source: &Arc<dyn SnapshotBytes>,
    body_start: usize,
    version: u32,
) -> Result<Vec<CollectionSnapshot>, String> {
    let bytes = source.bytes();
    let body = bytes
        .get(body_start..)
        .ok_or_else(|| "snapshot: body starts past end".to_string())?;
    decode_body_impl(body, version, Some((source, body_start)))
}

fn decode_body_impl(
    body: &[u8],
    version: u32,
    view: Option<(&Arc<dyn SnapshotBytes>, usize)>,
) -> Result<Vec<CollectionSnapshot>, String> {
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
                let len = usize::try_from(r.u64()?)
                    .map_err(|_| "snapshot: flat-clustered state too large".to_string())?;
                match view {
                    Some((source, body_start)) => {
                        let off = r.pos();
                        r.skip(len)?;
                        let base = body_start
                            .checked_add(off)
                            .ok_or_else(|| "snapshot: state offset overflow".to_string())?;
                        BackendSnapshot::FlatClusteredView {
                            source: Arc::clone(source),
                            base,
                            len,
                        }
                    }
                    None => BackendSnapshot::FlatClusteredBuilt(r.take(len)?.to_vec()),
                }
            }
            5 => {
                // clustered-hnsw 构建态：magic/版本等由 ClusteredHnsw::decode_state 校验。
                // EN: Clustered-hnsw built state; decode_state validates magic etc.
                let len = r.u64()? as usize;
                BackendSnapshot::ClusteredHnswBuilt(r.take(len)?.to_vec())
            }
            other => return Err(format!("snapshot: bad backend tag {other}")),
        };
        let index_type = match (index_code, tag) {
            (0, 0) => "flat",
            (1, 1 | 2) => "flat-clustered",
            (2, 3..=5) => "clustered-hnsw",
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
    #[cfg(test)]
    fn new() -> Self {
        Writer { bytes: Vec::new() }
    }
    /// 预留 `prefix` 字节（快照头部）后再追加内容，避免二次复制 body。
    /// EN: Reserve `prefix` bytes (snapshot header) before appending content so
    /// the body is never copied twice.
    fn with_prefix(prefix: usize) -> Self {
        Writer {
            bytes: vec![0u8; prefix],
        }
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
    fn pos(&self) -> usize {
        self.pos
    }
    fn skip(&mut self, n: usize) -> Result<(), String> {
        self.take(n).map(|_| ())
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

    /// 编码成内存快照字节（加密/上传由调用方负责；不产生临时文件）。
    /// EN: Encode into in-memory snapshot bytes (encryption/upload is the
    /// caller's job; no temp file is created).
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        encode_snapshot(self)
    }

    /// 从内存快照字节恢复数据库（同样执行 magic/版本/长度/CRC 校验）。
    /// EN: Restore a database from in-memory snapshot bytes (same magic/
    /// version/length/CRC validation as the file path).
    pub fn decode(bytes: &[u8]) -> Result<MobileDB, String> {
        decode_snapshot(bytes)
    }

    /// 从 `path` 加载数据库（新句柄）。
    ///
    /// 文件通过只读 mmap 读取：**调用方必须保证加载期间没有其他进程/线程
    /// 原地修改或截断该文件**。本库的 `save_to` 使用临时文件 + rename，满足
    /// 该前提；Vault 等宿主若原地覆写同一路径，请改用 `load_bytes`，或把覆写
    /// 也改成原子 rename。违反该前提可能触发 SIGBUS。
    /// EN: Load a database from `path` (new handle).
    ///
    /// The file is read through a read-only mmap: **the caller must ensure no
    /// other process/thread modifies or truncates the file in place while it is
    /// being loaded.** This library's `save_to` uses a temp file + rename and
    /// satisfies the requirement; if a Vault-style host overwrites the same path
    /// in place, use `load_bytes` instead, or make that overwrite atomic.
    /// Breaking the requirement can raise SIGBUS.
    pub fn load_from(path: impl AsRef<Path>) -> Result<MobileDB, String> {
        load_from(path.as_ref())
    }

    /// 以 view 模式从 `path` 加载：只解析元数据，簇向量首次检索时按簇解码。
    ///
    /// 不计算整文件 CRC（结构校验完整）；文件通过只读 mmap 引用，加载后
    /// **不得原地截断或改写该文件**（临时文件 + rename 原子替换是安全的）。
    /// 仅 flat-clustered 构建态按簇惰性解码；其他后端仍按原方式解码。
    /// EN: Load `path` in view mode: parse metadata only and decode cluster
    /// vectors on first search. The whole-file CRC is skipped (structural
    /// validation remains complete); the file is referenced through a
    /// read-only mmap and must not be truncated or overwritten in place after
    /// loading (atomic temp-file + rename is safe). Only flat-clustered built
    /// states decode lazily; other backends decode as before.
    pub fn load_view_from(path: impl AsRef<Path>) -> Result<MobileDB, String> {
        load_view_from(path.as_ref())
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
    fn parallel_crc_matches_single_pass() {
        // 覆盖小段（走单遍）、并行阈值边界与非整块长度。
        // EN: Cover small inputs (single pass), the parallel threshold boundary,
        // and a non-chunk-aligned length.
        for len in [0usize, 1, 4095, (1 << 20) - 1, 1 << 20, (1 << 20) + 7] {
            let data: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            assert_eq!(
                crc32_parallel(&data),
                crc32fast::hash(&data),
                "crc mismatch at len {len}"
            );
        }
    }

    #[test]
    fn view_load_matches_eager_and_materializes_on_write() {
        use houyi_vector::cluster::FlatStorageDtype;
        use houyi_vector::DistanceMetric;
        use rand::{Rng, SeedableRng};

        let dims = 16;
        let n = 600;
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut vectors = Vec::with_capacity(n * dims);
        for _ in 0..n * dims {
            vectors.push(rng.gen::<f32>() * 2.0 - 1.0);
        }
        let path =
            std::env::temp_dir().join(format!("houyi-mobile-view-{}.bin", std::process::id()));
        let db = MobileDB::new();
        db.get_or_create_with_config(
            "faces",
            dims,
            "flat-clustered",
            DistanceMetric::Cosine,
            FlatStorageDtype::F16,
        )
        .unwrap();
        db.insert_batch("faces", &ids, &vectors, dims).unwrap();
        db.build("faces").unwrap();
        db.save_to(&path).unwrap();

        let eager = MobileDB::load_from(&path).unwrap();
        let viewed = MobileDB::load_view_from(&path).unwrap();
        assert_eq!(viewed.count("faces").unwrap(), n);
        for qi in 0..4 {
            let q = &vectors[qi * dims..(qi + 1) * dims];
            assert_eq!(
                eager.search("faces", q, 10).unwrap(),
                viewed.search("faces", q, 10).unwrap()
            );
        }
        // 未写入的 view 保存直接复用原始构建态字节（不触发全量解码），
        // 往返后与 eager 检索一致。
        // EN: Saving an unwritten view reuses its raw built-state bytes (no full
        // decode) and roundtrips to eager-equivalent search results.
        let view_saved =
            std::env::temp_dir().join(format!("houyi-mobile-view-raw-{}.bin", std::process::id()));
        viewed.save_to(&view_saved).unwrap();
        let reloaded_view = MobileDB::load_from(&view_saved).unwrap();
        assert_eq!(reloaded_view.count("faces").unwrap(), n);
        for qi in 0..2 {
            let q = &vectors[qi * dims..(qi + 1) * dims];
            assert_eq!(
                eager.search("faces", q, 10).unwrap(),
                reloaded_view.search("faces", q, 10).unwrap()
            );
        }
        // 写入触发该簇及其余范数按需解码；之后新向量可被检索到，且可再次
        // 保存/加载。
        // EN: The write decodes the touched cluster (and norms) on demand; the
        // new vector is searchable and the database can be saved and loaded
        // again.
        let mut probe = vec![0.0f32; dims];
        probe[0] = 1.0;
        viewed.insert("faces", 9001, &probe).unwrap();
        assert_eq!(viewed.search("faces", &probe, 1).unwrap()[0].0, 9001);
        let path2 =
            std::env::temp_dir().join(format!("houyi-mobile-view2-{}.bin", std::process::id()));
        viewed.save_to(&path2).unwrap();
        let reloaded = MobileDB::load_from(&path2).unwrap();
        assert_eq!(reloaded.count("faces").unwrap(), n + 1);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&view_saved);
        let _ = std::fs::remove_file(&path2);
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
