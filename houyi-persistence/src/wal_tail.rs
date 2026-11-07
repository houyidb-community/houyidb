//! WAL 尾部读取器——复制 leader session 的 catch-up / 增量推送数据源。
//! WAL tail reader — catch-up and incremental shipping source for replication
//! leader sessions.
//!
//! 设计要点 / Design:
//! - **多段读取（P5.0.2）**：按 LSN 排序归档段 + 活动段，读完当前段自动跨到
//!   下一段；轮转对已打开的 fd 不可见，但读完旧 inode 后会重新扫描目录并接续。
//!   EN: Reads LSN-ordered archive + active segments and crosses segment
//!   boundaries automatically. Rotation is invisible to an open fd; after the old
//!   inode reaches EOF the reader rescans the directory and continues.
//! - **resume 定位 O(segments)**：open 时解析各段元数据，定位首个可能包含
//!   `lsn >= resume_lsn` 的段，再在该段内按完整 CRC 校验定位偏移。
//!   EN: open parses segment metadata and locates the first segment that may
//!   contain `lsn >= resume_lsn`, then finds the offset with full CRC validation.
//! - **撕裂尾停等下轮**：poll 撞见在途写的半条记录不是腐坏——静默停在最后
//!   有效边界，下轮重读。完整记录 CRC 失败则显式 Err。
//!   EN: A torn tail is an in-flight write, not corruption — stop at the valid
//!   boundary and retry later. A CRC failure on a complete record is an error.
//! - **缺段/重叠即错**：段链必须 LSN 连续；空洞意味着保留/轮转事故，不能静默
//!   让 follower 卡在错误位置。 / A gap or overlap in the segment chain is an error.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::encryption::Encryption;
use crate::{
    list_archived_segments, parse_v1_records, parse_v2_records_tail,
    parse_v2_records_tail_with_cipher, WalEntry, WalFormat, WAL_ENCRYPTED_HEADER_LEN,
    WAL_FORMAT_VERSION, WAL_FORMAT_VERSION_ENCRYPTED, WAL_HEADER_LEN, WAL_MAGIC,
};

/// 段元数据（读取器内部使用；按 first_lsn 排序）。
/// EN: Segment metadata (reader-internal; ordered by first_lsn).
#[derive(Clone)]
struct SegmentMeta {
    path: PathBuf,
    first_lsn: Option<u64>,
    last_lsn: Option<u64>,
    format: WalFormat,
    data_start: u64,
    key_id: Option<String>,
}

struct Cursor {
    file: File,
    /// 下一条待解析记录的文件偏移。
    /// EN: Offset of the next record to parse.
    offset: u64,
    format: WalFormat,
    key_id: Option<String>,
    encryption: Option<Arc<Encryption>>,
    path: PathBuf,
    last_lsn: Option<u64>,
}

pub struct WalTailReader {
    base_path: PathBuf,
    encryption: Option<Arc<Encryption>>,
    segments: Vec<SegmentMeta>,
    cursor: Cursor,
    first_available_lsn: u64,
}

impl WalTailReader {
    /// 打开 `path` 并定位到首条 `lsn >= resume_lsn` 的记录（跨全部段）。
    /// EN: Opens `path` and positions at the first record with `lsn >= resume_lsn`
    /// across all segments.
    pub fn open(path: &Path, resume_lsn: u64) -> io::Result<Self> {
        Self::open_with_encryption(path, resume_lsn, None)
    }

    /// 带解密句柄打开（P4.7.3）：v3 段必须提供 keyring。
    /// EN: Open with decryption; v3 segments require the keyring.
    pub fn open_with_encryption(
        path: &Path,
        resume_lsn: u64,
        encryption: Option<Arc<Encryption>>,
    ) -> io::Result<Self> {
        let segments = build_segments(path, encryption.as_deref())?;
        let first_available_lsn = segments
            .iter()
            .find_map(|m| m.first_lsn)
            .unwrap_or(resume_lsn);
        // 首个 last_lsn >= resume 的段；都落后时选最后一段（活动段）等待追加。
        // EN: first segment with last_lsn >= resume; if all are behind, choose the
        // last (active) segment and wait for appends.
        let idx = segments
            .iter()
            .position(|m| m.last_lsn.is_some_and(|l| l >= resume_lsn))
            .unwrap_or_else(|| segments.len().saturating_sub(1));
        let meta = segments[idx].clone();
        let cursor = open_cursor(&meta, resume_lsn, encryption.clone())?;
        Ok(Self {
            base_path: path.to_path_buf(),
            encryption,
            segments,
            cursor,
            first_available_lsn,
        })
    }

    /// 当前最早可读 LSN；没有任何段包含记录时返回打开时的 resume_lsn。
    /// EN: Earliest readable LSN; when no segment holds records, returns the
    /// open-time resume_lsn.
    pub fn first_available_lsn(&self) -> u64 {
        self.first_available_lsn
    }

    /// 增量读取自上次消费位以来的新记录；无新数据返回空 Vec。
    /// EN: Reads records appended since the last consumed offset; empty when idle.
    pub fn poll(&mut self) -> io::Result<Vec<WalEntry>> {
        loop {
            let (entries, consumed) = read_cursor(&mut self.cursor)?;
            if !entries.is_empty() {
                self.cursor.offset += consumed as u64;
                // 记录已消费到的 LSN：轮转后按此寻找下一段，不能沿用 open 时的
                // 段元数据（活动段在 open 之后仍会追加）。
                // EN: Track the last consumed LSN; after rotation the next segment
                // is chosen from this, not from the stale open-time active metadata.
                if let Some(last) = entries.last() {
                    self.cursor.last_lsn = Some(last.lsn);
                }
                return Ok(entries);
            }
            // 活动段未被轮转：确实没有新数据。
            // EN: The active segment was not rotated: genuinely idle.
            if self.cursor.path == self.base_path && !active_rotated(&self.cursor)? {
                return Ok(Vec::new());
            }
            self.refresh_segments()?;
            let Some(next) = self.next_segment() else {
                return Ok(Vec::new());
            };
            // 缺段即错：与已消费位置相邻的下一段必须从 last+1 开始。
            // EN: A missing segment is an error: the adjacent next segment must
            // start exactly at last + 1.
            if let (Some(first), Some(last)) = (next.first_lsn, self.cursor.last_lsn) {
                let expected = last.saturating_add(1);
                if first != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "wal segment chain non-contiguous after consumed lsn: \
                             next starts at {first}, expected {expected}"
                        ),
                    ));
                }
            }
            self.cursor = open_cursor(&next, next.first_lsn.unwrap_or(0), self.encryption.clone())?;
        }
    }

    /// 当前消费偏移（测试与 session 观测用）。
    /// EN: Current consumed offset (for tests and session observation).
    pub fn offset(&self) -> u64 {
        self.cursor.offset
    }

    /// 重新扫描归档段；活动段元数据也刷新（可能已轮转/追加）。
    /// EN: Rescan archives; the active segment metadata is refreshed too.
    fn refresh_segments(&mut self) -> io::Result<()> {
        self.segments = build_segments(&self.base_path, self.encryption.as_deref())?;
        self.first_available_lsn = self
            .segments
            .iter()
            .find_map(|m| m.first_lsn)
            .unwrap_or(self.first_available_lsn);
        Ok(())
    }

    /// 当前段之后的下一个段；优先按 LSN，空活动段作为最后兜底。
    /// EN: Next segment after the cursor, ordered by LSN; an empty active segment
    /// is the final fallback.
    fn next_segment(&self) -> Option<SegmentMeta> {
        let last = self.cursor.last_lsn?;
        let mut next = self
            .segments
            .iter()
            .filter(|m| m.first_lsn.is_some_and(|f| f > last))
            .min_by_key(|m| m.first_lsn)
            .cloned();
        if next.is_none() {
            // 活动段为空时没有 first_lsn；仍要打开它等待后续追加。
            // EN: An empty active segment has no first_lsn; open it to wait for appends.
            next = self
                .segments
                .iter()
                .find(|m| m.path == self.base_path && m.first_lsn.is_none())
                .cloned();
        }
        // 注意不能按 path 过滤：轮转后新活动段与原 path 同名但 inode 不同。
        // EN: Do not filter by path — after rotation the new active segment
        // reuses the same path with a different inode.
        next
    }
}

/// 读取当前游标所在段的新增记录；撕裂尾返回空 + consumed=0。
/// EN: Read new records from the cursor segment; a torn tail yields empty + consumed=0.
fn read_cursor(cursor: &mut Cursor) -> io::Result<(Vec<WalEntry>, usize)> {
    let file_len = cursor.file.metadata()?.len();
    if file_len < cursor.offset {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "wal shrank below consumed offset: len {file_len} < offset {}",
                cursor.offset
            ),
        ));
    }
    if file_len == cursor.offset {
        return Ok((Vec::new(), 0));
    }
    cursor.file.seek(SeekFrom::Start(cursor.offset))?;
    let mut buf = Vec::with_capacity((file_len - cursor.offset) as usize);
    cursor.file.read_to_end(&mut buf)?;
    parse_segment_tail(
        &buf,
        cursor.format,
        cursor.key_id.as_deref(),
        cursor.encryption.as_deref(),
    )
}

/// 打开指定段并定位到 `resume_lsn`（该段内首条 >= resume 的记录）。
/// EN: Open a segment and position at the first record >= resume_lsn within it.
fn open_cursor(
    meta: &SegmentMeta,
    resume_lsn: u64,
    encryption: Option<Arc<Encryption>>,
) -> io::Result<Cursor> {
    let mut file = File::open(&meta.path)?;
    file.seek(SeekFrom::Start(meta.data_start))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let (entries, consumed) = parse_segment_tail(
        &buf,
        meta.format,
        meta.key_id.as_deref(),
        encryption.as_deref(),
    )?;
    let overhead = record_overhead(meta.format);
    let mut offset = meta.data_start;
    let mut positioned = false;
    for e in &entries {
        if e.lsn >= resume_lsn {
            positioned = true;
            break;
        }
        offset += overhead + e.payload.len() as u64;
    }
    if !positioned {
        offset = meta.data_start + consumed as u64;
    }
    file.seek(SeekFrom::Start(offset))?;
    // 尚未消费任何记录时：定位在中段表示 resume 之前的记录已由调用方消费；
    // 定位在段尾表示本段全部已消费。两种情形都不能直接用段元数据 last_lsn
    // （活动段可能已追加）。
    // EN: With no records consumed yet, a mid-segment position means records
    // before resume were already consumed; an end position means the whole
    // segment was consumed. Neither may use the segment's open-time last_lsn.
    let last_lsn = if positioned {
        resume_lsn.checked_sub(1)
    } else {
        meta.last_lsn
    };
    Ok(Cursor {
        file,
        offset,
        format: meta.format,
        key_id: meta.key_id.clone(),
        encryption,
        path: meta.path.clone(),
        last_lsn,
    })
}

/// 构建并校验段链（归档 + 活动），按 first_lsn 排序。
/// EN: Build and validate the segment chain (archives + active), sorted by first_lsn.
fn build_segments(
    base_path: &Path,
    encryption: Option<&Encryption>,
) -> io::Result<Vec<SegmentMeta>> {
    let mut segments = Vec::new();
    for p in list_archived_segments(base_path)? {
        let meta = read_segment_meta(&p, encryption)?;
        if meta.first_lsn.is_some() {
            segments.push(meta);
        }
    }
    if base_path.exists() {
        segments.push(read_segment_meta(base_path, encryption)?);
    } else if segments.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("wal file not found: {}", base_path.display()),
        ));
    }
    segments.sort_by(|a, b| match (a.first_lsn, b.first_lsn) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    let mut prev: Option<&SegmentMeta> = None;
    for meta in &segments {
        if let (Some(first), Some(_last)) = (meta.first_lsn, meta.last_lsn) {
            if let Some(p) = prev {
                let expected = p.last_lsn.expect("previous non-empty").saturating_add(1);
                if first != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "wal segment chain non-contiguous: {} starts at lsn {first}, \
                             expected {expected}",
                            meta.path.display()
                        ),
                    ));
                }
            }
            prev = Some(meta);
        }
    }
    Ok(segments)
}

/// 读取单段元数据（格式/首末 LSN/数据起点/密钥 id）。
/// EN: Read one segment's metadata (format/first-last LSN/data start/key id).
fn read_segment_meta(path: &Path, encryption: Option<&Encryption>) -> io::Result<SegmentMeta> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let (format, data_start, key_id) = detect_format(&mut file, file_len, encryption)?;
    file.seek(SeekFrom::Start(data_start))?;
    let mut buf = Vec::with_capacity((file_len.saturating_sub(data_start)) as usize);
    file.read_to_end(&mut buf)?;
    let (entries, _) = parse_segment_tail(&buf, format, key_id.as_deref(), encryption)?;
    Ok(SegmentMeta {
        path: path.to_path_buf(),
        first_lsn: entries.first().map(|e| e.lsn),
        last_lsn: entries.last().map(|e| e.lsn),
        format,
        data_start,
        key_id,
    })
}

/// 格式识别（与 FileWal::open 同级）。
/// EN: Format detection (same as FileWal::open).
fn detect_format(
    file: &mut File,
    file_len: u64,
    encryption: Option<&Encryption>,
) -> io::Result<(WalFormat, u64, Option<String>)> {
    if file_len < WAL_HEADER_LEN as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "wal file too short: {file_len} < {WAL_HEADER_LEN} (torn debris, not tail-able)"
            ),
        ));
    }
    let mut head = [0u8; WAL_HEADER_LEN];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut head)?;
    if &head[..8] == WAL_MAGIC {
        let version = u32::from_be_bytes(head[8..12].try_into().unwrap());
        match version {
            WAL_FORMAT_VERSION => Ok((WalFormat::V2, WAL_HEADER_LEN as u64, None)),
            WAL_FORMAT_VERSION_ENCRYPTED => {
                if file_len < WAL_ENCRYPTED_HEADER_LEN as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "encrypted wal header truncated",
                    ));
                }
                let mut slot = [0u8; crate::encryption::KEY_ID_LEN];
                file.seek(SeekFrom::Start(WAL_HEADER_LEN as u64))?;
                file.read_exact(&mut slot)?;
                let key_id = crate::encryption::decode_key_id(&slot)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                let enc = encryption.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "encrypted WAL segment requires an encryption keyring",
                    )
                })?;
                if !enc.has_key(&key_id) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("encrypted WAL segment uses unknown key id {key_id:?}"),
                    ));
                }
                Ok((WalFormat::V3, WAL_ENCRYPTED_HEADER_LEN as u64, Some(key_id)))
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported wal format version {other} (expect \
                     {WAL_FORMAT_VERSION} or {WAL_FORMAT_VERSION_ENCRYPTED})"
                ),
            )),
        }
    } else {
        Ok((WalFormat::V1, 0, None))
    }
}

fn parse_segment_tail(
    buf: &[u8],
    format: WalFormat,
    key_id: Option<&str>,
    encryption: Option<&Encryption>,
) -> io::Result<(Vec<WalEntry>, usize)> {
    match (format, encryption, key_id) {
        (WalFormat::V3, Some(enc), Some(key)) => parse_v2_records_tail_with_cipher(buf, enc, key),
        (WalFormat::V3, _, _) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "encrypted WAL segment requires an encryption handle and key id",
        )),
        (WalFormat::V2, _, _) => parse_v2_records_tail(buf),
        (WalFormat::V1, _, _) => Ok(parse_v1_records(buf)),
    }
}

fn record_overhead(format: WalFormat) -> u64 {
    match format {
        WalFormat::V1 => 12,
        WalFormat::V2 => 16,
        WalFormat::V3 => 16 + 12 + 16,
    }
}

/// 活动文件是否已被轮转（rename 后新 inode）。Unix 用 inode 比较；其他平台
/// 退化为「路径长度小于已消费偏移」的保守判断。
/// EN: Whether the active file was rotated (new inode after rename). Unix compares
/// inodes; other platforms conservatively check path length against the offset.
fn active_rotated(cursor: &Cursor) -> io::Result<bool> {
    let path_meta = match std::fs::metadata(&cursor.path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(true),
        Err(e) => return Err(e),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let open_meta = cursor.file.metadata()?;
        Ok(path_meta.ino() != open_meta.ino())
    }
    #[cfg(not(unix))]
    {
        // 限制：非 Unix 平台无法比较 inode；若轮转后新活动段在下次 poll 前
        // 已写入足够数据，长度启发式会误判为未轮转并静默停住。生产平台为
        // Linux/macOS（Unix 分支），非 Unix 仅用于编译/测试，不承诺轮转检测。
        // EN: Limitation: non-Unix platforms cannot compare inodes. If the new
        // active segment grows before the next poll, the length heuristic may
        // conclude "not rotated" and silently stall. Production is Unix;
        // non-Unix is compile/test-only and does not promise rotation detection.
        Ok(path_meta.len() < cursor.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileWal, Wal};

    fn temp_wal_path() -> std::path::PathBuf {
        crate::unique_temp_path("houyi-waltail-test", "wal")
    }

    /// 手工编码 v2 记录（含正确 CRC），用于撕裂尾补全场景。
    /// EN: Hand-encodes a v2 record (with correct CRC) for torn-tail completion tests.
    fn v2_record_bytes(lsn: u64, payload: &[u8]) -> Vec<u8> {
        let mut h = crc32fast::Hasher::new();
        h.update(&lsn.to_be_bytes());
        h.update(&(payload.len() as u32).to_be_bytes());
        h.update(payload);
        let crc = h.finalize();
        let mut r = Vec::new();
        r.extend_from_slice(&lsn.to_be_bytes());
        r.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        r.extend_from_slice(&crc.to_be_bytes());
        r.extend_from_slice(payload);
        r
    }

    fn drain(tail: &mut WalTailReader) -> Vec<WalEntry> {
        let mut out = Vec::new();
        loop {
            let batch = tail.poll().unwrap();
            if batch.is_empty() {
                return out;
            }
            out.extend(batch);
        }
    }

    #[test]
    fn open_from_zero_reads_all_records() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            let _ = wal.append(b"alpha".to_vec());
            let _ = wal.append(b"beta".to_vec());
            let _ = wal.append(b"gamma".to_vec());
        }
        let mut tail = WalTailReader::open(&path, 0).unwrap();
        let entries = tail.poll().unwrap();
        let payloads: Vec<&[u8]> = entries.iter().map(|e| e.payload.as_slice()).collect();
        assert_eq!(
            payloads,
            vec![b"alpha".as_slice(), b"beta".as_slice(), b"gamma".as_slice()]
        );
        assert_eq!(entries[0].lsn, 0);
        assert_eq!(entries[2].lsn, 2);
        assert!(tail.poll().unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_resumes_from_first_lsn_ge_resume() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            for i in 0..5u8 {
                let _ = wal.append(vec![i]);
            }
        }
        let mut tail = WalTailReader::open(&path, 3).unwrap();
        let entries = tail.poll().unwrap();
        let lsns: Vec<u64> = entries.iter().map(|e| e.lsn).collect();
        assert_eq!(lsns, vec![3, 4]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_caught_up_positions_at_end_then_picks_up_appends() {
        let path = temp_wal_path();
        let mut wal = {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            let _ = wal.append(b"a".to_vec());
            let _ = wal.append(b"b".to_vec());
            wal
        };
        let mut tail = WalTailReader::open(&path, 2).unwrap();
        assert!(tail.poll().unwrap().is_empty());
        let _ = wal.append(b"c".to_vec());
        let entries = tail.poll().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lsn, 2);
        assert_eq!(entries[0].payload, b"c");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn torn_tail_waits_then_reads_completed_record() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            let _ = wal.append(b"good".to_vec());
        }
        let full = v2_record_bytes(1, b"late");
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&full[..10]).unwrap();
        }
        let mut tail = WalTailReader::open(&path, 0).unwrap();
        let entries = tail.poll().unwrap();
        assert_eq!(entries.len(), 1, "torn record must wait");
        assert_eq!(entries[0].payload, b"good");
        let offset_at_torn = tail.offset();
        assert!(tail.poll().unwrap().is_empty(), "still torn, no progress");
        assert_eq!(
            tail.offset(),
            offset_at_torn,
            "offset must not advance past torn record"
        );
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&full[10..]).unwrap();
        }
        let entries = tail.poll().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lsn, 1);
        assert_eq!(entries[0].payload, b"late");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v1_legacy_file_auto_detected() {
        let path = temp_wal_path();
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).unwrap();
            for (lsn, p) in [(0u64, b"old-a".as_slice()), (1, b"old-b"), (2, b"old-c")] {
                f.write_all(&lsn.to_be_bytes()).unwrap();
                f.write_all(&(p.len() as u32).to_be_bytes()).unwrap();
                f.write_all(p).unwrap();
            }
        }
        let mut tail = WalTailReader::open(&path, 1).unwrap();
        let entries = tail.poll().unwrap();
        let lsns: Vec<u64> = entries.iter().map(|e| e.lsn).collect();
        assert_eq!(lsns, vec![1, 2]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_error() {
        let path = temp_wal_path();
        let _ = std::fs::remove_file(&path);
        assert!(WalTailReader::open(&path, 0).is_err());
    }

    #[test]
    fn short_file_is_error() {
        let path = temp_wal_path();
        std::fs::write(&path, b"tiny").unwrap();
        let err = WalTailReader::open(&path, 0).err().unwrap();
        assert!(err.to_string().contains("too short"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_record_is_error_not_silent_stall() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            let _ = wal.append(b"good".to_vec());
        }
        let mut bad = v2_record_bytes(1, b"rotten");
        bad[12] ^= 0xFF;
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&bad).unwrap();
        }
        let err = WalTailReader::open(&path, 0).err().unwrap();
        assert!(err.to_string().contains("checksum mismatch"), "got: {err}");
        let _ = std::fs::remove_file(&path);

        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            let _ = wal.append(b"good".to_vec());
        }
        let mut tail = WalTailReader::open(&path, 0).unwrap();
        assert_eq!(tail.poll().unwrap().len(), 1);
        let mut bad = v2_record_bytes(1, b"rotten");
        bad[12] ^= 0xFF;
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&bad).unwrap();
        }
        let err = tail.poll().unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn shrunk_file_is_error() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            let _ = wal.append(b"x".to_vec());
        }
        let mut tail = WalTailReader::open(&path, 0).unwrap();
        assert_eq!(tail.poll().unwrap().len(), 1);
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(0).unwrap();
        drop(f);
        let err = tail.poll().unwrap_err();
        assert!(err.to_string().contains("wal shrank"));
        let _ = std::fs::remove_file(&path);
    }

    /// P5.0.2：跨轮转段读取（open 时已有归档 + 活动段）。
    /// EN: P5.0.2 — read across rotated segments (archives + active at open).
    #[test]
    fn reads_across_rotated_segments() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            for i in 0..3u8 {
                let _ = wal.append(vec![i]);
            }
            let _ = wal.rotate_archive().unwrap();
            for i in 3..5u8 {
                let _ = wal.append(vec![i]);
            }
        }
        let mut tail = WalTailReader::open(&path, 0).unwrap();
        let entries = drain(&mut tail);
        let lsns: Vec<u64> = entries.iter().map(|e| e.lsn).collect();
        assert_eq!(lsns, vec![0, 1, 2, 3, 4]);
        assert_eq!(tail.first_available_lsn(), 0);
        let _ = std::fs::remove_file(&path);
        cleanup_archives(&path);
    }

    /// P5.0.2：打开后发生轮转，poll 必须跨到新活动段。
    /// EN: P5.0.2 — rotation after open; poll must continue into the new active segment.
    #[test]
    fn poll_crosses_rotation_after_open() {
        let path = temp_wal_path();
        let mut wal = {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            let _ = wal.append(b"a".to_vec());
            wal
        };
        let mut tail = WalTailReader::open(&path, 0).unwrap();
        assert_eq!(tail.poll().unwrap().len(), 1);
        let _ = wal.rotate_archive().unwrap();
        let _ = wal.append(b"b".to_vec());
        let entries = drain(&mut tail);
        let lsns: Vec<u64> = entries.iter().map(|e| e.lsn).collect();
        assert_eq!(lsns, vec![1]);
        let _ = std::fs::remove_file(&path);
        cleanup_archives(&path);
    }

    /// P5.0.2：打开后保留删除尚未消费的中间段，poll 必须报缺段。
    /// EN: P5.0.2 — if retention deletes an unconsumed middle segment after
    /// open, poll must report the gap instead of silently skipping it.
    #[test]
    fn poll_detects_segment_gap_after_open() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            for _ in 0..2 {
                let _ = wal.append(b"a".to_vec());
            }
            let _ = wal.rotate_archive().unwrap();
            for _ in 0..2 {
                let _ = wal.append(b"b".to_vec());
            }
            let _ = wal.rotate_archive().unwrap();
            for _ in 0..2 {
                let _ = wal.append(b"c".to_vec());
            }
            let _ = wal.rotate_archive().unwrap();
            let _ = wal.append(b"d".to_vec());
        }
        let archives = crate::list_archived_segments(&path).unwrap();
        assert_eq!(archives.len(), 3);
        let mut tail = WalTailReader::open(&path, 0).unwrap();
        assert_eq!(tail.poll().unwrap().len(), 2);
        std::fs::remove_file(&archives[1]).unwrap();
        let err = tail.poll().unwrap_err();
        assert!(
            err.to_string().contains("non-contiguous"),
            "expected a gap error, got: {err}"
        );
        let _ = std::fs::remove_file(&path);
        cleanup_archives(&path);
    }

    /// P5.0.2：段链中间缺失必须报错，而不是静默停在错误位置。
    /// EN: P5.0.2 — a missing middle segment must error rather than silently stall.
    #[test]
    fn missing_middle_segment_is_error() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            for _ in 0..2 {
                let _ = wal.append(b"a".to_vec());
            }
            let _ = wal.rotate_archive().unwrap();
            for _ in 0..2 {
                let _ = wal.append(b"b".to_vec());
            }
            let _ = wal.rotate_archive().unwrap();
            for _ in 0..2 {
                let _ = wal.append(b"c".to_vec());
            }
            let _ = wal.rotate_archive().unwrap();
        }
        let archives = crate::list_archived_segments(&path).unwrap();
        assert_eq!(archives.len(), 3);
        std::fs::remove_file(&archives[1]).unwrap();
        let err = WalTailReader::open(&path, 0).err().unwrap();
        assert!(err.to_string().contains("non-contiguous"), "got: {err}");
        let _ = std::fs::remove_file(&path);
        cleanup_archives(&path);
    }

    fn cleanup_archives(path: &std::path::Path) {
        for p in crate::list_archived_segments(path).unwrap_or_default() {
            let _ = std::fs::remove_file(p);
        }
    }
}
