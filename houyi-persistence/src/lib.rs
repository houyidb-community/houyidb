use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use encryption::Encryption;

pub mod collection_snapshot_file;
pub mod collection_wal;
/// P4.5.4 故障注入点（feature `failpoints`；未启用时为空操作）。
/// Failpoint hooks — no-op unless the `failpoints` feature is enabled.
pub mod failpoint;
pub mod graph_snapshot_file;
pub mod ha_level;
// P4.7.3：静态加密（AES-256-GCM + 可插拔 keyring）。
// EN: P4.7.3: at-rest encryption (AES-256-GCM + pluggable keyring).
pub mod encryption;
// P4.6.1：复制握手日志前缀指纹（协议 v4）。
// EN: P4.6.1: replication handshake log-prefix fingerprint (protocol v4).
pub mod log_fingerprint;
// P4.5.5：复制层网络/时钟换型（生产 tokio，turmoil 测试走仿真实现）。
// EN: P4.5.5: replication network/clock swap (tokio in production, turmoil sim in tests).
pub mod net_runtime;
pub mod replication_follower;
pub mod replication_leader;
pub mod vector_snapshot;
pub mod vector_wal;
pub mod wal_async;
pub mod wal_replication;
pub mod wal_sidecar;
pub mod wal_tail;

// P4.5.0 属性化测试（proptest）：编解码纪律的不变量网，仅测试构建。
// EN: P4.5.0 property tests (proptest): invariant net for codec discipline, test builds only.
#[cfg(test)]
mod proptests;

pub use wal_async::WalAsyncHandle;

/// 命中一个故障注入点（见 [`failpoint`]）；未启用 `failpoints` feature 时为空操作。
/// Hit a failpoint (see [`failpoint`]); a no-op without the `failpoints` feature.
///
/// 用法 / Usage: `crate::fail_point!("wal_append_pre");` —— 展开为
/// `hit(name)?`，故只能用在返回 `io::Result<_>` 或 `From<io::Error>` 可达的
/// 函数里。EN: expands to `hit(name)?`, so the enclosing function must return a
/// result convertible from `io::Error`.
#[macro_export]
macro_rules! fail_point {
    ($name:literal) => {
        $crate::failpoint::hit($name)?
    };
}

use wal_sidecar::{read_wal_next_lsn_meta, wal_next_lsn_meta_path, write_wal_next_lsn_meta_atomic};

/// fuzz 专用入口（feature `fuzz-api`，P4.5.1）。
///
/// WHY 独立模块而非直接 `pub`：解析器本体保持 `pub(crate)`，生产 API 面不为测试
/// 扩大；独立 fuzz workspace（`fuzz/`）无法访问 crate 内部可见性，故用一个
/// feature 门控的薄封装把字节切片入口暴露出去。默认 feature 关闭，release 构建
/// 与生产路径完全不含此模块。
/// EN: Fuzz-only byte-slice entry points, gated behind the `fuzz-api` feature so
/// the production surface stays unchanged. The parsers remain `pub(crate)`.
#[cfg(feature = "fuzz-api")]
pub mod fuzz_api {
    use crate::WalEntry;

    /// 整文件解析（v1/v2 自动判别）：任意字节必须只返回「有效前缀」或 `Err`。
    /// EN: Whole-file parse (v1/v2 auto-detect): any bytes yield only a valid prefix or `Err`.
    pub fn parse_wal_file(data: &[u8]) -> std::io::Result<Vec<WalEntry>> {
        crate::parse_wal_file_entries(data)
    }

    /// v2 尾部增量解析：完整记录 CRC 不匹配 = `Err`（腐坏）；头/payload 不完整
    /// = 静默撕裂（返回已确认前缀）。
    /// EN: v2 tail incremental parse: a complete-record CRC mismatch = `Err` (corruption);
    /// an incomplete header/payload = silent torn tail (returns the confirmed prefix).
    pub fn parse_v2_tail(data: &[u8]) -> std::io::Result<(Vec<WalEntry>, usize)> {
        crate::parse_v2_records_tail(data)
    }
}

/// WAL 记录（v0.1）.
/// WAL entry for v0.1 alpha.
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub lsn: u64,
    pub payload: Vec<u8>,
}

/// 物理日志接口（后续扩展 XOR Delta/复制）.
/// Physical WAL contract with future XOR delta replication.
pub trait Wal: Send + Sync {
    fn append(&mut self, payload: Vec<u8>) -> WalEntry;
}

#[derive(Default)]
pub struct InMemoryWal {
    next_lsn: u64,
    logs: Vec<WalEntry>,
}

// ── WAL 磁盘格式（P4.1：版本号 + 每记录 CRC32）─────────────────────────────
//
// v2 文件布局：`[magic:8][version:u32 BE]` 文件头 + 记录序列
//   `[lsn:u64 BE][len:u32 BE][crc32:u32 BE][payload]`，
//   CRC32(IEEE) 覆盖 lsn‖len‖payload（头字段一并校验，LSN 腐坏也可检出）。
// v1（遗留）无文件头，记录为 `[lsn][len][payload]`，无校验。
//
// 兼容策略（升级不迁移）：
// - 新文件一律 v2（open 时写文件头）；
// - 已存在的 v1 文件继续按 v1 追加（单文件不混合格式），经 rotate/truncate
//   自然换代为 v2——生产 WAL 有截断/滚动周期，旧格式不会长期滞留；
// - 读路径按文件头逐段识别，归档段 v1/v2 混存可重放。
// 降级不兼容：旧二进制读 v2 文件会把 magic 当 LSN 解析出垃圾，属预期（版本号
// 的意义即让新版本知道自己读的是什么）。
//
// EN: v2 = 12-byte file header (magic + version) then records with per-record
// CRC32 over lsn‖len‖payload. Legacy v1 files keep appending in v1 (never mix
// formats in one file) and age out via rotate/truncate; readers detect the
// format per segment. Downgrade to old binaries is intentionally unsupported.

/// v2 文件魔数（ASCII "HOUYIWAL"）。
/// pub(crate)：WalTailReader 按同一魔数识别文件格式。
/// EN: v2 file magic (ASCII "HOUYIWAL"). pub(crate): WalTailReader detects format by the
/// same magic.
pub(crate) const WAL_MAGIC: &[u8; 8] = b"HOUYIWAL";
/// 当前明文 WAL 磁盘格式版本。
/// EN: Current plaintext WAL on-disk format version.
pub(crate) const WAL_FORMAT_VERSION: u32 = 2;
/// 加密 WAL 磁盘格式版本（P4.7.3）：文件头增加 16B key id。
/// EN: Encrypted WAL on-disk format version (P4.7.3): header adds a 16B key id.
pub(crate) const WAL_FORMAT_VERSION_ENCRYPTED: u32 = 3;
/// v2 文件头长度：magic(8) + version(4)。
/// EN: v2 header length: magic(8) + version(4).
pub(crate) const WAL_HEADER_LEN: usize = 12;
/// v3 文件头长度：magic(8) + version(4) + key_id(16)。
/// EN: v3 header length: magic(8) + version(4) + key_id(16).
pub(crate) const WAL_ENCRYPTED_HEADER_LEN: usize = 28;

/// WAL 段格式（按文件头逐段识别；同一目录允许混合历史）。
/// Per-segment WAL format; mixed history is allowed in one directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WalFormat {
    /// 遗留无校验格式（只读为主，继续按 v1 追加）。
    /// EN: Legacy checksum-less format (mostly read-only; appends continue as v1).
    V1,
    /// 明文 + 每记录 CRC32。
    /// EN: Plaintext + per-record CRC32.
    V2,
    /// 加密（AES-256-GCM）+ 每记录 CRC32（撕裂尾检测）。
    /// EN: Encrypted (AES-256-GCM) + per-record CRC32 (torn-tail detection).
    V3,
}

impl WalFormat {
    fn has_crc(self) -> bool {
        !matches!(self, Self::V1)
    }

    fn is_encrypted(self) -> bool {
        matches!(self, Self::V3)
    }
}

/// 一条记录的 CRC32（覆盖 lsn‖len‖payload）。
/// EN: CRC32 of one record (covers lsn‖len‖payload).
fn record_crc32(lsn: u64, payload: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(&lsn.to_be_bytes());
    h.update(&(payload.len() as u32).to_be_bytes());
    h.update(payload);
    h.finalize()
}

/// 写文件头（单次 write_all，避免半头状态窗口）。
/// Write the segment header in one syscall to avoid a half-header window.
fn write_wal_header(
    file: &mut File,
    format: WalFormat,
    encryption: Option<&Encryption>,
) -> io::Result<()> {
    let mut header = Vec::with_capacity(WAL_ENCRYPTED_HEADER_LEN);
    header.extend_from_slice(WAL_MAGIC);
    match format {
        WalFormat::V1 => return Ok(()),
        WalFormat::V2 => {
            header.extend_from_slice(&WAL_FORMAT_VERSION.to_be_bytes());
        }
        WalFormat::V3 => {
            let enc = encryption.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cannot write encrypted WAL without an encryption handle",
                )
            })?;
            header.extend_from_slice(&WAL_FORMAT_VERSION_ENCRYPTED.to_be_bytes());
            let id = crate::encryption::encode_key_id(enc.active_id())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
            header.extend_from_slice(&id);
        }
    }
    file.write_all(&header)
}

/// 编码一条记录。
///
/// v3 的 `len` 是 `nonce‖ciphertext‖tag` 的长度，CRC 同样覆盖这段密文
/// （只用于撕裂尾/位腐烂检测；真实性由 GCM tag 保证）。
/// EN: v3 stores `nonce||ciphertext||tag`; CRC covers that blob (torn-tail
/// detection only — authenticity comes from the GCM tag).
fn encode_record(
    lsn: u64,
    payload: &[u8],
    format: WalFormat,
    encryption: Option<&Encryption>,
) -> io::Result<Vec<u8>> {
    let body: Vec<u8> = if format.is_encrypted() {
        encryption
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cannot write encrypted WAL without an encryption handle",
                )
            })?
            .encrypt(payload)?
    } else {
        payload.to_vec()
    };
    let mut record = Vec::with_capacity(if format.has_crc() { 16 } else { 12 } + body.len());
    record.extend_from_slice(&lsn.to_be_bytes());
    record.extend_from_slice(&(body.len() as u32).to_be_bytes());
    if format.has_crc() {
        record.extend_from_slice(&record_crc32(lsn, &body).to_be_bytes());
    }
    record.extend_from_slice(&body);
    Ok(record)
}

/// 解析 WAL 文件字节：按文件头识别 v1/v2。
///
/// 撕裂写容忍（kill -9 / 掉电后尾部半条记录属正常）：v2 在「头部不完整 /
/// payload 不完整 / CRC 不匹配」处停止并 warn，返回之前的有效前缀——
/// 与 PostgreSQL 的 end-of-WAL 语义一致：校验失败点即日志终点。
/// EN: Torn-tail tolerant — v2 parsing stops (with a warn) at an incomplete
/// header/payload or CRC mismatch and returns the valid prefix; the first
//  checksum failure is by definition the end of the durable log.
// pub(crate)：WalTailReader（wal_tail.rs）复用同一磁盘格式解析，消除第二处手写解码。
// 无密钥入口仅测试与 fuzz 使用（生产一律走带 cipher 的版本）。
#[cfg(any(test, feature = "fuzz-api"))]
pub(crate) fn parse_wal_file_entries(data: &[u8]) -> io::Result<Vec<WalEntry>> {
    parse_wal_file_entries_with_cipher(data, None)
}

/// 带解密句柄的整文件解析（v3 必需；v1/v2 忽略）。
/// Parse a whole file; encrypted (v3) segments require the cipher.
pub(crate) fn parse_wal_file_entries_with_cipher(
    data: &[u8],
    encryption: Option<&Encryption>,
) -> io::Result<Vec<WalEntry>> {
    parse_wal_file_entries_with_len(data, encryption).map(|(entries, _)| entries)
}

/// 解析整文件并同时返回 **有效前缀字节数**（含文件头）——`FileWal::open`
/// 据此把活动文件截断到日志终点，使恢复后追加的记录始终落在重放可达区间。
/// EN: Also returns the valid-prefix byte length so `open` can truncate the
/// active file at end-of-log; post-recovery appends then stay replayable.
fn parse_wal_file_entries_with_len(
    data: &[u8],
    encryption: Option<&Encryption>,
) -> io::Result<(Vec<WalEntry>, usize)> {
    if data.len() >= WAL_HEADER_LEN && &data[..8] == WAL_MAGIC {
        let version = u32::from_be_bytes(data[8..12].try_into().unwrap());
        return match version {
            WAL_FORMAT_VERSION => {
                let (entries, consumed) = parse_v2_records(&data[WAL_HEADER_LEN..], None)?;
                Ok((entries, WAL_HEADER_LEN + consumed))
            }
            WAL_FORMAT_VERSION_ENCRYPTED => {
                if data.len() < WAL_ENCRYPTED_HEADER_LEN {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "encrypted wal header truncated",
                    ));
                }
                let enc = encryption.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "encrypted WAL segment requires an encryption keyring",
                    )
                })?;
                let key_id = crate::encryption::decode_key_id(&data[12..WAL_ENCRYPTED_HEADER_LEN])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                if !enc.has_key(&key_id) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("encrypted WAL segment uses unknown key id {key_id:?}"),
                    ));
                }
                let (entries, consumed) =
                    parse_v2_records(&data[WAL_ENCRYPTED_HEADER_LEN..], Some((enc, &key_id)))?;
                Ok((entries, WAL_ENCRYPTED_HEADER_LEN + consumed))
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported wal format version {other} (expect {WAL_FORMAT_VERSION} or \
                     {WAL_FORMAT_VERSION_ENCRYPTED})"
                ),
            )),
        };
    }
    Ok(parse_v1_records(data))
}

/// 记录解析：撕裂尾 / CRC 失败即日志终点（warn 后返回有效前缀）。
/// `decrypt` 为 Some 时按 v3 解密；完整记录的校验失败一律是腐坏（Err）。
/// 返回 `(条目, 消费字节数)`。
/// EN: Record parsing: a torn tail or CRC failure is the end of the log (warn, then
/// return the valid prefix). When `decrypt` is Some, v3 records are decrypted; a
/// validation failure on a complete record is always corruption (Err).
/// Returns `(entries, bytes consumed)`.
fn parse_v2_records(
    data: &[u8],
    decrypt: Option<(&Encryption, &str)>,
) -> io::Result<(Vec<WalEntry>, usize)> {
    parse_v2_records_inner(data, true, decrypt)
}

/// WalTailReader 增量喂法专用：撕裂尾（头/payload 不完整）= writer 在途写，
/// 属正常背压点，静默停等下轮（返回的消费字节数即停等边界）；但**完整记录的
/// CRC 不匹配是腐坏而非撕裂**——两种失败语义不同：撕裂会随 writer 补全自愈，
/// 腐坏永远不会。静默混同会让 tail reader 在腐坏点无声停摆（leader 继续追加，
/// follower 永远追不上且无一条告警），故 tail 模式对 CRC 不匹配返回 Err，
/// 由 session 层断连报错（重连后错误复现 = 可见的故障信号，而非静默停滞）。
/// EN: Quiet variant for incremental tailing — an incomplete header/payload is
/// an in-flight writer append and waits silently, but a CRC mismatch on a
/// fully-present record is corruption and surfaces as Err; conflating the two
/// would stall the tail reader forever without a single warning.
pub(crate) fn parse_v2_records_tail(data: &[u8]) -> io::Result<(Vec<WalEntry>, usize)> {
    parse_v2_records_inner(data, false, None)
}

/// 带解密的增量解析（v3）；`key_id` 来自文件头。
/// Incremental parser with decryption for v3 segments.
pub(crate) fn parse_v2_records_tail_with_cipher(
    data: &[u8],
    encryption: &Encryption,
    key_id: &str,
) -> io::Result<(Vec<WalEntry>, usize)> {
    parse_v2_records_inner(data, false, Some((encryption, key_id)))
}

fn parse_v2_records_inner(
    data: &[u8],
    warn_on_torn: bool,
    decrypt: Option<(&Encryption, &str)>,
) -> io::Result<(Vec<WalEntry>, usize)> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    // 长度字段来自（可能腐坏的）磁盘数据：记录端点一律 u64 计算，32-bit
    // 目标上 `idx + 16 + len` 的 usize 加法可溢出（debug panic/release 回绕）。
    // EN: Length fields come from (possibly corrupt) disk data: record endpoints are
    // computed in u64, since `idx + 16 + len` usize addition can overflow on 32-bit
    // targets (debug panic / release wrap).
    while data.len() - idx >= 16 {
        let lsn = u64::from_be_bytes(data[idx..idx + 8].try_into().unwrap());
        let len = u32::from_be_bytes(data[idx + 8..idx + 12].try_into().unwrap()) as usize;
        let crc = u32::from_be_bytes(data[idx + 12..idx + 16].try_into().unwrap());
        let end = idx as u64 + 16 + len as u64;
        if end > data.len() as u64 {
            if warn_on_torn {
                tracing::warn!(
                    offset = idx,
                    lsn,
                    "wal v2: torn tail (incomplete payload), treating as end of log"
                );
            }
            break;
        }
        let end = end as usize;
        let blob = &data[idx + 16..end];
        let actual = record_crc32(lsn, blob);
        if actual != crc {
            // 加密段：完整记录的校验失败只能是腐坏（撕裂写会先表现为长度不
            // 完整），一律 Err，绝不把中间损坏当作日志终点静默截断。
            // EN: a fully-present encrypted record that fails CRC is corruption,
            // never a torn tail — surface it instead of truncating silently.
            if decrypt.is_some() || !warn_on_torn {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "wal checksum mismatch at offset {idx} (lsn {lsn}, \
                         expected_crc {crc:#010x}, actual {actual:#010x}) — corruption, \
                         not torn tail"
                    ),
                ));
            }
            // 冷重放（整文件、写已停）：校验失败点即日志终点，与
            // PostgreSQL end-of-WAL 语义一致。
            // EN: Cold replay (whole file, writes stopped): the first failed check is the
            // end of the log, matching PostgreSQL end-of-WAL semantics.
            tracing::warn!(
                offset = idx,
                lsn,
                expected_crc = crc,
                actual_crc = actual,
                "wal v2: checksum mismatch, treating as end of log"
            );
            break;
        }
        let payload = match decrypt {
            Some((enc, key_id)) => enc.decrypt(key_id, blob)?,
            None => blob.to_vec(),
        };
        out.push(WalEntry { lsn, payload });
        idx = end;
    }
    // 残余 1..16 字节（循环条件自然退出）= 撕裂的记录头；break 路径已各自
    // warn（break 前必有 idx+16 <= len，此处条件不成立，不会重复告警）。
    // EN: Leftover 1..16 bytes (the loop exits naturally) = a torn record header; the
    // break paths each warn already (before any break idx+16 <= len, so this
    // condition cannot re-alert).
    if warn_on_torn && data.len() > idx && idx + 16 > data.len() {
        tracing::warn!(
            offset = idx,
            remaining = data.len() - idx,
            "wal v2: torn tail (incomplete record header), treating as end of log"
        );
    }
    Ok((out, idx))
}

/// v1 遗留记录解析（[lsn:u64][len:u32][payload] *，无校验）。
/// 返回 `(条目, 消费字节数)`；尾部不完整记录同样视为日志终点。
/// pub(crate)：WalTailReader 增量喂 v1 遗留文件复用。
/// EN: v1 legacy record parsing ([lsn:u64][len:u32][payload]*, no checksums).
/// Returns `(entries, bytes consumed)`; an incomplete tail record also marks the
/// end of the log. pub(crate): reused by WalTailReader for v1 incremental feeds.
pub(crate) fn parse_v1_records(data: &[u8]) -> (Vec<WalEntry>, usize) {
    let mut out = Vec::new();
    let mut idx = 0usize;
    // 与 v2 同理：长度字段可能腐坏，端点一律 u64 计算，32-bit 目标上溢出
    // 降级为「日志终点」而非 debug panic / release 回绕。
    // EN: Same as v2: length fields may be corrupt; endpoints are computed in u64 so
    // 32-bit overflow degrades to "end of log" instead of a debug panic / release wrap.
    while data.len() - idx >= 12 {
        let lsn = u64::from_be_bytes(data[idx..idx + 8].try_into().unwrap());
        let len = u32::from_be_bytes(data[idx + 8..idx + 12].try_into().unwrap()) as usize;
        let end = idx as u64 + 12 + len as u64;
        if end > data.len() as u64 {
            break;
        }
        let end = end as usize;
        out.push(WalEntry {
            lsn,
            payload: data[idx + 12..end].to_vec(),
        });
        idx = end;
    }
    (out, idx)
}

/// 从单文件解析 WAL 记录（v1/v2/v3 按文件头自动识别）。
/// EN: Parses WAL records from a single file (v1/v2/v3 detected from the header).
fn read_records_from_path(
    path: &Path,
    encryption: Option<&Encryption>,
) -> io::Result<Vec<WalEntry>> {
    let mut data = Vec::new();
    let mut f = File::open(path)?;
    f.read_to_end(&mut data)?;
    parse_wal_file_entries_with_cipher(&data, encryption)
}

/// 归档段文件名形如 `{stem}.wal.archived.{nanos}`，与 `with_extension("wal.archived.*")` 一致（活动文件可为 `*.wal` 或 `*.log` 等）。
/// EN: Archive segment names look like `{stem}.wal.archived.{nanos}`, matching
/// `with_extension("wal.archived.*")` (the active file may be `*.wal`, `*.log`, etc.).
fn archived_segment_marker(wal_path: &Path) -> Option<String> {
    let stem = wal_path.file_stem()?.to_str()?;
    Some(format!("{stem}.wal.archived."))
}

/// 列出 `wal_path` 同目录下归档段（按文件名排序，先于活动 `wal_path` 重放）。
/// EN: Lists archive segments beside `wal_path` (sorted by file name, replayed before
/// the active `wal_path`).
fn list_archived_segments(wal_path: &Path) -> io::Result<Vec<PathBuf>> {
    let Some(marker) = archived_segment_marker(wal_path) else {
        return Ok(Vec::new());
    };
    let parent = parent_dir_or_dot(wal_path);
    let mut paths: Vec<PathBuf> = fs::read_dir(parent)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with(&marker))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// 一次性迁移标记：rename 前落盘，归档段删完后移除。崩溃恢复据此区分
/// 「v3 活动段已含全量历史、归档是重复副本」与「普通轮转、归档仍有效」。
/// EN: one-shot migration marker written before the rename and removed after
/// the archives are deleted; crash recovery uses it to tell stale plaintext
/// duplicates from legitimate rotation archives.
fn encryption_migration_marker_path(wal_path: &Path) -> PathBuf {
    wal_path.with_extension("wal.encrypt.migrating")
}

/// 父目录归一：`Path::new("data.wal").parent()` 是 `Some("")` 而非 None，
/// 直接打开 "" 做 fsync/read_dir 会 ENOENT。
/// EN: normalize an empty parent (bare file names) to "." — `parent()` of a
/// bare file name is `Some("")`, not None.
pub fn parent_dir_or_dot(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// 目录项 fsync：迁移顺序（标记→rename→删归档）的每一步都要落盘，
/// 否则崩溃后可能「删了归档但 rename 未持久」= 丢历史。失败上抛。
/// EN: fsync the directory between migration steps; failure is fatal — an
/// unpersisted rename followed by archive deletion would lose history.
fn sync_parent_dir(path: &Path) -> io::Result<()> {
    fs::File::open(parent_dir_or_dot(path))?.sync_all()
}

/// 读 v3 段头里的密钥 id（轮换判定用）。 / EN: key id pinned by a v3 header.
fn read_v3_key_id(path: &Path) -> io::Result<String> {
    let mut head = [0u8; WAL_ENCRYPTED_HEADER_LEN];
    fs::File::open(path)?.read_exact(&mut head)?;
    crate::encryption::decode_key_id(&head[WAL_HEADER_LEN..WAL_ENCRYPTED_HEADER_LEN])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

/// 空 v3 段直接换头（无记录可归档）：截断后写新活动密钥的段头。
/// EN: re-head an empty v3 segment in place (nothing to archive).
fn rehead_empty_v3_segment(wal: &mut FileWal, enc: &Encryption) -> io::Result<()> {
    let _ = wal.file.take();
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(wal.path())?;
        write_wal_header(&mut f, WalFormat::V3, Some(enc))?;
        f.sync_all()?;
    }
    sync_parent_dir(wal.path())?;
    let mut reopened = OpenOptions::new()
        .read(true)
        .append(true)
        .open(wal.path())?;
    reopened.seek(SeekFrom::End(0))?;
    wal.file = Some(reopened);
    wal.format = WalFormat::V3;
    wal.active_bytes = WAL_ENCRYPTED_HEADER_LEN as u64;
    wal.active_first_lsn = None;
    if wal.next_lsn > 0 {
        write_wal_next_lsn_meta_atomic(wal.path(), wal.next_lsn)?;
    }
    Ok(())
}

/// 合并归档段 + 活动 WAL 中的全部记录（按 LSN 排序）。
/// EN: Merges all records from archive segments and the active WAL (sorted by LSN).
fn merge_all_segments(
    wal_path: &Path,
    encryption: Option<&Encryption>,
) -> io::Result<Vec<WalEntry>> {
    let mut all = Vec::new();
    for p in list_archived_segments(wal_path)? {
        all.extend(read_records_from_path(&p, encryption)?);
    }
    if wal_path.exists() {
        all.extend(read_records_from_path(wal_path, encryption)?);
    }
    all.sort_by_key(|e| e.lsn);
    Ok(all)
}

impl Wal for InMemoryWal {
    fn append(&mut self, payload: Vec<u8>) -> WalEntry {
        let entry = WalEntry {
            lsn: self.next_lsn,
            payload,
        };
        self.next_lsn += 1;
        self.logs.push(entry.clone());
        entry
    }
}

/// 归档段元数据（P5.0.2）：按 first_lsn 排序，与文件名/时钟无关。
/// EN: Archived-segment metadata (P5.0.2), sorted by first_lsn rather than
/// filename or wall clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalSegmentMeta {
    pub path: PathBuf,
    pub first_lsn: u64,
    pub last_lsn: u64,
    pub bytes: u64,
}

/// WAL 段统计（状态端点/保留决策共用）。
/// EN: WAL segment statistics (shared by status and retention decisions).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SegmentStats {
    pub first_available_lsn: u64,
    pub next_lsn: u64,
    pub active_bytes: u64,
    pub archived_bytes: u64,
    pub archives: Vec<WalSegmentMeta>,
}

/// 段保留结果（删除整段；活动文件永不删除）。
/// EN: Segment-retention outcome (whole segments only; the active file is never deleted).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetentionOutcome {
    pub deleted_segments: usize,
    pub bytes_freed: u64,
    pub first_available_lsn: u64,
    /// None = 已按预算执行（可能未删任何段）；Some = 被阻塞的原因。
    /// EN: None = ran within budget (may delete zero); Some = blocked reason.
    pub blocked_reason: Option<String>,
}

/// 文件 WAL：v2 格式（文件头 + 每记录 CRC32），v1 遗留文件只读兼容、按 v1 追加。
/// File WAL: v2 on-disk format (header + per-record CRC32); legacy v1 files stay
/// appendable in v1 and age out via rotate/truncate.
pub struct FileWal {
    path: PathBuf,
    /// `None` 仅在 `truncate_prefix_through` 重写文件瞬间；其余路径均为 `Some`。
    /// EN: `None` only during the instant `truncate_prefix_through` rewrites the file;
    /// every other path is `Some`.
    file: Option<File>,
    next_lsn: u64,
    /// 活动文件格式（决定追加编码；读路径逐段自动识别）。
    /// EN: Active file format (governs append encoding; readers auto-detect per segment).
    format: WalFormat,
    /// 静态加密句柄（P4.7.3）：Some 时新段写 v3，旧明文段在 open 时一次性
    /// 重写为 v3（迁移）；None 时保持明文。
    /// EN: when set, new segments are v3 and legacy plaintext segments are
    /// rewritten once on open.
    encryption: Option<Arc<Encryption>>,
    /// 写路径中毒标记（P4.6.3）：一次 write/flush/sync 失败后，文件可能停在
    /// 半条记录中间——继续追加会把撕裂点之后的所有新记录变成「不可重放区」
    /// （重放以撕裂点为日志终点），而这些新记录可能已被 ACK。中毒后所有写
    /// 路径一律报错，直到进程重启重新 open（open 会截断撕裂尾重建一致前缀）。
    /// 错误只记首个（后续错误是同一故障的回声，覆盖会丢失根因）。
    /// EN: sticky write-poison flag — after a failed write/flush/sync the file may
    /// stop mid-record; further appends would place acknowledged records after a
    /// torn boundary that replay treats as end-of-log. All writes error until the
    /// process reopens the WAL. The first failure is kept (later ones echo it).
    poisoned: Option<String>,
    /// 自动轮转阈值字节（0 = 关闭）；超过时在下一条记录前轮转。
    /// EN: Auto-rotation threshold in bytes (0 = disabled); rotate before the
    /// next record once exceeded.
    segment_bytes: u64,
    /// 活动文件当前字节数与首条记录 LSN（空活动段为 None）。
    /// EN: Active file size and first-record LSN (None for an empty segment).
    active_bytes: u64,
    active_first_lsn: Option<u64>,
    /// 归档段元数据，按 first_lsn 排序。
    /// EN: Archived segments sorted by first_lsn.
    archives: Vec<WalSegmentMeta>,
}

impl FileWal {
    /// 打开 WAL 文件并返回 **启动重放** 所需的条目快照（`next_lsn` 已对齐）。
    /// Open WAL file and return replayed entries for startup (`next_lsn` is aligned).
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<(Self, Vec<WalEntry>)> {
        Self::open_with_encryption(path, None)
    }

    /// 带静态加密打开（P4.7.3）。`encryption=Some` 时：v3 段解密读取；
    /// v1/v2 明文段在重放后**一次性重写为 v3**（含归档段合并），之后追加
    /// 全部加密。加密文件缺 keyring → 启动失败（fail-closed）。
    /// EN: with encryption, v3 segments are decrypted; legacy plaintext history
    /// is rewritten once into a single v3 segment. A missing key is fatal.
    pub fn open_with_encryption(
        path: impl AsRef<Path>,
        encryption: Option<Arc<Encryption>>,
    ) -> std::io::Result<(Self, Vec<WalEntry>)> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        let file_len = file.metadata()?.len();
        let default_format = if encryption.is_some() {
            WalFormat::V3
        } else {
            WalFormat::V2
        };
        // 格式定级：空文件/残缺文件（不够文件头也不够 v1 首条记录，只能是
        // 撕裂残骸）→ 按当前模式（加密/明文）重写文件头；已有内容按文件头识别。
        // EN: empty/torn files are reinitialized in the current mode; existing
        // content is classified by its header.
        let format = if file_len == 0 {
            write_wal_header(&mut file, default_format, encryption.as_deref())?;
            default_format
        } else if file_len < WAL_HEADER_LEN as u64 {
            file.set_len(0)?;
            write_wal_header(&mut file, default_format, encryption.as_deref())?;
            default_format
        } else {
            let mut head = [0u8; WAL_HEADER_LEN];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut head)?;
            file.seek(SeekFrom::End(0))?;
            if &head[..8] == WAL_MAGIC {
                let version = u32::from_be_bytes(head[8..12].try_into().unwrap());
                match version {
                    WAL_FORMAT_VERSION => WalFormat::V2,
                    WAL_FORMAT_VERSION_ENCRYPTED => {
                        if file_len < WAL_ENCRYPTED_HEADER_LEN as u64 {
                            // v3 头被撕裂：截断重写当前模式（数据本来就不可读）。
                            // EN: v3 header torn: truncate and rewrite in the current format
                            // (the data was unreadable anyway).
                            file.set_len(0)?;
                            write_wal_header(&mut file, default_format, encryption.as_deref())?;
                            default_format
                        } else {
                            WalFormat::V3
                        }
                    }
                    other => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unsupported wal format version {other} (expect \
                                 {WAL_FORMAT_VERSION} or {WAL_FORMAT_VERSION_ENCRYPTED})"
                            ),
                        ))
                    }
                }
            } else {
                WalFormat::V1
            }
        };
        // 撕裂尾截断：校验失败点/不完整记录即日志终点，但若撕裂字节留在原地，
        // append 游标会落在日志终点之后——恢复后追加的记录对重放不可达，
        // 二次崩溃即丢失「已确认」写。故 open 一律把活动文件截断到有效前缀
        // 末尾（PostgreSQL end-of-WAL 截断语义）。
        // EN: truncate the active file at end-of-log so post-recovery appends
        // stay replayable.
        let header_len = if format.is_encrypted() {
            WAL_ENCRYPTED_HEADER_LEN
        } else {
            WAL_HEADER_LEN
        };
        let active_entries = if file_len >= header_len as u64 {
            let mut data = Vec::with_capacity(file_len as usize);
            file.seek(SeekFrom::Start(0))?;
            file.read_to_end(&mut data)?;
            let (entries, valid_len) =
                parse_wal_file_entries_with_len(&data, encryption.as_deref())?;
            if (valid_len as u64) < file_len {
                tracing::warn!(
                    path = %path.display(),
                    file_len,
                    valid_len,
                    torn_bytes = file_len - valid_len as u64,
                    "wal: truncating torn tail to end of log"
                );
                file.set_len(valid_len as u64)?;
            }
            file.seek(SeekFrom::End(0))?;
            entries
        } else {
            Vec::new()
        };
        let final_active_len = file.metadata()?.len();
        let active_first_lsn = active_entries.first().map(|e| e.lsn);
        let mut wal = Self {
            path,
            file: Some(file),
            next_lsn: 0,
            format,
            encryption,
            poisoned: None,
            segment_bytes: 0,
            active_bytes: final_active_len,
            active_first_lsn,
            archives: Vec::new(),
        };
        // 迁移标记 + 已是 v3 ⇒ rename 已完成、归档段只是待删的明文副本。
        // 此时绝不能再次合并归档（重复 LSN 会破坏连续性/指纹）；直接清掉。
        // EN: marker + V3 active means the migration rename already happened;
        // the archives are stale plaintext duplicates. Merging them again would
        // duplicate LSNs and break contiguity/fingerprints — delete instead.
        let migration_marker = encryption_migration_marker_path(wal.path());
        let migration_completed = migration_marker.exists() && wal.format == WalFormat::V3;
        let mut replayed = Vec::new();
        if migration_completed {
            for p in list_archived_segments(wal.path())? {
                fs::remove_file(&p)?;
            }
            fs::remove_file(&migration_marker)?;
            sync_parent_dir(wal.path())?;
            tracing::info!(
                path = %wal.path().display(),
                "wal: finished an interrupted encryption migration (removed stale plaintext archives)"
            );
        } else {
            let encryption_for_read = wal.encryption.clone();
            for p in list_archived_segments(wal.path())? {
                let entries = read_records_from_path(&p, encryption_for_read.as_deref())?;
                if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
                    let bytes = fs::metadata(&p)?.len();
                    wal.archives.push(WalSegmentMeta {
                        path: p.clone(),
                        first_lsn: first.lsn,
                        last_lsn: last.lsn,
                        bytes,
                    });
                }
                replayed.extend(entries);
            }
            // 段顺序按 LSN，不按文件名（时钟回拨/跨进程命名可能乱序）。
            // EN: order segments by LSN, not filename (clock skew/cross-process
            // naming can invert lexical order).
            wal.archives.sort_by_key(|m| m.first_lsn);
        }
        let active_had_records = !active_entries.is_empty();
        replayed.extend(active_entries);
        replayed.sort_by_key(|e| e.lsn);
        // 一次性迁移：启用加密后把全部明文历史重写为单一 v3 段并删除旧段。
        // EN: one-time migration of the whole plaintext history into one v3 segment.
        if wal.encryption.is_some() && wal.format != WalFormat::V3 {
            wal.rewrite_all_encrypted(&replayed)?;
            wal.archives.clear();
            wal.active_bytes = fs::metadata(wal.path()).map(|m| m.len()).unwrap_or(0);
            wal.active_first_lsn = replayed.first().map(|e| e.lsn);
        }
        if let Some(last) = replayed.last() {
            wal.next_lsn = last.lsn + 1;
            let _ = std::fs::remove_file(wal_next_lsn_meta_path(wal.path()));
        } else {
            wal.next_lsn = read_wal_next_lsn_meta(wal.path())?.unwrap_or(0);
        }
        // 密钥轮换（P4.7.3）：v3 段头记录的是写入时的活动密钥 id。keyring 的
        // active 变了以后**必须换段**——继续向旧段追加会用新密钥写记录，而
        // 重放只信段头 id，新记录将永久不可解。旧段整体归档（保留旧 id 头）。
        // EN: a v3 segment header pins the key id used for its records. After
        // the active key changes, appending there would write new-key records
        // under an old-key header — permanently undecryptable. Archive the old
        // segment (it keeps its old header) and start a fresh one.
        if wal.format == WalFormat::V3 {
            let enc = wal.encryption.clone();
            if let Some(enc) = enc {
                let segment_key = read_v3_key_id(wal.path())?;
                if segment_key != enc.active_id() {
                    if !active_had_records {
                        rehead_empty_v3_segment(&mut wal, enc.as_ref())?;
                        tracing::info!(
                            old_key = %segment_key,
                            new_key = enc.active_id(),
                            "wal: encryption key rotated; empty segment re-headed"
                        );
                    } else {
                        let archived = wal.rotate_archive()?;
                        tracing::info!(
                            archived = ?archived,
                            old_key = %segment_key,
                            new_key = enc.active_id(),
                            "wal: encryption key rotated; previous segment archived"
                        );
                    }
                }
            }
        }
        Ok((wal, replayed))
    }

    /// 把全部历史重写为 v3 单段（tmp + fsync + rename），随后删除明文归档段。
    /// 删除失败按错误上抛：留下明文段会让「已启用加密」成为假象。
    /// EN: rewrite all history into one encrypted segment; leftover plaintext
    /// archives are an error, never a silent leak.
    fn rewrite_all_encrypted(&mut self, entries: &[WalEntry]) -> io::Result<()> {
        let enc = self.encryption.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "rewrite_all_encrypted requires an encryption handle",
            )
        })?;
        let tmp_path = self.path.with_extension("wal.encrypt.tmp");
        {
            let mut tmp = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            write_wal_header(&mut tmp, WalFormat::V3, Some(enc))?;
            for e in entries {
                tmp.write_all(&encode_record(e.lsn, &e.payload, WalFormat::V3, Some(enc))?)?;
            }
            tmp.sync_all()?;
        }
        // 迁移协议（崩溃安全，顺序不可交换）：
        //   1. 标记落盘（+ 目录 fsync）
        //   2. rename v3 tmp → 活动段（+ 目录 fsync）
        //   3. 删除明文归档（+ 目录 fsync）
        //   4. 删除标记
        // 任一步崩溃：若活动段仍是明文就走正常迁移重试；若已是 v3 且标记存在，
        // 下次 open 会清掉重复的明文归档（见 open 的 migration_completed）。
        // EN: marker → rename → delete archives → remove marker, each step
        // followed by a directory fsync; open() repairs the crash windows.
        let marker_path = encryption_migration_marker_path(&self.path);
        {
            let mut marker = fs::File::create(&marker_path)?;
            marker.write_all(
                format!(
                    "pid={} entries={} target=v3\n",
                    std::process::id(),
                    entries.len()
                )
                .as_bytes(),
            )?;
            marker.sync_all()?;
        }
        sync_parent_dir(&self.path)?;
        let _ = self.file.take();
        fs::rename(&tmp_path, &self.path)?;
        sync_parent_dir(&self.path)?;
        for p in list_archived_segments(&self.path)? {
            fs::remove_file(&p)?;
        }
        sync_parent_dir(&self.path)?;
        fs::remove_file(&marker_path)?;
        sync_parent_dir(&self.path)?;
        let mut reopened = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.path)?;
        reopened.seek(SeekFrom::End(0))?;
        self.file = Some(reopened);
        self.format = WalFormat::V3;
        tracing::info!(
            path = %self.path.display(),
            entries = entries.len(),
            "wal: migrated to encrypted format v3"
        );
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 中毒短路：健康时零成本（一个 Option 判别）。
    /// EN: poison gate — one Option check on the healthy path.
    fn ensure_healthy(&self) -> io::Result<()> {
        match &self.poisoned {
            Some(root) => Err(io::Error::other(format!(
                "wal '{}' is poisoned by an earlier write failure ({root}); reopen required",
                self.path.display()
            ))),
            None => Ok(()),
        }
    }

    /// 记录首个写失败并置毒（后续失败不再覆盖根因）。
    /// EN: record the first write failure and poison; later failures don't overwrite it.
    fn poison(&mut self, op: &'static str, err: &io::Error) {
        if self.poisoned.is_none() {
            self.poisoned = Some(format!("{op}: {err}"));
            tracing::error!(
                path = %self.path.display(),
                op,
                error = %err,
                "wal poisoned by write failure; all further writes rejected until reopen"
            );
        }
    }

    /// 测试注入：直接置毒以验证写路径短路（真实 IO 失败无法在单测里可移植地制造）。
    /// EN: test hook to verify the gate (real IO failures aren't portable in unit tests).
    #[cfg(test)]
    pub(crate) fn poison_for_test(&mut self, msg: &str) {
        self.poisoned = Some(msg.to_string());
    }

    #[cfg(test)]
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.is_some()
    }

    pub fn next_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// 配置自动轮转阈值（0 = 关闭）。仅在构造后、写入前调用。
    /// EN: Configure the auto-rotation threshold (0 disables). Call before writes.
    pub fn set_segment_bytes(&mut self, bytes: u64) {
        self.segment_bytes = bytes;
    }

    /// 当前最早可读 LSN（无记录时等于 next_lsn）。
    /// EN: Earliest readable LSN (equal to next_lsn when the log is empty).
    pub fn first_available_lsn(&self) -> u64 {
        self.archives
            .first()
            .map(|m| m.first_lsn)
            .or(self.active_first_lsn)
            .unwrap_or(self.next_lsn)
    }

    /// 段统计快照（状态端点/保留决策）。
    /// EN: Segment statistics snapshot (status and retention decisions).
    pub fn segment_stats(&self) -> SegmentStats {
        let archived_bytes = self.archives.iter().map(|m| m.bytes).sum();
        SegmentStats {
            first_available_lsn: self.first_available_lsn(),
            next_lsn: self.next_lsn,
            active_bytes: self.active_bytes,
            archived_bytes,
            archives: self.archives.clone(),
        }
    }

    /// 按整段删除最旧归档：仅删除 `last_lsn <= safe_floor` 的段，且最多删
    /// `max_deletes` 个；活动文件永不删除。删除后更新 first_available_lsn。
    /// EN: Delete oldest whole archives with `last_lsn <= safe_floor`, at most
    /// `max_deletes`; never touches the active file. Updates first_available_lsn.
    ///
    /// 最后一个归档被删且活动段为空时，先落盘 `next_lsn` 侧车——否则重启只能
    /// 从 0 重新分配 LSN，复制 resume 与快照 checkpoint 全部失义。
    /// EN: When deleting the last archive while the active segment is empty, write
    /// the durable next_lsn sidecar first; otherwise restart would reallocate from
    /// zero and invalidate replication resume/snapshot checkpoints.
    pub fn retain_segments(
        &mut self,
        safe_floor_lsn: u64,
        max_segments: usize,
        max_deletes: usize,
    ) -> io::Result<RetentionOutcome> {
        let first_available = self.first_available_lsn();
        if max_segments == 0 {
            return Ok(RetentionOutcome {
                deleted_segments: 0,
                bytes_freed: 0,
                first_available_lsn: first_available,
                blocked_reason: Some("retention_disabled".to_string()),
            });
        }
        if self.archives.len() <= max_segments {
            return Ok(RetentionOutcome {
                deleted_segments: 0,
                bytes_freed: 0,
                first_available_lsn: first_available,
                blocked_reason: None,
            });
        }
        let mut deleted = 0usize;
        let mut freed = 0u64;
        let mut blocked = None;
        while self.archives.len() > max_segments && deleted < max_deletes {
            let oldest = &self.archives[0];
            if oldest.last_lsn > safe_floor_lsn {
                blocked = Some(format!(
                    "floor_below_oldest_segment: safe_floor={safe_floor_lsn}, \
                     oldest_last_lsn={}",
                    oldest.last_lsn
                ));
                break;
            }
            // 删除最后一个归档且活动段为空：next_lsn 侧车必须先落盘。
            // EN: Deleting the last archive with an empty active segment requires
            // a durable next_lsn sidecar first.
            if self.archives.len() == 1 && self.active_first_lsn.is_none() {
                crate::wal_sidecar::write_wal_next_lsn_meta_atomic(&self.path, self.next_lsn)?;
            }
            let meta = self.archives[0].clone();
            fs::remove_file(&meta.path)?;
            self.archives.remove(0);
            deleted += 1;
            freed += meta.bytes;
        }
        if deleted > 0 {
            sync_parent_dir(&self.path)?;
        }
        Ok(RetentionOutcome {
            deleted_segments: deleted,
            bytes_freed: freed,
            first_available_lsn: self.first_available_lsn(),
            blocked_reason: blocked,
        })
    }

    /// 删除 LSN ≤ `through_lsn` 的记录并重写 WAL 文件（同目录下原子 `rename`）。
    ///
    /// **安全 / Safety**：仅删除**已确认**可丢弃的日志（通常与 `*.applied_lsn` 对齐）。当前索引仍为内存态、
    /// 无快照时，截断后若进程重启且 WAL 已空，**冷启动重放将无法恢复数据** — 须配合未来 checkpoint 快照或运维备份。
    /// **Safety**: Only drop log records that are safe to discard (typically aligned with `*.applied_lsn`).
    /// Without a durable snapshot, restarting after truncation may **lose data** if the WAL no longer contains history.
    ///
    /// 返回删除的记录条数。
    /// Returns the number of removed records.
    pub fn truncate_prefix_through(&mut self, through_lsn: u64) -> std::io::Result<usize> {
        let mut entries = self.replay_all()?;
        for p in list_archived_segments(&self.path)? {
            let _ = fs::remove_file(p);
        }
        let max_lsn_before = entries.last().map(|e| e.lsn);
        let before = entries.len();
        entries.retain(|e| e.lsn > through_lsn);
        let removed = before - entries.len();
        let next_lsn = if let Some(last) = entries.last() {
            last.lsn + 1
        } else {
            max_lsn_before.map(|m| m.saturating_add(1)).unwrap_or(0)
        };

        let tmp_path = self.path.with_extension("wal.checkpoint.tmp");
        {
            let mut tmp = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            // 重写保持当前格式：加密启用时一律 v3（open 已完成迁移），
            // 否则升级到 v2（v1 文件经截断自然换代）。
            // EN: rewrites keep the active format — v3 when encryption is on,
            // otherwise v2 (the natural v1 upgrade).
            let format = if self.encryption.is_some() {
                WalFormat::V3
            } else {
                WalFormat::V2
            };
            write_wal_header(&mut tmp, format, self.encryption.as_deref())?;
            for e in &entries {
                tmp.write_all(&encode_record(
                    e.lsn,
                    &e.payload,
                    format,
                    self.encryption.as_deref(),
                )?)?;
            }
            tmp.sync_all()?;
        }

        let _ = self.file.take();
        fs::rename(&tmp_path, &self.path)?;
        let mut reopened = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.path)?;
        reopened.seek(SeekFrom::End(0))?;
        self.file = Some(reopened);
        self.next_lsn = next_lsn;
        self.format = if self.encryption.is_some() {
            WalFormat::V3
        } else {
            WalFormat::V2
        };
        // 全量重写后归档段全部消失，活动文件即完整保留前缀。
        // EN: A full rewrite drops every archive; the active file is the retained prefix.
        self.archives.clear();
        self.active_bytes = fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        self.active_first_lsn = entries.first().map(|e| e.lsn);
        // 截断是全量重写（解析有效前缀 → tmp → fsync → rename）：成功后
        // 文件本身就是一致前缀，等价于一次 reopen，毒标记可以安全清除
        // （OCR 批次二十八）。失败路径不置毒——tmp+rename 在 rename 前
        // 失败时活动文件未被触碰，rename 后失败时新文件也已完整。
        // EN: a successful truncate rebuilds a consistent prefix (the reopen
        // equivalent), so the poison clears; failures leave the active file
        // either untouched (pre-rename) or fully rewritten (post-rename).
        self.poisoned = None;
        if entries.is_empty() {
            write_wal_next_lsn_meta_atomic(&self.path, next_lsn)?;
        } else {
            let _ = std::fs::remove_file(wal_next_lsn_meta_path(&self.path));
        }
        Ok(removed)
    }

    /// 将非空活动 WAL 归档为 `*.wal.archived.{nanos}`，并打开新的空活动文件；**LSN 单调延续**。
    /// Archive non-empty active WAL to `*.wal.archived.{nanos}`, open a fresh empty active file; **LSN continues**.
    pub fn rotate_archive(&mut self) -> io::Result<Option<PathBuf>> {
        if let Some(f) = self.file.as_mut() {
            f.flush()?;
        }
        let _ = self.file.take();
        let len = fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        // 「有记录才归档」：文件头不算数据。
        // EN: archive only when records exist — a header alone is not data.
        let header_len = match self.format {
            WalFormat::V1 => 0,
            WalFormat::V2 => WAL_HEADER_LEN as u64,
            WalFormat::V3 => WAL_ENCRYPTED_HEADER_LEN as u64,
        };
        let has_records = len > header_len;
        if !has_records {
            let mut f = OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(&self.path)?;
            f.seek(SeekFrom::End(0))?;
            if len == 0 {
                // 兜底：无头的空文件（理论上 open 已写头）补齐当前格式头。
                // EN: Fallback: a headerless empty file (open normally writes the header)
                // gets the current format header.
                let format = if self.encryption.is_some() {
                    WalFormat::V3
                } else {
                    WalFormat::V2
                };
                write_wal_header(&mut f, format, self.encryption.as_deref())?;
                self.format = format;
            }
            self.file = Some(f);
            self.active_bytes = len;
            self.active_first_lsn = None;
            write_wal_next_lsn_meta_atomic(&self.path, self.next_lsn)?;
            return Ok(None);
        }
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        // 纳秒 + 进程内序号：防止同一时钟周期内两次 rotate 生成同名归档互相覆盖
        // （时钟粒度粗于 1ns 的平台理论上可达）。
        // EN: nanos + process-local seq — prevents same-clock-tick double rotates
        // from overwriting each other's archive (reachable on platforms whose
        // clock granularity is coarser than 1ns).
        static ROTATE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = ROTATE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let archived = self
            .path
            .with_extension(format!("wal.archived.{nanos}.{seq}"));
        fs::rename(&self.path, &archived)?;
        let mut reopened = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&self.path)?;
        reopened.seek(SeekFrom::End(0))?;
        // 新段沿用当前格式（加密启用即 v3）。
        // EN: new segments keep the active format (v3 when encrypted).
        let format = if self.encryption.is_some() {
            WalFormat::V3
        } else {
            WalFormat::V2
        };
        write_wal_header(&mut reopened, format, self.encryption.as_deref())?;
        self.file = Some(reopened);
        self.format = format;
        // 归档段元数据：first 取运行态记录的首 LSN，last 取 next_lsn-1。
        // EN: Archive metadata: first is the active segment's first LSN; last is
        // next_lsn - 1.
        let first_lsn = self
            .active_first_lsn
            .unwrap_or_else(|| self.next_lsn.saturating_sub(1));
        self.archives.push(WalSegmentMeta {
            path: archived.clone(),
            first_lsn,
            last_lsn: self.next_lsn.saturating_sub(1),
            bytes: len,
        });
        self.archives.sort_by_key(|m| m.first_lsn);
        self.active_bytes = match format {
            WalFormat::V1 => 0,
            WalFormat::V2 => WAL_HEADER_LEN as u64,
            WalFormat::V3 => WAL_ENCRYPTED_HEADER_LEN as u64,
        };
        self.active_first_lsn = None;
        write_wal_next_lsn_meta_atomic(&self.path, self.next_lsn)?;
        Ok(Some(archived))
    }

    /// 归档段数量（不含活动文件）。
    /// EN: Number of archive segments (excluding the active file).
    pub fn archived_segment_count(&self) -> io::Result<usize> {
        Ok(self.archives.len())
    }

    /// 活动 WAL 文件字节数。
    /// EN: Size of the active WAL file in bytes.
    pub fn active_file_len(&self) -> io::Result<u64> {
        Ok(self.active_bytes)
    }

    pub fn replay_all(&mut self) -> std::io::Result<Vec<WalEntry>> {
        if let Some(f) = self.file.as_mut() {
            f.flush()?;
        }
        merge_all_segments(&self.path, self.encryption.as_deref())
    }

    /// 追加一条物理记录（**不** `flush`），供异步 WAL 批量刷盘合并 `flush` 调用。
    /// `next_lsn` 只在写盘成功后推进——写失败消耗 LSN 会让后续追加在磁盘上
    /// 留下空洞（文件缺该记录、侧车/内存却已越过），复制 resume 语义随之崩坏。
    /// EN: Append one record without flushing; async WAL batches multiple appends
    /// per `flush`. `next_lsn` advances only after a successful write — consuming
    /// an LSN for a failed write would leave an on-disk hole and corrupt
    /// replication resume semantics.
    pub fn append_no_flush(&mut self, payload: Vec<u8>) -> io::Result<WalEntry> {
        // 注入点（P4.5.4）：写前报错模拟 ENOSPC/EIO；写后 panic/pause 让 kill -9
        // 落在「记录已进内核缓冲、next_lsn 未推进」的窗口（撕裂写与侧车一致性）。
        // EN: pre-write error simulates ENOSPC/EIO; post-write panic/pause targets
        // the window where bytes are buffered but `next_lsn` has not advanced.
        self.ensure_healthy()?;
        // 自动轮转只在记录边界发生：先整体归档旧段，再写新记录。
        // EN: Auto-rotation happens only at record boundaries: archive the old
        // segment first, then write the new record.
        let header_len = match self.format {
            WalFormat::V1 => 0,
            WalFormat::V2 => WAL_HEADER_LEN as u64,
            WalFormat::V3 => WAL_ENCRYPTED_HEADER_LEN as u64,
        };
        if self.segment_bytes > 0
            && self.active_bytes >= self.segment_bytes
            && self.active_bytes > header_len
        {
            self.rotate_archive()?;
        }
        crate::fail_point!("wal_append_pre");
        let lsn = self.next_lsn;
        let record = encode_record(lsn, &payload, self.format, self.encryption.as_deref())?;
        if let Err(e) = self.file.as_mut().expect("wal file").write_all(&record) {
            // 写失败可能已落半条记录：置毒，绝不在撕裂点之后继续追加。
            // EN: a failed write may have left a torn prefix — poison rather than
            // append behind it.
            self.poison("append", &e);
            return Err(e);
        }
        crate::fail_point!("wal_append_post");
        self.active_bytes = self.active_bytes.saturating_add(record.len() as u64);
        if self.active_first_lsn.is_none() {
            self.active_first_lsn = Some(lsn);
        }
        self.next_lsn += 1;
        Ok(WalEntry { lsn, payload })
    }

    /// 将已写入内核缓冲区的 WAL 数据 `flush`（与 `append_no_flush` 配对）。
    /// Flush WAL file after one or more `append_no_flush` calls.
    pub fn flush_wal(&mut self) -> io::Result<()> {
        self.ensure_healthy()?;
        match self.file.as_mut().expect("wal file").flush() {
            Ok(()) => Ok(()),
            Err(e) => {
                self.poison("flush", &e);
                Err(e)
            }
        }
    }

    /// 以指定 LSN 追加一条记录（**不** `flush`）——复制 follower 保持 leader
    /// LSN 落盘，本地文件即复制进度（promote 后 LSN 连续、句柄零切换）。
    /// LSN 必须等于 `next_lsn`（连续无洞），否则拒绝：`lsn > next_lsn` 的空洞
    /// 意味着复制流脱序，静默接受会让 follower 与 leader 永久分叉且 promote
    /// 后污染新写；`lsn < next_lsn` 的过期/重复记录同样拒绝（at-least-once
    /// 重投应由 follower 幂等层先行吸收，漏到这里仍是脱序信号）。
    /// EN: Append one record with an explicit LSN (no flush) — follower keeps
    /// the leader's LSN space so its local file doubles as replication
    /// progress. Both forward gaps (stream desync) and stale/duplicate LSNs
    /// (redelivery that the idempotency layer should have absorbed) are
    /// rejected rather than silently accepted.
    pub fn append_entry_no_flush(&mut self, lsn: u64, payload: Vec<u8>) -> io::Result<WalEntry> {
        if lsn != self.next_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "wal append_entry lsn gap: got {lsn}, expect {}",
                    self.next_lsn
                ),
            ));
        }
        self.append_no_flush(payload)
    }

    /// fdatasync——复制 follower 的 DURABLE ACK 前置持久化屏障。
    /// EN: fdatasync durability barrier preceding the follower's DURABLE ack.
    pub fn sync_data(&mut self) -> io::Result<()> {
        self.ensure_healthy()?;
        // 注入点（P4.5.4）：fdatasync 前失败 ⇒ follower 不得发 DURABLE ACK。
        // EN: failure before fdatasync means the follower must not ack DURABLE.
        crate::fail_point!("wal_fsync_pre");
        match self.file.as_mut().expect("wal file").sync_data() {
            Ok(()) => Ok(()),
            Err(e) => {
                self.poison("sync_data", &e);
                Err(e)
            }
        }
    }
}

impl Wal for FileWal {
    fn append(&mut self, payload: Vec<u8>) -> WalEntry {
        let entry = self
            .append_no_flush(payload)
            .expect("failed to write wal record");
        self.flush_wal().expect("failed to flush wal file");
        entry
    }
}

/// 测试临时文件唯一命名：pid + 纳秒 + 进程内原子序号。
/// 仅靠纳秒时间戳会在并行测试同一时钟周期取值时撞名（Apple Silicon 时钟粒度
/// 粗于 1ns，CI 上实测两个并行测试拿到同一路径互相污染）。
/// EN: Unique temp filename for tests: pid + nanos + process-local atomic seq.
/// Nanos-only names collide when parallel tests read the clock in the same tick
/// (Apple Silicon granularity is coarser than 1ns; observed on CI as two
/// concurrent tests sharing one path and corrupting each other).
#[cfg(test)]
pub(crate) fn unique_temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{tag}-{}-{nanos}-{seq}.{ext}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::{
        encode_record, encryption_migration_marker_path, list_archived_segments,
        read_records_from_path, read_v3_key_id, unique_temp_path, write_wal_header, FileWal, Wal,
        WalFormat, WAL_FORMAT_VERSION_ENCRYPTED, WAL_HEADER_LEN, WAL_MAGIC,
    };
    use crate::encryption::{Encryption, LocalKeyProvider};
    use std::io::Write;
    use std::sync::Arc;

    /// 测试用 keyring（单一 id=k1 的 32B 密钥）。
    /// EN: Test keyring (a single 32B key with id=k1).
    fn test_encryption(byte: u8) -> Arc<Encryption> {
        let text = format!("k1={}", format!("{byte:02x}").repeat(32));
        Arc::new(
            Encryption::from_provider(Arc::new(
                LocalKeyProvider::parse(&text, None).expect("keyring"),
            ))
            .expect("encryption"),
        )
    }

    fn temp_wal_path() -> std::path::PathBuf {
        unique_temp_path("houyidb-wal", "log")
    }

    /// 手工编码一条 v1 遗留记录（构造 v1 测试文件用）。
    /// EN: Hand-encodes a v1 legacy record (used to build v1 test files).
    fn v1_record(lsn: u64, payload: &[u8]) -> Vec<u8> {
        let mut r = Vec::new();
        r.extend_from_slice(&lsn.to_be_bytes());
        r.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        r.extend_from_slice(payload);
        r
    }

    /// 只清理本测试自己 WAL 的归档段——全目录扫删会与并行测试互相误删
    /// （实测 file_wal_rotate_merges_on_reopen 的归档被邻居测试清掉而丢条目）。
    /// EN: Remove only this WAL's archived segments; a directory-wide sweep
    /// deletes sibling tests' archives under parallel runs.
    fn cleanup_own_archived_segments(path: &std::path::Path) {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let marker = format!("{stem}.wal.archived.");
        let parent = path.parent().unwrap();
        for e in std::fs::read_dir(parent).unwrap().flatten() {
            let p = e.path();
            if p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with(&marker))
                .unwrap_or(false)
            {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    /// P4.7.3：加密往返、落盘无明文、重开可读。
    /// EN: P4.7.3: encrypted roundtrip, no plaintext on disk, reopenable.
    #[test]
    fn encrypted_wal_roundtrip_and_no_plaintext_on_disk() {
        let path = temp_wal_path();
        let enc = test_encryption(7);
        {
            let (mut wal, _) = FileWal::open_with_encryption(&path, Some(Arc::clone(&enc)))
                .expect("open encrypted");
            wal.append(b"top-secret-vector".to_vec());
            wal.append(b"second-record".to_vec());
        }
        let raw = std::fs::read(&path).unwrap();
        let needle = b"top-secret-vector";
        assert!(
            !raw.windows(needle.len()).any(|w| w == needle),
            "plaintext must not appear in an encrypted WAL"
        );
        assert_eq!(&raw[..8], WAL_MAGIC);
        assert_eq!(
            u32::from_be_bytes(raw[8..12].try_into().unwrap()),
            WAL_FORMAT_VERSION_ENCRYPTED
        );
        {
            let (wal, entries) = FileWal::open_with_encryption(&path, Some(enc)).unwrap();
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].payload, b"top-secret-vector");
            assert_eq!(entries[1].payload, b"second-record");
            assert_eq!(wal.next_lsn(), 2);
        }
        let _ = std::fs::remove_file(path);
    }

    /// P4.7.3：错误密钥/缺失 keyring 必须失败，绝不静默返回空历史。
    /// EN: P4.7.3: wrong key or missing keyring must fail; never silently return an
    /// empty history.
    #[test]
    fn encrypted_wal_rejects_wrong_or_missing_key() {
        let path = temp_wal_path();
        {
            let (mut wal, _) =
                FileWal::open_with_encryption(&path, Some(test_encryption(1))).unwrap();
            wal.append(b"secret".to_vec());
        }
        // 同 id、不同密钥 → GCM 认证失败。
        // EN: Same id, different key → GCM authentication failure.
        let err = FileWal::open_with_encryption(&path, Some(test_encryption(2)))
            .err()
            .expect("wrong key must fail");
        assert!(
            err.to_string().contains("authentication") || err.to_string().contains("key"),
            "got: {err}"
        );
        // 完全没有 keyring → 拒绝打开加密段。
        // EN: No keyring at all → refuse to open the encrypted segment.
        assert!(FileWal::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    /// P4.7.3：启用加密时，既有明文历史一次性迁移为 v3 且不再有明文残留。
    /// EN: P4.7.3: with encryption enabled, existing plaintext history migrates to v3 in
    /// one pass and no plaintext remains.
    #[test]
    fn enabling_encryption_migrates_plaintext_history() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            wal.append(b"legacy-one".to_vec());
            wal.append(b"legacy-two".to_vec());
        }
        let enc = test_encryption(9);
        {
            let (mut wal, entries) =
                FileWal::open_with_encryption(&path, Some(Arc::clone(&enc))).unwrap();
            assert_eq!(entries.len(), 2, "history must survive migration");
            assert_eq!(entries[0].payload, b"legacy-one");
            wal.append(b"encrypted-three".to_vec());
        }
        let raw = std::fs::read(&path).unwrap();
        assert!(!raw.windows(10).any(|w| w == b"legacy-one"));
        assert_eq!(
            u32::from_be_bytes(raw[8..12].try_into().unwrap()),
            WAL_FORMAT_VERSION_ENCRYPTED
        );
        {
            let (_wal, entries) = FileWal::open_with_encryption(&path, Some(enc)).unwrap();
            let payloads: Vec<&[u8]> = entries.iter().map(|e| e.payload.as_slice()).collect();
            assert_eq!(
                payloads,
                vec![
                    b"legacy-one".as_slice(),
                    b"legacy-two".as_slice(),
                    b"encrypted-three".as_slice()
                ]
            );
        }
        let _ = std::fs::remove_file(path);
    }

    /// P4.7.3 崩溃窗口：迁移在 rename 之后、删归档之前被杀——下次打开必须
    /// 清掉重复的明文归档，而不是把它们再合并一遍（重复 LSN 会破坏指纹）。
    /// EN: crash between migration rename and archive deletion; the next open
    /// must remove the stale plaintext archives rather than replay duplicates.
    #[test]
    fn interrupted_encryption_migration_is_repaired() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).unwrap();
            wal.append(b"one".to_vec());
            wal.append(b"two".to_vec());
        }
        // 模拟已完成 rename 的 v3 活动段：内容与明文段一致。
        // EN: Simulates an already-renamed v3 active segment: same content as the
        // plaintext segment.
        let entries = read_records_from_path(&path, None).unwrap();
        let archive = path.parent().unwrap().join(format!(
            "{}.wal.archived.0001",
            path.file_stem().unwrap().to_str().unwrap()
        ));
        std::fs::copy(&path, &archive).unwrap();
        let enc = test_encryption(9);
        {
            let tmp = path.with_extension("sim.tmp");
            let mut f = std::fs::File::create(&tmp).unwrap();
            write_wal_header(&mut f, WalFormat::V3, Some(enc.as_ref())).unwrap();
            for e in &entries {
                f.write_all(
                    &encode_record(e.lsn, &e.payload, WalFormat::V3, Some(enc.as_ref())).unwrap(),
                )
                .unwrap();
            }
            f.sync_all().unwrap();
            std::fs::rename(&tmp, &path).unwrap();
        }
        let marker = encryption_migration_marker_path(&path);
        std::fs::write(&marker, b"pid=1 entries=2 target=v3\n").unwrap();

        let (_wal, replayed) = FileWal::open_with_encryption(&path, Some(enc)).unwrap();
        assert_eq!(
            replayed.len(),
            2,
            "stale archive must not duplicate history"
        );
        assert_eq!(replayed[0].payload, b"one");
        assert_eq!(replayed[1].payload, b"two");
        assert!(!archive.exists(), "plaintext archive removed");
        assert!(!marker.exists(), "migration marker cleared");
        let _ = std::fs::remove_file(path);
    }

    /// P4.7.3 轮换：active 密钥变化后必须换段，否则新记录会被旧段头 id 掩盖。
    /// EN: after rotating the active key, appends must go to a fresh segment —
    /// otherwise new-key records sit under an old-key header and never decrypt.
    #[test]
    fn encryption_key_rotation_archives_old_segment() {
        let path = temp_wal_path();
        let hex1 = format!("{:02x}", 1).repeat(32);
        let hex2 = format!("{:02x}", 2).repeat(32);
        let k1 = Arc::new(
            Encryption::from_provider(Arc::new(
                LocalKeyProvider::parse(&format!("k1={hex1}"), None).unwrap(),
            ))
            .unwrap(),
        );
        let both = Arc::new(
            Encryption::from_provider(Arc::new(
                LocalKeyProvider::parse(&format!("k1={hex1}\nactive=k2\nk2={hex2}"), None).unwrap(),
            ))
            .unwrap(),
        );
        {
            let (mut wal, _) = FileWal::open_with_encryption(&path, Some(k1)).unwrap();
            wal.append(b"a".to_vec());
            wal.append(b"b".to_vec());
        }
        {
            let (mut wal, entries) =
                FileWal::open_with_encryption(&path, Some(Arc::clone(&both))).unwrap();
            assert_eq!(entries.len(), 2, "old segment still replayable");
            assert_eq!(entries[0].payload, b"a");
            wal.append(b"c".to_vec());
            assert_eq!(
                read_v3_key_id(&path).unwrap(),
                "k2",
                "active segment must carry the new key id"
            );
        }
        let (_wal, entries) =
            FileWal::open_with_encryption(&path, Some(Arc::clone(&both))).unwrap();
        let payloads: Vec<&[u8]> = entries.iter().map(|e| e.payload.as_slice()).collect();
        assert_eq!(payloads, vec![b"a".as_slice(), b"b", b"c"]);

        // 轮换后 keyring 必须保留旧密钥，否则旧归档不可读（fail-closed）。
        // EN: After rotation the keyring must retain old keys, or old archives become
        // unreadable (fail-closed).
        let k2_only = Arc::new(
            Encryption::from_provider(Arc::new(
                LocalKeyProvider::parse(&format!("k2={hex2}"), None).unwrap(),
            ))
            .unwrap(),
        );
        assert!(FileWal::open_with_encryption(&path, Some(k2_only)).is_err());

        for p in list_archived_segments(&path).unwrap() {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_file(path);
    }

    /// P4.7.3：加密段的撕裂尾仍按日志终点截断（不误报成腐坏）。
    /// EN: P4.7.3: a torn tail in an encrypted segment is still truncated as end-of-log
    /// (not misreported as corruption).
    #[test]
    fn encrypted_torn_tail_truncates_to_previous_record() {
        let path = temp_wal_path();
        let enc = test_encryption(3);
        {
            let (mut wal, _) =
                FileWal::open_with_encryption(&path, Some(Arc::clone(&enc))).unwrap();
            wal.append(b"good-record".to_vec());
            wal.append(b"torn-record".to_vec());
        }
        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 4).unwrap();
        drop(f);
        let (_wal, entries) = FileWal::open_with_encryption(&path, Some(enc)).unwrap();
        assert_eq!(entries.len(), 1, "torn encrypted record is end-of-log");
        assert_eq!(entries[0].payload, b"good-record");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn file_wal_append_and_replay_roundtrip() {
        let path = temp_wal_path();
        {
            let (mut wal, initial) = FileWal::open(&path).expect("open file wal");
            assert!(initial.is_empty());
            let e1 = wal.append(b"alpha".to_vec());
            let e2 = wal.append(b"beta".to_vec());
            assert_eq!(e1.lsn, 0);
            assert_eq!(e2.lsn, 1);
        }
        {
            let (_wal, entries) = FileWal::open(&path).expect("reopen file wal");
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].payload, b"alpha".to_vec());
            assert_eq!(entries[1].payload, b"beta".to_vec());
        }
        let _ = std::fs::remove_file(path);
    }

    /// P4.6.3：写失败置毒后，所有写路径一律拒绝——绝不在可能的撕裂点之后
    /// 追加「已确认但不可重放」的记录；重新 open 后恢复。
    /// EN: after a write failure poisons the WAL, every write path rejects until
    /// reopen — no acknowledged records may land behind a torn prefix.
    #[test]
    fn poisoned_wal_rejects_writes_until_reopen() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            wal.append(b"before".to_vec());
            wal.poison_for_test("append: injected EIO");
            let err = wal.append_no_flush(b"after".to_vec()).unwrap_err();
            assert!(
                err.to_string().contains("poisoned"),
                "poison error must be explicit, got: {err}"
            );
            assert!(wal.flush_wal().is_err(), "flush must reject");
            assert!(wal.sync_data().is_err(), "sync must reject");
            // 失败不得推进 LSN。
            // EN: A failure must not advance LSN.
            assert_eq!(wal.next_lsn(), 1);
        }
        {
            // 重启（reopen）清除毒标记：open 的撕裂尾截断重建一致前缀。
            // EN: Reopen clears the poison marker: open's torn-tail truncation rebuilds a
            // consistent prefix.
            let (mut wal, entries) = FileWal::open(&path).expect("reopen");
            assert_eq!(entries.len(), 1);
            assert_eq!(wal.append(b"recovered".to_vec()).lsn, 1);
        }
        let _ = std::fs::remove_file(path);
    }

    /// OCR 批次二十八：截断 = 全量重写，成功后清除毒标记（唯一进程内自愈路径）。
    /// EN: a successful truncate rebuilds a consistent file and clears poison.
    #[test]
    fn successful_truncate_clears_poison() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            wal.append(b"first".to_vec());
            wal.poison_for_test("append: injected EIO");
            assert!(wal.append_no_flush(b"blocked".to_vec()).is_err());
            assert!(wal.is_poisoned());
            // 重写有效前缀（此处保留全部记录）后毒标记清除。
            // EN: After rewriting the valid prefix (all records kept here) the poison
            // marker is cleared.
            wal.truncate_prefix_through(0).expect("truncate rewrites");
            assert!(!wal.is_poisoned(), "successful truncate must heal");
            assert_eq!(wal.append(b"after".to_vec()).lsn, 1);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn file_wal_truncate_prefix_through_keeps_suffix_and_next_lsn() {
        let path = temp_wal_path();
        let _ = std::fs::remove_file(&path);
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            let _ = wal.append(b"a".to_vec());
            let _ = wal.append(b"b".to_vec());
            let _ = wal.append(b"c".to_vec());
            assert_eq!(wal.next_lsn(), 3);
        }
        {
            let (mut wal, _) = FileWal::open(&path).expect("reopen");
            let removed = wal.truncate_prefix_through(1).expect("truncate");
            assert_eq!(removed, 2);
            assert_eq!(wal.next_lsn(), 3);
        }
        {
            let (_wal, entries) = FileWal::open(&path).expect("reopen2");
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].lsn, 2);
            assert_eq!(entries[0].payload, b"c");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn file_wal_rotate_merges_on_reopen() {
        let path = temp_wal_path();
        let _ = std::fs::remove_file(&path);
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            let _ = wal.append(b"a".to_vec());
            let _ = wal.append(b"b".to_vec());
            let archived = wal.rotate_archive().expect("rotate").expect("path");
            assert!(archived.to_string_lossy().contains("archived"));
            let _ = wal.append(b"c".to_vec());
            assert_eq!(wal.next_lsn(), 3);
        }
        {
            let (_wal, entries) = FileWal::open(&path).expect("reopen");
            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0].payload, b"a");
            assert_eq!(entries[1].payload, b"b");
            assert_eq!(entries[2].payload, b"c");
        }
        let _ = std::fs::remove_file(&path);
        cleanup_own_archived_segments(&path);
    }

    // ── P4.1：v2 格式（文件头 + CRC32）专项 ────────────────────────────────
    // EN: P4.1: v2 format (file header + CRC32) specifics.

    /// 新文件落 v2 文件头，追加的记录带 CRC，重放还原。
    /// EN: New files write the v2 header; appended records carry CRCs and replay restores them.
    #[test]
    fn wal_v2_new_file_has_header_and_roundtrips() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            let _ = wal.append(b"x".to_vec());
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], WAL_MAGIC, "file must start with v2 magic");
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            2,
            "header version must be 2"
        );
        let (_wal, entries) = FileWal::open(&path).expect("reopen");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload, b"x");
        let _ = std::fs::remove_file(&path);
    }

    /// v1 遗留文件：无头也可重放；追加保持 v1（单文件不混合格式）。
    /// EN: v1 legacy file: replayable without a header; appends stay v1 (one file never
    /// mixes formats).
    #[test]
    fn wal_v1_legacy_file_replays_and_stays_v1() {
        let path = temp_wal_path();
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&v1_record(0, b"legacy-a")).unwrap();
            f.write_all(&v1_record(1, b"legacy-b")).unwrap();
        }
        {
            let (mut wal, entries) = FileWal::open(&path).expect("open legacy");
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].payload, b"legacy-a");
            assert_eq!(wal.next_lsn(), 2);
            let _ = wal.append(b"new-c".to_vec());
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_ne!(&bytes[..8], WAL_MAGIC, "v1 file must not gain a v2 header");
        let (_wal, entries) = FileWal::open(&path).expect("reopen");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[2].payload, b"new-c");
        let _ = std::fs::remove_file(&path);
    }

    /// 撕裂写（kill -9 尾部半条记录）：重放返回有效前缀，不 panic。
    /// EN: Torn write (kill -9 mid-record at the tail): replay returns the valid prefix
    /// and does not panic.
    #[test]
    fn wal_v2_torn_tail_replays_valid_prefix() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            let _ = wal.append(b"good-1".to_vec());
            let _ = wal.append(b"good-2".to_vec());
        }
        // 追加半条记录（只有 lsn + 半个 len 字段）模拟撕裂尾。
        // EN: Append half a record (lsn + half of the len field) to simulate a torn tail.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(&7u64.to_be_bytes()[..6]).unwrap();
        drop(f);
        let (wal, entries) = FileWal::open(&path).expect("reopen torn");
        assert_eq!(entries.len(), 2, "torn tail must be dropped");
        assert_eq!(entries[1].payload, b"good-2");
        assert_eq!(wal.next_lsn(), 2);
        let _ = std::fs::remove_file(&path);
    }

    /// 回归（OCR 裁决）：open 必须把活动文件截断到日志终点——否则 append
    /// 游标落在撕裂字节之后，恢复后追加的记录对重放不可达，二次崩溃即丢。
    /// Regression: open truncates the torn tail so post-recovery appends stay
    /// replayable across a second crash.
    #[test]
    fn wal_v2_open_truncates_torn_tail_and_appends_stay_replayable() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            let _ = wal.append(b"good-1".to_vec());
        }
        // 撕裂尾：完整记录头（lsn+len+crc）+ 半个 payload。
        // EN: Torn tail: complete record header (lsn+len+crc) + half the payload.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(&9u64.to_be_bytes()).unwrap();
        f.write_all(&10u32.to_be_bytes()).unwrap();
        f.write_all(&0u32.to_be_bytes()).unwrap();
        f.write_all(b"half").unwrap();
        drop(f);
        let len_with_torn = std::fs::metadata(&path).unwrap().len();

        let (mut wal, entries) = FileWal::open(&path).expect("reopen torn");
        assert_eq!(entries.len(), 1);
        let len_after_open = std::fs::metadata(&path).unwrap().len();
        assert!(
            len_after_open < len_with_torn,
            "open must truncate torn tail ({len_with_torn} -> {len_after_open})"
        );

        // 恢复后继续写入，二次「崩溃」（drop 不 flush 以外的清理）再重开：
        // 新追加的记录必须仍在重放可达区间。
        // EN: Keep writing after recovery, then "crash" again (drop performs no cleanup
        // beyond flush) and reopen: newly appended records must remain replay-reachable.
        let _ = wal.append(b"post-recovery".to_vec());
        drop(wal);
        let (_wal2, entries2) = FileWal::open(&path).expect("second reopen");
        let payloads: Vec<&[u8]> = entries2.iter().map(|e| e.payload.as_slice()).collect();
        assert_eq!(
            payloads,
            vec![b"good-1".as_slice(), b"post-recovery".as_slice()]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// CRC 不匹配 = 日志终点：腐坏记录及其后内容一律不重放。
    /// EN: A CRC mismatch = end of log: the corrupt record and everything after it are
    /// never replayed.
    #[test]
    fn wal_v2_checksum_mismatch_stops_replay() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            let _ = wal.append(b"intact".to_vec());
            let _ = wal.append(b"victim".to_vec());
        }
        // 翻转第二条记录 payload 的一个字节（offset = 头12 + 第一条16+6 + CRC4 之后）。
        // EN: Flip one byte of the second record's payload (offset = header 12 + first
        // record 16+6 + CRC 4).
        let mut bytes = std::fs::read(&path).unwrap();
        let second_payload_offset = WAL_HEADER_LEN + 16 + 6 + 16;
        bytes[second_payload_offset] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let (_wal, entries) = FileWal::open(&path).expect("reopen corrupted");
        assert_eq!(entries.len(), 1, "corrupted record must end the log");
        assert_eq!(entries[0].payload, b"intact");
        let _ = std::fs::remove_file(&path);
    }

    /// 截断重写 = v1→v2 自然换代点。
    /// EN: Truncate-and-rewrite is the natural v1→v2 generation switch point.
    #[test]
    fn wal_truncate_upgrades_v1_file_to_v2() {
        let path = temp_wal_path();
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&v1_record(0, b"old")).unwrap();
            f.write_all(&v1_record(1, b"kept")).unwrap();
        }
        {
            let (mut wal, _) = FileWal::open(&path).expect("open legacy");
            let removed = wal.truncate_prefix_through(0).expect("truncate");
            assert_eq!(removed, 1);
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], WAL_MAGIC, "truncate must rewrite as v2");
        let (_wal, entries) = FileWal::open(&path).expect("reopen");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload, b"kept");
        let _ = std::fs::remove_file(&path);
    }

    /// 仅含文件头的空 WAL 不归档（头部不是数据）；rotate 后新活动文件带头。
    /// EN: An empty WAL holding only a header is not archived (a header is not data);
    /// after rotate the new active file carries a header.
    #[test]
    fn wal_rotate_skips_header_only_file_and_new_active_has_header() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            assert!(
                wal.rotate_archive().expect("rotate").is_none(),
                "header-only file must not be archived"
            );
            let _ = wal.append(b"data".to_vec());
            assert!(wal.rotate_archive().expect("rotate").is_some());
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            &bytes[..8],
            WAL_MAGIC,
            "new active file must carry v2 header"
        );
        let _ = std::fs::remove_file(&path);
        cleanup_own_archived_segments(&path);
    }

    /// P5.0.2：自动轮转 + 按段保留 + 重开后 first_available/next 连续。
    /// EN: P5.0.2 — auto-rotation, whole-segment retention, and contiguous
    /// first_available/next after reopen.
    #[test]
    fn wal_segment_metadata_and_retention() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            wal.set_segment_bytes((WAL_HEADER_LEN + 1) as u64);
            for i in 0..3u8 {
                let _ = wal.append(vec![i]);
            }
            let stats = wal.segment_stats();
            assert_eq!(stats.first_available_lsn, 0);
            assert_eq!(stats.next_lsn, 3);
            assert_eq!(
                stats.archives.len(),
                2,
                "two rotations must archive two segments"
            );
            assert!(stats.active_bytes > 0);
            assert!(stats.archived_bytes > 0);

            let outcome = wal
                .retain_segments(0, 1, 8)
                .expect("retention must succeed");
            assert_eq!(outcome.deleted_segments, 1);
            assert!(outcome.bytes_freed > 0);
            assert_eq!(outcome.first_available_lsn, 1);
            assert_eq!(wal.first_available_lsn(), 1);

            // 旧段已删，floor 不够时不再删除。
            // EN: Older segments were removed; a low floor no longer deletes anything.
            let blocked = wal
                .retain_segments(0, 0, 8)
                .expect("disabled retention must not error");
            assert_eq!(blocked.deleted_segments, 0);
            assert_eq!(
                blocked.blocked_reason.as_deref(),
                Some("retention_disabled")
            );
        }
        let (_wal, entries) = FileWal::open(&path).expect("reopen");
        let lsns: Vec<u64> = entries.iter().map(|e| e.lsn).collect();
        assert_eq!(lsns, vec![1, 2], "retained history must remain contiguous");
        let _ = std::fs::remove_file(&path);
        cleanup_own_archived_segments(&path);
    }

    /// P5.0.2：safe_floor 低于最旧段末 LSN 时保留动作被阻塞而非越界删除。
    /// EN: P5.0.2 — a safe floor below the oldest segment's last LSN blocks
    /// retention instead of deleting needed history.
    #[test]
    fn wal_retention_respects_safe_floor() {
        let path = temp_wal_path();
        {
            let (mut wal, _) = FileWal::open(&path).expect("open");
            for i in 0..2u8 {
                let _ = wal.append(vec![i]);
            }
            let _ = wal.rotate_archive().expect("rotate");
            for i in 2..4u8 {
                let _ = wal.append(vec![i]);
            }
            let _ = wal.rotate_archive().expect("rotate");
            let _ = wal.append(b"tail".to_vec());
            let outcome = wal
                .retain_segments(0, 1, 8)
                .expect("retention must not error");
            assert_eq!(outcome.deleted_segments, 0);
            assert!(
                outcome
                    .blocked_reason
                    .as_deref()
                    .unwrap_or_default()
                    .contains("floor_below_oldest_segment"),
                "got: {:?}",
                outcome.blocked_reason
            );
        }
        let _ = std::fs::remove_file(&path);
        cleanup_own_archived_segments(&path);
    }
}

#[cfg(test)]
mod parent_dir_tests {
    use super::*;

    #[test]
    fn bare_filename_parent_normalizes_to_dot() {
        assert_eq!(parent_dir_or_dot(Path::new("data.wal")), Path::new("."));
        assert_eq!(
            parent_dir_or_dot(Path::new("/var/lib/houyi/data.wal")),
            Path::new("/var/lib/houyi")
        );
    }
}
