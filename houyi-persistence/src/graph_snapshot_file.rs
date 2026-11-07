//! HNSW 图快照文件信封（P4.4.0）：magic + checkpoint_lsn + CRC32 + 原子写。
//! EN: Graph snapshot file envelope (P4.4.0): magic + checkpoint_lsn + CRC32 +
//! atomic write.
//!
//! 分层：payload（HNSW 图 + 墓碑的二进制编解码）由 houyi-vector 的
//! `graph_snapshot` 模块定义，对本信封完全透明；本层只负责持久化完整性
//! （魔数/版本/长度/CRC32）与崩溃安全（tmp + fsync + rename）。
//! EN: Layering — the payload codec lives in houyi-vector and is opaque here;
//! this layer owns persistence integrity (magic/version/len/CRC32) and crash
//! safety (tmp + fsync + rename).
//!
//! 布局（全大端，与 WAL v2/v3 一致）：
//! Layout (all big-endian, matching WAL v2/v3):
//! ```text
//! v1（明文）:
//! magic[10] "HOUYIGRAPH" | version u32 | checkpoint_lsn u64 | payload_len u64 | crc32 u32 | payload
//! v2（加密，P4.7.3）:
//! magic[10] | version=2 u32 | checkpoint_lsn u64 | blob_len u64 | crc32 u32 | key_id[16] | blob
//! blob = nonce[12] ‖ ciphertext ‖ tag[16]
//! ```
//!
//! v2 的 `payload_len` 是密文 blob 长度，CRC 覆盖 **key_id‖blob**：key id
//! 槽位没有 AEAD 保护，必须由 CRC 兜底，否则槽内位翻转会被误判为「缺密钥」
//! 并把启动钉死。读取侧先做结构/CRC 校验再查密钥、最后 AEAD 解密；错误用
//! [`GraphSnapshotError`] 分类——损坏回退全量重放，缺密钥必须启动失败。
//! EN: v2 CRC covers key_id‖blob (the key-id slot has no AEAD protection).
//! Structure/CRC are checked before key lookup and AEAD; a typed error keeps
//! corruption (fallback) and missing key (fatal) apart.
//!
//! `checkpoint_lsn` 语义随调用方：benchmark 主索引对齐主 WAL LSN；collection
//! 路径对齐 collections.wal LSN 空间。重放规则：LSN ≤ checkpoint 的记录跳过
//! 向量插入（图已含），仍重建标量行。
//! EN: `checkpoint_lsn` semantics are caller-defined: the benchmark index uses
//! the main WAL LSN space; collections use the collections.wal LSN space.
//! Replay rule: records with LSN ≤ checkpoint skip vector insertion (the graph
//! already has them) but still rebuild scalar rows.

use crate::encryption::{decode_key_id, encode_key_id, Encryption, KEY_ID_LEN};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// 文件魔数（10 字节，区别于 payload 内的 8 字节 "HNSWGRPH"）。
/// EN: File magic (10B; the in-payload magic is a different 8B "HNSWGRPH").
pub const GRAPH_SNAPSHOT_MAGIC: &[u8; 10] = b"HOUYIGRAPH";
/// 明文信封版本（v1）。 / EN: Plaintext envelope version.
pub const GRAPH_SNAPSHOT_FILE_VERSION: u32 = 1;
/// 加密信封版本（v2，P4.7.3）。 / EN: Encrypted envelope version (v2).
pub const GRAPH_SNAPSHOT_FILE_VERSION_ENCRYPTED: u32 = 2;
/// 定长明文信封头字节数（从 magic 派生，单一事实源）。 / EN: Fixed plaintext
/// envelope header length (derived from the magic — single source of truth).
pub const GRAPH_SNAPSHOT_HEADER_LEN: usize = GRAPH_SNAPSHOT_MAGIC.len() + 4 + 8 + 8 + 4;
/// 加密信封头字节数（明文头 + key id 槽位）。 / EN: Encrypted header length.
pub const GRAPH_SNAPSHOT_ENCRYPTED_HEADER_LEN: usize = GRAPH_SNAPSHOT_HEADER_LEN + KEY_ID_LEN;
/// 缺密钥错误的消息前缀（用于人读诊断；**分类走类型**，不靠字符串）。
/// EN: human-readable prefix for missing-key errors; classification is by
/// type, never by string matching across crate boundaries.
pub const GRAPH_SNAPSHOT_MISSING_KEY: &str = "graph snapshot: missing encryption key";

/// 快照读取错误：调用方必须区分「致命缺密钥」与「可回退的损坏」。
/// 为什么用枚举而不是 `String`（OCR 批次三十五 MED）：字符串前缀分类会被
/// 任何一层 `format!("...: {e}")` 包装破坏——缺密钥一旦被误判为损坏，
/// 冷启动会静默全量重放并丢掉整段历史。类型让这种误判在编译期不可表达。
/// EN: typed error — a missing key must abort startup while corruption may fall
/// back to replay; string-prefix classification breaks under any wrapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphSnapshotError {
    /// 损坏/截断/CRC/解密失败：warn + 回退重放。 / fall back to full replay.
    /// EN: Corruption/truncation/CRC/decryption failure: warn and fall back to full replay.
    Corrupt(String),
    /// 有加密快照但密钥不可得：必须让启动失败。 / fatal at startup.
    /// EN: Encrypted snapshot present but key unavailable: startup must fail.
    MissingKey(String),
}

impl GraphSnapshotError {
    pub fn is_missing_key(&self) -> bool {
        matches!(self, Self::MissingKey(_))
    }

    fn corrupt(msg: impl Into<String>) -> Self {
        Self::Corrupt(msg.into())
    }
}

impl std::fmt::Display for GraphSnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt(msg) | Self::MissingKey(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for GraphSnapshotError {}

/// 解码后的快照文件内容。 / EN: Decoded snapshot file.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphSnapshotFile {
    /// 快照覆盖到的 LSN（调用方定义 LSN 空间）。
    /// EN: LSN up to which the snapshot covers (caller-defined LSN space).
    pub checkpoint_lsn: u64,
    /// 不透明 payload（houyi-vector 图快照字节）。
    /// EN: Opaque payload (houyi-vector graph snapshot bytes).
    pub payload: Vec<u8>,
}

/// `foo.wal` → `foo.graph_snapshot.bin`（benchmark 主索引沿用 wal 衍生路径；
/// collection 路径由调用方在 `snapshots/` 子目录下自行拼 `{name}.graph_snapshot.bin`）。
/// EN: wal-derived path helper for the benchmark index; collection callers
/// build their own `{name}.graph_snapshot.bin` under a `snapshots/` dir.
pub fn graph_snapshot_path_for_wal(wal_path: &Path) -> PathBuf {
    wal_path.with_extension("graph_snapshot.bin")
}

/// 编码明文信封（纯函数，便于测试）。 / EN: Encode the plaintext envelope.
pub fn encode_graph_snapshot_file(checkpoint_lsn: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(GRAPH_SNAPSHOT_HEADER_LEN + payload.len());
    out.extend_from_slice(GRAPH_SNAPSHOT_MAGIC);
    out.extend_from_slice(&GRAPH_SNAPSHOT_FILE_VERSION.to_be_bytes());
    out.extend_from_slice(&checkpoint_lsn.to_be_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    out.extend_from_slice(&crc32fast::hash(payload).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// 编码信封：`encryption = Some` 产出 v2（密文），None 产出 v1（明文）。
/// EN: Encode the envelope — v2 (encrypted) with a handle, v1 (plaintext) without.
pub fn encode_graph_snapshot_file_with_encryption(
    checkpoint_lsn: u64,
    payload: &[u8],
    encryption: Option<&Encryption>,
) -> io::Result<Vec<u8>> {
    let Some(enc) = encryption else {
        return Ok(encode_graph_snapshot_file(checkpoint_lsn, payload));
    };
    let blob = enc.encrypt(payload)?;
    let key_id = encode_key_id(enc.active_id())?;
    // CRC 覆盖 key_id‖blob：key id 槽位本身没有 AEAD 保护，必须由 CRC 兜底，
    // 否则槽内位翻转会被误判为「缺密钥」（OCR 批次三十五）。
    // EN: CRC covers key_id‖blob — the key-id slot has no AEAD protection, and
    // a bit flip there must read as corruption, not as a missing key.
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&key_id);
    hasher.update(&blob);
    let mut out = Vec::with_capacity(GRAPH_SNAPSHOT_ENCRYPTED_HEADER_LEN + blob.len());
    out.extend_from_slice(GRAPH_SNAPSHOT_MAGIC);
    out.extend_from_slice(&GRAPH_SNAPSHOT_FILE_VERSION_ENCRYPTED.to_be_bytes());
    out.extend_from_slice(&checkpoint_lsn.to_be_bytes());
    out.extend_from_slice(&(blob.len() as u64).to_be_bytes());
    out.extend_from_slice(&hasher.finalize().to_be_bytes());
    out.extend_from_slice(&key_id);
    out.extend_from_slice(&blob);
    Ok(out)
}

/// 解码并校验信封（明文 v1 或带对应密钥的 v2）。
/// 调用方按 [`GraphSnapshotError`] 分类：损坏回退重放，缺密钥致命。
/// EN: Decode either v1 or v2; callers branch on [`GraphSnapshotError`].
pub fn decode_graph_snapshot_file(bytes: &[u8]) -> Result<GraphSnapshotFile, GraphSnapshotError> {
    decode_graph_snapshot_file_with_encryption(bytes, None)
}

/// 带解密的解码（P4.7.3）。 / EN: Decode with decryption (P4.7.3).
pub fn decode_graph_snapshot_file_with_encryption(
    bytes: &[u8],
    encryption: Option<&Encryption>,
) -> Result<GraphSnapshotFile, GraphSnapshotError> {
    let (version, header_len) = envelope_version(bytes)?;
    if version == GRAPH_SNAPSHOT_FILE_VERSION {
        let (checkpoint_lsn, payload) = validate_body(bytes, header_len, header_len)?;
        return Ok(GraphSnapshotFile {
            checkpoint_lsn,
            payload: payload.to_vec(),
        });
    }
    let key_id = encrypted_key_id(bytes)?;
    // 先校验结构/长度/CRC（覆盖 key_id‖blob）再查密钥：key id 槽位的位翻转
    // 必须报损坏，不能冒充缺密钥把启动钉死（OCR 批次三十五）。
    // EN: structure/CRC (over key_id‖blob) first, key lookup second — a rotten
    // key-id slot must read as corruption, not as a fatal missing key.
    let (checkpoint_lsn, blob) = validate_body(bytes, header_len, GRAPH_SNAPSHOT_HEADER_LEN)?;
    let enc = require_key(encryption, &key_id)?;
    let payload = enc
        .decrypt(&key_id, blob)
        .map_err(|e| GraphSnapshotError::corrupt(format!("graph snapshot: decrypt failed: {e}")))?;
    Ok(GraphSnapshotFile {
        checkpoint_lsn,
        payload,
    })
}

/// 读取版本并返回对应的定长头长度。 / EN: Read the version and its header length.
fn envelope_version(bytes: &[u8]) -> Result<(u32, usize), GraphSnapshotError> {
    let min = GRAPH_SNAPSHOT_MAGIC.len() + 4;
    if bytes.len() < min {
        return Err(GraphSnapshotError::corrupt(format!(
            "graph snapshot: header truncated ({} < {min})",
            bytes.len()
        )));
    }
    if &bytes[..GRAPH_SNAPSHOT_MAGIC.len()] != GRAPH_SNAPSHOT_MAGIC {
        return Err(GraphSnapshotError::corrupt("graph snapshot: bad magic"));
    }
    let version = u32::from_be_bytes(bytes[10..14].try_into().unwrap());
    let header_len = match version {
        GRAPH_SNAPSHOT_FILE_VERSION => GRAPH_SNAPSHOT_HEADER_LEN,
        GRAPH_SNAPSHOT_FILE_VERSION_ENCRYPTED => GRAPH_SNAPSHOT_ENCRYPTED_HEADER_LEN,
        _ => {
            return Err(GraphSnapshotError::corrupt(format!(
                "graph snapshot: unsupported version {version}"
            )))
        }
    };
    if bytes.len() < header_len {
        return Err(GraphSnapshotError::corrupt(format!(
            "graph snapshot: header truncated ({} < {header_len})",
            bytes.len()
        )));
    }
    Ok((version, header_len))
}

/// v2 头里的 key id。 / EN: Key id from the v2 header.
fn encrypted_key_id(bytes: &[u8]) -> Result<String, GraphSnapshotError> {
    decode_key_id(&bytes[GRAPH_SNAPSHOT_HEADER_LEN..GRAPH_SNAPSHOT_ENCRYPTED_HEADER_LEN])
        .map_err(|e| GraphSnapshotError::corrupt(format!("graph snapshot: bad key id slot: {e}")))
}

fn missing_key(key_id: &str) -> GraphSnapshotError {
    GraphSnapshotError::MissingKey(format!(
        "{GRAPH_SNAPSHOT_MISSING_KEY} {key_id:?} (configure the keyring that holds it)"
    ))
}

fn require_key<'a>(
    encryption: Option<&'a Encryption>,
    key_id: &str,
) -> Result<&'a Encryption, GraphSnapshotError> {
    let enc = encryption.ok_or_else(|| {
        GraphSnapshotError::MissingKey(format!(
            "{GRAPH_SNAPSHOT_MISSING_KEY} {key_id:?} (snapshot is encrypted but no keyring is configured)"
        ))
    })?;
    if !enc.has_key(key_id) {
        return Err(missing_key(key_id));
    }
    Ok(enc)
}

/// 校验结构/长度/CRC 并返回 `(checkpoint_lsn, payload 切片)`。
/// `crc_from` 是 CRC 覆盖区起点：v1 为头尾（34），v2 从 key id 槽位（34）起，
/// 从而把 key id 纳入完整性保护。`payload` 始终从 `header_len` 起。
/// EN: validate structure/length/CRC; `crc_from` is 34 for both versions so v2
/// also protects the key-id slot, while the payload starts at `header_len`.
fn validate_body(
    bytes: &[u8],
    header_len: usize,
    crc_from: usize,
) -> Result<(u64, &[u8]), GraphSnapshotError> {
    let checkpoint_lsn = u64::from_be_bytes(bytes[14..22].try_into().unwrap());
    let payload_len_raw = u64::from_be_bytes(bytes[22..30].try_into().unwrap());
    let crc_expected = u32::from_be_bytes(bytes[30..34].try_into().unwrap());
    // 长度算术全程 checked：payload_len 来自文件（CRC 护不住 header 其余
    // 字段），u64::MAX 级值在 debug 下会让 HEADER+len 加法 panic、release 下
    // 回绕成误报——按信封纪律统一归为干净 Err（OCR 8eac6b8）。
    // EN: Length arithmetic is fully checked: payload_len comes from the file
    // (CRC does not protect the other header fields); u64::MAX-scale values
    // would panic on HEADER+len in debug and wrap into bogus values in release,
    // so they map to a clean Err per the envelope discipline (OCR 8eac6b8).
    let payload_len = usize::try_from(payload_len_raw).map_err(|_| {
        GraphSnapshotError::corrupt("graph snapshot: payload_len exceeds addressable size")
    })?;
    let total = header_len.checked_add(payload_len).ok_or_else(|| {
        GraphSnapshotError::corrupt("graph snapshot: payload_len overflows total length")
    })?;
    if bytes.len() < total {
        return Err(GraphSnapshotError::corrupt(format!(
            "graph snapshot: payload truncated ({} < {total})",
            bytes.len()
        )));
    }
    if bytes.len() > total {
        return Err(GraphSnapshotError::corrupt(format!(
            "graph snapshot: {} trailing bytes",
            bytes.len() - total
        )));
    }
    let payload = &bytes[header_len..];
    let crc_actual = crc32fast::hash(&bytes[crc_from..]);
    if crc_actual != crc_expected {
        return Err(GraphSnapshotError::corrupt(format!(
            "graph snapshot: crc32 mismatch (expected {crc_expected:#010x}, got {crc_actual:#010x})"
        )));
    }
    Ok((checkpoint_lsn, payload))
}

/// 原子写快照：同目录 tmp → fsync(file) → rename → fsync(dir)。
/// EN: Atomic write: same-dir tmp → fsync(file) → rename → fsync(dir).
///
/// 比 `vector_snapshot` 的 tmp+rename 多两次 fsync：快照是恢复根基，写频率
/// 极低，两个 syscall 换崩溃后文件必定完整可见（ rename 本身不保证数据落盘）。
///
/// 写盘流式进行（header 字段 + payload 分次 write_all），不为拼整 buffer
/// 再复制一遍 payload——图 payload 可达数百 MB（OCR 8eac6b8）。
/// tmp 文件名带 pid + 纳秒nonce：两个写者（线程/进程，如周期 checkpoint 与
/// 手动快照撞车）各写各的 tmp，后 rename 者赢，reader 侧 CRC 兜底；确定名
/// 的 tmp 会让两方字节交织进同一文件（OCR 8eac6b8）。
///
/// 崩溃残留纪律：进程在 rename 前崩溃会留下 `{name}.{pid}.{nonce}.tmp`，
/// 本模块不做恢复时清扫——调用方（server 启动路径）如需清理按该 glob 模式
/// 自行处理；残留文件无害（reader 只认最终名）。
pub fn write_graph_snapshot_atomic(
    path: &Path,
    checkpoint_lsn: u64,
    payload: &[u8],
) -> io::Result<()> {
    write_graph_snapshot_atomic_with_encryption(path, checkpoint_lsn, payload, None)
}

/// 带加密的原子写（P4.7.3）：先加密（唯一一次新分配），再按 v2 头流式写盘，
/// 不为拼整文件再复制一遍密文。
/// EN: Encrypt first (the only new allocation), then stream the v2 header and
/// blob to disk — no second full-size copy.
pub fn write_graph_snapshot_atomic_with_encryption(
    path: &Path,
    checkpoint_lsn: u64,
    payload: &[u8],
    encryption: Option<&Encryption>,
) -> io::Result<()> {
    // 加密在打开 tmp 之前完成：密钥错误不应留下半成品文件。
    // EN: Encrypt before creating the tmp — a key error must leave no residue.
    let encrypted = match encryption {
        Some(enc) => Some((encode_key_id(enc.active_id())?, enc.encrypt(payload)?)),
        None => None,
    };
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("graph_snapshot.bin");
    let dir = crate::parent_dir_or_dot(path);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dir.join(format!("{name}.{}.{nonce}.tmp", std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        use std::io::Write;
        f.write_all(GRAPH_SNAPSHOT_MAGIC)?;
        match &encrypted {
            Some((key_id, blob)) => {
                // 与 encode 的 CRC 覆盖一致：key_id‖blob。 / key_id‖blob.
                // EN: Matches encode's CRC coverage: key_id‖blob.
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(key_id);
                hasher.update(blob);
                f.write_all(&GRAPH_SNAPSHOT_FILE_VERSION_ENCRYPTED.to_be_bytes())?;
                f.write_all(&checkpoint_lsn.to_be_bytes())?;
                f.write_all(&(blob.len() as u64).to_be_bytes())?;
                f.write_all(&hasher.finalize().to_be_bytes())?;
                f.write_all(key_id)?;
                f.write_all(blob)?;
            }
            None => {
                f.write_all(&GRAPH_SNAPSHOT_FILE_VERSION.to_be_bytes())?;
                f.write_all(&checkpoint_lsn.to_be_bytes())?;
                f.write_all(&(payload.len() as u64).to_be_bytes())?;
                f.write_all(&crc32fast::hash(payload).to_be_bytes())?;
                f.write_all(payload)?;
            }
        }
        // 注入点（P4.5.4）：内容已写、尚未 fsync/rename —— panic/pause 让 kill -9
        // 落在「tmp 半成品残留」窗口，验证下次启动对残留 tmp 的处置与旧快照可用性。
        // EN: payload written but not yet synced/renamed — targets the window that
        // leaves a half-written tmp behind; validates restart behavior on residue.
        crate::fail_point!("graph_snapshot_write_mid");
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    // rename 的目录项持久化：崩溃后 tmp 不残留、新名必定可见。dir fsync 在
    // 个别平台/文件系统（如 Windows 旧 API）可能失败——不隐瞒，warn 明示
    // 崩溃耐久保证已降级（OCR 8eac6b8）。
    // EN: Directory-entry durability after rename: post-crash no tmp remains and
    // the new name must be visible. dir fsync may fail on some platforms or
    // filesystems (e.g. old Windows APIs) — warn that crash durability degraded.
    match fs::File::open(dir).and_then(|d| d.sync_all()) {
        Ok(()) => {}
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "graph snapshot: directory fsync failed — crash-durability of the rename degraded"
            );
        }
    }
    Ok(())
}

/// 读取快照文件：不存在 → `Ok(None)`；损坏 → [`GraphSnapshotError::Corrupt`]
/// （调用方 warn + 全量重放）；加密文件缺密钥 → [`GraphSnapshotError::MissingKey`]
/// （调用方必须让启动失败）。 / EN: Missing → `Ok(None)`; corrupt → fallback;
/// missing key → fatal.
pub fn read_graph_snapshot(path: &Path) -> Result<Option<GraphSnapshotFile>, GraphSnapshotError> {
    read_graph_snapshot_with_encryption(path, None)
}

/// 带解密的读取（P4.7.3）。 / EN: Read with decryption (P4.7.3).
pub fn read_graph_snapshot_with_encryption(
    path: &Path,
    encryption: Option<&Encryption>,
) -> Result<Option<GraphSnapshotFile>, GraphSnapshotError> {
    let mut bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        // ErrorKind 进消息：权限/盘满/瞬时 I/O 与「损坏」对调用方含义不同
        // （全量重放 vs 运维处置），字符串化不得丢 kind（OCR 8eac6b8）。
        // EN: Keep ErrorKind in the message: permission/disk-full/transient I/O
        // mean different things to callers than corruption (full replay vs ops
        // action); stringification must not drop the kind (OCR 8eac6b8).
        Err(e) => {
            return Err(GraphSnapshotError::corrupt(format!(
                "graph snapshot: read {}: {e} (kind: {:?})",
                path.display(),
                e.kind()
            )))
        }
    };
    let (version, header_len) = envelope_version(&bytes)?;
    // v2 复用 decode 的单一实现（OCR 批次三十五）：两份逐行相同的解密逻辑
    // 迟早会在错误分类/信封演进上分叉；v1 仍走零拷贝 drain。
    // EN: v2 delegates to the shared decoder; only v1 keeps the zero-copy drain.
    if version == GRAPH_SNAPSHOT_FILE_VERSION_ENCRYPTED {
        return decode_graph_snapshot_file_with_encryption(&bytes, encryption).map(Some);
    }
    let (checkpoint_lsn, _) = validate_body(&bytes, header_len, header_len)?;
    // drain 头即得 payload 所有权：整文件只有一次分配（图 payload 可达数百 MB）。
    // EN: Draining the header yields payload ownership with a single allocation
    // per file (graph payloads can reach hundreds of MB).
    bytes.drain(..header_len);
    Ok(Some(GraphSnapshotFile {
        checkpoint_lsn,
        payload: bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_encryption(id: &str, byte: u8) -> Encryption {
        let hex = format!("{byte:02x}").repeat(crate::encryption::KEY_LEN);
        let provider =
            crate::encryption::LocalKeyProvider::parse(&format!("{id}={hex}"), None).unwrap();
        Encryption::from_provider(std::sync::Arc::new(provider)).unwrap()
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "houyi-graphsnap-test-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn roundtrip_envelope() {
        let payload = b"fake-graph-payload".to_vec();
        let bytes = encode_graph_snapshot_file(42, &payload);
        let dec = decode_graph_snapshot_file(&bytes).unwrap();
        assert_eq!(
            dec,
            GraphSnapshotFile {
                checkpoint_lsn: 42,
                payload
            }
        );
    }

    #[test]
    fn decode_rejects_short_header() {
        let bytes = encode_graph_snapshot_file(1, b"abc");
        for cut in [0, 9, 10, GRAPH_SNAPSHOT_HEADER_LEN - 1] {
            assert!(
                decode_graph_snapshot_file(&bytes[..cut]).is_err(),
                "cut {cut} must fail"
            );
        }
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = encode_graph_snapshot_file(1, b"abc");
        bytes[0] ^= 0xFF;
        assert!(decode_graph_snapshot_file(&bytes)
            .unwrap_err()
            .to_string()
            .contains("bad magic"));
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let mut bytes = encode_graph_snapshot_file(1, b"abc");
        bytes[10..14].copy_from_slice(&7u32.to_be_bytes());
        assert!(decode_graph_snapshot_file(&bytes)
            .unwrap_err()
            .to_string()
            .contains("unsupported version 7"));
    }

    #[test]
    fn decode_rejects_crc_mismatch() {
        let mut bytes = encode_graph_snapshot_file(1, b"abcdef");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(decode_graph_snapshot_file(&bytes)
            .unwrap_err()
            .to_string()
            .contains("crc32 mismatch"));
    }

    #[test]
    fn decode_rejects_truncated_payload_and_trailing() {
        let bytes = encode_graph_snapshot_file(1, b"abcdef");
        let cut = bytes.len() - 2;
        assert!(decode_graph_snapshot_file(&bytes[..cut])
            .unwrap_err()
            .to_string()
            .contains("truncated"));
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(decode_graph_snapshot_file(&longer)
            .unwrap_err()
            .to_string()
            .contains("trailing"));
    }

    #[test]
    fn atomic_write_read_back_and_no_tmp_residue() {
        let dir = tmp_dir("atomic");
        let path = dir.join("c1.graph_snapshot.bin");
        write_graph_snapshot_atomic(&path, 77, b"payload-bytes").unwrap();
        let got = read_graph_snapshot(&path)
            .unwrap()
            .expect("snapshot exists");
        assert_eq!(got.checkpoint_lsn, 77);
        assert_eq!(got.payload, b"payload-bytes");
        // tmp 名带 pid+nonce（OCR 8eac6b8）：rename 后目录里不得有任何 *.tmp 残留。
        // EN: tmp names carry pid+nonce (OCR 8eac6b8): no *.tmp may remain after rename.
        let tmp_left = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(tmp_left, 0, "no tmp residue after rename");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_missing_returns_none() {
        let dir = tmp_dir("missing");
        let path = dir.join("nope.graph_snapshot.bin");
        assert_eq!(read_graph_snapshot(&path).unwrap(), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_corrupt_file_returns_err() {
        let dir = tmp_dir("corrupt");
        let path = dir.join("bad.graph_snapshot.bin");
        fs::write(&path, b"garbage").unwrap();
        assert!(read_graph_snapshot(&path).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    /// 回归（OCR 8eac6b8）：伪造的巨型 payload_len（CRC 只覆盖 payload，护不住
    /// header）必须是干净 Err——debug 下不得加法 panic，release 下不得回绕误报。
    /// EN: Regression (OCR 8eac6b8): a forged giant payload_len (CRC covers only
    /// the payload, not the header) must yield a clean Err — no add panic in
    /// debug and no wrap into bogus values in release.
    #[test]
    fn decode_rejects_length_overflow() {
        let mut bytes = encode_graph_snapshot_file(1, b"abc");
        bytes[22..30].copy_from_slice(&u64::MAX.to_be_bytes());
        let err = decode_graph_snapshot_file(&bytes).unwrap_err().to_string();
        assert!(
            err.contains("overflow") || err.contains("exceeds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn wal_derived_path() {
        let p = graph_snapshot_path_for_wal(Path::new("/data/drill.wal"));
        assert_eq!(p, Path::new("/data/drill.graph_snapshot.bin"));
    }

    /// P4.7.3：v2 信封的往返、明文不落盘、缺密钥必须可判别。
    /// EN: P4.7.3: v2 envelope roundtrip, no plaintext on disk, detectable missing key.
    #[test]
    fn encrypted_roundtrip_hides_payload_and_marks_missing_key() {
        let enc = test_encryption("k1", 7);
        let payload = b"vector-graph-secret-payload".to_vec();
        let bytes = encode_graph_snapshot_file_with_encryption(9, &payload, Some(&enc)).unwrap();
        let version = u32::from_be_bytes(bytes[10..14].try_into().unwrap());
        assert_eq!(version, GRAPH_SNAPSHOT_FILE_VERSION_ENCRYPTED);
        assert!(
            !bytes.windows(payload.len()).any(|w| w == payload),
            "plaintext payload must not appear in the encrypted envelope"
        );
        let got = decode_graph_snapshot_file_with_encryption(&bytes, Some(&enc)).unwrap();
        assert_eq!(got.checkpoint_lsn, 9);
        assert_eq!(got.payload, payload);

        // 未配置 keyring → 缺密钥（类型化，启动必须失败）。
        // EN: No keyring configured → missing key (typed; startup must fail).
        let err = decode_graph_snapshot_file_with_encryption(&bytes, None).unwrap_err();
        assert!(err.is_missing_key(), "got: {err}");
        // keyring 里没有该 id → 同样是缺密钥。
        // EN: The id is absent from the keyring → also a missing key.
        let other_id = test_encryption("k2", 8);
        let err = decode_graph_snapshot_file_with_encryption(&bytes, Some(&other_id)).unwrap_err();
        assert!(err.is_missing_key(), "got: {err}");
        // 同 id、不同密钥 → AEAD 失败（损坏/错密钥，不是缺密钥）。
        // EN: Same id but different key material → AEAD failure (corruption or
        // wrong key, not a missing key).
        let wrong_material = test_encryption("k1", 8);
        let err =
            decode_graph_snapshot_file_with_encryption(&bytes, Some(&wrong_material)).unwrap_err();
        assert!(!err.is_missing_key(), "got: {err}");
        assert!(err.to_string().contains("decrypt failed"), "got: {err}");

        // 篡改密文 → CRC 先报错（结构损坏 → 回退全量重放）。
        // EN: Tampered ciphertext → CRC fails first (structural corruption → full replay).
        let mut tampered = bytes.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        let err = decode_graph_snapshot_file_with_encryption(&tampered, Some(&enc)).unwrap_err();
        assert!(!err.is_missing_key());
        assert!(err.to_string().contains("crc32 mismatch"), "got: {err}");

        // 篡改 key id 槽位（仍是合法 id）：CRC 覆盖该槽位 → 归为损坏而非缺密钥。
        // EN: a still-valid-looking key id bit flip must classify as corruption.
        let mut key_tampered = bytes.clone();
        key_tampered[GRAPH_SNAPSHOT_HEADER_LEN] ^= 0x01; // 'k' → 'j'
        let err =
            decode_graph_snapshot_file_with_encryption(&key_tampered, Some(&enc)).unwrap_err();
        assert!(!err.is_missing_key(), "got: {err}");
        assert!(err.to_string().contains("crc32 mismatch"), "got: {err}");
    }

    /// P4.7.3：加密落盘 + 读回；迁移期明文 v1 在启用加密的节点上仍可读。
    /// EN: P4.7.3: encrypted write + readback; plaintext v1 stays readable on
    /// encrypted nodes during migration.
    #[test]
    fn encrypted_atomic_write_and_plaintext_compat() {
        let dir = tmp_dir("encrypted");
        let enc = test_encryption("k1", 3);
        let path = dir.join("c1.graph_snapshot.bin");
        write_graph_snapshot_atomic_with_encryption(&path, 12, b"top-secret-graph", Some(&enc))
            .unwrap();
        let raw = fs::read(&path).unwrap();
        assert!(
            !raw.windows(b"top-secret-graph".len())
                .any(|w| w == b"top-secret-graph"),
            "plaintext must not be written to disk"
        );
        let got = read_graph_snapshot_with_encryption(&path, Some(&enc))
            .unwrap()
            .expect("snapshot exists");
        assert_eq!(got.checkpoint_lsn, 12);
        assert_eq!(got.payload, b"top-secret-graph");
        let err = read_graph_snapshot_with_encryption(&path, None).unwrap_err();
        assert!(err.is_missing_key(), "got: {err}");

        let plain = dir.join("plain.graph_snapshot.bin");
        write_graph_snapshot_atomic(&plain, 5, b"plain").unwrap();
        let got = read_graph_snapshot_with_encryption(&plain, Some(&enc))
            .unwrap()
            .expect("plaintext snapshot readable during migration");
        assert_eq!(got.payload, b"plain");
        // 旧 API（无密钥）读明文仍工作。
        // EN: The old (key-less) API still reads plaintext.
        assert_eq!(
            read_graph_snapshot(&plain).unwrap().unwrap().payload,
            b"plain"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
