//! Collection 快照 payload 与 manifest（P5.0.2）。
//! Collection snapshot payload and manifest (P5.0.2).
//!
//! 快照文件本身复用 [`crate::graph_snapshot_file`] 的信封（checkpoint/CRC/AES-GCM/
//! 原子写）；本模块只定义信封内的 payload 与 `snapshots/collections.manifest`。
//! 标量行保存**完整物理页**（含 stale 行 + 墓碑），确保「快照 + 尾段重放」与全量
//! 重放逐行等价，不会因为快照隐式压实而改变过滤语义。
//! EN: The snapshot file reuses the graph-snapshot envelope (checkpoint/CRC/AES-GCM/
//! atomic write); this module defines only the payload and the manifest. Scalar rows
//! keep the complete physical page (stale rows + tombstones), so snapshot + tail
//! replay is row-for-row equivalent to full replay and cannot implicitly compact.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const COLLECTION_SNAPSHOT_MAGIC: &[u8; 9] = b"HOUYISNAP";
/// v2 增加 collection 的度量字节与 HNSW 参数；v1 仍可解码（按 index_type 历史
/// 语义推断度量、HNSW 用库默认）。v1 二进制无法读取 v2（版本号 fail-fast），
/// 降级前必须确认没有新快照。
/// EN: v2 adds the collection metric byte and HNSW parameters; v1 remains
/// decodable (metric inferred from index_type, HNSW defaults). A v1 binary
/// rejects v2 via the version check, so downgrades must not have v2 snapshots.
pub const COLLECTION_SNAPSHOT_VERSION: u32 = 2;
/// 解码仍接受的最早历史版本（v1）。
/// EN: Oldest historical version still accepted by the decoder (v1).
pub const COLLECTION_SNAPSHOT_MIN_VERSION: u32 = 1;
pub const COLLECTION_MANIFEST_MAGIC: &[u8; 8] = b"HOUYIMAN";
pub const COLLECTION_MANIFEST_VERSION: u32 = 1;
/// manifest JSON 上限（防御性；正常集合元数据远小于此）。
/// EN: Manifest JSON cap (defensive; normal metadata is far smaller).
pub const MAX_MANIFEST_BODY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_NAME_BYTES: usize = 4096;
const MAX_INDEX_TYPE_BYTES: usize = 64;

/// 快照内的向量形态。
/// EN: Vector shape inside a collection snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotVectorKind {
    Empty = 0,
    /// HNSW 图快照字节（由 houyi-vector 编码，本层不透明）。
    /// EN: HNSW graph snapshot bytes (opaque to this layer).
    HnswGraph = 1,
    /// 原始 f32 向量：[id u64][dims × f32 BE] 重复 count 次。
    /// EN: Raw f32 vectors: repeated [id u64][dims × f32 BE].
    RawF32 = 2,
}

impl SnapshotVectorKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Empty),
            1 => Some(Self::HnswGraph),
            2 => Some(Self::RawF32),
            _ => None,
        }
    }
}

/// 标量值（与 houyi-core::ScalarValue 同构；persistence 不反向依赖 core）。
/// EN: Scalar value (isomorphic to houyi-core::ScalarValue; persistence must not
/// depend on core).
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotScalar {
    I64(i64),
    F64(f64),
    Bool(bool),
    Text(String),
    Timestamp(i64),
    JsonText(String),
}

/// 一行物理标量（列名按字典序编码，保证确定性）。
/// EN: One physical scalar row (columns sorted for deterministic encoding).
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotScalarRow {
    pub id: u64,
    pub columns: Vec<(String, SnapshotScalar)>,
}

/// 快照中记录的 HNSW 图参数（v2+；仅 hnsw 后端）。
/// EN: HNSW graph parameters recorded in v2+ snapshots (hnsw backend only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotHnswParams {
    pub m: u32,
    pub ef_construction: u32,
    pub ef_search: u32,
}

/// 单个 collection 的快照 payload。
/// EN: Per-collection snapshot payload.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionSnapshotPayload {
    pub checkpoint_lsn: u64,
    pub name: String,
    pub dims: u64,
    pub index_type: String,
    pub ha_level: u8,
    /// 度量字节（0=L2/1=Cosine/2=IP/3=L1）；None = v1 旧快照，恢复时按
    /// index_type 推断（与旧服务端行为一致）。
    /// EN: Metric byte (0=L2/1=Cosine/2=IP/3=L1); None = legacy v1 snapshot,
    /// inferred from index_type at restore (matching the old server behavior).
    pub metric: Option<u8>,
    /// HNSW 参数；None = v1 快照或非 hnsw 后端。
    /// EN: HNSW parameters; None for v1 snapshots or non-hnsw backends.
    pub hnsw: Option<SnapshotHnswParams>,
    pub vector_kind: SnapshotVectorKind,
    pub vector_count: u64,
    pub vector_bytes: Vec<u8>,
    pub scalars: Vec<SnapshotScalarRow>,
    pub tombstones: Vec<u64>,
}

/// manifest 中的一个 collection 条目。
/// EN: One collection entry in the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionManifestEntry {
    pub name: String,
    /// 该 collection 快照覆盖到的 LSN（独立于其他 collection）。
    /// EN: LSN covered by this collection's snapshot (independent per collection).
    pub checkpoint_lsn: u64,
    pub dims: u64,
    pub index_type: String,
    pub ha_level: u8,
    /// 度量字节（v2+ 写入；旧 manifest 缺省为 None，加载时不强制比对）。
    /// EN: Metric byte (written by v2+; absent in legacy manifests, in which
    /// case loading does not enforce equality).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<u8>,
    pub vector_kind: u8,
    /// snapshots/ 下的裸文件名。
    /// EN: Bare filename under snapshots/.
    pub file: String,
    pub file_len: u64,
    /// 信封内明文 payload 的 CRC32。
    /// EN: CRC32 of the plaintext payload.
    pub payload_crc32: u32,
    pub vector_count: u64,
    pub scalar_rows: u64,
}

/// collections.manifest 的 JSON body。
/// EN: JSON body of collections.manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionManifest {
    pub format_version: u32,
    pub manifest_seq: u64,
    pub created_at_unix: u64,
    /// 本轮稳定名单后的全局屏障；新建 collection 的 Create LSN 必大于它。
    /// EN: Global barrier after the stable name set; a later Create has a greater LSN.
    pub global_barrier_lsn: u64,
    /// 所有 entry checkpoint 的最小值；WAL 截断只能到这里。
    /// EN: Minimum entry checkpoint; WAL truncation may not exceed it.
    pub safe_checkpoint_lsn: u64,
    pub collections: Vec<CollectionManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotDecodeError {
    Corrupt(String),
}

impl std::fmt::Display for SnapshotDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for SnapshotDecodeError {}

fn corrupt(msg: impl Into<String>) -> SnapshotDecodeError {
    SnapshotDecodeError::Corrupt(msg.into())
}

// ── payload 编解码 ─────────────────────────────────────────
// EN: Payload codec.

pub fn encode_collection_snapshot_payload(
    payload: &CollectionSnapshotPayload,
) -> io::Result<Vec<u8>> {
    // 编码侧与解码侧保持同一套合法性检查：否则写出一个自己读不回的 payload，
    // 把写入侧 bug 推迟成恢复期故障。
    // EN: Enforce the same invariants on encode as on decode; otherwise a writer
    // bug produces a payload this crate later refuses to read.
    if payload.dims == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection snapshot: dims must be > 0",
        ));
    }
    if crate::ha_level::HaLevel::from_code(payload.ha_level).is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "collection snapshot: invalid ha_level code {}",
                payload.ha_level
            ),
        ));
    }
    // v2 必须携带度量：快照脱离 WAL 后，度量只存在于这里，缺失会让恢复按
    // 历史默认重建（L2/IP/L1 集合被静默换成 Cosine 计分）。
    // EN: v2 must carry the metric — once the snapshot is detached from the WAL,
    // it is the only record; its absence would rebuild L2/IP/L1 collections as
    // cosine silently.
    let metric = payload.metric.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection snapshot: v2 payload requires a metric byte",
        )
    })?;
    if metric > 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("collection snapshot: invalid metric byte {metric}"),
        ));
    }
    // v2 的 HNSW 后端必须携带参数：缺失会让恢复退回库层默认值，静默改变
    // 集合的图参数（OCR #06）。反向同理——参数只属于 hnsw 后端。m 下限 2
    // 与库层 set_hnsw_m 一致（OCR #39）。
    // EN: A v2 hnsw backend must carry its parameters: absence would restore
    // library defaults and silently change the collection's graph parameters
    // (OCR #06). Conversely, params belong only to the hnsw backend. The m
    // lower bound of 2 matches the library's set_hnsw_m (OCR #39).
    match (payload.index_type.as_str(), payload.hnsw) {
        ("hnsw", None) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "collection snapshot: hnsw backend requires hnsw params",
            ));
        }
        (_, Some(_)) if payload.index_type != "hnsw" => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "collection snapshot: hnsw params require index_type 'hnsw'",
            ));
        }
        ("hnsw", Some(p)) => {
            if p.m < 2 || p.m > 1024 || p.ef_construction == 0 || p.ef_construction > 65_536 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "collection snapshot: hnsw params out of range",
                ));
            }
            if p.ef_search == 0 || p.ef_search > 65_536 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "collection snapshot: hnsw ef_search out of range",
                ));
            }
        }
        _ => {}
    }
    let name = payload.name.as_bytes();
    let index_type = payload.index_type.as_bytes();
    if name.len() > MAX_NAME_BYTES || name.len() > u16::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection snapshot: name too long",
        ));
    }
    if index_type.len() > MAX_INDEX_TYPE_BYTES || index_type.len() > u16::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection snapshot: index_type too long",
        ));
    }
    let vector_kind = payload.vector_kind as u8;
    match payload.vector_kind {
        SnapshotVectorKind::Empty => {
            if payload.vector_count != 0 || !payload.vector_bytes.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "collection snapshot: EMPTY vectors require count=0 and no bytes",
                ));
            }
        }
        SnapshotVectorKind::RawF32 => {
            let per = 8u64
                .checked_add(
                    payload.dims.checked_mul(4).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "dims overflow")
                    })?,
                )
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "dims overflow"))?;
            let expected = payload.vector_count.checked_mul(per).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "vector size overflow")
            })?;
            if expected != payload.vector_bytes.len() as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "collection snapshot: RAW_F32 bytes {} != count×per {}",
                        payload.vector_bytes.len(),
                        expected
                    ),
                ));
            }
        }
        SnapshotVectorKind::HnswGraph => {}
    }

    let mut out = Vec::new();
    out.extend_from_slice(COLLECTION_SNAPSHOT_MAGIC);
    out.extend_from_slice(&COLLECTION_SNAPSHOT_VERSION.to_be_bytes());
    out.extend_from_slice(&payload.checkpoint_lsn.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // flags (reserved)
    out.extend_from_slice(&(name.len() as u16).to_be_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(&payload.dims.to_be_bytes());
    out.extend_from_slice(&(index_type.len() as u16).to_be_bytes());
    out.extend_from_slice(index_type);
    out.push(payload.ha_level);
    out.push(metric);
    match payload.hnsw {
        None => out.push(0),
        Some(p) => {
            out.push(1);
            out.extend_from_slice(&p.m.to_be_bytes());
            out.extend_from_slice(&p.ef_construction.to_be_bytes());
            out.extend_from_slice(&p.ef_search.to_be_bytes());
        }
    }
    out.push(vector_kind);
    out.extend_from_slice(&payload.vector_count.to_be_bytes());
    out.extend_from_slice(&(payload.vector_bytes.len() as u64).to_be_bytes());
    out.extend_from_slice(&payload.vector_bytes);
    out.extend_from_slice(&(payload.scalars.len() as u64).to_be_bytes());
    for row in &payload.scalars {
        out.extend_from_slice(&row.id.to_be_bytes());
        out.extend_from_slice(&(row.columns.len() as u32).to_be_bytes());
        let mut columns: Vec<&(String, SnapshotScalar)> = row.columns.iter().collect();
        columns.sort_by(|a, b| a.0.cmp(&b.0));
        for (column, value) in columns {
            let cb = column.as_bytes();
            if cb.len() > MAX_NAME_BYTES || cb.len() > u16::MAX as usize {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "collection snapshot: column name too long",
                ));
            }
            out.extend_from_slice(&(cb.len() as u16).to_be_bytes());
            out.extend_from_slice(cb);
            encode_scalar(&mut out, value)?;
        }
    }
    out.extend_from_slice(&(payload.tombstones.len() as u64).to_be_bytes());
    for id in &payload.tombstones {
        out.extend_from_slice(&id.to_be_bytes());
    }
    Ok(out)
}

fn encode_scalar(out: &mut Vec<u8>, value: &SnapshotScalar) -> io::Result<()> {
    match value {
        SnapshotScalar::I64(v) => {
            out.push(1);
            out.extend_from_slice(&v.to_be_bytes());
        }
        SnapshotScalar::F64(v) => {
            out.push(2);
            out.extend_from_slice(&v.to_bits().to_be_bytes());
        }
        SnapshotScalar::Bool(v) => {
            out.push(3);
            out.push(u8::from(*v));
        }
        SnapshotScalar::Text(s) | SnapshotScalar::JsonText(s) => {
            let len: u32 = s
                .len()
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "scalar text too long"))?;
            out.push(if matches!(value, SnapshotScalar::Text(_)) {
                4
            } else {
                6
            });
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        SnapshotScalar::Timestamp(v) => {
            out.push(5);
            out.extend_from_slice(&v.to_be_bytes());
        }
    }
    Ok(())
}

pub fn decode_collection_snapshot_payload(
    bytes: &[u8],
) -> Result<CollectionSnapshotPayload, SnapshotDecodeError> {
    let mut r = ByteReader::new(bytes);
    if r.take(COLLECTION_SNAPSHOT_MAGIC.len())? != COLLECTION_SNAPSHOT_MAGIC {
        return Err(corrupt("collection snapshot: bad magic"));
    }
    let version = r.u32()?;
    if !(COLLECTION_SNAPSHOT_MIN_VERSION..=COLLECTION_SNAPSHOT_VERSION).contains(&version) {
        return Err(corrupt(format!(
            "collection snapshot: unsupported version {version}"
        )));
    }
    let checkpoint_lsn = r.u64()?;
    let flags = r.u32()?;
    if flags != 0 {
        return Err(corrupt("collection snapshot: reserved flags must be zero"));
    }
    let name = r.string_u16_bounded(MAX_NAME_BYTES)?;
    let dims = r.u64()?;
    if dims == 0 {
        return Err(corrupt("collection snapshot: dims must be > 0"));
    }
    let index_type = r.string_u16_bounded(MAX_INDEX_TYPE_BYTES)?;
    let ha_level = r.u8()?;
    if crate::ha_level::HaLevel::from_code(ha_level).is_none() {
        return Err(corrupt(format!(
            "collection snapshot: invalid ha_level code {ha_level}"
        )));
    }
    // v1 不携带度量/HNSW：保留 None，由调用方按 index_type 历史语义推断。
    // EN: v1 carries no metric/HNSW; keep None and let the caller infer from
    // index_type (legacy semantics).
    let (metric, hnsw) = if version >= 2 {
        let metric = r.u8()?;
        if metric > 3 {
            return Err(corrupt(format!(
                "collection snapshot: invalid metric byte {metric}"
            )));
        }
        let flag = r.u8()?;
        let hnsw = match flag {
            0 => None,
            1 => {
                let m = r.u32()?;
                let ef_construction = r.u32()?;
                let ef_search = r.u32()?;
                // m 下限 2 与库层 set_hnsw_m 一致（OCR #39）。
                // EN: The m lower bound of 2 matches the library's set_hnsw_m.
                if !(2..=1024).contains(&m) || !(1..=65_536).contains(&ef_construction) {
                    return Err(corrupt("collection snapshot: hnsw params out of range"));
                }
                if ef_search == 0 || ef_search > 65_536 {
                    return Err(corrupt("collection snapshot: hnsw ef_search out of range"));
                }
                Some(SnapshotHnswParams {
                    m,
                    ef_construction,
                    ef_search,
                })
            }
            other => {
                return Err(corrupt(format!(
                    "collection snapshot: invalid hnsw presence flag {other}"
                )))
            }
        };
        if hnsw.is_some() && index_type != "hnsw" {
            return Err(corrupt(
                "collection snapshot: hnsw params require index_type 'hnsw'",
            ));
        }
        // v2 的 hnsw 后端必须带参数：缺失会按库层默认恢复，静默改变图参数（OCR #06）。
        // EN: A v2 hnsw backend must carry params; absence would restore library
        // defaults and silently change graph parameters (OCR #06).
        if hnsw.is_none() && index_type == "hnsw" {
            return Err(corrupt(
                "collection snapshot: hnsw backend requires hnsw params",
            ));
        }
        (Some(metric), hnsw)
    } else {
        (None, None)
    };
    let vector_kind_raw = r.u8()?;
    let vector_kind = SnapshotVectorKind::from_u8(vector_kind_raw).ok_or_else(|| {
        corrupt(format!(
            "collection snapshot: unknown vector kind {vector_kind_raw}"
        ))
    })?;
    let vector_count = r.u64()?;
    let vector_bytes_len = r.u64()?;
    let vector_bytes = r.take_len(vector_bytes_len)?;
    match vector_kind {
        SnapshotVectorKind::Empty => {
            if vector_count != 0 || !vector_bytes.is_empty() {
                return Err(corrupt(
                    "collection snapshot: EMPTY vectors require count=0 and no bytes",
                ));
            }
        }
        SnapshotVectorKind::RawF32 => {
            let per = 8u64
                .checked_add(
                    dims.checked_mul(4)
                        .ok_or_else(|| corrupt("dims overflow"))?,
                )
                .ok_or_else(|| corrupt("dims overflow"))?;
            let expected = vector_count
                .checked_mul(per)
                .ok_or_else(|| corrupt("vector size overflow"))?;
            if expected != vector_bytes_len {
                return Err(corrupt(format!(
                    "collection snapshot: RAW_F32 bytes {vector_bytes_len} != count×per {expected}"
                )));
            }
        }
        SnapshotVectorKind::HnswGraph => {}
    }

    let scalar_row_count = r.u64()?;
    let min_row = 12u64; // id + column_count
    if scalar_row_count > (r.remaining() as u64) / min_row {
        return Err(corrupt(
            "collection snapshot: scalar row count exceeds remaining bytes",
        ));
    }
    let mut scalars = Vec::new();
    scalars
        .try_reserve(scalar_row_count as usize)
        .map_err(|_| corrupt("collection snapshot: scalar rows allocation failed"))?;
    for _ in 0..scalar_row_count {
        let id = r.u64()?;
        let column_count = r.u32()?;
        let min_col = 3u64; // name_len u16 + tag u8
        if column_count as u64 > (r.remaining() as u64) / min_col {
            return Err(corrupt(
                "collection snapshot: column count exceeds remaining bytes",
            ));
        }
        let mut columns = Vec::new();
        columns
            .try_reserve(column_count as usize)
            .map_err(|_| corrupt("collection snapshot: columns allocation failed"))?;
        let mut seen_columns = HashSet::with_capacity(column_count as usize);
        for _ in 0..column_count {
            let column = r.string_u16_bounded(MAX_NAME_BYTES)?;
            if !seen_columns.insert(column.clone()) {
                return Err(corrupt(format!(
                    "collection snapshot: duplicate column {column:?}"
                )));
            }
            let value = decode_scalar(&mut r)?;
            columns.push((column, value));
        }
        scalars.push(SnapshotScalarRow { id, columns });
    }
    let tombstone_count = r.u64()?;
    if tombstone_count > (r.remaining() as u64) / 8 {
        return Err(corrupt(
            "collection snapshot: tombstone count exceeds remaining bytes",
        ));
    }
    let mut tombstones = Vec::new();
    tombstones
        .try_reserve(tombstone_count as usize)
        .map_err(|_| corrupt("collection snapshot: tombstone allocation failed"))?;
    for _ in 0..tombstone_count {
        tombstones.push(r.u64()?);
    }
    if r.remaining() != 0 {
        return Err(corrupt(format!(
            "collection snapshot: trailing {} byte(s)",
            r.remaining()
        )));
    }
    Ok(CollectionSnapshotPayload {
        checkpoint_lsn,
        name,
        dims,
        index_type,
        ha_level,
        metric,
        hnsw,
        vector_kind,
        vector_count,
        vector_bytes,
        scalars,
        tombstones,
    })
}

fn decode_scalar(r: &mut ByteReader<'_>) -> Result<SnapshotScalar, SnapshotDecodeError> {
    let tag = r.u8()?;
    match tag {
        1 => Ok(SnapshotScalar::I64(r.i64()?)),
        2 => Ok(SnapshotScalar::F64(f64::from_bits(r.u64()?))),
        3 => Ok(SnapshotScalar::Bool(r.u8()? != 0)),
        4 => Ok(SnapshotScalar::Text(r.string_u32()?)),
        5 => Ok(SnapshotScalar::Timestamp(r.i64()?)),
        6 => Ok(SnapshotScalar::JsonText(r.string_u32()?)),
        other => Err(corrupt(format!(
            "collection snapshot: unknown scalar tag {other}"
        ))),
    }
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], SnapshotDecodeError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| corrupt("collection snapshot: length overflow"))?;
        if end > self.bytes.len() {
            return Err(corrupt("collection snapshot: truncated payload"));
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn take_len(&mut self, n: u64) -> Result<Vec<u8>, SnapshotDecodeError> {
        let n: usize = n
            .try_into()
            .map_err(|_| corrupt("collection snapshot: length too large"))?;
        Ok(self.take(n)?.to_vec())
    }

    fn u8(&mut self) -> Result<u8, SnapshotDecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SnapshotDecodeError> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, SnapshotDecodeError> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, SnapshotDecodeError> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, SnapshotDecodeError> {
        Ok(i64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string_u16_bounded(&mut self, max: usize) -> Result<String, SnapshotDecodeError> {
        let n = self.u16()? as usize;
        if n > max {
            return Err(corrupt("collection snapshot: string length exceeds cap"));
        }
        let s = std::str::from_utf8(self.take(n)?)
            .map_err(|_| corrupt("collection snapshot: invalid UTF-8"))?;
        Ok(s.to_string())
    }

    fn string_u32(&mut self) -> Result<String, SnapshotDecodeError> {
        let n = self.u32()? as usize;
        let s = std::str::from_utf8(self.take(n)?)
            .map_err(|_| corrupt("collection snapshot: invalid UTF-8"))?;
        Ok(s.to_string())
    }
}

// ── manifest 编解码 ────────────────────────────────────────
// EN: Manifest codec.

pub fn encode_collection_manifest(manifest: &CollectionManifest) -> io::Result<Vec<u8>> {
    validate_manifest(manifest).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let body = serde_json::to_vec(manifest)?;
    if body.len() as u64 > MAX_MANIFEST_BODY_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "collection manifest: body too large",
        ));
    }
    let mut out = Vec::with_capacity(24 + body.len());
    out.extend_from_slice(COLLECTION_MANIFEST_MAGIC);
    out.extend_from_slice(&COLLECTION_MANIFEST_VERSION.to_be_bytes());
    out.extend_from_slice(&(body.len() as u64).to_be_bytes());
    out.extend_from_slice(&crc32fast::hash(&body).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

pub fn decode_collection_manifest(bytes: &[u8]) -> Result<CollectionManifest, SnapshotDecodeError> {
    if bytes.len() < 24 {
        return Err(corrupt("collection manifest: header truncated"));
    }
    if &bytes[..8] != COLLECTION_MANIFEST_MAGIC {
        return Err(corrupt("collection manifest: bad magic"));
    }
    let version = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    if version != COLLECTION_MANIFEST_VERSION {
        return Err(corrupt(format!(
            "collection manifest: unsupported version {version}"
        )));
    }
    let body_len = u64::from_be_bytes(bytes[12..20].try_into().unwrap());
    if body_len > MAX_MANIFEST_BODY_BYTES {
        return Err(corrupt("collection manifest: body too large"));
    }
    let crc = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    let expected_len = 24usize
        .checked_add(
            usize::try_from(body_len)
                .map_err(|_| corrupt("collection manifest: body length overflow"))?,
        )
        .ok_or_else(|| corrupt("collection manifest: body length overflow"))?;
    if bytes.len() != expected_len {
        return Err(corrupt("collection manifest: trailing or truncated bytes"));
    }
    let body = &bytes[24..];
    if crc32fast::hash(body) != crc {
        return Err(corrupt("collection manifest: CRC mismatch"));
    }
    let manifest: CollectionManifest =
        serde_json::from_slice(body).map_err(|e| corrupt(format!("collection manifest: {e}")))?;
    validate_manifest(&manifest).map_err(corrupt)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &CollectionManifest) -> Result<(), String> {
    if manifest.format_version != 1 {
        return Err(format!(
            "collection manifest: unsupported format_version {}",
            manifest.format_version
        ));
    }
    let mut names = std::collections::HashSet::new();
    let mut min_checkpoint: Option<u64> = None;
    for entry in &manifest.collections {
        if entry.name.is_empty() {
            return Err("collection manifest: empty collection name".to_string());
        }
        if !names.insert(entry.name.clone()) {
            return Err(format!(
                "collection manifest: duplicate collection {:?}",
                entry.name
            ));
        }
        if !is_bare_file_name(&entry.file) {
            return Err(format!(
                "collection manifest: non-bare snapshot filename {:?}",
                entry.file
            ));
        }
        if entry.dims == 0 {
            return Err("collection manifest: dims must be > 0".to_string());
        }
        if SnapshotVectorKind::from_u8(entry.vector_kind).is_none() {
            return Err(format!(
                "collection manifest: unknown vector_kind {}",
                entry.vector_kind
            ));
        }
        if crate::ha_level::HaLevel::from_code(entry.ha_level).is_none() {
            return Err(format!(
                "collection manifest: invalid ha_level {}",
                entry.ha_level
            ));
        }
        // 度量字节与 payload 编码域一致（0..=3）：越界即损坏/未来格式，提前拒绝
        //（OCR #25）。None 是 v1 旧 manifest 的合法形态（无度量为准的约束）。
        // EN: The metric byte shares the payload's 0..=3 domain; anything else is
        // corruption or a future format (OCR #25). None is the legal legacy shape
        // of a v1 manifest (no metric-based constraint).
        if let Some(metric) = entry.metric {
            if metric > 3 {
                return Err(format!(
                    "collection manifest: invalid metric byte {metric} for {:?}",
                    entry.name
                ));
            }
        }
        if entry.checkpoint_lsn > manifest.global_barrier_lsn {
            return Err(format!(
                "collection manifest: {} checkpoint {} exceeds barrier {}",
                entry.name, entry.checkpoint_lsn, manifest.global_barrier_lsn
            ));
        }
        min_checkpoint =
            Some(min_checkpoint.map_or(entry.checkpoint_lsn, |m| m.min(entry.checkpoint_lsn)));
    }
    // 非空名单：safe = 各 entry checkpoint 的最小值；空名单（全部 collection 已
    // 删除）：safe = global_barrier_lsn —— 这是 WAL 可截断到的唯一安全点，
    // barrier 之后创建的 collection 其 Create LSN 必然更大、不会被误截。
    // EN: non-empty: safe = min entry checkpoint; empty (everything dropped):
    // safe = global_barrier_lsn, the only safe truncation point (post-barrier
    // Create LSNs are strictly greater and therefore never truncated).
    let expected_safe = min_checkpoint.unwrap_or(manifest.global_barrier_lsn);
    if manifest.safe_checkpoint_lsn != expected_safe {
        // 空名单时没有 entry，报「min entry checkpoint」会误导；统一用
        // expected safe checkpoint 措辞。两种来源在注释里已分别说明。
        // EN: an empty list has no entries, so "min entry checkpoint" would be
        // misleading; use a source-neutral "expected safe checkpoint" message.
        return Err(format!(
            "collection manifest: safe_checkpoint {} != expected safe checkpoint {expected_safe}",
            manifest.safe_checkpoint_lsn
        ));
    }
    Ok(())
}

fn is_bare_file_name(name: &str) -> bool {
    // `..` 作为子串且无分隔符时无害；只拒绝精确的穿越组件与路径分隔符。
    // EN: `..` as a substring is harmless without a separator; reject only exact
    // traversal components and path separators.
    !name.is_empty() && !name.contains('/') && !name.contains('\\') && name != "." && name != ".."
}

/// 原子写 manifest（tmp + fsync + rename + 父目录 fsync）。
/// EN: Atomic manifest write (tmp + fsync + rename + parent-dir fsync).
pub fn write_collection_manifest_atomic(
    path: &Path,
    manifest: &CollectionManifest,
) -> io::Result<()> {
    let bytes = encode_collection_manifest(manifest)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("collections.manifest");
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp = parent.join(format!(".{name}.{}.{nanos}.{seq}.tmp", std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    crate::sync_parent_dir(path)
}

pub fn read_collection_manifest(path: &Path) -> io::Result<Option<CollectionManifest>> {
    match std::fs::read(path) {
        Ok(bytes) => decode_collection_manifest(&bytes)
            .map(Some)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// 生成 collection 快照文件名：安全前缀 + 名称哈希 + checkpoint + epoch + seq。
/// EN: Build a collection snapshot filename: safe prefix + name hash + checkpoint +
/// epoch + seq.
///
/// 原始集合名不得直接进路径（可含 `/`/`..`）；非白名单字节百分号编码，并附
/// 精确名称的 FNV-1a 哈希避免截断/大小写碰撞。
/// EN: Raw names must not enter paths; non-allowlisted bytes are percent-encoded and
/// an FNV-1a hash of the exact name prevents truncation/case collisions.
pub fn snapshot_file_name(name: &str, checkpoint_lsn: u64, epoch_nanos: u128, seq: u64) -> String {
    let mut prefix = String::new();
    for b in name.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-') {
            prefix.push(*b as char);
        } else {
            prefix.push_str(&format!("%{b:02X}"));
        }
        if prefix.len() >= 64 {
            break;
        }
    }
    if prefix.is_empty() || prefix == "." || prefix == ".." {
        prefix = "c".to_string();
    }
    let hash = fnv1a64(name.as_bytes());
    format!("{prefix}.{hash:016x}.collection_snapshot.{checkpoint_lsn}.{epoch_nanos}.{seq}.bin")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// 快照目录（由 collections.wal 所在目录派生）。
/// EN: Snapshot directory derived from the collections.wal directory.
pub fn collection_snapshots_dir(collections_wal_path: &Path) -> PathBuf {
    collections_wal_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("snapshots")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_payload() -> CollectionSnapshotPayload {
        CollectionSnapshotPayload {
            checkpoint_lsn: 42,
            name: "docs".to_string(),
            dims: 4,
            index_type: "hnsw".to_string(),
            ha_level: 2,
            metric: Some(1),
            hnsw: Some(SnapshotHnswParams {
                m: 24,
                ef_construction: 400,
                ef_search: 256,
            }),
            vector_kind: SnapshotVectorKind::HnswGraph,
            vector_count: 3,
            vector_bytes: vec![1, 2, 3, 4, 5],
            scalars: vec![
                SnapshotScalarRow {
                    id: 1,
                    columns: vec![
                        ("color".into(), SnapshotScalar::Text("red".into())),
                        ("n".into(), SnapshotScalar::I64(-7)),
                    ],
                },
                SnapshotScalarRow {
                    id: 1,
                    columns: vec![("color".into(), SnapshotScalar::JsonText("null".into()))],
                },
            ],
            tombstones: vec![9, 10],
        }
    }

    #[test]
    fn payload_roundtrip_all_scalars_and_duplicate_ids() {
        let payload = CollectionSnapshotPayload {
            scalars: vec![SnapshotScalarRow {
                id: 5,
                columns: vec![
                    ("a".into(), SnapshotScalar::F64(1.5)),
                    ("b".into(), SnapshotScalar::Bool(true)),
                    ("c".into(), SnapshotScalar::Timestamp(123)),
                    ("d".into(), SnapshotScalar::JsonText("{\"x\":1}".into())),
                ],
            }],
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let decoded = decode_collection_snapshot_payload(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn payload_rejects_corruption_and_trailing_bytes() {
        let payload = sample_payload();
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let mut bad = bytes.clone();
        bad[0] ^= 0xFF;
        assert!(decode_collection_snapshot_payload(&bad).is_err());
        let mut bad = bytes.clone();
        bad.truncate(bytes.len() - 1);
        assert!(decode_collection_snapshot_payload(&bad).is_err());
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(decode_collection_snapshot_payload(&bad).is_err());
    }

    #[test]
    fn encode_rejects_invalid_dims_and_ha_level() {
        let mut payload = sample_payload();
        payload.dims = 0;
        assert!(encode_collection_snapshot_payload(&payload).is_err());
        let mut payload = sample_payload();
        payload.ha_level = 99;
        assert!(encode_collection_snapshot_payload(&payload).is_err());
    }

    /// 手工构造 v1 payload（无 metric/HNSW 字段）：锁定旧快照仍可解码且
    /// 度量/HNSW 为 None，由恢复方按 index_type 推断。
    /// EN: Hand-build a v1 payload (no metric/HNSW fields): legacy snapshots
    /// stay decodable with metric/HNSW None for the caller to infer.
    #[test]
    fn v1_payload_decode_is_backward_compatible() {
        let mut out = Vec::new();
        out.extend_from_slice(COLLECTION_SNAPSHOT_MAGIC);
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(&42u64.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(b"docs");
        out.extend_from_slice(&4u64.to_be_bytes());
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(b"hnsw");
        out.push(2);
        out.push(SnapshotVectorKind::HnswGraph as u8);
        out.extend_from_slice(&0u64.to_be_bytes());
        out.extend_from_slice(&0u64.to_be_bytes());
        out.extend_from_slice(&0u64.to_be_bytes()); // scalars
        out.extend_from_slice(&0u64.to_be_bytes()); // tombstones
        let decoded = decode_collection_snapshot_payload(&out).unwrap();
        assert_eq!(decoded.metric, None);
        assert_eq!(decoded.hnsw, None);
        assert_eq!(decoded.name, "docs");
        assert_eq!(decoded.index_type, "hnsw");
    }

    /// v2 编码必须携带度量；HNSW 参数只允许 hnsw 后端。
    /// EN: v2 encoding requires a metric; HNSW params require the hnsw backend.
    #[test]
    fn v2_encode_requires_metric_and_hnsw_backend() {
        let mut payload = sample_payload();
        payload.metric = None;
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        let mut payload = sample_payload();
        payload.metric = Some(9);
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        let mut payload = sample_payload();
        payload.index_type = "flat".into();
        assert!(encode_collection_snapshot_payload(&payload).is_err());
    }

    #[test]
    fn raw_vector_length_is_validated() {
        let mut payload = sample_payload();
        payload.vector_kind = SnapshotVectorKind::RawF32;
        payload.vector_count = 1;
        payload.vector_bytes = vec![0; 8 + 4 * 4 + 1];
        assert!(encode_collection_snapshot_payload(&payload).is_err());
    }

    #[test]
    fn manifest_roundtrip_and_safe_checkpoint_validation() {
        let manifest = CollectionManifest {
            format_version: 1,
            manifest_seq: 3,
            created_at_unix: 99,
            global_barrier_lsn: 50,
            safe_checkpoint_lsn: 40,
            collections: vec![
                CollectionManifestEntry {
                    name: "a".into(),
                    checkpoint_lsn: 40,
                    dims: 4,
                    index_type: "hnsw".into(),
                    ha_level: 0,
                    metric: Some(1),
                    vector_kind: 1,
                    file: snapshot_file_name("a", 40, 1, 1),
                    file_len: 10,
                    payload_crc32: 7,
                    vector_count: 1,
                    scalar_rows: 1,
                },
                CollectionManifestEntry {
                    name: "b".into(),
                    checkpoint_lsn: 45,
                    dims: 4,
                    index_type: "flat".into(),
                    ha_level: 1,
                    metric: Some(0),
                    vector_kind: 2,
                    file: snapshot_file_name("b", 45, 1, 2),
                    file_len: 11,
                    payload_crc32: 8,
                    vector_count: 2,
                    scalar_rows: 2,
                },
            ],
        };
        let bytes = encode_collection_manifest(&manifest).unwrap();
        assert_eq!(decode_collection_manifest(&bytes).unwrap(), manifest);

        let mut wrong = manifest.clone();
        wrong.safe_checkpoint_lsn = 44;
        assert!(encode_collection_manifest(&wrong).is_err());

        let mut pathlike = manifest;
        pathlike.collections[0].file = "../x.bin".into();
        assert!(encode_collection_manifest(&pathlike).is_err());
    }

    #[test]
    fn manifest_rejects_duplicate_names_and_checkpoint_above_barrier() {
        let base = CollectionManifestEntry {
            name: "a".into(),
            checkpoint_lsn: 40,
            dims: 4,
            index_type: "hnsw".into(),
            ha_level: 0,
            metric: Some(1),
            vector_kind: 1,
            file: snapshot_file_name("a", 40, 1, 1),
            file_len: 10,
            payload_crc32: 7,
            vector_count: 1,
            scalar_rows: 1,
        };
        let manifest = CollectionManifest {
            format_version: 1,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: 50,
            safe_checkpoint_lsn: 40,
            collections: vec![base.clone(), base.clone()],
        };
        assert!(
            encode_collection_manifest(&manifest).is_err(),
            "duplicate collection names must be rejected"
        );

        let mut above = manifest;
        above.collections = vec![base];
        above.collections[0].checkpoint_lsn = 60;
        above.safe_checkpoint_lsn = 60;
        assert!(
            encode_collection_manifest(&above).is_err(),
            "checkpoint beyond the barrier must be rejected"
        );
    }

    #[test]
    fn empty_manifest_safe_checkpoint_equals_barrier() {
        // 空名单（全部 collection 已删除）：safe 必须等于 barrier 才可截断 WAL。
        // EN: empty list (everything dropped): safe must equal the barrier so
        // WAL truncation is allowed.
        let manifest = CollectionManifest {
            format_version: 1,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: 7,
            safe_checkpoint_lsn: 7,
            collections: Vec::new(),
        };
        let bytes = encode_collection_manifest(&manifest).unwrap();
        assert_eq!(decode_collection_manifest(&bytes).unwrap(), manifest);

        let mut wrong = manifest;
        wrong.safe_checkpoint_lsn = 0;
        assert!(
            encode_collection_manifest(&wrong).is_err(),
            "empty manifest safe checkpoint must equal the barrier"
        );
    }

    #[test]
    fn manifest_crc_mismatch_is_rejected() {
        // barrier=0 与空名单 safe=0 合法；随后破坏 CRC 才能进入解码路径。
        // EN: barrier=0 with an empty list and safe=0 is valid; corrupt the CRC
        // afterwards so decoding is actually exercised.
        let manifest = CollectionManifest {
            format_version: 1,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: 0,
            safe_checkpoint_lsn: 0,
            collections: Vec::new(),
        };
        let mut bytes = encode_collection_manifest(&manifest).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(decode_collection_manifest(&bytes).is_err());
    }

    #[test]
    fn snapshot_filename_is_path_safe_and_unique() {
        let a = snapshot_file_name("../evil", 1, 2, 3);
        assert!(!a.contains('/'), "slashes must be encoded: {a}");
        let b = snapshot_file_name("A", 1, 2, 3);
        let c = snapshot_file_name("a", 1, 2, 3);
        assert_ne!(b, c, "hash must disambiguate case");
        assert!(is_bare_file_name(&a));
        assert!(is_bare_file_name(&b));
    }
}
