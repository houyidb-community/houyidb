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
/// v2 增加 collection 的度量字节与 HNSW 参数；v3 在 tombstones 之后追加稀疏
/// （sparse=bm25）倒排索引段（flags bit3 与段严格一一对应）。v1/v2 仍可解码；
/// 旧二进制读到更高版本会干净拒绝，降级前必须确认没有新快照。
/// EN: v2 adds the collection metric byte and HNSW parameters; v3 appends the
/// sparse (bm25) inverted-index section after tombstones (flags bit3 maps 1:1
/// to that section). v1/v2 remain decodable; older binaries cleanly reject a
/// higher version, so downgrades must not have v3 snapshots.
pub const COLLECTION_SNAPSHOT_VERSION: u32 = 3;
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
    /// PQ 构建态快照（仅 pq-flat）：不透明白状态字节（由 houyi-vector 编解
    /// 码），外框为 `[state_len u64][state][pending_count u64][pending rows]`；
    /// pending 行与 `RawF32` 同布局。旧二进制对未知 kind 干净拒绝。
    /// EN: PQ built-state snapshot (pq-flat only): opaque state bytes
    /// (encoded/decoded by houyi-vector) framed as
    /// `[state_len u64][state][pending_count u64][pending rows]`; pending rows
    /// share the `RawF32` layout. Older binaries cleanly reject the unknown kind.
    PqFlatState = 3,
    /// 缓冲的原始 f32 行（clustered / flat-clustered / pq-flat 未构建态或
    /// 待建行）：与 `RawF32` 相同行布局，但恢复时只装入写缓冲、不触发构建。
    /// 旧二进制对未知 kind 干净拒绝。
    /// EN: Buffered raw f32 rows (clustered / flat-clustered / pq-flat unbuilt
    /// or pending states): same row layout as `RawF32`, but restore loads them
    /// into the write buffer without triggering a build. Older binaries cleanly
    /// reject the unknown kind.
    BufferedRawF32 = 4,
}

impl SnapshotVectorKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Empty),
            1 => Some(Self::HnswGraph),
            2 => Some(Self::RawF32),
            3 => Some(Self::PqFlatState),
            4 => Some(Self::BufferedRawF32),
            _ => None,
        }
    }
}

/// 存储向量编码（payload flags bit0）。只影响 hnsw 后端的图内存储；`RawF32`
/// 行始终是 f32，因此显式 f16 不允许与 `RawF32` 组合。
/// EN: Stored vector encoding (payload flags bit0). It only affects the hnsw
/// backend's graph storage; `RawF32` rows are always f32, so f16 with `RawF32`
/// is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotVectorDtype {
    F32 = 0,
    F16 = 1,
}

impl SnapshotVectorDtype {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            _ => None,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// 图侧量化方式（payload flags bit1/bit2）。只影响 hnsw 后端的图内距离计算；
/// 主表 / Flat 行始终保存原始编码，因此量化不允许与 `RawF32` 行组合。
/// RaBitQ（1 bit/dim）额外要求度量仅 l2/cosine。
/// EN: Graph-side quantization (payload flags bit1/bit2). It only affects the
/// hnsw backend's in-graph distance computation; main-store/Flat rows always
/// keep raw encoding, so quantization with `RawF32` rows is rejected. RaBitQ
/// (1 bit/dim) additionally requires the metric to be l2/cosine only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotQuantization {
    None = 0,
    Int8 = 1,
    RaBitQ = 2,
}

impl SnapshotQuantization {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// 诊断可读名：错误信息必须出现 `int8`/`rabitq` 而非裸判别值。
    /// EN: Human-readable name for diagnostics: errors must say `int8`/`rabitq`
    /// rather than a bare discriminant.
    pub fn as_str(self) -> &'static str {
        match self {
            SnapshotQuantization::None => "none",
            SnapshotQuantization::Int8 => "int8",
            SnapshotQuantization::RaBitQ => "rabitq",
        }
    }

    /// 由快照字节解码；未知字节返回 None（调用方 fail-fast）。
    /// EN: Decode from the snapshot byte; unknown bytes return None (callers
    /// fail fast).
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(SnapshotQuantization::None),
            1 => Some(SnapshotQuantization::Int8),
            2 => Some(SnapshotQuantization::RaBitQ),
            _ => None,
        }
    }
}

/// payload `flags` 的 bit0：图向量按 f16 存储。旧二进制要求 flags == 0，
/// 因此 f16 快照对旧版本 fail-fast（不会误读成 f32），而 f32 快照仍可被旧
/// 版本读取（无需版本号 bump）。
/// EN: payload `flags` bit0: graph vectors stored as f16. Old binaries require
/// flags == 0, so f16 snapshots fail fast there (never misread as f32) while
/// f32 snapshots remain readable — no version bump needed.
pub const COLLECTION_SNAPSHOT_FLAG_VECTOR_F16: u32 = 0x01;

/// payload `flags` 的 bit1：图侧向量按 int8 量化。与 bit0（f16 主表编码）
/// 正交，可同时置位。旧二进制只接受 flags == 0，因此 int8 快照对旧版本
/// fail-fast（不会把 int8 图误读成原向量）。
/// EN: payload `flags` bit1: graph-side vectors are int8-quantized. Orthogonal
/// to bit0 (f16 main-store encoding) and both may be set. Old binaries accept
/// only flags == 0, so int8 snapshots fail fast there (never misread the int8
/// graph as raw vectors).
pub const COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8: u32 = 0x02;

/// payload `flags` 的 bit2：图侧向量按 RaBitQ（1 bit/dim）量化。与 bit1（int8）
/// 互斥、与 bit0（f16 主表编码）正交。旧二进制只接受 flags 的 bit0–bit1，
/// 因此 RaBitQ 快照对旧版本 fail-fast（不会把 1 bit 码误读成 int8/原向量）。
/// EN: payload `flags` bit2: graph-side vectors are RaBitQ-quantized (1 bit/dim).
/// Mutually exclusive with bit1 (int8) and orthogonal to bit0 (f16 main-store
/// encoding). Old binaries accept only bits 0–1, so RaBitQ snapshots fail fast
/// there (never misread the 1-bit codes as int8/raw vectors).
pub const COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ: u32 = 0x04;

/// payload `flags` 的 bit3：payload 在 tombstones 之后携带稀疏 BM25 倒排索引段
/// （v3 新增）。与量化位正交；旧二进制只支持版本 ≤ 2 / bit0–bit2，因此 v3
/// 快照对旧版本 fail-fast（不会静默丢掉稀疏索引后按 dense-only 恢复）。
/// EN: payload `flags` bit3: the payload carries the sparse BM25 inverted-index
/// section after tombstones (added in v3). Orthogonal to the quantization bits;
/// old binaries support only versions ≤ 2 / bits 0–2, so v3 snapshots fail
/// fast there (the sparse index is never silently dropped and restored
/// dense-only).
pub const COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25: u32 = 0x08;

/// 稀疏词项上限（与 houyi-vector 的 `MAX_SPARSE_TERM_ID` 保持一致；persistence
/// 不反向依赖 vector crate）。
/// EN: Sparse term cap (kept in sync with `MAX_SPARSE_TERM_ID` in houyi-vector;
/// persistence must not depend on the vector crate).
const MAX_SPARSE_TERM_ID: u32 = (1 << 24) - 1;

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

/// 稀疏 BM25 旁路索引的持久视图（v3+；仅 `sparse=bm25` 的 hnsw/flat 集合）。
/// `terms` 按 term_id 严格升序，term 内 postings 按文档 id 严格升序。
/// `doc_count` 是全局去重文档数；`total_len` 是全局权重和的持久值——解码只
/// 校验有限、≥ 0，且文档非空时 > 0，不重算（避免浮点求和顺序误差误拒合法
/// 快照）。posting 命中墓碑文档是合法状态（裸向量 + 稀疏复活同一 id），
/// 编码器与解码器均不拒绝。
/// EN: Persistent view of the sparse BM25 side index (v3+; only hnsw/flat
/// collections with `sparse=bm25`). `terms` ascend strictly by term_id and
/// postings ascend strictly by document id within each term. `doc_count` is the
/// globally distinct document count; `total_len` is the persisted global weight
/// sum — decode only checks finite, ≥ 0, and > 0 when documents exist, never
/// recomputing (avoiding float-order false rejections). Postings on tombstoned
/// documents are legitimate (bare-vector + sparse resurrection of the same id);
/// neither encoder nor decoder rejects them.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotSparseIndex {
    pub doc_count: u64,
    pub total_len: f64,
    pub terms: Vec<(u32, Vec<(u64, f32)>)>,
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
    /// 存储向量编码；None = v1/v2 旧快照（f32）。编码进 payload flags bit0。
    /// EN: Stored vector encoding; None = legacy v1/v2 snapshot (f32). Encoded
    /// in payload flags bit0.
    pub vector_dtype: Option<SnapshotVectorDtype>,
    /// 图侧量化；None = v1 旧快照（不量化），Some(None) = v2 未量化集合。
    /// 编码进 payload flags bit1（int8）/ bit2（RaBitQ），不新增 body 字节。
    /// 两者互斥；RaBitQ 额外要求度量为 l2/cosine。
    /// EN: Graph-side quantization; None = legacy v1 snapshot (none), Some(None)
    /// = v2 collection without quantization. Encoded in payload flags bit1
    /// (int8) / bit2 (RaBitQ) with no additional body bytes. The two bits are
    /// mutually exclusive; RaBitQ additionally requires the l2/cosine metric.
    pub quantization: Option<SnapshotQuantization>,
    pub vector_kind: SnapshotVectorKind,
    pub vector_count: u64,
    pub vector_bytes: Vec<u8>,
    pub scalars: Vec<SnapshotScalarRow>,
    pub tombstones: Vec<u64>,
    /// 稀疏 BM25 索引；None = sparse=none/旧快照。编码进 payload flags bit3，
    /// 且段内联在 tombstones 之后（v3）。
    /// EN: Sparse BM25 index; None = sparse=none/legacy snapshot. Encoded in
    /// payload flags bit3 with the section appended after tombstones (v3).
    pub sparse: Option<SnapshotSparseIndex>,
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
    /// 存储编码字节（0=f32/1=f16；新 manifest 写入，旧 manifest 缺省按 f32）。
    /// EN: Storage-encoding byte (0=f32/1=f16; written by new manifests, absent
    /// in legacy manifests where f32 is implied).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_dtype: Option<u8>,
    /// 图侧量化字节（int8 写 `Some(1)`、RaBitQ 写 `Some(2)`；缺省 = 未量化/旧
    /// manifest）。int8 要求 `format_version` ≥ 2，RaBitQ 要求 ≥ 3。
    /// EN: Graph-side quantization byte (int8 writes `Some(1)`, RaBitQ writes
    /// `Some(2)`; absent = none/legacy manifest). int8 requires
    /// `format_version` ≥ 2 and RaBitQ requires ≥ 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<u8>,
    /// 稀疏索引字节（bm25 写 `Some(1)`；缺省 = sparse=none/旧 manifest）。
    /// bm25 只允许 hnsw/flat，且要求 `format_version` = 4（条目存在即精确
    /// 推导为 4，低版本二进制不认识该字段会静默丢掉稀疏索引）。
    /// EN: Sparse-index byte (bm25 writes `Some(1)`; absent = sparse=none/legacy
    /// manifest). bm25 is allowed only on hnsw/flat and requires
    /// `format_version` = 4 (any such entry makes the exact derived version 4;
    /// older binaries ignore the field and silently drop the sparse index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<u8>,
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
    // 显式 f16 只属于 hnsw 后端，且不可能与 RawF32 行（固定 f32 编码）组合：
    // 任一违反都意味着调用方把 dtype 挂到了没有 dtype 存储路径的形态上。
    // EN: Explicit f16 belongs only to the hnsw backend and cannot combine with
    // RawF32 rows (fixed f32 encoding): either violation means the caller attached
    // the dtype to a shape with no dtype storage path.
    if payload.vector_dtype == Some(SnapshotVectorDtype::F16) {
        if payload.index_type != "hnsw" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "collection snapshot: vector_dtype f16 requires index_type 'hnsw'",
            ));
        }
        if matches!(
            payload.vector_kind,
            SnapshotVectorKind::RawF32 | SnapshotVectorKind::BufferedRawF32
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "collection snapshot: vector_dtype f16 cannot pair with {:?} rows",
                    payload.vector_kind
                ),
            ));
        }
    }
    // 量化只属于 hnsw 图存储：非 hnsw 后端与其他向量形态都没有量化解码路径，
    // 携带它就是写入侧 bug（fail-fast）。RaBitQ 另要求度量仅 l2(0)/cosine(1)。
    // EN: Quantization belongs solely to hnsw graph storage: non-hnsw backends
    // and other vector shapes have no quantized decode path, so carrying it is a
    // writer bug (fail fast). RaBitQ additionally requires metric l2(0)/cosine(1).
    if let Some(quantization) = payload.quantization {
        if quantization != SnapshotQuantization::None {
            if payload.index_type != "hnsw" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "collection snapshot: {} quantization requires index_type 'hnsw'",
                        quantization.as_str()
                    ),
                ));
            }
            if payload.vector_kind != SnapshotVectorKind::HnswGraph {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "collection snapshot: {} quantization requires HNSW_GRAPH vectors",
                        quantization.as_str()
                    ),
                ));
            }
            if quantization == SnapshotQuantization::RaBitQ && metric > 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "collection snapshot: rabitq quantization requires metric l2|cosine",
                ));
            }
        }
    }
    // 稀疏索引只属于 sparse=bm25 的 hnsw/flat 集合；语义校验（term/doc 升序、
    // 权重有限 > 0、doc_count 对账、total_len 自洽）与解码侧共用同一实现。
    // EN: The sparse index belongs only to sparse=bm25 hnsw/flat collections; the
    // semantic checks (term/doc ordering, finite > 0 weights, doc_count
    // reconciliation, total_len consistency) share one implementation with the
    // decoder.
    if let Some(sparse) = &payload.sparse {
        validate_snapshot_sparse(sparse, &payload.index_type)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
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
        SnapshotVectorKind::RawF32 | SnapshotVectorKind::BufferedRawF32 => {
            // kind 4 只属于懒构建后端（clustered/flat-clustered/pq-flat）：
            // 其他后端没有「先入缓冲、按需构建」的恢复路径。
            // EN: Kind 4 belongs only to lazy backends
            // (clustered/flat-clustered/pq-flat): other backends have no
            // "buffer now, build later" restore path.
            if payload.vector_kind == SnapshotVectorKind::BufferedRawF32
                && !is_lazy_backend(&payload.index_type)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "collection snapshot: BUFFERED_RAW_F32 requires index_type \
                         clustered|flat-clustered|pq-flat, got {:?}",
                        payload.index_type
                    ),
                ));
            }
            let expected = raw_rows_len(payload.dims, payload.vector_count)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            if expected != payload.vector_bytes.len() as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "collection snapshot: {:?} bytes {} != count×per {}",
                        payload.vector_kind,
                        payload.vector_bytes.len(),
                        expected
                    ),
                ));
            }
        }
        SnapshotVectorKind::PqFlatState => {
            if payload.index_type != "pq-flat" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "collection snapshot: PQ_FLAT_STATE requires index_type 'pq-flat', got {:?}",
                        payload.index_type
                    ),
                ));
            }
            if payload.vector_count == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "collection snapshot: PQ_FLAT_STATE requires count > 0",
                ));
            }
            validate_pq_flat_state_frame(&payload.vector_bytes, payload.dims, payload.vector_count)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        }
        SnapshotVectorKind::HnswGraph => {}
    }

    // 版本精确推导：携带稀疏段写 v3，否则沿用 v2（旧二进制对 v3 干净拒绝）。
    // EN: Exact version derivation: a sparse section writes v3, otherwise v2
    // (older binaries cleanly reject v3).
    let version: u32 = if payload.sparse.is_some() { 3 } else { 2 };
    let mut out = Vec::new();
    out.extend_from_slice(COLLECTION_SNAPSHOT_MAGIC);
    out.extend_from_slice(&version.to_be_bytes());
    out.extend_from_slice(&payload.checkpoint_lsn.to_be_bytes());
    let mut flags = match payload.vector_dtype {
        Some(SnapshotVectorDtype::F16) => COLLECTION_SNAPSHOT_FLAG_VECTOR_F16,
        _ => 0,
    };
    match payload.quantization {
        Some(SnapshotQuantization::Int8) => flags |= COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8,
        Some(SnapshotQuantization::RaBitQ) => flags |= COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ,
        Some(SnapshotQuantization::None) | None => {}
    }
    if payload.sparse.is_some() {
        flags |= COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25;
    }
    out.extend_from_slice(&flags.to_be_bytes());
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
    // 稀疏段（v3，位置式追加在 tombstones 之后）：
    // [doc_count u64][total_len f64][term_count u32]
    // 每 term：[term_id u32][posting_count u32][(doc u64, tf f32 BE)×posting_count]。
    // EN: Sparse section (v3, appended positionally after tombstones):
    // [doc_count u64][total_len f64][term_count u32]; per term: [term_id u32]
    // [posting_count u32][(doc u64, tf f32 BE) × posting_count].
    if let Some(sparse) = &payload.sparse {
        out.extend_from_slice(&sparse.doc_count.to_be_bytes());
        out.extend_from_slice(&sparse.total_len.to_bits().to_be_bytes());
        out.extend_from_slice(&(sparse.terms.len() as u32).to_be_bytes());
        for (term, postings) in &sparse.terms {
            out.extend_from_slice(&term.to_be_bytes());
            out.extend_from_slice(&(postings.len() as u32).to_be_bytes());
            for (doc, tf) in postings {
                out.extend_from_slice(&doc.to_be_bytes());
                out.extend_from_slice(&tf.to_bits().to_be_bytes());
            }
        }
    }
    Ok(out)
}

/// 校验稀疏段语义（编码/解码共用）：term 域与升序、posting 文档升序、权重
/// 有限 > 0、doc_count 与全局去重文档数对账、total_len 有限且与文档状态
/// 自洽（有文档时必须 > 0）、仅 hnsw/flat。
///
/// 这里**不**校验 posting 文档是否在标量墓碑集合中：墓碑只隐藏标量/载荷
/// 行，向量与稀疏数据跟随 upsert 复活（裸向量 + 稀疏重建同一 id 是合法
/// 写入路径）。若强制「posting 不得命中墓碑」，编码器会拒绝自己刚写下的
/// 合法状态。标量墓碑的「隐藏行」语义由读取侧的 `CollectionScalars` 在访问
/// 时解释；payload 携带完整物理页（含 stale 行与墓碑），本函数不做行级交叉
/// 校验，也不影响稀疏 postings。
///
/// EN: Validate sparse-section semantics (shared by encoder and decoder): term
/// domain/ordering, ascending posting docs, finite > 0 weights, doc_count
/// reconciled against globally distinct docs, total_len finite and positive
/// when documents exist, hnsw/flat only.
///
/// Posting docs are deliberately **not** checked against scalar tombstones:
/// tombstones only hide scalar/payload rows, while vectors and sparse data
/// follow upsert resurrection (a bare-vector + sparse rewrite of the same id is
/// legitimate). Enforcing "no posting may hit a tombstone" would make the
/// encoder reject a valid state it just wrote. The "hide rows" semantics of
/// scalar tombstones are interpreted at read time by `CollectionScalars`; the
/// payload carries full physical pages (stale rows plus tombstones), so this
/// function performs no row-level cross-check and says nothing about sparse
/// postings.
fn validate_snapshot_sparse(sparse: &SnapshotSparseIndex, index_type: &str) -> Result<(), String> {
    if index_type != "hnsw" && index_type != "flat" {
        return Err(format!(
            "collection snapshot: sparse bm25 requires index_type hnsw|flat, got {index_type:?}"
        ));
    }
    if !sparse.total_len.is_finite() || sparse.total_len < 0.0 {
        return Err("collection snapshot: sparse total_len must be finite and >= 0".to_string());
    }
    if sparse.doc_count > 0 && sparse.total_len == 0.0 {
        return Err(
            "collection snapshot: sparse total_len must be > 0 when doc_count > 0".to_string(),
        );
    }
    let mut prev_term: Option<u32> = None;
    // 去重文档集合同时用于 doc_count 对账（同一文档可出现在多个 term）。
    // EN: The distinct-doc set also reconciles doc_count (one doc may appear
    // under several terms).
    let mut docs: HashSet<u64> = HashSet::new();
    for (term, postings) in &sparse.terms {
        if *term == 0 || *term > MAX_SPARSE_TERM_ID {
            return Err(format!(
                "collection snapshot: sparse term {term} out of range 1..={MAX_SPARSE_TERM_ID}"
            ));
        }
        if let Some(prev) = prev_term {
            if *term <= prev {
                return Err(
                    "collection snapshot: sparse terms must be strictly ascending".to_string(),
                );
            }
        }
        prev_term = Some(*term);
        let mut prev_doc: Option<u64> = None;
        for (doc, tf) in postings {
            if !tf.is_finite() || *tf <= 0.0 {
                return Err(format!(
                    "collection snapshot: sparse weight for doc {doc} must be finite and > 0"
                ));
            }
            if let Some(prev) = prev_doc {
                if *doc <= prev {
                    return Err(format!(
                        "collection snapshot: sparse postings for term {term} must be strictly \
                         ascending by doc"
                    ));
                }
            }
            prev_doc = Some(*doc);
            docs.insert(*doc);
        }
    }
    if docs.len() as u64 != sparse.doc_count {
        return Err(format!(
            "collection snapshot: sparse doc_count {} != distinct postings {}",
            sparse.doc_count,
            docs.len()
        ));
    }
    Ok(())
}

/// 懒构建后端白名单（kind 4 的合法载体）。新增懒后端时必须同步扩展本名单，
/// 否则其快照编码会在此 fail-closed（拒绝整轮 manifest），不会静默走错路径。
/// EN: Lazy-backend allowlist (legal carriers for kind 4). Adding a lazy backend
/// requires extending this list; otherwise its snapshot encoding fails closed
/// here (rejecting the whole manifest) instead of silently taking a wrong path.
fn is_lazy_backend(index_type: &str) -> bool {
    matches!(index_type, "clustered" | "flat-clustered" | "pq-flat")
}

/// Raw 行布局期望字节数：count × (8 + dims × 4)。
/// EN: Expected raw-row byte length: count × (8 + dims × 4).
fn raw_rows_len(dims: u64, count: u64) -> Result<u64, String> {
    let per = 8u64
        .checked_add(
            dims.checked_mul(4)
                .ok_or_else(|| "dims overflow".to_string())?,
        )
        .ok_or_else(|| "dims overflow".to_string())?;
    count
        .checked_mul(per)
        .ok_or_else(|| "vector size overflow".to_string())
}

/// 校验 pq-flat 紧凑态外框（编码/解码共用）：`state_len > 0`、
/// `pending_count < vector_count`（kind 3 至少携带一条已建向量，
/// pending 等于 count 时语义与 kind 4 重复）、总长精确等于
/// `8 + state_len + 8 + pending_count × row_len`。
/// 状态字节本身对本层不透明（由 houyi-vector 解码时再校验）。
/// EN: Validate the pq-flat compact-state frame (shared by encoder and decoder):
/// `state_len > 0`, `pending_count < vector_count` (kind 3 always carries at
/// least one built vector; pending == count would duplicate kind-4 semantics),
/// and total length exactly `8 + state_len + 8 + pending_count × row_len`.
/// The state bytes stay opaque here and are validated by houyi-vector when decoded.
fn validate_pq_flat_state_frame(bytes: &[u8], dims: u64, vector_count: u64) -> Result<(), String> {
    if bytes.len() < 16 {
        return Err("collection snapshot: PQ_FLAT_STATE frame truncated".to_string());
    }
    let state_len = u64::from_be_bytes(
        bytes[0..8]
            .try_into()
            .map_err(|_| "collection snapshot: PQ_FLAT_STATE header".to_string())?,
    );
    if state_len == 0 {
        return Err("collection snapshot: PQ_FLAT_STATE empty state".to_string());
    }
    let pending_off = 8u64
        .checked_add(state_len)
        .ok_or_else(|| "collection snapshot: PQ_FLAT_STATE state_len overflow".to_string())?;
    let pending_off = usize::try_from(pending_off)
        .map_err(|_| "collection snapshot: PQ_FLAT_STATE state_len too large".to_string())?;
    let pending_end = pending_off
        .checked_add(8)
        .ok_or_else(|| "collection snapshot: PQ_FLAT_STATE length overflow".to_string())?;
    if bytes.len() < pending_end {
        return Err("collection snapshot: PQ_FLAT_STATE frame truncated".to_string());
    }
    let pending_count = u64::from_be_bytes(
        bytes[pending_off..pending_end]
            .try_into()
            .map_err(|_| "collection snapshot: PQ_FLAT_STATE header".to_string())?,
    );
    if pending_count >= vector_count {
        return Err(format!(
            "collection snapshot: PQ_FLAT_STATE pending {pending_count} must be < count \
             {vector_count} (at least one built vector)"
        ));
    }
    let expected = (pending_end as u64)
        .checked_add(raw_rows_len(dims, pending_count)?)
        .ok_or_else(|| "collection snapshot: PQ_FLAT_STATE length overflow".to_string())?;
    if expected != bytes.len() as u64 {
        return Err(format!(
            "collection snapshot: PQ_FLAT_STATE bytes {} != expected {expected}",
            bytes.len()
        ));
    }
    Ok(())
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
    if flags
        & !(COLLECTION_SNAPSHOT_FLAG_VECTOR_F16
            | COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8
            | COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ
            | COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25)
        != 0
    {
        return Err(corrupt(
            "collection snapshot: unsupported flags (only bits 0-3 may be set)",
        ));
    }
    // int8 与 RaBitQ 位互斥：同时置位说明写入方/读取方语义不一致（无合法形态）。
    // EN: The int8 and RaBitQ bits are mutually exclusive; both set means writer
    // and reader disagree (no legal shape exists).
    if flags & COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8 != 0
        && flags & COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ != 0
    {
        return Err(corrupt(
            "collection snapshot: int8 and rabitq quantization bits are mutually exclusive",
        ));
    }
    // v1 快照没有 dtype 位；携带 flags 说明写入方/读取方版本语义不一致，
    // 静默忽略会让 f16 数据被当作 f32 解读成损坏快照，必须 fail-fast。
    // EN: v1 snapshots have no dtype bit; any flags set means writer and reader
    // version semantics disagree. Silently ignoring them would let f16 data be
    // read as f32 (a corrupt snapshot), so fail fast instead.
    if version < 2 && flags != 0 {
        return Err(corrupt(
            "collection snapshot: v1 payload must not carry flags",
        ));
    }
    // 稀疏位与版本严格一一对应：v2 及更早不存在该段（置位说明读取方语义
    // 不一致），v3 必须携带该段（缺位会让稀疏索引被静默丢弃）。
    // EN: The sparse bit maps 1:1 to the version: v2 and earlier have no such
    // section (a set bit means reader/writer disagreement), and v3 must carry
    // it (a clear bit would silently drop the sparse index).
    if version < 3 && flags & COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25 != 0 {
        return Err(corrupt(
            "collection snapshot: sparse bit requires version >= 3",
        ));
    }
    if version >= 3 && flags & COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25 == 0 {
        return Err(corrupt(
            "collection snapshot: v3 payload requires the sparse bit",
        ));
    }
    // v2 起的 payload 按 flags bit0 解码 dtype：bit 未置位即 f32（包括旧服务端
    // 写的 v2 快照）。v1 不携带 dtype，保持 None。
    // EN: v2 payloads decode dtype from flags bit0: a clear bit means f32
    // (including v2 snapshots written by older servers). v1 carries no dtype.
    let vector_dtype = if version >= 2 {
        Some(if flags & COLLECTION_SNAPSHOT_FLAG_VECTOR_F16 != 0 {
            SnapshotVectorDtype::F16
        } else {
            SnapshotVectorDtype::F32
        })
    } else {
        None
    };
    // v2 起的 payload 按 flags bit1/bit2 解码量化：两位都未置位即未量化（包括
    // 旧服务端写的 v2 快照）；两位同时置位已在上方拒绝。v1 不携带量化，保持 None。
    // EN: v2 payloads decode quantization from flags bit1/bit2: neither set means
    // none (including v2 snapshots written by older servers); both set was
    // rejected above. v1 carries no quantization and keeps None.
    let quantization = if version >= 2 {
        Some(if flags & COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ != 0 {
            SnapshotQuantization::RaBitQ
        } else if flags & COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8 != 0 {
            SnapshotQuantization::Int8
        } else {
            SnapshotQuantization::None
        })
    } else {
        None
    };
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
    // f16 只属于 hnsw 图存储：非 hnsw 后端或 RawF32 行都没有 f16 解码路径，
    // 携带它就是损坏/写入侧 bug（fail-fast，避免按错误编码还原向量）。
    // EN: f16 belongs solely to hnsw graph storage — non-hnsw backends and
    // RawF32 rows have no f16 decode path, so carrying it means corruption or a
    // writer bug (fail fast instead of restoring vectors under the wrong encoding).
    if vector_dtype == Some(SnapshotVectorDtype::F16) {
        if index_type != "hnsw" {
            return Err(corrupt(
                "collection snapshot: vector_dtype f16 requires index_type 'hnsw'",
            ));
        }
        if matches!(
            vector_kind,
            SnapshotVectorKind::RawF32 | SnapshotVectorKind::BufferedRawF32
        ) {
            return Err(corrupt(format!(
                "collection snapshot: vector_dtype f16 cannot pair with {vector_kind:?} rows"
            )));
        }
    }
    // 量化只属于 hnsw 图存储：非 hnsw 后端或其他向量形态没有量化解码路径，
    // 携带它就是损坏/写入侧 bug（fail-fast）。RaBitQ 另要求度量仅 l2/cosine。
    // EN: Quantization belongs solely to hnsw graph storage: non-hnsw backends
    // and other vector shapes have no quantized decode path, so carrying it means
    // corruption or a writer bug (fail fast). RaBitQ additionally requires the
    // l2/cosine metric.
    if let Some(quantization) = quantization {
        if quantization != SnapshotQuantization::None {
            if index_type != "hnsw" {
                return Err(corrupt(format!(
                    "collection snapshot: {} quantization requires index_type 'hnsw'",
                    quantization.as_str()
                )));
            }
            if vector_kind != SnapshotVectorKind::HnswGraph {
                return Err(corrupt(format!(
                    "collection snapshot: {} quantization requires HNSW_GRAPH vectors",
                    quantization.as_str()
                )));
            }
            // v1 不携带度量：RaBitQ 位出现在 v1 已由上方 v1 flags 规则拒绝；
            // 走到这里 metric 必为 v2 显式值。
            // EN: v1 carries no metric, and a RaBitQ bit on v1 is already
            // rejected by the v1 flags rule above; reaching here means an
            // explicit v2 metric.
            if quantization == SnapshotQuantization::RaBitQ && matches!(metric, Some(m) if m > 1) {
                return Err(corrupt(
                    "collection snapshot: rabitq quantization requires metric l2|cosine",
                ));
            }
        }
    }
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
        SnapshotVectorKind::RawF32 | SnapshotVectorKind::BufferedRawF32 => {
            if vector_kind == SnapshotVectorKind::BufferedRawF32 && !is_lazy_backend(&index_type) {
                return Err(corrupt(format!(
                    "collection snapshot: BUFFERED_RAW_F32 requires index_type \
                     clustered|flat-clustered|pq-flat, got {index_type:?}"
                )));
            }
            let expected = raw_rows_len(dims, vector_count).map_err(corrupt)?;
            if expected != vector_bytes_len {
                return Err(corrupt(format!(
                    "collection snapshot: {vector_kind:?} bytes {vector_bytes_len} != count×per {expected}"
                )));
            }
        }
        SnapshotVectorKind::PqFlatState => {
            if index_type != "pq-flat" {
                return Err(corrupt(format!(
                    "collection snapshot: PQ_FLAT_STATE requires index_type 'pq-flat', \
                     got {index_type:?}"
                )));
            }
            if vector_count == 0 {
                return Err(corrupt(
                    "collection snapshot: PQ_FLAT_STATE requires count > 0",
                ));
            }
            validate_pq_flat_state_frame(&vector_bytes, dims, vector_count).map_err(corrupt)?;
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
    // v3 稀疏段：位置式固定跟在 tombstones 之后；长度用剩余字节做上界，
    // 防止 term/posting 计数触发放大分配。
    // EN: v3 sparse section: positionally fixed right after tombstones; counts
    // are bounded by the remaining bytes to prevent allocation amplification.
    let mut sparse: Option<SnapshotSparseIndex> = None;
    if version >= 3 {
        let doc_count = r.u64()?;
        let total_len = f64::from_bits(r.u64()?);
        let term_count = r.u32()?;
        // 每 term 至少 term_id u32 + posting_count u32。
        // EN: Each term needs at least term_id u32 + posting_count u32.
        if term_count as u64 > (r.remaining() as u64) / 8 {
            return Err(corrupt(
                "collection snapshot: sparse term count exceeds remaining bytes",
            ));
        }
        let mut terms = Vec::new();
        terms
            .try_reserve(term_count as usize)
            .map_err(|_| corrupt("collection snapshot: sparse terms allocation failed"))?;
        for _ in 0..term_count {
            let term = r.u32()?;
            let posting_count = r.u32()?;
            // 每条 posting 至少 doc u64 + tf f32。
            // EN: Each posting needs at least doc u64 + tf f32.
            if posting_count as u64 > (r.remaining() as u64) / 12 {
                return Err(corrupt(
                    "collection snapshot: sparse posting count exceeds remaining bytes",
                ));
            }
            let mut postings = Vec::new();
            postings
                .try_reserve(posting_count as usize)
                .map_err(|_| corrupt("collection snapshot: sparse postings allocation failed"))?;
            for _ in 0..posting_count {
                let doc = r.u64()?;
                let tf = f32::from_bits(r.u32()?);
                postings.push((doc, tf));
            }
            terms.push((term, postings));
        }
        let index = SnapshotSparseIndex {
            doc_count,
            total_len,
            terms,
        };
        validate_snapshot_sparse(&index, &index_type).map_err(corrupt)?;
        sparse = Some(index);
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
        vector_dtype,
        quantization,
        vector_kind,
        vector_count,
        vector_bytes,
        scalars,
        tombstones,
        sparse,
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
    // v1 = 无量化字段；v2 = 允许 entry 携带 int8 量化；v3 = 取值域加入 RaBitQ；
    // v4 = entry 可携带稀疏 bm25（写入方按 entries 精确推导：含稀疏写 4，否则
    // 含 RaBitQ 写 3、仅 int8 写 2、其余 1）。更早/更高版本均 fail-fast
    //（降级会静默丢字段）。
    // EN: v1 = no quantization field; v2 = entries may carry int8 quantization;
    // v3 = the value domain adds RaBitQ; v4 = entries may carry sparse bm25
    // (the writer derives the version exactly from entries: 4 with sparse, else
    // 3 with RaBitQ, 2 with int8 only, 1 otherwise). Older/newer versions fail
    // fast (downgrades silently drop fields).
    if !(1..=4).contains(&manifest.format_version) {
        return Err(format!(
            "collection manifest: unsupported format_version {}",
            manifest.format_version
        ));
    }
    let mut names = std::collections::HashSet::new();
    let mut min_checkpoint: Option<u64> = None;
    let mut saw_sparse = false;
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
        let vector_kind = SnapshotVectorKind::from_u8(entry.vector_kind).ok_or_else(|| {
            format!(
                "collection manifest: unknown vector_kind {}",
                entry.vector_kind
            )
        })?;
        // 懒构建后端的向量形态与 index_type 严格对应：kind 3（PQ 紧凑构建态）
        // 仅 pq-flat；kind 4（缓冲原始行）仅 clustered/flat-clustered/pq-flat。
        // 旧二进制对未知 kind 已在上面干净拒绝，无需 bump format_version。
        // EN: Lazy-backend vector kinds map strictly to index types: kind 3
        // (PQ compact built state) is pq-flat only; kind 4 (buffered raw rows)
        // is clustered/flat-clustered/pq-flat only. Older binaries already
        // cleanly reject the unknown kind above, so no format bump is needed.
        match vector_kind {
            SnapshotVectorKind::PqFlatState if entry.index_type != "pq-flat" => {
                return Err(format!(
                    "collection manifest: PQ_FLAT_STATE requires index_type 'pq-flat' for {:?}",
                    entry.name
                ));
            }
            SnapshotVectorKind::BufferedRawF32 if !is_lazy_backend(&entry.index_type) => {
                return Err(format!(
                    "collection manifest: BUFFERED_RAW_F32 requires index_type \
                     clustered|flat-clustered|pq-flat for {:?}",
                    entry.name
                ));
            }
            _ => {}
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
        // vector_dtype 域与 payload 编码一致（0=f32/1=f16），且 f16 只允许
        // hnsw 图存储；None 是旧 manifest 的合法形态（按 f32 隐含）。
        // EN: vector_dtype shares the payload domain (0=f32/1=f16) and f16 is
        // only legal on hnsw graph storage; None is the legal legacy shape
        // (f32 implied).
        if let Some(dtype) = entry.vector_dtype {
            if SnapshotVectorDtype::from_u8(dtype).is_none() {
                return Err(format!(
                    "collection manifest: unknown vector_dtype {dtype} for {:?}",
                    entry.name
                ));
            }
            if dtype == SnapshotVectorDtype::F16 as u8
                && (entry.index_type != "hnsw"
                    || entry.vector_kind != SnapshotVectorKind::HnswGraph as u8)
            {
                return Err(format!(
                    "collection manifest: f16 dtype requires hnsw graph storage for {:?}",
                    entry.name
                ));
            }
        }
        // quantization 域允许 1（int8）/2（RaBitQ），且只属于 hnsw 图存储。
        // int8 要求 format_version ≥ 2、RaBitQ ≥ 3——低版本二进制不认识该字段，
        // 静默忽略会把集合恢复成未量化（读侧因此把低版本 + quantization 判为
        // 损坏）。RaBitQ 另要求显式携带度量：缺席时无法证明 l2/cosine 约束，
        // 恢复会按历史默认 Cosine 重建，因此按损坏拒绝。
        // EN: The quantization domain allows 1 (int8)/2 (RaBitQ) and only on hnsw
        // graph storage. int8 requires format_version ≥ 2 and RaBitQ ≥ 3 — older
        // binaries ignore the field and would restore collections unquantized
        // (so low version + quantization is rejected as corrupt). RaBitQ also
        // requires an explicit metric: without one the l2/cosine constraint
        // cannot be checked and restore would fall back to the historical Cosine
        // default, so a missing metric is rejected as corrupt.
        if let Some(quantization) = entry.quantization {
            let quantization = SnapshotQuantization::from_u8(quantization).ok_or_else(|| {
                format!(
                    "collection manifest: unknown quantization {quantization} for {:?}",
                    entry.name
                )
            })?;
            // 0 = `none` 是已知变体但不是 manifest 的合法取值：未量化必须
            // 「字段缺席」，显式写 0 说明写入方/读取方语义不一致。
            // EN: 0 = `none` is a known variant but not a legal manifest value:
            // unquantized must be "field absent", and an explicit 0 means writer
            // and reader disagree.
            if quantization == SnapshotQuantization::None {
                return Err(format!(
                    "collection manifest: quantization '{}' is not a legal manifest value \
                     for {:?}",
                    quantization.as_str(),
                    entry.name
                ));
            }
            if entry.index_type != "hnsw"
                || entry.vector_kind != SnapshotVectorKind::HnswGraph as u8
            {
                return Err(format!(
                    "collection manifest: {} quantization requires hnsw graph storage for {:?}",
                    quantization.as_str(),
                    entry.name
                ));
            }
            if quantization == SnapshotQuantization::Int8 && manifest.format_version < 2 {
                return Err(format!(
                    "collection manifest: int8 quantization requires format_version 2 for {:?}",
                    entry.name
                ));
            }
            if quantization == SnapshotQuantization::RaBitQ {
                if manifest.format_version < 3 {
                    return Err(format!(
                        "collection manifest: rabitq quantization requires format_version 3 \
                         for {:?}",
                        entry.name
                    ));
                }
                if !matches!(entry.metric, Some(m) if m <= 1) {
                    return Err(format!(
                        "collection manifest: rabitq quantization requires an explicit \
                         metric l2|cosine for {:?}",
                        entry.name
                    ));
                }
            }
        }
        // sparse 域只允许 1（bm25），显式 0（none）与未知值都拒绝：未启用
        // 稀疏必须「字段缺席」。bm25 只允许 hnsw/flat，且要求 format_version
        // = 4——低版本二进制不认识该字段，会静默把集合恢复成 sparse=none。
        // EN: The sparse domain allows only 1 (bm25); an explicit 0 (none) and
        // unknown values are rejected — sparse off must mean "field absent".
        // bm25 is allowed only on hnsw/flat and requires format_version = 4:
        // older binaries do not know the field and would silently restore the
        // collection as sparse=none.
        if let Some(sparse) = entry.sparse {
            if sparse == 0 {
                return Err(format!(
                    "collection manifest: sparse 'none' is not a legal manifest value for {:?}",
                    entry.name
                ));
            }
            if sparse != 1 {
                return Err(format!(
                    "collection manifest: unknown sparse mode byte {sparse} for {:?}",
                    entry.name
                ));
            }
            if entry.index_type != "hnsw" && entry.index_type != "flat" {
                return Err(format!(
                    "collection manifest: sparse bm25 requires index_type hnsw|flat for {:?}",
                    entry.name
                ));
            }
            if manifest.format_version < 4 {
                return Err(format!(
                    "collection manifest: sparse bm25 requires format_version 4 for {:?}",
                    entry.name
                ));
            }
            saw_sparse = true;
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
    // format 4 ⇔ 至少一个稀疏条目：v4 二进制认识稀疏字段，无稀疏的清单写 4
    // 会让「按条目精确推导版本」的写入方语义失真，按损坏拒绝。
    // EN: format 4 ⇔ at least one sparse entry: a v4 binary knows the field, but
    // writing 4 without any sparse entry would distort the writer's exact
    // derived-version semantics, so it is rejected as corrupt.
    if manifest.format_version == 4 && !saw_sparse {
        return Err(
            "collection manifest: format_version 4 requires at least one sparse entry".to_string(),
        );
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
            vector_dtype: Some(SnapshotVectorDtype::F32),
            quantization: Some(SnapshotQuantization::None),
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
            sparse: None,
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
        assert_eq!(
            decoded.vector_dtype, None,
            "v1 carries no dtype (f32 implied)"
        );
        assert_eq!(
            decoded.quantization, None,
            "v1 carries no quantization (none implied)"
        );
        assert_eq!(decoded.name, "docs");
        assert_eq!(decoded.index_type, "hnsw");

        // v1 携带 flags（如 f16 位）必须 fail-fast，不得静默按 f32 解码。
        // EN: A v1 payload carrying flags (e.g. the f16 bit) must fail fast
        // rather than being silently decoded as f32.
        let mut with_flags = out.clone();
        with_flags[21..25].copy_from_slice(&COLLECTION_SNAPSHOT_FLAG_VECTOR_F16.to_be_bytes());
        assert!(decode_collection_snapshot_payload(&with_flags).is_err());
    }

    /// flags bit0 携带 f16：roundtrip 保真；v2 未置位的旧快照解码为显式 f32。
    /// EN: flags bit0 carries f16: faithful roundtrip; legacy v2 snapshots with
    /// a clear bit decode as explicit f32.
    #[test]
    fn payload_roundtrip_f16_dtype_bit() {
        let payload = CollectionSnapshotPayload {
            vector_dtype: Some(SnapshotVectorDtype::F16),
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        // flags 位于 magic(9)+version(4)+checkpoint(8) 之后。
        // EN: flags sit after magic(9)+version(4)+checkpoint(8).
        let flags = u32::from_be_bytes(bytes[21..25].try_into().unwrap());
        assert_eq!(flags, COLLECTION_SNAPSHOT_FLAG_VECTOR_F16);
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);
    }

    /// f16 只允许 hnsw 图存储：非 hnsw 后端与 RawF32 行在编码/解码两侧都拒绝。
    /// EN: f16 is hnsw-graph only: non-hnsw backends and RawF32 rows are rejected
    /// by both encoder and decoder.
    #[test]
    fn f16_dtype_rejected_outside_hnsw_graph() {
        let mut payload = CollectionSnapshotPayload {
            vector_dtype: Some(SnapshotVectorDtype::F16),
            ..sample_payload()
        };
        payload.vector_kind = SnapshotVectorKind::RawF32;
        payload.vector_count = 0;
        payload.vector_bytes.clear();
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        // 手工把合法 flat payload 的 flags bit0 置位：解码必须 fail-fast。
        // EN: Hand-set flags bit0 on a valid flat payload: the decoder must fail fast.
        let flat = CollectionSnapshotPayload {
            index_type: "flat".into(),
            metric: Some(0),
            hnsw: None,
            vector_dtype: Some(SnapshotVectorDtype::F32),
            vector_kind: SnapshotVectorKind::Empty,
            vector_count: 0,
            vector_bytes: Vec::new(),
            ..sample_payload()
        };
        let mut bytes = encode_collection_snapshot_payload(&flat).unwrap();
        bytes[24] |= COLLECTION_SNAPSHOT_FLAG_VECTOR_F16 as u8;
        assert!(decode_collection_snapshot_payload(&bytes).is_err());

        // v2 payload 携带 v3 稀疏位（bit3）必须 fail-fast：该版本不存在
        // 稀疏段，置位说明语义错配。
        // EN: A v2 payload carrying the v3 sparse bit (bit3) must fail fast:
        // that version has no sparse section, so a set bit means a semantic
        // mismatch.
        let mut bytes = encode_collection_snapshot_payload(&sample_payload()).unwrap();
        bytes[24] |= COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25 as u8;
        assert!(decode_collection_snapshot_payload(&bytes).is_err());

        // bit4 及以上的未知位仍然 fail-fast。
        // EN: Unknown bits from bit4 upward still fail fast.
        let mut bytes = encode_collection_snapshot_payload(&sample_payload()).unwrap();
        bytes[24] |= 0x10;
        assert!(decode_collection_snapshot_payload(&bytes).is_err());
    }

    /// flags bit1 携带 int8：roundtrip 保真（含与 f16 bit0 同时置位）。
    /// EN: flags bit1 carries int8: faithful roundtrip (including combined with
    /// the f16 bit0).
    #[test]
    fn payload_roundtrip_int8_quantization_bit() {
        let payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::Int8),
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let flags = u32::from_be_bytes(bytes[21..25].try_into().unwrap());
        assert_eq!(flags, COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8);
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        // f16 主表编码与 int8 图侧量化互不影响，两位可同时置位。
        // EN: f16 main-store encoding and int8 graph-side quantization are
        // independent; both bits may be set together.
        let payload = CollectionSnapshotPayload {
            vector_dtype: Some(SnapshotVectorDtype::F16),
            quantization: Some(SnapshotQuantization::Int8),
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let flags = u32::from_be_bytes(bytes[21..25].try_into().unwrap());
        assert_eq!(
            flags,
            COLLECTION_SNAPSHOT_FLAG_VECTOR_F16 | COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8
        );
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);
    }

    /// flags bit2 携带 RaBitQ：roundtrip 保真（与 f16 bit0 同时置位），并与
    /// int8 位互斥。
    /// EN: flags bit2 carries RaBitQ: faithful roundtrip (combined with f16 bit0)
    /// and mutually exclusive with the int8 bit.
    #[test]
    fn payload_roundtrip_rabitq_quantization_bit() {
        let payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::RaBitQ),
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let flags = u32::from_be_bytes(bytes[21..25].try_into().unwrap());
        assert_eq!(flags, COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ);
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        // f16 主表编码与 RaBitQ 图侧量化正交，两位可同时置位。
        // EN: f16 main-store encoding and RaBitQ graph-side quantization are
        // orthogonal; both bits may be set together.
        let payload = CollectionSnapshotPayload {
            vector_dtype: Some(SnapshotVectorDtype::F16),
            quantization: Some(SnapshotQuantization::RaBitQ),
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let flags = u32::from_be_bytes(bytes[21..25].try_into().unwrap());
        assert_eq!(
            flags,
            COLLECTION_SNAPSHOT_FLAG_VECTOR_F16 | COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ
        );
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        // int8 与 RaBitQ 位同时置位：解码 fail-fast（无合法形态）。
        // EN: Both int8 and RaBitQ bits set: the decoder fails fast (no legal
        // shape exists).
        let mut bytes = encode_collection_snapshot_payload(&payload).unwrap();
        bytes[24] |= COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8 as u8;
        assert!(decode_collection_snapshot_payload(&bytes).is_err());
    }

    /// RaBitQ 只允许 hnsw 图存储与 l2/cosine 度量；编码/解码两侧都拒绝违规组合。
    /// EN: RaBitQ allows hnsw graph storage and l2/cosine metrics only; both
    /// encoder and decoder reject violations.
    #[test]
    fn rabitq_quantization_requires_hnsw_graph_and_l2_or_cosine() {
        let mut payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::RaBitQ),
            ..sample_payload()
        };
        payload.index_type = "flat".into();
        payload.hnsw = None;
        // 诊断必须出现可读量化名（`rabitq`），不得只是一枚裸判别值。
        // EN: Diagnostics must carry the readable name (`rabitq`), not just a
        // bare discriminant byte.
        let err = encode_collection_snapshot_payload(&payload).unwrap_err();
        assert!(err.to_string().contains("rabitq"), "{err}");
        assert!(!err.to_string().contains("quantization 2"), "{err}");

        let mut payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::RaBitQ),
            ..sample_payload()
        };
        payload.vector_kind = SnapshotVectorKind::RawF32;
        payload.vector_count = 0;
        payload.vector_bytes.clear();
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        // ip(2)/l1(3) 度量与 RaBitQ 无距离核：编码侧拒绝。
        // EN: ip(2)/l1(3) have no RaBitQ distance kernel: the encoder rejects them.
        let payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::RaBitQ),
            metric: Some(2),
            ..sample_payload()
        };
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        // 手工置 bit2 的非 hnsw payload：解码 fail-fast。
        // EN: Hand-set bit2 on a non-hnsw payload: the decoder fails fast.
        let flat = CollectionSnapshotPayload {
            index_type: "flat".into(),
            metric: Some(0),
            hnsw: None,
            vector_dtype: Some(SnapshotVectorDtype::F32),
            vector_kind: SnapshotVectorKind::Empty,
            vector_count: 0,
            vector_bytes: Vec::new(),
            ..sample_payload()
        };
        let mut bytes = encode_collection_snapshot_payload(&flat).unwrap();
        bytes[24] |= COLLECTION_SNAPSHOT_FLAG_VECTOR_RABITQ as u8;
        assert!(decode_collection_snapshot_payload(&bytes).is_err());

        // 手工把 l2 payload 的度量字节改成 ip 后置 bit2：解码 fail-fast。
        // EN: Rewrite a l2 payload's metric byte to ip and set bit2: the decoder
        // fails fast.
        let payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::RaBitQ),
            metric: Some(0),
            ..sample_payload()
        };
        let mut bytes = encode_collection_snapshot_payload(&payload).unwrap();
        // 度量字节位于 ha_level 之后：magic(9)+ver(4)+lsn(8)+flags(4)+name_len(2)
        // +name(4)+dims(8)+index_len(2)+index(4)+ha(1)。
        // EN: The metric byte follows ha_level at magic(9)+ver(4)+lsn(8)+flags(4)
        // +name_len(2)+name(4)+dims(8)+index_len(2)+index(4)+ha(1).
        bytes[46] = 2;
        assert!(decode_collection_snapshot_payload(&bytes).is_err());
    }

    /// int8 只允许 hnsw 图存储：非 hnsw 后端与 RawF32/Empty 形态在编码/解码
    /// 两侧都拒绝。
    /// EN: int8 is hnsw-graph only: non-hnsw backends and RawF32/Empty shapes
    /// are rejected by both encoder and decoder.
    #[test]
    fn int8_quantization_rejected_outside_hnsw_graph() {
        let mut payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::Int8),
            ..sample_payload()
        };
        // 先清掉样本里的 hnsw 参数，否则会先撞「hnsw 参数只允许 hnsw」的校验，
        // 走不到 int8 的检查分支。
        // EN: Clear the sample's hnsw params first; otherwise the "hnsw params
        // require hnsw" check fires and the int8 branch is never exercised.
        payload.index_type = "flat".into();
        payload.hnsw = None;
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        let mut payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::Int8),
            ..sample_payload()
        };
        payload.vector_kind = SnapshotVectorKind::RawF32;
        payload.vector_count = 0;
        payload.vector_bytes.clear();
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        let mut payload = CollectionSnapshotPayload {
            quantization: Some(SnapshotQuantization::Int8),
            ..sample_payload()
        };
        payload.vector_kind = SnapshotVectorKind::Empty;
        payload.vector_count = 0;
        payload.vector_bytes.clear();
        assert!(encode_collection_snapshot_payload(&payload).is_err());

        // 手工把合法 flat payload 的 flags bit1 置位：解码必须 fail-fast。
        // EN: Hand-set flags bit1 on a valid flat payload: the decoder must fail fast.
        let flat = CollectionSnapshotPayload {
            index_type: "flat".into(),
            metric: Some(0),
            hnsw: None,
            vector_dtype: Some(SnapshotVectorDtype::F32),
            vector_kind: SnapshotVectorKind::Empty,
            vector_count: 0,
            vector_bytes: Vec::new(),
            ..sample_payload()
        };
        let mut bytes = encode_collection_snapshot_payload(&flat).unwrap();
        bytes[24] |= COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8 as u8;
        assert!(decode_collection_snapshot_payload(&bytes).is_err());

        // v1 快照携带 int8 位同样 fail-fast（旧版本语义不存在该字段）。
        // EN: A v1 payload carrying the int8 bit also fails fast (the field does
        // not exist in that historical version).
        let mut v1 = Vec::new();
        v1.extend_from_slice(COLLECTION_SNAPSHOT_MAGIC);
        v1.extend_from_slice(&1u32.to_be_bytes());
        v1.extend_from_slice(&42u64.to_be_bytes());
        v1.extend_from_slice(&COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8.to_be_bytes());
        v1.extend_from_slice(&4u16.to_be_bytes());
        v1.extend_from_slice(b"docs");
        v1.extend_from_slice(&4u64.to_be_bytes());
        v1.extend_from_slice(&4u16.to_be_bytes());
        v1.extend_from_slice(b"hnsw");
        v1.push(2);
        v1.push(SnapshotVectorKind::HnswGraph as u8);
        v1.extend_from_slice(&0u64.to_be_bytes());
        v1.extend_from_slice(&0u64.to_be_bytes());
        v1.extend_from_slice(&0u64.to_be_bytes());
        v1.extend_from_slice(&0u64.to_be_bytes());
        assert!(decode_collection_snapshot_payload(&v1).is_err());
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

    /// kind 4（缓冲原始行）：与 RawF32 同行布局，仅允许懒构建后端。
    /// EN: Kind 4 (buffered raw rows): same row layout as RawF32, lazy backends only.
    #[test]
    fn payload_roundtrip_buffered_raw_f32_kind() {
        let per = 8 + 4 * 4;
        let payload = CollectionSnapshotPayload {
            index_type: "clustered".to_string(),
            hnsw: None,
            metric: Some(0),
            vector_dtype: Some(SnapshotVectorDtype::F32),
            quantization: Some(SnapshotQuantization::None),
            vector_kind: SnapshotVectorKind::BufferedRawF32,
            vector_count: 3,
            vector_bytes: (0..3 * per).map(|i| i as u8).collect(),
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        for index_type in ["flat-clustered", "pq-flat"] {
            let mut p = payload.clone();
            p.index_type = index_type.to_string();
            let bytes = encode_collection_snapshot_payload(&p).unwrap();
            assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), p);
        }

        // 非懒构建后端 + kind 4 拒绝（flat 无需 hnsw 参数，能走到形态校验）。
        // EN: Kind 4 on a non-lazy backend is rejected (flat needs no hnsw params,
        // so the shape check is exercised).
        let mut bad = payload.clone();
        bad.index_type = "flat".to_string();
        assert!(encode_collection_snapshot_payload(&bad).is_err());

        // 行长度不匹配仍按 RawF32 规则拒绝。
        // EN: Row-length mismatch is rejected under the RawF32 rules.
        let mut bad = payload;
        bad.vector_count = 1;
        bad.vector_bytes = vec![0; per + 1];
        assert!(encode_collection_snapshot_payload(&bad).is_err());
    }

    /// kind 3（PQ 紧凑构建态）：外框精确校验 + 仅 pq-flat。
    /// EN: Kind 3 (PQ compact built state): exact frame validation and pq-flat only.
    #[test]
    fn payload_roundtrip_pq_flat_state_kind() {
        let per = 8 + 4 * 4;
        let state = vec![0xABu8; 16];
        let mut bytes3 = Vec::new();
        bytes3.extend_from_slice(&(state.len() as u64).to_be_bytes());
        bytes3.extend_from_slice(&state);
        bytes3.extend_from_slice(&2u64.to_be_bytes());
        for i in 0..2 * per {
            bytes3.push((i as u8).wrapping_mul(3));
        }
        let payload = CollectionSnapshotPayload {
            index_type: "pq-flat".to_string(),
            hnsw: None,
            metric: Some(0),
            vector_dtype: Some(SnapshotVectorDtype::F32),
            quantization: Some(SnapshotQuantization::None),
            vector_kind: SnapshotVectorKind::PqFlatState,
            vector_count: 5,
            vector_bytes: bytes3,
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        // 非 pq-flat 后端拒绝。
        // EN: Non-pq-flat backends are rejected.
        let mut wrong_type = payload.clone();
        wrong_type.index_type = "clustered".to_string();
        assert!(encode_collection_snapshot_payload(&wrong_type).is_err());

        // 空/截断/尾随/超量 pending 全部拒绝。
        // EN: Empty/truncated/trailing/over-count frames are all rejected.
        let mut empty = payload.clone();
        empty.vector_count = 0;
        empty.vector_bytes.clear();
        assert!(encode_collection_snapshot_payload(&empty).is_err());

        let mut truncated = payload.clone();
        truncated
            .vector_bytes
            .truncate(truncated.vector_bytes.len() - 1);
        assert!(encode_collection_snapshot_payload(&truncated).is_err());

        let mut trailing = payload.clone();
        trailing.vector_bytes.push(0);
        assert!(encode_collection_snapshot_payload(&trailing).is_err());

        let mut over = payload.clone();
        over.vector_count = 1; // pending_count=2 > vector_count
        assert!(encode_collection_snapshot_payload(&over).is_err());

        // 边界：pending_count == vector_count（kind 3 必须至少一条已建向量）。
        // EN: Boundary: pending_count == vector_count (kind 3 needs >= 1 built row).
        let mut equal = payload.clone();
        equal.vector_count = 2;
        assert!(encode_collection_snapshot_payload(&equal).is_err());

        // 前 8 字节伪造 state_len=0 必须拒绝。
        // EN: A forged state_len == 0 in the leading u64 must be rejected.
        let mut zero_state = payload.clone();
        zero_state.vector_bytes[..8].copy_from_slice(&0u64.to_be_bytes());
        assert!(encode_collection_snapshot_payload(&zero_state).is_err());

        // 解码侧：把 kind 4 payload 的 kind 字节改成 3（index_type 仍是
        // flat-clustered）必须 fail-fast。
        // EN: Decoder side: flipping a kind-4 payload's kind byte to 3 while the
        // index_type stays flat-clustered must fail fast.
        let sample = sample_payload();
        // 行宽由样例 dims 推导，构造与断言共用同一值，避免手工失配。
        // EN: Derive the row width from the sample dims and reuse it for both the
        // payload and the assertion, so the two cannot drift.
        let per = 8 + 4 * sample.dims as usize;
        let kind4 = CollectionSnapshotPayload {
            index_type: "flat-clustered".to_string(),
            hnsw: None,
            metric: Some(0),
            vector_kind: SnapshotVectorKind::BufferedRawF32,
            vector_count: 1,
            vector_bytes: vec![1; per],
            ..sample
        };
        // 行宽由 dims 单一来源推导；编码器确实按 dims 校验行宽——截短一行必须拒绝。
        // EN: The row width has a single source (dims); the encoder really checks
        // it — a one-byte-short row must be rejected.
        let mut short_row = kind4.clone();
        short_row.vector_bytes.truncate(per - 1);
        assert!(encode_collection_snapshot_payload(&short_row).is_err());
        let mut raw_bytes = encode_collection_snapshot_payload(&kind4).unwrap();
        let kind_off =
            9 + 4 + 8 + 4 + 2 + kind4.name.len() + 8 + 2 + kind4.index_type.len() + 1 + 1 + 1;
        assert_eq!(
            raw_bytes[kind_off],
            SnapshotVectorKind::BufferedRawF32 as u8,
            "kind byte offset must match the payload layout"
        );
        raw_bytes[kind_off] = SnapshotVectorKind::PqFlatState as u8;
        assert!(decode_collection_snapshot_payload(&raw_bytes).is_err());
    }

    /// manifest 的 kind 3/4 与 index_type 严格对应；未知 kind 继续拒绝。
    /// EN: Manifest kinds 3/4 map strictly to index types; unknown kinds stay rejected.
    #[test]
    fn manifest_accepts_lazy_backend_vector_kinds() {
        let base = CollectionManifestEntry {
            name: "a".into(),
            checkpoint_lsn: 10,
            dims: 4,
            index_type: "pq-flat".into(),
            ha_level: 0,
            metric: Some(0),
            vector_dtype: Some(0),
            quantization: None,
            sparse: None,
            vector_kind: SnapshotVectorKind::BufferedRawF32 as u8,
            file: snapshot_file_name("a", 10, 1, 1),
            file_len: 10,
            payload_crc32: 7,
            vector_count: 1,
            scalar_rows: 1,
        };
        let make = |entry: CollectionManifestEntry| CollectionManifest {
            format_version: 1,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: 20,
            safe_checkpoint_lsn: 10,
            collections: vec![entry],
        };
        assert!(encode_collection_manifest(&make(base.clone())).is_ok());

        // kind 4 在 clustered/flat-clustered 合法。
        // EN: Kind 4 is legal on clustered/flat-clustered.
        for index_type in ["clustered", "flat-clustered"] {
            let mut entry = base.clone();
            entry.index_type = index_type.to_string();
            assert!(encode_collection_manifest(&make(entry)).is_ok());
        }
        // kind 4 在 hnsw/flat 非法。
        // EN: Kind 4 is illegal on hnsw/flat.
        for index_type in ["hnsw", "flat"] {
            let mut entry = base.clone();
            entry.index_type = index_type.to_string();
            assert!(encode_collection_manifest(&make(entry)).is_err());
        }

        // kind 3 仅 pq-flat。
        // EN: Kind 3 is pq-flat only.
        let mut state = base.clone();
        state.vector_kind = SnapshotVectorKind::PqFlatState as u8;
        assert!(encode_collection_manifest(&make(state.clone())).is_ok());
        state.index_type = "clustered".into();
        assert!(encode_collection_manifest(&make(state)).is_err());

        // 未知 kind 仍然拒绝。
        // EN: Unknown kinds stay rejected.
        let mut unknown = base;
        unknown.vector_kind = 5;
        assert!(encode_collection_manifest(&make(unknown)).is_err());
        assert!(SnapshotVectorKind::from_u8(5).is_none());
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
                    vector_dtype: Some(0),
                    quantization: None,
                    sparse: None,
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
                    vector_dtype: Some(0),
                    quantization: None,
                    sparse: None,
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
            vector_dtype: Some(0),
            quantization: None,
            sparse: None,
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

    /// vector_dtype 域校验：越界字节与「f16 配非 hnsw/非图存储」都必须拒绝。
    /// EN: vector_dtype domain: out-of-range bytes and "f16 with non-hnsw
    /// backend / non-graph storage" must both be rejected.
    #[test]
    fn manifest_rejects_invalid_vector_dtype_bindings() {
        let base = CollectionManifestEntry {
            name: "a".into(),
            checkpoint_lsn: 10,
            dims: 4,
            index_type: "hnsw".into(),
            ha_level: 0,
            metric: Some(1),
            vector_dtype: Some(SnapshotVectorDtype::F16 as u8),
            quantization: None,
            sparse: None,
            vector_kind: SnapshotVectorKind::HnswGraph as u8,
            file: snapshot_file_name("a", 10, 1, 1),
            file_len: 10,
            payload_crc32: 7,
            vector_count: 1,
            scalar_rows: 1,
        };
        let make = |entry: CollectionManifestEntry| CollectionManifest {
            format_version: 1,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: 20,
            safe_checkpoint_lsn: 10,
            collections: vec![entry],
        };
        // 合法 f16/hnsw 基线可编码。
        // EN: The legal f16/hnsw baseline encodes.
        assert!(encode_collection_manifest(&make(base.clone())).is_ok());

        let mut unknown = base.clone();
        unknown.vector_dtype = Some(2);
        assert!(encode_collection_manifest(&make(unknown)).is_err());

        let mut flat = base.clone();
        flat.index_type = "flat".into();
        assert!(encode_collection_manifest(&make(flat)).is_err());

        let mut raw = base;
        raw.vector_kind = SnapshotVectorKind::RawF32 as u8;
        assert!(encode_collection_manifest(&make(raw)).is_err());
    }

    /// manifest quantization：int8(1) 要求 format_version ≥ 2，RaBitQ(2) 要求
    /// ≥ 3；两者都必须 hnsw 图存储，RaBitQ 另要求 l2/cosine 度量。字段缺席的
    /// 旧 manifest 继续可读且不被回写。
    /// EN: manifest quantization: int8(1) requires format_version ≥ 2 and
    /// RaBitQ(2) requires ≥ 3; both need hnsw graph storage, and RaBitQ also
    /// needs the l2/cosine metric. Legacy manifests without the field stay
    /// readable and never gain it on rewrite.
    #[test]
    fn manifest_quantization_requires_version_and_hnsw_graph() {
        let base = CollectionManifestEntry {
            name: "a".into(),
            checkpoint_lsn: 10,
            dims: 4,
            index_type: "hnsw".into(),
            ha_level: 0,
            metric: Some(1),
            vector_dtype: Some(0),
            quantization: Some(SnapshotQuantization::Int8.as_u8()),
            sparse: None,
            vector_kind: SnapshotVectorKind::HnswGraph as u8,
            file: snapshot_file_name("a", 10, 1, 1),
            file_len: 10,
            payload_crc32: 7,
            vector_count: 1,
            scalar_rows: 1,
        };
        let make = |entry: CollectionManifestEntry, format_version: u32| CollectionManifest {
            format_version,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: 20,
            safe_checkpoint_lsn: 10,
            collections: vec![entry],
        };
        // 合法基线：v2 + int8 + hnsw 图存储可编码并 roundtrip。
        // EN: legal baseline: v2 + int8 + hnsw graph storage encodes and roundtrips.
        let manifest = make(base.clone(), 2);
        let bytes = encode_collection_manifest(&manifest).unwrap();
        assert_eq!(decode_collection_manifest(&bytes).unwrap(), manifest);

        // 未量化形态不把字段写进 JSON（避免无谓的 manifest 变更）。
        // EN: unquantized shapes must not emit the field (no spurious manifest
        // changes).
        let plain = CollectionManifestEntry {
            quantization: None,
            ..base.clone()
        };
        let json = String::from_utf8(serde_json::to_vec(&plain).unwrap()).unwrap();
        assert!(!json.contains("quantization"));
        let legacy_manifest = make(plain, 1);
        let bytes = encode_collection_manifest(&legacy_manifest).unwrap();
        assert_eq!(decode_collection_manifest(&bytes).unwrap(), legacy_manifest);

        // v1 + int8 → 拒绝（v1 二进制会忽略字段并静默按未量化恢复）。
        // EN: v1 + int8 → rejected (v1 binaries ignore the field and silently
        // restore unquantized).
        assert!(encode_collection_manifest(&make(base.clone(), 1)).is_err());

        // 非 hnsw / 非图存储 → 拒绝。
        // EN: non-hnsw / non-graph storage → rejected.
        let mut flat = base.clone();
        flat.index_type = "flat".into();
        assert!(encode_collection_manifest(&make(flat, 2)).is_err());

        let mut raw = base.clone();
        raw.vector_kind = SnapshotVectorKind::RawF32 as u8;
        assert!(encode_collection_manifest(&make(raw, 2)).is_err());

        // 域外值（0/3）→ 拒绝。0 是已知的 `none` 变体，诊断必须区分
        //「已知值但不合法」与「未知字节」，不得统一报成 unknown。
        // EN: out-of-domain values (0/3) → rejected. 0 is the known `none`
        // variant, so the diagnostic must distinguish "known but illegal" from
        // "unknown byte" instead of reporting both as unknown.
        let mut zero = base.clone();
        zero.quantization = Some(0);
        let err = encode_collection_manifest(&make(zero, 2)).unwrap_err();
        assert!(
            err.to_string().contains("not a legal manifest value"),
            "{err}"
        );

        let mut unknown = base.clone();
        unknown.quantization = Some(3);
        assert!(encode_collection_manifest(&make(unknown, 2)).is_err());

        // RaBitQ：v3 + hnsw 图存储 + l2/cosine 度量合法并可 roundtrip。
        // EN: RaBitQ: v3 + hnsw graph storage + l2/cosine metric is legal and
        // roundtrips.
        let rabitq = CollectionManifestEntry {
            quantization: Some(SnapshotQuantization::RaBitQ.as_u8()),
            ..base.clone()
        };
        let manifest = make(rabitq.clone(), 3);
        let bytes = encode_collection_manifest(&manifest).unwrap();
        assert_eq!(decode_collection_manifest(&bytes).unwrap(), manifest);

        // v2 + RaBitQ → 拒绝（v2 二进制不认识该取值，会静默按未量化恢复）。
        // EN: v2 + RaBitQ → rejected (v2 binaries do not know the value and
        // would silently restore unquantized).
        assert!(encode_collection_manifest(&make(rabitq.clone(), 2)).is_err());

        // RaBitQ + 非 hnsw/非图存储 → 拒绝。
        // EN: RaBitQ + non-hnsw/non-graph storage → rejected.
        let mut rabitq_flat = rabitq.clone();
        rabitq_flat.index_type = "flat".into();
        assert!(encode_collection_manifest(&make(rabitq_flat, 3)).is_err());

        let mut rabitq_raw = rabitq.clone();
        rabitq_raw.vector_kind = SnapshotVectorKind::RawF32 as u8;
        assert!(encode_collection_manifest(&make(rabitq_raw, 3)).is_err());

        // RaBitQ + 度量缺席 → 拒绝：无法证明 l2/cosine 约束，恢复会按历史
        // 默认 Cosine 重建（OCR 219c5b1 #3）。
        // EN: RaBitQ + missing metric → rejected: the l2/cosine constraint
        // cannot be proven and restore would rebuild under the historical
        // Cosine default (OCR 219c5b1 #3).
        let mut rabitq_no_metric = rabitq.clone();
        rabitq_no_metric.metric = None;
        assert!(encode_collection_manifest(&make(rabitq_no_metric, 3)).is_err());

        // RaBitQ + ip/l1 度量 → 拒绝（1 bit 方向码没有对应距离核）。
        // EN: RaBitQ + ip/l1 metric → rejected (the 1-bit direction code has no
        // matching kernel).
        let mut rabitq_ip = rabitq;
        rabitq_ip.metric = Some(2);
        assert!(encode_collection_manifest(&make(rabitq_ip, 3)).is_err());

        // 版本域外（5）→ 拒绝；v3 本身合法（即使 entry 未量化）。
        // EN: out-of-domain version (5) → rejected; v3 itself is legal even
        // with an unquantized entry.
        let mut future = base;
        future.quantization = None;
        assert!(encode_collection_manifest(&make(future.clone(), 5)).is_err());
        assert!(encode_collection_manifest(&make(future, 3)).is_ok());
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

    /// v3 稀疏段基线：flat 集合、无标量、一条墓碑（9；posting 命中墓碑是
    /// 合法状态，编码器必须接受，见下方复活用例）。
    /// EN: v3 sparse-section baseline: flat collection, no scalars, one tombstone
    /// (9; postings may legitimately hit tombstones — the encoder must accept
    /// them, see the resurrection case below).
    fn sparse_payload() -> CollectionSnapshotPayload {
        CollectionSnapshotPayload {
            checkpoint_lsn: 42,
            name: "docs".to_string(),
            dims: 4,
            index_type: "flat".to_string(),
            ha_level: 2,
            metric: Some(0),
            hnsw: None,
            vector_dtype: Some(SnapshotVectorDtype::F32),
            quantization: Some(SnapshotQuantization::None),
            vector_kind: SnapshotVectorKind::Empty,
            vector_count: 0,
            vector_bytes: Vec::new(),
            scalars: Vec::new(),
            tombstones: vec![9],
            sparse: Some(SnapshotSparseIndex {
                doc_count: 3,
                total_len: 3.5,
                terms: vec![(1, vec![(1, 1.0), (2, 0.5)]), (7, vec![(3, 2.0)])],
            }),
        }
    }

    /// v3 稀疏段 roundtrip 保真：版本 3、bit3 置位；与 f16/int8 位正交组合。
    /// EN: v3 sparse section roundtrips faithfully: version 3 with bit3 set;
    /// orthogonally combinable with the f16/int8 bits.
    #[test]
    fn payload_roundtrip_sparse_v3_and_flag_mapping() {
        let payload = sparse_payload();
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let version = u32::from_be_bytes(bytes[9..13].try_into().unwrap());
        assert_eq!(version, 3);
        let flags = u32::from_be_bytes(bytes[21..25].try_into().unwrap());
        assert_eq!(flags, COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25);
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        // sparse 与 f16 主表编码、int8 图侧量化正交：三位同时置位仍 roundtrip。
        // EN: sparse is orthogonal to the f16 main-store encoding and int8
        // graph-side quantization: all three bits together still roundtrip.
        let payload = CollectionSnapshotPayload {
            sparse: sparse_payload().sparse,
            vector_dtype: Some(SnapshotVectorDtype::F16),
            quantization: Some(SnapshotQuantization::Int8),
            ..sample_payload()
        };
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        let flags = u32::from_be_bytes(bytes[21..25].try_into().unwrap());
        assert_eq!(
            flags,
            COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25
                | COLLECTION_SNAPSHOT_FLAG_VECTOR_F16
                | COLLECTION_SNAPSHOT_FLAG_VECTOR_INT8
        );
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        // v3 缺 bit3：稀疏段无法定位，解码必须 fail-fast（不得静默丢弃）。
        // EN: v3 without bit3: the sparse section cannot be located, so the
        // decoder must fail fast (never silently drop it).
        let payload = sparse_payload();
        let mut bytes = encode_collection_snapshot_payload(&payload).unwrap();
        bytes[24] &= !(COLLECTION_SNAPSHOT_FLAG_SPARSE_BM25 as u8);
        assert!(decode_collection_snapshot_payload(&bytes).is_err());
    }

    /// 稀疏段语义拒绝矩阵：backend/顺序/权重/doc_count/total_len/截断。
    /// EN: Sparse-section rejection matrix: backend/order/weights/doc_count/
    /// total_len/truncation.
    #[test]
    fn sparse_section_rejections() {
        let mut payload = sparse_payload();
        payload.index_type = "hii-l0".into();
        let err = encode_collection_snapshot_payload(&payload).unwrap_err();
        assert!(err.to_string().contains("hnsw|flat"), "{err}");

        let patch = |f: &dyn Fn(&mut SnapshotSparseIndex)| {
            let mut payload = sparse_payload();
            if let Some(sparse) = payload.sparse.as_mut() {
                f(sparse);
            }
            payload
        };
        let assert_encode_err = |payload: &CollectionSnapshotPayload| {
            assert!(
                encode_collection_snapshot_payload(payload).is_err(),
                "expected encode rejection"
            );
        };

        // term 必须严格升序且落在 1..=2^24-1。
        // EN: terms must ascend strictly and sit within 1..=2^24-1.
        assert_encode_err(&patch(&|s| s.terms.reverse()));
        assert_encode_err(&patch(&|s| s.terms[0].0 = 0));
        assert_encode_err(&patch(&|s| s.terms[1].0 = 1 << 24));

        // term 内文档必须严格升序、权重有限 > 0。
        // EN: docs within a term must ascend strictly with finite > 0 weights.
        assert_encode_err(&patch(&|s| s.terms[0].1.reverse()));
        assert_encode_err(&patch(&|s| s.terms[0].1[0].1 = 0.0));
        assert_encode_err(&patch(&|s| s.terms[0].1[0].1 = f32::NAN));
        assert_encode_err(&patch(&|s| s.terms[0].1[0].1 = f32::INFINITY));

        // doc_count 与全局去重文档数对账；total_len 有限且 ≥ 0。
        // EN: doc_count reconciles against globally distinct docs; total_len is
        // finite and ≥ 0.
        assert_encode_err(&patch(&|s| s.doc_count = 4));
        assert_encode_err(&patch(&|s| s.doc_count = 2));
        assert_encode_err(&patch(&|s| s.total_len = f64::NAN));
        assert_encode_err(&patch(&|s| s.total_len = -1.0));
        // 有文档时 total_len 必须 > 0，否则恢复后 avgdl=0 会静默返回空。
        // EN: With documents present total_len must be > 0; otherwise search
        // silently returns empty after recovery (avgdl=0).
        assert_encode_err(&patch(&|s| s.total_len = 0.0));

        // 墓碑只隐藏标量/载荷行；稀疏 posting 命中墓碑 id（裸向量 + 稀疏
        // 复活同一 id）是合法状态，编码器必须接受并原样往返。
        // EN: Tombstones hide scalar/payload rows only; a sparse posting on a
        // tombstoned id (bare-vector + sparse resurrection) is a legitimate
        // state that the encoder must accept and roundtrip faithfully.
        let mut payload = sparse_payload();
        if let Some(sparse) = payload.sparse.as_mut() {
            sparse.terms[0].1.push((9, 1.0));
            sparse.doc_count += 1;
            sparse.total_len += 1.0;
        }
        let bytes = encode_collection_snapshot_payload(&payload).unwrap();
        assert_eq!(decode_collection_snapshot_payload(&bytes).unwrap(), payload);

        // 解码侧：截断/截尾都 fail-fast（长度上界与 trailing 检查）。
        // EN: decoder side: truncation/trailing bytes fail fast (length bounds
        // and the trailing check).
        let mut bytes = encode_collection_snapshot_payload(&sparse_payload()).unwrap();
        bytes.truncate(bytes.len() - 1);
        assert!(decode_collection_snapshot_payload(&bytes).is_err());
        let mut bytes = encode_collection_snapshot_payload(&sparse_payload()).unwrap();
        bytes.push(0);
        assert!(decode_collection_snapshot_payload(&bytes).is_err());
    }

    /// manifest sparse：仅 bm25=1 合法、要求 hnsw/flat 与 format_version 4；
    /// format 4 必须至少有一个稀疏条目；字段缺席时不回写。
    /// EN: manifest sparse: only bm25=1 is legal, requires hnsw/flat and
    /// format_version 4; format 4 requires at least one sparse entry; the field
    /// is omitted when absent.
    #[test]
    fn manifest_sparse_requires_format4_and_hnsw_flat() {
        let base = CollectionManifestEntry {
            name: "a".into(),
            checkpoint_lsn: 10,
            dims: 4,
            index_type: "flat".into(),
            ha_level: 0,
            metric: Some(0),
            vector_dtype: Some(0),
            quantization: None,
            sparse: Some(1),
            vector_kind: SnapshotVectorKind::RawF32 as u8,
            file: snapshot_file_name("a", 10, 1, 1),
            file_len: 10,
            payload_crc32: 7,
            vector_count: 1,
            scalar_rows: 1,
        };
        let make = |entry: CollectionManifestEntry, format_version: u32| CollectionManifest {
            format_version,
            manifest_seq: 1,
            created_at_unix: 1,
            global_barrier_lsn: 20,
            safe_checkpoint_lsn: 10,
            collections: vec![entry],
        };
        // 合法基线：format 4 + spar=bm25 + flat 可编码并 roundtrip。
        // EN: legal baseline: format 4 + sparse=bm25 + flat encodes and roundtrips.
        let manifest = make(base.clone(), 4);
        let bytes = encode_collection_manifest(&manifest).unwrap();
        assert_eq!(decode_collection_manifest(&bytes).unwrap(), manifest);

        // 字段缺席：JSON 不出现 sparse，且 format 4 因无稀疏条目被拒。
        // EN: field absent: JSON omits sparse, and format 4 is rejected because
        // there is no sparse entry.
        let plain = CollectionManifestEntry {
            sparse: None,
            ..base.clone()
        };
        let json = String::from_utf8(serde_json::to_vec(&plain).unwrap()).unwrap();
        assert!(!json.contains("sparse"));
        assert!(encode_collection_manifest(&make(plain.clone(), 4)).is_err());

        // 版本不足 / backend 不合法 / 域外取值：全部拒绝。
        // EN: insufficient version / illegal backend / out-of-domain values:
        // all rejected.
        assert!(encode_collection_manifest(&make(base.clone(), 3)).is_err());

        let mut hnsw_only = base.clone();
        hnsw_only.index_type = "hii-l0".into();
        assert!(encode_collection_manifest(&make(hnsw_only, 4)).is_err());

        let mut zero = base.clone();
        zero.sparse = Some(0);
        let err = encode_collection_manifest(&make(zero, 4)).unwrap_err();
        assert!(
            err.to_string().contains("not a legal manifest value"),
            "{err}"
        );

        let mut unknown = base;
        unknown.sparse = Some(2);
        assert!(encode_collection_manifest(&make(unknown, 4)).is_err());
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
