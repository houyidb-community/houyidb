//! HNSW 图二进制快照（P4.4.0）：进程重启时直接载图，跳过昂贵的逐点建图。
//! EN: Binary HNSW graph snapshot (P4.4.0) — restore the graph directly on
//! restart instead of rebuilding it point-by-point.
//!
//! 分层说明：本文件只定义**纯 payload** 的编解码（不依赖 crc32fast/serde，
//! houyi-vector 保持零二进制依赖）；文件信封（magic/checkpoint_lsn/CRC32/
//! 原子写）在 houyi-persistence 的 `graph_snapshot_file.rs`，payload 对信封
//! 不透明。
//! EN: This file defines the pure payload codec only (no crc32fast/serde deps);
//! the file envelope (magic / checkpoint_lsn / CRC32 / atomic write) lives in
//! houyi-persistence, which treats the payload as opaque bytes.
//!
//! 布局（全大端 BE，与 WAL v2 文件字节序一致）：
//! Layout (all big-endian, matching WAL v2 file byte order):
//! ```text
//! [HEADER 108B]
//!   magic[8] "HNSWGRPH" | version u32 | dims u32 | metric u8 | flags u8 | rsv u16
//!   m u32 | ef_construction u32 | ef_search u32 | num_layers u32 | seed u64
//!   l0_mult f64 | upper_mult f64          ← 生效层乘数（LAYER_MULTS 布局依赖）
//!   node_count u64 | arena_len u64 | link_free_pair_count u64 | vector_count u64
//!   entry_point u64 (flags bit0 = 0 时无意义) | max_layer u32 | rsv u32
//!   flags: bit0 = HAS_ENTRY_POINT —— entry_point 是否存在靠 flag 表达而非
//!   u64::MAX 哨兵：VectorId 是普通 u64，u64::MAX 是合法 id，哨兵会与其碰撞
//!   （OCR ae8d7e3 HIGH）。
//!   bit1 = VECTOR_F16：仅 version≥2 允许。f32 快照写 version=1（旧二进制可读），
//!   f16 写 version=2（旧二进制以 UnsupportedVersion 明确拒绝，不做静默降级）。
//!   EN: bit1 = VECTOR_F16, allowed only when version>=2. f32 snapshots are
//!   EN: written as version=1 (readable by older binaries); f16 writes version=2
//!   EN: (older binaries reject with UnsupportedVersion instead of downgrading).
//! [nodes]      node_count × {id u64, link_base u64, max_layer u8, layer_len [u16;16]}
//! [link_arena] arena_len × u64
//! [link_free]  pair_count × {slots u64, base u64}   ← AHashMap<usize,Vec<usize>> 拍平
//! [vectors]    vector_count × {id u64, dims × f32-bits u32}
//! ```
//!
//! 关键设计决策（WHY）：
//! - `link_base` 是 arena 下标，其解释依赖 `layer_m`（即全局 `LAYER_MULTS` 与
//!   `NEIGHBOR_LINK_SLACK` 编译常量）。快照记录**生效乘数**，恢复时与当前进程
//!   乘数不一致即拒绝（弃用快照全量重放），避免静默错位。
//! - `m/ef_*` 不做硬性匹配：它们是运行时可调参数，恢复时直接**采用快照值**
//!   （调用方负责同步原子量）。硬性拒绝项只有 dims/metric/num_layers/乘数。
//! - `StdRng` 不序列化状态：恢复时用同一 seed 重播种。rng 只服务于后续插入的
//!   随机层抽取，不影响已有图结构的确定性。
//! - 邻接 id 的引用完整性（邻居必须存在于节点集）不在解码期校验——信封 CRC32
//!   已覆盖损坏场景，搜索路径另有悬空链接防御性跳过兜底。

use crate::hnsw_graph::{GraphVector, HnswGraph, HnswNode, SnapshotGraphParts, MAX_HNSW_LAYERS};
use crate::quantization::{Int8Vector, Quantization};
use crate::rabitq::{self, RaBitQVector};
use crate::stored_vector::{StoredVector, VectorDtype, VectorRef};
use crate::{DistanceMetric, VectorId};
use std::fmt;
use std::sync::Arc;

use ahash::AHashMap;

/// 快照 payload 魔数。 / EN: Payload magic.
pub(crate) const SNAPSHOT_MAGIC: &[u8; 8] = b"HNSWGRPH";
/// f32 快照写出的版本：保持旧二进制冷启动可读（快照只是建图加速，不是真相源）。
/// 解码 v1 时 flags 只允许 bit0，携带 f16 位即拒绝。
/// EN: Version written for f32 snapshots so older binaries can still cold-start
/// (a snapshot only accelerates graph construction; the WAL is the source of
/// truth). When decoding v1, only flags bit0 is allowed; carrying the f16 bit
/// is rejected.
pub(crate) const SNAPSHOT_VERSION_F32: u32 = 1;
/// dtype 标记版快照（f16 起使用）：旧二进制只认 v1，遇到 v2 会以
/// `UnsupportedVersion` 干净拒绝并回退全量 WAL 重放，而不是把 f16 图静默
/// 恢复成 f32、再在下次快照时固化降级。
/// EN: dtype-tagged snapshot version (used from f16 onward): old binaries only
/// accept v1 and cleanly reject v2 with `UnsupportedVersion`, falling back to
/// full WAL replay — instead of silently restoring an f16 graph as f32 and
/// then persisting that downgrade on the next snapshot.
pub(crate) const SNAPSHOT_VERSION_DTYPE: u32 = 2;
/// int8 图侧量化快照版本（P2）。v3 的 vectors 段为 `[id][scale][bias][i8 codes][f32 source]`：
/// codes 原样还原图侧距离行为，source 还原主表原始值（重启后精确 rescore 的
/// 依据）。旧二进制只认 v1/v2，遇到 v3 以 `UnsupportedVersion` 干净拒绝并回退
/// 全量 WAL 重放，而不是把 int8 图静默恢复成未量化并固化降级。
/// EN: int8 graph-quantization snapshot version (P2). The v3 vectors section is
/// `[id][scale][bias][i8 codes][f32 source]`: codes restore in-graph distance
/// behavior verbatim, source restores raw main-store values (the basis for
/// exact rescoring after restart). Old binaries only know v1/v2 and cleanly
/// reject v3 with `UnsupportedVersion`, falling back to full WAL replay instead
/// of silently restoring an int8 graph as unquantized and persisting that
/// downgrade.
pub(crate) const SNAPSHOT_VERSION_INT8: u32 = 3;
/// RaBitQ 图侧量化快照版本（P3）。v4 的 vectors 段为
/// `[id][norm f32][alpha f32][packed codes ceil(D/64)·u64][dims×f32 source]`：
/// codes/alpha/norm 原样还原图侧距离行为，source 还原主表原始值（精确 rescore
/// 的依据）。旧二进制只认 v1–v3，遇到 v4 以 `UnsupportedVersion` 干净拒绝并
/// 回退全量 WAL 重放，而不是把 RaBitQ 图静默恢复成未量化或 int8 并固化降级。
/// EN: RaBitQ graph-quantization snapshot version (P3). The v4 vectors section is
/// `[id][norm f32][alpha f32][packed codes ceil(D/64)·u64][dims×f32 source]`:
/// codes/alpha/norm restore in-graph distance behavior verbatim, source restores
/// raw main-store values (the basis for exact rescoring). Old binaries only know
/// v1–v3 and cleanly reject v4 with `UnsupportedVersion`, falling back to full
/// WAL replay instead of silently restoring a RaBitQ graph as unquantized/int8.
pub(crate) const SNAPSHOT_VERSION_RABITQ: u32 = 4;
/// 定长头字节数（见模块文档布局图）。 / EN: Fixed header length.
pub(crate) const HEADER_LEN: usize = 108;
/// 单节点记录字节数：id u64 + link_base u64 + max_layer u8 + layer_len 16×u16。
/// EN: Per-node record size.
pub(crate) const NODE_RECORD_LEN: usize = 8 + 8 + 1 + 2 * MAX_HNSW_LAYERS;
/// flags bit0：entry_point 字段存在。 / EN: flags bit0 — entry_point present.
const FLAG_HAS_ENTRY_POINT: u8 = 0x01;
/// flags bit1：向量存储为 f16（vectors 段仍按 f32 字节落盘，解码时下转）。
/// 仅 v2（`SNAPSHOT_VERSION_DTYPE`）允许该位；v1 携带该位即拒绝，避免把
/// 带 dtype 语义的快照按无 dtype 语义解释。该位缺席即 f32。
/// EN: flags bit1 — vectors are stored as f16 (the vectors section still writes
/// f32 bytes and down-converts on decode). Only v2 (`SNAPSHOT_VERSION_DTYPE`)
/// may carry it; v1 with this bit set is rejected so a dtype-bearing snapshot
/// is never interpreted under dtype-free semantics. Absent means f32.
const FLAG_VECTOR_F16: u8 = 0x02;
/// flags bit2：图侧 int8 量化。仅 v3 允许且**必须**置位（v3 只因 int8 存在）；
/// v3 可与 f16 位并存（主表编码与图侧量化正交）。
/// EN: flags bit2 — graph-side int8 quantization. Only v3 allows it, and v3
/// **requires** it (v3 exists solely for int8); it may coexist with the f16
/// bit since main-store encoding and graph-side quantization are orthogonal.
const FLAG_VECTOR_INT8: u8 = 0x04;
/// flags bit3：图侧 RaBitQ 量化。仅 v4 允许且**必须**置位；v4 可与 f16 位并存，
/// 但**禁止** int8 位（同一张图不可能同时采用两种图侧量化）。
/// EN: flags bit3 — graph-side RaBitQ quantization. Only v4 allows it, and v4
/// **requires** it; it may coexist with the f16 bit but must never carry the
/// int8 bit (one graph cannot use two graph-side quantization schemes).
const FLAG_VECTOR_RABITQ: u8 = 0x08;

/// 解码可接受的版本：v1（仅 f32）、v2（允许 dtype 位）、v3（int8）、v4（RaBitQ）。
/// EN: Versions accepted by the decoder: v1 (f32 only), v2 (dtype bit
/// allowed), v3 (int8), v4 (RaBitQ).
fn is_supported_version(version: u32) -> bool {
    (SNAPSHOT_VERSION_F32..=SNAPSHOT_VERSION_RABITQ).contains(&version)
}

/// 按版本校验 flags：v1 只认识 bit0，v2 加 f16 位，v3 再加 int8 位，v4 换成
/// rabitq 位（bit3 必置、int8 位禁置，可与 f16 位并存）；其余位一律拒绝
/// （未知位代表未来格式或损坏，不得静默忽略）。v3 必须携带 int8 位——v3 只因
/// int8 存在；v4 必须携带 rabitq 位——v4 只因 rabitq 存在，缺少对应位说明
/// 版本/标志被伪造。
/// EN: Validate flags per version: v1 knows bit0, v2 adds the f16 bit, v3 adds
/// the int8 bit, and v4 swaps to the rabitq bit (bit3 mandatory, int8 bit
/// forbidden, may coexist with the f16 bit); every other bit is rejected
/// (unknown bits mean a future format or corruption and must not be silently
/// ignored). v3 must carry the int8 bit — v3 exists solely for int8; v4 must
/// carry the rabitq bit — v4 exists solely for rabitq, so a missing bit means
/// the version/flags pair was forged.
fn validate_flags(version: u32, flags: u8) -> Result<(), GraphSnapshotError> {
    let allowed = match version {
        SNAPSHOT_VERSION_F32 => FLAG_HAS_ENTRY_POINT,
        SNAPSHOT_VERSION_DTYPE => FLAG_HAS_ENTRY_POINT | FLAG_VECTOR_F16,
        SNAPSHOT_VERSION_INT8 => FLAG_HAS_ENTRY_POINT | FLAG_VECTOR_F16 | FLAG_VECTOR_INT8,
        // v4 的 int8 位缺席即拒绝（不在 allowed 里）；rabitq 位必置。
        // EN: The v4 int8 bit is absent from `allowed` (rejected if set); the
        // rabitq bit is mandatory.
        SNAPSHOT_VERSION_RABITQ => FLAG_HAS_ENTRY_POINT | FLAG_VECTOR_F16 | FLAG_VECTOR_RABITQ,
        _ => return Err(GraphSnapshotError::InvalidStructure("flags")),
    };
    if flags & !allowed != 0 {
        return Err(GraphSnapshotError::InvalidStructure("flags"));
    }
    if version == SNAPSHOT_VERSION_INT8 && flags & FLAG_VECTOR_INT8 == 0 {
        return Err(GraphSnapshotError::InvalidStructure("flags"));
    }
    if version == SNAPSHOT_VERSION_RABITQ && flags & FLAG_VECTOR_RABITQ == 0 {
        return Err(GraphSnapshotError::InvalidStructure("flags"));
    }
    Ok(())
}

/// 恢复期的结构性匹配期望：任一不符即拒绝快照（调用方退化为全量重放）。
/// EN: Structural expectations for restore; any mismatch rejects the snapshot
/// (caller falls back to full WAL replay).
///
/// `m/ef_*` 刻意**不在**此列——它们是运行时可调参数，恢复时采用快照值。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SnapshotExpectations {
    pub dims: usize,
    pub metric: DistanceMetric,
    /// 目标图的向量 dtype；快照 flags 必须与其一致。
    /// EN: Target graph vector dtype; the snapshot flags must agree.
    pub dtype: VectorDtype,
    /// 目标图的图侧量化方式；快照 flags 必须与其一致（int8 不可静默降级为 none）。
    /// EN: Target graph quantization; the snapshot flags must agree (int8 must
    /// never downgrade silently to none).
    pub quantization: crate::quantization::Quantization,
    pub num_layers: usize,
    pub l0_mult: f64,
    pub upper_mult: f64,
}

impl SnapshotExpectations {
    /// 以一张现存图的结构参数 + 当前进程生效乘数构造期望。
    /// EN: Build expectations from an existing graph's structural params plus
    /// the process-wide effective multipliers.
    pub(crate) fn for_graph(g: &HnswGraph) -> Self {
        let (l0_mult, upper_mult) = crate::hnsw_graph::effective_layer_multipliers();
        Self {
            dims: g.dims,
            metric: g.metric,
            dtype: g.dtype(),
            quantization: g.quantization(),
            num_layers: g.num_layers,
            l0_mult,
            upper_mult,
        }
    }
}

/// 快照编解码错误。出现在 `HotBufferIndex::restore_graph_snapshot` 的公开
/// 签名中（下游按变体决定退化策略），故为 crate 公开类型。
/// EN: Snapshot codec errors. Public because it appears in the public
/// `restore_graph_snapshot` signature — downstream matches variants to pick
/// the fallback strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphSnapshotError {
    /// 魔数不匹配。
    /// EN: Magic mismatch.
    BadMagic,
    /// 不支持的格式版本。
    /// EN: Unsupported format version.
    UnsupportedVersion(u32),
    /// 输入在结构中途结束（给出短缺处偏移）。
    /// EN: Input ended mid-structure (offset of the shortfall).
    Truncated { at: usize },
    /// 最后一段之后存在多余字节。
    /// EN: Extra bytes after the last section.
    TrailingBytes { extra: usize },
    /// 未知的距离度量字节。
    /// EN: Unknown metric byte.
    InvalidMetric(u8),
    /// 结构期望不匹配（字段名）。
    /// EN: Structural expectation mismatch (field name).
    Mismatch(&'static str),
    /// 内容内部不一致（字段名）。
    /// EN: Internally inconsistent content (field name).
    InvalidStructure(&'static str),
}

impl fmt::Display for GraphSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad snapshot magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported snapshot version {v}"),
            Self::Truncated { at } => write!(f, "snapshot truncated at offset {at}"),
            Self::TrailingBytes { extra } => write!(f, "snapshot has {extra} trailing bytes"),
            Self::InvalidMetric(b) => write!(f, "invalid metric byte {b}"),
            Self::Mismatch(field) => write!(f, "snapshot field mismatch: {field}"),
            Self::InvalidStructure(field) => write!(f, "snapshot invalid structure: {field}"),
        }
    }
}

impl std::error::Error for GraphSnapshotError {}

// ---------------------------------------------------------------------------
// 小端序无关的 BE 读写游标（零依赖手卷）。
// BE cursor, hand-rolled to keep houyi-vector free of binary codec deps.
// ---------------------------------------------------------------------------

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    fn f64_bits(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_bits().to_be_bytes());
    }
    fn f32_bits(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_bits().to_be_bytes());
    }
    fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], GraphSnapshotError> {
        if self.buf.len() - self.pos < n {
            return Err(GraphSnapshotError::Truncated { at: self.pos });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, GraphSnapshotError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, GraphSnapshotError> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, GraphSnapshotError> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GraphSnapshotError> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64_bits(&mut self) -> Result<f64, GraphSnapshotError> {
        Ok(f64::from_bits(u64::from_be_bytes(
            self.take(8)?.try_into().unwrap(),
        )))
    }
    fn f32_bits(&mut self) -> Result<f32, GraphSnapshotError> {
        Ok(f32::from_bits(u32::from_be_bytes(
            self.take(4)?.try_into().unwrap(),
        )))
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn position(&self) -> usize {
        self.pos
    }
}

// ---------------------------------------------------------------------------
// HnswGraph 编解码实现。
// EN: HnswGraph encode/decode implementation.
// ---------------------------------------------------------------------------

/// 只读头部 peek：返回 `(dims, metric)`，不做全量解码。
/// EN: Header-only peek returning `(dims, metric)` without full decoding.
///
/// 用途：server 恢复 collection 快照时，需要先知道维度才能按写入侧同一
/// 构造参数创建后端（restore 的期望匹配在完整解码时才进行）。
/// EN: Used by server-side collection restore, which needs dims to construct
/// the backend with the same parameters as the write path (full expectation
/// matching still happens during complete decoding).
pub fn peek_graph_snapshot_dims_metric(
    payload: &[u8],
) -> Result<(usize, DistanceMetric), GraphSnapshotError> {
    let (dims, metric, _dtype) = peek_graph_snapshot_dims_metric_dtype(payload)?;
    Ok((dims, metric))
}

/// 只读头部 peek：返回 `(dims, metric, dtype)`，供恢复侧按写入侧同一 dtype
/// 构造后端；旧 `peek_graph_snapshot_dims_metric` 保留为忽略 dtype 的兼容入口。
/// EN: Header-only peek returning `(dims, metric, dtype)` so restore can build the
/// backend with the same dtype as the writer; the old dims/metric peek remains as
/// a compatibility entry that ignores dtype.
pub fn peek_graph_snapshot_dims_metric_dtype(
    payload: &[u8],
) -> Result<(usize, DistanceMetric, VectorDtype), GraphSnapshotError> {
    let (dims, metric, dtype, _quantization) =
        peek_graph_snapshot_dims_metric_dtype_quantization(payload)?;
    Ok((dims, metric, dtype))
}

/// 只读头部 peek：返回 `(dims, metric, dtype, quantization)`，供恢复侧按写入侧
/// 同一 dtype + 量化方式构造后端（int8 集合必须显式建 int8 图，缺失即静默分叉）。
/// EN: Header-only peek returning `(dims, metric, dtype, quantization)` so
/// restore can build the backend with the writer's dtype and quantization
/// (an int8 collection must construct an int8 graph; a missing value would
/// silently diverge).
pub fn peek_graph_snapshot_dims_metric_dtype_quantization(
    payload: &[u8],
) -> Result<(usize, DistanceMetric, VectorDtype, Quantization), GraphSnapshotError> {
    let mut r = Reader::new(payload);
    // 短于 magic 的输入判 BadMagic 而非 Truncated：peek 的职责是识别文件类型，
    // 连魔数都凑不齐的文件按「不是本格式」归类，调用方日志更准确。
    // EN: Input shorter than magic yields BadMagic, not Truncated: peek identifies the file type;
    // EN: a file that cannot even complete the magic is classified as not-this-format for better logs.
    if payload.len() < SNAPSHOT_MAGIC.len() || r.take(SNAPSHOT_MAGIC.len())? != SNAPSHOT_MAGIC {
        return Err(GraphSnapshotError::BadMagic);
    }
    let version = r.u32()?;
    if !is_supported_version(version) {
        return Err(GraphSnapshotError::UnsupportedVersion(version));
    }
    let dims = r.u32()? as usize;
    let metric_byte = r.u8()?;
    let metric = DistanceMetric::from_snapshot_byte(metric_byte)
        .ok_or(GraphSnapshotError::InvalidMetric(metric_byte))?;
    let flags = r.u8()?;
    validate_flags(version, flags)?;
    let dtype = if flags & FLAG_VECTOR_F16 != 0 {
        VectorDtype::F16
    } else {
        VectorDtype::F32
    };
    let quantization = quantization_from_flags(flags);
    Ok((dims, metric, dtype, quantization))
}

/// 单条向量记录的字节宽度（编码端估算容量与解码端预分配对账共用）。
/// EN: Byte width of one vector record, shared by encode capacity estimation
/// and decode prealloc reconciliation.
fn vector_record_len(quantization: Quantization, dims: usize) -> usize {
    match quantization {
        // id + norm + alpha + ceil(D/64) 个 u64 码字 + dims×f32 source。
        // EN: id + norm + alpha + ceil(D/64) u64 code words + dims×f32 source.
        Quantization::RaBitQ => 16 + 8 * rabitq::words_for_dims(dims) + 4 * dims,
        Quantization::Int8 => 16 + 5 * dims,
        Quantization::None => 8 + 4 * dims,
    }
}

/// 由 flags 的量化位推导图侧量化方式（bit2 int8 / bit3 rabitq；互斥已由
/// `validate_flags` 按版本钉死）。
/// EN: Derive graph quantization from the flags bits (bit2 int8 / bit3 rabitq;
/// mutual exclusion is pinned per version by `validate_flags`).
fn quantization_from_flags(flags: u8) -> Quantization {
    if flags & FLAG_VECTOR_RABITQ != 0 {
        Quantization::RaBitQ
    } else if flags & FLAG_VECTOR_INT8 != 0 {
        Quantization::Int8
    } else {
        Quantization::None
    }
}

impl HnswGraph {
    /// 序列化整图为 payload 字节。调用方需保证无并发写（持图读锁即可，
    /// client 写经 WAL/热缓冲不受阻）。
    /// EN: Serialize the whole graph. Caller must exclude concurrent mutation
    /// (a graph read lock suffices).
    pub(crate) fn encode_snapshot(&self) -> Vec<u8> {
        let (l0_mult, upper_mult) = crate::hnsw_graph::effective_layer_multipliers();
        let link_free_pairs: usize = self.link_free.values().map(Vec::len).sum();
        // 每条向量记录：v1/v2 为 id + dims×f32；v3 再加 scale/bias + dims×i8；
        // v4 为 norm/alpha + ceil(D/64) 个符号码字 + dims×f32 source。
        // EN: Per-vector record: v1/v2 = id + dims×f32; v3 adds scale/bias and
        // dims×i8; v4 carries norm/alpha + ceil(D/64) sign words + dims×f32 source.
        let vector_record_len = vector_record_len(self.quantization, self.dims);
        let est = HEADER_LEN
            + self.nodes.len() * NODE_RECORD_LEN
            + self.link_arena.len() * 8
            + link_free_pairs * 16
            + self.vectors.len() * vector_record_len;
        let mut w = Writer::with_capacity(est);

        // 版本随图侧量化与 dtype 决定：int8 写 v3、rabitq 写 v4（旧二进制对
        // 两者都必须显式拒绝）；未量化时 f32 写 v1（旧二进制可冷启动读图）、
        // f16 写 v2。新增量化方式或 dtype 必须在此扩展显式分支，编译器会强制
        // 同步版本/标志语义。
        // EN: The version follows graph quantization and dtype: int8 writes v3
        // and rabitq writes v4 (older binaries must reject both explicitly);
        // unquantized f32 writes v1 (older binaries can still cold-start) and
        // f16 writes v2. A new quantization or dtype must extend this explicit
        // branch, forcing version/flag semantics to be updated together.
        let version = match (self.quantization, self.dtype) {
            (Quantization::RaBitQ, _) => SNAPSHOT_VERSION_RABITQ,
            (Quantization::Int8, _) => SNAPSHOT_VERSION_INT8,
            (Quantization::None, VectorDtype::F32) => SNAPSHOT_VERSION_F32,
            (Quantization::None, VectorDtype::F16) => SNAPSHOT_VERSION_DTYPE,
        };

        // Header
        w.raw(SNAPSHOT_MAGIC);
        w.u32(version);
        w.u32(self.dims as u32);
        w.u8(self.metric.to_snapshot_byte());
        // entry_point 存在性进 flags：u64::MAX 是合法 VectorId，不能当哨兵；
        // bit1 记录向量 dtype（vectors 段始终写 f32 字节，解码时按该位下转）。
        // EN: entry_point presence goes into flags: u64::MAX is a valid VectorId, not a sentinel;
        // EN: bit1 records the vector dtype (the vectors section always writes f32 bytes and
        // EN: down-converts according to this bit on decode).
        let mut flags = 0u8;
        if self.entry_point.is_some() {
            flags |= FLAG_HAS_ENTRY_POINT;
        }
        if self.dtype == VectorDtype::F16 {
            flags |= FLAG_VECTOR_F16;
        }
        if self.quantization.is_int8() {
            flags |= FLAG_VECTOR_INT8;
        }
        if self.quantization.is_rabitq() {
            flags |= FLAG_VECTOR_RABITQ;
        }
        w.u8(flags);
        w.u16(0); // reserved
        w.u32(self.m as u32);
        w.u32(self.ef_construction as u32);
        w.u32(self.ef_search as u32);
        w.u32(self.num_layers as u32);
        w.u64(self.seed);
        w.f64_bits(l0_mult);
        w.f64_bits(upper_mult);
        w.u64(self.nodes.len() as u64);
        w.u64(self.link_arena.len() as u64);
        w.u64(link_free_pairs as u64);
        w.u64(self.vectors.len() as u64);
        w.u64(self.entry_point.unwrap_or(0));
        w.u32(self.max_layer as u32);
        w.u32(0); // reserved

        // Nodes（AHashMap 迭代顺序不确定，但内容全集确定；恢复后图等价与顺序无关）
        // EN: Nodes (AHashMap order is unspecified but the content set is fixed; equivalence is order-free).
        for (id, node) in &self.nodes {
            w.u64(*id);
            w.u64(node.link_base as u64);
            w.u8(node.max_layer);
            for len in node.layer_len {
                w.u16(len);
            }
        }

        // 链接 arena。
        // EN: Link arena.
        for id in &self.link_arena {
            w.u64(*id);
        }

        // Link free（拍平为 (slots, base) 对）
        // EN: Link free (flattened into (slots, base) pairs).
        for (slots, bases) in &self.link_free {
            for base in bases {
                w.u64(*slots as u64);
                w.u64(*base as u64);
            }
        }

        // Vectors。v1/v2 存归一化后的存储值（payload 统一写 f32 字节，f16 在此
        // 提升回 f32）。v3 额外前置 scale/bias 与 i8 codes，并追加**原始** source
        // （非反量化近似值）——重启后的精确 rescore 必须基于原始主表值。
        // EN: Vectors. v1/v2 hold normalized stored values (the payload always
        // writes f32 bytes; f16 is lifted here). v3 prepends scale/bias and the
        // i8 codes, then appends the **raw** source (not the dequantized
        // approximation) — exact rescoring after restart needs raw main-store
        // values.
        for (id, vec) in &self.vectors {
            w.u64(*id);
            debug_assert_eq!(vec.dims(), self.dims);
            match vec {
                GraphVector::Full(stored) => match stored.as_view() {
                    VectorRef::F32(v) => {
                        for x in v {
                            w.f32_bits(*x);
                        }
                    }
                    VectorRef::F16(v) => {
                        for x in v {
                            w.f32_bits(x.to_f32());
                        }
                    }
                },
                GraphVector::Int8 { codes, source } => {
                    w.f32_bits(codes.scale);
                    w.f32_bits(codes.bias);
                    for &c in &codes.codes {
                        w.u8(c as u8);
                    }
                    match source.as_view() {
                        VectorRef::F32(v) => {
                            for x in v {
                                w.f32_bits(*x);
                            }
                        }
                        VectorRef::F16(v) => {
                            for x in v {
                                w.f32_bits(x.to_f32());
                            }
                        }
                    }
                }
                GraphVector::RabitQ { codes, source } => {
                    debug_assert_eq!(
                        codes.codes.len(),
                        rabitq::words_for_dims(self.dims),
                        "rabitq code words must match ceil(D/64)"
                    );
                    w.f32_bits(codes.norm);
                    w.f32_bits(codes.alpha);
                    for &word in &codes.codes {
                        w.u64(word);
                    }
                    match source.as_view() {
                        VectorRef::F32(v) => {
                            for x in v {
                                w.f32_bits(*x);
                            }
                        }
                        VectorRef::F16(v) => {
                            for x in v {
                                w.f32_bits(x.to_f32());
                            }
                        }
                    }
                }
            }
        }

        w.buf
    }

    /// 从 payload 恢复整图；`expected` 结构期望不符即拒绝。
    /// 返回图与消费字节数（调用方可在其后拼接自有 section，如墓碑）。
    /// EN: Restore a graph from payload; rejects on structural mismatch.
    /// Returns the graph and bytes consumed (callers may append own sections).
    pub(crate) fn decode_snapshot(
        bytes: &[u8],
        expected: &SnapshotExpectations,
    ) -> Result<(Self, usize), GraphSnapshotError> {
        let mut r = Reader::new(bytes);

        // Header
        if r.take(SNAPSHOT_MAGIC.len())? != SNAPSHOT_MAGIC {
            return Err(GraphSnapshotError::BadMagic);
        }
        let version = r.u32()?;
        if !is_supported_version(version) {
            return Err(GraphSnapshotError::UnsupportedVersion(version));
        }
        let dims = r.u32()? as usize;
        let metric_byte = r.u8()?;
        let metric = DistanceMetric::from_snapshot_byte(metric_byte)
            .ok_or(GraphSnapshotError::InvalidMetric(metric_byte))?;
        let flags = r.u8()?;
        validate_flags(version, flags)?;
        let _reserved = r.u16()?;
        // dtype 由 flags bit1 表达；缺席即 f32，旧快照天然落在 f32。
        // EN: dtype comes from flags bit1; absent means f32, which old snapshots
        // naturally satisfy.
        let dtype = if flags & FLAG_VECTOR_F16 != 0 {
            VectorDtype::F16
        } else {
            VectorDtype::F32
        };
        // 图侧量化同样由 flags 表达；v3 必有 int8 位、v4 必有 rabitq 位
        // （validate_flags 已钉死）。
        // EN: Graph quantization is also expressed by flags; v3 always carries
        // the int8 bit and v4 the rabitq bit (validate_flags pins both).
        let quantization = quantization_from_flags(flags);
        let m = r.u32()? as usize;
        let ef_construction = r.u32()? as usize;
        let ef_search = r.u32()? as usize;
        let num_layers = r.u32()? as usize;
        let seed = r.u64()?;
        let l0_mult = r.f64_bits()?;
        let upper_mult = r.f64_bits()?;
        let node_count = r.u64()? as usize;
        let arena_len = r.u64()? as usize;
        let link_free_pair_count = r.u64()? as usize;
        let vector_count = r.u64()? as usize;
        let entry_point_raw = r.u64()?;
        let max_layer = r.u32()? as usize;
        let _reserved2 = r.u32()?;

        // 结构期望校验（硬拒绝项）
        // EN: Structural expectation checks (hard rejections).
        if dims != expected.dims {
            return Err(GraphSnapshotError::Mismatch("dims"));
        }
        if metric != expected.metric {
            return Err(GraphSnapshotError::Mismatch("metric"));
        }
        if dtype != expected.dtype {
            return Err(GraphSnapshotError::Mismatch("vector_dtype"));
        }
        if quantization != expected.quantization {
            return Err(GraphSnapshotError::Mismatch("quantization"));
        }
        if num_layers != expected.num_layers {
            return Err(GraphSnapshotError::Mismatch("num_layers"));
        }
        // 乘数决定 link_base 布局解释，必须位级一致（配置值经同一代码路径写入）。
        // EN: Multipliers govern link_base layout interpretation and must match bit-for-bit.
        if l0_mult.to_bits() != expected.l0_mult.to_bits()
            || upper_mult.to_bits() != expected.upper_mult.to_bits()
        {
            return Err(GraphSnapshotError::Mismatch("layer_multipliers"));
        }
        if num_layers == 0 || num_layers > MAX_HNSW_LAYERS {
            return Err(GraphSnapshotError::InvalidStructure("num_layers"));
        }
        // m 是布局解释参数（block_size/link_cap），restore 与 new 一样钳制到 >= 2；
        // 校验用原始 m、构图用钳制 m 会产生布局错位（OCR ae8d7e3），故低于钳制
        // 下限直接拒绝（调用方全量重放）。ef_* 只是运行参数，0→1 钳制无害，不拒。
        // EN: m is a layout parameter (block_size/link_cap); restore clamps to >= 2 like new.
        // EN: Validating raw m while building with clamped m misaligns the layout (OCR ae8d7e3),
        // EN: so values below the clamp floor are rejected (caller falls back to full replay).
        // EN: ef_* are runtime params; clamping 0→1 is harmless, so they are not rejected.
        if m < 2 {
            return Err(GraphSnapshotError::InvalidStructure("m"));
        }
        if max_layer >= num_layers && node_count > 0 {
            return Err(GraphSnapshotError::InvalidStructure("max_layer"));
        }
        if node_count != vector_count {
            return Err(GraphSnapshotError::InvalidStructure("vector_count"));
        }
        // 预分配守卫：header 声明的各段长度先与剩余字节对账（checked 算术），
        // 否则伪造的巨型 count 会把 with_capacity 变成 OOM/abort，而非干净的
        // Truncated 拒绝（OCR ae8d7e3）。remaining 可以更大（包装层可追加
        // 墓碑 trailer），故只比下界。
        // EN: Preallocation guard: reconcile declared section lengths with remaining bytes (checked math),
        // EN: otherwise forged giant counts turn with_capacity into OOM/abort instead of a clean
        // EN: Truncated rejection (OCR ae8d7e3). remaining may be larger (the wrapper appends a
        // EN: tombstone trailer), so only the lower bound is checked.
        // 记录宽度由图侧量化决定（v1/v2 无量化、v3 int8、v4 RaBitQ）；
        // header 的 dims 反算出 D = next_power_of_two，再得码字数。
        // EN: The record width follows graph quantization (v1/v2 plain, v3 int8,
        // v4 RaBitQ); D = next_power_of_two(dims) from the header yields the
        // word count.
        let vector_record_len = vector_record_len(quantization, dims);
        let declared = node_count
            .checked_mul(NODE_RECORD_LEN)
            .and_then(|n| arena_len.checked_mul(8).and_then(|a| n.checked_add(a)))
            .and_then(|n| {
                link_free_pair_count
                    .checked_mul(16)
                    .and_then(|p| n.checked_add(p))
            })
            .and_then(|n| {
                vector_count
                    .checked_mul(vector_record_len)
                    .and_then(|v| n.checked_add(v))
            });
        if declared.is_none_or(|need| need > r.remaining()) {
            return Err(GraphSnapshotError::Truncated { at: r.position() });
        }

        // Nodes
        let mut nodes: AHashMap<VectorId, HnswNode> = AHashMap::with_capacity(node_count);
        for _ in 0..node_count {
            let id = r.u64()?;
            let link_base = r.u64()? as usize;
            let node_max_layer = r.u8()?;
            let mut layer_len = [0u16; MAX_HNSW_LAYERS];
            for slot in layer_len.iter_mut() {
                *slot = r.u16()?;
            }
            if node_max_layer as usize >= num_layers {
                return Err(GraphSnapshotError::InvalidStructure("node.max_layer"));
            }
            // 块边界校验：base + block_size 不得越出 arena。
            // EN: Block bounds check: base + block_size must not exceed the arena.
            let block = crate::hnsw_graph::block_size_for_node(m, node_max_layer as usize);
            if link_base
                .checked_add(block)
                .is_none_or(|end| end > arena_len)
            {
                return Err(GraphSnapshotError::InvalidStructure("node.link_base"));
            }
            // 每层用量不得超过容量。
            // EN: Per-layer usage must not exceed capacity.
            for (l, len) in layer_len
                .iter()
                .enumerate()
                .take(node_max_layer as usize + 1)
            {
                if *len as usize > crate::hnsw_graph::link_cap(m, l) {
                    return Err(GraphSnapshotError::InvalidStructure("node.layer_len"));
                }
            }
            if nodes
                .insert(
                    id,
                    HnswNode {
                        link_base,
                        max_layer: node_max_layer,
                        layer_len,
                    },
                )
                .is_some()
            {
                return Err(GraphSnapshotError::InvalidStructure("node.duplicate"));
            }
        }

        // 链接 arena。
        // EN: Link arena.
        let mut link_arena = Vec::with_capacity(arena_len);
        for _ in 0..arena_len {
            link_arena.push(r.u64()?);
        }

        // Link free：base 必须全局唯一——合法编码器每块只释放一次，重复 base 只可能
        // 来自伪造内容；放过则 alloc_link_block 会把同一 arena 区间发给两个节点，
        // 静默别名互踩（OCR ae8d7e3）。
        // EN: Link free: base must be globally unique — a valid encoder frees each block once; duplicates
        // EN: can only be forged. Otherwise alloc_link_block hands the same arena range to two nodes,
        // EN: silently aliasing them (OCR ae8d7e3).
        let mut link_free: AHashMap<usize, Vec<usize>> = AHashMap::new();
        let mut seen_bases = ahash::AHashSet::with_capacity(link_free_pair_count);
        for _ in 0..link_free_pair_count {
            let slots = r.u64()? as usize;
            let base = r.u64()? as usize;
            if base.checked_add(slots).is_none_or(|end| end > arena_len) {
                return Err(GraphSnapshotError::InvalidStructure("link_free.base"));
            }
            if !seen_bases.insert(base) {
                return Err(GraphSnapshotError::InvalidStructure("link_free.duplicate"));
            }
            link_free.entry(slots).or_default().push(base);
        }

        // Vectors：v1/v2 的 f32 字节按 flags 记录的 dtype 一次性下转；v3 还原
        // int8 codes + 原始 source（source 同样按 dtype 下转）。
        // EN: Vectors: v1/v2 f32 bytes are down-converted once to the dtype in
        // flags; v3 restores int8 codes plus the raw source (also down-converted
        // to dtype).
        let mut vectors: AHashMap<VectorId, GraphVector> = AHashMap::with_capacity(vector_count);
        for _ in 0..vector_count {
            let id = r.u64()?;
            if !nodes.contains_key(&id) {
                return Err(GraphSnapshotError::InvalidStructure("vector.orphan"));
            }
            let entry = match quantization {
                Quantization::RaBitQ => {
                    let norm = r.f32_bits()?;
                    let alpha = r.f32_bits()?;
                    // 非有限的 norm/alpha 会让图内距离产出 NaN 并污染堆序；
                    // alpha ≤ 0 会让 `est_cos = b / alpha` 反向或爆炸。合法编码器
                    // （`RaBitQVector::from_rotated_unit` / 零范数哨兵）不可能产出，
                    // 按伪造拒绝。
                    // EN: Non-finite norm/alpha make in-graph distances NaN and
                    // corrupt heap ordering; alpha <= 0 flips or blows up
                    // `est_cos = b / alpha`. A valid encoder
                    // (`RaBitQVector::from_rotated_unit` / zero-norm sentinel)
                    // never produces them, so reject as forged.
                    if !norm.is_finite() || norm < 0.0 || !alpha.is_finite() || alpha <= 0.0 {
                        return Err(GraphSnapshotError::InvalidStructure("vector.quantization"));
                    }
                    let words = rabitq::words_for_dims(dims);
                    let mut codes = Vec::with_capacity(words);
                    for _ in 0..words {
                        codes.push(r.u64()?);
                    }
                    let mut source = Vec::with_capacity(dims);
                    for _ in 0..dims {
                        source.push(r.f32_bits()?);
                    }
                    GraphVector::RabitQ {
                        codes: Arc::new(RaBitQVector { codes, alpha, norm }),
                        source: Arc::new(StoredVector::from_f32(source, dtype)),
                    }
                }
                Quantization::Int8 => {
                    let scale = r.f32_bits()?;
                    let bias = r.f32_bits()?;
                    // 非有限的 scale/bias 会让图内距离产出 NaN 并污染堆序，
                    // 合法编码器（`Int8Vector::quantize`）不可能产出，按伪造拒绝。
                    // EN: Non-finite scale/bias make in-graph distances NaN and
                    // corrupt heap ordering; a valid encoder
                    // (`Int8Vector::quantize`) never produces them, so reject as forged.
                    if !scale.is_finite() || !bias.is_finite() {
                        return Err(GraphSnapshotError::InvalidStructure("vector.quantization"));
                    }
                    let mut codes = Vec::with_capacity(dims);
                    for _ in 0..dims {
                        codes.push(r.u8()? as i8);
                    }
                    let mut source = Vec::with_capacity(dims);
                    for _ in 0..dims {
                        source.push(r.f32_bits()?);
                    }
                    GraphVector::Int8 {
                        codes: Arc::new(Int8Vector { codes, scale, bias }),
                        source: Arc::new(StoredVector::from_f32(source, dtype)),
                    }
                }
                Quantization::None => {
                    let mut v = Vec::with_capacity(dims);
                    for _ in 0..dims {
                        v.push(r.f32_bits()?);
                    }
                    GraphVector::Full(Arc::new(StoredVector::from_f32(v, dtype)))
                }
            };
            // 重复 id 会静默覆盖并破坏 node/vector 双射（漏检的节点从 main_index
            // 缺席、被搜索静默跳过），与 nodes 段同纪律拒绝（OCR ae8d7e3）。
            // EN: Duplicate ids would silently overwrite and break the node/vector bijection (missing
            // EN: nodes are absent from main_index and skipped by search); rejected like the nodes section.
            if vectors.insert(id, entry).is_some() {
                return Err(GraphSnapshotError::InvalidStructure("vector.duplicate"));
            }
        }

        // entry_point 存在性由 flags 表达（u64::MAX 哨兵会与合法 id 碰撞）。
        // EN: entry_point presence is expressed by flags (a u64::MAX sentinel collides with valid ids).
        let entry_point = if flags & FLAG_HAS_ENTRY_POINT != 0 {
            if !nodes.contains_key(&entry_point_raw) {
                return Err(GraphSnapshotError::InvalidStructure("entry_point"));
            }
            Some(entry_point_raw)
        } else {
            None
        };
        // 双向不变量：空图必无 entry、非空图必有 entry（合法编码器不会产出
        // 后者；放过则下一次 insert 重置 max_layer，破坏「entry 必为顶层节点」）。
        // EN: Bidirectional invariant: an empty graph has no entry, a non-empty graph must have one
        // EN: (valid encoders never produce the latter; allowing it breaks entry-is-top-layer on insert).
        if nodes.is_empty() != entry_point.is_none() {
            return Err(GraphSnapshotError::InvalidStructure("entry_point"));
        }

        let graph = Self::restore_from_snapshot(SnapshotGraphParts {
            dims,
            metric,
            dtype,
            quantization,
            m,
            ef_construction,
            ef_search,
            num_layers,
            seed,
            nodes,
            link_arena,
            link_free,
            vectors,
            entry_point,
            max_layer,
        });
        Ok((graph, r.position()))
    }
}

// ---------------------------------------------------------------------------
// HotBufferIndex 包装层：图 payload + 墓碑 trailer。
// Tombstones MUST travel with the graph — they carry delete semantics.
// ---------------------------------------------------------------------------

use crate::HotBufferIndex;
use std::sync::atomic::Ordering;

/// 墓碑 trailer：count u64 + ids×u64，追加在图 payload 之后。
/// EN: Tombstone trailer appended after the graph payload.
impl HotBufferIndex {
    /// 编码整索引快照（图 + 墓碑）。调用前必须先 drain 构建队列
    /// （hot_buffer/builder_queue 不入快照）；快照期间持图读锁，client 写
    /// 经 WAL + 热缓冲照常进行，不受阻。
    /// EN: Encode the index snapshot (graph + tombstones). Drain the builder
    /// queue first — hot buffer / builder queue are not snapshotted. Holds the
    /// graph read lock; client writes proceed via WAL + hot buffer.
    pub async fn encode_graph_snapshot(&self) -> Vec<u8> {
        // 锁序 ts → hnsw，与全局锁序（ts → main → hnsw）一致；main 无需加锁，
        // 图内 vectors 即权威副本（merge 路径 main/图键集等价）。
        // EN: Lock order ts → hnsw matches the global order (ts → main → hnsw); main needs no lock,
        // EN: since the graph's vectors are the authoritative copy (main/graph keys match after merge).
        //
        // 一致性切点：ts 读锁在 async 侧全程持有（delete 路径先取 ts 写锁，
        // 故被挡在切点之外），图读锁在 spawn_blocking 内获取（挡住 merge）——
        // 与改造前的双锁持有语义完全相同。
        // EN: Consistency cut: the ts read lock is held for the whole async side (delete takes the ts
        // EN: write lock first, so it stays outside the cut); the graph read lock is taken inside
        // EN: spawn_blocking (blocking merge) — identical dual-lock semantics as before.
        //
        // 为何 spawn_blocking（OCR ae8d7e3）：encode 是 O(图) 纯 CPU，大图可耗
        // 时数百毫秒，在 tokio worker 上同步执行会停住整条 worker；周期 checkpoint
        // 场景必须挪到 blocking 池。
        // EN: Why spawn_blocking (OCR ae8d7e3): encode is O(graph) pure CPU and can take hundreds of
        // EN: milliseconds on large graphs; running it on a tokio worker stalls the whole worker, so
        // EN: periodic checkpoints must move to the blocking pool.
        let ts = self.tombstones.read().await;
        let hnsw = Arc::clone(&self.hnsw);
        let mut bytes = tokio::task::spawn_blocking(move || {
            let graph = crate::hnsw_read_guard(&hnsw);
            graph.encode_snapshot()
        })
        .await
        .expect("graph snapshot encode task panicked");
        bytes.extend_from_slice(&(ts.len() as u64).to_be_bytes());
        for id in ts.iter() {
            bytes.extend_from_slice(&id.to_be_bytes());
        }
        bytes
    }

    /// 从快照恢复整索引。仅允许在**空索引**上调用（启动恢复期）；
    /// 结构期望（dims/metric/num_layers/乘数）不符即拒绝，调用方退化为
    /// 全量 WAL 重放。恢复后 `hnsw_m/ef_*` 原子量同步为快照值。
    /// EN: Restore the index from a snapshot. Empty-target only (startup
    /// recovery); rejects on structural mismatch — caller falls back to full
    /// WAL replay. Syncs the `hnsw_*` atomics to snapshot values.
    pub async fn restore_graph_snapshot(&self, bytes: &[u8]) -> Result<(), GraphSnapshotError> {
        // 空目标守卫：恢复是启动期操作，覆盖非空索引必是调用错误。
        // EN: Empty-target guard: restore is a startup operation; overwriting a non-empty index is a bug.
        {
            let main = self.main_index.read().await;
            let graph = crate::hnsw_read_guard(&self.hnsw);
            if !graph.is_empty() || !main.is_empty() {
                return Err(GraphSnapshotError::InvalidStructure("non_empty_target"));
            }
        }
        let expected = {
            let graph = crate::hnsw_read_guard(&self.hnsw);
            SnapshotExpectations::for_graph(&graph)
        };
        let (graph, consumed) = HnswGraph::decode_snapshot(bytes, &expected)?;

        // 墓碑 trailer（同样先对账再预分配，防伪造 count 变 OOM）。
        // EN: Tombstone trailer (also reconcile before preallocating, so forged counts cannot OOM).
        let mut r = Reader::new(&bytes[consumed..]);
        let tombstone_count = r.u64()? as usize;
        if tombstone_count
            .checked_mul(8)
            .is_none_or(|need| need > r.remaining())
        {
            return Err(GraphSnapshotError::Truncated { at: r.position() });
        }
        let mut tombstones = ahash::AHashSet::with_capacity(tombstone_count);
        for _ in 0..tombstone_count {
            tombstones.insert(r.u64()?);
        }
        if r.remaining() != 0 {
            return Err(GraphSnapshotError::TrailingBytes {
                extra: r.remaining(),
            });
        }

        // 锁序 ts → main → hnsw（与 remove/merge 一致）；一次性备齐后替换，
        // 临界区内无 await（main/ts 的 tokio 写锁先行获取，parking_lot 图锁最后）。
        // EN: Lock order ts → main → hnsw (same as remove/merge); acquire all, then replace, with no
        // EN: await in the critical section (tokio write locks for main/ts first, parking_lot graph last).
        let mut ts = self.tombstones.write().await;
        let mut main = self.main_index.write().await;
        let mut slot = crate::hnsw_write_guard(&self.hnsw);

        // 空目标守卫复查（OCR ae8d7e3）：开场的读锁检查与此处之间存在
        // check-then-act 窗口，写入者溜进来时 main.clear()+图替换会静默丢写、
        // 原子量也已被快照值覆盖——写锁临界区内、原子量 store 之前再查一次。
        // EN: Empty-target recheck (OCR ae8d7e3): a check-then-act window exists between the opening
        // EN: read-lock check and here; a sneaking writer would lose data to main.clear()+swap and
        // EN: have its atomics overwritten — so recheck inside the write critical section before stores.
        if !slot.is_empty() || !main.is_empty() {
            return Err(GraphSnapshotError::InvalidStructure("non_empty_target"));
        }

        // 原子量先同步快照值，再落图——恢复后任何并发读到的参数与图一致。
        // EN: Sync atomics to snapshot values before installing the graph, so concurrent readers stay consistent.
        self.hnsw_m.store(graph.m, Ordering::Release);
        self.hnsw_ef_construction
            .store(graph.ef_construction, Ordering::Release);
        self.hnsw_ef_search
            .store(graph.ef_search, Ordering::Release);

        // 主表重建必须用原始 source（int8 图侧只是 codes，主表要精确 rescore
        // 原始值）；未量化图 source 即图向量本身，`Arc` 共享关系不变。
        // EN: The main map must rebuild from the raw source (an int8 graph only
        // holds codes while rescoring needs raw values); for unquantized graphs
        // source is the graph vector itself, so `Arc` sharing is unchanged.
        main.clear();
        for (id, v) in &graph.vectors {
            main.insert(*id, v.source_arc());
        }
        *ts = tombstones;
        *slot = graph;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_vector(dims: usize, seed: u64) -> Vec<f32> {
        (0..dims)
            .map(|i| ((seed as usize * 31 + i * 17 + 7) % 1000) as f32 / 1000.0)
            .collect()
    }

    fn build_graph(n: usize, dims: usize, m: usize, max_layers: usize) -> HnswGraph {
        let mut g = HnswGraph::new(dims, DistanceMetric::L2, m, 32, 16, 42, max_layers);
        for i in 0..n {
            let id = i as u64 + 1;
            g.insert(id, gen_vector(dims, id));
        }
        g
    }

    fn expectations_for(g: &HnswGraph) -> SnapshotExpectations {
        SnapshotExpectations::for_graph(g)
    }

    fn assert_graphs_equivalent(a: &HnswGraph, b: &HnswGraph) {
        assert_eq!(a.len(), b.len(), "node count");
        assert_eq!(a.edge_count_hint(), b.edge_count_hint(), "edge count");
        assert_eq!(a.entry_point(), b.entry_point(), "entry point");
        assert_eq!(a.max_layer(), b.max_layer(), "max layer");
        let mut av = a.export_vectors();
        av.sort_by_key(|(id, _)| *id);
        let mut bv = b.export_vectors();
        bv.sort_by_key(|(id, _)| *id);
        assert_eq!(av.len(), bv.len(), "vector count");
        for ((aid, avec), (bid, bvec)) in av.iter().zip(bv.iter()) {
            assert_eq!(aid, bid);
            // 快照按 f32 字节往返，f16 图两侧提升后应逐位一致。
            // EN: Snapshots roundtrip f32 bytes; lifting both sides of an f16 graph
            // must match bit-for-bit.
            assert_eq!(
                avec.to_f32_vec(),
                bvec.to_f32_vec(),
                "vector bits for {aid}"
            );
            let level = a.node_level(*aid).expect("node level");
            assert_eq!(Some(level), b.node_level(*bid), "node level for {aid}");
            for layer in 0..=level {
                assert_eq!(
                    a.neighbors_at_layer(*aid, layer),
                    b.neighbors_at_layer(*bid, layer),
                    "neighbors of {aid} at layer {layer}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_empty_graph() {
        let g = build_graph(0, 8, 4, 8);
        let bytes = g.encode_snapshot();
        let (restored, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.len(), 0);
        assert_eq!(restored.entry_point(), None);
    }

    #[test]
    fn roundtrip_graph_with_inserts_including_upper_layers() {
        let g = build_graph(200, 8, 4, 8);
        // 确保测试图确实含上层节点（否则多层路径未被覆盖）。
        // EN: Ensure the test graph really has upper-layer nodes (otherwise multi-layer paths are uncovered).
        let max_level = (1..=200u64)
            .filter_map(|id| g.node_level(id))
            .max()
            .expect("levels");
        assert!(max_level >= 1, "test graph must contain upper-layer nodes");
        let bytes = g.encode_snapshot();
        let (restored, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, bytes.len());
        assert_graphs_equivalent(&g, &restored);
    }

    /// f16 图快照（P1.4）：flags bit1 置位、peek 返回 f16、f32 线格式往返后
    /// 两侧提升值逐位一致（f16 值在 f32 中可精确表示）。
    /// EN: f16 graph snapshot (P1.4): flags bit1 is set, peek reports f16, and the
    /// f32 wire format roundtrips with bit-identical lifted values.
    #[test]
    fn roundtrip_f16_graph_preserves_dtype_and_vectors() {
        let dims = 8;
        let mut g =
            HnswGraph::new_with_dtype(dims, DistanceMetric::L2, VectorDtype::F16, 4, 32, 16, 42, 8);
        for i in 0..120u64 {
            g.insert(i + 1, gen_vector(dims, i + 1));
        }
        let bytes = g.encode_snapshot();
        // f16 快照写 v2（旧二进制必须显式拒绝）；flags 位于头偏移 17，bit1 = f16。
        // EN: f16 snapshots are written as v2 (older binaries must reject
        // explicitly); flags live at header offset 17, bit1 means f16.
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            SNAPSHOT_VERSION_DTYPE
        );
        assert_eq!(bytes[17] & FLAG_VECTOR_F16, FLAG_VECTOR_F16);
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype(&bytes).expect("peek"),
            (dims, DistanceMetric::L2, VectorDtype::F16)
        );
        let (restored, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.dtype(), VectorDtype::F16);
        assert_graphs_equivalent(&g, &restored);
    }

    /// 默认 f32 快照不受 f16 位影响；peek 明确报告 f32（旧快照兼容口径）。
    /// EN: Default f32 snapshots are untouched by the f16 bit; peek reports f32
    /// explicitly (the old-snapshot compatibility contract).
    #[test]
    fn f32_snapshot_peek_reports_f32() {
        let g = build_graph(32, 8, 4, 8);
        let bytes = g.encode_snapshot();
        // f32 保持 v1，旧二进制仍能冷启动读图。
        // EN: f32 stays v1 so older binaries can still cold-start from the graph.
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            SNAPSHOT_VERSION_F32
        );
        assert_eq!(bytes[17] & FLAG_VECTOR_F16, 0);
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype(&bytes).expect("peek"),
            (8, DistanceMetric::L2, VectorDtype::F32)
        );
    }

    /// 版本/标志门禁：v1 携带 f16 位与任意未知 flag 位都必须 fail-fast
    /// （decode 与 peek 口径一致），不得静默按无 dtype 语义降级。
    /// EN: Version/flag gates: v1 carrying the f16 bit and any unknown flag bit
    /// must fail fast (decode and peek agree), never silently downgrade to
    /// dtype-free semantics.
    #[test]
    fn version_and_flag_gates_reject_downgrade_and_unknown_bits() {
        let g32 = build_graph(8, 8, 4, 8);
        let mut g16 =
            HnswGraph::new_with_dtype(8, DistanceMetric::L2, VectorDtype::F16, 4, 32, 16, 42, 8);
        g16.insert(1, gen_vector(8, 1));
        let bytes16 = g16.encode_snapshot();

        // 把 v2 头改回 v1：f16 位在 v1 下无定义，必须拒绝。
        // EN: Rewrite the v2 header as v1: the f16 bit is undefined under v1
        // and must be rejected.
        let mut forged_v1 = bytes16.clone();
        forged_v1[8..12].copy_from_slice(&SNAPSHOT_VERSION_F32.to_be_bytes());
        assert_eq!(
            HnswGraph::decode_snapshot(&forged_v1, &expectations_for(&g32)).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype(&forged_v1).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );

        // v2 的未知位（bit2）同样拒绝。
        // EN: Unknown bits (bit2) are rejected on v2 as well.
        let mut unknown_bit = bytes16.clone();
        unknown_bit[17] |= 0x04;
        assert_eq!(
            HnswGraph::decode_snapshot(&unknown_bit, &expectations_for(&g16)).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype(&unknown_bit).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );
    }

    /// dtype 是恢复的硬期望：f16 快照不得写入 f32 目标（否则静默按错误精度
    /// 解释向量，后续检索与快照改写都会漂移）。
    /// EN: dtype is a hard restore expectation: an f16 snapshot must not land in
    /// an f32 target, which would silently reinterpret vectors at the wrong
    /// precision and drift both search and later re-encoded snapshots.
    #[test]
    fn decode_rejects_vector_dtype_mismatch() {
        let dims = 8;
        let mut g =
            HnswGraph::new_with_dtype(dims, DistanceMetric::L2, VectorDtype::F16, 4, 32, 16, 42, 8);
        g.insert(1, gen_vector(dims, 1));
        let bytes = g.encode_snapshot();
        let f32_graph = HnswGraph::new(dims, DistanceMetric::L2, 4, 32, 16, 42, 8);
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&f32_graph)).err(),
            Some(GraphSnapshotError::Mismatch("vector_dtype"))
        );
    }

    #[test]
    fn roundtrip_after_removals_exercises_link_free() {
        let mut g = build_graph(120, 8, 4, 8);
        for id in [3u64, 17, 42, 99] {
            assert!(g.remove_node(id), "remove {id}");
        }
        let bytes = g.encode_snapshot();
        let (restored, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, bytes.len());
        assert_graphs_equivalent(&g, &restored);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let g = build_graph(4, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        bytes[0] ^= 0xFF;
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::BadMagic)
        );
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let g = build_graph(4, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        bytes[8..12].copy_from_slice(&99u32.to_be_bytes());
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn decode_rejects_invalid_metric_byte() {
        let g = build_graph(4, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        bytes[16] = 7;
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::InvalidMetric(7))
        );
    }

    #[test]
    fn decode_rejects_truncation_at_section_boundaries() {
        let g = build_graph(50, 8, 4, 8);
        let bytes = g.encode_snapshot();
        // 头内、头尾、nodes 中、arena 中、vectors 中、末尾前一字节。
        // EN: Inside header, header end, mid-nodes, mid-arena, mid-vectors, one byte before end.
        let cuts = [
            0usize,
            8,
            HEADER_LEN - 1,
            HEADER_LEN,
            HEADER_LEN + NODE_RECORD_LEN + 3,
            HEADER_LEN + 50 * NODE_RECORD_LEN + 5,
            bytes.len() - 1,
        ];
        for cut in cuts {
            assert!(
                HnswGraph::decode_snapshot(&bytes[..cut], &expectations_for(&g)).is_err(),
                "cut at {cut} must fail"
            );
        }
    }

    // ── OCR 裁决批次十四（ae8d7e3 复审）回归 ─────────────────
    // 头字段偏移（BE，与 encode 顺序一致）：flags=17、m=[20..24]、
    // node_count=[60..68]、arena_len=[68..76]、link_free_pairs=[76..84]、
    // vector_count=[84..92]。

    /// 回归（OCR HIGH）：u64::MAX 是合法 VectorId，entry_point 存在性由 flags
    /// 表达——id = u64::MAX 的节点作 entry point 必须无损 roundtrip（旧哨兵
    /// 格式会把它静默解码成 None，图变成「有节点无入口」永远搜空）。
    /// EN: Regression (OCR HIGH): u64::MAX is a valid VectorId; entry presence comes from flags —
    /// EN: a u64::MAX entry point must roundtrip losslessly (the old sentinel decoded it to None).
    /// EN: The graph then had nodes but no entry and always searched empty.
    #[test]
    fn roundtrip_entry_point_id_u64_max() {
        let mut g = HnswGraph::new(8, DistanceMetric::L2, 4, 32, 16, 42, 8);
        g.insert(u64::MAX, gen_vector(8, 1));
        assert_eq!(g.entry_point(), Some(u64::MAX), "single node is the entry");
        let bytes = g.encode_snapshot();
        let (restored, _) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(
            restored.entry_point(),
            Some(u64::MAX),
            "u64::MAX entry must survive the roundtrip"
        );
        assert_eq!(restored.len(), 1);
    }

    /// 回归（OCR）：m 低于构造钳制下限（< 2）拒绝——校验若用原始 m、构图用
    /// 钳制 m，同一 arena 会被两种布局解释（越界写/静默错位）。
    /// EN: Regression (OCR): m below the clamp floor (< 2) is rejected — raw-m validation with
    /// EN: clamped-m building interprets the arena two ways (out-of-bounds writes / silent misalignment).
    #[test]
    fn decode_rejects_m_below_clamp_floor() {
        let g = build_graph(8, 8, 4, 8);
        for bad in [0u32, 1] {
            let mut bytes = g.encode_snapshot();
            bytes[20..24].copy_from_slice(&bad.to_be_bytes());
            assert_eq!(
                HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err(),
                Some(GraphSnapshotError::InvalidStructure("m")),
                "m={bad} must be rejected"
            );
        }
    }

    /// 回归（OCR）：非空图必须有 entry point（双向不变量；放过则下一次
    /// insert 重置 max_layer，破坏「entry 必为顶层节点」）。
    /// EN: Regression (OCR): a non-empty graph must have an entry point (bidirectional invariant;
    /// EN: allowing it lets the next insert reset max_layer and breaks entry-is-top-layer).
    #[test]
    fn decode_rejects_missing_entry_point_with_nodes() {
        let g = build_graph(8, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        bytes[17] = 0; // 清 FLAG_HAS_ENTRY_POINT
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::InvalidStructure("entry_point"))
        );
    }

    /// 回归（OCR）：header 声明的段长先与剩余字节对账——伪造巨型 count 必须
    /// 是干净的 Truncated，不得驱动 with_capacity 预分配（OOM/abort）。
    /// EN: Regression (OCR): declared section lengths are reconciled with remaining bytes — forged
    /// EN: giant counts must yield a clean Truncated, never drive with_capacity (OOM/abort).
    #[test]
    fn decode_rejects_oversized_declared_counts_before_prealloc() {
        let g = build_graph(4, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        bytes[60..68].copy_from_slice(&u64::MAX.to_be_bytes()); // node_count
        bytes[84..92].copy_from_slice(&u64::MAX.to_be_bytes()); // vector_count（保持相等）
        let err = HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err();
        assert!(
            matches!(err, Some(GraphSnapshotError::Truncated { .. })),
            "giant declared counts must be a clean Truncated, got {err:?}"
        );
    }

    /// 回归（OCR）：vectors 段重复 id 拒绝——静默覆盖会破坏 node/vector 双射，
    /// 漏检节点从 main_index 缺席、被搜索静默跳过。
    /// EN: Regression (OCR): duplicate ids in vectors are rejected — silent overwrite breaks the
    /// EN: node/vector bijection; missing nodes are absent from main_index and skipped by search.
    #[test]
    fn decode_rejects_duplicate_vector_id() {
        let g = build_graph(4, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        let node_count = u64::from_be_bytes(bytes[60..68].try_into().unwrap()) as usize;
        let arena_len = u64::from_be_bytes(bytes[68..76].try_into().unwrap()) as usize;
        let lf_pairs = u64::from_be_bytes(bytes[76..84].try_into().unwrap()) as usize;
        let voff = HEADER_LEN + node_count * NODE_RECORD_LEN + arena_len * 8 + lf_pairs * 16;
        let rec = 8 + 4 * 8; // id u64 + dims(8)×f32
        let first_id: [u8; 8] = bytes[voff..voff + 8].try_into().unwrap();
        bytes[voff + rec..voff + rec + 8].copy_from_slice(&first_id);
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::InvalidStructure("vector.duplicate"))
        );
    }

    /// 回归（OCR）：link_free 重复 base 拒绝——放过则 alloc_link_block 把同一
    /// arena 区间发给两个节点，链接存储静默别名互踩。
    /// EN: Regression (OCR): duplicate link_free bases are rejected — otherwise alloc_link_block
    /// EN: gives the same arena range to two nodes, silently aliasing link storage.
    #[test]
    fn decode_rejects_duplicate_link_free_base() {
        let mut g = build_graph(6, 8, 4, 8);
        assert!(g.remove_node(5));
        assert!(g.remove_node(6));
        let mut bytes = g.encode_snapshot();
        let node_count = u64::from_be_bytes(bytes[60..68].try_into().unwrap()) as usize;
        let arena_len = u64::from_be_bytes(bytes[68..76].try_into().unwrap()) as usize;
        let lf_pairs = u64::from_be_bytes(bytes[76..84].try_into().unwrap()) as usize;
        assert!(lf_pairs >= 2, "need >= 2 free pairs, got {lf_pairs}");
        let foff = HEADER_LEN + node_count * NODE_RECORD_LEN + arena_len * 8;
        // 第二对整对（slots+base）改成第一对：只改 base 时若两对 slots 不同，
        // base+slots 越界检查会先于 dup 检查触发（AHashMap 迭代序随机→闪烁）。
        // EN: Copy the whole second pair (slots+base) over the first: changing only base can make the
        // EN: base+slots bounds check fire before the dup check (random AHashMap order → flaky).
        let first_pair: [u8; 16] = bytes[foff..foff + 16].try_into().unwrap();
        bytes[foff + 16..foff + 32].copy_from_slice(&first_pair);
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::InvalidStructure("link_free.duplicate"))
        );
    }

    #[test]
    fn peek_reads_dims_and_metric_without_full_decode() {
        let g = build_graph(16, 8, 4, 8);
        let bytes = g.encode_snapshot();
        assert_eq!(
            peek_graph_snapshot_dims_metric(&bytes),
            Ok((8, DistanceMetric::L2))
        );
        // 只需要头部：尾部截断不影响 peek。
        // EN: Only the header is needed: tail truncation does not affect peek.
        assert_eq!(
            peek_graph_snapshot_dims_metric(&bytes[..HEADER_LEN]),
            Ok((8, DistanceMetric::L2))
        );
        assert_eq!(
            peek_graph_snapshot_dims_metric(&bytes[..4]).err(),
            Some(GraphSnapshotError::BadMagic)
        );
        assert!(peek_graph_snapshot_dims_metric(&[]).is_err());
    }

    #[test]
    fn decode_reports_consumed_so_wrapper_can_detect_trailing() {
        // 图级解码允许多余字节（HotBufferIndex 会在其后拼接墓碑 section）；
        // 调用方凭 consumed 检测 trailing。包装层（HotBufferIndex）对末尾强校验。
        // EN: Graph-level decoding allows trailing bytes (HotBufferIndex appends a tombstone section);
        // EN: callers use consumed to detect trailing data. The wrapper strictly validates the end.
        let g = build_graph(4, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        let exact_len = bytes.len();
        bytes.push(0xAB);
        let (_, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, exact_len);
    }

    #[test]
    fn decode_rejects_expectation_mismatches() {
        let g = build_graph(16, 8, 4, 8);
        let bytes = g.encode_snapshot();
        let base = expectations_for(&g);

        let mut e = base;
        e.dims += 1;
        assert!(matches!(
            HnswGraph::decode_snapshot(&bytes, &e),
            Err(GraphSnapshotError::Mismatch("dims"))
        ));

        let mut e = base;
        e.metric = DistanceMetric::Cosine;
        assert!(matches!(
            HnswGraph::decode_snapshot(&bytes, &e),
            Err(GraphSnapshotError::Mismatch("metric"))
        ));

        let mut e = base;
        e.num_layers += 1;
        assert!(matches!(
            HnswGraph::decode_snapshot(&bytes, &e),
            Err(GraphSnapshotError::Mismatch("num_layers"))
        ));

        let mut e = base;
        e.l0_mult *= 2.0;
        assert!(matches!(
            HnswGraph::decode_snapshot(&bytes, &e),
            Err(GraphSnapshotError::Mismatch("layer_multipliers"))
        ));
    }

    #[test]
    fn decode_rejects_corrupt_node_link_base() {
        let g = build_graph(8, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        // 第一个节点的 link_base 字段：HEADER_LEN + 8（跳过 id）。
        // EN: First node's link_base field: HEADER_LEN + 8 (skipping the id).
        let off = HEADER_LEN + 8;
        bytes[off..off + 8].copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(matches!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)),
            Err(GraphSnapshotError::InvalidStructure("node.link_base"))
        ));
    }

    #[test]
    fn decode_rejects_orphan_vector() {
        let g = build_graph(8, 8, 4, 8);
        let mut bytes = g.encode_snapshot();
        // vectors section 起始 = HEADER + nodes + arena + link_free。
        // 篡改第一个 vector 的 id 为不存在的大 id。
        // EN: Vectors section start = HEADER + nodes + arena + link_free.
        // EN: Tamper the first vector's id into a large nonexistent id.
        let node_count = 8usize;
        let arena_len = g.link_arena.len();
        let free_pairs: usize = g.link_free.values().map(Vec::len).sum();
        let vec_off = HEADER_LEN + node_count * NODE_RECORD_LEN + arena_len * 8 + free_pairs * 16;
        bytes[vec_off..vec_off + 8].copy_from_slice(&9_999_999u64.to_be_bytes());
        assert!(matches!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)),
            Err(GraphSnapshotError::InvalidStructure("vector.orphan"))
        ));
    }

    #[test]
    fn decode_adopts_snapshot_m_and_ef_without_rejection() {
        // m/ef 是运行时参数：恢复采用快照值，不做期望匹配。
        // EN: m/ef are runtime params: restore adopts snapshot values without expectation matching.
        let g = build_graph(16, 8, 4, 8);
        let bytes = g.encode_snapshot();
        let e = expectations_for(&g);
        let (restored, _) = HnswGraph::decode_snapshot(&bytes, &e).expect("decode");
        assert_eq!(restored.m, g.m);
        assert_eq!(restored.ef_search, g.ef_search);
        assert_eq!(restored.ef_construction, g.ef_construction);
    }

    // -----------------------------------------------------------------------
    // HotBufferIndex 包装层测试。
    // EN: HotBufferIndex wrapper tests.
    // -----------------------------------------------------------------------

    use crate::{HotBufferIndex, VectorIndex};

    fn hb_vector(dims: usize, seed: u64) -> Vec<f32> {
        (0..dims)
            .map(|i| (((seed as usize * 31 + i * 17 + 7) % 1000) as f32 / 1000.0) - 0.5)
            .collect()
    }

    async fn drain_all(idx: &HotBufferIndex) {
        loop {
            let report = idx.run_async_builder_once().await;
            if report.flushed == 0 {
                break;
            }
        }
    }

    async fn build_hotbuffer(n: usize, dims: usize) -> HotBufferIndex {
        let idx = HotBufferIndex::new(dims);
        let batch: Vec<_> = (0..n)
            .map(|i| {
                let id = i as u64 + 1;
                (id, hb_vector(dims, id))
            })
            .collect();
        idx.insert_batch(batch).await.expect("insert_batch");
        drain_all(&idx).await;
        idx
    }

    #[tokio::test]
    async fn hotbuffer_roundtrip_search_equivalence() {
        let dims = 16;
        let src = build_hotbuffer(300, dims).await;
        for id in [5u64, 123, 250] {
            assert!(src.remove(id).await, "remove {id}");
        }
        let bytes = src.encode_graph_snapshot().await;

        let dst = HotBufferIndex::new(dims);
        dst.restore_graph_snapshot(&bytes).await.expect("restore");

        for q in [1u64, 77, 901] {
            let query = hb_vector(dims, q);
            let a = src
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            let b = dst
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            assert_eq!(a, b, "scored top-10 for query seed {q}");
        }
    }

    /// f16 HotBuffer 快照往返（P1.4）：dtype 经 flags 传递，恢复后检索排序一致，
    /// 导出边界提升为 f32（快照线格式与逻辑备份均为 f32）。
    /// EN: f16 HotBuffer snapshot roundtrip (P1.4): dtype travels via flags, search
    /// ranking matches after restore, and exports lift to f32 at the boundary (both
    /// the snapshot wire format and logical backup are f32).
    #[tokio::test]
    async fn hotbuffer_f16_snapshot_roundtrip_and_search_equivalence() {
        let dims = 16;
        let src =
            HotBufferIndex::new_with_metric_and_dtype(dims, DistanceMetric::L2, VectorDtype::F16);
        let batch: Vec<_> = (0..200u64)
            .map(|i| (i + 1, hb_vector(dims, i + 1)))
            .collect();
        src.insert_batch(batch).await.expect("insert_batch");
        drain_all(&src).await;
        let bytes = src.encode_graph_snapshot().await;

        let dst =
            HotBufferIndex::new_with_metric_and_dtype(dims, DistanceMetric::L2, VectorDtype::F16);
        dst.restore_graph_snapshot(&bytes).await.expect("restore");
        assert_eq!(dst.vector_dtype(), VectorDtype::F16);

        for q in [1u64, 77, 190] {
            let query = hb_vector(dims, q);
            let a = src
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            let b = dst
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            assert_eq!(a, b, "f16 scored top-10 for query seed {q}");
        }
        let exported = dst.export_snapshot_vectors().await.expect("export");
        assert_eq!(exported.len(), 200);
        assert_eq!(exported[0].1.len(), dims);
    }

    /// f32 目标拒绝 f16 快照（dtype 不匹配必须显式失败，不得静默降精度）。
    /// EN: An f32 target rejects an f16 snapshot — dtype mismatch must fail
    /// loudly instead of silently dropping precision.
    #[tokio::test]
    async fn hotbuffer_restore_rejects_vector_dtype_mismatch() {
        let dims = 8;
        let src =
            HotBufferIndex::new_with_metric_and_dtype(dims, DistanceMetric::L2, VectorDtype::F16);
        src.insert_batch(vec![(1, hb_vector(dims, 1))])
            .await
            .expect("insert_batch");
        drain_all(&src).await;
        let bytes = src.encode_graph_snapshot().await;
        let dst = HotBufferIndex::new(dims);
        assert_eq!(
            dst.restore_graph_snapshot(&bytes).await.err(),
            Some(GraphSnapshotError::Mismatch("vector_dtype"))
        );
    }

    #[tokio::test]
    async fn hotbuffer_tombstones_preserved_and_resurrection_works() {
        let dims = 8;
        let src = build_hotbuffer(100, dims).await;
        for id in [7u64, 42] {
            assert!(src.remove(id).await);
        }
        let bytes = src.encode_graph_snapshot().await;

        let dst = HotBufferIndex::new(dims);
        dst.restore_graph_snapshot(&bytes).await.expect("restore");
        {
            let ts = dst.tombstones.read().await;
            assert!(ts.contains(&7));
            assert!(ts.contains(&42));
            assert_eq!(ts.len(), 2);
        }
        // 删除的 id 不可检索。
        // EN: Deleted ids are not searchable.
        let hits = dst
            .search_with_ef_override(&hb_vector(dims, 7), 100, None, None)
            .await;
        assert!(!hits.contains(&7));
        assert!(!hits.contains(&42));

        // 复活：重插同 id 清墓碑，合图后可检索。
        // 走 insert_batch（生产 collection 路径）：单条 insert 只进 hot_buffer，
        // 而候选生成在图非空时不含 hot（既有行为，非快照语义）。
        // EN: Resurrect: reinserting the same id clears the tombstone and makes it searchable after merge.
        // EN: Use insert_batch (the production collection path): a single insert only fills hot_buffer,
        // EN: and candidate generation excludes hot when the graph is non-empty (existing behavior).
        dst.insert_batch(vec![(7, hb_vector(dims, 7))])
            .await
            .expect("resurrect insert");
        drain_all(&dst).await;
        {
            let ts = dst.tombstones.read().await;
            assert!(!ts.contains(&7));
        }
        let hits = dst
            .search_with_ef_override(&hb_vector(dims, 7), 10, None, None)
            .await;
        assert_eq!(hits.first(), Some(&7), "resurrected id must be top-1");
    }

    #[tokio::test]
    async fn hotbuffer_restore_rebuilds_main_index_from_graph() {
        let dims = 8;
        let src = build_hotbuffer(150, dims).await;
        assert!(src.remove(9).await);
        let bytes = src.encode_graph_snapshot().await;

        let dst = HotBufferIndex::new(dims);
        dst.restore_graph_snapshot(&bytes).await.expect("restore");
        let main = dst.main_index.read().await;
        let graph = crate::hnsw_read_guard(&dst.hnsw);
        assert_eq!(main.len(), graph.vectors.len());
        assert!(!main.contains_key(&9));
        for (id, v) in graph.vectors.iter() {
            let mv = main.get(id).expect("main contains graph id");
            // 主表与图共享 source 句柄（未量化图 source 即图向量本身）。
            // EN: The main map shares the source handle with the graph (for an
            // unquantized graph, source is the graph vector itself).
            assert!(
                Arc::ptr_eq(mv, &v.source_arc()),
                "main must Arc-share the graph source"
            );
        }
    }

    #[tokio::test]
    async fn hotbuffer_restore_rejects_dims_mismatch() {
        let src = build_hotbuffer(16, 8).await;
        let bytes = src.encode_graph_snapshot().await;
        let dst = HotBufferIndex::new(32);
        assert_eq!(
            dst.restore_graph_snapshot(&bytes).await.err(),
            Some(GraphSnapshotError::Mismatch("dims"))
        );
    }

    #[tokio::test]
    async fn hotbuffer_restore_rejects_metric_mismatch() {
        let dims = 8;
        let src = build_hotbuffer(16, dims).await; // L2
        let bytes = src.encode_graph_snapshot().await;
        let dst = HotBufferIndex::new_with_metric(dims, DistanceMetric::Cosine);
        assert_eq!(
            dst.restore_graph_snapshot(&bytes).await.err(),
            Some(GraphSnapshotError::Mismatch("metric"))
        );
    }

    #[tokio::test]
    async fn hotbuffer_restore_rejects_truncated_and_trailing() {
        let src = build_hotbuffer(16, 8).await;
        let bytes = src.encode_graph_snapshot().await;
        let dst = HotBufferIndex::new(8);
        assert!(
            dst.restore_graph_snapshot(&bytes[..bytes.len() - 1])
                .await
                .is_err(),
            "truncated tombstone trailer must fail"
        );
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(
            dst.restore_graph_snapshot(&longer).await.err(),
            Some(GraphSnapshotError::TrailingBytes { extra: 1 })
        );
    }

    #[tokio::test]
    async fn hotbuffer_restore_rejects_non_empty_target() {
        let src = build_hotbuffer(16, 8).await;
        let bytes = src.encode_graph_snapshot().await;
        let dst = build_hotbuffer(4, 8).await;
        assert_eq!(
            dst.restore_graph_snapshot(&bytes).await.err(),
            Some(GraphSnapshotError::InvalidStructure("non_empty_target"))
        );
    }

    #[tokio::test]
    async fn hotbuffer_roundtrip_empty_index() {
        let src = HotBufferIndex::new(8);
        let bytes = src.encode_graph_snapshot().await;
        let dst = HotBufferIndex::new(8);
        dst.restore_graph_snapshot(&bytes).await.expect("restore");
        let graph = crate::hnsw_read_guard(&dst.hnsw);
        assert_eq!(graph.len(), 0);
    }

    #[tokio::test]
    async fn hotbuffer_restore_syncs_hnsw_param_atomics() {
        let src = build_hotbuffer(16, 8).await;
        let bytes = src.encode_graph_snapshot().await;
        let (sm, sec, ses) = {
            let g = crate::hnsw_read_guard(&src.hnsw);
            (g.m, g.ef_construction, g.ef_search)
        };
        let dst = HotBufferIndex::new(8);
        // 人为把目标原子量改成别的值，验证恢复会覆盖。
        // EN: Deliberately set target atomics to other values to verify restore overwrites them.
        dst.hnsw_m.store(99, Ordering::Relaxed);
        dst.restore_graph_snapshot(&bytes).await.expect("restore");
        assert_eq!(dst.hnsw_m.load(Ordering::Relaxed), sm);
        assert_eq!(dst.hnsw_ef_construction.load(Ordering::Relaxed), sec);
        assert_eq!(dst.hnsw_ef_search.load(Ordering::Relaxed), ses);
    }

    #[test]
    fn roundtrip_preserves_cosine_metric_vectors_verbatim() {
        // 快照存的是存储值（cosine 已单位化），恢复必须位级一致、不二次归一化。
        // EN: The snapshot stores stored values (cosine already normalized); restore must be bit-identical.
        let mut g = HnswGraph::new(8, DistanceMetric::Cosine, 4, 32, 16, 42, 8);
        for i in 1..=64u64 {
            g.insert(i, gen_vector(8, i));
        }
        let bytes = g.encode_snapshot();
        let (restored, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, bytes.len());
        assert_graphs_equivalent(&g, &restored);
    }

    // ── P2：图快照 v3（int8 codes + 原始 source） ────────────────
    // ── EN: P2: graph snapshot v3 (int8 codes + raw source) ──────

    fn build_int8_graph(
        n: usize,
        dims: usize,
        metric: DistanceMetric,
        dtype: VectorDtype,
    ) -> HnswGraph {
        let mut g = HnswGraph::new_with_dtype_and_quantization(
            dims,
            metric,
            dtype,
            Quantization::Int8,
            4,
            32,
            16,
            42,
            8,
        );
        for i in 0..n {
            let id = i as u64 + 1;
            g.insert(id, gen_vector(dims, id));
        }
        g
    }

    /// v3 往返：codes / scale / bias 位级一致，source 是**原始**主表值（不是
    /// 反量化近似值）——重启后的精确 rescore 依赖它。
    /// EN: v3 roundtrip: codes/scale/bias are bit-identical and `source` holds the
    /// **raw** main-store values (not the dequantized approximation) — exact
    /// rescoring after restart relies on it.
    #[test]
    fn roundtrip_int8_graph_preserves_codes_and_raw_source() {
        let g = build_int8_graph(64, 8, DistanceMetric::Cosine, VectorDtype::F32);
        let bytes = g.encode_snapshot();
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            SNAPSHOT_VERSION_INT8
        );
        let (restored, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.quantization(), Quantization::Int8);
        assert_graphs_equivalent(&g, &restored);
        for id in 1..=64u64 {
            let pick = |graph: &HnswGraph| {
                let gv = graph.graph_vector(id).expect("vector");
                match gv {
                    GraphVector::Int8 { codes, source } => (
                        codes.codes.clone(),
                        codes.scale.to_bits(),
                        codes.bias.to_bits(),
                        source.to_f32_vec(),
                    ),
                    GraphVector::Full(_) | GraphVector::RabitQ { .. } => {
                        panic!("int8 snapshot must restore GraphVector::Int8")
                    }
                }
            };
            let (ca, sa, ba, src_a) = pick(&g);
            let (cb, sb, bb, src_b) = pick(&restored);
            assert_eq!(ca, cb, "codes for {id}");
            assert_eq!((sa, ba), (sb, bb), "scale/bias bits for {id}");
            assert_eq!(src_a, src_b, "raw source for {id}");
        }
    }

    /// v3 与 f16 主表组合：两个 flag 位并存，source 按 f16 精确往返。
    /// EN: v3 combined with an f16 main store: both flag bits coexist and the
    /// f16 source roundtrips exactly.
    #[test]
    fn roundtrip_int8_f16_graph_combination() {
        let g = build_int8_graph(32, 8, DistanceMetric::Cosine, VectorDtype::F16);
        let bytes = g.encode_snapshot();
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            SNAPSHOT_VERSION_INT8
        );
        assert_eq!(bytes[17] & FLAG_VECTOR_INT8, FLAG_VECTOR_INT8, "int8 flag");
        assert_eq!(bytes[17] & FLAG_VECTOR_F16, FLAG_VECTOR_F16, "f16 flag");
        let (restored, _) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(restored.dtype(), VectorDtype::F16);
        assert_eq!(restored.quantization(), Quantization::Int8);
        assert_graphs_equivalent(&g, &restored);
    }

    /// 标志位门禁：v3 必须置 int8 位；int8 位不得出现在 v1 头上。
    /// EN: Flag gates: v3 must set the int8 bit, and the int8 bit must not
    /// appear on a v1 header.
    #[test]
    fn decode_rejects_int8_flag_and_version_mismatch() {
        let int8 = build_int8_graph(8, 8, DistanceMetric::L2, VectorDtype::F32);
        let mut bytes = int8.encode_snapshot();
        bytes[17] &= !FLAG_VECTOR_INT8;
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&int8)).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );

        let plain = build_graph(8, 8, 4, 8);
        let mut bytes = plain.encode_snapshot();
        bytes[17] |= FLAG_VECTOR_INT8;
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&plain)).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );
    }

    /// 双向量化期望校验：未量化目标拒绝 int8 快照（codes 会被当成 f32 主表值），
    /// int8 目标拒绝未量化快照（静默降级成未量化图）。
    /// EN: Bidirectional quantization expectations: an unquantized target rejects
    /// an int8 snapshot (codes would be read as f32 main-store values), and an
    /// int8 target rejects an unquantized snapshot (silent downgrade).
    #[test]
    fn decode_rejects_quantization_mismatch_in_both_directions() {
        let int8 = build_int8_graph(8, 8, DistanceMetric::L2, VectorDtype::F32);
        let plain = HnswGraph::new(8, DistanceMetric::L2, 4, 32, 16, 42, 8);
        let bytes = int8.encode_snapshot();
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&plain)).err(),
            Some(GraphSnapshotError::Mismatch("quantization"))
        );
        let plain_g = build_graph(8, 8, 4, 8);
        let bytes = plain_g.encode_snapshot();
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&int8)).err(),
            Some(GraphSnapshotError::Mismatch("quantization"))
        );
    }

    /// peek 四元组返回量化方式；旧 dtype/dims peek 行为不变。
    /// EN: The quartet peek reports quantization; legacy dtype/dims peeks are unchanged.
    #[test]
    fn peek_reports_quantization_and_keeps_legacy_entries() {
        let int8 = build_int8_graph(4, 8, DistanceMetric::Cosine, VectorDtype::F16);
        let bytes = int8.encode_snapshot();
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype_quantization(&bytes),
            Ok((
                8,
                DistanceMetric::Cosine,
                VectorDtype::F16,
                Quantization::Int8
            ))
        );
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype(&bytes),
            Ok((8, DistanceMetric::Cosine, VectorDtype::F16))
        );
        assert_eq!(
            peek_graph_snapshot_dims_metric(&bytes),
            Ok((8, DistanceMetric::Cosine))
        );
        let plain = build_graph(4, 8, 4, 8);
        let bytes = plain.encode_snapshot();
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype_quantization(&bytes),
            Ok((8, DistanceMetric::L2, VectorDtype::F32, Quantization::None))
        );
    }

    /// v3 段长对账：伪造巨型 count 必须是干净的 Truncated（按 v3 记录宽度）。
    /// EN: v3 section-length reconcile: forged giant counts must be a clean
    /// Truncated under the v3 record width.
    #[test]
    fn decode_rejects_oversized_declared_counts_for_v3() {
        let g = build_int8_graph(4, 8, DistanceMetric::L2, VectorDtype::F32);
        let mut bytes = g.encode_snapshot();
        bytes[60..68].copy_from_slice(&u64::MAX.to_be_bytes()); // node_count
        bytes[84..92].copy_from_slice(&u64::MAX.to_be_bytes()); // vector_count（保持相等）
        let err = HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err();
        assert!(
            matches!(err, Some(GraphSnapshotError::Truncated { .. })),
            "giant declared counts must be a clean Truncated, got {err:?}"
        );
    }

    /// v3 双射与有限性守卫：重复 id 拒绝；非有限 scale/bias 拒绝（否则图内距离
    /// 产出 NaN 污染堆序）。
    /// EN: v3 bijection and finiteness guards: duplicate ids are rejected; non-finite
    /// scale/bias are rejected (they would NaN out in-graph distances and corrupt heap order).
    #[test]
    fn decode_rejects_duplicate_and_non_finite_v3_vector_records() {
        let g = build_int8_graph(4, 8, DistanceMetric::L2, VectorDtype::F32);
        let bytes = g.encode_snapshot();
        let node_count = u64::from_be_bytes(bytes[60..68].try_into().unwrap()) as usize;
        let arena_len = u64::from_be_bytes(bytes[68..76].try_into().unwrap()) as usize;
        let lf_pairs = u64::from_be_bytes(bytes[76..84].try_into().unwrap()) as usize;
        let voff = HEADER_LEN + node_count * NODE_RECORD_LEN + arena_len * 8 + lf_pairs * 16;
        let rec = 16 + 5 * 8; // id u64 + scale/bias f32 + dims×i8 + dims×f32 source

        let mut dup = bytes.clone();
        let first_id: [u8; 8] = dup[voff..voff + 8].try_into().unwrap();
        dup[voff + rec..voff + rec + 8].copy_from_slice(&first_id);
        assert_eq!(
            HnswGraph::decode_snapshot(&dup, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::InvalidStructure("vector.duplicate"))
        );

        // scale 非有限（换为 NaN 位型）按伪造拒绝。
        // EN: non-finite scale (NaN bit pattern) is rejected as forged.
        let mut bad = bytes;
        bad[voff + 8..voff + 12].copy_from_slice(&f32::NAN.to_bits().to_be_bytes());
        assert_eq!(
            HnswGraph::decode_snapshot(&bad, &expectations_for(&g)).err(),
            Some(GraphSnapshotError::InvalidStructure("vector.quantization"))
        );
    }

    /// int8 HotBuffer 快照往返：恢复后主表必须是**原始**向量且与图 source 共享
    /// `Arc`，检索排序与源索引一致。
    /// EN: int8 HotBuffer snapshot roundtrip: the restored main store keeps raw
    /// vectors and Arc-shares them with the graph source; search ranking matches
    /// the source index.
    #[tokio::test]
    async fn hotbuffer_int8_snapshot_roundtrip_restores_raw_source() {
        let dims = 16;
        let mk = || {
            HotBufferIndex::new_with_metric_and_config_and_dtype_and_quantization(
                dims,
                DistanceMetric::L2,
                VectorDtype::F32,
                Quantization::Int8,
                &houyi_core::config::VectorConfig::default(),
                &houyi_core::config::HnswConfig::default(),
            )
        };
        let src = mk();
        let batch: Vec<_> = (0..64u64)
            .map(|i| (i + 1, hb_vector(dims, i + 1)))
            .collect();
        let expected: Vec<(u64, Vec<f32>)> = batch.clone();
        src.insert_batch(batch).await.expect("insert_batch");
        drain_all(&src).await;
        assert_eq!(src.vector_quantization(), Quantization::Int8);
        let bytes = src.encode_graph_snapshot().await;

        let dst = mk();
        dst.restore_graph_snapshot(&bytes).await.expect("restore");
        assert_eq!(dst.vector_quantization(), Quantization::Int8);

        for (id, raw) in &expected {
            let main = dst.main_index.read().await;
            let stored = main.get(id).expect("restored row must reach the main map");
            assert_eq!(stored.to_f32_vec(), *raw, "main keeps raw value for {id}");
        }
        for (id, _) in &expected {
            let main_arc = {
                let main = dst.main_index.read().await;
                Arc::clone(main.get(id).expect("restored row"))
            };
            let g = crate::hnsw_read_guard(&dst.hnsw);
            let gv = g.graph_vector(*id).expect("restored graph vector");
            assert!(
                matches!(gv, GraphVector::Int8 { .. }),
                "restored graph must stay int8"
            );
            assert!(
                Arc::ptr_eq(&main_arc, &gv.source_arc()),
                "main store and int8 source must share one Arc"
            );
        }

        for q in [1u64, 17, 42] {
            let query = hb_vector(dims, q);
            let a = src
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            let b = dst
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            assert_eq!(a, b, "int8 scored top-10 for query seed {q}");
        }
    }

    // ── P3：图快照 v4（RaBitQ codes + 原始 source） ────────────────
    // ── EN: P3: graph snapshot v4 (RaBitQ codes + raw source) ────

    fn build_rabitq_graph(
        n: usize,
        dims: usize,
        metric: DistanceMetric,
        dtype: VectorDtype,
    ) -> HnswGraph {
        let mut g = HnswGraph::new_with_dtype_and_quantization(
            dims,
            metric,
            dtype,
            Quantization::RaBitQ,
            4,
            32,
            16,
            42,
            8,
        );
        for i in 0..n {
            let id = i as u64 + 1;
            g.insert(id, gen_vector(dims, id));
        }
        g
    }

    /// v4 往返：codes/alpha/norm 位级一致，source 是**原始**主表值——重启后的
    /// 精确 rescore 依赖它。版本号与 bit3 标志必须同时出现。
    /// EN: v4 roundtrip: codes/alpha/norm are bit-identical and `source` holds
    /// **raw** main-store values (exact rescoring after restart depends on it).
    /// The version number and the bit3 flag must appear together.
    #[test]
    fn roundtrip_rabitq_graph_preserves_codes_and_raw_source() {
        let g = build_rabitq_graph(64, 8, DistanceMetric::Cosine, VectorDtype::F32);
        let bytes = g.encode_snapshot();
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            SNAPSHOT_VERSION_RABITQ
        );
        assert_eq!(
            bytes[17] & FLAG_VECTOR_RABITQ,
            FLAG_VECTOR_RABITQ,
            "v4 must set the rabitq flag"
        );
        assert_eq!(bytes[17] & FLAG_VECTOR_INT8, 0, "v4 must not set int8");
        let (restored, consumed) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.quantization(), Quantization::RaBitQ);
        assert_graphs_equivalent(&g, &restored);
        for id in 1..=64u64 {
            let pick = |graph: &HnswGraph| {
                let gv = graph.graph_vector(id).expect("vector");
                match gv {
                    GraphVector::RabitQ { codes, source } => (
                        codes.codes.clone(),
                        codes.alpha.to_bits(),
                        codes.norm.to_bits(),
                        source.to_f32_vec(),
                    ),
                    GraphVector::Full(_) | GraphVector::Int8 { .. } => {
                        panic!("rabitq snapshot must restore GraphVector::RabitQ")
                    }
                }
            };
            let (ca, aa, na, src_a) = pick(&g);
            let (cb, ab, nb, src_b) = pick(&restored);
            assert_eq!(ca, cb, "codes for {id}");
            assert_eq!((aa, na), (ab, nb), "alpha/norm bits for {id}");
            assert_eq!(src_a, src_b, "raw source for {id}");
        }
    }

    /// v4 与 f16 主表组合：bit3 与 bit1 并存，source 按 f16 精确往返。
    /// EN: v4 combined with an f16 main store: bit3 and bit1 coexist and the
    /// f16 source roundtrips exactly.
    #[test]
    fn roundtrip_rabitq_f16_graph_combination() {
        let g = build_rabitq_graph(32, 8, DistanceMetric::Cosine, VectorDtype::F16);
        let bytes = g.encode_snapshot();
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            SNAPSHOT_VERSION_RABITQ
        );
        assert_eq!(
            bytes[17] & FLAG_VECTOR_RABITQ,
            FLAG_VECTOR_RABITQ,
            "rabitq flag"
        );
        assert_eq!(bytes[17] & FLAG_VECTOR_F16, FLAG_VECTOR_F16, "f16 flag");
        let (restored, _) =
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).expect("decode");
        assert_eq!(restored.dtype(), VectorDtype::F16);
        assert_eq!(restored.quantization(), Quantization::RaBitQ);
        assert_graphs_equivalent(&g, &restored);
    }

    /// 标志位门禁：v4 必须置 bit3；bit3 不得出现在 v1 头上；v4 携带 int8 位
    /// （bit2）拒绝（一张图不可能同时采用两种图侧量化）。
    /// EN: Flag gates: v4 must set bit3; bit3 must not appear on a v1 header;
    /// v4 carrying the int8 bit (bit2) is rejected (one graph cannot use two
    /// graph-side quantization schemes).
    #[test]
    fn decode_rejects_rabitq_flag_and_version_mismatch() {
        let rabitq = build_rabitq_graph(8, 8, DistanceMetric::L2, VectorDtype::F32);
        let mut bytes = rabitq.encode_snapshot();
        bytes[17] &= !FLAG_VECTOR_RABITQ;
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&rabitq)).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );

        let plain = build_graph(8, 8, 4, 8);
        let mut bytes = plain.encode_snapshot();
        bytes[17] |= FLAG_VECTOR_RABITQ;
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&plain)).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );

        let mut bytes = rabitq.encode_snapshot();
        bytes[17] |= FLAG_VECTOR_INT8;
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&rabitq)).err(),
            Some(GraphSnapshotError::InvalidStructure("flags"))
        );
    }

    /// 量化期望双向拒绝：未量化/int8 目标拒绝 RaBitQ 快照，RaBitQ 目标拒绝
    /// 未量化快照——绝不允许静默改变图侧距离语义。
    /// EN: Bidirectional quantization expectations: unquantized/int8 targets
    /// reject a RaBitQ snapshot and a RaBitQ target rejects an unquantized
    /// snapshot — in-graph distance semantics never change silently.
    #[test]
    fn decode_rejects_rabitq_quantization_mismatch_in_both_directions() {
        let rabitq = build_rabitq_graph(8, 8, DistanceMetric::L2, VectorDtype::F32);
        let plain = HnswGraph::new(8, DistanceMetric::L2, 4, 32, 16, 42, 8);
        let int8 = build_int8_graph(8, 8, DistanceMetric::L2, VectorDtype::F32);
        let bytes = rabitq.encode_snapshot();
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&plain)).err(),
            Some(GraphSnapshotError::Mismatch("quantization"))
        );
        assert_eq!(
            HnswGraph::decode_snapshot(&bytes, &expectations_for(&int8)).err(),
            Some(GraphSnapshotError::Mismatch("quantization"))
        );
        let plain_bytes = build_graph(8, 8, 4, 8).encode_snapshot();
        assert_eq!(
            HnswGraph::decode_snapshot(&plain_bytes, &expectations_for(&rabitq)).err(),
            Some(GraphSnapshotError::Mismatch("quantization"))
        );
    }

    /// peek 四元组返回 RaBitQ；旧 peek 入口行为不变。
    /// EN: The quartet peek reports RaBitQ; legacy peek entries are unchanged.
    #[test]
    fn peek_reports_rabitq_and_keeps_legacy_entries() {
        let rabitq = build_rabitq_graph(4, 8, DistanceMetric::Cosine, VectorDtype::F16);
        let bytes = rabitq.encode_snapshot();
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype_quantization(&bytes),
            Ok((
                8,
                DistanceMetric::Cosine,
                VectorDtype::F16,
                Quantization::RaBitQ
            ))
        );
        assert_eq!(
            peek_graph_snapshot_dims_metric_dtype(&bytes),
            Ok((8, DistanceMetric::Cosine, VectorDtype::F16))
        );
    }

    /// v4 标量守卫：非法 alpha（非有限 / ≤0）与非法 norm（非有限 / <0）按伪造
    /// 拒绝（否则图内距离 NaN 污染堆序或反向爆炸）。
    /// EN: v4 scalar guards: invalid alpha (non-finite / <= 0) and invalid norm
    /// (non-finite / < 0) are rejected as forged (otherwise in-graph distances
    /// become NaN or blow up backwards).
    #[test]
    fn decode_rejects_invalid_rabitq_scalars() {
        let g = build_rabitq_graph(4, 8, DistanceMetric::L2, VectorDtype::F32);
        let bytes = g.encode_snapshot();
        let node_count = u64::from_be_bytes(bytes[60..68].try_into().unwrap()) as usize;
        let arena_len = u64::from_be_bytes(bytes[68..76].try_into().unwrap()) as usize;
        let lf_pairs = u64::from_be_bytes(bytes[76..84].try_into().unwrap()) as usize;
        let voff = HEADER_LEN + node_count * NODE_RECORD_LEN + arena_len * 8 + lf_pairs * 16;
        // 记录布局：id(8) | norm(4) | alpha(4) | codes(ceil(D/64)×8) | source(dims×4)。
        // EN: Record layout: id(8) | norm(4) | alpha(4) | codes(ceil(D/64)×8) | source(dims×4).
        let norm_at = voff + 8;
        let alpha_at = voff + 12;
        let cases: [(&str, usize, f32); 4] = [
            ("alpha NaN", alpha_at, f32::NAN),
            ("alpha zero", alpha_at, 0.0),
            ("norm NaN", norm_at, f32::NAN),
            ("norm negative", norm_at, -1.0),
        ];
        for (what, offset, value) in cases {
            let mut bad = bytes.clone();
            bad[offset..offset + 4].copy_from_slice(&value.to_bits().to_be_bytes());
            assert_eq!(
                HnswGraph::decode_snapshot(&bad, &expectations_for(&g)).err(),
                Some(GraphSnapshotError::InvalidStructure("vector.quantization")),
                "{what} must be rejected"
            );
        }
    }

    /// v4 段长对账：伪造巨型 count 必须是干净的 Truncated（按 v4 记录宽度）。
    /// EN: v4 section-length reconcile: forged giant counts must be a clean
    /// Truncated under the v4 record width.
    #[test]
    fn decode_rejects_oversized_declared_counts_for_v4() {
        let g = build_rabitq_graph(4, 8, DistanceMetric::L2, VectorDtype::F32);
        let mut bytes = g.encode_snapshot();
        bytes[60..68].copy_from_slice(&u64::MAX.to_be_bytes()); // node_count
        bytes[84..92].copy_from_slice(&u64::MAX.to_be_bytes()); // vector_count（保持相等）
        let err = HnswGraph::decode_snapshot(&bytes, &expectations_for(&g)).err();
        assert!(
            matches!(err, Some(GraphSnapshotError::Truncated { .. })),
            "giant declared counts must be a clean Truncated, got {err:?}"
        );
    }

    /// RaBitQ HotBuffer 快照往返：恢复后主表是**原始**向量且与图 source 共享
    /// `Arc`；检索排序与源索引一致（图内估计只用于取候选，最终打分在主表精确
    /// 计算）。
    /// EN: RaBitQ HotBuffer snapshot roundtrip: the restored main store keeps raw
    /// vectors and Arc-shares them with the graph source; search ranking matches
    /// the source index (in-graph estimates only propose candidates while final
    /// scores are computed exactly on the main store).
    #[tokio::test]
    async fn hotbuffer_rabitq_snapshot_roundtrip_restores_raw_source() {
        let dims = 16;
        let mk = || {
            HotBufferIndex::new_with_metric_and_config_and_dtype_and_quantization(
                dims,
                DistanceMetric::L2,
                VectorDtype::F32,
                Quantization::RaBitQ,
                &houyi_core::config::VectorConfig::default(),
                &houyi_core::config::HnswConfig::default(),
            )
        };
        let src = mk();
        let batch: Vec<_> = (0..64u64)
            .map(|i| (i + 1, hb_vector(dims, i + 1)))
            .collect();
        let expected: Vec<(u64, Vec<f32>)> = batch.clone();
        src.insert_batch(batch).await.expect("insert_batch");
        drain_all(&src).await;
        assert_eq!(src.vector_quantization(), Quantization::RaBitQ);
        let bytes = src.encode_graph_snapshot().await;

        let dst = mk();
        dst.restore_graph_snapshot(&bytes).await.expect("restore");
        assert_eq!(dst.vector_quantization(), Quantization::RaBitQ);

        for (id, raw) in &expected {
            let main = dst.main_index.read().await;
            let stored = main.get(id).expect("restored row must reach the main map");
            assert_eq!(stored.to_f32_vec(), *raw, "main keeps raw value for {id}");
        }
        for (id, _) in &expected {
            let main_arc = {
                let main = dst.main_index.read().await;
                Arc::clone(main.get(id).expect("restored row"))
            };
            let g = crate::hnsw_read_guard(&dst.hnsw);
            let gv = g.graph_vector(*id).expect("restored graph vector");
            assert!(
                matches!(gv, GraphVector::RabitQ { .. }),
                "restored graph must stay rabitq"
            );
            assert!(
                Arc::ptr_eq(&main_arc, &gv.source_arc()),
                "main store and rabitq source must share one Arc"
            );
        }

        for q in [1u64, 17, 42] {
            let query = hb_vector(dims, q);
            let a = src
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            let b = dst
                .search_scored_with_ef_override(&query, 10, None, None)
                .await;
            assert_eq!(a, b, "rabitq scored top-10 for query seed {q}");
        }
    }
}
