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

use crate::hnsw_graph::{HnswGraph, HnswNode, SnapshotGraphParts, MAX_HNSW_LAYERS};
use crate::{DistanceMetric, VectorId};
use std::fmt;
use std::sync::Arc;

use ahash::AHashMap;

/// 快照 payload 魔数。 / EN: Payload magic.
pub(crate) const SNAPSHOT_MAGIC: &[u8; 8] = b"HNSWGRPH";
/// 当前格式版本。 / EN: Current format version.
pub(crate) const SNAPSHOT_VERSION: u32 = 1;
/// 定长头字节数（见模块文档布局图）。 / EN: Fixed header length.
pub(crate) const HEADER_LEN: usize = 108;
/// 单节点记录字节数：id u64 + link_base u64 + max_layer u8 + layer_len 16×u16。
/// EN: Per-node record size.
pub(crate) const NODE_RECORD_LEN: usize = 8 + 8 + 1 + 2 * MAX_HNSW_LAYERS;
/// flags bit0：entry_point 字段存在。 / EN: flags bit0 — entry_point present.
const FLAG_HAS_ENTRY_POINT: u8 = 0x01;

/// 恢复期的结构性匹配期望：任一不符即拒绝快照（调用方退化为全量重放）。
/// EN: Structural expectations for restore; any mismatch rejects the snapshot
/// (caller falls back to full WAL replay).
///
/// `m/ef_*` 刻意**不在**此列——它们是运行时可调参数，恢复时采用快照值。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SnapshotExpectations {
    pub dims: usize,
    pub metric: DistanceMetric,
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
    let mut r = Reader::new(payload);
    // 短于 magic 的输入判 BadMagic 而非 Truncated：peek 的职责是识别文件类型，
    // 连魔数都凑不齐的文件按「不是本格式」归类，调用方日志更准确。
    // EN: Input shorter than magic yields BadMagic, not Truncated: peek identifies the file type;
    // EN: a file that cannot even complete the magic is classified as not-this-format for better logs.
    if payload.len() < SNAPSHOT_MAGIC.len() || r.take(SNAPSHOT_MAGIC.len())? != SNAPSHOT_MAGIC {
        return Err(GraphSnapshotError::BadMagic);
    }
    let version = r.u32()?;
    if version != SNAPSHOT_VERSION {
        return Err(GraphSnapshotError::UnsupportedVersion(version));
    }
    let dims = r.u32()? as usize;
    let metric_byte = r.u8()?;
    let metric = DistanceMetric::from_snapshot_byte(metric_byte)
        .ok_or(GraphSnapshotError::InvalidMetric(metric_byte))?;
    Ok((dims, metric))
}

impl HnswGraph {
    /// 序列化整图为 payload 字节。调用方需保证无并发写（持图读锁即可，
    /// client 写经 WAL/热缓冲不受阻）。
    /// EN: Serialize the whole graph. Caller must exclude concurrent mutation
    /// (a graph read lock suffices).
    pub(crate) fn encode_snapshot(&self) -> Vec<u8> {
        let (l0_mult, upper_mult) = crate::hnsw_graph::effective_layer_multipliers();
        let link_free_pairs: usize = self.link_free.values().map(Vec::len).sum();
        let est = HEADER_LEN
            + self.nodes.len() * NODE_RECORD_LEN
            + self.link_arena.len() * 8
            + link_free_pairs * 16
            + self.vectors.len() * (8 + self.dims * 4);
        let mut w = Writer::with_capacity(est);

        // Header
        w.raw(SNAPSHOT_MAGIC);
        w.u32(SNAPSHOT_VERSION);
        w.u32(self.dims as u32);
        w.u8(self.metric.to_snapshot_byte());
        // entry_point 存在性进 flags：u64::MAX 是合法 VectorId，不能当哨兵。
        // EN: entry_point presence goes into flags: u64::MAX is a valid VectorId, not a sentinel.
        w.u8(if self.entry_point.is_some() {
            FLAG_HAS_ENTRY_POINT
        } else {
            0
        });
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

        // Vectors（存的是归一化后的存储值，原样落盘原样恢复）
        // EN: Vectors (normalized stored values; written and restored verbatim).
        for (id, vec) in &self.vectors {
            w.u64(*id);
            debug_assert_eq!(vec.len(), self.dims);
            for x in vec.iter() {
                w.f32_bits(*x);
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
        if version != SNAPSHOT_VERSION {
            return Err(GraphSnapshotError::UnsupportedVersion(version));
        }
        let dims = r.u32()? as usize;
        let metric_byte = r.u8()?;
        let metric = DistanceMetric::from_snapshot_byte(metric_byte)
            .ok_or(GraphSnapshotError::InvalidMetric(metric_byte))?;
        let flags = r.u8()?;
        let _reserved = r.u16()?;
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
                    .checked_mul(8 + 4 * dims)
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

        // Vectors
        let mut vectors: AHashMap<VectorId, Arc<Vec<f32>>> = AHashMap::with_capacity(vector_count);
        for _ in 0..vector_count {
            let id = r.u64()?;
            if !nodes.contains_key(&id) {
                return Err(GraphSnapshotError::InvalidStructure("vector.orphan"));
            }
            let mut v = Vec::with_capacity(dims);
            for _ in 0..dims {
                v.push(r.f32_bits()?);
            }
            // 重复 id 会静默覆盖并破坏 node/vector 双射（漏检的节点从 main_index
            // 缺席、被搜索静默跳过），与 nodes 段同纪律拒绝（OCR ae8d7e3）。
            // EN: Duplicate ids would silently overwrite and break the node/vector bijection (missing
            // EN: nodes are absent from main_index and skipped by search); rejected like the nodes section.
            if vectors.insert(id, Arc::new(v)).is_some() {
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

        main.clear();
        for (id, v) in &graph.vectors {
            main.insert(*id, Arc::clone(v));
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
            assert_eq!(avec.as_slice(), bvec.as_slice(), "vector bits for {aid}");
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
            assert!(Arc::ptr_eq(mv, v), "main must Arc-share graph vectors");
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
}
