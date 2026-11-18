//! HouyiDB collections HTTP API 的 Rust 客户端（同步，ureq 阻塞式）。
//! EN: Official Rust client for the HouyiDB collections HTTP API (blocking, ureq).
//!
//! 与服务端的契约要点：
//! EN: Contract notes with the server:
//! - collection 在首次 upsert 时隐式创建，维度由首个向量定型；也可用
//!   [`Client::create_collection`]（PUT）显式定型 dims/metric/后端/HNSW 参数；
//!   EN: a collection is created implicitly on first upsert and its dimension is
//!   fixed by the first vector; [`Client::create_collection`] (PUT) may also
//!   create it explicitly with fixed dims/metric/backend/HNSW parameters;
//! - payload 键 `label` 为保留列，占用会 400（用顶层 `label` 字段）；
//!   EN: the payload key `label` is reserved (using it returns 400; use the
//!   top-level `label` field instead);
//! - 检索响应的 `score` 按创建时的度量换算：cosine/l2/l1 为 `1 - 内部距离`
//!   （cosine 夹到 [-1, 1]），ip 为原始点积；数值越大越近，但量纲随度量变化。
//!   EN: search `score` follows the collection metric: cosine/l2/l1 convert the
//!   kernel distance as `1 - d` (cosine clamped to [-1, 1]) and ip exposes the
//!   raw dot product; larger is closer, but the scale depends on the metric;
//! - [`stable_id`] 与 Python/TS/Go SDK、服务端 OpenAI 网关同一算法
//!   （BLAKE2b-8 大端 & 2^63-1），删除可跨 SDK 重放。
//!   EN: [`stable_id`] uses the same algorithm as the Python/TS/Go SDKs and the
//!   server OpenAI gateway (BLAKE2b-8 big-endian & 2^63-1), so deletes replay
//!   across SDKs.
//! - 稀疏词项由客户端给定（无分词器）；`sparse=bm25` 必须显式建集合，写入
//!   仅 HTTP/SDK（pgwire 不提供稀疏写入）。稀疏键缺省 = 不改，`[]` = 清空。
//!   EN: sparse terms are client-supplied (no tokenizer); `sparse=bm25` requires
//!   explicit creation and writes exist only over HTTP/SDK (pgwire offers no
//!   sparse writes). An absent sparse key means unchanged and `[]` clears it.
//! - 融合检索的 `score` 仅表达融合排序（RRF 或加权），不属于任何度量口径；
//!   `dense_score` 才按集合度量换算，`sparse_score` 为 BM25 原始分。
//!   EN: fusion `score` only expresses fused ranking (RRF or weighted) and
//!   belongs to no metric; `dense_score` follows the collection metric while
//!   `sparse_score` is the raw BM25 score.

use blake2::digest::generic_array::typenum::U8;
use blake2::Digest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// u64 正整数键空间的上半位留空（与服务端 `U64_MASK` 同口径）。
/// EN: Leaves the upper half of the u64 positive-key space unused (same rule as
/// the server's `U64_MASK`).
const U64_MASK: u64 = (1 << 63) - 1;

type Blake2b8 = blake2::Blake2b<U8>;

/// 把任意字符串 id 确定性映射为 HouyiDB 的 u64 向量 id。
///
/// 与服务端网关 `openai_gateway::stable_id` 逐字同款（`Blake2b<U8>` 即
/// blake2b digest_size=8 的参数化——digest 长度是算法参数而非截断）。
/// EN: Deterministic string→u64 id mapping, identical to the server gateway's
/// `openai_gateway::stable_id` (`Blake2b<U8>` is blake2b parameterized with
/// digest_size=8; the digest length is an algorithm parameter, not a truncation).
pub fn stable_id(text: &str) -> u64 {
    let mut h = Blake2b8::new();
    h.update(text.as_bytes());
    let digest = h.finalize();
    u64::from_be_bytes(digest.into()) & U64_MASK
}

/// 客户端错误：HTTP 非 2xx（状态码 + 服务端诊断文本）、传输层失败、响应
/// 违反协议形状，或调用方传入的参数在发出请求前就违反协议约束。
/// EN: Client error: HTTP non-2xx (status code + server diagnostic text), a
/// transport failure, a response violating the protocol shape, or an argument
/// rejected before the request is sent.
// 非穷尽枚举：后续新增变体不破坏下游的穷尽匹配。
// EN: Non-exhaustive enum: future variants do not break downstream exhaustive matches.
#[non_exhaustive]
#[derive(Debug)]
pub enum HouyiError {
    /// 服务端返回的非 2xx（状态码 + 纯文本诊断，axum 元组错误原样透出）。
    /// EN: Non-2xx returned by the server (status code + plain-text diagnostic;
    /// axum tuple errors passed through verbatim).
    Http { status: u16, message: String },
    /// 连接/读取等传输层失败。
    /// EN: Transport failure such as connect/read errors.
    Transport(String),
    /// 2xx 响应体违反协议形状（例如平行数组宽度错位）：确定性错误，
    /// 调用方不应重试。
    /// EN: A 2xx response body violates the protocol shape (e.g. misaligned
    /// parallel arrays): a deterministic error callers must not retry.
    InvalidResponse(String),
    /// 请求参数未通过客户端预校验（与服务端同口径的快速失败，未发出请求）。
    /// EN: Request arguments failed client-side pre-validation (the same limits
    /// as the server, failed fast without sending the request).
    InvalidRequest(String),
}

impl fmt::Display for HouyiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http { status, message } => write!(f, "houyidb HTTP {status}: {message}"),
            Self::Transport(e) => write!(f, "houyidb transport error: {e}"),
            Self::InvalidResponse(e) => write!(f, "houyidb invalid response: {e}"),
            Self::InvalidRequest(e) => write!(f, "houyidb invalid request: {e}"),
        }
    }
}

impl std::error::Error for HouyiError {}

/// 稀疏词项对 `(term_id, weight)`：词项 id 由客户端给定（无分词器），
/// 线上形状为 `[term, weight]` 二元数组；权重必须有限且 > 0。
/// EN: Sparse term pair `(term_id, weight)`: the term id is client-supplied
/// (no tokenizer) and the wire shape is a `[term, weight]` pair array; the
/// weight must be finite and > 0.
pub type SparseTerms = Vec<(u32, f32)>;

/// 稀疏词项 id 上界：24 位（与服务端一致；词表充足且编码紧凑）。
/// EN: Upper bound of sparse term ids: 24 bits (matches the server; ample
/// vocabulary headroom while staying compact).
pub const MAX_SPARSE_TERM_ID: u32 = (1 << 24) - 1;

/// 单文档稀疏词项原始条目上限（重复词项先求和再判重）。
/// EN: Raw sparse-pair cap per document (duplicates are summed before dedup).
pub const MAX_SPARSE_TERMS_PER_DOC: usize = 1024;

/// 单次稀疏查询的原始词项上限。
/// EN: Raw term cap for one sparse query.
pub const MAX_SPARSE_QUERY_TERMS: usize = 64;

/// RRF 平滑常数 `rrf_k` 的合法上界（下界为 1；与服务端一致）。
/// EN: Valid upper bound of the RRF smoothing constant `rrf_k` (lower bound is
/// 1; matches the server).
pub const MAX_RRF_K: u32 = 1000;

/// 客户端侧稀疏词项预校验（与服务端同口径的快速失败）：条目数、term 范围、
/// 权重有限且 > 0。空列表是否合法由调用方判定（查询必须非空，清空文档允许空）。
/// EN: Client-side sparse pre-validation (fail fast with the server's own
/// limits): pair count, term range, and finite weights > 0. Whether an empty
/// list is legal is decided by the caller (queries require non-empty; clearing
/// a document's terms allows empty).
fn validate_sparse_terms(
    terms: &[(u32, f32)],
    max_terms: usize,
    field: &str,
) -> Result<(), HouyiError> {
    if terms.len() > max_terms {
        return Err(HouyiError::InvalidRequest(format!(
            "{field} carries {} pairs, over the {max_terms}-pair limit",
            terms.len()
        )));
    }
    for &(term, weight) in terms {
        if term == 0 || term > MAX_SPARSE_TERM_ID {
            return Err(HouyiError::InvalidRequest(format!(
                "{field} term id {term} outside 1..={MAX_SPARSE_TERM_ID}"
            )));
        }
        // 权重 0/负数会让 BM25 的 tf/dl 失去意义，NaN/∞ 会污染全部累加分数。
        // EN: Zero or negative weights make BM25's tf/dl meaningless, and
        // NaN/∞ contaminates every accumulated score.
        if !weight.is_finite() || weight <= 0.0 {
            return Err(HouyiError::InvalidRequest(format!(
                "{field} term {term} weight {weight} must be finite and > 0"
            )));
        }
    }
    Ok(())
}

/// 一条待写入向量：id + 分量 + 可选 label/payload/稀疏词项。
/// EN: A vector to write: id + components + optional label/payload/sparse terms.
#[derive(Debug, Clone, Serialize)]
pub struct Vector {
    pub id: u64,
    pub values: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<HashMap<String, serde_json::Value>>,
    /// 稀疏词项（仅 `sparse=bm25` 集合）：None = 不改，Some([]) = 清空。
    /// EN: Sparse terms (only `sparse=bm25` collections): None means unchanged,
    /// `Some([])` clears.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseTerms>,
}

/// 过滤算子白名单（与服务端一致；多条 [`Filter`] 之间取交集）。
/// EN: Filter operator whitelist (matches the server; multiple [`Filter`] entries
/// are intersected).
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterOp {
    Eq,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
}

/// 一条结构化过滤条件。
/// EN: One structured filter condition.
#[derive(Debug, Clone, Serialize)]
pub struct Filter {
    pub column: String,
    pub op: FilterOp,
    pub value: serde_json::Value,
}

impl Filter {
    pub fn new(column: &str, op: FilterOp, value: impl Into<serde_json::Value>) -> Self {
        Self {
            column: column.to_string(),
            op,
            value: value.into(),
        }
    }
}

/// 单条命中：id + 相似度分数 + 可选 payload（include_payload 时）。
/// EN: One hit: id + similarity score + optional payload (when include_payload is set).
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: u64,
    pub score: f64,
    pub payload: Option<HashMap<String, serde_json::Value>>,
}

/// 检索响应：hits 按相似度从高到低排列。
/// EN: Search response: hits ordered from highest to lowest similarity.
#[derive(Debug, Clone, Default)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
}

impl SearchResult {
    pub fn ids(&self) -> Vec<u64> {
        self.hits.iter().map(|h| h.id).collect()
    }
    pub fn scores(&self) -> Vec<f64> {
        self.hits.iter().map(|h| h.score).collect()
    }
}

/// 检索响应的 wire 形状：ids/scores 平行数组 + payloads 以 id 十进制字符串为键。
/// EN: Wire shape of the search response: parallel ids/scores arrays plus
/// payloads keyed by the decimal id string.
#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    ids: Vec<u64>,
    #[serde(default)]
    scores: Vec<f64>,
    #[serde(default)]
    payloads: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl SearchResponse {
    fn into_result(self) -> SearchResult {
        let hits = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, &id)| SearchHit {
                id,
                score: self.scores.get(i).copied().unwrap_or(0.0),
                payload: self.payloads.get(&id.to_string()).cloned(),
            })
            .collect();
        SearchResult { hits }
    }
}

/// 融合响应的 fusion 参数块（与 ids 对齐的平行数组之外单列）。
/// EN: The fusion parameter block of a fusion response (separate from the
/// parallel arrays aligned with ids).
#[derive(Deserialize, Default)]
struct FusionParamsWire {
    #[serde(default)]
    mode: String,
    #[serde(default)]
    rrf_k: Option<u32>,
    #[serde(default)]
    alpha: Option<f32>,
}

/// 融合检索响应的 wire 形状：`dense_scores`/`sparse_scores` 是与 ids 对齐的
/// 平行数组，缺席模态为 JSON null。
/// EN: Wire shape of a fusion response: `dense_scores`/`sparse_scores` are
/// arrays parallel to ids, with JSON null for an absent modality.
#[derive(Deserialize)]
struct FusionResponse {
    #[serde(default)]
    ids: Vec<u64>,
    #[serde(default)]
    scores: Vec<f64>,
    #[serde(default)]
    dense_scores: Vec<Option<f64>>,
    #[serde(default)]
    sparse_scores: Vec<Option<f64>>,
    #[serde(default)]
    metric: String,
    #[serde(default)]
    fusion: FusionParamsWire,
    #[serde(default)]
    payloads: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl FusionResponse {
    /// 校验平行数组宽度：scores 必须与 ids 等宽；模态分数组缺席（空）时容忍，
    /// 出现时必须等宽。宽度错位说明响应被截断/损坏，继续按索引取值会把分数
    /// 静默错配给别的 id，这里改为显式失败（`InvalidResponse` 而非 `Transport`：
    /// 确定性协议违反，不可重试）。
    /// EN: Checks parallel-array widths: scores must equal ids in length;
    /// modality score arrays are tolerated when absent (empty) but must match
    /// when present. Misaligned widths mean a truncated/corrupt response where
    /// index-based access would silently pair scores with the wrong ids, so
    /// fail explicitly (`InvalidResponse`, not `Transport`: a deterministic
    /// protocol violation that must not be retried).
    fn validate_parity(&self) -> Result<(), HouyiError> {
        let ids = self.ids.len();
        if self.scores.len() != ids {
            return Err(HouyiError::InvalidResponse(format!(
                "fusion response scores has {} entries for {ids} ids",
                self.scores.len()
            )));
        }
        let check = |name: &str, len: usize| {
            if len != 0 && len != ids {
                return Err(HouyiError::InvalidResponse(format!(
                    "fusion response {name} has {len} entries for {ids} ids"
                )));
            }
            Ok(())
        };
        check("dense_scores", self.dense_scores.len())?;
        check("sparse_scores", self.sparse_scores.len())
    }

    fn into_result(self) -> FusionResult {
        let hits = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, &id)| FusionHit {
                id,
                score: self.scores.get(i).copied().unwrap_or(0.0),
                dense_score: self.dense_scores.get(i).copied().flatten(),
                sparse_score: self.sparse_scores.get(i).copied().flatten(),
                payload: self.payloads.get(&id.to_string()).cloned(),
            })
            .collect();
        FusionResult {
            hits,
            metric: self.metric,
            fusion_mode: self.fusion.mode,
            rrf_k: self.fusion.rrf_k,
            alpha: self.fusion.alpha,
        }
    }
}

/// 建集合时的 HNSW 参数三元组（必须三个全给，否则服务端 400）。
/// EN: HNSW parameter triple for explicit creation (all three or the server
/// returns 400).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

/// 显式建集合的选项（`Default` 等价服务端默认：余弦 + 默认后端 + 默认 HNSW）。
///
/// 显式创建把 dims/metric/后端/HNSW 参数一次定型：metric 与后端不可变，
/// 之后要改只能删除重建（旧图按旧度量构建，混用会让排序失去意义）。
/// EN: Options for explicit creation (`Default` equals server defaults: cosine,
/// default backend, default HNSW). Explicit creation fixes dims/metric/backend
/// and HNSW parameters at once; metric and backend are immutable afterwards —
/// changing them requires drop-and-recreate (old graphs were built under the old
/// metric, and mixing them would make ordering meaningless).
#[derive(Debug, Clone, Default)]
pub struct CreateCollectionOptions {
    /// `l2` / `cosine` / `ip` / `l1`（大小写不敏感，常用别名见服务端）。
    /// EN: l2/cosine/ip/l1 (case-insensitive; server accepts common aliases).
    pub metric: Option<String>,
    /// 后端类型：hnsw / pq-flat / flat / clustered / flat-clustered / hii-l0。
    /// 缺省取服务启动配置的默认后端。pq-flat 仅支持 l2、hii-l0 仅支持 cosine
    /// （不匹配服务端返回 400）。
    /// EN: Backend kind: hnsw / pq-flat / flat / clustered / flat-clustered /
    /// hii-l0. Omitted means the server's startup default backend. pq-flat is
    /// L2-only and hii-l0 Cosine-only (mismatches return 400).
    pub index_type: Option<String>,
    /// HNSW 三元组：仅 `index_type = Some("hnsw")` 时有效，三个字段必须全给；
    /// 与其他后端组合会被服务端拒绝（400）。
    /// EN: HNSW triple: valid only with `index_type = Some("hnsw")`, and all
    /// three fields are required; combining it with another backend is rejected
    /// by the server (400).
    pub hnsw: Option<HnswConfig>,
    /// 向量存储编码：`f32`（缺省）或 `f16`（仅 hnsw，省一半向量内存）。创建
    /// 后不可变；非 f32 与非 hnsw 组合会被服务端拒绝（400）。
    /// EN: Vector storage encoding: `f32` (default) or `f16` (hnsw only, halves
    /// vector memory). Immutable after creation; non-f32 with a non-hnsw backend
    /// is rejected by the server (400).
    pub vector_dtype: Option<String>,
    /// 图侧量化：`none`（缺省）、`int8` 或 `rabitq`（仅 hnsw，压缩图内距离
    /// 工作集；主表保留原始向量并在检索时精确重排）。`rabitq` 为 1 bit/dim，
    /// 仅支持 `metric = l2 | cosine`（`ip`/`l1` 返回 400）。创建后不可变；
    /// 非 none 与非 hnsw 组合会被服务端拒绝（400）。
    /// EN: Graph-side quantization: `none` (default), `int8`, or `rabitq`
    /// (hnsw only, shrinks the in-graph distance working set; the main store
    /// keeps raw vectors and reranks exactly at search time). `rabitq` packs
    /// 1 bit/dim and supports `metric = l2 | cosine` only (`ip`/`l1` return
    /// 400). Immutable after creation; non-none with a non-hnsw backend is
    /// rejected by the server (400).
    pub quantization: Option<String>,
    /// 稀疏旁路索引：`none`（缺省）或 `bm25`（客户端给定词项，仅 hnsw/flat，
    /// 创建后不可变）。`bm25` 与其他后端组合返回 400。
    /// EN: Sparse side index: `none` (default) or `bm25` (client-supplied
    /// terms, hnsw/flat-only, immutable after creation). Combining `bm25` with
    /// another backend returns 400.
    pub sparse: Option<String>,
}

/// 建集合响应：服务端回显实际生效的参数（同名已存在返回 409）。
/// EN: Creation response: the server echoes the effective parameters (an
/// existing name yields 409).
#[derive(Debug, Clone, Deserialize)]
pub struct CreateCollectionResult {
    pub created: bool,
    pub collection: String,
    pub dims: usize,
    pub metric: String,
    pub index_type: String,
    /// 实际生效的存储编码（`f32` | `f16`）；旧服务端不返回时为 None。
    /// EN: Effective storage encoding (`f32` | `f16`); None from older servers.
    #[serde(default)]
    pub vector_dtype: Option<String>,
    /// 实际生效的图侧量化（`none` | `int8` | `rabitq`）；旧服务端不返回时为
    /// None。
    /// EN: Effective graph-side quantization (`none` | `int8` | `rabitq`);
    /// None from older servers.
    #[serde(default)]
    pub quantization: Option<String>,
    /// 实际生效的稀疏索引模式（`none` | `bm25`）；旧服务端不返回时为 None
    /// （按可选字段容忍，不得报错）。
    /// EN: Effective sparse index mode (`none` | `bm25`); None from older
    /// servers (tolerated as an optional field, never an error).
    #[serde(default)]
    pub sparse: Option<String>,
    /// 仅 `index_type=hnsw` 时存在。
    /// EN: Present only for `index_type=hnsw`.
    #[serde(default)]
    pub hnsw: Option<HnswConfig>,
}

/// 检索/混合检索可选参数（`Default` 即全默认）。
/// EN: Optional search/hybrid-search parameters (`Default` means all defaults).
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub ef_search: Option<usize>,
    pub include_payload: bool,
    pub filters: Vec<Filter>,
}

/// 融合检索可选参数（`Default` 即 rrf + 服务端缺省 rrf_k=60/alpha=0.5）。
///
/// `fusion` 取 `rrf`（缺省）或 `weighted`；`rrf_k` 仅 rrf 模式参与计算，
/// `alpha` 仅 weighted 模式参与计算，但显式给出的值服务端一律严格校验
/// （越界 400，即使当前模式不参与计算）。
/// EN: Optional fusion-search parameters (`Default` is rrf plus the server
/// defaults rrf_k=60/alpha=0.5). `fusion` is `rrf` (default) or `weighted`;
/// `rrf_k` scores only under rrf and `alpha` only under weighted, yet the
/// server strictly validates explicit values (400 when out of range, even
/// under the inactive mode).
#[derive(Debug, Clone, Default)]
pub struct FusionOptions {
    pub ef_search: Option<usize>,
    pub include_payload: bool,
    pub filters: Vec<Filter>,
    pub fusion: Option<String>,
    pub rrf_k: Option<u32>,
    pub alpha: Option<f32>,
}

/// 一个融合命中：id + 融合分（仅排序语义）+ 两路原始分 + 可选 payload。
/// EN: One fusion hit: id + fused score (ranking only) + both raw modality
/// scores + optional payload.
#[derive(Debug, Clone)]
pub struct FusionHit {
    pub id: u64,
    pub score: f64,
    /// 按集合度量换算的稠密分（cosine/l2/l1 为 `1-d`、ip 为原始点积）；
    /// 仅稠密一路命中时存在。
    /// EN: Dense score converted under the collection metric (cosine/l2/l1 as
    /// `1-d`, ip as the raw dot product); present only when the dense path hit.
    pub dense_score: Option<f64>,
    /// BM25 原始分；仅稀疏一路命中时存在。
    /// EN: Raw BM25 score; present only when the sparse path hit.
    pub sparse_score: Option<f64>,
    pub payload: Option<HashMap<String, serde_json::Value>>,
}

/// 融合检索响应：hits 按融合分从高到低排列。
/// EN: Fusion response: hits ordered by fused score, highest first.
#[derive(Debug, Clone, Default)]
pub struct FusionResult {
    pub hits: Vec<FusionHit>,
    /// 仅用于解释 `dense_score` 的度量口径。
    /// EN: Metric that explains `dense_score` only.
    pub metric: String,
    /// 生效的融合模式（`rrf` | `weighted`）。
    /// EN: Effective fusion mode (`rrf` | `weighted`).
    pub fusion_mode: String,
    /// 仅 rrf 模式回显。
    /// EN: Echoed only under rrf.
    pub rrf_k: Option<u32>,
    /// 仅 weighted 模式回显。
    /// EN: Echoed only under weighted.
    pub alpha: Option<f32>,
}

/// HouyiDB 客户端入口（同步阻塞；`ureq::Agent` 连接池复用）。
/// EN: HouyiDB client entry point (synchronous, blocking; `ureq::Agent`
/// connection pool reuse).
#[derive(Clone)]
pub struct Client {
    agent: ureq::Agent,
    base_url: String,
    /// 可选管理 token：只附在 `/v1/admin/*` 请求上，普通读写请求绝不携带。
    /// EN: Optional admin token: attached only to `/v1/admin/*` requests, never to
    /// normal read/write calls.
    admin_token: Option<String>,
}

impl Client {
    /// 附加管理 token（仅用于管理端点，如删除整个 collection）。
    ///
    /// token 存于客户端，只在管理请求上发送 `Authorization: Bearer <token>`；
    /// 空串等价于未配置（不发送空凭据），未配置时管理请求照常发出（由服务端
    /// 401/403 决定，不在客户端 fail-fast——服务端可能未启用鉴权）。
    /// EN: Attach an admin token (used only by admin endpoints such as dropping a
    /// whole collection). It is stored on the client and sent as
    /// `Authorization: Bearer <token>` on admin requests only; an empty token
    /// means "unset" (no header), so the request still goes out and the server
    /// decides 401/403 (no client-side fail-fast — the server may run without
    /// auth).
    pub fn with_admin_token(mut self, token: impl Into<String>) -> Self {
        let token = token.into();
        self.admin_token = (!token.is_empty()).then_some(token);
        self
    }

    /// 构造客户端；`base_url` 如 `http://127.0.0.1:7000`。
    /// EN: Constructs a client; `base_url` such as `http://127.0.0.1:7000`.
    pub fn new(base_url: &str) -> Self {
        Self::with_timeout(base_url, Duration::from_secs(30))
    }

    /// 构造客户端并指定单次请求超时（连接+读取）。
    /// EN: Constructs a client with a per-request timeout (connect + read).
    pub fn with_timeout(base_url: &str, timeout: Duration) -> Self {
        // 关闭「状态码即错误」：SDK 需要把服务端纯文本诊断放进
        // HouyiError::Http，而 ureq 3 的 Error::StatusCode 不携带正文。
        // EN: disable "status code is error": the SDK needs the server's
        // plain-text diagnostic in HouyiError::Http, while ureq 3's
        // Error::StatusCode does not carry the body.
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                .timeout_global(Some(timeout))
                .http_status_as_error(false)
                .build(),
        );
        Self {
            agent,
            base_url: base_url.trim_end_matches('/').to_string(),
            admin_token: None,
        }
    }

    /// 返回一个 named collection 的操作句柄。
    /// EN: Returns a handle for a named collection.
    pub fn collection(&self, name: &str) -> Collection {
        Collection {
            agent: self.agent.clone(),
            // URL 路径段必须转义：未编码的 / ? # 会改写路由结构，
            // 表现为莫名其妙的 404 且 collection 永远建不出来。
            // EN: URL path segments must be escaped: an unencoded / ? # rewrites
            // the route structure, surfacing as a puzzling 404 with the
            // collection never being created.
            base: format!(
                "{}/v1/collections/{}",
                self.base_url,
                quote_path_segment(name)
            ),
            name: name.to_string(),
        }
    }

    /// 服务健康检查（不可达/非 ok 一律 false，不返回错误）。
    /// EN: Health check (unreachable/not-ok is always false; no error is returned).
    pub fn health(&self) -> bool {
        self.get("/v1/health")
            .ok()
            .and_then(|v: serde_json::Value| v.get("status")?.as_str().map(str::to_string))
            .as_deref()
            == Some("ok")
    }

    /// 列出全部 collection。
    /// EN: Lists all collections.
    pub fn list_collections(&self) -> Result<Vec<serde_json::Value>, HouyiError> {
        self.get("/v1/collections")
    }

    /// 主索引双速写入指标（hot/pending/flush_count 等，可观测性用）。
    /// EN: Primary index dual-speed write metrics (hot/pending/flush_count, for
    /// observability).
    pub fn index_status(&self) -> Result<serde_json::Value, HouyiError> {
        self.get("/v1/index/status")
    }

    /// 显式创建 collection（可指定度量/后端/HNSW 参数；需要写权限）。
    ///
    /// 与隐式创建（首次 upsert）不同，这里把 dims、metric、后端与 HNSW 参数
    /// 一次定型并由服务端回显；调用的都是普通写路径，不带管理 token。
    /// 同名已存在返回 409，参数非法返回 400，follower/只读返回 503。
    /// EN: Explicitly create a collection (metric/backend/HNSW parameters are
    /// settable; requires write permission). Unlike implicit creation (first
    /// upsert), dims, metric, backend and HNSW parameters are fixed here and
    /// echoed back. This is a normal write path, so the admin token is not
    /// attached. An existing name yields 409, invalid parameters 400, and
    /// followers/read-only nodes 503.
    pub fn create_collection(
        &self,
        name: &str,
        dims: usize,
        opts: &CreateCollectionOptions,
    ) -> Result<CreateCollectionResult, HouyiError> {
        let mut body = serde_json::json!({ "dims": dims });
        let map = body.as_object_mut().expect("request is an object");
        if let Some(metric) = &opts.metric {
            map.insert("metric".to_string(), serde_json::json!(metric));
        }
        if let Some(index_type) = &opts.index_type {
            map.insert("index_type".to_string(), serde_json::json!(index_type));
        }
        if let Some(vector_dtype) = &opts.vector_dtype {
            map.insert("vector_dtype".to_string(), serde_json::json!(vector_dtype));
        }
        if let Some(quantization) = &opts.quantization {
            map.insert("quantization".to_string(), serde_json::json!(quantization));
        }
        if let Some(sparse) = &opts.sparse {
            map.insert("sparse".to_string(), serde_json::json!(sparse));
        }
        if let Some(hnsw) = &opts.hnsw {
            map.insert("hnsw_m".to_string(), serde_json::json!(hnsw.m));
            map.insert(
                "hnsw_ef_construction".to_string(),
                serde_json::json!(hnsw.ef_construction),
            );
            map.insert(
                "hnsw_ef_search".to_string(),
                serde_json::json!(hnsw.ef_search),
            );
        }
        checked(
            self.agent
                .put(&format!(
                    "{}/v1/collections/{}",
                    self.base_url,
                    quote_path_segment(name)
                ))
                .send_json(body),
        )
    }

    /// 删除整个 collection（不可逆；需要 admin 权限）。
    ///
    /// 服务端是 WAL-first 删除：记录先落盘，崩溃后重放与 follower 复制都能
    /// 收敛到「已删除」；对不存在/已删除的 collection 返回 404。
    /// EN: Drop an entire collection (irreversible; admin permission required).
    /// The server deletes WAL-first, so crash replay and follower replication
    /// converge to "gone"; an unknown/already-dropped collection yields 404.
    pub fn delete_collection(&self, name: &str) -> Result<(), HouyiError> {
        let url = format!(
            "{}/v1/admin/collections/{}",
            self.base_url,
            quote_path_segment(name)
        );
        let mut req = self.agent.delete(&url);
        if let Some(token) = &self.admin_token {
            req = req.header("Authorization", &format!("Bearer {token}"));
        }
        let _: serde_json::Value = checked(req.call())?;
        Ok(())
    }

    fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, HouyiError> {
        checked(self.agent.get(&format!("{}{}", self.base_url, path)).call())
    }
}

/// 一个 named collection 的操作句柄（经 [`Client::collection`] 获取）。
/// EN: Handle for a named collection (obtained via [`Client::collection`]).
pub struct Collection {
    agent: ureq::Agent,
    base: String,
    name: String,
}

impl Collection {
    /// collection 名。
    /// EN: Collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 写入一批向量（collection 首次写入时隐式创建）。返回插入条数。
    /// EN: Writes a batch of vectors (the collection is created implicitly on
    /// first write). Returns the number of inserted rows.
    pub fn upsert(&self, vectors: &[Vector]) -> Result<usize, HouyiError> {
        // 稀疏词项在发出请求前按服务端口径校验：越界会在服务端整批 400，
        // 这里提前失败能给出具体到向量 id 的诊断。
        // EN: Validate sparse terms against the server's limits before sending:
        // out-of-range values would 400 the whole batch server-side, so fail
        // early with diagnostics that name the offending vector id.
        for vector in vectors {
            if let Some(terms) = &vector.sparse {
                validate_sparse_terms(
                    terms,
                    MAX_SPARSE_TERMS_PER_DOC,
                    &format!("vector {}", vector.id),
                )?;
            }
        }
        #[derive(Deserialize)]
        struct Inserted {
            inserted: usize,
        }
        let body: Inserted = checked(
            self.agent
                .post(&format!("{}/upsert", self.base))
                .send_json(serde_json::json!({ "vectors": vectors })),
        )?;
        Ok(body.inserted)
    }

    /// 纯向量 top-k 检索。
    /// EN: Pure vector top-k search.
    pub fn search(
        &self,
        vector: &[f32],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<SearchResult, HouyiError> {
        let body: SearchResponse = checked(
            self.agent
                .post(&format!("{}/search", self.base))
                .send_json(search_request(vector, k, opts, false)),
        )?;
        Ok(body.into_result())
    }

    /// 向量 × 标量过滤的混合检索（filters 取交集，仅 hnsw/flat 后端支持）。
    /// EN: Hybrid vector × scalar filter search (filters are intersected; only
    /// hnsw/flat backends are supported).
    pub fn hybrid_search(
        &self,
        vector: &[f32],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<SearchResult, HouyiError> {
        let body: SearchResponse = checked(
            self.agent
                .post(&format!("{}/hybrid/search", self.base))
                .send_json(search_request(vector, k, opts, true)),
        )?;
        Ok(body.into_result())
    }

    /// 稠密 + 稀疏融合检索（仅创建时开启 `sparse=bm25` 的集合）。
    ///
    /// 两路各自过取 4×k 后融合：rrf（缺省，`rrf_k` 平滑）或 weighted
    /// （两路 min-max 归一后 `alpha*dense + (1-alpha)*sparse`）。
    /// 稀疏查询词项为空或集合未启用稀疏时服务端返回 400。
    /// EN: Dense + sparse fusion search (only for collections created with
    /// `sparse=bm25`). Both paths over-fetch 4×k, then fuse: rrf (default, with
    /// `rrf_k` smoothing) or weighted (per-path min-max, then
    /// `alpha*dense + (1-alpha)*sparse`). The server returns 400 for an empty
    /// sparse query or a collection without sparse enabled.
    pub fn fusion_search(
        &self,
        vector: &[f32],
        sparse: &[(u32, f32)],
        k: usize,
        opts: &FusionOptions,
    ) -> Result<FusionResult, HouyiError> {
        // 与服务端同一份约束：查询词项非空且不超限、term/权重合法、
        // alpha∈[0,1]、rrf_k∈[1,1000]；先在本端拒绝，省一次必然 400 的往返。
        // EN: The same constraints the server enforces: a non-empty, bounded
        // sparse query with legal term/weight pairs, alpha within [0,1], and
        // rrf_k within [1,1000] — rejected here to spare a guaranteed-400 round
        // trip.
        if sparse.is_empty() {
            return Err(HouyiError::InvalidRequest(
                "fusion_search requires a non-empty sparse query".to_string(),
            ));
        }
        validate_sparse_terms(sparse, MAX_SPARSE_QUERY_TERMS, "sparse query")?;
        if let Some(alpha) = opts.alpha {
            if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                return Err(HouyiError::InvalidRequest(format!(
                    "alpha {alpha} must be finite and within [0, 1]"
                )));
            }
        }
        if let Some(rrf_k) = opts.rrf_k {
            if rrf_k == 0 || rrf_k > MAX_RRF_K {
                return Err(HouyiError::InvalidRequest(format!(
                    "rrf_k {rrf_k} outside 1..={MAX_RRF_K}"
                )));
            }
        }
        let body: FusionResponse = checked(
            self.agent
                .post(&format!("{}/fusion/search", self.base))
                .send_json(fusion_request(vector, sparse, k, opts)),
        )?;
        body.validate_parity()?;
        Ok(body.into_result())
    }

    /// 按 id 和/或过滤条件删除（幂等；两者取并集）。返回真实删除数。
    ///
    /// 按条件删除由服务端在标量页上求值：标量行随 upsert 同步落页，因此
    /// 刚写入、尚在异步构图队列中的向量也能被命中删除——这是「先检索拿 id
    /// 再删」无法保证的（检索在队列排空前看不到新写入）。
    /// EN: Deletes by id and/or filters (idempotent; the two sets are unioned).
    /// Returns the number actually deleted.
    ///
    /// EN: Filter deletes are evaluated server-side on scalar pages: scalar rows
    /// land synchronously with upsert, so vectors just written and still in the
    /// async graph-build queue can also be matched and deleted, which "search
    /// first, delete by id" cannot guarantee (search cannot see new writes until
    /// the queue drains).
    pub fn delete(&self, ids: &[u64], filters: &[Filter]) -> Result<usize, HouyiError> {
        if ids.is_empty() && filters.is_empty() {
            return Err(HouyiError::Transport(
                "delete requires at least one of ids or filters".to_string(),
            ));
        }
        #[derive(Deserialize)]
        struct Deleted {
            deleted: usize,
        }
        let body: Deleted = checked(
            self.agent
                .post(&format!("{}/delete", self.base))
                .send_json(serde_json::json!({ "ids": ids, "filters": filters })),
        )?;
        Ok(body.deleted)
    }

    /// collection 状态（indexed/pending/dims/index_type/scalar_row_count）。
    /// EN: Collection status (indexed/pending/dims/index_type/scalar_row_count).
    pub fn status(&self) -> Result<serde_json::Value, HouyiError> {
        checked(self.agent.get(&format!("{}/status", self.base)).call())
    }
}

fn search_request(
    vector: &[f32],
    k: usize,
    opts: &SearchOptions,
    hybrid: bool,
) -> serde_json::Value {
    let mut req = serde_json::json!({ "vector": vector, "k": k });
    let map = req.as_object_mut().expect("request is an object");
    if hybrid && !opts.filters.is_empty() {
        map.insert("filters".to_string(), serde_json::json!(opts.filters));
    }
    if let Some(ef) = opts.ef_search {
        map.insert("ef_search".to_string(), serde_json::json!(ef));
    }
    if opts.include_payload {
        map.insert("include_payload".to_string(), serde_json::json!(true));
    }
    req
}

/// 构造融合检索请求体：稀疏词项序列化为 `[term, weight]` 二元数组；
/// 融合模式与调参旋钮只在显式给出时携带（缺省由服务端补齐）。
/// EN: Builds the fusion request body: sparse terms serialize as
/// `[term, weight]` pairs; the fusion mode and tuning knobs are carried only
/// when explicitly set (the server fills the defaults otherwise).
fn fusion_request(
    vector: &[f32],
    sparse: &[(u32, f32)],
    k: usize,
    opts: &FusionOptions,
) -> serde_json::Value {
    let mut req = serde_json::json!({ "vector": vector, "sparse": sparse, "k": k });
    let map = req.as_object_mut().expect("request is an object");
    if !opts.filters.is_empty() {
        map.insert("filters".to_string(), serde_json::json!(opts.filters));
    }
    if let Some(ef) = opts.ef_search {
        map.insert("ef_search".to_string(), serde_json::json!(ef));
    }
    if opts.include_payload {
        map.insert("include_payload".to_string(), serde_json::json!(true));
    }
    if let Some(fusion) = &opts.fusion {
        map.insert("fusion".to_string(), serde_json::json!(fusion));
    }
    if let Some(rrf_k) = opts.rrf_k {
        map.insert("rrf_k".to_string(), serde_json::json!(rrf_k));
    }
    if let Some(alpha) = opts.alpha {
        map.insert("alpha".to_string(), serde_json::json!(alpha));
    }
    req
}

/// ureq 错误归一：HTTP 状态错误取服务端纯文本诊断；其余为传输层错误。
/// EN: Normalizes ureq errors: HTTP status errors take the server's plain-text
/// diagnostic; everything else is a transport error.
fn checked<T: serde::de::DeserializeOwned>(
    result: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<T, HouyiError> {
    match result {
        Ok(mut resp) => {
            let status = resp.status();
            if status.is_success() {
                resp.body_mut()
                    .read_json::<T>()
                    .map_err(|e| HouyiError::Transport(e.to_string()))
            } else {
                let message = resp
                    .body_mut()
                    .read_to_string()
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                Err(HouyiError::Http {
                    status: status.as_u16(),
                    message,
                })
            }
        }
        Err(e) => Err(HouyiError::Transport(e.to_string())),
    }
}

/// RFC 3986 unreserved（A-Za-z0-9-._~）以外的字节一律 %XX（UTF-8）。
/// 与 Python `quote(safe="")` 同口径，避免引入 percent-encoding 依赖。
/// EN: Percent-encodes every byte outside RFC 3986 unreserved (A-Za-z0-9-._~)
/// as %XX (UTF-8), matching Python `quote(safe="")` and avoiding a
/// percent-encoding dependency.
fn quote_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// stable_id 五方对拍：期望值由 Python SDK（hashlib.blake2b(digest_size=8)）
    /// 生成，与 TS/Go SDK、服务端网关同算法。跨语言一致是 delete 重放的前提。
    /// EN: stable_id cross-check across five parties: the expected values are
    /// generated by the Python SDK (hashlib.blake2b(digest_size=8)); the same
    /// algorithm as the TS/Go SDKs and the server gateway. Cross-language
    /// agreement is what makes delete replay possible.
    #[test]
    fn stable_id_matches_python() {
        let cases: &[(&str, u64)] = &[
            ("", 7252660547403494068),
            ("hello", 2861735919082615933),
            ("node-42", 6694704320175634731),
            ("苹果香蕉", 4490440666094115033),
            ("doc-1", 6265767295393789249),
            ("a/b?c#d", 4199517732852746375),
        ];
        for (text, want) in cases {
            assert_eq!(stable_id(text), *want, "stable_id({text:?})");
        }
    }

    #[test]
    fn stable_id_deterministic_positive_half() {
        let a = stable_id("same-input");
        assert_eq!(stable_id("same-input"), a);
        assert!(a <= U64_MASK);
    }

    #[test]
    fn quote_path_segment_encodes_reserved() {
        assert_eq!(quote_path_segment("docs"), "docs");
        assert_eq!(quote_path_segment("a/b?c#d"), "a%2Fb%3Fc%23d");
        assert_eq!(quote_path_segment("空 间"), "%E7%A9%BA%20%E9%97%B4");
    }

    /// 起一个只服务一次请求的本地监听器，返回 (base_url, 请求原文 join handle)。
    ///
    /// 按 Content-Length 读满请求体再应答：只停在 header 结束会让 PUT/POST
    /// 的 body 断言依赖 TCP 分片时机（偶发截断）。
    /// EN: One-shot local listener returning (base_url, join handle with the raw
    /// request). It reads the full body per Content-Length before replying:
    /// stopping at the header terminator would make PUT/POST body assertions
    /// depend on TCP segmentation timing (flaky truncation).
    fn serve_one_request(body: &'static str) -> (String, std::thread::JoinHandle<String>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).expect("read request");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                let Some(head_end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let header_len = head_end + 4;
                let content_length = String::from_utf8_lossy(&request[..header_len])
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        (name.eq_ignore_ascii_case("content-length"))
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                if request.len() >= header_len + content_length {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            String::from_utf8_lossy(&request).to_string()
        });
        (format!("http://{addr}"), handle)
    }

    /// 显式建集合：PUT 普通写路径（不带管理 token），metric/后端/HNSW 三元组
    /// 逐个落进 wire body，响应回显实际生效的度量。
    /// EN: explicit create uses the normal PUT write path (no admin token) and
    /// lands metric/backend/HNSW triple in the wire body; the response echoes
    /// the effective metric.
    #[test]
    fn create_collection_sends_metric_backend_and_hnsw() {
        let (base, server) = serve_one_request(
            r#"{"created":true,"collection":"docs","dims":4,"metric":"ip","index_type":"hnsw","vector_dtype":"f16","quantization":"int8","hnsw":{"m":16,"ef_construction":100,"ef_search":32}}"#,
        );
        let result = Client::new(&base)
            .with_admin_token("tok")
            .create_collection(
                "docs",
                4,
                &CreateCollectionOptions {
                    metric: Some("ip".to_string()),
                    index_type: Some("hnsw".to_string()),
                    hnsw: Some(HnswConfig {
                        m: 16,
                        ef_construction: 100,
                        ef_search: 32,
                    }),
                    vector_dtype: Some("f16".to_string()),
                    quantization: Some("int8".to_string()),
                    sparse: None,
                },
            )
            .expect("create");
        assert_eq!(result.metric, "ip");
        assert_eq!(result.vector_dtype.as_deref(), Some("f16"));
        assert_eq!(result.quantization.as_deref(), Some("int8"));
        assert_eq!(result.hnsw.map(|h| h.m), Some(16));
        let request = server.join().expect("server thread");
        assert!(
            request.starts_with("PUT /v1/collections/docs"),
            "unexpected request line: {request}"
        );
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").expect("body separator").1)
                .expect("request body is JSON");
        assert_eq!(body["dims"], 4);
        assert_eq!(body["metric"], "ip");
        assert_eq!(body["index_type"], "hnsw");
        assert_eq!(body["vector_dtype"], "f16");
        assert_eq!(body["quantization"], "int8");
        assert_eq!(body["hnsw_m"], 16);
        assert_eq!(body["hnsw_ef_construction"], 100);
        assert_eq!(body["hnsw_ef_search"], 32);
        assert!(
            !request.to_ascii_lowercase().contains("authorization:"),
            "create is a normal write and must not carry the admin token: {request}"
        );
    }

    /// rabitq 量化：wire body 原样携带 `"rabitq"`，响应回显同样解码出来。
    /// EN: rabitq quantization: the wire body carries `"rabitq"` verbatim and the
    /// echoed response decodes back to the same value.
    #[test]
    fn create_collection_sends_rabitq_quantization() {
        let (base, server) = serve_one_request(
            r#"{"created":true,"collection":"docs","dims":8,"metric":"cosine","index_type":"hnsw","vector_dtype":"f32","quantization":"rabitq","hnsw":{"m":16,"ef_construction":100,"ef_search":32}}"#,
        );
        let result = Client::new(&base)
            .with_admin_token("tok")
            .create_collection(
                "docs",
                8,
                &CreateCollectionOptions {
                    metric: Some("cosine".to_string()),
                    index_type: Some("hnsw".to_string()),
                    hnsw: Some(HnswConfig {
                        m: 16,
                        ef_construction: 100,
                        ef_search: 32,
                    }),
                    vector_dtype: Some("f32".to_string()),
                    quantization: Some("rabitq".to_string()),
                    sparse: None,
                },
            )
            .expect("create");
        assert_eq!(result.metric, "cosine");
        assert_eq!(result.quantization.as_deref(), Some("rabitq"));
        let request = server.join().expect("server thread");
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").expect("body separator").1)
                .expect("request body is JSON");
        assert_eq!(body["metric"], "cosine");
        assert_eq!(body["quantization"], "rabitq");
    }

    /// 不设置 quantization 时：wire body 不出现该键（保持旧服务端兼容），
    /// 且旧服务端响应缺席该字段时结果解码为 None（不创建自有键）。
    /// EN: when quantization is unset the wire body omits the key (old-server
    /// compatible), and a response from an older server without the field decodes
    /// to None (no key of our own is fabricated).
    #[test]
    fn create_collection_omits_quantization_when_unset() {
        let (base, server) = serve_one_request(
            r#"{"created":true,"collection":"docs","dims":4,"metric":"ip","index_type":"hnsw","hnsw":{"m":16,"ef_construction":100,"ef_search":32}}"#,
        );
        let result = Client::new(&base)
            .create_collection(
                "docs",
                4,
                &CreateCollectionOptions {
                    metric: Some("ip".to_string()),
                    index_type: Some("hnsw".to_string()),
                    hnsw: Some(HnswConfig {
                        m: 16,
                        ef_construction: 100,
                        ef_search: 32,
                    }),
                    vector_dtype: None,
                    quantization: None,
                    sparse: None,
                },
            )
            .expect("create");
        assert_eq!(result.quantization, None);
        assert_eq!(result.vector_dtype, None);
        let request = server.join().expect("server thread");
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").expect("body separator").1)
                .expect("request body is JSON");
        assert!(
            body.get("quantization").is_none(),
            "unset quantization must omit the wire key: {body}"
        );
        assert!(
            body.get("vector_dtype").is_none(),
            "unset vector_dtype must omit the wire key: {body}"
        );
        assert!(
            body.get("sparse").is_none(),
            "unset sparse must omit the wire key: {body}"
        );
    }

    /// 稀疏创建：wire body 携带 `sparse`，响应回显生效模式。
    /// EN: Sparse creation: the wire body carries `sparse` and the response
    /// echoes the effective mode.
    #[test]
    fn create_collection_sends_sparse() {
        let (base, server) = serve_one_request(
            r#"{"created":true,"collection":"docs","dims":8,"metric":"cosine","index_type":"hnsw","sparse":"bm25"}"#,
        );
        let result = Client::new(&base)
            .create_collection(
                "docs",
                8,
                &CreateCollectionOptions {
                    sparse: Some("bm25".to_string()),
                    ..Default::default()
                },
            )
            .expect("create");
        assert_eq!(result.sparse.as_deref(), Some("bm25"));
        let request = server.join().expect("server thread");
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").expect("body separator").1)
                .expect("request body is JSON");
        assert_eq!(body["dims"], 8);
        assert_eq!(body["sparse"], "bm25");
    }

    /// per-vector 稀疏形状：None 省略键（不改）、`Some([])` 为 `[]`（清空）、
    /// 词项队序列化为 `[term, weight]` 二元数组。
    /// EN: per-vector sparse shape: None omits the key (unchanged), `Some([])`
    /// serializes as `[]` (clear), and pairs as `[term, weight]` arrays.
    #[test]
    fn vector_sparse_wire_shape() {
        let unset = Vector {
            id: 1,
            values: vec![1.0],
            label: None,
            payload: None,
            sparse: None,
        };
        let wire = serde_json::to_value(&unset).expect("serialize");
        assert!(
            wire.get("sparse").is_none(),
            "unset sparse must omit the key: {wire}"
        );

        let set = Vector {
            id: 2,
            values: vec![1.0],
            label: None,
            payload: None,
            sparse: Some(vec![(3, 1.5), (5, 0.25)]),
        };
        let wire = serde_json::to_value(&set).expect("serialize");
        assert_eq!(wire["sparse"], serde_json::json!([[3, 1.5], [5, 0.25]]));

        let cleared = Vector {
            id: 3,
            values: vec![1.0],
            label: None,
            payload: None,
            sparse: Some(Vec::new()),
        };
        let wire = serde_json::to_value(&cleared).expect("serialize");
        assert_eq!(wire["sparse"], serde_json::json!([]));
    }

    /// 客户端预校验：非法稀疏词项与融合参数在任一请求发出前就失败
    /// （用例指向不可达端口，若校验漏网会以 Transport 失败而非本断言）。
    /// EN: Client pre-validation: illegal sparse terms and fusion params fail
    /// before any request goes out (cases target an unreachable port, so a
    /// validation miss would surface as Transport instead of this assertion).
    #[test]
    fn fusion_search_rejects_invalid_arguments_before_wire() {
        let client = Client::new("http://127.0.0.1:9");
        let col = client.collection("docs");
        let expect_invalid = |result: Result<FusionResult, HouyiError>| match result {
            Err(HouyiError::InvalidRequest(_)) => {}
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        let valid = [(1u32, 1.0f32)];
        expect_invalid(col.fusion_search(&[0.0], &[], 1, &FusionOptions::default()));
        let over: Vec<(u32, f32)> = (1..=65).map(|t| (t, 1.0)).collect();
        expect_invalid(col.fusion_search(&[0.0], &over, 1, &FusionOptions::default()));
        expect_invalid(col.fusion_search(&[0.0], &[(0, 1.0)], 1, &FusionOptions::default()));
        expect_invalid(col.fusion_search(
            &[0.0],
            &[(MAX_SPARSE_TERM_ID + 1, 1.0)],
            1,
            &FusionOptions::default(),
        ));
        expect_invalid(col.fusion_search(&[0.0], &[(1, 0.0)], 1, &FusionOptions::default()));
        expect_invalid(col.fusion_search(&[0.0], &[(1, f32::NAN)], 1, &FusionOptions::default()));
        expect_invalid(col.fusion_search(
            &[0.0],
            &valid,
            1,
            &FusionOptions {
                alpha: Some(1.5),
                ..Default::default()
            },
        ));
        expect_invalid(col.fusion_search(
            &[0.0],
            &valid,
            1,
            &FusionOptions {
                rrf_k: Some(0),
                ..Default::default()
            },
        ));
        expect_invalid(col.fusion_search(
            &[0.0],
            &valid,
            1,
            &FusionOptions {
                rrf_k: Some(MAX_RRF_K + 1),
                ..Default::default()
            },
        ));
    }

    /// upsert 的 per-vector 稀疏预校验与错误信息中的向量 id。
    /// EN: Per-vector sparse pre-validation on upsert, including the vector id
    /// named in the diagnostics.
    #[test]
    fn upsert_rejects_invalid_sparse_terms_before_wire() {
        let client = Client::new("http://127.0.0.1:9");
        let col = client.collection("docs");
        let vector = |sparse: Option<SparseTerms>| Vector {
            id: 7,
            values: vec![0.0; 4],
            label: None,
            payload: None,
            sparse,
        };
        let expect_invalid = |result: Result<usize, HouyiError>| match result {
            Err(HouyiError::InvalidRequest(message)) => {
                assert!(message.contains("vector 7"), "missing id in: {message}");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        };
        expect_invalid(col.upsert(&[vector(Some(vec![(0, 1.0)]))]));
        expect_invalid(col.upsert(&[vector(Some(vec![(MAX_SPARSE_TERM_ID + 1, 1.0)]))]));
        expect_invalid(col.upsert(&[vector(Some(vec![(1, 0.0)]))]));
        expect_invalid(col.upsert(&[vector(Some(vec![(1, f32::INFINITY)]))]));
        let over: SparseTerms = (1..=1025).map(|t| (t, 1.0)).collect();
        expect_invalid(col.upsert(&[vector(Some(over))]));
        // 空列表是合法输入（清空文档的稀疏词项），校验本身不得拒绝它。
        // EN: An empty list is legal input (clears the document's sparse
        // terms); the validator itself must not reject it.
        validate_sparse_terms(&[], MAX_SPARSE_TERMS_PER_DOC, "vector 7")
            .expect("empty sparse list is legal");
    }

    /// 融合响应平行数组宽度校验：scores 必须与 ids 等宽，模态数组缺席可容忍、
    /// 出现时必须等宽。
    /// EN: Fusion response parallel-array width validation: scores must match
    /// ids, modality arrays are tolerated when absent but must match when
    /// present.
    #[test]
    fn fusion_response_parity_rejects_misaligned_scores() {
        let parse = |json: &str| serde_json::from_str::<FusionResponse>(json).expect("wire");
        parse(r#"{"ids":[1,2],"scores":[1.0,0.5],"dense_scores":[0.1,null],"sparse_scores":[null,9.0]}"#)
            .validate_parity()
            .expect("aligned arrays pass");
        parse(r#"{"ids":[1,2],"scores":[1.0,0.5]}"#)
            .validate_parity()
            .expect("absent modality arrays are tolerated");
        let bad = parse(r#"{"ids":[1,2],"scores":[1.0]}"#);
        assert!(
            matches!(bad.validate_parity(), Err(HouyiError::InvalidResponse(_))),
            "short scores must be rejected"
        );
        let bad = parse(r#"{"ids":[1,2],"scores":[1.0,0.5],"dense_scores":[0.1]}"#);
        assert!(
            matches!(bad.validate_parity(), Err(HouyiError::InvalidResponse(_))),
            "short modality array must be rejected"
        );
    }

    /// 融合检索：路由、请求体（稀疏对/融合参数）与响应（平行分数组含
    /// null、fusion 参数块）逐项对齐。
    /// EN: Fusion search: route, request body (sparse pairs/fusion params), and
    /// response (parallel score arrays with nulls plus the fusion params block)
    /// line up item by item.
    #[test]
    fn fusion_search_wire_and_response() {
        let (base, server) = serve_one_request(
            r#"{"ids":[9,4],"scores":[0.75,0.02],"dense_scores":[0.9,null],"sparse_scores":[null,4.2],"metric":"cosine","fusion":{"mode":"weighted","rrf_k":null,"alpha":0.5},"payloads":{"9":{"text":"nine"}}}"#,
        );
        let result = Client::new(&base)
            .collection("docs")
            .fusion_search(
                &[1.0, 0.0],
                &[(7, 0.5), (8, 0.25)],
                2,
                &FusionOptions {
                    include_payload: true,
                    fusion: Some("weighted".to_string()),
                    alpha: Some(0.5),
                    ..Default::default()
                },
            )
            .expect("fusion");
        assert_eq!(
            result.hits.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![9, 4]
        );
        assert_eq!(result.hits[0].dense_score, Some(0.9));
        assert_eq!(result.hits[0].sparse_score, None);
        assert_eq!(result.hits[1].dense_score, None);
        assert_eq!(result.hits[1].sparse_score, Some(4.2));
        assert_eq!(
            result.hits[0]
                .payload
                .as_ref()
                .and_then(|p| p.get("text"))
                .and_then(|v| v.as_str()),
            Some("nine")
        );
        assert_eq!(result.metric, "cosine");
        assert_eq!(result.fusion_mode, "weighted");
        assert_eq!(result.rrf_k, None);
        assert_eq!(result.alpha, Some(0.5));

        let request = server.join().expect("server thread");
        assert!(
            request.starts_with("POST /v1/collections/docs/fusion/search"),
            "unexpected request line: {request}"
        );
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").expect("body separator").1)
                .expect("request body is JSON");
        assert_eq!(body["vector"], serde_json::json!([1.0, 0.0]));
        assert_eq!(body["sparse"], serde_json::json!([[7, 0.5], [8, 0.25]]));
        assert_eq!(body["k"], 2);
        assert_eq!(body["include_payload"], true);
        assert_eq!(body["fusion"], "weighted");
        assert_eq!(body["alpha"], 0.5);
        assert!(
            body.get("rrf_k").is_none(),
            "unset rrf_k must omit the wire key: {body}"
        );
        assert!(
            body.get("filters").is_none(),
            "empty filters must omit the wire key: {body}"
        );
    }

    /// 空 token 等价未配置（不发空 Authorization）；非空 token 只出现在管理请求上。
    /// EN: an empty token means unset (no empty Authorization header); a non-empty
    /// token is sent on admin requests.
    #[test]
    fn admin_token_empty_means_unset() {
        let (base, server) = serve_one_request(r#"{"deleted":true,"collection":"docs"}"#);
        Client::new(&base)
            .with_admin_token("")
            .delete_collection("docs")
            .expect("empty-token drop");
        let request = server.join().expect("server thread");
        assert!(
            request.starts_with("DELETE /v1/admin/collections/docs"),
            "unexpected request line: {request}"
        );
        assert!(
            !request.to_ascii_lowercase().contains("authorization:"),
            "empty admin token must not produce an Authorization header: {request}"
        );

        let (base, server) = serve_one_request(r#"{"deleted":true,"collection":"docs"}"#);
        Client::new(&base)
            .with_admin_token("tok")
            .delete_collection("docs")
            .expect("token drop");
        let request = server.join().expect("server thread");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer tok"),
            "admin token missing from request: {request}"
        );
    }

    /// 普通读写请求绝不可携带管理 token：把 Authorization 挪进通用 `get`
    /// 之类的回退必须被抓住。
    /// EN: normal read/write requests must never carry the admin token; a
    /// regression that attaches Authorization inside the shared `get` path must
    /// be caught.
    #[test]
    fn admin_token_withheld_from_normal_requests() {
        let (base, server) = serve_one_request(r#"{"deleted":true,"collection":"docs"}"#);
        let _ = Client::new(&base).with_admin_token("tok").health();
        let request = server.join().expect("server thread");
        assert!(
            request.starts_with("GET /v1/health"),
            "unexpected request line: {request}"
        );
        assert!(
            !request.to_ascii_lowercase().contains("authorization:"),
            "admin token must not leak onto normal requests: {request}"
        );
    }
}
