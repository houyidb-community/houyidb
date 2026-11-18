//! 稀疏向量倒排索引与 BM25 打分（P4）。
//!
//! 稀疏词项完全由客户端给出：`(term_id, weight)` 即事实，服务端不做分词、
//! 不做语言处理。索引由正向表（doc → terms）与倒排表（term → postings）组成，
//! 二者在同一把集合级写锁下同步更新，保证「先写即可查」。
//!
//! EN: Sparse-vector inverted index and BM25 scoring (P4). Terms are supplied
//! verbatim by clients as `(term_id, weight)`; the server performs no
//! tokenization or language processing. A forward table (doc -> terms) and an
//! inverted table (term -> postings) are updated together under the collection
//! write lock, guaranteeing read-your-write.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// 稀疏索引模式；`none` 为默认（不建旁路索引）。
/// EN: Sparse index mode; `none` is the default (no side index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMode {
    None,
    Bm25,
}

impl SparseMode {
    /// 稳定线格式串（创建请求/响应、WAL、清单共用）。
    /// EN: Stable wire string shared by create requests/responses, WAL, and the
    /// manifest.
    pub fn as_str(self) -> &'static str {
        match self {
            SparseMode::None => "none",
            SparseMode::Bm25 => "bm25",
        }
    }

    /// 解析创建参数；大小写不敏感，未知值返回 `None` 由调用方映射 400。
    /// EN: Parse a create parameter; case-insensitive, unknown values return
    /// `None` so callers can map to 400.
    pub fn parse_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Some(SparseMode::None),
            "bm25" => Some(SparseMode::Bm25),
            _ => None,
        }
    }

    /// 稳定持久码（0=none，1=bm25），供服务端原子字段使用；与线格式串独立
    /// 演进，避免把字符串解析放进热路径。
    /// EN: Stable persisted code (0=none, 1=bm25) for the server's atomic field;
    /// independent of the wire string so hot paths never re-parse text.
    pub fn code(self) -> u8 {
        match self {
            SparseMode::None => 0,
            SparseMode::Bm25 => 1,
        }
    }

    /// 从持久码还原；未知码返回 `None`（调用方按内存腐坏 fail-closed）。
    /// EN: Decode a persisted code; an unknown code returns `None` and the
    /// caller fails closed (memory corruption).
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(SparseMode::None),
            1 => Some(SparseMode::Bm25),
            _ => None,
        }
    }
}

/// term_id 上限（24 bit）：兼顾 1.6 千万词表与编码紧凑性。
/// EN: Upper term_id bound (24 bits): up to ~16.7M vocabulary entries while
/// keeping the encoding compact.
pub const MAX_SPARSE_TERM_ID: u32 = (1 << 24) - 1;
/// 每文档原始 `(term, weight)` 对数上限（在重复 term 求和之前按输入对数
/// 限流，求和后的不同 term 数只会更少）。
/// EN: Max raw `(term, weight)` pairs per document, enforced before duplicate
/// terms are summed; the distinct count after summing is therefore at most
/// this limit.
pub const MAX_SPARSE_TERMS_PER_DOC: usize = 1024;
/// 每次查询 term 数上限。
/// EN: Max terms per query.
pub const MAX_SPARSE_QUERY_TERMS: usize = 64;
/// BM25 词频饱和参数。
/// EN: BM25 term-frequency saturation parameter.
pub const BM25_K1: f32 = 1.2;
/// BM25 文档长度归一参数。
/// EN: BM25 document-length normalization parameter.
pub const BM25_B: f32 = 0.75;
/// 死 postings 压缩阈值：死行超过存活行的一半时触发重建，防止频繁更新
/// 放大内存。
/// EN: Dead-posting compaction threshold: rebuild once dead rows exceed half
/// of the live rows, bounding the memory blow-up from repeated updates.
const COMPACT_DEAD_RATIO: f64 = 0.5;

/// 内核输入校验错误；服务端边界负责映射 HTTP/pgwire 错误码。
/// EN: Kernel input-validation errors; server boundaries map them to HTTP /
/// pgwire error codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseError {
    /// 超过数量上限（文档 1024 / 查询 64，均按原始输入对数计）。
    /// EN: Term count exceeds the configured limit (both document and query
    /// limits count raw input pairs).
    TooManyTerms { got: usize, limit: usize },
    /// term_id 越界（允许 1..=MAX_SPARSE_TERM_ID）。
    /// EN: term_id outside `1..=MAX_SPARSE_TERM_ID`.
    TermOutOfRange { term: u32 },
    /// 权重非有限值（NaN/Inf）。
    /// EN: Weight is not finite (NaN or infinity).
    NonFiniteWeight { term: u32 },
    /// 权重非正（客户端语义要求 > 0）。
    /// EN: Weight is not positive (the wire contract requires > 0).
    NonPositiveWeight { term: u32 },
    /// 快照中的 term 列表非严格升序（含重复 term）。
    /// EN: Snapshot term list is not strictly ascending (duplicate term).
    TermsNotSorted { term: u32 },
    /// 快照中的 posting doc 列表非严格升序（含重复 doc）。
    /// EN: Snapshot postings are not strictly ascending (duplicate doc).
    PostingsNotSorted { term: u32 },
    /// posting 的 tf 非有限值。
    /// EN: Posting tf is not finite.
    NonFiniteTf { term: u32, doc: u64 },
    /// posting 的 tf 非正。
    /// EN: Posting tf is not positive.
    NonPositiveTf { term: u32, doc: u64 },
    /// 声明的 doc_count 与 postings 中出现的文档数不符。
    /// EN: Declared doc_count differs from the distinct docs seen in postings.
    DocCountMismatch { declared: u64, actual: u64 },
    /// total_len 非有限或为负。
    /// EN: total_len is non-finite or negative.
    InvalidTotalLen,
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseError::TooManyTerms { got, limit } => {
                write!(f, "too many sparse terms: {got} > {limit}")
            }
            SparseError::TermOutOfRange { term } => {
                write!(
                    f,
                    "sparse term id {term} out of range (1..={MAX_SPARSE_TERM_ID})"
                )
            }
            SparseError::NonFiniteWeight { term } => {
                write!(f, "sparse weight for term {term} is not finite")
            }
            SparseError::NonPositiveWeight { term } => {
                write!(f, "sparse weight for term {term} must be > 0")
            }
            SparseError::TermsNotSorted { term } => {
                write!(
                    f,
                    "snapshot sparse terms are not strictly ascending at {term}"
                )
            }
            SparseError::PostingsNotSorted { term } => {
                write!(
                    f,
                    "snapshot postings for term {term} are not strictly ascending"
                )
            }
            SparseError::NonFiniteTf { term, doc } => {
                write!(f, "snapshot tf for term {term} doc {doc} is not finite")
            }
            SparseError::NonPositiveTf { term, doc } => {
                write!(f, "snapshot tf for term {term} doc {doc} must be > 0")
            }
            SparseError::DocCountMismatch { declared, actual } => {
                write!(
                    f,
                    "snapshot sparse doc_count {declared} != distinct docs {actual}"
                )
            }
            SparseError::InvalidTotalLen => write!(f, "snapshot sparse total_len is invalid"),
        }
    }
}

impl std::error::Error for SparseError {}

/// 规范化词项列表：校验范围/权重，重复 term 求和，按 term 升序返回。
/// 文档与查询共用同一规则，保证 WAL/HTTP/快照三处口径一致。数量上限按
/// 原始输入对数执行（防重复对滥用），求和后的不同 term 数只会更少。
///
/// EN: Normalize a term list: validate ranges/weights, sum duplicate terms, and
/// return ascending by term id. Documents and queries share one rule so that
/// WAL, HTTP, and snapshots agree bit-for-bit. The count limit applies to raw
/// input pairs (anti-abuse); the distinct count after summing is never larger.
pub fn normalize_terms(
    input: &[(u32, f32)],
    max_terms: usize,
) -> Result<Vec<(u32, f32)>, SparseError> {
    if input.len() > max_terms {
        return Err(SparseError::TooManyTerms {
            got: input.len(),
            limit: max_terms,
        });
    }
    let mut sorted: Vec<(u32, f32)> = input.to_vec();
    sorted.sort_unstable_by_key(|(term, _)| *term);
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(sorted.len());
    for (term, weight) in sorted {
        if term == 0 || term > MAX_SPARSE_TERM_ID {
            return Err(SparseError::TermOutOfRange { term });
        }
        if !weight.is_finite() {
            return Err(SparseError::NonFiniteWeight { term });
        }
        if weight <= 0.0 {
            return Err(SparseError::NonPositiveWeight { term });
        }
        match out.last_mut() {
            Some((last_term, last_weight)) if *last_term == term => {
                *last_weight += weight;
                if !last_weight.is_finite() {
                    return Err(SparseError::NonFiniteWeight { term });
                }
            }
            _ => out.push((term, weight)),
        }
    }
    // 求和后仍可能因下溢变成 0，统一按非正拒绝。
    // EN: Summation may underflow to zero; reject such weights as non-positive.
    for (term, weight) in &out {
        if *weight <= 0.0 || !weight.is_finite() {
            return Err(SparseError::NonPositiveWeight { term: *term });
        }
    }
    Ok(out)
}

/// 倒排表中的一个 posting；`live=false` 表示该行已被文档重写/删除标记为死行。
/// EN: One posting in the inverted list; `live=false` marks a row dead because
/// the document was rewritten or removed.
#[derive(Debug, Clone, Copy)]
struct Posting {
    doc: u64,
    tf: f32,
    live: bool,
}

/// 单个 term 的倒排表；`df_live` 为该 term 当前存活文档数（BM25 的 df）。
/// EN: Inverted list for one term; `df_live` is the current live document
/// frequency (BM25's df).
#[derive(Debug, Default)]
struct TermPostings {
    postings: Vec<Posting>,
    df_live: u32,
}

/// 正向表条目：`(term, posting 下标, tf)` 与文档长度（Σ权重）。
/// 长度用 `f64` 累计：单项权重是 f32，但两项 3e38 在 f32 下求和会溢出到
/// `+inf`，污染 avgdl 与快照；f64 的余量足以覆盖 1024 项 f32 权重。
///
/// EN: Forward entry: `(term, posting index, tf)` plus the document length
/// (sum of weights). The length accumulates in `f64`: individual weights are
/// f32, but summing two 3e38 values in f32 overflows to `+inf` and would
/// poison avgdl and snapshots; f64 has enough headroom for 1024 f32 weights.
#[derive(Debug)]
struct DocEntry {
    terms: Vec<(u32, usize, f32)>,
    len: f64,
}

/// 集合级 BM25 倒排索引；整文档替换语义（重复 upsert 同 id 幂等）。
/// EN: Per-collection BM25 inverted index with whole-document replacement
/// semantics (re-upserting the same id is idempotent).
#[derive(Debug)]
pub struct Bm25Index {
    terms: HashMap<u32, TermPostings>,
    docs: HashMap<u64, DocEntry>,
    n_docs: u64,
    total_len: f64,
    k1: f32,
    b: f32,
    live_postings: u64,
    dead_postings: u64,
}

/// 状态/指标快照；`posting_count` 只计存活行，`dead_postings` 暴露放大程度。
/// EN: Stats snapshot; `posting_count` counts live rows only, while
/// `dead_postings` exposes amplification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseStats {
    pub doc_count: u64,
    pub term_count: u64,
    pub posting_count: u64,
    pub dead_postings: u64,
    pub avgdl: f64,
}

/// 持久化视图：只含存活 posting，term 升序、term 内 doc 升序。
/// EN: Persistence view: live postings only, terms ascending, docs ascending
/// within each term.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseSnapshotView {
    pub doc_count: u64,
    pub total_len: f64,
    pub terms: Vec<(u32, Vec<(u64, f32)>)>,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Index {
    /// 用默认 BM25 参数（k1=1.2，b=0.75）构造空索引。
    /// EN: Build an empty index with default BM25 parameters (k1=1.2, b=0.75).
    pub fn new() -> Self {
        Self::with_params(BM25_K1, BM25_B)
    }

    /// crate 内部测试/调参入口：显式指定 k1 与 b。参数非法（k1 非有限或为
    /// 负、b 不在 `[0,1]`）时直接 fail-fast，避免静默产生无意义的打分。
    /// 不对外公开：生产配置不允许调参，公开的 panic API 迟早被误用。
    /// 注意 `from_snapshot` 恒用默认参数，自定义参数不参与快照往返。
    ///
    /// EN: Crate-internal test/tuning entry point with explicit k1 and b.
    /// Invalid parameters (non-finite/negative k1, b outside `[0,1]`) fail fast
    /// instead of silently producing meaningless scores. Not public: production
    /// configuration does not allow tuning, and a public panicking API would
    /// eventually be misused. Note that `from_snapshot` always rebuilds with the
    /// default parameters, so custom parameters do not round-trip through
    /// snapshots.
    pub(crate) fn with_params(k1: f32, b: f32) -> Self {
        assert!(
            k1.is_finite() && k1 >= 0.0,
            "BM25 k1 must be finite and >= 0, got {k1}"
        );
        assert!(
            b.is_finite() && (0.0..=1.0).contains(&b),
            "BM25 b must be finite and in [0, 1], got {b}"
        );
        Self {
            terms: HashMap::new(),
            docs: HashMap::new(),
            n_docs: 0,
            total_len: 0.0,
            k1,
            b,
            live_postings: 0,
            dead_postings: 0,
        }
    }

    /// 整文档替换：先摘除旧 postings（标记死行并回退 df/N/total_len），再追加
    /// 规范化后的新词项。空列表等价于清空该文档的稀疏数据，仍是幂等操作；
    /// 该语义使 WAL covered/uncovered 两条重放分支都能安全地重复应用。
    ///
    /// EN: Whole-document replacement: retract old postings (mark dead and roll
    /// back df/N/total_len), then append the normalized new terms. An empty
    /// list clears the document's sparse data and is still idempotent, which is
    /// what makes both covered and uncovered WAL replay branches safe to apply
    /// repeatedly.
    pub fn upsert_doc(&mut self, id: u64, terms: &[(u32, f32)]) -> Result<(), SparseError> {
        let normalized = normalize_terms(terms, MAX_SPARSE_TERMS_PER_DOC)?;
        self.remove_doc_inner(id);
        self.insert_normalized(id, &normalized);
        Ok(())
    }

    /// 删除文档；不存在时幂等返回 `false`。
    /// EN: Remove a document; idempotent and returns `false` when absent.
    pub fn remove_doc(&mut self, id: u64) -> bool {
        self.remove_doc_inner(id)
    }

    fn remove_doc_inner(&mut self, id: u64) -> bool {
        let Some(entry) = self.docs.remove(&id) else {
            return false;
        };
        for (term, posting_idx, _) in &entry.terms {
            if let Some(tp) = self.terms.get_mut(term) {
                if let Some(posting) = tp.postings.get_mut(*posting_idx) {
                    if posting.live {
                        posting.live = false;
                        tp.df_live = tp.df_live.saturating_sub(1);
                        self.live_postings = self.live_postings.saturating_sub(1);
                        self.dead_postings += 1;
                    }
                }
            }
        }
        self.n_docs = self.n_docs.saturating_sub(1);
        self.total_len -= entry.len;
        // 浮点累计理论上可能与逐步求和的顺序不一致，为负或非有限时按正向表
        // 全量重算；文档数很少触发，代价可忽略。极端 f32 权重还会让「减掉大
        // 文档」把总和抵消成 0（小文档的长度被大数吞掉），此时 avgdl=0 会让
        // search 静默返回空——文档仍在时同样重算。
        // EN: In theory the running total may diverge from a fresh summation
        // order; on a non-finite/negative value, recompute from the forward
        // table. This is rare and the cost is negligible. Extreme f32 weights can
        // additionally cancel the total to 0 when a huge document is removed
        // (small lengths are swallowed by the big value); avgdl=0 would then
        // make search silently return empty, so recompute while documents remain.
        if !self.total_len.is_finite()
            || self.total_len < 0.0
            || (self.total_len == 0.0 && !self.docs.is_empty())
        {
            self.total_len = self.docs.values().map(|doc| doc.len).sum();
        }
        true
    }

    fn insert_normalized(&mut self, id: u64, normalized: &[(u32, f32)]) {
        if normalized.is_empty() {
            return;
        }
        let len: f64 = normalized.iter().map(|(_, weight)| *weight as f64).sum();
        let mut entries = Vec::with_capacity(normalized.len());
        for (term, tf) in normalized {
            let tp = self.terms.entry(*term).or_default();
            let posting_idx = tp.postings.len();
            tp.postings.push(Posting {
                doc: id,
                tf: *tf,
                live: true,
            });
            tp.df_live += 1;
            entries.push((*term, posting_idx, *tf));
        }
        self.docs.insert(
            id,
            DocEntry {
                terms: entries,
                len,
            },
        );
        self.n_docs += 1;
        self.total_len += len;
        self.live_postings += normalized.len() as u64;
    }

    /// BM25 检索：查询词项只决定候选集，权重不参与打分；文档侧权重即 tf。
    /// 返回 `(id, score)` 按分数降序、同分 id 升序，最多 `top_n` 条。
    ///
    /// 调用方必须先通过 HTTP/WAL/pgwire 边界校验；内核为防御性地把非法查询
    /// 归一为「无结果」（正常路径不可达）。
    ///
    /// EN: BM25 search: query terms select candidates only (query weights do not
    /// score); document-side weights are the tf. Returns `(id, score)` sorted by
    /// score desc then id asc, capped at `top_n`.
    ///
    /// Callers must validate at HTTP/WAL/pgwire boundaries first; defensively,
    /// an invalid query normalizes to an empty result (unreachable in normal
    /// paths).
    pub fn search(&self, query: &[(u32, f32)], top_n: usize) -> Vec<(u64, f32)> {
        self.search_inner(query, top_n, None)
    }

    /// 带候选白名单的 BM25 检索：白名单在**截断之前**应用，因此过滤后的
    /// top_n 一定填满（只要仍有命中），不会因先截断再过滤而丢失整个模态。
    /// `df`/`idf` 仍按集合全局统计计算，与未过滤检索同口径——过滤只决定
    /// 候选资格，不改变文档分数。
    ///
    /// EN: BM25 search restricted to an allow-list of candidate ids. The mask is
    /// applied **before truncation**, so the filtered top_n is always filled
    /// while matches remain; truncate-then-filter would silently drop the whole
    /// modality. `df`/`idf` stay collection-global, identical to unfiltered
    /// search — filtering decides candidate eligibility only, never scores.
    pub fn search_filtered(
        &self,
        query: &[(u32, f32)],
        top_n: usize,
        allowed: &HashSet<u64>,
    ) -> Vec<(u64, f32)> {
        self.search_inner(query, top_n, Some(allowed))
    }

    fn search_inner(
        &self,
        query: &[(u32, f32)],
        top_n: usize,
        allowed: Option<&HashSet<u64>>,
    ) -> Vec<(u64, f32)> {
        if top_n == 0 {
            return Vec::new();
        }
        let normalized = match normalize_terms(query, MAX_SPARSE_QUERY_TERMS) {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };
        if normalized.is_empty() || self.n_docs == 0 {
            return Vec::new();
        }
        // N=0 或 avgdl=0 时 BM25 无定义，直接返回空。
        // EN: BM25 is undefined for N=0 or avgdl=0; return empty in that case.
        let n = self.n_docs as f64;
        let avgdl = self.total_len / n;
        if !avgdl.is_finite() || avgdl <= 0.0 {
            return Vec::new();
        }
        let k1 = self.k1 as f64;
        let b = self.b as f64;
        let mut scores: HashMap<u64, f64> = HashMap::new();
        for (term, _query_weight) in &normalized {
            let Some(tp) = self.terms.get(term) else {
                // 未登录 term 贡献 0，不产生候选。
                // EN: An unknown term contributes 0 and yields no candidates.
                continue;
            };
            let df = tp.df_live as f64;
            if df <= 0.0 {
                continue;
            }
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for posting in &tp.postings {
                if !posting.live {
                    continue;
                }
                if let Some(allowed) = allowed {
                    if !allowed.contains(&posting.doc) {
                        continue;
                    }
                }
                let Some(doc) = self.docs.get(&posting.doc) else {
                    continue;
                };
                let dl = doc.len;
                let tf = posting.tf as f64;
                let denom = tf + k1 * (1.0 - b + b * dl / avgdl);
                let contribution = idf * tf * (k1 + 1.0) / denom;
                *scores.entry(posting.doc).or_insert(0.0) += contribution;
            }
        }
        let mut out: Vec<(u64, f32)> = scores
            .into_iter()
            .map(|(id, score)| (id, score as f32))
            .collect();
        sort_by_score_desc_id_asc(&mut out);
        out.truncate(top_n);
        out
    }

    /// 当死 postings 超过存活 postings 的一半时重建索引；返回是否执行了
    /// 压缩。重建从正向表出发，结果与压缩前完全一致（仅回收死行）。
    ///
    /// EN: Rebuild when dead postings exceed half of the live ones; returns
    /// whether compaction ran. Rebasing from the forward table preserves
    /// results exactly while reclaiming dead rows.
    pub fn compact_if_needed(&mut self) -> bool {
        if self.dead_postings == 0
            || (self.dead_postings as f64) <= COMPACT_DEAD_RATIO * (self.live_postings as f64)
        {
            return false;
        }
        let docs: Vec<(u64, Vec<(u32, f32)>)> = self
            .docs
            .iter()
            .map(|(id, entry)| {
                let terms = entry
                    .terms
                    .iter()
                    .map(|(term, _, tf)| (*term, *tf))
                    .collect();
                (*id, terms)
            })
            .collect();
        self.terms.clear();
        self.docs.clear();
        self.n_docs = 0;
        self.total_len = 0.0;
        self.live_postings = 0;
        self.dead_postings = 0;
        for (id, terms) in docs {
            self.insert_normalized(id, &terms);
        }
        true
    }

    /// 状态与指标视图。
    /// EN: Status and metrics view.
    pub fn stats(&self) -> SparseStats {
        let avgdl = if self.n_docs == 0 {
            0.0
        } else {
            self.total_len / self.n_docs as f64
        };
        SparseStats {
            doc_count: self.n_docs,
            term_count: self.terms.values().filter(|tp| tp.df_live > 0).count() as u64,
            posting_count: self.live_postings,
            dead_postings: self.dead_postings,
            avgdl,
        }
    }

    /// 导出文档的规范化稀疏词项（供 `/export` 与迁移使用）；不存在返回 `None`。
    /// EN: Export a document's normalized terms (for `/export` and migration);
    /// returns `None` when the document is absent.
    pub fn doc_terms(&self, id: u64) -> Option<Vec<(u32, f32)>> {
        self.docs.get(&id).map(|entry| {
            entry
                .terms
                .iter()
                .map(|(term, _, tf)| (*term, *tf))
                .collect()
        })
    }

    /// 持久化视图；编码器可对其排序后的结构做位置式写入。
    /// EN: Persistence view; encoders may write its sorted structure
    /// positionally.
    pub fn snapshot_view(&self) -> SparseSnapshotView {
        let mut term_ids: Vec<u32> = self
            .terms
            .iter()
            .filter(|(_, tp)| tp.df_live > 0)
            .map(|(term, _)| *term)
            .collect();
        term_ids.sort_unstable();
        let terms = term_ids
            .into_iter()
            .filter_map(|term| {
                let tp = self.terms.get(&term)?;
                let mut postings: Vec<(u64, f32)> = tp
                    .postings
                    .iter()
                    .filter(|posting| posting.live)
                    .map(|posting| (posting.doc, posting.tf))
                    .collect();
                postings.sort_unstable_by_key(|(doc, _)| *doc);
                Some((term, postings))
            })
            .collect();
        SparseSnapshotView {
            doc_count: self.n_docs,
            total_len: self.total_len,
            terms,
        }
    }

    /// 从持久化视图重建索引并校验结构不变量（排序、tf、doc_count）。
    /// `total_len` 按持久值信任，避免求和顺序误差导致误拒；但文档非空时
    /// 它必须为正（所有权重都 > 0），否则 avgdl=0 会让检索静默返回空。
    /// 重建恒用默认 BM25 参数，自定义参数不参与快照往返。
    ///
    /// EN: Rebuild from a persistence view and validate structural invariants
    /// (ordering, tf, doc_count). `total_len` is trusted as persisted to avoid
    /// false rejections from summation-order differences, but it must be
    /// positive when documents exist (all weights are > 0); otherwise avgdl=0
    /// silently yields empty search results. Rebuilding always uses the default
    /// BM25 parameters, so custom parameters never round-trip.
    pub fn from_snapshot(
        doc_count: u64,
        total_len: f64,
        terms: Vec<(u32, Vec<(u64, f32)>)>,
    ) -> Result<Self, SparseError> {
        if !total_len.is_finite() || total_len < 0.0 || (doc_count > 0 && total_len == 0.0) {
            return Err(SparseError::InvalidTotalLen);
        }
        let mut index = Self::new();
        let mut docs: HashMap<u64, DocEntry> = HashMap::new();
        let mut last_term: Option<u32> = None;
        for (term, postings) in terms {
            if term == 0 || term > MAX_SPARSE_TERM_ID {
                return Err(SparseError::TermOutOfRange { term });
            }
            if let Some(prev) = last_term {
                if term <= prev {
                    return Err(SparseError::TermsNotSorted { term });
                }
            }
            last_term = Some(term);
            let mut last_doc: Option<u64> = None;
            let mut term_postings = TermPostings::default();
            for (doc, tf) in postings {
                if let Some(prev) = last_doc {
                    if doc <= prev {
                        return Err(SparseError::PostingsNotSorted { term });
                    }
                }
                last_doc = Some(doc);
                if !tf.is_finite() {
                    return Err(SparseError::NonFiniteTf { term, doc });
                }
                if tf <= 0.0 {
                    return Err(SparseError::NonPositiveTf { term, doc });
                }
                let posting_idx = term_postings.postings.len();
                term_postings.postings.push(Posting {
                    doc,
                    tf,
                    live: true,
                });
                term_postings.df_live += 1;
                match docs.entry(doc) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        let entry = entry.get_mut();
                        entry.terms.push((term, posting_idx, tf));
                        entry.len += tf as f64;
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(DocEntry {
                            terms: vec![(term, posting_idx, tf)],
                            len: tf as f64,
                        });
                    }
                }
            }
            index.live_postings += term_postings.postings.len() as u64;
            index.terms.insert(term, term_postings);
        }
        let actual = docs.len() as u64;
        if actual != doc_count {
            return Err(SparseError::DocCountMismatch {
                declared: doc_count,
                actual,
            });
        }
        index.docs = docs;
        index.n_docs = actual;
        index.total_len = total_len;
        Ok(index)
    }

    /// 文档数（存活）。
    /// EN: Live document count.
    pub fn doc_count(&self) -> u64 {
        self.n_docs
    }

    /// 存活 posting 数。
    /// EN: Live posting count.
    pub fn live_postings(&self) -> u64 {
        self.live_postings
    }

    /// 死 posting 数。
    /// EN: Dead posting count.
    pub fn dead_postings(&self) -> u64 {
        self.dead_postings
    }
}

/// 按分数降序、id 升序排序（融合与 BM25 共用的稳定破平规则）。
/// NaN 不满足全序，先统一钳为 `-inf`（当作最低分），再用 `total_cmp` 排序，
/// 保证任何输入下顺序都是确定且无 panic 的。
///
/// EN: Sort by score descending then id ascending (the shared tie-break rule
/// for BM25 and fusion). NaN has no total order, so it is clamped to `-inf`
/// (lowest) and sorted with `total_cmp`, keeping the order deterministic and
/// panic-free for any input.
fn sort_by_score_desc_id_asc(items: &mut [(u64, f32)]) {
    items.sort_unstable_by(|a, b| {
        let a_score = if a.1.is_nan() { f32::NEG_INFINITY } else { a.1 };
        let b_score = if b.1.is_nan() { f32::NEG_INFINITY } else { b.1 };
        b_score.total_cmp(&a_score).then_with(|| a.0.cmp(&b.0))
    });
}

/// RRF 融合：`score = Σ 1/(rrf_k + rank)`，rank 从 1 开始；只在一路出现的
/// 文档只累加该路一项。两路输入必须已按各自分数降序排列。
///
/// EN: Reciprocal Rank Fusion: `score = sum 1/(rrf_k + rank)` with rank
/// starting at 1; documents present in only one list accumulate just that
/// list's term. Both inputs must already be sorted by their own scores
/// descending.
pub fn rrf_fuse(
    dense: &[(u64, f32)],
    sparse: &[(u64, f32)],
    k: usize,
    rrf_k: u32,
) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f64> = HashMap::new();
    let base = rrf_k as f64;
    for (idx, (id, _)) in dense.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (base + (idx + 1) as f64);
    }
    for (idx, (id, _)) in sparse.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (base + (idx + 1) as f64);
    }
    let mut out: Vec<(u64, f32)> = scores
        .into_iter()
        .map(|(id, score)| (id, score as f32))
        .collect();
    sort_by_score_desc_id_asc(&mut out);
    out.truncate(k);
    out
}

/// Min-Max 归一化到 [0,1]：单个元素（或所有值相等）归一为 1.0，空表返回空。
/// 非有限分数（NaN/Inf）先当作 0 参与 min/max，避免污染输出；输入里没有任何
/// 有限分数时全部归一为 0.0（不能落入「全等 → 1.0」，那会把坏分数抬成最高分）。
///
/// EN: Min-Max normalize into [0,1]: a single element (or an all-equal list)
/// normalizes to 1.0, and an empty list stays empty. Non-finite scores
/// (NaN/Inf) count as 0 for min/max so they cannot poison the output; when the
/// input contains no finite score at all, everything normalizes to 0.0 (the
/// all-equal -> 1.0 rule must not promote bad scores to the top).
pub fn normalize_min_max(items: &[(u64, f32)]) -> Vec<(u64, f32)> {
    let sanitize = |score: f32| if score.is_finite() { score } else { 0.0 };
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut any_finite = false;
    for (_, score) in items {
        if score.is_finite() {
            any_finite = true;
        }
        let score = sanitize(*score);
        min = min.min(score);
        max = max.max(score);
    }
    if items.is_empty() {
        return Vec::new();
    }
    if !any_finite {
        return items.iter().map(|(id, _)| (*id, 0.0)).collect();
    }
    let span = max - min;
    if !span.is_finite() || span <= 0.0 {
        return items.iter().map(|(id, _)| (*id, 1.0)).collect();
    }
    items
        .iter()
        .map(|(id, score)| (*id, (sanitize(*score) - min) / span))
        .collect()
}

/// 加权融合：`alpha*dense_norm + (1-alpha)*sparse_norm`；只在单路出现的文档
/// 另一路记 0 分。输入应为已归一化的候选列表；非有限分数按 0 处理。
///
/// EN: Weighted fusion: `alpha*dense_norm + (1-alpha)*sparse_norm`; a
/// document present in only one list scores 0 for the other modality. Inputs
/// are expected to be normalized candidate lists; non-finite scores count as 0.
pub fn weighted_fuse(
    dense_normed: &[(u64, f32)],
    sparse_normed: &[(u64, f32)],
    alpha: f32,
    k: usize,
) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f64> = HashMap::new();
    for (id, score) in dense_normed {
        let score = if score.is_finite() {
            *score as f64
        } else {
            0.0
        };
        *scores.entry(*id).or_insert(0.0) += alpha as f64 * score;
    }
    for (id, score) in sparse_normed {
        let score = if score.is_finite() {
            *score as f64
        } else {
            0.0
        };
        *scores.entry(*id).or_insert(0.0) += (1.0 - alpha as f64) * score;
    }
    let mut out: Vec<(u64, f32)> = scores
        .into_iter()
        .map(|(id, score)| (id, score as f32))
        .collect();
    sort_by_score_desc_id_asc(&mut out);
    out.truncate(k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    /// 独立参考实现：直接按定义逐文档计算 BM25，用于与内核结果对拍。
    /// EN: Independent reference implementation that computes BM25 per the
    /// definition document-by-document, used to cross-check the kernel.
    fn naive_bm25(docs: &HashMap<u64, Vec<(u32, f32)>>, query: &[(u32, f32)]) -> Vec<(u64, f32)> {
        let n = docs.len() as f64;
        if n == 0.0 {
            return Vec::new();
        }
        let total_len: f64 = docs
            .values()
            .map(|terms| terms.iter().map(|(_, w)| *w as f64).sum::<f64>())
            .sum();
        let avgdl = total_len / n;
        if avgdl <= 0.0 {
            return Vec::new();
        }
        let mut query_terms: Vec<u32> = query.iter().map(|(t, _)| *t).collect();
        query_terms.sort_unstable();
        query_terms.dedup();
        let mut out = Vec::new();
        for (id, terms) in docs {
            let dl: f64 = terms.iter().map(|(_, w)| *w as f64).sum();
            let mut score = 0.0f64;
            for term in &query_terms {
                let tf = terms
                    .iter()
                    .find(|(t, _)| *t == *term)
                    .map(|(_, w)| *w as f64);
                let Some(tf) = tf else { continue };
                let df = docs
                    .values()
                    .filter(|other| other.iter().any(|(t, _)| *t == *term))
                    .count() as f64;
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let k1 = BM25_K1 as f64;
                let b = BM25_B as f64;
                score += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl));
            }
            if score > 0.0 {
                out.push((*id, score as f32));
            }
        }
        sort_by_score_desc_id_asc(&mut out);
        out
    }

    fn normalized_model_terms(terms: &[(u32, f32)]) -> Vec<(u32, f32)> {
        normalize_terms(terms, MAX_SPARSE_TERMS_PER_DOC).unwrap()
    }

    #[test]
    fn sparse_mode_string_roundtrip_and_case_insensitive_parse() {
        for mode in [SparseMode::None, SparseMode::Bm25] {
            assert_eq!(SparseMode::parse_str(mode.as_str()), Some(mode));
        }
        assert_eq!(SparseMode::parse_str("BM25"), Some(SparseMode::Bm25));
        assert_eq!(SparseMode::parse_str("NoNe"), Some(SparseMode::None));
        assert_eq!(SparseMode::parse_str("pQ"), None);
    }

    #[test]
    fn sparse_mode_code_roundtrip_and_unknown_code() {
        for mode in [SparseMode::None, SparseMode::Bm25] {
            assert_eq!(SparseMode::from_code(mode.code()), Some(mode));
        }
        assert_eq!(SparseMode::from_code(2), None);
    }

    #[test]
    fn normalize_sums_duplicates_and_rejects_invalid_weights() {
        let normalized = normalize_terms(&[(2, 0.5), (1, 1.0), (2, 0.25)], 1024).unwrap();
        assert_eq!(normalized, vec![(1, 1.0), (2, 0.75)]);

        assert_eq!(
            normalize_terms(&[(0, 1.0)], 1024).unwrap_err(),
            SparseError::TermOutOfRange { term: 0 }
        );
        assert_eq!(
            normalize_terms(&[(MAX_SPARSE_TERM_ID + 1, 1.0)], 1024).unwrap_err(),
            SparseError::TermOutOfRange {
                term: MAX_SPARSE_TERM_ID + 1
            }
        );
        assert_eq!(
            normalize_terms(&[(1, f32::NAN)], 1024).unwrap_err(),
            SparseError::NonFiniteWeight { term: 1 }
        );
        assert_eq!(
            normalize_terms(&[(1, 0.0)], 1024).unwrap_err(),
            SparseError::NonPositiveWeight { term: 1 }
        );
        assert_eq!(
            normalize_terms(&[(1, -1.0)], 1024).unwrap_err(),
            SparseError::NonPositiveWeight { term: 1 }
        );
        assert_eq!(
            normalize_terms(&[(1, 1.0), (2, 1.0), (3, 1.0)], 2).unwrap_err(),
            SparseError::TooManyTerms { got: 3, limit: 2 }
        );
    }

    #[test]
    fn bm25_golden_values_match_definition() {
        let mut index = Bm25Index::new();
        // A 的长度 3（权重和），B 长度 1，C 长度 3；N=3，avgdl=7/3。
        // EN: A has length 3, B length 1, C length 3; N=3, avgdl=7/3.
        index.upsert_doc(1, &[(1, 2.0), (2, 1.0)]).unwrap();
        index.upsert_doc(2, &[(1, 1.0)]).unwrap();
        index.upsert_doc(3, &[(2, 3.0)]).unwrap();

        let single = index.search(&[(1, 1.0)], 10);
        assert_eq!(single.len(), 2);
        let by_id = |id: u64| single.iter().find(|(d, _)| *d == id).unwrap().1;
        // 独立公式：idf=ln(1+1.5/2.5)=ln(1.6)≈0.4700036292。
        // EN: Independent formula: idf=ln(1+1.5/2.5)=ln(1.6)≈0.4700036292.
        let idf = 0.470_003_63f32;
        let dl_a = 3.0f32;
        let avgdl = 7.0f32 / 3.0;
        let k1 = BM25_K1;
        let b = BM25_B;
        let expected_a = idf * 2.0 * (k1 + 1.0) / (2.0 + k1 * (1.0 - b + b * dl_a / avgdl));
        let expected_b = idf * 1.0 * (k1 + 1.0) / (1.0 + k1 * (1.0 - b + b * 1.0 / avgdl));
        assert!(
            approx(by_id(1), expected_a, 1e-5),
            "doc A score {}",
            by_id(1)
        );
        assert!(
            approx(by_id(2), expected_b, 1e-5),
            "doc B score {}",
            by_id(2)
        );
        assert!(
            expected_b > expected_a,
            "shorter doc must rank first for same tf"
        );

        let both = index.search(&[(1, 1.0), (2, 1.0)], 10);
        assert_eq!(both.len(), 3);
        let by_id2 = |id: u64| both.iter().find(|(d, _)| *d == id).unwrap().1;
        let expected_a2 =
            expected_a + idf * 1.0 * (k1 + 1.0) / (1.0 + k1 * (1.0 - b + b * dl_a / avgdl));
        assert!(approx(by_id2(1), expected_a2, 1e-5));
        // 融合前：三文档都在，顺序按分数降序。
        // EN: All three documents present, ordered by descending score.
        assert_eq!(both[0].0, 1);
    }

    #[test]
    fn unknown_terms_and_empty_index_return_no_candidates() {
        let mut index = Bm25Index::new();
        index.upsert_doc(7, &[(100, 1.0)]).unwrap();
        assert!(index.search(&[(999, 1.0)], 10).is_empty());
        assert!(Bm25Index::new().search(&[(1, 1.0)], 10).is_empty());
        assert!(index.search(&[], 10).is_empty());
        assert!(index.search(&[(100, 1.0)], 0).is_empty());
    }

    #[test]
    fn dead_postings_are_invisible_and_compaction_preserves_results() {
        let mut index = Bm25Index::new();
        index.upsert_doc(1, &[(1, 2.0)]).unwrap();
        assert_eq!(index.search(&[(1, 1.0)], 10).len(), 1);

        // 重写：旧 term1 posting 变为死行，新数据只含 term2。
        // EN: Rewrite: old term1 posting becomes dead; new data holds term2 only.
        index.upsert_doc(1, &[(2, 3.0)]).unwrap();
        assert!(index.search(&[(1, 1.0)], 10).is_empty());
        assert_eq!(index.search(&[(2, 1.0)], 10).len(), 1);
        let stats = index.stats();
        assert_eq!(stats.doc_count, 1);
        assert_eq!(stats.dead_postings, 1);
        assert_eq!(stats.posting_count, 1);

        // 再重写一次使 dead(2) > live(1)，触发压缩；结果不变。
        // EN: Rewrite again so dead(2) > live(1), triggering compaction with
        // unchanged results.
        index.upsert_doc(1, &[(3, 1.5)]).unwrap();
        assert_eq!(index.stats().dead_postings, 2);
        let before = index.search(&[(3, 1.0)], 10);
        assert!(index.compact_if_needed());
        let after = index.search(&[(3, 1.0)], 10);
        assert_eq!(before, after);
        assert_eq!(index.stats().dead_postings, 0);
        assert_eq!(index.stats().posting_count, 1);
    }

    #[test]
    fn filtered_search_applies_mask_before_truncation() {
        let mut index = Bm25Index::new();
        index.upsert_doc(10, &[(1, 5.0)]).unwrap();
        index.upsert_doc(20, &[(1, 3.0)]).unwrap();
        index.upsert_doc(30, &[(1, 1.0)]).unwrap();
        let all = index.search(&[(1, 1.0)], 10);
        assert_eq!(
            all.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
        // 若先截断 top-1 再过滤（只允许 30），结果会空；内核必须在过滤后
        // 再截断，才能为被过滤的模态保住候选。分数与未过滤检索同口径。
        // EN: Truncating top-1 before filtering (only 30 allowed) would yield
        // nothing; the kernel masks before truncation so the filtered modality
        // keeps its candidates. Scores match unfiltered search.
        let allowed: HashSet<u64> = [30].into_iter().collect();
        let filtered = index.search_filtered(&[(1, 1.0)], 1, &allowed);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0, 30);
        assert_eq!(filtered[0].1, all[2].1);
        // 空白名单恒为空结果。
        // EN: An empty allow-list always yields no results.
        assert!(index
            .search_filtered(&[(1, 1.0)], 10, &HashSet::new())
            .is_empty());
    }

    #[test]
    fn upsert_same_document_twice_is_idempotent() {
        let mut index = Bm25Index::new();
        index.upsert_doc(5, &[(1, 2.0), (2, 1.0)]).unwrap();
        let first = index.search(&[(1, 1.0), (2, 1.0)], 10);
        index.upsert_doc(5, &[(1, 2.0), (2, 1.0)]).unwrap();
        let second = index.search(&[(1, 1.0), (2, 1.0)], 10);
        assert_eq!(first, second);
        assert_eq!(index.doc_count(), 1);
    }

    #[test]
    fn empty_upsert_clears_document_sparse_data() {
        let mut index = Bm25Index::new();
        index.upsert_doc(1, &[(1, 1.0)]).unwrap();
        index.upsert_doc(2, &[(1, 2.0)]).unwrap();
        // `[]` 语义：清空该文档的稀疏数据但保留 upsert 词项语义。
        // EN: `[]` clears the document's sparse data while keeping upsert
        // semantics.
        index.upsert_doc(1, &[]).unwrap();
        let hits = index.search(&[(1, 1.0)], 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
        assert_eq!(index.doc_count(), 1);
        // 清空后正向表不再登记该文档；导出侧据此省略 sparse 字段。
        // EN: After clearing, the forward table no longer records the document;
        // exporters therefore omit the sparse field.
        assert_eq!(index.doc_terms(1), None);
        // 删除一个无稀疏数据的文档是幂等的。
        // EN: Removing a document with no sparse data is idempotent.
        assert!(!index.remove_doc(1));
        assert!(!index.remove_doc(999));
    }

    #[test]
    fn snapshot_view_roundtrip_preserves_scores_and_structure() {
        let mut index = Bm25Index::new();
        index.upsert_doc(3, &[(1, 2.0), (2, 0.5)]).unwrap();
        index.upsert_doc(1, &[(1, 1.0)]).unwrap();
        index.upsert_doc(2, &[(2, 3.0)]).unwrap();
        // 制造死行后导出，只能包含存活 posting。
        // EN: Create dead rows before exporting; only live postings may appear.
        index.upsert_doc(3, &[(1, 4.0), (2, 1.0)]).unwrap();

        let view = index.snapshot_view();
        assert_eq!(view.doc_count, 3);
        assert_eq!(
            view.terms.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec![1, 2]
        );
        for (_, postings) in &view.terms {
            assert!(postings.windows(2).all(|w| w[0].0 < w[1].0));
        }
        let rebuilt =
            Bm25Index::from_snapshot(view.doc_count, view.total_len, view.terms.clone()).unwrap();
        let query = [(1, 1.0), (2, 1.0)];
        assert_eq!(rebuilt.search(&query, 10), index.search(&query, 10));
        assert_eq!(rebuilt.stats().doc_count, index.stats().doc_count);
        assert_eq!(rebuilt.stats().posting_count, index.stats().posting_count);
        for id in [1, 2, 3] {
            assert_eq!(rebuilt.doc_terms(id), index.doc_terms(id));
        }
    }

    #[test]
    fn from_snapshot_rejects_corrupt_structures() {
        // term 必须严格升序：重复 term 拒绝。
        // EN: Terms must be strictly ascending; duplicates are rejected.
        assert_eq!(
            Bm25Index::from_snapshot(1, 1.0, vec![(1, vec![(1, 1.0)]), (1, vec![(2, 1.0)])])
                .unwrap_err(),
            SparseError::TermsNotSorted { term: 1 }
        );
        assert_eq!(
            Bm25Index::from_snapshot(1, 1.0, vec![(2, vec![(1, 1.0)]), (1, vec![(1, 1.0)])])
                .unwrap_err(),
            SparseError::TermsNotSorted { term: 1 }
        );
        // posting doc 必须严格升序。
        // EN: Posting docs must be strictly ascending.
        assert_eq!(
            Bm25Index::from_snapshot(2, 2.0, vec![(1, vec![(1, 1.0), (1, 1.0)])]).unwrap_err(),
            SparseError::PostingsNotSorted { term: 1 }
        );
        // tf 必须有限且 > 0。
        // EN: tf must be finite and > 0.
        assert_eq!(
            Bm25Index::from_snapshot(1, 1.0, vec![(1, vec![(1, f32::NAN)])]).unwrap_err(),
            SparseError::NonFiniteTf { term: 1, doc: 1 }
        );
        assert_eq!(
            Bm25Index::from_snapshot(1, 1.0, vec![(1, vec![(1, 0.0)])]).unwrap_err(),
            SparseError::NonPositiveTf { term: 1, doc: 1 }
        );
        // doc_count 与去重后的文档数必须一致。
        // EN: doc_count must equal the distinct document count.
        assert_eq!(
            Bm25Index::from_snapshot(2, 2.0, vec![(1, vec![(1, 1.0)])]).unwrap_err(),
            SparseError::DocCountMismatch {
                declared: 2,
                actual: 1
            }
        );
        assert_eq!(
            Bm25Index::from_snapshot(0, f64::NAN, Vec::new()).unwrap_err(),
            SparseError::InvalidTotalLen
        );
        assert_eq!(
            Bm25Index::from_snapshot(0, -1.0, Vec::new()).unwrap_err(),
            SparseError::InvalidTotalLen
        );
        // 文档非空时 total_len 必须为正，否则 avgdl=0 会静默返回空。
        // EN: With documents present total_len must be positive; otherwise
        // avgdl=0 would silently return empty results.
        assert_eq!(
            Bm25Index::from_snapshot(1, 0.0, vec![(1, vec![(1, 1.0)])]).unwrap_err(),
            SparseError::InvalidTotalLen
        );
    }

    #[test]
    fn doc_length_accumulates_in_f64_without_overflow() {
        // 两个 f32 3e38 在 f32 下求和溢出为 +inf；f64 累计必须保持有限。
        // EN: Two f32 3e38 values overflow to +inf when summed in f32; the f64
        // accumulator must stay finite.
        let mut index = Bm25Index::new();
        index.upsert_doc(1, &[(1, 3.0e38), (2, 3.0e38)]).unwrap();
        let view = index.snapshot_view();
        assert!(view.total_len.is_finite());
        assert!(view.total_len > 3.0e38f64);
        let rebuilt = Bm25Index::from_snapshot(view.doc_count, view.total_len, view.terms).unwrap();
        let hits = rebuilt.search(&[(1, 1.0)], 10);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].1.is_finite());
    }

    #[test]
    fn score_sort_treats_nan_as_lowest_and_is_deterministic() {
        let mut items = [(3u64, f32::NAN), (1, 0.5), (2, 0.5), (4, f32::NEG_INFINITY)];
        sort_by_score_desc_id_asc(&mut items);
        // 0.5 并列按 id 升序；NaN 与 -inf 都视为最低分，id 升序破平。
        // EN: 0.5 ties sort by id ascending; NaN and -inf both count as lowest,
        // tie-broken by id ascending.
        assert_eq!(
            items.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(items[0], (1, 0.5));
        assert_eq!(items[1], (2, 0.5));
        assert!(items[2].1.is_nan(), "NaN stays in place");
        assert_eq!(items[3], (4, f32::NEG_INFINITY));
    }

    #[test]
    fn normalization_treats_non_finite_scores_as_zero() {
        let normed = normalize_min_max(&[(1, f32::NAN), (2, 1.0), (3, 3.0)]);
        assert_eq!(normed[0], (1, 0.0));
        assert!(approx(normed[1].1, 1.0 / 3.0, 1e-6));
        assert!(approx(normed[2].1, 1.0, 1e-6));
        // 全部非有限：按 0 处理，不得落入「全等 → 1.0」把坏分数抬成最高分。
        // EN: All non-finite: treated as 0 — the all-equal -> 1.0 rule must not
        // promote bad scores to the maximum.
        assert_eq!(
            normalize_min_max(&[(1, f32::NAN), (2, f32::INFINITY)]),
            vec![(1, 0.0), (2, 0.0)]
        );
    }

    #[test]
    fn remove_doc_recomputes_total_len_when_cancellation_zeroes_it() {
        // 极端 f32 权重：巨大文档把总和撑大，小文档长度被吞掉；删除巨大文档
        // 后累计和抵消为 0，但小文档仍在——必须重算，否则 avgdl=0 静默空结果。
        // EN: Extreme f32 weights: the huge document inflates the total and
        // swallows the small one; removing the huge document cancels the
        // accumulator to 0 while the small document remains — recompute, or
        // avgdl=0 silently yields empty results.
        let mut index = Bm25Index::new();
        index.upsert_doc(1, &[(1, 3.0e38)]).unwrap();
        index.upsert_doc(2, &[(2, 1.0)]).unwrap();
        assert_eq!(index.stats().doc_count, 2);
        assert!(index.remove_doc(1));
        assert_eq!(index.stats().doc_count, 1);
        let hits = index.search(&[(2, 1.0)], 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
        assert_eq!(index.stats().avgdl, 1.0);
    }

    #[test]
    fn weighted_fuse_ignores_non_finite_scores() {
        let fused = weighted_fuse(&[(1, f32::NAN), (2, 1.0)], &[(2, f32::NAN)], 0.5, 2);
        let map: HashMap<u64, f32> = fused.iter().copied().collect();
        assert!(approx(map[&1], 0.0, 1e-6));
        assert!(approx(map[&2], 0.5, 1e-6));
    }

    #[test]
    #[should_panic(expected = "BM25 b")]
    fn with_params_rejects_out_of_range_b() {
        let _ = Bm25Index::with_params(1.2, -0.1);
    }

    #[test]
    fn rrf_fuse_sums_reciprocal_ranks_and_breaks_ties_by_id() {
        let dense = vec![(10, 0.9), (20, 0.8)];
        let sparse = vec![(20, 5.0), (30, 3.0)];
        let fused = rrf_fuse(&dense, &sparse, 3, 60);
        assert_eq!(fused.len(), 3);
        // 20: 1/62 + 1/61；10: 1/61；30: 1/62。
        // EN: 20: 1/62 + 1/61; 10: 1/61; 30: 1/62.
        let s20 = 1.0f32 / 62.0 + 1.0f32 / 61.0;
        let s10 = 1.0f32 / 61.0;
        let s30 = 1.0f32 / 62.0;
        assert_eq!(fused[0].0, 20);
        assert!(approx(fused[0].1, s20, 1e-6));
        assert_eq!(fused[1].0, 10);
        assert!(approx(fused[1].1, s10, 1e-6));
        assert_eq!(fused[2].0, 30);
        assert!(approx(fused[2].1, s30, 1e-6));

        // 单路输入：名次即最终排序；截断到 k。
        // EN: Single-list input keeps rank order and truncates to k.
        let only_dense = rrf_fuse(&[(3, 1.0), (4, 0.5)], &[], 1, 60);
        assert_eq!(only_dense, vec![(3, 1.0f32 / 61.0)]);
        assert!(rrf_fuse(&[], &[], 5, 60).is_empty());
        // 同分破平：两路各自 rank 相同时，id 小者在前。
        // EN: Tie-break: equal ranks from both lists order the smaller id first.
        let tie = rrf_fuse(&[(2, 1.0)], &[(1, 1.0)], 2, 60);
        assert_eq!(
            tie.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn min_max_normalization_handles_single_and_equal_values() {
        let normed = normalize_min_max(&[(1, 2.0), (2, 4.0), (3, 3.0)]);
        assert!(approx(normed[0].1, 0.0, 1e-6));
        assert!(approx(normed[1].1, 1.0, 1e-6));
        assert!(approx(normed[2].1, 0.5, 1e-6));
        // 单元素归一为 1.0；全等列表也归一为 1.0。
        // EN: A single element and an all-equal list normalize to 1.0.
        assert_eq!(normalize_min_max(&[(9, 3.0)]), vec![(9, 1.0)]);
        assert_eq!(
            normalize_min_max(&[(1, 2.0), (2, 2.0)]),
            vec![(1, 1.0), (2, 1.0)]
        );
        assert!(normalize_min_max(&[]).is_empty());
    }

    #[test]
    fn weighted_fuse_applies_alpha_and_missing_modality_zero() {
        let dense = normalize_min_max(&[(1, 10.0), (2, 20.0)]);
        let sparse = normalize_min_max(&[(2, 1.0), (3, 2.0)]);
        // 1: 0.5*0 + 0（dense 最小、sparse 缺席）；2: 0.5*1 + 0.5*0；
        // 3: 0.5*0 + 0.5*1。
        // EN: 1: 0.5*0 + 0; 2: 0.5*1 + 0.5*0; 3: 0.5*0 + 0.5*1.
        let fused = weighted_fuse(&dense, &sparse, 0.5, 3);
        let map: HashMap<u64, f32> = fused.iter().copied().collect();
        assert!(approx(map[&1], 0.0, 1e-6));
        assert!(approx(map[&2], 0.5, 1e-6));
        assert!(approx(map[&3], 0.5, 1e-6));
        // 2 与 3 同分，id 升序在前。
        // EN: 2 and 3 tie; the smaller id sorts first.
        assert_eq!(fused[0].0, 2);
        assert_eq!(fused[1].0, 3);

        // alpha=1 退化为纯 dense（sparse 权重为 0）。
        // EN: alpha=1 degenerates to pure dense (sparse weight 0).
        let pure_dense = weighted_fuse(&dense, &sparse, 1.0, 2);
        assert_eq!(
            pure_dense.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(weighted_fuse(&[], &[], 0.5, 3).is_empty());
    }

    proptest! {
        /// 与朴素模型对拍：随机 upsert/remove/compact 序列后，检索结果与
        /// 参考实现一致（分数按 f32 容差比较）。
        /// EN: Cross-check against a naive model: after a random sequence of
        /// upsert/remove/compact operations, search matches the reference
        /// implementation within an f32 tolerance.
        #[test]
        fn upsert_remove_search_matches_naive_model(
            ops in prop::collection::vec(
                (0u64..8, prop::collection::vec((1u32..6, 0.1f32..5.0), 0..5), any::<bool>()),
                0..30,
            ),
            query_terms in prop::collection::vec((1u32..6, 0.1f32..3.0), 1..4),
        ) {
            let mut index = Bm25Index::new();
            let mut model: HashMap<u64, Vec<(u32, f32)>> = HashMap::new();
            for (idx, (id, terms, remove)) in ops.iter().enumerate() {
                if *remove {
                    index.remove_doc(*id);
                    model.remove(id);
                } else {
                    index.upsert_doc(*id, terms).unwrap();
                    // 空稀疏列表等于清空：参考模型也不应把该文档计入 N。
                    // EN: An empty sparse list clears the document; the
                    // reference model must not count it toward N either.
                    let normalized = normalized_model_terms(terms);
                    if normalized.is_empty() {
                        model.remove(id);
                    } else {
                        model.insert(*id, normalized);
                    }
                }
                if idx % 7 == 6 {
                    index.compact_if_needed();
                }
            }
            let got = index.search(&query_terms, 16);
            let expected = naive_bm25(&model, &query_terms);
            prop_assert_eq!(got.len(), expected.len(), "hit count mismatch");
            for ((got_id, got_score), (exp_id, exp_score)) in got.iter().zip(expected.iter()) {
                prop_assert_eq!(got_id, exp_id);
                prop_assert!(
                    approx(*got_score, *exp_score, 1e-4),
                    "score mismatch for {}: {} vs {}", got_id, got_score, exp_score
                );
            }
        }

        /// 快照视图往返：任意索引导出再重建后，检索与统计一致。
        /// EN: Snapshot roundtrip: any index re-imported from its view yields
        /// identical search and stats.
        #[test]
        fn snapshot_roundtrip_matches_original(
            docs in prop::collection::vec(
                (0u64..8, prop::collection::vec((1u32..6, 0.1f32..5.0), 1..5)),
                1..10,
            ),
            query_terms in prop::collection::vec((1u32..6, 0.1f32..3.0), 1..4),
        ) {
            let mut index = Bm25Index::new();
            for (id, terms) in &docs {
                index.upsert_doc(*id, terms).unwrap();
            }
            let view = index.snapshot_view();
            let rebuilt = Bm25Index::from_snapshot(view.doc_count, view.total_len, view.terms).unwrap();
            prop_assert_eq!(rebuilt.stats().doc_count, index.stats().doc_count);
            prop_assert_eq!(rebuilt.stats().posting_count, index.stats().posting_count);
            prop_assert_eq!(rebuilt.search(&query_terms, 16), index.search(&query_terms, 16));
        }
    }
}
