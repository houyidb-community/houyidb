package com.houyidb.client;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 一条待写入向量：id + 分量 + 可选 label/payload/sparse。
 * EN: A vector to write: id + components + optional label/payload/sparse.
 */
public final class Types {

    private Types() {}

    /**
     * 稀疏词项 {term_id, weight}（term_id 取值 [1, 2^24-1]，weight 有限且为正；
     * 重复 term 由服务端求和）。
     * EN: Sparse term {term_id, weight} (term_id in [1, 2^24-1], weight finite
     * and positive; duplicate terms are summed by the server).
     */
    public record SparseTerm(int term, double weight) {}

    /**
     * 待写入向量。id 落在 [0, 2^63-1]（stable_id 或自有正整数键空间）。
     *
     * sparse 仅在 sparse=bm25 的集合上有效：null 表示不改该文档的稀疏词项，
     * 空列表（非 null）表示清空。
     * EN: Vector to write. id falls in [0, 2^63-1] (stable_id or your own
     * positive-key space). sparse applies only to sparse=bm25 collections:
     * null leaves the document's sparse terms unchanged, while an empty
     * (non-null) list clears them.
     */
    public static final class Vector {
        public final long id;
        public final double[] values;
        public final String label; // null = 不写入。 EN: null = omit.
        public final Map<String, Object> payload; // null = 不写入。 EN: null = omit.
        // 稀疏词项：null = 不改，空列表 = 清空。 EN: sparse: null = unchanged, empty list = clear.
        public final List<SparseTerm> sparse;

        public Vector(long id, double[] values, String label, Map<String, Object> payload) {
            this(id, values, label, payload, null);
        }

        public Vector(long id, double[] values, Map<String, Object> payload) {
            this(id, values, null, payload, null);
        }

        public Vector(long id, double[] values, String label, Map<String, Object> payload,
                List<SparseTerm> sparse) {
            this.id = id;
            this.values = values;
            this.label = label;
            this.payload = payload;
            this.sparse = sparse;
        }
    }

    /**
     * 过滤算子白名单（与服务端一致；多条 {@link Filter} 之间取交集）。
     * wire 值即小写字符串，直接用常量，无需枚举序列化层。
     * EN: Filter operator whitelist (matches the server; multiple {@link Filter}
     * entries are intersected). The wire value is the lowercase string, so
     * constants are used directly and no enum serialization layer is needed.
     */
    public static final class FilterOp {
        public static final String EQ = "eq";
        public static final String GT = "gt";
        public static final String GTE = "gte";
        public static final String LT = "lt";
        public static final String LTE = "lte";
        public static final String IN = "in";

        private FilterOp() {}
    }

    /**
     * 一条结构化过滤条件。value 原样进入 JSON（String/Number/Boolean/数组）。
     * EN: One structured filter condition. value goes into JSON as-is
     * (String/Number/Boolean/array).
     */
    public static final class Filter {
        public final String column;
        public final String op;
        public final Object value;

        public Filter(String column, String op, Object value) {
            this.column = column;
            this.op = op;
            this.value = value;
        }
    }

    /**
     * 单条命中：id + 相似度分数 + 可选 payload（include_payload 时）。
     * EN: One hit: id + similarity score + optional payload (when include_payload
     * is set).
     */
    public static final class SearchHit {
        public final long id;
        public final double score;
        public final Map<String, Object> payload; // null = 未请求或该条无 payload。 EN: null = not requested or no payload.

        SearchHit(long id, double score, Map<String, Object> payload) {
            this.id = id;
            this.score = score;
            this.payload = payload;
        }
    }

    /**
     * 检索响应：hits 按相似度从高到低排列。
     * EN: Search response: hits ordered from highest to lowest similarity.
     */
    public static final class SearchResult {
        public final List<SearchHit> hits;

        SearchResult(List<SearchHit> hits) {
            this.hits = hits;
        }

        public List<Long> ids() {
            List<Long> out = new ArrayList<>(hits.size());
            for (SearchHit h : hits) out.add(h.id);
            return out;
        }

        public List<Double> scores() {
            List<Double> out = new ArrayList<>(hits.size());
            for (SearchHit h : hits) out.add(h.score);
            return out;
        }
    }

    /**
     * 建集合时的 HNSW 参数三元组（必须三个全给，否则服务端 400）。
     * EN: HNSW parameter triple for explicit creation (all three are required
     * or the server returns 400).
     */
    public static final class HnswParams {
        public final int m;
        public final int efConstruction;
        public final int efSearch;

        public HnswParams(int m, int efConstruction, int efSearch) {
            this.m = m;
            this.efConstruction = efConstruction;
            this.efSearch = efSearch;
        }
    }

    /**
     * 显式建集合的选项（省略即服务端默认：余弦 + 默认后端 + 默认 HNSW）。
     * metric 与后端创建后不可变，改参数只能删除重建（旧图按旧度量构建）。
     * EN: Options for explicit creation (omitted means server defaults: cosine,
     * default backend, default HNSW). Metric and backend are immutable
     * afterwards; changing them requires drop-and-recreate (old graphs were
     * built under the old metric).
     */
    public static final class CreateCollectionOptions {
        public String metric; // l2 / cosine / ip / l1。 EN: l2/cosine/ip/l1.
        public String indexType; // hnsw / pq-flat / flat / ...。 EN: backend kind.
        // 存储编码 f32（缺省）/ f16（仅 hnsw）；创建后不可变。
        // EN: storage encoding f32 (default) / f16 (hnsw only); immutable after creation.
        public String vectorDtype;
        // 图侧量化 none（缺省）/ int8 / rabitq（仅 hnsw，压缩图内距离工作集；
        // 主表保留原始向量并在检索时精确重排）。rabitq 为 1 bit/dim，仅支持
        // metric l2|cosine（ip/l1 返回 400）；创建后不可变；非 none 与非 hnsw
        // 组合由服务端拒绝（400）。
        // EN: graph-side quantization none (default) / int8 / rabitq (hnsw only,
        // shrinks the in-graph distance working set; the main store keeps raw
        // vectors and reranks exactly at search time). rabitq packs 1 bit/dim
        // and supports metric l2|cosine only (ip/l1 return 400); immutable after
        // creation; non-none with a non-hnsw backend is rejected by the server
        // (400).
        public String quantization;
        // 稀疏索引模式 none（缺省）/ bm25（仅 hnsw/flat 后端）。开启后集合获得
        // 倒排稀疏索引并支持 fusionSearch；创建后不可变，隐式创建恒为 none。
        // EN: sparse index mode none (default) / bm25 (hnsw/flat backends only).
        // When enabled the collection gains an inverted sparse index and supports
        // fusionSearch; it is immutable after creation, and implicit creation is
        // always none.
        public String sparse;
        public HnswParams hnsw;

        public CreateCollectionOptions metric(String v) {
            this.metric = v;
            return this;
        }

        public CreateCollectionOptions indexType(String v) {
            this.indexType = v;
            return this;
        }

        public CreateCollectionOptions vectorDtype(String v) {
            this.vectorDtype = v;
            return this;
        }

        public CreateCollectionOptions quantization(String v) {
            this.quantization = v;
            return this;
        }

        public CreateCollectionOptions sparse(String v) {
            this.sparse = v;
            return this;
        }

        public CreateCollectionOptions hnsw(HnswParams v) {
            this.hnsw = v;
            return this;
        }
    }

    /**
     * 建集合响应：服务端回显实际生效的参数（hnsw 仅 hnsw 后端非 null）。
     * EN: Creation response echoing the effective parameters (hnsw is non-null
     * only for the hnsw backend).
     */
    public static final class CreateCollectionResult {
        public final boolean created;
        public final String collection;
        public final int dims;
        public final String metric;
        public final String indexType;
        // 实际生效的存储编码（f32 | f16）；旧服务端不返回时为 null。
        // EN: effective storage encoding (f32 | f16); null from older servers.
        public final String vectorDtype;
        // 实际生效的图侧量化（none | int8 | rabitq）；旧服务端不返回时为 null。
        // EN: effective graph-side quantization (none | int8 | rabitq); null from
        // older servers.
        public final String quantization;
        // 实际生效的稀疏模式（none | bm25）；旧服务端不返回时为 null。
        // EN: effective sparse mode (none | bm25); null from older servers.
        public final String sparse;
        public final HnswParams hnsw;

        CreateCollectionResult(boolean created, String collection, int dims,
                String metric, String indexType, String vectorDtype, String quantization,
                String sparse, HnswParams hnsw) {
            this.created = created;
            this.collection = collection;
            this.dims = dims;
            this.metric = metric;
            this.indexType = indexType;
            this.vectorDtype = vectorDtype;
            this.quantization = quantization;
            this.sparse = sparse;
            this.hnsw = hnsw;
        }
    }

    /**
     * 检索/混合检索可选参数（默认构造即全默认）。
     * EN: Optional search/hybrid-search parameters (default construction means
     * all defaults).
     */
    public static final class SearchOptions {
        public Integer efSearch;
        public boolean includePayload;
        public final List<Filter> filters = new ArrayList<>();

        public SearchOptions includePayload(boolean v) {
            this.includePayload = v;
            return this;
        }

        public SearchOptions efSearch(int v) {
            this.efSearch = v;
            return this;
        }

        public SearchOptions filter(String column, String op, Object value) {
            this.filters.add(new Filter(column, op, value));
            return this;
        }
    }

    /**
     * 融合检索可选参数（默认构造即服务端默认：rrf + rrf_k=60）。
     * fusion 取 rrf | weighted；rrfK 仅 rrf 生效，alpha 仅 weighted 生效
     * （alpha ∈ [0,1]）。filters/efSearch/includePayload 与 {@link SearchOptions}
     * 同语义。
     * EN: Optional fusion-search parameters (default construction means server
     * defaults: rrf with rrf_k=60). fusion is rrf | weighted; rrfK applies only
     * to rrf and alpha only to weighted (alpha in [0,1]). filters/efSearch/
     * includePayload carry the same semantics as {@link SearchOptions}.
     */
    public static final class FusionOptions {
        public Integer efSearch;
        public boolean includePayload;
        public final List<Filter> filters = new ArrayList<>();
        public String fusion;
        public Integer rrfK;
        public Double alpha;

        public FusionOptions includePayload(boolean v) {
            this.includePayload = v;
            return this;
        }

        public FusionOptions efSearch(int v) {
            this.efSearch = v;
            return this;
        }

        public FusionOptions filter(String column, String op, Object value) {
            this.filters.add(new Filter(column, op, value));
            return this;
        }

        public FusionOptions fusion(String v) {
            this.fusion = v;
            return this;
        }

        public FusionOptions rrfK(int v) {
            this.rrfK = v;
            return this;
        }

        public FusionOptions alpha(double v) {
            this.alpha = v;
            return this;
        }
    }

    /**
     * 单条融合命中：id + 融合分 + 两路原始分 + 可选 payload。
     *
     * 融合分（score）只表达排序（RRF 或加权），不属于任何度量口径；denseScore
     * 按集合度量换算、仅稠密一路命中时非 null；sparseScore 为 BM25 原始分、
     * 仅稀疏一路命中时非 null。
     * EN: One fusion hit: id + fused score + both raw modality scores + optional
     * payload. The fused score expresses ranking only (RRF or weighted) and
     * belongs to no metric; denseScore follows the collection metric and is
     * non-null only when the dense path hit; sparseScore is the raw BM25 score
     * and is non-null only when the sparse path hit.
     */
    public static final class FusionHit {
        public final long id;
        public final double score;
        public final Double denseScore;
        public final Double sparseScore;
        public final Map<String, Object> payload; // null = 未请求或该条无 payload。 EN: null = not requested or no payload.

        FusionHit(long id, double score, Double denseScore, Double sparseScore,
                Map<String, Object> payload) {
            this.id = id;
            this.score = score;
            this.denseScore = denseScore;
            this.sparseScore = sparseScore;
            this.payload = payload;
        }
    }

    /**
     * 融合检索响应：hits 按融合分从高到低排列，附生效参数回显。
     *
     * metric 仅用于解释 denseScore；fusionMode 为生效模式；rrfK 仅 rrf 模式
     * 回显、alpha 仅 weighted 模式回显（缺席为 null）。
     * EN: Fusion response: hits ordered by fused score, highest first, with the
     * effective parameters echoed. metric explains denseScore only; fusionMode
     * is the effective mode; rrfK is echoed only under rrf and alpha only under
     * weighted (null when absent).
     */
    public static final class FusionResult {
        public final List<FusionHit> hits;
        public final String metric;
        public final String fusionMode;
        public final Integer rrfK;
        public final Double alpha;

        FusionResult(List<FusionHit> hits, String metric, String fusionMode,
                Integer rrfK, Double alpha) {
            this.hits = hits;
            this.metric = metric;
            this.fusionMode = fusionMode;
            this.rrfK = rrfK;
            this.alpha = alpha;
        }

        public List<Long> ids() {
            List<Long> out = new ArrayList<>(hits.size());
            for (FusionHit h : hits) out.add(h.id);
            return out;
        }

        public List<Double> scores() {
            List<Double> out = new ArrayList<>(hits.size());
            for (FusionHit h : hits) out.add(h.score);
            return out;
        }
    }
}
