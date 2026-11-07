package com.houyidb.client;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 一条待写入向量：id + 分量 + 可选 label/payload。
 * EN: A vector to write: id + components + optional label/payload.
 */
public final class Types {

    private Types() {}

    /**
     * 待写入向量。id 落在 [0, 2^63-1]（stable_id 或自有正整数键空间）。
     * EN: Vector to write. id falls in [0, 2^63-1] (stable_id or your own
     * positive-key space).
     */
    public static final class Vector {
        public final long id;
        public final double[] values;
        public final String label; // null = 不写入。 EN: null = omit.
        public final Map<String, Object> payload; // null = 不写入。 EN: null = omit.

        public Vector(long id, double[] values, String label, Map<String, Object> payload) {
            this.id = id;
            this.values = values;
            this.label = label;
            this.payload = payload;
        }

        public Vector(long id, double[] values, Map<String, Object> payload) {
            this(id, values, null, payload);
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
        public HnswParams hnsw;

        public CreateCollectionOptions metric(String v) {
            this.metric = v;
            return this;
        }

        public CreateCollectionOptions indexType(String v) {
            this.indexType = v;
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
        public final HnswParams hnsw;

        CreateCollectionResult(boolean created, String collection, int dims,
                String metric, String indexType, HnswParams hnsw) {
            this.created = created;
            this.collection = collection;
            this.dims = dims;
            this.metric = metric;
            this.indexType = indexType;
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
}
