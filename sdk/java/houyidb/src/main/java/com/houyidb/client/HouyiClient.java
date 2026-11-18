package com.houyidb.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.houyidb.client.Types.CreateCollectionOptions;
import com.houyidb.client.Types.CreateCollectionResult;
import com.houyidb.client.Types.Filter;
import com.houyidb.client.Types.FusionHit;
import com.houyidb.client.Types.FusionOptions;
import com.houyidb.client.Types.FusionResult;
import com.houyidb.client.Types.HnswParams;
import com.houyidb.client.Types.SearchHit;
import com.houyidb.client.Types.SearchOptions;
import com.houyidb.client.Types.SearchResult;
import com.houyidb.client.Types.SparseTerm;
import com.houyidb.client.Types.Vector;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * HouyiDB 客户端入口（同步阻塞；{@link HttpClient} 连接池复用）。
 * 契约要点：collection 首次 upsert 隐式创建，维度由首个向量定型；payload 键
 * "label" 为保留列；检索响应 score 按 collection 度量换算（cosine 为 1 - d
 * 且钳制 [-1,1]；l2/l1 为 1 - d 不钳制；ip 为原始点积），越大越相似。
 * EN: HouyiDB client entry point (synchronous, blocking; {@link HttpClient}
 * connection pool reuse). Contract: a collection is created implicitly on first
 * upsert and its dimension is fixed by the first vector; payload key "label" is
 * reserved; search score is converted per the collection metric (cosine is
 * 1 - d clamped to [-1,1]; l2/l1 are 1 - d unclamped; ip is the raw dot
 * product), higher is closer.
 */
public final class HouyiClient {

    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    // 稀疏词项上限与服务端同口径：id ∈ [1, 2^24-1]，权重有限且 > 0；
    // 写路径每文档 ≤1024 条、查询 ≤64 条。
    // EN: Sparse term bounds match the server: id in [1, 2^24-1], weight finite
    // and > 0; at most 1024 pairs per document on writes and 64 per query.
    private static final int MAX_SPARSE_TERM_ID = (1 << 24) - 1;
    private static final int MAX_SPARSE_TERMS_PER_DOC = 1024;
    private static final int MAX_SPARSE_QUERY_TERMS = 64;

    private final HttpClient http;
    private final Duration timeout;
    private final String baseUrl;
    private final String adminToken;

    /**
     * 构造客户端；baseUrl 如 http://127.0.0.1:7000。
     * EN: Constructs a client; baseUrl such as http://127.0.0.1:7000.
     */
    public HouyiClient(String baseUrl) {
        this(baseUrl, DEFAULT_TIMEOUT, null);
    }

    /**
     * 构造客户端并指定单次请求超时（连接+读取）。
     * EN: Constructs a client with a per-request timeout (connect + read).
     */
    public HouyiClient(String baseUrl, Duration timeout) {
        this(baseUrl, timeout, null);
    }

    /**
     * 创建绑定管理 token 的客户端（默认超时）：token 只附在 /v1/admin/* 请求上，
     * 普通读写绝不携带；未配置（null/空串）时管理请求照常发出，由服务端决定
     * 401/403。
     * EN: Creates a client bound to an admin token (default timeout): the token is
     * attached only to /v1/admin/* requests. When unset (null/empty) the request
     * still goes out and the server decides 401/403.
     *
     * <p>用工厂而非 `(String, String)` 构造重载：后者会让既有
     * `new HouyiClient(url, null)` 与 `(String, Duration)` 重载二义。
     * EN: A factory instead of a `(String, String)` constructor overload: the
     * latter would make the existing `new HouyiClient(url, null)` ambiguous with
     * the `(String, Duration)` overload.
     */
    public static HouyiClient withAdminToken(String baseUrl, String adminToken) {
        return new HouyiClient(baseUrl, DEFAULT_TIMEOUT, adminToken);
    }

    /**
     * 构造客户端并同时指定超时与管理 token。
     * EN: Constructs a client with both a timeout and an admin token.
     */
    public HouyiClient(String baseUrl, Duration timeout, String adminToken) {
        this.http = HttpClient.newBuilder().connectTimeout(timeout).build();
        this.timeout = timeout;
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.adminToken = adminToken;
    }

    /**
     * 返回一个 named collection 的操作句柄。
     * EN: Returns a handle for a named collection.
     */
    public Collection collection(String name) {
        return new Collection(this, baseUrl + "/v1/collections/" + quotePathSegment(name), name);
    }

    /**
     * 服务健康检查（不可达/非 ok 一律 false，不抛异常）。
     * EN: Health check (returns false when unreachable or not ok; never throws).
     */
    public boolean health() {
        try {
            JsonNode body = getJson(baseUrl + "/v1/health");
            JsonNode status = body.get("status");
            return status != null && "ok".equals(status.asText());
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 列出全部 collection。
     * EN: Lists all collections.
     */
    public JsonNode listCollections() throws IOException, HouyiException {
        return getJson(baseUrl + "/v1/collections");
    }

    /**
     * 主索引双速写入指标（hot/pending/flush_count 等，可观测性用）。
     * EN: Primary index dual-speed write metrics (hot/pending/flush_count, for observability).
     */
    public JsonNode indexStatus() throws IOException, HouyiException {
        return getJson(baseUrl + "/v1/index/status");
    }

    /**
     * 显式创建 collection（可指定度量/后端/HNSW 参数；需要写权限）。
     *
     * 与隐式创建（首次 upsert）不同，dims、metric、后端与 HNSW 参数在这里
     * 一次定型并由服务端回显。普通写路径不带管理 token；同名已存在抛
     * {@link HouyiException}（409）、参数非法 400、follower/只读 503。
     * EN: Explicitly creates a collection (metric/backend/HNSW parameters are
     * settable; requires write permission). Unlike implicit creation (first
     * upsert), dims, metric, backend and HNSW parameters are fixed here and
     * echoed back. This is the normal write path without the admin token; an
     * existing name throws {@link HouyiException} (409), invalid parameters
     * 400, and followers/read-only nodes 503.
     */
    public CreateCollectionResult createCollection(
            String name, int dims, CreateCollectionOptions options)
            throws IOException, HouyiException {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("dims", dims);
        if (options != null) {
            if (options.metric != null) body.put("metric", options.metric);
            if (options.indexType != null) body.put("index_type", options.indexType);
            if (options.vectorDtype != null) body.put("vector_dtype", options.vectorDtype);
            if (options.quantization != null) body.put("quantization", options.quantization);
            if (options.sparse != null) body.put("sparse", options.sparse);
            if (options.hnsw != null) {
                body.put("hnsw_m", options.hnsw.m);
                body.put("hnsw_ef_construction", options.hnsw.efConstruction);
                body.put("hnsw_ef_search", options.hnsw.efSearch);
            }
        }
        JsonNode resp = putJson(baseUrl + "/v1/collections/" + quotePathSegment(name), body);
        // 非 hnsw 后端不回传 hnsw；显式 null 与缺键都按「无参数」处理。
        // EN: non-hnsw backends omit hnsw; both an explicit null and a missing
        // key mean "no parameters".
        JsonNode hnsw = resp.get("hnsw");
        HnswParams hnswParams = hnsw == null || hnsw.isNull() ? null : new HnswParams(
                hnsw.get("m").asInt(),
                hnsw.get("ef_construction").asInt(),
                hnsw.get("ef_search").asInt());
        // vector_dtype 旧服务端可能缺省：保持 null，不伪造 f32。
        // EN: older servers may omit vector_dtype: keep null and never fake f32.
        JsonNode vectorDtype = resp.get("vector_dtype");
        String vectorDtypeText =
                vectorDtype == null || vectorDtype.isNull() ? null : vectorDtype.asText();
        // quantization 旧服务端可能缺省：保持 null，不伪造 none。
        // EN: older servers may omit quantization: keep null and never fake none.
        JsonNode quantization = resp.get("quantization");
        String quantizationText =
                quantization == null || quantization.isNull() ? null : quantization.asText();
        // sparse 旧服务端可能缺省：保持 null，不伪造 none。
        // EN: older servers may omit sparse: keep null and never fake none.
        JsonNode sparse = resp.get("sparse");
        String sparseText = sparse == null || sparse.isNull() ? null : sparse.asText();
        return new CreateCollectionResult(
                resp.get("created").asBoolean(),
                resp.get("collection").asText(),
                resp.get("dims").asInt(),
                resp.get("metric").asText(),
                resp.get("index_type").asText(),
                vectorDtypeText,
                quantizationText,
                sparseText,
                hnswParams);
    }

    /**
     * 删除整个 collection（不可逆；需要 admin 权限）。
     *
     * 服务端 WAL-first：记录先落盘，崩溃重放与 follower 复制都收敛到「已删除」；
     * 不存在/已删除的 collection 抛 {@link HouyiException}（404）。
     * EN: Drops an entire collection (irreversible; admin permission required).
     * WAL-first on the server, so crash replay and follower replication converge to
     * "gone"; unknown/already-dropped collections throw {@link HouyiException} (404).
     */
    public void deleteCollection(String name) throws IOException, HouyiException {
        HttpRequest.Builder builder = HttpRequest.newBuilder(
                        URI.create(baseUrl + "/v1/admin/collections/" + quotePathSegment(name)))
                .timeout(timeout)
                .DELETE();
        // 管理请求才携带 Authorization；普通读写请求绝不上报 admin token。
        // EN: only admin requests carry Authorization; normal calls never do.
        if (adminToken != null && !adminToken.isEmpty()) {
            builder.header("Authorization", "Bearer " + adminToken);
        }
        send(builder.build());
    }

    private JsonNode getJson(String url) throws IOException, HouyiException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url)).timeout(timeout).GET().build();
        return send(req);
    }

    private JsonNode postJson(String url, ObjectNode body) throws IOException, HouyiException {
        return sendJson("POST", url, body);
    }

    private JsonNode putJson(String url, ObjectNode body) throws IOException, HouyiException {
        return sendJson("PUT", url, body);
    }

    private JsonNode sendJson(String method, String url, ObjectNode body)
            throws IOException, HouyiException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .method(method,
                        HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8))
                .build();
        return send(req);
    }

    /**
     * 非 2xx 时读取服务端纯文本诊断并抛 {@link HouyiException}。
     * EN: Reads the server's plain-text diagnostic on non-2xx and throws {@link HouyiException}.
     */
    private JsonNode send(HttpRequest req) throws IOException, HouyiException {
        HttpResponse<String> resp;
        try {
            resp = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("houyidb request interrupted", e);
        }
        if (resp.statusCode() < 200 || resp.statusCode() >= 300) {
            throw new HouyiException(resp.statusCode(), resp.body() == null ? "" : resp.body().trim());
        }
        try {
            return MAPPER.readTree(resp.body());
        } catch (IOException e) {
            throw new UncheckedIOException("houyidb response is not valid JSON", e);
        }
    }

    /**
     * 预校验稀疏词项：写路径与融合查询共用同一诊断，避免发出必然 400 的请求。
     * {@code maxTerms} 按用途给出（写 1024 / 查询 64）；null 列表只在写路径
     * 合法（表示不改），由调用方先处理。
     *
     * EN: Pre-validates sparse terms; the write and fusion-query paths share the
     * same diagnostics so a request that would surely 400 never goes out.
     * {@code maxTerms} depends on the use (1024 for writes / 64 for queries);
     * a null list is only legal on the write path (meaning unchanged) and is
     * handled by the caller first.
     */
    private static void validateSparseTerms(List<SparseTerm> terms, int maxTerms) {
        if (terms.size() > maxTerms) {
            throw new IllegalArgumentException(
                    "sparse term count " + terms.size() + " exceeds the limit " + maxTerms);
        }
        for (SparseTerm term : terms) {
            if (term == null) {
                throw new IllegalArgumentException(
                        "sparse term list must not contain null entries");
            }
            if (term.term() < 1 || term.term() > MAX_SPARSE_TERM_ID) {
                throw new IllegalArgumentException("sparse term id " + term.term()
                        + " out of range [1, " + MAX_SPARSE_TERM_ID + "]");
            }
            if (!Double.isFinite(term.weight()) || term.weight() <= 0) {
                throw new IllegalArgumentException(
                        "sparse term weight must be finite and > 0: " + term.weight());
            }
        }
    }

    private static ObjectNode vectorNode(Vector v) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("id", v.id);
        ArrayNode values = node.putArray("values");
        for (double x : v.values) values.add(x);
        if (v.label != null) node.put("label", v.label);
        if (v.payload != null) node.set("payload", MAPPER.valueToTree(v.payload));
        // null 省略该键（不改）；非 null（含空列表）输出数组，空数组即清空。
        // EN: null omits the key (unchanged); non-null (including an empty list)
        // emits an array, and an empty array clears the terms.
        if (v.sparse != null) {
            validateSparseTerms(v.sparse, MAX_SPARSE_TERMS_PER_DOC);
            ArrayNode sparse = node.putArray("sparse");
            for (SparseTerm term : v.sparse) {
                ArrayNode pair = sparse.addArray();
                pair.add(term.term());
                pair.add(term.weight());
            }
        }
        return node;
    }

    private static ObjectNode filterNode(Filter f) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("column", f.column);
        node.put("op", f.op);
        node.set("value", MAPPER.valueToTree(f.value));
        return node;
    }

    /**
     * RFC 3986 unreserved（A-Za-z0-9-._~）以外的字节一律 %XX（UTF-8），
     * 与 Python quote(safe="") 同口径：未编码的 / ? # 会改写路由结构，
     * 表现为莫名其妙的 404 且 collection 永远建不出来。
     * EN: Percent-encodes every byte outside RFC 3986 unreserved (A-Za-z0-9-._~)
     * as %XX (UTF-8), matching Python quote(safe=""): an unencoded / ? # rewrites
     * the route structure, surfacing as a puzzling 404 with the collection never
     * being created.
     */
    static String quotePathSegment(String s) {
        StringBuilder out = new StringBuilder(s.length());
        for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
            int c = b & 0xFF;
            boolean unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
                    || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_' || c == '~';
            if (unreserved) {
                out.append((char) c);
            } else {
                out.append('%');
                out.append(Character.toUpperCase(Character.forDigit((c >> 4) & 0xF, 16)));
                out.append(Character.toUpperCase(Character.forDigit(c & 0xF, 16)));
            }
        }
        return out.toString();
    }

    /**
     * 一个 named collection 的操作句柄（经 {@link #collection(String)} 获取）。
     * EN: Handle for a named collection (obtained via {@link #collection(String)}).
     */
    public static final class Collection {
        private final HouyiClient client;
        private final String base;
        private final String name;

        private Collection(HouyiClient client, String base, String name) {
            this.client = client;
            this.base = base;
            this.name = name;
        }

        /**
         * collection 名。
         * EN: Collection name.
         */
        public String name() {
            return name;
        }

        /**
         * 写入一批向量（collection 首次写入时隐式创建）。返回插入条数。
         * EN: Writes a batch of vectors (the collection is created implicitly on first
         * write). Returns the number of inserted rows.
         */
        public int upsert(List<Vector> vectors) throws IOException, HouyiException {
            ObjectNode body = MAPPER.createObjectNode();
            ArrayNode arr = body.putArray("vectors");
            for (Vector v : vectors) arr.add(vectorNode(v));
            return client.postJson(base + "/upsert", body).get("inserted").asInt();
        }

        /**
         * 纯向量 top-k 检索。
         * EN: Pure vector top-k search.
         */
        public SearchResult search(double[] vector, int k, SearchOptions options)
                throws IOException, HouyiException {
            return postSearch(base + "/search", vector, k, options, false);
        }

        /**
         * 向量 × 标量过滤的混合检索（filters 取交集，仅 hnsw/flat 后端支持）。
         * EN: Hybrid vector × scalar filter search (filters are intersected; only
         * hnsw/flat backends are supported).
         */
        public SearchResult hybridSearch(double[] vector, int k, SearchOptions options)
                throws IOException, HouyiException {
            return postSearch(base + "/hybrid/search", vector, k, options, true);
        }

        /**
         * dense + sparse 融合检索（RRF 或加权；集合须为 sparse=bm25）。
         *
         * 融合分只表达排序（不是任何度量口径）；每命中的 denseScore 按集合度量
         * 换算、sparseScore 为 BM25 原始分，某一路未命中时为 null。
         * EN: Dense + sparse fusion search (RRF or weighted; the collection must
         * be sparse=bm25). The fused score expresses ranking only (no metric
         * semantics); each hit's denseScore follows the collection metric and
         * sparseScore is the raw BM25 score, null when that modality missed.
         */
        public FusionResult fusionSearch(double[] vector, List<SparseTerm> sparse, int k,
                FusionOptions options) throws IOException, HouyiException {
            // 融合检索要求稀疏词项非空；在客户端先拒绝，避免发出必然 400 的请求。
            // EN: Fusion search requires a non-empty sparse term list; reject
            // client-side instead of sending a request that would surely 400.
            if (sparse == null || sparse.isEmpty()) {
                throw new IllegalArgumentException(
                        "fusionSearch requires a non-empty sparse term list");
            }
            validateSparseTerms(sparse, MAX_SPARSE_QUERY_TERMS);
            ObjectNode body = MAPPER.createObjectNode();
            ArrayNode values = body.putArray("vector");
            for (double x : vector) values.add(x);
            ArrayNode sparseTerms = body.putArray("sparse");
            for (SparseTerm term : sparse) {
                ArrayNode pair = sparseTerms.addArray();
                pair.add(term.term());
                pair.add(term.weight());
            }
            body.put("k", k);
            if (options != null) {
                if (!options.filters.isEmpty()) putFilters(body, options.filters);
                if (options.efSearch != null) body.put("ef_search", options.efSearch);
                if (options.includePayload) body.put("include_payload", true);
                if (options.fusion != null) body.put("fusion", options.fusion);
                if (options.rrfK != null) body.put("rrf_k", options.rrfK);
                if (options.alpha != null) body.put("alpha", options.alpha);
            }
            return toFusionResult(client.postJson(base + "/fusion/search", body));
        }

        private static void putFilters(ObjectNode body, List<Filter> filters) {
            ArrayNode arr = body.putArray("filters");
            for (Filter f : filters) arr.add(filterNode(f));
        }

        private SearchResult postSearch(String url, double[] vector, int k,
                SearchOptions options, boolean hybrid) throws IOException, HouyiException {
            ObjectNode body = MAPPER.createObjectNode();
            ArrayNode values = body.putArray("vector");
            for (double x : vector) values.add(x);
            body.put("k", k);
            if (hybrid && options != null && !options.filters.isEmpty()) {
                putFilters(body, options.filters);
            }
            if (options != null && options.efSearch != null) {
                body.put("ef_search", options.efSearch);
            }
            if (options != null && options.includePayload) {
                body.put("include_payload", true);
            }
            return toSearchResult(client.postJson(url, body));
        }

        /**
         * wire 形状：ids/scores 平行数组 + payloads 以 id 十进制字符串为键。
         * id 落在 [0, 2^63-1]，Java 有符号 long 无损承载（无 JS 的 2^53 问题）。
         * EN: Wire shape: parallel ids/scores arrays plus payloads keyed by the
         * decimal id string. Ids fall in [0, 2^63-1], which Java's signed long
         * carries losslessly (no JavaScript 2^53 issue).
         */
        @SuppressWarnings("unchecked")
        private static SearchResult toSearchResult(JsonNode body) {
            JsonNode ids = body.get("ids");
            JsonNode scores = body.get("scores");
            JsonNode payloads = body.get("payloads");
            List<SearchHit> hits = new ArrayList<>();
            int idCount = checkedIdCount("search", ids);
            if (idCount > 0) {
                for (int i = 0; i < idCount; i++) {
                    long id = checkedIdAt("search", ids, i);
                    double score = scores != null && i < scores.size() ? scores.get(i).asDouble() : 0.0;
                    Map<String, Object> payload = null;
                    if (payloads != null && payloads.has(Long.toString(id))) {
                        payload = MAPPER.convertValue(payloads.get(Long.toString(id)), Map.class);
                    }
                    hits.add(new SearchHit(id, score, payload));
                }
            }
            return new SearchResult(hits);
        }

        /**
         * 融合响应 wire 形状：ids/scores 平行数组 + dense_scores/sparse_scores
         * 平行分数组（缺席模态为 JSON null）+ metric + fusion 参数块 + 可选
         * payloads。可选字段缺席时按 null 容忍（旧服务端/精简响应）。
         * EN: Fusion wire shape: parallel ids/scores arrays plus the parallel
         * dense_scores/sparse_scores arrays (JSON null for an absent modality),
         * metric, the fusion params block, and optional payloads. Absent optional
         * fields fall back to null (older servers / slim responses).
         */
        @SuppressWarnings("unchecked")
        private static FusionResult toFusionResult(JsonNode body) {
            JsonNode ids = body.get("ids");
            JsonNode scores = body.get("scores");
            JsonNode denseScores = body.get("dense_scores");
            JsonNode sparseScores = body.get("sparse_scores");
            JsonNode payloads = body.get("payloads");
            int idCount = checkedIdCount("fusion", ids);
            // 融合分是排序键：宽度错位说明响应被截断/损坏，按协议违反显式失败，
            // 不能伪造成 0.0 让命中静默错排；同样，「存在但不是数组」也按腐坏
            // 拒绝，不能当作缺席吞掉。模态数组缺席容忍、出现时必须等宽。
            // EN: Fused scores are the ranking key; misaligned widths mean a
            // truncated/corrupt response, so fail explicitly as a protocol
            // violation instead of fabricating 0.0 and silently misordering.
            // Likewise a present-but-non-array value is rejected as corruption
            // rather than swallowed as absent. Modality arrays are tolerated
            // when absent but must match when present.
            int scoreCount = 0;
            if (scores != null && !scores.isNull()) {
                if (!scores.isArray()) {
                    throw new IllegalStateException(
                            "fusion response scores must be a JSON array");
                }
                scoreCount = scores.size();
            }
            if (scoreCount != idCount) {
                throw new IllegalStateException("fusion response scores has " + scoreCount
                        + " entries for " + idCount + " ids");
            }
            checkOptionalWidth("dense_scores", denseScores, idCount);
            checkOptionalWidth("sparse_scores", sparseScores, idCount);
            List<FusionHit> hits = new ArrayList<>();
            if (idCount > 0) {
                for (int i = 0; i < idCount; i++) {
                    long id = checkedIdAt("fusion", ids, i);
                    // asDouble() 会把 null/对象/文本静默转成 0.0；排序键必须先验
                    // 数值类型，否则腐坏响应会被伪造成合法分数。
                    // EN: asDouble() silently turns null/objects/text into 0.0;
                    // the ranking key must be type-checked first, otherwise a
                    // corrupt response is fabricated into a legal score.
                    if (!scores.get(i).isNumber()) {
                        throw new IllegalStateException("fusion response scores element "
                                + i + " is not a number");
                    }
                    double score = scores.get(i).asDouble();
                    Double denseScore = optionalScore("dense_scores", denseScores, i);
                    Double sparseScore = optionalScore("sparse_scores", sparseScores, i);
                    Map<String, Object> payload = null;
                    if (payloads != null && payloads.has(Long.toString(id))) {
                        payload = MAPPER.convertValue(payloads.get(Long.toString(id)), Map.class);
                    }
                    hits.add(new FusionHit(id, score, denseScore, sparseScore, payload));
                }
            }
            JsonNode fusion = body.get("fusion");
            JsonNode metric = body.get("metric");
            return new FusionResult(
                    hits,
                    metric == null || metric.isNull() ? null : metric.asText(),
                    fusion == null || fusion.isNull() ? null : optionalText(fusion.get("mode")),
                    // 数值缺席（JSON null 或缺键）一律回 null，不伪造默认值。
                    // EN: absent numbers (JSON null or missing key) become null;
                    // never fake a default.
                    optionalInt(fusion, "rrf_k"),
                    optionalDouble(fusion, "alpha"));
        }

        /**
         * ids 平行数组宽度：缺键/JSON null 视为空结果；存在但非数组按协议腐坏
         * 拒绝，不能当成空结果静默吞掉（与 scores 的类型硬化对称）。
         * EN: Width of the parallel ids array: a missing key / JSON null counts as
         * an empty result; a present-but-non-array value is rejected as protocol
         * corruption instead of being silently swallowed as empty (symmetric
         * with the scores type hardening).
         */
        private static int checkedIdCount(String responseName, JsonNode ids) {
            if (ids == null || ids.isNull()) {
                return 0;
            }
            if (!ids.isArray()) {
                throw new IllegalStateException(
                        responseName + " response ids must be a JSON array");
            }
            return ids.size();
        }

        /**
         * ids 元素必须是落在 [0, 2^63-1] 的 JSON 整数：asLong() 对文本/null 静默
         * 给 0，对超出 long 的 BigInteger 会静默截断，两者都会伪造命中 id。
         * EN: An ids element must be a JSON integer in [0, 2^63-1]: asLong()
         * silently yields 0 for text/null and silently truncates a BigInteger
         * beyond long, both fabricating a hit id.
         */
        private static long checkedIdAt(String responseName, JsonNode ids, int index) {
            JsonNode node = ids.get(index);
            if (!node.isIntegralNumber() || !node.canConvertToLong()) {
                throw new IllegalStateException(responseName + " response ids element "
                        + index + " is not a 64-bit signed integer");
            }
            long value = node.asLong();
            if (value < 0) {
                throw new IllegalStateException(responseName + " response ids element "
                        + index + " is negative (ids fall in [0, 2^63-1])");
            }
            return value;
        }

        /**
         * 模态分数组缺席（缺键/JSON null/空数组）容忍；出现时必须与 ids 等宽。
         * EN: A modality score array is tolerated when absent (missing key, JSON
         * null, or empty array) but must match the ids width when present.
         */
        private static void checkOptionalWidth(String name, JsonNode array, int idCount) {
            if (array == null || array.isNull()) {
                return;
            }
            if (!array.isArray()) {
                throw new IllegalStateException(
                        "fusion response " + name + " must be a JSON array");
            }
            int count = array.size();
            if (count != 0 && count != idCount) {
                throw new IllegalStateException("fusion response " + name + " has " + count
                        + " entries for " + idCount + " ids");
            }
        }

        private static Double optionalScore(String name, JsonNode scores, int index) {
            if (scores == null || !scores.isArray() || index >= scores.size()) {
                return null;
            }
            JsonNode node = scores.get(index);
            if (node.isNull()) {
                return null;
            }
            // 数值类型先验：asDouble() 对对象/文本会静默给 0.0，伪造模态分。
            // EN: Check the numeric type first: asDouble() silently yields 0.0
            // for objects/text, fabricating a modality score.
            if (!node.isNumber()) {
                throw new IllegalStateException("fusion response " + name + " element "
                        + index + " is not a number");
            }
            return node.asDouble();
        }

        private static String optionalText(JsonNode node) {
            return node == null || node.isNull() ? null : node.asText();
        }

        private static Integer optionalInt(JsonNode node, String field) {
            JsonNode value = node == null || node.isNull() ? null : node.get(field);
            return value == null || value.isNull() ? null : value.asInt();
        }

        private static Double optionalDouble(JsonNode node, String field) {
            JsonNode value = node == null || node.isNull() ? null : node.get(field);
            return value == null || value.isNull() ? null : value.asDouble();
        }

        /**
         * 按 id 和/或过滤条件删除（幂等；两者取并集）。返回真实删除数。
         * 按条件删除由服务端在标量页上求值：标量行随 upsert 同步落页，因此刚写入、
         * 尚在异步构图队列中的向量也能被命中删除——这是「先检索拿 id 再删」无法保证的。
         * EN: Deletes by id and/or filters (idempotent; the two sets are unioned).
         * Returns the number actually deleted. Filter deletes are evaluated
         * server-side on scalar pages: scalar rows land synchronously with upsert,
         * so vectors just written and still in the async graph-build queue can also
         * be matched and deleted, which "search first, delete by id" cannot guarantee.
         */
        public int delete(List<Long> ids, List<Filter> filters) throws IOException, HouyiException {
            if ((ids == null || ids.isEmpty()) && (filters == null || filters.isEmpty())) {
                throw new IllegalArgumentException("delete requires at least one of ids or filters");
            }
            ObjectNode body = MAPPER.createObjectNode();
            ArrayNode idArr = body.putArray("ids");
            if (ids != null) ids.forEach(idArr::add);
            ArrayNode filterArr = body.putArray("filters");
            if (filters != null) filters.forEach(f -> filterArr.add(filterNode(f)));
            return client.postJson(base + "/delete", body).get("deleted").asInt();
        }

        /**
         * collection 状态（indexed/pending/dims/index_type/scalar_row_count）。
         * EN: Collection status (indexed/pending/dims/index_type/scalar_row_count).
         */
        public JsonNode status() throws IOException, HouyiException {
            return client.getJson(base + "/status");
        }
    }
}
