package com.houyidb.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.houyidb.client.Types.CreateCollectionOptions;
import com.houyidb.client.Types.CreateCollectionResult;
import com.houyidb.client.Types.Filter;
import com.houyidb.client.Types.HnswParams;
import com.houyidb.client.Types.SearchHit;
import com.houyidb.client.Types.SearchOptions;
import com.houyidb.client.Types.SearchResult;
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
        return new CreateCollectionResult(
                resp.get("created").asBoolean(),
                resp.get("collection").asText(),
                resp.get("dims").asInt(),
                resp.get("metric").asText(),
                resp.get("index_type").asText(),
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

    private static ObjectNode vectorNode(Vector v) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("id", v.id);
        ArrayNode values = node.putArray("values");
        for (double x : v.values) values.add(x);
        if (v.label != null) node.put("label", v.label);
        if (v.payload != null) node.set("payload", MAPPER.valueToTree(v.payload));
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

        private SearchResult postSearch(String url, double[] vector, int k,
                SearchOptions options, boolean hybrid) throws IOException, HouyiException {
            ObjectNode body = MAPPER.createObjectNode();
            ArrayNode values = body.putArray("vector");
            for (double x : vector) values.add(x);
            body.put("k", k);
            if (hybrid && options != null && !options.filters.isEmpty()) {
                ArrayNode filters = body.putArray("filters");
                for (Filter f : options.filters) filters.add(filterNode(f));
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
            if (ids != null && ids.isArray()) {
                for (int i = 0; i < ids.size(); i++) {
                    long id = ids.get(i).asLong();
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
