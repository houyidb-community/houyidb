// P4 新增面的 wire 契约测试（JDK HttpServer，无真实服务端）：
// 逐向量 sparse 键存在性语义、创建 sparse 选项与旧服务端缺席容忍、
// 融合检索路由/请求体/响应平行分数组与 fusion 参数块。
//
// EN: Wire contract tests for the P4 additions (JDK HttpServer, no real
// server): per-vector sparse key-presence semantics, the create sparse option
// plus tolerance for older servers omitting it, and the fusion
// route/request body/response parallel score arrays plus the fusion params
// block.
package com.houyidb.client;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.houyidb.client.Types.CreateCollectionOptions;
import com.houyidb.client.Types.CreateCollectionResult;
import com.houyidb.client.Types.FusionHit;
import com.houyidb.client.Types.FusionOptions;
import com.houyidb.client.Types.FusionResult;
import com.houyidb.client.Types.SparseTerm;
import com.houyidb.client.Types.Vector;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class WireContractTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * 回放固定响应并记录最后一次请求（latch 保证跨线程可见性）。
     * EN: Replays a fixed response and records the last request (a latch
     * provides cross-thread visibility).
     */
    private static final class Capture {
        volatile String method;
        volatile String path;
        volatile String body;
        final CountDownLatch done = new CountDownLatch(1);
    }

    private interface ServerCall {
        void run(HouyiClient client) throws Exception;
    }

    private static void withServer(String resp, Capture cap, ServerCall call) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            byte[] raw = exchange.getRequestBody().readAllBytes();
            cap.method = exchange.getRequestMethod();
            cap.path = exchange.getRequestURI().getPath();
            cap.body = new String(raw, StandardCharsets.UTF_8);
            byte[] out = resp.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, out.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(out);
            }
            cap.done.countDown();
        });
        server.start();
        try {
            call.run(new HouyiClient("http://127.0.0.1:" + server.getAddress().getPort()));
            assertTrue(cap.done.await(2, TimeUnit.SECONDS), "capture server never replied");
        } finally {
            server.stop(0);
        }
    }

    /**
     * 缺省 null 省略键（不改），空列表输出 []（清空），词项输出 [term, weight]。
     * EN: null omits the key (unchanged), an empty list emits [] (clear), and
     * terms emit [term, weight] pairs.
     */
    @Test
    void vectorSparseWireShape() throws Exception {
        Capture cap = new Capture();
        withServer("{\"inserted\":3}", cap, client -> client.collection("docs").upsert(List.of(
                new Vector(1, new double[] {1.0}, null, null),
                new Vector(2, new double[] {1.0}, null, null, List.of()),
                new Vector(3, new double[] {1.0}, null, null,
                        List.of(new SparseTerm(3, 1.5), new SparseTerm(5, 0.25))))));
        JsonNode vectors = MAPPER.readTree(cap.body).get("vectors");
        assertFalse(vectors.get(0).has("sparse"), "null sparse must omit the key");
        assertTrue(vectors.get(1).get("sparse").isArray(), "empty sparse must serialize as []");
        assertEquals(0, vectors.get(1).get("sparse").size());
        JsonNode sparse = vectors.get(2).get("sparse");
        assertEquals(2, sparse.size());
        assertEquals(3, sparse.get(0).get(0).asInt());
        assertEquals(1.5, sparse.get(0).get(1).asDouble());
    }

    /**
     * 创建请求携带 sparse 且响应回显。
     * EN: The create request carries sparse and the response echoes it.
     */
    @Test
    void createCollectionSendsSparse() throws Exception {
        Capture cap = new Capture();
        // Java 无 out 参数，改用单元素数组搬运回调结果。
        // EN: no out-params in Java; capture the result through an array.
        CreateCollectionResult[] holder = new CreateCollectionResult[1];
        withServer("{\"created\":true,\"collection\":\"docs\",\"dims\":8,"
                + "\"metric\":\"cosine\",\"index_type\":\"flat\",\"sparse\":\"bm25\"}", cap,
                client -> holder[0] = client.createCollection("docs", 8,
                        new CreateCollectionOptions().indexType("flat").sparse("bm25")));
        assertEquals("PUT", cap.method);
        assertEquals("/v1/collections/docs", cap.path);
        JsonNode body = MAPPER.readTree(cap.body);
        assertEquals("bm25", body.get("sparse").asText());
        assertEquals("flat", body.get("index_type").asText());
        assertEquals("bm25", holder[0].sparse);
    }

    /**
     * 未设置 sparse 时请求体不带该键；旧服务端响应缺 sparse 时保持 null。
     * EN: The request omits the key when sparse is unset; a response without
     * sparse from an older server leaves it null.
     */
    @Test
    void createCollectionOmitsSparseWhenUnset() throws Exception {
        Capture cap = new Capture();
        CreateCollectionResult[] holder = new CreateCollectionResult[1];
        withServer("{\"created\":true,\"collection\":\"legacy\",\"dims\":4,"
                + "\"metric\":\"l2\",\"index_type\":\"flat\"}", cap,
                client -> holder[0] = client.createCollection("legacy", 4, null));
        assertFalse(MAPPER.readTree(cap.body).has("sparse"), "unset sparse must not be sent");
        assertNull(holder[0].sparse, "older-server response must leave sparse null");
    }

    /**
     * 写路径与融合查询在发请求前拒绝非法稀疏词项（词项范围/权重/数量），
     * 与 Rust/Python SDK 同口径，不把必然 400 的请求发到线上。
     * EN: The write path and fusion query reject illegal sparse terms (id range,
     * weight, count) before sending, matching the Rust/Python SDKs instead of
     * sending a request that would surely 400.
     */
    @Test
    void sparseTermsRejectedBeforeWire() {
        HouyiClient client =
                new HouyiClient("http://127.0.0.1:1", java.time.Duration.ofMillis(200));
        HouyiClient.Collection col = client.collection("docs");
        List<SparseTerm> tooMany = new java.util.ArrayList<>();
        for (int i = 1; i <= 65; i++) {
            tooMany.add(new SparseTerm(i, 1.0));
        }
        assertThrows(
                IllegalArgumentException.class,
                () -> col.upsert(List.of(new Vector(1, new double[] {1.0}, null, null,
                        List.of(new SparseTerm(0, 1.0))))),
                "term id 0 must be rejected");
        assertThrows(
                IllegalArgumentException.class,
                () -> col.upsert(List.of(new Vector(1, new double[] {1.0}, null, null,
                        List.of(new SparseTerm(1, Double.NaN))))),
                "NaN weight must be rejected");
        assertThrows(
                IllegalArgumentException.class,
                () -> col.upsert(List.of(new Vector(1, new double[] {1.0}, null, null,
                        List.of(new SparseTerm(1, 0.0))))),
                "zero weight must be rejected");
        assertThrows(
                IllegalArgumentException.class,
                () -> col.fusionSearch(new double[] {1.0},
                        List.of(new SparseTerm(16_777_216, 1.0)), 1, null),
                "query term id above 2^24-1 must be rejected");
        assertThrows(
                IllegalArgumentException.class,
                () -> col.fusionSearch(new double[] {1.0}, tooMany, 1, null),
                "more than 64 query terms must be rejected");
    }

    /**
     * 融合响应的平行分数组宽度错位必须显式失败，而不是伪造 0.0 静默错排。
     * EN: Misaligned parallel score arrays must fail explicitly instead of
     * fabricating 0.0 and silently misordering.
     */
    @Test
    void fusionResponseRejectsMisalignedScores() throws Exception {
        Capture cap = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":[1,2],\"scores\":[0.3]}", cap,
                client -> client.collection("docs").fusionSearch(
                        new double[] {1.0}, List.of(new SparseTerm(1, 1.0)), 2, null)));
    }

    /**
     * 存在但类型错误的字段按腐坏拒绝：模态字段非数组、分数元素非数值都不得
     * 被 asDouble()/缺席容忍静默吞掉。
     * EN: Present-but-wrong-typed fields are rejected as corruption: a non-array
     * modality field and a non-numeric score element must not be silently
     * swallowed by asDouble()/absent tolerance.
     */
    @Test
    void fusionResponseRejectsWrongTypedFields() throws Exception {
        Capture cap = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":[1],\"scores\":[0.3],\"dense_scores\":0.9}", cap,
                client -> client.collection("docs").fusionSearch(
                        new double[] {1.0}, List.of(new SparseTerm(1, 1.0)), 1, null)));
        Capture cap2 = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":[1,2],\"scores\":[null,0.2]}", cap2,
                client -> client.collection("docs").fusionSearch(
                        new double[] {1.0}, List.of(new SparseTerm(1, 1.0)), 2, null)));
    }

    /**
     * ids 平行数组与分数侧同样严格：存在但非数组、元素非整数都按腐坏拒绝，
     * 不能被 asLong()/空结果容忍静默吞掉。
     * EN: The parallel ids array is as strict as the score side: a
     * present-but-non-array value and non-integer elements are rejected as
     * corruption instead of being swallowed by asLong()/empty-result tolerance.
     */
    @Test
    void idsArrayTypeChecked() throws Exception {
        Capture cap = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":\"oops\"}", cap,
                client -> client.collection("docs").search(new double[] {1.0}, 1, null)));
        Capture cap2 = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":[\"x\"],\"scores\":[0.5]}", cap2,
                client -> client.collection("docs").search(new double[] {1.0}, 1, null)));
        Capture cap3 = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":{}, \"scores\":[0.5]}", cap3,
                client -> client.collection("docs").fusionSearch(
                        new double[] {1.0}, List.of(new SparseTerm(1, 1.0)), 1, null)));
        Capture cap4 = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":[null],\"scores\":[0.5]}", cap4,
                client -> client.collection("docs").fusionSearch(
                        new double[] {1.0}, List.of(new SparseTerm(1, 1.0)), 1, null)));
        // 超出 long 的 u64（Java 无法无损承载）与负数 id 同样显式拒绝，
        // 不能经 asLong() 静默回绕/伪造。
        // EN: u64 values beyond long (not losslessly representable in Java) and
        // negative ids are rejected explicitly instead of silently wrapping
        // through asLong().
        Capture cap5 = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":[9223372036854775808],\"scores\":[0.5]}", cap5,
                client -> client.collection("docs").search(new double[] {1.0}, 1, null)));
        Capture cap6 = new Capture();
        assertThrows(IllegalStateException.class, () -> withServer(
                "{\"ids\":[-1],\"scores\":[0.5]}", cap6,
                client -> client.collection("docs").search(new double[] {1.0}, 1, null)));
    }

    /**
     * 融合路由/请求体/响应逐项对齐。
     * EN: Fusion route/request body/response line up item by item.
     */
    @Test
    void fusionSearchWireAndResponse() throws Exception {
        Capture cap = new Capture();
        FusionResult[] holder = new FusionResult[1];
        withServer("{\"ids\":[9,4],\"scores\":[0.75,0.02],\"dense_scores\":[0.9,null],"
                + "\"sparse_scores\":[null,4.2],\"metric\":\"cosine\","
                + "\"fusion\":{\"mode\":\"rrf\",\"rrf_k\":60,\"alpha\":null},"
                + "\"payloads\":{\"9\":{\"text\":\"nine\"}}}", cap,
                client -> holder[0] = client.collection("docs").fusionSearch(
                        new double[] {1.0, 0.0},
                        List.of(new SparseTerm(7, 0.5), new SparseTerm(8, 0.25)), 2,
                        new FusionOptions().includePayload(true).fusion("weighted")
                                .rrfK(7).alpha(0.25)));
        assertEquals("POST", cap.method);
        assertEquals("/v1/collections/docs/fusion/search", cap.path);
        JsonNode body = MAPPER.readTree(cap.body);
        assertEquals(2, body.get("k").asInt());
        assertTrue(body.get("include_payload").asBoolean());
        assertEquals("weighted", body.get("fusion").asText());
        assertEquals(7, body.get("rrf_k").asInt());
        assertEquals(0.25, body.get("alpha").asDouble());
        assertEquals(2, body.get("sparse").size());

        FusionResult result = holder[0];
        assertEquals("rrf", result.fusionMode);
        assertEquals(60, result.rrfK);
        assertNull(result.alpha, "rrf mode must not echo alpha");
        assertEquals("cosine", result.metric);
        assertEquals(2, result.hits.size());
        FusionHit denseOnly = result.hits.get(0);
        assertEquals(9, denseOnly.id);
        assertEquals(0.9, denseOnly.denseScore);
        assertNull(denseOnly.sparseScore, "dense-only hit must have null sparse score");
        assertEquals("nine", denseOnly.payload.get("text"));
        FusionHit sparseOnly = result.hits.get(1);
        assertEquals(4, sparseOnly.id);
        assertNull(sparseOnly.denseScore, "sparse-only hit must have null dense score");
        assertEquals(4.2, sparseOnly.sparseScore);
    }

    /**
     * 响应缺可选字段时不报错，且不得伪造默认值。
     * EN: A response missing optional fields does not error and never fakes
     * defaults.
     */
    @Test
    void fusionResponseToleratesAbsentOptionalFields() throws Exception {
        Capture cap = new Capture();
        FusionResult[] holder = new FusionResult[1];
        withServer("{\"ids\":[1,2],\"scores\":[0.3,0.1]}", cap,
                client -> holder[0] = client.collection("docs").fusionSearch(
                        new double[] {1.0}, List.of(new SparseTerm(1, 1.0)), 2, null));
        FusionResult result = holder[0];
        assertEquals(2, result.hits.size());
        for (FusionHit hit : result.hits) {
            assertNull(hit.denseScore);
            assertNull(hit.sparseScore);
            assertNull(hit.payload);
        }
        assertNull(result.metric);
        assertNull(result.fusionMode);
        assertNull(result.rrfK);
        assertNull(result.alpha);
    }
}
