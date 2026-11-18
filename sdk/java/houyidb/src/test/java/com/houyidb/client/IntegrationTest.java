// houyidb Java SDK 与真实服务的集成测试。
// EN: Integration tests for the houyidb Java SDK against a live server.
//
// 前置：houyi-server 监听 127.0.0.1:17000（与其他语言 SDK 联调同一实例），
// 可用 HOUYIDB_URL 覆盖；服务不可达时经 JUnit Assumptions 整体跳过。
// EN: Prerequisite: houyi-server listens on 127.0.0.1:17000 (shared with the
// other language SDKs); override with HOUYIDB_URL; when unreachable, JUnit
// Assumptions skips the whole class.
// 运行：gradle test
// EN: Run with: gradle test
//
// 并行安全：各用例独占 collection 名（WAL 持久化 + 同 id upsert 幂等，重跑安全）；
// 检索类断言前一律 waitFlushed（向量检索看不到热缓冲中的数据）。
// EN: Parallel safety: each test owns its collection name (WAL persistence plus
// idempotent upsert by id make reruns safe); search assertions always call
// waitFlushed first (vector search cannot see data in the hot buffer).
package com.houyidb.client;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.houyidb.client.Types.CreateCollectionOptions;
import com.houyidb.client.Types.CreateCollectionResult;
import com.houyidb.client.Types.Filter;
import com.houyidb.client.Types.FilterOp;
import com.houyidb.client.Types.FusionHit;
import com.houyidb.client.Types.FusionOptions;
import com.houyidb.client.Types.FusionResult;
import com.houyidb.client.Types.HnswParams;
import com.houyidb.client.Types.SearchHit;
import com.houyidb.client.Types.SearchOptions;
import com.houyidb.client.Types.SearchResult;
import com.houyidb.client.Types.SparseTerm;
import com.houyidb.client.Types.Vector;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.bouncycastle.crypto.digests.Blake2bDigest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class IntegrationTest {

    private static final int TEST_DIMS = 16;

    private HouyiClient client;

    private static String baseUrl() {
        String v = System.getenv("HOUYIDB_URL");
        return v != null ? v.replaceAll("/+$", "") : "http://127.0.0.1:17000";
    }

    @BeforeEach
    void connect() {
        client = new HouyiClient(baseUrl(), Duration.ofSeconds(3));
        assumeTrue(client::health,
                () -> "houyi-server 不可达（" + baseUrl() + "）：先启动 "
                        + "./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000");
    }

    /**
     * 确定性伪 embedding（与其他语言 SDK 联调同款思路，自洽即可）。
     * EN: Deterministic pseudo-embedding (same idea as the other language SDKs;
     * it merely needs to be self-consistent).
     */
    private static double[] fakeEmbed(String text) {
        double[] out = new double[TEST_DIMS];
        for (int counter = 0; counter < TEST_DIMS; counter++) {
            Blake2bDigest digest = new Blake2bDigest(64);
            byte[] input = (text + ":" + counter).getBytes(StandardCharsets.UTF_8);
            digest.update(input, 0, input.length);
            byte[] sum = new byte[8];
            digest.doFinal(sum, 0);
            long v = ByteBuffer.wrap(sum).getLong();
            // 无符号换算：v 可能为负，先按无符号解释再归一化
            // EN: unsigned conversion: v may be negative, interpret it as unsigned
            // before normalizing
            double unsigned = v >= 0 ? (double) v : (double) v + 18446744073709551616.0;
            out[counter] = unsigned / 18446744073709551616.0 * 2.0 - 1.0;
        }
        return out;
    }

    /**
     * 等待异步构图队列排空（双速写入：flush 后才对检索可见）。
     * EN: Waits for the async graph-build queue to drain (dual-speed writes are
     * visible to search only after flush).
     */
    private static void waitFlushed(HouyiClient.Collection col) throws Exception {
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline) {
            if (col.status().get("pending_count").asLong() == 0) {
                return;
            }
            Thread.sleep(20);
        }
        throw new AssertionError("collection " + col.name() + " flush timeout");
    }

    private static Vector doc(String idText, String text, String category, int price) {
        return new Vector(StableId.of(idText), fakeEmbed(text),
                Map.of("text", text, "category", category, "price", price));
    }

    @Test
    void upsertSearchRoundTrip() throws Exception {
        HouyiClient.Collection col = client.collection("java_it");
        int inserted = col.upsert(List.of(
                doc("java-doc-0", "apple is a fruit", "fruit", 5),
                doc("java-doc-1", "banana is yellow", "fruit", 3),
                doc("java-doc-2", "laptop is electronics", "electronics", 999)));
        assertEquals(3, inserted);
        waitFlushed(col);

        SearchResult result = col.search(fakeEmbed("apple is a fruit"), 2,
                new SearchOptions().includePayload(true));
        assertEquals(2, result.hits.size());
        assertEquals("apple is a fruit", result.hits.get(0).payload.get("text"));
        assertTrue(result.hits.get(0).score >= result.hits.get(1).score,
                "scores must be descending");
    }

    @Test
    void hybridSearchFilters() throws Exception {
        HouyiClient.Collection col = client.collection("java_it_hybrid");
        col.upsert(List.of(
                doc("java-hybrid-0", "apple is a fruit", "fruit", 5),
                doc("java-hybrid-1", "banana is yellow", "fruit", 3),
                doc("java-hybrid-2", "laptop is electronics", "electronics", 999)));
        waitFlushed(col);

        SearchResult result = col.hybridSearch(fakeEmbed("apple is a fruit"), 3,
                new SearchOptions().includePayload(true)
                        .filter("category", FilterOp.EQ, "fruit")
                        .filter("price", FilterOp.LT, 4));
        assertEquals(1, result.hits.size(), "expected only banana");
        assertEquals("banana is yellow", result.hits.get(0).payload.get("text"));
    }

    /**
     * 异键集 payload：PAX 新列回填的端到端守卫（曾在此场景整列错位）。
     * EN: Disjoint payload key sets: end-to-end guard for PAX new-column backfill
     * (a whole column once got misaligned in this scenario).
     */
    @Test
    void mixedPayloadKeySets() throws Exception {
        HouyiClient.Collection col = client.collection("java_it_mixed_keys");
        col.upsert(List.of(
                new Vector(1, fakeEmbed("row-a"), Map.of("a", 1)),
                new Vector(2, fakeEmbed("row-b"), Map.of("b", 2))));
        // 过滤按标量页同步求值，但向量侧仍需构图完成才能命中。
        // EN: Filters are evaluated synchronously on scalar pages, but the vector
        // side still needs the graph build to finish before it can match.
        waitFlushed(col);

        SearchResult result = col.hybridSearch(fakeEmbed("row-b"), 5,
                new SearchOptions().includePayload(true).filter("b", FilterOp.EQ, 2));
        assertEquals(1, result.hits.size(), "filter b==2 must hit one row");
        SearchHit hit = result.hits.get(0);
        assertEquals(2, hit.id);
        assertFalse(hit.payload.containsKey("a"), "payload must not leak column 'a'");
    }

    @Test
    void largeUint64IdRoundTrip() throws Exception {
        HouyiClient.Collection col = client.collection("java_it_bigid");

        // StableId 落点常态是 19 位十进制（>2^53）；Java long 原生无损。
        // EN: StableId values are normally 19-digit decimals (>2^53); Java long is
        // lossless natively.
        long id = StableId.of("hello");
        col.upsert(List.of(new Vector(id, fakeEmbed("uint64 probe"), Map.of("tag", "big"))));
        waitFlushed(col);

        SearchResult result = col.hybridSearch(fakeEmbed("uint64 probe"), 1,
                new SearchOptions().includePayload(true).filter("tag", FilterOp.EQ, "big"));
        assertFalse(result.hits.isEmpty(), "no hit for big-id probe");
        assertEquals(id, result.hits.get(0).id, "u64 id round trip broken");

        assertTrue(col.delete(List.of(id), null) <= 1);
    }

    @Test
    void deleteByIdsAndFilters() throws Exception {
        HouyiClient.Collection col = client.collection("java_it_delete");

        long probeId = StableId.of("java-delete-me");
        col.upsert(List.of(new Vector(probeId, fakeEmbed("delete probe"), Map.of("tag", "del"))));
        // delete-by-filter 不依赖构图排空（标量页同步落页）。
        // EN: delete-by-filter does not depend on the graph-build drain (scalar
        // pages land synchronously).
        assertEquals(1, col.delete(null, List.of(new Filter("tag", FilterOp.EQ, "del"))));
        assertEquals(0, col.delete(List.of(probeId), null), "idempotent delete");
    }

    @Test
    void serverErrorSurfacesAsHouyiException() {
        HouyiClient.Collection col = client.collection("java_it_missing_collection");
        HouyiException ex = assertThrows(HouyiException.class, col::status);
        assertEquals(404, ex.statusCode());
    }

    @Test
    void deleteRequiresIdsOrFilters() {
        HouyiClient.Collection col = client.collection("java_it");
        assertThrows(IllegalArgumentException.class, () -> col.delete(null, null));
    }

    /**
     * collection 名含保留字符时路径转义（与 Python quote(safe="") 同口径）。
     * EN: Path escaping when a collection name contains reserved characters
     * (same rule as Python quote(safe="")).
     */
    @Test
    void collectionNameWithReservedChars() throws Exception {
        HouyiClient.Collection col = client.collection("java 名/a?b#c");
        col.upsert(List.of(new Vector(42, fakeEmbed("quoted"), Map.of("k", "v"))));
        assertEquals(1, col.delete(List.of(42L), null));
    }

    /**
     * 显式建集合回显度量与 HNSW 三元组；同名再建 409。
     * EN: Explicit creation echoes the metric and HNSW triple; a second create
     * of the same name yields 409.
     */
    @Test
    void createCollectionMetricAndHnsw() throws Exception {
        String name = "java_it_create_metric";
        // 清理只容忍 404（从未创建）；其他错误直接失败。
        // EN: Cleanup tolerates only 404 (never created); anything else fails.
        try {
            client.deleteCollection(name);
        } catch (HouyiException e) {
            assertEquals(404, e.statusCode(), "cleanup delete must succeed or 404");
        }

        CreateCollectionResult created = client.createCollection(name, TEST_DIMS,
                new CreateCollectionOptions()
                        .metric("ip")
                        .indexType("hnsw")
                        .hnsw(new HnswParams(16, 100, 32)));
        assertTrue(created.created);
        assertEquals(name, created.collection);
        assertEquals(TEST_DIMS, created.dims);
        assertEquals("ip", created.metric);
        assertEquals("hnsw", created.indexType);
        assertNotNull(created.hnsw, "hnsw echo must be present");
        assertEquals(16, created.hnsw.m);
        assertEquals(100, created.hnsw.efConstruction);
        assertEquals(32, created.hnsw.efSearch);

        HouyiException conflict = assertThrows(HouyiException.class,
                () -> client.createCollection(name, TEST_DIMS, null));
        assertEquals(409, conflict.statusCode());
    }

    /**
     * 显式建 f16 + int8 hnsw 集合：vector_dtype/quantization 透传、响应回显、
     * status 可观测。
     * EN: Explicitly creates an f16 + int8 hnsw collection: vector_dtype and
     * quantization are forwarded, echoed in the response, and observable via
     * status.
     */
    @Test
    void createCollectionF16Int8RoundTrip() throws Exception {
        String name = "java_it_f16_int8";
        try {
            client.deleteCollection(name);
        } catch (HouyiException e) {
            // 不存在（404）或启用了鉴权（401/403）：无需清理。
            // EN: missing (404) or auth enabled (401/403): nothing to clean up.
            assertTrue(e.statusCode() == 404 || e.statusCode() == 401 || e.statusCode() == 403,
                    "cleanup delete must succeed or be 404/401/403");
        }

        CreateCollectionResult created = client.createCollection(name, TEST_DIMS,
                new CreateCollectionOptions()
                        .metric("l2")
                        .indexType("hnsw")
                        .vectorDtype("f16")
                        .quantization("int8")
                        .hnsw(new HnswParams(16, 100, 32)));
        assertEquals("f16", created.vectorDtype);
        assertEquals("int8", created.quantization);
        assertEquals("hnsw", created.indexType);

        var status = client.collection(name).status();
        assertEquals("f16", status.get("vector_dtype").asText());
        assertEquals("int8", status.get("quantization").asText());

        client.deleteCollection(name);
    }

    /**
     * 显式建 rabitq（l2/cosine-only）hnsw 集合：请求透传、响应与 status 回显；
     * ip 组合返回 400。
     * EN: Explicitly creates a rabitq (l2/cosine-only) hnsw collection: the
     * request is forwarded, the response and status echo it, and the ip
     * combination returns 400.
     */
    @Test
    void createCollectionRabitqRoundTrip() throws Exception {
        String name = "java_it_rabitq";
        try {
            client.deleteCollection(name);
        } catch (HouyiException e) {
            // 不存在（404）或启用了鉴权（401/403）：无需清理。
            // EN: missing (404) or auth enabled (401/403): nothing to clean up.
            assertTrue(e.statusCode() == 404 || e.statusCode() == 401 || e.statusCode() == 403,
                    "cleanup delete must succeed or be 404/401/403");
        }

        CreateCollectionResult created = client.createCollection(name, TEST_DIMS,
                new CreateCollectionOptions()
                        .metric("cosine")
                        .indexType("hnsw")
                        .vectorDtype("f32")
                        .quantization("rabitq")
                        .hnsw(new HnswParams(16, 100, 32)));
        assertEquals("cosine", created.metric);
        assertEquals("hnsw", created.indexType);
        assertEquals("rabitq", created.quantization);

        var status = client.collection(name).status();
        assertEquals("rabitq", status.get("quantization").asText());

        HouyiException bad = assertThrows(HouyiException.class,
                () -> client.createCollection(name + "_ip", TEST_DIMS,
                        new CreateCollectionOptions()
                                .metric("ip")
                                .indexType("hnsw")
                                .quantization("rabitq")));
        assertEquals(400, bad.statusCode());

        client.deleteCollection(name);
    }

    /**
     * 删除整库后 404；同名重建得到空 collection。
     * EN: Drops a whole collection (404 afterwards) and expects a same-name
     * recreate to start empty.
     */
    @Test
    void deleteCollectionAndRecreate() throws Exception {
        String name = "java_it_admin_drop";
        HouyiClient.Collection col = client.collection(name);
        col.upsert(List.of(new Vector(7, fakeEmbed("drop me"), Map.of("tag", "drop"))));

        client.deleteCollection(name);

        HouyiException ex = assertThrows(HouyiException.class, col::status);
        assertEquals(404, ex.statusCode());

        // 同名重建：首次写入隐式创建，旧向量不得复活。
        // EN: recreate by name: the first write implicitly creates it and the old
        // vectors must not reappear.
        col.upsert(List.of(new Vector(8, fakeEmbed("fresh"), Map.of())));
        waitFlushed(col);
        assertEquals(1, col.status().get("indexed_count").asLong());
    }

    /**
     * 删除不存在的 collection 返回上游 404。
     * EN: Dropping an unknown collection surfaces the upstream 404.
     */
    @Test
    void deleteMissingCollectionIs404() {
        HouyiException ex = assertThrows(HouyiException.class,
                () -> client.deleteCollection("java_it_admin_never_created"));
        assertEquals(404, ex.statusCode());
    }

    /**
     * bm25 建集合 → 带稀疏 upsert → 融合检索（RRF/加权回显）→ sparse=[] 清空
     * 后仅剩稠密分。
     * EN: Creates a bm25 collection, upserts with sparse terms, runs a fusion
     * search (RRF/weighted echoes), and verifies that clearing with sparse=[]
     * leaves the document with a dense score only.
     */
    @Test
    void sparseCreateAndFusionSearch() throws Exception {
        String name = "java_it_fusion";
        try {
            client.deleteCollection(name);
        } catch (HouyiException e) {
            // 不存在（404）或启用了鉴权（401/403）：无需清理。
            // EN: missing (404) or auth enabled (401/403): nothing to clean up.
            assertTrue(e.statusCode() == 404 || e.statusCode() == 401 || e.statusCode() == 403,
                    "cleanup delete must succeed or be 404/401/403");
        }

        CreateCollectionResult created = client.createCollection(name, TEST_DIMS,
                new CreateCollectionOptions().indexType("flat").sparse("bm25"));
        assertEquals("bm25", created.sparse);
        assertEquals("flat", created.indexType);

        HouyiClient.Collection col = client.collection(name);
        col.upsert(List.of(
                new Vector(1, fakeEmbed("apple"), null, Map.of("text", "apple"),
                        List.of(new SparseTerm(7, 1.0), new SparseTerm(8, 0.5))),
                new Vector(2, fakeEmbed("banana"), null, Map.of("text", "banana"),
                        List.of(new SparseTerm(8, 1.0))),
                new Vector(3, fakeEmbed("cherry"), null, Map.of("text", "cherry"),
                        List.of(new SparseTerm(9, 1.0)))));
        waitFlushed(col);

        FusionResult result = col.fusionSearch(fakeEmbed("apple"),
                List.of(new SparseTerm(7, 1.0), new SparseTerm(8, 0.5)), 2,
                new FusionOptions().includePayload(true));
        assertFalse(result.hits.isEmpty(), "fusion must return hits");
        FusionHit top = result.hits.get(0);
        assertEquals(1, top.id);
        assertEquals("rrf", result.fusionMode);
        assertEquals(60, result.rrfK);
        assertNull(result.alpha, "rrf must not echo alpha");
        assertNotNull(top.denseScore, "both-modality hit must carry a dense score");
        assertNotNull(top.sparseScore, "both-modality hit must carry a sparse score");
        assertEquals("apple", top.payload.get("text"));
        assertEquals("cosine", result.metric, "flat default metric");

        FusionResult weighted = col.fusionSearch(fakeEmbed("apple"),
                List.of(new SparseTerm(7, 1.0)), 2,
                new FusionOptions().fusion("weighted").alpha(0.25));
        assertEquals("weighted", weighted.fusionMode);
        assertEquals(0.25, weighted.alpha);
        assertNull(weighted.rrfK, "weighted must not echo rrf_k");

        // sparse=[] 清空该文档词项（null 才是「不改」）；清空后仅剩稠密分。
        // EN: sparse=[] clears this document's terms (null means "unchanged");
        // afterwards only the dense score remains.
        col.upsert(List.of(new Vector(1, fakeEmbed("apple"), null,
                Map.of("text", "apple"), List.of())));
        waitFlushed(col);
        FusionResult cleared = col.fusionSearch(fakeEmbed("apple"),
                List.of(new SparseTerm(8, 1.0)), 3, null);
        FusionHit cleared1 = null;
        for (FusionHit hit : cleared.hits) {
            if (hit.id == 1) {
                cleared1 = hit;
            }
        }
        assertNotNull(cleared1, "cleared doc must stay reachable via the dense path");
        assertNotNull(cleared1.denseScore);
        assertNull(cleared1.sparseScore, "cleared doc must have no sparse score");

        client.deleteCollection(name);
    }

    /**
     * bm25 仅 hnsw/flat（clustered 组合 400）；隐式创建的 none 集合上融合检索 400。
     * EN: bm25 is hnsw/flat-only (a clustered combo is 400); fusion search on an
     * implicitly created none collection is 400.
     */
    @Test
    void sparseCreateGuards() throws Exception {
        String badName = "java_it_sparse_clustered";
        String implicitName = "java_it_sparse_implicit";
        for (String name : List.of(badName, implicitName)) {
            try {
                client.deleteCollection(name);
            } catch (HouyiException e) {
                assertTrue(e.statusCode() == 404 || e.statusCode() == 401
                        || e.statusCode() == 403, "cleanup delete must succeed or be 404/401/403");
            }
        }

        HouyiException bad = assertThrows(HouyiException.class,
                () -> client.createCollection(badName, TEST_DIMS,
                        new CreateCollectionOptions().indexType("clustered").sparse("bm25")));
        assertEquals(400, bad.statusCode(), "clustered+bm25 must be rejected");

        // 隐式创建恒为 none：融合检索被拒（400）。
        // EN: implicit creation is always none, so fusion search is rejected (400).
        client.collection(implicitName)
                .upsert(List.of(new Vector(1, fakeEmbed("implicit none"), Map.of())));
        HouyiException fusionErr = assertThrows(HouyiException.class,
                () -> client.collection(implicitName).fusionSearch(fakeEmbed("implicit none"),
                        List.of(new SparseTerm(1, 1.0)), 1, null));
        assertEquals(400, fusionErr.statusCode());

        client.deleteCollection(implicitName);
    }
}
