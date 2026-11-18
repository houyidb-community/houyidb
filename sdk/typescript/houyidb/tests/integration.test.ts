/**
 * houyidb TS SDK × 真实服务的集成测试。
 *
 * 前置：houyi-server 监听 127.0.0.1:17000（与 Python 联调同一实例），
 * 可用 HOUYIDB_URL 覆盖。运行：npm test
 *
 * EN: Integration tests for the houyidb TS SDK against a real server.
 *
 * EN: Prerequisite: houyi-server listens on 127.0.0.1:17000 (the same
 * instance used for Python integration), overridable via HOUYIDB_URL.
 * Run: npm test
 */

import { blake2b } from "@noble/hashes/blake2.js";
import { beforeAll, describe, expect, it } from "vitest";
import { HouyiClient, HouyiError, stableId } from "../src/index.js";

const BASE_URL = (process.env.HOUYIDB_URL ?? "http://127.0.0.1:17000").replace(
  /\/+$/,
  "",
);
const COLLECTION = "ts_it";
const DIMS = 16;

/**
 * 确定性伪 embedding（与 Python 联调同款思路，自洽即可，不跨语言对拍）。
 * EN: Deterministic pseudo-embedding (same idea as the Python integration;
 * self-consistency is enough, no cross-language exact comparison).
 */
function fakeEmbed(text: string, dims = DIMS): number[] {
  const out: number[] = [];
  let counter = 0;
  while (out.length < dims) {
    const block = blake2b(new TextEncoder().encode(`${text}:${counter}`), {
      dkLen: 8,
    });
    let v = 0n;
    for (const b of block) v = (v << 8n) | BigInt(b);
    out.push((Number(v) / 2 ** 64) * 2 - 1);
    counter += 1;
  }
  return out.slice(0, dims);
}

/**
 * 等待异步构图队列排空（双速写入：flush 后才对检索可见）。
 * EN: Wait for the async graph-building queue to drain (dual-speed writes:
 * visible to search only after flush).
 */
async function waitFlushed(client: HouyiClient, name: string, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const status = await client.collection(name).status();
    if (status.pending_count === 0) return;
    await new Promise((r) => setTimeout(r, 20));
  }
  throw new Error(`collection ${name} flush timeout`);
}

const client = new HouyiClient(BASE_URL);

beforeAll(async () => {
  if (!(await client.health())) {
    throw new Error(
      `houyi-server 不可达（${BASE_URL}）：先启动 ` +
        `./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000`,
    );
  }
});

describe("integration", () => {
  it("upsert + search round trip with payload", async () => {
    const col = client.collection(COLLECTION);
    const texts = ["apple is a fruit", "banana is yellow", "laptop is electronics"];
    const inserted = await col.upsert(
      texts.map((t, i) => ({
        id: stableId(`doc-${i}`),
        values: fakeEmbed(t),
        payload: { text: t, category: i < 2 ? "fruit" : "electronics", price: [5, 3, 999][i] },
      })),
    );
    expect(inserted).toBe(3);
    await waitFlushed(client, COLLECTION);

    const result = await col.search(fakeEmbed("apple is a fruit"), 2, {
      includePayload: true,
    });
    expect(result.hits.length).toBe(2);
    expect(result.hits[0].payload?.text).toBe("apple is a fruit");
    expect(result.scores[0]).toBeGreaterThanOrEqual(result.scores[1]);
  });

  it("hybrid search with intersected filters", async () => {
    const col = client.collection(COLLECTION);
    const result = await col.hybridSearch(fakeEmbed("apple is a fruit"), 3, {
      filters: [
        { column: "category", op: "eq", value: "fruit" },
        { column: "price", op: "lt", value: 4 },
      ],
      includePayload: true,
    });
    expect(result.hits.map((h) => h.payload?.text)).toEqual(["banana is yellow"]);
  });

  it("keeps >2^53 ids exact end to end", async () => {
    // stableId 落点常态是 19 位十进制，超 Number.MAX_SAFE_INTEGER——
    // 全链路（请求序列化 + 响应解析）都必须 bigint 无损。
    // EN: stableId output is typically 19 decimal digits, beyond
    // Number.MAX_SAFE_INTEGER—the whole path (request serialization +
    // response parsing) must be bigint-lossless.
    const col = client.collection(COLLECTION);
    const id = stableId("hello"); // 2861735919082615933n
    await col.upsert([{ id, values: fakeEmbed("bigint probe"), payload: { tag: "big" } }]);
    await waitFlushed(client, COLLECTION);
    const result = await col.hybridSearch(fakeEmbed("bigint probe"), 1, {
      filters: [{ column: "tag", op: "eq", value: "big" }],
      includePayload: true,
    });
    expect(result.hits[0].id).toBe(id);
    expect(result.hits[0].payload?.tag).toBe("big");
    await col.delete({ ids: [id] });
  });

  it("delete by ids and by filters", async () => {
    const col = client.collection(COLLECTION);
    const probeId = stableId("delete-me");
    await col.upsert([{ id: probeId, values: fakeEmbed("delete probe"), payload: { tag: "del" } }]);
    // delete-by-filter 不依赖构图排空（标量页同步落页）。
    // EN: delete-by-filter does not depend on graph-building drain (scalar
    // pages are written synchronously).
    expect(await col.delete({ filters: [{ column: "tag", op: "eq", value: "del" }] })).toBe(1);
    expect(await col.delete({ ids: [probeId] })).toBe(0); // 幂等 EN: idempotent
  });

  it("surfaces server errors as HouyiError", async () => {
    const col = client.collection("ts_it_missing_collection");
    await expect(col.status()).rejects.toMatchObject({
      name: "HouyiError",
      statusCode: 404,
    });
    await expect(col.search(fakeEmbed("x"), 1)).rejects.toBeInstanceOf(HouyiError);
  });

  it("delete requires ids or filters", async () => {
    await expect(client.collection(COLLECTION).delete({})).rejects.toThrow(
      /ids or filters/,
    );
  });

  it("drops a collection and recreates it empty", async () => {
    const name = "ts_it_admin_drop";
    const col = client.collection(name);
    await col.upsert([{ id: 7, values: fakeEmbed("drop me"), payload: { tag: "drop" } }]);

    await client.deleteCollection(name);

    await expect(col.status()).rejects.toMatchObject({
      name: "HouyiError",
      statusCode: 404,
    });

    // 同名重建：首次写入隐式创建，旧向量不得复活。
    // EN: recreate by name: the first write implicitly creates it and the old
    // vectors must not reappear.
    await col.upsert([{ id: 8, values: fakeEmbed("fresh") }]);
    await waitFlushed(client, name);
    expect((await col.status()).indexed_count).toBe(1);
  });

  it("returns 404 when dropping an unknown collection", async () => {
    await expect(
      client.deleteCollection("ts_it_admin_never_created"),
    ).rejects.toMatchObject({ name: "HouyiError", statusCode: 404 });
  });

  it("sparse create + fusion search round trip", async () => {
    const name = "ts_it_fusion";
    try {
      await client.deleteCollection(name);
    } catch (err) {
      // 清理只容忍 404（从未创建）；其他错误让用例失败。
      // EN: Cleanup tolerates only 404 (never created); anything else fails.
      expect(err).toMatchObject({ name: "HouyiError", statusCode: 404 });
    }
    const created = await client.createCollection(name, DIMS, {
      sparse: "bm25",
    });
    expect(created.sparse).toBe("bm25");

    const col = client.collection(name);
    await col.upsert([
      {
        id: 1,
        values: fakeEmbed("apple fruit"),
        payload: { text: "apple fruit" },
        sparse: [
          [11, 1.0],
          [12, 0.5],
        ],
      },
      {
        id: 2,
        values: fakeEmbed("banana yellow"),
        payload: { text: "banana yellow" },
        sparse: [[21, 1.0]],
      },
    ]);
    await waitFlushed(client, name);

    const rrf = await col.fusionSearch(fakeEmbed("apple fruit"), [[11, 1.0]], 2, {
      includePayload: true,
    });
    expect(rrf.fusionMode).toBe("rrf");
    expect(rrf.rrfK).toBe(60);
    expect(rrf.alpha).toBeNull();
    expect(rrf.ids[0]).toBe(1n);
    expect(rrf.hits[0]?.denseScore).not.toBeNull();
    expect(rrf.hits[0]?.sparseScore).not.toBeNull();
    expect(rrf.hits[0]?.payload?.text).toBe("apple fruit");

    const weighted = await col.fusionSearch(
      fakeEmbed("apple fruit"),
      [[11, 1.0]],
      2,
      { fusion: "weighted", alpha: 0.5 },
    );
    expect(weighted.fusionMode).toBe("weighted");
    expect(weighted.alpha).toBe(0.5);
    expect(weighted.rrfK).toBeNull();
    expect(weighted.ids[0]).toBe(1n);

    // 空数组清空该文档的稀疏词项：稠密一路仍命中，但无 BM25 分。
    // EN: An empty array clears the document's sparse terms: the dense path
    // still hits it, but there is no BM25 score.
    await col.upsert([
      { id: 1, values: fakeEmbed("apple fruit"), sparse: [] },
    ]);
    await waitFlushed(client, name);
    const cleared = await col.fusionSearch(fakeEmbed("apple fruit"), [[11, 1.0]], 2);
    const hitOne = cleared.hits.find((h) => h.id === 1n);
    expect(hitOne?.denseScore).not.toBeNull();
    expect(hitOne?.sparseScore).toBeNull();

    await client.deleteCollection(name);
  });
});
