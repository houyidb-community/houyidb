/**
 * 融合检索与逐向量稀疏字段的 wire 契约（假 fetch，无服务端）。
 *
 * 覆盖：sparse 键存在性语义（缺省不改 / [] 清空）、融合路由与请求体、
 * 响应平行分数组与 fusion 参数块、旧服务端可选字段缺席的容忍。
 * EN: Wire contracts for fusion search and the per-vector sparse field (fake
 * fetch, no server). Covers: sparse key-presence semantics (absent unchanged /
 * [] clear), the fusion route and request body, the response's parallel score
 * arrays plus fusion params block, and tolerance for optional fields absent
 * from older servers.
 */

import { describe, expect, it } from "vitest";
import { FusionResult, HouyiClient, type Vector } from "../src/index.js";
import { vectorToJson } from "../src/types.js";

interface Captured {
  url: string;
  method: string;
  body: string | undefined;
}

function recordingFetch(body: unknown, records: Captured[]): typeof fetch {
  const impl = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    records.push({
      url,
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? init.body : undefined,
    });
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
  return impl as unknown as typeof fetch;
}

describe("vector sparse field", () => {
  it("distinguishes absent (unchanged) from an empty array (clear)", () => {
    // 缺省键 = 不改，空数组 = 清空：序列化必须保留这个差异。
    // EN: absent key = unchanged, empty array = clear: serialization must
    // preserve the difference.
    const unset: Vector = { id: 1, values: [1.0] };
    expect(vectorToJson(unset)).not.toHaveProperty("sparse");

    const cleared: Vector = { id: 2, values: [1.0], sparse: [] };
    expect(vectorToJson(cleared).sparse).toEqual([]);

    const terms: Vector = {
      id: 3,
      values: [1.0],
      sparse: [
        [3, 1.5],
        [5, 0.25],
      ],
    };
    expect(vectorToJson(terms).sparse).toEqual([
      [3, 1.5],
      [5, 0.25],
    ]);
  });
});

describe("fusionSearch", () => {
  it("POSTs the route and body, then parses parallel scores and fusion echo", async () => {
    const records: Captured[] = [];
    const client = new HouyiClient("http://houyi.test", {
      fetch: recordingFetch(
        {
          ids: [9, 4],
          scores: [0.75, 0.02],
          dense_scores: [0.9, null],
          sparse_scores: [null, 4.2],
          metric: "cosine",
          fusion: { mode: "rrf", rrf_k: 60, alpha: null },
          payloads: { 9: { text: "nine" } },
        },
        records,
      ),
    });

    const result = await client
      .collection("docs")
      .fusionSearch([1.0, 0.0], [[7, 0.5], [8, 0.25]], 2, {
        includePayload: true,
      });

    expect(records).toHaveLength(1);
    expect(records[0]?.method).toBe("POST");
    expect(records[0]?.url).toBe(
      "http://houyi.test/v1/collections/docs/fusion/search",
    );
    expect(JSON.parse(records[0]?.body ?? "{}")).toEqual({
      vector: [1.0, 0.0],
      sparse: [
        [7, 0.5],
        [8, 0.25],
      ],
      k: 2,
      include_payload: true,
    });

    expect(result.fusionMode).toBe("rrf");
    expect(result.rrfK).toBe(60);
    expect(result.alpha).toBeNull();
    expect(result.metric).toBe("cosine");
    expect(result.ids).toEqual([9n, 4n]);
    expect(result.hits[0]?.denseScore).toBe(0.9);
    expect(result.hits[0]?.sparseScore).toBeNull();
    expect(result.hits[1]?.denseScore).toBeNull();
    expect(result.hits[1]?.sparseScore).toBe(4.2);
    expect(result.hits[0]?.payload).toEqual({ text: "nine" });
  });

  it("carries explicit filters and fusion knobs verbatim", async () => {
    const records: Captured[] = [];
    const client = new HouyiClient("http://houyi.test", {
      fetch: recordingFetch(
        {
          ids: [1],
          scores: [1.0],
          dense_scores: [0.8],
          sparse_scores: [3.1],
          metric: "l2",
          fusion: { mode: "weighted", rrf_k: null, alpha: 0.25 },
        },
        records,
      ),
    });

    await client.collection("docs").fusionSearch([2.0, 0.0], [[9, 1.0]], 1, {
      filters: [{ column: "tag", op: "eq", value: "x" }],
      efSearch: 48,
      fusion: "weighted",
      rrfK: 7,
      alpha: 0.25,
    });

    expect(JSON.parse(records[0]?.body ?? "{}")).toEqual({
      vector: [2.0, 0.0],
      sparse: [[9, 1.0]],
      k: 1,
      filters: [{ column: "tag", op: "eq", value: "x" }],
      ef_search: 48,
      fusion: "weighted",
      rrf_k: 7,
      alpha: 0.25,
    });
  });

  it("tolerates responses missing optional arrays and the fusion block", () => {
    // 精简/旧服务端响应：缺 dense_scores/sparse_scores/fusion/payloads 不报错。
    // EN: Trimmed/older-server responses: missing dense_scores/sparse_scores/
    // fusion/payloads do not error.
    const result = FusionResult.fromRawJson(
      JSON.stringify({ ids: [1, 2], scores: [0.3, 0.1] }),
    );
    expect(result.ids).toEqual([1n, 2n]);
    expect(result.hits.every((h) => h.denseScore === null)).toBe(true);
    expect(result.hits.every((h) => h.sparseScore === null)).toBe(true);
    expect(result.hits.every((h) => h.payload === null)).toBe(true);
    expect(result.fusionMode).toBeUndefined();
    expect(result.rrfK).toBeNull();
    expect(result.alpha).toBeNull();
  });

  it("rejects scores that are not parallel to ids", () => {
    // 融合分是排序键，短数组会让命中拿到 undefined 分值（静默错排）；
    // 模态数组缺席仍容忍，因此只对 scores 收紧。
    // EN: Fused scores are the ranking key; a short array would leave hits with
    // undefined scores (silent misordering). Modality arrays stay absent-
    // tolerant, so only scores is tightened.
    expect(() =>
      FusionResult.fromRawJson(JSON.stringify({ ids: [1, 2], scores: [0.3] })),
    ).toThrow(/scores length 1/);
  });
});
