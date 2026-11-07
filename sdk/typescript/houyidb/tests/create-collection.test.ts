import { describe, expect, it } from "vitest";
import { HouyiClient } from "../src/index.js";

/**
 * 显式建集合的 wire 契约：PUT 普通写路径、metric/后端/HNSW 三元组落进 body、
 * 路径段转义，且绝不携带 admin token（创建是写权限，不是管理权限）。
 * EN: Wire contract for explicit creation: a normal PUT write path, the
 * metric/backend/HNSW triple in the body, an escaped path segment, and never
 * the admin token (creation is write permission, not admin).
 */
interface Captured {
  url: string;
  method: string;
  headers: Headers;
  body: string | undefined;
}

function recordingFetch(records: Captured[]): typeof fetch {
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
      headers: new Headers(init?.headers),
      body: typeof init?.body === "string" ? init.body : undefined,
    });
    return new Response(
      JSON.stringify({
        created: true,
        collection: "docs/2026",
        dims: 4,
        metric: "ip",
        index_type: "hnsw",
        hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
      }),
      { status: 201, headers: { "content-type": "application/json" } },
    );
  };
  return impl as unknown as typeof fetch;
}

describe("createCollection", () => {
  it("PUTs metric/backend/HNSW without the admin token", async () => {
    const records: Captured[] = [];
    const client = new HouyiClient("http://houyi.test", {
      adminToken: "admin-token",
      fetch: recordingFetch(records),
    });

    const result = await client.createCollection("docs/2026", 4, {
      metric: "ip",
      indexType: "hnsw",
      hnsw: { m: 16, efConstruction: 100, efSearch: 32 },
    });

    expect(result.metric).toBe("ip");
    // 响应在客户端边界已映射成 camelCase（OCR #32）。
    // EN: The response is mapped to camelCase at the client boundary (OCR #32).
    expect(result.indexType).toBe("hnsw");
    expect(result.hnsw?.m).toBe(16);
    expect(result.hnsw?.efConstruction).toBe(100);
    expect(result.hnsw?.efSearch).toBe(32);
    expect(records).toHaveLength(1);
    expect(records[0]?.method).toBe("PUT");
    expect(records[0]?.url).toBe(
      "http://houyi.test/v1/collections/docs%2F2026",
    );
    expect(records[0]?.headers.get("authorization")).toBeNull();
    expect(JSON.parse(records[0]?.body ?? "{}")).toEqual({
      dims: 4,
      metric: "ip",
      index_type: "hnsw",
      hnsw_m: 16,
      hnsw_ef_construction: 100,
      hnsw_ef_search: 32,
    });
  });

  it("omits optional fields when options are defaulted", async () => {
    const records: Captured[] = [];
    const client = new HouyiClient("http://houyi.test", {
      fetch: recordingFetch(records),
    });

    await client.createCollection("docs", 8);

    expect(JSON.parse(records[0]?.body ?? "{}")).toEqual({ dims: 8 });
  });

  it("does not create an own hnsw key for non-HNSW backends", async () => {
    // 契约「hnsw 仅 hnsw 后端出现」：显式 `hnsw: undefined` 仍会创建自有键，
    // Object.hasOwn 会报告存在；必须用条件展开让键真正缺席（OCR 195e6be #09）。
    // EN: Contract "hnsw only appears for the hnsw backend": an explicit
    // `hnsw: undefined` still creates an own key that Object.hasOwn reports;
    // conditional spread is required for the key to be truly absent
    // (OCR 195e6be #09).
    const fetchImpl = async () =>
      new Response(
        JSON.stringify({
          created: true,
          collection: "flat",
          dims: 4,
          metric: "l2",
          index_type: "flat",
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      );
    const client = new HouyiClient("http://houyi.test", {
      fetch: fetchImpl as unknown as typeof fetch,
    });

    const result = await client.createCollection("flat", 4, {
      indexType: "flat",
      metric: "l2",
    });

    expect(Object.prototype.hasOwnProperty.call(result, "hnsw")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(result, "metric")).toBe(true);
    expect(result.indexType).toBe("flat");
  });
});
