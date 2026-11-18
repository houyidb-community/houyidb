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
        vector_dtype: "f16",
        quantization: "int8",
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
      vectorDtype: "f16",
      quantization: "int8",
    });

    expect(result.metric).toBe("ip");
    // 响应在客户端边界已映射成 camelCase（vector_dtype → vectorDtype）。
    // EN: The response is mapped to camelCase at the client boundary
    // (vector_dtype → vectorDtype).
    expect(result.vectorDtype).toBe("f16");
    expect(result.quantization).toBe("int8");
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
      vector_dtype: "f16",
      quantization: "int8",
      hnsw_m: 16,
      hnsw_ef_construction: 100,
      hnsw_ef_search: 32,
    });
  });

  it("PUTs rabitq quantization verbatim and decodes the echo", async () => {
    // rabitq 是图侧 1 bit/dim 量化：请求体原样携带、响应回显同样解码出来。
    // EN: rabitq is the graph-side 1 bit/dim quantization: the body carries it
    // verbatim and the echoed response decodes back to the same value.
    const records: Captured[] = [];
    const fetchImpl = async (input: RequestInfo | URL, init?: RequestInit) => {
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
          collection: "docs",
          dims: 8,
          metric: "cosine",
          index_type: "hnsw",
          vector_dtype: "f32",
          quantization: "rabitq",
          hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      );
    };
    const client = new HouyiClient("http://houyi.test", {
      fetch: fetchImpl as unknown as typeof fetch,
    });

    const result = await client.createCollection("docs", 8, {
      metric: "cosine",
      indexType: "hnsw",
      quantization: "rabitq",
    });

    expect(result.quantization).toBe("rabitq");
    expect(JSON.parse(records[0]?.body ?? "{}")).toEqual({
      dims: 8,
      metric: "cosine",
      index_type: "hnsw",
      quantization: "rabitq",
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

  it("does not create an own vectorDtype key when the server response omits it", async () => {
    // 旧服务端响应没有 vector_dtype 字段：`vectorDtype: undefined` 会创建自有键，
    // 破坏「字段缺席」的可观测契约；必须用条件展开让键真正缺席。
    // EN: Older server responses omit vector_dtype: `vectorDtype: undefined`
    // would create an own key and break the observable "field is absent"
    // contract; conditional spread is required for the key to be truly absent.
    const fetchImpl = async () =>
      new Response(
        JSON.stringify({
          created: true,
          collection: "legacy",
          dims: 4,
          metric: "l2",
          index_type: "flat",
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      );
    const client = new HouyiClient("http://houyi.test", {
      fetch: fetchImpl as unknown as typeof fetch,
    });

    const result = await client.createCollection("legacy", 4, {
      indexType: "flat",
      metric: "l2",
    });

    expect(Object.prototype.hasOwnProperty.call(result, "vectorDtype")).toBe(
      false,
    );
    expect(result.indexType).toBe("flat");
  });

  it("PUTs the sparse option and decodes the echo", async () => {
    // 稀疏开关是创建期显式选项：请求体原样携带、响应回显同一值。
    // EN: The sparse switch is an explicit create-time option: the body carries
    // it verbatim and the echoed response decodes back to the same value.
    const records: Captured[] = [];
    const fetchImpl = async (input: RequestInfo | URL, init?: RequestInit) => {
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
          collection: "docs",
          dims: 8,
          metric: "cosine",
          index_type: "flat",
          sparse: "bm25",
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      );
    };
    const client = new HouyiClient("http://houyi.test", {
      fetch: fetchImpl as unknown as typeof fetch,
    });

    const result = await client.createCollection("docs", 8, {
      indexType: "flat",
      sparse: "bm25",
    });

    expect(result.sparse).toBe("bm25");
    expect(JSON.parse(records[0]?.body ?? "{}")).toEqual({
      dims: 8,
      index_type: "flat",
      sparse: "bm25",
    });
  });

  it("does not create an own sparse key when the server response omits it", async () => {
    // 旧服务端（P4 之前）响应没有 sparse：条件展开让键真正缺席，不伪造 none。
    // EN: Pre-P4 server responses omit sparse: conditional spread keeps the key
    // truly absent and never fakes "none".
    const fetchImpl = async () =>
      new Response(
        JSON.stringify({
          created: true,
          collection: "legacy",
          dims: 4,
          metric: "l2",
          index_type: "flat",
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      );
    const client = new HouyiClient("http://houyi.test", {
      fetch: fetchImpl as unknown as typeof fetch,
    });

    const result = await client.createCollection("legacy", 4, {
      indexType: "flat",
      metric: "l2",
    });

    expect(Object.prototype.hasOwnProperty.call(result, "sparse")).toBe(false);
  });

  it("does not create an own quantization key when the server response omits it", async () => {
    // 旧服务端响应没有 quantization 字段：必须条件展开让键真正缺席；
    // 同时未设置该选项时 wire body 也不出现该键（旧服务端兼容）。
    // EN: Older server responses omit quantization: conditional spread keeps
    // the key truly absent; unset options also keep the wire body free of the
    // key (old-server compatibility).
    const fetchImpl = async () =>
      new Response(
        JSON.stringify({
          created: true,
          collection: "legacy",
          dims: 4,
          metric: "l2",
          index_type: "flat",
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      );
    const client = new HouyiClient("http://houyi.test", {
      fetch: fetchImpl as unknown as typeof fetch,
    });

    const result = await client.createCollection("legacy", 4, {
      indexType: "flat",
      metric: "l2",
    });

    expect(Object.prototype.hasOwnProperty.call(result, "quantization")).toBe(
      false,
    );
    expect(result.indexType).toBe("flat");
  });
});
