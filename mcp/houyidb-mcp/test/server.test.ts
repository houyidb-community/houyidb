import { afterEach, describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { loadConfig, type HouyidbMcpConfig } from "../src/config.js";
import { MAX_ERROR_BODY_CHARS } from "../src/constants.js";
import type { FetchLike, FetchResponseLike } from "../src/houyi-client.js";
import { createHouyidbServer } from "../src/server.js";

interface CapturedRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: string | undefined;
}

interface MockFetch {
  fetchImpl: FetchLike;
  calls: CapturedRequest[];
}

function createMockFetch(
  handler: (
    request: CapturedRequest,
  ) => FetchResponseLike | Promise<FetchResponseLike>,
): MockFetch {
  const calls: CapturedRequest[] = [];
  const fetchImpl: FetchLike = async (url, init = {}) => {
    const request: CapturedRequest = {
      url,
      method: init.method ?? "GET",
      headers: init.headers ?? {},
      body: init.body,
    };
    calls.push(request);
    return handler(request);
  };
  return { fetchImpl, calls };
}

function jsonResponse(body: unknown, status = 200): FetchResponseLike {
  return {
    status,
    text: async () => JSON.stringify(body),
  };
}

function textResponse(text: string, status = 200): FetchResponseLike {
  return {
    status,
    text: async () => text,
  };
}

function makeConfig(
  overrides: Partial<HouyidbMcpConfig> = {},
): HouyidbMcpConfig {
  return {
    baseUrl: "http://houyi.test:17000",
    readOnly: true,
    collections: new Set<string>(),
    ...overrides,
  };
}

const closers: Array<() => Promise<void>> = [];

afterEach(async () => {
  while (closers.length > 0) {
    const close = closers.pop();
    if (close !== undefined) await close();
  }
});

async function connect(
  config: HouyidbMcpConfig,
  fetchImpl: FetchLike,
): Promise<Client> {
  const server = createHouyidbServer(config, { fetchImpl });
  const client = new Client({ name: "houyidb-mcp-test", version: "0.0.0" });
  const [clientTransport, serverTransport] =
    InMemoryTransport.createLinkedPair();
  await Promise.all([
    server.connect(serverTransport),
    client.connect(clientTransport),
  ]);
  closers.push(async () => {
    await Promise.allSettled([client.close(), server.close()]);
  });
  return client;
}

interface TextResult {
  isError?: boolean;
  text: string;
}

async function callText(
  client: Client,
  name: string,
  args: Record<string, unknown>,
): Promise<TextResult> {
  const result = (await client.callTool({
    name,
    arguments: args,
  })) as {
    isError?: boolean;
    content?: Array<{ type?: string; text?: string }>;
  };
  const text = (result.content ?? [])
    .filter((item) => item.type === "text")
    .map((item) => item.text ?? "")
    .join("\n");
  return { ...(result.isError === undefined ? {} : { isError: result.isError }), text };
}

async function toolNames(client: Client): Promise<string[]> {
  const { tools } = await client.listTools();
  return tools.map((tool) => tool.name).sort();
}

describe("configuration", () => {
  it("uses safe defaults", () => {
    const config = loadConfig({});
    expect(config.baseUrl).toBe("http://127.0.0.1:17000");
    expect(config.authToken).toBeUndefined();
    expect(config.adminToken).toBeUndefined();
    expect(config.readOnly).toBe(true);
    expect(config.collections.size).toBe(0);
  });

  it("parses write mode, token, URL, and allowlist", () => {
    const config = loadConfig({
      HOUYIDB_URL: "http://houyi.local:18000///",
      HOUYI_AUTH_TOKEN: "  secret-token  ",
      HOUYI_ADMIN_TOKEN: "  admin-token  ",
      HOUYIDB_MCP_READ_ONLY: "0",
      HOUYIDB_MCP_COLLECTIONS: " docs, notes ,, docs ",
    });
    expect(config.baseUrl).toBe("http://houyi.local:18000");
    expect(config.authToken).toBe("secret-token");
    expect(config.adminToken).toBe("admin-token");
    expect(config.readOnly).toBe(false);
    expect([...config.collections]).toEqual(["docs", "notes"]);
  });

  it("rejects ambiguous read-only values instead of enabling writes", () => {
    expect(() => loadConfig({ HOUYIDB_MCP_READ_ONLY: "maybe" })).toThrow(
      /HOUYIDB_MCP_READ_ONLY/,
    );
  });

  it("rejects credentials embedded in the base URL", () => {
    expect(() =>
      loadConfig({ HOUYIDB_URL: "http://user:pass@houyi.local:17000" }),
    ).toThrow(/must not embed credentials/);
  });
});

describe("tool registration", () => {
  it("registers only read tools in read-only mode", async () => {
    const mock = createMockFetch(() => jsonResponse([]));
    const client = await connect(makeConfig(), mock.fetchImpl);
    expect(await toolNames(client)).toEqual([
      "collection_status",
      "fusion_search",
      "hybrid_search",
      "list_collections",
      "search",
    ]);
    expect(mock.calls).toHaveLength(0);
  });

  it("registers write tools when read-only is disabled", async () => {
    const mock = createMockFetch(() => jsonResponse([]));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );
    expect(await toolNames(client)).toEqual([
      "collection_status",
      "create_collection",
      "delete",
      "delete_collection",
      "fusion_search",
      "hybrid_search",
      "list_collections",
      "search",
      "upsert",
    ]);
  });
});

describe("collection allowlist", () => {
  it("filters list_collections by the allowlist", async () => {
    const mock = createMockFetch(() =>
      jsonResponse([
        { name: "docs", vectors: 10 },
        { name: "secret", vectors: 20 },
      ]),
    );
    const client = await connect(
      makeConfig({ collections: new Set(["docs"]) }),
      mock.fetchImpl,
    );
    const { isError, text } = await callText(client, "list_collections", {});
    expect(isError).not.toBe(true);
    expect(JSON.parse(text)).toEqual([{ name: "docs", vectors: 10 }]);
  });

  it("blocks every collection-scoped tool before making a request", async () => {
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({
        readOnly: false,
        collections: new Set(["docs"]),
      }),
      mock.fetchImpl,
    );

    const calls: Array<[string, Record<string, unknown>]> = [
      ["collection_status", { collection: "secret" }],
      ["search", { collection: "secret", vector: [1, 0], k: 1 }],
      [
        "hybrid_search",
        { collection: "secret", vector: [1, 0], k: 1, filters: [] },
      ],
      [
        "fusion_search",
        {
          collection: "secret",
          vector: [1, 0],
          sparse: [[1, 1]],
          k: 1,
        },
      ],
      [
        "upsert",
        { collection: "secret", vectors: [{ id: 1, values: [1, 0] }] },
      ],
      ["delete", { collection: "secret", ids: [1] }],
      ["delete_collection", { collection: "secret" }],
      ["create_collection", { collection: "secret", dims: 2 }],
    ];

    for (const [name, args] of calls) {
      const { isError, text } = await callText(client, name, args);
      expect(isError, name).toBe(true);
      expect(text, name).toContain("HOUYIDB_MCP_COLLECTIONS");
    }
    expect(mock.calls).toHaveLength(0);
  });
});

describe("search request mapping", () => {
  it("sends search options to the collection search endpoint", async () => {
    const mock = createMockFetch(() =>
      jsonResponse({
        ids: [7],
        scores: [0.99],
        payloads: { "7": { label: "doc-7", kind: "guide" } },
      }),
    );
    const client = await connect(
      makeConfig({
        collections: new Set(["docs"]),
        authToken: "top-secret",
      }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "search", {
      collection: "docs",
      vector: [0.25, -0.5, 1],
      k: 3,
      include_payload: true,
      ef_search: 128,
    });

    expect(isError).not.toBe(true);
    expect(JSON.parse(text)).toEqual({
      ids: [7],
      scores: [0.99],
      payloads: { "7": { label: "doc-7", kind: "guide" } },
    });
    expect(mock.calls).toHaveLength(1);
    const request = mock.calls[0];
    expect(request?.url).toBe(
      "http://houyi.test:17000/v1/collections/docs/search",
    );
    expect(request?.method).toBe("POST");
    expect(request?.headers.Authorization).toBe("Bearer top-secret");
    expect(request?.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(request?.body ?? "{}")).toEqual({
      vector: [0.25, -0.5, 1],
      k: 3,
      include_payload: true,
      ef_search: 128,
    });
  });

  it("defaults include_payload to false and omits ef_search", async () => {
    const mock = createMockFetch(() =>
      jsonResponse({ ids: [], scores: [] }),
    );
    const client = await connect(makeConfig(), mock.fetchImpl);

    await callText(client, "search", {
      collection: "docs",
      vector: [1, 2],
      k: 2,
    });

    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      vector: [1, 2],
      k: 2,
      include_payload: false,
    });
  });

  it("maps hybrid filters to the hybrid endpoint", async () => {
    const mock = createMockFetch(() =>
      jsonResponse({ ids: [1], scores: [0.5] }),
    );
    const client = await connect(makeConfig(), mock.fetchImpl);
    const filters = [
      { column: "category", op: "eq", value: "fruit" },
      { column: "year", op: "in", value: [2025, 2026] },
    ];

    await callText(client, "hybrid_search", {
      collection: "docs",
      vector: [1, 0],
      k: 5,
      filters,
      include_payload: true,
    });

    expect(mock.calls[0]?.url).toBe(
      "http://houyi.test:17000/v1/collections/docs/hybrid/search",
    );
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      vector: [1, 0],
      k: 5,
      filters,
      include_payload: true,
    });
  });

  it("maps fusion search to the fusion endpoint with defaults", async () => {
    // 缺省融合模式在本地显式补成 rrf；rrf_k/alpha 未给出时不转发，
    // 由服务端补齐缺省（60 / 0.5）。
    // EN: The fusion mode default is filled locally as rrf; rrf_k/alpha are
    // omitted when not supplied and defaulted server-side (60 / 0.5).
    const mock = createMockFetch(() =>
      jsonResponse({
        ids: [9],
        scores: [0.5],
        dense_scores: [0.9],
        sparse_scores: [2.1],
        metric: "cosine",
        fusion: { mode: "rrf", rrf_k: 60, alpha: null },
      }),
    );
    const client = await connect(makeConfig(), mock.fetchImpl);

    const { isError } = await callText(client, "fusion_search", {
      collection: "docs",
      vector: [1, 0],
      sparse: [
        [1, 0.9],
        [2, 0.4],
      ],
      k: 4,
      include_payload: true,
      ef_search: 64,
    });

    expect(isError).not.toBe(true);
    expect(mock.calls[0]?.url).toBe(
      "http://houyi.test:17000/v1/collections/docs/fusion/search",
    );
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      vector: [1, 0],
      sparse: [
        [1, 0.9],
        [2, 0.4],
      ],
      k: 4,
      filters: [],
      include_payload: true,
      ef_search: 64,
      fusion: "rrf",
    });
  });

  it("forwards weighted fusion knobs and rrf_k verbatim", async () => {
    // 显式调参值原样转发；本地不替调用方补另一模式的旋钮，
    // 也不改写信使值。
    // EN: Explicit tuning values are forwarded verbatim; the MCP layer never
    // fills the knob of the other mode nor rewrites the mode.
    const mock = createMockFetch(() =>
      jsonResponse({ ids: [], scores: [] }),
    );
    const client = await connect(makeConfig(), mock.fetchImpl);

    await callText(client, "fusion_search", {
      collection: "docs",
      vector: [1, 0],
      sparse: [[1, 0.9]],
      k: 2,
      fusion: "weighted",
      alpha: 0.3,
      rrf_k: 80,
    });

    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      vector: [1, 0],
      sparse: [[1, 0.9]],
      k: 2,
      filters: [],
      include_payload: false,
      fusion: "weighted",
      alpha: 0.3,
      rrf_k: 80,
    });
  });

  it("rejects out-of-range sparse terms before calling upstream", async () => {
    // term 边界与权重正性在 schema 层就该失败，不得把必然 400 的
    // 请求转发到服务端。
    // EN: Term bounds and weight positivity must fail in the schema layer;
    // requests the server would 400 must not be forwarded.
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(makeConfig(), mock.fetchImpl);

    const { isError, text } = await callText(client, "fusion_search", {
      collection: "docs",
      vector: [1, 0],
      sparse: [[0, 1]],
      k: 1,
    });
    expect(isError).toBe(true);
    expect(text.length).toBeGreaterThan(0);
    expect(mock.calls).toHaveLength(0);
  });

  it("encodes collection names used as path segments", async () => {
    const mock = createMockFetch(() => jsonResponse({ name: "docs/2026" }));
    const client = await connect(makeConfig(), mock.fetchImpl);

    await callText(client, "collection_status", {
      collection: "docs/2026",
    });

    expect(mock.calls[0]?.url).toBe(
      "http://houyi.test:17000/v1/collections/docs%2F2026/status",
    );
  });
});

describe("write request mapping", () => {
  it("maps create_collection to PUT with the normal token only", async () => {
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs/2026",
        dims: 4,
        metric: "ip",
        index_type: "hnsw",
        vector_dtype: "f16",
        quantization: "int8",
        hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
      }),
    );
    const client = await connect(
      makeConfig({
        readOnly: false,
        authToken: "write-token",
        adminToken: "admin-token",
      }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "create_collection", {
      collection: "docs/2026",
      dims: 4,
      metric: "ip",
      index_type: "hnsw",
      vector_dtype: "f16",
      quantization: "int8",
      hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
    });

    expect(isError).not.toBe(true);
    expect(JSON.parse(text)).toEqual({
      created: true,
      collection: "docs/2026",
      dims: 4,
      metric: "ip",
      index_type: "hnsw",
      vector_dtype: "f16",
      quantization: "int8",
      hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
    });
    const request = mock.calls[0];
    expect(request?.url).toBe(
      "http://houyi.test:17000/v1/collections/docs%2F2026",
    );
    expect(request?.method).toBe("PUT");
    // 显式创建是普通写路径：只带 auth token，绝不外泄 admin token。
    // EN: explicit creation is a normal write path: only the auth token, never
    // the admin token.
    expect(request?.headers.Authorization).toBe("Bearer write-token");
    expect(JSON.parse(request?.body ?? "{}")).toEqual({
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

  it("rejects f16 on a non-hnsw backend without calling upstream", async () => {
    // f16 只属于 hnsw 后端；显式非 hnsw 后端时本地先拒绝（服务端同口径）。
    // EN: f16 belongs to the hnsw backend only; an explicit non-hnsw backend is
    // rejected locally (the same rule as the server).
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "flat",
      vector_dtype: "f16",
    });

    expect(isError).toBe(true);
    expect(text).toContain("f16");
    expect(mock.calls.length).toBe(0);
  });

  it("rejects int8 on a non-hnsw backend without calling upstream", async () => {
    // int8 只属于 hnsw 后端；显式非 hnsw 后端时本地先拒绝（服务端同口径）。
    // EN: int8 belongs to the hnsw backend only; an explicit non-hnsw backend
    // is rejected locally (the same rule as the server).
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "flat",
      quantization: "int8",
    });

    expect(isError).toBe(true);
    expect(text).toContain("int8");
    expect(mock.calls.length).toBe(0);
  });

  it("rejects a mixed-case int8 alias on a non-hnsw backend without calling upstream", async () => {
    // 服务端量化解析去空白且大小写不敏感；本地预拒绝必须识别同一归一形式，
    // 否则会把服务端会 400 的组合静默转发（本地校验不得更宽于服务端语义）。
    // EN: The server parses quantization with trim + case-insensitive matching;
    // local pre-rejection must recognize the same normalized form or a
    // combination the server would 400 gets forwarded silently.
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "flat",
      quantization: "  INT8  ",
    });

    expect(isError).toBe(true);
    expect(text.toLowerCase()).toContain("int8");
    expect(mock.calls.length).toBe(0);
  });

  it("rejects rabitq on a non-hnsw backend without calling upstream", async () => {
    // rabitq 与 int8 一样只属于 hnsw 后端；显式非 hnsw 后端时本地先拒绝。
    // EN: rabitq, like int8, belongs to the hnsw backend only; an explicit
    // non-hnsw backend is rejected locally.
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 8,
      index_type: "flat",
      quantization: "rabitq",
    });

    expect(isError).toBe(true);
    expect(text.toLowerCase()).toContain("rabitq");
    expect(mock.calls.length).toBe(0);
  });

  it("rejects rabitq with an ip/l1 metric without calling upstream", async () => {
    // rabitq 仅支持 l2|cosine；ip/l1 在本地先拒绝（服务端同口径 400）。
    // metric 入参是闭合枚举，别名在 schema 层就被拒，不会到达此分支。
    // EN: rabitq supports l2|cosine only; ip/l1 are rejected locally (the
    // server returns the same 400). The metric input is a closed enum, so
    // aliases are rejected by the schema before this branch is reached.
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    for (const metric of ["ip", "l1"]) {
      const { isError, text } = await callText(client, "create_collection", {
        collection: "docs",
        dims: 8,
        metric,
        index_type: "hnsw",
        quantization: "rabitq",
      });
      expect(isError).toBe(true);
      expect(text).toContain("l2|cosine");
    }
    expect(mock.calls.length).toBe(0);
  });

  it("forwards rabitq with cosine", async () => {
    // cosine 合法：原样转发，不本地改写。
    // EN: cosine is valid: forward verbatim without local rewriting.
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 8,
        metric: "cosine",
        index_type: "hnsw",
        quantization: "rabitq",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 8,
      metric: "cosine",
      index_type: "hnsw",
      quantization: "rabitq",
    });
    expect(isError).not.toBe(true);
    expect(mock.calls.length).toBe(1);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 8,
      metric: "cosine",
      index_type: "hnsw",
      quantization: "rabitq",
    });
  });

  it("forwards a mixed-case quantization alias verbatim", async () => {
    // 归一化只用于本地判断；转发用原始值（服务端自己 trim/小写化），
    // 与 vector_dtype 的转发契约一致，本地不替用户改写请求体。
    // EN: Normalization is only for the local check; forwarding keeps the raw
    // value (the server trims/lowercases itself), matching the vector_dtype
    // forwarding contract — the MCP layer never rewrites the request body.
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 4,
        metric: "cosine",
        index_type: "hnsw",
        quantization: "int8",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "hnsw",
      quantization: " INT8 ",
    });

    expect(isError).not.toBe(true);
    expect(mock.calls.length).toBe(1);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 4,
      index_type: "hnsw",
      quantization: " INT8 ",
    });
  });

  it("forwards int8 without inferring index_type", async () => {
    // 省略 index_type 时不得推断后端、不得本地拒绝：服务端按部署的
    // default_index_type 解析（默认非 hnsw 时自行 400）。
    // EN: An omitted index_type must be neither inferred nor rejected locally:
    // the server resolves the deployment's default_index_type (its own 400 when
    // that default is not hnsw).
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 4,
        metric: "cosine",
        index_type: "hnsw",
        quantization: "int8",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      quantization: "int8",
    });

    expect(isError).not.toBe(true);
    expect(mock.calls.length).toBe(1);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 4,
      quantization: "int8",
    });
  });

  it("forwards an explicit none quantization on a non-hnsw backend", async () => {
    // 显式 "none" 对任何后端都合法：本地不得误拒（只有 int8/rabitq 需要 hnsw）。
    // EN: An explicit "none" is valid for any backend: local validation must
    // not reject it (only int8/rabitq require hnsw).
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 4,
        metric: "l2",
        index_type: "flat",
        quantization: "none",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "flat",
      metric: "l2",
      quantization: "none",
    });

    expect(isError).not.toBe(true);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 4,
      index_type: "flat",
      metric: "l2",
      quantization: "none",
    });
  });

  it("forwards f16 without inferring index_type", async () => {
    // 省略 index_type 时不得推断后端、也不得本地拒绝：服务端按部署的
    // default_index_type 解析（默认非 hnsw 时自行 400），与 hnsw 参数同纪律。
    // EN: An omitted index_type must be neither inferred nor rejected locally:
    // the server resolves the deployment's default_index_type (its own 400 when
    // that default is not hnsw), the same discipline as hnsw parameters.
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 4,
        metric: "cosine",
        index_type: "hnsw",
        vector_dtype: "f16",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      vector_dtype: "f16",
    });

    expect(isError).not.toBe(true);
    expect(mock.calls.length).toBe(1);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 4,
      vector_dtype: "f16",
    });
  });

  it("recognizes f16 aliases case-insensitively for the backend gate", async () => {
    // 服务端接受 f16 的别名且大小写不敏感；本地预拒绝必须同口径识别，
    // 不得因为 "HALF"/"F16" 这类合法别名可编译通过而漏拦。
    // EN: The server accepts f16 aliases case-insensitively; local
    // pre-rejection must recognize the same set, so a legal alias such as
    // "HALF"/"F16" cannot slip through the hnsw-only gate.
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 4,
        metric: "cosine",
        index_type: "hnsw",
        vector_dtype: "half",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const ok = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "hnsw",
      vector_dtype: "HALF",
    });
    expect(ok.isError).not.toBe(true);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 4,
      index_type: "hnsw",
      vector_dtype: "HALF",
    });

    const rejected = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "flat",
      vector_dtype: "half",
    });
    expect(rejected.isError).toBe(true);
    expect(rejected.text).toContain('index_type "hnsw"');
    expect(mock.calls.length).toBe(1);
  });

  it("rejects hnsw parameters on a non-hnsw backend without calling upstream", async () => {
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "pq-flat",
      hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
    });

    expect(isError).toBe(true);
    expect(text).toContain("hnsw");
    expect(mock.calls.length).toBe(0);
  });

  it("forwards hnsw parameters without inferring index_type", async () => {
    // 省略 index_type 时不得推断后端、也不得本地拒绝：服务端按部署的
    // default_index_type 解析（默认非 hnsw 时自行 400）。本地校验必须与
    // 服务端契约同宽（OCR 195e6be #02、cf1d19f #00）。
    // EN: An omitted index_type must be neither inferred nor rejected locally:
    // the server resolves the deployment's default_index_type (and returns its
    // own 400 when that default is not hnsw). Local validation must be no
    // stricter than the server contract (OCR 195e6be #02, cf1d19f #00).
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 4,
        metric: "cosine",
        index_type: "hnsw",
        hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      hnsw: { m: 16, ef_construction: 100, ef_search: 32 },
    });

    expect(isError).not.toBe(true);
    expect(mock.calls.length).toBe(1);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 4,
      hnsw_m: 16,
      hnsw_ef_construction: 100,
      hnsw_ef_search: 32,
    });
  });

  it("rejects a metric the backend cannot use without calling upstream", async () => {
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      metric: "ip",
      index_type: "hii-l0",
    });

    expect(isError).toBe(true);
    expect(text).toContain("cosine");
    expect(mock.calls.length).toBe(0);
  });

  it("fills the backend-mandated metric when omitted", async () => {
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 4,
        metric: "l2",
        index_type: "pq-flat",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    await callText(client, "create_collection", {
      collection: "docs",
      dims: 4,
      index_type: "pq-flat",
    });

    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 4,
      metric: "l2",
      index_type: "pq-flat",
    });
  });

  it("forwards sparse bm25 on flat and rejects it on other backends", async () => {
    // bm25 仅 hnsw|flat：显式其它后端本地先拒绝（与服务端 400 同口径），
    // 显式 flat 则原样转发 sparse 字段供服务端回显。
    // EN: bm25 is hnsw|flat-only: another explicit backend is rejected locally
    // (the server would 400); an explicit flat forwards the sparse field
    // verbatim for the server to echo.
    const mock = createMockFetch(() =>
      jsonResponse({
        created: true,
        collection: "docs",
        dims: 8,
        metric: "l2",
        index_type: "flat",
        sparse: "bm25",
      }),
    );
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const ok = await callText(client, "create_collection", {
      collection: "docs",
      dims: 8,
      index_type: "flat",
      sparse: "bm25",
    });
    expect(ok.isError).not.toBe(true);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      dims: 8,
      index_type: "flat",
      sparse: "bm25",
    });

    const rejected = await callText(client, "create_collection", {
      collection: "docs",
      dims: 8,
      index_type: "pq-flat",
      sparse: "bm25",
    });
    expect(rejected.isError).toBe(true);
    expect(rejected.text).toContain('index_type "hnsw"|"flat"');
    expect(mock.calls).toHaveLength(1);
  });

  it("maps upsert vectors to the upsert endpoint", async () => {
    const mock = createMockFetch(() => jsonResponse({ inserted: 1 }));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );
    const vectors = [
      {
        id: 42,
        values: [0.1, 0.2],
        label: "doc",
        payload: { category: "fruit" },
      },
    ];

    const { isError } = await callText(client, "upsert", {
      collection: "docs",
      vectors,
    });

    expect(isError).not.toBe(true);
    expect(mock.calls[0]?.url).toBe(
      "http://houyi.test:17000/v1/collections/docs/upsert",
    );
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({ vectors });
  });

  it("forwards per-vector sparse terms including the empty clear list", async () => {
    // sparse 键缺省 = 不改、[] = 清空；MCP 必须原样保留这一区别，
    // 不得把空数组剔除或补默认。
    // EN: An absent sparse key means unchanged while [] clears; the MCP layer
    // must preserve that distinction and neither strip empty arrays nor
    // inject a default.
    const mock = createMockFetch(() => jsonResponse({ inserted: 2 }));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );
    const vectors = [
      {
        id: 1,
        values: [1, 0],
        sparse: [
          [3, 1.2],
          [5, 0.4],
        ],
      },
      { id: 2, values: [0, 1], sparse: [] },
    ];

    const { isError } = await callText(client, "upsert", {
      collection: "docs",
      vectors,
    });

    expect(isError).not.toBe(true);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({ vectors });
  });

  it("forwards an explicit null sparse as-is (null means unchanged)", async () => {
    // 显式 null 与缺省同义（服务端视作不改），schema 必须放行且不得改写为
    // [] 或剔除键——否则“清空”语义会被 schema 层悄悄偷换。
    // EN: An explicit null means the same as absent (the server treats it as
    // unchanged); the schema must accept it and neither rewrite it to [] nor
    // drop the key, which would silently swap the "clear" semantics.
    const mock = createMockFetch(() => jsonResponse({ inserted: 1 }));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );
    const vectors = [{ id: 1, values: [1, 0], sparse: null }];

    const { isError } = await callText(client, "upsert", {
      collection: "docs",
      vectors,
    });

    expect(isError).not.toBe(true);
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({ vectors });
  });

  it("maps delete ids and filters, and rejects an empty delete", async () => {
    const mock = createMockFetch(() => jsonResponse({ deleted: 2 }));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );
    const filters = [{ column: "category", op: "eq", value: "fruit" }];

    await callText(client, "delete", {
      collection: "docs",
      ids: [1, 2],
      filters,
    });
    expect(mock.calls[0]?.url).toBe(
      "http://houyi.test:17000/v1/collections/docs/delete",
    );
    expect(JSON.parse(mock.calls[0]?.body ?? "{}")).toEqual({
      ids: [1, 2],
      filters,
    });

    const empty = await callText(client, "delete", { collection: "docs" });
    expect(empty.isError).toBe(true);
    expect(empty.text).toContain("at least one id or filter");
    expect(mock.calls).toHaveLength(1);
  });
});

describe("collection delete mapping", () => {
  it("issues an admin DELETE with the admin token only", async () => {
    const mock = createMockFetch(() =>
      jsonResponse({ deleted: true, collection: "docs" }),
    );
    const client = await connect(
      makeConfig({
        readOnly: false,
        authToken: "read-token",
        adminToken: "admin-token",
      }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "delete_collection", {
      collection: "docs",
    });

    expect(isError).not.toBe(true);
    expect(JSON.parse(text)).toEqual({ deleted: true, collection: "docs" });
    expect(mock.calls).toHaveLength(1);
    const request = mock.calls[0];
    expect(request?.url).toBe(
      "http://houyi.test:17000/v1/admin/collections/docs",
    );
    expect(request?.method).toBe("DELETE");
    expect(request?.headers.Authorization).toBe("Bearer admin-token");
    expect(request?.body).toBeUndefined();
  });

  it("rejects when HOUYI_ADMIN_TOKEN is not configured", async () => {
    const mock = createMockFetch(() => jsonResponse({}));
    const client = await connect(
      makeConfig({ readOnly: false }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "delete_collection", {
      collection: "docs",
    });

    expect(isError).toBe(true);
    expect(text).toContain("HOUYI_ADMIN_TOKEN");
    expect(mock.calls).toHaveLength(0);
  });

  it("encodes the collection name in the admin path", async () => {
    const mock = createMockFetch(() => jsonResponse({ deleted: true }));
    const client = await connect(
      makeConfig({ readOnly: false, adminToken: "admin-token" }),
      mock.fetchImpl,
    );

    await callText(client, "delete_collection", {
      collection: "docs/2026",
    });

    expect(mock.calls[0]?.url).toBe(
      "http://houyi.test:17000/v1/admin/collections/docs%2F2026",
    );
  });

  it("redacts the admin token in error messages", async () => {
    const mock = createMockFetch(() =>
      textResponse("forbidden admin-token", 403),
    );
    const client = await connect(
      makeConfig({ readOnly: false, adminToken: "admin-token" }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "delete_collection", {
      collection: "docs",
    });

    expect(isError).toBe(true);
    expect(text).toContain("HouyiDB HTTP 403");
    expect(text).toContain("[REDACTED]");
    expect(text).not.toContain("admin-token");
  });
});

describe("error mapping", () => {
  it("includes status and a redacted, truncated server body", async () => {
    const token = "top-secret";
    const mock = createMockFetch(() =>
      textResponse(
        `upstream exploded token=${token} ${"x".repeat(MAX_ERROR_BODY_CHARS * 2)}`,
        503,
      ),
    );
    const client = await connect(
      makeConfig({ authToken: token }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "collection_status", {
      collection: "docs",
    });

    expect(isError).toBe(true);
    expect(text).toContain("HouyiDB HTTP 503");
    expect(text).toContain("upstream exploded");
    expect(text).toContain("[REDACTED]");
    expect(text).not.toContain(token);
    expect(text).toContain("truncated");
    expect(text.length).toBeLessThan(MAX_ERROR_BODY_CHARS * 2);
  });

  it("maps transport failures to a readable tool error without the token", async () => {
    const token = "top-secret";
    const mock = createMockFetch(() => {
      throw new Error(`ECONNREFUSED token=${token}`);
    });
    const client = await connect(
      makeConfig({ authToken: token }),
      mock.fetchImpl,
    );

    const { isError, text } = await callText(client, "collection_status", {
      collection: "docs",
    });

    expect(isError).toBe(true);
    expect(text).toContain("Failed to reach HouyiDB");
    expect(text).toContain("[REDACTED]");
    expect(text).not.toContain(token);
  });

  it("maps invalid JSON to a readable tool error", async () => {
    const mock = createMockFetch(() => textResponse("not-json", 200));
    const client = await connect(makeConfig(), mock.fetchImpl);

    const { isError, text } = await callText(client, "collection_status", {
      collection: "docs",
    });

    expect(isError).toBe(true);
    expect(text).toContain("invalid JSON");
    expect(text).toContain("not-json");
  });
});
