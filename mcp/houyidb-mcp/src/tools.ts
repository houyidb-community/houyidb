/**
 * MCP 工具注册与 HouyiDB 请求映射。
 * EN: MCP tool registration and mapping to HouyiDB requests.
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import {
  assertCollectionAllowed,
  isCollectionAllowed,
  type HouyidbMcpConfig,
} from "./config.js";
import {
  BACKEND_METRIC_CONSTRAINTS,
  DEFAULT_FUSION_ALPHA,
  DEFAULT_FUSION_MODE,
  DEFAULT_RRF_K,
  F16_DTYPE_ALIASES,
  FUSION_MODES,
  INDEX_TYPES,
  MAX_ERROR_BODY_CHARS,
  MAX_HNSW_EF,
  MAX_HNSW_M,
  MAX_RRF_K,
  MAX_SEARCH_EF,
  MAX_SEARCH_K,
  MAX_SPARSE_QUERY_TERMS,
  MAX_SPARSE_TERM_ID,
  MAX_SPARSE_TERMS_PER_DOC,
  METRICS,
  MIN_HNSW_EF,
  MIN_HNSW_M,
  SPARSE_BM25_BACKENDS,
  SPARSE_MODES,
} from "./constants.js";
import { HouyiResponseError, HouyidbMcpError } from "./errors.js";
import type { HouyiClient } from "./houyi-client.js";
import { truncateBody } from "./houyi-client.js";
import {
  adminCollectionPath,
  collectionPath,
  collectionRootPath,
} from "./paths.js";

/**
 * collection 名称本身不能为空；白名单校验在处理器里执行。
 * EN: Collection names must be non-empty; allowlist checks run in the handler.
 */
const collectionSchema = z
  .string()
  .min(1)
  .describe("Collection name / collection 名称");

/**
 * 查询向量：至少一维且拒绝 NaN/Infinity。
 * EN: Query vector with at least one dimension and no NaN/Infinity.
 */
const vectorSchema = z
  .array(z.number().finite())
  .min(1)
  .describe("Dense query vector / 稠密查询向量");

/**
 * 单个稀疏词项对 `[term_id, weight]`：term 为客户端给定的词表 id，权重必须
 * 有限且为正；与服务端同边界，便于提前失败。
 * EN: One sparse term pair `[term_id, weight]`: the term is a client-supplied
 * vocabulary id and the weight must be finite and positive; bounds mirror the
 * server so invalid input fails early.
 */
const sparsePairSchema = z.tuple([
  z
    .number()
    .int()
    .min(1)
    .max(MAX_SPARSE_TERM_ID)
    .describe(`Term id, 1..${MAX_SPARSE_TERM_ID} / 词项 id`),
  z.number().finite().positive().describe("Term weight (> 0) / 词项权重"),
]);

/**
 * top-k 与服务端同边界，提前拒绝放大请求。
 * EN: Top-k shares the server bound so amplification is rejected early.
 */
const kSchema = z
  .number()
  .int()
  .min(1)
  .max(MAX_SEARCH_K)
  .describe(`Top-k, 1..${MAX_SEARCH_K} / 返回条数`);

/**
 * 可选 HNSW 搜索束宽。
 * EN: Optional HNSW search beam width.
 */
const efSearchSchema = z
  .number()
  .int()
  .min(1)
  .max(MAX_SEARCH_EF)
  .optional()
  .describe(`Optional HNSW ef_search, 1..${MAX_SEARCH_EF} / 可选 ef_search`);

/**
 * 是否回传标量 payload。
 * EN: Whether to return scalar payloads for each hit.
 */
const includePayloadSchema = z
  .boolean()
  .optional()
  .describe("Include payloads in hits (default false) / 是否回传 payload");

/**
 * 结构化过滤字面值：标量或标量数组（`in` 操作符）。
 * EN: Structured filter literal: scalar or scalar array (for the `in` operator).
 */
const filterLiteralSchema = z.union([
  z.string(),
  z.number().finite(),
  z.boolean(),
  z.array(z.union([z.string(), z.number().finite(), z.boolean()])),
]);

/**
 * 一条过滤条件，多个条件之间是 AND 交集。
 * EN: One filter condition; multiple filters are intersected with AND.
 */
const filterSchema = z.object({
  column: z
    .string()
    .min(1)
    .describe("Scalar column or reserved `label` column / 标量列名"),
  op: z
    .enum(["eq", "gt", "gte", "lt", "lte", "in"])
    .describe("Comparison operator / 比较操作符"),
  value: filterLiteralSchema.describe(
    "Comparison literal; arrays only for `in` / 比较字面值，仅 `in` 使用数组",
  ),
});

/**
 * 单个待写入向量。
 * EN: One vector to upsert.
 */
const upsertVectorSchema = z.object({
  id: z
    .number()
    .int()
    .min(0)
    .max(Number.MAX_SAFE_INTEGER)
    .describe("Vector id (safe integer range) / 向量 id（安全整数范围内）"),
  values: z.array(z.number().finite()).min(1).describe("Vector values / 向量分量"),
  label: z.string().optional().describe("Optional label / 可选标签"),
  payload: z
    .record(z.string(), z.unknown())
    .optional()
    .describe("Optional scalar payload / 可选标量 payload"),
  // nullable：显式 null 与缺省同义（服务端把 null 视作不改），schema 不得
  // 比线上契约更严；转发时 null 原样保留，语义由服务端归一。
  // EN: nullable: an explicit null means the same as absent (the server treats
  // null as unchanged), so the schema must not be stricter than the wire
  // contract; null is forwarded verbatim and normalized server-side.
  sparse: z
    .array(sparsePairSchema)
    .max(MAX_SPARSE_TERMS_PER_DOC)
    .nullable()
    .optional()
    .describe(
      "Sparse terms for a sparse=bm25 collection: absent = unchanged, " +
        "[] = clear; JSON null is treated as absent / 稀疏词项（仅 " +
        "sparse=bm25 集合）：缺省 = 不改，[] = 清空；JSON null 视同缺省",
    ),
});

/**
 * 建集合的 HNSW 参数三元组（与服务端边界一致；三个字段必须一起给）。
 * EN: HNSW parameter triple for explicit creation (bounds match the server;
 * the three fields must be supplied together).
 */
const hnswSchema = z
  .object({
    m: z
      .number()
      .int()
      .min(MIN_HNSW_M)
      .max(MAX_HNSW_M)
      .describe(`HNSW m, ${MIN_HNSW_M}..${MAX_HNSW_M} / 邻居数`),
    ef_construction: z
      .number()
      .int()
      .min(MIN_HNSW_EF)
      .max(MAX_HNSW_EF)
      .describe(`HNSW ef_construction, ${MIN_HNSW_EF}..${MAX_HNSW_EF} / 构图束宽`),
    ef_search: z
      .number()
      .int()
      .min(MIN_HNSW_EF)
      .max(MAX_HNSW_EF)
      .describe(
        `HNSW ef_search, ${MIN_HNSW_EF}..${MAX_HNSW_EF} at creation ` +
          `(search-time requests cap at ${MAX_SEARCH_EF}) / 创建期 ef_search ` +
          `（检索请求上限为 ${MAX_SEARCH_EF}）`,
      ),
  })
  .optional()
  .describe("Optional HNSW parameters (hnsw backend only) / 可选 HNSW 参数");

/**
 * 把任意成功返回值包装成 MCP 文本结果。
 * EN: Wraps any successful value as an MCP text result.
 */
function successResult(value: unknown): CallToolResult {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(value) ?? "null",
      },
    ],
  };
}

/**
 * 把异常转成出错结果。只暴露消息、不暴露堆栈，并限制长度。
 * EN: Converts failures into error results. Only the message (never a stack
 * trace) is exposed, and its length is capped.
 */
function errorResult(error: unknown): CallToolResult {
  const rawMessage =
    error instanceof Error ? error.message : String(error);
  const message = truncateBody(
    rawMessage.trim() || "unknown error",
    MAX_ERROR_BODY_CHARS,
  );
  return {
    content: [
      {
        type: "text",
        text: `[houyidb-mcp] ${message}`,
      },
    ],
    isError: true,
  };
}

/**
 * 统一执行工具逻辑，确保所有异常都变成可读的 `isError` 结果。
 * EN: Runs tool logic uniformly, turning every exception into a readable
 * `isError` result.
 */
async function runTool(
  operation: () => Promise<unknown>,
): Promise<CallToolResult> {
  try {
    return successResult(await operation());
  } catch (error) {
    return errorResult(error);
  }
}

/**
 * 读取上游 collection 列表并按白名单过滤。
 * EN: Reads upstream collections and filters them through the allowlist.
 */
async function listAllowedCollections(
  config: HouyidbMcpConfig,
  client: HouyiClient,
): Promise<unknown> {
  const data = await client.get("/v1/collections");
  if (!Array.isArray(data)) {
    throw new HouyiResponseError(
      "GET /v1/collections did not return a JSON array. " +
        "GET /v1/collections 未返回 JSON 数组。",
    );
  }
  if (config.collections.size === 0) return data;

  return data.filter((entry) => {
    if (typeof entry !== "object" || entry === null) return false;
    const name = (entry as { name?: unknown }).name;
    return typeof name === "string" && isCollectionAllowed(config, name);
  });
}

/**
 * 注册九类工具；只读模式只注册前五类，写工具根本不出现在 tools/list 中。
 * EN: Registers nine tools; read-only mode registers only the first five, so
 * write tools never appear in tools/list at all.
 */
export function registerTools(
  server: McpServer,
  config: HouyidbMcpConfig,
  client: HouyiClient,
): void {
  server.registerTool(
    "list_collections",
    {
      title: "List collections / 列出 collections",
      description:
        "List HouyiDB collections visible under HOUYIDB_MCP_COLLECTIONS. " +
        "列出 HOUYIDB_MCP_COLLECTIONS 白名单内的 HouyiDB collections。",
      annotations: {
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    () => runTool(() => listAllowedCollections(config, client)),
  );

  server.registerTool(
    "collection_status",
    {
      title: "Collection status / collection 状态",
      description:
        "Read indexed/pending counts, dimension, index type, and scalar rows " +
        "for one collection. 查询单个 collection 的已索引/待构图数量、维度、" +
        "索引类型与标量行数。",
      inputSchema: {
        collection: collectionSchema,
      },
      annotations: {
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    ({ collection }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        return client.get(collectionPath(collection, "status"));
      }),
  );

  server.registerTool(
    "search",
    {
      title: "Vector search / 向量检索",
      description:
        "Run top-k vector search in a collection. 在指定 collection 中执行 " +
        "top-k 向量检索。",
      inputSchema: {
        collection: collectionSchema,
        vector: vectorSchema,
        k: kSchema,
        include_payload: includePayloadSchema,
        ef_search: efSearchSchema,
      },
      annotations: {
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    ({ collection, vector, k, include_payload, ef_search }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        const payload: Record<string, unknown> = {
          vector,
          k,
          include_payload: include_payload ?? false,
        };
        if (ef_search !== undefined) payload.ef_search = ef_search;
        return client.post(collectionPath(collection, "search"), payload);
      }),
  );

  server.registerTool(
    "hybrid_search",
    {
      title: "Hybrid search / 混合检索",
      description:
        "Run vector search with structured scalar filters (AND semantics). " +
        "执行带结构化标量过滤（AND 交集）的向量检索。",
      inputSchema: {
        collection: collectionSchema,
        vector: vectorSchema,
        k: kSchema,
        filters: z
          .array(filterSchema)
          .optional()
          .describe("Structured filters / 结构化过滤条件"),
        include_payload: includePayloadSchema,
      },
      annotations: {
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    ({ collection, vector, k, filters, include_payload }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        return client.post(collectionPath(collection, "hybrid/search"), {
          vector,
          k,
          filters: filters ?? [],
          include_payload: include_payload ?? false,
        });
      }),
  );

  server.registerTool(
    "fusion_search",
    {
      title: "Fusion search / 稠密稀疏融合检索",
      description:
        "Run dense+sparse fusion search on a collection created with " +
        "sparse=bm25. Fusion is rrf (default) or weighted; `scores` only " +
        "ranks, while `dense_scores`/`sparse_scores` carry per-modality " +
        "values. 在创建时开启 sparse=bm25 的 collection 上执行稠密+稀疏融合" +
        "检索；融合模式为 rrf（缺省）或 weighted；`scores` 只表达排序，" +
        "`dense_scores`/`sparse_scores` 分别给出两路原始分。",
      inputSchema: {
        collection: collectionSchema,
        vector: vectorSchema,
        sparse: z
          .array(sparsePairSchema)
          .min(1)
          .max(MAX_SPARSE_QUERY_TERMS)
          .describe(
            `Sparse query terms, 1..${MAX_SPARSE_QUERY_TERMS} pairs / ` +
              `稀疏查询词项（1..${MAX_SPARSE_QUERY_TERMS} 对）`,
          ),
        k: kSchema,
        filters: z
          .array(filterSchema)
          .optional()
          .describe("Structured filters / 结构化过滤条件"),
        include_payload: includePayloadSchema,
        ef_search: efSearchSchema,
        fusion: z
          .enum(FUSION_MODES)
          .optional()
          .describe(
            `Fusion mode (default ${DEFAULT_FUSION_MODE}) / ` +
              `融合模式（缺省 ${DEFAULT_FUSION_MODE}）`,
          ),
        rrf_k: z
          .number()
          .int()
          .min(1)
          .max(MAX_RRF_K)
          .optional()
          .describe(
            `RRF smoothing constant, 1..${MAX_RRF_K} ` +
              `(default ${DEFAULT_RRF_K}, rrf only) / ` +
              `RRF 平滑常数（缺省 ${DEFAULT_RRF_K}，仅 rrf）`,
          ),
        alpha: z
          .number()
          .finite()
          .min(0)
          .max(1)
          .optional()
          .describe(
            "Weighted-fusion dense weight, 0..1 " +
              `(default ${DEFAULT_FUSION_ALPHA}, weighted only) / ` +
              `加权融合的稠密权重（缺省 ${DEFAULT_FUSION_ALPHA}，仅 weighted）`,
          ),
      },
      annotations: {
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    ({
      collection,
      vector,
      sparse,
      k,
      filters,
      include_payload,
      ef_search,
      fusion,
      rrf_k,
      alpha,
    }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        // 缺省值由服务端补齐；这里总是显式给出 fusion 模式，两个调参旋钮
        // 仅在调用方给出时转发——服务端对显式值一律严格校验（即使当前
        // 模式不参与计算），本地不再重复解释。
        // EN: Server-side fills the defaults; the fusion mode is always sent
        // explicitly while the two tuning knobs are forwarded only when the
        // caller supplied them — the server strictly validates explicit
        // values even under the inactive mode, so no local reinterpretation.
        const payload: Record<string, unknown> = {
          vector,
          sparse,
          k,
          filters: filters ?? [],
          include_payload: include_payload ?? false,
          fusion: fusion ?? DEFAULT_FUSION_MODE,
        };
        if (ef_search !== undefined) payload.ef_search = ef_search;
        if (rrf_k !== undefined) payload.rrf_k = rrf_k;
        if (alpha !== undefined) payload.alpha = alpha;
        return client.post(collectionPath(collection, "fusion/search"), payload);
      }),
  );

  if (config.readOnly) return;

  // 稀疏写入只经 HTTP/SDK；pgwire 本轮不提供稀疏写入（AUB1 行不变）。
  // EN: Sparse writes go through HTTP/SDK only; pgwire carries no sparse
  // writes this round (AUB1 rows are unchanged).
  server.registerTool(
    "upsert",
    {
      title: "Upsert vectors / 写入向量",
      description:
        "Upsert vectors and optional scalar payloads. The collection is created " +
        "on first write; implicit creation is always sparse=none, so sparse " +
        "entries require a collection explicitly created with sparse=bm25. " +
        "写入向量及可选标量 payload；collection 在首次写入时隐式创建，隐式" +
        "创建恒为 sparse=none，带稀疏词项写入需先显式创建 sparse=bm25 集合。",
      inputSchema: {
        collection: collectionSchema,
        vectors: z
          .array(upsertVectorSchema)
          .min(1)
          .describe("Vectors to upsert / 待写入向量"),
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    ({ collection, vectors }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        return client.post(collectionPath(collection, "upsert"), { vectors });
      }),
  );

  // 显式建集合：普通写路径（HOUYI_AUTH_TOKEN），不占用 HOUYI_ADMIN_TOKEN。
  // 显式存在同名 collection 时服务端 409；metric/后端创建后不可变。
  // EN: Explicit creation is a normal write path (HOUYI_AUTH_TOKEN) and does not
  // consume HOUYI_ADMIN_TOKEN. The server returns 409 when the name exists;
  // metric/backend are immutable after creation.
  server.registerTool(
    "create_collection",
    {
      title: "Create collection / 创建 collection",
      description:
        "Explicitly create a collection with fixed dims/metric/backend/HNSW " +
        "parameters. Metric and backend are immutable afterwards; an existing " +
        "name fails with 409. 显式创建 collection，一次定型维度、度量、后端与 " +
        "HNSW 参数；度量与后端创建后不可变，同名已存在返回 409。",
      inputSchema: {
        collection: collectionSchema,
        dims: z
          .number()
          .int()
          .min(1)
          .describe("Vector dimension / 向量维度"),
        metric: z
          .enum(METRICS)
          .optional()
          .describe(
            "Distance metric (default cosine; pq-flat=l2, hii-l0=cosine) / " +
              "距离度量（默认余弦；pq-flat=l2，hii-l0=cosine）",
          ),
        index_type: z
          .enum(INDEX_TYPES)
          .optional()
          .describe("Index backend / 索引后端"),
        vector_dtype: z
          .string()
          .optional()
          .describe(
            "Vector storage encoding: f32 (default) or f16 (halves vector " +
              "memory, hnsw-only, immutable after creation); the server also " +
              "accepts float32/fp32/float/single/float16/fp16/half case-insensitively " +
              "/ 向量存储编码：f32（缺省）或 f16（省一半向量内存，仅 hnsw，" +
              "创建后不可变）；服务端另接受 float32/fp32/float/single/float16/" +
              "fp16/half（大小写不敏感）",
          ),
        quantization: z
          .string()
          .optional()
          .describe(
            "Graph-side quantization: none (default), int8, or rabitq " +
              "(hnsw-only, shrinks the in-graph distance working set; the main " +
              "store keeps raw vectors and reranks exactly at search time; " +
              "immutable after creation); rabitq packs 1 bit/dim and supports " +
              "metric l2|cosine only (ip/l1 return 400); the server trims and " +
              "accepts case-insensitively / " +
              "图侧量化：none（缺省）、int8 或 rabitq（仅 hnsw，压缩图内距离" +
              "工作集；主表保留原始向量并在检索时精确重排；创建后不可变）；" +
              "rabitq 为 1 bit/dim，仅支持 metric l2|cosine（ip/l1 返回 400）；" +
              "服务端去空白且大小写不敏感",
          ),
        sparse: z
          .enum(SPARSE_MODES)
          .optional()
          .describe(
            "Sparse side index: none (default) or bm25 (client-supplied terms, " +
              "hnsw/flat-only, immutable after creation) / 稀疏旁路索引：" +
              "none（缺省）或 bm25（客户端给定词项，仅 hnsw/flat，创建后不可变）",
          ),
        hnsw: hnswSchema,
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    ({ collection, dims, metric, index_type, vector_dtype, quantization, sparse, hnsw }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        // 显式给出 index_type 时，本地先拒绝必然 400 的组合（服务端
        // validate_create 同口径）：HNSW 参数只属于 hnsw 后端；pq-flat 仅
        // L2、hii-l0 仅 Cosine（省略 metric 时按后端补默认）。省略
        // index_type 但带 hnsw 参数时不推断、也不本地拒绝：把参数原样转发，
        // 由服务端按部署的 default_index_type 解析（默认后端为 hnsw 时正常
        // 创建，否则服务端自行 400），本地校验因此不会严于服务端契约
        //（OCR 195e6be #02、cf1d19f #00）。本地拒绝统一用 HouyidbMcpError，
        // 与模块其他预期失败保持同一分类（OCR 195e6be #03）。
        // EN: When index_type is explicit, reject locally the combinations the
        // server's validate_create would reject with 400: HNSW parameters
        // belong to the hnsw backend only; pq-flat is L2-only and hii-l0 is
        // Cosine-only (the backend default fills an omitted metric). When
        // index_type is omitted but hnsw parameters are present, neither infer
        // nor reject locally: forward them as-is and let the server resolve the
        // deployment's default_index_type (normal creation when the default is
        // hnsw, a server-side 400 otherwise), so local validation is never
        // stricter than the server contract (OCR 195e6be #02, cf1d19f #00).
        // Local rejections use HouyidbMcpError so expected failures share the
        // module's taxonomy (OCR 195e6be #03).
        if (
          hnsw !== undefined &&
          index_type !== undefined &&
          index_type !== "hnsw"
        ) {
          throw new HouyidbMcpError(
            `hnsw parameters are only supported for index_type "hnsw" (got "${index_type}") / ` +
              `hnsw 参数只适用于 index_type "hnsw"（当前为 "${index_type}"）`,
          );
        }
        // f16 只属于 hnsw 后端：显式给出非 hnsw 后端时本地先拒绝（与服务端
        // validate_create 同口径）；省略 index_type 时原样转发，由服务端按
        // 部署的默认后端解析。
        // EN: f16 belongs to the hnsw backend only: reject locally when an
        // explicit non-hnsw backend is given (same as the server's
        // validate_create); when index_type is omitted, forward as-is and let
        // the server resolve the deployment's default backend.
        // 别名按服务端同口径归一（大小写不敏感）：只有 f16 系列别名需要
        // hnsw 后端，f32 别名（float32/fp32/float）不受后端限制。
        // EN: Normalize aliases exactly like the server (case-insensitive):
        // only f16-family aliases require the hnsw backend; f32 aliases
        // (float32/fp32/float) are unrestricted.
        const dtypeAlias = vector_dtype?.trim().toLowerCase();
        if (
          dtypeAlias !== undefined &&
          (F16_DTYPE_ALIASES as readonly string[]).includes(dtypeAlias) &&
          index_type !== undefined &&
          index_type !== "hnsw"
        ) {
          throw new HouyidbMcpError(
            `vector_dtype "${vector_dtype}" is only supported for index_type ` +
              `"hnsw" (got "${index_type}") / vector_dtype "${vector_dtype}" ` +
              `仅支持 index_type "hnsw"（当前为 "${index_type}"）`,
          );
        }
        // int8/rabitq 只属于 hnsw 后端：显式给出非 hnsw 后端时本地先拒绝
        // （与服务端 validate_create 同口径）；显式 "none" 对任何后端都合法；
        // 省略 index_type 时原样转发，由服务端按部署默认后端解析（默认非
        // hnsw 时服务端自行 400）。判断用与服务端同口径的归一别名（去空白 +
        // 大小写不敏感），转发仍用原始值——本地校验不得比服务端更严，未知串
        // 交由服务端 400。
        // EN: int8/rabitq belongs to the hnsw backend only: reject locally when
        // an explicit non-hnsw backend is given (same as the server's
        // validate_create); an explicit "none" is valid for any backend; when
        // index_type is omitted, forward as-is and let the server resolve the
        // deployment's default backend (a server-side 400 when it is not hnsw).
        // The check uses the alias normalized exactly like the server (trim +
        // case-insensitive) while forwarding the raw value — local validation
        // must not be stricter than the server, so unknown strings are left for
        // the server's 400.
        const quantizationAlias = quantization?.trim().toLowerCase();
        if (
          (quantizationAlias === "int8" || quantizationAlias === "rabitq") &&
          index_type !== undefined &&
          index_type !== "hnsw"
        ) {
          throw new HouyidbMcpError(
            `quantization "${quantization}" is only supported for index_type ` +
              `"hnsw" (got "${index_type}") / quantization "${quantization}" ` +
              `仅支持 index_type "hnsw"（当前为 "${index_type}"）`,
          );
        }
        // rabitq 仅支持 l2/cosine：metric 显式给出 ip/l1 时本地先拒绝（与
        // 服务端 validate_create 同口径）。metric 入参是闭合枚举（仅规范名
        // l2/cosine/ip/l1 能到这里），服务端接受的 euclidean/dot 等别名不在
        // MCP schema 范围内，无需在此归一。
        // EN: rabitq supports l2/cosine only: reject locally when metric is an
        // explicit ip/l1 (same as the server's validate_create). The metric
        // input is a closed enum (only the canonical l2/cosine/ip/l1 names reach
        // this handler), so server-side aliases such as euclidean/dot are out
        // of the MCP schema's scope and need no normalization here.
        if (
          quantizationAlias === "rabitq" &&
          metric !== undefined &&
          (metric === "ip" || metric === "l1")
        ) {
          throw new HouyidbMcpError(
            `quantization "rabitq" only supports metric l2|cosine (got ` +
              `"${metric}") / quantization "rabitq" 仅支持度量 l2|cosine ` +
              `（当前为 "${metric}"）`,
          );
        }
        // bm25 仅支持 hnsw/flat：显式给出其它后端时本地先拒绝（与服务端
        // validate_create 同口径）；省略 index_type 时原样转发，由服务端
        // 按部署默认后端解析（默认后端不支持时服务端自行 400）。
        // EN: bm25 supports hnsw/flat only: reject locally when another
        // backend is explicit (same as the server's validate_create); when
        // index_type is omitted, forward as-is and let the server resolve the
        // deployment's default backend (its own 400 when unsupported).
        if (
          sparse === "bm25" &&
          index_type !== undefined &&
          !SPARSE_BM25_BACKENDS.includes(index_type)
        ) {
          throw new HouyidbMcpError(
            `sparse "bm25" is only supported for index_type "hnsw"|"flat" ` +
              `(got "${index_type}") / sparse "bm25" 仅支持 index_type ` +
              `"hnsw"|"flat"（当前为 "${index_type}"）`,
          );
        }
        const constrainedMetric =
          index_type !== undefined
            ? BACKEND_METRIC_CONSTRAINTS[index_type]
            : undefined;
        if (
          constrainedMetric !== undefined &&
          metric !== undefined &&
          metric !== constrainedMetric
        ) {
          throw new HouyidbMcpError(
            `index_type "${index_type}" only supports metric ` +
              `"${constrainedMetric}" (got "${metric}") / 后端 ` +
              `"${index_type}" 仅支持度量 "${constrainedMetric}"`,
          );
        }
        const payload: Record<string, unknown> = { dims };
        const effectiveMetric = metric ?? constrainedMetric;
        if (effectiveMetric !== undefined) payload.metric = effectiveMetric;
        if (index_type !== undefined) {
          payload.index_type = index_type;
        }
        if (vector_dtype !== undefined) {
          payload.vector_dtype = vector_dtype;
        }
        if (quantization !== undefined) {
          payload.quantization = quantization;
        }
        if (sparse !== undefined) {
          payload.sparse = sparse;
        }
        if (hnsw !== undefined) {
          payload.hnsw_m = hnsw.m;
          payload.hnsw_ef_construction = hnsw.ef_construction;
          payload.hnsw_ef_search = hnsw.ef_search;
        }
        return client.put(collectionRootPath(collection), payload);
      }),
  );

  server.registerTool(
    "delete",
    {
      title: "Delete vectors / 删除向量",
      description:
        "Delete by ids and/or structured filters (the two sets are unioned). " +
        "按 id 和/或结构化过滤条件删除（两者取并集），至少提供其一。",
      inputSchema: {
        collection: collectionSchema,
        ids: z
          .array(z.number().int().min(0).max(Number.MAX_SAFE_INTEGER))
          .optional()
          .describe("Vector ids / 向量 id 列表"),
        filters: z
          .array(filterSchema)
          .optional()
          .describe("Structured filters / 结构化过滤条件"),
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: true,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    ({ collection, ids, filters }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        const idList = ids ?? [];
        const filterList = filters ?? [];
        if (idList.length === 0 && filterList.length === 0) {
          throw new HouyidbMcpError(
            "delete requires at least one id or filter. " +
              "delete 至少需要一个 id 或一条过滤条件。",
          );
        }
        return client.post(collectionPath(collection, "delete"), {
          ids: idList,
          filters: filterList,
        });
      }),
  );

  // 整库删除：不可逆，必须显式配置 HOUYI_ADMIN_TOKEN；仍受 collection 白名单约束。
  // EN: Whole-collection drop: irreversible, requires HOUYI_ADMIN_TOKEN, and
  // remains constrained by the collection allowlist.
  server.registerTool(
    "delete_collection",
    {
      title: "Delete collection / 删除整个 collection",
      description:
        "Drop an entire collection irreversibly (requires HOUYI_ADMIN_TOKEN). " +
        "不可逆地删除整个 collection（需要 HOUYI_ADMIN_TOKEN）。",
      inputSchema: {
        collection: collectionSchema,
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: true,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    ({ collection }) =>
      runTool(async () => {
        assertCollectionAllowed(config, collection);
        if (config.adminToken === undefined) {
          throw new HouyidbMcpError(
            "delete_collection requires HOUYI_ADMIN_TOKEN to be configured. " +
              "delete_collection 需要配置 HOUYI_ADMIN_TOKEN。",
          );
        }
        return client.deleteAdmin(adminCollectionPath(collection));
      }),
  );
}
