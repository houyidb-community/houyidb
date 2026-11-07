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
  INDEX_TYPES,
  MAX_ERROR_BODY_CHARS,
  MAX_HNSW_EF,
  MAX_HNSW_M,
  MAX_SEARCH_EF,
  MAX_SEARCH_K,
  METRICS,
  MIN_HNSW_EF,
  MIN_HNSW_M,
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
 * 注册八类工具；只读模式只注册前四类，写工具根本不出现在 tools/list 中。
 * EN: Registers eight tools; read-only mode registers only the first four, so
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

  if (config.readOnly) return;

  server.registerTool(
    "upsert",
    {
      title: "Upsert vectors / 写入向量",
      description:
        "Upsert vectors and optional scalar payloads. The collection is created " +
        "on first write. 写入向量及可选标量 payload；collection 在首次写入时" +
        "隐式创建。",
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
        hnsw: hnswSchema,
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    ({ collection, dims, metric, index_type, hnsw }) =>
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
