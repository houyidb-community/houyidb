/**
 * 服务级常量：版本、超时与 HouyiDB 的检索边界。
 * EN: Service constants: version, timeout, and HouyiDB search limits.
 */

export const SERVER_NAME = "houyidb-mcp";

// 版本号与 package.json / server.json 保持一致；发布时三处必须同步。
// EN: Keep the version in sync with package.json and server.json on release.
export const SERVER_VERSION = "0.1.0";

// 默认连接本机服务；客户端可通过 HOUYIDB_URL 覆盖。
// EN: Defaults to the local service; clients may override it with HOUYIDB_URL.
export const DEFAULT_BASE_URL = "http://127.0.0.1:17000";

// 单次上游请求的超时，避免 MCP 客户端无限等待失联服务。
// EN: Per-request upstream timeout so MCP clients never wait forever.
export const REQUEST_TIMEOUT_MS = 30_000;

// 错误正文上限，避免把超长诊断整体灌进模型上下文。
// EN: Cap error bodies so oversized diagnostics do not flood the model context.
export const MAX_ERROR_BODY_CHARS = 4_096;

// 与服务端 MAX_SEARCH_K 同口径，提前失败并给出可读原因。
// EN: Mirrors the server's MAX_SEARCH_K so failures are immediate and readable.
export const MAX_SEARCH_K = 10_000;

// 与服务端 MAX_SEARCH_EF 同口径，避免请求侧放大预分配。
// EN: Mirrors the server's MAX_SEARCH_EF to avoid request-side allocation blowups.
export const MAX_SEARCH_EF = 10_000;

// 显式建集合的 HNSW 三元组边界（与服务端 400 校验一致）：m ∈ [2,1024]，
// ef_construction / ef_search ∈ [1,65536]。注意创建期的 ef_search 上限
// 高于检索期的 MAX_SEARCH_EF（建集合只落参数，不预分配检索缓冲）。
// EN: HNSW triple bounds for explicit creation (matching the server's 400
// validation): m ∈ [2,1024], ef_construction / ef_search ∈ [1,65536]. The
// creation-time ef_search ceiling is higher than search-time MAX_SEARCH_EF
// because creation only persists parameters and allocates no search buffer.
export const MIN_HNSW_M = 2;
export const MAX_HNSW_M = 1_024;
export const MIN_HNSW_EF = 1;
export const MAX_HNSW_EF = 65_536;

// 度量与后端的字面量集合在此单点定义：zod schema 与后端/度量约束表共用，
// 避免键名拼错或字面量漂移后仍编译通过、静默绕过本地预拒绝（OCR 195e6be #01）。
// EN: The metric/backend literal sets are defined once here and shared by the
// zod schemas and the backend/metric constraint table, so a key typo or drifted
// literal cannot compile and silently disable the local pre-rejection
// (OCR 195e6be #01).
export const METRICS = ["l2", "cosine", "ip", "l1"] as const;
export type MetricName = (typeof METRICS)[number];

export const INDEX_TYPES = [
  "hnsw",
  "pq-flat",
  "flat",
  "clustered",
  "flat-clustered",
  "hii-l0",
] as const;
export type IndexTypeName = (typeof INDEX_TYPES)[number];

// 各后端唯一支持的度量：pq-flat 仅 L2（ADC 码本按平方欧氏距离训练），
// hii-l0 仅 Cosine。省略 metric 时据此派生默认值（与服务端一致）。
// EN: The only metric each backend supports: pq-flat is L2-only (its ADC
// codebooks are trained on squared Euclidean distance) and hii-l0 is
// Cosine-only. When metric is omitted the default is derived accordingly,
// matching the server.
export const BACKEND_METRIC_CONSTRAINTS: Partial<
  Record<IndexTypeName, MetricName>
> = {
  "pq-flat": "l2",
  "hii-l0": "cosine",
};
