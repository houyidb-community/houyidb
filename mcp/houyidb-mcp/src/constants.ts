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

// 服务端对 f16 的别名集合（大小写不敏感）：本地预拒绝必须识别所有别名，
// 同时不得比服务端更严——schema 因此放宽为 string，由服务端做最终校验。
// EN: f16 aliases accepted by the server (case-insensitive). Local
// pre-rejection must recognize every alias yet stay no stricter than the
// server, so the schema accepts a plain string and the server does final
// validation.
export const F16_DTYPE_ALIASES = ["f16", "float16", "fp16", "half"] as const;

// 稀疏向量边界与服务端同口径：term ∈ [1, 2^24-1]，每文档最多 1024 对，
// 查询最多 64 对（服务端对原始对数计数，重复 term 由服务端求和）。
// EN: Sparse bounds mirror the server: term ∈ [1, 2^24-1], at most 1024 raw
// pairs per document and 64 per query (counted before duplicate terms are
// summed server-side).
export const MAX_SPARSE_TERM_ID = (1 << 24) - 1;
export const MAX_SPARSE_TERMS_PER_DOC = 1_024;
export const MAX_SPARSE_QUERY_TERMS = 64;

// 创建期稀疏索引模式（大小写不敏感的服务端归一不在 MCP schema 范围）。
// EN: Sparse index modes accepted at creation (the server's
// case-insensitive normalization is out of the MCP schema's scope).
export const SPARSE_MODES = ["none", "bm25"] as const;

// bm25 是可过滤后端上的旁路倒排索引，与服务端 validate_create 同口径。
// EN: bm25 is a side inverted index on filterable backends, matching the
// server's validate_create.
export const SPARSE_BM25_BACKENDS: readonly IndexTypeName[] = ["hnsw", "flat"];

// 融合模式、RRF 平滑常数上限与服务端缺省值（缺省值在 schema 描述与请求
// 构造两处共用，避免字面量漂移）。
// EN: Fusion modes, the RRF smoothing-constant ceiling, and the server
// defaults (shared by the schema descriptions and request building so the
// literals cannot drift).
export const FUSION_MODES = ["rrf", "weighted"] as const;
export type FusionModeName = (typeof FUSION_MODES)[number];
export const MAX_RRF_K = 1_000;
// 缺省值用联合类型约束：字面量拼错会在编译期失败，而不是到服务端才 400。
// EN: The default is constrained by the union type so a typo fails at
// compile time instead of surfacing as a server-side 400.
export const DEFAULT_FUSION_MODE: FusionModeName = "rrf";
export const DEFAULT_RRF_K = 60;
export const DEFAULT_FUSION_ALPHA = 0.5;
