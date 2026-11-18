/**
 * HouyiDB 客户端公共类型（与 Python SDK 的数据类一一对应）。
 *
 * 与服务端的契约要点：
 * - collection 在首次 upsert 时隐式创建，维度由首个向量定型；
 * - payload 键 "label" 为保留列，占用会 400（用顶层 label 字段）；
 * - 检索响应的 score 是相似度，按 collection 度量换算（cosine 为 1 - d 且
 *   钳制 [-1,1]；l2/l1 为 1 - d 不钳制；ip 为原始点积），越大越相似；
 * - 向量 id 是 u64：一律以 bigint 暴露（Number 安全域只到 2^53-1），
 *   输入侧接受 bigint | number | string 并归一。
 *
 * EN: HouyiDB client public types (one-to-one with the Python SDK dataclasses).
 *
 * EN: Contract points with the server:
 * EN: - A collection is created implicitly on the first upsert; its dimensions
 * are fixed by the first vector;
 * EN: - The payload key "label" is a reserved column; using it returns 400
 * (use the top-level label field instead);
 * EN: - The score in search responses is a similarity converted per the
 * collection metric (cosine is 1 - d clamped to [-1,1]; l2/l1 are 1 - d
 * unclamped; ip is the raw dot product), larger is more similar;
 * EN: - Vector ids are u64: always exposed as bigint (the Number safe range
 * only reaches 2^53-1); the input side accepts bigint | number | string and
 * normalizes them.
 */

import { parseSearchResponse } from "./json.js";

/**
 * 向量 id 的输入形态（内部归一为 bigint）。
 * EN: Accepted input forms for a vector id (normalized to bigint internally).
 */
export type VectorIdInput = bigint | number | string;

export function toVectorId(id: VectorIdInput): bigint {
  const v = typeof id === "bigint" ? id : BigInt(id);
  if (v < 0n || v > (1n << 64n) - 1n) {
    throw new RangeError(`vector id out of u64 range: ${id}`);
  }
  return v;
}

/**
 * 稀疏词项对：``[term_id, weight]``，term_id 取值 ``[1, 2^24-1]``、weight
 * 有限且为正；重复 term 由服务端求和。
 * EN: A sparse term pair: ``[term_id, weight]`` with term_id in ``[1, 2^24-1]``
 * and a finite positive weight; duplicate terms are summed by the server.
 */
export type SparseTerm = [number, number];

/**
 * 一条待写入向量：id + 分量 + 可选 label/payload/sparse 词项。
 *
 * ``sparse`` 仅对 ``sparse: "bm25"`` 的集合有效：缺省表示不改该文档的稀疏
 * 词项，``[]`` 表示清空。
 * EN: A vector to write: id + components + optional label/payload/sparse
 * terms. ``sparse`` only applies to ``sparse: "bm25"`` collections: absent
 * means the document's sparse terms are unchanged, and ``[]`` clears them.
 */
export interface Vector {
  id: VectorIdInput;
  values: number[];
  label?: string;
  payload?: Record<string, unknown>;
  sparse?: SparseTerm[];
}

export function vectorToJson(v: Vector): Record<string, unknown> {
  const doc: Record<string, unknown> = { id: toVectorId(v.id), values: v.values };
  if (v.label !== undefined) doc.label = v.label;
  if (v.payload && Object.keys(v.payload).length > 0) doc.payload = v.payload;
  // 缺省键 = 不改，空数组 = 清空：两种语义必须靠「键存在性」区分，不能用空值
  // 到 undefined 的转换抹平。
  // EN: An absent key means unchanged and an empty array means clear: the two
  // semantics must be distinguished by key presence, never flattened by
  // converting empty values to undefined.
  if (v.sparse !== undefined) doc.sparse = v.sparse;
  return doc;
}

/**
 * 过滤算子白名单（与服务端一致；多条 Filter 之间取交集）。
 * EN: Filter operator whitelist (matches the server; multiple Filters are intersected).
 */
export type FilterOp = "eq" | "gt" | "gte" | "lt" | "lte" | "in";

/**
 * 一条结构化过滤条件。
 * EN: A structured filter condition.
 */
export interface Filter {
  column: string;
  op: FilterOp;
  value: unknown;
}

/**
 * 单条命中：id + 相似度分数 + 可选 payload（includePayload 时）。
 * EN: A single hit: id + similarity score + optional payload (when includePayload is set).
 */
export interface SearchHit {
  id: bigint;
  score: number;
  payload: Record<string, unknown> | null;
}

/**
 * 检索响应：hits 按相似度从高到低排列。
 * EN: Search response: hits are ordered by similarity from high to low.
 */
export class SearchResult {
  constructor(
    public readonly hits: SearchHit[],
    /**
     * 本次检索使用的度量（服务端回显；benchmark 直连路径可能省略）。
     * 有了它，调用方才能判断 l2/l1/ip 的越界分数是否合法，而不必再发一次
     * status 请求（OCR #30）。
     * EN: Metric used by this search (echoed by the server; omitted on the
     * benchmark path). It lets callers judge whether out-of-range l2/l1/ip
     * scores are legitimate without an extra status round-trip (OCR #30).
     */
    public readonly metric?: string,
  ) {}

  get ids(): bigint[] {
    return this.hits.map((h) => h.id);
  }

  get scores(): number[] {
    return this.hits.map((h) => h.score);
  }

  /**
   * 从原始响应文本解析（ids 数组经 bigint 防腐蚀处理，见 json.ts）。
   * payloads 的键是 id 的十进制字符串（服务端契约）。
   * EN: Parse from the raw response text (the ids array is protected against
   * corruption via bigint, see json.ts). Keys of payloads are the decimal
   * string form of ids (server contract).
   */
  static fromRawJson(raw: string): SearchResult {
    const body = parseSearchResponse(raw);
    const ids = body.ids ?? [];
    const scores = body.scores ?? [];
    const payloads = body.payloads;
    const hits = ids.map((i, idx) => {
      const id = BigInt(i);
      return {
        id,
        score: scores[idx],
        payload: payloads?.[id.toString()] ?? null,
      };
    });
    return new SearchResult(hits, body.metric);
  }
}

/**
 * 单条融合命中：id + 融合分 + 两路原始分 + 可选 payload。
 *
 * 融合分只表达排序（RRF 或加权），不属于任何度量口径；``denseScore`` 按
 * 集合度量换算、仅稠密一路命中时存在；``sparseScore`` 为 BM25 原始分、
 * 仅稀疏一路命中时存在。
 * EN: One fusion hit: id + fused score + both raw modality scores + optional
 * payload. The fused score only expresses ranking (RRF or weighted) and
 * belongs to no metric; ``denseScore`` is converted under the collection
 * metric and exists only when the dense path hit; ``sparseScore`` is the raw
 * BM25 score and exists only when the sparse path hit.
 */
export interface FusionHit {
  id: bigint;
  score: number;
  denseScore: number | null;
  sparseScore: number | null;
  payload: Record<string, unknown> | null;
}

/**
 * 融合检索响应：hits 按融合分从高到低排列，附生效参数回显。
 *
 * ``metric`` 仅用于解释 ``denseScore``；``fusionMode`` 为生效模式
 * （``rrf`` | ``weighted``）；``rrfK`` 仅 rrf 模式回显、``alpha`` 仅
 * weighted 模式回显（缺席为 null）。
 * EN: Fusion response: hits ordered by fused score, highest first, with the
 * effective parameters echoed. ``metric`` explains ``denseScore`` only;
 * ``fusionMode`` is the effective mode (``rrf`` | ``weighted``); ``rrfK`` is
 * echoed only under rrf and ``alpha`` only under weighted (absent is null).
 */
export class FusionResult {
  constructor(
    public readonly hits: FusionHit[],
    public readonly metric?: string,
    public readonly fusionMode?: string,
    public readonly rrfK: number | null = null,
    public readonly alpha: number | null = null,
  ) {}

  get ids(): bigint[] {
    return this.hits.map((h) => h.id);
  }

  get scores(): number[] {
    return this.hits.map((h) => h.score);
  }

  /**
   * 从原始响应文本解析（ids 数组经 bigint 防腐蚀处理，见 json.ts）。
   * 可选字段缺席（旧服务端/精简响应）时按 null 容忍。
   * EN: Parse from the raw response text (the ids array is protected against
   * corruption via bigint, see json.ts). Absent optional fields (older servers
   * or trimmed responses) are tolerated as null.
   */
  static fromRawJson(raw: string): FusionResult {
    const body = parseSearchResponse(raw);
    const ids = body.ids ?? [];
    const scores = body.scores ?? [];
    // scores 是融合排序键，服务端保证与 ids 等长；短数组会让命中拿到
    // undefined 分值（静默错排），因此按协议违例直接报错而不是伪造分值。
    // 模态数组仍按可选字段容忍：缺席表示该路全部未命中。
    // EN: scores is the fused ranking key and the server guarantees it is
    // parallel to ids; a short array would leave hits with undefined scores
    // (silent misordering), so treat it as a protocol violation instead of
    // faking values. Modality arrays stay optional-field tolerant: absent
    // means that path missed every hit.
    if (scores.length !== ids.length) {
      throw new Error(
        `malformed fusion response: scores length ${scores.length} != ids length ${ids.length}`,
      );
    }
    const denseScores = body.dense_scores ?? [];
    const sparseScores = body.sparse_scores ?? [];
    const payloads = body.payloads;
    const hits = ids.map((i, idx) => {
      const id = BigInt(i);
      return {
        id,
        score: scores[idx],
        denseScore: denseScores[idx] ?? null,
        sparseScore: sparseScores[idx] ?? null,
        payload: payloads?.[id.toString()] ?? null,
      };
    });
    const fusion = body.fusion;
    return new FusionResult(
      hits,
      body.metric,
      fusion?.mode,
      fusion?.rrf_k ?? null,
      fusion?.alpha ?? null,
    );
  }
}

/**
 * 服务端返回的非 2xx 响应（statusCode + 服务端诊断文本）。
 * EN: Non-2xx response returned by the server (statusCode + server diagnostic text).
 */
export class HouyiError extends Error {
  constructor(
    public readonly statusCode: number,
    message: string,
  ) {
    super(`houyidb HTTP ${statusCode}: ${message}`);
    this.name = "HouyiError";
  }
}
