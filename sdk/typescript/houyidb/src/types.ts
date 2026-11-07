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
 * 一条待写入向量：id + 分量 + 可选 label/payload。
 * EN: A vector to write: id + components + optional label/payload.
 */
export interface Vector {
  id: VectorIdInput;
  values: number[];
  label?: string;
  payload?: Record<string, unknown>;
}

export function vectorToJson(v: Vector): Record<string, unknown> {
  const doc: Record<string, unknown> = { id: toVectorId(v.id), values: v.values };
  if (v.label !== undefined) doc.label = v.label;
  if (v.payload && Object.keys(v.payload).length > 0) doc.payload = v.payload;
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
