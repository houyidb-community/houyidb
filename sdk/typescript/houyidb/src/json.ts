/**
 * bigint 安全的 JSON 序列化/解析。
 *
 * 为什么需要：HouyiDB 的向量 id 是 u64（stable_id 落到 63 位键空间），
 * 而 JS Number 安全整数只到 2^53-1（= 9007199254740991，16 位十进制）。
 * - 发送侧：JSON.stringify 不支持 bigint，原生抛 TypeError。
 * - 接收侧：JSON.parse 对 ≥17 位的整数静默丢低字节（精度腐蚀），
 *   命中回包的 id 会与写入时对不上——跨 SDK 重放删除直接失效。
 *
 * EN: bigint-safe JSON serialization/parsing.
 *
 * EN: Why this exists: HouyiDB vector ids are u64 (stable_id lands in the
 * 63-bit key space), while JS Number-safe integers only reach 2^53-1
 * (= 9007199254740991, 16 decimal digits).
 * EN: - Sending side: JSON.stringify does not support bigint and throws
 * TypeError natively.
 * EN: - Receiving side: JSON.parse silently drops low bytes for integers with
 * 17+ digits (precision corruption), so ids in search responses no longer
 * match the ids at write time—replayed cross-SDK deletes simply fail.
 */

const BIGINT_MARKER = "__houyidb_bigint__";

/**
 * 序列化请求体：bigint 以裸 JSON 数字字面量输出（服务端 serde_json 按 u64 解析）。
 * 经字符串占位符 + 二次替换实现（JSON.stringify 的 replacer 无法输出裸数字）。
 * EN: Serialize a request body: bigints are emitted as raw JSON number
 * literals (the server's serde_json parses them as u64). Implemented via
 * string placeholders plus a second replacement pass (JSON.stringify
 * replacers cannot emit raw numbers).
 */
export function stringify(value: unknown): string {
  return JSON.stringify(value, (_key, v) =>
    typeof v === "bigint" ? `${BIGINT_MARKER}${v}` : v,
  ).replace(new RegExp(`"${BIGINT_MARKER}(\\d+)"`, "g"), "$1");
}

/**
 * 解析响应体：把顶层 "ids" 数组里 ≥16 位的整数先加引号再 parse，
 * 由调用方归一为 bigint。阈值依据：≤15 位十进制必在 2^53 安全域内，
 * 16 位起可能越界——只处理 "ids" 数组，不碰 payload 里用户自存的数值。
 * EN: Parse a response body: integers with 16+ digits inside the top-level
 * "ids" array are quoted before parsing, then normalized to bigint by the
 * caller. Threshold rationale: up to 15 decimal digits always fit the 2^53
 * safe range, while 16 digits may overflow—only the "ids" array is handled,
 * never user-supplied numbers inside payloads.
 */
export function parseSearchResponse(raw: string): {
  ids?: Array<number | string>;
  scores?: number[];
  payloads?: Record<string, Record<string, unknown>>;
  /**
   * 本次检索使用的度量；benchmark 直连路径省略。有它调用方才能判断
   * l2/l1/ip 的越界分数是否合法（OCR #30）。
   * EN: Metric used by this search; omitted on the benchmark path. It lets
   * callers judge whether out-of-range l2/l1/ip scores are legitimate (OCR #30).
   */
  metric?: string;
} {
  const guarded = raw.replace(
    /("ids"\s*:\s*\[)([^\]]*)/,
    (_m, head: string, arr: string) =>
      head + arr.replace(/\b\d{16,}\b/g, '"$&"'),
  );
  return JSON.parse(guarded);
}
