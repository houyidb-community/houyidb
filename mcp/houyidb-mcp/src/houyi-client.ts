/**
 * HouyiDB HTTP 客户端（Node 18+ 全局 fetch）。
 *
 * 设计要点：
 * - 错误正文先脱敏再截断，token 不进入日志或 MCP 工具结果；
 * - 非 2xx 与 JSON 解析失败使用不同的错误类型，便于工具层给出可读提示；
 * - fetch 可注入，单元测试无需真实网络。
 *
 * EN: HouyiDB HTTP client built on Node 18+ global fetch.
 * EN: Design notes:
 * - Error bodies are redacted before truncation, keeping the token out of logs
 *   and MCP tool results;
 * - Non-2xx responses and JSON parse failures have distinct error types so the
 *   tool layer can produce readable diagnostics;
 * - fetch is injectable, so unit tests need no real network.
 */

import {
  MAX_ERROR_BODY_CHARS,
  REQUEST_TIMEOUT_MS,
} from "./constants.js";
import {
  HouyiHttpError,
  HouyiResponseError,
  HouyiTransportError,
} from "./errors.js";

/**
 * 最小 fetch 形状只依赖 MCP 服务实际使用的字段，避免绑定 DOM/undici 类型。
 * EN: Minimal fetch shape covering only fields this server uses, avoiding a
 * dependency on DOM or undici type definitions.
 */
export interface FetchResponseLike {
  status: number;
  text(): Promise<string>;
}

/**
 * 可注入的 fetch 实现。
 * EN: Injectable fetch implementation.
 */
export type FetchLike = (
  input: string,
  init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
    signal?: AbortSignal;
  },
) => Promise<FetchResponseLike>;

/**
 * 客户端构造参数。
 * EN: Client constructor options.
 */
export interface HouyiClientOptions {
  readonly baseUrl: string;
  readonly authToken?: string;
  /**
   * 可选管理 token：只附在管理请求（删除整个 collection）上，普通读写不带。
   * EN: Optional admin token: attached only to admin requests (dropping a
   * collection), never to normal read/write calls.
   */
  readonly adminToken?: string;
  readonly fetchImpl?: FetchLike;
  readonly timeoutMs?: number;
}

/**
 * 截断过长正文并显式标注，既保留诊断信息又限制上下文占用。
 * EN: Truncates oversized bodies with an explicit marker, retaining useful
 * diagnostics while bounding context usage.
 */
export function truncateBody(
  body: string,
  maxChars: number = MAX_ERROR_BODY_CHARS,
): string {
  if (body.length <= maxChars) return body;
  const omitted = body.length - maxChars;
  return `${body.slice(0, maxChars)}… [truncated ${omitted} chars]`;
}

/**
 * 从全局对象安全取 fetch，便于给出明确的运行时错误。
 * EN: Safely reads fetch from the global object so runtime errors stay explicit.
 */
function resolveFetch(fetchImpl?: FetchLike): FetchLike {
  if (fetchImpl) return fetchImpl;
  const globalFetch = (globalThis as { fetch?: FetchLike }).fetch;
  if (!globalFetch) {
    throw new HouyiTransportError(
      "Global fetch is unavailable; Node.js 18 or newer is required. " +
        "全局 fetch 不可用，需要 Node.js 18 或更高版本。",
    );
  }
  return globalFetch.bind(globalThis);
}

/**
 * HouyiDB collections API 的最小异步客户端。
 * EN: Minimal async client for the HouyiDB collections API.
 */
export class HouyiClient {
  readonly #baseUrl: string;
  readonly #authToken?: string;
  readonly #adminToken?: string;
  readonly #fetch: FetchLike;
  readonly #timeoutMs: number;

  constructor(options: HouyiClientOptions) {
    this.#baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.#authToken = options.authToken;
    this.#adminToken = options.adminToken;
    this.#fetch = resolveFetch(options.fetchImpl);
    this.#timeoutMs = options.timeoutMs ?? REQUEST_TIMEOUT_MS;
  }

  /**
   * GET 一个 JSON 端点。
   * EN: GET a JSON endpoint.
   */
  async get(path: string): Promise<unknown> {
    return this.#request("GET", path);
  }

  /**
   * POST JSON 到指定端点。
   * EN: POST JSON to an endpoint.
   */
  async post(path: string, payload: unknown): Promise<unknown> {
    return this.#request("POST", path, payload);
  }

  /**
   * PUT JSON 到指定端点（普通写路径，使用 auth token；绝不携带 admin token）。
   * EN: PUT JSON to an endpoint (normal write path using the auth token; never
   * the admin token).
   */
  async put(path: string, payload: unknown): Promise<unknown> {
    return this.#request("PUT", path, payload);
  }

  /**
   * DELETE 一个管理端点（使用 admin token；普通 token 绝不上报）。
   * EN: DELETE an admin endpoint (uses the admin token; the normal token is
   * never sent).
   */
  async deleteAdmin(path: string): Promise<unknown> {
    return this.#request("DELETE", path, undefined, true);
  }

  /**
   * 统一请求路径：超时、鉴权、错误映射与正文脱敏只在这里实现一次。
   * EN: Shared request path: timeout, auth, error mapping, and body redaction
   * are implemented exactly once here.
   */
  async #request(
    method: "GET" | "POST" | "PUT" | "DELETE",
    path: string,
    payload?: unknown,
    admin = false,
  ): Promise<unknown> {
    const headers: Record<string, string> = {
      Accept: "application/json",
    };
    const token = admin ? this.#adminToken : this.#authToken;
    if (token !== undefined) {
      headers.Authorization = `Bearer ${token}`;
    }

    let body: string | undefined;
    if (payload !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(payload);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.#timeoutMs);
    let response: FetchResponseLike;
    try {
      response = await this.#fetch(`${this.#baseUrl}${path}`, {
        method,
        headers,
        ...(body === undefined ? {} : { body }),
        signal: controller.signal,
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : String(error);
      if (controller.signal.aborted) {
        throw new HouyiTransportError(
          `HouyiDB request timed out after ${this.#timeoutMs} ms. ` +
            `HouyiDB 请求在 ${this.#timeoutMs} 毫秒后超时。`,
        );
      }
      throw new HouyiTransportError(
        `Failed to reach HouyiDB at ${this.#baseUrl}: ${this.#redact(message)}`,
      );
    } finally {
      clearTimeout(timer);
    }

    let rawText: string;
    try {
      rawText = await response.text();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : String(error);
      throw new HouyiResponseError(
        `Failed to read HouyiDB response body: ${this.#redact(message)}. ` +
          `读取 HouyiDB 响应正文失败: ${this.#redact(message)}。`,
      );
    }

    const safeText = this.#redact(rawText).trim();
    if (response.status < 200 || response.status >= 300) {
      throw new HouyiHttpError(
        response.status,
        truncateBody(safeText || "<empty response body>"),
      );
    }

    if (safeText === "") {
      throw new HouyiResponseError(
        `HouyiDB returned an empty body with HTTP ${response.status}. ` +
          `HouyiDB 在 HTTP ${response.status} 下返回了空正文。`,
      );
    }

    try {
      return JSON.parse(safeText) as unknown;
    } catch {
      throw new HouyiResponseError(
        `HouyiDB returned invalid JSON (HTTP ${response.status}): ` +
          `${truncateBody(safeText)}. ` +
          `HouyiDB 返回了非法 JSON（HTTP ${response.status}）。`,
      );
    }
  }

  /**
   * 将 token 从任意诊断文本中抹除。即使服务端意外回显凭据也不会泄漏。
   * EN: Removes the token from any diagnostic text, so even an upstream that
   * accidentally reflects credentials cannot leak them.
   */
  #redact(text: string): string {
    let out = text;
    for (const token of [this.#authToken, this.#adminToken]) {
      if (token !== undefined && token !== "") {
        out = out.split(token).join("[REDACTED]");
      }
    }
    return out;
  }
}
