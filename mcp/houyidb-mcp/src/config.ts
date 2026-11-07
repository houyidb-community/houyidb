/**
 * 环境变量解析与 collection 白名单。
 * EN: Environment parsing and collection allowlisting.
 */

import { DEFAULT_BASE_URL } from "./constants.js";
import { CollectionNotAllowedError, HouyidbMcpError } from "./errors.js";

/**
 * 服务运行配置（不可变）。
 * EN: Immutable runtime configuration.
 */
export interface HouyidbMcpConfig {
  readonly baseUrl: string;
  readonly authToken?: string;
  /**
   * 可选管理 token：只用于管理工具（删除整个 collection），绝不用于普通读写。
   * EN: Optional admin token: used by admin tools only (dropping a collection),
   * never by normal read/write calls.
   */
  readonly adminToken?: string;
  readonly readOnly: boolean;
  readonly collections: ReadonlySet<string>;
}

/**
 * 配置非法时抛出；消息不包含 token。
 * EN: Thrown for invalid configuration; messages never include the token.
 */
export class ConfigError extends HouyidbMcpError {}

const TRUE_VALUES = new Set(["1", "true", "yes", "on"]);
const FALSE_VALUES = new Set(["0", "false", "no", "off"]);

/**
 * 解析只读开关。默认 1；无法识别的值直接报错，避免静默开启写权限。
 * EN: Parses the read-only switch. Defaults to 1; unknown values fail fast
 * instead of silently enabling writes.
 */
function parseReadOnly(raw: string | undefined): boolean {
  const normalized = (raw ?? "").trim().toLowerCase();
  if (normalized === "") return true;
  if (TRUE_VALUES.has(normalized)) return true;
  if (FALSE_VALUES.has(normalized)) return false;
  throw new ConfigError(
    `HOUYIDB_MCP_READ_ONLY must be one of 1/0, true/false, yes/no, on/off; ` +
      `HOUYIDB_MCP_READ_ONLY 取值非法: "${raw}"`,
  );
}

/**
 * 校验并归一化服务地址，防止非 HTTP 协议或内嵌凭据。
 * EN: Validates and normalizes the base URL, rejecting non-HTTP protocols and
 * embedded credentials.
 */
function parseBaseUrl(raw: string | undefined): string {
  const candidate = (raw ?? "").trim() || DEFAULT_BASE_URL;
  let url: URL;
  try {
    url = new URL(candidate);
  } catch {
    throw new ConfigError(
      `HOUYIDB_URL is not a valid URL: "${candidate}". ` +
        `HOUYIDB_URL 不是合法 URL: "${candidate}"。`,
    );
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new ConfigError(
      `HOUYIDB_URL must use http or https; HOUYIDB_URL 必须使用 http 或 https。`,
    );
  }
  if (url.username !== "" || url.password !== "") {
    throw new ConfigError(
      `HOUYIDB_URL must not embed credentials; use HOUYI_AUTH_TOKEN instead. ` +
        `HOUYIDB_URL 不允许内嵌凭据，请改用 HOUYI_AUTH_TOKEN。`,
    );
  }
  if (url.search !== "" || url.hash !== "") {
    throw new ConfigError(
      `HOUYIDB_URL must not contain a query string or fragment. ` +
        `HOUYIDB_URL 不允许包含查询串或片段。`,
    );
  }
  return candidate.replace(/\/+$/, "");
}

/**
 * 从环境变量读取配置。空白的白名单表示“全部允许”。
 * EN: Reads configuration from the environment. An empty allowlist means all
 * collections are allowed.
 */
export function loadConfig(
  env: NodeJS.ProcessEnv = process.env,
): HouyidbMcpConfig {
  const collections = new Set(
    (env.HOUYIDB_MCP_COLLECTIONS ?? "")
      .split(",")
      .map((name) => name.trim())
      .filter((name) => name.length > 0),
  );
  const token = (env.HOUYI_AUTH_TOKEN ?? "").trim();
  const adminToken = (env.HOUYI_ADMIN_TOKEN ?? "").trim();

  return {
    baseUrl: parseBaseUrl(env.HOUYIDB_URL),
    ...(token === "" ? {} : { authToken: token }),
    ...(adminToken === "" ? {} : { adminToken }),
    readOnly: parseReadOnly(env.HOUYIDB_MCP_READ_ONLY),
    collections,
  };
}

/**
 * 判断 collection 是否对本次 MCP 会话可见。
 * EN: Whether a collection is visible to this MCP session.
 */
export function isCollectionAllowed(
  config: HouyidbMcpConfig,
  collection: string,
): boolean {
  return config.collections.size === 0 || config.collections.has(collection);
}

/**
 * 在发起任何 HTTP 请求前强制白名单校验。
 * EN: Enforces the allowlist before any HTTP request is made.
 */
export function assertCollectionAllowed(
  config: HouyidbMcpConfig,
  collection: string,
): void {
  if (!isCollectionAllowed(config, collection)) {
    throw new CollectionNotAllowedError(collection);
  }
}
