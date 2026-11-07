/**
 * MCP server 组装：配置 → HTTP 客户端 → 工具注册。
 * EN: MCP server assembly: config → HTTP client → tool registration.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { HouyidbMcpConfig } from "./config.js";
import { SERVER_NAME, SERVER_VERSION } from "./constants.js";
import { HouyiClient, type FetchLike } from "./houyi-client.js";
import { registerTools } from "./tools.js";

/**
 * 可注入依赖，便于单元测试替换网络层与超时。
 * EN: Injectable dependencies so unit tests can replace the network layer and
 * timeout.
 */
export interface CreateServerOptions {
  readonly fetchImpl?: FetchLike;
  readonly timeoutMs?: number;
}

/**
 * 构造服务器指令，清楚说明当前权限边界（不含任何凭据）。
 * EN: Builds server instructions that state the active permission boundary
 * without exposing any credential.
 */
function buildInstructions(config: HouyidbMcpConfig): string {
  const mode = config.readOnly
    ? "READ-ONLY: write tools are not registered. 只读模式：写工具未注册。"
    : "READ-WRITE: upsert/delete are enabled. 读写模式：upsert/delete 已启用。";
  const scope =
    config.collections.size === 0
      ? "All collections are visible. 全部 collections 可见。"
      : `Only allowlisted collections are visible: ${[...config.collections]
          .sort()
          .join(", ")}. 仅白名单 collections 可见。`;
  return `HouyiDB MCP server. ${mode} ${scope}`;
}

/**
 * 创建未连接的 MCP server；调用方负责绑定传输层。
 * EN: Creates an unconnected MCP server; the caller binds the transport.
 */
export function createHouyidbServer(
  config: HouyidbMcpConfig,
  options: CreateServerOptions = {},
): McpServer {
  const client = new HouyiClient({
    baseUrl: config.baseUrl,
    ...(config.authToken === undefined ? {} : { authToken: config.authToken }),
    ...(config.adminToken === undefined
      ? {}
      : { adminToken: config.adminToken }),
    ...(options.fetchImpl === undefined
      ? {}
      : { fetchImpl: options.fetchImpl }),
    ...(options.timeoutMs === undefined
      ? {}
      : { timeoutMs: options.timeoutMs }),
  });

  const server = new McpServer(
    {
      name: SERVER_NAME,
      version: SERVER_VERSION,
    },
    {
      instructions: buildInstructions(config),
    },
  );

  registerTools(server, config, client);
  return server;
}

/**
 * 以 stdio 传输运行服务器（MCP 客户端以子进程方式启动）。
 * EN: Runs the server over stdio (MCP clients launch it as a subprocess).
 */
export async function runStdioServer(
  config: HouyidbMcpConfig,
): Promise<McpServer> {
  const server = createHouyidbServer(config);
  await server.connect(new StdioServerTransport());
  return server;
}
