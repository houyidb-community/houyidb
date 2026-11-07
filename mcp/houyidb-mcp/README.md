# houyidb-mcp

> [English](README.en.md)

HouyiDB 的 Model Context Protocol（MCP）服务器。它通过 **stdio** 把 HouyiDB 的
collection 管理、向量检索、混合检索与可选写入能力暴露给 Claude Code、Cursor、
Cline 等 MCP 客户端。

- 默认**只读**：`upsert` / `create_collection` / `delete` / `delete_collection`
  在只读模式下根本不会注册。
- 内置 collection 白名单，所有 collection 维度工具在发请求前统一校验。
- 可选 `HOUYI_AUTH_TOKEN`，以 Bearer 方式透传，不在日志或工具错误中输出。
- 不引入额外 HTTP 客户端依赖，运行时使用 Node 18+ 全局 `fetch`。

## 环境要求

- Node.js 18 或更高版本（运行 MCP 服务器）。
- 一个可从当前进程访问的 HouyiDB HTTP 服务，默认 `http://127.0.0.1:17000`。

## 安装与运行（仓库内）

```bash
cd mcp/houyidb-mcp
npm install
npm run build
node dist/index.js
```

服务器使用标准输入/输出承载 MCP 协议，因此不要向 stdout 写普通日志。本实现只向
stderr 写启动错误。

包发布到 npm 后，也可以直接通过 `npx -y houyidb-mcp` 启动；当前仓库以本地构建
产物为准，未执行发布。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOUYIDB_URL` | `http://127.0.0.1:17000` | HouyiDB HTTP 基地址；仅允许 `http` / `https`，禁止内嵌凭据。 |
| `HOUYI_AUTH_TOKEN` | 空 | 可选 Bearer token，放入 `Authorization` 请求头（普通读写）。 |
| `HOUYI_ADMIN_TOKEN` | 空 | 可选管理员 Bearer token，**只**用于 `/v1/admin/*`（`delete_collection`）；未配置时该工具返回配置错误。 |
| `HOUYIDB_MCP_READ_ONLY` | `1` | `1/true/yes/on` 只读，`0/false/no/off` 启用写工具；其他值启动即报错。 |
| `HOUYIDB_MCP_COLLECTIONS` | 空 | 逗号分隔白名单；空值表示全部 collection 可见。名称首尾空白会被裁剪。 |

## 工具

| 工具 | 权限 | HTTP 映射 | 参数 |
|---|---|---|---|
| `list_collections` | 只读 | `GET /v1/collections` | 无；返回结果按白名单过滤。 |
| `collection_status` | 只读 | `GET /v1/collections/{name}/status` | `collection`。 |
| `search` | 只读 | `POST /v1/collections/{name}/search` | `collection`, `vector`, `k`, 可选 `include_payload`, 可选 `ef_search`。 |
| `hybrid_search` | 只读 | `POST /v1/collections/{name}/hybrid/search` | `collection`, `vector`, `k`, 可选 `filters`, 可选 `include_payload`。 |
| `upsert` | 读写 | `POST /v1/collections/{name}/upsert` | `collection`, `vectors`。 |
| `create_collection` | 读写 | `PUT /v1/collections/{name}` | `collection`, `dims`, 可选 `metric` / `index_type` / `hnsw`；普通写路径，只用 `HOUYI_AUTH_TOKEN`。 |
| `delete` | 读写 | `POST /v1/collections/{name}/delete` | `collection`, 可选 `ids`, 可选 `filters`；两者至少给其一。 |
| `delete_collection` | 读写（破坏性） | `DELETE /v1/admin/collections/{name}` | `collection`；不可逆，需要 `HOUYI_ADMIN_TOKEN`，仍受白名单约束。 |

### 参数说明

- `vector`：非空数字数组；服务端负责校验维度与 collection 定义一致。
- `k`：`1..10000`，与服务端上限一致。
- `ef_search`：可选 HNSW 搜索束宽，`1..10000`。
- `filters`：结构化过滤数组，条件之间为 AND：
  `{ "column": "category", "op": "eq", "value": "fruit" }`。
  `op` 支持 `eq` / `gt` / `gte` / `lt` / `lte` / `in`；`in` 的 `value` 是数组。
- `upsert.vectors`：
  `{ "id": 42, "values": [0.1, 0.2], "label": "可选", "payload": { "category": "fruit" } }`。
  首次写入会隐式创建 collection。
- `create_collection`：显式一次定型 `dims` 与可选 `metric`（`l2`/`cosine`/`ip`/`l1`）、
  `index_type` 与 `hnsw`（`{ m, ef_construction, ef_search }` 三项必须同时给出）。
  同名已存在返回 409；度量与后端创建后不可变。它走普通写路径，绝不携带
  `HOUYI_ADMIN_TOKEN`。
- `delete`：`ids` 与 `filters` 命中的集合取并集；空请求会被 MCP 层直接拒绝。
- `delete_collection`：无条件删除整个 collection，**不可逆**；服务端 WAL-first，
  崩溃重放与 follower 复制都会收敛到「已删除」。未知/重复删除返回 404；删除前
  建议先用导出端点备份。仅在读写模式注册，且必须配置 `HOUYI_ADMIN_TOKEN`。

`search` / `hybrid_search` 返回 HouyiDB 的原始 JSON（`ids`、`scores`、`metric`，
开启 `include_payload` 时还有 `payloads`）。`scores` 按集合度量换算、越大越相似：
`cosine` → `1 - distance` ∈ [-1, 1]，`l2` / `l1` → `1 - distance`（不截断），
`ip` → 点积本身。结构化过滤依赖服务端标量页，当前仅 `hnsw` / `flat` 后端支持；
标量页在 v1 为易失数据，具体语义以 HouyiDB 服务端文档为准。

## 客户端配置

Claude Code（项目级 `.mcp.json`）、Cursor（`.cursor/mcp.json`）和 Cline 的
MCP Servers JSON 均可使用同一段 stdio 配置：

```json
{
  "mcpServers": {
    "houyidb": {
      "command": "node",
      "args": ["/ABSOLUTE/PATH/TO/mcp/houyidb-mcp/dist/index.js"],
      "env": {
        "HOUYIDB_URL": "http://127.0.0.1:17000",
        "HOUYIDB_MCP_READ_ONLY": "1",
        "HOUYIDB_MCP_COLLECTIONS": "docs,faq"
      }
    }
  }
}
```

需要写入时把 `HOUYIDB_MCP_READ_ONLY` 改为 `0`。若 HouyiDB 启用了 RBAC 或网关
鉴权，在 `env` 中增加 `HOUYI_AUTH_TOKEN`；需要 `delete_collection` 时再增加
`HOUYI_ADMIN_TOKEN`（请只授予管理员 token）。修改客户端配置后重启对应 MCP 连接。

## Docker

```bash
docker build -t houyidb-mcp .
docker run -i --rm \
  -e HOUYIDB_URL=http://host.docker.internal:17000 \
  houyidb-mcp
```

`-i` 是 stdio MCP 的必要条件，不要加 `-t`。Linux 上访问宿主机时通常还需要：

```bash
docker run -i --rm --add-host=host.docker.internal:host-gateway \
  -e HOUYIDB_URL=http://host.docker.internal:17000 houyidb-mcp
```

也可以把 Docker 命令写进 MCP 客户端配置的 `command` / `args`。

## 安全与错误处理

- **只读默认开启**：这是 MCP 表面的保护，不替代 HouyiDB 的服务端 RBAC。
- **白名单**：所有带 collection 参数的工具在发起 HTTP 请求前校验
  `HOUYIDB_MCP_COLLECTIONS`；`list_collections` 的返回同样会过滤。
- **Token 不落日志**：服务只发送 Bearer 头，不打印请求头；若上游错误正文意外包含
  `HOUYI_AUTH_TOKEN` 或 `HOUYI_ADMIN_TOKEN`，均先替换为 `[REDACTED]` 再返回。
  管理 token 只用于 `/v1/admin/*`，普通读写绝不携带。
- **可读错误**：HTTP 非 2xx 会返回状态码与服务端正文；超长正文截断并标注长度。
  网络失败、超时、非法 JSON 也会转为 MCP `isError` 结果，而不是抛出堆栈。
- **输入边界**：`k` / `ef_search` 在 MCP 层提前限制；collection 路径段做 RFC 3986
  转义，避免 `/`、`?`、`#` 改写路由。

## 开发与测试

```bash
npm install
npm run build
npm test
```

测试使用 Vitest 与注入式 mock fetch，覆盖：

- 环境变量默认值与只读开关；
- 只读模式工具注册（写工具不可见）与读写模式完整工具集；
- collection 白名单过滤与所有 collection 级工具的预请求拦截；
- `search` / `hybrid_search` 请求体、鉴权头与路径转义；
- `upsert` / `create_collection` / `delete` 请求体、`create_collection` 只用普通
  token、空删除拒绝；
- 非 2xx、超长正文、token 脱敏、传输失败与非法 JSON 的错误映射。

## 发布元数据

- npm 包名：`houyidb-mcp`
- MCP Registry 名称：`io.github.houyidb-community/houyidb-mcp`
- 传输：stdio
- Registry 清单：[`server.json`](server.json)

## 许可证

Apache-2.0。完整条款见仓库根目录的 `LICENSE`。
