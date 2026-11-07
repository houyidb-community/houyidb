# houyidb-mcp

> [简体中文](README.md)

A Model Context Protocol (MCP) server for HouyiDB. It exposes HouyiDB collection
metadata, vector search, hybrid search, and opt-in writes to MCP clients such as
Claude Code, Cursor, and Cline over **stdio**.

- **Read-only by default**: `upsert`, `create_collection`, `delete`, and
  `delete_collection` are not registered at all in read-only mode.
- Built-in collection allowlisting is enforced before every collection-scoped
  HTTP request.
- Optional `HOUYI_AUTH_TOKEN` is forwarded as a Bearer token and never written to
  logs or tool errors.
- No extra HTTP client dependency: it uses the Node.js 18+ global `fetch`.

## Requirements

- Node.js 18 or newer to run the MCP server.
- A reachable HouyiDB HTTP service, defaulting to `http://127.0.0.1:17000`.

## Build and run (inside this repository)

```bash
cd mcp/houyidb-mcp
npm install
npm run build
node dist/index.js
```

The server carries the MCP protocol over stdin/stdout, so never write regular logs
to stdout. This implementation only writes startup failures to stderr.

After the package is published to npm, `npx -y houyidb-mcp` will also work. This
repository currently targets the local build and has not published the package.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HOUYIDB_URL` | `http://127.0.0.1:17000` | HouyiDB HTTP base URL. Only `http` / `https` are accepted; embedded credentials are rejected. |
| `HOUYI_AUTH_TOKEN` | empty | Optional Bearer token for normal reads/writes, sent in the `Authorization` header. |
| `HOUYI_ADMIN_TOKEN` | empty | Optional admin Bearer token used **only** for `/v1/admin/*` (`delete_collection`); the tool reports a configuration error when unset. |
| `HOUYIDB_MCP_READ_ONLY` | `1` | `1/true/yes/on` disables writes; `0/false/no/off` enables write tools. Any other value fails startup. |
| `HOUYIDB_MCP_COLLECTIONS` | empty | Comma-separated allowlist. Empty means all collections are visible. Surrounding whitespace is trimmed. |

## Tools

| Tool | Access | HTTP mapping | Arguments |
|---|---|---|---|
| `list_collections` | read-only | `GET /v1/collections` | None; results are filtered by the allowlist. |
| `collection_status` | read-only | `GET /v1/collections/{name}/status` | `collection`. |
| `search` | read-only | `POST /v1/collections/{name}/search` | `collection`, `vector`, `k`, optional `include_payload`, optional `ef_search`. |
| `hybrid_search` | read-only | `POST /v1/collections/{name}/hybrid/search` | `collection`, `vector`, `k`, optional `filters`, optional `include_payload`. |
| `upsert` | read-write | `POST /v1/collections/{name}/upsert` | `collection`, `vectors`. |
| `create_collection` | read-write | `PUT /v1/collections/{name}` | `collection`, `dims`, optional `metric` / `index_type` / `hnsw`; a normal write path using `HOUYI_AUTH_TOKEN` only. |
| `delete` | read-write | `POST /v1/collections/{name}/delete` | `collection`, optional `ids`, optional `filters`; at least one is required. |
| `delete_collection` | read-write (destructive) | `DELETE /v1/admin/collections/{name}` | `collection`; irreversible, requires `HOUYI_ADMIN_TOKEN`, still constrained by the allowlist. |

### Argument details

- `vector`: a non-empty array of numbers. The server checks its dimension against
  the collection definition.
- `k`: `1..10000`, matching the server-side cap.
- `ef_search`: optional HNSW search beam width, `1..10000`.
- `filters`: structured filter array; conditions are ANDed:
  `{ "column": "category", "op": "eq", "value": "fruit" }`.
  Supported `op` values are `eq` / `gt` / `gte` / `lt` / `lte` / `in`. For `in`,
  `value` must be an array.
- `upsert.vectors`:
  `{ "id": 42, "values": [0.1, 0.2], "label": "optional", "payload": { "category": "fruit" } }`.
  A collection is created implicitly on first write.
- `create_collection`: explicitly fixes `dims` plus optional `metric`
  (`l2`/`cosine`/`ip`/`l1`), `index_type`, and `hnsw` (all three of
  `{ m, ef_construction, ef_search }` must be supplied together). An existing
  name yields 409; metric and backend are immutable afterwards. It is a normal
  write path and never carries `HOUYI_ADMIN_TOKEN`.
- `delete`: ids and filter matches are unioned. An empty delete request is rejected
  at the MCP layer.
- `delete_collection`: unconditionally drops a whole collection and is
  **irreversible**. The server is WAL-first, so crash replay and follower
  replication converge to "gone". Unknown/already-dropped collections return 404;
  export via the export endpoint before deleting. It is registered only in
  read-write mode and requires `HOUYI_ADMIN_TOKEN`.

`search` / `hybrid_search` return HouyiDB's raw JSON (`ids`, `scores`, `metric`,
and `payloads` when `include_payload` is enabled). `scores` are metric-derived
and higher means more similar: `cosine` → `1 - distance` in [-1, 1], `l2` / `l1`
→ `1 - distance` (unclamped), `ip` → the raw dot product. Structured filters
rely on the server's scalar pages and are currently supported by the `hnsw` /
`flat` backends. Scalar pages are volatile in v1; see the HouyiDB server
documentation for the exact semantics.

## Client configuration

Claude Code (project-level `.mcp.json`), Cursor (`.cursor/mcp.json`), and the Cline
MCP Servers JSON all accept the same stdio configuration:

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

Set `HOUYIDB_MCP_READ_ONLY` to `0` to enable writes. If HouyiDB has RBAC or a
gateway in front of it, add `HOUYI_AUTH_TOKEN` to `env`; for `delete_collection`
also add `HOUYI_ADMIN_TOKEN` (grant it only to an admin identity). Restart the
relevant MCP connection after changing client configuration.

## Docker

```bash
docker build -t houyidb-mcp .
docker run -i --rm \
  -e HOUYIDB_URL=http://host.docker.internal:17000 \
  houyidb-mcp
```

`-i` is required for stdio MCP; do not add `-t`. On Linux, accessing the host
usually also needs:

```bash
docker run -i --rm --add-host=host.docker.internal:host-gateway \
  -e HOUYIDB_URL=http://host.docker.internal:17000 houyidb-mcp
```

You can also put the Docker command in the MCP client configuration's `command` /
`args`.

## Security and error handling

- **Read-only by default**: this protects the MCP surface only; it does not replace
  HouyiDB's server-side RBAC.
- **Allowlist**: every tool with a collection argument checks
  `HOUYIDB_MCP_COLLECTIONS` before making an HTTP request. `list_collections`
  results are filtered too.
- **No token in logs**: the server only sends the Bearer header and never prints
  request headers. If an upstream error body unexpectedly contains `HOUYI_AUTH_TOKEN`
  or `HOUYI_ADMIN_TOKEN`, it is replaced with `[REDACTED]` before being returned. The
  admin token is used only for `/v1/admin/*` and is never attached to normal
  reads/writes.
- **Readable errors**: HTTP non-2xx returns the status code and server body, with
  oversized bodies truncated and marked. Network failures, timeouts, and invalid
  JSON become MCP `isError` results instead of stack traces.
- **Input bounds**: `k` / `ef_search` are limited at the MCP layer, and collection
  path segments are RFC 3986 encoded so `/`, `?`, or `#` cannot rewrite routes.

## Development and tests

```bash
npm install
npm run build
npm test
```

Tests use Vitest with an injected mock fetch and cover:

- environment defaults and the read-only switch;
- read-only tool registration (write tools absent) and the full read-write toolset;
- allowlist filtering and pre-request blocking for every collection-scoped tool;
- `search` / `hybrid_search` request bodies, auth header, and path encoding;
- `upsert` / `create_collection` / `delete` request bodies, `create_collection`
  using only the normal token, and empty-delete rejection;
- non-2xx responses, oversized bodies, token redaction, transport failures, and
  invalid JSON.

## Package metadata

- npm package: `houyidb-mcp`
- MCP Registry name: `io.github.houyidb-community/houyidb-mcp`
- Transport: stdio
- Registry manifest: [`server.json`](server.json)

## License

Apache-2.0. See the repository root `LICENSE` for the full terms.
