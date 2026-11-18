# houyidb-mcp

> [简体中文](README.md)

A Model Context Protocol (MCP) server for HouyiDB. It exposes HouyiDB collection
metadata, vector search, hybrid search, dense+sparse fusion search, and opt-in
writes to MCP clients such as Claude Code, Cursor, and Cline over **stdio**.

- **Read-only by default**: `upsert`, `create_collection`, `delete`, and
  `delete_collection` are not registered at all in read-only mode.
- Built-in collection allowlisting is enforced before every collection-scoped
  HTTP request.
- Optional `HOUYI_AUTH_TOKEN` is forwarded as a Bearer token and never written to
  logs or tool errors.
- No extra HTTP client dependency: it uses the Node.js 20+ global `fetch`.

## Requirements

- Node.js 20 or newer to run the MCP server (Node 22 LTS recommended; `@hono/node-server` requires >=20).
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
| `fusion_search` | read-only | `POST /v1/collections/{name}/fusion/search` | `collection`, `vector`, `sparse`, `k`, optional `filters` / `include_payload` / `ef_search` / `fusion` / `rrf_k` / `alpha`; the collection must be `sparse=bm25`. |
| `upsert` | read-write | `POST /v1/collections/{name}/upsert` | `collection`, `vectors`. |
| `create_collection` | read-write | `PUT /v1/collections/{name}` | `collection`, `dims`, optional `metric` / `index_type` / `vector_dtype` / `quantization` / `sparse` / `hnsw`; a normal write path using `HOUYI_AUTH_TOKEN` only. |
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
- `sparse`: optional companion sparse terms for a dense vector,
  `[[term_id, weight], ...]`, where `term_id` is an integer in `1..2^24-1`
  and `weight` is finite and > 0; at most 1024 pairs per document (the server
  sums duplicate terms) and at most 64 pairs per fusion query. Only
  `sparse=bm25` collections accept it; any other collection returns 400. In
  `upsert`, an absent key (or JSON `null`) leaves the document sparse
  unchanged and `[]` clears it; writes use whole-document replacement.
  `fusion_search` requires `sparse` non-empty.
- `upsert.vectors`:
  `{ "id": 42, "values": [0.1, 0.2], "label": "optional", "payload": { "category": "fruit" } }`.
  A collection is created implicitly on first write.
- `create_collection`: explicitly fixes `dims` plus optional `metric`
  (`l2`/`cosine`/`ip`/`l1`), `index_type`, `vector_dtype` (`f32` default /
  `f16`), `quantization` (`none` default / `int8` / `rabitq`), `sparse`
  (`none` default / `bm25`), and `hnsw` (all
  three of `{ m, ef_construction, ef_search }` must be supplied together).
  `vector_dtype: "f16"` is hnsw-only, immutable after creation, and halves
  vector memory; the server also accepts aliases such as
  `float16/fp16/half` case-insensitively and MCP forwards them verbatim;
  `quantization: "int8"` (1 B/dim) and `"rabitq"` (1 bit/dim) compress only the
  HNSW graph-side copy while the main store keeps raw vectors for exact
  reranking; they are hnsw-only and immutable after creation, and an explicit
  `"none"` is legal on any backend. `rabitq` supports `metric` `l2`/`cosine`
  only (`ip`/`l1` return 400). An **explicit** non-hnsw `index_type` with
  `int8`/`rabitq` is rejected locally, and so is `rabitq` with an explicit
  `ip`/`l1` metric, while an omitted `index_type` is forwarded verbatim and
  resolved by the server defaults. `sparse: "bm25"` works on `hnsw` / `flat`
  backends only and is immutable after creation; implicitly created
  collections are always `none`; an **explicit** non-`hnsw`/`flat`
  `index_type` with `sparse: "bm25"` is rejected locally. An existing name
  yields 409; metric and backend are immutable afterwards. It is a normal
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

`fusion_search` requires both `vector` and `sparse` to be non-empty
(sparse-only retrieval is not offered yet). `fusion` defaults to `rrf`
(`score = Σ 1/(rrf_k + rank)`, `rrf_k` defaults to 60 and ranges 1..=1000);
`weighted` min-max normalizes each path and then computes
`alpha*dense + (1-alpha)*sparse` (`alpha` defaults to 0.5 and ranges 0..=1).
Explicitly supplied `rrf_k`/`alpha` values are always strictly validated even
under the inactive mode. `scores` only expresses fused ranking and belongs to
no metric semantics; `dense_scores` are metric-converted (`null` when the
document missed the dense path) and `sparse_scores` are raw BM25 (`null` when
it missed the sparse path). It is a pure read that never creates a collection
implicitly; a `sparse=none` collection returns 400; an older server without
the endpoint returns 404 passed through verbatim.

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

Both runtime and tests require Node.js 20+ (the shared floor of
`@hono/node-server` and Vitest 4; Node 22 LTS recommended).
中文：运行与测试均要求 Node.js 20+（`@hono/node-server` 与 Vitest 4 的共同下限；推荐 22 LTS）。

```bash
npm install
npm run build
npm test
```

Tests use Vitest with an injected mock fetch and cover:

- environment defaults and the read-only switch;
- read-only tool registration (write tools absent) and the full read-write toolset;
- allowlist filtering and pre-request blocking for every collection-scoped tool;
- `search` / `hybrid_search` / `fusion_search` request bodies (including the
  default `rrf` and `rrf_k`/`alpha` forwarding), auth header, and path encoding;
- `upsert` / `create_collection` / `delete` request bodies, `create_collection`
  using only the normal token, `vector_dtype` / `quantization` / `sparse`
  forwarding plus local rejection of an explicit non-hnsw backend (including
  `bm25` with an explicit non-`hnsw`/`flat` backend), and empty-delete
  rejection;
- non-2xx responses, oversized bodies, token redaction, transport failures, and
  invalid JSON.

## Package metadata

- npm package: `houyidb-mcp`
- MCP Registry name: `io.github.houyidb-community/houyidb-mcp`
- Transport: stdio
- Registry manifest: [`server.json`](server.json)

## License

Apache-2.0. See the repository root `LICENSE` for the full terms.
