# Changelog

> [中文](CHANGELOG.md)

## Unreleased

### Fixed
- **Payload filters only see the last row per id** (public issue #6): after
  re-upserting the same id, the old payload no longer participates in filtering.
  Before the fix, stale rows in the append-only page still matched — writing
  `{"color":"red"}` and then `{"color":"blue"}` made a `color=red` filter return
  the id while reading back the blue payload, and the behavior persisted across
  restarts. Filtering, delete-by-filter and payload read-back now share the
  last-wins rule; compaction only reclaims physical rows and never changes
  query results.

### Changed
- **English documentation navigation**: cross-links in English documents now
  consistently target `.en.md` files, and `SECURITY(.en).md` points security
  verification to the deployment guide and alert runbooks (the former
  `docs/ENTERPRISE_READINESS.md` is not part of the public snapshot); broken
  relative references are fixed. (Credit: @Aqiu16717, public PR #4)

## v0.2.0-beta.8 (2026-11-07)

### Added
- **`DELETE /v1/admin/collections/{name}`**: unconditional whole-collection
  drop (admin `Action::Admin`, audited automatically). Deletion is WAL-first:
  a `collection_drop` record is appended before the in-memory removal, so
  crash replay and follower live apply both converge to "gone". A same-name
  recreate starts empty and never resurrects the old vectors. Responds with
  `{"deleted":true,"collection":"<name>"}`; unknown/already-dropped 404,
  delete races 409, follower/read-only 503. `/v1/version` advertises
  `collection_delete`.
- **Six-language SDKs + MCP**: the Rust/Go/Python/TypeScript/.NET/Java clients
  gain an optional admin token (attached only to `/v1/admin/*` requests) and a
  `delete_collection` / `DeleteCollection` API; `houyidb-mcp` gains a
  `delete_collection` tool (registered only in read-write mode, requires
  `HOUYI_ADMIN_TOKEN`, and still honors the allowlist).
- **`PUT /v1/collections/{name}` explicit creation** (normal write permission,
  `Action::Write`): fixes `dims` / `metric` / `index_type` / the HNSW triple in
  one shot; parameters are validated before the WAL append. An existing name
  yields 409, and metric/backend are immutable afterwards. `/v1/version`
  advertises `collection_create` / `collection_metric`.
- **Collection-level distance metrics**: `l2` / `cosine` / `ip` / `l1` (the
  kernel always uses "smaller = closer"; HTTP scores convert at the single
  edge: cosine `1-d` clamped to [-1,1], l2/l1 `1-d` unclamped, ip the raw dot
  product). Support matrix: hnsw / flat / clustered / flat-clustered support
  all four; pq-flat only l2; hii-l0 only cosine. Collection/hybrid search
  responses gain a `metric` field (the benchmark path stays byte-identical).
- **Six-language SDK + MCP creation APIs**: `create_collection` /
  `createCollection` / `CreateCollection` / `CreateCollectionAsync` accept
  metric/backend/HNSW options and echo the effective parameters; the MCP server
  gains a `create_collection` tool that uses the normal `HOUYI_AUTH_TOKEN` and
  never the admin token.

### Changed
- **Replication protocol v8**: extends v7 (`collection_drop`) with
  `collection_create` metric/HNSW semantics (wire format unchanged). The
  fail-fast version gate isolates v8 from older deployments because old binaries
  silently skip unknown ops and rebuild create records as cosine (resurrected
  drops plus silent metric drift); upgrades must be coordinated cluster-wide.
- **Collection snapshot v2**: carries `metric` plus optional HNSW parameters;
  v1 decodes to the backend default metric. Graph snapshots carry the metric
  and reject mismatches; `collection_create` WAL records carry metric/HNSW too,
  so cold replay never resets historical collections to cosine.
- **Snapshots/retention**: a collection being dropped no longer aborts the
  snapshot cycle (it is skipped); after dropping everything the empty manifest
  commits `safe_checkpoint_lsn = global_barrier_lsn` so the WAL can be
  truncated; a collection created after the barrier has a strictly greater
  Create LSN, keeping truncation safe. Followers **never** publish empty-name
  manifests: their local WAL appends before apply, so `next_lsn - 1` may
  outrun a not-yet-applied create/upsert; they keep the previous manifest
  (temporarily using more local WAL) until a collection lands or they are
  promoted to leader.

### Upgrade
- Replication protocol **v8 is not rolling-compatible with v7 or earlier
  (beta.7 = v6)**: stop replication, upgrade every replica, then resume.
- **No downgrade**: never cold-replay v8 WAL with a v7 binary—it skips
  `collection_drop` and ignores the metric fields on create records,
  resurrecting dropped collections and silently drifting to cosine. Fall back
  to a backup or a delete-and-recreate (recreate with the original metric).

## v0.2.0-beta.7 (2026-11-02)

### Added
- `GET /v1/collections/{name}/export`: streaming NDJSON logical export
  (header + id-sorted records + footer). Built vectors are shared via `Arc`
  instead of cloned; payloads use last-write-wins and tombstones are skipped.
  `/v1/version` advertises `collection_export`.
- Collection list/status expose a read-only `ha_level` (changes still go
  through `POST /v1/collections/{name}/ha_level`, WAL-first).

### Fixed
- The concurrent-build regression test now adaptively waits for the queue to
  drain, removing a flaky Linux CI failure under load.

## v0.2.0-beta.6 (2026-10-30) — P5.0 storage lifecycle

### Added
- **Collections snapshots + manifest** (HOUYISNAP/HOUYIMAN): cold start skips
  records covered by a checkpoint; a truncated history without a manifest is
  fatal instead of replaying a gapped log.
- **Segmented WAL with bounded retention**: default 16 × 64 MiB, whole segments
  only; `/v1/wal/status`, `/metrics` and
  `POST /v1/admin/collections/compact` expose and drive retention.
- **Replication protocol v6**: handshakes carry the follower's earliest readable
  LSN, manifest checkpoint and stable follower_id; ANCHOR/HNDA negotiation;
  `BEHIND_RETAINED` with a full-resync runbook; the retention floor follows
  active followers' durable LSN (configurable disconnect grace).
- **In-memory scalar compaction**: checked every 10 minutes and compacted at
  ≥50% stale rows; status exposes physical/live/stale/tombstone counts;
  `POST /v1/collections/{name}/compact` triggers it manually.
- Promote replays the local WAL tail above each collection's `applied_lsn` and
  refuses to promote on failure.
- **Sentinel control-plane primitives (EE prerequisite, not paywalled)**:
  `cluster_state.json` persists the term (monotonic, corruption is fatal);
  leader leases with write fencing (`--require-leader-lease`, expiry returns
  503 `lease_expired` / pgwire 25006); `POST/DELETE/GET /v1/admin/lease`,
  `POST /v1/admin/demote`, `GET /v1/admin/replication/status`,
  `GET /v1/version`, `GET /v1/admin/whoami`, `GET /v1/admin/diagnostics`;
  promote accepts an optional `{term, ttl_ms}` to grant atomically; demote
  accepts `follower_of` (the new leader's replication address) to convert the
  old leader back into a follower at runtime and catch up without a restart.
  With everything off, behavior matches manual promote.
- **Replication quorum waits**: `--replication-sync-quorum N` (0 = any ACK,
  the default). With N > 0 a sync write waits until N active followers have
  DURABLE ≥ the write LSN (K-th largest).
- **Query log and stats**: `--query-log-capacity` / `--query-log-sample-every`
  (off by default). A bounded ring buffer records metadata only (no vectors,
  payloads or SQL); `GET /v1/admin/query-log` and
  `GET /v1/admin/query-stats` (p50/p95/p99).
- **Sentinel RBAC role**: a dedicated `sentinel_token` (ENV/TOML only, never
  a CLI flag) may only call lease/demote/promote/replication-status; admin
  keeps everything.
- Fuzz target `collection_snapshot_decode`.

### Changed
- **Behavior change (filtering)**: after compaction a filter matches only the
  last row per id (append-only pages previously let stale rows match).
- Startup self-check probes use reserved IDs and are removed; the benchmark
  `/v1/index/status` reports effective `distance_metric`, `hnsw_m` and
  `hnsw_ef_construction`.

### Upgrade
- Replication **v6 is not rolling-compatible with v5**: upgrade all replicas
  together.
- After the first WAL truncation, downgrading to an older binary is unsafe; use
  a backup or the full-resync path instead.
- Run `POST /v1/admin/collections/compact` once before upgrading so old history
  is covered by a snapshot.

## v0.2.0-beta.5 (2026-10-25)

### Added
- **Embedding integration (Route A)**: `docs/EMBEDDING_GUIDE.en.md` (Ollama / OpenAI /
  TEI / mock recipes), `examples/text_ingest.py` (chunk → automatic ingestion → search),
  `scripts/embed_smoke.sh` (end-to-end self-check), `scripts/mock_embedding_server.py`
  (semantic-free mock upstream). Added §2.6 to the trial manual; release packages natively include docs/examples/scripts.
- The trial manual supports prebuilt packages for both Apple Silicon and Intel platforms, plus an embedding chapter.

### Changed
- `async-trait 0.1.89 → 0.1.92`: Rust 1.99 adds `clippy::double_must_use`,
  which the old macro expansion triggers under `-D warnings`; after the upgrade, clippy is all green on 1.99.
- **CI/release fixes**: replaced 5 broken actions pinned to SHAs with valid v-tag commits;
  Release's `macos-x86_64` now cross-compiles on `macos-latest` (`macos-13` retired);
  the CI Docker build and Release pipelines are restored.

### Fixed
- Embedding example/script fixes: ignore the system proxy, friendly errors for non-UTF-8/missing files,
  chunk/overlap/batch/k parameter validation, and a smoke response parsing guard.
- Dependency security: `cmov 0.5.4`, `vitest ≥4.1.11` (TS SDK dev dependency),
  `tokio` workspace minimum `1.54.1`.

## v0.2.0-beta.4 (2026-10-25)

### Security
- **Dependency hardening**: `tokio 1.52.0 → 1.54.1` (RUSTSEC-2026-0386 `AsyncFd` I/O
  safety, unsound warning); `ctutils 0.4.2 → 0.4.5` (RUSTSEC-2026-0385
  non-constant-time masking; the chain is a dev dependency
  `tokio-postgres → postgres-protocol → digest → ctutils`).
  `cargo audit` recheck: 0 vulnerabilities / 0 warnings; `cargo test --all` all 660 tests green.

## v0.2.0-beta.3 (2026-10-25)

### Added
- **Filtered-search contiguous vector cache**: matched vectors are lazily copied into a contiguous `Vec<f32>` by label (Cosine
  pre-normalized), degrading filtered queries to sequential SIMD scans + top-k, invalidated after writes. Cohere 1M
  1% filter benchmark: ef128 **1365 QPS**, ef256 **1386 QPS**, recall 1.0.
- **HNSW experimental knobs** (disabled by default): `HOUYI_HNSW_KEEP_PRUNED=1` (keepPruned
  connections, mature recall .996→.9976), `HOUYI_HNSW_RERANK_EXPAND=N` (query-side
  neighbor expansion), `HOUYI_HNSW_COMMIT_CHUNK=N` (serial graph merge with short-lock commits).
- `/metrics` graph-build logs now include `graph` (number of nodes in the graph), used to diagnose coverage gaps between main and the graph.

### Changed
- **HNSW graph merge now defaults to "read-lock planning + short write-lock commit"**: ef_construction planning in
  `run_async_builder_once` moves outside the global write lock (completed in parallel under a read lock); only edge-connection commits hold a short write lock,
  grouped by `parallel_chunk_size × parallel_plan_group_chunks`. Streaming
  benchmark: 1000 serial queries at the 50% stage 104s → **3.9s**, p99 1.72s → **21ms**;
  80/100% checkpoints 206s/211s → **164s/201s**; mature recall unchanged (1× .9876,
  2× .996), post-write QPS 1× **1355/1353**. The commit window uses an independent gate that is mutually exclusive with `remove`
  (it does not take the tombstone lock, avoiding batch-write queuing); `HOUYI_HNSW_PLAN_APPLY=0` reverts to
  the old path. On the same day, Cohere 1M bulk-load total wall clock +11% (1055s → 1170s, recall/QPS
  on par; use the fallback switch when load time is sensitive). See
  `docs/benchmarks/houyidb-performance-2026-10.en.md`.
- **HNSW default connection density remains 2×M**: 1× would mark old 2× graph snapshots as incompatible and fall back to replay;
  if the WAL has already been truncated by a checkpoint, a data-loss window exists. Therefore the code default remains 2× (compatible
  with old snapshots); high-QPS scenarios should explicitly set `HOUYI_HNSW_LAYER0_MULT=1` (the VDB campaign
  default is 1×). On the same machine, 1× improves OpenAI/Cohere QPS by about 35–60%, while Cohere
  recall goes 0.9971 → 0.9883.

## v0.2.0-beta.2 (2026-10-21)

### Added
- **Correctness verification (P4.5)**: proptest property-based testing, cargo-fuzz, shuttle/loom, retrieval oracle,
  crash chaos, turmoil deterministic network simulation, Stateright model checking, Jepsen-style linearizability, soak sentinel.
- **Replication hardening (P4.6)**: protocol v5 (log prefix fingerprint + auth token), promote/shutdown mutual exclusion,
  HaLevel enum, FileWal poisoning, in-place apply retry, independent DNS/handshake timeouts, hermetic SDK integration.
- **Audit log (P4.7.0)**: SHA-256 hash chain, startup verification, poison on write failure, audit verification endpoint.
- **RBAC (P4.7.1)**: configuration-based principals admin/writer/reader + collection scopes.
- **Tenant quotas (P4.7.2)**: per-principal collection count/in-flight/write-rate limits (HTTP 429, pgwire 54000).
- **Encryption at rest (P4.7.3)**: AES-256-GCM; WAL v3, graph snapshot v2, legacy JSON snapshot encryption;
  local keyring + KeyProvider, key rotation, cross-key fingerprint consistency in replication.
- **Supply chain and operations (P4.8)**: CI cargo audit/deny + CycloneDX SBOM; alert runbooks and
  Prometheus rules; SLO/error budget; RTO/MTTR measurement scripts; Helm chart
  (StatefulSet/PVC/Secret); psql/psycopg2/pgbench external client conformance;
  removed the unmaintained hwloc native dependency (NUMA now reads sysfs).
- **pgwire compatibility**: accept BEGIN/COMMIT/ROLLBACK/SET as no-ops from standard clients
  (sent implicitly by libpq-family drivers), fixing psycopg2 connection failures.

### Changed
- **pgwire authorization**: Simple Query and Extended Execute share the rejection chain; quota rejections
  (54000) and authorization rejections (42501) are metered separately.
- **Dependency upgrades**: ureq 3 / rustls-pki-types (removing rustls 0.21 and the unmaintained
  rustls-pemfile); security upgrades for anyhow, crossbeam-epoch, tokio-postgres, rustls-webpki.
- **TLS PEM parsing** migrated to rustls-pki-types; Docker/CI no longer need libhwloc.

### Fixed
- Replication divergence (same LSN, different content) is no longer missed by an idempotent skip; startup without a key fails closed.
- pgwire SELECT implicitly creating a collection bypassing quotas; DELETE on an unknown collection falsely reporting 429.
- Code review batches 32 through 35, 24 items in total (including 3 HIGH).

## v0.2.0-beta (2026-05-13)

### Added
- **Multi-collection support**: `CollectionManager` manages collections with independent dimensions/indexes, routed over both HTTP and pgwire
- **Four index backends**: `--index-type hnsw|clustered|flat-clustered|flat`
- **ClusteredHnsw**: K-Means partitioning + parallel subgraph construction, 73% (50K) / 55% (1M) faster loading
- **FlatClustered**: K-Means partitioning + SIMD brute-force search, QPS 231 (highest at 50K)
- **Adaptive nprobe**: dynamically computes (k × 2500 / cluster_size) at query time, avoiding wasted compute for small k
- **Product Quantization**: globally shared codebook + SDC lookup-table retrieval, 32x compression at 128D (16B/vector)
- **PqFlatClustered + mmap reordering**: 15MB memory + 488MB disk, SIFT1M recall 0.937, latency 3.9ms
- **houyi-mobile**: standalone C ABI library, supporting native iOS/Android integration
- **houyi-mobile MobileDB**: multi-collection management, consistent with the server API
- **SIFT1M benchmark against MicroNN**: latency 3.9ms, recall 0.937, RAM 15MB
- **RegionPool lock-free**: CAS replaces Mutex
- **NEON SIMD completion**: 4-accumulator unrolling, full Cosine/L2/dot-product paths
- **Unified configuration system**: centralized in houyi-core/config.rs

### Changed
- **IndexBackend enum**: abstracts the four backends HNSW/ClusteredHnsw/FlatClustered/Flat
- **main.rs module split**: 6339 → 3040 lines (types.rs / tests.rs separated)
- **Vector data sharing via Arc**: main_index and HnswGraph share the same data, eliminating triple storage
- **PAX page cleanup**: removed decorative ptr/layout, vector columns changed to contiguous columnar storage
- **optimize() implementation**: HnswGraph.optimize_compact() heuristic neighbor reselection

### Fixed
- Scattered configuration: 41 env::var sites centralized into 21
- houyi-core zero Mutex (CAS replacement)
- SQL parser upgrade (sqlparser 0.62 AST)

### Known Issues
- 15 pgwire byte-level tests show a 1-byte deviation due to the IndexBackend refactor (no functional impact)
- ClusterGraph automatic splitting not completed (placeholder code)
- PQ not wired into --index-type
- simd-avx512 skeleton only

---

## v0.1.0-alpha (2026-04-24)

### Added
- Hyper-converged monolithic in-memory vector database skeleton
- HNSW graph engine + dual-speed write path
- pgwire + HTTP dual protocols
- WAL persistence + cold-start recovery
- PAX page abstraction + Region pool
