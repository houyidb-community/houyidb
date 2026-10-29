# Changelog

> [中文](CHANGELOG.md)

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
