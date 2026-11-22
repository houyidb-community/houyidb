# Changelog

> [中文](CHANGELOG.md)

## Unreleased

### Performance
- **houyi-mobile cold load**: snapshot decoding now reads metadata sequentially
  and copies whole cluster payloads in parallel; the file path uses a read-only
  mmap with parallel chunked CRC32, and cluster buffers are allocated without
  zero-initialization. Added view mode
  `houyi_mobile_db_load_view` / Kotlin `loadView` (mmap plus per-cluster lazy
  decode; the first write materializes every cluster). On-device 100k×128,
  time to first query: f16 52.3 → 11.5 ms with peak RSS 85.5 → 27.0 MB; i8
  32.0 → 10.1 ms with 49.5 → 21.0 MB. Load alone is 7.4–7.5 ms, level with
  USearch `view`. The snapshot format and existing API semantics are unchanged;
  view mode skips the whole-file CRC and the file must not be truncated or
  overwritten in place while mapped (see manual section 2.8).

## v0.2.0-beta.9 (2026-11-21)

### Added
- **houyi-mobile JNI/AAR**: new `houyi-mobile-jni` crate (native bindings for
  the Kotlin `cc.houyidb.mobile.HouyiDB` facade, with u64 search ids) and a
  `houyi-mobile/android` Gradle library project.
  `scripts/build_mobile_android_aar.sh` produces an arm64-v8a AAR (~404 KB;
  ~696 KB native library uncompressed), and `scripts/mobile_jni_smoke.sh`
  verifies JNI names/marshalling/u64/save-load on the host JVM. A C header is
  shipped at `include/houyi_mobile.h`.
- **houyi-mobile filter pushdown**: scalar rows now carry explicit vector ids
  (`insert_scalars_v2` / Kotlin `insertScalarsWithId`), and
  `FlatClustered::search_with_target_filtered` skips disallowed ids during
  candidate scanning (no post-filter shortfall under selective filters).
  `db_search_hybrid_u64` / Kotlin `searchHybrid` return u64 results. Mobile
  snapshots move to v3 with scalar row ids; v1/v2 keep row-index compatibility.
  The legacy scalar API retains row-index semantics.
- **houyi-mobile i8 storage**: `FlatStorageDtype` gains i8 (1 B per dimension
  + a 4-byte per-vector scale; flat-clustered only), wired through
  `build_with_storage_dtype` and C ABI/Kotlin `dtype=2`; f16/i8 share the cosine
  norm cache. Host measurements: real 7.3k×512 recall 0.996 / 4.3 MB file
  (f16 8.3 MB); 100k×192 recall 0.953 / 19.9 MB file (f16 38 MB), query
  latency at parity or slightly better. The server default f32 path is unchanged.
- **houyi-mobile search optimizations**: the f16 cosine path precomputes
  candidate/centroid norms (+4 B per vector, ~0.4 MB at 100k), reducing the
  query hot loop from dot + two norms to a single dot product. Added a
  configurable candidate budget via
  `houyi_mobile_{db_,}set_search_target` / Kotlin `setSearchBudget` (default
  2500/k). On real 7.3k×512 faces (host), `per_k=500` yields p50 ≈ 0.30 ms and
  recall 0.994 (default 2500: 0.43 ms / 1.000). The server's default f32 path
  does not enable the cache, leaving its behavior and performance unchanged.
- **houyi-mobile ABI v2**: batch writes (`insert_batch`/`db_insert_batch`) and
  u64 results (`search_u64`/`db_search_u64`; the legacy f32-id path loses
  precision above 2^24 and is no longer recommended). New
  `create_v2`/`db_collection_v2` take explicit metric (L2/Cosine/IP/L1) and
  dtype (f32/f16; f16 is flat-clustered only); reusing a name with a different
  configuration is rejected. A batch failure returns the first error and does
  not roll back rows already written. Snapshot format v2 records metric/dtype.
- **houyi-mobile f16 storage and save/load**: `flat-clustered` now uses f16
  per-cluster storage on mobile (`houyi-vector` gains `build_with_dtype`; the
  default `build` stays f32, so existing server collections and snapshots are
  unchanged). The 100k×192 index stays at ~37 MB resident (vs ~74 MB for f32);
  real 7.3k×512 recall@10 remains 1.0 at 0.47 ms, and hard 100k synthetic recall
  stays 0.955 (same as f32). New `houyi_mobile_db_save/load`: sibling temp file
  + fsync + rename, with magic/version/length/CRC32; a built flat-clustered
  stores its f16 state and loads without rebuilding, `flat` stores raw rows, and
  `clustered-hnsw` rebuilds on load. Measured 100k×192: 37.6 MB file, 0.10 s
  save, 0.064 s load; real 7.3k×512: 7.3 MB, 0.023/0.012 s (host).
- **houyi-mobile incremental writes and memory dedup**: after a
  `flat-clustered` build, inserts are assigned to the nearest centroid and are
  searchable immediately; overwriting an existing id updates in place or moves
  it across clusters without changing the count, and a later `build`
  reclusters from the current index. The separate write buffer is released
  after a successful build, so vectors are owned once (roughly halving the
  resident index at 100k×192). `clustered-hnsw` post-build inserts accept
  new ids (immediately searchable) and reject overwrites with -2 (recreate the index before
  upserting); `flat` is unchanged.
  Tests: incremental insert and cross-cluster upsert, C ABI immediate
  visibility; on real 7.3k×512 face embeddings recall@10=1.0, build 0.11 s,
  query 0.44 ms.
- **Instance observability**: `/metrics` gains the per-collection gauge
  `houyidb_collection_vectors{collection="..."}` (label values escaped per
  Prometheus rules, sourced from the cheap `indexed_count`), and
  `/v1/version` gains `build_sha`, the short git SHA embedded by
  `houyi-server/build.rs` at build time (`unknown` for source tarballs without
  `.git`; Docker build contexts have no `.git`, so CI passes
  `HOUYIDB_GIT_SHA` as a build-arg/ENV and images no longer always report
  `unknown`). Callers can use it to confirm the exact built commit of a running
  instance.
- **`vector_dtype` f16 storage**: `PUT /v1/collections/{name}` gains an
  optional `vector_dtype` (`f32` default / `f16`; case-insensitive with common
  aliases), allowed only for `index_type=hnsw` and immutable afterwards. f16
  halves resident vector memory; query vectors stay f32 and distances are
  computed against f16-rounded data (asymmetric quantization). The default f32
  behavior is bit-for-bit identical to older versions. Creation responses,
  list summaries, and status echo the effective `vector_dtype`;
  `/v1/version` advertises `vector_dtype_f16`. Collection WAL
  `collection_create` records and collection/graph snapshots (v2 flags bit0,
  manifest `vector_dtype`) carry the encoding, so cold restarts and snapshot
  restores preserve the dtype (f16 collections restore a typed hnsw backend
  from the snapshot header instead of silently falling back to f32). The
  standalone HNSW graph snapshot uses an asymmetric version gate: f32 still
  writes version=1 (older binaries can keep reading it), while f16 writes
  version=2 (older binaries reject it with `UnsupportedVersion` and fall back
  to a full WAL replay — never silently downgrade to f32 and freeze that on
  the next snapshot). An existing name with a different dtype yields 409;
  non-f32 with a non-hnsw backend yields 400.
- **Six-language SDK + MCP f16 support**: the Rust/Go/Python/TypeScript/.NET/
  Java creation APIs accept an optional `vector_dtype` and echo it (left unset
  when an older server omits it); the `houyidb-mcp` `create_collection` tool
  gains a `vector_dtype` parameter (rejecting f16 with an explicit non-hnsw
  backend locally, matching the server rule).
- **`quantization` graph-side int8**: `PUT /v1/collections/{name}` gains an
  optional `quantization` (`none` default / `int8`), allowed only for
  `index_type=hnsw`, immutable afterwards, and independent of `vector_dtype`.
  int8 compresses only the HNSW **graph-side** copy (per-vector affine,
  1 B/dim); the main store, hot buffer, and builder queue keep the raw
  (f32/f16) vectors. Search takes `k.max(ef)` × 2 graph candidates and reranks
  exactly on the raw vectors, so final top-k precision is unchanged. Cosine
  normalizes before quantizing; query vectors stay f32 (asymmetric); the
  zero-norm band keeps its original value. There is no net memory win for
  non-cosine collections (the main store must stay: 4→5 / 2→3 B/dim; only
  cosine drops, f32 8→5 and f16 4→3) — the win is in the graph distance
  working set and bandwidth. The default `none` behavior is bit-for-bit
  identical to older versions. Creation responses, list summaries, and status
  echo the effective `quantization`; `/v1/version` advertises
  `quantization_int8`; an existing name with a different quantization yields
  409; int8 with a non-hnsw backend yields 400 (an explicit `"none"` is legal
  on any backend).
- **Six-language SDK + MCP int8 support**: all six SDK creation APIs accept an
  optional `quantization` (unset keeps the key out of the request body; an
  older server that omits the field leaves it unset rather than defaulting to
  `none`); the `houyidb-mcp` `create_collection` tool gains a `quantization`
  parameter (rejecting int8 with an explicit non-hnsw backend locally).
- **`quantization` graph-side RaBitQ (1 bit/dim)**: the `quantization` value
  domain now spans `none` / `int8` / `rabitq`. RaBitQ requires
  `index_type=hnsw` with `metric=l2|cosine` (`ip` / `l1` yield 400 at
  creation; WAL replay and snapshot expectations fail fast too) and is
  immutable after creation. Kernel: D = dims rounded up to a power of two with
  zero padding; deterministic random rotation P = H_D ∘ diag(signs) (the seed
  depends only on D, bit-reproducible across processes/nodes/restarts); sign
  bits packed into `ceil(D/64)` u64 words plus `alpha=Σ|ō_i|` and `norm`. The
  query side pads and rotates once at the traversal entrance (O(D log D));
  asymmetric estimation `est_cos = clamp(sign_dot/alpha, -1, 1)` gives cosine
  distance `1−est` and squared L2 `max(0, q²+x²−2qx·est)`; stored×stored
  comparisons (construction heuristic/pruning) use Hamming distance plus the
  arcsine law. Zero-norm vectors use the sentinel `{codes=0, alpha=1, norm=0}`.
  Candidate width ×16 (code bandwidth matches int8 ×2: `16·D/8 = 2·D` bytes);
  like int8, the main store / hot buffer / builder queue keep the raw vectors
  and the final top-k is reranked exactly. Graph snapshot
  v4 (`FLAG_VECTOR_RABITQ=0x08` bit3 mandatory, int8 bit2 forbidden), collection
  payload flags bit2=`0x04`, manifest `quantization=2` with `format_version=3`;
  older binaries reject all of these cleanly — downgrades are forbidden. No
  net memory win outside cosine (main store unchanged; the graph side adds
  only ~136 B per vector at 768 dims, ~1/6 of int8) — the win is the in-graph
  distance working set and bandwidth.
- **Six-language SDK + MCP rabitq support**: all six SDK creation APIs accept
  `quantization="rabitq"` (unset keeps the key out of the request body; an
  older server that omits the field leaves it unset rather than defaulting to
  `none`); the `houyidb-mcp` `create_collection` tool extends its enum and
  local validation (int8/rabitq with an explicit non-hnsw backend and rabitq
  with an explicit ip/l1 metric are rejected locally).
- **Sparse vectors + BM25/IDF + dense+sparse fusion search (P4)**: `PUT
  /v1/collections/{name}` gains an optional `sparse` (`none` default / `bm25`),
  hnsw / flat backends only, immutable after creation, and implicit creation is
  always `none`; changing sparse on an existing name yields 409 and an illegal
  combination 400. Sparse terms are caller-supplied (no tokenizer):
  `(term_id, weight)` with term ∈ [1, 2^24-1], finite weight >0, ≤1024 raw
  pairs per document and ≤64 per query; duplicate terms are summed. Upsert
  entries may carry `"sparse": [[t,w],…]` (absent / null = unchanged, `[]` =
  clear; whole-document replacement, idempotent on WAL replay); a `sparse` key
  on a `sparse=none` collection yields 400. New endpoint
  `POST /v1/collections/{name}/fusion/search` (existing `/hybrid/search`
  semantics unchanged): dense and sparse must both be non-empty; both paths
  over-fetch 4×k and fuse via RRF (default `rrf_k=60`) or weighted (default
  `alpha=0.5`, per-path min-max normalization). `scores` **expresses fusion
  ranking only and belongs to no metric**; `dense_scores` follows the
  collection metric, `sparse_scores` is the raw BM25 score, and an absent
  modality is null. BM25 uses `k1=1.2`, `b=0.75`; deletes (by id / filters)
  remove sparse terms too; `status` echoes sparse stats, `/metrics` gains
  `houyidb_sparse_*` and `houyidb_fusion_searches_total`, `/v1/version`
  capabilities gain `sparse_bm25` / `fusion_search`, and `/export` appends an
  optional sparse field per record. Honest memory ledger: ~24 B per (doc, term)
  posting + 12–16 B forward index; 1M documents × 8 terms ≈ 330–400 MB (never
  plan on smaller per-entry numbers).
- **Six-language SDK + MCP sparse/fusion support**: the Rust/Go/Python/
  TypeScript/.NET/Java creation APIs accept an optional `sparse` (tolerating
  older servers that omit it), upsert entries carry sparse terms, and new
  `fusion_search` / `fusionSearch` / `FusionSearchAsync` methods return a
  response model where `dense_scores` / `sparse_scores` / the `fusion` block
  are all optional; the `houyidb-mcp` `create_collection` tool gains a `sparse`
  parameter (rejecting bm25 with a non-hnsw/flat backend locally),
  `upsert_vectors` accepts sparse, and a new `fusion_search` tool is added.
- **pgwire fusion query directives (P4)**: adds
  `/* houyi:sparse=t:w,… */`, `/* houyi:fusion=rrf|weighted */`, and
  `/* houyi:alpha=… */`; sparse and vector must appear together to enter fusion
  search. Sparse alone or fusion params alone → 42601; a missing collection
  directive or a `sparse=none` collection → 0A000; a missing collection →
  42P01; term/parameter violations → 22023. pgwire offers no sparse writes
  (AUB1 rows are unchanged).
- **Collection snapshot compaction for every backend (P5)**: clustered /
  flat-clustered / pq-flat previously could not be snapshotted, pinning
  collections-WAL retention at the old floor and reporting
  `unsupported_backend` in `/v1/wal/status`. All six backends now snapshot:
  - `vector_kind` gains 3 = built pq-flat compact state (centroids / codebook /
    cluster ids+codes; bit-identical restore, no retraining) and 4 =
    lazy-backend buffered raw rows. The extension is additive and bumps neither
    payload nor manifest version; older binaries cleanly reject unknown kinds
    at decode (`format_version` rules are unchanged).
  - clustered / flat-clustered snapshots = built raw vectors (boundary copies
    deduped by id) + pending rows; built pq-flat = compact built state +
    pending rows; unbuilt pq-flat = pending rows. Restore only refills the
    buffer and never auto-builds (matching a cold start via full WAL replay).
    `kind=4` is valid only for clustered/flat-clustered/pq-flat and `kind=3`
    only for pq-flat, cross-checked in both manifest and payload.
  - `export_snapshot_vectors` for lazy backends now returns built raw vectors
    plus pending rows; a built pq-flat has discarded its raw vectors, so export
    fails explicitly instead of silently returning partial data.

### Changed
- **Replication protocol v9**: extends v8's `collection_create` metric/HNSW
  semantics with `vector_dtype` (wire format unchanged). A v8 binary ignores
  the field and builds f16 collections as f32 (silent divergence), so v8/v9
  cannot be mixed and upgrades must be coordinated cluster-wide. **Never**
  replay v9 WAL on a v8 or older binary: f16 collections are reinterpreted as
  f32.
- **Replication protocol v10**: extends v9 with `quantization` (wire format
  unchanged). A v9 binary ignores the field and builds int8 collections as
  unquantized (silent divergence), so v9/v10 cannot be mixed and upgrades must
  be coordinated cluster-wide. **Never** replay v10 WAL on a v9 or older binary:
  int8 collections become unquantized graphs.
- **Replication protocol v11**: admits `rabitq` into the canonical
  `quantization` value domain on top of v10 (wire format unchanged). A v10
  binary treats the rabitq string as non-canonical and fails fast during
  replay (no silent downgrade) and rejects it at the handshake already; v10/v11
  cannot be mixed and upgrades must be coordinated cluster-wide. **Never**
  replay v11 WAL on a v10 or older binary.
- **Replication protocol v12**: adds sparse fields (`collection_create.sparse`
  and the optional `sparse` on upsert entries) on top of v11; the wire format
  remains additive. v11 and older binaries **silently ignore** those fields on
  decode (losing the sparse index while no snapshot exists), so v12 cannot be
  mixed with older versions and upgrades must be coordinated cluster-wide.
  **Never** replay v12 WAL on a v11 or older binary.
- **Snapshot version gates (sparse)**: collection snapshot payload v3 (flags
  bit3=`0x08` mandatory, v1/v2 must not set it; older binaries accept only
  1..=2 and reject v3 cleanly), with the sparse segment appended positionally
  after tombstones (term ascending, docs ascending within a term, cross-checked
  against tombstones); manifest entries gain an optional `sparse` (1=bm25) and
  a manifest containing any bm25 entry must use `format_version=4` (accepts
  1..=4, derived exactly). WAL sparse fields stay on the additive optional-field
  policy (`COLLECTION_WAL_VERSION` unchanged) with only the documented silent
  downgrade risk (take a snapshot before downgrading).
- **Snapshot version gates**: the standalone HNSW graph snapshot gains
  version=3 (records carry `[scale][bias][dims×i8][dims×f32 source]`,
  `FLAG_VECTOR_INT8=0x04`); v1/v2 binaries reject v3 with
  `UnsupportedVersion` and fall back to a full WAL replay. The collection
  snapshot payload stays version 2, with flags gaining bit1
  `VECTOR_INT8=0x02` (no new body bytes); the manifest entry gains an optional
  `quantization` (only int8 writes 1), and a manifest containing any int8 entry
  must use `format_version` 2 (older binaries accept only 1 and reject it
  explicitly instead of silently downgrading). RaBitQ adds graph snapshot
  version=4 (records carry `[id][norm][alpha][packed codes][dims×f32 source]`,
  with `FLAG_VECTOR_RABITQ=0x08` mandatory and the int8 bit2 forbidden; older
  binaries reject it with `UnsupportedVersion`). Collection `flags` gains
  bit2 `VECTOR_RABITQ=0x04` (mutually exclusive with bit1, no new body bytes),
  and the manifest `quantization` gains value 2 with `format_version` 3
  required for manifests containing any rabitq entry.
- **English documentation navigation**: cross-links in English documents now
  consistently target `.en.md` files, and `SECURITY(.en).md` points security
  verification to the deployment guide and alert runbooks (the former
  `docs/ENTERPRISE_READINESS.md` is not part of the public snapshot); broken
  relative references are fixed. (Credit: @Aqiu16717, public PR #4)

### Fixed
- **Dependency security remediation (36 Dependabot alerts: 10 critical /
  7 high / 18 medium / 1 low)**: Go SDK `golang.org/x/crypto` upgraded from
  v0.1.0 to v0.52.0 (`go` directive 1.17 -> 1.25; covers 18 x/crypto
  advisories including 8 critical); Python SDK now declares floors
  `requests>=2.34.2,<3` and `urllib3>=2.8.0,<3` (upper bounds mirror requests'
  own urllib3 constraint so consumers' existing caps are not widened) with
  `uv.lock` updated, and `requires-python` raised from >=3.9 to >=3.10 (the
  fixed versions require >=3.10 and Python 3.9 is EOL); both adapters move to
  >=3.10 with `houyidb>=0.2.0b8`. MCP deps `hono` 4.13.18,
  `@hono/node-server` 2.1.8, and `vitest` 4.1.12 (also fixing
  tinypool/@vitest/mocker critical/medium); `engines.node` is >=20, the
  shared floor of the production dependency `@hono/node-server` and Vitest 4,
  with `@types/node` aligned to ^20 and Node 22 LTS recommended in the README,
  and `npm audit` clean.
  Verification: Go `go test ./...`, Python wire contracts 7 passed, MCP clean
  `npm ci` + build + 43 tests. Note: local cargo audit still reports one
  `rustix` unsound warning (not a Dependabot entry; allowed warning).
- **Insufficient upper navigation beam in production-shaped legacy builds
  (residual islands measured by sm)**: at `c631596`, sm reproduced
  `clustered_1794652908806` with 100-row HTTP upserts (worker-threshold
  drains): recall@10 stayed at 0.9656 at ef=16/64/256/512, q45/q62 returned
  the wrong cluster (similarity 0.11-0.12 vs truth 0.95, ef-independent),
  while L0 was 100% reachable, the sweep added zero bridges, and neither a
  cold restart nor retriggering the sweep changed anything. Rebuilding the
  corpus bit-for-bit with mulberry32 localized it: with 3118 layer-1 nodes,
  a fixed upper beam of 64/128 cannot route the greedy descent from the
  entry to some clusters' in-layer representatives (layer >=2 and L0 are
  fully connected — the defect is purely insufficient beam coverage *within*
  a layer), and layer 1 additionally had 12 reachability orphans. Two fixes:
  (1) the upper beam now scales with layer size — estimate `nodes/m^layer`,
  width `max(HOUYI_HNSW_UPPER_EF, ceil(estimate/8))`, with the adaptive part
  capped at 4096 (the env value changed from a fixed width to a floor whose
  ceiling is the same 4096, and the new
  `HOUYI_HNSW_UPPER_EF_ADAPTIVE=0` switch disables adaptation for latency
  A/B; the build side stays greedy at 1); (2) the connectivity backstop now
  covers every layer, BFS-ing each layer and reconnecting broken components
  to their nearest reachable anchor. Layer populations are precomputed in a
  single node-map scan, and the healthy path walks only the reachable set,
  so the drain hook no longer rescans the whole graph once per layer under
  the write lock. Verification: on the exact production corpus legacy
  mean recall@10 rises from 0.9656 to **0.9953** (the only below-floor query
  is q30's TIE_EPS tied-neighbour noise), q45/q62 return to their truth
  clusters, default plan/apply stays at 1.0000, and all three official 50k
  cases pass. The gate adds a fourth 50k case,
  `legacy_prod_shape_batches_keep_all_layer_recall_50k` (exact sm corpus +
  the collection backend's `BuildConfig::default()` + ~300 rows per
  simulated worker tick), asserting 100% reachability on every layer,
  mean >= 0.99 at every ef, and q45/q62 recall >= 0.9. A new large-graph gate
  `large_graph_upper_navigation_keeps_cluster_at_quarter_million` (250k x 512 / 64
  clusters, ~15.6k layer-1 nodes) requires every returned id at ef=16..4096 to
  stay in the query's own cluster (no cross-cluster landings) and mean recall
  >= 0.99 at ef=8192; measured low-ef q8 recall can be 0 while its ids stay in
  cluster 8, and ef=8192 restores 1.0 — the large-graph bottleneck is L0 beam
  vs cluster size, not upper navigation. The adaptive cap is raised from 1024
  to 4096: 250k uses 1954 (~12.5%, below the cap) and 1M ~6.5%, preventing
  linear coverage decay. On this host, repeated 50k x 512 ef=256 A/B runs measure ~0.46-0.55
  ms with the fixed 64 beam vs ~0.64-0.90 ms adaptive (+0.18-0.35 ms,
  load-dependent);
  `HOUYI_HNSW_UPPER_EF_ADAPTIVE=0` rolls it back
  (`scripts/upper_beam_latency_ab.sh`), and nightly gains a 250k
  navigation job. All 50k/4653 recall
  gates also fix the ef sweep: `search_scored_with_ef_override` can only
  raise `ef_search`, so ef=16/64 previously replayed the inferred default
  beam; each round now pins 16/64/256/512 through the setter, and the new
  gate asserts per-tick flushed rows plus the legacy branch so it cannot
  silently measure the wrong build path.
- **Whole-cluster unreachable components left by serial/HTTP small-batch
  builds**: on clustered corpora, per-point serial inserts and HTTP 100-row
  requests accumulated into a drain can lose their only bridge to symmetric
  pruning, leaving components whose members all sit at >= `min_degree`
  (group repair only fills nodes below `min_degree` and cannot cover this
  "internally linked, no external bridge" shape). Fix: whenever a drain empties
  the build queue, BFS from the entry point at L0 and attach each unreachable
  component to the **vector-nearest reachable anchor** (spare capacity
  preferred, ties by id; up to 4 retries when the far side prunes the new edge),
  guaranteeing whole-graph reachability on the grouped and plan/apply paths
  alike. The official 50k gate gains an HTTP-shaped case (100-row requests
  drained at the worker threshold, covering a 300-row serial drain and a
  600-row grouped drain) and requires **100%** level-0 reachability.
- **HNSW upserts/deletes orphan third-party neighbors**: `remove_node` only
  strips the victim's reciprocal edges, so a neighbor can lose its last edge
  and fall to L0 degree 0 — unreachable from the entry point (the server
  backend model gate proved it on a 5-node rewrite sequence: node 8 fully
  isolated, zero hits). Fix: replace and delete paths run a local orphan
  repair over "batch ids + affected old neighbors" (the same id-window rule
  as group repair, cost scaling with affected neighbors rather than a full
  graph scan; still disabled by `HOUYI_HNSW_GROUP_REPAIR=off`). Adds three
  regression layers: server backend model differential (random
  upsert/delete/search streams over a small colliding id domain against a
  `HashMap` model, covering both the flat duplicate-id and graph-island
  incident classes), a kernel 5-node rewrite regression, and an HTTP
  stable-id reindex + cold-restart case (hnsw and flat); recall oracles now
  assert level-0 reachability and zero-degree invariants; the 50k
  clustered gate runs in nightly CI.
- **Duplicate-id counting and repeated top-k ids on the flat backend**: the
  flat write buffer only extended, so upserting the same id repeatedly
  accumulated rows — `indexed_count` came out inflated (3 after upserting
  1,2,1), one id occupied multiple top-k slots (`[1,1,2]`), and the stale
  vector kept being ranked. Fix: flat writes are now idempotent upserts —
  later writes within a batch win and rows with the same id are replaced
  across batches, matching the HNSW main graph's per-id update semantics.
  WAL replay and snapshot restore share this write channel, so restarts and
  restores dedupe as well. No read-side dedup cost is added. Storage keeps
  rows plus an id→row index behind a `RwLock`: upserts replace in place at
  O(batch), deletes are O(1) (`swap_remove` fixes up only the moved row), and
  concurrent read-only scans no longer serialize; `FlatStore` exposes
  read-only `rows`/`len`/`is_empty` accessors.
- **Whole-cluster islands at 50k×512 clustered corpora (both build paths)**:
  at 50k / 64 clusters (~781 points each) the adaptive group width (~195)
  stays below the cluster size, and both the plan/apply and legacy serial
  paths produced unreachable whole clusters (failing queries returned a
  completely wrong cluster with similarity 0.03–0.10 vs truth 0.95, and
  raising ef barely helped): the default plan/apply path without repair left
  only 359/50000 nodes forward-reachable (5003 new nodes with L0 degree 0).
  Two fixes: (1) the upper-layer descent beam is raised from per-layer greedy
  16 to 64 — the serial path's query #38 still landed in the wrong cluster at
  width 16 (wrong entry similarity 0.881 vs 0.0495 in the true cluster), while
  width 64 gives 0/64 wrong landings on both paths; (2) a post-commit orphan
  repair now runs by default (`HOUYI_HNSW_GROUP_REPAIR=4,16`; set `off` to
  disable): members with L0 degree <4 get bidirectional edges (L0 only) within
  a ±16 id window of the same batch. Plan/apply forward reachability recovers
  to 49958/50000 (99.92%; the remaining 42 nodes are small whole-cluster
  islands whose members already have degree ≥4, well above the ≥95% gate
  threshold), and the unrepaired serial path still reaches 96.18%; both hold
  mean recall 1.0000 at every ef and pass the ef=512 per-query floor 0.9.
  New release gate `./scripts/clustered_recall_50k_gate.sh` (two `#[ignore]`
  cases; debug takes minutes for the 50k brute-force truth plus build, so it
  stays out of per-commit `cargo test --all`). Gate and tunability: the upper
  beam is now configurable via `HOUYI_HNSW_UPPER_EF` (default 64, clamped to
  1..=1024, invalid values warn and fall back to 64; the build side
  intentionally keeps beam 1 — raising it only multiplies ef_construction
  cost and changes build semantics). Repair candidates are ranked in two tiers
  (healthy > low-degree); siblings already saturated at collection time are
  excluded outright (adding an edge to a saturated peer could trip its prune
  and revert the fresh edge), with a decision-time re-measurement catching
  candidates that fill up during the loop; window bounds use saturating
  arithmetic and the candidate reservation is capped by the actual per-node
  candidate count, so a typo'd huge window cannot reserve GBs under the write
  lock. When `GROUP_REPAIR` carries a window token that does not parse
  (e.g. `4,abc`) or extra fields (e.g. `4,16,32`), the whole value warns and
  falls back to the default instead of silently half-applying.
  The gates pin all three process-wide OnceLocks (PLAN_APPLY / GROUP_REPAIR /
  UPPER_EF) and assert the actually exercised build branch through the new
  `plan_apply` report field (preventing a silent fallback to the serial build
  while recall numbers still pass); `clustered_recall_50k_gate.sh` also clears
  KEEP_PRUNED / RERANK_EXPAND, runs single-threaded, and line-checks both `ok`
  results.
- **HNSW grouped build collapsed recall on clustered small corpora**: the
  plan/apply and legacy grouped paths planned each group against a fixed group
  width (default `chunk × group_chunks` = 1024) and only against the pre-group
  graph snapshot, so siblings inside a group were mutually invisible. On
  clustered data, in-batch siblings are each other's nearest neighbors, and
  symmetric pruning turned them into islands after commit: recall@10 stayed at
  6–50% for ef=16/64/256 regardless of ef, and level-0 BFS from the entry point
  reached only about half the nodes. Fix: adaptive group width via new
  `HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR` (default 256) and
  `HOUYI_HNSW_PLAN_GROUP_MIN` (default 16), with width =
  `min(configured, max(min, graph_len/divisor))`; `divisor=0` disables
  adaptation. Adaptively narrowed groups stay on the planned commit path (the
  legacy serial sub-branch redraws levels and discards the batch's pre-drawn
  levels, measured at only 91.3% mean recall). Two new regression gates
  (default and legacy; 4653×512 clustered corpus: level-0 reachability ≥95%,
  ef 16/64/256/512 recall mean ≥0.95; the per-query floor ≥0.9 is enforced at
  ef=512, which is stricter than this 512-d corpus's inferred default
  ef_search=256, and the other ef points additionally cap below-floor queries
  at 2 — with full 64-cluster coverage, measured #49 recall 0.7 and #55 recall
  0.0 are batch-boundary bridge variance, fully recovered at ef=512. The cap
  rejects a third broken cluster at every ef, while the historical fixed-width
  defect still fails the mean/count/reachability assertions immediately).
- **`HOUYI_HNSW_SEED_SERIAL_POINTS=0` did not work**: the positive-only
  `usize` parser turned an explicit zero into the default 256, making the
  documented "0 = disable" switch unreachable from the environment.
  `env_usize` is now derived from `env_usize_allow_zero`, so the two parsers
  cannot drift on absent/malformed handling.
- **One-shot build backends hid their state**: after `built=true`, clustered /
  flat-clustered / pq-flat send new writes only to the pending buffer where
  they never become searchable, and `pending_count` even returns to zero; no
  field previously expressed "rebuild required". `GET
  /v1/collections/{name}/status` now returns an optional `lazy_build`
  (`built` / `buffered` / `rebuild_required`; eager backends omit it). While a
  build is in flight `built` reports false so status never blocks on the build
  lock, and the docs state that the collection must be dropped and recreated for
  new data to become searchable.

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
