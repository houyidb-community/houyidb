# houyi-mobile on-device benchmarks (P0/P1 optimization log)

> [中文](HOUYIDB_MOBILE_BENCH.md)

Device: Huawei CLS-AL00 / Kirin 9010 / Android 12 / arm64-v8a / 11.3 GB RAM.
Method: core C ABI benchmark (no JNI facade), 100 queries × k=10, recall against
an on-device brute-force cosine top-10 ground truth; built with
`[profile.mobile]` (opt-level=s, LTO, strip); 3 alternating rounds, reporting
p50/p95 and recall. Battery was 24–37% and temperature 31–36 °C during runs;
background apps were stopped (Douyin / App Market / WeChat) but the environment
is still not ideal, so absolute numbers vary by ±3–10%.

## 128d × 100k (face-embedding workload, clustered synthetic data)

| Configuration | p50 / p95 | recall@10 | Snapshot | Cold load |
|---|---:|---:|---:|---:|
| flat-clustered f16 (default per_k=2500) | 1.44–1.60 ms / 2.0–2.5 ms | 0.9660 | 25.9 MB | 151–165 ms |
| flat-clustered i8 (default) | 1.21–1.38 ms / 1.7–2.0 ms | 0.9570 | 13.8 MB | 124–135 ms |
| clustered-hnsw f32 (before P1.5, raw rows + rebuild) | 2.49 ms / 2.96 ms | 0.9350 | 50.8 MB | **10.6 s** |
| clustered-hnsw f32 (after P1.5, built state) | 3.00 ms / 6.09 ms | 0.9350 | 77.4 MB | **243 ms** |

## 512d × 50k (CLIP reference, clustered synthetic data)

| Preset | p50 / p95 | recall@10 | Snapshot | Cold load |
|---|---:|---:|---:|---:|
| balanced (default) | 2.96–3.41 ms / 3.65–6.52 ms | 0.8570 | 50.8 MB | 300–361 ms |
| fast | 1.21–1.33 ms / 1.92–3.26 ms | 0.5110 | 50.8 MB | 293–371 ms |
| high_recall (f16) | 6.17–7.10 ms / 8.15–11.67 ms | **0.9990** | 50.8 MB | 297–305 ms |
| balanced (i8) | 2.21–2.28 ms / 2.50–3.76 ms | 0.8560 | 26.0 MB | 235–238 ms |
| high_recall (i8) | 4.15–4.17 ms / 4.82–5.15 ms | **0.9860** | 26.0 MB | 236–242 ms |

nprobe sweep (512d×50k, f16, one build, runtime `set_search_options`):

| nprobe | p50 | recall@10 |
|---:|---:|---:|
| 35 | 1.74 ms | 0.6060 |
| 78 | 2.24 ms | 0.7640 |
| 112 (build-time adaptive) | 3.31 ms | 0.8570 |
| 140 | 4.17 ms | 0.9130 |
| 168 | 4.77 ms | 0.9570 |
| 196 | 5.69 ms | 0.9810 |
| 224 (all clusters) | 6.53 ms | 0.9990 |

## Full-battery confirmation (100%, 31 °C, background stopped)

- 512d×50k (3-round medians): f16 balanced 3108 µs / 0.857, f16 high_recall
  6270 µs / **0.999**; i8 balanced 2146 µs / 0.856, i8 high_recall
  3885 µs / **0.986**.
- clustered-hnsw 128d×100k cold load: 237.0 / 240.3 / 239.5 ms (median 239 ms),
  recall 0.9350.

## Cold-load optimization (P2 follow-up, 128d×100k, alternating A/B in one session)

| Index | Before (warm median) | After | Speedup |
|---|---:|---:|---:|
| flat-clustered f16 (25.9 MB snapshot) | 176 ms | **95.5 ms** | 1.8× |
| flat-clustered i8 (13.8 MB snapshot) | 148.5 ms | **46.5 ms** | 3.2× |

How: decode now reads metadata sequentially and memcpys whole cluster payloads in
parallel (f16 keeps its u16 bit pattern, i8 stays bytes) instead of converting
element by element; the norm cache rebuild is parallel too. The snapshot format
and CRC validation are unchanged, and old snapshots still load. Against USearch
f32 64.5 MB at 53–97 ms, f16 is now in the same range and i8 is faster in
absolute terms.

(Persisting the norm cache into the snapshot was also tried: 80 vs 80 ms median
in A/B, no gain, reverted.)

## Cold-load round 2 (mmap + parallel CRC + no zero-init)

Building on the parallel bulk decode above:

- File snapshots load through a read-only mmap instead of a whole-file `read`,
  falling back to `fs::read` when mapping is unavailable.
- CRC32 is computed in parallel chunks and combined in order; the result is
  bit-identical to a single pass.
- Cluster payloads are no longer zero-initialized before being fully
  overwritten; `set_len` after the copy removes a 26 MB-class memset.
- The trailing-bytes check now runs before decoding, so malformed snapshots no
  longer pay the full decode cost.

Same-session alternating A/B (92% battery, 28 °C, background stopped; 6 loads
per round, median of the last 5):

| Index | Previous round | This round | Speedup |
|---|---:|---:|---:|
| flat-clustered f16 (26.6 MB snapshot) | 77.8 ms | **35.7 ms** | 2.2× |
| flat-clustered i8 (14.2 MB snapshot) | 38.3 ms | **27.8 ms** | 1.4× |

Cumulative versus the original per-element decode (176 / 148.5 ms, measured in
an earlier session; indicative only): about **4.9× / 5.3×**. Peak RSS is close between versions (f16 87 MB / i8 51 MB)
because the decoded buffers are still owned; the mmap gain is one fewer
whole-file copy and heap allocation, and mapped pages are file-backed and
reclaimable. See the next section for the exact same-data USearch comparison.

### USearch cold-load comparison (same device, same 100k×128 vectors)

Houyi exported the base vectors; USearch v2.26.6 built f32/f16/i8 indexes from the
same file. Three alternating rounds, median of 15 valid samples each. USearch's
`load` deserializes into RAM; `view` is a zero-copy mmap that defers page faults
to first access:

| Open mode | File | load | First query | Ready-to-query |
|---|---:|---:|---:|---:|
| Houyi f16 (28 °C) | 26.6 MB | 35.7 ms | ~0.8 ms | ~36.5 ms |
| Houyi f16 (31 °C) | 26.6 MB | 52.7 ms | ~0.9 ms | ~53.4 ms |
| Houyi i8 (28 °C) | 14.2 MB | 27.8 ms | ~0.8 ms | ~28.6 ms |
| Houyi i8 (31 °C) | 14.2 MB | 30.7 ms | ~1.1 ms | ~31.8 ms |
| USearch f16 `load` | 40.5 MB | 38.4 ms | 0.8 ms | 39.2 ms |
| USearch i8 `load` | 27.7 MB | 32.9 ms | 0.5 ms | 33.3 ms |
| USearch f32 `load` | 66.1 MB | 52.5 ms | 0.6 ms | 53.1 ms |
| **USearch f16 `view`** | 40.5 MB | 7.3 ms | 1.9 ms | **9.2 ms** |
| **USearch i8 `view`** | 27.7 MB | 8.5 ms | 1.1 ms | **9.7 ms** |
| **USearch f32 `view`** | 66.1 MB | 7.7 ms | 3.0 ms | **10.8 ms** |

Conclusions:

- Against `load` (deserialize into RAM): Houyi f16/i8 is slightly faster or on
  par at 28 °C; at 31 °C f16 falls behind as the CPU throttles, while i8 stays
  slightly ahead. USearch's load numbers are temperature-insensitive (38.4 /
  32.9 / 52.5 ms across rounds), whereas Houyi's eager decode depends more on
  CPU frequency.
- Against `view` (zero-copy mmap): **Houyi is 3–6× slower.** USearch defers the
  page-fault cost to the first query (1–3 ms), so it is ready in 9–11 ms.
- View mode (next section) brings loading level with USearch; time-to-first-query
  is still 1–2 ms behind, but peak memory is 40%+ lower.

## Cold-load round 3: view mode (mmap + per-cluster lazy decode)

New `loadView` / `houyi_mobile_db_load_view`: map the snapshot read-only and
parse metadata only; flat-clustered built-state clusters decode and cache on the
first search that touches them. The first write decodes only the clusters it
touches and their norms; saving an unwritten view reuses the raw built-state
bytes, while a written index is re-encoded.

Same-session alternating A/B (100k×128, 31 °C, warm page cache; median of 15
valid samples):

| Index | eager ready | **view ready** | eager peak RSS | **view peak RSS** | USearch view |
|---|---:|---:|---:|---:|---:|
| f16 | 52.3 ms | **11.5 ms** | 85.5 MB | **27.0 MB** | 9.4 ms / 48.1 MB |
| i8 | 32.0 ms | **10.1 ms** | 49.5 MB | **21.0 MB** | 9.0 ms / 35.8 MB |

- Load alone: Houyi view 7.5 ms (f16) / 7.4 ms (i8), level with USearch view's
  7.4–7.8 ms.
- First query: Houyi decodes the ~60 probed clusters (2.3–4 ms), while USearch's
  HNSW only touches the graph path (1.2–2.1 ms); total startup differs by 1–2 ms.
- Peak RSS is 41–44% lower than USearch view and 2.4–3.2× lower than eager.
- On truly cold storage both read from flash; the device cannot drop caches
  without root, but first-iteration samples were ~55 ms (view) and 46–70 ms
  (eager), indicative only.
- Cost: view mode skips the whole-file CRC and the file must not be truncated or
  overwritten in place while mapped; see manual section 2.8.
- Measurement note: Android `dumpsys meminfo` counts mapped file pages toward
  PSS/RSS, so total PSS does not necessarily drop in an app; native heap
  (anonymous memory) and a tighter peak distribution are view mode's main wins.

## Thread-count sweep (128d×100k, RAYON_NUM_THREADS, 3 runs each)

| Threads | f16 p50 median | i8 p50 median | Build |
|---:|---:|---:|---:|
| 2 | 3497 µs | 1659 µs | 11.4–12.5 s |
| 4 | 1667 µs | 1068 µs | 6.3–7.1 s |
| 6 | **1332 µs** | 1114 µs | 6.0–7.3 s |
| 8 (default) | 1530 µs | 1198 µs | 5.3–5.9 s |

Conclusion: 2 threads are unusable; 4/6/8 differ by 10–15% on queries and are
temperature-sensitive (the same configuration varied by up to 40% across
rounds), so there is **no stable winner**. Builds are always fastest with 8
threads, so the library keeps the default and apps may pick a count per device
via `houyi_mobile_set_threads`.

## Verdicts

### Effective

- **P0 in-memory snapshot**: `saveBytes/loadBytes` and `save_buffer/load_buffer`
  create no temp files and reject corrupt bytes; the on-device `bufcheck` passed
  (size query, short buffer, u64 id, bad magic).
- **P0 16 KB pages**: `llvm-readelf -l` reports `Align=0x4000` for all four LOAD
  segments.
- **P1.4 nprobe / presets**: 512d×50k high_recall reaches 0.999 (f16) / 0.986
  (i8); the default preset matches the historical baseline; an explicit nprobe
  is no longer silently squeezed back by the candidate budget.
- **P1.5 clustered-hnsw built-state snapshot**: 128d×100k cold load
  **10.6 s → 243 ms** (239 ms on the full-battery rerun) with unchanged recall;
  the snapshot grows from 50.8 MB to 77.4 MB (boundary replication + graph links).

### P2 additions (APIs, no latency change)

- Batch search `houyi_mobile_db_search_batch_u64` / Kotlin `searchBatch`: one
  lock for many queries, unchanged per-query latency and recall.
- Diagnostics `houyi_mobile_db_collection_stats` /
  `houyi_mobile_db_memory_usage` / Kotlin `collectionStats` / `memoryUsage`.
- Thread count `houyi_mobile_set_threads` / Kotlin `setThreads` (see the table
  above).

### Ineffective (reverted, not kept)

- **P1.3 fused f16 dot** (convert + dot in one loop, no scratch): f16 p50 went
  from 1.56–1.62 ms to **2.78–2.91 ms** (~1.8× slower). `half`'s vectorized
  conversion plus the hand-written NEON dot product wins.
- **P1.3 variant** (`map_init` scratch reuse only): f16 median 1534 → 1522 µs
  (−0.8%, within noise); i8, whose code did not change, drifted from 1368 to
  2109 µs, so the run carried no usable signal.
- **P1.6 i8 dotprod** (ARMv8.2 `vdotq_s32` plus query quantization): i8 p50
  median 1228 → 1234 µs (+0.5%) with recall 0.957 → 0.955. The first-round
  "−11%" was thermal drift on the previous run (1384 µs). Zero gain for an added
  quantization error, so the whole path was reverted.

## Guidance

- Below 10k vectors: plain `flat` (f32) is simpler; flat-clustered only wins
  clearly from ~50k upward.
- 50k–100k: `flat-clustered` f16 by default; use i8 under memory pressure
  (100k×128d snapshot is 13.8 MB).
- High-dimensional clustered data (512d) with insufficient default recall: try
  the `high_recall` preset (all clusters), then pick a latency/recall point from
  the sweep table via `set_search_options(nprobe=…)`.
- `clustered-hnsw` is for scenarios that need graph snapshots: loading is fast
  now, but its snapshot is larger and its default recall/latency still trail
  flat-clustered, so it stays out of the default recommendation.
