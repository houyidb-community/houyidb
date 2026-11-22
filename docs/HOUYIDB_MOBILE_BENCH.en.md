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
