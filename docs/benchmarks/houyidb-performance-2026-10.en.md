# HouyiDB Local Performance Measurements (2026-10)

> [中文](houyidb-performance-2026-10.md)

> This is a **single-machine engineering measurement record** comparing
> HouyiDB configurations (1×/2×, old/new graph-merge path, filter cache).
> It contains **no third-party comparison data** and is not an evaluation of
> any third-party product. The numbers hold only for the versions, parameters,
> machine, and dates stated here.

## 1. Environment and Method

| Item | Value |
|------|-------|
| Machine | Apple Silicon (arm64), 8 cores, 16GB RAM, macOS, AC power |
| Version | v0.2.0-beta.5 candidate builds (tables come from different dates; see labels) |
| Tool | VectorDBBench; method and parameters in `VDBBENCH_PROTOCOL.en.md` |
| Shared parameters | Non-streaming: k=10, m=24, ef_construction=400, COSINE, load_concurrency=8 |
| Query concurrency | 1 / 8 / 16, 30s each; ef_search swept per scenario |
| Build tuning | `PARALLEL_BATCH_BUILD=1`, `MAX_BUILD_BATCH=8192`, `SEED_SERIAL_POINTS=256` |

Each dataset is loaded once, then `--skip-load` runs different `ef_search`
values. The data directory is cleared and the server restarted before each
load. **Different tables come from different builds or code paths; they are
A/B comparisons within their own scenario and must not be combined into a
single score.**

Level-0 connectivity has two modes:

- **2×M (current default)**: recall-first, compatible with existing 2× graph
  snapshots;
- **1×M**: higher-QPS mode, slightly lower recall (about 0.9pp on Cohere).

Do not switch modes on an existing graph snapshot; confirm snapshot/WAL
integrity first or accept a full replay.

## 2. OpenAI 1536D / 50K

| Item | 2× (2026-10-22) | 1× (2026-10-22) |
|------|-----------------|------------------|
| Load | 81.2s (insert 42.6 + optimize 38.6) | **57.2s** (38.9 + 18.3) |
| ef64 recall | 0.9999 | 0.9999 |
| ef64 QPS (c8 / c16) | 426.1 / 400.3 | **686.0 / 680.3** |
| ef128 QPS | 403.9 | **663.5** |
| ef256 QPS | 411.8 | **652.6** |
| ef64 p99 | 9.6ms | **6.0ms** |

On the same dataset, 1× improves QPS by roughly 58–64% over 2× with unchanged
recall.

## 3. Cohere 768D / 1M

| Item | 2× (2026-10-21) | 1× (2026-10-22) |
|------|-----------------|------------------|
| Load | 1653.7s (insert 1465.1 + optimize 188.6) | **1012s** (825 + 187) |
| ef128 recall | **0.9971** | 0.9883 |
| ef128 QPS (c8 / c16) | 610 / 614 | **894 / 864** |
| ef256 recall | **0.9971** | 0.9883 |
| ef256 QPS (c8 / c16) | 608 / 627 | **833 / 839** |

1× gives higher QPS at about 0.9pp lower recall; use 2× for recall-sensitive
workloads and 1× for high QPS.

## 4. Label Filter (Cohere 1M, 1% labels)

| Item | 2× (2026-10-22) | 1× + contiguous vector cache (2026-10-23) |
|------|-----------------|------------------------------------------|
| Load | 1624.1s | **1301.6s** |
| ef128 recall | 1.0000 | 1.0000 |
| ef128 QPS (c8 / c16) | 697 / 698 | **1161.9 / 907.3** |
| ef256 recall | 1.0000 | 1.0000 |
| ef256 QPS (c8 / c16) | 688 / 747 | **1176.4 / 784.2** |

For high filter hit rates, HouyiDB lazily builds a contiguous vector cache per
label (Cosine vectors pre-normalized), turning filtered search into a sequential
SIMD scan + top-k. The cache is invalidated and rebuilt after writes; labels
over 256MB fall back to the old path. The first run reached ef128 1361/1365 and
ef256 1321/1386 QPS; the 2026-10-23 rerun under background load measured
1161.9/1176.4 (conc8). Both runs are well above the same implementation's old
path.

## 5. Streaming (Cohere 100K, insert_rate=500/s, k=100)

Graph merging now defaults to "plan under a read lock + commit under a short
write lock": planning runs in parallel under the HNSW read lock, and commit
holds the write lock for only 3–7ms per batch, so searches can read the graph
during construction.

| Item | Old path 1× (10-22) | New path 1× (10-23) | New path 2× (10-23) |
|------|---------------------|---------------------|---------------------|
| Insert | 200.1s | 200.1s | 200.1s |
| Optimize | 0.07s | **0.01s** | **0.01s** |
| Checkpoints 50/80/100/110 | 100.8/206.3/211.2/294.0s | 101.0/**164.4/201.0**/282.5s | 101.0/**165.3/201.0**/284.5s |
| Mature recall (100/110) | .988 / .988 | .9876 / .9876 | **.996 / .996** |
| 50% serial wall time / p99 | 104s / 1.72s | **3.9s / 21ms** | 6.3s / 25ms |
| 100% QPS (c5 / c10) | 996 / 1221 | **1103 / 1355** | 740 / 887 |
| 110% QPS (c5 / c10) | 990 / 1175 | **1101 / 1353** | 748 / 894 |

Final gate-build rerun (2026-10-24): checkpoints 100.95 / 164.69 / 200.90 /
283.25s, recall .5148 / .8307 / .988 / .988, serial p99 22 / 24.9 / 4.9 /
4.7ms, 100/110% QPS 999/1160 and 980/1214, consistent with the table within
run variance.

### Early-recall accounting

VectorDBBench streaming ground truth is a **static full-set** ground truth,
while inserts follow a prefix of the dataset row order. Measured GT@10 coverage
on CohereSmall is 0.5030 for the first 50% of rows and 0.8031 for the first
80%. The old path's 50% recall of .954 came from queries blocked behind the
graph-merge write lock (1000 queries took 104s), so most ran after the library
was nearly complete. The new path finishes in 3.9s and measures the coverage
ceiling (.515/.826).

Therefore **early recall must not be compared across different search speeds**;
use mature recall at 100%/110%. The new path costs about +11% total wall time
for bulk loading (Cohere 1M: 1170s vs 1055s); set
`HOUYI_HNSW_PLAN_APPLY=0` for load-time-sensitive workloads.

## 6. Conclusions

1. **1× vs 2×**: 1× improves QPS by about 35–64% on OpenAI/Cohere, at about
   0.9pp lower Cohere recall. Keep 2× as the default (snapshot compatibility)
   and set 1× explicitly for high-QPS workloads.
2. **Filtering**: the contiguous vector cache raises ef128/ef256 QPS from about
   700 to 1100+, with recall held at 1.0; cache invalidation adds one rebuild
   cost after writes.
3. **Streaming**: plan/apply cuts the 50% phase's 1000 serial queries from 104s
   to 3.9s and p99 from 1.72s to 21ms, with mature recall unchanged (1× .9876,
   2× .996).
4. All results are single-machine, single-run measurements. When comparing
   reports, verify the machine, mode (1×/2×), `ef_search`, concurrency, and
   interface path match.

## 7. Reproduce

```bash
# Method and parameters: VDBBENCH_PROTOCOL.en.md. This repository publishes no
# third-party comparison data.
export VDB_ROOT=/path/to/VectorDBBench
python3 benchmarks/vectordb_bench_houyidb/install.py --vdb-root "$VDB_ROOT"

# Start houyi-server first; set 1× / PLAN_APPLY before starting the server
VDB_ROOT="$VDB_ROOT" ./benchmarks/vectordb_bench_houyidb/run_bench.sh

# 1× mode (before server start: HOUYI_HNSW_LAYER0_MULT=1)
# Old graph-merge path (before server start: HOUYI_HNSW_PLAN_APPLY=0)
```
