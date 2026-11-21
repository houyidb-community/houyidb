# HouyiDB Mobile User Manual

> [中文](HOUYIDB_MOBILE_MANUAL.md)

> Lightweight vector index for native iOS / Android integration.  
> C ABI interface; build artifacts are static (`.a`) or dynamic (`.so` / `.dylib`) libraries.

---

## 1. Building

### 1.1 Building the static library

```bash
# macOS / iOS (aarch64)
cargo build --release -p houyi-mobile --target aarch64-apple-darwin
ls target/release/libhouyi_mobile.a   # ~19MB (release LTO)

# iOS (cross-compile; requires Xcode)
cargo build --release -p houyi-mobile --target aarch64-apple-ios

# Android (cross-compile; requires NDK)
cargo build --release -p houyi-mobile --target aarch64-linux-android
```

### 1.2 Linking into your project

- **iOS**: Drag `libhouyi_mobile.a` into Xcode and add the `libresolv.tbd` dependency
- **Android**: Place it under `app/src/main/jniLibs/arm64-v8a/` and call it via JNI

---

## 2. C API Reference

### 2.1 Creating an index

```c
void* houyi_mobile_create(uint32_t dims, const char* index_type);
```

| Parameter | Description |
|------|------|
| `dims` | Vector dimensions (e.g., 128, 768, 1536) |
| `index_type` | `"flat"` / `"flat-clustered"` / `"clustered-hnsw"` |
| Return | Index handle (NULL indicates failure) |

**Three index types**:

| Type | Memory | Latency | Recall | Use case |
|------|------|------|--------|------|
| `flat` | No compression | O(N) | 1.0 (exact) | N < 5K, offline evaluation |
| `flat-clustered` | f16 (~half of f32) / i8 (~half of f16 plus scale) | ~0.5–1ms | > 0.95 | **Recommended default: f16; use i8 under memory pressure** |
| `clustered-hnsw` | Graph structure | ~1ms | > 0.99 | N > 50K, high recall |

### 2.2 Inserting vectors

```c
int32_t houyi_mobile_insert(void* handle, uint64_t id, const float* vector, uint32_t len);
```

| Parameter | Description |
|------|------|
| `handle` | Index handle |
| `id` | Unique vector identifier |
| `vector` | Pointer to the float array |
| `len` | Array length (must equal dims) |
| Return | 0 on success; -1 for argument/dimension/missing collection; -2 for semantic rejection (e.g. overwriting an existing id after a clustered-hnsw build; rebuild first) |

Supports idempotent overwrite: inserting the same `id` again overwrites the old value.

> After a build:
> - `flat-clustered`: incremental inserts are assigned to the nearest centroid and are **searchable immediately**; overwriting an existing id updates in place or moves it across clusters without changing the count.
> - `clustered-hnsw`: post-build inserts of **new ids** are accepted (immediately searchable); overwriting an existing id returns -2, so recreate the index before a batch upsert to avoid stale copies.
> - `flat`: always scans the store directly; no build step is needed.
> - Calling `build` again reclusters from the current index (including incremental rows) and does not depend on the released write buffer.

### 2.3 Building the index

```c
uint32_t houyi_mobile_build(void* handle);
```

> Only `flat-clustered` and `clustered-hnsw` need this call. It is a no-op in `flat` mode.

Returns the total number of indexed vectors.

### 2.4 Searching

```c
uint32_t houyi_mobile_search(
    void* handle,
    const float* query,
    uint32_t dims,
    uint32_t k,
    float* results,      // 输出缓冲区 [id0, dist0, id1, dist1, ...]
    uint32_t max_results  // 缓冲区容量 (最多返回 max_results 个)
);
```

| Parameter | Description |
|------|------|
| `results` | Preallocated buffer; each result occupies 2 floats (id + distance) |
| `max_results` | Maximum number of results the results array can hold (≥ k) |
| Return | Number of results actually written |

### 2.5 Counting vectors

```c
uint32_t houyi_mobile_count(void* handle);
```

### 2.6 Destroying the index

```c
void houyi_mobile_destroy(void* handle);
```

### 2.7 ABI v2: batch writes / u64 results / explicit config

```c
void* houyi_mobile_create_v2(uint32_t dims, const char* index_type,
                             uint8_t metric, uint8_t dtype); // NULL on invalid
int32_t houyi_mobile_insert_batch(void* handle, const uint64_t* ids,
                                  const float* vectors, uint32_t count, uint32_t dims);
typedef struct { uint64_t id; float score; } HouyiMobileResult; // sizeof=16
uint32_t houyi_mobile_search_u64(void* handle, const float* query, uint32_t dims,
                                 uint32_t k, HouyiMobileResult* results, uint32_t max_results);
int32_t houyi_mobile_db_collection_v2(void* db, const char* name, uint32_t dims,
                                      const char* index_type, uint8_t metric, uint8_t dtype);
int32_t houyi_mobile_db_insert_batch(void* db, const char* name, const uint64_t* ids,
                                     const float* vectors, uint32_t count, uint32_t dims);
uint32_t houyi_mobile_db_search_u64(void* db, const char* name, const float* query,
                                    uint32_t dims, uint32_t k,
                                    HouyiMobileResult* results, uint32_t max_results);
```

- `metric`: 0=L2, 1=Cosine, 2=IP, 3=L1; `dtype`: 0=f32, 1=f16, 2=i8
  (f16/i8 are flat-clustered only; i8 uses 1 B per dimension plus a 4-byte
  per-vector scale). Legacy `create`/`db_collection` keep their
  historical defaults (flat=L2/f32, flat-clustered=Cosine/f16,
  clustered-hnsw=Cosine/f32).
- The old search writes `float` ids and loses precision above 2^24; **use the
  `*_u64` variants for all new code**.
- Batch writes take one lock and apply in order; they return 0/-1/-2 and do
  not roll back rows written before a failure.
- Reusing a collection name with a different configuration makes
  `db_collection_v2` return -1 (no silent reuse).

### 2.8 Saving / loading a database

```c
int32_t houyi_mobile_db_save(void* db, const char* path);  // 0 success, -1 failure
void*   houyi_mobile_db_load(const char* path);            // NULL on failure
```

- Saves the whole `MobileDB` (collections + scalar pages) by writing a sibling temp
  file, fsyncing, then renaming over the target. The file carries magic, version,
  body length, and CRC32; corrupt, truncated, or wrong-version files are rejected.
- `flat-clustered` persists its **f16/i8 built state** (centroids + per-cluster
  ids/vectors) and loads **without rebuilding**; `clustered-hnsw` (snapshot v4)
  persists its **built state** (centroids + every cluster graph + boundary
  replication) and also loads without rebuilding (100k cold load drops from
  10.6 s to sub-second); `flat` persists raw rows.
- Snapshot v4 only adds the clustered-hnsw built-state tag on top of v3; older
  binaries reject v4 cleanly, while this build still reads v1–v3 (v3
  clustered-hnsw raw rows keep the old rebuild-on-load path).
- No WAL: the caller owns the save point. Save at batch boundaries, off the UI thread.
- Measured on host (100k×192 f16): file 37.6 MB, save 0.10 s, load 0.064 s;
  real 7.3k×512: 7.3 MB, 0.023 s, 0.012 s. Verify on device.

**In-memory snapshot (no temp files, ready for a Vault)**:

```c
int32_t houyi_mobile_db_save_buffer(void* db, uint8_t* out, size_t capacity, size_t* written);
void*   houyi_mobile_db_load_buffer(const uint8_t* bytes, size_t len);
```

- Shares the file format and validation (magic/version/length/CRC32); corrupt
  bytes return NULL.
- Returns `0` on success, `-1` for invalid args (NULL db/written), `-2` when the
  buffer is too small. `*written` is the byte count on success and the required
  capacity on `-2`; query it with `save_buffer(db, NULL, 0, &written)`.
- The buffer stays caller-owned; nothing is allocated, freed, or retained.
  `bytes` is read only during `load_buffer`, so the caller may encrypt or wipe
  it immediately afterwards.
- Kotlin: `HouyiDB.saveBytes(handle): ByteArray` / `HouyiDB.loadBytes(bytes): Long`.

### 2.9 Search budget and recall presets

```c
int32_t houyi_mobile_db_set_search_options(void* db, const char* name,
                                           uint32_t per_k, uint32_t nprobe, uint32_t ef_search);
int32_t houyi_mobile_db_set_search_preset(void* db, const char* name, uint8_t preset);
int32_t houyi_mobile_db_set_build_nprobe(void* db, const char* name, uint32_t nprobe);
```

- A zero value in any option means "leave unchanged"; out-of-range values or a
  missing collection return -1.
- `nprobe` is the routed cluster count. The build-time ratio can be too small on
  high-dimensional clustered data such as 512d×50k: override it with
  `set_build_nprobe` before `build`, or raise routing on a built index with
  `set_search_options` (every cluster keeps its own data, so raising it only
  scans more).
- `ef_search` applies to `clustered-hnsw` only (written into every cluster
  graph).
- Presets:

| Preset | per_k | nprobe | ef_search (CH) | Use case |
|---|---:|---|---:|---|
| `0` fast | 500 | `max(4, clusters/16)` | 64 | first screen / interactive |
| `1` balanced | 2500 | build-time value | build-time value | default |
| `2` high_recall | 5000 | all clusters | 512 | hard cases like 512d×50k |

- An explicit nprobe wins over the `per_k` budget convergence (it is never
  squeezed back); in the C ABI `per_k=0` means "unchanged", so use the balanced
  preset to restore defaults.
- Measured recall/latency curve on device (512d×50k, f16, k=10): nprobe
  112→0.857 at 3.3 ms, 140→0.913 at 4.2 ms, 168→0.957 at 4.8 ms,
  196→0.981 at 5.7 ms, 224 (all)→0.999 at 6.5 ms. Pick a point in between with
  `set_search_options(nprobe=…)` instead of the presets.

```kotlin
HouyiDB.setSearchPreset(db, "faces", HouyiDB.SearchPreset.HIGH_RECALL)
HouyiDB.setSearchOptions(db, "faces", perK = 0, nprobe = 32, efSearch = 0)
HouyiDB.setBuildNprobe(db, "faces", 32)   // applies to the next build
```

### 2.10 Original search-budget notes (flat-clustered cosine)

```c
int32_t houyi_mobile_set_search_target(void* handle, uint32_t per_k);
int32_t houyi_mobile_db_set_search_target(void* db, const char* name, uint32_t per_k);
```

- `per_k` is the candidate budget per requested k: roughly `k * per_k` candidates
  are scanned. The default is **2500** (about 25k candidates at k=10) to protect
  recall.
- It only affects `flat-clustered` + Cosine; other backends return 0 without
  changing behavior. Kotlin: `HouyiDB.setSearchBudget(handle, name, perK)`.
- Real-face host benchmark (7.3k×512): `per_k=500` gives p50 ≈ 0.30 ms and
  recall@10 = 0.994; the 2500 default gives p50 ≈ 0.43 ms and recall = 1.000.
  Choose latency-first or recall-first accordingly; this budget is runtime state
  and is not persisted in snapshots.
- The f16/i8 cosine paths precompute candidate/centroid norms (+4 B per vector,
  about 0.4 MB at 100k) so the hot loop does one dot product; the server's
  default f32 path does not enable this cache.
- Device benchmark (Huawei CLS-AL00, k=10, median):

| Data / configuration | Default per_k=2500 | per_k=500 | recall@10 (500) |
|---|---:|---:|---:|
| Real 7.3k×512 f16 | 1.448 ms / 1.000 | 1.127 ms | 0.994 |
| Real 7.3k×512 i8 | 1.293 ms / 0.996 | 0.974 ms | 0.991 |
| Synthetic 100k×192 f16 | 2.125 ms / 0.955 | — | — |

  Device noise is higher than on host; treat these as order-of-magnitude numbers
  for this phone. The default budget favors recall; 500 suits first-screen or
  interactive queries.

---

## 3. Platform Examples

### 3.0 Android AAR

The repo now ships a `houyi-mobile/android` Gradle library project plus a JNI
bridge:

```bash
./scripts/build_mobile_android_aar.sh
# output: houyi-mobile/android/build/outputs/aar/houyi-mobile-android-release.aar (~404 KB)
./scripts/mobile_jni_smoke.sh   # host-JVM smoke test, no device required
```

Kotlin `cc.houyidb.mobile.HouyiDB` exposes `open/close/createCollection/
createCollection(metric,dtype)/insert/insertBatch/build/count/search/save/load/
saveBytes/loadBytes/insertScalarsWithId/searchHybrid/setSearchBudget/lastError`; `search` returns `SearchHit(id: Long, score: Float)`, eliminating
the f32-id precision limit. The AAR currently contains arm64-v8a only. Build
requirements: SDK 34, NDK >= 26, Gradle 8.9 (wrapper included), Java 17. The
native library is **16 KB page aligned** (`-Wl,-z,max-page-size=16384`; the
build script asserts `Align=0x4000` for every LOAD segment via `llvm-readelf
-l`), as required by Android 15+ 16 KB-page devices.


### 3.1 Swift (iOS)

```swift
import Foundation

class HouyiDB {
    private var handle: UnsafeMutableRawPointer?
    let dims: UInt32 = 128

    init?(indexType: String = "flat-clustered") {
        guard let h = houyi_mobile_create(dims, indexType.cString(using: .utf8)) else {
            return nil
        }
        self.handle = h
    }

    func insert(id: UInt64, vector: [Float]) {
        guard let h = handle, vector.count == dims else { return }
        houyi_mobile_insert(h, id, vector, dims)
    }

    func build() -> UInt32 {
        guard let h = handle else { return 0 }
        return houyi_mobile_build(h)
    }

    func search(query: [Float], k: UInt32 = 10) -> [(UInt64, Float)] {
        guard let h = handle, query.count == dims else { return [] }
        var results = [Float](repeating: 0, count: Int(k * 2))
        let n = houyi_mobile_search(h, query, dims, k, &results, k)
        var out: [(UInt64, Float)] = []
        for i in 0..<Int(n) {
            out.append((UInt64(results[i*2]), results[i*2+1]))
        }
        return out
    }

    deinit {
        if let h = handle { houyi_mobile_destroy(h) }
    }
}

// Usage
let db = HouyiDB(indexType: "flat-clustered")!
db.insert(id: 1, vector: [/* 128 floats */])
db.insert(id: 2, vector: [/* 128 floats */])
let count = db.build()
print("Indexed \(count) vectors")

let results = db.search(query: [/* 128 floats */], k: 10)
for (id, dist) in results {
    print("id=\(id) dist=\(dist)")
}
```

### 3.2 Kotlin (Android JNI)

```kotlin
class HouyiDB(private val dims: Int, indexType: String = "flat-clustered") {
    private var handle: Long = 0

    companion object {
        init { System.loadLibrary("houyi_mobile") }
    }

    // JNI declarations
    private external fun nativeCreate(dims: Int, type: String): Long
    private external fun nativeInsert(handle: Long, id: Long, vector: FloatArray): Int
    private external fun nativeBuild(handle: Long): Int
    private external fun nativeSearch(handle: Long, query: FloatArray, k: Int): FloatArray
    private external fun nativeDestroy(handle: Long)

    init {
        handle = nativeCreate(dims, indexType)
        if (handle == 0L) throw RuntimeException("Failed to create index")
    }

    fun insert(id: Long, vector: FloatArray) {
        require(vector.size == dims)
        nativeInsert(handle, id, vector)
    }

    fun build(): Int = nativeBuild(handle)

    fun search(query: FloatArray, k: Int = 10): List<Pair<Long, Float>> {
        val raw = nativeSearch(handle, query, k)
        return raw.toList().chunked(2).map { Pair(it[0].toLong(), it[1]) }
    }

    protected fun finalize() { nativeDestroy(handle) }
}
```

---

## 4. Full Workflow

```
1. houyi_mobile_create(128, "flat-clustered")   // 创建索引
2. houyi_mobile_insert(...) × N                  // 插入 N 个向量
3. houyi_mobile_build(...)                       // 构建聚类索引
4. houyi_mobile_search(...) × M                  // M 次检索
5. houyi_mobile_destroy(...)                     // 释放内存
```

---

## 5. Performance Tuning

Key parameters for `flat-clustered` mode (modify `houyi-vector/src/cluster/mod.rs` before compiling):

| Parameter | Default | Purpose |
|------|------|------|
| `ClusterConfig::kmeans_iters` | 15 | Number of K-Means iterations; higher values yield more precise clustering |
| `ClusterConfig::hnsw_ef_search` | 256 | Search candidate width (clustered-hnsw only) |
| `PqConfig::m` | dims/16 | Number of PQ segments; fewer segments give higher accuracy but lower compression ratio |
| `PqConfig::k` | 256 | Codebook size per segment |

---

## 6. Comparison with MicroNN

| Metric | MicroNN | HouyiDB Mobile |
|------|:---:|:---:|
| Latency | < 7 ms | **3.93 ms** |
| Recall@100 | ≥ 90% | **0.937** |
| RAM | ~10 MB | **15.4 MB** |
| Vector storage | Disk | Disk (mmap) |
| Interface | Swift/ObjC | C ABI (Swift/Kotlin/Flutter) |

---

## 7. Hybrid Search (Vector + Structured Filtering)

### 7.1 Swift Example

```swift
let db = houyi_mobile_db_open()

// Create collection
houyi_mobile_db_collection(db, "photos", 512, "flat-clustered")

// Insert vector + scalar labels
houyi_mobile_db_insert(db, "photos", 1, vector, 512)
// Preferred: write with an explicit vector id; hybrid search associates by id.
houyi_mobile_db_insert_scalars_v2(db, "photos", 1,
    "[{\"col\":\"tag\",\"val\":\"text:beach\"},{\"col\":\"date\",\"val\":\"ts:1717200000\"}]")
// The legacy houyi_mobile_db_insert_scalars keeps row-index compatibility;
// once explicit ids exist the legacy index may advance to a free slot, so use
// the v2 explicit-id API for mixed workloads.

// Hybrid search: tag=beach AND date>2024-06-01
// filters_json: [{"col":"tag","op":"eq","val":"text:beach"},{"col":"date","op":"gt","val":"ts:1717200000"}]
// New code uses u64 results; filters are pushed into the candidate scan for
// flat-clustered.
HouyiMobileResult results[10];
houyi_mobile_db_search_hybrid_u64(db, "photos", query, 512, 10, filters_json, results, 10)
```

### 7.2 Scalar Value Formats

| Type | Format | Example |
|:-----|:-----|:-----|
| Integer | `int:42` | `"int:42"` |
| Float | `float:0.8` | `"float:0.95"` |
| Text | `text:beach` | `"text:sunset"` |
| Timestamp | `ts:1717200000` | `"ts:1717200000"` (unix seconds) |

### 7.3 Filter Operations

| Operation | Description |
|:-----|:-----|
| `eq` | Equal to |
| `gt` | Greater than |
| `lt` | Less than |
| `gte` | Greater than or equal to |
| `lte` | Less than or equal to |
