# HouyiDB Mobile 使用手册

> [English](HOUYIDB_MOBILE_MANUAL.en.md)

> 轻量级向量索引，面向 iOS / Android 原生集成。  
> C ABI 接口，编译产物为静态库 (`.a`) 或动态库 (`.so` / `.dylib`)。

---

## 1. 构建

### 1.1 编译静态库

```bash
# macOS / iOS (aarch64)
cargo build --release -p houyi-mobile --target aarch64-apple-darwin
ls target/release/libhouyi_mobile.a   # ~19MB (release LTO)

# iOS (交叉编译，需 Xcode)
cargo build --release -p houyi-mobile --target aarch64-apple-ios

# Android (交叉编译，需 NDK)
cargo build --release -p houyi-mobile --target aarch64-linux-android
```

### 1.2 链接到项目

- **iOS**: 将 `libhouyi_mobile.a` 拖入 Xcode，添加 `libresolv.tbd` 依赖
- **Android**: 放入 `app/src/main/jniLibs/arm64-v8a/`，通过 JNI 调用

---

## 2. C API 参考

### 2.1 创建索引

```c
void* houyi_mobile_create(uint32_t dims, const char* index_type);
```

| 参数 | 说明 |
|------|------|
| `dims` | 向量维度 (如 128, 768, 1536) |
| `index_type` | `"flat"` / `"flat-clustered"` / `"clustered-hnsw"` |
| 返回 | 索引句柄 (NULL 表示失败) |

**三种索引类型**：

| 类型 | 内存 | 延迟 | Recall | 适用 |
|------|------|------|--------|------|
| `flat` | 无压缩 | O(N) | 1.0 (精确) | N < 5K, 离线评估 |
| `flat-clustered` | f16（约 f32 一半）/ i8（约 f16 一半，另有 scale） | ~0.5–1ms | > 0.95 | **推荐移动端默认（f16；内存紧张用 i8）** |
| `clustered-hnsw` | 图结构 | ~1ms | > 0.99 | N > 50K, 高 recall |

### 2.2 插入向量

```c
int32_t houyi_mobile_insert(void* handle, uint64_t id, const float* vector, uint32_t len);
```

| 参数 | 说明 |
|------|------|
| `handle` | 索引句柄 |
| `id` | 向量唯一标识 |
| `vector` | 浮点数组指针 |
| `len` | 数组长度 (必须 = dims) |
| 返回 | 0 成功；-1 参数/维度/集合缺失；-2 语义拒绝（如 CH 建图后覆盖已有 id，需 rebuild） |

支持幂等覆盖：同一 `id` 重复插入时覆盖旧值。

> 建图后行为：
> - `flat-clustered`：增量插入按最近质心归簇，**立即可查**；同 id 覆盖为跨簇搬移/原位更新，计数不变。
> - `clustered-hnsw`：建图后允许追加**新 id**（立即可查）；覆盖已有 id 返回 -2，需**重建索引**后批量 upsert，避免陈旧副本。
> - `flat`：始终直接扫描存储，无需单独建图。
> - 再次调用 `build` 会从当前索引（含增量行）重新聚类，不依赖已释放的写入缓冲。

### 2.3 构建索引

```c
uint32_t houyi_mobile_build(void* handle);
```

> 仅 `flat-clustered` 和 `clustered-hnsw` 需要调用。`flat` 模式下无操作。

返回已索引的向量总数。

### 2.4 检索

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

| 参数 | 说明 |
|------|------|
| `results` | 预分配缓冲区，每结果占 2 个 float (id + distance) |
| `max_results` | results 数组可容纳的最大结果数 (≥ k) |
| 返回 | 实际写入的结果数 |

### 2.5 查询数量

```c
uint32_t houyi_mobile_count(void* handle);
```

### 2.6 释放索引

```c
void houyi_mobile_destroy(void* handle);
```

### 2.7 ABI v2：批量写入 / u64 结果 / 显式配置

```c
void* houyi_mobile_create_v2(uint32_t dims, const char* index_type,
                             uint8_t metric, uint8_t dtype); // NULL 非法
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

- `metric`: 0=L2、1=Cosine、2=IP、3=L1；`dtype`: 0=f32、1=f16、2=i8（f16/i8
  仅 `flat-clustered`；i8 每维 1 B + 每向量 4 B scale）。旧 `create`/`db_collection` 保持历史默认（flat=L2/f32、
  flat-clustered=Cosine/f16、clustered-hnsw=Cosine/f32）。
- 旧检索写 `float` id，超过 2^24 会丢精度；**新代码一律用 `*_u64` 版本**。
- 批量写入一次加锁、按顺序应用；返回 0/-1/-2，失败前已写入的行不回滚。
- 同名 Collection 改用不同配置时 `db_collection_v2` 返回 -1（不做静默复用）。

### 2.8 保存 / 加载数据库

```c
int32_t houyi_mobile_db_save(void* db, const char* path);  // 0 成功，-1 失败
void*   houyi_mobile_db_load(const char* path);            // NULL 表示失败
```

- 保存整个 `MobileDB`（多 collection + 标量页）：先写同目录临时文件并 `fsync`，再
  `rename` 覆盖目标；文件含 magic/版本/长度/CRC32，损坏、截断、错版本一律拒绝。
- `flat-clustered` 保存 **f16/i8 构建态**（质心 + 分簇 id/向量），加载**不重建**；
  `clustered-hnsw` 自 v4 起保存**构建态**（质心 + 每簇图 + 边界复制），加载同样
  不重建（100k 冷加载从 10.6 s 降到亚秒级）；`flat` 保存原始行。
- 快照版本 v4 在 v3 基础上只新增 clustered-hnsw 构建态标签；旧二进制对 v4
  干净拒绝，新二进制仍可读取 v1–v3（v3 的 clustered-hnsw 原始行按旧逻辑重建）。
- 不引入 WAL：保存点由调用方控制；建议在写批次结束时调用，避开 UI 线程。
- 实测（host，100k×192 f16）：文件 37.6 MB，save 0.10 s，load 0.064 s；
  真实 7.3k×512：文件 7.3 MB，save 0.023 s，load 0.012 s。真机数值以实测为准。

**内存快照（不落临时文件，供 Vault 加密）**：

```c
int32_t houyi_mobile_db_save_buffer(void* db, uint8_t* out, size_t capacity, size_t* written);
void*   houyi_mobile_db_load_buffer(const uint8_t* bytes, size_t len);
```

- 与文件路径共用同一格式与校验（magic/版本/长度/CRC32）；坏字节返回 NULL。
- 返回码：`0` 成功、`-1` 参数非法（db/written 为 NULL）、`-2` 缓冲区不足。
  `*written` 成功时为字节数，`-2` 时为所需容量；用
  `save_buffer(db, NULL, 0, &written)` 查询容量。
- 缓冲区始终归调用方；本函数不分配/释放内存，也不保留引用。`bytes` 只在
  `load_buffer` 调用期间被读取，返回后可立即加密或清零。
- Kotlin：`HouyiDB.saveBytes(handle): ByteArray` / `HouyiDB.loadBytes(bytes): Long`。

**view 模式加载（低启动延迟 / 低峰值内存）**：

```c
void* houyi_mobile_db_load_view(const char* path);  // NULL 表示失败
```

- 只读 `mmap` 快照，加载只解析元数据；`flat-clustered` 构建态的簇向量在该簇
  第一次被检索时解码并缓存。100k×128 f16 实测：加载约 7 ms、到首次查询约
  11 ms，峰值 RSS 27 MB（eager 加载为 86 MB）。
- **不计算整文件 CRC**（结构校验仍完整）；需要完整性校验时用
  `houyi_mobile_db_load`。映射期间**不得原地截断或改写该文件**；`save` 的
  「临时文件 + rename」是安全的，Vault 等宿主原地覆写请改用 `load_buffer`。
- 首次写入只解码本次会触碰的簇及其范数，其余簇保持惰性。保存时：未写入过的
  视图直接复用原始构建态字节；写入后按当前数据重新编码。
- 已解码的簇会常驻缓存；查询覆盖全部簇后内存占用接近 eager，但映射页是文件页、
  可被内核回收。只要仍有簇未解码，快照字节源（mmap 或回退缓冲）就保持存活；
  全部簇解码并写入后自动释放。
- 仅 `flat-clustered` 构建态惰性解码；`flat` / `clustered-hnsw` 仍按原方式加载。
- view 模式要求磁盘上是**明文快照**；Vault/加密快照不能直接使用（继续用
  `loadBytes` 的 eager 路径，或等待后续加密 view）。不要把解密后的明文写入磁盘。
- Android `dumpsys meminfo` 口径下，mmap 的文件页会计入 PSS/RSS；view 的收益主要
  体现在 native heap（匿名内存）与峰值分布的收敛，总 PSS 不一定下降。
- Kotlin：`HouyiDB.loadView(path): Long`。

### 2.9 检索预算与召回预设

```c
int32_t houyi_mobile_db_set_search_options(void* db, const char* name,
                                           uint32_t per_k, uint32_t nprobe, uint32_t ef_search);
int32_t houyi_mobile_db_set_search_preset(void* db, const char* name, uint8_t preset);
int32_t houyi_mobile_db_set_build_nprobe(void* db, const char* name, uint32_t nprobe);
```

- 三个参数任一为 0 表示「保持不变」；越界或集合缺失返回 -1。
- `nprobe` 是路由簇数。构建期比例在 512d×50k 这类高维簇状数据上可能不足，
  可在 `build` 前用 `set_build_nprobe` 覆盖；已建图时用 `set_search_options`
  的运行期 nprobe 直接提高（所有簇都保存了自己的数据，提高只会多扫描）。
- `ef_search` 仅对 `clustered-hnsw` 生效（逐簇写入图实例）。
- 预设：

| 预设 | per_k | nprobe | ef_search（CH） | 适用 |
|---|---:|---|---:|---|
| `0` fast | 500 | `max(4, 簇数/16)` | 64 | 首屏/交互 |
| `1` balanced | 2500 | 构建期值 | 构建期值 | 默认 |
| `2` high_recall | 5000 | 全部簇 | 512 | 512d×50k 等难例 |

- 显式 nprobe 优先于 `per_k` 的预算收敛（不会被预算压回去）；`per_k=0` 的
  C ABI 语义是「保持不变」，恢复默认请用 balanced 预设。
- 512d×50k 真机实测的 recall–延迟曲线（f16，k=10）：nprobe 112→0.857 /
  3.3 ms，140→0.913 / 4.2 ms，168→0.957 / 4.8 ms，196→0.981 / 5.7 ms，
  224（全部）→0.999 / 6.5 ms。介于 fast 与 high_recall 之间时直接用
  `set_search_options(nprobe=…)` 选点。

```kotlin
HouyiDB.setSearchPreset(db, "faces", HouyiDB.SearchPreset.HIGH_RECALL)
HouyiDB.setSearchOptions(db, "faces", perK = 0, nprobe = 32, efSearch = 0)
HouyiDB.setBuildNprobe(db, "faces", 32)   // 下次 build 生效
```

### 2.10 批量检索与诊断

```c
int32_t houyi_mobile_db_search_batch_u64(void* db, const char* name,
                                         const float* queries, uint32_t dims,
                                         uint32_t nq, uint32_t k,
                                         HouyiMobileResult* results,
                                         uint32_t max_results_per_query, uint32_t* counts);
int32_t houyi_mobile_db_collection_stats(void* db, const char* name,
                                         HouyiMobileCollectionStats* out);
size_t  houyi_mobile_db_memory_usage(void* db);
```

- `search_batch_u64`：一次调用处理 `nq` 条查询，结果按行存放（第 i 条占
  `results[i*max_results_per_query..]`），`counts[i]` 写实际命中数。一次加锁，
  适合相册一次匹配多张脸；返回 0 成功、-1 参数/集合/维度错误。
- `collection_stats`：`count/dims/index_type/metric/dtype/built/nprobe/
  num_clusters`，配合 `memory_usage` 可在 App 侧选择 f16/i8 与预设。
- Kotlin：`searchBatch(...): List<List<SearchHit>>`、
  `collectionStats(...)`（`CollectionStats`）、`memoryUsage(handle)`。
- 批量内部仍按查询逐条执行（每次查询已按簇并行）；它的收益是减少 JNI/加锁
  次数，不会改变单查询的延迟与召回。
- 线程数：`houyi_mobile_set_threads(n)` / Kotlin `setThreads(n)`，须在首次
  检索/建图前调用（全局池只能初始化一次）。真机 128d×100k：2 线程明显变慢，
  4/6/8 查询中位数差约 10–15%，建图以 8 线程最快；建议建图用默认、交互检索
  按机型实测选定。

### 2.11 原有检索预算说明（flat-clustered 余弦）

```c
int32_t houyi_mobile_set_search_target(void* handle, uint32_t per_k);
int32_t houyi_mobile_db_set_search_target(void* db, const char* name, uint32_t per_k);
```

- `per_k` 是「每个 k 的候选预算」：实际扫描候选约 `k * per_k`，默认 **2500**
  （k=10 时约 25k 候选，保证高召回）。
- 仅对 `flat-clustered` + Cosine 生效；其他后端返回 0 但无效果。Kotlin：
  `HouyiDB.setSearchBudget(handle, name, perK)`。
- 人脸场景实测（真实 7.3k×512，host）：`per_k=500` 时 p50 ≈ 0.30 ms、
  recall@10 = 0.994；默认 2500 时 p50 ≈ 0.43 ms、recall = 1.000。可按
  「延迟优先」或「召回优先」选择；该预算不写入快照，属于运行期设置。
- f16/i8 余弦路径已预计算候选/质心范数（每向量 +4 B，100k 约 0.4 MB），
  查询热循环只做一次点积；服务端默认 f32 路径不启用该缓存。
- 真机实测（华为 CLS-AL00，k=10，中位数）：

| 数据/配置 | 默认 per_k=2500 | per_k=500 | recall@10（500） |
|---|---:|---:|---:|
| 真实 7.3k×512 f16 | 1.448 ms / 1.000 | 1.127 ms | 0.994 |
| 真实 7.3k×512 i8 | 1.293 ms / 0.996 | 0.974 ms | 0.991 |
| 合成 100k×192 f16 | 2.125 ms / 0.955 | — | — |

  真机噪声大于主机，数值仅代表该机型量级；默认预算优先保召回，预算 500
  适合首屏/交互查询。

---

## 3. 平台示例

### 3.0 Android AAR

仓库新增 `houyi-mobile/android` Gradle library 工程与 JNI 桥：

```bash
./scripts/build_mobile_android_aar.sh
# 产物 houyi-mobile/android/build/outputs/aar/houyi-mobile-android-release.aar（约 404 KB）
./scripts/mobile_jni_smoke.sh   # 主机 JVM 冒烟，无需真机
```

Kotlin `cc.houyidb.mobile.HouyiDB` 提供 `open/close/createCollection/
createCollection(metric,dtype)/insert/insertBatch/build/count/search/save/load/
saveBytes/loadBytes/insertScalarsWithId/searchHybrid/setSearchBudget/lastError`；`search` 返回 `SearchHit(id: Long, score: Float)`，不再有 f32 id
精度问题。AAR 当前仅含 arm64-v8a。构建需要 SDK 34 / NDK ≥26 / Gradle 8.9
（wrapper 已入库）/ Java 17。native 库按 **16 KB 页**对齐
（`-Wl,-z,max-page-size=16384`，构建脚本用 `llvm-readelf -l` 断言 LOAD
`Align=0x4000`），满足 Android 15+ 的 16 KB 页设备要求。


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

// 使用
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

## 4. 完整流程

```
1. houyi_mobile_create(128, "flat-clustered")   // 创建索引
2. houyi_mobile_insert(...) × N                  // 插入 N 个向量
3. houyi_mobile_build(...)                       // 构建聚类索引
4. houyi_mobile_search(...) × M                  // M 次检索
5. houyi_mobile_destroy(...)                     // 释放内存
```

---

## 5. 性能调参

`flat-clustered` 模式的关键参数（编译前修改 `houyi-vector/src/cluster/mod.rs`）：

| 参数 | 默认 | 作用 |
|------|------|------|
| `ClusterConfig::kmeans_iters` | 15 | K-Means 迭代次数，越大聚类越精确 |
| `ClusterConfig::hnsw_ef_search` | 256 | 检索候选宽度 (仅 clustered-hnsw) |
| `PqConfig::m` | dims/16 | PQ 段数，越少精度越高但压缩率越低 |
| `PqConfig::k` | 256 | 每段码本大小 |

---

## 6. 与 MicroNN 对标

| 指标 | MicroNN | HouyiDB Mobile |
|------|:---:|:---:|
| 延迟 | < 7 ms | **3.93 ms** |
| Recall@100 | ≥ 90% | **0.937** |
| RAM | ~10 MB | **15.4 MB** |
| 向量存储 | 磁盘 | 磁盘 (mmap) |
| 接口 | Swift/ObjC | C ABI (Swift/Kotlin/Flutter) |

---

## 7. 混合检索 (向量 + 结构化过滤)

### 7.1 Swift 示例

```swift
let db = houyi_mobile_db_open()

// 创建 Collection
houyi_mobile_db_collection(db, "photos", 512, "flat-clustered")

// 插入向量 + 标量标签
houyi_mobile_db_insert(db, "photos", 1, vector, 512)
// 推荐：显式向量 id 写入，混合检索按 id 关联
houyi_mobile_db_insert_scalars_v2(db, "photos", 1,
    "[{\"col\":\"tag\",\"val\":\"text:beach\"},{\"col\":\"date\",\"val\":\"ts:1717200000\"}]")
// 旧接口 houyi_mobile_db_insert_scalars 仍按行号兼容；若已写入显式 id，
// 行号可能顺延到空闲位，混合场景请只用 v2（显式 id）

// 混合检索: tag=beach AND date>2024-06-01
// filters_json: [{"col":"tag","op":"eq","val":"text:beach"},{"col":"date","op":"gt","val":"ts:1717200000"}]
// 新代码用 u64 结果；过滤条件在 flat-clustered 候选扫描阶段下推
HouyiMobileResult results[10];
houyi_mobile_db_search_hybrid_u64(db, "photos", query, 512, 10, filters_json, results, 10)
```

### 7.2 标量值格式

| 类型 | 格式 | 示例 |
|:-----|:-----|:-----|
| 整数 | `int:42` | `"int:42"` |
| 浮点 | `float:0.8` | `"float:0.95"` |
| 文本 | `text:beach` | `"text:sunset"` |
| 时间戳 | `ts:1717200000` | `"ts:1717200000"` (unix 秒) |

### 7.3 过滤操作

| 操作 | 说明 |
|:-----|:-----|
| `eq` | 等于 |
| `gt` | 大于 |
| `lt` | 小于 |
| `gte` | 大于等于 |
| `lte` | 小于等于 |
