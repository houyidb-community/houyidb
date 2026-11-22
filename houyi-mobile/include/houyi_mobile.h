// HouyiDB Mobile C ABI 头文件（iOS/Flutter/自定义 JNI 桥均可直接使用）。
// EN: HouyiDB Mobile C ABI header (usable by iOS/Flutter/custom JNI bridges).
//
// 生命周期：houyi_mobile_create* 返回的指针用 houyi_mobile_destroy 释放；
// houyi_mobile_db_open/load 返回的指针用 houyi_mobile_db_close 释放。
// EN: Lifetime: release houyi_mobile_create* handles with
// houyi_mobile_destroy; release houyi_mobile_db_open/load handles with
// houyi_mobile_db_close.
#ifndef HOUYI_MOBILE_H
#define HOUYI_MOBILE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HouyiMobileResult {
    uint64_t id;
    float score;
} HouyiMobileResult;

void* houyi_mobile_create(uint32_t dims, const char* index_type);
// metric：0=L2 1=Cosine 2=IP 3=L1；dtype：0=f32 1=f16 2=i8
// （f16/i8 仅 flat-clustered）。
// EN: metric: 0=L2 1=Cosine 2=IP 3=L1; dtype: 0=f32 1=f16 2=i8
// (f16/i8 only for flat-clustered).
void* houyi_mobile_create_v2(uint32_t dims, const char* index_type,
                             uint8_t metric, uint8_t dtype);
int32_t houyi_mobile_insert(void* handle, uint64_t id, const float* vector, uint32_t len);
int32_t houyi_mobile_insert_batch(void* handle, const uint64_t* ids,
                                  const float* vectors, uint32_t count, uint32_t dims);
uint32_t houyi_mobile_build(void* handle);
// Legacy: results 必须容纳 2*max_results 个 float（[id,score] 成对，id 为
// f32，>2^24 会丢精度）；64 位 id 请用 houyi_mobile_search_u64。
// EN: Legacy: results must hold 2*max_results floats ([id,score] pairs; ids are
// f32 and lose precision above 2^24); use search_u64 for 64-bit ids.
uint32_t houyi_mobile_search(void* handle, const float* query, uint32_t dims,
                             uint32_t k, float* results, uint32_t max_results);
uint32_t houyi_mobile_search_u64(void* handle, const float* query, uint32_t dims,
                                 uint32_t k, HouyiMobileResult* results,
                                 uint32_t max_results);
int32_t houyi_mobile_set_search_target(void* handle, uint32_t per_k);
uint32_t houyi_mobile_count(void* handle);
void houyi_mobile_destroy(void* handle);

void* houyi_mobile_db_open(void);
int32_t houyi_mobile_db_collection(void* db, const char* name, uint32_t dims,
                                   const char* index_type);
// dtype：0=f32 1=f16 2=i8（f16/i8 仅 flat-clustered）。
// EN: dtype: 0=f32 1=f16 2=i8 (f16/i8 only for flat-clustered).
int32_t houyi_mobile_db_collection_v2(void* db, const char* name, uint32_t dims,
                                      const char* index_type, uint8_t metric,
                                      uint8_t dtype);
int32_t houyi_mobile_db_insert(void* db, const char* name, uint64_t id,
                               const float* vector, uint32_t len);
int32_t houyi_mobile_db_insert_batch(void* db, const char* name, const uint64_t* ids,
                                     const float* vectors, uint32_t count, uint32_t dims);
uint32_t houyi_mobile_db_build(void* db, const char* name);
// Legacy: results 同样必须容纳 2*max_results 个 float（成对 id/score，id 为 f32）。
// EN: Legacy: results must also hold 2*max_results floats (paired id/score,
// f32 ids).
uint32_t houyi_mobile_db_search(void* db, const char* name, const float* query,
                                uint32_t dims, uint32_t k, float* results,
                                uint32_t max_results);
uint32_t houyi_mobile_db_search_u64(void* db, const char* name, const float* query,
                                    uint32_t dims, uint32_t k,
                                    HouyiMobileResult* results, uint32_t max_results);
int32_t houyi_mobile_db_set_search_target(void* db, const char* name, uint32_t per_k);
uint32_t houyi_mobile_db_count(void* db, const char* name);
int32_t houyi_mobile_db_drop(void* db, const char* name);
int32_t houyi_mobile_db_insert_scalars(void* db, const char* name, const char* tags_json);
// 显式向量 id 的标量写入；同 id 覆盖整行（旧行值清空）。标签 JSON 非法返回 -1。
// EN: Scalar write with an explicit vector id; the same id replaces the whole
// row (old values are cleared). Invalid tags JSON returns -1.
int32_t houyi_mobile_db_insert_scalars_v2(void* db, const char* name, uint64_t id,
                                          const char* tags_json);
// results 必须容纳 2*max_results 个 float（成对 id/score，id 为 f32）。
// filters_json 为 NULL 或非法 JSON 时返回 0（与无命中同码）；空串/纯空白
// 表示无过滤，正常返回结果。
// EN: results must hold 2*max_results floats (paired id/score, f32 ids).
// A NULL or invalid filters_json returns 0 (same code as no matches); an empty
// or whitespace-only string means no filter.
uint32_t houyi_mobile_db_search_hybrid(void* db, const char* name, const float* query,
                                       uint32_t dims, uint32_t k, const char* filters_json,
                                       float* results, uint32_t max_results);
// u64 结果；results 必须容纳 max_results 个 HouyiMobileResult（按
// sizeof(HouyiMobileResult) 计算，不假定固定字节数）。
// 过滤仅在 flat-clustered 的候选扫描内下推，flat/clustered-hnsw 仍为后过滤
// （可能不足 k）。filters_json 为 NULL 或非法 JSON 时返回 0（与无命中同码）；
// 空串/纯空白表示无过滤，正常返回结果。
// EN: u64 results; `results` must hold max_results HouyiMobileResult entries,
// sized by sizeof(HouyiMobileResult) rather than a fixed byte count. Filters
// are pushed into candidate scanning only for
// flat-clustered; flat/clustered-hnsw still post-filter (result count may be
// below k). A NULL or invalid filters_json returns 0 (same code as no
// matches); an empty or whitespace-only string means no filter.
uint32_t houyi_mobile_db_search_hybrid_u64(void* db, const char* name, const float* query,
                                           uint32_t dims, uint32_t k,
                                           const char* filters_json,
                                           HouyiMobileResult* results,
                                           uint32_t max_results);
// 初始化检索线程数（1..=256）。须在首次检索/建图前调用：rayon 全局池只能
// 初始化一次，池已存在返回 -1（保持默认线程数）。
// EN: Initialize the search thread count (1..=256). Must be called before the
// first search/build: the rayon global pool can be initialized only once, so
// -1 is returned when a pool already exists (default count stays).
int32_t houyi_mobile_set_threads(uint32_t threads);
// 集合诊断（C 布局）。built：flat 恒 0；nprobe/num_clusters：未建图或 flat 为 0。
// EN: Collection diagnostics (C layout). `built` is always 0 for flat;
// nprobe/num_clusters are 0 when unbuilt or flat.
typedef struct {
    uint32_t count;
    uint32_t dims;
    uint8_t index_type; // 0=flat 1=flat-clustered 2=clustered-hnsw
    uint8_t metric;     // 同 create_v2
    uint8_t dtype;      // 0=f32 1=f16 2=i8
    uint8_t built;
    uint32_t nprobe;
    uint32_t num_clusters;
    uint32_t reserved;
} HouyiMobileCollectionStats;
// 填充诊断；0 成功、-1 参数非法/集合缺失。
// EN: Fill diagnostics; 0 on success, -1 for bad arguments/missing collection.
int32_t houyi_mobile_db_collection_stats(void* db, const char* name,
                                         HouyiMobileCollectionStats* out);
// 全库堆内存估算（字节）；db 为 NULL 返回 0。
// EN: Estimated heap usage across collections in bytes; 0 for a NULL db.
size_t houyi_mobile_db_memory_usage(void* db);
// 批量检索：results 按行存放，第 i 条查询占 results[i*max_results_per_query..]，
// counts[i] 写实际命中数（≤ max_results_per_query 且 ≤ k）。
// 返回 0 成功、-1 参数非法/集合缺失/维度不符。
// EN: Batch search: row-major results, query i owns
// results[i*max_results_per_query..]; counts[i] gets its hit count (≤
// max_results_per_query and ≤ k). Returns 0 on success, -1 for bad arguments,
// a missing collection, or a dimension mismatch.
int32_t houyi_mobile_db_search_batch_u64(void* db, const char* name, const float* queries,
                                         uint32_t dims, uint32_t nq, uint32_t k,
                                         HouyiMobileResult* results,
                                         uint32_t max_results_per_query, uint32_t* counts);
// 检索选项：per_k 候选预算、nprobe 路由簇数、ef_search（仅 clustered-hnsw）。
// 任一参数传 0 表示该项保持不变；越界/集合缺失返回 -1，成功返回 0。
// nprobe 建图后夹取到簇数；提高只会多扫描（所有簇都保存了自己的数据）。
// EN: Search options: per_k candidate budget, nprobe routed cluster count, and
// ef_search (clustered-hnsw only). Pass 0 to leave a field unchanged; returns
// 0 on success and -1 for out-of-range values or a missing collection. nprobe
// is clamped to the cluster count after a build; raising it only scans more.
int32_t houyi_mobile_db_set_search_options(void* db, const char* name, uint32_t per_k,
                                           uint32_t nprobe, uint32_t ef_search);
// 预设：0=fast（per_k 500、少量簇）、1=balanced（默认）、2=high_recall
// （per_k 5000、全簇路由、clustered-hnsw ef 512）。非法码返回 -1。
// EN: Presets: 0=fast (per_k 500, few clusters), 1=balanced (default),
// 2=high_recall (per_k 5000, all clusters, clustered-hnsw ef 512). Invalid
// codes return -1.
int32_t houyi_mobile_db_set_search_preset(void* db, const char* name, uint8_t preset);
// 下一次 houyi_mobile_db_build 使用的 nprobe；0 = 自适应公式。已建图需重建
// 才生效（运行期请用 set_search_options 的 nprobe）。
// EN: nprobe for the next houyi_mobile_db_build; 0 = adaptive formula. A built
// index needs a rebuild (use set_search_options nprobe at runtime).
int32_t houyi_mobile_db_set_build_nprobe(void* db, const char* name, uint32_t nprobe);
int32_t houyi_mobile_db_save(void* db, const char* path);
// 从文件加载数据库；失败返回 NULL。内部用只读 mmap：加载期间调用方不得让
// 其他进程/线程原地截断或改写该文件（「临时文件 + rename」原子替换是安全的；
// 必须原地覆写时请改用 houyi_mobile_db_load_buffer）。违反该前提可能 SIGBUS。
// EN: Load a database from a file; returns NULL on failure. Uses a read-only
// mmap internally: while loading, the caller must not let another process or
// thread truncate or overwrite the file in place (atomic temp-file + rename is
// safe; use houyi_mobile_db_load_buffer when an in-place overwrite is required).
// Breaking this precondition can raise SIGBUS.
void* houyi_mobile_db_load(const char* path);
// 内存快照：把整个数据库编码进调用方缓冲区，**不写任何临时文件**（加密/上传由
// 调用方负责）。返回 0 成功、-1 参数非法（db/written 为 NULL）、-2 缓冲区不足。
// *written：成功时为实际字节数；返回 -2 时为所需容量，调用方可扩容后重试。
// 查询所需容量：传 out=NULL、capacity=0 会返回 -2 并写入 *written。
// 缓冲区始终归调用方所有；本函数不分配也不释放内存，字节在调用返回后不再被引用。
// EN: In-memory snapshot: encode the whole database into a caller-provided
// buffer and write **no temp file** (encryption/upload is the caller's job).
// Returns 0 on success, -1 for invalid args (NULL db/written), -2 when the
// buffer is too small. `*written` gets the byte count on success and the
// required capacity on -2; query the size with out=NULL, capacity=0. The
// buffer stays caller-owned; this call allocates/frees nothing and retains no
// reference to it afterwards.
int32_t houyi_mobile_db_save_buffer(void* db, uint8_t* out, size_t capacity, size_t* written);
// 从内存字节恢复数据库；失败返回 NULL（坏 magic/CRC/版本/截断/尾随字节）。
// bytes 只在本调用期间读取，返回后即可加密/清零；返回值用 houyi_mobile_db_close 释放。
// EN: Restore from memory bytes; NULL on failure (bad magic/CRC/version/
// truncation/trailing bytes). `bytes` is read only during this call and may be
// encrypted/wiped afterwards; release the result with houyi_mobile_db_close.
void* houyi_mobile_db_load_buffer(const uint8_t* bytes, size_t len);
void houyi_mobile_db_close(void* db);

#ifdef __cplusplus
}
#endif

#endif
