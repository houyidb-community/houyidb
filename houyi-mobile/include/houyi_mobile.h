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
int32_t houyi_mobile_db_save(void* db, const char* path);
void* houyi_mobile_db_load(const char* path);
void houyi_mobile_db_close(void* db);

#ifdef __cplusplus
}
#endif

#endif
