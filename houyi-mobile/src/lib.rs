//! HouyiDB Mobile: 轻量级向量索引，C ABI 接口，面向 iOS / Android。
//! C-compatible API for mobile vector search.
//!
//! 双模式 / Dual mode:
//!   - 单 Collection: houyi_mobile_create/insert/search  (向后兼容)
//!   - 多 Collection: houyi_mobile_db_open/collection/insert/search

// C ABI 边界：extern "C" 导出函数必须解引用原始指针，而 unsafe 无法跨 FFI 约束调用方。
// EN: FFI entry points deref raw pointers by design; `unsafe` cannot be enforced across the C ABI.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod backend;
pub mod db;
mod filter;
mod snapshot;
mod storage;

use backend::Backend;
use houyi_vector::cluster::FlatStorageDtype;
use houyi_vector::DistanceMetric;
use std::sync::Mutex;

/// 移动端索引句柄。
/// EN: Mobile-side index handle.
pub struct MobileIndex {
    inner: Mutex<Backend>,
    dims: usize,
}

impl MobileIndex {
    pub fn new(dims: usize, index_type: &str) -> Result<Self, String> {
        Ok(MobileIndex {
            inner: Mutex::new(Backend::new(dims, index_type)?),
            dims,
        })
    }

    /// 显式 metric/dtype 的索引句柄。
    /// EN: Index handle with an explicit metric/dtype.
    pub fn new_with_config(
        dims: usize,
        index_type: &str,
        metric: DistanceMetric,
        dtype: FlatStorageDtype,
    ) -> Result<Self, String> {
        let backend = Backend::new_with_config(dims, index_type, metric, dtype)?;
        Ok(MobileIndex {
            inner: Mutex::new(backend),
            dims,
        })
    }

    /// 插入/覆盖。建图后的 flat-clustered 立即可查；维度或其他拒绝返回 Err。
    /// EN: Insert or overwrite. A built flat-clustered index is searchable
    /// immediately; dimension or other rejections return Err.
    pub fn insert(&self, id: u64, vector: &[f32]) -> Result<(), String> {
        self.inner.lock().unwrap().insert(id, vector)
    }

    /// 构建/重聚类；返回索引原子数。
    /// EN: Build or recluster; returns the indexed count.
    pub fn build(&self) -> usize {
        self.inner.lock().unwrap().build()
    }

    /// 检索 top-k，返回 (ids, distances)；查询长度不等于 dims 时返回空列表。
    /// EN: Search top-k, returning (ids, distances); a query whose length does
    /// not equal dims yields an empty list.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.inner.lock().unwrap().search(query, k)
    }

    pub fn vector_count(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// 调整 flat-clustered 余弦检索的候选预算（每 k），默认 2500。
    /// EN: Adjust the flat-clustered cosine candidate budget per k (default 2500).
    pub fn set_search_target(&self, per_k: usize) -> Result<(), String> {
        self.inner.lock().unwrap().set_search_target(per_k)
    }

    /// 批量插入：一次加锁，按顺序应用；失败返回首个错误（之前的行已生效）。
    /// EN: Batch insert with a single lock, applied in order; returns the first
    /// error (rows before it remain applied).
    pub fn insert_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<(), String> {
        let expected = ids
            .len()
            .checked_mul(self.dims)
            .ok_or_else(|| "batch size overflow".to_string())?;
        if vectors.len() != expected {
            return Err(format!(
                "batch vectors have {} values, expected {} ({} rows × {} dims)",
                vectors.len(),
                expected,
                ids.len(),
                self.dims
            ));
        }
        let mut guard = self.inner.lock().unwrap();
        for (row, &id) in ids.iter().enumerate() {
            let start = row * self.dims;
            let vector = &vectors[start..start + self.dims];
            guard.insert(id, vector)?;
        }
        Ok(())
    }
}

/// C ABI v2 检索结果（64 位 id + f32 分数）。
/// C 侧布局为 `u64 id; f32 score;`，对齐后 `sizeof=16`（score 后 4 B padding）。
/// EN: C ABI v2 search result (64-bit id + f32 score). The C layout is
/// `u64 id; f32 score;`; alignment makes `sizeof == 16` (4 B padding after score).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HouyiMobileResult {
    pub id: u64,
    pub score: f32,
}

/// 解析标量标签 JSON（`[{"col":"tag","val":"text:beach"},...]`）。
/// 使用 serde_json，值里的逗号/括号/转义引号不会截断；解析失败由调用方报错。
/// EN: Parse the scalar-tags JSON (`[{"col":"tag","val":"text:beach"},...]`) with
/// serde_json, so commas/braces/escaped quotes inside values do not truncate;
/// parse failures are reported by the caller.
pub fn parse_tag_pairs(json_str: &str) -> Result<Vec<(String, String)>, String> {
    let value: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("invalid tags JSON: {e}"))?;
    let array = value
        .as_array()
        .ok_or_else(|| "tags JSON must be an array".to_string())?;
    let mut tags = Vec::with_capacity(array.len());
    for item in array {
        let obj = item
            .as_object()
            .ok_or_else(|| "tags JSON items must be objects".to_string())?;
        let col = obj
            .get("col")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "tags JSON item missing string 'col'".to_string())?;
        let val = obj
            .get("val")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "tags JSON item missing string 'val'".to_string())?;
        tags.push((col.to_string(), val.to_string()));
    }
    Ok(tags)
}

/// 解析混合检索过滤 JSON（`[{"col":"tag","op":"eq","val":"text:beach"},...]`）。
/// serde_json 负责字符串边界；`eq` 之外的未知 op 是硬错误，不再静默按 eq 处理。
/// EN: Parse the hybrid-filter JSON
/// (`[{"col":"tag","op":"eq","val":"text:beach"},...]`). serde_json handles
/// string boundaries; ops other than the documented set are hard errors and
/// are no longer silently treated as `eq`.
pub fn parse_filter_ops(json_str: &str) -> Result<Vec<crate::filter::FilterOp>, String> {
    use crate::filter::{parse_scalar, FilterOp, FilterOpType};
    let value: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("invalid filters JSON: {e}"))?;
    let array = value
        .as_array()
        .ok_or_else(|| "filters JSON must be an array".to_string())?;
    let mut filters = Vec::with_capacity(array.len());
    for item in array {
        let obj = item
            .as_object()
            .ok_or_else(|| "filters JSON items must be objects".to_string())?;
        let col = obj
            .get("col")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "filter item missing string 'col'".to_string())?;
        let op = obj
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "filter item missing string 'op'".to_string())?;
        let raw = obj
            .get("val")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "filter item missing string 'val'".to_string())?;
        let value = parse_scalar(raw)?;
        let op = match op {
            "eq" => FilterOpType::Eq,
            "gt" => FilterOpType::Gt,
            "lt" => FilterOpType::Lt,
            "gte" => FilterOpType::GtEq,
            "lte" => FilterOpType::LtEq,
            other => return Err(format!("invalid filter op {other:?}")),
        };
        filters.push(FilterOp {
            column: col.to_string(),
            op,
            value,
        });
    }
    Ok(filters)
}

/// 解析过滤 JSON，空串/纯空白表示无过滤；C ABI 与 JNI 共用同一策略。
/// EN: Parse filter JSON where an empty/whitespace string means "no filter";
/// shared by the C ABI and JNI so the policy cannot drift.
pub fn parse_filter_ops_allow_empty(
    json_str: &str,
) -> Result<Vec<crate::filter::FilterOp>, String> {
    if json_str.trim().is_empty() {
        Ok(Vec::new())
    } else {
        parse_filter_ops(json_str)
    }
}

/// 把后端插入错误映射为 C 返回码：参数/维度/缺集合 = -1，语义拒绝
/// （如 clustered-hnsw 建图后拒绝覆盖）= -2。
/// EN: Map backend insert errors to C return codes: argument/dimension/
/// missing-collection = -1; semantic rejection (e.g. clustered-hnsw refusing a
/// post-build overwrite) = -2.
fn insert_error_code(message: &str) -> i32 {
    if message.starts_with(backend::ERR_INVALID) || message.contains("not found") {
        -1
    } else {
        -2
    }
}

// ── C ABI ─────────────────────────────────────────────────
// 移动端原生调用：iOS (Swift/ObjC), Android (JNI/Kotlin)
// EN: -- C ABI --
// EN: Mobile native calls: iOS (Swift/ObjC), Android (JNI/Kotlin)

use std::ffi::{c_char, CStr};

/// 创建索引句柄。
/// index_type: "flat" | "flat-clustered" | "clustered-hnsw"
/// 返回句柄指针，失败返回 null。
#[no_mangle]
pub extern "C" fn houyi_mobile_create(dims: u32, index_type: *const c_char) -> *mut MobileIndex {
    if index_type.is_null() || dims == 0 {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(index_type) };
    let t = c_str.to_str().unwrap_or("flat");
    match MobileIndex::new(dims as usize, t) {
        Ok(idx) => Box::into_raw(Box::new(idx)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// ABI v2 创建索引：显式 metric/dtype。
/// metric: 0=L2 1=Cosine 2=IP 3=L1；dtype: 0=f32 1=f16 2=i8（f16/i8 仅
/// flat-clustered）。非法组合/维度返回 NULL。
/// EN: ABI v2 create with explicit metric/dtype. metric 0=L2, 1=Cosine,
/// 2=IP, 3=L1; dtype 0=f32, 1=f16, 2=i8 (f16/i8 are flat-clustered only).
/// Invalid combinations/dims return NULL.
#[no_mangle]
pub extern "C" fn houyi_mobile_create_v2(
    dims: u32,
    index_type: *const std::ffi::c_char,
    metric: u8,
    dtype: u8,
) -> *mut MobileIndex {
    if index_type.is_null() || dims == 0 {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(index_type) };
    let index_type = match c_str.to_str() {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    let metric = match DistanceMetric::from_snapshot_byte(metric) {
        Some(value) => value,
        None => return std::ptr::null_mut(),
    };
    let dtype = match FlatStorageDtype::from_snapshot_byte(dtype) {
        Some(value) => value,
        None => return std::ptr::null_mut(),
    };
    match MobileIndex::new_with_config(dims as usize, index_type, metric, dtype) {
        Ok(index) => Box::into_raw(Box::new(index)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 插入向量。vector 长度必须等于 dims。
/// 返回 0 成功、-1 参数/维度错误、-2 语义拒绝（如 CH 建图后覆盖已有 id）。
/// EN: Insert a vector; the length must equal dims. Returns 0 on success, -1 for
/// argument/dimension errors, and -2 for semantic rejection (e.g. a post-build
/// clustered-hnsw overwrite).
#[no_mangle]
pub extern "C" fn houyi_mobile_insert(
    handle: *mut MobileIndex,
    id: u64,
    vector: *const f32,
    len: u32,
) -> i32 {
    if handle.is_null() || vector.is_null() {
        return -1;
    }
    let idx = unsafe { &*handle };
    // FFI 边界先校验长度：错误长度既不能进入索引，也不能被静默丢弃。
    // EN: Validate the length at the FFI boundary: a wrong-length row must
    // neither enter the index nor be dropped silently.
    if len as usize != idx.dims {
        return -1;
    }
    let v = unsafe { std::slice::from_raw_parts(vector, len as usize) };
    match idx.insert(id, v) {
        Ok(()) => 0,
        Err(message) => insert_error_code(&message),
    }
}

/// 构建/定稿索引。返回索引向量数。
/// EN: Build/finalize the index. Returns the number of indexed vectors.
#[no_mangle]
pub extern "C" fn houyi_mobile_build(handle: *mut MobileIndex) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let idx = unsafe { &*handle };
    idx.build() as u32
}

/// 检索 top-k。结果写入 results 缓冲区: [id0, dist0, id1, dist1, ...]。
/// 返回实际写入的结果数。
/// EN: Search top-k. Results are written into the results buffer:
/// EN: [id0, dist0, id1, dist1, ...].
/// EN: Returns the number of results actually written.
#[no_mangle]
pub extern "C" fn houyi_mobile_search(
    handle: *mut MobileIndex,
    query: *const f32,
    dims: u32,
    k: u32,
    results: *mut f32, // [id as f32, dist] × k
    max_results: u32,
) -> u32 {
    if handle.is_null() || query.is_null() || results.is_null() {
        return 0;
    }
    let idx = unsafe { &*handle };
    if dims as usize != idx.dims {
        return 0;
    }
    let q = unsafe { std::slice::from_raw_parts(query, dims as usize) };
    let found = idx.search(q, k as usize);
    let n = found.len().min(max_results as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(results, n * 2) };
    for (i, (id, dist)) in found.iter().take(n).enumerate() {
        out[i * 2] = *id as f32;
        out[i * 2 + 1] = *dist;
    }
    n as u32
}

/// 批量插入（ABI v2）。ids/vectors 由调用方保证长度为 count 与 count*dims。
/// 返回 0 成功、-1 参数/维度、-2 语义拒绝；失败前的行已生效。
/// EN: Batch insert (ABI v2). The caller guarantees `count` ids and
/// `count*dims` floats. Returns 0, -1 for argument/dimension errors, -2 for
/// semantic rejection; rows before the failure remain applied.
#[no_mangle]
pub extern "C" fn houyi_mobile_insert_batch(
    handle: *mut MobileIndex,
    ids: *const u64,
    vectors: *const f32,
    count: u32,
    dims: u32,
) -> i32 {
    if handle.is_null() || (count > 0 && (ids.is_null() || vectors.is_null())) {
        return -1;
    }
    let idx = unsafe { &*handle };
    if dims as usize != idx.dims {
        return -1;
    }
    let count = count as usize;
    let expected = match count.checked_mul(idx.dims) {
        Some(value) => value,
        None => return -1,
    };
    let ids_slice = if count == 0 {
        &[] as &[u64]
    } else {
        unsafe { std::slice::from_raw_parts(ids, count) }
    };
    let vectors_slice = if expected == 0 {
        &[] as &[f32]
    } else {
        unsafe { std::slice::from_raw_parts(vectors, expected) }
    };
    match idx.insert_batch(ids_slice, vectors_slice) {
        Ok(()) => 0,
        Err(message) => insert_error_code(&message),
    }
}

/// 检索 top-k（ABI v2，u64 id）。results 为 `HouyiMobileResult[max_results]`。
/// EN: Search top-k (ABI v2, u64 ids). `results` is
/// `HouyiMobileResult[max_results]`.
#[no_mangle]
pub extern "C" fn houyi_mobile_search_u64(
    handle: *mut MobileIndex,
    query: *const f32,
    dims: u32,
    k: u32,
    results: *mut HouyiMobileResult,
    max_results: u32,
) -> u32 {
    if handle.is_null() || query.is_null() || results.is_null() {
        return 0;
    }
    let idx = unsafe { &*handle };
    if dims as usize != idx.dims {
        return 0;
    }
    let q = unsafe { std::slice::from_raw_parts(query, dims as usize) };
    let found = idx.search(q, k as usize);
    let n = found.len().min(max_results as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(results, n) };
    for (slot, (id, score)) in out.iter_mut().zip(found.iter().take(n)) {
        slot.id = *id;
        slot.score = *score;
    }
    n as u32
}

/// 调整候选预算（每 k）；0 或 >100000 返回 -1。
/// EN: Set the candidate budget per k; 0 or >100000 returns -1.
#[no_mangle]
pub extern "C" fn houyi_mobile_set_search_target(handle: *mut MobileIndex, per_k: u32) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let idx = unsafe { &*handle };
    match idx.set_search_target(per_k as usize) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// 获取已索引向量数。
/// EN: Get the number of indexed vectors.
#[no_mangle]
pub extern "C" fn houyi_mobile_count(handle: *mut MobileIndex) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let idx = unsafe { &*handle };
    idx.vector_count() as u32
}

/// 释放索引。
/// EN: Free the index.
#[no_mangle]
pub extern "C" fn houyi_mobile_destroy(handle: *mut MobileIndex) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

// ── MobileDB C API (多 Collection) ─────────────────────────
// EN: -- MobileDB C API (multiple collections) --

/// 打开/创建数据库 / Open or create a multi-collection database.
/// EN: Open or create a multi-collection database.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_open() -> *mut db::MobileDB {
    Box::into_raw(Box::new(db::MobileDB::new()))
}

/// 获取或创建 Collection / Get or create a named collection.
/// 首次调用时锁定维度，后续调用 dims 参数被忽略。
/// EN: Get or create a named collection.
/// EN: The first call locks the dimensions; later calls ignore the dims argument.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_collection(
    db_handle: *mut db::MobileDB,
    name: *const std::ffi::c_char,
    dims: u32,
    index_type: *const std::ffi::c_char,
) -> i32 {
    if db_handle.is_null() || name.is_null() || index_type.is_null() || dims == 0 {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(name) };
    let coll_name = match c_name.to_str() {
        Ok(value) if !value.is_empty() && value.len() <= 4096 => value,
        _ => return -1,
    };
    let c_type = unsafe { std::ffi::CStr::from_ptr(index_type) };
    let idx_type = match c_type.to_str() {
        Ok("flat") => "flat",
        Ok("flat-clustered") => "flat-clustered",
        Ok("clustered-hnsw") => "clustered-hnsw",
        _ => return -1,
    };
    match db.get_or_create(coll_name, dims as usize, idx_type) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// ABI v2 获取/创建 Collection：显式 metric/dtype（0=f32 1=f16 2=i8）；
/// 同名不同配置返回 -1。
/// EN: ABI v2 get-or-create collection with explicit metric/dtype
/// (0=f32, 1=f16, 2=i8); a same-name collection with a different
/// configuration returns -1.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_collection_v2(
    db_handle: *mut db::MobileDB,
    name: *const std::ffi::c_char,
    dims: u32,
    index_type: *const std::ffi::c_char,
    metric: u8,
    dtype: u8,
) -> i32 {
    if db_handle.is_null() || name.is_null() || index_type.is_null() || dims == 0 {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(name) };
    let coll_name = match c_name.to_str() {
        Ok(value) if !value.is_empty() && value.len() <= 4096 => value,
        _ => return -1,
    };
    let c_type = unsafe { std::ffi::CStr::from_ptr(index_type) };
    let idx_type = match c_type.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    let metric = match DistanceMetric::from_snapshot_byte(metric) {
        Some(value) => value,
        None => return -1,
    };
    let dtype = match FlatStorageDtype::from_snapshot_byte(dtype) {
        Some(value) => value,
        None => return -1,
    };
    match db.get_or_create_with_config(coll_name, dims as usize, idx_type, metric, dtype) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// 向 Collection 插入向量 / Insert vector into named collection.
/// 返回 0 成功、-1 参数/维度/缺集合、-2 语义拒绝（同单索引）。
/// EN: Returns 0 on success, -1 for argument/dimension/missing collection, and
/// -2 for semantic rejection (same as the single-index API).
#[no_mangle]
pub extern "C" fn houyi_mobile_db_insert(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    id: u64,
    vector: *const f32,
    len: u32,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() || vector.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = c_name.to_str().unwrap_or("default");
    let v = unsafe { std::slice::from_raw_parts(vector, len as usize) };
    match db.insert(name, id, v) {
        Ok(()) => 0,
        Err(message) => insert_error_code(&message),
    }
}

/// 构建 Collection 索引 / Build collection index.
/// EN: Build the collection index.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_build(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
) -> u32 {
    if db_handle.is_null() || coll_name.is_null() {
        return 0;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = c_name.to_str().unwrap_or("default");
    db.build(name).unwrap_or(0) as u32
}

/// 检索 / Search within a collection.
/// EN: Search within a collection.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_search(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    query: *const f32,
    dims: u32,
    k: u32,
    results: *mut f32,
    max_results: u32,
) -> u32 {
    if db_handle.is_null() || coll_name.is_null() || query.is_null() || results.is_null() {
        return 0;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = c_name.to_str().unwrap_or("default");
    let q = unsafe { std::slice::from_raw_parts(query, dims as usize) };
    let found = db.search(name, q, k as usize).unwrap_or_default();
    let n = found.len().min(max_results as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(results, n * 2) };
    for (i, (id, dist)) in found.iter().take(n).enumerate() {
        out[i * 2] = *id as f32;
        out[i * 2 + 1] = *dist;
    }
    n as u32
}

/// 设置检索选项：候选预算（per_k）、路由簇数（nprobe）、ef_search（仅
/// clustered-hnsw）。任一参数为 0 表示该项保持不变。
/// 返回 0 成功、-1 参数/集合缺失/越界。
/// EN: Set search options: candidate budget (per_k), routed clusters (nprobe),
/// and ef_search (clustered-hnsw only). A zero value leaves that field
/// unchanged. Returns 0 on success, -1 for bad arguments/missing collection.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_set_search_options(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    per_k: u32,
    nprobe: u32,
    ef_search: u32,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    match db.set_search_options(name, per_k, nprobe, ef_search) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// 应用检索预设：0=fast、1=balanced、2=high_recall；其他值返回 -1。
/// EN: Apply a search preset: 0=fast, 1=balanced, 2=high_recall; other values
/// return -1.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_set_search_preset(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    preset: u8,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() {
        return -1;
    }
    let Some(preset) = backend::SearchPreset::from_code(preset) else {
        return -1;
    };
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    match db.set_search_preset(name, preset) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// 下一次 build 使用的 nprobe（0 = 自适应公式）。
/// EN: nprobe for the next build (0 = adaptive formula).
#[no_mangle]
pub extern "C" fn houyi_mobile_db_set_build_nprobe(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    nprobe: u32,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    match db.set_build_nprobe(name, nprobe) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// 保存整个数据库到路径（原子写 + CRC 校验）。
/// 返回 0 成功、-1 参数/IO/序列化失败。
/// EN: Save the whole database to a path (atomic write + CRC). Returns 0 on
/// success; -1 for argument/IO/serialization failures.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_save(
    db_handle: *mut db::MobileDB,
    path: *const std::ffi::c_char,
) -> i32 {
    if db_handle.is_null() || path.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_path = unsafe { std::ffi::CStr::from_ptr(path) };
    let path_str = match c_path.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    match db.save_to(path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// 从路径加载数据库；失败返回 NULL（坏 magic/CRC/版本/截断）。
/// 返回的指针必须用 `houyi_mobile_db_close` 释放（不能 free 或传给 destroy）。
/// EN: Load a database from a path; returns NULL on failure (bad magic/CRC/
/// version/truncation). The returned pointer must be released with
/// `houyi_mobile_db_close` (not `free`, and not `houyi_mobile_destroy`).
#[no_mangle]
pub extern "C" fn houyi_mobile_db_load(path: *const std::ffi::c_char) -> *mut db::MobileDB {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let c_path = unsafe { std::ffi::CStr::from_ptr(path) };
    let path_str = match c_path.to_str() {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };
    match db::MobileDB::load_from(path_str) {
        Ok(db) => Box::into_raw(Box::new(db)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 把数据库编码进调用方缓冲区（不写任何临时文件）。
///
/// 返回 0 成功、-1 参数非法（db 或 written 为空）、-2 缓冲区不足。
/// `*written`：成功时为写入字节数；返回 -2 时为所需容量，调用方可据此扩容
/// 重试。查询所需容量：传 `out=NULL, capacity=0` 会返回 -2 并写 `*written`。
/// 缓冲区所有权始终属于调用方；本函数不分配、不释放内存。
/// EN: Encode the database into a caller-provided buffer (no temp files).
/// Returns 0 on success, -1 for invalid arguments (null db/written), -2 when
/// the buffer is too small. `*written` receives the byte count on success and
/// the required capacity on -2; query the size with `out=NULL, capacity=0`.
/// The buffer stays owned by the caller; this function allocates/frees nothing.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_save_buffer(
    db_handle: *mut db::MobileDB,
    out: *mut u8,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    if db_handle.is_null() || written.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let bytes = match db.encode() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    let required = bytes.len();
    unsafe { *written = required };
    if out.is_null() || capacity < required {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, required);
    }
    0
}

/// 从内存字节恢复数据库；失败返回 NULL（坏 magic/CRC/版本/截断/尾随字节）。
/// 返回的指针必须用 `houyi_mobile_db_close` 释放。
/// `bytes` 仅在本调用期间被读取，不会被保存；调用方可立即加密/清零。
/// EN: Restore a database from in-memory bytes; returns NULL on failure (bad
/// magic/CRC/version/truncation/trailing bytes). Release the returned pointer
/// with `houyi_mobile_db_close`. `bytes` is read only during this call and is
/// never retained, so the caller may encrypt/wipe it immediately.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_load_buffer(bytes: *const u8, len: usize) -> *mut db::MobileDB {
    if bytes.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes, len) };
    match db::MobileDB::decode(slice) {
        Ok(db) => Box::into_raw(Box::new(db)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 向 Collection 批量插入（ABI v2）；失败前的行已生效。
/// EN: Batch insert into a collection (ABI v2); rows before the failure remain
/// applied.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_insert_batch(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    ids: *const u64,
    vectors: *const f32,
    count: u32,
    dims: u32,
) -> i32 {
    if db_handle.is_null()
        || coll_name.is_null()
        || (count > 0 && (ids.is_null() || vectors.is_null()))
    {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    let count = count as usize;
    let expected = match count.checked_mul(dims as usize) {
        Some(value) => value,
        None => return -1,
    };
    let ids_slice = if count == 0 {
        &[] as &[u64]
    } else {
        unsafe { std::slice::from_raw_parts(ids, count) }
    };
    let vectors_slice = if expected == 0 {
        &[] as &[f32]
    } else {
        unsafe { std::slice::from_raw_parts(vectors, expected) }
    };
    match db.insert_batch(name, ids_slice, vectors_slice, dims as usize) {
        Ok(()) => 0,
        Err(message) => insert_error_code(&message),
    }
}

/// Collection 内检索（ABI v2，u64 id）。
/// EN: Search within a collection (ABI v2, u64 ids).
#[no_mangle]
pub extern "C" fn houyi_mobile_db_search_u64(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    query: *const f32,
    dims: u32,
    k: u32,
    results: *mut HouyiMobileResult,
    max_results: u32,
) -> u32 {
    if db_handle.is_null() || coll_name.is_null() || query.is_null() || results.is_null() {
        return 0;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return 0,
    };
    let q = unsafe { std::slice::from_raw_parts(query, dims as usize) };
    let found = db.search(name, q, k as usize).unwrap_or_default();
    let n = found.len().min(max_results as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(results, n) };
    for (slot, (id, score)) in out.iter_mut().zip(found.iter().take(n)) {
        slot.id = *id;
        slot.score = *score;
    }
    n as u32
}

/// 调整 Collection 候选预算（每 k）；0 或 >100000 返回 -1。
/// EN: Set the collection candidate budget per k; 0 or >100000 returns -1.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_set_search_target(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    per_k: u32,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    match db.set_search_target(name, per_k as usize) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// 获取 Collection 向量数 / Get vector count in a collection.
/// EN: Get the vector count of a collection.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_count(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
) -> u32 {
    if db_handle.is_null() || coll_name.is_null() {
        return 0;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = c_name.to_str().unwrap_or("default");
    db.count(name).unwrap_or(0) as u32
}

/// 删除 Collection / Drop a collection.
/// EN: Drop a collection.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_drop(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = c_name.to_str().unwrap_or("default");
    if db.drop(name) {
        0
    } else {
        -1
    }
}

/// 插入标量标签 / Insert scalar tags for a row.
/// tags_json: JSON array like [{"col":"tag","val":"text:beach"},{"col":"date","val":"ts:1717200000"}]
#[no_mangle]
pub extern "C" fn houyi_mobile_db_insert_scalars(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    tags_json: *const std::ffi::c_char,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() || tags_json.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = c_name.to_str().unwrap_or("default");
    let c_json = unsafe { std::ffi::CStr::from_ptr(tags_json) };
    let json_str = match c_json.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };

    // 简单 JSON 解析 [{"col":"tag","val":"text:beach"},...]
    // EN: Simple JSON parsing [{"col":"tag","val":"text:beach"},...]
    let owned = match parse_tag_pairs(json_str) {
        Ok(value) => value,
        Err(_) => return -1,
    };
    let refs: Vec<(&str, &str)> = owned
        .iter()
        .map(|(col, val)| (col.as_str(), val.as_str()))
        .collect();
    db.insert_scalars(name, &refs).map(|_| 0).unwrap_or(-1)
}

/// 带显式向量 id 的标量写入（混合检索按 id 关联；旧 API 仍按行号兼容）。
/// EN: Scalar write with an explicit vector id (hybrid search associates by id;
/// the legacy API keeps row-index compatibility).
#[no_mangle]
pub extern "C" fn houyi_mobile_db_insert_scalars_v2(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    id: u64,
    tags_json: *const std::ffi::c_char,
) -> i32 {
    if db_handle.is_null() || coll_name.is_null() || tags_json.is_null() {
        return -1;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    let c_json = unsafe { std::ffi::CStr::from_ptr(tags_json) };
    let json_str = match c_json.to_str() {
        Ok(value) => value,
        Err(_) => return -1,
    };
    let owned = match parse_tag_pairs(json_str) {
        Ok(value) => value,
        Err(_) => return -1,
    };
    let refs: Vec<(&str, &str)> = owned
        .iter()
        .map(|(col, val)| (col.as_str(), val.as_str()))
        .collect();
    db.insert_scalars_with_id(name, id, &refs)
        .map(|_| 0)
        .unwrap_or(-1)
}

/// 混合检索 / Hybrid search with structured filters.
/// filters_json: JSON array of filter ops
///   [{"col":"tag","op":"eq","val":"text:beach"},{"col":"date","op":"gt","val":"ts:1717200000"}]
#[no_mangle]
pub extern "C" fn houyi_mobile_db_search_hybrid(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    query: *const f32,
    dims: u32,
    k: u32,
    filters_json: *const std::ffi::c_char,
    results: *mut f32,
    max_results: u32,
) -> u32 {
    if db_handle.is_null()
        || coll_name.is_null()
        || query.is_null()
        || filters_json.is_null()
        || results.is_null()
    {
        return 0;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = c_name.to_str().unwrap_or("default");
    let q = unsafe { std::slice::from_raw_parts(query, dims as usize) };
    let c_json = unsafe { std::ffi::CStr::from_ptr(filters_json) };
    let json_str = match c_json.to_str() {
        Ok(value) => value,
        Err(_) => return 0,
    };

    // 解析 filters JSON → FilterOp vec
    // EN: Parse filters JSON → FilterOp vec.
    let filters = match parse_filter_ops_allow_empty(json_str) {
        Ok(value) => value,
        Err(_) => return 0,
    };

    let found = db
        .search_hybrid(name, q, k as usize, &filters)
        .unwrap_or_default();
    let n = found.len().min(max_results as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(results, n * 2) };
    for (i, (id, dist)) in found.iter().take(n).enumerate() {
        out[i * 2] = *id as f32;
        out[i * 2 + 1] = *dist;
    }
    n as u32
}

/// 混合检索（u64 结果，按 id 过滤下推）。
/// EN: Hybrid search with u64 results and id-based filter pushdown.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn houyi_mobile_db_search_hybrid_u64(
    db_handle: *mut db::MobileDB,
    coll_name: *const std::ffi::c_char,
    query: *const f32,
    dims: u32,
    k: u32,
    filters_json: *const std::ffi::c_char,
    results: *mut HouyiMobileResult,
    max_results: u32,
) -> u32 {
    if db_handle.is_null()
        || coll_name.is_null()
        || query.is_null()
        || filters_json.is_null()
        || results.is_null()
    {
        return 0;
    }
    let db = unsafe { &*db_handle };
    let c_name = unsafe { std::ffi::CStr::from_ptr(coll_name) };
    let name = match c_name.to_str() {
        Ok(value) => value,
        Err(_) => return 0,
    };
    let q = unsafe { std::slice::from_raw_parts(query, dims as usize) };
    let c_json = unsafe { std::ffi::CStr::from_ptr(filters_json) };
    let json_str = match c_json.to_str() {
        Ok(value) => value,
        Err(_) => return 0,
    };
    let filters = match parse_filter_ops_allow_empty(json_str) {
        Ok(value) => value,
        Err(_) => return 0,
    };
    let found = db
        .search_hybrid(name, q, k as usize, &filters)
        .unwrap_or_default();
    let n = found.len().min(max_results as usize);
    let out = unsafe { std::slice::from_raw_parts_mut(results, n) };
    for (slot, (id, score)) in out.iter_mut().zip(found.iter().take(n)) {
        slot.id = *id;
        slot.score = *score;
    }
    n as u32
}

/// 关闭数据库 / Close database and free all resources.
/// EN: Close the database and free all resources.
#[no_mangle]
pub extern "C" fn houyi_mobile_db_close(db_handle: *mut db::MobileDB) {
    if !db_handle.is_null() {
        unsafe {
            drop(Box::from_raw(db_handle));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_flat_create_insert_search() {
        let dims = 4u32;
        let t = CString::new("flat").unwrap();
        let handle = houyi_mobile_create(dims, t.as_ptr());
        assert!(!handle.is_null());

        // Insert
        let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
        let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
        assert_eq!(houyi_mobile_insert(handle, 1, v1.as_ptr(), 4), 0);
        assert_eq!(houyi_mobile_insert(handle, 2, v2.as_ptr(), 4), 0);
        assert_eq!(houyi_mobile_count(handle), 2);

        // Search
        let q: [f32; 4] = [1.0, 0.1, 0.0, 0.0];
        let mut results = vec![0.0f32; 4]; // 2 results × 2 fields
        let n = houyi_mobile_search(handle, q.as_ptr(), 4, 1, results.as_mut_ptr(), 2);
        assert_eq!(n, 1);
        // 最近邻应该是 id=1
        // EN: The nearest neighbor should be id=1.
        assert_eq!(results[0] as u64, 1);

        houyi_mobile_destroy(handle);
    }

    #[test]
    fn test_flat_clustered_build_search() {
        let dims = 8u32;
        let n = 200;
        let t = CString::new("flat-clustered").unwrap();
        let handle = houyi_mobile_create(dims, t.as_ptr());

        // 插入随机向量
        // EN: Insert random vectors
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vectors = Vec::new();
        for i in 0..n {
            let mut v = Vec::with_capacity(dims as usize);
            for _ in 0..dims {
                v.push(rng.gen::<f32>());
            }
            houyi_mobile_insert(handle, i as u64, v.as_ptr(), dims);
            vectors.push(v);
        }
        assert_eq!(houyi_mobile_count(handle), n as u32);

        // Build
        let count = houyi_mobile_build(handle);
        assert!(count > 0);

        // 检索：查询向量应命中自身
        // EN: Search: the query should find itself
        let q = &vectors[50];
        let mut results = vec![0.0f32; 20];
        let n_found = houyi_mobile_search(handle, q.as_ptr(), dims, 1, results.as_mut_ptr(), 10);
        assert_eq!(n_found, 1);
        assert_eq!(results[0] as u64, 50);

        houyi_mobile_destroy(handle);
    }

    /// 建图后插入必须立即可查；同 id 覆盖不增加计数（去重后 store 已释放，
    /// 由索引单独持有向量）。
    /// EN: Post-build inserts must be searchable immediately, and an upsert of
    /// an existing id must not change the count (the store is released after
    /// build and the index alone owns the vectors).
    #[test]
    fn test_flat_clustered_incremental_insert_visible() {
        let dims = 8u32;
        let n = 200;
        let t = CString::new("flat-clustered").unwrap();
        let handle = houyi_mobile_create(dims, t.as_ptr());
        assert!(!handle.is_null());

        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let mut vectors = Vec::new();
        for i in 0..n {
            let mut v = Vec::with_capacity(dims as usize);
            for _ in 0..dims {
                v.push(rng.gen::<f32>());
            }
            assert_eq!(houyi_mobile_insert(handle, i as u64, v.as_ptr(), dims), 0);
            vectors.push(v);
        }
        assert!(houyi_mobile_build(handle) > 0);
        assert_eq!(houyi_mobile_count(handle), n as u32);

        // 新 id 用一个独立方向：立即可查，计数 +1。
        // EN: Insert a new id along a distinct direction: searchable
        // immediately, count +1.
        let mut probe = vec![0.0f32; dims as usize];
        probe[0] = 1.0;
        assert_eq!(houyi_mobile_insert(handle, 9999, probe.as_ptr(), dims), 0);
        assert_eq!(houyi_mobile_count(handle), (n + 1) as u32);
        let mut results = vec![0.0f32; 20];
        let found = houyi_mobile_search(handle, probe.as_ptr(), dims, 5, results.as_mut_ptr(), 10);
        assert!(found >= 1);
        assert!(
            (0..found as usize).any(|i| results[i * 2] as u64 == 9999),
            "new id must be visible immediately"
        );

        // 同 id 覆盖到 10 号附近：计数不变，新位置可查（与 10 号同向时同分，
        // 因此按 top-k 包含性断言）。
        // EN: Upsert id 50 near vector 10: the count stays put and the new
        // position is searchable (same direction ties with id 10, so assert
        // containment in top-k).
        let mut moved = vectors[10].clone();
        moved[0] += 0.0001;
        assert_eq!(houyi_mobile_insert(handle, 50, moved.as_ptr(), dims), 0);
        assert_eq!(houyi_mobile_count(handle), (n + 1) as u32);
        let found = houyi_mobile_search(handle, moved.as_ptr(), dims, 5, results.as_mut_ptr(), 10);
        assert!(
            (0..found as usize).any(|i| results[i * 2] as u64 == 50),
            "upserted id must remain visible"
        );

        houyi_mobile_destroy(handle);
    }

    /// FFI 维度防护：错误长度返回 -1 且不写入；CH 建图后新 id 可追加即查，
    /// 覆盖已有 id 返回 -2 且计数不变。
    /// EN: FFI dimension guard: a wrong length returns -1 and writes nothing.
    /// A built clustered-hnsw appends new ids (immediately searchable) and
    /// rejects overwriting an existing id with -2 without inflating the count.
    #[test]
    fn test_insert_rejections_are_loud() {
        let dims = 4u32;
        let t = CString::new("flat-clustered").unwrap();
        let handle = houyi_mobile_create(dims, t.as_ptr());
        assert!(!handle.is_null());
        assert_eq!(
            houyi_mobile_insert(handle, 1, [1.0f32, 0.0, 0.0].as_ptr(), 3),
            -1
        );
        assert_eq!(houyi_mobile_count(handle), 0);
        houyi_mobile_destroy(handle);

        let t = CString::new("clustered-hnsw").unwrap();
        let handle = houyi_mobile_create(dims, t.as_ptr());
        let mut vectors = Vec::new();
        for i in 0..10u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            assert_eq!(houyi_mobile_insert(handle, i, v.as_ptr(), dims), 0);
            vectors.push(v);
        }
        assert!(houyi_mobile_build(handle) > 0);
        assert_eq!(houyi_mobile_count(handle), 10);
        // 新 id 建图后允许追加并立即可查；同 id 覆盖返回 -2（重建后 upsert）。
        // EN: New ids may be appended after build and are immediately searchable;
        // overwriting an existing id returns -2 (rebuild to upsert).
        // 用独立方向的新向量，避免与既有共线向量的余弦并列导致 top-k 截断。
        // EN: Use a distinct direction so the appended id is not cut off by a
        // cosine tie with the existing collinear vectors.
        let appended = [0.0f32, 1.0, 0.0, 0.0];
        assert_eq!(houyi_mobile_insert(handle, 22, appended.as_ptr(), dims), 0);
        assert_eq!(houyi_mobile_count(handle), 11);
        let mut results = vec![0.0f32; 20];
        let found =
            houyi_mobile_search(handle, appended.as_ptr(), dims, 5, results.as_mut_ptr(), 10);
        assert!(
            (0..found as usize).any(|i| results[i * 2] as u64 == 22),
            "appended clustered-hnsw id must be immediately searchable"
        );
        let v = vectors[2].clone();
        assert_eq!(
            houyi_mobile_insert(handle, 2, v.as_ptr(), dims),
            -2,
            "post-build clustered-hnsw overwrite must be rejected"
        );
        assert_eq!(houyi_mobile_count(handle), 11);
        // 重建索引保留全部行（覆盖/upsert 需重建后重新灌入数据）。
        // EN: A rebuild preserves every row (overwrites/upserts require
        // recreating the index and re-inserting the data).
        assert!(houyi_mobile_build(handle) > 0);
        houyi_mobile_destroy(handle);
    }

    /// 持久化往返：flat-clustered 构建态（含增量行）、flat、CH、标量过滤
    /// 全部恢复；损坏 magic/CRC/截断干净拒绝。
    /// EN: Persistence roundtrip: built flat-clustered state (including
    /// incremental rows), flat, CH, and scalar filters all restore; bad
    /// magic/CRC/truncation are cleanly rejected.
    #[test]
    fn test_db_snapshot_roundtrip_and_corruption() {
        use rand::Rng;
        use rand::SeedableRng;
        let path = std::env::temp_dir().join(format!(
            "houyi-mobile-snap-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = db::MobileDB::new();
        let dims = 8usize;
        let n = 200usize;
        db.get_or_create("faces", dims, "flat-clustered").unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(202);
        let mut vectors = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
            db.insert("faces", i as u64, &v).unwrap();
            vectors.push(v);
        }
        assert!(db.build("faces").unwrap() > 0);
        let mut probe = vec![0.0f32; dims];
        probe[0] = 1.0;
        db.insert("faces", 9999, &probe).unwrap();
        let moved = vectors[10].clone();
        db.insert("faces", 50, &moved).unwrap();
        assert_eq!(db.count("faces").unwrap(), n + 1);

        db.get_or_create("raw", 4, "flat").unwrap();
        for i in 0..5u64 {
            let v = vec![i as f32, 0.0, 0.0, 1.0];
            db.insert("raw", i, &v).unwrap();
        }
        db.get_or_create("tags", 4, "flat").unwrap();
        for (i, tag) in ["a", "b", "a"].iter().enumerate() {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            db.insert("tags", i as u64, &v).unwrap();
            db.insert_scalars("tags", &[("tag", &format!("text:{tag}"))])
                .unwrap();
        }
        db.get_or_create("fc-pending", 4, "flat-clustered").unwrap();
        for i in 0..3u64 {
            db.insert("fc-pending", i, &[i as f32, 0.0, 0.0, 1.0])
                .unwrap();
        }
        db.get_or_create("ch-pending", 4, "clustered-hnsw").unwrap();
        for i in 0..3u64 {
            db.insert("ch-pending", i, &[0.0, i as f32, 0.0, 1.0])
                .unwrap();
        }
        db.get_or_create("ch", 4, "clustered-hnsw").unwrap();
        let mut ch_vectors = Vec::new();
        for i in 0..10u64 {
            let v: Vec<f32> = (0..4).map(|_| rng.gen::<f32>()).collect();
            db.insert("ch", i, &v).unwrap();
            ch_vectors.push(v);
        }
        assert!(db.build("ch").unwrap() > 0);

        db.save_to(&path).unwrap();
        assert!(path.exists(), "snapshot file must exist");
        let loaded = db::MobileDB::load_from(&path).unwrap();
        assert_eq!(loaded.count("faces").unwrap(), n + 1);
        let found = loaded.search("faces", &probe, 5).unwrap();
        assert!(found.iter().any(|(id, _)| *id == 9999));
        let found = loaded.search("faces", &moved, 5).unwrap();
        assert!(found.iter().any(|(id, _)| *id == 50));
        assert_eq!(loaded.count("raw").unwrap(), 5);
        let found = loaded.search("tags", &[0.0f32, 0.0, 0.0, 1.0], 5).unwrap();
        assert_eq!(found.len(), 3);
        use crate::filter::{FilterOp, FilterOpType, ScalarVal};
        let filtered = loaded
            .search_hybrid(
                "tags",
                &[0.0f32, 0.0, 0.0, 1.0],
                5,
                &[FilterOp {
                    column: "tag".into(),
                    op: FilterOpType::Eq,
                    value: ScalarVal::Text("a".into()),
                }],
            )
            .unwrap();
        assert_eq!(filtered.len(), 2, "scalar rows must survive snapshot");
        assert_eq!(loaded.count("ch").unwrap(), 10);
        let found = loaded.search("ch", &ch_vectors[5], 3).unwrap();
        assert!(found.iter().any(|(id, _)| *id == 5));
        // 未建图的两种惰性后端也要往返：count 保留、随后 build/检索可用。
        // EN: Both unbuilt lazy backends must roundtrip: count survives and a
        // later build/search works.
        assert_eq!(loaded.count("fc-pending").unwrap(), 3);
        assert!(loaded.build("fc-pending").unwrap() > 0);
        let found = loaded
            .search("fc-pending", &[0.0f32, 0.0, 0.0, 1.0], 3)
            .unwrap();
        assert_eq!(found.len(), 3);
        assert_eq!(loaded.count("ch-pending").unwrap(), 3);
        assert!(loaded.build("ch-pending").unwrap() > 0);
        let found = loaded
            .search("ch-pending", &[0.0f32, 1.0, 0.0, 1.0], 3)
            .unwrap();
        assert!(!found.is_empty(), "CH search must work after load+build");

        // C ABI 入口同样覆盖：保存 → 加载指针 → count → close；坏路径拒绝。
        // EN: Cover the C ABI entry points too: save -> load pointer -> count ->
        // close, with bad paths rejected.
        let abi_db = houyi_mobile_db_open();
        let name = CString::new("abi").unwrap();
        let ty = CString::new("flat-clustered").unwrap();
        assert_eq!(
            houyi_mobile_db_collection(abi_db, name.as_ptr(), 4, ty.as_ptr()),
            0
        );
        let v = [1.0f32, 0.0, 0.0, 0.0];
        assert_eq!(
            houyi_mobile_db_insert(abi_db, name.as_ptr(), 1, v.as_ptr(), 4),
            0
        );
        let abi_path =
            std::env::temp_dir().join(format!("houyi-mobile-abi-{}.bin", std::process::id()));
        let abi_path_c = CString::new(abi_path.to_str().unwrap()).unwrap();
        assert_eq!(houyi_mobile_db_save(abi_db, abi_path_c.as_ptr()), 0);
        let abi_loaded = houyi_mobile_db_load(abi_path_c.as_ptr());
        assert!(!abi_loaded.is_null());
        assert_eq!(houyi_mobile_db_count(abi_loaded, name.as_ptr()), 1);
        assert_eq!(houyi_mobile_db_save(abi_db, std::ptr::null()), -1);
        assert!(houyi_mobile_db_load(std::ptr::null()).is_null());
        houyi_mobile_db_close(abi_db);
        houyi_mobile_db_close(abi_loaded);
        let _ = std::fs::remove_file(&abi_path);

        // 损坏样本：截断 / 坏 magic / CRC 翻转都必须拒绝。
        // EN: Corrupted samples (truncated, bad magic, flipped CRC) must be
        // rejected.
        let bytes = std::fs::read(&path).unwrap();
        let truncated = std::env::temp_dir().join(format!(
            "houyi-mobile-snap-trunc-{}.bin",
            std::process::id()
        ));
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
        assert!(db::MobileDB::load_from(&truncated).is_err());
        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        let bad_magic_path = std::env::temp_dir().join(format!(
            "houyi-mobile-snap-magic-{}.bin",
            std::process::id()
        ));
        std::fs::write(&bad_magic_path, &bad_magic).unwrap();
        assert!(db::MobileDB::load_from(&bad_magic_path).is_err());
        let mut bad_crc = bytes;
        bad_crc[20] ^= 0xFF;
        let bad_crc_path =
            std::env::temp_dir().join(format!("houyi-mobile-snap-crc-{}.bin", std::process::id()));
        std::fs::write(&bad_crc_path, &bad_crc).unwrap();
        assert!(db::MobileDB::load_from(&bad_crc_path).is_err());
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&truncated);
        let _ = std::fs::remove_file(&bad_magic_path);
        let _ = std::fs::remove_file(&bad_crc_path);
    }

    /// clustered-hnsw 构建态持久化：保存的是图而非原始行，加载不重建，
    /// 检索结果与保存前一致；同时验证文件与内存快照两条路径。
    /// EN: Clustered-hnsw persists its built graphs (not raw rows); loading does
    /// not rebuild and search results match. Covers both file and buffer paths.
    #[test]
    fn test_clustered_hnsw_built_state_persistence() {
        use rand::Rng;
        use rand::SeedableRng;
        let dims = 8usize;
        let n = 300usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(31337);
        let db = db::MobileDB::new();
        db.get_or_create_with_config(
            "ch",
            dims,
            "clustered-hnsw",
            DistanceMetric::Cosine,
            FlatStorageDtype::F32,
        )
        .unwrap();
        let mut data = Vec::with_capacity(n * dims);
        for _ in 0..n {
            for _ in 0..dims {
                data.push(rng.gen::<f32>() * 2.0 - 1.0);
            }
        }
        for i in 0..n {
            db.insert("ch", i as u64, &data[i * dims..(i + 1) * dims])
                .unwrap();
        }
        db.build("ch").unwrap();
        let q = &data[..dims];
        let before: Vec<(u64, f32)> = db.search("ch", q, 5).unwrap();
        assert_eq!(before.len(), 5);

        let bytes = db.encode().unwrap();
        let loaded = db::MobileDB::decode(&bytes).unwrap();
        assert_eq!(loaded.count("ch").unwrap(), n);
        let after = loaded.search("ch", q, 5).unwrap();
        assert_eq!(after.len(), before.len());
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1.0e-5);
        }

        // 文件路径同样走构建态。
        // EN: The file path uses the built state as well.
        let path =
            std::env::temp_dir().join(format!("houyi-mobile-ch-state-{}.bin", std::process::id()));
        db.save_to(&path).unwrap();
        let from_file = db::MobileDB::load_from(&path).unwrap();
        assert_eq!(from_file.count("ch").unwrap(), n);
        assert_eq!(from_file.search("ch", q, 5).unwrap()[0].0, before[0].0);
        let _ = std::fs::remove_file(&path);

        // 坏字节拒绝。
        // EN: Corrupt bytes are rejected.
        let mut corrupt = bytes.clone();
        corrupt[0] ^= 0xFF;
        assert!(db::MobileDB::decode(&corrupt).is_err());

        // v3 快照仍可读：把版本字节改回 3（头部不参与 CRC），解码必须成功。
        // EN: v3 snapshots stay readable: rewrite the version bytes to 3 (the
        // header is outside the CRC) and decoding must still succeed.
        let mut v3 = bytes.clone();
        v3[8..12].copy_from_slice(&3u32.to_le_bytes());
        let from_v3 = db::MobileDB::decode(&v3).expect("v3 snapshot must load");
        assert_eq!(from_v3.count("ch").unwrap(), n);
        assert_eq!(from_v3.search("ch", q, 5).unwrap()[0].0, before[0].0);
    }

    /// 检索预设与 nprobe 覆盖：构建期 nprobe=1 时随机查询召回很低，
    /// high_recall 预设（全簇路由）必须显著提升；非法参数必须拒绝。
    /// EN: Presets and nprobe overrides: with a build-time nprobe of 1, random
    /// queries recall poorly; the high_recall preset (all clusters) must lift
    /// it clearly, and invalid options must be rejected.
    #[test]
    fn test_search_presets_and_nprobe_override() {
        use crate::backend::SearchPreset;
        use rand::Rng;
        use rand::SeedableRng;
        let dims = 8usize;
        let clusters = 20usize;
        let per_cluster = 50usize;
        let n = clusters * per_cluster;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242);
        let mut centers = vec![0f32; clusters * dims];
        for x in &mut centers {
            *x = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let mut data = Vec::with_capacity(n * dims);
        for i in 0..n {
            let c = i / per_cluster;
            for d in 0..dims {
                // 噪声足够大，使 top-10 真的跨簇：否则 nprobe=1 本身就能覆盖。
                // EN: Noise must be large enough that top-10 spans clusters;
                // otherwise nprobe=1 already covers the true neighbours.
                data.push(centers[c * dims + d] + (rng.gen::<f32>() - 0.5) * 0.8);
            }
        }
        let db = db::MobileDB::new();
        for name in ["approx", "exact"] {
            db.get_or_create_with_config(
                name,
                dims,
                if name == "approx" {
                    "flat-clustered"
                } else {
                    "flat"
                },
                DistanceMetric::Cosine,
                FlatStorageDtype::F32,
            )
            .unwrap();
        }
        for i in 0..n {
            let v = &data[i * dims..(i + 1) * dims];
            db.insert("approx", i as u64, v).unwrap();
            db.insert("exact", i as u64, v).unwrap();
        }
        // 构建期 nprobe=1：随机查询只能扫到约 1/20 的簇。
        // EN: Build-time nprobe=1: a random query only scans ~1/20 of clusters.
        db.set_build_nprobe("approx", 1).unwrap();
        assert!(db.build("approx").unwrap() > 0);

        let recall = |db: &db::MobileDB| -> f64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(7);
            let mut hits = 0usize;
            let total = 20usize * 10;
            for _ in 0..20 {
                let q: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                let truth: ahash::AHashSet<u64> = db
                    .search("exact", &q, 10)
                    .unwrap()
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                let got: ahash::AHashSet<u64> = db
                    .search("approx", &q, 10)
                    .unwrap()
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                hits += truth.intersection(&got).count();
            }
            hits as f64 / total as f64
        };
        let low = recall(&db);
        db.set_search_preset("approx", SearchPreset::HighRecall)
            .unwrap();
        let high = recall(&db);
        assert!(
            high > low + 0.3,
            "high_recall preset must lift recall: low={low} high={high}"
        );
        // 回到默认档并恢复构建期 nprobe。
        // EN: Back to the balanced preset restores the build-time nprobe.
        db.set_search_preset("approx", SearchPreset::Balanced)
            .unwrap();
        // 0 值字段保持不变，其余越界拒绝。
        // EN: Zero fields stay unchanged; out-of-range values are rejected.
        assert!(db.set_search_options("approx", 0, 0, 0).is_ok());
        assert!(db.set_search_options("approx", 0, 0, 5000).is_err());
        assert!(db.set_search_options("missing", 0, 0, 0).is_err());

        // C ABI：预设码与选项越界返回 -1。
        // EN: C ABI: bad preset codes and options return -1.
        let name = CString::new("approx").unwrap();
        let abi = houyi_mobile_db_open();
        let ty = CString::new("flat-clustered").unwrap();
        assert_eq!(
            houyi_mobile_db_collection_v2(abi, name.as_ptr(), dims as u32, ty.as_ptr(), 1, 0),
            0
        );
        for (id, v) in data.chunks_exact(dims).enumerate().take(40) {
            assert_eq!(
                houyi_mobile_db_insert(abi, name.as_ptr(), id as u64, v.as_ptr(), dims as u32),
                0
            );
        }
        assert_eq!(
            houyi_mobile_db_set_search_options(abi, name.as_ptr(), 0, 0, 0),
            0
        );
        assert_eq!(
            houyi_mobile_db_set_search_options(abi, name.as_ptr(), 0, 0, 5000),
            -1
        );
        assert_eq!(
            houyi_mobile_db_set_search_options(std::ptr::null_mut(), name.as_ptr(), 0, 0, 0),
            -1
        );
        assert_eq!(houyi_mobile_db_set_search_preset(abi, name.as_ptr(), 2), 0);
        assert_eq!(houyi_mobile_db_set_search_preset(abi, name.as_ptr(), 9), -1);
        assert_eq!(houyi_mobile_db_set_build_nprobe(abi, name.as_ptr(), 0), 0);
        houyi_mobile_db_close(abi);
    }

    /// 内存快照：encode/decode 往返、空库、坏 magic/CRC/截断拒绝，以及
    /// C ABI buffer 入口（查询容量 / 缓冲区不足 / 空指针 / 坏字节）。
    /// EN: In-memory snapshot: encode/decode roundtrip, empty DB, bad magic/CRC/
    /// truncation rejection, and the C ABI buffer entry points (size query,
    /// too-small buffer, null pointers, corrupt bytes).
    #[test]
    fn test_snapshot_buffer_roundtrip_and_rejection() {
        use rand::Rng;
        use rand::SeedableRng;
        // 空库也要能往返。
        // EN: An empty database must roundtrip too.
        let empty = db::MobileDB::new();
        let empty_bytes = empty.encode().unwrap();
        let restored = db::MobileDB::decode(&empty_bytes).unwrap();
        assert!(restored.list().is_empty());

        let db = db::MobileDB::new();
        let dims = 6usize;
        db.get_or_create("faces", dims, "flat-clustered").unwrap();
        db.get_or_create("raw", dims, "flat").unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let mut probes = Vec::new();
        for i in 0..64u64 {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
            db.insert("faces", i, &v).unwrap();
            db.insert("raw", i, &v).unwrap();
            probes.push(v);
        }
        assert!(db.build("faces").unwrap() > 0);
        db.insert_scalars_with_id("faces", 3, &[("tag", "text:keep")])
            .unwrap();

        let bytes = db.encode().unwrap();
        let loaded = db::MobileDB::decode(&bytes).unwrap();
        assert_eq!(loaded.count("faces").unwrap(), 64);
        assert_eq!(loaded.count("raw").unwrap(), 64);
        let found = loaded.search("faces", &probes[3], 5).unwrap();
        assert!(
            found.iter().take(1).any(|(id, _)| *id == 3),
            "top-1 must survive encode/decode"
        );
        assert!(!loaded.search("raw", &probes[9], 3).unwrap().is_empty());

        // 坏 magic / 坏 CRC / 截断都必须拒绝。
        // EN: Bad magic/CRC/truncation must all be rejected.
        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(db::MobileDB::decode(&bad_magic).is_err());
        let mut bad_crc = bytes.clone();
        let last = bad_crc.len() - 1;
        bad_crc[last] ^= 0xFF;
        assert!(db::MobileDB::decode(&bad_crc).is_err());
        assert!(db::MobileDB::decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(db::MobileDB::decode(&bytes[..8]).is_err());

        // C ABI buffer：容量查询 → 缓冲区不足 → 成功 → 加载。
        // EN: C ABI buffer: size query -> too small -> success -> load.
        let abi_db = houyi_mobile_db_open();
        assert!(!abi_db.is_null());
        let name = CString::new("buf").unwrap();
        let ty = CString::new("flat-clustered").unwrap();
        assert_eq!(
            houyi_mobile_db_collection_v2(abi_db, name.as_ptr(), 4, ty.as_ptr(), 1, 1),
            0
        );
        let ids = [1u64, 2, 3];
        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        assert_eq!(
            houyi_mobile_db_insert_batch(
                abi_db,
                name.as_ptr(),
                ids.as_ptr(),
                vectors.as_ptr(),
                3,
                4
            ),
            0
        );
        assert!(houyi_mobile_db_build(abi_db, name.as_ptr()) > 0);

        let mut required = 0usize;
        assert_eq!(
            houyi_mobile_db_save_buffer(abi_db, std::ptr::null_mut(), 0, &mut required),
            -2
        );
        assert!(required > 0, "size query must report required bytes");
        let mut small = vec![0u8; required - 1];
        let mut written = 0usize;
        assert_eq!(
            houyi_mobile_db_save_buffer(abi_db, small.as_mut_ptr(), small.len(), &mut written),
            -2
        );
        assert_eq!(written, required, "short buffer must report required size");
        let mut buffer = vec![0u8; required];
        assert_eq!(
            houyi_mobile_db_save_buffer(abi_db, buffer.as_mut_ptr(), buffer.len(), &mut written),
            0
        );
        assert_eq!(written, required);

        let loaded_abi = houyi_mobile_db_load_buffer(buffer.as_ptr(), buffer.len());
        assert!(!loaded_abi.is_null());
        assert_eq!(
            houyi_mobile_db_count(loaded_abi, name.as_ptr()),
            3,
            "buffer roundtrip must keep the vector count"
        );
        houyi_mobile_db_close(loaded_abi);

        // 非法参数与坏字节。
        // EN: Invalid arguments and corrupt bytes.
        let mut sink = 0usize;
        assert_eq!(
            houyi_mobile_db_save_buffer(
                std::ptr::null_mut(),
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut sink
            ),
            -1
        );
        assert_eq!(
            houyi_mobile_db_save_buffer(
                abi_db,
                buffer.as_mut_ptr(),
                buffer.len(),
                std::ptr::null_mut()
            ),
            -1
        );
        assert!(houyi_mobile_db_load_buffer(std::ptr::null(), 10).is_null());
        assert!(houyi_mobile_db_load_buffer(buffer.as_ptr(), 0).is_null());
        let mut corrupted = buffer.clone();
        corrupted[0] ^= 0xFF;
        assert!(houyi_mobile_db_load_buffer(corrupted.as_ptr(), corrupted.len()).is_null());
        houyi_mobile_db_close(abi_db);
    }

    /// ABI v2：批量插入与 u64 结果（>2^24 的 id 不再丢精度），DB 路径同样覆盖。
    /// EN: ABI v2: batch insert and u64 results (ids above 2^24 keep full
    /// precision), including the DB path.
    /// 非法创建永远返回 NULL/-1，不再跨 FFI panic（未知类型/超大 dims）。
    /// EN: Invalid creation always returns NULL/-1 instead of panicking across
    /// the FFI boundary (unknown type / oversized dims).
    #[test]
    fn test_invalid_creation_is_degraded_not_panicking() {
        let bogus = CString::new("bogus").unwrap();
        // 旧入口保留历史降级（未知类型映射 clustered-hnsw），但不 panic；
        // v2 显式配置则严格拒绝。
        // EN: The legacy entry keeps its historical fallback (unknown type maps
        // to clustered-hnsw) without panicking; v2 explicit config rejects.
        let fallback = houyi_mobile_create(4, bogus.as_ptr());
        assert!(!fallback.is_null());
        houyi_mobile_destroy(fallback);
        assert!(houyi_mobile_create_v2(4, bogus.as_ptr(), 0, 0).is_null());
        let flat = CString::new("flat").unwrap();
        assert!(houyi_mobile_create(100_000, flat.as_ptr()).is_null());
        assert!(houyi_mobile_create_v2(4, flat.as_ptr(), 9, 0).is_null());
        let db = houyi_mobile_db_open();
        let name = CString::new("bad").unwrap();
        assert_eq!(
            houyi_mobile_db_collection(db, name.as_ptr(), 4, bogus.as_ptr()),
            -1
        );
        assert_eq!(
            houyi_mobile_db_collection_v2(db, name.as_ptr(), 4, flat.as_ptr(), 1, 1),
            -1
        );
        assert_eq!(
            houyi_mobile_db_collection(db, name.as_ptr(), 100_000, flat.as_ptr()),
            -1
        );
        houyi_mobile_db_close(db);
    }

    #[test]
    fn test_abi_v2_batch_and_u64_ids() {
        let dims = 4u32;
        let t = CString::new("flat").unwrap();
        let handle = houyi_mobile_create_v2(dims, t.as_ptr(), 0, 0);
        assert!(!handle.is_null());
        let ids = [9_007_199_254_740_993u64, 42u64];
        let vectors = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(
            houyi_mobile_insert_batch(
                handle,
                ids.as_ptr(),
                vectors.as_ptr(),
                ids.len() as u32,
                dims
            ),
            0
        );
        let query = [1.0f32, 0.0, 0.0, 0.0];
        let mut results = vec![HouyiMobileResult { id: 0, score: 0.0 }; 4];
        let found = houyi_mobile_search_u64(
            handle,
            query.as_ptr(),
            dims,
            2,
            results.as_mut_ptr(),
            results.len() as u32,
        );
        assert_eq!(found, 2);
        assert_eq!(results[0].id, ids[0], "u64 id must not be rounded");
        assert_eq!(results[1].id, ids[1]);
        // 批量维度错误不写入。
        // EN: A batch dimension mismatch writes nothing.
        assert_eq!(
            houyi_mobile_insert_batch(
                handle,
                ids.as_ptr(),
                vectors.as_ptr(),
                ids.len() as u32,
                dims + 1
            ),
            -1
        );
        assert_eq!(houyi_mobile_count(handle), 2);
        houyi_mobile_destroy(handle);

        // DB 批量路径 + u64 检索。
        // EN: DB batch path + u64 search.
        let db = houyi_mobile_db_open();
        let name = CString::new("people").unwrap();
        let fc = CString::new("flat-clustered").unwrap();
        assert_eq!(
            houyi_mobile_db_collection_v2(db, name.as_ptr(), dims, fc.as_ptr(), 1, 1),
            0
        );
        assert_eq!(
            houyi_mobile_db_insert_batch(
                db,
                name.as_ptr(),
                ids.as_ptr(),
                vectors.as_ptr(),
                ids.len() as u32,
                dims
            ),
            0
        );
        assert!(houyi_mobile_db_build(db, name.as_ptr()) > 0);
        let mut db_results = vec![HouyiMobileResult { id: 0, score: 0.0 }; 4];
        let found = houyi_mobile_db_search_u64(
            db,
            name.as_ptr(),
            query.as_ptr(),
            dims,
            2,
            db_results.as_mut_ptr(),
            db_results.len() as u32,
        );
        assert_eq!(found, 2);
        assert_eq!(db_results[0].id, ids[0]);
        assert_eq!(db_results[1].id, ids[1]);
        houyi_mobile_db_close(db);
    }

    /// 过滤解析：未知 op 与非法值必须硬错误，不能静默改成 eq/丢条件。
    /// EN: Filter parsing must reject unknown ops and invalid values instead of
    /// silently falling back to eq or dropping the condition.
    #[test]
    fn test_parse_filter_ops_rejects_bad_op_and_value() {
        assert!(parse_filter_ops("[{\"col\":\"tag\",\"op\":\"ge\",\"val\":\"text:a\"}]").is_err());
        assert!(parse_filter_ops("[{\"col\":\"n\",\"op\":\"eq\",\"val\":\"int:abc\"}]").is_err());
        // 空串/纯空白是“无过滤”，非法 JSON 仍然是错误。
        // EN: Empty/whitespace means no filter; malformed JSON stays an error.
        assert!(parse_filter_ops_allow_empty("  ").unwrap().is_empty());
        assert!(parse_filter_ops_allow_empty("not-json").is_err());
    }

    /// 混合检索按向量 id 过滤（不再把标量行号当 id）；持久化后仍按 id。
    /// EN: Hybrid filtering associates by vector id (not scalar row index) and
    /// keeps that association across a snapshot roundtrip.
    #[test]
    fn test_hybrid_filter_by_vector_id() {
        use crate::filter::{FilterOp, FilterOpType, ScalarVal};
        let dims = 4usize;
        let db = db::MobileDB::new();
        db.get_or_create("hybrid", dims, "flat-clustered").unwrap();
        for id in [10u64, 20, 30] {
            let mut v = vec![0.0f32; dims];
            v[(id as usize / 10) % dims] = 1.0;
            db.insert("hybrid", id, &v).unwrap();
        }
        db.build("hybrid").unwrap();
        db.insert_scalars_with_id("hybrid", 10, &[("tag", "text:a")])
            .unwrap();
        db.insert_scalars_with_id("hybrid", 20, &[("tag", "text:b")])
            .unwrap();
        db.insert_scalars_with_id("hybrid", 30, &[("tag", "text:a")])
            .unwrap();
        let filters = [FilterOp {
            column: "tag".into(),
            op: FilterOpType::Eq,
            value: ScalarVal::Text("a".into()),
        }];
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let found = db.search_hybrid("hybrid", &q, 10, &filters).unwrap();
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|(id, _)| *id == 10 || *id == 30));
        // 同 id 更新标签：旧标签不得再命中。
        // EN: Updating tags for the same id must drop the old tag match.
        db.insert_scalars_with_id("hybrid", 10, &[("tag", "text:c")])
            .unwrap();
        let found = db.search_hybrid("hybrid", &q, 10, &filters).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, 30);
        let path =
            std::env::temp_dir().join(format!("houyi-mobile-hybrid-{}.bin", std::process::id()));
        db.save_to(&path).unwrap();
        let loaded = db::MobileDB::load_from(&path).unwrap();
        let found = loaded.search_hybrid("hybrid", &q, 10, &filters).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, 30);
        let _ = std::fs::remove_file(&path);

        // C ABI v2：显式 id 写入 + u64 hybrid 检索。
        // EN: C ABI v2: explicit-id scalar write + u64 hybrid search.
        let handle = houyi_mobile_db_open();
        let name = CString::new("abi-hybrid").unwrap();
        let ty = CString::new("flat-clustered").unwrap();
        assert_eq!(
            houyi_mobile_db_collection(handle, name.as_ptr(), 4, ty.as_ptr()),
            0
        );
        let v0 = [1.0f32, 0.0, 0.0, 0.0];
        let v1 = [0.0f32, 1.0, 0.0, 0.0];
        assert_eq!(
            houyi_mobile_db_insert(handle, name.as_ptr(), 10, v0.as_ptr(), 4),
            0
        );
        assert_eq!(
            houyi_mobile_db_insert(handle, name.as_ptr(), 20, v1.as_ptr(), 4),
            0
        );
        let tags = CString::new("[{\"col\":\"tag\",\"val\":\"text:a\"}]").unwrap();
        let tags_b = CString::new("[{\"col\":\"tag\",\"val\":\"text:b\"}]").unwrap();
        assert_eq!(
            houyi_mobile_db_insert_scalars_v2(handle, name.as_ptr(), 10, tags.as_ptr()),
            0
        );
        assert_eq!(
            houyi_mobile_db_insert_scalars_v2(handle, name.as_ptr(), 20, tags_b.as_ptr()),
            0
        );
        assert!(houyi_mobile_db_build(handle, name.as_ptr()) > 0);
        let filters = CString::new("[{\"col\":\"tag\",\"op\":\"eq\",\"val\":\"text:a\"}]").unwrap();
        let mut out = vec![HouyiMobileResult { id: 0, score: 0.0 }; 4];
        let found = houyi_mobile_db_search_hybrid_u64(
            handle,
            name.as_ptr(),
            v0.as_ptr(),
            4,
            2,
            filters.as_ptr(),
            out.as_mut_ptr(),
            out.len() as u32,
        );
        assert_eq!(found, 1);
        assert_eq!(out[0].id, 10);
        houyi_mobile_db_close(handle);
    }

    /// i8 存储：mobile DB 创建/检索/持久化，以及 C ABI dtype=2 路径。
    /// EN: i8 storage: mobile DB create/search/persistence plus the C ABI
    /// dtype=2 path.
    #[test]
    fn test_i8_storage_config_and_persistence() {
        let dims = 4usize;
        let db = db::MobileDB::new();
        db.get_or_create_with_config(
            "i8",
            dims,
            "flat-clustered",
            DistanceMetric::Cosine,
            FlatStorageDtype::I8,
        )
        .unwrap();
        for i in 0..10u64 {
            let mut v = vec![0.0f32; dims];
            v[(i as usize) % dims] = 1.0;
            v[(i as usize + 1) % dims] = 0.1;
            db.insert("i8", i, &v).unwrap();
        }
        assert!(db.build("i8").unwrap() > 0);
        assert_eq!(db.count("i8").unwrap(), 10);
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let found = db.search("i8", &q, 3).unwrap();
        assert!(!found.is_empty());
        let path = std::env::temp_dir().join(format!("houyi-mobile-i8-{}.bin", std::process::id()));
        db.save_to(&path).unwrap();
        let loaded = db::MobileDB::load_from(&path).unwrap();
        let col = loaded.get("i8").unwrap();
        assert_eq!(col.dtype, FlatStorageDtype::I8);
        assert_eq!(loaded.count("i8").unwrap(), 10);
        assert!(!loaded.search("i8", &q, 3).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);

        // C ABI dtype=2 路径。
        // EN: C ABI dtype=2 path.
        let handle = houyi_mobile_db_open();
        let name = CString::new("abi-i8").unwrap();
        let ty = CString::new("flat-clustered").unwrap();
        assert_eq!(
            houyi_mobile_db_collection_v2(handle, name.as_ptr(), 4, ty.as_ptr(), 1, 2),
            0
        );
        let v = [1.0f32, 0.0, 0.0, 0.0];
        assert_eq!(
            houyi_mobile_db_insert(handle, name.as_ptr(), 1, v.as_ptr(), 4),
            0
        );
        let v2 = [0.0f32, 1.0, 0.0, 0.0];
        assert_eq!(
            houyi_mobile_db_insert(handle, name.as_ptr(), 2, v2.as_ptr(), 4),
            0
        );
        assert!(houyi_mobile_db_build(handle, name.as_ptr()) > 0);
        let mut out = vec![HouyiMobileResult { id: 0, score: 0.0 }; 4];
        let found = houyi_mobile_db_search_u64(
            handle,
            name.as_ptr(),
            v.as_ptr(),
            4,
            2,
            out.as_mut_ptr(),
            out.len() as u32,
        );
        assert_eq!(found, 2);
        assert_eq!(out[0].id, 1);
        assert_eq!(out[1].id, 2);
        houyi_mobile_db_close(handle);
    }

    /// 显式 metric/dtype 创建与持久化；同名不同配置拒绝。
    /// EN: Explicit metric/dtype creation and persistence; same-name
    /// configurations are rejected.
    #[test]
    fn test_metric_dtype_config_and_persistence() {
        let db = db::MobileDB::new();
        assert!(db
            .get_or_create_with_config(
                "metric-l1",
                4,
                "flat",
                DistanceMetric::L1,
                FlatStorageDtype::F32
            )
            .is_ok());
        assert!(db
            .get_or_create_with_config(
                "metric-l1",
                4,
                "flat",
                DistanceMetric::L2,
                FlatStorageDtype::F32
            )
            .is_err());
        let col = db.get("metric-l1").unwrap();
        assert_eq!(col.metric, DistanceMetric::L1);
        assert_eq!(col.dtype, FlatStorageDtype::F32);
        db.insert("metric-l1", 1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let path =
            std::env::temp_dir().join(format!("houyi-mobile-config-{}.bin", std::process::id()));
        db.save_to(&path).unwrap();
        let loaded = db::MobileDB::load_from(&path).unwrap();
        let col = loaded.get("metric-l1").unwrap();
        assert_eq!(col.metric, DistanceMetric::L1);
        assert_eq!(col.dtype, FlatStorageDtype::F32);
        let found = loaded
            .search("metric-l1", &[0.0f32, 1.0, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(found[0].0, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_db_multi_collection() {
        let db = db::MobileDB::new();
        // 创建两个维度不同的 Collection
        // EN: Create two collections with different dims
        let photos = db.get_or_create("photos", 128, "flat").unwrap();
        let docs = db.get_or_create("docs", 256, "flat").unwrap();
        assert_eq!(photos.dims, 128);
        assert_eq!(docs.dims, 256);

        // 向 Collection 插入向量
        // EN: Insert into collections
        let v128: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let v256: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        db.insert("photos", 1, &v128).unwrap();
        db.insert("docs", 2, &v256).unwrap();
        assert_eq!(db.count("photos").unwrap(), 1);
        assert_eq!(db.count("docs").unwrap(), 1);

        // Search
        let r = db.search("photos", &v128, 1).unwrap();
        assert_eq!(r[0].0, 1);
        let r = db.search("docs", &v256, 1).unwrap();
        assert_eq!(r[0].0, 2);

        // 维度不匹配
        // EN: Dim mismatch
        assert!(db.insert("photos", 3, &v256).is_err());

        // List
        let names = db.list();
        assert!(names.contains(&"photos".to_string()));
        assert!(names.contains(&"docs".to_string()));
    }

    /// MobileDB 的 flat-clustered 增量流程：插入→建图→增量→覆盖→重建，
    /// 与单索引句柄共用 backend 状态机（防止两份实现漂移）。
    /// EN: MobileDB flat-clustered incremental flow: insert -> build ->
    /// incremental -> overwrite -> rebuild, sharing the single backend state
    /// machine with the handle so the two entry points cannot drift.
    #[test]
    fn test_db_flat_clustered_incremental_flow() {
        use rand::Rng;
        use rand::SeedableRng;
        let dims = 8usize;
        let n = 200;
        let db = db::MobileDB::new();
        db.get_or_create("faces", dims, "flat-clustered").unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut vectors = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
            db.insert("faces", i as u64, &v).unwrap();
            vectors.push(v);
        }
        assert_eq!(db.count("faces").unwrap(), n);
        assert!(db.build("faces").unwrap() > 0);
        assert_eq!(db.count("faces").unwrap(), n);

        let mut probe = vec![0.0f32; dims];
        probe[0] = 1.0;
        db.insert("faces", 9999, &probe).unwrap();
        assert_eq!(db.count("faces").unwrap(), n + 1);
        let found = db.search("faces", &probe, 5).unwrap();
        assert!(found.iter().any(|(id, _)| *id == 9999));

        let moved = vectors[10].clone();
        db.insert("faces", 50, &moved).unwrap();
        assert_eq!(db.count("faces").unwrap(), n + 1);
        let found = db.search("faces", &moved, 5).unwrap();
        assert!(found.iter().any(|(id, _)| *id == 50));

        // 维度错误通过 DB 入口传播。
        // EN: Dimension errors propagate through the DB entry point.
        assert!(db.insert("faces", 7, &[0.0f32; 4]).is_err());
        assert_eq!(db.count("faces").unwrap(), n + 1);

        let rebuilt = db.build("faces").unwrap();
        assert_eq!(rebuilt, n + 1);
        assert_eq!(db.count("faces").unwrap(), n + 1);
    }

    /// MobileDB 的 clustered-hnsw 建图后追加新 id、拒绝覆盖已有 id。
    /// EN: MobileDB appends new ids and rejects overwriting an existing id on
    /// a built clustered-hnsw collection.
    #[test]
    fn test_db_clustered_hnsw_appends_new_ids_and_rejects_overwrite() {
        let dims = 4usize;
        let db = db::MobileDB::new();
        db.get_or_create("people", dims, "clustered-hnsw").unwrap();
        for i in 0..10u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            db.insert("people", i, &v).unwrap();
        }
        assert!(db.build("people").unwrap() > 0);
        assert_eq!(db.count("people").unwrap(), 10);
        let v = vec![5.0f32, 0.0, 0.0, 0.0];
        assert!(db.insert("people", 77, &v).is_ok());
        assert_eq!(db.count("people").unwrap(), 11);
        assert!(db.insert("people", 5, &v).is_err());
        assert_eq!(db.count("people").unwrap(), 11);
    }
}
