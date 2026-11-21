//! Android JNI 绑定：Kotlin `cc.houyidb.mobile.HouyiDB` 的 native 实现。
//! Android JNI bridge for the Kotlin object `cc.houyidb.mobile.HouyiDB`.
//!
//! 所有入口都在 JNI 边界把错误转成返回码并写入线程安全的 lastError，
//! 不把 Rust panic/字符串细节泄漏到 Java 层。
//! EN: Every entry converts errors to return codes and stores them in a
//! thread-safe last-error slot; no Rust panic or raw detail crosses into Java.

use houyi_mobile::db::MobileDB;
use houyi_vector::cluster::FlatStorageDtype;
use houyi_vector::DistanceMetric;
use jni::objects::{JFloatArray, JLongArray, JObject, JString};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

// 错误槽按调用线程隔离：Kotlin 同步调用后立即读 lastError，coroutine 在同一
// 调用栈内不会切线程；共享全局槽会被其他线程的成功调用清空。
// EN: The error slot is thread-local: Kotlin reads lastError synchronously
// after the native call and a coroutine cannot switch threads within that
// stack frame; a shared global slot would be cleared by another thread's
// success.
thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

// 活句柄表：拒绝重复 close / 已关闭句柄的后续调用（避免 double free/UAF）。
// EN: Live-handle registry: reject double close and calls on closed handles
// (avoiding double free/UAF).
static HANDLES: LazyLock<Mutex<HashSet<usize>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn register_handle(value: jlong) {
    if value != 0 {
        if let Ok(mut handles) = HANDLES.lock() {
            handles.insert(value as usize);
        }
    }
}

fn handle_from(value: jlong) -> Option<&'static MobileDB> {
    if value == 0 {
        return None;
    }
    let live = HANDLES
        .lock()
        .map(|handles| handles.contains(&(value as usize)))
        .unwrap_or(false);
    if !live {
        return None;
    }
    Some(unsafe { &*(value as *const MobileDB) })
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 4096 {
        return Err("invalid collection name".to_string());
    }
    Ok(())
}

fn string_arg(env: &mut JNIEnv<'_>, value: &JString<'_>) -> Result<String, String> {
    env.get_string(value)
        .map(|s| s.into())
        .map_err(|e| format!("invalid string argument: {e}"))
}

fn metric_from(value: jint) -> Option<DistanceMetric> {
    u8::try_from(value)
        .ok()
        .and_then(DistanceMetric::from_snapshot_byte)
}

fn dtype_from(value: jint) -> Option<FlatStorageDtype> {
    u8::try_from(value)
        .ok()
        .and_then(FlatStorageDtype::from_snapshot_byte)
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeOpen(
    _env: JNIEnv<'_>,
    _this: JObject<'_>,
) -> jlong {
    clear_error();
    let handle = Box::into_raw(Box::new(MobileDB::new())) as jlong;
    register_handle(handle);
    handle
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeClose(
    _env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    let removed = HANDLES
        .lock()
        .map(|mut handles| handles.remove(&(handle as usize)))
        .unwrap_or(false);
    if removed {
        unsafe {
            drop(Box::from_raw(handle as *mut MobileDB));
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeCreateCollectionLegacy(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    dims: jint,
    index_type: JString<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let (Ok(name), Ok(index_type)) = (
        string_arg(&mut env, &name),
        string_arg(&mut env, &index_type),
    ) else {
        set_error("invalid string argument");
        return -1;
    };
    if let Err(message) = validate_name(&name) {
        set_error(message);
        return -1;
    }
    if dims <= 0 || dims > 65_536 {
        set_error("invalid dims");
        return -1;
    }
    if !matches!(
        index_type.as_str(),
        "flat" | "flat-clustered" | "clustered-hnsw"
    ) {
        set_error("invalid index type");
        return -1;
    }
    match db.get_or_create(&name, dims as usize, &index_type) {
        Ok(_) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeCreateCollection(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    dims: jint,
    index_type: JString<'_>,
    metric: jint,
    dtype: jint,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let (Ok(name), Ok(index_type)) = (
        string_arg(&mut env, &name),
        string_arg(&mut env, &index_type),
    ) else {
        set_error("invalid string argument");
        return -1;
    };
    if let Err(message) = validate_name(&name) {
        set_error(message);
        return -1;
    }
    if dims <= 0 || dims > 65_536 {
        set_error("invalid dims");
        return -1;
    }
    let (Some(metric), Some(dtype)) = (metric_from(metric), dtype_from(dtype)) else {
        set_error("invalid metric/dtype code");
        return -1;
    };
    match db.get_or_create_with_config(&name, dims as usize, &index_type, metric, dtype) {
        Ok(_) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeInsert(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    id: jlong,
    vector: JFloatArray<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    let len = match env.get_array_length(&vector) {
        Ok(value) => value as usize,
        Err(_) => {
            set_error("invalid vector array");
            return -1;
        }
    };
    let mut values = vec![0.0f32; len];
    if env.get_float_array_region(&vector, 0, &mut values).is_err() {
        set_error("invalid vector array");
        return -1;
    }
    match db.insert(&name, id as u64, &values) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeInsertBatch(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    ids: JLongArray<'_>,
    vectors: JFloatArray<'_>,
    count: jint,
    dims: jint,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    let id_len = env.get_array_length(&ids).unwrap_or(0) as usize;
    let vec_len = env.get_array_length(&vectors).unwrap_or(0) as usize;
    if count < 0 || dims <= 0 {
        set_error("batch count/dims exceed arrays");
        return -1;
    }
    let count = count as usize;
    if count > id_len || dims <= 0 {
        set_error("batch count/dims exceed arrays");
        return -1;
    }
    let expected = match count.checked_mul(dims as usize) {
        Some(value) if value <= vec_len => value,
        _ => {
            set_error("batch vectors shorter than count*dims");
            return -1;
        }
    };
    let mut id_values = vec![0i64; count];
    if env.get_long_array_region(&ids, 0, &mut id_values).is_err() {
        set_error("invalid id array");
        return -1;
    }
    let mut vec_values = vec![0.0f32; expected];
    if env
        .get_float_array_region(&vectors, 0, &mut vec_values)
        .is_err()
    {
        set_error("invalid vector array");
        return -1;
    }
    let ids_u64: Vec<u64> = id_values.into_iter().map(|v| v as u64).collect();
    match db.insert_batch(&name, &ids_u64, &vec_values, dims as usize) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeBuild(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    match db.build(&name) {
        Ok(count) => count as jint,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeInsertScalarsWithId(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    id: jlong,
    tags_json: JString<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let (Ok(name), Ok(tags_json)) = (
        string_arg(&mut env, &name),
        string_arg(&mut env, &tags_json),
    ) else {
        set_error("invalid string argument");
        return -1;
    };
    let owned = match houyi_mobile::parse_tag_pairs(&tags_json) {
        Ok(value) => value,
        Err(message) => {
            set_error(message);
            return -1;
        }
    };
    let refs: Vec<(&str, &str)> = owned
        .iter()
        .map(|(col, val)| (col.as_str(), val.as_str()))
        .collect();
    match db.insert_scalars_with_id(&name, id as u64, &refs) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSearchHybridU64(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    query: JFloatArray<'_>,
    k: jint,
    filters_json: JString<'_>,
    out_ids: JLongArray<'_>,
    out_scores: JFloatArray<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let (Ok(name), Ok(filters_json)) = (
        string_arg(&mut env, &name),
        string_arg(&mut env, &filters_json),
    ) else {
        set_error("invalid string argument");
        return -1;
    };
    if k < 0 {
        set_error("invalid k");
        return -1;
    }
    let query_len = match env.get_array_length(&query) {
        Ok(value) => value as usize,
        Err(_) => {
            set_error("invalid query array");
            return -1;
        }
    };
    let mut q = vec![0.0f32; query_len];
    if env.get_float_array_region(&query, 0, &mut q).is_err() {
        set_error("invalid query array");
        return -1;
    }
    // 与 C ABI 共用同一“空串=无过滤”策略，避免两套 FFI 语义漂移。
    // EN: Use the same empty-means-no-filter helper as the C ABI so the two
    // FFI surfaces cannot drift.
    let filters = match houyi_mobile::parse_filter_ops_allow_empty(&filters_json) {
        Ok(value) => value,
        Err(message) => {
            set_error(message);
            return -1;
        }
    };
    let found = match db.search_hybrid(&name, &q, k as usize, &filters) {
        Ok(value) => value,
        Err(message) => {
            set_error(message);
            return -1;
        }
    };
    let out_len = match (
        env.get_array_length(&out_ids),
        env.get_array_length(&out_scores),
    ) {
        (Ok(ids), Ok(scores)) => ids.min(scores) as usize,
        _ => {
            set_error("invalid result arrays");
            return -1;
        }
    };
    let n = found.len().min(out_len);
    let mut ids = vec![0i64; n];
    let mut scores = vec![0.0f32; n];
    for (i, (id, score)) in found.iter().take(n).enumerate() {
        ids[i] = *id as i64;
        scores[i] = *score;
    }
    if env.set_long_array_region(&out_ids, 0, &ids).is_err()
        || env.set_float_array_region(&out_scores, 0, &scores).is_err()
    {
        set_error("failed to write result arrays");
        return -1;
    }
    n as jint
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSetSearchTarget(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    per_k: jint,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    if per_k <= 0 {
        set_error("invalid search target per k");
        return -1;
    }
    match db.set_search_target(&name, per_k as usize) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeCount(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    match db.count(&name) {
        Ok(count) => count as jint,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSearch(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    query: JFloatArray<'_>,
    k: jint,
    out_ids: JLongArray<'_>,
    out_scores: JFloatArray<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    let query_len = env.get_array_length(&query).unwrap_or(0) as usize;
    let mut values = vec![0.0f32; query_len];
    if env.get_float_array_region(&query, 0, &mut values).is_err() {
        set_error("invalid query array");
        return -1;
    }
    if k < 0 {
        set_error("invalid k");
        return -1;
    }
    let found = match db.search(&name, &values, k as usize) {
        Ok(values) => values,
        Err(message) => {
            set_error(message);
            return -1;
        }
    };
    let out_len = env
        .get_array_length(&out_ids)
        .unwrap_or(0)
        .min(env.get_array_length(&out_scores).unwrap_or(0)) as usize;
    let n = found.len().min(out_len);
    let mut ids = vec![0i64; n];
    let mut scores = vec![0.0f32; n];
    for (i, (id, score)) in found.iter().take(n).enumerate() {
        ids[i] = *id as i64;
        scores[i] = *score;
    }
    if env.set_long_array_region(&out_ids, 0, &ids).is_err()
        || env.set_float_array_region(&out_scores, 0, &scores).is_err()
    {
        set_error("failed to write result arrays");
        return -1;
    }
    n as jint
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSave(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    path: JString<'_>,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(path) = string_arg(&mut env, &path) else {
        set_error("invalid path");
        return -1;
    };
    match db.save_to(&path) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeLoad(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    path: JString<'_>,
) -> jlong {
    clear_error();
    let Ok(path) = string_arg(&mut env, &path) else {
        set_error("invalid path");
        return 0;
    };
    match MobileDB::load_from(&path) {
        Ok(db) => {
            let handle = Box::into_raw(Box::new(db)) as jlong;
            register_handle(handle);
            handle
        }
        Err(message) => {
            set_error(message);
            0
        }
    }
}

/// 把整个数据库编码成 Java byte[]（不落任何临时文件，供 Vault 加密后保存）。
/// 失败返回 null 并写入 lastError。
/// EN: Encode the whole database into a Java byte[] (no temp file, so the
/// caller's Vault can encrypt it in place). Returns null on failure.
#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSaveBytes(
    env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
) -> jni::sys::jbyteArray {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return std::ptr::null_mut();
    };
    let bytes = match db.encode() {
        Ok(value) => value,
        Err(message) => {
            set_error(message);
            return std::ptr::null_mut();
        }
    };
    match env.byte_array_from_slice(&bytes) {
        Ok(array) => array.into_raw(),
        Err(e) => {
            set_error(format!("cannot allocate byte array: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// 从 Java byte[] 恢复数据库；失败返回 0 并写入 lastError。
/// 入参字节只在本调用内读取，Kotlin 侧随后可加密/清零。
/// EN: Restore a database from a Java byte[]; returns 0 on failure. The bytes
/// are read only during this call, so Kotlin may encrypt/wipe them afterwards.
#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeLoadBytes(
    env: JNIEnv<'_>,
    _this: JObject<'_>,
    bytes: jni::objects::JByteArray<'_>,
) -> jlong {
    clear_error();
    let data = match env.convert_byte_array(&bytes) {
        Ok(value) => value,
        Err(e) => {
            set_error(format!("invalid byte array: {e}"));
            return 0;
        }
    };
    match MobileDB::decode(&data) {
        Ok(db) => {
            let handle = Box::into_raw(Box::new(db)) as jlong;
            register_handle(handle);
            handle
        }
        Err(message) => {
            set_error(message);
            0
        }
    }
}

/// 设置检索选项（0 = 该项保持不变）；失败返回 -1 并写入 lastError。
/// EN: Set search options (0 leaves a field unchanged); -1 on failure.
#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSetSearchOptions(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    per_k: jint,
    nprobe: jint,
    ef_search: jint,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    let (Ok(per_k), Ok(nprobe), Ok(ef_search)) = (
        u32::try_from(per_k),
        u32::try_from(nprobe),
        u32::try_from(ef_search),
    ) else {
        set_error("negative search option");
        return -1;
    };
    match db.set_search_options(&name, per_k, nprobe, ef_search) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

/// 应用检索预设（0/1/2）；失败返回 -1。
/// EN: Apply a search preset (0/1/2); -1 on failure.
#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSetSearchPreset(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    preset: jint,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    let Ok(code) = u8::try_from(preset) else {
        set_error("invalid preset");
        return -1;
    };
    let Some(preset) = houyi_mobile::backend::SearchPreset::from_code(code) else {
        set_error("invalid preset");
        return -1;
    };
    match db.set_search_preset(&name, preset) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

/// 下一次 build 使用的 nprobe（0 = 自适应）；失败返回 -1。
/// EN: nprobe for the next build (0 = adaptive); -1 on failure.
#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeSetBuildNprobe(
    mut env: JNIEnv<'_>,
    _this: JObject<'_>,
    handle: jlong,
    name: JString<'_>,
    nprobe: jint,
) -> jint {
    clear_error();
    let Some(db) = handle_from(handle) else {
        set_error("invalid database handle");
        return -1;
    };
    let Ok(name) = string_arg(&mut env, &name) else {
        set_error("invalid collection name");
        return -1;
    };
    let Ok(nprobe) = u32::try_from(nprobe) else {
        set_error("negative nprobe");
        return -1;
    };
    match db.set_build_nprobe(&name, nprobe) {
        Ok(()) => 0,
        Err(message) => {
            set_error(message);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cc_houyidb_mobile_HouyiDB_nativeLastError(
    env: JNIEnv<'_>,
    _this: JObject<'_>,
) -> jstring {
    let message = LAST_ERROR.with(|slot| slot.borrow().clone());
    match env.new_string(message) {
        Ok(value) => value.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}
