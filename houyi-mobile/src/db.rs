//! MobileDB: 多 Collection 管理（与服务器 CollectionManager 一脉相承）。
//! Multi-collection manager matching the server-side CollectionManager API.
//!
//! 用法 / Usage:
//!   let db = MobileDB::new();
//!   db.get_or_create("photos", 512, "flat-clustered");
//!   db.insert("photos", 1, &vec![0.1; 512])?;
//!   db.build("photos")?;
//!   let results = db.search("photos", &query, 10)?;

use crate::backend::{Backend, BackendSnapshot, ERR_INVALID};
use crate::filter::{parse_scalar, FilterOp, ScalarStore, ScalarStoreSnapshot, ScalarVal};
use houyi_vector::cluster::FlatStorageDtype;
use houyi_vector::DistanceMetric;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 单集合持久化视图（save/load 编解码的中间结构）。
/// EN: Persistence view of one collection (intermediate form for the
/// save/load codec).
pub(crate) struct CollectionSnapshot {
    pub name: String,
    pub dims: usize,
    pub index_type: String,
    pub metric: DistanceMetric,
    pub dtype: FlatStorageDtype,
    pub backend: BackendSnapshot,
    pub scalars: ScalarStoreSnapshot,
}

/// Collection：独立维度 + 独立索引后端。
/// A named collection with its own dims and index backend.
pub struct Collection {
    pub name: String,
    pub dims: usize,
    pub index_type: String,
    pub metric: DistanceMetric,
    pub dtype: FlatStorageDtype,
    backend: Mutex<Backend>,
    /// 标量列存储（用于混合检索）/ Scalar store for hybrid search
    /// EN: Scalar store for hybrid search.
    scalars: Mutex<ScalarStore>,
}

/// MobileDB：多 Collection 管理器（与服务器 CollectionManager 一脉相承）。
/// Multi-collection database — same semantics as server-side CollectionManager.
pub struct MobileDB {
    collections: Mutex<HashMap<String, Arc<Collection>>>,
}

impl Default for MobileDB {
    fn default() -> Self {
        Self::new()
    }
}

impl MobileDB {
    /// 按名称排序导出所有集合，保证同一状态写成稳定字节。
    /// EN: Export all collections sorted by name so the same state encodes to
    /// stable bytes.
    pub(crate) fn snapshot_collections(&self) -> Result<Vec<CollectionSnapshot>, String> {
        // 只在取集合列表时持全局锁；逐集合克隆/编码在锁外执行，避免 save
        // 长时间阻塞 get/list/drop。
        // EN: Hold the global lock only while cloning the collection list;
        // per-collection snapshotting runs outside it so a save cannot block
        // get/list/drop for seconds.
        let entries: Vec<Arc<Collection>> = {
            let guard = self.collections.lock().unwrap();
            let mut names: Vec<&String> = guard.keys().collect();
            names.sort();
            names
                .into_iter()
                .filter_map(|name| guard.get(name).cloned())
                .collect()
        };
        let mut out = Vec::with_capacity(entries.len());
        for col in entries {
            let backend = col
                .backend
                .lock()
                .unwrap()
                .snapshot()
                .map_err(|e| format!("collection '{}': {e}", col.name))?;
            out.push(CollectionSnapshot {
                name: col.name.clone(),
                dims: col.dims,
                index_type: col.index_type.clone(),
                metric: col.metric,
                dtype: col.dtype,
                backend,
                scalars: col.scalars.lock().unwrap().snapshot(),
            });
        }
        Ok(out)
    }

    /// 从持久化视图重建数据库；重名/维度不一致/未知后端返回 Err。
    /// EN: Rebuild a database from persistence views; duplicate names,
    /// dimension mismatches, and unknown backends return Err.
    pub(crate) fn from_snapshots(snapshots: Vec<CollectionSnapshot>) -> Result<Self, String> {
        let mut collections = HashMap::with_capacity(snapshots.len());
        for snapshot in snapshots {
            if snapshot.name.is_empty() {
                return Err("snapshot: empty collection name".to_string());
            }
            if snapshot.dims == 0 {
                return Err(format!(
                    "snapshot: collection '{}' has zero dims",
                    snapshot.name
                ));
            }
            if collections.contains_key(&snapshot.name) {
                return Err(format!(
                    "snapshot: duplicate collection '{}'",
                    snapshot.name
                ));
            }
            let backend = Backend::restore(
                snapshot.backend,
                snapshot.dims,
                &snapshot.index_type,
                snapshot.metric,
                snapshot.dtype,
            )
            .map_err(|e| format!("collection '{}': {e}", snapshot.name))?;
            let scalars = ScalarStore::from_snapshot(snapshot.scalars)
                .map_err(|e| format!("collection '{}': {e}", snapshot.name))?;
            collections.insert(
                snapshot.name.clone(),
                Arc::new(Collection {
                    name: snapshot.name,
                    dims: snapshot.dims,
                    index_type: snapshot.index_type,
                    metric: snapshot.metric,
                    dtype: snapshot.dtype,
                    backend: Mutex::new(backend),
                    scalars: Mutex::new(scalars),
                }),
            );
        }
        Ok(MobileDB {
            collections: Mutex::new(collections),
        })
    }

    /// 创建空数据库 / Create empty database.
    /// EN: Create an empty database.
    pub fn new() -> Self {
        MobileDB {
            collections: Mutex::new(HashMap::new()),
        }
    }

    /// 获取或创建 Collection / Get or create a named collection.
    /// 返回 Arc<Collection>（锁外安全持有）/ Returns Arc so caller can hold reference safely.
    /// EN: Get or create a named collection.
    /// EN: Returns Arc so the caller can hold the reference safely outside the lock.
    pub fn get_or_create(
        &self,
        name: &str,
        dims: usize,
        index_type: &str,
    ) -> Result<Arc<Collection>, String> {
        if name.is_empty() || name.len() > 4096 {
            return Err("invalid collection name".to_string());
        }
        let normalized = match index_type {
            "flat" | "flat-clustered" | "clustered-hnsw" => index_type,
            _ => "clustered-hnsw",
        };
        let (metric, dtype) = Backend::legacy_config(normalized);
        // 校验与构造都放在 entry 内：已存在的集合不应为一次 get 分配
        // 1024×dims 的缓冲，也不应在无效 dims 时 panic。
        // EN: Validation and construction happen inside the entry closure so a
        // get on an existing collection neither allocates a 1024×dims buffer
        // nor panics on invalid dims.
        Backend::validate_config(dims, normalized, metric, dtype)?;
        let mut guard = self.collections.lock().unwrap();
        if let Some(existing) = guard.get(name) {
            return Ok(existing.clone());
        }
        let backend = Backend::new(dims, normalized)?;
        let collection = Arc::new(Collection {
            name: name.to_string(),
            dims,
            index_type: normalized.to_string(),
            metric,
            dtype,
            backend: Mutex::new(backend),
            scalars: Mutex::new(ScalarStore::new()),
        });
        guard.insert(name.to_string(), collection.clone());
        Ok(collection)
    }

    /// 显式 metric/dtype 创建；同名不同配置返回 Err（不做静默覆盖）。
    /// EN: Create with explicit metric/dtype; a same-name collection with a
    /// different configuration returns Err instead of being silently reused.
    pub fn get_or_create_with_config(
        &self,
        name: &str,
        dims: usize,
        index_type: &str,
        metric: DistanceMetric,
        dtype: FlatStorageDtype,
    ) -> Result<Arc<Collection>, String> {
        Backend::validate_config(dims, index_type, metric, dtype)?;
        let mut guard = self.collections.lock().unwrap();
        if let Some(existing) = guard.get(name) {
            if existing.dims != dims
                || existing.index_type != index_type
                || existing.metric != metric
                || existing.dtype != dtype
            {
                return Err(format!(
                    "collection '{name}' already exists with a different configuration"
                ));
            }
            return Ok(existing.clone());
        }
        let backend = Backend::new_with_config(dims, index_type, metric, dtype)?;
        let collection = Arc::new(Collection {
            name: name.to_string(),
            dims,
            index_type: index_type.to_string(),
            metric,
            dtype,
            backend: Mutex::new(backend),
            scalars: Mutex::new(ScalarStore::new()),
        });
        guard.insert(name.to_string(), collection.clone());
        Ok(collection)
    }

    /// 获取 Collection / Get a collection by name.
    /// EN: Get a collection by name.
    pub fn get(&self, name: &str) -> Option<Arc<Collection>> {
        self.collections.lock().unwrap().get(name).cloned()
    }

    /// 插入向量到指定 Collection / Insert vector into named collection.
    /// EN: Insert a vector into a named collection.
    pub fn insert(&self, coll_name: &str, id: u64, vector: &[f32]) -> Result<(), String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        if vector.len() != col.dims {
            return Err(format!(
                "{ERR_INVALID}dim mismatch: {} vs {}",
                vector.len(),
                col.dims
            ));
        }
        let result = col.backend.lock().unwrap().insert(id, vector);
        result
    }

    /// 批量插入：一次加锁、按顺序应用；失败返回首个错误（之前的行已生效）。
    /// EN: Batch insert with one lock, applied in order; returns the first
    /// error (rows before it remain applied).
    pub fn insert_batch(
        &self,
        coll_name: &str,
        ids: &[u64],
        vectors: &[f32],
        dims: usize,
    ) -> Result<(), String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        if dims != col.dims {
            return Err(format!("{ERR_INVALID}dim mismatch: {dims} vs {}", col.dims));
        }
        let expected = ids
            .len()
            .checked_mul(dims)
            .ok_or_else(|| "batch size overflow".to_string())?;
        if vectors.len() != expected {
            return Err(format!(
                "{ERR_INVALID}batch vectors have {} values, expected {expected}",
                vectors.len()
            ));
        }
        let mut guard = col.backend.lock().unwrap();
        for (row, &id) in ids.iter().enumerate() {
            let start = row * dims;
            guard.insert(id, &vectors[start..start + dims])?;
        }
        Ok(())
    }

    /// 调整 Collection 的候选预算（每 k）；仅 flat-clustered 生效。
    /// EN: Adjust the collection's candidate budget per k; flat-clustered only.
    pub fn set_search_target(&self, coll_name: &str, per_k: usize) -> Result<(), String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        let result = col.backend.lock().unwrap().set_search_target(per_k);
        result
    }

    /// 构建 Collection 索引 / Build collection index.
    /// EN: Build the collection index.
    pub fn build(&self, coll_name: &str) -> Result<usize, String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        let mut guard = col.backend.lock().unwrap();
        Ok(guard.build())
    }

    /// 在 Collection 中检索 / Search within a collection.
    /// EN: Search within a collection.
    pub fn search(
        &self,
        coll_name: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(u64, f32)>, String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        if query.len() != col.dims {
            return Err(format!(
                "{ERR_INVALID}dim mismatch: {} vs {}",
                query.len(),
                col.dims
            ));
        }
        let guard = col.backend.lock().unwrap();
        Ok(guard.search(query, k))
    }

    /// Collection 向量数 / Vector count in a collection.
    /// EN: Vector count in a collection.
    pub fn count(&self, coll_name: &str) -> Result<usize, String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        let guard = col.backend.lock().unwrap();
        Ok(guard.len())
    }

    /// 列出所有 Collection / List all collection names.
    /// EN: List all collection names.
    pub fn list(&self) -> Vec<String> {
        self.collections.lock().unwrap().keys().cloned().collect()
    }

    /// 插入标量标签 / Insert scalar tags for a row.
    /// `tags`: [("col_name", "value"), ...] — value 格式: "int:42" | "float:0.8" | "text:beach" | "ts:1717200000"
    /// EN: Insert scalar tags for a row.
    /// EN: `tags`: [("col_name", "value"), ...] — value format: "int:42" | "float:0.8"
    /// EN: | "text:beach" | "ts:1717200000"
    pub fn insert_scalars(&self, coll_name: &str, tags: &[(&str, &str)]) -> Result<(), String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        let row: Vec<(&str, ScalarVal)> = tags
            .iter()
            .map(|(col_name, val)| parse_scalar(val).map(|sv| (*col_name, sv)))
            .collect::<Result<_, _>>()?;
        col.scalars.lock().unwrap().push_row(&row);
        Ok(())
    }

    /// 以显式向量 id 写入标量行；混合检索按 id 关联（旧 API 仍按行号兼容）。
    /// EN: Insert scalar tags for an explicit vector id; hybrid search associates
    /// by id (the legacy API keeps row-index compatibility).
    pub fn insert_scalars_with_id(
        &self,
        coll_name: &str,
        id: u64,
        tags: &[(&str, &str)],
    ) -> Result<(), String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        let row: Vec<(&str, ScalarVal)> = tags
            .iter()
            .map(|(col_name, val)| parse_scalar(val).map(|sv| (*col_name, sv)))
            .collect::<Result<_, _>>()?;
        col.scalars.lock().unwrap().upsert_row_with_id(id, &row);
        Ok(())
    }

    /// 混合检索：先结构化过滤 → 再向量检索 / Hybrid search: filter → vector ANN.
    /// EN: Hybrid search: structured filter first → then vector search.
    pub fn search_hybrid(
        &self,
        coll_name: &str,
        query: &[f32],
        k: usize,
        filters: &[FilterOp],
    ) -> Result<Vec<(u64, f32)>, String> {
        let col = self
            .get(coll_name)
            .ok_or_else(|| format!("collection '{coll_name}' not found"))?;
        if query.len() != col.dims {
            return Err(format!(
                "{ERR_INVALID}dim mismatch: {} vs {}",
                query.len(),
                col.dims
            ));
        }
        let filter_ids = {
            let guard = col.scalars.lock().unwrap();
            guard.filter_ids(filters)
        };
        let guard = col.backend.lock().unwrap();
        // `None` = 无过滤条件；`Some(empty)` = 有条件但无命中 → 结果必须为空。
        // 传入后端做索引内过滤，避免先搜后过滤导致高选择性时不足 k。
        // EN: `None` = no filters; `Some(empty)` = matched nothing — the latter
        // must yield an empty result, not skip the mask. The mask goes into the
        // backend so selective filters do not shrink post-filtered results.
        match filter_ids {
            None => Ok(guard.search(query, k)),
            Some(filter_ids) => {
                let allowed: ahash::AHashSet<u64> = filter_ids.into_iter().collect();
                Ok(guard.search_filtered(query, k, Some(&allowed)))
            }
        }
    }

    /// 删除 Collection / Drop a collection.
    /// EN: Drop a collection.
    pub fn drop(&self, name: &str) -> bool {
        self.collections.lock().unwrap().remove(name).is_some()
    }
}
