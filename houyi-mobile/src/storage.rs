//! 轻量向量存储：内存池 + 可选 mmap 持久化。
//! Lightweight vector storage: in-memory pool + optional mmap persistence.
//!
//! 移动端内存受限，使用平坦数组 + ID 索引，避免 HashMap 开销。
//! Mobile-friendly: flat arrays + ID lookup, minimal overhead.

use std::collections::HashMap;

/// 向量存储（内存池）。
/// EN: Vector store (in-memory pool).
pub struct VectorStore {
    dims: usize,
    ids: Vec<u64>,
    vectors: Vec<f32>,
    id_to_idx: HashMap<u64, usize>,
}

impl VectorStore {
    pub fn new(dims: usize, capacity: usize) -> Self {
        VectorStore {
            dims,
            ids: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity * dims),
            id_to_idx: HashMap::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// 插入向量（幂等覆盖）。
    /// Insert or update a vector.
    pub fn upsert(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dims, "dim mismatch");
        if let Some(&idx) = self.id_to_idx.get(&id) {
            // 覆盖已有向量
            // EN: Overwrite an existing vector.
            let start = idx * self.dims;
            self.vectors[start..start + self.dims].copy_from_slice(vector);
        } else {
            let idx = self.ids.len();
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);
            self.id_to_idx.insert(id, idx);
        }
    }

    /// 按内部索引获取向量切片。
    /// EN: Get a vector slice by internal index.
    #[cfg(test)]
    pub fn get_by_idx(&self, idx: usize) -> Option<&[f32]> {
        if idx < self.ids.len() {
            let start = idx * self.dims;
            Some(&self.vectors[start..start + self.dims])
        } else {
            None
        }
    }

    /// 获取原始 ID 列表和向量数据（供 Flat 检索）。
    /// EN: Get the raw ID list and vector data (for Flat search).
    pub fn raw_data(&self) -> (&[u64], &[f32]) {
        (&self.ids, &self.vectors)
    }

    /// 校验维度后写入；FFI 边界用 Result 代替 assert，避免跨 C ABI panic。
    /// EN: Validate dimensions then upsert; the FFI boundary returns a Result
    /// instead of asserting so a bad length cannot panic across the C ABI.
    pub fn try_upsert(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dims {
            return Err(format!(
                "dim mismatch: expected {}, got {}",
                self.dims,
                vector.len()
            ));
        }
        self.upsert(id, vector);
        Ok(())
    }

    /// 导出逐行 (id, vector) 快照（持久化用）。
    /// EN: Export row-wise (id, vector) pairs for persistence.
    pub fn rows(&self) -> Vec<(u64, Vec<f32>)> {
        let dims = self.dims;
        self.ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, self.vectors[i * dims..(i + 1) * dims].to_vec()))
            .collect()
    }

    /// 取走 ID/向量数据用于建图（保留空存储占位，随后由索引状态替换）。
    /// EN: Take ids/vectors for a build (leaves an empty store placeholder that
    /// the backend state replaces with the built index).
    pub fn take_for_build(&mut self) -> (Vec<u64>, Vec<f32>) {
        let ids = std::mem::take(&mut self.ids);
        let vectors = std::mem::take(&mut self.vectors);
        self.id_to_idx.clear();
        (ids, vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_and_lookup() {
        let mut store = VectorStore::new(4, 10);
        store.upsert(1, &[0.1, 0.2, 0.3, 0.4]);
        store.upsert(2, &[0.5, 0.6, 0.7, 0.8]);
        assert_eq!(store.len(), 2);
        assert_eq!(store.get_by_idx(0).unwrap(), &[0.1, 0.2, 0.3, 0.4]);

        // 幂等覆盖
        // EN: Idempotent overwrite.
        store.upsert(1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store.len(), 2);
        assert_eq!(store.get_by_idx(0).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
