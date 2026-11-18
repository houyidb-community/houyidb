//! PQ-FlatClustered: K-Means 分区 + 全局 PQ 码本 + mmap 重排序。
//! PQ in-memory (16MB), vectors on disk. Two-phase: PQ coarse → exact rerank.
//!
//! 内存: 128D × 1M → 16MB codes + 8MB IDs + 128KB codebook ≈ 24MB in RAM
//! 磁盘: 512MB vectors via mmap (按需读取, 每次查 ~200KB)

use super::kmeans;
use super::router;
use crate::pq::{PqConfig, PqIndex};
use crate::DistanceMetric;
use rayon::prelude::*;
use std::fs;
use std::io::Read;

/// 快照状态魔数。
/// EN: Snapshot state magic.
const PQ_STATE_MAGIC: &[u8; 8] = b"HOUYIPQ1";
/// 头部长度：magic(8) + 8 × u32。
/// EN: Header length: magic (8) + 8 × u32.
const PQ_STATE_HEADER_LEN: usize = 8 + 8 * 4;
/// PQ 码本大小上限（8-bit 码）。
/// EN: Upper bound of the PQ codebook size (8-bit codes).
const MAX_PQ_K: usize = 256;

pub struct PqFlatClustered {
    dims: usize,
    num_clusters: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    cluster_ids: Vec<Vec<u64>>,
    cluster_codes: Vec<Vec<u8>>,
    codebook: Vec<f32>,
    pq_config: PqConfig,
    /// mmap 文件路径（存储全量原始向量，用于重排序）
    /// EN: mmap file path (stores all raw vectors for reranking).
    vecs_path: Option<String>,
}

impl PqFlatClustered {
    pub fn build(
        vectors: &[f32],
        ids: &[u64],
        dims: usize,
        metric: DistanceMetric,
        num_clusters: usize,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        Self::build_with_mmap(
            vectors,
            ids,
            dims,
            metric,
            num_clusters,
            kmeans_iters,
            seed,
            None,
        )
    }

    /// 构建并同时写 mmap 文件。mmap 模式下重排按「id == 输入行号」定位原始行，
    /// 入口强制顺序 id（`0..n`），不满足即 panic。
    // 构造参数即构建配置全集；引入 builder 对单一调用点无收益。
    // EN: Builds and writes the mmap file at the same time. In mmap mode rerank
    // locates raw rows by `id == input row index`; the entry point enforces
    // sequential ids (`0..n`) and panics otherwise.
    // EN: Constructor params are the full build config; a builder adds no value here.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_mmap(
        vectors: &[f32],
        ids: &[u64],
        dims: usize,
        _metric: DistanceMetric,
        num_clusters: usize,
        kmeans_iters: usize,
        seed: u64,
        mmap_path: Option<&str>,
    ) -> Self {
        use std::borrow::Cow;

        assert!(dims > 0, "pq-flat build: dims must be > 0");
        let expected_len = ids
            .len()
            .checked_mul(dims)
            .expect("pq-flat build: vectors/ids shape overflow");
        assert_eq!(
            vectors.len(),
            expected_len,
            "pq-flat build: vectors/ids shape mismatch"
        );

        // mmap 重排按「id == 输入行号」定位原始行，非顺序 id 会静默重排错行；
        // 这里强制契约（顺序 id 0..n），不满足即 fail-fast。
        // EN: mmap rerank locates raw rows by `id == input row index`; any other
        // id layout would silently rerank the wrong rows. Enforce the contract
        // (sequential ids 0..n) and fail fast otherwise.
        if mmap_path.is_some() {
            for (i, &id) in ids.iter().enumerate() {
                assert_eq!(
                    id, i as u64,
                    "pq-flat mmap build requires sequential ids (id == row index)"
                );
            }
        }

        // 同一 id 的多次写入采用最后写入获胜：懒缓冲在构建前可能积累同 id 多行，
        // 去重后编码态才可被解码器接受，且与服务端 last-write-wins 语义一致。
        // 无重复时不复制，避免大集合的整表拷贝。
        // EN: Repeated writes of the same id use last-write-wins: the lazy buffer
        // may hold several rows per id before a build, and deduping keeps the
        // encoded state decodable while matching server last-write-wins semantics.
        // With no duplicates nothing is copied (no full-matrix copy).
        let n_in = ids.len();
        let (deduped_ids, deduped_vectors): (Cow<'_, [u64]>, Cow<'_, [f32]>) = {
            let mut last_idx: ahash::AHashMap<u64, usize> = ahash::AHashMap::with_capacity(n_in);
            for (i, &id) in ids.iter().enumerate() {
                last_idx.insert(id, i);
            }
            if last_idx.len() < n_in {
                let keep: Vec<usize> = (0..n_in).filter(|&i| last_idx[&ids[i]] == i).collect();
                let mut dids = Vec::with_capacity(keep.len());
                let mut dvecs = Vec::with_capacity(keep.len() * dims);
                for &i in &keep {
                    dids.push(ids[i]);
                    dvecs.extend_from_slice(&vectors[i * dims..(i + 1) * dims]);
                }
                (Cow::Owned(dids), Cow::Owned(dvecs))
            } else {
                (Cow::Borrowed(ids), Cow::Borrowed(vectors))
            }
        };
        let n = deduped_ids.len();
        let di = deduped_ids.as_ref();
        let dv = deduped_vectors.as_ref();

        let pq_cfg = PqConfig::for_dims(dims);
        let sub_dims = pq_cfg.sub_dims(dims);
        let m = pq_cfg.m;
        let k = pq_cfg.k;

        // 写 mmap 文件（原始向量，扁平存储）：按输入行号原样写（不去重），
        // 与重排的 id == 行号定位契约一致；聚类态可用去重后的向量。
        // EN: Write the mmap file (raw vectors, flat layout) in input row order
        // (no dedup), matching the id == row-index rerank contract; the cluster
        // state may use the deduped vectors.
        if let Some(path) = mmap_path {
            let mut data = Vec::with_capacity(16 + n_in * dims * 4);
            data.extend_from_slice(&(n_in as u64).to_le_bytes());
            data.extend_from_slice(&(dims as u64).to_le_bytes());
            for v in vectors.chunks_exact(dims) {
                for &x in v {
                    data.extend_from_slice(&x.to_le_bytes());
                }
            }
            fs::write(path, &data).expect("write mmap");
        }

        // 1) K-Means（kmeans 实际返回 min(num_clusters, n) 个中心，n=1 时为 1）。
        // EN: 1) K-Means (returns min(num_clusters, n) centroids, i.e. 1 when n=1).
        let km = kmeans::kmeans(
            dv,
            dims,
            num_clusters.min(n).max(2),
            kmeans_iters,
            seed,
            DistanceMetric::L2,
        );
        // 存储字段必须用实际中心数，否则 n=1 时状态会写出 num_clusters=2 而只有
        // 1 个中心，解码器无法闭合。cluster_sz/nprobe 同样按实际数推导。
        // EN: Use the actual centroid count; storing the requested 2 for n=1 would
        // desync num_clusters from centroids and break decode. cluster_sz/nprobe
        // derive from the actual count too.
        let c = km.centroids.len() / dims;

        // 2) 全局 PQ 码本
        // EN: 2) Global PQ codebook.
        let global_pq = PqIndex::train(dv, dims, &pq_cfg);
        let codebook = global_pq.codebooks;

        // 3) 分配 + 分组
        // EN: 3) Assign + group.
        let assignments = router::assign_vectors(dv, dims, &km.centroids, DistanceMetric::L2);
        let groups = router::group_by_cluster(&assignments, c);

        // 4) 每聚类编码
        // EN: 4) Encode each cluster.
        let cluster_data: Vec<(Vec<u64>, Vec<u8>)> = groups
            .par_iter()
            .map(|vec_indices| {
                let mut cids = Vec::with_capacity(vec_indices.len());
                let mut codes = Vec::with_capacity(vec_indices.len() * m);
                for &vi in vec_indices {
                    let v = &dv[vi * dims..(vi + 1) * dims];
                    cids.push(di[vi]);
                    for seg in 0..m {
                        let sub = &v[seg * sub_dims..(seg + 1) * sub_dims];
                        let cb_start = seg * k * sub_dims;
                        let best = nearest_centroid_u8(
                            sub,
                            &codebook[cb_start..cb_start + k * sub_dims],
                            sub_dims,
                            k,
                        );
                        codes.push(best);
                    }
                }
                (cids, codes)
            })
            .collect();
        let (cluster_ids, cluster_codes): (Vec<_>, Vec<_>) = cluster_data.into_iter().unzip();

        let cluster_sz = n / c;
        let target = 25_000;
        // 下界不得超过上界：c ∈ {1,2,3}（小数据集）时 clamp(4, c) 会 panic。
        // EN: lower bound must not exceed upper bound — clamp(4, c) panics for c < 4.
        let nprobe = (target / cluster_sz.max(1)).clamp(4.min(c), c);

        PqFlatClustered {
            dims,
            num_clusters: c,
            nprobe,
            centroids: km.centroids,
            cluster_ids,
            cluster_codes,
            codebook,
            pq_config: pq_cfg,
            vecs_path: mmap_path.map(|s| s.to_string()),
        }
    }

    /// 两阶段检索: PQ 粗筛 → 从 mmap 读原始向量精确重排。
    /// EN: Two-stage search: PQ coarse filter → read raw vectors from mmap and rerank exactly.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if query.len() != self.dims {
            return Vec::new();
        }
        let k = k.max(1);
        let m = self.pq_config.m;
        let cb_k = self.pq_config.k;
        let sub_dims = self.pq_config.sub_dims(self.dims);

        let total = self.total_vectors();
        let cluster_sz = (total / self.num_clusters).max(1);
        let target = k.saturating_mul(2500);
        let nprobe = self.nprobe.min((target / cluster_sz).max(4));

        let top_clusters = router::route_query(
            query,
            &self.centroids,
            self.dims,
            nprobe,
            DistanceMetric::L2,
        );

        // Phase 1: PQ 查表粗筛
        // EN: Phase 1: PQ table-lookup coarse filter.
        let mut dist_table = vec![0.0f32; m * cb_k];
        for seg in 0..m {
            let q_sub = &query[seg * sub_dims..(seg + 1) * sub_dims];
            for c in 0..cb_k {
                let start = seg * cb_k * sub_dims + c * sub_dims;
                dist_table[seg * cb_k + c] =
                    l2_dist(q_sub, &self.codebook[start..start + sub_dims], sub_dims);
            }
        }

        let coarse_k = k.saturating_mul(20).max(500); // PQ粗筛候选数, 调此参数平衡 recall/latency
        let mut candidates: Vec<(u64, f32)> = top_clusters
            .par_iter()
            .flat_map(|&cid| {
                let codes = &self.cluster_codes[cid];
                let ids = &self.cluster_ids[cid];
                let n = codes.len() / m;
                if n == 0 {
                    return Vec::new();
                }
                let mut local: Vec<(u64, f32)> = (0..n)
                    .map(|i| {
                        let mut d = 0.0;
                        for seg in 0..m {
                            d += dist_table[seg * cb_k + codes[i * m + seg] as usize];
                        }
                        (ids[i], d)
                    })
                    .collect();
                local.sort_by(|a, b| a.1.total_cmp(&b.1));
                local.truncate(coarse_k);
                local
            })
            .collect();
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        // 聚类态 id 全局唯一（构建去重、快照解码拒绝重复，且每 id 只属一个簇），
        // 跨簇合并天然无重复，无需每查询分配去重集合。
        // EN: Cluster-state ids are globally unique (build dedups, snapshot decode
        // rejects duplicates, and each id belongs to one cluster), so merged
        // candidates are duplicate-free; no per-query dedup allocation is needed.
        candidates.truncate(coarse_k);

        // Phase 2: 从 mmap 读原始向量，精确重排
        // EN: Phase 2: read raw vectors from mmap and rerank exactly.
        if let Some(ref path) = self.vecs_path {
            let mut file = match fs::File::open(path) {
                Ok(f) => f,
                Err(_) => {
                    candidates.truncate(k);
                    return candidates;
                }
            };
            // 读取 header: [n:u64][dims:u64]
            // EN: Read header: [n:u64][dims:u64].
            let mut h = [0u8; 16];
            file.read_exact(&mut h).unwrap();
            let file_n = u64::from_le_bytes(h[0..8].try_into().unwrap()) as usize;
            let file_dims = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
            assert_eq!(file_dims, self.dims);

            let mut reranked: Vec<(u64, f32)> = candidates
                .iter()
                .filter_map(|(id, _)| {
                    // 需要根据 id 找到在原始文件中的位置
                    // id → 原始索引: 简单情况 id = 原始索引
                    // EN: Need to locate the id in the raw file.
                    // EN: id → raw index: in the simple case id = raw index.
                    let idx = *id as usize;
                    if idx >= file_n {
                        return None;
                    }
                    let offset = 16 + idx * self.dims * 4;
                    let mut buf = vec![0u8; self.dims * 4];
                    file.read_exact_at(&mut buf, offset as u64).ok()?;
                    let mut v = vec![0.0f32; self.dims];
                    for i in 0..self.dims {
                        v[i] = f32::from_le_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap());
                    }
                    let d = l2_dist(query, &v, self.dims);
                    Some((*id, d))
                })
                .collect();

            reranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            reranked.truncate(k);
            return reranked;
        }

        // 无 mmap 回退: 纯 PQ 结果
        // EN: No-mmap fallback: pure PQ results.
        candidates.truncate(k);
        candidates
    }

    pub fn total_vectors(&self) -> usize {
        self.cluster_ids.iter().map(|v| v.len()).sum()
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// 编码紧凑构建态快照（无原始向量：服务端构建时未落地 mmap 文件）。
    /// 线格式：
    /// `magic(8) | dims,nclusters,nprobe,m,k,sub_dims,centroid_len,codebook_len (u32 BE)`
    /// `| centroids(f32 BE) | codebook(f32 BE) | 每聚类 { count(u32 BE), ids(u64 BE)×count, codes(u8)×count×m }`
    ///
    /// 快照恢复时原样装回构建态（不做重训/重分配），检索结果逐位不变。
    /// 编码器拒绝 mmap 构建态（原始向量未随状态导出，恢复会静默降级为纯 PQ）
    /// 并防御重复 id，保证「编码器输出必可被解码器接受」闭合。
    /// EN: Encodes the compact built state (no raw vectors: the server build
    /// never wrote an mmap file). Wire layout:
    /// `magic (8) | dims,nclusters,nprobe,m,k,sub_dims,centroid_len,codebook_len (u32 BE)`
    /// `| centroids (f32 BE) | codebook (f32 BE) | per cluster { count (u32 BE), ids (u64 BE) × count, codes (u8) × count × m }`.
    /// Restoring installs the built state verbatim (no retraining/reassignment),
    /// so search results stay bit-identical. The encoder rejects mmap-built
    /// states (raw vectors are not exported and restore would silently degrade
    /// to pure PQ) and guards against duplicate ids, keeping the
    /// encoder/decoder pair closed.
    pub fn encode_snapshot_state(&self) -> Result<Vec<u8>, String> {
        if self.vecs_path.is_some() {
            return Err(
                "pq state: mmap-built state carries no raw vectors; snapshot would \
                 silently degrade rerank results"
                    .to_string(),
            );
        }
        let m = self.pq_config.m;
        let sub_dims = self.pq_config.sub_dims(self.dims);
        let mut out = Vec::new();
        out.extend_from_slice(PQ_STATE_MAGIC);
        for v in [
            self.dims,
            self.num_clusters,
            self.nprobe,
            m,
            self.pq_config.k,
            sub_dims,
            self.centroids.len(),
            self.codebook.len(),
        ] {
            out.extend_from_slice(&(v as u32).to_be_bytes());
        }
        for &x in &self.centroids {
            out.extend_from_slice(&x.to_be_bytes());
        }
        for &x in &self.codebook {
            out.extend_from_slice(&x.to_be_bytes());
        }
        let mut ids_seen = ahash::AHashSet::with_capacity(self.total_vectors());
        for (ids, codes) in self.cluster_ids.iter().zip(self.cluster_codes.iter()) {
            out.extend_from_slice(&(ids.len() as u32).to_be_bytes());
            for &id in ids {
                if !ids_seen.insert(id) {
                    return Err(format!("pq state: duplicate id {id}"));
                }
                out.extend_from_slice(&id.to_be_bytes());
            }
            out.extend_from_slice(codes);
        }
        Ok(out)
    }

    /// 解码构建态快照并做全量一致性校验（magic/长度/字段语义/浮点有限/码值/重复 id）。
    /// 解码结果无 mmap 回退（`vecs_path = None`），检索走纯 PQ 路径。
    /// EN: Decodes the built-state snapshot with full consistency validation
    /// (magic/length/field semantics/finite floats/code range/duplicate ids).
    /// The decoded index has no mmap path (`vecs_path = None`) and searches via
    /// the pure-PQ path.
    pub fn decode_snapshot_state(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < PQ_STATE_HEADER_LEN {
            return Err("pq state truncated: header".to_string());
        }
        if &bytes[..8] != PQ_STATE_MAGIC {
            return Err("pq state bad magic".to_string());
        }
        let mut cur = 8usize;
        let dims = read_u32(bytes, &mut cur)? as usize;
        let num_clusters = read_u32(bytes, &mut cur)? as usize;
        let nprobe = read_u32(bytes, &mut cur)? as usize;
        let m = read_u32(bytes, &mut cur)? as usize;
        let k = read_u32(bytes, &mut cur)? as usize;
        let sub_dims = read_u32(bytes, &mut cur)? as usize;
        let centroid_len = read_u32(bytes, &mut cur)? as usize;
        let codebook_len = read_u32(bytes, &mut cur)? as usize;

        if dims == 0 {
            return Err("pq state: dims must be > 0".to_string());
        }
        if m == 0 || sub_dims == 0 || dims != m.saturating_mul(sub_dims) {
            return Err("pq state: dims != m * sub_dims".to_string());
        }
        if k == 0 || k > MAX_PQ_K {
            return Err(format!("pq state: k out of range (1..={MAX_PQ_K})"));
        }
        // n=1 的集合实际只有 1 个中心；允许 1 个簇以闭合编码器输出。
        // EN: A single-row collection has exactly one centroid; allow a single
        // cluster so the encoder's output always decodes.
        if num_clusters == 0 {
            return Err("pq state: num_clusters must be >= 1".to_string());
        }
        if nprobe == 0 || nprobe > num_clusters {
            return Err("pq state: nprobe out of range".to_string());
        }
        if centroid_len != num_clusters.saturating_mul(dims) {
            return Err("pq state: centroid_len mismatch".to_string());
        }
        if codebook_len != m.saturating_mul(k).saturating_mul(sub_dims) {
            return Err("pq state: codebook_len mismatch".to_string());
        }

        // 先按剩余字节校验浮点段长度，防止伪造头部触发超大分配。
        // EN: Bound the float section against the actual byte buffer before
        // allocating, so forged headers cannot trigger huge allocations.
        let rest = (bytes.len() - cur) as u64;
        let float_bytes = ((centroid_len as u64) + (codebook_len as u64))
            .checked_mul(4)
            .ok_or_else(|| "pq state: float section length overflow".to_string())?;
        if float_bytes > rest {
            return Err("pq state truncated: floats".to_string());
        }

        let mut centroids = Vec::with_capacity(centroid_len);
        for _ in 0..centroid_len {
            let x = read_f32(bytes, &mut cur)?;
            if !x.is_finite() {
                return Err("pq state: non-finite centroid".to_string());
            }
            centroids.push(x);
        }
        let mut codebook = Vec::with_capacity(codebook_len);
        for _ in 0..codebook_len {
            let x = read_f32(bytes, &mut cur)?;
            if !x.is_finite() {
                return Err("pq state: non-finite codebook".to_string());
            }
            codebook.push(x);
        }

        let mut cluster_ids: Vec<Vec<u64>> = Vec::with_capacity(num_clusters);
        let mut cluster_codes: Vec<Vec<u8>> = Vec::with_capacity(num_clusters);
        let mut ids_seen = ahash::AHashSet::new();
        for _ in 0..num_clusters {
            let count = read_u32(bytes, &mut cur)? as usize;
            let need = (count as u64)
                .checked_mul(8 + m as u64)
                .ok_or_else(|| "pq state: cluster block length overflow".to_string())?;
            if need > (bytes.len() - cur) as u64 {
                return Err("pq state truncated: cluster block".to_string());
            }
            let mut ids = Vec::with_capacity(count);
            for _ in 0..count {
                let id = read_u64(bytes, &mut cur)?;
                if !ids_seen.insert(id) {
                    return Err(format!("pq state: duplicate id {id}"));
                }
                ids.push(id);
            }
            let code_len = count
                .checked_mul(m)
                .ok_or_else(|| "pq state: code length overflow".to_string())?;
            let codes = bytes[cur..cur + code_len].to_vec();
            cur += code_len;
            if k < MAX_PQ_K {
                if let Some(&bad) = codes.iter().find(|&&c| c as usize >= k) {
                    return Err(format!("pq state: code {bad} out of range (k={k})"));
                }
            }
            cluster_ids.push(ids);
            cluster_codes.push(codes);
        }
        if cur != bytes.len() {
            return Err("pq state: trailing bytes".to_string());
        }

        Ok(PqFlatClustered {
            dims,
            num_clusters,
            nprobe,
            centroids,
            cluster_ids,
            cluster_codes,
            codebook,
            // train_iters 只在训练期使用，装回构建态时不再训练，取值不影响检索。
            // EN: train_iters only matters during training; restoring never
            // retrains, so the value does not affect search.
            pq_config: PqConfig {
                m,
                k,
                train_iters: 15,
            },
            vecs_path: None,
        })
    }

    pub fn memory_mb(&self) -> f64 {
        let cb = self.codebook.len() * 4;
        let ib: usize = self.cluster_ids.iter().map(|v| v.len() * 8).sum();
        let cob: usize = self.cluster_codes.iter().map(|c| c.len()).sum();
        (cb + ib + cob) as f64 / 1024.0 / 1024.0
    }
    pub fn disk_mb(&self) -> f64 {
        self.vecs_path
            .as_ref()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0)
            .unwrap_or(0.0)
    }
}

/// 快照状态解码的定长读取（越界返回错误而非 panic）。
/// EN: Fixed-width reads for snapshot state decoding (returns an error instead of panicking).
fn read_u32(bytes: &[u8], cur: &mut usize) -> Result<u32, String> {
    let end = (*cur)
        .checked_add(4)
        .ok_or_else(|| "pq state: offset overflow".to_string())?;
    let slice = bytes
        .get(*cur..end)
        .ok_or_else(|| "pq state truncated".to_string())?;
    *cur = end;
    let arr: [u8; 4] = slice
        .try_into()
        .map_err(|_| "pq state: bad u32 field".to_string())?;
    Ok(u32::from_be_bytes(arr))
}

fn read_u64(bytes: &[u8], cur: &mut usize) -> Result<u64, String> {
    let end = (*cur)
        .checked_add(8)
        .ok_or_else(|| "pq state: offset overflow".to_string())?;
    let slice = bytes
        .get(*cur..end)
        .ok_or_else(|| "pq state truncated".to_string())?;
    *cur = end;
    let arr: [u8; 8] = slice
        .try_into()
        .map_err(|_| "pq state: bad u64 field".to_string())?;
    Ok(u64::from_be_bytes(arr))
}

fn read_f32(bytes: &[u8], cur: &mut usize) -> Result<f32, String> {
    Ok(f32::from_bits(read_u32(bytes, cur)?))
}

#[inline]
fn l2_dist(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..len {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn nearest_centroid_u8(sub: &[f32], cb: &[f32], sub_dims: usize, k: usize) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::MAX;
    for c in 0..k {
        let d = l2_dist(sub, &cb[c * sub_dims..(c + 1) * sub_dims], sub_dims);
        if d < best_d {
            best_d = d;
            best = c as u8;
        }
    }
    best
}

// read_exact_at 兼容垫片。
// EN: read_exact_at polyfill.
trait ReadAt {
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> std::io::Result<()>;
}
impl ReadAt for fs::File {
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        use std::io::Seek;
        self.seek(std::io::SeekFrom::Start(offset))?;
        self.read_exact(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceMetric;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// 回归：小数据集（c < 4）时 nprobe 的 clamp 下界不得超过上界，否则 build panic。
    /// EN: regression — for c < 4 the nprobe clamp lower bound must not exceed
    /// the upper bound (pre-existing panic on small collections).
    #[test]
    fn test_pq_flat_small_collection_no_clamp_panic() {
        let dims = 64;
        let n = 6;
        let mut rng = StdRng::seed_from_u64(7);
        let mut vecs = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vecs.push(rng.gen::<f32>());
        }
        let pqf = PqFlatClustered::build(&vecs, &ids, dims, DistanceMetric::L2, 3, 5, 0);
        assert_eq!(pqf.total_vectors(), n);
    }

    /// 快照状态编解码必须逐位保真：解码后检索结果与编码前完全一致。
    /// EN: Snapshot state round-trip must be bit-faithful: search results are
    /// identical before and after the round-trip.
    #[test]
    fn test_pq_flat_state_roundtrip_preserves_search() {
        let dims = 64;
        let n = 200;
        let mut rng = StdRng::seed_from_u64(4242);
        let mut vecs = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vecs.push(rng.gen::<f32>());
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let pqf = PqFlatClustered::build(&vecs, &ids, dims, DistanceMetric::L2, c, 10, 0);

        let bytes = pqf.encode_snapshot_state().expect("encode state");
        let decoded = PqFlatClustered::decode_snapshot_state(&bytes).expect("decode state");
        assert_eq!(decoded.dims(), dims);
        assert_eq!(decoded.num_clusters(), pqf.num_clusters());
        assert_eq!(decoded.total_vectors(), pqf.total_vectors());
        // 无 mmap 路径必须保持纯 PQ 语义（与原索引一致）。
        // EN: The no-mmap path must stay pure-PQ, matching the original index.
        assert_eq!(decoded.disk_mb(), 0.0);

        let mut qrng = StdRng::seed_from_u64(999);
        for _ in 0..10 {
            let q: Vec<f32> = (0..dims).map(|_| qrng.gen::<f32>()).collect();
            let before = pqf.search(&q, 5);
            let after = decoded.search(&q, 5);
            assert_eq!(before, after, "search results must be bit-identical");
        }
    }

    /// 解码必须拒绝腐坏状态：坏 magic / 截断 / 尾随字节 / 重复 id。
    /// EN: Decoding must reject corrupt state: bad magic, truncation, trailing
    /// bytes, duplicate ids.
    #[test]
    fn test_pq_flat_state_decode_rejects_corruption() {
        let dims = 64;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(7);
        let mut vecs = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n * dims {
            vecs.push(rng.gen::<f32>());
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let pqf = PqFlatClustered::build(&vecs, &ids, dims, DistanceMetric::L2, c, 10, 0);
        let bytes = pqf.encode_snapshot_state().expect("encode state");

        // 坏 magic。
        // EN: Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(PqFlatClustered::decode_snapshot_state(&bad).is_err());

        // 截断（去掉最后一个字节）。
        // EN: Truncation (drop the last byte).
        let truncated = &bytes[..bytes.len() - 1];
        assert!(PqFlatClustered::decode_snapshot_state(truncated).is_err());

        // 尾随字节。
        // EN: Trailing byte.
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(PqFlatClustered::decode_snapshot_state(&trailing).is_err());

        // 头部字段不一致：m 改为 m+1 后 dims != m*sub_dims。
        // EN: Inconsistent header: bumping m breaks dims == m*sub_dims.
        let mut bad_m = bytes.clone();
        let m = u32::from_be_bytes(bytes[20..24].try_into().unwrap()) as usize;
        bad_m[20..24].copy_from_slice(&((m + 1) as u32).to_be_bytes());
        assert!(PqFlatClustered::decode_snapshot_state(&bad_m).is_err());

        // 重复 id：把聚类 0 的第二个 id 改成第一个 id。
        // EN: Duplicate id: overwrite cluster 0's second id with the first.
        let mut dup = bytes.clone();
        let first_cluster = first_cluster_block_offset(&dup);
        let count =
            u32::from_be_bytes(dup[first_cluster..first_cluster + 4].try_into().unwrap()) as usize;
        assert!(count >= 2, "cluster 0 must have at least 2 members");
        let id0 = dup[first_cluster + 4..first_cluster + 12].to_vec();
        dup[first_cluster + 12..first_cluster + 20].copy_from_slice(&id0);
        assert!(PqFlatClustered::decode_snapshot_state(&dup).is_err());
    }

    /// 单行集合（n=1）的实际簇数为 1，编码态必须能被解码器闭合接受。
    /// EN: A single-row collection has exactly one cluster; its encoded state
    /// must be accepted by the decoder (encoder/decoder closure).
    #[test]
    fn test_pq_flat_single_vector_state_roundtrip() {
        let dims = 32;
        let vecs: Vec<f32> = (0..dims).map(|i| i as f32 / 16.0).collect();
        let pqf = PqFlatClustered::build(&vecs, &[42], dims, DistanceMetric::L2, 3, 5, 0);
        assert_eq!(pqf.num_clusters(), 1);
        assert_eq!(pqf.total_vectors(), 1);

        let bytes = pqf
            .encode_snapshot_state()
            .expect("encode single-row state");
        let decoded = PqFlatClustered::decode_snapshot_state(&bytes).expect("decode state");
        assert_eq!(decoded.num_clusters(), 1);
        assert_eq!(decoded.total_vectors(), 1);
        assert_eq!(decoded.search(&vecs, 1), pqf.search(&vecs, 1));
    }

    /// 构建前对同一 id 的多次写入必须是最后写入获胜，且编码态无重复 id。
    /// EN: Repeated writes of one id before a build are last-write-wins, and the
    /// encoded state carries no duplicate ids.
    #[test]
    fn test_pq_flat_duplicate_id_build_last_write_wins() {
        let dims = 16;
        let mut rng = StdRng::seed_from_u64(11);
        let row_a: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        let row_b: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        let row_a_last: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        let mut vecs = Vec::new();
        vecs.extend_from_slice(&row_a);
        vecs.extend_from_slice(&row_b);
        vecs.extend_from_slice(&row_a_last);
        let pqf = PqFlatClustered::build(&vecs, &[5, 7, 5], dims, DistanceMetric::L2, 2, 5, 0);
        assert_eq!(pqf.total_vectors(), 2);

        // 去重保留「最后出现位置」顺序：id 7 前、id 5（最后一行）后。
        // EN: Dedup keeps last-occurrence order: id 7 first, id 5 (last row) second.
        let mut expected_vecs = Vec::new();
        expected_vecs.extend_from_slice(&row_b);
        expected_vecs.extend_from_slice(&row_a_last);
        let expected =
            PqFlatClustered::build(&expected_vecs, &[7, 5], dims, DistanceMetric::L2, 2, 5, 0);
        let mut qrng = StdRng::seed_from_u64(99);
        for _ in 0..5 {
            let q: Vec<f32> = (0..dims).map(|_| qrng.gen()).collect();
            assert_eq!(pqf.search(&q, 2), expected.search(&q, 2));
        }

        let bytes = pqf.encode_snapshot_state().expect("encode deduped state");
        let decoded = PqFlatClustered::decode_snapshot_state(&bytes).expect("decode state");
        assert_eq!(decoded.total_vectors(), 2);
    }

    /// mmap 模式下非顺序 id 会使重排按 id 定位到错误原始行，必须 fail-fast。
    /// EN: Non-sequential ids in mmap mode would rerank against wrong raw rows;
    /// the build must fail fast.
    #[test]
    #[should_panic(expected = "sequential ids")]
    fn test_pq_flat_mmap_build_rejects_non_sequential_ids() {
        let dims = 8;
        let ids: Vec<u64> = vec![0, 1, 0];
        let vecs: Vec<f32> = (0..3 * dims).map(|i| (i % 13) as f32 / 13.0).collect();
        // 顺序 id 断言在写 mmap 文件之前触发 panic，文件不会落盘，无需清理。
        // EN: The sequential-id assert panics before the mmap file is written,
        // so no file is created and no cleanup is needed.
        let tmp = std::env::temp_dir().join(format!("pq_mmap_dup_{}.bin", std::process::id()));
        PqFlatClustered::build_with_mmap(
            &vecs,
            &ids,
            dims,
            DistanceMetric::L2,
            2,
            3,
            0,
            Some(tmp.to_str().unwrap()),
        );
    }

    /// mmap 文件按输入行号原样写入，供重排按 id 定位原始行。
    /// EN: The mmap file keeps every input row in input order so rerank can
    /// locate raw rows by id.
    #[test]
    fn test_pq_flat_mmap_file_preserves_input_rows() {
        let dims = 8;
        let ids: Vec<u64> = vec![0, 1, 2];
        let vecs: Vec<f32> = (0..3 * dims).map(|i| (i % 13) as f32 / 13.0).collect();
        let tmp = std::env::temp_dir().join(format!("pq_mmap_{}.bin", std::process::id()));
        let pqf = PqFlatClustered::build_with_mmap(
            &vecs,
            &ids,
            dims,
            DistanceMetric::L2,
            2,
            3,
            0,
            Some(tmp.to_str().unwrap()),
        );
        assert_eq!(pqf.total_vectors(), 3, "sequential ids keep every row");
        let data = fs::read(&tmp).expect("read mmap");
        assert_eq!(
            u64::from_le_bytes(data[..8].try_into().unwrap()),
            3,
            "mmap header keeps the input row count"
        );
        assert_eq!(
            data.len(),
            16 + 3 * dims * 4,
            "mmap body keeps every input row"
        );
        for (i, v) in vecs.iter().enumerate() {
            let off = 16 + i * 4;
            let got = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            assert_eq!(got.to_bits(), v.to_bits(), "row {i} must match input");
        }
        let _ = fs::remove_file(tmp);
    }

    /// mmap 构建态不携带原始向量，快照编码必须显式拒绝而非静默降级。
    /// EN: An mmap-built state carries no raw vectors; snapshot encoding must
    /// fail explicitly instead of silently degrading rerank.
    #[test]
    fn test_pq_flat_encode_rejects_mmap_state() {
        let dims = 8;
        let n = 32;
        let vecs: Vec<f32> = (0..n * dims).map(|i| (i % 17) as f32 / 17.0).collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let tmp = std::env::temp_dir().join(format!("pq_state_{}.bin", std::process::id()));
        let pqf = PqFlatClustered::build_with_mmap(
            &vecs,
            &ids,
            dims,
            DistanceMetric::L2,
            4,
            5,
            0,
            Some(tmp.to_str().unwrap()),
        );
        let err = pqf
            .encode_snapshot_state()
            .expect_err("mmap state must not encode");
        assert!(err.contains("raw vectors"), "unexpected error: {err}");
        let _ = fs::remove_file(tmp);
    }

    /// 定位第一个聚类块的字节偏移（仅测试用，按线格式手工计算）。
    /// EN: Byte offset of the first cluster block (test-only manual walk).
    fn first_cluster_block_offset(bytes: &[u8]) -> usize {
        let u32_at =
            |i: usize| u32::from_be_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap());
        let centroid_len = u32_at(6) as usize;
        let codebook_len = u32_at(7) as usize;
        8 + 8 * 4 + centroid_len * 4 + codebook_len * 4
    }

    #[test]
    fn test_pq_mmap_search() {
        let dims = 64;
        let n = 200;
        let mut rng = StdRng::seed_from_u64(42);
        let mut vecs = Vec::with_capacity(n * dims);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            for _ in 0..dims {
                vecs.push(rng.gen::<f32>());
            }
        }
        let c = (n as f64).sqrt().ceil() as usize;
        let tmp = std::env::temp_dir().join(format!("pq_test_{}.bin", std::process::id()));
        let pqf = PqFlatClustered::build_with_mmap(
            &vecs,
            &ids,
            dims,
            DistanceMetric::L2,
            c,
            10,
            0,
            Some(tmp.to_str().unwrap()),
        );
        assert_eq!(pqf.total_vectors(), n);
        assert!(pqf.memory_mb() < 2.0);
        assert!(pqf.disk_mb() > 0.0);

        let mut found = 0;
        for i in 0..20 {
            let q = &vecs[i * dims..(i + 1) * dims];
            let r = pqf.search(q, 1);
            if r.first().map(|x| x.0) == Some(ids[i]) {
                found += 1;
            }
        }
        assert!(found as f32 / 20.0 > 0.5);
        let _ = fs::remove_file(tmp);
    }
}
