//! 服务端后端语义的模型差分门禁：用参考模型对拍 `IndexBackend` 的在线操作流。
//!
//! 事故背景：`IndexBackend::Flat` 的写入缓冲只做 `extend`，同一 id 重复 upsert
//! 会累积多行——`indexed_count` 虚高、top-k 出现重复 id、旧向量继续参与排序。
//! 内核 `retrieval_oracle` 只覆盖 `HotBufferIndex`（且其随机数据策略显式去重 id），
//! 服务端封装层当时没有任何模型对照。本门禁把「小 id 域强制碰撞 + 同 id 换向量
//! + 删除后重插」的随机操作流同时喂给后端与 `HashMap` 参考模型，逐项对拍：
//!   计数、top-k 唯一性、删除不可见、重插可见、精确后端的逐位排序、近似后端的
//!   召回下限。任何后端变体出现同样的「追加而非替换」回归都会在这里立即失败。
//!
//! Model-differential gate for the server-side backend semantics: drive
//! `IndexBackend` with random operation streams against a `HashMap` reference
//! model. Background: `IndexBackend::Flat` only extended its buffer, so
//! re-upserting an id accumulated rows (inflated `indexed_count`, duplicate
//! ids in top-k, stale vectors still ranked). The kernel oracle only covered
//! `HotBufferIndex` and its datasets keep ids unique by policy, leaving the
//! server wrapper without any model comparison. This gate forces id collisions
//! in a small domain, rewrites vectors for existing ids, deletes and reinserts,
//! and cross-checks counts, top-k uniqueness, delete visibility, reinsert
//! visibility, exact backends' ordering, and approximate backends' recall.
//!
//! 变体豁免：`Clustered` / `FlatClustered` / `PqFlat` 不进入本在线流——懒构建
//! 后端需要显式的「构图后写入/查询」生命周期，`PqFlat` 构图后丢弃原始向量，
//! 无法与暴力模型逐位对拍。它们由各自的单测与生命周期门禁覆盖；新增后端变体
//! 必须登记到 `cases()` 或在此写明豁免理由。
//! EN: Exemptions: `Clustered` / `FlatClustered` / `PqFlat` stay out of this
//! online stream — lazy backends need an explicit build-then-write/search
//! lifecycle, and built `PqFlat` discards raw vectors so it cannot be
//! cross-checked bit-for-bit against a brute-force model. They are covered by
//! their own unit and lifecycle gates; a new backend variant must be registered
//! in `cases()` or justify its exemption here.

use houyi_server::index_backend::IndexBackend;
use std::collections::BTreeMap;

const DIMS: usize = 8;
/// 小 id 域：随机 upsert 必然反复命中同 id，复现「同 id 换向量」的写入语义。
/// EN: Small id domain: random upserts necessarily hit existing ids, reproducing
/// the per-id rewrite semantics.
const ID_DOMAIN: u64 = 24;
const STEPS: usize = 240;
const K_MAX: usize = 8;

/// 确定性 LCG：门禁失败可用固定 seed 逐位复现。
/// EN: Deterministic LCG so a failing gate reproduces bit-for-bit from its seed.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 10_000) as f32 / 10_000.0
    }
}

/// 六条互不相同的基向量：`(id, 版本)` 决定基向量再叠加微小扰动，既制造近似
/// 重复（同基向量、不同 id）也避免查询恰好与某行浮点相等。
/// EN: Six distinct base vectors; `(id, version)` selects one plus tiny noise,
/// creating near-duplicates (same base, different ids) without exact collisions.
fn vector_for(id: u64, version: u64) -> Vec<f32> {
    let base = ((id as usize) * 5 + (version as usize) * 3) % 6;
    let mut v = vec![0.0f32; DIMS];
    v[base % DIMS] = 1.0;
    v[(base + 1) % DIMS] = 1.0 + base as f32 * 0.1;
    v[(base + 3) % DIMS] = 0.5;
    let mut rng = Lcg::new(id.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(version));
    for x in &mut v {
        *x += (rng.unit() - 0.5) * 0.02;
    }
    v
}

/// 参考模型：id → 当前向量（后写覆盖先写，删除即移除）。
/// EN: Reference model: id → current vector (later writes win, delete removes).
type Model = BTreeMap<u64, Vec<f32>>;

fn brute_topk(model: &Model, query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = model
        .iter()
        .map(|(id, v)| (*id, houyi_vector::distance::cosine(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// 把写入推进到可检索状态：HNSW 先排空异步构图队列，HII L0 重建小图；
/// Flat 两者都是 no-op。
/// EN: Make writes searchable: HNSW drains its async build queue, HII L0
/// rebuilds its small graph; both are no-ops for Flat.
async fn sync(backend: &IndexBackend) {
    // 只有 HNSW 有异步构图队列；HII L0 的 `pending_build_count` 语义是
    // 「memtable 行数」而不是待构图数，按它循环会永不退出——必须按变体分流。
    // EN: Only HNSW has an async build queue. HII L0's `pending_build_count`
    // reports memtable rows, not pending builds, so looping on it never exits;
    // dispatch by variant instead.
    if let IndexBackend::Hnsw(_) = backend {
        while backend.pending_build_count().await > 0 {
            backend.run_async_builder_once().await;
        }
    }
    backend.optimize_build().await;
}

struct Case {
    name: &'static str,
    make: fn() -> IndexBackend,
    supports_delete: bool,
    /// 精确后端必须与暴力参考逐位一致；近似后端只要求召回下限。
    /// EN: Exact backends must match brute force bit-for-bit; approximate
    /// backends only need a recall floor.
    exact: bool,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "hnsw",
            make: || IndexBackend::new_hnsw(DIMS),
            supports_delete: true,
            exact: false,
        },
        Case {
            name: "flat",
            make: || IndexBackend::new_flat(DIMS),
            supports_delete: true,
            exact: true,
        },
        Case {
            name: "hii-l0",
            make: || IndexBackend::new_hii_l0(DIMS),
            supports_delete: false,
            exact: false,
        },
    ]
}

/// 近似后端的召回下限：小图 + 小语料下理应接近 1.0；阈值只拦结构性断裂
/// （不可达孤岛、删除未生效、排序通路漂移），不拦单次采样抖动。
/// EN: Recall floor for approximate backends: tiny graphs should be near 1.0;
/// the bound only catches structural breakage (unreachable islands, ignored
/// deletes, drifted ranking paths), not sampling jitter.
const APPROX_RECALL_MIN: f32 = 0.8;

#[tokio::test]
async fn backend_model_differential_holds_under_colliding_ids() {
    // 显式钉死补救开关：A/B 环境里残留 `off` 会关掉孤点补救，让本门禁改测
    // 另一套（有意关闭的）行为；与 50k 门禁的 env 纪律一致。
    //
    // 该钉死依赖两个前提，改动时须保持：(1) `group_repair_params` 是进程级
    // OnceLock，只读一次，所以本文件必须是**单测试二进制**且在任何构图之前
    // 设置；(2) 新增测试不得在本测试之前构造 HNSW/调用修复路径，否则开关已
    // 被缓存，本行静默失效。把新测试加进本文件时先复核这两点。
    // EN: Pin the repair knob explicitly: a leftover `off` in an A/B shell
    // would disable orphan repair and silently make this gate measure the
    // intentionally-disabled behavior, matching the 50k gates' env discipline.
    //
    // Two preconditions must hold for this pin: (1) `group_repair_params` is a
    // process-wide OnceLock read once, so this file must stay a single-test
    // binary and the pin must precede any build; (2) a new test must not build
    // an HNSW index or touch repair before this test, or the knob is already
    // cached and the pin silently no-ops. Re-check both when adding tests here.
    std::env::set_var("HOUYI_HNSW_GROUP_REPAIR", "4,16");
    for case in cases() {
        let backend = (case.make)();
        let mut model = Model::new();
        let mut versions: BTreeMap<u64, u64> = BTreeMap::new();
        let mut rng = Lcg::new(0xC0FF_EE00_1234_5678);

        for step in 0..STEPS {
            match rng.next() % 10 {
                0..=4 => {
                    // 批内/跨批重复 id：后写覆盖先写，模型同步替换。
                    // EN: Duplicate ids within and across batches: later writes
                    // win and the model replaces accordingly.
                    let n = 1 + (rng.next() % 4) as usize;
                    let mut batch = Vec::with_capacity(n);
                    for _ in 0..n {
                        let id = rng.next() % ID_DOMAIN;
                        let ver = versions.entry(id).or_insert(0);
                        *ver += 1;
                        let v = vector_for(id, *ver);
                        model.insert(id, v.clone());
                        batch.push((id, v));
                    }
                    backend.insert_batch(batch).await.unwrap();
                    sync(&backend).await;
                }
                5..=6 if case.supports_delete => {
                    if let Some((id, _)) = model.iter().next() {
                        let id = *id;
                        let removed = model.remove(&id).expect("picked id exists");
                        assert!(
                            backend.delete(id).await.unwrap(),
                            "{}: delete of a live id must report true",
                            case.name
                        );
                        // 用被删向量立即查询：墓碑/删除路径失效会把它顶到第一。
                        // EN: Query immediately with the removed vector; a broken
                        // tombstone/delete path would rank it first.
                        let k = model.len() + 4;
                        let got = backend.search(&removed, k, None).await;
                        assert!(
                            !got.contains(&id),
                            "{}: deleted id {id} still visible",
                            case.name
                        );
                        // 重插：必须复活并按新向量排序。走 `insert_batch`
                        // （服务端 upsert 的真实入口；单条 `insert` 只进 HNSW
                        // 热缓冲、不建图）。
                        // EN: Reinsert must resurrect it and rank by the new
                        // vector. Use `insert_batch` (the server's real upsert
                        // entry; a single `insert` only fills the HNSW hot
                        // buffer without a graph build).
                        let ver = versions.entry(id).or_insert(0);
                        *ver += 1;
                        let v = vector_for(id, *ver);
                        backend.insert_batch(vec![(id, v.clone())]).await.unwrap();
                        sync(&backend).await;
                        model.insert(id, v.clone());
                        assert_eq!(
                            backend.search(&v, 1, None).await,
                            vec![id],
                            "{}: reinserted id not visible",
                            case.name
                        );
                        assert_eq!(
                            backend.indexed_count().await,
                            model.len(),
                            "{}: count drift after delete+reinsert",
                            case.name
                        );
                    }
                }
                7..=9 => {
                    if model.is_empty() {
                        continue;
                    }
                    let id = rng.next() % ID_DOMAIN;
                    // 用该 id 已存的版本（而不是 +1 后丢弃）：查询方向应贴近
                    // 一条真实存活行，才能覆盖陈旧向量/排序语义。
                    // EN: Use the id's stored version (not an incremented value
                    // that is discarded): the query direction should sit near a
                    // real live row so stale-vector/ordering semantics are
                    // actually exercised.
                    let ver = versions.get(&id).copied().unwrap_or(0);
                    let query = vector_for(id, ver);
                    let k = (1 + (rng.next() % K_MAX as u64)) as usize;
                    let scored = backend
                        .search_scored_with_ef_override(&query, k, None, None)
                        .await;
                    let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();

                    // 计数与唯一性：flat 追加式写入会在这里立即暴露。
                    // EN: Count and uniqueness: append-style writes fail here at once.
                    assert_eq!(
                        backend.indexed_count().await,
                        model.len(),
                        "{}: indexed_count drift at step {step}",
                        case.name
                    );
                    let unique: std::collections::BTreeSet<u64> = ids.iter().copied().collect();
                    assert_eq!(
                        unique.len(),
                        ids.len(),
                        "{}: duplicate id in top-k at step {step}: {ids:?}",
                        case.name
                    );
                    for rid in &ids {
                        assert!(
                            model.contains_key(rid),
                            "{}: returned id {rid} absent from model at step {step}",
                            case.name
                        );
                    }
                    // 距离必须升序（越小越近）；排序通路漂移会破坏该合同。
                    // EN: Distances must ascend (smaller = closer); a drifted
                    // ranking path breaks this contract.
                    for pair in scored.windows(2) {
                        assert!(
                            pair[0].1 <= pair[1].1 + 1e-6,
                            "{}: non-ascending distances at step {step}: {:?}",
                            case.name,
                            scored
                        );
                    }

                    let truth = brute_topk(&model, &query, k);
                    if case.exact {
                        assert_eq!(
                            ids, truth,
                            "{}: exact backend diverged from brute force at step {step}",
                            case.name
                        );
                    } else {
                        let hit = ids.iter().filter(|id| truth.contains(id)).count();
                        let recall = hit as f32 / truth.len() as f32;
                        assert!(
                            recall >= APPROX_RECALL_MIN,
                            "{}: recall {recall:.2} < {APPROX_RECALL_MIN} at step {step} \
                             (got={ids:?}, truth={truth:?})",
                            case.name
                        );
                    }
                }
                _ => {}
            }

            assert_eq!(
                backend.indexed_count().await,
                model.len(),
                "{}: count must track the model after step {step}",
                case.name
            );
        }

        // 清空后检索必须为空（墓碑/残留行不得复活）。
        // EN: After deleting everything, search must be empty (no resurrection).
        if case.supports_delete {
            for id in model.keys().copied().collect::<Vec<_>>() {
                assert!(backend.delete(id).await.unwrap(), "{}: delete", case.name);
            }
            model.clear();
            sync(&backend).await;
            assert_eq!(backend.indexed_count().await, 0, "{}: clear", case.name);
            assert!(
                backend.search(&vector_for(1, 1), 4, None).await.is_empty(),
                "{}: search after full delete must be empty",
                case.name
            );
        }
    }
}
