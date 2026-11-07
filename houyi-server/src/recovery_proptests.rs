//! P4.5.0 恢复状态机：随机 op 序列 + 随机崩溃点 → 冷重放 vs 参考模型全比对。
//! Recovery state machine (P4.5.0): random op sequences with random crash
//! points, cold-replayed and compared field-by-field against a reference model.
//!
//! 覆盖的不变量 / Invariants:
//! - 冷重放是写入侧的严格对偶：任意 upsert/delete/set_ha_level 序列重放后，
//!   collection 集 / dims / ha_level / 向量集（id→values 精确相等）/ 标量行数
//!   / payload 回读（last-wins + 墓碑）与参考模型完全一致。
//! - 墓碑语义：删除后向量与 payload 双不可见；带标量重 upsert 复活；裸向量
//!   重 upsert 不清标量墓碑。
//! - 撕裂尾：崩溃点写入的半条记录被 open 截断，其后追加的记录重放可达。
//! - 历史截断韧性：指向不存在 collection 的 delete/set_ha_level 告警跳过。
//! - checkpoint 分界（独立 property）：图快照 + 尾段重放 ≡ 全量重放——
//!   `lsn ≤ checkpoint` 的记录跳过后端向量写但标量行/墓碑全量重建。

use crate::collection::CollectionManager;
use crate::collection_lifecycle::{load_collection_snapshots, write_collection_snapshots};
use crate::recovery::{
    replay_collections_wal_into_manager, replay_collections_wal_into_manager_with_mode,
    GraphSnapshotMetrics,
};
use crate::scalar_json::build_collection_scalar_row;
use crate::types::{WalHandle, WalWriteMode};
use houyi_core::ScalarValue;
use houyi_persistence::collection_wal::{
    encode_collection_create, encode_collection_delete, encode_collection_set_ha_level,
    encode_collection_upsert_batch, CollectionWalItem,
};
use houyi_persistence::ha_level::HaLevel;
use houyi_persistence::FileWal;
use proptest::prelude::*;
use proptest_state_machine::{prop_state_machine, ReferenceStateMachine, StateMachineTest};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// 状态机维度：4 维足够触发编解码/重放全路径，又保持 HNSW 构图廉价。
/// EN: State-machine dimensions: 4 exercises full codec/replay paths while
/// EN: keeping HNSW builds cheap.
const DIMS: usize = 4;
/// collection 名字池：小池促碰撞（同 collection 多 op 交错才有状态演化）。
/// EN: Collection name pool: a small pool forces collisions (interleaved ops on one collection
/// EN: evolve state).
const NAMES: [&str; 3] = ["alpha", "beta", "gamma"];
/// id 池：小池促删除/重插入/墓碑碰撞。
/// EN: Id pool: a small pool forces delete/reinsert/tombstone collisions.
const ID_SPACE: u64 = 64;

// ── 生成器域 ────────────────────────────────────────────────────
// EN: Generator domain.

/// 标量 payload 值（扁平 JSON，无 null——null 与 PAX 缺失列占位不可区分，
/// 属记录的已知取舍，不进属性域）。
/// EN: Scalar payload values (flat JSON, no null — null is indistinguishable
/// EN: from the PAX missing-column placeholder; a documented trade-off, excluded
/// EN: from the domain).
#[derive(Clone, Debug)]
enum ScalarJson {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

impl ScalarJson {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Int(v) => serde_json::Value::from(*v),
            Self::Float(v) => serde_json::Value::from(*v),
            Self::Bool(v) => serde_json::Value::from(*v),
            Self::Text(v) => serde_json::Value::from(v.clone()),
        }
    }
}

fn arb_scalar_json() -> impl Strategy<Value = ScalarJson> {
    prop_oneof![
        (-1_000_000i64..1_000_000).prop_map(ScalarJson::Int),
        // 粗粒度值域：f64→JSON→f64 精确往返，浮点等价性不属本状态机职责。
        // EN: Coarse value domain: exact f64->JSON->f64 round-trip; float equivalence
        // EN: is not this state machine's job.
        (-1024i32..1024).prop_map(|n| ScalarJson::Float(n as f64 * 0.5)),
        any::<bool>().prop_map(ScalarJson::Bool),
        "[a-z]{1,8}".prop_map(ScalarJson::Text),
    ]
}

fn arb_payload() -> impl Strategy<Value = Vec<(String, ScalarJson)>> {
    prop::collection::vec(("[a-z]{1,8}", arb_scalar_json()), 0..3).prop_map(|pairs| {
        // label 列由 label 字段渲染；payload 里的同名列会触发覆盖语义，避让。
        // EN: The label column is rendered from the label field; a same-named payload
        // EN: key triggers overwrite semantics, so avoid it.
        let mut seen = BTreeSet::new();
        pairs
            .into_iter()
            .filter(|(k, _)| k != "label" && seen.insert(k.clone()))
            .collect::<Vec<_>>()
    })
}

/// 向量分量：有限且量级有界。上界不是随意取——图路径 `normalize_for_cosine`
/// 用 **f32** 累加 norm²，|x| ≳ 1e19 时 norm² 上溢为 inf、`inv = inf.sqrt().recip()`
/// 归零，图内向量被静默归零（hnsw_graph.rs 记录的「图路径 f32 累加」已知延后项
/// 的最坏形态，已作为独立发现上报）。本属性聚焦恢复语义，值域取远离该边界的
/// 一般范围，把极端值行为留给该发现项单独裁决。
///
/// 直接从有界区间生成（而非 `any::<f32>()` + 过滤器）：全值域里 NaN/±inf 与
/// |x| > 1e6 占大头，过滤式采样浪费生成熵、拖慢收缩，且与区间生成覆盖同一集合
/// （OCR 批次修订）。
/// EN: Bounded magnitudes on purpose — the graph path accumulates norm² in f32,
/// so |x| ≳ 1e19 overflows to inf and zeroes the in-graph vector (worst case of
/// the known f32-accumulation deferral). Generated straight from the bounded
/// range rather than filtered from the full f32 domain: same set, less entropy
/// wasted, better shrinking.
fn arb_values() -> impl Strategy<Value = [f32; DIMS]> {
    prop::array::uniform4(-1.0e6f32..1.0e6)
}

fn payload_to_json(
    payload: &[(String, ScalarJson)],
) -> Option<serde_json::Map<String, serde_json::Value>> {
    if payload.is_empty() {
        return None;
    }
    Some(
        payload
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json()))
            .collect(),
    )
}

// ── 参考模型 ────────────────────────────────────────────────────
// EN: Reference model.

/// 一次 upsert 携带的标量行原始值快照（label + payload 键值对）：payload 回读
/// 期望值由它与写入侧同一构造函数推出。
/// EN: Raw scalar-row snapshot carried by one upsert (label + payload key/value
/// EN: pairs): expected payload read-back is derived via the same write-side constructor.
type ScalarSnapshot = Option<(Option<String>, Vec<(String, ScalarJson)>)>;

#[derive(Clone, Debug)]
enum Transition {
    Upsert {
        name_ix: usize,
        id: u64,
        values: [f32; DIMS],
        label: Option<String>,
        payload: Vec<(String, ScalarJson)>,
    },
    Delete {
        name_ix: usize,
        id: u64,
    },
    SetHaLevel {
        name_ix: usize,
        level: HaLevel,
    },
    /// 崩溃并冷重放：`torn` 携带写入文件尾的半条记录残骸（模拟撕裂写）。
    /// EN: Crash and cold replay: `torn` carries the half-record remnant written at
    /// EN: EOF (simulated torn write).
    CrashAndReplay {
        torn: Option<Vec<u8>>,
    },
}

#[derive(Clone, Debug, Default)]
struct RefState {
    /// BTreeMap：确定序遍历，比对输出稳定。
    /// EN: BTreeMap: deterministic iteration order keeps comparison output stable.
    collections: BTreeMap<String, RefCollection>,
}

#[derive(Clone, Debug)]
struct RefCollection {
    ha_level: HaLevel,
    /// append-only 标量行总数（含已被墓碑/覆盖的陈旧行——重放侧逐行重建，
    /// 行数本身就是不变量）。
    /// EN: Total append-only scalar rows (including stale rows from tombstones/
    /// EN: overwrites — replay rebuilds row by row, so the row count itself is the invariant).
    scalar_row_count: usize,
    rows: BTreeMap<u64, RefRow>,
}

#[derive(Clone, Debug, Default)]
struct RefRow {
    values: [f32; DIMS],
    /// 向量侧存在性（upsert 置真 / delete 置假，last-wins）。
    /// EN: Vector-side presence (upsert sets true / delete sets false, last-wins).
    vec_present: bool,
    /// 标量墓碑：delete 置位；**携带非标量行**的 upsert 清除（复活）；
    /// 裸向量 upsert 不触碰（与「裸向量不建行」同约定）。
    /// EN: Scalar tombstone: delete sets it; an upsert carrying a non-scalar row
    /// EN: clears it (resurrection); a bare-vector upsert leaves it untouched
    /// EN: (same convention as "bare vectors build no row").
    scalar_tombstoned: bool,
    /// 最后一次非空标量行（label + payload 原始值），payload 回读的期望值。
    /// EN: Latest non-empty scalar row (label + raw payload), the expected payload read-back.
    last_scalar: ScalarSnapshot,
}

impl RefRow {
    fn new(values: [f32; DIMS]) -> Self {
        Self {
            values,
            ..Self::default()
        }
    }
}

struct RecoveryRef;

impl ReferenceStateMachine for RecoveryRef {
    type State = RefState;
    type Transition = Transition;

    fn init_state() -> BoxedStrategy<Self::State> {
        Just(RefState::default()).boxed()
    }

    fn transitions(_state: &Self::State) -> BoxedStrategy<Self::Transition> {
        prop_oneof![
            8 => (
                0..NAMES.len(),
                0..ID_SPACE,
                arb_values(),
                prop::option::of("[a-z]{1,8}"),
                arb_payload(),
            )
                .prop_map(|(name_ix, id, values, label, payload)| Transition::Upsert {
                    name_ix,
                    id,
                    values,
                    label,
                    payload,
                }),
            3 => (0..NAMES.len(), 0..ID_SPACE)
                .prop_map(|(name_ix, id)| Transition::Delete { name_ix, id }),
            1 => (0..NAMES.len(), arb_ha_level())
                .prop_map(|(name_ix, level)| Transition::SetHaLevel { name_ix, level }),
            // ~13% 崩溃率：每次崩溃都是一次全量冷重放 + 模型比对。
            // EN: ~13% crash rate: every crash means a full cold replay plus model comparison.
            2 => prop::option::of(prop::collection::vec(any::<u8>(), 1..24))
                .prop_map(|torn| Transition::CrashAndReplay { torn }),
        ]
        .boxed()
    }

    fn apply(mut state: Self::State, transition: &Self::Transition) -> Self::State {
        match transition {
            Transition::Upsert {
                name_ix,
                id,
                values,
                label,
                payload,
            } => {
                let name = NAMES[*name_ix].to_string();
                let col = state.collections.entry(name).or_insert(RefCollection {
                    ha_level: HaLevel::Async,
                    scalar_row_count: 0,
                    rows: BTreeMap::new(),
                });
                let row = col.rows.entry(*id).or_insert_with(|| RefRow::new(*values));
                row.values = *values;
                row.vec_present = true;
                // 与写入侧同一函数判定「是否建行」：裸向量不建行、不清墓碑。
                // EN: Same write-side function decides row building: bare vectors build no
                // EN: row and clear no tombstone.
                let built = build_collection_scalar_row(
                    label.as_deref(),
                    payload_to_json(payload).as_ref(),
                );
                if !built.is_empty() {
                    col.scalar_row_count += 1;
                    row.scalar_tombstoned = false;
                    row.last_scalar = Some((label.clone(), payload.clone()));
                }
            }
            Transition::Delete { name_ix, id } => {
                // 目标 collection 缺失 = 历史截断场景：重放告警跳过，模型同步 no-op。
                // EN: Missing target collection = truncated history: replay warns and skips;
                // EN: model sync is a no-op.
                if let Some(col) = state.collections.get_mut(NAMES[*name_ix]) {
                    let row = col
                        .rows
                        .entry(*id)
                        .or_insert_with(|| RefRow::new([0.0; DIMS]));
                    row.vec_present = false;
                    row.scalar_tombstoned = true;
                }
            }
            Transition::SetHaLevel { name_ix, level } => {
                if let Some(col) = state.collections.get_mut(NAMES[*name_ix]) {
                    col.ha_level = *level;
                }
            }
            Transition::CrashAndReplay { .. } => {
                // 崩溃不改参考模型——重放必须还原崩溃前已确认的全部状态。
                // EN: A crash does not change the reference model — replay must restore all
                // EN: states acked before it.
            }
        }
        state
    }
}

// ── SUT ─────────────────────────────────────────────────────────

struct RecoverySut {
    dir: PathBuf,
    wal_path: PathBuf,
    wal: Option<FileWal>,
    /// 已落 create 记录的 collection（对偶写入侧「仅新建时落 create」）。
    /// EN: Collections with a create record (mirrors write side: create only on first insert).
    created: BTreeSet<String>,
    rt: tokio::runtime::Runtime,
}

impl RecoverySut {
    fn append(&mut self, payload: Vec<u8>) {
        let wal = self.wal.as_mut().expect("wal handle");
        wal.append_no_flush(payload).expect("wal append");
        wal.flush_wal().expect("wal flush");
    }
}

/// 临时目录的 RAII 清理：属性断言失败会提前 `return`、状态机 `apply` 内 panic 会
/// 跳过 `teardown`，只在成功路径 `remove_dir_all` 会把目录漏在 /tmp（OCR 批次修订）。
/// EN: RAII cleanup for temp dirs — a failing assert returns early and a panic
/// inside `apply` skips `teardown`, so success-path-only removal leaks the dir.
struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn unique_temp_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::AtomicU64;
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "{tag}-{}-{nanos}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

struct RecoveryStateMachineTest;

impl StateMachineTest for RecoveryStateMachineTest {
    type SystemUnderTest = RecoverySut;
    type Reference = RecoveryRef;

    fn init_test(_ref_state: &RefState) -> Self::SystemUnderTest {
        let dir = unique_temp_dir("houyidb-recovery-sm");
        let wal_path = dir.join("houyidb.collections.wal");
        let (wal, entries) = FileWal::open(&wal_path).expect("open fresh collections wal");
        assert!(entries.is_empty(), "fresh wal must be empty");
        RecoverySut {
            dir,
            wal_path,
            wal: Some(wal),
            created: BTreeSet::new(),
            rt: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("tokio runtime"),
        }
    }

    fn apply(
        mut state: Self::SystemUnderTest,
        ref_state: &RefState,
        transition: Transition,
    ) -> Self::SystemUnderTest {
        match transition {
            Transition::Upsert {
                name_ix,
                id,
                values,
                label,
                payload,
            } => {
                let name = NAMES[name_ix];
                if state.created.insert(name.to_string()) {
                    state.append(
                        encode_collection_create(name, DIMS, "hnsw").expect("encode create"),
                    );
                }
                let item = CollectionWalItem {
                    id,
                    values: values.to_vec(),
                    label,
                    payload: payload_to_json(&payload),
                };
                state.append(
                    encode_collection_upsert_batch(name, DIMS, &[item]).expect("encode upsert"),
                );
            }
            Transition::Delete { name_ix, id } => {
                state.append(
                    encode_collection_delete(NAMES[name_ix], &[id]).expect("encode delete"),
                );
            }
            Transition::SetHaLevel { name_ix, level } => {
                state.append(
                    encode_collection_set_ha_level(NAMES[name_ix], level)
                        .expect("encode set_ha_level"),
                );
            }
            Transition::CrashAndReplay { torn } => {
                // 崩溃：句柄收尾落盘后整体丢弃（进程内存态消亡）。
                // EN: Crash: after the handle flushes, drop everything (in-process memory state
                // EN: disappears).
                {
                    let mut wal = state.wal.take().expect("wal handle");
                    wal.sync_data().expect("sync before crash");
                    drop(wal);
                }
                // 撕裂写：崩溃点半条记录残骸直接追加在文件尾（绕过 WAL 编码）。
                // EN: Torn write: append the half-record remnant at the crash point directly at
                // EN: EOF (bypassing WAL encoding).
                if let Some(bytes) = torn {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new()
                        .append(true)
                        .open(&state.wal_path)
                        .expect("open for torn append");
                    f.write_all(&bytes).expect("write torn bytes");
                    f.sync_data().expect("sync torn bytes");
                }
                // 冷启动：open 截断撕裂尾 → 重放进全新 manager → 与模型全比对。
                // EN: Cold start: open truncates the torn tail -> replay into a fresh manager ->
                // EN: compare fully with the model.
                let (wal, entries) = FileWal::open(&state.wal_path).expect("reopen after crash");
                let manager = CollectionManager::new(DIMS, "hnsw");
                state
                    .rt
                    .block_on(replay_collections_wal_into_manager(
                        &entries,
                        &manager,
                        &HashMap::new(),
                    ))
                    .expect("cold replay must succeed");
                assert_manager_matches_model(&manager, ref_state, &state.rt);
                state.wal = Some(wal);
            }
        }
        state
    }

    fn teardown(mut state: Self::SystemUnderTest, ref_state: RefState) {
        // 终局兜底比对：`apply` 只在 CrashAndReplay 点比对 SUT vs 模型，未抽到崩溃
        // 的序列全程零校验（2/14 崩溃权重下，短用例绝大多数无崩溃）。teardown 拿得到
        // 最终 RefState，补一次冷重放比对，让每个用例都有端到端校验。
        // EN: Terminal cold-replay comparison — a sequence that never draws a crash
        // transition would otherwise never verify SUT against the model.
        state.wal = None; // 每次 append 已 flush，直接放句柄
        let (wal, entries) = FileWal::open(&state.wal_path).expect("reopen wal for terminal check");
        let manager = CollectionManager::new(DIMS, "hnsw");
        state
            .rt
            .block_on(replay_collections_wal_into_manager(
                &entries,
                &manager,
                &HashMap::new(),
            ))
            .expect("terminal cold replay must succeed");
        assert_manager_matches_model(&manager, &ref_state, &state.rt);
        drop(wal);

        // 清理挂在 `RecoverySut::drop` 上：`apply` 内断言失败时 teardown 不会执行，
        // 只有 Drop 能在展开路径上兜住临时目录。
        // EN: Cleanup hangs off `RecoverySut::drop`: when an assert fails inside `apply`,
        // EN: teardown is skipped; only Drop can cover the temp dir on the unwind path.
        drop(state);
    }
}

impl Drop for RecoverySut {
    fn drop(&mut self) {
        // 先放句柄再删目录（与 `FileWal` 的落盘约定无关，纯为删除顺序确定）。
        // EN: Drop the handle before deleting the dir (unrelated to `FileWal` flush rules;
        // EN: only for deterministic delete order).
        self.wal = None;
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

prop_state_machine! {
    #![proptest_config(ProptestConfig {
        // 每个用例含多次全量重放 + HNSW 构图，预算高于纯编解码属性；
        // nightly 经 PROPTEST_CASES 放大。
        // EN: Each case runs multiple full replays + HNSW builds, so the budget is above
        // EN: pure codec properties; nightly scales it via PROPTEST_CASES.
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn recovery_state_machine(sequential 1..48 => RecoveryStateMachineTest);
}

// ── 模型比对 ────────────────────────────────────────────────────
// EN: Model comparison.

/// 重放后的 manager 与参考模型逐字段比对（崩溃恢复的核心不变量）。
/// Compare a replayed manager field-by-field against the reference model —
/// the core crash-recovery invariant.
fn assert_manager_matches_model(
    manager: &CollectionManager,
    model: &RefState,
    rt: &tokio::runtime::Runtime,
) {
    let mut names = manager.list();
    names.sort();
    let expected: Vec<&String> = model.collections.keys().collect();
    assert_eq!(names.len(), expected.len(), "collection set diverged");
    for (got, want) in names.iter().zip(expected.iter()) {
        assert_eq!(got, *want, "collection set diverged");
    }

    for (name, refcol) in &model.collections {
        let col = manager
            .get(name)
            .expect("collection must exist after replay");
        assert_eq!(col.dims, DIMS, "dims diverged for '{name}'");
        assert_eq!(
            col.ha_level(),
            refcol.ha_level,
            "ha_level diverged for '{name}'"
        );

        // 向量侧：导出集 = 模型中 vec_present 的行（id → values 精确相等）。
        // EN: Vector side: exported set = rows with vec_present in the model (id -> values
        // EN: exactly equal).
        let exported: BTreeMap<u64, Vec<f32>> = rt
            .block_on(col.backend.export_snapshot_vectors())
            .expect("export vectors")
            .into_iter()
            .collect();
        let expected_vectors: BTreeMap<u64, Vec<f32>> = refcol
            .rows
            .iter()
            .filter(|(_, r)| r.vec_present)
            .map(|(id, r)| (*id, r.values.to_vec()))
            .collect();
        assert_eq!(
            exported.len(),
            expected_vectors.len(),
            "vector id set diverged for '{name}': exported {:?} vs expected {:?}",
            exported.keys().collect::<Vec<_>>(),
            expected_vectors.keys().collect::<Vec<_>>()
        );
        for (id, want) in &expected_vectors {
            assert_eq!(
                exported.get(id),
                Some(want),
                "vector values diverged for '{name}' id={id}"
            );
        }

        let scalars = col.scalars.lock().expect("scalars lock");
        assert_eq!(
            scalars.row_count(),
            refcol.scalar_row_count,
            "scalar row count diverged for '{name}'"
        );
        // payload 回读：last-wins + 墓碑缺席。期望值经写入侧同一构造函数派生，
        // 比对即「重放 == 写入」的端到端语义等价。
        // EN: Payload read-back: last-wins + tombstone absence. Expected values are derived
        // EN: via the same write-side constructor, so comparison is end-to-end replay == write.
        let all_ids: Vec<u64> = refcol.rows.keys().copied().collect();
        let payloads = scalars.payloads_for(&all_ids);
        let mut expected_payloads: HashMap<u64, HashMap<String, ScalarValue>> = HashMap::new();
        for (id, row) in &refcol.rows {
            if row.scalar_tombstoned {
                continue;
            }
            if let Some((label, payload)) = &row.last_scalar {
                let built = build_collection_scalar_row(
                    label.as_deref(),
                    payload_to_json(payload).as_ref(),
                );
                if !built.is_empty() {
                    expected_payloads.insert(*id, built);
                }
            }
        }
        assert_eq!(
            payloads.len(),
            expected_payloads.len(),
            "payload id set diverged for '{name}': got {:?} vs expected {:?}",
            payloads.keys().collect::<Vec<_>>(),
            expected_payloads.keys().collect::<Vec<_>>()
        );
        for (id, want) in &expected_payloads {
            assert_eq!(
                payloads.get(id),
                Some(want),
                "payload diverged for '{name}' id={id}"
            );
        }
    }
}

// ── 构建器自身的独立期望 ────────────────────────────────────────
// EN: Builder's own independent expectations.

/// 状态机的期望 payload 由**同一个** `build_collection_scalar_row` 派生，故对
/// 构建器内部的键合并 bug 是盲的（两边一起错 → 属性不可见）。本用例改用手写期望
/// 钉住碰撞语义：payload 键先写入、label 字段后写入并覆盖（last-wins），无 label
/// 字段时 payload 的 `label` 键原样保留（不注入空值）。
/// EN: The state machine derives expected payloads through the same builder, so a
/// builder-internal key-merge bug is invisible to it. Hand-written expectations
/// pin the collision semantics instead.
#[test]
fn scalar_row_builder_label_key_collision_is_last_wins() {
    let payload = serde_json::json!({"label": "from_payload", "n": 7, "ok": true});
    let map = payload.as_object().expect("json object");

    let row = build_collection_scalar_row(Some("from_field"), Some(map));
    assert_eq!(row.len(), 3, "label 键只能出现一次（被 label 字段覆盖）");
    assert_eq!(
        row.get("label"),
        Some(&ScalarValue::Text("from_field".to_string()))
    );
    assert_eq!(row.get("n"), Some(&ScalarValue::Int64(7)));
    assert_eq!(row.get("ok"), Some(&ScalarValue::Bool(true)));

    let row = build_collection_scalar_row(None, Some(map));
    assert_eq!(
        row.get("label"),
        Some(&ScalarValue::Text("from_payload".to_string())),
        "无 label 字段时 payload 的 label 键必须原样保留"
    );
}

// ── checkpoint 分界独立 property ────────────────────────────────
// EN: Standalone property for the checkpoint boundary.

/// 单 collection 的 op（checkpoint 属性专用：名字固定，聚焦 LSN 分界语义）。
/// EN: Ops for a single collection (checkpoint property only: fixed name, focused on LSN boundary
/// EN: semantics).
#[derive(Clone, Debug)]
enum CollOp {
    Upsert {
        id: u64,
        values: [f32; DIMS],
        label: Option<String>,
        payload: Vec<(String, ScalarJson)>,
    },
    Delete {
        id: u64,
    },
    SetHaLevel {
        level: HaLevel,
    },
}

/// 枚举全值域生成器（0..=2 与 HaLevel 一一对应）。
/// Generator over the full enum domain.
fn arb_ha_level() -> impl Strategy<Value = HaLevel> {
    prop::sample::select(vec![HaLevel::Async, HaLevel::SemiSync, HaLevel::FullSync])
}

fn arb_coll_op() -> impl Strategy<Value = CollOp> {
    prop_oneof![
        6 => (
            0..ID_SPACE,
            arb_values(),
            prop::option::of("[a-z]{1,8}"),
            arb_payload(),
        )
            .prop_map(|(id, values, label, payload)| CollOp::Upsert {
                id,
                values,
                label,
                payload,
            }),
        2 => (0..ID_SPACE).prop_map(|id| CollOp::Delete { id }),
        1 => arb_ha_level().prop_map(|level| CollOp::SetHaLevel { level }),
    ]
}

fn encode_coll_op(op: &CollOp) -> Vec<u8> {
    match op {
        CollOp::Upsert {
            id,
            values,
            label,
            payload,
        } => {
            let item = CollectionWalItem {
                id: *id,
                values: values.to_vec(),
                label: label.clone(),
                payload: payload_to_json(payload),
            };
            encode_collection_upsert_batch("c", DIMS, &[item]).expect("encode upsert")
        }
        CollOp::Delete { id } => encode_collection_delete("c", &[*id]).expect("encode delete"),
        CollOp::SetHaLevel { level } => {
            encode_collection_set_ha_level("c", *level).expect("encode set_ha_level")
        }
    }
}

/// 向量集等价（快照路 vs 全量路的口径对齐，见下方 WHY）。
///
/// WHY 不逐字节相等：HNSW 图内只存 Cosine 归一化空间向量，图快照存的就是这份
/// 归一化向量；`restore_graph_snapshot` 把它灌进 `main_index`，而全量重放路径的
/// `main_index` 装的是客户端**原始**向量（热缓冲/待构图队列全程保原始，merge 时
/// 图内另存归一化副本）。故两条恢复路径的 `export_snapshot_vectors` 在 Cosine
/// 下方向等价、数值口径不同——这是 P4.4.0 快照格式的固有取舍（快照不保原始
/// 向量，只有归一化空间可恢复），非本期引入。本属性按**产品语义**断言：距离
/// 口径（方向）等价 + 零范数带向量原样保留（图路径对该带不做单位化）。
/// 已作为待裁决项上报：是否让快照额外保真原始向量（体积翻倍 + 格式 bump）。
/// EN: Not byte-equal by design — the graph snapshot only carries the
/// unit-normalized vectors (all HNSW stores), and restore feeds those into
/// `main_index`, while the full-replay path keeps the client's original vectors
/// there. Both paths are equivalent under the Cosine distance semantics this
/// backend promises; preserving original vectors across snapshots is a P4.4.0
/// format trade-off flagged for review, not introduced here.
fn assert_vectors_equivalent(a: &[(u64, Vec<f32>)], b: &[(u64, Vec<f32>)], ctx: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{ctx}: vector id count diverged: {:?} vs {:?}",
        a.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        b.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );
    for ((aid, av), (bid, bv)) in a.iter().zip(b.iter()) {
        assert_eq!(aid, bid, "{ctx}: vector id order diverged");
        // f32 累加，与图路径 `normalize_for_cosine`（hnsw_graph.rs:255）**同一表达式**：
        // 只有累加精度逐位一致，本分支才与图内实际决策同支。改用 f64 会在阈值
        // `f32::EPSILON` 的一个 ulp 内给出相反判定——图已单位化而断言走「原样保留」
        // 分支（或反之），把正确实现报成失败（OCR 批次修订）。
        // 上溢不构成风险：`arb_values` 已限 |x| ≤ 1e6 ⇒ norm² ≤ 4e12 ≪ f32::MAX。
        // EN: f32 accumulation mirroring the graph path exactly — an f64 sum drifts
        // within one ulp of the band edge and can pick the opposite branch from the
        // one the graph actually took. Overflow is out of scope: the generator
        // bounds |x| ≤ 1e6.
        let norm2: f32 = av.iter().map(|x| x * x).sum();
        if norm2 <= houyi_vector::ZERO_NORM_BAND_NORM2 {
            // 零范数带（阈值取 houyi-vector 的**同一个**常量，非按值复刻：阈值
            // 变更时此处随之联动，不会静默失配）：图路径原样保留，两路必须逐字节相等。
            // EN: Zero-norm band (threshold uses the **same** constant in houyi-vector, not
            // EN: a copied value, so it follows changes instead of silently drifting): the
            // EN: graph path keeps vectors as-is; both paths must be byte-identical.
            assert_eq!(
                av, bv,
                "{ctx}: zero-norm-band vector id={aid} must be verbatim"
            );
        } else {
            // `distance::cosine` 返回**距离**（1 - cos_sim），同向 ⇒ 距离 ≈ 0。
            // EN: `distance::cosine` returns **distance** (1 - cos_sim); same direction => distance
            // EN: ~ 0.
            let dist = houyi_vector::distance::cosine(av, bv);
            assert!(
                dist < 1e-5,
                "{ctx}: snapshot vector id={aid} must be the unit-normalized form of \
                 the original (cosine distance {dist}); full={av:?} snap={bv:?}"
            );
        }
    }
}

/// 两个 manager 逐字段等价（checkpoint 属性：快照路 vs 全量路互为 oracle）。
/// EN: Two managers are field-by-field equivalent (snapshot path vs full path are mutual oracles).
fn assert_managers_equivalent(
    a: &CollectionManager,
    b: &CollectionManager,
    rt: &tokio::runtime::Runtime,
) {
    let mut names_a = a.list();
    let mut names_b = b.list();
    names_a.sort();
    names_b.sort();
    assert_eq!(names_a, names_b, "collection sets diverged");
    for name in names_a {
        let ca = a.get(&name).expect("collection in a");
        let cb = b.get(&name).expect("collection in b");
        assert_eq!(ca.dims, cb.dims, "dims diverged for '{name}'");
        assert_eq!(
            ca.ha_level(),
            cb.ha_level(),
            "ha_level diverged for '{name}'"
        );
        let va = rt
            .block_on(ca.backend.export_snapshot_vectors())
            .expect("export a");
        let vb = rt
            .block_on(cb.backend.export_snapshot_vectors())
            .expect("export b");
        assert_vectors_equivalent(&va, &vb, &name);

        let all_ids: Vec<u64> = va.iter().map(|(id, _)| *id).collect();
        let sa = ca.scalars.lock().expect("scalars a");
        let sb = cb.scalars.lock().expect("scalars b");
        assert_eq!(
            sa.row_count(),
            sb.row_count(),
            "scalar row count diverged for '{name}'"
        );
        assert_eq!(
            sa.payloads_for(&all_ids),
            sb.payloads_for(&all_ids),
            "payload read-back diverged for '{name}'"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    /// checkpoint 分界不变量：图快照 + `lsn > checkpoint` 尾段重放 ≡ 全量重放。
    /// 覆盖恢复路径最危险的 off-by-one 面：分界处记录「向量跳、标量建」。
    /// EN: Checkpoint-boundary invariant: graph snapshot + replay of the `lsn > checkpoint`
    /// EN: tail is exactly a full replay. Covers the riskiest off-by-one face: at the
    /// EN: boundary a record is "vector skipped, scalar built".
    #[test]
    fn graph_snapshot_plus_tail_replay_equals_full_replay(
        ops in prop::collection::vec(arb_coll_op(), 2..24),
        cp_seed in any::<proptest::sample::Index>(),
    ) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime");
        let dir = unique_temp_dir("houyidb-checkpoint-prop");
        // 守卫声明在 manager 之前 ⇒ 最后析构（manager 先释放，再删目录）。
        // EN: The guard is declared before the manager => dropped last (manager freed first, then
        // EN: dir removed).
        let _dir_guard = TempDirGuard(dir.clone());
        let wal_path = dir.join("houyidb.collections.wal");

        // 全部记录落 WAL（create + ops）。
        // EN: All records are written to the WAL (create + ops).
        let mut records = vec![encode_collection_create("c", DIMS, "hnsw").expect("encode create")];
        records.extend(ops.iter().map(encode_coll_op));
        {
            let (mut wal, _) = FileWal::open(&wal_path).expect("open wal");
            for r in &records {
                wal.append_no_flush(r.clone()).expect("append");
            }
            wal.sync_data().expect("sync");
        }
        let (_, entries) = FileWal::open(&wal_path).expect("reopen wal");
        assert_eq!(entries.len(), records.len());

        // 分界点：create 之后、最后一个 op 之前（两侧都至少有一条 op）。
        // EN: Split point: after create, before the last op (both sides have at least one op).
        let cp_ops = 1 + cp_seed.index(ops.len() - 1);
        // create 占 lsn 0，op i 占 lsn i（从 1 起）；覆盖 create + 前 cp_ops 个 op。
        // EN: create takes lsn 0, op i takes lsn i (starting at 1); covers create + first cp_ops
        // EN: ops.
        let checkpoint_lsn = cp_ops as u64;

        // 路径 A：全量重放（无快照）。
        // EN: Path A: full replay (no snapshot).
        let manager_full = CollectionManager::new(DIMS, "hnsw");
        rt.block_on(replay_collections_wal_into_manager(
            &entries,
            &manager_full,
            &HashMap::new(),
        ))
        .expect("full replay");

        // 路径 B：前缀重放 → 图快照 → 新 manager 恢复快照 → 带 checkpoint 重放全量。
        // EN: Path B: prefix replay -> graph snapshot -> new manager restores snapshot ->
        // EN: replay the rest with checkpoint.
        let manager_snap_src = CollectionManager::new(DIMS, "hnsw");
        rt.block_on(replay_collections_wal_into_manager(
            &entries[..=checkpoint_lsn as usize],
            &manager_snap_src,
            &HashMap::new(),
        ))
        .expect("prefix replay");
        let snapshot_bytes = {
            let col = manager_snap_src.get("c").expect("collection c");
            rt.block_on(crate::recovery::drain_pending_builder(&col.backend));
            rt.block_on(col.backend.encode_graph_snapshot())
                .expect("hnsw backend must encode a graph snapshot")
        };
        let manager_snap = CollectionManager::new(DIMS, "hnsw");
        {
            let (col, _) = manager_snap
                .get_or_create_flagged("c", Some(DIMS))
                .expect("create c");
            rt.block_on(col.backend.restore_graph_snapshot(&snapshot_bytes))
                .expect("hnsw backend must accept a graph snapshot")
                .expect("snapshot restore must succeed");
        }
        let mut checkpoints = HashMap::new();
        checkpoints.insert("c".to_string(), checkpoint_lsn);
        rt.block_on(replay_collections_wal_into_manager(
            &entries,
            &manager_snap,
            &checkpoints,
        ))
        .expect("checkpoint replay");

        assert_managers_equivalent(&manager_full, &manager_snap, &rt);
    }

    /// P5.0.2：collection 快照（HNSW 图 + 标量物理视图 + 墓碑 + HA）加尾段重放，
    /// 必须与全量重放逐字段等价；与上面的 legacy 图快照属性互补。
    /// EN: P5.0.2 — collection snapshots (HNSW graph + physical scalars +
    /// tombstones + HA) plus tail replay must equal full replay field-by-field;
    /// complements the legacy graph-snapshot property above.
    #[test]
    fn collection_snapshot_plus_tail_replay_equals_full_replay(
        ops in prop::collection::vec(arb_coll_op(), 2..24),
        cp_seed in any::<proptest::sample::Index>(),
    ) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime");
        let dir = unique_temp_dir("houyidb-collection-snapshot-prop");
        let _dir_guard = TempDirGuard(dir.clone());
        let wal_path = dir.join("houyidb.collections.wal");

        let mut records = vec![encode_collection_create("c", DIMS, "hnsw").expect("encode create")];
        records.extend(ops.iter().map(encode_coll_op));
        {
            let (mut wal, _) = FileWal::open(&wal_path).expect("open wal");
            for r in &records {
                wal.append_no_flush(r.clone()).expect("append");
            }
            wal.sync_data().expect("sync");
        }
        let (_, entries) = FileWal::open(&wal_path).expect("reopen wal");
        assert_eq!(entries.len(), records.len());

        let cp_ops = 1 + cp_seed.index(ops.len() - 1);
        let checkpoint_lsn = cp_ops as u64;

        // 路径 A：全量重放。
        // EN: Path A — full replay.
        let manager_full = Arc::new(CollectionManager::new(DIMS, "hnsw"));
        rt.block_on(replay_collections_wal_into_manager(
            &entries,
            &manager_full,
            &HashMap::new(),
        ))
        .expect("full replay");

        // 路径 B：前缀重放 → collection 快照 + manifest → 新 manager 恢复 → 尾段重放。
        // EN: Path B — prefix replay -> collection snapshot + manifest -> fresh
        // manager restore -> tail replay.
        let manager_prefix = Arc::new(CollectionManager::new(DIMS, "hnsw"));
        rt.block_on(replay_collections_wal_into_manager(
            &entries[..=checkpoint_lsn as usize],
            &manager_prefix,
            &HashMap::new(),
        ))
        .expect("prefix replay");
        let (wal, _) = WalHandle::open(&wal_path, WalWriteMode::Sync).expect("open wal handle");
        let metrics = Arc::new(GraphSnapshotMetrics::default());
        let outcome = rt.block_on(write_collection_snapshots(
            &manager_prefix,
            &Arc::new(wal),
            &wal_path,
            None,
            &metrics,
            // 属性测试模拟 leader/单机语义（同步 apply），允许空 manifest。
            // EN: The proptest models leader/standalone semantics (synchronous
            // apply), so empty manifests are allowed.
            true,
        ));
        let manifest = outcome
            .manifest
            .expect("collection snapshots must commit for hnsw");
        assert_eq!(manifest.collections[0].checkpoint_lsn, checkpoint_lsn);

        let manager_snap = Arc::new(CollectionManager::new(DIMS, "hnsw"));
        let loaded = rt
            .block_on(load_collection_snapshots(
                &manager_snap,
                &wal_path,
                0,
                None,
                &metrics,
            ))
            .expect("load collection snapshots")
            .expect("manifest must load");
        rt.block_on(replay_collections_wal_into_manager_with_mode(
            &entries,
            &manager_snap,
            &loaded.checkpoints,
            true,
        ))
        .expect("manifest checkpoint replay");

        assert_managers_equivalent(&manager_full, &manager_snap, &rt);
    }
}
