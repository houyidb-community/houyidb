//! JSON 编码的 Collection WAL 记录（v1）。
//! Collection WAL records as JSON (v1): create / upsert_batch / delete / drop.
//!
//! 为什么独立成文件而不复用主向量 WAL：主 WAL 的 checkpoint/rotate 以「主索引快照
//! 已覆盖的 LSN」为界截断；Collection 数据（当前）没有快照，混入同一文件会被
//! checkpoint 静默清掉，持久化形同虚设。独立文件让两条生命周期互不相干。
//! EN: Why a separate file from the main vector WAL — the main WAL's
//! checkpoint/rotate truncates at the LSN covered by the main-index snapshot, and
//! collection data has no snapshot, so sharing one file would let a checkpoint
//! silently erase collection records. Separate files keep the lifecycles apart.
//!
//! 标量行的持久化策略：存「用户原始 JSON」（label + payload），不落 ScalarValue。
//! 写入侧的标量行本来就由 `json_value_to_scalar` 从这份 JSON 派生，重放走同一
//! 函数即逐字节语义等价（Int64/Float64/Bool/Text/JsonText 全覆盖；Timestamp 不会
//! 从 JSON 路径产生）。由此 persistence 层无需反向依赖 houyi-core。
//! EN: Scalar rows are persisted as the user's original JSON (label + payload), not
//! as ScalarValue. The in-memory row was derived from exactly this JSON via
//! `json_value_to_scalar`, so replaying through the same function is semantically
//! identical — and the persistence crate stays free of a houyi-core dependency.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 协议版本 / Protocol version for the `v` field.
/// EN: Protocol version carried in the `v` field.
pub const COLLECTION_WAL_VERSION: u32 = 1;

const OP_COLLECTION_CREATE: &str = "collection_create";
const OP_COLLECTION_UPSERT_BATCH: &str = "collection_upsert_batch";
const OP_COLLECTION_DELETE: &str = "collection_delete";
const OP_COLLECTION_SET_HA_LEVEL: &str = "collection_set_ha_level";
/// 整库删除 op：删除 collection 自身（区别于 `collection_delete` 的按 id 删向量）。
/// EN: Whole-collection delete op (distinct from `collection_delete`, which
/// removes individual vectors by id).
const OP_COLLECTION_DROP: &str = "collection_drop";

/// 集合创建的可选元数据（度量 + HNSW 参数）；缺省字段表示旧记录语义。
/// EN: Optional collection-creation metadata (metric + HNSW parameters); absent
/// fields mean legacy-record semantics.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CollectionCreateOptions {
    /// 规范度量串（l2|cosine|ip|l1）；None = 旧记录，按 index_type 历史语义推断。
    /// EN: Canonical metric string (l2|cosine|ip|l1); None = legacy record.
    pub metric: Option<String>,
    pub hnsw_m: Option<usize>,
    pub hnsw_ef_construction: Option<usize>,
    pub hnsw_ef_search: Option<usize>,
}

impl CollectionCreateOptions {
    /// 规范度量集合：写入侧只写这些串，解码侧对未知串 fail-fast（腐败/未来版本）。
    /// EN: Canonical metrics only — the writer emits these strings and the decoder
    /// fails fast on anything else (corruption / future version).
    pub const CANONICAL_METRICS: [&'static str; 4] = ["l2", "cosine", "ip", "l1"];
}

/// 由主 WAL 路径派生 collections WAL 路径：`/data/houyidb.wal` → `/data/houyidb.collections.wal`。
/// Derive the collections WAL path from the main WAL path.
pub fn collections_wal_path_for(wal_path: &Path) -> PathBuf {
    let stem = wal_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "houyidb".to_string());
    wal_path.with_file_name(format!("{stem}.collections.wal"))
}

/// 重放侧解码结果 / Decoded record for cold replay.
/// EN: Decoded record consumed by cold replay.
#[derive(Debug, Clone, PartialEq)]
pub enum CollectionWalRecord {
    /// 隐式/显式建 Collection（dims/index_type/metric 为创建时定型值）。
    /// `options` 缺省即旧记录：度量按 index_type 历史语义推断、HNSW 参数取库层默认。
    /// EN: Implicit/explicit collection creation (dims/index_type/metric fixed at
    /// creation). Default `options` means a legacy record: metric inferred from
    /// index_type and library-default HNSW parameters.
    Create {
        name: String,
        dims: usize,
        index_type: String,
        options: CollectionCreateOptions,
    },
    /// 一批 upsert（向量 + 可选 label/payload 原始 JSON）/ One upsert batch.
    /// EN: One upsert batch (vectors plus optional raw label/payload JSON).
    UpsertBatch {
        name: String,
        dim: usize,
        items: Vec<CollectionWalItem>,
    },
    /// 一批删除（过滤删除在写入侧已解析成具体 id）/ Resolved delete ids.
    /// EN: One delete batch (filtered deletes are resolved to concrete ids at write time).
    Delete { name: String, ids: Vec<u64> },
    /// 设置 collection 的 HA 级别（P4.4.2）/ Set the collection's HA level.
    /// 解码即得枚举：非法级别在解码侧 fail-fast，下游不再面对可解析失败的
    /// 字符串（P4.6.3 类型硬化）。EN: decoded straight to the enum so no
    /// downstream consumer faces an unvalidated string (P4.6.3).
    SetHaLevel {
        name: String,
        ha_level: crate::ha_level::HaLevel,
    },
    /// 删除整个 collection（无条件、不可逆）/ Drop the whole collection
    /// (unconditional and irreversible).
    /// EN: The runtime removes the collection from the manager; replay and the
    /// follower apply path do the same, which keeps a deleted collection from
    /// being resurrected by older WAL records.
    Drop { name: String },
}

impl CollectionWalRecord {
    /// 记录作用的 collection 名（重放按 collection checkpoint 跳整条记录时用）。
    /// EN: Target collection name (used to skip whole records by per-collection
    /// checkpoint during replay).
    pub fn collection_name(&self) -> &str {
        match self {
            Self::Create { name, .. }
            | Self::UpsertBatch { name, .. }
            | Self::Delete { name, .. }
            | Self::SetHaLevel { name, .. }
            | Self::Drop { name } => name,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollectionWalItem {
    pub id: u64,
    pub values: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Map<String, serde_json::Value>>,
}

/// 线上信封：单结构体承载全部 op，缺省字段不序列化（与 vector_wal 同风格）。
/// Wire envelope: one struct for all ops; absent fields are omitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionWalEnvelope {
    v: u32,
    op: String,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dims: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    index_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    items: Option<Vec<CollectionWalItem>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ids: Option<Vec<u64>>,
    // P4.4.2：不 bump v——v≠1 的旧记录会被整体静默跳过，新 op 对旧二进制
    // 天然兼容（未知 op 跳过），无需动版本号。
    // EN: P4.4.2: no version bump — records with v != 1 are skipped wholesale,
    // and new ops are naturally compatible with old binaries (unknown op skip).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ha_level: Option<String>,
    // 集合级度量与 HNSW 参数（显式创建写入，旧记录缺省）。
    // EN: Collection metric and HNSW parameters (written by explicit creation;
    // absent in legacy records).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metric: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hnsw_m: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hnsw_ef_construction: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hnsw_ef_search: Option<usize>,
}

/// 编码 collection 创建记录 / Encode a collection-create record.
/// EN: Encodes a collection-create record.
pub fn encode_collection_create(
    name: &str,
    dims: usize,
    index_type: &str,
) -> Result<Vec<u8>, String> {
    encode_collection_create_with_options(
        name,
        dims,
        index_type,
        &CollectionCreateOptions::default(),
    )
}

/// 编码带度量/HNSW 参数的创建记录（显式 `PUT /v1/collections/{name}` 使用）。
/// 度量串必须是 [`CollectionCreateOptions::CANONICAL_METRICS`] 之一，避免把
/// 别名或任意串写进 WAL 后让解码侧成为唯一校验点。
/// EN: Encode a create record carrying metric/HNSW parameters (explicit creation).
/// The metric must be one of the canonical strings so the WAL never carries an
/// alias or arbitrary text that only the decoder would vet.
pub fn encode_collection_create_with_options(
    name: &str,
    dims: usize,
    index_type: &str,
    options: &CollectionCreateOptions,
) -> Result<Vec<u8>, String> {
    if let Some(metric) = options.metric.as_deref() {
        if !CollectionCreateOptions::CANONICAL_METRICS.contains(&metric) {
            return Err(format!(
                "non-canonical collection metric {metric:?} (expect l2|cosine|ip|l1)"
            ));
        }
    }
    // HNSW 参数与后端/度量的耦合必须在写入侧闭合（解码侧同规则）：半套参数、
    // 给非 hnsw 后端、以及「有 HNSW 参数却缺度量」都会让重放静默补默认值
    //（度量退回 Cosine），与创建时实际语义分叉。m 下限 2 与库层一致。
    // EN: Couple HNSW parameters with the backend and metric on the write side
    // (the decoder mirrors this): a partial set, params for a non-hnsw backend,
    // or HNSW params without a metric would replay with silent defaults (metric
    // falls back to cosine), diverging from creation-time semantics. The m lower
    // bound of 2 matches the library.
    match (
        options.hnsw_m,
        options.hnsw_ef_construction,
        options.hnsw_ef_search,
    ) {
        (None, None, None) => {}
        (Some(m), Some(efc), Some(efs)) => {
            if index_type != "hnsw" {
                return Err(format!(
                    "hnsw parameters require index_type 'hnsw' (got {index_type:?})"
                ));
            }
            if options.metric.is_none() {
                return Err("hnsw parameters require an explicit metric".to_string());
            }
            if !(2..=1024).contains(&m)
                || !(1..=65_536).contains(&efc)
                || !(1..=65_536).contains(&efs)
            {
                return Err(format!(
                    "hnsw parameters out of range \
                     (m={m}, ef_construction={efc}, ef_search={efs})"
                ));
            }
        }
        _ => {
            return Err(
                "hnsw_m/hnsw_ef_construction/hnsw_ef_search must be provided together".to_string(),
            )
        }
    }
    let env = CollectionWalEnvelope {
        v: COLLECTION_WAL_VERSION,
        op: OP_COLLECTION_CREATE.to_string(),
        name: name.to_string(),
        dims: Some(dims),
        index_type: Some(index_type.to_string()),
        items: None,
        ids: None,
        ha_level: None,
        metric: options.metric.clone(),
        hnsw_m: options.hnsw_m,
        hnsw_ef_construction: options.hnsw_ef_construction,
        hnsw_ef_search: options.hnsw_ef_search,
    };
    serde_json::to_vec(&env).map_err(|e| e.to_string())
}

/// 编码一批 collection upsert / Encode a collection upsert batch.
/// EN: Encodes a collection upsert batch.
pub fn encode_collection_upsert_batch(
    name: &str,
    dim: usize,
    items: &[CollectionWalItem],
) -> Result<Vec<u8>, String> {
    let env = CollectionWalEnvelope {
        v: COLLECTION_WAL_VERSION,
        op: OP_COLLECTION_UPSERT_BATCH.to_string(),
        name: name.to_string(),
        dims: Some(dim),
        index_type: None,
        items: Some(items.to_vec()),
        ids: None,
        ha_level: None,
        metric: None,
        hnsw_m: None,
        hnsw_ef_construction: None,
        hnsw_ef_search: None,
    };
    serde_json::to_vec(&env).map_err(|e| e.to_string())
}

/// 编码 collection HA 级别变更（P4.4.2）/ Encode a collection HA-level change.
///
/// 收**枚举**而非字符串或裸码：规范字符串由本函数内部渲染——「落盘必为规范
/// 串」与「非法级别不可表达」由类型签名保证（release 下同样成立）。（OCR
/// 批次十八；P4.6.3 起码值由枚举携带）
/// EN: Takes the **enum** — the canonical string is rendered internally, so
/// "persisted value is canonical" and "invalid level is unrepresentable" both
/// hold by construction in all build profiles.
pub fn encode_collection_set_ha_level(
    name: &str,
    level: crate::ha_level::HaLevel,
) -> Result<Vec<u8>, String> {
    let env = CollectionWalEnvelope {
        v: COLLECTION_WAL_VERSION,
        op: OP_COLLECTION_SET_HA_LEVEL.to_string(),
        name: name.to_string(),
        dims: None,
        index_type: None,
        items: None,
        ids: None,
        ha_level: Some(level.as_str().to_string()),
        metric: None,
        hnsw_m: None,
        hnsw_ef_construction: None,
        hnsw_ef_search: None,
    };
    serde_json::to_vec(&env).map_err(|e| e.to_string())
}

/// 编码一批 collection 删除 / Encode a collection delete batch.
/// EN: Encodes a collection delete batch.
pub fn encode_collection_delete(name: &str, ids: &[u64]) -> Result<Vec<u8>, String> {
    let env = CollectionWalEnvelope {
        v: COLLECTION_WAL_VERSION,
        op: OP_COLLECTION_DELETE.to_string(),
        name: name.to_string(),
        dims: None,
        index_type: None,
        items: None,
        ids: Some(ids.to_vec()),
        ha_level: None,
        metric: None,
        hnsw_m: None,
        hnsw_ef_construction: None,
        hnsw_ef_search: None,
    };
    serde_json::to_vec(&env).map_err(|e| e.to_string())
}

/// 编码整库删除记录 / Encode a whole-collection drop record.
/// EN: Encodes a whole-collection drop record.
pub fn encode_collection_drop(name: &str) -> Result<Vec<u8>, String> {
    let env = CollectionWalEnvelope {
        v: COLLECTION_WAL_VERSION,
        op: OP_COLLECTION_DROP.to_string(),
        name: name.to_string(),
        dims: None,
        index_type: None,
        items: None,
        ids: None,
        ha_level: None,
        metric: None,
        hnsw_m: None,
        hnsw_ef_construction: None,
        hnsw_ef_search: None,
    };
    serde_json::to_vec(&env).map_err(|e| e.to_string())
}

/// 解码一条 WAL payload；非 JSON、旧版或未知 op 返回 `Ok(None)`（向前兼容跳过）。
/// Decode one WAL payload; non-JSON / legacy / unknown ops yield `Ok(None)`.
pub fn decode_collection_wal_payload(
    payload: &[u8],
) -> Result<Option<CollectionWalRecord>, String> {
    if payload.is_empty() || payload[0] != b'{' {
        return Ok(None);
    }
    let env: CollectionWalEnvelope = match serde_json::from_slice(payload) {
        Ok(e) => e,
        Err(err) => {
            // 区分「外来/历史记录」（静默跳过）与「腐败的 collection 记录」：
            // WAL 帧格式无校验和，JSON 可解析性是唯一的腐败探测器——若 op 是已知
            // collection op 而整体解析失败，这是数据腐败（位衰减/截断/未来写入侧
            // bug），静默跳过等于重放时无声丢数据，必须 fail-fast。
            // EN: Distinguish foreign/legacy records (skip silently) from corrupt
            // collection records: the WAL frame format has no checksum, so JSON
            // validity is the only corruption detector. A known collection op
            // whose body fails to parse means corruption (bit rot / truncation /
            // a future writer bug) — silently skipping it would lose data on
            // replay, so fail fast instead.
            #[derive(Deserialize)]
            struct OpProbe {
                op: String,
            }
            if let Ok(probe) = serde_json::from_slice::<OpProbe>(payload) {
                if matches!(
                    probe.op.as_str(),
                    OP_COLLECTION_CREATE
                        | OP_COLLECTION_UPSERT_BATCH
                        | OP_COLLECTION_DELETE
                        | OP_COLLECTION_SET_HA_LEVEL
                        | OP_COLLECTION_DROP
                ) {
                    return Err(format!(
                        "corrupt collection WAL record (op={}): {err}",
                        probe.op
                    ));
                }
            }
            return Ok(None);
        }
    };
    if env.v != COLLECTION_WAL_VERSION {
        return Ok(None);
    }
    match env.op.as_str() {
        OP_COLLECTION_CREATE => {
            let dims = env
                .dims
                .ok_or_else(|| "collection_create missing dims".to_string())?;
            let index_type = env
                .index_type
                .ok_or_else(|| "collection_create missing index_type".to_string())?;
            // 度量串必须在规范集合内：未知串 = 写入侧 bug/位衰减/未来版本，fail-fast
            //（与 ha_level 同纪律）。HNSW 三项必须全有或全无，避免半套参数被静默补默认。
            // EN: The metric string must be canonical — an unknown value means a
            // writer bug, bit rot, or a future version, so fail fast (same policy as
            // ha_level). The three HNSW fields must be all-present or all-absent so
            // half a parameter set is never silently completed with defaults.
            if let Some(metric) = env.metric.as_deref() {
                if !CollectionCreateOptions::CANONICAL_METRICS.contains(&metric) {
                    return Err(format!(
                        "collection_create (collection '{}') has non-canonical metric {metric:?}",
                        env.name
                    ));
                }
            }
            // 有 HNSW 参数时必须同时给出度量与非 hnsw 后端校验：缺度量会让重放
            // 退回 Cosine（与创建时度量静默分叉），非 hnsw 后端则意味着写入侧 bug
            // 或被篡改的记录（OCR #05/#24）。m 下限 2 与库层 set_hnsw_m 一致。
            // EN: HNSW params require a metric and an hnsw backend: a missing metric
            // would replay as cosine (silent divergence from creation semantics), and
            // a non-hnsw backend means a writer bug or tampered record (OCR #05/#24).
            // The m lower bound of 2 matches the library's set_hnsw_m.
            let has_hnsw = env.hnsw_m.is_some()
                || env.hnsw_ef_construction.is_some()
                || env.hnsw_ef_search.is_some();
            if has_hnsw {
                if env.metric.is_none() {
                    return Err(format!(
                        "collection_create (collection '{}') has hnsw params without a metric",
                        env.name
                    ));
                }
                if index_type != "hnsw" {
                    return Err(format!(
                        "collection_create (collection '{}') has hnsw params for index_type {:?}",
                        env.name, index_type
                    ));
                }
            }
            let hnsw = match (env.hnsw_m, env.hnsw_ef_construction, env.hnsw_ef_search) {
                (None, None, None) => None,
                (Some(m), Some(efc), Some(efs)) => {
                    if !(2..=1024).contains(&m)
                        || !(1..=65_536).contains(&efc)
                        || !(1..=65_536).contains(&efs)
                    {
                        return Err(format!(
                            "collection_create (collection '{}') has out-of-range hnsw params \
                             (m={m}, ef_construction={efc}, ef_search={efs})",
                            env.name
                        ));
                    }
                    Some((m, efc, efs))
                }
                _ => {
                    return Err(format!(
                        "collection_create (collection '{}') has partial hnsw params",
                        env.name
                    ))
                }
            };
            Ok(Some(CollectionWalRecord::Create {
                name: env.name,
                dims,
                index_type,
                options: CollectionCreateOptions {
                    metric: env.metric,
                    hnsw_m: hnsw.map(|(m, _, _)| m),
                    hnsw_ef_construction: hnsw.map(|(_, efc, _)| efc),
                    hnsw_ef_search: hnsw.map(|(_, _, efs)| efs),
                },
            }))
        }
        OP_COLLECTION_UPSERT_BATCH => {
            let dim = env
                .dims
                .ok_or_else(|| "collection_upsert_batch missing dims".to_string())?;
            let items = env
                .items
                .ok_or_else(|| "collection_upsert_batch missing items".to_string())?;
            for it in &items {
                if it.values.len() != dim {
                    return Err(format!(
                        "dim mismatch: item id={} len={} expected dim={}",
                        it.id,
                        it.values.len(),
                        dim
                    ));
                }
            }
            Ok(Some(CollectionWalRecord::UpsertBatch {
                name: env.name,
                dim,
                items,
            }))
        }
        OP_COLLECTION_DELETE => {
            let ids = env
                .ids
                .ok_or_else(|| "collection_delete missing ids".to_string())?;
            Ok(Some(CollectionWalRecord::Delete {
                name: env.name,
                ids,
            }))
        }
        OP_COLLECTION_SET_HA_LEVEL => {
            let ha_level = env
                .ha_level
                .ok_or_else(|| "collection_set_ha_level missing ha_level".to_string())?;
            // 落盘值必须可解析：不可解析 = 写入侧 bug 或位衰减，fail-fast
            // 而不是带着垃圾级别继续（重放侧与实时 apply 共用本解码）。
            // 已知权衡：未来版本若新增级别字符串，本二进制降级重放会在本臂
            // hard-fail（与未知 op 静默跳过不对称）——v1 接受：同级别表由单一
            // 发布序列控制，而位衰减防护（fail-fast）是现役威胁；新增级别时
            // 发布注记必须标注「旧二进制不可降级重放该 WAL」。（OCR 批次十八）
            // EN: A persisted level must parse — an unparseable value means a
            // writer bug or bit rot, so fail fast. Known trade-off: a future
            // release adding a level string would hard-brick this binary's cold
            // replay on downgrade (asymmetric with unknown ops, which skip).
            // Accepted for v1: bit-rot protection is the live threat; release
            // notes for any new level must flag the downgrade restriction.
            let level = crate::ha_level::HaLevel::parse(&ha_level)
                .map_err(|e| format!("collection_set_ha_level (collection '{}'): {e}", env.name))?;
            Ok(Some(CollectionWalRecord::SetHaLevel {
                name: env.name,
                ha_level: level,
            }))
        }
        OP_COLLECTION_DROP => Ok(Some(CollectionWalRecord::Drop { name: env.name })),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_items() -> Vec<CollectionWalItem> {
        let mut payload = serde_json::Map::new();
        payload.insert("tag".to_string(), serde_json::json!("p0"));
        payload.insert("score".to_string(), serde_json::json!(0.5));
        vec![
            CollectionWalItem {
                id: 1,
                values: vec![0.1, 0.2],
                label: Some("doc-a".to_string()),
                payload: Some(payload),
            },
            CollectionWalItem {
                id: 2,
                values: vec![0.3, 0.4],
                label: None,
                payload: None,
            },
        ]
    }

    #[test]
    fn roundtrip_create() {
        let bytes = encode_collection_create("docs", 768, "hnsw").unwrap();
        let rec = decode_collection_wal_payload(&bytes).unwrap().unwrap();
        assert_eq!(
            rec,
            CollectionWalRecord::Create {
                name: "docs".to_string(),
                dims: 768,
                index_type: "hnsw".to_string(),
                options: CollectionCreateOptions::default(),
            }
        );
    }

    #[test]
    fn roundtrip_create_with_metric_and_hnsw_options() {
        let options = CollectionCreateOptions {
            metric: Some("ip".to_string()),
            hnsw_m: Some(32),
            hnsw_ef_construction: Some(200),
            hnsw_ef_search: Some(64),
        };
        let bytes = encode_collection_create_with_options("docs", 128, "hnsw", &options).unwrap();
        let env: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(env["v"], 1, "envelope stays v1");
        assert_eq!(env["metric"], "ip");
        assert_eq!(env["hnsw_m"], 32);
        let rec = decode_collection_wal_payload(&bytes).unwrap().unwrap();
        assert_eq!(
            rec,
            CollectionWalRecord::Create {
                name: "docs".to_string(),
                dims: 128,
                index_type: "hnsw".to_string(),
                options,
            }
        );
        // 旧记录形态不受影响：无新字段时解码为默认 options。
        // EN: Legacy shape is unaffected: absent new fields decode to default options.
        let legacy = br#"{"v":1,"op":"collection_create","name":"d","dims":4,"index_type":"flat"}"#;
        match decode_collection_wal_payload(legacy).unwrap().unwrap() {
            CollectionWalRecord::Create { options, .. } => {
                assert_eq!(options, CollectionCreateOptions::default())
            }
            other => panic!("unexpected record: {other:?}"),
        }
    }

    #[test]
    fn create_options_validation_fails_closed() {
        // 非规范度量串：编码侧与解码侧都拒绝。
        // EN: Non-canonical metric: rejected by both writer and decoder.
        let bad = CollectionCreateOptions {
            metric: Some("dot".to_string()),
            ..Default::default()
        };
        assert!(encode_collection_create_with_options("d", 4, "hnsw", &bad).is_err());
        let corrupt =
            br#"{"v":1,"op":"collection_create","name":"d","dims":4,"index_type":"hnsw","metric":"manhattan"}"#;
        assert!(decode_collection_wal_payload(corrupt).is_err());
        // 半套 HNSW 参数视为腐败。
        // EN: A partial HNSW parameter set is corruption.
        let partial = br#"{"v":1,"op":"collection_create","name":"d","dims":4,"index_type":"hnsw","hnsw_m":16}"#;
        assert!(decode_collection_wal_payload(partial).is_err());
        // 超界参数 fail-fast。
        // EN: Out-of-range parameters fail fast.
        let out = br#"{"v":1,"op":"collection_create","name":"d","dims":4,"index_type":"hnsw","hnsw_m":4096,"hnsw_ef_construction":400,"hnsw_ef_search":256}"#;
        assert!(decode_collection_wal_payload(out).is_err());
        // 有 HNSW 参数却缺度量：重放不得静默退回 Cosine（OCR #05）。
        // EN: HNSW params without a metric must not replay as cosine (OCR #05).
        let no_metric = br#"{"v":1,"op":"collection_create","name":"d","dims":4,"index_type":"hnsw","hnsw_m":16,"hnsw_ef_construction":400,"hnsw_ef_search":256}"#;
        assert!(decode_collection_wal_payload(no_metric).is_err());
        // HNSW 参数给非 hnsw 后端：写入侧 bug/篡改，fail-fast（OCR #24）。
        // EN: HNSW params for a non-hnsw backend mean a writer bug or tampering
        // (OCR #24).
        let wrong_backend = br#"{"v":1,"op":"collection_create","name":"d","dims":4,"index_type":"flat","metric":"l2","hnsw_m":16,"hnsw_ef_construction":400,"hnsw_ef_search":256}"#;
        assert!(decode_collection_wal_payload(wrong_backend).is_err());
        // m=1 低于库层下限：编码/解码两侧都拒绝（OCR #39）。
        // EN: m=1 is below the library bound; both writer and decoder reject it.
        let m_one = br#"{"v":1,"op":"collection_create","name":"d","dims":4,"index_type":"hnsw","metric":"l2","hnsw_m":1,"hnsw_ef_construction":400,"hnsw_ef_search":256}"#;
        assert!(decode_collection_wal_payload(m_one).is_err());

        // 编码侧同规则：半套参数、非 hnsw 后端、缺度量、m=1。
        // EN: The writer mirrors these rules: partial set, non-hnsw backend,
        // missing metric, m=1.
        let partial = CollectionCreateOptions {
            metric: Some("l2".to_string()),
            hnsw_m: Some(16),
            ..Default::default()
        };
        assert!(encode_collection_create_with_options("d", 4, "hnsw", &partial).is_err());
        let wrong_backend = CollectionCreateOptions {
            metric: Some("l2".to_string()),
            hnsw_m: Some(16),
            hnsw_ef_construction: Some(400),
            hnsw_ef_search: Some(256),
        };
        assert!(encode_collection_create_with_options("d", 4, "flat", &wrong_backend).is_err());
        let mut no_metric = wrong_backend.clone();
        no_metric.metric = None;
        assert!(encode_collection_create_with_options("d", 4, "hnsw", &no_metric).is_err());
        let mut m_one = wrong_backend.clone();
        m_one.hnsw_m = Some(1);
        assert!(encode_collection_create_with_options("d", 4, "hnsw", &m_one).is_err());
    }

    #[test]
    fn roundtrip_upsert_batch() {
        let items = sample_items();
        let bytes = encode_collection_upsert_batch("docs", 2, &items).unwrap();
        let rec = decode_collection_wal_payload(&bytes).unwrap().unwrap();
        assert_eq!(
            rec,
            CollectionWalRecord::UpsertBatch {
                name: "docs".to_string(),
                dim: 2,
                items,
            }
        );
    }

    #[test]
    fn roundtrip_delete() {
        let bytes = encode_collection_delete("docs", &[7, 8, 9]).unwrap();
        let rec = decode_collection_wal_payload(&bytes).unwrap().unwrap();
        assert_eq!(
            rec,
            CollectionWalRecord::Delete {
                name: "docs".to_string(),
                ids: vec![7, 8, 9],
            }
        );
    }

    #[test]
    fn roundtrip_drop() {
        let bytes = encode_collection_drop("docs").unwrap();
        let rec = decode_collection_wal_payload(&bytes).unwrap().unwrap();
        assert_eq!(
            rec,
            CollectionWalRecord::Drop {
                name: "docs".to_string(),
            }
        );
    }

    #[test]
    fn drop_wire_shape_and_corruption() {
        // 信封保持 v1：旧二进制把未知 op 当外来记录静默跳过；新二进制按 drop 语义处理。
        // EN: The envelope stays v1: old binaries skip the unknown op, while this
        // binary applies the drop semantics.
        let bytes = encode_collection_drop("docs").unwrap();
        let env: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(env["v"], 1);
        assert_eq!(env["op"], "collection_drop");
        assert_eq!(env["name"], "docs");

        // 已知 op 但缺 name → 整体解析失败必须 fail-fast（不能静默跳过）。
        // EN: Known op missing `name` → must fail fast (never silently skip).
        let missing = br#"{"v":1,"op":"collection_drop"}"#;
        let err = decode_collection_wal_payload(missing).unwrap_err();
        assert!(err.contains("corrupt collection WAL record"), "got: {err}");

        // 未来版信封：版本不符跳过（与既有记录同纪律）。
        // EN: Future envelope version → skipped, same as existing record kinds.
        let future = br#"{"v":2,"op":"collection_drop","name":"docs"}"#;
        assert!(decode_collection_wal_payload(future).unwrap().is_none());
    }

    #[test]
    fn dim_mismatch_is_error_not_silent_skip() {
        let mut items = sample_items();
        items[0].values = vec![1.0];
        let bytes = encode_collection_upsert_batch("docs", 2, &items).unwrap();
        assert!(decode_collection_wal_payload(&bytes).is_err());
    }

    #[test]
    fn unknown_op_and_legacy_payloads_are_skipped() {
        assert!(decode_collection_wal_payload(b"bootstrap")
            .unwrap()
            .is_none());
        assert!(decode_collection_wal_payload(b"").unwrap().is_none());
        // 主向量 WAL 的记录混入（理论不应发生）也不得炸掉重放。
        // EN: Main-vector-WAL records mixed in (should not happen) must not break replay.
        let foreign = br#"{"v":1,"op":"vector_upsert_batch","dim":2,"items":[]}"#;
        assert!(decode_collection_wal_payload(foreign).unwrap().is_none());
        // 未来版本的记录：版本不符跳过。
        // EN: Future-version record: skipped on version mismatch.
        let future = br#"{"v":2,"op":"collection_delete","name":"d","ids":[1]}"#;
        assert!(decode_collection_wal_payload(future).unwrap().is_none());
    }

    #[test]
    fn corrupt_known_op_is_error_not_silent_skip() {
        // op 已知但 body 损坏（dims 类型错误）→ 必须 Err（重放 fail-fast），
        // 不得当作外来记录静默跳过（否则重放无声丢数据）。
        // EN: Known op with a corrupt body (dims type error) → must Err (replay
        // fail-fast); never silently skip it as foreign (that would lose data).
        let corrupt = br#"{"v":1,"op":"collection_create","name":"d","dims":"not-a-number"}"#;
        let err = decode_collection_wal_payload(corrupt).unwrap_err();
        assert!(err.contains("corrupt collection WAL record"), "got: {err}");
        // op 未知且 body 损坏 → 仍是外来记录，静默跳过。
        // EN: Unknown op with a corrupt body → still foreign, silently skipped.
        let foreign = br#"{"v":1,"op":"something_else","name":"d","dims":"bad"}"#;
        assert!(decode_collection_wal_payload(foreign).unwrap().is_none());
        // 整体非 JSON（连 op 都探测不出）→ 静默跳过（帧层已负责截断尾丢弃）。
        // EN: Not JSON at all (op cannot be probed) → silently skipped (frame layer drops tails).
        assert!(
            decode_collection_wal_payload(b"{\"op\":\"collection_create\",")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn roundtrip_set_ha_level() {
        use crate::ha_level::HaLevel;
        let bytes = encode_collection_set_ha_level("docs", HaLevel::SemiSync).unwrap();
        let rec = decode_collection_wal_payload(&bytes).unwrap().unwrap();
        assert_eq!(
            rec,
            CollectionWalRecord::SetHaLevel {
                name: "docs".to_string(),
                ha_level: HaLevel::SemiSync,
            }
        );
    }

    #[test]
    fn encode_set_ha_level_always_persists_canonical_string() {
        // 回归（OCR dcace87[2]）：「落盘必为规范串」由枚举 + 内部渲染保证。
        // P4.6.3 起非法级别在类型层不可表达（旧 u8 版本此处是运行时守卫）。
        // EN: canonical-by-construction now; invalid levels are unrepresentable.
        use crate::ha_level::HaLevel;
        for level in [HaLevel::Async, HaLevel::SemiSync, HaLevel::FullSync] {
            let bytes = encode_collection_set_ha_level("docs", level).unwrap();
            let env: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(env["ha_level"], level.as_str());
        }
    }

    #[test]
    fn set_ha_level_compat_and_corruption() {
        // 旧二进制视角：未知 op 整体可解析 → 静默跳过（向前兼容纪律不变）。
        // EN: An old binary skips the unknown op silently.
        let bytes =
            encode_collection_set_ha_level("docs", crate::ha_level::HaLevel::FullSync).unwrap();
        let env: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(env["op"], "collection_set_ha_level");
        assert_eq!(env["ha_level"], "full_sync");

        // op 已知但 ha_level 不可解析 → 腐败 fail-fast（OpProbe 列表同步覆盖）。
        // EN: Known op with an unparsable ha_level → corruption fail-fast (OpProbe covers it).
        let corrupt = br#"{"v":1,"op":"collection_set_ha_level","name":"d","ha_level":"sycn"}"#;
        assert!(decode_collection_wal_payload(corrupt).is_err());
        // 缺 ha_level 字段 → Err（missing 字段消息）。
        // EN: Missing ha_level field → Err (missing-field message).
        let missing = br#"{"v":1,"op":"collection_set_ha_level","name":"d"}"#;
        let err = decode_collection_wal_payload(missing).unwrap_err();
        assert!(err.contains("missing ha_level"), "got: {err}");
        // body 损坏到整体解析失败 → OpProbe fail-fast（不是静默跳过）。
        // EN: Body corrupt enough to fail full parsing → OpProbe fail-fast (not skipped).
        let broken = br#"{"v":1,"op":"collection_set_ha_level","name":"d","ha_level":42}"#;
        let err = decode_collection_wal_payload(broken).unwrap_err();
        assert!(err.contains("corrupt collection WAL record"), "got: {err}");
    }

    #[test]
    fn derived_path_sits_next_to_main_wal() {
        let p = collections_wal_path_for(Path::new("/data/houyidb.wal"));
        assert_eq!(p, PathBuf::from("/data/houyidb.collections.wal"));
    }
}
