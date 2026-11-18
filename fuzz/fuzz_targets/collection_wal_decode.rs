#![no_main]
//! 目标 6：collections WAL payload 解码（P4）。
//! EN: Target 6: collections WAL payload decoding (P4).
//!
//! 断言的不变量 / Invariants:
//! - 任意字节喂给 `decode_collection_wal_payload` 只返回 `Ok(None)`（外来/旧
//!   记录跳过）、`Ok(Some(..))`（合法记录）或 `Err(..)`（已知 op 但正文腐败），
//!   不得 panic、不得 OOM（伪造的 items/ids/稀疏数组量级由 serde 与 checked
//!   算术拦住）。
//! EN: - Arbitrary bytes into `decode_collection_wal_payload` may only return
//! EN:   `Ok(None)` (foreign/legacy record skipped), `Ok(Some(..))` (valid record)
//! EN:   or `Err(..)` (known op with a corrupt body): never panic and never OOM
//! EN:   (forged item/id/sparse array sizes are stopped by serde and checked
//! EN:   arithmetic).
//!
//! 覆盖策略 / Coverage strategy: JSON 结构门会挡住随机变异，语料里放真实编码
//! 器输出（create 含 `sparse=bm25`、upsert 含稀疏词项 / 空列表 / 缺省、delete、
//! drop、set_ha_level），变异才落在 JSON 字段、稀疏数组与 op 分派上。
//! EN: The JSON structure gate blocks random mutation, so the corpus carries real
//! EN: encoder outputs (create with `sparse=bm25`, upsert with sparse terms /
//! EN: an empty list / an absent field, delete, drop, set_ha_level) so mutations
//! EN: land on JSON fields, the sparse array, and op dispatch.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    use houyi_persistence::collection_wal::decode_collection_wal_payload;

    let _ = decode_collection_wal_payload(data);
});
