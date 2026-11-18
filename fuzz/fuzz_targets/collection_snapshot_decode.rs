#![no_main]
//! 目标 5：collections 快照 payload 与 manifest 解码（P5.0.5）。
//! EN: Target 5: collections snapshot payload and manifest decoding (P5.0.5).
//!
//! 断言的不变量 / Invariants:
//! - 任意字节喂给 `decode_collection_snapshot_payload` / `decode_collection_manifest`
//!   只返回 `Err`，不得 panic、不得 OOM（伪造的 u64 级计数必须走 checked 算术与
//!   `try_reserve`，而不是先分配后校验）。
//! EN: - Arbitrary bytes into `decode_collection_snapshot_payload` /
//! EN:   `decode_collection_manifest` may only return `Err`: never panic and never
//! EN:   OOM (forged u64-scale counts must go through checked arithmetic and
//! EN:   `try_reserve`, not allocate-then-validate).
//!
//! 覆盖策略 / Coverage strategy: magic/version 会挡住随机变异，语料里放真实
//! payload 与 manifest（未量化 v1、int8 v2/v3、RaBitQ：payload flags
//! bit2=0x04、图快照 v4、manifest format 3，以及稀疏 v3：payload flags
//! bit3=0x08、tombstones 之后的倒排段、manifest format 4 与条目 sparse=1），
//! 变异才能真正落在计数、列名、标量 tag、稀疏 term/posting 排序与四个版本号
//! 的交叉校验上。
//! EN: magic/version blocks random mutation, so the corpus carries real payloads
//! and manifests (unquantized v1, int8 v2/v3, RaBitQ: payload flags bit2=0x04,
//! graph snapshot v4, manifest format 3, and sparse v3: payload flags bit3=0x08,
//! the inverted-index section after tombstones, manifest format 4 with entry
//! sparse=1) so mutations land on counts, column names, scalar tags, sparse
//! term/posting ordering, and the four version marks' cross-checks.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    use houyi_persistence::collection_snapshot_file::{
        decode_collection_manifest, decode_collection_snapshot_payload,
    };

    let _ = decode_collection_snapshot_payload(data);
    let _ = decode_collection_manifest(data);

    // 结构感知的第二入口：跳过 1 字节 selector，让 fuzzer 用同一语料分别打两条
    // 解码器，而不必让两条 magic 在同一份前缀上竞争。
    // EN: A structure-aware second entry: skip one selector byte so the fuzzer can
    // target either decoder with one corpus without two magics competing for the
    // same prefix.
    if let Some((_, rest)) = data.split_first() {
        let _ = decode_collection_snapshot_payload(rest);
        let _ = decode_collection_manifest(rest);
    }
});
