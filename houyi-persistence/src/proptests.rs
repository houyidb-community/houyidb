//! P4.5.0 属性化测试基座：WAL v2 / collections WAL / 复制协议帧的编解码不变量。
//! Property-based test foundation (P4.5.0): codec invariants for the WAL v2
//! on-disk format, the collections WAL envelope, and the replication frames.
//!
//! 覆盖的不变量 / Invariants covered:
//! - WAL v2：encode→parse roundtrip 等价；任意截断只产出**原始记录的前缀**
//!   （撕裂尾 = 日志终点）；单字节翻转在整文件解析下同样只产出前缀（首个
//!   校验失败点即终点），tail 增量模式下腐坏（完整记录 CRC 不匹配）必须
//!   报错而非静默停等；`append_entry_no_flush` 拒绝 LSN 空洞/回退且失败
//!   不消耗 LSN。
//! - collections WAL：五种记录 encode→decode roundtrip；任意字节不 panic；
//!   未知 op / v≠1 静默跳过（Ok(None)）；已知 op 腐败 fail-fast（Err）。
//! - 复制帧：全帧型 roundtrip；任意字节不 panic；64MB payload 守卫与
//!   ACK stage / ERROR kind 校验不被绕过。

use crate::collection_wal::{
    decode_collection_wal_payload, encode_collection_create_with_options, encode_collection_delete,
    encode_collection_drop, encode_collection_set_ha_level, encode_collection_upsert_batch,
    CollectionCreateOptions, CollectionWalItem, CollectionWalRecord, COLLECTION_WAL_VERSION,
};
use crate::ha_level::HaLevel;
use crate::wal_replication::{
    decode_error_msg, decode_frame_header, decode_handshake_prefix, decode_handshake_token,
    decode_welcome, encode_ack, encode_data_header, encode_error, encode_handshake,
    encode_heartbeat, encode_welcome, frame_header_len, FrameHeader, ACK_STAGE_DURABLE,
    ACK_STAGE_RECEIVED, ERROR_KINDS, ERROR_KIND_FOLLOWER_AHEAD, ERROR_KIND_GENERIC,
    HANDSHAKE_PREFIX_LEN, MAX_PAYLOAD_BYTES,
};
use crate::{
    encode_record, parse_v2_records_tail, parse_wal_file_entries, unique_temp_path, FileWal,
    WalEntry, WAL_HEADER_LEN, WAL_MAGIC,
};
use proptest::prelude::*;

/// WAL 属性测试的 payload 尺寸上限：覆盖空记录与多记录拼接即可，无需大 payload
/// （大 payload 的撕裂行为与小的同构，且 256B 已跨多条记录）。
/// EN: Payload size cap for WAL property tests: empty and multi-record payloads
/// suffice (large payloads tear like small ones, and 256B already spans records).
const MAX_PROP_PAYLOAD: usize = 256;

/// 临时文件的 RAII 清理。属性断言失败会提前 `return`/panic，只在末尾
/// `remove_file` 会把文件漏在 /tmp —— 清理必须挂在 Drop 上（OCR 批次修订）。
/// EN: RAII cleanup for temp files: a failing `prop_assert!` returns early, so a
/// trailing `remove_file` leaks the file; the drop guard cannot be skipped.
struct TempFileGuard(std::path::PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// 手工拼帧时的字段偏移。`wal_replication` 只导出整帧长度（`ACK_LEN` 等），
/// 没有「帧内第几个字节是什么」的常量——协议布局改动时这里必须跟着改，故把
/// 偏移集中在此锚定一次，用例里不再出现裸字面量（OCR 批次修订）。
/// ERROR 头 = tag(1) + kind(1) + msg_len u16 BE(2)；ACK = tag/stream/stage/lsn8。
/// EN: Field offsets for hand-built frames. `wal_replication` exports whole-frame
/// lengths (`ACK_LEN` etc.) but no per-byte layout constants — protocol layout
/// changes must be mirrored here, so offsets are anchored once and tests avoid
/// bare literals (OCR batch revision).
/// ERROR header = tag(1) + kind(1) + msg_len u16 BE(2); ACK = tag/stream/stage/lsn8.
const ERROR_HEADER_LEN: usize = 4;
const ACK_STAGE_OFFSET: usize = 2;

/// 生成一组 LSN 连续（从 0 起）的随机记录。
/// EN: Generates a random record set with contiguous LSNs starting at 0.
fn arb_entries() -> impl Strategy<Value = Vec<(u64, Vec<u8>)>> {
    prop::collection::vec(
        prop::collection::vec(any::<u8>(), 0..MAX_PROP_PAYLOAD),
        0..12,
    )
    .prop_map(|payloads| {
        payloads
            .into_iter()
            .enumerate()
            .map(|(i, p)| (i as u64, p))
            .collect::<Vec<_>>()
    })
}

/// 把记录集编码成完整 v2 文件字节（文件头 + 记录序列）。
/// EN: Encodes the record set into full v2 file bytes (header + record sequence).
fn encode_v2_file(entries: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(WAL_MAGIC);
    data.extend_from_slice(&crate::WAL_FORMAT_VERSION.to_be_bytes());
    for (lsn, payload) in entries {
        data.extend_from_slice(&encode_record(*lsn, payload, crate::WalFormat::V2, None).unwrap());
    }
    data
}

fn assert_prefix(actual: &[WalEntry], expected_prefix_source: &[(u64, Vec<u8>)]) {
    assert!(
        actual.len() <= expected_prefix_source.len(),
        "parsed {} entries from {} encoded — replay must never invent records",
        actual.len(),
        expected_prefix_source.len()
    );
    for (i, e) in actual.iter().enumerate() {
        assert_eq!(
            e.lsn, expected_prefix_source[i].0,
            "entry {i} lsn diverged — parsed stream is not a prefix of the encoded stream"
        );
        assert_eq!(
            e.payload, expected_prefix_source[i].1,
            "entry {i} payload diverged — parsed stream is not a prefix of the encoded stream"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        // 本地门禁分钟级预算；nightly 经 PROPTEST_CASES 放大。
        // EN: Minute-scale budget for local gates; nightly scales via PROPTEST_CASES.
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// WAL v2 roundtrip：任意记录集 encode → 整文件解析等价还原。
    /// EN: WAL v2 roundtrip: encode any record set → whole-file parse restores it exactly.
    #[test]
    fn wal_v2_roundtrip(entries in arb_entries()) {
        let data = encode_v2_file(&entries);
        let parsed = parse_wal_file_entries(&data).expect("clean v2 file must parse");
        prop_assert_eq!(parsed.len(), entries.len());
        assert_prefix(&parsed, &entries);
    }

    /// 任意截断（含文件头中间）：解析要么版本门报错，要么产出原始记录的前缀，
    /// 绝不 panic、绝不凭空多出记录。
    /// EN: Any truncation (including inside the header): parsing either fails the
    /// version gate or yields a prefix of the original records — never panics and
    /// never invents records.
    #[test]
    fn wal_v2_truncation_yields_prefix(entries in arb_entries(), cut in 0usize..4096) {
        let data = encode_v2_file(&entries);
        let cut = cut.min(data.len());
        let truncated = &data[..cut];
        // 截断点落在文件头内：magic 不完整 → 按 v1 遗留解析（静默容错路径），
        // 只要求不 panic；落在记录区 → 必须是前缀。
        // EN: Truncation inside the header: incomplete magic → parsed via the legacy
        // v1 path (silent tolerance), only no-panic is required; inside records →
        // the result must be a prefix.
        if cut >= WAL_HEADER_LEN {
            let parsed = parse_wal_file_entries(truncated).expect("v2 header intact: torn tail must be tolerated");
            assert_prefix(&parsed, &entries);
        } else {
            let _ = parse_wal_file_entries(truncated);
        }
    }

    /// 单字节翻转纪律（腐坏面）：整文件解析在首个校验失败点停止，产出前缀；
    /// tail 增量模式对「完整记录 CRC 不匹配」必须 Err（腐坏 ≠ 撕裂，不容静默）。
    /// EN: Single-byte flip discipline (corruption side): whole-file parsing stops
    /// at the first failed check and yields a prefix; tail incremental mode must
    /// Err on a complete-record CRC mismatch (corruption != torn tail).
    #[test]
    fn wal_v2_bitflip_discipline(
        entries in arb_entries().prop_filter("at least one record", |e| !e.is_empty()),
        pos_seed in any::<usize>(),
    ) {
        let data = encode_v2_file(&entries);
        // 翻转点限定在记录区（跳过文件头——magic 翻转会触发 v1 兼容路径，单独覆盖）。
        // EN: Flip point restricted to the record area (the header is skipped —
        // magic flips take the v1 compat path and are covered separately).
        let pos = WAL_HEADER_LEN + pos_seed % (data.len() - WAL_HEADER_LEN);
        let mut corrupted = data.clone();
        corrupted[pos] ^= 0x5A;

        // 整文件（冷重放语义）：产出前缀，且最多走到被翻转记录的前一条。
        // EN: Whole file (cold replay semantics): yields a prefix, stopping at or
        // before the flipped record.
        let parsed = parse_wal_file_entries(&corrupted)
            .expect("bit flip inside records must not break the parser itself");
        assert_prefix(&parsed, &entries);
        prop_assert!(parsed.len() < entries.len(),
            "flip at byte {pos} must terminate the log at or before the corrupted record");

        // tail 模式：翻转落在长度字段 → 越界呈撕裂语义（Ok(前缀)）；落在 lsn /
        // CRC 字段 / payload → 完整记录 CRC 不匹配 → 腐坏 Err。**两种都是合法结果**
        // （Err 正是「腐坏不静默」的纪律），故只对 Ok 侧设断言。断言必须是**严格**
        // 前缀：长度字段变更要么越界截断、要么改变 CRC 覆盖区间，任何一条记录被
        // 翻转都不可能整条流原样通过——只断「是前缀」对空结果平凡成立 = 属性真空
        // （OCR 批次修订）。
        // EN: Tail mode: a flip in the length field → out of bounds reads as a torn
        // tail (Ok(prefix)); in the lsn / CRC field / payload → complete-record CRC
        // mismatch → corruption Err. **Both are legal** (Err is the "no silent
        // corruption" rule), so only the Ok side asserts. The assertion must be a
        // **strict** prefix: a length change either truncates out of bounds or
        // shifts CRC coverage, so no flipped record can pass the stream unchanged —
        // asserting only "is a prefix" holds vacuously for an empty result (OCR).
        if let Ok((tail_entries, _)) = parse_v2_records_tail(&corrupted[WAL_HEADER_LEN..]) {
            assert_prefix(&tail_entries, &entries);
            prop_assert!(
                tail_entries.len() < entries.len(),
                "flip at byte {pos} must terminate the tail at or before the corrupted record"
            );
        }
    }

    /// 文件头 magic 翻转 → 按 v1 遗留路径解析：只要求不 panic（v1 无校验，
    /// 垃圾进垃圾出属记录的兼容策略，不是安全属性）。
    /// EN: Header magic flip → parsed via the legacy v1 path: only no-panic is
    /// required (v1 has no checksums; garbage-in/garbage-out is a recorded
    /// compatibility policy, not a security property).
    #[test]
    fn wal_v2_magic_flip_never_panics(entries in arb_entries(), pos in 0usize..WAL_MAGIC.len()) {
        let mut data = encode_v2_file(&entries);
        data[pos] ^= 0x01;
        let _ = parse_wal_file_entries(&data);
    }

    /// 版本号翻转 → 版本门 fail-fast（Err），绝不按错误版本误解析。
    /// 翻转域由常量派生（`WAL_MAGIC.len()..WAL_HEADER_LEN`）而非字面量 8..12：
    /// 文件头布局一旦变化，字面量会静默打到 magic 上——本用例名不副实，
    /// 而 magic 用例则退化成翻版本号（OCR 批次修订）。
    /// EN: Version-number flip → version gate fail-fast (Err); never parse under
    /// the wrong version. The flip range derives from constants
    /// (`WAL_MAGIC.len()..WAL_HEADER_LEN`), not the literal 8..12: if the header
    /// layout changes, literals would silently hit magic, making this test
    /// misnamed while the magic test degrades into a version flip (OCR revision).
    #[test]
    fn wal_v2_version_flip_rejected(entries in arb_entries(), pos in WAL_MAGIC.len()..WAL_HEADER_LEN) {
        let mut data = encode_v2_file(&entries);
        data[pos] ^= 0x01;
        // 翻转后版本仍为 2 的概率存在（bit 翻转回自身不可能——XOR 0x01 必改值），
        // 故必然触发版本门。
        // EN: The version cannot stay 2 after the flip (a bit cannot flip to itself —
        // XOR 0x01 always changes the value), so the version gate necessarily triggers.
        prop_assert!(parse_wal_file_entries(&data).is_err());
    }

    /// tail 解析器对任意字节不 panic（在线增量的鲁棒性底线）。
    /// EN: The tail parser never panics on arbitrary bytes (robustness floor for
    /// online incremental reads).
    #[test]
    fn wal_v2_tail_parse_never_panics(data in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = parse_v2_records_tail(&data);
    }

    /// FileWal LSN 纪律：随机 append 尝试序列下，仅 lsn == next_lsn 被接受；
    /// 被拒的尝试不消耗 LSN、不污染后续追加；最终重放等于被接受的前缀序列。
    /// EN: FileWal LSN discipline: over a random sequence of append attempts, only
    /// lsn == next_lsn is accepted; rejected attempts consume no LSN and do not
    /// pollute later appends; final replay equals the accepted prefix sequence.
    #[test]
    fn file_wal_append_lsn_gap_rejected(
        attempts in prop::collection::vec((any::<u8>(), prop::bool::ANY), 1..24),
    ) {
        let path = unique_temp_path("houyidb-wal-prop-lsn", "log");
        let _guard = TempFileGuard(path.clone());
        let (mut wal, initial) = FileWal::open(&path).expect("open fresh wal");
        prop_assert!(initial.is_empty());

        let mut accepted: Vec<Vec<u8>> = Vec::new();
        let mut expected_next = 0u64;
        for (i, (payload_byte, make_gap)) in attempts.into_iter().enumerate() {
            let candidate = if make_gap && expected_next > 0 {
                // 空洞（跳号）或回退（旧号）交替制造。
                // EN: Alternate between holes (skipped LSN) and regressions (stale LSN).
                if i % 2 == 0 { expected_next + 1 } else { expected_next - 1 }
            } else if make_gap {
                expected_next + 7
            } else {
                expected_next
            };
            let payload = vec![payload_byte; (i % 5) + 1];
            let result = wal.append_entry_no_flush(candidate, payload.clone());
            if candidate == expected_next {
                let entry = result.expect("in-sequence append must succeed");
                prop_assert_eq!(entry.lsn, expected_next);
                accepted.push(payload);
                expected_next += 1;
            } else {
                prop_assert!(result.is_err(),
                    "gap/stale lsn {candidate} must be rejected at next_lsn {expected_next}");
                prop_assert_eq!(wal.next_lsn(), expected_next,
                    "rejected append must not consume an LSN");
            }
        }
        wal.flush_wal().expect("flush");
        drop(wal);

        let (_reopened, replayed) = FileWal::open(&path).expect("reopen");
        prop_assert_eq!(replayed.len(), accepted.len());
        for (i, e) in replayed.iter().enumerate() {
            prop_assert_eq!(e.lsn, i as u64);
            prop_assert_eq!(&e.payload, &accepted[i]);
        }
    }

    // ── collections WAL（collection WAL 属性）────────────────────
    // EN: collections WAL section.

    /// 五种 collection 记录的 encode → decode roundtrip。
    /// EN: encode → decode roundtrip for all five collection record kinds.
    #[test]
    fn collection_wal_roundtrip(record in arb_collection_wal_record()) {
        let encoded = encode_record_for(&record);
        let decoded = decode_collection_wal_payload(&encoded).expect("self-encoded record must decode");
        prop_assert_eq!(decoded, Some(record));
    }

    /// 任意字节不 panic；非 JSON / v≠1 / 未知 op 一律静默跳过（Ok(None)）。
    /// EN: Arbitrary bytes never panic; non-JSON / v != 1 / unknown op are silently
    /// skipped (Ok(None)).
    #[test]
    fn collection_wal_arbitrary_bytes_never_panic(data in prop::collection::vec(any::<u8>(), 0..256)) {
        let _ = decode_collection_wal_payload(&data);
    }

    /// 未知 op 的合法 JSON 信封 → Ok(None)（向前兼容跳过）。
    /// EN: Valid JSON envelope with an unknown op → Ok(None) (forward-compatible skip).
    #[test]
    fn collection_wal_unknown_op_skipped(op in "[a-z_]{4,24}") {
        let payload = serde_json::json!({
            "v": COLLECTION_WAL_VERSION,
            "op": op,
            "name": "c",
        });
        // 已知 op 不在本用例射程（由 roundtrip 与腐败用例覆盖）。
        // EN: Known ops are out of scope here (covered by roundtrip and corruption tests).
        prop_assume!(!matches!(
            op.as_str(),
            "collection_create" | "collection_upsert_batch" | "collection_delete" | "collection_set_ha_level" | "collection_drop"
        ));
        let decoded = decode_collection_wal_payload(payload.to_string().as_bytes())
            .expect("valid JSON with unknown op must not fail");
        prop_assert_eq!(decoded, None);
    }

    /// v≠1 的合法信封 → Ok(None)（旧版/未来版整体跳过）。
    /// EN: Valid envelope with v != 1 → Ok(None) (old/future versions skipped wholesale).
    #[test]
    fn collection_wal_version_mismatch_skipped(v in (COLLECTION_WAL_VERSION + 1)..1000) {
        let payload = serde_json::json!({
            "v": v,
            "op": "collection_create",
            "name": "c",
            "dims": 4,
            "index_type": "hnsw",
        });
        let decoded = decode_collection_wal_payload(payload.to_string().as_bytes())
            .expect("version mismatch must be a skip, not an error");
        prop_assert_eq!(decoded, None);
    }

    /// 已知 op + 腐败字段 → fail-fast Err（JSON 可解析性是 WAL 帧唯一的
    /// 腐败探测器，静默跳过 = 重放无声丢数据）。
    /// EN: Known op + corrupt fields → fail-fast Err (JSON parsability is the WAL
    /// frame's only corruption detector; a silent skip means silent data loss on replay).
    #[test]
    fn collection_wal_corrupt_known_op_fails(op_idx in 0usize..5) {
        // 构造 op 合法但必填字段缺失/错型的信封。版本号取常量而非字面量：
        // 版本 bump 时本用例必须跟着走「已知 op」分支，硬编码会静默退化成
        // 「版本不匹配 → Ok(None)」的假通过。
        // EN: Build envelopes with a valid op but missing/mistyped required fields.
        // The version comes from the constant, not a literal: on a version bump this
        // test must follow the "known op" branch; a hardcoded value would silently
        // degrade into a false pass via "version mismatch → Ok(None)".
        let v = COLLECTION_WAL_VERSION;
        let payload = match op_idx {
            0 => serde_json::json!({"v": v, "op": "collection_create", "name": "c"}),
            1 => serde_json::json!({"v": v, "op": "collection_upsert_batch", "name": "c", "dims": 4}),
            2 => serde_json::json!({"v": v, "op": "collection_delete", "name": "c"}),
            3 => serde_json::json!({"v": v, "op": "collection_set_ha_level", "name": "c"}),
            _ => serde_json::json!({"v": v, "op": "collection_drop"}),
        };
        prop_assert!(decode_collection_wal_payload(payload.to_string().as_bytes()).is_err(),
            "known op with missing required fields must fail fast");
    }

    // ── 复制协议帧 ──────────────────────────────────────────────
    // EN: Replication protocol frames section.

    /// 全帧型 roundtrip（HANDSHAKE/WELCOME/DATA 头/HEARTBEAT/ACK/ERROR）。
    /// EN: Roundtrip for every frame type (HANDSHAKE/WELCOME/DATA header/HEARTBEAT/ACK/ERROR).
    #[test]
    fn repl_frame_roundtrip(
        resume_lsn in any::<u64>(),
        leader_next in any::<u64>(),
        log_hash in any::<u64>(),
        stream in 0u8..4,
        lsn in any::<u64>(),
        payload_len in 0u32..1024,
        stage in prop::sample::select(vec![ACK_STAGE_RECEIVED, ACK_STAGE_DURABLE]),
        kind in prop::sample::select(ERROR_KINDS),
        msg in ".{0,128}",
    ) {
        let follower_id = [0x5Au8; 16];
        let hs = encode_handshake(resume_lsn, 1, log_hash, 0, 0, follower_id, "tok")?;
        let prefix = decode_handshake_prefix(&hs).unwrap();
        prop_assert_eq!(
            (
                prefix.resume_lsn,
                prefix.stream_mask,
                prefix.log_hash,
                prefix.log_start_lsn,
                prefix.snapshot_checkpoint,
                prefix.follower_id,
            ),
            (resume_lsn, 1, log_hash, 0, 0, follower_id)
        );
        prop_assert_eq!(prefix.token_len, 3);
        prop_assert_eq!(
            decode_handshake_token(&hs[HANDSHAKE_PREFIX_LEN..]).unwrap(),
            "tok"
        );

        let wl = encode_welcome(leader_next, 0);
        let welcome = decode_welcome(&wl).unwrap();
        prop_assert_eq!(welcome.leader_next_lsn, leader_next);
        prop_assert_eq!(welcome.leader_first_available_lsn, 0);

        let dh = encode_data_header(stream, lsn, payload_len);
        prop_assert_eq!(
            decode_frame_header(&dh).unwrap(),
            FrameHeader::Data { stream, lsn, payload_len }
        );

        let hb = encode_heartbeat(stream, leader_next);
        prop_assert_eq!(
            decode_frame_header(&hb).unwrap(),
            FrameHeader::Heartbeat { stream, leader_next_lsn: leader_next }
        );

        let ack = encode_ack(stream, stage, lsn);
        prop_assert_eq!(
            decode_frame_header(&ack).unwrap(),
            FrameHeader::Ack { stream, stage, lsn }
        );

        let err = encode_error(kind, &msg);
        let head = decode_frame_header(&err[..ERROR_HEADER_LEN]).unwrap();
        let want_len = msg.len().min(crate::wal_replication::MAX_ERROR_MSG_BYTES) as u16;
        prop_assert_eq!(head, FrameHeader::Error { kind, msg_len: want_len });
        let decoded_msg = decode_error_msg(&err[ERROR_HEADER_LEN..]);
        prop_assert_eq!(decoded_msg, msg);
    }

    /// 任意字节不 panic（会话层对不可信对端的底线）。
    /// EN: Arbitrary bytes never panic (floor for a session layer facing untrusted peers).
    #[test]
    fn repl_frame_arbitrary_bytes_never_panic(data in prop::collection::vec(any::<u8>(), 0..64)) {
        let _ = decode_frame_header(&data);
        let _ = frame_header_len(data.first().copied().unwrap_or(0));
    }

    /// 64MB payload 守卫不得被绕过：超限的 DATA 头必须 Err。
    /// EN: The 64MB payload guard must not be bypassable: an oversized DATA header must Err.
    #[test]
    fn repl_data_payload_guard(oversize in (MAX_PAYLOAD_BYTES + 1)..u32::MAX) {
        let header = encode_data_header(0, 42, oversize);
        prop_assert!(decode_frame_header(&header).is_err(),
            "payload_len {oversize} must be rejected by the 64MB guard");
    }

    /// ACK stage 校验：非法 stage（解码侧）必须 Err——与编码侧归一纪律对偶。
    /// 域含 **0**（`ACK_STAGE_RECEIVED = 1` / `DURABLE = 2`，0 是历史二义值：
    /// `lsn = 0` 原点 vs 未初始化 stage，必须被拒）。
    /// EN: ACK stage validation: an invalid stage (decode side) must Err — the dual
    /// of encoder-side normalization. The domain includes **0**
    /// (`ACK_STAGE_RECEIVED = 1` / `DURABLE = 2`; 0 is a historic ambiguous value:
    /// `lsn = 0` origin vs uninitialized stage — it must be rejected).
    #[test]
    fn repl_ack_invalid_stage_rejected(stage in prop_oneof![0u8..1, 3u8..=u8::MAX]) {
        let mut ack = encode_ack(0, ACK_STAGE_RECEIVED, 7);
        ack[ACK_STAGE_OFFSET] = stage;
        prop_assert!(decode_frame_header(&ack).is_err(),
            "invalid ack stage {stage} must be rejected at decode");
    }
}

// ── 守卫接受侧边界（非随机：定值断言，随机域无意义）──────────────
// EN: Guard acceptance-side boundaries (fixed assertions; random domains are meaningless).

/// `payload_len == MAX_PAYLOAD_BYTES` 与 `msg_len == MAX_ERROR_MSG_BYTES` 必须 Ok。
/// 只测拒绝侧会漏掉「守卫被收紧成 `>=` / `<` 的 off-by-one 回归」——那会把合法
/// 上界帧拒掉，且线上表现为「最大合法帧永不通过」（fuzz 目标同款断言，两处独立）。
/// EN: `payload_len == MAX_PAYLOAD_BYTES` and `msg_len == MAX_ERROR_MSG_BYTES` must
/// be Ok. Testing only the reject side misses off-by-one regressions where a guard
/// is tightened to `>=` / `<` — that rejects legal upper-bound frames, showing up
/// online as "the largest legal frame never passes" (same assertion in the fuzz
/// target, independently).
#[test]
fn repl_frame_guard_boundaries_accepted() {
    let data = encode_data_header(0, 42, MAX_PAYLOAD_BYTES);
    assert_eq!(
        decode_frame_header(&data).unwrap(),
        FrameHeader::Data {
            stream: 0,
            lsn: 42,
            payload_len: MAX_PAYLOAD_BYTES,
        }
    );

    let max_msg = crate::wal_replication::MAX_ERROR_MSG_BYTES;
    let mut err = encode_error(ERROR_KIND_GENERIC, "x");
    err[2..ERROR_HEADER_LEN].copy_from_slice(&(max_msg as u16).to_be_bytes());
    assert_eq!(
        decode_frame_header(&err[..ERROR_HEADER_LEN]).unwrap(),
        FrameHeader::Error {
            kind: ERROR_KIND_GENERIC,
            msg_len: max_msg as u16,
        }
    );
}

// ── collections WAL 记录生成器 ──────────────────────────────────
// EN: collections WAL record generators.

/// 生成有限 f32（排除 NaN：PartialEq 语义与 JSON roundtrip 的共同前提）。
/// EN: Generates finite f32 (NaN excluded: shared precondition for PartialEq and JSON roundtrip).
fn arb_finite_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("finite", |f| f.is_finite())
}

fn arb_collection_name() -> impl Strategy<Value = String> {
    prop::sample::select(vec!["alpha", "beta", "gamma"]).prop_map(str::to_string)
}

/// 标量 payload 值：扁平 JSON（Int/Bool/Text/Float），不含 null——null 与
/// PAX 页缺失列占位同形，回读侧本就不可区分，属性测试不掺这个已知取舍。
/// EN: Scalar payload values: flat JSON (Int/Bool/Text/Float), no null — null has
/// the same shape as the PAX page's missing-column placeholder and is inherently
/// indistinguishable on readback, so property tests avoid this known trade-off.
fn arb_json_scalar() -> impl Strategy<Value = serde_json::Value> {
    prop_oneof![
        (-1_000_000i64..1_000_000).prop_map(serde_json::Value::from),
        any::<bool>().prop_map(serde_json::Value::from),
        "[a-z ]{0,16}".prop_map(serde_json::Value::from),
        // 限定在 f64→f32→JSON 全链路精确往返的粗粒度值域，避免浮点噪音
        // 掩盖真实回归（浮点 roundtrip 等价性由 wal/帧属性覆盖）。
        // EN: Restricted to a coarse range that roundtrips exactly through
        // f64→f32→JSON so float noise cannot mask real regressions (float roundtrip
        // equivalence is covered by the wal/frame properties).
        (-1024i32..1024).prop_map(|n| serde_json::Value::from(n as f64 * 0.5)),
    ]
}

fn arb_payload_map() -> impl Strategy<Value = serde_json::Map<String, serde_json::Value>> {
    prop::collection::vec(("[a-z]{1,8}", arb_json_scalar()), 0..4).prop_map(|pairs| {
        pairs
            .into_iter()
            .filter(|(k, _)| k != "label") // label 列由 label 字段渲染，避让覆盖语义
            .collect::<serde_json::Map<String, serde_json::Value>>()
    })
}

/// 创建选项生成器：覆盖默认、仅度量、完整 HNSW 三元组三种合法形态，
/// 让属性测试同时打新字段的 roundtrip 与「hnsw 必须带度量/仅 hnsw 后端」
/// 的编码约束（非法组合由下方单元测试的点测覆盖，不进 roundtrip 生成器）。
/// EN: Create-options generator covering the three legal shapes — default,
/// metric-only, and the full HNSW triple — so the property test exercises the
/// new fields' roundtrip plus the encoder's "HNSW requires a metric / only on
/// the hnsw backend" constraints. Invalid combinations stay in the focused
/// unit tests rather than the roundtrip generator.
fn arb_create_options() -> impl Strategy<Value = CollectionCreateOptions> {
    let metric = prop::sample::select(vec!["cosine", "l2", "ip", "l1"]);
    prop_oneof![
        Just(CollectionCreateOptions::default()),
        // Select 不实现 Copy：复用同一策略必须显式 clone。
        // EN: Select is not Copy, so reuse requires an explicit clone.
        metric.clone().prop_map(|m| CollectionCreateOptions {
            metric: Some(m.to_string()),
            ..Default::default()
        }),
        metric.prop_flat_map(|m| {
            (
                Just(m.to_string()),
                2usize..=1024,
                1usize..=65_536,
                1usize..=65_536,
            )
                .prop_map(|(metric, m, efc, efs)| CollectionCreateOptions {
                    metric: Some(metric),
                    hnsw_m: Some(m),
                    hnsw_ef_construction: Some(efc),
                    hnsw_ef_search: Some(efs),
                })
        }),
    ]
}

fn arb_collection_wal_record() -> impl Strategy<Value = CollectionWalRecord> {
    prop_oneof![
        (arb_collection_name(), 1usize..64, arb_create_options()).prop_map(
            |(name, dims, options)| CollectionWalRecord::Create {
                name,
                dims,
                index_type: "hnsw".to_string(),
                options,
            }
        ),
        (
            arb_collection_name(),
            prop::collection::vec(
                (
                    any::<u64>(),
                    prop::collection::vec(arb_finite_f32(), 4..=4),
                    prop::option::of("[a-z]{0,8}"),
                    prop::option::of(arb_payload_map()),
                ),
                0..6,
            ),
        )
            .prop_map(|(name, items)| CollectionWalRecord::UpsertBatch {
                name,
                dim: 4,
                items: items
                    .into_iter()
                    .map(|(id, values, label, payload)| CollectionWalItem {
                        id,
                        values,
                        label,
                        payload,
                    })
                    .collect(),
            }),
        (
            arb_collection_name(),
            prop::collection::vec(any::<u64>(), 0..8),
        )
            .prop_map(|(name, ids)| CollectionWalRecord::Delete { name, ids }),
        (arb_collection_name(), 0u8..=2).prop_map(|(name, code)| {
            CollectionWalRecord::SetHaLevel {
                name,
                // 合法码域 0..=2 与枚举一一对应（HaLevel::from_code 全射）。
                // EN: Legal code domain 0..=2 maps one-to-one to the enum
                // (HaLevel::from_code is surjective).
                ha_level: HaLevel::from_code(code).expect("0..=2 are all valid levels"),
            }
        }),
        arb_collection_name().prop_map(|name| CollectionWalRecord::Drop { name }),
    ]
}

/// 按记录的 op 分派到对应编码函数（roundtrip 属性的对偶编码入口）。
/// EN: Dispatches on the record's op to the matching encoder (the dual entry for
/// the roundtrip property).
fn encode_record_for(record: &CollectionWalRecord) -> Vec<u8> {
    match record {
        CollectionWalRecord::Create {
            name,
            dims,
            index_type,
            options,
        } => encode_collection_create_with_options(name, *dims, index_type, options).unwrap(),
        CollectionWalRecord::UpsertBatch { name, dim, items } => {
            encode_collection_upsert_batch(name, *dim, items).unwrap()
        }
        CollectionWalRecord::Delete { name, ids } => encode_collection_delete(name, ids).unwrap(),
        CollectionWalRecord::SetHaLevel { name, ha_level } => {
            encode_collection_set_ha_level(name, *ha_level).unwrap()
        }
        CollectionWalRecord::Drop { name } => encode_collection_drop(name).unwrap(),
    }
}

/// 供 ERROR kind 生成器使用：两个合法 kind 都在射程内（编译期锚定常量存在性）。
/// EN: Used by the ERROR kind generator: both legal kinds are in range (a
/// compile-time anchor that the constants exist).
#[allow(dead_code)]
const _: (u8, u8) = (ERROR_KIND_GENERIC, ERROR_KIND_FOLLOWER_AHEAD);
