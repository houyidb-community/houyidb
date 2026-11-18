#![no_main]
//! 目标 3：复制协议 v10 帧解码（HANDSHAKE / WELCOME / DATA / HEARTBEAT / ACK /
//! ERROR）。
//! EN: Target 3: replication protocol v10 frame decoding (HANDSHAKE / WELCOME / DATA /
//! EN: HEARTBEAT / ACK / ERROR).
//!
//! 断言的不变量 / Invariants:
//! EN: Invariants:
//! - 任意字节进任何**解码器**只返回 `Ok`/`Err`，不得 panic；未知 tag 由
//!   `frame_header_len` 先拒绝（`decode_frame_header` 的 `unreachable!` 因此
//!   不可达）。
//! EN: - Arbitrary bytes into any **decoder** return only `Ok`/`Err`, never panic;
//! EN:   unknown tags are rejected by `frame_header_len` first (so
//! EN:   `decode_frame_header`'s `unreachable!` is unreachable).
//! - 64MB payload 守卫不得绕过：DATA `payload_len > MAX_PAYLOAD_BYTES` 必须 Err
//!   （否则会话层会按长度预分配 → 远端可控 OOM）；`== MAX` 必须**接受**
//!   （防守卫被收紧成 `>=` 的回归）。
//! EN: - The 64MB payload guard must not be bypassable: DATA
//! EN:   `payload_len > MAX_PAYLOAD_BYTES` must be Err (otherwise the session layer
//! EN:   preallocates by length → remote-controlled OOM); `== MAX` must be
//! EN:   **accepted** (guards against a regression tightening the guard to `>=`).
//! - 1KB 错误消息守卫同理（`msg_len > MAX_ERROR_MSG_BYTES` 必须 Err）。
//! EN: - Same for the 1KB error message guard
//! EN:   (`msg_len > MAX_ERROR_MSG_BYTES` must be Err).
//! - ACK stage 门：stage ∉ {RECEIVED, DURABLE} 必须 Err。
//! EN: - ACK stage gate: stage ∉ {RECEIVED, DURABLE} must be Err.
//! - 无前置条件的编码器（DATA 头 / HEARTBEAT / HANDSHAKE / WELCOME）对任意字段
//!   值不得 panic，且「编码→解码」必须回读一致。
//! EN: - Encoders without preconditions (DATA header / HEARTBEAT / HANDSHAKE /
//! EN:   WELCOME) must not panic for any field values, and "encode→decode" must
//! EN:   round-trip.
//!
//! 关于 `encode_ack` / `encode_error` 的前置条件 / Preconditions: 这两个编码器
//! 文档化了「`stage` / `kind` 须为常量」并在调用侧 `debug_assert!` 守卫（刻意的
//! 「调用侧编程错误」防线，见 wal_replication.rs:200 / :217）。它们是**内部**接口
//! ——线上的任意字节只从解码侧进入，永远到不了编码器。故本目标对这两个编码器先
//! **归一**到合法常量再调用；「任意 stage/kind 字节」的解码面由下面手工拼帧的
//! 守卫断言覆盖（与 wal_replication.rs 既有测试同法，那里同样绕开编码器构造非法
//! stage 帧）。
//! EN: Preconditions of `encode_ack` / `encode_error`: both encoders document that
//! EN: `stage` / `kind` must be constants and guard with `debug_assert!` at the call
//! EN: site (a deliberate "caller programming error" defense, see
//! EN: wal_replication.rs:200 / :217). They are **internal** interfaces — arbitrary
//! EN: bytes from the wire only enter on the decode side and never reach the
//! EN: encoders. So this target first **normalizes** these two encoders to legal
//! EN: constants; the decode surface for "arbitrary stage/kind bytes" is covered by
//! EN: the manually assembled frame guard assertions below (same approach as the
//! EN: existing wal_replication.rs tests, which also bypass the encoders to build
//! EN: illegal stage frames).
//!
//! 结构感知 / Structure-aware: 用 `Unstructured` 把输入前若干字节解成结构化字段
//! （tag/stream/lsn/stage/kind/payload_len/msg_len），余下字节 `take_rest()` 作原始
//! 载荷——变异因此落在载荷语法上而非「长度字段自洽性」上；语料种子仍是可读的
//! 「18 字节头字段 + 真实帧字节」。
//! EN: Structure-aware: `Unstructured` decodes the leading bytes into structured
//! EN: fields (tag/stream/lsn/stage/kind/payload_len/msg_len), and `take_rest()`
//! EN: yields the raw payload; mutations therefore land on payload syntax rather
//! EN: than "length-field consistency"; corpus seeds remain readable
//! EN: "18-byte header fields + real frame bytes".

use arbitrary::Unstructured;
use houyi_persistence::wal_replication as proto;
use libfuzzer_sys::fuzz_target;

/// 结构化字段 + 原始载荷。刻意不用 `#[derive(Arbitrary)]`：derive 的 `Vec<u8>` 会
/// 先读长度前缀，手工语料种子（可读的「头字段 + 载荷」字节）无法映射成有意义输入；
/// `take_rest()` 让种子尾部原样成为载荷。
/// EN: Structured fields + raw payload. Deliberately not `#[derive(Arbitrary)]`: the
/// EN: derived `Vec<u8>` would read a length prefix first, so hand-written corpus seeds
/// EN: (readable "header fields + payload" bytes) could not map to meaningful inputs;
/// EN: `take_rest()` makes the seed tail the payload verbatim.
struct FrameSpec {
    tag: u8,
    stream: u8,
    lsn: u64,
    stage: u8,
    kind: u8,
    payload_len: u32,
    msg_len: u16,
    raw: Vec<u8>,
}

/// ACK stage 是否合法（解码门的判定基准）。
/// EN: Whether the ACK stage is legal (the baseline for the decode gate).
fn legal_stage(stage: u8) -> bool {
    stage == proto::ACK_STAGE_RECEIVED || stage == proto::ACK_STAGE_DURABLE
}

/// ERROR kind 是否合法（编码器前置条件的判定基准）。
/// EN: Whether the ERROR kind is legal (the baseline for the encoder precondition).
fn legal_kind(kind: u8) -> bool {
    proto::ERROR_KINDS.contains(&kind)
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let tag = u.arbitrary::<u8>().unwrap_or(proto::FRAME_DATA);
    let stream = u.arbitrary::<u8>().unwrap_or(0);
    let lsn = u.arbitrary::<u64>().unwrap_or(0);
    let stage = u.arbitrary::<u8>().unwrap_or(proto::ACK_STAGE_DURABLE);
    let kind = u.arbitrary::<u8>().unwrap_or(proto::ERROR_KIND_GENERIC);
    let payload_len = u.arbitrary::<u32>().unwrap_or(0);
    let msg_len = u.arbitrary::<u16>().unwrap_or(0);
    let spec = FrameSpec {
        tag,
        stream,
        lsn,
        stage,
        kind,
        payload_len,
        msg_len,
        raw: u.take_rest().to_vec(),
    };

    // ── 1) 原始字节路径（真实攻击面）────────────────────────────
    // EN: -- 1) Raw byte path (real attack surface) --
    let _ = proto::decode_frame_header(&spec.raw);
    if let Ok(prefix) = proto::decode_handshake_prefix(&spec.raw) {
        let end = proto::HANDSHAKE_PREFIX_LEN + usize::from(prefix.token_len);
        if spec.raw.len() >= end {
            let _ = proto::decode_handshake_token(&spec.raw[proto::HANDSHAKE_PREFIX_LEN..end]);
        }
    }
    let _ = proto::decode_welcome(&spec.raw);
    let _ = proto::decode_error_msg(&spec.raw);
    let _ = proto::frame_header_len(spec.tag);

    // ── 2) DATA：payload_len 守卫（手工拼头，解码器是唯一权威判定点）──
    // EN: -- 2) DATA: payload_len guard (manual header; the decoder is the only authority) --
    let mut data_hdr = [0u8; proto::DATA_HEADER_LEN];
    data_hdr[0] = proto::FRAME_DATA;
    data_hdr[1] = spec.stream;
    data_hdr[2..10].copy_from_slice(&spec.lsn.to_be_bytes());
    data_hdr[10..14].copy_from_slice(&spec.payload_len.to_be_bytes());
    match proto::decode_frame_header(&data_hdr) {
        Ok(proto::FrameHeader::Data {
            stream,
            lsn,
            payload_len,
        }) => {
            assert!(
                payload_len <= proto::MAX_PAYLOAD_BYTES,
                "DATA header accepted payload_len {payload_len} past the guard"
            );
            assert_eq!(
                (stream, lsn, payload_len),
                (spec.stream, spec.lsn, spec.payload_len),
                "DATA header fields must round-trip"
            );
        }
        Ok(other) => panic!("DATA header decoded as {other:?}"),
        Err(_) => assert!(
            spec.payload_len > proto::MAX_PAYLOAD_BYTES,
            "DATA header rejected payload_len {} within the guard",
            spec.payload_len
        ),
    }
    // 边界接受侧：== MAX 必须 Ok（守卫收紧成 >= 的回归会在这里炸）。
    // EN: Boundary-accept side: == MAX must be Ok (a regression tightening the guard to >=
    // EN: would blow up here).
    let mut max_hdr = data_hdr;
    max_hdr[10..14].copy_from_slice(&proto::MAX_PAYLOAD_BYTES.to_be_bytes());
    assert!(
        proto::decode_frame_header(&max_hdr).is_ok(),
        "payload_len == MAX_PAYLOAD_BYTES must be accepted"
    );
    // 无前置条件的编码器：任意字段值不得 panic，且与手工头逐字节一致。
    // EN: Encoders without preconditions: no panic for any field values, and byte-for-byte
    // EN: agreement with the manual header.
    assert_eq!(
        proto::encode_data_header(spec.stream, spec.lsn, spec.payload_len),
        data_hdr,
        "encode_data_header must agree with the manual DATA header"
    );

    // ── 3) ACK：stage 门只放行 RECEIVED/DURABLE ──────────────────
    // EN: -- 3) ACK: the stage gate only admits RECEIVED/DURABLE --
    let mut ack_hdr = [0u8; proto::ACK_LEN];
    ack_hdr[0] = proto::FRAME_ACK;
    ack_hdr[1] = spec.stream;
    ack_hdr[2] = spec.stage;
    ack_hdr[3..11].copy_from_slice(&spec.lsn.to_be_bytes());
    match proto::decode_frame_header(&ack_hdr) {
        Ok(proto::FrameHeader::Ack { stream, stage, lsn }) => {
            assert!(
                legal_stage(stage),
                "ACK stage gate accepted illegal stage {stage}"
            );
            assert_eq!(
                (stream, stage, lsn),
                (spec.stream, spec.stage, spec.lsn),
                "ACK fields must round-trip"
            );
        }
        Ok(other) => panic!("ACK frame decoded as {other:?}"),
        Err(_) => assert!(
            !legal_stage(spec.stage),
            "ACK stage gate rejected a legal stage {}",
            spec.stage
        ),
    }
    // 编码器侧（合法 stage）：往返一致。
    // EN: Encoder side (legal stage): round-trip consistent.
    let stage_enc = if legal_stage(spec.stage) {
        spec.stage
    } else {
        proto::ACK_STAGE_RECEIVED
    };
    let ack = proto::encode_ack(spec.stream, stage_enc, spec.lsn);
    match proto::decode_frame_header(&ack) {
        Ok(proto::FrameHeader::Ack { stage, lsn, .. }) => {
            assert_eq!(stage, stage_enc, "encoded ACK stage must round-trip");
            assert_eq!(lsn, spec.lsn, "encoded ACK lsn must round-trip");
        }
        other => panic!("encoded ACK decoded as {other:?}"),
    }

    // ── 4) ERROR：msg_len 1KB 守卫 + 编码器截断语义 ──────────────
    // EN: -- 4) ERROR: 1KB msg_len guard + encoder truncation semantics --
    let mut err_hdr = [0u8; 4];
    err_hdr[0] = proto::FRAME_ERROR;
    err_hdr[1] = spec.kind;
    err_hdr[2..4].copy_from_slice(&spec.msg_len.to_be_bytes());
    match proto::decode_frame_header(&err_hdr) {
        Ok(proto::FrameHeader::Error { kind, msg_len }) => {
            assert!(
                msg_len as usize <= proto::MAX_ERROR_MSG_BYTES,
                "ERROR header accepted msg_len {msg_len} past the guard"
            );
            assert_eq!(
                (kind, msg_len),
                (spec.kind, spec.msg_len),
                "ERROR header fields must round-trip"
            );
        }
        Ok(other) => panic!("ERROR frame decoded as {other:?}"),
        Err(_) => assert!(
            spec.msg_len as usize > proto::MAX_ERROR_MSG_BYTES,
            "ERROR header rejected msg_len {} within the guard",
            spec.msg_len
        ),
    }
    // 边界接受侧：== 1KB 必须 Ok。
    // EN: Boundary-accept side: == 1KB must be Ok.
    let mut max_err = err_hdr;
    max_err[2..4].copy_from_slice(&(proto::MAX_ERROR_MSG_BYTES as u16).to_be_bytes());
    assert!(
        proto::decode_frame_header(&max_err).is_ok(),
        "msg_len == MAX_ERROR_MSG_BYTES must be accepted"
    );
    // 编码器截断：超长消息必须被截到守卫内（定长用例，避免每轮 64KB 分配）。
    // EN: Encoder truncation: oversized messages must be truncated within the guard
    // EN: (fixed-size case to avoid a 64KB allocation every round).
    let kind_enc = if legal_kind(spec.kind) {
        spec.kind
    } else {
        proto::ERROR_KIND_GENERIC
    };
    let err = proto::encode_error(kind_enc, &"x".repeat(proto::MAX_ERROR_MSG_BYTES + 1));
    match proto::decode_frame_header(&err[..4]) {
        Ok(proto::FrameHeader::Error { kind, msg_len }) => {
            assert_eq!(kind, kind_enc);
            assert_eq!(msg_len as usize, proto::MAX_ERROR_MSG_BYTES);
        }
        other => panic!("encoded ERROR header decoded as {other:?}"),
    }

    // ── 5) 定长帧：编码→解码必须回读一致 ────────────────────────
    // EN: -- 5) Fixed-length frames: encode→decode must round-trip --
    let hb = proto::encode_heartbeat(spec.stream, spec.lsn);
    assert_eq!(
        proto::decode_frame_header(&hb).expect("heartbeat header must decode"),
        proto::FrameHeader::Heartbeat {
            stream: spec.stream,
            leader_next_lsn: spec.lsn,
        }
    );
    // HANDSHAKE 前缀在 v4/v6/v10 依次扩展（内容指纹、保留起点 + 快照
    // checkpoint、follower 身份）；编码→解码必须逐字段回读一致。
    // EN: The HANDSHAKE prefix grew across v4/v6/v10 (content fingerprint,
    // EN: retention start + snapshot checkpoint, follower id); encode→decode
    // EN: must round-trip every field.
    let follower_id = [0x5A_u8; 16];
    let log_start_lsn = spec.lsn.wrapping_add(1);
    let snapshot_checkpoint = spec.lsn.wrapping_add(2);
    let hs = proto::encode_handshake(
        spec.lsn,
        spec.payload_len,
        0xABCD_EF01,
        log_start_lsn,
        snapshot_checkpoint,
        follower_id,
        "tok",
    )
    .expect("handshake encode");
    let prefix = proto::decode_handshake_prefix(&hs).expect("handshake prefix must decode");
    assert_eq!(prefix.resume_lsn, spec.lsn);
    assert_eq!(prefix.stream_mask, spec.payload_len);
    assert_eq!(prefix.log_hash, 0xABCD_EF01);
    assert_eq!(prefix.log_start_lsn, log_start_lsn);
    assert_eq!(prefix.snapshot_checkpoint, snapshot_checkpoint);
    assert_eq!(prefix.follower_id, follower_id);
    assert_eq!(
        proto::decode_handshake_token(&hs[proto::HANDSHAKE_PREFIX_LEN..]).expect("token"),
        "tok"
    );
    let wl = proto::encode_welcome(spec.lsn, log_start_lsn);
    assert_eq!(
        proto::decode_welcome(&wl).expect("welcome must decode"),
        proto::Welcome {
            leader_next_lsn: spec.lsn,
            leader_first_available_lsn: log_start_lsn,
        }
    );
});
