//! 复制协议 v8 — 纯编解码层（无 IO），全大端字节序（对齐 WAL v2 文件格式）。
//! Replication protocol v8 — pure codec layer (no IO), big-endian throughout
//! (aligned with the WAL v2 on-disk format, eliminating the v1 LE/BE split).
//! v3 = v2 + ERROR 帧 kind 字节；v4 = v3 + HANDSHAKE 日志前缀指纹（P4.6.1）；
//! v5 = v4 + HANDSHAKE 认证 token（P4.6.3）；v6 = v5 + follower 保留基线与
//! 锚点协商（P5.0.4：`log_start_lsn` / `snapshot_checkpoint` / `follower_id`，
//! ANCHOR/HNDA，`BEHIND_RETAINED`，WELCOME 携带 leader first-available）；
//! v7 = v6 + `collection_drop`；v8 = v7 + `collection_create` 的 metric/HNSW
//! 参数语义（线格式不变，版本门只为阻止旧二进制静默按 Cosine 重建）。
//! v3 = v2 + ERROR kind byte; v4 = v3 + log-prefix fingerprint (P4.6.1);
//! v5 = v4 + shared-secret auth token in HANDSHAKE (P4.6.3); v6 = v5 +
//! follower retention baseline and anchor negotiation (P5.0.4);
//! v7 = v6 + `collection_drop`; v8 = v7 + metric/HNSW parameters carried by
//! `collection_create` records (wire format unchanged; the gate only stops old
//! binaries from silently rebuilding collections with Cosine).
//!
//! 帧布局 / Frame layouts:
//! ```text
//! Follower → Leader: HANDSHAKE "HNDK" + version:u32 + resume_lsn:u64 + stream_mask:u32 + log_hash:u64
//!                    + log_start_lsn:u64 + snapshot_checkpoint:u64 + follower_id[16] + token_len:u16 + token (62B+token)
//! Leader → Follower: ANCHOR    "ANCH" + version:u32 + anchor_lsn:u64                     (16B)
//! Follower → Leader: HNDA      "HNDA" + version:u32 + anchor_lsn:u64 + log_hash:u64      (24B)
//! Leader → Follower: WELCOME   "WLCM" + version:u32 + leader_next_lsn:u64
//!                    + leader_first_available_lsn:u64                                  (24B)
//! Leader → Follower: ERROR     0x0F + kind:u8 + len:u16 + msg                            (4B+len)
//! Leader → Follower: DATA      0x01 + stream:u8 + lsn:u64 + len:u32 + payload            (14B+len)
//! Leader → Follower: HEARTBEAT 0x02 + stream:u8 + leader_next_lsn:u64                    (10B)
//! Follower → Leader: ACK       0x03 + stream:u8 + stage:u8 + lsn:u64                     (11B)
//! ```
//!
//! `log_hash` = `range_hash(log_start_lsn, resume_lsn)`；v5 的整段前缀哈希是
//! `log_start_lsn = 0` 的特例。`log_start_lsn` 是 follower 本地最早可读 LSN，
//! leader 用它判断是否需要锚点协商（ANCHOR/HNDA）。
//! EN: `log_hash` is `range_hash(log_start_lsn, resume_lsn)`; the v5 whole-prefix
//! hash is the `log_start_lsn = 0` special case. `log_start_lsn` is the
//! follower's earliest readable LSN and drives ANCHOR/HNDA negotiation.
//!
//! 认证语义 / Auth semantics: token 是运维在 leader 与 follower 配置的共享
//! 秘密（明文传输——v1 定位是可信网络；TLS 是 v2 项）。双方都配置时必须
//! 常量时间比较相等；任一侧未配置时只接受空 token（fail-closed：给无认证
//! leader 配了 token 的 follower 会握手失败而不是“降级可用”）。
//! EN: the token is a shared secret configured on both sides (sent in clear —
//! v1 assumes a trusted network; TLS is a v2 item). When both sides configure
//! it, comparison is constant-time; when either side is unconfigured only the
//! empty token is accepted (fail-closed, never silently unauthenticated).
//!
//! 语义要点 / Semantics:
//! - `resume_lsn` = follower 本地 next_lsn（下一条期望的 LSN），消除空 WAL 下溢；
//!   重叠推送由 follower 幂等吸收（lsn < expected 跳过仍 ACK，= expected apply，
//!   > expected 视为空洞，断连重 catch-up）。
//! - ACK stage：1 = RECEIVED（帧解析完成），2 = DURABLE（fsync + apply 完成）。
//! - 版本不符即 fail-fast（ERROR 帧 / 解码错误），不做 v1 兼容——v1 骨架零生产引用。

use std::io;

/// 复制协议版本。v2 全大端，与 v1（小端、一次性快照推送）不兼容。
/// v3 = v2 + ERROR 帧加 kind 字节（3B→4B 头）：线格式变更必须 bump 版本——
/// 同号混布会过版本门再错解 ERROR 帧，把 follower_ahead（脑裂信号）静默
/// 降级为普通重连（OCR 批次十七）。
/// v4 = v3 + HANDSHAKE 增加 64 位日志前缀指纹：旧主等长日志重连不再被静默
/// 接受（P4.5.6 模型反例；P4.6.1 修复）。
/// v5 = v4 + HANDSHAKE 认证 token（P4.6.3）；v6 = v5 + 保留基线与锚点协商（P5.0.4）。
/// v7 = v6 + `collection_drop` 语义：线格式未变，但旧二进制实时 apply 会把未知
/// op 静默跳过，导致 follower 永久保留已删 collection（静默分叉），因此必须
/// 用版本门隔离 v6/v7 混布。跨 v7 需要全集群协调升级（版本门 fail-fast，
/// 不支持滚动混跑）；降级到 v6 二进制会跳过 drop 记录并复活 collection，
/// 禁止旧二进制重放 v7 WAL。
/// v8 = v7 + `collection_create` 信封新增 `metric` / `hnsw_*` 可选字段
/// （P5.1）：线格式未变，但 v7 二进制解码时忽略未知字段，follower 会按
/// 默认 Cosine 建集合——全是静默分叉，因此 v7/v8 同样禁止混跑（协调式升级）。
/// 降级限制：v7 二进制重放 v8 WAL 会把非 Cosine 集合建成 Cosine，禁止。
/// EN: v3 = v2 + kind byte in ERROR frames. v4 = v3 + log-prefix fingerprint
/// (P4.6.1). v5 = v4 + auth token (P4.6.3). v6 = v5 + retention baseline and
/// anchor negotiation (P5.0.4). v7 = v6 + `collection_drop` semantics: the wire
/// format is unchanged, but an old binary silently skips the unknown op during
/// live apply, so the follower would retain a dropped collection (silent
/// divergence). The version gate must therefore separate v6/v7; crossing v7
/// requires a coordinated full-cluster upgrade (fail-fast gate, no mixed
/// rolling deployment). Downgrading skips drop records and resurrects
/// collections — old binaries must never replay v7 WAL.
/// v8 = v7 + optional `metric`/`hnsw_*` fields in the `collection_create`
/// envelope (P5.1). The wire format is unchanged, but a v7 binary ignores the
/// unknown fields during decode and rebuilds the collection with the default
/// Cosine metric — silent divergence again, so v7/v8 must not be mixed either
/// (coordinated upgrade). A v7 binary replaying a v8 WAL would recreate
/// non-Cosine collections as Cosine — downgrade is forbidden.
pub const REPL_PROTOCOL_VERSION: u32 = 8;

/// 流标识：collections.wal（v1 唯一流）。
/// EN: Stream id: collections.wal (the only v1 stream).
pub const STREAM_COLLECTIONS: u8 = 0;
/// 流标识：主 WAL（benchmark 路径），v2 预留。
/// EN: Stream id: main WAL (benchmark path), reserved for v2.
pub const STREAM_MAIN_WAL: u8 = 1;

/// stream_mask 中 collections.wal 对应的位。
/// EN: Bit in stream_mask corresponding to collections.wal.
pub const STREAM_MASK_COLLECTIONS: u32 = 1 << STREAM_COLLECTIONS;

/// ACK 阶段：帧已解析接收（未落盘）。
/// EN: ACK stage: frame parsed and received (not yet durable).
pub const ACK_STAGE_RECEIVED: u8 = 1;
/// ACK 阶段：已 fsync 并 apply（HotBuffer 可查）。
/// EN: ACK stage: fsynced and applied (visible in HotBuffer).
pub const ACK_STAGE_DURABLE: u8 = 2;

/// ERROR 帧 kind：通用错误（诊断消息仅供日志）。
/// EN: ERROR frame kind: generic error (the message is for logs only).
pub const ERROR_KIND_GENERIC: u8 = 0;
/// ERROR 帧 kind：follower_ahead（脑裂信号）。结构化字段承载安全判定——
/// 靠匹配人类可读消息文本做分类，任一侧改文案都会让脑裂识别静默失效。
/// EN: Split-brain classification rides a structured kind byte, never the
/// human-readable message text.
pub const ERROR_KIND_FOLLOWER_AHEAD: u8 = 1;
/// ERROR 帧 kind：日志分叉（同 resume_lsn 但内容指纹不一致）。
///
/// 与 follower_ahead 一样是**人工处置**信号：不得自动重试收敛，必须清空
/// 落后节点数据目录后全量重同步。EN: manual-intervention signal, same class
/// as follower_ahead — the lagging node must be wiped and re-synced.
pub const ERROR_KIND_DIVERGED: u8 = 2;
/// ERROR 帧 kind：认证失败（token 缺失/不匹配）。
///
/// 配置错误类信号：重试无意义，follower 以长退避反复告警直到运维修正 token。
/// EN: configuration-error signal — retries cannot heal it; the follower logs
/// and backs off until the token is fixed.
pub const ERROR_KIND_AUTH: u8 = 3;
/// ERROR 帧 kind：follower 所需历史已被保留策略删除（P5.0.4）。
///
/// 与 diverged 同属人工处置：不能自动重试，必须按 runbook 做全量重同步；
/// follower 收到后长退避并告警，绝不自动清库。
/// EN: The follower needs history that retention already deleted (P5.0.4).
/// Manual intervention like diverged: long backoff and a runbook-driven full
/// resync; the follower never wipes itself automatically.
pub const ERROR_KIND_BEHIND_RETAINED: u8 = 4;

/// 全部合法 ERROR kind——编码侧校验的单一事实来源：新增 kind 只改这里，
/// 杜绝「常量加了、校验漏改」的静默放行（kind 承载脑裂分类，放行即降级）。
/// 切片而非定长数组：新增 kind 是字面意义的一处编辑（OCR dcace87[1]）。
/// EN: Single source of truth for valid ERROR kinds — adding a kind touches
/// only this slice, so the encoder guard can never drift out of sync.
pub const ERROR_KINDS: &[u8] = &[
    ERROR_KIND_GENERIC,
    ERROR_KIND_FOLLOWER_AHEAD,
    ERROR_KIND_DIVERGED,
    ERROR_KIND_AUTH,
    ERROR_KIND_BEHIND_RETAINED,
];

/// 单帧 payload 上限（防恶意长度撑爆内存，沿用 v1 守卫）。
/// EN: Per-frame payload cap (stops malicious lengths from exhausting memory;
/// carried over from the v1 guard).
pub const MAX_PAYLOAD_BYTES: u32 = 64 * 1024 * 1024;
/// ERROR 帧消息上限（编码侧截断，解码侧守卫）。
/// EN: ERROR frame message cap (encoder truncates, decoder guards).
pub const MAX_ERROR_MSG_BYTES: usize = 1024;
/// 认证 token 长度上限（字节）：共享秘密是短串；上限防恶意长度撑爆握手缓冲。
/// EN: auth token length cap — prevents a malicious handshake length.
pub const MAX_TOKEN_BYTES: usize = 256;

/// 配置通道的 token 校验（启动期 fail-fast，而不是每会话在编码时失败）。
/// 空白与非空白由调用方按配置纪律判断；这里只负责长度上限。
/// EN: startup validation for configured tokens — an oversized token must fail
/// at config load, not on every session's encode.
pub fn validate_auth_token(token: &str) -> Result<(), String> {
    if token.len() > MAX_TOKEN_BYTES {
        return Err(format!(
            "replication auth token is {} bytes, exceeding the {MAX_TOKEN_BYTES}-byte limit",
            token.len()
        ));
    }
    Ok(())
}

pub const FRAME_DATA: u8 = 0x01;
pub const FRAME_HEARTBEAT: u8 = 0x02;
pub const FRAME_ACK: u8 = 0x03;
pub const FRAME_ERROR: u8 = 0x0F;

const MAGIC_HANDSHAKE: &[u8; 4] = b"HNDK";
const MAGIC_WELCOME: &[u8; 4] = b"WLCM";
const MAGIC_ANCHOR: &[u8; 4] = b"ANCH";
const MAGIC_HANDSHAKE_ANCHORED: &[u8; 4] = b"HNDA";

/// HANDSHAKE 定长前缀（magic4 + version4 + resume_lsn8 + stream_mask4 +
/// log_hash8 + log_start_lsn8 + snapshot_checkpoint8 + follower_id16 +
/// token_len2）；token 紧随其后。
/// EN: fixed handshake prefix; the token follows.
pub const HANDSHAKE_PREFIX_LEN: usize = 62;
/// WELCOME 帧全长（magic4 + version4 + leader_next_lsn8 +
/// leader_first_available_lsn8）。
/// EN: WELCOME frame total length (magic4 + version4 + next8 + first_available8).
pub const WELCOME_LEN: usize = 24;
/// ANCHOR 帧全长（magic4 + version4 + anchor_lsn8）。
/// EN: ANCHOR frame total length (magic4 + version4 + anchor_lsn8).
pub const ANCHOR_LEN: usize = 16;
/// HANDSHAKE_ANCHORED 帧全长（magic4 + version4 + anchor_lsn8 + log_hash8）。
/// EN: HANDSHAKE_ANCHORED total length (magic4 + version4 + anchor8 + hash8).
pub const HANDSHAKE_ANCHORED_LEN: usize = 24;
/// DATA 帧头全长（tag1 + stream1 + lsn8 + len4），payload 紧随其后。
/// 头与 payload 分离编码以便 write_vectored 零拷贝推送。
/// EN: DATA header total length (tag1 + stream1 + lsn8 + len4); the payload follows
/// immediately. Header and payload are encoded separately for zero-copy
/// write_vectored pushes.
pub const DATA_HEADER_LEN: usize = 14;
/// HEARTBEAT 帧全长（tag1 + stream1 + leader_next_lsn8）。
/// EN: HEARTBEAT frame total length (tag1 + stream1 + leader_next_lsn8).
pub const HEARTBEAT_LEN: usize = 10;
/// ACK 帧全长（tag1 + stream1 + stage1 + lsn8）。
/// EN: ACK frame total length (tag1 + stream1 + stage1 + lsn8).
pub const ACK_LEN: usize = 11;

fn invalid_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// ── HANDSHAKE / WELCOME（握手 / 欢迎帧）──────────────────────────
// EN: HANDSHAKE / WELCOME section.

/// HANDSHAKE 定长前缀的解码结果。
/// Decoded fixed prefix of a HANDSHAKE frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandshakePrefix {
    pub resume_lsn: u64,
    pub stream_mask: u32,
    /// v6 范围指纹 `range_hash(log_start_lsn, resume_lsn)`。
    /// EN: v6 range fingerprint `range_hash(log_start_lsn, resume_lsn)`.
    pub log_hash: u64,
    /// follower 本地最早可读 LSN（无截断时为 0）。
    /// EN: Follower's earliest readable LSN (0 when history is untruncated).
    pub log_start_lsn: u64,
    /// follower 本地 manifest 的 checkpoint（无 manifest 为 0）。
    /// EN: Follower-local manifest checkpoint (0 when no manifest exists).
    pub snapshot_checkpoint: u64,
    /// 稳定 follower 身份（跨重连保留 retention grace）。
    /// EN: Stable follower identity (retention grace survives reconnects).
    pub follower_id: [u8; 16],
    /// 紧随前缀之后的 token 字节数 / token byte count following the prefix.
    /// EN: Number of token bytes following the fixed prefix.
    pub token_len: u16,
}

/// 编码 HANDSHAKE 帧（Follower → Leader）。
/// token 超限返回 Err——编码侧拒绝构造无法被对端合法解码的帧。
/// EN: rejects oversized tokens instead of emitting an undecodable frame.
#[allow(clippy::too_many_arguments)]
pub fn encode_handshake(
    resume_lsn: u64,
    stream_mask: u32,
    log_hash: u64,
    log_start_lsn: u64,
    snapshot_checkpoint: u64,
    follower_id: [u8; 16],
    token: &str,
) -> io::Result<Vec<u8>> {
    let token = token.as_bytes();
    if token.len() > MAX_TOKEN_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "replication token too long: {} > {MAX_TOKEN_BYTES}",
                token.len()
            ),
        ));
    }
    let mut buf = Vec::with_capacity(HANDSHAKE_PREFIX_LEN + token.len());
    buf.extend_from_slice(MAGIC_HANDSHAKE);
    buf.extend_from_slice(&REPL_PROTOCOL_VERSION.to_be_bytes());
    buf.extend_from_slice(&resume_lsn.to_be_bytes());
    buf.extend_from_slice(&stream_mask.to_be_bytes());
    buf.extend_from_slice(&log_hash.to_be_bytes());
    buf.extend_from_slice(&log_start_lsn.to_be_bytes());
    buf.extend_from_slice(&snapshot_checkpoint.to_be_bytes());
    buf.extend_from_slice(&follower_id);
    buf.extend_from_slice(&(token.len() as u16).to_be_bytes());
    buf.extend_from_slice(token);
    Ok(buf)
}

/// 解码 HANDSHAKE 定长前缀，返回字段与 token 长度（token 本体由调用方
/// 再读 `token_len` 字节）。magic / version 不符即 fail-fast；token 长度
/// 超上限也在这里拒绝（解码唯一权威判定点，调用方不得自行放宽）。
/// EN: decodes the fixed prefix; the caller then reads `token_len` bytes. Bad
/// magic/version and oversized tokens are rejected here (single authority).
pub fn decode_handshake_prefix(buf: &[u8]) -> io::Result<HandshakePrefix> {
    if buf.len() < HANDSHAKE_PREFIX_LEN {
        return Err(invalid_data(format!(
            "handshake prefix too short: {} < {HANDSHAKE_PREFIX_LEN}",
            buf.len()
        )));
    }
    if &buf[0..4] != MAGIC_HANDSHAKE {
        return Err(invalid_data("handshake bad magic"));
    }
    let version = u32::from_be_bytes(buf[4..8].try_into().unwrap());
    if version != REPL_PROTOCOL_VERSION {
        return Err(invalid_data(format!(
            "protocol version mismatch: got {version}, want {REPL_PROTOCOL_VERSION}"
        )));
    }
    let resume_lsn = u64::from_be_bytes(buf[8..16].try_into().unwrap());
    let stream_mask = u32::from_be_bytes(buf[16..20].try_into().unwrap());
    let log_hash = u64::from_be_bytes(buf[20..28].try_into().unwrap());
    let log_start_lsn = u64::from_be_bytes(buf[28..36].try_into().unwrap());
    let snapshot_checkpoint = u64::from_be_bytes(buf[36..44].try_into().unwrap());
    let mut follower_id = [0u8; 16];
    follower_id.copy_from_slice(&buf[44..60]);
    let token_len = u16::from_be_bytes(buf[60..62].try_into().unwrap());
    if usize::from(token_len) > MAX_TOKEN_BYTES {
        return Err(invalid_data(format!(
            "handshake token too long: {token_len} > {MAX_TOKEN_BYTES}"
        )));
    }
    Ok(HandshakePrefix {
        resume_lsn,
        stream_mask,
        log_hash,
        log_start_lsn,
        snapshot_checkpoint,
        follower_id,
        token_len,
    })
}

/// 解码 token 本体（UTF-8 无效即 Err）。
/// EN: decodes the token body; invalid UTF-8 is an error.
pub fn decode_handshake_token(buf: &[u8]) -> io::Result<String> {
    String::from_utf8(buf.to_vec()).map_err(|_| invalid_data("handshake token is not valid UTF-8"))
}

/// 认证比较（常量时间）：未配置 token 的对端只接受空 token。
///
/// 为什么恒定时间（WHY）：token 是低熵共享秘密，逐字节提前返回会泄露前缀
/// 长度信息；这里对两串逐字节 XOR 累积、长度差异并入比较结果，时间只与
/// 长度有关。网络抖动使其在实践中难以利用，但成本为零，没有理由不做。
/// EN: constant-time comparison; an unconfigured side accepts only the empty
/// token. Length differences fold into the result rather than early-returning.
pub fn tokens_match(expected: Option<&str>, presented: &str) -> bool {
    let expected = expected.unwrap_or("");
    let (a, b) = (expected.as_bytes(), presented.as_bytes());
    // 长度差用 usize 累积：u8 会截断 256 的整数倍（OCR 批次二十八
    // security：空 token vs 256 个 NUL 曾误判相等）。
    // EN: accumulate length deltas in usize — u8 truncated multiples of 256
    // (empty vs 256 NULs compared equal).
    let mut diff: usize = a.len() ^ b.len();
    let n = a.len().max(b.len());
    for i in 0..n {
        // 缺位补 0（长度已并入 diff，补齐不会掩盖不等）。
        // EN: Missing positions are zero-filled (length is folded into diff, so
        // padding cannot mask inequality).
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

/// WELCOME 的解码结果。
/// EN: Decoded WELCOME payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Welcome {
    pub leader_next_lsn: u64,
    pub leader_first_available_lsn: u64,
}

/// 编码 WELCOME 帧（Leader → Follower）。
/// EN: Encodes a WELCOME frame (Leader → Follower).
pub fn encode_welcome(leader_next_lsn: u64, leader_first_available_lsn: u64) -> [u8; WELCOME_LEN] {
    let mut buf = [0u8; WELCOME_LEN];
    buf[0..4].copy_from_slice(MAGIC_WELCOME);
    buf[4..8].copy_from_slice(&REPL_PROTOCOL_VERSION.to_be_bytes());
    buf[8..16].copy_from_slice(&leader_next_lsn.to_be_bytes());
    buf[16..24].copy_from_slice(&leader_first_available_lsn.to_be_bytes());
    buf
}

/// 解码 WELCOME 帧。magic / version 不符即 fail-fast。
/// EN: Decodes a WELCOME frame; a magic/version mismatch fails fast.
pub fn decode_welcome(buf: &[u8]) -> io::Result<Welcome> {
    if buf.len() < WELCOME_LEN {
        return Err(invalid_data(format!(
            "welcome too short: {} < {WELCOME_LEN}",
            buf.len()
        )));
    }
    if &buf[0..4] != MAGIC_WELCOME {
        return Err(invalid_data("welcome bad magic"));
    }
    let version = u32::from_be_bytes(buf[4..8].try_into().unwrap());
    if version != REPL_PROTOCOL_VERSION {
        return Err(invalid_data(format!(
            "protocol version mismatch: got {version}, want {REPL_PROTOCOL_VERSION}"
        )));
    }
    Ok(Welcome {
        leader_next_lsn: u64::from_be_bytes(buf[8..16].try_into().unwrap()),
        leader_first_available_lsn: u64::from_be_bytes(buf[16..24].try_into().unwrap()),
    })
}

/// 编码 ANCHOR 帧（Leader → Follower）：leader 要求 follower 以
/// `anchor_lsn` 为新基线重算范围指纹。
/// EN: Encodes an ANCHOR frame: the leader asks the follower to recompute its
/// range fingerprint from `anchor_lsn`.
pub fn encode_anchor(anchor_lsn: u64) -> [u8; ANCHOR_LEN] {
    let mut buf = [0u8; ANCHOR_LEN];
    buf[0..4].copy_from_slice(MAGIC_ANCHOR);
    buf[4..8].copy_from_slice(&REPL_PROTOCOL_VERSION.to_be_bytes());
    buf[8..16].copy_from_slice(&anchor_lsn.to_be_bytes());
    buf
}

/// 解码 ANCHOR 帧。
/// EN: Decodes an ANCHOR frame.
pub fn decode_anchor(buf: &[u8]) -> io::Result<u64> {
    if buf.len() < ANCHOR_LEN {
        return Err(invalid_data(format!(
            "anchor too short: {} < {ANCHOR_LEN}",
            buf.len()
        )));
    }
    if &buf[0..4] != MAGIC_ANCHOR {
        return Err(invalid_data("anchor bad magic"));
    }
    let version = u32::from_be_bytes(buf[4..8].try_into().unwrap());
    if version != REPL_PROTOCOL_VERSION {
        return Err(invalid_data(format!(
            "protocol version mismatch: got {version}, want {REPL_PROTOCOL_VERSION}"
        )));
    }
    Ok(u64::from_be_bytes(buf[8..16].try_into().unwrap()))
}

/// HANDSHAKE_ANCHORED 的解码结果。
/// EN: Decoded HANDSHAKE_ANCHORED payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandshakeAnchored {
    pub anchor_lsn: u64,
    pub log_hash: u64,
}

/// 编码 HANDSHAKE_ANCHORED（Follower → Leader）。
/// EN: Encodes HANDSHAKE_ANCHORED (Follower → Leader).
pub fn encode_handshake_anchored(anchor_lsn: u64, log_hash: u64) -> [u8; HANDSHAKE_ANCHORED_LEN] {
    let mut buf = [0u8; HANDSHAKE_ANCHORED_LEN];
    buf[0..4].copy_from_slice(MAGIC_HANDSHAKE_ANCHORED);
    buf[4..8].copy_from_slice(&REPL_PROTOCOL_VERSION.to_be_bytes());
    buf[8..16].copy_from_slice(&anchor_lsn.to_be_bytes());
    buf[16..24].copy_from_slice(&log_hash.to_be_bytes());
    buf
}

/// 解码 HANDSHAKE_ANCHORED。
/// EN: Decodes HANDSHAKE_ANCHORED.
pub fn decode_handshake_anchored(buf: &[u8]) -> io::Result<HandshakeAnchored> {
    if buf.len() < HANDSHAKE_ANCHORED_LEN {
        return Err(invalid_data(format!(
            "handshake_anchored too short: {} < {HANDSHAKE_ANCHORED_LEN}",
            buf.len()
        )));
    }
    if &buf[0..4] != MAGIC_HANDSHAKE_ANCHORED {
        return Err(invalid_data("handshake_anchored bad magic"));
    }
    let version = u32::from_be_bytes(buf[4..8].try_into().unwrap());
    if version != REPL_PROTOCOL_VERSION {
        return Err(invalid_data(format!(
            "protocol version mismatch: got {version}, want {REPL_PROTOCOL_VERSION}"
        )));
    }
    Ok(HandshakeAnchored {
        anchor_lsn: u64::from_be_bytes(buf[8..16].try_into().unwrap()),
        log_hash: u64::from_be_bytes(buf[16..24].try_into().unwrap()),
    })
}

// ── 流帧（DATA / HEARTBEAT / ACK / ERROR）────────────────────
// EN: Stream frames (DATA / HEARTBEAT / ACK / ERROR).

/// 编码 DATA 帧头；payload 由调用方以 write_vectored 紧随其后零拷贝推送。
/// EN: Encodes a DATA frame header; the caller pushes the payload right after with
/// a zero-copy write_vectored.
pub fn encode_data_header(stream: u8, lsn: u64, payload_len: u32) -> [u8; DATA_HEADER_LEN] {
    let mut buf = [0u8; DATA_HEADER_LEN];
    buf[0] = FRAME_DATA;
    buf[1] = stream;
    buf[2..10].copy_from_slice(&lsn.to_be_bytes());
    buf[10..14].copy_from_slice(&payload_len.to_be_bytes());
    buf
}

/// 编码 HEARTBEAT 帧（Leader → Follower，携带 leader_next_lsn 供滞后计算）。
/// EN: Encodes a HEARTBEAT frame (Leader → Follower) carrying leader_next_lsn for
/// lag computation.
pub fn encode_heartbeat(stream: u8, leader_next_lsn: u64) -> [u8; HEARTBEAT_LEN] {
    let mut buf = [0u8; HEARTBEAT_LEN];
    buf[0] = FRAME_HEARTBEAT;
    buf[1] = stream;
    buf[2..10].copy_from_slice(&leader_next_lsn.to_be_bytes());
    buf
}

/// 未知 ACK stage 归一为 RECEIVED（release 强制，OCR 6bfb210[3]）：
/// 宁可少报 durable，不可虚报。独立成纯函数以便 debug 构建下可测
/// （debug_assert 在调用侧保 dev 期响亮）。
/// EN: Unknown ACK stages normalize to RECEIVED (enforced in release, OCR
/// 6bfb210[3]): under-report durable rather than over-report. Kept as a pure
/// function so it stays testable in debug builds (the caller's debug_assert
/// keeps dev builds loud).
fn normalize_ack_stage(stage: u8) -> u8 {
    if stage == ACK_STAGE_RECEIVED || stage == ACK_STAGE_DURABLE {
        stage
    } else {
        tracing::error!(stage, "unknown ACK stage coerced to RECEIVED");
        ACK_STAGE_RECEIVED
    }
}

/// 未知 ERROR kind 归一为 GENERIC（release 强制，OCR dcace87[1]）：帧仍
/// 合法、不 panic，分类降级但留下告警痕迹。独立纯函数同理可测（6bfb210[8]）。
/// EN: Unknown ERROR kinds normalize to GENERIC (enforced in release, OCR
/// dcace87[1]): the frame stays valid and never panics; classification degrades
/// but an alert is logged. Pure function for the same testability (6bfb210[8]).
fn normalize_error_kind(kind: u8) -> u8 {
    if ERROR_KINDS.contains(&kind) {
        kind
    } else {
        tracing::error!(kind, "unknown ERROR kind coerced to GENERIC");
        ERROR_KIND_GENERIC
    }
}

/// 编码 ACK 帧（Follower → Leader）。`stage` 须为 ACK_STAGE_* 常量。
/// EN: Encodes an ACK frame (Follower → Leader). `stage` must be an ACK_STAGE_*
/// constant.
pub fn encode_ack(stream: u8, stage: u8, lsn: u64) -> [u8; ACK_LEN] {
    debug_assert!(stage == ACK_STAGE_RECEIVED || stage == ACK_STAGE_DURABLE);
    let stage = normalize_ack_stage(stage);
    let mut buf = [0u8; ACK_LEN];
    buf[0] = FRAME_ACK;
    buf[1] = stream;
    buf[2] = stage;
    buf[3..11].copy_from_slice(&lsn.to_be_bytes());
    buf
}

/// 编码 ERROR 帧（Leader → Follower）。`kind` 须为 ERROR_KIND_* 常量。
/// 超长消息按上限截断（按 char 边界安全截断）。
/// EN: Encodes an ERROR frame (Leader → Follower). `kind` must be an ERROR_KIND_*
/// constant; oversized messages are truncated at the cap (on a char boundary).
pub fn encode_error(kind: u8, msg: &str) -> Vec<u8> {
    // 与 encode_ack 的 stage 断言对称：kind 字节承载脑裂分类，非法值会让
    // follower_ahead 静默降级为通用重连（OCR 批次十七）——编码侧尽早炸。
    // release 同样强制（OCR dcace87[1]）：debug_assert-only 会在 release
    // 下放行未知 kind，脑裂分类静默退化。
    // EN: Symmetric with encode_ack's stage assertion: the kind byte carries
    // split-brain classification, and an invalid value would silently degrade
    // follower_ahead into a generic reconnect (OCR batch 17) — fail early at the
    // encoder. Also enforced in release (OCR dcace87[1]): debug_assert-only would
    // let unknown kinds through in release, silently degrading split-brain detection.
    debug_assert!(ERROR_KINDS.contains(&kind), "unknown ERROR kind {kind}");
    let kind = normalize_error_kind(kind);
    let mut m = msg;
    if m.len() > MAX_ERROR_MSG_BYTES {
        let mut end = MAX_ERROR_MSG_BYTES;
        while !m.is_char_boundary(end) {
            end -= 1;
        }
        m = &m[..end];
    }
    let mut buf = Vec::with_capacity(4 + m.len());
    buf.push(FRAME_ERROR);
    buf.push(kind);
    buf.extend_from_slice(&(m.len() as u16).to_be_bytes());
    buf.extend_from_slice(m.as_bytes());
    buf
}

/// 各帧 tag 对应的固定头长度（**含** tag 本身，**不含**变长 payload/消息）。
/// DATA → 14（另有 payload），ERROR → 4（另有消息），其余即全长。
/// 未知 tag 返回 Err——会话层应断连（协议对端不可信）。
/// EN: Fixed header length per frame tag (**including** the tag, **excluding** the
/// variable payload/message). DATA → 14 (plus payload), ERROR → 4 (plus message);
/// the rest are total lengths. Unknown tags return Err — the session layer should
/// disconnect (an untrusted protocol peer).
pub fn frame_header_len(tag: u8) -> io::Result<usize> {
    match tag {
        FRAME_DATA => Ok(DATA_HEADER_LEN),
        FRAME_HEARTBEAT => Ok(HEARTBEAT_LEN),
        FRAME_ACK => Ok(ACK_LEN),
        FRAME_ERROR => Ok(4),
        _ => Err(invalid_data(format!("unknown frame tag: 0x{tag:02x}"))),
    }
}

/// 解码后的帧头（变长部分由会话层按长度继续读取）。
/// EN: Decoded frame header (the session layer reads the variable part by length).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameHeader {
    Data {
        stream: u8,
        lsn: u64,
        payload_len: u32,
    },
    Heartbeat {
        stream: u8,
        leader_next_lsn: u64,
    },
    Ack {
        stream: u8,
        stage: u8,
        lsn: u64,
    },
    Error {
        kind: u8,
        msg_len: u16,
    },
}

/// 解码帧头。`buf` 为完整帧头字节（含 tag，长度 = `frame_header_len(tag)`）。
/// 校验：DATA payload_len ≤ 64MB；ACK stage ∈ {1,2}；ERROR msg_len ≤ 1024。
/// EN: Decodes a frame header. `buf` holds the complete header (including tag,
/// length = `frame_header_len(tag)`). Checks: DATA payload_len <= 64MB; ACK stage
/// in {1,2}; ERROR msg_len <= 1024.
pub fn decode_frame_header(buf: &[u8]) -> io::Result<FrameHeader> {
    if buf.is_empty() {
        return Err(invalid_data("empty frame header"));
    }
    let tag = buf[0];
    let want = frame_header_len(tag)?;
    if buf.len() < want {
        return Err(invalid_data(format!(
            "frame header too short: {} < {want} (tag 0x{tag:02x})",
            buf.len()
        )));
    }
    match tag {
        FRAME_DATA => {
            let stream = buf[1];
            let lsn = u64::from_be_bytes(buf[2..10].try_into().unwrap());
            let payload_len = u32::from_be_bytes(buf[10..14].try_into().unwrap());
            if payload_len > MAX_PAYLOAD_BYTES {
                return Err(invalid_data(format!(
                    "payload too large: {payload_len} > {MAX_PAYLOAD_BYTES}"
                )));
            }
            Ok(FrameHeader::Data {
                stream,
                lsn,
                payload_len,
            })
        }
        FRAME_HEARTBEAT => {
            let stream = buf[1];
            let leader_next_lsn = u64::from_be_bytes(buf[2..10].try_into().unwrap());
            Ok(FrameHeader::Heartbeat {
                stream,
                leader_next_lsn,
            })
        }
        FRAME_ACK => {
            let stream = buf[1];
            let stage = buf[2];
            if stage != ACK_STAGE_RECEIVED && stage != ACK_STAGE_DURABLE {
                return Err(invalid_data(format!("invalid ack stage: {stage}")));
            }
            let lsn = u64::from_be_bytes(buf[3..11].try_into().unwrap());
            Ok(FrameHeader::Ack { stream, stage, lsn })
        }
        FRAME_ERROR => {
            let kind = buf[1];
            let msg_len = u16::from_be_bytes(buf[2..4].try_into().unwrap());
            if msg_len as usize > MAX_ERROR_MSG_BYTES {
                return Err(invalid_data(format!(
                    "error message too large: {msg_len} > {MAX_ERROR_MSG_BYTES}"
                )));
            }
            Ok(FrameHeader::Error { kind, msg_len })
        }
        _ => unreachable!("frame_header_len already rejected unknown tag"),
    }
}

/// 解码 ERROR 帧消息体（诊断用途，UTF-8 有损解码不 fail）。
/// EN: Decodes an ERROR frame message body (diagnostics only; lossy UTF-8 never fails).
pub fn decode_error_msg(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── HANDSHAKE ─────────────────────────────────────────────

    const FID: [u8; 16] = [
        0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE,
        0xAF,
    ];

    fn decode_full(buf: &[u8]) -> io::Result<(HandshakePrefix, String)> {
        let prefix = decode_handshake_prefix(&buf[..HANDSHAKE_PREFIX_LEN])?;
        let start = HANDSHAKE_PREFIX_LEN;
        let end = start + usize::from(prefix.token_len);
        if buf.len() < end {
            return Err(invalid_data("handshake token truncated"));
        }
        Ok((prefix, decode_handshake_token(&buf[start..end])?))
    }

    #[test]
    fn handshake_roundtrip() {
        let buf = encode_handshake(
            0x0102_0304_0506_0708,
            0x0A0B_0C0D,
            0x1122_3344_5566_7788,
            0x2122_3344_5566_7788,
            0x3132_3344_5566_7788,
            FID,
            "s3cret",
        )
        .unwrap();
        let (prefix, token) = decode_full(&buf).unwrap();
        assert_eq!(prefix.resume_lsn, 0x0102_0304_0506_0708);
        assert_eq!(prefix.stream_mask, 0x0A0B_0C0D);
        assert_eq!(prefix.log_hash, 0x1122_3344_5566_7788);
        assert_eq!(prefix.log_start_lsn, 0x2122_3344_5566_7788);
        assert_eq!(prefix.snapshot_checkpoint, 0x3132_3344_5566_7788);
        assert_eq!(prefix.follower_id, FID);
        assert_eq!(prefix.token_len, 6);
        assert_eq!(token, "s3cret");
    }

    #[test]
    fn handshake_with_empty_token_roundtrips() {
        let buf = encode_handshake(0, 1, 0, 0, 0, FID, "").unwrap();
        assert_eq!(buf.len(), HANDSHAKE_PREFIX_LEN);
        let (prefix, token) = decode_full(&buf).unwrap();
        assert_eq!(prefix.token_len, 0);
        assert!(token.is_empty());
    }

    #[test]
    fn handshake_bytes_are_big_endian() {
        // 钉死线格式：BE + 字段偏移。防回归 v1 的 LE/BE 分裂。
        // EN: Pin the wire format: big-endian and field offsets. Guards against the
        // v1 LE/BE split regression.
        let buf = encode_handshake(
            0x0102_0304_0506_0708,
            0x0A0B_0C0D,
            0x1122_3344_5566_7788,
            0x2122_3344_5566_7788,
            0x3132_3344_5566_7788,
            FID,
            "ab",
        )
        .unwrap();
        assert_eq!(&buf[0..4], b"HNDK");
        assert_eq!(&buf[4..8], &REPL_PROTOCOL_VERSION.to_be_bytes());
        assert_eq!(
            &buf[8..16],
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
        assert_eq!(&buf[16..20], &[0x0A, 0x0B, 0x0C, 0x0D]);
        assert_eq!(
            &buf[20..28],
            &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
        );
        assert_eq!(
            &buf[28..36],
            &[0x21, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
        );
        assert_eq!(
            &buf[36..44],
            &[0x31, 0x32, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
        );
        assert_eq!(&buf[44..60], &FID);
        assert_eq!(&buf[60..62], &[0x00, 0x02]);
        assert_eq!(&buf[62..], b"ab");
        assert_eq!(buf.len(), HANDSHAKE_PREFIX_LEN + 2);
    }

    #[test]
    fn handshake_rejects_bad_magic() {
        let mut buf = encode_handshake(0, 1, 0, 0, 0, FID, "").unwrap();
        buf[0] = b'X';
        assert!(decode_handshake_prefix(&buf).is_err());
    }

    #[test]
    fn handshake_rejects_version_mismatch() {
        let mut buf = encode_handshake(0, 1, 0, 0, 0, FID, "").unwrap();
        buf[4..8].copy_from_slice(&1u32.to_be_bytes()); // v1
        let err = decode_handshake_prefix(&buf).unwrap_err();
        assert!(err.to_string().contains("version mismatch"));
    }

    #[test]
    fn handshake_rejects_truncated_prefix() {
        let buf = encode_handshake(0, 1, 0, 0, 0, FID, "").unwrap();
        assert!(decode_handshake_prefix(&buf[..HANDSHAKE_PREFIX_LEN - 1]).is_err());
        assert!(decode_handshake_prefix(&[]).is_err());
    }

    #[test]
    fn handshake_rejects_oversized_token_both_directions() {
        let too_long = "x".repeat(MAX_TOKEN_BYTES + 1);
        assert!(encode_handshake(0, 1, 0, 0, 0, FID, &too_long).is_err());

        // 解码侧是唯一权威：手工把 token_len 改成超限值也必须拒绝。
        // EN: The decoder is the sole authority: a hand-crafted oversized token_len
        // must also be rejected.
        let mut buf = encode_handshake(0, 1, 0, 0, 0, FID, "").unwrap();
        buf[60..62].copy_from_slice(&((MAX_TOKEN_BYTES + 1) as u16).to_be_bytes());
        let err = decode_handshake_prefix(&buf).unwrap_err();
        assert!(err.to_string().contains("token too long"), "got: {err}");
    }

    #[test]
    fn token_match_semantics() {
        // 未配置只接受空串；配置后必须精确相等。
        // EN: Unconfigured accepts only the empty string; once configured, an exact
        // match is required.
        assert!(tokens_match(None, ""));
        assert!(!tokens_match(None, "x"));
        assert!(tokens_match(Some("abc"), "abc"));
        assert!(!tokens_match(Some("abc"), "abd"));
        assert!(!tokens_match(Some("abc"), "ab"));
        assert!(!tokens_match(Some("abc"), "abcd"));
        assert!(tokens_match(Some(""), ""));
        // 回归（OCR 批次二十八）：长度差 256 的整数倍曾被 u8 截断为 0。
        // EN: Regression (OCR batch 28): length differences in multiples of 256 were
        // once truncated by u8 to 0.
        let nuls = "\u{0}".repeat(MAX_TOKEN_BYTES);
        assert!(!tokens_match(None, &nuls));
        assert!(!tokens_match(Some("tok"), &format!("tok{nuls}")));
    }

    // ── ANCHOR / HANDSHAKE_ANCHORED（锚点协商）─────────────────
    // EN: ANCHOR / HANDSHAKE_ANCHORED section.

    #[test]
    fn anchor_roundtrip_and_bytes() {
        let buf = encode_anchor(0x0102_0304_0506_0708);
        assert_eq!(&buf[0..4], b"ANCH");
        assert_eq!(&buf[4..8], &REPL_PROTOCOL_VERSION.to_be_bytes());
        assert_eq!(
            &buf[8..16],
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
        assert_eq!(decode_anchor(&buf).unwrap(), 0x0102_0304_0506_0708);
        assert_eq!(buf.len(), ANCHOR_LEN);
    }

    #[test]
    fn anchor_rejects_bad_magic_version_truncated() {
        let mut buf = encode_anchor(0);
        buf[0] = b'X';
        assert!(decode_anchor(&buf).is_err());
        let mut buf = encode_anchor(0);
        buf[4..8].copy_from_slice(&1u32.to_be_bytes());
        assert!(decode_anchor(&buf).is_err());
        let buf = encode_anchor(0);
        assert!(decode_anchor(&buf[..ANCHOR_LEN - 1]).is_err());
    }

    #[test]
    fn handshake_anchored_roundtrip_and_bytes() {
        let buf = encode_handshake_anchored(0x0102_0304_0506_0708, 0x1122_3344_5566_7788);
        assert_eq!(&buf[0..4], b"HNDA");
        assert_eq!(&buf[4..8], &REPL_PROTOCOL_VERSION.to_be_bytes());
        assert_eq!(
            &buf[8..16],
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
        assert_eq!(
            &buf[16..24],
            &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
        );
        let got = decode_handshake_anchored(&buf).unwrap();
        assert_eq!(got.anchor_lsn, 0x0102_0304_0506_0708);
        assert_eq!(got.log_hash, 0x1122_3344_5566_7788);
        assert_eq!(buf.len(), HANDSHAKE_ANCHORED_LEN);
    }

    #[test]
    fn handshake_anchored_rejects_bad_magic_version_truncated() {
        let mut buf = encode_handshake_anchored(0, 0);
        buf[0] = b'X';
        assert!(decode_handshake_anchored(&buf).is_err());
        let mut buf = encode_handshake_anchored(0, 0);
        buf[4..8].copy_from_slice(&1u32.to_be_bytes());
        assert!(decode_handshake_anchored(&buf).is_err());
        let buf = encode_handshake_anchored(0, 0);
        assert!(decode_handshake_anchored(&buf[..HANDSHAKE_ANCHORED_LEN - 1]).is_err());
    }

    // ── WELCOME ───────────────────────────────────────────────

    #[test]
    fn welcome_roundtrip() {
        let buf = encode_welcome(0x0102_0304_0506_0708, 0x1112_1314_1516_1718);
        assert_eq!(
            decode_welcome(&buf).unwrap(),
            Welcome {
                leader_next_lsn: 0x0102_0304_0506_0708,
                leader_first_available_lsn: 0x1112_1314_1516_1718,
            }
        );
    }

    #[test]
    fn welcome_bytes_are_big_endian() {
        let buf = encode_welcome(0x0102_0304_0506_0708, 0x1112_1314_1516_1718);
        assert_eq!(&buf[0..4], b"WLCM");
        assert_eq!(&buf[4..8], &REPL_PROTOCOL_VERSION.to_be_bytes());
        assert_eq!(
            &buf[8..16],
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
        assert_eq!(
            &buf[16..24],
            &[0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]
        );
        assert_eq!(buf.len(), WELCOME_LEN);
    }

    #[test]
    fn welcome_rejects_bad_magic_version_truncated() {
        let mut buf = encode_welcome(0, 0);
        buf[0] = b'X';
        assert!(decode_welcome(&buf).is_err());

        let mut buf = encode_welcome(0, 0);
        buf[4..8].copy_from_slice(&99u32.to_be_bytes());
        assert!(decode_welcome(&buf).is_err());

        let buf = encode_welcome(0, 0);
        assert!(decode_welcome(&buf[..WELCOME_LEN - 1]).is_err());
    }

    // ── DATA ──────────────────────────────────────────────────

    #[test]
    fn data_header_roundtrip() {
        let buf = encode_data_header(STREAM_MAIN_WAL, 0x0102_0304_0506_0708, 12345);
        let h = decode_frame_header(&buf).unwrap();
        assert_eq!(
            h,
            FrameHeader::Data {
                stream: STREAM_MAIN_WAL,
                lsn: 0x0102_0304_0506_0708,
                payload_len: 12345,
            }
        );
    }

    #[test]
    fn data_header_bytes_are_big_endian() {
        let buf = encode_data_header(0x01, 0x0102_0304_0506_0708, 0x0A0B_0C0D);
        let want: &[u8] = &[
            0x01, 0x01, // tag, stream
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // lsn BE
            0x0A, 0x0B, 0x0C, 0x0D, // len BE
        ];
        assert_eq!(&buf[..], want);
        assert_eq!(buf.len(), DATA_HEADER_LEN);
    }

    #[test]
    fn data_header_rejects_oversize_payload() {
        let buf = encode_data_header(0, 0, MAX_PAYLOAD_BYTES + 1);
        let err = decode_frame_header(&buf).unwrap_err();
        assert!(err.to_string().contains("payload too large"));
        // 边界：恰好 64MB 允许
        // EN: Boundary: exactly 64MB is allowed.
        let buf = encode_data_header(0, 0, MAX_PAYLOAD_BYTES);
        assert!(decode_frame_header(&buf).is_ok());
    }

    #[test]
    fn data_header_rejects_truncated() {
        let buf = encode_data_header(0, 0, 0);
        assert!(decode_frame_header(&buf[..DATA_HEADER_LEN - 1]).is_err());
    }

    // ── HEARTBEAT ─────────────────────────────────────────────

    #[test]
    fn heartbeat_roundtrip_and_bytes() {
        let buf = encode_heartbeat(0x01, 0x0102_0304_0506_0708);
        let want: &[u8] = &[0x02, 0x01, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(&buf[..], want);
        assert_eq!(
            decode_frame_header(&buf).unwrap(),
            FrameHeader::Heartbeat {
                stream: 0x01,
                leader_next_lsn: 0x0102_0304_0506_0708,
            }
        );
    }

    // ── ACK ───────────────────────────────────────────────────

    #[test]
    fn ack_roundtrip_and_bytes() {
        let buf = encode_ack(0x01, ACK_STAGE_DURABLE, 0x0102_0304_0506_0708);
        let want: &[u8] = &[
            0x03, 0x01, 0x02, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        ];
        assert_eq!(&buf[..], want);
        assert_eq!(
            decode_frame_header(&buf).unwrap(),
            FrameHeader::Ack {
                stream: 0x01,
                stage: ACK_STAGE_DURABLE,
                lsn: 0x0102_0304_0506_0708,
            }
        );
    }

    #[test]
    fn ack_rejects_invalid_stage() {
        for bad in [0u8, 3, 255] {
            let buf = encode_ack_raw_stage(bad);
            let err = decode_frame_header(&buf).unwrap_err();
            assert!(err.to_string().contains("invalid ack stage"), "stage={bad}");
        }
    }

    /// 绕过 encode_ack 的 debug_assert + 归一化构造非法 stage 帧（解码侧
    /// fail-fast 测试需要真正的非法字节）。
    /// EN: Bypasses encode_ack's debug_assert + normalization to build an
    /// invalid-stage frame (the decoder fail-fast test needs truly invalid bytes).
    fn encode_ack_raw_stage(stage: u8) -> [u8; ACK_LEN] {
        let mut buf = [0u8; ACK_LEN];
        buf[0] = FRAME_ACK;
        buf[2] = stage;
        buf
    }

    #[test]
    fn normalize_helpers_coerce_unknown_codes() {
        // 归一化分支的 debug 构建可测性（OCR 6bfb210[8]）：assert 留在调用
        // 侧，归一化是纯函数——分支被反转/删除时本测试即红。
        // EN: Debug-build testability of the normalization branches (OCR 6bfb210[8]):
        // the assert stays at the caller while normalization is a pure function —
        // reversing/removing a branch turns this test red.
        assert_eq!(normalize_ack_stage(9), ACK_STAGE_RECEIVED);
        assert_eq!(normalize_ack_stage(ACK_STAGE_DURABLE), ACK_STAGE_DURABLE);
        assert_eq!(normalize_error_kind(9), ERROR_KIND_GENERIC);
        assert_eq!(
            normalize_error_kind(ERROR_KIND_FOLLOWER_AHEAD),
            ERROR_KIND_FOLLOWER_AHEAD
        );
    }

    // ── ERROR ─────────────────────────────────────────────────

    #[test]
    fn error_roundtrip_and_bytes() {
        let buf = encode_error(ERROR_KIND_FOLLOWER_AHEAD, "bad");
        let want: &[u8] = &[0x0F, 0x01, 0x00, 0x03, b'b', b'a', b'd'];
        assert_eq!(&buf[..], want);
        assert_eq!(
            decode_frame_header(&buf[..4]).unwrap(),
            FrameHeader::Error {
                kind: ERROR_KIND_FOLLOWER_AHEAD,
                msg_len: 3
            }
        );
        assert_eq!(decode_error_msg(&buf[4..]), "bad");
    }

    #[test]
    fn error_truncates_oversize_message() {
        let long = "x".repeat(MAX_ERROR_MSG_BYTES + 500);
        let buf = encode_error(ERROR_KIND_GENERIC, &long);
        assert_eq!(buf.len(), 4 + MAX_ERROR_MSG_BYTES);
        // 多字节 UTF-8 截断在 char 边界
        // EN: Multi-byte UTF-8 truncation happens on a char boundary.
        let multi = "汉".repeat(MAX_ERROR_MSG_BYTES); // 每字 3B
        let buf = encode_error(ERROR_KIND_GENERIC, &multi);
        let msg_len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
        assert!(msg_len <= MAX_ERROR_MSG_BYTES);
        assert_eq!(msg_len % 3, 0, "must cut at char boundary");
        assert!(std::str::from_utf8(&buf[4..]).is_ok());
    }

    #[test]
    fn error_header_rejects_oversize_len() {
        let buf = [FRAME_ERROR, ERROR_KIND_GENERIC, 0xFF, 0xFF]; // 65535 > 1024
        assert!(decode_frame_header(&buf).is_err());
    }

    // ── 通用 ──────────────────────────────────────────────────
    // EN: General section.

    #[test]
    fn frame_header_len_table() {
        assert_eq!(frame_header_len(FRAME_DATA).unwrap(), DATA_HEADER_LEN);
        assert_eq!(frame_header_len(FRAME_HEARTBEAT).unwrap(), HEARTBEAT_LEN);
        assert_eq!(frame_header_len(FRAME_ACK).unwrap(), ACK_LEN);
        assert_eq!(frame_header_len(FRAME_ERROR).unwrap(), 4);
        assert!(frame_header_len(0x00).is_err());
        assert!(frame_header_len(0x7F).is_err());
    }

    #[test]
    fn decode_rejects_empty_and_unknown() {
        assert!(decode_frame_header(&[]).is_err());
        assert!(decode_frame_header(&[0x77; 16]).is_err());
    }

    #[test]
    fn protocol_version_is_pinned() {
        // 钉死版本：线格式任何变更必须 bump。
        // v3 = v2 + ERROR kind 字节；v4 = v3 + 日志前缀指纹；v5 = v4 + 认证 token；
        // v6 = v5 + 保留基线与锚点协商（P5.0.4）；
        // v7 = v6 + `collection_drop` 语义（线格式不变，版本门只为阻止旧二进制
        //      静默跳过 drop 导致 follower 保留已删集合）；
        // v8 = v7 + `collection_create` 的 metric/HNSW 参数（版本门阻止旧二进制
        //      静默按 Cosine 重建，v7/v8 同样禁止混跑）。
        // 版本号变动必须显式改测试；协议不做跨版本兼容。
        // EN: Pin the version: any wire-format change must bump it.
        // v3 = v2 + ERROR kind byte; v4 = v3 + log-prefix fingerprint;
        // v5 = v4 + auth token; v6 = v5 + retention baseline/anchor negotiation;
        // v7 = v6 + `collection_drop` semantics (wire format unchanged; the gate only
        // blocks old binaries from silently skipping drop and retaining the collection);
        // v8 = v7 + metric/HNSW params in `collection_create` (the gate stops old
        // binaries from silently rebuilding with Cosine; no mixed v7/v8 either).
        // Version changes must explicitly touch the test; the protocol is not
        // cross-version compatible.
        assert_eq!(REPL_PROTOCOL_VERSION, 8);
    }
}
