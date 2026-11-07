//! 语料种子生成（P4.5.1）：用**真实编码器**为 4 个 fuzz target 产出起始语料。
//! EN: Corpus seed generation (P4.5.1): real encoders produce the starting
//! EN: corpora for the 4 fuzz targets.
//!
//! 为什么需要 / Why: 解析器的输入受 CRC32 / 长度前缀 / 自洽长度门保护，纯随机
//! 字节几乎必然在第一个门就被拒——fuzzer 的预算会耗在「猜结构」上。语料种子从
//! **合法结构**出发，变异才落在语义上（图结构、墓碑 trailer、SQL 文本、帧字段）。
//! EN: Why: parser inputs are guarded by CRC32 / length prefixes / self-consistent
//! EN: length checks, so random bytes almost always fail the first gate and the
//! EN: fuzzer budget is spent guessing structure. Seeds start from valid
//! EN: structures, so mutations land on semantics (graph, tombstones, SQL, frames).
//!
//! 用法 / Usage:
//! EN: Usage: run the command below.
//! ```text
//! cd fuzz && cargo +nightly run --example seed_corpus
//! ```
//! 幂等：重复运行覆盖同名种子。语料 `fuzz/corpus/` 入库（不是 .gitignore 项）。
//! EN: Idempotent: reruns overwrite same-named seeds. `fuzz/corpus/` is checked
//! EN: into git (not gitignored).

use std::fs;
use std::path::{Path, PathBuf};

/// 语料根目录（编译期锚定，避免依赖 cwd）。
/// EN: Corpus root directory (anchored at compile time, no cwd dependency).
fn corpus_dir(target: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("corpus")
        .join(target)
}

fn write_seed(dir: &Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join(name), bytes)
}

/// 进程唯一临时路径（seed 生成期的一次性 scratch，非测试）。
/// EN: Process-unique temporary path (one-off scratch during seed generation, not a test).
fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "houyi_seed_{tag}_{}_{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    seed_wal()?;
    seed_repl()?;
    seed_pgwire()?;
    seed_graph()?;
    seed_collection_snapshot()?;
    println!(
        "语料种子已写入 {}",
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("corpus")
            .display()
    );
    Ok(())
}

/// 目标 1：真实 v2 WAL 文件（含撕裂尾变体）+ 三种 collection 记录。
/// EN: Target 1: real v2 WAL file (with a torn-tail variant) + three collection records.
fn seed_wal() -> Result<(), Box<dyn std::error::Error>> {
    use houyi_persistence::collection_wal::{
        encode_collection_create, encode_collection_delete, encode_collection_set_ha_level,
        encode_collection_upsert_batch, CollectionWalItem,
    };
    use houyi_persistence::FileWal;

    let dir = corpus_dir("wal_v2_decode");
    let tmp = temp_path("wal");
    let (mut wal, _) = FileWal::open(&tmp)?;
    wal.append_no_flush(b"{\"op\":\"create\",\"name\":\"items\"}".to_vec())?;
    wal.append_no_flush(vec![0xA5u8; 64])?;
    wal.append_no_flush(b"tail-record".to_vec())?;
    wal.flush_wal()?;
    drop(wal);
    let bytes = fs::read(&tmp)?;
    fs::remove_file(&tmp).ok();

    write_seed(&dir, "v2_real_wal", &bytes)?;
    // 撕裂尾：末 3 字节被砍 → 尾记录不完整，解析应静默截断在有效前缀。
    // EN: Torn tail: last 3 bytes chopped off → tail record incomplete; parsing should
    // EN: silently truncate at the valid prefix.
    write_seed(&dir, "v2_torn_tail", &bytes[..bytes.len() - 3])?;
    // v2 文件头 + 版本号错位 → 版本门。`11` 硬编码了「magic 8 字节 + version u32 BE
    // 在 8..12」的布局，故生成期自检：布局一旦漂移，这里立刻失败，而不是悄悄产出
    // 打不中版本门的失准种子（与 v2_real_wal 近乎重复 = 覆盖静默退化）。
    // EN: v2 file header with a misplaced version → version gate. `11` hard-codes the layout
    // EN: "magic 8 bytes + version u32 BE at 8..12", so generation self-checks: a layout
    // EN: drift fails here immediately instead of silently emitting a seed that misses the
    // EN: version gate (nearly identical to v2_real_wal = silent coverage regression).
    let mut bad_version = bytes.clone();
    bad_version[11] = 9;
    assert!(
        houyi_persistence::fuzz_api::parse_wal_file(&bad_version).is_err()
            && houyi_persistence::fuzz_api::parse_wal_file(&bytes).is_ok(),
        "v2_bad_version 种子必须命中版本门（WAL 文件头布局漂移？）"
    );
    write_seed(&dir, "v2_bad_version", &bad_version)?;

    // collection 记录：真实 JSON envelope，走 op 分派（create/upsert_batch/
    // delete/set_ha_level）。upsert_batch 是唯一解析 `items`（f32 值数组 + 可选
    // label/payload）的记录，且含 dims 不匹配的 fail-fast 分支——确定性种子把这些
    // 分支直接钉进起始语料，不留给变异去猜。
    // EN: collection record: real JSON envelope dispatched by op (create/upsert_batch/
    // EN: delete/set_ha_level). upsert_batch is the only record that parses `items` (f32
    // EN: value array + optional label/payload), and it has the dims-mismatch fail-fast
    // EN: branch. The deterministic seed pins these branches into the starting corpus
    // EN: instead of leaving them for mutation to guess.
    write_seed(
        &dir,
        "collection_create",
        &encode_collection_create("items", 2, "hnsw")?,
    )?;
    let items = vec![
        CollectionWalItem {
            id: 1,
            values: vec![0.1, 0.2],
            label: Some("a".to_string()),
            payload: Some(
                serde_json::json!({"n": 7, "t": "x"})
                    .as_object()
                    .expect("json object")
                    .clone(),
            ),
        },
        CollectionWalItem {
            id: 2,
            values: vec![0.3, 0.4],
            label: None,
            payload: None,
        },
    ];
    write_seed(
        &dir,
        "collection_upsert_batch",
        &encode_collection_upsert_batch("items", 2, &items)?,
    )?;
    write_seed(
        &dir,
        "collection_delete",
        &encode_collection_delete("items", &[1, 2, 3])?,
    )?;
    write_seed(
        &dir,
        "collection_set_ha_level",
        &encode_collection_set_ha_level("items", houyi_persistence::ha_level::HaLevel::FullSync)?,
    )?;
    Ok(())
}

/// 目标 3：复制协议 v3 各类帧，前面补 18 字节结构化字段区（tag/stream/lsn/
/// stage/kind/payload_len/msg_len），使 `Unstructured` 解出的字段合法、`take_rest()`
/// 恰好是真实帧字节。
/// EN: Target 3: replication protocol v3 frames of every kind, prefixed with an 18-byte
/// EN: structured field area (tag/stream/lsn/stage/kind/payload_len/msg_len) so
/// EN: `Unstructured` yields legal fields and `take_rest()` is exactly the real frame
/// EN: bytes.
fn seed_repl() -> Result<(), Box<dyn std::error::Error>> {
    use houyi_persistence::wal_replication as proto;

    let dir = corpus_dir("repl_frame_decode");
    let prefix = |payload_len: u32, msg_len: u16| -> Vec<u8> {
        let mut p = Vec::with_capacity(18);
        p.push(proto::FRAME_DATA);
        p.push(proto::STREAM_COLLECTIONS);
        p.extend_from_slice(&42u64.to_le_bytes()); // arbitrary 读 u64 用 from_le_bytes
        p.push(proto::ACK_STAGE_DURABLE);
        p.push(proto::ERROR_KIND_GENERIC);
        p.extend_from_slice(&payload_len.to_le_bytes());
        p.extend_from_slice(&msg_len.to_le_bytes());
        p
    };

    // DATA：头 + 真实 payload。
    // EN: DATA: header + real payload.
    let payload = b"{\"op\":\"upsert\",\"name\":\"items\",\"ids\":[1,2,3]}";
    let mut data =
        proto::encode_data_header(proto::STREAM_COLLECTIONS, 42, payload.len() as u32).to_vec();
    data.extend_from_slice(payload);
    let mut seed = prefix(payload.len() as u32, 0);
    seed.extend_from_slice(&data);
    write_seed(&dir, "data_frame", &seed)?;

    // HANDSHAKE / WELCOME / HEARTBEAT / ACK：定长帧。
    // EN: HANDSHAKE / WELCOME / HEARTBEAT / ACK: fixed-length frames.
    let mut seed = prefix(0, 0);
    seed.extend_from_slice(&proto::encode_handshake(
        42,
        proto::STREAM_MASK_COLLECTIONS,
        0xDEAD_BEEF_CAFE_F00D,
        0,
        0,
        [7u8; 16],
        "seed-token",
    )?);
    write_seed(&dir, "handshake", &seed)?;

    let mut seed = prefix(0, 0);
    seed.extend_from_slice(&proto::encode_welcome(43, 0));
    write_seed(&dir, "welcome", &seed)?;

    let mut seed = prefix(0, 0);
    seed.extend_from_slice(&proto::encode_heartbeat(proto::STREAM_COLLECTIONS, 42));
    write_seed(&dir, "heartbeat", &seed)?;

    let mut seed = prefix(0, 0);
    seed.extend_from_slice(&proto::encode_ack(
        proto::STREAM_COLLECTIONS,
        proto::ACK_STAGE_DURABLE,
        42,
    ));
    write_seed(&dir, "ack_durable", &seed)?;

    // ERROR：kind + 消息体（msg_len 与真实长度一致）。
    // EN: ERROR: kind + message body (msg_len matches the real length).
    let msg = "follower_ahead";
    let mut seed = prefix(0, msg.len() as u16);
    seed.extend_from_slice(&proto::encode_error(proto::ERROR_KIND_GENERIC, msg));
    write_seed(&dir, "error_frame", &seed)?;
    Ok(())
}

/// 目标 4：合法 tag + 该 tag 的真实报文体（组装后帧自洽，直接落 SQL 解析路径）。
/// EN: Target 4: a legal tag + the real message body for that tag (the assembled frame
/// EN: is self-consistent and goes straight into the SQL parse path).
fn seed_pgwire() -> Result<(), Box<dyn std::error::Error>> {
    let dir = corpus_dir("pgwire_message_decode");

    let mut seed = vec![b'Q'];
    seed.extend_from_slice(b"SELECT id FROM items LIMIT 10\0");
    write_seed(&dir, "query_select", &seed)?;

    // 负长度字段（0xFFFFFFFF → i32 -1）：P4.5.1 首轮 fuzz 崩溃样本的最小化形态。
    // 旧实现 `i32 as usize` 符号扩展 → `len + 1` 加法溢出 panic；修复后必须 Err。
    // 种子入库 = 该路径永久留在覆盖集里（回归测试另见 pgwire.rs 单元测试）。
    // EN: Negative length field (0xFFFFFFFF → i32 -1): minimized form of the first-round
    // EN: P4.5.1 fuzz crash. The old `i32 as usize` sign extension → `len + 1` addition
    // EN: overflow panic; after the fix it must return Err. Checking the seed into the repo
    // EN: keeps this path in the coverage set permanently (regression test in pgwire.rs).
    let mut seed = vec![b'Q'];
    seed.extend_from_slice(&(-1i32).to_be_bytes());
    seed.push(0);
    write_seed(&dir, "query_negative_length", &seed)?;

    // Parse 消息：name\0 query\0 num_params(i16 BE) + n×oid(i32 BE)。
    // EN: Parse: name\0 query\0 num_params(i16 BE) + n×oid(i32 BE).
    let mut seed = vec![b'P'];
    seed.extend_from_slice(b"stmt1\0SELECT 1\0");
    seed.extend_from_slice(&0i16.to_be_bytes());
    write_seed(&dir, "parse_zero_params", &seed)?;

    let mut seed = vec![b'P'];
    seed.extend_from_slice(b"stmt2\0SELECT id FROM items WHERE id = $1\0");
    seed.extend_from_slice(&1i16.to_be_bytes());
    seed.extend_from_slice(&23i32.to_be_bytes()); // int4 oid
    write_seed(&dir, "parse_one_param", &seed)?;

    // Bind 消息：portal\0 stmt\0 n_fmt(i16) [fmt×n]
    // n_params(i16) [len(i32)+val] n_res(i16) [fmt]
    // EN: Bind: portal\0 stmt\0 n_fmt(i16) [fmt×n]
    // EN: n_params(i16) [len(i32)+val] n_res(i16) [fmt]
    let mut seed = vec![b'B'];
    seed.extend_from_slice(b"portal1\0stmt1\0");
    seed.extend_from_slice(&0i16.to_be_bytes()); // n_fmt = 0
    seed.extend_from_slice(&1i16.to_be_bytes()); // n_params = 1
    seed.extend_from_slice(&5i32.to_be_bytes()); // 参数长度 5
    seed.extend_from_slice(b"12345");
    seed.extend_from_slice(&0i16.to_be_bytes()); // n_result_formats = 0
    write_seed(&dir, "bind_one_param", &seed)?;

    // startup packet：tag 字节 + protocol(3.0) + key\0value\0… + 终止 \0。
    // EN: startup packet: tag byte + protocol(3.0) + key\0value\0… + terminating \0.
    let mut seed = vec![b'Q'];
    seed.extend_from_slice(&196608i32.to_be_bytes());
    seed.extend_from_slice(b"user\0houyi\0database\0houyi\0\0");
    write_seed(&dir, "startup_packet", &seed)?;
    Ok(())
}

/// 目标 2：真实图快照 payload（空图）+ 带墓碑 trailer 的 payload + 真实信封文件。
/// EN: Target 2: real graph snapshot payload (empty graph) + payload with tombstone
/// EN: trailer + real envelope file.
fn seed_graph() -> Result<(), Box<dyn std::error::Error>> {
    use houyi_persistence::graph_snapshot_file::encode_graph_snapshot_file;

    let dir = corpus_dir("graph_snapshot_restore");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let index = houyi_vector::HotBufferIndex::new(4);
    let payload = rt.block_on(index.encode_graph_snapshot());

    // 空墓碑集：payload 末尾 8 字节是 count=0（大端）。以此为锚点替换成 2 条墓碑，
    // 让「墓碑 count 对账」路径在语料里就可达（否则只能靠变异猜到 trailer 位置）。
    // EN: Empty tombstone set: the last 8 bytes of the payload are count=0 (big-endian).
    // EN: Use that anchor to substitute 2 tombstones so the "tombstone count reconciliation"
    // EN: path is reachable from the corpus (otherwise mutation must guess the trailer).
    assert!(
        payload.len() >= 8 && payload[payload.len() - 8..].iter().all(|b| *b == 0),
        "空图 payload 应以 8 字节全零墓碑 count 结尾"
    );
    let mut with_tombstones = payload[..payload.len() - 8].to_vec();
    with_tombstones.extend_from_slice(&2u64.to_be_bytes());
    with_tombstones.extend_from_slice(&7u64.to_be_bytes());
    with_tombstones.extend_from_slice(&9u64.to_be_bytes());

    // 结构感知布局：前 8 字节 LSN + payload。
    // EN: Structure-aware layout: first 8 bytes LSN + payload.
    let mut seed = 42u64.to_le_bytes().to_vec();
    seed.extend_from_slice(&payload);
    write_seed(&dir, "lsn_plus_payload", &seed)?;

    let mut seed = 42u64.to_le_bytes().to_vec();
    seed.extend_from_slice(&with_tombstones);
    write_seed(&dir, "lsn_plus_payload_tombstones", &seed)?;

    // 信封层：真实信封文件（magic/version/长度/CRC 全合法，供变异打头部）。
    // EN: Envelope layer: real envelope file (magic/version/length/CRC all valid, so
    // EN: mutations can hit the header).
    write_seed(
        &dir,
        "envelope_valid",
        &encode_graph_snapshot_file(42, &payload),
    )?;
    write_seed(
        &dir,
        "envelope_tombstones",
        &encode_graph_snapshot_file(42, &with_tombstones),
    )?;
    Ok(())
}

/// 目标 5：真实 collections 快照 payload + manifest 各一份（P5.0.5）。
/// EN: Target 5: one real collections snapshot payload and manifest (P5.0.5).
fn seed_collection_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    use houyi_persistence::collection_snapshot_file::{
        encode_collection_manifest, encode_collection_snapshot_payload, CollectionManifest,
        CollectionManifestEntry, CollectionSnapshotPayload, SnapshotScalar, SnapshotScalarRow,
        SnapshotVectorKind,
    };

    let dir = corpus_dir("collection_snapshot_decode");
    let payload = CollectionSnapshotPayload {
        checkpoint_lsn: 7,
        name: "docs".to_string(),
        dims: 4,
        index_type: "hnsw".to_string(),
        ha_level: 1,
        metric: Some(1),
        hnsw: Some(houyi_persistence::collection_snapshot_file::SnapshotHnswParams {
            m: 24,
            ef_construction: 400,
            ef_search: 256,
        }),
        vector_kind: SnapshotVectorKind::Empty,
        vector_count: 0,
        vector_bytes: Vec::new(),
        scalars: vec![
            SnapshotScalarRow {
                id: 1,
                columns: vec![
                    ("flag".to_string(), SnapshotScalar::Bool(true)),
                    ("g".to_string(), SnapshotScalar::I64(2)),
                    ("label".to_string(), SnapshotScalar::Text("label_1p".to_string())),
                    ("payload".to_string(), SnapshotScalar::JsonText("{\"g\":2}".to_string())),
                    ("score".to_string(), SnapshotScalar::F64(0.5)),
                    ("ts".to_string(), SnapshotScalar::Timestamp(1_700_000_000)),
                ],
            },
            SnapshotScalarRow {
                id: 2,
                columns: vec![("g".to_string(), SnapshotScalar::I64(0))],
            },
        ],
        tombstones: vec![3, 9],
    };
    write_seed(&dir, "payload.bin", &encode_collection_snapshot_payload(&payload)?)?;

    let manifest = CollectionManifest {
        format_version: 1,
        manifest_seq: 1,
        created_at_unix: 1_700_000_000,
        global_barrier_lsn: 9,
        safe_checkpoint_lsn: 7,
        collections: vec![CollectionManifestEntry {
            name: "docs".to_string(),
            checkpoint_lsn: 7,
            dims: 4,
            index_type: "hnsw".to_string(),
            ha_level: 1,
            metric: Some(1),
            vector_kind: SnapshotVectorKind::Empty as u8,
            file: "docs.collection_snapshot.7.1.0.bin".to_string(),
            file_len: 128,
            payload_crc32: 0x1234_5678,
            vector_count: 0,
            scalar_rows: 2,
        }],
    };
    write_seed(&dir, "manifest.bin", &encode_collection_manifest(&manifest)?)?;
    Ok(())
}
