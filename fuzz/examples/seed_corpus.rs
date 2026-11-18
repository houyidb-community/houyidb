//! 语料种子生成（P4.5.1）：用**真实编码器**为 6 个 fuzz target 产出起始语料。
//! EN: Corpus seed generation (P4.5.1): real encoders produce the starting
//! EN: corpora for the 6 fuzz targets.
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

// int8 图种子经 `HotBufferIndex` 的异步写入入口建图，异步方法来自该 trait。
// EN: int8 graph seeds build their graph through the async write entry of
// EN: `HotBufferIndex`, whose methods come from this trait.
use houyi_vector::VectorIndex;

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
    seed_collection_wal()?;
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
            sparse: None,
        },
        CollectionWalItem {
            id: 2,
            values: vec![0.3, 0.4],
            label: None,
            payload: None,
            sparse: None,
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

    // int8 图快照（v3：codes + source）：只有真实 int8 构建器能产出 flags=0x04
    // 的 v3 payload。解码器先比对 flags 派生的 quantization、再解析向量段，因此
    // 「按量化版本选择解码布局」分支只在**期望 int8** 的恢复目标上才可达；
    // fuzz target 同时以未量化与 int8 两种目标恢复同一语料。
    // EN: int8 graph snapshot (v3: codes + source): only a real int8 builder
    // emits a v3 payload with flags=0x04. The decoder compares flags-derived
    // quantization before parsing the vector records, so the "pick the decode
    // layout by quantization version" branch is reachable only on an *int8-
    // expecting* restore target — the fuzz target restores this corpus into
    // both an unquantized and an int8 graph.
    let int8_payload = build_graph_snapshot(
        &rt,
        4,
        vec![
            (1, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
        ],
        houyi_vector::Quantization::Int8,
    );
    let mut seed = 42u64.to_le_bytes().to_vec();
    seed.extend_from_slice(&int8_payload);
    write_seed(&dir, "lsn_plus_payload_int8", &seed)?;
    write_seed(
        &dir,
        "envelope_int8",
        &encode_graph_snapshot_file(42, &int8_payload),
    )?;

    // rabitq 图快照（v4：norm/alpha/packed codes/source，flags bit3=0x08）：
    // 同样只有真实 rabitq 构建器能产出；fuzzer 需要「期望 rabitq」的恢复
    // 目标才能深入 v4 向量段与位打包校验。
    // EN: rabitq graph snapshot (v4: norm/alpha/packed codes/source, flags
    // bit3=0x08): likewise only a real rabitq builder emits it; the fuzzer
    // needs a *rabitq-expecting* restore target to reach the v4 vector records
    // and packed-bits validation.
    let rabitq_payload = build_graph_snapshot(
        &rt,
        4,
        vec![
            (1, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
        ],
        houyi_vector::Quantization::RaBitQ,
    );
    let mut seed = 42u64.to_le_bytes().to_vec();
    seed.extend_from_slice(&rabitq_payload);
    write_seed(&dir, "lsn_plus_payload_rabitq", &seed)?;
    write_seed(
        &dir,
        "envelope_rabitq",
        &encode_graph_snapshot_file(42, &rabitq_payload),
    )?;
    Ok(())
}

/// 构建并排空一个真实量化图，返回对应版本的快照 payload（int8 → v3，
/// rabitq → v4）。
/// 图节点数断言：`indexed_count` 只反映主表，无法发现被图插入拒绝的行；节点
/// 少于期望会静默产出退化但仍可解码的种子。
/// 字节可复现性：节点/向量段按 AHashMap 迭代序写出，故重跑生成的内容等价、
/// 字节不同；入库样本只是任一有效快照，不承诺逐字节重现。
/// EN: Build and drain a real quantized graph, returning the matching snapshot
/// payload (int8 → v3, rabitq → v4).
/// The graph-node assertion matters: `indexed_count` only tracks the main map, so
/// rows rejected by graph insertion would not fail it — a short graph would
/// silently ship a degraded but still decodable seed.
/// Byte reproducibility: nodes/vectors are written in AHashMap iteration order,
/// so regenerating yields equivalent content with different bytes; the checked-in
/// sample is any one valid snapshot, not a byte-reproducible one.
fn build_graph_snapshot(
    rt: &tokio::runtime::Runtime,
    dims: usize,
    rows: Vec<(u64, Vec<f32>)>,
    quantization: houyi_vector::Quantization,
) -> Vec<u8> {
    let expected = rows.len();
    let index = houyi_vector::HotBufferIndex::new_with_metric_and_dtype_and_quantization(
        dims,
        houyi_vector::DistanceMetric::L2,
        houyi_vector::VectorDtype::F32,
        quantization,
    );
    rt.block_on(async {
        index
            .insert_batch(rows)
            .await
            .expect("seed rows must enter the builder queue");
        while index.pending_build_count().await > 0 {
            index.run_async_builder_once().await;
        }
    });
    assert_eq!(
        rt.block_on(index.graph_node_count()),
        expected,
        "seed graph must link every inserted row"
    );
    rt.block_on(index.encode_graph_snapshot())
}

/// 目标 5：真实 collections 快照 payload + manifest 各一份（P5.0.5）。
/// EN: Target 5: one real collections snapshot payload and manifest (P5.0.5).
fn seed_collection_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    use houyi_persistence::collection_snapshot_file::{
        encode_collection_manifest, encode_collection_snapshot_payload, CollectionManifest,
        CollectionManifestEntry, CollectionSnapshotPayload, SnapshotQuantization, SnapshotScalar,
        SnapshotScalarRow, SnapshotVectorDtype, SnapshotVectorKind,
    };

    let dir = corpus_dir("collection_snapshot_decode");
    let payload = CollectionSnapshotPayload {
        checkpoint_lsn: 7,
        name: "docs".to_string(),
        dims: 4,
        index_type: "hnsw".to_string(),
        ha_level: 1,
        metric: Some(1),
        hnsw: Some(
            houyi_persistence::collection_snapshot_file::SnapshotHnswParams {
                m: 24,
                ef_construction: 400,
                ef_search: 256,
            },
        ),
        vector_dtype: Some(SnapshotVectorDtype::F32),
        quantization: None,
        // 未量化：保持 v1 manifest 兼容样本。
        // EN: unquantized: keeps the v1-manifest compatibility sample.
        vector_kind: SnapshotVectorKind::Empty,
        vector_count: 0,
        vector_bytes: Vec::new(),
        scalars: vec![
            SnapshotScalarRow {
                id: 1,
                columns: vec![
                    ("flag".to_string(), SnapshotScalar::Bool(true)),
                    ("g".to_string(), SnapshotScalar::I64(2)),
                    (
                        "label".to_string(),
                        SnapshotScalar::Text("label_1p".to_string()),
                    ),
                    (
                        "payload".to_string(),
                        SnapshotScalar::JsonText("{\"g\":2}".to_string()),
                    ),
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
        sparse: None,
    };
    write_seed(
        &dir,
        "payload.bin",
        &encode_collection_snapshot_payload(&payload)?,
    )?;

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
            vector_dtype: Some(0),
            quantization: None,
            sparse: None,
            // 仅 int8 写 Some(1)；此处保持未量化。
            // EN: only int8 writes Some(1); this sample stays unquantized.
            vector_kind: SnapshotVectorKind::Empty as u8,
            file: "docs.collection_snapshot.7.1.0.bin".to_string(),
            file_len: 128,
            payload_crc32: 0x1234_5678,
            vector_count: 0,
            scalar_rows: 2,
        }],
    };
    write_seed(
        &dir,
        "manifest.bin",
        &encode_collection_manifest(&manifest)?,
    )?;

    // int8 变体：真实 v3 图 payload 装进 HNSW_GRAPH 形态的 collection payload，
    // 以及 format_version=2 的 manifest（entry quantization=Some(1)）。让解码器
    // 从语料就能到达 int8 校验分支（flags bit1 / manifest v2），而不是靠变异猜
    // 位置；同时保留上面的未量化样本作为 v1 兼容对照。
    // EN: int8 variant: a real v3 graph payload inside an HNSW_GRAPH-shaped
    // collection payload, plus a format_version=2 manifest (entry
    // quantization=Some(1)). The decoder reaches the int8 validation branches
    // (payload flags bit1 / manifest v2) straight from the corpus instead of
    // mutation guessing positions; the unquantized sample above stays as the
    // v1-compatibility counterpart.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let int8_graph = build_graph_snapshot(
        &rt,
        4,
        vec![
            (1, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
        ],
        houyi_vector::Quantization::Int8,
    );
    let int8_graph_len = int8_graph.len() as u64;
    let int8_payload = CollectionSnapshotPayload {
        checkpoint_lsn: 7,
        name: "int8docs".to_string(),
        dims: 4,
        index_type: "hnsw".to_string(),
        ha_level: 1,
        metric: Some(0),
        hnsw: Some(
            houyi_persistence::collection_snapshot_file::SnapshotHnswParams {
                m: 24,
                ef_construction: 400,
                ef_search: 256,
            },
        ),
        vector_dtype: Some(SnapshotVectorDtype::F32),
        quantization: Some(SnapshotQuantization::Int8),
        vector_kind: SnapshotVectorKind::HnswGraph,
        vector_count: 3,
        vector_bytes: int8_graph,
        scalars: Vec::new(),
        tombstones: Vec::new(),
        sparse: None,
    };
    write_seed(
        &dir,
        "payload_int8.bin",
        &encode_collection_snapshot_payload(&int8_payload)?,
    )?;

    let int8_manifest = CollectionManifest {
        format_version: 2,
        manifest_seq: 2,
        created_at_unix: 1_700_000_000,
        global_barrier_lsn: 9,
        safe_checkpoint_lsn: 7,
        collections: vec![CollectionManifestEntry {
            name: "int8docs".to_string(),
            checkpoint_lsn: 7,
            dims: 4,
            index_type: "hnsw".to_string(),
            ha_level: 1,
            metric: Some(0),
            vector_dtype: Some(0),
            quantization: Some(SnapshotQuantization::Int8.as_u8()),
            sparse: None,
            vector_kind: SnapshotVectorKind::HnswGraph as u8,
            file: "int8docs.collection_snapshot.7.2.0.bin".to_string(),
            file_len: int8_graph_len,
            payload_crc32: 0,
            vector_count: 3,
            scalar_rows: 0,
        }],
    };
    write_seed(
        &dir,
        "manifest_int8.bin",
        &encode_collection_manifest(&int8_manifest)?,
    )?;

    // rabitq 变体（P3）：payload flags bit2=0x04，manifest format_version=3、
    // entry quantization=Some(2)。三个版本号（payload flag / 图快照 v4 /
    // manifest format 3）必须同源一致，种子让解码器从语料直达这些校验分支。
    // EN: rabitq variant (P3): payload flags bit2=0x04, manifest
    // format_version=3 and entry quantization=Some(2). The three version marks
    // (payload flag / graph snapshot v4 / manifest format 3) must agree; the
    // seed lands the decoder directly on those validation branches.
    let rabitq_graph = build_graph_snapshot(
        &rt,
        4,
        vec![
            (1, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
        ],
        houyi_vector::Quantization::RaBitQ,
    );
    let rabitq_graph_len = rabitq_graph.len() as u64;
    let rabitq_payload = CollectionSnapshotPayload {
        checkpoint_lsn: 7,
        name: "rabitqdocs".to_string(),
        dims: 4,
        index_type: "hnsw".to_string(),
        ha_level: 1,
        metric: Some(0),
        hnsw: Some(
            houyi_persistence::collection_snapshot_file::SnapshotHnswParams {
                m: 24,
                ef_construction: 400,
                ef_search: 256,
            },
        ),
        vector_dtype: Some(SnapshotVectorDtype::F32),
        quantization: Some(SnapshotQuantization::RaBitQ),
        vector_kind: SnapshotVectorKind::HnswGraph,
        vector_count: 3,
        vector_bytes: rabitq_graph,
        scalars: Vec::new(),
        tombstones: Vec::new(),
        sparse: None,
    };
    write_seed(
        &dir,
        "payload_rabitq.bin",
        &encode_collection_snapshot_payload(&rabitq_payload)?,
    )?;

    let rabitq_manifest = CollectionManifest {
        format_version: 3,
        manifest_seq: 3,
        created_at_unix: 1_700_000_000,
        global_barrier_lsn: 9,
        safe_checkpoint_lsn: 7,
        collections: vec![CollectionManifestEntry {
            name: "rabitqdocs".to_string(),
            checkpoint_lsn: 7,
            dims: 4,
            index_type: "hnsw".to_string(),
            ha_level: 1,
            metric: Some(0),
            vector_dtype: Some(0),
            quantization: Some(SnapshotQuantization::RaBitQ.as_u8()),
            sparse: None,
            vector_kind: SnapshotVectorKind::HnswGraph as u8,
            file: "rabitqdocs.collection_snapshot.7.3.0.bin".to_string(),
            file_len: rabitq_graph_len,
            payload_crc32: 0,
            vector_count: 3,
            scalar_rows: 0,
        }],
    };
    write_seed(
        &dir,
        "manifest_rabitq.bin",
        &encode_collection_manifest(&rabitq_manifest)?,
    )?;

    // 稀疏变体（P4）：payload flags bit3=0x08，倒排段内联在 tombstones 之后
    // （v3）；manifest format_version=4、条目 sparse=Some(1)。语料让解码器直达
    // term/posting 升序、tf 有限正数与墓碑交叉校验分支。
    // EN: sparse variant (P4): payload flags bit3=0x08 with the inverted-index
    // section appended after tombstones (v3); manifest format_version=4 and entry
    // sparse=Some(1). The seed lands the decoder directly on the term/posting
    // ascending order, finite-positive tf, and tombstone cross-check branches.
    let sparse_payload = CollectionSnapshotPayload {
        checkpoint_lsn: 7,
        name: "sparse_docs".to_string(),
        dims: 4,
        index_type: "flat".to_string(),
        ha_level: 1,
        metric: Some(0),
        hnsw: None,
        vector_dtype: None,
        quantization: None,
        vector_kind: SnapshotVectorKind::Empty,
        vector_count: 0,
        vector_bytes: Vec::new(),
        scalars: vec![SnapshotScalarRow {
            id: 1,
            columns: vec![("g".to_string(), SnapshotScalar::I64(2))],
        }],
        tombstones: vec![9],
        sparse: Some(
            houyi_persistence::collection_snapshot_file::SnapshotSparseIndex {
                doc_count: 2,
                // 文档 1（term 7: 1.0 + term 8: 0.5）与文档 2（term 7: 2.0）。
                // EN: doc 1 (term 7: 1.0 + term 8: 0.5) and doc 2 (term 7: 2.0).
                total_len: 3.5,
                terms: vec![(7, vec![(1, 1.0), (2, 2.0)]), (8, vec![(1, 0.5)])],
            },
        ),
    };
    write_seed(
        &dir,
        "payload_sparse.bin",
        &encode_collection_snapshot_payload(&sparse_payload)?,
    )?;

    let sparse_manifest = CollectionManifest {
        format_version: 4,
        manifest_seq: 4,
        created_at_unix: 1_700_000_000,
        global_barrier_lsn: 9,
        safe_checkpoint_lsn: 7,
        collections: vec![CollectionManifestEntry {
            name: "sparse_docs".to_string(),
            checkpoint_lsn: 7,
            dims: 4,
            index_type: "flat".to_string(),
            ha_level: 1,
            metric: Some(0),
            vector_dtype: None,
            quantization: None,
            sparse: Some(1),
            vector_kind: SnapshotVectorKind::Empty as u8,
            file: "sparse_docs.collection_snapshot.7.4.0.bin".to_string(),
            file_len: 128,
            payload_crc32: 0x1234_5678,
            vector_count: 0,
            scalar_rows: 1,
        }],
    };
    write_seed(
        &dir,
        "manifest_sparse.bin",
        &encode_collection_manifest(&sparse_manifest)?,
    )?;
    Ok(())
}

/// 目标 6：真实 collections WAL payload（create 含 `sparse=bm25`、upsert 含
/// 稀疏词项 / 空列表 / 缺省、delete、drop、set_ha_level）。
/// EN: Target 6: real collections WAL payloads (create with `sparse=bm25`,
/// upsert with sparse terms / an empty list / an absent field, delete, drop,
/// set_ha_level).
fn seed_collection_wal() -> Result<(), Box<dyn std::error::Error>> {
    use houyi_persistence::collection_wal::{
        encode_collection_create_with_options, encode_collection_delete, encode_collection_drop,
        encode_collection_set_ha_level, encode_collection_upsert_batch, CollectionCreateOptions,
        CollectionWalItem,
    };

    let dir = corpus_dir("collection_wal_decode");
    let mut options = CollectionCreateOptions::default();
    options.metric = Some("l2".to_string());
    options.sparse = Some("bm25".to_string());
    write_seed(
        &dir,
        "create_bm25.json",
        &encode_collection_create_with_options("sparse_docs", 4, "hnsw", &options)?,
    )?;

    let items = vec![
        CollectionWalItem {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("a".to_string()),
            payload: None,
            sparse: Some(vec![(7, 1.0), (8, 0.5)]),
        },
        CollectionWalItem {
            id: 2,
            values: vec![0.0, 1.0, 0.0, 0.0],
            label: None,
            payload: None,
            // 空列表 = 清空该文档稀疏（区别于缺省 = 不改）。
            // EN: an empty list clears the document's sparse terms (distinct from
            // absence, which leaves them unchanged).
            sparse: Some(Vec::new()),
        },
        CollectionWalItem {
            id: 3,
            values: vec![0.0, 0.0, 1.0, 0.0],
            label: None,
            payload: None,
            sparse: None,
        },
    ];
    write_seed(
        &dir,
        "upsert_sparse.json",
        &encode_collection_upsert_batch("sparse_docs", 4, &items)?,
    )?;
    write_seed(
        &dir,
        "delete.json",
        &encode_collection_delete("sparse_docs", &[1, 3])?,
    )?;
    write_seed(&dir, "drop.json", &encode_collection_drop("sparse_docs")?)?;
    write_seed(
        &dir,
        "set_ha_level.json",
        &encode_collection_set_ha_level(
            "sparse_docs",
            houyi_persistence::ha_level::HaLevel::FullSync,
        )?,
    )?;
    Ok(())
}
