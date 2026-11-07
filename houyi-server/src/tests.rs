/// 测试用集群控制句柄（只读打开，不落盘）。 EN: Test-only cluster handle.
fn test_cluster() -> std::sync::Arc<houyi_server::cluster::ClusterControl> {
    // 每次调用独占目录（时间戳可能同纳秒重复，加原子序号）。
    // EN: per-call directory (timestamps can collide; add an atomic counter).
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "houyi-bin-test-cluster-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::sync::Arc::new(
        houyi_server::cluster::ClusterControl::open(&dir.join("test.wal"), false)
            .expect("cluster control"),
    )
}

use super::{run_pgwire_listener, run_pgwire_listener_with_metrics};
use houyi_core::{HouyiPage, ScalarPredicate, ScalarValue};
use houyi_query::pgwire::BindParamValue;
use houyi_server::collection::CollectionManager;
use houyi_server::http_handler::{benchmark_pgwire_status, benchmark_wal_checkpoint};
use houyi_server::index_backend::IndexBackend;
use houyi_server::pgwire_handler::{
    decode_houyi_upsert_batch_bytea_binary, decode_houyi_vector_bytea_binary,
    execute_hybrid_with_scalar_bitset, execute_query, pgwire_bind_param_to_sql_literal,
    pgwire_float_list_to_directive_text,
};
use houyi_server::types::{
    BenchmarkHttpState, InflightWatermark, PgWireMetrics, PgWireSharedState, WalHandle,
    WalWriteMode, DEFAULT_MAX_HTTP_INFLIGHT,
};
use houyi_vector::{HotBufferIndex, VectorIndex};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_postgres::{types::Type, NoTls};

fn encode_pg_float4_array_binary(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&1_i32.to_be_bytes()); // ndim
    out.extend_from_slice(&0_i32.to_be_bytes()); // hasnull
    out.extend_from_slice(&700_i32.to_be_bytes()); // FLOAT4OID
    out.extend_from_slice(&(values.len() as i32).to_be_bytes()); // dim len
    out.extend_from_slice(&1_i32.to_be_bytes()); // lower bound
    for value in values {
        out.extend_from_slice(&4_i32.to_be_bytes());
        out.extend_from_slice(&value.to_bits().to_be_bytes());
    }
    out
}

fn encode_houyi_vector_bytea_binary(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    out
}

fn encode_houyi_vector_bytea_binary_v1(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + values.len() * 4);
    out.extend_from_slice(b"AVB1");
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    out.extend_from_slice(&encode_houyi_vector_bytea_binary(values));
    out
}

fn encode_houyi_upsert_batch_bytea_binary_v1(dim: u32, items: &[(u64, Vec<f32>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"AUB1");
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for (id, v) in items {
        assert_eq!(v.len(), dim as usize);
        out.extend_from_slice(&id.to_le_bytes());
        for x in v {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    out
}

fn read_single_column_format_and_oid(row_desc_body: &[u8]) -> (u32, i16) {
    let mut idx = 2usize;
    while row_desc_body[idx] != 0 {
        idx += 1;
    }
    idx += 1;
    idx += 4; // table oid
    idx += 2; // attr num
    let type_oid = i32::from_be_bytes(row_desc_body[idx..idx + 4].try_into().unwrap()) as u32;
    idx += 4;
    idx += 2; // type size
    idx += 4; // type modifier
    let format = i16::from_be_bytes(row_desc_body[idx..idx + 2].try_into().unwrap());
    (type_oid, format)
}

async fn startup_pgwire_test_conn(addr: std::net::SocketAddr) -> TcpStream {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();
    stream
}

async fn assert_binary_plain_scalar_result(sql: &[u8], expected_oid: u32, expected_bytes: &[u8]) {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = startup_pgwire_test_conn(addr).await;

    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut desc = Vec::new();
    desc.push(b'D');
    let desc_body = b"P\0";
    desc.extend_from_slice(&((4 + desc_body.len()) as i32).to_be_bytes());
    desc.extend_from_slice(desc_body);
    stream.write_all(&desc).await.unwrap();
    let mut t_header = [0_u8; 5];
    stream.read_exact(&mut t_header).await.unwrap();
    assert_eq!(t_header[0], b'T');
    let t_len = i32::from_be_bytes(t_header[1..5].try_into().unwrap()) as usize;
    let mut t_body = vec![0_u8; t_len - 4];
    stream.read_exact(&mut t_body).await.unwrap();
    let (type_oid, format) = read_single_column_format_and_oid(&t_body);
    assert_eq!(type_oid, expected_oid);
    assert_eq!(format, 1);

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut d_header = [0_u8; 5];
    stream.read_exact(&mut d_header).await.unwrap();
    assert_eq!(d_header[0], b'D');
    let d_len = i32::from_be_bytes(d_header[1..5].try_into().unwrap()) as usize;
    let mut d_body = vec![0_u8; d_len - 4];
    stream.read_exact(&mut d_body).await.unwrap();
    let value_len = i32::from_be_bytes(d_body[2..6].try_into().unwrap()) as usize;
    assert_eq!(value_len, expected_bytes.len());
    assert_eq!(&d_body[6..6 + value_len], expected_bytes);
}

async fn connect_tokio_postgres_client(addr: std::net::SocketAddr) -> tokio_postgres::Client {
    let config = format!(
        "host={} port={} user=tester dbname=houyi",
        addr.ip(),
        addr.port()
    );
    let (client, connection) = tokio_postgres::connect(&config, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

#[test]
fn pgwire_binary_scalar_param_decode_supports_phase1_types() {
    assert_eq!(
        pgwire_bind_param_to_sql_literal(16, 1, &BindParamValue::Binary(vec![1_u8]),).unwrap(),
        "TRUE"
    );
    assert_eq!(
        pgwire_bind_param_to_sql_literal(
            20,
            1,
            &BindParamValue::Binary(42_i64.to_be_bytes().to_vec()),
        )
        .unwrap(),
        "42"
    );
    assert_eq!(
        pgwire_bind_param_to_sql_literal(
            700,
            1,
            &BindParamValue::Binary(3.5_f32.to_bits().to_be_bytes().to_vec()),
        )
        .unwrap(),
        "3.5"
    );
    assert_eq!(
        pgwire_bind_param_to_sql_literal(
            701,
            1,
            &BindParamValue::Binary(7.25_f64.to_bits().to_be_bytes().to_vec()),
        )
        .unwrap(),
        "7.25"
    );
}

#[test]
fn pgwire_binary_float4_array_param_decodes_to_vector_directive_text() {
    let encoded = encode_pg_float4_array_binary(&[1.0, 0.25, 3.5, -4.0]);
    assert_eq!(
        pgwire_bind_param_to_sql_literal(1021, 1, &BindParamValue::Binary(encoded)).unwrap(),
        "1,0.25,3.5,-4"
    );
}

#[test]
fn pgwire_binary_bytea_vector_param_decodes_to_directive_text() {
    let encoded = encode_houyi_vector_bytea_binary(&[1.0, 0.25, 3.5, -4.0]);
    assert_eq!(
        decode_houyi_vector_bytea_binary(&encoded, None)
            .map(|v| pgwire_float_list_to_directive_text(&v))
            .unwrap(),
        "1,0.25,3.5,-4"
    );
}

#[test]
fn pgwire_binary_bytea_vector_v1_param_decodes_to_directive_text() {
    let encoded = encode_houyi_vector_bytea_binary_v1(&[1.0, 0.25, 3.5, -4.0]);
    assert_eq!(
        decode_houyi_vector_bytea_binary(&encoded, Some(4))
            .map(|v| pgwire_float_list_to_directive_text(&v))
            .unwrap(),
        "1,0.25,3.5,-4"
    );
}

#[test]
fn pgwire_binary_bytea_vector_rejects_dim_mismatch() {
    let encoded = encode_houyi_vector_bytea_binary_v1(&[1.0, 0.25, 3.5, -4.0]);
    let err = decode_houyi_vector_bytea_binary(&encoded, Some(3)).unwrap_err();
    assert!(err.contains("server vector_dim"), "unexpected error: {err}");
}

#[test]
fn pgwire_binary_bytea_upsert_batch_decodes_v1() {
    let payload = encode_houyi_upsert_batch_bytea_binary_v1(
        4,
        &[
            (7, vec![1.0_f32, 0.0, 0.0, 0.0]),
            (9, vec![0.0_f32, 1.0, 0.0, 0.0]),
        ],
    );
    let batch = decode_houyi_upsert_batch_bytea_binary(&payload, Some(4)).unwrap();
    assert_eq!(batch.dim, 4);
    assert_eq!(batch.items.len(), 2);
    assert_eq!(batch.items[0].0, 7);
    assert_eq!(batch.items[1].0, 9);
}

#[test]
fn pgwire_binary_bytea_upsert_batch_rejects_dim_mismatch() {
    let payload =
        encode_houyi_upsert_batch_bytea_binary_v1(4, &[(7, vec![1.0_f32, 0.0, 0.0, 0.0])]);
    let err = decode_houyi_upsert_batch_bytea_binary(&payload, Some(3)).unwrap_err();
    assert!(err.contains("server vector_dim"), "unexpected error: {err}");
}

#[test]
fn pgwire_binary_float4_array_rejects_wrong_element_oid() {
    let mut encoded = encode_pg_float4_array_binary(&[1.0, 2.0]);
    encoded[8..12].copy_from_slice(&701_i32.to_be_bytes());
    let err =
        pgwire_bind_param_to_sql_literal(1021, 1, &BindParamValue::Binary(encoded)).unwrap_err();
    assert!(
        err.contains("element OID mismatch"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn vector_wal_replay_applies_batch_to_index() {
    use super::replay_vector_wal_into_index;
    use houyi_persistence::vector_wal::encode_vector_upsert_batch;
    use houyi_persistence::FileWal;
    use houyi_vector::DistanceMetric;

    let path = std::env::temp_dir().join(format!(
        "houyidb-wal-replay-test-{}.wal",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let dim = 4usize;
    let batch = vec![(42u64, vec![0.1f32, 0.2, 0.3, 0.4])];
    let payload = encode_vector_upsert_batch(dim, &batch).unwrap();
    {
        let (mut wal, _) = FileWal::open(&path).unwrap();
        use houyi_persistence::Wal;
        wal.append(payload);
    }
    let (_wal, entries) = FileWal::open(&path).unwrap();
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new_with_metric(
        dim,
        DistanceMetric::Cosine,
    )));
    let n = replay_vector_wal_into_index(&entries, &index, dim, None)
        .await
        .unwrap();
    assert_eq!(n, 1);
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn vector_wal_replay_skips_lsn_at_or_below_checkpoint() {
    use super::replay_vector_wal_into_index;
    use houyi_persistence::vector_wal::encode_vector_upsert_batch;
    use houyi_persistence::FileWal;
    use houyi_persistence::Wal;
    use houyi_vector::DistanceMetric;

    let path = std::env::temp_dir().join(format!(
        "houyidb-wal-replay-minlsn-{}.wal",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let dim = 4usize;
    let batch1 = vec![(1u64, vec![0.1f32, 0.2, 0.3, 0.4])];
    let batch2 = vec![(2u64, vec![0.5f32, 0.6, 0.7, 0.8])];
    {
        let (mut wal, _) = FileWal::open(&path).unwrap();
        wal.append(encode_vector_upsert_batch(dim, &batch1).unwrap());
        wal.append(encode_vector_upsert_batch(dim, &batch2).unwrap());
    }
    let (_wal, entries) = FileWal::open(&path).unwrap();
    assert!(entries.len() >= 2);
    let cp = entries[0].lsn;
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new_with_metric(
        dim,
        DistanceMetric::Cosine,
    )));
    let n = replay_vector_wal_into_index(&entries, &index, dim, Some(cp))
        .await
        .unwrap();
    assert_eq!(n, 1);
    let _ = std::fs::remove_file(&path);
}

/// 同 id 两条 WAL 批次按 LSN 顺序重放后，检索语义与最终向量一致（last-wins，客户端重试安全）。
/// EN: Replaying two WAL batches for the same id in LSN order keeps search semantics
/// EN: equal to the final vector (last-wins, safe for client retries).
#[tokio::test]
async fn vector_wal_replay_same_id_last_wins_across_batches() {
    use super::drain_pending_builder;
    use super::replay_vector_wal_into_index;
    use houyi_persistence::vector_wal::encode_vector_upsert_batch;
    use houyi_persistence::FileWal;
    use houyi_persistence::Wal;
    use houyi_vector::DistanceMetric;

    let path = std::env::temp_dir().join(format!(
        "houyidb-wal-replay-lastwins-{}.wal",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let dim = 4usize;
    let batch_first = vec![(1u64, vec![1.0_f32, 0.0, 0.0, 0.0])];
    let batch_second = vec![(1u64, vec![0.0_f32, 1.0, 0.0, 0.0])];
    {
        let (mut wal, _) = FileWal::open(&path).unwrap();
        wal.append(encode_vector_upsert_batch(dim, &batch_first).unwrap());
        wal.append(encode_vector_upsert_batch(dim, &batch_second).unwrap());
    }
    let (_wal, entries) = FileWal::open(&path).unwrap();
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new_with_metric(
        dim,
        DistanceMetric::Cosine,
    )));
    let n = replay_vector_wal_into_index(&entries, &index, dim, None)
        .await
        .unwrap();
    assert_eq!(n, 2);
    drain_pending_builder(&index).await;
    let q = vec![0.0_f32, 1.0, 0.0, 0.0];
    let top = index.search_with_ef(&q, 1, None, Some(64)).await;
    assert_eq!(top, vec![1], "last batch vector should win for id=1");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn execute_query_applies_pushed_down_bitset() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(1, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();
    index.insert(2, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();
    index.insert(3, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();

    let ids = execute_query(
        &index,
        "SELECT id FROM docs WHERE id IN (2, 3)",
        &[0.1, 0.2, 0.3, 0.4],
        10,
        None,
    )
    .await
    .unwrap();
    assert_eq!(ids, vec![2, 3]);
}

#[tokio::test]
async fn execute_query_without_where_returns_all_up_to_k() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(11, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();
    index.insert(12, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();

    let ids = execute_query(
        &index,
        "SELECT id FROM docs",
        &[0.1, 0.2, 0.3, 0.4],
        1,
        None,
    )
    .await
    .unwrap();
    assert_eq!(ids, vec![11]);
}

#[tokio::test]
async fn scalar_bitset_is_applied_to_vector_search() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(0, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();
    index.insert(1, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();
    index.insert(2, vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();

    let mut page = HouyiPage::new(4096);
    for tag in ["p0", "p1", "p1"] {
        let mut scalars = std::collections::HashMap::new();
        scalars.insert("tag".to_string(), ScalarValue::Text(tag.to_string()));
        page.insert_row(scalars, None);
    }

    let ids = execute_hybrid_with_scalar_bitset(
        &index,
        &page,
        "tag",
        ScalarPredicate::Eq(ScalarValue::Text("p1".to_string())),
        &[0.1, 0.2, 0.3, 0.4],
        10,
    )
    .await;
    assert_eq!(ids, vec![1, 2]);
}

#[tokio::test]
async fn pgwire_handshake_returns_auth_and_ready() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut packet = Vec::new();
    packet.extend_from_slice(&len.to_be_bytes());
    packet.extend_from_slice(&payload);
    stream.write_all(&packet).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    assert_eq!(auth[0], b'R');

    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    assert_eq!(ps1_header[0], b'S');
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();

    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    assert_eq!(ps2_header[0], b'S');
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();

    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();
    assert_eq!(ready[0], b'Z');
    assert_eq!(ready[5], b'I');
}

#[tokio::test]
async fn pgwire_simple_query_returns_command_complete_and_ready() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    // 启动握手。
    // EN: startup handshake
    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    // 消费握手输出。
    // EN: consume handshake outputs
    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // 发送简单查询。
    // EN: send simple query
    let query_sql = b"SELECT 1\0";
    let q_len = (query_sql.len() + 4) as i32;
    let mut q = Vec::new();
    q.push(b'Q');
    q.extend_from_slice(&q_len.to_be_bytes());
    q.extend_from_slice(query_sql);
    stream.write_all(&q).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let val_len = i32::from_be_bytes(data_row_body[2..6].try_into().unwrap());
    let text = std::str::from_utf8(&data_row_body[6..6 + val_len as usize]).unwrap();
    assert_eq!(text, "1");

    let mut complete_header = [0_u8; 5];
    stream.read_exact(&mut complete_header).await.unwrap();
    assert_eq!(complete_header[0], b'C');
    let complete_len = i32::from_be_bytes(complete_header[1..5].try_into().unwrap()) as usize;
    let mut complete_body = vec![0_u8; complete_len - 4];
    stream.read_exact(&mut complete_body).await.unwrap();

    let mut ready2 = [0_u8; 6];
    stream.read_exact(&mut ready2).await.unwrap();
    assert_eq!(ready2[0], b'Z');
    assert_eq!(ready2[5], b'I');
}

#[tokio::test]
async fn pgwire_extended_query_parse_bind_execute_sync() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready0 = [0_u8; 6];
    stream.read_exact(&mut ready0).await.unwrap();

    let parse_body = b"\0SELECT 1\0\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_body.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_body);
    stream.write_all(&parse).await.unwrap();

    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let bind_body = b"\0\0\x00\x00\x00\x00\x00\x00";
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(bind_body);
    stream.write_all(&bind).await.unwrap();

    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();

    let mut complete_header = [0_u8; 5];
    stream.read_exact(&mut complete_header).await.unwrap();
    assert_eq!(complete_header[0], b'C');
    let complete_len = i32::from_be_bytes(complete_header[1..5].try_into().unwrap()) as usize;
    let mut complete_body = vec![0_u8; complete_len - 4];
    stream.read_exact(&mut complete_body).await.unwrap();

    let mut sync = Vec::new();
    sync.push(b'S');
    sync.extend_from_slice(&4_i32.to_be_bytes());
    stream.write_all(&sync).await.unwrap();

    let mut ready2 = [0_u8; 6];
    stream.read_exact(&mut ready2).await.unwrap();
    assert_eq!(ready2[0], b'Z');
}

#[tokio::test]
async fn pgwire_vector_search_returns_id_rows() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index
        .insert(42, vec![1.0_f32, 0.0, 0.0, 0.0])
        .await
        .unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let qtext = "/* houyi:vector=1,0,0,0 */ SELECT id FROM docs LIMIT 5\0";
    let q_len = (qtext.len() + 4) as i32;
    let mut q = Vec::new();
    q.push(b'Q');
    q.extend_from_slice(&q_len.to_be_bytes());
    q.extend_from_slice(qtext.as_bytes());
    stream.write_all(&q).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let col_count = i16::from_be_bytes(data_row_body[0..2].try_into().unwrap());
    assert_eq!(col_count, 1);
    let val_len = i32::from_be_bytes(data_row_body[2..6].try_into().unwrap());
    let text = std::str::from_utf8(&data_row_body[6..6 + val_len as usize]).unwrap();
    assert_eq!(text, "42");

    let mut complete_header = [0_u8; 5];
    stream.read_exact(&mut complete_header).await.unwrap();
    assert_eq!(complete_header[0], b'C');
    let complete_len = i32::from_be_bytes(complete_header[1..5].try_into().unwrap()) as usize;
    let mut complete_body = vec![0_u8; complete_len - 4];
    stream.read_exact(&mut complete_body).await.unwrap();
    assert!(complete_body.starts_with(b"SELECT "));

    let mut ready2 = [0_u8; 6];
    stream.read_exact(&mut ready2).await.unwrap();
    assert_eq!(ready2[0], b'Z');
}

#[tokio::test]
async fn pgwire_extended_bind_one_param_for_vector_search() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql = b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs WHERE id IN ($1) LIMIT 5\0\x00\x01\x00\x00\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let val_len = i32::from_be_bytes(data_row_body[2..6].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&data_row_body[6..6 + val_len]).unwrap();
    assert_eq!(text, "7");

    let mut seen_ids = vec![text.to_string()];
    loop {
        let mut h = [0_u8; 5];
        stream.read_exact(&mut h).await.unwrap();
        let msg_len = i32::from_be_bytes(h[1..5].try_into().unwrap()) as usize;
        let mut body = vec![0_u8; msg_len - 4];
        stream.read_exact(&mut body).await.unwrap();
        match h[0] {
            b'D' => {
                let val_len = i32::from_be_bytes(body[2..6].try_into().unwrap()) as usize;
                let value = std::str::from_utf8(&body[6..6 + val_len])
                    .unwrap()
                    .to_string();
                seen_ids.push(value);
            }
            b'C' => break,
            other => panic!("unexpected message tag before CommandComplete: {other}"),
        }
    }
    assert!(seen_ids.iter().any(|id| id == "7"));
}

#[tokio::test]
async fn pgwire_extended_bind_one_binary_int4_param_for_vector_search() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql =
            b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs WHERE id IN ($1) LIMIT 5\0\x00\x01\x00\x00\x00\x17";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one shared format code
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // binary
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one param
    bind_body.extend_from_slice(&4_i32.to_be_bytes());
    bind_body.extend_from_slice(&7_i32.to_be_bytes());
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let val_len = i32::from_be_bytes(data_row_body[2..6].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&data_row_body[6..6 + val_len]).unwrap();
    assert_eq!(text, "7");

    let mut seen_ids = vec![text.to_string()];
    loop {
        let mut h = [0_u8; 5];
        stream.read_exact(&mut h).await.unwrap();
        let msg_len = i32::from_be_bytes(h[1..5].try_into().unwrap()) as usize;
        let mut body = vec![0_u8; msg_len - 4];
        stream.read_exact(&mut body).await.unwrap();
        match h[0] {
            b'D' => {
                let val_len = i32::from_be_bytes(body[2..6].try_into().unwrap()) as usize;
                let value = std::str::from_utf8(&body[6..6 + val_len])
                    .unwrap()
                    .to_string();
                seen_ids.push(value);
            }
            b'C' => break,
            other => panic!("unexpected message tag before CommandComplete: {other}"),
        }
    }
    assert!(seen_ids.iter().any(|id| id == "7"));
}

#[tokio::test]
async fn pgwire_extended_bind_binary_float4_array_for_vector_directive() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql =
        b"\0/* houyi:vector=$1 */ SELECT id FROM docs LIMIT 5\0\x00\x01\x00\x00\x03\xfd";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let encoded_vec = encode_pg_float4_array_binary(&[1.0, 0.0, 0.0, 0.0]);
    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one shared format code
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // binary
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one param
    bind_body.extend_from_slice(&(encoded_vec.len() as i32).to_be_bytes());
    bind_body.extend_from_slice(&encoded_vec);
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let val_len = i32::from_be_bytes(data_row_body[2..6].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&data_row_body[6..6 + val_len]).unwrap();
    assert_eq!(text, "7");

    let mut seen_ids = vec![text.to_string()];
    loop {
        let mut h = [0_u8; 5];
        stream.read_exact(&mut h).await.unwrap();
        let msg_len = i32::from_be_bytes(h[1..5].try_into().unwrap()) as usize;
        let mut body = vec![0_u8; msg_len - 4];
        stream.read_exact(&mut body).await.unwrap();
        match h[0] {
            b'D' => {
                let val_len = i32::from_be_bytes(body[2..6].try_into().unwrap()) as usize;
                let value = std::str::from_utf8(&body[6..6 + val_len])
                    .unwrap()
                    .to_string();
                seen_ids.push(value);
            }
            b'C' => break,
            other => panic!("unexpected message tag before CommandComplete: {other}"),
        }
    }
    assert!(seen_ids.iter().any(|id| id == "7"));
}

#[tokio::test]
async fn pgwire_extended_bind_binary_bytea_for_vector_directive() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = startup_pgwire_test_conn(addr).await;

    let parse_sql =
        b"\0/* houyi:vector=$1 */ SELECT id FROM docs LIMIT 5\0\x00\x01\x00\x00\x00\x11";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let encoded_vec = encode_houyi_vector_bytea_binary_v1(&[1.0, 0.0, 0.0, 0.0]);
    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&(encoded_vec.len() as i32).to_be_bytes());
    bind_body.extend_from_slice(&encoded_vec);
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let text_len = i32::from_be_bytes(data_row_body[2..6].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&data_row_body[6..6 + text_len]).unwrap();
    assert_eq!(text, "7");
}

#[tokio::test]
async fn pgwire_extended_bind_two_params_for_vector_search() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql = b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs WHERE id IN ($1,$2) LIMIT 5\0\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&2_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"9");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut seen_ids = Vec::<String>::new();
    loop {
        let mut h = [0_u8; 5];
        stream.read_exact(&mut h).await.unwrap();
        let msg_len = i32::from_be_bytes(h[1..5].try_into().unwrap()) as usize;
        let mut body = vec![0_u8; msg_len - 4];
        stream.read_exact(&mut body).await.unwrap();
        match h[0] {
            b'D' => {
                let val_len = i32::from_be_bytes(body[2..6].try_into().unwrap()) as usize;
                let text = std::str::from_utf8(&body[6..6 + val_len])
                    .unwrap()
                    .to_string();
                seen_ids.push(text);
            }
            b'C' => break,
            other => panic!("unexpected message tag before CommandComplete: {other}"),
        }
    }
    assert!(seen_ids.iter().any(|s| s == "7"));
    assert!(seen_ids.iter().any(|s| s == "9"));
}

#[tokio::test]
async fn pgwire_execute_with_max_rows_emits_portal_suspended() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql = b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs WHERE id IN ($1,$2) LIMIT 5\0\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&2_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"9");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex1 = Vec::new();
    ex1.push(b'E');
    let mut ex1_body = vec![0_u8];
    ex1_body.extend_from_slice(&1_i32.to_be_bytes());
    ex1.extend_from_slice(&((4 + ex1_body.len()) as i32).to_be_bytes());
    ex1.extend_from_slice(&ex1_body);
    stream.write_all(&ex1).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut d1_header = [0_u8; 5];
    stream.read_exact(&mut d1_header).await.unwrap();
    assert_eq!(d1_header[0], b'D');
    let d1_len = i32::from_be_bytes(d1_header[1..5].try_into().unwrap()) as usize;
    let mut d1_body = vec![0_u8; d1_len - 4];
    stream.read_exact(&mut d1_body).await.unwrap();
    let id1_len = i32::from_be_bytes(d1_body[2..6].try_into().unwrap()) as usize;
    let id1 = std::str::from_utf8(&d1_body[6..6 + id1_len])
        .unwrap()
        .to_string();
    assert!(id1 == "7" || id1 == "9");

    let mut s1 = [0_u8; 5];
    stream.read_exact(&mut s1).await.unwrap();
    assert_eq!(s1[0], b's');
    let s1_len = i32::from_be_bytes(s1[1..5].try_into().unwrap()) as usize;
    assert_eq!(s1_len, 4);

    let mut ex2 = Vec::new();
    ex2.push(b'E');
    let mut ex2_body = vec![0_u8];
    ex2_body.extend_from_slice(&1_i32.to_be_bytes());
    ex2.extend_from_slice(&((4 + ex2_body.len()) as i32).to_be_bytes());
    ex2.extend_from_slice(&ex2_body);
    stream.write_all(&ex2).await.unwrap();

    let mut d2_header = [0_u8; 5];
    stream.read_exact(&mut d2_header).await.unwrap();
    assert_eq!(d2_header[0], b'D');
    let d2_len = i32::from_be_bytes(d2_header[1..5].try_into().unwrap()) as usize;
    let mut d2_body = vec![0_u8; d2_len - 4];
    stream.read_exact(&mut d2_body).await.unwrap();
    let id2_len = i32::from_be_bytes(d2_body[2..6].try_into().unwrap()) as usize;
    let id2 = std::str::from_utf8(&d2_body[6..6 + id2_len])
        .unwrap()
        .to_string();
    assert!(id2 == "7" || id2 == "9");
    assert_ne!(id1, id2);

    let mut c2 = [0_u8; 5];
    stream.read_exact(&mut c2).await.unwrap();
    assert_eq!(c2[0], b'C');
}

#[tokio::test]
async fn pgwire_sync_clears_pending_portal_state() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // 为 IN ($1,$2) 执行 Parse + Bind。
    // EN: Parse + Bind for IN ($1,$2)
    let parse_sql = b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs WHERE id IN ($1,$2) LIMIT 5\0\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&2_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"9");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    // 首次 Execute，max_rows=1 => T + D + s。
    // EN: First execute with max_rows=1 => T + D + s
    let mut ex1 = Vec::new();
    ex1.push(b'E');
    let mut ex1_body = vec![0_u8];
    ex1_body.extend_from_slice(&1_i32.to_be_bytes());
    ex1.extend_from_slice(&((4 + ex1_body.len()) as i32).to_be_bytes());
    ex1.extend_from_slice(&ex1_body);
    stream.write_all(&ex1).await.unwrap();

    let mut t = [0_u8; 5];
    stream.read_exact(&mut t).await.unwrap();
    assert_eq!(t[0], b'T');
    let t_len = i32::from_be_bytes(t[1..5].try_into().unwrap()) as usize;
    let mut t_body = vec![0_u8; t_len - 4];
    stream.read_exact(&mut t_body).await.unwrap();

    let mut d = [0_u8; 5];
    stream.read_exact(&mut d).await.unwrap();
    assert_eq!(d[0], b'D');
    let d_len = i32::from_be_bytes(d[1..5].try_into().unwrap()) as usize;
    let mut d_body = vec![0_u8; d_len - 4];
    stream.read_exact(&mut d_body).await.unwrap();

    let mut s = [0_u8; 5];
    stream.read_exact(&mut s).await.unwrap();
    assert_eq!(s[0], b's');
    let s_len = i32::from_be_bytes(s[1..5].try_into().unwrap()) as usize;
    assert_eq!(s_len, 4);

    // Sync 应清空挂起的 portal 状态。
    // EN: Sync should clear pending portal state.
    let mut sync = Vec::new();
    sync.push(b'S');
    sync.extend_from_slice(&4_i32.to_be_bytes());
    stream.write_all(&sync).await.unwrap();

    let mut ready2 = [0_u8; 6];
    stream.read_exact(&mut ready2).await.unwrap();
    assert_eq!(ready2[0], b'Z');

    // 不重新 Bind 再次 Execute 应从头重启 portal（先返回 RowDescription）。
    // EN: Execute again without re-Bind should restart the portal from the beginning
    // EN: (RowDescription first).
    let mut ex2 = Vec::new();
    ex2.push(b'E');
    let mut ex2_body = vec![0_u8];
    ex2_body.extend_from_slice(&1_i32.to_be_bytes());
    ex2.extend_from_slice(&((4 + ex2_body.len()) as i32).to_be_bytes());
    ex2.extend_from_slice(&ex2_body);
    stream.write_all(&ex2).await.unwrap();

    let mut d2 = [0_u8; 5];
    stream.read_exact(&mut d2).await.unwrap();
    assert_eq!(d2[0], b'D');
    let d2_len = i32::from_be_bytes(d2[1..5].try_into().unwrap()) as usize;
    let mut d2_body = vec![0_u8; d2_len - 4];
    stream.read_exact(&mut d2_body).await.unwrap();

    let mut s2 = [0_u8; 5];
    stream.read_exact(&mut s2).await.unwrap();
    assert_eq!(s2[0], b's');
}

#[tokio::test]
async fn pgwire_describe_portal_works_before_and_after_portal_suspended() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // 执行 Parse + Bind。
    // EN: Parse + Bind
    let parse_sql = b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs WHERE id IN ($1,$2) LIMIT 5\0\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&2_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"9");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    // Execute 前 Describe(P)：应返回 RowDescription。
    // EN: Describe(P) before Execute: should return RowDescription
    let mut desc = Vec::new();
    desc.push(b'D');
    let desc_body = b"P\0";
    desc.extend_from_slice(&((4 + desc_body.len()) as i32).to_be_bytes());
    desc.extend_from_slice(desc_body);
    stream.write_all(&desc).await.unwrap();
    let mut t1 = [0_u8; 5];
    stream.read_exact(&mut t1).await.unwrap();
    assert_eq!(t1[0], b'T');
    let t1_len = i32::from_be_bytes(t1[1..5].try_into().unwrap()) as usize;
    let mut t1_body = vec![0_u8; t1_len - 4];
    stream.read_exact(&mut t1_body).await.unwrap();

    // Execute(max_rows=1)：期望一行 + 挂起。
    // EN: Execute(max_rows=1): expect row + suspended
    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&1_i32.to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut d = [0_u8; 5];
    stream.read_exact(&mut d).await.unwrap();
    assert_eq!(d[0], b'D');
    let d_len = i32::from_be_bytes(d[1..5].try_into().unwrap()) as usize;
    let mut d_body = vec![0_u8; d_len - 4];
    stream.read_exact(&mut d_body).await.unwrap();
    let mut s = [0_u8; 5];
    stream.read_exact(&mut s).await.unwrap();
    assert_eq!(s[0], b's');

    // 挂起期间 Describe(P)：仍应返回 RowDescription。
    // EN: Describe(P) while suspended: still should return RowDescription
    stream.write_all(&desc).await.unwrap();
    let mut t3 = [0_u8; 5];
    stream.read_exact(&mut t3).await.unwrap();
    assert_eq!(t3[0], b'T');
    let t3_len = i32::from_be_bytes(t3[1..5].try_into().unwrap()) as usize;
    let mut t3_body = vec![0_u8; t3_len - 4];
    stream.read_exact(&mut t3_body).await.unwrap();
}

#[tokio::test]
async fn wal_checkpoint_http_truncates_to_applied_sidecar() {
    use axum::{body::Body, http::Request, routing::post, Router};
    use houyi_persistence::wal_sidecar::{sidecar_path_for_wal, write_applied_lsn_atomic};
    use houyi_persistence::{FileWal, Wal};
    use tower::ServiceExt;

    let path = std::env::temp_dir().join(format!("houyidb-wal-ckpt-{}.wal", std::process::id()));
    let _ = std::fs::remove_file(&path);
    {
        let (mut wal, _) = FileWal::open(&path).unwrap();
        let _ = wal.append(b"a".to_vec());
        let _ = wal.append(b"b".to_vec());
    }
    let applied_sidecar = sidecar_path_for_wal(&path);
    let _ = std::fs::remove_file(&applied_sidecar);
    write_applied_lsn_atomic(&applied_sidecar, 1).unwrap();

    let wal = Arc::new(WalHandle::open(&path, WalWriteMode::Async).unwrap().0);
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    let collections_wal = Arc::new(
        WalHandle::open(
            &houyi_persistence::collection_wal::collections_wal_path_for(&path),
            WalWriteMode::Async,
        )
        .unwrap()
        .0,
    );
    let app = Router::new()
        .route("/v1/wal/checkpoint", post(benchmark_wal_checkpoint))
        .with_state(BenchmarkHttpState {
            audit: None,
            encryption: None,
            promote: None,
            collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
            index,
            wal: Arc::clone(&wal),
            collections_wal,
            vector_dim: 4,
            pgwire_metrics: Arc::new(PgWireMetrics::default()),
            applied_sidecar: applied_sidecar.clone(),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path: path.clone(),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
            graph_snapshot_metrics: Arc::new(
                houyi_server::recovery::GraphSnapshotMetrics::default(),
            ),
            role: houyi_server::types::RoleState::default(),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: houyi_server::types::shared_replication_metrics(None),
            lifecycle: std::sync::Arc::new(
                houyi_server::collection_lifecycle::CollectionLifecycleState::for_test(),
            ),
        });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/wal/checkpoint")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);

    let (_wal2, entries) = FileWal::open(&path).unwrap();
    assert!(entries.is_empty());
    assert_eq!(_wal2.next_lsn(), 2);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&applied_sidecar);
    let _ = std::fs::remove_file(houyi_persistence::collection_wal::collections_wal_path_for(
        &path,
    ));
    let _ = std::fs::remove_file(houyi_persistence::wal_sidecar::wal_next_lsn_meta_path(
        &path,
    ));
}

#[tokio::test]
async fn pgwire_unsupported_message_returns_error_response() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    // 启动握手。
    // EN: startup handshake
    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    // 消费握手。
    // EN: consume handshake
    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // 非扩展查询子集的前端报文：仅 ErrorResponse，需 Sync 后才 ReadyForQuery。
    // EN: Frontend message outside the extended-query subset: only ErrorResponse;
    // EN: ReadyForQuery arrives after Sync.
    let mut unsupported = Vec::new();
    unsupported.push(0x7F);
    unsupported.extend_from_slice(&4_i32.to_be_bytes());
    stream.write_all(&unsupported).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();

    let mut sync = Vec::new();
    sync.push(b'S');
    sync.extend_from_slice(&4_i32.to_be_bytes());
    stream.write_all(&sync).await.unwrap();

    let mut ready2 = [0_u8; 6];
    stream.read_exact(&mut ready2).await.unwrap();
    assert_eq!(ready2[0], b'Z');
}

#[tokio::test]
async fn pgwire_bind_binary_text_param_succeeds() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // Parse 未命名语句，显式 TEXTOID 参数。
    // EN: Parse unnamed statement with explicit TEXTOID parameter.
    let parse_sql = b"\0SELECT id FROM docs WHERE name = $1\0\x00\x01\x00\x00\x00\x19";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    // Bind 参数格式码 = 1（二进制）：text oid 的二进制 utf8 现在被接受。
    // EN: Bind with binary param format code = 1 => text oid binary utf8 is now accepted.
    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one param format code follows
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // binary format
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one param
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&0_i16.to_be_bytes()); // text result format
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();

    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');
}

#[tokio::test]
async fn pgwire_bind_param_count_mismatch_returns_protocol_violation() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // Parse 声明一个绑定参数。
    // EN: Parse declares one bind parameter.
    let parse_sql = b"\0SELECT id FROM docs WHERE id IN ($1)\0\x00\x01\x00\x00\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    // Bind 发送零个参数 => 不匹配。
    // EN: Bind sends zero params => mismatch.
    let bind_body = b"\0\0\x00\x00\x00\x00\x00\x00";
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(bind_body);
    stream.write_all(&bind).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(
        err_body.windows(7).any(|w| w == b"C08P01\0"),
        "expected SQLSTATE 08P01 in error body, got {:?}",
        err_body
    );
}

#[tokio::test]
async fn pgwire_describe_statement_returns_declared_param_oids() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // Parse 带一个显式 TEXTOID 参数。
    // EN: Parse with one explicit TEXTOID parameter.
    let parse_sql = b"\0SELECT id FROM docs WHERE id IN ($1)\0\x00\x01\x00\x00\x00\x19";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    // Describe 语句。
    // EN: Describe statement.
    let desc_body = b"S\0";
    let mut describe = Vec::new();
    describe.push(b'D');
    describe.extend_from_slice(&((4 + desc_body.len()) as i32).to_be_bytes());
    describe.extend_from_slice(desc_body);
    stream.write_all(&describe).await.unwrap();

    // ParameterDescription
    let mut t_header = [0_u8; 5];
    stream.read_exact(&mut t_header).await.unwrap();
    assert_eq!(t_header[0], b't');
    let t_len = i32::from_be_bytes(t_header[1..5].try_into().unwrap()) as usize;
    let mut t_body = vec![0_u8; t_len - 4];
    stream.read_exact(&mut t_body).await.unwrap();
    let count = i16::from_be_bytes(t_body[0..2].try_into().unwrap());
    assert_eq!(count, 1);
    let oid = i32::from_be_bytes(t_body[2..6].try_into().unwrap());
    assert_eq!(oid, 25);

    // RowDescription
    let mut row_header = [0_u8; 5];
    stream.read_exact(&mut row_header).await.unwrap();
    assert_eq!(row_header[0], b'T');
}

#[tokio::test]
async fn pgwire_bind_rejects_unsupported_declared_param_oid() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    // Parse 声明 JSONBOID (3802)，不在当前支持子集内。
    // EN: Parse declares JSONBOID (3802), outside the currently supported subset.
    let parse_sql = b"\0SELECT id FROM docs WHERE id = $1\0\x00\x01\x00\x00\x0e\xda";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(
        err_body.windows(7).any(|w| w == b"C0A000\0"),
        "expected SQLSTATE 0A000 in error body, got {:?}",
        err_body
    );
}

#[tokio::test]
async fn pgwire_bind_multiple_result_formats_returns_feature_not_supported() {
    let addr = run_pgwire_listener(
        "127.0.0.1:0",
        Some(Arc::new(PgWireSharedState {
            replication_metrics: houyi_server::types::shared_replication_metrics(None),
            collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
            index: IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4))),
            vector_dim: 4,
            pgwire_hnsw_m: None,
            pgwire_hnsw_ef_construction: None,
            wal: None,
            collections_wal: None,
            role: houyi_server::types::RoleState::default(),
            cluster: test_cluster(),
            query_log: None,
        })),
    )
    .await
    .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql = b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs LIMIT 5\0\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    // 单列 portal 必须拒绝多个结果格式码。
    // EN: Single-column portal must reject multiple result format codes.
    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes()); // param fmt count
    bind_body.extend_from_slice(&0_i16.to_be_bytes()); // num params
    bind_body.extend_from_slice(&2_i16.to_be_bytes()); // result fmt count
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(
        err_body.windows(7).any(|w| w == b"C0A000\0"),
        "expected SQLSTATE 0A000 in error body, got {:?}",
        err_body
    );
}

#[tokio::test]
async fn pgwire_execute_binary_result_format_returns_int8_payload() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql =
            b"\0/* houyi:vector=1,0,0,0 */ SELECT id FROM docs WHERE id IN ($1,$2) LIMIT 5\0\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&2_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"9");
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one result format code
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // binary
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut desc = Vec::new();
    desc.push(b'D');
    let desc_body = b"P\0";
    desc.extend_from_slice(&((4 + desc_body.len()) as i32).to_be_bytes());
    desc.extend_from_slice(desc_body);
    stream.write_all(&desc).await.unwrap();
    let mut t_header = [0_u8; 5];
    stream.read_exact(&mut t_header).await.unwrap();
    assert_eq!(t_header[0], b'T');
    let t_len = i32::from_be_bytes(t_header[1..5].try_into().unwrap()) as usize;
    let mut t_body = vec![0_u8; t_len - 4];
    stream.read_exact(&mut t_body).await.unwrap();
    let (type_oid, format) = read_single_column_format_and_oid(&t_body);
    assert_eq!(type_oid, 20);
    assert_eq!(format, 1);

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&1_i32.to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut d1_header = [0_u8; 5];
    stream.read_exact(&mut d1_header).await.unwrap();
    assert_eq!(d1_header[0], b'D');
    let d1_len = i32::from_be_bytes(d1_header[1..5].try_into().unwrap()) as usize;
    let mut d1_body = vec![0_u8; d1_len - 4];
    stream.read_exact(&mut d1_body).await.unwrap();
    let value_len = i32::from_be_bytes(d1_body[2..6].try_into().unwrap()) as usize;
    assert_eq!(value_len, 8);
    let value = i64::from_be_bytes(d1_body[6..14].try_into().unwrap());
    assert!(value == 7 || value == 9);

    let mut suspended = [0_u8; 5];
    stream.read_exact(&mut suspended).await.unwrap();
    assert_eq!(suspended[0], b's');
}

#[tokio::test]
async fn pgwire_execute_binary_result_format_returns_bool_payload() {
    assert_binary_plain_scalar_result(b"\0SELECT true\0\x00\x00", 16, &[1_u8]).await;
}

#[tokio::test]
async fn pgwire_execute_binary_result_format_returns_int4_payload() {
    assert_binary_plain_scalar_result(b"\0SELECT 42\0\x00\x00", 23, &42_i32.to_be_bytes()).await;
}

#[tokio::test]
async fn pgwire_execute_binary_result_format_returns_float4_payload() {
    assert_binary_plain_scalar_result(
        b"\0SELECT 3.5\0\x00\x00",
        700,
        &3.5_f32.to_bits().to_be_bytes(),
    )
    .await;
}

#[tokio::test]
async fn pgwire_execute_binary_result_format_returns_float8_payload() {
    assert_binary_plain_scalar_result(
        b"\0SELECT 1.0e39\0\x00\x00",
        701,
        &1.0e39_f64.to_bits().to_be_bytes(),
    )
    .await;
}

#[tokio::test]
async fn tokio_postgres_text_typed_query_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let rows = client
        .query_typed("SELECT $1", &[(&"hello", Type::TEXT)])
        .await
        .unwrap();
    let row = &rows[0];
    let value: String = row.get(0);
    assert_eq!(value, "hello");
}

#[tokio::test]
async fn tokio_postgres_binary_typed_query_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let rows = client
        .query_typed("SELECT $1", &[(&42_i32, Type::INT4)])
        .await
        .unwrap();
    let row = &rows[0];
    let value: i32 = row.get(0);
    assert_eq!(value, 42);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_text_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT 'hello'").await.unwrap();
    let row = client.query_one(&stmt, &[]).await.unwrap();
    let value: String = row.get(0);
    assert_eq!(value, "hello");
}

#[tokio::test]
async fn tokio_postgres_prepare_query_binary_int4_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT 42").await.unwrap();
    let row = client.query_one(&stmt, &[]).await.unwrap();
    let value: i32 = row.get(0);
    assert_eq!(value, 42);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_text_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT $1::text").await.unwrap();

    let row1 = client.query_one(&stmt, &[&"hello"]).await.unwrap();
    let value1: String = row1.get(0);
    assert_eq!(value1, "hello");

    let row2 = client.query_one(&stmt, &[&"world"]).await.unwrap();
    let value2: String = row2.get(0);
    assert_eq!(value2, "world");
}

#[tokio::test]
async fn pgwire_named_statement_binary_int4_param_returns_binary_payload() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = startup_pgwire_test_conn(addr).await;

    let parse_sql = b"s1\0SELECT $1::int4\0\x00\x00";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();

    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut desc_stmt = Vec::new();
    desc_stmt.push(b'D');
    let desc_stmt_body = b"Ss1\0";
    desc_stmt.extend_from_slice(&((4 + desc_stmt_body.len()) as i32).to_be_bytes());
    desc_stmt.extend_from_slice(desc_stmt_body);
    stream.write_all(&desc_stmt).await.unwrap();

    let mut pd_header = [0_u8; 5];
    stream.read_exact(&mut pd_header).await.unwrap();
    assert_eq!(pd_header[0], b't');
    let pd_len = i32::from_be_bytes(pd_header[1..5].try_into().unwrap()) as usize;
    let mut pd_body = vec![0_u8; pd_len - 4];
    stream.read_exact(&mut pd_body).await.unwrap();
    assert_eq!(&pd_body[0..2], &1_i16.to_be_bytes());
    assert_eq!(i32::from_be_bytes(pd_body[2..6].try_into().unwrap()), 23);

    let mut t_header = [0_u8; 5];
    stream.read_exact(&mut t_header).await.unwrap();
    assert_eq!(t_header[0], b'T');
    let t_len = i32::from_be_bytes(t_header[1..5].try_into().unwrap()) as usize;
    let mut t_body = vec![0_u8; t_len - 4];
    stream.read_exact(&mut t_body).await.unwrap();
    let (type_oid, format) = read_single_column_format_and_oid(&t_body);
    assert_eq!(type_oid, 23);
    assert_eq!(format, 0);

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0s1\0");
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&4_i32.to_be_bytes());
    bind_body.extend_from_slice(&41_i32.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();

    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut d_header = [0_u8; 5];
    stream.read_exact(&mut d_header).await.unwrap();
    assert_eq!(d_header[0], b'D');
    let d_len = i32::from_be_bytes(d_header[1..5].try_into().unwrap()) as usize;
    let mut d_body = vec![0_u8; d_len - 4];
    stream.read_exact(&mut d_body).await.unwrap();
    let value_len = i32::from_be_bytes(d_body[2..6].try_into().unwrap()) as usize;
    assert_eq!(value_len, 4);
    assert_eq!(&d_body[6..10], &41_i32.to_be_bytes());
}

#[tokio::test]
async fn tokio_postgres_prepare_query_binary_int4_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT $1::int4").await.unwrap();

    let row1 = client.query_one(&stmt, &[&41_i32]).await.unwrap();
    let value1: i32 = row1.get(0);
    assert_eq!(value1, 41);

    let row2 = client.query_one(&stmt, &[&42_i32]).await.unwrap();
    let value2: i32 = row2.get(0);
    assert_eq!(value2, 42);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_binary_int8_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT $1::int8").await.unwrap();

    let row1 = client.query_one(&stmt, &[&41_i64]).await.unwrap();
    let value1: i64 = row1.get(0);
    assert_eq!(value1, 41);

    let row2 = client.query_one(&stmt, &[&42_i64]).await.unwrap();
    let value2: i64 = row2.get(0);
    assert_eq!(value2, 42);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_binary_float4_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT $1::float4").await.unwrap();

    let row1 = client.query_one(&stmt, &[&1.5_f32]).await.unwrap();
    let value1: f32 = row1.get(0);
    assert_eq!(value1, 1.5_f32);

    let row2 = client.query_one(&stmt, &[&2.5_f32]).await.unwrap();
    let value2: f32 = row2.get(0);
    assert_eq!(value2, 2.5_f32);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_binary_float8_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT $1::float8").await.unwrap();

    let row1 = client.query_one(&stmt, &[&1.5_f64]).await.unwrap();
    let value1: f64 = row1.get(0);
    assert_eq!(value1, 1.5_f64);

    let row2 = client.query_one(&stmt, &[&2.5_f64]).await.unwrap();
    let value2: f64 = row2.get(0);
    assert_eq!(value2, 2.5_f64);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_binary_bool_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT $1::bool").await.unwrap();

    let row1 = client.query_one(&stmt, &[&true]).await.unwrap();
    let value1: bool = row1.get(0);
    assert!(value1);

    let row2 = client.query_one(&stmt, &[&false]).await.unwrap();
    let value2: bool = row2.get(0);
    assert!(!value2);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_multi_column_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client.prepare("SELECT 42, 'hello'").await.unwrap();
    let row = client.query_one(&stmt, &[]).await.unwrap();
    let value1: i32 = row.get(0);
    let value2: String = row.get(1);
    assert_eq!(value1, 42);
    assert_eq!(value2, "hello");
}

#[tokio::test]
async fn tokio_postgres_prepare_query_multi_column_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client
        .prepare("SELECT $1::int4, $2::text, $3::float8")
        .await
        .unwrap();

    let row1 = client
        .query_one(&stmt, &[&41_i32, &"alpha", &1.5_f64])
        .await
        .unwrap();
    let value11: i32 = row1.get(0);
    let value12: String = row1.get(1);
    let value13: f64 = row1.get(2);
    assert_eq!(value11, 41);
    assert_eq!(value12, "alpha");
    assert_eq!(value13, 1.5_f64);

    let row2 = client
        .query_one(&stmt, &[&42_i32, &"beta", &2.5_f64])
        .await
        .unwrap();
    let value21: i32 = row2.get(0);
    let value22: String = row2.get(1);
    let value23: f64 = row2.get(2);
    assert_eq!(value21, 42);
    assert_eq!(value22, "beta");
    assert_eq!(value23, 2.5_f64);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_multi_column_bool_combo_param_reuse_smoke() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client
        .prepare("SELECT $1::bool, $2::int8, $3::text")
        .await
        .unwrap();

    let row1 = client
        .query_one(&stmt, &[&true, &41_i64, &"alpha"])
        .await
        .unwrap();
    let value11: bool = row1.get(0);
    let value12: i64 = row1.get(1);
    let value13: String = row1.get(2);
    assert!(value11);
    assert_eq!(value12, 41);
    assert_eq!(value13, "alpha");

    let row2 = client
        .query_one(&stmt, &[&false, &42_i64, &"beta"])
        .await
        .unwrap();
    let value21: bool = row2.get(0);
    let value22: i64 = row2.get(1);
    let value23: String = row2.get(2);
    assert!(!value21);
    assert_eq!(value22, 42);
    assert_eq!(value23, "beta");
}

#[tokio::test]
async fn tokio_postgres_prepare_query_bytea_vector_param_smoke() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    index.insert(7, vec![1.0_f32, 0.0, 0.0, 0.0]).await.unwrap();
    index.insert(9, vec![0.0_f32, 1.0, 0.0, 0.0]).await.unwrap();
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let stmt = client
        .prepare("/* houyi:vector=$1 */ SELECT id FROM docs LIMIT 1")
        .await
        .unwrap();
    let payload = encode_houyi_vector_bytea_binary_v1(&[1.0_f32, 0.0, 0.0, 0.0]);
    let row = client.query_one(&stmt, &[&payload]).await.unwrap();
    let value: i64 = row.get(0);
    assert_eq!(value, 7);
}

#[tokio::test]
async fn tokio_postgres_prepare_query_bytea_upsert_batch_smoke() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let upsert_stmt = client
        .prepare_typed("/* houyi:upsert_batch=$1 */ SELECT 1", &[Type::BYTEA])
        .await
        .unwrap();
    let payload = encode_houyi_upsert_batch_bytea_binary_v1(
        4,
        &[
            (7, vec![1.0_f32, 0.0, 0.0, 0.0]),
            (9, vec![0.0_f32, 1.0, 0.0, 0.0]),
        ],
    );
    let row = client.query_one(&upsert_stmt, &[&payload]).await.unwrap();
    let inserted: i64 = row.get(0);
    assert_eq!(inserted, 2);

    let rep = index.run_async_builder_once().await;
    assert_eq!(
        rep.flushed, 2,
        "insert_batch queues vectors; builder must flush before ANN search"
    );

    let query_stmt = client
        .prepare("/* houyi:vector=$1 */ SELECT id FROM docs LIMIT 1")
        .await
        .unwrap();
    let q = encode_houyi_vector_bytea_binary_v1(&[1.0_f32, 0.0, 0.0, 0.0]);
    let row = client.query_one(&query_stmt, &[&q]).await.unwrap();
    let id: i64 = row.get(0);
    assert_eq!(id, 7);
}

/// 只读副本（P4.4.1）：role=follower 时 pgwire upsert_batch 必须以
/// SQLSTATE 25006（read_only_sql_transaction，PostgreSQL 只读事务同款）拒写，
/// 连接保持可用（错误后 ReadyForQuery 照常，检索路径不受影响），且零落盘。
/// EN: Follower replicas reject pgwire upsert_batch with SQLSTATE 25006 while
/// the connection stays usable and nothing is written.
#[tokio::test]
async fn pgwire_upsert_batch_follower_rejects_with_25006() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::new(houyi_server::types::ReplicationRole::Follower),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let upsert_stmt = client
        .prepare_typed("/* houyi:upsert_batch=$1 */ SELECT 1", &[Type::BYTEA])
        .await
        .unwrap();
    let payload =
        encode_houyi_upsert_batch_bytea_binary_v1(4, &[(7, vec![1.0_f32, 0.0, 0.0, 0.0])]);
    let err = client
        .query_one(&upsert_stmt, &[&payload])
        .await
        .expect_err("follower must reject upsert_batch");
    assert_eq!(
        err.code(),
        Some(&tokio_postgres::error::SqlState::READ_ONLY_SQL_TRANSACTION),
        "expected SQLSTATE 25006, got: {err}"
    );
    assert_eq!(
        index.hot_buffer_count().await,
        0,
        "rejected write must not reach the index"
    );

    // 错误后连接仍可用：检索路径（只读）不受影响，返回空结果集。
    // EN: The connection stays usable after the error: the read path is unaffected
    // EN: and returns an empty result set.
    let query_stmt = client
        .prepare("/* houyi:vector=$1 */ SELECT id FROM docs LIMIT 1")
        .await
        .unwrap();
    let q = encode_houyi_vector_bytea_binary_v1(&[1.0_f32, 0.0, 0.0, 0.0]);
    let rows = client.query(&query_stmt, &[&q]).await.unwrap();
    assert!(rows.is_empty());
}

/// 对称用例（OCR 批次十六）：带 collection 指令的 upsert 同样 25006，且
/// CollectionManager 零创建零落行——守卫必须挡在路由之前，而不是只护住
/// 主索引 else 分支（未来把检查挪进分支内的重构会被本测试当场抓住）。
/// EN: The collection-directive branch must also reject with 25006 before
/// routing, leaving the CollectionManager untouched.
#[tokio::test]
async fn pgwire_upsert_batch_collection_follower_rejects_with_25006() {
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    let collections = std::sync::Arc::new(CollectionManager::new(4, "hnsw"));
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::clone(&collections),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::new(houyi_server::types::ReplicationRole::Follower),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    let upsert_stmt = client
        .prepare_typed(
            "/* houyi:collection=pgw */ /* houyi:upsert_batch=$1 */ SELECT 1",
            &[Type::BYTEA],
        )
        .await
        .unwrap();
    let payload =
        encode_houyi_upsert_batch_bytea_binary_v1(4, &[(7, vec![1.0_f32, 0.0, 0.0, 0.0])]);
    let err = client
        .query_one(&upsert_stmt, &[&payload])
        .await
        .expect_err("follower must reject collection upsert_batch");
    assert_eq!(
        err.code(),
        Some(&tokio_postgres::error::SqlState::READ_ONLY_SQL_TRANSACTION),
        "expected SQLSTATE 25006, got: {err}"
    );
    assert!(
        collections.get("pgw").is_none(),
        "rejected write must not create the collection"
    );

    // 同连接的 SELECT 针对不存在的 collection：follower 必须 42P01 而非
    // 隐式创建（幻影 collection 会进 /v1/collections、与 leader 分叉）。
    // EN: A SELECT against a missing collection returns 42P01 on followers
    // instead of implicitly creating a phantom.
    let query_stmt = client
        .prepare("/* houyi:collection=ghost */ /* houyi:vector=$1 */ SELECT id FROM docs LIMIT 1")
        .await
        .unwrap();
    let q = encode_houyi_vector_bytea_binary_v1(&[1.0_f32, 0.0, 0.0, 0.0]);
    let err = client
        .query(&query_stmt, &[&q])
        .await
        .expect_err("follower SELECT on missing collection must 42P01");
    assert_eq!(
        err.code(),
        Some(&tokio_postgres::error::SqlState::UNDEFINED_TABLE),
        "expected SQLSTATE 42P01, got: {err}"
    );
    assert!(
        collections.get("ghost").is_none(),
        "SELECT must not implicitly create the collection on a follower"
    );
}

#[tokio::test]
async fn pgwire_upsert_batch_execute_increments_metrics() {
    use std::sync::atomic::Ordering;

    let metrics = Arc::new(PgWireMetrics::default());
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: None,
        collections_wal: None,
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener_with_metrics("127.0.0.1:0", Some(pg), Arc::clone(&metrics))
        .await
        .unwrap();
    let mut stream = startup_pgwire_test_conn(addr).await;

    let parse_sql = b"\0/* houyi:upsert_batch=$1 */ SELECT 1\0\x00\x01\x00\x00\x00\x11";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let payload = encode_houyi_upsert_batch_bytea_binary_v1(
        4,
        &[
            (7, vec![1.0_f32, 0.0, 0.0, 0.0]),
            (9, vec![0.0_f32, 1.0, 0.0, 0.0]),
        ],
    );
    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&(payload.len() as i32).to_be_bytes());
    bind_body.extend_from_slice(&payload);
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    assert_eq!(data_row_header[0], b'D');
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let text_len = i32::from_be_bytes(data_row_body[2..6].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&data_row_body[6..6 + text_len]).unwrap();
    assert_eq!(text, "2");

    let mut complete_header = [0_u8; 5];
    stream.read_exact(&mut complete_header).await.unwrap();
    assert_eq!(complete_header[0], b'C');
    let complete_len = i32::from_be_bytes(complete_header[1..5].try_into().unwrap()) as usize;
    let mut complete_body = vec![0_u8; complete_len - 4];
    stream.read_exact(&mut complete_body).await.unwrap();
    let _ = complete_body;

    assert_eq!(
        metrics.upsert_batch_execute_count.load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        metrics.upsert_batch_rows_inserted.load(Ordering::Relaxed),
        2
    );
}

/// pgwire 写入路径的 WAL 覆盖（此前全部 pgwire 测试装配 wal=None/collections_wal=None，
/// WAL-first 分支无任何真实落盘验证）：无 collection 指令的 upsert_batch 必须落主 WAL
/// （vector_upsert_batch）；带 collection 指令的必须落 collections WAL（create +
/// upsert_batch）。
/// EN: WAL coverage for the pgwire write path (previously every pgwire test
/// assembled wal=None/collections_wal=None, leaving the WAL-first branches
/// unverified): a directive-less upsert_batch must land in the main WAL
/// (vector_upsert_batch); one carrying a collection directive must land in the
/// collections WAL (create + upsert_batch).
#[tokio::test]
async fn pgwire_upsert_batch_persists_to_wals() {
    let path = std::env::temp_dir().join(format!(
        "houyidb-pgwire-wal-test-{}-{}.wal",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    let collections_path = houyi_persistence::collection_wal::collections_wal_path_for(&path);
    let _ = std::fs::remove_file(&collections_path);
    let (wal, _) = super::WalHandle::open(&path, super::WalWriteMode::Async).unwrap();
    let (collections_wal, _) =
        super::WalHandle::open(&collections_path, super::WalWriteMode::Async).unwrap();
    let index = IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4)));
    let pg = Arc::new(PgWireSharedState {
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
        index: index.clone(),
        vector_dim: 4,
        pgwire_hnsw_m: None,
        pgwire_hnsw_ef_construction: None,
        wal: Some(Arc::new(wal)),
        collections_wal: Some(Arc::new(collections_wal)),
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
    });
    let addr = run_pgwire_listener("127.0.0.1:0", Some(pg.clone()))
        .await
        .unwrap();
    let client = connect_tokio_postgres_client(addr).await;

    // 无指令批次 → 主 WAL。
    // EN: Batch without a directive -> main WAL.
    let plain_stmt = client
        .prepare_typed("/* houyi:upsert_batch=$1 */ SELECT 1", &[Type::BYTEA])
        .await
        .unwrap();
    let payload =
        encode_houyi_upsert_batch_bytea_binary_v1(4, &[(7, vec![1.0_f32, 0.0, 0.0, 0.0])]);
    let row = client.query_one(&plain_stmt, &[&payload]).await.unwrap();
    let inserted: i64 = row.get(0);
    assert_eq!(inserted, 1);

    // 带 collection 指令批次 → collections WAL（含隐式 create）。
    // EN: Batch with a collection directive -> collections WAL (including implicit create).
    let coll_stmt = client
        .prepare_typed(
            "/* houyi:collection=pgw */ /* houyi:upsert_batch=$1 */ SELECT 1",
            &[Type::BYTEA],
        )
        .await
        .unwrap();
    let payload = encode_houyi_upsert_batch_bytea_binary_v1(
        4,
        &[
            (11, vec![0.0_f32, 1.0, 0.0, 0.0]),
            (12, vec![0.0_f32, 0.0, 1.0, 0.0]),
        ],
    );
    let row = client.query_one(&coll_stmt, &[&payload]).await.unwrap();
    let inserted: i64 = row.get(0);
    assert_eq!(inserted, 2);

    // WalHandle::append 同步等待写线程确认，此处直接读文件即可。
    // EN: WalHandle::append blocks until the writer thread acks, so the files
    // are readable here without extra flushing.
    let (_w, main_entries) = houyi_persistence::FileWal::open(&path).unwrap();
    assert_eq!(
        main_entries.len(),
        1,
        "main WAL must hold exactly one record"
    );
    let (_w2, coll_entries) = houyi_persistence::FileWal::open(&collections_path).unwrap();
    assert_eq!(
        coll_entries.len(),
        2,
        "collections WAL must hold create + upsert_batch"
    );
    let first =
        houyi_persistence::collection_wal::decode_collection_wal_payload(&coll_entries[0].payload)
            .unwrap();
    let second =
        houyi_persistence::collection_wal::decode_collection_wal_payload(&coll_entries[1].payload)
            .unwrap();
    use houyi_persistence::collection_wal::CollectionWalRecord;
    // 隐式创建也记录**生效**的度量与 HNSW 参数：重放时按记录精确重建，而不是
    // 由 index_type 推断（否则未来默认度量变化会让历史集合静默漂移）。
    // EN: Implicit creation also logs the **effective** metric and HNSW params so
    // replay reconstructs exactly what was serving, rather than inferring from
    // index_type (a future default change would silently drift old collections).
    //
    // 期望值取创建后集合的实际参数，而不是写死 16/400/256：这些值由
    // `VectorConfig/HnswConfig::from_env()` 推导，`HOUYI_VECTOR_AUTO_HNSW`,
    // `HOUYI_VECTOR_RECALL_OPTIMIZED` 等环境变量会改变默认，写死会让测试因
    // 无关配置偶发失败（OCR #41）。
    // EN: Expect the created collection's effective params instead of literals
    // 16/400/256: those derive from `VectorConfig/HnswConfig::from_env()`, so
    // `HOUYI_VECTOR_AUTO_HNSW`, `HOUYI_VECTOR_RECALL_OPTIMIZED` etc. shift the
    // defaults and literals fail for unrelated config reasons (OCR #41).
    let col = pg
        .collections
        .get("pgw")
        .expect("implicit create must register pgw");
    let params = col.hnsw_params();
    assert_eq!(
        first,
        Some(CollectionWalRecord::Create {
            name: "pgw".to_string(),
            dims: 4,
            index_type: "hnsw".to_string(),
            options: houyi_persistence::collection_wal::CollectionCreateOptions {
                metric: Some(col.metric().as_str().to_string()),
                hnsw_m: Some(params.m),
                hnsw_ef_construction: Some(params.ef_construction),
                hnsw_ef_search: Some(params.ef_search),
            },
        })
    );
    match second {
        Some(CollectionWalRecord::UpsertBatch { name, dim, items }) => {
            assert_eq!(name, "pgw");
            assert_eq!(dim, 4);
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].id, 11);
            assert_eq!(items[1].id, 12);
        }
        other => panic!("expected upsert batch record, got {other:?}"),
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&collections_path);
    let _ = std::fs::remove_file(houyi_persistence::wal_sidecar::wal_next_lsn_meta_path(
        &path,
    ));
    let _ = std::fs::remove_file(houyi_persistence::wal_sidecar::wal_next_lsn_meta_path(
        &collections_path,
    ));
}

#[tokio::test]
async fn pgwire_bind_with_shared_text_param_format_succeeds() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql =
        b"\0SELECT id FROM docs WHERE id IN ($1,$2)\0\x00\x02\x00\x00\x00\x19\x00\x00\x00\x19";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&1_i16.to_be_bytes()); // one shared param format
    bind_body.extend_from_slice(&0_i16.to_be_bytes()); // text
    bind_body.extend_from_slice(&2_i16.to_be_bytes()); // two params
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"9");
    bind_body.extend_from_slice(&0_i16.to_be_bytes()); // result fmt count
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();

    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');
}

#[tokio::test]
async fn pgwire_close_statement_clears_statement_and_portal_state() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql = b"\0SELECT id FROM docs WHERE id = $1\0\x00\x01\x00\x00\x00\x19";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let close_stmt_body = b"S\0";
    let mut close_stmt = Vec::new();
    close_stmt.push(b'C');
    close_stmt.extend_from_slice(&((4 + close_stmt_body.len()) as i32).to_be_bytes());
    close_stmt.extend_from_slice(close_stmt_body);
    stream.write_all(&close_stmt).await.unwrap();

    let mut cc = [0_u8; 5];
    stream.read_exact(&mut cc).await.unwrap();
    assert_eq!(cc[0], b'3');

    // Describe(S) 现在应因语句缺失而失败。
    // EN: Describe(S) should now fail with missing statement.
    let desc_stmt_body = b"S\0";
    let mut desc_stmt = Vec::new();
    desc_stmt.push(b'D');
    desc_stmt.extend_from_slice(&((4 + desc_stmt_body.len()) as i32).to_be_bytes());
    desc_stmt.extend_from_slice(desc_stmt_body);
    stream.write_all(&desc_stmt).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(err_body.windows(7).any(|w| w == b"C26000\0"));

    // portal 已消失，Execute 也应失败。
    // EN: Execute should also fail because the portal is gone.
    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut err2_header = [0_u8; 5];
    stream.read_exact(&mut err2_header).await.unwrap();
    assert_eq!(err2_header[0], b'E');
    let err2_len = i32::from_be_bytes(err2_header[1..5].try_into().unwrap()) as usize;
    let mut err2_body = vec![0_u8; err2_len - 4];
    stream.read_exact(&mut err2_body).await.unwrap();
    assert!(err2_body.windows(7).any(|w| w == b"C34000\0"));
}

#[tokio::test]
async fn pgwire_close_portal_preserves_statement_but_clears_portal_state() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let parse_sql = b"\0SELECT id FROM docs WHERE id = $1\0\x00\x01\x00\x00\x00\x19";
    let mut parse = Vec::new();
    parse.push(b'P');
    parse.extend_from_slice(&((4 + parse_sql.len()) as i32).to_be_bytes());
    parse.extend_from_slice(parse_sql);
    stream.write_all(&parse).await.unwrap();
    let mut pc = [0_u8; 5];
    stream.read_exact(&mut pc).await.unwrap();
    assert_eq!(pc[0], b'1');

    let mut bind_body = Vec::new();
    bind_body.extend_from_slice(b"\0\0");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i16.to_be_bytes());
    bind_body.extend_from_slice(&1_i32.to_be_bytes());
    bind_body.extend_from_slice(b"7");
    bind_body.extend_from_slice(&0_i16.to_be_bytes());
    let mut bind = Vec::new();
    bind.push(b'B');
    bind.extend_from_slice(&((4 + bind_body.len()) as i32).to_be_bytes());
    bind.extend_from_slice(&bind_body);
    stream.write_all(&bind).await.unwrap();
    let mut bc = [0_u8; 5];
    stream.read_exact(&mut bc).await.unwrap();
    assert_eq!(bc[0], b'2');

    let close_portal_body = b"P\0";
    let mut close_portal = Vec::new();
    close_portal.push(b'C');
    close_portal.extend_from_slice(&((4 + close_portal_body.len()) as i32).to_be_bytes());
    close_portal.extend_from_slice(close_portal_body);
    stream.write_all(&close_portal).await.unwrap();

    let mut cc = [0_u8; 5];
    stream.read_exact(&mut cc).await.unwrap();
    assert_eq!(cc[0], b'3');

    // 语句仍在，Describe(S) 仍应成功。
    // EN: Describe(S) should still work because the statement remains.
    let desc_stmt_body = b"S\0";
    let mut desc_stmt = Vec::new();
    desc_stmt.push(b'D');
    desc_stmt.extend_from_slice(&((4 + desc_stmt_body.len()) as i32).to_be_bytes());
    desc_stmt.extend_from_slice(desc_stmt_body);
    stream.write_all(&desc_stmt).await.unwrap();

    let mut param_desc_header = [0_u8; 5];
    stream.read_exact(&mut param_desc_header).await.unwrap();
    assert_eq!(param_desc_header[0], b't');
    let param_desc_len = i32::from_be_bytes(param_desc_header[1..5].try_into().unwrap()) as usize;
    let mut param_desc_body = vec![0_u8; param_desc_len - 4];
    stream.read_exact(&mut param_desc_body).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    assert_eq!(row_desc_header[0], b'T');
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();

    // portal 已被显式关闭，Execute 应失败。
    // EN: Execute should fail because the portal was explicitly closed.
    let mut ex = Vec::new();
    ex.push(b'E');
    let mut ex_body = vec![0_u8];
    ex_body.extend_from_slice(&(-1_i32).to_be_bytes());
    ex.extend_from_slice(&((4 + ex_body.len()) as i32).to_be_bytes());
    ex.extend_from_slice(&ex_body);
    stream.write_all(&ex).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(err_body.windows(7).any(|w| w == b"C34000\0"));

    // 语句仍存在，Re-Bind 应成功。
    // EN: Re-Bind should succeed because the statement still exists.
    stream.write_all(&bind).await.unwrap();
    let mut bc2 = [0_u8; 5];
    stream.read_exact(&mut bc2).await.unwrap();
    assert_eq!(bc2[0], b'2');
}

#[tokio::test]
async fn benchmark_pgwire_status_reports_simple_query_traffic() {
    use axum::{body::Body, http::Request, routing::get, Router};
    use tower::ServiceExt;

    let metrics = Arc::new(PgWireMetrics::default());
    let pgwire_addr = run_pgwire_listener_with_metrics("127.0.0.1:0", None, Arc::clone(&metrics))
        .await
        .unwrap();
    let mut stream = TcpStream::connect(pgwire_addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let query_sql = b"SELECT 1\0";
    let q_len = (query_sql.len() + 4) as i32;
    let mut q = Vec::new();
    q.push(b'Q');
    q.extend_from_slice(&q_len.to_be_bytes());
    q.extend_from_slice(query_sql);
    stream.write_all(&q).await.unwrap();

    let mut row_desc_header = [0_u8; 5];
    stream.read_exact(&mut row_desc_header).await.unwrap();
    let row_desc_len = i32::from_be_bytes(row_desc_header[1..5].try_into().unwrap()) as usize;
    let mut row_desc_body = vec![0_u8; row_desc_len - 4];
    stream.read_exact(&mut row_desc_body).await.unwrap();
    let mut data_row_header = [0_u8; 5];
    stream.read_exact(&mut data_row_header).await.unwrap();
    let data_row_len = i32::from_be_bytes(data_row_header[1..5].try_into().unwrap()) as usize;
    let mut data_row_body = vec![0_u8; data_row_len - 4];
    stream.read_exact(&mut data_row_body).await.unwrap();
    let mut complete_header = [0_u8; 5];
    stream.read_exact(&mut complete_header).await.unwrap();
    let complete_len = i32::from_be_bytes(complete_header[1..5].try_into().unwrap()) as usize;
    let mut complete_body = vec![0_u8; complete_len - 4];
    stream.read_exact(&mut complete_body).await.unwrap();
    let mut ready2 = [0_u8; 6];
    stream.read_exact(&mut ready2).await.unwrap();

    let path = std::env::temp_dir().join(format!(
        "houyidb-pgwire-metrics-test-{}.wal",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let (wal, _) = super::WalHandle::open(&path, super::WalWriteMode::Async).unwrap();
    let app = Router::new()
        .route("/v1/pgwire/status", get(benchmark_pgwire_status))
        .with_state(BenchmarkHttpState {
            audit: None,
            encryption: None,
            promote: None,
            collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
            index: IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4))),
            wal: Arc::new(wal),
            collections_wal: Arc::new(
                super::WalHandle::open(
                    &houyi_persistence::collection_wal::collections_wal_path_for(&path),
                    super::WalWriteMode::Async,
                )
                .unwrap()
                .0,
            ),
            vector_dim: 4,
            pgwire_metrics: Arc::clone(&metrics),
            applied_sidecar: path.with_extension("applied_lsn"),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path: path.clone(),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
            graph_snapshot_metrics: Arc::new(
                houyi_server::recovery::GraphSnapshotMetrics::default(),
            ),
            role: houyi_server::types::RoleState::default(),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: houyi_server::types::shared_replication_metrics(None),
            lifecycle: std::sync::Arc::new(
                houyi_server::collection_lifecycle::CollectionLifecycleState::for_test(),
            ),
        });
    let req = Request::builder()
        .method("GET")
        .uri("/v1/pgwire/status")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: super::PgWireMetricsSnapshot = serde_json::from_slice(&body).unwrap();

    assert_eq!(body.handshake_count, 1);
    assert_eq!(body.simple_query_count, 1);
    assert!(body.frontend_messages >= 1);
    assert!(body.sql_text_bytes >= "SELECT 1".len() as u64);
    assert_eq!(body.upsert_batch_execute_count, 0);
    assert_eq!(body.upsert_batch_rows_inserted, 0);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(houyi_persistence::collection_wal::collections_wal_path_for(
        &path,
    ));
}

#[tokio::test]
async fn pgwire_simple_query_copy_sql_returns_feature_not_supported() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let query_sql = b"COPY docs FROM STDIN\0";
    let q_len = (query_sql.len() + 4) as i32;
    let mut q = Vec::new();
    q.push(b'Q');
    q.extend_from_slice(&q_len.to_be_bytes());
    q.extend_from_slice(query_sql);
    stream.write_all(&q).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(
        err_body.windows(7).any(|w| w == b"C0A000\0"),
        "expected SQLSTATE 0A000 in error body, got {:?}",
        err_body
    );

    let mut ready2 = [0_u8; 6];
    stream.read_exact(&mut ready2).await.unwrap();
    assert_eq!(ready2[0], b'Z');
}

#[tokio::test]
async fn pgwire_copydata_returns_feature_not_supported() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let mut copy_data = Vec::new();
    copy_data.push(b'd');
    copy_data.extend_from_slice(&4_i32.to_be_bytes());
    stream.write_all(&copy_data).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(
        err_body.windows(7).any(|w| w == b"C0A000\0"),
        "expected SQLSTATE 0A000 in error body, got {:?}",
        err_body
    );
}

#[tokio::test]
async fn pgwire_copyfail_returns_feature_not_supported_with_reason() {
    let addr = run_pgwire_listener("127.0.0.1:0", None).await.unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let mut payload = Vec::new();
    payload.extend_from_slice(&196608_i32.to_be_bytes());
    payload.extend_from_slice(b"user\0tester\0database\0houyi\0\0");
    let len = (payload.len() + 4) as i32;
    let mut startup = Vec::new();
    startup.extend_from_slice(&len.to_be_bytes());
    startup.extend_from_slice(&payload);
    stream.write_all(&startup).await.unwrap();

    let mut auth = [0_u8; 9];
    stream.read_exact(&mut auth).await.unwrap();
    let mut ps1_header = [0_u8; 5];
    stream.read_exact(&mut ps1_header).await.unwrap();
    let ps1_len = i32::from_be_bytes(ps1_header[1..5].try_into().unwrap()) as usize;
    let mut ps1_body = vec![0_u8; ps1_len - 4];
    stream.read_exact(&mut ps1_body).await.unwrap();
    let mut ps2_header = [0_u8; 5];
    stream.read_exact(&mut ps2_header).await.unwrap();
    let ps2_len = i32::from_be_bytes(ps2_header[1..5].try_into().unwrap()) as usize;
    let mut ps2_body = vec![0_u8; ps2_len - 4];
    stream.read_exact(&mut ps2_body).await.unwrap();
    let mut ready = [0_u8; 6];
    stream.read_exact(&mut ready).await.unwrap();

    let reason = b"client aborted copy\0";
    let mut copy_fail = Vec::new();
    copy_fail.push(b'f');
    copy_fail.extend_from_slice(&((4 + reason.len()) as i32).to_be_bytes());
    copy_fail.extend_from_slice(reason);
    stream.write_all(&copy_fail).await.unwrap();

    let mut err_header = [0_u8; 5];
    stream.read_exact(&mut err_header).await.unwrap();
    assert_eq!(err_header[0], b'E');
    let err_len = i32::from_be_bytes(err_header[1..5].try_into().unwrap()) as usize;
    let mut err_body = vec![0_u8; err_len - 4];
    stream.read_exact(&mut err_body).await.unwrap();
    assert!(
        err_body.windows(7).any(|w| w == b"C0A000\0"),
        "expected SQLSTATE 0A000 in error body, got {:?}",
        err_body
    );
    assert!(
        err_body
            .windows("client aborted copy".len())
            .any(|w| w == b"client aborted copy"),
        "expected copy fail reason in error body, got {:?}",
        err_body
    );
}
// ── Task 2: Collection 生产写入路径（label + payload + 内存标量页）──────────
// EN: Task 2 regression tests — collection upsert stores label/payload in
// a per-collection in-memory PAX scalar page, and hybrid search filters on it.
use axum::extract::Path;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;

/// 临时文件守卫：断言失败提前 unwind 时也删除 WAL/sidecar，避免 /tmp 泄漏。
/// Drop guard for temp WAL/sidecar files: cleanup still runs when an assert fails.
struct TempFileCleanup(Vec<std::path::PathBuf>);
impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        for path in &self.0 {
            if path.is_dir() {
                let _ = std::fs::remove_dir_all(path);
            } else {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

fn collection_test_state(dim: usize) -> (BenchmarkHttpState, TempFileCleanup) {
    collection_test_state_with_index(dim, "hnsw")
}

/// 构造指定 Collection 默认索引类型的测试态（flat/clustered 回归用）。
/// Test state with a chosen collection default index type (flat/clustered regressions).
fn collection_test_state_with_index(
    dim: usize,
    index_type: &str,
) -> (BenchmarkHttpState, TempFileCleanup) {
    collection_test_state_full(dim, index_type, None)
}

/// 完整测试态构造：可注入 Collection 异步构图配置（模拟生产 worker 行为）。
/// Full test-state constructor: optionally inject the collection builder config
/// (mirrors the production worker wiring).
fn collection_test_state_full(
    dim: usize,
    index_type: &str,
    builder_cfg: Option<houyi_server::collection::CollectionBuilderConfig>,
) -> (BenchmarkHttpState, TempFileCleanup) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    // 每个测试独立目录：snapshots/ 由 WAL 所在父目录派生，共用 /tmp 会让
    // 并行测试互相看到对方的 manifest（P5.0.3 status 测试暴露的隔离缺陷）。
    // EN: One directory per test: snapshots/ derives from the WAL parent, so a
    // shared /tmp lets parallel tests see each other's manifests (isolation bug
    // exposed by the P5.0.3 status test).
    let dir = std::env::temp_dir().join(format!(
        "houyidb-coll-test-{}-{}-{}",
        std::process::id(),
        nanos,
        seq
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let wal_path = dir.join("bench.wal");
    let sidecar = wal_path.with_extension("applied");
    let wal = Arc::new(WalHandle::open(&wal_path, WalWriteMode::Async).unwrap().0);
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
    let collections_wal = Arc::new(
        WalHandle::open(&collections_wal_path, WalWriteMode::Async)
            .unwrap()
            .0,
    );
    let manager = match builder_cfg {
        Some(cfg) => CollectionManager::new(dim, index_type).with_builder_config(cfg),
        None => CollectionManager::new(dim, index_type),
    };
    let state = BenchmarkHttpState {
        audit: None,
        encryption: None,
        promote: None,
        index: IndexBackend::new_hnsw(dim),
        collections: Arc::new(manager),
        wal,
        collections_wal,
        vector_dim: dim,
        pgwire_metrics: Arc::new(PgWireMetrics::default()),
        applied_sidecar: sidecar.clone(),
        applied_sidecar_lock: Arc::new(Mutex::new(())),
        wal_path: wal_path.clone(),
        label_index: Arc::new(RwLock::new(HashMap::new())),
        openai_upstream: None,
        http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
        graph_snapshot_metrics: Arc::new(houyi_server::recovery::GraphSnapshotMetrics::default()),
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        lifecycle: std::sync::Arc::new(
            houyi_server::collection_lifecycle::CollectionLifecycleState::for_test(),
        ),
    };
    (state, TempFileCleanup(vec![dir]))
}

fn json_payload(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    v.as_object()
        .expect("payload must be a JSON object")
        .clone()
}

async fn collection_upsert(
    state: &BenchmarkHttpState,
    name: &str,
    vectors: Vec<houyi_server::types::UpsertVector>,
) {
    use houyi_server::http_handler::handle_collection_upsert;
    let expected = vectors.len();
    let req = houyi_server::types::UpsertRequest {
        vectors,
        index: None,
    };
    let resp = handle_collection_upsert(State(state.clone()), Path(name.to_string()), Json(req))
        .await
        .expect("collection upsert should succeed");
    assert_eq!(resp.inserted, expected);

    // 生产环境由异步 builder worker 持续把 builder_queue 刷入主图；
    // 直调 handler 的单测没有 worker，这里同步排空，保证写入对检索可见。
    // EN: Production drains builder_queue via the async worker; unit tests call
    // handlers directly, so flush synchronously to make the batch searchable.
    if let Some(col) = state.collections.get(name) {
        for _ in 0..1000 {
            if col.backend.run_async_builder_once().await.flushed == 0 {
                break;
            }
        }
    }
}

async fn collection_hybrid(
    state: &BenchmarkHttpState,
    name: &str,
    query: Vec<f32>,
    k: usize,
    filters: Vec<(&str, &str, serde_json::Value)>,
) -> Vec<u64> {
    use houyi_server::http_handler::{handle_collection_hybrid_search, HybridSearchRequest};
    use houyi_server::types::HybridSearchFilter;
    let req = HybridSearchRequest {
        vector: query,
        include_payload: false,
        k,
        ef_search: None,
        filters: filters
            .into_iter()
            .map(|(column, op, value)| HybridSearchFilter {
                column: column.to_string(),
                op: op.to_string(),
                value,
            })
            .collect(),
    };
    handle_collection_hybrid_search(
        State(state.clone()),
        Path(name.to_string()),
        None,
        Json(req),
    )
    .await
    .expect("hybrid search should succeed")
    .0
    .ids
}

/// 与 [`collection_hybrid`] 相同，但保留 Result 以便断言 400/404。
/// Same as `collection_hybrid` but keeps the Result for 400/404 assertions.
async fn collection_hybrid_result(
    state: &BenchmarkHttpState,
    name: &str,
    query: Vec<f32>,
    k: usize,
    filters: Vec<(&str, &str, serde_json::Value)>,
) -> Result<houyi_server::types::SearchResponse, (StatusCode, String)> {
    use houyi_server::http_handler::{handle_collection_hybrid_search, HybridSearchRequest};
    use houyi_server::types::HybridSearchFilter;
    let req = HybridSearchRequest {
        vector: query,
        include_payload: false,
        k,
        ef_search: None,
        filters: filters
            .into_iter()
            .map(|(column, op, value)| HybridSearchFilter {
                column: column.to_string(),
                op: op.to_string(),
                value,
            })
            .collect(),
    };
    handle_collection_hybrid_search(
        State(state.clone()),
        Path(name.to_string()),
        None,
        Json(req),
    )
    .await
    .map(|Json(resp)| resp)
}

// ── collection 删除 / include_payload ───────────────────
// EN: collection delete / include_payload.

/// 直调 collection delete handler（保留 Result 以断言 400/404）。
/// Call the collection delete handler directly (Result kept for 400/404 assertions).
async fn collection_delete_call(
    state: &BenchmarkHttpState,
    name: &str,
    ids: Vec<u64>,
) -> Result<houyi_server::types::DeleteResponse, (StatusCode, String)> {
    collection_delete_full(state, name, ids, Vec::new()).await
}

/// 直调 collection delete handler（可同时带 filters，覆盖 delete-by-filter 路径）。
/// Call the collection delete handler with optional filters (delete-by-filter path).
async fn collection_delete_full(
    state: &BenchmarkHttpState,
    name: &str,
    ids: Vec<u64>,
    filters: Vec<houyi_server::types::HybridSearchFilter>,
) -> Result<houyi_server::types::DeleteResponse, (StatusCode, String)> {
    use houyi_server::http_handler::handle_collection_delete;
    use houyi_server::types::DeleteRequest;
    handle_collection_delete(
        State(state.clone()),
        Path(name.to_string()),
        Json(DeleteRequest { ids, filters }),
    )
    .await
    .map(|Json(resp)| resp)
}

/// 直调 collection search handler 并指定 include_payload。
/// Call the collection search handler with an explicit include_payload flag.
async fn collection_search_with_payload(
    state: &BenchmarkHttpState,
    name: &str,
    vector: Vec<f32>,
    k: usize,
    include_payload: bool,
) -> houyi_server::types::SearchResponse {
    use houyi_server::http_handler::handle_collection_search;
    let req = houyi_server::types::SearchRequest {
        vector,
        k,
        filter_gt_id: None,
        ef_search: None,
        filter_label: None,
        include_payload,
    };
    handle_collection_search(
        State(state.clone()),
        Path(name.to_string()),
        None,
        Json(req),
    )
    .await
    .expect("collection search should succeed")
    .0
}

fn three_docs() -> Vec<houyi_server::types::UpsertVector> {
    use houyi_server::types::UpsertVector;
    vec![
        UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("fruit".to_string()),
            payload: Some(json_payload(
                serde_json::json!({"name": "apple", "price": 3}),
            )),
        },
        UpsertVector {
            id: 2,
            values: vec![0.0, 1.0, 0.0, 0.0],
            label: Some("electronics".to_string()),
            payload: Some(json_payload(
                serde_json::json!({"name": "phone", "price": 4999}),
            )),
        },
        UpsertVector {
            id: 3,
            values: vec![0.0, 0.0, 1.0, 0.0],
            label: None,
            payload: Some(json_payload(
                serde_json::json!({"name": "novel", "price": 45}),
            )),
        },
    ]
}

/// delete：命中从检索消失、deleted 计数真实、重复删除幂等（0）。
/// EN: delete removes hits, reports the true count, and is idempotent.
#[tokio::test]
async fn collection_delete_removes_hits_and_is_idempotent() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(&state, "c", three_docs()).await;

    let resp = collection_delete_call(&state, "c", vec![1, 2, 999])
        .await
        .expect("delete should succeed");
    assert_eq!(resp.deleted, 2, "only existing ids count; 999 skipped");

    let found = collection_search(&state, "c", vec![1.0, 0.0, 0.0, 0.0], 3, None).await;
    assert!(!found.ids.contains(&1), "deleted id=1 must be invisible");
    let found = collection_search(&state, "c", vec![0.0, 1.0, 0.0, 0.0], 3, None).await;
    assert!(!found.ids.contains(&2), "deleted id=2 must be invisible");
    let found = collection_search(&state, "c", vec![0.0, 0.0, 1.0, 0.0], 3, None).await;
    assert_eq!(found.ids, vec![3], "id=3 survives");

    let again = collection_delete_call(&state, "c", vec![1, 2])
        .await
        .expect("re-delete should succeed");
    assert_eq!(again.deleted, 0, "idempotent re-delete counts nothing");
}

/// delete 后混合检索的过滤命中也不再返回该 id（标量墓碑生效）。
/// EN: after delete, filtered hybrid search no longer surfaces the id (scalar tombstone).
#[tokio::test]
async fn collection_delete_excludes_id_from_hybrid_filters() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(&state, "c", three_docs()).await;
    collection_delete_call(&state, "c", vec![2]).await.unwrap();

    let hits = collection_hybrid(
        &state,
        "c",
        vec![0.0, 1.0, 0.0, 0.0],
        10,
        vec![("label", "eq", serde_json::json!("electronics"))],
    )
    .await;
    assert!(
        hits.is_empty(),
        "deleted id must not match filters: {hits:?}"
    );
}

/// include_payload=true：响应携带 label + payload；=false：无 payloads 字段。
/// EN: include_payload=true attaches label+payload; false keeps the field absent.
#[tokio::test]
async fn collection_search_include_payload_round_trip() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(&state, "c", three_docs()).await;

    let resp = collection_search_with_payload(&state, "c", vec![1.0, 0.0, 0.0, 0.0], 2, true).await;
    let payloads = resp
        .payloads
        .expect("payloads must be present when requested");
    let doc1 = payloads.get("1").expect("id=1 payload");
    assert_eq!(doc1["label"], serde_json::json!("fruit"));
    assert_eq!(doc1["name"], serde_json::json!("apple"));
    assert_eq!(doc1["price"], serde_json::json!(3));

    let plain = collection_search(&state, "c", vec![1.0, 0.0, 0.0, 0.0], 2, None).await;
    assert!(
        plain.payloads.is_none(),
        "payloads must stay absent without the flag (response shape unchanged)"
    );
}

/// 端到端「宕机重启」回归（本修复的核心场景）：collections 路径 upsert/delete 全链路
/// WAL-first 落盘 → 丢弃全部内存态（模拟进程崩溃）→ 仅从 collections WAL 重放 →
/// 向量可检索、payload 原样回读、删除语义保持。修复前这条链路上数据全丢。
/// EN: End-to-end crash-restart regression (the core scenario of this fix): the
/// collections upsert/delete path persists WAL-first; after dropping all in-memory
/// state (simulated crash) a cold replay from the collections WAL alone restores
/// searchable vectors, payload read-back, and delete semantics. Before the fix
/// everything on this path was lost.
#[tokio::test]
async fn collections_restart_recovers_vectors_payloads_and_deletes() {
    let (state, _cleanup) = collection_test_state(4);
    let wal_path = state.wal_path.clone();
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
    collection_upsert(&state, "restart", three_docs()).await;
    collection_delete_call(&state, "restart", vec![2])
        .await
        .unwrap();
    // 模拟宕机：进程内存态（Manager、WAL 句柄）整体丢弃。
    // EN: Simulated crash — every in-memory handle (manager, WAL) is dropped.
    drop(state);

    // 冷启动：仅由 collections WAL 重建（与 main 装配同一入口）。
    // EN: Cold start — rebuild solely from the collections WAL (same entry as main).
    let (_w, entries) = houyi_persistence::FileWal::open(&collections_wal_path).unwrap();
    let manager = std::sync::Arc::new(houyi_server::collection::CollectionManager::new(4, "hnsw"));
    let stats = houyi_server::recovery::replay_collections_wal_into_manager(
        &entries,
        &manager,
        &std::collections::HashMap::new(),
    )
    .await
    .expect("cold replay must succeed");
    assert_eq!(stats.creates, 1);
    assert_eq!(stats.upsert_rows, 3);
    assert_eq!(stats.scalar_rows, 3);
    assert_eq!(stats.tombstones, 1);

    // 用重放后的 Manager 装配一个只读视角 state，走公开检索 handler 验证。
    // EN: Assemble a read-side state over the replayed manager and verify through
    // the public search handler.
    let wal = std::sync::Arc::new(
        super::WalHandle::open(&wal_path, super::WalWriteMode::Async)
            .unwrap()
            .0,
    );
    let collections_wal = std::sync::Arc::new(
        super::WalHandle::open(&collections_wal_path, super::WalWriteMode::Async)
            .unwrap()
            .0,
    );
    let state2 = BenchmarkHttpState {
        audit: None,
        encryption: None,
        promote: None,
        index: IndexBackend::new_hnsw(4),
        collections: manager,
        wal,
        collections_wal,
        vector_dim: 4,
        pgwire_metrics: std::sync::Arc::new(PgWireMetrics::default()),
        applied_sidecar: wal_path.with_extension("applied"),
        applied_sidecar_lock: std::sync::Arc::new(Mutex::new(())),
        wal_path: wal_path.clone(),
        label_index: std::sync::Arc::new(RwLock::new(std::collections::HashMap::new())),
        openai_upstream: None,
        http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
        graph_snapshot_metrics: Arc::new(houyi_server::recovery::GraphSnapshotMetrics::default()),
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        lifecycle: std::sync::Arc::new(
            houyi_server::collection_lifecycle::CollectionLifecycleState::for_test(),
        ),
    };
    let resp =
        collection_search_with_payload(&state2, "restart", vec![1.0, 0.0, 0.0, 0.0], 5, true).await;
    assert!(
        resp.ids.contains(&1),
        "id 1 survives restart: {:?}",
        resp.ids
    );
    assert!(
        resp.ids.contains(&3),
        "id 3 survives restart: {:?}",
        resp.ids
    );
    assert!(
        !resp.ids.contains(&2),
        "pre-crash delete must survive restart: {:?}",
        resp.ids
    );
    let payloads = resp.payloads.expect("payloads after restart");
    assert_eq!(payloads["1"]["name"], serde_json::json!("apple"));
    assert_eq!(payloads["1"]["label"], serde_json::json!("fruit"));
    assert!(
        !payloads.contains_key("2"),
        "deleted id must not read back after restart"
    );
}

/// ha_level 端到端「宕机重启」（P4.4.2）：端点设置 WAL-first 落盘 → 丢弃全部内存态
/// → 冷重放恢复级别；从未设置过的 collection 恢复为默认 async；指向不存在
/// collection 的 set 记录（历史截断场景）重放告警跳过而非卡死启动。
/// EN: Crash-restart restores the ha_level set via the endpoint (WAL-first);
/// untouched collections come back at the default async; a set record naming a
/// missing collection is skipped with a warning, not a startup failure.
#[tokio::test]
async fn collections_restart_recovers_ha_level() {
    let (state, _cleanup) = collection_test_state(4);
    let wal_path = state.wal_path.clone();
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
    collection_upsert(&state, "docs", three_docs()).await;
    collection_upsert(&state, "plain", three_docs()).await;
    use houyi_server::http_handler::handle_collection_set_ha_level;
    let resp = handle_collection_set_ha_level(
        State(state.clone()),
        Path("docs".to_string()),
        Json(houyi_server::types::SetHaLevelRequest {
            ha_level: "full_sync".to_string(),
        }),
    )
    .await
    .expect("set ha_level must succeed")
    .0;
    assert_eq!(resp.ha_level, "full_sync");
    // 模拟宕机：进程内存态整体丢弃（WAL 句柄随 drop 收尾落盘）。
    // EN: Simulated crash: drop all in-process memory state (the WAL handle flushes on drop).
    drop(state);

    let (_w, entries) = houyi_persistence::FileWal::open(&collections_wal_path).unwrap();
    // 追加一条指向不存在 collection 的 set 记录（模拟历史被截断后的残留）：
    // 重放必须跳过它并照常完成，不得把启动卡死在元数据畸形上。
    // EN: Append a set record pointing at a nonexistent collection (leftover from truncated
    // EN: history): replay must skip it and finish normally, not stall startup on bad metadata.
    let mut entries = entries;
    entries.push(houyi_persistence::WalEntry {
        lsn: entries.last().map(|e| e.lsn + 1).unwrap_or(1),
        payload: houyi_persistence::collection_wal::encode_collection_set_ha_level(
            "ghost",
            houyi_persistence::ha_level::HaLevel::SemiSync,
        )
        .unwrap(),
    });
    let manager = std::sync::Arc::new(houyi_server::collection::CollectionManager::new(4, "hnsw"));
    houyi_server::recovery::replay_collections_wal_into_manager(
        &entries,
        &manager,
        &std::collections::HashMap::new(),
    )
    .await
    .expect("cold replay must succeed");
    assert_eq!(
        manager.get("docs").unwrap().ha_level(),
        houyi_persistence::ha_level::HaLevel::FullSync,
        "level set pre-crash must survive restart"
    );
    assert_eq!(
        manager.get("plain").unwrap().ha_level(),
        houyi_persistence::ha_level::HaLevel::Async,
        "untouched collection keeps the default"
    );
}

/// hybrid 检索同样支持 include_payload；被删除的 id 不回读 payload。
/// EN: hybrid search honors include_payload too; deleted ids have no payload.
#[tokio::test]
async fn collection_hybrid_include_payload_and_delete_tombstone() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(&state, "c", three_docs()).await;
    collection_delete_call(&state, "c", vec![1]).await.unwrap();

    use houyi_server::http_handler::{handle_collection_hybrid_search, HybridSearchRequest};
    use houyi_server::types::HybridSearchFilter;
    let req = HybridSearchRequest {
        vector: vec![1.0, 0.0, 0.0, 0.0],
        k: 3,
        include_payload: true,
        ef_search: None,
        filters: vec![HybridSearchFilter {
            column: "price".to_string(),
            op: "gt".to_string(),
            value: serde_json::json!(1),
        }],
    };
    let resp = handle_collection_hybrid_search(
        State(state.clone()),
        Path("c".to_string()),
        None,
        Json(req),
    )
    .await
    .expect("hybrid search should succeed")
    .0;
    assert!(!resp.ids.contains(&1));
    let payloads = resp.payloads.expect("payloads present");
    assert!(!payloads.contains_key("1"), "deleted id must not read back");
    assert_eq!(payloads["2"]["name"], serde_json::json!("phone"));
}

/// 删除后携带新 payload 重写同一 id：复活且回读为最新值（last-wins）。
/// EN: re-upsert with fresh payload after delete resurrects; read-back is last-wins.
#[tokio::test]
async fn collection_reupsert_after_delete_resurrects_with_latest_payload() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(&state, "c", three_docs()).await;
    collection_delete_call(&state, "c", vec![1]).await.unwrap();

    collection_upsert(
        &state,
        "c",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(
                serde_json::json!({"name": "apple-pro", "price": 9}),
            )),
        }],
    )
    .await;

    let resp = collection_search_with_payload(&state, "c", vec![1.0, 0.0, 0.0, 0.0], 1, true).await;
    assert_eq!(resp.ids, vec![1], "resurrected id searchable again");
    let payloads = resp.payloads.expect("payloads present");
    assert_eq!(
        payloads["1"]["name"],
        serde_json::json!("apple-pro"),
        "read-back must be the latest row"
    );
    assert!(
        payloads["1"].get("label").is_none(),
        "stale label from the superseded row must not leak into the latest row"
    );
}

/// 不支持删除的后端（clustered）fail-fast 400，且标量页不被半删除污染。
/// EN: backends without delete support (clustered) fail fast with 400 and the
/// scalar page is not polluted by a half-delete.
#[tokio::test]
async fn collection_delete_unsupported_backend_returns_400() {
    let (state, _cleanup) = collection_test_state_with_index(4, "clustered");
    collection_upsert(&state, "c", three_docs()).await;
    let err = collection_delete_call(&state, "c", vec![1])
        .await
        .expect_err("clustered backend must refuse delete");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err.1.contains("unsupported"), "got: {}", err.1);
    // 半删除防护：clustered 需 optimize_build 后才可检索；构图后验证「向量仍在 +
    // payload 仍可回读（标量墓碑未登记）」。
    // EN: Half-delete guard: clustered rows need optimize_build before search; after building,
    // EN: verify the vector is still present and the payload still reads back (no scalar
    // EN: tombstone).
    let col = state.collections.get("c").expect("collection exists");
    assert_eq!(col.backend.optimize_build().await, 3);
    let resp = collection_search_with_payload(&state, "c", vec![1.0, 0.0, 0.0, 0.0], 1, true).await;
    assert_eq!(
        resp.ids,
        vec![1],
        "refused delete must leave the vector intact"
    );
    let payloads = resp.payloads.expect("payloads present");
    assert_eq!(
        payloads["1"]["name"],
        serde_json::json!("apple"),
        "scalar page must stay intact after a refused delete"
    );
}

/// 回归（WAL 投毒）：不支持删除的后端 delete-by-ids 必须 400 且不得在 collections
/// WAL 留下 collection_delete 记录。修复前 WAL append 先于 backend.delete——400
/// 拒绝的请求已落盘，而重放对 backend.delete 的 Err fail-fast（不变式：Err 即后端
/// 从不支持 delete，这种记录不可能被写出），一条坏请求让服务永远无法冷启动。
/// EN: Regression (WAL poisoning): a refused delete on an unsupported backend must
/// not leave a collection_delete record in the WAL. Before the fix the WAL append
/// preceded backend.delete, so a 400-rejected request was persisted — and replay
/// fails fast on the backend Err, bricking cold start.
#[tokio::test]
async fn collection_delete_unsupported_backend_leaves_no_wal_record() {
    let (state, _cleanup) = collection_test_state_with_index(4, "clustered");
    let wal_path = state.wal_path.clone();
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
    collection_upsert(&state, "c", three_docs()).await;
    let err = collection_delete_call(&state, "c", vec![1])
        .await
        .expect_err("clustered backend must refuse delete");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    drop(state);

    // 冷启动重放必须成功，且恢复出全部 3 条向量（无任何 delete 生效）。
    // EN: Cold replay must succeed and restore all 3 vectors (no delete took effect).
    let (_w, entries) = houyi_persistence::FileWal::open(&collections_wal_path).unwrap();
    let manager = std::sync::Arc::new(houyi_server::collection::CollectionManager::new(
        4,
        "clustered",
    ));
    let stats = houyi_server::recovery::replay_collections_wal_into_manager(
        &entries,
        &manager,
        &std::collections::HashMap::new(),
    )
    .await
    .expect("cold replay must not be poisoned by a refused delete");
    assert_eq!(stats.upsert_rows, 3);
    assert_eq!(
        stats.tombstones, 0,
        "a 400-rejected delete must not persist a tombstone record"
    );
    assert_eq!(stats.delete_batches, 0);
}

/// 回归（WAL/内存顺序分叉）：并发同 id upsert/delete 下，冷启动重放终态必须等于
/// 内存已确认终态。拆锁版本有两类分叉：delete 记录可先于 upsert 落 WAL 而墓碑晚于
/// upsert 标量行（重放复活崩溃前已确认删除的 id）；两批并发 upsert 也可 WAL A,B
/// 而内存 B,A（last-writer 分叉）。write_lock 串行化「WAL append + 内存变更」后，
/// WAL 顺序 = 内存顺序，重放必然收敛。
/// EN: Regression (WAL/memory order divergence): under concurrent same-id
/// upsert/delete, cold replay must converge to the acknowledged in-memory state.
/// The split-lock version could persist delete before upsert while tombstoning
/// after the upsert's scalar row (replay resurrects a pre-crash acknowledged
/// delete), and two concurrent upserts could land WAL A,B but memory B,A
/// (last-writer divergence). The write_lock serializes "WAL append + mutation"
/// so WAL order equals memory order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_upsert_delete_replay_converges_to_memory_state() {
    use houyi_server::http_handler::handle_collection_upsert;
    use houyi_server::types::{UpsertRequest, UpsertVector};
    let (state, _cleanup) = collection_test_state(4);
    let wal_path = state.wal_path.clone();
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&wal_path);
    collection_upsert(&state, "cc", three_docs()).await;

    // 60 对并发同 id upsert/delete：胜负未定，但无论谁赢，两侧终态必须一致。
    // EN: 60 concurrent same-id upsert/delete pairs — the winner is
    // indeterminate, but both sides must agree on the final state.
    let mut handles = Vec::new();
    for round in 0..60u64 {
        let s = state.clone();
        handles.push(tokio::spawn(async move {
            let mut payload = serde_json::Map::new();
            payload.insert("round".to_string(), serde_json::json!(round));
            let req = UpsertRequest {
                vectors: vec![UpsertVector {
                    id: 1,
                    values: vec![1.0, 0.0, 0.0, 0.0],
                    label: Some("fruit".to_string()),
                    payload: Some(payload),
                }],
                index: None,
            };
            // 直调 handler 不排空：builder 排空与 write_lock 串行化语义无关。
            // EN: Call the handler directly without draining — builder draining
            // is orthogonal to write_lock serialization.
            let _ = handle_collection_upsert(State(s), Path("cc".to_string()), Json(req)).await;
        }));
        let s = state.clone();
        handles.push(tokio::spawn(async move {
            let _ = collection_delete_call(&s, "cc", vec![1]).await;
        }));
    }
    for h in handles {
        h.await.expect("upsert/delete task panicked");
    }

    // 内存终态（排空后走公开检索路径）：向量命中 + payload 回读（墓碑 ⇒ 无）。
    // EN: In-memory final state via the public search path after draining:
    // vector hit + payload read-back (tombstone ⇒ absent).
    if let Some(col) = state.collections.get("cc") {
        for _ in 0..1000 {
            if col.backend.run_async_builder_once().await.flushed == 0 {
                break;
            }
        }
    }
    let memory =
        collection_search_with_payload(&state, "cc", vec![1.0, 0.0, 0.0, 0.0], 5, true).await;
    let memory_vector_alive = memory.ids.contains(&1);
    let memory_payload_alive = memory
        .payloads
        .as_ref()
        .is_some_and(|p| p.contains_key("1"));
    drop(state);

    // 冷启动重放终态（replay 末尾已 drain_pending_builder），同样走检索路径。
    // EN: Cold-replay final state (replay drains the pending builder at the
    // end), verified through the same search path.
    let (_w, entries) = houyi_persistence::FileWal::open(&collections_wal_path).unwrap();
    let manager = std::sync::Arc::new(houyi_server::collection::CollectionManager::new(4, "hnsw"));
    houyi_server::recovery::replay_collections_wal_into_manager(
        &entries,
        &manager,
        &std::collections::HashMap::new(),
    )
    .await
    .expect("cold replay must succeed");
    let wal = std::sync::Arc::new(
        super::WalHandle::open(&wal_path, super::WalWriteMode::Async)
            .unwrap()
            .0,
    );
    let collections_wal = std::sync::Arc::new(
        super::WalHandle::open(&collections_wal_path, super::WalWriteMode::Async)
            .unwrap()
            .0,
    );
    let state2 = BenchmarkHttpState {
        audit: None,
        encryption: None,
        promote: None,
        index: IndexBackend::new_hnsw(4),
        collections: manager,
        wal,
        collections_wal,
        vector_dim: 4,
        pgwire_metrics: std::sync::Arc::new(PgWireMetrics::default()),
        applied_sidecar: wal_path.with_extension("applied"),
        applied_sidecar_lock: std::sync::Arc::new(Mutex::new(())),
        wal_path: wal_path.clone(),
        label_index: std::sync::Arc::new(RwLock::new(std::collections::HashMap::new())),
        openai_upstream: None,
        http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
        graph_snapshot_metrics: Arc::new(houyi_server::recovery::GraphSnapshotMetrics::default()),
        role: houyi_server::types::RoleState::default(),
        cluster: test_cluster(),
        query_log: None,
        replication_metrics: houyi_server::types::shared_replication_metrics(None),
        lifecycle: std::sync::Arc::new(
            houyi_server::collection_lifecycle::CollectionLifecycleState::for_test(),
        ),
    };
    let replayed =
        collection_search_with_payload(&state2, "cc", vec![1.0, 0.0, 0.0, 0.0], 5, true).await;
    let replay_vector_alive = replayed.ids.contains(&1);
    let replay_payload_alive = replayed
        .payloads
        .as_ref()
        .is_some_and(|p| p.contains_key("1"));

    assert_eq!(
        memory_payload_alive, replay_payload_alive,
        "scalar/tombstone state: replay must converge to the acknowledged memory state"
    );
    assert_eq!(
        memory_vector_alive, replay_vector_alive,
        "vector state: replay must converge to the acknowledged memory state"
    );
}

/// 不存在的 collection：delete 返回 404。
/// EN: deleting from an unknown collection yields 404.
#[tokio::test]
async fn collection_delete_nonexistent_collection_returns_404() {
    let (state, _cleanup) = collection_test_state(4);
    let err = collection_delete_call(&state, "nope", vec![1])
        .await
        .expect_err("unknown collection must 404");
    assert_eq!(err.0, StatusCode::NOT_FOUND);
}

/// delete-by-filter 必须覆盖「刚 upsert、尚在 builder_queue 待构图」的向量：
/// 标量行随 upsert 同步落页，过滤求值能命中它们；remove() 同时清扫 queue/hot/main。
/// 这是「先检索拿 id 再删」无法保证的（检索看不到队列内向量），也是服务端提供
/// filters 形态删除的根本原因。
/// EN: delete-by-filter must cover freshly upserted vectors still sitting in
/// builder_queue: scalar rows land synchronously at upsert so filter evaluation
/// sees them, and remove() sweeps queue/hot/main alike. A search-then-delete
/// flow cannot guarantee this (search cannot see queued vectors) — the whole
/// reason the server offers filter-shaped delete.
#[tokio::test]
async fn collection_delete_by_filter_covers_queued_vectors() {
    use houyi_server::http_handler::handle_collection_upsert;
    use houyi_server::types::{HybridSearchFilter, UpsertRequest};
    let (state, _cleanup) = collection_test_state(4);
    // 直调 upsert handler，不做同步排空——三条向量全部滞留 builder_queue。
    // Call the upsert handler directly without draining — all three vectors
    // remain queued for graph building.
    let req = UpsertRequest {
        vectors: three_docs(),
        index: None,
    };
    let _ = handle_collection_upsert(State(state.clone()), Path("c".to_string()), Json(req))
        .await
        .expect("collection upsert should succeed");

    let resp = collection_delete_full(
        &state,
        "c",
        Vec::new(),
        vec![HybridSearchFilter {
            column: "label".to_string(),
            op: "eq".to_string(),
            value: serde_json::json!("fruit"),
        }],
    )
    .await
    .expect("delete-by-filter should succeed");
    assert_eq!(resp.deleted, 1, "queued id=1 (label=fruit) must be deleted");

    // 排空 builder queue：被删 id 不得借 flush 复活进图。
    // Drain the queue: the deleted id must not resurrect into the graph.
    let col = state.collections.get("c").expect("collection exists");
    for _ in 0..1000 {
        if col.backend.run_async_builder_once().await.flushed == 0 {
            break;
        }
    }
    let found = collection_search(&state, "c", vec![1.0, 0.0, 0.0, 0.0], 3, None).await;
    assert!(
        !found.ids.contains(&1),
        "deleted queued id must stay invisible after flush: {:?}",
        found.ids
    );
    assert_eq!(found.ids.len(), 2, "ids 2/3 survive");
    // 标量墓碑同步生效：过滤命中为空。
    // EN: The scalar tombstone takes effect synchronously: filter hits are empty.
    let hits = collection_hybrid(
        &state,
        "c",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("label", "eq", serde_json::json!("fruit"))],
    )
    .await;
    assert!(hits.is_empty(), "scalar tombstone must exclude the id");
}

/// ids 与 filters 并集删除：两边命中的 id 都被删除。
/// EN: ids and filters union — matches from both sides are deleted.
#[tokio::test]
async fn collection_delete_by_filter_unions_with_explicit_ids() {
    use houyi_server::types::HybridSearchFilter;
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(&state, "c", three_docs()).await;
    let resp = collection_delete_full(
        &state,
        "c",
        vec![3], // 显式删 id=3
        vec![HybridSearchFilter {
            column: "label".to_string(),
            op: "eq".to_string(),
            value: serde_json::json!("electronics"), // 过滤命中 id=2
        }],
    )
    .await
    .expect("union delete should succeed");
    assert_eq!(resp.deleted, 2, "id=3 (explicit) + id=2 (filter)");
    let found = collection_search(&state, "c", vec![1.0, 0.0, 0.0, 0.0], 3, None).await;
    assert_eq!(found.ids, vec![1], "only id=1 survives");
}

/// ids 与 filters 同时为空：400（空删除请求属于调用方错误，显式拒绝）。
/// EN: empty ids + empty filters → 400 (a no-op delete request is a caller error).
#[tokio::test]
async fn collection_delete_empty_request_returns_400() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(&state, "c", three_docs()).await;
    let err = collection_delete_full(&state, "c", Vec::new(), Vec::new())
        .await
        .expect_err("empty delete request must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
}

/// delete-by-filter 在不支持过滤语义的后端上 fail-fast 400，且不产生半删除
/// （显式 ids 部分也不执行）。
/// EN: delete-by-filter fails fast with 400 on backends without filter semantics,
/// and does not half-delete (explicit ids are not applied either).
#[tokio::test]
async fn collection_delete_by_filter_unsupported_backend_returns_400() {
    use houyi_server::types::HybridSearchFilter;
    let (state, _cleanup) = collection_test_state_with_index(4, "clustered");
    collection_upsert(&state, "c", three_docs()).await;
    let err = collection_delete_full(
        &state,
        "c",
        vec![3],
        vec![HybridSearchFilter {
            column: "label".to_string(),
            op: "eq".to_string(),
            value: serde_json::json!("fruit"),
        }],
    )
    .await
    .expect_err("clustered backend must refuse delete-by-filter");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err.1.contains("unsupported"), "got: {}", err.1);
    // 半删除防护：显式 id=3 也不得被删（clustered 需 optimize_build 后才可检索）。
    // EN: Half-delete guard: explicit id=3 must not be deleted either (clustered needs
    // EN: optimize_build).
    let col = state.collections.get("c").expect("collection exists");
    assert_eq!(col.backend.optimize_build().await, 3);
    let found = collection_search(&state, "c", vec![0.0, 0.0, 1.0, 0.0], 1, None).await;
    assert_eq!(
        found.ids,
        vec![3],
        "explicit ids must survive a refused delete"
    );
}

// ── OpenAI 兼容 /v1/embeddings 网关 ─────────────────────
// EN: OpenAI-compatible /v1/embeddings gateway.

/// 起一个假上游：OpenAI 形状应答，embedding 由原文确定性生成（4 维）；
/// model == "rate-limited" 时回 429（验证错误透传 / 4xx 不计熔断），
/// model == "boom" 时回 500（验证 5xx 计熔断失败）。
/// Spin up a fake upstream speaking the OpenAI embeddings shape; deterministic
/// 4-dim embeddings; model "rate-limited" yields 429 to test error forwarding,
/// model "boom" yields 500 to test breaker-relevant failures.
async fn spawn_fake_openai_upstream() -> String {
    use axum::routing::post;
    use axum::Router;
    async fn fake_embeddings(
        Json(req): Json<serde_json::Value>,
    ) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
        if req["model"].as_str() == Some("rate-limited") {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"message":"slow down","type":"rate_limit"}}"#.to_string(),
            ));
        }
        if req["model"].as_str() == Some("boom") {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"{"error":{"message":"upstream exploded","type":"server_error"}}"#.to_string(),
            ));
        }
        let inputs: Vec<String> = match &req["input"] {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
        let data: Vec<serde_json::Value> = inputs
            .iter()
            .enumerate()
            .map(|(i, text)| {
                serde_json::json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": [text.len() as f32, i as f32, 0.5, -0.5],
                })
            })
            .collect();
        Ok(Json(serde_json::json!({
            "object": "list",
            "data": data,
            "model": req["model"],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        })))
    }
    let app = Router::new().route("/v1/embeddings", post(fake_embeddings));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn openai_gateway_state(upstream: Option<String>) -> (BenchmarkHttpState, TempFileCleanup) {
    let (mut state, cleanup) = collection_test_state(4);
    state.openai_upstream = upstream.map(|base_url| {
        Arc::new(houyi_server::types::OpenAIUpstreamConfig::new(
            base_url,
            Some("test-key".to_string()),
            std::time::Duration::from_secs(5),
        ))
    });
    (state, cleanup)
}

async fn openai_embeddings_call(
    state: &BenchmarkHttpState,
    body: serde_json::Value,
) -> Result<serde_json::Value, (StatusCode, String)> {
    use houyi_server::openai_gateway::handle_openai_embeddings;
    handle_openai_embeddings(State(state.clone()), Json(body))
        .await
        .map(|Json(v)| v)
}

/// 未配置上游：501（明确未启用，而不是 404 让调用方怀疑路径拼错）。
/// EN: No upstream configured: 501 (explicitly disabled, not 404 that makes callers
/// EN: suspect a typo in the path).
#[tokio::test]
async fn openai_gateway_unconfigured_returns_501() {
    let (state, _cleanup) = openai_gateway_state(None);
    let err = openai_embeddings_call(&state, serde_json::json!({"model": "m", "input": "hi"}))
        .await
        .expect_err("unconfigured gateway must 501");
    assert_eq!(err.0, StatusCode::NOT_IMPLEMENTED);
    assert!(err.1.contains("not configured"), "got: {}", err.1);
}

/// 纯代理（无 collection 字段）：响应原样回传，不创建任何 collection。
/// EN: Pure proxy (no collection field): the response passes through unchanged;
/// EN: no collection is created.
#[tokio::test]
async fn openai_gateway_transparent_proxy_round_trip() {
    let upstream = spawn_fake_openai_upstream().await;
    let (state, _cleanup) = openai_gateway_state(Some(upstream));
    let resp = openai_embeddings_call(
        &state,
        serde_json::json!({"model": "fake-1", "input": ["hello", "world!"]}),
    )
    .await
    .expect("proxy should succeed");
    assert_eq!(resp["object"], "list");
    assert_eq!(resp["data"].as_array().unwrap().len(), 2);
    assert_eq!(resp["data"][0]["embedding"][0], 5.0); // "hello".len()
    assert!(resp.get("houyi").is_none(), "no ingest without collection");
    assert!(
        state.collections.get("fake-1").is_none(),
        "transparent proxy must not create collections"
    );
}

/// 上游业务错误：状态码与正文原样透传（不包信封、不降级成 500）。
/// EN: Upstream business error: status code and body pass through unchanged
/// EN: (no envelope, no downgrade to 500).
#[tokio::test]
async fn openai_gateway_forwards_upstream_error() {
    let upstream = spawn_fake_openai_upstream().await;
    let (state, _cleanup) = openai_gateway_state(Some(upstream));
    let err = openai_embeddings_call(
        &state,
        serde_json::json!({"model": "rate-limited", "input": "hi"}),
    )
    .await
    .expect_err("upstream 429 must be forwarded");
    assert_eq!(err.0, StatusCode::TOO_MANY_REQUESTS);
    assert!(err.1.contains("rate_limit"), "got: {}", err.1);
}

// ── P4.3.3 OpenAI 上游熔断 ──────────────────────────────────────
// EN: P4.3.3 OpenAI upstream circuit breaker.

/// 连续 5xx 越阈值后熔断开路：后续请求 503 快速失败（不再透传、不打上游），
/// 指标三件套（state/open_total/rejected_total）同步。
/// EN: After consecutive 5xx crosses the threshold the breaker opens: later requests
/// EN: fast-fail with 503 (no passthrough, no upstream call); the three metrics
/// EN: (state/open_total/rejected_total) follow.
#[tokio::test]
async fn openai_breaker_trips_open_after_consecutive_5xx_and_fast_fails() {
    use houyi_server::breaker::BREAKER_FAILURE_THRESHOLD;
    let upstream = spawn_fake_openai_upstream().await;
    let (state, _cleanup) = openai_gateway_state(Some(upstream));
    let breaker = &state.openai_upstream.as_ref().unwrap().breaker;

    // 未越阈值：5xx 原样透传 500，断路器仍闭合。
    // EN: Below threshold: 5xx passes through as 500; the breaker stays closed.
    for _ in 0..BREAKER_FAILURE_THRESHOLD {
        let err =
            openai_embeddings_call(&state, serde_json::json!({"model": "boom", "input": "hi"}))
                .await
                .expect_err("upstream 500 must be forwarded");
        assert_eq!(err.0, StatusCode::INTERNAL_SERVER_ERROR, "got: {}", err.1);
    }
    assert_eq!(breaker.state_code(), 1, "threshold reached must open");
    assert_eq!(breaker.open_total(), 1);

    // 开路后：503 快速失败，错误文案点名熔断器。
    // EN: Once open: 503 fast-fail; the error text names the circuit breaker.
    let err = openai_embeddings_call(&state, serde_json::json!({"model": "boom", "input": "hi"}))
        .await
        .expect_err("open breaker must fast-fail");
    assert_eq!(err.0, StatusCode::SERVICE_UNAVAILABLE);
    assert!(err.1.contains("circuit breaker"), "got: {}", err.1);
    assert_eq!(breaker.rejected_open_total(), 1);
}

/// 4xx（429）不计熔断：连续超阈值次数的 4xx 后断路器仍闭合、请求仍透传。
/// EN: 4xx (429) does not trip the breaker: after more than the threshold of 4xx
/// EN: the breaker stays closed and requests still pass through.
#[tokio::test]
async fn openai_breaker_ignores_4xx_failures() {
    use houyi_server::breaker::BREAKER_FAILURE_THRESHOLD;
    let upstream = spawn_fake_openai_upstream().await;
    let (state, _cleanup) = openai_gateway_state(Some(upstream));
    let breaker = &state.openai_upstream.as_ref().unwrap().breaker;

    for _ in 0..BREAKER_FAILURE_THRESHOLD + 2 {
        let err = openai_embeddings_call(
            &state,
            serde_json::json!({"model": "rate-limited", "input": "hi"}),
        )
        .await
        .expect_err("429 must keep being forwarded");
        assert_eq!(
            err.0,
            StatusCode::TOO_MANY_REQUESTS,
            "4xx must never fast-fail via the breaker: {}",
            err.1
        );
    }
    assert_eq!(breaker.state_code(), 0, "4xx must not trip the breaker");
    assert_eq!(breaker.open_total(), 0);
    assert_eq!(breaker.rejected_open_total(), 0);
}

/// 自动入库：collection 字段触发 upsert——id 取 stable_id(原文)，原文进 payload，
/// 响应注入 houyi 扩展字段；检索可回读原文。
/// EN: Auto-ingest: the collection field triggers upsert — id = stable_id(text), text goes
/// EN: into payload, the response injects houyi extension fields; search reads the text back.
#[tokio::test]
async fn openai_gateway_auto_ingest_upserts_with_stable_ids() {
    use houyi_server::openai_gateway::stable_id;
    let upstream = spawn_fake_openai_upstream().await;
    let (state, _cleanup) = openai_gateway_state(Some(upstream));
    let resp = openai_embeddings_call(
        &state,
        serde_json::json!({
            "model": "fake-1",
            "input": ["hello", "world!"],
            "collection": "oa",
        }),
    )
    .await
    .expect("auto-ingest should succeed");
    assert_eq!(resp["houyi"]["collection"], "oa");
    assert_eq!(resp["houyi"]["upserted"], 2);
    // OpenAI 形状部分不受影响。
    // EN: The OpenAI-shaped part is unaffected.
    assert_eq!(resp["data"].as_array().unwrap().len(), 2);

    // 排空构图队列后检索：stable_id("hello") 命中，payload 回读原文。
    // EN: After draining the build queue, search finds stable_id("hello") and payload returns the
    // EN: text.
    let col = state.collections.get("oa").expect("collection created");
    for _ in 0..1000 {
        if col.backend.run_async_builder_once().await.flushed == 0 {
            break;
        }
    }
    let found =
        collection_search_with_payload(&state, "oa", vec![5.0, 0.0, 0.5, -0.5], 1, true).await;
    let expected = stable_id("hello");
    assert_eq!(found.ids, vec![expected], "id must be stable_id(text)");
    let payloads = found.payloads.expect("payloads present");
    assert_eq!(payloads[&expected.to_string()]["text"], "hello");
}

/// 自动入库 + token 数组输入：无法还原原文，显式 400（不静默入库无 payload 的向量）。
/// EN: Auto-ingest with token-array input: the text cannot be reconstructed, so return 400
/// EN: (never silently ingest a payload-less vector).
#[tokio::test]
async fn openai_gateway_auto_ingest_rejects_token_arrays() {
    let upstream = spawn_fake_openai_upstream().await;
    let (state, _cleanup) = openai_gateway_state(Some(upstream));
    let err = openai_embeddings_call(
        &state,
        serde_json::json!({"model": "m", "input": [[1, 2, 3]], "collection": "oa"}),
    )
    .await
    .expect_err("token arrays must be rejected for auto-ingest");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err.1.contains("token arrays"), "got: {}", err.1);
}

/// 自动入库 + 空 input 数组：400（与 OpenAI 同口径，且避免建出空 collection）。
/// EN: Auto-ingest with an empty input array: 400 (matches OpenAI, and avoids
/// EN: creating an empty collection).
#[tokio::test]
async fn openai_gateway_auto_ingest_rejects_empty_input() {
    let upstream = spawn_fake_openai_upstream().await;
    let (state, _cleanup) = openai_gateway_state(Some(upstream));
    let err = openai_embeddings_call(
        &state,
        serde_json::json!({"model": "m", "input": [], "collection": "oa"}),
    )
    .await
    .expect_err("empty input must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn collection_hybrid_eq_text_payload_matches_without_quote_bug() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "c1",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("fruit".to_string()),
            payload: Some(json_payload(
                serde_json::json!({"color": "red", "price": 10}),
            )),
        }],
    )
    .await;

    // JSON 字符串 "red" 必须按裸字符串匹配；旧实现序列化成 Text("\"red\"") 永不命中。
    // EN: JSON string "red" must match as a bare string; the old impl serialized
    // EN: Text("\"red\"") and never matched.
    let red = collection_hybrid(
        &state,
        "c1",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(
        red,
        vec![1],
        "eq on text payload must match raw JSON string"
    );

    let blue = collection_hybrid(
        &state,
        "c1",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("blue"))],
    )
    .await;
    assert!(blue.is_empty(), "non-matching eq must return empty");
}

#[tokio::test]
async fn collection_hybrid_numeric_int_and_float_filters() {
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "c2",
        vec![
            UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"price": 10, "score": 1.5}))),
            },
            UpsertVector {
                id: 2,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                // score 也给出浮点值：缺列会被页补成 JsonText("null")，
                // 而 houyi-core 跨类型 compare 返回 Equal，会污染 gte/lte 语义（既有行为）。
                // EN: score also yields a float: a missing column is filled as JsonText("null"),
                // EN: and houyi-core cross-type compare returns Equal, polluting gte/lte semantics.
                payload: Some(json_payload(serde_json::json!({"price": 1, "score": 0.5}))),
            },
        ],
    )
    .await;

    let gt = collection_hybrid(
        &state,
        "c2",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("price", "gt", serde_json::json!(5))],
    )
    .await;
    assert_eq!(gt, vec![1], "gt price=5 must match only id 1");

    let lt = collection_hybrid(
        &state,
        "c2",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("price", "lt", serde_json::json!(5))],
    )
    .await;
    assert!(!lt.contains(&1), "lt price=5 must not match id 1");
    assert_eq!(lt, vec![2]);

    let gte_float = collection_hybrid(
        &state,
        "c2",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("score", "gte", serde_json::json!(1.5))],
    )
    .await;
    assert_eq!(gte_float, vec![1], "gte on float payload must match id 1");

    let lte_float = collection_hybrid(
        &state,
        "c2",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("score", "lte", serde_json::json!(1.5))],
    )
    .await;
    // 两个浮点 score（1.5 / 0.5）均 ≤ 1.5；按与查询距离排序 id1 在前。
    // EN: Both float scores (1.5 / 0.5) are <= 1.5; ordered by distance to the query, id1 first.
    assert_eq!(
        lte_float,
        vec![1, 2],
        "lte on float payload must match both"
    );

    let lte_float_low = collection_hybrid(
        &state,
        "c2",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("score", "lte", serde_json::json!(1.4))],
    )
    .await;
    assert_eq!(lte_float_low, vec![2], "lte score=1.4 must match only id 2");
}

#[tokio::test]
async fn collection_hybrid_label_column_filter() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "c3",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("fruit".to_string()),
            payload: None,
        }],
    )
    .await;

    let hit = collection_hybrid(
        &state,
        "c3",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("label", "eq", serde_json::json!("fruit"))],
    )
    .await;
    assert_eq!(hit, vec![1], "label column eq must match id 1");

    let miss = collection_hybrid(
        &state,
        "c3",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("label", "eq", serde_json::json!("veg"))],
    )
    .await;
    assert!(miss.is_empty());
}

#[tokio::test]
async fn collection_hybrid_multiple_filters_intersect() {
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "c4",
        vec![
            UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(
                    serde_json::json!({"color": "red", "price": 10}),
                )),
            },
            UpsertVector {
                id: 2,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(
                    serde_json::json!({"color": "blue", "price": 100}),
                )),
            },
        ],
    )
    .await;

    let both = collection_hybrid(
        &state,
        "c4",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![
            ("color", "eq", serde_json::json!("red")),
            ("price", "gt", serde_json::json!(5)),
        ],
    )
    .await;
    assert_eq!(both, vec![1], "intersection of color+price must be id 1");

    let disjoint = collection_hybrid(
        &state,
        "c4",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![
            ("color", "eq", serde_json::json!("red")),
            ("price", "gt", serde_json::json!(100)),
        ],
    )
    .await;
    assert!(disjoint.is_empty(), "disjoint intersection must be empty");
}

#[tokio::test]
async fn collection_hybrid_filter_nonexistent_column_returns_empty() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "c5",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("fruit".to_string()),
            payload: None,
        }],
    )
    .await;

    let ids = collection_hybrid(
        &state,
        "c5",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("no_such_column", "eq", serde_json::json!("x"))],
    )
    .await;
    assert!(ids.is_empty());
}

#[tokio::test]
async fn collection_hybrid_no_filters_falls_back_to_plain_search() {
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "c6",
        vec![
            UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: None,
            },
            UpsertVector {
                id: 2,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                payload: None,
            },
        ],
    )
    .await;

    let ids = collection_hybrid(&state, "c6", vec![1.0, 0.0, 0.0, 0.0], 10, vec![]).await;
    assert!(
        ids.contains(&1) && ids.contains(&2),
        "no filters must return both vectors, got {ids:?}"
    );
}

#[tokio::test]
async fn collection_hybrid_reupsert_same_id_dedupes_results() {
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::types::UpsertVector;
    // 同一 id 两次 upsert，标量页 append-only 产生两行，均映射到 id 1。
    // EN: Two upserts of the same id: the append-only scalar page keeps two rows, both mapped to id
    // EN: 1.
    collection_upsert(
        &state,
        "c7",
        vec![UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"color": "red"}))),
        }],
    )
    .await;
    collection_upsert(
        &state,
        "c7",
        vec![UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"color": "blue"}))),
        }],
    )
    .await;

    let ids = collection_hybrid(
        &state,
        "c7",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("blue"))],
    )
    .await;
    assert_eq!(
        ids.iter().filter(|&&x| x == 1).count(),
        1,
        "same vector id must appear at most once, got {ids:?}"
    );

    // 压实前：append-only 旧行仍在页内，red 的旧值依旧命中。
    // EN: Before compaction the append-only stale row still matches red.
    let stale = collection_hybrid(
        &state,
        "c7",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(
        stale,
        vec![1],
        "pre-compaction: superseded value still matches (documented behavior)"
    );

    // 压实后：只保留最后一行，旧值不再命中。
    // EN: After compaction only the last row remains; the stale value no longer matches.
    let col = state.collections.get("c7").expect("collection exists");
    col.compact_scalars(true, 0.5)
        .await
        .expect("compaction must succeed");
    let red_after = collection_hybrid(
        &state,
        "c7",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert!(
        red_after.is_empty(),
        "post-compaction: superseded value must not match, got {red_after:?}"
    );
    let blue_after = collection_hybrid(
        &state,
        "c7",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("blue"))],
    )
    .await;
    assert_eq!(blue_after, vec![1], "latest value must still match");
}

/// P5.0.1：status 暴露物理/可见/stale 行与 applied_lsn；管理端点压实后只剩最后一行。
/// EN: P5.0.1 — status exposes physical/live/stale rows and applied_lsn; the
/// admin compaction endpoint keeps only the last row per id.
#[tokio::test]
async fn collection_scalar_compaction_endpoint_and_status() {
    use houyi_server::http_handler::{
        handle_collection_compact, handle_collection_status, ScalarCompactRequest,
    };
    use houyi_server::types::UpsertVector;
    let (state, _cleanup) = collection_test_state(4);
    for color in ["red", "blue"] {
        collection_upsert(
            &state,
            "compact1",
            vec![UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": color}))),
            }],
        )
        .await;
    }

    let status = handle_collection_status(State(state.clone()), Path("compact1".to_string()))
        .await
        .unwrap()
        .0;
    assert_eq!(status["scalar_physical_rows"], 2);
    assert_eq!(status["scalar_live_rows"], 1);
    assert_eq!(status["scalar_stale_rows"], 1);
    assert_eq!(status["scalar_tombstone_ids"], 0);
    assert_eq!(
        status["applied_lsn"], 2,
        "create LSN 0 + two upsert LSNs 1/2 → applied 2"
    );

    let resp = handle_collection_compact(
        State(state.clone()),
        Path("compact1".to_string()),
        Json(ScalarCompactRequest { force: true }),
    )
    .await
    .unwrap()
    .0;
    assert_eq!(resp["dropped_rows"], 1);
    assert_eq!(resp["physical_after"], 1);

    let status = handle_collection_status(State(state), Path("compact1".to_string()))
        .await
        .unwrap()
        .0;
    assert_eq!(status["scalar_physical_rows"], 1);
    assert_eq!(status["scalar_live_rows"], 1);
    assert_eq!(status["scalar_stale_rows"], 0);
    assert!(status["last_compaction_unix"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn collection_upsert_rejects_reserved_label_payload_key() {
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::http_handler::handle_collection_upsert;
    use houyi_server::types::{UpsertRequest, UpsertVector};
    // payload 夹带保留键 "label"：必须 400，且不得静默覆盖顶层 label 语义。
    // EN: Payload carrying the reserved key "label" must 400 and never silently override
    // EN: top-level label semantics.
    let req = UpsertRequest {
        vectors: vec![UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("top-level-tag".to_string()),
            payload: Some(json_payload(serde_json::json!({
                "label": "sneaky",
                "color": "red"
            }))),
        }],
        index: None,
    };
    let err = handle_collection_upsert(State(state.clone()), Path("c8".to_string()), Json(req))
        .await
        .expect_err("reserved payload key must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    // 校验先于任何写入：Collection 不应被隐式创建。
    // EN: Validation precedes any write: no collection may be implicitly created.
    assert!(
        state.collections.get("c8").is_none(),
        "no collection may be created when request validation fails"
    );
}

// ── 评审修订回归：flat mask / filter 边界 / 混合批次 / 隔离 ──────────────────
// EN: Review-fix regressions — flat mask honored, filter validation,
// mixed-batch alignment, per-collection isolation, 400/404 cases.

#[tokio::test]
async fn flat_backend_hybrid_filter_honors_mask() {
    // FIX-1 回归：flat 后端曾完全忽略过滤 mask，返回未过滤结果。
    // 让 blue 比 red 更靠近查询向量：旧实现 k=1 会错误返回 blue(id=2)。
    // EN: blue is nearer the query; an ignored mask would return id 2 for k=1.
    let (state, _cleanup) = collection_test_state_with_index(4, "flat");
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "flat1",
        vec![
            UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "red"}))),
            },
            UpsertVector {
                id: 2,
                values: vec![0.9, 0.1, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "blue"}))),
            },
        ],
    )
    .await;

    let only_red = collection_hybrid(
        &state,
        "flat1",
        vec![0.9, 0.1, 0.0, 0.0],
        1,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(only_red, vec![1], "flat backend must score ONLY masked ids");

    let only_red_topk = collection_hybrid(
        &state,
        "flat1",
        vec![0.9, 0.1, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(
        only_red_topk,
        vec![1],
        "flat filtered search must return exactly the red id"
    );
}

#[tokio::test]
async fn clustered_backend_filtered_hybrid_search_rejected_with_400() {
    // FIX-1(c)：非 hnsw/flat 索引尚不支持标量过滤——必须明确 400，
    // 不能静默丢掉过滤条件返回未过滤结果。
    // EN: Filtered hybrid search on unsupported backends must fail fast (400).
    let (state, _cleanup) = collection_test_state_with_index(4, "clustered");
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "cl1",
        vec![UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"color": "red"}))),
        }],
    )
    .await;

    let err = collection_hybrid_result(
        &state,
        "cl1",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await
    .expect_err("filtered hybrid on clustered must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(
        err.1.contains("clustered"),
        "error must name the unsupported index type, got: {}",
        err.1
    );

    // 无过滤条件的检索不应被误伤（路由本身可用）。
    // Unfiltered search on the same collection still routes fine.
    let ok = collection_hybrid_result(&state, "cl1", vec![1.0, 0.0, 0.0, 0.0], 10, vec![]).await;
    assert!(ok.is_ok(), "unfiltered hybrid must not be rejected");
}

#[tokio::test]
async fn hybrid_filter_in_op_actually_filters() {
    // FIX-4(a)：in 操作符的「引号对称」回归——in 列表必须真实收敛结果。
    // EN: The `in` op must actually restrict results (quote-fix symmetry).
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "in1",
        vec![
            UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "red"}))),
            },
            UpsertVector {
                id: 2,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "blue"}))),
            },
            UpsertVector {
                id: 3,
                values: vec![0.0, 0.0, 1.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "green"}))),
            },
        ],
    )
    .await;

    let ids = collection_hybrid(
        &state,
        "in1",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "in", serde_json::json!(["red", "green"]))],
    )
    .await;
    assert_eq!(
        ids,
        vec![1, 3],
        "in [red, green] must return ids 1 and 3 only"
    );
}

#[tokio::test]
async fn hybrid_mixed_batch_bare_and_payload_vectors_stay_aligned() {
    // FIX-4(b)：同一批次里裸向量与 payload 向量混排时，只有带标量的向量建行，
    // row_to_id 必须与页行严格对齐（过滤 red/blue 不得错位命中裸向量）。
    // EN: Bare and payload vectors in one batch — only payload vectors create
    // rows; row_to_id must stay aligned so filters hit the right vector ids.
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "mix1",
        vec![
            UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: None,
            },
            UpsertVector {
                id: 2,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "red"}))),
            },
            UpsertVector {
                id: 3,
                values: vec![0.0, 0.0, 1.0, 0.0],
                label: None,
                payload: None,
            },
            UpsertVector {
                id: 4,
                values: vec![0.0, 0.0, 0.0, 1.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "blue"}))),
            },
        ],
    )
    .await;

    let reds = collection_hybrid(
        &state,
        "mix1",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(
        reds,
        vec![2],
        "red filter must hit id 2 despite bare rows around it"
    );

    let blues = collection_hybrid(
        &state,
        "mix1",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("blue"))],
    )
    .await;
    assert_eq!(blues, vec![4], "blue filter must hit id 4");

    let all = collection_hybrid(&state, "mix1", vec![1.0, 0.0, 0.0, 0.0], 10, vec![]).await;
    for expected in [1_u64, 2, 3, 4] {
        assert!(
            all.contains(&expected),
            "unfiltered search must see id {expected}, got {all:?}"
        );
    }
}

#[tokio::test]
async fn hybrid_filters_isolated_per_collection() {
    // FIX-4(c)：c1 的标量行绝不能泄漏进 c2 的过滤结果。
    // EN: c1's payload rows must never leak into c2's filtered results.
    let (state, _cleanup) = collection_test_state(4);
    use houyi_server::types::UpsertVector;
    collection_upsert(
        &state,
        "iso-a",
        vec![UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"color": "red"}))),
        }],
    )
    .await;
    collection_upsert(
        &state,
        "iso-b",
        vec![
            UpsertVector {
                id: 2,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "blue"}))),
            },
            UpsertVector {
                id: 3,
                values: vec![0.0, 0.0, 1.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "red"}))),
            },
        ],
    )
    .await;

    let a_red = collection_hybrid(
        &state,
        "iso-a",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(a_red, vec![1], "c1 red must be only id 1");

    let b_red = collection_hybrid(
        &state,
        "iso-b",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(b_red, vec![3], "c2 red must be only id 3 (no leak from c1)");

    let b_blue = collection_hybrid(
        &state,
        "iso-b",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("blue"))],
    )
    .await;
    assert_eq!(b_blue, vec![2], "c2 blue must be only id 2");
}

#[tokio::test]
async fn hybrid_filter_in_with_non_array_value_rejected_400() {
    // FIX-2：in 的右值必须是 JSON 数组，否则 400（旧实现静默按「无匹配」处理）。
    // EN: `in` requires a JSON array value; anything else is 400, not silent empty.
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "bad1",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"color": "red"}))),
        }],
    )
    .await;

    let err = collection_hybrid_result(
        &state,
        "bad1",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "in", serde_json::json!("red"))],
    )
    .await
    .expect_err("in with scalar value must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(
        err.1.contains("'in'"),
        "message must mention the in operator: {}",
        err.1
    );
}

#[tokio::test]
async fn hybrid_filter_in_with_nested_array_element_rejected_400() {
    // in 列表元素与标量操作符同规则：元素不得是数组/对象/null，否则 400，
    // 不能静默退化成 JsonText 比较产生零命中（或更糟的脏命中）。
    // EN: `in` elements follow the scalar-op shape rule — array/object/null
    // elements must yield 400 rather than silently comparing JsonText.
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "bad1e",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"color": "red"}))),
        }],
    )
    .await;

    let err = collection_hybrid_result(
        &state,
        "bad1e",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "in", serde_json::json!([[1, 2]]))],
    )
    .await
    .expect_err("in with nested-array element must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(
        err.1.contains("'in'"),
        "message must mention the in operator: {}",
        err.1
    );
}

#[tokio::test]
async fn hybrid_filter_gt_with_array_value_rejected_400() {
    // FIX-2：标量比较操作符右值不得是数组/对象/null。
    // EN: Scalar comparison ops reject array/object/null literals with 400.
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "bad2",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"price": 10}))),
        }],
    )
    .await;

    let err = collection_hybrid_result(
        &state,
        "bad2",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("price", "gt", serde_json::json!([1, 2]))],
    )
    .await
    .expect_err("gt with array value must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);

    let err_null = collection_hybrid_result(
        &state,
        "bad2",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("price", "eq", serde_json::Value::Null)],
    )
    .await
    .expect_err("eq with null value must be rejected");
    assert_eq!(err_null.0, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn hybrid_unknown_op_rejected_400() {
    // FIX-4(e)：未知操作符必须 400。
    // EN: FIX-4(e): unknown operators must return 400.
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "bad3",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: Some(json_payload(serde_json::json!({"color": "red"}))),
        }],
    )
    .await;

    let err = collection_hybrid_result(
        &state,
        "bad3",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "regex", serde_json::json!("red.*"))],
    )
    .await
    .expect_err("unknown op must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(
        err.1.contains("regex"),
        "error must name the unknown op: {}",
        err.1
    );
}

#[tokio::test]
async fn hybrid_search_nonexistent_collection_returns_404() {
    // FIX-4(f)：不存在的 Collection → 404。
    // EN: FIX-4(f): a nonexistent collection -> 404.
    let (state, _cleanup) = collection_test_state(4);
    let err = collection_hybrid_result(
        &state,
        "no-such-collection",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await
    .expect_err("hybrid on missing collection must fail");
    assert_eq!(err.0, StatusCode::NOT_FOUND);
}

// ── Task 3：SearchResponse.scores 相似度分数链路 ──────────────
// EN: Task 3: SearchResponse.scores similarity-score path.

/// 直调 collection search handler，返回完整响应（含 scores）。
/// Call the collection search handler directly; returns the full scored response.
async fn collection_search(
    state: &BenchmarkHttpState,
    name: &str,
    vector: Vec<f32>,
    k: usize,
    ef_search: Option<usize>,
) -> houyi_server::types::SearchResponse {
    use houyi_server::http_handler::handle_collection_search;
    let req = houyi_server::types::SearchRequest {
        vector,
        k,
        filter_gt_id: None,
        ef_search,
        filter_label: None,
        include_payload: false,
    };
    handle_collection_search(
        State(state.clone()),
        Path(name.to_string()),
        None,
        Json(req),
    )
    .await
    .expect("collection search should succeed")
    .0
}

/// 直调 benchmark `/v1/vectors/search` handler。
/// Call the benchmark search handler directly.
async fn benchmark_search_call(
    state: &BenchmarkHttpState,
    req: houyi_server::types::SearchRequest,
) -> houyi_server::types::SearchResponse {
    use houyi_server::http_handler::benchmark_search;
    benchmark_search(State(state.clone()), None, Json(req))
        .await
        .expect("benchmark search should succeed")
        .0
}

/// scores 与 ids 等长对齐，且相似度非升序（越大越相似）。
/// scores must be aligned with ids and non-increasing.
fn assert_scores_aligned_and_non_increasing(resp: &houyi_server::types::SearchResponse) {
    assert_eq!(resp.ids.len(), resp.scores.len(), "ids/scores 必须等长对齐");
    for pair in resp.scores.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "scores 必须非升序（相似度从高到低）: {} < {}",
            pair[0],
            pair[1]
        );
    }
}

/// HNSW Collection 检索：精确匹配向量排第一，score ≈ 1，ids/scores 对齐。
/// Exact-match vector ranks first with score ~1; ids/scores aligned.
#[tokio::test]
async fn collection_search_scores_aligned_and_exact_match_one() {
    let (state, _cleanup) = collection_test_state(4);
    let filler: Vec<[f32; 4]> = vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    ];
    let vectors = filler
        .into_iter()
        .enumerate()
        .map(|(i, v)| houyi_server::types::UpsertVector {
            id: (i + 1) as u64,
            values: v.to_vec(),
            label: None,
            payload: None,
        })
        .chain(std::iter::once(houyi_server::types::UpsertVector {
            id: 42,
            values: vec![2.0, 0.0, 3.0, 0.0],
            label: None,
            payload: None,
        }))
        .collect();
    collection_upsert(&state, "c1", vectors).await;

    let resp = collection_search(&state, "c1", vec![2.0, 0.0, 3.0, 0.0], 5, None).await;
    assert_eq!(resp.ids.len(), 5);
    assert_scores_aligned_and_non_increasing(&resp);
    assert_eq!(resp.ids[0], 42, "精确匹配向量必须排第一");
    assert!(
        (resp.scores[0] - 1.0).abs() < 1e-4,
        "精确匹配 score 应 ≈ 1，实际 {}",
        resp.scores[0]
    );
}

/// 空结果时 ids 与 scores 同为空数组（不能只有 ids 空）。
/// Empty result → both ids and scores empty.
#[tokio::test]
async fn collection_search_empty_result_empty_scores() {
    use houyi_server::http_handler::handle_collection_upsert;
    let (state, _cleanup) = collection_test_state(4);
    // 空批次 upsert 仅用于创建 Collection（search 不会自动建 Collection）。
    // An empty batch creates the collection; search never auto-creates one.
    let created = handle_collection_upsert(
        State(state.clone()),
        Path("empty".to_string()),
        Json(houyi_server::types::UpsertRequest {
            vectors: Vec::new(),
            index: None,
        }),
    )
    .await
    .expect("empty-batch upsert should create collection");
    assert_eq!(created.inserted, 0);

    let resp = collection_search(&state, "empty", vec![1.0, 0.0, 0.0, 0.0], 10, None).await;
    assert!(resp.ids.is_empty());
    assert!(resp.scores.is_empty());
}

/// flat 后端按余弦相似度排序：同向不同模长的 [10,0,0,0] 对 query [1,0,0,0]
/// 必须排第一（score ≈ 1）；旧实现用裸 L2² 会把它排到最后。
/// Flat ranks by cosine: same-direction different-magnitude vector ranks first;
/// the old raw-L2² scoring ranked it last.
#[tokio::test]
async fn flat_collection_search_scores_use_cosine_ranking() {
    let (state, _cleanup) = collection_test_state_with_index(4, "flat");
    collection_upsert(
        &state,
        "fc",
        vec![
            houyi_server::types::UpsertVector {
                id: 1,
                values: vec![10.0, 0.0, 0.0, 0.0],
                label: None,
                payload: None,
            },
            houyi_server::types::UpsertVector {
                id: 2,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                payload: None,
            },
        ],
    )
    .await;

    let resp = collection_search(&state, "fc", vec![1.0, 0.0, 0.0, 0.0], 2, None).await;
    assert_eq!(resp.ids, vec![1, 2], "flat 必须按余弦距离而非裸 L2² 排序");
    assert_scores_aligned_and_non_increasing(&resp);
    assert!(
        (resp.scores[0] - 1.0).abs() < 1e-4,
        "同向向量 score 应 ≈ 1，实际 {}",
        resp.scores[0]
    );
}

/// 混合检索：标量过滤后 ids/sccores 仍对齐，且与 id-only helper 结果一致。
/// Filtered hybrid search keeps ids/scores aligned and matches the id-only helper.
#[tokio::test]
async fn hybrid_filtered_search_scores_align_with_filtered_ids() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "hc",
        vec![
            houyi_server::types::UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "red"}))),
            },
            houyi_server::types::UpsertVector {
                id: 2,
                values: vec![0.9, 0.1, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "red"}))),
            },
            houyi_server::types::UpsertVector {
                id: 3,
                values: vec![0.0, 1.0, 0.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "blue"}))),
            },
            houyi_server::types::UpsertVector {
                id: 4,
                values: vec![0.0, 0.0, 1.0, 0.0],
                label: None,
                payload: Some(json_payload(serde_json::json!({"color": "blue"}))),
            },
        ],
    )
    .await;

    let resp = collection_hybrid_result(
        &state,
        "hc",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await
    .expect("filtered hybrid search should succeed");
    assert_eq!(resp.ids, vec![1, 2]);
    assert_scores_aligned_and_non_increasing(&resp);

    // 与既有 id-only helper 的结果交叉验证顺序一致。
    // Cross-check ordering against the pre-existing id-only helper.
    let ids_only = collection_hybrid(
        &state,
        "hc",
        vec![1.0, 0.0, 0.0, 0.0],
        10,
        vec![("color", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(resp.ids, ids_only);
}

/// benchmark 检索：label/id 过滤的 fetch_k 放大 + retain + truncate 之后，
/// ids/scores 仍对齐；k=0 时两者皆空。
/// After fetch-multiply/retain/truncate, ids/scores stay aligned; k=0 → both empty.
#[tokio::test]
async fn benchmark_search_scores_survive_fetch_filter_truncate() {
    let (state, _cleanup) = collection_test_state(4);
    let mut batch: Vec<(u64, Vec<f32>)> = (1..29u64)
        .map(|id| {
            let v: Vec<f32> = (0..4)
                .map(|j| ((id * 13 + j * 7) % 17) as f32 / 17.0 + 0.05)
                .collect();
            (id, v)
        })
        .collect();
    batch.push((42, vec![0.0, 0.0, 5.0, 0.0]));
    state
        .index
        .insert_batch(batch)
        .await
        .expect("insert_batch should succeed");
    for _ in 0..1000 {
        if state.index.run_async_builder_once().await.flushed == 0 {
            break;
        }
    }

    // 带过滤条件 → fetch_k = k*4，验证放大后再截断不破坏对齐。
    // A filter triggers fetch_k = k*4; verify alignment after the multiply+truncate.
    let resp = benchmark_search_call(
        &state,
        houyi_server::types::SearchRequest {
            vector: vec![0.0, 0.0, 1.0, 0.0],
            k: 5,
            filter_gt_id: Some(0),
            ef_search: None,
            filter_label: None,
            include_payload: false,
        },
    )
    .await;
    assert_eq!(resp.ids.len(), 5);
    assert_scores_aligned_and_non_increasing(&resp);
    assert_eq!(resp.ids[0], 42, "精确匹配向量必须排第一");
    assert!(
        (resp.scores[0] - 1.0).abs() < 1e-4,
        "精确匹配 score 应 ≈ 1，实际 {}",
        resp.scores[0]
    );

    let empty = benchmark_search_call(
        &state,
        houyi_server::types::SearchRequest {
            vector: vec![0.0, 0.0, 1.0, 0.0],
            k: 0,
            filter_gt_id: None,
            ef_search: None,
            filter_label: None,
            include_payload: false,
        },
    )
    .await;
    assert!(empty.ids.is_empty());
    assert!(empty.scores.is_empty());
}

/// benchmark label + id 双重过滤：分数随 id 对一起 retain，不错位。
/// Label + id filters retain score pairs together — no misalignment.
#[tokio::test]
async fn benchmark_search_filtered_scores_remain_aligned() {
    use houyi_server::http_handler::benchmark_upsert;
    let (state, _cleanup) = collection_test_state(4);
    let upserted = benchmark_upsert(
        State(state.clone()),
        Json(houyi_server::types::UpsertRequest {
            vectors: vec![
                houyi_server::types::UpsertVector {
                    id: 1,
                    values: vec![1.0, 0.0, 0.0, 0.0],
                    label: Some("red".to_string()),
                    payload: None,
                },
                houyi_server::types::UpsertVector {
                    id: 2,
                    values: vec![1.0, 1.0, 0.0, 0.0],
                    label: Some("red".to_string()),
                    payload: None,
                },
                houyi_server::types::UpsertVector {
                    id: 3,
                    values: vec![0.9, 0.0, 0.1, 0.0],
                    label: Some("red".to_string()),
                    payload: None,
                },
                houyi_server::types::UpsertVector {
                    id: 4,
                    values: vec![0.0, 1.0, 0.0, 0.0],
                    label: Some("blue".to_string()),
                    payload: None,
                },
                houyi_server::types::UpsertVector {
                    id: 5,
                    values: vec![0.8, 0.0, 0.2, 0.0],
                    label: Some("red".to_string()),
                    payload: None,
                },
            ],
            index: None,
        }),
    )
    .await
    .expect("benchmark upsert should succeed");
    assert_eq!(upserted.inserted, 5);
    for _ in 0..1000 {
        if state.index.run_async_builder_once().await.flushed == 0 {
            break;
        }
    }

    let resp = benchmark_search_call(
        &state,
        houyi_server::types::SearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k: 2,
            filter_gt_id: Some(2),
            ef_search: None,
            filter_label: Some("red".to_string()),
            include_payload: false,
        },
    )
    .await;
    assert_eq!(
        resp.ids,
        vec![3, 5],
        "label=red 且 id>2 只剩 3、5，按距离升序"
    );
    assert_scores_aligned_and_non_increasing(&resp);
}

/// OCR 回归：NaN/±Inf 向量必须在任何副作用之前被 benchmark upsert 拒绝（400）：
/// 毒记录不得落 WAL（否则冷启动重放被库层拒绝而中止），label_index 不得被污染，
/// 索引行数不增长。混合批次（合法行 + 毒行）整批失败、合法行同样不落。
/// EN: NaN/±Inf vectors are rejected (400) before ANY side effect: no WAL append,
/// no label_index pollution, no index growth. A mixed (good+poisoned) batch fails
/// atomically — even the good row is not applied.
#[tokio::test]
async fn benchmark_upsert_rejects_non_finite_before_wal() {
    use houyi_server::http_handler::benchmark_upsert;
    use houyi_server::types::{UpsertRequest, UpsertVector};
    let (state, _cleanup) = collection_test_state(4);
    let lsn_before = state.wal.next_lsn();

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // 毒行带 label：旧实现先写 label_index 再校验，映射会被脏 id 污染。
        // The poisoned row carries a label — the old ordering polluted label_index.
        let req = UpsertRequest {
            vectors: vec![UpsertVector {
                id: 99,
                values: vec![1.0, bad, 0.0, 0.0],
                label: Some("poison".to_string()),
                payload: None,
            }],
            index: None,
        };
        let err = benchmark_upsert(State(state.clone()), Json(req))
            .await
            .expect_err("non-finite component must be rejected");
        assert_eq!(
            err.0,
            StatusCode::BAD_REQUEST,
            "NaN/Inf is a client error, not a bare 500"
        );
    }

    // 混合批次：合法 label 行 + 毒行 → 整批 400，谁都不许落（原子性）。
    // Mixed batch: one good labeled row + one poisoned row → whole batch rejected.
    let mixed = UpsertRequest {
        vectors: vec![
            UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: Some("good".to_string()),
                payload: None,
            },
            UpsertVector {
                id: 2,
                values: vec![0.0, f32::NAN, 0.0, 0.0],
                label: Some("good".to_string()),
                payload: None,
            },
        ],
        index: None,
    };
    assert_eq!(
        benchmark_upsert(State(state.clone()), Json(mixed))
            .await
            .expect_err("mixed batch must be rejected")
            .0,
        StatusCode::BAD_REQUEST
    );

    // 维度错误同样先于 label_index 写入。
    // Dim mismatch is rejected before the label write as well.
    let dim_req = UpsertRequest {
        vectors: vec![UpsertVector {
            id: 3,
            values: vec![1.0, 0.0, 0.0],
            label: Some("wrong-dim".to_string()),
            payload: None,
        }],
        index: None,
    };
    assert_eq!(
        benchmark_upsert(State(state.clone()), Json(dim_req))
            .await
            .expect_err("dim mismatch must be rejected")
            .0,
        StatusCode::BAD_REQUEST
    );

    assert_eq!(
        state.wal.next_lsn(),
        lsn_before,
        "no poisoned record may be appended to the WAL"
    );
    // 收窄锁守卫作用域：块结束即释放，避免守卫生命周期跨后面的 .await
    // （clippy::await_holding_lock 不识别显式 drop）。
    // EN: Scope the guard to an inner block so its lifetime ends before the
    // following .await (the lint ignores explicit drop).
    {
        let labels = state.label_index.read().unwrap();
        for tainted in ["poison", "good", "wrong-dim"] {
            assert!(
                labels.get(tainted).is_none(),
                "label {tainted:?} must not exist after failed validation"
            );
        }
    }
    assert_eq!(
        state.index.indexed_count().await,
        0,
        "no row may be indexed when validation fails"
    );
}

/// OCR 回归：Collection upsert 遇 NaN/±Inf 返回 400，且不隐式创建 Collection、
/// 不落任何行。
/// EN: Collection upsert returns 400 for NaN/±Inf, creates no collection and
/// stores no rows.
#[tokio::test]
async fn collection_upsert_rejects_non_finite() {
    use houyi_server::http_handler::handle_collection_upsert;
    use houyi_server::types::{UpsertRequest, UpsertVector};
    let (state, _cleanup) = collection_test_state(4);

    let req = UpsertRequest {
        vectors: vec![UpsertVector {
            id: 1,
            values: vec![1.0, f32::NAN, 0.0, 0.0],
            label: None,
            payload: None,
        }],
        index: None,
    };
    let err = handle_collection_upsert(State(state.clone()), Path("c-nan".to_string()), Json(req))
        .await
        .expect_err("non-finite component must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(
        state.collections.get("c-nan").is_none(),
        "no collection may be created when input validation fails"
    );
}

/// OCR 回归（查询侧）：NaN/±Inf 查询向量在进索引前以 400 拒绝（距离恒 NaN、排序
/// 未定义）；维度不符同样 400；合法查询不受影响。
/// EN: NaN/±Inf query vectors are rejected with 400 before touching the index;
/// wrong dims also 400; valid queries still work.
#[tokio::test]
async fn benchmark_search_rejects_non_finite_query() {
    use houyi_server::http_handler::{benchmark_search, benchmark_upsert};
    use houyi_server::types::{SearchRequest, UpsertRequest, UpsertResponse, UpsertVector};
    let (state, _cleanup) = collection_test_state(4);

    let _: Json<UpsertResponse> = benchmark_upsert(
        State(state.clone()),
        Json(UpsertRequest {
            vectors: vec![UpsertVector {
                id: 1,
                values: vec![1.0, 0.0, 0.0, 0.0],
                label: None,
                payload: None,
            }],
            index: None,
        }),
    )
    .await
    .expect("seed upsert succeeds");
    // 排空异步构图队列，使种子行进索引可检索。
    // EN: Drain the async builder so seed rows become searchable.
    for _ in 0..1000 {
        if state.index.run_async_builder_once().await.flushed == 0 {
            break;
        }
    }

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = benchmark_search(
            State(state.clone()),
            None,
            Json(SearchRequest {
                vector: vec![1.0, bad, 0.0, 0.0],
                k: 5,
                filter_gt_id: None,
                ef_search: None,
                filter_label: None,
                include_payload: false,
            }),
        )
        .await
        .expect_err("non-finite query must be rejected");
        assert_eq!(
            err.0,
            StatusCode::BAD_REQUEST,
            "NaN/Inf query is a client error"
        );
    }

    let dim_err = benchmark_search(
        State(state.clone()),
        None,
        Json(SearchRequest {
            vector: vec![1.0, 0.0, 0.0],
            k: 5,
            filter_gt_id: None,
            ef_search: None,
            filter_label: None,
            include_payload: false,
        }),
    )
    .await
    .expect_err("wrong-dim query must be rejected");
    assert_eq!(dim_err.0, StatusCode::BAD_REQUEST);

    // k=0 不得让坏输入短路成 200：同一条坏查询无论 k 取值都必须 400（契约一致）。
    // k=0 must not short-circuit an invalid query to 200 — the same bad input is
    // rejected with 400 regardless of k.
    for bad_vector in [
        vec![1.0_f32, f32::NAN, 0.0, 0.0],
        vec![1.0_f32, f32::INFINITY, 0.0, 0.0],
        vec![1.0_f32, 0.0, 0.0],
    ] {
        let err = benchmark_search(
            State(state.clone()),
            None,
            Json(SearchRequest {
                vector: bad_vector,
                k: 0,
                filter_gt_id: None,
                ef_search: None,
                filter_label: None,
                include_payload: false,
            }),
        )
        .await
        .expect_err("invalid query must be rejected even when k=0");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }
    // 对照：合法的 k=0 查询仍是 200 空结果。
    // EN: Control: a valid k=0 query stays 200-empty.
    let ok_zero = benchmark_search(
        State(state.clone()),
        None,
        Json(SearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k: 0,
            filter_gt_id: None,
            ef_search: None,
            filter_label: None,
            include_payload: false,
        }),
    )
    .await
    .expect("valid k=0 query succeeds");
    assert!(ok_zero.0.ids.is_empty());

    // 合法查询不受影响。
    // EN: A valid query still works.
    let ok = benchmark_search(
        State(state.clone()),
        None,
        Json(SearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k: 1,
            filter_gt_id: None,
            ef_search: None,
            filter_label: None,
            include_payload: false,
        }),
    )
    .await
    .expect("valid query succeeds");
    assert_eq!(ok.0.ids, vec![1]);
}

/// OCR 回归：存储侧历史毒行（此处直接灌进 HotBufferIndex——它只校验维度，模拟
/// pg_wire binary float4/float8 旧写入路径的 NaN 脏向量）产生的 NaN 距离永远不得
/// 进入响应：过滤路径按 4k 过取，毒行即便落在取回集合里，也必须在 truncate 前
/// 被剔除，有限候选照常返回，序列化 JSON 不得出现 `null`。
/// 说明（可观测性边界）：HNSW 热路径用 total_cmp 排序（NaN 恒在末尾），今天
/// truncate 单独也不会让 NaN 占住 top-k；truncate 前 retain 是面向「非有限距离
/// 排到有限候选之前」的排序器/度量的结构性防线，响应转换函数内的丢弃仍是最后
/// 安全网。本测试锁定对外可观测契约。
/// EN: A legacy poisoned stored row (inserted straight into HotBufferIndex, which
/// checks dims only — modeling old pg_wire binary float writes) yields NaN distance
/// and must never reach the response: it is dropped before truncate during the 4k
/// over-fetch filter path, finite candidates survive, and serialized JSON has no
/// null. HNSW sorts with total_cmp (NaN last) today, so pre-truncate retain is
/// structural defense for sorters where non-finite can outrank finite; the converter
/// keeps the final net. This test locks the observable contract.
#[tokio::test]
async fn benchmark_search_drops_poisoned_hit_before_truncate_and_backfills() {
    use houyi_server::http_handler::benchmark_search;
    use houyi_server::types::SearchRequest;
    let (state, _cleanup) = collection_test_state(4);
    // 2 条有限行 + 1 条毒行，全部直接写热缓冲（绕过适配层闸门——闸门建立后
    // IndexBackend::insert 本身已拒绝非有限分量，必须下沉到库层直插才能模拟存量毒数据），
    // 立即可检索。
    // EN: Seed straight into HotBufferIndex; the adapter gate now rejects non-finite
    // components itself, so legacy poison must be injected one layer below.
    if let houyi_server::index_backend::IndexBackend::Hnsw(idx) = &state.index {
        idx.insert(1, vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0, 0.0]).await.unwrap();
        idx.insert(99, vec![1.0, f32::NAN, 0.0, 0.0]).await.unwrap();
    } else {
        panic!("collection_test_state default backend must be hnsw");
    }

    // 带 id 过滤 → fetch_k = 4k = 20，毒行确定落在取回集合中；k=5 > 有限行数。
    // The id filter triggers 4k over-fetch (20); k=5 exceeds the finite-row count.
    let resp = benchmark_search(
        State(state.clone()),
        None,
        Json(SearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k: 5,
            filter_gt_id: Some(0),
            ef_search: None,
            filter_label: None,
            include_payload: false,
        }),
    )
    .await
    .expect("search with poisoned stored row still succeeds");
    assert_eq!(resp.0.ids, vec![1, 2], "poisoned id 99 must never appear");
    assert_eq!(resp.0.scores.len(), 2);
    assert!(
        resp.0.scores.iter().all(|s| s.is_finite()),
        "no non-finite score may be returned"
    );
    let json = serde_json::to_string(&resp.0).unwrap();
    assert!(
        !json.contains("null"),
        "serialized scores must not contain null: {json}"
    );
}

/// OCR 回归（查询侧）：Collection search 遇 NaN/±Inf 查询返回 400，合法查询正常。
/// EN: Collection search rejects NaN/±Inf queries with 400.
#[tokio::test]
async fn collection_search_rejects_non_finite_query() {
    use houyi_server::http_handler::handle_collection_search;
    use houyi_server::types::SearchRequest;
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "c-q",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: None,
        }],
    )
    .await;

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = handle_collection_search(
            State(state.clone()),
            Path("c-q".to_string()),
            None,
            Json(SearchRequest {
                vector: vec![1.0, bad, 0.0, 0.0],
                k: 5,
                filter_gt_id: None,
                ef_search: None,
                filter_label: None,
                include_payload: false,
            }),
        )
        .await
        .expect_err("non-finite query must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    let ok = collection_search(&state, "c-q", vec![1.0, 0.0, 0.0, 0.0], 1, None).await;
    assert_eq!(ok.ids, vec![1]);
}

/// OCR 回归（查询侧）：hybrid search 同样在谓词构造/加锁前拒绝非有限查询（400）。
/// EN: Hybrid search rejects non-finite queries with 400 before predicates/locks.
#[tokio::test]
async fn collection_hybrid_search_rejects_non_finite_query() {
    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "c-h",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("red".to_string()),
            payload: None,
        }],
    )
    .await;

    // 带过滤条件也要在解释谓词之前因非有限向量 400（顺序：维度/有限性 → 谓词）。
    // Even with filters, finiteness is checked before predicate construction.
    let err = collection_hybrid_result(
        &state,
        "c-h",
        vec![1.0, f32::NAN, 0.0, 0.0],
        5,
        vec![("label", "eq", serde_json::json!("red"))],
    )
    .await
    .expect_err("non-finite hybrid query must be rejected");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);

    let ids = collection_hybrid(
        &state,
        "c-h",
        vec![1.0, 0.0, 0.0, 0.0],
        1,
        vec![("label", "eq", serde_json::json!("red"))],
    )
    .await;
    assert_eq!(ids, vec![1]);
}

/// OCR 第五轮回归（纯机械不变量）：过取候选必须「先剔除非有限距离、再截断到 k」，
/// 占据前部槽位的毒行（图后端 partial_cmp(NaN)=Equal 可让 NaN 落在任意位置）
/// 被剔除后由其后已取回的有限候选回填；先 truncate 再丢弃会让响应缩水。
/// EN: Over-fetched candidates drop non-finite distances BEFORE truncating to k,
/// so a poisoned top slot is backfilled by already-fetched finite candidates.
#[test]
fn finite_pairs_truncated_to_k_backfills_after_dropping_non_finite() {
    use houyi_server::http_handler::finite_pairs_truncated_to_k;
    let pairs = vec![(99, f32::NAN), (1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)];
    let kept = finite_pairs_truncated_to_k(pairs, 3);
    assert_eq!(
        kept,
        vec![(1, 0.1), (2, 0.2), (3, 0.3)],
        "finite rows behind the NaN slot must backfill the response"
    );
    let with_inf = vec![
        (1, 0.0),
        (2, f32::INFINITY),
        (3, 0.5),
        (4, f32::NEG_INFINITY),
    ];
    assert_eq!(
        finite_pairs_truncated_to_k(with_inf, 3),
        vec![(1, 0.0), (3, 0.5)]
    );
}

/// OCR 第五轮回归（collection search）：毒行经库层直插模拟闸门建立前的历史
/// pg_wire binary 脏数据；collection 路径与 benchmark 路径同一纪律（4k 过取 +
/// truncate 前剔除），毒行不得出现、响应不含 null。当前 HNSW 热缓冲 total_cmp
/// 把 NaN 排在末尾，本夹具锁的是可观察契约（结构防线）；回填机械不变量由
/// `finite_pairs_truncated_to_k_backfills_after_dropping_non_finite` 单独锁死。
/// EN: Poison seeded at library level models legacy pg_wire binary rows; the
/// collection path applies the same 4k over-fetch + pre-truncate drop discipline.
#[tokio::test]
async fn collection_search_drops_poisoned_hit() {
    let (state, _cleanup) = collection_test_state(4);
    // 经公开 upsert handler 创建 collection（get_or_create 为 pub(crate)，bin 测试不可见），
    // 再经公开的 collections.get() 拿后端，库层直插模拟闸门建立前的存量毒数据。
    // EN: Create the collection through the public upsert handler (get_or_create is
    // pub(crate) and invisible to bin tests), then seed legacy poison via library insert.
    collection_upsert(
        &state,
        "poison",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: None,
        }],
    )
    .await;
    let col = state
        .collections
        .get("poison")
        .expect("collection created by upsert");
    if let houyi_server::index_backend::IndexBackend::Hnsw(idx) = &col.backend {
        idx.insert(2, vec![0.0, 1.0, 0.0, 0.0]).await.unwrap();
        idx.insert(99, vec![1.0, f32::NAN, 0.0, 0.0]).await.unwrap();
    } else {
        panic!("collection_test_state default backend must be hnsw");
    }

    let resp = collection_search(&state, "poison", vec![1.0, 0.0, 0.0, 0.0], 5, None).await;
    assert_eq!(resp.ids, vec![1, 2], "poisoned id 99 must never appear");
    assert!(resp.scores.iter().all(|s| s.is_finite()));
    let json = serde_json::to_string(&resp).unwrap();
    assert!(
        !json.contains("null"),
        "serialized scores must not contain null: {json}"
    );
}

/// OCR 第五轮回归（hybrid search，无过滤分支）：与 collection search 同一过取/剔除
/// 纪律；此前 hybrid 只在响应转换时丢弃非有限距离（缩水且无回填）。
/// EN: The hybrid handler (unfiltered branch here) shares the over-fetch +
/// pre-truncate drop discipline; it previously dropped non-finite pairs only at
/// response conversion, shrinking the result with no backfill.
#[tokio::test]
async fn collection_hybrid_search_drops_poisoned_hit() {
    let (state, _cleanup) = collection_test_state(4);
    // 同 collection_search 毒行回归：经公开 upsert 建集合，库层直插存量毒数据。
    // EN: Same seeding route as the collection-search regression: create via public
    // upsert, then inject the legacy poison row directly at library level.
    collection_upsert(
        &state,
        "poison-h",
        vec![houyi_server::types::UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: None,
        }],
    )
    .await;
    let col = state
        .collections
        .get("poison-h")
        .expect("collection created by upsert");
    if let houyi_server::index_backend::IndexBackend::Hnsw(idx) = &col.backend {
        idx.insert(2, vec![0.0, 1.0, 0.0, 0.0]).await.unwrap();
        idx.insert(99, vec![1.0, f32::NAN, 0.0, 0.0]).await.unwrap();
    } else {
        panic!("collection_test_state default backend must be hnsw");
    }

    let resp = collection_hybrid_result(&state, "poison-h", vec![1.0, 0.0, 0.0, 0.0], 5, vec![])
        .await
        .expect("unfiltered hybrid search succeeds");
    assert_eq!(resp.ids, vec![1, 2]);
    assert!(resp.scores.iter().all(|s| s.is_finite()));
    assert!(!serde_json::to_string(&resp).unwrap().contains("null"));
}

/// OCR 第五轮回归（benchmark search 无过滤分支）：过取不再只对过滤查询生效——
/// 无 filter 时同样按 4k 过取并在 truncate 前剔毒，三条检索路径口径一致。
/// EN: The unfiltered benchmark path now over-fetches 4k and drops non-finite
/// hits before truncate too, aligning all three search handlers.
#[tokio::test]
async fn benchmark_search_unfiltered_drops_poisoned_hit() {
    use houyi_server::http_handler::benchmark_search;
    use houyi_server::types::SearchRequest;
    let (state, _cleanup) = collection_test_state(4);
    if let houyi_server::index_backend::IndexBackend::Hnsw(idx) = &state.index {
        idx.insert(1, vec![1.0, 0.0, 0.0, 0.0]).await.unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0, 0.0]).await.unwrap();
        idx.insert(99, vec![1.0, f32::NAN, 0.0, 0.0]).await.unwrap();
    } else {
        panic!("collection_test_state default backend must be hnsw");
    }
    let resp = benchmark_search(
        State(state.clone()),
        None,
        Json(SearchRequest {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k: 5,
            filter_gt_id: None,
            ef_search: None,
            filter_label: None,
            include_payload: false,
        }),
    )
    .await
    .expect("search with poisoned stored row still succeeds");
    assert_eq!(resp.0.ids, vec![1, 2]);
    assert!(resp.0.scores.iter().all(|s| s.is_finite()));
    assert!(!serde_json::to_string(&resp.0).unwrap().contains("null"));
}

/// 回归（端到端冒烟发现的生产缺口）：注入 builder 配置的 CollectionManager 必须为
/// Collection 的 HNSW 后端启动异步构图 worker —— upsert 后无需手动 drain，worker
/// 自动把待构图队列 flush 成可检索状态。修复前生产仅主索引有 worker，Collection
/// 写入永远滞留在 builder_queue，检索恒空。
/// EN: Regression (production gap caught by smoke testing): a manager with builder
/// config must start the async builder worker for collection HNSW backends — after
/// upsert, vectors become searchable with no manual drain. Previously only the main
/// index had a worker; collection writes were stuck in builder_queue forever.
#[tokio::test]
async fn collection_hnsw_upsert_becomes_searchable_via_builder_worker() {
    use houyi_server::collection::CollectionBuilderConfig;
    use houyi_server::http_handler::{handle_collection_search, handle_collection_upsert};
    use houyi_server::types::{SearchRequest, UpsertRequest, UpsertVector};
    use houyi_vector::FlushPolicy;

    let (state, _cleanup) = collection_test_state_full(
        4,
        "hnsw",
        Some(CollectionBuilderConfig {
            threshold: 1,
            tick_ms: 5,
            policy: FlushPolicy::Hybrid,
        }),
    );

    // 直调 handler，不做单测惯用的手动 run_async_builder_once 排空。
    // EN: Call the handler directly — deliberately no manual builder drain.
    let req = UpsertRequest {
        vectors: vec![UpsertVector {
            id: 7,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: None,
            payload: None,
        }],
        index: None,
    };
    let upserted =
        handle_collection_upsert(State(state.clone()), Path("c-w".to_string()), Json(req))
            .await
            .expect("upsert should succeed");
    assert_eq!(upserted.inserted, 1);

    let search = |state: BenchmarkHttpState| async move {
        handle_collection_search(
            State(state),
            Path("c-w".to_string()),
            None,
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: 1,
                filter_gt_id: None,
                ef_search: None,
                filter_label: None,
                include_payload: false,
            }),
        )
        .await
        .expect("search should succeed")
        .0
    };

    // 轮询等待 worker flush（5 tick 内应完成；上限 5s 防悬挂）。
    // EN: Poll until the worker flushes (should take a few ticks; cap at 5s).
    let mut resp = search(state.clone()).await;
    for _ in 0..500 {
        if resp.ids == vec![7] {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        resp = search(state.clone()).await;
    }
    assert_eq!(
        resp.ids,
        vec![7],
        "builder worker must flush the collection pending queue"
    );
    assert_eq!(resp.scores.len(), 1);
    assert!((resp.scores[0] - 1.0).abs() < 1e-5);
}

/// /readyz 行为守卫（P4.0.5）：空队列 → 200 + status=ready，响应携带水位线与队列
/// 观测字段（摘流决策方可归因）。
/// EN: Empty builder queue → 200 ready; backlog/watermark fields keep the
/// drain decision attributable.
#[tokio::test]
async fn readyz_reports_ready_with_backlog_fields() {
    use axum::{body::Body, http::Request, routing::get, Router};
    use tower::ServiceExt;

    let (state, _cleanup) = collection_test_state(4);
    state
        .index
        .insert(7, vec![0.1, 0.2, 0.3, 0.4])
        .await
        .unwrap();
    let app = Router::new()
        .route("/readyz", get(houyi_server::http_handler::benchmark_readyz))
        .with_state(state);

    let req = Request::builder()
        .method("GET")
        .uri("/readyz")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], serde_json::json!("ready"));
    assert_eq!(v["pending_build"], serde_json::json!(0));
    assert_eq!(v["hot_buffer"], serde_json::json!(1));
    assert_eq!(v["indexed"], serde_json::json!(0));
    assert!(v["watermark"].is_number());
}

/// /readyz 超水位分支（P4.0.5）：待构图队列 > 水位线 → 503 + status=not_ready，
/// 响应字段与 200 分支同形（K8s 摘流期间仍可观测积压归因）。
/// EN: Builder queue above watermark → 503 not_ready with the same body shape,
/// so backlog attribution stays observable while drained from rotation.
#[tokio::test]
async fn readyz_reports_not_ready_above_watermark() {
    use axum::{body::Body, http::Request, routing::get, Router};
    use tower::ServiceExt;

    let (state, _cleanup) = collection_test_state(4);
    // 从生产常量派生（而非另写 65537 字面量）：水位线调参时本测试同步跟踪，
    // 不会以一个令人困惑的 "expected 503, got 200" 静默失真。
    // EN: Derived from production constants (not a separate 65537 literal): tuning the
    // EN: watermark updates this test too, avoiding a confusing "expected 503, got 200" drift.
    const OVER_WATERMARK: usize = houyi_server::http_handler::READYZ_PENDING_WATERMARK + 1;
    let batch: Vec<(u64, Vec<f32>)> = (0..OVER_WATERMARK as u64)
        .map(|id| (id, vec![0.01, 0.02, 0.03, 0.04]))
        .collect();
    state.index.insert_batch(batch).await.unwrap();
    let app = Router::new()
        .route("/readyz", get(houyi_server::http_handler::benchmark_readyz))
        .with_state(state);

    let req = Request::builder()
        .method("GET")
        .uri("/readyz")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], serde_json::json!("not_ready"));
    assert_eq!(v["pending_build"], serde_json::json!(OVER_WATERMARK));
    assert!(v["hot_buffer"].is_number());
    assert!(v["indexed"].is_number());
    assert!(v["watermark"].is_number());
}

/// `/metrics` 行为守卫（P4.0.4）：Prometheus text exposition 契约——
/// content-type 为 text/plain; version=0.0.4，8 个核心指标名各自出现且带数值行。
/// EN: `/metrics` behavior guard (P4.0.4): the Prometheus text exposition contract —
/// EN: content-type is text/plain; version=0.0.4, and 8 core metric names each appear with a value
/// EN: line.
#[tokio::test]
async fn metrics_exposes_prometheus_text_format() {
    use axum::{body::Body, http::Request, routing::get, Router};
    use tower::ServiceExt;

    let (state, _cleanup) = collection_test_state(4);
    // 播种非零且互异的 pgwire 计数：全零快照无法暴露同类型字段交叉接线
    // （如 handshakes 误接 frontend_messages），数值来源必须可区分。
    // EN: Seed distinct non-zero pgwire counters so the guard also verifies
    // value provenance, not just metric-name presence.
    state
        .pgwire_metrics
        .handshake_count
        .store(17, std::sync::atomic::Ordering::Relaxed);
    state
        .pgwire_metrics
        .simple_query_count
        .store(23, std::sync::atomic::Ordering::Relaxed);
    state
        .pgwire_metrics
        .frontend_messages
        .store(97, std::sync::atomic::Ordering::Relaxed);
    // 图快照计数（P4.4.0e）同样播种互异非零值，验证 4 个新系列的接线来源。
    // EN: Graph snapshot counters (P4.4.0e) also seed distinct nonzero values to verify the 4 new
    // EN: series.
    state
        .graph_snapshot_metrics
        .writes_total
        .store(5, std::sync::atomic::Ordering::Relaxed);
    state
        .graph_snapshot_metrics
        .loads_total
        .store(7, std::sync::atomic::Ordering::Relaxed);
    state
        .graph_snapshot_metrics
        .rejects_total
        .store(3, std::sync::atomic::Ordering::Relaxed);
    state
        .graph_snapshot_metrics
        .bytes_written_total
        .store(12345, std::sync::atomic::Ordering::Relaxed);
    let app = Router::new()
        .route(
            "/metrics",
            get(houyi_server::http_handler::benchmark_metrics),
        )
        .with_state(state);

    let req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
    let ct = res
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    // 抓取器靠 version 参数协商暴露格式，只查 text/plain 前缀会放过版本漂移。
    // EN: Scrapers negotiate the format via the version parameter; checking only the
    // EN: text/plain prefix would miss version drift.
    assert!(
        ct.starts_with("text/plain") && ct.contains("version=0.0.4"),
        "content-type must be Prometheus text exposition v0.0.4, got {ct}"
    );
    let body = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    for name in [
        "houyidb_index_hot_buffer",
        "houyidb_index_indexed_vectors",
        "houyidb_index_pending_build",
        "houyidb_index_flush_total",
        "houyidb_index_flushed_items_total",
        "houyidb_wal_active_bytes",
        "houyidb_pgwire_handshakes_total",
        "houyidb_pgwire_queries_total",
        "houyidb_graph_snapshot_writes_total",
        "houyidb_graph_snapshot_loads_total",
        "houyidb_graph_snapshot_rejects_total",
        "houyidb_graph_snapshot_bytes_written_total",
    ] {
        // 每个指标须同时有 HELP/TYPE 元信息与一行 `name <value>` 样本。
        // EN: Each metric needs both HELP/TYPE metadata and a `name <value>` sample line.
        assert!(
            text.contains(&format!("# HELP {name} ")),
            "missing HELP for {name}"
        );
        assert!(
            text.contains(&format!("# TYPE {name} ")),
            "missing TYPE for {name}"
        );
        let sample = text
            .lines()
            .find(|l| l.starts_with(&format!("{name} ")))
            .unwrap_or_else(|| panic!("missing sample line for {name}"));
        let value = sample.rsplit(' ').next().unwrap();
        assert!(
            value.parse::<u64>().is_ok(),
            "sample value not numeric for {name}: {value}"
        );
    }
    // 播种值精确落位：验证指标与来源字段的接线关系（防同类型字段互换）。
    // EN: Seeded values land exactly: verifies metric-to-source wiring (prevents swapping same-type
    // EN: fields).
    assert!(
        text.lines()
            .any(|l| l == "houyidb_pgwire_handshakes_total 17"),
        "handshakes sample must equal seeded 17"
    );
    assert!(
        text.lines().any(|l| l == "houyidb_pgwire_queries_total 23"),
        "queries sample must equal seeded 23"
    );
    for (name, seeded) in [
        ("houyidb_graph_snapshot_writes_total", "5"),
        ("houyidb_graph_snapshot_loads_total", "7"),
        ("houyidb_graph_snapshot_rejects_total", "3"),
        ("houyidb_graph_snapshot_bytes_written_total", "12345"),
    ] {
        assert!(
            text.lines().any(|l| l == format!("{name} {seeded}")),
            "{name} sample must equal seeded {seeded}"
        );
    }
}

/// OpenAPI 文档守卫：内嵌 spec 必须是合法 JSON，且核心路由一条不缺——
/// 防止以后改路由忘了同步 houyi-server/openapi.json。
/// EN: OpenAPI doc guard: the embedded spec must be valid JSON and include every core route —
/// EN: prevents forgetting to sync houyi-server/openapi.json when routes change.
#[test]
fn openapi_spec_is_valid_json_and_covers_core_routes() {
    let spec: serde_json::Value =
        serde_json::from_str(houyi_server::http_handler::OPENAPI_SPEC).expect("spec must parse");
    assert_eq!(spec["openapi"], serde_json::json!("3.1.0"));
    for path in [
        "/v1/health",
        "/readyz",
        "/metrics",
        "/v1/openapi.json",
        "/v1/collections",
        "/v1/collections/{name}/upsert",
        "/v1/collections/{name}/search",
        "/v1/collections/{name}/hybrid/search",
        "/v1/collections/{name}/delete",
        "/v1/collections/{name}/status",
        "/v1/index/status",
        "/v1/embeddings",
        // Sentinel 控制面（EE M1）。 EN: Sentinel control plane (EE M1).
        "/v1/version",
        "/v1/admin/lease",
        "/v1/admin/promote",
        "/v1/admin/demote",
        "/v1/admin/replication/status",
        "/v1/admin/whoami",
        "/v1/admin/diagnostics",
        "/v1/admin/query-log",
        "/v1/admin/query-stats",
    ] {
        assert!(
            spec["paths"].get(path).is_some(),
            "openapi.json missing path {path}"
        );
    }

    // /readyz 两个状态码必须 $ref 命名组件，且组件字段集与 Rust
    // ReadyzResponse 结构体字段一一对应（双向漂移守卫）。
    // EN: The two /readyz status codes must $ref named components, and the component field set
    // EN: must match the Rust ReadyzResponse struct one-to-one (two-way drift guard).
    let component = &spec["components"]["schemas"]["ReadyzResponse"];
    assert!(component.is_object(), "missing named ReadyzResponse schema");
    for code in ["200", "503"] {
        let reference = &spec["paths"]["/readyz"]["get"]["responses"][code]["content"]
            ["application/json"]["schema"]["$ref"];
        assert_eq!(
            reference.as_str(),
            Some("#/components/schemas/ReadyzResponse"),
            "/readyz {code} must $ref the named component"
        );
    }
    // 真·双向守卫：schema 字段集不再对手写清单比对（手写清单本身会漂移），
    // 而是序列化一个真实 ReadyzResponse 样本取其 JSON 键集——结构体加/删/
    // 改名任一侧字段都会在一侧爆炸（serde 重命名同样被捕获）。
    // EN: Bidirectional guard: compare the schema properties against the JSON
    // key set of a serialized ReadyzResponse sample, not a hardcoded list.
    let sample = houyi_server::types::ReadyzResponse {
        status: "ready",
        pending_build: 0,
        hot_buffer: 0,
        indexed: 0,
        watermark: 0,
    };
    let sample_value = serde_json::to_value(&sample).expect("serialize ReadyzResponse");
    let mut rust_fields: Vec<&str> = sample_value
        .as_object()
        .expect("ReadyzResponse must serialize to a JSON object")
        .keys()
        .map(|s| s.as_str())
        .collect();
    rust_fields.sort_unstable();
    let mut schema_fields: Vec<&str> = component["properties"]
        .as_object()
        .expect("properties object")
        .keys()
        .map(|s| s.as_str())
        .collect();
    schema_fields.sort_unstable();
    assert_eq!(
        schema_fields, rust_fields,
        "ReadyzResponse schema fields must mirror the serialized Rust struct key set"
    );
}

/// 认证中间件行为守卫（P4.2.1）：探针豁免、/metrics 纳入保护、
/// Bearer 与 X-Api-Key 双通道、无凭据/错误凭据一律 401。
/// EN: Auth middleware behavior guard (P4.2.1): probes exempt, /metrics protected,
/// EN: Bearer and X-Api-Key both accepted, missing/wrong credentials always 401.
#[tokio::test]
async fn require_auth_guards_routes_and_exempts_probes() {
    use axum::{body::Body, http::Request, http::StatusCode, routing::get, Router};
    use std::sync::Arc;
    use tower::ServiceExt;

    let auth_state = Arc::new(houyi_server::rbac::AuthzState::new(
        Arc::new(
            houyi_server::rbac::AuthzPolicy::from_config(Some("s3cret"), &[])
                .expect("legacy policy"),
        ),
        None,
    ));
    let app = Router::new()
        .route("/v1/health", get(|| async { "ok" }))
        .route("/readyz", get(|| async { "ready" }))
        .route("/metrics", get(|| async { "m" }))
        .route("/v1/collections", get(|| async { "data" }))
        .layer(axum::middleware::from_fn_with_state(
            auth_state,
            houyi_server::http_handler::require_auth,
        ));

    async fn status(
        app: &Router,
        uri: &str,
        auth: Option<(&'static str, &'static str)>,
    ) -> StatusCode {
        let mut b = Request::builder().method("GET").uri(uri);
        if let Some((k, v)) = auth {
            b = b.header(k, v);
        }
        app.clone()
            .oneshot(b.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    // 探针豁免（orchestrator 不带凭据）。
    // EN: Probes are exempt (orchestrators carry no credentials).
    assert_eq!(status(&app, "/v1/health", None).await, StatusCode::OK);
    assert_eq!(status(&app, "/readyz", None).await, StatusCode::OK);
    // 无凭据 401——/metrics 同样受保护（运行形态信息不外泄）。
    // EN: No credentials -> 401; /metrics is protected too (no runtime-shape leak).
    assert_eq!(
        status(&app, "/v1/collections", None).await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        status(&app, "/metrics", None).await,
        StatusCode::UNAUTHORIZED
    );
    // 错误凭据 401。
    // EN: Wrong credentials -> 401.
    assert_eq!(
        status(
            &app,
            "/v1/collections",
            Some(("authorization", "Bearer nope"))
        )
        .await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        status(&app, "/v1/collections", Some(("x-api-key", "nope"))).await,
        StatusCode::UNAUTHORIZED
    );
    // Bearer 与 X-Api-Key 双通道放行。
    // EN: Both Bearer and X-Api-Key are accepted.
    assert_eq!(
        status(
            &app,
            "/v1/collections",
            Some(("authorization", "Bearer s3cret"))
        )
        .await,
        StatusCode::OK
    );
    assert_eq!(
        status(&app, "/v1/collections", Some(("x-api-key", "s3cret"))).await,
        StatusCode::OK
    );
}

/// 401 契约与 Bearer scheme 形状钉（OCR 359d85d[2][10][11]）：
/// - 401 必须是 text/plain 且携带 `WWW-Authenticate: Bearer`（RFC 7235 §3.1）；
/// - auth-scheme 大小写不敏感（bearer/BEARER 同义）；scheme 后 1*SP 容忍；
/// - 裸 token、无空格粘连、其他 scheme 一律 401——这些行为是被钉住的
///   决定，不是实现巧合。
/// EN: 401 contract and Bearer scheme shape pinning (OCR 359d85d[2][10][11]):
/// EN: - 401 must be text/plain with `WWW-Authenticate: Bearer` (RFC 7235 §3.1);
/// EN: - auth-scheme is case-insensitive (bearer/BEARER equivalent); 1*SP after scheme tolerated;
/// EN: - bare token, glued token without space, and other schemes are always 401 — these
/// EN:   are pinned decisions, not implementation coincidences.
#[tokio::test]
async fn require_auth_401_contract_and_bearer_scheme_shapes() {
    use axum::{body::Body, http::Request, http::StatusCode, routing::get, Router};
    use std::sync::Arc;
    use tower::ServiceExt;

    let auth_state = Arc::new(houyi_server::rbac::AuthzState::new(
        Arc::new(
            houyi_server::rbac::AuthzPolicy::from_config(Some("s3cret"), &[])
                .expect("legacy policy"),
        ),
        None,
    ));
    let app = Router::new()
        .route("/v1/collections", get(|| async { "data" }))
        .layer(axum::middleware::from_fn_with_state(
            auth_state,
            houyi_server::http_handler::require_auth,
        ));

    async fn call(app: &Router, auth: Option<&str>) -> axum::response::Response {
        let mut b = Request::builder().method("GET").uri("/v1/collections");
        if let Some(v) = auth {
            b = b.header("authorization", v);
        }
        app.clone()
            .oneshot(b.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    // 401 契约：text/plain 正文 + WWW-Authenticate 质询头。
    // EN: 401 contract: text/plain body plus a WWW-Authenticate challenge header.
    let resp = call(&app, None).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        resp.headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer"),
        "401 must carry WWW-Authenticate: Bearer (RFC 7235 §3.1)"
    );
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        content_type.starts_with("text/plain"),
        "401 body must be text/plain, got {content_type}"
    );

    // scheme 大小写不敏感（RFC 7235）。
    // EN: Scheme is case-insensitive (RFC 7235).
    for header in ["bearer s3cret", "BEARER s3cret", "Bearer s3cret"] {
        assert_eq!(
            call(&app, Some(header)).await.status(),
            StatusCode::OK,
            "{header:?} should authenticate"
        );
    }
    // scheme 后 1*SP 容忍（token 前导空格 trim，本体不 trim）。
    // EN: Tolerate 1*SP after the scheme (trim leading token spaces only; body untouched).
    assert_eq!(
        call(&app, Some("Bearer  s3cret")).await.status(),
        StatusCode::OK
    );
    // 拒绝形状：裸 token / 无空格粘连 / 其他 scheme / token 带尾随空格。
    // EN: Rejected shapes: bare token / glued without space / other scheme / trailing-space token.
    for header in [
        "s3cret",
        "Bearers3cret",
        "Basic s3cret",
        "Bearer s3cret ",
        "Bearer ",
    ] {
        assert_eq!(
            call(&app, Some(header)).await.status(),
            StatusCode::UNAUTHORIZED,
            "{header:?} must be rejected"
        );
    }
}

/// 在途水位线中间件行为守卫（P4.2.3）：占满后业务路由 503，
/// /v1/health /readyz /metrics 豁免；许可归还后恢复放行。
/// EN: In-flight watermark middleware guard (P4.2.3): when saturated, business routes
/// EN: return 503; /v1/health /readyz /metrics stay exempt; permits returning restores admission.
#[tokio::test]
async fn limit_http_inflight_rejects_overflow_and_exempts_probes() {
    use axum::{body::Body, http::Request, http::StatusCode, routing::get, Router};
    use std::sync::Arc;
    use tower::ServiceExt;

    let limiter = Arc::new(InflightWatermark::new(1));
    let app = Router::new()
        .route("/v1/health", get(|| async { "ok" }))
        .route("/readyz", get(|| async { "ready" }))
        .route("/metrics", get(|| async { "m" }))
        .route("/v1/collections", get(|| async { "data" }))
        .layer(axum::middleware::from_fn_with_state(
            limiter.clone(),
            houyi_server::http_handler::limit_http_inflight,
        ));

    async fn status(app: &Router, uri: &str) -> StatusCode {
        app.clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    // 占住唯一名额（模拟一个在途慢请求）。
    // EN: Occupy the only slot (simulating one in-flight slow request).
    let permit = limiter.try_acquire().expect("first slot");
    assert_eq!(
        status(&app, "/v1/collections").await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(limiter.rejected_total(), 1);
    // 探针与指标在饱和时仍必须可达。
    // EN: Probes and metrics must remain reachable while saturated.
    assert_eq!(status(&app, "/v1/health").await, StatusCode::OK);
    assert_eq!(status(&app, "/readyz").await, StatusCode::OK);
    assert_eq!(status(&app, "/metrics").await, StatusCode::OK);

    // 许可归还后恢复放行。
    // EN: Traffic is admitted again after the permit returns.
    drop(permit);
    assert_eq!(status(&app, "/v1/collections").await, StatusCode::OK);
    // 请求完成后名额应已自动归还（RAII）——再放行一次证明无泄漏。
    // EN: The slot returns automatically after completion (RAII); admitting once more proves no
    // EN: leak.
    assert_eq!(status(&app, "/v1/collections").await, StatusCode::OK);
    assert_eq!(limiter.active(), 0);
}

/// 内存水位线写入面接线守卫（P4.2.4）：max=1 字节时——
/// Linux（有 RSS 源）upsert 被 503 拒绝且不计入索引；无 RSS 源的平台 fail-open 放行。
/// EN: Memory-watermark write-path wiring guard (P4.2.4): with max=1 byte —
/// EN: Linux (RSS source) rejects upserts with 503 and does not index them; platforms without RSS
/// EN: fail open.
#[tokio::test]
async fn memory_watermark_guards_http_upsert_path() {
    use axum::{body::Body, http::Request, http::StatusCode, routing::post, Router};
    use houyi_server::http_handler::benchmark_upsert;
    use houyi_server::mem::MemoryWatermark;
    use tower::ServiceExt;

    let path = std::env::temp_dir().join(format!(
        "houyidb-mem-watermark-test-{}-{}.wal",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    let (wal, _) = super::WalHandle::open(&path, super::WalWriteMode::Async).unwrap();
    let metrics = std::sync::Arc::new(PgWireMetrics {
        memory_watermark: std::sync::Arc::new(MemoryWatermark::new(1)),
        ..Default::default()
    });
    let app = Router::new()
        .route("/v1/vectors/upsert", post(benchmark_upsert))
        .with_state(BenchmarkHttpState {
            audit: None,
            encryption: None,
            promote: None,
            collections: std::sync::Arc::new(CollectionManager::new(4, "hnsw")),
            index: IndexBackend::Hnsw(Arc::new(HotBufferIndex::new(4))),
            wal: Arc::new(wal),
            collections_wal: Arc::new(
                super::WalHandle::open(
                    &houyi_persistence::collection_wal::collections_wal_path_for(&path),
                    super::WalWriteMode::Async,
                )
                .unwrap()
                .0,
            ),
            vector_dim: 4,
            pgwire_metrics: Arc::clone(&metrics),
            applied_sidecar: path.with_extension("applied_lsn"),
            applied_sidecar_lock: Arc::new(Mutex::new(())),
            wal_path: path.clone(),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            openai_upstream: None,
            http_inflight: Arc::new(InflightWatermark::new(DEFAULT_MAX_HTTP_INFLIGHT as u64)),
            graph_snapshot_metrics: Arc::new(
                houyi_server::recovery::GraphSnapshotMetrics::default(),
            ),
            role: houyi_server::types::RoleState::default(),
            cluster: test_cluster(),
            query_log: None,
            replication_metrics: houyi_server::types::shared_replication_metrics(None),
            lifecycle: std::sync::Arc::new(
                houyi_server::collection_lifecycle::CollectionLifecycleState::for_test(),
            ),
        });

    let body = serde_json::json!({
        "vectors": [{ "id": 1, "values": [1.0, 0.0, 0.0, 0.0] }]
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/vectors/upsert")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();

    match houyi_server::mem::current_rss_bytes() {
        Some(rss) if rss > 1 => {
            assert_eq!(
                res.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "tripped watermark must 503 the upsert"
            );
            assert_eq!(metrics.memory_watermark.rejected_total(), 1);
        }
        _ => {
            // 无 RSS 源（macOS 等）fail-open：写入照常成功。
            // EN: No RSS source (macOS etc.) fails open: writes still succeed.
            assert_eq!(res.status(), StatusCode::OK, "fail-open must admit");
            assert_eq!(metrics.memory_watermark.rejected_total(), 0);
        }
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(houyi_persistence::collection_wal::collections_wal_path_for(
        &path,
    ));
    let _ = std::fs::remove_file(path.with_extension("applied_lsn"));
}

// ── P4.3.1 TOML 配置文件 ────────────────────────────────────────
// EN: P4.3.1 TOML config file.

/// TOML 应用语义：写出的键覆盖默认，未写出的键（worker_tick_ms 等）保持 default。
/// 注：并非真正的"全键"往返——ServerConfigFile 覆盖的是运维常用键子集，
/// 新增可配置键时应同步扩展本夹具。
/// EN: TOML apply semantics: written keys override defaults; unwritten keys (worker_tick_ms
/// EN: etc.) keep their defaults. Note: this is not a true "all keys" round-trip —
/// EN: ServerConfigFile covers the operational subset; extend this fixture when adding keys.
#[test]
fn toml_config_file_applies_written_keys_only() {
    let dir = std::env::temp_dir().join(format!(
        "houyidb-toml-cfg-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("houyidb.toml");
    std::fs::write(
        &path,
        r#"
vector_dim = 768
benchmark_http_addr = "0.0.0.0:8000"
pgwire_addr = "0.0.0.0:15432"
flush_policy = "interval"
wal_write_mode = "sync"
worker_threshold = 64
max_pgwire_connections = 50
max_http_inflight = 128
max_memory_bytes = 1073741824
slow_query_ms = 250
snapshot_interval_secs = 120
snapshot_wal_threshold_bytes = 1048576
collections_scalar_compact_interval_secs = 42
collections_scalar_compact_stale_ratio = 0.25
tls_cert_path = "/etc/houyi/cert.pem"
tls_key_path = "/etc/houyi/key.pem"
auth_token = "file-token"
index_type = "hii-l0"
"#,
    )
    .unwrap();

    let mut cfg = super::ServerConfig::default();
    super::apply_toml_config_file(&mut cfg, &path).expect("valid toml must apply");
    assert_eq!(cfg.vector_dim, 768);
    assert_eq!(cfg.benchmark_http_addr, "0.0.0.0:8000");
    assert_eq!(cfg.pgwire_addr, "0.0.0.0:15432");
    assert_eq!(cfg.flush_policy, houyi_vector::FlushPolicy::IntervalOnly);
    assert_eq!(cfg.wal_write_mode, super::WalWriteMode::Sync);
    assert_eq!(cfg.worker_threshold, 64);
    assert_eq!(cfg.max_pgwire_connections, 50);
    assert_eq!(cfg.max_http_inflight, 128);
    assert_eq!(cfg.max_memory_bytes, 1073741824);
    assert_eq!(cfg.slow_query_ms, 250);
    assert_eq!(cfg.snapshot_interval_secs, 120);
    assert_eq!(cfg.snapshot_wal_threshold_bytes, 1048576);
    assert_eq!(cfg.collections_scalar_compact_interval_secs, 42);
    assert_eq!(cfg.collections_scalar_compact_stale_ratio, 0.25);
    assert_eq!(cfg.tls_cert_path.as_deref(), Some("/etc/houyi/cert.pem"));
    assert_eq!(cfg.tls_key_path.as_deref(), Some("/etc/houyi/key.pem"));
    assert_eq!(cfg.auth_token.as_deref(), Some("file-token"));
    assert_eq!(cfg.index_type, "hii-l0");
    // 未写出的键保持默认。
    // EN: Unwritten keys keep their defaults.
    let default = super::ServerConfig::default();
    assert_eq!(cfg.worker_tick_ms, default.worker_tick_ms);
    assert_eq!(cfg.openai_timeout_secs, default.openai_timeout_secs);
    assert_eq!(cfg.pgwire_hnsw_m, default.pgwire_hnsw_m);

    std::fs::remove_dir_all(&dir).ok();
}

/// 图快照触发旋钮默认值（P4.4.0e）：周期 600s、WAL 增量阈值 256MB，默认开启；
/// 0 = 关闭对应通道（不校验下限，0 是合法语义而非配置笔误）。
/// EN: Graph snapshot trigger defaults (P4.4.0e): 600s period, 256MB WAL delta, enabled;
/// EN: 0 disables the channel (no lower-bound check; 0 is legal semantics, not a typo).
#[test]
fn snapshot_trigger_defaults_are_on() {
    let cfg = super::ServerConfig::default();
    assert_eq!(cfg.snapshot_interval_secs, 600);
    assert_eq!(cfg.snapshot_wal_threshold_bytes, 268_435_456);
    assert_eq!(cfg.collections_scalar_compact_interval_secs, 600);
    assert_eq!(cfg.collections_scalar_compact_stale_ratio, 0.5);
}

// ── 复制配置（P4.4.1）────────────────────────────────────
// EN: Replication config (P4.4.1).

/// role 枚举解析：三值 roundtrip + 笔误 fast-fail（不静默落 disabled——
/// 运维写 "folower" 时静默落单机是脑裂级事故）。
/// EN: Role parse roundtrip; a typo is fatal — silently falling back to
/// standalone would be a split-brain-class incident.
#[test]
fn replication_role_parse_roundtrip_and_typo_fatal() {
    use houyi_server::types::{parse_replication_role, ReplicationRole};
    assert_eq!(
        parse_replication_role("disabled").unwrap(),
        ReplicationRole::Disabled
    );
    assert_eq!(
        parse_replication_role("leader").unwrap(),
        ReplicationRole::Leader
    );
    assert_eq!(
        parse_replication_role("follower").unwrap(),
        ReplicationRole::Follower
    );
    assert_eq!(
        parse_replication_role(" Leader ").unwrap(),
        ReplicationRole::Leader
    );
    // enum→string 方向（as_str 供启动日志 replication_role 字段与配置回显）——
    // 双向都过才算 roundtrip。
    // EN: Complete the roundtrip by asserting parse(as_str(role)) for all variants.
    for role in [
        ReplicationRole::Disabled,
        ReplicationRole::Leader,
        ReplicationRole::Follower,
    ] {
        assert_eq!(parse_replication_role(role.as_str()).unwrap(), role);
    }
    let err = parse_replication_role("folower").unwrap_err();
    assert!(
        err.contains("folower"),
        "error must echo the bad value, got: {err}"
    );
    // 默认值 = 单机。
    // EN: Default = standalone.
    let cfg = super::ServerConfig::default();
    assert_eq!(cfg.role, ReplicationRole::Disabled);
    assert_eq!(cfg.replication_listen_addr, "0.0.0.0:9876");
    assert_eq!(cfg.replication_queue_batches, 4096);
    assert!(cfg.replication_leader_addr.is_none());
}

/// 组合校验：follower 缺 leader 地址 / 空 listen 地址 / 队列 0 / sync 模式
/// 全部 fast-fail；合法组合放行（含 disabled + sync 不受影响）。
/// EN: Combo validation — missing peer addr, empty listen addr, zero queue,
/// and sync WAL mode all fail fast; valid combos pass.
#[test]
fn replication_config_validation() {
    use houyi_server::types::ReplicationRole;
    let base = super::ServerConfig::default();
    super::validate_config(&base).expect("default (disabled) must pass");

    // follower 缺 leader_addr → 拒绝
    // EN: follower without leader_addr -> reject.
    let mut cfg = base.clone();
    cfg.role = ReplicationRole::Follower;
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("replication-leader-addr"), "got: {err}");

    // follower + leader_addr → 放行
    // EN: follower + leader_addr -> accept.
    cfg.replication_leader_addr = Some("10.0.0.1:9876".to_string());
    super::validate_config(&cfg).expect("follower with leader addr must pass");

    // follower + 空/畸形 listen_addr → 拒绝（promote 在 follower 上原地 bind
    // 该地址起 leader 监听，failover 中途才炸不可接受，启动期 fast-fail）。
    // EN: promote binds listen_addr in place on the follower — reject empty or
    // malformed addresses at boot, not mid-failover.
    cfg.replication_listen_addr = " ".to_string();
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("replication_listen_addr"), "got: {err}");
    cfg.replication_listen_addr = "not-an-addr".to_string();
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("replication_listen_addr"), "got: {err}");
    cfg.replication_listen_addr = "0.0.0.0:9876".to_string();
    super::validate_config(&cfg).expect("follower with valid listen addr must pass");

    // leader + 空 listen_addr → 拒绝；默认地址 → 放行
    // EN: leader + empty listen_addr -> reject; default address -> accept.
    let mut cfg = base.clone();
    cfg.role = ReplicationRole::Leader;
    super::validate_config(&cfg).expect("leader with default listen addr must pass");
    cfg.replication_listen_addr = "  ".to_string();
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("replication_listen_addr"), "got: {err}");

    // 队列 0 → 拒绝（第一条 publish 即踢，等同自我 DoS）
    // EN: Queue 0 -> reject (the first publish kicks, i.e. self-DoS).
    let mut cfg = base.clone();
    cfg.role = ReplicationRole::Leader;
    cfg.replication_queue_batches = 0;
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("replication_queue_batches"), "got: {err}");

    // role≠disabled + wal_write_mode=sync → 拒绝（复制依赖异步 WAL 原语）
    // EN: role != disabled + wal_write_mode=sync -> reject (replication needs async WAL
    // EN: primitives).
    let mut cfg = base.clone();
    cfg.role = ReplicationRole::Leader;
    cfg.wal_write_mode = super::WalWriteMode::Sync;
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("wal_write_mode"), "got: {err}");

    // follower + sync 同样拒绝——这条是承重组合：follower 装配走
    // collections_wal.async_handle().expect(...)，validate 是它 panic 路径上
    // 唯一的守卫；按角色拆分校验的重构不能让 follower 变体静默丢失。
    // EN: Follower+Sync is the load-bearing case — it is the only guard on the
    // async_handle().expect() panic path in follower assembly.
    let mut cfg = base.clone();
    cfg.role = ReplicationRole::Follower;
    cfg.replication_leader_addr = Some("10.0.0.1:9876".to_string());
    cfg.wal_write_mode = super::WalWriteMode::Sync;
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("wal_write_mode"), "got: {err}");

    // disabled + sync 不受复制校验影响
    // EN: disabled + sync is unaffected by replication validation.
    let mut cfg = base.clone();
    cfg.wal_write_mode = super::WalWriteMode::Sync;
    super::validate_config(&cfg).expect("disabled + sync must pass");

    // sync 超时 0 → 拒绝（0 = 每笔写静默降级 async，必为笔误；fast-fail 优于
    // 带着「看似半同步实则纯异步」的错觉运行）。
    // EN: timeout 0 would silently degrade every write to async — a typo, so fatal.
    // 校验挂在 role≠disabled 分支（与 queue_batches 同纪律：disabled 时该值
    // 是死配置，不误伤），故本用例显式切到 leader。
    let mut cfg = base.clone();
    cfg.role = ReplicationRole::Leader;
    cfg.replication_sync_timeout_ms = 0;
    let err = super::validate_config(&cfg).unwrap_err();
    assert!(err.contains("replication_sync_timeout_ms"), "got: {err}");
}

/// TOML 镜像：复制四键经文件通道生效，枚举笔误 fast-fail。
/// EN: TOML channel applies all four replication keys; enum typo is fatal.
#[test]
fn toml_config_replication_keys() {
    use houyi_server::types::ReplicationRole;
    let dir = std::env::temp_dir().join(format!(
        "houyidb-toml-repl-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("houyidb.toml");
    std::fs::write(
        &path,
        r#"
role = "follower"
replication_listen_addr = "0.0.0.0:19876"
replication_leader_addr = "10.0.0.2:9876"
replication_queue_batches = 1024
replication_sync_timeout_ms = 500
replication_auth_token = "shared-secret"
default_ha_level = "semi_sync"
audit_log_path = "/tmp/houyidb-audit.toml-test.log"
audit_include_reads = true
audit_max_bytes = 1048576

[[auth_principals]]
name = "ingest"
role = "writer"
token = "ingest-token"
collections = ["docs"]
max_collections = 5
max_inflight = 8
writes_per_sec = 100

[[auth_principals]]
name = "analytics"
role = "reader"
token_sha256 = "eb759c4f289ff540bc4f7b7a3af9e2dda169e7cdfefe42f732e4442325ce655b"
"#,
    )
    .unwrap();
    let mut cfg = super::ServerConfig::default();
    super::apply_toml_config_file(&mut cfg, &path).expect("valid toml must apply");
    assert_eq!(cfg.role, ReplicationRole::Follower);
    assert_eq!(cfg.replication_listen_addr, "0.0.0.0:19876");
    assert_eq!(
        cfg.replication_leader_addr.as_deref(),
        Some("10.0.0.2:9876")
    );
    assert_eq!(
        cfg.replication_auth_token.as_deref(),
        Some("shared-secret"),
        "replication auth token must load from TOML"
    );
    assert_eq!(cfg.replication_queue_batches, 1024);
    assert_eq!(cfg.replication_sync_timeout_ms, 500);
    assert_eq!(
        cfg.default_ha_level,
        houyi_persistence::ha_level::HaLevel::SemiSync
    );
    assert_eq!(
        cfg.audit_log_path.as_deref(),
        Some(std::path::Path::new("/tmp/houyidb-audit.toml-test.log")),
        "audit path must load from TOML"
    );
    assert!(cfg.audit_include_reads);
    assert_eq!(cfg.audit_max_bytes, 1_048_576);

    // RBAC 主体（P4.7.1）：两条 [[auth_principals]] 解析并可通过策略校验。
    // EN: RBAC principals (P4.7.1): both [[auth_principals]] parse and pass policy validation.
    assert_eq!(cfg.auth_principals.len(), 2);
    assert_eq!(cfg.auth_principals[0].name, "ingest");
    assert_eq!(cfg.auth_principals[0].role, "writer");
    assert_eq!(
        cfg.auth_principals[0].token.as_deref(),
        Some("ingest-token")
    );
    assert_eq!(
        cfg.auth_principals[0].collections.as_deref(),
        Some(&["docs".to_string()][..])
    );
    assert_eq!(cfg.auth_principals[1].name, "analytics");
    let policy = houyi_server::rbac::AuthzPolicy::from_config(
        cfg.auth_token.as_deref(),
        &cfg.auth_principals,
    )
    .expect("configured principals must validate");
    assert_eq!(policy.len(), 2);
    let ingest = policy.resolve("ingest-token").unwrap();
    assert_eq!(ingest.max_collections, Some(5));
    assert_eq!(ingest.max_inflight, 8);
    assert_eq!(ingest.writes_per_sec, Some(100));
    let reader = policy.resolve("analytics-token").unwrap();
    assert_eq!(reader.role, houyi_server::rbac::Role::Reader);
    assert_eq!(reader.max_inflight, 64, "default in-flight cap");

    // default_ha_level 笔误 fast-fail（与 role 同纪律：配置不可信不落默认）。
    // EN: default_ha_level typos fast-fail (same rule as role: untrusted config never falls back).
    std::fs::write(&path, r#"default_ha_level = "sycn""#).unwrap();
    let mut cfg = super::ServerConfig::default();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(
        err.contains("sycn"),
        "typo must echo the bad value, got: {err}"
    );

    // 复制认证 token 空白 = fail-closed（运维以为认证已开）。
    // EN: Blank replication auth token = fail-closed (operator believes auth is on).
    std::fs::write(&path, "replication_auth_token = \"   \"").unwrap();
    let mut cfg = super::ServerConfig::default();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(
        err.contains("replication_auth_token") && err.contains("blank"),
        "blank replication token must be fatal, got: {err}"
    );

    // 审计路径空白 = fail-closed（运维以为审计已开）。
    // EN: Blank audit path = fail-closed (operator believes audit is on).
    std::fs::write(&path, "audit_log_path = \"   \"").unwrap();
    let mut cfg = super::ServerConfig::default();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(
        err.contains("audit_log_path") && err.contains("blank"),
        "blank audit path must be fatal, got: {err}"
    );

    // 超长 token 启动期拒绝（OCR 批次二十八）：不能拖到每会话 encode 失败。
    // EN: Over-long token rejected at startup (OCR batch 28): cannot defer to per-session
    // EN: encode failure.
    let too_long = format!(
        "replication_auth_token = \"{}\"",
        "x".repeat(houyi_persistence::wal_replication::MAX_TOKEN_BYTES + 1)
    );
    std::fs::write(&path, too_long).unwrap();
    let mut cfg = super::ServerConfig::default();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(
        err.contains("exceeding"),
        "oversized replication token must be fatal, got: {err}"
    );

    std::fs::write(&path, r#"role = "primary""#).unwrap();
    let mut cfg = super::ServerConfig::default();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(
        err.contains("primary"),
        "typo must echo the bad value, got: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// 枚举字符串笔误必须 fast-fail，不静默落回默认。
/// EN: Misspelled enum strings must fast-fail, never silently fall back to the default.
#[test]
fn toml_config_enum_typo_is_fatal() {
    let dir = std::env::temp_dir().join(format!(
        "houyidb-toml-cfg-typo-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.toml");
    std::fs::write(&path, "flush_policy = \"bogus\"\n").unwrap();
    let mut cfg = super::ServerConfig::default();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(
        err.contains("bogus"),
        "error should name the bad value: {err}"
    );

    std::fs::write(&path, "wal_write_mode = \"sometimes\"\n").unwrap();
    assert!(super::apply_toml_config_file(&mut cfg, &path).is_err());

    std::fs::remove_dir_all(&dir).ok();
}

/// 未知键（如拼错的 `vector_dims`）必须 fast-fail，不静默落默认——
/// deny_unknown_fields 的回归护栏：键名笔误与枚举笔误同属"配置不可信"。
/// EN: Unknown keys (typos like `vector_dims`) must fast-fail instead of
/// silently falling back to defaults — regression guard for deny_unknown_fields.
#[test]
fn toml_config_unknown_key_is_fatal() {
    let dir = std::env::temp_dir().join(format!(
        "houyidb-toml-cfg-unknown-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("unknown.toml");
    std::fs::write(&path, "vector_dims = 768\n").unwrap();
    let mut cfg = super::ServerConfig::default();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(
        err.contains("vector_dims"),
        "error should name the bad key: {err}"
    );
    // 被拒的配置不得半应用：vector_dim 保持默认。
    // EN: Rejected config must not be half-applied: vector_dim keeps its default.
    assert_eq!(cfg.vector_dim, super::ServerConfig::default().vector_dim);

    std::fs::remove_dir_all(&dir).ok();
}

/// 空白 auth_token 必须 fail-closed（OCR 359d85d[6]）：运维写了
/// `auth_token = "  "` 时以为认证已开——静默降级 None 等于裸奔，
/// 启动期报错才是正确行为。trim 后存储（比较侧不 trim）。
/// EN: Blank auth_token must fail closed (OCR 359d85d[6]): `auth_token = "  "` makes the
/// EN: operator believe auth is on — silently degrading to None is running naked; a startup
/// EN: error is correct. Store the trimmed value (comparison side does not trim).
#[test]
fn toml_blank_auth_token_is_fatal_fail_closed() {
    let mut cfg = super::ServerConfig::default();
    let file: super::ServerConfigFile = toml::from_str("auth_token = \"   \"\n").unwrap();
    let err = file.apply_to(&mut cfg).unwrap_err();
    assert!(
        err.contains("auth_token"),
        "error should name the key: {err}"
    );
    assert!(
        cfg.auth_token.is_none(),
        "rejected blank token must not be applied"
    );

    // 带外围空白的合法 token：存 trim 后的值，否则永远匹配不上客户端。
    // EN: Valid token with surrounding spaces: store the trimmed value, or clients never match.
    let file: super::ServerConfigFile = toml::from_str("auth_token = \"  s3cret  \"\n").unwrap();
    file.apply_to(&mut cfg).unwrap();
    assert_eq!(cfg.auth_token.as_deref(), Some("s3cret"));
}

/// 文件缺失/语法错误均 fast-fail；空文件 = 全默认。
/// EN: Missing file or syntax errors fast-fail; an empty file means all defaults.
#[test]
fn toml_config_missing_or_malformed_is_fatal_empty_is_noop() {
    let mut cfg = super::ServerConfig::default();
    let missing = std::path::Path::new("/nonexistent/houyidb-config.toml");
    let err = super::apply_toml_config_file(&mut cfg, missing).unwrap_err();
    assert!(err.contains("read config file"), "got: {err}");

    let dir = std::env::temp_dir().join(format!(
        "houyidb-toml-cfg-malformed-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("malformed.toml");
    std::fs::write(&path, "vector_dim = [not a number\n").unwrap();
    let err = super::apply_toml_config_file(&mut cfg, &path).unwrap_err();
    assert!(err.contains("parse config file"), "got: {err}");

    std::fs::write(&path, "# nothing but a comment\n").unwrap();
    super::apply_toml_config_file(&mut cfg, &path).expect("empty toml is a no-op");
    assert_eq!(cfg.vector_dim, super::ServerConfig::default().vector_dim);

    std::fs::remove_dir_all(&dir).ok();
}

// ── P4.3.2 慢查询日志 ───────────────────────────────────────────
// EN: P4.3.2 slow query log.

/// 慢查询日志（P4.3.2）：阈值 0 = 禁用不打；阈值启用后超阈值检索打 warn。
/// 正向触发用 flat 后端暴力扫描 + 循环采样：单次扫描耗时不可精确控制，
/// 加重到 dim=96 × 60k 行（单次稳定数毫秒），40 次采样中至少一次 >= 1ms
/// 在重载 CI 上也是稳妥的（调度/缺页抖动）。
/// EN: threshold 0 stays quiet; once enabled, an over-threshold search warns.
/// The firing path uses a flat brute-force scan plus sampling, weighted to
/// dim-96 × 60k rows (each scan stably takes several ms), so at least one of
/// 40 scans crossing 1ms holds even on a loaded CI box (jitter included).
#[tokio::test]
async fn slow_query_log_quiet_when_disabled_and_fires_over_threshold() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    // 捕获 warn 输出的内存 writer（set_default 线程局部订阅器，不动全局）。
    // EN: In-memory writer capturing warn output (thread-local via set_default, not global).
    #[derive(Clone)]
    struct CaptureWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("capture lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> MakeWriter<'a> for CaptureWriter {
        type Writer = CaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    let buf = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(CaptureWriter(Arc::clone(&buf)))
        .with_ansi(false)
        .finish();
    // #[tokio::test] 默认 current_thread 运行时 + 直调 handler 内联轮询，
    // 线程局部订阅器覆盖整个测试体（无跨线程丢日志风险）。
    // EN: #[tokio::test] uses the default current_thread runtime + inline handler polling,
    // EN: so the thread-local subscriber covers the whole test body (no cross-thread log loss).
    let _guard = tracing::subscriber::set_default(subscriber);

    let captured = || String::from_utf8_lossy(&buf.lock().expect("capture lock")).to_string();

    let (state, _cleanup) = collection_test_state_with_index(96, "flat");
    // 60k × dim96 暴力扫描：单次稳定数毫秒，配合循环采样保证至少一次越线。
    // EN: 60k x dim96 brute-force scan: a few ms per call; loop sampling guarantees one threshold
    // EN: hit.
    let vectors: Vec<houyi_server::types::UpsertVector> = (0..60_000u64)
        .map(|i| houyi_server::types::UpsertVector {
            id: i + 1,
            values: (0..96).map(|d| ((i + d as u64) % 7) as f32).collect(),
            label: None,
            payload: None,
        })
        .collect();
    collection_upsert(&state, "c1", vectors).await;

    let query: Vec<f32> = vec![1.0; 96];
    // 阈值 0（默认）= 禁用：检索正常返回且日志安静。
    // EN: Threshold 0 (default) = disabled: search returns normally and logs stay quiet.
    let resp = collection_search(&state, "c1", query.clone(), 3, None).await;
    assert!(!resp.ids.is_empty());
    assert!(
        !captured().contains("slow http"),
        "threshold 0 must stay quiet: {}",
        captured()
    );

    // 阈值 1ms：循环采样，至少一次扫描越线打 warn。
    // EN: Threshold 1ms: loop sampling until at least one scan exceeds it and logs warn.
    state
        .pgwire_metrics
        .slow_query_ms
        .store(1, AtomicOrdering::Relaxed);
    let mut fired = false;
    for _ in 0..40 {
        let _ = collection_search(&state, "c1", query.clone(), 3, None).await;
        if captured().contains("slow http collection search") {
            fired = true;
            break;
        }
    }
    assert!(
        fired,
        "threshold 1ms must fire at least once over 40 brute-force scans"
    );
    // 日志不得携带向量本体（只带 collection/k/elapsed 等元数据）。
    // EN: Logs must not contain the vector itself (only metadata: collection/k/elapsed).
    let out = captured();
    assert!(
        !out.contains("[1.0, 1.0"),
        "query vector must not leak into the log: {out}"
    );
}

// ---------- OCR 批次十五：快照阈值基线与关停 ENV 解析 ----------
// EN: OCR batch 15: snapshot threshold baseline and shutdown ENV parsing.

#[test]
fn wal_threshold_due_requires_known_baseline() {
    // 基线未知（读失败）只建基线，不得把整个存量 WAL 计为增量误触发。
    // EN: Unknown baseline (read failure): only record it; never count the whole WAL as delta.
    assert!(!super::wal_threshold_due(None, Some(1 << 30), 256));
    // 当前读失败同样不触发。
    // EN: A failed current read does not trigger either.
    assert!(!super::wal_threshold_due(Some(100), None, 256));
    // 双未知。
    // EN: Both unknown.
    assert!(!super::wal_threshold_due(None, None, 256));
}

#[test]
fn wal_threshold_due_fires_on_delta_and_saturates_on_shrink() {
    // 增量达到阈值触发；不足不触发。
    // EN: Triggers when the delta reaches the threshold; not below it.
    assert!(super::wal_threshold_due(Some(100), Some(100 + 256), 256));
    assert!(!super::wal_threshold_due(Some(100), Some(100 + 255), 256));
    // WAL 变小（外部事故）saturating 归零，不得回绕成巨大增量误触发。
    // EN: A shrinking WAL (external incident) saturates to zero, never wraps into a huge delta.
    assert!(!super::wal_threshold_due(Some(1000), Some(10), 256));
    // 阈值 0 = 禁用通道。
    // EN: Threshold 0 = disable the channel.
    assert!(!super::wal_threshold_due(Some(0), Some(u64::MAX), 0));
}

#[test]
fn shutdown_snapshot_enabled_parses_disable_values_strictly() {
    // 缺省 / 空串 → 默认 ON。
    // EN: Missing / empty string -> default ON.
    assert!(super::shutdown_snapshot_enabled(None));
    assert!(super::shutdown_snapshot_enabled(Some("")));
    // 显式关闭族（大小写与空白不敏感）。
    // EN: Explicit off family (case- and whitespace-insensitive).
    for v in ["0", "false", "off", "no", "FALSE", "Off", " 0 ", " No"] {
        assert!(
            !super::shutdown_snapshot_enabled(Some(v)),
            "{v:?} must disable"
        );
    }
    // 显式开启族（常见布尔写法，不触发误报警告）与无法识别的值都落 ON
    // （关停路径不 fast-fail）。
    // EN: Explicit on family (common boolean spellings, no false warning) and unrecognized
    // EN: values both fall to ON (the shutdown path does not fast-fail).
    for v in ["1", "true", "yes", "on", "TRUE", " On "] {
        assert!(
            super::shutdown_snapshot_enabled(Some(v)),
            "{v:?} must enable"
        );
    }
    assert!(super::shutdown_snapshot_enabled(Some("ture"))); // 笔误也得保住快照
}

// ══════════════════════════════════════════════════════════════════════════
// P4.5.3 过滤 TLP（SQLancer 三分区映射）／Filter TLP
//
// 目标：给「HTTP 结构化过滤 → 掩码 → 向量检索」这条链加一道**无外部真值**的
// 正确性网。三条腿：
//
// 1. **差分对照**——测试侧独立实现一套过滤语义（`tlp_reference_matches`），
//    与被测链（`filter_predicate` JSON→谓词 + PAX 页过滤 + 行号→id + 掩码检索）
//    逐 id 对照。参考实现刻意不复用内核 `compare`，避免「同一 bug 两边同错」。
// 2. **三分区（TLP）**——按恒存在的 `tag ∈ {"a","b"}` 把数据集划成互斥且穷尽
//    的两半，于是对**任意**谓词合取 P：`Q(P) = Q(P∧tag=a) ⊎ Q(P∧tag=b)`，
//    且两侧不交。这比「P / ¬P」更契合本 API——HTTP 过滤面只支持**合取**
//    （无 `or`/`not`），而划分维恰好能表达补集，且断言的是「集合被完整切分」。
// 3. **墓碑不可见**——删除的 id 在任何过滤/向量组合下都不得出现。
//
// 为什么用 flat 索引（WHY）：本组断言关心的是**集合**，而 HNSW 近似会漏邻，
// 使「Q(P) 恰好等于参考集合」在近似下不可判。flat 是精确扫描，掩码分支
// （≤50K 走 rayon 暴力）在两条索引路径上都是精确的；ANN 的近似行为由
// `houyi-vector/tests/retrieval_oracle.rs` 单独覆盖。
//
// EN: Filter TLP — differential against an independent reference implementation,
// a two-way partition on a always-present `tag` column (the HTTP filter DSL only
// supports conjunctions, so a partition is the right way to express the
// complement), and tombstone invisibility. The `flat` index is used so the
// assertions are decidable (HNSW approximation is covered by the vector oracle).
// ══════════════════════════════════════════════════════════════════════════

// proptest 预导入（注意：Rust 的 `use` 是**模块级**的，位置不影响作用域——
// 它等效于放在文件头；当前能编译只是因为全模块没有重名项。若将来要真正
// 隔离命名空间，应把这节包进嵌套 `mod`）。
// EN: `use` is module-scoped regardless of position — this is equivalent to a
// file-top import today; a nested `mod` is needed for real isolation.
use proptest::prelude::*;

/// 一行测试数据：`id` + 两个过滤列 + 向量。
/// 为什么 `p` 可缺列：payload 少写一个键时，PAX 页会以 `JsonText("null")` 补齐
/// （见 `HouyiPage::insert_row`）——这正是「范围比较命中缺列行」那类静默假阳性
/// 的入口，必须在随机数据里真实出现。
/// EN: `p` may be absent so the page-padding path (`JsonText("null")`) is
/// genuinely exercised — that padding is the entry point for silent
/// false-positive range matches.
#[derive(Debug, Clone)]
struct TlpRow {
    id: u64,
    p: Option<i64>,
    tag: &'static str,
    vector: Vec<f32>,
}

/// 测试侧的过滤条件模型（与 HTTP DSL 同形：平坦合取，无嵌套）。
/// Test-side filter model — flat conjunction, same shape as the HTTP DSL.
#[derive(Debug, Clone)]
struct TlpFilter {
    column: &'static str,
    op: &'static str,
    value: serde_json::Value,
}

const TLP_DIM: usize = 8;

/// splitmix64：确定性向量/数据集生成，不引外部 RNG（避免版本 API 漂移）。
/// EN: splitmix64: deterministic vector/dataset generation without an external RNG
/// EN: (avoids version API drift).
struct TlpRng(u64);

impl TlpRng {
    fn new(seed: u64) -> Self {
        TlpRng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// [0, 1) 均匀分布。
    /// EN: uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn vector(&mut self) -> Vec<f32> {
        (0..TLP_DIM).map(|_| self.unit()).collect()
    }
}

/// 由（随机 payload 形状, 种子）物化成完整行集：id 唯一且与行号一致。
/// Materialize rows from the random payload shapes; ids are unique.
fn tlp_rows(shapes: &[(Option<i64>, &'static str)], seed: u64) -> Vec<TlpRow> {
    let mut rng = TlpRng::new(seed);
    shapes
        .iter()
        .enumerate()
        .map(|(idx, (p, tag))| TlpRow {
            id: idx as u64 + 1,
            p: *p,
            tag,
            vector: rng.vector(),
        })
        .collect()
}

/// **独立参考实现**：给定行与条件，判定是否命中。
///
/// 语义刻意与内核分开写（不复用 `houyi_core::compare`）：
/// - `p` 缺列 ⇒ 该列在页里是 `JsonText("null")`，**任何**比较/相等/`in` 都不命中；
/// - `p` 有值（整数）⇒ 数值比较，`eq`/`in` 按整数相等；
/// - `tag` 恒存在 ⇒ 文本比较。
///
/// EN: Independent reference. A missing `p` is `JsonText("null")` on the page and
/// therefore matches nothing; an integer `p` compares numerically; `tag` is
/// always present and compares as text.
fn tlp_reference_matches(row: &TlpRow, filter: &TlpFilter) -> bool {
    match filter.column {
        "p" => {
            let Some(p) = row.p else {
                // 缺列 ⇒ 与任何标量都不可比、不相等。
                // EN: Missing column => incomparable/unequal to any scalar.
                return false;
            };
            match filter.op {
                "eq" => filter.value.as_i64() == Some(p),
                "gt" => filter.value.as_i64().is_some_and(|v| p > v),
                "lt" => filter.value.as_i64().is_some_and(|v| p < v),
                "gte" => filter.value.as_i64().is_some_and(|v| p >= v),
                "lte" => filter.value.as_i64().is_some_and(|v| p <= v),
                "in" => filter
                    .value
                    .as_array()
                    .expect("in literal is an array")
                    .iter()
                    .any(|v| v.as_i64() == Some(p)),
                other => panic!("unexpected op in reference: {other}"),
            }
        }
        "tag" => match filter.op {
            // 标量操作符取字符串字面值；`in` 取数组（不能对数组调 `as_str`）。
            // EN: Scalar ops read a string literal; `in` reads the array.
            "eq" => filter.value.as_str() == Some(row.tag),
            "gt" => filter.value.as_str().is_some_and(|t| row.tag > t),
            "lt" => filter.value.as_str().is_some_and(|t| row.tag < t),
            "gte" => filter.value.as_str().is_some_and(|t| row.tag >= t),
            "lte" => filter.value.as_str().is_some_and(|t| row.tag <= t),
            "in" => filter
                .value
                .as_array()
                .expect("in literal is an array")
                .iter()
                .any(|v| v.as_str() == Some(row.tag)),
            other => panic!("unexpected op in reference: {other}"),
        },
        other => panic!("unexpected column in reference: {other}"),
    }
}

/// 参考集合：合取（所有条件同时命中）后的存活 id，升序。
/// Reference id set for a conjunction over live (non-tombstoned) rows.
fn tlp_reference_ids(rows: &[TlpRow], dead: &[u64], filters: &[TlpFilter]) -> Vec<u64> {
    let mut ids: Vec<u64> = rows
        .iter()
        .filter(|r| !dead.contains(&r.id))
        .filter(|r| filters.iter().all(|f| tlp_reference_matches(r, f)))
        .map(|r| r.id)
        .collect();
    ids.sort_unstable();
    ids
}

/// 随机 payload 形状（`p` 可缺列，`tag` 恒存在）。
/// EN: Random payload shape (`p` may be missing; `tag` is always present).
fn tlp_arb_shapes() -> impl Strategy<Value = Vec<(Option<i64>, &'static str)>> {
    prop::collection::vec(
        (
            prop::option::of(0i64..20),
            prop_oneof![Just("a"), Just("b")],
        ),
        1..=12,
    )
}

/// 随机合取条件（0~3 条叶子，可跨列组合）。
/// EN: Random conjunction (0-3 leaves, combinable across columns).
fn tlp_arb_filters() -> impl Strategy<Value = Vec<TlpFilter>> {
    prop::collection::vec(tlp_arb_leaf(), 0..=3)
}

fn tlp_arb_leaf() -> impl Strategy<Value = TlpFilter> {
    prop_oneof![
        (0i64..20).prop_map(|v| TlpFilter {
            column: "p",
            op: "eq",
            value: serde_json::json!(v),
        }),
        (0i64..20).prop_map(|v| TlpFilter {
            column: "p",
            op: "gt",
            value: serde_json::json!(v),
        }),
        (0i64..20).prop_map(|v| TlpFilter {
            column: "p",
            op: "lt",
            value: serde_json::json!(v),
        }),
        (0i64..20).prop_map(|v| TlpFilter {
            column: "p",
            op: "gte",
            value: serde_json::json!(v),
        }),
        (0i64..20).prop_map(|v| TlpFilter {
            column: "p",
            op: "lte",
            value: serde_json::json!(v),
        }),
        prop::collection::vec(0i64..20, 1..=3).prop_map(|vs| TlpFilter {
            column: "p",
            op: "in",
            value: serde_json::json!(vs),
        }),
        prop_oneof![Just("a"), Just("b")].prop_map(|t| TlpFilter {
            column: "tag",
            op: "eq",
            value: serde_json::json!(t),
        }),
        prop::collection::vec(prop_oneof![Just("a"), Just("b")], 1..=2).prop_map(|ts| {
            TlpFilter {
                column: "tag",
                op: "in",
                value: serde_json::json!(ts),
            }
        }),
    ]
}

/// 把行集写入一个**全新**的 flat collection（每例独立状态，杜绝跨例污染）。
/// 返回 (state, cleanup, collection 名)。
/// EN: Write the row set into a **brand-new** flat collection (per-case state, no cross-case
/// EN: leakage).
/// EN: Returns (state, cleanup, collection name).
async fn tlp_setup(rows: &[TlpRow], suffix: u64) -> (BenchmarkHttpState, TempFileCleanup, String) {
    let (state, cleanup) = collection_test_state_with_index(TLP_DIM, "flat");
    let name = format!("tlp{suffix}");
    let vectors: Vec<houyi_server::types::UpsertVector> = rows
        .iter()
        .map(|r| {
            let mut payload = serde_json::Map::new();
            if let Some(p) = r.p {
                payload.insert("p".to_string(), serde_json::json!(p));
            }
            payload.insert("tag".to_string(), serde_json::json!(r.tag));
            houyi_server::types::UpsertVector {
                id: r.id,
                values: r.vector.clone(),
                label: None,
                payload: Some(payload),
            }
        })
        .collect();
    collection_upsert(&state, &name, vectors).await;
    (state, cleanup, name)
}

/// 跑一次过滤检索并返回**排序后的 id 集合**。
/// `k` 取「行数 + 余量」以枚举整个命中集（掩码分支按距离排序后截断到 k）。
/// Run one filtered search and return the sorted id SET. `k` exceeds the row
/// count so the whole masked set is enumerated (the mask branch sorts by
/// distance then truncates to `k`).
async fn tlp_query(
    state: &BenchmarkHttpState,
    name: &str,
    query: &[f32],
    filters: &[TlpFilter],
) -> Vec<u64> {
    let filters: Vec<(&str, &str, serde_json::Value)> = filters
        .iter()
        .map(|f| (f.column, f.op, f.value.clone()))
        .collect();
    let mut ids = collection_hybrid(state, name, query.to_vec(), 64, filters).await;
    ids.sort_unstable();
    ids
}

fn tlp_sorted(mut ids: Vec<u64>) -> Vec<u64> {
    ids.sort_unstable();
    ids
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

    /// 差分：被测链的命中集必须与独立参考实现逐 id 一致。
    /// Differential: the filter chain must equal the independent reference.
    #[test]
    fn tlp_filter_chain_matches_independent_reference(
        shapes in tlp_arb_shapes(),
        filters in tlp_arb_filters(),
        seed in any::<u64>(),
    ) {
        let rows = tlp_rows(&shapes, seed);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (state, _cleanup, name) = tlp_setup(&rows, seed).await;
            let query = TlpRng::new(seed ^ 0xF00D).vector();
            let got = tlp_query(&state, &name, &query, &filters).await;
            let want = tlp_reference_ids(&rows, &[], &filters);
            prop_assert_eq!(&got, &want, "rows={:?} filters={:?}", rows, filters);
            Ok(())
        })?;
    }

    /// TLP 三分区：按恒存在的 `tag` 划分后，两侧不交且并集恰等于整体命中集。
    /// Partition: splitting on the always-present `tag` must cut the result set
    /// into two disjoint halves whose union is the whole.
    #[test]
    fn tlp_partition_on_tag_is_disjoint_and_exhaustive(
        shapes in tlp_arb_shapes(),
        filters in tlp_arb_filters(),
        seed in any::<u64>(),
    ) {
        let rows = tlp_rows(&shapes, seed);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (state, _cleanup, name) = tlp_setup(&rows, seed).await;
            let query = TlpRng::new(seed ^ 0xBEEF).vector();

            let all = tlp_query(&state, &name, &query, &filters).await;
            let mut a = filters.to_vec();
            a.push(TlpFilter { column: "tag", op: "eq", value: serde_json::json!("a") });
            let mut b = filters.to_vec();
            b.push(TlpFilter { column: "tag", op: "eq", value: serde_json::json!("b") });
            let set_a = tlp_query(&state, &name, &query, &a).await;
            let set_b = tlp_query(&state, &name, &query, &b).await;

            // 不交。
            // EN: Disjoint.
            let inter: Vec<u64> = set_a.iter().filter(|id| set_b.contains(id)).copied().collect();
            prop_assert!(inter.is_empty(), "partitions overlap: {:?}", inter);
            // 并集 = 整体（`tag` 恒存在且 ∈ {a,b} ⇒ 穷尽）。
            // EN: Union = whole (tag is always present and in {a,b}, hence exhaustive).
            prop_assert_eq!(tlp_sorted([set_a.clone(), set_b.clone()].concat()), all.clone());
            // 两侧各自也是参考解。
            // EN: Each side also matches the reference solution.
            prop_assert_eq!(&set_a, &tlp_reference_ids(&rows, &[], &a));
            prop_assert_eq!(&set_b, &tlp_reference_ids(&rows, &[], &b));
            Ok(())
        })?;
    }

    /// 合取单调性：加一个条件只能缩小命中集（过滤不得放宽）。
    /// Conjunction monotonicity: adding a conjunct may only shrink the result.
    #[test]
    fn tlp_adding_conjunct_only_shrinks(
        shapes in tlp_arb_shapes(),
        filters in tlp_arb_filters(),
        extra in tlp_arb_leaf(),
        seed in any::<u64>(),
    ) {
        let rows = tlp_rows(&shapes, seed);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (state, _cleanup, name) = tlp_setup(&rows, seed).await;
            let query = TlpRng::new(seed ^ 0x5EED).vector();
            let base = tlp_query(&state, &name, &query, &filters).await;
            let mut wider = filters.clone();
            wider.push(extra.clone());
            let narrowed = tlp_query(&state, &name, &query, &wider).await;
            for id in &narrowed {
                prop_assert!(
                    base.contains(id),
                    "conjunct {} ({:?}) introduced id {} not in the base set {:?}",
                    id, extra, id, base
                );
            }
            Ok(())
        })?;
    }

    /// 墓碑：删除的 id 在**任何**过滤组合下都不可见（且参考实现同步剔除）。
    /// Tombstones: a deleted id is invisible under every filter combination.
    #[test]
    fn tlp_tombstoned_ids_are_invisible_everywhere(
        shapes in tlp_arb_shapes(),
        filters in tlp_arb_filters(),
        victim_pick in any::<prop::sample::Index>(),
        seed in any::<u64>(),
    ) {
        let rows = tlp_rows(&shapes, seed);
        let victim = rows[victim_pick.index(rows.len())].id;
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (state, _cleanup, name) = tlp_setup(&rows, seed).await;
            let query = TlpRng::new(seed ^ 0xDEAD).vector();

            let resp = collection_delete_call(&state, &name, vec![victim]).await
                .expect("delete should succeed");
            prop_assert!(resp.deleted >= 1, "delete reported {:?}", resp);

            // 带条件与不带条件都不许出现。
            // EN: Must not appear with or without filters.
            let filtered = tlp_query(&state, &name, &query, &filters).await;
            let unfiltered = tlp_query(&state, &name, &query, &[]).await;
            prop_assert!(!filtered.contains(&victim), "deleted id {} still filtered in: {:?}", victim, filtered);
            prop_assert!(!unfiltered.contains(&victim), "deleted id {} still visible: {:?}", victim, unfiltered);
            // 参考实现也剔除后必须逐位一致。
            // EN: After the reference also excludes it, results must match exactly.
            let want = tlp_reference_ids(&rows, &[victim], &filters);
            prop_assert_eq!(&filtered, &want, "post-delete reference mismatch");
            Ok(())
        })?;
    }

    /// 掩码决定集合、向量只决定次序：同一过滤条件配两个不同查询向量，命中集必须相同。
    /// The mask determines the SET; the vector only orders it.
    #[test]
    fn tlp_result_set_is_query_independent(
        shapes in tlp_arb_shapes(),
        filters in tlp_arb_filters(),
        seed in any::<u64>(),
    ) {
        let rows = tlp_rows(&shapes, seed);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (state, _cleanup, name) = tlp_setup(&rows, seed).await;
            let q1 = TlpRng::new(seed ^ 0x1111).vector();
            let q2 = TlpRng::new(seed ^ 0x2222).vector();
            let s1 = tlp_query(&state, &name, &q1, &filters).await;
            let s2 = tlp_query(&state, &name, &q2, &filters).await;
            prop_assert_eq!(&s1, &s2, "result set changed with the query vector");
            Ok(())
        })?;
    }
}

/// 确定性回归：缺列与跨类型在**服务器过滤链**上不得被范围比较命中。
///
/// 这是 `houyi-core/tests/filter_oracle.rs` 那个假阳性缺陷的服务器侧镜像——
/// 缺陷在 `compare` 的 `_ => Equal` 兜底，但 HTTP 路径还有一层「JSON→Scalar」
/// 映射与 PAX 缺列补齐，必须单独钉死，否则内核修好而服务端仍可能放宽。
/// EN: Server-side mirror of the silent false-positive fixed in `houyi-core`:
/// the JSON→scalar mapping plus PAX missing-column padding is a second surface.
#[tokio::test]
async fn tlp_range_filters_never_match_missing_or_cross_type_payloads() {
    let rows = vec![
        TlpRow {
            id: 1,
            p: Some(10),
            tag: "a",
            vector: vec![0.1; TLP_DIM],
        },
        TlpRow {
            id: 2,
            p: None,
            tag: "a",
            vector: vec![0.2; TLP_DIM],
        }, // 缺 p 列
        TlpRow {
            id: 3,
            p: Some(3),
            tag: "b",
            vector: vec![0.3; TLP_DIM],
        },
    ];
    let (state, _cleanup, name) = tlp_setup(&rows, 4242).await;
    let query = vec![0.15_f32; TLP_DIM];

    let gte = tlp_query(
        &state,
        &name,
        &query,
        &[TlpFilter {
            column: "p",
            op: "gte",
            value: serde_json::json!(5),
        }],
    )
    .await;
    assert_eq!(gte, vec![1], "gte must only match the numeric row");

    // 缺列行（id 2）不得因 `_ => Equal` 式兜底混进范围结果：`p <= 2` 对
    // p=10 与 p=3 都为假、对缺列行为「不可比」，故必须是空集。
    // EN: The missing-column row (id 2) must not slip into range results via a `_ => Equal`
    // EN: fallback: `p <= 2` is false for p=10 and p=3 and incomparable for the missing
    // EN: column, so the result must be empty.
    let lte_low = tlp_query(
        &state,
        &name,
        &query,
        &[TlpFilter {
            column: "p",
            op: "lte",
            value: serde_json::json!(2),
        }],
    )
    .await;
    assert!(
        lte_low.is_empty(),
        "lte below every value must be empty, got {lte_low:?}"
    );

    // 正常范围：两条有值的行都命中，缺列行仍不得出现。
    // EN: Normal range: both valued rows match; the missing-column row must still be absent.
    let lte = tlp_query(
        &state,
        &name,
        &query,
        &[TlpFilter {
            column: "p",
            op: "lte",
            value: serde_json::json!(100),
        }],
    )
    .await;
    assert_eq!(
        lte,
        vec![1, 3],
        "lte must match numeric rows only (no missing-column row)"
    );

    // 交叉类型：`tag` 是字符串列，对它做数值范围比较必须空集（不得类型混淆命中）。
    // EN: Cross-type: `tag` is a string column; numeric range comparison must be empty
    // EN: (no type-confusion hits).
    let cross = tlp_query(
        &state,
        &name,
        &query,
        &[TlpFilter {
            column: "tag",
            op: "gt",
            value: serde_json::json!(0),
        }],
    )
    .await;
    assert!(
        cross.is_empty(),
        "numeric range over a text column must match nothing, got {cross:?}"
    );
}

/// 确定性回归：`in` 的空数组 ⇒ 空结果（不得被当成「无条件」放宽）。
/// Regression: an empty `in` literal must yield nothing, not "no condition".
#[tokio::test]
async fn tlp_empty_in_literal_yields_empty_result() {
    let rows = vec![
        TlpRow {
            id: 1,
            p: Some(1),
            tag: "a",
            vector: vec![0.1; TLP_DIM],
        },
        TlpRow {
            id: 2,
            p: Some(2),
            tag: "b",
            vector: vec![0.2; TLP_DIM],
        },
    ];
    let (state, _cleanup, name) = tlp_setup(&rows, 777).await;
    let query = vec![0.15_f32; TLP_DIM];

    let ids = tlp_query(
        &state,
        &name,
        &query,
        &[TlpFilter {
            column: "p",
            op: "in",
            value: serde_json::json!([]),
        }],
    )
    .await;
    assert!(ids.is_empty(), "empty `in` must match nothing, got {ids:?}");
}

/// P5.0.3：管理员 compact 端点提交 manifest、设置租约，并返回保留结果。
/// EN: P5.0.3 — the admin compact endpoint commits a manifest, sets the lease,
/// and returns the retention result.
#[tokio::test]
async fn collections_compact_endpoint_commits_manifest_and_sets_lease() {
    use houyi_server::http_handler::{benchmark_wal_status, handle_admin_collections_compact};
    use houyi_server::types::{CollectionsCompactRequest, UpsertVector};

    let (state, _cleanup) = collection_test_state(4);
    collection_upsert(
        &state,
        "docs",
        vec![UpsertVector {
            id: 1,
            values: vec![1.0, 0.0, 0.0, 0.0],
            label: Some("a".to_string()),
            payload: None,
        }],
    )
    .await;

    // 先确认 status 的 collections 视图存在且未提交 manifest。
    // EN: First confirm the status collections view exists and has no manifest.
    let before = benchmark_wal_status(State(state.clone())).await.unwrap().0;
    assert!(!before.collections.manifest_present);
    assert_eq!(before.collections.archived_segments, 0);

    let Json(resp) = handle_admin_collections_compact(
        State(state.clone()),
        Json(CollectionsCompactRequest {
            rotate: false,
            hold_retention_secs: Some(120),
        }),
    )
    .await
    .unwrap();
    assert!(resp.manifest_committed);
    assert!(resp.manifest_seq.is_some());
    assert!(resp.retention.lease_until_unix > 0);
    // 测试态 max_segments=0：保留明确报告为 disabled。
    // EN: Test state uses max_segments=0, so retention reports disabled.
    assert_eq!(
        resp.retention.blocked_reason.as_deref(),
        Some("retention_disabled")
    );

    let after = benchmark_wal_status(State(state)).await.unwrap().0;
    assert!(after.collections.manifest_present);
    assert_eq!(after.collections.manifest_seq, resp.manifest_seq);
    assert_eq!(
        after.collections.safe_checkpoint_lsn,
        resp.safe_checkpoint_lsn
    );
    assert!(after.collections.retention_lease_until_unix > 0);
}

/// P5.0.3：follower 的管理员 compact 必须 503（本地周期任务另行执行）。
/// EN: P5.0.3 — the admin compact endpoint returns 503 on followers (the local
/// periodic task runs separately).
#[tokio::test]
async fn collections_compact_endpoint_rejects_followers() {
    use houyi_server::http_handler::handle_admin_collections_compact;
    use houyi_server::types::{CollectionsCompactRequest, ReplicationRole};

    let (mut state, _cleanup) = collection_test_state(4);
    state.role = houyi_server::types::RoleState::new(ReplicationRole::Follower);
    let err =
        handle_admin_collections_compact(State(state), Json(CollectionsCompactRequest::default()))
            .await
            .unwrap_err();
    assert_eq!(err.0, StatusCode::SERVICE_UNAVAILABLE);
    assert!(err.1.contains("read_only"));
}
