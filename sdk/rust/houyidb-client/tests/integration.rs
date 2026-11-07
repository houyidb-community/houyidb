//! houyidb-client Rust SDK 与真实服务的集成测试。
//! EN: Integration tests for the houyidb-client Rust SDK against a live server.
//!
//! hermetic 纪律（P4.6.4）：**只有显式设置 `HOUYIDB_URL` 才连接真实服务**；
//! 未设置时所有用例确定性跳过——`cargo test --all` 不再依赖「17000 上恰好
//! 有（且状态不陈旧）的服务」这一环境巧合。要跑真集成：
//! `./scripts/sdk_integration.sh`（脚本自建临时服务 + 临时 WAL，跑完清理）。
//! EN: hermetic discipline (P4.6.4): tests connect only when `HOUYIDB_URL` is
//! explicitly set and skip deterministically otherwise; `cargo test --all` no
//! longer relies on the environmental coincidence of a (non-stale) server on
//! port 17000. For real integration: `./scripts/sdk_integration.sh` (spins up a
//! throwaway server plus a temporary WAL and cleans up afterwards).
//!
//! 并行安全：Rust 测试默认多线程并行，因此每个用例独占 collection 名
//! （共享 collection 会因写入时序互相干扰）；WAL 持久化 + 同 id upsert 幂等，
//! 重跑安全。检索类断言前一律 wait_flushed（向量检索看不到热缓冲中的数据）。
//! EN: parallel safety: Rust tests run multi-threaded by default, so each test
//! owns its collection name (a shared collection would interfere through write
//! timing); WAL persistence plus idempotent upsert by id make reruns safe.
//! Search assertions always call wait_flushed first (vector search cannot see
//! data in the hot buffer).

use houyidb_client::{
    stable_id, Client, Collection, CreateCollectionOptions, Filter, FilterOp, HnswConfig,
    HouyiError, SearchOptions, Vector,
};
use std::collections::HashMap;
use std::time::Duration;

const TEST_DIMS: usize = 16;

/// 显式给定的服务地址；未设置/空白 ⇒ None（用例跳过，测试面保持 hermetic）。
/// EN: explicit URL or None; unset means skip so `cargo test --all` is hermetic.
fn base_url() -> Option<String> {
    std::env::var("HOUYIDB_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// 确定性伪 embedding（与 Python/TS/Go 联调同款思路，自洽即可）。
/// EN: Deterministic pseudo-embedding (same idea as the Python/TS/Go SDKs; it
/// merely needs to be self-consistent).
fn fake_embed(text: &str) -> Vec<f32> {
    (0..TEST_DIMS)
        .map(|counter| {
            let digest = blake2::Blake2b::<blake2::digest::generic_array::typenum::U8>::default();
            let mut h = digest;
            blake2::Digest::update(&mut h, format!("{text}:{counter}").as_bytes());
            let out = blake2::Digest::finalize(h);
            let v = u64::from_be_bytes(out.into());
            (v as f64 / 18446744073709551616.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn payload(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// 未配置 URL 或服务不可达时返回 None（用例跳过），可达返回 (Client, Collection)。
/// EN: None when unconfigured or unreachable (test skips).
fn connect(name: &str) -> Option<(Client, Collection)> {
    let Some(url) = base_url() else {
        eprintln!("HOUYIDB_URL 未设置：跳过集成用例（真实运行用 ./scripts/sdk_integration.sh）");
        return None;
    };
    let client = Client::with_timeout(&url, Duration::from_secs(3));
    if !client.health() {
        // 显式配置了 URL 却不可达 = 环境错误，必须响亮；不能静默跳过让
        // 「以为在跑集成、实际全跳过」蒙混过关。
        // EN: an explicitly configured but unreachable URL is loud, not silent.
        panic!("HOUYIDB_URL={url} 已设置但服务不可达（health 失败）");
    }
    let col = client.collection(name);
    Some((client, col))
}

/// 等待异步构图队列排空（双速写入：flush 后才对检索可见）。
/// EN: Waits for the async graph-build queue to drain (dual-speed writes are
/// visible to search only after flush).
fn wait_flushed(col: &Collection) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if let Ok(status) = col.status() {
            if status
                .get("pending_count")
                .and_then(|v| v.as_u64())
                .map(|n| n == 0)
                .unwrap_or(false)
            {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("collection {} flush timeout", col.name());
}

#[test]
fn upsert_search_round_trip() {
    let Some((_client, col)) = connect("rust_it") else {
        return;
    };

    let texts = [
        "apple is a fruit",
        "banana is yellow",
        "laptop is electronics",
    ];
    let categories = ["fruit", "fruit", "electronics"];
    let prices = [5, 3, 999];
    let vectors: Vec<Vector> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| Vector {
            id: stable_id(&format!("rust-doc-{i}")),
            values: fake_embed(text),
            label: None,
            payload: Some(payload(&[
                ("text", (*text).into()),
                ("category", categories[i].into()),
                ("price", prices[i].into()),
            ])),
        })
        .collect();

    let inserted = col.upsert(&vectors).expect("upsert");
    assert_eq!(inserted, 3);
    wait_flushed(&col);

    let opts = SearchOptions {
        include_payload: true,
        ..Default::default()
    };
    let result = col
        .search(&fake_embed("apple is a fruit"), 2, &opts)
        .expect("search");
    assert_eq!(result.hits.len(), 2);
    let top = result.hits[0].payload.as_ref().expect("payload");
    assert_eq!(
        top.get("text").and_then(|v| v.as_str()),
        Some("apple is a fruit")
    );
    assert!(
        result.hits[0].score >= result.hits[1].score,
        "scores must be descending"
    );
}

#[test]
fn hybrid_search_filters() {
    let Some((_client, col)) = connect("rust_it_hybrid") else {
        return;
    };

    let texts = [
        "apple is a fruit",
        "banana is yellow",
        "laptop is electronics",
    ];
    let categories = ["fruit", "fruit", "electronics"];
    let prices = [5, 3, 999];
    let vectors: Vec<Vector> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| Vector {
            id: stable_id(&format!("rust-hybrid-{i}")),
            values: fake_embed(text),
            label: None,
            payload: Some(payload(&[
                ("text", (*text).into()),
                ("category", categories[i].into()),
                ("price", prices[i].into()),
            ])),
        })
        .collect();
    col.upsert(&vectors).expect("upsert");
    wait_flushed(&col);

    let opts = SearchOptions {
        include_payload: true,
        filters: vec![
            Filter::new("category", FilterOp::Eq, "fruit"),
            Filter::new("price", FilterOp::Lt, 4),
        ],
        ..Default::default()
    };
    let result = col
        .hybrid_search(&fake_embed("apple is a fruit"), 3, &opts)
        .expect("hybrid search");
    assert_eq!(
        result.hits.len(),
        1,
        "expected only banana, got {:?}",
        result.hits
    );
    let top = result.hits[0].payload.as_ref().expect("payload");
    assert_eq!(
        top.get("text").and_then(|v| v.as_str()),
        Some("banana is yellow")
    );
}

/// 异键集 payload：PAX 新列回填的端到端守卫（曾在此场景整列错位）。
/// EN: Disjoint payload key sets: end-to-end guard for PAX new-column backfill
/// (a whole column once got misaligned in this scenario).
#[test]
fn mixed_payload_key_sets() {
    let Some((_client, col)) = connect("rust_it_mixed_keys") else {
        return;
    };

    col.upsert(&[
        Vector {
            id: 1,
            values: fake_embed("row-a"),
            label: None,
            payload: Some(payload(&[("a", 1.into())])),
        },
        Vector {
            id: 2,
            values: fake_embed("row-b"),
            label: None,
            payload: Some(payload(&[("b", 2.into())])),
        },
    ])
    .expect("upsert");
    // 过滤按标量页同步求值，但向量侧仍需构图完成才能命中。
    // EN: Filters are evaluated synchronously on scalar pages, but the vector
    // side still needs the graph build to finish before it can match.
    wait_flushed(&col);

    let opts = SearchOptions {
        include_payload: true,
        filters: vec![Filter::new("b", FilterOp::Eq, 2)],
        ..Default::default()
    };
    let result = col
        .hybrid_search(&fake_embed("row-b"), 5, &opts)
        .expect("hybrid search");
    assert_eq!(
        result.hits.len(),
        1,
        "filter b==2 must hit one row, got {:?}",
        result.hits
    );
    assert_eq!(result.hits[0].id, 2);
    let p = result.hits[0].payload.as_ref().expect("payload");
    assert!(
        !p.contains_key("a"),
        "payload must not leak column 'a': {p:?}"
    );
}

#[test]
fn large_u64_id_round_trip() {
    let Some((_client, col)) = connect("rust_it_bigid") else {
        return;
    };

    // stable_id 落点常态是 19 位十进制（>2^53）；Rust u64 原生无损。
    // EN: stable_id values are normally 19-digit decimals (>2^53); Rust u64 is
    // lossless natively.
    let id = stable_id("hello");
    col.upsert(&[Vector {
        id,
        values: fake_embed("uint64 probe"),
        label: None,
        payload: Some(payload(&[("tag", "big".into())])),
    }])
    .expect("upsert");
    wait_flushed(&col);

    let opts = SearchOptions {
        include_payload: true,
        filters: vec![Filter::new("tag", FilterOp::Eq, "big")],
        ..Default::default()
    };
    let result = col
        .hybrid_search(&fake_embed("uint64 probe"), 1, &opts)
        .expect("hybrid search");
    assert!(!result.hits.is_empty(), "no hit for big-id probe");
    assert_eq!(result.hits[0].id, id, "u64 id round trip broken");

    let deleted = col.delete(&[id], &[]).expect("cleanup delete");
    assert!(deleted <= 1);
}

#[test]
fn delete_by_ids_and_filters() {
    let Some((_client, col)) = connect("rust_it_delete") else {
        return;
    };

    let probe_id = stable_id("rust-delete-me");
    col.upsert(&[Vector {
        id: probe_id,
        values: fake_embed("delete probe"),
        label: None,
        payload: Some(payload(&[("tag", "del".into())])),
    }])
    .expect("upsert");

    // delete-by-filter 不依赖构图排空（标量页同步落页）。
    // EN: delete-by-filter does not depend on the graph-build drain (scalar
    // pages land synchronously).
    let deleted = col
        .delete(&[], &[Filter::new("tag", FilterOp::Eq, "del")])
        .expect("delete by filter");
    assert_eq!(deleted, 1);
    let again = col.delete(&[probe_id], &[]).expect("idempotent delete");
    assert_eq!(again, 0);
}

#[test]
fn server_error_surfaces_as_houyi_error() {
    let Some((_client, col)) = connect("rust_it_missing_collection") else {
        return;
    };

    match col.status() {
        Err(HouyiError::Http { status, .. }) => assert_eq!(status, 404),
        other => panic!("expected HTTP 404 error, got {other:?}"),
    }
}

#[test]
fn delete_requires_ids_or_filters() {
    let Some((_client, col)) = connect("rust_it") else {
        return;
    };
    assert!(col.delete(&[], &[]).is_err(), "empty delete must fail");
}

/// collection 名含保留字符时路径转义（与 Python quote(safe="") 同口径）。
/// EN: Path escaping when a collection name contains reserved characters
/// (same rule as Python quote(safe="")).
#[test]
fn collection_name_with_reserved_chars() {
    let Some((client, _)) = connect("rust_it") else {
        return;
    };
    let col = client.collection("rust 名/a?b#c");
    let vectors = vec![Vector {
        id: 42,
        values: fake_embed("quoted"),
        label: None,
        payload: Some(payload(&[("k", "v".into())])),
    }];
    col.upsert(&vectors).expect("upsert to quoted collection");
    let deleted = col.delete(&[42], &[]).expect("cleanup");
    assert_eq!(deleted, 1);
}

/// 删除整个 collection：删除后 404，同名重建得到空 collection。
/// EN: Dropping a whole collection: status becomes 404 afterwards, and a
/// same-name recreate starts empty.
#[test]
fn delete_collection_and_recreate() {
    let Some((client, col)) = connect("rust_it_admin_drop") else {
        return;
    };

    col.upsert(&[Vector {
        id: 7,
        values: fake_embed("drop me"),
        label: None,
        payload: Some(payload(&[("tag", "drop".into())])),
    }])
    .expect("upsert");

    client
        .delete_collection("rust_it_admin_drop")
        .expect("delete collection");

    match col.status() {
        Err(HouyiError::Http { status, .. }) => assert_eq!(status, 404),
        other => panic!("expected HTTP 404 after drop, got {other:?}"),
    }

    // 同名重建：首次写入隐式创建，旧向量不得复活。
    // EN: recreate by name: the first write implicitly creates it and the old
    // vectors must not reappear.
    col.upsert(&[Vector {
        id: 8,
        values: fake_embed("fresh"),
        label: None,
        payload: None,
    }])
    .expect("recreate");
    wait_flushed(&col);
    let status = col.status().expect("status after recreate");
    assert_eq!(
        status.get("indexed_count").and_then(|v| v.as_u64()),
        Some(1),
        "recreated collection must contain only the new vector"
    );
}

/// 删除不存在的 collection 返回上游 404。
/// EN: Dropping an unknown collection surfaces the upstream 404.
#[test]
fn delete_missing_collection_is_404() {
    let Some((client, _)) = connect("rust_it_admin_missing") else {
        return;
    };
    match client.delete_collection("rust_it_admin_never_created") {
        Err(HouyiError::Http { status, .. }) => assert_eq!(status, 404),
        other => panic!("expected HTTP 404 for unknown collection, got {other:?}"),
    }
}

/// 显式建集合（IP 度量）：响应回显生效参数，检索分数为原始点积（越大越近）。
/// EN: Explicit creation with the IP metric: the response echoes the effective
/// parameters and search scores are raw dot products (larger = closer).
#[test]
fn create_collection_ip_metric_round_trip() {
    let Some((client, col)) = connect("rust_it_create_ip") else {
        return;
    };

    // 先删同名残留：同名已存在会 409，删除保证用例可重复运行。
    // 清理只容忍 404（从未创建）；其他错误直接失败。
    // EN: Drop any leftover same-name collection first: an existing name yields
    // 409 on create, and deletion keeps the test rerun-safe. Cleanup tolerates
    // only 404 (never created); anything else fails.
    match client.delete_collection("rust_it_create_ip") {
        Ok(()) | Err(HouyiError::Http { status: 404, .. }) => {}
        Err(other) => panic!("cleanup delete failed: {other:?}"),
    }

    let created = client
        .create_collection(
            "rust_it_create_ip",
            TEST_DIMS,
            &CreateCollectionOptions {
                metric: Some("ip".to_string()),
                index_type: Some("hnsw".to_string()),
                hnsw: Some(HnswConfig {
                    m: 16,
                    ef_construction: 100,
                    ef_search: 64,
                }),
            },
        )
        .expect("explicit create");
    assert!(created.created);
    assert_eq!(created.collection, "rust_it_create_ip");
    assert_eq!(created.dims, TEST_DIMS);
    assert_eq!(created.metric, "ip");
    assert_eq!(created.index_type, "hnsw");
    let hnsw = created.hnsw.expect("hnsw parameters echoed");
    assert_eq!(
        (hnsw.m, hnsw.ef_construction, hnsw.ef_search),
        (16, 100, 64)
    );

    // 两个共线向量：IP 分数等于原始点积，查询向量对长向量的分数更高。
    // EN: Two collinear vectors: IP scores are the raw dot products, so the query
    // scores higher against the longer vector.
    let mut short = vec![0.0_f32; TEST_DIMS];
    short[0] = 1.0;
    let mut long = vec![0.0_f32; TEST_DIMS];
    long[0] = 2.0;
    col.upsert(&[
        Vector {
            id: 1,
            values: short.clone(),
            label: None,
            payload: None,
        },
        Vector {
            id: 2,
            values: long,
            label: None,
            payload: None,
        },
    ])
    .expect("upsert");
    wait_flushed(&col);

    let result = col
        .search(&short, 2, &SearchOptions::default())
        .expect("search");
    assert_eq!(result.ids(), vec![2, 1], "IP ranks the larger dot first");
    let scores = result.scores();
    assert!(
        (scores[0] - 2.0).abs() < 1e-5 && (scores[1] - 1.0).abs() < 1e-5,
        "IP scores must be raw dot products, got {scores:?}"
    );

    let status = col.status().expect("status");
    assert_eq!(status.get("metric").and_then(|v| v.as_str()), Some("ip"));
}
