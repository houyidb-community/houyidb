//! HouyiDB server — 启动入口：配置解析、子系统组装、pgwire/HTTP 监听。
//! Main entrypoint: config parsing, subsystem assembly, pgwire/HTTP listeners.

use houyi_core::config::BuildConfig;
use houyi_core::{
    detect_numa_topology, HouyiPage, NumaPolicy, RegionPool, ScalarPredicate, ScalarValue,
};
use houyi_persistence::wal_sidecar::{
    read_applied_lsn, sidecar_path_for_wal, write_applied_lsn_atomic,
};
use houyi_query::{PgWireQuery, QueryPlanner};
use houyi_server::collection::{CollectionBuilderConfig, CollectionManager};
use houyi_server::http_handler::{bench_build_config, run_benchmark_http_api};
use houyi_server::index_backend::IndexBackend;
use houyi_server::pgwire_handler::{
    execute_hybrid_with_scalar_bitset, execute_query, handle_pgwire_simple_query_loop,
    pgwire_accept_session, pgwire_connection_admit,
};
use houyi_server::recovery::*;
use houyi_server::types::*;
use houyi_vector::{DistanceMetric, HotBufferIndex};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::time::{self, Duration};
const FEATURE_PORTABLE: bool = true;
#[cfg(not(feature = "portable"))]
const FEATURE_PORTABLE: bool = false;

#[cfg(feature = "simd-avx512")]
const FEATURE_SIMD_AVX512: bool = true;
#[cfg(not(feature = "simd-avx512"))]
const FEATURE_SIMD_AVX512: bool = false;

#[cfg(feature = "cxl-memory")]
const FEATURE_CXL_MEMORY: bool = true;
#[cfg(not(feature = "cxl-memory"))]
const FEATURE_CXL_MEMORY: bool = false;

#[cfg(feature = "ha-raft")]
const FEATURE_HA_RAFT: bool = true;
#[cfg(not(feature = "ha-raft"))]
const FEATURE_HA_RAFT: bool = false;

/// TOML 配置加载（P4.3.1）：`--config <path>`（CLI 优先）或 `HOUYI_CONFIG` 指定；
/// 内容在合并链中位于 default 之上、ENV/CLI 之下。文件缺失/解析失败 fast-fail。
/// EN: Load the optional TOML config; the path itself is CLI > ENV, while the
/// file's values sit below ENV/CLI flags in the merge chain.
fn load_toml_config_base() -> Result<ServerConfig, String> {
    let mut cfg = ServerConfig::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    // last-wins：与其他旗标的循环覆写语义对齐（--config a.toml --config b.toml
    // 取 b）。first-wins 会与 CLI 主体解析行为发散，同一个进程里两套覆盖规则。
    // EN: last-wins, matching the CLI loop's overwrite semantics — a
    // first-wins scan here would give the same process two conflicting
    // override rules.
    let path = args
        .windows(2)
        .rfind(|w| w[0] == "--config")
        .map(|w| w[1].clone())
        .or_else(|| std::env::var("HOUYI_CONFIG").ok())
        .filter(|p| !p.trim().is_empty());
    let Some(path) = path else {
        return Ok(cfg);
    };
    apply_toml_config_file(&mut cfg, std::path::Path::new(&path))?;
    tracing::info!(path = %path, "loaded toml config");
    Ok(cfg)
}

/// 读 + 解析 + 应用单个 TOML 文件（独立函数便于单测）。
/// EN: Read, parse, and apply one TOML file (separated for unit testing).
fn apply_toml_config_file(cfg: &mut ServerConfig, path: &std::path::Path) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read config file {}: {e}", path.display()))?;
    let file: ServerConfigFile =
        toml::from_str(&text).map_err(|e| format!("parse config file {}: {e}", path.display()))?;
    // 凭据落在 TOML 里时检查文件权限：组/他人可读的 token 与明文写进
    // 世界可读文件等价，warn 提示收紧（不 fast-fail——权限模型因部署而异，
    // 如容器内 root-only 挂载）。仅 unix 有 mode 语义，其他平台跳过。
    // EN: When a credential lives in the TOML, check file permissions — a
    // group/world-readable token file is plaintext exposure; warn to tighten
    // (not fatal: permission models vary by deployment, e.g. root-only mounts
    // in containers). Unix-only; other platforms skip.
    #[cfg(unix)]
    if file.auth_token.is_some()
        || file.replication_auth_token.is_some()
        || file.sentinel_token.is_some()
        || file
            .auth_principals
            .iter()
            .any(|p| p.token.is_some() || p.token_sha256.is_some())
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mode = meta.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                // 三种凭据同一纪律（OCR 批次二十八 + EE M1）：漏检任何一种
                // 都会让该共享秘密静默逃过权限告警。
                // EN: all three credentials share the hygiene check (OCR batch 28
                // + EE M1); missing one lets that secret skip the warning.
                tracing::warn!(
                    path = %path.display(),
                    mode = format!("{mode:o}"),
                    "config file contains a credential (auth_token / \
                     replication_auth_token / sentinel_token) but is \
                     group/world-accessible; consider chmod 600"
                );
            }
        }
    }
    file.apply_to(cfg)
}

fn parse_config_from_env(mut cfg: ServerConfig) -> Result<ServerConfig, String> {
    if let Ok(v) = std::env::var("HOUYI_SERVER_WORKER_THRESHOLD") {
        cfg.worker_threshold = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_WORKER_THRESHOLD parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_WORKER_TICK_MS") {
        cfg.worker_tick_ms = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_WORKER_TICK_MS parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_FLUSH_POLICY") {
        cfg.flush_policy = parse_flush_policy(&v)?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_METRICS_INTERVAL_SECS") {
        cfg.metrics_interval_secs = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_METRICS_INTERVAL_SECS parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_HNSW_EF_SEARCH") {
        cfg.hnsw_ef_search = Some(
            v.parse::<usize>()
                .map_err(|e| format!("HOUYI_SERVER_HNSW_EF_SEARCH parse error: {e}"))?,
        );
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_VECTOR_DIM") {
        cfg.vector_dim = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_VECTOR_DIM parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_BENCHMARK_HTTP_ADDR") {
        cfg.benchmark_http_addr = v;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_PGWIRE_ADDR") {
        cfg.pgwire_addr = v;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_WAL_PATH") {
        cfg.wal_path = std::path::PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_WAL_WRITE_MODE") {
        cfg.wal_write_mode =
            parse_wal_write_mode(&v).map_err(|e| format!("HOUYI_SERVER_WAL_WRITE_MODE: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_PGWIRE_HNSW_M") {
        cfg.pgwire_hnsw_m = Some(
            v.parse::<usize>()
                .map_err(|e| format!("HOUYI_SERVER_PGWIRE_HNSW_M parse error: {e}"))?,
        );
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_PGWIRE_HNSW_EFC") {
        cfg.pgwire_hnsw_ef_construction = Some(
            v.parse::<usize>()
                .map_err(|e| format!("HOUYI_SERVER_PGWIRE_HNSW_EFC parse error: {e}"))?,
        );
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_INDEX_TYPE") {
        cfg.index_type = v;
    }
    if let Ok(v) = std::env::var("HOUYI_OPENAI_UPSTREAM") {
        cfg.openai_upstream = Some(v).filter(|s| !s.trim().is_empty());
    }
    if let Ok(v) = std::env::var("HOUYI_OPENAI_TIMEOUT_SECS") {
        cfg.openai_timeout_secs = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_OPENAI_TIMEOUT_SECS parse error: {e}"))?;
    }
    // 认证令牌（P4.2.1）仅 ENV 注入——不接 CLI 旗标（ps 可见即泄漏）。
    // fail-closed（OCR 359d85d[6]）：设置了却为空白 = 运维以为认证已开、
    // 实际静默放开——这种配置错误必须在启动期炸掉，不能降级为 None。
    // 存 trim 后的值：比较侧不 trim，" s3cret " 之类的值若原样存将永远
    // 匹配不上任何客户端。
    // EN: The auth token (P4.2.1) is ENV-only — no CLI flag (visible in `ps`).
    // fail-closed (OCR 359d85d[6]): set but blank means ops believe auth is on
    // while it is silently open — such a config error must fail at startup,
    // never degrade to None. Store the trimmed value: the comparison side does
    // not trim, so a value like " s3cret " would never match any client.
    if let Ok(v) = std::env::var("HOUYI_AUTH_TOKEN") {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(
                "HOUYI_AUTH_TOKEN is set but blank; refusing to start unauthenticated \
                 (unset it to disable auth explicitly)"
                    .to_string(),
            );
        }
        cfg.auth_token = Some(trimmed.to_string());
    }
    // TLS（P4.2.2）：证书/私钥只是路径而非密钥本体，CLI/ENV 双通道均可。
    // EN: TLS (P4.2.2): cert/key are paths rather than key material, so CLI/ENV both work.
    if let Ok(v) = std::env::var("HOUYI_TLS_CERT") {
        cfg.tls_cert_path = Some(v).filter(|s| !s.trim().is_empty());
    }
    if let Ok(v) = std::env::var("HOUYI_TLS_KEY") {
        cfg.tls_key_path = Some(v).filter(|s| !s.trim().is_empty());
    }
    // 水位线（P4.2.3）：连接/在途上限，非敏感配置，CLI/ENV 双通道。
    // EN: Watermarks (P4.2.3): connection/in-flight caps are non-sensitive, CLI/ENV both work.
    if let Ok(v) = std::env::var("HOUYI_SERVER_MAX_PGWIRE_CONNECTIONS") {
        cfg.max_pgwire_connections = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_MAX_PGWIRE_CONNECTIONS parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_MAX_HTTP_INFLIGHT") {
        cfg.max_http_inflight = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_MAX_HTTP_INFLIGHT parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_MAX_MEMORY_BYTES") {
        cfg.max_memory_bytes = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_MAX_MEMORY_BYTES parse error: {e}"))?;
    }
    // 慢查询阈值（P4.3.2）：0 = 禁用（opt-in），非敏感配置，CLI/ENV 双通道。
    // EN: Slow-query threshold (P4.3.2): 0 disables it (opt-in); non-sensitive, CLI/ENV both work.
    if let Ok(v) = std::env::var("HOUYI_SERVER_SLOW_QUERY_MS") {
        cfg.slow_query_ms = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_SLOW_QUERY_MS parse error: {e}"))?;
    }
    // 图快照自动触发（P4.4.0e）：0 = 关闭对应通道。
    // EN: Graph snapshot auto-trigger (P4.4.0e): 0 disables the corresponding channel.
    if let Ok(v) = std::env::var("HOUYI_SERVER_SNAPSHOT_INTERVAL_SECS") {
        cfg.snapshot_interval_secs = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_SNAPSHOT_INTERVAL_SECS parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_SNAPSHOT_WAL_THRESHOLD_BYTES") {
        cfg.snapshot_wal_threshold_bytes = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_SNAPSHOT_WAL_THRESHOLD_BYTES parse error: {e}"))?;
    }
    // 标量页自动压实（P5.0.1）：0 = 关闭自动压实，管理端点仍可触发。
    // EN: Scalar auto-compaction (P5.0.1): 0 disables the periodic path; the
    // admin endpoint still works.
    if let Ok(v) = std::env::var("HOUYI_SERVER_COLLECTIONS_SCALAR_COMPACT_INTERVAL_SECS") {
        cfg.collections_scalar_compact_interval_secs = v.parse::<u64>().map_err(|e| {
            format!("HOUYI_SERVER_COLLECTIONS_SCALAR_COMPACT_INTERVAL_SECS parse error: {e}")
        })?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_COLLECTIONS_SCALAR_COMPACT_STALE_RATIO") {
        cfg.collections_scalar_compact_stale_ratio = v.parse::<f64>().map_err(|e| {
            format!("HOUYI_SERVER_COLLECTIONS_SCALAR_COMPACT_STALE_RATIO parse error: {e}")
        })?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_COLLECTIONS_WAL_SEGMENT_BYTES") {
        cfg.collections_wal_segment_bytes = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_COLLECTIONS_WAL_SEGMENT_BYTES parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_COLLECTIONS_WAL_MAX_SEGMENTS") {
        cfg.collections_wal_max_segments = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_COLLECTIONS_WAL_MAX_SEGMENTS parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_COLLECTIONS_WAL_FOLLOWER_GRACE_SECS") {
        cfg.collections_wal_follower_grace_secs = v.parse::<u64>().map_err(|e| {
            format!("HOUYI_SERVER_COLLECTIONS_WAL_FOLLOWER_GRACE_SECS parse error: {e}")
        })?;
    }
    // 复制（P4.4.1）：role 枚举笔误 fast-fail（与 flush_policy 同纪律）。
    // EN: Replication (P4.4.1): a role enum typo fast-fails (same discipline as flush_policy).
    if let Ok(v) = std::env::var("HOUYI_SERVER_ROLE") {
        cfg.role = houyi_server::types::parse_replication_role(&v)
            .map_err(|e| format!("HOUYI_SERVER_ROLE: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_REPLICATION_LISTEN_ADDR") {
        cfg.replication_listen_addr = v;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_REPLICATION_LEADER_ADDR") {
        cfg.replication_leader_addr = Some(v).filter(|s| !s.trim().is_empty());
    }
    // 复制认证（P4.6.3）：空白 = 运维以为认证已开，启动期报错（fail-closed）。
    // EN: Replication auth (P4.6.3): blank means ops think auth is on — error at startup.
    if let Ok(v) = std::env::var("HOUYI_SERVER_REPLICATION_AUTH_TOKEN") {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(
                "HOUYI_SERVER_REPLICATION_AUTH_TOKEN is blank; unset it to disable \
                 replication auth explicitly"
                    .to_string(),
            );
        }
        // 长度上限启动期校验（OCR 批次二十八）：否则每个会话在 encode 时
        // 失败，形成无限重连而不是清晰的配置错误。
        // EN: Startup validation of the length cap (OCR batch 28): otherwise every
        // session fails at encode time, causing endless reconnects instead of a clear
        // configuration error.
        houyi_persistence::wal_replication::validate_auth_token(trimmed)
            .map_err(|e| format!("HOUYI_SERVER_REPLICATION_AUTH_TOKEN: {e}"))?;
        cfg.replication_auth_token = Some(trimmed.to_string());
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_REPLICATION_QUEUE_BATCHES") {
        cfg.replication_queue_batches = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_REPLICATION_QUEUE_BATCHES parse error: {e}"))?;
    }
    // P4.4.2：sync 等待超时 + 新建 collection 默认 HA 级别（枚举笔误 fast-fail）。
    // EN: P4.4.2: sync-wait timeout + default HA level for new collections (enum typo
    // fast-fails).
    if let Ok(v) = std::env::var("HOUYI_SERVER_REPLICATION_SYNC_TIMEOUT_MS") {
        cfg.replication_sync_timeout_ms = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_REPLICATION_SYNC_TIMEOUT_MS parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_DEFAULT_HA_LEVEL") {
        cfg.default_ha_level = houyi_persistence::ha_level::HaLevel::parse(&v)
            .map_err(|e| format!("HOUYI_SERVER_DEFAULT_HA_LEVEL: {e}"))?;
    }
    // 审计（P4.7.0）：空白路径 = 运维以为审计已开，启动期报错（fail-closed）。
    // EN: Auditing (P4.7.0): a blank path means ops believe auditing is on —
    // error at startup (fail-closed).
    if let Ok(v) = std::env::var("HOUYI_SERVER_AUDIT_LOG") {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(
                "HOUYI_SERVER_AUDIT_LOG is blank; unset it to disable auditing explicitly"
                    .to_string(),
            );
        }
        cfg.audit_log_path = Some(std::path::PathBuf::from(trimmed));
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_AUDIT_INCLUDE_READS") {
        cfg.audit_include_reads = match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "off" | "no" => false,
            other => {
                return Err(format!(
                    "HOUYI_SERVER_AUDIT_INCLUDE_READS: invalid boolean {other:?} \
                     (use 1/0, true/false, yes/no, on/off)"
                ))
            }
        };
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_AUDIT_MAX_BYTES") {
        cfg.audit_max_bytes = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_AUDIT_MAX_BYTES parse error: {e}"))?;
    }
    // 静态加密（P4.7.3）：空白路径 = 运维以为加密已开，启动期报错。
    // EN: Encryption at rest (P4.7.3): a blank path means ops believe encryption is
    // on — error at startup.
    if let Ok(v) = std::env::var("HOUYI_SERVER_ENCRYPTION_KEY_FILE") {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(
                "HOUYI_SERVER_ENCRYPTION_KEY_FILE is blank; unset it to disable \
                 encryption at rest explicitly"
                    .to_string(),
            );
        }
        cfg.encryption_key_file = Some(std::path::PathBuf::from(trimmed));
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_ENCRYPTION_KEY_ID") {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(
                "HOUYI_SERVER_ENCRYPTION_KEY_ID is blank; unset it to use the \
                 keyring's active= selection"
                    .to_string(),
            );
        }
        cfg.encryption_key_id = Some(trimmed.to_string());
    }
    // Sentinel 控制面（EE M1）：token 属敏感项，仅 ENV/TOML——CLI 参数对 ps
    // 可见，与 HOUYI_AUTH_TOKEN 同规。空白值 fail-closed。
    // EN: sentinel control plane (EE M1): the token is ENV/TOML only, matching
    // the auth-token discipline; blank values fail closed.
    if let Ok(v) = std::env::var("HOUYI_SERVER_SENTINEL_TOKEN") {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(
                "HOUYI_SERVER_SENTINEL_TOKEN is blank; unset it to disable the \
                 sentinel control plane explicitly"
                    .to_string(),
            );
        }
        cfg.sentinel_token = Some(trimmed.to_string());
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_REQUIRE_LEADER_LEASE") {
        cfg.require_leader_lease = match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "off" | "no" => false,
            other => {
                return Err(format!(
                    "HOUYI_SERVER_REQUIRE_LEADER_LEASE: invalid boolean {other:?} \
                     (use 1/0, true/false, yes/no, on/off)"
                ))
            }
        };
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_LEADER_LEASE_MAX_TTL_MS") {
        cfg.leader_lease_max_ttl_ms = v
            .parse::<u64>()
            .map_err(|e| format!("HOUYI_SERVER_LEADER_LEASE_MAX_TTL_MS parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_REPLICATION_SYNC_QUORUM") {
        cfg.replication_sync_quorum = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_REPLICATION_SYNC_QUORUM parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_QUERY_LOG_CAPACITY") {
        cfg.query_log_capacity = v
            .parse::<usize>()
            .map_err(|e| format!("HOUYI_SERVER_QUERY_LOG_CAPACITY parse error: {e}"))?;
    }
    if let Ok(v) = std::env::var("HOUYI_SERVER_QUERY_LOG_SAMPLE_EVERY") {
        cfg.query_log_sample_every = v
            .parse::<u32>()
            .map_err(|e| format!("HOUYI_SERVER_QUERY_LOG_SAMPLE_EVERY parse error: {e}"))?;
    }
    Ok(cfg)
}

fn print_help() {
    println!("HouyiDB server usage:");
    println!("  cargo run -p houyi-server --bin houyi-server -- [options]");
    println!();
    println!("Options (CLI > ENV > default):");
    println!("  --worker-threshold <usize>     ENV HOUYI_SERVER_WORKER_THRESHOLD");
    println!("  --worker-tick-ms <u64>         ENV HOUYI_SERVER_WORKER_TICK_MS");
    println!("  --flush-policy <name>          ENV HOUYI_SERVER_FLUSH_POLICY");
    println!("                                 name: threshold | interval | hybrid");
    println!("  --metrics-interval-secs <u64>  ENV HOUYI_SERVER_METRICS_INTERVAL_SECS");
    println!(
        "  --hnsw-ef-search <usize>       ENV HOUYI_SERVER_HNSW_EF_SEARCH (omit = use inferred)"
    );
    println!("  --vector-dim <usize>           ENV HOUYI_SERVER_VECTOR_DIM");
    println!("  --benchmark-http-addr <addr>   ENV HOUYI_SERVER_BENCHMARK_HTTP_ADDR");
    println!("  --pgwire-addr <addr>           ENV HOUYI_SERVER_PGWIRE_ADDR");
    println!("  HNSW: HOUYI_HNSW_PARALLEL_BATCH_BUILD / HOUYI_HNSW_MAX_BUILD_BATCH / HOUYI_HNSW_PARALLEL_MIN_GRAPH / HOUYI_HNSW_SEED_SERIAL_POINTS");
    println!("        startup defaults if unset: PARALLEL=1, MAX_BUILD_BATCH=8192 (MIN_GRAPH unset -> 4096; SEED_SERIAL unset -> 256 in houyi-vector)");
    println!("  HNSW: POST /v1/optimize 每次结束发一条日志(tracing info)；HOUYI_HNSW_OPTIMIZE_LOG_FILE=path 再追加到文件；HOUYI_HNSW_OPTIMIZE_LOG_QUIET=1 仅写文件");
    println!("  --pgwire-hnsw-m <usize>        ENV HOUYI_SERVER_PGWIRE_HNSW_M (optional; match VectorDBBench --m)");
    println!("  --pgwire-hnsw-efc <usize>      ENV HOUYI_SERVER_PGWIRE_HNSW_EFC (optional; match --ef-construction)");
    println!("  --index-type <type>            ENV HOUYI_SERVER_INDEX_TYPE (hnsw|clustered|flat-clustered|flat|pq-flat|hii-l0)");
    println!("  --openai-upstream <url>        ENV HOUYI_OPENAI_UPSTREAM (OpenAI 兼容 embeddings 网关上游 base URL; 不设置 = /v1/embeddings 返回 501)");
    println!("  --openai-timeout-secs <u64>    ENV HOUYI_OPENAI_TIMEOUT_SECS (default 30)");
    println!("  上游 API key 仅经 ENV HOUYI_OPENAI_API_KEY 注入（不接 CLI 旗标——ps 可见即泄漏）");
    println!("  认证(P4.2.1): ENV HOUYI_AUTH_TOKEN=token 启用(仅 ENV, 同不上 CLI); HTTP /v1/* 除 /v1/health /readyz 需 'Authorization: Bearer <token>' 或 'X-Api-Key: <token>'; pgwire 走明文密码质询(仅配合 TLS/localhost 部署)");
    println!("  --tls-cert <path>              ENV HOUYI_TLS_CERT (PEM 证书链; 与 --tls-key 成对, HTTP/pgwire 共用)");
    println!(
        "  --tls-key <path>               ENV HOUYI_TLS_KEY (PEM 私钥; 不设置 = 两监听均明文)"
    );
    println!("  --max-pgwire-connections <n>   ENV HOUYI_SERVER_MAX_PGWIRE_CONNECTIONS (default 200; 超限握手后 FATAL 53300)");
    println!("  --max-http-inflight <n>        ENV HOUYI_SERVER_MAX_HTTP_INFLIGHT (default 512; 超限 503, /v1/health /readyz /metrics 豁免)");
    println!("  --max-memory-bytes <n>         ENV HOUYI_SERVER_MAX_MEMORY_BYTES (default 0=禁用; RSS 超限拒绝写入 HTTP 503/pgwire 53200; 仅 Linux 有 RSS 源)");
    println!("  --slow-query-ms <n>            ENV HOUYI_SERVER_SLOW_QUERY_MS (default 0=禁用; pgwire/HTTP 检索超阈值打 warn 日志)");
    println!("  --snapshot-interval-secs <n>   ENV HOUYI_SERVER_SNAPSHOT_INTERVAL_SECS (default 600; 0=关闭周期图快照)");
    println!("  --snapshot-wal-threshold-bytes <n>  ENV HOUYI_SERVER_SNAPSHOT_WAL_THRESHOLD_BYTES (default 268435456; 0=关闭 WAL 增量阈值触发)");
    println!("  --collections-scalar-compact-interval-secs <n>  ENV HOUYI_SERVER_COLLECTIONS_SCALAR_COMPACT_INTERVAL_SECS (default 600; 0=关闭标量自动压实)");
    println!("  --collections-scalar-compact-stale-ratio <f>    ENV HOUYI_SERVER_COLLECTIONS_SCALAR_COMPACT_STALE_RATIO (default 0.5; 0..=1)");
    println!("  --collections-wal-segment-bytes <n>             ENV HOUYI_SERVER_COLLECTIONS_WAL_SEGMENT_BYTES (default 67108864; 0=只手动轮转)");
    println!("  --collections-wal-max-segments <n>              ENV HOUYI_SERVER_COLLECTIONS_WAL_MAX_SEGMENTS (default 16; 0=关闭删除)");
    println!("  --collections-wal-follower-grace-secs <n>       ENV HOUYI_SERVER_COLLECTIONS_WAL_FOLLOWER_GRACE_SECS (default 60; leader 断连/启动/promote 宽限)");
    println!("  关停图快照(P4.4.0e)默认开启: ENV HOUYI_SHUTDOWN_SNAPSHOT=0 关闭 (行为变更——此前默认关闭、=1 开启, 且写出格式由 JSON 向量快照切换为 HNSW 图快照)");
    println!(
        "  复制(P4.4.1): --role disabled|leader|follower  ENV HOUYI_SERVER_ROLE (default disabled)"
    );
    println!("  --replication-listen-addr <addr>  ENV HOUYI_SERVER_REPLICATION_LISTEN_ADDR (default 0.0.0.0:9876; role=leader 生效)");
    println!("  --replication-leader-addr <addr>  ENV HOUYI_SERVER_REPLICATION_LEADER_ADDR (role=follower 必填)");
    println!("  (无 CLI 旗标) 复制认证共享秘密          ENV HOUYI_SERVER_REPLICATION_AUTH_TOKEN / TOML replication_auth_token (leader 与 follower 必须一致; 缺省不认证)");
    println!("  --replication-queue-batches <n>   ENV HOUYI_SERVER_REPLICATION_QUEUE_BATCHES (default 4096; 慢 follower 积满即踢)");
    println!("  --replication-sync-timeout-ms <n> ENV HOUYI_SERVER_REPLICATION_SYNC_TIMEOUT_MS (default 2000; semi/full_sync 等 follower ACK 上限, 超时降级 async + 指标)");
    println!("  --replication-sync-quorum <n>  ENV HOUYI_SERVER_REPLICATION_SYNC_QUORUM (default 0=任一 ACK; >0 要求 n 个活跃 follower 的 DURABLE ≥ 写入 LSN, 会降低可用性)");
    println!("  (无 CLI 旗标) sentinel 控制面秘密     ENV HOUYI_SERVER_SENTINEL_TOKEN / TOML sentinel_token (sentinel 角色: lease/demote/promote/replication-status; 需同时配置数据面 token)");
    println!("  --require-leader-lease         ENV HOUYI_SERVER_REQUIRE_LEADER_LEASE (default false; 开启后 leader 必须有有效租约才可写, 过期 503 lease_expired)");
    println!("  --leader-lease-max-ttl-ms <n>  ENV HOUYI_SERVER_LEADER_LEASE_MAX_TTL_MS (default 60000; 硬上限同一常量)");
    println!("  --query-log-capacity <n>       ENV HOUYI_SERVER_QUERY_LOG_CAPACITY (default 0=关闭; 有界查询日志, 只记元数据)");
    println!("  --query-log-sample-every <n>   ENV HOUYI_SERVER_QUERY_LOG_SAMPLE_EVERY (default 1=全量; 每 n 条记 1 条)");
    println!("  --default-ha-level <name>       ENV HOUYI_SERVER_DEFAULT_HA_LEVEL (default async; 新建 collection 默认 HA 级别: async|semi_sync|full_sync)");
    println!("  --audit-log-path <path>        ENV HOUYI_SERVER_AUDIT_LOG (default 关闭; 开启后启动即校验哈希链, 链断拒绝启动)");
    println!("  --audit-include-reads          ENV HOUYI_SERVER_AUDIT_INCLUDE_READS (default false; 读请求也入审计)");
    println!("  --audit-max-bytes <n>          ENV HOUYI_SERVER_AUDIT_MAX_BYTES (default 268435456; 段轮转阈值, 0=不轮转; 旧段不删除)");
    println!("  --encryption-key-file <path>   ENV HOUYI_SERVER_ENCRYPTION_KEY_FILE (keyring 文件; 开启 WAL/快照静态加密)");
    println!("  --encryption-key-id <id>       ENV HOUYI_SERVER_ENCRYPTION_KEY_ID (新写入的活动密钥 id; 缺省取 keyring active=)");
    println!("  (TOML) [[auth_principals]]     RBAC 主体: name/role(admin|writer|reader)/token 或 token_sha256/collections(可选, [\"*\"]=全部); legacy auth_token 仍为全权 admin");
    println!("                                 运行期设置: POST /v1/collections/{{name}}/ha_level {{\"ha_level\":\"semi_sync\"}} (WAL-first, 重放恢复; collection 级 > 默认)");
    println!("                                 role=follower 时写端点 503/pgwire 25006 只读; 复制要求 --wal-write-mode async");
    println!("                                 复制范围仅 collections.wal(collection 路径); benchmark 主索引(POST /v1/vectors/upsert 与 pgwire 默认索引)不复制——failover 后新主无该路径数据");
    println!("                                 复制协议 v1 无认证/TLS: 任何可达监听端口的机器均可全量拉取数据或伪造 follower 拖垮 leader, 必须靠网络隔离(内网/防火墙)保护");
    println!("  --config <path>                ENV HOUYI_CONFIG (TOML 配置文件; 合并优先级 CLI 旗标 > ENV > TOML > default; 文件缺键落默认, 枚举笔误 fast-fail)");
    println!("  --wal-path <path>              ENV HOUYI_SERVER_WAL_PATH (vector WAL file)");
    println!("  --wal-write-mode async|sync    ENV HOUYI_SERVER_WAL_WRITE_MODE (default async)");
    println!("                                 async = background WAL writer; sync = mutex + per-record flush.");
    println!("                                 Startup-only; change requires process restart.");
    println!("  -h, --help                     Show this help and exit");
}

/// 主 WAL + collections.wal 当前字节数之和（图快照阈值通道的输入）。
/// Async 模式 active_file_len 要穿有界通道等写线程应答（阻塞 recv），移出
/// runtime worker；加 5s 上限——写线程挂死时令本 tick 的阈值检查降级跳过
/// （返回 None），而不是让整个快照循环停摆。
/// EN: Sum of active WAL bytes for the snapshot threshold trigger. Reads are
/// moved off the runtime and bounded at 5s — a wedged writer thread degrades
/// this tick instead of stalling the snapshot loop.
async fn total_wal_bytes(wal_main: &Arc<WalHandle>, wal_coll: &Arc<WalHandle>) -> Option<u64> {
    let m = Arc::clone(wal_main);
    let c = Arc::clone(wal_coll);
    let read = tokio::task::spawn_blocking(move || {
        // 轮转后 active 变小、归档变大；只算 active 会让阈值通道误判「WAL 变小」。
        // EN: After rotation the active file shrinks while archives grow; counting
        // only active bytes would make the threshold channel see false shrinkage.
        let a = m.segment_stats().ok()?;
        let b = c.segment_stats().ok()?;
        Some(a.active_bytes + a.archived_bytes + b.active_bytes + b.archived_bytes)
    });
    match time::timeout(Duration::from_secs(5), read).await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "graph snapshot: wal size read task join failed, skipping threshold check this tick");
            None
        }
        Err(_) => {
            tracing::warn!(
                "graph snapshot: wal size read timed out, skipping threshold check this tick"
            );
            None
        }
    }
}

/// WAL 增量阈值判定（图快照阈值通道）。基线未知（启动/上轮读失败）时**只
/// 建基线不触发**——基线读失败落 0 会把整个存量 WAL 计为「新增量」，启动
/// 30s 后白白触发一次数百 MB 全量快照。
/// EN: Threshold trigger with explicit baseline semantics — an unknown
/// baseline only gets established, never fires; falling back to 0 would
/// count the entire pre-existing WAL as fresh delta and spuriously trigger
/// a full snapshot right after boot.
fn wal_threshold_due(last_bytes: Option<u64>, current: Option<u64>, threshold: u64) -> bool {
    match (last_bytes, current) {
        (Some(last), Some(cur)) => threshold > 0 && cur.saturating_sub(last) >= threshold,
        _ => false,
    }
}

/// 关停图快照开关解析（HOUYI_SHUTDOWN_SNAPSHOT）。默认 ON 后「裸串 != "0"」
/// 的写法会让 "false"/"off"/尾部空格/笔误全部静默保持开启——与文档的
/// 「=0 关闭」承诺不符。显式枚举关闭值（trim + 大小写不敏感），其余非空
/// 值 warn 后落默认 ON（关停路径不 fast-fail）。
/// EN: Strict parse for the shutdown-snapshot env knob — explicit disable
/// values, unrecognized non-empty values warn and fall back to ON.
fn shutdown_snapshot_enabled(raw: Option<&str>) -> bool {
    match raw {
        None => true,
        Some(v) => {
            let v = v.trim();
            if v.is_empty() {
                return true;
            }
            let lower = v.to_ascii_lowercase();
            if matches!(lower.as_str(), "0" | "false" | "off" | "no") {
                return false;
            }
            // 显式开启族一并识别（true/yes/on 是常见布尔写法）：能识别就不
            // 打 warn—— recurring 误报会训练运维忽略告警，稀释真笔误的信号。
            // EN: Recognize the explicit enable family too (true/yes/on are common
            // booleans): when recognized, do not warn — recurring false alarms train
            // operators to ignore alerts and dilute the signal from real typos.
            if !matches!(lower.as_str(), "1" | "true" | "yes" | "on") {
                tracing::warn!(value = %v,
                    "HOUYI_SHUTDOWN_SNAPSHOT: unrecognized value, defaulting to ON (use =0/false/off/no to disable)");
            }
            true
        }
    }
}

fn parse_value<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    raw.parse::<T>()
        .map_err(|e| format!("{flag} parse error: {e}"))
}

fn parse_config_from_cli(mut cfg: ServerConfig) -> Result<ServerConfig, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            // -h/--help 由 main 的 argv 预扫描统一处理（先于日志/配置加载），
            // 此循环不再承接——跑到这里说明预扫描放行了它（如作为别的旗标的值），
            // 按未知旗标报错。
            // EN: -h/--help is handled by main's argv pre-scan (before logging/config
            // load); this loop does not handle it — reaching here means the pre-scan
            // let it through (e.g. as another flag's value), so report an unknown flag.
            "--worker-threshold" => {
                i += 1;
                let v = args.get(i).ok_or("--worker-threshold requires a value")?;
                cfg.worker_threshold = parse_value("--worker-threshold", v)?;
            }
            "--worker-tick-ms" => {
                i += 1;
                let v = args.get(i).ok_or("--worker-tick-ms requires a value")?;
                cfg.worker_tick_ms = parse_value("--worker-tick-ms", v)?;
            }
            "--flush-policy" => {
                i += 1;
                let v = args.get(i).ok_or("--flush-policy requires a value")?;
                cfg.flush_policy = parse_flush_policy(v)?;
            }
            "--metrics-interval-secs" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--metrics-interval-secs requires a value")?;
                cfg.metrics_interval_secs = parse_value("--metrics-interval-secs", v)?;
            }
            "--hnsw-ef-search" => {
                i += 1;
                let v = args.get(i).ok_or("--hnsw-ef-search requires a value")?;
                cfg.hnsw_ef_search = Some(parse_value("--hnsw-ef-search", v)?);
            }
            "--vector-dim" => {
                i += 1;
                let v = args.get(i).ok_or("--vector-dim requires a value")?;
                cfg.vector_dim = parse_value("--vector-dim", v)?;
            }
            "--benchmark-http-addr" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--benchmark-http-addr requires a value")?;
                cfg.benchmark_http_addr = v.to_string();
            }
            "--pgwire-addr" => {
                i += 1;
                let v = args.get(i).ok_or("--pgwire-addr requires a value")?;
                cfg.pgwire_addr = v.to_string();
            }
            "--wal-path" => {
                i += 1;
                let v = args.get(i).ok_or("--wal-path requires a value")?;
                cfg.wal_path = std::path::PathBuf::from(v);
            }
            "--wal-write-mode" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--wal-write-mode requires async or sync")?;
                cfg.wal_write_mode = parse_wal_write_mode(v)?;
            }
            "--pgwire-hnsw-m" => {
                i += 1;
                let v = args.get(i).ok_or("--pgwire-hnsw-m requires a value")?;
                cfg.pgwire_hnsw_m = Some(parse_value("--pgwire-hnsw-m", v)?);
            }
            "--pgwire-hnsw-efc" => {
                i += 1;
                let v = args.get(i).ok_or("--pgwire-hnsw-efc requires a value")?;
                cfg.pgwire_hnsw_ef_construction = Some(parse_value("--pgwire-hnsw-efc", v)?);
            }
            "--index-type" => {
                i += 1;
                let v = args.get(i).ok_or(
                    "--index-type requires hnsw|clustered|flat-clustered|flat|pq-flat|hii-l0",
                )?;
                cfg.index_type = v.to_string();
            }
            "--openai-upstream" => {
                i += 1;
                let v = args.get(i).ok_or("--openai-upstream requires a value")?;
                cfg.openai_upstream = Some(v.to_string());
            }
            "--openai-timeout-secs" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--openai-timeout-secs requires a value")?;
                cfg.openai_timeout_secs = parse_value("--openai-timeout-secs", v)?;
            }
            // 空白过滤对齐 ENV 语义：空/纯空白视为未设置（Some("") 会在
            // 启动期以「读文件失败」报错，把配置笔误伪装成文件问题）。
            // EN: Whitespace filtering mirrors ENV semantics — an empty or
            // all-blank value counts as unset; Some("") would otherwise
            // surface as a file-read error, disguising a config typo.
            "--tls-cert" => {
                i += 1;
                let v = args.get(i).ok_or("--tls-cert requires a value")?;
                cfg.tls_cert_path = Some(v.to_string()).filter(|s| !s.trim().is_empty());
            }
            "--tls-key" => {
                i += 1;
                let v = args.get(i).ok_or("--tls-key requires a value")?;
                cfg.tls_key_path = Some(v.to_string()).filter(|s| !s.trim().is_empty());
            }
            "--max-pgwire-connections" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--max-pgwire-connections requires a value")?;
                cfg.max_pgwire_connections = parse_value("--max-pgwire-connections", v)?;
            }
            "--max-http-inflight" => {
                i += 1;
                let v = args.get(i).ok_or("--max-http-inflight requires a value")?;
                cfg.max_http_inflight = parse_value("--max-http-inflight", v)?;
            }
            "--max-memory-bytes" => {
                i += 1;
                let v = args.get(i).ok_or("--max-memory-bytes requires a value")?;
                cfg.max_memory_bytes = parse_value("--max-memory-bytes", v)?;
            }
            "--slow-query-ms" => {
                i += 1;
                let v = args.get(i).ok_or("--slow-query-ms requires a value")?;
                cfg.slow_query_ms = parse_value("--slow-query-ms", v)?;
            }
            "--snapshot-interval-secs" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--snapshot-interval-secs requires a value")?;
                cfg.snapshot_interval_secs = parse_value("--snapshot-interval-secs", v)?;
            }
            "--snapshot-wal-threshold-bytes" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--snapshot-wal-threshold-bytes requires a value")?;
                cfg.snapshot_wal_threshold_bytes =
                    parse_value("--snapshot-wal-threshold-bytes", v)?;
            }
            "--collections-scalar-compact-interval-secs" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--collections-scalar-compact-interval-secs requires a value")?;
                cfg.collections_scalar_compact_interval_secs =
                    parse_value("--collections-scalar-compact-interval-secs", v)?;
            }
            "--collections-scalar-compact-stale-ratio" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--collections-scalar-compact-stale-ratio requires a value")?;
                cfg.collections_scalar_compact_stale_ratio =
                    parse_value("--collections-scalar-compact-stale-ratio", v)?;
            }
            "--collections-wal-segment-bytes" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--collections-wal-segment-bytes requires a value")?;
                cfg.collections_wal_segment_bytes =
                    parse_value("--collections-wal-segment-bytes", v)?;
            }
            "--collections-wal-max-segments" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--collections-wal-max-segments requires a value")?;
                cfg.collections_wal_max_segments =
                    parse_value("--collections-wal-max-segments", v)?;
            }
            "--collections-wal-follower-grace-secs" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--collections-wal-follower-grace-secs requires a value")?;
                cfg.collections_wal_follower_grace_secs =
                    parse_value("--collections-wal-follower-grace-secs", v)?;
            }
            "--role" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--role requires disabled|leader|follower")?;
                cfg.role = houyi_server::types::parse_replication_role(v)
                    .map_err(|e| format!("--role: {e}"))?;
            }
            "--replication-listen-addr" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--replication-listen-addr requires a value")?;
                cfg.replication_listen_addr = v.to_string();
            }
            "--replication-leader-addr" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--replication-leader-addr requires a value")?;
                cfg.replication_leader_addr = Some(v.to_string()).filter(|s| !s.trim().is_empty());
            }
            "--replication-queue-batches" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--replication-queue-batches requires a value")?;
                cfg.replication_queue_batches = parse_value("--replication-queue-batches", v)?;
            }
            "--replication-sync-timeout-ms" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--replication-sync-timeout-ms requires a value")?;
                cfg.replication_sync_timeout_ms = parse_value("--replication-sync-timeout-ms", v)?;
            }
            "--default-ha-level" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--default-ha-level requires async|semi_sync|full_sync")?;
                cfg.default_ha_level = houyi_persistence::ha_level::HaLevel::parse(v)
                    .map_err(|e| format!("--default-ha-level: {e}"))?;
            }
            "--audit-log-path" => {
                i += 1;
                let v = args.get(i).ok_or("--audit-log-path requires a value")?;
                // 空白与 ENV/TOML 同纪律 fail-closed：包装脚本传空变量时
                // 不得静默关掉审计（OCR 批次三十三）。
                // EN: a blank CLI value is fatal, matching ENV/TOML — it must
                // not silently disable auditing.
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    return Err(
                        "--audit-log-path requires a non-empty path; omit the flag to \
                         disable auditing"
                            .to_string(),
                    );
                }
                cfg.audit_log_path = Some(std::path::PathBuf::from(trimmed));
            }
            "--audit-include-reads" => {
                cfg.audit_include_reads = true;
            }
            // Sentinel 控制面（EE M1）：token 不走 CLI（ps 可见），其余非敏感。
            // EN: sentinel control plane (EE M1) — the token stays off the CLI.
            "--require-leader-lease" => {
                cfg.require_leader_lease = true;
            }
            "--leader-lease-max-ttl-ms" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--leader-lease-max-ttl-ms requires a value")?;
                cfg.leader_lease_max_ttl_ms = parse_value("--leader-lease-max-ttl-ms", v)?;
            }
            "--replication-sync-quorum" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--replication-sync-quorum requires a value")?;
                cfg.replication_sync_quorum = parse_value("--replication-sync-quorum", v)?;
            }
            "--query-log-capacity" => {
                i += 1;
                let v = args.get(i).ok_or("--query-log-capacity requires a value")?;
                cfg.query_log_capacity = parse_value("--query-log-capacity", v)?;
            }
            "--query-log-sample-every" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--query-log-sample-every requires a value")?;
                cfg.query_log_sample_every = parse_value("--query-log-sample-every", v)?;
            }
            "--audit-max-bytes" => {
                i += 1;
                let v = args.get(i).ok_or("--audit-max-bytes requires a value")?;
                cfg.audit_max_bytes = parse_value("--audit-max-bytes", v)?;
            }
            "--encryption-key-file" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--encryption-key-file requires a value")?;
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    return Err(
                        "--encryption-key-file requires a non-empty path; omit the flag \
                         to disable encryption at rest"
                            .to_string(),
                    );
                }
                cfg.encryption_key_file = Some(std::path::PathBuf::from(trimmed));
            }
            "--encryption-key-id" => {
                i += 1;
                let v = args.get(i).ok_or("--encryption-key-id requires a value")?;
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    return Err(
                        "--encryption-key-id requires a non-empty id; omit the flag to \
                         use the keyring's active= selection"
                            .to_string(),
                    );
                }
                cfg.encryption_key_id = Some(trimmed.to_string());
            }
            // 已在 load_toml_config_base 消费（路径发现），此处仅吞掉避免 unknown。
            // EN: Already consumed by load_toml_config_base (path discovery); swallow
            // it here only to avoid reporting it as unknown.
            "--config" => {
                i += 1;
                if args.get(i).is_none() {
                    return Err("--config requires a value".to_string());
                }
            }
            _ => return Err(format!("unknown option: {flag}. use --help for usage")),
        }
        i += 1;
    }

    validate_config(&cfg)?;
    Ok(cfg)
}

/// 配置组合校验（独立于 argv 解析，供单元测试直调）。
/// EN: Cross-field config validation, separated from argv parsing so unit
/// tests can call it directly.
fn validate_config(cfg: &ServerConfig) -> Result<(), String> {
    if cfg.worker_threshold == 0 {
        return Err("worker_threshold must be > 0".to_string());
    }
    if cfg.worker_tick_ms == 0 {
        return Err("worker_tick_ms must be > 0".to_string());
    }
    if cfg.metrics_interval_secs == 0 {
        return Err("metrics_interval_secs must be > 0".to_string());
    }
    if cfg.hnsw_ef_search == Some(0) {
        return Err("hnsw_ef_search must be > 0 when set".to_string());
    }
    if cfg.vector_dim == 0 {
        return Err("vector_dim must be > 0".to_string());
    }
    if cfg.benchmark_http_addr.trim().is_empty() {
        return Err("benchmark_http_addr must not be empty".to_string());
    }
    if cfg.pgwire_addr.trim().is_empty() {
        return Err("pgwire_addr must not be empty".to_string());
    }
    if cfg.openai_timeout_secs == 0 {
        return Err("openai_timeout_secs must be > 0".to_string());
    }
    // TLS 证书与私钥必须成对配置——只给一个是典型的部署笔误，fast-fail
    // 比静默半开（一耳明文一耳密文）安全。
    // EN: TLS certificate and private key must be configured as a pair — supplying
    // only one is a typical deployment typo; fast-fail is safer than silently running
    // half-open (one listener plaintext, one encrypted).
    if cfg.tls_cert_path.is_some() != cfg.tls_key_path.is_some() {
        return Err(
            "--tls-cert 与 --tls-key（或 HOUYI_TLS_CERT/HOUYI_TLS_KEY）必须成对出现".to_string(),
        );
    }
    // 水位线为 0 = 拒绝一切连接/请求，等同于自我 DoS，按配置笔误 fast-fail。
    // EN: A watermark of 0 rejects every connection/request, tantamount to self-DoS;
    // treat it as a config typo and fast-fail.
    if cfg.max_pgwire_connections == 0 {
        return Err("max_pgwire_connections must be > 0".to_string());
    }
    if cfg.max_http_inflight == 0 {
        return Err("max_http_inflight must be > 0".to_string());
    }
    // 标量压实阈值必须是有限 0..=1；interval=0 是合法的「关闭自动压实」。
    // EN: The stale ratio must be finite and in 0..=1; interval=0 legally
    // disables auto-compaction.
    if !cfg.collections_scalar_compact_stale_ratio.is_finite()
        || !(0.0..=1.0).contains(&cfg.collections_scalar_compact_stale_ratio)
    {
        return Err(format!(
            "collections_scalar_compact_stale_ratio must be finite and in 0..=1, got {}",
            cfg.collections_scalar_compact_stale_ratio
        ));
    }
    // 段太小会让轮转/保留开销压过收益；0 = 只手动轮转是合法值。
    // EN: Tiny segments cost more than they save; 0 = manual rotation only is valid.
    if cfg.collections_wal_segment_bytes != 0 && cfg.collections_wal_segment_bytes < 1024 * 1024 {
        return Err(format!(
            "collections_wal_segment_bytes must be 0 or >= 1 MiB, got {}",
            cfg.collections_wal_segment_bytes
        ));
    }
    // 复制（P4.4.1）组合校验：缺对端地址是典型部署笔误，fast-fail 优于
    // 启动后空跑。队列 0 = 每 follower 第一条 publish 即踢，等同自我 DoS。
    // EN: Replication combo validation — missing peer addresses and a zero
    // queue are config typos that must fail fast, not degrade at runtime.
    use houyi_server::types::ReplicationRole;
    match cfg.role {
        ReplicationRole::Disabled => {}
        ReplicationRole::Leader => {
            if cfg.replication_listen_addr.trim().is_empty() {
                return Err(
                    "replication_listen_addr must not be empty when role=leader".to_string()
                );
            }
        }
        ReplicationRole::Follower => {
            if cfg.replication_leader_addr.is_none() {
                return Err(
                    "--replication-leader-addr (or HOUYI_SERVER_REPLICATION_LEADER_ADDR) is required when role=follower"
                        .to_string(),
                );
            }
            // promote 会在 follower 上原地 bind listen_addr 起 leader 监听——
            // 空/畸形地址必须启动期 fast-fail，而不是 failover 中途才炸。
            // EN: promote binds listen_addr in place on the follower — an
            // empty/malformed address must fail at boot, not mid-failover.
            if cfg.replication_listen_addr.trim().is_empty() {
                return Err(
                    "replication_listen_addr must not be empty when role=follower (promote binds it)"
                        .to_string(),
                );
            }
            if cfg
                .replication_listen_addr
                .parse::<std::net::SocketAddr>()
                .is_err()
            {
                return Err(format!(
                    "replication_listen_addr must be a valid socket address, got {:?}",
                    cfg.replication_listen_addr
                ));
            }
        }
    }
    if cfg.role != ReplicationRole::Disabled {
        if cfg.replication_queue_batches == 0 {
            return Err("replication_queue_batches must be > 0".to_string());
        }
        // 复制依赖异步 WAL 原语（ShipHub 扇出 / 显式 LSN 追加 / follower
        // group-commit fsync），sync 模式没有对应通道。
        // EN: Replication needs the async WAL (ship fan-out, explicit-LSN
        // append, follower fsync barrier); sync mode has no such path.
        if cfg.wal_write_mode != WalWriteMode::Async {
            return Err("replication requires wal_write_mode=async".to_string());
        }
        // 0 超时 = 每笔 sync 写立即静默降级——与「配了 HA」的运维预期相反，
        // 按配置笔误 fast-fail（P4.4.2）。
        // EN: A zero timeout would silently degrade every sync write — treat
        // it as a config typo, not a feature.
        if cfg.replication_sync_timeout_ms == 0 {
            return Err("replication_sync_timeout_ms must be > 0".to_string());
        }
    }
    // 只给 id 不给 keyring 是部署笔误：运维以为加密已开，实际什么都没发生。
    // EN: a key id without a keyring is a typo that must not silently no-op.
    if cfg.encryption_key_id.is_some() && cfg.encryption_key_file.is_none() {
        return Err(
            "encryption_key_id is set but encryption_key_file is not; configure the \
             keyring or remove both"
                .to_string(),
        );
    }
    // 租约 TTL 上限（EE M1）：0 会让所有 grant 失败；超过硬上限会放大时钟
    // 偏斜风险。两者都按配置笔误 fast-fail。
    // EN: lease TTL cap (EE M1): 0 rejects every grant; above the hard cap it
    // widens clock-skew exposure. Both are config typos.
    if cfg.leader_lease_max_ttl_ms == 0
        || cfg.leader_lease_max_ttl_ms > houyi_server::cluster::MAX_LEASE_TTL_MS
    {
        return Err(format!(
            "leader_lease_max_ttl_ms must be in 1..={}ms, got {}",
            houyi_server::cluster::MAX_LEASE_TTL_MS,
            cfg.leader_lease_max_ttl_ms
        ));
    }
    if cfg.query_log_sample_every == 0 {
        return Err("query_log_sample_every must be > 0 (1 = record every query)".to_string());
    }
    // 只配置 sentinel token 会把整个数据面锁死（策略非空 → 所有端点要认证，
    // 而没有任何数据面主体）——运维本意几乎不可能是这样，fast-fail。
    // EN: a lone sentinel token would lock the whole data plane (non-empty
    // policy requires auth, but no data-plane principal exists) — fail fast.
    if cfg.sentinel_token.is_some() && cfg.auth_token.is_none() && cfg.auth_principals.is_empty() {
        return Err(
            "sentinel_token requires auth_token or [[auth_principals]]; otherwise the \
             data plane has no principal and every request is rejected"
                .to_string(),
        );
    }
    Ok(())
}

/// 加载静态加密句柄（P4.7.3）。keyring 不可读/格式错/活动键不存在一律启动
/// 失败——静默降级为明文会让「已开启加密」的错觉持续到第一次事故。
/// EN: load the encryption handle. Any keyring problem is fatal; silently
/// falling back to plaintext would invalidate the operator's assumption.
fn load_encryption(
    cfg: &ServerConfig,
) -> std::io::Result<Option<Arc<houyi_persistence::encryption::Encryption>>> {
    let Some(path) = cfg.encryption_key_file.as_deref() else {
        return Ok(None);
    };
    // keyring 是密钥本体所在，权限卫生比含 token 的 TOML 更严格。
    // EN: the keyring holds the keys themselves — same hygiene check, louder.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mode = meta.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                tracing::warn!(
                    path = %path.display(),
                    mode = format!("{mode:o}"),
                    "encryption keyring is group/world-accessible; chmod 600 is \
                     strongly recommended"
                );
            }
        }
    }
    let enc = houyi_persistence::encryption::Encryption::from_key_file(
        path,
        cfg.encryption_key_id.as_deref(),
    )
    .map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!(
                "load encryption keyring {} (key id {:?}): {e}",
                path.display(),
                cfg.encryption_key_id
            ),
        )
    })?;
    tracing::info!(active_key = enc.active_id(), "encryption at rest enabled");
    Ok(Some(Arc::new(enc)))
}

#[allow(dead_code)]
async fn run_pgwire_listener(
    addr: &str,
    pg: Option<Arc<PgWireSharedState>>,
) -> std::io::Result<SocketAddr> {
    run_pgwire_listener_with_metrics(addr, pg, Arc::new(PgWireMetrics::default())).await
}

async fn run_pgwire_listener_with_metrics(
    addr: &str,
    pg: Option<Arc<PgWireSharedState>>,
    metrics: Arc<PgWireMetrics>,
) -> std::io::Result<SocketAddr> {
    // 兼容旧签名（测试用）：永不触发的关停信号，不启用认证与 TLS。
    // EN: Legacy signature for tests — a shutdown future that never fires, no auth, no TLS.
    run_pgwire_listener_with_metrics_and_shutdown(
        addr,
        pg,
        metrics,
        None,
        None,
        std::future::pending(),
    )
    .await
}

/// 带关停信号的 pgwire 监听（P4.0.3）：信号触发后停止 accept，让存量连接任务
/// 随进程退出自然收尾（pgwire 为长连接协议，主动 drain 会话超出本子任务范围）。
/// `authz`（P4.2.1 认证 → P4.7.1 RBAC）每连接共享；握手质询明文密码并解析主体。
/// EN: Shutdown-aware pgwire listener: on signal, stop accepting; in-flight
/// sessions end with process exit (draining long-lived pg sessions is out of scope).
async fn run_pgwire_listener_with_metrics_and_shutdown<F>(
    addr: &str,
    pg: Option<Arc<PgWireSharedState>>,
    metrics: Arc<PgWireMetrics>,
    authz: Option<Arc<houyi_server::rbac::AuthzPolicy>>,
    tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
    shutdown: F,
) -> std::io::Result<SocketAddr>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    tokio::spawn(async move {
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                _ = &mut shutdown => {
                    tracing::info!("pgwire listener: shutdown signal, stop accepting");
                    break;
                }
                accepted = listener.accept() => {
                    match accepted {
                        Ok((stream, _)) => {
                            let pg_conn = pg.clone();
                            let metrics_conn = Arc::clone(&metrics);
                            let authz_conn = authz.clone();
                            let tls_conn = tls_acceptor.clone();
                            tokio::spawn(async move {
                                // TLS 协商（可选升级）+ 认证握手统一在入口内完成；
                                // 失败的连接（协商/握手 Err）直接丢弃。握手成功
                                // 返回已认证主体，供每条语句授权与审计使用。
                                // EN: the handshake returns the authenticated
                                // principal used for per-statement authorization.
                                if let Ok((mut session, principal)) = pgwire_accept_session(
                                    stream,
                                    metrics_conn.as_ref(),
                                    authz_conn.as_deref(),
                                    tls_conn.as_ref(),
                                )
                                .await
                                {
                                    // 连接水位线（P4.2.3）：握手后裁决，超进发
                                    // FATAL 53300；许可随本任务结束自动归还。
                                    // EN: Connection watermark (P4.2.3): decided after the
                                    // handshake; over-limit emits FATAL 53300; the permit
                                    // returns automatically when this task ends.
                                    if let Some(_permit) =
                                        pgwire_connection_admit(&mut session, &metrics_conn).await
                                    {
                                        let _ = handle_pgwire_simple_query_loop(
                                            &mut session,
                                            pg_conn,
                                            metrics_conn,
                                            principal,
                                        )
                                        .await;
                                    }
                                }
                            });
                        }
                        Err(_) => break,
                    }
                }
            }
        }
    });
    Ok(local_addr)
}

// 恢复相关函数已迁移到 houyi_server::recovery。
// EN: recovery functions moved to houyi_server::recovery.

/// 安装 tracing 订阅器（P4.0 可观测性地基）。
/// Install the tracing subscriber (P4.0 observability foundation).
///
/// - 分级：`RUST_LOG`（EnvFilter），未设置时默认 `info`。
/// - 输出：stdout（ANSI）恒开；`HOUYI_LOG_DIR=<dir>` 时再追加 tracing-appender
///   日轮转文件（`<dir>/houyidb.log.YYYY-MM-DD`，非阻塞写入）。
/// - 库 crate 只发事件，只有本二进制安装订阅器；返回的 guard 必须活到进程退出，
///   否则文件缓冲丢尾部日志。
fn init_tracing() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::prelude::*;
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let stdout_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stdout);
    let (file_layer, guard) = match std::env::var("HOUYI_LOG_DIR") {
        Ok(dir) if !dir.trim().is_empty() => {
            let appender = tracing_appender::rolling::daily(dir.trim(), "houyidb.log");
            let (nb, guard) = tracing_appender::non_blocking(appender);
            (
                Some(
                    tracing_subscriber::fmt::layer()
                        .with_ansi(false)
                        .with_writer(nb),
                ),
                Some(guard),
            )
        }
        _ => (None, None),
    };
    tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer)
        .with(file_layer)
        .init();
    guard
}

/// 安装 panic hook（P4.0.2）：panic 统一经 tracing 报错（含源码位置与 payload），
/// 再链回默认 hook 保留 stderr 原文与 RUST_BACKTRACE 栈。
/// Install a panic hook that reports via tracing (location + payload), then chains
/// to the default hook to keep the stderr original and RUST_BACKTRACE stack.
///
/// WHY：tokio 按 task 捕获 panic（不断言进程），默认只往 stderr 打一行——后台
/// builder/WAL 线程 panic 在生产上会被终端噪音淹没、文件日志完全漏记。入库后
/// panic 成为可检索事件（RUST_LOG / 日志文件一致可见）。
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("<non-string payload>");
        // hook 全局生效（主线程 panic 同样经过），消息保持中性并带线程名，
        // 避免把主线程 panic 误标为 worker 线程误导排障。
        // EN: The hook is process-global — keep the message neutral and include
        // the thread name so main-thread panics aren't mislabeled as worker ones.
        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("<unnamed>");
        tracing::error!(
            panic.location = %location,
            panic.payload = %payload,
            panic.thread = %thread_name,
            "panic hook caught panic"
        );
        default_hook(info);
    }));
}

/// 等待关停信号（P4.0.3）：Unix 下 SIGTERM 与 SIGINT(ctrl_c) 等值对待；
/// 非 Unix 仅 ctrl_c。返回信号名供关停日志归因。处理器注册失败以 io::Error
/// 传播（main 返回 Result，无需在生产路径 panic）。
/// Await a shutdown signal: SIGTERM and SIGINT on Unix, ctrl_c elsewhere.
/// Handler-registration failures propagate instead of panicking.
/// 读取或生成稳定的 follower 身份（P5.0.4）：人工重同步前保持不变，
/// leader 侧的 retention grace 才能跨重连识别同一副本。
/// EN: Loads or generates the stable follower identity (P5.0.4); it survives
/// reconnects until a manual resync deletes the file.
fn load_or_create_follower_id(wal_path: &std::path::Path) -> std::io::Result<[u8; 16]> {
    use rand::RngCore;
    use std::io::Write as _;

    let dir = wal_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let path = dir.join("replication.follower_id");
    match std::fs::read(&path) {
        Ok(bytes) => {
            let id: [u8; 16] = bytes.as_slice().try_into().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{} must contain exactly 16 bytes, found {}; delete the file to \
                         issue a fresh follower identity (this orphans the leader's grace \
                         entry for the old id and may force a full resync)",
                        path.display(),
                        bytes.len()
                    ),
                )
            })?;
            return Ok(id);
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let mut id = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut id);
    let tmp = dir.join("replication.follower_id.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&id)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    // 目录项也要落盘：崩溃后 follower_id 丢失会换身份，旧 grace 记录失联。
    // EN: fsync the directory too: losing the id across a crash would orphan the
    // old grace entry.
    #[cfg(unix)]
    {
        let dir_handle = std::fs::File::open(dir)?;
        dir_handle.sync_all()?;
    }
    Ok(id)
}

async fn shutdown_signal() -> std::io::Result<&'static str> {
    #[cfg(unix)]
    {
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            r = tokio::signal::ctrl_c() => {
                r?;
                Ok("SIGINT")
            }
            _ = sigterm.recv() => Ok("SIGTERM"),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await?;
        Ok("SIGINT")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // help 先于一切（含日志初始化与配置加载）：坏的 HOUYI_CONFIG 不应挡住
    // `-h`——求助路径必须永远可达，哪怕环境完全配错。argv 预扫描是唯一的
    // help 入口（parse_config_from_cli 不再承接，避免两条 help 路径发散）。
    // 已知取舍：`-h` 出现在别的旗标值位（如 `--tls-cert -h`）也会打 help——
    // 求助永远赢，比"参数笔误报 unknown"更符合求助语义。
    // EN: help precedes everything, including tracing init and config
    // loading — the way to ask for help stays reachable under any misconfig.
    // This argv pre-scan is the single help entry; `-h` in a value position
    // also prints help (asking for help always wins).
    if std::env::args().skip(1).any(|a| a == "-h" || a == "--help") {
        print_help();
        std::process::exit(0);
    }
    // 文件日志的非阻塞 guard：持有到 main 返回，进程退出前冲刷尾部缓冲。
    // EN: Keep the non-blocking file guard alive until exit so buffered logs flush.
    let _log_guard = init_tracing();
    install_panic_hook();
    // 合并链（P4.3.1）：default → TOML(--config/HOUYI_CONFIG) → ENV → CLI 旗标。
    // EN: Merge chain: default → TOML → ENV → CLI flags.
    let file_cfg = load_toml_config_base().map_err(std::io::Error::other)?;
    let env_cfg = parse_config_from_env(file_cfg).map_err(std::io::Error::other)?;
    let cfg = parse_config_from_cli(env_cfg).map_err(std::io::Error::other)?;
    let bench_build = bench_build_config(BuildConfig::from_env());

    tracing::info!("HouyiDB v{} starting...", env!("CARGO_PKG_VERSION"));
    tracing::info!("Compliance check: cursor rules active.");
    let hnsw_ef_display = cfg
        .hnsw_ef_search
        .map(|e| e.to_string())
        .unwrap_or_else(|| "inferred".to_string());
    let wal_mode_s = match cfg.wal_write_mode {
        WalWriteMode::Async => "async",
        WalWriteMode::Sync => "sync",
    };
    tracing::info!(
        worker_threshold = cfg.worker_threshold,
        worker_tick_ms = cfg.worker_tick_ms,
        policy = ?cfg.flush_policy,
        metrics_interval_secs = cfg.metrics_interval_secs,
        hnsw_ef_search = %hnsw_ef_display,
        vector_dim = cfg.vector_dim,
        benchmark_http_addr = %cfg.benchmark_http_addr,
        pgwire_addr = %cfg.pgwire_addr,
        wal_path = %cfg.wal_path.display(),
        wal_write_mode = wal_mode_s,
        replication_role = cfg.role.as_str(),
        "config"
    );
    tracing::info!(
        portable = FEATURE_PORTABLE,
        simd_avx512 = FEATURE_SIMD_AVX512,
        cxl_memory = FEATURE_CXL_MEMORY,
        ha_raft = FEATURE_HA_RAFT,
        "features"
    );

    // 1) 初始化 HouyiBus: Region 池 + NUMA 感知分配。
    // EN: 1) Initialize HouyiBus: Region pool + NUMA-aware allocation.
    let numa_info = detect_numa_topology();
    tracing::info!(
        numa_nodes = numa_info.numa_nodes,
        source = %numa_info.source,
        "houyi-core topology"
    );
    let region_pool = RegionPool::new(NumaPolicy::Interleaved, 1 << 20, numa_info.numa_nodes);
    let alloc = region_pool
        .allocate(1 << 20)
        .ok_or_else(|| std::io::Error::other("region allocation failed"))?;
    let page = HouyiPage::new_in_region(1 << 20, alloc);
    tracing::info!(
        capacity_bytes = page.capacity(),
        numa_node = ?page.numa_node(),
        "houyi-core ready: aligned page"
    );
    for usage in region_pool.usage_snapshot() {
        tracing::info!(
            region = usage.region_id,
            numa_node = usage.numa_node,
            used = usage.used,
            capacity = usage.capacity,
            "houyi-core region usage"
        );
    }

    // 静态加密（P4.7.3）：必须先于任何 WAL/快照打开——旧明文历史在打开时
    // 一次性迁移，此后所有写入与快照都走密文。
    // EN: load before opening anything — legacy plaintext history is migrated on
    // open, and every later write/snapshot stays encrypted.
    let encryption = load_encryption(&cfg)?;

    // 2) 向量 WAL：打开并读取快照；HTTP upsert 成功后将 JSON 记录追加到同一文件。
    // EN: 2) Vector WAL: open it and read the snapshot; after a successful HTTP
    // upsert, append the JSON record to the same file.
    let applied_sidecar = sidecar_path_for_wal(&cfg.wal_path);
    let applied_lsn_before = read_applied_lsn(&applied_sidecar)?;
    let (wal, wal_entries) = WalHandle::open_with_ship_and_encryption(
        &cfg.wal_path,
        cfg.wal_write_mode,
        None,
        encryption.clone(),
    )?;
    let wal_max_lsn = wal_entries.last().map(|e| e.lsn).unwrap_or(0);
    tracing::info!(
        wal_path = %cfg.wal_path.display(),
        replayed_entries = wal_entries.len(),
        next_lsn = wal.next_lsn(),
        applied_lsn_sidecar = applied_lsn_before,
        wal_max_lsn,
        "houyi-persistence ready"
    );
    if applied_lsn_before > wal_max_lsn && wal_max_lsn > 0 {
        tracing::warn!(
            applied_lsn_before,
            wal_max_lsn,
            "houyi-persistence: applied_lsn sidecar ahead of wal max; possible truncated WAL"
        );
    }

    // 3) 初始化查询与向量子系统占位。
    // EN: 3) Initialize query and vector subsystem placeholders.
    let plan = QueryPlanner::plan(PgWireQuery {
        sql: "SELECT id FROM docs WHERE tag='p0'".to_string(),
    });
    tracing::info!(?plan, "houyi-query ready");

    let index = if cfg.index_type == "clustered" {
        let idx = IndexBackend::new_clustered(cfg.vector_dim);
        tracing::info!("houyi-vector: ClusteredHnsw backend");
        idx
    } else if cfg.index_type == "flat-clustered" {
        let idx = IndexBackend::new_flat_clustered(cfg.vector_dim);
        tracing::info!("houyi-vector: FlatClustered backend (K-Means + SIMD brute-force)");
        idx
    } else if cfg.index_type == "flat" {
        let idx = IndexBackend::new_flat(cfg.vector_dim);
        tracing::info!("houyi-vector: Flat backend (pure SIMD brute-force, recall=1.0)");
        idx
    } else if cfg.index_type == "pq-flat" {
        let idx = IndexBackend::new_pq_flat(cfg.vector_dim);
        tracing::info!("houyi-vector: PqFlat backend (PQ-compressed, ~16MB for 1M×128D)");
        idx
    } else if cfg.index_type == "hii-l0" {
        let idx = IndexBackend::new_hii_l0(cfg.vector_dim);
        tracing::info!(
            "houyi-vector: HII L0 backend (memtable + small HNSW M=4, real-time writes)"
        );
        tracing::info!("houyi-vector: HII L1/L2 TODO — pending GPU/NVMe hardware");
        idx
    } else {
        // 将 layer_m 乘数注入 HNSW 全局缓存（进程启动一次，热路径无开销）。
        // 默认 2×M（召回优先，兼容旧图快照）；高 QPS 场景显式设
        // HOUYI_HNSW_LAYER0_MULT=1。
        // Push layer degree multipliers once at startup (hot-path zero-cost).
        // Default 2×M (recall-first, snapshot-compatible); set =1 for higher QPS.
        let hnsw_cfg = houyi_core::config::HnswConfig::from_env();
        houyi_vector::set_layer_multipliers(hnsw_cfg.layer0_mult, hnsw_cfg.upper_mult);

        let mut hns = HotBufferIndex::new_with_metric(cfg.vector_dim, DistanceMetric::Cosine);
        hns.set_build_config(bench_build);
        // pgwire 构建参数在启动时即生效，而不是等首批 upsert：`/v1/index/status`
        // 从启动起就报告真实生效值，基准客户端可在写入前核对。
        // EN: Apply pgwire build parameters at startup instead of on the first
        // upsert so /v1/index/status reports the effective values immediately.
        if let Some(m) = cfg.pgwire_hnsw_m {
            hns.set_hnsw_m(m).map_err(std::io::Error::other)?;
        }
        if let Some(efc) = cfg.pgwire_hnsw_ef_construction {
            hns.set_hnsw_ef_construction(efc)
                .map_err(std::io::Error::other)?;
        }
        if let Some(ef) = cfg.hnsw_ef_search {
            hns.set_hnsw_ef_search(ef).map_err(std::io::Error::other)?;
        }
        tracing::info!(
            m = hns.hnsw_m(),
            ef_construction = hns.hnsw_ef_construction_value(),
            ef_search = hns.hnsw_ef_search(),
            layer0_mult = hnsw_cfg.layer0_mult,
            upper_mult = hnsw_cfg.upper_mult,
            "houyi-vector: hnsw effective parameters"
        );
        IndexBackend::Hnsw(Arc::new(hns))
    };
    // P4.4.0：图快照优先（二进制 HNSW，免重建图）；缺席/损坏回退旧 JSON 向量快照，
    // 再退化为全量 WAL 重放。图快照的 checkpoint 与 JSON 快照语义相同（主 WAL LSN 空间）。
    // EN: Graph snapshot first (binary HNSW, no graph rebuild); fall back to the
    // legacy JSON vector snapshot, then to full WAL replay.
    let graph_snapshot_metrics = Arc::new(GraphSnapshotMetrics::default());
    // P5.0.3：快照锁/段上限/保留租约与计数器在 HTTP、周期任务之间共享。
    // EN: P5.0.3 snapshot lock, segment cap, retention lease and counters are
    // shared between HTTP and the periodic task.
    let lifecycle = Arc::new(
        houyi_server::collection_lifecycle::CollectionLifecycleState::new(
            cfg.collections_wal_max_segments,
            houyi_server::collection_lifecycle::DEFAULT_MAX_DELETES_PER_RUN,
        ),
    );
    let snapshot_checkpoint = match apply_graph_snapshot_if_present_with_encryption(
        &cfg.wal_path,
        &index,
        &graph_snapshot_metrics,
        encryption.as_deref(),
    )
    .await?
    {
        Some(cp) => Some(cp),
        None => {
            // 期望度量必须取索引实际值：pq-flat 如实在接口上报告 L2，硬编码
            // Cosine 会让它自己写出的新快照在下次启动时被判为度量不符。
            // EN: The expected metric must come from the live index: pq-flat now
            // truthfully reports L2, so hard-coding Cosine would reject the very
            // snapshots the new binary writes.
            apply_vector_snapshot_if_present_with_encryption(
                &cfg.wal_path,
                &index,
                cfg.vector_dim,
                index.distance_metric(),
                encryption.as_deref(),
            )
            .await?
        }
    };
    let replay_n =
        replay_vector_wal_into_index(&wal_entries, &index, cfg.vector_dim, snapshot_checkpoint)
            .await?;
    tracing::info!(
        batches = replay_n,
        min_lsn_exclusive = ?snapshot_checkpoint,
        "houyi-vector: vector WAL replay applied from HTTP upsert log"
    );
    drain_pending_builder(&index).await;
    let snapshot_cp = snapshot_checkpoint.unwrap_or(0);
    let applied_after_recovery = wal_max_lsn.max(snapshot_cp);
    write_applied_lsn_atomic(&applied_sidecar, applied_after_recovery)?;
    let worker = index.start_async_builder_worker(
        cfg.worker_threshold,
        cfg.worker_tick_ms,
        cfg.flush_policy,
    );
    let wal = Arc::new(wal);
    let pgwire_metrics = {
        // P4.2.3：连接水位线上限随配置注入（默认 200，见 types::DEFAULT_MAX_PGWIRE_CONNECTIONS）。
        // P4.2.4：内存水位线同挂（0 = 禁用）。P4.3.2：慢查询阈值同挂（0 = 禁用）。
        // EN: P4.2.3: connection watermark injected from config (default 200, see
        // types::DEFAULT_MAX_PGWIRE_CONNECTIONS). P4.2.4: memory watermark likewise
        // (0 = disabled). P4.3.2: slow-query threshold likewise (0 = disabled).
        Arc::new(PgWireMetrics {
            connection_watermark: Arc::new(InflightWatermark::new(
                cfg.max_pgwire_connections as u64,
            )),
            memory_watermark: Arc::new(houyi_server::mem::MemoryWatermark::new(
                cfg.max_memory_bytes,
            )),
            slow_query_ms: std::sync::atomic::AtomicU64::new(cfg.slow_query_ms),
            ..Default::default()
        })
    };
    let obs_index = index.clone();
    let obs_pgwire_metrics = Arc::clone(&pgwire_metrics);
    let observer = tokio::spawn(async move {
        let mut ticker = time::interval(Duration::from_secs(cfg.metrics_interval_secs));
        loop {
            ticker.tick().await;
            let pending = obs_index.pending_build_count().await;
            let main = obs_index.indexed_count().await;
            let hot = obs_index.hot_buffer_count().await;
            let graph = obs_index.graph_node_count().await;
            let metrics = obs_index.builder_metrics();
            tracing::info!(
                hot,
                main,
                graph,
                pending,
                flushes = metrics.flush_count,
                total_flushed = metrics.total_flushed_items,
                max_pending = metrics.max_pending_queue,
                last_flush = metrics.last_flush_items,
                "[metrics]"
            );
            let pg = obs_pgwire_metrics.snapshot();
            tracing::info!(
                handshakes = pg.handshake_count,
                frontend = pg.frontend_messages,
                decode_errors = pg.decode_errors,
                auth_failures = pg.auth_failures,
                query = pg.simple_query_count,
                parse = pg.parse_count,
                bind = pg.bind_count,
                execute = pg.execute_count,
                describe = pg.describe_count,
                close = pg.close_count,
                sync = pg.sync_count,
                flush = pg.flush_count,
                copy = pg.copy_count,
                terminate = pg.terminate_count,
                unsupported = pg.unsupported_count,
                sql_text_bytes = pg.sql_text_bytes,
                bind_param_text_bytes = pg.bind_param_text_bytes,
                payload_bytes = pg.frontend_payload_bytes,
                "[pgwire]"
            );
        }
    });
    // 自检探针使用保留 ID，避免重启时覆盖 WAL 重放出的真实 0/1/2；自检结束后
    // 删除，保证主索引从空态开始（VDB 客户端的 drop_old 检查依赖这一点）。
    // EN: Self-check probes use reserved IDs so a restart cannot overwrite real
    // rows at 0/1/2; they are removed afterwards so the main index starts empty
    // (the VectorDBBench client's drop_old enforcement relies on it).
    let sample_vector = vec![0.1_f32; cfg.vector_dim];
    let sample_ids = [u64::MAX - 2, u64::MAX - 1, u64::MAX];
    for id in sample_ids {
        index.insert(id, sample_vector.clone()).await?;
    }
    let query_ids = execute_query(
        &index,
        &format!("SELECT id FROM docs WHERE id IN ({})", sample_ids[2]),
        &sample_vector,
        5,
        None,
    )
    .await?;
    tracing::info!(?query_ids, "houyi-vector ready: query_filtered_ids");

    let mut page = HouyiPage::new(4096);
    for tag in ["p0", "p1", "p1"] {
        let mut scalars = std::collections::HashMap::new();
        scalars.insert("tag".to_string(), ScalarValue::Text(tag.to_string()));
        page.insert_row(scalars, None);
    }
    let hybrid_ids = execute_hybrid_with_scalar_bitset(
        &index,
        &page,
        "tag",
        ScalarPredicate::Eq(ScalarValue::Text("p1".to_string())),
        &[0.1, 0.2, 0.3, 0.4],
        10,
    )
    .await;
    tracing::info!(?hybrid_ids, "houyi-vector ready: scalar_bitset_hybrid_ids");
    for id in sample_ids {
        match index.delete(id).await {
            Ok(_) => {}
            // clustered / pq-flat / hii-l0 当前不支持 delete；保留探针不影响
            // 真实数据（保留 ID 不会与数据集冲突），但该后端无法从空态验收。
            // EN: Those backends do not support delete today; the reserved-ID
            // probes cannot collide with dataset rows, but such a backend cannot
            // pass an empty-index check.
            Err(err) => tracing::warn!(id, %err, "houyi-vector ready: probe cleanup skipped"),
        }
    }

    // Collection HNSW 后端与主索引同源的异步构图配置：缺了它 Collection 写入永不
    // 可检索（insert_batch 只进待构图队列）。
    // EN: Same builder knobs as the main index — without them collection HNSW writes
    // stay invisible (insert_batch only enqueues).
    let collection_builder_cfg = CollectionBuilderConfig {
        threshold: cfg.worker_threshold,
        tick_ms: cfg.worker_tick_ms,
        policy: cfg.flush_policy,
    };

    // Collections：HTTP/pgwire 共享的同一 Manager + 独立 WAL（独立于主 WAL 的理由见
    // houyi_persistence::collection_wal 模块文档——主 WAL 的 checkpoint 会清掉无快照
    // 的 collection 记录）。先重放再起监听，重启前写入的 collection 全量恢复。
    // EN: A single CollectionManager shared by HTTP and pgwire, plus a dedicated WAL
    // (see houyi_persistence::collection_wal for why it must not share the main WAL —
    // its checkpoint would erase unsnapshotted collection records). Replay BEFORE
    // the listeners start so pre-restart collections come back in full.
    let collections_wal_path =
        houyi_persistence::collection_wal::collections_wal_path_for(&cfg.wal_path);
    // 复制扇出枢纽（P4.4.1）：role≠disabled 时装配——leader 用于 publish；
    // follower 同样装 hub（其 AppendEntry 批次也会 publish，级联 v1 不支持，
    // 但 P4.4.3 promote 后 hub 直接可用、WAL 句柄零切换）。hub 必须先于
    // WAL open 创建并注入写线程。
    // EN: The ship hub exists whenever replication is on — followers get one
    // too so promote (P4.4.3) needs no handle swap. It must be created BEFORE
    // opening the WAL so the writer thread owns it.
    let ship_hub = if cfg.role != houyi_server::types::ReplicationRole::Disabled {
        Some(std::sync::Arc::new(
            houyi_persistence::replication_leader::ShipHub::new(cfg.replication_queue_batches),
        ))
    } else {
        None
    };
    let (collections_wal, collections_wal_entries) = WalHandle::open_with_config(
        &collections_wal_path,
        cfg.wal_write_mode,
        ship_hub.clone(),
        encryption.clone(),
        houyi_persistence::wal_async::WalOpenConfig {
            segment_bytes: cfg.collections_wal_segment_bytes,
        },
    )?;
    let collections = std::sync::Arc::new(
        CollectionManager::new(cfg.vector_dim, &cfg.index_type)
            .with_builder_config(collection_builder_cfg)
            .with_default_ha_level(cfg.default_ha_level),
    );
    // P5.0.2：优先恢复 collection / manifest 快照；没有 manifest 时回退 P4.4 的
    // 逐 collection 图快照。`first_available > 0` 且 manifest 不可用时，加载器
    // 直接报错——截断历史没有快照无法安全恢复。
    // EN: Prefer collection/manifest snapshots; fall back to P4.4 graph snapshots
    // when no manifest exists. With first_available > 0, the loader fails on an
    // unusable manifest — truncated history cannot be safely replayed without one.
    let collections_first_available = collections_wal.first_available_lsn().unwrap_or(0);
    let loaded = houyi_server::collection_lifecycle::load_collection_snapshots(
        &collections,
        &collections_wal_path,
        collections_first_available,
        encryption.clone(),
        &graph_snapshot_metrics,
    )
    .await?;
    let (graph_checkpoints, manifest_mode, manifest_safe_checkpoint) = match loaded {
        Some(loaded) => (
            loaded.checkpoints,
            true,
            Some(loaded.manifest.safe_checkpoint_lsn),
        ),
        None => (
            apply_collection_graph_snapshots_with_encryption(
                &collections,
                &collections_wal_path,
                &graph_snapshot_metrics,
                encryption.as_deref(),
            )
            .await?,
            false,
            None,
        ),
    };
    let collections_stats = houyi_server::recovery::replay_collections_wal_into_manager_with_mode(
        &collections_wal_entries,
        &collections,
        &graph_checkpoints,
        manifest_mode,
    )
    .await?;
    // 缓存 manifest checkpoint：follower 握手只读原子量，不做磁盘 IO。
    // EN: Cache the manifest checkpoint so follower handshakes never do disk IO.
    lifecycle.set_manifest_checkpoint(manifest_safe_checkpoint.unwrap_or(0));
    tracing::info!(
        path = %collections_wal_path.display(),
        first_available_lsn = collections_first_available,
        manifest_mode,
        manifest_safe_checkpoint = ?manifest_safe_checkpoint,
        creates = collections_stats.creates,
        upsert_batches = collections_stats.upsert_batches,
        upsert_batches_skipped_snapshot = collections_stats.upsert_batches_skipped_snapshot,
        upsert_rows = collections_stats.upsert_rows,
        scalar_rows = collections_stats.scalar_rows,
        delete_batches = collections_stats.delete_batches,
        delete_batches_skipped_snapshot = collections_stats.delete_batches_skipped_snapshot,
        tombstones = collections_stats.tombstones,
        records_skipped_snapshot = collections_stats.records_skipped_snapshot,
        graph_snapshots = graph_checkpoints.len(),
        "houyi-persistence: collections WAL replay"
    );
    let collections_wal = Arc::new(collections_wal);

    // ── 复制装配（P4.4.1）────────────────────────────────────
    // 重放完成后启动：resume_lsn = 本地 next_lsn（重放已追平本地历史）。
    // 运行态角色进共享原子量（HTTP 只读拦截 / pgwire 25006 / 指标共用；
    // P4.4.3 promote 运行中切换）。leader/follower 句柄留到关停序列统一 shutdown。
    // EN: Start replication after replay (resume_lsn = local next_lsn). The
    // runtime role lives in a shared atomic (read-only guards + metrics);
    // leader/follower handles are shut down in the teardown sequence below.
    let role_state = houyi_server::types::RoleState::new(cfg.role);
    // leader 启动后先留出 follower 重连窗口：期间不删除段，避免旧 follower
    // 还没握手就被截断。promote 路径在 P5.0.4 后续接线同样设置。
    // EN: Give followers a reconnect window right after leader startup: no
    // deletion until it expires, so old followers can handshake first. The
    // promote path will set the same hold in the follow-up P5.0.4 wiring.
    if role_state.get() == houyi_server::types::ReplicationRole::Leader
        && cfg.collections_wal_follower_grace_secs > 0
    {
        lifecycle.hold_retention(cfg.collections_wal_follower_grace_secs);
    }
    // 运维警示（OCR 批次十六/十八）：范围与安全边界必须在启动日志里可见——
    // 两条警示对 leader 与 follower 同样成立（监听端口暴露警示适用于任何
    // 可被 promote 的节点；「主索引不复制」的意外正是纯 follower 部署会撞上
    // 的），提升到 role match 之上，任何复制角色启动都打。
    // EN: Both boundary warnings apply to every replication-enabled node (a
    // pure follower is exactly the deployment that gets promoted), so they are
    // hoisted above the role match.
    if cfg.role != houyi_server::types::ReplicationRole::Disabled {
        tracing::warn!(
            "replication covers collections.wal only — the benchmark main index \
             (POST /v1/vectors/upsert, pgwire default index) is NOT replicated; \
             a promoted follower starts empty on that path"
        );
        tracing::warn!(
            "replication protocol has no authentication/TLS — any host reaching \
             the listen port can pull the full dataset or exhaust the leader \
             with fake follower sessions; network isolation is mandatory"
        );
    }
    // leader/follower 句柄共享槽（P4.4.3）：promote 端点与关停序列经同一槽位
    // 交接句柄——promote 后新 leader 句柄对关停序列可见，杜绝泄漏。
    // EN: One shared slot for the live replication handles so the promote
    // endpoint and the shutdown sequence never lose track of a handle.
    let replication_handles: houyi_server::types::SharedReplicationHandles = std::sync::Arc::new(
        std::sync::Mutex::new(houyi_server::types::ReplicationHandles::default()),
    );
    // follower 身份与快照 checkpoint 在 leader/follower/demote 三处共用：
    // 运行时 demote→follower 转换（M3）需要它们，而不只在启动为 follower 时。
    // EN: follower identity and the snapshot checkpoint are shared by leader,
    // follower and runtime demote conversion (M3), not just follower startup.
    let follower_id = if cfg.role == houyi_server::types::ReplicationRole::Disabled {
        None
    } else {
        Some(load_or_create_follower_id(&collections_wal_path)?)
    };
    let snapshot_checkpoint: std::sync::Arc<dyn Fn() -> u64 + Send + Sync> = {
        let lifecycle_for_cp = std::sync::Arc::clone(&lifecycle);
        std::sync::Arc::new(move || lifecycle_for_cp.manifest_checkpoint())
    };
    let replication_metrics: houyi_server::types::SharedReplicationMetrics = match cfg.role {
        houyi_server::types::ReplicationRole::Disabled => {
            houyi_server::types::shared_replication_metrics(None)
        }
        houyi_server::types::ReplicationRole::Leader => {
            let hub = ship_hub.clone().expect("ship hub assembled for leader");
            let watermarks =
                std::sync::Arc::new(houyi_persistence::replication_leader::AckWatermarks::new());
            let wal_for_next = Arc::clone(&collections_wal);
            let leader_next_lsn: std::sync::Arc<dyn Fn() -> u64 + Send + Sync> =
                std::sync::Arc::new(move || wal_for_next.next_lsn());
            let wal_for_first = Arc::clone(&collections_wal);
            let leader_first_available: std::sync::Arc<
                dyn Fn() -> std::io::Result<u64> + Send + Sync,
            > = std::sync::Arc::new(move || wal_for_first.first_available_lsn());
            let gate = lifecycle.retention_gate();
            let leader = houyi_persistence::replication_leader::ReplicationLeader::start(
                &cfg.replication_listen_addr,
                hub,
                collections_wal_path.clone(),
                std::sync::Arc::clone(&leader_next_lsn),
                std::sync::Arc::clone(&watermarks),
                houyi_persistence::replication_leader::LeaderSessionConfig::for_replication(
                    cfg.replication_queue_batches,
                    cfg.replication_auth_token.clone(),
                    encryption.clone(),
                )
                .with_retention(leader_first_available, gate),
            )
            .await?;
            tracing::info!(
                listen_addr = %cfg.replication_listen_addr,
                bound = ?leader.local_addr().ok(),
                "replication leader ready"
            );
            let view = houyi_server::types::ReplicationMetricsView::Leader {
                hub: ship_hub.clone().expect("ship hub assembled for leader"),
                watermarks,
                leader_next_lsn,
                sync_timeout: std::time::Duration::from_millis(cfg.replication_sync_timeout_ms),
                sync_quorum: cfg.replication_sync_quorum,
            };
            replication_handles.lock().unwrap().leader = Some(leader);
            houyi_server::types::shared_replication_metrics(Some(view))
        }
        houyi_server::types::ReplicationRole::Follower => {
            let leader_addr = cfg
                .replication_leader_addr
                .clone()
                .expect("leader addr validated");
            let wal_async = collections_wal
                .async_handle()
                .expect("async wal handle (config validated async mode)");
            let follower_metrics = std::sync::Arc::new(
                houyi_persistence::replication_follower::FollowerMetrics::default(),
            );
            let apply_sink =
                std::sync::Arc::new(houyi_server::replication_apply::CollectionsApplySink::new(
                    std::sync::Arc::clone(&collections),
                ));
            let poison_skipped = apply_sink.poison_skipped_handle();
            let follower_id = follower_id.expect("follower id for follower role");
            let follower = houyi_persistence::replication_follower::ReplicationFollower::start(
                leader_addr.clone(),
                wal_async,
                apply_sink,
                std::sync::Arc::clone(&follower_metrics),
                houyi_persistence::replication_follower::FollowerConfig::for_replication(
                    cfg.replication_auth_token.clone(),
                    encryption.clone(),
                )
                .with_follower_identity(follower_id, std::sync::Arc::clone(&snapshot_checkpoint)),
            )?;
            tracing::info!(
                leader_addr = %leader_addr,
                follower_id = %follower_id
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
                "replication follower ready (read-only)"
            );
            let view = houyi_server::types::ReplicationMetricsView::Follower {
                metrics: follower_metrics,
                poison_skipped,
            };
            replication_handles.lock().unwrap().follower = Some(follower);
            houyi_server::types::shared_replication_metrics(Some(view))
        }
    };

    // promote 上下文（P4.4.3）：role≠disabled 即装配——follower 靠它完成
    // promote；leader 起始角色同样装配（状态单点触达纪律），端点入口 409。
    // EN: Assemble the promote context whenever replication is on.
    let promote_ctx = if cfg.role != houyi_server::types::ReplicationRole::Disabled {
        Some(std::sync::Arc::new(houyi_server::types::PromoteContext {
            handles: std::sync::Arc::clone(&replication_handles),
            promote_gate: tokio::sync::Mutex::new(()),
            hub: ship_hub
                .clone()
                .expect("ship hub assembled when replication on"),
            listen_addr: cfg.replication_listen_addr.clone(),
            collections_wal_path: collections_wal_path.clone(),
            collections_wal: Arc::clone(&collections_wal),
            queue_batches: cfg.replication_queue_batches,
            sync_timeout: std::time::Duration::from_millis(cfg.replication_sync_timeout_ms),
            sync_quorum: cfg.replication_sync_quorum,
            lifecycle: std::sync::Arc::clone(&lifecycle),
            follower_grace_secs: cfg.collections_wal_follower_grace_secs,
            replication_auth_token: cfg.replication_auth_token.clone(),
            encryption: encryption.clone(),
            collections: std::sync::Arc::clone(&collections),
            follower_id: follower_id.expect("follower id when replication is on"),
            snapshot_checkpoint: std::sync::Arc::clone(&snapshot_checkpoint),
        }))
    } else {
        None
    };

    // OpenAI 兼容 embeddings 网关：key 仅经 ENV 注入（CLI 对 ps 可见，不接旗标）。
    // EN: OpenAI-compatible embeddings gateway: the key arrives only via ENV (CLI
    // arguments are visible to ps, so no flag is offered).
    let openai_upstream = cfg.openai_upstream.clone().map(|base_url| {
        let api_key = std::env::var("HOUYI_OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        Arc::new(houyi_server::types::OpenAIUpstreamConfig::new(
            base_url,
            api_key,
            Duration::from_secs(cfg.openai_timeout_secs),
        ))
    });
    if let Some(upstream) = &openai_upstream {
        tracing::info!(?upstream, "openai gateway");
    }

    // 集群控制原语（EE M1）：term 持久化在 WAL 目录的 cluster_state.json，
    // 损坏 fail-closed（启动失败优于带着未知 term 继续写）。租约默认关闭，
    // 关闭时所有写闸门直接放行，行为与手动 promote 一致。
    // EN: cluster primitives (EE M1): the term lives next to the WAL in
    // cluster_state.json and corruption is fatal. Leases are off by default;
    // while off, every write gate is a no-op, matching manual promote.
    let cluster = std::sync::Arc::new(
        houyi_server::cluster::ClusterControl::open_with_max_ttl(
            &cfg.wal_path,
            cfg.require_leader_lease,
            cfg.leader_lease_max_ttl_ms,
        )
        .map_err(|e| format!("cluster state store {}: {e}", cfg.wal_path.display()))?,
    );
    tracing::info!(
        term = cluster.term(),
        require_leader_lease = cfg.require_leader_lease,
        max_ttl_ms = cluster.max_ttl_ms(),
        "cluster control"
    );
    if cfg.require_leader_lease && cfg.sentinel_token.is_none() {
        tracing::warn!(
            "require_leader_lease is on without sentinel_token: leases must be granted \
             manually via POST /v1/admin/lease or the cluster stays read-only"
        );
    }

    // 查询日志（EE M1）：容量 0 = 关闭（None），写路径零开销。
    // EN: query log (EE M1); capacity 0 disables it and the write path stays free.
    let query_log: Option<std::sync::Arc<houyi_server::query_log::QueryLog>> =
        (cfg.query_log_capacity > 0).then(|| {
            std::sync::Arc::new(houyi_server::query_log::QueryLog::new(
                cfg.query_log_capacity,
                cfg.query_log_sample_every,
            ))
        });

    // 认证令牌（P4.2.1）：HTTP 与 pgwire 共享同一份 Arc<str>；只记录是否启用，
    // 永不记录令牌本体。
    // EN: Shared auth token handle — log only whether auth is on, never the token.
    // RBAC（P4.7.1）：legacy auth_token 作为全权 admin，配置主体追加；sentinel
    // 专用秘密（EE M1）作为独立控制面主体。
    // 校验失败 fast-fail（重复 token/非法角色/坏哈希都在启动期暴露）。
    // EN: build the authorization policy; invalid principals are fatal at startup.
    let auth_policy = houyi_server::rbac::AuthzPolicy::from_config_with_sentinel(
        cfg.auth_token.as_deref(),
        &cfg.auth_principals,
        cfg.sentinel_token.as_deref(),
    )
    .map_err(|e| format!("auth principals: {e}"))?;
    let authz: Option<Arc<houyi_server::rbac::AuthzPolicy>> =
        (!auth_policy.is_empty()).then(|| Arc::new(auth_policy));
    tracing::info!(
        auth_enabled = authz.is_some(),
        principals = authz.as_ref().map(|p| p.len()).unwrap_or(0),
        "auth config"
    );

    // TLS acceptor（P4.2.2）：证书/私钥成对已在配置校验保证；HTTP 与 pgwire
    // 共享同一份 rustls acceptor（单证书）。加载失败 fast-fail 优于半开监听。
    // EN: Shared rustls acceptor for HTTP and pgwire (single certificate);
    // pairing was validated during config parsing; load failure is fatal.
    let tls_acceptor: Option<tokio_rustls::TlsAcceptor> =
        match (&cfg.tls_cert_path, &cfg.tls_key_path) {
            (Some(cert), Some(key)) => Some(houyi_server::tls::load_tls_acceptor(
                std::path::Path::new(cert),
                std::path::Path::new(key),
            )?),
            _ => None,
        };
    tracing::info!(tls_enabled = tls_acceptor.is_some(), "tls config");

    // 审计日志（P4.7.0）：开启时启动即校验既有哈希链，链断拒绝启动。
    // EN: open the audit log and verify its hash chain before serving.
    let audit: Option<Arc<houyi_server::audit::AuditLog>> = match &cfg.audit_log_path {
        Some(path) => Some(Arc::new(houyi_server::audit::AuditLog::open(
            path.clone(),
            cfg.audit_include_reads,
            cfg.audit_max_bytes,
        )?)),
        None => None,
    };
    tracing::info!(
        audit_enabled = audit.is_some(),
        audit_path = ?cfg.audit_log_path,
        audit_include_reads = cfg.audit_include_reads,
        "audit config"
    );

    // HTTP 在途水位线（P4.2.3）：中间件占位 + /metrics 读数共用一份。
    // EN: Shared in-flight watermark for the middleware and /metrics.
    let http_inflight = Arc::new(InflightWatermark::new(cfg.max_http_inflight as u64));
    tracing::info!(
        max_http_inflight = cfg.max_http_inflight,
        max_pgwire_connections = cfg.max_pgwire_connections,
        slow_query_ms = cfg.slow_query_ms,
        "watermark config"
    );

    // 关停广播（P4.0.3）：watch 通道一处 send，HTTP 优雅关停与 pgwire accept 退出
    // 各自消费；已触发的值对新 watch 者立即可见（无 notify 丢信号竞态）。
    // EN: One watch channel fans the shutdown signal out to the HTTP graceful-drain
    // and the pgwire accept loop; a fired value is immediately visible to late
    // subscribers (no lost-signal race).
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let http_shutdown = {
        let mut rx = shutdown_rx.clone();
        async move {
            let _ = rx.changed().await;
        }
    };
    let (benchmark_http_addr, http_serve_task) = run_benchmark_http_api(
        &cfg.benchmark_http_addr,
        index.clone(),
        Arc::clone(&wal),
        Arc::clone(&collections_wal),
        Arc::clone(&lifecycle),
        std::sync::Arc::clone(&collections),
        cfg.vector_dim,
        Arc::clone(&pgwire_metrics),
        applied_sidecar.clone(),
        cfg.wal_path.clone(),
        openai_upstream,
        authz.clone(),
        tls_acceptor.clone(),
        http_inflight,
        Arc::clone(&graph_snapshot_metrics),
        role_state.clone(),
        std::sync::Arc::clone(&replication_metrics),
        promote_ctx.clone(),
        audit,
        encryption.clone(),
        std::sync::Arc::clone(&cluster),
        query_log.clone(),
        http_shutdown,
    )
    .await?;
    tracing::info!("Listening on benchmark_http://{benchmark_http_addr}");

    let pgwire_state = Arc::new(PgWireSharedState {
        index: index.clone(),
        collections: std::sync::Arc::clone(&collections),
        vector_dim: cfg.vector_dim,
        pgwire_hnsw_m: cfg.pgwire_hnsw_m,
        pgwire_hnsw_ef_construction: cfg.pgwire_hnsw_ef_construction,
        wal: Some(Arc::clone(&wal)),
        collections_wal: Some(Arc::clone(&collections_wal)),
        role: role_state.clone(),
        replication_metrics: std::sync::Arc::clone(&replication_metrics),
        cluster: std::sync::Arc::clone(&cluster),
        query_log: query_log.clone(),
    });
    let pgwire_shutdown = {
        let mut rx = shutdown_rx.clone();
        async move {
            let _ = rx.changed().await;
        }
    };
    let pgwire_addr = run_pgwire_listener_with_metrics_and_shutdown(
        &cfg.pgwire_addr,
        Some(pgwire_state),
        Arc::clone(&pgwire_metrics),
        authz.clone(),
        tls_acceptor,
        pgwire_shutdown,
    )
    .await?;
    tracing::info!("Listening on pg_wire://{pgwire_addr}");

    // ── 标量页自动压实（P5.0.1）────────────────────────────────
    // 内存优化，不写 WAL；每个 collection 独立锁，压实失败只告警不影响服务。
    // 首 tick 吞掉：刚启动没有 stale 行，且冷重放可能仍在收尾。
    // EN: In-memory optimization (no WAL); per-collection locks; failures are
    // warnings only. The first tick is swallowed: startup has no stale rows and
    // cold replay may still be finishing.
    let scalar_compaction_task = if cfg.collections_scalar_compact_interval_secs > 0 {
        let manager = std::sync::Arc::clone(&collections);
        let interval = cfg.collections_scalar_compact_interval_secs;
        let ratio = cfg.collections_scalar_compact_stale_ratio;
        Some(tokio::spawn(async move {
            let mut ticker = time::interval(Duration::from_secs(interval));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                for name in manager.list() {
                    let Some(col) = manager.get(&name) else {
                        continue;
                    };
                    match col.compact_scalars(false, ratio).await {
                        Ok(result) if result.skipped_reason.is_none() => {
                            tracing::info!(
                                collection = %name,
                                dropped_rows = result.dropped_rows,
                                physical_after = result.physical_after,
                                "scalar compaction: ok"
                            );
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!(
                                collection = %name,
                                error = %e,
                                "scalar compaction failed"
                            );
                        }
                    }
                }
            }
        }))
    } else {
        None
    };

    // ── 图快照自动触发（P4.4.0e）────────────────────────────────
    // 同一写出例程（GraphSnapshotWriter::write_all）服务三个通道：周期间隔、
    // WAL 增量阈值、关停。周期间隔与阈值共用一个 tick 循环（tick = min(30s,
    // interval)）；快照锁保证周期任务与关停快照不并发编码同一图。
    // EN: One writer routine serves all three triggers (interval, WAL-delta
    // threshold, shutdown); interval/threshold share one tick loop, and a
    // mutex keeps the periodic task and the shutdown snapshot from encoding
    // the same graph concurrently.
    let graph_snapshot_writer = Arc::new(GraphSnapshotWriter {
        index: index.clone(),
        wal_path: cfg.wal_path.clone(),
        applied_sidecar: applied_sidecar.clone(),
        collections: std::sync::Arc::clone(&collections),
        collections_wal: Arc::clone(&collections_wal),
        collections_wal_path: collections_wal_path.clone(),
        metrics: Arc::clone(&graph_snapshot_metrics),
        encryption: encryption.clone(),
        lifecycle: Arc::clone(&lifecycle),
        roles: role_state.clone(),
    });
    let snapshot_task = if cfg.snapshot_interval_secs > 0 || cfg.snapshot_wal_threshold_bytes > 0 {
        let writer = Arc::clone(&graph_snapshot_writer);
        let lock = Arc::clone(&lifecycle);
        let wal_main = Arc::clone(&wal);
        let wal_coll = Arc::clone(&collections_wal);
        let interval_secs = cfg.snapshot_interval_secs;
        let threshold = cfg.snapshot_wal_threshold_bytes;
        Some(tokio::spawn(async move {
            // tick 粒度：阈值通道需要周期性检查；interval < 30s 时跟随 interval。
            // EN: Tick granularity: the threshold channel needs periodic checks;
            // follow the interval when it is below 30s.
            let tick_secs = if interval_secs > 0 {
                interval_secs.min(30)
            } else {
                30
            };
            let mut ticker = time::interval(Duration::from_secs(tick_secs));
            // interval 的首 tick 立即返回——吞掉它：启动即快照没有意义（刚恢复完）。
            // EN: The interval's first tick returns immediately — swallow it:
            // snapshotting at startup is pointless (recovery just finished).
            ticker.tick().await;
            let mut last_at = std::time::Instant::now();
            // 基线 Option 化：启动读失败（None）只等首个成功读建基线——
            // 落 0 会把整个存量 WAL 计为增量，启动即误触发一次全量快照。
            // EN: Baseline as Option: a failed startup read (None) waits for the first
            // successful read to set the baseline — defaulting to 0 would count the
            // whole existing WAL as an increment and falsely fire a full snapshot.
            let mut last_bytes: Option<u64> = total_wal_bytes(&wal_main, &wal_coll).await;
            loop {
                ticker.tick().await;
                let due_interval =
                    interval_secs > 0 && last_at.elapsed() >= Duration::from_secs(interval_secs);
                let current = total_wal_bytes(&wal_main, &wal_coll).await;
                // 增量用 saturating_sub：WAL 轮换/截断让文件变小时增量归零而非
                // 下溢——截断后重放成本已重置，不触发正是期望语义。
                // EN: Use saturating_sub for the increment: WAL rotation/truncation that
                // shrinks the file zeroes the increment instead of underflowing — replay
                // cost is reset after truncation, so not triggering is intended.
                let due_threshold = wal_threshold_due(last_bytes, current, threshold);
                if last_bytes.is_none() && current.is_some() {
                    // 建立基线（不触发）：首个成功读即「上次快照后」的起点。
                    // EN: Establish the baseline (without triggering): the first
                    // successful read is the starting point "since the last snapshot".
                    last_bytes = current;
                }
                if !due_interval && !due_threshold {
                    continue;
                }
                tracing::info!(
                    due_interval,
                    due_threshold,
                    "graph snapshot: auto trigger fired"
                );
                {
                    let _guard = lock.snapshot_lock.lock().await;
                    writer.write_all().await;
                }
                last_at = std::time::Instant::now();
                // 无条件覆盖（OCR 批次十七）：快照已把当前 WAL 全量收入，
                // 读失败时保留旧基线会让下一个成功读把「快照前存量」计为增量
                // 再误触发一次；落 None 走「只建基线不触发」语义干净重基线。
                // EN: Overwrite unconditionally — after a snapshot the whole WAL
                // is covered, so a failed read must re-baseline (None) rather
                // than keep a stale pre-snapshot baseline.
                last_bytes = current;
            }
        }))
    } else {
        None
    };

    // ── collections WAL 段保留（P5.0.3）──────────────────────────
    // 与快照解耦：每 30s 读一次当前 manifest 的安全 checkpoint，按整段删除
    // 最旧归档。leader 在 P5.0.4 之前一律 blocked（没有 follower floor）；
    // follower/单机用本地 manifest。租约由管理员 compact 端点设置。
    // EN: Retention is decoupled from snapshots: every 30s read the current
    // manifest's safe checkpoint and delete oldest whole archives. Leaders are
    // blocked until P5.0.4 (no follower floor); followers/standalone use the
    // local manifest. The admin compact endpoint sets the lease.
    let retention_task = if cfg.collections_wal_max_segments > 0 {
        let wal = Arc::clone(&collections_wal);
        let path = collections_wal_path.clone();
        let roles = role_state.clone();
        let life = Arc::clone(&lifecycle);
        let hub = ship_hub.clone();
        let gate = lifecycle.retention_gate();
        let follower_grace = Duration::from_secs(cfg.collections_wal_follower_grace_secs);
        Some(tokio::spawn(async move {
            let mut ticker = time::interval(Duration::from_secs(30));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                // 写锁覆盖「采样 follower floor → 删除段」：与握手的读锁互斥，
                // 不存在「握手校验通过后段被删」的窗口。
                // EN: the write lock spans floor sampling and segment deletion,
                // mutually exclusive with handshake read locks.
                let _gate = gate.write().await;
                let wal = Arc::clone(&wal);
                let path = path.clone();
                let roles = roles.clone();
                let life = Arc::clone(&life);
                let hub = hub.clone();
                match tokio::task::spawn_blocking(move || {
                    let follower_floor = hub
                        .as_ref()
                        .and_then(|h| h.min_active_durable_lsn(follower_grace));
                    houyi_server::collection_lifecycle::run_retention_with_follower_floor(
                        &wal,
                        &path,
                        &roles,
                        &life,
                        follower_floor,
                    )
                })
                .await
                {
                    Ok(report) if report.deleted_segments > 0 => tracing::info!(
                        deleted_segments = report.deleted_segments,
                        bytes_freed = report.bytes_freed,
                        first_available_lsn = report.first_available_lsn,
                        "collections retention: deleted whole segments"
                    ),
                    Ok(report) => tracing::debug!(
                        archived_segments = report.archived_segments,
                        blocked_reason = ?report.blocked_reason,
                        "collections retention: no deletion"
                    ),
                    Err(e) => tracing::warn!(error = %e, "collections retention task join failed"),
                }
            }
        }))
    } else {
        None
    };

    // ── follower 进度 GC（P5.0.4）──────────────────────────────
    // min_active_durable_lsn 同时负责清理过期断连进度；保留关闭时也必须运行，
    // 否则 follower 频繁重连会让 progress map 持续增长。
    // EN: min_active_durable_lsn also reclaims expired disconnected progress;
    // it must run even when retention is disabled, or follower churn grows the
    // progress map for the process lifetime.
    let progress_gc_task = ship_hub.as_ref().map(|hub| {
        let hub = Arc::clone(hub);
        let grace = Duration::from_secs(cfg.collections_wal_follower_grace_secs);
        tokio::spawn(async move {
            let mut ticker = time::interval(Duration::from_secs(60));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let _ = hub.min_active_durable_lsn(grace);
            }
        })
    });

    // ── 优雅关停序列（P4.0.3）────────────────────────────────────
    // 顺序即语义：先摘流（停止 accept + drain 在途请求），再停 builder 并排空
    // 队列（写入可检索），最后刷 WAL + 侧车——任何一步提前都可能丢「已确认」写。
    // EN: Order matters: stop routing first (stop accept, drain in-flight), then
    // stop the builder and drain its queue (writes become searchable), finally
    // flush WAL + sidecar — reordering can lose acknowledged writes.
    let sig = shutdown_signal().await?;
    tracing::info!(signal = sig, "shutdown signal received, draining");
    let _ = shutdown_tx.send(true);
    observer.abort();
    // 周期快照任务先摘：abort 丢弃其 future（持锁即释放），下方的关停快照
    // 才能独占快照锁；进行中（spawn_blocking 内）的原子写不可被取消，会
    // 后台写完，tmp+rename 保证不产生撕裂文件。
    // EN: Abort the periodic task first so the shutdown snapshot owns the
    // write lock; an in-flight blocking write runs to completion in the
    // background (atomic rename — no torn files).
    if let Some(task) = snapshot_task {
        task.abort();
    }
    if let Some(task) = retention_task {
        task.abort();
    }
    if let Some(task) = progress_gc_task {
        task.abort();
    }
    if let Some(task) = scalar_compaction_task {
        task.abort();
    }
    // 等 HTTP drain 真正完成：axum 停止 accept 并等在途请求处理完毕才返回。
    // 不 await 则 builder/WAL 排空与在途写竞争，main 返回时 runtime drop
    // 还会中途掐断 drain（客户端看到连接重置）。
    // EN: Await the HTTP drain — otherwise builder/WAL teardown races in-flight
    // writes, and runtime drop on main's return aborts the drain mid-flight.
    // drain 必须有上限：在途请求挂死（慢客户端/死锁）会让关停永远停滞，
    // 编排器随后 SIGKILL 反而丢更多。HOUYI_SHUTDOWN_DRAIN_TIMEOUT_SECS，
    // 默认 30s，超时 warn 后继续收尾（放弃等待，但不放弃 WAL/侧车）。
    // EN: The drain needs an upper bound — a hung in-flight request would
    // stall shutdown forever. Timeout warns and continues teardown.
    // 解析失败必须可见：静默落默认会让运维的笔误（如 "30s"）在关停时才暴露。
    // EN: A parse failure must be visible — silently falling back would hide
    // an operator's typo (e.g. "30s") until shutdown time.
    let drain_timeout_secs: u64 = match std::env::var("HOUYI_SHUTDOWN_DRAIN_TIMEOUT_SECS") {
        Ok(v) => match v.parse::<u64>() {
            Ok(0) => {
                tracing::warn!(
                    "HOUYI_SHUTDOWN_DRAIN_TIMEOUT_SECS=0 skips waiting for in-flight \
                     requests (drain aborts immediately)"
                );
                0
            }
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(
                    value = %v,
                    "HOUYI_SHUTDOWN_DRAIN_TIMEOUT_SECS is not a valid u64, falling back to 30s"
                );
                30
            }
        },
        Err(_) => 30,
    };
    // serve 任务 panic（JoinError）不立即返回：先完成 WAL/侧车收尾，
    // 再把失败传导为 main 非零退出——收尾优先，失败不掩盖。
    // EN: A panicked serve task is recorded, teardown (WAL/sidecar) still
    // runs to completion, and the failure is then propagated as a non-zero exit.
    let mut serve_join_err: Option<String> = None;
    match time::timeout(Duration::from_secs(drain_timeout_secs), http_serve_task).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::error!(error = %e, "benchmark http serve task join failed");
            serve_join_err = Some(format!("benchmark http serve task join failed: {e}"));
        }
        Err(_) => {
            tracing::warn!(
                timeout_secs = drain_timeout_secs,
                "http drain timed out, continuing shutdown teardown"
            );
        }
    }
    // 复制先停（P4.4.1）：HTTP drain 之后、builder/WAL 收尾之前——此后数据面
    // 只出不进（leader 不再扇出 / follower 不再 apply），关停快照与 WAL 排空
    // 面对的就是静止状态。join 挂死（慢 follower 会话/退避睡眠）不能拖死关停：
    // 5s 上限 warn 继续，线程随进程退出回收。
    // EN: Stop replication after the HTTP drain and before builder/WAL teardown
    // so the shutdown snapshot sees a quiescent data plane. A wedged join must
    // not stall shutdown — bound at 5s, warn, and continue.
    // P4.5.5：复制句柄关停本身已是 async（task abort + 5s 收尾），不再经
    // spawn_blocking。std 锁只用于同步取出句柄，锁在 await 前释放。
    // EN: replication shutdown is async now (task abort + 5s budget); the std
    // lock is released before the first await.
    // 放进独立 task 再用 timeout 等：超时只取消等待、**不取消关停本身**
    // （drop JoinHandle = detach）。旧 spawn_blocking 版本同语义；直接
    // `timeout(shutdown_future)` 会在超时点丢掉整个 future，把已从槽位取出的
    // 句柄连回收一起取消（OCR 批次二十三）。
    // EN: spawn + timed join preserves the old "timeout detaches, does not
    // cancel" semantics; timing out the future directly would discard the
    // already-taken handles mid-teardown.
    let handles_for_shutdown = std::sync::Arc::clone(&replication_handles);
    let promote_ctx_for_shutdown = promote_ctx.clone();
    let replication_shutdown = tokio::spawn(async move {
        // 与 promote 互斥（P4.6.2）：HTTP drain 超时后 serve task 可能仍被
        // detached 运行并接受 promote——若不取同一把 gate，关停可能取走句柄
        // 与 promote 起新 leader 并发，收尾结果取决于时序。拿 gate 后，
        // promote 要么已完成（其 leader 会被下面的取槽收割），要么排在关停后
        // 失败（服务已关）。
        // EN: serialize against promote — after an HTTP drain timeout the
        // detached serve task may still promote; without this gate, teardown
        // and promote race over the same handles.
        let _promote_gate = match &promote_ctx_for_shutdown {
            Some(ctx) => Some(ctx.promote_gate.lock().await),
            None => None,
        };
        // 共享槽取出全部句柄：起始角色与 promote 换入的 leader 都在此处收尾。
        // 毒锁恢复：某线程 panic 后锁被污染，关停收尾不能因此跳过句柄回收。
        // EN: Recover from a poisoned lock — teardown must still reclaim the
        // replication handles even if some thread panicked while holding it.
        let (leader, follower) = {
            let mut handles = handles_for_shutdown
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (handles.leader.take(), handles.follower.take())
        };
        if let Some(l) = leader {
            l.shutdown().await;
        }
        if let Some(f) = follower {
            f.shutdown().await;
        }
    });
    if tokio::time::timeout(Duration::from_secs(5), replication_shutdown)
        .await
        .is_err()
    {
        tracing::warn!("replication shutdown join timed out (5s), continuing teardown");
    }
    if let Some(w) = worker {
        w.shutdown().await;
    }
    drain_pending_builder(&index).await;
    // 关停图快照（P4.4.0e）默认开启——**行为变更**：此前 HOUYI_SHUTDOWN_SNAPSHOT=1
    // 才开启且写 JSON 向量快照（大数据集导出代价高）；现默认写 HNSW 图快照
    // （主索引 + 全部 collection，二进制格式、重启免重建图），仅 =0 显式关闭。
    // JSON 快照保留为加载侧回退（既有文件仍可恢复）与手动 /v1/wal/snapshot
    // 端点，关停/周期通道不再写。WAL 重放仍是持久性根基；快照失败仅告警，
    // 不阻断后续 WAL/侧车收尾。此刻 builder 已停、队列已排空（上方 drain），
    // 周期任务已 abort——写出的即是最终一致状态。
    // EN: Shutdown graph snapshot is now ON by default (behavior change — it
    // used to be opt-in JSON via HOUYI_SHUTDOWN_SNAPSHOT=1); =0 disables.
    // Failures warn and never block WAL/sidecar teardown.
    if shutdown_snapshot_enabled(std::env::var("HOUYI_SHUTDOWN_SNAPSHOT").ok().as_deref()) {
        let _guard = lifecycle.snapshot_lock.lock().await;
        graph_snapshot_writer.write_all().await;
    }
    // next_lsn 与 shutdown 必须贴进同一 spawn_blocking 且在 shutdown 之前
    // 捕获：Async 模式 shutdown 后写线程退出、通道接收端随之 drop，此后
    // next_lsn() 询问恒失败返回 0，最终侧车写将永远被跳过（OCR 裁决发现的
    // 真实 bug）。两处阻塞调用同房也消除「读取在 worker、shutdown 在阻塞池」
    // 的调度间隙。
    // EN: Capture next_lsn inside the same spawn_blocking, BEFORE shutdown() —
    // once the async writer thread exits, next_lsn() queries fail and return 0,
    // silently disabling the final monotonic sidecar write.
    let wal_main = Arc::clone(&wal);
    let wal_coll = Arc::clone(&collections_wal);
    let final_lsn = match tokio::task::spawn_blocking(move || {
        let lsn = wal_main.next_lsn().saturating_sub(1);
        wal_main.shutdown();
        wal_coll.shutdown();
        lsn
    })
    .await
    {
        Ok(lsn) => lsn,
        Err(e) => {
            // join 失败 = 闭包 panic，WAL 排空状态不可知；final_lsn 保守取 0，
            // 下方单调比较下不会误写侧车。
            // EN: join failure = closure panic; the WAL drain state is unknown, so
            // final_lsn conservatively takes 0; the monotonic comparison below then
            // avoids a mistaken sidecar write.
            tracing::error!(error = %e, "wal shutdown task join failed");
            0
        }
    };
    // 侧车只升不降（与 upsert 路径同口径）：final_lsn 为关停前最后分配的
    // LSN；shutdown drain 保证所有已 append 记录落盘，故其即持久点（不声称
    // 「已应用到索引」——重放以 WAL 为准重建索引，语义自洽）。
    // EN: Monotonic sidecar: final_lsn is the last allocated LSN; the drain
    // flushed every appended record, so it marks the durable point.
    // 读失败（真实故障，非缺文件）时跳过最终写：未知现值就无法保证单调，
    // 宁可不写也不能让小值覆盖大值击穿单调防线。
    // EN: Skip the final write when the sidecar read genuinely fails — without
    // the current value, monotonicity can't be verified.
    match read_applied_lsn(&applied_sidecar) {
        Ok(sidecar_lsn) => {
            if final_lsn > sidecar_lsn {
                if let Err(e) = write_applied_lsn_atomic(&applied_sidecar, final_lsn) {
                    tracing::error!(error = %e, "final applied_lsn sidecar write failed");
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "read applied_lsn sidecar failed, skipping final sidecar write");
        }
    }
    let metrics = index.builder_metrics();
    tracing::info!(?metrics, final_lsn, "builder metrics on shutdown");
    // serve 任务 panic 的失败在最后传导：WAL/侧车收尾已完成，此刻返回 Err
    // 让进程以非零码退出（编排器可据此告警），且不丢任何收尾动作。
    // EN: A panicked serve task's failure propagates last: WAL/sidecar teardown is
    // already done, and returning Err here exits non-zero (so orchestrators can
    // alert) without losing any teardown action.
    if let Some(msg) = serve_join_err {
        return Err(msg.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
