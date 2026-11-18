//! 回归：HTTP upsert → WAL 落盘后杀进程 → 同一路径冷启动 → 检索仍返回预期 id。
//! Regression: WAL-first upsert survives process death; cold restart replays into index and search works.
//!
//! 与单元测试中 `replay_vector_wal_into_index` 不同，这里走 **真实子进程** + **`/v1/vectors/*`**，覆盖 CLI、监听与序列化路径。

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn unique_test_dir() -> PathBuf {
    // pid + 纳秒 + 进程内原子序号：本二进制两个测试并行跑，同时钟周期取值会
    // 撞名（Apple Silicon 时钟粒度粗于 1ns，d605c33 在 houyi-persistence 修过
    // 同根因）。撞名后一方的 clear_wal_artifacts/remove_dir_all 会删掉另一方
    // 正在使用的目录，表现为子进程 WAL open ENOENT 启动即退、stdout EOF。
    // EN: pid + nanos + in-process atomic sequence: two tests in this binary run in parallel,
    // EN: and same-clock-tick values collide (Apple Silicon clock granularity is coarser than
    // EN: 1ns; d605c33 fixed the same root cause in houyi-persistence). On collision one side's
    // EN: clear_wal_artifacts/remove_dir_all deletes the other's live directory, showing up as
    // EN: child WAL open ENOENT at startup and stdout EOF.
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "houyidb-restart-recovery-{}-{nanos}-{seq}",
        std::process::id()
    ))
}

fn clear_wal_artifacts(wal: &Path) {
    let _ = std::fs::remove_file(wal);
    // 与 `houyi-persistence::wal_sidecar::sidecar_path_for_wal` 一致：`foo.wal` → `foo.applied_lsn`
    // EN: Matches `houyi-persistence::wal_sidecar::sidecar_path_for_wal`:
    // EN: `foo.wal` -> `foo.applied_lsn`.
    let parent = wal.parent().unwrap_or_else(|| Path::new("."));
    let stem = wal.file_stem().and_then(|s| s.to_str()).unwrap_or("wal");
    let sidecar = parent.join(format!("{stem}.applied_lsn"));
    let _ = std::fs::remove_file(&sidecar);
    // 对应路径：`foo.wal` → `foo.vector_snapshot.json` / `foo.wal.next_lsn`。
    // EN: `foo.wal` -> `foo.vector_snapshot.json` / `foo.wal.next_lsn`.
    let _ = std::fs::remove_file(wal.with_extension("vector_snapshot.json"));
    let _ = std::fs::remove_file(wal.with_extension("wal.next_lsn"));
}

fn drain_reader_to_eof(mut r: impl Read + Send + 'static) {
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        while r.read(&mut buf).unwrap_or(0) > 0 {}
    });
}

/// 子进程守卫：断言 panic 沿栈展开时也能 kill + wait，防孤儿 houyi-server
/// 泄漏（占着端口/WAL 影响后续用例，CI 上悬挂到 job 超时）。
/// EN: RAII guard that reaps the child even on panic unwind.
struct ChildGuard(Option<std::process::Child>);

impl ChildGuard {
    fn new(child: std::process::Child) -> Self {
        Self(Some(child))
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// 从 stdout 读到 `Listening on benchmark_http://`；随后后台排空 stdout/stderr，避免关闭管道读端导致子进程 **SIGPIPE/EPIPE**。
///
/// 子串匹配而非行前缀：P4.0 起服务器日志走 tracing（行首是 时间戳+级别+target，
/// 级别/target 还带 ANSI 颜色），消息文本出现在行内而非行首。
///
/// stderr 全程捕获到共享缓冲（只留尾部）：子进程启动即退出时（stdout EOF），
/// 其死因只在 stderr——丢弃它曾让 CI 上的间歇失败无法归因。
/// EN: stderr is captured into a shared tail buffer so early child exits are
/// diagnosable from the returned error.
fn read_benchmark_http_base(
    mut child: std::process::Child,
    timeout: Duration,
) -> std::io::Result<(std::process::Child, String)> {
    const MARKER: &str = "Listening on benchmark_http://";
    let stderr_tail: std::sync::Arc<std::sync::Mutex<String>> =
        std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    if let Some(mut stderr) = child.stderr.take() {
        let buf = std::sync::Arc::clone(&stderr_tail);
        std::thread::spawn(move || {
            let mut chunk = [0u8; 8192];
            while let Ok(n) = stderr.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                let mut b = buf.lock().unwrap();
                b.push_str(&String::from_utf8_lossy(&chunk[..n]));
                // 只保留尾部 ~8KB：长跑日志不撑爆内存，启动死因一定在尾部。
                // drain 的字节下标可能落在多字节字符中间（String::drain 对非
                // char 边界直接 panic）——先对齐到边界再删，宁多留几字节。
                // EN: Keep only the last ~8KB: long-run logs do not blow up memory, and startup
                // EN: failures are always at the tail. The drain byte index may land mid multi-byte
                // EN: character (String::drain panics on non-char boundaries) — align to the
                // EN: boundary first, keeping a few extra bytes.
                let len = b.len();
                if len > 8192 {
                    let mut cut = len - 8192;
                    while cut < len && !b.is_char_boundary(cut) {
                        cut += 1;
                    }
                    b.drain(..cut);
                }
            }
        });
    }
    let stdout = child.stdout.take().expect("stdout piped");
    let start = Instant::now();
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    while start.elapsed() < timeout {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        if let Some(pos) = line.find(MARKER) {
            // tracing 的 ANSI 颜色只包裹级别/target，消息文本无颜色；防御性截掉
            // 可能的转义序列与空白，只留 host:port。
            // EN: tracing ANSI colors wrap only level/target, not message text; defensively strip
            // EN: possible escape sequences and whitespace, keeping only host:port.
            let rest = &line[pos + MARKER.len()..];
            let host_port: String = rest
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '\u{1b}')
                .collect();
            // 必须保持管道打开并持续 drain，否则服务器后续日志可能因写端 broken pipe 异常退出。
            // EN: Must keep the pipe open and drain continuously, or later server logs may
            // EN: exit on a broken pipe.
            drain_reader_to_eof(reader);
            return Ok((child, format!("http://{}", host_port)));
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    // 等一拍让 stderr drain 线程把残余字节收进缓冲。
    // EN: Wait one tick for the stderr drain thread to collect remaining bytes.
    std::thread::sleep(Duration::from_millis(50));
    let tail = stderr_tail.lock().unwrap();
    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!(
            "timeout waiting for Listening on benchmark_http://; child stderr tail: {}",
            tail.trim()
        ),
    ))
}

fn post_json(base: &str, path: &str, body: &serde_json::Value) -> serde_json::Value {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let mut resp = ureq::post(&url)
        .header("Content-Type", "application/json; charset=utf-8")
        .send_json(body.clone())
        .unwrap_or_else(|e| panic!("POST {url} failed: {e}"));
    resp.body_mut()
        .read_json()
        .unwrap_or_else(|e| panic!("decode JSON from {url}: {e}"))
}

/// DELETE 请求：返回 `(status, body)`。
/// EN: DELETE request returning `(status, body)`.
///
/// `http_status_as_error(false)` 让 4xx/5xx 也走 `Ok`：ureq 的
/// `Error::StatusCode` 不携带响应体，开启默认行为会丢掉服务端诊断文本，
/// 断言失败时只剩空串（OCR #09）。
/// EN: `http_status_as_error(false)` keeps 4xx/5xx on the `Ok` path: ureq's
/// `Error::StatusCode` carries no response body, so the default behavior drops
/// the server's diagnostic text and failed assertions print an empty string
/// (OCR #09).
fn delete_status(base: &str, path: &str) -> (u16, String) {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    match ureq::delete(&url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
    {
        Ok(mut resp) => {
            let status = resp.status().as_u16();
            let body = resp.body_mut().read_to_string().unwrap_or_default();
            (status, body)
        }
        Err(e) => panic!("DELETE {url} failed: {e}"),
    }
}

/// GET 状态码（业务状态端点：未知 collection 走 404）。
/// EN: GET status code (a missing collection status is 404).
fn get_status(base: &str, path: &str) -> u16 {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    match ureq::get(&url).call() {
        Ok(resp) => resp.status().as_u16(),
        Err(ureq::Error::StatusCode(code)) => code,
        Err(e) => panic!("GET {url} failed: {e}"),
    }
}

/// GET JSON（2xx）；非 2xx panic 并带路径。
/// EN: GET JSON (2xx); non-2xx panics with the path.
fn get_json(base: &str, path: &str) -> serde_json::Value {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let mut resp = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    resp.body_mut()
        .read_json()
        .unwrap_or_else(|e| panic!("decode JSON from {url}: {e}"))
}

/// GET 纯文本（如 `/metrics`）；非 2xx panic 并带路径。
/// EN: GET plain text (e.g. `/metrics`); non-2xx panics with the path.
fn get_text(base: &str, path: &str) -> String {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let mut resp = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    resp.body_mut()
        .read_to_string()
        .unwrap_or_else(|e| panic!("read text from {url}: {e}"))
}

/// 从 Prometheus 文本中取单值；缺失或不可解析即 panic（测试要确定证据）。
/// 只接受「名称 + 空格 + 数字」形态：裸前缀（未来 `..._created` 样本）与
/// 标签样例都不算匹配，避免把别的样本静默当成目标值（OCR fc30113 #0）。
/// EN: Read a single value from Prometheus text; missing/unparsable panics
/// (the test wants deterministic evidence). Only the exact `name value` shape
/// matches — bare prefixes (future `..._created` samples) and labeled samples
/// never silently count as the target metric (OCR fc30113 #0).
fn metric_value(text: &str, name: &str) -> f64 {
    text.lines()
        .find_map(|line| {
            let rest = line.strip_prefix(name)?;
            rest.strip_prefix(' ')?.trim().parse().ok()
        })
        .unwrap_or_else(|| panic!("metric {name} missing or unparsable in /metrics"))
}

/// PUT JSON：返回 `(status, body)`（4xx/5xx 不 panic，供校验矩阵断言）。
/// EN: PUT JSON returning `(status, body)`; 4xx/5xx never panic so validation
/// matrices can assert them.
fn put_json_status(base: &str, path: &str, body: &serde_json::Value) -> (u16, String) {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    // 与 `delete_status` 同理：保留 4xx/5xx 的响应体供断言打印（OCR #09）。
    // EN: Same rationale as `delete_status`: keep the 4xx/5xx body for assertions (OCR #09).
    match ureq::put(&url)
        .header("Content-Type", "application/json; charset=utf-8")
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(body.clone())
    {
        Ok(mut resp) => {
            let status = resp.status().as_u16();
            let body = resp.body_mut().read_to_string().unwrap_or_default();
            (status, body)
        }
        Err(e) => panic!("PUT {url} failed: {e}"),
    }
}

/// 轮询 collection 状态直到 `expected` 个向量全部可检索。
/// EN: Poll the collection status until all `expected` vectors are searchable.
///
/// 仅看 `pending_count == 0` 不够：异步 builder 会先排空队列、后把点写入主
/// 索引，该窗口内 pending 已为 0 但检索仍看不到新点，会让「精确 ids/scores」
/// 断言偶发失败。必须同时要求 hot 缓冲清空且 indexed 达到期望值（OCR #31）。
/// EN: `pending_count == 0` alone is insufficient: the async builder drains its
/// queue before inserting into the main index, so pending is already 0 while
/// search still cannot see the new points, making exact ids/scores assertions
/// flaky. Require an empty hot buffer and indexed >= expected as well (OCR #31).
///
/// 非 200 / 非 JSON 立即失败：404（collection 丢失）或 500 是真实故障，
/// 继续轮询只会把原因拖到超时后丢失（OCR #10）。
/// EN: Fail fast on non-200/non-JSON: a 404 (missing collection) or 500 is a
/// genuine failure; polling would only defer and lose the cause (OCR #10).
fn wait_for_collection_ready(base: &str, name: &str, expected: u64, timeout: Duration) {
    let path = format!("/v1/collections/{name}/status");
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let start = Instant::now();
    let mut last = String::from("no response yet");
    while start.elapsed() < timeout {
        // 单次请求也要受剩余预算约束：ureq 3 默认 `timeout_global=None`，若服务端
        // 接受连接却不应答（例如 handler 死锁），`call()` 会无限阻塞，循环的
        // 超时判断永远不再执行，整个 CI job 只能等外层超时（OCR 195e6be #13）。
        // EN: Bound each request by the remaining budget too: ureq 3 defaults
        // `timeout_global=None`, so a server that accepts the connection but never
        // answers (e.g. a deadlocked handler) blocks `call()` forever, the loop's
        // timeout check is never re-evaluated, and the whole CI job waits for the
        // outer timeout (OCR 195e6be #13).
        let remaining = timeout.saturating_sub(start.elapsed());
        match ureq::get(&url)
            .config()
            .http_status_as_error(false)
            .timeout_global(Some(remaining))
            .build()
            .call()
        {
            Ok(mut resp) => {
                let status = resp.status().as_u16();
                let body = resp.body_mut().read_to_string().unwrap_or_default();
                if status != 200 {
                    panic!("GET {url} returned {status}: {body}");
                }
                let v: serde_json::Value = serde_json::from_str(&body)
                    .unwrap_or_else(|e| panic!("decode JSON from {url}: {e}; body: {body}"));
                // 字段是接口契约（见 handle_collection_status），缺失/改名必须
                // 大声失败；用默认值会把错误形态变成 10 秒后的不可解释超时
                // （OCR 195e6be #07）。
                // EN: The fields are part of the endpoint contract (see
                // handle_collection_status); absence/rename must fail loudly.
                // Defaults would turn the broken shape into an inexplicable
                // timeout after 10s (OCR 195e6be #07).
                let pending = v["pending_count"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("status missing pending_count: {body}"));
                let hot = v["hot_count"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("status missing hot_count: {body}"));
                let indexed = v["indexed_count"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("status missing indexed_count: {body}"));
                if pending == 0 && hot == 0 && indexed >= expected {
                    return;
                }
                last = format!("pending={pending} hot={hot} indexed={indexed} (need {expected})");
            }
            Err(e) => panic!("GET {url} failed: {e}"),
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timeout waiting for {url}: {last}");
}

/// 检索并返回 `(ids, scores, metric)`（metric 是响应里回显的度量串）。
/// EN: search and return `(ids, scores, metric)` (metric is echoed by the response).
fn search_ids_scores(
    base: &str,
    name: &str,
    vector: [f32; 4],
    k: usize,
) -> (Vec<u64>, Vec<f32>, String) {
    let resp = post_json(
        base,
        &format!("/v1/collections/{name}/search"),
        &serde_json::json!({ "vector": vector, "k": k, "ef_search": 128 }),
    );
    let ids = resp["ids"]
        .as_array()
        .expect("ids array")
        .iter()
        .map(|x| x.as_u64().expect("id u64"))
        .collect();
    let scores = resp["scores"]
        .as_array()
        .expect("scores array")
        .iter()
        .map(|x| x.as_f64().expect("score f64") as f32)
        .collect();
    let metric = resp["metric"]
        .as_str()
        .expect("search response must echo the metric")
        .to_string();
    (ids, scores, metric)
}

/// 列出所有 collection 名（`GET /v1/collections` 是对象数组）。
/// EN: list every collection name (`GET /v1/collections` returns an object array).
fn list_collection_names(base: &str) -> Vec<String> {
    let url = format!("{}/v1/collections", base.trim_end_matches('/'));
    let mut resp = ureq::get(&url).call().expect("GET /v1/collections");
    let body: serde_json::Value = resp.body_mut().read_json().expect("decode collections");
    body.as_array()
        .expect("collections array")
        .iter()
        .filter_map(|c| c.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .collect()
}

/// `Listening on...` 打印后，`axum::serve` 的 task 可能尚未开始 accept；轮询直到健康检查成功。
/// EN: After `Listening on...` is printed, the `axum::serve` task may not have started
/// EN: accepting; poll until the health check succeeds.
fn wait_for_health_ok(base: &str, timeout: Duration) {
    let url = format!("{}/v1/health", base.trim_end_matches('/'));
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Ok(mut resp) = ureq::get(&url).call() {
            if let Ok(v) = resp.body_mut().read_json::<serde_json::Value>() {
                if v["status"] == "ok" {
                    return;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timeout waiting for GET {url} status=ok");
}

/// 写入落在异步构图队列里，optimize 后才会出现在检索结果中；VDB 的加载阶段
/// 同样先调用 optimize 再测量，这里保持相同的可见性语义。
/// EN: Writes sit in the async build queue until optimize; VDB's load stage also
/// optimizes before measuring, so mirror that visibility contract here.
fn wait_for_index_ready(base: &str, timeout: Duration) {
    let _ = post_json(base, "/v1/optimize", &serde_json::json!({}));
    let url = format!("{}/v1/index/status", base.trim_end_matches('/'));
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Ok(mut resp) = ureq::get(&url).call() {
            if let Ok(v) = resp.body_mut().read_json::<serde_json::Value>() {
                if v["pending_count"].as_u64().unwrap_or(0) == 0 {
                    return;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timeout waiting for GET {url} pending_count=0");
}

/// 启动 `houyi-server` 子进程（HTTP / PG 端口均为 0，由内核分配）；返回 `(child, benchmark base URL)`。
/// EN: Start the `houyi-server` child process (HTTP / PG ports are 0, kernel-assigned);
/// EN: returns `(child, benchmark base URL)`.
fn spawn_houyi_server(wal: &Path) -> (std::process::Child, String) {
    // Cargo 在 `cargo test` 时注入 `CARGO_BIN_EXE_<bin>`（`houyi-server` → `..._houyi_server`）。
    // EN: Cargo injects `CARGO_BIN_EXE_<bin>` during `cargo test` (`houyi-server` ->
    // EN: `..._houyi_server`).
    let bin = std::env::var("CARGO_BIN_EXE_houyi-server")
        .or_else(|_| std::env::var("CARGO_BIN_EXE_houyi_server"))
        .expect("binary path missing — run via `cargo test -p houyi-server`");
    let child = Command::new(bin)
        .args([
            "--benchmark-http-addr",
            "127.0.0.1:0",
            "--pgwire-addr",
            "127.0.0.1:0",
            "--wal-path",
            wal.to_str().expect("utf8 wal path"),
            "--vector-dim",
            "4",
            "--worker-threshold",
            "8",
            "--worker-tick-ms",
            "20",
            "--metrics-interval-secs",
            "120",
            "--pgwire-hnsw-m",
            "24",
            "--pgwire-hnsw-efc",
            "400",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn houyi-server");
    let (child, base) =
        read_benchmark_http_base(child, Duration::from_secs(45)).expect("read benchmark http url");
    (child, base)
}

#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn fresh_server_reports_empty_index_and_effective_hnsw_params() {
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let _child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let url = format!("{}/v1/index/status", base.trim_end_matches('/'));
    let mut resp = ureq::get(&url).call().expect("GET /v1/index/status");
    let status: serde_json::Value = resp
        .body_mut()
        .read_json()
        .expect("decode /v1/index/status");
    // 启动自检的保留探针必须被清理：VDB 客户端把「索引为空」当作 drop_old
    // 的准入条件；残留探针会让每一次冷启动都被误判为脏索引。
    // EN: The startup probes must be gone: the VDB client treats an empty index
    // as the drop_old precondition, so leftovers would reject every cold start.
    assert_eq!(
        status["hot_count"], 0,
        "self-check probes not removed: {status}"
    );
    assert_eq!(status["indexed_count"], 0, "{status}");
    assert_eq!(status["pending_count"], 0, "{status}");
    // 构建参数必须在启动时就生效并可观测，客户端据此在测量前核对。
    // EN: Build parameters must be effective and observable at startup so the
    // client can verify them before measuring.
    assert_eq!(status["distance_metric"], "cosine", "{status}");
    assert_eq!(status["hnsw_m"], 24, "{status}");
    assert_eq!(status["hnsw_ef_construction"], 400, "{status}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn wal_persisted_vectors_survive_server_restart() {
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let upsert_body = serde_json::json!({
        "vectors": [
            { "id": 1, "values": [1.0_f32, 0.0, 0.0, 0.0] },
            { "id": 2, "values": [0.0_f32, 1.0, 0.0, 0.0] }
        ]
    });

    let (child1, base) = spawn_houyi_server(&wal);
    let child1 = ChildGuard::new(child1);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let up = post_json(&base, "/v1/vectors/upsert", &upsert_body);
    assert_eq!(up["inserted"], 2);
    // 异步构建队列中的点尚不可检索；VDB 的加载阶段同样先 optimize 再测量。
    // EN: Queued points are not searchable yet; VDB likewise optimizes before measuring.
    wait_for_index_ready(&base, Duration::from_secs(10));

    let search = post_json(
        &base,
        "/v1/vectors/search",
        &serde_json::json!({
            "vector": [1.0_f32, 0.0, 0.0, 0.0],
            "k": 2,
            "ef_search": 32
        }),
    );
    let ids_before: Vec<u64> = search["ids"]
        .as_array()
        .expect("ids array")
        .iter()
        .map(|x| x.as_u64().expect("id u64"))
        .collect();
    assert!(
        ids_before.contains(&1),
        "before restart: expected id 1 in top-k, got {:?}",
        ids_before
    );

    drop(child1); // SIGKILL + reap，随后同 WAL 路径冷启动

    let (child2, base2) = spawn_houyi_server(&wal);
    let child2 = ChildGuard::new(child2);
    wait_for_health_ok(&base2, Duration::from_secs(5));
    wait_for_index_ready(&base2, Duration::from_secs(10));

    let search_after = post_json(
        &base2,
        "/v1/vectors/search",
        &serde_json::json!({
            "vector": [1.0_f32, 0.0, 0.0, 0.0],
            "k": 2,
            "ef_search": 32
        }),
    );
    let ids_after: Vec<u64> = search_after["ids"]
        .as_array()
        .expect("ids array")
        .iter()
        .map(|x| x.as_u64().expect("id u64"))
        .collect();
    assert!(
        ids_after.contains(&1),
        "after restart: expected id 1 in top-k, got {:?}",
        ids_after
    );

    // 先停服务再清理工件：活进程握着 WAL/侧车句柄，清理不应与其并发。
    // EN: Stop the server before cleaning artifacts: a live process holds WAL/sidecar handles,
    // EN: so cleanup must not race with it.
    drop(child2);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 故障注入（P4.1.2）：写入负载进行中 SIGKILL —— WAL 尾部可能是撕裂的半条
/// 记录（异步写线程批量 flush 被杀在半途），重启后：
/// 1) 服务必须能起（v2 CRC + 撕裂尾容忍，重放不 panic）；
/// 2) 所有「已收到 200 确认」的向量必须可检索（page cache 在进程死后仍存活，
///    ack 即持久）；在途未确认的请求允许丢失（客户端未收到成功，可安全重试）。
///
/// EN: SIGKILL mid-write-load: the WAL tail may be a torn partial record.
/// After restart every acknowledged vector must be searchable; unacknowledged
/// in-flight writes may be lost (client never saw success).
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn wal_survives_sigkill_under_write_load() {
    use std::sync::{Arc, Mutex};

    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    // id → 确定性向量（方向随 id 变化，检索自身向量必得 score=1 的 top 组）。
    // EN: id -> deterministic vector (direction varies with id; searching its own vector yields
    // EN: the score=1 top group).
    fn vec_for_id(id: u64) -> [f32; 4] {
        [
            (id % 7 + 1) as f32,
            (id % 5 + 1) as f32,
            (id % 3 + 1) as f32,
            (id % 11 + 1) as f32,
        ]
    }

    let (child1, base) = spawn_houyi_server(&wal);
    let child1 = ChildGuard::new(child1);
    wait_for_health_ok(&base, Duration::from_secs(5));

    // 后台写入线程：持续批量 upsert；被杀后连接断开自然出错退出。
    // EN: Background writer thread: continuous batched upserts; after the kill the connection
    // EN: drops and it exits with an error naturally.
    let acked: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    // 写入线程最后一次错误：被杀后的连接错误是预期退出路径，被杀前的错误
    // 是真实故障——断言失败时并入诊断，省得盲猜是负载不足还是服务器早崩。
    // EN: Writer thread's last error: after the kill, a connection error is the expected exit
    // EN: path; before the kill it is a real fault — include it in assertion diagnostics
    // EN: instead of guessing whether load was insufficient or the server died early.
    let writer_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let writer = {
        let base = base.clone();
        let acked = Arc::clone(&acked);
        let writer_err = Arc::clone(&writer_err);
        std::thread::spawn(move || {
            const BATCH: u64 = 20;
            let mut next: u64 = 1_000_000; // 避开 main 自检向量 id 0..2
            loop {
                let vectors: Vec<serde_json::Value> = (next..next + BATCH)
                    .map(|id| {
                        let v = vec_for_id(id);
                        serde_json::json!({ "id": id, "values": v })
                    })
                    .collect();
                let url = format!("{}/v1/vectors/upsert", base.trim_end_matches('/'));
                // ureq 3：非 2xx 响应走 Err(Error::StatusCode)，连接错误走
                // 其余 Err 变体——Ok 即成功，无独立 status 分支可判。
                // EN: ureq 3: non-2xx responses yield Err(Error::StatusCode); connection errors
                // EN: use other Err variants — Ok means success, no separate status branch.
                match ureq::post(&url)
                    .header("Content-Type", "application/json; charset=utf-8")
                    .send_json(serde_json::json!({ "vectors": vectors }))
                {
                    Ok(_) => {
                        acked.lock().unwrap().extend(next..next + BATCH);
                        next += BATCH;
                    }
                    Err(e) => {
                        *writer_err.lock().unwrap() = Some(format!("POST {url} => {e}"));
                        break; // 服务器已被杀：连接错误即退出
                    }
                }
            }
        })
    };

    // 等足够多批次确认后，在写入流中段 SIGKILL（保证杀在写路径活跃期）。
    // 写入线程若提前死亡（如服务器在负载期崩溃），立即退出轮询而非干等
    // 15s——下方断言会带着 writer_err 诊断爆炸。
    // EN: After enough acked batches, SIGKILL mid-stream (ensures the write path is active).
    // EN: If the writer dies early (e.g. the server crashes under load), exit the poll
    // EN: immediately instead of waiting 15s — the assertions below explode with writer_err
    // EN: diagnostics.
    let start = Instant::now();
    loop {
        let n = acked.lock().unwrap().len();
        if n >= 200 || writer.is_finished() || start.elapsed() > Duration::from_secs(15) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(child1); // SIGKILL + reap（ChildGuard）
                  // join 结果必须检查：writer panic（而非记录错误）时若静默吞掉，
                  // 后面的 acked 断言会在一个不完整的数据集上"通过"。
                  // EN: The join result must be checked: if a writer panic (not a recorded error)
                  // EN: is swallowed, the later acked assertions "pass" on an incomplete dataset.
    writer.join().expect("writer thread must not panic");
    // join 之后才取快照：写入线程已停，快照即最终全集。先 take 后杀的旧顺序
    // 存在竞态窗口——窗口内确认的 id 会漏出校验集（已确认却未验证）。
    // EN: Take the snapshot only after join: the writer has stopped, so it is the final set.
    // EN: The old take-then-kill order had a race window where acked ids escaped verification.
    let acked_ids: Vec<u64> = std::mem::take(&mut *acked.lock().unwrap());
    let last_writer_err = writer_err.lock().unwrap().take();
    assert!(
        acked_ids.len() >= 200,
        "load phase must acknowledge >= 200 vectors, got {}; last writer error: {:?}",
        acked_ids.len(),
        last_writer_err
    );

    // 重启：撕裂 WAL 不得阻碍启动。
    // EN: Restart: a torn WAL must not block startup.
    let (child2, base2) = spawn_houyi_server(&wal);
    let child2 = ChildGuard::new(child2);
    wait_for_health_ok(&base2, Duration::from_secs(10));

    // 抽样已确认 id（首/尾各 5 + 均匀 10 个），每个必须落在自身向量检索的
    // top-8（score=1 同分组内，k=8 吸收同分平局）。
    // EN: Sample acked ids (first/last 5 + 10 uniform); each must land in the top-8 of its
    // EN: own-vector search (score=1 tie group; k=8 absorbs equal-score ties).
    let mut sample: Vec<u64> = Vec::new();
    sample.extend_from_slice(&acked_ids[..5]);
    sample.extend_from_slice(&acked_ids[acked_ids.len() - 5..]);
    let step = (acked_ids.len() / 10).max(1);
    sample.extend(acked_ids.iter().step_by(step).take(10).copied());
    for id in sample {
        let v = vec_for_id(id);
        let res = post_json(
            &base2,
            "/v1/vectors/search",
            &serde_json::json!({ "vector": v, "k": 8, "ef_search": 64 }),
        );
        let ids: Vec<u64> = res["ids"]
            .as_array()
            .expect("ids array")
            .iter()
            .map(|x| x.as_u64().expect("id u64"))
            .collect();
        assert!(
            ids.contains(&id),
            "acknowledged id {id} must be searchable after SIGKILL restart, got {ids:?}"
        );
    }

    // 先停服务再清理工件（同测试一）。
    // EN: Stop the server before cleaning artifacts (same as test one).
    drop(child2);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 管理删除（`DELETE /v1/admin/collections/{name}`）：200 后 list/status 立即消失，
/// 且 WAL-first 的 `collection_drop` 记录必须让同一 WAL 冷启动后仍保持「已删除」。
/// EN: admin drop: after 200 the list/status hide the collection immediately, and
/// the WAL-first `collection_drop` record keeps it gone across a cold restart.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn dropped_collection_stays_dropped_after_restart() {
    const NAME: &str = "docs";
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child1, base) = spawn_houyi_server(&wal);
    let child1 = ChildGuard::new(child1);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let up = post_json(
        &base,
        &format!("/v1/collections/{NAME}/upsert"),
        &serde_json::json!({
            "vectors": [{ "id": 1, "values": [1.0_f32, 0.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(up["inserted"], 1);
    assert!(
        list_collection_names(&base).contains(&NAME.to_string()),
        "collection must exist before the drop"
    );

    let (code, body) = delete_status(&base, &format!("/v1/admin/collections/{NAME}"));
    assert_eq!(code, 200, "drop response body: {body}");
    let resp: serde_json::Value = serde_json::from_str(&body).expect("drop response JSON");
    assert_eq!(resp["deleted"], true);
    assert_eq!(resp["collection"], NAME);

    assert!(
        !list_collection_names(&base).contains(&NAME.to_string()),
        "dropped collection must disappear from the list"
    );
    assert_eq!(
        get_status(&base, &format!("/v1/collections/{NAME}/status")),
        404,
        "status of a dropped collection must be 404"
    );
    // 重复删除同样是 404（幂等失败而非 500）。
    // EN: a repeat drop is 404 too (idempotent failure, never 500).
    let (code, _) = delete_status(&base, &format!("/v1/admin/collections/{NAME}"));
    assert_eq!(code, 404);

    // SIGKILL + reap，随后同 WAL 路径冷启动。
    // EN: SIGKILL + reap, then cold start on the same WAL path.
    drop(child1);

    let (child2, base2) = spawn_houyi_server(&wal);
    let child2 = ChildGuard::new(child2);
    wait_for_health_ok(&base2, Duration::from_secs(5));
    assert!(
        !list_collection_names(&base2).contains(&NAME.to_string()),
        "collection_drop must survive cold replay"
    );
    assert_eq!(
        get_status(&base2, &format!("/v1/collections/{NAME}/status")),
        404,
        "dropped collection must stay gone after restart"
    );

    drop(child2);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 名为 `compact` 的 collection 必须经静态管理路由删除：matchit 静态段优先于
/// `{name}`，若没把 DELETE 挂到静态路由，这个 DELETE 会 405 且 collection 删不掉。
/// EN: a collection named `compact` must drop through the static admin route:
/// matchit prefers the static segment over `{name}`, so without mounting DELETE
/// there this DELETE is 405 and the collection can never be dropped.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn compact_named_collection_drops_through_static_route() {
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let up = post_json(
        &base,
        "/v1/collections/compact/upsert",
        &serde_json::json!({
            "vectors": [{ "id": 1, "values": [1.0_f32, 0.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(up["inserted"], 1);

    let (code, body) = delete_status(&base, "/v1/admin/collections/compact");
    assert_eq!(code, 200, "drop response body: {body}");
    let resp: serde_json::Value = serde_json::from_str(&body).expect("drop response JSON");
    assert_eq!(resp["collection"], "compact");
    assert!(!list_collection_names(&base).contains(&"compact".to_string()));
    assert_eq!(get_status(&base, "/v1/collections/compact/status"), 404);

    drop(child);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 显式创建（`PUT /v1/collections/{name}`，P5.1）：metric/HNSW 参数在状态、列表、
/// 检索分数与冷重启后都保持一致；重复创建 409。
/// EN: explicit create: metric/HNSW params stay consistent across status, list,
/// search scores, and a cold restart; a duplicate create is 409.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn explicit_create_ip_metric_survives_restart() {
    const NAME: &str = "ip-docs";
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let body = serde_json::json!({
        "dims": 4,
        "metric": "ip",
        "index_type": "hnsw",
        "hnsw_m": 16,
        "hnsw_ef_construction": 100,
        "hnsw_ef_search": 32
    });
    let (code, resp) = put_json_status(&base, &format!("/v1/collections/{NAME}"), &body);
    assert_eq!(code, 201, "create response: {resp}");
    let resp: serde_json::Value = serde_json::from_str(&resp).expect("create response JSON");
    assert_eq!(resp["created"], true);
    assert_eq!(resp["collection"], NAME);
    assert_eq!(resp["dims"], 4);
    assert_eq!(resp["metric"], "ip");
    assert_eq!(resp["index_type"], "hnsw");
    assert_eq!(resp["hnsw"]["m"], 16);
    assert_eq!(resp["hnsw"]["ef_construction"], 100);
    assert_eq!(resp["hnsw"]["ef_search"], 32);

    // 同名重复创建必须 409，而不是静默改写度量/参数。
    // EN: a duplicate create must be 409, never a silent metric/param rewrite.
    let (code, _) = put_json_status(&base, &format!("/v1/collections/{NAME}"), &body);
    assert_eq!(code, 409);

    let status = get_json(&base, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(status["metric"], "ip", "{status}");
    assert_eq!(status["hnsw"]["m"], 16, "{status}");
    let list = get_json(&base, "/v1/collections");
    let entry = list
        .as_array()
        .expect("collections array")
        .iter()
        .find(|c| c["name"] == NAME)
        .expect("created collection in list");
    assert_eq!(entry["metric"], "ip", "{entry}");
    assert_eq!(entry["hnsw"]["ef_search"], 32, "{entry}");

    // ip 的 HTTP 分数是原始点积：query=[1,0,0,0] 对 [1,2,3,4]/[2,0,0,0] 得 1/2，
    // 距离为 -dot 升序，因此 id 9（点积 2）排前、id 5（点积 1）随后。
    // EN: ip scores are raw dot products: query [1,0,0,0] against [1,2,3,4] /
    // [2,0,0,0] yields 1 / 2; distance is -dot ascending, so id 9 (dot 2) leads
    // and id 5 (dot 1) follows.
    let up = post_json(
        &base,
        &format!("/v1/collections/{NAME}/upsert"),
        &serde_json::json!({
            "vectors": [
                { "id": 5, "values": [1.0_f32, 2.0, 3.0, 4.0] },
                { "id": 9, "values": [2.0_f32, 0.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(up["inserted"], 2);
    wait_for_collection_ready(&base, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "ip");
    assert_eq!(ids, vec![9, 5], "ip ordering by descending dot product");
    assert!(
        (scores[0] - 2.0).abs() < 1e-6,
        "dot product score, got {scores:?}"
    );
    assert!(
        (scores[1] - 1.0).abs() < 1e-6,
        "dot product score, got {scores:?}"
    );

    // 冷重启：快照/WAL 必须带回 metric 与 HNSW 参数，重建的是 ip 而非 Cosine。
    // EN: cold restart: snapshot/WAL must restore metric and HNSW params, rebuilding
    // ip rather than Cosine.
    drop(child);
    let (child2, base2) = spawn_houyi_server(&wal);
    let child2 = ChildGuard::new(child2);
    wait_for_health_ok(&base2, Duration::from_secs(5));
    let status = get_json(&base2, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(
        status["metric"], "ip",
        "metric must survive restart: {status}"
    );
    assert_eq!(status["hnsw"]["m"], 16, "{status}");
    wait_for_collection_ready(&base2, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base2, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "ip");
    assert_eq!(ids, vec![9, 5], "ip ordering must survive restart");
    assert!(
        (scores[0] - 2.0).abs() < 1e-6,
        "dot product score, got {scores:?}"
    );
    assert!(
        (scores[1] - 1.0).abs() < 1e-6,
        "dot product score, got {scores:?}"
    );

    drop(child2);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// P1.7：f16 存储编码经 HTTP 创建 → WAL → 冷重启完整闭环。创建响应与状态回显
/// f16；f16 写入的向量检索结果与 f32 参考一致；WAL 重放后编码不变（不会退回 f32）。
/// 同时验证非法编码/非 hnsw 组合在落 WAL 前 400。
/// EN: P1.7 — f16 storage encoding through the full HTTP create → WAL → cold
/// restart loop. Create/status echo f16; searches over f16-stored vectors match
/// the f32 reference; WAL replay must preserve the encoding (never fall back to
/// f32). Illegal encodings / non-hnsw combos are 400 before any WAL record.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn f16_collection_survives_restart_with_same_dtype() {
    const NAME: &str = "f16-docs";
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let body = serde_json::json!({
        "dims": 4,
        "metric": "l2",
        "index_type": "hnsw",
        "hnsw_m": 16,
        "hnsw_ef_construction": 100,
        "hnsw_ef_search": 32,
        "vector_dtype": "f16"
    });
    let (code, resp) = put_json_status(&base, &format!("/v1/collections/{NAME}"), &body);
    assert_eq!(code, 201, "create response: {resp}");
    let resp: serde_json::Value = serde_json::from_str(&resp).expect("create response JSON");
    assert_eq!(resp["created"], true);
    assert_eq!(resp["vector_dtype"], "f16", "{resp}");
    assert_eq!(resp["index_type"], "hnsw", "{resp}");

    // 状态端点同样回显存储编码；旧 f32 集合应回显 f32（缺省）。
    // EN: the status endpoint echoes the encoding too; legacy f32 collections
    // echo f32 (the default).
    let status = get_json(&base, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(status["vector_dtype"], "f16", "{status}");

    // 选全部可被 f16 精确表示的值：f16 存储后的距离/排序必须与 f32 参考一致
    // （2、3、4、29 都在 f16 精确网格上，可用 1e-6 严格比较）。
    // EN: values chosen to be exactly representable in f16: distances/ordering
    // after f16 storage must match the f32 reference exactly (2, 3, 4, 29 all
    // sit on the f16 grid; strict 1e-6 comparison is safe).
    let up = post_json(
        &base,
        &format!("/v1/collections/{NAME}/upsert"),
        &serde_json::json!({
            "vectors": [
                { "id": 5, "values": [1.0_f32, 2.0, 3.0, 4.0] },
                { "id": 9, "values": [2.0_f32, 0.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(up["inserted"], 2);
    wait_for_collection_ready(&base, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![9, 5], "l2 ordering over f16 storage");
    assert!(
        (scores[0] - 0.0).abs() < 1e-6,
        "distance 1 → score 0, got {scores:?}"
    );
    assert!(
        (scores[1] + 28.0).abs() < 1e-6,
        "distance 29 → score -28, got {scores:?}"
    );

    // 冷重启：WAL 重放必须保留 vector_dtype=f16，而不是按缺省 f32 重建。
    // EN: cold restart: WAL replay must preserve vector_dtype=f16 rather than
    // rebuilding with the f32 default.
    drop(child);
    let (child2, base2) = spawn_houyi_server(&wal);
    let child2 = ChildGuard::new(child2);
    wait_for_health_ok(&base2, Duration::from_secs(5));
    let status = get_json(&base2, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(
        status["vector_dtype"], "f16",
        "dtype must survive restart: {status}"
    );
    assert_eq!(status["index_type"], "hnsw", "{status}");
    wait_for_collection_ready(&base2, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base2, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![9, 5], "l2 ordering must survive restart");
    assert!((scores[0] - 0.0).abs() < 1e-6, "{scores:?}");
    assert!((scores[1] + 28.0).abs() < 1e-6, "{scores:?}");

    // 非法编码 / 非 hnsw 组合必须在落 WAL 前 400（重启后不会留下幽灵集合）。
    // EN: illegal encodings / non-hnsw combos are 400 before any WAL record
    // (no ghost collection can appear after a restart).
    let (code, _) = put_json_status(
        &base2,
        "/v1/collections/bad-encoding",
        &serde_json::json!({"dims": 4, "index_type": "hnsw", "vector_dtype": "bf16"}),
    );
    assert_eq!(code, 400);
    let (code, _) = put_json_status(
        &base2,
        "/v1/collections/flat-f16",
        &serde_json::json!({"dims": 4, "index_type": "flat", "vector_dtype": "f16"}),
    );
    assert_eq!(code, 400);
    assert_eq!(
        get_status(&base2, "/v1/collections/bad-encoding/status"),
        404
    );
    assert_eq!(get_status(&base2, "/v1/collections/flat-f16/status"), 404);

    drop(child2);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// P2：图侧 int8 量化经 HTTP 创建 → WAL → 冷重启完整闭环。检索分数由主表原始
/// 向量精确重排，必须与 f32 参考逐位一致；重启后 quantization=int8 不变（不会
/// 静默退回未量化图）。非法值/非 hnsw 组合落 WAL 前 400。
/// EN: P2 — graph-side int8 quantization through the full HTTP create → WAL →
/// cold restart loop. Scores are exact-rescored on the raw main-store vectors and
/// must match the f32 reference exactly; after restart quantization stays int8
/// (never silently unquantized). Illegal values / non-hnsw combos are 400 before
/// any WAL record.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn int8_collection_survives_restart_and_matches_f32_rescore() {
    const NAME: &str = "int8-docs";
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let body = serde_json::json!({
        "dims": 4,
        "metric": "l2",
        "index_type": "hnsw",
        "hnsw_m": 16,
        "hnsw_ef_construction": 100,
        "hnsw_ef_search": 32,
        "quantization": "int8"
    });
    let (code, resp) = put_json_status(&base, &format!("/v1/collections/{NAME}"), &body);
    assert_eq!(code, 201, "create response: {resp}");
    let resp: serde_json::Value = serde_json::from_str(&resp).expect("create response JSON");
    assert_eq!(resp["created"], true);
    assert_eq!(resp["quantization"], "int8", "{resp}");
    assert_eq!(resp["vector_dtype"], "f32", "{resp}");

    let status = get_json(&base, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(status["quantization"], "int8", "{status}");
    // capability 探测同步宣告新特性，供 EE 客户端协商。
    // EN: the capability probe announces the feature for EE client negotiation.
    let version = get_json(&base, "/v1/version");
    let caps: Vec<&str> = version["capabilities"]
        .as_array()
        .expect("capabilities array")
        .iter()
        .filter_map(|c| c.as_str())
        .collect();
    assert!(
        caps.contains(&"quantization_int8"),
        "capabilities: {caps:?}"
    );

    // 与 f16 测试同一组值：int8 只影响图内距离，最终分数按主表 f32 精确计算，
    // 因此可与 f32 参考严格比较（1e-6）。
    // EN: same values as the f16 test: int8 affects only in-graph distance; final
    // scores are exact on the raw f32 main store and compare strictly (1e-6).
    let up = post_json(
        &base,
        &format!("/v1/collections/{NAME}/upsert"),
        &serde_json::json!({
            "vectors": [
                { "id": 5, "values": [1.0_f32, 2.0, 3.0, 4.0] },
                { "id": 9, "values": [2.0_f32, 0.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(up["inserted"], 2);
    wait_for_collection_ready(&base, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![9, 5], "l2 ordering over int8 graph");
    assert!(
        (scores[0] - 0.0).abs() < 1e-6,
        "distance 1 → score 0 (exact rescore), got {scores:?}"
    );
    assert!(
        (scores[1] + 28.0).abs() < 1e-6,
        "distance 29 → score -28 (exact rescore), got {scores:?}"
    );

    // 冷重启：WAL 重放必须保留 quantization=int8，而不是按未量化重建。
    // EN: cold restart: WAL replay must preserve quantization=int8 rather than
    // rebuilding unquantized.
    drop(child);
    let (child2, base2) = spawn_houyi_server(&wal);
    let child2 = ChildGuard::new(child2);
    wait_for_health_ok(&base2, Duration::from_secs(5));
    let status = get_json(&base2, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(
        status["quantization"], "int8",
        "quantization must survive restart: {status}"
    );
    assert_eq!(status["index_type"], "hnsw", "{status}");
    wait_for_collection_ready(&base2, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base2, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![9, 5], "l2 ordering must survive restart");
    assert!((scores[0] - 0.0).abs() < 1e-6, "{scores:?}");
    assert!((scores[1] + 28.0).abs() < 1e-6, "{scores:?}");

    // 同名再声明不同量化（缺省 none）→ 409，不得改写既有集合。metric 必须与
    // 创建时一致（l2），否则会先在 metric 比对处 409，碰不到量化比对。
    // EN: re-requesting the same name with a different quantization (default
    // none) → 409; the existing collection must not be rewritten. The metric
    // must match creation time (l2) or the metric check 409s first and the
    // quantization comparison is never reached.
    let (code, _) = put_json_status(
        &base2,
        &format!("/v1/collections/{NAME}"),
        &serde_json::json!({"dims": 4, "index_type": "hnsw", "metric": "l2"}),
    );
    assert_eq!(code, 409);

    // 非法值 / 非 hnsw 组合必须在落 WAL 前 400（重启后不会留下幽灵集合）。
    // EN: illegal values / non-hnsw combos are 400 before any WAL record (no
    // ghost collection can appear after a restart).
    let (code, _) = put_json_status(
        &base2,
        "/v1/collections/bad-quantization",
        &serde_json::json!({"dims": 4, "index_type": "hnsw", "quantization": "int4"}),
    );
    assert_eq!(code, 400);
    let (code, _) = put_json_status(
        &base2,
        "/v1/collections/flat-int8",
        &serde_json::json!({"dims": 4, "index_type": "flat", "quantization": "int8"}),
    );
    assert_eq!(code, 400);
    assert_eq!(
        get_status(&base2, "/v1/collections/bad-quantization/status"),
        404
    );
    assert_eq!(get_status(&base2, "/v1/collections/flat-int8/status"), 404);

    drop(child2);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// P3：图侧 rabitq 量化经 HTTP 创建 → WAL → 冷重启完整闭环。检索分数由主表
/// 原始向量精确重排，必须与 f32 参考逐位一致；重启后 quantization=rabitq 不变
/// （不会静默退回未量化图）。ip/l1/非法值/非 hnsw 组合落 WAL 前 400。
/// EN: P3 — graph-side rabitq quantization through the full HTTP create → WAL →
/// cold restart loop. Scores are exact-rescored on the raw main-store vectors and
/// must match the f32 reference exactly; after restart quantization stays rabitq
/// (never silently unquantized). ip/l1, illegal values and non-hnsw combos are
/// 400 before any WAL record.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn rabitq_collection_survives_restart_and_matches_f32_rescore() {
    const NAME: &str = "rabitq-docs";
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let body = serde_json::json!({
        "dims": 4,
        "metric": "l2",
        "index_type": "hnsw",
        "hnsw_m": 16,
        "hnsw_ef_construction": 100,
        "hnsw_ef_search": 32,
        "quantization": "rabitq"
    });
    let (code, resp) = put_json_status(&base, &format!("/v1/collections/{NAME}"), &body);
    assert_eq!(code, 201, "create response: {resp}");
    let resp: serde_json::Value = serde_json::from_str(&resp).expect("create response JSON");
    assert_eq!(resp["created"], true);
    assert_eq!(resp["quantization"], "rabitq", "{resp}");
    assert_eq!(resp["vector_dtype"], "f32", "{resp}");

    let status = get_json(&base, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(status["quantization"], "rabitq", "{status}");
    // capability 探测同步宣告新特性，供 EE 客户端协商。
    // EN: the capability probe announces the feature for EE client negotiation.
    let version = get_json(&base, "/v1/version");
    let caps: Vec<&str> = version["capabilities"]
        .as_array()
        .expect("capabilities array")
        .iter()
        .filter_map(|c| c.as_str())
        .collect();
    assert!(
        caps.contains(&"quantization_rabitq"),
        "capabilities: {caps:?}"
    );

    // 与 int8 测试同一组值：rabitq 只影响图内候选，最终分数按主表 f32 精确计算，
    // 因此可与 f32 参考严格比较（1e-6）。
    // EN: same values as the int8 test: rabitq affects only in-graph candidates;
    // final scores are exact on the raw f32 main store and compare strictly (1e-6).
    let up = post_json(
        &base,
        &format!("/v1/collections/{NAME}/upsert"),
        &serde_json::json!({
            "vectors": [
                { "id": 5, "values": [1.0_f32, 2.0, 3.0, 4.0] },
                { "id": 9, "values": [2.0_f32, 0.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(up["inserted"], 2);
    wait_for_collection_ready(&base, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![9, 5], "l2 ordering over rabitq graph");
    assert!(
        (scores[0] - 0.0).abs() < 1e-6,
        "distance 1 → score 0 (exact rescore), got {scores:?}"
    );
    assert!(
        (scores[1] + 28.0).abs() < 1e-6,
        "distance 29 → score -28 (exact rescore), got {scores:?}"
    );

    // 周期快照默认 600s，测试内不会自行发生；显式触发一轮完整快照，让冷启动
    // 真正走快照恢复路径（图快照 v4 + manifest format 3 + payload flag bit2），
    // 而不是只重放 WAL create（OCR 728e994 #0）。
    // EN: The periodic snapshot defaults to 600s and never fires inside a test;
    // trigger a full cycle explicitly so the cold start really exercises the
    // snapshot restore path (graph snapshot v4 + manifest format 3 + payload flag
    // bit2) instead of merely replaying the WAL create (OCR 728e994 #0).
    let resp = post_json(
        &base,
        "/v1/admin/collections/compact",
        &serde_json::json!({}),
    );
    assert_eq!(
        resp["manifest_committed"], true,
        "snapshot cycle must commit a manifest: {resp}"
    );

    // 冷重启：快照恢复必须保留 quantization=rabitq，而不是按未量化重建；
    // 检查点之后的 WAL 仍照常重放。
    // EN: cold restart: snapshot restore must preserve quantization=rabitq
    // rather than rebuilding unquantized; WAL past the checkpoint still replays.
    drop(child);
    let (child2, base2) = spawn_houyi_server(&wal);
    let child2 = ChildGuard::new(child2);
    wait_for_health_ok(&base2, Duration::from_secs(5));
    // 单节点保留不会截断 WAL，快照恢复失败会静默回退到全量重放；用启动计数
    // 证明本轮确实走了 manifest 快照恢复路径（OCR 2e7e669 #1）。
    // EN: Single-node retention never truncates the WAL, so a broken snapshot
    // restore would silently fall back to a full replay; startup counters prove
    // the manifest snapshot path was actually taken (OCR 2e7e669 #1).
    let metrics = get_text(&base2, "/metrics");
    assert!(
        metric_value(&metrics, "houyidb_graph_snapshot_loads_total") >= 1.0,
        "snapshot restore must run: {metrics}"
    );
    assert_eq!(
        metric_value(&metrics, "houyidb_graph_snapshot_rejects_total"),
        0.0,
        "no snapshot may be rejected: {metrics}"
    );
    let status = get_json(&base2, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(
        status["quantization"], "rabitq",
        "quantization must survive restart: {status}"
    );
    assert_eq!(status["index_type"], "hnsw", "{status}");
    wait_for_collection_ready(&base2, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base2, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![9, 5], "l2 ordering must survive restart");
    assert!((scores[0] - 0.0).abs() < 1e-6, "{scores:?}");
    assert!((scores[1] + 28.0).abs() < 1e-6, "{scores:?}");

    // 同名再声明不同量化（缺省 none）→ 409；metric 必须与创建时一致（l2），
    // 否则会先在 metric 比对处 409，碰不到量化比对。
    // EN: re-requesting the same name with a different quantization (default
    // none) → 409; the metric must match creation time (l2) or the metric
    // check 409s first and the quantization comparison is never reached.
    let (code, _) = put_json_status(
        &base2,
        &format!("/v1/collections/{NAME}"),
        &serde_json::json!({"dims": 4, "index_type": "hnsw", "metric": "l2"}),
    );
    assert_eq!(code, 409);

    // rabitq 只接受 l2/cosine、非 hnsw 组合与非法值：全部在落 WAL 前 400，
    // 重启后不得留下幽灵集合。
    // EN: rabitq accepts only l2/cosine; non-hnsw combos and illegal values are
    // 400 before any WAL record, leaving no ghost collections after restart.
    let (code, resp) = put_json_status(
        &base2,
        "/v1/collections/rabitq-ip",
        &serde_json::json!({
            "dims": 4, "index_type": "hnsw", "metric": "ip",
            "quantization": "rabitq"
        }),
    );
    assert_eq!(code, 400, "rabitq+ip must be rejected: {resp}");
    let (code, _) = put_json_status(
        &base2,
        "/v1/collections/rabitq-l1",
        &serde_json::json!({
            "dims": 4, "index_type": "hnsw", "metric": "l1",
            "quantization": "rabitq"
        }),
    );
    assert_eq!(code, 400);
    let (code, _) = put_json_status(
        &base2,
        "/v1/collections/bad-rabitq",
        &serde_json::json!({
            "dims": 4, "index_type": "hnsw", "quantization": "sq8"
        }),
    );
    assert_eq!(code, 400);
    let (code, _) = put_json_status(
        &base2,
        "/v1/collections/flat-rabitq",
        &serde_json::json!({
            "dims": 4, "index_type": "flat", "quantization": "rabitq"
        }),
    );
    assert_eq!(code, 400);
    for name in ["rabitq-ip", "rabitq-l1", "bad-rabitq", "flat-rabitq"] {
        assert_eq!(
            get_status(&base2, &format!("/v1/collections/{name}/status")),
            404,
            "failed create {name} must not exist on the live server"
        );
    }

    // 非法 create 若漏写 WAL，只有下一次重放才会显形：再冷启一次复验 404
    //（同一次重启也确认快照恢复后的 rabitq 集合仍然完整）。OCR 728e994 #1。
    // EN: A leaked WAL record from an illegal create only materializes on the
    // next replay: cold-start once more and re-assert 404 (the same restart also
    // confirms the snapshot-restored rabitq collection stays intact). OCR
    // 728e994 #1.
    drop(child2);
    let (child3, base3) = spawn_houyi_server(&wal);
    let child3 = ChildGuard::new(child3);
    wait_for_health_ok(&base3, Duration::from_secs(5));
    let metrics = get_text(&base3, "/metrics");
    assert!(
        metric_value(&metrics, "houyidb_graph_snapshot_loads_total") >= 1.0,
        "snapshot restore must run on the second restart: {metrics}"
    );
    assert_eq!(
        metric_value(&metrics, "houyidb_graph_snapshot_rejects_total"),
        0.0,
        "no snapshot may be rejected on the second restart: {metrics}"
    );
    let status = get_json(&base3, &format!("/v1/collections/{NAME}/status"));
    assert_eq!(
        status["quantization"], "rabitq",
        "quantization must survive a second restart: {status}"
    );
    wait_for_collection_ready(&base3, NAME, 2, Duration::from_secs(10));
    let (ids, _, metric) = search_ids_scores(&base3, NAME, [1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![9, 5], "l2 ordering must survive a second restart");
    for name in ["rabitq-ip", "rabitq-l1", "bad-rabitq", "flat-rabitq"] {
        assert_eq!(
            get_status(&base3, &format!("/v1/collections/{name}/status")),
            404,
            "failed create {name} must not appear after replay"
        );
    }

    drop(child3);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 显式创建的参数校验矩阵：非法度量、后端不支持的度量、HNSW 参数残缺/错配后端、
/// 维度冲突与已存在同名的 409，都必须在落 WAL 前失败。
/// EN: explicit-create validation matrix: bad metric, metric unsupported by the
/// backend, partial/misplaced HNSW params, dims conflict, and name collision —
/// all must fail before any WAL record is written.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn explicit_create_validation_matrix() {
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let cases: &[(&str, serde_json::Value, u16)] = &[
        (
            "zero-dims",
            serde_json::json!({"dims": 0, "metric": "l2"}),
            400,
        ),
        (
            "unknown-metric",
            serde_json::json!({"dims": 4, "metric": "bogus"}),
            400,
        ),
        // pq-flat 内部恒为平方 L2，不接受 cosine。
        // EN: pq-flat is squared-L2 internally and rejects cosine.
        (
            "pq-flat-cosine",
            serde_json::json!({"dims": 4, "index_type": "pq-flat", "metric": "cosine"}),
            400,
        ),
        // hii-l0 内部单位化，只接受 cosine。
        // EN: hii-l0 normalizes internally and only accepts cosine.
        (
            "hii-l0-l2",
            serde_json::json!({"dims": 4, "index_type": "hii-l0", "metric": "l2"}),
            400,
        ),
        (
            "partial-hnsw",
            serde_json::json!({"dims": 4, "hnsw_m": 16}),
            400,
        ),
        (
            "hnsw-on-flat",
            serde_json::json!({
                "dims": 4,
                "index_type": "flat",
                "hnsw_m": 16,
                "hnsw_ef_construction": 100,
                "hnsw_ef_search": 32
            }),
            400,
        ),
        (
            "unknown-backend",
            serde_json::json!({"dims": 4, "index_type": "diskann"}),
            400,
        ),
    ];
    for (name, body, want) in cases {
        let (code, resp) = put_json_status(&base, &format!("/v1/collections/{name}"), body);
        assert_eq!(code, *want, "case {name}: {resp}");
    }

    // 失败的创建不得留下任何 collection（WAL 未落记录）。
    // EN: failed creates must leave no collection behind (no WAL record).
    let list = get_json(&base, "/v1/collections");
    assert!(
        list.as_array().expect("array").is_empty(),
        "validation failures must not create collections: {list}"
    );

    // 隐式集合已存在：显式创建同名（同维度或异维度）都按「名字已占用」409。
    // EN: an implicit collection exists, so an explicit create with the same name
    // (same or different dims) is 409 "name already exists".
    let up = post_json(
        &base,
        "/v1/collections/implicit/upsert",
        &serde_json::json!({
            "vectors": [{ "id": 1, "values": [1.0_f32, 0.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(up["inserted"], 1);
    let (code, _) = put_json_status(
        &base,
        "/v1/collections/implicit",
        &serde_json::json!({"dims": 4, "metric": "l2"}),
    );
    assert_eq!(code, 409);
    let (code, _) = put_json_status(
        &base,
        "/v1/collections/implicit",
        &serde_json::json!({"dims": 8, "metric": "l2"}),
    );
    assert_eq!(code, 409);

    drop(child);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// l2 分数语义：`score = 1 - L2²`（不钳制）。使用 flat 后端（精确顺序扫描），
/// 零查询向量对 [1,0,0,0] / [3,4,0,0] 的分数应为 0 / -24。
/// EN: l2 score semantics (`score = 1 - L2²`, unclamped) on the exact flat
/// backend: with a zero query, [1,0,0,0] / [3,4,0,0] score 0 / -24.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn l2_metric_scores_are_unclamped_one_minus_squared_distance() {
    const NAME: &str = "l2-docs";
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    let (code, resp) = put_json_status(
        &base,
        &format!("/v1/collections/{NAME}"),
        &serde_json::json!({"dims": 4, "metric": "l2", "index_type": "flat"}),
    );
    assert_eq!(code, 201, "create response: {resp}");
    let resp: serde_json::Value = serde_json::from_str(&resp).expect("create response JSON");
    assert_eq!(resp["metric"], "l2");
    assert!(resp.get("hnsw").is_none(), "flat must omit hnsw: {resp}");

    let up = post_json(
        &base,
        &format!("/v1/collections/{NAME}/upsert"),
        &serde_json::json!({
            "vectors": [
                { "id": 3, "values": [3.0_f32, 4.0, 0.0, 0.0] },
                { "id": 1, "values": [1.0_f32, 0.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(up["inserted"], 2);
    wait_for_collection_ready(&base, NAME, 2, Duration::from_secs(10));
    let (ids, scores, metric) = search_ids_scores(&base, NAME, [0.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(metric, "l2");
    assert_eq!(ids, vec![1, 3], "ascending squared L2: 1 then 25");
    assert!((scores[0] - 0.0).abs() < 1e-6, "1 - 1², got {scores:?}");
    assert!(
        (scores[1] - (-24.0)).abs() < 1e-6,
        "1 - 25 must stay out of [-1,1], got {scores:?}"
    );

    drop(child);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 消费形态回归：**稳定 id 重索引**（同一批 id 换向量重写）+ 冷重启，覆盖
/// `hnsw` 与 `flat` 两种后端。知识库/bench 的真实同步会反复写同一批 chunk id：
/// 旧 flat 实现按追加处理——`indexed_count` 翻倍、top-k 出现重复 id、旧向量
/// 继续参与排序；HNSW 也可能在重写邻居时孤立第三方节点。本用例把这条真实
/// 工作流固化为门禁：重索引后计数不增、top-k 唯一、旧向量不再命中、新向量
/// 排序正确，冷重启后全部保持。
/// EN: Consumer-shaped regression: **stable-id reindex** (rewriting the same id
/// set with new vectors) plus a cold restart, across both `hnsw` and `flat`.
/// Real knowledge-base/bench syncs rewrite the same chunk ids repeatedly; the
/// old flat implementation appended (doubling `indexed_count`, repeating ids
/// in top-k, stale vectors still ranked), and HNSW could orphan third-party
/// nodes when rewriting their neighbors. This pins that real workflow: counts
/// do not grow, top-k ids stay unique, stale vectors disappear, new vectors
/// rank correctly, and everything survives a cold restart.
#[test]
#[cfg_attr(not(unix), ignore = "uses std::process::Child::kill (unix)")]
fn stable_id_reindex_is_idempotent_across_backends_and_restart() {
    const N: u64 = 64;
    const COLLECTIONS: [(&str, &str); 2] = [("reindex-hnsw", "hnsw"), ("reindex-flat", "flat")];
    let dir = unique_test_dir();
    let _ = std::fs::create_dir_all(&dir);
    let wal = dir.join("bench.wal");
    clear_wal_artifacts(&wal);

    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));

    for (name, index_type) in COLLECTIONS {
        let (code, resp) = put_json_status(
            &base,
            &format!("/v1/collections/{name}"),
            &serde_json::json!({"dims": 4, "metric": "l2", "index_type": index_type}),
        );
        assert_eq!(code, 201, "create {name}: {resp}");

        // 第一版：id i → [i, 0, 0, 1]。
        // EN: First version: id i -> [i, 0, 0, 1].
        let rows: Vec<serde_json::Value> = (0..N)
            .map(|i| serde_json::json!({"id": i, "values": [i as f32, 0.0_f32, 0.0, 1.0]}))
            .collect();
        let up = post_json(
            &base,
            &format!("/v1/collections/{name}/upsert"),
            &serde_json::json!({ "vectors": rows }),
        );
        assert_eq!(up["inserted"], N, "{name}: first load inserted");
        wait_for_collection_ready(&base, name, N, Duration::from_secs(10));

        // 重索引：同一批 id 换向量 → [0, i, 0, 1]（真实同步的更新形态）。
        // EN: Reindex: same ids, new vectors -> [0, i, 0, 1].
        let rows: Vec<serde_json::Value> = (0..N)
            .map(|i| serde_json::json!({"id": i, "values": [0.0_f32, i as f32, 0.0, 1.0]}))
            .collect();
        let up = post_json(
            &base,
            &format!("/v1/collections/{name}/upsert"),
            &serde_json::json!({ "vectors": rows }),
        );
        assert_eq!(up["inserted"], N, "{name}: reindex inserted");
        wait_for_collection_ready(&base, name, N, Duration::from_secs(10));
        let status = get_json(&base, &format!("/v1/collections/{name}/status"));
        assert_eq!(
            status["indexed_count"], N,
            "{name}: reindex must replace, not append: {status}"
        );

        // 旧向量不得再命中：用 v1 的 id=63 向量查询，重索引后它应完全退出 top-k。
        // EN: The stale vector must be gone: query with v1's id=63 vector; after
        // reindexing it must drop out of the top-k entirely.
        let (ids, _, metric) = search_ids_scores(&base, name, [63.0, 0.0, 0.0, 1.0], 10);
        assert_eq!(metric, "l2");
        assert!(
            !ids.contains(&63),
            "{name}: stale vector still ranked: {ids:?}"
        );
        assert_eq!(
            ids.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            ids.len(),
            "{name}: duplicate ids in top-k after reindex: {ids:?}"
        );

        // 新向量排序：用 v2 的 id=63 向量查询，id=63 必须排第一且结果唯一。
        // EN: New-vector ordering: query with v2's id=63 vector; id=63 must
        // rank first and the result set stays unique.
        let (ids, _, _) = search_ids_scores(&base, name, [0.0, 63.0, 0.0, 1.0], 10);
        assert_eq!(
            ids.first().copied(),
            Some(63),
            "{name}: new vector not ranked first: {ids:?}"
        );
        assert_eq!(
            ids.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            ids.len(),
            "{name}: duplicate ids after reindex: {ids:?}"
        );
    }

    // 冷重启：WAL 重放 / 快照恢复必须保持「每 id 一行」的替换语义。
    // EN: Cold restart: WAL replay / snapshot restore must preserve per-id
    // replacement (one row per id).
    drop(child);
    let (child, base) = spawn_houyi_server(&wal);
    let child = ChildGuard::new(child);
    wait_for_health_ok(&base, Duration::from_secs(5));
    for (name, _) in COLLECTIONS {
        wait_for_collection_ready(&base, name, N, Duration::from_secs(10));
        let status = get_json(&base, &format!("/v1/collections/{name}/status"));
        assert_eq!(
            status["indexed_count"], N,
            "{name}: restart must not duplicate rows: {status}"
        );
        let (ids, _, _) = search_ids_scores(&base, name, [0.0, 63.0, 0.0, 1.0], 10);
        assert_eq!(
            ids.first().copied(),
            Some(63),
            "{name}: new vector lost after restart: {ids:?}"
        );
        assert_eq!(
            ids.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            ids.len(),
            "{name}: duplicate ids after restart: {ids:?}"
        );
    }

    drop(child);
    clear_wal_artifacts(&wal);
    let _ = std::fs::remove_dir_all(&dir);
}
