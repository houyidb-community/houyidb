//! OpenAI 兼容 `/v1/embeddings` 网关：透明代理 + 可选自动入库。
//! OpenAI-compatible `/v1/embeddings` gateway: transparent proxy + optional auto-ingest.
//!
//! 设计定位（为什么是这个形态）：
//! - HouyiDB 是向量数据库，内核**没有** embedding 推理能力；网关把 OpenAI 形状的
//!   请求转发到配置的上游（OpenAI / Ollama / vLLM 等任意兼容端点），自己不算向量。
//! - 不带 `collection` 字段时是完全透明的代理：请求体原样转发（未知字段不丢），
//!   上游响应（含错误状态码与正文）原样回传——"把 base_url 从 OpenAI 换成
//!   HouyiDB"即可工作。
//! - 带 `collection`（HouyiDB 扩展字段，OpenAI SDK 经 `extra_body` 传入）时，
//!   网关把上游返回的 embedding 顺手 upsert 进该 collection：id 取
//!   `stable_id(text)`（与 Python SDK 同一 blake2b-8 方案，重跑幂等），原文进
//!   payload `{"text": ...}`（与两个 adapter 的 text_key 约定一致），响应里注入
//!   `houyi: {collection, upserted}` 扩展字段（OpenAI 客户端忽略未知字段，安全）。
//! - 自动入库要求 input 是字符串（数组）；token 数组无法还原原文，显式 400。

use crate::http_handler::handle_collection_upsert;
use crate::types::{BenchmarkHttpState, UpsertRequest, UpsertVector};
use axum::extract::{Path, State};
use axum::Json;
use blake2::digest::consts::U8;
use blake2::digest::Digest;

/// 单次请求的 input 条数上限（与 OpenAI 的 2048 对齐，防网关被超大批次当漏斗）。
/// EN: Per-request input item cap (aligned with OpenAI's 2048; prevents the
/// gateway from being used as a funnel for oversized batches).
const MAX_INPUT_ITEMS: usize = 2048;

/// u64 正整数键空间的上半位留空（与 Python SDK `_U64_MASK` 同口径）。
/// EN: Leave the upper half of the u64 positive key space empty (same
/// convention as the Python SDK's `_U64_MASK`).
const U64_MASK: u64 = (1 << 63) - 1;

type Blake2b8 = blake2::Blake2b<U8>;

/// 与 Python SDK `houyidb.stable_id` 完全一致的确定性哈希：
/// blake2b(text, digest_size=8) 大端 u64 & (2^63-1)。两侧一致才能跨 SDK 重放删除。
/// EN: Deterministic hash identical to the Python SDK's `houyidb.stable_id`:
/// blake2b(text, digest_size=8) as big-endian u64 & (2^63-1). Both sides must
/// agree so cross-SDK delete replays land on the same ids.
pub fn stable_id(text: &str) -> u64 {
    let mut h = Blake2b8::new();
    h.update(text.as_bytes());
    let digest = h.finalize();
    u64::from_be_bytes(digest.into()) & U64_MASK
}

type HttpResult<T> = Result<T, (axum::http::StatusCode, String)>;

/// `POST /v1/embeddings` — OpenAI 兼容嵌入端点。
/// EN: `POST /v1/embeddings` — OpenAI-compatible embeddings endpoint.
pub async fn handle_openai_embeddings(
    State(state): State<BenchmarkHttpState>,
    Json(body): Json<serde_json::Value>,
) -> HttpResult<Json<serde_json::Value>> {
    let upstream = state.openai_upstream.clone().ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            "openai embeddings gateway is not configured \
             (set --openai-upstream / HOUYI_OPENAI_UPSTREAM)"
                .to_string(),
        )
    })?;

    // 网关扩展字段不进上游；其余字段原样转发（未知字段透明——这是"兼容"的本义）。
    // 先借后取：扩展字段与待入库原文从借用中提取，再拿走 body 所有权删字段——
    // 避免为删一个字段深拷贝整个请求体（ingress 上限 64MB）。
    // EN: Gateway-only fields never go upstream; everything else is forwarded
    // verbatim (unknown fields pass through — that is the point of "compatible").
    // Borrow before taking: extract extension fields and ingest texts from the
    // borrow, then take ownership of body and delete fields — avoids deep-copying
    // the whole request just to remove one field (ingress cap is 64MB).
    let collection = body
        .get("collection")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    // 只读副本（P4.4.1 修正）：带 collection 的嵌入请求最终要入库——follower
    // 在上游调用之前就 503，不为一笔必拒的写烧上游配额与秒级延迟。
    // 纯代理模式（无 collection 字段）在 follower 上保持可用。
    // EN: A embeddings request carrying `collection` ends in a write — reject it
    // on followers before the billable upstream call; pure proxy mode (no
    // collection field) stays available on followers.
    if collection.is_some() {
        crate::http_handler::reject_if_read_only(&state)?;
    }

    // 自动入库只在需要时做形状校验：input 必须是字符串/字符串数组（token 数组
    // 无法还原原文）。纯代理路径不校验，把校验权完全留给上游。
    // EN: Auto-ingest validates shape only when needed: input must be a string or
    // string array (token arrays cannot recover the original text). The pure proxy
    // path does not validate and leaves that entirely to the upstream.
    let texts: Option<Vec<String>> = if collection.is_some() {
        Some(extract_texts(&body)?)
    } else {
        None
    };

    let mut forward = body;
    if let Some(obj) = forward.as_object_mut() {
        obj.remove("collection");
    }

    let url = format!("{}/v1/embeddings", upstream.base_url.trim_end_matches('/'));
    let upstream_cfg = upstream.clone();
    // 熔断准入（P4.3.2→P4.3.3）：放在所有本地校验之后、spawn_blocking 之前——
    // 准入后每一条出路都会回报 on_success/on_failure，探测名额不会卡死；
    // Open/HalfOpen 时 503 快速失败，不把压力继续倒给病了的上游。
    // EN: Breaker admission after all local validation, right before the blocking
    // upstream call — every path past this point reports back, so the half-open
    // probe slot can never get stuck; open/half-open fast-fails 503 instead of
    // piling onto a sick upstream.
    let breaker = upstream.breaker.clone();
    let epoch = match breaker.admit() {
        Ok(epoch) => epoch,
        // 快速失败文案带上拒绝那一刻的状态（admit 判定与文案同一份快照）：
        // Open（冷却中）与 HalfOpen（探测在途）都表现为 503，但运维含义
        // 不同——与 houyidb_openai_breaker_state 指标对齐，避免 flapping
        // 期间的误导性诊断。
        // EN: The fast-fail message names the state at the instant of
        // rejection (same snapshot as the decision) — Open (cooling) and
        // HalfOpen (probe in flight) both surface as 503 but mean different
        // things to an operator correlating against the breaker-state metric.
        Err(state) => {
            // 用 breaker 导出的状态常量匹配（编码单一定义点）；未知状态
            // 如实渲染编号，不冒充 open——错标比未知更糟。
            // EN: Match on the breaker's exported state constants (single
            // definition point); unknown states render as-is rather than
            // masquerading as open — a wrong label is worse than none.
            let state_word = match state {
                crate::breaker::STATE_HALF_OPEN => "half-open (probe in flight)".to_string(),
                crate::breaker::STATE_OPEN => "open".to_string(),
                other => format!("in unknown state {other}"),
            };
            return Err((
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "openai upstream circuit breaker is {state_word}; request fast-failed \
                     without touching the upstream"
                ),
            ));
        }
    };
    // ureq 是同步客户端：整个调用（发送 + 响应体读取 + JSON 解析）都放进
    // spawn_blocking——只把 send_json 放进去会把 into_json/into_string 的阻塞
    // socket 读留在 async worker 上（有界仅由超时兜底，并发请求足以拖垮执行器）。
    // Agent 共享（连接池），避免每请求完整握手。不引入 reqwest/hyper 二套 HTTP 栈。
    // 熔断回报直接走 upstream_cfg.breaker（config 自带 Arc 共享），
    // 不再为闭包单克隆一个句柄——join error 路径用外层那一个。
    // EN: ureq is synchronous: the whole call (send + response read + JSON parsing)
    // goes into spawn_blocking — only send_json would leave the blocking
    // into_json/into_string socket reads on an async worker (bounded only by the
    // timeout; concurrent requests can exhaust the executor). The Agent is shared
    // (connection pool) to avoid a full handshake per request; no second
    // reqwest/hyper HTTP stack. The breaker report goes straight through
    // upstream_cfg.breaker (config already shares the Arc) — no extra handle clone
    // for the closure; the join-error path uses the outer one.
    let resp: HttpResult<serde_json::Value> = tokio::task::spawn_blocking(move || {
        let mut call = upstream_cfg
            .agent
            .post(&url)
            .header("content-type", "application/json");
        if let Some(key) = &upstream_cfg.api_key {
            call = call.header("authorization", &format!("Bearer {key}"));
        }
        match call.send_json(forward) {
            // Agent 配置了 http_status_as_error(false)：2xx 与 4xx/5xx 都走这里。
            // EN: The Agent sets http_status_as_error(false): both 2xx and 4xx/5xx land here.
            Ok(mut r) => {
                let status = r.status();
                if status.is_success() {
                    match r.body_mut().read_json() {
                        Ok(v) => {
                            upstream_cfg.breaker.on_success(epoch);
                            Ok(v)
                        }
                        // 2xx 但响应体非 JSON：上游违约（病了），计熔断失败。
                        // EN: 2xx but the body is not JSON: the upstream broke its contract
                        // (sick) — count a breaker failure.
                        Err(e) => {
                            upstream_cfg.breaker.on_failure(epoch, true);
                            Err((
                                axum::http::StatusCode::BAD_GATEWAY,
                                format!("upstream returned non-JSON body: {e}"),
                            ))
                        }
                    }
                } else {
                    // 上游业务错误（4xx/5xx）：状态码与正文原样透传，不包信封。
                    // 正文读取失败时也要给出诊断，而不是哑默的空串。
                    // 熔断口径：5xx 计失败（上游病了）；4xx 是请求形状问题——
                    // Closed 下不计也不清零，HalfOpen 探测收到 4xx 视为探测通过
                    // （上游答了 = 活着，见 breaker 模块文档）。
                    // EN: 5xx counts; 4xx neither counts nor resets while Closed,
                    // and resolves a HalfOpen probe as alive.
                    let code = status.as_u16();
                    upstream_cfg.breaker.on_failure(epoch, code >= 500);
                    let text = r
                        .body_mut()
                        .read_to_string()
                        .unwrap_or_else(|e| format!("(failed to read upstream error body: {e})"));
                    Err((
                        axum::http::StatusCode::from_u16(code)
                            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY),
                        text,
                    ))
                }
            }
            Err(e) => {
                upstream_cfg.breaker.on_failure(epoch, true);
                Err((
                    axum::http::StatusCode::BAD_GATEWAY,
                    format!("upstream call failed: {e}"),
                ))
            }
        }
    })
    .await
    .map_err(|e| {
        // 阻塞任务 panic/取消：视同传输失败计熔断（准入已发，必须回报，
        // 否则半开探测名额卡死）。
        // EN: A panicked/cancelled blocking task counts as a transport-class
        // failure — admission was granted, so a report is mandatory or the
        // half-open probe slot would stick.
        breaker.on_failure(epoch, true);
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("openai gateway worker join error: {e}"),
        )
    })?;

    let mut resp_body = resp?;

    if let (Some(name), Some(texts)) = (collection, texts) {
        let embeddings = extract_embeddings(&resp_body, texts.len())?;
        let vectors: Vec<UpsertVector> = texts
            .iter()
            .zip(embeddings)
            .map(|(text, values)| UpsertVector {
                id: stable_id(text),
                values,
                label: None,
                payload: Some(
                    serde_json::json!({"text": text})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
                // 嵌入网关不产生稀疏词项（无分词器）。
                // EN: The embedding gateway produces no sparse terms (no tokenizer).
                sparse: None,
            })
            .collect();
        // 复用 collection upsert handler 的全部分支（保留键校验/维度定型/标量落页/
        // 双速写入），不在网关里另写一条入库路径。
        // EN: Reuse every branch of the collection upsert handler (reserved-key checks /
        // dimension fixing / scalar page writes / dual-speed writes) instead of writing a
        // second ingest path in the gateway.
        let Json(upserted) = handle_collection_upsert(
            State(state.clone()),
            Path(name.clone()),
            Json(UpsertRequest {
                vectors,
                index: None,
            }),
        )
        .await?;
        if let Some(obj) = resp_body.as_object_mut() {
            obj.insert(
                "houyi".to_string(),
                serde_json::json!({"collection": name, "upserted": upserted.inserted}),
            );
        }
    }

    Ok(Json(resp_body))
}

/// 从请求体提取待入库原文：string | [string]；空数组与超上限显式 400。
/// EN: Extract texts to ingest from the request body: string | [string]; empty
/// arrays and over-cap sizes get an explicit 400.
fn extract_texts(body: &serde_json::Value) -> HttpResult<Vec<String>> {
    match body.get("input") {
        Some(serde_json::Value::String(s)) => Ok(vec![s.clone()]),
        Some(serde_json::Value::Array(arr)) => {
            if arr.is_empty() {
                return Err((
                    axum::http::StatusCode::BAD_REQUEST,
                    "input must not be an empty array".to_string(),
                ));
            }
            if arr.len() > MAX_INPUT_ITEMS {
                return Err((
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("input exceeds the {MAX_INPUT_ITEMS}-item limit"),
                ));
            }
            arr.iter()
                .map(|v| {
                    v.as_str().map(str::to_string).ok_or_else(|| {
                        (
                            axum::http::StatusCode::BAD_REQUEST,
                            "auto-ingest requires string input(s); token arrays are not \
                             supported (drop the 'collection' field for a transparent proxy)"
                                .to_string(),
                        )
                    })
                })
                .collect()
        }
        _ => Err((
            axum::http::StatusCode::BAD_REQUEST,
            "input is required and must be a string or an array of strings".to_string(),
        )),
    }
}

/// 从上游响应提取 embedding 矩阵并按 index 对齐；形状不符一律 502（上游违约）。
/// EN: Extract the embedding matrix from the upstream response and align by index;
/// any shape mismatch is a 502 (upstream breach).
fn extract_embeddings(body: &serde_json::Value, expected: usize) -> HttpResult<Vec<Vec<f32>>> {
    let bad = |msg: &str| {
        (
            axum::http::StatusCode::BAD_GATEWAY,
            format!("upstream returned malformed embeddings: {msg}"),
        )
    };
    let data = body
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| bad("missing 'data' array"))?;
    if data.len() != expected {
        return Err(bad(&format!(
            "'data' has {} item(s), expected {expected}",
            data.len()
        )));
    }
    let mut by_index: Vec<Option<Vec<f32>>> = vec![None; expected];
    for item in data {
        let index = item
            .get("index")
            .and_then(|i| i.as_u64())
            .ok_or_else(|| bad("missing 'index'"))? as usize;
        if index >= expected {
            return Err(bad(&format!("index {index} out of range")));
        }
        let embedding = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| bad("missing 'embedding' array"))?;
        let mut values = Vec::with_capacity(embedding.len());
        for v in embedding {
            let f = v.as_f64().ok_or_else(|| bad("non-numeric component"))? as f32;
            if !f.is_finite() {
                return Err(bad("non-finite component"));
            }
            values.push(f);
        }
        by_index[index] = Some(values);
    }
    by_index
        .into_iter()
        .enumerate()
        .map(|(i, v)| v.ok_or_else(|| bad(&format!("missing index {i}"))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与 Python SDK `houyidb.stable_id` 的跨语言对拍：这些常量由 Python 侧
    /// `hashlib.blake2b(text, digest_size=8)` 产出，两侧必须恒等，否则跨 SDK
    /// 的 delete 重放会静默打偏。
    /// EN: Cross-language parity with the Python SDK's `houyidb.stable_id`: these
    /// constants come from Python's `hashlib.blake2b(text, digest_size=8)`; both
    /// sides must be identical, or cross-SDK delete replays silently miss.
    #[test]
    fn stable_id_matches_python_sdk() {
        assert_eq!(stable_id("x"), 5395104992458594383);
        assert_eq!(stable_id("doc-1"), 6265767295393789249);
        assert_eq!(stable_id(""), 7252660547403494068);
        assert_eq!(stable_id("apple is a fruit"), 4470867300576208512);
        assert!(stable_id("x") < (1 << 63));
    }
}
