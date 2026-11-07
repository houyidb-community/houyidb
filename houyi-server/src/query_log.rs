//! 查询日志（EE M1 数据源）：有界 ring buffer + 聚合统计。
//! Query log (EE M1 data source): bounded ring buffer plus aggregate stats.
//!
//! 只记录**元数据**（时间、主体、路由、collection、k、状态码、延迟、命中数、
//! 是否带过滤）——向量、payload、SQL 原文一律不入日志。容量 0 = 关闭，写路径
//! 零开销（`record` 的三处调用点都先判断 `Option`）。
//! EN: metadata only — timestamps, principal, route, collection, k, status,
//! latency, hit count and a filter flag. Vectors, payloads and raw SQL are
//! never recorded. Capacity 0 disables the log; write paths check the Option
//! first, so the disabled case costs nothing.
//!
//! 并发模型：单个 `Mutex<VecDeque>` 是有意的取舍（与「禁止全局 Mutex」铁律
//! 的关系）——本路径不在核心计算内，且仅在运维显式开启时存在；临界区只做
//! 一次 push/pop，不触碰索引与 WAL。若将来成为热点，按 collection 分片即可，
//! 对外形状不变。
//! EN: the single mutex is a deliberate trade-off: this path is not on the
//! core compute path, only exists when explicitly enabled, and the critical
//! section is one push/pop that never touches the index or WAL. If it ever
//! becomes hot, shard by collection without changing the public shape.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// 单条查询记录。 EN: One recorded query.
#[derive(Debug, Clone, Serialize)]
pub struct QueryLogEntry {
    /// Unix 毫秒时间戳。 EN: Unix milliseconds.
    pub ts_ms: u64,
    /// 进程内单调记录 id（采样后仍连续，便于定位丢弃）。 EN: Monotonic record id.
    pub request_id: u64,
    /// 主体名（匿名 = "-"）；永不记录 token。 EN: Principal name ("-" = anonymous).
    pub principal: String,
    /// 路由标签：collection_search / hybrid_search / benchmark_search / pgwire。
    /// EN: route label.
    pub route: &'static str,
    pub collection: Option<String>,
    pub k: usize,
    /// HTTP 状态码；pgwire 成功记 200，失败按 SQLSTATE 归类为 400/500。
    /// EN: HTTP status; pgwire records 200 on success and 400/500 (by SQLSTATE
    /// class) on failure.
    pub status: u16,
    pub latency_ms: f64,
    pub result_count: usize,
    /// 是否携带标量过滤。 EN: Whether a scalar filter was applied.
    pub filtered: bool,
}

/// 每 collection 的聚合统计。 EN: Per-collection aggregate stats.
#[derive(Debug, Clone, Serialize)]
pub struct CollectionQueryStats {
    pub collection: String,
    pub count: u64,
    pub errors: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// 查询统计视图。 EN: Query stats view.
#[derive(Debug, Clone, Serialize)]
pub struct QueryStats {
    pub recorded_total: u64,
    pub dropped_total: u64,
    /// 当前缓冲区内样本数（统计口径 = 现存条目，容量有界）。
    /// EN: samples currently buffered (stats are computed over the live buffer).
    pub buffered: usize,
    pub by_collection: Vec<CollectionQueryStats>,
}

/// `/v1/admin/query-log` 查询参数。 EN: Query-string filters for the log endpoint.
#[derive(Debug, Default, Deserialize)]
pub struct QueryLogQuery {
    /// 只返回 ts_ms >= since 的记录。 EN: Only entries at or after this timestamp.
    pub since: Option<u64>,
    pub collection: Option<String>,
    /// 只返回 latency_ms >= min_ms 的记录。 EN: Only entries at or above this latency.
    pub min_ms: Option<f64>,
    /// 返回条数上限（默认 100，硬上限 = 容量）。 EN: Limit (default 100).
    pub limit: Option<usize>,
}

/// 有界查询日志。 EN: Bounded query log.
pub struct QueryLog {
    capacity: usize,
    sample_every: u32,
    seq: AtomicU64,
    recorded_total: AtomicU64,
    dropped_total: AtomicU64,
    entries: Mutex<VecDeque<QueryLogEntry>>,
}

impl QueryLog {
    /// `capacity` 必须 > 0（调用方在配置校验期保证）；`sample_every` = 0 视为 1。
    /// EN: capacity must be > 0 (validated at startup); sample_every 0 means 1.
    pub fn new(capacity: usize, sample_every: u32) -> Self {
        Self {
            capacity: capacity.max(1),
            sample_every: sample_every.max(1),
            seq: AtomicU64::new(1),
            recorded_total: AtomicU64::new(0),
            dropped_total: AtomicU64::new(0),
            entries: Mutex::new(VecDeque::with_capacity(capacity.clamp(1, 4096))),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 记录一条查询；采样按全局序号（1/N），被采样掉的也计入 request_id 连续段。
    /// min_ms 过滤、状态与命中数由调用方填好。
    /// EN: record one query; sampling is by global sequence (1-in-N), and skipped
    /// queries still consume a request id so ids stay traceable.
    pub fn record(&self, entry: QueryLogEntry) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        if !seq.is_multiple_of(u64::from(self.sample_every)) {
            return;
        }
        let entry = QueryLogEntry {
            request_id: seq,
            ..entry
        };
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if entries.len() >= self.capacity {
            entries.pop_front();
            self.dropped_total.fetch_add(1, Ordering::Relaxed);
        }
        entries.push_back(entry);
        self.recorded_total.fetch_add(1, Ordering::Relaxed);
    }

    /// 按条件读取最近记录（新→旧），最多 `limit` 条（默认 100）。
    /// EN: read recent entries (newest first), at most `limit` (default 100).
    pub fn recent(&self, query: &QueryLogQuery) -> Vec<QueryLogEntry> {
        let limit = query.limit.unwrap_or(100).min(self.capacity);
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        entries
            .iter()
            .rev()
            .filter(|e| query.since.is_none_or(|since| e.ts_ms >= since))
            .filter(|e| {
                query
                    .collection
                    .as_deref()
                    .is_none_or(|name| e.collection.as_deref() == Some(name))
            })
            .filter(|e| query.min_ms.is_none_or(|min| e.latency_ms >= min))
            .take(limit)
            .cloned()
            .collect()
    }

    /// 按 collection 聚合当前缓冲区（count/errors/p50/p95/p99）。
    /// EN: aggregate the live buffer per collection (count/errors/p50/p95/p99).
    pub fn stats(&self) -> QueryStats {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut buckets: BTreeMap<String, (u64, u64, Vec<f64>)> = BTreeMap::new();
        for entry in entries.iter() {
            let key = entry
                .collection
                .clone()
                .unwrap_or_else(|| "<global>".to_string());
            let bucket = buckets.entry(key).or_insert_with(|| (0, 0, Vec::new()));
            bucket.0 += 1;
            if entry.status >= 400 {
                bucket.1 += 1;
            }
            bucket.2.push(entry.latency_ms);
        }
        let by_collection = buckets
            .into_iter()
            .map(|(collection, (count, errors, mut latencies))| {
                latencies.sort_by(|a, b| a.total_cmp(b));
                CollectionQueryStats {
                    collection,
                    count,
                    errors,
                    p50_ms: percentile(&latencies, 0.50),
                    p95_ms: percentile(&latencies, 0.95),
                    p99_ms: percentile(&latencies, 0.99),
                }
            })
            .collect();
        QueryStats {
            recorded_total: self.recorded_total.load(Ordering::Relaxed),
            dropped_total: self.dropped_total.load(Ordering::Relaxed),
            buffered: entries.len(),
            by_collection,
        }
    }
}

/// 最近秩百分位（空集合 = 0）。索引向上取整，样本少时 p95 == max。
/// EN: nearest-rank percentile (empty = 0); with few samples p95 = max.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[rank]
}

/// 当前 Unix 毫秒（墙钟只用于展示，不参与租约/排序判定）。
/// EN: current Unix milliseconds (wall clock for display only).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(collection: Option<&str>, latency_ms: f64, status: u16) -> QueryLogEntry {
        QueryLogEntry {
            ts_ms: now_ms(),
            request_id: 0,
            principal: "tester".to_string(),
            route: "collection_search",
            collection: collection.map(str::to_string),
            k: 10,
            status,
            latency_ms,
            result_count: 3,
            filtered: false,
        }
    }

    #[test]
    fn ring_buffer_evicts_and_counts_drops() {
        let log = QueryLog::new(3, 1);
        for i in 0..5 {
            log.record(entry(Some("docs"), i as f64, 200));
        }
        let recent = log.recent(&QueryLogQuery::default());
        assert_eq!(recent.len(), 3);
        // 最新在前：5 条里保留最后 3 条（延迟 2/3/4）。
        // EN: Newest first: only the last three of five survive (latency 2/3/4).
        assert_eq!(recent[0].latency_ms, 4.0);
        assert_eq!(recent[2].latency_ms, 2.0);
        let stats = log.stats();
        assert_eq!(stats.buffered, 3);
        assert_eq!(stats.dropped_total, 2);
        assert_eq!(stats.recorded_total, 5);
    }

    #[test]
    fn sampling_records_one_in_n_but_keeps_ids_monotonic() {
        let log = QueryLog::new(16, 2);
        for _ in 0..6 {
            log.record(entry(Some("docs"), 1.0, 200));
        }
        let recent = log.recent(&QueryLogQuery::default());
        assert_eq!(recent.len(), 3);
        // 采样保留偶数序号（2/4/6），request_id 连续可追溯。
        // EN: Sampling keeps the even sequence numbers (2/4/6); request ids stay
        // traceable.
        assert_eq!(
            recent.iter().map(|e| e.request_id).collect::<Vec<_>>(),
            vec![6, 4, 2]
        );
    }

    #[test]
    fn filters_and_percentiles() {
        let log = QueryLog::new(100, 1);
        log.record(entry(Some("a"), 1.0, 200));
        log.record(entry(Some("a"), 9.0, 500));
        log.record(entry(Some("b"), 4.0, 200));
        let recent = log.recent(&QueryLogQuery {
            collection: Some("a".to_string()),
            min_ms: Some(2.0),
            ..Default::default()
        });
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].latency_ms, 9.0);
        let stats = log.stats();
        let a = stats
            .by_collection
            .iter()
            .find(|s| s.collection == "a")
            .expect("collection a");
        assert_eq!(a.count, 2);
        assert_eq!(a.errors, 1);
        assert_eq!(a.p50_ms, 1.0);
        assert_eq!(a.p95_ms, 9.0);
    }
}
