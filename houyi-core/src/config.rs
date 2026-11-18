//! 统一配置系统：集中管理所有 `HOUYI_*` 环境变量，在进程启动时解析一次。
//! Centralized configuration: all `HOUYI_*` env vars parsed once at startup.
//!
//! 各 crate 通过 `&Config` 传参，不再各自读取 `std::env::var`。

/// 解析布尔型环境变量（`"1"`/`"true"`/`"yes"`/`"on"` → true；`"0"`/`"false"`/`"no"`/`"off"` → false）。
/// Parse boolean env var; case-insensitive.
pub fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(s) if s.is_empty() => default,
        Ok(s) => {
            let s = s.to_ascii_lowercase();
            match s.as_str() {
                "0" | "false" | "no" | "off" => false,
                "1" | "true" | "yes" | "on" => true,
                _ => default,
            }
        }
        Err(_) => default,
    }
}

/// 解析允许 `0` 的 `usize` 环境变量：`0` 是有意义的配置值（如「关闭自适应」），
/// 不做正数过滤；缺失或非法时返回 `default`。
/// EN: Parses a `usize` env var where `0` is a meaningful value (e.g. "disable
/// adaptation"), without the positive-only filter; falls back to `default` when
/// absent or malformed.
fn env_usize_allow_zero(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// 解析 `usize` 环境变量，不存在或为 `0` 时返回 `default`。
/// 由 `env_usize_allow_zero` 派生，避免两个解析器在「缺失/非法」语义上漂移。
/// EN: Parses a `usize` env var; returns `default` when unset or zero. Derived
/// from `env_usize_allow_zero` so the two parsers cannot drift on
/// absent/malformed handling.
pub fn env_usize(name: &str, default: usize) -> usize {
    let n = env_usize_allow_zero(name, default);
    if n > 0 {
        n
    } else {
        default
    }
}

/// 解析可选 `usize`（不存在 → None）。
/// EN: Parses an optional `usize` (absent → None).
pub fn env_usize_optional(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
}

// ── HNSW 图参数 ────────────────────────────────────────────
// EN: ── HNSW graph parameters ──────────────────────────────

#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// 每层最大邻接数。
    /// EN: Max neighbors per layer (M).
    pub m: usize,
    /// 构图 greedy 搜索列表宽度。
    /// EN: Build-time beam width.
    pub ef_construction: usize,
    /// 查询 beam 宽度。
    /// EN: Query-time beam width.
    pub ef_search: usize,
    /// 最大层数上限。
    /// EN: Max possible layers.
    pub max_layers: usize,
    /// 随机种子。
    /// EN: RNG seed for level assignment.
    pub seed: u64,
    /// Level 0 连接数乘数。2.0 = 2M（默认，召回优先，兼容旧图快照）；
    /// 1.0 = M（高 QPS，需显式设置 `HOUYI_HNSW_LAYER0_MULT=1`）。
    /// Level-0 degree multiplier. 2.0 = 2M (default, recall-first,
    /// snapshot-compatible); 1.0 = M (higher QPS, opt-in via env).
    pub layer0_mult: f64,
    /// 上层连接数乘数。
    /// EN: Upper-layer degree multiplier.
    pub upper_mult: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 24,
            ef_construction: 400,
            ef_search: 512,
            max_layers: 10,
            seed: 0xA7E5_72D8_B7DB_BEEF,
            layer0_mult: 2.0,
            upper_mult: 0.5,
        }
    }
}

impl HnswConfig {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        c.max_layers = env_usize("HOUYI_HNSW_MAX_LAYERS", 10).clamp(1, 16);
        c.seed = std::env::var("HOUYI_HNSW_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(c.seed);
        c.layer0_mult = std::env::var("HOUYI_HNSW_LAYER0_MULT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&f| f > 0.0)
            .unwrap_or(c.layer0_mult);
        c.upper_mult = std::env::var("HOUYI_HNSW_UPPER_MULT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&f| f > 0.0)
            .unwrap_or(c.upper_mult);
        c
    }
}

// ── 批量构图参数 ────────────────────────────────────────────
// EN: ── Batch build parameters ─────────────────────────────

#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// 单次 drain 最大向量数。
    /// EN: Max vectors per drain (clamped to [64, 256*1024]).
    pub max_build_batch: usize,
    /// 一次性取走队列全部待构图项。
    /// EN: Drain entire pending queue in one shot.
    pub optimize_drain_all: bool,
    /// 打印构图阶段耗时分解。
    /// EN: Print per-stage timing breakdown.
    pub breakdown_enabled: bool,
    /// 收集精确边数量指标（遍历全图）。
    /// EN: Collect exact edge counts (full graph scan).
    pub collect_edge_metrics: bool,
    /// 启用并行批量构图。
    /// EN: Enable parallel batch HNSW build.
    pub parallel_batch_build: bool,
    /// 并行构图最小批大小。
    /// EN: Min batch size for parallel path.
    pub parallel_min_batch: usize,
    /// 并行构图前图内最小点数。
    /// EN: Min graph size before parallel is eligible.
    pub parallel_min_graph: usize,
    /// 并行 chunk 大小。
    /// EN: Parallel planning chunk size.
    pub parallel_chunk_size: usize,
    /// 并行构图最小 chunk 大小。
    /// EN: Min chunk to use parallel (else serial).
    pub parallel_min_chunk: usize,
    /// 每组 chunk 数量（一次提交）。
    /// EN: Chunks per apply group.
    pub parallel_plan_group_chunks: usize,
    /// 规划组宽自适应的图规模除数：自适应上限 = `max(plan_group_min, 图点数/该值)`，
    /// 最终组宽再与配置组宽取 min；0 = 关闭自适应，始终使用
    /// `chunk × group_chunks` 的固定组宽。
    /// EN: Graph-size divisor for adaptive plan-group width: the adaptive cap is
    /// `max(plan_group_min, graph_len / divisor)` and the final width is the
    /// minimum with the configured width; 0 disables adaptation and keeps the
    /// fixed `chunk × group_chunks` width.
    pub plan_group_graph_divisor: usize,
    /// 自适应上限的下限：`max(该值, 图点数/divisor)` 作为上限，再与配置组宽取
    /// min——若配置组宽本身更小，最终组宽仍可低于该值。计划/应用路径在图上点数
    /// 达到 `seed_serial_points` 之前不会进入分组，因此不存在「空图也按此播种」。
    /// EN: Floor for the adaptive cap (`max(this, graph_len / divisor)`); the
    /// final width is `min(configured, cap)`, so a smaller configured width stays
    /// below this value. The plan/apply path does not group until the graph
    /// reaches `seed_serial_points`, so it never seeds an empty graph.
    pub plan_group_min: usize,
    /// 图内点数 < K 时从 fresh 头单线程插入。
    /// EN: Seed ST insert threshold (0 = disable).
    pub seed_serial_points: usize,
    /// 串行合图时每 N 点释放一次写锁（0 = 关闭）。用于压低边写边搜的尾延迟。
    /// Release the graph write lock every N inserts in the serial merge path
    /// (0 = disabled). Bounds search wait during streaming builds.
    pub commit_chunk: usize,
    /// 每轮 builder drain 后 yield。
    /// EN: Yield after each drain round.
    pub builder_yield: bool,
    /// Bulk 构图 ef_construction 截断上限（None = 不截断）。
    /// EN: Cap for bulk ef.
    pub bulk_ef_construction_cap: Option<usize>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            max_build_batch: 512,
            optimize_drain_all: false,
            breakdown_enabled: false,
            collect_edge_metrics: false,
            parallel_batch_build: false,
            parallel_min_batch: 512,
            parallel_min_graph: 4096,
            parallel_chunk_size: 256,
            parallel_min_chunk: 64,
            parallel_plan_group_chunks: 4,
            plan_group_graph_divisor: 256,
            plan_group_min: 16,
            seed_serial_points: 256,
            commit_chunk: 0,
            builder_yield: false,
            bulk_ef_construction_cap: None,
        }
    }
}

impl BuildConfig {
    pub fn from_env() -> Self {
        Self {
            max_build_batch: env_usize("HOUYI_HNSW_MAX_BUILD_BATCH", 512).clamp(64, 256 * 1024),
            optimize_drain_all: env_bool("HOUYI_HNSW_OPTIMIZE_DRAIN_ALL", false),
            breakdown_enabled: env_bool("HOUYI_HNSW_OPTIMIZE_BREAKDOWN", false),
            collect_edge_metrics: env_bool("HOUYI_HNSW_COLLECT_EDGE_METRICS", false),
            parallel_batch_build: env_bool("HOUYI_HNSW_PARALLEL_BATCH_BUILD", false),
            parallel_min_batch: env_usize("HOUYI_HNSW_PARALLEL_MIN_BATCH", 512).max(1),
            parallel_min_graph: env_usize("HOUYI_HNSW_PARALLEL_MIN_GRAPH", 4096),
            parallel_chunk_size: env_usize("HOUYI_HNSW_PARALLEL_CHUNK_SIZE", 256).max(32),
            parallel_min_chunk: env_usize("HOUYI_HNSW_PARALLEL_MIN_CHUNK", 64).max(8),
            parallel_plan_group_chunks: env_usize("HOUYI_HNSW_PARALLEL_PLAN_GROUP_CHUNKS", 4)
                .max(1),
            // 自适应组宽默认开启：4.6k 语料实测固定组宽 1024 会让 recall@10 从
            // ~100% 掉到 6–50%（组内兄弟不可见），而 1M 图上 1024 组宽仅占图的
            // 0.1%，兄弟盲区可忽略——用图规模的比例上限兼顾两边。
            // EN: Adaptive width is on by default: at a 4.6k corpus a fixed 1024
            // group collapsed recall@10 from ~100% to 6–50% (sibling blindness),
            // while at 1M vectors a 1024 group is only 0.1% of the graph and the
            // blind spot is negligible — a graph-size proportion covers both.
            // 0 是文档承诺的「关闭自适应」开关，必须用允许 0 的解析：
            // `env_usize` 会把 0 当非法值回落到默认 256，使开关失效。
            // EN: 0 is the documented "disable adaptation" switch and needs the
            // allow-zero parser — `env_usize` treats 0 as invalid and falls back
            // to the default 256, making the switch ineffective.
            plan_group_graph_divisor: env_usize_allow_zero(
                "HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR",
                256,
            ),
            plan_group_min: env_usize("HOUYI_HNSW_PLAN_GROUP_MIN", 16),
            // `0 = 关闭` 是有意义的值（`insert` 分支据此判断），必须允许零值：
            // `env_usize` 会把显式 0 回落成默认 256，使开关从环境变量不可达。
            // EN: `0 = disable` is meaningful (the insert path branches on it) and
            // needs the allow-zero parser — `env_usize` turns an explicit 0 into
            // the default 256, making the switch unreachable from the env.
            seed_serial_points: env_usize_allow_zero("HOUYI_HNSW_SEED_SERIAL_POINTS", 256),
            commit_chunk: env_usize("HOUYI_HNSW_COMMIT_CHUNK", 0),
            builder_yield: env_bool("HOUYI_HNSW_BUILDER_YIELD", false),
            bulk_ef_construction_cap: match std::env::var("HOUYI_HNSW_BULK_EF_CONSTRUCTION")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
            {
                None => None,
                Some(0) => None,
                Some(cap) => Some(cap),
            },
        }
    }
}

/// 计算自适应规划组宽：`min(configured, max(min, graph_len / divisor))`，结果
/// 至少为 1；`divisor == 0` 关闭收敛（返回 configured）。
///
/// 为什么按「图规模的比例」而不是固定小批：组内规划对组前图快照可见、组内兄弟
/// 互不可见。成簇数据里同批兄弟互为最近邻，固定大组会让新节点在提交后被对称剪
/// 枝剪成孤岛；组宽随图增长（约为图的 1/divisor）后，早期小图用小组保质量，大
/// 图仍用大组保吞吐。默认 divisor=256、min=16 来自 4653×512 知识库语料实测
/// （fixed 1024：recall@10 6–50%；组 16：100%）。
///
/// Computes the adaptive plan-group width: `min(configured, max(min,
/// graph_len / divisor))`, at least 1; `divisor == 0` disables adaptation and
/// returns `configured`. Planning sees only the pre-group snapshot and siblings
/// inside a group are invisible to each other, so a fixed large group turns
/// clustered siblings into islands after symmetric pruning. Scaling the group
/// with the graph (~1/divisor) keeps small-graph quality and large-graph
/// throughput. Defaults (256/16) come from the 4653×512 knowledge-base corpus
/// (fixed 1024: recall@10 6–50%; group 16: 100%).
pub fn effective_plan_group_size(
    configured: usize,
    graph_len: usize,
    min: usize,
    divisor: usize,
) -> usize {
    if divisor == 0 {
        return configured.max(1);
    }
    let adaptive_cap = (graph_len / divisor).max(min);
    configured.min(adaptive_cap).max(1)
}

// ── 向量索引参数 ────────────────────────────────────────────
// EN: ── Vector index parameters ─────────────────────────────

#[derive(Debug, Clone)]
pub struct VectorConfig {
    pub auto_hnsw: bool,
    pub recall_optimized: bool,
    pub estimated_vectors: Option<usize>,
    pub force_flat_search: bool,
    pub sanity_check: bool,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            auto_hnsw: true,
            recall_optimized: false,
            estimated_vectors: None,
            force_flat_search: false,
            sanity_check: false,
        }
    }
}

impl VectorConfig {
    pub fn from_env() -> Self {
        Self {
            auto_hnsw: env_bool("HOUYI_VECTOR_AUTO_HNSW", true),
            recall_optimized: env_bool("HOUYI_VECTOR_RECALL_OPTIMIZED", false),
            estimated_vectors: env_usize_optional("HOUYI_VECTOR_ESTIMATED_VECTORS"),
            force_flat_search: env_bool("HOUYI_VECTOR_FORCE_FLAT", false),
            sanity_check: env_bool("HOUYI_VECTOR_SANITY_CHECK", false),
        }
    }
}

// ── 优化日志参数 ────────────────────────────────────────────
// EN: ── Optimize log parameters ─────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct OptimizeLogConfig {
    pub quiet: bool,
    pub log_file: Option<String>,
}

impl OptimizeLogConfig {
    pub fn from_env() -> Self {
        Self {
            quiet: env_bool("HOUYI_HNSW_OPTIMIZE_LOG_QUIET", false),
            log_file: std::env::var("HOUYI_HNSW_OPTIMIZE_LOG_FILE")
                .ok()
                .filter(|s| !s.trim().is_empty()),
        }
    }
}

// ── 聚合配置 ────────────────────────────────────────────────
// EN: ── Aggregate config ───────────────────────────────────

/// 进程级统一配置，启动时调用 `HouyiConfig::load()` 一次。
/// Process-level config; call `HouyiConfig::load()` once at startup.
#[derive(Debug, Clone)]
pub struct HouyiConfig {
    pub hnsw: HnswConfig,
    pub build: BuildConfig,
    pub vector: VectorConfig,
    pub optimize_log: OptimizeLogConfig,
}

impl HouyiConfig {
    pub fn load() -> Self {
        Self {
            hnsw: HnswConfig::from_env(),
            build: BuildConfig::from_env(),
            vector: VectorConfig::from_env(),
            optimize_log: OptimizeLogConfig::from_env(),
        }
    }
}

/// 串行化 crate 内会修改/读取进程级 `env` 的测试，避免并行执行时互相污染。
/// 同进程的其它测试模块（如 `cxl_memory`）也必须使用本锁：`set_var` 与并发
/// `var()` 读取在任何模块间都是同一个进程级竞争。
/// 某用例持锁 panic 会毒化互斥量；回收 `into_inner` 让其余用例继续报告
/// 自身结果，而不是级联失败在「锁毒化」上掩盖真正的根因。
/// EN: Serializes crate-wide tests that mutate or read process env so parallel
/// cases cannot contaminate each other. Other test modules in the same process
/// (e.g. `cxl_memory`) must share this lock: `set_var` racing a concurrent
/// `var()` read is one process-wide race regardless of module. A test
/// panicking while holding the guard poisons the mutex; recovering via
/// `into_inner` keeps the remaining cases meaningful instead of cascading on
/// a poison error that hides the root cause.
#[cfg(test)]
pub(crate) fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// 测试结束（含 panic）后恢复环境变量快照，供 crate 内所有 env 测试共用。
/// 快照存 `OsString` 而非 `String`：`var().ok()` 会把 `NotUnicode` 折叠成
/// `None`，drop 时便对原本存在的非 UTF-8 值执行 `remove_var`，静默删除而非
/// 恢复——CXL 路径这类文件系统变量在 Unix 上就可能包含非 UTF-8 字节。
/// EN: Restores an env-var snapshot on drop (including on panic); shared by all
/// env tests in the crate. The snapshot is an `OsString`, not a `String`:
/// `var().ok()` collapses `NotUnicode` into `None`, so drop would `remove_var`
/// a pre-existing non-UTF-8 value instead of restoring it — filesystem-backed
/// variables such as the CXL paths can contain non-UTF-8 bytes on Unix.
#[cfg(test)]
pub(crate) struct EnvRestore {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

#[cfg(test)]
impl EnvRestore {
    pub(crate) fn new(key: &'static str) -> Self {
        Self {
            key,
            prev: std::env::var_os(key),
        }
    }
}

#[cfg(test)]
impl Drop for EnvRestore {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_bool_parses_variants() {
        // 这些测试并不实际设置环境变量——它们测试解析逻辑。读取进程 `env` 也
        // 必须与改写用例串行：`set_var` 与无锁 `var()` 并发正是本锁要消除的竞争。
        // EN: These tests don't actually set env vars — they test the parsing
        // logic. Reading process env must also serialize with mutating cases:
        // concurrent `set_var` and an unlocked `var()` read is exactly the race
        // this lock exists to remove.
        let _guard = env_test_lock();
        // 未设置环境变量时，env_bool 返回默认值。
        // EN: When no env var is set, env_bool returns the default.
        assert!(env_bool("HOUYI_NONEXISTENT_TEST_VAR_12345", true));
        assert!(!env_bool("HOUYI_NONEXISTENT_TEST_VAR_12345", false));
    }

    #[test]
    fn default_configs_are_valid() {
        let hnsw = HnswConfig::default();
        assert!(hnsw.m >= 2);
        assert!(hnsw.ef_construction >= 1);
        assert!(hnsw.ef_search >= 1);
        assert!(hnsw.max_layers >= 1);

        let build = BuildConfig::default();
        assert!(build.max_build_batch >= 64);
        assert!(build.parallel_chunk_size >= 32);

        let vec = VectorConfig::default();
        assert!(vec.auto_hnsw);
        assert!(!vec.force_flat_search);
    }

    #[test]
    fn effective_plan_group_size_converges_with_graph() {
        // 空图/小图受下限约束；4.6k 图给出约 18 的组宽；大图回到配置值；
        // divisor=0 关闭自适应；配置值本身更小时不被放大。
        // EN: Empty/small graphs are bounded by the floor; a 4.6k graph yields
        // about 18; large graphs return to the configured width; divisor=0
        // disables adaptation; a smaller configured width is never inflated.
        assert_eq!(effective_plan_group_size(1024, 0, 16, 256), 16);
        assert_eq!(effective_plan_group_size(1024, 4653, 16, 256), 18);
        assert_eq!(effective_plan_group_size(1024, 1_000_000, 16, 256), 1024);
        assert_eq!(effective_plan_group_size(1024, 4653, 16, 0), 1024);
        assert_eq!(effective_plan_group_size(8, 4653, 16, 256), 8);
        assert_eq!(effective_plan_group_size(0, 0, 0, 256), 1);
    }

    #[test]
    fn plan_group_divisor_env_zero_disables_adaptation() {
        // 回归：`env_usize` 的正数过滤会把 0 回落成默认 256，使文档承诺的
        // 「divisor=0 关闭自适应」无法从环境变量生效。纯函数测试无法覆盖解析层。
        // EN: Regression — `env_usize`'s positive-only filter turns 0 into the
        // default 256, so the documented "divisor=0 disables adaptation" never
        // takes effect via the environment. The pure-function test cannot catch
        // the parsing layer.
        let _guard = env_test_lock();
        let _restore = EnvRestore::new("HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR");
        std::env::set_var("HOUYI_HNSW_PLAN_GROUP_GRAPH_DIVISOR", "0");
        let build = BuildConfig::from_env();
        assert_eq!(
            build.plan_group_graph_divisor, 0,
            "divisor=0 must disable adaptation, not fall back to the default"
        );
    }

    #[test]
    fn seed_serial_points_env_zero_disables_seeding() {
        // 同属 `env_usize` 吞掉显式 0 的缺陷类：`0 = 关闭种子串行点`是文档承诺
        // 的开关，字段解析与消费分处两处，只有 env 级测试能钉住解析层不回落。
        // EN: Same defect class — `env_usize` swallowing an explicit 0. `0 =
        // disable seed serial points` is a documented switch, and parsing and
        // consumption live in different places; only an env-level test can pin
        // that the parser does not fall back to the default.
        let _guard = env_test_lock();
        let _restore = EnvRestore::new("HOUYI_HNSW_SEED_SERIAL_POINTS");
        std::env::set_var("HOUYI_HNSW_SEED_SERIAL_POINTS", "0");
        let build = BuildConfig::from_env();
        assert_eq!(
            build.seed_serial_points, 0,
            "0 must disable seed serial points, not fall back to the default 256"
        );
    }

    #[test]
    fn houyi_config_load_returns_valid_defaults() {
        // 读取 `env` 期间与其它改写用例串行，避免同进程竞争。
        // EN: Serialize with env-mutating cases to avoid same-process races.
        let _guard = env_test_lock();
        let cfg = HouyiConfig::load();
        assert!(cfg.hnsw.m >= 2);
        assert!(cfg.build.max_build_batch >= 64);
    }
}
