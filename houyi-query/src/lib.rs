pub mod pgwire;

use sqlparser::ast::{Expr, SetExpr, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

/// 简化版 pg_wire 查询表示.
/// Simplified pg_wire query representation.
#[derive(Debug, Clone)]
pub struct PgWireQuery {
    pub sql: String,
}

/// pg_wire Simple Query 扩展：块注释嵌入查询向量（无 extended query 时仍可做混合检索）。
/// EN: Block-comment embedding for query vector when extended-query protocol is not used.
#[derive(Debug, Clone, PartialEq)]
pub enum HouyiPgWireSql {
    /// 无 `houyi:vector` 指令；由服务端决定占位或拒绝。
    /// EN: No `houyi:vector` directive; the server decides whether to stub or reject.
    Plain(String),
    /// 已剥离指令；`k` 来自 `LIMIT n`，缺省为 10。
    /// EN: Directive already stripped; `k` comes from `LIMIT n`, default 10.
    VectorSearch {
        sql_for_planner: String,
        query_vector: Vec<f32>,
        k: usize,
        /// 来自 `/* houyi:ef_search=N */`；与 HTTP `ef_search` 语义一致。
        /// EN: From `/* houyi:ef_search=N */`; same semantics as HTTP `ef_search`.
        ef_search_override: Option<usize>,
    },
    /// 稀疏+稠密融合检索（P4）：`/* houyi:sparse=t:w,… */` 与 `/* houyi:vector=… */`
    /// 必须同时存在；`fusion`/`alpha` 携带原始指令值（语义校验在服务端，映射 22023）。
    /// EN: Sparse+dense fusion search (P4): the sparse and vector directives must
    /// both be present; `fusion`/`alpha` carry raw directive values (semantic
    /// validation is server-side, mapping to 22023).
    FusionSearch {
        sql_for_planner: String,
        query_vector: Vec<f32>,
        /// 原始 `(term, weight)` 对；重复求和由服务端内核完成。
        /// EN: Raw `(term, weight)` pairs; duplicate summing is done by the kernel.
        query_sparse: Vec<(u32, f32)>,
        k: usize,
        ef_search_override: Option<usize>,
        fusion: Option<String>,
        alpha: Option<f32>,
    },
}

// 所有指令共用的前缀：快路径只扫一次 `contains`，各指令常量必须由它开头。
// EN: Prefix shared by every directive: the fast path scans `contains` once,
// and each directive constant must start with it.
const HOUYI_DIRECTIVE_PREFIX: &str = "/* houyi:";
const HOUYI_VECTOR_DIRECTIVE: &str = "/* houyi:vector=";
// 指令前缀：在 SQL 注释中嵌入查询向量、ef_search、collection 与融合参数。
// EN: Directive prefixes: embed query vector, ef_search, collection, and fusion params in SQL comments.
const HOUYI_EF_SEARCH_DIRECTIVE: &str = "/* houyi:ef_search=";
const HOUYI_COLLECTION_DIRECTIVE: &str = "/* houyi:collection=";
const HOUYI_SPARSE_DIRECTIVE: &str = "/* houyi:sparse=";
const HOUYI_FUSION_DIRECTIVE: &str = "/* houyi:fusion=";
const HOUYI_ALPHA_DIRECTIVE: &str = "/* houyi:alpha=";

/// 剥离某类指令：一次扫描收集所有出现区间，每个值都经 `parse` 校验（生效值
/// 为最后一次，与 ef_search「last wins」策略一致），最后一次性重建字符串。
/// 线性成本：重复指令不再逐条 O(len) 重新分配（OCR f42415a：此前数千条重复
/// 注释会把解析放大成二次成本）。
/// EN: Strip one directive family: one scan collects every occurrence span, each
/// value goes through `parse` (the last one wins, same "last wins" policy as
/// ef_search), and the string is rebuilt once. Linear cost: repeated directives
/// no longer reallocate O(len) per occurrence (OCR f42415a: thousands of
/// duplicate comments used to amplify parsing into quadratic cost).
fn strip_houyi_value_directives<T>(
    sql: &str,
    prefix: &str,
    label: &str,
    mut parse: impl FnMut(&str) -> Result<T, String>,
) -> Result<(String, Option<T>), String> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut last: Option<T> = None;
    // 指令前缀只可能出现在注释开头，区间不会重叠，顺序扫描即可。
    // EN: The prefix can only start a comment, so spans never overlap; a
    // forward scan suffices.
    let mut search_from = 0;
    while let Some(rel) = sql[search_from..].find(prefix) {
        let start = search_from + rel;
        let after = &sql[start + prefix.len()..];
        let end_rel = after
            .find("*/")
            .ok_or_else(|| format!("unterminated houyi:{label} comment"))?;
        let value = after[..end_rel].trim();
        last = Some(parse(value)?);
        let remove_end = start + prefix.len() + end_rel + 2;
        spans.push((start, remove_end));
        search_from = remove_end;
    }
    if spans.is_empty() {
        return Ok((sql.trim().to_string(), None));
    }
    // 每个非指令片段按两侧边界 trim，非空片段以单个空格重连——与逐条剥离
    // 的语义逐字一致（含指令位于首/尾与纯指令语句）。
    // EN: Trim each non-directive fragment at its boundaries and rejoin non-empty
    // fragments with single spaces — byte-for-byte the same result as the former
    // per-occurrence stripping (including directives at either end and
    // directive-only statements).
    let mut out = String::with_capacity(sql.len());
    let mut cursor = 0;
    for (start, end) in spans {
        let part = sql[cursor..start].trim();
        if !part.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(part);
        }
        cursor = end;
    }
    let tail = sql[cursor..].trim();
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(tail);
    }
    Ok((out, last))
}

fn strip_houyi_ef_search_directives(sql: &str) -> Result<(String, Option<usize>), String> {
    strip_houyi_value_directives(sql, HOUYI_EF_SEARCH_DIRECTIVE, "ef_search", |value| {
        let n: usize = value
            .parse()
            .map_err(|_| format!("invalid houyi:ef_search value: {value}"))?;
        if n == 0 {
            return Err("houyi:ef_search must be > 0".to_string());
        }
        Ok(n)
    })
}

/// 解析 `/* houyi:sparse=t:w[,t:w…] */` 结构（整数 term + 浮点权重）。
/// 语义域（term ∈ 1..=MAX、权重有限 > 0、数量上限、重复求和）由服务端内核
/// `houyi_vector::sparse::normalize_terms` 统一校验；本层只拒绝结构错误，
/// 让越界值走 22023 而非 42601。
/// EN: Parse the sparse directive structure (integer term + float weight). The
/// semantic domain (term in 1..=MAX, finite > 0 weights, count cap, duplicate
/// summing) is checked server-side by the kernel `normalize_terms`; this layer
/// rejects only structural errors so out-of-domain values surface as 22023
/// instead of 42601.
fn parse_sparse_directive(raw: &str) -> Result<Vec<(u32, f32)>, String> {
    let mut out = Vec::new();
    for token in raw.split(',') {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        let (term_str, weight_str) = t
            .split_once(':')
            .ok_or_else(|| format!("invalid sparse term pair {t:?} (expect term:weight)"))?;
        let term: u32 = term_str
            .trim()
            .parse()
            .map_err(|_| format!("invalid sparse term id in {t:?} (expect unsigned integer)"))?;
        let weight: f32 = weight_str
            .trim()
            .parse()
            .map_err(|_| format!("invalid sparse weight in {t:?} (expect float)"))?;
        out.push((term, weight));
    }
    if out.is_empty() {
        return Err("houyi:sparse directive is empty".to_string());
    }
    Ok(out)
}

/// 从 SQL 注释中提取 collection 指令 / Extract collection directive from SQL comment.
/// `/* houyi:collection=my_collection */` → `Some("my_collection")`
pub fn extract_collection_directive(sql: &str) -> Option<String> {
    let start = sql.find(HOUYI_COLLECTION_DIRECTIVE)?;
    let after = &sql[start + HOUYI_COLLECTION_DIRECTIVE.len()..];
    let end = after.find("*/")?;
    let name = after[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// 供扩展查询 Describe 的 `RowDescription` 单列名：向量/融合检索为 `id`，否则为 `value`（与 Simple Query 结果列一致）。
/// EN: `RowDescription` column for extended-query Describe: `id` for vector/fusion search, else `value`.
pub fn pgwire_row_column_name_for_sql(sql: &str) -> Result<&'static str, String> {
    match parse_houyi_pgwire_sql(sql) {
        Ok(HouyiPgWireSql::VectorSearch { .. } | HouyiPgWireSql::FusionSearch { .. }) => Ok("id"),
        Ok(HouyiPgWireSql::Plain(_)) => Ok("value"),
        Err(e) => Err(e),
    }
}

/// 解析 `/* houyi:vector=… */` 的浮点列表（剥离由通用 helper 完成）。
/// 与 sparse/fusion/alpha 一致：每次出现都校验、最后一次生效（OCR f42415a：
/// 此前只解析第一条，重复向量指令会静默组合出「首条稠密 + 末条稀疏」）。
/// EN: Parse the float list of a `houyi:vector` directive (stripping is done by
/// the shared helper). Like sparse/fusion/alpha, every occurrence is validated
/// and the last one wins (OCR f42415a: only the first occurrence used to be
/// parsed, letting duplicate vector directives silently pair the *first* dense
/// vector with the *last* sparse list).
fn parse_vector_directive(raw: &str) -> Result<Vec<f32>, String> {
    let mut query_vector = Vec::new();
    for token in raw.split(',') {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        let f: f32 = t
            .parse()
            .map_err(|_| format!("invalid float in houyi:vector: {t}"))?;
        query_vector.push(f);
    }
    if query_vector.is_empty() {
        return Err("houyi:vector list is empty".to_string());
    }
    Ok(query_vector)
}

/// 解析 Simple Query 文本：提取 `houyi:vector`、`houyi:ef_search`、`houyi:sparse`、
/// `houyi:fusion`、`houyi:alpha` 指令与可选 `LIMIT n`。
/// 稀疏与向量指令同时存在 → 融合检索；只有稀疏（或只有融合参数）→ 语法错误。
/// EN: Parses Simple Query text: vector/ef_search/sparse/fusion/alpha directives
/// plus optional `LIMIT n`. Sparse + vector present → fusion search; sparse alone
/// (or fusion params alone) → syntax error.
pub fn parse_houyi_pgwire_sql(sql: &str) -> Result<HouyiPgWireSql, String> {
    // 普通 SQL 不含任何 houyi 指令时直接返回：下面五次整串扫描（每次无命中
    // 也会 trim 并重新分配返回串）退化为一次 contains；仅保留最终 `Plain` 的
    // 一次 trim/to_string。各指令常量与 `HOUYI_DIRECTIVE_PREFIX` 的包含关系
    // 由单测锁定，该判定因此是构造上成立的安全超集。
    // EN: Plain SQL without any houyi directive returns immediately: the five
    // full-string scans below (each trims and reallocates even on a miss)
    // collapse into one `contains`, leaving only the final `Plain` trim/to_string.
    // A unit test pins that every directive constant starts with
    // `HOUYI_DIRECTIVE_PREFIX`, so the predicate is a safe superset by
    // construction.
    if !sql.contains(HOUYI_DIRECTIVE_PREFIX) {
        return Ok(HouyiPgWireSql::Plain(sql.trim().to_string()));
    }
    let (work, ef_search_override) = strip_houyi_ef_search_directives(sql)?;
    let (work, sparse_directive) = strip_houyi_value_directives(
        &work,
        HOUYI_SPARSE_DIRECTIVE,
        "sparse",
        parse_sparse_directive,
    )?;
    let (work, fusion) =
        strip_houyi_value_directives(&work, HOUYI_FUSION_DIRECTIVE, "fusion", |value| {
            if value.is_empty() {
                Err("houyi:fusion value is empty".to_string())
            } else {
                Ok(value.to_string())
            }
        })?;
    let (work, alpha) =
        strip_houyi_value_directives(&work, HOUYI_ALPHA_DIRECTIVE, "alpha", |value| {
            value
                .parse::<f32>()
                .map_err(|_| format!("invalid float in houyi:alpha: {value}"))
        })?;

    // 向量指令与其余指令同规：全部出现都校验、最后一次生效。
    // EN: The vector directive follows the same rule as the others: every
    // occurrence is validated and the last one wins.
    let (work, vector_directive) = strip_houyi_value_directives(
        &work,
        HOUYI_VECTOR_DIRECTIVE,
        "vector",
        parse_vector_directive,
    )?;
    let Some(query_vector) = vector_directive else {
        // 稀疏/融合参数脱离向量检索没有意义；静默忽略会让客户端误以为生效，
        // 在解析层直接拒绝（映射 42601）。裸 ef_search 指令仍是历史容错
        // （只剥离不报错，由 `parse_pgwire_strips_ef_only_plain` 固化）：
        // 与全部硬失败的新指令不同属**有意保留**的不对称，避免破坏既有客户端。
        // EN: Sparse/fusion params are meaningless without vector search;
        // silently ignoring them would let clients believe they took effect, so
        // reject at the parse layer (mapped to 42601). A bare ef_search directive
        // stays historical tolerance (stripped, not rejected; pinned by
        // `parse_pgwire_strips_ef_only_plain`) — a deliberately kept asymmetry
        // with the strictly validated new directives, to avoid breaking existing
        // clients.
        if sparse_directive.is_some() {
            return Err(
                "houyi:sparse requires a houyi:vector directive (fusion search needs \
                 both modalities)"
                    .to_string(),
            );
        }
        if fusion.is_some() || alpha.is_some() {
            return Err(
                "houyi:fusion/houyi:alpha require a houyi:sparse and houyi:vector directive"
                    .to_string(),
            );
        }
        return Ok(HouyiPgWireSql::Plain(work));
    };
    if (fusion.is_some() || alpha.is_some()) && sparse_directive.is_none() {
        return Err("houyi:fusion/houyi:alpha require a houyi:sparse directive".to_string());
    }
    let sql_for_planner = work;
    let k = extract_limit_k(&sql_for_planner).unwrap_or(10);
    if k == 0 {
        return Err("LIMIT must be >= 1 for vector search".to_string());
    }
    match sparse_directive {
        Some(query_sparse) => Ok(HouyiPgWireSql::FusionSearch {
            sql_for_planner,
            query_vector,
            query_sparse,
            k,
            ef_search_override,
            fusion,
            alpha,
        }),
        None => Ok(HouyiPgWireSql::VectorSearch {
            sql_for_planner,
            query_vector,
            k,
            ef_search_override,
        }),
    }
}

/// AST-based：从 SQL 中提取 `LIMIT n` 子句。
/// EN: AST-based: extracts the `LIMIT n` clause from SQL.
/// 提取 `LIMIT n`（查询日志/统计用；与规划器共用同一实现）。
/// EN: extract `LIMIT n`; shared with the planner and query logging.
pub fn extract_limit_k(sql: &str) -> Option<usize> {
    let dialect = GenericDialect {};
    let ast = Parser::parse_sql(&dialect, sql).ok()?;
    let stmt = ast.first()?;
    let query = match stmt {
        Statement::Query(q) => q,
        _ => return None,
    };
    let limit_clause = query.limit_clause.as_ref()?;
    let limit_expr = match limit_clause {
        sqlparser::ast::LimitClause::LimitOffset { limit, .. } => limit.as_ref()?,
        sqlparser::ast::LimitClause::OffsetCommaLimit { limit, .. } => limit,
    };
    match limit_expr {
        Expr::Value(v) => v.value.to_string().parse::<usize>().ok(),
        _ => None,
    }
}

/// 第一版执行计划：保留混合检索规划接口.
/// v0.1 execution plan with hybrid-search placeholder.
#[derive(Debug, Clone)]
pub enum ExecutionPlan {
    RawSql(String),
    HybridSearch {
        sql: String,
        pushed_down_bitset: Option<Vec<u64>>,
    },
}

pub struct QueryPlanner;

impl QueryPlanner {
    pub fn plan(query: PgWireQuery) -> ExecutionPlan {
        let bitset = extract_filter_ids(&query.sql);
        // 若 WHERE 子句带有可过滤的 id，则视为混合检索。
        // EN: If we have a WHERE clause with filterable ids, it's a hybrid search.
        if bitset.is_some() || query.sql.to_ascii_uppercase().contains("WHERE") {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset: bitset,
                sql: query.sql,
            }
        } else {
            ExecutionPlan::RawSql(query.sql)
        }
    }
}

/// AST-based：从 WHERE 子句中提取可下推的 id 过滤条件。
/// EN: AST-based: extracts pushdown-able id filters from the WHERE clause.
fn extract_filter_ids(sql: &str) -> Option<Vec<u64>> {
    let dialect = GenericDialect {};
    let ast = Parser::parse_sql(&dialect, sql).ok()?;
    let stmt = ast.first()?;
    let query = match stmt {
        Statement::Query(q) => q,
        _ => return None,
    };
    // 导航路径：Query → body (SetExpr) → Select → selection (WHERE 子句)。
    // EN: Navigate: Query → body (SetExpr) → Select → selection (WHERE clause).
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s,
        _ => return None,
    };
    let selection = select.selection.as_ref()?;
    walk_expr_for_ids(selection)
}

/// 递归遍历表达式树，收集 `id IN (...)` / `id = n` / AND / OR 中的 id 集合。
/// EN: Recursively collects ids from `id IN (...)`, `id = n`, AND, and OR.
/// 支持: IN, =, >, <, >=, <=, AND, OR, BETWEEN
/// EN: Supports: IN, =, >, <, >=, <=, AND, OR, BETWEEN.
fn walk_expr_for_ids(expr: &Expr) -> Option<Vec<u64>> {
    match expr {
        // 示例：
        // EN: Example:
        // WHERE id IN (1, 2, 3)
        Expr::InList {
            expr: inner,
            list,
            negated: false,
            ..
        } => {
            if !is_id_column(inner) {
                return None;
            }
            let mut ids = Vec::new();
            for item in list {
                match item {
                    Expr::Value(v) => {
                        ids.push(v.value.to_string().parse::<u64>().ok()?);
                    }
                    _ => return None,
                }
            }
            if ids.is_empty() {
                None
            } else {
                Some(ids)
            }
        }
        // 示例：
        // EN: Example:
        // WHERE id = 5
        Expr::BinaryOp { left, op, right } if *op == sqlparser::ast::BinaryOperator::Eq => {
            if !is_id_column(left) {
                return None;
            }
            match right.as_ref() {
                Expr::Value(v) => Some(vec![v.value.to_string().parse::<u64>().ok()?]),
                _ => None,
            }
        }
        // WHERE id > 5 / id >= 5 / id < 5 / id <= 5  → 不能转为精确 bitset，返回 None
        // EN: WHERE id >/>=/</<= 5 cannot become an exact bitset; returns None.
        Expr::BinaryOp { op, .. }
            if *op == sqlparser::ast::BinaryOperator::Gt
                || *op == sqlparser::ast::BinaryOperator::GtEq
                || *op == sqlparser::ast::BinaryOperator::Lt
                || *op == sqlparser::ast::BinaryOperator::LtEq =>
        {
            None // 范围过滤无法转为有限 id 集合
        }
        // WHERE id BETWEEN 5 AND 10 → 不能转为精确 bitset
        // EN: WHERE id BETWEEN 5 AND 10 cannot become an exact bitset.
        Expr::Between { .. } => None,
        // WHERE ... AND ... → AND 合并
        // EN: WHERE ... AND ... → merge with AND.
        Expr::BinaryOp { left, op, right } if *op == sqlparser::ast::BinaryOperator::And => {
            match (walk_expr_for_ids(left), walk_expr_for_ids(right)) {
                (Some(mut l), Some(r)) => {
                    l.extend(r);
                    l.sort_unstable();
                    l.dedup();
                    Some(l)
                }
                (Some(ids), None) => Some(ids),
                (None, Some(ids)) => Some(ids),
                (None, None) => None,
            }
        }
        // WHERE ... OR ... → OR 合并
        // EN: WHERE ... OR ... → merge with OR.
        Expr::BinaryOp { left, op, right } if *op == sqlparser::ast::BinaryOperator::Or => {
            match (walk_expr_for_ids(left), walk_expr_for_ids(right)) {
                (Some(mut l), Some(r)) => {
                    l.extend(r);
                    l.sort_unstable();
                    l.dedup();
                    Some(l)
                }
                (Some(ids), _) => Some(ids),
                (_, Some(ids)) => Some(ids),
                (None, None) => None,
            }
        }
        _ => None,
    }
}

/// 判断表达式是否为 `id` 列引用。
/// EN: Checks whether the expression is an `id` column reference.
fn is_id_column(expr: &Expr) -> bool {
    match expr {
        Expr::Identifier(ident) => ident.value.eq_ignore_ascii_case("id"),
        Expr::CompoundIdentifier(parts) => {
            parts.len() == 2 && parts[1].value.eq_ignore_ascii_case("id")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        extract_collection_directive, parse_houyi_pgwire_sql, pgwire_row_column_name_for_sql,
        ExecutionPlan, HouyiPgWireSql, PgWireQuery, QueryPlanner, HOUYI_ALPHA_DIRECTIVE,
        HOUYI_COLLECTION_DIRECTIVE, HOUYI_DIRECTIVE_PREFIX, HOUYI_EF_SEARCH_DIRECTIVE,
        HOUYI_FUSION_DIRECTIVE, HOUYI_SPARSE_DIRECTIVE, HOUYI_VECTOR_DIRECTIVE,
    };

    // ── 原有测试（保持兼容）─────────────────────────────────
    // EN: ── Original tests (kept compatible) ─────────────────

    #[test]
    fn pgwire_row_column_name_vector_search_is_id() {
        assert_eq!(
            pgwire_row_column_name_for_sql(
                "/* houyi:vector=0.1,0.2,0.3,0.4 */ SELECT id FROM t LIMIT 3"
            )
            .unwrap(),
            "id"
        );
    }

    #[test]
    fn pgwire_row_column_name_plain_is_value() {
        assert_eq!(pgwire_row_column_name_for_sql("SELECT 1").unwrap(), "value");
    }

    #[test]
    fn pgwire_row_column_name_propagates_parse_error() {
        let err = pgwire_row_column_name_for_sql("/* houyi:vector= */ SELECT 1").unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn extract_collection_directive_parses_name() {
        assert_eq!(
            extract_collection_directive("/* houyi:collection=my_coll */ SELECT 1"),
            Some("my_coll".to_string())
        );
        assert_eq!(extract_collection_directive("SELECT 1"), None);
        assert_eq!(
            extract_collection_directive("/* houyi:collection= */ SELECT 1"),
            None
        );
    }

    #[test]
    fn plan_non_where_sql_as_raw() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT 1".to_string(),
        });
        match plan {
            ExecutionPlan::RawSql(sql) => assert_eq!(sql, "SELECT 1"),
            _ => panic!("expected raw plan"),
        }
    }

    #[test]
    fn plan_where_id_in_as_hybrid_with_bitset() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM docs WHERE id IN (7, 9, 13)".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => assert_eq!(pushed_down_bitset, Some(vec![7, 9, 13])),
            _ => panic!("expected hybrid plan"),
        }
    }

    #[test]
    fn plan_where_without_id_in_keeps_none_bitset() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM docs WHERE tag = 'p0'".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => assert!(pushed_down_bitset.is_none()),
            _ => panic!("expected hybrid plan"),
        }
    }

    #[test]
    fn parse_pgwire_plain_without_directive() {
        let p = parse_houyi_pgwire_sql("SELECT 1").unwrap();
        match p {
            HouyiPgWireSql::Plain(s) => assert_eq!(s, "SELECT 1"),
            _ => panic!("expected plain"),
        }
    }

    #[test]
    fn parse_pgwire_vector_strips_directive_and_reads_limit() {
        let p = parse_houyi_pgwire_sql(
            "/* houyi:vector=0.1, 0.2, 0.3, 0.4 */ SELECT id FROM docs WHERE id IN (2,3) LIMIT 5",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::VectorSearch {
                sql_for_planner,
                query_vector,
                k,
                ef_search_override,
            } => {
                assert_eq!(query_vector, vec![0.1, 0.2, 0.3, 0.4]);
                assert_eq!(k, 5);
                assert_eq!(ef_search_override, None);
                assert_eq!(
                    sql_for_planner,
                    "SELECT id FROM docs WHERE id IN (2,3) LIMIT 5"
                );
                let plan = QueryPlanner::plan(PgWireQuery {
                    sql: sql_for_planner,
                });
                match plan {
                    ExecutionPlan::HybridSearch {
                        pushed_down_bitset, ..
                    } => assert_eq!(pushed_down_bitset, Some(vec![2, 3])),
                    _ => panic!("expected hybrid"),
                }
            }
            _ => panic!("expected vector search"),
        }
    }

    #[test]
    fn parse_pgwire_vector_default_k_10_without_limit() {
        let p = parse_houyi_pgwire_sql("/* houyi:vector=1,0,0,0*/SELECT id FROM t").unwrap();
        match p {
            HouyiPgWireSql::VectorSearch { k, .. } => assert_eq!(k, 10),
            _ => panic!("expected vector search"),
        }
    }

    #[test]
    fn parse_pgwire_ef_search_stripped_and_applies_to_vector_search() {
        let p = parse_houyi_pgwire_sql(
            "/* houyi:ef_search=64 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM docs LIMIT 3",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::VectorSearch {
                k,
                ef_search_override,
                sql_for_planner,
                ..
            } => {
                assert_eq!(k, 3);
                assert_eq!(ef_search_override, Some(64));
                assert_eq!(sql_for_planner, "SELECT id FROM docs LIMIT 3");
            }
            _ => panic!("expected vector search"),
        }
    }

    #[test]
    fn parse_pgwire_strips_ef_only_plain() {
        let p = parse_houyi_pgwire_sql("/* houyi:ef_search=32 */ SELECT 1").unwrap();
        match p {
            HouyiPgWireSql::Plain(s) => assert_eq!(s, "SELECT 1"),
            _ => panic!("expected plain"),
        }
    }

    // ── P4 融合检索指令测试 ─────────────────────────────────
    // EN: ── P4 fusion-search directive tests ─────────────────

    #[test]
    fn pgwire_row_column_name_fusion_search_is_id() {
        assert_eq!(
            pgwire_row_column_name_for_sql(
                "/* houyi:sparse=1:0.5,2:1.2 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t LIMIT 3"
            )
            .unwrap(),
            "id"
        );
    }

    #[test]
    fn parse_pgwire_fusion_strips_directives_and_keeps_collection() {
        let p = parse_houyi_pgwire_sql(
            "/* houyi:collection=docs */ /* houyi:sparse=1:0.5, 2:1.2 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM docs LIMIT 4",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::FusionSearch {
                sql_for_planner,
                query_vector,
                query_sparse,
                k,
                ef_search_override,
                fusion,
                alpha,
            } => {
                assert_eq!(query_vector, vec![1.0, 0.0, 0.0, 0.0]);
                assert_eq!(query_sparse, vec![(1, 0.5), (2, 1.2)]);
                assert_eq!(k, 4);
                assert_eq!(ef_search_override, None);
                // 缺省融合参数由服务端补齐（rrf / alpha=0.5）。
                // EN: Defaults are filled server-side (rrf / alpha=0.5).
                assert_eq!(fusion, None);
                assert_eq!(alpha, None);
                assert_eq!(
                    sql_for_planner,
                    "/* houyi:collection=docs */ SELECT id FROM docs LIMIT 4"
                );
            }
            _ => panic!("expected fusion search"),
        }
    }

    #[test]
    fn parse_pgwire_fusion_weighted_alpha_and_ef() {
        let p = parse_houyi_pgwire_sql(
            "/* houyi:ef_search=32 */ /* houyi:alpha=0.25 */ /* houyi:fusion=WEIGHTED */ /* houyi:sparse=7:2 */ /* houyi:vector=0,1,0,0 */ SELECT id FROM t LIMIT 2",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::FusionSearch {
                k,
                ef_search_override,
                fusion,
                alpha,
                ..
            } => {
                assert_eq!(k, 2);
                assert_eq!(ef_search_override, Some(32));
                // 模式串保持大小写原样，语义解析（大小写不敏感）在服务端。
                // EN: The mode string preserves its case; semantic parsing
                // (case-insensitive) happens server-side.
                assert_eq!(fusion.as_deref(), Some("WEIGHTED"));
                assert_eq!(alpha, Some(0.25));
            }
            _ => panic!("expected fusion search"),
        }
    }

    #[test]
    fn parse_pgwire_fusion_keeps_duplicate_term_pairs_raw() {
        // 解析层保留原始对；重复求和由服务端内核 normalize_terms 完成
        // （HTTP 与 pgwire 共用同一规则）。
        // EN: The parser keeps raw pairs; duplicate summing is done by the
        // server-side kernel normalize_terms (shared by HTTP and pgwire).
        let p = parse_houyi_pgwire_sql(
            "/* houyi:sparse=3:1.0,3:0.5 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::FusionSearch { query_sparse, .. } => {
                assert_eq!(query_sparse, vec![(3, 1.0), (3, 0.5)]);
            }
            _ => panic!("expected fusion search"),
        }
    }

    #[test]
    fn parse_pgwire_sparse_without_vector_is_error() {
        let err = parse_houyi_pgwire_sql("/* houyi:sparse=1:0.5 */ SELECT id FROM t").unwrap_err();
        assert!(err.contains("houyi:vector"), "got: {err}");
    }

    #[test]
    fn parse_pgwire_fusion_directive_without_sparse_is_error() {
        let err = parse_houyi_pgwire_sql(
            "/* houyi:fusion=rrf */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t",
        )
        .unwrap_err();
        assert!(err.contains("houyi:sparse"), "got: {err}");
        let err = parse_houyi_pgwire_sql(
            "/* houyi:alpha=0.5 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t",
        )
        .unwrap_err();
        assert!(err.contains("houyi:sparse"), "got: {err}");
    }

    #[test]
    fn parse_pgwire_sparse_malformed_pairs_are_syntax_errors() {
        for raw in ["1", "x:1", "1:y", "1:0.5,2"] {
            let sql = format!(
                /* 样例 / EN: sample */
                "/* houyi:sparse={raw} */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t"
            );
            assert!(
                parse_houyi_pgwire_sql(&sql).is_err(),
                "{raw} must be rejected"
            );
        }
    }

    #[test]
    fn parse_pgwire_sparse_empty_list_is_error() {
        let err = parse_houyi_pgwire_sql(
            "/* houyi:sparse= */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t",
        )
        .unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn parse_pgwire_fusion_empty_value_is_error() {
        let err = parse_houyi_pgwire_sql(
            "/* houyi:fusion= */ /* houyi:sparse=1:1 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t",
        )
        .unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn directive_constants_all_start_with_shared_prefix() {
        // 快路径只认 `HOUYI_DIRECTIVE_PREFIX`：任何指令常量拼写漂移都会让
        // 该指令静默退化为 Plain，因此在此穷举锁定包含关系。
        // EN: The fast path only recognizes `HOUYI_DIRECTIVE_PREFIX`; any
        // directive constant that drifts would silently degrade to Plain, so
        // this test exhaustively pins the containment.
        for directive in [
            HOUYI_VECTOR_DIRECTIVE,
            HOUYI_EF_SEARCH_DIRECTIVE,
            HOUYI_COLLECTION_DIRECTIVE,
            HOUYI_SPARSE_DIRECTIVE,
            HOUYI_FUSION_DIRECTIVE,
            HOUYI_ALPHA_DIRECTIVE,
        ] {
            assert!(
                directive.starts_with(HOUYI_DIRECTIVE_PREFIX),
                "directive {directive:?} must start with {HOUYI_DIRECTIVE_PREFIX:?}"
            );
        }
    }

    #[test]
    fn parse_pgwire_sparse_duplicate_directives_last_wins() {
        // 每次出现都校验；生效值为最后一个（与 ef_search 指令同策略）。
        // EN: Every occurrence is validated; the last one wins (same policy as
        // ef_search).
        let p = parse_houyi_pgwire_sql(
            "/* houyi:sparse=1:0.5 */ /* houyi:sparse=2:1.0,3:0.25 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t LIMIT 5",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::FusionSearch { query_sparse, .. } => {
                assert_eq!(query_sparse, vec![(2, 1.0), (3, 0.25)]);
            }
            _ => panic!("expected fusion search"),
        }
    }

    #[test]
    fn parse_pgwire_vector_duplicate_directives_last_wins() {
        // 与 sparse 指令同规：每次出现都校验，生效值为最后一个。
        // EN: Same rule as the sparse directive: every occurrence is validated
        // and the last one wins.
        let p = parse_houyi_pgwire_sql(
            "/* houyi:vector=9,9,9,9 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t LIMIT 3",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::VectorSearch { query_vector, .. } => {
                assert_eq!(query_vector, vec![1.0, 0.0, 0.0, 0.0]);
            }
            _ => panic!("expected vector search"),
        }
    }

    #[test]
    fn parse_pgwire_fusion_alpha_non_finite_stays_semantic() {
        // parser 只做结构解析；NaN/inf 的语义拒绝由服务端映射 22023。
        // EN: The parser only does structural parsing; NaN/inf are rejected
        // semantically server-side (22023).
        let p = parse_houyi_pgwire_sql(
            "/* houyi:alpha=NaN */ /* houyi:sparse=1:1 */ /* houyi:vector=1,0,0,0 */ SELECT id FROM t",
        )
        .unwrap();
        match p {
            HouyiPgWireSql::FusionSearch { alpha, .. } => {
                assert!(alpha.expect("alpha present").is_nan());
            }
            _ => panic!("expected fusion search"),
        }
    }

    // ── 新增 AST 解析测试 ────────────────────────────────────
    // EN: ── New AST parsing tests ─────────────────────────────

    #[test]
    fn extract_id_equals_single() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM docs WHERE id = 42".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => assert_eq!(pushed_down_bitset, Some(vec![42])),
            _ => panic!("expected hybrid plan with single id"),
        }
    }

    #[test]
    fn extract_id_in_list() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM docs WHERE id IN (1, 2, 3)".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => assert_eq!(pushed_down_bitset, Some(vec![1, 2, 3])),
            _ => panic!("expected hybrid plan"),
        }
    }

    #[test]
    fn plan_where_and_combines_ids() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM t WHERE id = 7 AND id = 9".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => {
                let mut ids = pushed_down_bitset.unwrap();
                ids.sort_unstable();
                assert_eq!(ids, vec![7, 9]);
            }
            _ => panic!("expected hybrid plan"),
        }
    }

    #[test]
    fn plan_range_where_returns_none_bitset() {
        // id > 5 AND id < 10 无法转换为有限 id 集合。
        // EN: id > 5 AND id < 10 can't be converted to a finite id set.
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM docs WHERE id > 5 AND id < 10".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => assert!(pushed_down_bitset.is_none()),
            _ => panic!("expected hybrid plan"),
        }
    }

    #[test]
    fn plan_simple_select_without_where_is_raw() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT 42".to_string(),
        });
        match plan {
            ExecutionPlan::RawSql(_) => {}
            _ => panic!("expected raw plan"),
        }
    }

    // ── 新增 SQL 模式测试 ────────────────────────────────
    // EN: ── New SQL mode tests ────────────────────────────────

    #[test]
    fn plan_or_combines_ids() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM t WHERE id = 7 OR id = 9".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => {
                let mut ids = pushed_down_bitset.unwrap();
                ids.sort_unstable();
                assert_eq!(ids, vec![7, 9]);
            }
            _ => panic!("expected hybrid"),
        }
    }

    #[test]
    fn plan_between_returns_none_bitset() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM t WHERE id BETWEEN 5 AND 10".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => {
                assert!(pushed_down_bitset.is_none());
            }
            _ => panic!("expected hybrid"),
        }
    }

    #[test]
    fn plan_gt_returns_none_bitset() {
        let plan = QueryPlanner::plan(PgWireQuery {
            sql: "SELECT id FROM t WHERE id > 5".to_string(),
        });
        match plan {
            ExecutionPlan::HybridSearch {
                pushed_down_bitset, ..
            } => {
                assert!(pushed_down_bitset.is_none());
            }
            _ => panic!("expected hybrid"),
        }
    }
}
