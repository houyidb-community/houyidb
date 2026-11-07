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
}

const HOUYI_VECTOR_DIRECTIVE: &str = "/* houyi:vector=";
// 指令前缀：在 SQL 注释中嵌入查询向量、ef_search 与 collection。
// EN: Directive prefixes: embed query vector, ef_search, and collection in SQL comments.
const HOUYI_EF_SEARCH_DIRECTIVE: &str = "/* houyi:ef_search=";
const HOUYI_COLLECTION_DIRECTIVE: &str = "/* houyi:collection=";

fn strip_houyi_ef_search_directives(sql: &str) -> Result<(String, Option<usize>), String> {
    let mut owned = sql.trim().to_string();
    let mut last_ef: Option<usize> = None;
    while let Some(start) = owned.find(HOUYI_EF_SEARCH_DIRECTIVE) {
        let after = &owned[start + HOUYI_EF_SEARCH_DIRECTIVE.len()..];
        let end_rel = after
            .find("*/")
            .ok_or_else(|| "unterminated houyi:ef_search comment".to_string())?;
        let num_str = after[..end_rel].trim();
        let n: usize = num_str
            .parse()
            .map_err(|_| format!("invalid houyi:ef_search value: {num_str}"))?;
        if n == 0 {
            return Err("houyi:ef_search must be > 0".to_string());
        }
        last_ef = Some(n);
        let remove_end = start + HOUYI_EF_SEARCH_DIRECTIVE.len() + end_rel + 2;
        let mut new_s = String::new();
        new_s.push_str(owned[..start].trim_end());
        let tail = owned[remove_end..].trim_start();
        if !new_s.is_empty() && !tail.is_empty() {
            new_s.push(' ');
        }
        new_s.push_str(tail);
        owned = new_s.trim().to_string();
    }
    Ok((owned, last_ef))
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

/// 供扩展查询 Describe 的 `RowDescription` 单列名：向量检索为 `id`，否则为 `value`（与 Simple Query 结果列一致）。
/// EN: `RowDescription` column for extended-query Describe: `id` for vector search, else `value`.
pub fn pgwire_row_column_name_for_sql(sql: &str) -> Result<&'static str, String> {
    match parse_houyi_pgwire_sql(sql) {
        Ok(HouyiPgWireSql::VectorSearch { .. }) => Ok("id"),
        Ok(HouyiPgWireSql::Plain(_)) => Ok("value"),
        Err(e) => Err(e),
    }
}

/// 解析 Simple Query 文本：提取 `/* houyi:vector=... */`、`/* houyi:ef_search=N */` 与可选 `LIMIT n`。
/// EN: Parses Simple Query text: `houyi:vector`, `houyi:ef_search`, and optional `LIMIT n`.
pub fn parse_houyi_pgwire_sql(sql: &str) -> Result<HouyiPgWireSql, String> {
    let (work, ef_search_override) = strip_houyi_ef_search_directives(sql)?;
    if !work.contains(HOUYI_VECTOR_DIRECTIVE) {
        return Ok(HouyiPgWireSql::Plain(work));
    }
    let start = work
        .find(HOUYI_VECTOR_DIRECTIVE)
        .ok_or_else(|| "houyi:vector directive missing".to_string())?;
    let after = &work[start + HOUYI_VECTOR_DIRECTIVE.len()..];
    let end_rel = after
        .find("*/")
        .ok_or_else(|| "unterminated houyi:vector comment".to_string())?;
    let vec_part = after[..end_rel].trim();
    let mut query_vector = Vec::new();
    for token in vec_part.split(',') {
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
    let directive_end = start + HOUYI_VECTOR_DIRECTIVE.len() + end_rel + 2;
    let mut sql_for_planner = String::new();
    sql_for_planner.push_str(work[..start].trim_end());
    let tail = work[directive_end..].trim_start();
    if !sql_for_planner.is_empty() && !tail.is_empty() {
        sql_for_planner.push(' ');
    }
    sql_for_planner.push_str(tail);
    let sql_for_planner = sql_for_planner.trim().to_string();

    let k = extract_limit_k(&sql_for_planner).unwrap_or(10);
    if k == 0 {
        return Err("LIMIT must be >= 1 for vector search".to_string());
    }
    Ok(HouyiPgWireSql::VectorSearch {
        sql_for_planner,
        query_vector,
        k,
        ef_search_override,
    })
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
        ExecutionPlan, HouyiPgWireSql, PgWireQuery, QueryPlanner,
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
