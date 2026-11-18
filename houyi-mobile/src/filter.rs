//! 轻量标量过滤器：标签/时间戳/数值过滤 → bitset。
//! Lightweight scalar filter: tag/timestamp/numeric → bitset for hybrid search.
//!
//! 与服务器 PAX 谓词树一脉相承，移动端用平坦列存储。
//! Mirrors server-side ScalarPredicate tree, optimized for mobile.

use std::collections::HashMap;

/// 标量列类型 / Scalar column type
/// EN: Scalar column type.
#[derive(Debug, Clone)]
pub enum ColType {
    Int,
    Float,
    Text,
    Timestamp, // unix seconds (i64)
}

/// 标量列存储 / Columnar scalar store
/// EN: Columnar scalar store.
pub struct ScalarStore {
    rows: usize,
    /// 每行对应的向量 id；旧 API 未提供 id 时按行号 0..n 填充。
    /// EN: Vector id for each scalar row; legacy APIs without an id use the
    /// row index 0..n.
    row_ids: Vec<u64>,
    /// id → 行号（显式 id 写入/覆盖 O(1)，避免逐行线性扫描）。
    /// EN: Map id to row index so explicit-id upserts are O(1) instead of a
    /// linear scan per write.
    row_index: HashMap<u64, usize>,
    columns: HashMap<String, (ColType, Vec<ScalarVal>)>,
}

/// 标量页快照：列按名称排序，保证同一状态写成稳定字节。
/// EN: Scalar-page snapshot: columns sorted by name so the same state always
/// encodes to stable bytes.
#[derive(Debug, Clone)]
pub(crate) struct ScalarStoreSnapshot {
    pub rows: usize,
    pub row_ids: Vec<u64>,
    pub columns: Vec<(String, ColType, Vec<ScalarVal>)>,
}

#[derive(Debug, Clone)]
pub enum ScalarVal {
    Int(i64),
    Float(f64),
    Text(String),
    Timestamp(i64),
    /// 缺列的**不可碰撞**占位：不是用户数据，任何等值/范围比较都不得命中。
    /// 旧实现以 `Text("")` 补齐，`eq ""` / `in [""]` 会误命中缺列行。
    /// EN: non-collidable padding for missing columns; it must never match.
    Missing,
}

impl ScalarStore {
    /// 创建空存储 / Create empty store
    /// EN: Create an empty store.
    pub fn new() -> Self {
        ScalarStore {
            rows: 0,
            row_ids: Vec::new(),
            row_index: HashMap::new(),
            columns: HashMap::new(),
        }
    }

    pub(crate) fn snapshot(&self) -> ScalarStoreSnapshot {
        let mut columns: Vec<(String, ColType, Vec<ScalarVal>)> = self
            .columns
            .iter()
            .map(|(name, (ty, values))| (name.clone(), ty.clone(), values.clone()))
            .collect();
        columns.sort_by(|a, b| a.0.cmp(&b.0));
        ScalarStoreSnapshot {
            rows: self.rows,
            row_ids: self.row_ids.clone(),
            columns,
        }
    }

    pub(crate) fn from_snapshot(snapshot: ScalarStoreSnapshot) -> Result<Self, String> {
        if snapshot.row_ids.len() != snapshot.rows {
            return Err(format!(
                "scalar row ids have {} entries, expected {}",
                snapshot.row_ids.len(),
                snapshot.rows
            ));
        }
        let mut columns = HashMap::with_capacity(snapshot.columns.len());
        for (name, ty, values) in snapshot.columns {
            if values.len() != snapshot.rows {
                return Err(format!(
                    "scalar column '{name}' has {} values, expected {}",
                    values.len(),
                    snapshot.rows
                ));
            }
            if columns.insert(name.clone(), (ty, values)).is_some() {
                return Err(format!("duplicate scalar column '{name}'"));
            }
        }
        let mut row_index = HashMap::with_capacity(snapshot.row_ids.len());
        for (idx, &id) in snapshot.row_ids.iter().enumerate() {
            if row_index.insert(id, idx).is_some() {
                return Err(format!("duplicate scalar row id {id}"));
            }
        }
        Ok(ScalarStore {
            rows: snapshot.rows,
            row_ids: snapshot.row_ids,
            row_index,
            columns,
        })
    }

    /// 添加一行标量数据 / Append a row of scalars
    /// 行号兼容仅在显式 id 尚未占用该行号时成立；混合使用时请一律改用显式 id。
    /// EN: Append a row of scalar data. Row-index compatibility only holds while
    /// explicit ids have not occupied that index; mixed usage should use
    /// explicit ids exclusively.
    pub fn push_row(&mut self, row: &[(&str, ScalarVal)]) {
        // 旧 API 按行号取 id；显式 id 可能已占用该行号，因此跳到首个空闲 id，
        // 避免同一 id 出现两行（否则 row_index 与 row_ids 会指向不同行）。
        // EN: Legacy callers derive the id from the row count; explicit ids may
        // already occupy that slot, so advance to the first free id. Otherwise
        // one id would map to two rows and row_index would disagree with row_ids.
        let mut id = self.rows as u64;
        while self.row_index.contains_key(&id) {
            id += 1;
        }
        self.write_row_with_id(id, row);
    }

    /// 以显式向量 id 写入标量行（混合检索按 id 关联，而不是行号）；同 id
    /// 已存在时整行覆盖而非追加，保证 row_ids 唯一。内部写入原语。
    /// EN: Write a scalar row for an explicit vector id (hybrid search associates
    /// by id, not row index). An existing id is replaced wholesale instead of
    /// appended, so row_ids stays unique. Internal write primitive.
    fn write_row_with_id(&mut self, id: u64, row: &[(&str, ScalarVal)]) {
        if let Some(&idx) = self.row_index.get(&id) {
            self.overwrite_row_at(idx, row);
            return;
        }
        for (col, val) in row {
            let (_col_type, vec) = self
                .columns
                .entry(col.to_string())
                .or_insert_with(|| (val.col_type(), Vec::new()));
            vec.push(val.clone());
        }
        // 补齐缺失列 / fill missing columns with null-like placeholder
        // EN: Fill missing columns with a null-like placeholder.
        for (_t, vec) in self.columns.values_mut() {
            while vec.len() <= self.rows {
                vec.push(ScalarVal::Missing);
            }
        }
        self.row_index.insert(id, self.row_ids.len());
        self.row_ids.push(id);
        self.rows += 1;
    }

    /// 以显式 id 覆盖写入整行标量（不存在则追加）；旧行值全部先置 Missing，
    /// 避免旧标签在更新后仍命中。这是显式 id 写入的规范入口。
    /// EN: Replace the whole scalar row for an explicit id (append if absent);
    /// prior values are cleared to Missing so stale tags cannot keep matching.
    /// This is the canonical explicit-id entry point.
    pub fn upsert_row_with_id(&mut self, id: u64, row: &[(&str, ScalarVal)]) {
        self.write_row_with_id(id, row);
    }

    /// 覆盖指定行：先把该行所有列置 Missing，再写入新列值。
    /// EN: Overwrite one row by clearing every column at that index, then
    /// writing the new values.
    fn overwrite_row_at(&mut self, idx: usize, row: &[(&str, ScalarVal)]) {
        for (_ty, vec) in self.columns.values_mut() {
            if let Some(slot) = vec.get_mut(idx) {
                *slot = ScalarVal::Missing;
            }
        }
        for (col, val) in row {
            let (_ty, vec) = self
                .columns
                .entry(col.to_string())
                .or_insert_with(|| (val.col_type(), vec![ScalarVal::Missing; self.rows]));
            vec.resize(self.rows, ScalarVal::Missing);
            vec[idx] = val.clone();
        }
    }

    /// 过滤 → 匹配行索引列表。
    /// Filter → matching row indices.
    ///
    /// 返回值区分三种状态（**调用方必须区分**，否则「无过滤条件」与「命中
    /// 为空」会被混为一谈）：
    /// - `None`：没有过滤条件 → 调用方应跳过掩码；
    /// - `Some(vec![])`：有条件但无命中（含未知列）→ 调用方必须返回空结果。
    ///
    /// 为什么必须区分（WHY）：旧实现统一返回空 Vec，`search_hybrid` 把
    /// 「空」当「不过滤」跳过掩码——未知列或零命中的过滤反而被放宽成**全量**
    /// 结果（OCR 批次二十六 HIGH）。
    /// EN: `None` = no filter, `Some(empty)` = matched nothing (including an
    /// unknown column). Conflating them made `search_hybrid` skip the mask and
    /// return the full result set.
    pub fn filter(&self, filters: &[FilterOp]) -> Option<Vec<u64>> {
        if filters.is_empty() {
            return None;
        }
        if self.rows == 0 {
            return Some(Vec::new());
        }
        let mut bitset = vec![true; self.rows];
        for f in filters {
            // 未知列 ⇒ 空结果（与 houyi-core 的 `filter_multi_column` 同口径）：
            // 旧实现把查不到的列当「无此条件」跳过，静默放宽了结果集。
            // EN: Unknown column ⇒ empty result (same contract as houyi-core);
            // skipping it silently widened the result set.
            let Some((_t, col)) = self.columns.get(&f.column) else {
                return Some(Vec::new());
            };
            let mut col_match = vec![false; self.rows];
            for (idx, val) in col.iter().enumerate() {
                col_match[idx] = f.matches(val);
            }
            for i in 0..self.rows {
                bitset[i] = bitset[i] && col_match[i];
            }
        }
        Some(
            bitset
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| if m { Some(i as u64) } else { None })
                .collect(),
        )
    }

    /// 过滤 → 匹配的**向量 id** 列表（混合检索用；行号只用于内部列访问）。
    /// EN: Filter to the matching **vector ids** for hybrid search; row indices
    /// remain internal to column access.
    pub fn filter_ids(&self, filters: &[FilterOp]) -> Option<Vec<u64>> {
        self.filter(filters).map(|indices| {
            indices
                .into_iter()
                .map(|i| self.row_ids[i as usize])
                .collect()
        })
    }
}

impl ScalarVal {
    fn col_type(&self) -> ColType {
        match self {
            ScalarVal::Int(_) => ColType::Int,
            ScalarVal::Float(_) => ColType::Float,
            ScalarVal::Text(_) => ColType::Text,
            ScalarVal::Timestamp(_) => ColType::Timestamp,
            ScalarVal::Missing => ColType::Text,
        }
    }
}

/// 过滤操作 / Filter operation
/// EN: Filter operation.
#[derive(Debug, Clone)]
pub struct FilterOp {
    pub column: String,
    pub op: FilterOpType,
    pub value: ScalarVal,
}

#[derive(Debug, Clone)]
pub enum FilterOpType {
    Eq,
    Gt,
    Lt,
    GtEq,
    LtEq,
    In(Vec<ScalarVal>),
}

impl FilterOp {
    fn matches(&self, val: &ScalarVal) -> bool {
        use std::cmp::Ordering::{Equal, Greater, Less};
        match &self.op {
            FilterOpType::Eq => scalar_eq(val, &self.value),
            FilterOpType::Gt => compare(val, &self.value) == Some(Greater),
            FilterOpType::Lt => compare(val, &self.value) == Some(Less),
            FilterOpType::GtEq => matches!(compare(val, &self.value), Some(Greater | Equal)),
            FilterOpType::LtEq => matches!(compare(val, &self.value), Some(Less | Equal)),
            FilterOpType::In(list) => list.iter().any(|v| scalar_eq(v, val)),
        }
    }
}

impl PartialEq for ScalarVal {
    fn eq(&self, other: &Self) -> bool {
        // 同类型浮点用 `==`（与 houyi-core 的 `ScalarValue` 派生语义一致：
        // NaN != NaN、-0.0 == 0.0）；跨类型由 `scalar_eq` 做数值提升。
        // EN: Same-type floats use `==` (matching houyi-core's derived `ScalarValue`
        // EN: semantics: NaN != NaN, -0.0 == 0.0); cross-type comparison goes through
        // EN: `scalar_eq` numeric promotion.
        matches!((self, other),
            (ScalarVal::Int(a), ScalarVal::Int(b)) if a == b)
            || matches!((self, other),
            (ScalarVal::Float(a), ScalarVal::Float(b)) if a == b)
            || matches!((self, other),
            (ScalarVal::Text(a), ScalarVal::Text(b)) if a == b)
            || matches!((self, other),
            (ScalarVal::Timestamp(a), ScalarVal::Timestamp(b)) if a == b)
            || matches!((self, other), (ScalarVal::Missing, ScalarVal::Missing))
    }
}

/// 标量相等：数值类跨 Int/Float 按数值语义；`Missing` 与任何用户值都不等。
/// Scalar equality — numeric promotion across Int/Float; `Missing` never
/// matches user data.
fn scalar_eq(a: &ScalarVal, b: &ScalarVal) -> bool {
    use ScalarVal::*;
    match (a, b) {
        (Missing, _) | (_, Missing) => false,
        (Int(x), Float(y)) => (*x as f64) == *y,
        (Float(x), Int(y)) => *x == (*y as f64),
        _ => a == b,
    }
}

/// 标量值比较；**类型不可比时返回 `None`**（不再退化为 `Equal`）。
/// Typed scalar comparison — `None` when the operands are not comparable.
///
/// 为什么必须区分「相等」与「不可比」（WHY）：旧实现以 `_ => Equal` 兜底，而
/// `GtEq`/`LtEq` 把 `Equal` 视为命中——于是 `price >= 5` 会命中缺列行
/// （`push_row` 以 `Text("")` 补齐）以及类型不同的行，产生静默假阳性。与
/// `houyi-core` 的 `compare` 同口径修复（见 `houyi-core/src/lib.rs`）。
/// EN: The old `_ => Equal` fallback made `GtEq`/`LtEq` accept incomparable
/// operands (missing-column padding `Text("")`, or a different type) — silent
/// false positives. Mirrors the fix in `houyi-core`'s `compare`.
fn compare(a: &ScalarVal, b: &ScalarVal) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (ScalarVal::Int(x), ScalarVal::Int(y)) => Some(x.cmp(y)),
        (ScalarVal::Float(x), ScalarVal::Float(y)) => Some(x.total_cmp(y)),
        (ScalarVal::Int(x), ScalarVal::Float(y)) => Some((*x as f64).total_cmp(y)),
        (ScalarVal::Float(x), ScalarVal::Int(y)) => Some(x.total_cmp(&(*y as f64))),
        (ScalarVal::Timestamp(x), ScalarVal::Timestamp(y)) => Some(x.cmp(y)),
        (ScalarVal::Text(x), ScalarVal::Text(y)) => Some(x.cmp(y)),
        // Missing 是占位不是数据：与任何值都不可比。
        // EN: Missing is a placeholder, not data: it is incomparable with any value.
        _ => None,
    }
}

/// 解析标量值字符串 / Parse a scalar value from its wire string.
/// "int:42" | "float:0.8" | "text:beach" | "ts:1717200000"
pub(crate) fn parse_scalar(raw: &str) -> Result<ScalarVal, String> {
    if let Some(v) = raw.strip_prefix("int:") {
        v.parse()
            .map(ScalarVal::Int)
            .map_err(|e| format!("invalid int scalar {raw:?}: {e}"))
    } else if let Some(v) = raw.strip_prefix("float:") {
        v.parse()
            .map(ScalarVal::Float)
            .map_err(|e| format!("invalid float scalar {raw:?}: {e}"))
    } else if let Some(v) = raw.strip_prefix("text:") {
        Ok(ScalarVal::Text(v.to_string()))
    } else if let Some(v) = raw.strip_prefix("ts:") {
        v.parse()
            .map(ScalarVal::Timestamp)
            .map_err(|e| format!("invalid timestamp scalar {raw:?}: {e}"))
    } else {
        Ok(ScalarVal::Text(raw.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_and_legacy_writes_keep_unique_row_ids() {
        let mut store = ScalarStore::new();
        store.upsert_row_with_id(1, &[("tag", ScalarVal::Text("explicit".into()))]);
        // 旧 API 的 id 由行号推导，此时行号 0 被显式 id 1 之后的下一个空闲
        // 行号本应是 1，但 1 已占用；必须跳过而不是生成重复 id。
        // EN: The legacy id comes from the row count; slot 1 is already taken,
        // so it must advance instead of emitting a duplicate id.
        store.push_row(&[("tag", ScalarVal::Text("legacy".into()))]);
        let snap = store.snapshot();
        assert_eq!(snap.rows, 2);
        let unique: std::collections::HashSet<u64> = snap.row_ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            2,
            "row ids must stay unique: {:?}",
            snap.row_ids
        );

        // 同 id 再写必须整行覆盖，不能新增第三行。
        // EN: Rewriting the same id must replace the row, not add a third one.
        store.upsert_row_with_id(1, &[("tag", ScalarVal::Text("replaced".into()))]);
        let snap = store.snapshot();
        assert_eq!(snap.rows, 2);
        let ids = store.filter_ids(&[FilterOp {
            column: "tag".into(),
            op: FilterOpType::Eq,
            value: ScalarVal::Text("replaced".into()),
        }]);
        assert_eq!(ids, Some(vec![1]));
    }

    #[test]
    fn test_scalar_filter() {
        let mut store = ScalarStore::new();
        store.push_row(&[
            ("tag", ScalarVal::Text("beach".into())),
            ("date", ScalarVal::Timestamp(1717200000)),
        ]);
        store.push_row(&[
            ("tag", ScalarVal::Text("sunset".into())),
            ("date", ScalarVal::Timestamp(1717300000)),
        ]);
        store.push_row(&[
            ("tag", ScalarVal::Text("beach".into())),
            ("date", ScalarVal::Timestamp(1717400000)),
        ]);

        // 过滤 tag=beach
        // EN: Filter tag=beach
        let ids = store
            .filter(&[FilterOp {
                column: "tag".into(),
                op: FilterOpType::Eq,
                value: ScalarVal::Text("beach".into()),
            }])
            .expect("filters were provided");
        assert_eq!(ids, vec![0, 2]);

        // 过滤 date > 1717250000
        // EN: Filter date > 1717250000
        let ids = store
            .filter(&[FilterOp {
                column: "date".into(),
                op: FilterOpType::Gt,
                value: ScalarVal::Timestamp(1717250000),
            }])
            .expect("filters were provided");
        assert_eq!(ids, vec![1, 2]);

        // 组合条件：tag=beach AND date > 1717250000
        // EN: Combined: tag=beach AND date > 1717250000
        let ids = store
            .filter(&[
                FilterOp {
                    column: "tag".into(),
                    op: FilterOpType::Eq,
                    value: ScalarVal::Text("beach".into()),
                },
                FilterOp {
                    column: "date".into(),
                    op: FilterOpType::Gt,
                    value: ScalarVal::Timestamp(1717250000),
                },
            ])
            .expect("filters were provided");
        assert_eq!(ids, vec![2]); // only row 2: tag=beach AND date=1717400000 > 1717250000
    }

    /// 回归：范围比较在类型不可比时不得命中——旧实现 `_ => Equal` + `GtEq` 把
    /// `Equal` 视为命中，`price >= 5` 会命中「缺 price 列」的行（补齐 `Text("")`）
    /// 与 `price` 是字符串的行。与 houyi-core `tests/filter_oracle.rs` 同口径。
    /// Regression: range ops must miss on incomparable types (missing-column
    /// padding and cross-type values), mirroring houyi-core's filter oracle.
    #[test]
    fn test_range_filter_never_matches_missing_or_cross_type() {
        let mut store = ScalarStore::new();
        store.push_row(&[("price", ScalarVal::Int(10))]);
        store.push_row(&[]); // 缺 price 列 ⇒ 以 Text("") 补齐
        store.push_row(&[("price", ScalarVal::Text("unknown".into()))]);

        let gte = store
            .filter(&[FilterOp {
                column: "price".into(),
                op: FilterOpType::GtEq,
                value: ScalarVal::Int(5),
            }])
            .expect("filters were provided");
        assert_eq!(gte, vec![0], "gte must only match the numeric row");

        let lte = store
            .filter(&[FilterOp {
                column: "price".into(),
                op: FilterOpType::LtEq,
                value: ScalarVal::Int(100),
            }])
            .expect("filters were provided");
        assert_eq!(lte, vec![0], "lte must only match the numeric row");
    }

    /// 回归：Int/Float 混写同列时范围比较按数值语义。
    /// EN: Regression: Int/Float mixed in the same column compares numerically for range
    /// EN: ops.
    #[test]
    fn test_numeric_cross_type_range() {
        let mut store = ScalarStore::new();
        store.push_row(&[("p", ScalarVal::Int(10))]);
        store.push_row(&[("p", ScalarVal::Float(3.0))]);

        let gt = store
            .filter(&[FilterOp {
                column: "p".into(),
                op: FilterOpType::Gt,
                value: ScalarVal::Float(5.5),
            }])
            .expect("filters were provided");
        assert_eq!(gt, vec![0]);
        let lt = store
            .filter(&[FilterOp {
                column: "p".into(),
                op: FilterOpType::Lt,
                value: ScalarVal::Float(5.5),
            }])
            .expect("filters were provided");
        assert_eq!(lt, vec![1]);
    }

    /// 回归：未知列 ⇒ 空结果（不得被当作「无此条件」跳过）。
    /// EN: Regression: unknown column ⇒ empty result (must not be treated as "no such
    /// EN: condition" and skipped).
    #[test]
    fn test_unknown_column_yields_empty() {
        let mut store = ScalarStore::new();
        store.push_row(&[("tag", ScalarVal::Text("beach".into()))]);
        let ids = store
            .filter(&[FilterOp {
                column: "nope".into(),
                op: FilterOpType::Eq,
                value: ScalarVal::Text("beach".into()),
            }])
            .expect("filters were provided");
        assert!(ids.is_empty(), "unknown column must match nothing");
    }
}

/// 回归（OCR 批次二十六）：`eq` / `in` 与范围比较同口径做数值提升。
/// EN: Regression (OCR batch 26): `eq` / `in` use the same numeric promotion as range
/// EN: comparison.
#[test]
fn test_equality_promotes_int_float() {
    let mut store = ScalarStore::new();
    store.push_row(&[("p", ScalarVal::Float(10.0))]);
    store.push_row(&[("p", ScalarVal::Int(10))]);
    store.push_row(&[("p", ScalarVal::Int(11))]);

    let eq = store
        .filter(&[FilterOp {
            column: "p".into(),
            op: FilterOpType::Eq,
            value: ScalarVal::Int(10),
        }])
        .expect("filters were provided");
    assert_eq!(eq, vec![0, 1]);
    let in_ids = store
        .filter(&[FilterOp {
            column: "p".into(),
            op: FilterOpType::In(vec![ScalarVal::Float(10.0)]),
            value: ScalarVal::Float(10.0),
        }])
        .expect("filters were provided");
    assert_eq!(in_ids, vec![0, 1]);
}

/// 回归（OCR 批次二十六）：缺列占位 `Missing` 不得与用户数据（含空串）
/// 发生等值命中。
/// EN: Regression (OCR batch 26): the missing-column `Missing` placeholder must not
/// EN: match user data (including the empty string) under equality.
#[test]
fn test_missing_padding_never_equals_user_values() {
    let mut store = ScalarStore::new();
    store.push_row(&[("p", ScalarVal::Text("".into()))]);
    store.push_row(&[]); // 缺列 ⇒ Missing
    let ids = store
        .filter(&[FilterOp {
            column: "p".into(),
            op: FilterOpType::Eq,
            value: ScalarVal::Text("".into()),
        }])
        .expect("filters were provided");
    assert_eq!(ids, vec![0], "missing padding must not match eq \"\"");
}

/// 回归（OCR 批次二十六 HIGH）：未知列过滤在 `search_hybrid` 层必须得到
/// **空结果**，不得因为「掩码为空」被放宽成全量结果。
/// EN: Regression (OCR batch 26 HIGH): unknown-column filters must yield an **empty
/// EN: result** at the `search_hybrid` layer; an empty mask must not be widened to all
/// EN: rows.
#[test]
fn test_search_hybrid_unknown_column_returns_empty() {
    use crate::db::MobileDB;
    let db = MobileDB::new();
    db.get_or_create("c", 4, "flat").unwrap();
    db.insert("c", 0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    db.insert("c", 1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
    db.insert_scalars("c", &[("tag", "text:a")]).unwrap();
    db.insert_scalars("c", &[("tag", "text:b")]).unwrap();

    let all = db
        .search_hybrid("c", &[1.0, 0.0, 0.0, 0.0], 10, &[])
        .unwrap();
    assert_eq!(all.len(), 2, "no filters must return everything");

    let unknown = db
        .search_hybrid(
            "c",
            &[1.0, 0.0, 0.0, 0.0],
            10,
            &[FilterOp {
                column: "nope".into(),
                op: FilterOpType::Eq,
                value: ScalarVal::Text("a".into()),
            }],
        )
        .unwrap();
    assert!(
        unknown.is_empty(),
        "unknown column must yield nothing, got {unknown:?}"
    );

    let hit = db
        .search_hybrid(
            "c",
            &[1.0, 0.0, 0.0, 0.0],
            10,
            &[FilterOp {
                column: "tag".into(),
                op: FilterOpType::Eq,
                value: ScalarVal::Text("a".into()),
            }],
        )
        .unwrap();
    assert_eq!(hit.len(), 1, "known column must filter, got {hit:?}");
}
