//! pgwire 协议处理：握手、Simple Query、Extended Query、向量编解码。
//! pgwire protocol: handshake, simple/extended query, vector encode/decode.

use crate::index_backend::IndexBackend;
use crate::types::*;
use houyi_core::bus::MultiTaskBusRouter;
use houyi_core::interface::QueryTask;
use houyi_core::{HouyiPage, ScalarPredicate};
use houyi_query::pgwire::{
    decode_frontend_message, decode_startup_packet, encode_authentication_cleartext_password,
    encode_authentication_ok, encode_bind_complete, encode_close_complete, encode_command_complete,
    encode_error_response, encode_parameter_description_oids, encode_parameter_status,
    encode_parse_complete, encode_portal_suspended, encode_ready_for_query, BindParamValue,
    FrontendMessage, GSSENC_REQUEST_CODE, SSL_REQUEST_CODE,
};
use houyi_query::{
    parse_houyi_pgwire_sql, ExecutionPlan, HouyiPgWireSql, PgWireQuery, QueryPlanner,
};
use houyi_vector::VectorId;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::TlsAcceptor;
fn pgwire_decode_error_code(err_msg: &str) -> &'static str {
    if err_msg.contains("only text parameter format is supported")
        || err_msg.contains("null parameter is not supported")
        || err_msg.contains("prepared statement parameters not supported")
        || err_msg.contains("unsupported parameter type OID")
        || err_msg.contains("unsupported result format code")
        || err_msg.contains("unsupported result format count")
        || err_msg.contains("unsupported parameter OID/format combination")
        || err_msg.contains("unsupported binary parameter type OID")
    {
        "0A000"
    } else {
        "08P01"
    }
}

fn pgwire_supported_param_oid(oid: u32) -> bool {
    matches!(oid, 0 | 16 | 17 | 20 | 23 | 25 | 700 | 701 | 1021 | 1043)
}

fn pgwire_supported_param_oid_and_format(oid: u32, format: i16) -> bool {
    match format {
        0 => pgwire_supported_param_oid(oid),
        1 => matches!(oid, 0 | 16 | 17 | 20 | 23 | 25 | 700 | 701 | 1021 | 1043),
        _ => false,
    }
}

fn pgwire_escape_sql_text(raw: &str) -> String {
    raw.replace('\'', "''")
}

pub fn pgwire_float_list_to_directive_text(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn pgwire_split_plain_select_exprs(sql: &str) -> Option<Vec<String>> {
    let trimmed = sql.trim().trim_end_matches(';').trim();
    let upper = trimmed.to_ascii_uppercase();
    if !upper.starts_with("SELECT ") {
        return None;
    }
    let exprs = trimmed[6..].trim();
    if exprs.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_single_quotes = false;
    for ch in exprs.chars() {
        match ch {
            '\'' => {
                in_single_quotes = !in_single_quotes;
                current.push(ch);
            }
            ',' if !in_single_quotes => {
                let expr = current.trim();
                if expr.is_empty() {
                    return None;
                }
                out.push(expr.to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let tail = current.trim();
    if tail.is_empty() {
        return None;
    }
    out.push(tail.to_string());
    Some(out)
}

fn pgwire_result_columns_for_sql_with_params(
    sql: &str,
    param_oids: &[u32],
) -> Result<Vec<PgWireColumnMeta>, String> {
    if sql.contains("/* houyi:upsert_batch=$") {
        return Ok(vec![PgWireColumnMeta {
            name: "inserted".to_string(),
            type_oid: 20,
        }]);
    }
    if sql.contains("/* houyi:vector=$") {
        return Ok(vec![PgWireColumnMeta {
            name: "id".to_string(),
            type_oid: 20,
        }]);
    }
    match parse_houyi_pgwire_sql(sql) {
        Ok(HouyiPgWireSql::VectorSearch { .. }) => Ok(vec![PgWireColumnMeta {
            name: "id".to_string(),
            type_oid: 20,
        }]),
        Ok(HouyiPgWireSql::Plain(s)) => {
            let exprs = pgwire_split_plain_select_exprs(&s).ok_or_else(|| {
                "only simple SELECT literal[, literal...] is supported".to_string()
            })?;
            let columns = exprs
                .iter()
                .enumerate()
                .map(|(idx, expr)| {
                    let type_oid =
                        if let Some((_placeholder_idx, oid)) = pgwire_placeholder_cast_oid(expr) {
                            oid
                        } else if let Some(placeholder) = expr.strip_prefix('$') {
                            if let Ok(param_idx) = placeholder.parse::<usize>() {
                                if param_idx >= 1 && param_idx <= param_oids.len() {
                                    let oid = param_oids[param_idx - 1];
                                    if oid == 0 {
                                        25
                                    } else {
                                        oid
                                    }
                                } else {
                                    25
                                }
                            } else {
                                25
                            }
                        } else if let Some((oid, _value)) = pgwire_plain_cast_value(expr) {
                            oid
                        } else {
                            pgwire_plain_scalar_oid(expr).unwrap_or(25)
                        };
                    PgWireColumnMeta {
                        name: if exprs.len() == 1 {
                            "value".to_string()
                        } else {
                            format!("column{}", idx + 1)
                        },
                        type_oid,
                    }
                })
                .collect();
            Ok(columns)
        }
        Err(e) => Err(e),
    }
}

fn pgwire_plain_scalar_oid(expr: &str) -> Option<u32> {
    let lower = expr.to_ascii_lowercase();
    if lower == "true" || lower == "false" {
        Some(16)
    } else if expr.contains('.') {
        if expr.parse::<f32>().map(|v| v.is_finite()).unwrap_or(false) {
            Some(700)
        } else if expr.parse::<f64>().map(|v| v.is_finite()).unwrap_or(false) {
            Some(701)
        } else {
            None
        }
    } else if expr.parse::<i32>().is_ok() {
        Some(23)
    } else if expr.parse::<i64>().is_ok() {
        Some(20)
    } else {
        None
    }
}

fn pgwire_type_name_oid(type_name: &str) -> Option<u32> {
    match type_name.trim().to_ascii_lowercase().as_str() {
        "bool" | "boolean" => Some(16),
        "int4" | "integer" => Some(23),
        "int8" | "bigint" => Some(20),
        "text" => Some(25),
        "varchar" | "character varying" => Some(1043),
        "float4" | "real" => Some(700),
        "float8" | "double precision" => Some(701),
        _ => None,
    }
}

fn pgwire_placeholder_cast_oid(expr: &str) -> Option<(usize, u32)> {
    let (placeholder, ty) = expr.split_once("::")?;
    let idx = placeholder.strip_prefix('$')?.parse::<usize>().ok()?;
    let oid = pgwire_type_name_oid(ty)?;
    Some((idx, oid))
}

fn pgwire_plain_cast_value(expr: &str) -> Option<(u32, String)> {
    let (raw, ty) = expr.split_once("::")?;
    let oid = pgwire_type_name_oid(ty)?;
    let value = match oid {
        16 => {
            let lowered = raw.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "true" | "t" | "1" => "true".to_string(),
                "false" | "f" | "0" => "false".to_string(),
                _ => return None,
            }
        }
        20 => raw.trim().parse::<i64>().ok()?.to_string(),
        23 => raw.trim().parse::<i32>().ok()?.to_string(),
        25 | 1043 => raw.trim().trim_matches('\'').to_string(),
        700 => raw.trim().parse::<f32>().ok()?.to_string(),
        701 => raw.trim().parse::<f64>().ok()?.to_string(),
        _ => return None,
    };
    Some((oid, value))
}

fn infer_pgwire_param_oids(sql: &str, param_count: usize, provided_oids: Vec<u32>) -> Vec<u32> {
    let mut oids = if provided_oids.is_empty() && param_count > 0 {
        vec![0_u32; param_count]
    } else {
        provided_oids
    };
    if oids.len() < param_count {
        oids.resize(param_count, 0);
    }
    if let Some(exprs) = pgwire_split_plain_select_exprs(sql) {
        for expr in exprs {
            if let Some((idx, oid)) = pgwire_placeholder_cast_oid(&expr) {
                if idx >= 1 && idx <= oids.len() && oids[idx - 1] == 0 {
                    oids[idx - 1] = oid;
                }
            }
        }
    }
    if let Some(idx) = pgwire_vector_placeholder_param_index(sql) {
        if idx >= 1 && idx <= oids.len() && oids[idx - 1] == 0 {
            oids[idx - 1] = 17;
        }
    }
    if let Some(idx) = pgwire_upsert_batch_placeholder_param_index(sql) {
        if idx >= 1 && idx <= oids.len() && oids[idx - 1] == 0 {
            oids[idx - 1] = 17;
        }
    }
    oids
}

fn pgwire_statement_entry(conn: &PgWireConnState, name: &str) -> Option<PreparedStatementEntry> {
    if name.is_empty() {
        conn.unnamed_stmt
            .as_ref()
            .map(|sql| PreparedStatementEntry {
                sql: sql.clone(),
                param_count: conn.unnamed_param_count,
                param_oids: conn.unnamed_param_oids.clone(),
            })
    } else {
        conn.named_statements.get(name).cloned()
    }
}

fn infer_pgwire_placeholder_count(sql: &str) -> usize {
    let bytes = sql.as_bytes();
    let mut i = 0usize;
    let mut max_n = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(n) = sql[start..end].parse::<usize>() {
                    max_n = max_n.max(n);
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
    max_n
}

fn pgwire_normalize_result_formats(
    result_formats: &[i16],
    column_count: usize,
) -> Result<Vec<i16>, String> {
    match result_formats {
        [] => Ok(vec![0; column_count]),
        [fmt] if *fmt == 0 || *fmt == 1 => Ok(vec![*fmt; column_count]),
        [fmt] => Err(format!(
            "unsupported result format code {} (only text/binary are supported)",
            fmt
        )),
        many if many.len() == column_count => {
            if let Some(bad) = many.iter().find(|fmt| **fmt != 0 && **fmt != 1) {
                Err(format!(
                    "unsupported result format code {} (only text/binary are supported)",
                    bad
                ))
            } else {
                Ok(many.to_vec())
            }
        }
        many => Err(format!(
            "unsupported result format count {} for {}-column portal",
            many.len(),
            column_count
        )),
    }
}

pub(crate) fn encode_row_description_columns(
    columns: &[PgWireColumnMeta],
    formats: &[i16],
) -> Result<Vec<u8>, String> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&(columns.len() as i16).to_be_bytes());
    for (idx, col) in columns.iter().enumerate() {
        let format = *formats.get(idx).unwrap_or(&0);
        payload.extend_from_slice(col.name.as_bytes());
        payload.push(0);
        payload.extend_from_slice(&0_i32.to_be_bytes());
        payload.extend_from_slice(&0_i16.to_be_bytes());
        payload.extend_from_slice(&(col.type_oid as i32).to_be_bytes());
        let type_size = match col.type_oid {
            20 => 8_i16,
            23 | 700 => 4_i16,
            16 => 1_i16,
            701 => 8_i16,
            25 | 1043 => -1_i16,
            _ => -1_i16,
        };
        payload.extend_from_slice(&type_size.to_be_bytes());
        payload.extend_from_slice(&0_i32.to_be_bytes());
        payload.extend_from_slice(&format.to_be_bytes());
    }
    let len = (payload.len() + 4) as i32;
    let mut out = Vec::new();
    out.push(b'T');
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

pub(crate) fn encode_data_row_columns(
    values: &[String],
    columns: &[PgWireColumnMeta],
    formats: &[i16],
) -> Result<Vec<u8>, String> {
    if values.len() != columns.len() {
        return Err(format!(
            "data row column/value count mismatch: values={}, columns={}",
            values.len(),
            columns.len()
        ));
    }
    let mut payload = Vec::new();
    payload.extend_from_slice(&(values.len() as i16).to_be_bytes());
    for (idx, value) in values.iter().enumerate() {
        let type_oid = columns[idx].type_oid;
        let format = *formats.get(idx).unwrap_or(&0);
        let data = match (type_oid, format) {
            (16, 1) => match value.to_ascii_lowercase().as_str() {
                "true" | "t" | "1" => vec![1_u8],
                "false" | "f" | "0" => vec![0_u8],
                other => {
                    return Err(format!(
                        "failed to encode bool binary result: invalid value {}",
                        other
                    ))
                }
            },
            (23, 1) => value
                .parse::<i32>()
                .map(|v| v.to_be_bytes().to_vec())
                .map_err(|e| format!("failed to encode int4 binary result: {e}"))?,
            (20, 1) => value
                .parse::<i64>()
                .map(|v| v.to_be_bytes().to_vec())
                .map_err(|e| format!("failed to encode int8 binary result: {e}"))?,
            (700, 1) => value
                .parse::<f32>()
                .map(|v| v.to_bits().to_be_bytes().to_vec())
                .map_err(|e| format!("failed to encode float4 binary result: {e}"))?,
            (701, 1) => value
                .parse::<f64>()
                .map(|v| v.to_bits().to_be_bytes().to_vec())
                .map_err(|e| format!("failed to encode float8 binary result: {e}"))?,
            (_, 1) => value.as_bytes().to_vec(),
            (_, 0) => value.as_bytes().to_vec(),
            (_, other) => {
                return Err(format!(
                    "unsupported result format code {} (only text/binary are supported)",
                    other
                ))
            }
        };
        payload.extend_from_slice(&(data.len() as i32).to_be_bytes());
        payload.extend_from_slice(&data);
    }
    let len = (payload.len() + 4) as i32;
    let mut out = Vec::new();
    out.push(b'D');
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// PostgreSQL `FLOAT4[]` binary layout:
/// ndim(i32), has_null(i32), elem_oid(i32), dims[(len(i32), lower(i32))...], elements[(len(i32), bytes)...]
/// PostgreSQL `FLOAT4[]` 二进制布局：
/// ndim(i32), has_null(i32), elem_oid(i32), dims[(len(i32), lower(i32))...], elements[(len(i32), bytes)...]
fn decode_pg_float4_array_binary(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() < 12 {
        return Err("float4[] binary payload too short".to_string());
    }
    let ndim = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let has_null = i32::from_be_bytes(bytes[4..8].try_into().unwrap());
    let elem_oid = i32::from_be_bytes(bytes[8..12].try_into().unwrap());
    if ndim < 0 {
        return Err(format!("float4[] binary has invalid ndim {}", ndim));
    }
    if ndim == 0 {
        return Ok(Vec::new());
    }
    if ndim != 1 {
        return Err(format!(
            "float4[] binary only supports one dimension for now, got {}",
            ndim
        ));
    }
    if has_null != 0 {
        return Err("float4[] binary with NULL elements is not supported".to_string());
    }
    if elem_oid != 700 {
        return Err(format!(
            "float4[] binary element OID mismatch: expected 700, got {}",
            elem_oid
        ));
    }
    let dims_bytes = 12 + (ndim as usize) * 8;
    if bytes.len() < dims_bytes {
        return Err("float4[] binary payload truncated in dimension header".to_string());
    }
    let len = i32::from_be_bytes(bytes[12..16].try_into().unwrap());
    if len < 0 {
        return Err(format!(
            "float4[] binary has invalid dimension length {}",
            len
        ));
    }
    let expected = len as usize;
    let mut offset = dims_bytes;
    let mut out = Vec::with_capacity(expected);
    for _ in 0..expected {
        if bytes.len() < offset + 4 {
            return Err("float4[] binary payload truncated before element length".to_string());
        }
        let item_len = i32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if item_len != 4 {
            return Err(format!(
                "float4[] binary element length mismatch: expected 4, got {}",
                item_len
            ));
        }
        if bytes.len() < offset + 4 {
            return Err("float4[] binary payload truncated before element bytes".to_string());
        }
        let value_bits = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
        out.push(f32::from_bits(value_bits));
        offset += 4;
    }
    if offset != bytes.len() {
        return Err("float4[] binary payload has trailing bytes".to_string());
    }
    Ok(out)
}

/// HouyiDB `BYTEA` 向量负载支持两种格式：
/// 1. legacy：连续 little-endian `f32`
/// 2. v1：`b"AVB1" + dim(u32 le) + payload[f32 le...]`
///
/// HouyiDB `BYTEA` vector payload supports:
/// 1. legacy contiguous little-endian `f32`
/// 2. v1 `b"AVB1" + dim(u32 le) + payload[f32 le...]`
pub fn decode_houyi_vector_bytea_binary(
    bytes: &[u8],
    expected_dim: Option<usize>,
) -> Result<Vec<f32>, String> {
    const HEADER: &[u8; 4] = b"AVB1";
    let (declared_dim, payload) = if bytes.len() >= 8 && &bytes[0..4] == HEADER {
        let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        (Some(dim), &bytes[8..])
    } else {
        (None, bytes)
    };
    if payload.len() % 4 != 0 {
        return Err(format!(
            "bytea vector payload length must be a multiple of 4, got {}",
            payload.len()
        ));
    }
    let actual_dim = payload.len() / 4;
    if let Some(dim) = declared_dim {
        if dim != actual_dim {
            return Err(format!(
                "bytea vector header dim {} does not match payload dim {}",
                dim, actual_dim
            ));
        }
    }
    if let Some(expected) = expected_dim {
        if actual_dim != expected {
            return Err(format!(
                "bytea vector dim {} != server vector_dim {}",
                actual_dim, expected
            ));
        }
    }
    let mut out = Vec::with_capacity(actual_dim);
    for chunk in payload.chunks_exact(4) {
        out.push(f32::from_bits(u32::from_le_bytes(
            chunk.try_into().unwrap(),
        )));
    }
    Ok(out)
}

/// HouyiDB `BYTEA` 批量 upsert 负载 v1：
/// `b"AUB1" + dim(u32 le) + count(u32 le) + items[count]`
/// `item := id(u64 le) + vec[f32 le; dim]`
///
/// HouyiDB `BYTEA` batch upsert payload v1:
/// `b"AUB1" + dim(u32 le) + count(u32 le) + items[count]`
/// `item := id(u64 le) + vec[f32 le; dim]`
pub fn decode_houyi_upsert_batch_bytea_binary(
    bytes: &[u8],
    expected_dim: Option<usize>,
) -> Result<HouyiUpsertBatch, String> {
    const HEADER: &[u8; 4] = b"AUB1";
    if bytes.len() < 12 || &bytes[0..4] != HEADER {
        return Err("bytea upsert_batch payload must start with AUB1 header".to_string());
    }
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if dim == 0 {
        return Err("bytea upsert_batch dim must be > 0".to_string());
    }
    if let Some(expected) = expected_dim {
        if dim != expected {
            return Err(format!(
                "bytea upsert_batch dim {} != server vector_dim {}",
                dim, expected
            ));
        }
    }
    let item_len = 8_usize
        .checked_add(
            dim.checked_mul(4)
                .ok_or("bytea upsert_batch dim overflow")?,
        )
        .ok_or("bytea upsert_batch item length overflow")?;
    let expected_len = 12_usize
        .checked_add(
            item_len
                .checked_mul(count)
                .ok_or("bytea upsert_batch count overflow")?,
        )
        .ok_or("bytea upsert_batch payload length overflow")?;
    if bytes.len() != expected_len {
        return Err(format!(
            "bytea upsert_batch payload length mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        ));
    }
    let mut offset = 12;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        let id = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            let bits = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;
            v.push(f32::from_bits(bits));
        }
        items.push((id, v));
    }
    debug_assert_eq!(offset, bytes.len());
    Ok(HouyiUpsertBatch { dim, items })
}

fn pgwire_vector_placeholder_param_index(sql: &str) -> Option<usize> {
    let marker = "/* houyi:vector=";
    let start = sql.find(marker)?;
    let after = &sql[start + marker.len()..];
    let end = after.find("*/")?;
    let expr = after[..end].trim();
    let idx = expr.strip_prefix('$')?.parse::<usize>().ok()?;
    Some(idx)
}

fn pgwire_upsert_batch_placeholder_param_index(sql: &str) -> Option<usize> {
    let marker = "/* houyi:upsert_batch=";
    // 从 `houyi:upsert_batch=$N` 形式的上游 SQL 标记中取参数序号。
    // EN: Take the parameter index from the upstream SQL marker `houyi:upsert_batch=$N`.
    let start = sql.find(marker)?;
    let after = &sql[start + marker.len()..];
    let end = after.find("*/")?;
    let expr = after[..end].trim();
    let idx = expr.strip_prefix('$')?.parse::<usize>().ok()?;
    Some(idx)
}

/// 将 Bind 参数规范化为 SQL 文本字面量，先保证协议正确性，再继续优化二进制快路径。
/// Normalize bind parameters into SQL literals first; optimize binary fast-path later.
pub fn pgwire_bind_param_to_sql_literal(
    oid: u32,
    format: i16,
    param: &BindParamValue,
) -> Result<String, String> {
    match (format, param) {
        (0, BindParamValue::Text(raw)) => match oid {
            0 | 25 | 1043 => Ok(format!("'{}'", pgwire_escape_sql_text(raw))),
            16 => {
                let lowered = raw.trim().to_ascii_lowercase();
                match lowered.as_str() {
                    "t" | "true" | "1" => Ok("TRUE".to_string()),
                    "f" | "false" | "0" => Ok("FALSE".to_string()),
                    _ => Err(format!("invalid bool text parameter: {}", raw)),
                }
            }
            20 => raw
                .trim()
                .parse::<i64>()
                .map(|v| v.to_string())
                .map_err(|e| format!("invalid int8 text parameter: {e}")),
            23 => raw
                .trim()
                .parse::<i32>()
                .map(|v| v.to_string())
                .map_err(|e| format!("invalid int4 text parameter: {e}")),
            700 => raw
                .trim()
                .parse::<f32>()
                .map(|v| v.to_string())
                .map_err(|e| format!("invalid float4 text parameter: {e}")),
            701 => raw
                .trim()
                .parse::<f64>()
                .map(|v| v.to_string())
                .map_err(|e| format!("invalid float8 text parameter: {e}")),
            1021 => {
                let trimmed = raw.trim();
                let inner = trimmed
                    .strip_prefix('{')
                    .and_then(|s| s.strip_suffix('}'))
                    .unwrap_or(trimmed);
                let mut values = Vec::new();
                for token in inner.split(',') {
                    let token = token.trim();
                    if token.is_empty() {
                        continue;
                    }
                    let value = token
                        .parse::<f32>()
                        .map_err(|e| format!("invalid float4[] text parameter element: {e}"))?;
                    values.push(value);
                }
                Ok(pgwire_float_list_to_directive_text(&values))
            }
            other => Err(format!("unsupported parameter type OID {}", other)),
        },
        (1, BindParamValue::Binary(bytes)) => match oid {
            0 | 25 | 1043 => String::from_utf8(bytes.clone())
                .map(|s| format!("'{}'", pgwire_escape_sql_text(&s)))
                .map_err(|e| format!("invalid text binary parameter utf8: {e}")),
            16 => match bytes.as_slice() {
                [0] => Ok("FALSE".to_string()),
                [1] => Ok("TRUE".to_string()),
                _ => Err(format!(
                    "invalid bool binary parameter length/value: {:?}",
                    bytes
                )),
            },
            20 => {
                if bytes.len() != 8 {
                    return Err(format!(
                        "invalid int8 binary parameter length: expected 8, got {}",
                        bytes.len()
                    ));
                }
                Ok(i64::from_be_bytes(bytes[..8].try_into().unwrap()).to_string())
            }
            23 => {
                if bytes.len() != 4 {
                    return Err(format!(
                        "invalid int4 binary parameter length: expected 4, got {}",
                        bytes.len()
                    ));
                }
                Ok(i32::from_be_bytes(bytes[..4].try_into().unwrap()).to_string())
            }
            700 => {
                if bytes.len() != 4 {
                    return Err(format!(
                        "invalid float4 binary parameter length: expected 4, got {}",
                        bytes.len()
                    ));
                }
                Ok(f32::from_bits(u32::from_be_bytes(bytes[..4].try_into().unwrap())).to_string())
            }
            701 => {
                if bytes.len() != 8 {
                    return Err(format!(
                        "invalid float8 binary parameter length: expected 8, got {}",
                        bytes.len()
                    ));
                }
                Ok(f64::from_bits(u64::from_be_bytes(bytes[..8].try_into().unwrap())).to_string())
            }
            1021 => decode_pg_float4_array_binary(bytes)
                .map(|values| pgwire_float_list_to_directive_text(&values)),
            other => Err(format!("unsupported binary parameter type OID {}", other)),
        },
        (0, BindParamValue::Binary(_)) => {
            Err("bind parameter payload/format mismatch: expected text payload".to_string())
        }
        (1, BindParamValue::Text(_)) => {
            Err("bind parameter payload/format mismatch: expected binary payload".to_string())
        }
        (other, _) => Err(format!(
            "unsupported parameter format code {} (only text/binary are supported)",
            other
        )),
    }
}

/// 按 PostgreSQL `$1/$2...` 占位符做最小文本替换（仅用于未命名 prepared 的 text 参数）。
/// Minimal `$1/$2...` placeholder substitution for unnamed prepared text params.
fn apply_bind_params(sql: &str, params: &[String]) -> Result<String, String> {
    let mut out = sql.to_string();
    // 逆序替换，避免 `$1` 误改 `$10`。
    // EN: Reverse order prevents `$1` from accidentally rewriting `$10`.
    for (idx, p) in params.iter().enumerate().rev() {
        let token = format!("${}", idx + 1);
        if !out.contains(&token) {
            return Err(format!("missing placeholder {token} in prepared SQL"));
        }
        out = out.replace(&token, p);
    }
    Ok(out)
}

fn is_copy_sql(sql: &str) -> bool {
    sql.trim_start().to_ascii_uppercase().starts_with("COPY ")
}

/// 事务/会话控制语句的 CommandComplete 标签；None = 普通查询。
/// 为什么必须支持（libpq 系客户端的硬约束）：psycopg2/JDBC 会在执行前发
/// `BEGIN`，连接池还会发 `SET`/`DEALLOCATE`；拒绝它们会让标准客户端**无法
/// 建立可用连接**。v1 没有多语句事务语义，按命令成功应答（不返回行）比
/// 0A000 更兼容；真正的原子性由 WAL/复制保证，不依赖 SQL 事务。
/// EN: map transaction/session control to a CommandComplete tag. psycopg2/JDBC
/// issue BEGIN before queries; rejecting it breaks standard clients. v1 has no
/// multi-statement transactions — durability comes from WAL/replication.
fn pgwire_noop_command(sql: &str) -> Option<&'static str> {
    let first = sql
        .trim_start()
        .split(|c: char| c.is_whitespace() || c == ';')
        .next()
        .unwrap_or("")
        .to_ascii_uppercase();
    match first.as_str() {
        "BEGIN" | "START" => Some("BEGIN"),
        "COMMIT" | "END" => Some("COMMIT"),
        "ROLLBACK" | "ABORT" => Some("ROLLBACK"),
        "SET" => Some("SET"),
        "RESET" => Some("RESET"),
        "DEALLOCATE" => Some("DEALLOCATE"),
        "DISCARD" => Some("DISCARD"),
        _ => None,
    }
}

/// 计算 SQL 查询结果行，供 Simple Query 与 Extended Execute 复用。
/// Build query result rows for Simple Query and extended Execute.
async fn pgwire_collect_rows(
    pg: Option<&PgWireSharedState>,
    sql: &str,
) -> Result<(Vec<PgWireColumnMeta>, Vec<Vec<String>>), (String, String)> {
    match pg {
        Some(state) => match parse_houyi_pgwire_sql(sql) {
            Ok(HouyiPgWireSql::VectorSearch {
                sql_for_planner,
                query_vector,
                k,
                ef_search_override,
            }) => {
                // 放大面封边先于任何副作用（尤其先于 get_or_create 隐式建 Collection）：
                // pgwire 的 `LIMIT n` 与 `/* houyi:ef_search=N */` 与 HTTP 同源，都要经
                // k.max(ef) → 16×ann_k 访问器预分配；不封上界时一条 SQL 即可 abort 进程。
                // 与 HTTP 共用同一判定函数；SQLSTATE 22023 = invalid_parameter_value。
                // EN: Enforce the shared k/ef cap before any side effect (notably before
                // get_or_create implicitly creates a collection). pgwire LIMIT/ef feed the
                // same k.max(ef) → 16×ann_k preallocation as HTTP; an unbounded SQL could
                // abort the process. SQLSTATE 22023 = invalid_parameter_value.
                if let Some(msg) =
                    crate::http_handler::search_limit_violation(k, ef_search_override)
                {
                    return Err(("22023".to_string(), msg));
                }
                // 路由：有 Collection 指令时走 Collection，否则走默认 index（向后兼容）
                // EN: Routing: use a Collection when the directive is present, else the
                // default index (backward compatible).
                let coll_name = houyi_query::extract_collection_directive(&sql_for_planner);
                let _col_holder; // keep Arc alive
                let backend_used: &IndexBackend;
                let dims_to_check: usize;
                if let Some(ref name) = coll_name {
                    // 只读副本（P4.4.1 修正）：follower 上 SELECT 不得隐式建
                    // collection——get_or_create 会在复制流之外 mutate 本地状态
                    // （幻影 collection 进 /v1/collections、与 leader 分叉、重启
                    // 静默消失）。缺名按 PostgreSQL 语义回 42P01 undefined_table，
                    // 与 HTTP 检索的 get()+404 契约对齐。
                    // EN: On followers a SELECT must not implicitly create the
                    // collection (that mutates local state outside the replication
                    // stream); resolve with get() and return 42P01 when missing.
                    let col = if state.role.is_follower() {
                        state.collections.get(name).ok_or_else(|| {
                            (
                                "42P01".to_string(),
                                format!("undefined collection \"{name}\""),
                            )
                        })?
                    } else {
                        state
                            .collections
                            .get_or_create(name, Some(query_vector.len()))
                            .map_err(|e| ("XX000".to_string(), e))?
                    };
                    dims_to_check = col.dims;
                    _col_holder = col;
                    backend_used = &_col_holder.backend;
                } else {
                    // follower 上的默认（benchmark）主索引查询：**有意放行**的只读
                    // 路径——主索引不在复制范围内，这里通常返回空结果。与 collection
                    // 分支的响亮 42P01 不同，本路径是「静默空成功」：failover 排查时
                    // 别把它误认为数据完整（OCR 批次十八；写路径已在 Execute 被
                    // 25006 拦截，本路径无 mutation 风险）。
                    // EN: Follower queries against the default benchmark index are an
                    // intentionally allowed read path — the main index is not
                    // replicated, so this typically returns empty rows (silent
                    // empty-success, by design; see startup warning).
                    backend_used = &state.index;
                    dims_to_check = state.vector_dim;
                }
                if query_vector.len() != dims_to_check {
                    return Err((
                        "22000".to_string(),
                        format!(
                            "query vector dim {} != expected dim {}",
                            query_vector.len(),
                            dims_to_check
                        ),
                    ));
                }
                let ids = execute_query(
                    backend_used,
                    &sql_for_planner,
                    &query_vector,
                    k,
                    ef_search_override,
                )
                .await
                .map_err(|e| ("XX000".to_string(), e))?;
                Ok((
                    vec![PgWireColumnMeta {
                        name: "id".to_string(),
                        type_oid: 20,
                    }],
                    ids.into_iter().map(|id| vec![id.to_string()]).collect(),
                ))
            }
            Ok(HouyiPgWireSql::Plain(s)) => {
                if is_copy_sql(&s) {
                    Err(("0A000".to_string(), "COPY SQL is not supported".to_string()))
                } else {
                    let exprs = pgwire_split_plain_select_exprs(&s).ok_or_else(|| {
                        (
                            "0A000".to_string(),
                            "only simple SELECT literal[, literal...] is supported".to_string(),
                        )
                    })?;
                    let mut columns = Vec::with_capacity(exprs.len());
                    let mut row = Vec::with_capacity(exprs.len());
                    for (idx, expr) in exprs.iter().enumerate() {
                        let (oid, value) = pgwire_plain_cast_value(expr).unwrap_or_else(|| {
                            (
                                pgwire_plain_scalar_oid(expr).unwrap_or(25),
                                expr.trim_matches('\'').to_string(),
                            )
                        });
                        columns.push(PgWireColumnMeta {
                            name: if exprs.len() == 1 {
                                "value".to_string()
                            } else {
                                format!("column{}", idx + 1)
                            },
                            type_oid: oid,
                        });
                        row.push(value);
                    }
                    Ok((columns, vec![row]))
                }
            }
            Err(e) => Err(("42601".to_string(), e)),
        },
        None => match parse_houyi_pgwire_sql(sql) {
            Ok(HouyiPgWireSql::Plain(s)) => {
                if is_copy_sql(&s) {
                    Err(("0A000".to_string(), "COPY SQL is not supported".to_string()))
                } else {
                    let exprs = pgwire_split_plain_select_exprs(&s).ok_or_else(|| {
                        (
                            "0A000".to_string(),
                            "only simple SELECT literal[, literal...] is supported".to_string(),
                        )
                    })?;
                    let mut columns = Vec::with_capacity(exprs.len());
                    let mut row = Vec::with_capacity(exprs.len());
                    for (idx, expr) in exprs.iter().enumerate() {
                        let (oid, value) = pgwire_plain_cast_value(expr).unwrap_or_else(|| {
                            (
                                pgwire_plain_scalar_oid(expr).unwrap_or(25),
                                expr.trim_matches('\'').to_string(),
                            )
                        });
                        columns.push(PgWireColumnMeta {
                            name: if exprs.len() == 1 {
                                "value".to_string()
                            } else {
                                format!("column{}", idx + 1)
                            },
                            type_oid: oid,
                        });
                        row.push(value);
                    }
                    Ok((columns, vec![row]))
                }
            }
            _ => Err((
                "0A000".to_string(),
                "vector search requires pg_wire shared state".to_string(),
            )),
        },
    }
}

/// SQLSTATE → 查询日志状态码：客户端错误类记 400，其余记 500。
/// EN: SQLSTATE → query-log status: client-fault classes are 400, else 500.
fn pgwire_log_status(sqlstate: &str) -> u16 {
    match sqlstate.get(0..2) {
        Some("0A" | "22" | "25" | "26" | "34" | "42") => 400,
        _ => 500,
    }
}

/// 记录一条 pgwire 查询（含主体、k、过滤标志）。关闭时零开销。
/// EN: record one pgwire query (principal, k, filter flag); free when disabled.
#[allow(clippy::too_many_arguments)]
fn record_pgwire_query(
    pg: Option<&PgWireSharedState>,
    principal: Option<&crate::rbac::Principal>,
    sql: &str,
    status: u16,
    latency_ms: f64,
    result_count: usize,
) {
    let Some(log) = pg.as_ref().and_then(|pg| pg.query_log.as_ref()) else {
        return;
    };
    log.record(crate::query_log::QueryLogEntry {
        ts_ms: crate::query_log::now_ms(),
        request_id: 0,
        principal: principal
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "-".to_string()),
        route: "pgwire",
        collection: houyi_query::extract_collection_directive(sql),
        k: houyi_query::extract_limit_k(sql).unwrap_or(10),
        status,
        latency_ms,
        result_count,
        // 与规划器一致：任何 WHERE 都按过滤路径处理（含不可下推的谓词）。
        // EN: mirrors the planner: any WHERE is treated as filtered, including
        // non-pushdownable predicates.
        filtered: sql.to_ascii_uppercase().contains("WHERE"),
    });
}

/// 执行一条 SQL（Simple Query 与 Extended Execute 共用）。
/// Execute one SQL statement (shared by Simple Query and extended Execute).
async fn pgwire_dispatch_sql<S>(
    stream: &mut S,
    pg: Option<&PgWireSharedState>,
    sql: &str,
    principal: Option<&crate::rbac::Principal>,
) -> std::io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // 事务/会话控制：只回 CommandComplete（不发行描述/数据行）。
    // EN: Transaction/session control: reply with CommandComplete only (no row
    // description/data rows).
    if let Some(tag) = pgwire_noop_command(sql) {
        return stream.write_all(&encode_command_complete(tag)).await;
    }
    // 查询日志（EE M1）：只记元数据，主体/K/过滤标志取自会话与 SQL。
    // EN: query log (EE M1), metadata only; principal, k and the filter flag
    // come from the session and the SQL.
    let t0 = std::time::Instant::now();
    let outcome = pgwire_collect_rows(pg, sql).await;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match &outcome {
        Ok((_, rows)) => record_pgwire_query(pg, principal, sql, 200, latency_ms, rows.len()),
        Err((code, _)) => {
            record_pgwire_query(pg, principal, sql, pgwire_log_status(code), latency_ms, 0)
        }
    }
    match outcome {
        Ok((columns, rows)) => {
            let formats = vec![0; columns.len()];
            let row_description = encode_row_description_columns(&columns, &formats)
                .map_err(std::io::Error::other)?;
            stream.write_all(&row_description).await?;
            for row in &rows {
                let encoded = encode_data_row_columns(row, &columns, &formats)
                    .map_err(std::io::Error::other)?;
                stream.write_all(&encoded).await?;
            }
            stream
                .write_all(&encode_command_complete(&format!("SELECT {}", rows.len())))
                .await?;
        }
        Err((code, msg)) => {
            stream
                .write_all(&encode_error_response("ERROR", &code, &msg))
                .await?;
        }
    }
    Ok(())
}

pub async fn execute_query(
    index: &IndexBackend,
    sql: &str,
    query_vector: &[f32],
    k: usize,
    ef_search_override: Option<usize>,
) -> Result<Vec<VectorId>, String> {
    let router = MultiTaskBusRouter::new(64);
    let plan = QueryPlanner::plan(PgWireQuery {
        sql: sql.to_string(),
    });
    let task = match plan {
        ExecutionPlan::RawSql(sql_text) => QueryTask {
            task_id: 1,
            sql: sql_text,
            query_vector: query_vector.to_vec(),
            k,
            filter_mask: None,
            ef_search_override,
        },
        ExecutionPlan::HybridSearch {
            sql: sql_text,
            pushed_down_bitset,
        } => QueryTask {
            task_id: 1,
            sql: sql_text,
            query_vector: query_vector.to_vec(),
            k,
            filter_mask: pushed_down_bitset,
            ef_search_override,
        },
    };
    router
        .publish_query(task)
        .map_err(|_| "query task bus is full".to_string())?;
    let result = consume_one_query_task(index, &router).await;
    let m = router.metrics_snapshot();
    tracing::debug!(
        q_pub = m.published_query,
        q_consume = m.consumed_query,
        b_pub = m.published_build,
        p_pub = m.published_persist,
        "router metrics"
    );
    result
}

/// 原始 PAX 页上的过滤检索（启动自检探针用）：直接按谓词求 bitset，不经过
/// [`crate::collection::CollectionScalars::matching_ids`] 的 last-wins 过滤。
/// 仅适用于单行/无重写的页；collection 的真实过滤必须走 `matching_ids`，否则
/// 同 id 重写后旧值会重新命中（公共 issue #6）。
/// EN: Filtered search over a raw PAX page (startup self-check probe only): builds
/// the bitset straight from the predicate, bypassing the last-wins filter in
/// [`crate::collection::CollectionScalars::matching_ids`]. Only safe for
/// single-row/non-rewritten pages; real collection filtering must go through
/// `matching_ids`, or stale values match after same-id re-upserts (public issue #6).
pub async fn execute_hybrid_with_scalar_bitset(
    index: &IndexBackend,
    page: &HouyiPage,
    column: &str,
    predicate: ScalarPredicate,
    query_vector: &[f32],
    k: usize,
) -> Vec<VectorId> {
    let bitset = page.filter_to_bitset(column, &predicate);
    index.search(query_vector, k, Some(&bitset)).await
}

async fn consume_one_query_task(
    index: &IndexBackend,
    router: &MultiTaskBusRouter,
) -> Result<Vec<VectorId>, String> {
    let Some(q) = router.try_consume_query() else {
        return Err("task bus is empty".to_string());
    };
    Ok(index
        .search_with_ef(
            &q.query_vector,
            q.k,
            q.filter_mask.as_deref(),
            q.ef_search_override,
        )
        .await)
}

/// pgwire 会话流（P4.2.2）：明文 TCP 或 TLS 升级后的流。统一实现
/// AsyncRead/AsyncWrite，握手与查询循环对两种传输复用同一份代码。
/// EN: Session stream — plain TCP or TLS-upgraded; one AsyncRead/AsyncWrite
/// impl lets handshake and query loops stay transport-agnostic.
pub enum PgwireStream {
    Plain(TcpStream),
    Tls(Box<tokio_rustls::server::TlsStream<TcpStream>>),
}

impl AsyncRead for PgwireStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_read(cx, buf),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for PgwireStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_write(cx, buf),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_flush(cx),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_shutdown(cx),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_shutdown(cx),
        }
    }
}

/// peek 直到凑满 `buf`（tokio `peek` 允许短读）。对端关闭返回 UnexpectedEof。
/// EN: Peek until `buf` is full (tokio `peek` may return short reads). A peer
/// close returns UnexpectedEof.
async fn peek_exact(stream: &TcpStream, buf: &mut [u8]) -> std::io::Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = stream.peek(&mut buf[filled..]).await?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "pgwire: connection closed during SSL negotiation",
            ));
        }
        filled += n;
    }
    Ok(())
}

/// SSLRequest 协商（P4.2.2）：peek（不消费）启动包按码分流——
/// - SSLRequest → 消费并回 'S'，完成 TLS 握手，返回 Tls 流；
/// - GSSENCRequest → 消费并回 'N'（不支持 GSSAPI），继续看下一包；
/// - 其他（直接 StartupMessage）→ 原样返回明文流（sslmode=prefer 语义；
///   强制 TLS-only 需要 pg_hba 式配置面，超出 beta 范围）。
///
/// EN: peek-based SSLRequest negotiation: 'S' + TLS handshake on SSLRequest,
/// 'N' on GSSENC, pass-through for direct plaintext startup (prefer semantics).
async fn negotiate_pgwire_tls(
    mut stream: TcpStream,
    acceptor: &TlsAcceptor,
) -> std::io::Result<PgwireStream> {
    loop {
        let mut head = [0_u8; 8];
        peek_exact(&stream, &mut head).await?;
        let len = i32::from_be_bytes(head[0..4].try_into().unwrap()) as usize;
        let code = i32::from_be_bytes(head[4..8].try_into().unwrap());
        match code {
            SSL_REQUEST_CODE if len == 8 => {
                // 消费 SSLRequest 后回 'S' 并升级。
                // EN: Consume the SSLRequest, reply 'S', and upgrade.
                let mut sink = [0_u8; 8];
                stream.read_exact(&mut sink).await?;
                stream.write_all(b"S").await?;
                let tls = acceptor.accept(stream).await.map_err(|e| {
                    std::io::Error::new(e.kind(), format!("pgwire tls handshake: {e}"))
                })?;
                return Ok(PgwireStream::Tls(Box::new(tls)));
            }
            GSSENC_REQUEST_CODE if len == 8 => {
                let mut sink = [0_u8; 8];
                stream.read_exact(&mut sink).await?;
                stream.write_all(b"N").await?;
                continue;
            }
            _ => return Ok(PgwireStream::Plain(stream)),
        }
    }
}

/// pgwire 连接入口：TLS 协商（可选升级）→ 认证握手，返回就绪会话流。
/// `tls_acceptor` 为 None 时行为与 P4.2.1 一致（SSLRequest 固定回 'N'）。
/// EN: Connection entry — optional TLS upgrade, then the auth handshake;
/// returns the ready session stream.
pub async fn pgwire_accept_session(
    stream: TcpStream,
    metrics: &PgWireMetrics,
    authz: Option<&crate::rbac::AuthzPolicy>,
    tls_acceptor: Option<&TlsAcceptor>,
) -> std::io::Result<(PgwireStream, Option<crate::rbac::Principal>)> {
    let mut session = match tls_acceptor {
        Some(acceptor) => negotiate_pgwire_tls(stream, acceptor).await?,
        None => PgwireStream::Plain(stream),
    };
    let principal = handle_pgwire_handshake(&mut session, metrics, authz).await?;
    Ok((session, principal))
}

/// 连接水位线裁决（P4.2.3）：握手完成后、进入查询循环前调用。
/// 超限时发送协议内 FATAL 53300（与 Postgres `too_many_connections` 同款），
/// 返回 None；否则返回 RAII 许可，随会话任务结束自动归还。
/// 裁决放在握手后而非 accept 时：客户端已说完 startup，错误以标准
/// ErrorResponse 送达（accept 时直接关闭只会让客户端看到不明 EOF）。
/// EN: Post-handshake admission control — on overflow send FATAL 53300 (mirrors
/// Postgres) and return None; otherwise return the RAII permit.
pub async fn pgwire_connection_admit<S>(
    session: &mut S,
    metrics: &PgWireMetrics,
) -> Option<InflightPermit>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    match metrics.connection_watermark.try_acquire() {
        Some(permit) => Some(permit),
        None => {
            let _ = session
                .write_all(&encode_error_response(
                    "FATAL",
                    "53300",
                    "sorry, too many clients",
                ))
                .await;
            None
        }
    }
}

/// 认证失败的固定应答延迟（OCR 359d85d[8]）：没有限速时，网络可达客户端可以
/// 按连接速度在线爆破令牌。250ms 固定延迟把单连接猜测速率压到 4 次/秒——
/// 不防分布式爆破（那需要 per-peer 计数+指数退避，超出 beta 范围），但把
/// "零成本"变成"有成本"。失败路径持连接更久是水位的既定取舍（连接水位线
/// P4.2.3 兜底总量）。
/// EN: Fixed delay before every auth-failure reply — bounds per-connection
/// online guessing to 4 attempts/s. Not a distributed-brute-force defense
/// (that needs per-peer backoff, out of beta scope); the connection watermark
/// bounds how many delayed failures can be held open.
const AUTH_FAILURE_DELAY: std::time::Duration = std::time::Duration::from_millis(250);

async fn auth_failure_delay() {
    tokio::time::sleep(AUTH_FAILURE_DELAY).await;
}

/// pgwire 握手：SSLRequest 协商 → （可选）明文密码认证 → AuthenticationOk。
/// `auth_token` 来自 HOUYI_AUTH_TOKEN；None = 不启用认证（本机开发默认）。
/// 泛型流：明文 TCP 与 TLS 升级流共用（P4.2.2）。
/// EN: Startup negotiation, optional cleartext-password challenge (beta
/// simplification — deploy with TLS or on localhost), then AuthenticationOk.
/// Generic over the stream so plain TCP and TLS sessions share the code.
pub async fn handle_pgwire_handshake<S>(
    stream: &mut S,
    metrics: &PgWireMetrics,
    authz: Option<&crate::rbac::AuthzPolicy>,
) -> std::io::Result<Option<crate::rbac::Principal>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        let mut len_buf = [0_u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let len = i32::from_be_bytes(len_buf) as usize;
        if len < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid startup packet length",
            ));
        }
        let mut payload = vec![0_u8; len - 4];
        stream.read_exact(&mut payload).await?;
        let mut packet = Vec::with_capacity(len);
        packet.extend_from_slice(&len_buf);
        packet.extend_from_slice(&payload);
        let startup = decode_startup_packet(&packet).map_err(std::io::Error::other)?;

        if startup.protocol_version == SSL_REQUEST_CODE {
            stream.write_all(b"N").await?;
            continue;
        }

        // GSSENC（P4.2.2）：明文路径同样显式回 'N'——psql 可能先发 GSSENC
        // 再发 SSLRequest，当普通启动包处理会协议错乱。
        // EN: Reply 'N' to GSSENC on the plaintext path too — psql may send
        // it before SSLRequest; treating it as a startup packet corrupts the
        // protocol state.
        if startup.protocol_version == GSSENC_REQUEST_CODE {
            stream.write_all(b"N").await?;
            continue;
        }

        // 认证主体（P4.7.1）：策略非空即质询；密码即 token，解析失败统一
        // 28P01（不区分「未知 token」与「错误 token」，避免身份枚举）。
        // EN: challenge when the policy is non-empty; password = token; a
        // single 28P01 for unknown/wrong avoids principal enumeration.
        let mut principal = None;
        if let Some(policy) = authz.filter(|p| !p.is_empty()) {
            stream
                .write_all(&encode_authentication_cleartext_password())
                .await?;
            // 密码消息（'p'）：[类型:1][长度:4][密码 cstring]。
            // EN: PasswordMessage('p'): [type:1][len:4][password cstring].
            let mut header = [0_u8; 5];
            stream.read_exact(&mut header).await?;
            if header[0] != b'p' {
                metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
                auth_failure_delay().await;
                let _ = stream
                    .write_all(&encode_error_response(
                        "FATAL",
                        "28000",
                        "expected password message",
                    ))
                    .await;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "pgwire auth: expected password message",
                ));
            }
            let len = i32::from_be_bytes(header[1..5].try_into().unwrap()) as usize;
            // 密码上限 4 KiB：超出即畸形/攻击载荷，拒于边界。
            // EN: Password cap 4 KiB: anything larger is malformed/attack payload,
            // rejected at the boundary.
            if !(5..=4104).contains(&len) {
                metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
                auth_failure_delay().await;
                let _ = stream
                    .write_all(&encode_error_response(
                        "FATAL",
                        "28000",
                        "invalid password message length",
                    ))
                    .await;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "pgwire auth: invalid password message length",
                ));
            }
            let mut payload = vec![0_u8; len - 4];
            stream.read_exact(&mut payload).await?;
            let password = payload.strip_suffix(&[0]).unwrap_or(&payload);
            let token = std::str::from_utf8(password).unwrap_or("");
            match policy.resolve(token) {
                Some(p) => principal = Some(p.clone()),
                None => {
                    // 28P01 = invalid_password；日志不记录密码明文。
                    // 独立计数（OCR 359d85d[12]）：凭据拒绝不再混进 decode_errors。
                    // EN: 28P01 = invalid_password; logs never record the password in clear.
                    // Counted separately (OCR 359d85d[12]): credential rejections no longer
                    // mix into decode_errors.
                    metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
                    auth_failure_delay().await;
                    let _ = stream
                        .write_all(&encode_error_response(
                            "FATAL",
                            "28P01",
                            "password authentication failed",
                        ))
                        .await;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "pgwire auth: password authentication failed",
                    ));
                }
            }
        }

        stream.write_all(&encode_authentication_ok()).await?;
        stream
            .write_all(&encode_parameter_status("server_version", "16.0-houyi"))
            .await?;
        stream
            .write_all(&encode_parameter_status("client_encoding", "UTF8"))
            .await?;
        stream.write_all(&encode_ready_for_query()).await?;
        metrics.handshake_count.fetch_add(1, Ordering::Relaxed);
        return Ok(principal);
    }
}

/// collection 写分支带出的 sync 等待上下文（P4.4.2）：具名字段替代位置元组，
/// 等待点不再靠记忆 (col, ha_level, lsn) 的次序。
/// EN: Sync-wait context yielded by the collection write branch — named fields
/// instead of a positional tuple.
struct CollSyncWait {
    col: std::sync::Arc<crate::collection::Collection>,
    ha_level: houyi_persistence::ha_level::HaLevel,
    lsn: u64,
}

/// 集合配额（P4.7.2）：pgwire 语句若会隐式新建 collection，先检查主体配额。
/// 只约束新建；已有 collection 的更新不受限。SQLSTATE 54000 = 程序限制。
///
/// 为什么**不**按读写动作分账（OCR 批次三十四 HIGH）：带 collection 指令的
/// SELECT 向量检索同样走 `get_or_create`——只读主体/达到上限的主体可以用
/// `SELECT` 无限新建 collection，绕过配额。判定只看「名字是否已存在」；
/// 读取既有 collection 永远不受限。
/// EN: do NOT gate on read/write — a SELECT vector search with a collection
/// directive also implicitly creates; existence alone decides. Reads against
/// existing collections are never blocked. 54000 = program_limit_exceeded.
fn check_collection_quota(
    state: Option<&PgWireSharedState>,
    principal: &crate::rbac::Principal,
    sql: &str,
) -> Result<(), (String, String)> {
    let Some(name) = houyi_query::extract_collection_directive(sql) else {
        return Ok(());
    };
    let Some(state) = state else {
        return Ok(());
    };
    if state.collections.get(&name).is_some() {
        return Ok(());
    }
    let existing = state.collections.list();
    if principal.collection_quota_exceeded(&existing, &name) {
        return Err((
            "54000".to_string(),
            format!(
                "collection quota exceeded for principal '{}' (max_collections={:?})",
                principal.name, principal.max_collections
            ),
        ));
    }
    Ok(())
}

/// 逐语句拒绝链（RBAC 42501 + 配额 54000）：Simple Query 与 Extended Execute
/// 共用同一入口，两个协议面不可能各自漂移。
/// EN: one denial chain shared by Simple Query and Extended Execute so the two
/// protocol surfaces cannot drift apart.
fn check_pgwire_denial(
    state: Option<&PgWireSharedState>,
    principal: &crate::rbac::Principal,
    sql: &str,
) -> Result<(), (String, String)> {
    crate::rbac::authorize_sql(principal, sql)?;
    check_collection_quota(state, principal, sql)
}

/// 拒绝计数分账：配额拒绝（54000）与权限拒绝（42501）分开记录，
/// 配额背压不得伪装成安全事件（OCR 批次三十四）。
/// EN: quota rejections (54000) get their own counter; routine backpressure
/// must not look like an authorization attack.
fn record_pgwire_denial(metrics: &PgWireMetrics, code: &str) {
    if code == "54000" {
        metrics.quota_denials.fetch_add(1, Ordering::Relaxed);
    } else {
        metrics.authz_denials.fetch_add(1, Ordering::Relaxed);
    }
}

pub async fn handle_pgwire_simple_query_loop<S>(
    stream: &mut S,
    pg: Option<Arc<PgWireSharedState>>,
    metrics: Arc<PgWireMetrics>,
    principal: Option<crate::rbac::Principal>,
) -> std::io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut conn = PgWireConnState::default();
    loop {
        let mut header = [0_u8; 5];
        if stream.read_exact(&mut header).await.is_err() {
            return Ok(());
        }
        let len = i32::from_be_bytes(header[1..5].try_into().unwrap()) as usize;
        if len < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid frontend message length",
            ));
        }
        let mut payload = vec![0_u8; len - 4];
        stream.read_exact(&mut payload).await?;
        metrics
            .frontend_payload_bytes
            .fetch_add(payload.len() as u64, Ordering::Relaxed);
        let mut full = Vec::with_capacity(len + 1);
        full.extend_from_slice(&header);
        full.extend_from_slice(&payload);
        metrics.frontend_messages.fetch_add(1, Ordering::Relaxed);
        let msg = match decode_frontend_message(&full).map_err(std::io::Error::other) {
            Ok(m) => m,
            Err(e) => {
                metrics.decode_errors.fetch_add(1, Ordering::Relaxed);
                let code = pgwire_decode_error_code(&e.to_string());
                stream
                    .write_all(&encode_error_response("ERROR", code, &e.to_string()))
                    .await?;
                continue;
            }
        };
        let pg_ref = pg.as_deref();
        match msg {
            FrontendMessage::Query(sql) => {
                metrics.simple_query_count.fetch_add(1, Ordering::Relaxed);
                metrics
                    .sql_text_bytes
                    .fetch_add(sql.len() as u64, Ordering::Relaxed);
                // 慢查询计时（P4.3.2）：阈值 0 = 禁用，超阈值打 warn（SQL 清洗防
                // 刷屏/防日志注入）。计时窗口刻意覆盖整个 dispatch（含响应编码与
                // 逐行 write_all）——客户端背压/网络慢拖长的端到端延迟同样是
                // 「这条查询慢」的运维事实（OCR 批次三确认保留）。
                // EN: The timed window deliberately spans the whole dispatch
                // (response encoding + row writes included) — client backpressure
                // is operationally part of "this query was slow".
                // RBAC（P4.7.1）：每条语句执行前判定角色与 collection 作用域；
                // 拒绝以标准 ErrorResponse（42501）返回，不进入任何副作用路径。
                // EN: authorize every statement before any side effect.
                if let Some(p) = principal.as_ref() {
                    let denial = check_pgwire_denial(pg_ref, p, &sql);
                    if let Err((code, msg)) = denial {
                        record_pgwire_denial(&metrics, &code);
                        stream
                            .write_all(&encode_error_response("ERROR", &code, &msg))
                            .await?;
                        // Simple Query 的响应即使出错也必须以 ReadyForQuery
                        // 收尾，否则 libpq 系客户端会一直等待（OCR 批次三十二
                        // HIGH）。正常路径的 RFQ 在 dispatch 之后统一发送。
                        // EN: Simple Query error responses still require
                        // ReadyForQuery, or libpq clients block waiting.
                        stream.write_all(&encode_ready_for_query()).await?;
                        continue;
                    }
                }
                let t0 = std::time::Instant::now();
                pgwire_dispatch_sql(stream, pg_ref, &sql, principal.as_ref()).await?;
                let elapsed_ms = t0.elapsed().as_millis() as u64;
                let threshold = metrics.slow_query_ms.load(Ordering::Relaxed);
                if crate::types::slow_query_exceeded(elapsed_ms, threshold) {
                    tracing::warn!(
                        elapsed_ms,
                        threshold_ms = threshold,
                        sql = %crate::types::sanitize_sql_for_log(&sql),
                        "slow pgwire simple query"
                    );
                }
                stream.write_all(&encode_ready_for_query()).await?;
            }
            FrontendMessage::Parse {
                name,
                query,
                num_params,
                param_oids,
            } => {
                metrics.parse_count.fetch_add(1, Ordering::Relaxed);
                metrics
                    .sql_text_bytes
                    .fetch_add(query.len() as u64, Ordering::Relaxed);
                let inferred_param_count = if num_params == 0 {
                    infer_pgwire_placeholder_count(&query)
                } else {
                    num_params
                };
                let inferred_param_oids =
                    infer_pgwire_param_oids(&query, inferred_param_count, param_oids);
                if !name.is_empty() {
                    conn.named_statements.insert(
                        name,
                        PreparedStatementEntry {
                            sql: query,
                            param_count: inferred_param_count,
                            param_oids: inferred_param_oids,
                        },
                    );
                    if conn.bound_statement_name.as_deref() == Some("") {
                        conn.bound_unnamed_sql = None;
                        conn.unnamed_result_formats.clear();
                        conn.pending_portal = None;
                    }
                    stream.write_all(&encode_parse_complete()).await?;
                } else {
                    conn.unnamed_stmt = Some(query);
                    conn.unnamed_param_count = inferred_param_count;
                    conn.unnamed_param_oids = inferred_param_oids;
                    conn.bound_unnamed_sql = None;
                    conn.bound_statement_name = None;
                    conn.bound_unnamed_row_description_sent = false;
                    conn.unnamed_result_formats.clear();
                    conn.pending_portal = None;
                    conn.pending_upsert_batch = None;
                    stream.write_all(&encode_parse_complete()).await?;
                }
            }
            FrontendMessage::Bind {
                portal,
                statement,
                params,
                param_formats,
                result_formats,
            } => {
                metrics.bind_count.fetch_add(1, Ordering::Relaxed);
                let param_bytes: usize = params.iter().map(|p| p.encoded_len()).sum();
                metrics
                    .bind_param_text_bytes
                    .fetch_add(param_bytes as u64, Ordering::Relaxed);
                if !portal.is_empty() {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "0A000",
                            "only unnamed portal is supported",
                        ))
                        .await?;
                } else if pgwire_statement_entry(&conn, &statement).is_none() {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "26000",
                            "prepared statement does not exist",
                        ))
                        .await?;
                } else if params.len()
                    != pgwire_statement_entry(&conn, &statement)
                        .map(|entry| entry.param_count)
                        .unwrap_or_default()
                {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "08P01",
                            &format!(
                                "bind parameter count mismatch: expected {}, got {}",
                                pgwire_statement_entry(&conn, &statement)
                                    .map(|entry| entry.param_count)
                                    .unwrap_or_default(),
                                params.len()
                            ),
                        ))
                        .await?;
                } else if pgwire_statement_entry(&conn, &statement)
                    .map(|entry| entry.param_oids)
                    .unwrap_or_default()
                    .iter()
                    .any(|oid| !pgwire_supported_param_oid(*oid))
                {
                    let unsupported = pgwire_statement_entry(&conn, &statement)
                        .map(|entry| entry.param_oids)
                        .unwrap_or_default()
                        .iter()
                        .copied()
                        .find(|oid| !pgwire_supported_param_oid(*oid))
                        .unwrap_or_default();
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "0A000",
                            &format!(
                                "unsupported parameter type OID {} (only unknown/text is supported)",
                                unsupported
                            ),
                        ))
                        .await?;
                } else if pgwire_statement_entry(&conn, &statement)
                    .map(|entry| entry.param_oids)
                    .unwrap_or_default()
                    .iter()
                    .copied()
                    .zip(param_formats.iter().copied())
                    .any(|(oid, format)| !pgwire_supported_param_oid_and_format(oid, format))
                {
                    let (bad_oid, bad_format) = pgwire_statement_entry(&conn, &statement)
                        .map(|entry| entry.param_oids)
                        .unwrap_or_default()
                        .iter()
                        .copied()
                        .zip(param_formats.iter().copied())
                        .find(|(oid, format)| !pgwire_supported_param_oid_and_format(*oid, *format))
                        .unwrap_or((0, 0));
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "0A000",
                            &format!(
                                "unsupported parameter OID/format combination: oid={}, format={} (only configured scalar/vector binary codecs are supported)",
                                bad_oid, bad_format
                            ),
                        ))
                        .await?;
                } else {
                    let statement_entry = pgwire_statement_entry(&conn, &statement).unwrap_or(
                        PreparedStatementEntry {
                            sql: String::new(),
                            param_count: 0,
                            param_oids: Vec::new(),
                        },
                    );
                    let columns = match pgwire_result_columns_for_sql_with_params(
                        &statement_entry.sql,
                        &statement_entry.param_oids,
                    ) {
                        Ok(columns) => columns,
                        Err(e) => {
                            stream
                                .write_all(&encode_error_response("ERROR", "42601", &e))
                                .await?;
                            continue;
                        }
                    };
                    let result_formats =
                        match pgwire_normalize_result_formats(&result_formats, columns.len()) {
                            Ok(formats) => formats,
                            Err(e) => {
                                let code = pgwire_decode_error_code(&e);
                                stream
                                    .write_all(&encode_error_response("ERROR", code, &e))
                                    .await?;
                                continue;
                            }
                        };
                    let vector_param_idx =
                        pgwire_vector_placeholder_param_index(&statement_entry.sql);
                    let upsert_batch_param_idx =
                        pgwire_upsert_batch_placeholder_param_index(&statement_entry.sql);
                    let mut pending_upsert_batch: Option<HouyiUpsertBatch> = None;
                    let bound_params = statement_entry
                        .param_oids
                        .iter()
                        .copied()
                        .zip(param_formats.iter().copied())
                        .zip(params.iter())
                        .enumerate()
                        .map(|(idx, ((oid, format), param))| {
                            if oid == 17
                                && format == 1
                                && vector_param_idx == Some(idx + 1)
                            {
                                match param {
                                    BindParamValue::Binary(bytes) => {
                                        let expected_dim =
                                            pg_ref.as_ref().map(|state| state.vector_dim);
                                        let values = decode_houyi_vector_bytea_binary(
                                            bytes,
                                            expected_dim,
                                        )?;
                                        Ok(pgwire_float_list_to_directive_text(&values))
                                    }
                                    BindParamValue::Text(_) => Err(
                                        "bytea vector payload currently requires binary format"
                                            .to_string(),
                                    ),
                                }
                            } else {
                                if oid == 17 && format == 1 && upsert_batch_param_idx == Some(idx + 1)
                                {
                                    match param {
                                        BindParamValue::Binary(bytes) => {
                                            let expected_dim =
                                                pg_ref.as_ref().map(|state| state.vector_dim);
                                            let batch =
                                                decode_houyi_upsert_batch_bytea_binary(bytes, expected_dim)?;
                                            pending_upsert_batch = Some(batch);
                                            Ok("NULL".to_string())
                                        }
                                        BindParamValue::Text(_) => Err(
                                            "bytea upsert_batch payload currently requires binary format"
                                                .to_string(),
                                        ),
                                    }
                                } else {
                                    pgwire_bind_param_to_sql_literal(oid, format, param)
                                }
                            }
                        })
                        .collect::<Result<Vec<_>, _>>();
                    let bound_params = match bound_params {
                        Ok(values) => values,
                        Err(e) => {
                            let code = pgwire_decode_error_code(&e);
                            stream
                                .write_all(&encode_error_response("ERROR", code, &e))
                                .await?;
                            continue;
                        }
                    };
                    let bound = apply_bind_params(&statement_entry.sql, &bound_params);
                    match bound {
                        Ok(sql) => {
                            conn.bound_unnamed_sql = Some(sql);
                            let is_named_statement = !statement.is_empty();
                            conn.bound_statement_name = Some(if statement.is_empty() {
                                "".to_string()
                            } else {
                                statement
                            });
                            // 命名 statement 在 prepare/Describe(S) 阶段已拿到列描述；再次 Execute 不应重复发送 RowDescription。
                            // Named statements already expose row metadata during prepare/Describe(S), so Execute should not resend it.
                            conn.bound_unnamed_row_description_sent = is_named_statement;
                            conn.unnamed_result_formats = result_formats;
                            conn.pending_portal = None;
                            conn.pending_upsert_batch = pending_upsert_batch;
                            stream.write_all(&encode_bind_complete()).await?;
                        }
                        Err(e) => {
                            stream
                                .write_all(&encode_error_response("ERROR", "08P01", &e))
                                .await?;
                        }
                    }
                }
            }
            FrontendMessage::Describe { kind, name } => {
                metrics.describe_count.fetch_add(1, Ordering::Relaxed);
                if kind != b'S' && kind != b'P' {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "0A000",
                            "Describe kind must be S or P",
                        ))
                        .await?;
                } else if kind == b'S' {
                    let Some(statement_entry) = pgwire_statement_entry(&conn, &name) else {
                        stream
                            .write_all(&encode_error_response(
                                "ERROR",
                                "26000",
                                "prepared statement does not exist",
                            ))
                            .await?;
                        continue;
                    };
                    match pgwire_result_columns_for_sql_with_params(
                        &statement_entry.sql,
                        &statement_entry.param_oids,
                    ) {
                        Ok(columns) => {
                            stream
                                .write_all(&encode_parameter_description_oids(
                                    &statement_entry.param_oids,
                                ))
                                .await?;
                            let formats = vec![0; columns.len()];
                            let row_description =
                                match encode_row_description_columns(&columns, &formats) {
                                    Ok(encoded) => encoded,
                                    Err(e) => {
                                        stream
                                            .write_all(&encode_error_response("ERROR", "0A000", &e))
                                            .await?;
                                        continue;
                                    }
                                };
                            stream.write_all(&row_description).await?;
                            conn.bound_unnamed_row_description_sent = true;
                        }
                        Err(e) => {
                            stream
                                .write_all(&encode_error_response("ERROR", "42601", &e))
                                .await?;
                        }
                    }
                } else if conn.bound_unnamed_sql.is_none() {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "34000",
                            "unnamed portal does not exist",
                        ))
                        .await?;
                } else {
                    let stmt_name = conn.bound_statement_name.clone().unwrap_or_default();
                    let Some(statement_entry) = pgwire_statement_entry(&conn, &stmt_name) else {
                        stream
                            .write_all(&encode_error_response(
                                "ERROR",
                                "26000",
                                "prepared statement does not exist",
                            ))
                            .await?;
                        continue;
                    };
                    match pgwire_result_columns_for_sql_with_params(
                        &statement_entry.sql,
                        &statement_entry.param_oids,
                    ) {
                        Ok(columns) => {
                            let result_formats = match pgwire_normalize_result_formats(
                                &conn.unnamed_result_formats,
                                columns.len(),
                            ) {
                                Ok(formats) => formats,
                                Err(e) => {
                                    let code = pgwire_decode_error_code(&e);
                                    stream
                                        .write_all(&encode_error_response("ERROR", code, &e))
                                        .await?;
                                    continue;
                                }
                            };
                            let row_description =
                                match encode_row_description_columns(&columns, &result_formats) {
                                    Ok(encoded) => encoded,
                                    Err(e) => {
                                        stream
                                            .write_all(&encode_error_response("ERROR", "0A000", &e))
                                            .await?;
                                        continue;
                                    }
                                };
                            stream.write_all(&row_description).await?;
                            conn.bound_unnamed_row_description_sent = true;
                        }
                        Err(e) => {
                            stream
                                .write_all(&encode_error_response("ERROR", "42601", &e))
                                .await?;
                        }
                    }
                }
            }
            FrontendMessage::Execute { portal, max_rows } => {
                metrics.execute_count.fetch_add(1, Ordering::Relaxed);
                if !portal.is_empty() {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "0A000",
                            "only unnamed portal is supported",
                        ))
                        .await?;
                    continue;
                }
                if conn.bound_unnamed_sql.is_none() {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "34000",
                            "unnamed portal does not exist (Bind required)",
                        ))
                        .await?;
                    continue;
                }
                let Some(ref sql) = conn.bound_unnamed_sql else {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "26000",
                            "unnamed prepared statement does not exist",
                        ))
                        .await?;
                    continue;
                };
                // Extended 协议同样逐语句授权（Bind/Execute 可绕过 Simple Query）。
                // EN: authorize extended-protocol executes too.
                if let Some(p) = principal.as_ref() {
                    let denial = check_pgwire_denial(pg_ref, p, sql);
                    if let Err((code, msg)) = denial {
                        record_pgwire_denial(&metrics, &code);
                        stream
                            .write_all(&encode_error_response("ERROR", &code, &msg))
                            .await?;
                        continue;
                    }
                }
                // 事务/会话控制：Extended 路径同样只回 CommandComplete。
                // EN: Transaction/session control: the Extended path likewise replies with
                // CommandComplete only.
                if conn.pending_portal.is_none() {
                    if let Some(tag) = pgwire_noop_command(sql) {
                        stream.write_all(&encode_command_complete(tag)).await?;
                        continue;
                    }
                }
                if conn.pending_portal.is_none() {
                    // 慢查询计时（P4.3.2）：upsert 与查询两路径共用同一阈值。
                    // EN: Slow-query timing (P4.3.2): upsert and query paths share the same
                    // threshold.
                    let t0 = std::time::Instant::now();
                    if let Some(batch) = conn.pending_upsert_batch.take() {
                        let Some(state) = pg_ref else {
                            stream
                                .write_all(&encode_error_response(
                                    "ERROR",
                                    "0A000",
                                    "upsert_batch requires pgwire shared state",
                                ))
                                .await?;
                            continue;
                        };
                        if batch.dim != state.vector_dim {
                            stream
                                .write_all(&encode_error_response(
                                    "ERROR",
                                    "22000",
                                    &format!(
                                        "upsert_batch dim {} != server vector_dim {}",
                                        batch.dim, state.vector_dim
                                    ),
                                ))
                                .await?;
                            continue;
                        }
                        // 内存水位线（P4.2.4）：校验后、WAL/索引写入前裁决——
                        // 超进发 53200（out_of_memory 类），与 HTTP 503 同语义。
                        // EN: Memory watermark admission after validation, before
                        // WAL/index side effects; SQLSTATE 53200 mirrors the HTTP 503.
                        if let Err(msg) = metrics.memory_watermark.check_admission() {
                            stream
                                .write_all(&encode_error_response("ERROR", "53200", &msg))
                                .await?;
                            continue;
                        }
                        // 只读副本（P4.4.1）：follower 拒写，SQLSTATE 25006
                        // （read_only_sql_transaction，PostgreSQL 只读事务同款）
                        // ——覆盖下方 collection/主索引两个写入分支；检索天然放行。
                        // EN: Followers reject upserts with 25006, guarding both
                        // write branches below; reads pass through untouched.
                        if state.role.is_follower() {
                            stream
                                .write_all(&encode_error_response(
                                    "ERROR",
                                    "25006",
                                    "read_only: follower replica rejects writes",
                                ))
                                .await?;
                            continue;
                        }
                        // 租约 fencing（EE M1）：与 HTTP 同一句柄；未开启租约要求
                        // 时恒放行。25006 与只读副本同码——客户端语义一致。
                        // EN: lease fencing (EE M1) shares the HTTP handle; a no-op
                        // when leases are not required. 25006 matches follower
                        // semantics for clients.
                        if let Err(e) = state.cluster.check_writable() {
                            stream
                                .write_all(&encode_error_response(
                                    "ERROR",
                                    "25006",
                                    &format!("read_only: {e}"),
                                ))
                                .await?;
                            continue;
                        }
                        // 路由：有 Collection 指令时走 Collection，否则走默认 index（向后兼容）
                        // EN: Routing: use a Collection when the directive is present, else the
                        // default index (backward compatible).
                        let inserted = batch.items.len() as i64;
                        let upsert_sql = conn.bound_unnamed_sql.as_deref().unwrap_or("");
                        let coll_name = houyi_query::extract_collection_directive(upsert_sql);
                        // collection 分支带出（col, 锁内捕获的 ha_level, 最后一条
                        // WAL LSN）供锁释放后的 sync 等待（P4.4.2 语义，pgwire 于
                        // P4.4.3 补齐——此前 pgwire 客户端写 full_sync collection
                        // 实际只得 async 语义）。主索引分支不在复制范围内，无等待语义。
                        // EN: The collection branch yields (col, ha_level captured
                        // under the write lock, last WAL LSN) for the post-lock sync
                        // wait; the main-index path is not replicated.
                        let coll_sync: Option<CollSyncWait> = if let Some(ref name) = coll_name {
                            let col = state
                                .collections
                                .get_or_create(name, Some(batch.dim))
                                .map_err(std::io::Error::other)?;
                            // 写互斥（与 HTTP 路径同纪律）：WAL append 与 insert_batch
                            // 同序，避免并发同 id 批次的 WAL/内存顺序分叉。
                            // EN: Write mutex, same discipline as the HTTP path — the
                            // WAL append and insert_batch share one order so
                            // concurrent same-id batches cannot diverge.
                            // 删除竞态：SQLSTATE 40001（serialization_failure）——
                            // 客户端可安全重试；与 HTTP 边界 409 同语义。
                            // EN: Deletion race — SQLSTATE 40001 (serialization_failure),
                            // safely retryable client-side; same semantics as HTTP 409.
                            let _write = match col.begin_write().await {
                                Ok(guard) => guard,
                                Err(_) => {
                                    stream
                                        .write_all(&encode_error_response(
                                            "ERROR",
                                            "40001",
                                            &format!("collection '{name}' is being deleted"),
                                        ))
                                        .await?;
                                    continue;
                                }
                            };
                            // WAL-first（与 HTTP collection upsert 同纪律）：pgwire 批量
                            // 不携带 label/payload，items 只有 id+values。测试装配
                            // （collections_wal=None）跳过落盘。
                            // EN: WAL-first, same discipline as the HTTP collection upsert.
                            // pgwire batches carry no label/payload; test assemblies
                            // (collections_wal=None) skip persistence.
                            let mut last_wal_lsn = None;
                            if let Some(ref cw) = state.collections_wal {
                                // 建集合记录先于本批 upsert 落盘（首个写锁持有者负责，
                                // 参数取自 collection 自身；详见 ensure_create_logged）。
                                // EN: the create record precedes this upsert batch — the
                                // first write-lock holder logs it from the collection's own
                                // parameters (see ensure_create_logged).
                                crate::collection::ensure_create_logged(cw, name, &col)
                                    .map_err(std::io::Error::other)?;
                                let wal_items: Vec<
                                    houyi_persistence::collection_wal::CollectionWalItem,
                                > = batch
                                    .items
                                    .iter()
                                    .map(|(id, values)| {
                                        houyi_persistence::collection_wal::CollectionWalItem {
                                            id: *id,
                                            values: values.clone(),
                                            label: None,
                                            payload: None,
                                        }
                                    })
                                    .collect();
                                let payload = houyi_persistence::collection_wal::encode_collection_upsert_batch(
                                    name,
                                    batch.dim,
                                    &wal_items,
                                )
                                .map_err(std::io::Error::other)?;
                                // sync 等待只等本写**最后一条** WAL LSN（create+
                                // upsert 两条时 upsert 的 LSN 有序覆盖前一条）。
                                // EN: Wait on this write's **last** WAL LSN only (create + upsert:
                                // the upsert LSN orders after and covers the earlier one).
                                last_wal_lsn = Some(cw.append(payload)?.lsn);
                            }
                            if let Some(m) = state.pgwire_hnsw_m {
                                if m != col.backend.hnsw_m() {
                                    let _ = col.backend.set_hnsw_m(m);
                                }
                            }
                            if let Some(efc) = state.pgwire_hnsw_ef_construction {
                                if efc != col.backend.hnsw_ef_construction_value() {
                                    let _ = col.backend.set_hnsw_ef_construction(efc);
                                }
                            }
                            col.backend
                                .insert_batch(batch.items)
                                .await
                                .map_err(std::io::Error::other)?;
                            if let Some(lsn) = last_wal_lsn {
                                col.note_applied_lsn(lsn);
                            }
                            // 级别在 _write 锁内捕获（与 HTTP 同纪律）：定格在本写
                            // WAL LSN 处的在效级别，防并发 set_ha_level 降级抢语义。
                            // EN: Level captured inside the _write lock (same discipline as HTTP):
                            // pin the level in effect at this write's WAL LSN, preventing a
                            // concurrent set_ha_level downgrade from stealing the semantics.
                            let ha_level = col.ha_level();
                            last_wal_lsn.map(|lsn| CollSyncWait {
                                col: col.clone(),
                                ha_level,
                                lsn,
                            })
                        } else {
                            if let Some(m) = state.pgwire_hnsw_m {
                                if m != state.index.hnsw_m() {
                                    let _ = state.index.set_hnsw_m(m);
                                }
                            }
                            if let Some(efc) = state.pgwire_hnsw_ef_construction {
                                if efc != state.index.hnsw_ef_construction_value() {
                                    let _ = state.index.set_hnsw_ef_construction(efc);
                                }
                            }
                            // 主索引路径此前完全绕过 WAL（pgwire 特有的持久化缺口，
                            // 与 HTTP benchmark_upsert 同纪律补齐：WAL-first）。
                            // EN: The main-index path previously bypassed the WAL
                            // entirely — a pgwire-specific durability gap; close it
                            // WAL-first like the HTTP benchmark_upsert.
                            if let Some(ref main_wal) = state.wal {
                                let payload =
                                    houyi_persistence::vector_wal::encode_vector_upsert_batch(
                                        batch.dim,
                                        &batch.items,
                                    )
                                    .map_err(std::io::Error::other)?;
                                main_wal.append(payload)?;
                            }
                            state
                                .index
                                .insert_batch(batch.items)
                                .await
                                .map_err(std::io::Error::other)?;
                            None
                        };
                        // sync 等待（写锁已随分支结束释放——等待是纯观测，持锁等待
                        // 会把并发写串行到 sync RTT 上；与 HTTP 路径同纪律）。
                        // EN: The write lock was released with the branch; waiting is
                        // pure observation, same discipline as the HTTP path.
                        if let Some(wait) = coll_sync {
                            crate::http_handler::sync_wait_if_required(
                                &state.replication_metrics,
                                &wait.col,
                                wait.ha_level,
                                wait.lsn,
                            )
                            .await;
                        }
                        metrics
                            .upsert_batch_execute_count
                            .fetch_add(1, Ordering::Relaxed);
                        metrics
                            .upsert_batch_rows_inserted
                            .fetch_add(inserted as u64, Ordering::Relaxed);
                        let elapsed_ms = t0.elapsed().as_millis() as u64;
                        let threshold = metrics.slow_query_ms.load(Ordering::Relaxed);
                        if crate::types::slow_query_exceeded(elapsed_ms, threshold) {
                            tracing::warn!(
                                elapsed_ms,
                                threshold_ms = threshold,
                                rows = inserted,
                                sql = %crate::types::sanitize_sql_for_log(sql),
                                "slow pgwire upsert batch"
                            );
                        }
                        let upsert_columns = vec![PgWireColumnMeta {
                            name: "inserted".to_string(),
                            type_oid: 20,
                        }];
                        let result_formats = match pgwire_normalize_result_formats(
                            &conn.unnamed_result_formats,
                            upsert_columns.len(),
                        ) {
                            Ok(formats) => formats,
                            Err(e) => {
                                let code = pgwire_decode_error_code(&e);
                                stream
                                    .write_all(&encode_error_response("ERROR", code, &e))
                                    .await?;
                                continue;
                            }
                        };
                        if !conn.bound_unnamed_row_description_sent {
                            let row_description = match encode_row_description_columns(
                                &upsert_columns,
                                &result_formats,
                            ) {
                                Ok(encoded) => encoded,
                                Err(e) => {
                                    stream
                                        .write_all(&encode_error_response("ERROR", "0A000", &e))
                                        .await?;
                                    continue;
                                }
                            };
                            stream.write_all(&row_description).await?;
                            conn.bound_unnamed_row_description_sent = true;
                        }
                        conn.pending_portal = Some(PendingPortalResult {
                            columns: upsert_columns,
                            total_rows: 1,
                            rows: vec![vec![inserted.to_string()]],
                            next_idx: 0,
                        });
                    }
                    if conn.pending_portal.is_none() {
                        match pgwire_collect_rows(pg_ref, sql).await {
                            Ok((columns, rows)) => {
                                record_pgwire_query(
                                    pg_ref,
                                    principal.as_ref(),
                                    sql,
                                    200,
                                    t0.elapsed().as_secs_f64() * 1000.0,
                                    rows.len(),
                                );
                                let elapsed_ms = t0.elapsed().as_millis() as u64;
                                let threshold = metrics.slow_query_ms.load(Ordering::Relaxed);
                                if crate::types::slow_query_exceeded(elapsed_ms, threshold) {
                                    tracing::warn!(
                                        elapsed_ms,
                                        threshold_ms = threshold,
                                        rows = rows.len(),
                                        sql = %crate::types::sanitize_sql_for_log(sql),
                                        "slow pgwire extended query"
                                    );
                                }
                                let result_formats = match pgwire_normalize_result_formats(
                                    &conn.unnamed_result_formats,
                                    columns.len(),
                                ) {
                                    Ok(formats) => formats,
                                    Err(e) => {
                                        let code = pgwire_decode_error_code(&e);
                                        stream
                                            .write_all(&encode_error_response("ERROR", code, &e))
                                            .await?;
                                        continue;
                                    }
                                };
                                if !conn.bound_unnamed_row_description_sent {
                                    let row_description = match encode_row_description_columns(
                                        &columns,
                                        &result_formats,
                                    ) {
                                        Ok(encoded) => encoded,
                                        Err(e) => {
                                            stream
                                                .write_all(&encode_error_response(
                                                    "ERROR", "0A000", &e,
                                                ))
                                                .await?;
                                            continue;
                                        }
                                    };
                                    stream.write_all(&row_description).await?;
                                    conn.bound_unnamed_row_description_sent = true;
                                }
                                conn.pending_portal = Some(PendingPortalResult {
                                    columns,
                                    total_rows: rows.len(),
                                    rows,
                                    next_idx: 0,
                                });
                            }
                            Err((code, msg)) => {
                                record_pgwire_query(
                                    pg_ref,
                                    principal.as_ref(),
                                    sql,
                                    pgwire_log_status(&code),
                                    t0.elapsed().as_secs_f64() * 1000.0,
                                    0,
                                );
                                stream
                                    .write_all(&encode_error_response("ERROR", &code, &msg))
                                    .await?;
                                continue;
                            }
                        }
                    }
                }
                let Some(mut pending) = conn.pending_portal.take() else {
                    continue;
                };
                let remaining = pending.rows.len().saturating_sub(pending.next_idx);
                let emit_n = if max_rows == 0 || max_rows == u32::MAX {
                    remaining
                } else {
                    remaining.min(max_rows as usize)
                };
                let end = pending.next_idx + emit_n;
                let result_formats = match pgwire_normalize_result_formats(
                    &conn.unnamed_result_formats,
                    pending.columns.len(),
                ) {
                    Ok(formats) => formats,
                    Err(e) => {
                        let code = pgwire_decode_error_code(&e);
                        stream
                            .write_all(&encode_error_response("ERROR", code, &e))
                            .await?;
                        continue;
                    }
                };
                for row in &pending.rows[pending.next_idx..end] {
                    let encoded =
                        match encode_data_row_columns(row, &pending.columns, &result_formats) {
                            Ok(msg) => msg,
                            Err(e) => {
                                let code = pgwire_decode_error_code(&e);
                                stream
                                    .write_all(&encode_error_response("ERROR", code, &e))
                                    .await?;
                                continue;
                            }
                        };
                    stream.write_all(&encoded).await?;
                }
                pending.next_idx = end;
                if pending.next_idx < pending.rows.len() {
                    conn.pending_portal = Some(pending);
                    stream.write_all(&encode_portal_suspended()).await?;
                } else {
                    stream
                        .write_all(&encode_command_complete(&format!(
                            "SELECT {}",
                            pending.total_rows
                        )))
                        .await?;
                }
            }
            FrontendMessage::Close { kind, name } => {
                metrics.close_count.fetch_add(1, Ordering::Relaxed);
                if kind == b'S' {
                    if name.is_empty() {
                        conn.unnamed_stmt = None;
                        conn.unnamed_param_count = 0;
                        conn.unnamed_param_oids.clear();
                    } else {
                        conn.named_statements.remove(&name);
                    }
                    if conn.bound_statement_name.as_deref() == Some(name.as_str())
                        || (name.is_empty() && conn.bound_statement_name.as_deref() == Some(""))
                    {
                        conn.bound_unnamed_sql = None;
                        conn.bound_statement_name = None;
                        conn.bound_unnamed_row_description_sent = false;
                        conn.unnamed_result_formats.clear();
                        conn.pending_portal = None;
                        conn.pending_upsert_batch = None;
                    }
                    stream.write_all(&encode_close_complete()).await?;
                } else if kind == b'P' {
                    conn.bound_unnamed_sql = None;
                    conn.bound_statement_name = None;
                    conn.bound_unnamed_row_description_sent = false;
                    conn.unnamed_result_formats.clear();
                    conn.pending_portal = None;
                    conn.pending_upsert_batch = None;
                    stream.write_all(&encode_close_complete()).await?;
                } else {
                    stream
                        .write_all(&encode_error_response(
                            "ERROR",
                            "0A000",
                            "Close kind must be S or P",
                        ))
                        .await?;
                }
            }
            FrontendMessage::Sync => {
                metrics.sync_count.fetch_add(1, Ordering::Relaxed);
                conn.pending_portal = None;
                conn.pending_upsert_batch = None;
                stream.write_all(&encode_ready_for_query()).await?;
            }
            FrontendMessage::Flush => {
                metrics.flush_count.fetch_add(1, Ordering::Relaxed);
                stream.flush().await?;
            }
            FrontendMessage::CopyData | FrontendMessage::CopyDone => {
                metrics.copy_count.fetch_add(1, Ordering::Relaxed);
                stream
                    .write_all(&encode_error_response(
                        "ERROR",
                        "0A000",
                        "COPY protocol is not supported",
                    ))
                    .await?;
            }
            FrontendMessage::CopyFail(client_msg) => {
                metrics.copy_count.fetch_add(1, Ordering::Relaxed);
                let msg = if client_msg.trim().is_empty() {
                    "COPY protocol is not supported".to_string()
                } else {
                    format!("COPY protocol is not supported: {}", client_msg.trim())
                };
                // client_msg 是客户端可控的原始载荷：原样落日志会让任意连接方
                // (1) 用大消息灌爆持久化日志文件、(2) 内嵌换行/控制符伪造日志行。
                // 只记长度 + 截断转义前缀。
                // EN: client_msg is client-controlled — log byte length plus a
                // truncated, escape_default'd prefix instead of the raw payload
                // (log flooding / forged log lines otherwise).
                let preview: String = client_msg
                    .chars()
                    .take(128)
                    .collect::<String>()
                    .escape_default()
                    .collect();
                tracing::warn!(
                    client_msg_len = client_msg.len(),
                    client_msg_preview = %preview,
                    "pgwire copy-fail from client"
                );
                stream
                    .write_all(&encode_error_response("ERROR", "0A000", &msg))
                    .await?;
            }
            FrontendMessage::Terminate => {
                metrics.terminate_count.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            FrontendMessage::Unsupported(tag) => {
                metrics.unsupported_count.fetch_add(1, Ordering::Relaxed);
                let msg = format!("unsupported frontend message: {}", tag as char);
                stream
                    .write_all(&encode_error_response("ERROR", "0A000", &msg))
                    .await?;
            }
        }
    }
}

/// pgwire 边缘的检索放大面封边回归（OCR 复扫：HTTP 加上 k 上界后，pgwire 的
/// `LIMIT n` / `/* houyi:ef_search=N */` 仍是裸 usize，可驱动同样的 16×ann_k
/// 访问器预分配直至 alloc-abort）。
/// EN: Edge regression for the pgwire retrieval amplification surface — after the
/// HTTP k cap landed, pgwire `LIMIT n` / ef_search directive stayed unbounded and
/// could drive the same 16×ann_k visit-tracker preallocation to alloc-abort.
#[cfg(test)]
mod search_limit_edge_tests {
    use super::*;
    use crate::collection::CollectionManager;
    use crate::http_handler::{MAX_SEARCH_EF, MAX_SEARCH_K};

    fn pg_state() -> Arc<PgWireSharedState> {
        Arc::new(PgWireSharedState {
            replication_metrics: crate::types::shared_replication_metrics(None),
            index: IndexBackend::new_hnsw(4),
            collections: Arc::new(CollectionManager::new(4, "hnsw")),
            vector_dim: 4,
            pgwire_hnsw_m: None,
            pgwire_hnsw_ef_construction: None,
            wal: None,
            collections_wal: None,
            role: crate::types::RoleState::default(),
            cluster: std::sync::Arc::new(
                crate::cluster::ClusterControl::open(
                    &std::env::temp_dir()
                        .join(format!(
                            "houyi-pgwire-test-cluster-{}-{}-{}",
                            std::process::id(),
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_nanos(),
                            {
                                static SEQ: std::sync::atomic::AtomicU64 =
                                    std::sync::atomic::AtomicU64::new(0);
                                SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            }
                        ))
                        .join("test.wal"),
                    false,
                )
                .expect("cluster control"),
            ),
            query_log: None,
        })
    }

    /// 超大 LIMIT 必须在进入检索链前以 22023（invalid_parameter_value）拒绝。
    /// EN: An over-cap LIMIT is rejected with 22023 before reaching the retrieval chain.
    #[tokio::test]
    async fn pgwire_rejects_over_cap_limit() {
        let pg = pg_state();
        let sql = format!(
            /* 样例 / EN: sample */
            "/* houyi:vector=1,0,0,0 */ SELECT id FROM docs LIMIT {}",
            MAX_SEARCH_K + 1
        );
        let (sqlstate, msg) = pgwire_collect_rows(Some(&pg), &sql)
            .await
            .err()
            .expect("over-cap LIMIT must be rejected");
        assert_eq!(sqlstate, "22023");
        assert!(
            msg.contains(&MAX_SEARCH_K.to_string()),
            "error must name the bound, got: {msg}"
        );
    }

    /// 超大 houyi:ef_search 指令同源放大（ann_k = k.max(ef)），同样 22023 拒绝。
    /// EN: An over-cap ef_search directive drives the same ann_k path — rejected too.
    #[tokio::test]
    async fn pgwire_rejects_over_cap_ef_search_directive() {
        let pg = pg_state();
        let sql = format!(
            /* 样例 / EN: sample */
            "/* houyi:ef_search={} */ /* houyi:vector=1,0,0,0 */ SELECT id FROM docs LIMIT 10",
            MAX_SEARCH_EF + 1
        );
        let (sqlstate, msg) = pgwire_collect_rows(Some(&pg), &sql)
            .await
            .err()
            .expect("over-cap ef_search must be rejected");
        assert_eq!(sqlstate, "22023");
        assert!(
            msg.contains(&MAX_SEARCH_EF.to_string()),
            "error must name the bound, got: {msg}"
        );
    }

    /// 恰好等于上界的 LIMIT 必须放行（空索引返回 0 行，仅验证边缘不误伤边界值）。
    /// EN: LIMIT exactly at the cap passes (empty index → 0 rows; boundary value
    /// must not be rejected).
    #[tokio::test]
    async fn pgwire_accepts_limit_at_cap() {
        let pg = pg_state();
        let sql = format!(
            /* 样例 / EN: sample */
            "/* houyi:vector=1,0,0,0 */ SELECT id FROM docs LIMIT {MAX_SEARCH_K}"
        );
        let (_meta, rows) = pgwire_collect_rows(Some(&pg), &sql)
            .await
            .expect("LIMIT == cap must be accepted");
        assert!(rows.is_empty());
    }
}

/// P4.7.2 配额裁决（OCR 批次三十四）：SELECT 隐式建集合、SQLSTATE 54000、
/// 计数分账。
/// EN: quota adjudication — implicit creates via SELECT, 54000, metric split.
#[cfg(test)]
mod quota_tests {
    use super::*;
    use crate::collection::CollectionManager;

    fn pg_state() -> Arc<PgWireSharedState> {
        Arc::new(PgWireSharedState {
            replication_metrics: crate::types::shared_replication_metrics(None),
            index: IndexBackend::new_hnsw(4),
            collections: Arc::new(CollectionManager::new(4, "hnsw")),
            vector_dim: 4,
            pgwire_hnsw_m: None,
            pgwire_hnsw_ef_construction: None,
            wal: None,
            collections_wal: None,
            role: crate::types::RoleState::default(),
            cluster: std::sync::Arc::new(
                crate::cluster::ClusterControl::open(
                    &std::env::temp_dir()
                        .join(format!(
                            "houyi-pgwire-test-cluster-{}-{}-{}",
                            std::process::id(),
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_nanos(),
                            {
                                static SEQ: std::sync::atomic::AtomicU64 =
                                    std::sync::atomic::AtomicU64::new(0);
                                SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            }
                        ))
                        .join("test.wal"),
                    false,
                )
                .expect("cluster control"),
            ),
            query_log: None,
        })
    }

    fn principal_with_quota(max_collections: usize) -> crate::rbac::Principal {
        let policy = crate::rbac::AuthzPolicy::from_config(
            None,
            &[PrincipalConfig {
                name: "w".into(),
                role: "writer".into(),
                token: Some("quota-token".into()),
                token_sha256: None,
                collections: None,
                max_collections: Some(max_collections),
                max_inflight: None,
                writes_per_sec: None,
            }],
        )
        .unwrap();
        policy.resolve("quota-token").unwrap().clone()
    }

    /// 回归（OCR HIGH）：带 collection 指令的 SELECT 会隐式建集合，
    /// 配额必须按「名字是否存在」判定，而不是按读写动作。
    /// EN: Regression (OCR HIGH): a SELECT with a collection directive implicitly
    /// creates a collection; quota must be judged by whether the name exists, not
    /// by the read/write action.
    #[tokio::test]
    async fn quota_covers_implicit_create_via_select() {
        let pg = pg_state();
        pg.collections.get_or_create("docs", Some(4)).unwrap();
        let p = principal_with_quota(1);
        let read_existing = /* 样例 / EN: sample */"/* houyi:collection=docs */ /* houyi:vector=1,0,0,0 */ \
                             SELECT id FROM docs LIMIT 1";
        assert!(
            check_pgwire_denial(Some(&pg), &p, read_existing).is_ok(),
            "reading an existing collection is never blocked"
        );
        let read_new = /* 样例 / EN: sample */"/* houyi:collection=new */ /* houyi:vector=1,0,0,0 */ \
                        SELECT id FROM docs LIMIT 1";
        let (code, msg) = check_pgwire_denial(Some(&pg), &p, read_new)
            .expect_err("SELECT must not bypass the collection quota");
        assert_eq!(code, "54000");
        assert!(msg.contains("max_collections"), "got: {msg}");
    }

    #[test]
    fn transaction_control_is_classified() {
        assert_eq!(pgwire_noop_command("BEGIN"), Some("BEGIN"));
        assert_eq!(pgwire_noop_command("  start transaction"), Some("BEGIN"));
        assert_eq!(pgwire_noop_command("COMMIT;"), Some("COMMIT"));
        assert_eq!(pgwire_noop_command("ROLLBACK"), Some("ROLLBACK"));
        assert_eq!(pgwire_noop_command("SET search_path TO x"), Some("SET"));
        assert_eq!(pgwire_noop_command("SELECT 1"), None);
    }

    #[test]
    fn denial_counters_are_separated() {
        let metrics = PgWireMetrics::default();
        record_pgwire_denial(&metrics, "54000");
        record_pgwire_denial(&metrics, "42501");
        assert_eq!(metrics.quota_denials.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.authz_denials.load(Ordering::Relaxed), 1);
    }

    async fn read_backend<S: AsyncRead + Unpin>(stream: &mut S) -> (u8, Vec<u8>) {
        let mut header = [0u8; 5];
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            stream.read_exact(&mut header),
        )
        .await
        .expect("backend message timed out")
        .expect("read backend header");
        let len = i32::from_be_bytes(header[1..5].try_into().unwrap()) as usize;
        let mut payload = vec![0u8; len - 4];
        stream.read_exact(&mut payload).await.unwrap();
        (header[0], payload)
    }

    /// 线级回归：Simple Query 配额拒绝必须是 ErrorResponse(54000) +
    /// ReadyForQuery，且只增加 quota_denials。
    /// EN: Wire-level regression: a Simple Query quota rejection must be
    /// ErrorResponse(54000) + ReadyForQuery, and only bump quota_denials.
    #[tokio::test]
    async fn simple_query_quota_rejection_is_54000_with_ready_for_query() {
        let pg = pg_state();
        pg.collections.get_or_create("docs", Some(4)).unwrap();
        let principal = principal_with_quota(1);
        let metrics = Arc::new(PgWireMetrics::default());
        let task_metrics = Arc::clone(&metrics);
        let (mut client, mut server) = tokio::io::duplex(64 * 1024);
        let task = tokio::spawn(async move {
            handle_pgwire_simple_query_loop(&mut server, Some(pg), task_metrics, Some(principal))
                .await
        });

        let sql = /* 样例 / EN: sample */"/* houyi:collection=new */ /* houyi:vector=1,0,0,0 */ \
                   SELECT id FROM docs LIMIT 1";
        let mut frame = vec![b'Q'];
        frame.extend_from_slice(&((sql.len() + 5) as i32).to_be_bytes());
        frame.extend_from_slice(sql.as_bytes());
        frame.push(0);
        client.write_all(&frame).await.unwrap();

        let (kind, payload) = read_backend(&mut client).await;
        assert_eq!(kind, b'E', "expected ErrorResponse");
        assert!(
            payload.windows(5).any(|w| w == b"54000"),
            "expected SQLSTATE 54000, got payload: {payload:?}"
        );
        let (kind, _) = read_backend(&mut client).await;
        assert_eq!(
            kind, b'Z',
            "Simple Query errors must end with ReadyForQuery"
        );

        drop(client);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), task).await;
        assert_eq!(metrics.quota_denials.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.authz_denials.load(Ordering::Relaxed), 0);
    }
}
