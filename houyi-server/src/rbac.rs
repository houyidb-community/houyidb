//! 配置式 RBAC（P4.7.1）。
//! Config-based RBAC (P4.7.1).
//!
//! 模型 / Model:
//! - **主体 Principal**：`name`（审计用）、`role`、可选 `collections` 作用域、
//!   token 的 SHA-256 哈希（明文不落配置也可落，但内存里只留哈希）。
//! - **角色 Role**：`admin`（全部）> `writer`（读 + 写）> `reader`（只读）。
//! - **动作 Action**：Read / Write / Admin。HTTP 由方法与路径推导，pgwire
//!   由 SQL 首关键字推导（只读：SELECT/SHOW/EXPLAIN/VALUES/TABLE，其余按写）。
//! - **作用域**：`collections=None` = 全部；列表中的 `"*"` 等价全部。带作用域
//!   的主体只能访问列出的 collection；主索引/embeddings/管理面属于全局数据，
//!   带作用域主体默认不可写（读元数据端点仍放行）。
//!
//! 为什么先做配置式而不是动态管理 API（WHY）：当前是单机 + 手动 promote，
//! 动态 token 需要持久化与复制一致性（否则 promote 后 token 丢失/权限错配），
//! 会把认证和复制两条主线同时拖入风险区。配置可版本化、可 GitOps、无引导
//! 问题；接口上保留 `AuthzPolicy` 这一层，将来接目录/OIDC 不动客户端。
//! EN: config-first because dynamic tokens require persistence + replication
//! consistency before manual promote is mature. The policy layer is the seam
//! for a future catalog/OIDC provider.
//!
//! 安全细节 / Security: token 仅以 SHA-256 哈希驻留；比较走常量时间；未知
//! token 401、已知但越权 403；原始凭据永不进日志/审计（审计记主体名）。
//! EN: hashes at rest, constant-time compare, 401 vs 403 split, raw tokens
//! never logged.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde::Serialize;

use crate::audit::sha256_hex;
use crate::types::{constant_time_eq, PrincipalConfig};

/// 角色。
/// EN: Role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Role {
    Admin,
    Writer,
    Reader,
    /// 控制面角色（EE M1）：只能访问 lease/demote/promote/replication-status。
    /// 由专用 `sentinel_token` 派生，不与数据面读写权限交叉。
    /// EN: control-plane role (EE M1) derived from the dedicated sentinel token;
    /// grants only lease/demote/promote/replication-status.
    Sentinel,
}

impl Role {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "admin" => Ok(Self::Admin),
            "writer" => Ok(Self::Writer),
            "reader" => Ok(Self::Reader),
            "sentinel" => Ok(Self::Sentinel),
            other => Err(format!(
                "invalid role {other:?}: expect admin | writer | reader | sentinel"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admin => "admin",
            Self::Writer => "writer",
            Self::Reader => "reader",
            Self::Sentinel => "sentinel",
        }
    }
}

/// 动作。
/// EN: Action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Read,
    Write,
    Admin,
    /// sentinel 控制面动作（lease/demote/promote/replication-status）。
    /// EN: sentinel control actions (lease/demote/promote/replication-status).
    Sentinel,
    /// 只读元数据（`/v1/version`）：任何已认证角色可读，sentinel 也需要它做
    /// 能力探测——不归入数据面 Read，避免 sentinel 借此读 collection。
    /// EN: read-only metadata (`/v1/version`) open to every authenticated role;
    /// kept separate from data-plane Read so the sentinel role cannot read data.
    Meta,
}

/// 已认证主体。
/// EN: Authenticated principal.
#[derive(Clone)]
pub struct Principal {
    pub name: String,
    pub role: Role,
    /// None = 全部；Some 列表含 "*" 亦为全部。
    /// EN: None = all; a Some list containing "*" also means all.
    pub collections: Option<Vec<String>>,
    /// 可见 collection 数上限（仅约束新建）。
    /// EN: Upper bound on visible collections (applies to new ones only).
    pub max_collections: Option<usize>,
    /// 在途请求上限；0 = 不限。
    /// EN: In-flight request limit; 0 = unlimited.
    pub max_inflight: usize,
    /// 写请求速率（次/秒）；None = 不限。
    /// EN: Write request rate (per second); None = unlimited.
    pub writes_per_sec: Option<u32>,
    token_hash: String,
    /// 每主体在途许可（max_inflight=0 时为 None）。
    /// EN: Per-principal in-flight permit (None when max_inflight = 0).
    inflight: Option<Arc<tokio::sync::Semaphore>>,
    /// 每主体写令牌桶。
    /// EN: Per-principal write token bucket.
    write_bucket: Option<Arc<std::sync::Mutex<TokenBucket>>>,
}

/// 简单令牌桶（每主体独立，短临界区；不在全局锁上）。
/// Per-principal token bucket with a tiny critical section.
#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last: std::time::Instant,
    rate: f64,
    burst: f64,
}

impl TokenBucket {
    fn new(rate: u32) -> Self {
        let r = f64::from(rate);
        Self {
            tokens: r,
            last: std::time::Instant::now(),
            rate: r,
            burst: r,
        }
    }

    fn try_take(&mut self) -> bool {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + elapsed * self.rate).min(self.burst);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

fn build_principal(
    name: String,
    role: Role,
    collections: Option<Vec<String>>,
    token_hash: String,
    max_collections: Option<usize>,
    max_inflight: usize,
    writes_per_sec: Option<u32>,
) -> Principal {
    Principal {
        name,
        role,
        collections,
        max_collections,
        max_inflight,
        writes_per_sec,
        token_hash,
        inflight: (max_inflight > 0).then(|| Arc::new(tokio::sync::Semaphore::new(max_inflight))),
        write_bucket: writes_per_sec.map(|r| Arc::new(std::sync::Mutex::new(TokenBucket::new(r)))),
    }
}

// token 哈希即使是哈希也是凭据材料（弱 token 可离线爆破）：Debug 明确
// 打码，且不派生 Serialize——避免未来 `serde_json::to_*` 或结构化日志
// 意外把哈希带出去（OCR 批次三十二）。
// EN: the token hash is credential material (weak tokens are offline-crackable);
// redact it in Debug and never serialize the principal.
impl std::fmt::Debug for Principal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Principal")
            .field("name", &self.name)
            .field("role", &self.role)
            .field("collections", &self.collections)
            .field("max_collections", &self.max_collections)
            .field("max_inflight", &self.max_inflight)
            .field("writes_per_sec", &self.writes_per_sec)
            .field("token_hash", &"<redacted>")
            .finish()
    }
}

/// 作用域匹配的单一实现（Principal 与 AuthenticatedPrincipal 共用）。
/// 为什么抽成自由函数（OCR 批次三十四）：两处曾各写一份 `None`/`"*"` 语义，
/// 今天等价、明天就可能分叉——HTTP 用快照副本、pgwire 用原主体，分叉会让
/// 两个接口面看到不同的可见集合。
/// EN: one implementation of scope matching shared by `Principal` and
/// `AuthenticatedPrincipal`; the two copies had to stay in lockstep forever.
fn scope_matches(collections: &Option<Vec<String>>, name: &str) -> bool {
    match collections {
        None => true,
        Some(list) => list.iter().any(|c| c == "*" || c == name),
    }
}

/// 集合配额判定（单一实现）：已存在不算新建；只数作用域内名称。
/// EN: single quota implementation — existing names are never counted.
fn scope_quota_exceeded(
    max_collections: Option<usize>,
    collections: &Option<Vec<String>>,
    existing: &[String],
    target: &str,
) -> bool {
    let Some(limit) = max_collections else {
        return false;
    };
    if existing.iter().any(|n| n == target) {
        return false;
    }
    existing
        .iter()
        .filter(|n| scope_matches(collections, n))
        .count()
        >= limit
}

impl Principal {
    fn scope_all(&self) -> bool {
        match &self.collections {
            None => true,
            Some(list) => list.iter().any(|c| c == "*"),
        }
    }

    /// 授权判定。`collection=None` + `global_data=true` 表示主索引/管理面等
    /// 全局数据资源；带作用域主体默认拒绝（fail-closed）。
    /// EN: authorize an action; scoped principals are denied global data by default.
    pub fn allows(&self, action: Action, collection: Option<&str>, global_data: bool) -> bool {
        let role_ok = match action {
            Action::Read => matches!(self.role, Role::Admin | Role::Writer | Role::Reader),
            Action::Write => matches!(self.role, Role::Admin | Role::Writer),
            Action::Admin => self.role == Role::Admin,
            // admin 隐含控制面权限（排障/手动故障转移）；sentinel 角色只有这一项。
            // EN: admin implies control-plane access (manual failover); the
            // sentinel role has nothing else.
            Action::Sentinel => matches!(self.role, Role::Admin | Role::Sentinel),
            Action::Meta => true,
        };
        if !role_ok {
            return false;
        }
        if self.scope_all() {
            return true;
        }
        match collection {
            Some(name) => self.may_see(name),
            None => {
                // 全局数据（主索引/embeddings/admin）需要 "*" 或未设作用域；
                // 纯元数据读（如列出 collections）对有作用域主体放行。
                // EN: Global data (main index/embeddings/admin) requires "*" or no scope;
                // EN: pure metadata reads (e.g. listing collections) are allowed for scoped principals.
                if global_data {
                    false
                } else {
                    action == Action::Read
                }
            }
        }
    }

    /// 名称是否在主体作用域内（列表过滤与配额计数共用）。
    /// Whether a collection name is visible to this principal.
    pub fn may_see(&self, name: &str) -> bool {
        scope_matches(&self.collections, name)
    }

    /// 新建 `target` 是否会突破 max_collections（已存在则不算新建）。
    /// 并发创建下的 check-then-create 竞态可略微超限（v1 接受并文档化）。
    /// EN: would creating `target` exceed the quota? Existing names never count.
    pub fn collection_quota_exceeded(&self, existing: &[String], target: &str) -> bool {
        scope_quota_exceeded(self.max_collections, &self.collections, existing, target)
    }

    /// 写速率许可（未配置 = 放行）。
    /// EN: write-rate admission.
    pub fn try_acquire_write(&self) -> bool {
        match &self.write_bucket {
            None => true,
            Some(bucket) => {
                let mut b = bucket.lock().unwrap_or_else(|e| e.into_inner());
                b.try_take()
            }
        }
    }

    /// 在途许可（max_inflight=0 = 不限）。返回的 permit 必须持有到响应完成。
    /// EN: in-flight permit held until the response completes.
    pub fn try_acquire_inflight(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        match &self.inflight {
            None => None,
            Some(sem) => sem.clone().try_acquire_owned().ok(),
        }
    }

    pub fn has_inflight_limit(&self) -> bool {
        self.inflight.is_some()
    }
}

/// 中间件状态：策略 + 可选的 collection 目录（集合配额需要知道现有名字）。
/// Middleware state: policy plus an optional catalog for collection quotas.
#[derive(Clone)]
pub struct AuthzState {
    pub policy: Arc<AuthzPolicy>,
    pub collections: Option<Arc<crate::collection::CollectionManager>>,
}

impl AuthzState {
    pub fn new(
        policy: Arc<AuthzPolicy>,
        collections: Option<Arc<crate::collection::CollectionManager>>,
    ) -> Self {
        Self {
            policy,
            collections,
        }
    }
}

/// 配置式主体策略（不可变，热加载时整体替换）。
/// Immutable policy; hot reload swaps the whole Arc.
#[derive(Debug, Clone, Default)]
pub struct AuthzPolicy {
    principals: Arc<Vec<Principal>>,
}

impl AuthzPolicy {
    /// 从配置构建：legacy `auth_token` 作为全权 admin，`[[auth_principals]]`
    /// 追加主体。校验失败返回 Err（启动期 fail-fast）。
    /// EN: build from config; the legacy token becomes an unscoped admin.
    pub fn from_config(
        legacy_token: Option<&str>,
        configs: &[PrincipalConfig],
    ) -> Result<Self, String> {
        Self::from_config_with_sentinel(legacy_token, configs, None)
    }

    /// `from_config` + sentinel 专用秘密（EE M1）。沿用既有构造函数的调用点
    /// 不受影响——没有 sentinel token 时行为逐字节不变。
    /// EN: `from_config` plus the sentinel secret (EE M1); existing callers are
    /// unaffected — without a sentinel token behavior is unchanged.
    pub fn from_config_with_sentinel(
        legacy_token: Option<&str>,
        configs: &[PrincipalConfig],
        sentinel_token: Option<&str>,
    ) -> Result<Self, String> {
        let mut principals: Vec<Principal> = Vec::new();
        if let Some(token) = legacy_token {
            if token.is_empty() {
                return Err("auth_token is empty".to_string());
            }
            principals.push(build_principal(
                "legacy-admin".to_string(),
                Role::Admin,
                None,
                sha256_hex(token.as_bytes()),
                None,
                64,
                None,
            ));
        }
        for cfg in configs {
            let name = cfg.name.trim();
            if name.is_empty() {
                return Err("auth_principals: empty name".to_string());
            }
            if principals.iter().any(|p| p.name == name) {
                return Err(format!("auth_principals: duplicate name {name:?}"));
            }
            let role =
                Role::parse(&cfg.role).map_err(|e| format!("auth_principals[{name}]: {e}"))?;
            let token_hash = match (&cfg.token, &cfg.token_sha256) {
                (Some(token), None) => {
                    if token.is_empty() {
                        return Err(format!(
                            "auth_principals[{name}]: token is empty (use token_sha256 \
                             or remove the principal)"
                        ));
                    }
                    sha256_hex(token.as_bytes())
                }
                (None, Some(hash)) => {
                    let hash = hash.trim().to_ascii_lowercase();
                    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                        return Err(format!(
                            "auth_principals[{name}]: token_sha256 must be 64 hex chars"
                        ));
                    }
                    hash
                }
                _ => {
                    return Err(format!(
                        "auth_principals[{name}]: exactly one of token / token_sha256 \
                         is required"
                    ))
                }
            };
            if principals.iter().any(|p| p.token_hash == token_hash) {
                return Err(format!(
                    "auth_principals[{name}]: token duplicates another principal"
                ));
            }
            let collections = match &cfg.collections {
                None => None,
                Some(list) => {
                    if list.is_empty() {
                        return Err(format!(
                            "auth_principals[{name}]: collections is empty (omit it for all)"
                        ));
                    }
                    let mut seen = Vec::with_capacity(list.len());
                    for c in list {
                        let c = c.trim();
                        if c.is_empty() {
                            return Err(format!(
                                "auth_principals[{name}]: empty collection scope entry"
                            ));
                        }
                        if !seen.iter().any(|s: &String| s == c) {
                            seen.push(c.to_string());
                        }
                    }
                    Some(seen)
                }
            };
            // 配额校验：max_collections / writes_per_sec 的 0 视为非法
            //（要么省略表示不限，要么给正数）。max_inflight 是例外：0 表示
            // 不限，None 表示默认 64——两种含义都必须可表达。
            // EN: 0 is invalid for max_collections / writes_per_sec (omit for
            // unlimited), but max_inflight deliberately accepts 0 = unlimited.
            let max_collections = match cfg.max_collections {
                Some(0) => {
                    return Err(format!(
                        "auth_principals[{name}]: max_collections must be > 0 (omit for unlimited)"
                    ))
                }
                other => other,
            };
            let max_inflight = cfg.max_inflight.unwrap_or(64);
            let writes_per_sec = match cfg.writes_per_sec {
                Some(0) => {
                    return Err(format!(
                        "auth_principals[{name}]: writes_per_sec must be > 0 (omit for unlimited)"
                    ))
                }
                other => other,
            };
            principals.push(build_principal(
                name.to_string(),
                role,
                collections,
                token_hash,
                max_collections,
                max_inflight,
                writes_per_sec,
            ));
        }
        // Sentinel 专用秘密（EE M1）：独立主体，避免把控制面权限混进数据面
        // token；空白值 fail-closed（与 auth_token 同规）。
        // EN: the sentinel secret becomes its own principal so control-plane
        // authority never rides a data-plane token; a blank value is fatal.
        if let Some(token) = sentinel_token {
            if token.is_empty() {
                return Err("sentinel_token is empty (unset it to disable)".to_string());
            }
            if principals.iter().any(|p| p.name == "sentinel") {
                return Err(
                    "sentinel_token conflicts with an auth_principals entry named \
                     \"sentinel\"; rename the principal"
                        .to_string(),
                );
            }
            let token_hash = sha256_hex(token.as_bytes());
            if principals.iter().any(|p| p.token_hash == token_hash) {
                return Err(
                    "sentinel_token duplicates another principal's token; use a \
                     dedicated secret"
                        .to_string(),
                );
            }
            principals.push(build_principal(
                "sentinel".to_string(),
                Role::Sentinel,
                None,
                token_hash,
                None,
                64,
                None,
            ));
        }
        Ok(Self {
            principals: Arc::new(principals),
        })
    }

    /// 空策略 = 不启用认证（单机开发默认）。
    /// An empty policy means auth is disabled (single-node dev default).
    pub fn is_empty(&self) -> bool {
        self.principals.is_empty()
    }

    pub fn len(&self) -> usize {
        self.principals.len()
    }

    /// 常量时间解析 token → 主体。未知 token 返回 None。
    /// Resolve a token to a principal with constant-time comparison.
    pub fn resolve(&self, token: &str) -> Option<&Principal> {
        let hash = sha256_hex(token.as_bytes());
        self.principals
            .iter()
            .find(|p| constant_time_eq(p.token_hash.as_bytes(), hash.as_bytes()))
    }
}

/// HTTP 动作推导：管理面与高危运维写操作需要 Admin。
/// Map an HTTP request to an action; admin surfaces require Admin.
pub fn http_action(method: &Method, path: &str) -> Action {
    let m = method.as_str();
    // sentinel 控制面（EE M1）：独立动作，admin 隐含放行，sentinel 角色只此一项。
    // lease 的 GET 用于观察租约，也归控制面（不暴露给普通 reader）。
    // EN: sentinel control surface (EE M1) — admin implies it, the sentinel
    // role gets exactly this set; even the lease GET stays off the reader role.
    if path == "/v1/version" {
        return Action::Meta;
    }
    if matches!(
        path,
        "/v1/admin/lease"
            | "/v1/admin/demote"
            | "/v1/admin/promote"
            | "/v1/admin/replication/status"
    ) {
        return Action::Sentinel;
    }
    if path.starts_with("/v1/admin")
        || path.ends_with("/ha_level")
        || path.ends_with("/compact")
        || path == "/metrics"
        || matches!(
            path,
            "/v1/wal/checkpoint" | "/v1/wal/rotate" | "/v1/wal/snapshot" | "/v1/optimize"
        )
    {
        return Action::Admin;
    }
    if m == "GET" || m == "HEAD" || m == "OPTIONS" {
        return Action::Read;
    }
    // POST 检索端点是读语义；其余 POST 视为写。
    // EN: POST search endpoints are read semantics; all other POSTs count as writes.
    if path.ends_with("/search") || path == "/v1/hybrid/search" {
        Action::Read
    } else {
        Action::Write
    }
}

/// 路径中的 collection 名（`/v1/collections/{name}/...`）。
/// Extract the collection name from an HTTP path.
pub fn http_collection(path: &str) -> Option<&str> {
    let rest = path.strip_prefix("/v1/collections/")?;
    let name = rest.split('/').next().unwrap_or("");
    (!name.is_empty()).then_some(name)
}

/// 该 HTTP 请求是否可能新建 collection：显式 `PUT /v1/collections/{name}`
/// 与隐式 `POST /v1/collections/{name}/upsert`。
///
/// 检索/删除/改 HA 对未知名字不会创建，配额不得把它们当成新建——否则
/// `DELETE` 未知名字会先得到 429 而非 404（OCR 批次三十四）。
/// EN: explicit `PUT /v1/collections/{name}` and implicit upsert both create;
/// search/delete/ha_level never do, so quota must not count them as creations.
pub fn http_creates_collection(method: &Method, path: &str) -> bool {
    let Some(name) = http_collection(path) else {
        return false;
    };
    // collection 名之后的剩余路径：空前缀即 collection 根。尾斜杠 `/v1/collections/x/`
    // 不是建集合：axum 0.8（matchit）不做尾斜杠归一，路由器只注册了
    // `/v1/collections/{name}`，该请求本就 404；若把它算作新建，配额满时会在
    // 路由前先返回 429，与「429 不得替代 404」的既有纪律冲突（OCR #38）。
    // EN: remainder after the name; an empty rest is the collection root.
    // A trailing slash (`/v1/collections/x/`) is not a creation: axum 0.8
    // (matchit) does not normalize trailing slashes and the router only
    // registers `/v1/collections/{name}`, so such a request is a 404; counting
    // it as a creation would return 429 at quota before routing, violating the
    // "429 must not replace 404" discipline (OCR #38).
    let rest = &path["/v1/collections/".len() + name.len()..];
    (method == Method::PUT && rest.is_empty()) || (method == Method::POST && rest == "/upsert")
}

/// 全局数据资源（主索引/嵌入/管理面）：带作用域主体默认不可访问。
/// Global data resources (main index/embeddings/admin) for scope decisions.
pub fn http_global_data(path: &str) -> bool {
    path.starts_with("/v1/vectors")
        || path.starts_with("/v1/admin")
        || path == "/v1/hybrid/search"
        || path == "/v1/embeddings"
        || path == "/v1/optimize"
        || path == "/metrics"
        || path.starts_with("/v1/wal/")
        || path.starts_with("/v1/index/")
}

/// 已认证主体（写入请求/响应扩展：处理器做列表过滤与配额，审计取名字）。
/// Authenticated principal carried in request/response extensions.
#[derive(Debug, Clone)]
pub struct AuthenticatedPrincipal {
    pub name: String,
    /// 角色（whoami 端点展示；审计仍以主体名为准）。
    /// EN: role (shown by whoami; audit still keys on the principal name).
    pub role: Role,
    /// None = 全部；列表过滤与配额判定使用。
    /// EN: None = all; used by list filtering and quota checks.
    pub collections: Option<Vec<String>>,
    pub max_collections: Option<usize>,
}

impl AuthenticatedPrincipal {
    fn from_principal(p: &Principal) -> Self {
        Self {
            name: p.name.clone(),
            role: p.role,
            collections: p.collections.clone(),
            max_collections: p.max_collections,
        }
    }

    /// 名称是否可见（None / "*" 为全部）；与 [`Principal::may_see`] 同一实现。
    /// EN: Whether a name is visible (None / "*" = all); same impl as [`Principal::may_see`].
    pub fn may_see(&self, name: &str) -> bool {
        scope_matches(&self.collections, name)
    }

    /// 新建 `target` 是否超配额（已存在不算新建）；与 Principal 同一实现。
    /// EN: Whether creating `target` exceeds quota (existing is not new); same impl as Principal.
    pub fn collection_quota_exceeded(&self, existing: &[String], target: &str) -> bool {
        scope_quota_exceeded(self.max_collections, &self.collections, existing, target)
    }
}

/// 429 配额拒绝（带主体扩展供审计；Retry-After 仅对速率限制有意义，
/// 但在途/集合配额也可能很快恢复，统一给 1 秒）。
/// 429 quota rejection carrying the principal for auditing.
fn quota_rejected(principal: &Principal, reason: &'static str) -> Response {
    tracing::warn!(
        principal = %principal.name,
        role = principal.role.as_str(),
        reason,
        "rbac: quota exceeded"
    );
    let mut response = (StatusCode::TOO_MANY_REQUESTS, reason).into_response();
    response.headers_mut().insert(
        header::RETRY_AFTER,
        axum::http::HeaderValue::from_static("1"),
    );
    response
        .extensions_mut()
        .insert(AuthenticatedPrincipal::from_principal(principal));
    response
}

/// 认证 + 授权中间件（P4.2.1 认证 → P4.7.1 RBAC）。
/// Authentication + authorization middleware.
///
/// 401 = 未知/缺失 token（带 WWW-Authenticate）；403 = 已知主体越权。
/// EN: 401 for unknown/missing credentials; 403 for known-but-forbidden.
pub async fn require_auth(
    State(state): State<Arc<AuthzState>>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    if path == "/v1/health" || path == "/readyz" {
        return next.run(req).await;
    }
    let Some(principal) = extract_tokens(req.headers())
        .into_iter()
        .find_map(|t| state.policy.resolve(t))
    else {
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "unauthorized",
        )
            .into_response();
    };
    let action = http_action(req.method(), path);
    let collection = http_collection(path);
    if !principal.allows(action, collection, http_global_data(path)) {
        tracing::warn!(
            principal = %principal.name,
            role = principal.role.as_str(),
            method = %req.method(),
            path,
            "rbac: insufficient privilege"
        );
        // 403 时主体已知：把主体带进响应扩展，让审计记录
        // `principal:<name>` 而不是回退到 token 指纹（越权记录恰恰最需要
        // 人可读身份）。401 保持指纹——那里主体确实未知。
        // EN: attach the known principal to 403 responses for the audit actor.
        let mut response = (StatusCode::FORBIDDEN, "forbidden").into_response();
        response
            .extensions_mut()
            .insert(AuthenticatedPrincipal::from_principal(principal));
        return response;
    }
    // 写速率配额（每主体令牌桶）。
    // EN: Write-rate quota (per-principal token bucket).
    if action == Action::Write && !principal.try_acquire_write() {
        return quota_rejected(principal, "write rate limit exceeded");
    }
    // 集合配额：只约束「可能隐式新建」的端点（upsert），且只约束新建。
    // 删除/检索/改 HA 对未知名字不会创建；把它们按写动作计入配额会让
    // DELETE 未知名字得到 429 而非 404（OCR 批次三十四）。
    // EN: only endpoints that can create (upsert) are quota-gated — a delete
    // of an unknown name must stay a 404, not a 429.
    if action == Action::Write && http_creates_collection(req.method(), path) {
        if let (Some(name), Some(collections)) = (collection, &state.collections) {
            if collections.get(name).is_none() {
                let existing = collections.list();
                if principal.collection_quota_exceeded(&existing, name) {
                    return quota_rejected(principal, "collection quota exceeded");
                }
            }
        }
    }
    // 在途配额：许可必须在响应体写完前一直持有。缓冲体在 handler 返回时
    // 即已生成，但流式响应（如 collection 导出）返回的只是 body 流；把
    // permit 移进 body wrapper，保证慢客户端读流的全过程都计入在途。
    // EN: the permit must live until the response body is fully written. A
    // buffered body is complete when the handler returns, but streaming
    // responses (e.g. collection export) only return the body stream — moving
    // the permit into the body wrapper counts slow-client streaming too.
    let permit = if principal.has_inflight_limit() {
        match principal.try_acquire_inflight() {
            Some(p) => Some(p),
            None => return quota_rejected(principal, "too many in-flight requests"),
        }
    } else {
        None
    };
    let auth_ext = AuthenticatedPrincipal::from_principal(principal);
    // 请求扩展供处理器做列表过滤/集合配额；响应扩展供审计取主体名。
    // EN: Request extensions feed list filtering/collection quota; response
    // EN: extensions let audit read the principal name.
    let mut req = req;
    req.extensions_mut().insert(auth_ext.clone());
    let mut response = next.run(req).await;
    response.extensions_mut().insert(auth_ext);
    response.map(|body| body_holding_permit(body, permit))
}

/// 把在途许可挂到响应体上：body 耗尽或丢弃时许可才释放。
/// EN: attach the in-flight permit to the body; it is released only when the
/// body is exhausted or dropped.
fn body_holding_permit(body: Body, permit: Option<tokio::sync::OwnedSemaphorePermit>) -> Body {
    let Some(permit) = permit else {
        return body;
    };
    let stream = body.into_data_stream();
    let guarded =
        futures_util::stream::unfold((stream, permit), |(mut stream, permit)| async move {
            stream.next().await.map(|item| (item, (stream, permit)))
        });
    Body::from_stream(guarded)
}

/// 提取全部候选凭据（Bearer 与 X-Api-Key 都尝试，scheme 大小写不敏感，
/// token 不做 trim）。为什么保留两个候选（OCR 批次三十二）：只取 Bearer
/// 会让「过期的 Bearer + 有效的 X-Api-Key」被误 401——这是客户端可见的
/// 兼容性回退。
/// EN: return every candidate — a stale Bearer must not shadow a valid
/// X-Api-Key (client-visible regression).
pub fn extract_tokens(headers: &HeaderMap) -> Vec<&str> {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .and_then(|(scheme, token)| {
            scheme
                .eq_ignore_ascii_case("bearer")
                .then(|| token.trim_start_matches(' '))
        });
    let api_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
    bearer
        .into_iter()
        .chain(api_key)
        .filter(|t| !t.is_empty())
        .collect()
}

// ── pgwire 授权 ────────────────────────────────────────────────
// EN: pgwire authorization.

/// 去掉前导空白与 SQL 注释（块注释可反复出现，行注释到行尾）。
/// 本项目的指令形态就是「前导注释 + SELECT」，按首 token 分类会把主读路径
/// 误判为写（OCR 批次三十二 HIGH）。
/// EN: skip leading whitespace and SQL comments before extracting the keyword;
/// the documented directive form starts with a block comment.
fn strip_leading_sql_comments(sql: &str) -> &str {
    let mut s = sql.trim_start();
    loop {
        if let Some(rest) = s.strip_prefix("/*") {
            match rest.find("*/") {
                Some(end) => s = rest[end + 2..].trim_start(),
                None => return s,
            }
        } else if let Some(rest) = s.strip_prefix("--") {
            match rest.find('\n') {
                Some(end) => s = rest[end + 1..].trim_start(),
                None => return s,
            }
        } else {
            return s;
        }
    }
}

/// SQL 动作推导：只读关键字之外一律按写处理（fail-closed）。
/// Conservative SQL classification: anything but an explicit read is a write.
pub fn sql_action(sql: &str) -> Action {
    let s = strip_leading_sql_comments(sql);
    // 多语句（分号后仍有内容）一律按写：Simple Query 可能承载多条语句，
    // 只看首关键字会让「SELECT 1; DELETE …」绕过只读判定。
    // EN: multi-statement payloads are writes; otherwise a read prefix could
    // smuggle a write past the classifier.
    let body = s.trim_end().trim_end_matches(';');
    if body.contains(';') {
        return Action::Write;
    }
    let first = body
        .split(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .to_ascii_uppercase();
    match first.as_str() {
        "SELECT" | "SHOW" | "EXPLAIN" | "VALUES" | "TABLE" => Action::Read,
        // WITH 可能是只读 CTE 也可能是写 CTE：保守按写。
        // EN: WITH may be a read-only or a write CTE: conservatively treat it as a write.
        _ => Action::Write,
    }
}

/// pgwire 单条语句授权；返回 (SQLSTATE, message) 供 ErrorResponse。
/// Authorize one pgwire statement; returns (SQLSTATE, message) on denial.
pub fn authorize_sql(principal: &Principal, sql: &str) -> Result<(), (String, String)> {
    let action = sql_action(sql);
    let collection = houyi_query::extract_collection_directive(sql);
    let global_data = collection.is_none();
    if principal.allows(action, collection.as_deref(), global_data) {
        Ok(())
    } else {
        Err((
            "42501".to_string(),
            format!(
                "insufficient_privilege: principal '{}' (role {}) may not {} {}",
                principal.name,
                principal.role.as_str(),
                match action {
                    Action::Read => "read",
                    Action::Write => "write",
                    Action::Admin => "administer",
                    Action::Sentinel => "run sentinel control on",
                    Action::Meta => "inspect",
                },
                collection
                    .as_deref()
                    .map(|c| format!("collection '{c}'"))
                    .unwrap_or_else(|| "the main index".to_string())
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(name: &str, role: &str, token: &str, collections: Option<&[&str]>) -> PrincipalConfig {
        PrincipalConfig {
            name: name.to_string(),
            role: role.to_string(),
            token: Some(token.to_string()),
            token_sha256: None,
            collections: collections.map(|c| c.iter().map(|s| s.to_string()).collect()),
            max_collections: None,
            max_inflight: None,
            writes_per_sec: None,
        }
    }

    fn policy() -> AuthzPolicy {
        AuthzPolicy::from_config(
            Some("legacy-token"),
            &[
                cfg("ingest", "writer", "ingest-token", Some(&["docs"])),
                cfg("analytics", "reader", "analytics-token", None),
                cfg("ops", "admin", "ops-token", None),
            ],
        )
        .expect("valid policy")
    }

    #[test]
    fn resolves_tokens_and_legacy_admin() {
        let p = policy();
        assert_eq!(p.resolve("legacy-token").unwrap().name, "legacy-admin");
        assert_eq!(p.resolve("ingest-token").unwrap().role, Role::Writer);
        assert!(p.resolve("wrong").is_none());
        // 明文 token 不驻留（结构体里只有哈希），Debug 连哈希也打码。
        // EN: Plaintext tokens are not kept (only hashes in the struct); Debug masks even hashes.
        let debug = format!("{:?}", p);
        assert!(!debug.contains("ingest-token"));
        assert!(
            !debug.contains(&sha256_hex(b"ingest-token")),
            "token hash must be redacted in Debug"
        );
    }

    /// sentinel 角色只有控制面动作；数据面读写与 admin 全部拒绝。
    /// EN: the sentinel role has control-plane actions only.
    #[test]
    fn sentinel_principal_is_control_plane_only() {
        let p = AuthzPolicy::from_config_with_sentinel(
            Some("legacy-token"),
            &[cfg("ingest", "writer", "ingest-token", Some(&["docs"]))],
            Some("sentinel-secret"),
        )
        .expect("valid policy");
        let sentinel = p.resolve("sentinel-secret").expect("sentinel token");
        assert_eq!(sentinel.name, "sentinel");
        assert_eq!(sentinel.role, Role::Sentinel);
        assert!(sentinel.allows(Action::Sentinel, None, true));
        assert!(!sentinel.allows(Action::Read, Some("docs"), false));
        assert!(!sentinel.allows(Action::Write, Some("docs"), false));
        assert!(!sentinel.allows(Action::Admin, None, true));
        // admin 仍隐含控制面权限。
        // EN: Admin still implies control-plane access.
        assert!(p
            .resolve("legacy-token")
            .unwrap()
            .allows(Action::Sentinel, None, true));
        // 空 sentinel token / 与既有 token 重复 / 与同名主体冲突都 fast-fail。
        // EN: A blank sentinel token, a duplicate token, or a name clash with an
        // existing principal all fail fast.
        assert!(AuthzPolicy::from_config_with_sentinel(None, &[], Some("")).is_err());
        assert!(
            AuthzPolicy::from_config_with_sentinel(None, &[], Some("dup")).is_ok(),
            "unique secret is accepted"
        );
        let dup =
            AuthzPolicy::from_config_with_sentinel(Some("same-secret"), &[], Some("same-secret"));
        assert!(dup.unwrap_err().contains("duplicates"));
        let name_conflict = AuthzPolicy::from_config_with_sentinel(
            None,
            &[cfg("sentinel", "reader", "reader-token", None)],
            Some("sentinel-secret"),
        );
        assert!(name_conflict.unwrap_err().contains("named"));
    }

    #[test]
    fn rbac_role_and_scope_matrix() {
        let p = policy();
        let reader = p.resolve("analytics-token").unwrap();
        assert!(reader.allows(Action::Read, Some("docs"), false));
        assert!(!reader.allows(Action::Write, Some("docs"), false));
        assert!(!reader.allows(Action::Admin, None, true));

        let writer = p.resolve("ingest-token").unwrap();
        assert!(writer.allows(Action::Read, Some("docs"), false));
        assert!(writer.allows(Action::Write, Some("docs"), false));
        assert!(!writer.allows(Action::Write, Some("other"), false));
        assert!(
            !writer.allows(Action::Write, None, true),
            "scoped writer must not touch global data"
        );
        // 有作用域的读元数据（列出 collections）放行。
        // EN: Scoped principals may read metadata (list collections).
        assert!(writer.allows(Action::Read, None, false));

        let admin = p.resolve("ops-token").unwrap();
        assert!(admin.allows(Action::Admin, None, true));
        assert!(admin.allows(Action::Write, Some("anything"), false));
    }

    #[test]
    fn rejects_bad_principals() {
        let dup = AuthzPolicy::from_config(
            None,
            &[cfg("a", "reader", "t", None), cfg("b", "reader", "t", None)],
        );
        assert!(dup.unwrap_err().contains("duplicates"));
        let bad_role = AuthzPolicy::from_config(None, &[cfg("a", "root", "t", None)]);
        assert!(bad_role.unwrap_err().contains("invalid role"));
        let neither = AuthzPolicy::from_config(
            None,
            &[PrincipalConfig {
                name: "a".into(),
                role: "reader".into(),
                token: None,
                token_sha256: None,
                collections: None,
                max_collections: None,
                max_inflight: None,
                writes_per_sec: None,
            }],
        );
        assert!(neither.unwrap_err().contains("exactly one"));
        let bad_hash = AuthzPolicy::from_config(
            None,
            &[PrincipalConfig {
                name: "a".into(),
                role: "reader".into(),
                token: None,
                token_sha256: Some("zz".into()),
                collections: None,
                max_collections: None,
                max_inflight: None,
                writes_per_sec: None,
            }],
        );
        assert!(bad_hash.unwrap_err().contains("64 hex"));
    }

    #[test]
    fn http_action_and_scope_classification() {
        assert_eq!(http_action(&Method::GET, "/v1/collections"), Action::Read);
        assert_eq!(
            http_action(&Method::POST, "/v1/collections/docs/search"),
            Action::Read
        );
        assert_eq!(
            http_action(&Method::POST, "/v1/collections/docs/upsert"),
            Action::Write
        );
        assert_eq!(
            http_action(&Method::POST, "/v1/collections/docs/ha_level"),
            Action::Admin
        );
        assert_eq!(
            http_action(&Method::POST, "/v1/collections/docs/compact"),
            Action::Admin
        );
        assert_eq!(
            http_action(&Method::POST, "/v1/admin/promote"),
            Action::Sentinel
        );
        assert_eq!(
            http_action(&Method::POST, "/v1/admin/lease"),
            Action::Sentinel
        );
        assert_eq!(
            http_action(&Method::GET, "/v1/admin/replication/status"),
            Action::Sentinel
        );
        assert_eq!(
            http_action(&Method::POST, "/v1/admin/demote"),
            Action::Sentinel
        );
        // 查询分析端点仍是 admin，不下放给 sentinel 角色。
        // EN: Query-analysis endpoints stay admin-only; the sentinel role gets
        // nothing here.
        assert_eq!(
            http_action(&Method::GET, "/v1/admin/query-log"),
            Action::Admin
        );
        assert_eq!(http_action(&Method::GET, "/v1/version"), Action::Meta);
        // DELETE 落在 /v1/admin/*：管理动作，writer/reader 一律 403。
        // EN: DELETE under /v1/admin/* is an admin action (writer/reader get 403).
        assert_eq!(
            http_action(&Method::DELETE, "/v1/admin/collections/docs"),
            Action::Admin
        );
        assert_eq!(http_collection("/v1/collections/docs/search"), Some("docs"));
        assert_eq!(http_collection("/v1/collections"), None);
        assert!(http_global_data("/v1/vectors/upsert"));
        assert!(!http_global_data("/v1/collections/docs/search"));
        // PUT collection 根是写动作；只有显式建/隐式 upsert 计入新建配额。
        // EN: PUT on a collection root is a write; only explicit create and
        // implicit upsert count as creations for the quota.
        assert_eq!(
            http_action(&Method::PUT, "/v1/collections/docs"),
            Action::Write
        );
        assert!(http_creates_collection(
            &Method::PUT,
            "/v1/collections/docs"
        ));
        assert!(http_creates_collection(
            &Method::POST,
            "/v1/collections/docs/upsert"
        ));
        assert!(!http_creates_collection(
            &Method::GET,
            "/v1/collections/docs"
        ));
        assert!(!http_creates_collection(
            &Method::DELETE,
            "/v1/collections/docs"
        ));
        assert!(!http_creates_collection(
            &Method::PUT,
            "/v1/collections/docs/search"
        ));
        assert!(!http_creates_collection(
            &Method::POST,
            "/v1/collections/docs/ha_level"
        ));
        // 尾斜杠不是建集合：路由器不归一尾斜杠，该路径只会 404；
        // 配额不得把它提前变成 429（OCR #38）。
        // EN: A trailing slash is not a creation: the router does not normalize
        // it, so the path only ever 404s; quota must not turn it into 429 (OCR #38).
        assert!(!http_creates_collection(
            &Method::PUT,
            "/v1/collections/docs/"
        ));
    }

    /// 中间件端到端：角色 + collection 作用域 + 401/403 区分。
    /// EN: middleware end-to-end: roles, collection scopes, 401 vs 403.
    #[tokio::test]
    async fn rbac_middleware_allows_and_denies() {
        use axum::body::Body;
        use axum::routing::{delete, get, post};
        use axum::Router;
        use tower::ServiceExt;

        let auth_state = Arc::new(AuthzState::new(Arc::new(policy()), None));
        let app = Router::new()
            .route("/v1/collections/{name}/upsert", post(|| async { "ok" }))
            .route("/v1/collections/{name}/search", post(|| async { "ok" }))
            .route("/v1/collections", get(|| async { "list" }))
            .route("/v1/admin/collections/{name}", delete(|| async { "ok" }))
            .route("/v1/admin/audit/verify", get(|| async { "admin" }))
            .route("/v1/vectors/upsert", post(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&auth_state),
                require_auth,
            ));

        async fn call(app: &Router, method: &str, uri: &str, token: Option<&str>) -> StatusCode {
            let mut b = Request::builder().method(method).uri(uri);
            if let Some(t) = token {
                b = b.header("authorization", format!("Bearer {t}"));
            }
            app.clone()
                .oneshot(b.body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status()
        }

        // 未知/缺失凭据 → 401。
        // EN: Unknown/missing credentials -> 401.
        assert_eq!(
            call(&app, "GET", "/v1/collections", None).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            call(&app, "GET", "/v1/collections", Some("nope")).await,
            StatusCode::UNAUTHORIZED
        );
        // reader：可读、不可写、不可管理。
        // EN: reader: can read, cannot write or manage.
        assert_eq!(
            call(
                &app,
                "POST",
                "/v1/collections/docs/search",
                Some("analytics-token")
            )
            .await,
            StatusCode::OK
        );
        assert_eq!(
            call(
                &app,
                "POST",
                "/v1/collections/docs/upsert",
                Some("analytics-token")
            )
            .await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            call(
                &app,
                "GET",
                "/v1/admin/audit/verify",
                Some("analytics-token")
            )
            .await,
            StatusCode::FORBIDDEN
        );
        // writer：作用域内可写，作用域外与全局数据拒绝。
        // EN: writer: writes within scope; outside scope and global data are denied.
        assert_eq!(
            call(
                &app,
                "POST",
                "/v1/collections/docs/upsert",
                Some("ingest-token")
            )
            .await,
            StatusCode::OK
        );
        assert_eq!(
            call(
                &app,
                "POST",
                "/v1/collections/other/upsert",
                Some("ingest-token")
            )
            .await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            call(&app, "POST", "/v1/vectors/upsert", Some("ingest-token")).await,
            StatusCode::FORBIDDEN
        );
        // 元数据读放行（列出 collections）。
        // EN: Metadata reads are allowed (list collections).
        assert_eq!(
            call(&app, "GET", "/v1/collections", Some("ingest-token")).await,
            StatusCode::OK
        );
        // admin：全局数据与管理面均可。
        // EN: admin: global data and the management surface are allowed.
        assert_eq!(
            call(&app, "POST", "/v1/vectors/upsert", Some("ops-token")).await,
            StatusCode::OK
        );
        assert_eq!(
            call(&app, "GET", "/v1/admin/audit/verify", Some("ops-token")).await,
            StatusCode::OK
        );
        // 整库删除只有 admin 能做；writer/reader 均 403，admin 放行。
        // EN: whole-collection drop is admin-only; writer/reader are 403.
        assert_eq!(
            call(
                &app,
                "DELETE",
                "/v1/admin/collections/docs",
                Some("analytics-token")
            )
            .await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            call(
                &app,
                "DELETE",
                "/v1/admin/collections/docs",
                Some("ingest-token")
            )
            .await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            call(
                &app,
                "DELETE",
                "/v1/admin/collections/docs",
                Some("ops-token")
            )
            .await,
            StatusCode::OK
        );

        // 过期 Bearer 不得遮蔽有效的 X-Api-Key（兼容性回归）。
        // EN: An invalid Bearer must not shadow a valid X-Api-Key (compat regression).
        let req = Request::builder()
            .method("GET")
            .uri("/v1/collections")
            .header("authorization", "Bearer stale-token")
            .header("x-api-key", "analytics-token")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::OK
        );

        // 403 响应必须携带主体名（审计记录越权者身份）。
        // EN: 403 responses must carry the principal name (audit records the violator).
        let req = Request::builder()
            .method("POST")
            .uri("/v1/collections/docs/upsert")
            .header("authorization", "Bearer analytics-token")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let principal = resp
            .extensions()
            .get::<AuthenticatedPrincipal>()
            .expect("403 must carry the authenticated principal for auditing");
        assert_eq!(principal.name, "analytics");
    }

    /// 配额语义：集合上限只约束新建；在途与写速率按主体生效。
    /// EN: quota semantics — creation-only collection cap, per-principal
    /// inflight and write-rate limits.
    #[test]
    fn rbac_quotas() {
        let p = AuthzPolicy::from_config(
            None,
            &[PrincipalConfig {
                name: "limited".into(),
                role: "writer".into(),
                token: Some("limited-token".into()),
                token_sha256: None,
                collections: None,
                max_collections: Some(2),
                max_inflight: Some(1),
                writes_per_sec: Some(1),
            }],
        )
        .expect("valid quota config");
        let principal = p.resolve("limited-token").unwrap();
        let existing = vec!["a".to_string(), "b".to_string()];
        assert!(
            !principal.collection_quota_exceeded(&existing, "a"),
            "update never blocked"
        );
        assert!(
            principal.collection_quota_exceeded(&existing, "c"),
            "new collection over cap"
        );
        assert!(principal.try_acquire_write(), "first write within burst");
        assert!(!principal.try_acquire_write(), "burst exhausted");
        let permit = principal.try_acquire_inflight().expect("first in-flight");
        assert!(
            principal.try_acquire_inflight().is_none(),
            "second in-flight rejected"
        );
        drop(permit);
        assert!(
            principal.try_acquire_inflight().is_some(),
            "permit released"
        );

        // 0 值配置在启动期拒绝。
        // EN: Zero-valued config is rejected at startup.
        let zero = AuthzPolicy::from_config(
            None,
            &[PrincipalConfig {
                name: "z".into(),
                role: "reader".into(),
                token: Some("z".into()),
                token_sha256: None,
                collections: None,
                max_collections: Some(0),
                max_inflight: None,
                writes_per_sec: None,
            }],
        );
        assert!(zero.unwrap_err().contains("max_collections"));
    }

    /// 流式响应的在途许可：必须持有到 body 读完，不能随 handler 返回释放。
    /// EN: a streaming response holds its in-flight permit until the body is
    /// drained, not just until the handler returns.
    #[tokio::test]
    async fn inflight_permit_spans_streaming_bodies() {
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let policy = AuthzPolicy::from_config(
            None,
            &[PrincipalConfig {
                name: "streamer".into(),
                role: "reader".into(),
                token: Some("stream-token".into()),
                token_sha256: None,
                collections: None,
                max_collections: None,
                max_inflight: Some(1),
                writes_per_sec: None,
            }],
        )
        .expect("policy");
        let state = Arc::new(AuthzState::new(Arc::new(policy), None));
        let app = Router::new()
            .route(
                "/v1/collections/{name}/export",
                get(|| async {
                    let stream = futures_util::stream::once(async {
                        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from_static(b"row\n"))
                    });
                    Body::from_stream(stream)
                }),
            )
            .layer(axum::middleware::from_fn_with_state(state, require_auth));

        let request = || {
            Request::builder()
                .uri("/v1/collections/docs/export")
                .header(header::AUTHORIZATION, "Bearer stream-token")
                .body(Body::empty())
                .unwrap()
        };
        let first = app.clone().oneshot(request()).await.expect("first");
        assert_eq!(first.status(), StatusCode::OK);
        // body 未读：许可仍被占用。 / EN: the body is not drained yet.
        let second = app.clone().oneshot(request()).await.expect("second");
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        let _ = axum::body::to_bytes(first.into_body(), 1024)
            .await
            .expect("body");
        let third = app.clone().oneshot(request()).await.expect("third");
        assert_eq!(third.status(), StatusCode::OK);
    }

    /// 中间件集合配额：已有 1 个、上限 1 时，新建第二个返回 429，更新已有放行。
    /// EN: middleware enforces the collection cap on creates only.
    #[tokio::test]
    async fn rbac_collection_quota_middleware() {
        use axum::body::Body;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        let manager = Arc::new(crate::collection::CollectionManager::new(4, "hnsw"));
        manager.get_or_create("a", Some(4)).unwrap();
        let policy = Arc::new(
            AuthzPolicy::from_config(
                None,
                &[PrincipalConfig {
                    name: "w".into(),
                    role: "writer".into(),
                    token: Some("w-token".into()),
                    token_sha256: None,
                    collections: None,
                    max_collections: Some(1),
                    max_inflight: None,
                    writes_per_sec: None,
                }],
            )
            .unwrap(),
        );
        let state = Arc::new(AuthzState::new(policy, Some(Arc::clone(&manager))));
        let app = Router::new()
            .route("/v1/collections/{name}/upsert", post(|| async { "ok" }))
            .route("/v1/collections/{name}/delete", post(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(state, require_auth));

        async fn call(app: &Router, action: &str, name: &str) -> StatusCode {
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(format!("/v1/collections/{name}/{action}"))
                        .header("authorization", "Bearer w-token")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
                .status()
        }

        assert_eq!(
            call(&app, "upsert", "a").await,
            StatusCode::OK,
            "update existing"
        );
        assert_eq!(
            call(&app, "upsert", "b").await,
            StatusCode::TOO_MANY_REQUESTS,
            "creating beyond max_collections"
        );
        // 删除未知名字不会创建 collection：不得被集合配额改成 429
        //（OCR 批次三十四）。真实 handler 会返回 404，中间件必须放行到 handlers。
        // EN: Deleting an unknown name creates no collection, so quota must not turn
        // EN: it into 429 (OCR batch 34). The real handler returns 404; middleware
        // EN: must pass it through to the handlers.
        assert_eq!(
            call(&app, "delete", "b").await,
            StatusCode::OK,
            "delete must not be quota-gated as an implicit create"
        );
    }

    /// 中间件写速率：1 次/秒，连续两个写请求第二个 429。
    /// EN: write-rate limit rejects the second immediate write.
    #[tokio::test]
    async fn rbac_write_rate_middleware() {
        use axum::body::Body;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        let policy = Arc::new(
            AuthzPolicy::from_config(
                None,
                &[PrincipalConfig {
                    name: "r".into(),
                    role: "writer".into(),
                    token: Some("r-token".into()),
                    token_sha256: None,
                    collections: None,
                    max_collections: None,
                    max_inflight: None,
                    writes_per_sec: Some(1),
                }],
            )
            .unwrap(),
        );
        let state = Arc::new(AuthzState::new(policy, None));
        let app = Router::new()
            .route("/v1/collections/{name}/upsert", post(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(state, require_auth));

        async fn call(app: &Router) -> StatusCode {
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/collections/a/upsert")
                        .header("authorization", "Bearer r-token")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
                .status()
        }

        assert_eq!(call(&app).await, StatusCode::OK);
        assert_eq!(call(&app).await, StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn sql_authorization() {
        let p = policy();
        let writer = p.resolve("ingest-token").unwrap();
        // 有作用域主体：带 collection 指令的读/写放行，主索引读与元数据读拒绝。
        // EN: Scoped principal: reads/writes with a collection directive pass;
        // EN: main-index and metadata reads are denied.
        assert!(authorize_sql(writer, "SELECT * FROM t /* houyi:collection=docs */").is_ok());
        assert_eq!(
            authorize_sql(writer, "SELECT * FROM t").unwrap_err().0,
            "42501",
            "scoped principal must not read the main index"
        );
        assert_eq!(authorize_sql(writer, "SHOW TABLES").unwrap_err().0, "42501");
        let write_scoped = "INSERT INTO t VALUES (1) /* houyi:collection=docs */";
        assert!(authorize_sql(writer, write_scoped).is_ok());
        let write_other = "INSERT INTO t VALUES (1) /* houyi:collection=other */";
        assert_eq!(authorize_sql(writer, write_other).unwrap_err().0, "42501");
        // 无 collection 指令的写 = 主索引：带作用域主体拒绝。
        // EN: A write without a collection directive targets the main index;
        // EN: scoped principals are denied.
        assert_eq!(
            authorize_sql(writer, "INSERT INTO t VALUES (1)")
                .unwrap_err()
                .0,
            "42501"
        );
        let reader = p.resolve("analytics-token").unwrap();
        assert!(authorize_sql(reader, "SELECT 1").is_ok());
        // 本项目的规范形态是「前导块注释 + SELECT」：必须仍判定为只读。
        // EN: The documented shape is "leading block comment + SELECT": must still classify as
        // EN: read.
        assert!(authorize_sql(reader, "/* houyi:collection=docs */ SELECT * FROM t").is_ok());
        assert!(authorize_sql(reader, "/* a */ -- line\nSELECT 1").is_ok());
        // 多语句不得借只读前缀夹带写。
        // EN: Multi-statement SQL must not smuggle a write behind a read-only prefix.
        assert_eq!(
            authorize_sql(reader, "SELECT 1; DELETE FROM t")
                .unwrap_err()
                .0,
            "42501"
        );
        assert_eq!(
            authorize_sql(reader, "DELETE FROM t").unwrap_err().0,
            "42501"
        );
    }
}
