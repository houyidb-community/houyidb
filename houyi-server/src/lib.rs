//! HouyiDB server: HTTP API + pg_wire protocol handling + WAL coordination.
//! 服务子系统：HTTP API + pg_wire 协议处理 + WAL 协调。

pub mod audit;
pub mod breaker;
pub mod cluster;
pub mod collection;
pub mod collection_lifecycle;
pub mod http_handler;
pub mod index_backend;
pub mod mem;
pub mod openai_gateway;
pub mod pgwire_handler;
pub mod query_log;
pub mod rbac;
pub mod recovery;
pub mod replication_apply;
// P4.5.0 恢复状态机属性测试（proptest + proptest-state-machine），仅测试构建。
// EN: P4.5.0 recovery state-machine property tests (proptest + proptest-state-machine),
// test builds only.
#[cfg(test)]
mod recovery_proptests;
mod scalar_json;
pub mod tls;
pub mod types;
