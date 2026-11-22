//! HouyiDB 官方 Rust 门面：重导出 [`houyidb_client`] 的全部公开 API。
//! EN: Official HouyiDB Rust facade re-exporting the full public API of
//! [`houyidb_client`].
//!
//! 需要更细的依赖/版本控制时，可直接依赖 `houyidb-client`。
//! EN: Depend on `houyidb-client` directly for finer dependency/version control.
#![forbid(unsafe_code)]

pub use houyidb_client::*;
