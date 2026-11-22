# houyidb

HouyiDB 官方 Rust 门面：完整重导出 [`houyidb-client`](https://crates.io/crates/houyidb-client)
的公开 API。
EN: Official HouyiDB Rust facade re-exporting the full public API of
[`houyidb-client`](https://crates.io/crates/houyidb-client).

## 安装 / Install

```toml
[dependencies]
houyidb = "0.2.0-beta.9"
```

`cargo add houyidb` 会选中当前预发布版本；也可以显式指定
`cargo add houyidb@0.2.0-beta.9`。
EN: `cargo add houyidb` selects the current pre-release; you can also pin it
explicitly with `cargo add houyidb@0.2.0-beta.9`.

用法与 `houyidb-client` 完全一致，示例见
[`houyidb-client` README](https://crates.io/crates/houyidb-client)。
EN: The API is identical to `houyidb-client`; see its README for examples.
