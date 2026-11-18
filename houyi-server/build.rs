// 构建时嵌入 git 短 SHA（`/version` 的 build_sha），让运行中的实例能自证版本。
// 源码发布包没有 `.git` 时回退为 `unknown`；build script 失败不得阻断构建。
//
// 外部传入的 `HOUYIDB_GIT_SHA` 只接受 7–40 位 ASCII 十六进制字符，并归一化为
// 8 位小写短 SHA：cargo 逐行解析 build script 的 stdout，未校验的值若含换行
// 可以注入额外 `cargo:` 指令；同时这也保持「短 SHA」文档契约与 git 路径一致。
// 非法值不会被采用（只打印不含该值的 warning，避免二次注入），回退 git 探测。
//
// EN: Embed the short git SHA at build time for `/version`'s build_sha so a
// running instance can identify its exact source. Source tarballs without
// `.git` fall back to `unknown`; a build-script failure must not block builds.
//
// An externally supplied `HOUYIDB_GIT_SHA` is accepted only if it is 7–40 ASCII
// hex characters, and is normalized to an 8-character lowercase short SHA:
// cargo parses build-script stdout line by line, so an unvalidated value with
// an embedded newline could inject extra `cargo:` directives; this also keeps
// the documented "short SHA" contract aligned with the git path. Invalid
// values are ignored (a warning without the value avoids re-injection) and the
// git probe is used instead.

use std::process::Command;

/// 归一化 SHA（外部传入或 git 探测）：合法时取前 8 位小写；非法返回 None。
/// EN: Normalize a SHA (external or git-probed): first 8 lowercase chars when
/// valid, None otherwise.
fn normalize_short_sha(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let is_hex =
        (7..=40).contains(&trimmed.len()) && trimmed.chars().all(|c| c.is_ascii_hexdigit());
    is_hex.then(|| trimmed[..trimmed.len().min(8)].to_ascii_lowercase())
}

fn main() {
    // 外部显式提供的值优先：Docker 构建上下文排除了 `.git`，CI 需通过
    // build-arg/ENV 传入提交 SHA，否则镜像内永远是 unknown。
    // EN: An externally supplied value wins: Docker build contexts exclude
    // `.git`, so CI must pass the commit SHA via build-arg/ENV; otherwise every
    // image would report `unknown`.
    println!("cargo:rerun-if-env-changed=HOUYIDB_GIT_SHA");
    if let Ok(raw) = std::env::var("HOUYIDB_GIT_SHA") {
        if let Some(sha) = normalize_short_sha(&raw) {
            println!("cargo:rustc-env=HOUYIDB_GIT_SHA={sha}");
            return;
        }
        if !raw.trim().is_empty() {
            println!("cargo:warning=ignoring HOUYIDB_GIT_SHA: expected 7-40 ASCII hex characters");
        }
    }

    // HEAD、分支 ref 与 packed-refs 变化时重跑；打包源码没有这些路径，
    // cargo 会忽略。packed-refs 覆盖 `git gc`/`git pack-refs` 后的分支尖端，
    // 只盯 loose refs 会让嵌入的 SHA 变陈旧。
    // EN: Rerun when HEAD, branch refs, or packed-refs change; cargo ignores
    // missing paths in source packages. packed-refs covers branch tips after
    // `git gc`/`git pack-refs`; watching loose refs alone would go stale.
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs");
    println!("cargo:rerun-if-changed=../.git/packed-refs");

    // git 输出也走同一归一化：`--short=8` 在仓库内 8 位有歧义时会返回更长
    // 的缩写，且 lossy 解码可能带入异常字节；统一为 7–40 位十六进制后取
    // 8 位，两条构建路径的 build_sha 形态才真正对称。
    // EN: Route git output through the same normalizer: `--short=8` returns a
    // longer abbreviation when 8 chars are ambiguous in the repository, and
    // lossy decoding may carry odd bytes; normalizing to 7-40 hex chars and
    // taking 8 keeps both build paths symmetric.
    let sha = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                normalize_short_sha(&s)
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=HOUYIDB_GIT_SHA={sha}");
}
