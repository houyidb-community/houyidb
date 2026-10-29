# Contributing to HouyiDB

> [中文](CONTRIBUTING.md)

Thank you for your interest in contributing. HouyiDB is an open-core project: this repository (the community edition) is licensed under Apache-2.0, and enterprise capabilities are developed in a separate private repository. Before submitting a contribution, please read
[`LICENSING.en.md`](LICENSING.en.md).

## 1. Contributor License Agreement (CLA)

All external contributions must first sign [`CLA.en.md`](CLA.en.md) (individuals; employed contributors require company authorization).
Current process: state the following in the PR description

> I have read the CLA Document and I hereby sign the CLA.

Maintainers will record the signature. PRs without a signed CLA will not be merged. The CLA allows the project to use the same contribution under both Apache-2.0 and a commercial license, without transferring copyright ownership.

## 2. Development Environment

```bash
# 稳定版 Rust（仓库 rust-version 为最低要求）
cargo build --all
cargo test --all
```

Local gates (same criteria as CI; all must be green before submitting):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all
./scripts/community_boundary_check.sh
./scripts/security_audit.sh        # requires cargo-audit/deny/cyclonedx
```

The development repository additionally runs replication, concurrency, turmoil,
and chaos drills for WAL/replication/HA changes. The public repository keeps
unit/integration tests and the reproducible benchmark adapter.

## 3. Code and Commit Conventions

- Technical discussions and comments use Simplified Chinese; public APIs and key algorithms are bilingual (Chinese and English).
- Comments explain "why", not restate "what was done".
- Commit messages: `<type>: <description>` (feat/fix/refactor/docs/test/chore/perf/ci),
  stating the motivation and impact scope; performance changes must include measurement data.
- One PR focuses on one thing; do not reorder unrelated code along the way.
- Historical bugs must be covered by regression tests; performance gates and correctness gates must not be bypassed.
- For security vulnerabilities, do **not** open a public issue; see [`SECURITY.en.md`](SECURITY.en.md).

## 4. Boundary Discipline (Important)

- The public repository must not contain enterprise-edition code, private dependencies, private registries, or cross-repository path dependencies;
  CI's `community_boundary_check.sh` will reject them.
- Do not pre-build empty abstractions for a "future commercial edition"; add extension points only for real requirements.
- Do not remove or weaken existing security controls (RBAC, audit, encryption, quotas) to simplify implementations.

## 5. Review and Merge

1. PRs must pass all CI (test/clippy/fmt/security/boundary).
2. At least one maintainer approval; core persistence/replication/security changes require two.
3. Merge method: squash merge (maintaining a linear history and clear motivation).
4. Releases are tagged by maintainers; the community edition and enterprise edition are released separately.

## 6. Code of Conduct

By participating in this project, you agree to [`CODE_OF_CONDUCT.en.md`](CODE_OF_CONDUCT.en.md).
