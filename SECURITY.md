# HouyiDB 安全政策（Security Policy）

> [English](SECURITY.en.md)
>
> 本文为中文翻译，法律效力以英文原文（SECURITY.en.md）为准。

## 支持的版本（Supported versions）

HouyiDB 目前处于 beta 阶段（`0.2.x`）。安全修复面向 `main` 上的最新发布标签提供；较旧的 beta 标签可能不会收到补丁。

## 报告漏洞（Reporting a vulnerability）

**请勿为安全漏洞提交公开 issue。**

请使用 GitHub 的私密漏洞报告（private vulnerability reporting）：

1. 打开 <https://github.com/houyidb-community/houyidb/security/advisories>
2. 点击 **Report a vulnerability**
3. 请包含：受影响的版本/标签、部署模式（standalone/leader/follower）、复现步骤、影响，以及任何建议的修复方案。

如果该 advisory 表单不可用，请先通过仓库的私密渠道联系维护者，然后再公开披露。

## 响应时限（What to expect）

| 阶段 | 目标时限 |
|-------|--------|
| 确认收到报告 | 72 小时 |
| 初步分诊 / 严重性定级 | 7 天 |
| 修复或缓解方案 | 30 天（严重问题：尽可能快） |
| 协调披露（coordinated disclosure） | 修复后 90 天，以双方约定为准 |

除非你希望匿名，我们会在 advisory 中致谢报告者。

## 范围（Scope）

属于范围：

- 本仓库中的 Rust 服务器与库；
- 认证/授权（RBAC、令牌处理、pgwire/TLS）；
- 静态加密（at-rest encryption：WAL v3、快照 v2、keyring）与审计哈希链（audit hash chain）；
- 复制协议（replication protocol，v5）、promote/failover 逻辑、WAL 恢复；
- 通过畸形的 pgwire/HTTP 输入实施拒绝服务攻击；
- 已发布二进制中的供应链（supply-chain）问题。

不属于范围：

- 需要攻陷主机/root 权限的漏洞（威胁模型见 `houyi-persistence/src/encryption.rs`）；
- 未绕过任何安全控制的错误配置（例如有意禁用认证）；
- 在 HouyiDB 中没有可达路径的第三方依赖（请报告给上游，但我们仍希望知情）；
- 社会工程学（social engineering）。

## 安全港（Safe harbor）

对于尊重隐私、不降低服务质量并遵循协调披露原则的善意研究，我们不会采取法律行动。请勿对你并不拥有或未获授权测试的系统进行测试。

## 安全特性与验证（Security features and verification）

- 安全配置与部署边界见[部署指南](docs/DEPLOYMENT_GUIDE.md)；故障处理见[告警 Runbooks](docs/runbooks/README.md)。
- 在本地运行 `./scripts/security_audit.sh` 可执行 cargo audit/deny/SBOM。
- 审计链验证：`GET /v1/admin/audit/verify`（需要 Admin 令牌）。
