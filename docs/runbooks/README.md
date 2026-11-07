# HouyiDB 告警处置手册（Runbook）

> [English](README.en.md)

> 面向 on-call 工程师的可执行操作卡。**不是**设计文档：每条都必须能在
> 5 分钟内照做，并明确“禁止做什么”。告警规则见
> `deploy/prometheus/houyidb-alerts.yml`（`runbook_url` 指向本目录）。
> EN: executable on-call cards. The alert file links here via `runbook_url`.

## 通用约定

- **先保现场，再做恢复**：不要删除/截断 WAL、快照、审计段；先把文件复制到
  安全目录，再按手册处置。
- **只读优先**：能通过 promote 切换读写的场景，不要重启两个节点。
- **禁止绕过安全控制**：缺密钥/审计链断时，禁止临时关闭加密或审计启动。
- 每次事故保留一份时间线（触发指标、命令、恢复时间），供 RTO/MTTR 统计与
  复盘。

## 索引

| 告警 / 事件 | 严重度 | 手册 |
|-------------|--------|------|
| 复制日志分歧（diverged） / follower_ahead | Critical | [replication-divergence.md](replication-divergence.md) |
| 复制滞后、慢 follower 被踢 | Warning | [replication-lag.md](replication-lag.md) |
| follower 落后于保留范围（BEHIND_RETAINED） | Critical | [collections-retention-resync.md](collections-retention-resync.md) |
| WAL 中毒（写入失败后拒绝写） | Critical | [wal-poison.md](wal-poison.md) |
| 加密快照/WAL 缺密钥导致启动失败 | Critical | [encryption-key-missing.md](encryption-key-missing.md) |
| 审计链校验失败 / 审计写错误 | Critical | [audit-chain-broken.md](audit-chain-broken.md) |
| 配额拒绝或认证失败风暴 | Warning | [quota-and-auth-storm.md](quota-and-auth-storm.md) |
| 内存水位 / 磁盘不足 / 快照被拒 | Warning | [resource-watermarks.md](resource-watermarks.md) |

## 通用诊断入口

```bash
# 指标（直连本机 benchmark HTTP 端口）
curl -fsS http://127.0.0.1:17000/metrics | grep -E 'houyidb_(replication|wal|audit|pgwire_quota|memory)'

# 就绪状态
curl -fsS http://127.0.0.1:17000/readyz

# 服务日志中的关键模式
#   wal poisoned by write failure
#   replication diverged / follower_ahead
#   AES-GCM authentication failed
#   missing encryption key
#   audit chain broken
```
