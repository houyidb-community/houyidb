# Runbook：落后保留范围的全量重同步

> [English](collections-retention-resync.en.md)

**告警**：follower 日志出现 `BEHIND RETAINED`；或
`increase(houyidb_replication_behind_retained_rejections_total[10m]) > 0`。

## 症状与影响

- follower 所需的历史已被 leader 的段保留删除（默认 16 段 × 64 MiB），
  复制无法从断点续传；follower 保持只读并长退避（60s 一次），**不会自动清库**。
- leader 的写路径不受影响；该 follower 的 RPO 持续扩大，直到人工重同步。

## 前 5 分钟

```bash
# leader：确认保留 floor 与 manifest
curl -fsS http://<leader>:17000/v1/wal/status | \
  python3 -c 'import json,sys; c=json.load(sys.stdin)["collections"]; print(json.dumps(c, indent=2))'
curl -fsS http://<leader>:17000/metrics | grep -E \
  'collections_(snapshot_safe_checkpoint_lsn|wal_first_available_lsn|segments_deleted_total)'

# follower：确认是 BEHIND_RETAINED 而不是网络/认证/分叉
curl -fsS http://<follower>:17000/metrics | grep -E \
  'replication_(behind_retained_rejections_total|leader_first_available_lsn|auth_rejections_total|diverged_rejections_total)'
```

- `auth/diverged` 增长 → 先按对应 runbook 处理，不要重同步。
- 只有 `behind_retained` 增长才是本 runbook 的适用范围。

## 重同步（建立新基线）

1. **停 follower**（保留其日志备查）。
2. leader 侧提交完整快照并轮转，同时用租约阻止保留删除：
   ```bash
   curl -fsS -X POST http://<leader>:17000/v1/admin/collections/compact \
     -H 'content-type: application/json' \
     -d '{"rotate":true,"hold_retention_secs":3600}'
   ```
   应答必须 `manifest_committed=true`，记录 `manifest_seq` / `safe_checkpoint_lsn`。
3. 把 leader 的以下文件复制到 follower 数据目录（覆盖前先清空 follower 的旧基线）：
   - `snapshots/`（manifest + 本轮生成的全部快照文件）；
   - collections WAL 活动段与 `*.wal.archived.*` 归档段；
   - `<collections-wal>.next_lsn`；
   - 删除 follower 的 `replication.follower_id`（新身份 = 明确的重同步语义）。
   同机/已挂载路径可直接运行：
   ```bash
   ./scripts/collections_resync.sh \
     --leader-http http://<leader>:17000 \
     --leader-wal /data/leader/houyidb.collections.wal \
     --follower-wal /data/follower/houyidb.collections.wal \
     --hold-secs 3600 --yes
   ```
4. 以 `--role follower` 启动 follower，等待握手。校验：
   ```bash
   curl -fsS http://<follower>:17000/v1/wal/status
   curl -fsS http://<follower>:17000/metrics | grep -E \
     'replication_(lag_entries|behind_retained_rejections_total|leader_first_available_lsn)'
   ```
   `lag_entries` 归零；`behind_retained` 不再增长。
5. leader 的租约到期前保留会继续；重同步完成后可等待自然到期，或再次调用
   compact 并传 `hold_retention_secs:0` 显式清除租约。

## 风险与边界

- **RPO**：重同步期间该 follower 不提供服务；新基线建立后，从快照点起的写入
  才进入其复制历史。停写窗口（LB 静默写）可把 RPO 压到 0。
- **加密**：follower 必须持有与 leader 相同的 keyring，否则启动时解密快照/WAL
  失败（fail-closed）。
- **主索引不复制**：benchmark 主索引（`/v1/vectors/*`）不在 collections 复制
  范围内；本 runbook 不处理它。
- **不要**用删除 leader 段、放宽 `collections_wal_max_segments` 的方式“修复”；
  那只会扩大下一轮落后窗口。
