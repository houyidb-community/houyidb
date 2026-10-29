# Runbook: Replication Lag / Slow Follower Kicked

> [中文](replication-lag.md)

**Alert**: `houyidb_replication_lag_entries > 10000` for 10 minutes;
or `increase(houyidb_replication_slow_follower_kicks_total[10m]) > 0`

## Symptoms and Impact

- The follower's received LSN lags behind the leader; semi_sync/full_sync writes may degrade
  to async due to wait timeouts (`houyidb_replication_sync_timeouts_total` rises).
- When the follower queue fills up, the follower is kicked (protecting the WAL write thread and other followers), then automatically reconnects and
  catches up; data is not necessarily lost, but the RPO window widens.

## First 5 Minutes

```bash
curl -fsS http://<leader>:17000/metrics | grep -E \
  'replication_(lag_entries|sync_timeouts_total|slow_follower_kicks_total|followers_active|shipped_batches_total)'
curl -fsS http://<follower>:17000/metrics | grep -E \
  'replication_(recv_lsn|durable_lsn|apply_failures_total|reconnects_total|fsync_total)'
# Network and disk
ping -c 3 <follower>; ssh <follower> 'df -h /data; iostat -x 1 3'
```

## Handling

1. First determine whether the cause is **bandwidth/disk** or **apply blocking**:
   - The follower's `apply_failures_total` / `apply_retries_total` increases → switch to
     [wal-poison.en.md](wal-poison.en.md) to check apply errors and disk.
   - The follower's disk I/O is saturated or `df` is nearly full → expand capacity or clean up archives (do not delete the active WAL).
   - Network packet loss/latency → fix the network; do not blindly increase the queue depth `replication_queue_batches`
     to catch up (it strains the WAL write thread's memory).
2. **Do not** perform writes on the follower or do a manual promote to "save effort".
3. Being kicked is a protection mechanism: after the follower recovers, it will catch up automatically; if it is repeatedly kicked over a long period,
   reduce the write rate or upgrade the follower configuration.

## Recovery Verification

- `lag_entries` keeps decreasing and reaches zero;
- `slow_follower_kicks_total` stops increasing;
- `sync_timeouts_total` stops increasing and semi/full_sync write latency recovers.

## Escalation

Lag exceeding 30 minutes or an unacceptable RPO risk → escalate to the DBA and evaluate a temporary downgrade (async)
plus a backfill strategy.
