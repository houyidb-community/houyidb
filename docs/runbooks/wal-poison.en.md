# Runbook: WAL Poisoned (Writes Rejected After a Write Failure)

> [中文](wal-poison.md)

**Trigger**: log contains `wal poisoned by write failure; all further writes rejected`
(a Loki rule can be configured; on the Prometheus side it manifests as sustained 5xx on write requests and stalled `wal` metrics)

## Symptoms and Impact

- After the first fsync/write failure, the WAL handle enters a poisoned state: all subsequent writes fail immediately,
  avoiding further appends to a torn log that would create unrecoverable holes.
- Read requests remain available; replication followers may stall.

## First 5 Minutes

```bash
# 1) Find the root-cause error (the first poison log carries op and io error)
journalctl -u houyidb --since '30 min ago' | grep -A2 'wal poisoned'
# 2) Disk/filesystem
df -h /data; df -i /data; dmesg | tail -30
# 3) Permissions and read-only mounts
mount | grep ' /data '; ls -l /data/*.wal
# 4) Replication status (check whether the follower can still catch up)
curl -fsS http://127.0.0.1:17000/metrics | grep houyidb_replication_
```

## Handling

1. **Do not**: manually truncate/edit the active WAL, delete `.wal` files, or restart while skipping fsync.
2. Fix the root cause: expand the disk/clean up safely deletable archives, fix read-only mounts/permissions, handle EIO.
3. After the root cause is eliminated, **restart the node** (the poisoned state is cleared only on reopen). Before restarting, copy the log file for the
   damaged window to the incident directory for reference.
4. If it is poisoned again immediately after restart: stop trying and switch to [replication-divergence.en.md](replication-divergence.en.md)
   to evaluate rebuilding from a replica; do not let both nodes run in a degraded state.

## Recovery Verification

- `/readyz` passes after restart;
- A write probe (one collection upsert) succeeds;
- The cold-start log has no anomalies other than `truncating torn tail`;
- The follower's `lag_entries` reaches zero.

## Escalation

Any suspected data loss (record count after restart is lower than the write-side ledger) → P0; preserve the WAL and snapshots and
escalate to the DBA/storage lead.
