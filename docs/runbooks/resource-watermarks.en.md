# Runbook: Memory Watermark / Low Disk / Snapshot Rejected

> [中文](resource-watermarks.md)

**Alerts**:
- `increase(houyidb_memory_watermark_rejected_total[5m]) > 0`
- `node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes < 0.1`
- `increase(houyidb_graph_snapshot_rejects_total[1h]) > 0`
- `increase(houyidb_replication_apply_failures_total[5m]) > 0`

## Symptoms and Impact

- Memory watermark: writes are rejected (HTTP 503 / pgwire 53200), while reads remain available.
- Low disk: WAL/snapshot writes fail → may trigger WAL poisoning (switch to the wal-poison runbook).
- Snapshot rejected: cold start falls back to an earlier snapshot or full WAL replay; startup slows down but data is usually not lost.

## First 5 Minutes

```bash
df -h /data; df -i /data
curl -fsS http://127.0.0.1:17000/metrics | grep -E \
  'process_rss_bytes|memory_watermark|graph_snapshot_rejects|wal_active_bytes'
du -sh /data/* 2>/dev/null | sort -h | tail -10
# Reason for snapshot rejection
journalctl -u houyidb --since '2 hours ago' | grep -E 'graph snapshot|snapshot discarded'
```

## Handling

1. **Memory watermark**: confirm whether RSS truly exceeds the limit (`houyidb_process_rss_bytes`);
   temporarily shed load/scale out; check for abnormally large queries or collection bloat.
   `max_memory_bytes=0` disables throttling — production should set a reasonable value.
2. **Disk**: clean up content that is **safe to delete** (old backups/archive copies/already-migrated plaintext segments);
   never delete the active WAL or the most recent snapshot directly; prefer expanding capacity.
3. **Snapshot rejected**: read the CRC/length/key reason from the logs:
   - CRC/truncation → file corruption; keep the corrupted sample and let the next snapshot overwrite it;
   - Missing key → switch to [encryption-key-missing.en.md](encryption-key-missing.en.md);
   - Recovery failure → check graph payload compatibility; perform a full replay if necessary.
4. If writes have already failed and poisoning occurred, switch to [wal-poison.en.md](wal-poison.en.md).

## Recovery Verification

- The watermark rejection count stops increasing and a write probe succeeds;
- `df` available space > 20%;
- In the next periodic snapshot round, `writes_total` increases and `rejects_total` does not;
- The follower catches up (if affected).

## Escalation

Disk cannot be expanded within 30 minutes or the WAL is already poisoned → escalate to the DBA/infrastructure team; any deletion
in the data directory requires two-person confirmation.
