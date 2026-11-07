# Runbook: full resync when a follower falls behind retention

> [中文](collections-retention-resync.md)

**Alert**: the follower logs `BEHIND RETAINED`; or
`increase(houyidb_replication_behind_retained_rejections_total[10m]) > 0`.

## Symptoms and impact

- History the follower needs has already been removed by leader segment
  retention (default 16 segments × 64 MiB); replication cannot resume from the
  follower's position. The follower stays read-only and backs off (60s), and
  **never wipes itself**.
- The leader's write path is unaffected; this follower's RPO grows until a
  manual resync.

## First 5 minutes

```bash
# Leader: retention floor and manifest
curl -fsS http://<leader>:17000/v1/wal/status | \
  python3 -c 'import json,sys; c=json.load(sys.stdin)["collections"]; print(json.dumps(c, indent=2))'
curl -fsS http://<leader>:17000/metrics | grep -E \
  'collections_(snapshot_safe_checkpoint_lsn|wal_first_available_lsn|segments_deleted_total)'

# Follower: confirm BEHIND_RETAINED, not network/auth/divergence
curl -fsS http://<follower>:17000/metrics | grep -E \
  'replication_(behind_retained_rejections_total|leader_first_available_lsn|auth_rejections_total|diverged_rejections_total)'
```

- Growing `auth`/`diverged` counters: follow their runbooks first, do not resync.
- This runbook applies only when `behind_retained` is growing.

## Resync (establish a new baseline)

1. **Stop the follower** (keep its logs for diagnosis).
2. On the leader, commit a full snapshot and rotate while holding a retention lease:
   ```bash
   curl -fsS -X POST http://<leader>:17000/v1/admin/collections/compact \
     -H 'content-type: application/json' \
     -d '{"rotate":true,"hold_retention_secs":3600}'
   ```
   The response must have `manifest_committed=true`; record `manifest_seq` and
   `safe_checkpoint_lsn`.
3. Copy the following from the leader to the follower data directory (clear the
   follower's old baseline first):
   - `snapshots/` (manifest plus every snapshot file of this generation);
   - the active collections WAL and all `*.wal.archived.*` segments;
   - `<collections-wal>.next_lsn`;
   - delete the follower's `replication.follower_id` (a new identity is the
     explicit resync semantics).
   For same-host or mounted paths, run:
   ```bash
   ./scripts/collections_resync.sh \
     --leader-http http://<leader>:17000 \
     --leader-wal /data/leader/houyidb.collections.wal \
     --follower-wal /data/follower/houyidb.collections.wal \
     --hold-secs 3600 --yes
   ```
4. Start the follower with `--role follower` and wait for the handshake. Verify:
   ```bash
   curl -fsS http://<follower>:17000/v1/wal/status
   curl -fsS http://<follower>:17000/metrics | grep -E \
     'replication_(lag_entries|behind_retained_rejections_total|leader_first_available_lsn)'
   ```
   `lag_entries` returns to zero and `behind_retained` stops growing.
5. The leader lease keeps retention paused until it expires; after the resync
   completes, either wait it out or call compact again with
   `hold_retention_secs:0` to clear it explicitly.

## Risks and limits

- **RPO**: the follower serves nothing during the resync; only writes after the
  new snapshot point enter its replicated history. Silence writes at the load
  balancer to drive RPO to zero.
- **Encryption**: the follower must hold the same keyring as the leader, or it
  fails closed when decrypting snapshots/WAL.
- **The benchmark main index is not replicated** (`/v1/vectors/*`); this runbook
  does not cover it.
- Do **not** "fix" this by deleting leader segments or raising
  `collections_wal_max_segments`; that only widens the next lag window.
