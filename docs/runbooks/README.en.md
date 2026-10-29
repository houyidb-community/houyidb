# HouyiDB Alert Runbooks

> [中文](README.md)

> Executable on-call cards for on-call engineers. **Not** a design document: every card must be
> followable within 5 minutes and must explicitly state "what not to do". Alert rules are in
> `deploy/prometheus/houyidb-alerts.yml` (`runbook_url` points to this directory).
> EN: executable on-call cards. The alert file links here via `runbook_url`.

## General Conventions

- **Preserve the scene first, then recover**: do not delete/truncate WAL, snapshots, or audit segments; copy
  the files to a safe directory first, then proceed per the runbook.
- **Read-only first**: where promote can switch read/write, do not restart both nodes.
- **Never bypass security controls**: when a key is missing or the audit chain is broken, do not start the
  service with encryption or auditing temporarily disabled.
- Keep a timeline for every incident (triggering metrics, commands, recovery time) for RTO/MTTR statistics and
  postmortems.

## Index

| Alert / Event | Severity | Runbook |
|-------------|--------|------|
| Replication log divergence (diverged) / follower_ahead | Critical | [replication-divergence.en.md](replication-divergence.en.md) |
| Replication lag, slow follower kicked | Warning | [replication-lag.en.md](replication-lag.en.md) |
| WAL poisoned (writes rejected after a write failure) | Critical | [wal-poison.en.md](wal-poison.en.md) |
| Missing encryption key for encrypted snapshot/WAL causes startup failure | Critical | [encryption-key-missing.en.md](encryption-key-missing.en.md) |
| Audit chain verification failure / audit write error | Critical | [audit-chain-broken.en.md](audit-chain-broken.en.md) |
| Quota rejection or authentication failure storm | Warning | [quota-and-auth-storm.en.md](quota-and-auth-storm.en.md) |
| Memory watermark / low disk / snapshot rejected | Warning | [resource-watermarks.en.md](resource-watermarks.en.md) |

## General Diagnostic Entry Points

```bash
# Metrics (directly against the local benchmark HTTP port)
curl -fsS http://127.0.0.1:17000/metrics | grep -E 'houyidb_(replication|wal|audit|pgwire_quota|memory)'

# Readiness state
curl -fsS http://127.0.0.1:17000/readyz

# Key patterns in the service logs
#   wal poisoned by write failure
#   replication diverged / follower_ahead
#   AES-GCM authentication failed
#   missing encryption key
#   audit chain broken
```
