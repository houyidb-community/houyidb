# HouyiDB SLO and Error Budgets

> [中文](SLO.md)

> Version: 2026-10-21 (initial). Target values will be calibrated quarterly after the first production data / benchmark baseline;
> any relaxation must have a change record and a capacity assessment.
> EN: initial SLOs. Calibrate quarterly or after the first production baseline.

## 1. Service Boundary

| Surface | Entry point | Description |
|----|------|------|
| Data plane (HTTP) | `/v1/collections/{name}/...` | upsert/search/delete/hybrid |
| Data plane (pgwire) | SQL Simple/Extended | Shares the same CollectionManager as HTTP |
| Management plane | `/v1/admin/*`, `/metrics` | Not counted toward data-plane availability |
| Probes | `/v1/health`, `/readyz` | Used only for availability determination; not counted toward request success rate |

## 2. SLIs and Targets

### 2.1 Availability

- **SLI**: successful requests / valid requests. Success = not 5xx and not timed out; 4xx (auth/quota/
  client errors) are excluded from the denominator, but a **spike** in 401/403/429 triggers security/capacity alerts.
- **Target**: **99.9% / calendar month** (43m 12s of allowed downtime).
- **Data source**: request counts and status codes from the ingress load balancer/Blackbox probes; or request
  counters added later in the HTTP/pgwire middleware.

### 2.2 Latency

| Operation | SLI | Target |
|------|-----|------|
| Vector search | p99 end-to-end | ≤ 100ms (k ≤ 10, ≤ 1M vectors, HNSW) |
| upsert (async/semi) | p99 end-to-end | ≤ 200ms |
| upsert (full_sync) | p99 end-to-end | ≤ 2× `replication_sync_timeout_ms` (default 2s → 4s cap) |
| pgwire simple query | p99 | Same criteria as HTTP |

Latency counts successful requests only; timeouts/5xx are charged to the availability error budget.
Targets follow deployment-side observations and the self-benchmark data in `VDBBENCH_PROTOCOL.en.md`.

### 2.3 Replication and Durability (RPO / RTO)

| HA level | RPO target | Description |
|---------|----------|------|
| full_sync | **0** | An acknowledged write must be durable on the follower; asserted by full_sync ACK semantics and failure drills |
| semi_sync | 0 (degrades on wait timeout) | Degrades to async on timeout; the window is tracked by `sync_timeouts_total` |
| async | Bounded (≤ 1s) | Constrained by the WAL flush policy; not a durability guarantee |

- **RTO target**: single-node restart recovery ≤ 60s (with a graph snapshot, 1M-vector scale); manual promote
  ≤ 30s. Measure it with the restart/promote drills in the deployment guide.
- Replication lag SLI: `houyidb_replication_lag_entries` p95 ≤ 1000 and 5-minute mean
  ≤ 100; beyond that, follow the [Replication Lag Runbook](runbooks/replication-lag.en.md).

## 3. Error Budget Policy

- Monthly budget = 0.1% × minutes in the month; consumed by the availability SLI.
- **Burn rate alerts**:
  - Fast burn: 1-hour window burn rate > 14.4 (2% budget/hour) → **page**;
  - Slow burn: 6-hour window burn rate > 6 (5% budget/6 hours) → ticket.
- Budget exhausted (remaining < 10%):
  1. Freeze non-urgent feature merges and prioritize reliability fixes;
  2. Review recent changes and incidents;
  3. Resuming releases requires double confirmation from the DBA/owner.
- A healthy budget does not exempt security/correctness gates (SLOs are not a reason to lower quality).

## 4. Integration with Alerts/Drills

- Prometheus rules: `deploy/prometheus/houyidb-alerts.yml` (runbook_url points to
  `docs/runbooks/`).
- RTO/MTTR: keep a record of every drill for quarterly calibration.
- Incident severity: P0 (data loss/unavailability > 30m/security incident), P1 (availability affected),
  P2 (single node/non-core); P0/P1 require a postmortem and an update to the SLO or Runbook.

## 5. Pending Measurements (Honest Current Gaps)

- HTTP total request/latency histograms currently depend on external probes or gateway metrics; later,
  `houyidb_http_requests_total{status}` and latency histograms can be added in middleware (out of scope for this initial SLO).
- RTO targets need at least 3 drill data points (across different data-size tiers) before they can be committed as promises.
