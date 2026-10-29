# HouyiDB Deployment Guide

> [中文](DEPLOYMENT_GUIDE.md)

> Applies to v0.2.0-beta.5. HouyiDB is a **single-node, memory-first** vector database;
> high availability is manual leader/follower promotion, with no automatic election.
> During beta, use it first for functional validation and non-critical workloads.

## 1. Deployment Forms

| Form | Use case | Entry point |
|------|----------|------|
| Docker Compose | Quick start, single-node trial | `docker-compose.yml` |
| Prebuilt binary + systemd | Linux single-node deployment | Release package in `docs/USER_MANUAL.en.md` |
| Source build | Custom features / development and debugging | `cargo build --release -p houyi-server` |
| Kubernetes Helm | In-cluster single node or primary/standby | `deploy/helm/houyidb/` |

## 2. Ports and Data Directory

| Purpose | Default address | Override |
|------|----------|----------|
| HTTP API | `127.0.0.1:7000` | `--benchmark-http-addr` / `HOUYI_SERVER_BENCHMARK_HTTP_ADDR` |
| pgwire | `127.0.0.1:5432` | `--pgwire-addr` / `HOUYI_SERVER_PGWIRE_ADDR` |
| Replication stream (leader only) | `0.0.0.0:9876` | `--replication-listen-addr` / `HOUYI_SERVER_REPLICATION_LISTEN_ADDR` |

The data directory is determined by `--wal-path`, from which the following are derived in the same directory:

```
<wal-path>                     # 向量 WAL
<stem>.collections.wal         # collection 元数据/向量/payload 的 WAL
<stem>.applied_lsn             # 已应用 LSN 旁路文件
<stem>.wal.next_lsn            # 下一 LSN 旁路文件
snapshots/                     # 各 collection 的 HNSW 图快照
```

**Production must set `--wal-path` explicitly**: the default falls under the system temp directory, and a reboot or `/tmp` cleanup will lose data. The directory must be writable by the service user and included in backups.

> `*.collections.wal` currently only grows and is never compacted. Sustained high-frequency writes will keep it growing; reserve capacity for write volume during capacity planning, and run cold-start recovery drills periodically.

## 3. Docker Compose (Recommended Starting Point)

```bash
docker compose up -d --build
curl -fsS http://127.0.0.1:7000/v1/health
```

Key conventions of `docker-compose.yml`:

- The `houyidb-data` data volume is mounted at `/data` in the container, and the image sets
  `HOUYI_SERVER_WAL_PATH=/data/houyidb.wal`;
- HTTP 7000 and pgwire 5432 are mapped to the host;
- Optional OpenAI-compatible gateway: after setting `HOUYI_OPENAI_UPSTREAM` (and
  `HOUYI_OPENAI_API_KEY`), `/v1/embeddings` becomes available.

On port conflicts, change only the host-side mapping (keep 7000/5432 inside the container):

```bash
docker run -d --name houyidb \
  -p 17000:7000 -p 15432:5432 \
  -v houyidb-data:/data houyidb:latest
```

On macOS, the AirPlay Receiver may occupy 5000/7000; prefer changing the host port.

In production, enable authentication via environment variables or TOML (do not write it into the image):

```yaml
environment:
  HOUYI_AUTH_TOKEN: "${HOUYI_AUTH_TOKEN}"
  HOUYI_SERVER_WAL_WRITE_MODE: "async"
```

## 4. Binary + systemd

```bash
# 1) Prepare the user and directories
sudo useradd --system --home /var/lib/houyidb --shell /usr/sbin/nologin houyidb
sudo install -d -o houyidb -g houyidb -m 750 /var/lib/houyidb /etc/houyidb
sudo install -o houyidb -g houyidb -m 750 houyi-server /usr/local/bin/houyi-server

# 2) Environment file (mode 600; especially important when it contains a token)
sudo install -o root -g houyidb -m 640 /dev/null /etc/houyidb/houyidb.env
```

`/etc/houyidb/houyidb.env`:

```bash
HOUYI_SERVER_BENCHMARK_HTTP_ADDR=0.0.0.0:7000
HOUYI_SERVER_PGWIRE_ADDR=0.0.0.0:5432
HOUYI_SERVER_WAL_PATH=/var/lib/houyidb/houyidb.wal
HOUYI_SERVER_WAL_WRITE_MODE=async
HOUYI_AUTH_TOKEN=<长随机串>
# Optionally use TOML for full configuration such as RBAC/TLS/encryption:
# HOUYI_CONFIG=/etc/houyidb/config.toml
```

`/etc/systemd/system/houyidb.service`:

```ini
[Unit]
Description=HouyiDB server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=houyidb
Group=houyidb
EnvironmentFile=/etc/houyidb/houyidb.env
ExecStart=/usr/local/bin/houyi-server
Restart=on-failure
RestartSec=2
LimitNOFILE=65536
# SIGTERM triggers a graceful shutdown (including a shutdown graph snapshot); allow enough time to avoid a forced kill
KillSignal=SIGTERM
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now houyidb
curl -fsS http://127.0.0.1:7000/v1/health
```

## 5. Building from Source

Requires stable Rust (1.87+ recommended):

```bash
cargo build --release -p houyi-server
./target/release/houyi-server --help
```

Docker builds default to a generic cross-CPU path; for AVX2:

```bash
docker build --build-arg CARGO_FEATURES=simd-avx2 -t houyidb:avx2 .
```

## 6. Kubernetes (Helm)

Single node:

```bash
helm install houyi ./deploy/helm/houyidb -n houyidb --create-namespace \
  --set persistence.size=100Gi
```

The default image is `ghcr.io/houyidb-community/houyidb:v0.2.0-beta.5`;
override `image.repository` / `image.tag` for a self-built image.

Primary/standby uses **two releases** (v1 has no automatic election; increasing
`replicaCount` only yields multiple independent single nodes):

```bash
helm install houyi-leader ./deploy/helm/houyidb -n houyidb \
  --set role=leader --set replication.authTokenSecret=houyi-replication

helm install houyi-follower ./deploy/helm/houyidb -n houyidb \
  --set role=follower \
  --set replication.leaderAddr=houyi-leader-houyidb.houyidb.svc:9876 \
  --set replication.authTokenSecret=houyi-replication
```

For Secret conventions and how TLS/encryption/RBAC are mounted, see
`deploy/helm/houyidb/README.en.md`; pass the full `config.toml` through a Secret, and do not
pass credentials via `--set` (they remain in the Helm release and shell history).

## 7. Configuration Precedence and TOML

Precedence: **CLI flags > environment variables > TOML > defaults**. The configuration file is
specified by `--config` or `HOUYI_CONFIG`; unknown keys, enum typos, and empty tokens all cause
startup failure (fail-fast / fail-closed).

```toml
# /etc/houyidb/config.toml
benchmark_http_addr = "0.0.0.0:7000"
pgwire_addr = "0.0.0.0:5432"
wal_path = "/var/lib/houyidb/houyidb.wal"
wal_write_mode = "async"          # async | sync（复制要求 async）
vector_dim = 1536
index_type = "hnsw"               # hnsw|clustered|flat-clustered|flat|pq-flat|hii-l0
snapshot_interval_secs = 600

# TLS (HTTP and pgwire share the same certificate pair)
tls_cert_path = "/etc/houyidb/tls.crt"
tls_key_path  = "/etc/houyidb/tls.key"

# Encryption at rest (AES-256-GCM; each keyring line is `id=<64-hex>` and may add `active=<id>`)
encryption_key_file = "/etc/houyidb/keyring"

# Audit log (SHA-256 hash chain; verified at startup; a broken chain refuses to start)
audit_log_path = "/var/lib/houyidb/audit.log"
audit_include_reads = false

# RBAC: role = admin | writer | reader; choose either token or token_sha256
[[auth_principals]]
name = "app-writer"
role = "writer"
token_sha256 = "<sha256-hex>"
collections = ["docs", "images"]
max_collections = 8
max_inflight = 64
writes_per_sec = 1000
```

- `auth_token` / `HOUYI_AUTH_TOKEN` is the legacy **full admin** channel;
  use `[[auth_principals]]` for multiple principals.
- Unknown credentials return 401, and a known principal exceeding its scope returns 403;
  quota overruns return HTTP 429 and pgwire `SQLSTATE 54000`.
- The pgwire password is the principal token; use it only over TLS or on localhost.
- Losing the keyring makes WAL/snapshots undecryptable; back it up separately from the data,
  with permissions 600.

## 8. Minimal Production Combination of Authentication, TLS, and Encryption

1. Terminate TLS at a reverse proxy, or configure `tls_cert_path` / `tls_key_path` on the service
   directly (applies to HTTP and pgwire simultaneously);
2. Enable `HOUYI_AUTH_TOKEN` or `[[auth_principals]]`; never run an unauthenticated instance
   on the public internet;
3. If encryption at rest is needed, create the keyring before the first start (missing keys
   fail closed);
4. Once the audit log is enabled, verify the hash chain with
   `GET /v1/admin/audit/verify`.

> The replication port has no TLS, only shared-token authentication; it must be isolated on an
> internal network or by firewall.

## 9. Backup and Recovery

The recommended practice is "stop writes, then copy the entire data directory":

```bash
sudo systemctl stop houyidb
sudo tar czf houyidb-backup-$(date +%F).tgz \
  -C /var/lib houyidb
sudo systemctl start houyidb
```

- The directory must include the WAL, `*.collections.wal`, sidecar files, `snapshots/`, and
  the keyring (if encryption is enabled);
- While running, WAL and snapshot files are still changing; a hot copy **cannot guarantee
  cross-file consistency**; for critical data use stop-write backups, or use primary/standby
  (after the follower catches up, back up/promote on it);
- Restore: extract the backup to the original path, set ownership to `houyidb`; on startup,
  WAL and graph snapshots are replayed automatically;
- Periodically run cold-start drills with a copy to verify the backup is usable, not merely
  that the files exist.

## 10. Primary/Standby and Manual Failover

```bash
# leader
houyi-server --role leader \
  --replication-listen-addr 0.0.0.0:9876 \
  --wal-path /data/houyidb.wal

# follower (start with an empty directory; full sync happens automatically)
houyi-server --role follower \
  --replication-leader-addr 10.0.0.10:9876 \
  --wal-path /data/houyidb.wal
```

- leader/follower must be configured with the same `replication_auth_token` (ENV or TOML);
- Replication requires `wal_write_mode=async`;
- Collection-level HA: `async` / `semi_sync` / `full_sync`, configured via
  `POST /v1/collections/{name}/ha_level`; acknowledged writes under `full_sync`
  satisfy RPO=0;
- Promotion can only be executed on a follower:

```bash
curl -fsS -X POST http://<follower>:7000/v1/admin/promote \
  -H "Authorization: Bearer $TOKEN"
# {"promoted":true,"next_lsn":...}
```

- promote is idempotent: returns 409 if already leader, and 400 if `role=disabled`;
- After failover, the **old leader must be stopped** and its data directory cleared, then it must
  reconnect to the new leader as a follower to avoid split-brain:
  `rm -rf /data/* && houyi-server --role follower ...`;
- Replication covers only the collection path; `/v1/vectors` (the benchmark main index) is not
  replicated, so after failover the new leader has no data for that path. Use the Collections API
  for production data;
- v1 has no automatic election/quorum: monitor replication lag, then promote manually after
  human judgment.

## 11. Observability

| Endpoint | Purpose |
|------|------|
| `GET /v1/health` | Liveness check (auth-exempt) |
| `GET /readyz` | Readiness check (auth-exempt) |
| `GET /metrics` | Prometheus text metrics (auth-exempt) |
| `GET /v1/index/status` | Main index status |
| `GET /v1/collections/{name}/status` | Status of a single collection |
| `GET /v1/pgwire/status` | pgwire session statistics |

- Alert rules: `deploy/prometheus/houyidb-alerts.yml`, each alert carries a runbook link
  (`docs/runbooks/`);
- SLOs and error budgets: `docs/SLO.en.md`;
- Logs go to stdout/stderr (`tracing`); adjust with `RUST_LOG=info`;
  `--slow-query-ms` prints slow queries.

## 12. Upgrade and Rollback

1. Back up the data directory first (see §9);
2. Single node: stop the service → replace the binary/image → start → verify `/v1/health` and
   sample retrievals;
3. Primary/standby: upgrade the follower first and confirm it has caught up → promote → upgrade
   the old leader → clear its data and reconnect as a follower;
4. The first start after an upgrade replays WAL/graph snapshots as needed; the time grows with
   data volume;
5. **No downgrade compatibility is promised**: restore the pre-upgrade backup before rolling
   back; do not open new-format data directly with the old binary.

## 13. Troubleshooting

| Symptom | Resolution |
|------|------|
| `Address already in use` | Change the port; on macOS check for AirPlay occupying 5000/7000 |
| Startup fails with a missing-key error | Keyring not configured or wrong path; once encryption is enabled, the service cannot start without a key |
| Audit chain verification fails | Log truncated/tampered; restore from backup or follow the security incident process |
| Replication `AUTH REJECTED` | `replication_auth_token` differs between the two sides |
| Follower read-only 503 / pgwire 25006 | Expected: followers do not accept writes; promote first |
| Replication lag keeps growing | Check the network and the follower disk; see `docs/runbooks/replication-lag.en.md` |
| Retrieval dimension 400 | The query vector length must equal the dimension of the collection's first write |
| Filter returns no results | Confirm the backend is hnsw/flat and that column names and operators are valid |
| WAL replay takes very long | Large data volume or `collections.wal` uncompacted for a long time; evaluate snapshots and capacity |

For more trial/client questions, see `docs/USER_MANUAL.en.md`; for protocol details, see
`docs/HOUYIDB_SERVER_MANUAL.en.md`.
