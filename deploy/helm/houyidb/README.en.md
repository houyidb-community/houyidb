# HouyiDB Helm Chart

> [中文](README.md)

> A single release = a single node; **primary/standby uses two releases**, not replicaCount.
> v1 provides manual promote only; no automatic leader election/quorum.

## Installation (Standalone)

```bash
# Default image: ghcr.io/houyidb-community/houyidb:v0.2.0-beta.5
helm install houyi ./deploy/helm/houyidb \
  --namespace houyidb --create-namespace \
  --set persistence.size=100Gi
```

To use a self-built image, override `image.repository` / `image.tag`:

```bash
docker build -t registry.example.com/houyidb:v0.2.0-beta.5 .
docker push registry.example.com/houyidb:v0.2.0-beta.5
helm install houyi ./deploy/helm/houyidb -n houyidb \
  --set image.repository=registry.example.com/houyidb \
  --set image.tag=v0.2.0-beta.5
```

## Primary/Standby (Manual Failover)

```bash
# leader
helm install houyi-leader ./deploy/helm/houyidb -n houyidb \
  --set role=leader \
  --set replication.authTokenSecret=houyi-replication

# follower (leaderAddr points to the leader's Service + replication port)
helm install houyi-follower ./deploy/helm/houyidb -n houyidb \
  --set role=follower \
  --set replication.leaderAddr=houyi-leader-houyidb.houyidb.svc:9876 \
  --set replication.authTokenSecret=houyi-replication

# Trigger a manual promote (see the risk boundaries in docs/runbooks/replication-divergence.en.md)
kubectl -n houyidb exec houyi-follower-houyidb-0 -- \
  curl -fsS -X POST http://127.0.0.1:7000/v1/admin/promote
```

## Security (Recommended in Production)

```bash
# RBAC / full configuration (including token/principals) goes through a Secret
kubectl -n houyidb create secret generic houyi-config \
  --from-file=config.toml=./config.toml

# Encryption-at-rest keyring (0600; the key name in the Secret must be keyring)
kubectl -n houyidb create secret generic houyi-keyring \
  --from-file=keyring=./keyring

# TLS
kubectl -n houyidb create secret tls houyi-tls --cert=tls.crt --key=tls.key

helm upgrade --install houyi ./deploy/helm/houyidb -n houyidb \
  --set config.existingSecret=houyi-config \
  --set encryption.existingSecret=houyi-keyring \
  --set tls.enabled=true --set tls.existingSecret=houyi-tls
```

- Do not pass tokens/keys via `--set` (they end up in the Helm release Secret and shell history).
- The data volume defaults to `50Gi` RWO; the encryption keyring is mounted with `0400`.
- Pods run as non-root (10001), with a read-only root filesystem and an `emptyDir` for `/tmp`.
- Alert rules: set `monitoring.prometheusRule.enabled=true` and paste the groups from
  `deploy/prometheus/houyidb-alerts.yml` (requires the Prometheus Operator CRD).

## Verification

```bash
helm lint ./deploy/helm/houyidb
helm template houyi ./deploy/helm/houyidb > /tmp/houyidb-rendered.yaml
kubectl apply --dry-run=server -f /tmp/houyidb-rendered.yaml
```
