# HouyiDB Helm Chart

> [English](README.en.md)

> 单 release = 单节点；**主备是两个 release**，不是 replicaCount。
> v1 只有手动 promote，不做自动选主/quorum。

## 安装（单机）

```bash
# 默认镜像：ghcr.io/houyidb-community/houyidb:v0.2.0-beta.8
helm install houyi ./deploy/helm/houyidb \
  --namespace houyidb --create-namespace \
  --set persistence.size=100Gi
```

使用自建镜像时覆盖 `image.repository` / `image.tag`：

```bash
docker build -t registry.example.com/houyidb:v0.2.0-beta.8 .
docker push registry.example.com/houyidb:v0.2.0-beta.8
helm install houyi ./deploy/helm/houyidb -n houyidb \
  --set image.repository=registry.example.com/houyidb \
  --set image.tag=v0.2.0-beta.8
```

## 主备（手动 failover）

```bash
# leader
helm install houyi-leader ./deploy/helm/houyidb -n houyidb \
  --set role=leader \
  --set replication.authTokenSecret=houyi-replication

# follower（leaderAddr 指向 leader 的 Service + replication 端口）
helm install houyi-follower ./deploy/helm/houyidb -n houyidb \
  --set role=follower \
  --set replication.leaderAddr=houyi-leader-houyidb.houyidb.svc:9876 \
  --set replication.authTokenSecret=houyi-replication

# 触发手动 promote（见 docs/runbooks/replication-divergence.md 的风险边界）
kubectl -n houyidb exec houyi-follower-houyidb-0 -- \
  curl -fsS -X POST http://127.0.0.1:7000/v1/admin/promote
```

## 安全（推荐生产开启）

```bash
# RBAC / 完整配置（含 token/principals）走 Secret
kubectl -n houyidb create secret generic houyi-config \
  --from-file=config.toml=./config.toml

# 静态加密 keyring（0600；Secret 内 key 名必须为 keyring）
kubectl -n houyidb create secret generic houyi-keyring \
  --from-file=keyring=./keyring

# TLS
kubectl -n houyidb create secret tls houyi-tls --cert=tls.crt --key=tls.key

helm upgrade --install houyi ./deploy/helm/houyidb -n houyidb \
  --set config.existingSecret=houyi-config \
  --set encryption.existingSecret=houyi-keyring \
  --set tls.enabled=true --set tls.existingSecret=houyi-tls
```

- 不要用 `--set` 传 token/密钥（会进 Helm release Secret 与命令历史）。
- 数据卷默认 `50Gi` RWO；加密 keyring 以 `0400` 挂载。
- Pod 以非 root（10001）运行、只读根文件系统、`/tmp` emptyDir。
- 告警规则：`monitoring.prometheusRule.enabled=true` 并粘贴
  `deploy/prometheus/houyidb-alerts.yml` 的 groups（需 Prometheus Operator CRD）。

## 验证

```bash
helm lint ./deploy/helm/houyidb
helm template houyi ./deploy/helm/houyidb > /tmp/houyidb-rendered.yaml
kubectl apply --dry-run=server -f /tmp/houyidb-rendered.yaml
```
