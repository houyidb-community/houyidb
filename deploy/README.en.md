# HouyiDB Deployment Assets

> [中文](README.md)

For the complete deployment procedure (Docker, binary + systemd, source build,
Kubernetes, authentication/TLS/encryption, backup and primary/standby), see
[`docs/DEPLOYMENT_GUIDE.en.md`](../docs/DEPLOYMENT_GUIDE.en.md).

This directory contains deployment-related configuration assets only:

```
deploy/
├── helm/houyidb/            # Helm chart（单机 / 主备两个 release）
├── prometheus/              # 告警规则
├── macos/houyi-server       # 历史预编译二进制（发布包以 GitHub Release 为准）
└── python/houyidb_client.py # 历史演示客户端（新项目请用 sdk/python/houyidb）
```

- Container images: `Dockerfile` and `docker-compose.yml` in the repository root;
- Kubernetes: see `deploy/helm/houyidb/README.en.md`;
- Python / TypeScript / Go / Java / .NET / Rust clients: see `sdk/`.
