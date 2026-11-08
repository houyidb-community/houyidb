# Security Policy

> [中文](SECURITY.md)

## Supported versions

HouyiDB is currently in beta (`0.2.x`). Security fixes are provided for the
latest release tag on `main`; older beta tags may not receive patches.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Use GitHub's private vulnerability reporting:

1. Open <https://github.com/houyidb-community/houyidb/security/advisories>
2. Click **Report a vulnerability**
3. Include: affected version/tag, deployment mode (standalone/leader/follower),
   reproduction steps, impact, and any suggested fix.

If the advisory form is unavailable, contact the maintainers through the
repository's private channels before disclosing publicly.

## What to expect

| Stage | Target |
|-------|--------|
| Acknowledge report | 72 hours |
| Initial triage / severity | 7 days |
| Fix or mitigation plan | 30 days (critical: as fast as possible) |
| Coordinated disclosure | 90 days after fix, by agreement |

We credit reporters in the advisory unless you prefer to remain anonymous.

## Scope

In scope:

- The Rust server and libraries in this repository;
- Authentication/authorization (RBAC, token handling, pgwire/TLS);
- At-rest encryption (WAL v3, snapshot v2, keyring) and audit hash chain;
- Replication protocol (v5), promote/failover logic, WAL recovery;
- Denial of service via malformed pgwire/HTTP input;
- Supply-chain issues in shipped binaries.

Out of scope:

- Vulnerabilities requiring a compromised host/root (see threat model in
  `houyi-persistence/src/encryption.rs`);
- Misconfiguration with no security control bypass (e.g., intentionally
  disabling auth);
- Third-party dependencies without a reachable path in HouyiDB (report upstream,
  but we still want to know);
- Social engineering.

## Safe harbor

We will not pursue legal action for good-faith research that respects privacy,
does not degrade service, and follows coordinated disclosure. Do not test
against systems you do not own or have permission to test.

## Security features and verification

- For security configuration and deployment boundaries, see the
  [Deployment Guide](docs/DEPLOYMENT_GUIDE.en.md); for incident handling, see the
  [Alert Runbooks](docs/runbooks/README.en.md).
- Run `./scripts/security_audit.sh` for cargo audit/deny/SBOM locally.
- Audit chain verification:
  `GET /v1/admin/audit/verify` (Admin token required).
