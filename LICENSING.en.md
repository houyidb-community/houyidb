# Licensing

> [中文](LICENSING.md)

HouyiDB uses an **open-core** model. This file explains which code is under
which license; the legally binding texts are `LICENSE` (Apache-2.0) and the
commercial agreement delivered with the enterprise edition.

## 1. This repository: Apache-2.0

All source code, documentation, scripts, and configuration in this repository
(`houyidb`) are licensed under the **Apache License, Version 2.0**. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

You may use, modify, distribute, and commercially host this code, including in
closed-source products, provided you comply with Apache-2.0 (preserve notices,
include a copy of the license, etc.).

## 2. Enterprise edition: commercial license

The private `houyidb-enterprise` repository and its distributed binaries are
**not** open source. They are licensed under a commercial agreement. Enterprise
features include SSO/SCIM, external KMS/HSM, audit WORM/retention, fleet
management, and certified builds. The community repository contains the complete
single-node engine; it never depends on enterprise code.

Enterprise code is never committed to this repository, and this repository has
no build-time or runtime dependency on it.

## 3. Contributions

Unless a file explicitly states otherwise, contributions accepted into this
repository are licensed under Apache-2.0. Contributors must sign
[`CLA.en.md`](CLA.en.md) so the project can also use the contribution in the
commercial edition. This is the standard dual-licensing CLA model; it does not
transfer copyright ownership.

## 4. Third-party dependencies

Third-party crates remain under their own licenses. `deny.toml` restricts the
allowlist to licenses suitable for commercial distribution; adding a new
license requires explicit review. Run `./scripts/security_audit.sh` before
adding dependencies.

## 5. Trademarks

The Apache-2.0 license does not grant trademark rights. See
[`TRADEMARK.en.md`](TRADEMARK.en.md) for the HouyiDB name and logo policy.

## 6. Questions

For commercial licensing, enterprise evaluation, or relicensing requests,
contact the maintainers (see `CONTRIBUTING.en.md`). This document is a summary,
not legal advice; the signed agreements and `LICENSE` control.
