# SDK Release Guide

> [中文](SDK_RELEASE.md)

> Goal: publish all six SDKs to their package registries and keep a repeatable
> CI release path. Account creation, 2FA, terms acceptance, and payments are
> owner-only; once tokens are configured, the publish commands can run locally
> or in GitHub Actions.

## 0. Current State

- Name availability checked on 2026-10-26: PyPI `houyidb`, npm `houyidb`,
  crates.io `houyidb-client`, and NuGet `HouyiDb` all return 404; the Maven
  Central group `top.houyidb` has no artifacts yet.
- Local artifacts: `./scripts/package_sdks.sh` builds everything into
  `dist/sdk/`.
- CI: `.github/workflows/publish-sdks.yml` builds artifacts by default; it only
  publishes on release events or manual dispatch with `publish=true`.
- The Maven Central group is now `top.houyidb`, matching the owned
  `houyidb.top` domain; the Java package name remains `com.houyidb.client`
  (independent of Maven coordinates).

## 1. Publish Order

| Order | Registry | Package | Account | CI Secret |
|-------|----------|---------|---------|-----------|
| 1 | PyPI | `houyidb` / `langchain-houyidb` / `llama-index-vector-stores-houyidb` | pypi.org | Trusted Publishing (no token) |
| 2 | npm | `houyidb` / `houyidb-mcp` | npmjs.com | `NPM_TOKEN` |
| 3 | crates.io | `houyidb-client` | crates.io | `CARGO_REGISTRY_TOKEN` |
| 4 | NuGet | `HouyiDb` | nuget.org | Trusted Publishing (`NUGET_USER`) |
| 5 | Maven Central | `top.houyidb:houyidb-client` | Sonatype Central Portal | GPG + Portal token |
| 6 | Go | `github.com/houyidb-community/houyidb/sdk/go/houyidb` | none | none (tag) |

Publish PyPI/npm/crates/NuGet first, then Maven Central (longest process),
then the Go tag.

## 2. PyPI (Trusted Publishing Recommended)

1. Register at <https://pypi.org> and enable 2FA.
2. **Important**: PyPI pending publishers are unique per
   `(owner, repository, workflow, environment)`. A monorepo cannot reuse one
   configuration for multiple not-yet-created projects; the third attempt
   fails with `A pending trusted publisher matching this configuration has
   already been registered for a different project name`. Use one environment
   per package:

   | PyPI Project Name | Environment |
   |-------------------|-------------|
   | `houyidb` | `pypi` |
   | `langchain-houyidb` | `pypi-langchain` |
   | `llama-index-vector-stores-houyidb` | `pypi-llama` |

   All three use Owner `houyidb-community`, Repository `houyidb`, Workflow
   `publish-sdks.yml`.
3. Create the `pypi`, `pypi-langchain`, and `pypi-llama` environments in GitHub
   repository Settings (no secret needed).
4. Push a `v*` tag or run `Publish SDKs` manually with publish enabled.
5. Manual fallback:

```bash
python3 -m build
python3 -m twine upload dist/*
```

## 3. npm

1. Register at <https://www.npmjs.com> and enable 2FA.
2. Create a **Granular Access Token** (Read and Publish, scoped to `houyidb`
   and `houyidb-mcp`), or use `npm login`.
3. Add the `NPM_TOKEN` secret to the GitHub repository.
4. Locally:

```bash
cd sdk/typescript/houyidb
npm ci && npm run build
npm publish --provenance --access public
```

MCP package:

```bash
cd mcp/houyidb-mcp
npm ci && npm run build
npm publish --access public
```

> npm publishes are irreversible (you can only deprecate); inspect the tarball
> with `npm pack` before publishing.

## 4. crates.io

1. Sign in at <https://crates.io> with GitHub.
2. Account Settings → API Tokens → create a token.
3. Add the `CARGO_REGISTRY_TOKEN` secret to the GitHub repository.
4. Locally:

```bash
cargo login <token>
cargo publish -p houyidb-client
```

> The first publish reserves `houyidb-client`; versions cannot be overwritten.

## 5. NuGet (Trusted Publishing, no API key)

1. Sign in at <https://www.nuget.org>, click your username →
   **Trusted Publishing** → **Add policy**.
2. Configure the GitHub policy:
   - Owner: your nuget.org user (or the organization that owns `HouyiDb`)
   - Repository Owner: `houyidb-community`
   - Repository: `houyidb`
   - Workflow File: `publish-sdks.yml` (filename only)
   - Environment: `nuget`
   - Scope: allow **Push new packages and package versions**; glob `HouyiDb`
     (use `HouyiDb*` if the UI expects a wildcard)
3. Add the GitHub repository secret `NUGET_USER`; its value is your nuget.org
   **profile name**, not your email.
4. No `NUGET_API_KEY` is needed; the `NuGet/login` workflow step exchanges a
   GitHub OIDC token for a temporary key valid for one hour.
5. Pack locally only:

```bash
cd sdk/dotnet
dotnet pack src/HouyiDb/HouyiDb.csproj -c Release -o dist
```

> You do not need to disable 2FA; Trusted Publishing does not require OTP in
> CI. A new policy may be temporarily active for seven days and becomes
> permanent after the first successful publish.

## 6. Maven Central (Sonatype Central Portal)

1. Register at <https://central.sonatype.com>.
2. Create the namespace `top.houyidb`; the portal provides a DNS TXT record to
   add to `houyidb.top` and wait for verification.
3. Generate a GPG key and publish the public key:

```bash
gpg --gen-key
gpg --keyserver keys.openpgp.org --send-keys <KEY_ID>
```

4. Build and sign (add the `signing` plugin to
   `sdk/java/houyidb/build.gradle`, using
   `ORG_GRADLE_PROJECT_signingKey` / `signingPassword` environment variables;
   never commit the private key).
5. Build the bundle and upload it to the Central Portal:

```bash
cd sdk/java/houyidb
gradle publishAllPublicationsToLocalBuildRepository
# zip build/repo and upload it via Central Portal's Upload Bundle
```

6. After validation, click Publish; artifacts usually appear in Maven Central
   within minutes.

> If you do not want to maintain GPG/Central yet, ship PyPI/npm/NuGet/crates
> first and add Maven Central later. Java users can still use the jar from the
> GitHub Release.

## 7. Go Submodule

Go has no central package upload; tag the submodule after release:

```bash
git tag sdk/go/houyidb/v0.2.0-beta.8
git push origin sdk/go/houyidb/v0.2.0-beta.8
```

Users:

```bash
go get github.com/houyidb-community/houyidb/sdk/go/houyidb@v0.2.0-beta.8
```

> The root tag `v0.2.0-beta.8` is not the submodule version; use the
> `sdk/go/houyidb/vX.Y.Z` form.

## 8. Post-Release Verification

```bash
pip index versions houyidb
npm view houyidb version
cargo search houyidb-client
dotnet package search HouyiDb
curl -s "https://search.maven.org/solrsearch/select?q=g:top.houyidb&rows=5&wt=json"
```

Install each package once and run a minimal example:

```bash
python3 -m venv /tmp/houyi-sdk && /tmp/houyi-sdk/bin/pip install houyidb
npm install houyidb
```

## 9. Security Rules

- Tokens live only in GitHub Secrets or local environment variables; never in
  the repository, logs, or screenshots.
- Enable 2FA everywhere; use least-privilege tokens and rotate them.
- Confirm version consistency before publishing: Python `0.2.0b8`,
  npm/Java/.NET/Rust `0.2.0-beta.8`, Go tag `v0.2.0-beta.8`.
- Publishing is irreversible; run `./scripts/package_sdks.sh` and inspect
  `dist/sdk/` first.
