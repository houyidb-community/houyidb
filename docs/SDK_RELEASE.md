# SDK 包发布手册

> [English](SDK_RELEASE.en.md)

> 目标：把六个 SDK 发布到各自的包管理器，并保留可复现的 CI 发布路径。
> 账号注册、2FA、条款接受与付费只能由项目所有者完成；Token 配置好后，
> 发布命令可以本地执行，也可以交给 GitHub Actions。

## 0. 当前状态

- 包名可用性已查询（2026-10-26）：PyPI `houyidb`、npm `houyidb`、
  crates.io `houyidb-client`、NuGet `HouyiDb` 均返回 404；Maven Central
  `top.houyidb` 组下无构件。
- 本地产物：`./scripts/package_sdks.sh` 会构建到 `dist/sdk/`。
- CI 流水线：`.github/workflows/publish-sdks.yml`，默认只构建 artifact；
  只有 release 事件或手动勾选 `publish=true` 才真正上传。
- Maven Central 组名已改为 `top.houyidb`，与已拥有的 `houyidb.top` 域名
  对应；Java 包名仍是 `com.houyidb.client`（不影响 Maven 坐标）。

## 1. 发布顺序

| 顺序 | 平台 | 包 | 账号 | CI Secret |
|------|------|----|------|-----------|
| 1 | PyPI | `houyidb` / `langchain-houyidb` / `llama-index-vector-stores-houyidb` | pypi.org | Trusted Publishing（无 Token） |
| 2 | npm | `houyidb` / `houyidb-mcp` | npmjs.com | `NPM_TOKEN` |
| 3 | crates.io | `houyidb-client` | crates.io | `CARGO_REGISTRY_TOKEN` |
| 4 | NuGet | `HouyiDb` | nuget.org | Trusted Publishing（`NUGET_USER`） |
| 5 | Maven Central | `top.houyidb:houyidb-client` | Sonatype Central Portal | GPG + Portal Token |
| 6 | Go | `github.com/houyidb-community/houyidb/sdk/go/houyidb` | 无 | 无（打 tag） |

先发 PyPI/npm/crates/NuGet，再发 Maven Central（流程最长），最后打 Go tag。

## 2. PyPI（推荐 Trusted Publishing）

1. 注册 <https://pypi.org>，开启 2FA。
2. **重要**：PyPI 的 pending publisher 以
   `(owner, repository, workflow, environment)` 唯一。同一个 monorepo 不能给
   多个未创建项目复用同一组配置，否则第三个会报
   `A pending trusted publisher matching this configuration has already been
   registered for a different project name`。每个包使用独立的 environment：

   | PyPI Project Name | Environment |
   |-------------------|-------------|
   | `houyidb` | `pypi` |
   | `langchain-houyidb` | `pypi-langchain` |
   | `llama-index-vector-stores-houyidb` | `pypi-llama` |

   三个项目的 Owner 都是 `houyidb-community`，Repository 都是 `houyidb`，
   Workflow 都是 `publish-sdks.yml`。
3. 在 GitHub 仓库 Settings → Environments 创建
   `pypi`、`pypi-langchain`、`pypi-llama`（无需 secret）。
4. 推送 `v*` tag 或手动运行 `Publish SDKs` 并勾选 publish。
5. 本地手动发布（备用）：

```bash
python3 -m build
python3 -m twine upload dist/*
```

## 3. npm

1. 注册 <https://www.npmjs.com>，开启 2FA。
2. 创建 **Granular Access Token**（Read and Publish，限定 `houyidb` 与
   `houyidb-mcp`），或使用 `npm login`。
3. 在 GitHub 仓库添加 secret `NPM_TOKEN`。
4. 本地：

```bash
cd sdk/typescript/houyidb
npm ci && npm run build
npm publish --provenance --access public
```

MCP 包：

```bash
cd mcp/houyidb-mcp
npm ci && npm run build
npm publish --access public
```

> npm 包名一旦发布不可撤回（只能 deprecate）；发布前先 `npm pack` 检查
> tarball 内容。

## 4. crates.io

1. 用 GitHub 登录 <https://crates.io>。
2. Account Settings → API Tokens → 创建 token。
3. 在 GitHub 仓库添加 secret `CARGO_REGISTRY_TOKEN`。
4. 本地：

```bash
cargo login <token>
cargo publish -p houyidb-client
```

> 首次发布会占用 `houyidb-client` 名称；版本号不可覆盖。

## 5. NuGet（Trusted Publishing，无需 API Key）

1. 登录 <https://www.nuget.org>，点击用户名 → **Trusted Publishing** →
   **Add policy**。
2. 填写 GitHub 策略：
   - Owner：你的 nuget.org 个人账号（或拥有 `HouyiDb` 的组织）
   - Repository Owner：`houyidb-community`
   - Repository：`houyidb`
   - Workflow File：`publish-sdks.yml`（只写文件名）
   - Environment：`nuget`
   - Scope：允许 **Push new packages and package versions**，Glob 填
     `HouyiDb`（如果界面要求通配，可用 `HouyiDb*`）
3. 在 GitHub 仓库添加 secret `NUGET_USER`，值是 nuget.org 的**profile
   name**，不是邮箱。
4. 不需要 `NUGET_API_KEY`；workflow 里的 `NuGet/login` 会用 GitHub OIDC
   换一个 1 小时有效的临时 key。
5. 本地只做打包验证：

```bash
cd sdk/dotnet
dotnet pack src/HouyiDb/HouyiDb.csproj -c Release -o dist
```

> 2FA 不需要关闭；Trusted Publishing 在 CI 里不要求 OTP。策略首次创建后
> 可能处于 7 天临时激活期，第一次成功发布后会转为永久。

## 6. Maven Central（Sonatype Central Portal）

1. 注册 <https://central.sonatype.com>。
2. 创建 namespace `top.houyidb`；Portal 会给出 DNS TXT 记录，加到
   `houyidb.top` 的 DNS，等待验证通过。
3. 生成 GPG 密钥并发布公钥：

```bash
gpg --gen-key
gpg --keyserver keys.openpgp.org --send-keys <KEY_ID>
```

4. 构建并签名（在 `sdk/java/houyidb/build.gradle` 增加 `signing` 插件，
   使用 `ORG_GRADLE_PROJECT_signingKey` / `signingPassword` 环境变量；
   不要把私钥写进仓库）。
5. 生成 bundle 并上传到 Central Portal：

```bash
cd sdk/java/houyidb
gradle publishAllPublicationsToLocalBuildRepository
# 将 build/repo 打成 zip，通过 Central Portal 的 Upload Bundle 上传
```

6. Portal 验证通过后点击 Publish；通常在数分钟内出现在 Maven Central。

> 如果不想维护 GPG/Central 流程，可先只发 PyPI/npm/NuGet/crates，Maven
> Central 后补。Java 用户仍可通过 GitHub Release 的 jar 使用。

## 7. Go 子模块

Go 没有中心化包上传；发布后打子模块 tag：

```bash
git tag sdk/go/houyidb/v0.2.0-beta.8
git push origin sdk/go/houyidb/v0.2.0-beta.8
```

用户使用：

```bash
go get github.com/houyidb-community/houyidb/sdk/go/houyidb@v0.2.0-beta.8
```

> 仓库根 tag `v0.2.0-beta.8` 不会自动成为子模块版本；必须使用
> `sdk/go/houyidb/vX.Y.Z` 形式。

## 8. 发布后验证

```bash
pip index versions houyidb
npm view houyidb version
cargo search houyidb-client
dotnet package search HouyiDb
curl -s "https://search.maven.org/solrsearch/select?q=g:top.houyidb&rows=5&wt=json"
```

再分别安装一次并跑最小示例：

```bash
python3 -m venv /tmp/houyi-sdk && /tmp/houyi-sdk/bin/pip install houyidb
npm install houyidb
```

## 9. 安全规则

- Token 只放 GitHub Secrets 或本机环境变量，绝不进入仓库、日志或截图；
- 所有平台开启 2FA；Token 最小权限、定期轮换；
- 发布前确认版本号一致：Python `0.2.0b8`、npm/Java/.NET/Rust
  `0.2.0-beta.8`、Go tag `v0.2.0-beta.8`；
- 发布是不可逆动作，执行前先 `./scripts/package_sdks.sh` 并检查
  `dist/sdk/` 内容。
