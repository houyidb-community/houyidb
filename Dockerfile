# HouyiDB 一体化镜像：多阶段构建，运行时仅含二进制 + curl（健康检查）。
#
#   docker build -t houyidb .
#   docker run -d -p 7000:7000 -p 5432:5432 -v houyidb-data:/data houyidb
#
# P4.8 起无原生 C 依赖（NUMA 拓扑改读 sysfs，移除未维护的 hwloc crate）。

# ---------- 构建阶段 ----------
FROM rust:1-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# .dockerignore 已排除 target/.git/docs/SDK 构建产物；整仓拷贝换取最简单可靠的构建，
# 代价是源码任何改动都会使依赖层缓存失效（无 cargo-chef）。
COPY . .

# 默认 portable 路径：无 SIMD 编译期特化，运行时跨 CPU 通用（amd64/arm64 均可）。
# 如需 AVX2 构建：docker build --build-arg CARGO_FEATURES=simd-avx2 .
#
# 构建上下文排除了 `.git`（.dockerignore），build.rs 的 git 探测必然失败；
# CI/本地要嵌入提交标识必须显式传入（镜像 `/v1/version` 的 build_sha 来源）。
# EN: The build context excludes `.git` (.dockerignore), so build.rs cannot
# probe git. Pass the commit identity explicitly to embed it as `/version`'s
# build_sha.
ARG CARGO_FEATURES=""
ARG HOUYIDB_GIT_SHA=""
RUN if [ -n "$CARGO_FEATURES" ]; then \
        cargo build --release -p houyi-server --features "$CARGO_FEATURES"; \
    else \
        cargo build --release -p houyi-server; \
    fi

# ---------- 运行阶段 ----------
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home houyidb \
    && mkdir -p /data && chown houyidb:houyidb /data

COPY --from=builder /build/target/release/houyi-server /usr/local/bin/houyi-server

USER houyidb
VOLUME ["/data"]

# 容器内必须监听 0.0.0.0（默认 127.0.0.1 只在容器内可达）；WAL 落 /data 卷持久化。
ENV HOUYI_SERVER_BENCHMARK_HTTP_ADDR=0.0.0.0:7000 \
    HOUYI_SERVER_PGWIRE_ADDR=0.0.0.0:5432 \
    HOUYI_SERVER_WAL_PATH=/data/houyidb.wal

EXPOSE 7000 5432

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7000/v1/health || exit 1

CMD ["houyi-server"]
