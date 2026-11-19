#!/usr/bin/env bash
# 50k 大簇召回门禁（本地纪律项，默认不进每次提交的 `cargo test --all`）。
# EN: 50k large-cluster recall gate (local discipline, not part of the
# EN: per-commit `cargo test --all`).
#
# 背景：50k×512 / 64 簇（每簇 ~781 点）下，自适应组宽 ~195 仍小于簇规模，
# 默认 plan/apply 与 legacy 串行两条构图路径都曾出现整簇孤岛：失败查询返回
# 完全错误的簇（相似度 0.03~0.10，真值 0.95），加大 ef 几乎不改善。
# 本脚本以 release 跑四个 `#[ignore]` 用例（debug 下 50k 暴力真值 + 构图需
# 数分钟，超出每次提交的分钟级预算）：
#   1. `plan_apply_grouped_build_keeps_clustered_recall_50k`（默认 plan/apply +
#      500 条/批）；
#   2. `legacy_serial_build_keeps_clustered_recall_50k`（=0 + 500 条/批的逐点
#      串行路径）；
#   3. `legacy_http_batches_keep_clustered_recall_50k`（=0 + 100 条/批的 HTTP
#      形态：worker 阈值式排空，覆盖 300 行串行与 600 行分组两种 drain）；
#   4. `legacy_prod_shape_batches_keep_all_layer_recall_50k`（=0 + 精确复刻 sm
#      生产 mulberry32 语料 + 服务端 collection 的真实 BuildConfig 默认值 +
#      约 300 行/tick 的 worker 节奏；断言全层可达 100%、每个 ef 均值 ≥0.99、
#      q45/q62 在每个 ef ≥0.9，仅允许 q30 的 TIE_EPS 并列噪声低于下限）。
# 用例 1–3 沿用 4653 口径：level-0 可达率 ≥95%（用例 3 额外要求 100%）、
# EF_SWEEP 每点均值 ≥RECALL_MIN，且每个查询在每个 ef 都不得低于单查询下限
# 0.9（ef 扫描已真正钉到底层束宽，低 ef 不再复用推断默认，零容差）。
# 用例 4 更严：逐层 100% 可达、每个 ef 均值 ≥0.99，且只放行 q30 的并列近邻
# 噪声，避免把「另一个破簇」藏在低 ef 容差里。
# EN: Background: at 50k x 512 / 64 clusters (~781 points per cluster) the
# EN: adaptive group width ~195 stays below the cluster size, and both the
# EN: default plan/apply and the legacy serial build paths showed whole-cluster
# EN: islands: failing queries returned a completely wrong cluster (similarity
# EN: 0.03-0.10 vs truth 0.95) and raising ef barely helped. This script runs
# EN: four `#[ignore]` cases in release (the 50k brute-force truth plus build
# EN: take minutes in debug, beyond the per-commit minute budget):
# EN:   1. `plan_apply_grouped_build_keeps_clustered_recall_50k` (default
# EN:      plan/apply + 500-row requests);
# EN:   2. `legacy_serial_build_keeps_clustered_recall_50k` (=0 + 500-row
# EN:      requests, the per-point serial path);
# EN:   3. `legacy_http_batches_keep_clustered_recall_50k` (=0 + 100-row
# EN:      HTTP-shaped requests drained at the worker threshold, covering both
# EN:      a 300-row serial drain and a 600-row grouped drain);
# EN:   4. `legacy_prod_shape_batches_keep_all_layer_recall_50k` (=0 + an exact
# EN:      replica of sm's production mulberry32 corpus + the real collection
# EN:      `BuildConfig` defaults + ~300 rows per worker tick; asserts 100%
# EN:      all-layer reachability, mean >= 0.99 at every ef, and q45/q62 >= 0.9
# EN:      at every ef; only q30's TIE_EPS tied-neighbour noise may fall below).
# EN: Cases 1-3 keep the 4653 semantics: level-0 reachability >= 95% (case 3
# EN: additionally requires 100%), mean >= RECALL_MIN at every EF_SWEEP point,
# EN: and no query may fall below the per-query floor 0.9 at any ef (the ef
# EN: sweep now pins the real beam, so low ef no longer replays the inferred
# EN: default — zero tolerance). Case 4 is stricter: 100% reachability on every
# EN: layer, mean >= 0.99 at every ef, and only q30's tied-neighbour noise is
# EN: allowed below the floor — a second broken cluster cannot hide behind
# EN: low-ef tolerance.
#
# 用法 / Usage（仓库根目录 / from repo root）:
#   ./scripts/clustered_recall_50k_gate.sh
#
# 环境纪律：仅剔除与本门禁相关的进程级开关——`HOUYI_HNSW_PLAN_APPLY` /
# `HOUYI_HNSW_GROUP_REPAIR` / `HOUYI_HNSW_UPPER_EF` /
# `HOUYI_HNSW_UPPER_EF_ADAPTIVE` 由用例自身再钉一遍，`HOUYI_HNSW_BULK_EF_CONSTRUCTION` / `HOUYI_HNSW_KEEP_PRUNED` / `HOUYI_HNSW_RERANK_EXPAND` 未被用例钉住、残留会
# 改写构图/查询行为，也在本脚本清空。其余环境保留，避免破坏 rustup / 自定义
# CARGO_HOME 等工具链设置。
# EN: Env discipline: only the process-wide knobs relevant to this gate are
# EN: stripped — `HOUYI_HNSW_PLAN_APPLY` / `HOUYI_HNSW_GROUP_REPAIR` /
# EN: `HOUYI_HNSW_UPPER_EF` / `HOUYI_HNSW_UPPER_EF_ADAPTIVE` are pinned again
# EN: by the cases themselves, while `HOUYI_HNSW_BULK_EF_CONSTRUCTION` / `HOUYI_HNSW_KEEP_PRUNED` / `HOUYI_HNSW_RERANK_EXPAND` are not pinned by
# EN: the cases and a leftover would change build/search behavior, so the
# EN: script clears them too. Everything else stays so rustup / custom
# EN: CARGO_HOME setups are not broken.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# 模板必须以连续的 X 结尾：BSD/macOS mktemp 要求如此，后缀放在 X 之后会
# 在本机直接报错退出。
# EN: The template must end with the run of X's: BSD/macOS mktemp requires it,
# EN: and a suffix after the X's errors out on those hosts.
LOG="$(mktemp "${TMPDIR:-/tmp}/clustered_recall_50k.XXXXXX")"
trap 'rm -f "${LOG}"' EXIT

echo "==> clustered_recall_50k: release, plan/apply + legacy serial + legacy HTTP batches + prod-shape batches (ignore-only)"
env -u HOUYI_HNSW_PLAN_APPLY -u HOUYI_HNSW_GROUP_REPAIR -u HOUYI_HNSW_UPPER_EF \
  -u HOUYI_HNSW_UPPER_EF_ADAPTIVE -u HOUYI_HNSW_BULK_EF_CONSTRUCTION \
  -u HOUYI_HNSW_KEEP_PRUNED -u HOUYI_HNSW_RERANK_EXPAND cargo test --release \
  -p houyi-vector --test plan_apply_group_recall --test legacy_group_recall \
  --test legacy_http_group_recall --test sm_legacy_repro \
  -- --ignored --nocapture --test-threads=1 50k | tee "${LOG}"

# 四条 50k 用例都必须真实执行并通过：过滤词写错或用例改名会让输出只剩 0 条
# `ok` 行而脚本仍以退出码 0 结束，因此逐行断言（-F 规避正则元字符）。
# EN: All four 50k cases must really run and pass: a typo in the filter or a
# EN: renamed case would leave zero `ok` lines while the script still exits 0,
# EN: so assert each expected line explicitly (-F avoids regex metacharacters).
for name in plan_apply_grouped_build_keeps_clustered_recall_50k \
            legacy_serial_build_keeps_clustered_recall_50k \
            legacy_http_batches_keep_clustered_recall_50k \
            legacy_prod_shape_batches_keep_all_layer_recall_50k; do
  if ! grep -Fq "test ${name} ... ok" "${LOG}"; then
    echo "missing passing line for ${name} in gate output" >&2
    exit 1
  fi
done
echo "==> clustered_recall_50k: pass"
