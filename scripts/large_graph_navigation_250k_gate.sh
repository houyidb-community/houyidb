#!/usr/bin/env bash
# 250k 大图导航门禁（本地/nightly 纪律项，默认不进每次提交的 `cargo test --all`）。
# EN: 250k large-graph navigation gate (local/nightly discipline, not part of the
# EN: per-commit `cargo test --all`).
#
# 背景：50k 生产形态门禁只能覆盖 layer-1 ~3.1k 节点的导航；250k×512 / 64 簇
# （每簇 ~3906 点、layer-1 ~15.6k 节点）才能压出「上层已能进入正确簇，但
# L0 束宽小于簇规模时簇内 top-k 仍会丢」的规模效应。本门禁断言：
#   1. ef=16…4096 的所有返回 id 必须留在查询自己的簇（不允许跨簇错落）；
#   2. ef=8192 均值召回 ≥0.99、无查询低于 0.9（图结构完整，宽束可恢复）。
# 250k 实测：低 ef 的 q8 召回可为 0，但返回 id 始终在簇 8；ef=8192 恢复 1.0。
# EN: Background: the 50k production-shaped gate only covers ~3.1k layer-1
# EN: nodes; 250k x 512 / 64 clusters (~3906 points per cluster, ~15.6k layer-1
# EN: nodes) is the scale where upper navigation already lands in the correct
# EN: cluster while an L0 beam smaller than the cluster still loses intra-cluster
# EN: top-k. This gate asserts: (1) every returned id at ef=16..4096 stays in
# EN: the query's own cluster (no cross-cluster landings); (2) at ef=8192 mean
# EN: recall >= 0.99 with zero below-floor queries (structurally complete graph,
# EN: recoverable with a wide beam). Measured at 250k: low-ef q8 recall can be 0
# EN: but its ids stay in cluster 8; ef=8192 restores 1.0000.
#
# 用法 / Usage（仓库根目录 / from repo root）:
#   ./scripts/large_graph_navigation_250k_gate.sh
#
# 环境纪律：清空会改写构图/查询行为的进程级开关（与 50k 门禁同口径），
# 用例自身再钉 PLAN_APPLY=0 / GROUP_REPAIR=4,16 / ADAPTIVE=1。
# EN: Env discipline: strip the process-wide build/search switches (same as the
# 50k gate); the case itself pins PLAN_APPLY=0 / GROUP_REPAIR=4,16 /
# ADAPTIVE=1.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG="$(mktemp "${TMPDIR:-/tmp}/large_graph_navigation_250k.XXXXXX")"
trap 'rm -f "${LOG}"' EXIT

echo "==> large_graph_navigation_250k: release, prod-shape 250k x 512 (ignore-only)"
env -u HOUYI_HNSW_PLAN_APPLY -u HOUYI_HNSW_GROUP_REPAIR -u HOUYI_HNSW_UPPER_EF \
  -u HOUYI_HNSW_UPPER_EF_ADAPTIVE -u HOUYI_HNSW_BULK_EF_CONSTRUCTION \
  -u HOUYI_HNSW_KEEP_PRUNED -u HOUYI_HNSW_RERANK_EXPAND cargo test --release -p houyi-vector \
  --test sm_legacy_repro large_graph_upper_navigation_keeps_cluster_at_quarter_million \
  -- --ignored --nocapture --test-threads=1 | tee "${LOG}"

# 用例必须真实执行并通过：过滤词写错或用例改名会让输出只剩 0 条 `ok` 行。
# EN: The case must really run and pass: a typo in the filter or a renamed case
# would leave zero `ok` lines.
if ! grep -Fq "test large_graph_upper_navigation_keeps_cluster_at_quarter_million ... ok" "${LOG}"; then
  echo "missing passing line for large_graph_upper_navigation_keeps_cluster_at_quarter_million" >&2
  exit 1
fi
echo "==> large_graph_navigation_250k: pass"
