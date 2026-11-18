"""HouyiDB collections HTTP API 客户端。

设计约束（与服务端契约一一对应）：

- collection 在首次 upsert 时隐式创建，维度由首个向量定型；
- payload 键 ``label`` 为保留列，占用会 400（用顶层 ``label`` 字段）；
- 检索响应的 ``score`` 是相似度，按 collection 度量换算（cosine 为
  ``1 - d`` 且钳制 ``[-1,1]``；l2/l1 为 ``1 - d`` 不钳制；ip 为原始点积），
  越大越相似；
- 显式建集合可指定 metric/后端/HNSW 参数；metric 与后端创建后不可变，
  改参数只能删除重建；
- ``include_payload=True`` 时响应携带每命中的 label+payload（键为 id 字符串）；
- delete 按 id 幂等删除，返回真实删除数；
- collection 的标量页 v1 易失（进程重启后 payload/过滤失效，向量索引不受影响）；
- 稀疏词项由客户端给定（无分词器）；``sparse="bm25"`` 必须显式建集合，且仅
  ``hnsw``/``flat`` 后端可用；写入时缺省 ``sparse`` 键表示不改、``[]`` 表示清空；
- 融合检索 ``scores`` 只表达融合排序（RRF 或加权），``dense_scores`` 才按集合
  度量换算，``sparse_scores`` 为 BM25 原始分。

EN: HouyiDB collections HTTP API client. The design constraints mirror the
server contract one-to-one:

- collections are created implicitly on first upsert; dimensions are fixed by
  the first vector;
- the ``label`` payload key is a reserved column; using it returns 400 (use the
  top-level ``label`` field instead);
- ``score`` in search responses is similarity converted per the collection
  metric (cosine is ``1 - d`` clamped to ``[-1,1]``; l2/l1 are ``1 - d``
  unclamped; ip is the raw dot product); larger is more similar;
- explicit creation can set the metric/backend/HNSW parameters; metric and
  backend are immutable afterwards, so changing them requires drop-and-recreate;
- with ``include_payload=True`` each hit carries label+payload (keyed by id
  string);
- delete is idempotent by id and returns the number actually deleted;
- the collection scalar page in v1 is volatile (payload/filters are lost after
  a restart; the vector index is unaffected);
- sparse terms are client-supplied (no tokenizer); ``sparse="bm25"`` requires
  explicit creation and is available on the ``hnsw``/``flat`` backends only; in
  writes an absent ``sparse`` key means unchanged and ``[]`` clears it;
- fusion ``scores`` only expresses fused ranking (RRF or weighted); only
  ``dense_scores`` are converted under the collection metric, and
  ``sparse_scores`` are raw BM25 scores.
"""

from __future__ import annotations

from urllib.parse import quote

import hashlib
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

#: u64 正整数键空间的上半位留空（与服务端 u64 id 对齐，避免符号位歧义）。
#: EN: Leave the upper half of the u64 key space unused (aligned with the
#: server's u64 ids to avoid sign-bit ambiguity).
_U64_MASK = (1 << 63) - 1

#: 稀疏词项 term_id 上界（与服务端 MAX_SPARSE_TERM_ID 一致）。
#: EN: Upper bound of a sparse term id (matches the server's
#: MAX_SPARSE_TERM_ID).
_MAX_SPARSE_TERM_ID = (1 << 24) - 1

#: 写入/查询的原始词项对上限（重复 term 在服务端求和前计数）。
#: EN: Raw pair limits for writes/queries (counted before server-side
#: duplicate-term summation).
_MAX_SPARSE_TERMS_PER_DOC = 1024
_MAX_SPARSE_QUERY_TERMS = 64


def _sparse_terms_to_wire(
    terms: Sequence[Tuple[int, float]], max_pairs: int
) -> List[List[Any]]:
    """把 ``(term_id, weight)`` 序列编码为线格式 ``[[term, weight], …]``。

    客户端先做形状校验而不是依赖服务端：``int(t)`` 对 ``1.9``/``True`` 会静默
    截断成另一个 term，写进 WAL 后无法察觉（服务端只会看到合法 term）。这里
    用 ``operator.index`` 拒绝非整数（含 bool），并校验区间与有限正权重，与
    文档承诺一致；重复 term 的求和仍归服务端。

    EN: Encode a ``(term_id, weight)`` sequence into the wire form
    ``[[term, weight], …]``. Shape validation happens client-side rather than
    relying on the server: ``int(t)`` silently truncates ``1.9``/``True`` into a
    *different* term, which once in the WAL is invisible (the server only sees a
    valid term). ``operator.index`` rejects non-integers (including bool), and
    the bounds/finite-positive-weight checks match the documented contract;
    duplicate-term summation remains server-side.
    """
    if len(terms) > max_pairs:
        raise ValueError(
            f"sparse terms exceed the limit of {max_pairs} raw pairs per request "
            f"(got {len(terms)})"
        )
    out: List[List[Any]] = []
    for item in terms:
        try:
            term, weight = item
        except (TypeError, ValueError):
            raise ValueError(
                f"sparse term must be a (term_id, weight) pair, got {item!r}"
            ) from None
        if isinstance(term, bool):
            raise ValueError(f"sparse term id must be an integer, got bool ({term!r})")
        try:
            term_id = operator.index(term)
        except TypeError:
            raise ValueError(
                f"sparse term id must be an integer, got {type(term).__name__} ({term!r})"
            ) from None
        if not 1 <= term_id <= _MAX_SPARSE_TERM_ID:
            raise ValueError(
                f"sparse term id {term_id} out of range 1..{_MAX_SPARSE_TERM_ID}"
            )
        if isinstance(weight, bool):
            raise ValueError(
                f"sparse weight must be a number, got bool ({weight!r})"
            )
        try:
            weight_value = float(weight)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "sparse weight must be a finite number > 0, got "
                f"{type(weight).__name__} ({weight!r})"
            ) from None
        if not math.isfinite(weight_value) or weight_value <= 0:
            raise ValueError(
                f"sparse weight must be finite and > 0, got {weight_value!r}"
            )
        out.append([term_id, weight_value])
    return out


def stable_id(text: str) -> int:
    """把任意字符串 id 确定性映射为 HouyiDB 的 u64 向量 id。

    adapter 层共用：langchain 的 str ids、llama-index 的 node_id 都经此函数
    落到 u64 键空间；同一输入恒得同一输出，因此 delete(ids) 可以重放同一映射。

    EN: Deterministically map an arbitrary string id to a HouyiDB u64 vector
    id. Shared by the adapters: LangChain str ids and LlamaIndex node_ids land
    in the u64 key space through this function; the same input always yields
    the same output, so delete(ids) can replay the same mapping.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _U64_MASK


class HouyiError(Exception):
    """服务端返回的非 2xx 响应（status_code + 服务端诊断文本）。

    EN: Non-2xx response from the server (status_code + server diagnostic text).
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"houyidb HTTP {status_code}: {message}")


@dataclass
class Vector:
    """一条待写入向量：id + 分量 + 可选 label/payload/sparse 词项。

    ``sparse`` 仅对 ``sparse="bm25"`` 的集合有效：``None``（缺省）表示不改该
    文档的稀疏词项，``[]`` 表示清空；每项为 ``(term_id, weight)``，term_id
    取值 ``[1, 2^24-1]``、weight 有限且为正，重复 term 由服务端求和。

    EN: A vector to write: id + components + optional label/payload/sparse
    terms. ``sparse`` only applies to ``sparse="bm25"`` collections: ``None``
    (default) leaves the document's sparse terms unchanged, while ``[]``
    clears them; each item is ``(term_id, weight)`` with term_id in
    ``[1, 2^24-1]`` and a finite positive weight; duplicate terms are summed by
    the server.
    """

    id: int
    values: Sequence[float]
    label: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = None
    sparse: Optional[Sequence[Tuple[int, float]]] = None

    def to_json(self) -> Dict[str, Any]:
        doc: Dict[str, Any] = {"id": self.id, "values": list(self.values)}
        if self.label is not None:
            doc["label"] = self.label
        if self.payload:
            doc["payload"] = dict(self.payload)
        if self.sparse is not None:
            doc["sparse"] = _sparse_terms_to_wire(self.sparse, _MAX_SPARSE_TERMS_PER_DOC)
        return doc


@dataclass
class Filter:
    """一条结构化过滤条件（多条之间取交集）。

    op 白名单与服务端一致：``eq`` / ``gt`` / ``lt`` / ``gte`` / ``lte`` / ``in``。

    EN: A structured filter condition (multiple conditions intersect). The op
    whitelist matches the server: ``eq`` / ``gt`` / ``lt`` / ``gte`` / ``lte``
    / ``in``.
    """

    column: str
    op: str
    value: Any

    def to_json(self) -> Dict[str, Any]:
        return {"column": self.column, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class HnswParams:
    """建集合时的 HNSW 参数三元组（必须三个全给，否则服务端 400）。

    EN: HNSW parameter triple for explicit creation (all three are required or
    the server returns 400).
    """

    m: int
    ef_construction: int
    ef_search: int


@dataclass
class SearchHit:
    """单条命中：id + 相似度分数 + 可选 payload（include_payload 时）。

    EN: A single hit: id + similarity score + optional payload (with
    include_payload).
    """

    id: int
    score: float
    payload: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """检索响应的数据类化：hits 按相似度从高到低排列。

    EN: Typed search response: hits ordered by similarity, highest first.
    """

    hits: List[SearchHit] = field(default_factory=list)

    @property
    def ids(self) -> List[int]:
        return [h.id for h in self.hits]

    @property
    def scores(self) -> List[float]:
        return [h.score for h in self.hits]

    @classmethod
    def from_json(cls, body: Mapping[str, Any]) -> "SearchResult":
        ids: Sequence[int] = body.get("ids", [])
        scores: Sequence[float] = body.get("scores", [])
        payloads: Optional[Mapping[str, Any]] = body.get("payloads")
        hits = [
            SearchHit(
                id=i,
                score=s,
                payload=dict(payloads[str(i)]) if payloads and str(i) in payloads else None,
            )
            for i, s in zip(ids, scores)
        ]
        return cls(hits=hits)


@dataclass
class FusionHit:
    """单条融合命中：id + 融合分 + 两路原始分 + 可选 payload。

    融合分只表达排序（RRF 或加权），不属于任何度量口径；``dense_score`` 按
    集合度量换算、仅稠密一路命中时存在；``sparse_score`` 为 BM25 原始分、
    仅稀疏一路命中时存在。

    EN: One fusion hit: id + fused score + both raw modality scores + optional
    payload. The fused score only expresses ranking (RRF or weighted) and
    belongs to no metric; ``dense_score`` is converted under the collection
    metric and exists only when the dense path hit; ``sparse_score`` is the
    raw BM25 score and exists only when the sparse path hit.
    """

    id: int
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None


@dataclass
class FusionResult:
    """融合检索响应：hits 按融合分从高到低排列，附生效参数回显。

    ``metric`` 仅用于解释 ``dense_score``；``fusion_mode`` 为生效模式
    （``rrf`` | ``weighted``）；``rrf_k`` 仅 rrf 模式回显、``alpha`` 仅
    weighted 模式回显。

    EN: Fusion response: hits ordered by fused score, highest first, with the
    effective parameters echoed. ``metric`` explains ``dense_score`` only;
    ``fusion_mode`` is the effective mode (``rrf`` | ``weighted``);
    ``rrf_k`` is echoed only under rrf and ``alpha`` only under weighted.
    """

    hits: List[FusionHit] = field(default_factory=list)
    metric: str = ""
    fusion_mode: str = ""
    rrf_k: Optional[int] = None
    alpha: Optional[float] = None

    @property
    def ids(self) -> List[int]:
        return [h.id for h in self.hits]

    @property
    def scores(self) -> List[float]:
        return [h.score for h in self.hits]

    @classmethod
    def from_json(cls, body: Mapping[str, Any]) -> "FusionResult":
        ids: Sequence[int] = body.get("ids", [])
        scores: Sequence[float] = body.get("scores", [])
        dense_scores: Sequence[Optional[float]] = body.get("dense_scores", [])
        sparse_scores: Sequence[Optional[float]] = body.get("sparse_scores", [])
        # 融合分是排序键：宽度错位说明响应被截断/损坏，zip 会静默丢命中或
        # 错配分数，因此与 Rust/TypeScript SDK 同口径显式失败；模态分数组
        # 缺席（空）仍容忍，出现时必须等宽。
        # EN: Fused scores are the ranking key; misaligned widths mean a
        # truncated/corrupt response where zip would silently drop hits or
        # mispair scores, so fail explicitly like the Rust/TypeScript SDKs.
        # Modality arrays absent (empty) stay tolerated but must match when
        # present.
        id_count = len(ids)
        if len(scores) != id_count:
            raise ValueError(
                f"fusion response scores has {len(scores)} entries for {id_count} ids"
            )
        for name, arr in (("dense_scores", dense_scores), ("sparse_scores", sparse_scores)):
            if len(arr) not in (0, id_count):
                raise ValueError(
                    f"fusion response {name} has {len(arr)} entries for {id_count} ids"
                )
        payloads: Optional[Mapping[str, Any]] = body.get("payloads")
        fusion: Mapping[str, Any] = body.get("fusion") or {}
        hits = [
            FusionHit(
                id=i,
                score=s,
                dense_score=(
                    dense_scores[idx] if idx < len(dense_scores) else None
                ),
                sparse_score=(
                    sparse_scores[idx] if idx < len(sparse_scores) else None
                ),
                payload=dict(payloads[str(i)]) if payloads and str(i) in payloads else None,
            )
            for idx, (i, s) in enumerate(zip(ids, scores))
        ]
        return cls(
            hits=hits,
            metric=str(body.get("metric", "")),
            fusion_mode=str(fusion.get("mode", "")),
            rrf_k=fusion.get("rrf_k"),
            alpha=fusion.get("alpha"),
        )


class Collection:
    """一个 named collection 的操作句柄（经 :class:`HouyiClient.collection` 获取）。

    EN: Operation handle for one named collection (obtained via
    :class:`HouyiClient.collection`).
    """

    def __init__(self, client: "HouyiClient", name: str):
        self._client = client
        self.name = name
        # URL 路径段必须转义：`requests` 原样保留 `/`/`?`/`#`——含这些字符的名字
        # 会改写路由结构，表现为莫名其妙的 404 且 collection 永远建不出来。
        # EN: The URL path segment must be escaped: `requests` keeps `/`, `?`,
        # and `#` as-is — names containing them rewrite the route structure,
        # showing up as baffling 404s and a collection that can never be created.
        self._quoted = quote(name, safe="")

    def upsert(self, vectors: Iterable[Vector | Mapping[str, Any]]) -> int:
        """写入一批向量（collection 首次写入时隐式创建）。返回插入条数。

        EN: Write a batch of vectors (the collection is created implicitly on
        the first write). Returns the number of inserted rows.
        """
        docs = [v.to_json() if isinstance(v, Vector) else dict(v) for v in vectors]
        body = self._client._post(f"/v1/collections/{self._quoted}/upsert", {"vectors": docs})
        return int(body["inserted"])

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        ef_search: Optional[int] = None,
        include_payload: bool = False,
    ) -> SearchResult:
        """纯向量 top-k 检索。

        EN: Pure vector top-k search.
        """
        req: Dict[str, Any] = {"vector": list(vector), "k": k}
        if ef_search is not None:
            req["ef_search"] = ef_search
        if include_payload:
            req["include_payload"] = True
        body = self._client._post(f"/v1/collections/{self._quoted}/search", req)
        return SearchResult.from_json(body)

    def hybrid_search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        filters: Optional[Iterable[Filter | Mapping[str, Any]]] = None,
        ef_search: Optional[int] = None,
        include_payload: bool = False,
    ) -> SearchResult:
        """向量 × 标量过滤的混合检索（filters 取交集，仅 hnsw/flat 后端支持）。

        EN: Hybrid vector x scalar-filter search (filters intersect; only the
        hnsw/flat backends support this).
        """
        req: Dict[str, Any] = {"vector": list(vector), "k": k}
        if filters:
            req["filters"] = [
                f.to_json() if isinstance(f, Filter) else dict(f) for f in filters
            ]
        if ef_search is not None:
            req["ef_search"] = ef_search
        if include_payload:
            req["include_payload"] = True
        body = self._client._post(f"/v1/collections/{self._quoted}/hybrid/search", req)
        return SearchResult.from_json(body)

    def fusion_search(
        self,
        vector: Sequence[float],
        sparse: Sequence[Tuple[int, float]],
        k: int,
        *,
        filters: Optional[Iterable[Filter | Mapping[str, Any]]] = None,
        ef_search: Optional[int] = None,
        include_payload: bool = False,
        fusion: Optional[str] = None,
        rrf_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> FusionResult:
        """稠密 + 稀疏融合检索（仅 ``sparse="bm25"`` 的集合；两路均需非空）。

        两路各过取 4×k 后融合：``fusion="rrf"``（缺省，``rrf_k`` 缺省 60）按
        倒数排名相加；``fusion="weighted"`` 在过滤后候选上各路 min-max 归一后
        按 ``alpha*dense + (1-alpha)*sparse`` 相加（``alpha`` 缺省 0.5）。稀疏
        查询每项为 ``(term_id, weight)``，最多 64 个原始对。filters 取交集，
        仅 hnsw/flat 后端支持。

        EN: Dense + sparse fusion search (only for ``sparse="bm25"``
        collections; both modalities must be non-empty). Both paths over-fetch
        4x k, then fuse: ``fusion="rrf"`` (default, ``rrf_k`` defaults to 60)
        sums reciprocal ranks; ``fusion="weighted"`` min-max normalizes each
        modality over the filtered candidates and sums
        ``alpha*dense + (1-alpha)*sparse`` (``alpha`` defaults to 0.5). Sparse
        query items are ``(term_id, weight)`` with at most 64 raw pairs.
        Filters intersect and are supported on hnsw/flat backends only.
        """
        req: Dict[str, Any] = {
            "vector": list(vector),
            "sparse": _sparse_terms_to_wire(sparse, _MAX_SPARSE_QUERY_TERMS),
            "k": k,
        }
        if filters:
            req["filters"] = [
                f.to_json() if isinstance(f, Filter) else dict(f) for f in filters
            ]
        if ef_search is not None:
            req["ef_search"] = ef_search
        if include_payload:
            req["include_payload"] = True
        if fusion is not None:
            req["fusion"] = fusion
        if rrf_k is not None:
            req["rrf_k"] = rrf_k
        if alpha is not None:
            req["alpha"] = alpha
        body = self._client._post(
            f"/v1/collections/{self._quoted}/fusion/search", req
        )
        return FusionResult.from_json(body)

    def delete(
        self,
        ids: Optional[Iterable[int]] = None,
        *,
        filters: Optional[Iterable[Filter | Mapping[str, Any]]] = None,
    ) -> int:
        """按 id 和/或过滤条件删除（幂等；两者取并集）。返回真实删除数。

        按条件删除由服务端在标量页上求值：标量行随 upsert 同步落页，因此
        刚写入、尚在异步构图队列中的向量也能被命中删除——这是「先检索拿 id
        再删」无法保证的（检索在队列排空前看不到新写入）。

        EN: Delete by id and/or filters (idempotent; the two sets are unioned).
        Returns the number of rows actually deleted.

        Filter deletion is evaluated by the server on the scalar page: scalar
        rows land on the page synchronously with upsert, so vectors that were
        just written and are still in the async build queue can be matched and
        deleted — something "search for ids, then delete" cannot guarantee
        (search does not see new writes until the queue drains).
        """
        id_list = [int(i) for i in ids] if ids is not None else []
        filter_list = (
            [f.to_json() if isinstance(f, Filter) else dict(f) for f in filters]
            if filters
            else []
        )
        if not id_list and not filter_list:
            raise ValueError("delete requires at least one of ids or filters")
        body = self._client._post(
            f"/v1/collections/{self._quoted}/delete",
            {"ids": id_list, "filters": filter_list},
        )
        return int(body["deleted"])

    def status(self) -> Dict[str, Any]:
        """collection 状态（indexed/pending/dims/index_type/scalar_row_count）。

        EN: Collection status (indexed/pending/dims/index_type/scalar_row_count).
        """
        return self._client._get(f"/v1/collections/{self._quoted}/status")


class HouyiClient:
    """HouyiDB 客户端入口。

    :param base_url: 服务地址，如 ``http://127.0.0.1:7000``。
    :param timeout: 单次请求超时秒数（连接+读取）。
    :param session: 可选的 ``requests.Session``（连接池复用/自定义适配器）。
    :param admin_token: 可选管理 token，只附在 ``/v1/admin/*`` 请求上。

    EN: HouyiDB client entry point.

    :param base_url: Server address, e.g. ``http://127.0.0.1:7000``.
    :param timeout: Per-request timeout in seconds (connect + read).
    :param session: Optional ``requests.Session`` (connection pooling/custom
        adapters).
    :param admin_token: Optional admin token attached only to ``/v1/admin/*``
        requests.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7000",
        *,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
        admin_token: Optional[str] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._admin_token = admin_token

    def create_collection(
        self,
        name: str,
        dims: int,
        *,
        metric: Optional[str] = None,
        index_type: Optional[str] = None,
        hnsw: Optional[HnswParams] = None,
        vector_dtype: Optional[str] = None,
        quantization: Optional[str] = None,
        sparse: Optional[str] = None,
    ) -> Dict[str, Any]:
        """显式创建 collection（可指定度量/后端/HNSW 参数/存储编码；需要写权限）。

        与隐式创建（首次 upsert）不同，dims、metric、后端与 HNSW 参数在这里
        一次定型并由服务端回显；metric 与后端创建后不可变，改参数只能删除
        重建。普通写路径不带管理 token；同名已存在抛 ``HouyiError(409)``、
        参数非法 400、follower/只读 503。

        EN: Explicitly create a collection (metric/backend/HNSW parameters are
        settable; requires write permission). Unlike implicit creation (first
        upsert), dims, metric, backend and HNSW parameters are fixed here and
        echoed back; metric and backend are immutable afterwards, so changing
        them requires drop-and-recreate. This is the normal write path without
        the admin token; an existing name raises ``HouyiError(409)``, invalid
        parameters 400, and followers/read-only nodes 503.

        ``vector_dtype`` 可选 ``"f32"``（缺省）或 ``"f16"``（仅 hnsw，省一半
        向量内存）；创建后不可变，非 f32 与非 hnsw 组合由服务端拒绝（400）。
        EN: ``vector_dtype`` may be ``"f32"`` (default) or ``"f16"`` (hnsw only,
        halves vector memory); it is immutable after creation, and non-f32
        with a non-hnsw backend is rejected by the server (400).

        ``quantization`` 可选 ``"none"``（缺省）、``"int8"`` 或 ``"rabitq"``
        （仅 hnsw，压缩图内距离工作集；主表保留原始向量并在检索时精确重排）。
        rabitq 为 1 bit/dim，仅支持 metric l2|cosine（ip/l1 返回 400）；创建后
        不可变，非 none 与非 hnsw 组合由服务端拒绝（400）。
        EN: ``quantization`` may be ``"none"`` (default), ``"int8"``, or
        ``"rabitq"`` (hnsw only, shrinks the in-graph distance working set; the
        main store keeps raw vectors and reranks exactly at search time). rabitq
        packs 1 bit/dim and supports metric l2|cosine only (ip/l1 return 400);
        it is immutable after creation, and non-none with a non-hnsw backend is
        rejected by the server (400).

        ``sparse`` 可选 ``"none"``（缺省）或 ``"bm25"``（仅 hnsw/flat 后端），
        开启后集合获得倒排稀疏索引并支持 :meth:`Collection.fusion_search`；
        创建后不可变，隐式创建的集合恒为 none。响应回显生效值；旧服务端响应
        可能缺少该字段，调用方应按可选字段容忍。
        EN: ``sparse`` may be ``"none"`` (default) or ``"bm25"`` (hnsw/flat
        backends only); when enabled the collection gains an inverted sparse
        index and supports :meth:`Collection.fusion_search`; it is immutable
        after creation, and implicitly created collections are always none. The
        response echoes the effective value; older servers may omit the field,
        so callers should treat it as optional.
        """
        # dims 一旦建集合就不可变（只能删除重建），因此绝不做 int() 截断：
        # 3.9→3、True→1 都会把调用方笔误固化成永久错误的维度。这里与
        # bool 显式区分（bool 是 int 子类）并在本地 fail-fast（OCR #12）。
        # EN: dims is immutable once the collection exists (only drop-and-recreate
        # can change it), so never truncate via int(): 3.9 -> 3 or True -> 1 would
        # fossilize a caller typo into a permanently wrong dimension. Distinguish
        # bool explicitly (bool is an int subclass) and fail fast locally (OCR #12).
        if not isinstance(dims, int) or isinstance(dims, bool) or dims <= 0:
            raise ValueError(f"dims must be a positive int (got {dims!r})")
        payload: Dict[str, Any] = {"dims": dims}
        if metric is not None:
            payload["metric"] = metric
        if index_type is not None:
            payload["index_type"] = index_type
        if vector_dtype is not None:
            payload["vector_dtype"] = vector_dtype
        if quantization is not None:
            payload["quantization"] = quantization
        if sparse is not None:
            payload["sparse"] = sparse
        if hnsw is not None:
            payload["hnsw_m"] = hnsw.m
            payload["hnsw_ef_construction"] = hnsw.ef_construction
            payload["hnsw_ef_search"] = hnsw.ef_search
        return self._put(f"/v1/collections/{quote(name, safe='')}", payload)

    def delete_collection(self, name: str) -> None:
        """删除整个 collection（不可逆；需要 admin 权限）。

        服务端 WAL-first：记录先落盘，崩溃重放与 follower 复制都收敛到
        「已删除」；不存在或已删除的 collection 抛 ``HouyiError(404)``。

        EN: Drop an entire collection (irreversible; admin permission required).
        The server is WAL-first, so crash replay and follower replication
        converge to "gone"; unknown/already-dropped collections raise
        ``HouyiError(404)``.
        """
        path = f"/v1/admin/collections/{quote(name, safe='')}"
        resp = self._session.delete(
            f"{self._base_url}{path}",
            timeout=self._timeout,
            headers=self._admin_headers(),
        )
        self._checked(resp)

    def collection(self, name: str) -> Collection:
        return Collection(self, name)

    def health(self) -> bool:
        try:
            body = self._get("/v1/health")
        except (HouyiError, requests.RequestException):
            return False
        return body.get("status") == "ok"

    def list_collections(self) -> List[Dict[str, Any]]:
        return list(self._get("/v1/collections"))

    def index_status(self) -> Dict[str, Any]:
        """主索引双速写入指标（hot/pending/flush_count 等，可观测性用）。

        EN: Primary index dual-speed write metrics (hot/pending/flush_count,
        etc.; for observability).
        """
        return self._get("/v1/index/status")

    # ── 内部传输 ──────────────────────────────────────────
    # EN: Internal transport.

    def _get(self, path: str) -> Any:
        resp = self._session.get(f"{self._base_url}{path}", timeout=self._timeout)
        return self._checked(resp)

    def _admin_headers(self) -> Dict[str, str]:
        """管理请求的 Authorization 头；未配置 token 时返回空表。

        EN: Authorization header for admin requests; empty when no token is set.
        """
        if self._admin_token:
            return {"Authorization": f"Bearer {self._admin_token}"}
        return {}

    def _post(self, path: str, payload: Mapping[str, Any]) -> Any:
        resp = self._session.post(
            f"{self._base_url}{path}", json=payload, timeout=self._timeout
        )
        return self._checked(resp)

    def _put(self, path: str, payload: Mapping[str, Any]) -> Any:
        resp = self._session.put(
            f"{self._base_url}{path}", json=payload, timeout=self._timeout
        )
        return self._checked(resp)

    @staticmethod
    def _checked(resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            # 服务端错误体是纯文本诊断（axum 元组错误），原样透出。
            # EN: The server error body is a plain-text diagnostic (axum tuple
            # error); surface it verbatim.
            raise HouyiError(resp.status_code, resp.text.strip() or resp.reason)
        return resp.json()
