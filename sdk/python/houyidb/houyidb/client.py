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
- collection 的标量页 v1 易失（进程重启后 payload/过滤失效，向量索引不受影响）。

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
  a restart; the vector index is unaffected).
"""

from __future__ import annotations

from urllib.parse import quote

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import requests

#: u64 正整数键空间的上半位留空（与服务端 u64 id 对齐，避免符号位歧义）。
#: EN: Leave the upper half of the u64 key space unused (aligned with the
#: server's u64 ids to avoid sign-bit ambiguity).
_U64_MASK = (1 << 63) - 1


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
    """一条待写入向量：id + 分量 + 可选 label/payload。

    EN: A vector to write: id + components + optional label/payload.
    """

    id: int
    values: Sequence[float]
    label: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        doc: Dict[str, Any] = {"id": self.id, "values": list(self.values)}
        if self.label is not None:
            doc["label"] = self.label
        if self.payload:
            doc["payload"] = dict(self.payload)
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
    ) -> Dict[str, Any]:
        """显式创建 collection（可指定度量/后端/HNSW 参数；需要写权限）。

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
