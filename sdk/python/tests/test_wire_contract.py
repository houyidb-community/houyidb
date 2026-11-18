"""Python SDK wire 契约（无服务端；假传输记录请求、回放响应）。

覆盖 P4 新增面：创建 sparse 选项、逐向量 sparse 形状（缺省不改 / [] 清空）、
融合检索路由与请求体、响应平行分数组与 fusion 参数块、旧服务端缺字段容忍。

EN: Python SDK wire contracts (no server; a fake transport records requests and
replays responses). Covers the P4 additions: the create sparse option, per-vector
sparse shapes (absent means unchanged / [] clears), the fusion search route and
request body, the response's parallel score arrays plus the fusion params block,
and tolerance for fields absent from older servers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from houyidb import FusionResult, HouyiClient, Vector


class _FakeResponse:
    """最小 ``requests.Response`` 替身：只暴露 SDK 用到的字段。

    EN: Minimal ``requests.Response`` stand-in exposing only what the SDK uses.
    """

    def __init__(self, body: Any, status_code: int = 200):
        self.status_code = status_code
        self.text = ""
        self.reason = "OK"
        self._body = body

    def json(self) -> Any:
        return self._body


class _RecordingSession:
    """记录 (method, url, body) 并按固定响应回放。

    EN: Records (method, url, body) and replays a fixed response.
    """

    def __init__(self, body: Any):
        self.calls: List[Dict[str, Any]] = []
        self._body = body

    def _record(self, method: str, url: str, body: Optional[Any]) -> _FakeResponse:
        self.calls.append({"method": method, "url": url, "body": body})
        return _FakeResponse(self._body)

    def get(self, url: str, timeout: Optional[float] = None) -> _FakeResponse:
        return self._record("GET", url, None)

    def post(
        self, url: str, json: Optional[Any] = None, timeout: Optional[float] = None
    ) -> _FakeResponse:
        return self._record("POST", url, json)

    def put(
        self, url: str, json: Optional[Any] = None, timeout: Optional[float] = None
    ) -> _FakeResponse:
        return self._record("PUT", url, json)

    def delete(self, url: str, timeout: Optional[float] = None, **_: Any) -> _FakeResponse:
        return self._record("DELETE", url, None)


def _client(body: Any) -> tuple[HouyiClient, _RecordingSession]:
    session = _RecordingSession(body)
    return HouyiClient("http://127.0.0.1:1", session=session), session


def test_create_collection_sparse_option_and_old_server_echo_tolerance():
    """显式创建携带 ``sparse``；旧服务端响应缺该字段时原样返回不报错。

    EN: Explicit creation carries ``sparse``; when an older server omits the
    field from its response, the raw dict is returned without error.
    """
    client, session = _client({"created": True, "collection": "docs", "dims": 8})
    created = client.create_collection("docs", 8, index_type="flat", sparse="bm25")
    call = session.calls[-1]
    assert call["method"] == "PUT"
    assert call["url"].endswith("/v1/collections/docs")
    assert call["body"]["sparse"] == "bm25"
    assert call["body"]["index_type"] == "flat"
    # 旧服务端没有 sparse 回显：SDK 不解析、不报错。
    # EN: An older server has no sparse echo: the SDK neither parses nor errors.
    assert created["collection"] == "docs"
    assert "sparse" not in created

    unset = client.create_collection("docs2", 8)
    assert "sparse" not in session.calls[-1]["body"]
    assert "index_type" not in session.calls[-1]["body"]
    assert unset["dims"] == 8


def test_vector_sparse_wire_shape():
    """None 省略键（不改）、空列表序列化为 ``[]``（清空）、词项为二元数组。

    EN: None omits the key (unchanged), an empty list serializes as ``[]``
    (clear), and terms are ``[term, weight]`` pairs.
    """
    unset = Vector(id=1, values=[1.0, 0.0])
    assert "sparse" not in unset.to_json()

    cleared = Vector(id=2, values=[1.0, 0.0], sparse=[])
    assert cleared.to_json()["sparse"] == []

    set_terms = Vector(id=3, values=[1.0, 0.0], sparse=[(3, 1.5), (5, 0.25)])
    assert set_terms.to_json()["sparse"] == [[3, 1.5], [5, 0.25]]


def test_vector_sparse_pairs_rejected_before_wire():
    """客户端在序列化前拒绝非法词项：int() 截断/布尔/越界/非正权重都不会入 WAL。

    EN: Illegal pairs are rejected before serialization — int() truncation,
    bools, out-of-range ids and non-positive weights never reach the WAL.
    """
    with pytest.raises(ValueError, match="must be an integer"):
        Vector(id=1, values=[1.0], sparse=[(1.9, 1.0)]).to_json()
    with pytest.raises(ValueError, match="bool"):
        Vector(id=1, values=[1.0], sparse=[(True, 1.0)]).to_json()
    with pytest.raises(ValueError, match="out of range"):
        Vector(id=1, values=[1.0], sparse=[(0, 1.0)]).to_json()
    with pytest.raises(ValueError, match="finite and > 0"):
        Vector(id=1, values=[1.0], sparse=[(1, float("nan"))]).to_json()
    with pytest.raises(ValueError, match="finite and > 0"):
        Vector(id=1, values=[1.0], sparse=[(1, 0.0)]).to_json()
    # 非数值权重与布尔权重同样走 ValueError，而不是裸 TypeError/隐式 1.0。
    # EN: Non-numeric and boolean weights also raise ValueError rather than a
    # bare TypeError or a silent 1.0.
    with pytest.raises(ValueError, match="finite number"):
        Vector(id=1, values=[1.0], sparse=[(1, None)]).to_json()
    with pytest.raises(ValueError, match="got bool"):
        Vector(id=1, values=[1.0], sparse=[(1, True)]).to_json()
    # 溢出到 float 的大整数同样归入 ValueError 诊断，而不是裸 OverflowError。
    # EN: Huge integers that overflow float conversion are also normalized into
    # the ValueError diagnostic rather than a bare OverflowError.
    with pytest.raises(ValueError, match="finite number"):
        Vector(id=1, values=[1.0], sparse=[(1, 10**400)]).to_json()


def test_fusion_search_route_body_and_response():
    """融合路由/请求体/响应逐项对齐；可选旋钮缺省时不出现在请求体。

    EN: Fusion route/request body/response line up item by item; optional knobs
    stay out of the request body when unset.
    """
    body = {
        "ids": [9, 4],
        "scores": [0.75, 0.02],
        "dense_scores": [0.9, None],
        "sparse_scores": [None, 4.2],
        "metric": "cosine",
        "fusion": {"mode": "rrf", "rrf_k": 60, "alpha": None},
        "payloads": {"9": {"text": "nine"}},
    }
    client, session = _client(body)
    result = client.collection("docs").fusion_search(
        [1.0, 0.0], [(7, 0.5), (8, 0.25)], 2, include_payload=True
    )
    call = session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/collections/docs/fusion/search")
    assert call["body"] == {
        "vector": [1.0, 0.0],
        "sparse": [[7, 0.5], [8, 0.25]],
        "k": 2,
        "include_payload": True,
    }
    assert result.fusion_mode == "rrf"
    assert result.rrf_k == 60
    assert result.alpha is None
    assert result.metric == "cosine"
    assert [h.id for h in result.hits] == [9, 4]
    assert result.hits[0].dense_score == 0.9
    assert result.hits[0].sparse_score is None
    assert result.hits[1].dense_score is None
    assert result.hits[1].sparse_score == 4.2
    assert result.hits[0].payload == {"text": "nine"}


def test_fusion_search_carries_explicit_knobs():
    """显式给出的 filters/ef/融合旋钮全部原样上送。

    EN: Explicit filters/ef/fusion knobs are all sent verbatim.
    """
    client, session = _client(
        {
            "ids": [1],
            "scores": [1.0],
            "dense_scores": [0.8],
            "sparse_scores": [3.1],
            "metric": "l2",
            "fusion": {"mode": "weighted", "rrf_k": None, "alpha": 0.25},
        }
    )
    from houyidb import Filter

    client.collection("docs").fusion_search(
        [2.0, 0.0],
        [(9, 1.0)],
        1,
        filters=[Filter("tag", "eq", "x")],
        ef_search=48,
        fusion="weighted",
        rrf_k=7,
        alpha=0.25,
    )
    sent = session.calls[-1]["body"]
    assert sent["filters"] == [{"column": "tag", "op": "eq", "value": "x"}]
    assert sent["ef_search"] == 48
    assert sent["fusion"] == "weighted"
    assert sent["rrf_k"] == 7
    assert sent["alpha"] == 0.25


def test_fusion_response_tolerates_absent_optional_fields():
    """响应缺 dense_scores/sparse_scores/metric/fusion/payloads 时不报错。

    EN: A response missing dense_scores/sparse_scores/metric/fusion/payloads
    does not error.
    """
    result = FusionResult.from_json({"ids": [1, 2], "scores": [0.3, 0.1]})
    assert [h.id for h in result.hits] == [1, 2]
    assert all(h.dense_score is None for h in result.hits)
    assert all(h.sparse_score is None for h in result.hits)
    assert all(h.payload is None for h in result.hits)
    assert result.metric == ""
    assert result.fusion_mode == ""
    assert result.rrf_k is None
    assert result.alpha is None


def test_fusion_response_rejects_misaligned_parallel_arrays():
    """scores 必须与 ids 等宽；模态数组缺席容忍、出现时必须等宽。

    EN: scores must match ids in length; modality arrays are tolerated when
    absent but must match when present.
    """
    result = FusionResult.from_json(
        {"ids": [1, 2], "scores": [0.3, 0.1], "dense_scores": [0.9, None]}
    )
    assert [h.dense_score for h in result.hits] == [0.9, None]

    with pytest.raises(ValueError, match="scores has 1 entries for 2 ids"):
        FusionResult.from_json({"ids": [1, 2], "scores": [0.3]})
    with pytest.raises(ValueError, match="dense_scores has 1 entries for 2 ids"):
        FusionResult.from_json(
            {"ids": [1, 2], "scores": [0.3, 0.1], "dense_scores": [0.9]}
        )
