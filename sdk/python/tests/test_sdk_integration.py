"""houyidb SDK × 真实服务的联调：upsert/search/hybrid/delete/include_payload。

EN: Integration test for the houyidb SDK against a real server:
upsert/search/hybrid/delete/include_payload.
"""

from __future__ import annotations

import pytest
from houyidb import Filter, HouyiClient, HouyiError, HnswParams, Vector, stable_id

from conftest import fake_embed, wait_flushed

COLLECTION = "sdk_it"

DOCS = [
    ("apple is a fruit", "fruit", 5),
    ("banana is yellow", "fruit", 3),
    ("laptop is electronics", "electronics", 999),
]


@pytest.fixture()
def col(base_url):
    client = HouyiClient(base_url)
    assert client.health()
    c = client.collection(COLLECTION)
    # 固定 id，重跑幂等（upsert 同 id 覆盖）。
    # EN: Fixed ids keep reruns idempotent (upsert overwrites the same id).
    c.upsert(
        Vector(
            id=i + 1,
            values=fake_embed(text),
            payload={"text": text, "category": cat, "price": price},
        )
        for i, (text, cat, price) in enumerate(DOCS)
    )
    wait_flushed(c)
    return c


def test_upsert_and_status(col):
    status = col.status()
    assert status["dims"] == 16
    assert status["scalar_row_count"] >= len(DOCS)


def test_search_with_payload_round_trip(col):
    q = fake_embed("apple is a fruit")
    result = col.search(q, 2, include_payload=True)
    assert len(result.hits) == 2
    top = result.hits[0]
    assert top.id == 1  # 自相似最高 / EN: self-similarity is highest
    assert top.payload is not None
    assert top.payload["text"] == "apple is a fruit"
    assert top.payload["category"] == "fruit"
    assert top.payload["price"] == 5
    # score 语义：余弦相似度，越大越相似
    # EN: score semantics: cosine similarity, larger is more similar
    assert result.scores[0] >= result.scores[1]


def test_hybrid_search_filters(col):
    q = fake_embed("apple is a fruit")
    result = col.hybrid_search(
        q, 3, filters=[Filter("category", "eq", "fruit"), Filter("price", "lt", 4)]
    )
    assert result.ids == [2]  # 只有 banana 同时满足 / EN: only banana satisfies both


def test_hybrid_search_in_operator(col):
    q = fake_embed("laptop is electronics")
    result = col.hybrid_search(
        q, 3, filters=[Filter("category", "in", ["electronics", "toys"])]
    )
    assert result.ids == [3]


def test_delete_removes_hit_and_is_idempotent(col):
    assert col.delete([2]) == 1
    result = col.search(fake_embed("banana is yellow"), 3)
    assert 2 not in result.ids
    assert col.delete([2]) == 0  # 幂等 / EN: idempotent
    assert col.delete([424242]) == 0  # 不存在的 id 静默跳过 / EN: missing ids are skipped silently


def test_reupsert_after_delete_resurrects_with_latest_payload(col):
    col.delete([1])
    col.upsert(
        [Vector(id=1, values=fake_embed("apple is a fruit"),
                payload={"text": "apple v2", "category": "fruit", "price": 7})]
    )
    wait_flushed(col)
    result = col.search(fake_embed("apple is a fruit"), 1, include_payload=True)
    assert result.hits[0].id == 1
    assert result.hits[0].payload["text"] == "apple v2"
    assert result.hits[0].payload["price"] == 7


def test_delete_by_filter_covers_fresh_writes(col):
    """按条件删除：无需等待 flush，刚写入（尚在构图队列）的向量也能命中。

    EN: Delete by filter: no need to wait for flush; vectors just written
    (still in the build queue) can be matched too.
    """
    c = col._client.collection("sdk_it_dbf")
    c.upsert(
        Vector(id=i + 1, values=fake_embed(t), payload={"tag": tag})
        for i, (t, tag) in enumerate([("x1", "gone"), ("x2", "gone"), ("x3", "keep")])
    )
    # 故意不 wait_flushed：标量行同步落页，delete-by-filter 立即生效。
    # EN: Deliberately no wait_flushed: scalar rows land on the page
    # synchronously, so delete-by-filter takes effect immediately.
    assert c.delete(filters=[Filter("tag", "eq", "gone")]) == 2
    wait_flushed(c)
    result = c.search(fake_embed("x1"), 3, include_payload=True)
    assert result.ids == [3]
    assert c.delete(filters=[Filter("tag", "eq", "gone")]) == 0  # 幂等 / EN: idempotent


def test_delete_requires_ids_or_filters(col):
    with pytest.raises(ValueError, match="ids or filters"):
        col.delete()


def test_stable_id_deterministic_and_u63():
    a, b = stable_id("doc-1"), stable_id("doc-1")
    assert a == b
    assert 0 <= a < (1 << 63)
    assert stable_id("doc-1") != stable_id("doc-2")


def test_unknown_collection_status_404(base_url):
    c = HouyiClient(base_url).collection("no_such_collection_xyz")
    with pytest.raises(HouyiError) as exc_info:
        c.status()
    assert exc_info.value.status_code == 404


def test_create_collection_metric_and_hnsw(base_url):
    """显式建集合回显度量与 HNSW 三元组；同名再建 409。
    EN: Explicit creation echoes the metric and HNSW triple; a second create
    of the same name yields 409.
    """
    client = HouyiClient(base_url)
    name = "py_it_create_metric"
    # 清理只容忍 404（从未创建）；其他错误直接让用例失败。
    # EN: Cleanup tolerates only 404 (never created); anything else fails the test.
    try:
        client.delete_collection(name)
    except HouyiError as exc:
        assert exc.status_code == 404, exc
    created = client.create_collection(
        name,
        16,
        metric="ip",
        index_type="hnsw",
        hnsw=HnswParams(m=16, ef_construction=100, ef_search=32),
    )
    assert created["created"] is True
    assert created["collection"] == name
    assert created["metric"] == "ip"
    assert created["index_type"] == "hnsw"
    assert created["hnsw"] == {"m": 16, "ef_construction": 100, "ef_search": 32}
    with pytest.raises(HouyiError) as exc_info:
        client.create_collection(name, 16, metric="ip", index_type="hnsw")
    assert exc_info.value.status_code == 409
    client.delete_collection(name)


def test_delete_collection_and_recreate(base_url):
    """删除整库后 404；同名重建得到空 collection。
    EN: Dropping a collection yields 404 afterwards; a same-name recreate
    starts empty.
    """
    client = HouyiClient(base_url)
    name = "py_it_admin_drop"
    col = client.collection(name)
    col.upsert([Vector(id=7, values=fake_embed("drop me"), payload={"tag": "drop"})])

    client.delete_collection(name)

    with pytest.raises(HouyiError) as exc_info:
        col.status()
    assert exc_info.value.status_code == 404

    col.upsert([Vector(id=8, values=fake_embed("fresh"))])
    wait_flushed(col)
    assert col.status()["indexed_count"] == 1


def test_delete_missing_collection_404(base_url):
    client = HouyiClient(base_url)
    with pytest.raises(HouyiError) as exc_info:
        client.delete_collection("py_it_admin_never_created")
    assert exc_info.value.status_code == 404
