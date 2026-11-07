"""HouyiDB Python SDK — collections HTTP API 的薄封装。

只封装传输与类型：collection 路由、upsert/search/hybrid_search/delete、
响应数据类化。框架语义（Document、filter DSL、embedding 职责）不属于本层，
由 langchain-houyidb / llama-index-vector-stores-houyidb 等 adapter 承担。

EN: Thin wrapper over the collections HTTP API. This layer only covers
transport and types: collection routes, upsert/search/hybrid_search/delete,
and response dataclasses. Framework semantics (Document, filter DSL, embedding
responsibilities) belong to the adapters such as langchain-houyidb /
llama-index-vector-stores-houyidb.
"""

from .client import (
    Collection,
    Filter,
    HouyiClient,
    HouyiError,
    HnswParams,
    SearchHit,
    SearchResult,
    Vector,
    stable_id,
)

__all__ = [
    "Collection",
    "Filter",
    "HouyiClient",
    "HouyiError",
    "HnswParams",
    "SearchHit",
    "SearchResult",
    "Vector",
    "stable_id",
]

__version__ = "0.2.0b8"
