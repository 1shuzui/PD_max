"""RAG 预留路由（501）。"""

from __future__ import annotations

from fastapi import APIRouter, Response

router = APIRouter()


@router.get(
    "/search",
    summary="知识库检索（预留）",
    description="RAG 检索接口，当前未实现。",
)
async def knowledge_search() -> Response:
    return Response(status_code=501, content="知识库 RAG 尚未实现")


@router.post(
    "/ingest",
    summary="知识库文档入库（预留）",
    description="RAG 入库接口，当前未实现。",
)
async def knowledge_ingest() -> Response:
    return Response(status_code=501, content="知识库 RAG 尚未实现")
