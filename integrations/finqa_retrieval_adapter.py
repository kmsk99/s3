"""FastAPI /retrieve adapter — Weaviate hybrid + Cohere rerank wrapped as s3 paper's
HTTP retrieval contract (see s3/s3/search/retrieval_server.py lines 326-358).

Request:
    POST /retrieve
    {"queries": [...], "topk": 8, "return_scores": true}

Response:
    {"result": [[{"document": {"contents": "Title\\nText"}, "score": 0.83}, ...], ...]}

The "Title\\nText" format matches what s3's generation_s3.py _passages2string()
splits on (first line = title, rest = text). For FinQA we synthesize Title from
the chunk's source_type + chunk_index since FinQA docs don't carry titles.

Run:
    uvicorn finqa_retrieval_adapter:app --host 0.0.0.0 --port 3000 --workers 4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

REPO_ROOT = Path(os.getenv(
    "FINQA_REPO_ROOT",
    "/home/ubuntu/fin-qa-research"
))
sys.path.insert(0, str(REPO_ROOT / "src" / "finqa_common" / "src"))

import weaviate  # noqa: E402
import requests  # noqa: E402

# Bedrock support kept as fallback when OpenRouter not configured.
try:
    from finqa_common.utils import (
        get_bedrock_client,
        get_embedding,
        rerank_with_cohere,
    )
except Exception:
    get_bedrock_client = None
    get_embedding = None
    rerank_with_cohere = None


COLLECTION_NAME = os.getenv("FINQA_COLLECTION", "FinQA_Chunking_Markdown")
EMBEDDING_MODEL = os.getenv("FINQA_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
RERANK_MODEL = os.getenv("FINQA_RERANK_MODEL", "cohere.rerank-v3-5:0")
HYBRID_ALPHA = float(os.getenv("FINQA_HYBRID_ALPHA", "0.7"))
TOP_K_INITIAL = int(os.getenv("FINQA_TOP_K_INITIAL", "50"))
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080")
_HP = WEAVIATE_URL.replace("http://", "").replace("https://", "")
WEAVIATE_HOST = _HP.split(":")[0]
WEAVIATE_PORT = int(_HP.split(":")[1]) if ":" in _HP else 8080
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# OpenRouter — when key is set, route rerank through OpenRouter and skip
# Bedrock-based embedding (use BM25-only Weaviate retrieval).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_RERANK_MODEL = os.getenv("OPENROUTER_RERANK_MODEL", "cohere/rerank-v3.5")
USE_OPENROUTER = bool(OPENROUTER_API_KEY)


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    # FinQA-specific extension: scope retrieval to a single document id.
    # Optional — if absent the search runs across the whole collection (paper default).
    doc_id: Optional[str] = None


app = FastAPI()
_weaviate_client = None
_bedrock_client = None


def _ensure_clients():
    global _weaviate_client, _bedrock_client
    if _weaviate_client is None:
        _weaviate_client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
    if _bedrock_client is None and not USE_OPENROUTER and get_bedrock_client is not None:
        _bedrock_client = get_bedrock_client(region=AWS_REGION)


def _openrouter_rerank(query: str, documents: list, top_n: int) -> list:
    """Call OpenRouter Cohere rerank. Returns documents reordered with rerank_score."""
    payload = {
        "model": OPENROUTER_RERANK_MODEL,
        "query": query,
        "documents": [d.get("text", "") for d in documents],
        "top_n": min(top_n, len(documents)),
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/rerank",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    out = []
    for entry in body.get("results", []):
        idx = entry["index"]
        if 0 <= idx < len(documents):
            doc = dict(documents[idx])
            doc["rerank_score"] = entry.get("relevance_score")
            out.append(doc)
    return out


def _format_doc(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Weaviate+Cohere chunk into s3's {document:{contents,...}} shape."""
    source_type = chunk.get("source_type", "?")
    chunk_idx = chunk.get("chunk_index", "?")
    chunk_id = chunk.get("id") or f"{source_type}_{chunk_idx}"
    title = f"[{source_type}] chunk {chunk_idx} (id={chunk_id})"
    text = chunk.get("text", "")
    return {
        "document": {"contents": f"{title}\n{text}"},
        "score": float(
            chunk.get("rerank_score")
            or chunk.get("hybrid_score")
            or 0.0
        ),
    }


def _retrieve_single(
    query: str, topk: int, doc_id: Optional[str]
) -> List[Dict[str, Any]]:
    coll = _weaviate_client.collections.get(COLLECTION_NAME)
    if USE_OPENROUTER:
        # BM25-only retrieval (no Bedrock embedding); rerank via OpenRouter
        if doc_id:
            flt = weaviate.classes.query.Filter.by_property("doc_id").equal(doc_id)
            res = coll.query.bm25(
                query=query, query_properties=["text"],
                limit=TOP_K_INITIAL, return_metadata=["score"], filters=flt,
            )
        else:
            res = coll.query.bm25(
                query=query, query_properties=["text"],
                limit=TOP_K_INITIAL, return_metadata=["score"],
            )
    else:
        embedding = get_embedding(query, EMBEDDING_MODEL, _bedrock_client)
        if not embedding:
            return []
        if doc_id:
            flt = weaviate.classes.query.Filter.by_property("doc_id").equal(doc_id)
            res = coll.query.hybrid(
                query=query, vector=embedding, query_properties=["text"],
                alpha=HYBRID_ALPHA, limit=TOP_K_INITIAL, return_metadata=["score"], filters=flt,
            )
        else:
            res = coll.query.hybrid(
                query=query, vector=embedding, query_properties=["text"],
                alpha=HYBRID_ALPHA, limit=TOP_K_INITIAL, return_metadata=["score"],
            )
    chunks = [
        {
            "id": f"{o.properties.get('doc_id','')}_{o.properties.get('chunk_index','')}",
            "text": o.properties.get("text", ""),
            "original_index": o.properties.get("original_index"),
            "chunk_index": o.properties.get("chunk_index"),
            "source_type": o.properties.get("source_type", "?"),
            "hybrid_score": o.metadata.score if hasattr(o.metadata, "score") else None,
        }
        for o in res.objects
    ]
    if not chunks:
        return []
    if USE_OPENROUTER:
        reranked = _openrouter_rerank(query, chunks, topk)
    else:
        reranked = rerank_with_cohere(
            query=query, documents=chunks, model_id=RERANK_MODEL,
            bedrock_client=_bedrock_client, top_n=topk,
            weaviate_client=_weaviate_client,
        )
    return [_format_doc(c) for c in reranked[:topk]]


@app.on_event("startup")
def _startup() -> None:
    _ensure_clients()


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    if not request.queries:
        raise HTTPException(status_code=400, detail="empty queries")
    topk = request.topk or 8
    _ensure_clients()
    result = [_retrieve_single(q, topk, request.doc_id) for q in request.queries]
    return {"result": result}


@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME}


if __name__ == "__main__":
    uvicorn.run(
        "finqa_retrieval_adapter:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "3000")),
        workers=int(os.getenv("WORKERS", "1")),
    )
