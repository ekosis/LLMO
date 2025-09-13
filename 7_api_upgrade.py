"""
LLM-SEO Search API (extended)
- Vector search over chunked/embedded product data (Chroma)
- Indexing pipeline endpoints (/upsert, /reembed, /delete)
- Query planner (light): filter parsing + hybrid scoring
- Optional LLM re-ranking hook
- NEW: image_url in Match schema/response
- NEW: /events/query, /events/click logging + CTR-based learning-to-rank (demo)
"""

import hashlib
import json
import os
import re
import time
from math import log
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import chromadb
from chromadb.api.types import Where
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from openai import OpenAI

# ----------------------
# App & Storage
# ----------------------
app = FastAPI(
    title="LLM-SEO Search API",
    version="0.5.0",
    description="""
Vector search API for LLM tools/plugins.
Distances returned by Chroma are typically cosine distance (smaller = more similar).
This API also returns `similarity = 1 - distance` for easier interpretation.
""",
)

INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index/chroma"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RERANK_WITH_LLM = os.getenv("RERANK_WITH_LLM", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
SERVER_API_KEY = os.getenv("SERVER_API_KEY")
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "10"))

chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(COLLECTION_NAME)

# ----------------------
# Security / Rate limit
# ----------------------
_last_req_ts: float = 0.0
def rate_limit():
    global _last_req_ts
    now = time.time()
    min_interval = 1.0 / max(RATE_LIMIT_RPS, 0.1)
    if now - _last_req_ts < min_interval:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _last_req_ts = now

def check_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not REQUIRE_API_KEY:
        return
    if not SERVER_API_KEY:
        raise HTTPException(500, "Server API key not configured")
    if x_api_key != SERVER_API_KEY:
        raise HTTPException(401, "Invalid API key")

# ----------------------
# Event logs (demo)  NEW
# ----------------------
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = LOG_DIR / "stats.json"

def _load_stats() -> Dict[str, Any]:
    if STATS_PATH.exists():
        try:
            return json.loads(STATS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_stats(stats: Dict[str, Any]) -> None:
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

# ----------------------
# Schemas
# ----------------------
class ProductChunk(BaseModel):
    id: str = Field(..., description="Unique chunk id")
    text: str = Field(..., description="Chunk text")
    source_url: Optional[HttpUrl] = None
    label: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    product_name: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "USD"
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    content_id: Optional[str] = None
    image_url: Optional[HttpUrl] = None  # (optional) allow upsert with image

class UpsertRequest(BaseModel):
    chunks: List[ProductChunk]

class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    label: Optional[str] = None
    content_id: Optional[str] = None

class SearchRequest(BaseModel):
    q: str
    k: int = 8
    label: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    rerank: bool = False

class Match(BaseModel):
    id: str
    label: Optional[str]
    brand: Optional[str]
    category: Optional[str]
    text: str
    product_name: Optional[str]
    price: Optional[float]
    currency: Optional[str] = "USD"
    content_id: Optional[str]
    source_url: Optional[str]
    image_url: Optional[str] = None     # NEW
    token_start: Optional[int]
    token_end: Optional[int]
    distance: float
    similarity: float
    highlight: Optional[str] = None

class SearchResponse(BaseModel):
    matches: List[Match]
    took_ms: int

# ----------------------
# Embeddings (with LRU cache + fallback)
# ----------------------
@lru_cache(maxsize=10_000)
def _embed_cached(text: str, model: str) -> Tuple[float, ...]:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(model=model, input=[text])
        return tuple(resp.data[0].embedding)
    except Exception:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        size = 512
        repeats = size // len(digest) + 1
        extended = (digest * repeats)[:size]
        return tuple(b / 255 for b in extended)

def embed(text: str) -> List[float]:
    return list(_embed_cached(text, EMBEDDING_MODEL))

# ----------------------
# Simple Query Planner (light)
# ----------------------
_price_pat = re.compile(r"(\d+(?:\.\d+)?)\s*(?:원|won|usd|\$)?", re.I)
def parse_filters(q: str) -> Dict[str, Any]:
    q_norm = q.strip()
    prices = [float(m.group(1)) for m in _price_pat.finditer(q_norm)]
    min_price = max_price = None
    if prices:
        if any(x in q_norm for x in ["이하", "under", "<="]):
            max_price = max(prices)
        elif any(x in q_norm for x in ["이상", "over", ">="]):
            min_price = min(prices)

    tokens = re.findall(r"[A-Za-z가-힣0-9\-\+]{2,}", q_norm)
    brand = tokens[0] if tokens else None
    category = tokens[1] if len(tokens) > 1 else None
    return {"brand": brand, "category": category, "min_price": min_price, "max_price": max_price}

def hybrid_score(query: str, doc_text: str, distance: float) -> float:
    sim = 1.0 - distance
    q_terms = set(re.findall(r"\w+", query.lower()))
    d_terms = set(re.findall(r"\w+", doc_text.lower()))
    overlap = len(q_terms & d_terms) / (len(q_terms) + 1e-6)
    return 0.8 * sim + 0.2 * min(overlap, 1.0)

def highlight_snippet(q: str, text: str, window: int = 200) -> str:
    q_terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 1]
    if not q_terms:
        return text[:window]
    pos = min((text.lower().find(t) for t in q_terms if t in text.lower()), default=0)
    start = max(0, pos - window // 3)
    end = min(len(text), start + window)
    return text[start:end]

# ----------------------
# Core search
# ----------------------
def _where_from_filters(
    label: Optional[str],
    brand: Optional[str],
    category: Optional[str],
    min_price: Optional[float],
    max_price: Optional[float],
) -> Where:
    where: Dict[str, Any] = {}
    if label:
        where["label"] = label
    if brand:
        where["brand"] = brand
    if category:
        where["category"] = category
    if min_price is not None or max_price is not None:
        price_cond: Dict[str, Any] = {}
        if min_price is not None:
            price_cond["$gte"] = float(min_price)
        if max_price is not None:
            price_cond["$lte"] = float(max_price)
        where["price"] = price_cond
    return where or None  # type: ignore

def _maybe_llm_rerank(query: str, matches: List[Match]) -> List[Match]:
    if not RERANK_WITH_LLM or not OPENAI_API_KEY:
        return matches
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = "Query: {}\nItems:\n{}".format(
            query,
            "\n".join(f"- [{i}] {m.product_name or ''} :: {m.text[:200]}" for i, m in enumerate(matches)),
        )
        resp = client.chat.completions.create(
            model=os.getenv("RERANK_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": f"Score relevance 0..1 for each item.\n{prompt}\nReturn JSON list of floats."}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        scores = data.get("scores", [])
        if len(scores) == len(matches):
            paired = list(zip(matches, scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            return [p[0] for p in paired]
        return matches
    except Exception:
        return matches

def run_search(
    q: str,
    k: int = 8,
    label: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    adaptive: bool = True,
    rerank: bool = False,
) -> SearchResponse:
    if collection is None or collection.count() == 0:
        raise HTTPException(500, "index not built")

    t0 = time.time()

    # planner enrichment
    parsed = parse_filters(q)
    brand = brand or parsed["brand"]
    category = category or parsed["category"]
    if min_price is None:
        min_price = parsed["min_price"]
    if max_price is None:
        max_price = parsed["max_price"]

    where = _where_from_filters(label, brand, category, min_price, max_price)
    q_vec = embed(q)

    # adaptive candidate count
    n_raw = max(k + 7, 10) if (adaptive and len(q.split()) <= 3) else max(k + 3, 8)
    result = collection.query(query_embeddings=[q_vec], n_results=n_raw, where=where)

    metas = result.get("metadatas", [[]])[0]
    docs = result.get("documents", [[]])[0]
    dists = result.get("distances", [[]])[0]

    # Build matches
    candidates: List[Match] = []
    seen_ids = set()
    for md, doc, dist in zip(metas, docs, dists):
        cid_key = md.get("id") or md.get("content_id") or ""
        if cid_key in seen_ids:
            continue
        seen_ids.add(cid_key)
        sim = 1.0 - float(dist)
        candidates.append(
            Match(
                id=md.get("id"),
                label=md.get("label"),
                brand=md.get("brand"),
                category=md.get("category"),
                text=doc,
                product_name=md.get("product_name"),
                price=md.get("price"),
                currency=md.get("currency", "USD"),
                content_id=md.get("content_id"),
                source_url=md.get("source_url"),
                image_url=md.get("image_url"),  # NEW
                token_start=md.get("token_start"),
                token_end=md.get("token_end"),
                distance=float(dist),
                similarity=sim,
                highlight=highlight_snippet(q, doc),
            )
        )

    # CTR-based learning-to-rank  NEW
    stats = _load_stats()
    clicks = stats.get("clicks", {})
    alpha = float(os.getenv("LTR_ALPHA", "0.05"))  # demo weight
    def boosted_score(m: Match) -> float:
        base = hybrid_score(q, m.text, m.distance)
        cid = str(m.content_id or m.id)
        c = clicks.get(cid, 0)
        return base + alpha * log(1 + c)

    candidates.sort(key=lambda m: boosted_score(m), reverse=True)
    matches = candidates[:k]

    # Optional LLM re-rank
    if rerank:
        matches = _maybe_llm_rerank(q, matches)

    took_ms = int((time.time() - t0) * 1000)
    return SearchResponse(matches=matches, took_ms=took_ms)

# ----------------------
# Indexing pipeline
# ----------------------
def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _embed_many(texts: List[str]) -> List[List[float]]:
    return [embed(t) for t in texts]

@app.post("/upsert", dependencies=[Depends(rate_limit), Depends(check_api_key)], response_model=Dict[str, int])
def upsert(req: UpsertRequest):
    ids = [c.id for c in req.chunks]
    docs = [c.text for c in req.chunks]
    metas = [c.model_dump() for c in req.chunks]
    embs = _embed_many(docs)
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return {"upserted": len(ids)}

@app.post("/reembed", dependencies=[Depends(rate_limit), Depends(check_api_key)], response_model=Dict[str, int])
def reembed(label: Optional[str] = None):
    where = {"label": label} if label else None
    res = collection.get(where=where, include=["documents", "metadatas"])
    if not res or not res.get("ids"):
        return {"reembedded": 0}
    docs = res["documents"]
    ids = res["ids"]
    metas = res["metadatas"]
    embs = _embed_many(docs)
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return {"reembedded": len(ids)}

@app.post("/delete", dependencies=[Depends(rate_limit), Depends(check_api_key)], response_model=Dict[str, int])
def delete(req: DeleteRequest):
    where: Dict[str, Any] = {}
    if req.label:
        where["label"] = req.label
    if req.content_id:
        where["content_id"] = req.content_id
    count = 0
    if req.ids:
        collection.delete(ids=req.ids)
        count += len(req.ids)
    if where:
        got = collection.get(where=where)
        ids = got.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            count += len(ids)
    return {"deleted": count}

# ----------------------
# Search endpoint
# ----------------------
@app.post("/search", response_model=SearchResponse, dependencies=[Depends(rate_limit)])
def search(req: SearchRequest):
    try:
        resp = run_search(
            q=req.q,
            k=req.k,
            label=req.label,
            brand=req.brand,
            category=req.category,
            min_price=req.min_price,
            max_price=req.max_price,
            adaptive=True,
            rerank=req.rerank,
        )
        return resp
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ----------------------
# Events (demo)  NEW
# ----------------------
@app.post("/events/query")
def log_query(event: Dict[str, Any]):
    # event: {"query": str, "ts": int}
    with (LOG_DIR / "query.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"ok": True}

@app.post("/events/click")
def log_click(event: Dict[str, Any]):
    # event: {"query": str, "item_id": str, "content_id": str, "rank": int, "ts": int}
    with (LOG_DIR / "click.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    stats = _load_stats()
    cid = str(event.get("content_id") or event.get("item_id") or "")
    if cid:
        stats.setdefault("clicks", {})
        stats["clicks"][cid] = stats["clicks"].get(cid, 0) + 1
        _save_stats(stats)
    return {"ok": True}

# ----------------------
# Health & metadata
# ----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "count": collection.count() if collection else 0,
        "embedding_model": EMBEDDING_MODEL,
        "distance_metric": "cosine",
        "similarity_definition": "similarity = 1 - distance (cosine)",
    }

# Optional: ChatGPT plugin manifest (classic plugin spec)
@app.get("/.well-known/ai-plugin.json")
def plugin_manifest():
    host = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
    return {
        "schema_version": "v1",
        "name_for_human": "LLM-SEO Search",
        "name_for_model": "llm_seo_search",
        "description_for_model": "Search product chunks by semantics and filters; returns citations and highlights.",
        "auth": {"type": "none" if not REQUIRE_API_KEY else "user_http"},
        "api": {"type": "openapi", "url": f"{host}/openapi.json"},
        "logo_url": f"{host}/logo.png",
        "contact_email": "support@example.com",
        "legal_info_url": f"{host}/legal",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
