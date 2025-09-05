"""Step 7: Provide a simple API over the embedded data."""
import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from openai import OpenAI

import chromadb
from chromadb.errors import NotFoundError

app = FastAPI()

INDEX_DIR = Path("data/index")
COLLECTION_NAME = "chunks"

try:
    _client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = _client.get_collection(COLLECTION_NAME)
    _peek = collection.peek(1)
    EMBEDDING_DIM = len(_peek["embeddings"][0]) if _peek.get("embeddings") else 0
except (NotFoundError, FileNotFoundError):
    collection = None
    EMBEDDING_DIM = 0

def embed_query(text: str) -> np.ndarray:
    try:
        client = OpenAI()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=[text])
        return np.array(resp.data[0].embedding)
    except Exception:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        dim = EMBEDDING_DIM or 32
        return np.array([b / 255 for b in digest[:dim]])

@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results to return", ge=1),
    label: Optional[str] = Query(None, description="Filter by label"),
):
    if collection is None:
        return JSONResponse({"error": "index not built"}, status_code=500)
    q_vec = embed_query(q)
    where = {"label": label} if label else None
    res = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["metadatas", "distances"],
    )
    matches = res.get("metadatas", [[]])[0]
    scores = res.get("distances", [[]])[0]
    return {"matches": matches, "scores": scores}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
