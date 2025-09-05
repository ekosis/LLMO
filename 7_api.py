"""Step 7: Provide a simple API over the embedded data."""
import hashlib
import os
from pathlib import Path
from typing import Any, List

import chromadb
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

INDEX_DIR = Path("data/index/chroma")
chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
try:
    collection = chroma_client.get_collection("chunks")
except Exception:
    collection = None

embedding_dim: int | None = None
if collection is not None:
    try:
        peek = collection.peek()
        if peek.get("embeddings"):
            embedding_dim = len(peek["embeddings"][0])
    except Exception:
        pass


def embed_query(text: str) -> List[float]:
    try:
        client = OpenAI()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=[text])
        return resp.data[0].embedding
    except Exception:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        size = embedding_dim or 32
        repeats = size // len(digest) + 1
        extended = (digest * repeats)[:size]
        return [b / 255 for b in extended]


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, description="Number of results"),
    label: str | None = Query(None, description="Optional label filter"),
) -> Any:
    if collection is None or collection.count() == 0:
        return JSONResponse({"error": "index not built"}, status_code=500)
    q_vec = embed_query(q)
    where = {"label": label} if label else None
    result = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
        where=where,
    )
    matches = result.get("metadatas", [[]])[0]
    scores = result.get("distances", [[]])[0]
    return {"matches": matches, "scores": scores}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
