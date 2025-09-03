"""Step 7: Provide a simple API over the embedded data."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

INDEX_DIR = Path("data/index")
EMB_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.jsonl"

embeddings = np.load(EMB_PATH) if EMB_PATH.exists() else None
meta = [json.loads(l) for l in META_PATH.open()] if META_PATH.exists() else []

def embed_query(text: str):
    try:
        client = OpenAI()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=[text])
        return np.array(resp.data[0].embedding)
    except Exception:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return np.array([b / 255 for b in digest[: embeddings.shape[1]]])

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.get("/search")
def search(q: str = Query(..., description="Search query")):
    if embeddings is None or not meta:
        return JSONResponse({"error": "index not built"}, status_code=500)
    q_vec = embed_query(q)
    sims = [cosine(q_vec, emb) for emb in embeddings]
    idx = int(np.argmax(sims))
    return {"match": meta[idx], "score": sims[idx]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
