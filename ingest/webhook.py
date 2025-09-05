"""GitHub webhook handler for incremental ingestion.

This server listens for GitHub push events and re-runs the
classification → extraction → chunking → embedding pipeline only for
files reported as changed. Results are upserted into a Chroma collection
so that the index stays up to date.
"""

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, Request

# 2_crawler uses a numeric prefix; load it dynamically
_C_RAW = Path(__file__).with_name("2_crawler.py")
spec = importlib.util.spec_from_file_location("crawler", _C_RAW)
crawler = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(crawler)  # type: ignore
crawl = crawler.crawl  # type: ignore[attr-defined]
slugify = crawler.slugify  # type: ignore[attr-defined]

# Similarly load numbered agent modules
_P_RAW = Path(__file__).parent.parent / "agents" / "3_page_classifier.py"
spec_p = importlib.util.spec_from_file_location("page_classifier", _P_RAW)
page_classifier = importlib.util.module_from_spec(spec_p)
assert spec_p and spec_p.loader
spec_p.loader.exec_module(page_classifier)  # type: ignore
iter_blocks = page_classifier.iter_blocks  # type: ignore[attr-defined]
classify_block = page_classifier.classify_block  # type: ignore[attr-defined]

_E_RAW = Path(__file__).parent.parent / "agents" / "6_embedder.py"
spec_e = importlib.util.spec_from_file_location("embedder", _E_RAW)
embedder = importlib.util.module_from_spec(spec_e)
assert spec_e and spec_e.loader
spec_e.loader.exec_module(embedder)  # type: ignore
embed_texts = embedder.embed_texts  # type: ignore[attr-defined]

try:  # Optional dependency; the server can still run without it
    import chromadb
except Exception:  # pragma: no cover - chroma not installed
    chromadb = None  # type: ignore

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - OpenAI not installed
    OpenAI = None  # type: ignore

app = FastAPI()

# Initialise Chroma collection if available
if chromadb is not None:
    CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection("documents")
else:  # pragma: no cover
    collection = None  # type: ignore


def _openai_client():
    """Return an OpenAI client or None if unavailable."""
    if OpenAI is None:
        return None
    try:
        return OpenAI()
    except Exception:
        return None


async def process_url(url: str) -> None:
    """Fetch a URL and push its chunks into Chroma."""
    # Fetch latest content
    html_path = await crawl(url)
    html = html_path.read_text(encoding="utf-8")

    # Classify blocks
    client = _openai_client()
    blocks = list(iter_blocks(html))
    classified = [classify_block(client, b) | {"text": b} for b in blocks]

    # Build chunks
    chunks = [
        {
            "id": f"{slugify(url)}_{i}",
            "text": c["text"],
            "label": c.get("label", "other"),
        }
        for i, c in enumerate(classified)
    ]

    # Embed and upsert
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, client)
    if collection is not None:
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=texts,
            metadatas=[{"label": c["label"], "source": url} for c in chunks],
            embeddings=vectors,
        )


def changed_files(payload: dict) -> Iterable[str]:
    """Extract added/modified files from a GitHub webhook payload."""
    for commit in payload.get("commits", []):
        for key in ("added", "modified"):
            for path in commit.get(key, []):
                yield path


@app.post("/github-webhook")
async def github_webhook(request: Request) -> dict:
    payload = await request.json()
    files = list({*changed_files(payload)})
    base = os.getenv("RAW_BASE_URL", "")
    urls = [base + f for f in files]
    await asyncio.gather(*(process_url(u) for u in urls))
    return {"processed": files}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
