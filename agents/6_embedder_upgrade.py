"""Step 6: Embed chunks and store vectors (with image_url metadata)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI


def _fallback_dimension() -> int:
    raw = os.getenv("FALLBACK_EMBED_DIM")
    try:
        return int(raw) if raw else 1536
    except (TypeError, ValueError):
        return 1536


def _fallback_embed(text: str, size: int) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    repeats = size // len(digest) + 1
    return [b / 255 for b in (digest * repeats)[:size]]


def embed_texts(texts: List[str], client: OpenAI | None) -> List[List[float]]:
    """
    Batch-embed texts with OpenAI; fall back to deterministic hash vectors if API unavailable.
    """
    if client is None:
        size = _fallback_dimension()
        return [_fallback_embed(text, size) for text in texts]
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    input_path = Path("data/chunks/chunks.jsonl")
    index_dir = Path("data/index/chroma")
    if not input_path.exists():
        print(f"Input file {input_path} not found")
        return
    index_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    ids: List[str] = []
    labels: List[str] = []
    token_starts: List[Optional[int]] = []
    token_ends: List[Optional[int]] = []
    product_names: List[Optional[str]] = []
    prices: List[Optional[float]] = []
    content_ids: List[Optional[str]] = []
    image_urls: List[Optional[str]] = []  # NEW

    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line)
            texts.append(record.get("text", ""))
            ids.append(str(record.get("id")))
            labels.append(record.get("label"))
            token_starts.append(record.get("token_start"))
            token_ends.append(record.get("token_end"))
            product_names.append(record.get("product_name"))
            prices.append(record.get("price"))
            content_ids.append(record.get("content_id"))
            image_urls.append(record.get("image_url"))  # NEW

    # Embed
    try:
        client = OpenAI()
    except Exception:
        client = None
    vectors = embed_texts(texts, client)

    # Upsert to Chroma
    chroma_client = chromadb.PersistentClient(path=str(index_dir))
    collection = chroma_client.get_or_create_collection("chunks")

    metadatas: List[Dict[str, Any]] = []
    for i, l, t, ts, te, pn, pr, cid, img in zip(
        ids, labels, texts, token_starts, token_ends, product_names, prices, content_ids, image_urls
    ):
        md: Dict[str, Any] = {"id": i, "label": l, "text": t}
        if ts is not None:
            md["token_start"] = ts
        if te is not None:
            md["token_end"] = te
        if pn is not None:
            md["product_name"] = pn
        if pr is not None:
            md["price"] = pr
        if cid is not None:
            md["content_id"] = cid
        if img is not None:
            md["image_url"] = img  # NEW
        metadatas.append(md)

    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    print(f"Upserted {len(vectors)} embeddings into Chroma at {index_dir}")


if __name__ == "__main__":
    main()
