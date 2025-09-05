"""Step 6: Embed chunks and store vectors."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI


def embed_texts(texts: List[str], client: OpenAI | None) -> List[List[float]]:
    if client is None:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [b / 255 for b in digest[:32]]
            vectors.append(vec)
        return vectors
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

    try:
        client = OpenAI()
    except Exception:
        client = None
    vectors = embed_texts(texts, client)

    chroma_client = chromadb.PersistentClient(path=str(index_dir))
    collection = chroma_client.get_or_create_collection("chunks")

    metadatas: List[Dict[str, Any]] = []
    for i, l, t, ts, te, pn, pr, cid in zip(
        ids, labels, texts, token_starts, token_ends, product_names, prices, content_ids
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
        metadatas.append(md)

    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    print(f"Upserted {len(vectors)} embeddings into Chroma at {index_dir}")


if __name__ == "__main__":
    main()

