"""Step 6: Embed chunks and store vectors."""
import hashlib
import json
import os
from pathlib import Path
from typing import List

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
    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line)
            texts.append(record.get("text", ""))
            ids.append(str(record.get("id")))
            labels.append(record.get("label"))

    try:
        client = OpenAI()
    except Exception:
        client = None
    vectors = embed_texts(texts, client)

    chroma_client = chromadb.PersistentClient(path=str(index_dir))
    collection = chroma_client.get_or_create_collection("chunks")
    metadatas = [
        {"id": i, "label": l, "text": t}
        for i, l, t in zip(ids, labels, texts)
    ]
    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    print(f"Upserted {len(vectors)} embeddings into Chroma at {index_dir}")

if __name__ == "__main__":
    main()
