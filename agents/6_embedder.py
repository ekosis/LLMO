"""Step 6: Embed chunks and store vectors."""
import hashlib
import json
import os
from pathlib import Path
from typing import List

import numpy as np
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
    output_dir = Path("data/index")
    if not input_path.exists():
        print(f"Input file {input_path} not found")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    texts: List[str] = []
    meta: List[dict] = []
    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line)
            texts.append(record.get("text", ""))
            meta.append({"id": record.get("id"), "label": record.get("label")})
    try:
        client = OpenAI()
    except Exception:
        client = None
    vectors = embed_texts(texts, client)
    np.save(output_dir / "embeddings.npy", np.array(vectors))
    with (output_dir / "meta.jsonl").open("w", encoding="utf-8") as outfile:
        for m in meta:
            outfile.write(json.dumps(m) + "\n")
    print(f"Stored {len(vectors)} embeddings in {output_dir}")

if __name__ == "__main__":
    main()
