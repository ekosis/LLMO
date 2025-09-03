import os
import json
import re
from typing import List, Dict, Iterable

import numpy as np
from dotenv import load_dotenv
import faiss
from openai import OpenAI

load_dotenv()

# ---------- Chunking utilities ----------

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]

def chunk_text(text: str, sentences_per_chunk: int = 5) -> Iterable[str]:
    """Yield semantically coherent chunks based on sentence boundaries."""
    sentences = _split_sentences(text)
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk]).strip()
        if chunk:
            yield chunk

# ---------- Document loading ----------

def load_documents() -> List[Dict[str, str]]:
    """Load items, notices and FAQs from data/*.json if they exist.

    Each JSON file is expected to contain a list of objects with an ``id`` and
    ``text`` (or ``content``) field. Missing files are skipped.
    """
    docs: List[Dict[str, str]] = []
    for name in ["items", "notices", "faqs"]:
        path = os.path.join("data", f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            text = entry.get("text") or entry.get("content") or ""
            doc_id = entry.get("id") or entry.get("question") or f"{name}-{len(docs)}"
            docs.append({"id": f"{name}-{doc_id}", "text": text})
    return docs

# ---------- Embedding and persistence ----------

def embed_chunks(chunks: List[Dict[str, str]], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed the provided chunks using OpenAI embeddings."""
    client = OpenAI()
    texts = [c["text"] for c in chunks]
    response = client.embeddings.create(model=model, input=texts)
    vectors = np.array([d.embedding for d in response.data], dtype="float32")
    return vectors

def save_embeddings(vectors: np.ndarray, chunks: List[Dict[str, str]], out_dir: str = "data/embeddings") -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), vectors)
    meta_path = os.path.join(out_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

# ---------- FAISS index ----------

def build_faiss_index(vectors: np.ndarray, out_dir: str = "data/index") -> None:
    os.makedirs(out_dir, exist_ok=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, os.path.join(out_dir, "chunks.faiss"))

# ---------- Retrieval ----------

def _load_metadata(meta_path: str) -> List[Dict[str, str]]:
    metadata: List[Dict[str, str]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

class Retriever:
    """Simple wrapper around a FAISS index for semantic search."""

    def __init__(self, index_path: str = "data/index/chunks.faiss", meta_path: str = "data/embeddings/metadata.jsonl", model: str = "text-embedding-3-small") -> None:
        self.index = faiss.read_index(index_path)
        self.metadata = _load_metadata(meta_path)
        self.client = OpenAI()
        self.model = model

    def query(self, text: str, k: int = 5) -> List[Dict[str, str]]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        q = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(q, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[int(idx)]
            results.append({"score": float(dist), "metadata": meta})
        return results

# ---------- Main pipeline ----------

def run_pipeline() -> None:
    docs = load_documents()
    chunks: List[Dict[str, str]] = []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            chunks.append({"id": doc["id"], "text": chunk})
    if not chunks:
        raise ValueError("No documents found to embed.")
    vectors = embed_chunks(chunks)
    save_embeddings(vectors, chunks)
    build_faiss_index(vectors)

if __name__ == "__main__":
    run_pipeline()
