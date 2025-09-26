#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refresh.py (upgrade-aware)
- Detects changed raw HTML files (data/raw/*.html) via file hash
- Rebuilds ONLY the changed files through the upgraded pipeline:
  (blocks with html) -> classify (LLM+heuristic) -> features (LLM+heuristic+image_url)
  -> chunk (propagate image_url) -> embed -> upsert to Chroma
- Deletes previous chunks for a file before replacing them (manifest tracked)
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
from bs4 import BeautifulSoup
from openai import OpenAI

# ----------------------------
# Settings / Paths
# ----------------------------
RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index/chroma")
MANIFEST = Path("data/index/manifest.json")
COLLECTION = os.getenv("COLLECTION_NAME", "chunks")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Chunking params (match step 5)
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "200"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger("refresh")

# ----------------------------
# Tokenizer (tiktoken optional)
# ----------------------------
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None

def _encode(text: str):
    if _ENCODER is not None:
        return _ENCODER.encode(text)
    return text.split()

def _decode(tokens) -> str:
    if _ENCODER is not None:
        return _ENCODER.decode(tokens)
    return " ".join(tokens)

def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text.split())

# ----------------------------
# Step 3: block extract + classify (upgraded)
#  - iter_blocks returns {"text", "html"}
#  - classify_block(client, text) -> {"label","confidence"}
# ----------------------------
PROMPT_CLASSIFY = (
    "You are a classifier that labels website blocks. "
    "Classify the block below as one of: product, product_detail, category_list, review, "
    "content_summary, notice, contact, or other. "
    "Return a JSON object with keys 'label' and 'confidence' (0-1).\n\n"
    "BLOCK:\n{block}\n"
)

def heuristic_label(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["contact", "email", "phone", "address"]):
        return "contact"
    if any(k in t for k in ["cart", "buy", "price", "add to cart", "basket", "checkout"]):
        return "product_detail"
    if any(k in t for k in ["review", "rating", "stars", "testimonial", "feedback"]):
        return "review"
    if any(k in t for k in ["summary", "article", "blog", "news", "content"]):
        return "content_summary"
    if any(k in t for k in ["notice", "policy", "terms", "legal"]):
        return "notice"
    if any(k in t for k in ["category", "categories", "department", "shop by"]):
        return "category_list"
    return "other"

def classify_block(client: Optional[OpenAI], block_text: str) -> Dict[str, Any]:
    label, confidence = "other", 0.0
    if client is not None:
        try:
            resp = client.responses.create(
                model=CHAT_MODEL,
                input=PROMPT_CLASSIFY.format(block=block_text[:4000]),
            )
            content = resp.output_text
            try:
                data = json.loads(content)
                label = str(data.get("label", "other"))
                conf_raw = data.get("confidence", 0)
                confidence = float(conf_raw) if isinstance(conf_raw, (int, float, str)) else 0.0
                confidence = max(0.0, min(1.0, confidence))
            except Exception:
                logger.warning("LLM classify JSON parse failed: %s", content[:160])
        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)
    if confidence < 0.6:
        label = heuristic_label(block_text)
    return {"label": label, "confidence": confidence}

def iter_blocks(html: str) -> Iterable[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    for el in soup.find_all(["div", "p", "li"]):
        text = el.get_text(strip=True)
        if not text:
            continue
        yield {"text": text, "html": el.prettify()}

# ----------------------------
# Step 4: feature extract (upgraded)
#  - LLM + heuristic merge
#  - image_url from HTML block
# ----------------------------
PRICE_RE = re.compile(r"(?:\$|USD\s*|KRW\s*|₩)?\s*(\d+(?:\.\d{2})?)")
CONTENT_ID_RE = re.compile(r"\b(?:content[\s_-]*id|product[\s_-]*id|id)\b[:\s]*([A-Za-z0-9_-]+)", re.I)

PROMPT_EXTRACT = (
    "You are an extraction agent that returns JSON. Given the text below, "
    "identify a product or content name, any price (number only) and a "
    "content or product identifier if present. Use keys 'product_name', "
    "'price' and 'content_id'. Missing values should be null.\n\nTEXT:\n{text}\n"
)

def _llm_features(client: Optional[OpenAI], text: str) -> Dict[str, Any]:
    if client is None:
        return {"product_name": None, "price": None, "content_id": None}
    try:
        resp = client.responses.create(
            model=CHAT_MODEL, input=PROMPT_EXTRACT.format(text=text[:4000])
        )
        data = json.loads(resp.output_text)
        price_val = data.get("price")
        try:
            price_val = float(price_val) if price_val is not None else None
        except Exception:
            price_val = None
        return {
            "product_name": data.get("product_name"),
            "price": price_val,
            "content_id": data.get("content_id"),
        }
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        return {"product_name": None, "price": None, "content_id": None}

def _heuristic_features(text: str, label: str) -> Dict[str, Any]:
    features: Dict[str, Any] = {"product_name": None, "price": None, "content_id": None}
    price_match = PRICE_RE.search(text)
    cid_match = CONTENT_ID_RE.search(text)
    if label in {"product", "product_detail"}:
        name_end = len(text)
        if price_match:
            try:
                features["price"] = float(price_match.group(1))
            except Exception:
                features["price"] = None
            price_pos = text.lower().find("price")
            name_end = min(name_end, price_pos if price_pos != -1 else price_match.start())
        if cid_match:
            features["content_id"] = cid_match.group(1)
            name_end = min(name_end, cid_match.start())
        features["product_name"] = (text[:name_end].strip() or None)
    elif cid_match:
        features["content_id"] = cid_match.group(1)
    return features

def _extract_image_url(html_raw: Optional[str]) -> Optional[str]:
    if not html_raw:
        return None
    try:
        soup = BeautifulSoup(html_raw, "html.parser")
        img = soup.find("img", src=True) or soup.find("img", attrs={"data-src": True})
        if img:
            return img.get("src") or img.get("data-src")
    except Exception:
        return None
    return None

# ----------------------------
# Step 5: chunk (propagate image_url)
# ----------------------------
def chunk_features(records: List[Dict[str, Any]], file_id: str,
                   max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> List[Dict[str, Any]]:
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    chunks: List[Dict[str, Any]] = []
    idx = 0
    for rec in records:
        tokens = _encode(rec.get("text", ""))
        if not tokens:
            continue
        product_fields = {
            "product_name": rec.get("product_name"),
            "price": rec.get("price"),
            "content_id": rec.get("content_id"),
            "image_url": rec.get("image_url"),
        }
        label = rec.get("label", "other")
        for start in range(0, len(tokens), step):
            end = min(start + max_tokens, len(tokens))
            chunk_text = _decode(tokens[start:end])
            chunks.append({
                "id": f"{file_id}_{idx}",
                "label": label,
                "text": chunk_text,
                "token_start": start,
                "token_end": end,
                **product_fields,
            })
            idx += 1
    return chunks

# ----------------------------
# Step 6: embeddings (fallback if OpenAI unavailable)
# ----------------------------
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


def embed_texts(texts: List[str], client: Optional[OpenAI]) -> List[List[float]]:
    if client is None:
        size = _fallback_dimension()
        return [_fallback_embed(t, size) for t in texts]
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# ----------------------------
# Manifest helpers
# ----------------------------
def load_manifest() -> Dict[str, Any]:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(manifest: Dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# ----------------------------
# File processing
# ----------------------------
def process_file(html_file: Path, collection, client: Optional[OpenAI]) -> List[str]:
    html = html_file.read_text(encoding="utf-8", errors="ignore")
    blocks: List[Dict[str, Any]] = []
    for bo in iter_blocks(html):
        text = bo["text"]
        res = classify_block(client, text)
        blocks.append({"text": text, "html": bo.get("html"), **res})

    # features: LLM + heuristic (+ image_url)
    features: List[Dict[str, Any]] = []
    for r in blocks:
        text = r.get("text", "").strip()
        label = r.get("label", "other")
        f = {"label": label, "text": text, "token_count": count_tokens(text),
             "product_name": None, "price": None, "content_id": None, "image_url": None}
        # LLM first
        llm = _llm_features(client, text)
        f.update({k: v for k, v in llm.items() if v not in (None, "", [])})
        # Heuristic fill
        heur = _heuristic_features(text, label)
        for k, v in heur.items():
            if f.get(k) in (None, "", []):
                f[k] = v
        # image_url
        img = _extract_image_url(r.get("html"))
        if img:
            f["image_url"] = img
        features.append(f)

    # chunk
    file_id = html_file.stem
    chunks = chunk_features(features, file_id=file_id)

    # embed & upsert
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    vectors = embed_texts(texts, client)

    metadatas: List[Dict[str, Any]] = []
    for c in chunks:
        md: Dict[str, Any] = {"id": c["id"], "label": c["label"], "text": c["text"]}
        if c.get("token_start") is not None:
            md["token_start"] = c["token_start"]
        if c.get("token_end") is not None:
            md["token_end"] = c["token_end"]
        if c.get("product_name") is not None:
            md["product_name"] = c["product_name"]
        if c.get("price") is not None:
            md["price"] = c["price"]
        if c.get("content_id") is not None:
            md["content_id"] = c["content_id"]
        if c.get("image_url") is not None:
            md["image_url"] = c["image_url"]
        metadatas.append(md)

    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    logger.info("Refreshed %s (%d chunks)", html_file.name, len(chunks))
    return ids

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not RAW_DIR.exists():
        logger.error("Raw directory %s not found", RAW_DIR)
        return

    try:
        client = OpenAI()
    except Exception:
        client = None

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION)

    manifest = load_manifest()
    changed: List[str] = []

    for html_file in sorted(RAW_DIR.glob("*.html")):
        cur_hash = file_hash(html_file)
        entry = manifest.get(html_file.name)
        if entry and entry.get("hash") == cur_hash:
            continue  # unchanged
        # delete old chunks if existed
        if entry and entry.get("chunk_ids"):
            try:
                collection.delete(ids=entry["chunk_ids"])
            except Exception:
                pass
        # reprocess
        ids = process_file(html_file, collection, client)
        manifest[html_file.name] = {"hash": cur_hash, "chunk_ids": ids}
        changed.append(html_file.name)

    save_manifest(manifest)
    if changed:
        logger.info("Updated %d documents: %s", len(changed), ", ".join(changed))
    else:
        logger.info("No documents changed")

if __name__ == "__main__":
    main()
