import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from importlib import import_module
from openai import OpenAI

page_classifier = import_module("agents.3_page_classifier")
feature_extractor = import_module("agents.4_feature_extractor")
chunker_mod = import_module("agents.5_chunker")
embedder_mod = import_module("agents.6_embedder")

iter_blocks = page_classifier.iter_blocks
classify_block = page_classifier.classify_block
PRICE_RE = feature_extractor.PRICE_RE
CONTENT_ID_RE = feature_extractor.CONTENT_ID_RE
count_tokens = feature_extractor.count_tokens
_encode = chunker_mod._encode
_decode = chunker_mod._decode
embed_texts = embedder_mod.embed_texts

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

MANIFEST = Path("data/index/manifest.json")


def load_manifest() -> Dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(manifest: Dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def extract_features_from_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for record in blocks:
        text = record.get("text", "").strip()
        label = record.get("label", "other")
        price_match = PRICE_RE.search(text)
        cid_match = CONTENT_ID_RE.search(text)
        feature: Dict[str, Any] = {
            "label": label,
            "text": text,
            "token_count": count_tokens(text),
            "product_name": None,
            "price": None,
            "content_id": None,
        }
        if label in {"product", "product_detail"}:
            name_end = len(text)
            if price_match:
                feature["price"] = float(price_match.group(1))
                price_pos = text.lower().find("price")
                if price_pos != -1:
                    name_end = min(name_end, price_pos)
                else:
                    name_end = min(name_end, price_match.start())
            if cid_match:
                feature["content_id"] = cid_match.group(1)
                name_end = min(name_end, cid_match.start())
            feature["product_name"] = text[:name_end].strip() or None
        elif cid_match:
            feature["content_id"] = cid_match.group(1)
        features.append(feature)
    return features


def chunk_features(features: List[Dict[str, Any]], file_id: str, max_tokens: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    chunks: List[Dict[str, Any]] = []
    idx = 0
    for record in features:
        tokens = _encode(record.get("text", ""))
        if not tokens:
            continue
        label = record.get("label", "other")
        product_fields = {
            "product_name": record.get("product_name"),
            "price": record.get("price"),
            "content_id": record.get("content_id"),
        }
        for start in range(0, len(tokens), step):
            end = min(start + max_tokens, len(tokens))
            chunk_text = _decode(tokens[start:end])
            chunk = {
                "id": f"{file_id}_{idx}",
                "label": label,
                "text": chunk_text,
                "token_start": start,
                "token_end": end,
                **product_fields,
            }
            chunks.append(chunk)
            idx += 1
    return chunks


def process_file(html_file: Path, collection, client: OpenAI | None) -> List[str]:
    html = html_file.read_text(encoding="utf-8")
    blocks = []
    for block in iter_blocks(html):
        result = classify_block(client, block)
        blocks.append({"text": block, **result})
    features = extract_features_from_blocks(blocks)
    chunks = chunk_features(features, html_file.stem)
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
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
        metadatas.append(md)
    vectors = embed_texts(texts, client)
    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    logger.info("Refreshed %s (%d chunks)", html_file.name, len(chunks))
    return ids


def main() -> None:
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        logger.error("Raw directory %s not found", raw_dir)
        return
    try:
        client = OpenAI()
    except Exception:
        client = None
    chroma_client = chromadb.PersistentClient(path="data/index/chroma")
    collection = chroma_client.get_or_create_collection("chunks")

    manifest = load_manifest()
    changed: List[str] = []
    for html_file in raw_dir.glob("*.html"):
        current_hash = file_hash(html_file)
        entry = manifest.get(html_file.name)
        if entry and entry.get("hash") == current_hash:
            continue
        if entry and entry.get("chunk_ids"):
            collection.delete(ids=entry["chunk_ids"])
        ids = process_file(html_file, collection, client)
        manifest[html_file.name] = {"hash": current_hash, "chunk_ids": ids}
        changed.append(html_file.name)
    save_manifest(manifest)
    if changed:
        logger.info("Updated %d documents: %s", len(changed), ", ".join(changed))
    else:
        logger.info("No documents changed")

if __name__ == "__main__":
    main()
