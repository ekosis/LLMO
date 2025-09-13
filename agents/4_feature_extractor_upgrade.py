"""Step 4: Extract basic features from classified page blocks (with image_url).

This module reads JSONL produced by step 3 (page classifier) where each record
contains both plain text and original HTML per block. It extracts:
- product_name / price / content_id (LLM + heuristic merge)
- image_url (from the first <img src="..."> found in the block's HTML)

Notes:
- `features["text"]` is ALWAYS plain text (record["text"]), so downstream steps
  (chunking / embedding) operate on clean text rather than raw HTML.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken may be unavailable
    _ENCODER = None


PRICE_RE = re.compile(r"(?:\$|USD\s*|KRW\s*|₩)?\s*(\d+(?:\.\d{2})?)")
CONTENT_ID_RE = re.compile(r"\b(?:content[\s_-]*id|product[\s_-]*id|id)\b[:\s]*([A-Za-z0-9_-]+)", re.I)

PROMPT_TEMPLATE = (
    "You are an extraction agent that returns JSON. Given the text below, "
    "identify a product or content name, any price (number only) and a "
    "content or product identifier if present. Use keys 'product_name', "
    "'price' and 'content_id'. Missing values should be null.\n\nTEXT:\n{text}\n"
)


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using tiktoken if available."""
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text.split())


def _heuristic_features(text: str, label: str) -> Dict[str, Any]:
    """Return feature dictionary using simple regular expressions."""
    features: Dict[str, Any] = {
        "product_name": None,
        "price": None,
        "content_id": None,
    }

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
            if price_pos != -1:
                name_end = min(name_end, price_pos)
            else:
                name_end = min(name_end, price_match.start())
        if cid_match:
            features["content_id"] = cid_match.group(1)
            name_end = min(name_end, cid_match.start())
        # naive product name guess: prefix up to price/id hints
        cand = text[:name_end].strip()
        features["product_name"] = cand or features["product_name"]
    else:
        if cid_match:
            features["content_id"] = cid_match.group(1)

    return features


def _llm_features(client: Optional[OpenAI], text: str) -> Dict[str, Any]:
    """Attempt to extract features using an LLM (optional)."""
    if client is None:
        return {"product_name": None, "price": None, "content_id": None}

    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    try:
        response = client.responses.create(
            model=model, input=PROMPT_TEMPLATE.format(text=text[:4000])
        )
        data = json.loads(response.output_text)
        # normalize types
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
    except Exception as exc:  # pragma: no cover - network/JSON issues
        logger.warning("LLM extraction failed: %s", exc)
        return {"product_name": None, "price": None, "content_id": None}


def _extract_image_url(html_raw: Optional[str]) -> Optional[str]:
    """Return the first image URL from a block's HTML, if any."""
    if not html_raw:
        return None
    try:
        soup = BeautifulSoup(html_raw, "html.parser")
        # Prefer <img src=...>; fall back to data-src if needed.
        img = soup.find("img", src=True) or soup.find("img", attrs={"data-src": True})
        if img:
            return img.get("src") or img.get("data-src")
    except Exception:
        return None
    return None


def extract_features(input_path: Path, output_path: Path) -> None:
    """Read classified blocks and write feature dictionaries to *output_path*."""
    try:
        client = OpenAI()
    except Exception as exc:  # pragma: no cover - OpenAI may be unavailable
        logger.warning("OpenAI client unavailable: %s", exc)
        client = None

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            record = json.loads(line)

            # IMPORTANT: always use plain text for features["text"]
            plain_text = (record.get("text") or "").strip()
            html_raw = record.get("html")  # provided by upgraded step 3

            label = record.get("label", "other")
            features: Dict[str, Any] = {
                "label": label,
                "text": plain_text,
                "token_count": count_tokens(plain_text),
                "product_name": None,
                "price": None,
                "content_id": None,
                "image_url": None,  # NEW
            }

            # LLM extractor (optional)
            llm_feats = _llm_features(client, plain_text)
            for key, val in llm_feats.items():
                features[key] = val

            # Heuristic merge (only fill missing)
            heur_feats = _heuristic_features(plain_text, label)
            for key, val in heur_feats.items():
                if features.get(key) in (None, "", []):
                    features[key] = val

            # Image URL extraction from HTML block
            img = _extract_image_url(html_raw)
            if img:
                features["image_url"] = img

            outfile.write(json.dumps(features) + "\n")

    logger.info("Wrote features to %s", output_path)


def main() -> None:
    input_path = Path("data/parsed/page_types.jsonl")
    output_path = Path("data/parsed/features.jsonl")
    if not input_path.exists():
        print(f"Input file {input_path} not found")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_features(input_path, output_path)
    print(f"Wrote features to {output_path}")


if __name__ == "__main__":
    main()
