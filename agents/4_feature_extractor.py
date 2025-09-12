"""Step 4: Extract basic features from classified page blocks.

Historically this module used only regular expressions to pull product names,
prices and content identifiers from each block.  That approach works for simple
commerce pages but does not generalise well to domains like game stores or
other digital content.  This revision adds an optional LLM based extractor –
similar to :mod:`agents.3_page_classifier` – that attempts to parse structured
information from arbitrary text.  The LLM output is merged with the previous
heuristic rules so that the script still functions without API access.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken may be unavailable
    _ENCODER = None


PRICE_RE = re.compile(r"(?:\$|USD\s*)(\d+(?:\.\d{2})?)")
CONTENT_ID_RE = re.compile(r"\b(?:content[\s_-]*id|id)\b[:\s]*([A-Za-z0-9_-]+)", re.I)

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
    # Fallback: rough estimate based on whitespace splitting
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
            features["price"] = float(price_match.group(1))
            price_pos = text.lower().find("price")
            if price_pos != -1:
                name_end = min(name_end, price_pos)
            else:
                name_end = min(name_end, price_match.start())
        if cid_match:
            features["content_id"] = cid_match.group(1)
            name_end = min(name_end, cid_match.start())
        features["product_name"] = text[:name_end].strip() or None
    elif cid_match:
        features["content_id"] = cid_match.group(1)

    return features


def _llm_features(client: OpenAI, text: str) -> Dict[str, Any]:
    """Attempt to extract features using an LLM."""

    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    try:
        response = client.responses.create(
            model=model, input=PROMPT_TEMPLATE.format(text=text[:4000])
        )
        data = json.loads(response.output_text)
        return {
            "product_name": data.get("product_name"),
            "price": data.get("price"),
            "content_id": data.get("content_id"),
        }
    except Exception as exc:  # pragma: no cover - network/JSON issues
        logger.warning("LLM extraction failed: %s", exc)
        return {"product_name": None, "price": None, "content_id": None}


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
            text = (record.get("html") or record.get("text") or "").strip()
            label = record.get("label", "other")

            features: Dict[str, Any] = {
                "label": label,
                "text": text,
                "token_count": count_tokens(text),
                "product_name": None,
                "price": None,
                "content_id": None,
            }

            if client is not None:
                llm_feats = _llm_features(client, text)
                for key, val in llm_feats.items():
                    features[key] = val

            heur_feats = _heuristic_features(text, label)
            for key, val in heur_feats.items():
                if features.get(key) in (None, "", []):
                    features[key] = val

            outfile.write(json.dumps(features) + "\n")

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
