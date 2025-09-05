"""Step 4: Extract basic features from classified page blocks.

This version extracts additional structured information for commerce and
content pages.  When a block is labelled as a product or product detail we try
to pull out the product name and price.  If a block references a content
identifier (e.g. ``Content ID: 123``) we capture that as ``content_id``.

All extracted values are heuristic and meant only for demonstration purposes.
"""

import json
import re
from pathlib import Path

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken may be unavailable
    _ENCODER = None


PRICE_RE = re.compile(r"(?:\$|USD\s*)(\d+(?:\.\d{2})?)")
CONTENT_ID_RE = re.compile(
    r"\b(?:content[\s_-]*id|id)\b[:\s]*([A-Za-z0-9_-]+)", re.I
)


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using tiktoken if available."""

    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Fallback: rough estimate based on whitespace splitting
    return len(text.split())

def extract_features(input_path: Path, output_path: Path) -> None:
    """Read classified blocks and write feature dictionaries to *output_path*."""

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            record = json.loads(line)
            text = (record.get("html") or record.get("text") or "").strip()
            label = record.get("label", "other")

            features = {
                "label": label,
                "text": text,
                "token_count": count_tokens(text),
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
