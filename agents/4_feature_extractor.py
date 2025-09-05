"""Step 4: Extract basic features from classified page blocks.

This step now parses additional structured metadata from the text based on the
assigned label. For example, product blocks may expose ``price`` and
``product_name`` fields while any block can surface a ``content_id`` if present.
"""

import json
import re
from pathlib import Path


def parse_fields(label: str, text: str) -> dict:
    """Heuristically extract structured fields from ``text``.

    Parameters
    ----------
    label: str
        The block label (e.g. ``product``).
    text: str
        The block text from which to extract fields.

    Returns
    -------
    dict
        A dictionary containing any extracted fields.
    """

    fields: dict = {}
    # Content id can appear for any label.
    id_match = re.search(r"ID[:\s]+(\w+)", text, re.IGNORECASE)
    if id_match:
        fields["content_id"] = id_match.group(1)

    if label == "product":
        # Price formatted as $12.34
        price_match = re.search(r"\$([0-9]+(?:\.[0-9]{2})?)", text)
        if price_match:
            fields["price"] = price_match.group(1)

        # Determine product name by stripping other fields from text
        name = text
        if price_match:
            name = name.split(price_match.group(0))[0]
        if id_match:
            name = name.split(id_match.group(0))[0]
        name = name.replace("Only", "").strip(" -:\n")
        if name:
            fields["product_name"] = name

    return fields


def extract_features(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            text = (record.get("html") or record.get("text") or "").strip()
            label = record.get("label", "other")
            features = {"label": label, "text": text}
            fields = parse_fields(label, text)
            features.update({k: v for k, v in fields.items() if v is not None})
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
