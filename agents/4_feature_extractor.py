"""Step 4: Extract basic features from classified page blocks."""
import json
from pathlib import Path

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken may be unavailable
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using tiktoken if available."""

    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Fallback: rough estimate based on whitespace splitting
    return len(text.split())

def extract_features(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            text = (record.get("html") or record.get("text") or "").strip()
            label = record.get("label", "other")
            features = {
                "label": label,
                "text": text,
                "token_count": count_tokens(text),
            }
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
