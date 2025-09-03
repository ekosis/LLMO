"""Step 4: Extract basic features from classified page blocks."""
import json
from pathlib import Path

def extract_features(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            text = (record.get("html") or record.get("text") or "").strip()
            label = record.get("label", "other")
            features = {"label": label, "text": text}
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
