"""Step 5: Group extracted features into chunks."""
import json
from pathlib import Path

def chunk_features(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for idx, line in enumerate(infile):
            record = json.loads(line)
            chunk = {"id": idx, "label": record.get("label", "other"), "text": record.get("text", "")}
            outfile.write(json.dumps(chunk) + "\n")

def main() -> None:
    input_path = Path("data/parsed/features.jsonl")
    output_path = Path("data/chunks/chunks.jsonl")
    if not input_path.exists():
        print(f"Input file {input_path} not found")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_features(input_path, output_path)
    print(f"Wrote chunks to {output_path}")

if __name__ == "__main__":
    main()
