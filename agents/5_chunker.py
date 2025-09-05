"""Step 5: Group extracted features into chunks."""
import json
from pathlib import Path

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken may be unavailable
    _ENCODER = None


def _encode(text: str):
    if _ENCODER is not None:
        return _ENCODER.encode(text)
    return text.split()


def _decode(tokens) -> str:
    if _ENCODER is not None:
        return _ENCODER.decode(tokens)
    return " ".join(tokens)


def chunk_features(
    input_path: Path,
    output_path: Path,
    max_tokens: int = 200,
    overlap: int = 50,
) -> None:
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        idx = 0
        for line in infile:
            record = json.loads(line)
            text = record.get("text", "")
            label = record.get("label", "other")

            # Capture product related fields if present so they can flow through to
            # the final chunks and ultimately the vector store metadata.
            product_fields = {
                "product_name": record.get("product_name"),
                "price": record.get("price"),
                "content_id": record.get("content_id"),
            }

            tokens = _encode(text)
            if not tokens:
                continue
            for start in range(0, len(tokens), step):
                end = min(start + max_tokens, len(tokens))
                chunk_text = _decode(tokens[start:end])
                chunk = {
                    "id": idx,
                    "label": label,
                    "text": chunk_text,
                    "token_start": start,
                    "token_end": end,
                    **product_fields,
                }
                outfile.write(json.dumps(chunk) + "\n")
                idx += 1

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
