import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are a classifier that labels website blocks. "
    "Classify the block below as one of: product, notice, contact, or other. "
    "Return a JSON object with keys 'label' and 'confidence' (0-1).\n\n"
    "BLOCK:\n{block}\n"
)

def heuristic_label(text: str) -> str:
    """Rule-based fallback classification."""
    text = text.lower()
    if any(k in text for k in ["contact", "email", "phone", "address"]):
        return "contact"
    if any(k in text for k in ["cart", "buy", "price", "add to cart"]):
        return "product"
    if any(k in text for k in ["notice", "policy", "terms", "legal"]):
        return "notice"
    return "other"

def classify_block(client: OpenAI | None, block: str) -> Dict[str, Any]:
    """Classify a single block using the LLM with heuristic fallback."""
    label, confidence = "other", 0.0
    if client is not None:
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        try:
            response = client.responses.create(
                model=model,
                input=PROMPT_TEMPLATE.format(block=block[:4000]),
            )
            content = response.output_text
            data = json.loads(content)
            label = data.get("label", "other")
            confidence = float(data.get("confidence", 0))
        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)

    if confidence < 0.6:
        heuristic = heuristic_label(block)
        logger.info("Low confidence %.2f; using heuristic '%s'", confidence, heuristic)
        label = heuristic
    logger.info("Label: %s (conf %.2f)", label, confidence)
    return {"label": label, "confidence": confidence}

def process_blocks(input_path: Path, output_path: Path) -> None:
    try:
        client = OpenAI()
    except Exception as exc:
        logger.warning("OpenAI client unavailable: %s", exc)
        client = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            block = record.get("html") or record.get("json") or record.get("text", "")
            result = classify_block(client, block)
            record.update(result)
            outfile.write(json.dumps(record) + "\n")

def main() -> None:
    input_path = Path(os.getenv("PAGE_BLOCKS_PATH", "data/parsed/page_blocks.jsonl"))
    output_path = Path("data/parsed/page_types.jsonl")
    if not input_path.exists():
        logger.error("Input file %s not found", input_path)
        return
    process_blocks(input_path, output_path)

if __name__ == "__main__":
    main()
