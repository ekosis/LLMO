import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable

from bs4 import BeautifulSoup
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are a classifier that labels website blocks. "
    "Classify the block below as one of: product, product_detail, category_list, review, "
    "content_summary, notice, contact, or other. "
    "Return a JSON object with keys 'label' and 'confidence' (0-1).\n\n"
    "BLOCK:\n{block}\n"
)

def heuristic_label(text: str) -> str:
    text = text.lower()
    if any(k in text for k in ["contact", "email", "phone", "address"]):
        return "contact"
    if any(k in text for k in ["cart", "buy", "price", "add to cart", "basket", "checkout"]):
        return "product_detail"
    if any(k in text for k in ["review", "rating", "stars", "testimonial", "feedback"]):
        return "review"
    if any(k in text for k in ["summary", "article", "blog", "news", "content"]):
        return "content_summary"
    if any(k in text for k in ["notice", "policy", "terms", "legal"]):
        return "notice"
    if any(k in text for k in ["category", "categories", "department", "shop by"]):
        return "category_list"
    return "other"

def classify_block(client: OpenAI | None, block: str) -> Dict[str, Any]:
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
    return {"label": label, "confidence": confidence}

def iter_blocks(html: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "lxml")
    for el in soup.find_all(["div", "p", "li"]):
        text = el.get_text(strip=True)
        if text:
            yield text

def process_html_dir(input_dir: Path, output_path: Path) -> None:
    try:
        client = OpenAI()
    except Exception as exc:
        logger.warning("OpenAI client unavailable: %s", exc)
        client = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for html_file in input_dir.glob("*.html"):
            html = html_file.read_text(encoding="utf-8")
            for block in iter_blocks(html):
                result = classify_block(client, block)
                record = {"text": block}
                record.update(result)
                outfile.write(json.dumps(record) + "\n")
    logger.info("Wrote classifications to %s", output_path)

def main() -> None:
    input_dir = Path("data/raw")
    output_path = Path("data/parsed/page_types.jsonl")
    if not input_dir.exists():
        logger.error("Input directory %s not found", input_dir)
        return
    process_html_dir(input_dir, output_path)

if __name__ == "__main__":
    main()
