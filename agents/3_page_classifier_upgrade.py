#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Classify page blocks (with HTML preserved per block).

- Extract <div>/<p>/<li> blocks
- For each block:
  - Keep both plain text and original HTML ("text", "html")
  - Classify with LLM (optional), fallback to heuristic
- Write JSONL: {"text": str, "html": str, "label": str, "confidence": float}
"""

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
    t = text.lower()
    if any(k in t for k in ["contact", "email", "phone", "address"]):
        return "contact"
    if any(k in t for k in ["cart", "buy", "price", "add to cart", "basket", "checkout"]):
        return "product_detail"
    if any(k in t for k in ["review", "rating", "stars", "testimonial", "feedback"]):
        return "review"
    if any(k in t for k in ["summary", "article", "blog", "news", "content"]):
        return "content_summary"
    if any(k in t for k in ["notice", "policy", "terms", "legal"]):
        return "notice"
    if any(k in t for k in ["category", "categories", "department", "shop by"]):
        return "category_list"
    return "other"

def classify_block(client: OpenAI | None, block_text: str) -> Dict[str, Any]:
    label, confidence = "other", 0.0
    if client is not None:
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        try:
            resp = client.responses.create(
                model=model,
                input=PROMPT_TEMPLATE.format(block=block_text[:4000]),
            )
            content = resp.output_text
            try:
                data = json.loads(content)
                label = str(data.get("label", "other"))
                conf_raw = data.get("confidence", 0) # 방어적 파싱
                confidence = float(conf_raw) if isinstance(conf_raw, (int, float, str)) else 0.0 # 범위 클램프
                confidence = max(0.0, min(1.0, confidence))
            except Exception as exc:
                logger.warning("LLM JSON parse failed: %s | raw=%s", exc, content[:200])
        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)

    if confidence < 0.6:
        h = heuristic_label(block_text)
        logger.info("Low confidence %.2f; using heuristic '%s'", confidence, h)
        label = h

    return {"label": label, "confidence": confidence}

def iter_blocks(html: str) -> Iterable[Dict[str, str]]:
    """Yield dict blocks with both plain text and original HTML."""
    soup = BeautifulSoup(html, "lxml")
    for el in soup.find_all(["div", "p", "li"]):
        text = el.get_text(strip=True)
        if not text:
            continue
        html_block = el.prettify()
        yield {"text": text, "html": html_block}

def process_html_dir(input_dir: Path, output_path: Path) -> None:
    # OpenAI 클라이언트는 선택사항: 없으면 휴리스틱만 사용
    try:
        client = OpenAI()
    except Exception as exc:
        logger.warning("OpenAI client unavailable: %s (fallback to heuristics)", exc)
        client = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for html_file in sorted(input_dir.glob("*.html")):
            html = html_file.read_text(encoding="utf-8", errors="ignore")
            for block_obj in iter_blocks(html):
                text = block_obj["text"]
                html_block = block_obj["html"]
                result = classify_block(client, text)
                record = {"text": text, "html": html_block}
                record.update(result)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
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
