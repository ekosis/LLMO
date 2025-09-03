from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from openai import OpenAI


class AttributeExtractor:
    """Extract attributes from text blocks using an LLM."""

    def __init__(
        self,
        domain: str,
        schema_dir: str = "configs",
        output_path: str = "data/parsed/items.jsonl",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = OpenAI()
        self.domain = domain
        self.model = model
        self.schema = self._load_schema(schema_dir)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_schema(self, schema_dir: str) -> Optional[Dict[str, str]]:
        """Load a per-domain schema from YAML if available."""
        schema_path = Path(schema_dir) / f"{self.domain}_schema.yaml"
        if not schema_path.exists():
            return None
        with schema_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fields = data.get("fields")
        if isinstance(fields, dict):
            return fields
        if isinstance(fields, list):
            return {field: "" for field in fields}
        return None

    def _build_prompt(self, block_text: str) -> str:
        base = (
            "Extract structured key-value data from the following text. "
            "Respond with a JSON object."
        )
        if self.schema:
            keys = ", ".join(self.schema.keys())
            base += f" Use the following keys when applicable: {keys}."
        return f"{base}\n\nText:\n{block_text}"

    def _call_llm(self, prompt: str) -> Dict[str, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception:
            return {}

    def extract(self, blocks: List[Dict[str, str]]) -> List[Dict[str, Dict[str, str]]]:
        """Extract attributes for each block and persist them."""
        results = []
        for block in blocks:
            text = block.get("text", "")
            prompt = self._build_prompt(text)
            attributes = self._call_llm(prompt)
            item = {"block": block, "attributes": attributes}
            results.append(item)
            self._write_item(item)
        return results

    def _write_item(self, item: Dict[str, Dict[str, str]]) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sample_blocks = [{"text": "Widget 3000 by ACME costs $10."}]
    extractor = AttributeExtractor(domain="product")
    extractor.extract(sample_blocks)
