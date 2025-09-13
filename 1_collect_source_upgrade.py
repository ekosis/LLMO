#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_collect_source.py (upgraded)
- Demo sample HTML generation (optional: --demo)
- Real web/app crawling with Playwright + server-side sanitization
- Output: data/raw/<slug>.html
"""

import argparse
import asyncio
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

# ---- Settings ----------------------------------------------------------------

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 요소 제거 셀렉터(클라이언트/서버 공통 후보)
AD_SELECTORS = [
    "script",
    "style",
    "header",
    "footer",
    "nav",
    "iframe",
    "[class*=ad]",
    "[id*=ad]",
    "[class*=banner]",
    "[id*=banner]",
    "[class*=promo]",
    "[id*=promo]",
    ".cookie-banner",
    ".cookie",
]

CLIENT_SIDE_REMOVE_JS = f"""
(() => {{
  const selectors = {AD_SELECTORS!r};
  const toRemove = new Set();
  for (const sel of selectors) {{
    try {{
      document.querySelectorAll(sel).forEach(node => toRemove.add(node));
    }} catch (e) {{}}
  }}
  toRemove.forEach(n => n.remove());
}})();
"""

# ---- Utils -------------------------------------------------------------------

def slugify(url: str) -> str:
    s = re.sub(r"^https?://", "", url.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "index"

def sanitize_html_server_side(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # 제거
    for sel in AD_SELECTORS:
        try:
            for node in soup.select(sel):
                node.decompose()
        except Exception:
            pass
    # 공백 정리
    for tag in soup.find_all(text=True):
        if isinstance(tag, str):
            compact = re.sub(r"\s+\n", "\n", tag)
            compact = re.sub(r"[ \t]{2,}", " ", compact)
            if compact != tag:
                tag.replace_with(compact)
    return str(soup)

# ---- Demo HTML ---------------------------------------------------------------

DEMO_SOURCE_URL = "https://github.com/Kuzma02/Electronics-eCommerce-Shop-With-Admin-Dashboard-NextJS-NodeJS"

DEMO_HTML = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Singitronic Demo | LLM-SEO</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>body{{font-family:system-ui, sans-serif; margin:2rem;}}</style>
</head>
<body>
  <main>
    <h1>Singitronic Electronics – Demo Page</h1>
    <p>Source: <a href="{DEMO_SOURCE_URL}">{DEMO_SOURCE_URL}</a></p>

    <section class="product" data-content-id="sku123">
      <h2 class="product__name">Ultra TV 55</h2>
      <p class="product__price">₩950,000</p>
      <p class="product__desc">55-inch 4K OLED TV with HDR, thin bezel, smart features.</p>
    </section>

    <section class="product-detail" data-content-id="sku123">
      <h3>Details</h3>
      <ul>
        <li>Panel: OLED 4K</li>
        <li>Size: 55-inch</li>
        <li>HDR: Yes</li>
      </ul>
    </section>

    <section class="notice">
      <h3>Notice</h3>
      <p>Free shipping for orders over ₩500,000.</p>
    </section>

    <section class="contact">
      <h3>Contact</h3>
      <p>Email: support@singitronic.example</p>
    </section>
  </main>
</body>
</html>
"""

def write_demo(out_path: Path) -> Path:
    out_path.write_text(DEMO_HTML, encoding="utf-8")
    print(f"[demo] Wrote sample HTML -> {out_path}")
    return out_path

# ---- Real Crawler (Playwright) ----------------------------------------------

async def crawl(url: str, out_path: Optional[Path] = None, timeout_ms: int = 30000) -> Path:
    """
    Render the page with Playwright, remove noisy elements client-side,
    then sanitize again server-side and save to data/raw/<slug>.html
    """
    try:
        from playwright.async_api import async_playwright  # lazy import
    except Exception as e:
        raise RuntimeError(
            "Playwright is not installed. Install with:\n"
            "  pip install playwright\n"
            "  playwright install"
        ) from e

    out_path = out_path or RAW_DIR.joinpath(f"{slugify(url)}.html")

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url, timeout=timeout_ms)
        # client-side remove
        await page.evaluate(CLIENT_SIDE_REMOVE_JS)
        # wait minimal network idle
        await page.wait_for_timeout(500)
        html = await page.content()
        await context.close()
        await browser.close()

    # server-side sanitize
    cleaned = sanitize_html_server_side(html)
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"[crawl] Saved cleaned HTML -> {out_path}")
    return out_path

# ---- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect raw source (demo or real crawl) into data/raw/")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--demo", action="store_true", help="Write demo HTML instead of crawling a URL")
    g.add_argument("--url", type=str, help="URL to crawl (if omitted with --demo off, an error is raised)")
    parser.add_argument("--out", type=str, help="Optional output file path (defaults to data/raw/<slug>.html)")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Playwright navigation timeout (ms)")
    args = parser.parse_args()

    if args.demo:
        out = Path(args.out) if args.out else RAW_DIR.joinpath("sample.html")
        write_demo(out)
        return

    if not args.url:
        parser.error("either --demo or --url must be provided")

    out = Path(args.out) if args.out else RAW_DIR.joinpath(f"{slugify(args.url)}.html")
    asyncio.run(crawl(args.url, out_path=out, timeout_ms=args.timeout_ms))

if __name__ == "__main__":
    main()