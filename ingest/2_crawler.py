import asyncio
import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Comment
from playwright.async_api import async_playwright

# CSS selectors for advertising and layout elements to remove.
AD_SELECTORS: Iterable[str] = [
    "script",
    "noscript",
    "style",
    "iframe",
    "header",
    "footer",
    "nav",
    "aside",
    "[id*='ad']",
    "[class*='ad']",
    "[id*='banner']",
    "[class*='banner']",
    "[id*='promo']",
    "[class*='promo']",
    ".cookie-banner",
]

def slugify(url: str) -> str:
    """Turn a URL into a safe filename."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")

async def fetch_page(url: str) -> str:
    """Render a page in a headless browser and return HTML."""
    async with async_playwright() as p:
        browser = await p.firefox.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        # Remove unwanted elements in browser context
        for selector in AD_SELECTORS:
            await page.eval_on_selector_all(selector, "els => els.forEach(el => el.remove())")
        html = await page.content()
        await browser.close()
    return html

def sanitize_html(html: str) -> str:
    """Strip advertising/layout elements and return clean HTML."""
    soup = BeautifulSoup(html, "lxml")
    for selector in AD_SELECTORS:
        for el in soup.select(selector):
            el.decompose()
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()
    return soup.prettify()

async def crawl(url: str, output_dir: Path | str = "data/raw") -> Path:
    """Crawl a URL and write sanitized HTML to output_dir."""
    raw_html = await fetch_page(url)
    clean_html = sanitize_html(raw_html)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{slugify(url)}.html"
    out_file.write_text(clean_html, encoding="utf-8")
    return out_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crawl a web page and save sanitized HTML.")
    parser.add_argument("url", help="URL to crawl")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    args = parser.parse_args()
    asyncio.run(crawl(args.url, args.out))
