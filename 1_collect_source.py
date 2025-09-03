"""Step 1: Prepare sample website/app sources for demonstration.
Writes a small HTML file into data/raw/ so later steps can operate."""
from pathlib import Path

SAMPLE_HTML = """
<html>
  <body>
    <h1>Sample Shop</h1>
    <div class="product">
      <h2>Widget</h2>
      <p>Price: $9.99</p>
    </div>
    <div class="notice">
      <p>We will be closed on holidays.</p>
    </div>
    <div class="contact">
      <p>Contact us at info@example.com</p>
    </div>
  </body>
</html>
"""

def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sample.html"
    out_file.write_text(SAMPLE_HTML.strip(), encoding="utf-8")
    print(f"Wrote sample source to {out_file}")

if __name__ == "__main__":
    main()
