import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup

# ----------------------------
# Config
# ----------------------------
RAW_DIR = Path("data/raw")
API_BASE = os.getenv("LLM_SEO_API", "http://localhost:8000")

st.set_page_config(
    page_title="LLM-SEO Demo: Baseline vs Plugin",
    layout="wide",
)

# ----------------------------
# Utils
# ----------------------------
def load_raw_docs() -> List[Dict[str, Any]]:
    """Load raw HTML files and extract plain text blocks for naive keyword search."""
    docs = []
    for p in sorted(RAW_DIR.glob("*.html")):
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            docs.append({"path": str(p), "text": text})
        except Exception as e:
            st.warning(f"Failed to read {p}: {e}")
    return docs

def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9가-힣]{2,}", s.lower())

def keyword_score(query: str, text: str) -> Tuple[float, List[str]]:
    """Very simple keyword match score = sum of term frequencies."""
    q_terms = tokenize(query)
    t_terms = tokenize(text)
    tf: Dict[str, int] = {}
    for t in t_terms:
        tf[t] = tf.get(t, 0) + 1
    score = sum(tf.get(qt, 0) for qt in q_terms)
    matched = [qt for qt in q_terms if tf.get(qt, 0) > 0]
    return float(score), matched

def snippet(text: str, query: str, window: int = 180) -> str:
    q_terms = tokenize(query)
    pos: Optional[int] = None
    low = text.lower()
    for qt in q_terms:
        i = low.find(qt)
        if i != -1:
            pos = i
            break
    if pos is None:
        return text[:window] + ("..." if len(text) > window else "")
    start = max(0, pos - window // 3)
    end = min(len(text), start + window)
    return (text[start:end] + ("..." if end < len(text) else "")).strip()

def post_event(endpoint: str, payload: Dict[str, Any]) -> None:
    """Fire-and-forget event logging."""
    try:
        requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=5)
    except Exception:
        pass

def plugin_search(query: str,
                  k: int = 8,
                  rerank: bool = False,
                  brand: Optional[str] = None,
                  category: Optional[str] = None,
                  min_price: Optional[float] = None,
                  max_price: Optional[float] = None) -> Dict[str, Any]:
    """Call backend /search with optional filters and log query event."""
    # 질의 이벤트 로깅
    post_event("/events/query", {"query": query, "ts": int(time.time() * 1000)})

    url = f"{API_BASE}/search"
    payload: Dict[str, Any] = {
        "q": query,
        "k": k,
        "rerank": rerank,
        "brand": brand or None,
        "category": category or None,
        "min_price": float(min_price) if min_price not in (None, 0) else None,
        "max_price": float(max_price) if max_price not in (None, 0) else None,
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=30)
    took = int((time.time() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"error": f"Non-JSON response: {resp.text[:200]}", "status": resp.status_code}
    data["_http_ms"] = took
    data["_status"] = resp.status_code
    data["_payload"] = payload
    return data

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("LLM-SEO Demo Controls")

default_query = "Samsung TV under $1000"
query = st.sidebar.text_input("Query", value=default_query)

top_k = st.sidebar.slider("Top K", min_value=3, max_value=20, value=8, step=1)
use_rerank = st.sidebar.checkbox("LLM re-rank (plugin side)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters (Plugin)")
brand = st.sidebar.text_input("brand", value="")
category = st.sidebar.text_input("category", value="")
min_price = st.sidebar.number_input("min_price", min_value=0.0, step=1.0, value=0.0)
max_price = st.sidebar.number_input("max_price", min_value=0.0, step=1.0, value=0.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**API base**")
st.sidebar.code(API_BASE, language="text")

st.title("🔎 Baseline vs LLM-SEO Plugin (Side-by-side Demo)")

st.write(
    "Left: naive keyword search over raw HTML files (`data/raw/*.html`). "
    "Right: your LLM-SEO plugin (`/search`) results from vector DB (chunks/embeddings)."
)

st.info(
    "No raw HTML yet? Run: `python 1_collect_source_upgrade.py --demo` "
    "then steps 3–6 to build the index.",
    icon="ℹ️",
)

# ----------------------------
# Columns
# ----------------------------
col_left, col_right = st.columns(2, gap="large")

# ----------------------------
# Left: Baseline keyword search on raw HTML
# ----------------------------
with col_left:
    st.subheader("Baseline (Raw HTML keyword search)")
    raw_docs = load_raw_docs()
    if not raw_docs:
        st.info("No raw HTML found in `data/raw/`. Run `python 1_collect_source_upgrade.py --demo` first.")
    else:
        scored = []
        for d in raw_docs:
            score, matched = keyword_score(query, d["text"])
            if score > 0:
                scored.append((score, matched, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            st.warning("No matches found with simple keyword scoring.")
        else:
            for s, matched, d in scored[:top_k]:
                st.markdown(f"**Score:** {s:.0f}  \n`{d['path']}`")
                st.caption(f"Matched: {', '.join(matched) if matched else '(none)'}")
                st.write(snippet(d["text"], query))
                st.divider()

# ----------------------------
# Right: Plugin search (Vector + Filters + Cards)
# ----------------------------
with col_right:
    st.subheader("Plugin (/search)")
    if st.button("Run Plugin Search", type="primary"):
        data = plugin_search(
            query,
            k=top_k,
            rerank=use_rerank,
            brand=brand or None,
            category=category or None,
            min_price=min_price if min_price else None,
            max_price=max_price if max_price else None,
        )
        if data.get("error"):
            st.error(f"API error: {data['error']}")
        elif data.get("_status", 500) != 200:
            st.error(f"HTTP {data['_status']} from API")
        else:
            matches = data.get("matches", [])
            st.caption(f"API {data.get('_http_ms', 0)} ms · k={len(matches)} · payload={json.dumps(data.get('_payload', {}), ensure_ascii=False)}")
            if not matches:
                st.warning("No plugin matches.")
            else:
                for i, m in enumerate(matches, start=1):
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        # 썸네일
                        img_url = m.get("image_url")
                        if img_url:
                            try:
                                c1.image(img_url, use_column_width=True)
                            except Exception:
                                pass
                        # 제목/가격
                        title = m.get("product_name") or "(no product name)"
                        price = m.get("price")
                        currency = m.get("currency") or ""
                        c2.markdown(f"### #{i} · {title}")
                        if isinstance(price, (int, float)):
                            c2.caption(f"{currency} {price:,.0f}" if currency and currency != "USD" else f"${price:,.2f}")
                        # 하이라이트/텍스트
                        hl = m.get("highlight") or m.get("text", "")[:200]
                        c2.write(hl)
                        # CTA 버튼(클릭 이벤트 로깅)
                        src = m.get("source_url")
                        if src:
                            if c2.button("View product", key=f"cta_{i}"):
                                post_event("/events/click", {
                                    "query": query,
                                    "item_id": m.get("id"),
                                    "content_id": m.get("content_id"),
                                    "rank": i,
                                    "ts": int(time.time() * 1000),
                                })
                                st.markdown(f"[Open link]({src})")
                    st.markdown("")  # spacing
