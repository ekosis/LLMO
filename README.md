# LLMO SaaS MVP (Singitronic target)

This is a minimal end-to-end MVP that:
1) fetches UI sources from a GitHub repo (e.g., Singitronic Next.js shop),
2) parses semantic units (items, FAQs, notices, docs),
3) chunks & embeds them,
4) builds a FAISS vector store,
5) answers user questions via RAG with GPT-4o (with citations).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
echo "OPENAI_API_KEY=YOUR_KEY" > .env

python 0_main.py     # Or run each step individually below
python 7_api.py      # Start FastAPI search API
```

### Run steps individually
```bash
python 1_fetch.py     # input: GitHub repo URL (default: Singitronic)
python 2_parse.py     # outputs parsed JSONL files
python 3_chunk_embed.py
python 4_vector_store.py
python 5_query.py     # interactive multi-turn Q&A; optionally save JSON
python 7_api.py       # optional: start API server
```

## Notes
- **Fetching**: For Next.js repos, rendered HTML is not present; we ingest UI source files as "webpage sources" for this MVP.
- **Chunking**: Unit-aware (items as a single chunk, FAQ Q/A pairs, notices/docs chunked by paragraphs).
- **Embeddings**: `text-embedding-3-small` by default (set `EMBEDDING_MODEL` env to change).
- **Chat model**: `gpt-4o` by default (set `CHAT_MODEL` env to change).
- **Index**: FAISS (cosine similarity via normalized vectors).

## Folders
```
data/
  raw/          # sources.jsonl
  parsed/       # items.jsonl, faqs.jsonl, notices.jsonl, docs.jsonl
  chunks/       # chunks.jsonl
  embeddings/   # embeddings.npy, meta.jsonl
  index/        # faiss.index, index_meta.jsonl
  answers/      # saved RAG answers
workspace/      # git clone target
```

## Security
- Only public repo sources are ingested in this MVP. Do not ingest secrets.
- Use `.env` for API keys (never commit your keys).

## License
This sample is provided "as is" for MVP/demo purposes.
