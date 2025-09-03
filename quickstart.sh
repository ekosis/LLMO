#!/bin/bash
# Quick start script chaining steps 1-7 for demonstration.
set -e

python 1_collect_source.py
# Step 2: optionally crawl a real site
python ingest/2_crawler.py https://github.com/Kuzma02/Electronics-eCommerce-Shop-With-Admin-Dashboard-NextJS-NodeJS --out data/raw || true  # Singitronic.com
python agents/3_page_classifier.py
python agents/4_feature_extractor.py
python agents/5_chunker.py
python agents/6_embedder.py
# Step 7: launch the API (Ctrl+C to stop)
# python 7_api.py
