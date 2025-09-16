# LLMO 소개 

LLMO(LLM search engine - Optimization, 한국어 표기 '르모')는 
웹 페이지 및 모바일 어플리케이션의 소스를 자동으로 LLM 검색에 **최적화**하도록 전처리하고,
벡터DB를 통해 플러그인/툴에서 활용할 수 있는 검색 API를 제공합니다.  

이 프로젝트는 웹/앱의 소스를 **LLM 검색에 최적화**하도록 전처리하고,  
Streamlit 데모를 통해 Baseline(단순 크롤링 키워드) vs Plugin(LLM-SEO 벡터 검색) 비교 시연이 가능합니다.  

This project preprocesses web/app sources to **optimize them for LLM search**,  
stores them in a vector database, and exposes a search API consumable by LLM plugins/tools.  
A Streamlit demo allows side-by-side comparison of Baseline (naive keyword search) vs Plugin (LLM-SEO vector search).

## LLMO 작동 및 파이프라인(Pipeline)

LLMO는 웹/앱 소스를 수집하여 RAG(Retrieval-Augmented Generation) 파이프라인을 구축합니다
실제 사이트를 대상으로 다음 과정을 거쳐 검색 API를 제공합니다.

1. **수집 (Collect)**
   ```bash
   python 1_collect_source_upgrade.py --demo
   ```
   - `--demo`: 예시 HTML(Singitronic.com 샘플) 저장 (store sample HTML)
   - 실제 운영 시: 고객 웹/앱 크롤링 → `data/raw/*.html` 저장 (crawl client site)

2. **페이지 분류 (Page Classification)**
   ```bash
   python 3_page_classifier_upgrade.py
   ```
   - HTML 블록(`text`, `html`) 추출 (extract text+html)
   - LLM + 휴리스틱 분류 (classify via LLM+heuristics)
   - 출력: `data/parsed/page_types.jsonl`

3. **특징 추출 (Feature Extraction)**
   ```bash
   python 4_feature_extractor_upgrade.py
   ```
   - 상품명, 가격, content_id 추출 (extract product_name, price, content_id)
   - **image_url**: 첫 번째 `<img>` 태그 (first image tag)
   - 출력: `data/parsed/features.jsonl`

4. **청킹 (Chunking)**
   ```bash
   python 5_chunker_upgrade.py
   ```
   - 텍스트를 max_tokens 단위로 청킹 (split into chunks)
   - 상품 메타 전파 (propagate metadata: product_name, price, content_id, image_url)
   - 출력: `data/chunks/chunks.jsonl`

5. **임베딩 & 벡터DB 업서트 (Embedding & Upsert)**
   ```bash
   python 6_embedder_upgrade.py
   ```
   - OpenAI Embedding API (fallback: hash)
   - ChromaDB 업서트 (upsert to ChromaDB)
   - 메타데이터에 `image_url` 포함 (include image_url)

6. **검색 API (Search API)**
   ```bash
   uvicorn 7_api_upgrade:app --reload
   ```
   - `/search`: 의미론 검색 + 필터 (semantic search + filters)
   - `/upsert`, `/reembed`, `/delete`: 인덱스 관리 (index mgmt)
   - `/events/query`, `/events/click`: 이벤트 로깅 (event logging)
   - CTR 기반 리랭킹 데모 (CTR-based reranking demo)

7. **리프레시 (Refresh)**
   ```bash
   python refresh_upgrade.py
   ```
   - 파일 해시 추적 (track file hashes)
   - 변경된 문서만 재처리 (reprocess changed docs)
   - 기존 청크 삭제 후 교체 (delete old chunks, insert new)

8. **Streamlit 데모 (Streamlit Demo)**
   ```bash
   streamlit run app_streamlit.py
   ```
   - 좌측: Baseline (keyword search)
   - 우측: Plugin (vector search + metadata)
   - 카드 UI: 이미지/가격/CTA 버튼 (card UI with image/price/CTA)
   - 클릭 이벤트 CTR 반영 (click → CTR rerank)

## 특징 (Features)

- HTML 블록 단위의 휴리스틱 및 LLM기반 분석 (Block-level text+HTML Analysis by Heristic and LLM based)
- 이미지/가격/상품ID 휴리스틱 및 LLM기반 추출 → 벡터DB 저장 (Extract metadata by Heristic and LLM based → vector DB)
- API 기반 의미론 검색 + 필터 (Semantic search + filters)
- 이벤트 로깅 → CTR 리랭킹 (Event logging → CTR reranking demo)
- Baseline vs Plugin 비교 UI (Side-by-side comparison)

## 디렉토리 구조 (Directory Layout)

```
data/
  raw/                # HTML 원본 (raw HTML)
  parsed/             # page_types.jsonl, features.jsonl
  chunks/             # chunks.jsonl
  index/chroma/       # 벡터 인덱스 (vector index)
  index/manifest.json # 파일 해시/청크ID (manifest)
  logs/               # query.log, click.log, stats.json
```

## 빠른 실행 (Quickstart, Demo Mode)

```bash
python 1_collect_source_upgrade.py --demo
python 3_page_classifier_upgrade.py
python 4_feature_extractor_upgrade.py
python 5_chunker_upgrade.py
python 6_embedder_upgrade.py
uvicorn 7_api_upgrade:app --reload
streamlit run app_streamlit.py
```

## 라이선스
이 프로젝트는 데모 목적의 예시 코드이며 "as is" 상태로 제공됩니다.
모든 권한은 본 Github 운영자 개인에게 있습니다.
