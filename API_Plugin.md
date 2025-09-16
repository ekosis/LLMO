# API Plugin Specification

본 문서는 LLM/플러그인이 사용할 수 있는 Search API 및 이벤트/관리용 API 명세입니다.  
This document specifies the Search API and auxiliary endpoints for LLMs/plugins.  

## 검색 (Search)

### `POST /search`

**요청 (Request)**
```json
{
  "q": "Samsung TV under $1000",
  "k": 8,
  "label": "product",
  "brand": "Samsung",
  "category": "TV",
  "min_price": 500,
  "max_price": 1000,
  "rerank": false
}
```

**응답 (Response)**
```json
{
  "matches": [
    {
      "id": "sku123_0",
      "label": "product",
      "brand": "Samsung",
      "category": "TV",
      "text": "Samsung 55-inch 4K OLED TV with HDR...",
      "product_name": "Samsung OLED 55",
      "price": 950.0,
      "currency": "USD",
      "content_id": "sku123",
      "source_url": "https://example.com/samsung-oled55",
      "image_url": "https://example.com/samsung-oled55.png",
      "token_start": 0,
      "token_end": 180,
      "distance": 0.08,
      "similarity": 0.92,
      "highlight": "…4K OLED TV with HDR…"
    }
  ],
  "took_ms": 42
}
```

## 인덱스 관리 (Index Management)

### `POST /upsert`
- 신규 청크 등록 (Insert new chunks)

### `POST /reembed`
- 기존 문서 재임베딩 (Re-embed docs)

### `POST /delete`
- 특정 ID/label/content_id 삭제 (Delete by ID/label/content_id)

## 이벤트 (Events, Demo)

CTR 리랭킹 데모용 이벤트 API  
Event APIs for CTR-based reranking demo.

### `POST /events/query`
```json
{ "query": "Samsung TV under $1000", "ts": 1737000000000 }
```

### `POST /events/click`
```json
{
  "query": "Samsung TV under $1000",
  "item_id": "sku123_0",
  "content_id": "sku123",
  "rank": 1,
  "ts": 1737000001000
}
```

---

## 기타 (Misc)

- `/health`: 상태 확인 (health check)
- `/openapi.json`: OpenAPI 스펙 (schema)
- `/.well-known/ai-plugin.json`: ChatGPT plugin manifest

## 유사도 지표 (Similarity Metric)

- **Chroma distance**: cosine distance (0 = identical, 1 = dissimilar)  
- **Similarity 변환 (transform)**: `similarity = 1 - distance`
- `highlight`는 질의와 매칭된 텍스트 일부를 제공합니다.

-------



# LLM-SEO Search Plugin API

외부 LLM(GPT, Claude 등)이 상품/문서 데이터를 청킹·임베딩한 벡터 DB에 접근하여 검색할 수 있도록 제공되는 플러그인입니다.

## 플러그인 접근

- **OpenAPI 스펙**: `/openapi.json`에서 제공되어 ChatGPT Plugin, LangChain Tool, LangGraph 등과 자동으로 연동할 수 있습니다.
- **ChatGPT 플러그인 매니페스트**: `/.well-known/ai-plugin.json`를 통해 플러그인 이름, 설명, 스펙 URL을 제공합니다.
- **인증**: `REQUIRE_API_KEY=true`인 경우 헤더에 `X-API-Key`를 포함합니다.
