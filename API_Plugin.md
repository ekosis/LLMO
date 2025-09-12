# LLM-SEO Search Plugin API

외부 LLM(GPT, Claude 등)이 상품/문서 데이터를 청킹·임베딩한 벡터 DB에 접근하여 검색할 수 있도록 제공되는 플러그인입니다.

## 플러그인 접근

- **OpenAPI 스펙**: `/openapi.json`에서 제공되어 ChatGPT Plugin, LangChain Tool, LangGraph 등과 자동으로 연동할 수 있습니다.
- **ChatGPT 플러그인 매니페스트**: `/.well-known/ai-plugin.json`를 통해 플러그인 이름, 설명, 스펙 URL을 제공합니다.
- **인증**: `REQUIRE_API_KEY=true`인 경우 헤더에 `X-API-Key`를 포함합니다.

## 주요 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
|POST|`/upsert`|새 문서/상품을 인덱스에 추가합니다.|
|POST|`/reembed`|기존 데이터를 임베딩 모델로 재처리합니다.|
|POST|`/delete`|ID, 라벨 또는 콘텐츠 ID로 데이터 삭제|
|POST|`/search`|검색 질의를 수행하여 유사한 청크 반환|
|GET|`/health`|상태 및 인덱스 정보 확인|
|GET|`/.well-known/ai-plugin.json`|ChatGPT 플러그인 매니페스트|
|GET|`/openapi.json`|OpenAPI 스펙|

## 검색 흐름

1. **질의 전처리**: 간단한 Query Planner가 브랜드, 카테고리, 가격 범위를 추출하여 필터를 구성합니다.
2. **벡터 검색**: 쿼리를 임베딩 후 ChromaDB에서 유사한 청크를 검색합니다.
3. **하이브리드 스코어링**: 의미론적 유사도와 키워드 중첩을 결합합니다.
4. **선택적 LLM 리랭킹**: `rerank=true`일 때 OpenAI 모델을 호출해 관련성을 재평가합니다.
5. **하이라이트 생성**: 질의와 관련된 텍스트 일부를 포함합니다.

## 검색 요청 예시
```http
POST /search
Content-Type: application/json
{
  "q": "삼성 100만원 이하 55인치 TV",
  "k": 8,
  "rerank": true
}
```

## 검색 응답 예시
```json
{
  "matches": [
    {
      "id": "sku123#desc#1",
      "label": "prod",
      "brand": "Samsung",
      "category": "TV",
      "text": "삼성 울트라 55인치 4K OLED TV, HDR 지원...",
      "product_name": "Ultra TV 55",
      "price": 950000,
      "currency": "KRW",
      "content_id": "sku123",
      "source_url": "https://shop.example/p/sku123",
      "token_start": 0,
      "token_end": 180,
      "distance": 0.12,
      "similarity": 0.88,
      "highlight": "...삼성 울트라 55인치 4K OLED..."
    }
  ],
  "took_ms": 47
}
```

`distance`는 코사인 거리이며 작을수록 더 가깝고, `similarity = 1 - distance`로 변환됩니다. `highlight`는 질의와 매칭된 텍스트 일부를 제공합니다.

## 활용 사례
- 쇼핑몰 상품 검색 및 추천
- 내부 FAQ나 문서 검색
- 리뷰 분석 등 임베딩 기반 검색 기능

## 실행
```bash
python 7_api.py
```
실행 후 브라우저나 LLM 도구에서 `/search`, `/upsert` 등 엔드포인트를 호출하여 사용할 수 있습니다.
