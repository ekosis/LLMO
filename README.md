# LLMO 소개 

미래에는 검색의 영향력이 줄어들 것입니다. 
LLMO(LLM Optimization, 한국어 표기로는 '르모')는 웹 페이지 및 모바일 어플리케이션을 자동으로 LLM 검색에 최적화 될 수 있도록 유지 및 보수하는 SaaS를 지향합니다.
과거 모든 웹 페이지 및 모바일 어플리케이션들이 검색엔진최적화(SEO)를 지향했던 것처럼, 앞으로는 LLM Optimization이 필요합니다.
따라서 판매할 물건을 나열(Curation)하는 아마존, 쿠팡과 같은 쇼핑 페이지부터, 콘텐츠를 나열하는 유튜브 같은 모든 인터넷 페이지들이 LLMO의 고객입니다.

## LLMO 작동

LLMO는 GitHub 저장소의 웹/앱 소스를 수집하여 RAG(Retrieval-Augmented Generation) 파이프라인을 구축합니다
실제 사이트를 대상으로 다음 과정을 거쳐 검색 API를 제공합니다.

## 특징
- GitHub 또는 실제 웹 페이지에서 HTML 소스를 수집
- 페이지 블록을 LLM/휴리스틱으로 분류
- 분류된 블록에서 LLM/휴리스틱으로 특징을 추출하고 청크 단위로 묶음
- 임베딩을 생성하여 FAISS 인덱스를 구축
- 임베딩 기반 검색 API 제공

## 설치
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# OpenAI API 키 설정 (선택 사항)
echo "OPENAI_API_KEY=YOUR_KEY" > .env
```

## 환경 변수
실행 시 다음 환경 변수를 사용할 수 있습니다.
- `OPENAI_API_KEY` : OpenAI API 키(설정하지 않으면 해시 기반 임베딩 사용)
- `CHAT_MODEL` : 페이지 분류와 특징 추출에 사용할 모델 이름(기본값 `gpt-4o-mini`)
- `EMBEDDING_MODEL` : 임베딩 생성에 사용할 모델 이름(기본값 `text-embedding-3-small`)
- `PORT` : API 서버 포트(기본값 `8000`)

## 빠른 실행
모든 단계를 연속으로 실행하려면 제공된 스크립트를 사용합니다.
```bash
bash quickstart.sh
```

임베딩이 생성된 후 API가 실행되면 다음과 같이 간단한 검색을 시도해 볼 수 있습니다.
```bash
curl "http://localhost:8000/search?q=sample"
```

## 실행 순서
아래 명령을 순서대로 실행하여 파이프라인을 구축합니다.
```bash
python 1_collect_source.py                      # 예시 HTML 생성
python ingest/2_crawler.py <URL> --out data/raw # 실제 사이트 크롤링(옵션)
python agents/3_page_classifier.py              # 페이지 블록 분류
python agents/4_feature_extractor.py            # 특징 추출
python agents/5_chunker.py                      # 청크 생성
python agents/6_embedder.py                     # 임베딩 및 인덱스 저장
python 7_api.py                                 # 간단한 검색 API 실행
```

## 디렉터리 구조
```
data/
  raw/          # 원본 HTML
  parsed/       # 분류 결과 및 특징
  chunks/       # 청크 데이터
  index/        # 임베딩 및 메타 정보
```

## 라이선스
이 프로젝트는 데모 목적의 예시 코드이며 "as is" 상태로 제공됩니다.
모든 권한은 본 Github 운영자 개인에게 있습니다.
