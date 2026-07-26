# RAG-from-Scratch
## RAG에 대한 Low Level 구현을 위한 Repository

# 폴더 구조
```text
RAG-from-Scratch/
├── core/                   # RAG 파이프라인의 뼈대가 되는 추상 클래스 및 데이터 클래스
│   ├── base.py             # BaseChunker, BaseEmbedder 등 기본 인터페이스 및 타입 정의
│   └── pipeline.py         # 각 모듈을 조립하여 전체 흐름을 제어하는 파이프라인
│
├── modules/                # 실제 작동하는 세부 기능 구현체 모음
│   ├── chunker.py          # 긴 텍스트를 적절한 크기로 자르는 텍스트 분할 모듈
│   ├── embedder.py         # 텍스트를 다차원 벡터로 변환하는 임베딩 모듈 (진행률 및 배치 처리 포함)
│   ├── vector_store.py     # FAISS 기반 초고속 벡터 검색 및 디스크 저장/로드 모듈
│   └── generator.py        # 검색된 문서를 바탕으로 LLM(Qwen 등)이 답변을 생성하는 모듈
│
├── dbs/                    # [자동 생성] 텍스트 데이터를 벡터화하여 영구 저장하는 로컬 DB 폴더
│   └── wiki_ko_db/         # 한국어 위키백과 데이터베이스 폴더 (5만 개)
│       ├── docs.json       # 원본 텍스트 및 메타데이터가 저장된 JSON 파일
│       └── faiss.index     # FAISS 벡터 압축 인덱스 파일
│
├── single_test/            # 각 모듈이 정상 작동하는지 개별적으로 검증하는 단위 테스트 폴더
│   ├── test_generator.py   # LLM 답변 생성 기능 단독 테스트
│   └── test_retrieval.py   # FAISS 검색 및 임베딩 단독 테스트
│
├── build_db.py             # [오프라인 DB 구축] Hugging Face 등 외부 데이터를 받아와 청킹/임베딩 후 DB를 굽는 스크립트
├── main.py                 # [실전 RAG 서빙] 구축된 DB를 로드하여 사용자와 터미널에서 질의응답을 주고받는 대화형 챗봇 스크립트
│
└── requirements.txt        # 프로젝트 실행에 필요한 라이브러리 및 의존성 버전 관리 파일
```

# 가상환경 세팅 가이드

## 가상 환경 생성
```bash
conda create -n rag_env python=3.11 -y
```

## 가상 환경 활성화
```bash
conda activate rag_env
```

## 하드웨어 가속을 지원하는 PyTorch 2.4.0 (CUDA 12.1) 설치 (RTX 3060 기준)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 이외의 파일들 설치
```bash
pip install -r requirements.txt
```

# Vector DB 구축
```bash
python build_db.py (2023.11.01 기준 한국어 wikipedia 5만개 데이터 사용)
```

# Roadmap & Experimental Points

## 1. Chunking (데이터 전처리)
*   **현재 구현:** `SimpleChunker` - 고정 크기(400자) 및 오버랩(50자) 기반의 단순 슬라이딩 윈도우 분할.
*   **고도화 포인트 (Next Step):**
    *   **Recursive Character Chunking:** 문단, 문장, 단어 단위로 쪼개어 문맥 단절 최소화.
    *   **Small2Big (Parent Document Retriever):** 임베딩/검색은 촘촘하게(작은 청크), 생성 모델에는 전체 문맥(부모 문서) 전달.
    *   **Semantic Chunking:** 문장 간 임베딩 유사도를 계산해 문맥이 바뀌는 지점에서 텍스트 분할.
*   **To Read:** *RAPTOR (Sarthi et al., 2024)*

## 2. Embedding (벡터 변환)
*   **현재 구현:** `BasicEmbedder` - `ko-sroberta-multitask` 모델을 이용한 Dense Vector 추출 및 Batch 처리.
*   **고도화 포인트 (Next Step):**
    *   **모델 교체:** 최대 8,192 토큰 및 다국어 처리가 가능한 SOTA 모델(`BAAI/bge-m3`)로 업그레이드.
    *   **Hybrid Embedding:** 의미(Semantic) 기반의 Dense 검색과 고유명사/키워드 중심의 Sparse(BM25, SPLADE) 검색 병행.
*   **To Read:** *SPLADE (Formal et al., 2021)*, *BGE M3-Embedding (Chen et al., 2024)*

## 3. Vector Store & Retriever (검색 모듈)
*   **현재 구현:** `FAISSVectorStore` - L2 정규화 + Inner Product(내적) 기반의 완전 탐색(Flat) 인덱싱.
*   **고도화 포인트 (Next Step):**
    *   **Retriever 모듈 독립 및 Hybrid Search:** FAISS 결과와 BM25 결과를 융합하는 RRF(Reciprocal Rank Fusion) 알고리즘 적용.
    *   **Re-ranking (교차 인코더):** `bge-reranker` 등을 활용해 1차 검색된 문서들의 연관성을 재계산하여 정밀한 Top-K 선별.
*   **To Read:** *Lost in the Middle (Liu et al., 2023)*, *Reciprocal Rank Fusion (Cormack et al., 2009)*

## 4. Generator (답변 생성)
*   **현재 구현:** `LocalGenerator` - `Qwen2.5-3B-Instruct` (fp16) 모델 로드 및 프롬프트 엔지니어링을 통한 Hallucination 제어 (Fallback 로직).
*   **고도화 포인트 (Next Step):**
    *   **Query Rewriting:** 사용자의 모호한 질문을 검색에 최적화된 키워드형 쿼리로 자동 변환.
*   **To Read:** *Self-RAG (Asai et al., 2023)*, *Query Rewriting for RAG (Ma et al., 2023)*

# GPU 모니터링
```bash
watch -n 1 nvidia-smi
```

# 실행
```bash
python main.py
```