# RAG-from-Scratch
## RAG에 대한 Low Level 구현을 위한 Repository

# 폴더 구조
```text
RAG-from-Scratch/
├── core/                    # RAG 파이프라인의 뼈대가 되는 추상 클래스 및 데이터 클래스 (DTO)
│   ├── base.py              # 모듈 간 데이터 규약(EmbedResult 등) 및 기본 인터페이스 정의
│   └── pipeline.py          # 각 모듈을 조립하여 전체 흐름(데이터 적재/질의응답)을 제어하는 파이프라인
│
├── modules/                 # 실제 작동하는 세부 기능 구현체 모음
│   ├── chunker.py           # 긴 텍스트를 적절한 크기로 자르는 텍스트 분할 (Simple, Semantic, Recursive)
│   ├── embedder.py          # 텍스트를 다차원 벡터로 변환하는 임베딩 모듈
│   ├── vector_store.py      # FAISS 기반 초고속 벡터 검색 및 디스크 저장/로드 (창고 역할)
│   ├── retriever.py         # 단일/다중 DB를 조합하여 최적의 검색 전략을 관리하는 모듈
│   └── generator.py         # 검색된 문서를 바탕으로 LLM(Qwen 등)이 답변을 생성하는 모듈
│
├── dbs/                     # [자동 생성] 텍스트 데이터를 벡터화하여 영구 저장하는 로컬 DB 폴더
│   ├── wiki_ko_simple_db/   # 단순 글자 수 기반 분할 DB
│   ├── wiki_ko_semantic_db/ # 의미(유사도) 기반 분할 DB
│   └── wiki_ko_recursive_db/# 규칙 기반 재귀적 분할 DB
│
├── build_db_baseline.py     # [오프라인 DB 구축] Simple Chunker 파이프라인 실행 스크립트
├── build_db_semantic.py     # [오프라인 DB 구축] Semantic Chunker 파이프라인 실행 스크립트
├── build_db_recursive.py    # [오프라인 DB 구축] Recursive Chunker 파이프라인 실행 스크립트
├── main.py                  # 구축된 DB를 로드하여 질의응답을 주고받는 챗봇 스크립트
│
└── requirements.txt         # 프로젝트 실행에 필요한 라이브러리 및 의존성 버전 관리 파일
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
# 1. 단순 글자 수 기반 분할 DB 구축
python build_db_baseline.py
```

```bash
# 2. 의미 기반 분할 DB 구축
python build_db_proposed.py
```

```bash
#3. 규칙 기반 재귀적 분할 DB 구축
python build_db_recursive.py
```

# Per Module Experimental Points

## 1. Chunking (데이터 전처리)
*   **현재 구현:** 
    * `SimpleChunker` - 고정 크기(400자) 및 오버랩(50자) 기반의 단순 슬라이딩 윈도우 분할.
    * `SemanticChunker` - 문장 간 임베딩 코사인 유사도를 계산해 문맥이 단절되는 지점을 동적으로 분할.
    * `RecursiveChunker` - 지정된 구분자(\n\n, \n, 공백 등)의 우선순위에 따라 텍스트를 재귀적으로 쪼개어 문맥 훼손을 최소화

*   **고도화 포인트 (Next Step):**
    *   **Small2Big (Parent Document Retriever):** 임베딩/검색은 촘촘하게(작은 청크), 생성 모델에는 전체 문맥(부모 문서) 전달.
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
##


# Step-by-Step RAG RoadMap

* **[Step 1.1] 순수 Dense 검색기**
  * *구현:* FAISS (L2 정규화가 적용된 코사인 유사도).
  * *과제:* 노이즈 문서를 걸러내기 위한 유사도 임계값(Threshold) 컷오프 로직 구현.
* **[Step 1.2] 순수 Sparse 검색기 (BM25 바닥부터 구현)**
  * *구현:* 무거운 검색 엔진(Elasticsearch 등) 없이 순수 파이썬 딕셔너리와 math 모듈 사용. TF, IDF(Robertson-Spärck Jones 공식), 문서 길이 정규화를 수기로 직접 구현.
  * *연구 포인트:* Dense 검색이 실패하고 BM25가 성공하는 특정 실패 모드(예: 고유 명사, 제품 일련번호 등) 분석.
* **[Step 1.3] 하이브리드 앙상블 (RRF)**
  * *구현:* Dense와 Sparse의 순위를 융합하는 RRF (Reciprocal Rank Fusion, Cormack et al., 2009) 알고리즘 적용.
  * *연구 목표:* Dense 단독, Sparse 단독, Hybrid 구성 간의 성능 격차(Recall@K) 정량화.

---

* **[Step 2.1] 재정렬 (Re-ranking)**
  * *개념:* 상위 K개의 검색된 문서를 Cross-Encoder 기반 모델로 정밀 채점하여 진짜 핵심 문서 Top-3만 추출.
  * *구현:* `BGE-Reranker` 등 경량화된 리랭커 모델 도입.
* **[Step 2.2] 문맥 압축 및 MMR**
  * *개념:* 검색된 문서들이 질문과 관련성이 있으면서도 서로 다른 시각을 담도록 보장하기 위해 MMR (Maximal Marginal Relevance) 적용. LLM 어텐션 저하를 막기 위한 문서 배치(Position Tuning) 최적화.

---

* **[Step 3.1] 질문 재작성 (Query Rewriting)**
  * *구현:* 경량 로컬 LLM을 활용해 사용자의 모호한 질문을 검색 최적화된 키워드 중심으로 변환.
* **[Step 3.2] HyDE (Hypothetical Document Embeddings)**
  * *개념:* LLM으로 가짜 정답(가설)을 먼저 생성한 뒤, 해당 가설 문장을 임베딩하여 벡터 공간을 검색함으로써 검색 정합성 극대화.

---

* **[Step 4.1] 시맨틱 청킹 (Semantic Chunking)**
  * *개념:* 연속된 문장 간의 임베딩 거리 변화를 계산하여 문맥이 전환되는 지점에서만 동적으로 청크 분할.
* **[Step 4.2] RAPTOR (계층적 트리 구조 검색)**
  * *개념:* 청크들을 클러스터링하고 LLM으로 재귀적으로 요약하여, 거시적 주제(숲)와 미세한 디테일(나무)을 동시에 검색할 수 있는 트리 구조 인덱스 구축.

---

* **[Step 5.1] RAGAS 프레임워크 통합**
  * *평가 지표:*
    * **Faithfulness (충실도):** 생성된 답변이 검색된 문맥에 기반하고 있는가? (할루시네이션 검증)
    * **Answer Relevance (답변 관련성):** 답변이 사용자의 질문을 직접적으로 해결하는가?
    * **Context Precision (검색 정밀도):** 검색기가 정답 문서를 최상단에 배치했는가?
    * **Context Recall (검색 재현율):** 검색된 청크에 정답을 위한 모든 정보가 포함되어 있는가?
  * *연구 목표:* Phase 1~4 모듈을 켜고 끄는 **Ablation Study**를 수행하여 성능 변화(Delta)를 표로 도출.

---

* **[Step 6.1] 적응형 라우팅 에이전트 (Adaptive Routing)**
  * *구현:* 질문의 의도를 파악하는 LLM 분류기 도입 (예: 단순 연산은 툴 호출, 최신 정보는 웹 검색, 전공 지식은 FAISS 호출).
* **[Step 6.2] 자기 교정 에이전트 (Self-RAG & CRAG)**
  * *구현:* 1차 답변 생성 후, 검색된 증거와 비교해 스스로 답변을 평가(Critique)하고, 불일치가 감지되면 2차 검색이나 검색 범위 확장을 유발하는 피드백 루프 구축.

---

# GPU 모니터링
```bash
watch -n 1 nvidia-smi
```

# 실행
```bash
python main.py
```