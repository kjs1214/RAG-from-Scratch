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

# GPU 모니터링
```bash
watch -n 1 nvidia-smi
```

# 실행
```bash
python main.py
```