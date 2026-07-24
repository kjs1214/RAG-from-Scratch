# RAG-from-Scratch
## RAG에 대한 Low Level 구현을 위한 Repository

# 폴더 구조
```text
RAG-from-Scratch/
├── core/                   
│   ├── base.py            
│   └── pipeline.py         
│
├── modules/                
│   ├── chunker.py          
│   ├── embedder.py         
│   ├── vector_store.py     
│   └── generator.py        
│
├── single_test/          
│   ├── test_generator.py            
│   └── test_retrieval.py
│
├── main.py  
│
└── requirements.txt
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

# GPU 모니터링
```bash
watch -n 1 nvidia-smi
```

# 빠른 실행
```bash
python main.py
```