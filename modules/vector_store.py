# modules/vector_store.py
import os
import json
import faiss
import numpy as np
from dataclasses import asdict
from typing import List

from core.base import BaseVectorStore, Document, RetrievalResult

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # 코사인 유사도(Cosine Similarity)와 동일한 결과를 내기 위해 내적(Inner Product) 인덱스 사용
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []

    def add_documents(self, docs: List[Document], embeddings: np.ndarray):
        """임베딩 벡터를 L2 정규화하여 FAISS 인덱스에 밀어넣음"""
        # 코사인 유사도를 IP로 계산하려면 벡터 정규화가 필수
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(docs)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """질문 벡터와 가장 내적 값이 높은(거리가 가까운) 문서를 초고속 검색"""
        if self.index.ntotal == 0:
            return []
            
        # 질문 벡터도 정규화
        faiss.normalize_L2(query_embedding)
        
        # FAISS 검색 (distances: 점수, indices: 문서 인덱스 번호)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:  # 유효한 인덱스인 경우만
                results.append(RetrievalResult(doc=self.documents[idx], score=float(score)))
        return results

    def save(self, save_dir: str):
        """디스크에 DB 영구 저장 (FAISS 인덱스 + 문서 JSON)"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. FAISS 벡터 저장
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        
        # 2. 문서 원본 데이터 저장 (dataclass를 dict로 변환)
        docs_dict = [asdict(doc) for doc in self.documents]
        with open(os.path.join(save_dir, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(docs_dict, f, ensure_ascii=False, indent=2)
            
        print(f"[VectorStore] DB 디스크 저장 완료 -> {save_dir}")

    def load(self, load_dir: str):
        """디스크에서 메모리로 DB 적재"""
        # 1. FAISS 벡터 로드
        self.index = faiss.read_index(os.path.join(load_dir, "faiss.index"))
        
        # 2. 문서 원본 데이터 로드
        with open(os.path.join(load_dir, "docs.json"), "r", encoding="utf-8") as f:
            docs_dict = json.load(f)
            self.documents = [Document(**d) for d in docs_dict]
            
        print(f"[VectorStore] DB 디스크 로드 완료 (총 문서 수: {self.index.ntotal})")