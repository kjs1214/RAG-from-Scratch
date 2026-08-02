import os
import faiss
import pickle
import numpy as np
from core.base import BaseVectorStore, Document, EmbedResult, SearchResult
from typing import List

class FAISSVectorStore(BaseVectorStore):
    """[대조군 창고] 순수 Dense 벡터 전용 DB"""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # L2 거리(유클리디안) 기반의 완전 탐색 인덱스 사용
        self.index = faiss.IndexFlatL2(dimension) 
        self.doc_store = {} # 정수 인덱스와 Document 객체를 매핑하는 딕셔너리

    def add_documents(self, documents: List[Document], embed_result: EmbedResult):
        if embed_result.dense is not None:
            # 1. FAISS 인덱스에 벡터 추가
            self.index.add(embed_result.dense)
            
            # 2. 벡터의 인덱스와 Document 매핑 저장
            start_idx = len(self.doc_store)
            for i, doc in enumerate(documents):
                self.doc_store[start_idx + i] = doc

    def search(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        if query_embed_result.dense is None or self.index.ntotal == 0:
            return []
            
        # 1. FAISS 검색 수행 (거리가 짧을수록 유사함)
        distances, indices = self.index.search(query_embed_result.dense, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx != -1 and idx in self.doc_store:
                # L2 거리를 유사도 점수로 변환 (거리가 0에 가까울수록 1에 가까워지도록 단순화)
                # (원하시는 코사인 유사도 방식이 있다면 나중에 IP 인덱스로 변경 가능합니다)
                score = 1.0 / (1.0 + float(distances[0][i]))
                results.append(SearchResult(doc=self.doc_store[idx], score=score))
                
        return results

    def save(self, path: str):
        """FAISS 인덱스와 메타데이터(doc_store)를 디스크에 저장"""
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "doc_store.pkl"), "wb") as f:
            pickle.dump(self.doc_store, f)

    def load(self, path: str):
        """디스크에서 FAISS 인덱스와 메타데이터 로드"""
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "doc_store.pkl"), "rb") as f:
            self.doc_store = pickle.load(f)


class BM25SparseStore(BaseVectorStore):
    """[추후 실험군 창고] 키워드(Sparse) 전용 역색인 DB 뼈대"""
    def add_documents(self, documents: List[Document], embed_result: EmbedResult):
        if embed_result.sparse is not None:
            pass # TODO: 추후 구현

    def search(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        return [] # TODO: 추후 구현
        
    def save(self, path: str):
        pass
        
    def load(self, path: str):
        pass