import os
import faiss
import pickle
import numpy as np
from core.base import BaseVectorStore, Document, EmbedResult, SearchResult
from typing import List

class FAISSVectorStore(BaseVectorStore):
    """[대조군 창고] 순수 Dense 벡터 전용 DB (코사인 유사도 기반)"""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # 코사인 유사도 계산을 위한 내적(IP) 인덱스 사용
        self.index = faiss.IndexFlatIP(dimension) 
        self.doc_store = {}

    def add_documents(self, documents: List[Document], embed_result: EmbedResult):
        if embed_result.dense is not None:
            # FAISS 처리를 위한 타입 변환 및 L2 정규화 (IP + 정규화 = 코사인 유사도)
            vectors = np.array(embed_result.dense, dtype=np.float32)
            faiss.normalize_L2(vectors)
            
            self.index.add(vectors)
            
            start_idx = len(self.doc_store)
            for i, doc in enumerate(documents):
                self.doc_store[start_idx + i] = doc

    def search(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        if query_embed_result.dense is None or self.index.ntotal == 0:
            return []
            
        # 질문 벡터 정규화 후 검색
        query_vector = np.array(query_embed_result.dense, dtype=np.float32)
        faiss.normalize_L2(query_vector)
            
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx != -1 and idx in self.doc_store:
                # 정규화된 내적 값이므로 그대로 코사인 유사도로 사용 (-1.0 ~ 1.0)
                score = float(distances[0][i])
                results.append(SearchResult(doc=self.doc_store[idx], score=score))
                
        return results

    def save(self, path: str):
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "doc_store.pkl"), "wb") as f:
            pickle.dump(self.doc_store, f)

    def load(self, path: str):
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