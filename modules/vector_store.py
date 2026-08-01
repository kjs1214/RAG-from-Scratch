from core.base import BaseVectorStore, Document, EmbedResult, SearchResult
from typing import List

class FAISSVectorStore(BaseVectorStore):
    """[대조군 창고] 순수 Dense 벡터 전용 DB"""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None # 추후 faiss.IndexFlatIP 등 초기화
        self.doc_store = {}

    def add_documents(self, documents: List[Document], embed_result: EmbedResult):
        if embed_result.dense is not None:
            # TODO: FAISS 인덱스에 embed_result.dense 추가 및 doc_store 매핑
            pass

    def search(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        # TODO: query_embed_result.dense 로 FAISS 서치 후 반환
        return []

class BM25SparseStore(BaseVectorStore):
    """[실험군 창고] 키워드(Sparse) 전용 역색인 DB"""
    def add_documents(self, documents: List[Document], embed_result: EmbedResult):
        if embed_result.sparse is not None:
            # TODO: sparse 딕셔너리 기반 역색인 구축
            pass

    def search(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        # TODO: query_embed_result.sparse 기반 키워드 매칭
        return []