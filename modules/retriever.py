from core.base import BaseRetriever, BaseVectorStore, EmbedResult, SearchResult
from typing import List

class BaselineRetriever(BaseRetriever):
    """단일 DB에 의존하는 단순 검색기"""
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        # 그냥 창고(VectorStore)에 검색을 패스만 함
        return self.vector_store.search(query_embed_result, top_k=top_k)


class HybridRRFRetriever(BaseRetriever):
    """Dense와 Sparse를 모두 검색하여 RRF로 점수를 융합하는 검색기"""
    def __init__(self, dense_store: BaseVectorStore, sparse_store: BaseVectorStore):
        self.dense_store = dense_store
        self.sparse_store = sparse_store

    def retrieve(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        # 1. 양쪽 창고 동시 검색 (융합을 위해 2배수로 뽑음)
        dense_results = self.dense_store.search(query_embed_result, top_k=top_k * 2)
        sparse_results = self.sparse_store.search(query_embed_result, top_k=top_k * 2)

        # 2. RRF 알고리즘 융합
        fused_results = self._apply_rrf(dense_results, sparse_results)
        
        return fused_results[:top_k]

    def _apply_rrf(self, dense_res: List[SearchResult], sparse_res: List[SearchResult], k=60) -> List[SearchResult]:
        # TODO: Reciprocal Rank Fusion 점수 계산 알고리즘 구현
        return dense_res # (임시 반환)