from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

# ==========================================
# DTO (Data Transfer Objects) - 표준 규격
# ==========================================

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmbedResult:
    """임베더가 무조건 뱉어야 하는 표준 상자"""
    dense: np.ndarray 
    sparse: Optional[List[Dict[str, float]]] = None 
    dimension: int = 768

@dataclass
class SearchResult:
    """검색기가 뱉어내는 매칭 결과"""
    doc: Document
    score: float


# ==========================================
# Interfaces - 모듈별 필수 구현 메서드
# ==========================================

class BaseChunker(ABC):
    @abstractmethod
    def split(self, text: str, doc_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Document]:
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> EmbedResult:
        pass

class BaseVectorStore(ABC):
    """오직 데이터 적재와 단순 거리 계산(Search)만 담당"""
    @abstractmethod
    def add_documents(self, documents: List[Document], embed_result: EmbedResult):
        pass

    @abstractmethod
    def search(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        pass

class BaseRetriever(ABC):
    """여러 Vector DB를 조합해 검색 전략(Retrieve) 수행"""
    @abstractmethod
    def retrieve(self, query_embed_result: EmbedResult, top_k: int = 3) -> List[SearchResult]:
        pass

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieved_results: List[SearchResult]) -> str:
        pass