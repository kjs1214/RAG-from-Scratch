from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch

"""
Data Model Define
"""
@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    doc: Document
    score: float

"""
Interface Define
"""
class BaseChunker(ABC):
    @abstractmethod
    def split(self, text: str, doc_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Document]:
        """텍스트를 분할하여 Document 리스트로 반환"""
        pass

class BaseEmbedder(ABC):
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """문서 리스트를 배치 처리하여 벡터(List[float])로 변환"""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """사용자 질문 1개를 벡터로 변환"""
        pass

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, docs: List[Document], embeddings: List[List[float]]):
        """문서와 임베딩 벡터를 저장소에 저장"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """질의 벡터와 유사한 문서를 top_k개 반환"""
        pass

class BaseGenerator(ABC):
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int = 512) -> str:
        """검색된 컨텍스트를 바탕으로 답변 생성"""
        pass