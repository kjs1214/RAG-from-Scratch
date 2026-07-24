# core/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    doc: Document
    score: float

class BaseChunker(ABC):
    @abstractmethod
    def split(self, text: str, doc_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Document]:
        """긴 텍스트를 여러 개의 작은 Document 구조체로 쪼개어 반환"""
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, docs: List[Document], embeddings: np.ndarray):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        pass
        
    @abstractmethod
    def save(self, save_dir: str):
        pass

    @abstractmethod
    def load(self, load_dir: str):
        pass

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int = 256) -> str:
        pass