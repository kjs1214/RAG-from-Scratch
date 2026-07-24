from typing import Any, Dict, List
from core.base import BaseChunker, Document

class SimpleChunker(BaseChunker):
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, doc_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Document]:
        if metadata is None: metadata = {}
        docs = []
        step = self.chunk_size - self.overlap
        chunk_idx = 0
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i+self.chunk_size].strip()
            if not chunk_text: continue
            
            doc_id = f"{doc_id_prefix}_chunk_{chunk_idx}"
            docs.append(Document(doc_id=doc_id, content=chunk_text, metadata=metadata))
            chunk_idx += 1
            
            if i + self.chunk_size >= len(text): break
            
        return docs