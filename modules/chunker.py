from typing import Any, Dict, List, Callable
from core.base import BaseChunker, Document
import re
import numpy as np

'''
[Simple Chunker]
    텍스트의 의미나 문법적 경계(단어, 문장)를 고려하지 않고, 
    지정된 글자 수(chunk_size)와 겹침(overlap)에 따라 기계적으로 텍스트를 자릅니다.

    Limitations:
        문장이나 단어의 중간이 무작위로 절단될 확률이 높아, 임베딩 모델(Dense Encoder)이 
        해당 텍스트의 벡터를 생성할 때 문맥(Context)이 심각하게 훼손되는 문제
'''
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
    
'''
[Semantic Chunker]
    임베딩 모델을 활용해 인접한 문장 간의 코사인 유사도를 계산하고, 
    유사도가 급격히 하락하는 지점을 문맥이 바뀌는 경계로 판단하여 텍스트를 분할합니다.
'''
class SemanticChunker(BaseChunker):
    def __init__(self, embed_func: Callable[[List[str]], np.ndarray], percentile_threshold: int = 20):
        """
        :param embed_func: 문장 리스트를 받아 임베딩 벡터(np.ndarray)를 반환하는 콜백 함수
        :param percentile_threshold: 하위 N%의 유사도를 가진 지점을 문맥 경계로 간주 (기본 하위 20%)
        """
        self.embed_func = embed_func
        self.percentile_threshold = percentile_threshold

    def _split_into_sentences(self, text: str) -> List[str]:
        """한국어 특성을 고려한 1차 문장 분리"""
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def split(self, text: str, doc_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Document]:
        if metadata is None: metadata = {}
        
        sentences = self._split_into_sentences(text)
        
        # 문장이 하나뿐이면 그대로 반환
        if len(sentences) <= 1:
            return [Document(doc_id=f"{doc_id_prefix}_chunk_0", content=text, metadata=metadata)]

        # 1. 각 문장을 임베딩 벡터로 변환
        embeddings = self.embed_func(sentences)

        # 2. 인접한 문장(i 와 i+1) 간의 코사인 유사도 계산
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i+1])
            # 코사인 유사도 공식을 이용해 두 문장의 의미적 거리 계산
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)

        # 3. 문맥 단절 경계값(Threshold) 설정
        # 유사도 중 하위 percentile_threshold(예: 20%)에 해당하는 값을 컷오프 기준으로 설정
        threshold = np.percentile(similarities, self.percentile_threshold)

        docs = []
        current_chunk = [sentences[0]]
        chunk_idx = 0

        # 4. 유사도를 바탕으로 문맥이 이어지는 문장들을 하나의 Document로 병합
        for i, sim in enumerate(similarities):
            if sim < threshold:
                # 유사도가 기준치보다 낮음 -> "문맥이 바뀌었다"고 판단하고 분할(Split)
                chunk_content = " ".join(current_chunk).strip()
                docs.append(Document(
                    doc_id=f"{doc_id_prefix}_chunk_{chunk_idx}",
                    content=chunk_content,
                    metadata=metadata.copy()
                ))
                chunk_idx += 1
                current_chunk = [sentences[i+1]]
            else:
                # 문맥이 이어짐 -> 현재 덩어리에 문장 추가
                current_chunk.append(sentences[i+1])

        # 마지막에 남은 문장들 처리
        if current_chunk:
            chunk_content = " ".join(current_chunk).strip()
            if chunk_content:
                docs.append(Document(
                    doc_id=f"{doc_id_prefix}_chunk_{chunk_idx}",
                    content=chunk_content,
                    metadata=metadata.copy()
                ))

        return docs