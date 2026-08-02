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
    
class RecursiveChunker(BaseChunker):
    """
    [Recursive Chunker]
    지정된 구분자(separators)의 우선순위에 따라 텍스트를 재귀적으로 분할합니다.
    일반적으로 '문단 -> 문장 -> 단어' 순으로 쪼개어 의미 훼손을 최소화합니다.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # 1순위: 이중 줄바꿈(문단), 2순위: 단일 줄바꿈(문장), 3순위: 공백(단어), 4순위: 글자 단위
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_recursively(self, text: str, separators: List[str]) -> List[str]:
        """재귀적으로 텍스트를 쪼개는 내부 함수"""
        if not separators:
            return [text]

        separator = separators[0]
        # 현재 구분자로 텍스트를 나눌 수 있는지 확인 (빈 문자열이면 무조건 글자 단위 분할)
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                # 쪼개진 조각이 여전히 제한 크기보다 크면, 다음 우선순위 구분자로 재귀 호출
                if len(separators) > 1:
                    good_splits.extend(self._split_recursively(s, separators[1:]))
                else:
                    # 더 이상 구분자가 없으면 강제로 자름
                    good_splits.append(s)
                    
        return good_splits

    def split(self, text: str, doc_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Document]:
        if metadata is None: metadata = {}
        
        # 1. 텍스트를 최대한 의미 단위로 잘게 쪼갬
        splits = self._split_recursively(text, self.separators)
        
        docs = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0

        # 2. 쪼개진 조각들을 chunk_size를 넘지 않는 선에서 다시 이어 붙임 (Merge)
        for s in splits:
            s_len = len(s)
            if current_length + s_len > self.chunk_size and current_chunk:
                # 덩어리가 꽉 차면 Document 상자로 포장
                chunk_text = "".join(current_chunk).strip()
                if chunk_text:
                    docs.append(Document(
                        doc_id=f"{doc_id_prefix}_chunk_{chunk_idx}",
                        content=chunk_text,
                        metadata=metadata.copy()
                    ))
                    chunk_idx += 1
                
                # Overlap 처리 로직: 끝부분 조각들을 남겨서 다음 덩어리에 포함
                overlap_length = 0
                overlap_chunk = []
                for prev_s in reversed(current_chunk):
                    if overlap_length + len(prev_s) <= self.overlap:
                        overlap_chunk.insert(0, prev_s)
                        overlap_length += len(prev_s)
                    else:
                        break
                        
                current_chunk = overlap_chunk
                current_length = sum(len(c) for c in current_chunk)

            current_chunk.append(s)
            current_length += s_len

        # 마지막 찌꺼기 덩어리 처리
        if current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if chunk_text:
                docs.append(Document(
                    doc_id=f"{doc_id_prefix}_chunk_{chunk_idx}",
                    content=chunk_text,
                    metadata=metadata.copy()
                ))

        return docs