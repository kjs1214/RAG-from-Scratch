import time
from typing import List, Dict, Any
from .base import BaseChunker, BaseEmbedder, BaseVectorStore, BaseGenerator

class RAGPipeline:
    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        generator: BaseGenerator,
    ):
        # 4개의 핵심 모듈을 조립
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator

    def build_index(self, raw_documents: List[Dict[str, Any]], batch_size: int = 32):
        """
        [1. Chunking] -> [2. Embedding] -> [3. Vector Store 저장] 
        """
        print("[Pipeline] 인덱싱 시작...")
        start_time = time.time()
        
        # 1. 텍스트 분할 (Chunking)
        # 로직: raw_documents를 순회하며 self.chunker.split() 호출

        # 2. 벡터화 (Embedding)
        # 로직: self.embedder.embed_texts() 호출

        # 3. 저장소에 추가
        # 로직: self.vector_store.add_documents() 호출
        
        elapsed = time.time() - start_time
        print(f"[Pipeline] 인덱싱 완료 (소요 시간: {elapsed:.2f}초)")

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        [1. 질문 벡터화] -> [2. Retrieval 검색] -> [3. Generator 답변 생성]
        """
        start_time = time.time()
        
        # 1. 질의 벡터화
        # query_vec = self.embedder.embed_query(question)

        # 2. 문서 검색
        # retrieved_results = self.vector_store.search(query_vec, top_k=top_k)

        # 3. 답변 생성
        # answer = self.generator.generate(question, retrieved_results)
        
        elapsed = time.time() - start_time
        
        return {
            "question": question,
            "answer": "아직 Generator 로직이 구현되지 않았습니다.",  # 임시 반환
            "latency_seconds": round(elapsed, 3)
        }