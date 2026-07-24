# core/pipeline.py
import time
from typing import List, Dict, Any
from .base import BaseChunker, BaseEmbedder, BaseVectorStore, BaseGenerator, Document

class RAGPipeline:
    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        generator: BaseGenerator,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        print("[Pipeline] 모든 모듈 조립 완료.")

    def build_index(self, raw_documents: List[Dict[str, Any]]):
        """[1. 텍스트 분할] -> [2. 벡터화] -> [3. DB 적재]"""
        print(f"[Pipeline] 총 {len(raw_documents)}개 원본 문서 인덱싱 시작...")
        start_time = time.time()
        
        all_chunks: List[Document] = []
        
        # 1. 텍스트 분할 (Chunking)
        for raw_doc in raw_documents:
            chunks = self.chunker.split(
                text=raw_doc["content"], 
                doc_id_prefix=raw_doc["id"],
                metadata={"title": raw_doc.get("title", "")}
            )
            all_chunks.extend(chunks)
            
        print(f"  -> {len(all_chunks)}개의 청크(Chunk)로 분할됨.")

        # 2. 벡터화 (Embedding)
        texts_to_embed = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.encode(texts_to_embed)

        # 3. Vector DB 저장
        self.vector_store.add_documents(all_chunks, embeddings)
        
        elapsed = time.time() - start_time
        print(f"[Pipeline] 인덱싱 완료 (소요 시간: {elapsed:.2f}초)")

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """[1. 질문 벡터화] -> [2. 문서 검색] -> [3. 답변 생성]"""
        start_time = time.time()
        
        # 2차원 배열 (1, 768) 형태 유지
        query_embedding = self.embedder.encode([question])

        # 2. FAISS 기반 초고속 검색
        retrieved_results = self.vector_store.search(query_embedding, top_k=top_k)

        # 3. LLM을 통한 최종 답변 생성
        answer = self.generator.generate(question, retrieved_results)
        
        elapsed = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": [
                {"score": res.score, "doc": res.doc} for res in retrieved_results
            ],
            "latency_seconds": round(elapsed, 3)
        }