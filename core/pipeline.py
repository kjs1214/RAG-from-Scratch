import time
from typing import List, Dict, Any
from .base import BaseChunker, BaseEmbedder, BaseVectorStore, BaseRetriever, BaseGenerator, Document, EmbedResult

class RAGPipeline:
    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_stores: List[BaseVectorStore], # 적재할 창고 목록 (FAISS, BM25 등)
        retriever: BaseRetriever,             # 검색을 지휘할 대장
        generator: BaseGenerator,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_stores = vector_stores
        self.retriever = retriever
        self.generator = generator
        print("[Pipeline] Modular RAG 시스템 조립 완료.")

    def build_index(self, raw_documents: List[Dict[str, Any]]):
        """오프라인 DB 구축: 텍스트 분할 -> 벡터화 -> 모든 창고에 적재"""
        print(f"[Pipeline] {len(raw_documents)}개 원본 문서 인덱싱 시작...")
        
        all_chunks: List[Document] = []
        for raw_doc in raw_documents:
            all_chunks.extend(self.chunker.split(
                text=raw_doc["content"], 
                doc_id_prefix=raw_doc["id"],
                metadata={"title": raw_doc.get("title", "")}
            ))

        # 2. 임베딩 (표준 상자 반환)
        texts_to_embed = [chunk.content for chunk in all_chunks]
        embed_result: EmbedResult = self.embedder.encode(texts_to_embed)

        # 3. 등록된 모든 창고(DB)에 데이터 적재
        for store in self.vector_stores:
            store.add_documents(all_chunks, embed_result)
            
        print("[Pipeline] 인덱싱 및 저장 완료.")

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """실시간 질의응답: 임베딩 -> 검색 대장에게 토스 -> LLM 생성"""
        start_time = time.time()
        
        # 1. 질문 임베딩 (표준 상자)
        query_embed_result: EmbedResult = self.embedder.encode([question])

        # 2. Retriever(검색기)에게 모든 검색 전략을 위임
        retrieved_results = self.retriever.retrieve(query_embed_result, top_k=top_k)

        # 3. LLM 답변 생성
        answer = self.generator.generate(question, retrieved_results)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": [{"score": res.score, "doc": res.doc} for res in retrieved_results],
            "latency_seconds": round(time.time() - start_time, 3)
        }