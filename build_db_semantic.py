import os
import time
from datasets import load_dataset
from modules.chunker import SemanticChunker
from modules.embedder import BasicEmbedder
from modules.vector_store import FAISSVectorStore
from modules.retriever import BaselineRetriever
from modules.generator import LocalGenerator
from core.pipeline import RAGPipeline

def load_hf_wikipedia_korean(num_docs: int = 50000):
    print(f"Hugging Face에서 한국어 위키백과 문서 {num_docs}개를 다운로드합니다...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    documents = []
    for i, doc in enumerate(dataset):
        if i >= num_docs: break
        if doc["text"].strip():
            documents.append({"id": str(doc["id"]), "title": doc["title"], "content": doc["text"]})
    print(f"다운로드 완료: 총 {len(documents)}개 문서 확보.")
    return documents

def main():
    # [Proposed용 독립 경로]
    save_db_path = "./dbs/wiki_ko_semantic_db"
    start_time = time.time()

    print("========== [Proposed DB 인덱싱 시작 (Semantic Chunker)] ==========")
    raw_docs = load_hf_wikipedia_korean(num_docs=50000)
    
    if not raw_docs:
        return

    embedder = BasicEmbedder()
    
    # 1. SemanticChunker 사용 
    # [핵심] 리턴된 EmbedResult 상자에서 dense(numpy 배열)만 꺼내서 청커에게 전달하는 람다 함수 적용!
    chunker = SemanticChunker(
        embed_func=lambda texts: embedder.encode(texts).dense,
        percentile_threshold=20
    )
    
    vector_store = FAISSVectorStore(dimension=768)

    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_stores=[vector_store],
        retriever=None,
        generator=None
    )

    pipeline.build_index(raw_documents=raw_docs)

    os.makedirs(os.path.dirname(save_db_path), exist_ok=True)
    vector_store.save(save_db_path)
    
    elapsed = time.time() - start_time
    print(f"\n'{save_db_path}'에 저장 완료. 소요 시간: {elapsed:.2f}초")

if __name__ == "__main__":
    main()