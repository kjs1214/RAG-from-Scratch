import os
import time
from datasets import load_dataset

from modules.chunker import RecursiveChunker
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
        if i >= num_docs:
            break
        if doc["text"].strip():
            documents.append({
                "id": str(doc["id"]),
                "title": doc["title"],
                "content": doc["text"]
            })
            
    print(f"다운로드 완료: 총 {len(documents)}개 문서 확보.")
    return documents

def main():
    # [수정 포인트 1] 저장 경로를 Recursive 전용 폴더로 변경!
    save_db_path = "./dbs/wiki_ko_recursive_db"
    start_time = time.time()

    print("========== [Recursive DB 인덱싱 시작] ==========")
    raw_docs = load_hf_wikipedia_korean(num_docs=50000)
    
    if not raw_docs:
        print("데이터를 불러오지 못했습니다.")
        return

    # [수정 포인트 2] 모듈 레고 블록 생성 (Retriever와 Generator 포함)
    chunker = RecursiveChunker(chunk_size=400, overlap=50)
    embedder = BasicEmbedder()
    vector_store = FAISSVectorStore(dimension=768)

    # [수정 포인트 3] 파이프라인 조립
    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_stores=[vector_store],
        retriever=None,
        generator=None
    )

    # [수정 포인트 4] 귀찮은 for문 다 지우고 파이프라인에게 원본 문서 통째로 넘김!
    print("\n텍스트 분할 및 벡터 DB 적재 파이프라인 가동...")
    pipeline.build_index(raw_documents=raw_docs)

    # 5. 디스크 저장
    os.makedirs(save_db_path, exist_ok=True)
    vector_store.save(save_db_path)
    
    elapsed = time.time() - start_time
    print(f"\n'{save_db_path}'에 저장 완료. 소요 시간: {elapsed:.2f}초")

if __name__ == "__main__":
    main()