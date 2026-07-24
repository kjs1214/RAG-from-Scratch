import os
import time
from datasets import load_dataset
from modules.chunker import SimpleChunker
from modules.embedder import BasicEmbedder
from modules.vector_store import FAISSVectorStore

# 총 50000개의 문서를 다운로드
def load_hf_wikipedia_korean(num_docs: int = 50000):
    """허깅페이스에서 한국어 위키백과 원문을 다운로드하여 리스트로 반환"""
    print(f"Hugging Face에서 한국어 위키백과 문서 {num_docs}개를 다운로드합니다...")
    # 'wikimedia/wikipedia'의 '20231101.ko' 버전(한국어) 로드
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    
    documents = []
    for i, doc in enumerate(dataset):
        if i >= num_docs:
            break
        
        # 원문(Text)이 비어있지 않은 경우에만 추가
        if doc["text"].strip():
            documents.append({
                "id": str(doc["id"]),
                "title": doc["title"],
                "content": doc["text"]
            })
            
    print(f"다운로드 완료: 총 {len(documents)}개 문서 확보.")
    return documents

def main():
    save_db_path = "./dbs/wiki_ko_db"
    start_time = time.time()

    print("========== [DB 인덱싱 시작] ==========")
    # 1. 외부 데이터 로드 (위키백과 1,000개 문서)
    raw_docs = load_hf_wikipedia_korean(num_docs=50000)
    
    if not raw_docs:
        print("데이터를 불러오지 못했습니다.")
        return

    # 2. 모듈 초기화
    # 위키백과는 한 문서가 매우 길기 때문에 청크 사이즈를 조금 넉넉하게 잡았습니다.
    chunker = SimpleChunker(chunk_size=400, overlap=50)
    embedder = BasicEmbedder()
    vector_store = FAISSVectorStore(dimension=768)

    # 3. 청킹 (문서 분할)
    print("\n텍스트 분할(Chunking) 진행 중...")
    all_chunks = []
    for doc in raw_docs:
        chunks = chunker.split(text=doc["content"], doc_id_prefix=doc["id"], metadata={"title": doc["title"]})
        all_chunks.extend(chunks)
    print(f"  -> 총 {len(raw_docs)}개 원본 문서에서 {len(all_chunks)}개 청크(Chunk) 생성 완료.")

    # 4. 임베딩 및 Vector DB 적재 (GPU 집중 구간)
    print(f"\n임베딩 및 DB 적재 진행 중... (총 {len(all_chunks)}개 연산)")
    texts = [c.content for c in all_chunks]
    embeddings = embedder.encode(texts)
    vector_store.add_documents(all_chunks, embeddings)

    # 5. 디스크 저장
    os.makedirs(os.path.dirname(save_db_path), exist_ok=True)
    vector_store.save(save_db_path)
    
    elapsed = time.time() - start_time
    print(f"\n'{save_db_path}'에 저장 완료.")

if __name__ == "__main__":
    main()