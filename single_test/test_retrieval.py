import sys
import os

# 상위(루트) 디렉토리를 파이썬 모듈 검색 경로에 추가 (C/C++의 -I include 경로 지정과 동일)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from core.base import Document
from modules.embedder import BasicEmbedder
from modules.vector_store import FAISSVectorStore

def test_modules():
    embedder = BasicEmbedder()
    vector_store = FAISSVectorStore(dimension=768)

    # 1. 가짜 문서 생성 (Document 규격 준수)
    docs = [
        Document(doc_id="1", content="건국대학교 컴퓨터공학부는 리눅스 시스템 프로그래밍을 가르친다."),
        Document(doc_id="2", content="FAISS는 C++로 작성된 초고속 벡터 검색 라이브러리다."),
    ]
    
    # 2. 임베딩 및 적재
    print("문서 벡터화 중...")
    embeddings = embedder.encode([d.content for d in docs])
    vector_store.add_documents(docs, embeddings)
    
    # 3. 디스크 저장 테스트 (dbs 폴더는 프로젝트 루트 아래 생성되도록 경로 지정)
    vector_store.save("./dbs/test_db")
    
    # 4. 검색 테스트
    query = "FAISS가 뭐야?"
    q_vec = embedder.encode([query])
    results = vector_store.search(q_vec, top_k=1)
    
    print(f"\n질문: {query}")
    print(f"검색 결과: {results[0].doc.content} (점수: {results[0].score:.4f})")

if __name__ == "__main__":
    test_modules()