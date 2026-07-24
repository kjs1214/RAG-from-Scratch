import os
import time
from modules.embedder import BasicEmbedder
from modules.vector_store import FAISSVectorStore
from modules.generator import LocalGenerator

def main():
    db_path = "./dbs/wiki_ko_db"
    
    print("========== [RAG 대화 test] ==========")
    
    # 1. 모듈 초기화
    embedder = BasicEmbedder()
    vector_store = FAISSVectorStore(dimension=768)
    generator = LocalGenerator()

    # 2. DB 적재 확인 및 로드 (최초 1회만 실행)
    if os.path.exists(os.path.join(db_path, "faiss.index")):
        vector_store.load(db_path)
    else:
        print(f"'{db_path}'에 DB가 없습니다. 먼저 build_db.py를 실행해서 DB를 생성해 주세요.")
        return

    SIMILARITY_THRESHOLD = 0.6
    
    print("\n준비 완료! 질문을 자유롭게 입력하세요.")
    print("종료하려면 'exit', 'quit' 또는 '종료'를 입력하세요.\n")

    # 3. 무한 루프로 챗봇 세션 시작
    while True:
        print("=" * 60)
        query = input("사용자 질문: ").strip()
        
        # 종료 조건
        if query.lower() in ['exit', 'quit', '종료']:
            print("대화를 종료합니다.")
            break
            
        # 빈 입력 방지
        if not query:
            continue

        start_time = time.time()
        
        # 임베딩 및 검색
        query_vec = embedder.encode([query])
        retrieved_results = vector_store.search(query_vec, top_k=3)
        
        use_rag = False
        if retrieved_results and retrieved_results[0].score >= SIMILARITY_THRESHOLD:
            use_rag = True

        # 분기 처리 (RAG vs Fallback)
        if use_rag:
            print(f"\n[RAG 모드] DB에서 관련 문서를 찾았습니다. (최고 유사도: {retrieved_results[0].score:.4f})")
            for i, res in enumerate(retrieved_results):
                print(f"  [{i+1}] score: {res.score:.4f} | {res.doc.content[:80]}...")
                
            answer = generator.generate(query=query, contexts=retrieved_results)
        else:
            print(f"\n[Fallback 모드] DB에 관련 정보가 없어 LLM 내부 지식으로 답변합니다.")
            answer = generator.generate(query=query, contexts=[])

        elapsed = round(time.time() - start_time, 3)
        
        print(f"\n답변 (소요 시간: {elapsed}초):")
        print(answer)
        print("\n")

if __name__ == "__main__":
    main()