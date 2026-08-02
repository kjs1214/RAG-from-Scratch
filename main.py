import os
import time
from modules.embedder import BasicEmbedder
from modules.vector_store import FAISSVectorStore
from modules.retriever import BaselineRetriever
from modules.generator import LocalGenerator
from core.pipeline import RAGPipeline

def get_smart_snippet(query: str, content: str, window_size: int = 120) -> str:
    """질문에 포함된 키워드를 찾아 그 주변 텍스트만 쏙 뽑아주는 스마트 스니펫 함수"""
    content = content.replace('\n', ' ')
    
    # 질문에서 2글자 이상인 단어들만 키워드로 추출
    keywords = [w for w in query.split() if len(w) >= 2]
    
    match_idx = -1
    for kw in keywords:
        idx = content.find(kw)
        if idx != -1:
            match_idx = idx
            break
            
    # 키워드를 못 찾았으면 앞에서부터 자름
    if match_idx == -1:
        return content[:window_size] + "..." if len(content) > window_size else content
        
    # 키워드를 중심으로 앞뒤 텍스트 자르기
    start = max(0, match_idx - 30)
    end = min(len(content), match_idx + window_size - 30)
    
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(content) else ""
    
    return prefix + content[start:end] + suffix


def main():
    # 저장해둔 Recursive DB 경로 지정
    db_path = "./dbs/wiki_ko_recursive_db"

    print("========== [RAG 챗봇 시스템] ==========")
    print("1. 임베딩 모델 로드 중...")
    embedder = BasicEmbedder()
    
    print(f"2. 벡터 DB 불러오는 중... ({db_path})")
    vector_store = FAISSVectorStore(dimension=768)
    vector_store.load(db_path)

    print("3. 검색기(Retriever) 및 로컬 LLM(Generator) 로드 중...")
    retriever = BaselineRetriever(vector_store=vector_store)
    generator = LocalGenerator()

    # 파이프라인 조립
    pipeline = RAGPipeline(
        chunker=None,
        embedder=embedder,
        vector_stores=[vector_store],
        retriever=retriever,
        generator=generator
    )
    
    print("=" * 60)
    print(" RAG 챗봇을 시작합니다. (종료하려면 'q' 또는 'quit' 입력)")
    print("=" * 60)

    while True:
        user_query = input("\n👤 사용자: ")
        
        if user_query.lower() in ['q', 'quit', 'exit']:
            print("\n챗봇을 종료합니다.")
            break
            
        if not user_query.strip():
            continue

        print("챗봇: (답변 생성 중...)")
        
        try:
            result = pipeline.query(question=user_query, top_k=3)
            
            answer = result["answer"]
            latency = result["latency_seconds"]
            retrieved_docs = result["retrieved_docs"]
            
            print(f"\n{answer}")
            print(f"\n[소요 시간: {latency}초]")
            
            # 내부 지식으로 답변했는지 확인
            if "[내부 지식 답변]" in answer or "[NOT_FOUND]" in answer:
                print("-" * 60)
                print("[참고 문서 없음] 내부 지식으로 답변을 생성했습니다.")
                print("-" * 60)
            else:
                print("-" * 60)
                print("[참고 문서 (답변의 근간이 된 텍스트)]")
                for i, doc_info in enumerate(retrieved_docs):
                    title = doc_info["doc"].metadata.get("title", "제목 없음")
                    raw_content = doc_info["doc"].content
                    
                    content_snippet = get_smart_snippet(user_query, raw_content)
                        
                    print(f"  {i+1}. {title} (L2 Distance: {doc_info['score']:.4f})")
                    print(f"     ➔ {content_snippet}\n")
                print("-" * 60)
                
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()