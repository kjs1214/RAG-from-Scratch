from core.pipeline import RAGPipeline
from modules.chunker import SimpleChunker
from modules.embedder import BasicEmbedder
from modules.vector_store import FAISSVectorStore # 이름 수정됨
from modules.generator import LocalGenerator

def main():
    print("========== [RAG 시스템 초기화] ==========")
    # 1. 모듈 초기화 (부품 생성)
    chunker = SimpleChunker(chunk_size=150, overlap=30)
    embedder = BasicEmbedder()
    vector_store = FAISSVectorStore(dimension=768)
    generator = LocalGenerator()

    # 2. 파이프라인 조립
    rag = RAGPipeline(chunker, embedder, vector_store, generator)

    # 3. 문서 데이터 로드 및 인덱싱
    print("\n========== [문서 적재 및 인덱싱] ==========")
    sample_docs = [
        {"id": "doc1", "title": "건국대 연구실", "content": "강지석 학생은 현재 건국대학교 심철준 교수님 연구실 소속이다. 주로 리눅스 환경과 C/C++ 시스템 프로그래밍을 다룬다."},
        {"id": "doc2", "title": "RAG 연구", "content": "RAG 파이프라인 구축 시 LangChain 같은 고수준 프레임워크를 배제하고 바닥부터 구현하면 실험과 평가의 투명성을 극대화할 수 있다."},
        {"id": "doc3", "title": "ELION Lab", "content": "또한 서재형 교수님이 이끄는 ELION Lab에서 인턴으로 근무하며 실시간 비디오 연기 감지 시스템을 개발 중이다."}
    ]
    rag.build_index(sample_docs)

    # 4. 질의응답(Q&A) 테스트
    print("\n========== [질의응답 테스트] ==========")
    query = "강지석 학생은 어느 교수님 연구실에 소속되어 있으며, 어떤 언어를 주로 사용하나요?"
    print(f"🗣️ 사용자 질문: {query}")
    
    result = rag.query(question=query, top_k=2)
    
    print("\n[Retrieved Contexts (검색된 문서)]")
    for doc_info in result["retrieved_docs"]:
        # 파이프라인에서 전달한 딕셔너리 구조에 맞춰 출력
        print(f" - 유사도: {doc_info['score']:.4f} | {doc_info['doc'].content}")
        
    print(f"\nLLM 답변 (소요 시간: {result['latency_seconds']}초)")
    print(result["answer"])

if __name__ == "__main__":
    main()