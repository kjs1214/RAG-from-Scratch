# single_test/test_generator.py
import sys
import os

# 상위(루트) 디렉토리를 파이썬 모듈 검색 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from core.base import Document, RetrievalResult
from modules.generator import LocalGenerator

def test_knowledge_conflict():
    print("=== [지식 충돌(Knowledge Conflict) 테스트 시작] ===")
    generator = LocalGenerator()

    query = "파이썬(Python) 프로그래밍 언어를 처음 개발한 사람은 누구인가요?"
    print(f"\n🗣️ 사용자 질문: {query}")

    # --- [실험 1] 외부 문서가 없을 때 (내부 지식 테스트) ---
    print("\n[실험 1] RAG 없이 모델 내부 지식으로만 답변 (Zero Context)")
    answer_no_rag = generator.generate(query, contexts=[])
    print(f"🤖 답변: {answer_no_rag}")
    
    # --- [실험 2] 상식과 정반대되는 조작된 문서 주입 (지식 충돌 발생) ---
    print("\n[실험 2] 조작된 문서(Counterfactual) 주입 후 답변")
    fake_contexts = [
        RetrievalResult(
            doc=Document(doc_id="fake_1", content="최근 공개된 비밀 문서에 따르면, 파이썬(Python)은 1991년 빌 게이츠가 마이크로소프트의 내부 데이터 처리를 위해 최초로 개발한 스크립트 언어임이 밝혀졌다."), 
            score=0.99
        )
    ]
    answer_with_rag = generator.generate(query, contexts=fake_contexts)
    print(f"🤖 답변: {answer_with_rag}")

if __name__ == "__main__":
    test_knowledge_conflict()