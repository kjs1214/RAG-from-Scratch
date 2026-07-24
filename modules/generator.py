# modules/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

from core.base import BaseGenerator, RetrievalResult

class LocalGenerator(BaseGenerator):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Generator] '{model_name}' 모델을 {self.device}에 로드 중... (로우레벨 제어 모드)")
        
        # pipeline 대신 Tokenizer와 Model을 직접 로드합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int = 256) -> str:
        # 1. 텍스트 프롬프트 구성 (이전과 동일)
        context_text = "\n".join([f"- {res.doc.content}" for res in contexts])
        messages = [
            {"role": "system", "content": "주어진 참조 문서만을 바탕으로 질문에 답변하세요. 문서와 상식이 달라도 문서를 따르세요."},
            {"role": "user", "content": f"참조 문서:\n{context_text}\n\n질문: {query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 2. 텍스트를 PyTorch 텐서(숫자)로 변환하여 GPU에 올림
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 3. 모델 추론 (output_scores=True 가 연구의 핵심입니다!)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                output_scores=True,           # 각 단어를 뱉을 때의 로짓(Logit) 값을 모두 저장
                return_dict_in_generate=True  # 결과를 딕셔너리 형태로 반환
            )
        
        # 4. 입력된 프롬프트 길이를 제외하고, 새롭게 생성된 토큰들만 추출
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[0, input_length:]
        
        # 5. 토큰을 다시 텍스트로 디코딩
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # (선택) OOM 방지
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return answer