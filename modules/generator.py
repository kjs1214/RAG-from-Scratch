import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from core.base import BaseGenerator, SearchResult

class LocalGenerator(BaseGenerator):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Generator] '{model_name}' 모델을 {self.device}에 로드 중...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

    def _generate_text(self, messages: List[dict], max_new_tokens: int) -> str:
        """중복되는 텍스트 생성 코드를 모아둔 헬퍼 메서드"""
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                output_scores=True,           
                return_dict_in_generate=True  
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[0, input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return answer

    def generate(self, query: str, contexts: List[SearchResult], max_new_tokens: int = 512, system_prompt: str = None) -> str:
        
        if contexts:
            # 문서 기반 1차 시도 프롬프트
            context_text = "\n".join([f"[문서 {i+1}] {res.doc.content}" for i, res in enumerate(contexts)])
            default_rag_prompt = "주어진 참조 문서만을 바탕으로 답변하세요. 참조 문서에 질문에 대한 단서가 전혀 없다면 절대 지어내지 말고 정확히 '[NOT_FOUND]' 라고만 답변하세요."
            sp = system_prompt if system_prompt else default_rag_prompt
            
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": f"참조 문서:\n{context_text}\n\n질문: {query}"}
            ]
            
            # 1차 생성
            answer = self._generate_text(messages, max_new_tokens)
            
            # 문서에 정답이 없는 경우
            if "[NOT_FOUND]" in answer:
                fallback_sp = "당신은 친절하고 유용한 AI 어시스턴트입니다. 당신이 알고 있는 지식을 바탕으로 친절하게 답변해 주세요."
                fallback_messages = [
                    {"role": "system", "content": fallback_sp},
                    {"role": "user", "content": query}
                ]
                
                # 내부 지식으로 2차 생성
                internal_answer = self._generate_text(fallback_messages, max_new_tokens)
                return f"[내부 지식 답변]\n{internal_answer}"
            
            # 검색이 성공한 경우
            return f"[위키백과 검색 답변]\n{answer}"
            
        else:
            # 문서가 아예 안 들어온 경우 바로 내부 지식 사용
            sp = system_prompt if system_prompt else "당신은 친절하고 유용한 AI 어시스턴트입니다."
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": query}
            ]
            internal_answer = self._generate_text(messages, max_new_tokens)
            return f"[내부 지식 답변]\n{internal_answer}"