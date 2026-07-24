import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

from core.base import BaseGenerator, RetrievalResult

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

    # system_prompt 파라미터 추가
    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int = 512, system_prompt: str = None) -> str:
        
        if contexts:
            # RAG 모드
            context_text = "\n".join([f"[문서 {i+1}] {res.doc.content}" for i, res in enumerate(contexts)])
            
            # 지식을 섞지 말고, 문서에 없으면 모른다고 방어하도록 하드코딩
            default_rag_prompt = "주어진 참조 문서만을 바탕으로 답변하세요. 참조 문서에 질문에 대한 단서가 전혀 없다면 절대 당신의 지식으로 지어내지 말고 '문서에서 해당 정보를 찾을 수 없습니다.'라고만 답변하세요."
            sp = system_prompt if system_prompt else default_rag_prompt
            
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": f"참조 문서:\n{context_text}\n\n질문: {query}"}
            ]
        else:
            # Fallback 모드
            sp = system_prompt if system_prompt else "당신은 친절하고 유용한 AI 보조입니다."
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": query}
            ]
        
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