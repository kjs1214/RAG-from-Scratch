import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
# [수정됨] 표준 규격인 EmbedResult 임포트
from core.base import BaseEmbedder, EmbedResult 

class BasicEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] '{model_name}' 모델을 {self.device}에 로드 중...")
        self.model = SentenceTransformer(model_name, device=self.device)

    # [수정됨] 반환 타입을 EmbedResult로 변경
    def encode(self, texts: list[str]) -> EmbedResult:
        """
        텍스트 리스트를 입력받아 FAISS 연산에 최적화된 float32 타입의 numpy 배열 반환
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=32,             
            show_progress_bar=True,    
            convert_to_numpy=True
        )
        # [수정됨] numpy 배열을 EmbedResult DTO에 담아서 반환
        return EmbedResult(
            dense=embeddings.astype(np.float32), 
            dimension=768
        )