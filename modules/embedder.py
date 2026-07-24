import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from core.base import BaseEmbedder

class BasicEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        # RTX 3060(단일 GPU) 사용을 위한 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] '{model_name}' 모델을 {self.device}에 로드 중...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 리스트를 입력받아 FAISS 연산에 최적화된 float32 타입의 numpy 배열 반환
        """
        # convert_to_numpy=True 옵션으로 텐서가 아닌 numpy 배열로 즉시 반환받음
        embeddings = self.model.encode(
            texts, 
            batch_size=32,             # 32개 단위로 쪼개서 GPU 연산
            show_progress_bar=True,    # 터미널에 진행률(%) 표시
            convert_to_numpy=True
            )
        return embeddings.astype(np.float32)