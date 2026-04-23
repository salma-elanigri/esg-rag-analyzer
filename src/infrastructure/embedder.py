import logging
from typing import List
from src.core.interfaces import IEmbedder
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class Embedder(IEmbedder, Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logging.info(f"Loading Embedder sentence transformer {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:  # renamed from embed_text
        embeddings = self.model.encode(text)
        return embeddings.tolist()
