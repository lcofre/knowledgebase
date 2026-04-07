from typing import List, Any
from llama_index.embeddings.ollama import OllamaEmbedding

class NomicOllamaEmbedding(OllamaEmbedding):
    """
    A custom wrapper for OllamaEmbedding to support Nomic's required prefixes.
    Nomic requires 'search_query: ' for queries and 'search_document: ' for documents.
    """
    def _get_query_embedding(self, query: str) -> List[float]:
        return super()._get_query_embedding(f"search_query: {query}")

    def _get_text_embedding(self, text: str) -> List[float]:
        return super()._get_text_embedding(f"search_document: {text}")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await super()._aget_query_embedding(f"search_query: {query}")

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await super()._aget_text_embedding(f"search_document: {text}")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        prefixed_texts = [f"search_document: {t}" for t in texts]
        return super()._get_text_embeddings(prefixed_texts)
